"""R122 small-amplitude (decayed-regime) linear operators of the teacher and of the frozen HCA.

Both laws are ``x' = clamp(k(inv) . (a, sal, col, b))`` with invariants that are quadratic
in the state. As the amplitude goes to zero the invariants vanish, the softmax salience
weights become uniform and the clamp is inactive, so one composed pass is a *linear* map
of the relation states. It is obtained here in two independent ways:

* ``probe``     columns ``F(eps e_i) / eps`` from the real step functions (the real R119
  EXACT_GUARDED kernel for the HCA, the frozen torch teacher for the reference) at
  ``eps = 2^-70``, where the R119 guards prove the gates equal their bias values and the
  softmax weights equal ``1/n`` exactly. Nothing analytical replaces the kernel.
* ``analytic``  the same operator assembled from the eight limiting gate values and the
  window link operators (``U_d U_s^T``), with MEAN reconciliation.

Agreement of the two shows that, in the decayed regime, the HCA differs from its teacher
*only* through the eight limiting gate values. The spectra then give the asymptotic decay
rate and phase velocity of each system.

The R104 controller (alpha = beta = 0.5) is linear: ``1.5 M - 0.5 I``. The R107 controller
(alpha != beta) contains the radial projection ``s (s.d)/(s.s)``, which is homogeneous but
not linear, so no matrix is reported for it.
"""
from __future__ import annotations

import math

import numpy as np

from research.yang_mills_hca.r118_scalable_awr import engine as E118
from research.yang_mills_hca.r118_scalable_awr import provenance as P
from research.yang_mills_hca.r118_scalable_awr.registry import blocks_to_matrix

from . import harness as H

PROBE_EXPONENT = -70


def copy_rows(S, c: int):
    idx = np.nonzero(S.row_copy == c)[0]
    return idx[np.argsort(S.row_local[idx], kind="stable")]


def probe(step_fn, S, exponent: int = PROBE_EXPONENT):
    """Linear operator of one copy's subsystem from the real step function (all copies probed in parallel)."""
    rows = [copy_rows(S, c) for c in range(H.COPIES)]
    n = 8 * len(rows[0])
    if any(len(r) != len(rows[0]) for r in rows) or any(
            not np.array_equal(S.row_local[r], S.row_local[rows[0]]) for r in rows):
        raise RuntimeError("COPY_STRUCTURE")
    eps = math.ldexp(1.0, exponent)
    M = np.empty((n, n))
    for start in range(0, n, H.COPIES):
        X = np.zeros((len(S.rows), 8))
        cols = []
        for c in range(H.COPIES):
            i = start + c
            if i < n:
                X[rows[c][i // 8], i % 8] = eps
                cols.append((c, i))
        Y = step_fn(X)
        for c, i in cols:
            M[:, i] = np.ldexp(Y[rows[c]].reshape(-1), -exponent)
    return M


def gate_limits(checkpoint: str):
    """The eight gate values of a frozen checkpoint at zero invariants: sigmoid / tanh of the biases."""
    _, b, _ = E118.load_weights(checkpoint)
    return tuple((1.0 / (1.0 + math.exp(-float(b[q])))) if q in (0, 4) else math.tanh(float(b[q])) for q in range(8))


def teacher_limits():
    zero = [0.0] * 4
    return H.teacher_coefficients(zero, zero, zero, zero)


def analytic(S, gates):
    """Decayed-regime operator of copy 0 from limiting gates and window links (MEAN reconciliation)."""
    reg, st = S.reg, S.st
    rows0 = copy_rows(S, 0)
    pos = {int(S.rows[r]): j for j, r in enumerate(rows0)}     # pool row -> local index within copy 0
    n = 8 * len(rows0)
    acc = np.zeros((n, n))
    count = np.zeros(len(rows0))
    k = gates
    eye = np.eye(4)
    for hnd in S.h.tolist():
        slot_rel, slot_row = st.slot_rel[hnd], st.slot_row[hnd]
        if not any(int(r) in pos for r in slot_row if r >= 0):
            continue
        links = reg.window_links(slot_rel)
        for d in range(4):
            if slot_row[d] < 0:
                continue
            j = pos[int(slot_row[d])]
            count[j] += 1
            ra, rb = slice(8 * j, 8 * j + 4), slice(8 * j + 4, 8 * j + 8)
            acc[ra, ra] += k[0] * eye
            acc[ra, rb] += k[3] * eye
            acc[rb, rb] += k[4] * eye
            acc[rb, ra] += k[7] * eye
            mine = [lk for lk in links if lk[0] == d]
            for (_, s, _p, inc_d, inc_s) in mine:
                L = blocks_to_matrix(reg.frames[inc_d]) @ blocks_to_matrix(reg.frames[inc_s]).T / len(mine)
                js = pos[int(slot_row[s])]
                sa, sb = slice(8 * js, 8 * js + 4), slice(8 * js + 4, 8 * js + 8)
                acc[ra, sa] += k[1] * L
                acc[ra, sb] += k[2] * L
                acc[rb, sa] += k[5] * L
                acc[rb, sb] += k[6] * L
    return acc / np.repeat(count, 8)[:, None]


def spectrum(M):
    lam = np.linalg.eigvals(M)
    order = np.argsort(-np.abs(lam), kind="stable")
    lam = lam[order]
    top = lam[0]
    return {"spectral_radius": float(abs(top)), "dominant_real": float(top.real), "dominant_imag": float(abs(top.imag)),
            "dominant_phase_deg_per_step": float(abs(math.degrees(math.atan2(top.imag, top.real)))),
            "second_modulus": float(np.abs(lam[min(2 if abs(top.imag) > 0 else 1, len(lam) - 1)])),
            "negative_real_max_modulus": float(max([abs(x) for x in lam if x.real < 0], default=0.0))}


def powers_readout(S, M, x0_copy, steps: int = H.HMAX):
    """Trajectory of one copy under the linear operator: list of states [n] for h = 0..steps."""
    out = [x0_copy]
    for _ in range(steps):
        out.append(M @ out[-1])
    return np.array(out)


def analyze_system(S, oracle, checkpoints=P.CHECKPOINTS):
    """Operators, analytic agreement and spectra for one composed system."""
    Mt = probe(lambda X: S.teacher_step(oracle, X), S)
    Mt2 = probe(lambda X: S.teacher_step(oracle, X), S, PROBE_EXPONENT - 10)
    At = analytic(S, teacher_limits())
    scale_t = float(np.abs(Mt).max())
    rec = {"dimension": int(Mt.shape[0]),
           "teacher": {"probe_homogeneity_max_abs": float(np.abs(Mt - Mt2).max()),
                       "analytic_vs_probe_max_rel": float(np.abs(At - Mt).max() / scale_t), **spectrum(Mt)},
           "checkpoints": {}}
    ops = {"TEACHER": Mt}
    worst_h, worst_a = rec["teacher"]["probe_homogeneity_max_abs"], rec["teacher"]["analytic_vs_probe_max_rel"]
    for ck in checkpoints:
        Mh = probe(lambda X, ck=ck: S.step(X, ck, "NATIVE"), S)
        Mh2 = probe(lambda X, ck=ck: S.step(X, ck, "NATIVE"), S, PROBE_EXPONENT - 10)
        Ah = analytic(S, gate_limits(ck))
        item = {"probe_homogeneity_max_abs": float(np.abs(Mh - Mh2).max()),
                "analytic_vs_probe_max_rel": float(np.abs(Ah - Mh).max() / float(np.abs(Mh).max())),
                "operator_mismatch_rel_fro": float(np.linalg.norm(Mh - Mt) / np.linalg.norm(Mt)),
                "native": spectrum(Mh)}
        item["native"]["radius_ratio_vs_teacher"] = item["native"]["spectral_radius"] / rec["teacher"]["spectral_radius"]
        item["native"]["phase_minus_teacher_deg_per_step"] = (item["native"]["dominant_phase_deg_per_step"]
                                                              - rec["teacher"]["dominant_phase_deg_per_step"])
        worst_h = max(worst_h, item["probe_homogeneity_max_abs"])
        worst_a = max(worst_a, item["analytic_vs_probe_max_rel"])
        ops[ck + "|NATIVE"] = Mh
        if not H.is_signed(ck):
            Mc = probe(lambda X, ck=ck: S.step(X, ck, "R114"), S)
            lin = 1.5 * Mh - 0.5 * np.eye(Mh.shape[0])
            item["r104_controller_vs_1p5M_minus_0p5I_max_abs"] = float(np.abs(Mc - lin).max())
            item["r104"] = spectrum(Mc)
            item["r104"]["radius_ratio_vs_teacher"] = item["r104"]["spectral_radius"] / rec["teacher"]["spectral_radius"]
            item["r104"]["phase_minus_teacher_deg_per_step"] = (item["r104"]["dominant_phase_deg_per_step"]
                                                                - rec["teacher"]["dominant_phase_deg_per_step"])
            worst_a = max(worst_a, item["r104_controller_vs_1p5M_minus_0p5I_max_abs"] / float(np.abs(Mh).max()))
            ops[ck + "|R114"] = Mc
        rec["checkpoints"][ck] = item
    rec["max_probe_homogeneity_abs"] = worst_h
    rec["max_analytic_vs_probe_rel"] = worst_a
    return rec, ops


def linear_trajectory_residual(S, ops, X0, T, XS_by_key):
    """Largest relative deviation between real trajectories and operator powers (per copy, all steps).

    ``XS_by_key`` maps the keys of ``ops`` (other than TEACHER) to real trajectories ``[21, rows, 8]``.
    """
    worst = {}
    for key, M in ops.items():
        real = T if key == "TEACHER" else XS_by_key.get(key)
        if real is None:
            continue
        w = 0.0
        for c in range(H.COPIES):
            rows = copy_rows(S, c)
            traj = powers_readout(S, M, X0[rows].reshape(-1))
            ref = real[:, rows].reshape(real.shape[0], -1)
            num = np.sqrt(((traj - ref) ** 2).sum(-1))
            den = np.sqrt((ref ** 2).sum(-1))
            w = max(w, float((num[den > 0] / den[den > 0]).max()))
        worst[key] = w
    return worst
