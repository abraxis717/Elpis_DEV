"""R118-B sparse relational registry: participants, relations and incidence.

Immutable after construction (arrays are made read-only); built in one bulk,
vectorized index-construction phase. Storage is O(P + R + M):

=====================  =================  ====================================
array                  shape / dtype      meaning
=====================  =================  ====================================
participant_ids        [P] uint64         sorted stable 64-bit participant IDs
relation_ids           [R] uint64         sorted stable 64-bit relation IDs
rel_offsets            [R+1] int64        CSR row pointers into incidences
members                [M] int32          participant index, ascending per relation
frames                 [M, 8] float64     U[r,p] as two row-major 2x2 blocks
inc_relation           [M] int32          relation index of each incidence
part_offsets           [P+1] int64        reverse CSR row pointers
part_incidences        [M] int32          incidence positions grouped by participant
=====================  =================  ====================================

Frames are stored losslessly as the two diagonal 2x2 blocks of the 4x4
SO(2)xSO(2) matrix (off-block entries are exactly zero by contract), because
historical non-flat frames are floating-point products that are *not* exactly
of ``[[c,-s],[s,c]]`` form; a (cos, sin) encoding would perturb them.

No dense participant x participant (or relation x relation) structure exists.
Link operators between relations are derived on demand inside 4-slot windows.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


class RegistryError(ValueError):
    def __init__(self, code: str, detail: str = ""):
        super().__init__(code + (":" + detail if detail else ""))
        self.code = code


@dataclass(frozen=True)
class RegistryLimits:
    max_participants: int = 1 << 24
    max_relations: int = 1 << 26
    max_arity: int = 16                 # historical fixtures: 4; native kernel hard limit 64
    max_participant_degree: int = 1 << 20
    frame_tolerance: float = 1e-12      # |B B^T - I| and |det B - 1| per 2x2 block

    def __post_init__(self):
        if not 1 <= self.max_arity <= 64:
            raise RegistryError("ARITY_LIMIT_OUT_OF_RANGE")


def rotation_blocks(theta1, theta2, exact_libm: bool = True) -> np.ndarray:
    """``rotation(th1, th2)`` entries as [..., 8] blocks.

    ``exact_libm=True`` evaluates cos/sin with ``math`` (bit-identical to the historical
    ``portable_mechanics.rotation``); ``False`` uses vectorized numpy for large synthetic
    registries, where only the SO(2) contract matters, not libm bit-equality.
    """
    import math

    t1 = np.asarray(theta1, dtype=np.float64)
    t2 = np.asarray(theta2, dtype=np.float64)
    if exact_libm:
        c1 = np.vectorize(math.cos, otypes=[np.float64])(t1)
        s1 = np.vectorize(math.sin, otypes=[np.float64])(t1)
        c2 = np.vectorize(math.cos, otypes=[np.float64])(t2)
        s2 = np.vectorize(math.sin, otypes=[np.float64])(t2)
    else:
        c1, s1, c2, s2 = np.cos(t1), np.sin(t1), np.cos(t2), np.sin(t2)
    return np.stack([c1, -s1, s1, c1, c2, -s2, s2, c2], axis=-1)


def blocks_to_matrix(blocks) -> np.ndarray:
    b = np.asarray(blocks, dtype=np.float64)
    m = np.zeros(b.shape[:-1] + (4, 4))
    m[..., 0, 0], m[..., 0, 1], m[..., 1, 0], m[..., 1, 1] = b[..., 0], b[..., 1], b[..., 2], b[..., 3]
    m[..., 2, 2], m[..., 2, 3], m[..., 3, 2], m[..., 3, 3] = b[..., 4], b[..., 5], b[..., 6], b[..., 7]
    return m


def matrix_to_blocks(m) -> np.ndarray:
    m = np.asarray(m, dtype=np.float64)
    off = np.concatenate([m[..., 0:2, 2:4].reshape(m.shape[:-2] + (4,)), m[..., 2:4, 0:2].reshape(m.shape[:-2] + (4,))], -1)
    if np.any(off != 0.0):
        raise RegistryError("FRAME_NOT_BLOCK_DIAGONAL")
    return np.stack([m[..., 0, 0], m[..., 0, 1], m[..., 1, 0], m[..., 1, 1],
                     m[..., 2, 2], m[..., 2, 3], m[..., 3, 2], m[..., 3, 3]], axis=-1)


def _check_frames(frames: np.ndarray, tol: float) -> None:
    if frames.ndim != 2 or frames.shape[1] != 8:
        raise RegistryError("FRAME_SHAPE")
    if not np.all(np.isfinite(frames)):
        raise RegistryError("FRAME_NONFINITE")
    for b in (0, 4):
        a, bb, c, d = (frames[:, b + i] for i in range(4))
        orth = np.maximum.reduce([np.abs(a * a + bb * bb - 1.0), np.abs(c * c + d * d - 1.0), np.abs(a * c + bb * d)])
        det = np.abs(a * d - bb * c - 1.0)
        if frames.shape[0] and (orth.max() > tol or det.max() > tol):
            raise RegistryError("FRAME_NOT_SO2", f"block={b // 4} orth={orth.max():.3e} det={det.max():.3e}")


class Registry:
    """Read-only sparse hypergraph. Construct with ``build_registry``."""

    def __init__(self, **arrays):
        for name, arr in arrays.items():
            if isinstance(arr, np.ndarray):
                arr.setflags(write=False)
            setattr(self, name, arr)
        self.num_participants = int(self.participant_ids.shape[0])
        self.num_relations = int(self.relation_ids.shape[0])
        self.num_incidences = int(self.members.shape[0])

    # -- lookups (vectorized, refuse unknown IDs) ---------------------------
    @staticmethod
    def _lookup(sorted_ids: np.ndarray, ids, code: str) -> np.ndarray:
        q = np.asarray(ids, dtype=np.uint64).reshape(-1)
        pos = np.searchsorted(sorted_ids, q)
        ok = pos < sorted_ids.shape[0]
        ok[ok] = sorted_ids[pos[ok]] == q[ok]
        if not ok.all():
            raise RegistryError(code, f"{int((~ok).sum())} unknown")
        return pos.astype(np.int32)

    def participant_index(self, ids) -> np.ndarray:
        return self._lookup(self.participant_ids, ids, "UNKNOWN_PARTICIPANT")

    def relation_index(self, ids) -> np.ndarray:
        return self._lookup(self.relation_ids, ids, "UNKNOWN_RELATION")

    def arity(self, rel) -> np.ndarray:
        rel = np.asarray(rel)
        return (self.rel_offsets[rel + 1] - self.rel_offsets[rel]).astype(np.int64)

    def members_of(self, r: int) -> np.ndarray:
        return self.members[self.rel_offsets[r]:self.rel_offsets[r + 1]]

    def incidences_of_participant(self, p: int) -> np.ndarray:
        return self.part_incidences[self.part_offsets[p]:self.part_offsets[p + 1]]

    def participant_degree(self) -> np.ndarray:
        return np.diff(self.part_offsets)

    def nbytes(self) -> dict:
        names = ("participant_ids", "relation_ids", "rel_offsets", "members", "frames", "inc_relation",
                 "part_offsets", "part_incidences")
        out = {n: int(getattr(self, n).nbytes) for n in names}
        out["total"] = sum(out.values())
        return out

    def window_links(self, slot_rel) -> list:
        """Canonical window links for 4 slot relations (-1 = empty), historical order.

        Returns [(dst, src, participant_index, incidence_dst, incidence_src)], ordered by
        dst, then src, then participant index (``portable_mechanics.link_builder`` order).
        """
        out = []
        for d in range(4):
            rd = int(slot_rel[d])
            if rd < 0:
                continue
            md = self.members_of(rd)
            od = int(self.rel_offsets[rd])
            for s in range(4):
                rs = int(slot_rel[s])
                if s == d or rs < 0:
                    continue
                ms = self.members_of(rs)
                os_ = int(self.rel_offsets[rs])
                shared, id_, is_ = np.intersect1d(md, ms, assume_unique=True, return_indices=True)
                for p, i, j in zip(shared.tolist(), id_.tolist(), is_.tolist()):
                    out.append((d, s, p, od + i, os_ + j))
        return out


def build_registry(participant_ids, relation_ids, relation_offsets, member_ids, frames,
                   limits: RegistryLimits = RegistryLimits()) -> Registry:
    """Bulk, vectorized construction with explicit contracts (raises RegistryError)."""
    pid = np.asarray(participant_ids, dtype=np.uint64).reshape(-1)
    rid = np.asarray(relation_ids, dtype=np.uint64).reshape(-1)
    off = np.asarray(relation_offsets, dtype=np.int64).reshape(-1)
    mid = np.asarray(member_ids, dtype=np.uint64).reshape(-1)
    fr = np.ascontiguousarray(np.asarray(frames, dtype=np.float64))
    P_, R_, M_ = pid.shape[0], rid.shape[0], mid.shape[0]
    if P_ > limits.max_participants:
        raise RegistryError("PARTICIPANT_CAPACITY", str(P_))
    if R_ > limits.max_relations:
        raise RegistryError("RELATION_CAPACITY", str(R_))
    if off.shape[0] != R_ + 1 or (R_ + 1 and (off[0] != 0 or off[-1] != M_)):
        raise RegistryError("OFFSETS")
    arity = np.diff(off)
    if R_ and (arity.min() < 1 or arity.max() > limits.max_arity):
        raise RegistryError("ARITY_CONTRACT", f"min={arity.min()} max={arity.max()} limit={limits.max_arity}")
    if fr.shape != (M_, 8):
        raise RegistryError("FRAME_SHAPE")
    _check_frames(fr, limits.frame_tolerance)
    # Participants: sorted unique stable IDs.
    porder = np.argsort(pid, kind="stable")
    pid_sorted = pid[porder]
    if P_ > 1 and np.any(pid_sorted[1:] == pid_sorted[:-1]):
        raise RegistryError("DUPLICATE_PARTICIPANT")
    # Relations: sorted unique stable IDs.
    rorder = np.argsort(rid, kind="stable")
    rid_sorted = rid[rorder]
    if R_ > 1 and np.any(rid_sorted[1:] == rid_sorted[:-1]):
        raise RegistryError("DUPLICATE_RELATION")
    # Member IDs -> participant indices (refuse unknown).
    pos = np.searchsorted(pid_sorted, mid)
    ok = pos < P_
    ok[ok] = pid_sorted[pos[ok]] == mid[ok]
    if not ok.all():
        raise RegistryError("UNKNOWN_MEMBER", f"{int((~ok).sum())} unknown")
    member_idx = pos.astype(np.int64)
    # Incidence relation (in input relation order) -> new relation index.
    inc_rel_input = np.repeat(np.arange(R_, dtype=np.int64), arity)
    new_index_of_input = np.empty(R_, dtype=np.int64)
    new_index_of_input[rorder] = np.arange(R_, dtype=np.int64)
    inc_rel = new_index_of_input[inc_rel_input]
    # Canonical incidence order: by (relation index, participant index).
    order = np.lexsort((member_idx, inc_rel))
    inc_rel_s = inc_rel[order]
    mem_s = member_idx[order]
    if M_ > 1 and np.any((inc_rel_s[1:] == inc_rel_s[:-1]) & (mem_s[1:] == mem_s[:-1])):
        raise RegistryError("DUPLICATE_MEMBER_IN_RELATION")
    fr_s = np.ascontiguousarray(fr[order])
    rel_offsets = np.zeros(R_ + 1, dtype=np.int64)
    np.cumsum(np.bincount(inc_rel_s, minlength=R_), out=rel_offsets[1:])
    # Reverse index: incidences grouped by participant, ascending relation within each.
    rorder2 = np.lexsort((inc_rel_s, mem_s))
    pdeg = np.bincount(mem_s, minlength=P_)
    if P_ and pdeg.max(initial=0) > limits.max_participant_degree:
        raise RegistryError("PARTICIPANT_DEGREE_CONTRACT", f"max={pdeg.max()} limit={limits.max_participant_degree}")
    part_offsets = np.zeros(P_ + 1, dtype=np.int64)
    np.cumsum(pdeg, out=part_offsets[1:])
    if M_ >= (1 << 31) or R_ >= (1 << 31) or P_ >= (1 << 31):
        raise RegistryError("INDEX_WIDTH")
    return Registry(
        participant_ids=pid_sorted,
        relation_ids=rid_sorted,
        rel_offsets=rel_offsets,
        members=mem_s.astype(np.int32),
        frames=fr_s,
        inc_relation=inc_rel_s.astype(np.int32),
        part_offsets=part_offsets,
        part_incidences=rorder2.astype(np.int32),
        limits=limits,
    )
