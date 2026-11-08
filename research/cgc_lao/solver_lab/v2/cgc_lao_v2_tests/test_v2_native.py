"""Native C11 component: layout, bitwise parity with the scalar reference, tolerance parity with the vectorised
learner, fail-closed inputs, horizon, binding, poisoning, zeroisation, extremes, and static boundedness.

The library is compiled from source into a temporary directory. Without a C compiler these tests skip, unless
ELPIS_REQUIRE_NATIVE=1 (CI), where a missing compiler is a failure.
"""
import ctypes as C
import os
import re
import unittest

import numpy as np

from cgc_lao_v2_tests import V2
from cgc_lao_lab.env import logits
from cgc_lao_v2 import agents2, candidates2, env2, harness2, native
from cgc_lao_v2.scalar2 import ScalarRMF2

REQUIRE = os.environ.get("ELPIS_REQUIRE_NATIVE") == "1"
CONFIGS = [  # (config name, provider mode, acquisition)
    ("RMF_NOEVENT", "DUPLICATE", 0), ("SAFE4_P8", "FIRST_ONLY", 0), ("RMF5_E25", "DUPLICATE", 0),
    ("RMF5B_E25", "SECOND_EVENT", 0), ("RMF5_E25", "DUPLICATE", 1)]
REGIMES = ("CLEAN", "FLIP40", "FLIP48", "CORR50", "EVENT40", "QUNIF")


def native_or_skip(case):
    try:
        return native.library()
    except native.NativeUnavailable as exc:
        if REQUIRE:
            raise
        case.skipTest(str(exc))


def drive_pair(cfg, mode, acq, batch, e, check):
    hyps, prior = candidates2.BAYES_CONFIGS[cfg]["hypotheses"], candidates2.BAYES_CONFIGS[cfg]["prior"]
    s = ScalarRMF2(hyps, prior, acquisition=acq)
    n = native.NativeRMF(hyps, prior, acquisition=acq)
    contract = env2.CONTRACTS["DUP2_4"]
    z = logits(batch.router_w, batch.x[e])
    for t in range(8):
        a_s, a_n = s.select(list(z[t])), n.select(list(z[t]))
        check(a_s == a_n, ("action", t, a_s, a_n))
        y = [int(v) for v in batch.y[e, t, a_s, :contract.reads[t]]]
        if mode == "FIRST_ONLY" or len(y) == 1:
            steps = [(y[:1], True)]
        elif mode == "DUPLICATE":
            steps = [(y, True)]
        else:
            steps = [(y[:1], True), (y[1:], False)]
        for reads, adv in steps:
            s.update(a_s, float(z[t, a_s]), reads, adv)
            n.update(a_s, float(z[t, a_s]), reads, adv)
            check(s.mu == n.mu and s.v == n.v and s.lw == n.lw, ("state", t))
            check(s.counts == n.counts and s.t == n.t, ("counts", t))
    zq = logits(batch.router_w, batch.xq[e])
    for j in range(16):
        check(s.predictive(list(zq[j])) == n.predictive(list(zq[j])), ("predictive", j))
        check(s.decide(list(zq[j])) == n.decide(list(zq[j])), ("decision", j))
    check(s.weights() == n.weights() and s.state_correction() == n.state_correction(), ("readout",))
    return n


class TestNativeComponent(unittest.TestCase):
    def setUp(self):
        self.lib = native_or_skip(self)

    def test_layout_and_declared_sizes(self):
        self.assertEqual(self.lib.cgc_lao_rmf_state_size(), 480)
        self.assertEqual(C.sizeof(native.State), 480)
        self.assertEqual(self.lib.cgc_lao_rmf_config_size(), 520)
        for cfg, scalars in (("RMF_NOEVENT", 31), ("SAFE4_P8", 31), ("RMF5_E25", 38)):
            c = candidates2.BAYES_CONFIGS[cfg]
            self.assertEqual(native.NativeRMF(c["hypotheses"], c["prior"]).state_scalars(), scalars)

    def test_bitwise_parity_with_scalar_reference(self):
        for cfg, mode, acq in CONFIGS:
            for regime in REGIMES:
                batch = env2.episodes("DEV", 255, regime, 0, 6)
                for e in range(batch.count):
                    def check(ok, where):
                        self.assertTrue(ok, (cfg, mode, acq, regime, e, where))
                    drive_pair(cfg, mode, acq, batch, e, check)

    def test_tolerance_parity_with_vectorised_learner(self):
        near_ties = 0
        for cfg, mode, acq in CONFIGS[:4]:
            c = candidates2.BAYES_CONFIGS[cfg]
            batch = env2.episodes("DEV", 254, "FLIP40", 0, 64)
            agent = agents2.BayesAgent2(c["hypotheses"], c["prior"], acquisition="probe" if acq == 0 else "greedy")
            out = harness2.rollout2(agent, batch, env2.CONTRACTS["DUP2_4"], mode)
            for e in range(batch.count):
                n = native.NativeRMF(c["hypotheses"], c["prior"], acquisition=acq)
                z = logits(batch.router_w, batch.x[e])
                same = True
                for t in range(8):
                    a = n.select(list(z[t]))
                    if a != out.actions[e, t]:
                        same = False
                        break
                    y = [int(v) for v in batch.y[e, t, a, :env2.CONTRACTS["DUP2_4"].reads[t]]]
                    if mode == "FIRST_ONLY" or len(y) == 1:
                        n.update(a, float(z[t, a]), y[:1])
                    elif mode == "DUPLICATE":
                        n.update(a, float(z[t, a]), y)
                    else:
                        n.update(a, float(z[t, a]), y[:1])
                        n.update(a, float(z[t, a]), y[1:], advance=False)
                if not same:
                    near_ties += 1
                    continue
                self.assertTrue(np.allclose(n.mu, agent.filter.mu[e], atol=1e-9, rtol=0))
                self.assertTrue(np.allclose(n.v, agent.filter.v[e], atol=1e-9, rtol=0))
                self.assertTrue(np.allclose(n.lw, agent.filter.lw[e], atol=1e-9, rtol=0))
                zq = logits(batch.router_w, batch.xq[e])
                for j in range(16):
                    if n.decide(list(zq[j])) != out.decisions[e, j]:
                        p = n.predictive(list(zq[j]))
                        gaps = [abs(p[i] - p[k]) for i in range(3) for k in range(i + 1, 3)] + [abs(max(p) - 0.6)]
                        self.assertLess(min(gaps), 1e-9)
                        near_ties += 1
        self.assertLessEqual(near_ties, 2)

    def test_fail_closed_inputs_leave_state_unchanged(self):
        c = candidates2.BAYES_CONFIGS["RMF_NOEVENT"]
        n = native.NativeRMF(c["hypotheses"], c["prior"])
        n.update(n.select([0.2, -0.1, 0.4]), 0.4, [1, 1])
        before = n.state_bytes()
        for args in ((3, 0.1, [1]), (0, float("nan"), [1]), (0, float("inf"), [1]), (0, 0.1, [2]),
                     (0, 0.1, []), (0, 0.1, [1, 0, 1])):
            with self.assertRaises(native.NativeError) as cm:
                n.update(*args)
            self.assertEqual(str(cm.exception), "E_INPUT")
            self.assertEqual(n.state_bytes(), before)
        with self.assertRaises(native.NativeError):
            n.decide([0.0, float("nan"), 0.0])
        self.assertEqual(n.state_bytes(), before)

    def test_horizon_contradictions_extremes_binding_zeroisation(self):
        c = candidates2.BAYES_CONFIGS["RMF_NOEVENT"]
        n = native.NativeRMF(c["hypotheses"], c["prior"])
        n.update(1, 0.4, [1, 0])                                  # disagreeing duplicate readings
        w = n.weights()
        self.assertEqual(w[0], 0.0)                               # the clean channel is impossible
        for t in range(7):
            n.update(t % 3, [-745.0, 745.0, -1e6, 1e6, 0.0, -0.0, 3.0][t], [t % 2])
        self.assertTrue(all(np.isfinite(n.mu).ravel()) and min(np.ravel(n.v)) >= 1e-6)
        with self.assertRaises(native.NativeError) as cm:
            n.update(0, 0.1, [1])
        self.assertEqual(str(cm.exception), "E_HORIZON")
        with self.assertRaises(native.NativeError):
            n.select([0.0, 0.0, 0.0])
        n.update(0, 0.1, [1], advance=False)                      # an extra independent event is allowed
        other = native.NativeRMF([(0.3, 0.0)], [1.0])
        rc = self.lib.cgc_lao_rmf_weights(C.byref(n.st), C.byref(other.cfg), (C.c_double * 8)())
        self.assertEqual(native.STATUS[rc], "E_STATE")            # a state is bound to its configuration
        n.zeroize()
        self.assertEqual(n.state_bytes(), bytes(C.sizeof(native.State)))
        with self.assertRaises(native.NativeError) as cm:
            n.weights()
        self.assertEqual(str(cm.exception), "E_STATE")
        n.reset()
        self.assertEqual(n.t, 0)
        self.assertEqual(len(n.weights()), 4)
        with self.assertRaises(native.NativeError) as cm:
            native.NativeRMF([(0.5, 0.0)], [1.0])
        self.assertEqual(str(cm.exception), "E_CONFIG")
        near = native.NativeRMF([(0.0, 0.0), (0.49999, 0.0), ([0.45, 0.4999], 0.5)], [1, 1, 1], horizon=16)
        for t in range(16):
            near.update(t % 3, 3.0 * np.sin(t), [t % 2, (t // 2) % 2])
        self.assertTrue(all(np.isfinite(near.lw)) and all(np.isfinite(np.ravel(near.mu))))


class TestNativeSourceIsBounded(unittest.TestCase):
    SRC = (V2 / "native" / "src" / "cgc_lao_rmf.c").read_text(encoding="utf-8")

    def code(self):
        text = re.sub(r"/\*.*?\*/", "", self.SRC, flags=re.S)
        return re.sub(r"//[^\n]*", "", text)

    def test_no_allocation_io_or_mutable_globals(self):
        code = self.code()
        for banned in ("malloc", "calloc", "realloc", "free(", "alloca", "fopen", "fwrite", "printf", "getenv",
                       "socket", "rand(", "time(", "thread"):
            self.assertNotIn(banned, code, banned)
        objects = [line for line in code.splitlines() if line.startswith("static ") and "(" not in line.split("=")[0]]
        self.assertGreaterEqual(len(objects), 7)          # the constants and the quadrature tables
        for line in objects:                              # every file-scope object is immutable
            self.assertTrue(line.startswith("static const "), line)

    def test_call_graph_is_acyclic(self):
        code = self.code()
        bodies, pos = {}, 0
        for m in re.finditer(r"(?m)^[a-z_][\w \*]*?\b(\w+)\(([^;{]*)\)\s*\{", code):
            name, start = m.group(1), m.end()
            depth, i = 1, start
            while depth:
                depth += {"{": 1, "}": -1}.get(code[i], 0)
                i += 1
            bodies[name] = code[start:i]
            pos = i
        self.assertGreater(len(bodies), 10)
        graph = {f: {g for g in bodies if g != f and re.search(rf"\b{g}\(", body)} | ({f} if re.search(rf"\b{f}\(", body) else set())
                 for f, body in bodies.items()}
        state = {}

        def visit(f):
            self.assertNotEqual(state.get(f), 1, f"recursion through {f}")
            if state.get(f) == 2:
                return
            state[f] = 1
            for g in graph[f]:
                visit(g)
            state[f] = 2
        for f in graph:
            visit(f)
        del pos


if __name__ == "__main__":
    unittest.main()
