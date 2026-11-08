"""Post-freeze host-admission regression tests (no v4 holdout access)."""
from __future__ import annotations

import json
import os
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
for path in (ROOT / "v2" / "src", ROOT / "src"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import numpy as np  # noqa: E402
from cgc_lao_v2.host import HostError, run_episode  # noqa: E402
from cgc_lao_v2.providers import (  # noqa: E402
    AuditedMechanism, MechanismRegistry, ProviderDeclaration, Reading,
    slb2_digest,
)
from host_closure.strict_host import StrictResearchHost, strict_admit  # noqa: E402

FROZEN_V4 = json.loads((ROOT / "v4" / "FROZEN_V4.json").read_text(encoding="utf-8"))
FROZEN = {
    "PRIMARY": FROZEN_V4["candidates"]["PRIMARY"]["config"],
    "FALLBACK": FROZEN_V4["candidates"]["FALLBACK"]["config"],
    "horizon": 8,
    "reads": [2, 2, 2, 2, 1, 1, 1, 1],
}
REGISTRY = MechanismRegistry.slb2()
ROUTER = np.zeros((3, 4), dtype=np.float64)
CONTEXT = np.zeros(4, dtype=np.float64)
QUERIES = np.zeros((1, 4), dtype=np.float64)


def declaration(relation="INDEPENDENT_REREAD", capacity=2, bound=None,
                mechanism="slb2/independent-reread"):
    return ProviderDeclaration("test-provider", relation, capacity,
                               slb2_digest(mechanism), bound)


def readings(expert: int, t: int, n: int):
    event = f"test/e{t}/a{expert}"
    return [Reading(event, f"{event}/r{i}", f"channel-{i}", expert, (t + i) % 2)
            for i in range(n)]


class TestAdmissionBoundary(unittest.TestCase):
    def test_independent_provider_with_one_reading_cannot_be_contract_one(self):
        got = strict_admit(declaration(capacity=1), REGISTRY)
        self.assertEqual((got.contract, got.configuration, got.mode),
                         ("II", "FALLBACK", "FIRST_ONLY"))
        host = StrictResearchHost(FROZEN, REGISTRY, backend="scalar")
        self.assertEqual(host.initialize(declaration(capacity=1)).contract, "II")

    def test_correlated_provider_with_one_reading_cannot_be_contract_one(self):
        d = declaration("CORRELATED", 1, 0.1, "slb2/correlated-0.1")
        self.assertEqual(strict_admit(d, REGISTRY, 0.25).contract, "II")

    def test_unknown_relation_is_never_admitted(self):
        self.assertEqual(strict_admit(declaration("TELEPATHY"), REGISTRY).contract, "II")

    def test_nonfinite_and_out_of_range_declared_bounds_fail_closed(self):
        for bad in (float("nan"), float("inf"), -0.1, 1.1, True):
            d = declaration("CORRELATED", 2, bad, "slb2/correlated-0.1")
            self.assertEqual(strict_admit(d, REGISTRY, 0.25).contract, "II", bad)
        d = declaration("INDEPENDENT_REREAD", 2, float("nan"))
        self.assertEqual(strict_admit(d, REGISTRY).contract, "II")

    def test_nonfinite_audited_bound_is_not_trusted(self):
        d = declaration("CORRELATED", 2, 0.1, "slb2/correlated-0.1")
        broken = MechanismRegistry({d.mechanism_sha256:
            AuditedMechanism("bad", "CORRELATED", float("nan"), "test")})
        self.assertEqual(strict_admit(d, broken, 0.25).contract, "II")

    def test_global_rho_admission_threshold_cannot_exceed_qualified_range(self):
        for bad in (float("nan"), float("inf"), -0.1, 0.3, True):
            with self.assertRaisesRegex(ValueError, "RHO_ADMISSION_OUT_OF_QUALIFIED_RANGE"):
                StrictResearchHost(FROZEN, REGISTRY, rho_admit=bad, backend="scalar")

    def test_valid_audited_correlation_within_v4_bound_is_admitted(self):
        d = declaration("CORRELATED", 2, 0.1, "slb2/correlated-0.1")
        self.assertEqual(strict_admit(d, REGISTRY, 0.25).contract, "I")
        self.assertEqual(strict_admit(d, REGISTRY, 0.0).contract, "II")

    def test_reject_unsupported_schedule_and_bad_capacity_types(self):
        wrong = {**FROZEN, "reads": [1] * 8}
        with self.assertRaisesRegex(ValueError, "UNQUALIFIED_READING_SCHEDULE"):
            StrictResearchHost(wrong, REGISTRY, backend="scalar")
        for bad in (2.0, 0, 3, True):
            self.assertEqual(strict_admit(declaration(capacity=bad), REGISTRY).contract, "II")


class TestStrictEpisode(unittest.TestCase):
    def host(self, backend="scalar", decl=None):
        h = StrictResearchHost(FROZEN, REGISTRY, backend=backend)
        self.assertEqual(h.initialize(decl or declaration()).contract, "I")
        h.reset_episode(ROUTER)
        return h

    def _assert_scalar_zero(self, host):
        learner = host.learner
        self.assertTrue(all(v == 0.0 for v in learner.lw))
        self.assertTrue(all(v == 0.0 for row in learner.mu for v in row))
        self.assertTrue(all(v == 0.0 for row in learner.v for v in row))
        self.assertEqual(learner.counts, [0, 0, 0])
        self.assertEqual(learner.t, 0)

    def test_missing_required_duplicate_poisons_and_zeroizes(self):
        host = self.host()
        a = host.propose_expert(CONTEXT)
        with self.assertRaisesRegex(HostError, "QUALIFIED_READING_COUNT_MISMATCH"):
            host.observe_selected(a, readings(a, 0, 1))
        self.assertEqual(host.ep.status, "FAILED")
        self._assert_scalar_zero(host)
        with self.assertRaises(HostError):
            host.propose_expert(CONTEXT)
        # A brand-new episode can proceed, with no inherited evidence.
        host.reset_episode(ROUTER)
        self.assertEqual(host.learner.t, 0)
        self.assertEqual(host.ep.status, "ACTIVE")
        a = host.propose_expert(CONTEXT)
        host.observe_selected(a, readings(a, 0, 2))
        self.assertEqual(host.learner.t, 1)

    def test_all_eight_steps_enforce_exact_read_schedule(self):
        host = self.host()
        for t, required in enumerate(FROZEN["reads"]):
            a = host.propose_expert(CONTEXT)
            host.observe_selected(a, readings(a, t, required))
        self.assertEqual(len(host.propose_terminal_decision(QUERIES)), 1)
        self.assertEqual(host.close_episode()["status"], "TERMINAL")
        self._assert_scalar_zero(host)

    def test_fifth_step_single_reading_is_valid(self):
        host = self.host()
        for t in range(5):
            a = host.propose_expert(CONTEXT)
            host.observe_selected(a, readings(a, t, 2 if t < 4 else 1))
        self.assertEqual(host.ep.t, 5)

    def test_extra_reading_fails_closed_not_silently_dropped(self):
        host = self.host()
        a = host.propose_expert(CONTEXT)
        with self.assertRaisesRegex(HostError, "QUALIFIED_READING_COUNT_MISMATCH"):
            host.observe_selected(a, readings(a, 0, 3))
        self._assert_scalar_zero(host)

    def test_fallback_receives_first_only(self):
        h = StrictResearchHost(FROZEN, REGISTRY, backend="scalar")
        self.assertEqual(h.initialize(declaration(capacity=1)).mode, "FIRST_ONLY")
        h.reset_episode(ROUTER)
        for t in range(8):
            a = h.propose_expert(CONTEXT)
            h.observe_selected(a, readings(a, t, 1))
        self.assertEqual(len(h.propose_terminal_decision(QUERIES)), 1)
        h.close_episode()
        self._assert_scalar_zero(h)

    def test_bounded_read_iterator(self):
        host = self.host()
        a = host.propose_expert(CONTEXT)
        # A finite but oversized iterable is materialized only up to 3 items.
        with self.assertRaisesRegex(HostError, "QUALIFIED_READING_COUNT_MISMATCH"):
            host.observe_selected(a, (r for r in readings(a, 0, 10)))
        self._assert_scalar_zero(host)

    def test_invalid_duplicate_metadata_does_not_reach_learner(self):
        host = self.host()
        a = host.propose_expert(CONTEXT)
        bad = readings(a, 0, 2)
        bad[1] = Reading(bad[1].event_id, bad[0].reading_id,
                         bad[1].channel_id, a, bad[1].value)
        with self.assertRaisesRegex(HostError, "READING_REPLAYED"):
            host.observe_selected(a, bad)
        self._assert_scalar_zero(host)

    def test_native_missing_duplicate_zeroizes_when_backend_available(self):
        try:
            from cgc_lao_v2 import native
            native.library()
        except Exception as exc:
            if os.environ.get("ELPIS_REQUIRE_NATIVE") == "1":
                raise
            self.skipTest(f"native research library not available: {exc}")
        host = self.host("native")
        a = host.propose_expert(CONTEXT)
        with self.assertRaisesRegex(HostError, "QUALIFIED_READING_COUNT_MISMATCH"):
            host.observe_selected(a, readings(a, 0, 1))
        self.assertEqual(host.learner.state_bytes(), bytes(len(host.learner.state_bytes())))


if __name__ == "__main__":
    unittest.main()
