"""v4 closure: the research host runs the FROZEN v4 learner end to end (Milestone E); sealed records replay
from the frozen sources (bit for bit under the recorded profile, within a claim-preserving tolerance in any
runtime; docs/CI_POLICY.md); the disposition statuses and the OOD characterisation (rho_admit) are recomputed
from the sealed records with the preregistered rules (Milestone B)."""
import hashlib
import json
import os
import unittest

from cgc_lao_v4_tests import V4
from cgc_lao_lab import evaluate as ev1
from cgc_lao_lab import stats
from cgc_lao_v2 import agents2, env2, harness2, native, report2
from cgc_lao_v2.host import ResearchHost, admit, run_episode
from cgc_lao_v2.providers import MechanismRegistry, ProviderDeclaration, SimulatedProvider, slb2_digest
from cgc_lao_v4 import study4

REQUIRE = os.environ.get("ELPIS_REQUIRE_NATIVE") == "1"
EVIDENCE = V4 / "evidence"
FROZEN_V4 = json.loads((V4 / "FROZEN_V4.json").read_text(encoding="utf-8"))
CONTRACT = env2.CONTRACTS["DUP2_4"]
FROZEN = {"PRIMARY": FROZEN_V4["candidates"]["PRIMARY"]["config"],
          "FALLBACK": FROZEN_V4["candidates"]["FALLBACK"]["config"],
          "horizon": 8, "reads": list(CONTRACT.reads)}
REGISTRY = MechanismRegistry.slb2()
RHO_GRID = ((0.0, "FLIP40"), (0.1, "CORR10"), (0.25, "CORR25"), (0.5, "CORR50"), (0.75, "CORR75"),
            (0.9, "CORR90"), (1.0, "EVENT40"))


def load(path):
    return json.loads(path.read_text(encoding="utf-8"))


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def declare(relation, mechanism, bound=None):
    return ProviderDeclaration("p", relation, 2, slb2_digest(mechanism) if mechanism else "0" * 64, bound)


class TestV4HostEndToEnd(unittest.TestCase):
    """Contract I runs the frozen primary, Contract II the frozen fallback, through the host with the native
    component, decision for decision equal to the vectorised frozen learners (near-ties excepted)."""

    def setUp(self):
        try:
            native.library()
        except native.NativeUnavailable as exc:
            if REQUIRE:
                raise
            self.skipTest(str(exc))

    def check(self, regime, decl, behaviour, role, mode, n=48):
        batch = study4.episodes("DEV", 250, regime, 0, n)
        cfg = FROZEN[role]
        ref = harness2.rollout2(agents2.BayesAgent2(cfg["hypotheses"], cfg["prior"]), batch, CONTRACT, mode)
        host = ResearchHost(FROZEN, REGISTRY, 0.0, "native")
        adm = host.initialize(decl)
        self.assertEqual((adm.configuration, adm.mode), (role, mode))
        mismatches = 0
        for e in range(n):
            got = run_episode(host, batch, e, SimulatedProvider(batch, e, decl, behaviour, FROZEN["reads"]))
            mismatches += sum(int(a != b) for a, b in zip(got, ref.decisions[e]))
        self.assertLessEqual(mismatches, 1, (regime, role))

    def test_contract_one_runs_the_frozen_primary(self):
        self.assertEqual(FROZEN_V4["candidates"]["PRIMARY"]["mode"], "DUPLICATE")
        for regime in ("CLEAN", "FLIP40", "FLIP45"):
            self.check(regime, declare("INDEPENDENT_REREAD", "slb2/independent-reread"), "reread", "PRIMARY",
                       "DUPLICATE")

    def test_contract_two_runs_the_frozen_fallback(self):
        self.assertEqual(FROZEN_V4["candidates"]["FALLBACK"]["mode"], "FIRST_ONLY")
        for regime, decl, behaviour in (("EVENT40", declare("SHARED_SOURCE", "slb2/shared-source"), "reread"),
                                        ("FLIP40", declare("INDEPENDENT_REREAD", None), "reread"),
                                        ("CLEAN", declare("COPY", "slb2/independent-reread"), "copy")):
            self.check(regime, decl, behaviour, "FALLBACK", "FIRST_ONLY")


def replays():
    """(phase, record, recorded, recomputed): router seed 0 of every sealed v4 qualification record, recomputed from
    the frozen sources in the current runtime, for NOOP, the frozen primary and the frozen fallback."""
    labels = ("NOOP", FROZEN_V4["candidates"]["PRIMARY"]["name"], FROZEN_V4["candidates"]["FALLBACK"]["name"])
    for phase in ("VALIDATION", "HOLDOUT", "OOD"):
        path = EVIDENCE / phase.lower() / "qualification.json"
        if not path.exists():
            continue
        rec = load(path)
        regime = "FLIP45" if "FLIP45" in rec["regimes"] else "CORR25"
        specs = {k: {**rec["arms"][k], "factory": study4.factory(rec["arms"][k]["arm"], **rec["arms"][k]["overrides"])}
                 for k in labels}
        got = study4.evaluate_seed(rec["partition"], rec["seeds"][0], [regime], specs, rec["episodes_per_seed"])
        for label in labels:
            for metric, values in rec["results"][regime][label].items():
                yield phase, rec, values[0], got[regime][label][metric], (phase, regime, label, metric)


def claim_margins(rec, primary, fallback) -> list:
    """Distance of every decision bound of the v4 gates and family claims from its threshold."""
    sc, tau = report2.scales(), report2.TAU_SAFE
    g, f = report2.primary_gates(rec, primary), report2.fallback_gates(rec, fallback)
    out = [g["G1"]["hi"], g["G2"]["hi"], g["G3"]["hi"] - sc["delta_clean"], g["G4"]["hi"] - sc["delta_all"],
           g["G5"]["hi"] - sc["online_budget"], g["G9"]["vs_RMF_V1"]["hi"], g["G9"]["vs_NOOP"]["hi"] - tau,
           g["G10"]["hi"] - report2.V1_CLEAN_MARGIN, f["F1"]["hi"], f["F2"]["hi"], f["F4"]["hi"],
           f["F5"]["hi"] - sc["online_budget"]]
    out += [s["hi"] for s in g["G7"]["shifts"].values()] + [s["hi"] for s in f["F6"]["shifts"].values()]
    out += [v["hi"] - tau for v in f["F3"]["regimes"].values()]
    out += [x for fam in g["families"].values() for x in (fam["hi"], fam["hi"] - tau)]
    return [abs(x) for x in out]


class TestV4Closure(unittest.TestCase):
    # The recorded profile: what the records' ``environment`` names. BLAS threading is immaterial here (no product
    # is larger than 3 x 4), so the thread variable is not part of the profile.
    PROFILE = ("python", "numpy", "machine", "cpu")
    TOLERANCE = 1e-7

    def test_historical_replay_is_exact_under_the_recorded_profile(self):
        """HISTORICAL REPLAY (docs/CI_POLICY.md): bit for bit (the 10 recorded significant digits), demanded only
        when the current runtime matches the environment the record names."""
        current, exact, mismatch = ev1.environment(), 0, set()
        for phase, rec, recorded, value, where in replays():
            differs = [k for k in self.PROFILE if rec["environment"].get(k) != current.get(k)]
            if differs:
                mismatch.add((phase, tuple((k, rec["environment"].get(k), current.get(k)) for k in differs)))
                continue
            self.assertEqual(float(f"{value:.10g}"), recorded, where)
            exact += 1
        if not exact:
            self.skipTest(f"HISTORICAL_REPLAY_ENVIRONMENT_MISMATCH: exact replay is demanded only under the "
                          f"recorded profile: {sorted(mismatch)}")

    def test_current_runtime_regression(self):
        """Not skippable: in whatever runtime runs the suite, every recomputed per-seed value is within TOLERANCE of
        the record. A shift of at most TOLERANCE in every per-seed value moves any interval bound by at most about
        3 x TOLERANCE, and the test checks that every v4 gate and family claim sits more than ten times that far
        from its threshold (the closest, FLIP45 on VALIDATION, is 2.2e-5 away), so agreement within the tolerance
        cannot change a claim. Floating-point differences between runtimes are of order 1e-14; one changed
        decision moves a per-seed value by about 1e-5 and fails the test."""
        primary, fallback = FROZEN_V4["candidates"]["PRIMARY"]["name"], FROZEN_V4["candidates"]["FALLBACK"]["name"]
        checked = 0
        for phase, rec, recorded, value, where in replays():
            self.assertLessEqual(abs(value - recorded), self.TOLERANCE, where)
            checked += 1
        self.assertGreater(checked, 0)
        for phase in ("VALIDATION", "HOLDOUT"):
            path = EVIDENCE / phase.lower() / "qualification.json"
            if path.exists():
                self.assertGreater(min(claim_margins(load(path), primary, fallback)), 30 * self.TOLERANCE, phase)

    def test_disposition_statuses_follow_the_gates(self):
        path = EVIDENCE / "DISPOSITION_V4.json"
        if not path.exists():
            self.skipTest("no v4 disposition yet")
        disp = load(path)
        for phase, s in disp["phases"].items():
            self.assertEqual(s["record_sha256"], sha(EVIDENCE / phase.lower() / "qualification.json"), phase)
            self.assertEqual(s["native_parity_sha256"], sha(EVIDENCE / phase.lower() / "native_parity.json"), phase)
            self.assertTrue(all(s["native_parity"].values()), phase)
        both = set(disp["phases"]) == {"VALIDATION", "HOLDOUT"}
        primary_ok = both and all(all(s["primary"].values()) for s in disp["phases"].values())
        fallback_ok = both and all(all(s["fallback"].values()) for s in disp["phases"].values())
        self.assertEqual(disp["primary"]["name"], FROZEN_V4["candidates"]["PRIMARY"]["name"])
        self.assertEqual(disp["fallback"]["name"], FROZEN_V4["candidates"]["FALLBACK"]["name"])
        self.assertEqual(disp["primary"]["status"] == "QUALIFIED_SYNTHETIC_CONTRACT_I", primary_ok)
        self.assertEqual(disp["fallback"]["status"] == "QUALIFIED_SYNTHETIC_CONTRACT_II_CONSERVATIVE", fallback_ok)

    def test_rho_admit_is_recomputed_and_bounds_admission(self):
        path = EVIDENCE / "ood" / "characterization.json"
        if not path.exists():
            self.skipTest("OOD not opened yet")
        ch, rec = load(path), load(EVIDENCE / "ood" / "qualification.json")
        self.assertEqual(ch["record_sha256"], sha(EVIDENCE / "ood" / "qualification.json"))
        primary = FROZEN_V4["candidates"]["PRIMARY"]["name"]
        rho_admit = None
        for rho, regime in RHO_GRID:
            if stats.paired(report2.series(rec, regime, primary), report2.series(rec, regime, "NOOP"))["hi"] >= 0:
                break
            rho_admit = rho
        self.assertEqual(ch["rho_admit"], rho_admit)
        for bound, mechanism in ((0.1, "slb2/correlated-0.1"), (0.5, "slb2/correlated-0.5")):
            got = admit(declare("CORRELATED", mechanism, bound), REGISTRY, rho_admit if rho_admit is not None else 0.0)
            self.assertEqual(got.contract == "I", rho_admit is not None and bound <= rho_admit, bound)


if __name__ == "__main__":
    unittest.main()
