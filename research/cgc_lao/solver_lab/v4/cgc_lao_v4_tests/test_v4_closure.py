"""v4 closure: the research host runs the FROZEN v4 learner end to end (Milestone E); sealed records replay
bitwise from the frozen sources; the disposition statuses and the OOD characterisation (rho_admit) are
recomputed from the sealed records with the preregistered rules (Milestone B)."""
import hashlib
import json
import os
import unittest

from cgc_lao_v4_tests import V4
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


class TestV4Closure(unittest.TestCase):
    def test_sealed_records_replay_bitwise(self):
        """Deterministic replay: one router seed of each sealed qualification record, recomputed from the frozen
        sources, equals the recorded per-seed values exactly (records keep 10 significant digits)."""
        for phase in ("VALIDATION", "HOLDOUT", "OOD"):
            path = EVIDENCE / phase.lower() / "qualification.json"
            if not path.exists():
                continue
            rec = load(path)
            regime = "FLIP45" if "FLIP45" in rec["regimes"] else "CORR25"
            labels = ("NOOP", FROZEN_V4["candidates"]["PRIMARY"]["name"], FROZEN_V4["candidates"]["FALLBACK"]["name"])
            specs = {k: {**rec["arms"][k], "factory": study4.factory(rec["arms"][k]["arm"], **rec["arms"][k]["overrides"])}
                     for k in labels}
            got = study4.evaluate_seed(rec["partition"], rec["seeds"][0], [regime], specs, rec["episodes_per_seed"])
            for label in labels:
                for metric, values in rec["results"][regime][label].items():
                    self.assertEqual(float(f"{got[regime][label][metric]:.10g}"), values[0], (phase, label, metric))

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
