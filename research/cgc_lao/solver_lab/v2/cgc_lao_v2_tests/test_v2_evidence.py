"""v2 evidence integrity: preregistered scales, recomputed selection, the freeze, frozen plans, native parity
records, recomputed disposition and chronology. No science is run here."""
import json
import subprocess
import unittest

from cgc_lao_v2_tests import V2
from cgc_lao_lab import evaluate as ev1
from cgc_lao_v2 import evaluate2, freeze2, report2

EVIDENCE = V2 / "evidence"
FROZEN = V2 / "FROZEN_V2.json"
DISPOSITION = EVIDENCE / "DISPOSITION_V2.json"


def load(path):
    return json.loads(path.read_text(encoding="utf-8"))


def git(*args):
    try:
        return subprocess.run(["git", *args], cwd=V2, capture_output=True, text=True, check=True, timeout=60).stdout.strip()
    except (OSError, subprocess.CalledProcessError, subprocess.TimeoutExpired):
        return None


class TestV2Evidence(unittest.TestCase):
    def test_records_are_sealed_against_the_current_specs(self):
        records = [p for p in sorted(EVIDENCE.rglob("*.json")) if p.name != "DISPOSITION_V2.json"]
        self.assertTrue(records)
        for path in records:
            rec = load(path)
            self.assertEqual(rec["schema"], "cgc-lao-solver-lab.record.v2", path)
            self.assertEqual(rec["spec_sha256"]["slb2"], ev1.file_digest(evaluate2.SPEC2), path)
            self.assertEqual(rec["spec_sha256"]["slb1"], ev1.file_digest(ev1.SPEC), path)

    def test_preregistered_scales_and_selection_are_recomputed(self):
        gates = load(EVIDENCE / "train" / "levels.json")["gates"]
        prereg = (V2 / "PREREGISTRATION_V2.md").read_text(encoding="utf-8")
        for key in ("delta_clean", "delta_all", "online_budget"):
            self.assertIn(f"{gates[key]:.8f}", prereg, key)
        sel = load(V2 / "SELECTION_V2.json")
        r1 = load(V2 / sel["deciding_round"]["record"])
        self.assertEqual(report2.select_primary(r1)["chosen"], sel["primary"]["chosen"])
        self.assertEqual(report2.select_fallback(r1)["chosen"], sel["fallback"]["chosen"])

    def test_freeze_binds_the_current_sources(self):
        if not FROZEN.exists():
            self.skipTest("no v2 freeze yet")
        frozen = load(FROZEN)
        self.assertEqual(frozen["source_files_sha256"], evaluate2.source_file_digests())
        self.assertEqual(frozen["native_files_sha256"], evaluate2.native_file_digests())
        self.assertEqual(frozen["coefficients_sha256"], evaluate2.coefficient_digests())
        self.assertIn(frozen["root_cmake_wiring"], freeze2.ROOT_CMAKE.read_text(encoding="utf-8"))
        sel = load(V2 / "SELECTION_V2.json")
        self.assertEqual(frozen["candidates"]["PRIMARY"]["name"], sel["primary"]["chosen"])
        self.assertEqual(frozen["candidates"]["FALLBACK"]["name"], sel["fallback"]["chosen"])

    def test_qualification_records_ran_the_frozen_plan(self):
        if not FROZEN.exists():
            self.skipTest("no v2 freeze yet")
        frozen = load(FROZEN)
        for phase, plan in frozen["plans"].items():
            path = EVIDENCE / phase.lower() / "qualification.json"
            if not path.exists():
                continue
            rec = load(path)
            self.assertEqual(rec["source_files_sha256"], frozen["source_files_sha256"], phase)
            self.assertEqual(rec["native_files_sha256"], frozen["native_files_sha256"], phase)
            self.assertEqual((rec["partition"], rec["seeds"], rec["episodes_per_seed"], rec["regimes"]),
                             (plan["partition"], list(range(*plan["seeds"])), plan["episodes"], plan["regimes"]), phase)
            parity = EVIDENCE / phase.lower() / "native_parity.json"
            if phase != "OOD":
                self.assertTrue(parity.exists(), phase)
            if parity.exists():
                p = load(parity)
                self.assertEqual(p["source_files_sha256"], frozen["source_files_sha256"])
                for role, arm in p["arms"].items():
                    self.assertTrue(arm["pass"] and arm["bitwise_failures"] == 0 and arm["decision_mismatches"] == 0,
                                    (phase, role))

    def test_recorded_disposition_is_recomputed(self):
        if not DISPOSITION.exists():
            self.skipTest("no v2 disposition yet")
        disp = load(DISPOSITION)
        frozen = load(FROZEN)
        primary, fallback = frozen["candidates"]["PRIMARY"]["name"], frozen["candidates"]["FALLBACK"]["name"]
        for phase in ("VALIDATION", "HOLDOUT"):
            path = EVIDENCE / phase.lower() / "qualification.json"
            if not path.exists():
                continue
            rec = load(path)
            g = report2.primary_gates(rec, primary)
            f = report2.fallback_gates(rec, fallback)
            for k in ("G1", "G2", "G3", "G4", "G5", "G7", "G9", "G10"):
                self.assertEqual(g[k]["pass"], disp["phases"][phase]["primary"][k], (phase, k))
            for k in ("F1", "F2", "F3", "F4", "F5", "F6"):
                self.assertEqual(f[k]["pass"], disp["phases"][phase]["fallback"][k], (phase, k))
            self.assertEqual({fam: v["claim"] for fam, v in g["families"].items()}, disp["phases"][phase]["families"])

    def test_chronology(self):
        if git("rev-parse", "--is-shallow-repository") != "false":
            self.skipTest("needs a full git history")

        def added(rel):
            out = git("log", "--format=%H", "--diff-filter=A", "--", rel)
            return out.splitlines()[-1] if out else None

        order = [added("PREREGISTRATION_V2.md")]
        self.assertIsNotNone(order[0])
        for rel in ("SELECTION_V2.json", "FROZEN_V2.json", "evidence/validation/qualification.json",
                    "evidence/holdout/qualification.json", "evidence/ood/qualification.json"):
            commit = added(rel)
            if commit is None:
                break
            self.assertTrue(order[-1] != commit and git("merge-base", "--is-ancestor", order[-1], commit) is not None, rel)
            order.append(commit)
        for path in sorted((EVIDENCE / "dev").glob("*.json")):
            commit = added(str(path.relative_to(V2)))
            if commit is not None:
                self.assertTrue(order[0] != commit and git("merge-base", "--is-ancestor", order[0], commit) is not None, path)


if __name__ == "__main__":
    unittest.main()
