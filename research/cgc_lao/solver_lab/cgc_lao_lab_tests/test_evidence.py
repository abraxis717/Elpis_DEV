"""Evidence integrity: preregistered numbers, the freeze, run plans, recomputed dispositions, chronology.

These tests run no science: they recompute every recorded decision from the per-seed numbers in the
committed records under the frozen law, and check that VALIDATION and HOLDOUT ran exactly the frozen
source on the planned seeds.
"""
import json
import subprocess
import unittest

from cgc_lao_lab_tests import LAB
from cgc_lao_lab import evaluate, report
from cgc_lao_lab.candidates import CANDIDATES

EVIDENCE = LAB / "evidence"
FROZEN = LAB / "FROZEN.json"
DISPOSITION = EVIDENCE / "DISPOSITION.json"


def load(path):
    return json.loads(path.read_text(encoding="utf-8"))


def git(*args):
    try:
        out = subprocess.run(["git", *args], cwd=LAB, capture_output=True, text=True, check=True, timeout=60)
    except (OSError, subprocess.CalledProcessError, subprocess.TimeoutExpired):
        return None
    return out.stdout.strip()


class TestEvidenceIntegrity(unittest.TestCase):
    def test_records_are_sealed(self):
        records = sorted(EVIDENCE.rglob("*.json"))
        self.assertTrue(records)
        for path in records:
            rec = load(path)
            if path.name == "DISPOSITION.json":
                continue
            self.assertEqual(rec["schema"], "cgc-lao-solver-lab.record.v1", path)
            self.assertEqual(len(rec["source_sha256"]), 64, path)
            self.assertEqual(rec["spec_sha256"], evaluate.file_digest(evaluate.SPEC), path)

    def test_preregistered_gate_scales_equal_the_train_record(self):
        gates = load(EVIDENCE / "train" / "levels.json")["gates"]
        prereg = (LAB / "PREREGISTRATION.md").read_text(encoding="utf-8")
        for key in ("delta_clean", "delta_all", "online_budget"):
            self.assertIn(f"{gates[key]:.8f}", prereg, key)
        self.assertEqual(report.gate_scales(), gates)
        self.assertEqual((gates["clean_reference"], gates["all_reference"]), ("CLEAN_BAYES", "CGC_REFERENCE"))

    def test_freeze_binds_the_current_source(self):
        if not FROZEN.exists():
            self.skipTest("no freeze yet")
        frozen = load(FROZEN)
        self.assertEqual(frozen["source_files_sha256"], evaluate.source_file_digests())
        self.assertEqual(frozen["spec_sha256"], evaluate.file_digest(evaluate.SPEC))
        self.assertEqual(frozen["coefficients_sha256"], evaluate.coefficient_digests())
        for contract, cand in frozen["candidates"].items():
            self.assertEqual(cand["config"], CANDIDATES[cand["name"]], contract)

    def phase_records(self, phase):
        frozen = load(FROZEN)
        out = {}
        for contract in frozen["candidates"]:
            path = EVIDENCE / phase.lower() / f"{contract.lower()}.json"
            if path.exists():
                out[contract] = load(path)
        return frozen, out

    def test_validation_and_holdout_ran_the_frozen_source_and_plan(self):
        if not FROZEN.exists():
            self.skipTest("no freeze yet")
        for phase in ("VALIDATION", "HOLDOUT"):
            frozen, recs = self.phase_records(phase)
            plan = frozen["plans"][phase]
            for contract, rec in recs.items():
                self.assertEqual(rec["source_files_sha256"], frozen["source_files_sha256"], (phase, contract))
                self.assertEqual(rec["partition"], plan["partition"])
                self.assertEqual(rec["seeds"], list(range(*plan["seeds"])))
                self.assertEqual(rec["episodes_per_seed"], plan["episodes"])
                self.assertEqual(rec["contract"], contract)
                self.assertEqual(rec["regimes"], frozen["regimes"])

    def test_recorded_disposition_is_recomputed_from_per_seed_numbers(self):
        if not DISPOSITION.exists():
            self.skipTest("no disposition yet")
        disp = load(DISPOSITION)
        frozen = load(FROZEN)
        for contract, cand in frozen["candidates"].items():
            phases = {}
            for phase in ("VALIDATION", "HOLDOUT"):
                ref = load(EVIDENCE / phase.lower() / "original.json")
                rec = load(EVIDENCE / phase.lower() / f"{contract.lower()}.json")
                phases[phase] = (rec, ref)
            got = report.disposition(phases, cand["name"], disp[contract]["state_scalars"], disp["tests_passed"])
            self.assertEqual(got["status"], disp[contract]["status"], contract)
            self.assertEqual(got["failing"], disp[contract]["failing"], contract)
            for phase in phases:
                for g in report.CORE_GATES:
                    self.assertAlmostEqual(got["phases"][phase][g]["hi"], disp[contract]["phases"][phase][g]["hi"],
                                           places=12, msg=(contract, phase, g))

    def test_chronology(self):
        if git("rev-parse", "--is-shallow-repository") != "false":
            self.skipTest("needs a full git history")

        def added(rel):
            out = git("log", "--format=%H", "--diff-filter=A", "--", rel)
            return out.splitlines()[-1] if out else None

        def before(a, b):
            return a is not None and b is not None and git("merge-base", "--is-ancestor", a, b) is not None

        prereg = added("PREREGISTRATION.md")
        self.assertIsNotNone(prereg)
        order = [prereg]
        for rel in ("FROZEN.json", "evidence/validation/original.json", "evidence/holdout/original.json"):
            commit = added(rel)
            if commit is None:
                break
            self.assertTrue(before(order[-1], commit) and order[-1] != commit, rel)
            order.append(commit)
        for path in sorted((EVIDENCE / "dev").glob("*.json")):
            commit = added(str(path.relative_to(LAB)))
            if commit is not None:
                self.assertTrue(before(prereg, commit) and prereg != commit, path.name)


if __name__ == "__main__":
    unittest.main()
