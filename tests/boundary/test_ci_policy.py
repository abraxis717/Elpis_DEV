"""CI policy gate, static part (docs/CI_POLICY.md).

The jobs that run the evidence chronology gates must see the history those gates prove (a shallow
checkout turned the Retention R0 chronology test into a bare KeyError on main, 63da0bf), the Scientific
authority job must exist and run the evidence suites, and the merge policy must say what it says.
"""
from __future__ import annotations

import re

from ._system import REPO

CI = REPO / ".github" / "workflows" / "ci.yml"
POLICY = REPO / "docs" / "CI_POLICY.md"
# Jobs whose pytest run includes the evidence chronology gates.
FULL_HISTORY_JOBS = ("python", "airgap", "authority", "lanes")


def _jobs() -> dict:
    body = CI.read_text(encoding="utf-8").split("\njobs:\n", 1)[1]
    parts = re.split(r"(?m)^  ([A-Za-z0-9_-]+):\n", body)
    return {parts[i]: parts[i + 1] for i in range(1, len(parts), 2)}


def _checkout_options(block: str) -> str:
    found = re.search(r"(?m)^ {6}- uses: actions/checkout@v\d+\n((?: {8,}\S.*\n)*)", block)
    assert found, "job has no actions/checkout step"
    return found.group(1)


def test_jobs_that_run_chronology_gates_check_out_full_history():
    jobs = _jobs()
    for name in FULL_HISTORY_JOBS:
        assert re.search(r"(?m)^ +fetch-depth: *0 *$", _checkout_options(jobs[name])), (
            f"job {name!r} runs evidence chronology gates but does not check out full history (fetch-depth: 0)")


def test_scientific_authority_job_runs_the_evidence_suites_in_a_declared_environment():
    block = _jobs()["authority"]
    assert "name: Scientific authority" in block
    assert "pytest -rs --lane scientific" in block   # every tests/research laboratory but the historical ones
    assert "tests/ECS" in block
    assert '"numpy==1.26.4"' in block and 'OPENBLAS_NUM_THREADS: "1"' in block
    assert re.search(r'ctest .*-R "\^ECS', block), "native executor/reference tests not run"


# -- lanes (tests/lanes.py): every test has exactly one lane, every lane an owning job -------------------------

def _test_files():
    return sorted(p.relative_to(REPO).as_posix() for p in (REPO / "tests").rglob("test_*.py"))


def test_every_test_file_has_a_lane_and_every_rule_is_live():
    from tests.lanes import LANES, RULES, lane_of

    files = _test_files()
    for path in files:
        assert lane_of(path) in LANES, path
    for prefix, lane in RULES:
        assert lane in LANES
        path, _, name = prefix.partition("::")
        live = [f for f in files if f.startswith(path)]
        assert live, f"stale lane rule: {prefix}"
        if name:   # a function rule names a test that exists
            assert any(re.search(rf"(?m)^def {re.escape(name)}\w*\(", (REPO / f).read_text()) for f in live), prefix
    # The research corpus is never in the default FAST lane.
    assert not [f for f in files if f.startswith("tests/research/") and lane_of(f) == "fast"]


def test_every_lane_has_an_owning_ci_job():
    jobs = _jobs()
    owners = {"fast": ("python", "--lane fast"), "scientific": ("authority", "--lane scientific"),
              "stress": ("lanes", "--lane stress"), "historical": ("lanes", "--lane historical")}
    from tests.lanes import LANES

    assert set(owners) == set(LANES)
    for lane, (job, needle) in owners.items():
        assert needle in jobs[job], (lane, job)
    assert "pytest --lane all" in jobs["airgap"]   # every Python lane, offline
    assert "tests/boundary/test_donor_parity.py" in jobs["parity"]   # its HISTORICAL test needs the donor
    for job in ("native", "sanitizers"):
        assert "ctest --test-dir build" in jobs[job]   # NATIVE and sanitizer STRESS lanes


def test_policy_states_the_merge_rules():
    text = " ".join(POLICY.read_text(encoding="utf-8").split())
    for needle in ("**No merge on red.**", "**One green run is not a pass.**",
                   "NONDETERMINISTIC / ENVIRONMENT-SENSITIVE", "Do not pick the green run",
                   "corrective work takes priority over new science", "Scientific authority",
                   "never edited, re-run or regenerated to make CI green"):
        assert needle in text, needle
