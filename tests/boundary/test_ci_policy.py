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
FULL_HISTORY_JOBS = ("python", "airgap", "authority")


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
    for suite in ("tests/research/ecs_cognition_r0", "tests/research/ecs_retention_r0",
                  "tests/research/ecs_runtime_r1", "tests/ECS_G"):
        assert suite in block, suite
    assert '"numpy==1.26.4"' in block and 'OPENBLAS_NUM_THREADS: "1"' in block
    assert re.search(r'ctest .*-R "\^ECS_G', block), "native executor/reference tests not run"


def test_policy_states_the_merge_rules():
    text = " ".join(POLICY.read_text(encoding="utf-8").split())
    for needle in ("**No merge on red.**", "**One green run is not a pass.**",
                   "NONDETERMINISTIC / ENVIRONMENT-SENSITIVE", "Do not pick the green run",
                   "corrective work takes priority over new science", "Scientific authority",
                   "never edited, re-run or regenerated to make CI green"):
        assert needle in text, needle
