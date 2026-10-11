"""Test lanes: which qualification surface owns every test (docs/CI_POLICY.md, "Lanes").

| Lane | What | Who runs it |
|---|---|---|
| FAST | production contracts, runtime, continuity adapter, boundary, small integration | bare ``pytest`` (the default), Python 3.11 / 3.12 jobs, ``tests/qualify.sh fast`` |
| STRESS | multi-process contention, the publisher's exhaustive process-death matrix, long-history scaling | ``pytest --lane stress``: the "Python stress and historical" job |
| SCIENTIFIC | frozen ECS science, retention and H-ECS evidence and replay | ``pytest --lane scientific``: the Scientific authority job |
| HISTORICAL | the DSV4.1 tower, the ECS dynamics laboratory, donor parity, the retained noncanonical model-execution mechanics of ``elpis.inference`` (no production consumer: docs/PYTHON_AUTHORITY_CENSUS.md) | ``pytest --lane historical``: the "Python stress and historical" job; donor parity: the Donor parity job |

NATIVE (ctest: ECS, K1, FMS, continuity, RuntimeCore, H-ECS mechanics) and the sanitizer STRESS builds are
CMake/ctest lanes; the offline job runs every Python lane (``--lane all``) with no network.

A bare ``pytest`` runs FAST only: ordinary local qualification never executes the research corpus by
accident. Explicit test paths run what they name; ``--lane`` filters any selection to one lane.
``tests/boundary/test_ci_policy.py`` proves every test file has exactly one lane and every lane an owning job.
"""
from __future__ import annotations

LANES = ("fast", "stress", "scientific", "historical")

# Ordered rules, first match wins: a repository-relative path prefix, optionally ``path::test_name`` (a test
# function name prefix, matching every parametrization).
RULES: tuple[tuple[str, str], ...] = (
    ("tests/boundary/test_donor_parity.py", "historical"),
    ("tests/research/dsv41_tower/", "historical"),
    ("tests/research/ecs_dynamics/", "historical"),
    ("tests/research/", "scientific"),
    # elpis.inference: the DSV4 codec boundary the runtime uses is FAST; the retained model-execution mechanics
    # (transaction, sequence, steering, principal, experts, rows, targets, speculative) have no production consumer.
    ("tests/inference/transaction/test_scaling.py", "stress"),
    ("tests/inference/test_text.py", "fast"),
    ("tests/inference/test_recipe_render.py", "fast"),
    ("tests/inference/test_context.py", "fast"),
    ("tests/inference/", "historical"),
    ("tests/pipeline/canonical_publisher/test_protocol.py::test_seeded_process_contention", "stress"),
    ("tests/pipeline/canonical_publisher/test_protocol.py::test_production_readers_during_successive_publications",
     "stress"),
    ("tests/pipeline/canonical_publisher/test_protocol.py::test_process_death_recovery", "stress"),
    # Mutation adequacy of the red-team regressions: a fresh interpreter per mutant over a sandbox copy.
    ("tests/boundary/test_redteam_mutations.py::test_the_regression_kills_the_mutant", "stress"),
    ("tests/", "fast"),
)


def lane_of(nodeid: str) -> str:
    """The lane of a pytest node id (``path::name[params]``) or a file path."""
    for prefix, lane in RULES:
        if "::" in prefix:
            if nodeid.startswith(prefix):
                return lane
        elif nodeid.split("::", 1)[0].startswith(prefix):
            return lane
    raise LookupError(f"no lane for {nodeid}")
