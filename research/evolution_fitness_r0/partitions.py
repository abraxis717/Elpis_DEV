"""Frozen world partitions, negative baselines and the declarative agent policy format. RESEARCH_ONLY. NO_CLAIM.

| Partition | Spec | Worlds | Role |
|---|---|---|---|
| CALIBRATION | corridor 9 | 4 | evaluator determinism and replay self-check |
| DEVELOPMENT | corridor 9 | 8 | where proposals may be developed (the promotion law's EVOLVE partition) |
| HELD_OUT | corridor 9 | 12 | the held-out fitness delta |
| OOD | corridor 17 | 8 | the out-of-distribution delta (a longer corridor) |

World sets are pairwise disjoint. Each partition's manifest digest binds its spec and world ids; the digests are
recorded write-once in ``frozen/partitions.json`` and verified by the tests. Negative baselines (``ALWAYS_STAY``,
``ALWAYS_LEFT``, ``AWAY``) frame the measure: a candidate must beat the best of them on HELD_OUT to pass
correctness.

An agent is *data*, never code: ``agent/policy.json`` maps every observation (:data:`~.environment.OBSERVATIONS`)
to an action. The evaluator never executes anything from a workspace.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from types import MappingProxyType

from .environment import OBSERVATIONS, Action, EnvironmentSpec

POLICY_SCHEMA = "elpis.research.evolution-fitness.policy.v1"
PARTITION_SCHEMA = "elpis.research.evolution-fitness.partition.v1"
POLICY_PATH = "agent/policy.json"
FROZEN = Path(__file__).resolve().parent / "frozen" / "partitions.json"

IN_DISTRIBUTION, OUT_OF_DISTRIBUTION = EnvironmentSpec(9), EnvironmentSpec(17)

PARTITIONS = MappingProxyType({
    "CALIBRATION": (IN_DISTRIBUTION, tuple(f"cal-{i}" for i in range(4))),
    "DEVELOPMENT": (IN_DISTRIBUTION, tuple(f"dev-{i}" for i in range(8))),
    "HELD_OUT": (IN_DISTRIBUTION, tuple(f"held-{i}" for i in range(12))),
    "OOD": (OUT_OF_DISTRIBUTION, tuple(f"ood-{i}" for i in range(8))),
})


def partition_digest(name: str) -> str:
    spec, worlds = PARTITIONS[name]
    body = json.dumps({"partition": name, "spec": spec.digest, "worlds": list(worlds)}, sort_keys=True,
                      separators=(",", ":")).encode()
    return hashlib.sha256(PARTITION_SCHEMA.encode() + b"\0" + body).hexdigest()


def promotion_partitions() -> dict[str, str]:
    """The four partitions as the promotion law names them (DEVELOPMENT is its EVOLVE partition)."""
    return {"EVOLVE": partition_digest("DEVELOPMENT"), "CALIBRATION": partition_digest("CALIBRATION"),
            "HELD_OUT": partition_digest("HELD_OUT"), "OOD": partition_digest("OOD")}


def _table(choose) -> MappingProxyType:
    return MappingProxyType({o: choose(int(o.split(":")[0])) for o in OBSERVATIONS})


NEGATIVE_BASELINES = MappingProxyType({
    "ALWAYS_STAY": _table(lambda direction: Action.STAY),
    "ALWAYS_LEFT": _table(lambda direction: Action.LEFT),
    "AWAY": _table(lambda direction: Action.LEFT if direction > 0 else Action.RIGHT),
})


class PolicyRefusal(ValueError):
    """A workspace policy that is not exactly the declarative policy format."""


def policy_document(table) -> bytes:
    return json.dumps({"schema": POLICY_SCHEMA, "table": {o: Action(table[o]).value for o in OBSERVATIONS}},
                      sort_keys=True, separators=(",", ":")).encode()


def load_policy(root) -> MappingProxyType:
    """The policy table of a workspace (``agent/policy.json``), strictly validated; never executed."""
    path = Path(root) / POLICY_PATH
    try:
        data = json.loads(path.read_bytes())
    except (OSError, ValueError) as exc:
        raise PolicyRefusal("no well-formed agent/policy.json") from exc
    if type(data) is not dict or set(data) != {"schema", "table"} or data["schema"] != POLICY_SCHEMA:
        raise PolicyRefusal("policy fields")
    table = data["table"]
    if type(table) is not dict or set(table) != set(OBSERVATIONS):
        raise PolicyRefusal("the table must map exactly every observation")
    try:
        return MappingProxyType({o: Action(table[o]) for o in OBSERVATIONS})
    except ValueError as exc:
        raise PolicyRefusal("unknown action") from exc
