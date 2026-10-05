"""The single admitted exception to the Retention R1/R2/R3 canonical-vocabulary guards.

Those guards assert that canonical ECS_G code names no retention mechanism (consolidation, protection,
reconditioning, rehearsal) while none has qualified. Retention R3 ended OUTCOME_A, which its preregistration names
as the condition for a native K1 milestone; that milestone must itself pass its differential qualification
(research/ecs_k1_native). Only when both hold may exactly the milestone's files use the vocabulary; every other
canonical ECS_G file, and every R1/R2/R3 record, specification and verdict, is unchanged.

The exception is void unless the R3 QUAL record is byte-for-byte the recorded one (sha256 pinned here, as in
tests/research/ecs_retention_r3/test_evidence.py) and states OUTCOME_A, and one of ADMITTED_NATIVE_RECORDS verifies
completely (byte pin, internal digest, plan, clean harness commit, R3 binding) and states QUALIFIED. A file merely
claiming "QUALIFIED" admits nothing. ecsg-k1-native.v1 recorded NOT_QUALIFIED (docs/research/ECS_K1_NATIVE_RESULTS.md)
and is never admitted; while the K1 files exist without an admitted record, the R1/R2/R3 guards fail by design.
"""
from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path

R3_QUAL = Path("research/ecs_retention_r3/evidence/qual/ecsg-retention-r3.v1.qual.json")
R3_QUAL_SHA256 = "83ce2d61f30bb2927e1288995de5eb00ce41ff48767d54faedde88c8266c20e6"

K1_FILES = frozenset({
    "src/elpis/ECS_G/k1.py",
    "native/ECS_G/include/elpis/ecsg_k1.h",
    "native/ECS_G/include/elpis/ecsg_k1_fms.h",
    "native/ECS_G/src/ecsg_k1.c",
    "native/ECS_G/src/ecsg_k1_fms.c",
    "native/ECS_G/src/ecsg_k1_internal.h",
    "native/ECS_G/tests/test_ecsg_k1.c",
    "native/ECS_G/tests/test_ecsg_k1_alloc.c",
    "native/ECS_G/tests/test_ecsg_k1_fms.c",
    "native/ECS_G/tests/test_ecsg_k1_fms_alloc.c",
    "native/ECS_G/tests/test_ecsg_k1_fms_faults.c",
    "native/ECS_G/tests/test_ecsg_k1_performance.c",
})


def r3_outcome_a(repo: Path) -> bool:
    path = repo / R3_QUAL
    if not path.is_file():
        return False
    data = path.read_bytes()
    if hashlib.sha256(data).hexdigest() != R3_QUAL_SHA256:
        return False
    return json.loads(data)["body"]["outcome"] == "OUTCOME_A"


NATIVE_DOMAIN = "elpis.research.ecs-k1-native"

# The only native qualification records that may admit K1 into canonical code: each one byte-pinned, with the
# plan, the clean harness commit and the R3 authority it must be bound to. Empty until a plan version qualifies.
ADMITTED_NATIVE_RECORDS: dict = {
    "research/ecs_k1_native/evidence/ecsg-k1-native.v2.attestation.json": {
        "sha256": "73863d9b91e45e710c94bc3a13f752e82b3dbc0472b0dc6cf09a8e2f461de836",
        "experiment": "ecsg-k1-native.v2",
        "plan": "research/ecs_k1_native/specs/ecsg-k1-native.v2.plan.json",
        "plan_sha256": "6720e607d887f1a49545e3e5a6cfdc8b641991795a990f8fdd2157b0b86ea7eb",
        "harness_commit": "f720e4a906b0f9c1190536e6123de9c7f4e6160e",
        "r3_qual_digest": "7e9417b3fbe74329c34bc83daf24bfdf3dcf2ea01ce9d500791ba7e2e08a2627",
        "raw_evidence_sha256": "c96fa766b61bd05ceb9181ef0566123e6ba6a0afbca7e10367eaf580421994c2",
        "raw_qualification_digest": "fbea9d43e11706d789c6aac575d3412b0a6ada0c83bd4c0872d2ef2c55cf03aa",
    },
}


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def native_digest(kind: str, value) -> str:
    """The research/ecs_k1_native record digest (domain-separated SHA-256 of canonical JSON), without NumPy."""
    canonical = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True, allow_nan=False)
    return hashlib.sha256(f"{NATIVE_DOMAIN}.{kind}.v1".encode("ascii") + b"\0" + canonical.encode("ascii")).hexdigest()


def native_record_admitted(repo: Path, relative: str, pin: dict) -> bool:
    """Admit only the byte-pinned K1N-v2 attestation, bound to the immutable plan,
    clean harness, R3 authority and externally retained raw qualification identity."""
    path = repo / relative
    plan = repo / pin["plan"]

    if not path.is_file() or not plan.is_file():
        return False

    if _sha256(path) != pin["sha256"]:
        return False

    if _sha256(plan) != pin["plan_sha256"]:
        return False

    try:
        att = json.loads(path.read_bytes())
    except (OSError, json.JSONDecodeError):
        return False

    gates = att.get("gates", {})

    return (
        att.get("schema") == "elpis.ecsg.k1n2.qualification-attestation.v1"
        and att.get("experiment") == pin["experiment"]
        and att.get("verdict") == "QUALIFIED"
        and att.get("run_complete") is True
        and att.get("failed_gates") == []
        and set(gates) == {"E1", "E2", "E3", "E4", "E5", "L1", "L2", "D1_Q", "D1_F"}
        and all(gates.values())
        and att.get("world_counts") == {"Q": 32, "F": 32}
        and att.get("plan_sha256") == pin["plan_sha256"]
        and att.get("harness_commit") == pin["harness_commit"]
        and att.get("r3_qual_digest") == pin["r3_qual_digest"]
        and att.get("raw_evidence_sha256") == pin["raw_evidence_sha256"]
        and att.get("raw_qualification_digest") == pin["raw_qualification_digest"]
        and r3_outcome_a(repo)
    )


def native_qualified(repo: Path) -> bool:
    """A specifically admitted, fully bound native qualification record states QUALIFIED. Any other file under
    research/ecs_k1_native/evidence, whatever it claims, admits nothing."""
    return any(native_record_admitted(repo, relative, pin) for relative, pin in ADMITTED_NATIVE_RECORDS.items())


def admitted(repo: Path) -> frozenset:
    """Repository-relative files that may name the K1 mechanism: the K1 milestone's, and only after R3 OUTCOME_A
    and a QUALIFIED native differential qualification."""
    return K1_FILES if r3_outcome_a(repo) and native_qualified(repo) else frozenset()


def field_admitted(repo: Path, text: str, vocabulary: re.Pattern) -> bool:
    """An ELPIS_SYSTEM.json ECS_G field may use the vocabulary only after R3 OUTCOME_A, and only in sentences that
    name K1."""
    sentences = [s for s in re.split(r"(?<=[.;])\s+", text) if vocabulary.search(s)]
    return not sentences or (r3_outcome_a(repo) and native_qualified(repo) and all("K1" in s for s in sentences))
