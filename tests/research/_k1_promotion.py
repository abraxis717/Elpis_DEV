"""The single admitted exception to the Retention R1/R2/R3 canonical-vocabulary guards.

Those guards assert that canonical ECS_G code names no retention mechanism (consolidation, protection,
reconditioning, rehearsal) while none has qualified. Retention R3 ended OUTCOME_A, which its preregistration names
as the sole condition for a native K1 milestone. From then on, exactly the files of that milestone may use the
vocabulary; every other canonical ECS_G file, and every R1/R2/R3 record, specification and verdict, is unchanged.

The exception is void unless the R3 QUAL record is byte-for-byte the recorded one (sha256 pinned here, as in
tests/research/ecs_retention_r3/test_evidence.py) and states OUTCOME_A. The native milestone's own scientific gate
(differential qualification against every R3 QUAL world) is research/ecs_k1_native, tested in
tests/research/ecs_k1_native.
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


def admitted(repo: Path) -> frozenset:
    """Repository-relative files that may name the K1 mechanism: the K1 milestone's, and only after R3 OUTCOME_A."""
    return K1_FILES if r3_outcome_a(repo) else frozenset()


def field_admitted(repo: Path, text: str, vocabulary: re.Pattern) -> bool:
    """An ELPIS_SYSTEM.json ECS_G field may use the vocabulary only after R3 OUTCOME_A, and only in sentences that
    name K1."""
    sentences = [s for s in re.split(r"(?<=[.;])\s+", text) if vocabulary.search(s)]
    return not sentences or (r3_outcome_a(repo) and all("K1" in s for s in sentences))
