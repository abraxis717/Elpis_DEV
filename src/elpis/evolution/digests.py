"""Domain-framed identity for evolution-path and promotion records.

Path assertions, transition receipts, harness manifests, evaluation evidence
and selection receipts are identified by the shared canonical identity
(``elpis.identity.content_digest``). Organism, genotype, lineage, fitness and
population records use their own persisted encoding in
:mod:`elpis.evolution.canonical`.
"""
from __future__ import annotations

import hashlib
from pathlib import Path

from elpis.identity import content_digest


def domain_digest(domain: str, payload: object) -> str:
    return content_digest(domain, payload)


def require_digest(value: object) -> None:
    if type(value) is not str or len(value) != 64:
        raise ValueError("digest must be 64 lowercase hex characters")
    if any(ch not in "0123456789abcdef" for ch in value):
        raise ValueError("digest must be lowercase hex")


def sha256_file(path: str | Path) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()
