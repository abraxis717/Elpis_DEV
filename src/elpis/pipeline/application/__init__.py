"""Capability application against shadow capability state.

Applies unapplied consumption artifacts against shadow capability records with
deterministic guards, atomicity, replay protection and a durable SQLite
application ledger. Never mutates canonical capability state and never
activates a model.
"""
from .canonical import canonical_json, canonical_digest, check_hex64
from .errors import ApplicationError, ApplicationRejected, AuthorityViolation
from .shadow_state import ShadowCapabilityState
from .application import apply_artifact
from .ledger import ApplicationLedger, ledger_head_digest
from .durable_ledger import DurableApplicationLedger

__all__ = [
    "canonical_json", "canonical_digest", "check_hex64",
    "ApplicationError", "ApplicationRejected", "AuthorityViolation",
    "ShadowCapabilityState",
    "apply_artifact",
    "ApplicationLedger", "ledger_head_digest",
    "DurableApplicationLedger",
]
