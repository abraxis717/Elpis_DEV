"""ELPIS SHARED-STATE CONFORMANCE CONTRACT R0 — backend-independent.

Authority: 00_MASTER_CONTRACT_R0.md (ELPIS UNIFIED RALPH/AWR GDN CONTRACT,
coordinating Elpis DEV, 2026-11-04).  This module transcribes the frozen
behavioral API into an executable, machine-checkable form.  It does NOT
implement a backend, a backbone, or any learned operator.

Every name and semantic below is contract-level.  A backend (Spark native
GDN, Yang-Mills pure geometry, or the local reference) must supply an
adapter that satisfies these checks without redefining them.
"""
from __future__ import annotations
import math
from dataclasses import dataclass, FrozenInstanceError
from types import MappingProxyType

CONTRACT_SCHEMA = "elpis.gdn.session.r0"
CONTRACT_REVISION = "R0"

# ---------------------------------------------------------------- operations
OPS = (
    "begin_session", "observe", "read_exact", "upsert_exact", "read_assoc",
    "propose_relation", "transport_and_aggregate", "apply_delta",
    "reinforce", "decay", "evict", "step", "reset", "halt",
)
OP_SET = frozenset(OPS)

# ------------------------------------------------------------------- receipts
STATUS_OK = "OK"
STATUS_REJECTED = "REJECTED"
STATUS_NOT_FOUND = "NOT_FOUND"
STATUS_PENDING = "PENDING"
RECEIPT_STATUSES = (STATUS_OK, STATUS_REJECTED, STATUS_NOT_FOUND, STATUS_PENDING)

# Contract-level rejection codes.  A backend may add detail text but not
# change the code family, otherwise trace attribution becomes meaningless.
REJECT_CODES = frozenset((
    "INVALID_ADDRESS", "DUPLICATE_IDENTITY", "CAPACITY_EXCEEDED",
    "OP_BUDGET", "STEP_BUDGET", "UNADMITTED_RELATION", "UNKNOWN_PARTICIPANT",
    "ROLE_MISMATCH", "ORDINAL_MISMATCH", "FLAGS_MISMATCH",
    "PROVENANCE_MISMATCH", "STALE_TICKET", "UNBOUND_FRAME",
    "UNAUTHORIZED_WRITE", "PROTECTED_RECORD", "DISCONNECTED_PARTICIPANT",
    "INVALID_VALUE", "NONFINITE_VALUE", "STALE_PROPOSAL", "UNKNOWN_RELATION",
))

# ------------------------------------------------------------------ data types
Vec4 = tuple  # exactly four finite floats, float64 reference precision
VecN = tuple


@dataclass(frozen=True)
class SessionSpec:
    """Frozen session configuration (master contract `Configuration` block)."""
    schema: str = CONTRACT_SCHEMA
    model_ref: str = "REFERENCE_NOT_NATIVE_GDN"
    model_sha256: str = ""
    kv_window_tokens: int = 16
    max_active_relations: int = 12
    max_active_entities: int = 13
    max_control_records: int = 64
    max_ops_per_tick: int = 64
    max_steps_per_call: int = 4096
    arithmetic_dtype: str = "float64"
    exact_address_namespace: tuple = (1, 1_000_003)
    relational_namespace: tuple = (0, 4096)
    gauge_rep: str = "SO2x2-R4"
    conv_state_slots: int = 4
    conv_state_dim: int = 8
    finite_bound: float = 1.0e6
    reset_policy: str = "EMPTY"
    salient_temperature: float = 8.0
    transport_tolerance: float = 1.0e-9


def validate_spec(spec) -> tuple:
    """Return tuple of human-readable violations; empty tuple means valid."""
    bad = []
    if not isinstance(spec, SessionSpec):
        return ("SPEC_TYPE",)
    if spec.schema != CONTRACT_SCHEMA:
        bad.append("SCHEMA_MISMATCH")
    if spec.reset_policy != "EMPTY":
        bad.append("RESET_POLICY_NOT_EMPTY")
    if spec.gauge_rep != "SO2x2-R4":
        bad.append("GAUGE_REP_MISMATCH")
    if spec.arithmetic_dtype not in ("float64", "float32"):
        bad.append("DTYPE_UNSUPPORTED")
    for name in ("kv_window_tokens", "max_active_relations",
                 "max_active_entities", "max_control_records",
                 "max_ops_per_tick", "max_steps_per_call",
                 "conv_state_slots", "conv_state_dim"):
        v = getattr(spec, name)
        if type(v) is not int or v < 0:
            bad.append(f"{name}_NOT_NONNEG_INT")
    if spec.kv_window_tokens < 0:
        bad.append("KV_WINDOW_NEGATIVE")
    if spec.max_active_relations < 1:
        bad.append("RELATION_CAPACITY_ZERO")
    if spec.max_active_entities < 1:
        bad.append("ENTITY_CAPACITY_ZERO")
    for name in ("exact_address_namespace", "relational_namespace"):
        ns = getattr(spec, name)
        if (not isinstance(ns, tuple) or len(ns) != 2
                or any(type(x) is not int for x in ns)
                or ns[0] > ns[1]):
            bad.append(f"{name}_MALFORMED")
    if not (math.isfinite(spec.finite_bound) and spec.finite_bound > 0):
        bad.append("FINITE_BOUND_INVALID")
    if not (math.isfinite(spec.salient_temperature) and spec.salient_temperature > 0):
        bad.append("SALIENT_TEMPERATURE_INVALID")
    if not (math.isfinite(spec.transport_tolerance) and spec.transport_tolerance > 0):
        bad.append("TRANSPORT_TOLERANCE_INVALID")
    return tuple(bad)


# ------------------------------------------------------------------ identities
@dataclass(frozen=True)
class AdmissionRecord:
    """Immutable, source-attested authority for one relation identity.

    Authority is carried by provenance/participant/role/ordinal/ticket fields.
    A geometric vector NEVER contributes to admission.
    """
    relation_id: str
    provenance: str
    participants: tuple          # tuple of (entity_id, role, ordinal)
    flags: tuple                 # canonical sorted tuple of str
    ticket: str
    issued_tick: int
    valid_until_tick: int
    protected: bool = False


class AdmissionRegistry:
    """Immutable registry of admitted relation identities."""
    def __init__(self, records):
        table = {}
        for rec in records:
            if not isinstance(rec, AdmissionRecord):
                raise ValueError("REGISTRY_RECORD_TYPE")
            if rec.relation_id in table:
                raise ValueError("REGISTRY_DUPLICATE_RELATION_ID")
            table[rec.relation_id] = rec
        self._table = MappingProxyType(table)

    def get(self, relation_id):
        return self._table.get(relation_id)

    def ids(self):
        return tuple(sorted(self._table))


# --------------------------------------------------------------- contract values
@dataclass(frozen=True)
class Proposal:
    """Typed, uncommitted relational proposal (PROPOSE_RELATION payload)."""
    relation_id: str
    participant: int
    role: str
    ordinal: int
    flags: tuple
    source_frame: str
    value4: tuple
    ticket: str
    issued_tick: int
    provenance: str = ""


@dataclass(frozen=True)
class Receipt:
    status: str
    code: str = ""
    detail: str = ""


def receipt(status, code="", detail=""):
    if status not in RECEIPT_STATUSES:
        raise ValueError("BAD_RECEIPT_STATUS")
    if status == STATUS_REJECTED and code not in REJECT_CODES:
        raise ValueError(f"UNKNOWN_REJECT_CODE:{code}")
    return Receipt(status, code, detail)


def vec4(values, bound):
    """Strict contract-level validation of a signed R4 activation."""
    if not isinstance(values, (tuple, list)) or len(values) != 4:
        raise receipt_invalid("INVALID_VALUE", "VECTOR4_SHAPE")
    out = []
    for x in values:
        if type(x) not in (int, float):
            raise receipt_invalid("INVALID_VALUE", "VECTOR4_TYPE")
        f = float(x)
        if not math.isfinite(f):
            raise receipt_invalid("NONFINITE_VALUE", "VECTOR4_NONFINITE")
        if abs(f) > bound:
            raise receipt_invalid("NONFINITE_VALUE", "VECTOR4_OUT_OF_BOUNDS")
        out.append(f)
    return tuple(out)


class receipt_invalid(Exception):
    """Internal carrier for a contract-level invalid-value receipt."""
    def __init__(self, code, detail):
        self.code, self.detail = code, detail


# ------------------------------------------------------------- adapter protocols
STATE_BANK_METHODS = ("begin_session", "observe", "read_exact", "upsert_exact",
                      "read_assoc", "propose_relation", "transport_and_aggregate",
                      "apply_delta", "reinforce", "decay", "evict", "step",
                      "reset", "halt")
# Introspection surface required for backend-independent evaluation.
INTROSPECTION_METHODS = ("snapshot", "occupancy", "policy")
TRANSPORT_METHODS = ("to_canonical", "to_view", "bind_frame", "list_frames")


def check_adapter(bank, label="STATE_BANK"):
    """Structural conformance of an adapter to the frozen API.

    Returns a list of missing/malformed members.  A backend that cannot
    satisfy this cannot be qualified at any integration level.
    """
    missing = []
    for m in STATE_BANK_METHODS + INTROSPECTION_METHODS:
        if not callable(getattr(bank, m, None)):
            missing.append(f"{label}.{m}:MISSING")
    return missing


def check_transport(transport, label="TRANSPORT"):
    missing = []
    for m in TRANSPORT_METHODS:
        if not callable(getattr(transport, m, None)):
            missing.append(f"{label}.{m}:MISSING")
    return missing


# --------------------------------------------------------------- canonical form
def canonical_state(state):
    """Deterministic, order-normalised, float-tolerant canonical form.

    Used for trace comparison and failure attribution.  Floats are rounded to
    12 decimals so that float64 reassociation noise cannot masquerade as a
    semantic difference; the harness compares with tolerance separately.
    """
    def norm(x):
        if isinstance(x, float):
            return round(x, 12) if abs(x) < 1e12 else float("nan")
        if isinstance(x, (tuple, list)):
            return [norm(v) for v in x]
        if isinstance(x, dict):
            return {k: norm(v) for k, v in sorted(x.items())}
        return x
    return norm(state)


def digest_state(state):
    import json
    blob = json.dumps(canonical_state(state), sort_keys=True, separators=(",", ":"))
    return __import__("hashlib").sha256(blob.encode()).hexdigest()


# --------------------------------------------------------------- op vocabulary
def op(name, *args):
    if name not in OP_SET:
        raise ValueError(f"UNKNOWN_OP:{name}")
    return (name,) + args


OP_BEGIN_SESSION = "begin_session"
OP_RESET = "reset"
OP_HALT = "halt"
