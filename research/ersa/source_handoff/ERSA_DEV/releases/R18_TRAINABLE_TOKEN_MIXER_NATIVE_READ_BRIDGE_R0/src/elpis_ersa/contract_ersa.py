"""ELPIS ERS CONTRACT R0 — backend-neutral relational-attention contract.

Schema:      elpis.ersa.session.r0   (revision R0)
Derived from: elpis.gdn.session.r0 revision R0 (frozen shared-state contract,
              authority 00_MASTER_CONTRACT_R0.md, transcribed by
              UnifiedStateConformance_R0/src/elpis_ssc/contract_r0.py,
              sha256 pinned in reports/source_census_R0.json).

This module does NOT change the predecessor.  It is a versioned, backend-neutral
extension: the predecessor's 14 shared-state operations remain the authority for
state-bank conformance (see adapter_ssc.py), while this schema adds the
attention/retrieval surface the predecessor never defined.

The rejection-code family is inherited UNCHANGED: a backend may add detail text,
never a new code family, otherwise trace attribution becomes meaningless.
"""
from __future__ import annotations
import math
from dataclasses import dataclass
from types import MappingProxyType

SCHEMA = "elpis.ersa.session.r1"
REVISION = "R1"
DERIVED_FROM_SCHEMA = "elpis.ersa.session.r0"
DERIVED_FROM_REVISION = "R0"

# ------------------------------------------------------------- inherited codes
REJECT_CODES = frozenset((
    "INVALID_ADDRESS", "DUPLICATE_IDENTITY", "CAPACITY_EXCEEDED",
    "OP_BUDGET", "STEP_BUDGET", "UNADMITTED_RELATION", "UNKNOWN_PARTICIPANT",
    "ROLE_MISMATCH", "ORDINAL_MISMATCH", "FLAGS_MISMATCH",
    "PROVENANCE_MISMATCH", "STALE_TICKET", "UNBOUND_FRAME",
    "UNAUTHORIZED_WRITE", "PROTECTED_RECORD", "DISCONNECTED_PARTICIPANT",
    "INVALID_VALUE", "NONFINITE_VALUE", "STALE_PROPOSAL", "UNKNOWN_RELATION",
))
# R1 additive extension (versioned proposal; inherited family above unchanged).
R1_REJECT_CODES = frozenset((
    "STAGE_UNKNOWN", "STAGE_MODIFIED", "STAGE_REPLAYED",
    "STAGE_FOREIGN_SESSION", "PENDING_STAGE_LIMIT",
    "STEP_BUDGET_INVALID", "PROTECTED_CAPACITY_FULL",
    "FIELD_VALIDATION",
))
REJECT_CODES = REJECT_CODES | R1_REJECT_CODES
STATUS_OK, STATUS_REJECTED, STATUS_NOT_FOUND, STATUS_PENDING = (
    "OK", "REJECTED", "NOT_FOUND", "PENDING")
RECEIPT_STATUSES = (STATUS_OK, STATUS_REJECTED, STATUS_NOT_FOUND, STATUS_PENDING)

# --------------------------------------------------------------- op vocabulary
# Mandated attention surface (absent from the predecessor schema).
ATTENTION_OPS = (
    "begin_session", "insert_relation", "query_candidates", "select",
    "aggregate", "propose_update", "commit_validated_update", "evict", "reset",
)
# Inherited control surface, semantics unchanged from the predecessor.
CONTROL_OPS = ("step", "halt", "snapshot", "occupancy", "policy")
OPS = ATTENTION_OPS + CONTROL_OPS
OP_SET = frozenset(OPS)


@dataclass(frozen=True)
class SessionSpec:
    """Bounded-by-construction session configuration."""
    schema: str = SCHEMA
    model_ref: str = "REFERENCE_NOT_NATIVE_GDN"
    model_sha256: str = ""
    # inherited capacity fields
    max_active_relations: int = 12
    max_active_entities: int = 13
    max_control_records: int = 64
    max_ops_per_tick: int = 64
    max_steps_per_call: int = 4096
    exact_address_namespace: tuple = (1, 1_000_003)
    relational_namespace: tuple = (0, 4096)
    arithmetic_dtype: str = "float64"
    finite_bound: float = 1.0e6
    conv_state_slots: int = 4
    conv_state_dim: int = 8
    salient_temperature: float = 8.0
    transport_tolerance: float = 1.0e-9
    gauge_rep: str = "SO2x2-R4"
    reset_policy: str = "EMPTY"
    # ERSA additions — every one bounded
    max_candidates: int = 8            # Stage A output ceiling
    product_key_subkeys: int = 4       # k
    product_key_bits: int = 10         # b per sub-key
    adjacency_degree: int = 4          # bounded neighbour lists
    key_feature_dim: int = 4           # discrete key feature width
    eviction_policy: str = "UTILITY_LRU_AUTHORITY"
    pending_stage_limit: int = 64      # inherited from ARWTransportPort
    halflife_ticks: float = 64.0       # freshness decay
    relevance_eps: float = 1.0e-12


def validate_spec(spec) -> tuple:
    bad = []
    if not isinstance(spec, SessionSpec):
        return ("SPEC_TYPE",)
    if spec.schema != SCHEMA:
        bad.append("SCHEMA_MISMATCH")
    if spec.reset_policy != "EMPTY":
        bad.append("RESET_POLICY_NOT_EMPTY")
    if spec.gauge_rep != "SO2x2-R4":
        bad.append("GAUGE_REP_MISMATCH")
    if spec.arithmetic_dtype not in ("float64", "float32"):
        bad.append("DTYPE_UNSUPPORTED")
    for name in ("max_active_relations", "max_active_entities",
                 "max_control_records", "max_ops_per_tick",
                 "max_steps_per_call", "conv_state_slots", "conv_state_dim",
                 "max_candidates", "product_key_subkeys", "product_key_bits",
                 "adjacency_degree", "key_feature_dim", "pending_stage_limit"):
        v = getattr(spec, name)
        if type(v) is not int or v < 0:
            bad.append(f"{name}_NOT_NONNEG_INT")
    if spec.max_active_relations < 1:
        bad.append("RELATION_CAPACITY_ZERO")
    if spec.max_candidates < 1:
        bad.append("CANDIDATE_CEILING_ZERO")
    if spec.product_key_subkeys < 1 or spec.product_key_bits < 1:
        bad.append("PRODUCT_KEY_MALFORMED")
    if spec.adjacency_degree < 0:
        bad.append("ADJACENCY_NEGATIVE")
    for name in ("finite_bound", "salient_temperature", "transport_tolerance",
                 "halflife_ticks", "relevance_eps"):
        v = getattr(spec, name)
        if not (isinstance(v, float) and math.isfinite(v) and v > 0):
            bad.append(f"{name}_INVALID")
    if spec.eviction_policy not in ("UTILITY_LRU_AUTHORITY", "FIFO", "LRU",
                                    "UTILITY_ONLY", "REJECT"):
        bad.append("EVICT_POLICY_UNKNOWN")
    ns_ok = True
    for name in ("exact_address_namespace", "relational_namespace"):
        ns = getattr(spec, name)
        if (not isinstance(ns, tuple) or len(ns) != 2
                or any(type(x) is not int for x in ns) or ns[0] > ns[1]):
            bad.append(f"{name}_MALFORMED")
    return tuple(bad)


# ------------------------------------------------------------------- receipts
@dataclass(frozen=True)
class Receipt:
    status: str
    code: str = ""
    detail: str = ""


class _Invalid(Exception):
    def __init__(self, code, detail=""):
        self.code, self.detail = code, detail


def receipt(status, code="", detail=""):
    if status not in RECEIPT_STATUSES:
        raise ValueError("BAD_RECEIPT_STATUS")
    if status == STATUS_REJECTED and code not in REJECT_CODES:
        raise ValueError(f"UNKNOWN_REJECT_CODE:{code}")
    return Receipt(status, code, detail)


def vec4(values, bound):
    """Strict contract-level validation of a signed R4 activation."""
    if not isinstance(values, (tuple, list)) or len(values) != 4:
        raise _Invalid("INVALID_VALUE", "VECTOR4_SHAPE")
    out = []
    for x in values:
        if type(x) not in (int, float):
            raise _Invalid("INVALID_VALUE", "VECTOR4_TYPE")
        f = float(x)
        if not math.isfinite(f):
            raise _Invalid("NONFINITE_VALUE", "VECTOR4_NONFINITE")
        if abs(f) > bound:
            raise _Invalid("NONFINITE_VALUE", "VECTOR4_OUT_OF_BOUNDS")
        out.append(f)
    return tuple(out)


def key_features(values, dim):
    """Discrete, signed key features.  Identity is NOT numeric proximity."""
    if not isinstance(values, (tuple, list)) or len(values) != dim:
        raise _Invalid("INVALID_VALUE", "KEY_FEATURE_SHAPE")
    out = []
    for x in values:
        if type(x) is not int:
            raise _Invalid("INVALID_VALUE", "KEY_FEATURE_TYPE")
        out.append(int(x))
    return tuple(out)


# ------------------------------------------------------- authority primitives
@dataclass(frozen=True)
class AdmissionRecord:
    """Immutable, source-attested authority for one relation identity.

    Authority is carried by provenance/role/flags/ticket.  A geometric vector
    NEVER contributes to admission (suite B05 of the predecessor harness).
    """
    relation_id: int
    provenance: int
    role: int
    ordinal: int
    flags: tuple
    ticket: int
    issued_tick: int
    valid_until_tick: int = 1 << 62
    protected: bool = False


class AdmissionRegistry:
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


DENIED_PROVENANCE = frozenset()  # per-session set supplied by the authority gate


# ---------------------------------------------------------- adapter protocols
ATTENTION_METHODS = ATTENTION_OPS
INTROSPECTION_METHODS = ("snapshot", "occupancy", "policy")
TRANSPORT_METHODS = ("to_canonical", "to_view", "bind_frame", "list_frames")


def check_adapter(bank, label="ATTENTION_BANK"):
    missing = []
    for m in ATTENTION_METHODS + INTROSPECTION_METHODS:
        if not callable(getattr(bank, m, None)):
            missing.append(f"{label}.{m}:MISSING")
    return missing


def check_transport(transport, label="TRANSPORT"):
    missing = []
    for m in TRANSPORT_METHODS:
        if not callable(getattr(transport, m, None)):
            missing.append(f"{label}.{m}:MISSING")
    return missing


# ------------------------------------------------------------- canonical form
def canonical_state(state):
    def norm(x):
        if isinstance(x, float):
            return round(x, 12) if abs(x) < 1e12 else float("nan")
        if isinstance(x, (tuple, list)):
            return [norm(v) for v in x]
        if isinstance(x, dict):
            return {k: norm(v) for k, v in sorted(x.items())}
        if isinstance(x, (tuple,)):
            return [norm(v) for v in x]
        return x
    return norm(state)


def digest_state(state):
    import json, hashlib
    blob = json.dumps(canonical_state(state), sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(blob.encode()).hexdigest()


def op(name, *args):
    if name not in OP_SET:
        raise ValueError(f"UNKNOWN_OP:{name}")
    return (name,) + args


# flag bits (shared with the predecessor's flag vocabulary)
FLAG_PROTECTED = 1
FLAG_SUPERSEDED = 2
FLAG_SEALED = 4
FLAG_NAMES = {FLAG_PROTECTED: "PROTECTED", FLAG_SUPERSEDED: "SUPERSEDED",
              FLAG_SEALED: "SEALED"}
