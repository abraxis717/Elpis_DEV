"""ERSA RELATION BANK R5 — bounded CPU reference model for relational attention.

REFERENCE_NOT_NATIVE_GDN / REFERENCE_NOT_YM_AUTHORITY.  Every production
container is a fixed-size array allocated at begin_session: there is no
unbounded Python list, dict, or cache serving as session state.  Transient
return values are bounded by the session's candidate ceiling.

Identity is LOGICAL (relation_id in the contract namespace) and is separate
from physical SLOT index; a bounded open-addressing map binds them.  Numerical
proximity of geometry never grants identity or authority.

Determinism: candidate/selection order is the canonical total order
(-round(score,12), relation_id); float sums use math.fsum.
"""
from __future__ import annotations
import math
import numpy as np
from . import contract_ersa as C
from . import geometry_ersa as G

MIX_A = 0x9E3779B97F4A7C15
MASK64 = (1 << 64) - 1


def _splitmix(x: int) -> int:
    x = (x + MIX_A) & MASK64
    x = ((x ^ (x >> 30)) * 0xBF58476D1CE4E5B9) & MASK64
    x = ((x ^ (x >> 27)) * 0x94D049BB133111EB) & MASK64
    return x ^ (x >> 31)


def key_tuple(row) -> tuple:
    return tuple(int(x) for x in row)


def key_fingerprint(key) -> int:
    h = 0xCBF29CE484222325
    for v in key:
        h = _splitmix(h ^ (int(v) & MASK64))
    return h & MASK64


def subkeys(key, k: int, bits: int):
    """Locality-sensitive product-key sub-keys.

    Sub-key j hashes a SLICE of the discrete key (component j mod dim), so
    relations agreeing on any component share a bucket.  Hashing the whole key
    instead would make every sub-key table a near-exact index and destroy
    associative recall — that is the failure mode this design avoids.
    """
    out = []
    d = max(len(key), 1)
    for j in range(k):
        comp = int(key[j % d])
        out.append(_splitmix((comp & MASK64) ^ ((j + 1) * 0x100000001B3))
                   & ((1 << bits) - 1))
    return tuple(out)


RELEVANCE_WEIGHTS = {
    "qcompat": 1.00, "utility": 0.50, "freshness": 0.35, "credibility": 0.40,
    "contradiction": -0.60, "goalrel": 0.45, "protected": 0.30,
    "geomcompat": 0.55,
}


def _cos(a, b) -> float:
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    na = float(np.sqrt(np.sum(a * a)))
    nb = float(np.sqrt(np.sum(b * b)))
    if na < 1e-14 or nb < 1e-14:
        return 0.0
    return float(np.sum(a * b) / (na * nb))


class AdmissionRegistryView:
    """Bounded and duplicate-rejecting copy of immutable admissions.

    A generator is consumed at most limit+1 times, never an unbounded stream.
    No source can silently replace an earlier admission with the same identity.
    """
    def __init__(self, records, limit):
        self._table = {}
        if isinstance(records, C.AdmissionRegistry):
            records = (records.get(i) for i in records.ids())
        for count, rec in enumerate(records):
            if count >= limit:
                raise C._Invalid("CAPACITY_EXCEEDED", "REGISTRY_TOO_LARGE")
            if type(rec) is not C.AdmissionRecord:
                raise C._Invalid("FIELD_VALIDATION", "ADMISSION_RECORD_TYPE")
            if (type(rec.relation_id) is not int or
                    type(rec.ticket) is not int or
                    type(rec.provenance) is not int or
                    type(rec.role) is not int or
                    type(rec.ordinal) is not int or
                    type(rec.issued_tick) is not int or
                    type(rec.valid_until_tick) is not int or
                    rec.issued_tick < 0 or
                    rec.valid_until_tick < rec.issued_tick or
                    type(rec.protected) is not bool or
                    type(rec.flags) is not tuple or
                    any(type(f) is not str or f not in C.FLAG_NAMES.values()
                        for f in rec.flags) or
                    rec.flags != tuple(sorted(set(rec.flags)))):
                raise C._Invalid("FIELD_VALIDATION", "ADMISSION_RECORD_MALFORMED")
            if rec.relation_id in self._table:
                raise C._Invalid("DUPLICATE_IDENTITY", "ADMISSION_DUPLICATE")
            self._table[rec.relation_id] = rec

    def get(self, relation_id):
        return self._table.get(int(relation_id))

    def ids(self):
        return tuple(sorted(self._table))


class ErsBank:
    CAP_SLOTS = None

    def __init__(self):
        self._open = False
        self._spec = None
        self._arrays = None
        self._index = None
        self._tick = 0
        self._ops_this_tick = 0
        self._steps = 0
        self._halted = False
        self._pending = None
        self._pending_used = 0
        self._stage_index = {}
        self._pending_n = 0
        self._counters = None
        # Monotonic, bounded version clock for this bank lifetime. A value
        # cycle cannot revive a stale stage (ABA prevention).
        self._revision_clock = 0

    # ------------------------------------------------------------- lifecycle
    def begin_session(self, spec=None, registry=None, transport=None,
                      authority=None):
        if self._open:
            return C.receipt(C.STATUS_REJECTED, "OP_BUDGET",
                             "SESSION_ALREADY_OPEN")
        spec = spec or C.SessionSpec()
        bad = C.validate_spec(spec)
        if bad:
            return C.receipt(C.STATUS_REJECTED, "INVALID_ADDRESS",
                             "SPEC:" + ";".join(bad))
        # Resource envelope for this qualified CPU-reference configuration.
        # Validate BEFORE any allocations (especially 1<<product_key_bits).
        if (spec.max_active_relations > 8192 or
                spec.max_control_records > 1048576 or
                spec.max_candidates > 8192 or
                spec.pending_stage_limit > 8192 or
                spec.product_key_subkeys > 16 or
                spec.product_key_bits > 16 or
                spec.key_feature_dim > 32 or
                spec.adjacency_degree > 64):
            return C.receipt(C.STATUS_REJECTED, "CAPACITY_EXCEEDED",
                             "ERSA_R3_RESOURCE_ENVELOPE")
        cap = spec.max_active_relations
        try:
            reg = AdmissionRegistryView(registry or (),
                                        max(spec.max_control_records, cap))
        except C._Invalid as exc:
            return C.receipt(C.STATUS_REJECTED, exc.code, exc.detail)
        self._spec = spec
        self._registry = reg
        self._transport = transport
        a = dict(authority or {})
        self._authority = {
            "denied": frozenset(a.get("denied_provenances", ())),
            "expired": frozenset(a.get("expired_tickets", ())),
            "protected": frozenset(a.get("protected_relations", ())),
        }
        m = dict(a.get("modes", {}))
        self._modes = {"discovery": m.get("discovery", "indexed"),
                       "selection": m.get("selection", "ersa")}
        self._arrays = {
            "live": np.zeros(cap, dtype=np.uint8),
            "rid": -np.ones(cap, dtype=np.int64),
            "key": np.zeros((cap, spec.key_feature_dim), dtype=np.int64),
            "prov": np.zeros(cap, dtype=np.int64),
            "part": np.zeros(cap, dtype=np.int64),
            "role": np.zeros(cap, dtype=np.int64),
            "ord": np.zeros(cap, dtype=np.int64),
            "flags": np.zeros(cap, dtype=np.int64),
            "ticket": np.zeros(cap, dtype=np.int64),
            "issued": np.zeros(cap, dtype=np.int64),
            "value": np.zeros((cap, 4), dtype=np.float64),
            "revision": np.zeros(cap, dtype=np.int64),
            "util": np.zeros(cap, dtype=np.float64),
            "cred": np.zeros(cap, dtype=np.float64),
            "goal": np.zeros((cap, spec.key_feature_dim), dtype=np.float64),
            "contra": -np.ones(cap, dtype=np.int64),
            "angle": np.zeros((cap, 2), dtype=np.float64),
        }
        tcap = 1
        while tcap < 2 * cap:
            tcap <<= 1
        nb = 1 << spec.product_key_bits
        k = spec.product_key_subkeys
        self._index = {
            "rid_table": np.zeros(tcap, dtype=np.int64),   # slot+1, 0 empty
            "rid_cap": tcap,
            "table": np.zeros(tcap, dtype=np.int64),       # key exact index
            "table_cap": tcap,
            "pk_head": np.zeros((k, nb), dtype=np.int64),
            "pk_next": np.zeros((k, cap), dtype=np.int64),
            "nbr": np.zeros((cap, max(spec.adjacency_degree, 1)),
                            dtype=np.int64),
            "nbr_n": np.zeros(cap, dtype=np.int64),
        }
        # Zero means zero stages, exactly as accepted by validate_spec().
        self._pending = [None] * int(spec.pending_stage_limit)
        self._pending_used = 0
        self._stage_index = {}
        self._epoch = int(getattr(self, "_epoch", 0)) + 1
        self._pending_n = 0
        self._counters = {"live": 0, "inserts": 0, "evictions": 0,
                          "rejections": 0, "reinforcements": 0,
                          "queries": 0, "proposals": 0, "commits": 0,
                          "logical_tokens": 0}
        self._tick = 0
        self._ops_this_tick = 0
        self._steps = 0
        self._halted = False
        self._open = True
        return C.receipt(C.STATUS_OK, "", "SESSION_OPEN")

    def _budget(self):
        if not self._open:
            return False
        if self._halted:
            return False
        if self._ops_this_tick >= self._spec.max_ops_per_tick:
            return False
        self._ops_this_tick += 1
        return True

    def _admitted_rid(self, rid) -> bool:
        """Authority test on the LOGICAL identity. Geometry never contributes."""
        rec = self._registry.get(int(rid))
        if rec is None:
            return False
        if rec.provenance in self._authority["denied"]:
            return False
        if rec.ticket in self._authority["expired"]:
            return False
        return self._tick <= rec.valid_until_tick

    def _slot_of(self, rid):
        s = self._rid_find(int(rid))
        if s is None:
            return None
        return s if self._arrays["live"][s] else None

    def _is_protected(self, slot):
        """Three independent protection declarations have OR semantics."""
        rid = int(self._arrays["rid"][slot])
        rec = self._registry.get(rid)
        return (bool(int(self._arrays["flags"][slot]) & C.FLAG_PROTECTED)
                or rid in self._authority["protected"]
                or (rec is not None and rec.protected is True))

    def _next_revision(self):
        # On overflow refuse writes; never wrap and make a stale version valid.
        if self._revision_clock >= (1 << 63) - 1:
            return None
        return self._revision_clock + 1

    # --------------------------------------------------------------- identity
    def _rid_find(self, rid):
        idx = self._index
        cap = idx["rid_cap"]
        i = _splitmix(int(rid) & MASK64) % cap
        while idx["rid_table"][i]:
            s = int(idx["rid_table"][i]) - 1
            if int(self._arrays["rid"][s]) == int(rid):
                return s
            i = (i + 1) % cap
        return None

    def _rid_put(self, rid, slot):
        idx = self._index
        cap = idx["rid_cap"]
        i = _splitmix(int(rid) & MASK64) % cap
        while idx["rid_table"][i]:
            i = (i + 1) % cap
        idx["rid_table"][i] = slot + 1

    def _rid_remove(self, rid):
        idx = self._index
        cap = idx["rid_cap"]
        i = _splitmix(int(rid) & MASK64) % cap
        while idx["rid_table"][i]:
            s = int(idx["rid_table"][i]) - 1
            if int(self._arrays["rid"][s]) == int(rid):
                break
            i = (i + 1) % cap
        if not idx["rid_table"][i]:
            return
        idx["rid_table"][i] = 0
        j = (i + 1) % cap
        while idx["rid_table"][j]:
            s = int(idx["rid_table"][j]) - 1
            r = int(self._arrays["rid"][s])
            idx["rid_table"][j] = 0
            self._rid_put(r, s)
            j = (j + 1) % cap

    # --------------------------------------------------------------- insertion
    def insert_relation(self, relation_id, key, value4, provenance=0,
                        participant=0, role=0, ordinal=0, flags=0,
                        ticket=0, utility=1.0, credibility=1.0, goal=None,
                        contradicts=-1, angle=(0.0, 0.0)):
        if not self._budget():
            return C.receipt(C.STATUS_REJECTED, "OP_BUDGET", "OPS_PER_TICK")
        rid = int(relation_id)
        spec = self._spec
        lo, hi = spec.relational_namespace
        if not (lo <= rid <= hi):
            return C.receipt(C.STATUS_REJECTED, "INVALID_ADDRESS",
                             "RID_OUT_OF_NS")
        bad = self._validate_meta(provenance=provenance,
                                  participant=participant, role=role,
                                  ordinal=ordinal, flags=flags,
                                  ticket=ticket, utility=utility,
                                  credibility=credibility, goal=goal,
                                  contradicts=contradicts, angle=angle)
        if bad:
            self._counters["rejections"] += 1
            return C.receipt(C.STATUS_REJECTED, bad[0], bad[1])
        gv = tuple(float(g) for g in goal) if goal is not None else None
        av = (float(angle[0]), float(angle[1]))
        cptr = int(contradicts)
        a = self._arrays
        if self._slot_of(rid) is not None:
            return C.receipt(C.STATUS_REJECTED, "DUPLICATE_IDENTITY",
                             "RID_LIVE")
        rec = self._registry.get(rid)
        if rec is None:
            return C.receipt(C.STATUS_REJECTED, "UNADMITTED_RELATION",
                             "NO_ADMISSION_RECORD")
        if (rec.provenance != int(provenance) or rec.role != int(role)
                or rec.ordinal != int(ordinal)):
            return C.receipt(C.STATUS_REJECTED, "PROVENANCE_MISMATCH",
                             "AUTHORITY_FIELDS_DISAGREE")
        if rec.ticket != int(ticket):
            return C.receipt(C.STATUS_REJECTED, "STALE_TICKET",
                             "TICKET_DISAGREE")
        if rec.flags != tuple(sorted(self._flag_names(flags))):
            return C.receipt(C.STATUS_REJECTED, "FLAGS_MISMATCH",
                             "FLAGS_DISAGREE")
        if not self._admitted_rid(rid):
            return C.receipt(C.STATUS_REJECTED, "UNAUTHORIZED_WRITE",
                             "DENIED_OR_EXPIRED")
        try:
            v = C.vec4(value4, spec.finite_bound)
            kf = C.key_features(key, spec.key_feature_dim)
        except C._Invalid as exc:
            return C.receipt(C.STATUS_REJECTED, exc.code, exc.detail)
        next_revision = self._next_revision()
        if next_revision is None:
            return C.receipt(C.STATUS_REJECTED, "STALE_PROPOSAL",
                             "REVISION_CLOCK_EXHAUSTED")
        if self._counters["live"] >= spec.max_active_relations:
            victim = self._evict_victim(exclude_rid=rid)
            if victim is None:
                self._counters["rejections"] += 1
                return C.receipt(
                    C.STATUS_REJECTED,
                    "PROTECTED_CAPACITY_FULL"
                    if self._all_protected() else "CAPACITY_EXCEEDED",
                    "NO_AUTHORIZED_VICTIM")
            self._evict_locked(victim)
        slot = self._free_slot()
        if slot is None:
            return C.receipt(C.STATUS_REJECTED, "CAPACITY_EXCEEDED",
                             "NO_FREE_SLOT")
        a["live"][slot] = 1
        a["rid"][slot] = rid
        a["key"][slot] = kf
        a["prov"][slot] = int(provenance)
        a["part"][slot] = int(participant)
        a["role"][slot] = int(role)
        a["ord"][slot] = int(ordinal)
        a["flags"][slot] = int(flags)
        a["ticket"][slot] = int(ticket)
        a["issued"][slot] = self._tick
        a["value"][slot] = v
        a["revision"][slot] = next_revision
        self._revision_clock = next_revision
        a["util"][slot] = float(utility)
        a["cred"][slot] = float(credibility)
        if gv is not None:
            a["goal"][slot] = gv
        a["contra"][slot] = cptr
        a["angle"][slot] = av
        if self._transport is not None:
            if int(participant) in self._transport.participant_angles:
                self._transport.bind_frame(rid, int(participant))
                self._transport.set_relation_frame(rid, (float(angle[0]),
                                                         float(angle[1])))
        self._rid_put(rid, slot)
        self._index_insert(slot, kf)
        self._adj_insert(int(participant), slot)
        self._counters["live"] += 1
        self._counters["inserts"] += 1
        return C.receipt(C.STATUS_OK, "", "INSERTED")

    def _validate_meta(self, **kw):
        """R1: type + dimensionality + finiteness + bounds for every numeric
        metadata field. Checked BEFORE any victim choice or mutation."""
        spec = self._spec
        b = spec.finite_bound

        def num(name, x, lo, hi):
            if isinstance(x, bool) or not isinstance(x, (int, float)):
                return ("FIELD_VALIDATION", f"{name}:TYPE")
            if not math.isfinite(float(x)):
                return ("NONFINITE_VALUE", f"{name}:NONFINITE")
            if not (lo <= float(x) <= hi):
                return ("INVALID_VALUE", f"{name}:OUT_OF_BOUNDS")
            return None

        for name, x, lo, hi in (("utility", kw["utility"], 0.0, 4.0),
                                ("credibility", kw["credibility"], 0.0, 1.0)):
            bad = num(name, x, lo, hi)
            if bad:
                return bad

        def vec(name, x, width):
            if x is None:
                return None
            try:
                seq = tuple(x)
            except TypeError:
                return ("FIELD_VALIDATION", f"{name}:TYPE")
            if len(seq) != width:
                return ("FIELD_VALIDATION", f"{name}:WIDTH")
            for g in seq:
                if isinstance(g, bool) or not isinstance(g, (int, float)):
                    return ("FIELD_VALIDATION", f"{name}:ELEM_TYPE")
                if not math.isfinite(float(g)):
                    return ("NONFINITE_VALUE", f"{name}:NONFINITE")
                if not (-b <= float(g) <= b):
                    return ("INVALID_VALUE", f"{name}:OUT_OF_BOUNDS")
            return None

        for name, x, w in (("goal", kw["goal"], spec.key_feature_dim),
                           ("angle", kw["angle"], 2)):
            bad = vec(name, x, w)
            if bad:
                return bad
        for name in ("provenance", "participant", "role", "ordinal",
                     "ticket"):
            x = kw[name]
            if isinstance(x, bool) or not isinstance(x, int):
                return ("FIELD_VALIDATION", f"{name}:TYPE")
        c = kw["contradicts"]
        if isinstance(c, bool) or not isinstance(c, int):
            return ("FIELD_VALIDATION", "contradicts:TYPE")
        if c < -1 or c > 1_000_003:
            return ("INVALID_VALUE", "contradicts:OUT_OF_RANGE")
        f = kw["flags"]
        if isinstance(f, bool) or not isinstance(f, int) or f < 0:
            return ("FIELD_VALIDATION", "flags:TYPE")
        allowed = 0
        for bit in C.FLAG_NAMES:
            allowed |= bit
        if f & ~allowed:
            return ("FIELD_VALIDATION", "flags:UNKNOWN_BITS")
        return None

    def _all_protected(self):
        a = self._arrays
        for s in range(len(a["live"])):
            if a["live"][s] and not self._is_protected(s):
                return False
        return True

    def _free_slot(self):
        a = self._arrays
        for s in range(len(a["live"])):
            if not a["live"][s]:
                return s
        return None

    def _flag_names(self, flags):
        out = []
        for bit, name in C.FLAG_NAMES.items():
            if int(flags) & bit:
                out.append(name)
        return out

    # ------------------------------------------------------------------ index
    def _index_insert(self, slot, kf):
        idx = self._index
        fp = key_fingerprint(kf)
        self._table_put(fp, slot)
        k = self._spec.product_key_subkeys
        b = self._spec.product_key_bits
        for j, sk in enumerate(subkeys(kf, k, b)):
            idx["pk_next"][j][slot] = idx["pk_head"][j][sk]
            idx["pk_head"][j][sk] = slot + 1

    def _table_put(self, fp, slot):
        idx = self._index
        cap = idx["table_cap"]
        i = fp % cap
        while idx["table"][i]:
            i = (i + 1) % cap
        idx["table"][i] = slot + 1

    def _table_find(self, fp):
        idx = self._index
        cap = idx["table_cap"]
        i = fp % cap
        while idx["table"][i]:
            s = int(idx["table"][i]) - 1
            if key_fingerprint(key_tuple(self._arrays["key"][s])) == fp:
                return s
            i = (i + 1) % cap
        return None

    def _index_remove(self, slot, kf):
        idx = self._index
        k = self._spec.product_key_subkeys
        b = self._spec.product_key_bits
        for j, sk in enumerate(subkeys(kf, k, b)):
            prev_head = True
            cur = int(idx["pk_head"][j][sk])
            steps = 0
            while cur and steps <= len(idx["pk_next"][j]):
                if cur - 1 == slot:
                    if prev_head:
                        idx["pk_head"][j][sk] = idx["pk_next"][j][cur - 1]
                    else:
                        idx["pk_next"][j][prev] = idx["pk_next"][j][cur - 1]
                    break
                prev = cur - 1
                prev_head = False
                cur = int(idx["pk_next"][j][cur - 1])
                steps += 1
            idx["pk_next"][j][slot] = 0
        self._table_remove(key_fingerprint(kf))

    def _table_remove(self, fp):
        idx = self._index
        cap = idx["table_cap"]
        i = fp % cap
        while idx["table"][i]:
            s = int(idx["table"][i]) - 1
            if key_fingerprint(key_tuple(self._arrays["key"][s])) == fp:
                break
            i = (i + 1) % cap
        if not idx["table"][i]:
            return
        idx["table"][i] = 0
        j = (i + 1) % cap
        while idx["table"][j]:
            s = int(idx["table"][j]) - 1
            fpj = key_fingerprint(key_tuple(self._arrays["key"][s]))
            idx["table"][j] = 0
            self._table_put(fpj, s)
            j = (j + 1) % cap

    def _adj_insert(self, participant, slot):
        idx = self._index
        deg = idx["nbr"].shape[1]
        p = int(participant) % idx["nbr"].shape[0]
        if idx["nbr_n"][p] < deg:
            idx["nbr"][p][int(idx["nbr_n"][p])] = slot
            idx["nbr_n"][p] += 1

    def _adj_remove(self, participant, slot):
        idx = self._index
        p = int(participant) % idx["nbr"].shape[0]
        n = int(idx["nbr_n"][p])
        for i in range(n):
            if int(idx["nbr"][p][i]) == slot:
                idx["nbr"][p][i] = idx["nbr"][p][n - 1]
                idx["nbr"][p][n - 1] = 0
                idx["nbr_n"][p] = n - 1
                return

    # -------------------------------------------------------------- discovery
    def query_candidates(self, query_key, query_goal=None, participant=None,
                         limit=None, qdir=None, exact_relation_id=None):
        """Discover for fuzzy inference or select an explicitly named identity.

        A key, cosine score, or hash fingerprint is never proof of identity.
        Exact identity queries use the admitted relation-id map and return at
        most one record; they do not claim semantic discovery from a fuzzy key.
        """
        if not self._budget():
            self._last_slots = []
            self._last_candidates = []
            self._last_scored_count = 0
            return C.receipt(C.STATUS_REJECTED, "OP_BUDGET", "OPS_PER_TICK")
        self._last_slots = []
        self._last_candidates = []
        self._last_scored_count = 0
        self._counters["queries"] += 1
        spec = self._spec
        if exact_relation_id is not None:
            if type(exact_relation_id) is not int:
                return C.receipt(C.STATUS_REJECTED, "INVALID_ADDRESS",
                                 "EXACT_RELATION_ID_TYPE")
            rid = exact_relation_id
            slot = self._slot_of(rid)
            if slot is None:
                return C.receipt(C.STATUS_REJECTED, "UNKNOWN_RELATION",
                                 "EXACT_RELATION_NOT_LIVE")
            if not self._admitted_rid(rid):
                return C.receipt(C.STATUS_REJECTED, "UNAUTHORIZED_WRITE",
                                 "EXACT_RELATION_UNADMITTED")
            if self._transported_value(slot) is None:
                return C.receipt(C.STATUS_REJECTED, "UNBOUND_FRAME",
                                 "EXACT_RELATION_UNBOUND_FRAME")
            self._last_slots = [slot]
            self._last_candidates = [rid]
            return C.receipt(C.STATUS_OK, "", "CANDIDATES:"+str(rid)+":identity")
        limit = int(limit if limit is not None else spec.max_candidates)
        limit = max(1, min(limit, spec.max_candidates))
        kf = C.key_features(query_key, spec.key_feature_dim)
        found = {}
        slot = self._table_find(key_fingerprint(kf))
        if slot is not None and self._arrays["live"][slot]:
            found[int(slot)] = "exact"
        k, b = spec.product_key_subkeys, spec.product_key_bits
        idx = self._index
        hits = {}
        for j, sk in enumerate(subkeys(kf, k, b)):
            cur = int(idx["pk_head"][j][sk])
            steps = 0
            while cur and steps <= spec.max_active_relations:
                s = cur - 1
                if self._arrays["live"][s]:
                    hits[s] = hits.get(s, 0) + 1
                    if s not in found:
                        found[s] = "product_key"
                cur = int(idx["pk_next"][j][s])
                steps += 1
        if participant is not None:
            p = int(participant) % idx["nbr"].shape[0]
            for i in range(int(idx["nbr_n"][p])):
                s = int(idx["nbr"][p][i])
                if self._arrays["live"][s] and s not in found:
                    found[s] = "adjacency"
        ordered = (sorted(s for s in found if found[s] == "exact") +
                   sorted(s for s in found if found[s] != "exact"))
        mode_d = self._modes["discovery"]
        if mode_d == "score_complete":
            # Complete bounded scan: guarantees no live, admitted record is
            # missed by the discovery index, NOT task-grounded semantic recall.
            # The O(capacity) scan cost is always observable through diagnostics.
            found = {s: "complete" for s in range(spec.max_active_relations)
                     if self._arrays["live"][s]}
        if mode_d in ("score_indexed", "score_complete"):
            # Evaluate the whole bounded index-hit union before truncation.
            # This remains approximate when relevant records have no index hits.
            valid=[]
            for s in found:
                rid=int(self._arrays["rid"][s])
                if not self._admitted_rid(rid):
                    continue
                tv=self._transported_value(s)
                if tv is None:continue
                feats=self._features(s, query_key, query_goal,
                                     qdir if qdir is not None else query_key,tv)
                valid.append((-round(self._score(feats),12),rid,s))
            valid.sort()
            self._last_scored_count = len(valid)
            ordered=[s for _,_,s in valid[:limit]]
        elif mode_d == "dense":
            ordered = [s for s in range(spec.max_active_relations)
                       if self._arrays["live"][s]]
        elif mode_d == "exact_only":
            ordered = [s for s in ordered if found.get(s) == "exact"]
        elif mode_d == "identity_first":
            ordered = self._order_identity_first(found, hits, limit)
        elif mode_d == "channels":
            ordered = self._order_channels(found, hits, participant, limit)
        else:
            ordered = ordered[:limit]
        self._last_slots = ordered
        self._last_candidates = [int(self._arrays["rid"][s]) for s in ordered]
        return C.receipt(C.STATUS_OK, "", "CANDIDATES:" +
                         ",".join(f"{self._arrays['rid'][s]}:"
                                  f"{found.get(s, 'dense')}" for s in ordered))

    # ------------------------------------------------- candidate ordering (R2)
    def _order_identity_first(self, found, hits, limit):
        """Candidate (a): identity-first hierarchical retrieval.

        Tier 1 exact identity (never displaced by truncation), tier 2
        adjacency, tier 3 product-key ranked by the number of matching
        sub-keys (bounded collision resolution), then slot for determinism.
        """
        exact = sorted(s for s in found if found[s] == "exact")
        adj = sorted(s for s in found if found[s] == "adjacency")
        pk = [s for s in found if found[s] == "product_key"]
        pk.sort(key=lambda s: (-hits.get(s, 0), s))
        return (exact + adj + pk)[:max(1, int(limit))]

    def _order_channels(self, found, hits, participant, limit):
        """Candidate (b): explicit protected-evidence and contradiction
        channels plus per-provenance coverage in the product-key tier.

        Channels: exact identity, protected records, adjacency,
        contradiction partners of the seeded candidates, then product-key
        candidates interleaved round-robin across provenances (each group
        ranked by sub-key hits) so a single dominant provenance cannot
        monopolise the candidate ceiling.
        """
        a = self._arrays
        n = self._spec.max_active_relations
        exact = sorted(s for s in found if found[s] == "exact")
        adj = sorted(s for s in found if found[s] == "adjacency")
        prot = [s for s in range(n)
                if self._arrays["live"][s] and
                (int(a["flags"][s]) & C.FLAG_PROTECTED
                 or int(a["rid"][s]) in self._authority["protected"])]
        seeds = exact + adj
        contra = []
        for s in seeds:
            target = int(a["contra"][s])
            if target <= 0:          # -1 = no contradiction pointer
                continue
            contra += [t for t in range(n)
                       if self._arrays["live"][t]
                       and int(a["contra"][t]) == target and t not in seeds]
        pk = [s for s in found if found[s] == "product_key"]
        groups = {}
        for s in pk:
            groups.setdefault(int(a["prov"][s]), []).append(s)
        for g in groups.values():
            g.sort(key=lambda s: (-hits.get(s, 0), s))
        rr, depth = [], 0
        while any(len(g) > depth for g in groups.values()):
            for g in groups.values():
                if len(g) > depth:
                    rr.append(g[depth])
            depth += 1
        out, seen = [], set()
        for s in exact + prot + adj + contra + rr:
            if s not in seen:
                seen.add(s)
                out.append(s)
        return out[:max(1, int(limit))]

    # ------------------------------------------------------------- relevance
    def _features(self, slot, qkey, qgoal, qdir, tv=None):
        a = self._arrays
        rid = int(a["rid"][slot])
        kf = a["key"][slot].astype(float)
        qk = np.asarray(qkey, dtype=float)
        qcompat = _cos(kf, qk)
        age = max(0, self._tick - int(a["issued"][slot]))
        fresh = math.exp(-age / self._spec.halflife_ticks)
        util = float(a["util"][slot])
        cred = float(a["cred"][slot])
        contra = int(a["contra"][slot])
        contra_slot = self._slot_of(contra)
        contradiction = (-1.0 if (contra_slot is not None
                                  and self._admitted_rid(contra)) else 0.0)
        goal = a["goal"][slot]
        goalrel = _cos(goal, np.asarray(qgoal, dtype=float)) if qgoal else 0.0
        protected = 1.0 if self._is_protected(slot) else 0.0
        if tv is None:
            tv = self._transported_value(slot)
        geom = _cos(np.asarray(tv or (0.0,) * 4, dtype=float),
                    np.asarray(qdir, dtype=float))
        return {"qcompat": qcompat, "utility": util, "freshness": fresh,
                "credibility": cred, "contradiction": contradiction,
                "goalrel": goalrel, "protected": protected,
                "geomcompat": geom}

    def _score(self, feats):
        return math.fsum(RELEVANCE_WEIGHTS[k] * v for k, v in feats.items())

    def _transported_value(self, slot):
        if self._transport is None:
            return tuple(self._arrays["value"][slot])
        rid = int(self._arrays["rid"][slot])
        try:
            return tuple(self._transport.to_canonical(
                rid, int(self._arrays["part"][slot]),
                tuple(self._arrays["value"][slot])))
        except PermissionError:
            return None

    def select(self, query_key, query_goal=None, participant=None, k=None,
               qdir=None, exact_relation_id=None):
        if not self._budget():
            self._last_slots = []
            self._last_selection = []
            self._last_slots_sel = []
            self._last_evidence = []
            return C.receipt(C.STATUS_REJECTED, "OP_BUDGET", "OPS_PER_TICK")
        spec = self._spec
        k = int(k if k is not None else spec.max_candidates)
        k = max(1, min(k, spec.max_candidates))
        discovery = self.query_candidates(query_key, query_goal, participant,
                                          qdir=qdir,
                                          exact_relation_id=exact_relation_id)
        if discovery.status != C.STATUS_OK:
            self._last_slots = []
            self._last_candidates = []
            self._last_selection = []
            self._last_slots_sel = []
            self._last_evidence = []
            return discovery
        slots = getattr(self, "_last_slots", [])
        scored = []
        self._excluded = []
        a = self._arrays
        for s in slots:
            rid = int(a["rid"][s])
            if not self._admitted_rid(rid) or int(a["prov"][s]) in self._authority["denied"]:
                self._excluded.append((rid, "AUTHORITY_DENIED"))
                continue
            tv = self._transported_value(s)
            if tv is None:
                self._excluded.append((rid, "UNBOUND_FRAME"))
                continue
            feats = self._features(s, query_key, query_goal,
                                   qdir if qdir is not None else query_key, tv)
            scored.append((-round(self._score(feats), 12), rid, s, feats))
        mode = self._modes["selection"]
        scored.sort()
        if mode == "utility_only":
            scored.sort(key=lambda t: (-round(float(a["util"][t[2]]), 12),
                                       t[1]))
        elif mode in ("fifo", "lru"):
            scored.sort(key=lambda t: (int(a["issued"][t[2]]), t[1]))
        elif mode == "random":
            scored.sort(key=lambda t: (_splitmix(t[1] * 7919 + self._tick),
                                       t[1]))
        chosen = scored[:k]
        self._last_selection = [(rid, f) for _n, rid, _s, f in chosen]
        self._last_slots_sel = [s for _n, _rid, s, _f in chosen]
        return C.receipt(C.STATUS_OK, "", "SELECTED:" +
                         ",".join(str(rid) for _n, rid, _s, _f in chosen))

    def aggregate(self, query_key, query_goal=None, participant=None,
                  k=None, qdir=None, exact_relation_id=None):
        if not self._budget():
            return C.receipt(C.STATUS_REJECTED, "OP_BUDGET", "OPS_PER_TICK")
        sel = self.select(query_key, query_goal, participant, k, qdir,
                          exact_relation_id=exact_relation_id)
        if sel.status != C.STATUS_OK:
            self._last_selection = []
            self._last_slots_sel = []
            self._last_evidence = []
            self._last_aggregate = None
            self._last_collective = None
            self._counters["rejections"] += 1
            return C.receipt(C.STATUS_REJECTED, sel.code or "OP_BUDGET",
                             "NESTED_SELECT_REJECTED:" + str(sel.code))
        a = self._arrays
        transported, weights, evidence = [], [], []
        for (rid, feats), slot in zip(self._last_selection,
                                      getattr(self, "_last_slots_sel", [])):
            v = self._transported_value(slot)
            if v is None:
                continue
            transported.append(v)
            weights.append(self._score(feats))
            evidence.append({"relation_id": int(rid),
                             "provenance": int(a["prov"][slot]),
                             "weight": self._score(feats),
                             "authority": "admitted",
                             "utility": float(a["util"][slot])})
        self._last_evidence = evidence
        salient, collective = G.aggregate_signed(transported, weights)
        self._last_aggregate = salient
        self._last_collective = collective
        return C.receipt(C.STATUS_OK, "", "AGGREGATE:" +
                         ",".join(f"{x:.12g}" for x in salient))

    # --------------------------------------- explicit read-set / identity seam
    def _clear_read(self):
        self._last_slots = []
        self._last_candidates = []
        self._last_slots_sel = []
        self._last_selection = []
        self._last_evidence = []
        self._last_aggregate = None
        self._last_collective = None
        self._last_scored_count = 0

    def aggregate_scoped(self, relation_ids, query_key, query_goal=None,
                         qdir=None, *, revision_pins=None, expected_epoch=None):
        """Explicit bounded identity-set read; NOT semantic discovery.

        Caller must independently establish right to request this set.  Pins
        bind reads to bank generation, preventing ABA/stale-context reuse.
        Refuse the ENTIRE set if any member is missing, denied, stale, or
        transport-unbound. No partial evidence or residual old aggregate.
        """
        self._clear_read()
        if not self._budget():
            return C.receipt(C.STATUS_REJECTED, "OP_BUDGET", "OPS_PER_TICK")
        return self._aggregate_scoped_validated(
            relation_ids,query_key,query_goal,qdir,revision_pins,expected_epoch)

    def _aggregate_scoped_validated(self,relation_ids,query_key,query_goal,
                                   qdir,revision_pins,expected_epoch):
        # Internal entry only, after one op-budget charge. Never exposed as a
        # public control surface or authority-bearing token.
        spec = self._spec
        if (type(relation_ids) not in (tuple, list) or not relation_ids or
                len(relation_ids) > spec.max_candidates or
                any(type(x) is not int for x in relation_ids) or
                len(set(relation_ids)) != len(relation_ids)):
            return C.receipt(C.STATUS_REJECTED, "INVALID_ADDRESS", "BAD_RELATION_SET")
        if expected_epoch is not None and (type(expected_epoch) is not int or
                expected_epoch != self._epoch):
            return C.receipt(C.STATUS_REJECTED, "STALE_TICKET", "SESSION_EPOCH")
        if revision_pins is not None and (
                type(revision_pins) not in (tuple, list) or
                len(revision_pins) != len(relation_ids) or
                any(type(x) is not int or x < 1 for x in revision_pins)):
            return C.receipt(C.STATUS_REJECTED, "INVALID_ADDRESS", "BAD_VERSION_PINS")
        try:
            qk = C.key_features(query_key, spec.key_feature_dim)
            qd = C.vec4(qdir if qdir is not None else qk, spec.finite_bound)
            qg = (C.vec4(query_goal, spec.finite_bound)
                  if query_goal is not None and spec.key_feature_dim == 4
                  else query_goal)
            if (qg is not None and (type(qg) not in (tuple, list) or
                    len(qg) != spec.key_feature_dim or
                    any(type(x) not in (int, float) or not math.isfinite(x)
                        for x in qg))):
                raise C._Invalid("INVALID_VALUE", "BAD_GOAL")
        except C._Invalid as exc:
            return C.receipt(C.STATUS_REJECTED, exc.code, exc.detail)
        a = self._arrays
        collected = []
        for i, rid in enumerate(relation_ids):
            slot = self._slot_of(rid)
            if slot is None:
                return C.receipt(C.STATUS_REJECTED, "UNKNOWN_RELATION", "RELATION_SET_NOT_LIVE")
            if (not self._admitted_rid(rid) or
                    int(a['prov'][slot]) in self._authority['denied']):
                return C.receipt(C.STATUS_REJECTED, "UNAUTHORIZED_WRITE", "RELATION_SET_DENIED")
            if revision_pins is not None and int(a['revision'][slot]) != revision_pins[i]:
                return C.receipt(C.STATUS_REJECTED, "STALE_TICKET", "RELATION_GENERATION")
            value = self._transported_value(slot)
            if value is None:
                return C.receipt(C.STATUS_REJECTED, "UNBOUND_FRAME", "RELATION_SET_FRAME")
            features = self._features(slot,qk,qg,qd,value)
            collected.append((rid,slot,value,features,self._score(features)))
        # Commit only when all records are validated; stable caller order.
        self._last_candidates = list(relation_ids)
        self._last_slots = [z[1] for z in collected]
        self._last_selection = [(z[0],z[3]) for z in collected]
        self._last_slots_sel = list(self._last_slots)
        self._last_evidence = [dict(relation_id=z[0],
                provenance=int(a['prov'][z[1]]),weight=z[4],
                authority='admitted_not_query_permission',
                utility=float(a['util'][z[1]])) for z in collected]
        self._last_aggregate,self._last_collective = G.aggregate_signed(
            [z[2] for z in collected], [z[4] for z in collected])
        return C.receipt(C.STATUS_OK,"","SCOPED_SET:"+
                         ",".join(str(x) for x in relation_ids))

    def aggregate_reciprocal_contradiction(self, seed_relation_id, query_key,
                                           query_goal=None, qdir=None,
                                           *, revision_pins=None,
                                           expected_epoch=None):
        """Read only a reciprocally declared contradiction edge.

        No fuzzy nearest-neighbor substitution. Reject dangling, unidirectional
        or unauthorized links; both endpoints have to fit the candidate budget.
        """
        self._clear_read()
        if not self._budget():
            return C.receipt(C.STATUS_REJECTED,'OP_BUDGET','OPS_PER_TICK')
        if type(seed_relation_id) is not int:
            return C.receipt(C.STATUS_REJECTED,'INVALID_ADDRESS','SEED_TYPE')
        if not self._open or self._spec.max_candidates < 2:
            return C.receipt(C.STATUS_REJECTED,'CAPACITY_EXCEEDED','PAIR_CEILING')
        s = self._slot_of(seed_relation_id)
        if s is None:
            return C.receipt(C.STATUS_REJECTED,'UNKNOWN_RELATION','SEED_NOT_LIVE')
        partner = int(self._arrays['contra'][s])
        t = self._slot_of(partner)
        if partner < 0 or partner == seed_relation_id or t is None or int(self._arrays['contra'][t]) != seed_relation_id:
            return C.receipt(C.STATUS_REJECTED,'UNADMITTED_RELATION','NONRECIPROCAL_CONTRADICTION')
        return self._aggregate_scoped_validated(
            (seed_relation_id,partner),query_key,query_goal,qdir,
            revision_pins,expected_epoch)

    # ---------------------------------------------------------- proposal/commit
    def propose_update(self, target, delta4, gain=0.25):
        if not self._budget():
            return C.receipt(C.STATUS_REJECTED, "OP_BUDGET", "OPS_PER_TICK")
        rid = int(target)
        slot = self._slot_of(rid)
        a = self._arrays
        if slot is None:
            return C.receipt(C.STATUS_REJECTED, "UNKNOWN_RELATION",
                             "TARGET_DEAD")
        if not self._admitted_rid(rid):
            return C.receipt(C.STATUS_REJECTED, "UNAUTHORIZED_WRITE",
                             "TARGET_NOT_ADMITTED")
        if (int(a["flags"][slot]) & C.FLAG_SEALED) or (
                self._is_protected(slot) and gain > 0.5):
            return C.receipt(C.STATUS_REJECTED, "PROTECTED_RECORD",
                             "PROTECTED_GAIN_LIMIT")
        if not (0.0 <= float(gain) <= 1.0) or not math.isfinite(float(gain)):
            return C.receipt(C.STATUS_REJECTED, "INVALID_VALUE", "GAIN_RANGE")
        try:
            d = C.vec4(delta4, self._spec.finite_bound)
        except C._Invalid as exc:
            return C.receipt(C.STATUS_REJECTED, exc.code, exc.detail)
        old = tuple(float(x) for x in a["value"][slot])
        nxt = tuple(math.fsum((old[i], float(gain) * d[i]))
                    for i in range(4))
        if self._pending_used >= len(self._pending):
            self._counters["rejections"] += 1
            return C.receipt(C.STATUS_REJECTED, "PENDING_STAGE_LIMIT",
                             "PENDING_STAGES_FULL")
        tok = self._next_token()
        if tok is None:
            return C.receipt(C.STATUS_REJECTED, "STAGE_UNKNOWN",
                             "TOKEN_COUNTER_EXHAUSTED")
        entries = ((rid, old, nxt),)
        idx = next(i for i, p in enumerate(self._pending) if p is None)
        self._pending[idx] = {
            "token": tok, "session_epoch": self._epoch,
            "issued_tick": self._tick,
            "target_revisions": ((rid, int(a["revision"][slot])),),
            "entries": tuple((int(r),
                              tuple(float(x) for x in p),
                              tuple(float(x) for x in c))
                             for (r, p, c) in entries)}
        self._stage_index[tok] = idx
        self._pending_used += 1
        self._counters["proposals"] += 1
        return {"token": tok, "entries": entries, "issued_tick": self._tick}

    def _next_token(self):
        if getattr(self, "_token", 0) >= (1 << 63) - 1:
            return None
        self._token = getattr(self, "_token", 0) + 1
        return self._token

    def commit_validated_update(self, stage):
        """Single-target, version-bound commit with an undo journal.

        Stage entries are authenticated against the private issuance record.
        The physical slot is resolved exactly once and never reconstructed in
        the write path; missing targets cannot become NumPy broadcast writes.
        """
        if not self._budget():
            return C.receipt(C.STATUS_REJECTED, "OP_BUDGET", "OPS_PER_TICK")
        if type(stage) is not dict or type(stage.get("token")) is not int:
            return C.receipt(C.STATUS_REJECTED, "STAGE_UNKNOWN", "BAD_STAGE")
        tok = stage["token"]
        idx = self._stage_index.get(tok)
        auth = (self._pending[idx] if type(idx) is int
                and 0 <= idx < len(self._pending) else None)
        if auth is None or auth.get("token") != tok:
            return C.receipt(C.STATUS_REJECTED, "STAGE_UNKNOWN",
                             "TOKEN_NOT_ISSUED_OR_ALREADY_CONSUMED")
        if auth["session_epoch"] != self._epoch:
            return C.receipt(C.STATUS_REJECTED, "STAGE_FOREIGN_SESSION",
                             "SESSION_EPOCH_MISMATCH")
        if (type(stage.get("issued_tick")) is not int or
                stage["issued_tick"] != auth["issued_tick"]):
            return C.receipt(C.STATUS_REJECTED, "STAGE_MODIFIED",
                             "ISSUED_TICK_DISAGREE")
        raw = stage.get("entries")
        if (type(raw) is not tuple or len(raw) != 1 or
                any(type(e) is not tuple or len(e) != 3 or
                    type(e[0]) is not int or type(e[1]) is not tuple or
                    type(e[2]) is not tuple or len(e[1]) != 4 or
                    len(e[2]) != 4 for e in raw)):
            return C.receipt(C.STATUS_REJECTED, "STAGE_MODIFIED",
                             "ENTRIES_SHAPE")
        try:
            sub = tuple((r, tuple(float(x) for x in p),
                         tuple(float(x) for x in c)) for r, p, c in raw)
        except (TypeError, ValueError, OverflowError):
            return C.receipt(C.STATUS_REJECTED, "STAGE_MODIFIED",
                             "ENTRIES_MALFORMED")
        if sub != auth["entries"]:
            return C.receipt(C.STATUS_REJECTED, "STAGE_MODIFIED",
                             "ENTRIES_DIVERGE_FROM_ISSUED")
        (rid, prev, cur), = sub
        slot = self._slot_of(rid)
        if slot is None:
            return C.receipt(C.STATUS_REJECTED, "UNKNOWN_RELATION",
                             "TARGET_DEAD")
        a = self._arrays
        # Both value and generation must agree. The generation is monotonically
        # renewed on insertion and on every committed value update.
        if (auth.get("target_revisions") !=
                ((rid, int(a["revision"][slot])),) or
                tuple(float(x) for x in a["value"][slot]) != prev):
            return C.receipt(C.STATUS_REJECTED, "STALE_PROPOSAL",
                             "STALE_ARW_STATE_OR_GENERATION")
        if not self._admitted_rid(rid):
            return C.receipt(C.STATUS_REJECTED, "UNAUTHORIZED_WRITE",
                             "NOT_ADMITTED")
        try:
            C.vec4(cur, self._spec.finite_bound)
        except C._Invalid as exc:
            return C.receipt(C.STATUS_REJECTED, exc.code, exc.detail)
        new_version = self._next_revision()
        if new_version is None:
            return C.receipt(C.STATUS_REJECTED, "STALE_PROPOSAL",
                             "REVISION_CLOCK_EXHAUSTED")
        # Local, bounded undo journal (one 4-vector, no full-bank copy).
        try:
            old_row = np.array(a["value"][slot], copy=True)
            old_revision = int(a["revision"][slot])
            old_used = self._pending_used
            old_commits = self._counters["commits"]
            old_clock = self._revision_clock
        except Exception:
            return C.receipt(C.STATUS_REJECTED, "INVALID_VALUE",
                             "COMMIT_PREPARE_ABORT")
        try:
            # All post-write bookkeeping is guarded by the same undo journal.
            a["value"][slot] = cur
            a["revision"][slot] = new_version
            self._revision_clock = new_version
            self._pending[idx] = None
            del self._stage_index[tok]
            self._pending_used = old_used - 1
            self._counters["commits"] = old_commits + 1
        except Exception:
            # A bookkeeping or NumPy write failure must not report success.
            # Restore the validated physical slot, never re-resolve by rid.
            try:
                a["value"][slot] = old_row
                a["revision"][slot] = old_revision
                self._revision_clock = old_clock
                self._pending[idx] = auth
                self._stage_index[tok] = idx
                self._pending_used = old_used
                self._counters["commits"] = old_commits
            except Exception:
                # Failure to roll back means this session is no longer safe.
                self._halted = True
                return C.receipt(C.STATUS_REJECTED, "INVALID_VALUE",
                                 "COMMIT_ROLLBACK_FAILED_SESSION_HALTED")
            return C.receipt(C.STATUS_REJECTED, "INVALID_VALUE",
                             "COMMIT_ATOMIC_ABORT")
        return C.receipt(C.STATUS_OK, "", "COMMITTED_ATOMIC:1")

    # ---------------------------------------------------------------- eviction
    def _evict_victim(self, exclude_rid=None):
        """Authorized eviction: protected and denied records are never victims."""
        a = self._arrays
        spec = self._spec
        live = [s for s in range(spec.max_active_relations)
                if a["live"][s] and int(a["rid"][s]) != int(exclude_rid)]
        if not live:
            return None
        eligible = [s for s in live if not self._is_protected(s)]
        if not eligible:
            return None
        pool = eligible
        pol = spec.eviction_policy
        if pol == "REJECT":
            return None
        if pol in ("FIFO", "LRU"):
            return min(pool, key=lambda s: (int(a["issued"][s]),
                                            int(a["rid"][s])))
        if pol == "UTILITY_ONLY":
            return min(pool, key=lambda s: (float(a["util"][s]),
                                            int(a["rid"][s])))
        return min(pool, key=lambda s: (float(a["util"][s]),
                                        int(a["issued"][s]),
                                        int(a["rid"][s])))

    def _evict_locked(self, slot):
        a = self._arrays
        rid = int(a["rid"][slot])
        self._index_remove(slot, key_tuple(a["key"][slot]))
        self._adj_remove(int(a["part"][slot]), slot)
        self._rid_remove(rid)
        a["live"][slot] = 0
        a["rid"][slot] = -1
        self._counters["live"] -= 1
        self._counters["evictions"] += 1

    def evict(self, relation_id):
        if not self._budget():
            return C.receipt(C.STATUS_REJECTED, "OP_BUDGET", "OPS_PER_TICK")
        rid = int(relation_id)
        slot = self._slot_of(rid)
        a = self._arrays
        if slot is None:
            return C.receipt(C.STATUS_REJECTED, "UNKNOWN_RELATION", "NOT_LIVE")
        if self._is_protected(slot):
            return C.receipt(C.STATUS_REJECTED, "PROTECTED_RECORD",
                             "EVICT_REFUSED")
        self._evict_locked(slot)
        return C.receipt(C.STATUS_OK, "", "EVICTED")

    # --------------------------------------------------------------- control
    def reinforce(self, relation_id, coactivation=1.0, feedback=1.0):
        if not self._budget():
            return C.receipt(C.STATUS_REJECTED, "OP_BUDGET", "OPS_PER_TICK")
        rid = int(relation_id)
        slot = self._slot_of(rid)
        a = self._arrays
        if slot is None:
            return C.receipt(C.STATUS_REJECTED, "UNKNOWN_RELATION", "NOT_LIVE")
        if not self._admitted_rid(rid):
            return C.receipt(C.STATUS_REJECTED, "UNAUTHORIZED_WRITE",
                             "NOT_ADMITTED")
        a["util"][slot] = min(4.0, float(a["util"][slot])
                              + 0.25 * float(coactivation) * float(feedback))
        self._counters["reinforcements"] += 1
        return C.receipt(C.STATUS_OK, "", "REINFORCED")

    def decay(self, ticks=1):
        if not self._budget():
            return C.receipt(C.STATUS_REJECTED, "OP_BUDGET", "OPS_PER_TICK")
        a = self._arrays
        spec = self._spec
        for s in range(spec.max_active_relations):
            if a["live"][s]:
                a["util"][s] *= math.exp(-float(ticks) / spec.halflife_ticks)
        return C.receipt(C.STATUS_OK, "", "DECAYED")

    def step(self, action_budget=1):
        if not self._open:
            return C.receipt(C.STATUS_REJECTED, "OP_BUDGET", "NO_SESSION")
        if self._halted:
            return C.receipt(C.STATUS_REJECTED, "STEP_BUDGET", "HALTED")
        if isinstance(action_budget, bool) or not isinstance(action_budget, int):
            return C.receipt(C.STATUS_REJECTED, "STEP_BUDGET_INVALID",
                             "TYPE")
        if action_budget <= 0:
            return C.receipt(C.STATUS_REJECTED, "STEP_BUDGET_INVALID",
                             "NONPOSITIVE")
        if action_budget > self._spec.max_steps_per_call:
            return C.receipt(C.STATUS_REJECTED, "STEP_BUDGET_INVALID",
                             "OVER_CALL_BUDGET")
        if self._steps + action_budget > (self._spec.max_ops_per_tick
                                          * max(1, self._spec.max_steps_per_call)):
            return C.receipt(C.STATUS_REJECTED, "STEP_BUDGET",
                             "CUMULATIVE_WORK_BOUND")
        self._tick = int(self._tick + action_budget)
        self._steps += 1
        self._ops_this_tick = 0
        return C.receipt(C.STATUS_OK, "", "STEPPED")

    def halt(self):
        self._halted = True
        return C.receipt(C.STATUS_OK, "", "HALTED")

    def reset(self):
        """EMPTY reset policy: every slot/control record returns to empty."""
        self._arrays = None
        self._index = None
        self._counters = None
        self._tick = 0
        self._steps = 0
        self._ops_this_tick = 0
        self._halted = False
        self._pending = None
        self._pending_used = 0
        self._stage_index = {}
        self._pending_n = 0
        self._registry = None
        self._transport = None
        self._authority = None
        self._spec = None
        self._open = False
        for attr in ("_last_candidates", "_last_selection", "_last_aggregate", "_last_scored_count",
                     "_last_slots", "_last_slots_sel", "_last_evidence",
                     "_excluded", "_last_collective"):
            self.__dict__.pop(attr, None)
        return C.receipt(C.STATUS_OK, "", "RESET_EMPTY")

    def occupancy(self):
        if self._arrays is None:
            return {"live": 0, "capacity": 0, "tick": 0}
        return {"live": int(self._counters["live"]),
                "capacity": int(self._spec.max_active_relations),
                "tick": int(self._tick),
                "inserts": int(self._counters["inserts"]),
                "evictions": int(self._counters["evictions"]),
                "rejections": int(self._counters["rejections"]),
                "queries": int(self._counters["queries"]),
                "proposals": int(self._counters["proposals"]),
                "commits": int(self._counters["commits"]),
                "reinforcements": int(self._counters["reinforcements"])}

    def policy(self):
        return {"schema": self._spec.schema if self._spec else "",
                "eviction": self._spec.eviction_policy if self._spec else "",
                "gauge": self._spec.gauge_rep if self._spec else "",
                "product_key": (self._spec.product_key_subkeys,
                                self._spec.product_key_bits)
                                if self._spec else (),
                "discovery_mode": (self._modes["discovery"] if self._spec else ""),
                "score_indexed_guarantee": "RANKED_OVER_INDEX_HITS_ONLY"}

    def snapshot(self):
        if self._arrays is None:
            return {"relations": {}, "control": {}, "tick": 0}
        a, spec = self._arrays, self._spec
        rel = {}
        for s in range(spec.max_active_relations):
            if a["live"][s]:
                rid = int(a["rid"][s])
                rel[rid] = {"key": key_tuple(a["key"][s]),
                            "provenance": int(a["prov"][s]),
                            "participant": int(a["part"][s]),
                            "role": int(a["role"][s]),
                            "ordinal": int(a["ord"][s]),
                            "flags": int(a["flags"][s]),
                            "ticket": int(a["ticket"][s]),
                            "issued": int(a["issued"][s]),
                            "value": tuple(float(x) for x in a["value"][s]),
                            "utility": float(a["util"][s]),
                            "credibility": float(a["cred"][s]),
                            "contradicts": int(a["contra"][s]),
                            "angle": tuple(float(x) for x in a["angle"][s])}
        return {"relations": rel, "tick": int(self._tick),
                "control": {"steps": self._steps, "halted": self._halted}}
