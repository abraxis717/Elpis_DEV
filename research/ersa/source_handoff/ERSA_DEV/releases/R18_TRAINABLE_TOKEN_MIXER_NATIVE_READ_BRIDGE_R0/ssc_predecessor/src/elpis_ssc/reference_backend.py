"""REFERENCE STATE BANK — explicitly NOT a native Gated DeltaNet backend.

    REFERENCE_NOT_NATIVE_GDN

This is a pure-Python contract-compatible state machine whose only purpose is
to give the conformance harness something executable to qualify against while
Spark develops the real physical GDN bank.  A Python list of slots is NOT an
in-model recurrent state; nothing here certifies physical GDN read/write,
effective physical capacity, native semantic admission, or long-context
retrieval fidelity.

Implementation strategy (deliberately different from the oracle):
  * fixed-width slot arrays + index maps for entities, relations, control
    records, convolution placeholders and the rolling KV window;
  * monotonic logical token counter separate from the recurrent tick;
  * per-tick operation budget enforced before any mutation.
"""
from __future__ import annotations
import math
from dataclasses import dataclass

from . import contract_r0 as C  # package-relative import (see __init__)

REFERENCE_MARKER = "REFERENCE_NOT_NATIVE_GDN"

POLICIES = ("FIFO", "LRU", "HEBBIAN_ONLY", "UTILITY_HEBBIAN")
UTILITY_DECAY_RATE = 3.5e-4          # source-derived: G5.13A age() constant
UTILITY_INIT = 0.020                 # source-derived: G5.13A initial score
HEBBIAN_COACT_GAIN = 0.20            # source-derived: G5.13A
UTILITY_REWARD_GAIN = 0.34           # source-derived: G5.13A


@dataclass
class EntityRecord:
    address: int
    value: tuple
    born: int
    last_read: int
    pinned: bool


@dataclass
class RelationRecord:
    relation_id: str
    provenance: str
    participant: int
    role: str
    ordinal: int
    flags: tuple
    value4: tuple
    frame: str
    utility: float
    utility_tick: int
    born: int
    last_read: int
    protected: bool


@dataclass
class ControlRecord:
    op: str
    tick: int
    status: str
    code: str


class ReferenceBank:
    """One session, one authoritative bounded bank (reference semantics)."""

    def __init__(self, spec, registry, transport, policy="UTILITY_HEBBIAN",
                 write_authority="fixture:write:r0"):
        bad = C.validate_spec(spec)
        if bad:
            raise ValueError("SPEC_INVALID:" + ",".join(bad))
        if policy not in POLICIES:
            raise ValueError("UNKNOWN_POLICY")
        self.spec, self.registry, self.transport = spec, registry, transport
        self.policy_name, self.write_authority = policy, write_authority
        self.reset()

    # ------------------------------------------------------------- lifecycle
    def begin_session(self, spec=None, registry=None, transport=None):
        if spec is not None and spec != self.spec:
            raise ValueError("SPEC_SWAP_FORBIDDEN")
        if registry is not None:
            self.registry = registry
        if transport is not None:
            self.transport = transport
        self.reset()
        return C.receipt(C.STATUS_OK)

    def reset(self):
        s = self.spec
        self.entity_slots = [None] * s.max_active_entities
        self.entity_index = {}
        self.relation_slots = [None] * s.max_active_relations
        self.relation_index = {}
        self.control_slots = [None] * s.max_control_records
        self.control_count = 0
        self.conv = [tuple(0.0 for _ in range(s.conv_state_dim))
                     for _ in range(s.conv_state_slots)]
        self.kv = [None] * (s.kv_window_tokens if s.kv_window_tokens else 0)
        self.kv_count = 0
        self.kv_write = 0
        self.committed = set()
        self.tick = 0
        self.logical_tokens = 0
        self.ops_this_tick = 0
        self.issued = {}
        self.committed = set()
        self.evict_count = 0
        self.reject_count = 0
        self.reinforcements = 0
        self.halted = False
        return C.receipt(C.STATUS_OK)

    def policy(self):
        return self.policy_name

    # ------------------------------------------------------------- budgets
    def _budget(self):
        if self.ops_this_tick >= self.spec.max_ops_per_tick:
            self.reject_count += 1
            return C.receipt(C.STATUS_REJECTED, "OP_BUDGET")
        self.ops_this_tick += 1
        return None

    def _control(self, op, status, code):
        s = self.spec
        if s.max_control_records:
            self.control_slots[self.control_count % s.max_control_records] = \
                ControlRecord(op, self.tick, status, code)
            self.control_count += 1

    def _advance(self, ticks):
        if ticks > self.spec.max_steps_per_call:
            return C.receipt(C.STATUS_REJECTED, "STEP_BUDGET")
        self.tick += ticks
        self.ops_this_tick = 0
        return None

    # ------------------------------------------------------------- admission
    def _validate(self, proposal):
        reg = self.registry.get(proposal.relation_id)
        if reg is None:
            return C.receipt(C.STATUS_REJECTED, "UNADMITTED_RELATION")
        if proposal.provenance != reg.provenance:
            return C.receipt(C.STATUS_REJECTED, "PROVENANCE_MISMATCH")
        match = None
        for (ent, role, ordinal) in reg.participants:
            if ent == proposal.participant:
                match = (role, ordinal)
                break
        if match is None:
            return C.receipt(C.STATUS_REJECTED, "UNKNOWN_PARTICIPANT")
        if match[0] != proposal.role:
            return C.receipt(C.STATUS_REJECTED, "ROLE_MISMATCH")
        if match[1] != proposal.ordinal:
            return C.receipt(C.STATUS_REJECTED, "ORDINAL_MISMATCH")
        if tuple(sorted(proposal.flags)) != tuple(reg.flags):
            return C.receipt(C.STATUS_REJECTED, "FLAGS_MISMATCH")
        if proposal.ticket != reg.ticket:
            return C.receipt(C.STATUS_REJECTED, "STALE_TICKET")
        if self.tick > reg.valid_until_tick:
            return C.receipt(C.STATUS_REJECTED, "STALE_TICKET")
        if proposal.participant not in self.entity_index:
            return C.receipt(C.STATUS_REJECTED, "DISCONNECTED_PARTICIPANT")
        if proposal.source_frame not in self.transport.list_frames():
            return C.receipt(C.STATUS_REJECTED, "UNBOUND_FRAME")
        try:
            C.vec4(proposal.value4, self.spec.finite_bound)
        except C.receipt_invalid as exc:
            return C.receipt(C.STATUS_REJECTED, exc.code, exc.detail)
        return None

    def propose_relation(self, proposal):
        bad = self._budget()
        if bad:
            return bad
        bad = self._validate(proposal)
        if bad:
            self.reject_count += 1
            self._control("propose_relation", bad.status, bad.code)
            return bad
        self.issued[id(proposal)] = (self.tick, proposal)
        self._control("propose_relation", C.STATUS_PENDING, "")
        return C.receipt(C.STATUS_PENDING)

    def apply_delta(self, proposal):
        bad = self._budget()
        if bad:
            return bad
        entry = self.issued.get(id(proposal))
        if entry is None:
            return C.receipt(C.STATUS_REJECTED, "STALE_PROPOSAL")
        issued_tick, _ = entry
        if id(proposal) in self.committed:
            self.reject_count += 1
            self._control("apply_delta", C.STATUS_REJECTED, "DUPLICATE_IDENTITY")
            return C.receipt(C.STATUS_REJECTED, "DUPLICATE_IDENTITY")
        bad = self._validate(proposal)
        if bad:
            self.reject_count += 1
            self._control("apply_delta", bad.status, bad.code)
            return bad
        try:
            canonical = C.vec4(self.transport.to_canonical(
                proposal.relation_id, proposal.source_frame, proposal.value4),
                self.spec.finite_bound)
        except (KeyError, PermissionError, ValueError):
            self.reject_count += 1
            self._control("apply_delta", C.STATUS_REJECTED, "UNBOUND_FRAME")
            return C.receipt(C.STATUS_REJECTED, "UNBOUND_FRAME")
        rid = proposal.relation_id
        if rid in self.relation_index:
            rec = self.relation_slots[self.relation_index[rid]]
            rec.value4 = canonical
            rec.frame = proposal.relation_id
            rec.last_read = self.tick
            self.committed.add(id(proposal))
            self._control("apply_delta", C.STATUS_OK, "")
            return C.receipt(C.STATUS_OK)
        try:
            slot = self._free_relation_slot()
        except C.receipt_invalid as exc:
            self.reject_count += 1
            self._control("apply_delta", C.STATUS_REJECTED, exc.code)
            return C.receipt(C.STATUS_REJECTED, exc.code)
        reg = self.registry.get(rid)
        self.relation_slots[slot] = RelationRecord(
            rid, reg.provenance, proposal.participant, proposal.role,
            proposal.ordinal, tuple(sorted(proposal.flags)), canonical,
            proposal.relation_id, UTILITY_INIT, self.tick, self.tick,
            self.tick, reg.protected)
        self.relation_index[rid] = slot
        self.committed.add(id(proposal))
        self._control("apply_delta", C.STATUS_OK, "")
        return C.receipt(C.STATUS_OK)

    def _free_relation_slot(self):
        for i, r in enumerate(self.relation_slots):
            if r is None:
                return i
        victim = self._relation_victim()
        if victim is None:
            raise C.receipt_invalid("CAPACITY_EXCEEDED", "no evictable relation")
        self._drop_relation(victim)
        return next(i for i, r in enumerate(self.relation_slots) if r is None)

    def _age(self, rec):
        return rec.utility * math.exp(-UTILITY_DECAY_RATE * (self.tick - rec.utility_tick))

    def _relation_victim(self):
        live = [r for r in self.relation_slots if r is not None]
        if not live:
            return None
        if self.policy_name == "UTILITY_HEBBIAN":
            free = [r for r in live if not r.protected]
            live = free if free else live
        if self.policy_name == "FIFO":
            victim = min(live, key=lambda r: (r.born, r.relation_id))
        elif self.policy_name == "LRU":
            victim = min(live, key=lambda r: (r.last_read, r.born, r.relation_id))
        else:
            victim = min(live, key=lambda r: (self._age(r), r.born, r.relation_id))
        return victim.relation_id

    def _drop_relation(self, rid):
        slot = self.relation_index.pop(rid)
        self.relation_slots[slot] = None
        self.evict_count += 1

    # ------------------------------------------------------------- exact space
    def _check_address(self, address):
        lo, hi = self.spec.exact_address_namespace
        if type(address) is not int or not lo <= address <= hi:
            return C.receipt(C.STATUS_REJECTED, "INVALID_ADDRESS")
        return None

    def read_exact(self, address):
        bad = self._budget()
        if bad:
            return bad
        bad = self._check_address(address)
        if bad:
            return bad
        slot = self.entity_index.get(address)
        if slot is None:
            return C.receipt(C.STATUS_NOT_FOUND)
        rec = self.entity_slots[slot]
        rec.last_read = self.tick
        self._control("read_exact", C.STATUS_OK, "")
        return C.receipt(C.STATUS_OK, "", rec.value)

    def upsert_exact(self, address, value, authority):
        bad = self._budget()
        if bad:
            return bad
        bad = self._check_address(address)
        if bad:
            return bad
        if authority != self.write_authority:
            self.reject_count += 1
            self._control("upsert_exact", C.STATUS_REJECTED, "UNAUTHORIZED_WRITE")
            return C.receipt(C.STATUS_REJECTED, "UNAUTHORIZED_WRITE")
        try:
            vec = C.vec4(value, self.spec.finite_bound)
        except C.receipt_invalid as exc:
            return C.receipt(C.STATUS_REJECTED, exc.code, exc.detail)
        if address in self.entity_index:
            rec = self.entity_slots[self.entity_index[address]]
            rec.value = vec
            rec.last_read = self.tick
            self._control("upsert_exact", C.STATUS_OK, "")
            return C.receipt(C.STATUS_OK)
        try:
            slot = self._free_entity_slot()
        except C.receipt_invalid as exc:
            self.reject_count += 1
            self._control("upsert_exact", C.STATUS_REJECTED, exc.code)
            return C.receipt(C.STATUS_REJECTED, exc.code)
        self.entity_slots[slot] = EntityRecord(address, vec, self.tick, self.tick, False)
        self.entity_index[address] = slot
        self._control("upsert_exact", C.STATUS_OK, "")
        return C.receipt(C.STATUS_OK)

    def _free_entity_slot(self):
        for i, e in enumerate(self.entity_slots):
            if e is None:
                return i
        referenced = {r.participant for r in self.relation_slots if r is not None}
        live = [e for e in self.entity_slots if e is not None and not e.pinned]
        if not live:
            raise C.receipt_invalid("CAPACITY_EXCEEDED", "no evictable entity")
        # Incidence closure: an entity that still anchors a live relation is
        # only evicted once nothing unreferenced remains.
        orphans = [e for e in live if e.address not in referenced]
        victim = min(orphans or live, key=lambda e: (e.last_read, e.born, e.address))
        self._drop_entity(victim.address)
        return next(i for i, e in enumerate(self.entity_slots) if e is None)

    def _drop_entity(self, address):
        # Incidence closure: erasing a participant erases relations bound to it.
        slot = self.entity_index.pop(address)
        self.entity_slots[slot] = None
        self.evict_count += 1
        for rid in [r for r, slot in self.relation_index.items()
                    if self.relation_slots[slot].participant == address]:
            self._drop_relation(rid)

    # ---------------------------------------------------------- associative
    def read_assoc(self, key):
        bad = self._budget()
        if bad:
            return bad
        hits = []
        for rid in sorted(self.relation_index):
            rec = self.relation_slots[self.relation_index[rid]]
            if rec.participant == key:
                hits.append((self._age(rec), rec.value4, rid))
        if not hits:
            return C.receipt(C.STATUS_NOT_FOUND)
        hits.sort(key=lambda h: (-h[0], h[2]))
        score, value, rid = hits[0]
        rec = self.relation_slots[self.relation_index[rid]]
        rec.last_read = self.tick
        self._control("read_assoc", C.STATUS_OK, "")
        return C.receipt(C.STATUS_OK, "", (value, score, len(hits)))

    # ---------------------------------------------------------- aggregation
    def transport_and_aggregate(self, proposals):
        bad = self._budget()
        if bad:
            return bad
        transported = []
        for p in proposals:
            entry = self.issued.get(id(p))
            if entry is None:
                return C.receipt(C.STATUS_REJECTED, "STALE_PROPOSAL")
            bad = self._validate(p)
            if bad:
                self.reject_count += 1
                self._control("transport_and_aggregate", bad.status, bad.code)
                return bad
            try:
                transported.append(C.vec4(self.transport.to_canonical(
                    p.relation_id, p.source_frame, p.value4), self.spec.finite_bound))
            except (KeyError, PermissionError, ValueError):
                self.reject_count += 1
                self._control("transport_and_aggregate", C.STATUS_REJECTED, "UNBOUND_FRAME")
                return C.receipt(C.STATUS_REJECTED, "UNBOUND_FRAME")
        if not transported:
            return C.receipt(C.STATUS_REJECTED, "INVALID_VALUE", "EMPTY_ROUND")
        norms = [math.sqrt(math.fsum(v * v for v in vec)) for vec in transported]
        exponents = [self.spec.salient_temperature * n for n in norms]
        top = math.fsum(math.exp(t) for t in exponents)
        weights = [math.exp(t) / top for t in exponents]
        salient = tuple(math.fsum(weights[i] * transported[i][j]
                                  for i in range(len(transported)))
                        for j in range(4))
        collective = tuple(math.fsum(v[j] for v in transported) / len(transported)
                           for j in range(4))
        self._control("transport_and_aggregate", C.STATUS_OK, "")
        return C.receipt(C.STATUS_OK, "", (salient, collective))

    # ---------------------------------------------------------- utility ops
    def reinforce(self, relation_id, coactivation, feedback):
        bad = self._budget()
        if bad:
            return bad
        for x in (coactivation, feedback):
            if type(x) not in (int, float) or not math.isfinite(x) or not -1 <= x <= 1:
                return C.receipt(C.STATUS_REJECTED, "INVALID_VALUE", "bounded feedback")
        slot = self.relation_index.get(relation_id)
        if slot is None:
            return C.receipt(C.STATUS_NOT_FOUND)
        rec = self.relation_slots[slot]
        self.reinforcements += 1
        rec.last_read = self.tick
        if self.policy_name in ("HEBBIAN_ONLY", "UTILITY_HEBBIAN"):
            value = self._age(rec) + HEBBIAN_COACT_GAIN * max(coactivation, 0.0)
            if self.policy_name == "UTILITY_HEBBIAN":
                value += UTILITY_REWARD_GAIN * feedback
                if feedback < 0:
                    rec.protected = False
            rec.utility = max(0.0, min(1.0, value))
            rec.utility_tick = self.tick
        self._control("reinforce", C.STATUS_OK, "")
        return C.receipt(C.STATUS_OK)

    def decay(self, ticks):
        bad = self._budget()
        if bad:
            return bad
        bad = self._advance(ticks)
        if bad:
            return bad
        self._control("decay", C.STATUS_OK, "")
        return C.receipt(C.STATUS_OK)

    def evict(self, relation_id):
        bad = self._budget()
        if bad:
            return bad
        slot = self.relation_index.get(relation_id)
        if slot is None:
            return C.receipt(C.STATUS_NOT_FOUND)
        rec = self.relation_slots[slot]
        if rec.protected and self.policy_name == "UTILITY_HEBBIAN":
            return C.receipt(C.STATUS_REJECTED, "PROTECTED_RECORD")
        self._drop_relation(relation_id)
        self._control("evict", C.STATUS_OK, "")
        return C.receipt(C.STATUS_OK)

    def step(self, action_budget=1):
        bad = self._budget()
        if bad:
            return bad
        if type(action_budget) is not int or action_budget < 0:
            return C.receipt(C.STATUS_REJECTED, "STEP_BUDGET")
        bad = self._advance(min(action_budget, self.spec.max_steps_per_call))
        if bad:
            return bad
        self._control("step", C.STATUS_OK, "")
        return C.receipt(C.STATUS_OK)

    # ---------------------------------------------------------- rolling KV
    def observe(self, tokens):
        bad = self._budget()
        if bad:
            return bad
        if not isinstance(tokens, tuple) or not tokens:
            return C.receipt(C.STATUS_REJECTED, "INVALID_VALUE", "EMPTY_TOKENS")
        for t in tokens:
            if type(t) is not int or t < 0:
                return C.receipt(C.STATUS_REJECTED, "INVALID_VALUE", "TOKEN_TYPE")
        if len(tokens) > self.spec.max_steps_per_call:
            return C.receipt(C.STATUS_REJECTED, "STEP_BUDGET")
        for tok in tokens:
            self.logical_tokens += 1
            self.tick += 1
            self.ops_this_tick = 0
            w = self.spec.kv_window_tokens
            if w:
                self.kv[self.kv_write % w] = (tok, self.logical_tokens)
                self.kv_write += 1
                self.kv_count = min(w, self.kv_count + 1)
            for i in range(self.spec.conv_state_slots):
                prev = self.conv[i]
                phase = ((tok + 7 * i) % 13) / 13.0
                self.conv[i] = prev[1:] + (phase,)
        self._control("observe", C.STATUS_OK, "")
        return C.receipt(C.STATUS_OK, "", self.logical_tokens)

    def window_tokens(self):
        w = self.spec.kv_window_tokens
        if not w:
            return ()
        start = (self.kv_write - self.kv_count) % w
        return tuple(self.kv[(start + i) % w] for i in range(self.kv_count))

    # ---------------------------------------------------------- terminal
    def halt(self):
        self.halted = True
        self._control("halt", C.STATUS_OK, "")
        return C.receipt(C.STATUS_OK, "", "REFERENCE_TERMINAL")

    # ---------------------------------------------------------- introspection
    def _control_chronological(self):
        cap = self.spec.max_control_records
        if not cap:
            return []
        if self.control_count <= cap:
            live = [c for c in self.control_slots if c is not None]
        else:
            split = self.control_count % cap
            live = [c for c in self.control_slots[split:] if c is not None] + \
                   [c for c in self.control_slots[:split] if c is not None]
        return [(c.op, c.tick, c.status, c.code) for c in live]

    def occupancy(self):
        return {
            "entities": len(self.entity_index),
            "entity_capacity": self.spec.max_active_entities,
            "relations": len(self.relation_index),
            "relation_capacity": self.spec.max_active_relations,
            "control_records": min(self.control_count, self.spec.max_control_records),
            "window": self.kv_count,
            "window_capacity": self.spec.kv_window_tokens,
            "tick": self.tick,
            "logical_tokens": self.logical_tokens,
            "evictions": self.evict_count,
            "rejections": self.reject_count,
            "reinforcements": self.reinforcements,
        }

    def snapshot(self):
        return {
            "entities": sorted(
                (e.address, tuple(round(x, 12) for x in e.value), e.born,
                 e.last_read, e.pinned)
                for e in self.entity_slots if e is not None),
            "relations": sorted(
                (r.relation_id, r.provenance, r.participant, r.role, r.ordinal,
                 r.flags, tuple(round(x, 12) for x in r.value4), r.frame,
                 round(r.utility, 12), r.utility_tick, r.born, r.last_read,
                 r.protected)
                for r in self.relation_slots if r is not None),
            "control": self._control_chronological(),
            "conv": [tuple(round(x, 12) for x in slot) for slot in self.conv],
            "window": self.window_tokens(),
            "tick": self.tick,
            "logical_tokens": self.logical_tokens,
            "evictions": self.evict_count,
            "rejections": self.reject_count,
            "reinforcements": self.reinforcements,
        }
