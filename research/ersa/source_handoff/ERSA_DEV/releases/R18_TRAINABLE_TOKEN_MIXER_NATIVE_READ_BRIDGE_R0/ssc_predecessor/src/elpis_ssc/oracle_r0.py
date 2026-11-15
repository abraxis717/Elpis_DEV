"""INDEPENDENT TEST ORACLE — expected results derived from the contract text.

SEPARABILITY REQUIREMENT.  The oracle is the second implementation of the
contract, written from the contract rules with a deliberately different
strategy from the backend under test:

  backend  : fixed slot arrays + index maps + ring buffers
  oracle   : plain dicts + ordered lists + explicit re-scan victim search

Neither imports the other.  A backend therefore cannot qualify by producing
its own expected answers: every observable is compared against this model,
and every injected defect must be detected by the comparison, not by the
backend's own self-report.

Constants marked SOURCE-DERIVED come from the G5.13A parent authority
(elastic_awr_r0.py, SHA-256 33f75070...); they are reproduced here so that
parent scenarios can be replayed, not copied as an implementation.
"""
from __future__ import annotations
import math
from dataclasses import dataclass

from . import contract_r0 as C

UTILITY_DECAY_RATE = 3.5e-4
UTILITY_INIT = 0.020
HEBBIAN_COACT_GAIN = 0.20
UTILITY_REWARD_GAIN = 0.34


class OracleModel:
    """Expected state after replaying a contract-level operation trace."""

    def __init__(self, spec, registry, transport, policy="UTILITY_HEBBIAN",
                 write_authority="fixture:write:r0"):
        bad = C.validate_spec(spec)
        if bad:
            raise ValueError("SPEC_INVALID:" + ",".join(bad))
        if policy not in ("FIFO", "LRU", "HEBBIAN_ONLY", "UTILITY_HEBBIAN"):
            raise ValueError("UNKNOWN_POLICY")
        self.spec, self.registry, self.transport = spec, registry, transport
        self.policy_name, self.write_authority = policy, write_authority
        self.entities = {}
        self.relations = {}
        self.control = []
        self.conv = [[0.0] * spec.conv_state_dim for _ in range(spec.conv_state_slots)]
        self.kv = []
        self.tick = 0
        self.logical = 0
        self.ops_this_tick = 0
        self.issued = set()
        self.committed = set()
        self.evictions = 0
        self.rejections = 0
        self.reinforcements = 0

    # ---------------------------------------------------------------- budgets
    def _budget(self):
        if self.ops_this_tick >= self.spec.max_ops_per_tick:
            self.rejections += 1
            return C.receipt(C.STATUS_REJECTED, "OP_BUDGET")
        self.ops_this_tick += 1
        return None

    def _control(self, op, status, code):
        if self.spec.max_control_records:
            self.control.append((op, self.tick, status, code))
            keep = self.spec.max_control_records
            if len(self.control) > keep:
                del self.control[:len(self.control) - keep]

    def _age(self, rec):
        return rec["utility"] * math.exp(-UTILITY_DECAY_RATE * (self.tick - rec["utility_tick"]))

    # ------------------------------------------------------------- admission
    def _validate(self, p):
        reg = self.registry.get(p.relation_id)
        if reg is None:
            return C.receipt(C.STATUS_REJECTED, "UNADMITTED_RELATION")
        if p.provenance != reg.provenance:
            return C.receipt(C.STATUS_REJECTED, "PROVENANCE_MISMATCH")
        same = [t for t in reg.participants if t[0] == p.participant]
        if not same:
            return C.receipt(C.STATUS_REJECTED, "UNKNOWN_PARTICIPANT")
        role, ordinal = same[0][1], same[0][2]
        if role != p.role:
            return C.receipt(C.STATUS_REJECTED, "ROLE_MISMATCH")
        if ordinal != p.ordinal:
            return C.receipt(C.STATUS_REJECTED, "ORDINAL_MISMATCH")
        if tuple(sorted(p.flags)) != tuple(reg.flags):
            return C.receipt(C.STATUS_REJECTED, "FLAGS_MISMATCH")
        if p.ticket != reg.ticket:
            return C.receipt(C.STATUS_REJECTED, "STALE_TICKET")
        if self.tick > reg.valid_until_tick:
            return C.receipt(C.STATUS_REJECTED, "STALE_TICKET")
        if p.participant not in self.entities:
            return C.receipt(C.STATUS_REJECTED, "DISCONNECTED_PARTICIPANT")
        if p.source_frame not in self.transport.list_frames():
            return C.receipt(C.STATUS_REJECTED, "UNBOUND_FRAME")
        try:
            C.vec4(p.value4, self.spec.finite_bound)
        except C.receipt_invalid as exc:
            return C.receipt(C.STATUS_REJECTED, exc.code, exc.detail)
        return None

    def _victim(self):
        live = list(self.relations)
        if not live:
            return None
        if self.policy_name == "UTILITY_HEBBIAN":
            free = [k for k in live if not self.relations[k]["protected"]]
            live = free if free else live
        if self.policy_name == "FIFO":
            return sorted(live, key=lambda k: (self.relations[k]["born"], k))[0]
        if self.policy_name == "LRU":
            return sorted(live, key=lambda k: (self.relations[k]["last_read"],
                                               self.relations[k]["born"], k))[0]
        return sorted(live, key=lambda k: (self._age(self.relations[k]),
                                           self.relations[k]["born"], k))[0]

    # ------------------------------------------------------------ operations
    def begin_session(self, *a):
        self.evictions = self.rejections = self.reinforcements = 0
        self.entities, self.relations, self.control = {}, {}, []
        self.conv = [[0.0] * self.spec.conv_state_dim for _ in range(self.spec.conv_state_slots)]
        self.kv, self.tick, self.logical, self.ops_this_tick = [], 0, 0, 0
        self.issued, self.evictions, self.rejections, self.reinforcements = set(), 0, 0, 0
        self.committed = set()
        return C.receipt(C.STATUS_OK)

    def observe(self, tokens):
        bad = self._budget()
        if bad:
            return bad
        if not isinstance(tokens, tuple) or not tokens:
            return C.receipt(C.STATUS_REJECTED, "INVALID_VALUE")
        if any(type(t) is not int or t < 0 for t in tokens):
            return C.receipt(C.STATUS_REJECTED, "INVALID_VALUE")
        if len(tokens) > self.spec.max_steps_per_call:
            return C.receipt(C.STATUS_REJECTED, "STEP_BUDGET")
        for tok in tokens:
            self.logical += 1
            self.tick += 1
            self.ops_this_tick = 0
            if self.spec.kv_window_tokens:
                self.kv.append((tok, self.logical))
                if len(self.kv) > self.spec.kv_window_tokens:
                    self.kv = self.kv[-self.spec.kv_window_tokens:]
            for i in range(self.spec.conv_state_slots):
                phase = ((tok + 7 * i) % 13) / 13.0
                self.conv[i] = self.conv[i][1:] + [phase]
        self._control("observe", C.STATUS_OK, "")
        return C.receipt(C.STATUS_OK, "", self.logical)

    def read_exact(self, address):
        bad = self._budget() or self._address(address)
        if bad:
            return bad
        if address not in self.entities:
            return C.receipt(C.STATUS_NOT_FOUND)
        rec = self.entities[address]
        rec[2] = self.tick
        self._control("read_exact", C.STATUS_OK, "")
        return C.receipt(C.STATUS_OK, "", tuple(rec[0]))

    def upsert_exact(self, address, value, authority):
        bad = self._budget() or self._address(address)
        if bad:
            return bad
        if authority != self.write_authority:
            self.rejections += 1
            self._control("upsert_exact", C.STATUS_REJECTED, "UNAUTHORIZED_WRITE")
            return C.receipt(C.STATUS_REJECTED, "UNAUTHORIZED_WRITE")
        try:
            vec = C.vec4(value, self.spec.finite_bound)
        except C.receipt_invalid as exc:
            return C.receipt(C.STATUS_REJECTED, exc.code, exc.detail)
        if address in self.entities:
            self.entities[address][0] = list(vec)
            self.entities[address][2] = self.tick
            self._control("upsert_exact", C.STATUS_OK, "")
            return C.receipt(C.STATUS_OK)
        if len(self.entities) >= self.spec.max_active_entities:
            unpinned = [a for a, r in self.entities.items() if not r[3]]
            if not unpinned:
                return C.receipt(C.STATUS_REJECTED, "CAPACITY_EXCEEDED")
            referenced = {r["participant"] for r in self.relations.values()}
            orphans = [a for a in unpinned if a not in referenced]
            victim = sorted(orphans or unpinned, key=lambda a: (self.entities[a][2],
                                                                self.entities[a][1],
                                                                a))[0]
            del self.entities[victim]
            self.evictions += 1
            for rid in [k for k, rec in self.relations.items()
                        if rec["participant"] == victim]:
                del self.relations[rid]
                self.evictions += 1
        self.entities[address] = [list(vec), self.tick, self.tick, False]
        self._control("upsert_exact", C.STATUS_OK, "")
        return C.receipt(C.STATUS_OK)

    def read_assoc(self, key):
        bad = self._budget()
        if bad:
            return bad
        hits = sorted(((-self._age(r), rid) for rid, r in self.relations.items()
                       if r["participant"] == key))
        if not hits:
            return C.receipt(C.STATUS_NOT_FOUND)
        neg_age, rid = hits[0]
        self.relations[rid]["last_read"] = self.tick
        self._control("read_assoc", C.STATUS_OK, "")
        return C.receipt(C.STATUS_OK, "", (tuple(self.relations[rid]["value4"]),
                                           -neg_age, len(hits)))

    def propose_relation(self, p):
        bad = self._budget()
        if bad:
            return bad
        bad = self._validate(p)
        if bad:
            self.rejections += 1
            self._control("propose_relation", bad.status, bad.code)
            return bad
        self.issued.add(id(p))
        self._control("propose_relation", C.STATUS_PENDING, "")
        return C.receipt(C.STATUS_PENDING)

    def _transport(self, p):
        try:
            return C.vec4(self.transport.to_canonical(
                p.relation_id, p.source_frame, p.value4), self.spec.finite_bound)
        except (KeyError, PermissionError, ValueError):
            return None

    def apply_delta(self, p):
        bad = self._budget()
        if bad:
            return bad
        if id(p) not in self.issued:
            return C.receipt(C.STATUS_REJECTED, "STALE_PROPOSAL")
        if id(p) in self.committed:
            self.rejections += 1
            self._control("apply_delta", C.STATUS_REJECTED, "DUPLICATE_IDENTITY")
            return C.receipt(C.STATUS_REJECTED, "DUPLICATE_IDENTITY")
        bad = self._validate(p)
        if bad:
            self.rejections += 1
            self._control("apply_delta", bad.status, bad.code)
            return bad
        canonical = self._transport(p)
        if canonical is None:
            self.rejections += 1
            self._control("apply_delta", C.STATUS_REJECTED, "UNBOUND_FRAME")
            return C.receipt(C.STATUS_REJECTED, "UNBOUND_FRAME")
        rid = p.relation_id
        if rid in self.relations:
            self.relations[rid]["value4"] = canonical
            self.relations[rid]["last_read"] = self.tick
            self.committed.add(id(p))
            self._control("apply_delta", C.STATUS_OK, "")
            return C.receipt(C.STATUS_OK)
        if len(self.relations) >= self.spec.max_active_relations:
            victim = self._victim()
            if victim is None:
                return C.receipt(C.STATUS_REJECTED, "CAPACITY_EXCEEDED")
            del self.relations[victim]
            self.evictions += 1
        reg = self.registry.get(rid)
        self.relations[rid] = {
            "relation_id": rid, "provenance": reg.provenance,
            "participant": p.participant, "role": p.role, "ordinal": p.ordinal,
            "flags": tuple(sorted(p.flags)), "value4": canonical,
            "frame": rid, "utility": UTILITY_INIT, "utility_tick": self.tick,
            "born": self.tick, "last_read": self.tick, "protected": reg.protected,
        }
        self.committed.add(id(p))
        self._control("apply_delta", C.STATUS_OK, "")
        return C.receipt(C.STATUS_OK)

    def transport_and_aggregate(self, proposals):
        bad = self._budget()
        if bad:
            return bad
        transported = []
        for p in proposals:
            if id(p) not in self.issued:
                return C.receipt(C.STATUS_REJECTED, "STALE_PROPOSAL")
            bad = self._validate(p)
            if bad:
                self.rejections += 1
                self._control("transport_and_aggregate", bad.status, bad.code)
                return bad
            vec = self._transport(p)
            if vec is None:
                self.rejections += 1
                self._control("transport_and_aggregate", C.STATUS_REJECTED, "UNBOUND_FRAME")
                return C.receipt(C.STATUS_REJECTED, "UNBOUND_FRAME")
            transported.append(vec)
        if not transported:
            return C.receipt(C.STATUS_REJECTED, "INVALID_VALUE")
        norms = [math.sqrt(sum(v * v for v in vec)) for vec in transported]
        ex = [self.spec.salient_temperature * n for n in norms]
        total = sum(math.exp(t) for t in ex)
        w = [math.exp(t) / total for t in ex]
        salient = tuple(sum(w[i] * transported[i][j] for i in range(len(transported)))
                        for j in range(4))
        collective = tuple(sum(v[j] for v in transported) / len(transported)
                           for j in range(4))
        self._control("transport_and_aggregate", C.STATUS_OK, "")
        return C.receipt(C.STATUS_OK, "", (salient, collective))

    def reinforce(self, rid, coactivation, feedback):
        bad = self._budget()
        if bad:
            return bad
        for x in (coactivation, feedback):
            if type(x) not in (int, float) or not math.isfinite(x) or not -1 <= x <= 1:
                return C.receipt(C.STATUS_REJECTED, "INVALID_VALUE")
        if rid not in self.relations:
            return C.receipt(C.STATUS_NOT_FOUND)
        rec = self.relations[rid]
        self.reinforcements += 1
        rec["last_read"] = self.tick
        if self.policy_name in ("HEBBIAN_ONLY", "UTILITY_HEBBIAN"):
            value = self._age(rec) + HEBBIAN_COACT_GAIN * max(coactivation, 0.0)
            if self.policy_name == "UTILITY_HEBBIAN":
                value += UTILITY_REWARD_GAIN * feedback
                if feedback < 0:
                    rec["protected"] = False
            rec["utility"] = max(0.0, min(1.0, value))
            rec["utility_tick"] = self.tick
        self._control("reinforce", C.STATUS_OK, "")
        return C.receipt(C.STATUS_OK)

    def decay(self, ticks):
        bad = self._budget()
        if bad:
            return bad
        if ticks > self.spec.max_steps_per_call:
            return C.receipt(C.STATUS_REJECTED, "STEP_BUDGET")
        self.tick += ticks
        self.ops_this_tick = 0
        self._control("decay", C.STATUS_OK, "")
        return C.receipt(C.STATUS_OK)

    def evict(self, rid):
        bad = self._budget()
        if bad:
            return bad
        if rid not in self.relations:
            return C.receipt(C.STATUS_NOT_FOUND)
        if self.relations[rid]["protected"] and self.policy_name == "UTILITY_HEBBIAN":
            return C.receipt(C.STATUS_REJECTED, "PROTECTED_RECORD")
        del self.relations[rid]
        self.evictions += 1
        self._control("evict", C.STATUS_OK, "")
        return C.receipt(C.STATUS_OK)

    def step(self, budget=1):
        bad = self._budget()
        if bad:
            return bad
        if type(budget) is not int or budget < 0:
            return C.receipt(C.STATUS_REJECTED, "STEP_BUDGET")
        if budget > self.spec.max_steps_per_call:
            return C.receipt(C.STATUS_REJECTED, "STEP_BUDGET")
        self.tick += budget
        self.ops_this_tick = 0
        self._control("step", C.STATUS_OK, "")
        return C.receipt(C.STATUS_OK)

    def reset(self):
        self.entities, self.relations, self.control = {}, {}, []
        self.conv = [[0.0] * self.spec.conv_state_dim for _ in range(self.spec.conv_state_slots)]
        self.kv, self.tick, self.logical, self.ops_this_tick = [], 0, 0, 0
        self.issued = set()
        self.committed = set()
        self.evictions = self.rejections = self.reinforcements = 0
        return C.receipt(C.STATUS_OK)

    def halt(self):
        self._control("halt", C.STATUS_OK, "")
        return C.receipt(C.STATUS_OK, "", "REFERENCE_TERMINAL")

    # ------------------------------------------------------------ introspection
    def _address(self, address):
        lo, hi = self.spec.exact_address_namespace
        if type(address) is not int or not lo <= address <= hi:
            return C.receipt(C.STATUS_REJECTED, "INVALID_ADDRESS")
        return None

    def occupancy(self):
        return {"entities": len(self.entities),
                "entity_capacity": self.spec.max_active_entities,
                "relations": len(self.relations),
                "relation_capacity": self.spec.max_active_relations,
                "control_records": len(self.control),
                "window": len(self.kv),
                "window_capacity": self.spec.kv_window_tokens,
                "tick": self.tick, "logical_tokens": self.logical,
                "evictions": self.evictions, "rejections": self.rejections,
                "reinforcements": self.reinforcements}

    def snapshot(self):
        return {
            "entities": sorted((a, tuple(round(x, 12) for x in r[0]), r[1], r[2], r[3])
                                for a, r in self.entities.items()),
            "relations": sorted(
                (r["relation_id"], r["provenance"], r["participant"], r["role"],
                 r["ordinal"], r["flags"], tuple(round(x, 12) for x in r["value4"]),
                 r["frame"], round(r["utility"], 12), r["utility_tick"], r["born"],
                 r["last_read"], r["protected"])
                for r in self.relations.values()),
            "control": list(self.control),
            "conv": [tuple(round(x, 12) for x in slot) for slot in self.conv],
            "window": tuple(self.kv),
            "tick": self.tick, "logical_tokens": self.logical,
            "evictions": self.evictions, "rejections": self.rejections,
            "reinforcements": self.reinforcements,
        }


# ------------------------------------------------------------- trace machinery
def replay(model, trace):
    """Execute a contract-level trace against any contract-compatible model."""
    receipts = []
    for op in trace:
        name, args = op[0], op[1:]
        fn = getattr(model, name)
        receipts.append((name, fn(*args)))
    return receipts


def replay_one(model, op):
    name, args = op[0], op[1:]
    return getattr(model, name)(*args)


def compare_snapshots(a, b, tol=1e-9):
    """Field-wise comparison with float tolerance.  Returns divergences."""
    out = []
    for key in sorted(set(a) | set(b)):
        va, vb = a.get(key), b.get(key)
        if not _close(va, vb, tol):
            out.append((key, va, vb))
    return out


def _close(x, y, tol):
    if type(x) is float or type(y) is float:
        if isinstance(x, float) and isinstance(y, float):
            return abs(x - y) <= tol
        return x == y
    if isinstance(x, (tuple, list)) and isinstance(y, (tuple, list)):
        if len(x) != len(y):
            return False
        return all(_close(i, j, tol) for i, j in zip(x, y))
    if isinstance(x, dict) and isinstance(y, dict):
        return x == y
    return x == y


def compare_receipts(ra, rb):
    """True when two receipts disagree semantically.

    A missing receipt is a contract violation by the adapter, not a crash.
    """
    if ra is None or rb is None:
        return True
    if ra.status != rb.status or ra.code != rb.code:
        return True
    return not _close(ra.detail, rb.detail, 1e-9)
