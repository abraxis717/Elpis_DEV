"""ADVERSARIAL DEFECT INJECTIONS — deliberately broken adapters.

These are NOT implementations under qualification.  They exist so the harness
can be red-teamed: every mutant must be REJECTED by the conformance
comparison, with attribution.  A test suite that accepts one of these is a
blocking failure and the harness must not report qualification.
"""
from __future__ import annotations
from . import reference_backend as RB
from . import contract_r0 as C
from . import geometry_r0 as G


class MutantIgnoreAuthority(RB.ReferenceBank):
    """Defect: skips the admission/authority check entirely."""
    def _validate(self, proposal):
        return None


class MutantStaleTicket(RB.ReferenceBank):
    """Defect: accepts an expired admission ticket."""
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
        if match is None:
            return C.receipt(C.STATUS_REJECTED, "UNKNOWN_PARTICIPANT")
        if match[0] != proposal.role:
            return C.receipt(C.STATUS_REJECTED, "ROLE_MISMATCH")
        if match[1] != proposal.ordinal:
            return C.receipt(C.STATUS_REJECTED, "ORDINAL_MISMATCH")
        if proposal.ticket != reg.ticket:
            return C.receipt(C.STATUS_REJECTED, "STALE_TICKET")
        return None  # expired ticket and disconnected participant accepted


class MutantRetainedRecord(RB.ReferenceBank):
    """Defect: one relation record survives RESET."""
    def reset(self):
        out = RB.ReferenceBank.reset(self)
        reg = self.registry
        rid = reg.ids()[0] if reg.ids() else "MUTANT:STALE"
        rec = RB.RelationRecord(rid, "mutant", 0, "evidence", 0, ("synthetic",),
                                (0.0, 0.0, 0.0, 0.0), rid, 0.02, 0, 0, 0, False)
        self.relation_slots[0] = rec
        self.relation_index[rid] = 0
        return out

    def begin_session(self, *a, **k):
        return RB.ReferenceBank.begin_session(self, *a, **k)


class MutantDirtyBoundary(RB.ReferenceBank):
    """Defect: convolution state and rolling KV inherit across sessions."""
    def reset(self):
        conv = getattr(self, "conv", None)
        kv = getattr(self, "kv", None)
        kvc = getattr(self, "kv_count", None)
        kvw = getattr(self, "kv_write", None)
        out = RB.ReferenceBank.reset(self)
        if conv is not None:
            self.conv, self.kv = conv, kv
            self.kv_count, self.kv_write = kvc, kvw
        return out


class MutantCapacityOverflow(RB.ReferenceBank):
    """Defect: grows past the declared relation capacity instead of evicting."""
    def _free_relation_slot(self):
        for i, r in enumerate(self.relation_slots):
            if r is None:
                return i
        self.relation_slots.append(None)
        return len(self.relation_slots) - 1


class MutantDeniedAggregation(RB.ReferenceBank):
    """Defect: an unadmitted proposal is allowed to influence aggregation."""
    def transport_and_aggregate(self, proposals):
        bad = self._budget()
        if bad:
            return bad
        transported = []
        for p in proposals:
            try:
                transported.append(C.vec4(self.transport.to_canonical(
                    p.relation_id, p.source_frame, p.value4), self.spec.finite_bound))
            except (KeyError, PermissionError, ValueError):
                transported.append(tuple(float(x) for x in p.value4))
        if not transported:
            return C.receipt(C.STATUS_REJECTED, "INVALID_VALUE", "EMPTY_ROUND")
        norms = [G._norm(v) for v in transported]
        ex = [self.spec.salient_temperature * n for n in norms]
        total = G.math.fsum(G.math.exp(t) for t in ex)
        w = [G.math.exp(t) / total for t in ex]
        salient = tuple(G.math.fsum(w[i] * transported[i][j]
                                    for i in range(len(transported))) for j in range(4))
        collective = tuple(G.math.fsum(v[j] for v in transported) / len(transported)
                           for j in range(4))
        self._control("transport_and_aggregate", C.STATUS_OK, "")
        return C.receipt(C.STATUS_OK, "", (salient, collective))


class MutantSquaredAggregation(RB.ReferenceBank):
    """Defect: signed aggregation replaced by componentwise squared mean."""
    def transport_and_aggregate(self, proposals):
        base = RB.ReferenceBank.transport_and_aggregate(self, proposals)
        if base.status != C.STATUS_OK:
            return base
        salient, collective = base.detail
        sq = tuple(sum(x * x for x in collective) / 4.0 for _ in range(4))
        return C.receipt(C.STATUS_OK, "", (salient, sq))


class MutantUnboundedWindow(RB.ReferenceBank):
    """Defect: rolling cache retains every token ever observed."""
    def __init__(self, *a, **k):
        RB.ReferenceBank.__init__(self, *a, **k)
        self._all_tokens = []

    def observe(self, tokens):
        out = RB.ReferenceBank.observe(self, tokens)
        self._all_tokens.extend((t, i) for i, t in enumerate(tokens, start=1))
        return out

    def window_tokens(self):
        return tuple(self._all_tokens)

    def occupancy(self):
        occ = RB.ReferenceBank.occupancy(self)
        occ["window"] = len(self._all_tokens)
        return occ

    def snapshot(self):
        snap = RB.ReferenceBank.snapshot(self)
        snap["window"] = tuple(self._all_tokens)
        return snap

    def reset(self):
        RB.ReferenceBank.reset(self)
        self._all_tokens = []


class MutantPreValidationMutation(RB.ReferenceBank):
    """Defect: mutates a relation before proposal validation completes."""
    def apply_delta(self, proposal):
        bad = self._budget()
        if bad:
            return bad
        rid = proposal.relation_id
        if rid in self.relation_index:
            rec = self.relation_slots[self.relation_index[rid]]
            rec.value4 = tuple(float(x) * 1.5 for x in rec.value4)
        return RB.ReferenceBank.apply_delta(self, proposal)


class MutantNoDuplicateRejection(RB.ReferenceBank):
    """Defect: the same committed proposal may be applied repeatedly."""
    def apply_delta(self, proposal):
        committed = getattr(self, "_committed_backup", None)
        self._committed_backup = set(self.committed)
        self.committed = set()
        out = RB.ReferenceBank.apply_delta(self, proposal)
        self.committed = self._committed_backup | ({id(proposal)}
                                                   if out.status == C.STATUS_OK else set())
        return out


MUTANTS = {
    "ignore_authority": MutantIgnoreAuthority,
    "stale_ticket_accepted": MutantStaleTicket,
    "retained_record_after_reset": MutantRetainedRecord,
    "dirty_session_boundary": MutantDirtyBoundary,
    "capacity_overflow": MutantCapacityOverflow,
    "denied_source_aggregation": MutantDeniedAggregation,
    "squared_signed_aggregation": MutantSquaredAggregation,
    "unbounded_mock_window": MutantUnboundedWindow,
    "prevalidation_mutation": MutantPreValidationMutation,
    "duplicate_update_accepted": MutantNoDuplicateRejection,
}
