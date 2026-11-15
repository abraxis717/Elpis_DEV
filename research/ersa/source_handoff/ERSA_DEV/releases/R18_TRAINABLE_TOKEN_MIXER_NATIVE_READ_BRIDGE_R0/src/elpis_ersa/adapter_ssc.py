"""Fail-closed predecessor boundary (ERSA R3 repair candidate).

The SSC predecessor's exact entity/address namespace is NOT equivalent to
ERSA's admitted logical relation identities. The R2 adapter silently hashed
strings into relation IDs and executed writes on coincidentally matching IDs.
That is a demonstrated fail-open mapping defect; no faithful mapping contract
was included in the ERSA R0/R2 handoff. Refuse all unsupported operations.

This is intentionally *not* a conformant replacement for SSC.  Integration
requires an explicit, independently verified, versioned identity binding.
"""
from . import bank_ersa as Bk, contract_ersa as C


class SscAdapter:
    predecessor_compatible = False
    bridge_disposition = 'HOLD_UNMAPPED_PREDECESSOR_IDENTITY_SPACE'

    def __init__(self):
        self._bank = None

    def begin_session(self, spec=None, registry=None, transport=None):
        self._bank = Bk.ErsBank()
        return self._bank.begin_session(spec, registry, transport, None)

    def reset(self):
        if self._bank is None:
            return C.receipt(C.STATUS_REJECTED, 'OP_BUDGET', 'NO_SESSION')
        return self._bank.reset()

    def policy(self):
        p = self._bank.policy() if self._bank else {}
        return dict(p, predecessor_compatible=False,
                    bridge_disposition=self.bridge_disposition)

    def occupancy(self):
        return (self._bank.occupancy() if self._bank else
                {'live': 0, 'capacity': 0, 'tick': 0})

    def snapshot(self):
        return (self._bank.snapshot() if self._bank else
                {'relations': {}, 'control': {}, 'tick': 0})

    @staticmethod
    def _hold(op, code='UNADMITTED_RELATION'):
        return C.receipt(C.STATUS_REJECTED, code,
                         f'{op}:UNMAPPED_PREDECESSOR_IDENTITY_SPACE')

    # All old SSC entity/address operations must refuse until proven mapped.
    def read_exact(self, address):
        return self._hold('read_exact')

    def read_assoc(self, key):
        return self._hold('read_assoc')

    def observe(self, ids):
        return self._hold('observe')

    def upsert_exact(self, address, value, authority):
        return self._hold('upsert_exact')

    def propose_relation(self, proposal):
        return self._hold('propose_relation')

    def apply_delta(self, proposal):
        return self._hold('apply_delta', 'STAGE_UNKNOWN')

    def transport_and_aggregate(self, proposals):
        return self._hold('transport_and_aggregate')

    def reinforce(self, relation_id, coactivation, feedback):
        return self._hold('reinforce')

    def decay(self, ticks):
        return self._hold('decay')

    def evict(self, relation_id):
        return self._hold('evict')

    def step(self, action_budget=1):
        return self._hold('step', 'STEP_BUDGET')

    def halt(self):
        # Halting is safe even though no SSC mutation is permitted.
        if self._bank is None:
            return self._hold('halt', 'OP_BUDGET')
        return self._bank.halt()
