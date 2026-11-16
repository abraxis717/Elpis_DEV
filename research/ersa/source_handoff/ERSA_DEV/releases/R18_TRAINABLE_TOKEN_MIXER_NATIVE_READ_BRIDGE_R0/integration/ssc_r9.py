"""ERSA R9: guarded SSC 14-op reference forwarding + native exact-state shadow.

The genuine SSC ReferenceBank is authoritative, *not* the C shadow.
No untrusted source can receive write permission from a native lookup.
The monitor prevents accepted over-budget steps and negative decays at a
versioned R9 boundary without changing frozen SSC R0 files.
"""
from __future__ import annotations
import threading

from elpis_ssc import contract_r0 as SC
from elpis_ssc.reference_backend import ReferenceBank
from integration.ssc_r8 import MonitoredSscSession, SscReadProjection
from native.ersa_state_r9 import NativeState, StateHold

class GuardHold(RuntimeError):
    def __init__(self,code):
        self.code=code
        super().__init__(code)

class GuardedSscReference:
    """One owner for the SSC reference bank; native entity state is a checked mirror.

    Native mirror publication is atomic but not an SSC/native distributed
    transaction. Post-commit shadow failure puts the entire guard in HOLD.
    A foreign backend alias is explicitly outside its coherency guarantee.
    """
    def __init__(self,bank,library):
        if type(bank) is not ReferenceBank or SC.check_adapter(bank):
            raise GuardHold('REAL_SSC_REFERENCE_REQUIRED')
        spec=bank.spec
        if (SC.validate_spec(spec) or spec.max_active_entities>8192 or
            spec.max_active_entities<1 or spec.max_active_relations>8192):
            raise GuardHold('SSC_SPEC_NOT_BOUNDED')
        self._lock=threading.RLock()
        self._owner=MonitoredSscSession(bank)
        self._registry=bank.registry
        self._transport=bank.transport
        self._spec=spec
        lo,hi=spec.exact_address_namespace
        if not -(1<<63)<=lo<=hi<1<<63:
            raise GuardHold('SSC_ADDRESS_RANGE_NOT_NATIVE')
        self._shadow=NativeState(library,spec.max_active_entities,lo,hi,spec.finite_bound)
        self._held=False
        self._version=-1
        self._digest=''
        try:
            self._publish()
        except BaseException:
            self._shadow.close()
            raise

    @property
    def generation(self):
        with self._lock:return self._owner.generation

    def close(self):
        with self._lock:
            if self._shadow:self._shadow.close()
            self._held=True

    def _source_snapshot(self):
        return self._owner.snapshot()

    def _check(self):
        if self._held:raise GuardHold('SSC_GUARD_HELD')
        if (self._owner.spec is not self._spec or
            self._owner.registry is not self._registry or
            self._owner._bank.transport is not self._transport):
            self._held=True
            raise GuardHold('SSC_EXTERNAL_AUTHORITY_SWAP')
        if self._owner.generation!=self._version:
            self._held=True
            raise GuardHold('SSC_UNMONITORED_GENERATION')
        if SC.digest_state(self._source_snapshot())!=self._digest:
            self._held=True
            raise GuardHold('SSC_UNMONITORED_MUTATION')

    def _publish(self):
        snapshot=self._source_snapshot()
        if (type(snapshot) is not dict or
            type(snapshot.get('entities')) is not list or
            len(snapshot['entities'])>self._spec.max_active_entities):
            raise GuardHold('SSC_SNAPSHOT_UNMAPPABLE')
        new=NativeState.replace_from_rows(self._shadow,snapshot['entities'])
        try:
            if new.count != len(snapshot['entities']):
                raise GuardHold('NATIVE_COUNT_DIVERGENCE')
            for row in snapshot['entities']:
                if new.peek(row[0]) != tuple(row[1]):
                    raise GuardHold('NATIVE_VALUE_DIVERGENCE')
            digest=SC.digest_state(snapshot)
            gen=self._owner.generation
        except BaseException:
            new.close()
            raise
        previous=self._shadow
        self._shadow=new
        self._version=gen
        self._digest=digest
        previous.close()

    def _reject_time(self,op,args,kwargs):
        if op not in ('step','decay'):return None
        if len(args)>1:return 'STEP_BUDGET'
        valid_key='action_budget' if op=='step' else 'ticks'
        if kwargs and (set(kwargs)!={valid_key} or args):return 'STEP_BUDGET'
        value=kwargs.get(valid_key,1 if op=='step' else None) if kwargs else (
            args[0] if args else (1 if op=='step' else None))
        if type(value) is not int or not 0<=value<=self._spec.max_steps_per_call:
            return 'STEP_BUDGET'
        return None

    def _call(self,name,*args,**kwargs):
        with self._lock:
            self._check()
            if name=='begin_session' and (args or kwargs):
                # No registry/transport/spec rebinding through this guard.
                # SSC identity changes require a new guard instance and epoch.
                self._owner._version += 1
                self._publish()
                raise GuardHold('SSC_SESSION_REBIND_REQUIRES_NEW_OWNER')
            invalid=self._reject_time(name,args,kwargs)
            if invalid:
                # Even a rejected operation invalidates previously captured
                # projections. No SSC backend call and no clock rewind.
                self._owner._version += 1
                self._publish()
                return SC.receipt(SC.STATUS_REJECTED,invalid)
            try:
                receipt=getattr(self._owner,name)(*args,**kwargs)
                self._publish()
                return receipt
            except BaseException:
                # A backend exception can occur after a partial write. HOLD.
                self._held=True
                raise

    def __getattr__(self,name):
        if name not in SC.STATE_BANK_METHODS:raise AttributeError(name)
        return lambda *a,**kw:self._call(name,*a,**kw)

    def snapshot(self):
        with self._lock:
            self._check()
            return self._source_snapshot()

    def occupancy(self):
        with self._lock:
            self._check()
            return self._owner.occupancy()

    def policy(self):
        with self._lock:
            self._check()
            return self._owner.policy()

    def peek_exact_native(self,address):
        """Read-only mirror diagnostic, NOT an SSC `read_exact` or authority."""
        with self._lock:
            self._check()
            val=self._shadow.peek(address)
            snap=self._source_snapshot()
            expected={row[0]:tuple(row[1]) for row in snap['entities']}.get(address)
            if val!=expected:
                self._held=True
                raise GuardHold('NATIVE_PEEK_MISMATCH')
            return val

    def relation_projection(self):
        """R8 projection remains SSC-owned and read-only."""
        with self._lock:
            self._check()
            return SscReadProjection(self._owner)
