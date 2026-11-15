"""ERSA R8: SSC R0 -> ERSA R7 *read-only* identity projection.

The SSC predecessor owns the 14-op state bank. This module does not implement,
forward, or authorize SSC mutations through ERSA. The native ERSA reader only
reads a bounded, immutable-at-capture projection of SSC canonical relation
values. SSC's exact entity namespace, rolling KV/conv state and proposal ticket
semantics are NEVER claimed to be implemented by ERSA.

The bridge requires a monitored session owner. Version and snapshot checks are
best-effort process-local coherency checks, NOT external source attestation.
"""
from __future__ import annotations
from dataclasses import dataclass
import hashlib
import json
import math
import threading

from elpis_ersa import bank_ersa as EB, contract_ersa as EC
from elpis_ssc import contract_r0 as SC
from native.ersa_read_r7 import NativeReadPath

SSC_METHODS = SC.STATE_BANK_METHODS

class ProjectionHold(Exception):
    """One typed fail-closed reason, never represented as an SSC receipt."""
    def __init__(self, code: str):
        super().__init__(code)
        self.code = code


def _digest(snapshot):
    # SSC canonical representation. A digest is a coherency check, not a MAC.
    return SC.digest_state(snapshot)


class MonitoredSscSession:
    """Explicit owner wrapper; all SSC operations increment a monotonic version.

    Do not expose/modify the inner bank through another alias during execution.
    An outside alias is detectable only when it changes the snapshot, and an
    identical reset ABA cannot be detected without this versioned owner.
    """
    def __init__(self, backend):
        missing = SC.check_adapter(backend)
        if missing:
            raise ProjectionHold('SSC_BACKEND_API_MISSING')
        self._bank = backend
        self._version = 0
        self._lock = threading.RLock()

    @property
    def generation(self):
        with self._lock:
            return self._version

    @property
    def registry(self):
        return self._bank.registry

    @property
    def spec(self):
        return self._bank.spec

    def snapshot(self):
        with self._lock:
            return self._bank.snapshot()

    def occupancy(self):
        with self._lock:
            return self._bank.occupancy()

    def policy(self):
        with self._lock:
            return self._bank.policy()

    def __getattr__(self, name):
        if name not in SSC_METHODS:
            raise AttributeError(name)
        fn = getattr(self._bank, name)
        def call(*args, **kwargs):
            with self._lock:
                # Conservatively invalidate captures even on rejected ops.
                # Never hide backend errors or normalize its receipts.
                try:
                    return fn(*args, **kwargs)
                finally:
                    self._version += 1
        return call


def _int64(n):
    return type(n) is int and -(1 << 63) < n < (1 << 63)


def _key(participant, role, ordinal, rid):
    if not all(_int64(x) for x in (participant, role, ordinal, rid)):
        raise ProjectionHold('IDENTITY_KEY_OUT_OF_RANGE')
    return (participant, role, ordinal, rid)


@dataclass(frozen=True)
class RelationProjection:
    ssc_id: str
    numeric_id: int
    source_value: tuple
    query_key: tuple
    source_flags: tuple
    source_protected: bool


@dataclass(frozen=True)
class ProjectionRead:
    status: str
    code: str
    ssc_id: str = ''
    numeric_id: int = -1
    value4: tuple = ()
    salient: tuple = ()
    generation: int = -1


class SscReadProjection:
    """One snapshot-bound, bounded ERSA relation view.

    The full immutable SSC AdmissionRegistry establishes a stable bijection:
    lexical identity ordering maps each SSC str ID to its own ERSA int ID.
    Projection uses R7 *real* admitted record storage and native kernels.
    This is not a state-bank replacement and it never credits semantic reads.
    """
    def __init__(self, owner: MonitoredSscSession, *, max_registry=8192):
        if type(owner) is not MonitoredSscSession:
            raise ProjectionHold('OWNER_WRAPPER_REQUIRED')
        if type(max_registry) is not int or not 1 <= max_registry <= 8192:
            raise ProjectionHold('REGISTRY_CAP_INVALID')
        self._owner = owner
        self._owner_version = owner.generation
        self._owner_registry = owner.registry
        self._ssc_snapshot = owner.snapshot()
        self._snapshot_digest = _digest(self._ssc_snapshot)
        self._session_epoch = self._owner_version
        self._frozen_registry_digest = None
        self._native = None
        self._records = {}
        self._ersa_bank = None
        self._max_registry = max_registry
        self._construct()
        self._check_current()

    def _check_current(self):
        if (self._owner.generation != self._owner_version or
                self._owner.registry is not self._owner_registry or
                _digest(self._owner.snapshot()) != self._snapshot_digest):
            raise ProjectionHold('SSC_VIEW_STALE')

    def _construct(self):
        spec = self._owner.spec
        if SC.validate_spec(spec):
            raise ProjectionHold('SSC_SPEC_INVALID')
        if spec.max_active_relations > 8192 or spec.max_active_relations < 1:
            raise ProjectionHold('SSC_CAPACITY_UNSUPPORTED')
        ids = self._owner_registry.ids()
        if len(ids) > self._max_registry:
            raise ProjectionHold('SSC_REGISTRY_LIMIT')
        if not all(type(rid) is str and rid for rid in ids):
            raise ProjectionHold('SSC_ID_MALFORMED')
        if len(set(ids)) != len(ids):
            raise ProjectionHold('SSC_ID_DUPLICATE')
        if ids != tuple(sorted(ids)):
            raise ProjectionHold('SSC_ID_ORDER')
        mapping = {rid: i + 1 for i, rid in enumerate(ids)}
        snap = self._ssc_snapshot
        if type(snap) is not dict or type(snap.get('relations')) is not list:
            raise ProjectionHold('SSC_SNAPSHOT_SHAPE')
        if len(snap['relations']) > spec.max_active_relations:
            raise ProjectionHold('SSC_SNAPSHOT_EXCEEDS_CAP')
        provenance = sorted({self._owner_registry.get(rid).provenance for rid in ids})
        roles = sorted({role for rid in ids for _,role,_ in self._owner_registry.get(rid).participants})
        tickets = sorted({self._owner_registry.get(rid).ticket for rid in ids})
        if not all(type(v) is str for v in provenance + roles + tickets):
            raise ProjectionHold('SSC_AUTHORITY_TYPE')
        pmap = {p:i+1 for i,p in enumerate(provenance)}
        rmap = {role:i+1 for i,role in enumerate(roles)}
        tmap = {t:i+1 for i,t in enumerate(tickets)}
        records = []
        seen = set()
        inserted = []
        for row in snap['relations']:
            if type(row) not in (list,tuple) or len(row)!=13:
                raise ProjectionHold('SSC_RELATION_SHAPE')
            (rid, prov, part, role, ordinal, flags, value, frame, utility,
             utility_tick, born, last_read, protected) = row
            if type(rid) is not str or rid not in mapping or rid in seen:
                raise ProjectionHold('SSC_LIVE_ID_MISSING_OR_DUPLICATE')
            seen.add(rid)
            reg=self._owner_registry.get(rid)
            if (prov != reg.provenance or
                    (part,role,ordinal) not in reg.participants or
                    tuple(flags)!=reg.flags or
                    type(protected) is not bool or
                    protected != reg.protected):
                # SSC permits runtime protection changes; ERSA immutable
                # admission does not represent that behavior losslessly.
                raise ProjectionHold('SSC_AUTHORITY_OR_PROTECTION_UNMAPPABLE')
            if type(part) is not int or type(ordinal) is not int:
                raise ProjectionHold('SSC_PARTICIPANT_TYPE')
            if type(snap.get('tick')) is not int or snap['tick']>reg.valid_until_tick:
                raise ProjectionHold('SSC_ADMISSION_EXPIRED')
            try:
                valid_vec = SC.vec4(value, spec.finite_bound)
            except (SC.receipt_invalid, TypeError, ValueError) as exc:
                raise ProjectionHold('SSC_VECTOR_INVALID') from exc
            if tuple(valid_vec) != tuple(value):
                raise ProjectionHold('SSC_VECTOR_INCOMPATIBLE')
            if type(utility) not in (int,float) or not math.isfinite(utility):
                raise ProjectionHold('SSC_UTILITY_NONFINITE')
            n=mapping[rid]
            f=('PROTECTED',) if protected else ()
            key=_key(part, rmap[role], ordinal, n)
            records.append(EC.AdmissionRecord(n,pmap[prov],rmap[role],ordinal,
                f,tmap[reg.ticket],reg.issued_tick,reg.valid_until_tick,protected))
            inserted.append((rid,n,key,tuple(float(x) for x in value),
                             pmap[prov],part,rmap[role],ordinal,f,
                             tmap[reg.ticket],float(utility),protected))
        cap=max(1,spec.max_active_relations)
        es=EC.SessionSpec(max_active_relations=cap,max_candidates=min(8,cap),
                max_control_records=max(64, len(records)),
                max_ops_per_tick=max(64,cap*2+16),
                relational_namespace=(0,max(4096,len(ids)+1)),
                finite_bound=float(spec.finite_bound))
        bank=EB.ErsBank()
        beg=bank.begin_session(es,records)
        if beg.status != 'OK':
            raise ProjectionHold('ERSA_PROJECTION_START:'+beg.code)
        for (rid,n,key,val,prov,part,role,ordn,flags,tkt,util,protected) in inserted:
            ff=EC.FLAG_PROTECTED if protected else 0
            out=bank.insert_relation(n,key,val,provenance=prov,participant=part,
                    role=role,ordinal=ordn,flags=ff,ticket=tkt,
                    utility=max(0.0,min(1.0,util)),credibility=1.0)
            if out.status!='OK':
                raise ProjectionHold('ERSA_PROJECTION_ADMISSION:'+out.code)
            self._records[rid]=RelationProjection(rid,n,val,key,tuple(self._owner_registry.get(rid).flags),protected)
        # Identity projection must reproduce each exact value, not just count.
        banksnap=bank.snapshot()['relations']
        if len(banksnap)!=len(self._records):
            raise ProjectionHold('ERSA_RELATION_COUNT_MISMATCH')
        for v in self._records.values():
            if v.numeric_id not in banksnap or banksnap[v.numeric_id]['value']!=v.source_value:
                raise ProjectionHold('ERSA_VALUE_MISMATCH')
        self._ersa_bank=bank
        self._mapping=mapping

    @property
    def active_ids(self):
        return tuple(sorted(self._records))

    @property
    def projection_count(self):
        return len(self._records)

    def read(self, ssc_id, native_path: NativeReadPath):
        try:
            self._check_current()
            if type(ssc_id) is not str:
                raise ProjectionHold('SSC_ID_TYPE')
            if ssc_id not in self._mapping:
                raise ProjectionHold('SSC_UNADMITTED_ID')
            rec=self._records.get(ssc_id)
            if rec is None:
                raise ProjectionHold('SSC_ID_NOT_RESIDENT')
            if type(native_path) is not NativeReadPath:
                raise ProjectionHold('NATIVE_READER_REQUIRED')
            out=native_path.read(self._ersa_bank,rec.query_key,
                          qdir=(1.,0.,0.,0.),k=1,exact_relation_id=rec.numeric_id)
            if out.status!='OK' or out.relation_ids!=(rec.numeric_id,):
                raise ProjectionHold('ERSA_NATIVE_IDENTITY_FAILURE')
            if any(abs(a-b)>1e-9 for a,b in zip(out.salient,rec.source_value)):
                raise ProjectionHold('ERSA_NATIVE_VALUE_DIVERGENCE')
            self._check_current()
            return ProjectionRead('OK','',ssc_id,rec.numeric_id,rec.source_value,
                                  out.salient,self._owner_version)
        except ProjectionHold as exc:
            return ProjectionRead('HOLD',exc.code)

    def read_exact_native(self,*args,**kwargs):
        """Intentionally absent: SSC exact entity-address reads are not mapped."""
        raise ProjectionHold('SSC_ENTITY_NAMESPACE_UNSUPPORTED')
