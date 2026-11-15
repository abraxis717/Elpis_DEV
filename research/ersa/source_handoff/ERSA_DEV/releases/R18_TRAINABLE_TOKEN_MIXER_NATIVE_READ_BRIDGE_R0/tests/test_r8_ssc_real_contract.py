"""Independent R8 tests using the genuine seven pinned SSC modules.

A deliberately small *test-only identity transport* replaces the SSC fixture
module's missing transport_reference.py dependency. This is not the official
SSC fixture suite and must never be described as full predecessor replay.
"""
from __future__ import annotations
from dataclasses import replace
import hashlib
import math
import pytest

from elpis_ssc import contract_r0 as SC, oracle_r0 as SO, reference_backend as SB, harness_r0 as SH
from elpis_ersa import adapter_ssc as legacy
from integration.ssc_r8 import MonitoredSscSession,SscReadProjection,ProjectionHold
from native.ersa_read_r7 import NativeReadPath
from native.ersa_native import NativeOps, build_native


class IdentityTestTransport:
    """Small, explicit test port, NOT a substitute predecessor source file."""
    def __init__(self,frames=('p101','p102','p103','p104')):
        self.frames=tuple(frames)
    def bind_frame(self,relation_id,frame_id,operator):
        if frame_id not in self.frames:raise KeyError(frame_id)
    def list_frames(self):return self.frames
    def to_canonical(self,rid,frame,vec4):
        if frame not in self.frames:raise KeyError(frame)
        return tuple(vec4)
    def to_view(self,rid,frame,vec4):
        if frame not in self.frames:raise KeyError(frame)
        return tuple(vec4)


def context(names=('R1','R2','R3'),cap=4,valid_until=1000,protected=()):
    reg=SC.AdmissionRegistry([SC.AdmissionRecord(
        rid,'prov:'+rid,((101+i,'evidence',0),),('synthetic',),
        'ticket:'+rid,0,valid_until,rid in protected)
        for i,rid in enumerate(names)])
    spec=SC.SessionSpec(kv_window_tokens=8,max_active_relations=cap,
        max_active_entities=8,max_control_records=64,max_ops_per_tick=64,
        max_steps_per_call=256)
    trans=IdentityTestTransport()
    bank=SB.ReferenceBank(spec,reg,trans)
    oracle=SO.OracleModel(spec,reg,trans)
    return spec,reg,trans,bank,oracle


def proposal(rid='R1',part=101,role='evidence',vec=(0.2,-0.1,0.0,0.5),
             provenance=None,ticket=None,frame=None):
    return SC.Proposal(rid,part,role,0,('synthetic',),frame or f'p{part}',
        vec,ticket or 'ticket:'+rid,0,provenance or 'prov:'+rid)


def populated(n=2, names=('R1','R2','R3'), cap=4, protected=(),monitor=False):
    spec,reg,transport,bank,oracle=context(names,cap,protected=protected)
    m=MonitoredSscSession(bank) if monitor else bank
    assert m.begin_session().status=='OK'
    for i in range(n):
        rid=names[i]; part=101+i
        assert m.upsert_exact(part,(0.1*i,0,0,0),'fixture:write:r0').status=='OK'
        p=proposal(rid,part,vec=(.25+i*.1,-.2,0,.4))
        assert m.propose_relation(p).status=='PENDING'
        assert m.apply_delta(p).status=='OK'
    return m,bank,oracle


@pytest.fixture(scope='session')
def native():
    import shutil
    lib=build_native();ops=NativeReadPath(NativeOps(lib))
    yield ops
    shutil.rmtree(lib.parent)


def test_ssc_actual_api_not_ersa_schema():
    _,reg,transport,bank,_=context()
    assert not SC.check_adapter(bank)
    assert not SC.check_transport(transport)
    assert SC.CONTRACT_SCHEMA=='elpis.gdn.session.r0'
    assert len(SC.STATE_BANK_METHODS)==14


@pytest.mark.parametrize('bad', [1,None,'',object()])
def test_projection_rejects_invalid_owner(bad):
    with pytest.raises(ProjectionHold):SscReadProjection(bad)


@pytest.mark.parametrize('rid', ['R1','R2','R3'])
def test_mapped_native_reads_exact(rid,native):
    owner,backend,_=populated(n=3,monitor=True)
    p=SscReadProjection(owner)
    out=p.read(rid,native)
    assert out.status=='OK' and out.code==''
    assert out.value4==dict((r[0],r[6]) for r in backend.snapshot()['relations'])[rid]
    assert max(abs(a-b) for a,b in zip(out.value4,out.salient)) < 1e-9
    assert len(set(p._mapping.values())) == len(p._mapping)


@pytest.mark.parametrize('bad', [None,1,1.0,True,b'R1',('R1',)])
def test_native_read_identity_type_rejected(bad,native):
    owner,_,_=populated(monitor=True)
    assert SscReadProjection(owner).read(bad,native).status=='HOLD'


@pytest.mark.parametrize('missing', ['R3','R404','R0'])
def test_nonresident_or_unadmitted_fails_closed(missing,native):
    owner,_,_=populated(monitor=True)
    out=SscReadProjection(owner).read(missing,native)
    assert out.status=='HOLD' and out.value4==() and out.salient==()


@pytest.mark.parametrize('method,args',[
    ('reset',()),('step',(1,)),('decay',(1,)),('observe',((17,),)),
    ('read_exact',(101,)),('evict',('R1',)),('reinforce',('R1',.3,.4)),
])
def test_versioned_ssc_operations_invalidate_projection(method,args,native):
    owner,_,_=populated(monitor=True)
    projection=SscReadProjection(owner)
    getattr(owner,method)(*args)
    assert projection.read('R1',native).code=='SSC_VIEW_STALE'


def test_identical_reset_aba_blocked_by_owner_generation(native):
    owner,_,_=populated(n=0,monitor=True)
    p=SscReadProjection(owner)
    digest=p._snapshot_digest
    assert owner.reset().status=='OK'
    assert SC.digest_state(owner.snapshot())==digest
    assert p.read('R1',native).code=='SSC_VIEW_STALE'


def test_outside_alias_snapshot_change_detected(native):
    owner,backend,_=populated(monitor=True)
    p=SscReadProjection(owner)
    backend.upsert_exact(102,(0,0,0,0),'fixture:write:r0')
    assert p.read('R1',native).code=='SSC_VIEW_STALE'


@pytest.mark.parametrize('rid', ['R1','R2'])
def test_no_ssc_mutation_during_native_projection(rid,native):
    owner,bank,_=populated(monitor=True)
    snap=SC.digest_state(bank.snapshot())
    p=SscReadProjection(owner)
    assert p.read(rid,native).status=='OK'
    assert SC.digest_state(bank.snapshot())==snap


def test_original_ssc_string_names_never_coerce_to_numeric(native):
    owner,bank,_=populated(n=2,names=('01','1','R3'),monitor=True)
    p=SscReadProjection(owner)
    x=p.read('01',native);y=p.read('1',native)
    assert x.status==y.status=='OK' and x.numeric_id!=y.numeric_id
    assert x.ssc_id=='01' and y.ssc_id=='1'


def test_registry_over_limit_hold_without_allocation():
    owner,_,_=populated(n=0,monitor=True)
    with pytest.raises(ProjectionHold,match='SSC_REGISTRY_LIMIT'):
        SscReadProjection(owner,max_registry=2)


def test_capacity_too_large_hold():
    _,reg,trans,bank,_=context(cap=8193)
    owner=MonitoredSscSession(bank)
    with pytest.raises(ProjectionHold,match='SSC_CAPACITY_UNSUPPORTED'):
        SscReadProjection(owner)


def test_ssc_runtime_unprotection_not_silently_mapped():
    owner,backend,_=populated(n=1,protected=('R1',),monitor=True)
    assert backend.snapshot()['relations'][0][-1] is True
    assert owner.reinforce('R1',.1,-1.).status=='OK'
    assert backend.snapshot()['relations'][0][-1] is False
    with pytest.raises(ProjectionHold,match='SSC_AUTHORITY_OR_PROTECTION_UNMAPPABLE'):
        SscReadProjection(owner)


def test_frozen_legacy_adapter_refuses_ssc_write():
    adapter=legacy.SscAdapter()
    assert not adapter.predecessor_compatible
    assert adapter.upsert_exact(101,(0,0,0,0),'fixture:write:r0').status=='REJECTED'
    assert adapter.propose_relation(proposal()).status=='REJECTED'


def test_ssc_entity_address_not_available_in_ersa_projection():
    owner,_,_=populated(monitor=True)
    p=SscReadProjection(owner)
    with pytest.raises(ProjectionHold,match='SSC_ENTITY_NAMESPACE_UNSUPPORTED'):
        p.read_exact_native(101)


def test_refuses_unknown_native_reader():
    owner,_,_=populated(monitor=True)
    p=SscReadProjection(owner)
    assert p.read('R1',None).code=='NATIVE_READER_REQUIRED'


@pytest.mark.parametrize('seed', range(8))
def test_real_ssc_reference_vs_independent_oracle(seed):
    spec,reg,transport,bank,oracle=context(cap=3)
    ops=[SC.op('begin_session'),SC.op('observe',(seed+1,)),
         SC.op('upsert_exact',101,(seed/16,0.,0.,0.),'fixture:write:r0'),
         SC.op('upsert_exact',102,(.2,0.,0.,0.),'fixture:write:r0')]
    p=proposal('R1',101,vec=(.1+seed/64,-.3,0.,.2))
    q=proposal('R2',102,vec=(-.2,.1,0.,.2))
    ops += [SC.op('propose_relation',p),SC.op('apply_delta',p),
            SC.op('propose_relation',q),SC.op('apply_delta',q),
            SC.op('read_assoc',101),SC.op('transport_and_aggregate',(p,)),
            SC.op('reinforce','R1',.25,-.1),SC.op('decay',2),
            SC.op('step',1),SC.op('read_exact',101),SC.op('evict','R2'),
            SC.op('reset'),SC.op('halt')]
    result=SH.evaluate(bank,oracle,ops,checkpoints=(2,6,12,16),tol=1e-9)
    assert result['divergences']==[],str(result['divergences'][:2])
    assert result['first_divergence_index'] is None


@pytest.mark.parametrize('seed', range(8))
def test_real_ssc_invalid_operations_parity(seed):
    spec,reg,transport,bank,oracle=context()
    valid=proposal('R1',101)
    ops=[SC.op('begin_session'),
         SC.op('upsert_exact',101,(0,0,0,0),'WRONG_AUTHORITY'),
         SC.op('upsert_exact',999999999,(0,0,0,0),'fixture:write:r0'),
         SC.op('upsert_exact',101,(float('inf'),0,0,0),'fixture:write:r0'),
         SC.op('upsert_exact',101,(0,0,0,0),'fixture:write:r0'),
         SC.op('propose_relation',replace(valid,ticket='WRONG')),
         SC.op('propose_relation',replace(valid,role='wrong')),
         SC.op('propose_relation',replace(valid,provenance='wrong')),
         SC.op('apply_delta',valid),
         SC.op('propose_relation',valid),
         SC.op('apply_delta',valid),
         SC.op('apply_delta',valid),
         SC.op('step',1),SC.op('decay',2),
         SC.op('read_exact',-seed-1)]
    result=SH.evaluate(bank,oracle,ops,checkpoints=(1,8,len(ops)-1))
    assert result['divergences']==[],str(result['divergences'][:2])


def test_projection_verified_against_ssc_oracle_after_replay(native):
    spec,reg,tr,bank,oracle=context()
    mon=MonitoredSscSession(bank)
    p1=proposal('R1',101,( 'evidence' ))
    # Standalone SSC operation trace driven through a monitored owner.
    trace=[SC.op('begin_session'),SC.op('observe',(1,2)),
        SC.op('upsert_exact',101,(0,0,0,0),'fixture:write:r0'),
        SC.op('propose_relation',p1),SC.op('apply_delta',p1)]
    result=SH.evaluate(mon,oracle,trace,checkpoints=(len(trace)-1,))
    assert not result['divergences']
    p=SscReadProjection(mon)
    got=p.read('R1',native)
    assert got.status=='OK'
    assert got.value4==oracle.snapshot()['relations'][0][6]


def test_frozen_ssc_predecessor_step_budget_divergence_detected():
    """Unmodified SSC reference incorrectly clips over-budget step.

    KEEP this negative control: a genuine frozen predecessor defect, not a
    candidate repair. It blocks FULL SSC parity.
    """
    spec,reg,tr,bank,oracle=context()
    trace=[SC.op('begin_session'),SC.op('step',spec.max_steps_per_call+1)]
    result=SH.evaluate(bank,oracle,trace,checkpoints=(len(trace)-1,))
    assert result['first_divergence_index']==1
    assert any(x[2]=='receipt' and x[1]=='step' for x in result['divergences'])


def test_frozen_ssc_negative_decay_acceptance_documented():
    """Frozen SSC oracle and reference both accept negative time advances.

    Agreement is not soundness. A new backend must reject this upstream.
    """
    spec,reg,tr,bank,oracle=context()
    before=bank.snapshot()['tick']
    a=bank.decay(-2);b=oracle.decay(-2)
    assert a.status==b.status=='OK'
    assert bank.snapshot()['tick']==before-2
    assert oracle.snapshot()['tick']==before-2


@pytest.fixture(scope='session')
def ssc_native():
    from native.ssc_compat_r8 import build_ssc_reduce,SscNativeReduce
    import shutil
    lib=build_ssc_reduce()
    obj=SscNativeReduce(lib)
    yield obj
    shutil.rmtree(lib.parent)


@pytest.mark.parametrize('seed',range(20))
def test_ssc_native_reducer_against_real_ssc_geometry(seed,ssc_native):
    import random
    from elpis_ssc.geometry_r0 import aggregate_signed
    rng=random.Random(seed)
    n=1+seed*3
    vectors=[tuple(rng.uniform(-1.,1.) for _ in range(4)) for _ in range(n)]
    a,b=ssc_native.reduce(vectors,temperature=8.0)
    ea,eb=aggregate_signed(vectors,temperature=8.0)
    assert max(abs(x-y) for x,y in zip(a,ea))<1e-11
    assert max(abs(x-y) for x,y in zip(b,eb))<1e-11


def test_ssc_aggregate_is_not_ersa_signed_l1(ssc_native,native):
    from elpis_ssc.geometry_r0 import aggregate_signed
    v=((.2,.0,.0,.0),(1.,.1,.0,.0))
    ssc,collect=ssc_native.reduce(v)
    py_ssc,_=aggregate_signed(v)
    ersa,_=native.ops.aggregate(v,[1.,1.])
    assert max(abs(a-b) for a,b in zip(ssc,py_ssc))<1e-12
    assert max(abs(a-b) for a,b in zip(ssc,ersa))>0.1


@pytest.mark.parametrize('bad',[(),((1,2,3),),((1,2,3,4,5),),((float('nan'),0,0,0),),
                                 ((float('inf'),0,0,0),)])
def test_native_ssc_invalid_vector_fail_closed(bad,ssc_native):
    from native.ssc_compat_r8 import SscReduceError
    with pytest.raises(SscReduceError):ssc_native.reduce(bad)


@pytest.mark.parametrize('bad',[float('nan'),float('inf'),-1.,0.,True,'8'])
def test_native_ssc_invalid_temperature_fail_closed(bad,ssc_native):
    from native.ssc_compat_r8 import SscReduceError
    with pytest.raises(SscReduceError):ssc_native.reduce(((1.,0.,0.,0.),),bad)


def test_ssc_native_can_match_real_reference_transport_aggregate(ssc_native):
    spec,reg,tr,b,o=context()
    assert b.begin_session().status=='OK'
    p=proposal('R1',101,vec=(.3,.1,0.,0.))
    q=proposal('R2',102,vec=(-.1,.2,.5,0.))
    for part in (101,102):
        assert b.upsert_exact(part,(0.,0.,0.,0.),'fixture:write:r0').status=='OK'
    assert b.propose_relation(p).status=='PENDING'
    assert b.propose_relation(q).status=='PENDING'
    rec=b.transport_and_aggregate((p,q))
    assert rec.status=='OK'
    expected=rec.detail
    observed=ssc_native.reduce((p.value4,q.value4),spec.salient_temperature)
    for actual,want in zip(observed,expected):
        assert max(abs(a-b) for a,b in zip(actual,want))<1e-12


def test_mid_read_ssc_mutation_rejected(native,monkeypatch):
    owner,_,_=populated(monitor=True)
    view=SscReadProjection(owner)
    original=NativeReadPath.read
    def mutated(self,*a,**kw):
        result=original(self,*a,**kw)
        assert owner.step(1).status=='OK'
        return result
    monkeypatch.setattr(NativeReadPath,'read',mutated)
    result=view.read('R1',native)
    assert result.status=='HOLD' and result.code=='SSC_VIEW_STALE'
    assert result.salient==()
