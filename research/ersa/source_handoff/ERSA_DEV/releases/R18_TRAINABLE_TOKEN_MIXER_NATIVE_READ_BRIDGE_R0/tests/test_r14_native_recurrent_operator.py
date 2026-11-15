"""ERSA R14: independent differential / negative tests against frozen R13."""
from __future__ import annotations
import ctypes as ct
import dataclasses
import numpy as np
import pytest
from integration import ersa_controller_r13 as r13
from integration.ersa_recur_r14 import CRecur,Receipt,compile_recur,run_on_owner
from integration.ersa_selector_r12 import (
    open_state, score_native_source, CSelector, cleanup_compilations, train as train_selector,
)
from native.ersa_native import NativeOps,build_native
from integration.ssc_native_r11 import compile_r11

@pytest.fixture(scope='module')
def environment():
    paths=[build_native(),score_native_source(),compile_r11(),r13.compile_halt(),compile_recur()]
    try:
        ops=NativeOps(paths[0]);sel=CSelector(paths[1]);halt=r13.CHalt(paths[3]);engine=CRecur(paths[4])
        wh,loss0,loss1=r13.train_halt(seed=7213,episodes=160,epochs=30)
        ws,sloss0,sloss1=train_selector(seed=10001,episodes=120,epochs=28)
        assert loss1<loss0 and sloss1<sloss0
        yield ops,sel,halt,engine,paths[2],ws,wh
    finally:cleanup_compilations(*paths)

@pytest.mark.parametrize('family',['positive','hard_pointer','no_terminal','missing_goal'])
@pytest.mark.parametrize('seed',[11,12,13,14,15,16,17,18])
@pytest.mark.parametrize('policy',['learned','keygoal_learned_halt','strong','fixed2'])
def test_native_recurrence_exact_path_and_status_parity(environment,family,seed,policy):
    ops,sel,halt,engine,lib,ws,wh=environment
    case=r13.make_chain(20000+seed*719,family=family)
    with open_state(lib,case.episode) as owner:
        selector_w=ws if policy=='learned' else r13.KEY_GOAL
        host_policy='learned' if policy=='keygoal_learned_halt' else policy
        host=r13.execute(owner,ops,sel,halt,case,selector_w,wh,policy=host_policy)
        native=run_on_owner(owner,engine,case,selector_w,wh,policy=policy)
        if host['status']=='OK' and len(set(host['path']))!=len(host['path']):
            # New R14 fail-closed check: an R13 fixed2 duplicate-stop is unsafe.
            assert native['status']=='HOLD_CYCLE' and native['path']==host['path']
            return
        assert native['status']==host['status'],(family,seed,policy,host,native)
        assert native['path']==host['path'],(family,seed,policy,host,native)
        if host['status']=='OK':
            assert native['steps']==host['steps']
            assert native['answer']==host['answer']
        assert len(native['path'])<=4

@pytest.mark.parametrize('seed',[181,182,183,184,185,186,187,188])
def test_stale_state_before_native_execution_rejected(environment,seed):
    _,_,_,engine,lib,ws,wh=environment
    case=r13.make_chain(seed,hops=2)
    with open_state(lib,case.episode) as owner:
        def inject(o,stage):
            if stage=='before':
                assert o.upsert_exact(1000,(.4,.1,.2,.3),'fixture:write:r0').status=='OK'
        out=run_on_owner(owner,engine,case,ws,wh,inject=inject)
        assert out['status']=='HOLD_STALE_GENERATION' and out['path']==()

@pytest.mark.parametrize('seed',[191,192,193,194,195,196,197,198])
def test_stale_state_after_native_execution_rejected(environment,seed):
    _,_,_,engine,lib,ws,wh=environment
    case=r13.make_chain(seed,hops=3)
    with open_state(lib,case.episode) as owner:
        def inject(o,stage):
            if stage=='after':
                assert o.upsert_exact(1000,(.4,.1,.2,.3),'fixture:write:r0').status=='OK'
        out=run_on_owner(owner,engine,case,ws,wh,inject=inject)
        assert out['status']=='HOLD_STALE_GENERATION' and out['path']==()

@pytest.mark.parametrize('invalid',[257,-1,.5,float('nan'),'1',None])
def test_untrusted_authority_mask_never_wraps_into_admission(environment,invalid):
    _,_,_,engine,lib,ws,wh=environment
    case=r13.make_chain(8210)
    allow=[1]*len(case.episode.keys);allow[0]=invalid
    with open_state(lib,case.episode) as owner:
        out=run_on_owner(owner,engine,case,ws,wh,allowed=allow)
        assert out['status']=='HOLD_AUTHORITY'

@pytest.mark.parametrize('seed',[700,701,702,703,704,705,706,707])
def test_unauthorized_terminal_is_never_in_native_path(environment,seed):
    _,_,_,engine,lib,ws,wh=environment
    case=r13.make_chain(seed,family='positive',hops=2)
    allow=np.ones(len(case.episode.keys),dtype=np.uint8)
    allow[case.path[-1]]=0
    with open_state(lib,case.episode) as owner:
        out=run_on_owner(owner,engine,case,ws,wh,policy='strong',allowed=allow)
        assert case.path[-1] not in out['path']

@pytest.mark.parametrize('seed',[811,812,813,814,815,816,817,818])
def test_all_denied_must_abstain_without_path(environment,seed):
    _,_,_,engine,lib,ws,wh=environment
    case=r13.make_chain(seed)
    with open_state(lib,case.episode) as owner:
        out=run_on_owner(owner,engine,case,ws,wh,allowed=np.zeros(24,dtype=np.uint8))
        assert out=={'status':'HOLD_NO_CANDIDATE','path':()}

@pytest.mark.parametrize('seed',[921,922,923,924])
def test_no_terminal_never_claims_answer(environment,seed):
    _,_,_,engine,lib,ws,wh=environment
    case=r13.make_chain(seed,family='no_terminal')
    with open_state(lib,case.episode) as owner:
        for policy in ('learned','keygoal_learned_halt','strong'):
            assert run_on_owner(owner,engine,case,ws,wh,policy=policy)['status']!='OK'

@pytest.mark.parametrize('column',[0,1,2,3])
def test_malformed_exact_state_is_rejected_atomically(environment,column):
    _,_,_,engine,_,ws,wh=environment
    case=r13.make_chain(991)
    values=case.episode.values.copy();values[1,column]=float('inf')
    with pytest.raises(ValueError):engine.run(case.episode,values,ws,wh)

@pytest.mark.parametrize('column',[0,1,2,3,4,5,6,7])
def test_nonfinite_selector_weight_rejected(environment,column):
    _,_,_,engine,_,ws,wh=environment
    case=r13.make_chain(992)
    malicious=ws.copy();malicious[column]=float('nan')
    with pytest.raises(ValueError):engine.run(case.episode,case.episode.values,malicious,wh)

@pytest.mark.parametrize('column',[0,1,2,3,4,5])
def test_nonfinite_halt_weight_rejected(environment,column):
    _,_,_,engine,_,ws,wh=environment
    case=r13.make_chain(993)
    malicious=wh.copy();malicious[column]=float('inf')
    with pytest.raises(ValueError):engine.run(case.episode,case.episode.values,ws,malicious)

@pytest.mark.parametrize('step',[0,1,2,3,4,5])
def test_receipt_untouched_on_native_validation_failure(environment,step):
    _,_,_,engine,_,ws,wh=environment
    case=r13.make_chain(994)
    vals=engine._prepare(case.episode,case.episode.values,ws,wh)
    n,*arr=vals
    keys,goals,values,meta,issued,mask,sw,hw,qkey,qgoal,qdir=arr
    mask=mask.copy();mask[step]=2
    D=ct.POINTER(ct.c_double);I=ct.POINTER(ct.c_int64);U=ct.POINTER(ct.c_uint8)
    receipt=Receipt(99,93,(ct.c_int32*4)(88,87,86,85),(ct.c_double*4)(11.,12.,13.,14.))
    rc=engine.f(n,keys.ctypes.data_as(D),goals.ctypes.data_as(D),values.ctypes.data_as(D),
       meta.ctypes.data_as(D),issued.ctypes.data_as(I),mask.ctypes.data_as(U),
       sw.ctypes.data_as(D),hw.ctypes.data_as(D),qkey.ctypes.data_as(D),
       qgoal.ctypes.data_as(D),qdir.ctypes.data_as(D),4,0,.015,ct.byref(receipt))
    assert rc!=0
    assert receipt.status==99 and receipt.count==93
    assert tuple(receipt.path)==(88,87,86,85)
    assert tuple(receipt.answer)==(11.,12.,13.,14.)

def test_negative_and_oversized_hop_request_rejected(environment):
    _,_,_,engine,_,ws,wh=environment
    case=r13.make_chain(995)
    for hop in (-1,0,5,100):
        with pytest.raises(ValueError):engine.run(case.episode,case.episode.values,ws,wh,max_hops=hop)

def test_read_only_native_operator_never_mutates_snapshot(environment):
    _,_,_,engine,_,ws,wh=environment
    case=r13.make_chain(996)
    ep=case.episode
    copies=[x.copy() for x in (ep.keys,ep.goals,ep.values,ep.meta,ep.issued,ep.query_key,ep.query_goal,ep.query_dir)]
    engine.run(ep,ep.values,ws,wh)
    for cur,old in zip((ep.keys,ep.goals,ep.values,ep.meta,ep.issued,ep.query_key,ep.query_goal,ep.query_dir),copies):
        assert np.array_equal(cur,old)

def test_native_8192_capacity_is_bounded_and_has_valid_receipt(environment):
    _,_,_,engine,_,ws,wh=environment
    from integration.ersa_selector_r12 import make_episode
    ep=make_episode(997,n=8192)
    out=engine.run(ep,ep.values,r13.KEY_GOAL,wh,policy='strong')
    assert len(out['path'])<=4 and out['status'] in ('OK','HOLD_UNCERTAIN_STOP','HOLD_BUDGET','HOLD_AMBIGUOUS')

def test_unscoped_authority_is_not_promoted_to_production(environment):
    _,_,_,engine,lib,ws,wh=environment
    case=r13.make_chain(998)
    with open_state(lib,case.episode) as owner:
        got=run_on_owner(owner,engine,case,ws,wh)
        assert got['status'] in ('OK','HOLD_UNCERTAIN_STOP','HOLD_BUDGET','HOLD_AMBIGUOUS','HOLD_CYCLE')
        assert 'trust' not in got and 'rights' not in got
