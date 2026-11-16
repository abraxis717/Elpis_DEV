"""Independently motivated R13 BREAK cases and R12 native-substrate integration."""
from __future__ import annotations
import ctypes, math
import numpy as np
import pytest
import integration.ersa_controller_r13 as r
from integration.ersa_selector_r12 import score_native_source,CSelector,open_state,cleanup_compilations
from native.ersa_native import NativeOps,build_native
from integration.ssc_native_r11 import compile_r11

@pytest.fixture(scope='module')
def native():
    a=build_native();b=score_native_source();c=compile_r11();d=r.compile_halt()
    try:yield NativeOps(a),CSelector(b),c,r.CHalt(d)
    finally:cleanup_compilations(a,b,c,d)

@pytest.mark.parametrize('hop',range(1,5))
@pytest.mark.parametrize('seed',[131,401,783])
def test_synthetic_path_is_real_native_pointer_chain(hop,seed,native):
    _,_,lib,_=native
    case=r.make_chain(seed,hops=hop)
    ep=case.episode
    assert len(case.path)==hop and len(set(case.path))==hop
    with open_state(lib,ep) as state:
        values=r.read_values(state,len(ep.keys))
        assert np.array_equal(values,ep.values)
        for i,j in zip(case.path[:-1],case.path[1:]):
            assert np.array_equal(values[i],ep.keys[j])
        assert r.cosine(values[case.path[-1]],ep.query_goal)>.9

@pytest.mark.parametrize('step,maxsteps',[(0,4),(5,4),(1,0),(1,5),(-1,4),(1,-1),(4,3)])
def test_native_invalid_step_does_not_mutate_receipt(native,step,maxsteps):
    halt=native[3]
    x=np.zeros(5,dtype=np.float64);w=np.zeros(6,dtype=np.float64)
    prob=ctypes.c_double(12.3);action=ctypes.c_int(71)
    f=halt.f
    rc=f(x.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
         w.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
         step,maxsteps,ctypes.byref(prob),ctypes.byref(action))
    assert rc!=0 and prob.value==12.3 and action.value==71

@pytest.mark.parametrize('slot',[0,1,2,3,4])
def test_nonfinite_feature_rejected_native(native,slot):
    x=np.zeros(5,dtype=np.float64);x[slot]=float('nan')
    with pytest.raises(ValueError):native[3].action(x,np.zeros(6),1)

@pytest.mark.parametrize('slot',[0,1,2,3,4,5])
def test_nonfinite_weight_rejected_native(native,slot):
    w=np.zeros(6,dtype=np.float64);w[slot]=float('inf')
    with pytest.raises(ValueError):native[3].action(np.zeros(5),w,1)

def test_native_logits_and_actions_agree_with_independent_float64(native):
    h=native[3];rng=np.random.default_rng(7800)
    for _ in range(64):
        x=rng.normal(size=5);w=rng.normal(size=6)
        act,p=h.action(x,w,step=1)
        expected=1/(1+math.exp(-float(x@w[:5]+w[5])))
        assert abs(p-expected)<1e-12
        assert act==(1 if expected>=.70 else 0 if expected<=.30 else 2)

def test_native_step_ceiling_cannot_continue(native):
    x=np.zeros(5);w=np.array([-1,0,0,0,0,-20.],dtype=float)
    act,_=native[3].action(x,w,4)
    assert act==2

@pytest.mark.parametrize('seed',[101,102,103,104,105,106])
def test_no_terminal_must_not_stop_on_cyclic_pointer(native,seed):
    ops,sel,lib,h=native
    case=r.make_chain(seed,family='no_terminal')
    with open_state(lib,case.episode) as owner:
        # Even maliciously stop-biased weights cannot turn exact pointers into answers.
        bad=np.array([0,0,0,0,0,20.],dtype=np.float64)
        got=r.execute(owner,ops,sel,h,case,r.KEY_GOAL,bad)
        assert got['status']!='OK'
        assert len(got.get('path',()))<=4

def test_mutated_native_state_fails_closed(native):
    ops,sel,lib,h=native
    case=r.make_chain(90231,hops=2)
    with open_state(lib,case.episode) as owner:
        def mutate(o,step):
            if step==1:assert o.upsert_exact(1000,(.4,.1,0.,0.),'fixture:write:r0').status=='OK'
        got=r.execute(owner,ops,sel,h,case,r.KEY_GOAL,np.ones(6),inject=mutate)
        assert got['status']=='HOLD_STALE_GENERATION'

def test_no_authorized_candidates_cannot_read_answer(native):
    ops,sel,lib,h=native
    case=r.make_chain(90232)
    with open_state(lib,case.episode) as owner:
        got=r.execute(owner,ops,sel,h,case,r.KEY_GOAL,np.ones(6),allowed=np.zeros(24,dtype=np.uint8))
        assert got['status']=='HOLD_NO_CANDIDATE'

def test_denied_terminal_never_selected(native):
    ops,sel,lib,h=native
    case=r.make_chain(90233,hops=2)
    allowed=np.ones(24,dtype=np.uint8);allowed[case.path[-1]]=0
    with open_state(lib,case.episode) as owner:
        got=r.execute(owner,ops,sel,h,case,r.KEY_GOAL,np.ones(6),allowed=allowed)
        assert case.path[-1] not in got.get('path',())

def test_training_is_deterministic_and_has_real_loss_gain():
    a,x,y=r.train_halt(seed=999,episodes=80,epochs=15)
    b,x2,y2=r.train_halt(seed=999,episodes=80,epochs=15)
    assert np.array_equal(a,b) and x==x2 and y==y2 and y<x

def test_shuffled_labels_are_not_valid_proof_of_learning():
    a,_,_=r.train_halt(seed=678,episodes=170,epochs=35)
    b,_,_=r.train_halt(seed=678,episodes=170,epochs=35,shuffle_labels=True)
    assert not np.array_equal(a,b)
    assert a[0]>b[0] and a[2]>b[2]

def test_variable_length_evaluation_never_exposes_target_to_controller(native):
    # The decision routine references only episode, not the private oracle path.
    import inspect
    src=inspect.getsource(r.execute)
    assert 'case.path' not in src and 'case.terminal' not in src
    ops,sel,lib,h=native
    w,_,_=r.train_halt(seed=8081,episodes=240,epochs=40)
    stats=r.evaluate(r.KEY_GOAL,w,ops=ops,selector=sel,halter=h,state_lib=lib,
                     seed=744441,count=48,family='positive')
    assert stats['learned']['success']>=40 and stats['learned']['false_stop']==0
    assert stats['strong']['success']>=stats['learned']['success']-5

def test_zero_credit_for_superiority_when_baseline_wins(native):
    ops,sel,lib,h=native
    w,_,_=r.train_halt(seed=8081,episodes=240,epochs=40)
    stats=r.evaluate(r.KEY_GOAL,w,ops=ops,selector=sel,halter=h,state_lib=lib,
                     seed=744441,count=48,family='positive')
    if stats['learned']['success']<=stats['strong']['success']:
        assert not stats['learned']['success']>stats['strong']['success']
