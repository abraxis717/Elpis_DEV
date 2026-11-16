"""R12 independent controls; no frozen R2 answers used for training."""
import ctypes, hashlib, math, random, shutil
import numpy as np
import pytest
from integration.ersa_selector_r12 import (
  make_episode,features_py,train,score_native_source,CSelector,open_state,
  read_values,native_features,two_hop,FROZEN_WEIGHTS,cleanup_compilations)
from native.ersa_native import NativeOps,build_native
from integration.ssc_native_r11 import compile_r11

@pytest.fixture(scope='module')
def native():
    a=build_native();b=score_native_source();c=compile_r11()
    try: yield NativeOps(a),CSelector(b),c
    finally:cleanup_compilations(a,b,c)

@pytest.mark.parametrize('seed',range(20))
def test_c_feature_reference_parity(native,seed):
    ops,sel,lib=native; ep=make_episode(seed+10000)
    got=native_features(ops,ep,ep.query_key,ep.query_goal,ep.values)
    expected=features_py(ep,ep.query_key,ep.query_goal)
    assert np.max(np.abs(got-expected))<2e-12
    w=np.array([.3,1.,.07,-.2,.5,.1,-.9,1.2])
    assert np.max(np.abs(sel.score(got,w)-got@w))<2e-12

@pytest.mark.parametrize('seed',range(12))
def test_unchanged_c_owned_state_parity(native,seed):
    ops,sel,lib=native;ep=make_episode(seed+9000)
    with open_state(lib,ep) as owner:
        values=read_values(owner,len(ep.values))
        assert np.array_equal(values,ep.values)
        a=native_features(ops,ep,ep.query_key,ep.query_goal,values)
        b=features_py(ep,ep.query_key,ep.query_goal)
        assert np.max(np.abs(a-b))<2e-12

@pytest.mark.parametrize('seed',range(8))
def test_two_hop_bounded_native_admission(native,seed):
    ops,sel,lib=native;ep=make_episode(seed+6000)
    with open_state(lib,ep) as owner:
        weights=np.array([2.,0.,0.,0.,0.,5.,0.,0.])
        res=two_hop(owner,ops,sel,ep,weights)
        assert res['status']=='OK' and len(res['ids'])==2 and res['steps']==2
        assert res['ids']==(ep.source,ep.target)
        assert res['answer']==tuple(ep.values[ep.target])

@pytest.mark.parametrize('bad',[0,1,3,-1,True])
def test_step_limit_failclosed(native,bad):
    ops,sel,lib=native;ep=make_episode(400)
    with open_state(lib,ep) as owner:
        res=two_hop(owner,ops,sel,ep,FROZEN_WEIGHTS,step_limit=bad)
        assert res['status']!='OK'

@pytest.mark.parametrize('mask',[[0]*24,[2]*24,[1]*23,[1.5]*24])
def test_authority_mask_failclosed(native,mask):
    ops,sel,lib=native;ep=make_episode(401)
    with open_state(lib,ep) as owner:
        r=two_hop(owner,ops,sel,ep,FROZEN_WEIGHTS,allowed=mask)
        assert r['status']!='OK'

def test_revoked_source_never_chosen(native):
    ops,sel,lib=native;ep=make_episode(402)
    mask=[1]*24;mask[ep.source]=0
    with open_state(lib,ep) as owner:
        r=two_hop(owner,ops,sel,ep,np.array([2.,0,0,0,0,5.,0,0]),allowed=mask)
        assert r['status']=='OK'
        assert ep.source not in r['ids']

def test_mutation_between_read_and_score_blocks(native):
    ops,sel,lib=native;ep=make_episode(403)
    with open_state(lib,ep) as owner:
        def mutate(o,hop):
            if hop==0:
                assert o.upsert_exact(1000,(.1,.2,.3,.4),'fixture:write:r0').status=='OK'
        res=two_hop(owner,ops,sel,ep,FROZEN_WEIGHTS,inject=mutate)
        assert res['status']=='HOLD_STALE_GENERATION'

def test_native_c_invalid_is_atomic(native):
    ops,sel,lib=native
    x=np.ones((2,8),dtype='float64');w=np.ones(8)
    output=np.full(2,123.4,dtype='float64')
    bad=x.copy();bad[1,7]=float('nan')
    rc=sel.f(2,bad.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
            w.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
            output.ctypes.data_as(ctypes.POINTER(ctypes.c_double)))
    assert rc!=0 and np.all(output==123.4)
    with pytest.raises(ValueError):sel.score(x,np.array([float('inf')]+[0]*7))
    with pytest.raises(ValueError):sel.score(np.zeros((8193,8)),w)

def test_training_deterministic_finite_and_improved():
    a,initial,final=train(seed=3091,episodes=80,epochs=30)
    b,initial2,final2=train(seed=3091,episodes=80,epochs=30)
    assert np.array_equal(a,b) and initial==initial2 and final==final2
    assert final<initial and np.isfinite(a).all() and a.shape==(8,)

@pytest.mark.parametrize('seed',[100,101,102,103])
def test_withheld_goal_signal_triggers_abstention(native,seed):
    ops,sel,lib=native
    w,_,_=train(episodes=128,epochs=50,seed=9001)
    abstained=0
    for i in range(16):
        ep=make_episode(800000+seed*100+i*1009,regime='no_goal')
        with open_state(lib,ep) as owner:
            out=two_hop(owner,ops,sel,ep,w,min_margin=.2)
            abstained+=out['status']=='HOLD_AMBIGUOUS'
    assert abstained>=12

def test_train_label_permutation_negative_control(native):
    ops,sel,lib=native
    w,_,_=train(seed=7110,episodes=240,epochs=65)
    corrupted,_,_=train(seed=7110,episodes=240,epochs=65,shuffle_labels=True)
    assert not np.array_equal(w,corrupted)
    hits={'trained':0,'shuffled':0}
    for i in range(48):
        ep=make_episode(990000+i*65537)
        with open_state(lib,ep) as owner:
            for label,ww in (('trained',w),('shuffled',corrupted)):
                out=two_hop(owner,ops,sel,ep,ww)
                hits[label]+=out['status']=='OK' and out['ids']==(ep.source,ep.target)
    assert hits['trained']>=42 and hits['shuffled']<=hits['trained']-20

def test_pointer_must_be_followed_not_target_label(native):
    ops,sel,lib=native
    ep=make_episode(908300)
    w=np.array([4.,0.,0.,0.,0.,5.,0.,0.])
    with open_state(lib,ep) as owner:
        original=two_hop(owner,ops,sel,ep,w)
    assert original['status']=='OK' and original['ids']==(ep.source,ep.target)
    other=next(i for i in range(len(ep.keys)) if i not in (ep.source,ep.target))
    ep.values[ep.source]=ep.keys[other].copy()
    ep.goals[other]=ep.goals[ep.target].copy()
    with open_state(lib,ep) as owner:
        altered=two_hop(owner,ops,sel,ep,w)
    assert altered['status']=='OK' and altered['ids'][0]==ep.source
    assert altered['ids'][1]!=ep.target

def test_learned_vs_strong_deterministic_and_frozen(native):
    from integration.ersa_selector_r12 import evaluate
    ops,sel,lib=native
    weights,initial,final=train(seed=2900,episodes=384,epochs=80)
    b={'frozen':FROZEN_WEIGHTS,'key_only':np.array([1.,0,0,0,0,0,0,0]),
       'key_goal':np.array([1.,0,0,0,0,1.,0,0])}
    stats=evaluate(weights,selector=sel,ops=ops,owner_lib=lib,seed=7744000,
                   count=96,baselines=b,min_margin=.2)
    assert stats['learned']['success']>=90
    assert stats['learned']['success']>stats['frozen']['success']+20
    assert stats['learned']['success']>stats['key_only']['success']+15
    # Explicitly retain the harder comparator; do not require spurious lift.
    assert stats['key_goal']['success']>=85
