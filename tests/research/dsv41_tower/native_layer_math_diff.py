from __future__ import annotations
import ctypes as C
from pathlib import Path
from types import SimpleNamespace
import sys
import numpy as np
from research.dsv41_tower.numerics import hc_mixes
from research.dsv41_tower.moe import route, expert
P=C.POINTER(C.c_float); U32P=C.POINTER(C.c_uint32)
def ptr(a): return a.ctypes.data_as(P)
def load(path):
    lib=C.CDLL(str(Path(path).resolve()))
    lib.elpis_dsv41_native_abi_version.restype=C.c_uint32
    lib.elpis_dsv41_native_capabilities.restype=C.c_uint64
    lib.elpis_dsv41_hc_mixes_f32.argtypes=[P,P,P,P,P,P,P,P,C.c_size_t,C.c_size_t,C.c_size_t,C.c_float,C.c_float]
    lib.elpis_dsv41_hc_mixes_f32.restype=C.c_int
    lib.elpis_dsv41_route_f32.argtypes=[P,P,P,U32P,P,P,C.c_size_t,C.c_size_t,C.c_size_t,C.c_uint32,C.c_float,C.c_int,C.c_float]
    lib.elpis_dsv41_route_f32.restype=C.c_int
    lib.elpis_dsv41_expert_f32.argtypes=[P,P,P,P,C.c_float,C.c_int,C.c_float,P,P,P,C.c_size_t,C.c_size_t]
    lib.elpis_dsv41_expert_f32.restype=C.c_int
    return lib
def cfg(**kw):
    d=dict(hc_mult=4,hc_sinkhorn_iters=20,norm_eps=1e-6,hc_eps=1e-6,score_func='softmax',active_experts=2,norm_topk_prob=True,gate_temp=1.0,route_scale=1.0,swiglu_limit=0.0)
    d.update(kw); return SimpleNamespace(**d)
def main():
    if len(sys.argv)!=2: raise SystemExit('usage: native_layer_math_diff.py LIB')
    lib=load(sys.argv[1]); assert lib.elpis_dsv41_native_abi_version()==1; assert lib.elpis_dsv41_native_capabilities() & 0xFFF == 0xFFF
    rng=np.random.default_rng(41202); worst=dict(hc_pre=0.0,hc_post=0.0,hc_comb=0.0,route=0.0,expert=0.0)
    for copies,dim,iters in ((2,4,1),(2,8,5),(4,8,20),(4,16,7)):
        c=cfg(hc_mult=copies,hc_sinkhorn_iters=iters); mix_dim=(2+copies)*copies
        for _ in range(32):
            stream=rng.normal(size=(copies,dim)).astype('<f4'); fn=rng.normal(size=(mix_dim,copies*dim)).astype('<f4'); scale=rng.normal(size=3).astype('<f4'); base=rng.normal(size=mix_dim).astype('<f4')
            pre=np.empty(copies,dtype='<f4'); post=np.empty(copies,dtype='<f4'); comb=np.empty((copies,copies),dtype='<f4'); scratch=np.empty(mix_dim,dtype='<f4')
            assert lib.elpis_dsv41_hc_mixes_f32(ptr(stream),ptr(fn),ptr(scale),ptr(base),ptr(pre),ptr(post),ptr(comb),ptr(scratch),copies,dim,iters,c.norm_eps,c.hc_eps)==0
            ep,epo,ec=hc_mixes(stream,fn,scale,base,c)
            worst['hc_pre']=max(worst['hc_pre'],float(np.max(np.abs(pre-ep)))); worst['hc_post']=max(worst['hc_post'],float(np.max(np.abs(post-epo)))); worst['hc_comb']=max(worst['hc_comb'],float(np.max(np.abs(comb-ec))))
            np.testing.assert_allclose(pre,ep,rtol=5e-6,atol=2e-6); np.testing.assert_allclose(post,epo,rtol=5e-6,atol=2e-6); np.testing.assert_allclose(comb,ec,rtol=8e-6,atol=3e-6)
    score_ids={'softmax':0,'sigmoid':1,'sqrtsoftplus':2}
    for score in score_ids:
      for active,normalize in ((1,True),(2,True),(2,False),(4,True)):
        dim,experts=16,7; c=cfg(score_func=score,active_experts=active,norm_topk_prob=normalize,gate_temp=.7,route_scale=1.3)
        for _ in range(48):
            x=rng.normal(size=dim).astype('<f4'); w=rng.normal(size=(experts,dim)).astype('<f4'); b=rng.normal(size=experts).astype('<f4'); w[2]=w[1]; b[2]=b[1]
            ch=np.empty(active,dtype=np.uint32); vals=np.empty(active,dtype='<f4'); scratch=np.empty(experts,dtype='<f4')
            assert lib.elpis_dsv41_route_f32(ptr(x),ptr(w),ptr(b),ch.ctypes.data_as(U32P),ptr(vals),ptr(scratch),dim,experts,active,score_ids[score],c.gate_temp,int(normalize),c.route_scale)==0
            ech,ev=route(x,w,b,c); np.testing.assert_array_equal(ch.astype(np.int64),ech); worst['route']=max(worst['route'],float(np.max(np.abs(vals-ev)))); np.testing.assert_allclose(vals,ev,rtol=6e-6,atol=3e-6)
    for dim,edim,limit in ((4,7,0.0),(16,13,.4),(32,19,3.0)):
      for weighted in (0,1):
        for _ in range(48):
            x=rng.normal(size=dim).astype('<f4'); tensors=[rng.normal(size=(edim,dim)).astype('<f4'),rng.normal(size=(edim,dim)).astype('<f4'),rng.normal(size=(dim,edim)).astype('<f4')]
            rw=np.float32(.37); gate=np.empty(edim,dtype='<f4'); up=np.empty(edim,dtype='<f4'); out=np.empty(dim,dtype='<f4')
            assert lib.elpis_dsv41_expert_f32(ptr(x),ptr(tensors[0]),ptr(tensors[1]),ptr(tensors[2]),float(rw),weighted,float(limit),ptr(gate),ptr(up),ptr(out),dim,edim)==0
            exp=expert(x,tensors,rw if weighted else None,limit); worst['expert']=max(worst['expert'],float(np.max(np.abs(out-exp)))); np.testing.assert_allclose(out,exp,rtol=8e-6,atol=5e-6)
    print('PASS_NATIVE_LAYER_MATH_DIFFERENTIAL')
    for k,v in worst.items(): print(f'{k}_max_abs={v:.9g}')
if __name__=='__main__': main()
