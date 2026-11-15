"""ERSA R6 native stateless kernels. No native authority/admission surface.

Compile from this directory into /dev/shm ONLY, never install a .so or
mutate the project worktree. C ABI inputs require a prevalidated, authoritative
R5 bank; higher-level code must not treat a native rank as a right to read.
"""
from __future__ import annotations
import ctypes as ct
from pathlib import Path
import subprocess
import tempfile
import os
import math
import numpy as np

MAX=8192
P_DOUBLE=ct.POINTER(ct.c_double)
P_I64=ct.POINTER(ct.c_int64)
P_I32=ct.POINTER(ct.c_int32)
P_U8=ct.POINTER(ct.c_uint8)

class NativeError(ValueError): pass

def build_native(cc='cc'):
    tmp=Path(tempfile.mkdtemp(prefix='ersa-native-',dir='/dev/shm'))
    src=Path(__file__).resolve().with_name('ersa_native_r0.c')
    out=tmp/'libersa_native_r0.so'
    env=dict(os.environ,TMPDIR='/dev/shm')
    p=subprocess.run([cc,'-std=c11','-O2','-fPIC','-shared','-fno-fast-math',
                      '-o',str(out),str(src)],env=env,stdout=subprocess.PIPE,
                     stderr=subprocess.STDOUT,text=True,timeout=30)
    if p.returncode: raise RuntimeError('NATIVE_CC_FAILED:'+p.stdout[-1500:])
    return out

class NativeOps:
    def __init__(self, libpath):
        self.lib=ct.CDLL(str(libpath))
        self.lib.ersa_score_batch_v1.argtypes=[ct.c_int64,P_DOUBLE,P_DOUBLE]
        self.lib.ersa_score_batch_v1.restype=ct.c_int
        self.lib.ersa_rank_canonical_v1.argtypes=[ct.c_int64,ct.c_int64,P_DOUBLE,P_I64,P_U8,P_I32,P_I64]
        self.lib.ersa_rank_canonical_v1.restype=ct.c_int
        self.lib.ersa_signed_reduce_v1.argtypes=[ct.c_int64,P_DOUBLE,P_DOUBLE,P_DOUBLE,P_DOUBLE]
        self.lib.ersa_signed_reduce_v1.restype=ct.c_int
    @staticmethod
    def mat(x,ncol):
        a=np.ascontiguousarray(x,dtype=np.float64)
        if a.ndim!=2 or a.shape[1]!=ncol or a.shape[0]>MAX:
            raise NativeError('BAD_MATRIX_SHAPE')
        if not np.isfinite(a).all():raise NativeError('NONFINITE_INPUT')
        return a
    @staticmethod
    def ptr(a,typ=P_DOUBLE):return a.ctypes.data_as(typ)
    def score(self,features):
        x=self.mat(features,8); y=np.zeros(len(x),np.float64)
        rc=self.lib.ersa_score_batch_v1(len(x),self.ptr(x),self.ptr(y))
        if rc:raise NativeError('SCORE_RC:'+str(rc))
        return y
    def rank(self,scores,rids,eligible,k):
        s=np.ascontiguousarray(scores,dtype=np.float64)
        # Reject before lossy casting: uint8 eligibility and int64 relation
        # identifiers are *authority-adjacent* values, never wrap aliases.
        try:
            raw_ids=list(rids)
            raw_eligibility=list(eligible)
        except (TypeError,ValueError):
            raise NativeError('BAD_RANK_INPUT') from None
        if (any(isinstance(v,(bool,np.bool_)) or not isinstance(v,(int,np.integer))
                or not -(1<<63)<=int(v)<(1<<63) for v in raw_ids) or
            any(isinstance(v,(bool,np.bool_)) or not isinstance(v,(int,np.integer))
                or int(v) not in (0,1) for v in raw_eligibility)):
            raise NativeError('BAD_RANK_ID_OR_ELIGIBILITY')
        ids=np.ascontiguousarray(raw_ids,dtype=np.int64)
        eligible=np.ascontiguousarray(raw_eligibility,dtype=np.uint8)
        if s.ndim!=1 or not 0<=len(s)<=MAX or ids.shape!=s.shape or eligible.shape!=s.shape:
            raise NativeError('BAD_RANK_SHAPE')
        if isinstance(k,bool) or not isinstance(k,(int,np.integer)) or not 1<=k<=MAX:
            raise NativeError('BAD_K')
        if not np.all((eligible==0)|(eligible==1)):
            raise NativeError('INVALID_ELIGIBILITY')
        if not np.isfinite(s[eligible.astype(bool)]).all():
            raise NativeError('NONFINITE_SCORE')
        # Receives CANONICAL scores (R5 uses Python round(score,12)). Native
        # score() is diagnostic until floating rounding parity is certified.
        out=np.zeros(k,dtype=np.int32); count=ct.c_int64(-1)
        rc=self.lib.ersa_rank_canonical_v1(len(s),int(k),self.ptr(s),
             self.ptr(ids,P_I64),self.ptr(eligible,P_U8),self.ptr(out,P_I32),ct.byref(count))
        if rc:raise NativeError('RANK_RC:'+str(rc))
        return out[:count.value].tolist()
    def aggregate(self,vectors,weights):
        v=self.mat(vectors,4)
        w=np.ascontiguousarray(weights,dtype=np.float64)
        if w.ndim!=1 or len(w)!=len(v) or not np.isfinite(w).all():
            raise NativeError('WEIGHT_SHAPE_OR_NONFINITE')
        s=np.zeros(4,dtype=np.float64); c=np.zeros(4,dtype=np.float64)
        rc=self.lib.ersa_signed_reduce_v1(len(v),self.ptr(v),self.ptr(w),self.ptr(s),self.ptr(c))
        if rc:raise NativeError('REDUCE_RC:'+str(rc))
        return tuple(s.tolist()),tuple(c.tolist())

# R7: native feature extraction. This module never owns memory authority.
def _feature_batch(self,keys,goals,vectors,metadata,issued,tick,halflife,
                   query_key,query_goal,query_direction):
    key=self.mat(keys, len(query_key))
    dim=key.shape[1]
    if not 1<=dim<=32:raise NativeError('BAD_FEATURE_DIM')
    goal=self.mat(goals,dim)
    vec=self.mat(vectors,4)
    meta=self.mat(metadata,4)
    n=len(key)
    if any(len(arr)!=n for arr in (goal,vec,meta)):
        raise NativeError('FEATURE_ROW_COUNT')
    if np.any((meta[:,2]!=0)&(meta[:,2]!=-1)) or np.any((meta[:,3]!=0)&(meta[:,3]!=1)):
        raise NativeError('BAD_FEATURE_FLAGS')
    if (type(tick) is not int or not 0<=tick<(1<<63) or
            not np.isfinite(halflife) or halflife<=0):
        raise NativeError('BAD_TIME')
    issued=np.asarray(issued)
    if issued.ndim!=1 or len(issued)!=n or issued.dtype.kind not in ('i','u'):
        raise NativeError('BAD_ISSUED_SHAPE')
    if np.any(issued<0) or np.any(issued>tick):
        raise NativeError('BAD_TIME')
    issued=np.ascontiguousarray(issued,dtype=np.int64)
    qk=np.ascontiguousarray(query_key,dtype=np.float64)
    qd=np.ascontiguousarray(query_direction,dtype=np.float64)
    qg=(np.ascontiguousarray(query_goal,dtype=np.float64) if query_goal is not None else np.zeros(dim,dtype=np.float64))
    if (qk.shape!=(dim,) or qg.shape!=(dim,) or qd.shape!=(4,) or
            not np.isfinite(qk).all() or not np.isfinite(qg).all() or not np.isfinite(qd).all()):
        raise NativeError('BAD_QUERY')
    out=np.empty((n,8),dtype=np.float64)
    rc=self.lib.ersa_features_v2(n,dim,self.ptr(key),self.ptr(goal),self.ptr(vec),
        self.ptr(meta),self.ptr(issued,P_I64),tick,float(halflife),
        self.ptr(qk),self.ptr(qg),self.ptr(qd),int(query_goal is not None),self.ptr(out))
    if rc:raise NativeError('FEATURE_RC:'+str(rc))
    return out

_orig_init=NativeOps.__init__
def _r7_init(self,libpath):
    _orig_init(self,libpath)
    self.lib.ersa_features_v2.argtypes=[ct.c_int64,ct.c_int64,P_DOUBLE,P_DOUBLE,
        P_DOUBLE,P_DOUBLE,P_I64,ct.c_int64,ct.c_double,P_DOUBLE,P_DOUBLE,P_DOUBLE,
        ct.c_int32,P_DOUBLE]
    self.lib.ersa_features_v2.restype=ct.c_int
NativeOps.__init__=_r7_init
NativeOps.features=_feature_batch
