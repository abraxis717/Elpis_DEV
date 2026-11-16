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
        ids=np.ascontiguousarray(rids,dtype=np.int64)
        eligible=np.ascontiguousarray(eligible,dtype=np.uint8)
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
