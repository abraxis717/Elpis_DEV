"""ERSA R14: single-call C recurrence over a generation-pinned SSC exact-state view.

This is not a GPU neural layer or direct in-place C state-bank ownership.
Frozen host provides admitted metadata and an exact C-owned state snapshot.
The C kernel performs every select / continue / stop decision within one call.
"""
from __future__ import annotations
import ctypes as ct
import pathlib, shutil, subprocess, tempfile
import numpy as np
from integration.ersa_controller_r13 import KEY_GOAL, MAX_HOPS
from integration.ersa_selector_r12 import read_values

class Receipt(ct.Structure):
    _fields_ = [('status',ct.c_int32),('count',ct.c_int32),
                ('path',ct.c_int32*4),('answer',ct.c_double*4)]

LABELS={0:'OK',1:'HOLD_UNCERTAIN_STOP',2:'HOLD_BUDGET',
        3:'HOLD_NO_CANDIDATE',4:'HOLD_AMBIGUOUS',5:'HOLD_CYCLE'}
POLICIES={'learned':0,'keygoal_learned_halt':1,'strong':2,'fixed2':3}

def compile_recur():
    src=pathlib.Path(__file__).resolve().parents[1]/'native/ersa_recur_r14.c'
    d=pathlib.Path(tempfile.mkdtemp(dir='/dev/shm',prefix='ersa14-recur-'))
    so=d/'libersa14.so'
    p=subprocess.run(['cc','-std=c11','-O2','-Wall','-Wextra','-Werror',
      '-fno-fast-math','-fPIC','-shared',str(src),'-o',str(so),'-lm'],
      capture_output=True,text=True,timeout=30)
    if p.returncode:
        shutil.rmtree(d);raise RuntimeError('C_BUILD_FAILED:'+p.stderr)
    return so

class CRecur:
    def __init__(self,lib):
        self.lib=ct.CDLL(str(lib));self.f=self.lib.ersa14_recur
        D=ct.POINTER(ct.c_double);I=ct.POINTER(ct.c_int64);U=ct.POINTER(ct.c_uint8)
        self.f.argtypes=[ct.c_int64,D,D,D,D,I,U,D,D,D,D,D,ct.c_int,ct.c_int,
                         ct.c_double,ct.POINTER(Receipt)]
        self.f.restype=ct.c_int
    def _prepare(self,ep,values,selector_w,halt_w,allowed=None,policy='learned',max_hops=4,
                 margin_norm=.015):
        n=len(ep.keys)
        if not 1<=n<=8192 or max_hops!=MAX_HOPS or policy not in POLICIES:
            raise ValueError('INVALID_BOUNDS_OR_POLICY')
        keys=np.ascontiguousarray(ep.keys,dtype=np.float64)
        goals=np.ascontiguousarray(ep.goals,dtype=np.float64)
        vals=np.ascontiguousarray(values,dtype=np.float64)
        meta=np.ascontiguousarray(ep.meta,dtype=np.float64)
        if np.asarray(ep.issued).dtype.kind not in 'iu':raise ValueError('INVALID_ISSUED_TYPE')
        issued=np.ascontiguousarray(ep.issued,dtype=np.int64)
        qkey=np.ascontiguousarray(ep.query_key,dtype=np.float64)
        qgoal=np.ascontiguousarray(ep.query_goal,dtype=np.float64)
        qdir=np.ascontiguousarray(ep.query_dir,dtype=np.float64)
        sw=np.ascontiguousarray(selector_w if policy=='learned' else KEY_GOAL,dtype=np.float64)
        hw=np.ascontiguousarray(halt_w if halt_w is not None else np.zeros(6),dtype=np.float64)
        raw_mask=np.asarray(np.ones(n,dtype=np.uint8) if allowed is None else allowed)
        if raw_mask.dtype.kind not in 'ibu' or raw_mask.shape!=(n,) or not np.isin(raw_mask,[0,1]).all():
            raise ValueError('INVALID_PERMISSION_MASK')
        mask=np.ascontiguousarray(raw_mask,dtype=np.uint8)
        if (keys.shape!=(n,4) or goals.shape!=(n,4) or vals.shape!=(n,4) or
            meta.shape!=(n,4) or issued.shape!=(n,) or mask.shape!=(n,) or
            sw.shape!=(8,) or hw.shape!=(6,) or qkey.shape!=(4,) or
            qgoal.shape!=(4,) or qdir.shape!=(4,) or
            not np.isin(mask,[0,1]).all() or not np.isfinite(margin_norm)):
            raise ValueError('INVALID_INPUT_SCHEMA')
        return (n,keys,goals,vals,meta,issued,mask,sw,hw,qkey,qgoal,qdir)
    def run(self,ep,values,selector_w,halt_w,*,allowed=None,policy='learned',max_hops=4,
            margin_norm=.015):
        vals=self._prepare(ep,values,selector_w,halt_w,allowed,policy,max_hops,margin_norm)
        n,*arrays=vals
        keys,goals,values,meta,issued,mask,sw,hw,qkey,qgoal,qdir=arrays
        D=ct.POINTER(ct.c_double);I=ct.POINTER(ct.c_int64);U=ct.POINTER(ct.c_uint8)
        result=Receipt()
        rc=self.f(n,keys.ctypes.data_as(D),goals.ctypes.data_as(D),
            values.ctypes.data_as(D),meta.ctypes.data_as(D),issued.ctypes.data_as(I),
            mask.ctypes.data_as(U),sw.ctypes.data_as(D),hw.ctypes.data_as(D),
            qkey.ctypes.data_as(D),qgoal.ctypes.data_as(D),qdir.ctypes.data_as(D),
            max_hops,POLICIES[policy],float(margin_norm),ct.byref(result))
        if rc!=0:raise ValueError('NATIVE_RECUR_REJECT:'+str(rc))
        if result.status not in LABELS or not 0<=result.count<=max_hops:
            raise RuntimeError('INVALID_NATIVE_RECEIPT')
        path=tuple(int(result.path[i]) for i in range(result.count))
        out={'status':LABELS[result.status],'path':path}
        if result.status==0:
            out.update(answer=tuple(result.answer),steps=result.count)
        return out

def run_on_owner(owner,engine,case,selector_w,halt_w,*,policy='learned',
                 allowed=None,inject=None,margin_norm=.015):
    """Reads one exact C-owned state snapshot and validates its generation.

    The after-run generation check detects concurrent state replacement during
    execution. This is an optimistic read, NOT a lease or atomic native snapshot.
    """
    ep=case.episode
    if allowed is not None and (len(allowed)!=len(ep.keys) or
                                not np.isin(allowed,[0,1]).all()):
        return {'status':'HOLD_AUTHORITY','path':()}
    try:
        # R11 advances its generation on *every* accepted exact-read operation.
        # Therefore do not compare before/after generation across read_values.
        # Its pinned RLock serializes materialization and native recurrence with
        # other owner operations; generation is pinned after read materialization.
        with owner._lock:
            vals=read_values(owner,len(ep.keys))
            pinned=owner.generation
            if inject is not None:inject(owner,'before')
            if owner.generation!=pinned:return {'status':'HOLD_STALE_GENERATION','path':()}
            result=engine.run(ep,vals,selector_w,halt_w,policy=policy,allowed=allowed,
                              margin_norm=margin_norm)
            if inject is not None:inject(owner,'after')
            if owner.generation!=pinned:return {'status':'HOLD_STALE_GENERATION','path':()}
            return result
    except (ValueError,RuntimeError,TypeError,OverflowError):
        return {'status':'HOLD_INVALID_NATIVE','path':()}
