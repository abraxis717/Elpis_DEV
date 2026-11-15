"""R36: C-owned inference slot state for R35 donor-backed ERSA write/read.

This does not replace the R11 SSC entity authority. No FLA Triton kernels are used.
The C recurrence is an independent CPU implementation; Torch remains training.
"""
import ctypes
import os
from pathlib import Path
import subprocess
import tempfile
import shutil
import numpy as np
import torch
from torch.nn import functional as F
from r35.slot_write import SlotState, ERSASlotWriteRead
from integration.ersa_trainable_layer_r18 import ERSAState

HERE=Path(__file__).resolve().parent
C_SOURCE=HERE.parent/'native/ersa_slot_r36.c'


def build_native():
    tmp=Path(os.environ.get('TMPDIR','/dev/shm'))
    if not str(tmp.resolve()).startswith('/dev/shm'):
        raise RuntimeError('COMPILATION_MUST_USE_TMPFS')
    out=Path(tempfile.mkdtemp(prefix='ersa-r36-native-',dir=tmp))
    libpath=out/'slot.so'
    proc=subprocess.run(['cc','-O2','-std=c11','-Wall','-Wextra','-fPIC','-shared','-pthread',
                         str(C_SOURCE),'-lm','-o',str(libpath)],capture_output=True,text=True,timeout=30)
    if proc.returncode:
        shutil.rmtree(out)
        raise RuntimeError('NATIVE_COMPILE_FAILED:'+proc.stderr[-1000:])
    lib=ctypes.CDLL(str(libpath))
    lib.ersa36_new.restype=ctypes.c_void_p
    lib.ersa36_new.argtypes=[ctypes.c_uint32]*3+[ctypes.c_uint64]
    lib.ersa36_free.argtypes=[ctypes.c_void_p]
    lib.ersa36_import.argtypes=[ctypes.c_void_p,ctypes.c_void_p,ctypes.c_void_p,ctypes.c_uint64]
    lib.ersa36_export.argtypes=[ctypes.c_void_p,ctypes.c_void_p,ctypes.c_void_p,ctypes.c_uint64]
    lib.ersa36_step.argtypes=[ctypes.c_void_p]+[ctypes.c_void_p]*6+[ctypes.c_uint64]
    lib.ersa36_generation.argtypes=[ctypes.c_void_p]
    lib.ersa36_generation.restype=ctypes.c_uint64
    return lib,out


def f32(a):
    if isinstance(a,torch.Tensor): a=a.detach().cpu().contiguous().numpy()
    return np.ascontiguousarray(a,dtype=np.float32)

def ptr(a):return ctypes.c_void_p(a.ctypes.data)

class CSlotState:
    def __init__(self,lib,b,d,n,generation=0):
        self.lib=lib;self.b=b;self.d=d;self.n=n
        self.p=lib.ersa36_new(b,d,n,generation)
        if not self.p:raise ValueError('INVALID_OR_UNBOUNDED_SLOT_IMAGE')
    @property
    def step(self):return int(self.lib.ersa36_generation(self.p))
    def close(self):
        if self.p:self.lib.ersa36_free(self.p);self.p=None
    def __enter__(self):return self
    def __exit__(self,*a):self.close()
    def __del__(self):self.close()
    def import_torch(self,state):
        if state.hk.shape!=(self.b,1,self.d,self.n) or state.hv.shape!=(self.b,1,self.n,self.d) or state.step!=self.step:
            raise ValueError('STATE_SHAPE_OR_GENERATION_MISMATCH')
        hk=f32(state.hk);hv=f32(state.hv)
        if self.lib.ersa36_import(self.p,ptr(hk),ptr(hv),self.step):raise ValueError('INVALID_IMPORT')
    def export_torch(self,device='cpu'):
        hk=np.empty((self.b,1,self.d,self.n),dtype=np.float32)
        hv=np.empty((self.b,1,self.n,self.d),dtype=np.float32)
        if self.lib.ersa36_export(self.p,ptr(hk),ptr(hv),self.step):raise RuntimeError('INVALID_EXPORT')
        return SlotState(torch.from_numpy(hk).to(device),torch.from_numpy(hv).to(device),self.step)
    def step_once(self,q,k,v,s,g,*,expected=None):
        if expected is None:expected=self.step
        arrays=[f32(a) for a in (q,k,v,s,g)]
        if [a.shape for a in arrays]!=[(self.b,self.d)]*3+[(self.b,self.n)]*2:
            raise ValueError('INVALID_STEP_TENSOR_SHAPE')
        y=np.empty((self.b,self.d),dtype=np.float32)
        rc=self.lib.ersa36_step(self.p,*map(ptr,arrays),ptr(y),expected)
        if rc:raise ValueError('NATIVE_STEP_REJECTED:'+str(rc))
        return torch.from_numpy(y)


@torch.no_grad()
def infer_native(model,x,state=None,*,write_allowed=None,slot_allowed=None,lib=None,return_aux=False):
    """Inference-only R35 frozen weights, native-owned slot state per call.

    Host evaluates learned projections and R18 relational reads. C exclusively
    evolves hk/hv. Conversion to Torch is a transient read snapshot, not authority.
    """
    if not isinstance(model,ERSASlotWriteRead):raise TypeError('EXPECTED_R35_MODEL')
    if x.device.type!='cpu' or x.dtype!=torch.float32 or x.ndim!=3 or x.shape[-1]!=model.d_model or not torch.isfinite(x).all():
        raise ValueError('INVALID_INPUT')
    b,t,d=x.shape;n=model.num_slots
    if not (1<=b<=128 and 1<=t<=model.max_steps and b*t*n*d*2<=2_000_000):raise ValueError('INVALID_BOUNDS')
    if write_allowed is None:write_allowed=torch.ones((b,t),dtype=torch.bool)
    if slot_allowed is None:slot_allowed=torch.ones((b,n),dtype=torch.bool)
    if write_allowed.shape!=(b,t) or slot_allowed.shape!=(b,n) or write_allowed.dtype!=torch.bool or slot_allowed.dtype!=torch.bool or write_allowed.device.type!='cpu' or slot_allowed.device.type!='cpu':
        raise ValueError('BAD_ADMISSION')
    if state is None:state=model.initial_state(b)
    if not isinstance(state,SlotState) or state.hk.shape!=(b,1,d,n) or state.hv.shape!=(b,1,n,d) or state.step<0 or state.step+t>2**31-1 or not torch.isfinite(state.hk).all() or not torch.isfinite(state.hv).all():
        raise ValueError('BAD_INITIAL_STATE')
    if lib is None:raise ValueError('EXPLICIT_NATIVE_LIB_REQUIRED')
    with CSlotState(lib,b,d,n,state.step) as native:
        native.import_torch(state)
        outs=[];slots=[]
        keybank=model.slot_keys[None].expand(b,n,d)
        meta=torch.zeros(b,n,4)
        issued=torch.zeros(b,n,dtype=torch.int64)
        for i in range(t):
            xi=torch.where(write_allowed[:,i,None],x[:,i],torch.zeros_like(x[:,i]))
            scores=model.router(xi).masked_fill(~slot_allowed,-1e9)
            chosen=torch.zeros((b,n),dtype=torch.bool)
            chosen.scatter_(1,scores.topk(model.topk,dim=-1).indices,True)
            chosen=chosen & slot_allowed & write_allowed[:,i,None]
            g=chosen.float()*(-F.softplus(model.forget(xi))/8.)
            s=-torch.expm1(g)
            native.step_once(model.q_proj(xi),model.k_proj(xi),model.v_proj(xi),s,g)
            snapshot=native.export_torch()
            ersa=ERSAState(keys=keybank,goals=keybank,values=snapshot.hv[:,0],
                           metadata=meta,issued=issued,allowed=slot_allowed,
                           tick=64,generation=state.step+i)
            y,_,_=model.ersa(x[:,i:i+1],ersa,return_aux=True)
            outs.append(y);slots.append(chosen)
        final=native.export_torch()
    result=torch.cat(outs,dim=1)
    if return_aux:return result,final,{'write_slots':torch.stack(slots,dim=1),'native_state_bytes':2*b*d*n*4}
    return result,final
