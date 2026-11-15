"""Native SSC signed dual-view (norm-softmax) compatibility, separate ABI.

Inputs MUST already be admission-checked canonical vectors. No persistent
state, no transport authority, and no change to ERSA's R7 signed-L1 reducer.
"""
from __future__ import annotations
import ctypes as c
from pathlib import Path
import subprocess,tempfile
import numpy as np

class SscReduceError(ValueError):pass

def build_ssc_reduce():
    d=Path(tempfile.mkdtemp(prefix='ersa-ssc-r8-',dir='/dev/shm'))
    src=Path(__file__).with_name('ersa_ssc_reduce_r8.c')
    out=d/'libersa_ssc_r8.so'
    p=subprocess.run(['cc','-std=c11','-O2','-fPIC','-shared',
        '-fno-fast-math',str(src),'-o',str(out),'-lm'],
        stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True,timeout=30)
    if p.returncode:
        raise RuntimeError('SSC_NATIVE_CC_FAILED:'+p.stdout[-1000:])
    return out

class SscNativeReduce:
    def __init__(self,lib):
        self._lib=c.CDLL(str(lib))
        pp=c.POINTER(c.c_double)
        self._fn=self._lib.ersa_ssc_softmax_r8
        self._fn.argtypes=[c.c_int64,pp,c.c_double,pp,pp]
        self._fn.restype=c.c_int

    def reduce(self,transported,temperature=8.0):
        if type(temperature) not in (int,float) or isinstance(temperature,bool):
            raise SscReduceError('INVALID_TEMPERATURE_TYPE')
        try:
            x=np.asarray(transported,dtype=np.float64)
        except (TypeError,ValueError,OverflowError) as exc:
            raise SscReduceError('INVALID_VECTOR_TYPE') from exc
        if x.ndim!=2 or x.shape[1]!=4 or not 1<=len(x)<=8192:
            raise SscReduceError('INVALID_SHAPE_OR_CAP')
        if not x.flags.c_contiguous:x=np.ascontiguousarray(x)
        a=np.zeros(4,dtype=np.float64);b=np.zeros(4,dtype=np.float64)
        ptr=c.POINTER(c.c_double)
        rc=self._fn(len(x),x.ctypes.data_as(ptr),float(temperature),
            a.ctypes.data_as(ptr),b.ctypes.data_as(ptr))
        if rc:raise SscReduceError('SSC_NATIVE_RC:'+str(rc))
        return tuple(a.tolist()),tuple(b.tolist())
