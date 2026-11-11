"""R125 diagnostic-only preclamp census: instrument a volatile copy of the pinned R119 C kernel.

The original R119 kernel remains the authority for *all* causal rollouts. The probe
kernel must produce bitwise-identical one-step outputs or the experiment NONQUALs.
Source patch is in memory; compiler input/output live only under /dev/shm, and the
compiled library is unlinked once loaded. Not a proposed runtime component.
"""
from __future__ import annotations

import ctypes
import hashlib
import subprocess
from research.yang_mills_hca.r118_scalable_awr import provenance as P
from research.yang_mills_hca.r119_numerical_hotpath import native_r119 as N

EXPECTED_R119_SOURCE='125e138b11d0189bc46093b257ec0885ea7e49e875c67fc11b4f91ca21d291e2'

# Instance-local raw proposals, prior to R119's exact clamp guard or actual clamp.
# This is a census only, not a change to the algorithm's decisions.
C_HEADER = r'''
static uint64_t awr125_probe_calls=0, awr125_probe_hits=0;
static double awr125_probe_max=0.0;
void awr125_reset(void) {awr125_probe_calls=0;awr125_probe_hits=0;awr125_probe_max=0.0;}
void awr125_report(uint64_t *calls, uint64_t *hits, double *mx) {
  *calls=awr125_probe_calls; *hits=awr125_probe_hits; *mx=awr125_probe_max;
}
static inline void awr125_precensus(const double *v) {
  double n=sqrt(dot4(v,v));
  awr125_probe_calls++;
  if(n>awr125_probe_max) awr125_probe_max=n;
  if(4.0 / (n + 1e-30) < 1.0) awr125_probe_hits++;
}
'''


def instrument_source():
    raw=N.KERNEL_SOURCE.read_bytes()
    if hashlib.sha256(raw).hexdigest()!=EXPECTED_R119_SOURCE:
        raise RuntimeError('UNPINNED_R119_C_SOURCE')
    source=raw.decode('utf-8')
    # Include after dot4 declaration to make probe independent of declaration order.
    anchor='static inline void clamp4(double *v) {'
    if source.count(anchor)!=1: raise RuntimeError('CLAMP_DEFINITION_ANCHOR')
    source=source.replace(anchor,C_HEADER+'\n'+anchor,1)
    anchor='        if (gmask & AWR_G_CLAMP) {'
    if source.count(anchor)!=1: raise RuntimeError('PRECLAMP_CALL_SITE_ANCHOR')
    source=source.replace(anchor,'        awr125_precensus(ua);\n        awr125_precensus(ub);\n'+anchor,1)
    return source


def build_probe():
    code=instrument_source().encode()
    with P.volatile_scratch('elpis_r125_probe_') as scratch:
        c=scratch/'instrumented.c'; so=scratch/'probe.so'
        c.write_bytes(code)
        p=subprocess.run([N.compiler(),*N.KERNEL_CFLAGS,str(c),'-o',str(so),'-lm'],capture_output=True,text=True)
        if p.returncode: raise RuntimeError('INSTRUMENT_COMPILE:'+p.stderr[-1200:])
        lib=ctypes.CDLL(str(so))
        # Explicit bindings matching r119_numerical_hotpath/native_r119.py.
        vp,i32,u32,f64=ctypes.c_void_p,ctypes.c_int32,ctypes.c_uint32,ctypes.c_double
        lib.awr119_abi_version.restype=ctypes.c_int
        lib.awr119_compact.argtypes=[i32,vp,vp,i32,vp,vp,vp,vp,vp,vp,vp,i32]
        lib.awr119_compact.restype=ctypes.c_int
        lib.awr119_release_marks.argtypes=[i32,vp,vp]
        lib.awr119_release_marks.restype=None
        lib.awr119_execute.argtypes=[i32,vp,vp,i32,vp,vp,vp,vp,vp,vp,vp,vp,vp,vp,i32,f64,f64,i32,i32,vp,vp,vp,vp,i32,u32,f64,vp]
        lib.awr119_execute.restype=ctypes.c_int
        lib.awr125_reset.argtypes=[]
        lib.awr125_reset.restype=None
        lib.awr125_report.argtypes=[vp,vp,vp]
        lib.awr125_report.restype=None
        if lib.awr119_abi_version()!=N.KERNEL_ABI: raise RuntimeError('INSTRUMENT_ABI_DRIFT')
    return lib,hashlib.sha256(code).hexdigest()


def measured_step(S,incoming,checkpoint,frac,teacher_k4,lib):
    from research.yang_mills_hca.r123_k4_prospective.qualify import engine
    eng=engine(S,checkpoint,frac,teacher_k4)
    eng._k119=lib
    lib.awr125_reset()
    S.load(incoming)
    ret=eng.step(S.h)
    if not ret.exact or ret.disposition!='EXECUTED': raise RuntimeError('INSTRUMENT_STEP_REFUSED:'+str(ret.disposition))
    calls=ctypes.c_uint64(); hits=ctypes.c_uint64(); m=ctypes.c_double()
    lib.awr125_report(ctypes.byref(calls),ctypes.byref(hits),ctypes.byref(m))
    if not (0 <= hits.value <= calls.value and calls.value>0): raise RuntimeError('INSTRUMENT_COUNTER_INVALID')
    return S.dump(),{'proposals':calls.value,'clamp_hits':hits.value,'max_preclamp_norm':m.value}
