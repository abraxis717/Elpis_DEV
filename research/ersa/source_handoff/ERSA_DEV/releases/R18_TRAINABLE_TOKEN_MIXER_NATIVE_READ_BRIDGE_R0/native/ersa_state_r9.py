"""ERSA R9 fixed-capacity native *shadow* state, not a source of admission authority.

Every rebuild constructs a new bounded C state, checks row validity first, and
publishes it only when all operations succeed. No numpy or persistent files.
"""
from __future__ import annotations
import ctypes as ct
import math
import os
from pathlib import Path
import subprocess
import tempfile

class StateHold(ValueError):
    pass

CARR=ct.c_double*4

def compile_state(cc='cc'):
    tmp=Path(tempfile.mkdtemp(prefix='ersa-r9-',dir='/dev/shm'))
    source=Path(__file__).with_suffix('.c')
    output=tmp/'libersa_state_r9.so'
    args=[cc,'-std=c11','-O2','-fPIC','-shared','-fno-fast-math','-Wall','-Wextra',
          '-Werror','-o',str(output),str(source)]
    res=subprocess.run(args,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,
                       text=True,timeout=30,env=dict(os.environ,TMPDIR='/dev/shm'))
    if res.returncode:
        raise StateHold('C_COMPILATION_FAILED:'+res.stdout[-1200:])
    return output

class NativeState:
    def __init__(self,library,capacity,lo,hi,bound):
        if (type(capacity) is not int or not 1<=capacity<=8192 or
            type(lo) is not int or type(hi) is not int or
            not -(1<<63)<=lo<=hi<(1<<63) or
            type(bound) not in (int,float) or not math.isfinite(bound) or bound<=0):
            raise StateHold('NATIVE_STATE_CONFIG_INVALID')
        self.lib=library if isinstance(library,ct.CDLL) else ct.CDLL(str(library))
        l=self.lib
        l.ersa_state_create.argtypes=[ct.c_int64,ct.c_int64,ct.c_int64,ct.c_double]
        l.ersa_state_create.restype=ct.c_void_p
        l.ersa_state_free.argtypes=[ct.c_void_p]
        l.ersa_state_generation.argtypes=[ct.c_void_p]
        l.ersa_state_generation.restype=ct.c_uint64
        l.ersa_state_count.argtypes=[ct.c_void_p]
        l.ersa_state_count.restype=ct.c_int64
        l.ersa_state_capacity.argtypes=[ct.c_void_p]
        l.ersa_state_capacity.restype=ct.c_int64
        l.ersa_state_upsert.argtypes=[ct.c_void_p,ct.c_int64,ct.POINTER(ct.c_double),ct.c_uint64]
        l.ersa_state_upsert.restype=ct.c_int
        l.ersa_state_delete.argtypes=[ct.c_void_p,ct.c_int64,ct.c_uint64]
        l.ersa_state_delete.restype=ct.c_int
        l.ersa_state_peek.argtypes=[ct.c_void_p,ct.c_int64,ct.POINTER(ct.c_double)]
        l.ersa_state_peek.restype=ct.c_int
        self.cap,self.lo,self.hi,self.bound=capacity,lo,hi,float(bound)
        self.ptr=l.ersa_state_create(capacity,lo,hi,self.bound)
        if not self.ptr:raise StateHold('NATIVE_STATE_ALLOC_FAIL')

    def close(self):
        if getattr(self,'ptr',None):
            self.lib.ersa_state_free(self.ptr)
            self.ptr=None
    def __enter__(self):return self
    def __exit__(self,*args):self.close()
    @property
    def generation(self):return int(self.lib.ersa_state_generation(self._p()))
    @property
    def count(self):return int(self.lib.ersa_state_count(self._p()))
    def _p(self):
        if not self.ptr:raise StateHold('NATIVE_STATE_CLOSED')
        return self.ptr
    def _addr(self,address):
        if type(address) is not int or not -(1<<63)<=address<(1<<63):
            raise StateHold('NATIVE_ADDRESS_TYPE')
        return address
    def upsert(self,address,value,expected):
        address=self._addr(address)
        if (type(expected) is not int or expected<0 or expected>(1<<64)-1):
            raise StateHold('NATIVE_GENERATION_TYPE')
        if (type(value) not in (tuple,list) or len(value)!=4 or
            any(type(v) not in (int,float) or not math.isfinite(v) or abs(v)>self.bound for v in value)):
            raise StateHold('NATIVE_VALUE_INVALID')
        cval=CARR(*map(float,value))
        rc=self.lib.ersa_state_upsert(self._p(),address,cval,expected)
        if rc:raise StateHold('NATIVE_UPSERT_RC:'+str(rc))
    def delete(self,address,expected):
        if type(expected) is not int or not 0<=expected<1<<64:raise StateHold('NATIVE_GENERATION_TYPE')
        rc=self.lib.ersa_state_delete(self._p(),self._addr(address),expected)
        if rc:raise StateHold('NATIVE_DELETE_RC:'+str(rc))
    def peek(self,address):
        buf=CARR()
        rc=self.lib.ersa_state_peek(self._p(),self._addr(address),buf)
        if rc==6:return None
        if rc:raise StateHold('NATIVE_READ_RC:'+str(rc))
        return tuple(buf)

    @classmethod
    def replace_from_rows(cls,old,rows):
        """All-or-nothing native publication: old remains intact on failure."""
        candidate=cls(old.lib,old.cap,old.lo,old.hi,old.bound)
        try:
            if type(rows) not in (list,tuple) or len(rows)>old.cap:
                raise StateHold('NATIVE_ROW_CAP')
            seen=set()
            for row in rows:
                if type(row) not in (tuple,list) or len(row)!=5:
                    raise StateHold('NATIVE_ROW_SCHEMA')
                address,value,*_=row
                if address in seen:raise StateHold('NATIVE_DUPLICATE_ADDRESS')
                seen.add(address)
                candidate.upsert(address,value,candidate.generation)
            return candidate
        except BaseException:
            candidate.close()
            raise
