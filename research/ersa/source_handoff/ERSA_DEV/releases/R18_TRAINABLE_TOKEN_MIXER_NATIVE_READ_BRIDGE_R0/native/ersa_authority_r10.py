"""Opaque, fixed-capacity C RAM-state authority with CAS and zeroizing free.

No Python state copy survives between operations. The Python host computes
candidate transitions, but only a successful C generation-checked replace
commits them. No claim of C-native SSC operation execution.
"""
from __future__ import annotations
import ctypes as C
from pathlib import Path
import subprocess, tempfile, os

class NativeAuthorityHold(RuntimeError):pass

def compile_authority(cc='cc'):
    target=Path(tempfile.mkdtemp(dir='/dev/shm',prefix='ersa-r10-'))/'libersa_authority_r10.so'
    source=Path(__file__).with_suffix('.c')
    res=subprocess.run([cc,'-std=c11','-O2','-pthread','-Wall','-Wextra','-Werror',
        '-fno-fast-math','-fPIC','-shared','-o',str(target),str(source)],
        stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True,timeout=30,
        env=dict(os.environ,TMPDIR='/dev/shm'))
    if res.returncode:raise NativeAuthorityHold('COMPILE_FAILED:'+res.stdout[-2000:])
    return target

class NativeAuthority:
    def __init__(self,library,max_bytes=16*1024*1024):
        if type(max_bytes) is not int or not 128<=max_bytes<=32*1024*1024:
            raise NativeAuthorityHold('CAPACITY_INVALID')
        self.lib=library if isinstance(library,C.CDLL) else C.CDLL(str(library))
        l=self.lib
        l.ersa_authority_new.argtypes=[C.c_size_t];l.ersa_authority_new.restype=C.c_void_p
        l.ersa_authority_free.argtypes=[C.c_void_p]
        l.ersa_authority_generation.argtypes=[C.c_void_p]
        l.ersa_authority_generation.restype=C.c_uint64
        l.ersa_authority_length.argtypes=[C.c_void_p]
        l.ersa_authority_length.restype=C.c_size_t
        l.ersa_authority_capacity.argtypes=[C.c_void_p]
        l.ersa_authority_capacity.restype=C.c_size_t
        l.ersa_authority_replace.argtypes=[C.c_void_p,C.c_uint64,C.c_void_p,C.c_size_t]
        l.ersa_authority_replace.restype=C.c_int
        l.ersa_authority_copy.argtypes=[C.c_void_p,C.c_uint64,C.c_void_p,C.c_size_t]
        l.ersa_authority_copy.restype=C.c_int
        self.ptr=l.ersa_authority_new(max_bytes)
        if not self.ptr:raise NativeAuthorityHold('ALLOC_FAIL')
    def __enter__(self):return self
    def __exit__(self,*args):self.close()
    def close(self):
        if getattr(self,'ptr',None):self.lib.ersa_authority_free(self.ptr);self.ptr=None
    def _p(self):
        if not self.ptr:raise NativeAuthorityHold('CLOSED')
        return self.ptr
    @property
    def generation(self):return int(self.lib.ersa_authority_generation(self._p()))
    @property
    def length(self):return int(self.lib.ersa_authority_length(self._p()))
    @property
    def capacity(self):return int(self.lib.ersa_authority_capacity(self._p()))
    def replace(self,data,expected):
        if type(data) is not bytes or not data:raise NativeAuthorityHold('INVALID_IMAGE')
        if type(expected) is not int or not 0<=expected<1<<64:
            raise NativeAuthorityHold('INVALID_GENERATION')
        # a local copy is the candidate; live authoritative bytes are never
        # exposed to a potentially-mutating Python view.
        cbuf=C.create_string_buffer(data,len(data))
        rc=self.lib.ersa_authority_replace(self._p(),expected,cbuf,len(data))
        if rc:raise NativeAuthorityHold('CAS_REJECT_'+str(rc))
    def read(self,expected=None):
        version=self.generation
        if expected is not None and (type(expected) is not int or version!=expected):
            raise NativeAuthorityHold('STALE_READ')
        length=self.length
        if not length:return b'',version
        buf=C.create_string_buffer(length)
        rc=self.lib.ersa_authority_copy(self._p(),version,buf,length)
        if rc:raise NativeAuthorityHold('COPY_REJECT_'+str(rc))
        return buf.raw,version
