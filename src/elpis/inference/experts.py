"""Exact expert materialization and fixed-order CPU SwiGLU execution.

Architectural reference: DeepSeek V4.1 inference/model.py Expert/MoE, MIT,
UPSTREAM_REVISION_UNPINNED; see LICENSES/PROVENANCE.md.
Equality contract preregistered: identical F32 bytes, kernel and reduction
order must yield bitwise-identical resident and streamed CPU outputs.
"""
from contextlib import contextmanager
from dataclasses import dataclass
from threading import RLock
from time import perf_counter_ns
import numpy as np
from .contracts import Code, digest_value, identity, integer, require
from elpis.substrate.digests import raw_digest


@dataclass(frozen=True)
class ExpertTensor:
    role: str
    shape: tuple[int,int]
    asset: str
    offset: int
    content: str
    dtype: str='F32_LE'
    quantization: str='NONE'
    packing: str='ROW_MAJOR'

    def __post_init__(self):
        require(self.role in ('gate','up','down'))
        require(type(self.shape) is tuple and len(self.shape)==2)
        for d in self.shape: integer(d,1)
        integer(self.offset); digest_value(self.asset); digest_value(self.content)
        require((self.dtype,self.quantization,self.packing)==('F32_LE','NONE','ROW_MAJOR'),
                Code.UNSUPPORTED,'expert representation')

    @property
    def size(self): return self.shape[0]*self.shape[1]*4


@dataclass(frozen=True)
class ExpertManifest:
    model: str
    layer: int
    expert_id: int
    tensors: tuple[ExpertTensor,...]
    abi: str='ELPIS_SWIGLU_F32_CPU_R0'

    def __post_init__(self):
        require(bool(self.model) and self.abi=='ELPIS_SWIGLU_F32_CPU_R0')
        integer(self.layer); integer(self.expert_id)
        require(type(self.tensors) is tuple and tuple(t.role for t in self.tensors)==('gate','up','down'),detail='tensor roles/order')
        gate,up,down=self.tensors
        require(gate.shape==up.shape and down.shape==gate.shape[::-1],detail='expert shapes')

    @property
    def digest(self): return identity('expert-manifest',self)


def expert_kernel(x,tensors):
    gate,up,down=tensors
    g=np.asarray(x,dtype='<f4')@gate
    with np.errstate(over='ignore'):
        activation=g/(1+np.exp(-g))
    result=(activation*(np.asarray(x,dtype='<f4')@up))@down
    require(np.all(np.isfinite(result)),Code.ENCODING,'expert nonfinite output')
    return result.astype('<f4')


class ExpertBank:
    def __init__(self,provider,manifests,*,model,expected_digests,staging_bytes=1<<20):
        integer(staging_bytes,1)
        require(type(manifests) is tuple and len(manifests)>0)
        require(tuple(m.digest for m in manifests)==expected_digests,Code.IDENTITY,'expert manifest pin')
        require(all(m.model==model for m in manifests),Code.IDENTITY,'expert model')
        self.manifests={(m.layer,m.expert_id):m for m in manifests}
        require(len(self.manifests)==len(manifests),Code.IDENTITY,'duplicate expert')
        for m in manifests:
            require(sum(t.size for t in m.tensors)*3<=staging_bytes,Code.LIMIT,'expert staging budget')
            for t in m.tensors:
                require(t.asset in provider._assets,Code.MISSING,'expert asset')
                require(t.offset+t.size<=provider._assets[t.asset][1].size,detail='expert tensor range')
        self.provider=provider; self.model=model; self.staging_budget=staging_bytes
        self._lock=RLock(); self.staged_bytes=0; self.high_water=0; self.last_metrics={}
        self._bindings_verified=False

    @property
    def digest(self):
        return identity('expert-bank',tuple(m.digest for _,m in sorted(self.manifests.items())))

    def manifest(self,layer,expert):
        require((layer,expert) in self.manifests,Code.MISSING,'exact selected expert')
        return self.manifests[layer,expert]

    def _decode(self,manifest,raws):
        require(len(raws)==3,detail='expert tensor count')
        tensors=[]
        for tensor,raw in zip(manifest.tensors,raws):
            require(type(raw) is bytes and len(raw)==tensor.size and raw_digest(raw)==tensor.content,
                    Code.INTEGRITY,'expert tensor content')
            value=np.frombuffer(raw,dtype='<f4').reshape(tensor.shape)
            require(np.all(np.isfinite(value)),Code.ENCODING,'expert tensor nonfinite')
            tensors.append(value)
        return tuple(tensors)

    def _read(self,manifest):
        raws=[]
        for tensor in manifest.tensors:
            data=bytearray(); offset=tensor.offset
            while len(data)<tensor.size:
                page_size=self.provider._assets[tensor.asset][1].page_size
                count=min(tensor.size-len(data),page_size-offset%page_size)
                with self.provider.acquire(tensor.asset,offset,count) as lease: data.extend(lease.read())
                offset+=count
            raws.append(bytes(data))
        return raws

    @contextmanager
    def _staged(self,manifest):
        reservation=sum(t.size for t in manifest.tensors)*3
        require(self.staged_bytes+reservation<=self.staging_budget,Code.LIMIT,'expert staging exhausted')
        self.staged_bytes+=reservation; self.high_water=max(self.high_water,self.staged_bytes)
        try:
            yield
        finally:
            self.staged_bytes-=reservation

    @contextmanager
    def acquire(self,layer,expert):
        with self._lock:
            manifest=self.manifest(layer,expert)
            with self._staged(manifest):
                yield self._decode(manifest,self._read(manifest))

    def admit(self,*,resident=None):
        """Verify expert bytes against their manifests once, before a sequence.

        With ``resident`` (raw bytes per (layer, expert), as for ``execute``)
        every manifest is verified and decoded once and the returned admission
        holds the read-only arrays. Without it, each manifest's content binding
        to its asset range is verified once per bank (the substrate still
        verifies every cold page it faults in); later reads decode without
        re-hashing. Either way the admission is the only input
        :meth:`execute_admitted` accepts.
        """
        with self._lock:
            if resident is None:
                if not self._bindings_verified:
                    for key in sorted(self.manifests):
                        with self.acquire(*key):
                            pass
                    self._bindings_verified=True
                return ExpertAdmission(self,None)
            require(type(resident) is dict and set(resident)>=set(self.manifests),Code.MISSING,
                    'resident experts must cover the bank')
            return ExpertAdmission(self,{key:self._decode(self.manifests[key],resident[key])
                                         for key in sorted(self.manifests)})

    def execute(self,x,*,model,layer,route,weights,resident=None,shared=()):
        require(model==self.model,Code.IDENTITY,'execution model')
        require(type(route) is tuple and len(set(route))==len(route) and len(route)==len(weights),detail='expert route')
        require(type(shared) is tuple and len(set(shared))==len(shared) and not set(shared).intersection(route))
        require(all(np.isfinite(w) for w in weights),detail='routing weights')
        x=np.asarray(x,dtype='<f4')
        require(x.ndim==1 and np.all(np.isfinite(x)),detail='expert input')
        result=np.zeros_like(x); materialize_ns=execution_ns=byte_count=0
        for expert,weight in tuple(zip(route,weights))+tuple((i,1.0) for i in shared):
            m=self.manifest(layer,expert)
            require(x.shape==(m.tensors[0].shape[0],),detail='expert input dimension')
            byte_count+=sum(t.size for t in m.tensors)
            start=perf_counter_ns()
            if resident is None:
                with self.acquire(layer,expert) as tensors:
                    materialize_ns+=perf_counter_ns()-start
                    begin=perf_counter_ns()
                    result+=np.float32(weight)*expert_kernel(x,tensors)
                    execution_ns+=perf_counter_ns()-begin
            else:
                require((layer,expert) in resident,Code.MISSING,'resident exact expert')
                tensors=self._decode(m,resident[layer,expert])
                materialize_ns+=perf_counter_ns()-start
                begin=perf_counter_ns()
                result+=np.float32(weight)*expert_kernel(x,tensors)
                execution_ns+=perf_counter_ns()-begin
        self.last_metrics=dict(selected_experts=len(route)+len(shared),expert_bytes=byte_count,
                               materialization_ns=materialize_ns,execution_ns=execution_ns,
                               staging_high_water=self.high_water,transfer_overlap_ns=None,
                               staging_high_water_kind='configured_reservation')
        return result


    def execute_admitted(self,x,*,admission,model,layer,route,weights,shared=()):
        """Stream-lane execution over an admission: no per-call content hashing.

        Arithmetic is :func:`expert_kernel` on the same F32 arrays as
        :meth:`execute`, so results are bitwise identical.
        """
        require(type(admission) is ExpertAdmission and admission.bank is self,Code.IDENTITY,'expert admission')
        require(model==self.model,Code.IDENTITY,'execution model')
        require(type(route) is tuple and len(set(route))==len(route) and len(route)==len(weights),detail='expert route')
        require(type(shared) is tuple and len(set(shared))==len(shared) and not set(shared).intersection(route))
        x=np.asarray(x,dtype='<f4')
        result=np.zeros_like(x); materialize_ns=execution_ns=byte_count=0
        for expert,weight in tuple(zip(route,weights))+tuple((i,1.0) for i in shared):
            m=self.manifest(layer,expert)
            require(x.shape==(m.tensors[0].shape[0],),detail='expert input dimension')
            byte_count+=sum(t.size for t in m.tensors)
            start=perf_counter_ns()
            if admission.resident is not None:
                tensors=admission.resident[layer,expert]
                materialize_ns+=perf_counter_ns()-start
                begin=perf_counter_ns()
                result+=np.float32(weight)*expert_kernel(x,tensors)
                execution_ns+=perf_counter_ns()-begin
            else:
                with self._lock, self._staged(m):
                    tensors=tuple(np.frombuffer(raw,dtype='<f4').reshape(t.shape)
                                  for t,raw in zip(m.tensors,self._read(m)))
                    materialize_ns+=perf_counter_ns()-start
                    begin=perf_counter_ns()
                    result+=np.float32(weight)*expert_kernel(x,tensors)
                    execution_ns+=perf_counter_ns()-begin
        self.last_metrics=dict(selected_experts=len(route)+len(shared),expert_bytes=byte_count,
                               materialization_ns=materialize_ns,execution_ns=execution_ns,
                               staging_high_water=self.high_water,transfer_overlap_ns=None,
                               staging_high_water_kind='configured_reservation')
        return result


class ExpertAdmission:
    """Expert bytes admitted for one bank before a sequence. Not provenance.

    ``resident`` holds verified read-only arrays per (layer, expert), or is
    None when experts are read through the substrate on use.
    """
    __slots__=('bank','resident')

    def __init__(self,bank,resident):
        self.bank=bank; self.resident=resident
