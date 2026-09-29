"""Driver-neutral target contract: tensors, latent inputs, target state and step receipts.

A target driver (the first is :mod:`elpis.inference.drivers.dsv4`) implements
:class:`Target`: it starts a :class:`NeuralState` bound to one context
snapshot and advances it one token at a time, returning the next state and a
:class:`StepReceipt` that binds every input, asset and route it used. The
transaction layer replays and verifies targets only through this contract.
Arithmetic of the numerical profile is explicit CPU F32.
"""
from dataclasses import dataclass
import os
import platform
from typing import Protocol
import numpy as np
from .associative import History
from .contracts import Code, ProposalOnly, identity, integer, require, raw_digest
from .global_context import GlobalCandidate,IndexResult,StreamCandidate


def numerical_profile():
    return identity(
        'numerical-execution-profile.v2',
        dict(
            numpy=np.__version__,
            numpy_config=np.show_config(mode='dicts'),
            system=platform.system(),
            machine=platform.machine(),
            declared_env=dict(
                openblas_coretype=os.environ.get('OPENBLAS_CORETYPE'),
                openblas_num_threads=os.environ.get('OPENBLAS_NUM_THREADS'),
                omp_num_threads=os.environ.get('OMP_NUM_THREADS'),
            ),
        ),
    )


def vector(x): return tuple(float(v) for v in np.asarray(x,dtype='<f4'))


def softmax(x):
    x=np.asarray(x,dtype='<f4'); e=np.exp(x-np.max(x))
    return e/np.sum(e,dtype=np.float32)


@dataclass(frozen=True)
class Tensor:
    shape: tuple[int,...]
    data: bytes

    def __post_init__(self):
        require(type(self.shape) is tuple and len(self.shape)>0)
        count=1
        for n in self.shape: count*=integer(n,1)
        require(type(self.data) is bytes and len(self.data)==count*4,detail='F32 tensor size')
        require(np.all(np.isfinite(self.array())),Code.ENCODING,'model tensor nonfinite')

    def array(self): return np.frombuffer(self.data,dtype='<f4').reshape(self.shape)

    @property
    def digest(self): return identity('tensor',dict(shape=self.shape,dtype='F32_LE',packing='ROW_MAJOR',content=raw_digest(self.data)))


@dataclass(frozen=True)
class LatentProjection:
    channel: str
    source_schema: str
    model: str
    weights: Tensor
    version: int=0

    def __post_init__(self):
        require(self.channel in ('M','G','X','R') and bool(self.source_schema) and bool(self.model))
        require(len(self.weights.shape)==2 and self.version==0)

    @property
    def digest(self):
        return identity('latent-projection',dict(channel=self.channel,source_schema=self.source_schema,
                        model=self.model,weights=self.weights.digest,version=self.version))


@dataclass(frozen=True)
class LatentInput(ProposalOnly):
    channel: str
    source_schema: str
    source: str
    context_snapshot: str
    projection: str
    values: tuple[float,...]

    @property
    def digest(self): return identity('latent-input',self)


@dataclass(frozen=True)
class TargetConfig:
    model: str
    tokenizer: str
    vocab: int
    dimension: int
    local_window: int
    compression: int
    global_top_k: int
    expert_ids: tuple[int,...]
    active_experts: int
    shared_experts: tuple[int,...]=()
    layer: int=0
    max_tokens: int=4096

    def __post_init__(self):
        require(bool(self.model) and bool(self.tokenizer))
        for n in (self.vocab,self.dimension,self.local_window,self.compression,self.global_top_k,self.active_experts,self.max_tokens): integer(n,1)
        require(type(self.expert_ids) is tuple and len(set(self.expert_ids))==len(self.expert_ids))
        require(self.active_experts<=len(self.expert_ids) and not set(self.shared_experts).intersection(self.expert_ids))
        for i in self.expert_ids+self.shared_experts: integer(i)

    @property
    def digest(self): return identity('target-config',self)


@dataclass(frozen=True)
class NeuralState:
    model: str
    context_snapshot: str
    numerical_profile: str
    tokens: tuple[int,...]
    history: History
    local_keys: tuple[tuple[float,...],...]=()
    local_values: tuple[tuple[float,...],...]=()
    pending: tuple[tuple,...]=()  # (K,V,compression score)
    global_pool: tuple[GlobalCandidate,...]=()
    index: IndexResult | None=None
    hidden: tuple[float,...]=()
    logits: tuple[float,...]=()

    @property
    def digest(self): return identity('neural-state',self)


@dataclass(frozen=True)
class StreamingNeuralState:
    """Ephemeral model-local state of an active sequence. Not authority, not durable.

    Deliberately has no digest/provenance API; the global pool holds
    ``StreamCandidate`` entries without provenance. Only finalization at the
    commit boundary produces a :class:`NeuralState`.
    """
    model: str
    context_snapshot: str
    numerical_profile: str
    tokens: tuple[int,...]
    history: History
    local_keys: tuple[tuple[float,...],...]=()
    local_values: tuple[tuple[float,...],...]=()
    pending: tuple[tuple,...]=()
    global_pool: tuple=()
    hidden: tuple[float,...]=()
    logits: tuple[float,...]=()
    step: 'StreamStep | None'=None

    def record(self):
        """The bounded part of this state finalization needs (tokens and pool are rebuilt)."""
        return StreamRecord(self.step,self.history,self.local_keys,self.local_values,self.pending,
                            self.hidden,self.logits)


@dataclass(frozen=True, slots=True)
class StreamStep:
    """What one stream step observed: model data only, kept for the commit boundary."""
    token: int
    rows: tuple            # address rows for the token, every layer (as HashResult.rows)
    memory: tuple[float,...]
    query: tuple[float,...]
    route: tuple[int,...]
    selected: tuple[str,...]   # object ids of the selected global candidates
    candidate: StreamCandidate | None


@dataclass(frozen=True, slots=True)
class StreamRecord:
    step: StreamStep
    history: History
    local_keys: tuple
    local_values: tuple
    pending: tuple
    hidden: tuple[float,...]
    logits: tuple[float,...]


@dataclass(frozen=True)
class StepReceipt:
    model: str
    tokenizer: str
    numerical_profile: str
    input_state: str
    context_snapshot: str
    token: int
    scheme: str
    parameters: str
    associative_result: str
    bank: str
    rows: tuple[int,...]
    global_index: str
    latents: tuple[str,...]
    expert_route: tuple[int,...]
    expert_contents: tuple[str,...]
    assets: tuple[str,...]
    output_state: str

    @property
    def digest(self): return identity('target-step',self)


class Target(Protocol):
    """What the transaction, speculative verification and steering require of a driver."""

    config: TargetConfig
    model_identity: str
    numerical_profile: str
    last_metrics: dict

    def initial(self, context_snapshot: str) -> NeuralState: ...

    def step(self, state: NeuralState, token: int, *, expected_state: str,
             latents: tuple = (), **options) -> tuple[NeuralState, StepReceipt]: ...

    def admit_stream(self, *, resident_experts=None): ...

    def initial_stream(self, context_snapshot: str) -> StreamingNeuralState: ...

    def resume_stream(self, state: NeuralState) -> StreamingNeuralState: ...

    def finalize_stream(self, committed: NeuralState, records: tuple, *,
                        latents: tuple = ()) -> tuple[tuple[NeuralState, StepReceipt], ...]: ...

    def stream_step(self, state: StreamingNeuralState, token: int, *, admission,
                    latents: tuple = ()) -> StreamingNeuralState: ...

