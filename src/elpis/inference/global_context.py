"""DeepSeek-derived compressed context selection, independently of placement.

Reference: DeepSeek4.1/00_V41_REFERENCE/inference/model.py Indexer,
Compressor and shared attention runtime; MIT, UPSTREAM_REVISION_UNPINNED.
This implementation adopts causal completed groups, rectified side-attention, hierarchical
candidate blocks and index reuse. No QSA backbone or FP4 arithmetic is adopted.
"""
from dataclasses import dataclass
from enum import Enum
import math
import numpy as np
from .contracts import Code, digest_value, identity, integer, require


class IndexMode(str,Enum):
    FULL='FULL'
    REUSE='REUSE'
    REINDEX='REINDEX'


@dataclass(frozen=True)
class GlobalCandidate:
    object_id: str
    model: str
    projection: str
    end_position: int
    key: tuple[float,...]
    value: tuple[float,...]
    source: str

    def __post_init__(self):
        require(bool(self.object_id) and bool(self.model)); digest_value(self.projection); digest_value(self.source)
        integer(self.end_position)
        require(type(self.key) is tuple and type(self.value) is tuple and len(self.key)>0 and len(self.key)==len(self.value))
        require(all(type(x) is float and math.isfinite(x) for x in self.key+self.value),detail='global latent')

    @property
    def digest(self): return identity('global-candidate',self)


@dataclass(frozen=True)
class IndexConfig:
    model: str
    projection: str
    dimension: int
    top_k: int
    block_size: int
    candidate_blocks: int

    def __post_init__(self):
        require(bool(self.model)); digest_value(self.projection)
        for n in (self.dimension,self.top_k,self.block_size,self.candidate_blocks): integer(n,1)

    @property
    def digest(self): return identity('global-index.config',self)


@dataclass(frozen=True)
class IndexResult:
    config: str
    pool: str
    context_snapshot: str
    query: tuple[float,...]
    position: int
    mode: IndexMode
    selected: tuple[str,...]
    predecessor: str | None

    @property
    def digest(self): return identity('global-index.result',self)


def select_global(config,pool,query,*,context_snapshot,position,mode=IndexMode.FULL,
                  previous=None,expected_previous=None):
    require(type(config) is IndexConfig and type(mode) is IndexMode)
    digest_value(context_snapshot); integer(position)
    require(type(pool) is tuple and len({c.object_id for c in pool})==len(pool),detail='global candidate pool')
    for c in pool:
        require(type(c) is GlobalCandidate and (c.model,c.projection,len(c.key))==
                (config.model,config.projection,config.dimension),Code.IDENTITY,'global representation')
    query=tuple(float(x) for x in query)
    require(len(query)==config.dimension and all(math.isfinite(x) for x in query),detail='global query')
    pool_id=identity('global-pool',pool)
    if mode is IndexMode.FULL:
        require(previous is None and expected_previous is None,detail='FULL has no predecessor')
    else:
        require(type(previous) is IndexResult and previous.digest==expected_previous,Code.STALE,'index predecessor')
        require(previous.config==config.digest and previous.context_snapshot==context_snapshot,
                Code.STALE,'index context/config')
        require(position>=previous.position,Code.STALE,'index position')
    if mode is IndexMode.REUSE:
        require(previous.pool==pool_id,Code.STALE,'reuse requires exact shared pool')
        selected=previous.selected
    else:
        eligible=sorted((c for c in pool if c.end_position<=position),key=lambda c:(c.end_position,c.object_id))
        scores={c.digest:max(0.0,float(np.dot(np.asarray(query,dtype='<f4'),np.asarray(c.key,dtype='<f4')))) for c in eligible}
        blocks=[eligible[i:i+config.block_size] for i in range(0,len(eligible),config.block_size)]
        ranked=sorted(enumerate(blocks),key=lambda pair:(-max(scores[c.digest] for c in pair[1]),pair[0]))
        candidates=[c for _,block in ranked[:config.candidate_blocks] for c in block]
        winners=sorted(candidates,key=lambda c:(-scores[c.digest],c.end_position,c.object_id))[:config.top_k]
        selected=tuple(c.digest for c in sorted(winners,key=lambda c:(c.end_position,c.object_id)))
    return IndexResult(config.digest,pool_id,context_snapshot,query,position,mode,selected,
                       None if previous is None else previous.digest)


@dataclass(frozen=True, slots=True)
class StreamCandidate:
    """A compressed KV entry inside an active sequence. Deliberately no provenance.

    Finalization turns it into a :class:`GlobalCandidate` whose ``source`` is
    the ``native-kv-source`` identity over the committed tokens, exactly as the
    legacy step would have written it.
    """
    object_id: str
    end_position: int
    key: tuple[float,...]
    value: tuple[float,...]


def select_global_stream(config, pool, query, *, position):
    """FULL-mode selection of :func:`select_global` without identity work.

    Scores are keyed by ``object_id`` (unique within a pool) where the legacy
    selection keys them by candidate digest; a score depends only on the key,
    so both select the same candidates in the same order.
    """
    require(type(config) is IndexConfig)
    integer(position)
    require(type(pool) is tuple and
            len({c.object_id for c in pool}) == len(pool), detail='stream global pool')
    query=tuple(float(x) for x in query)
    require(len(query)==config.dimension and all(math.isfinite(x) for x in query),detail='global query')
    for c in pool:
        require(type(c) is StreamCandidate and len(c.key)==config.dimension and len(c.value)==config.dimension,
                Code.IDENTITY,'stream global representation')
    eligible=sorted((c for c in pool if c.end_position<=position),
                    key=lambda c:(c.end_position,c.object_id))
    scores={c.object_id:max(0.0,float(np.dot(np.asarray(query,dtype='<f4'),
                                             np.asarray(c.key,dtype='<f4'))))
            for c in eligible}
    blocks=[eligible[i:i+config.block_size] for i in range(0,len(eligible),config.block_size)]
    ranked=sorted(enumerate(blocks),
                  key=lambda pair:(-max(scores[c.object_id] for c in pair[1]),pair[0]))
    candidates=[c for _,block in ranked[:config.candidate_blocks] for c in block]
    winners=sorted(candidates,
                   key=lambda c:(-scores[c.object_id],c.end_position,c.object_id))[:config.top_k]
    return tuple(sorted(winners,key=lambda c:(c.end_position,c.object_id)))


def selected_values(pool,result):
    require(identity('global-pool',pool)==result.pool,Code.STALE,'selected pool')
    by_digest={c.digest:c for c in pool}
    require(all(d in by_digest for d in result.selected),Code.IDENTITY,'selected context')
    return tuple(by_digest[d] for d in result.selected)
