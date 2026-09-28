"""Immutable context lifetimes and model-view compaction, without evidence mutation.

Concept reference: ElpisDonors/ZCode/context/context-types.ts,
compaction/compact-contract.ts, state/session-fork.ts; Apache-2.0,
revision 872ad960de7ec172591f7e1952f7849229f94521. Elpis-native contracts.
"""
from dataclasses import dataclass, replace
from enum import Enum
from .contracts import Code, ProposalOnly, digest_value, identity, integer, require


class Lifetime(str,Enum):
    STABLE='STABLE'
    DYNAMIC='DYNAMIC'
    EPHEMERAL='EPHEMERAL'
    SPECULATIVE='SPECULATIVE'


@dataclass(frozen=True)
class ContextItem(ProposalOnly):
    object_id: str
    source: str
    content: bytes
    lifetime: Lifetime

    def __post_init__(self):
        require(type(self.content) is bytes and bool(self.object_id) and bool(self.source))
        require(type(self.lifetime) is Lifetime)

    @property
    def digest(self): return identity('context.item',self)


@dataclass(frozen=True)
class Snapshot(ProposalOnly):
    branch: str
    generation: int
    predecessor: str | None
    canonical: tuple[ContextItem,...]
    visible: tuple[ContextItem,...]
    retired: tuple[str,...]=()

    def __post_init__(self):
        integer(self.generation)
        require(bool(self.branch))
        if self.predecessor is not None: digest_value(self.predecessor)
        for items in (self.canonical,self.visible):
            require(type(items) is tuple and all(type(i) is ContextItem for i in items))
            require(len({i.object_id for i in items})==len(items),detail='duplicate context object')
        require(type(self.retired) is tuple and len(set(self.retired))==len(self.retired))
        require(not set(self.retired).intersection(i.object_id for i in self.visible),detail='retired visible object')

    @property
    def digest(self): return identity('context.snapshot',self)


def initial_snapshot(branch='main'):
    return Snapshot(branch,0,None,(),())


def _check(snapshot,expected):
    require(type(snapshot) is Snapshot and snapshot.digest==expected,Code.STALE,'context snapshot predecessor')


def append_context(snapshot,item,*,expected):
    _check(snapshot,expected)
    require(type(item) is ContextItem)
    require(item.object_id not in {i.object_id for i in snapshot.canonical+snapshot.visible},Code.IDENTITY,'canonical overwrite')
    require(item.object_id not in snapshot.retired,Code.IDENTITY,'retired identity reuse')
    return replace(snapshot,generation=snapshot.generation+1,predecessor=snapshot.digest,
                   canonical=snapshot.canonical+(item,),visible=snapshot.visible+(item,))


def fork_context(snapshot,branch,*,expected):
    _check(snapshot,expected)
    require(bool(branch) and branch!=snapshot.branch,detail='distinct fork identity')
    return replace(snapshot,branch=branch,generation=0,predecessor=snapshot.digest)


def retire_context(snapshot,object_id,*,expected,reason):
    _check(snapshot,expected)
    matches=[i for i in snapshot.visible if i.object_id==object_id]
    require(len(matches)==1 and bool(reason),detail='retirement object/reason')
    require(matches[0].lifetime in (Lifetime.EPHEMERAL,Lifetime.SPECULATIVE),Code.INVALID,'stable/dynamic retirement requires compaction')
    return replace(snapshot,generation=snapshot.generation+1,predecessor=snapshot.digest,
                   visible=tuple(i for i in snapshot.visible if i.object_id!=object_id),
                   retired=snapshot.retired+(object_id,))


@dataclass(frozen=True)
class CompactionRecord(ProposalOnly):
    input_snapshot: str
    compacted: tuple[str,...]
    preserved: tuple[str,...]
    replacement: str
    policy: str
    reason: str
    output_snapshot: str

    @property
    def digest(self): return identity('context.compaction',self)


def compact_context(snapshot,start,stop,replacement,*,expected,replacement_digest,policy,reason):
    _check(snapshot,expected)
    integer(start); integer(stop,1)
    require(start<stop<=len(snapshot.visible),detail='compaction boundary')
    require(replacement.digest==replacement_digest,Code.IDENTITY,'replacement digest')
    require(bool(policy) and bool(reason),detail='compaction policy/reason')
    require(replacement.object_id not in {i.object_id for i in snapshot.canonical+snapshot.visible},
            Code.IDENTITY,'summary must not overwrite canonical evidence')
    visible=snapshot.visible[:start]+(replacement,)+snapshot.visible[stop:]
    out=replace(snapshot,generation=snapshot.generation+1,predecessor=snapshot.digest,visible=visible)
    record=CompactionRecord(snapshot.digest,tuple(i.digest for i in snapshot.visible[start:stop]),
                           tuple(i.digest for i in snapshot.visible[:start]+snapshot.visible[stop:]),
                           replacement.digest,policy,reason,out.digest)
    return out,record


def verify_compaction(before,after,record):
    require(type(before) is Snapshot and type(after) is Snapshot and
            type(record) is CompactionRecord,detail='compaction record types')
    require(bool(record.policy) and bool(record.reason),detail='compaction policy/reason')
    require(record.input_snapshot==before.digest and
            record.output_snapshot==after.digest and
            after.predecessor==before.digest,Code.STALE,'compaction lineage')
    require(after.branch==before.branch,Code.IDENTITY,'compaction branch changed')
    require(after.generation==before.generation+1,Code.IDENTITY,'compaction generation')
    require(after.retired==before.retired,Code.IDENTITY,'compaction retired set changed')
    require(before.canonical==after.canonical,Code.IDENTITY,'canonical evidence changed')
    input_ids=tuple(i.digest for i in before.visible)
    require(record.compacted and all(d in input_ids for d in record.compacted))
    start=input_ids.index(record.compacted[0]); stop=start+len(record.compacted)
    require(input_ids[start:stop]==record.compacted,detail='noncontiguous compaction')
    require(input_ids[:start]+input_ids[stop:]==record.preserved,detail='preserved region')
    require(len(after.visible)==len(before.visible)-len(record.compacted)+1,
            Code.IDENTITY,'compaction output geometry')
    replacement=after.visible[start]
    require(replacement.digest==record.replacement,Code.IDENTITY,'compaction replacement')
    require(replacement.object_id not in {i.object_id for i in before.canonical+before.visible},
            Code.IDENTITY,'summary must not overwrite canonical evidence')
    expected=replace(
        before,generation=before.generation+1,predecessor=before.digest,
        visible=before.visible[:start]+(replacement,)+before.visible[stop:]
    )
    require(after==expected,Code.IDENTITY,'compaction output snapshot')
    require(record.output_snapshot==expected.digest,Code.IDENTITY,'compaction output digest')
