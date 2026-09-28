from dataclasses import replace
import pytest
from elpis.inference.context import *
from elpis.inference.contracts import ContractError


def populated():
    s=initial_snapshot()
    for i in range(4):
        item=ContextItem(str(i),'canonical-evidence',str(i).encode(),Lifetime.DYNAMIC)
        s=append_context(s,item,expected=s.digest)
    return s


def test_compaction_preserves_canonical_evidence_and_forks():
    s=populated()
    summary=ContextItem('summary','tiny-model',b'lossy',Lifetime.DYNAMIC)
    out,record=compact_context(s,0,2,summary,expected=s.digest,replacement_digest=summary.digest,policy='fixture-v1',reason='bounded context')
    verify_compaction(s,out,record)
    assert out.canonical==s.canonical and out.visible[0]==summary
    fork=fork_context(out,'child',expected=out.digest)
    item=ContextItem('temporary','tool',b'data',Lifetime.EPHEMERAL)
    child=append_context(fork,item,expected=fork.digest)
    assert child.visible!=out.visible and out.digest==record.output_snapshot
    retired=retire_context(child,'temporary',expected=child.digest,reason='turn end')
    assert retired.visible==out.visible and retired.canonical[-1]==item


def test_context_planted_defects():
    s=populated(); replacement=ContextItem('summary','model',b'lossy',Lifetime.DYNAMIC)
    with pytest.raises(ContractError,match='STALE'): append_context(s,replacement,expected='0'*64)
    with pytest.raises(ContractError,match='overwrite'): append_context(s,s.canonical[0],expected=s.digest)
    with pytest.raises(ContractError): retire_context(s,'0',expected=s.digest,reason='attempt')
    for bounds in [(-1,2),(2,1),(0,5)]:
        with pytest.raises(ContractError):
            compact_context(s,*bounds,replacement,expected=s.digest,replacement_digest=replacement.digest,policy='p',reason='r')
    with pytest.raises(ContractError,match='replacement'):
        compact_context(s,0,2,replacement,expected=s.digest,replacement_digest='0'*64,policy='p',reason='r')
    out,rec=compact_context(s,0,2,replacement,expected=s.digest,replacement_digest=replacement.digest,policy='p',reason='r')
    for mutated in [replace(rec,input_snapshot='0'*64),replace(rec,replacement='0'*64),replace(rec,preserved=())]:
        with pytest.raises(ContractError): verify_compaction(s,out,mutated)
    assert not replacement.semantic_authority and not out.mutation_authority
    with pytest.raises((AttributeError,TypeError)):
        object.__setattr__(out,'semantic_authority',True)


def test_compaction_verifier_binds_entire_output_snapshot():
    s=populated()
    replacement=ContextItem('summary-x','model',b'lossy',Lifetime.DYNAMIC)
    out,record=compact_context(
        s,1,3,replacement,expected=s.digest,replacement_digest=replacement.digest,
        policy='p',reason='r'
    )
    verify_compaction(s,out,record)
    for mutated in (
        replace(out,branch='forged'),
        replace(out,generation=out.generation+1),
        replace(out,retired=('forged',)),
    ):
        with pytest.raises(ContractError):
            verify_compaction(s,mutated,replace(record,output_snapshot=mutated.digest))


def test_compaction_verifier_enforces_producer_policy_reason_and_identity_rules():
    s=populated()
    summary=ContextItem('summary-parity','model',b'lossy',Lifetime.DYNAMIC)
    out,record=compact_context(
        s,0,2,summary,expected=s.digest,replacement_digest=summary.digest,
        policy='p',reason='r'
    )
    verify_compaction(s,out,record)

    for forged_record in (
        replace(record,policy=''),
        replace(record,reason=''),
    ):
        with pytest.raises(ContractError):
            verify_compaction(s,out,forged_record)

    reused=ContextItem(s.visible[0].object_id,'model',b'forged',Lifetime.DYNAMIC)
    forged_out=replace(
        s,
        generation=s.generation+1,
        predecessor=s.digest,
        visible=(reused,)+s.visible[2:],
    )
    forged_record=CompactionRecord(
        s.digest,
        tuple(i.digest for i in s.visible[:2]),
        tuple(i.digest for i in s.visible[2:]),
        reused.digest,
        'p',
        'r',
        forged_out.digest,
    )
    with pytest.raises(ContractError,match='overwrite canonical evidence'):
        verify_compaction(s,forged_out,forged_record)
