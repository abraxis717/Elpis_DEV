import pytest
from elpis.inference.prefetch import *
from elpis.inference.contracts import ContractError
from .test_structural import adapt,proposal_fixture


def test_prefetch_off_on_identical_bytes_wrong_prediction(provider):
    f,path,m,a=provider
    proposal=adapt(proposal_fixture())[0]
    catalog={'python-memory':(RangeHint(a,0,16,2,10,proposal.digest),
                               RangeHint(a,112,16,1,10,proposal.digest))}
    plan=plan_prefetch(committed_state='1'*64,context_snapshot=proposal.context_snapshot,step=0,
                       proposals=(proposal,),catalog=catalog)
    with f.acquire(a,0,16) as lease: baseline=lease.read()
    f.evict()
    results=execute_prefetch(f,plan,state='1'*64,context_snapshot=proposal.context_snapshot,step=0)
    reads=f.stats()['pread_bytes']
    with f.acquire(a,0,16) as lease: assert lease.read()==baseline
    assert f.stats()['pread_bytes']==reads
    comparison=compare_prefetch(results,((a,0,16),))
    assert comparison['useful_hits']==1 and comparison['wasted_pread_bytes']==16
    assert not plan.execution_authority


def test_expiry_and_stale_prefetch(provider):
    f,path,m,a=provider; p=adapt(proposal_fixture())[0]
    plan=plan_prefetch(committed_state='1'*64,context_snapshot=p.context_snapshot,step=2,
                      proposals=(p,),catalog={'python-memory':(RangeHint(a,0,16,1,1,p.digest),)})
    assert plan.hints==()
    with pytest.raises(ContractError,match='STALE'):
        execute_prefetch(f,plan,state='0'*64,context_snapshot=p.context_snapshot,step=2)
