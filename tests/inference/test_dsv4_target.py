from dataclasses import replace
import numpy as np
import pytest
from elpis.inference.drivers.dsv4.fixtures import make_fixture
from elpis.inference.target import LatentInput
from elpis.inference.contracts import ContractError
from elpis.inference.prefetch import RangeHint,PrefetchPlan,execute_prefetch


@pytest.mark.parametrize('scheme',['DSV41_ENGRAM','QWEN38_PLE'])
@pytest.mark.parametrize('dimension',[4,8])
def test_target_exercises_all_planes_and_resident_exactness(provider,tmp_path,scheme,dimension):
    f,*_=provider; target,resident,metadata=make_fixture(f,tmp_path/'neural',dimension=dimension,scheme_name=scheme)
    a=target.initial('1'*64); b=a
    for token in (1,2,3,4,5,6):
        a,receipt=target.step(a,token,expected_state=a.digest)
        b,expected=target.step(b,token,expected_state=b.digest,resident_experts=resident)
        assert a==b and receipt==expected
    assert len(a.local_keys)==4 and len(a.global_pool)==3 and len(a.index.selected)==2
    assert len(a.logits)==16 and np.all(np.isfinite(a.logits))
    assert receipt.rows and receipt.expert_route and receipt.assets
    assert metadata['training']=='NONE'


def test_neural_prefetch_and_routing_hint_equivalence(provider,tmp_path):
    f,*_=provider; target,resident,metadata=make_fixture(f,tmp_path/'neural')
    state=target.initial('1'*64)
    expected,receipt=target.step(state,1,expected_state=state.digest)
    f.evict()
    # Incorrect expert prediction: it can only warm physical pages.
    hint=RangeHint(metadata['expert_asset'],0,16,1,10,'2'*64)
    plan=PrefetchPlan(state.digest,state.context_snapshot,0,(hint,),())
    execute_prefetch(f,plan,state=state.digest,context_snapshot=state.context_snapshot,step=0)
    actual,other=target.step(state,1,expected_state=state.digest)
    assert actual==expected and receipt==other
    projection=target.projections['R']
    latent=LatentInput('R',projection.source_schema,'3'*64,state.context_snapshot,projection.digest,(1.,2.,3.,4.))
    hinted,record=target.step(state,1,expected_state=state.digest,latents=(latent,))
    assert hinted==expected and record.expert_route==receipt.expert_route
    assert record.latents!=receipt.latents
    with pytest.raises(ContractError):
        target.step(state,1,expected_state=state.digest,latents=(replace(latent,projection='0'*64),))
