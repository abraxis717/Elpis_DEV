from dataclasses import replace
import pytest
from elpis.inference.global_context import *
from elpis.inference.contracts import ContractError


def test_full_reuse_reindex_causal_deterministic():
    cfg=IndexConfig('tiny','1'*64,2,2,2,2)
    pool=tuple(GlobalCandidate(str(i),'tiny','1'*64,i,(float(i),1.0),(1.0,float(i)),'2'*64) for i in range(8))
    first=select_global(cfg,pool,(1.,0.),context_snapshot='3'*64,position=4)
    assert [c.object_id for c in selected_values(pool,first)]==['3','4']
    assert first==select_global(cfg,pool,(1.,0.),context_snapshot='3'*64,position=4)
    reuse=select_global(cfg,pool,(-1.,0.),context_snapshot='3'*64,position=4,
                        mode=IndexMode.REUSE,previous=first,expected_previous=first.digest)
    assert reuse.selected==first.selected
    reindex=select_global(cfg,pool,(-1.,0.),context_snapshot='3'*64,position=4,
                         mode=IndexMode.REINDEX,previous=reuse,expected_previous=reuse.digest)
    assert reindex.selected!=first.selected


def test_stale_pool_context_and_wrong_representation():
    cfg=IndexConfig('tiny','1'*64,2,1,1,1)
    pool=(GlobalCandidate('0','tiny','1'*64,0,(1.,2.),(2.,1.),'2'*64),)
    first=select_global(cfg,pool,(1.,0.),context_snapshot='3'*64,position=0)
    for changes in [dict(pool=()),dict(context_snapshot='4'*64),dict(expected_previous='0'*64)]:
        kwargs=dict(config=cfg,pool=pool,query=(1.,0.),context_snapshot='3'*64,position=0,
                    mode=IndexMode.REUSE,previous=first,expected_previous=first.digest)
        kwargs.update(changes)
        with pytest.raises(ContractError): select_global(**kwargs)
    with pytest.raises(ContractError):
        select_global(cfg,(replace(pool[0],model='wrong'),),(1.,0.),context_snapshot='3'*64,position=0)
