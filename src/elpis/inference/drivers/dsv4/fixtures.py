"""Explicit synthetic fixture intake, never a production parameter generator.

No training or donor evaluation data is used. Every artifact binds seed,
geometry, tokenizer, schema and resulting content. This module is opt-in.
"""
import numpy as np
from ...associative import AddressScheme,DSV41Parameters,QwenPLEParameters
from ...contracts import Bank,require
from ...experts import ExpertBank,ExpertManifest,ExpertTensor
from elpis.substrate.file_assets import bounded_path,inspect_asset
from elpis.substrate.digests import raw_digest
from ...target import TargetConfig,Tensor,LatentProjection
from .target import CompactTarget
from ...rows import RowEngine,RowTable,row_representation


def make_fixture(provider,directory,*,dimension=4,scheme_name='DSV41_ENGRAM',seed=917):
    directory=bounded_path(provider.root,directory); directory.mkdir(parents=True,exist_ok=True)
    require(scheme_name in ('DSV41_ENGRAM','QWEN38_PLE'))
    tokenizer='synthetic-raw16-'+scheme_name
    model='fixture-model-'+str(dimension)+'-'+scheme_name
    if scheme_name=='DSV41_ENGRAM':
        parameters=DSV41Parameters(tokenizer,tuple(i%8 for i in range(16)),8,0,(0,),3,1,dimension,
                                   ((3,5,7),),((101,103),),((0,101),),(204,))
    else:
        parameters=QwenPLEParameters(tokenizer,16,15,3,1,dimension*2,(3,5,7),(101,103),(0,101),204)
    scheme=AddressScheme(parameters,expected_digest=parameters.digest,tokenizer=tokenizer,scheme=parameters.schema)
    rng=np.random.default_rng(seed)
    def tensor(shape,scale=.1):
        a=(rng.normal(size=shape)*scale).astype('<f4')
        return Tensor(tuple(shape),a.tobytes())
    memory=tensor((204,dimension))
    memory_path=directory/'associative.dat'; memory_path.write_bytes(memory.data)
    manifest=inspect_asset(provider.root,memory_path,16)
    memory_asset=provider.register(memory_path,manifest,expected_manifest=manifest.digest)
    bank=Bank('fixture-bank',model,tokenizer,parameters.schema,parameters.digest,0,204,dimension,
               manifest.content,row_representation(memory_asset,0,204,dimension,'F32_LE'))
    rows=RowEngine(provider,RowTable(bank,memory_asset,0,'F32_LE'),expected_bank=bank.digest)
    expert_data=[]; resident={}
    for i in range(4):
        resident[0,i]=tuple(tensor((dimension,dimension)).data for _ in range(3))
        expert_data.extend(resident[0,i])
    expert_path=directory/'experts.dat'; expert_path.write_bytes(b''.join(expert_data))
    em=inspect_asset(provider.root,expert_path,16)
    expert_asset=provider.register(expert_path,em,expected_manifest=em.digest)
    experts=[]; offset=0
    for i in range(4):
        tensors=[]
        for role,data in zip(('gate','up','down'),resident[0,i]):
            tensors.append(ExpertTensor(role,(dimension,dimension),expert_asset,offset,raw_digest(data)))
            offset+=len(data)
        experts.append(ExpertManifest(model,0,i,tuple(tensors)))
    experts=tuple(experts)
    eb=ExpertBank(provider,experts,model=model,expected_digests=tuple(e.digest for e in experts))
    config=TargetConfig(model,tokenizer,16,dimension,4,2,2,(0,1,2),2,(3,))
    weights={name:tensor(shape) for name,shape in {'embedding':(16,dimension),'q':(dimension,dimension),
              'k':(dimension,dimension),'v':(dimension,dimension),'out':(dimension,16),
              'router':(dimension,3),'compress':(dimension,1)}.items()}
    projections=tuple(LatentProjection(channel,schema,model,tensor((dimension,dimension)))
                      for channel,schema in [('M','associative-row.r0'),('G','structural-latent.r0'),
                                             ('X','executive-latent.r0'),('R','routing-proposal.r0')])
    target=CompactTarget(config,weights,projections,scheme,rows,eb)
    return target,resident,dict(seed=seed,training='NONE',model=target.model_identity,
                configuration=config.digest,parameter_artifact=parameters.digest,
                memory_asset=memory_asset,expert_asset=expert_asset,
                evaluation_data='synthetic qualification inputs; no donor benchmark data')
