from dataclasses import replace
import numpy as np
import pytest
from elpis.inference.experts import ExpertBank,ExpertManifest,ExpertTensor
from elpis.substrate.file_assets import inspect_asset
from elpis.substrate.digests import raw_digest
from elpis.inference.contracts import ContractError


def expert_fixture(f,tmp_path,dimension=4,count=4):
    rng=np.random.default_rng(917)
    chunks=[]; records=[]; resident={}; offset=0
    for expert in range(count):
        raws=tuple((rng.normal(size=(dimension,dimension))*.1).astype('<f4').tobytes() for _ in range(3))
        resident[0,expert]=raws
        records.append((expert,offset,raws)); chunks.extend(raws); offset+=sum(map(len,raws))
    path=tmp_path/'experts.dat'; path.write_bytes(b''.join(chunks))
    m=inspect_asset(f.root,path,16); asset=f.register(path,m,expected_manifest=m.digest)
    manifests=[]
    for expert,offset,raws in records:
        tensors=[]
        for role,raw in zip(('gate','up','down'),raws):
            tensors.append(ExpertTensor(role,(dimension,dimension),asset,offset,raw_digest(raw))); offset+=len(raw)
        manifests.append(ExpertManifest('fixture-model',0,expert,tuple(tensors)))
    manifests=tuple(manifests)
    bank=ExpertBank(f,manifests,model='fixture-model',expected_digests=tuple(m.digest for m in manifests))
    return bank,resident,path


def test_streamed_resident_bitwise_equality(provider,tmp_path):
    f,*_=provider; bank,resident,path=expert_fixture(f,tmp_path)
    x=np.arange(4,dtype='<f4')
    kw=dict(model='fixture-model',layer=0,route=(2,0),weights=(.75,.25),shared=(3,))
    expected=bank.execute(x,**kw,resident=resident)
    streamed=bank.execute(x,**kw)
    assert expected.tobytes()==streamed.tobytes()
    assert bank.staged_bytes==0 and bank.high_water<=bank.staging_budget
    f.evict()
    assert bank.execute(x,**kw).tobytes()==expected.tobytes()


def test_expert_identity_shape_content_and_missing(provider,tmp_path):
    f,*_=provider; bank,resident,path=expert_fixture(f,tmp_path)
    kw=dict(model='fixture-model',layer=0,route=(0,),weights=(1.0,))
    for change in [dict(model='wrong'),dict(layer=9),dict(route=(99,)),dict(route=(0,0),weights=(.5,.5))]:
        with pytest.raises(ContractError): bank.execute(np.ones(4),**dict(kw,**change))
    swapped=dict(resident); swapped[0,0]=resident[0,1]
    with pytest.raises(ContractError,match='INTEGRITY'): bank.execute(np.ones(4),**kw,resident=swapped)
    tensor=bank.manifest(0,0).tensors[0]
    for changed in [dict(dtype='F16'),dict(quantization='UNKNOWN'),dict(packing='COLUMN_MAJOR')]:
        with pytest.raises(ContractError): replace(tensor,**changed)
    with pytest.raises(ContractError):
        ExpertManifest('fixture-model',0,0,(replace(tensor,shape=(3,4)),)+bank.manifest(0,0).tensors[1:])
    with pytest.raises(ContractError):
        m=bank.manifest(0,0)
        ExpertBank(f,(m,m),model=m.model,expected_digests=(m.digest,m.digest))
    path.write_bytes(b'corrupt')
    with pytest.raises(ContractError,match='INTEGRITY'): bank.execute(np.ones(4),**kw)
    assert bank.staged_bytes==0 and f.stats()['pinned']==0
