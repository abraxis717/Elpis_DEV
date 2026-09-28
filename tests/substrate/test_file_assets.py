from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
import os
from pathlib import Path
import pytest
from elpis.substrate.file_assets import inspect_asset
from elpis.substrate.synthetic import SyntheticFileAssets as FMSFileAssets
from elpis.substrate.contracts import ContractError


@pytest.fixture
def provider(native_workspace, fms_file_library):
    root=native_workspace
    f=FMSFileAssets(root=root,library=fms_file_library,warm_bytes=64,staging_bytes=128)
    path=root/'asset.dat'; path.write_bytes(bytes(range(128)))
    m=inspect_asset(root,path,16)
    asset=f.register(path,m,expected_manifest=m.digest)
    yield f,path,m,asset
    f.close()


def test_ranges_cache_accounting_and_leases(provider):
    f,path,m,a=provider
    with f.acquire(a,13,29,tier='HOT') as lease:
        assert lease.actual_tier=='WARM'
        assert lease.read()==bytes(range(13,42))
        with pytest.raises(ContractError,match='BUSY'): f.evict(a)
        with pytest.raises(ContractError,match='UNSUPPORTED'): lease.bind_fence(object())
    with pytest.raises(ContractError,match='CLOSED'): lease.read()
    with pytest.raises(ContractError,match='CLOSED'): lease.release()
    before=f.stats()['pread_bytes']
    with f.acquire(a,15,1) as lease: assert lease.read()==b'\x0f'
    assert f.stats()['pread_bytes']==before
    stats=f.stats()
    assert stats['warm']<=64 and stats['ram']<=64 and stats['external_cold']==128
    assert stats['native_storage']==0 and stats['pinned']==0
    assert stats['staging_high_water']<=128 and not stats['buffered_page_cache_accounted']
    f.evict()
    assert f.stats()['pages']==0 and path.read_bytes()==bytes(range(128))


def test_changed_truncated_corrupt_metadata(provider):
    f,path,m,a=provider
    with pytest.raises(ContractError,match='IDENTITY'):
        f.register(path,replace(m,content='0'*64),expected_manifest=m.digest)
    with pytest.raises(ContractError,match='IDENTITY'):
        f.register(path,replace(m,pages=('0'*64,)+m.pages[1:]),expected_manifest=m.digest)
    path.write_bytes(b'broken')
    with pytest.raises(ContractError,match='INTEGRITY'): f.acquire(a,0,1)


def test_wrong_page_map_cannot_expose_bytes(provider,tmp_path):
    f,path,m,a=provider
    bad=replace(m,pages=('0'*64,)+m.pages[1:])
    with pytest.raises(ContractError,match='INTEGRITY'):
        f.register(path,bad,expected_manifest=bad.digest)
    assert f.stats()['pinned']==0 and f.stats()['pages']==0


def test_reads_retry_interrupt_short_fail_closed(provider,monkeypatch):
    f,path,m,a=provider
    real=os.pread
    calls=[]
    def interrupted(fd,n,offset):
        calls.append(offset)
        if len(calls)==1: raise InterruptedError()
        return real(fd,min(n,3),offset)
    monkeypatch.setattr(os,'pread',interrupted)
    with f.acquire(a,0,16) as lease: assert lease.read()==bytes(range(16))
    assert len(calls)>2
    f.evict()
    monkeypatch.setattr(os,'pread',lambda *a: b'')
    with pytest.raises(ContractError,match='IO'): f.acquire(a,0,16)
    assert f.stats()['pinned']==0


def test_concurrent_acquire_and_capacity(provider):
    f,path,m,a=provider
    def read(i):
        for _ in range(10):
            with f.acquire(a,i,1) as lease:
                assert lease.read()==bytes([i])
    with ThreadPoolExecutor(max_workers=4) as pool:
        list(pool.map(read,[1,17,65,113]))
    leases=[f.acquire(a,i,1) for i in (1,17,33,49)]
    try:
        with pytest.raises(ContractError,match='LIMIT'): f.acquire(a,65,1)
    finally:
        for lease in leases: lease.release()
    assert f.stats()['pinned']==0


def test_absent_assets_invalid_ranges_and_cleanup(provider):
    f,path,m,a=provider
    with pytest.raises(ContractError,match='MISSING'): f.acquire('0'*64,0,1)
    for offset,length in [(-1,1),(128,1),(0,0),(True,1)]:
        with pytest.raises(ContractError): f.acquire(a,offset,length)
    with pytest.raises(ContractError,match='LIMIT'): f.acquire(a,0,128)
    with pytest.raises(RuntimeError):
        with f.acquire(a,0,1): raise RuntimeError('caller failure')
    assert f.stats()['pinned']==0


def test_hot_reject_and_storage_limits(provider,fms_file_library):
    f,path,m,a=provider
    with FMSFileAssets(root=f.root,library=fms_file_library,warm_bytes=64,
                       storage_bytes=128,hot_absent_policy='REJECT') as reject:
        a=reject.register(path,m,expected_manifest=m.digest)
        with pytest.raises(ContractError,match='UNSUPPORTED'): reject.acquire(a,0,1,tier='HOT')
        assert reject.stats()['pinned']==0
        other=replace(m,content='0'*64)
        with pytest.raises(ContractError,match='LIMIT'): reject.register(path,other,expected_manifest=other.digest)
