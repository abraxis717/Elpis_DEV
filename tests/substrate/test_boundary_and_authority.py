"""Successor authority/path/native adversaries. All self-authorization is test-only."""
from dataclasses import asdict, replace
import ctypes
import errno
import hashlib
import json
import os
from pathlib import Path
import sys

import pytest

from elpis.substrate import boundary
from elpis.substrate.authority import AssetIdentity, NativeIdentity, PinnedAuthority
from elpis.substrate.contracts import Code, ContractError
from elpis.substrate.digests import identity, raw_digest
from elpis.substrate.file_assets import AssetManifest, FMSFileAssets, inspect_asset
from elpis.substrate.synthetic import SyntheticFileAssets

from ..conftest import require_native_library
APPROVED = b'externally approved asset bytes!!'


def manifest_for(data=APPROVED, page_size=16):
    pages = tuple(raw_digest(data[i:i+page_size]) for i in range(0, len(data), page_size))
    content = identity('file-asset.content-map', dict(size=len(data), page_size=page_size, pages=pages))
    return AssetManifest(len(data), page_size, content, pages)


def catalog(native=None, asset=None, provenance='deployment'):
    # Test vectors simulate a separately configured deployment catalog. Expected
    # asset identity is computed from APPROVED, never from the candidate file.
    m = manifest_for()
    asset = asset or AssetIdentity('weights', len(APPROVED), 16,
                                   hashlib.sha256(APPROVED).hexdigest(), m.digest)
    document = json.dumps(dict(schema='elpis.inference-authority.v1',
                               source='test deployment authority vector v1', provenance=provenance,
                               assets=[asdict(asset)],
                               libraries=[] if native is None else [asdict(native)])).encode()
    return PinnedAuthority(document, expected_sha256=hashlib.sha256(document).hexdigest())


@pytest.fixture
def secure_root(tmp_path):
    try:
        with boundary.RootCapability(tmp_path):
            pass
    except ContractError as exc:
        if exc.code == Code.UNSUPPORTED:
            pytest.skip('strong Linux boundary unavailable: ' + str(exc))
        raise
    return tmp_path


@pytest.fixture
def compiled_native():
    if sys.platform != 'linux':
        pytest.skip('Linux native qualification unavailable')
    return require_native_library('elpis_fms_file_assets').read_bytes()


@pytest.fixture
def setup_native(secure_root, compiled_native):
    library = secure_root/'provider.so'
    library.write_bytes(compiled_native)
    # Build output is explicitly self-authorized for this synthetic test fixture.
    native = NativeIdentity('provider', len(compiled_native), hashlib.sha256(compiled_native).hexdigest())
    return secure_root, library, native


def provider_for(setup, authority=None):
    root, library, native = setup
    return FMSFileAssets(root=root, library=library, authority=authority or catalog(native),
                         library_id='provider', warm_bytes=64, staging_bytes=128)


def test_catalog_independent_pin_and_immutable_records():
    with pytest.raises(ContractError, match='independent catalog pin'):
        PinnedAuthority(b'{}', expected_sha256='0'*64)
    authority = catalog()
    with pytest.raises(TypeError):
        authority.assets['forged'] = authority.assets['weights']
    with pytest.raises(AttributeError):
        authority.source = 'attacker'


@pytest.mark.parametrize('document', [b'{"schema":1,"schema":2}', b'null', b'[]', b'not json'])
def test_malformed_catalog_fails_closed(document):
    with pytest.raises(ContractError):
        PinnedAuthority(document, expected_sha256=hashlib.sha256(document).hexdigest())


def test_valid_asset_and_native_provider(setup_native):
    root, _, _ = setup_native
    candidate = root/'weights.dat'
    candidate.write_bytes(APPROVED)
    with provider_for(setup_native) as provider:
        asset = provider.register(candidate, manifest_for(), asset_id='weights')
        with provider.acquire(asset, 3, 20) as lease:
            assert lease.read() == APPROVED[3:23]
        assert provider._native.lib._name.startswith('/proc/self/fd/')
        assert provider.stats()['native_storage'] == 0


@pytest.mark.parametrize('mode', ['wrong-initial', 'swap-after-inspection', 'forged-self-manifest', 'wrong-raw-pin'])
def test_independent_asset_identity_rejects_candidates(setup_native, mode):
    root, _, native = setup_native
    candidate = root/'weights.dat'
    candidate.write_bytes(APPROVED)
    observation = inspect_asset(root, candidate, 16)
    authority = catalog(native)
    if mode == 'wrong-raw-pin':
        authority = catalog(native, replace(authority.assets['weights'], sha256='0'*64))
    else:
        if mode == 'swap-after-inspection':
            candidate.rename(root/'previous-weights.dat')
        candidate.write_bytes(b'X'*len(APPROVED))
        if mode == 'forged-self-manifest':
            observation = inspect_asset(root, candidate, 16)
        elif mode == 'wrong-initial':
            observation = manifest_for()
    with provider_for(setup_native, authority) as provider:
        with pytest.raises(ContractError, match='IDENTITY|INTEGRITY'):
            provider.register(candidate, observation, asset_id='weights', expected_manifest=observation.digest)
        assert not provider._assets and provider.stats()['pages'] == 0


def test_old_self_manifest_pattern_and_synthetic_authority_rejected(setup_native):
    root, library, native = setup_native
    candidate = root/'weights.dat'
    candidate.write_bytes(APPROVED)
    m = inspect_asset(root, candidate, 16)
    with provider_for(setup_native) as provider:
        with pytest.raises(ContractError, match='authorized asset identifier'):
            provider.register(candidate, m, expected_manifest=m.digest)
    for authority in (None, catalog(native, provenance='synthetic-test'), m):
        with pytest.raises(ContractError, match='deployment authority'):
            FMSFileAssets(root=root, library=library, authority=authority, library_id='provider')


@pytest.mark.parametrize('kind', ['final-link', 'intermediate-link', 'traversal', 'absolute-outside', 'fifo'])
def test_path_adversaries(secure_root, kind):
    root = secure_root/'trusted'
    root.mkdir()
    outside = secure_root/'outside'
    outside.mkdir()
    (outside/'asset').write_bytes(APPROVED)
    if kind == 'final-link':
        (root/'asset').symlink_to(outside/'asset')
        path = 'asset'
    elif kind == 'intermediate-link':
        (root/'linked').symlink_to(outside, target_is_directory=True)
        path = 'linked/asset'
    elif kind == 'traversal':
        path = '../outside/asset'
    elif kind == 'absolute-outside':
        path = outside/'asset'
    else:
        os.mkfifo(root/'pipe')
        path = 'pipe'
    with boundary.RootCapability(root) as cap:
        with pytest.raises(ContractError):
            cap.open_file(path)
    with pytest.raises(ContractError):
        inspect_asset(root, path, 16)


def test_root_symlink_rejected(secure_root):
    (secure_root/'alias').symlink_to(secure_root, target_is_directory=True)
    with pytest.raises(ContractError):
        boundary.RootCapability(secure_root/'alias')


def test_root_rename_does_not_redirect_capability(secure_root):
    root = secure_root/'root'
    root.mkdir()
    (root/'asset').write_bytes(APPROVED)
    with boundary.RootCapability(root) as cap:
        root.rename(secure_root/'old-root')
        root.mkdir()
        (root/'asset').write_bytes(b'forged')
        fd = cap.open_file('asset')
        try:
            assert os.pread(fd, len(APPROVED), 0) == APPROVED
        finally:
            os.close(fd)


def test_intermediate_swap_at_open_rejected(secure_root, monkeypatch):
    (secure_root/'dir').mkdir()
    (secure_root/'dir'/'asset').write_bytes(APPROVED)
    with boundary.RootCapability(secure_root) as cap:
        real = boundary._openat2
        def swapped(fd, path, flags):
            (secure_root/'dir').rename(secure_root/'moved')
            (secure_root/'dir').symlink_to(secure_root/'moved', target_is_directory=True)
            return real(fd, path, flags)
        monkeypatch.setattr(boundary, '_openat2', swapped)
        with pytest.raises(ContractError):
            cap.open_file('dir/asset')


def test_asset_rename_after_admission_cannot_expose_replacement(setup_native):
    root, _, _ = setup_native
    path = root/'asset'
    path.write_bytes(APPROVED)
    with provider_for(setup_native) as provider:
        asset = provider.register(path, manifest_for(), asset_id='weights')
        path.rename(root/'original')
        path.write_bytes(b'X'*len(APPROVED))
        try:
            with provider.acquire(asset, 0, 16) as lease:
                assert lease.read() == APPROVED[:16]
        except ContractError as exc:
            assert exc.code == Code.INTEGRITY  # rename may change retained inode ctime


def test_duplicate_admission_revalidates_path(setup_native):
    root, _, _ = setup_native
    path = root/'asset'
    path.write_bytes(APPROVED)
    with provider_for(setup_native) as provider:
        provider.register(path, manifest_for(), asset_id='weights')
        path.unlink()
        path.symlink_to(root/'provider.so')
        with pytest.raises(ContractError):
            provider.register(path, manifest_for(), asset_id='weights')


def test_wrong_native_digest_never_reaches_dlopen(setup_native, monkeypatch):
    _, _, native = setup_native
    real = ctypes.CDLL
    def guarded(path, *args, **kwargs):
        assert path is None, 'unverified executable reached dlopen'
        return real(path, *args, **kwargs)
    monkeypatch.setattr(ctypes, 'CDLL', guarded)
    with pytest.raises(ContractError, match='native SHA-256'):
        provider_for(setup_native, catalog(replace(native, sha256='0'*64)))


def test_native_substitution_before_open_rejected(setup_native):
    _, library, _ = setup_native
    library.write_bytes(b'X'*library.stat().st_size)
    with pytest.raises(ContractError, match='native SHA-256'):
        provider_for(setup_native)


@pytest.mark.parametrize('mutation', ['rename', 'in-place'])
def test_native_substitution_at_dlopen_uses_sealed_bytes(setup_native, monkeypatch, mutation):
    import fcntl
    root, library, native = setup_native
    real = ctypes.CDLL
    # Keep existing process-lifetime capabilities alive; isolate this test's
    # cache so the actual dlopen boundary is exercised for each attack.
    monkeypatch.setattr(boundary, '_LOADED_NATIVE_OBJECTS', {})
    seen = []
    def substituted(path, *args, **kwargs):
        if path is not None:
            assert path.startswith('/proc/self/fd/')
            fd = int(path.rsplit('/', 1)[1])
            assert fcntl.fcntl(fd, fcntl.F_GET_SEALS) & fcntl.F_SEAL_WRITE
            with pytest.raises(OSError):
                os.pwrite(fd, b'X', 0)
            if mutation == 'rename':
                library.rename(root/'original.so')
            library.write_bytes(b'X'*native.size)
            seen.append(path)
        return real(path, *args, **kwargs)
    monkeypatch.setattr(ctypes, 'CDLL', substituted)
    with provider_for(setup_native) as provider:
        assert provider.stats()['pages'] == 0
    assert seen


def test_native_symlink_rejected(setup_native):
    root, library, _ = setup_native
    library.rename(root/'real.so')
    library.symlink_to(root/'real.so')
    with pytest.raises(ContractError):
        provider_for(setup_native)


def test_explicit_synthetic_fixture_is_distinct(setup_native):
    root, library, _ = setup_native
    path = root/'fixture'
    path.write_bytes(b'self authorized fixture')
    with SyntheticFileAssets(root=root, library=library) as provider:
        m = inspect_asset(root, path, 16)
        asset = provider.register(path, m, expected_manifest=m.digest)
        with provider.acquire(asset, 0, 4) as lease:
            assert lease.read() == b'self'
        assert provider.authority.provenance == 'synthetic-test'


def test_non_linux_fails_closed_without_path_open(tmp_path, monkeypatch):
    monkeypatch.setattr(boundary.sys, 'platform', 'darwin')
    monkeypatch.setattr(os, 'open', lambda *a, **k: pytest.fail('pathname fallback'))
    with pytest.raises(ContractError, match='UNSUPPORTED'):
        boundary.RootCapability(tmp_path)


@pytest.mark.parametrize('error', [errno.ENOSYS, errno.EINVAL, errno.EPERM, errno.EAGAIN, errno.ELOOP, errno.EXDEV])
def test_openat2_errors_never_downgrade(secure_root, monkeypatch, error):
    class MissingSyscall:
        restype = None
        def __call__(self, *args):
            ctypes.set_errno(error)
            return -1
    class FakeLibc:
        syscall = MissingSyscall()
    monkeypatch.setattr(ctypes, 'CDLL', lambda *a, **k: FakeLibc())
    with pytest.raises(ContractError) as caught:
        boundary.RootCapability(secure_root)
    assert caught.value.code == (Code.UNSUPPORTED if error in (errno.ENOSYS, errno.EINVAL, errno.EPERM) else Code.IO)


def test_repeated_native_contexts_share_only_verified_code(setup_native):
    with provider_for(setup_native) as first, provider_for(setup_native) as second:
        assert first._native.lib is second._native.lib
        assert first._native.ctx.value != second._native.ctx.value
    _, library, _ = setup_native
    library.write_bytes(b'X'*library.stat().st_size)
    with pytest.raises(ContractError, match='native SHA-256'):
        provider_for(setup_native)


def test_asset_swap_after_open_is_descriptor_bound(setup_native, monkeypatch):
    root, _, _ = setup_native
    path = root/'asset'
    path.write_bytes(APPROVED)
    with provider_for(setup_native) as provider:
        real = provider._boundary.open_file
        def swapped(candidate):
            fd = real(candidate)
            path.rename(root/'original')
            path.write_bytes(b'X'*len(APPROVED))
            return fd
        monkeypatch.setattr(provider._boundary, 'open_file', swapped)
        asset = provider.register(path, manifest_for(), asset_id='weights')
        with provider.acquire(asset, 0, len(APPROVED)) as lease:
            assert lease.read() == APPROVED


def test_native_memfd_unavailable_fails_closed(setup_native, monkeypatch):
    def unavailable(*args):
        raise OSError(errno.ENOSYS, 'not supported')
    monkeypatch.setattr(os, 'memfd_create', unavailable)
    with pytest.raises(ContractError, match='UNSUPPORTED'):
        provider_for(setup_native)


def test_native_sealing_failure_never_loads(setup_native, monkeypatch):
    import fcntl
    real = ctypes.CDLL
    def no_seals(*args):
        raise OSError(errno.EPERM, 'seal denied')
    def guarded(path, *args, **kwargs):
        assert path is None
        return real(path, *args, **kwargs)
    monkeypatch.setattr(fcntl, 'fcntl', no_seals)
    monkeypatch.setattr(ctypes, 'CDLL', guarded)
    with pytest.raises(ContractError, match='UNSUPPORTED'):
        provider_for(setup_native)


def test_failed_admission_closes_descriptor_and_root_close_is_final(setup_native, monkeypatch):
    root, _, _ = setup_native
    path = root/'asset'
    path.write_bytes(b'X'*len(APPROVED))
    with provider_for(setup_native) as provider:
        real = provider._boundary.open_file
        opened = []
        def capture(candidate):
            fd = real(candidate)
            opened.append(fd)
            return fd
        monkeypatch.setattr(provider._boundary, 'open_file', capture)
        with pytest.raises(ContractError):
            provider.register(path, manifest_for(), asset_id='weights')
        with pytest.raises(OSError):
            os.fstat(opened[-1])
    with pytest.raises(ContractError, match='CLOSED'):
        provider._boundary.open_file(path)


def test_asset_mutation_during_admission_rejected(setup_native, monkeypatch):
    root, _, _ = setup_native
    path = root/'asset'
    path.write_bytes(APPROVED)
    with provider_for(setup_native) as provider:
        real = os.pread
        changed = []
        def mutating(fd, count, offset):
            data = real(fd, count, offset)
            if not changed:
                path.write_bytes(b'X'*len(APPROVED))
                changed.append(True)
            return data
        monkeypatch.setattr(os, 'pread', mutating)
        with pytest.raises(ContractError, match='INTEGRITY'):
            provider.register(path, manifest_for(), asset_id='weights')
        assert not provider._assets


def test_nul_root_is_rejected_before_syscall(tmp_path, monkeypatch):
    monkeypatch.setattr(os, 'open', lambda *a, **k: pytest.fail('invalid root reached OS'))
    with pytest.raises(ContractError, match='absolute trusted root'):
        boundary.RootCapability(str(tmp_path) + '\x00/ignored')


def test_geometry_checked_before_admission_reads(setup_native, monkeypatch):
    root, _, _ = setup_native
    path = root/'asset'
    path.write_bytes(APPROVED + b'unapproved extension')
    with provider_for(setup_native) as provider:
        monkeypatch.setattr(os, 'pread', lambda *a: pytest.fail('wrong-size asset read'))
        with pytest.raises(ContractError, match='asset geometry'):
            provider.register(path, manifest_for(), asset_id='weights')


@pytest.mark.parametrize('invalid_call',[1,2])
def test_invalid_native_tier_releases_current_and_prior_leases(setup_native,monkeypatch,invalid_call):
    root, _, _ = setup_native
    path = root/'asset-tier'
    path.write_bytes(APPROVED)
    with provider_for(setup_native) as provider:
        asset = provider.register(path, manifest_for(), asset_id='weights')
        real = provider._native.acquire
        calls = 0
        def invalid_once(oid,tier):
            nonlocal calls
            calls += 1
            lease,ptr,actual = real(oid,tier)
            return lease,ptr,(7 if calls == invalid_call else actual)
        monkeypatch.setattr(provider._native,'acquire',invalid_once)
        with pytest.raises(ContractError,match='native lease tier'):
            provider.acquire(asset,0,20)
        assert provider.stats()['pinned'] == 0
        assert all(value == 0 for value in provider._pins.values())
