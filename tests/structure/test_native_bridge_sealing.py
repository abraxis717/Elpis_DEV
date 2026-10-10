"""Real bridge ELF security tests. No monkeypatched digest decisions.

Failure paths are observed as specific fail-closed outcomes. The actual
substrate Linux openat2 + sealed memfd implementation is exercised.
"""
from __future__ import annotations

import hashlib
from pathlib import Path

import pytest

from elpis.pipeline.ingress import IngressError, IngressLibrary
from elpis.structure.retrieval.errors import RetrievalLibraryError
from elpis.structure.retrieval.hacf import RetrievalLibrary
from elpis.structure.native_loader import load_structure_bridge
from elpis.substrate.authority import PinnedAuthority
from elpis.substrate.contracts import ContractError
from .native_bridge_fixture import pin_bridge
from ..conftest import require_native_library


@pytest.fixture(params=[('elpis_retrieval_bridge', RetrievalLibrary, RetrievalLibraryError),
                        ('elpis_ingress_bridge', IngressLibrary, IngressError)])
def specimen(request, tmp_path):
    library_id, constructor, error_type = request.param
    target, root, authority = pin_bridge(require_native_library(library_id), tmp_path, library_id)
    return target, root, authority, constructor, error_type, library_id


def _abi_export(wrapper):
    """Observed ABI function, not a fabricated common wrapper property.

    ctypes caches the resolved function on its CDLL instance. Identity of this
    actual export also tests that verified equal binaries reuse the sealed ELF.
    """
    if isinstance(wrapper, RetrievalLibrary):
        return wrapper.lib.elpis_retrieval_bridge_abi_version, 2
    if isinstance(wrapper, IngressLibrary):
        return wrapper.bridge_abi, 1
    raise TypeError("unknown bridge wrapper")


def test_correct_pin_loads_real_bridge(specimen):
    path, root, authority, constructor, _, _ = specimen
    lib = constructor(path, root=root, authority=authority)
    abi_export, expected_abi = _abi_export(lib)
    assert abi_export() == expected_abi


def test_unpinned_path_can_never_load(specimen):
    path, _, _, constructor, error_type, _ = specimen
    with pytest.raises(error_type) as caught:
        constructor(path)
    assert caught.value.code == 'LIB_AUTHORITY'


def test_missing_root_refused(specimen):
    path, _, authority, constructor, error_type, _ = specimen
    with pytest.raises(error_type) as caught:
        constructor(path, authority=authority)
    assert caught.value.code == 'LIB_AUTHORITY'


def test_wrong_digest_refused(specimen, tmp_path):
    path, root, _, constructor, error_type, library_id = specimen
    _, _, wrong = pin_bridge(path, tmp_path / 'wrong-digest', library_id, sha_override='0' * 64)
    with pytest.raises(error_type) as caught:
        constructor(path, root=root, authority=wrong)
    assert caught.value.code == 'LIB_AUTHORITY'
    assert 'native SHA-256' in str(caught.value)


def test_wrong_library_identity_refused(specimen, tmp_path):
    path, root, _, constructor, error_type, library_id = specimen
    wrong_id = 'elpis_ingress_bridge' if library_id == 'elpis_retrieval_bridge' else 'elpis_retrieval_bridge'
    _, _, wrong = pin_bridge(path, tmp_path / 'wrong-id', wrong_id)
    with pytest.raises(error_type) as caught:
        constructor(path, root=root, authority=wrong)
    assert caught.value.code == 'LIB_AUTHORITY'
    assert 'not in deployment authority' in str(caught.value)


def test_symlink_and_path_escape_refused(specimen, tmp_path):
    path, root, authority, constructor, error_type, _ = specimen
    alias = root / 'symlink-bridge.so'
    alias.symlink_to(path)
    with pytest.raises(error_type) as caught:
        constructor(alias, root=root, authority=authority)
    assert caught.value.code == 'LIB_AUTHORITY'
    with pytest.raises(error_type) as caught:
        constructor(path, root=tmp_path / 'other-root', authority=authority)
    assert caught.value.code == 'LIB_AUTHORITY'


def test_replacement_must_be_reverified_even_with_cached_code(specimen):
    path, root, authority, constructor, error_type, _ = specimen
    before = constructor(path, root=root, authority=authority)
    abi_export, expected_abi = _abi_export(before)
    assert abi_export() == expected_abi
    path.write_bytes(b'tampered ELF content')
    with pytest.raises(error_type) as caught:
        constructor(path, root=root, authority=authority)
    assert caught.value.code == 'LIB_AUTHORITY'


def test_same_verified_bytes_share_sealed_code(specimen):
    path, root, authority, constructor, _, _ = specimen
    first = constructor(path, root=root, authority=authority)
    second = constructor(path, root=root, authority=authority)
    first_export, expected_abi = _abi_export(first)
    second_export, repeated_abi = _abi_export(second)
    assert expected_abi == repeated_abi
    assert first_export() == second_export() == expected_abi
    assert first_export is second_export


def test_deployment_only_provenance(specimen):
    path, root, authority, constructor, error_type, _ = specimen
    import json
    from dataclasses import asdict
    catalog = json.dumps({'schema': authority.schema, 'source': 'synthetic testing',
                          'provenance': 'synthetic-test', 'assets': [],
                          'libraries': [asdict(x) for x in authority.libraries.values()]},
                         sort_keys=True, separators=(',', ':')).encode()
    synthetic = PinnedAuthority(catalog, expected_sha256=hashlib.sha256(catalog).hexdigest())
    with pytest.raises(error_type) as caught:
        constructor(path, root=root, authority=synthetic)
    assert caught.value.code == 'LIB_AUTHORITY'


def test_wrong_bridge_abi_refused(tmp_path):
    # Correct independent byte pin, but wrong exported symbol family.
    wrong = require_native_library('elpis_ingress_bridge')
    path, root, authority = pin_bridge(wrong, tmp_path, 'elpis_retrieval_bridge')
    with pytest.raises(RetrievalLibraryError) as caught:
        RetrievalLibrary(path, root=root, authority=authority)
    assert caught.value.code == 'ABI_MISMATCH'


def test_wrong_ingress_bridge_abi_refused(tmp_path):
    # The foreign bridge has a valid ELF pin but no ingress ABI exports.
    wrong = require_native_library('elpis_retrieval_bridge')
    path, root, authority = pin_bridge(wrong, tmp_path, 'elpis_ingress_bridge')
    with pytest.raises(IngressError) as caught:
        IngressLibrary(path, root=root, authority=authority)
    assert caught.value.code == 'ABI_MISMATCH'
