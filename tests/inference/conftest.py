"""Inference fixtures: an explicitly loaded native file-asset provider and the DSV4 driver.

All fixtures are synthetic: no weights are shipped or downloaded. The native
library comes from the build tree (see tests/conftest.py); when
ELPIS_REQUIRE_NATIVE=1 a missing library is a failure, never a skip.
"""
from __future__ import annotations

import pytest

from elpis.inference.drivers.dsv4.fixtures import make_fixture
from elpis.inference.transaction import InferenceEngine
from elpis.substrate.file_assets import inspect_asset
from elpis.substrate.synthetic import SyntheticFileAssets


@pytest.fixture
def tmp_path(native_workspace):
    """Per-test scratch beneath the provider's trusted root.

    File assets resolve only beneath one root without following symlinks, so
    every fixture path in these tests must live inside that root.
    """
    path = native_workspace / "scratch"
    path.mkdir()
    return path


@pytest.fixture
def provider(native_workspace, fms_file_library):
    root = native_workspace
    f = SyntheticFileAssets(root=root, library=fms_file_library, warm_bytes=64, staging_bytes=128)
    path = root / "asset.dat"
    path.write_bytes(bytes(range(128)))
    m = inspect_asset(root, path, 16)
    asset = f.register(path, m, expected_manifest=m.digest)
    yield f, path, m, asset
    f.close()


@pytest.fixture
def target(provider, native_workspace):
    f, *_ = provider
    t, resident, meta = make_fixture(f, native_workspace / "target")
    return t, resident, meta


@pytest.fixture
def make_rt(native_workspace, fms_file_library):
    providers = []

    def build(dimension=4, name="target"):
        provider = SyntheticFileAssets(root=native_workspace, library=fms_file_library,
                                       warm_bytes=64, staging_bytes=128)
        providers.append(provider)
        target, _, _ = make_fixture(provider, native_workspace / name, dimension=dimension)
        return InferenceEngine(target)

    yield build
    for provider in providers:
        provider.close()


@pytest.fixture
def rt(make_rt):
    return make_rt()
