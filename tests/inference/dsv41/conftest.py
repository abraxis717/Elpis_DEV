import os
from pathlib import Path
import pytest

from elpis.inference.drivers.dsv41.fixtures import make_fixture, fixture_config
from elpis.inference.drivers.dsv41.engram import build_parameters
from elpis.inference.text import V41Tokenizer, RECIPE_SHA256
from elpis.substrate.boundary import RootCapability
from elpis.substrate.synthetic import SyntheticFileAssets


@pytest.fixture(scope="session")
def v41():
    path = os.environ.get("ELPIS_V41_TOKENIZER")
    if not path:
        pytest.skip("ELPIS_V41_TOKENIZER required; unconfigured is not tower qualification")
    path = Path(path)
    with RootCapability(path.parent) as root:
        return V41Tokenizer.load(root, path.name, expected_sha256=RECIPE_SHA256)


@pytest.fixture(scope="session")
def tower_config(v41):
    return fixture_config(v41)


@pytest.fixture(scope="session")
def address_parameters(v41, tower_config):
    return build_parameters(v41, tower_config)


@pytest.fixture
def tower(native_workspace, fms_file_library, v41, tower_config, address_parameters):
    provider = SyntheticFileAssets(root=native_workspace, library=fms_file_library,
                                   warm_bytes=1 << 20, staging_bytes=1 << 16)
    target, meta = make_fixture(provider, native_workspace / "tower", v41,
                                config=tower_config, parameters=address_parameters)
    yield target, meta
    provider.close()
