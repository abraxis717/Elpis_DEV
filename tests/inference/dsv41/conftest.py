import os
from pathlib import Path
import pytest

from elpis.inference.drivers.dsv41.fixtures import make_fixture, fixture_config
from elpis.inference.drivers.dsv41.engram import build_parameters
from elpis.inference.text import V41Tokenizer, RECIPE_SHA256
from elpis.substrate.boundary import RootCapability
from elpis.substrate.synthetic import SyntheticFileAssets


# Qualification mode: a skipped DSV4.1 test is never a pass. Both reference paths
# must be bound, and the tower, YTS-R0 provider-stream, Native Clock R0 and Native
# Materializer R1 suites must be collected; any skip becomes a failure.
QUALIFY = os.environ.get("ELPIS_DSV41_QUALIFY") == "1"
_HERE = Path(__file__).parent


def pytest_collection_modifyitems(session, config, items):
    if not QUALIFY:
        return
    for name in ("ELPIS_V41_TOKENIZER", "ELPIS_TOWER_DONORS"):
        if not os.environ.get(name):
            raise pytest.UsageError(f"ELPIS_DSV41_QUALIFY=1 requires {name}")
    ours = [i for i in items if Path(str(i.fspath)).parent == _HERE]
    counts = {name: sum(Path(str(i.fspath)).name == name for i in ours)
              for name in ("test_differential.py", "test_tower.py", "test_provider_stream.py",
                           "test_provider_stream_faults.py", "test_native_clock.py",
                           "test_native_materializer.py")}
    if not all(counts.values()):
        raise pytest.UsageError(f"ELPIS_DSV41_QUALIFY=1 collected {counts}; every suite is required")


@pytest.hookimpl(hookwrapper=True)
def pytest_runtest_makereport(item, call):
    outcome = yield
    report = outcome.get_result()
    if QUALIFY and report.skipped and Path(str(item.fspath)).parent == _HERE:
        report.outcome = "failed"
        report.longrepr = f"ELPIS_DSV41_QUALIFY=1: a skip is not qualification: {report.longrepr}"


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
