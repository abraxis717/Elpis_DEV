import json
import os
import subprocess
import sys

from ._helpers import REPO, SRC_PATHS, digest_vector, scripted

PROBE = """
import json, sys
from pathlib import Path
sys.path[:0] = %r
from elpis.inference.drivers.dsv4.fixtures import make_fixture
from elpis.substrate.synthetic import SyntheticFileAssets
from elpis.inference.transaction import InferenceEngine
from tests.inference.steered._helpers import digest_vector, scripted
work, root, library = Path(sys.argv[1]), Path(sys.argv[2]), sys.argv[3]
provider = SyntheticFileAssets(root=root, library=library, warm_bytes=64, staging_bytes=128)
try:
    target, _, _ = make_fixture(provider, work / 'target')
    _, _, epochs = scripted(InferenceEngine(target), greedy_tokens=5, block_size=1)
    print(json.dumps(digest_vector(epochs), sort_keys=True))
finally:
    provider.close()
"""


def test_same_process_repeatability(make_rt):
    first, second = make_rt(), make_rt(name='second')
    _, _, a = scripted(first, greedy_tokens=5, block_size=1)
    _, _, b = scripted(second, greedy_tokens=5, block_size=1)
    _, _, c = scripted(first, greedy_tokens=5, block_size=1)
    assert digest_vector(a) == digest_vector(b) == digest_vector(c)


def test_fresh_process_hash_seed_determinism(make_rt, tmp_path, native_workspace, fms_file_library):
    _, _, epochs = scripted(make_rt(), greedy_tokens=5, block_size=1)
    expected = json.dumps(digest_vector(epochs), sort_keys=True)
    for seed in ('0', '717', '845813583'):
        work = tmp_path / ('seed-' + seed)
        work.mkdir()
        probe = subprocess.run([sys.executable, '-c', PROBE % (SRC_PATHS + [str(REPO)],), str(work),
                                str(native_workspace), str(fms_file_library)], capture_output=True,
                               text=True, env=dict(os.environ, PYTHONHASHSEED=seed, PYTHONDONTWRITEBYTECODE='1'))
        assert probe.returncode == 0, probe.stderr
        assert probe.stdout.strip() == expected, seed


def test_no_contamination_from_unrelated_preceding_request(make_rt):
    contaminated, clean = make_rt(), make_rt(name='clean')
    scripted(contaminated, request_id='req-Z', greedy_tokens=7, block_size=2)
    _, _, after = scripted(contaminated, greedy_tokens=5, block_size=1)
    _, _, alone = scripted(clean, greedy_tokens=5, block_size=1)
    assert digest_vector(after) == digest_vector(alone)
