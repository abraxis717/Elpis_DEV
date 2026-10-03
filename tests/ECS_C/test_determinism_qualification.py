"""Independent fresh-process histories agree across hash seeds and restarts.

For every (PYTHONHASHSEED, restart cadence) pair the same scenario is built in
a fresh process and then replayed in another fresh process. Every build must
equal its replay, and all builds must produce identical canonical output
(state root, event bytes and full projection).
"""
from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import sys

import elpis

SCENARIO = Path(__file__).with_name("_determinism_scenario.py")
IMPORT_ROOT = str(Path(elpis.__file__).resolve().parents[1])


def _run(directory, *args, seed):
    env = dict(os.environ, PYTHONHASHSEED=seed, PYTHONPATH=IMPORT_ROOT, PYTHONDONTWRITEBYTECODE="1")
    out = subprocess.run([sys.executable, str(SCENARIO), str(directory), *args],
                         capture_output=True, text=True, env=env, check=True, timeout=120)
    return json.loads(out.stdout)


def test_independent_hash_seed_and_restart_histories(tmp_path):
    results = []
    for seed in ["0", "1", "42", "123456789", "random"]:
        for restart in [0, 1, 3, 7]:
            directory = tmp_path / f"history-{seed}-{restart}"
            live = _run(directory, "--restart-every", str(restart), seed=seed)
            fresh = _run(directory, "--replay", seed=seed)
            assert fresh == live
            results.append(live)
    assert len(results) == 20
    assert all(value == results[0] for value in results)
