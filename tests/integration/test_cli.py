"""The operator interface (``python -m elpis.runtime.cli``) over the build's libraries, as separate processes.

RAM-only (RuntimeCore ABI v8): the libraries are read, read-only, and admitted only against the operator's catalog pin;
there is no home and nothing is written.

* ``catalog`` measures the libraries read-only and prints their catalog and its SHA-256 (the pin);
* a session founds a source, learns numerically, queries, records typed HACF evidence (observe-only: no K1 learning)
  and looks it up, all in memory; ``close`` discards the session and the next starts at the lineage's initial
  identity; a NEW process starts there too (nothing is resumed: byte-identical first QUERY answers, no LOOKUP answer);
* refusals: every retired persistent command (init, create, publish, candidates, reconcile, --home, --publish); a
  changed library or a wrong pin (pin drift) refuses every start; a missing pin; unauthorized LEARN and evidence; a
  script whose declared expectation is not met exits non-zero;
* no process writes anything (the working directory stays empty).

Every output is machine-readable JSON. Mechanics only: the evidence and stimuli are fixtures.
"""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

import pytest

from ..conftest import native_build_dir, require_native_library

REPO = Path(__file__).resolve().parents[2]
_LIBS = ("native/runtime/libelpis_runtime.so", "native/runtime/libelpis_hacf_observation.so",
         "native/ECS_G/libelpis_ecsg_math.so", "native/ECS_G/libelpis_ecsg_k1.so",
         "native/ECS_G/libelpis_ecsg_k1_fms.so", "native/ECS_C/libelpis_ecs_memory.so",
         "native/runtime/composite/libelpis_ecs_composite.so")


def node(name):
    return hashlib.sha256(b"cli-fixture-node:" + name.encode()).hexdigest()


def prov(name):
    return hashlib.sha256(b"cli-fixture-provenance:" + name.encode()).hexdigest()


X = [[((i * 7 + j * 3) % 11 - 5) / 16.0 for j in range(6)] for i in range(4)]
LEARN = {"op": "learn", "x": X, "y": [0.1, -0.2, 0.05, 0.3], "schedule": [[2, 3], [2, 5]], "rate": 0.002,
         "grant": "test operator: fixture numeric experience"}
EDGE = {"relation": "hyperedge:1", "participants": [{"node": node("alice"), "role": "incidence:1"},
                                                    {"node": node("acme"), "role": "incidence:2"},
                                                    {"node": node("engineer"), "role": "incidence:3"}]}
# The operator founds source 1 in the session (the native host's op): typed evidence is admitted only from an ACTIVE
# source of the session.
SOURCE = {"op": "source_found", "label": "fixture source 1", "source_authority": 1, "per_turn": 8,
          "scope_label": "fixture scope"}
EVIDENCE = {"op": "evidence", "items": [dict(EDGE, provenance=prov("record-17"), authority=1, polarity="ASSERT",
                                             tick=1)], "grant": "test operator: source 1", "sources": [1]}
LOOKUP = {"op": "lookup", "relation": "hyperedge:1", "target_role": "incidence:3",
          "cues": [{"role": "incidence:1", "node": node("alice")}, {"role": "incidence:2", "node": node("acme")}]}


@pytest.fixture(scope="module")
def build():
    for name in ("elpis_runtime", "elpis_ecs_composite", "elpis_hacf_observation", "elpis_ecsg_k1_fms"):
        require_native_library(name)
    return native_build_dir()


def cli(source, *args, script=None, pin=None, cwd="/"):
    env = dict(os.environ, PYTHONPATH=str(REPO / "src"), PYTHONDONTWRITEBYTECODE="1")
    cmd = [sys.executable, "-m", "elpis.runtime.cli", "--native-build", str(source)]
    if pin is not None:
        cmd += ["--catalog-sha256", pin]
    cmd += ["--seed", "7", *args]
    if script is not None:
        cmd += ["session", "--script", "-"]
    run = subprocess.run(cmd, input=json.dumps(script) if script is not None else None, capture_output=True,
                         text=True, env=env, cwd=cwd, timeout=300)
    out = [json.loads(line) for line in run.stdout.splitlines() if line.strip()]
    assert run.stderr == "" or run.returncode != 0, run.stderr
    return run.returncode, out


def _pin(build):
    rc, out = cli(build, "catalog")
    assert rc == 0 and out[0]["ok"], out
    return out[0]["catalog_sha256"]


def _copy_build(build, root, flip=None):
    for rel in _LIBS:
        (root / rel).parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(build / rel, root / rel)
    if flip is not None:
        data = bytearray((root / flip).read_bytes())
        data[-1] ^= 1
        (root / flip).write_bytes(bytes(data))
    return root


def test_the_catalog_measures_the_libraries_read_only(build, tmp_path):
    before = {r: hashlib.sha256((build / r).read_bytes()).hexdigest() for r in _LIBS}
    rc, out = cli(build, "catalog", cwd=tmp_path)
    assert rc == 0 and out[0]["ok"] and len(out[0]["catalog_sha256"]) == 64
    pinned = {e["library_id"]: e["sha256"] for e in out[0]["catalog"]["libraries"]}
    assert pinned == {Path(r).name[3:-3]: h for r, h in before.items()}
    assert list(tmp_path.iterdir()) == []                                  # nothing written
    assert {r: hashlib.sha256((build / r).read_bytes()).hexdigest() for r in _LIBS} == before


def test_a_session_works_in_memory_and_nothing_survives_it(build, tmp_path):
    pin = _pin(build)
    rc, out = cli(build, pin=pin, cwd=tmp_path, script=[
        {"op": "open"}, SOURCE, {"op": "query", "x": X}, LEARN, {"op": "query", "x": X}, EVIDENCE, LOOKUP,
        {"op": "live"}, {"op": "status"}, {"op": "close"}, {"op": "live"}, LOOKUP, {"op": "query", "x": X}])
    assert rc == 0 and all(o["ok"] for o in out), out
    opened, source, before, learned, after, evidence, found, live, status, closed, fresh, lost, again = out
    initial = opened["initial_identity"]
    assert opened["anchored"]["identity"] == initial and opened["anchored"]["state_format"] == "COMPOSITE"
    assert source["lifecycle"] == "ACTIVE" and source["source_authority"] == 1
    assert learned["durability"] == "VOLATILE" and after["values"] != before["values"]
    assert evidence["durability"] == "VOLATILE" and evidence["k1_steps"] == 0
    assert found["found"] and found["answer"] == node("engineer")
    assert live["live"]["identity"] != initial
    assert status["installation"]["persistence"] == "NONE" and status["abi"] == {"runtime": 8, "composite": 1}
    assert closed["closed"] and closed["discarded"]["learned"]
    assert fresh["live"]["identity"] == initial and not lost["found"]
    assert again["values_sha256"] == before["values_sha256"]               # the initial state answers again
    # A new process: the same initial identity and answers; nothing was kept to resume.
    rc, out = cli(build, pin=pin, cwd=tmp_path, script=[{"op": "live"}, {"op": "query", "x": X}, LOOKUP])
    assert rc == 0 and out[0]["live"]["identity"] == initial
    assert out[1]["values_sha256"] == before["values_sha256"] and not out[2]["found"]
    assert list(tmp_path.iterdir()) == []                                  # no process wrote anything


@pytest.mark.parametrize("args", [("init",), ("create",), ("publish",), ("candidates",), ("reconcile",),
                                  ("--home", "/nowhere", "status"), ("learn", "--publish")])
def test_every_retired_persistent_command_is_refused(build, tmp_path, args):
    rc, out = cli(build, *args, pin=_pin(build), cwd=tmp_path)
    assert (rc, out[0]["code"]) == (2, "OPERATION_RETIRED"), out
    assert list(tmp_path.iterdir()) == []


def test_retired_operations_in_a_script_are_refused_and_unmet_expectations_fail(build, tmp_path):
    pin = _pin(build)
    rc, out = cli(build, pin=pin, script=[{"op": "publish", "expect": "OPERATION_RETIRED"},
                                          {"op": "candidates", "expect": "OPERATION_RETIRED"},
                                          {"op": "reconcile", "expect": "OPERATION_RETIRED"}])
    assert rc == 0 and all(o["met"] for o in out), out
    rc, out = cli(build, pin=pin, script=[dict(LEARN, grant=None, expect="LEARN_UNAUTHORIZED"),
                                          dict(EVIDENCE, grant=None, expect="OBSERVATION_UNAUTHORIZED"),
                                          SOURCE, dict(EVIDENCE, sources=[2], expect="OBSERVATION_AUTHORITY")])
    assert rc == 0 and [o["met"] for o in out if "met" in o] == [True, True, True], out
    rc, out = cli(build, pin=pin, script=[dict(LEARN, expect="LEARN_UNAUTHORIZED")])   # it succeeds: unmet
    assert rc == 3 and out[0]["ok"] and not out[0]["met"]


def test_pin_drift_and_a_missing_pin_refuse_every_start(build, tmp_path):
    pin = _pin(build)
    changed = _copy_build(build, tmp_path / "changed", flip="native/runtime/composite/libelpis_ecs_composite.so")
    rc, out = cli(changed, "status", pin=pin)
    assert (rc, out[0]["code"]) == (4, "INSTALL_PIN"), out
    runtime = _copy_build(build, tmp_path / "runtime", flip="native/runtime/libelpis_runtime.so")
    rc, out = cli(runtime, "status", pin=pin)
    assert (rc, out[0]["code"]) == (4, "INSTALL_PIN"), out
    rc, out = cli(build, "status", pin="0" * 64)
    assert (rc, out[0]["code"]) == (4, "INSTALL_PIN"), out
    rc, out = cli(build, "status")
    assert (rc, out[0]["code"]) == (2, "CLI_INPUT"), out
