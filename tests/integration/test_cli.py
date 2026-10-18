"""The operator interface (``python -m elpis.runtime.cli``) over a real deployment, as separate processes.

* init pins an operator build's libraries into the home; afterwards the build tree is never used (deleted here);
* create anchors one ECS composite lineage; a session learns numerically, queries, records typed HACF evidence
  (observe-only: no K1 learning), looks it up and publishes recoverably; a NEW process resumes exactly that state
  (byte-identical QUERY answers, the same LOOKUP, live identity == durable identity);
* volatile work not published is lost at exit, and the API says so;
* refusals: an existing deployment or lineage is never overwritten; a symlinked or group-writable home; a changed
  library, catalog, pin or ABI (pin drift) refuses every start; unauthorized LEARN and evidence; a script whose
  declared expectation is not met exits non-zero;
* an interrupted publication (the testing RuntimeCore build parks the durability worker after the checkpoint slot
  is synced; the process is SIGKILLed) leaves an unresolved candidate: cognitive operations are refused until the
  operator adopts or discards exactly that candidate.

Every output is machine-readable JSON. Mechanics only: the evidence and stimuli are fixtures.
"""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys

import pytest

from ..conftest import native_build_dir, require_native_library

REPO = Path(__file__).resolve().parents[2]
_LIBS = ("native/runtime/libelpis_runtime.so", "native/runtime/libelpis_runtime_testing.so",
         "native/runtime/libelpis_hacf_observation.so", "native/ECS_G/libelpis_ecsg_math.so",
         "native/ECS_G/libelpis_ecsg_k1.so", "native/ECS_G/libelpis_ecsg_k1_fms.so", "native/ECS_C/libelpis_ecs_memory.so",
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
EVIDENCE = {"op": "evidence", "items": [dict(EDGE, provenance=prov("record-17"), authority=1, polarity="ASSERT",
                                             tick=1)], "grant": "test operator: source 1", "sources": [1]}
LOOKUP = {"op": "lookup", "relation": "hyperedge:1", "target_role": "incidence:3",
          "cues": [{"role": "incidence:1", "node": node("alice")}, {"role": "incidence:2", "node": node("acme")}]}


@pytest.fixture(scope="module")
def build(tmp_path_factory):
    """A copy of just the libraries a deployment needs, laid out as a build tree (deleted after init)."""
    for name in ("elpis_runtime", "elpis_runtime_testing", "elpis_ecs_composite", "elpis_hacf_observation"):
        require_native_library(name)
    return native_build_dir()


def _copy_build(build, root):
    for rel in _LIBS:
        (root / rel).parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(build / rel, root / rel)
    return root


def cli(home, *args, script=None):
    env = dict(os.environ, PYTHONPATH=str(REPO / "src"), PYTHONDONTWRITEBYTECODE="1")
    cmd = [sys.executable, "-m", "elpis.runtime.cli", "--home", str(home), *args]
    if script is not None:
        cmd += ["session", "--script", "-"]
    run = subprocess.run(cmd, input=json.dumps(script) if script is not None else None, capture_output=True,
                         text=True, env=env, cwd="/", timeout=300)
    out = [json.loads(line) for line in run.stdout.splitlines() if line.strip()]
    assert run.stderr == "" or run.returncode != 0, run.stderr
    return run.returncode, out


def _deploy(build, tmp_path, *, testing=False):
    staging = _copy_build(build, tmp_path / "build-copy")
    home = tmp_path / "home"
    rc, out = cli(home, "init", "--native-build", str(staging), *(["--testing"] if testing else []))
    assert rc == 0 and out[0]["ok"], out
    shutil.rmtree(staging)                                    # never used again
    rc, out = cli(home, "create", "--seed", "7")
    assert rc == 0 and out[0]["ok"] and out[0]["state_format"] == "COMPOSITE", out
    return home, out[0]["identity"]


def test_a_session_learns_records_evidence_publishes_and_a_new_process_resumes_it_exactly(build, tmp_path):
    home, initial = _deploy(build, tmp_path)
    rc, out = cli(home, script=[{"op": "query", "x": X}, LEARN, {"op": "query", "x": X}, EVIDENCE, LOOKUP,
                                {"op": "live"}, {"op": "publish"}, {"op": "status"}])
    assert rc == 0 and all(o["ok"] for o in out), out
    before, learned, after, evidence, found, live, published, status = out
    assert learned["durability"] == "VOLATILE" and learned["epoch"] == [0, 8] and before["values"] != after["values"]
    assert evidence["k1_steps"] == 0 and evidence["durability"] == "VOLATILE"
    assert (found["found"], found["answer"]) == (True, node("engineer"))
    assert live["durable"]["identity"] == initial != live["live"]["identity"]     # volatile until published
    assert published["ticket"]["acknowledged"] and published["durable"]["identity"] == live["live"]["identity"]
    assert status["recovery"]["disposition"] == "RESUMABLE" and status["fault"] is None
    assert status["abi"] == {"runtime": 5, "composite": 1} and "ECS_CODEC_UNQUALIFIED" in status["capabilities"]["text"]
    # A new process: exactly the published state.
    rc, again = cli(home, script=[{"op": "query", "x": X}, LOOKUP, {"op": "live"}])
    assert rc == 0, again
    assert again[0]["values_sha256"] == after["values_sha256"] and again[0]["values"] == after["values"]
    assert again[1]["answer"] == node("engineer")
    assert again[2]["live"]["identity"] == again[2]["durable"]["identity"] == published["durable"]["identity"]


def test_volatile_work_is_lost_at_exit_and_the_api_says_so(build, tmp_path):
    home, initial = _deploy(build, tmp_path)
    rc, out = cli(home, "learn", "--x", json.dumps(X), "--y", "[0.1,-0.2,0.05,0.3]", "--schedule", "[[4,2]]",
                  "--grant", "test operator")
    assert rc == 0 and out[0]["durability"] == "VOLATILE" and out[0]["identity"] != initial
    rc, out = cli(home, "status")
    assert out[0]["live"]["identity"] == out[0]["durable"]["identity"] == initial      # lost, as reported
    rc, out = cli(home, "learn", "--x", json.dumps(X), "--y", "[0.1,-0.2,0.05,0.3]", "--schedule", "[[4,2]]",
                  "--grant", "test operator", "--publish")
    assert rc == 0 and out[1]["ticket"]["acknowledged"]
    rc, status = cli(home, "status")
    assert status[0]["durable"]["identity"] == out[0]["identity"] != initial


def test_refusals_and_expectations_are_reported_and_never_overwrite(build, tmp_path):
    home, _ = _deploy(build, tmp_path)
    assert cli(home, "create")[1][0]["code"] == "LINEAGE_EXISTS"
    rc, out = cli(home, "init", "--native-build", str(build))
    assert (rc, out[0]["code"]) == (4, "DEPLOYMENT_EXISTS")
    rc, out = cli(home, script=[dict(LEARN, grant=None, expect="LEARN_UNAUTHORIZED"),
                                dict(EVIDENCE, grant=None, expect="OBSERVATION_UNAUTHORIZED"),
                                dict(EVIDENCE, sources=[2], expect="OBSERVATION_AUTHORITY")])
    assert rc == 0 and all(o["met"] for o in out), out
    rc, out = cli(home, script=[dict(LEARN, expect="LEARN_UNAUTHORIZED")])           # it succeeds: not as expected
    assert rc == 3 and out[0]["ok"] and not out[0]["met"]
    link = tmp_path / "link"
    link.symlink_to(home)
    assert cli(link, "status")[1][0]["code"] == "DEPLOYMENT_PATH"
    open_home = tmp_path / "open"
    open_home.mkdir(mode=0o777)
    open_home.chmod(0o777)
    assert cli(open_home, "init", "--native-build", str(build))[1][0]["code"] == "DEPLOYMENT_PATH"


@pytest.mark.parametrize("target,code", [("lib/libelpis_runtime.so", "RUNTIME_UNPINNED"),
                                         ("lib/libelpis_ecs_composite.so", "DEPLOYMENT_PIN"),
                                         ("native-catalog.json", "DEPLOYMENT_PIN"),
                                         ("elpis.json:pin", "DEPLOYMENT_PIN"),
                                         ("elpis.json:abi", "DEPLOYMENT_ABI")])
def test_pin_drift_refuses_every_start(build, tmp_path, target, code):
    home, _ = _deploy(build, tmp_path)
    if target.startswith("elpis.json:"):
        path = home / "elpis.json"
        config = json.loads(path.read_text())
        if target.endswith("pin"):
            config["catalog_sha256"] = "0" * 64
        else:
            config["abi"] = {"runtime": 4, "composite": 1}
        path.write_text(json.dumps(config))
    else:
        path = home / target
        path.chmod(0o600)
        data = bytearray(path.read_bytes())
        data[-1] ^= 0x01
        path.write_bytes(bytes(data))
    rc, out = cli(home, "status")
    assert (rc, out[0]["ok"], out[0]["code"]) == (4, False, code), out


_CRASH = r"""
import json, sys, time
from array import array
from pathlib import Path
sys.path.insert(0, sys.argv[1] + "/src")
from elpis.runtime.cognition import LearnAuthority, Stimulus
from elpis.runtime.deployment import Deployment
s = Deployment(Path(sys.argv[2])).session()
x = array("d", [((i * 7 + j * 3) % 11 - 5) / 16.0 for i in range(4) for j in range(6)])
s.runtime.run_numeric_learn(s.state, Stimulus(x, array("d", [0.1, -0.2, 0.05, 0.3]), array("Q", [4, 3]), dim=6),
                            LearnAuthority(0.002, "test operator: crash fixture"))
candidate = s.runtime.live_identity(s.state)[0]
s.runtime._core.testing_durable_fault(2, int(sys.argv[3]), 0)      # park the durability worker at the step
s.runtime.publish_recoverable(s.state, wait=False)
while s.runtime.durability().parked != int(sys.argv[3]):
    time.sleep(0.005)
print(json.dumps({"candidate": candidate.hex()}), flush=True)
time.sleep(3600)
"""


@pytest.mark.parametrize("action", ["adopt", "discard"])
def test_an_interrupted_publication_is_reconciled_only_by_the_operator(build, tmp_path, action):
    home, initial = _deploy(build, tmp_path, testing=True)
    env = dict(os.environ, PYTHONDONTWRITEBYTECODE="1")
    child = subprocess.Popen([sys.executable, "-c", _CRASH, str(REPO), str(home), "5"], stdout=subprocess.PIPE,
                             stderr=subprocess.PIPE, text=True, env=env)
    try:
        line = child.stdout.readline()
        assert line, child.stderr.read()
        candidate = json.loads(line)["candidate"]
    finally:
        child.send_signal(signal.SIGKILL)
        child.wait(timeout=30)
    assert child.returncode == -signal.SIGKILL
    rc, out = cli(home, "candidates")
    assert out[0]["recovery"]["disposition"] == "CANDIDATE_UNRESOLVED"
    assert (out[0]["recovery"]["authorized"], out[0]["recovery"]["candidate"]) == (initial, candidate)
    rc, out = cli(home, "query", "--x", json.dumps(X))
    assert (rc, out[0]["code"]) == (3, "LINEAGE_CANDIDATE_UNRESOLVED")              # no implicit adoption
    rc, out = cli(home, "reconcile", "--" + action, candidate)
    assert rc == 0 and out[0]["recovery"]["disposition"] == "RESUMABLE", out
    expected = candidate if action == "adopt" else initial
    assert out[0]["durable"]["identity"] == expected
    rc, out = cli(home, script=[{"op": "live"}, {"op": "query", "x": X}])
    assert rc == 0 and out[0]["live"]["identity"] == out[0]["durable"]["identity"] == expected
