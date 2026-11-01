"""The RAM-only native operator host (``native/runtime/elpis``, docs/NATIVE_HOST.md) as separate processes with a minimal
environment (no PYTHONPATH, no LD_* variables): the production entry point runs no interpreter and writes nothing.

* closure: the binary links only the C library; a live process maps only the C runtime and the six sealed admitted
  objects, has no writable file descriptor and no writable shared file mapping, runs with core dumps and file growth
  disabled, and its descriptors are a fixed set that long sessions never grow;
* the catalog: measured read-only from the libraries and the host binary; any drift of a library or the host binary,
  or a library naming an unadmitted dependency, refuses every start;
* the host's answers are the Python operator interface's: the same initial identity, QUERY values, LEARN transition,
  relation keys, LOOKUP answer and component identities;
* a session learns and records typed evidence admitted by the session's ACTIVE sources, in memory; ``close`` and a new
  process both start again at the lineage's initial identity: nothing is published, recovered or resumed;
* operator events (lifecycle, versions, capacity, reclamation) are typed events of the volatile session;
* every retired persistent command is refused; containment refuses a privileged process and writable storage on a
  standard descriptor, and the contained testing host's deliberate write attempts are refused while the same
  uncontained user may write there; the swap policy locks the process memory or refuses;
* a killed process leaves nothing behind.

As root, every host process runs as uid 65534 (the host refuses a privileged process), reading the build tree and a
world-readable scratch directory. Mechanics only: the stimuli and the evidence are fixtures.
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
import tempfile

import pytest

from ..conftest import native_build_dir

REPO = Path(__file__).resolve().parents[2]
BUILD = native_build_dir()
# An installed product (docs/NATIVE_HOST.md, "Installation"): the host in <prefix>/bin, the libraries flat in
# <prefix>/lib/elpis (read with --libraries). Otherwise the build tree (--native-build).
PREFIX = Path(os.environ["ELPIS_NATIVE_HOST_PREFIX"]) if os.environ.get("ELPIS_NATIVE_HOST_PREFIX") else None
LIBDIR = PREFIX / "lib" / "elpis" if PREFIX else None
HOST = PREFIX / "bin" / "elpis" if PREFIX else BUILD / "native" / "runtime" / "elpis"
HOST_TESTING = BUILD / "native" / "runtime" / "elpis_testing"   # test support: never installed
_LIBS = ("native/ECS_G/libelpis_ecsg_math.so", "native/ECS_G/libelpis_ecsg_k1.so", "native/ECS_C/libelpis_ecs_memory.so",
         "native/runtime/composite/libelpis_ecs_composite.so", "native/runtime/libelpis_hacf_observation.so",
         "native/runtime/libelpis_runtime.so")
ENV = {"PATH": "/usr/bin:/bin", "LC_ALL": "C"}
UNPRIVILEGED = ["setpriv", "--reuid=65534", "--regid=65534", "--clear-groups", "--"] if os.geteuid() == 0 else []

pytestmark = pytest.mark.skipif(not (HOST.exists() and HOST_TESTING.exists()),
                                reason=f"the native host is not built under {BUILD}")


def node(name):
    return hashlib.sha256(b"cli-fixture-node:" + name.encode()).hexdigest()


def prov(name):
    return hashlib.sha256(b"cli-fixture-provenance:" + name.encode()).hexdigest()


# The Python operator interface's fixtures (tests/integration/test_cli.py), shared by the differential below.
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
SOURCE = {"op": "source_found", "label": "fixture source 1", "source_authority": 1, "per_turn": 8,
          "scope_label": "fixture scope"}


def evidence(tick, authority=1, polarity="ASSERT", **extra):
    item = dict(EDGE, provenance=prov(f"record-{tick}"), authority=authority, polarity=polarity, tick=tick)
    return dict({"op": "evidence", "items": [item], "grant": "test operator", "sources": [authority]}, **extra)


# ---- process helpers -------------------------------------------------------------------------------------------

def _source_args(source=None):
    if source is not None:
        return ["--libraries" if LIBDIR else "--native-build", str(source)]
    return ["--libraries", str(LIBDIR)] if LIBDIR else ["--native-build", str(BUILD)]


def run_host(*args, script=None, testing=False, env=None, prefix=None, stdout=subprocess.PIPE):
    """Run the host as given (no source or pin added); returns (exit status, the JSON lines)."""
    cmd = [*(UNPRIVILEGED if prefix is None else prefix), str(HOST_TESTING if testing else HOST), *args]
    text = None
    if script is not None:
        cmd += ["session", "--script", "-"]
        text = "\n".join(json.dumps(op) for op in script) + "\n"
    run = subprocess.run(cmd, input=text, stdout=stdout, stderr=subprocess.PIPE, text=True, env=env or ENV, cwd="/",
                         timeout=300)
    assert run.stderr == "", run.stderr
    lines = run.stdout.splitlines() if stdout is subprocess.PIPE else []
    return run.returncode, [json.loads(line) for line in lines if line.strip()]


_PINS = {}


def pin(testing=False, source=None):
    """The catalog pin of a host variant over a library set (`catalog`, measured read-only)."""
    key = (testing, str(source))
    if key not in _PINS:
        rc, out = run_host(*_source_args(source), "catalog", testing=testing)
        assert rc == 0 and out[0]["ok"], out
        _PINS[key] = out[0]["catalog_sha256"]
    return _PINS[key]


def host(*args, script=None, testing=False, source=None, seed=7, **kwargs):
    """One host command (or a session over `script`) over the pinned libraries."""
    return run_host(*_source_args(source), "--catalog-sha256", pin(testing, source), "--seed", str(seed), *args,
                    script=script, testing=testing, **kwargs)


@pytest.fixture
def shared():
    """A scratch directory the unprivileged host may read (pytest's own temporary directories are private)."""
    root = Path(tempfile.mkdtemp(prefix="elpis-host-"))
    root.chmod(0o755)
    try:
        yield root
    finally:
        shutil.rmtree(root)


class Live:
    """A testing-host session that pauses at each ``testing_pause``: the harness inspects or kills it there."""

    def __init__(self, script, root):
        path = root / f"script-{os.getpid()}-{id(self)}.jsonl"
        path.write_text("\n".join(json.dumps(op) for op in script) + "\n")
        path.chmod(0o644)
        self.proc = subprocess.Popen([*UNPRIVILEGED, str(HOST_TESTING), *_source_args(), "--catalog-sha256",
                                      pin(True), "--seed", "7", "session", "--script", str(path)],
                                     stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
                                     text=True, env=ENV, cwd="/")
        self.out = []

    def until_pause(self):
        for line in self.proc.stdout:
            op = json.loads(line)
            self.out.append(op)
            if op.get("op") == "testing_pause" and op.get("paused"):
                return op
        raise AssertionError(("the session ended before a pause", self.out))

    def resume(self):
        self.proc.stdin.write("r")
        self.proc.stdin.flush()

    def kill(self):
        self.proc.send_signal(signal.SIGKILL)
        self.proc.wait(timeout=60)

    def finish(self):
        for line in self.proc.stdout:
            self.out.append(json.loads(line))
        return self.proc.wait(timeout=300), self.out


def _staged(root, rel):
    """Where library `rel` (a build-tree path) sits in a staged copy: flat for an installed product."""
    return root / (Path(rel).name if LIBDIR else rel)


def _staging(root, flip=None, patch=None):
    for rel in _LIBS:
        target = _staged(root, rel)
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(LIBDIR / Path(rel).name if LIBDIR else BUILD / rel, target)
    for p in [root, *root.rglob("*")]:
        p.chmod(0o755 if p.is_dir() else 0o644)
    if flip is not None:
        data = bytearray(_staged(root, flip).read_bytes())
        data[-1] ^= 1
        _staged(root, flip).write_bytes(bytes(data))
    if patch is not None:
        rel, old, new = patch
        data = _staged(root, rel).read_bytes()
        assert data.count(old) == 1, (rel, old)
        _staged(root, rel).write_bytes(data.replace(old, new))
    return root


# ---- closure ---------------------------------------------------------------------------------------------------

def _dynamic(path):
    text = subprocess.run(["readelf", "-d", str(path)], capture_output=True, text=True, check=True).stdout
    needed = [line.split("[", 1)[1].rstrip("]") for line in text.splitlines() if "(NEEDED)" in line]
    paths = [line for line in text.splitlines() if "(RUNPATH)" in line or "(RPATH)" in line]
    return needed, paths


# A sanitizer build adds its runtime (and that runtime's own dependencies) to every process; nothing else.
_SANITIZERS = ("libasan.so", "libubsan.so", "libtsan.so")
SANITIZED = any(n.startswith(_SANITIZERS) for n in (_dynamic(HOST)[0] if HOST.exists() else ()))


def test_the_host_links_only_the_c_library():
    for exe in (HOST, HOST_TESTING):
        needed, paths = _dynamic(exe)
        needed = [n for n in needed if not n.startswith(_SANITIZERS)]
        assert needed == ["libc.so.6"] and paths == [], (exe, needed, paths)
        data = exe.read_bytes()
        assert b"libpython" not in data and b"Py_Initialize" not in data


def _proc(pid, name):
    try:
        return Path(f"/proc/{pid}/{name}").read_text()
    except PermissionError:
        pytest.skip("a non-dumpable process's /proc entries are readable only with CAP_SYS_PTRACE (run as root)")


def _descriptors(pid):
    out = []
    for fd in sorted(Path(f"/proc/{pid}/fd").iterdir(), key=lambda p: int(p.name)):
        try:
            out.append((fd.name, os.readlink(fd), Path(f"/proc/{pid}/fdinfo/{fd.name}").read_text()))
        except FileNotFoundError:
            pass
    return out


def test_a_live_host_maps_no_interpreter_writes_nowhere_and_its_descriptors_never_grow(shared):
    work = [LEARN, {"op": "query", "x": X}, {"op": "status"}, {"op": "authority"}, {"op": "live"}]
    script = [SOURCE, evidence(1), {"op": "testing_pause"}] + work * 10 + [{"op": "testing_pause"}] + \
        [evidence(t) for t in range(2, 12)] + [{"op": "close"}] + work * 10 + [{"op": "testing_pause"}]
    live = Live(script, shared)
    seen = []
    for _ in range(3):
        live.until_pause()
        pid = live.proc.pid
        maps = _proc(pid, "maps")
        mapped = [line.split(maxsplit=5) for line in maps.splitlines()]
        paths = sorted({m[5] for m in mapped if len(m) == 6 and (m[5].startswith("/") or m[5].startswith("/memfd:"))})
        assert not [p for p in paths if "python" in p.lower()], paths
        sealed = {(m[3], m[4]) for m in mapped if len(m) == 6 and m[5].startswith("/memfd:elpis-admitted")}
        assert len(sealed) == 6, paths                        # six distinct admitted objects, each from its seal
        # No writable shared mapping of any file: a write there could reach storage.
        assert not [m for m in mapped if len(m) == 6 and m[1][1] == "w" and m[1][3] == "s"], maps
        others = {os.path.basename(p) for p in paths if not p.startswith("/memfd:")}
        if SANITIZED:
            others = {o for o in others if not o.startswith(_SANITIZERS + ("libstdc++.so", "libm.so"))
                      and not (o.startswith("tsan.") and o.endswith(" (deleted)"))}
        assert others <= {"elpis_testing", "libc.so.6", "libm.so.6", "ld-linux-x86-64.so.2", "ld-linux-aarch64.so.1",
                          "libgcc_s.so.1"}, others
        limits = _proc(pid, "limits")
        assert [line.split()[-3:-1] for line in limits.splitlines() if line.startswith(("Max core file size",
                                                                                         "Max file size"))] == \
            [["0", "0"], ["0", "0"]], limits
        descriptors = _descriptors(pid)
        for fd, target, info in descriptors:
            flags = int(next(line for line in info.splitlines() if line.startswith("flags:")).split()[1], 8)
            writable = flags & os.O_ACCMODE in (os.O_WRONLY, os.O_RDWR)
            # Writable descriptors are the pipes to the harness (and inert devices); nothing on storage.
            assert not writable or target.startswith(("pipe:", "/dev/null", "anon_inode:")), (fd, target, flags)
        seen.append([(fd, target) for fd, target, _ in descriptors])
        live.resume()
    rc, out = live.finish()
    assert rc == 0 and all(op.get("ok", True) for op in out), [op for op in out if not op.get("ok", True)]
    # The same descriptors at every pause: 20 learning turns, 11 evidence turns and a new session opened nothing.
    assert seen[0] == seen[1] == seen[2], seen
    assert len(seen[0]) <= 16, seen[0]
    assert sum("memfd:elpis-admitted" in target for _, target in seen[0]) == 6


# ---- the catalog and pins ---------------------------------------------------------------------------------------

def test_the_catalog_pins_the_host_and_every_library_read_only(shared):
    staging = _staging(shared / "libs")
    before = {rel: hashlib.sha256(_staged(staging, rel).read_bytes()).hexdigest() for rel in _LIBS}
    rc, out = run_host(*_source_args(staging), "catalog")
    assert rc == 0 and out[0]["ok"], out
    catalog = out[0]["catalog"]
    assert catalog["host_sha256"] == hashlib.sha256(HOST.read_bytes()).hexdigest()
    assert catalog["abi"] == {"runtime": 8, "composite": 1}
    assert {e["id"]: e["sha256"] for e in catalog["libraries"]} == \
        {Path(rel).name[3:-3]: digest for rel, digest in before.items()}
    assert [e["id"] for e in catalog["libraries"]] == ["elpis_ecsg_math", "elpis_ecsg_k1", "elpis_ecs_memory",
                                                       "elpis_ecs_composite", "elpis_hacf_observation",
                                                       "elpis_runtime"]
    assert {rel: hashlib.sha256(_staged(staging, rel).read_bytes()).hexdigest() for rel in _LIBS} == before
    assert sorted(p.name for p in shared.iterdir()) == ["libs"]           # nothing written beside the copies


def test_a_library_naming_an_unadmitted_dependency_is_refused_before_it_loads(shared):
    staging = _staging(shared / "libs", patch=("native/runtime/libelpis_hacf_observation.so", b"\0libc.so.6\0",
                                                b"\0libX.so.6\0"))
    rc, out = run_host(*_source_args(staging), "catalog")
    assert (rc, out[0]["code"]) == (4, "INSTALL_PIN"), out
    assert "libelpis_hacf_observation.so depends on libX.so.6" in out[0]["detail"]


@pytest.mark.parametrize("what", ["library", "runtime", "host", "pin"])
def test_drift_refuses_every_start(shared, what):
    if what == "library":
        staging = _staging(shared / "libs", flip="native/ECS_C/libelpis_ecs_memory.so")
        rc, out = run_host(*_source_args(staging), "--catalog-sha256", pin(), "status")
    elif what == "runtime":
        staging = _staging(shared / "libs", flip="native/runtime/libelpis_runtime.so")
        rc, out = run_host(*_source_args(staging), "--catalog-sha256", pin(), "status")
    elif what == "host":   # another host binary under the production pin (the catalog names the host's own bytes)
        rc, out = run_host(*_source_args(), "--catalog-sha256", pin(), "status", testing=True)
    else:
        rc, out = run_host(*_source_args(), "--catalog-sha256", "0" * 64, "status")
    assert (rc, out[0]["code"]) == (4, "INSTALL_PIN"), out


# ---- the host answers as the Python operator interface ------------------------------------------------------------

def _sanitizer_preload():
    """Under a sanitizer build the Python process loading sanitized libraries needs the runtime preloaded."""
    if not SANITIZED:
        return {}
    runtime = next(n for n in _dynamic(HOST)[0] if n.startswith(_SANITIZERS))
    found = subprocess.run(["gcc", f"-print-file-name={runtime}"], capture_output=True, text=True).stdout.strip()
    return {"LD_PRELOAD": found} if found and os.path.isabs(found) else {}


def _python_cli(*args, script=None):
    env = dict(os.environ, PYTHONPATH=str(REPO / "src"), PYTHONDONTWRITEBYTECODE="1", **_sanitizer_preload())
    # The Python reference is an offline tool, not part of the installed product (whose lib/elpis holds the host's six
    # libraries, not the K1 FMS adapter the reference also admits): it always reads the build tree, while the host
    # under test may be the installed one.
    source = ["--native-build", str(BUILD)]
    catalog = subprocess.run([sys.executable, "-m", "elpis.runtime.cli", *source, "catalog"], capture_output=True,
                             text=True, env=env, timeout=300)
    python_pin = json.loads(catalog.stdout)["catalog_sha256"]
    cmd = [sys.executable, "-m", "elpis.runtime.cli", *source, "--catalog-sha256", python_pin, "--seed", "7", *args]
    if script is not None:
        cmd += ["session", "--script", "-"]
    run = subprocess.run(cmd, input=json.dumps(script) if script is not None else None, capture_output=True,
                         text=True, env=env, timeout=300)
    return run.returncode, [json.loads(line) for line in run.stdout.splitlines() if line.strip()]


def test_the_native_host_answers_exactly_as_the_python_operator_interface():
    script = [{"op": "live"}, SOURCE, {"op": "query", "x": X}, LEARN, {"op": "query", "x": X}, EVIDENCE, LOOKUP,
              {"op": "live"}]
    rc, native = host(script=script)
    assert rc == 0 and all(o["ok"] for o in native), native
    rc, python = _python_cli(script=script)
    assert rc == 0 and all(o["ok"] for o in python), python
    n_live0, n_src, n_q0, n_learn, n_q1, n_ev, n_found, n_live1 = native
    p_live0, p_src, p_q0, p_learn, p_q1, p_ev, p_found, p_live1 = python
    assert n_live0["live"] == p_live0["live"]                               # the same initial composite
    assert n_src["id"] == p_src["id"] or n_src["source_authority"] == p_src["source_authority"]
    assert (n_q0["values_sha256"], n_q1["values_sha256"]) == (p_q0["values_sha256"], p_q1["values_sha256"])
    assert (n_learn["epoch"], n_learn["generation"], n_learn["identity"]) == \
        (p_learn["epoch"], p_learn["generation"], p_learn["identity"])
    assert [e["relation_key"] for e in n_ev["entries"]] == [e["relation_key"] for e in p_ev["entries"]]
    assert n_ev["identity"] == p_ev["identity"]
    assert (n_found["found"], n_found["answer"]) == (p_found["found"], p_found["answer"]) == (True, node("engineer"))
    assert n_live1["live"] == p_live1["live"]


# ---- the session: in memory only ------------------------------------------------------------------------------------

def test_a_session_works_in_memory_and_nothing_survives_it():
    rc, out = host(script=[{"op": "open"}, SOURCE, {"op": "query", "x": X}, LEARN, EVIDENCE, LOOKUP, {"op": "live"},
                           {"op": "close"}, {"op": "live"}, LOOKUP, {"op": "query", "x": X}, {"op": "authority"}])
    assert rc == 0 and all(o["ok"] for o in out), out
    opened, source, q0, learned, recorded, found, live, closed, fresh, lost, again, authority = out
    initial = opened["live"]["identity"]
    assert opened["session"]["durability"] == "VOLATILE" and opened["session"]["ordinal"] == 1
    assert learned["durability"] == recorded["durability"] == "VOLATILE"
    assert found["found"] and live["live"]["identity"] != initial
    assert closed["closed"] and closed["discarded"]["learned"]
    assert fresh["live"]["identity"] == initial and not lost["found"]
    assert again["values_sha256"] == q0["values_sha256"]
    assert authority["session"]["ordinal"] == 2 and authority["session"]["lineage"] != opened["session"]["lineage"]
    assert [e["kind"] for e in authority["entities"]] == ["OPERATOR", "RUNTIME"]   # the source died with session 1
    rc, out = host(script=[{"op": "live"}, LOOKUP, {"op": "query", "x": X}])
    assert rc == 0 and out[0]["live"]["identity"] == initial and not out[1]["found"]
    assert out[2]["values_sha256"] == q0["values_sha256"]


def test_a_killed_process_leaves_nothing(shared):
    before = sorted(p.name for p in shared.iterdir())
    live = Live([SOURCE, LEARN, evidence(1), {"op": "testing_pause"}, LEARN], shared)
    live.until_pause()
    live.kill()
    assert live.proc.returncode == -signal.SIGKILL
    scripts = [p.name for p in shared.iterdir() if p.name.startswith("script-")]
    assert sorted(p.name for p in shared.iterdir() if p.name not in scripts) == before
    rc, out = host(script=[{"op": "live"}, LOOKUP])
    assert rc == 0 and out[0]["live"]["identity"] == out[0]["initial_identity"] and not out[1]["found"]


# ---- refusals ----------------------------------------------------------------------------------------------------

@pytest.mark.parametrize("args", [("init",), ("create",), ("publish",), ("reconcile",), ("candidates",),
                                  ("authority-reconcile",), ("--home", "/nowhere", "status"),
                                  ("--contained", "status"), ("learn", "--publish")])
def test_every_retired_persistent_command_is_refused(args):
    rc, out = host(*args)
    assert (rc, out[0]["code"]) == (2, "OPERATION_RETIRED"), out


def test_refusals_are_reported_with_stable_codes_and_change_nothing():
    script = [SOURCE, {"op": "live"},
              dict(LEARN, grant=None, expect="LEARN_UNAUTHORIZED"),
              dict(LEARN, rate=-1, expect="LEARNING_RATE"),
              dict(LEARN, grant="   ", expect="LEARN_AUTHORITY"),
              dict(EVIDENCE, grant=None, expect="OBSERVATION_UNAUTHORIZED"),
              dict(EVIDENCE, sources=[2], expect="OBSERVATION_AUTHORITY"),
              evidence(2, authority=2, expect="RUNTIME_OBSERVATION_REFUSED"),       # no ACTIVE source 2
              dict(LEARN, schedule=[[4, 1048576]], expect="COGNITION_FUEL_EXCEEDED"),
              {"op": "text", "input": "hello", "expect": "ECS_CODEC_UNQUALIFIED"},
              {"op": "publish", "expect": "OPERATION_RETIRED"},
              {"op": "nonsense", "expect": "CLI_INPUT"},
              {"op": "live"}]
    rc, out = host(script=script)
    assert rc == 0 and all(o.get("met", o["ok"]) for o in out), out
    assert out[1]["live"] == out[-1]["live"]


def test_operator_events_are_typed_events_of_the_volatile_session():
    rc, out = host(script=[SOURCE, evidence(1),
                           {"op": "source_lifecycle", "source_authority": 1, "to": "DORMANT"},
                           evidence(2, expect="RUNTIME_OBSERVATION_REFUSED"),          # a DORMANT source is refused
                           {"op": "source_lifecycle", "source_authority": 1, "to": "ACTIVE"},
                           {"op": "source_version", "source_authority": 1, "expected": 5, "version": 6,
                            "expect": "ECS_C_SOURCE_STALE"},
                           {"op": "source_version", "source_authority": 1, "expected": 1, "version": 2},
                           evidence(3),
                           {"op": "source_reclaim", "source_authority": 1, "expect": "ECS_C_LIFECYCLE"},
                           {"op": "authority"}])
    assert rc == 0 and all(o.get("met", o["ok"]) for o in out), out
    source = [e for e in out[-1]["entities"] if e["kind"] == "SOURCE"]
    assert len(source) == 1 and source[0]["lifecycle"] == "ACTIVE" and source[0]["source_version"] == 2


def test_entity_capacity_is_refused_and_reclamation_frees_a_row():
    second = dict(SOURCE, label="fixture source 2", source_authority=2)
    rc, out = host("--capacity", "3", script=[
        SOURCE, dict(second, expect="ECS_C_CAPACITY"),
        {"op": "source_lifecycle", "source_authority": 1, "to": "TERMINATED"},
        {"op": "source_reclaim", "source_authority": 1},
        second, evidence(1, authority=2), {"op": "authority"}])
    assert rc == 0 and all(o.get("met", o["ok"]) for o in out), out
    assert [e["source_authority"] for e in out[-1]["entities"] if e["kind"] == "SOURCE"] == [2]


# ---- containment ------------------------------------------------------------------------------------------------

def test_containment_refuses_what_it_cannot_contain(shared):
    if os.geteuid() == 0:                                     # a privileged process is never contained
        rc, out = run_host(*_source_args(), "catalog", prefix=[])
        assert (rc, out[0]["code"]) == (4, "CONTAINMENT_REFUSED"), out
    sink = shared / "stdout"
    with open(sink, "w") as f:                                 # writable storage on a standard descriptor
        run = subprocess.run([*UNPRIVILEGED, str(HOST), *_source_args(), "catalog"], stdout=f,
                             stderr=subprocess.PIPE, text=True, env=ENV, cwd="/", timeout=300)
    # Refused, and not even the refusal was written there (it went to standard error, a pipe).
    assert run.returncode == 4 and sink.read_text() == "" and "CONTAINMENT_REFUSED" in run.stderr
    # Standard error on storage (standard output a pipe): a refusal, a usage error and a retired command each leave
    # it empty, whatever is refused.
    for args in ((*_source_args(), "catalog"), ("bogus",), ("init",)):
        sink.write_text("")
        with open(sink, "w") as f:
            run = subprocess.run([*UNPRIVILEGED, str(HOST), *args], stdout=subprocess.PIPE, stderr=f, text=True,
                                 env=ENV, cwd="/", timeout=300)
        assert run.returncode in (2, 4) and sink.read_text() == "", (args, run.returncode)


def test_the_contained_hosts_deliberate_write_attempts_are_all_refused(shared):
    probe = shared / "probe"
    probe.mkdir()
    probe.chmod(0o777)
    existing = probe / "existing"
    existing.write_text("existing")
    existing.chmod(0o666)
    # Uncontained, the same user may create and remove a file there: the refusals below are the containment's.
    subprocess.run([*UNPRIVILEGED, "sh", "-c", f"echo x > {probe}/control && rm {probe}/control"], check=True)
    kinds = [("new", "create"), ("existing", "write"), ("existing", "extend"), ("existing", "remove"),
             ("existing", "rename"), ("dir", "mkdir")]
    rc, out = host(testing=True, script=[{"op": "testing_write_probe", "path": str(probe / name), "kind": kind,
                                          "offset": 64} for name, kind in kinds] + [{"op": "status"}])
    assert rc == 0 and len(out) == len(kinds) + 1, out
    assert all(o["errno"] != 0 for o in out[:-1]), out
    containment = out[-1]["containment"]
    assert (containment["restricted"], containment["write_grants"], containment["core_limit"],
            containment["fsize_limit"], containment["dumpable"]) == (True, 0, 0, 0, False), containment
    assert sorted(p.name for p in probe.iterdir()) == ["existing"] and existing.read_text() == "existing"


def test_the_swap_policy_locks_the_process_memory_or_refuses(shared):
    swaps = shared / "swaps"
    swaps.write_text("Filename\tType\tSize\tUsed\tPriority\n/swapfile file 4194300 0 -2\n")
    swaps.chmod(0o644)
    env = dict(ENV, ELPIS_TESTING_PROC_SWAPS=str(swaps))
    rc, out = run_host(*_source_args(), "catalog", testing=True, env=env,
                       prefix=["prlimit", "--memlock=0:0", *UNPRIVILEGED])
    assert (rc, out[0]["code"]) == (4, "SWAP_POLICY"), out              # active swap and no lockable memory
    rc, out = host(testing=True, env=env, script=[{"op": "status"}])
    if rc == 4:   # this machine's RLIMIT_MEMLOCK cannot hold the host: refused, never run unlocked
        assert out[0]["code"] == "SWAP_POLICY", out
    else:
        assert rc == 0 and out[0]["containment"]["memory_locked"] and out[0]["containment"]["swap_areas"] == 1
