"""The native operator host (``native/runtime/elpis``, docs/NATIVE_HOST.md) over real deployments, as separate processes
with a minimal environment (no PYTHONPATH, no LD_* variables): the production entry point runs no interpreter.

* closure: the binary links only the C library; a live process maps only the C runtime and the seven sealed admitted
  objects, and its descriptors are a fixed set that long sessions never grow;
* init pins an operator build's libraries (the build tree is deleted afterwards) and the host binary itself; any
  drift of a library, the catalog, the configuration's ABI, the host binary or the build variant refuses every start;
* the host's answers are the Python operator interface's: the same initial identity, QUERY values, LEARN transition,
  relation keys, LOOKUP answer and component identities, over another durable authority (ECS_C, not continuity);
* a session learns, records typed evidence admitted by the ECS_C authority's ACTIVE sources, publishes recoverably,
  and a new process resumes exactly that state; unpublished work is lost at exit and the host says so;
* process death at every step of a durable publication (SIGKILL while the testing build parks the worker there)
  recovers exactly as the durability law says, with no operator reconciliation of the authority;
* copied-back authority records and a corrupt newest record refuse the runtime until the operator reconciles exactly
  what the authority names; the K1 checkpoint candidate is then reconciled separately;
* the home is a fixed file set: its files, sizes and recovery work do not change over many publications and
  restarts; entity capacity is refused before any write and reclamation frees a row.

Mechanics only: the stimuli and the evidence are fixtures; nothing here is a scientific result.
"""
from __future__ import annotations

import errno
import hashlib
import json
import os
from pathlib import Path
import shutil
import signal
import stat
import subprocess
import sys
import tempfile

import pytest

from ..conftest import native_build_dir

REPO = Path(__file__).resolve().parents[2]
BUILD = native_build_dir()
# An installed product (docs/NATIVE_HOST.md, "Installation"): the host in <prefix>/bin, the libraries flat in
# <prefix>/lib/elpis, provisioned with `init --libraries`. Otherwise the build tree.
PREFIX = Path(os.environ["ELPIS_NATIVE_HOST_PREFIX"]) if os.environ.get("ELPIS_NATIVE_HOST_PREFIX") else None
LIBDIR = PREFIX / "lib" / "elpis" if PREFIX else None
HOST = PREFIX / "bin" / "elpis" if PREFIX else BUILD / "native" / "runtime" / "elpis"
HOST_TESTING = PREFIX / "bin" / "elpis_testing" if PREFIX else BUILD / "native" / "runtime" / "elpis_testing"
_LIBS = ("native/ECS_G/libelpis_ecsg_math.so", "native/ECS_G/libelpis_ecsg_k1.so", "native/ECS_C/libelpis_ecs_memory.so",
         "native/runtime/composite/libelpis_ecs_composite.so", "native/runtime/libelpis_hacf_observation.so",
         "native/ECS_C/authority/libelpis_ecs_c_authority.so", "native/runtime/libelpis_runtime.so",
         "native/runtime/libelpis_runtime_testing.so")
ENV = {"PATH": "/usr/bin:/bin", "LC_ALL": "C"}

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

def host(home, *args, script=None, testing=False, env=None):
    """Run one host command (or a session over `script`); returns (exit status, the JSON lines)."""
    cmd = [str(HOST_TESTING if testing else HOST), "--home", str(home), *args]
    text = None
    if script is not None:
        cmd += ["session", "--script", "-"]
        text = "\n".join(json.dumps(op) for op in script) + "\n"
    run = subprocess.run(cmd, input=text, capture_output=True, text=True, env=env or ENV, cwd="/", timeout=300)
    assert run.stderr == "", run.stderr
    return run.returncode, [json.loads(line) for line in run.stdout.splitlines() if line.strip()]


class Live:
    """A testing-host session that pauses at each ``testing_pause``: the harness inspects or kills it there."""

    def __init__(self, home, script, tmp_path):
        path = tmp_path / f"script-{os.getpid()}-{id(self)}.jsonl"
        path.write_text("\n".join(json.dumps(op) for op in script) + "\n")
        self.proc = subprocess.Popen([str(HOST_TESTING), "--home", str(home), "session", "--script", str(path)],
                                     stdin=subprocess.PIPE, stdout=subprocess.PIPE, text=True, env=ENV, cwd="/")
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


def _staging(tmp_path, name="build-copy"):
    root = tmp_path / name
    for rel in _LIBS:
        _staged(root, rel).parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(LIBDIR / Path(rel).name if LIBDIR else BUILD / rel, _staged(root, rel))
    return root


def _init_args(staging):
    return ["--libraries" if LIBDIR else "--native-build", str(staging)]


def deploy(tmp_path, *, testing=False, capacity=None, create=True, name="home"):
    staging = _staging(tmp_path)
    home = tmp_path / name
    extra = ["--capacity", str(capacity)] if capacity else []
    rc, out = host(home, "init", *_init_args(staging), *extra, testing=testing)
    assert rc == 0 and out[0]["ok"], out
    shutil.rmtree(staging)                                    # never used again
    if not create:
        return home, None
    rc, out = host(home, "create", "--seed", "7", testing=testing)
    assert rc == 0 and out[0]["ok"] and out[0]["state_format"] == "COMPOSITE", out
    return home, out[0]["identity"]


def files(*roots):
    """Every file under the roots, with its size (the persistence footprint)."""
    out = {}
    for root in roots:
        for p in sorted(Path(root).rglob("*")):
            if p.is_file():
                out[str(p.relative_to(root.parent))] = p.stat().st_size
    return out


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


def _mapped_objects(pid):
    objects = set()
    for line in Path(f"/proc/{pid}/maps").read_text().splitlines():
        parts = line.split(maxsplit=5)
        if len(parts) == 6 and (parts[5].startswith("/") or parts[5].startswith("/memfd:")):
            objects.add((parts[3], parts[4], parts[5]))     # device, inode, path
    return objects


def _descriptors(pid):
    out = []
    for fd in sorted(Path(f"/proc/{pid}/fd").iterdir(), key=lambda p: int(p.name)):
        try:
            out.append(os.readlink(fd))
        except FileNotFoundError:
            pass
    return out


def test_a_live_host_maps_no_interpreter_and_its_descriptors_never_grow(tmp_path):
    home, _ = deploy(tmp_path, testing=True)
    work = [LEARN, {"op": "publish"}, {"op": "query", "x": X}, {"op": "status"}, {"op": "authority"}]
    script = [SOURCE, evidence(1), {"op": "testing_pause"}] + work * 10 + [{"op": "testing_pause"}] + \
        [evidence(t) for t in range(2, 12)] + work * 10 + [{"op": "testing_pause"}]
    live = Live(home, script, tmp_path)
    seen = []
    for _ in range(3):
        live.until_pause()
        pid = live.proc.pid
        objects = _mapped_objects(pid)
        paths = sorted({p for _, _, p in objects})
        assert not [p for p in paths if "python" in p.lower()], paths
        sealed = {(dev, ino) for dev, ino, p in objects if p.startswith("/memfd:elpis-admitted")}
        assert len(sealed) == 7, paths                        # seven distinct admitted objects, each from its seal
        others = {os.path.basename(p) for p in paths if not p.startswith("/memfd:")}
        if SANITIZED:                                         # the sanitizer runtime and what it needs
            others = {o for o in others if not o.startswith(_SANITIZERS + ("libstdc++.so", "libm.so"))}
        # The C runtime (libm too: a build may link the ECS_G libraries against it), the loader, the unwinder.
        assert others <= {"elpis_testing", "libc.so.6", "libm.so.6", "ld-linux-x86-64.so.2", "ld-linux-aarch64.so.1",
                          "libgcc_s.so.1"}, others
        seen.append(_descriptors(pid))
        live.resume()
    rc, out = live.finish()
    assert rc == 0 and all(op.get("ok", True) for op in out), [op for op in out if not op.get("ok", True)]
    # The same descriptors at every pause: 30 publications and 20 evidence turns opened nothing that stayed open.
    assert seen[0] == seen[1] == seen[2], seen
    assert len(seen[0]) <= 32, seen[0]
    assert sum("memfd:elpis-admitted" in d for d in seen[0]) == 7


# ---- deployment and pins ---------------------------------------------------------------------------------------

def test_init_pins_the_libraries_and_the_host_and_the_build_tree_is_never_used_again(tmp_path):
    staging = _staging(tmp_path)
    expected = {Path(rel).name: hashlib.sha256(_staged(staging, rel).read_bytes()).hexdigest() for rel in _LIBS}
    home = tmp_path / "home"
    rc, out = host(home, "init", *_init_args(staging))
    assert rc == 0, out
    shutil.rmtree(staging)
    pins = out[0]["libraries"]
    assert set(pins) == {"elpis_ecsg_math", "elpis_ecsg_k1", "elpis_ecs_memory", "elpis_ecs_composite",
                         "elpis_hacf_observation", "elpis_ecs_c_authority", "elpis_runtime"}
    for library_id, pin in pins.items():
        assert pin["sha256"] == expected[f"lib{library_id}.so"]
    assert out[0]["host_sha256"] == hashlib.sha256(HOST.read_bytes()).hexdigest()
    assert sorted(p.name for p in home.iterdir()) == ["authority", "catalog", "host.cfg", "lib", "state"]
    assert oct(stat.S_IMODE(home.stat().st_mode)) == "0o700"
    assert all(stat.S_IMODE(p.stat().st_mode) == 0o500 for p in (home / "lib").iterdir())
    rc, out = host(home, "status")
    assert rc == 0 and out[0]["lineage"] is None and out[0]["authority"]["disposition"] == "CLEAN"
    assert out[0]["abi"] == {"runtime": 6, "composite": 1, "ecs_c": 1}
    assert out[0]["capabilities"]["text"].startswith("UNAVAILABLE: ECS_CODEC_UNQUALIFIED")


def test_a_library_naming_an_unadmitted_dependency_is_refused_before_it_loads(tmp_path):
    # A pinned library is still checked: one whose dynamic section names a library that is neither admitted before
    # it nor part of the platform is refused at init, before the loader could resolve that name anywhere.
    staging = _staging(tmp_path)
    lib = _staged(staging, "native/runtime/libelpis_hacf_observation.so")
    data = lib.read_bytes()
    needed = [n for n in _dynamic(lib)[0]]
    assert "libc.so.6" in needed and data.count(b"\0libc.so.6\0") == 1, needed
    lib.write_bytes(data.replace(b"\0libc.so.6\0", b"\0libX.so.6\0"))
    rc, out = host(tmp_path / "home", "init", *_init_args(staging))
    assert (rc, out[0]["code"]) == (4, "DEPLOYMENT_PIN"), out
    assert "libelpis_hacf_observation.so depends on libX.so.6" in out[0]["detail"]
    assert not (tmp_path / "home" / "host.cfg").exists()     # nothing was published


def _drift(home, what):
    if what == "library":
        lib = home / "lib" / "libelpis_ecs_memory.so"
        lib.chmod(0o700)
        data = bytearray(lib.read_bytes())
        data[-1] ^= 1
        lib.write_bytes(bytes(data))
    elif what == "catalog":
        cat = home / "catalog"
        cat.chmod(0o600)
        cat.write_text(cat.read_text().replace('"schema"', ' "schema"', 1))
    elif what == "abi":
        cfg = home / "host.cfg"
        cfg.write_text(cfg.read_text().replace('"runtime":6', '"runtime":5'))
    elif what == "pin":
        cfg = home / "host.cfg"
        doc = json.loads(cfg.read_text())
        doc["catalog_sha256"] = "0" * 64
        cfg.write_text(json.dumps(doc))


@pytest.mark.parametrize("what,code", [("library", "DEPLOYMENT_PIN"), ("catalog", "DEPLOYMENT_PIN"),
                                       ("abi", "DEPLOYMENT_ABI"), ("pin", "DEPLOYMENT_PIN")])
def test_drift_refuses_every_start(tmp_path, what, code):
    home, _ = deploy(tmp_path)
    _drift(home, what)
    for args in (("status",), ("query", "--x", json.dumps(X))):
        rc, out = host(home, *args)
        assert (rc, out[0]["ok"], out[0]["code"]) == (4, False, code), out


def test_another_host_binary_or_variant_is_refused(tmp_path):
    home, _ = deploy(tmp_path)
    other = tmp_path / "elpis-other"
    other.write_bytes(HOST.read_bytes() + b"\0")             # runs exactly as before; its bytes differ
    other.chmod(0o700)
    run = subprocess.run([str(other), "--home", str(home), "status"], capture_output=True, text=True, env=ENV)
    assert run.returncode == 4 and json.loads(run.stdout)["code"] == "DEPLOYMENT_PIN"
    rc, out = host(home, "status", testing=True)
    assert (rc, out[0]["code"]) == (4, "DEPLOYMENT_CONFIG")   # the testing host never runs a production deployment
    testing_home, _ = deploy(tmp_path, testing=True, name="testing-home")
    rc, out = host(testing_home, "status")
    assert (rc, out[0]["code"]) == (4, "DEPLOYMENT_CONFIG")   # nor the production host a testing deployment


def test_the_home_path_is_checked_and_nothing_is_overwritten(tmp_path):
    home, _ = deploy(tmp_path)
    rc, out = host(home, "init", "--native-build", str(BUILD))
    assert (rc, out[0]["code"]) == (4, "DEPLOYMENT_EXISTS")
    rc, out = host(home, "create")
    assert (rc, out[0]["code"]) == (3, "LINEAGE_EXISTS")
    link = tmp_path / "link"
    link.symlink_to(home)
    assert host(link, "status")[1][0]["code"] == "DEPLOYMENT_PATH"
    assert host(Path("relative/home"), "status")[1][0]["code"] == "DEPLOYMENT_PATH"
    assert host(Path(str(home) + "/../home"), "status")[1][0]["code"] == "DEPLOYMENT_PATH"
    home.chmod(0o770)
    assert host(home, "status")[1][0]["code"] == "DEPLOYMENT_PATH"
    home.chmod(0o700)
    rc, out = host(tmp_path / "missing", "status")
    assert (rc, out[0]["code"]) == (4, "DEPLOYMENT_MISSING")


def _inodes(home):
    return {str(q.relative_to(home)): (q.stat().st_ino, q.stat().st_size) for q in sorted(home.rglob("*")) if q.is_file()}


def _parked(cmd, timeout=120):
    """Run a testing-host command until its watcher reports the park, then SIGKILL it (a real process death)."""
    proc = subprocess.Popen(cmd, stdin=subprocess.DEVNULL, stdout=subprocess.PIPE, text=True, env=ENV, cwd="/")
    lines = []
    try:
        for line in proc.stdout:
            lines.append(json.loads(line))
            if lines[-1].get("op") == "testing_parked":
                break
        assert lines and lines[-1].get("op") == "testing_parked", lines
    finally:
        proc.send_signal(signal.SIGKILL)
        proc.wait(timeout=timeout)
    return lines


# Death inside create's anchor at each control point (docs/RUNTIME_CORE.md, "Control-operation crash points"):
# whether the authority holds the anchor, and the authority's recovery disposition after the death.
ANCHOR = {1: (False, "CLEAN"), 2: (False, "CLEAN"), 3: (False, "CLEAN"), 4: (False, "TRAILING_INCOMPLETE"),
          5: (True, "INTERRUPTED_COMPLETED"), 6: (True, "INTERRUPTED_COMPLETED")}


@pytest.mark.parametrize("point", sorted(ANCHOR))
def test_an_interrupted_create_is_completed_by_the_same_create(tmp_path, point):
    reference, initial = deploy(tmp_path, testing=True, name="reference")
    home, _ = deploy(tmp_path, testing=True, create=False)
    _parked([str(HOST_TESTING), "--home", str(home), "create", "--seed", "7", "--testing-anchor-park", str(point),
             "--testing-fault-arg", "100"])
    anchored, disposition = ANCHOR[point]
    rc, out = host(home, script=[{"op": "authority"}, {"op": "status"}], testing=True)
    assert rc == 0 and out[0]["report"]["disposition"] == disposition and out[0]["reconciliation"] is None, out
    assert out[1]["recovery"]["disposition"] == ("RESUMABLE" if anchored else "NOTHING_TO_RECOVER"), out[1]
    rc, out = host(home, "create", "--seed", "8", testing=True)
    assert (rc, out[0]["code"]) == (3, "LINEAGE_EXISTS")      # another lineage: never overwritten
    rc, out = host(home, "create", "--seed", "7", testing=True)
    assert rc == 0 and out[0]["ok"] and out[0]["identity"] == initial and out[0]["resumed"] is anchored, out
    rc, out = host(home, script=[SOURCE, LEARN, evidence(1), {"op": "publish"}], testing=True)
    assert rc == 0 and all(o["ok"] for o in out), out
    # The same 17 files as an uninterrupted create, the lineage's and the slots' at the same sizes (host.cfg names the
    # home's own path).
    sizes = lambda h: {k: v[1] for k, v in _inodes(h).items() if k != "host.cfg"}
    assert sorted(_inodes(home)) == sorted(_inodes(reference)) and sizes(home) == sizes(reference)
    rc, out = host(home, "create", "--seed", "7", testing=True)
    assert (rc, out[0]["code"]) == (3, "LINEAGE_EXISTS")      # past its anchor: the lineage has moved on


def test_create_removes_only_its_own_debris_and_a_created_lineage_is_never_touched(tmp_path):
    home, _ = deploy(tmp_path, create=False)
    (home / "lineage.cfg.new").write_text("partial")          # what a death inside the first publication leaves
    (home / "checkpoint.new").mkdir()
    (home / "checkpoint.new" / "k1-checkpoint.a").write_bytes(b"\0" * 64)
    rc, out = host(home, "create", "--seed", "7")
    assert rc == 0 and out[0]["resumed"] is False, out
    assert not (home / "lineage.cfg.new").exists() and not (home / "checkpoint.new").exists()
    before = _inodes(home)
    rc, out = host(home, "create", "--seed", "7")             # an untouched lineage at its initial identity
    assert rc == 0 and out[0]["resumed"] is True and _inodes(home) == before, out
    # Debris that is not this host's own is refused, and nothing is removed.
    other, _ = deploy(tmp_path, create=False, name="other")
    (other / "checkpoint.new").mkdir()
    (other / "checkpoint.new" / "notes").write_text("someone else's")
    rc, out = host(other, "create", "--seed", "7")
    assert (rc, out[0]["code"]) == (3, "LINEAGE_EXISTS") and (other / "checkpoint.new" / "notes").exists()


def test_a_parked_checkpoint_reconciliation_killed_there_resolves_exactly(tmp_path):
    home, old = deploy(tmp_path, testing=True)
    live = Live(home, [SOURCE, LEARN, {"op": "testing_durable_fault", "action": 2, "step": 5, "arg": 0},
                       {"op": "publish", "wait": False}, {"op": "testing_wait_parked", "parked": 5},
                       {"op": "testing_pause"}], tmp_path)
    live.until_pause()
    live.kill()
    new = live.out[1]["identity"]
    rc, out = host(home, "status", testing=True)
    assert out[0]["recovery"]["disposition"] == "CANDIDATE_UNRESOLVED" and out[0]["recovery"]["candidate"] == new
    script = tmp_path / "discard.jsonl"
    script.write_text("\n".join(json.dumps(o) for o in [
        {"op": "testing_control_fault", "kind": 3, "action": 2, "point": 2, "arg": 0},
        {"op": "reconcile", "action": "discard", "candidate": new}]) + "\n")
    lines = _parked([str(HOST_TESTING), "--home", str(home), "session", "--script", str(script)])
    assert lines[-1]["parked"] == 302 and lines[0]["armed"] is True
    rc, out = host(home, script=[{"op": "authority"}, {"op": "status"}], testing=True)
    assert out[0]["report"]["disposition"] == "CLEAN" and out[0]["reconciliation"] is None, out[0]
    assert out[1]["recovery"]["disposition"] == "RESUMABLE" and out[1]["recovery"]["candidate"] is None  # withdrawn
    assert out[1]["durable"]["identity"] == old


# ---- the operator surface --------------------------------------------------------------------------------------

def _sanitizer_preload():
    """The sanitizer runtimes the host links, by path: an uninstrumented interpreter loading sanitizer-built
    libraries must preload them (the native host links them itself)."""
    text = subprocess.run(["ldd", str(HOST)], capture_output=True, text=True, check=True).stdout
    paths = [line.split("=>", 1)[1].split("(", 1)[0].strip() for line in text.splitlines()
             if "=>" in line and line.strip().startswith(_SANITIZERS)]
    return {"LD_PRELOAD": " ".join(paths), "ASAN_OPTIONS": "detect_leaks=0"}


def _python_cli(home, *args, script=None):
    env = dict(os.environ, PYTHONPATH=str(REPO / "src"), PYTHONDONTWRITEBYTECODE="1")
    if SANITIZED:
        env.update(_sanitizer_preload())
    cmd = [sys.executable, "-m", "elpis.runtime.cli", "--home", str(home), *args]
    if script is not None:
        cmd += ["session", "--script", "-"]
    run = subprocess.run(cmd, input=json.dumps(script) if script is not None else None, capture_output=True,
                         text=True, env=env, cwd="/", timeout=300)
    return run.returncode, [json.loads(line) for line in run.stdout.splitlines() if line.strip()]


def test_the_native_host_answers_exactly_as_the_python_operator_interface(tmp_path):
    # The Python deployment pins its own library set (K1 FMS included) and publishes to continuity; the composite's
    # identities are the authority-independent part, so they must agree byte for byte.
    py_staging = tmp_path / "py-build"
    for rel in _LIBS + ("native/ECS_G/libelpis_ecsg_k1_fms.so",):
        (py_staging / rel).parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(BUILD / rel, py_staging / rel)
    py_home = tmp_path / "py-home"
    assert _python_cli(py_home, "init", "--native-build", str(py_staging))[0] == 0
    rc, created = _python_cli(py_home, "create", "--seed", "7")
    assert rc == 0, created
    script = [{"op": "query", "x": X}, LEARN, {"op": "query", "x": X}, EVIDENCE, LOOKUP, {"op": "live"}]
    rc, py = _python_cli(py_home, script=script)
    assert rc == 0 and all(o["ok"] for o in py), py
    home, initial = deploy(tmp_path)
    assert initial == created[0]["identity"]
    rc, out = host(home, script=[SOURCE] + script)
    assert rc == 0 and all(o["ok"] for o in out), out
    native = out[1:]
    for a, b in zip(py, native):
        assert a["op"] == b["op"]
    assert native[0]["values"] == py[0]["values"] and native[0]["values_sha256"] == py[0]["values_sha256"]
    for key in ("durability", "experiences", "rows", "steps", "epoch", "generation", "grant", "identity"):
        assert native[1][key] == py[1][key], key
    assert native[2]["values"] == py[2]["values"] and native[2]["values"] != native[0]["values"]
    for key in ("durability", "k1_steps", "generation", "entries", "identity"):
        assert native[3][key] == py[3][key], key
    for key in ("found", "answer", "matches", "conflicts", "contradicted", "uncertain", "state_identity"):
        assert native[4][key] == py[4][key], key
    assert native[4]["answer"] == node("engineer")
    assert native[5]["live"]["identity"] == py[5]["live"]["identity"]
    assert native[5]["live"]["k1_identity"] == py[5]["live"]["k1_identity"]
    assert native[5]["live"]["memory_identity"] == py[5]["live"]["memory_identity"]


def test_a_session_publishes_and_a_new_process_resumes_it_exactly(tmp_path):
    home, initial = deploy(tmp_path)
    rc, out = host(home, script=[SOURCE, {"op": "query", "x": X}, LEARN, {"op": "query", "x": X}, EVIDENCE, LOOKUP,
                                 {"op": "live"}, {"op": "publish"}, {"op": "status"}, {"op": "authority"}])
    assert rc == 0 and all(o["ok"] for o in out), out
    _, _, learned, after, recorded, found, live, published, status, authority = out
    assert learned["durability"] == "VOLATILE" and recorded["durability"] == "VOLATILE" and recorded["k1_steps"] == 0
    assert live["durable"]["identity"] == initial != live["live"]["identity"]
    assert published["ticket"]["acknowledged"] and published["durable"]["identity"] == live["live"]["identity"]
    assert status["recovery"]["disposition"] == "RESUMABLE" and status["fault"] is None
    assert status["footprint"]["capabilities"] == 0           # no memory capability is ever set (LEARN unqualified)
    assert authority["authority"]["evidence_count"] == 1      # the publication recorded the evidence summary
    assert [e["kind"] for e in authority["entities"]] == ["OPERATOR", "RUNTIME", "SOURCE"]
    key = recorded["entries"][0]["relation_key"]
    rc, again = host(home, script=[{"op": "query", "x": X}, LOOKUP, {"op": "live"}, {"op": "slot", "key": key}])
    assert rc == 0, again
    assert again[0]["values"] == after["values"] and again[1]["answer"] == node("engineer")
    assert again[2]["live"]["identity"] == again[2]["durable"]["identity"] == published["durable"]["identity"]
    slot = again[3]
    assert slot["found"] and slot["assert_count"] == 1 and slot["last_authority"] == 1
    assert slot["hyperedge"] == slot["structure_ref"] == recorded["entries"][0]["hyperedge"]
    assert {p["node"] for p in slot["participants"]} == {node("alice"), node("acme"), node("engineer")}


def test_unpublished_work_is_lost_at_exit_and_the_host_says_so(tmp_path):
    home, initial = deploy(tmp_path)
    rc, out = host(home, "learn", "--x", json.dumps(X), "--y", "[0.1,-0.2,0.05,0.3]", "--schedule", "[[4,2]]",
                   "--rate", "0.002", "--grant", "test operator")
    assert rc == 0 and out[0]["durability"] == "VOLATILE" and out[0]["identity"] != initial
    rc, out = host(home, "status")
    assert out[0]["live"]["identity"] == out[0]["durable"]["identity"] == initial      # lost, as reported
    rc, out = host(home, script=[LEARN, {"op": "close"}, {"op": "live"}])
    assert out[1]["volatile_work_discarded"] is True and out[2]["live"]["identity"] == initial
    rc, out = host(home, "learn", "--x", json.dumps(X), "--y", "[0.1,-0.2,0.05,0.3]", "--schedule", "[[4,2]]",
                   "--rate", "0.002", "--grant", "test operator", "--publish")
    assert rc == 0 and out[1]["ticket"]["acknowledged"]
    assert host(home, "status")[1][0]["durable"]["identity"] == out[0]["identity"] != initial


def test_refusals_are_reported_with_stable_codes_and_change_nothing(tmp_path):
    home, initial = deploy(tmp_path)
    source2 = dict(SOURCE, label="fixture source 2", source_authority=2, per_turn=1, scope_label="fixture scope 2")
    rc, out = host(home, script=[
        SOURCE, source2,
        dict(LEARN, grant=None, expect="LEARN_UNAUTHORIZED"),
        dict(LEARN, rate=0, expect="LEARNING_RATE"),
        dict(LEARN, grant="  ", expect="LEARN_AUTHORITY"),
        dict(LEARN, schedule=[[4, 0]], expect="STIMULUS"),
        dict(LEARN, y=[0.1], expect="STIMULUS"),
        {"op": "query", "x": [[1, 2, 3]], "expect": "CLI_INPUT"},
        dict(EVIDENCE, grant=None, expect="OBSERVATION_UNAUTHORIZED"),
        dict(EVIDENCE, sources=[2], expect="OBSERVATION_AUTHORITY"),
        evidence(1, authority=3, expect="RUNTIME_OBSERVATION_REFUSED"),       # not a source of the authority
        dict(evidence(2, authority=2), items=[evidence(2, authority=2)["items"][0], evidence(3, authority=2)["items"][0]],
             expect="RUNTIME_OBSERVATION_REFUSED"),                            # past source 2's bound of 1 per turn
        evidence(4, authority=5, expect="OBSERVATION_STRUCTURE"),            # structure: assertion authority 0..3
        {"op": "text", "input": "hello", "expect": "ECS_CODEC_UNQUALIFIED"},
        {"op": "no_such_op", "expect": "CLI_INPUT"},
        {"op": "reconcile", "action": "discard", "candidate": "0" * 64, "expect": "LINEAGE_BOUND"},
        {"op": "authority_reconcile", "kind": "ROLLBACK", "digest": "0" * 64, "expect": "ECS_C_NOT_RECONCILABLE"},
        {"op": "source_found", "label": "x", "source_authority": 9, "expect": "CLI_INPUT"},   # no scope
        # One present source row per memory source-authority identifier (freed only by reclamation).
        dict(SOURCE, label="fixture source 1 again", scope_label="fixture scope 3", expect="ECS_C_SCOPE_CONFLICT"),
        {"op": "live"}])
    assert rc == 0 and all(o.get("met", o["ok"]) for o in out), [o for o in out if not o.get("met", o["ok"])]
    assert out[-1]["live"]["identity"] == initial                              # nothing above changed the state
    # An unmet expectation exits 3; a malformed script or option exits 2 and runs nothing.
    rc, out = host(home, script=[dict(LEARN, expect="LEARN_UNAUTHORIZED")])
    assert rc == 3 and out[0]["met"] is False
    run = subprocess.run([str(HOST), "--home", str(home), "session", "--script", "-"], input='{"op": "query"\n',
                         capture_output=True, text=True, env=ENV)
    assert run.returncode == 2 and json.loads(run.stdout)["code"] == "CLI_INPUT"
    rc, out = host(home, "learn", "--bogus", "1")
    assert (rc, out[0]["code"]) == (2, "CLI_INPUT")
    rc, out = host(home, "text", "--input", "hello")
    assert (rc, out[0]["code"]) == (3, "ECS_CODEC_UNQUALIFIED")
    assert host(home, "status")[1][0]["durable"]["identity"] == initial


def test_operator_events_reach_the_authority_directly_and_through_the_held_runtime(tmp_path):
    home, _ = deploy(tmp_path)
    # Directly (no runtime open in this process): founding; then through the runtime that holds the authority.
    rc, out = host(home, script=[SOURCE, {"op": "query", "x": X}, evidence(1),
                                 {"op": "source_lifecycle", "source_authority": 1, "to": "DORMANT"},
                                 evidence(2, expect="RUNTIME_OBSERVATION_REFUSED"),
                                 {"op": "source_lifecycle", "source_authority": 1, "to": "ACTIVE"}, evidence(3),
                                 {"op": "source_version", "source_authority": 1, "expected": 2, "version": 3,
                                  "expect": "ECS_C_SOURCE_STALE"},              # a stale expectation: refused
                                 {"op": "source_version", "source_authority": 1, "expected": 1, "version": 2},
                                 {"op": "authority"}])
    assert rc == 0 and all(o.get("met", o["ok"]) for o in out), out
    source = [e for e in out[-1]["entities"] if e["kind"] == "SOURCE"][0]
    assert (source["lifecycle"], source["source_version"], source["source_authority"]) == ("ACTIVE", 2, 1)
    # A fresh process reads the same rows directly.
    rc, again = host(home, "authority")
    assert rc == 0 and again[0]["entities"] == out[-1]["entities"]
    assert again[0]["authority"]["held"] is False and again[0]["reconciliation"] is None


def test_entity_capacity_is_refused_before_any_write_and_reclamation_frees_a_row(tmp_path):
    home, _ = deploy(tmp_path, capacity=4)                    # the operator, the runtime and two sources
    s = lambda n: dict(SOURCE, label=f"source {n}", source_authority=n, scope_label=f"scope {n}")
    rc, out = host(home, script=[s(1), s(2), dict(s(3), expect="ECS_C_CAPACITY"), {"op": "authority"}])
    assert rc == 0 and all(o.get("met", o["ok"]) for o in out), out
    generation = out[-1]["authority"]["generation"]
    rc, out = host(home, script=[{"op": "query", "x": X},                      # the runtime holds the authority now
                                 dict(s(3), expect="ECS_C_CAPACITY"),          # still full, through the runtime
                                 {"op": "source_lifecycle", "source_authority": 2, "to": "TERMINATED"},
                                 {"op": "source_reclaim", "source_authority": 2}, s(3), {"op": "authority"}])
    assert rc == 0 and all(o.get("met", o["ok"]) for o in out), out
    # Terminate, reclaim, found and activate: four transitions; the refusals wrote nothing.
    assert out[-1]["authority"]["generation"] == generation + 4
    assert sorted(e["source_authority"] for e in out[-1]["entities"] if e["kind"] == "SOURCE") == [1, 3]


# ---- process death and the authority's reconciliation ---------------------------------------------------------

# Death while the testing build parks the durability worker at each step (docs/RUNTIME_CORE.md, "Durability crash
# matrix"): the durable identity a restart resumes, whether the K1 checkpoint keeps an unresolved candidate, and the
# authority's recovery disposition (never one that needs the operator: process death is an accident, not an attack).
CRASH = {1: ("old", False, "CLEAN"), 2: ("old", False, "CLEAN"), 3: ("old", False, "CLEAN"),
         4: ("old", True, "CLEAN"), 5: ("old", True, "CLEAN"), 6: ("old", True, "TRAILING_INCOMPLETE"),
         7: ("new", False, "INTERRUPTED_COMPLETED"), 8: ("new", False, "INTERRUPTED_COMPLETED"),
         9: ("new", False, "CLEAN")}


@pytest.mark.parametrize("step", sorted(CRASH))
def test_process_death_at_every_durable_step_recovers_exactly(tmp_path, step):
    home, old = deploy(tmp_path, testing=True)
    live = Live(home, [LEARN, {"op": "testing_durable_fault", "action": 2, "step": step, "arg": 100},
                       {"op": "publish", "wait": False}, {"op": "testing_wait_parked", "parked": step},
                       {"op": "testing_pause"}], tmp_path)
    live.until_pause()
    live.kill()
    new = live.out[0]["identity"]
    durable, candidate, disposition = CRASH[step]
    rc, out = host(home, script=[{"op": "authority"}, {"op": "status"}], testing=True)
    assert rc == 0, out
    authority, status = out
    assert authority["report"]["disposition"] == disposition and authority["reconciliation"] is None
    assert status["session_refusal"] is None
    assert status["durable"]["identity"] == (new if durable == "new" else old)
    if not candidate:
        assert status["recovery"]["disposition"] == "RESUMABLE" and status["recovery"]["candidate"] is None
        assert status["live"]["identity"] == status["durable"]["identity"]
        return
    # The checkpoint holds the unpublished capture: cognition is refused until the operator resolves exactly it.
    assert status["recovery"] == {"disposition": "CANDIDATE_UNRESOLVED", "authorized": old, "candidate": new,
                                  "envelope_bytes": 0}
    action = "adopt" if step % 2 else "discard"
    rc, out = host(home, script=[dict(LEARN, expect="LINEAGE_NOT_RESUMABLE"),
                                 {"op": "reconcile", "action": action, "candidate": "0" * 64,
                                  "expect": "CHECKPOINT_INVALID"},
                                 {"op": "reconcile", "action": action, "candidate": new}, {"op": "live"}],
                   testing=True)
    assert rc == 0 and all(o.get("met", o["ok"]) for o in out), out
    assert out[3]["live"]["identity"] == out[3]["durable"]["identity"] == (new if action == "adopt" else old)


# A publication that fails (the testing build's fault plans): the ticket and the host's refusal code, then what a
# restart finds. FAILED changes nothing durable; UNCERTAIN is resolved by restart to exactly one complete record.
FAILURES = {3: ("DURABLE_FAILED", "CHECKPOINT_IO", "RESUMABLE", "old", "CLEAN"),          # slot write fails
            4: ("DURABLE_FAILED", "CHECKPOINT_IO", "RESUMABLE", "old", "CLEAN"),          # slot torn, then fails
            5: ("DURABLE_FAILED", "CHECKPOINT_IO", "RESUMABLE", "old", "CLEAN"),          # slot sync fails
            6: ("DURABLE_FAILED", "CONTINUITY_PUBLICATION_REFUSED", "RESUMABLE", "old", "CLEAN"),   # record write
            7: ("DURABLE_FAILED", "CONTINUITY_PUBLICATION_REFUSED", "RESUMABLE", "old", "TRAILING_INCOMPLETE"),
            8: ("DURABLE_UNCERTAIN", "CONTINUITY_PUBLICATION_UNCERTAIN", "CANDIDATE_UNRESOLVED", "old", "CLEAN"),
            9: ("DURABLE_UNCERTAIN", "CONTINUITY_PUBLICATION_UNCERTAIN", "RESUMABLE", "new",
                "INTERRUPTED_COMPLETED"),                                                 # sync failed, durable
            10: ("DURABLE_FAILED", "CHECKPOINT_IO", "RESUMABLE", "old", "CLEAN")}         # read-back differs


@pytest.mark.parametrize("action", sorted(FAILURES))
def test_a_failed_publication_is_refused_with_its_ticket_and_restart_resolves_it(tmp_path, action):
    code, cause, disposition, durable, report = FAILURES[action]
    home, old = deploy(tmp_path, testing=True)
    rc, out = host(home, script=[LEARN, {"op": "testing_durable_fault", "action": action, "step": 0, "arg": 64},
                                 {"op": "publish"}, {"op": "publish"}], testing=True)
    new = out[0]["identity"]
    failed, again = out[2], out[3]
    assert (rc, failed["ok"], failed["code"]) == (3, False, code), failed
    assert failed["ticket"]["code"] == cause and failed["ticket"]["acknowledged"] is False
    assert failed["ticket"]["status"] == code.removeprefix("DURABLE_")
    if code == "DURABLE_FAILED":
        assert failed["durable"]["identity"] == old           # nothing durable changed
        assert again["ok"] and again["ticket"]["acknowledged"] and again["durable"]["identity"] == new
        return
    assert failed["durable"] is None and not again["ok"]     # unknown until restart: the lane refuses until reopen
    rc, out = host(home, script=[{"op": "authority"}, {"op": "status"}], testing=True)
    assert out[0]["report"]["disposition"] == report and out[0]["reconciliation"] is None
    assert out[1]["recovery"]["disposition"] == disposition
    assert out[1]["durable"]["identity"] == (new if durable == "new" else old)


def _slot_bytes(home):
    return [(home / "state" / f"authority.{c}").read_bytes() for c in "ab"]


def _write_slots(home, slots):
    for c, data in zip("ab", slots):
        with open(home / "state" / f"authority.{c}", "r+b") as f:
            f.write(data)


def test_copied_back_records_need_the_operators_exact_reconciliation(tmp_path):
    home, _ = deploy(tmp_path)
    rc, out = host(home, script=[LEARN, {"op": "publish"}])
    first = out[0]["identity"]
    saved = _slot_bytes(home)
    rc, out = host(home, script=[LEARN, {"op": "publish"}])
    second = out[0]["identity"]
    _write_slots(home, saved)                                  # genuine, authenticated, older: a rollback
    rc, out = host(home, script=[{"op": "status"}, {"op": "authority"}, dict(LEARN, expect="RUNTIME_AUTHORITY"),
                                 {"op": "query", "x": X, "expect": "RUNTIME_AUTHORITY"}])
    assert rc == 0 and all(o.get("met", o["ok"]) for o in out[2:]), out
    status, authority = out[0], out[1]
    assert status["session_refusal"]["code"] == "RUNTIME_AUTHORITY"
    assert authority["report"]["disposition"] == "ROLLBACK" and authority["report"]["mode"] == "ROLLBACK"
    reconciliation = authority["reconciliation"]
    assert reconciliation["kind"] == "ROLLBACK"
    rc, out = host(home, script=[
        {"op": "authority_reconcile", "kind": "ROLLBACK", "digest": "1" * 64, "expect": "AUTHORITY_RECONCILE_MISMATCH"},
        {"op": "authority_reconcile", "kind": "REPIN", "digest": reconciliation["digest"],
         "expect": "AUTHORITY_RECONCILE_MISMATCH"},
        {"op": "authority_reconcile", "kind": "ROLLBACK", "digest": reconciliation["digest"]},
        {"op": "status"}])
    assert rc == 0 and all(o.get("met", o["ok"]) for o in out), out
    assert out[2]["generation"] == reconciliation["abandoned_generation"] + 1
    status = out[3]
    # The authority continues from the record the operator accepted; the newer checkpoint slot is a candidate.
    assert status["durable"]["identity"] == first
    assert status["recovery"]["disposition"] == "CANDIDATE_UNRESOLVED" and status["recovery"]["candidate"] == second
    rc, out = host(home, script=[{"op": "reconcile", "action": "discard", "candidate": second}, {"op": "live"},
                                 {"op": "authority"}])
    assert rc == 0 and out[1]["live"]["identity"] == first and out[2]["authority"]["disposition"] == "CLEAN"


def test_a_corrupt_newest_record_is_never_silently_skipped(tmp_path):
    home, initial = deploy(tmp_path)
    rc, out = host(home, script=[LEARN, {"op": "publish"}])
    published = out[0]["identity"]
    rc, out = host(home, "authority")
    generations = out[0]["report"]["slot_generations"]
    newest = "ab"[generations.index(max(generations))]
    path = home / "state" / f"authority.{newest}"
    data = bytearray(path.read_bytes())
    data[200] ^= 0xFF
    path.write_bytes(bytes(data))
    rc, out = host(home, script=[{"op": "authority"}, dict(LEARN, expect="RUNTIME_AUTHORITY")])
    assert rc == 0 and out[1]["met"], out
    report, reconciliation = out[0]["report"], out[0]["reconciliation"]
    assert report["disposition"] == "CORRUPT_ACKNOWLEDGED" and reconciliation["kind"] == "ROLLBACK"
    # Only the operator's explicit acceptance of the predecessor reopens it; the capture it lost is a candidate.
    rc, out = host(home, script=[{"op": "authority_reconcile", "kind": "ROLLBACK", "digest": reconciliation["digest"]},
                                 {"op": "status"}])
    assert rc == 0 and out[1]["durable"]["identity"] == initial
    assert out[1]["recovery"]["candidate"] == published


# ---- boundedness ------------------------------------------------------------------------------------------------

def _round(tick0, turns):
    ops = []
    for t in range(turns):
        ops += [LEARN, evidence(tick0 + t), {"op": "publish"}]
    return ops


def _bounded(tmp_path, rounds, turns):
    home, _ = deploy(tmp_path)
    assert host(home, script=[SOURCE])[0] == 0
    baseline = files(home)
    assert sorted(baseline) == ["home/authority/deployment.key", "home/authority/witness.a",
                                "home/authority/witness.b", "home/catalog", "home/checkpoint/k1-checkpoint.a",
                                "home/checkpoint/k1-checkpoint.b", "home/host.cfg",
                                "home/lib/libelpis_ecs_c_authority.so", "home/lib/libelpis_ecs_composite.so",
                                "home/lib/libelpis_ecs_memory.so", "home/lib/libelpis_ecsg_k1.so",
                                "home/lib/libelpis_ecsg_math.so", "home/lib/libelpis_hacf_observation.so",
                                "home/lib/libelpis_runtime.so", "home/lineage.cfg", "home/state/authority.a",
                                "home/state/authority.b"], baseline
    work = None
    for r in range(rounds):
        rc, out = host(home, script=_round(1 + r * turns, turns))
        assert rc == 0 and all(o["ok"] for o in out), [o for o in out if not o["ok"]]
        assert files(home) == baseline                         # no file created, removed or resized
        rc, out = host(home, "authority")
        report = out[0]["report"]
        recovery = (report["reads"], report["read_bytes"], report["replays"])
        work = work or recovery
        assert recovery == work                                 # recovery work does not grow with history
        assert out[0]["authority"]["publications"] == 1 + (r + 1) * turns   # the anchor, then every publication
    return home


def test_the_home_is_a_fixed_file_set_over_many_publications_and_restarts(tmp_path):
    _bounded(tmp_path, rounds=4, turns=15)


def test_long_runs_keep_the_home_fixed(tmp_path):
    # STRESS lane (tests/lanes.py): 1,000 publications over 20 restarts.
    _bounded(tmp_path, rounds=20, turns=50)


# ---- containment ------------------------------------------------------------------------------------------------

# A contained host refuses a privileged process (as elpis_autonomous_run does): as root, these tests run the host as
# an unprivileged user (uid 65534) over a deployment that user owns.
UNPRIVILEGED = ["setpriv", "--reuid=65534", "--regid=65534", "--clear-groups", "--"] if os.geteuid() == 0 else []


@pytest.fixture
def user_dir(tmp_path):
    if not UNPRIVILEGED:
        yield tmp_path
        return
    root = Path(tempfile.mkdtemp(prefix="elpis-contained-"))
    os.chown(root, 65534, 65534)
    try:
        yield root
    finally:
        shutil.rmtree(root, ignore_errors=True)


def host_as(home, *args, script=None, testing=False, contained=False, stdout=subprocess.PIPE):
    """As `host`, as the deployment's (unprivileged) owner, optionally contained."""
    cmd = UNPRIVILEGED + [str(HOST_TESTING if testing else HOST), "--home", str(home)] + \
        (["--contained"] if contained else []) + list(args)
    text = None
    if script is not None:
        cmd += ["session", "--script", "-"]
        text = "\n".join(json.dumps(op) for op in script) + "\n"
    run = subprocess.run(cmd, input=text, stdout=stdout, stderr=subprocess.PIPE, text=True, env=ENV, cwd="/",
                         timeout=300)
    assert run.stderr == "", run.stderr
    lines = run.stdout.splitlines() if stdout is subprocess.PIPE else []
    return run.returncode, [json.loads(line) for line in lines if line.strip()]


def deploy_as(root, *, testing=True):
    staging = _staging(root, "build")
    if UNPRIVILEGED:
        for p in [staging, *staging.rglob("*")]:
            os.chown(p, 65534, 65534)
    home = root / "home"
    rc, out = host_as(home, "init", *_init_args(staging), testing=testing)
    assert rc == 0, out
    rc, out = host_as(home, "create", "--seed", "7", testing=testing)
    assert rc == 0, out
    shutil.rmtree(staging)
    return home, out[0]["identity"]


def probe(path, kind, offset=0):
    return {"op": "testing_write_probe", "path": str(path), "kind": kind, "offset": offset}


def test_a_contained_host_operates_and_writes_only_its_fixed_slots(user_dir):
    home, initial = deploy_as(user_dir)
    before = files(home)
    slots = [f for f in before if f.startswith(("home/state/", "home/checkpoint/", "home/authority/witness"))]
    assert len(slots) == 6
    largest = max(before[f] for f in slots)
    # Uncontained, the same probes create a file and write the configuration: containment is what refuses them.
    rc, out = host_as(home, script=[probe(home / "new-file", "create"), probe(home / "host.cfg", "write")],
                      testing=True)
    assert [o["errno"] for o in out] == [0, 0]
    rc, out = host_as(home, script=[
        SOURCE, LEARN, evidence(1), {"op": "publish"}, LOOKUP,
        probe(home / "new-file", "create"), probe(home / "state" / "authority.c", "create"),
        probe(home / "host.cfg", "write"), probe(home / "catalog", "write"),
        probe(home / "state" / "authority.a", "write"),                      # a granted slot, rewritten in place
        probe(home / "checkpoint" / "k1-checkpoint.a", "extend", largest),   # past every fixed slot
        {"op": "status"}], testing=True, contained=True)
    assert rc == 0 and all(o["ok"] for o in out), out
    published = out[3]["durable"]["identity"]
    assert out[4]["answer"] == node("engineer") and published != initial
    assert [o["errno"] for o in out if o["op"] == "testing_write_probe"] == \
        [errno.EACCES, errno.EACCES, errno.EACCES, errno.EACCES, 0, errno.EFBIG]
    assert files(home) == before                               # nothing created, removed or resized
    # A new contained process resumes exactly the published state.
    rc, again = host_as(home, script=[{"op": "live"}, LOOKUP], testing=True, contained=True)
    assert rc == 0 and again[0]["live"]["identity"] == again[0]["durable"]["identity"] == published
    assert again[1]["answer"] == node("engineer")


def test_containment_refuses_what_it_cannot_contain(user_dir, tmp_path):
    home, _ = deploy_as(user_dir)
    rc, out = host_as(home, "create", contained=True, testing=True)
    assert (rc, out[0]["code"]) == (2, "CLI_INPUT")            # provisioning is never contained
    sink = user_dir / "out.json"
    with open(sink, "w") as f:
        if UNPRIVILEGED:
            os.chown(sink, 65534, 65534)
        rc, _ = host_as(home, "status", contained=True, testing=True, stdout=f)
    assert rc == 4 and json.loads(sink.read_text())["code"] == "CONTAINMENT_REFUSED"   # a writable file as stdout
    slot = home / "checkpoint" / "k1-checkpoint.b"
    with open(slot, "ab") as f:                                # a slot that is not exactly its fixed file
        f.write(b"\0")
    rc, out = host_as(home, "status", contained=True, testing=True)
    assert (rc, out[0]["code"]) == (4, "CONTAINMENT_REFUSED") and "k1-checkpoint.b" in out[0]["detail"]
    if os.geteuid() == 0:                                      # a privileged process is never contained
        root_home, _ = deploy(tmp_path)
        rc, out = host(root_home, "--contained", "status")
        assert (rc, out[0]["code"]) == (4, "CONTAINMENT_REFUSED")
