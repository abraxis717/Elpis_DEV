"""The Python reference's process gate (elpis.runtime.environment; docs/ELPIS_MISSION.md, Amendment 3).

A Runtime opens only in a process that writes no bytecode cache, cannot be core-dumped and keeps its memory out of
swap; otherwise it refuses ``ENVIRONMENT_REFUSED`` before any session exists. The offline CLI refuses a process
started with bytecode writing enabled, and refuses to write anything onto storage standard output, even its refusal.
The swap cases run in child interpreters: locking every page of the test process itself would outlive the test.
"""
from __future__ import annotations

import ctypes
import os
from pathlib import Path
import resource
import shutil
import subprocess
import sys
import tempfile

import pytest

from elpis.runtime import Runtime
from elpis.runtime.composition import CompositionError
from elpis.runtime.environment import require_volatile_process, swap_areas

from ..conftest import native_build_dir, require_native_library, runtime_config

REPO = Path(__file__).resolve().parents[2]
SWAPS = "Filename\t\t\t\tType\t\tSize\t\tUsed\t\tPriority\n"
ACTIVE = SWAPS + "/swapfile                               file\t\t4194300\t\t0\t\t-2\n"


def _dumpable() -> int:
    libc = ctypes.CDLL(None)
    return libc.prctl(3, 0, 0, 0, 0)   # PR_GET_DUMPABLE


def test_a_runtime_opens_only_once_the_process_cannot_dump_or_write_bytecode(monkeypatch):
    library = Path(require_native_library("elpis_runtime"))
    monkeypatch.setattr(sys, "dont_write_bytecode", False)
    runtime = Runtime(runtime_config(library))
    with pytest.raises(CompositionError) as refused:
        runtime.open()
    assert refused.value.code == "ENVIRONMENT_REFUSED" and "bytecode" in str(refused.value)
    assert runtime.fault is not None                # nothing was opened: the runtime is not live
    monkeypatch.setattr(sys, "dont_write_bytecode", True)
    with Runtime(runtime_config(library)) as opened:
        c = opened.containment
        assert c.core_limit == (0, 0) and c.dumpable == 0 and not c.bytecode_writes
        assert resource.getrlimit(resource.RLIMIT_CORE) == (0, 0) and _dumpable() == 0
        assert c.swap_areas == swap_areas() and c.memory_locked is (c.swap_areas > 0)


def test_the_swap_policy_is_read_and_an_unknown_policy_refuses(tmp_path):
    none, active = tmp_path / "none", tmp_path / "active"
    none.write_text(SWAPS)
    active.write_text(ACTIVE)
    assert swap_areas(none) == 0 and swap_areas(active) == 1
    for unknown in (tmp_path / "absent", tmp_path / "empty"):
        if unknown.name == "empty":
            unknown.write_text("")
        with pytest.raises(CompositionError) as refused:
            require_volatile_process(proc_swaps=unknown)
        assert refused.value.code == "ENVIRONMENT_REFUSED" and "swap policy is unknown" in str(refused.value)


_PROBE = """
import json, sys
from elpis.runtime.environment import require_volatile_process
from elpis.runtime.composition import CompositionError
try:
    c = require_volatile_process(proc_swaps=sys.argv[1])
    status = [l for l in open("/proc/self/status") if l.startswith("VmLck:")][0].split()[1]
    print(json.dumps({"ok": True, "locked": c.memory_locked, "vmlck_kib": int(status)}))
except CompositionError as exc:
    print(json.dumps({"ok": False, "code": exc.code, "detail": str(exc)}))
"""


def _probe(swaps: Path, *prefix: str):
    env = {"PATH": os.environ.get("PATH", "/usr/bin:/bin"), "PYTHONPATH": str(REPO / "src"),
           "PYTHONDONTWRITEBYTECODE": "1"}
    run = subprocess.run([*prefix, sys.executable, "-B", "-c", _PROBE, str(swaps)], capture_output=True,
                         text=True, env=env, cwd="/", timeout=120)
    assert run.returncode == 0, run.stderr
    import json
    return json.loads(run.stdout)


def test_active_swap_locks_the_process_memory_or_refuses():
    shared = Path(tempfile.mkdtemp(prefix="elpis-swaps-"))   # readable by the unprivileged probe below
    try:
        shared.chmod(0o755)
        active = shared / "active"
        active.write_text(ACTIVE)
        active.chmod(0o644)
        _check_swap_policy(active)
    finally:
        shutil.rmtree(shared)


def _check_swap_policy(active: Path):
    out = _probe(active)
    if os.geteuid() == 0 or resource.getrlimit(resource.RLIMIT_MEMLOCK)[0] == resource.RLIM_INFINITY:
        assert out["ok"] and out["locked"] and out["vmlck_kib"] > 0, out   # locked: nothing can be paged out
    else:
        assert not out["ok"] and out["code"] == "ENVIRONMENT_REFUSED", out
    setpriv = shutil.which("setpriv")
    if os.geteuid() == 0 and setpriv:
        # Without the IPC_LOCK capability and with a finite lock limit, the lock cannot hold the process: refused,
        # never attempted (a partial lock would turn later allocations into failures).
        unprivileged = _probe(active, setpriv, "--reuid=65534", "--regid=65534", "--clear-groups",
                              "--inh-caps=-all", "--bounding-set=-all", "prlimit", "--memlock=8388608:8388608", "--")
        assert not unprivileged["ok"] and "memory lock limit" in unprivileged["detail"], unprivileged


def _cli_env(**extra):
    env = {"PATH": os.environ.get("PATH", "/usr/bin:/bin"), "PYTHONPATH": str(REPO / "src")}
    env.update(extra)
    return env


def test_the_cli_refuses_an_interpreter_writing_bytecode_and_storage_stdout(tmp_path):
    build = native_build_dir()
    # Bytecode writing enabled (redirected into the test's scratch directory, so the refusal costs the repository
    # nothing): refused before any operation.
    run = subprocess.run([sys.executable, "-m", "elpis.runtime.cli", "--native-build", str(build), "catalog"],
                         capture_output=True, text=True, timeout=300, cwd="/",
                         env=_cli_env(PYTHONPYCACHEPREFIX=str(tmp_path / "pycache")))
    assert run.returncode == 4 and '"ENVIRONMENT_REFUSED"' in run.stdout and "-B" in run.stdout, run
    # Standard output on storage: refused, and not a byte written there (the reason went to standard error).
    sink = tmp_path / "stdout"
    with open(sink, "w") as f:
        run = subprocess.run([sys.executable, "-B", "-m", "elpis.runtime.cli", "--native-build", str(build),
                              "catalog"], stdout=f, stderr=subprocess.PIPE, text=True, timeout=300, cwd="/",
                             env=_cli_env())
    assert run.returncode == 4 and sink.read_text() == "" and "ENVIRONMENT_REFUSED" in run.stderr
    # Standard error on storage as well: still refused, and both files stay empty.
    err = tmp_path / "stderr"
    with open(sink, "w") as f, open(err, "w") as g:
        run = subprocess.run([sys.executable, "-B", "-m", "elpis.runtime.cli", "--native-build", str(build),
                              "catalog"], stdout=f, stderr=g, timeout=300, cwd="/", env=_cli_env())
    assert run.returncode == 4 and sink.read_text() == "" and err.read_text() == ""
    # Standard error alone on storage (standard output a pipe): refused, reported on standard output.
    with open(err, "w") as g:
        run = subprocess.run([sys.executable, "-B", "-m", "elpis.runtime.cli", "--native-build", str(build),
                              "catalog"], stdout=subprocess.PIPE, stderr=g, text=True, timeout=300, cwd="/",
                             env=_cli_env())
    assert run.returncode == 4 and '"ENVIRONMENT_REFUSED"' in run.stdout and err.read_text() == ""
