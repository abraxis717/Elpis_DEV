#!/usr/bin/env python3
"""research/volatile_runtime_v0: zero filesystem writes on the volatile ECS_C session's tested paths.

    python3 -I research/volatile_runtime_v0/zero_write.py --repo . --build build \
        --out research/volatile_runtime_v0/results

Evidence for phases V1 and V2 of the volatile-native migration (docs/ELPIS_MISSION.md, Amendment 3;
docs/VOLATILE_RUNTIME_CENSUS.md, section 6). It covers the session's own test paths only, not any composed runtime
path. It needs root (to build a read-only mount namespace and drop to uid 65534), Linux x86-64, strace, unshare,
setpriv, a C compiler and Landlock ABI >= 5. It fails closed if any of them is missing.

Tested paths: built binaries. Nothing is rebuilt here except the probe.
  P1 test_ecs_c_session_abi         the session's C ABI end to end
  P2 test_ecs_c_relational_session  the relational view over the session and the real ECS_C memory
  P3 cargo tests                    the session crate's release test binary (ECS_C.cargo_test_session)
  P4 nomem_probe                    memory exhaustion under RLIMIT_AS: NOMEM, nothing spilled

Each path runs in three modes:
  plain     no tracer: exit status, wall time, peak RSS of that process alone (wait4).
  observed  as root, so any write would be permitted, under strace -f -y. The claim: no write is even attempted.
  confined  inside a private mount namespace, with every mount but /proc remounted read-only (topmost mount per
            mount point verified). It runs as uid/gid 65534 with no groups and no capabilities, under
            elpis_autonomous_run with no grant (Landlock: no write, create, remove, rename, truncate or device ioctl;
            seccomp; descriptors >= 3 closed), with RLIMIT_CORE 0 and an empty environment, under strace -f -y.

A run passes when it exits 0 with zero write-class events on storage. strace writes its log into a pipe that this
script drains, so no file is created even for the trace.

Negative controls: real writes, each detected as an event. They show that the detector is not blind and that each
containment layer refuses on its own:
  root unconfined     creation, rename, sync and removal all succeed
  uid 65534 alone     a write into a directory that uid 65534 owns succeeds: the uid layer alone is not containment
  Landlock alone      the same write is refused EACCES
  full confinement    the same write is refused EROFS

The results carry no host-local path: the repository and the build tree appear as <repo> and <build>.
"""
from __future__ import annotations

import argparse
import ctypes
import hashlib
import json
import os
import platform
import re
import shutil
import subprocess
import sys
import threading
import time
from pathlib import Path

SCHEMA = "elpis.volatile-runtime-v0.zero-write.v1"
NOBODY = 65534

TRACED = ("open", "openat", "openat2", "creat", "write", "writev", "pwrite64", "pwritev", "pwritev2", "rename",
          "renameat", "renameat2", "unlink", "unlinkat", "mkdir", "mkdirat", "rmdir", "truncate", "ftruncate",
          "fallocate", "fsync", "fdatasync", "sync", "syncfs", "sync_file_range", "msync", "link", "linkat", "symlink",
          "symlinkat", "mknod", "mknodat", "chmod", "fchmod", "fchmodat", "chown", "fchown", "lchown", "fchownat",
          "utimensat", "setxattr", "lsetxattr", "fsetxattr", "removexattr", "lremovexattr", "fremovexattr", "mmap",
          "copy_file_range", "sendfile", "splice", "io_uring_setup", "name_to_handle_at", "open_by_handle_at")
ALWAYS_WRITE = set(TRACED) - {"open", "openat", "openat2", "creat", "write", "writev", "pwrite64", "pwritev",
                              "pwritev2", "mmap"}
WRITE_FLAGS = re.compile(r"\bO_(?:WRONLY|RDWR|CREAT|TRUNC|APPEND|TMPFILE)\b")
CALL = re.compile(r"^(\d+)\s+(\w+)\((.*)\)\s+=\s+(-?\d+|\?|0x[0-9a-f]+)(?:\s+(\w+))?")
NOT_STORAGE = ("pipe:", "socket:", "anon_inode:", "/dev/pts/", "/dev/null", "/dev/tty", "/memfd:")

CONFINE = r'''
set -eu
fd="$1"; launcher="$2"; ro="$3"; shift 3
if [ "$ro" = 1 ]; then
  awk '{print $5}' /proc/self/mountinfo | sort -u | while read -r m; do
    case "$m" in /proc|/proc/*) continue ;; esac
    mount -o remount,bind,ro "$m" 2>/dev/null || echo "remount failed: $m" >&2
  done
  rw=$(awk '{o[$5] = $6} END {for (m in o) if (o[m] !~ /^ro(,|$)/ && m !~ /^\/proc(\/|$)/) print m}' \
       /proc/self/mountinfo | sort | tr '\n' ' ')
  if [ -n "$rw" ]; then echo "VOLATILE_V0_CONFINEMENT_REFUSED: writable mounts remain: $rw" >&2; exit 97; fi
fi
ulimit -c 0
if [ "$launcher" = none ]; then
  exec strace -f -qq -y -s 96 -e trace="$TRACED" -e signal=none -o "/proc/self/fd/$fd" -- \
    setpriv --reuid=65534 --regid=65534 --clear-groups --inh-caps=-all --bounding-set=-all -- "$@"
fi
exec strace -f -qq -y -s 96 -e trace="$TRACED" -e signal=none -o "/proc/self/fd/$fd" -- \
  setpriv --reuid=65534 --regid=65534 --clear-groups --inh-caps=-all --bounding-set=-all -- "$launcher" -- "$@"
'''


class Fail(SystemExit):
    def __init__(self, message: str):
        super().__init__(f"VOLATILE_V0_REFUSED: {message}")


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


class Sanitizer:
    def __init__(self, repo: Path, build: Path):
        self.pairs = sorted(((str(build), "<build>"), (str(repo), "<repo>")), key=lambda p: -len(p[0]))

    def __call__(self, text: str) -> str:
        for raw, alias in self.pairs:
            text = text.replace(raw, alias)
        return text


def calls(log: str):
    """The complete call lines of a strace -f log: an interrupted call ("<unfinished ...>") is joined with its
    "<... name resumed>" continuation; a call its process never returned from is kept, with the result "?"."""
    pending: dict[str, str] = {}
    for raw in log.splitlines():
        m = re.match(r"^(\d+)\s+(.*)$", raw.strip())
        if not m:
            continue
        pid, rest = m.groups()
        if rest.endswith("<unfinished ...>"):
            pending[pid] = rest[: -len("<unfinished ...>")].rstrip()
            continue
        resumed = re.match(r"^<\.\.\. \w+ resumed>\s?(.*)$", rest)
        if resumed:
            if pid not in pending:
                continue
            rest = pending.pop(pid) + resumed.group(1)
        yield f"{pid} {rest}"
    for pid, rest in pending.items():
        yield f"{pid} {rest}) = ?"


def write_events(log: str) -> list[str]:
    """Write-class events against storage in a strace -f -y log, attempted or successful."""
    events = []
    for raw in calls(log):
        m = CALL.match(raw.strip())
        if not m:
            continue
        name, args = m.group(2), m.group(3)
        if name in ALWAYS_WRITE:
            events.append(raw.strip())
        elif name in ("open", "openat", "openat2", "creat"):
            path = re.search(r'"([^"]*)"', args)
            if (name == "creat" or WRITE_FLAGS.search(args)) and not (
                    path and path.group(1).startswith(("/dev/null", "/dev/pts/", "/dev/tty"))):
                events.append(raw.strip())
        elif name in ("write", "writev", "pwrite64", "pwritev", "pwritev2"):
            target = args.split(",", 1)[0]
            target = target.split("<", 1)[1].rstrip(">") if "<" in target else ""
            if target and not target.startswith(NOT_STORAGE):
                events.append(raw.strip())
        elif name == "mmap":
            parts = [a.strip() for a in args.split(",")]
            if len(parts) >= 5 and "PROT_WRITE" in parts[2] and "MAP_SHARED" in parts[3] and \
                    not parts[4].startswith("-1"):
                target = parts[4].split("<", 1)[1].rstrip(">") if "<" in parts[4] else ""
                if not target.startswith(NOT_STORAGE):
                    events.append(raw.strip())
    return events


def syscall_counts(log: str) -> dict[str, int]:
    counts: dict[str, int] = {}
    for raw in calls(log):
        m = CALL.match(raw.strip())
        if m:
            counts[m.group(2)] = counts.get(m.group(2), 0) + 1
    return dict(sorted(counts.items()))


def traced(argv: list[str], mode: str, launcher: Path | None, timeout: int) -> tuple[subprocess.CompletedProcess, str,
                                                                                         float]:
    """Run argv under strace. mode: observed, confined, uid_only (no launcher, writable mounts) or landlock_only."""
    r, w = os.pipe()
    chunks: list[bytes] = []

    def drain():
        while True:
            block = os.read(r, 1 << 16)
            if not block:
                return
            chunks.append(block)

    thread = threading.Thread(target=drain)
    thread.start()
    env = {"PATH": "/usr/bin:/bin", "TRACED": ",".join(TRACED)}
    if mode == "observed":
        cmd = ["strace", "-f", "-qq", "-y", "-s", "96", "-e", "trace=" + ",".join(TRACED), "-e", "signal=none",
               "-o", f"/proc/self/fd/{w}", "--", *argv]
    else:
        ro = "1" if mode == "confined" else "0"
        use = "none" if mode == "uid_only" else str(launcher)
        cmd = ["unshare", "--mount", "--propagation", "private", "--", "/bin/sh", "-c", CONFINE, "confine", str(w),
               use, ro, *argv]
    start = time.monotonic()
    try:
        run = subprocess.run(cmd, stdin=subprocess.DEVNULL, capture_output=True, text=True, pass_fds=(w,), env=env,
                             cwd="/", timeout=timeout)
    finally:
        os.close(w)
        thread.join()
        os.close(r)
    return run, b"".join(chunks).decode("utf-8", "replace"), time.monotonic() - start


def plain(argv: list[str], timeout: int) -> dict:
    start = time.monotonic()
    proc = subprocess.Popen(argv, stdin=subprocess.DEVNULL, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                            env={"PATH": "/usr/bin:/bin"}, cwd="/")
    out: dict[str, bytes] = {}
    readers = [threading.Thread(target=lambda k=k, s=s: out.__setitem__(k, s.read()))
               for k, s in (("stdout", proc.stdout), ("stderr", proc.stderr))]
    for t in readers:
        t.start()
    deadline = start + timeout
    while True:
        pid, status, usage = os.wait4(proc.pid, os.WNOHANG)
        if pid:
            break
        if time.monotonic() > deadline:
            proc.kill()
            raise Fail(f"timeout: {argv[0]}")
        time.sleep(0.01)
    for t in readers:
        t.join()
    proc.returncode = os.waitstatus_to_exitcode(status)
    return {"exit": proc.returncode, "seconds": round(time.monotonic() - start, 3), "peak_rss_kib": usage.ru_maxrss}


def tail(text: str, clean: Sanitizer, lines: int = 6) -> list[str]:
    return [clean(line) for line in text.strip().splitlines()[-lines:]]


def landlock_abi() -> int:
    libc = ctypes.CDLL(None, use_errno=True)
    libc.syscall.restype = ctypes.c_long
    return int(libc.syscall(444, None, 0, 1))      # landlock_create_ruleset(NULL, 0, LANDLOCK_CREATE_RULESET_VERSION)


def first_line(cmd: list[str]) -> str:
    try:
        out = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
    except (OSError, subprocess.TimeoutExpired):
        return "unavailable"
    return (out.stdout or out.stderr).strip().splitlines()[0] if (out.stdout or out.stderr).strip() else "unavailable"


def mount_of(target: str) -> str:
    best = ("", "")
    for line in Path("/proc/self/mountinfo").read_text().splitlines():
        fields = line.split()
        point, fstype = fields[4], fields[fields.index("-") + 1]
        if (target == point or target.startswith(point.rstrip("/") + "/")) and len(point) >= len(best[0]):
            best = (point, fstype)
    return best[1]


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--repo", type=Path, required=True)
    ap.add_argument("--build", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--timeout", type=int, default=900)
    args = ap.parse_args(argv)
    repo, build, out = args.repo.resolve(), args.build.resolve(), args.out.resolve()
    clean = Sanitizer(repo, build)

    # Fail closed on the prerequisites.
    if platform.system() != "Linux" or platform.machine() != "x86_64":
        raise Fail("Linux x86-64 is required (elpis_autonomous_run's seccomp table is x86-64 only)")
    if os.geteuid() != 0:
        raise Fail("root is required to build the read-only mount namespace and to drop to uid 65534")
    for tool in ("strace", "unshare", "setpriv", "nm", "cc", "git", "mount", "awk"):
        if shutil.which(tool) is None:
            raise Fail(f"{tool} is not available")
    abi = landlock_abi()
    if abi < 5:
        raise Fail(f"Landlock ABI >= 5 is required (found {abi})")
    session_lib = build / "native/ECS_C/session/libelpis_ecs_c_session.so"
    launcher = build / "native/runtime/elpis_autonomous_run"
    p1 = build / "native/ECS_C/session/test_ecs_c_session_abi"
    p2 = build / "native/ECS_C/test_ecs_c_relational_session"
    p3s = sorted((build / "native/ECS_C/session/cargo-test/release/deps").glob("elpis_ecs_c_session-*"))
    p3s = [p for p in p3s if p.is_file() and os.access(p, os.X_OK) and p.suffix == ""]
    for path in (session_lib, launcher, p1, p2):
        if not path.is_file():
            raise Fail(f"{clean(str(path))} is not built")
    if len(p3s) != 1:
        raise Fail(f"exactly one cargo test binary expected, found {len(p3s)} (build ECS_C.cargo_test_session)")
    p3 = p3s[0]

    # The probe: built here, into the build tree, from its committed source.
    probe_dir = build / "research/volatile_runtime_v0"
    probe_dir.mkdir(parents=True, exist_ok=True)
    p4 = probe_dir / "nomem_probe"
    subprocess.run(["cc", "-std=c11", "-O2", "-Wall", "-Wextra", "-Werror",
                    "-I", str(repo / "native/ECS_C/session/include"),
                    str(repo / "research/volatile_runtime_v0/nomem_probe.c"), str(session_lib),
                    f"-Wl,-rpath,{session_lib.parent}", "-o", str(p4)], check=True)

    # Static: what the session library can call at all.
    undefined = subprocess.run(["nm", "-D", "--undefined-only", str(session_lib)], capture_output=True, text=True,
                               check=True).stdout
    imports = sorted({line.split()[-1].split("@")[0] for line in undefined.splitlines() if line.split()})
    defined = subprocess.run(["nm", "-D", "--defined-only", str(session_lib)], capture_output=True, text=True,
                             check=True).stdout
    exports = sorted({line.split()[-1] for line in defined.splitlines() if len(line.split()) >= 3})
    lib = ctypes.CDLL(str(session_lib))
    lib.elpis_ecs_c_session_bytes.restype = ctypes.c_uint64
    lib.elpis_ecs_c_session_bytes.argtypes = [ctypes.c_uint32]
    footprint = {str(c): lib.elpis_ecs_c_session_bytes(c) for c in (1, 2, 16, 64, 256, 1024, 4096, 4097)}

    paths = {"P1": [str(p1)], "P2": [str(p2)], "P3": [str(p3), "--test-threads=1"], "P4": [str(p4)]}
    runs = {}
    for key, cmd in paths.items():
        entry = {"argv": [clean(a) for a in cmd], "sha256": sha256(Path(cmd[0])), "plain": plain(cmd, args.timeout)}
        for mode in ("observed", "confined"):
            run, log, seconds = traced(cmd, mode, launcher, args.timeout)
            events = write_events(log)
            entry[mode] = {"exit": run.returncode, "seconds": round(seconds, 3), "traced_calls": syscall_counts(log),
                           "write_events": [clean(e) for e in events], "stdout_tail": tail(run.stdout, clean),
                           "stderr_tail": tail(run.stderr, clean)}
        runs[key] = entry

    # Negative controls.
    control_dir = build / "research/volatile_runtime_v0/control"
    if control_dir.exists():
        shutil.rmtree(control_dir)
    control_dir.mkdir(parents=True)
    os.chown(control_dir, NOBODY, NOBODY)
    controls = {}
    target = control_dir / "probe"
    observed, log, _ = traced(["/bin/sh", "-c", f'echo x > "{target}.a" && mv "{target}.a" "{target}.b" && sync && '
                                                 f'rm "{target}.b"'], "observed", launcher, args.timeout)
    controls["root_unconfined"] = {"exit": observed.returncode, "write_events": [clean(e) for e in write_events(log)]}
    for mode in ("uid_only", "landlock_only", "confined"):
        name = mode if mode != "confined" else "full_confinement"
        run, log, _ = traced(["/usr/bin/touch", str(control_dir / f"{name}.file")], mode, launcher, args.timeout)
        controls[name] = {"exit": run.returncode, "write_events": [clean(e) for e in write_events(log)],
                          "stderr_tail": tail(run.stderr, clean, 2),
                          "file_created": (control_dir / f"{name}.file").exists()}
    shutil.rmtree(control_dir)

    # Verdict.
    problems = []
    for key, entry in runs.items():
        if entry["plain"]["exit"] != 0:
            problems.append(f"{key} plain exit {entry['plain']['exit']}")
        for mode in ("observed", "confined"):
            if entry[mode]["exit"] != 0:
                problems.append(f"{key} {mode} exit {entry[mode]['exit']}")
            if entry[mode]["write_events"]:
                problems.append(f"{key} {mode}: {len(entry[mode]['write_events'])} write-class events")
    root = controls["root_unconfined"]
    names = {CALL.match(e).group(2) for e in root["write_events"] if CALL.match(e)}
    if not (root["exit"] == 0 and {"rename", "renameat", "renameat2"} & names and {"unlink", "unlinkat"} & names
            and "sync" in names and any(WRITE_FLAGS.search(e) for e in root["write_events"])):
        problems.append("control root_unconfined: the detector missed a real write")
    if not (controls["uid_only"]["exit"] == 0 and controls["uid_only"]["file_created"]
            and controls["uid_only"]["write_events"]):
        problems.append("control uid_only: the write into the uid's own directory did not happen as expected")
    for name, errno_name in (("landlock_only", "EACCES"), ("full_confinement", "EROFS")):
        c = controls[name]
        if c["exit"] == 0 or c["file_created"] or not any(errno_name in e for e in c["write_events"]):
            problems.append(f"control {name}: expected a refused, recorded attempt ({errno_name})")
    session_imports = set(imports) - {"_ITM_deregisterTMCloneTable", "_ITM_registerTMCloneTable", "__cxa_finalize",
                                      "__gmon_start__"}
    allowed = {"abort", "free", "getrandom", "malloc", "memcpy", "memmove", "memset", "memcmp", "bcmp",
               "posix_memalign"}
    if not session_imports <= allowed:
        problems.append(f"session library imports {sorted(session_imports - allowed)}")

    head = subprocess.run(["git", "-C", str(repo), "rev-parse", "HEAD"], capture_output=True, text=True,
                          check=True).stdout.strip()
    dirty = subprocess.run(["git", "-C", str(repo), "status", "--porcelain", "--", "native",
                            "research/volatile_runtime_v0", "tests/boundary/test_volatile_runtime.py"],
                           capture_output=True, text=True, check=True).stdout.strip()
    environment = {
        "kernel": platform.release(), "machine": platform.machine(), "landlock_abi": abi,
        "strace": first_line(["strace", "-V"]), "cc": first_line(["cc", "--version"]),
        "rustc": first_line(["rustc", "-V"]), "python": platform.python_version(),
        "swap_total_kib": int(re.search(r"SwapTotal:\s+(\d+)", Path("/proc/meminfo").read_text()).group(1)),
        "core_pattern": Path("/proc/sys/kernel/core_pattern").read_text().strip(),
        "root_fs": mount_of("/"), "tmp_fs": mount_of("/tmp"),
    }
    document = {
        "schema": SCHEMA,
        "classification": "DEV evidence for phases V1/V2; the session's own test paths only; not a composed runtime",
        "source_commit": head, "source_tree_clean": not dirty,
        "environment": environment,
        "session_library": {"sha256": sha256(session_lib), "imports": imports, "exports": exports,
                            "footprint_bytes": footprint},
        "traced_syscalls": list(TRACED),
        "runs": runs, "controls": controls,
        "verdict": "ZERO_WRITE_PASS" if not problems else "FAIL", "problems": problems,
    }
    out.mkdir(parents=True, exist_ok=True)
    text = json.dumps(document, indent=1, sort_keys=True) + "\n"
    assert str(repo) not in text and str(build) not in text, "a host-local path leaked into the results"
    (out / "zero_write_v0.json").write_text(text)
    env_text = "\n".join(f"{k}: {v}" for k, v in environment.items()) + \
        f"\nsource_commit: {head} (tree clean for native/, the harness and the gate: {not dirty})\n" + \
        f"run: {time.strftime('%Y-%m-%d %H:%M UTC', time.gmtime())}\n"
    (out / "ENVIRONMENT.txt").write_text(env_text)
    sources = sorted(
        [p for p in (repo / "native/ECS_C/session").rglob("*") if p.is_file() and "target" not in p.parts] +
        [repo / "native/ECS_C/relational/tests/test_ecs_c_relational_session.c",
         repo / "native/ECS_C/relational/src/ecs_c_relational.c",
         repo / "native/ECS_C/relational/include/elpis/ecs_c_relational.h",
         repo / "native/runtime/containment/elpis_autonomous_run.c",
         repo / "research/volatile_runtime_v0/nomem_probe.c", repo / "research/volatile_runtime_v0/zero_write.py",
         repo / "tests/boundary/test_volatile_runtime.py"])
    manifest = {"schema": "elpis.volatile-runtime-v0.manifest.v1", "source_commit": head,
                "sources": {str(p.relative_to(repo)): sha256(p) for p in sources},
                "results": {name: sha256(out / name) for name in ("ENVIRONMENT.txt", "zero_write_v0.json")}}
    (out / "MANIFEST.json").write_text(json.dumps(manifest, indent=1, sort_keys=True) + "\n")
    print(f"{document['verdict']}: {len(runs)} paths x 3 modes; problems: {problems or 'none'}")
    return 0 if not problems else 1


if __name__ == "__main__":
    sys.exit(main())
