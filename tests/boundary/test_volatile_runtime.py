"""The volatile-native migration (docs/ELPIS_MISSION.md, Amendment 3; docs/VOLATILE_RUNTIME_CENSUS.md).

Phases V0 (census), V1 (the volatile ECS_C session) and V2 (the relational view rebound to it):

* The census is complete. Every persistent writer of the registry (elpis.runtime.persistence) has exactly one
  volatile disposition, and every Python and Rust source that writes persistent storage belongs to a registered
  writer (tests/boundary/test_autonomous_persistence.py scans the C/C++ sources). The RuntimeCore functions and the
  Runtime operations the census names as persistent are the ones the code declares.
* It fails closed. While anything is nonconforming, the authority documents say the migration is INCOMPLETE, and no
  document claims the runtime is diskless.
* The volatile ECS_C session has no persistent path. It is no_std, with no dependency and no build script. Its extern
  surface is memory, abort and getrandom only. Its header names no path or descriptor. Its built library imports no
  file or I/O function.
* Its test paths make no write-class system call on any file. This is traced with strace when the build and strace
  exist, and is never an implicit PASS.
"""
from __future__ import annotations

import ast
import inspect
import os
import re
import shutil
import subprocess
import tomllib

import pytest

from elpis.runtime import persistence as P
from elpis.runtime.composition import Runtime

from ..conftest import native_build_dir, native_required
from ._system import REPO

SESSION = REPO / "native/ECS_C/session"
DOCS = REPO / "docs"
INCOMPLETE = "VOLATILE MIGRATION: INCOMPLETE"


def _missing(message: str):
    if native_required():
        pytest.fail(message)
    pytest.skip(message + " (never an implicit PASS)")


# ------------------------------------------------------------------------------------------------- the census


def test_every_writer_has_exactly_one_volatile_disposition():
    ids = {w.id for w in P.WRITERS}
    assert set(P.VOLATILE_CENSUS) == ids, sorted(set(P.VOLATILE_CENSUS) ^ ids)
    for wid, entry in P.VOLATILE_CENSUS.items():
        assert type(entry.disposition) is P.VolatileDisposition and entry.action.strip(), wid
        assert entry.phase in {"V3", "V4", "-"}, (wid, entry.phase)
        if entry.disposition is P.VolatileDisposition.NONCONFORMING_RUNTIME:
            assert entry.phase in {"V3", "V4"}, wid           # a nonconforming writer has a retirement phase
    by = lambda d: {w for w, e in P.VOLATILE_CENSUS.items() if e.disposition is d}   # noqa: E731
    # Every writer the ordinary path or the native host reaches writes operational state: none conforms.
    assert P.AUTONOMOUS_WRITERS | P.NATIVE_HOST_AUTONOMOUS_WRITERS <= by(P.VolatileDisposition.NONCONFORMING_RUNTIME)
    assert {"native_host_deployment", "python_reference_deployment", "ecs_c_authority_provisioning",
            "k1_checkpoint_provisioning", "hgram_fixed_capacity_store", "canonical_grid81_publication"} <= by(
        P.VolatileDisposition.NONCONFORMING_RUNTIME)
    # What already refuses is the registry's prohibited set, and only that.
    prohibited = {w.id for w in P.WRITERS if w.classification is P.WriterClass.PROHIBITED_FROM_AUTONOMOUS_RUNTIME}
    assert by(P.VolatileDisposition.RETIRED) == prohibited
    # Offline research is never runtime; an offline artifact producer is never an autonomous writer.
    research = {w.id for w in P.WRITERS if w.classification is P.WriterClass.OFFLINE_RESEARCH_ONLY}
    assert by(P.VolatileDisposition.DEVELOPMENT_ONLY) == research
    assert not by(P.VolatileDisposition.OFFLINE_ARTIFACT_PRODUCER) & (P.AUTONOMOUS_WRITERS |
                                                                     P.NATIVE_HOST_AUTONOMOUS_WRITERS)
    # The one volatile replacement that exists is the session, and nothing composes it yet.
    assert P.VOLATILE_REPLACEMENTS == {
        "ecs_c_authority_slots": ("native/ECS_C/session", "ECSC_VOLATILE_SESSION_R0", "not composed")}


# A persistent write from Python: write-mode opens, path mutation, syncs, temporary files, databases, file logging,
# writable or file-backed arrays. Read-only opens (os.O_RDONLY) are not writes.
_PY_WRITES = re.compile(
    r"""\bopen\([^)\n]*,\s*['"][^'"]*[wax+]|\bmode\s*=\s*['"][^'"]*[wax+]|\.write_text\(|\.write_bytes\("""
    r"""|\bos\.(?:rename|replace|remove|unlink|rmdir|removedirs|mkdir|makedirs|fsync|fdatasync|truncate|ftruncate"""
    r"""|link|symlink|mkfifo|mknod|sync)\(|\bshutil\.(?:copy|copy2|copyfile|copytree|move|rmtree)\("""
    r"""|\btempfile\.|\bsqlite3\.connect\(|\bFileHandler\b|RotatingFileHandler|\.mkdir\(|\.touch\(|\.unlink\("""
    r"""|\.rmdir\(|\.symlink_to\(|\.hardlink_to\(|\bos\.O_(?:WRONLY|RDWR|CREAT|TRUNC|APPEND)\b"""
    r"""|\b(?:np|numpy)\.(?:save|savez|savetxt|memmap)\(|\.tofile\(|ACCESS_WRITE|\bshelve\.|\bdbm\.""")


def _python_writer_modules() -> set[str]:
    return {entry.split(":")[0] for w in P.WRITERS for entry in w.python}


def test_every_python_source_that_writes_is_a_registered_writer():
    registered = _python_writer_modules()
    writers, unregistered = set(), []
    for path in sorted((REPO / "src").rglob("*.py")):
        text = path.read_text(encoding="utf-8")
        if not _PY_WRITES.search(text):
            continue
        module = ".".join(path.relative_to(REPO / "src").with_suffix("").parts)
        writers.add(module)
        if module not in registered:
            unregistered.append((module, sorted({m.group(0) for m in _PY_WRITES.finditer(text)})))
    assert not unregistered, ("classify each persistent writer in elpis.runtime.persistence", unregistered)
    # The scan is not vacuous: it finds the writers the census of V0 found.
    assert {"elpis.runtime.deployment", "elpis.runtime.acceptance", "elpis.runtime.release",
            "elpis.runtime.authority", "elpis.pipeline.canonical.publisher"} <= writers, sorted(writers)


# A persistent write from Rust: write opens, creation, sync, rename, removal, positional writes, resizing.
_RS_WRITES = re.compile(r"\bFile::create\b|\.write\(true\)|\.create(?:_new)?\(true\)|\.append\(true\)|\.truncate\("
                        r"true\)|\bsync_data\(|\bsync_all\(|\bfs::(?:rename|remove_file|remove_dir\w*|create_dir\w*|"
                        r"write|copy|hard_link)\b|\bDirBuilder\b|\bwrite_all_at\(|\bwrite_at\(|\bset_len\(")


def test_every_rust_source_that_writes_is_a_registered_writer():
    classified = {entry.split(":", 1)[0] for w in P.WRITERS for entry in w.native}
    writers = set()
    for path in sorted((REPO / "native").rglob("*.rs")):
        rel = path.relative_to(REPO).as_posix()
        if "/target/" in rel or path.name == "tests.rs" or "/tests/" in rel:
            continue
        if _RS_WRITES.search(path.read_text(encoding="utf-8")):
            writers.add(rel)
    assert writers == {"native/ECS_C/authority/src/store.rs", "native/runtime/src/checkpoint.rs"}, sorted(writers)
    assert writers <= classified, sorted(writers - classified)
    # The session crate is not among them, and never will be: it is no_std (below).
    assert not any(w.startswith("native/ECS_C/session/") for w in writers)


def _declared_runtime_functions() -> set[str]:
    header = (REPO / "native/runtime/include/elpis/runtime.h").read_text(encoding="utf-8")
    return set(re.findall(r"\b(elpis_runtime_\w+)\s*\(", header))


def test_the_persistent_runtime_core_abi_and_operations_are_the_declared_ones():
    declared = _declared_runtime_functions()
    assert P.RUNTIME_CORE_PERSISTENT_ABI <= declared, sorted(P.RUNTIME_CORE_PERSISTENT_ABI - declared)
    # ABI v7 is still the implemented version (v8, over the volatile session, retires these functions).
    ffi = (REPO / "native/runtime/src/ffi.rs").read_text(encoding="utf-8")
    assert re.search(r"pub const ABI_VERSION: u32 = 7;", ffi)
    assert "RuntimeCore C ABI v7" in (REPO / "native/runtime/include/elpis/runtime.h").read_text(encoding="utf-8")
    operations = {name for name, _ in inspect.getmembers(Runtime) if not name.startswith("_")}
    assert P.NONCONFORMING_OPERATIONS <= operations, sorted(P.NONCONFORMING_OPERATIONS - operations)
    # The composer is constructed over the durable authority's directory.
    tree = ast.parse((REPO / "src/elpis/runtime/composition.py").read_text(encoding="utf-8"))
    config = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "RuntimeConfig")
    fields = {n.target.id for n in config.body if isinstance(n, ast.AnnAssign) and isinstance(n.target, ast.Name)}
    assert "authority_dir" in fields
    assert P.VOLATILE_MIGRATION_COMPLETE is False


_DISKLESS_CLAIM = re.compile(r"(?i)\b(?:the\s+)?(?:runtime|elpis)\s+is\s+(?:now\s+)?(?:fully\s+|entirely\s+)?"
                             r"(?:diskless|volatile[- ]only)\b")


def test_the_documents_fail_closed_while_the_migration_is_incomplete():
    authority = {name: (DOCS / name).read_text(encoding="utf-8") for name in (
        "ELPIS_MISSION.md", "VOLATILE_RUNTIME_CENSUS.md", "ECS_C_SESSION.md", "NONCLAIMS.md", "ECS_C_AUTHORITY.md",
        "RUNTIME_CORE.md", "NATIVE_HOST.md", "NATIVE_SPLIT_CENSUS.md")}
    for name, text in authority.items():
        assert (INCOMPLETE in text) is (not P.VOLATILE_MIGRATION_COMPLETE), name
    assert "Elpis is not yet diskless" in authority["NONCLAIMS.md"]
    assert "Amendment 3" in authority["ELPIS_MISSION.md"]
    claims = []
    for path in [REPO / "README.md", REPO / "ELPIS_SYSTEM.json", *sorted(DOCS.rglob("*.md"))]:
        for m in _DISKLESS_CLAIM.finditer(path.read_text(encoding="utf-8")):
            claims.append((path.relative_to(REPO).as_posix(), m.group(0)))
    assert not claims, claims
    system = (REPO / "ELPIS_SYSTEM.json").read_text(encoding="utf-8")
    assert INCOMPLETE in system


# ------------------------------------------------------------------------------------------------- the session

def _code(text: str) -> str:
    text = re.sub(r"/\*.*?\*/", "", text, flags=re.S)
    return "\n".join(line.split("//")[0] for line in text.splitlines())


def test_the_session_crate_has_no_persistent_path():
    # Unconditionally no_std, with std only in the test build: outside the tests, no std item (std::fs, std::io,
    # std::env, std::thread, std::time) can even be named; the compiler enforces it.
    lib = _code((SESSION / "src/lib.rs").read_text(encoding="utf-8"))
    assert re.search(r"^#!\[no_std\]$", lib, flags=re.M) and "cfg_attr" not in lib
    assert re.findall(r"(#\[[^\]]*\]\s*)?extern crate std;", lib) == ["#[cfg(test)]\n"], "std only in the tests"
    assert not (SESSION / "build.rs").exists()
    manifest = tomllib.loads((SESSION / "Cargo.toml").read_text(encoding="utf-8"))
    for table in ("dependencies", "build-dependencies", "dev-dependencies"):
        assert not manifest.get(table), table
    assert manifest["lib"]["crate-type"] == ["rlib"] and manifest["features"]["default"] == []
    lock = tomllib.loads((SESSION / "Cargo.lock").read_text(encoding="utf-8"))
    assert [p["name"] for p in lock["package"]] == ["elpis_ecs_c_session"]
    # Its foreign surface: memory, abort and the kernel's random source.
    extern = set()
    for path in sorted((SESSION / "src").glob("*.rs")):
        for block in re.findall(r'extern\s+"C"\s*\{(.*?)\n\}', _code(path.read_text(encoding="utf-8")), flags=re.S):
            extern |= set(re.findall(r"\bfn\s+(\w+)\s*\(", block))
    assert extern == {"malloc", "free", "posix_memalign", "abort", "getrandom"}, sorted(extern)
    # The kinds that existed only for persistence are refused as retired, never applied.
    session = (SESSION / "src/session.rs").read_text(encoding="utf-8")
    assert "PersistenceRetired" in session and "Code::PersistenceRetired" in (SESSION / "src/error.rs").read_text(
        encoding="utf-8")


def test_the_session_header_has_no_path_or_descriptor():
    for name in ("ecs_c_session.h", "ecs_c_types.h"):
        text = _code((SESSION / "include/elpis" / name).read_text(encoding="utf-8"))
        for word in ("path", "dir", "file", "FILE", "fd", "mode_t", "off_t"):
            assert not re.search(rf"\b{word}\b", text), (name, word)
        params = re.findall(r"\(([^;{}]*)\)\s*;", text)
        assert not any("char" in p for p in params), (name, [p for p in params if "char" in p])


# The library may import memory, abort and the kernel's random source; nothing that names a file, descriptor or I/O.
_SESSION_IMPORTS = {"abort", "free", "getrandom", "malloc", "memcpy", "memmove", "memset", "memcmp", "bcmp",
                    "posix_memalign"}
_WEAK_RUNTIME = {"_ITM_deregisterTMCloneTable", "_ITM_registerTMCloneTable", "__cxa_finalize", "__gmon_start__"}


def _session_library():
    found = sorted(p for p in native_build_dir().rglob("libelpis_ecs_c_session.so")
                   if p.is_file() and not any(part.startswith("cargo") for part in p.relative_to(
                       native_build_dir()).parts))
    if not found:
        _missing(f"libelpis_ecs_c_session.so not built under {native_build_dir()}")
    return found[0]


def test_the_built_session_library_imports_no_file_or_io_function():
    library = _session_library()
    nm = shutil.which("nm")
    if nm is None:
        _missing("nm is not available")
    table = subprocess.run([nm, "-D", "--undefined-only", str(library)], capture_output=True, text=True,
                           check=True).stdout
    imports = {line.split()[-1].split("@")[0] for line in table.splitlines() if line.split()}
    assert imports - _WEAK_RUNTIME <= _SESSION_IMPORTS, sorted(imports - _WEAK_RUNTIME - _SESSION_IMPORTS)
    assert {"malloc", "free", "getrandom"} <= imports


# The write-class system calls a test path must never make on a file. Writes to pipes, terminals and /dev/null are
# not storage; neither are anonymous or memfd mappings.
_TRACED = ("open,openat,openat2,creat,write,writev,pwrite64,pwritev,pwritev2,rename,renameat,renameat2,unlink,"
           "unlinkat,mkdir,mkdirat,rmdir,truncate,ftruncate,fallocate,fsync,fdatasync,sync,syncfs,sync_file_range,"
           "msync,"
           "link,linkat,symlink,symlinkat,mknod,mknodat,chmod,fchmod,fchmodat,chown,fchown,lchown,fchownat,"
           "utimensat,setxattr,lsetxattr,fsetxattr,removexattr,lremovexattr,fremovexattr,mmap,copy_file_range,"
           "sendfile,splice,io_uring_setup,name_to_handle_at,open_by_handle_at")
_ALWAYS_WRITE = {"rename", "renameat", "renameat2", "unlink", "unlinkat", "mkdir", "mkdirat", "rmdir", "truncate",
                 "ftruncate", "fallocate", "fsync", "fdatasync", "sync", "syncfs", "sync_file_range", "msync", "link",
                 "linkat", "symlink", "symlinkat", "mknod", "mknodat", "chmod", "fchmod", "fchmodat", "chown",
                 "fchown", "lchown", "fchownat", "utimensat", "setxattr", "lsetxattr", "fsetxattr", "removexattr",
                 "lremovexattr", "fremovexattr", "copy_file_range", "sendfile", "splice", "io_uring_setup",
                 "name_to_handle_at", "open_by_handle_at"}
_WRITE_FLAGS = re.compile(r"\bO_(?:WRONLY|RDWR|CREAT|TRUNC|APPEND|TMPFILE)\b")
_CALL = re.compile(r"^(?:\[pid\s+\d+\]\s+|\d+\s+)?(\w+)\((.*)\)\s+=\s+(-?\d+|\?)")


def _not_storage(fd_text: str) -> bool:
    target = fd_text.split("<", 1)[1].rstrip(">") if "<" in fd_text else ""
    return target.startswith(("pipe:", "socket:", "anon_inode:", "/dev/pts/", "/dev/null", "/dev/tty",
                              "/memfd:")) or target == ""


def _calls(log: str):
    """Complete call lines: an interrupted call ("<unfinished ...>") joined with its "<... name resumed>" rest; a
    call never returned from kept with the result "?"."""
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
    """The write-class events a strace -f -y log records against storage (attempted or successful)."""
    events = []
    for raw in _calls(log):
        m = _CALL.match(raw.strip())
        if not m:
            continue
        name, args = m.group(1), m.group(2)
        if name in _ALWAYS_WRITE:
            events.append(raw.strip())
        elif name in ("open", "openat", "openat2", "creat"):
            if name == "creat" or _WRITE_FLAGS.search(args):
                path = re.search(r'"([^"]*)"', args)
                if not (path and path.group(1).startswith(("/dev/null", "/dev/pts/", "/dev/tty"))):
                    events.append(raw.strip())
        elif name in ("write", "writev", "pwrite64", "pwritev", "pwritev2"):
            if not _not_storage(args.split(",", 1)[0]):
                events.append(raw.strip())
        elif name == "mmap":
            parts = [a.strip() for a in args.split(",")]
            if len(parts) >= 5 and "PROT_WRITE" in parts[2] and "MAP_SHARED" in parts[3] and \
                    not parts[4].startswith("-1") and not _not_storage(parts[4]):
                events.append(raw.strip())
    return events


def _strace_run(command: list[str], tmp_path) -> tuple[subprocess.CompletedProcess, list[str]]:
    strace = shutil.which("strace")
    if strace is None:
        _missing("strace is not available")
    log = tmp_path / "trace.log"   # written by the tracer (the test harness), never by the traced process
    run = subprocess.run([strace, "-f", "-qq", "-y", "-s", "96", "-e", "trace=" + _TRACED, "-e", "signal=none",
                          "-o", str(log), "--", *command], capture_output=True, text=True, timeout=900,
                         env={"PATH": os.environ.get("PATH", "/usr/bin:/bin")}, cwd="/")
    if run.returncode != 0 and "ptrace" in run.stderr:
        _missing("ptrace is not permitted here: " + run.stderr.strip()[:200])
    return run, write_events(log.read_text(encoding="utf-8", errors="replace"))


def test_the_write_detector_is_not_blind(tmp_path):
    # A negative control: the detector must see a real creation, rename, sync and removal.
    probe = tmp_path / "probe"
    run, events = _strace_run(["/bin/sh", "-c", f"echo x > {probe}.a && mv {probe}.a {probe}.b && sync && "
                                                f"rm {probe}.b"], tmp_path)
    assert run.returncode == 0, run.stderr
    names = {_CALL.match(e).group(1) for e in events}
    assert {"rename", "renameat", "renameat2"} & names and {"unlink", "unlinkat"} & names and "sync" in names, events
    assert any(_WRITE_FLAGS.search(e) for e in events), events


@pytest.mark.parametrize("binary", ["ECS_C/session/test_ecs_c_session_abi", "ECS_C/test_ecs_c_relational_session"])
def test_the_session_test_paths_make_no_filesystem_write(binary, tmp_path):
    path = native_build_dir() / "native" / binary
    if not path.is_file():
        _missing(f"{binary} not built under {native_build_dir()}")
    run, events = _strace_run([str(path)], tmp_path)
    assert run.returncode == 0, (run.stdout[-2000:], run.stderr[-2000:])
    assert events == [], events[:20]
