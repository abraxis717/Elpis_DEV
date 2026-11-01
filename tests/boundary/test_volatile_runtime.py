"""The volatile-native migration (docs/ELPIS_MISSION.md, Amendment 3; docs/VOLATILE_RUNTIME_CENSUS.md).

Phases V0 to V5. Each property is checked against the code and the built binaries:

* The census is closed. Every persistent writer of the registry (elpis.runtime.persistence) has exactly one final
  disposition, and every active-runtime writer is RETIRED, REPLACED_WITH_VOLATILE or READ_ONLY_STATIC_RESOURCE.
* Every Python source that writes persistent storage is a registered offline producer or development code; no Rust
  source writes at all.
* RuntimeCore is ABI v8: no function that writes, opens for writing or resumes persistent storage is declared or
  exported, and the Runtime has no persistent operation or configuration.
* No binary the runtime loads (the production host, every library it admits, the bridges the Python runtime loads,
  the file-asset library) imports a write-class C library function.
* The documents state the migration as it is: VOLATILE MIGRATION: COMPLETE exactly when every closure condition
  holds, INCOMPLETE otherwise. No document claims the runtime is diskless or that the operating system writes
  nothing.
* The closure is evidenced: condition 11's composed-runtime qualification is the recorded run, read back here.
* The volatile ECS_C session has no persistent path (no_std, no dependency, memory/abort/getrandom only, a header
  without paths), and its test paths make no write-class system call (strace, when available; never an implicit
  PASS).
"""
from __future__ import annotations

import ast
import hashlib
import inspect
import json
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
COMPLETE = "VOLATILE MIGRATION: COMPLETE"


def _missing(message: str):
    if native_required():
        pytest.fail(message)
    pytest.skip(message + " (never an implicit PASS)")


# ------------------------------------------------------------------------------------------------- the census


def test_every_writer_has_exactly_one_final_disposition():
    for w in P.WRITERS:
        assert type(w.disposition) is P.Disposition, w.id
        if w.runtime:
            assert w.disposition in P.RUNTIME_DISPOSITIONS, w.id
        else:
            assert w.disposition in (P.Disposition.OFFLINE_ARTIFACT_PRODUCER, P.Disposition.DEVELOPMENT_ONLY), w.id
    assert "NONCONFORMING" not in json.dumps([w.disposition.value for w in P.WRITERS])
    by = {d: {w.id for w in P.WRITERS if w.disposition is d} for d in P.Disposition}
    # Everything the persistent runtime wrote is retired or replaced; COLD is read only.
    assert {"k1_checkpoint_slots", "k1_checkpoint_provisioning", "runtime_durability_lane", "ecs_c_authority_provisioning",
            "fms_posix_cold_store", "hgram_fixed_capacity_store", "runtime_canonical_publication"} <= \
        by[P.Disposition.RETIRED]
    assert {"ecs_c_authority_slots", "native_host_deployment", "python_reference_deployment",
            "evolution_reservation", "operator_authority_events"} <= by[P.Disposition.REPLACED_WITH_VOLATILE]
    assert {"pinned_native_libraries", "fms_cold_backing", "hgram_retained_store"} <= \
        by[P.Disposition.READ_ONLY_STATIC_RESOURCE]
    assert P.VOLATILE_REPLACEMENTS["ecs_c_authority_slots"].startswith("the volatile ECS_C session")


# A persistent write from Python: write-mode opens, path mutation, syncs, temporary files, databases, file logging,
# writable or file-backed arrays. Read-only opens (os.O_RDONLY) are not writes.
_PY_WRITES = re.compile(
    r"""\bopen\([^)\n]*,\s*['"][^'"]*[wax+]|\bmode\s*=\s*['"][^'"]*[wax+]|\.write_text\(|\.write_bytes\("""
    r"""|\bos\.(?:rename|replace|remove|unlink|rmdir|removedirs|mkdir|makedirs|fsync|fdatasync|truncate|ftruncate"""
    r"""|link|symlink|mkfifo|mknod|sync)\(|\bshutil\.(?:copy|copy2|copyfile|copytree|move|rmtree)\("""
    r"""|\btempfile\.|\bsqlite3\.connect\(|\bFileHandler\b|RotatingFileHandler|\.mkdir\(|\.touch\(|\.unlink\("""
    r"""|\.rmdir\(|\.symlink_to\(|\.hardlink_to\(|\bos\.O_(?:WRONLY|RDWR|CREAT|TRUNC|APPEND)\b"""
    r"""|\b(?:np|numpy)\.(?:save|savez|savetxt|memmap)\(|\.tofile\(|ACCESS_WRITE|\bshelve\.|\bdbm\.""")


# Functions that name write access modes only to refuse a descriptor opened with them (the offline CLI's storage-stdio
# check, as the native host's): their names are not writes. Everything else in the module is still scanned.
_CHECKS_ONLY = {"elpis.runtime.cli": "_storage"}


def _without_function(text: str, name: str) -> str:
    tree = ast.parse(text)
    node = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == name)
    body = ast.get_source_segment(text, node)
    assert "open(" not in body and "fdopen" not in body, "a refusal check must not open anything"
    return text.replace(body, "")


def test_every_python_source_that_writes_is_a_registered_writer():
    registered = {entry.split(":")[0] for w in P.WRITERS
                  if w.disposition in (P.Disposition.OFFLINE_ARTIFACT_PRODUCER, P.Disposition.DEVELOPMENT_ONLY)
                  for entry in w.python}
    writers, unregistered = set(), []
    for path in sorted((REPO / "src").rglob("*.py")):
        text = path.read_text(encoding="utf-8")
        module = ".".join(path.relative_to(REPO / "src").with_suffix("").parts)
        if module in _CHECKS_ONLY:
            text = _without_function(text, _CHECKS_ONLY[module])
        if not _PY_WRITES.search(text):
            continue
        writers.add(module)
        if module not in registered:
            unregistered.append((module, sorted({m.group(0) for m in _PY_WRITES.finditer(text)})))
    assert not unregistered, ("classify each persistent writer in elpis.runtime.persistence", unregistered)
    # The scan is not vacuous: it finds the offline and development writers.
    assert {"elpis.runtime.acceptance", "elpis.runtime.release", "elpis.pipeline.canonical.publisher",
            "elpis.evolution.materialize"} <= writers, sorted(writers)
    # None of the composer's own modules writes.
    composer = {"elpis.runtime.composition", "elpis.runtime.core", "elpis.runtime.cognition", "elpis.runtime.authority",
                "elpis.runtime.deployment", "elpis.runtime.cli", "elpis.runtime.context", "elpis.runtime.environment",
                "elpis.substrate.residency", "elpis.evolution.promotion", "elpis.evolution.policy"}
    assert not writers & composer, sorted(writers & composer)


# A persistent write from Rust: write opens, creation, sync, rename, removal, positional writes, resizing.
_RS_WRITES = re.compile(r"\bFile::create\b|\.write\(true\)|\.create(?:_new)?\(true\)|\.append\(true\)|\.truncate\("
                        r"true\)|\bsync_data\(|\bsync_all\(|\bfs::(?:rename|remove_file|remove_dir\w*|create_dir\w*|"
                        r"write|copy|hard_link)\b|\bDirBuilder\b|\bwrite_all_at\(|\bwrite_at\(|\bset_len\(")


def test_no_rust_source_writes_persistent_storage():
    writers = set()
    for path in sorted((REPO / "native").rglob("*.rs")):
        rel = path.relative_to(REPO).as_posix()
        if "/target/" in rel or path.name == "tests.rs" or "/tests/" in rel:
            continue
        if _RS_WRITES.search(path.read_text(encoding="utf-8")):
            writers.add(rel)
    assert writers == set(), sorted(writers)
    assert not (REPO / "native/runtime/src/checkpoint.rs").exists() and not (REPO / "native/runtime/src/durable.rs").exists()


def _declared_runtime_functions() -> set[str]:
    header = (REPO / "native/runtime/include/elpis/runtime.h").read_text(encoding="utf-8")
    return set(re.findall(r"\b(elpis_runtime_\w+)\s*\(", header))


def test_the_runtime_core_abi_has_no_persistent_function():
    declared = _declared_runtime_functions()
    assert not declared & P.RETIRED_RUNTIME_CORE_ABI, sorted(declared & P.RETIRED_RUNTIME_CORE_ABI)
    assert P.RUNTIME_CORE_PERSISTENT_ABI == frozenset()
    assert "elpis_runtime_create" in declared
    ffi = (REPO / "native/runtime/src/ffi.rs").read_text(encoding="utf-8")
    assert re.search(r"pub const ABI_VERSION: u32 = 8;", ffi)
    assert re.search(r"\bELPIS_RUNTIME_ABI_V8 = 8u\b", (REPO / "native/runtime/include/elpis/runtime.h").read_text(
        encoding="utf-8"))
    inventory = json.loads((REPO / "native/ABI_INVENTORY.json").read_text(encoding="utf-8"))["libraries"]
    exported = set(inventory["libelpis_runtime.so"]["exports"])
    assert not exported & P.RETIRED_RUNTIME_CORE_ABI and "libelpis_runtime_testing.so" not in inventory
    # The Runtime: no persistent operation, no persistent configuration.
    operations = {name for name, _ in inspect.getmembers(Runtime) if not name.startswith("_")}
    for retired in ("publish_recoverable", "publish_identity", "durability", "recover_k1", "discard_k1_candidate",
                    "adopt_k1_candidate", "import_composite", "publish_canonical"):
        assert retired not in operations, retired
    tree = ast.parse((REPO / "src/elpis/runtime/composition.py").read_text(encoding="utf-8"))
    config = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "RuntimeConfig")
    fields = {n.target.id for n in config.body if isinstance(n, ast.AnnAssign) and isinstance(n.target, ast.Name)}
    assert not {"authority_dir", "checkpoint_dir", "home", "durability"} & fields, fields


# C library functions that write, create, remove, rename, link, resize or sync storage, or make temporary files.
# write/fwrite (to a pipe or terminal checked at startup) and open/fopen (read-only modes, checked in the sources below
# and on the running process by strace) are not in the set.
_WRITE_CLASS = {"rename", "renameat", "renameat2", "unlink", "unlinkat", "mkdir", "mkdirat", "rmdir", "ftruncate",
                "ftruncate64", "truncate", "truncate64", "fallocate", "fallocate64", "posix_fallocate",
                "posix_fallocate64", "fsync", "fdatasync", "sync", "syncfs", "sync_file_range", "pwrite", "pwrite64",
                "pwritev", "pwritev2", "link", "linkat", "symlink", "symlinkat", "creat", "creat64", "msync", "mkstemp",
                "mkstemp64", "mkstemps", "mkostemp", "mkdtemp", "tmpfile", "tmpfile64", "tempnam", "tmpnam", "chmod",
                "fchmod", "fchmodat", "chown", "fchown", "lchown", "fchownat", "utimensat", "futimens", "utime",
                "utimes", "setxattr", "fsetxattr", "lsetxattr", "removexattr", "fremovexattr", "copy_file_range",
                "sendfile", "splice", "mknod", "mknodat", "mkfifo", "remove"}
# What the runtime loads: the production host and the six libraries it admits; the volatile session RuntimeCore
# embeds; the bridges and libraries the Python runtime loads.
_LOADED = ("native/runtime/elpis", "native/runtime/libelpis_runtime.so", "native/runtime/libelpis_hacf_observation.so",
           "native/runtime/composite/libelpis_ecs_composite.so", "native/ECS_G/libelpis_ecsg_math.so",
           "native/ECS_G/libelpis_ecsg_k1.so", "native/ECS_G/libelpis_ecsg_k1_fms.so", "native/ECS_G/libelpis_ecsg_fms.so",
           "native/ECS_C/libelpis_ecs_memory.so", "native/ECS_C/session/libelpis_ecs_c_session.so",
           "native/substrate/libelpis_fms_file_assets.so", "native/structure/libelpis_retrieval_bridge.so",
           "native/pipeline/libelpis_ingress_bridge.so", "native/substrate/libelpis_execution.so")


def test_no_binary_the_runtime_loads_imports_a_write_function():
    nm = shutil.which("nm")
    if nm is None:
        _missing("nm is not available")
    build = native_build_dir()
    for rel in _LOADED:
        path = build / rel
        if not path.is_file():
            _missing(f"{rel} is not built under {build}")
        table = subprocess.run([nm, "-D", "--undefined-only", str(path)], capture_output=True, text=True,
                               check=True).stdout
        imports = {line.split()[-1].split("@")[0] for line in table.splitlines() if line.split()}
        assert not imports & _WRITE_CLASS, (rel, sorted(imports & _WRITE_CLASS))
    # The testing host is the one binary with writers (its probe), and it is never installed.
    testing = build / "native/runtime/elpis_testing"
    if testing.is_file():
        table = subprocess.run([nm, "-D", "--undefined-only", str(testing)], capture_output=True, text=True,
                               check=True).stdout
        assert {"rename", "unlink", "mkdir"} & {l.split()[-1].split("@")[0] for l in table.splitlines() if l.split()}
    # fopen in what the runtime links opens for reading only.
    for source in ("native/runtime/host/src/elpis_host.c", "native/structure/semantic/src/graph/snapshot_manifest.c"):
        modes = re.findall(r'\bfopen\s*\([^;]*?,\s*"([^"]*)"\s*\)', (REPO / source).read_text(encoding="utf-8"))
        assert modes and all(set(m) <= set("rb") for m in modes), (source, modes)


_DISKLESS_CLAIM = re.compile(r"(?i)\b(?:the\s+)?(?:runtime|elpis)\s+is\s+(?:now\s+)?(?:fully\s+|entirely\s+)?"
                             r"(?:diskless|volatile[- ]only)\b")
_OS_CLAIM = re.compile(r"(?i)\b(?:the\s+)?(?:operating system|whole system|host|machine)\s+(?:performs|makes|issues)\s+"
                       r"(?:no|zero)\s+(?:NAND\s+|storage\s+|disk\s+)?writes\b")


def test_the_documents_state_the_migration_as_it_is():
    authority = {name: (DOCS / name).read_text(encoding="utf-8") for name in (
        "ELPIS_MISSION.md", "VOLATILE_RUNTIME_CENSUS.md", "ECS_C_SESSION.md", "NONCLAIMS.md", "ECS_C_AUTHORITY.md",
        "RUNTIME_CORE.md", "NATIVE_HOST.md", "NATIVE_SPLIT_CENSUS.md")}
    state = COMPLETE if P.VOLATILE_MIGRATION_COMPLETE else INCOMPLETE
    other = INCOMPLETE if P.VOLATILE_MIGRATION_COMPLETE else COMPLETE
    for name, text in authority.items():
        assert state in text and other not in text, name
    system = (REPO / "ELPIS_SYSTEM.json").read_text(encoding="utf-8")
    assert state in system and other not in system
    assert "Amendment 3" in authority["ELPIS_MISSION.md"]
    # Precise claims only: the runtime issues no persistent write; nothing more is claimed of the machine.
    assert "outside Elpis's process-control boundary" in authority["NONCLAIMS.md"]
    claims = []
    for path in [REPO / "README.md", REPO / "ELPIS_SYSTEM.json", *sorted(DOCS.rglob("*.md"))]:
        text = path.read_text(encoding="utf-8")
        for pattern in (_DISKLESS_CLAIM, _OS_CLAIM):
            for m in pattern.finditer(text):
                claims.append((path.relative_to(REPO).as_posix(), m.group(0)))
    assert not claims, claims


def test_the_closure_is_evidenced():
    assert [c.number for c in P.CLOSURE] == list(range(1, 15))
    for c in P.CLOSURE:
        assert c.statement.strip() and c.evidence, c.number
    assert P.VOLATILE_MIGRATION_COMPLETE is (all(c.holds for c in P.CLOSURE)
                                             and not P.REMAINING_DEPENDENCIES)
    composed = next(c for c in P.CLOSURE if c.number == 11)
    results = REPO / P.COMPOSED_QUALIFICATION
    if composed.holds:
        document = json.loads(results.read_text(encoding="utf-8"))
        assert document["verdict"] == "ZERO_WRITE_PASS" and document["problems"] == [], document["problems"]
        assert document["source_tree_clean"] is True
        manifest = json.loads((results.parent / "MANIFEST.json").read_text(encoding="utf-8"))
        for name, digest in manifest["results"].items():
            assert hashlib.sha256((results.parent / name).read_bytes()).hexdigest() == digest, name
        # It is the composed runtime: the host, RuntimeCore, the Python reference and FMS/HACF all ran, every traced
        # path in both traced modes with zero write events, and the recorded run is of this branch's history.
        runs = document["runs"]
        assert {"host_session", "host_long", "host_restarts", "python_cli", "python_fms_hacf", "fms_cold", "hgram_read",
                "runtime_core.session", "runtime_core.composite", "runtime_core.hot_io"} <= set(runs), sorted(runs)
        traced = [(name, mode) for name, run in runs.items() for mode in ("observed", "confined")
                  if isinstance(run, dict) and mode in run]
        assert len(traced) >= 20, traced
        for name, mode in traced:
            assert runs[name][mode]["exit"] == 0 and runs[name][mode]["write_events"] == [], (name, mode)
        # The recorded commit is in this branch's history. A shallow checkout (CI's) does not hold that history, so
        # there the ancestry is not checkable; everything above still is.
        commit = document["source_commit"]
        known = subprocess.run(["git", "-C", str(REPO), "cat-file", "-e", f"{commit}^{{commit}}"],
                               capture_output=True).returncode == 0
        shallow = subprocess.run(["git", "-C", str(REPO), "rev-parse", "--is-shallow-repository"],
                                 capture_output=True, text=True).stdout.strip() == "true"
        assert known or shallow, f"the recorded commit {commit} is unknown to a full clone"
        if known:
            assert subprocess.run(["git", "-C", str(REPO), "merge-base", "--is-ancestor", commit, "HEAD"],
                                  capture_output=True).returncode == 0, commit


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
