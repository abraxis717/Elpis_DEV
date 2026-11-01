"""Elpis SHALL NOT autonomously mutate persistent storage (elpis.runtime.persistence; docs/ELPIS_MISSION.md,
Amendment 3).

Architectural properties of the census, checked against the code:

* every public Runtime operation is classified exactly once (autonomous or operator), and none writes;
* the registry binds to the code: a live entry point (read-only, offline or development) resolves; a retired or
  replaced persistent entry point is absent, in code and not just in a test's reach;
* no composer reaches a writer: the modules a fresh interpreter loads for the Python runtime include no module that
  hosts an offline or development writer, and the autonomous operations reference none;
* the native host creates RuntimeCore over the volatile session only (ABI v8) and builds no FMS store;
* every native C/C++ source that performs a persistent write is a classified offline producer or development code,
  and no bridge that Python loads calls one;
* the H-gram writers and the persistent HACF corpus do not exist; the retrieval epoch's FMS is RAM-only;
* importing the runtime installs no persistent log sink.

The same law is proven on real native libraries under a write trap in tests/integration/test_autonomous_no_growth.py,
and on the composed runtime under strace in research/volatile_runtime_v1.
"""
from __future__ import annotations

import ast
import importlib
import inspect
import os
import re
import subprocess
import sys
from pathlib import Path

import pytest

from elpis.runtime import persistence as P
from elpis.runtime.composition import Runtime

REPO = Path(__file__).resolve().parents[2]
SRC = REPO / "src" / "elpis"
NATIVE = REPO / "native"
LIVE = {P.Disposition.READ_ONLY_STATIC_RESOURCE, P.Disposition.OFFLINE_ARTIFACT_PRODUCER,
        P.Disposition.DEVELOPMENT_ONLY}


def _public_operations() -> set[str]:
    return {name for name, _ in inspect.getmembers(Runtime) if not name.startswith("_")}


def test_every_runtime_operation_is_classified_exactly_once():
    operations = _public_operations()
    assert not (P.AUTONOMOUS_OPERATIONS & P.OPERATOR_OPERATIONS)
    assert P.AUTONOMOUS_OPERATIONS | P.OPERATOR_OPERATIONS == operations, (
        "classify every public Runtime operation in elpis.runtime.persistence",
        sorted(operations ^ (P.AUTONOMOUS_OPERATIONS | P.OPERATOR_OPERATIONS)))
    assert P.NONCONFORMING_OPERATIONS == frozenset() and P.AUTONOMOUS_WRITERS == frozenset()
    assert P.NATIVE_HOST_AUTONOMOUS_WRITERS == frozenset()


def _code(text: str, suffix: str) -> str:
    """Code without comments (C, C++, Rust)."""
    if suffix in (".c", ".h", ".cpp", ".hpp", ".rs"):
        text = re.sub(r"/\*.*?\*/", "", text, flags=re.S)
        return "\n".join(line.split("//")[0] for line in text.splitlines())
    return text


def _resolve(entry: str):
    module, qualname = entry.split(":")
    try:
        target = importlib.import_module(module)
    except ModuleNotFoundError:
        return None
    for part in qualname.split("."):
        if not hasattr(target, part):
            return None
        target = getattr(target, part)
    return target


def _native_present(entry: str) -> bool:
    path, symbol = entry.split(":", 1)
    file = REPO / path
    return file.is_file() and symbol in _code(file.read_text(encoding="utf-8", errors="replace"), file.suffix)


def test_writer_registry_binds_to_the_code():
    ids = [w.id for w in P.WRITERS]
    assert len(ids) == len(set(ids))
    for w in P.WRITERS:
        assert type(w.disposition) is P.Disposition and w.bound.strip(), w.id
        if w.disposition in LIVE:
            for entry in w.python:
                assert callable(_resolve(entry)), (w.id, entry)
            for entry in w.native:
                assert _native_present(entry), (w.id, entry)
        else:
            # Retired or replaced: every persistent entry point it had is gone from the code. A file that still
            # exists must not contain the symbol (comments may name it as history).
            for entry in w.python:
                assert _resolve(entry) is None, (w.id, entry)
            for entry in w.native:
                assert not _native_present(entry), (w.id, entry)
        if w.disposition in (P.Disposition.REPLACED_WITH_VOLATILE, P.Disposition.READ_ONLY_STATIC_RESOURCE):
            assert w.replacement.strip(), w.id
        if w.runtime:
            assert w.disposition in P.RUNTIME_DISPOSITIONS, w.id
    # The retired residency parameter is not a parameter at all.
    from elpis.substrate.residency import Context
    assert set(inspect.signature(Context).parameters) == {"library", "warm_bytes", "max_objects"}


def _defined_tests(path: Path) -> set[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"))
    return {n.name for n in tree.body if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))}


def _ctest_names() -> set[str]:
    names = set()
    for cmake in list(NATIVE.rglob("CMakeLists.txt")) + [REPO / "CMakeLists.txt"]:
        text = cmake.read_text(encoding="utf-8")
        for label, name in re.findall(r"elpis_add_test\((\w+)\s+(\w+)", text):
            names.add(f"{label}.{name}")
        names |= set(re.findall(r"add_test\(NAME (\w+\.\w+)", text))
        for name in re.findall(r"semantic_test\((\w+)", text):
            names.add(f"structure.{name}")
    return names


def test_every_piece_of_evidence_exists():
    ctests = _ctest_names()
    nodes = [e for w in P.WRITERS for e in w.evidence] + [e for c in P.CLOSURE for e in c.evidence]
    for node in nodes:
        if node.startswith("ctest:"):
            assert node[len("ctest:"):] in ctests, node
        elif "::" in node:
            path, name = node.split("::")
            assert name in _defined_tests(REPO / path), node
        else:
            assert node == P.COMPOSED_QUALIFICATION, node


# Modules hosting an offline or development writer: a composer's import closure must contain none of them.
def _writer_modules() -> set[str]:
    return {entry.split(":")[0] for w in P.WRITERS if w.disposition in (P.Disposition.OFFLINE_ARTIFACT_PRODUCER,
                                                                         P.Disposition.DEVELOPMENT_ONLY)
            for entry in w.python}


_CLOSURE_PROBE = """
import json, sys
import elpis.runtime, elpis.runtime.composition, elpis.runtime.cli, elpis.runtime.cognition, elpis.runtime.context
import elpis.structure.retrieval.hacf, elpis.pipeline.ingress, elpis.ECS.k1, elpis.substrate.residency
print(json.dumps(sorted(m for m in sys.modules if m == "elpis" or m.startswith("elpis."))))
"""


def test_no_composer_imports_an_offline_or_development_writer():
    """In a fresh interpreter, the Python runtime (and the bindings it uses) loads no module that hosts a writer."""
    probe = subprocess.run([sys.executable, "-B", "-c", _CLOSURE_PROBE], capture_output=True, text=True,
                           env={"PYTHONPATH": str(REPO / "src"), "PATH": os.environ.get("PATH", "")}, timeout=120)
    assert probe.returncode == 0, probe.stderr
    import json
    loaded = set(json.loads(probe.stdout))
    writers = _writer_modules()
    assert {"elpis.evolution.materialize", "elpis.pipeline.canonical.publisher", "elpis.runtime.acceptance"} <= writers
    assert not loaded & writers, sorted(loaded & writers)
    # The autonomous operations reference no writer by name either.
    forbidden = {entry.split(":")[1].split(".")[0] for w in P.WRITERS
                 if w.disposition in (P.Disposition.OFFLINE_ARTIFACT_PRODUCER, P.Disposition.DEVELOPMENT_ONLY)
                 for entry in w.python}
    tree = ast.parse((SRC / "runtime" / "composition.py").read_text(encoding="utf-8"))
    runtime = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "Runtime")
    for node in runtime.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name in P.AUTONOMOUS_OPERATIONS:
            names = {n.id for n in ast.walk(node) if isinstance(n, ast.Name)} | \
                    {n.attr for n in ast.walk(node) if isinstance(n, ast.Attribute)}
            assert not names & forbidden, (node.name, names & forbidden)


def test_each_composer_holds_only_the_volatile_session():
    host = "".join((REPO / "native/runtime/host/src" / f).read_text(encoding="utf-8")
                   for f in sorted(os.listdir(REPO / "native/runtime/host/src")))
    assert re.search(r"\belpis_runtime_create\b", host) and "elpis_runtime_create_ecs_c" not in host
    assert not re.search(r"\b(fms_|elpis_fms_|cold_root)", host)
    core = (REPO / "src/elpis/runtime/core.py").read_text(encoding="utf-8")
    assert "elpis_runtime_create" in core and "elpis_runtime_create_ecs_c" not in core
    for path in (REPO / "src/elpis").rglob("*.py"):
        if path == REPO / "src/elpis/runtime/persistence.py":
            continue
        text = path.read_text(encoding="utf-8")
        assert not re.search(r"libelpis_continuity|libelpis_ecs_c_authority|\belpis_ecs_c_(?:provision|open|"
                             r"reconcile\w*|import\w*)\b|\belpis_continuity_(?!snapshot\b|evolution\b)\w+", text), path


def _native_sources():
    for path in NATIVE.rglob("*"):
        if path.suffix in {".c", ".cpp", ".h", ".rs"} and "tests" not in path.parts:
            yield path


# A persistent write: create/write opens, positional writes, renames, links, unlinks, truncation, directory creation,
# syncs, temporary files.
_WRITES = re.compile(r"\b(pwrite|pwritev|renameat2?|rename|fsync|fdatasync|mkdirat|mkdir|unlinkat|unlink|ftruncate|"
                     r"creat|mkstemp|mkdtemp|tmpfile|posix_fallocate|fallocate)\s*\(|\b(?:sym)?link(?:at)?\s*\([^,()]*,"
                     r"|O_CREAT|O_WRONLY|O_RDWR")
# Sources that name write flags only to refuse them, or only for the testing host's probe (compiled out of the
# production host; the binary gate in test_volatile_runtime.py checks the production host imports no writer).
_REFUSES_WRITES = {"native/runtime/containment/elpis_autonomous_run.c", "native/runtime/host/src/host_contain.c",
                   "native/runtime/host/src/host_admit.c"}
# The bridges Python loads (elpis.structure.retrieval.hacf, elpis.pipeline.ingress).
_LOADED_BRIDGES = ("native/structure/bridge/retrieval_bridge.c", "native/pipeline/ingress/bridge/ingress_bridge.c")


def test_every_native_source_that_writes_is_a_classified_producer_or_development_code():
    classified = {entry.split(":", 1)[0]: w for w in P.WRITERS for entry in w.native
                  if w.disposition in (P.Disposition.OFFLINE_ARTIFACT_PRODUCER, P.Disposition.DEVELOPMENT_ONLY)}
    writers = {str(p.relative_to(REPO)) for p in _native_sources()
               if p.suffix in {".c", ".cpp"} and _WRITES.search(_code(p.read_text(encoding="utf-8", errors="replace"),
                                                                        p.suffix))}
    unclassified = sorted(writers - set(classified) - _REFUSES_WRITES)
    assert not unclassified, ("classify each persistent writer in elpis.runtime.persistence", unclassified)
    # The testing host's probe is the only writer in the host's source, and it is compiled only into the testing host.
    host = (REPO / "native/runtime/host/src/elpis_host.c").read_text(encoding="utf-8")
    at = re.search(r"static int op_testing_write_probe\(", host)
    assert at, "the testing host's write probe"
    guard = host.rfind("#ifdef ELPIS_HOST_TESTING", 0, at.start())
    assert guard >= 0 and host.find("#endif", guard) > at.start(), "the probe is compiled only into the testing host"


def test_no_bridge_python_loads_calls_an_offline_writer():
    symbols = {entry.split(":", 1)[1] for w in P.WRITERS if w.disposition is P.Disposition.OFFLINE_ARTIFACT_PRODUCER
               for entry in w.native if re.fullmatch(r"\w+", entry.split(":", 1)[1])}
    assert {"elpis_write_requirement_set", "elpis_embedding_write_vector", "semantic_snapshot_write"} <= symbols
    pattern = re.compile(r"\b(" + "|".join(sorted(symbols)) + r")\s*\(")
    for path in _LOADED_BRIDGES:
        found = sorted(set(pattern.findall((REPO / path).read_text(encoding="utf-8"))))
        assert not found, (path, found)


def test_the_hgram_writers_do_not_exist():
    kernel = NATIVE / "structure" / "hacf"
    assert not (kernel / "src" / "kernel" / "hgram_init.c").exists()
    for symbol in ("elpis_hgram_init_once", "elpis_hgram_assoc_store_preapproved"):
        users = sorted(str(p.relative_to(REPO)) for p in _native_sources()
                       if re.search(rf"\b{symbol}\s*\(", _code(p.read_text(encoding="utf-8", errors="replace"),
                                                               p.suffix)))
        assert not users, (symbol, users)
    store = _code((kernel / "src" / "kernel" / "hgram_store.c").read_text(encoding="utf-8"), ".c")
    assert "O_RDWR" not in store and "O_WRONLY" not in store and "O_CREAT" not in store
    assert not re.search(r"\b(pwrite|fsync|fdatasync|posix_fallocate|ftruncate|unlinkat|LOCK_EX)\b", store)


def test_hacf_has_no_persistent_corpus_or_ingress_path_and_the_epoch_fms_is_ram_only():
    corpus = (NATIVE / "structure" / "hacf" / "include" / "elpis" / "corpus.h").read_text(encoding="utf-8")
    opens = {line.split("(")[0].split()[-1] for line in corpus.splitlines() if "elpis_corpus_open" in line}
    assert opens == {"elpis_corpus_open_ephemeral"}, opens
    ingress = (NATIVE / "pipeline" / "ingress" / "bridge" / "ingress_bridge.c").read_text(encoding="utf-8")
    body = ingress.split("int elpis_ingress_env_open(", 1)[1].split("\n}\n", 1)[0]
    assert "persistent ingress retired" in body and "open_ephemeral" not in body
    bridge = (NATIVE / "structure" / "bridge" / "retrieval_bridge.c").read_text(encoding="utf-8")
    assert "fms_pal_posix_create_ram_only()" in bridge and "fms_pal_posix_create(" not in bridge
    assert "cfg.tier_budget[FMS_COLD] = 0;" in bridge and "cfg.domain_ceiling[FMS_DOM_STORAGE] = 0;" in bridge


_LOG_PROBE = """
import logging, sys
import elpis.runtime, elpis.runtime.composition, elpis.runtime.cognition
import elpis.structure.retrieval.hacf, elpis.pipeline.ingress
loggers = [logging.getLogger()] + [l for l in logging.Logger.manager.loggerDict.values()
                                   if isinstance(l, logging.Logger)]
handlers = [(l.name, type(h).__name__) for l in loggers for h in l.handlers]
print(handlers)
sys.exit(1 if handlers else 0)
"""


def test_importing_the_runtime_installs_no_persistent_log_sink():
    # A clean interpreter: the test runner's own logging capture is not Elpis's.
    probe = subprocess.run([sys.executable, "-B", "-c", _LOG_PROBE], capture_output=True, text=True,
                           env={"PYTHONPATH": str(REPO / "src")}, timeout=120)
    assert probe.returncode == 0, (probe.stdout, probe.stderr)


@pytest.mark.parametrize("name", ["persistent_hacf_corpus", "persistent_log_sink", "fms_posix_cold_store",
                                  "hgram_fixed_capacity_store", "k1_checkpoint_slots", "runtime_durability_lane"])
def test_the_retired_writers_stay_retired(name):
    assert P.writer(name).disposition is P.Disposition.RETIRED
