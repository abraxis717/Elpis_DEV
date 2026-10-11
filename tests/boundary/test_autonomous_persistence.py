"""Zero autonomously ever-expanding Elpis-owned persistence (elpis.runtime.persistence).

Architectural properties, not keyword bans:

* every public Runtime operation is classified exactly once (autonomous or operator);
* the persistent-writer registry resolves to real code, and each writer has exactly one class;
* the autonomous Runtime operations and the modules they delegate to reference no operator writer;
* native writers documented as having no Python binding have none anywhere in ``src/``;
* the H-gram is provisioned and written only by its operator command and its explicit entry points;
* the HACF corpus and ingress have no persistent path, and the retrieval epoch's FMS is RAM-only;
* importing the runtime installs no persistent log sink.

The same table is proven on real native libraries under a write trap in
``tests/integration/test_autonomous_no_growth.py``.
"""
from __future__ import annotations

import ast
import importlib
import inspect
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


def _public_operations() -> set[str]:
    return {name for name, _ in inspect.getmembers(Runtime) if not name.startswith("_")}


def test_every_runtime_operation_is_classified_exactly_once():
    operations = _public_operations()
    assert not (P.AUTONOMOUS_OPERATIONS & P.OPERATOR_OPERATIONS)
    assert P.AUTONOMOUS_OPERATIONS | P.OPERATOR_OPERATIONS == operations, (
        "classify every public Runtime operation in elpis.runtime.persistence",
        sorted(operations ^ (P.AUTONOMOUS_OPERATIONS | P.OPERATOR_OPERATIONS)))


def test_writer_registry_resolves_to_real_code_with_one_class_each():
    ids = [w.id for w in P.WRITERS]
    assert len(ids) == len(set(ids))
    for w in P.WRITERS:
        assert type(w.classification) is P.WriterClass and w.bound.strip(), w.id
        if not (w.python or w.native):
            # Only the absence of a writer may be documented without an entry point.
            assert w.classification in (P.WriterClass.OFFLINE_RESEARCH_ONLY,
                                        P.WriterClass.PROHIBITED_FROM_AUTONOMOUS_RUNTIME), w.id
        for entry in w.python:
            module, qualname = entry.split(":")
            target = importlib.import_module(module)
            for part in qualname.split("."):
                target = getattr(target, part)
            assert callable(target), entry
        for entry in w.native:
            path, symbol = entry.split(":", 1)
            assert symbol and symbol in (REPO / path).read_text(encoding="utf-8"), entry
    assert P.AUTONOMOUS_WRITERS == {"continuity_slots", "fms_posix_cold_store", "k1_checkpoint_slots"}


def _operator_writer_names() -> set[str]:
    """Names by which Python code would reach a non-autonomous writer."""
    names = set()
    for w in P.WRITERS:
        if w.classification is P.WriterClass.FIXED_CAPACITY_AUTONOMOUS:
            continue
        for entry in w.python:
            module, qualname = entry.split(":")
            parts = qualname.split(".")
            names.add(parts[0])            # the function or the writer class
    return names


def _referenced_names(node: ast.AST) -> set[str]:
    out = set()
    for n in ast.walk(node):
        if isinstance(n, ast.Name):
            out.add(n.id)
        elif isinstance(n, ast.Attribute):
            out.add(n.attr)
        elif isinstance(n, (ast.Import, ast.ImportFrom)):
            out.update(a.asname or a.name.split(".")[-1] for a in n.names)
    return out


def test_autonomous_runtime_operations_reference_no_operator_writer():
    forbidden = _operator_writer_names()
    assert {"publish_candidate", "atomic_materialize", "apply_artifact"} <= forbidden
    tree = ast.parse((SRC / "runtime" / "composition.py").read_text(encoding="utf-8"))
    runtime = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "Runtime")
    methods = {n.name: n for n in runtime.body if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))}
    for name in P.AUTONOMOUS_OPERATIONS:
        node = methods.get(name)
        if node is None:      # a property or attribute: resolved through the class, checked above
            continue
        assert not (_referenced_names(node) & forbidden), (name, _referenced_names(node) & forbidden)
    # The modules the autonomous turn delegates to reference no operator writer at all.
    for module in ("cognition.py", "core.py", "edges.py"):
        names = _referenced_names(ast.parse((SRC / "runtime" / module).read_text(encoding="utf-8")))
        assert not (names & forbidden), (module, names & forbidden)


def _native_symbol(entry: str) -> str:
    return entry.split(":", 1)[1]


def test_operator_native_writers_have_no_python_binding():
    unbound = [w for w in P.WRITERS if w.native and not w.python
               and w.classification is P.WriterClass.OPERATOR_EXPLICIT_BOUNDED]
    symbols = {_native_symbol(e) for w in unbound for e in w.native if _native_symbol(e).startswith(("elpis_", "semantic_"))}
    assert {"elpis_hgram_init_once", "elpis_hgram_assoc_store_preapproved", "semantic_snapshot_write",
            "semantic_b2b_head_cas", "semantic_b2c_publish_one"} <= symbols
    pattern = re.compile(r"\b(" + "|".join(sorted(symbols)) + r")\b")
    registry = SRC / "runtime" / "persistence.py"
    for path in SRC.rglob("*.py"):
        if path == registry:
            continue
        found = sorted(set(pattern.findall(path.read_text(encoding="utf-8"))))
        assert not found, (path.relative_to(REPO), found)


def _native_sources():
    for path in NATIVE.rglob("*"):
        if path.suffix in {".c", ".cpp", ".h", ".rs"} and "tests" not in path.parts:
            yield path


def test_hgram_is_provisioned_and_written_only_through_its_explicit_entry_points():
    kernel = NATIVE / "structure" / "hacf"
    allowed = {kernel / "src" / "kernel" / "hgram_store.c", kernel / "src" / "kernel" / "hgram_init.c",
               kernel / "include" / "elpis" / "hgram_store.h"}
    for symbol, callers in (("elpis_hgram_init_once", allowed),
                            ("elpis_hgram_assoc_store_preapproved", allowed - {kernel / "src" / "kernel" / "hgram_init.c"})):
        users = {p for p in _native_sources() if symbol in p.read_text(encoding="utf-8", errors="replace")}
        assert users <= callers, (symbol, sorted(str(p.relative_to(REPO)) for p in users - callers))


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
    probe = subprocess.run([sys.executable, "-c", _LOG_PROBE], capture_output=True, text=True,
                           env={"PYTHONPATH": str(REPO / "src")}, timeout=120)
    assert probe.returncode == 0, (probe.stdout, probe.stderr)


@pytest.mark.parametrize("name", ["persistent_hacf_corpus", "persistent_log_sink"])
def test_prohibited_autonomous_writers_stay_prohibited(name):
    assert P.writer(name).classification is P.WriterClass.PROHIBITED_FROM_AUTONOMOUS_RUNTIME
