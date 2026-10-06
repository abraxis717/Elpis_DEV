"""The runtime receipt history is bounded, and the authority documents say so.

Static gates over source and documents. The behavioural proofs (finite disk
across compactions, bounded restart replay, crash matrix, retention) live in
tests/ECS_C/test_runtime_history_bounded.py and test_compaction_base.py; these
gates keep the code and the authority from drifting back to an unbounded or
lifetime-claiming design.
"""
from __future__ import annotations

import ast
import dataclasses
import inspect
import json
import re
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
HISTORY = REPO / "src/elpis/runtime/history.py"
NATIVE = REPO / "src/elpis/runtime/native_history.py"
GENERATIONS = REPO / "src/elpis/ECS_C/generations.py"
COMPACTION = REPO / "src/elpis/ECS_C/compaction.py"

STALE_CLAIMS = (
    "re-reads the event log",
    "recording is idempotent:",
    "recording is idempotent and",
    "opening replays and verifies the whole event chain",
    "ecs turns are not recorded",
)


def test_policy_has_only_finite_integer_bounds():
    from elpis.runtime.history import RuntimeHistoryPolicy

    fields = dataclasses.fields(RuntimeHistoryPolicy)
    assert {f.name for f in fields} == {
        "max_segment_bytes", "max_segment_events", "max_checkpoint_bytes", "max_directory_bytes"}
    default = RuntimeHistoryPolicy()
    for field in fields:
        # No Optional / None / zero "unlimited" encoding exists.
        assert field.type in ("int", int), field
        value = getattr(default, field.name)
        assert type(value) is int and 0 < value <= RuntimeHistoryPolicy.MAX_BOUND
    assert default.max_directory_bytes >= default.required_directory_bytes


def test_runtime_config_policy_default_is_finite():
    from elpis.runtime import RuntimeConfig
    from elpis.runtime.history import RuntimeHistoryPolicy

    field = {f.name: f for f in dataclasses.fields(RuntimeConfig)}["history_policy"]
    assert field.default == RuntimeHistoryPolicy()


def test_runtime_binds_only_the_bounded_native_segment_session():
    text = NATIVE.read_text(encoding="utf-8")
    calls = set(re.findall(r"elpis_ecsc_runtime_session_\w+", text))
    assert "elpis_ecsc_runtime_session_open_segment" in calls
    # The original whole-log open (no base, no policy) is never used.
    assert "elpis_ecsc_runtime_session_open" not in calls


def test_record_hot_path_has_no_python_write_fallback():
    from elpis.runtime.history import ReceiptHistory

    source = inspect.getsource(ReceiptHistory.record)
    for forbidden in ("_ports", "propose", "run_until_quiescent", "_kernel", "events("):
        assert forbidden not in source, forbidden


def test_no_archive_or_compression_in_the_history_path():
    banned = {"gzip", "zlib", "lzma", "bz2", "tarfile", "zipfile", "shutil"}
    for path in (HISTORY, NATIVE, GENERATIONS, COMPACTION):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        names = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                names |= {a.name.split(".")[0] for a in node.names}
            elif isinstance(node, ast.ImportFrom) and node.module:
                names.add(node.module.split(".")[0])
        assert not names & banned, (path, names & banned)


def test_retention_errors_are_stable_codes_in_source():
    text = HISTORY.read_text(encoding="utf-8")
    for code in ("HISTORY_STORAGE_CAPACITY", "HISTORY_COMPACTION_REQUIRED", "HISTORY_COMPACTION_FAILED",
                 "HISTORY_CHECKPOINT_INVALID", "HISTORY_GENERATION_MISMATCH", "HISTORY_BELOW_RETENTION_FLOOR",
                 "HISTORY_LEGACY_MIGRATION_FAILED", "HISTORY_NATIVE_SEGMENT_MISMATCH"):
        assert code in text, code


def test_authority_documents_drop_lifetime_claims_and_state_the_bounds():
    docs = {name: (REPO / name).read_text(encoding="utf-8")
            for name in ("ELPIS_SYSTEM.json", "docs/ARCHITECTURE.md", "docs/NONCLAIMS.md")}
    for name, text in docs.items():
        lowered = text.lower()
        found = [claim for claim in STALE_CLAIMS if claim in lowered]
        assert not found, (name, found)

    architecture = docs["docs/ARCHITECTURE.md"]
    for phrase in ("RuntimeHistoryPolicy", "retention_floor", "only within the retained window",
                   "ecs.compaction-checkpoint.v1", "SEGMENT_FULL", "HISTORY_BELOW_RETENTION_FLOOR"):
        assert phrase in architecture, phrase
    nonclaims = docs["docs/NONCLAIMS.md"]
    for phrase in ("No lifetime history", "only within the retained window", "Crash model"):
        assert phrase in nonclaims, phrase

    system = json.loads(docs["ELPIS_SYSTEM.json"])
    subs = {s["id"]: s for s in system["subsystems"]}
    ids = subs["ECS_C"]["protocol_identifiers_retained"]
    assert {"ecs.compaction-checkpoint.v1", "ecs.generation-manifest.v1",
            "ecs.context-projection.retained.v1"} <= set(ids)
    runtime = subs["runtime"]
    assert "only within the retained window" in runtime["mutation_authority"]
    assert any("retention floor" in item for item in runtime["incomplete_interfaces"])
    assert any("lifetime idempotence" in item for item in system["nonclaims"])
