"""There is exactly one ECS, and continuity is not one.

* ``elpis.ECS`` exists and is the canonical cognitive/dynamical substrate;
  ``elpis.continuity`` exists; ``elpis.ECS_G`` and ``elpis.ECS_C`` do not.
* Dependency directions: runtime composes ECS and continuity; ECS imports no
  inference, runtime or continuity; continuity imports no ECS, runtime,
  inference or native code; inference imports neither.
* The canonical turn and Runtime.run_turn reach no history, event, scheduler,
  message-bus, projection, compaction, topology or duplicate-index machinery.
* A text scan of every tracked file finds the retired subsystem names only in
  explicitly enumerated immutable evidence or protocol identifiers, each with
  a path and a reason (no pattern exemptions).
"""
from __future__ import annotations

import ast
import hashlib
import inspect
import os
from pathlib import Path
import re
import subprocess
import sys

import pytest

from ._mission import imports_of
from ._system import REPO, SYSTEM


def _clean_import(module: str) -> subprocess.CompletedProcess:
    env = {"PATH": os.environ.get("PATH", ""), "PYTHONPATH": str(REPO / "src"), "PYTHONDONTWRITEBYTECODE": "1"}
    return subprocess.run([sys.executable, "-c", f"import {module}"], capture_output=True, text=True, env=env)


def test_one_ecs_and_continuity_exist_and_the_retired_names_do_not():
    assert _clean_import("elpis.ECS").returncode == 0
    assert _clean_import("elpis.continuity").returncode == 0
    for retired in ("elpis.ECS_G", "elpis.ECS_C"):
        result = _clean_import(retired)
        assert result.returncode != 0 and "ModuleNotFoundError" in result.stderr, retired
    for path in ("src/elpis/ECS_C", "src/elpis/ECS_G", "native/ECS_C", "native/ECS_G", "tests/ECS_C", "tests/ECS_G"):
        assert not (REPO / path).exists(), path
    ids = [s["id"] for s in SYSTEM["subsystems"]]
    assert "ECS" in ids and "continuity" in ids
    assert not [i for i in ids if i.startswith("ECS_")]


def _imports(package: str) -> set[str]:
    names = set()
    for path in (REPO / "src" / "elpis" / package).rglob("*.py"):
        names |= imports_of(REPO, path)
    return names


def _under(name: str, prefix: str) -> bool:
    return name == prefix or name.startswith(prefix + ".")


def test_dependency_directions_keep_runtime_as_the_only_composer():
    ecs = _imports("ECS")
    assert not [n for n in ecs if any(_under(n, p) for p in
                                      ("elpis.inference", "elpis.runtime", "elpis.continuity", "research"))]
    continuity = _imports("continuity")
    assert not [n for n in continuity if _under(n, "elpis") and not _under(n, "elpis.continuity")]
    assert not [n for n in continuity if n.split(".")[0] in ("numpy", "research")]
    # The continuity authority is the Rust crate: it depends on nothing but the Rust standard library, and it
    # knows nothing of ECS, K1 execution, runtime or inference (it stores a K1 digest, never K1 state).
    crate = REPO / "native/continuity"
    cargo = (crate / "Cargo.toml").read_text()
    assert re.search(r"(?ms)^\[dependencies\]\s*(?:#[^\n]*\n\s*)*(?=^\[|\Z)", cargo), "no crate dependencies"
    assert "[build-dependencies]" not in cargo and "[dev-dependencies]" not in cargo
    rust = "\n".join(p.read_text() for p in (crate / "src").rglob("*.rs"))
    for foreign in ("elpis_ecsg", "ecsg_", "extern crate", "#[link(", "dsv", "inference"):
        assert foreign not in rust, foreign
    # RuntimeCore embeds the continuity crate and nothing else; it links no ECS code (K1 is reached only through
    # caller-supplied function tables) and knows nothing of inference or DSV.
    runtime_crate = REPO / "native/runtime"
    cargo = (runtime_crate / "Cargo.toml").read_text()
    deps = cargo[cargo.index("[dependencies]"):].split("\n[", 1)[0]
    assert re.findall(r"(?m)^(\w+)\s*=", deps) == ["elpis_continuity"], deps
    assert "[build-dependencies]" not in cargo and "[dev-dependencies]" not in cargo
    rust = "\n".join(line.split("//")[0] for p in (runtime_crate / "src").rglob("*.rs")
                     for line in p.read_text().splitlines())   # code only: doc comments cite the K1 ABI it mirrors
    for foreign in ("elpis_ecsg", "extern crate", "#[link(", "dlopen", "dsv", "inference"):
        assert foreign not in rust, foreign
    inference = _imports("inference")
    assert not [n for n in inference if _under(n, "elpis.ECS") or _under(n, "elpis.continuity")]
    cognition = imports_of(REPO, REPO / "src/elpis/runtime/cognition.py")
    assert any(_under(n, "elpis.ECS") for n in cognition)
    composition = imports_of(REPO, REPO / "src/elpis/runtime/composition.py")
    assert any(_under(n, "elpis.continuity") for n in composition)
    subs = {s["id"]: s for s in SYSTEM["subsystems"]}
    assert {"ECS", "continuity"} <= set(subs["runtime"]["depends_on"])
    assert "continuity" not in subs["ECS"]["depends_on"] and "ECS" not in subs["continuity"]["depends_on"]


# Machinery the canonical hot path must never reach (docs/CONTINUITY.md).
FORBIDDEN_HOT_PATH = re.compile(
    r"(?i)\b(receipt|record\(|records\(|history|event_index|enqueue|processed|scheduler|mailbox|entity_port|"
    r"propose|run_until_quiescent|projection|compact|segment|topology|replay|duplicate|retention_floor)\w*")


def test_the_canonical_turn_has_no_history_machinery_statically():
    import elpis.runtime.cognition as cognition
    from elpis.runtime.composition import Runtime
    from elpis.runtime.core import RuntimeCore, describe

    for source in (inspect.getsource(cognition), inspect.getsource(Runtime.run_turn), inspect.getsource(describe),
                   inspect.getsource(RuntimeCore.turn_begin), inspect.getsource(RuntimeCore.turn_commit),
                   inspect.getsource(RuntimeCore.turn_abort)):
        code = ast.unparse(_strip_docstrings(ast.parse(_dedent(source))))
        assert not FORBIDDEN_HOT_PATH.findall(code), FORBIDDEN_HOT_PATH.findall(code)
    # RuntimeCore's managed turn (Rust): lineage, the native transaction and one publication, nothing else.
    core = (REPO / "native/runtime/src/core.rs").read_text()
    turn = core[core.index("// -- K1 lineage"):core.index("// -- evolution")]
    code = "\n".join(line.split("//")[0] for line in turn.splitlines())
    assert not FORBIDDEN_HOT_PATH.findall(code), FORBIDDEN_HOT_PATH.findall(code)


def _dedent(source: str) -> str:
    import textwrap
    return textwrap.dedent(source)


def _strip_docstrings(tree):
    for node in ast.walk(tree):
        body = getattr(node, "body", None)
        if (isinstance(body, list) and body and isinstance(body[0], ast.Expr)
                and isinstance(body[0].value, ast.Constant) and isinstance(body[0].value.value, str)):
            node.body = body[1:] or [ast.Pass()]
    return tree


def test_continuity_surface_is_current_authority_only():
    from elpis.continuity import ContinuityStore

    public = {n for n in vars(ContinuityStore) if not n.startswith("_")}
    # testing_fault / testing_counters reach only the separately built testing library; the production
    # library has no such symbols and the adapter refuses them (tests/continuity/test_adapter.py).
    assert public == {"open", "close", "snapshot", "anchor_cognition", "commit_cognition_transition",
                      "reserve_evolution_assertion", "commit_evolution_transition", "testing_fault",
                      "testing_counters"}
    header = (REPO / "native/continuity/include/elpis/continuity.h").read_text()
    exported = set(re.findall(r"\b(elpis_continuity_\w+)\s*\(", header))
    assert exported == {f"elpis_continuity_{n}" for n in (
        "abi_version", "record_size", "code_name", "evolution_digest", "record_encode", "record_decode",
        "store_create", "store_destroy", "store_open", "store_close", "store_snapshot", "anchor_cognition",
        "commit_cognition", "reserve_evolution", "finalize_evolution")}, exported
    banned = re.compile(r"(?i)\b(entit(?:y|ies)|mailbox\w*|scheduler\w*|topology|projection\w*|replay\w*|"
                        r"compact\w*|segment\w*|retention\w*|events?|event_log|receipt_history|propose\w*)\b")
    for path in (REPO / "src/elpis/continuity").rglob("*.py"):
        # Identifiers only: the legacy-layout names it refuses (``events.log``) are string data, not machinery.
        names = set()
        for node in ast.walk(ast.parse(path.read_text())):
            if isinstance(node, ast.Name):
                names.add(node.id)
            elif isinstance(node, ast.Attribute):
                names.add(node.attr)
            elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                names.add(node.name)
            elif isinstance(node, ast.arg):
                names.add(node.arg)
            elif isinstance(node, ast.alias):
                names.add(node.asname or node.name)
        found = [n for n in names if banned.search(n.replace("_", " "))]
        assert not found, (path, found)
    for path in (REPO / "native/continuity/src").rglob("*.rs"):
        if path.name == "tests.rs":
            continue
        # Rust identifiers (fn/struct/enum/variant/field names); comments and string data excluded.
        code = re.sub(r"//[^\n]*|\"(?:\\.|[^\"\\])*\"", "", path.read_text())
        found = [n for n in re.findall(r"\b[A-Za-z_][A-Za-z0-9_]*\b", code) if banned.search(
            re.sub(r"(?<=[a-z])(?=[A-Z])", " ", n).replace("_", " "))]
        assert not found, (path, sorted(set(found)))


# -- the residue scan ----------------------------------------------------------------------------------

# A retired subsystem token: ECS_C / ECS_G / ecs_c / ecs_g / ecsc / ecsg / ECSC / ECSG as a word or identifier part. It is not part
# of a longer word (``ECS_CODEC_UNQUALIFIED`` and ``ECSCodecMap`` are other names) except as a CamelCase prefix
# (``ECSGLibrary``).
RETIRED_TOKEN = re.compile(r"(?<![A-Za-z0-9])(?:ECS_[CG]|ecs_[cg]|ecs[cg]|ECS[CG])(?:(?![A-Za-z0-9])|(?=[A-Z][a-z]))")

# The retained, qualified native ECS ABI and persisted protocol identifiers (docs/ARCHITECTURE.md, ECS section;
# ELPIS_SYSTEM.json ``protocol_identifiers_retained``). These are identifier *forms*; they are accepted only inside
# the files enumerated in NATIVE_ABI_FILES below, never globally.
NATIVE_ABI_FORM = re.compile(
    r"\w*ecsg_\w*"                      # elpis_ecsg_* symbols, ecsg_*.h/.c files, libelpis_ecsg_* / test_ecsg_* targets
    r"|ELPIS_ECSG_\w*"                   # ELPIS_ECSG_* ABI constants
    r"|ECSG(?:Library|Error)\b"          # the ctypes binding classes
    r"|ecsg-k1-native(?:\.v\d+)?"         # the K1 native qualification authority id
    r"|elpis\.ecsg\.[\w.-]+")              # persisted schema / digest-domain strings

# Immutable evidence: any form may appear (recorded names, measured paths, frozen source text).
FROZEN_EVIDENCE = {
    "research/ecs_cognition_r0/": "Closed Cognition R0 laboratory: byte-frozen source and recorded evidence.",
    "research/ecs_k1_native/": "K1 native qualification: byte-pinned plans, harness and the ecsg-k1-native attestation.",
    "research/ecs_retention_r0/": "Closed Retention R0 laboratory: byte-frozen source and recorded evidence.",
    "research/ecs_retention_r1/": "Closed Retention R1 laboratory: byte-frozen source and recorded evidence.",
    "research/ecs_retention_r2/": "Closed Retention R2 laboratory: byte-frozen source and recorded evidence.",
    "research/ecs_retention_r3/": "Closed Retention R3 laboratory: byte-frozen source and recorded evidence.",
    "research/ecs_runtime_r1/": "Runtime R1 performance laboratory: recorded paths and digests of the measured sources.",
    "docs/research/": "Result reports of the closed laboratories, cited by digest; never rewritten.",
    "tests/research/": "Guards over the recorded evidence: they quote recorded paths and state the one rename.",
    "research/__init__.py": "Frozen-laboratory import name elpis.ECS_G resolved to elpis.ECS, research-only.",
    "tests/boundary/test_one_ecs.py": "This gate names the retired tokens it rejects.",
    "research/hecs_r1/evidence/": "H-ECS R1 write-once evidence: each file embeds the frozen specification (K1 ABI name).",
    "research/hecs_r2/evidence/": "H-ECS R2 compact phase records: each names the K1 library it ran against (K1 ABI name).",
    "research/hecs_r3/unresolved/successor_dev_3_of_8/frozen_boundary/":
        "H-ECS R3 successor frozen boundary: the digest-pinned supervisor source names the K1 library (K1 ABI name).",
    "research/hecs_r3/closure/successor_dev_8_of_8/reconstruction/":
        "H-ECS R3 successor reconstruction record: the adapter source names the K1 library it ran (K1 ABI name).",
    "research/hecs_r3/successor_c2/freeze/source/supervisor/":
        "H-ECS R3 successor C2 frozen supervisor source: names the K1 library it runs (K1 ABI name).",
}

# Native sources whose exact bytes are recorded by research/ecs_runtime_r1/evidence (measured and sanitized
# sources) and pinned by the mission gate; their comments, macros and PASS strings keep the old name.
BYTE_BOUND = {
    "native/ECS/include/elpis/ecsg_executor.h", "native/ECS/include/elpis/ecsg_math.h",
    "native/ECS/include/elpis/ecsg_state.h", "native/ECS/src/ecsg_executor.c", "native/ECS/src/ecsg_math.c",
    "native/ECS/src/ecsg_state.c", "native/ECS/tests/test_ecsg_executor.c",
    "native/ECS/tests/test_ecsg_executor_alloc.c", "native/ECS/tests/test_ecsg_executor_txn.c",
    "native/ECS/tests/test_ecsg_math.c", "native/ECS/tests/test_ecsg_snapshot.c", "native/ECS/tests/test_ecsg_state.c",
}
BYTE_BOUND_REASON = "Exact bytes recorded by research/ecs_runtime_r1/evidence; unchanged across the rename."

_ABI_SYSTEM = "System authority: classifies the retained native ABI and persisted protocol identifiers."
_ABI_DOC = "Documents native ECS ABI, library and target names and persisted protocol identifiers."
_ABI_NATIVE = "Native ECS source, build or test: declares, implements or links the qualified native ABI."
_ABI_BINDING = "ECS binding: loads the native ABI and names persisted schema and digest-domain identifiers."
_ABI_TEST = "Exercises or pins the native ABI and persisted protocol identifiers."
_HECS_K1 = "H-ECS R0 drives each level's K1 state through the qualified native ABI (elpis_ecsg_k1)."
_HECS_R1_K1 = "H-ECS R1 drives each level's K1 state through the qualified native ABI (elpis_ecsg_k1)."
_HECS_R2_K1 = "H-ECS R2 drives each level's K1 state through the qualified native ABI (elpis_ecsg_k1)."
_RUNTIME_K1 = "RuntimeCore reaches K1 only through tables of the qualified native ABI's entry points (elpis_ecsg_k1)."

# Files that may contain retired tokens only as NATIVE_ABI_FORM identifiers.
NATIVE_ABI_FILES = {
    "ELPIS_SYSTEM.json": _ABI_SYSTEM,
    "docs/ARCHITECTURE.md": _ABI_DOC,
    "docs/COGNITION_R0.md": _ABI_DOC,
    "docs/ECS_K1_RUNTIME.md": _ABI_DOC,
    "docs/ECS_MUTABLE_FMS_R0.md": _ABI_DOC,
    "docs/ECS_RUNTIME_R1.md": _ABI_DOC,
    "docs/performance/ECS_K1_RUNTIME.md": _ABI_DOC,
    "docs/performance/ECS_RUNTIME_R1.md": _ABI_DOC,
    "native/ECS/CMakeLists.txt": _ABI_NATIVE,
    "native/ECS/README.md": _ABI_NATIVE,
    "native/ECS/include/elpis/ecsg_fms.h": _ABI_NATIVE,
    "native/ECS/include/elpis/ecsg_k1.h": _ABI_NATIVE,
    "native/ECS/include/elpis/ecsg_k1_fms.h": _ABI_NATIVE,
    "native/ECS/src/ecsg_fms.c": _ABI_NATIVE,
    "native/ECS/src/ecsg_k1.c": _ABI_NATIVE,
    "native/ECS/src/ecsg_k1_fms.c": _ABI_NATIVE,
    "native/ECS/src/ecsg_k1_internal.h": _ABI_NATIVE,
    "native/ECS/tests/test_ecsg_fms.c": _ABI_NATIVE,
    "native/ECS/tests/test_ecsg_fms_alloc.c": _ABI_NATIVE,
    "native/ECS/tests/test_ecsg_fms_performance.c": _ABI_NATIVE,
    "native/ECS/tests/test_ecsg_k1.c": _ABI_NATIVE,
    "native/ECS/tests/test_ecsg_k1_alloc.c": _ABI_NATIVE,
    "native/ECS/tests/test_ecsg_k1_fms.c": _ABI_NATIVE,
    "native/ECS/tests/test_ecsg_k1_fms_alloc.c": _ABI_NATIVE,
    "native/ECS/tests/test_ecsg_k1_fms_faults.c": _ABI_NATIVE,
    "native/ECS/tests/test_ecsg_k1_performance.c": _ABI_NATIVE,
    "src/elpis/ECS/__init__.py": _ABI_BINDING,
    "src/elpis/ECS/cognition.py": _ABI_BINDING,
    "src/elpis/ECS/k1.py": _ABI_BINDING,
    "src/elpis/ECS/native.py": _ABI_BINDING,
    "src/elpis/ECS/residency.py": _ABI_BINDING,
    "tests/ECS/test_cognition_r0_contract.py": _ABI_TEST,
    "tests/ECS/test_cognition_r0_core.py": _ABI_TEST,
    "tests/ECS/test_executor_binding.py": _ABI_TEST,
    "tests/ECS/test_executor_differential.py": _ABI_TEST,
    "tests/ECS/test_k1_runtime.py": _ABI_TEST,
    "tests/ECS/test_math_r0.py": _ABI_TEST,
    "tests/ECS/test_mutable_fms.py": _ABI_TEST,
    "tests/ECS/test_native_binding.py": _ABI_TEST,
    "tests/ECS/test_snapshot_r0.py": _ABI_TEST,
    "tests/ECS/test_stateful_recurrence_r0.py": _ABI_TEST,
    "tests/boundary/_mission.py": _ABI_TEST,
    "tests/boundary/test_k1_commit_identity.py": _ABI_TEST,
    "tests/boundary/test_k1_runtime.py": _ABI_TEST,
    "tests/boundary/test_k1_state_digest.py": _ABI_TEST,
    "tests/boundary/test_mutable_fms.py": _ABI_TEST,
    "tests/integration/test_codec_ecs_turn.py": _ABI_TEST,
    "tests/integration/test_runtime_hot_path.py": _ABI_TEST,
    "native/continuity/qualify.sh": "Builds the ECS K1 libraries (native target names) the continuity runtime tests load.",
    "docs/RUNTIME_CORE.md": _RUNTIME_K1,
    "native/runtime/CMakeLists.txt": _RUNTIME_K1,
    "native/runtime/include/elpis/runtime.h": _RUNTIME_K1,
    "native/runtime/src/ffi.rs": _RUNTIME_K1,
    "native/runtime/src/lib.rs": _RUNTIME_K1,
    "native/runtime/src/substrate.rs": _RUNTIME_K1,
    "native/runtime/src/tests.rs": _RUNTIME_K1,
    "native/runtime/tests/test_runtime_abi.c": _RUNTIME_K1,
    "native/runtime/tests/test_runtime_lifecycle.c": _RUNTIME_K1,
    "src/elpis/runtime/core.py": _RUNTIME_K1,
    "research/hecs_r1/CMakeLists.txt": _HECS_R1_K1,
    "research/hecs_r1/README.md": _HECS_R1_K1,
    "research/hecs_r1/rust/src/k1.rs": _HECS_R1_K1,
    "research/hecs_r1/rust/src/main.rs": _HECS_R1_K1,
    "research/hecs_r1/rust/src/spec.rs": _HECS_R1_K1,
    "research/hecs_r1/rust/src/tests.rs": _HECS_R1_K1,
    "research/hecs_r1/specs/hecs-r1.v1.spec.json": _HECS_R1_K1,
    "research/hecs_r2/CMakeLists.txt": _HECS_R2_K1,
    "research/hecs_r2/README.md": _HECS_R2_K1,
    "research/hecs_r2/specs/hecs-r2.v1.spec.json": _HECS_R2_K1,
    "research/hecs_r2/rust/src/k1.rs": _HECS_R2_K1,
    "research/hecs_r2/rust/src/main.rs": _HECS_R2_K1,
    "research/hecs_r2/rust/src/spec.rs": _HECS_R2_K1,
    "research/hecs_r2/rust/src/tests.rs": _HECS_R2_K1,
    "research/hecs_r0/CMakeLists.txt": _HECS_K1,
    "research/hecs_r0/PREREGISTRATION.md": _HECS_K1,
    "research/hecs_r0/README.md": _HECS_K1,
    "research/hecs_r0/rust/src/k1.rs": _HECS_K1,
    "research/hecs_r0/rust/src/main.rs": _HECS_K1,
    "research/hecs_r0/rust/src/spec.rs": _HECS_K1,
    "research/hecs_r0/rust/src/tests.rs": _HECS_K1,
    "research/hecs_r0/specs/hecs-r0.v1.spec.json": _HECS_K1,
    "research/hecs_r0/evidence/dev/hecs-r0.v1.calibration.json": _HECS_K1,
}


def _tracked() -> list[str]:
    out = subprocess.run(["git", "ls-files", "-z"], cwd=REPO, capture_output=True, check=True).stdout
    return [p for p in out.decode().split("\0") if p]


def _frozen(path: str) -> bool:
    return any(path == key or (key.endswith("/") and path.startswith(key)) for key in FROZEN_EVIDENCE)


def _text(path: str) -> str | None:
    try:
        return (REPO / path).read_text(encoding="utf-8")
    except (UnicodeDecodeError, FileNotFoundError, IsADirectoryError):
        return None


def test_retired_names_appear_only_in_enumerated_evidence_and_abi_identifiers():
    violations = []
    for path in _tracked():
        if _frozen(path) or path in BYTE_BOUND:
            continue
        text = _text(path)
        if text is None:
            continue
        hits = list(RETIRED_TOKEN.finditer(text))
        if not hits:
            continue
        if path not in NATIVE_ABI_FILES:
            violations.append((path, text[hits[0].start():hits[0].end() + 24]))
            continue
        spans = [m.span() for m in NATIVE_ABI_FORM.finditer(text)]
        for hit in hits:
            if not any(a <= hit.start() and hit.end() <= b for a, b in spans):
                violations.append((path, text[max(0, hit.start() - 16):hit.end() + 24]))
    assert not violations, violations


def test_every_exemption_is_real_and_reasoned():
    tracked = _tracked()
    for key, why in FROZEN_EVIDENCE.items():
        assert why.strip(), key
        members = [p for p in tracked if p == key or (key.endswith("/") and p.startswith(key))]
        assert any(RETIRED_TOKEN.search(_text(p) or "") for p in members), f"stale exemption: {key}"
    for path, why in NATIVE_ABI_FILES.items():
        assert why.strip() and RETIRED_TOKEN.search(_text(path) or ""), f"stale exemption: {path}"
    assert BYTE_BOUND_REASON.strip()
    evidence = "".join(p.read_text() for p in (REPO / "research/ecs_runtime_r1/evidence").glob("*.json"))
    for path in BYTE_BOUND:
        digest = hashlib.sha256((REPO / path).read_bytes()).hexdigest()
        assert digest in evidence, f"{path} is no longer the recorded bytes; it is not byte-bound"
    assert not set(NATIVE_ABI_FILES) & BYTE_BOUND
