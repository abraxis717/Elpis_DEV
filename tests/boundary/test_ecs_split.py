"""The ECS has two native owners, ECS_G and ECS_C, and only RuntimeCore composes them (docs/NATIVE_SPLIT_CENSUS.md).

The successor of the one-ECS gate (tests/boundary/test_one_ecs.py, retired with the one-ECS tree by Stage B of
the native split). It keeps every prohibition that still applies and adds the split's:

* ``native/ECS_G`` (geometry: math, Runtime R1 executor, mutable FMS, K1, K1 FMS) and ``native/ECS_C`` (the memory
  and the authority) exist, ``native/ECS`` does not; ELPIS_SYSTEM.json declares ``ECS_G`` and ``ECS_C`` and no ``ECS``.
* Dependency directions, in the sources, the build graph and the authority: ECS_G depends on the substrate only;
  ECS_C on the substrate and, while its authority embeds the continuity crate (until Stage F), on continuity; neither
  ever on the other or on RuntimeCore; the composite (both domains in one object) is RuntimeCore's; RuntimeCore's
  crate still embeds only continuity and links no ECS code. The built binaries' link graph is checked by
  tests/boundary/test_abi_inventory.py and tests/boundary/test_ecs_c_authority.py.
* Python is not the target execution architecture: no ``elpis.ECS_G`` / ``elpis.ECS_C`` package exists (the
  frozen laboratories' ``elpis.ECS_G`` import name resolves only through research/__init__.py), the deleted ECS_C
  Python kernel and its native history ABI never reappear in canonical code.
* The canonical turn reaches no history, event, scheduler, message-bus, projection, compaction, topology or
  duplicate-index machinery, and RuntimeCore's HOT functions touch no store, lane, sync or digest.
* The native sources the Runtime R1 evidence measured are byte-identical at their recorded paths.
"""
from __future__ import annotations

import ast
import hashlib
import inspect
import json
import os
from pathlib import Path
import re
import subprocess
import sys

from ._mission import ecs_independence, imports_of
from ._system import REPO, SYSTEM


def _clean_import(module: str) -> subprocess.CompletedProcess:
    env = {"PATH": os.environ.get("PATH", ""), "PYTHONPATH": str(REPO / "src"), "PYTHONDONTWRITEBYTECODE": "1"}
    return subprocess.run([sys.executable, "-c", f"import {module}"], capture_output=True, text=True, env=env)


def test_the_two_native_ecs_domains_exist_and_the_one_ecs_tree_does_not():
    assert (REPO / "native/ECS_G").is_dir() and (REPO / "native/ECS_C").is_dir()
    assert not (REPO / "native/ECS").exists()
    assert (REPO / "native/runtime/composite/src/ecs_composite.c").is_file()
    ids = [s["id"] for s in SYSTEM["subsystems"]]
    assert "ECS_G" in ids and "ECS_C" in ids and "ECS" not in ids and "continuity" in ids
    subs = {s["id"]: s for s in SYSTEM["subsystems"]}
    assert set(subs["ECS_G"]["native_targets"]) == {"elpis_ecsg_math", "elpis_ecsg_fms", "elpis_ecsg_k1",
                                                    "elpis_ecsg_k1_fms"}
    assert subs["ECS_C"]["native_targets"] == ["elpis_ecs_memory", "elpis_ecs_c_authority", "elpis_ecs_c_hca"]
    assert "elpis_ecs_composite" in subs["runtime"]["native_targets"]
    # Python bindings keep their historical package; no Python ECS_G / ECS_C package or kernel exists.
    assert _clean_import("elpis.ECS").returncode == 0 and _clean_import("elpis.continuity").returncode == 0
    for absent in ("elpis.ECS_G", "elpis.ECS_C"):
        result = _clean_import(absent)
        assert result.returncode != 0 and "ModuleNotFoundError" in result.stderr, absent
    for path in ("src/elpis/ECS_C", "src/elpis/ECS_G", "tests/ECS_C", "tests/ECS_G"):
        assert not (REPO / path).exists(), path


def _imports(package: str) -> set[str]:
    names = set()
    for path in (REPO / "src" / "elpis" / package).rglob("*.py"):
        names |= imports_of(REPO, path)
    return names


def _under(name: str, prefix: str) -> bool:
    return name == prefix or name.startswith(prefix + ".")


def _cmake_links(text: str) -> set[str]:
    """Every elpis_* target named in a CMake file's target_link_libraries / LIBS lists."""
    names = set()
    for block in re.findall(r"target_link_libraries\(([^)]*)\)", text) + re.findall(r"\bLIBS\b([^)]*)", text):
        names |= set(re.findall(r"\belpis_\w+", block))
    return names


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
    for foreign in ("elpis_ecsg", "ecsg_", "elpis_ecs_mem", "extern crate", "#[link(", "dsv", "inference"):
        assert foreign not in rust, foreign
    # RuntimeCore embeds the continuity crate and, since ABI v6, the ECS_C authority crate without its C ABI (the
    # sole composer of the durable authority), and nothing else; it links no ECS code (K1 and the composite are reached
    # only through caller-supplied function tables) and knows nothing of inference or DSV.
    runtime_crate = REPO / "native/runtime"
    cargo = (runtime_crate / "Cargo.toml").read_text()
    deps = cargo[cargo.index("[dependencies]"):].split("\n[", 1)[0]
    assert re.findall(r"(?m)^(\w+)\s*=", deps) == ["elpis_continuity", "elpis_ecs_c_authority"], deps
    assert 'elpis_ecs_c_authority = { path = "../ECS_C/authority", default-features = false }' in deps
    assert "[build-dependencies]" not in cargo and "[dev-dependencies]" not in cargo
    lock = (runtime_crate / "Cargo.lock").read_text()
    assert sorted(re.findall(r'(?m)^name = "([^"]+)"', lock)) == ["elpis_continuity", "elpis_ecs_c_authority",
                                                                  "elpis_runtime"]
    rust = "\n".join(line.split("//")[0] for p in (runtime_crate / "src").rglob("*.rs")
                     for line in p.read_text().splitlines())   # code only: doc comments cite the K1 ABI it mirrors
    for foreign in ("elpis_ecsg", "extern crate", "#[link(", "dlopen", "dsv", "inference", "ecs_c_authority::ffi"):
        assert foreign not in rust, foreign
    # The ECS_C authority (docs/ECS_C_AUTHORITY.md) embeds the continuity crate (SHA-256, the evolution digest, the
    # legacy register reader) and nothing else until continuity folds into ECS_C (census, section 9): no ECS_G, no
    # memory, no RuntimeCore, no codec. Its build script only keeps the embedded crate's C ABI out of its exports.
    authority = REPO / "native/ECS_C/authority"
    cargo = (authority / "Cargo.toml").read_text()
    deps = cargo[cargo.index("[dependencies]"):].split("\n[", 1)[0]
    assert re.findall(r"(?m)^(\w+)\s*=", deps) == ["elpis_continuity"], deps
    assert "[build-dependencies]" not in cargo and "[dev-dependencies]" not in cargo
    lock = (authority / "Cargo.lock").read_text()
    assert sorted(re.findall(r'(?m)^name = "([^"]+)"', lock)) == ["elpis_continuity", "elpis_ecs_c_authority"]
    rust = "\n".join(line.split("//")[0] for p in (authority / "src").rglob("*.rs") for line in p.read_text().splitlines())
    for foreign in ("elpis_ecsg", "ecsg_", "elpis_ecs_mem", "elpis_runtime", "extern crate", "#[link(", "dlopen",
                    "dsv", "inference"):
        assert foreign not in rust, foreign
    build = "\n".join(line.split("//")[0] for line in (authority / "build.rs").read_text().splitlines())
    assert re.findall(r'println!\("([^"]*)"\)', build) == ["cargo:rustc-cdylib-link-arg=-Wl,--exclude-libs,ALL"]
    # Only for its own C ABI library: Cargo passes a dependency's cdylib link arguments on to the dependent's cdylib,
    # where it would hide RuntimeCore's re-export of continuity's C ABI (test_abi_inventory.py checks the binaries).
    assert 'if std::env::var_os("CARGO_FEATURE_C_ABI").is_some() {' in build
    inference = _imports("inference")
    assert not [n for n in inference if _under(n, "elpis.ECS") or _under(n, "elpis.continuity")]
    cognition = imports_of(REPO, REPO / "src/elpis/runtime/cognition.py")
    assert any(_under(n, "elpis.ECS") for n in cognition)
    composition = imports_of(REPO, REPO / "src/elpis/runtime/composition.py")
    assert any(_under(n, "elpis.continuity") for n in composition)
    subs = {s["id"]: s for s in SYSTEM["subsystems"]}
    assert {"ECS_G", "ECS_C", "continuity"} <= set(subs["runtime"]["depends_on"])
    # ECS_C depends on continuity only while its authority embeds the continuity crate (until Stage F).
    assert subs["ECS_G"]["depends_on"] == ["substrate"]
    assert subs["ECS_C"]["depends_on"] == ["substrate", "continuity"]
    for ecs in ("ECS_G", "ECS_C"):
        assert ecs not in subs["continuity"]["depends_on"]
    # The build graph: each domain links only itself and the substrate; the composite (both) is RuntimeCore's.
    geometric = _cmake_links((REPO / "native/ECS_G/CMakeLists.txt").read_text())
    memory = set().union(*(_cmake_links(p.read_text()) for p in (REPO / "native/ECS_C").rglob("CMakeLists.txt")))
    assert not [n for n in geometric if n.startswith(("elpis_ecs_", "elpis_runtime", "elpis_continuity"))], geometric
    assert not [n for n in memory if n.startswith(("elpis_ecsg_", "elpis_runtime", "elpis_continuity"))], memory
    composite = (REPO / "native/runtime/composite/CMakeLists.txt").read_text()
    assert "elpis_add_library(runtime elpis_ecs_composite" in composite
    for domain in ("ECS_G", "ECS_C"):
        text = "\n".join(p.read_text() for p in (REPO / f"native/{domain}").rglob("CMakeLists.txt"))
        labels = set(re.findall(r"elpis_add_(?:library|test|cargo_target)\((\w+)\s", text))
        labels |= set(re.findall(r"LABELS (\w+)", text))
        assert labels == {domain}, (domain, labels)   # every target of the tree belongs to its domain
    # The source include graph (the mission gate's check 4): no include across the two domains.
    assert ecs_independence(REPO) == []


def _write(root: Path, name: str, text: str) -> None:
    path = root / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def test_the_independence_check_rejects_an_include_across_the_two_domains(tmp_path):
    _write(tmp_path, "native/ECS_G/src/ecsg_math.c", '#include "elpis/ecsg_math.h"\n')
    _write(tmp_path, "native/ECS_C/memory/src/ecs_memory.c", '#include "elpis/ecs_memory.h"\n#include "elpis/sha256.h"\n')
    assert ecs_independence(tmp_path) == []
    _write(tmp_path, "native/ECS_C/memory/src/ecs_memory.c", '#include "elpis/ecs_memory.h"\n#include "elpis/ecsg_k1.h"\n')
    _write(tmp_path, "native/ECS_G/src/ecsg_k1.c", '#include "elpis/ecs_memory.h"\n')
    _write(tmp_path, "native/ECS_G/src/ecsg_state.c", '#include "elpis/runtime.h"\n')
    found = ecs_independence(tmp_path)
    assert {"native/ECS_C/memory/src/ecs_memory.c: includes elpis/ecsg_k1.h",
            "native/ECS_G/src/ecsg_k1.c: includes elpis/ecs_memory.h",
            "native/ECS_G/src/ecsg_state.c: includes elpis/runtime.h"} <= set(found), found


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
    # RuntimeCore's HOT operations (Rust): the live lineage and the native transaction, nothing else. Each HOT
    # function body is checked on its own: no history machinery, and (ABI v4, docs/RUNTIME_CORE.md "Live and
    # durable authority") no store, no durability lane, no checkpoint, no publication and no sync. A managed commit
    # is volatile; durable publication is the separate, explicit durable_request.
    core = (REPO / "native/runtime/src/core.rs").read_text()
    durable_machinery = re.compile(r"\b(with_stores|Stores|lane|checkpoint|publish\w*|sync\w*|snapshot_write|"
                                   r"continuity|commit_cognition\w*|digest|leased_identity)\b")
    for name in _HOT_FUNCTIONS:
        body = _rust_fn(core, name)
        code = "\n".join(line.split("//")[0] for line in body.splitlines())
        assert not FORBIDDEN_HOT_PATH.findall(code), (name, FORBIDDEN_HOT_PATH.findall(code))
        assert not durable_machinery.findall(code), (name, durable_machinery.findall(code))


# The HOT functions of RuntimeCore: QUERY, the managed turn and OBSERVE, with their helpers. Binding (`reconcile`)
# reads the durable identity once per binding and is checked separately by the runtime laws.
_HOT_FUNCTIONS = ("query", "query_inner", "turn_begin", "begin", "schedule_once", "run_schedule", "take_turn",
                  "turn_commit", "turn_observe", "turn_abort", "set_live")


def _rust_fn(source: str, name: str) -> str:
    """The body of `fn name` in `source` (brace-matched; comments and strings in these bodies hold no braces)."""
    m = re.search(r"\bfn " + re.escape(name) + r"\b[^{]*\{", source)
    assert m, name
    depth, i = 1, m.end()
    while depth:
        depth += {"{": 1, "}": -1}.get(source[i], 0)
        i += 1
    return source[m.end():i - 1]


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
        "commit_cognition", "reserve_evolution", "finalize_evolution",
        # The state-format-tagged lineage (record v3, docs/ECS_COMPOSITE_STATE.md): still one identity digest, now
        # with the format it names, and the explicit legacy K1 -> composite import. No state, no history.
        "anchor_cognition_as", "commit_cognition_as", "import_cognition")}, exported
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


# -- the retired donor ----------------------------------------------------------------------------------------------

# The deleted ECS_C event-history kernel (CONT-5, c96e64d) and the receipt history (CONT-3): its native ABI, its Python
# kernel and its runtime receipt modules. Not the target architecture (docs/NATIVE_SPLIT_CENSUS.md, sections 5a and 6):
# they appear only in enumerated immutable records, never in canonical code or live documentation.
RETIRED_DONOR = re.compile(
    r"\b(?:elpis_ecsc_\w+|ELPIS_ECSC_\w+|ecsc_(?:history|log|session)\w*|elpis\.ECS_C\b|elpis\.ecs\.kernel|"
    r"ReceiptHistory|NativeHistorySession|native_history|runtime\.history)")

# Immutable records that name the donor (frozen source text).
FROZEN_EVIDENCE = {
    "research/ecs_dynamics/": "Closed ECS dynamics laboratory (frozen, no longer run): its source imports the "
                              "pre-rename elpis.ecs kernel it measured.",
}
# Live files that name the retired donor in order to inventory or forbid it.
TRANSITION_RECORDS = {
    "docs/NATIVE_SPLIT_CENSUS.md": "Stage A census of the native split: inventories the deleted ECS_C donor by name.",
    "docs/CONTINUITY.md": "The continuity consumer census that retired the receipt history names what it deleted.",
    "tests/integration/test_runtime_hot_path.py": "Asserts the retired receipt-history modules stay absent.",
    "tests/boundary/test_ecs_split.py": "This gate names the retired donor identifiers it rejects.",
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


def test_the_retired_donor_appears_only_in_enumerated_records():
    violations = []
    for path in _tracked():
        if _frozen(path) or path in TRANSITION_RECORDS:
            continue
        text = _text(path)
        if text is None:
            continue
        hit = RETIRED_DONOR.search(text)
        if hit:
            violations.append((path, text[max(0, hit.start() - 16):hit.end() + 24]))
    assert not violations, violations


def test_every_exemption_is_real_and_reasoned():
    tracked = _tracked()
    for key, why in FROZEN_EVIDENCE.items():
        assert why.strip(), key
        members = [p for p in tracked if p == key or (key.endswith("/") and p.startswith(key))]
        assert any(RETIRED_DONOR.search(_text(p) or "") for p in members), f"stale exemption: {key}"
    for path, why in TRANSITION_RECORDS.items():
        assert why.strip() and path in tracked and RETIRED_DONOR.search(_text(path) or ""), f"stale exemption: {path}"


# Native sources whose exact bytes are recorded by research/ecs_runtime_r1/evidence (measured and sanitized
# sources), under these very paths: the split moved them back to the paths the evidence recorded.
BYTE_BOUND = (
    "native/ECS_G/include/elpis/ecsg_executor.h", "native/ECS_G/include/elpis/ecsg_math.h",
    "native/ECS_G/include/elpis/ecsg_state.h", "native/ECS_G/src/ecsg_executor.c", "native/ECS_G/src/ecsg_math.c",
    "native/ECS_G/src/ecsg_state.c", "native/ECS_G/tests/test_ecsg_executor.c",
    "native/ECS_G/tests/test_ecsg_executor_alloc.c", "native/ECS_G/tests/test_ecsg_executor_txn.c",
    "native/ECS_G/tests/test_ecsg_math.c", "native/ECS_G/tests/test_ecsg_snapshot.c",
    "native/ECS_G/tests/test_ecsg_state.c",
)


def test_the_runtime_r1_measured_sources_are_the_recorded_bytes_at_their_recorded_paths():
    evidence = "".join(p.read_text() for p in (REPO / "research/ecs_runtime_r1/evidence").glob("*.json"))
    for path in BYTE_BOUND:
        digest = hashlib.sha256((REPO / path).read_bytes()).hexdigest()
        assert digest in evidence, f"{path} is no longer the recorded bytes"
        assert f'"{path}"' in evidence, f"{path} is not a path the evidence recorded"
