"""The durable ECS_C authority is RETIRED (docs/ELPIS_MISSION.md, Amendment 3; docs/ECS_C_AUTHORITY.md).

It held two record slots, an authenticated rollback witness and a deployment key on persistent storage, provisioned
them, recovered from them and imported continuity's legacy register. The volatile-native migration removed it: no
source, no build target, no library in the ABI inventory or the system manifest, no Cargo dependency, no link, no
include, no load. The ECS_C transition law it consumed is owned by the volatile session crate (native/ECS_C/session),
which RuntimeCore ABI v8 holds instead, and that crate can reach no filesystem. Git history keeps the durable code.
"""
from __future__ import annotations

import json
import re

from ._system import REPO

SESSION = REPO / "native/ECS_C/session"
_TOKENS = ("elpis/ecs_c_authority.h", "libelpis_ecs_c_authority", "elpis_ecs_c_authority", "ECS_C/authority")
_SCANNED = (".c", ".h", ".cpp", ".rs", ".toml", ".cmake", ".py", ".sh")


def _code(text: str, suffix: str) -> str:
    """The text without comments (C/C++/Rust // and block comments, Python/CMake/shell # comments, and docstrings)."""
    if suffix in (".c", ".h", ".cpp", ".rs"):
        text = re.sub(r"/\*.*?\*/", "", text, flags=re.S)
        return "\n".join(line.split("//")[0] for line in text.splitlines())
    if suffix == ".py":
        text = re.sub(r'"""[\s\S]*?"""', "", text)
    return "\n".join(line.split("#")[0] for line in text.splitlines())


def test_the_durable_authority_has_no_source_target_or_library():
    assert not (REPO / "native/ECS_C/authority").exists()
    assert "add_subdirectory(authority)" not in (REPO / "native/ECS_C/CMakeLists.txt").read_text(encoding="utf-8")
    inventory = json.loads((REPO / "native/ABI_INVENTORY.json").read_text(encoding="utf-8"))["libraries"]
    assert not [name for name in inventory if "ecs_c_authority" in name]
    assert not [s for entry in inventory.values() for s in entry["exports"]
                if s.startswith(("elpis_ecs_c_open", "elpis_ecs_c_provision", "elpis_ecs_c_recover",
                                 "elpis_ecs_c_reconcile", "elpis_continuity_"))]
    system = json.loads((REPO / "ELPIS_SYSTEM.json").read_text(encoding="utf-8"))
    ecs_c = next(s for s in system["subsystems"] if s["id"] == "ECS_C")
    assert "elpis_ecs_c_authority" not in ecs_c["native_targets"] and ecs_c["persistence"].startswith("NONE")
    assert any(r["path"] == "native/ECS_C/authority" for r in system["post_migration_retirements"])


def test_nothing_builds_links_includes_or_loads_the_durable_authority():
    """Code, not prose: comments and docstrings may name the retired authority as history; no build file, source or
    script may reach it. Research instruments pinned to their recorded runs are history and are not scanned, and the
    guards that assert its absence (the census, the boundary tests) are not reaches."""
    offenders = []
    for root in ("native", "src", "containment", "tests"):
        base = REPO / root
        if not base.exists():
            continue
        for path in base.rglob("*"):
            if not path.is_file() or (path.suffix not in _SCANNED and path.name != "CMakeLists.txt"):
                continue
            rel = path.relative_to(REPO)
            # The guards name the retired authority to assert its absence: the census and the boundary tests.
            if "target" in rel.parts or rel.as_posix().startswith("tests/boundary/") or \
                    rel.as_posix() == "src/elpis/runtime/persistence.py":
                continue
            code = _code(path.read_text(encoding="utf-8", errors="replace"),
                         ".cmake" if path.name == "CMakeLists.txt" else path.suffix)
            if any(token in code for token in _TOKENS):
                offenders.append(str(path.relative_to(REPO)))
    assert not offenders, offenders
    for crate in (REPO / "native/runtime", SESSION):
        manifest = (crate / "Cargo.toml").read_text(encoding="utf-8")
        assert "elpis_ecs_c_authority" not in manifest
        assert "elpis_ecs_c_authority" not in (crate / "Cargo.lock").read_text(encoding="utf-8")


def test_the_session_owns_the_law_and_can_reach_no_filesystem():
    lib = (SESSION / "src/lib.rs").read_text(encoding="utf-8")
    assert "#![no_std]" in lib
    manifest = (SESSION / "Cargo.toml").read_text(encoding="utf-8")
    assert re.search(r"(?ms)^\[dependencies\]\s*\n(?:#[^\n]*\n|\s*\n)*(?:\[|\Z)", manifest), "no dependencies"
    for path in (SESSION / "src").glob("*.rs"):
        code = _code(path.read_text(encoding="utf-8"), ".rs")
        if path.name == "tests.rs":
            continue                                   # #[cfg(test)] only (extern crate std in tests)
        for forbidden in ("std::fs", "std::io", "std::net", "std::process", "std::env", "File::", "OpenOptions",
                          "libc::open", "fn open(", "pwrite", "fsync"):
            assert forbidden not in code, (path.name, forbidden)
    assert (SESSION / "src/model.rs").is_file() and (SESSION / "src/error.rs").is_file()
