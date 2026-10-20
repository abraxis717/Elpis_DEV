"""The ECS_C authority's ABI and persistence bounds are one declaration, checked in the sources (docs/ECS_C_AUTHORITY.md).

* The C header, the Rust FFI and the ABI inventory name the same functions: the production library exports the
  header's functions except the two testing hooks, the testing library all of them, and neither exports the retired
  standalone continuity's ``elpis_continuity_*`` ABI (the binaries are checked by test_abi_inventory.py).
* The persistence surface is the fixed file set of the census's persistence invariant (section 5a): the crate
  names exactly five files, creates files only while provisioning and never renames or removes one afterwards.
* The record and witness sizes in the header, the Rust layout and the docs agree.
"""
from __future__ import annotations

import json
import re

from ._system import REPO

CRATE = REPO / "native/ECS_C/authority"
HEADER = (CRATE / "include/elpis/ecs_c_authority.h").read_text(encoding="utf-8")
TESTING = {"elpis_ecs_c_testing_arm", "elpis_ecs_c_testing_counters"}


def _rust(name: str) -> str:
    return (CRATE / "src" / name).read_text(encoding="utf-8")


def _code(text: str) -> str:
    return "\n".join(line.split("//")[0] for line in text.splitlines())


def test_the_header_the_ffi_and_the_inventory_declare_one_abi():
    declared = set(re.findall(r"\b(elpis_ecs_c_\w+)\s*\(", HEADER))
    ffi = _rust("ffi.rs")
    defined = set(re.findall(r'#\[no_mangle\]\s*pub (?:unsafe )?extern "C" fn (\w+)', ffi))
    assert defined == declared, sorted(defined ^ declared)
    gated = set(re.findall(r'#\[cfg\(feature = "testing"\)\]\s*#\[no_mangle\]\s*pub (?:unsafe )?extern "C" fn (\w+)', ffi))
    assert gated == TESTING
    # The C ABI is the default `c-abi` feature: an embedding crate (RuntimeCore) builds without it.
    assert '#[cfg(feature = "c-abi")]\npub mod ffi;' in _rust("lib.rs")
    assert 'default = ["c-abi"]' in (CRATE / "Cargo.toml").read_text(encoding="utf-8")
    inventory = json.loads((REPO / "native/ABI_INVENTORY.json").read_text(encoding="utf-8"))["libraries"]
    production = inventory["libelpis_ecs_c_authority.so"]
    testing = inventory["libelpis_ecs_c_authority_testing.so"]
    assert production == {"soname": "libelpis_ecs_c_authority.so", "elpis_needed": [],
                          "exports": sorted(declared - TESTING)}
    assert testing == {"soname": "libelpis_ecs_c_authority_testing.so", "elpis_needed": [],
                       "exports": sorted(declared)}
    assert not [s for e in (production, testing) for s in e["exports"] if s.startswith("elpis_continuity_")]


def test_the_persistence_surface_is_the_fixed_file_set():
    store = _code(_rust("store.rs"))
    assert 'STATE_FILES: [&str; 2] = ["authority.a", "authority.b"]' in store
    assert 'KEY_FILE: &str = "deployment.key"' in store
    assert 'WITNESS_FILES: [&str; 2] = ["witness.a", "witness.b"]' in store
    # Every file creation, rename and removal goes through Io, and Io's are called only by provisioning.
    for primitive in ("create_new(", "fs::rename(", "fs::remove_file(", "create_dir"):
        assert store.count(primitive) == 1, primitive
    provisioning = store[store.index("pub(crate) fn provision_with("):]
    preparing = store[store.index("fn prepare_dir("):store.index("struct Recovered")]
    for call in ("io.create(", "io.rename(", "io.unlink("):
        outside = store.replace(provisioning, "").count(call)
        assert outside == 0, call
    assert "create_dir" in preparing
    # No crate source writes anything but the fixed files: no log, no segments, no generations, no compaction.
    for name in ("model.rs", "record.rs", "store.rs", "legacy.rs", "ffi.rs", "lib.rs", "error.rs", "hmac.rs"):
        code = _code(_rust(name))
        assert not re.search(r"\b(OpenOptions::new\(\)\.append|append\(true\)|create\(true\)|"
                             r"set_len|tempfile|mkstemp)\b", code), name
    legacy = _code(_rust("legacy.rs"))
    assert "write" not in legacy, "the continuity directory is read only"


def test_record_and_witness_sizes_agree_with_the_header_and_the_docs():
    record = _code(_rust("record.rs"))
    model = _code(_rust("model.rs"))
    assert "pub const WITNESS_BYTES: usize = 240;" in record
    assert "#define ELPIS_ECS_C_WITNESS_BYTES 240u" in HEADER
    for constant, value in (("STATE_FIXED", 512), ("ENTITY_BYTES", 160), ("EVENT_BYTES", 512), ("PAYLOAD_AT", 80)):
        assert f"pub const {constant}: usize = {value};" in model, constant
    assert "pub const HEADER_BYTES: usize = 256;" in record and "pub const TRAILER_BYTES: usize = 64;" in record
    assert "#define ELPIS_ECS_C_MAX_CAPACITY 4096u" in HEADER and "pub const MAX_CAPACITY: u32 = 4096;" in model
    doc = (REPO / "docs/ECS_C_AUTHORITY.md").read_text(encoding="utf-8")
    assert "R = 1344 + 160 × capacity" in doc and "240 bytes" in doc
