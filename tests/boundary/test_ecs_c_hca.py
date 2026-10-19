"""The ECS_C HCA native research substrate stays research: gated, consumer-free, I/O-free and one declared ABI
(docs/ECS_C_HCA.md).

* Gated: no library, executable or Python module outside its own directory and its tests uses it. Not RuntimeCore,
  not the composite, not the memory, not the deployment's admitted library set, not src/. It enables no law in the
  canonical runtime.
* Nothing persists and nothing runs on its own: the source has no file, clock, environment, randomness or thread
  call.
* One ABI: the header's functions are the source's exported functions and the ABI inventory's entry; the
  library links no other Elpis library. The B2 schedule domain is the reference's byte for byte (the differential
  depends on it; tests/research/ecsc_hca_native).
"""
from __future__ import annotations

import json
import re

from ._system import REPO

HCA = REPO / "native/ECS_C/hca"
HEADER = (HCA / "include/elpis/ecs_c_hca.h").read_text(encoding="utf-8")
SOURCE = (HCA / "src/ecs_c_hca.c").read_text(encoding="utf-8")


def _code(text: str) -> str:
    text = re.sub(r"/\*.*?\*/", "", text, flags=re.S)
    return "\n".join(line.split("//")[0] for line in text.splitlines())


def test_nothing_canonical_uses_the_research_substrate():
    names = ("ecs_c_hca", "elpis_hca_", "ecsc_hca")
    offenders = []
    for path in sorted((REPO / "native").rglob("*")):
        if not path.is_file() or path.suffix not in (".c", ".h", ".rs", ".txt", ".cmake", ".toml"):
            continue
        rel = path.relative_to(REPO).as_posix()
        if rel.startswith("native/ECS_C/hca/") or rel == "native/ECS_C/CMakeLists.txt":
            continue
        if any(n in path.read_text(encoding="utf-8", errors="replace") for n in names):
            offenders.append(rel)
    for path in sorted((REPO / "src").rglob("*.py")):
        if any(n in path.read_text(encoding="utf-8") for n in names):
            offenders.append(path.relative_to(REPO).as_posix())
    assert not offenders, offenders
    # In its own CMake file: one library and its tests, and nothing else links it.
    cmake = (REPO / "native/ECS_C/CMakeLists.txt").read_text(encoding="utf-8")
    linked = [line for line in cmake.splitlines() if "elpis_ecs_c_hca" in line and "LIBS" in line]
    assert linked == ["elpis_add_test(ECS_C test_ecs_c_hca SOURCES hca/tests/test_ecs_c_hca.c LIBS elpis_ecs_c_hca "
                      "elpis_hash TIMEOUT 120)"], linked
    assert "elpis_ecs_c_hca" not in (REPO / "src/elpis/runtime/deployment.py").read_text(encoding="utf-8")


def test_it_performs_no_io_and_starts_nothing():
    code = _code(SOURCE)
    for call in ("fopen", "open(", "openat", "read(", "write(", "pwrite", "pread", "mmap", "fsync", "fdatasync",
                 "rename", "unlink", "mkdir", "pthread", "fork", "exec", "getenv", "time(", "clock_gettime",
                 "rand(", "random(", "printf", "stdio.h", "unistd.h", "fcntl.h"):
        assert call not in code, call
    # One allocation per graph and per allocated session; nothing else allocates (test_ecs_c_hca_alloc proves it).
    assert code.count("calloc(") == 2 and "malloc(" not in code and "realloc(" not in code


def test_the_header_the_source_and_the_inventory_declare_one_abi():
    declared = set(re.findall(r"\b(elpis_hca_\w+)\s*\(", _code(HEADER)))
    defined = set(re.findall(r"(?m)^(?:[\w ]+\*?\s*)\b(elpis_hca_\w+)\s*\(", _code(SOURCE)))
    assert declared == defined, sorted(declared ^ defined)
    inventory = json.loads((REPO / "native/ABI_INVENTORY.json").read_text(encoding="utf-8"))["libraries"]
    entry = inventory["libelpis_ecs_c_hca.so"]
    assert entry["soname"] == "libelpis_ecs_c_hca.so" and entry["elpis_needed"] == []
    assert sorted(e for e in entry["exports"] if e.startswith("elpis_hca_")) == sorted(declared)
    # Beyond its own ABI only the substrate's SHA-256 helpers, as libelpis_ecs_memory.so carries them.
    assert set(entry["exports"]) - declared == set(inventory["libelpis_ecs_memory.so"]["exports"]) - {
        e for e in inventory["libelpis_ecs_memory.so"]["exports"] if e.startswith("elpis_ecs_mem_")}


def test_the_schedule_domain_is_the_references():
    reference = (REPO / "research/ecsc_hca_r0/b2.py").read_text(encoding="utf-8")
    domain = re.search(r'SCHEDULE_DOMAIN = b"([^"]*)\\0"', reference).group(1)
    assert f'static const uint8_t SCHEDULE_DOMAIN[{len(domain) + 1}] = "{domain}";' in SOURCE
