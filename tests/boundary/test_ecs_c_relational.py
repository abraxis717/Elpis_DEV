"""The ECS_C relational view stays DEV: consumer-free, I/O-free, HCA-free and one declared ABI
(docs/ECS_C_RELATIONAL.md).

* Not composed: no library, executable or Python module outside its own directory and its tests uses it. Not
  RuntimeCore, not the composite, not the memory, not the deployment's admitted library set, not src/.
* Isolated from HCA research: the view neither names nor links the HCA substrate, and no function of its ABI takes a
  salience, a candidate or an HCA object. The research interface (research/ecsc_dynamic_relational_r0) reads the
  view's export one way.
* Nothing persists and nothing runs on its own: the source has no file, clock, environment, randomness or thread
  call, and one allocation per view.
* One ABI: the header's functions are the source's exported functions and the ABI inventory's entry; the library
  links only the memory.
"""
from __future__ import annotations

import json
import re

from ._system import REPO

REL = REPO / "native/ECS_C/relational"
HEADER = (REL / "include/elpis/ecs_c_relational.h").read_text(encoding="utf-8")
SOURCE = (REL / "src/ecs_c_relational.c").read_text(encoding="utf-8")


def _code(text: str) -> str:
    text = re.sub(r"/\*.*?\*/", "", text, flags=re.S)
    return "\n".join(line.split("//")[0] for line in text.splitlines())


def test_nothing_canonical_uses_the_relational_view():
    names = ("ecs_c_relational", "elpis_ecs_c_rel_")
    offenders = []
    for path in sorted((REPO / "native").rglob("*")):
        if not path.is_file() or path.suffix not in (".c", ".h", ".rs", ".txt", ".cmake", ".toml", ".cpp", ".hpp"):
            continue
        rel = path.relative_to(REPO).as_posix()
        if rel.startswith("native/ECS_C/relational/") or rel == "native/ECS_C/CMakeLists.txt":
            continue
        if any(n in path.read_text(encoding="utf-8", errors="replace") for n in names):
            offenders.append(rel)
    for path in sorted((REPO / "src").rglob("*.py")):
        if any(n in path.read_text(encoding="utf-8") for n in names):
            offenders.append(path.relative_to(REPO).as_posix())
    assert not offenders, offenders
    # In its own CMake file: one library and its tests, and nothing else links it.
    cmake = (REPO / "native/ECS_C/CMakeLists.txt").read_text(encoding="utf-8")
    linked = [line.strip() for line in cmake.splitlines() if "elpis_ecs_c_relational" in line and "LIBS" in line]
    assert linked == ['LIBS elpis_ecs_c_relational "${CMAKE_CURRENT_BINARY_DIR}/authority/libelpis_ecs_c_authority.so"'], linked
    assert "elpis_ecs_c_relational" not in (REPO / "src/elpis/runtime/deployment.py").read_text(encoding="utf-8")


def test_it_is_isolated_from_hca_research():
    for text in (_code(HEADER), _code(SOURCE)):
        assert "hca" not in text.lower(), "the view names no HCA object"
        for word in ("salience", "SALIENCE", "candidate_relation", "activation"):
            assert word not in text, word
    # Every input of the ABI is admitted evidence, an authority table or a coordinate: no feedback channel.
    declared = re.findall(r"\b(elpis_ecs_c_rel_\w+)\s*\(([^;]*)\);", _code(HEADER), flags=re.S)
    for name, params in declared:
        assert "hca" not in params.lower() and "score" not in params.lower(), name
    cmake = (REPO / "native/ECS_C/CMakeLists.txt").read_text(encoding="utf-8")
    block = cmake[cmake.index("elpis_add_library(ECS_C elpis_ecs_c_relational"):]
    assert "elpis_ecs_c_hca" not in block


def test_it_performs_no_io_and_starts_nothing():
    code = _code(SOURCE)
    for call in ("fopen", "open", "openat", "read", "write", "pwrite", "pread", "mmap", "fsync", "fdatasync",
                 "rename", "unlink", "mkdir", "fork", "getenv", "time", "clock_gettime", "rand", "random",
                 "printf", "fprintf", "puts"):
        assert not re.search(r"(?<![\w.>])" + call + r"\s*\(", code), call
    for word in ("pthread", "exec", "stdio.h", "unistd.h", "fcntl.h", "thread"):
        assert word not in code, word
    # One allocation per view (test_ecs_c_relational_alloc proves none per call).
    assert code.count("calloc(") == 1 and "malloc(" not in code and "realloc(" not in code
    # The test-only damage hook is compiled into the tests only.
    hook = code.index("int ecs_c_rel_testing_damage(")
    assert code.rfind("#ifdef ELPIS_ECS_C_REL_TESTING", 0, hook) > code.rfind("#endif", 0, hook)


def test_the_header_the_source_and_the_inventory_declare_one_abi():
    declared = set(re.findall(r"\b(elpis_ecs_c_rel_\w+)\s*\(", _code(HEADER)))
    defined = set(re.findall(r"(?m)^(?:[\w ]+\*?\s*)\b(elpis_ecs_c_rel_\w+)\s*\(", _code(SOURCE)))
    assert declared == defined, sorted(declared ^ defined)
    inventory = json.loads((REPO / "native/ABI_INVENTORY.json").read_text(encoding="utf-8"))["libraries"]
    entry = inventory["libelpis_ecs_c_relational.so"]
    assert entry["soname"] == "libelpis_ecs_c_relational.so"
    assert entry["elpis_needed"] == ["libelpis_ecs_memory.so"]
    assert entry["exports"] == sorted(declared)


def test_it_reads_the_authority_layout_but_links_no_authority_code():
    # The authority header supplies elpis_ecs_c_entity (a layout); no authority function is called.
    code = _code(SOURCE)
    authority_calls = set(re.findall(r"\b(elpis_ecs_c_(?!rel_)\w+)\s*\(", code))
    assert not authority_calls, authority_calls
    memory_calls = set(re.findall(r"\b(elpis_ecs_mem_\w+)\s*\(", code))
    assert memory_calls == {"elpis_ecs_mem_relation_key"}, memory_calls
