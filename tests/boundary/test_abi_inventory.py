"""The native ABI inventory (native/ABI_INVENTORY.json) is what the build actually exports.

Every listed library is built exactly once under its file name (cargo and test-copy directories excluded), carries
its recorded SONAME, exports exactly its recorded ``elpis_*`` / ``ELPIS_*`` dynamic symbols and links exactly its
recorded Elpis libraries (``DT_NEEDED``): the ownership graph is checked on the binaries, not only in the sources. A
source relocation (docs/NATIVE_SPLIT_CENSUS.md) must leave this file unchanged; a change of exports or of the Elpis
link graph is a versioned change recorded here in the same commit.
"""
from __future__ import annotations

import json
import shutil
import subprocess

import pytest

from ..conftest import native_build_dir, native_required
from ._system import REPO

INVENTORY = json.loads((REPO / "native" / "ABI_INVENTORY.json").read_text(encoding="utf-8"))


def _built(name: str):
    build = native_build_dir()
    found = sorted(p for p in build.rglob(name) if p.is_file() and not any(
        part.startswith("cargo") or part == "testing-lib" for part in p.relative_to(build).parts))
    if not found:
        message = f"{name} not built under {build}"
        if native_required():
            pytest.fail(message)
        pytest.skip(message + " (never an implicit PASS)")
    return found


def _tool(name: str) -> str:
    path = shutil.which(name)
    if path is None:
        pytest.skip(f"{name} is not available")
    return path


def test_the_inventory_is_well_formed():
    assert INVENTORY["schema"] == "elpis.native-abi-inventory.v1"
    for name, entry in INVENTORY["libraries"].items():
        assert name.startswith("lib") and name.endswith(".so"), name
        assert entry["exports"] == sorted(set(entry["exports"])) and entry["exports"], name
        assert entry["soname"].startswith("lib") and entry["soname"].endswith(".so"), name
        assert entry["elpis_needed"] == sorted(entry["elpis_needed"]), name
        assert set(entry["elpis_needed"]) <= {e["soname"] for e in INVENTORY["libraries"].values()}, name


@pytest.mark.parametrize("name", sorted(INVENTORY["libraries"]))
def test_every_library_exports_exactly_its_recorded_symbols(name):
    entry = INVENTORY["libraries"][name]
    found = _built(name)
    assert len(found) == 1, found            # one build product per library name
    nm, readelf = _tool("nm"), _tool("readelf")
    dynamic = subprocess.run([readelf, "-d", str(found[0])], capture_output=True, text=True, check=True).stdout
    soname = [line.split("[", 1)[1].rstrip("]") for line in dynamic.splitlines() if "(SONAME)" in line]
    assert soname == [entry["soname"]], (name, soname)
    needed = sorted(line.split("[", 1)[1].rstrip("]") for line in dynamic.splitlines() if "(NEEDED)" in line)
    assert [n for n in needed if n.startswith("libelpis_")] == entry["elpis_needed"], (name, needed)
    table = subprocess.run([nm, "-D", "--defined-only", str(found[0])], capture_output=True, text=True,
                           check=True).stdout
    exports = sorted({line.split()[-1] for line in table.splitlines()
                      if len(line.split()) >= 3 and line.split()[1] in "TDRBV"
                      and line.split()[-1].startswith(("elpis_", "ELPIS_"))})
    assert exports == entry["exports"], (name, sorted(set(exports) ^ set(entry["exports"])))
