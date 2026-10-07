"""Authority gates for mutable ECS residency. This is mechanics, not a new cognitive law."""
import ast
import re

from . import _mission as M
from ._system import REPO


def test_fms_rejects_ecs_semantic_logic(tmp_path):
    p = tmp_path / "native/substrate/src/fms_core.c"
    p.parent.mkdir(parents=True)
    for source in ("void elpis_ecsg_learn(void);", "int retention;", "int g1;", "int s3;"):
        p.write_text(source)
        assert M.fms_genericity(tmp_path)


def test_only_the_residency_adapter_can_include_generic_fms(tmp_path):
    p = tmp_path / "native/ECS/src/ecsg_math.c"
    p.parent.mkdir(parents=True)
    p.write_text('#include "elpis/fms.h"\n')
    assert M.ecs_independence(tmp_path)
    p.write_text('#include "elpis/dsv41.h"\n')
    assert M.ecs_independence(tmp_path)


def test_adapter_uses_only_public_executor_and_fms_abis():
    source = (REPO / "native/ECS/src/ecsg_fms.c").read_text()
    assert "elpis_ecsg_executor_restore(" in source
    assert "elpis_ecsg_executor_snapshot_write(" in source
    assert "fms_lease_acquire(" in source and "fms_lease_release(" in source
    assert "fms_register(" in source and "fms_unregister(" in source
    for forbidden in ("ecsg_executor_internal", "make_borrowed", "bind_borrowed", "PyObject", "PyGILState", "DSV"):
        assert forbidden not in source
    for forbidden_call in ("fopen(", "pwrite(", "pread("):
        assert not re.search(r"(?<![\\w])" + re.escape(forbidden_call), source)


def test_python_control_plane_is_additive_and_does_not_reimplement_the_hot_path():
    path = REPO / "src/elpis/ECS/residency.py"
    tree = ast.parse(path.read_text())
    imports = M.imports_of(REPO, path)
    assert not [n for n in imports if n.startswith("numpy") or n.startswith("research")]
    # FMSRuntime creates an actual existing Executor instance with a native ABI proxy;
    # it does not contain query/learn/step numerical implementations.
    text = path.read_text()
    assert "Executor.__new__(Executor)" in text
    for name in ("def forward(", "def learn(", "def step("):
        assert name not in text


def test_runtime_r1_measured_sources_are_not_replaced_by_the_adapter():
    adapter = (REPO / "native/ECS/src/ecsg_fms.c").read_text()
    header = (REPO / "native/ECS/include/elpis/ecsg_fms.h").read_text()
    assert '#include "elpis/ecsg_fms.h"' in adapter
    assert '#include "elpis/ecsg_executor.h"' in header
    assert '#include "elpis/fms.h"' in header
    assert '#include "elpis/ecsg_executor_internal.h"' not in adapter + header
    # The historical evidence suite independently pins exact bytes of the measured files (under their recorded
    # paths, resolved to the current tree through the one stated rename).
    evidence = (REPO / "tests/research/ecs_runtime_r1/test_evidence.py").read_text()
    assert '/src/ecsg_executor.c"' in evidence and '/native.py"' in evidence
    assert "measured_file_is_current" in evidence
