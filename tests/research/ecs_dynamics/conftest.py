"""The laboratory needs NumPy (the ``test`` extra); without it these tests are not collected."""
import importlib.util

if importlib.util.find_spec("numpy") is None:
    collect_ignore_glob = ["test_*.py"]

# ECS_C_HISTORICAL_IMPORT_COMPAT_R0
#
# The committed research/ecs_dynamics laboratory is a frozen scientific
# object whose source digest must remain byte-identical. Its historical
# public-kernel experiment imports elpis.ecs.*, which was the package name
# when the laboratory was frozen.
#
# Production now names that same continuity/history kernel elpis.ECS_C.
# This test-only alias permits replay/qualification of the frozen laboratory
# without recreating a production elpis.ecs package or changing frozen source.
import importlib
import sys

_ecs_c = importlib.import_module("elpis.ECS_C")
sys.modules.setdefault("elpis.ecs", _ecs_c)

for _name in ("kernel", "persistence", "topology_analysis"):
    sys.modules.setdefault(
        "elpis.ecs." + _name,
        importlib.import_module("elpis.ECS_C." + _name),
    )

del _name, _ecs_c
