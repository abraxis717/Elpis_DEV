"""Research laboratories. RESEARCH_ONLY. NO_RUNTIME_AUTHORITY.

Nothing here is packaged, installed or imported by ``src/elpis``; the boundary
suite enforces that (``research`` is a forbidden import for production code).

Frozen laboratory import names
------------------------------
The closed laboratories (``ecs_cognition_r0``, ``ecs_retention_r0..r3``,
``ecs_runtime_r1``, ``ecs_k1_native``) are byte-frozen scientific objects:
their source digests are part of their recorded evidence, so their text can
never change. They import the ECS under the module name it had when they were
frozen, ``elpis.ECS_G``. The one canonical ECS is now ``elpis.ECS``.

Importing ``research`` installs one lazy import-name mapping, and only for
these frozen laboratories: ``elpis.ECS_G[.<module>]`` resolves to the *same*
module object as ``elpis.ECS[.<module>]``. It is a name, not a second
implementation, and it does not exist in the ``elpis`` distribution:
``import elpis.ECS_G`` fails in any process that has not imported
``research`` (tests/boundary/test_one_ecs.py).
"""
from __future__ import annotations

import importlib
import importlib.abc
import importlib.util
import sys

_FROZEN_NAME = "elpis.ECS_G"
_CANONICAL_NAME = "elpis.ECS"


class _FrozenLaboratoryImportName(importlib.abc.MetaPathFinder, importlib.abc.Loader):
    def find_spec(self, fullname, path=None, target=None):
        if fullname == _FROZEN_NAME or fullname.startswith(_FROZEN_NAME + "."):
            return importlib.util.spec_from_loader(fullname, self)
        return None

    def create_module(self, spec):
        return importlib.import_module(_CANONICAL_NAME + spec.name[len(_FROZEN_NAME):])

    def exec_module(self, module):
        """The canonical module is already executed; nothing is re-run."""


if not any(isinstance(finder, _FrozenLaboratoryImportName) for finder in sys.meta_path):
    sys.meta_path.insert(0, _FrozenLaboratoryImportName())
