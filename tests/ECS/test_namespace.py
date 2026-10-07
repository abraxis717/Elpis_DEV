"""Importing the ECS package is inert: it loads no numpy and exposes no implicit kernel or step."""

import importlib
import sys


def test_ecs_namespace_import_is_inert():
    before = set(sys.modules)
    module = importlib.import_module("elpis.ECS")
    after = set(sys.modules)

    assert module.__name__ == "elpis.ECS"
    assert "numpy" not in after - before
    assert not hasattr(module, "Kernel")
    assert not hasattr(module, "step")
