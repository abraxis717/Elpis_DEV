"""Materialized ECS_G namespace carries no runtime authority yet."""

import importlib
import sys


def test_ecs_g_namespace_is_materialized_and_inert():
    before = set(sys.modules)
    module = importlib.import_module("elpis.ECS_G")
    after = set(sys.modules)

    assert module.__name__ == "elpis.ECS_G"
    assert "numpy" not in after - before
    assert not hasattr(module, "Kernel")
    assert not hasattr(module, "step")
