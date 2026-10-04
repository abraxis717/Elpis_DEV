"""The laboratory mechanics need NumPy (the ``test`` extra); without it they are not collected.

The preregistration guard (test_preregistration.py) needs neither NumPy nor the native library and always runs.
"""
import importlib.util

if importlib.util.find_spec("numpy") is None:
    collect_ignore_glob = ["test_mechanics.py", "test_evidence.py", "test_runtime_regression.py"]
