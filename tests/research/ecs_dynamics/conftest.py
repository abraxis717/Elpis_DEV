"""The laboratory needs NumPy (the ``test`` extra); without it these tests are not collected."""
import importlib.util

if importlib.util.find_spec("numpy") is None:
    collect_ignore_glob = ["test_*.py"]
