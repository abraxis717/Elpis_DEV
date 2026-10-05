"""The current-runtime regression needs NumPy (the ``test`` extra) and the native build; without NumPy it is not
collected. The plan and record checks need neither and always run."""
import importlib.util

if importlib.util.find_spec("numpy") is None:
    collect_ignore_glob = ["test_runtime_regression.py"]
