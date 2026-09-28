"""Shared helpers for boundary tests: the system authority and repository files."""
from __future__ import annotations

import ast
import json
import os
from pathlib import Path
import subprocess

REPO = Path(__file__).resolve().parents[2]
SYSTEM = json.loads((REPO / "ELPIS_SYSTEM.json").read_text(encoding="utf-8"))
MIGRATION = json.loads((REPO / "migration" / "BETA_MIGRATION.json").read_text(encoding="utf-8"))

_EXCLUDED_DIRS = {".git", ".venv", "venv", "__pycache__", ".pytest_cache", ".eggs"}


def live_subsystems():
    return [s for s in SYSTEM["subsystems"] if s["status"] != "PLANNED"]


def subsystem_ids():
    return [s["id"] for s in SYSTEM["subsystems"]]


def repository_files() -> list[Path]:
    """Tracked files when run from a Git checkout, otherwise a filtered walk."""
    try:
        out = subprocess.run(
            ["git", "ls-files", "-z"], cwd=REPO, capture_output=True, check=True
        ).stdout
        names = [n for n in out.decode("utf-8").split("\0") if n]
        if names:
            return [REPO / n for n in names if (REPO / n).exists()]
    except (OSError, subprocess.CalledProcessError):
        pass
    files = []
    for dirpath, dirnames, filenames in os.walk(REPO):
        dirnames[:] = [
            d for d in dirnames
            if d not in _EXCLUDED_DIRS and not d.startswith("build") and not d.endswith(".egg-info")
        ]
        files.extend(Path(dirpath) / f for f in filenames)
    return files


def python_modules(root: Path):
    for path in sorted(root.rglob("*.py")):
        if "__pycache__" in path.parts:
            continue
        yield path


def module_name(path: Path) -> str:
    parts = list(path.relative_to(REPO / "src").with_suffix("").parts)
    if parts[-1] == "__init__":
        parts = parts[:-1]
    return ".".join(parts)


def imported_modules(path: Path) -> set[str]:
    """Absolute names of every module imported by ``path`` (relative imports resolved)."""
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    current = module_name(path).split(".")
    # The package that relative level 1 refers to.
    package = current if path.name == "__init__.py" else current[:-1]
    names: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            names.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            if node.level == 0:
                if node.module:
                    names.add(node.module)
                continue
            base = package[: len(package) - (node.level - 1)]
            if node.module:
                names.add(".".join(base + [node.module]))
            else:
                names.update(".".join(base + [alias.name]) for alias in node.names)
    return names
