"""Bind frozen evidence to the current tree across the one ECS rename (ECS_G -> ECS).

Recorded evidence names the files it measured under the paths they had when it
was recorded (``src/elpis/ECS_G/...``, ``native/ECS_G/...``) together with their
SHA-256. The evidence is immutable and is never rewritten. The ECS was later
renamed to its canonical name (docs/CONTINUITY.md, docs/ARCHITECTURE.md). This
module states that rename exactly and proves, per file, that the current file is
the measured file:

* native C sources, headers and tests moved byte-for-byte: the current bytes
  must have the recorded SHA-256;
* the measured Python modules changed only by the rename itself: their
  docstrings, and the literal text ``ECS_G`` -> ``ECS`` in string constants
  (error messages). The recorded SHA-256 must match the git blob at the
  evidence's own recorded head, and the current module must be equal to that
  blob as a Python AST once docstrings are removed and that one textual rename
  is applied. Any other change (code, names, constants, imports) fails.
"""
from __future__ import annotations

import ast
import hashlib
from pathlib import Path
import subprocess

REPO = Path(__file__).resolve().parents[2]

# The rename, exactly. Nothing else is mapped.
RENAMED_ROOTS = (
    ("src/elpis/ECS_G/", "src/elpis/ECS/"),
    ("native/ECS_G/", "native/ECS/"),
    ("tests/ECS_G/", "tests/ECS/"),
)


def current_path(recorded: str) -> str:
    for old, new in RENAMED_ROOTS:
        if recorded.startswith(old):
            return new + recorded[len(old):]
    return recorded


def _sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _blob(commit: str, path: str) -> bytes | None:
    result = subprocess.run(["git", "show", f"{commit}:{path}"], cwd=REPO, capture_output=True)
    return result.stdout if result.returncode == 0 else None


class _Normalize(ast.NodeTransformer):
    """Drop docstrings; apply the rename to string constants of the recorded side only."""

    def __init__(self, rename: bool):
        self.rename = rename

    def _strip(self, node):
        body = getattr(node, "body", None)
        if (body and isinstance(body[0], ast.Expr) and isinstance(body[0].value, ast.Constant)
                and isinstance(body[0].value.value, str)):
            node.body = body[1:] or [ast.Pass()]
        return node

    def visit_Module(self, node):
        self.generic_visit(node)
        return self._strip(node)

    def visit_ClassDef(self, node):
        self.generic_visit(node)
        return self._strip(node)

    def visit_FunctionDef(self, node):
        self.generic_visit(node)
        return self._strip(node)

    visit_AsyncFunctionDef = visit_FunctionDef

    def visit_Constant(self, node):
        if self.rename and isinstance(node.value, str):
            return ast.copy_location(ast.Constant(node.value.replace("ECS_G", "ECS")), node)
        return node


def _normalized(source: bytes, *, rename: bool) -> str:
    tree = _Normalize(rename).visit(ast.parse(source))
    return ast.dump(tree, include_attributes=False)


def measured_file_is_current(recorded_path: str, recorded_sha: str, recorded_head: str) -> bool:
    """True iff the current file is the file the evidence measured, modulo the stated rename."""
    current = REPO / current_path(recorded_path)
    if not current.is_file():
        return False
    data = current.read_bytes()
    if _sha(data) == recorded_sha:
        return True
    if not recorded_path.endswith(".py"):
        return False
    blob = _blob(recorded_head, recorded_path)
    if blob is None or _sha(blob) != recorded_sha:
        return False
    return _normalized(blob, rename=True) == _normalized(data, rename=False)


def renamed_lab_binding(protocol) -> dict:
    """A closed laboratory's frozen file-binding constants, re-pointed at the renamed tree.

    The laboratories' source text is frozen, so their binding constants still name the pre-rename layout and can
    no longer be hashed. Tests apply these values (in memory only; the laboratory files are never edited) so the
    frozen ``implementation()`` can run against the current tree. The bound *names* then differ from every
    recorded binding, so every historical replay correctly reports a stale binding and is skipped rather than
    claimed; nothing new can be frozen against a record.
    """
    values = {}
    for attr in ("_BOUND_GLOBS", "_BOUND_EXTRA", "BOUND_FILES"):
        if hasattr(protocol, attr):
            values[attr] = tuple(current_path(name) for name in getattr(protocol, attr))
    return values


def apply_renamed_lab_binding(protocol) -> None:
    """For historical-replay subprocesses: apply :func:`renamed_lab_binding` to the process."""
    for attr, value in renamed_lab_binding(protocol).items():
        setattr(protocol, attr, value)
