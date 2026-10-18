"""K1 Recovery R0 v4: the unchanged-law premise, bound to source.

The specification asks the ABI v4 durability law of the ABI v5 library. That is meaningful only while the law's
implementation is the ABI v4 one. ``check(repo, v4, commit)`` verifies it mechanically against ``v4`` (the last ABI v4
commit, ``plan.json`` ``premise.abi_v4_commit``) and returns every way it does not hold:

* P1 the continuity authority (``native/continuity``) is byte-identical;
* P2 the ABI v4 durability-law test files are byte-identical (they run in CI against the v5 library);
* P3 every native recovery and crash test of ABI v4 (``tests.rs`` ``mod recovery``) exists with an identical body;
* P4 the durability lane and the checkpoint store (``durable.rs``, ``checkpoint.rs``) are identical once their
  testing-only items (``cfg(feature = "testing")``, ``cfg(any(test, feature = "testing"))``) and comments are removed.

What it cannot decide mechanically it records with a digest instead of claiming: the control operations in ``core.rs``
were restructured to carry the testing-only crash points; their production equivalence rests on the review recorded in
the specification (identical exported symbols and call targets), and the ABI v5 turn changes in ``core.rs`` are outside
the durability law (they are Cognitive Boundary's subject).
"""
from __future__ import annotations

import hashlib
import re
import subprocess
from pathlib import Path

PREMISE_SCHEMA = "elpis.k1-recovery-r0-v4.premise.v1"
LAW_TESTS = ("tests/integration/test_k1_recovery.py", "tests/integration/test_durable_process_kill.py",
             "tests/integration/test_durable_ticket_law.py", "tests/integration/test_composite_process_kill.py",
             "tests/integration/test_autonomous_no_growth.py")
LAW_SOURCES = ("native/runtime/src/durable.rs", "native/runtime/src/checkpoint.rs")
_TESTING = re.compile(r'#\[cfg\((?:feature = "testing"|any\(test, feature = "testing"\))\)\]')


def _show(repo: Path, commit: str, path: str) -> str:
    return subprocess.run(["git", "-C", str(repo), "show", f"{commit}:{path}"], capture_output=True, text=True,
                          check=True).stdout


def _tree(repo: Path, commit: str, path: str) -> str:
    return subprocess.run(["git", "-C", str(repo), "rev-parse", f"{commit}:{path}"], capture_output=True, text=True,
                          check=True).stdout.strip()


def _strip_comments(text: str) -> str:
    return "\n".join(line.split("//", 1)[0].rstrip() for line in text.splitlines())


def _end_of_item(text: str, start: int) -> int:
    """The end of the item or statement starting at ``start``: its matching brace block, or its ``;`` / ``,`` outside
    every bracket (a parameter list's commas do not end a function)."""
    depth, nest, i = 0, 0, start
    while i < len(text):
        ch = text[i]
        if ch in "([":
            nest += 1
        elif ch in ")]":
            nest -= 1
        elif ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return i + 1
            if depth < 0:
                return i
        elif ch in ";," and depth == 0 and nest == 0:
            return i + 1
        i += 1
    return i


def production(text: str) -> str:
    """The source with every testing-only item removed and comments and blank lines dropped (whitespace-normalized):
    what a production build of the file contains."""
    text = _strip_comments(text)
    while True:
        m = _TESTING.search(text)
        if not m:
            break
        text = text[:m.start()] + text[_end_of_item(text, m.end()):]
    return "\n".join(" ".join(line.split()) for line in text.splitlines() if line.strip())


def _test_functions(text: str, module: str) -> dict:
    """``{name: body}`` of every ``#[test] fn`` inside ``mod <module> {`` (nested modules included)."""
    start = text.find(f"mod {module} {{")
    if start < 0:
        return {}
    body = text[start:_end_of_item(text, text.index("{", start))]
    out = {}
    for m in re.finditer(r"#\[test\]\s*fn (\w+)\(\)", body):
        open_brace = body.index("{", m.end())
        out[m.group(1)] = " ".join(body[open_brace:_end_of_item(body, open_brace)].split())
    return out


def check(repo: Path, v4: str, commit: str = "HEAD") -> dict:
    problems = []
    if _tree(repo, v4, "native/continuity") != _tree(repo, commit, "native/continuity"):
        problems.append("P1: native/continuity differs from the ABI v4 commit")
    for path in LAW_TESTS:
        if _show(repo, v4, path) != _show(repo, commit, path):
            problems.append(f"P2: {path} differs from the ABI v4 commit")
    old = _test_functions(_show(repo, v4, "native/runtime/src/tests.rs"), "recovery")
    new = _test_functions(_show(repo, commit, "native/runtime/src/tests.rs"), "recovery")
    if not old:
        problems.append("P3: no ABI v4 recovery tests were found")
    for name, body in old.items():
        if new.get(name) != body:
            problems.append(f"P3: the ABI v4 recovery test {name} is missing or changed")
    for path in LAW_SOURCES:
        if production(_show(repo, v4, path)) != production(_show(repo, commit, path)):
            problems.append(f"P4: the production durability law in {path} differs from the ABI v4 commit")
    core_diff = subprocess.run(["git", "-C", str(repo), "diff", v4, commit, "--", "native/runtime/src/core.rs"],
                               capture_output=True, text=True, check=True).stdout
    return {"schema": PREMISE_SCHEMA, "abi_v4_commit": v4, "commit": subprocess.run(
                ["git", "-C", str(repo), "rev-parse", commit], capture_output=True, text=True, check=True).stdout.strip(),
            "abi_v4_recovery_tests": sorted(old), "problems": problems,
            "reviewed_not_mechanical": {"native/runtime/src/core.rs": {
                "diff_sha256": hashlib.sha256(core_diff.encode()).hexdigest(),
                "basis": "control operations restructured for the testing-only crash points (production equivalence "
                         "per the 2026-10-18 review: identical exported symbols and call targets); the ABI v5 turn "
                         "changes are outside the durability law"}}}
