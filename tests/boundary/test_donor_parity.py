"""Parity with the read-only donor at the migration basis.

These checks need a checkout of the donor at the basis commit, named by
``ELPIS_DONOR_ROOT``. Without it they skip; CI's parity job sets
``ELPIS_REQUIRE_DONOR=1`` so a missing donor is a failure there. The donor is
only read: git plumbing reads and a Python subprocess that imports donor
modules with bytecode writing disabled.
"""
from __future__ import annotations

import filecmp
import json
import os
from pathlib import Path
import subprocess
import sys

import pytest

from ._system import REPO, SYSTEM

DONOR = os.environ.get("ELPIS_DONOR_ROOT")


@pytest.fixture(scope="module")
def donor() -> Path:
    if not DONOR:
        if os.environ.get("ELPIS_REQUIRE_DONOR") == "1":
            pytest.fail("ELPIS_REQUIRE_DONOR=1 but ELPIS_DONOR_ROOT is not set")
        pytest.skip("ELPIS_DONOR_ROOT not set (donor parity runs in the CI parity job)")
    root = Path(DONOR).resolve()
    assert (root / ".git").exists(), root
    return root


def _git(root: Path, *args: str) -> str:
    return subprocess.run(["git", "-C", str(root), *args], capture_output=True, text=True,
                          check=True).stdout.strip()


def test_donor_checkout_is_the_unmodified_basis(donor):
    assert _git(donor, "rev-parse", "HEAD") == SYSTEM["donor"]["migration_basis_commit"]
    assert _git(donor, "rev-parse", "HEAD^{tree}") == SYSTEM["donor"]["migration_basis_tree"]
    assert _git(donor, "status", "--porcelain", "--untracked-files=no") == ""


BYTE_IDENTICAL = [
    ("tests/fixtures/inference/dsv41.json", "tests/fixtures/inference_r0/dsv41.json"),
    ("tests/fixtures/inference/qwen38.json", "tests/fixtures/inference_r0/qwen38.json"),
    ("LICENSE", "LICENSE"),
] + [(f"LICENSES/{name}.txt", f"components/InferenceInfrastructure/licenses/{name}.txt")
     for name in ("Apache-2.0", "DS4-MIT", "DeepSeek-V41-MIT", "DeepSpec-MIT")]


@pytest.mark.parametrize(("ours", "theirs"), BYTE_IDENTICAL, ids=[p[0] for p in BYTE_IDENTICAL])
def test_migrated_files_are_byte_identical(donor, ours, theirs):
    assert (REPO / ours).read_bytes() == (donor / theirs).read_bytes()


def test_historical_grid81_canonical_fixture_is_byte_identical(donor):
    ours = REPO / "tests/fixtures/grid81/Canonical/Grid81"
    theirs = donor / "components/Grid81/state/Canonical/Grid81"
    names = sorted(p.relative_to(ours).as_posix() for p in ours.rglob("*") if p.is_file())
    assert names == sorted(p.relative_to(theirs).as_posix() for p in theirs.rglob("*") if p.is_file())
    match, mismatch, errors = filecmp.cmpfiles(ours, theirs, names, shallow=False)
    assert not mismatch and not errors and len(match) == len(names)


_PROBE = r"""
import hashlib, json, sys, tempfile
side = sys.argv[1]
d = lambda s: hashlib.sha256(s.encode()).hexdigest()
if side == "donor":
    from elpis_evolution_path_gate import EvolutionPathAssertion, HarnessManifest, content_map_digest
    from elpis_ecs.kernel import Kernel
    from elpis.canonical_identity import content_digest
else:
    # v0 is a retired persisted assertion schema; its identity stays computable.
    from elpis.evolution import EvolutionPathAssertionV0 as EvolutionPathAssertion
    from elpis.evolution import HarnessManifest, content_map_digest
    from elpis.ECS_C.kernel import Kernel
    from elpis.identity import content_digest
a = EvolutionPathAssertion("ep", d("s"), 2, d("h"), d("p"), d("c"), d("y"), ("x/y",), 1, 2,
                           d("r"), d("e"), d("pj"), d("he"), d("root"))
m = HarnessManifest(1, d("parent"), d("cand"), d("path"), d("edit"),
                    (("a.txt", d("a")), ("b/c.txt", d("b"))), d("cfg"), d("tool"), d("pol"), d("build"))
with tempfile.TemporaryDirectory() as tmp:
    with Kernel(tmp).open() as k:
        x = k.found_entity("population"); y = k.found_entity("environment")
        k.run_until_quiescent()
        k.entity_port(y).propose(x, b"observation")
        k.run_until_quiescent()
        root = k.state_root_digest()
        heads = [e["event_digest"] for e in k.events()]
print(json.dumps({"assertion": a.digest, "manifest": m.digest,
                  "content_map": content_map_digest((("a", d("a")),)),
                  "identity": content_digest("elpis.parity.v1", {"k": [1, "x", None]}),
                  "ecs_state_root": root, "ecs_events": heads}, sort_keys=True))
"""


def _probe(side: str, paths: list[Path]) -> dict:
    env = {"PATH": os.environ.get("PATH", ""), "PYTHONDONTWRITEBYTECODE": "1", "PYTHONHASHSEED": "0",
           "PYTHONPATH": os.pathsep.join(str(p) for p in paths)}
    result = subprocess.run([sys.executable, "-c", _PROBE, side], capture_output=True, text=True, env=env)
    assert result.returncode == 0, result.stderr
    return json.loads(result.stdout)


def test_persisted_identities_match_the_donor(donor):
    theirs = _probe("donor", [donor / "components/EvolutionPathGate/src", donor / "ECS/runtime", donor / "src"])
    ours = _probe("ours", [REPO / "src"])
    assert ours == theirs
