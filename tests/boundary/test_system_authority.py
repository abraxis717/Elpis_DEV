"""ELPIS_SYSTEM.json is the single system authority, and the repository obeys it."""
from __future__ import annotations

import re
import tomllib

import pytest

from ._system import REPO, SYSTEM, live_subsystems, repository_files, subsystem_ids

EXPECTED_SUBSYSTEMS = ["substrate", "ecs", "structure", "pipeline", "evolution", "inference", "runtime"]
STATUSES = {"PLANNED", "OPERATIONAL", "OPERATIONAL_INCOMPLETE_INTEGRATION"}


def test_authority_schema_and_subsystem_set():
    assert SYSTEM["schema"] == "elpis.system.v1"
    assert subsystem_ids() == EXPECTED_SUBSYSTEMS
    for sub in SYSTEM["subsystems"]:
        assert sub["status"] in STATUSES, sub["id"]
        assert sub["python_package"] == "elpis." + sub["id"]
        assert set(sub["depends_on"]) <= set(EXPECTED_SUBSYSTEMS) - {sub["id"]}


def test_subsystem_dependencies_are_acyclic():
    graph = {s["id"]: set(s["depends_on"]) for s in SYSTEM["subsystems"]}
    visiting, done = set(), set()

    def visit(node):
        assert node not in visiting, f"dependency cycle through {node}"
        if node in done:
            return
        visiting.add(node)
        for dep in graph[node]:
            visit(dep)
        visiting.discard(node)
        done.add(node)

    for node in graph:
        visit(node)


@pytest.mark.parametrize("sub", SYSTEM["subsystems"], ids=lambda s: s["id"])
def test_declared_roots_exist_and_planned_subsystems_are_not_materialized(sub):
    package_dir = REPO / "src" / "elpis" / sub["id"]
    if sub["status"] == "PLANNED":
        assert not package_dir.exists(), f"{sub['id']} materialized without being declared live"
        assert not sub["implementation_roots"] and not sub["native_targets"]
        return
    assert package_dir.is_dir()
    assert sub["implementation_roots"], sub["id"]
    for root in sub["implementation_roots"] + sub["test_roots"]:
        assert (REPO / root).exists(), root
    assert any((REPO / root).rglob("test_*.py") for root in sub["test_roots"]), sub["id"]
    for key in ("purpose", "mutation_authority", "runtime_participation", "model_dependency", "maturity"):
        assert sub.get(key), (sub["id"], key)


def test_every_python_package_is_owned():
    owned = {"elpis." + s["id"] for s in live_subsystems()}
    shared = {m["module"] for m in SYSTEM["shared_modules"]}
    src = REPO / "src" / "elpis"
    for entry in sorted(src.iterdir()):
        if entry.name in ("__pycache__",):
            continue
        if entry.name == "__init__.py":
            continue
        name = "elpis." + (entry.stem if entry.is_file() else entry.name)
        assert name in owned | shared, f"unclassified python surface {entry.relative_to(REPO)}"
    for module in SYSTEM["shared_modules"]:
        assert (REPO / module["path"]).is_file()


def test_every_top_level_surface_is_classified():
    surfaces = SYSTEM["repository_surfaces"]
    top_level = {path.relative_to(REPO).parts[0] for path in repository_files()}
    unclassified = sorted(top_level - set(surfaces))
    assert not unclassified, f"unclassified top-level surfaces: {unclassified}"
    missing = sorted(name for name in surfaces if not (REPO / name).exists())
    assert not missing, f"declared surfaces missing: {missing}"


def test_forbidden_beta_structure_is_absent():
    policy = SYSTEM["boundary_policy"]
    for name in policy["forbidden_top_level_paths"]:
        assert not (REPO / name).exists(), f"beta surface resurrected: {name}"
    patterns = [re.compile(p) for p in policy["forbidden_directory_name_patterns"]]
    forbidden_files = set(policy["forbidden_files"])
    for path in repository_files():
        rel = path.relative_to(REPO)
        assert rel.name not in forbidden_files, f"beta authority file resurrected: {rel}"
        for part in rel.parts[:-1]:
            assert not any(p.match(part) for p in patterns), f"beta directory taxonomy resurrected: {rel}"


def test_distribution_identity_is_development_only():
    project = tomllib.loads((REPO / "pyproject.toml").read_text(encoding="utf-8"))["project"]
    assert project["name"] == SYSTEM["identity"]["python_distribution"] == "elpis-dev"
    assert project["name"] not in ("elpis", "elpisai")
    assert ".dev" in project["version"]
    assert "Private :: Do Not Upload" in project["classifiers"]
    assert "scripts" not in project, "no console entry point claims the beta `elpis` command"
    assert SYSTEM["identity"]["release_authority"] == "NONE_DEFERRED"


def test_no_publication_workflow():
    workflows = REPO / ".github" / "workflows"
    for path in sorted(workflows.glob("*.y*ml")) if workflows.is_dir() else []:
        text = path.read_text(encoding="utf-8").lower()
        for marker in ("pypi", "twine", "gh-action-pypi-publish", "id-token: write", "release:"):
            assert marker not in text, f"{path.name} carries publication authority marker {marker!r}"
