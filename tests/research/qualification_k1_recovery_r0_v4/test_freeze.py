"""K1 Recovery R0 v4: freeze verification authenticates the executing object, not a well-formed JSON file.

Each test changes exactly one thing the freeze binds and shows the refusal names it: the checkout (commit, tree,
cleanliness), the instrument (plan, runner, checker, freeze producer, specification), the subject source, the
interpreter and the package it actually imports (isolated mode makes PYTHONPATH irrelevant, so the import itself is
measured), the build-tree libraries, the deployed home/lib copies and their pins, the reported ABI, the toolchain,
the platform and the filesystem. A QUAL run that is refused creates nothing. No QUAL input is opened here.
"""
from __future__ import annotations

import copy
import importlib.util
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

import pytest

from ...conftest import native_build_dir, require_runtime_library

REPO = Path(__file__).resolve().parents[3]
INSTRUMENT = REPO / "research" / "qualification" / "k1_recovery_r0_v4"


def _load(name: str):
    spec = importlib.util.spec_from_file_location(f"k1r4_{name}", INSTRUMENT / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


freeze = _load("freeze")

_OBSERVE = r"""
import importlib.util, json, sys
from pathlib import Path
spec = importlib.util.spec_from_file_location("f", sys.argv[1])
f = importlib.util.module_from_spec(spec); spec.loader.exec_module(f)
print(json.dumps(f.observe(native_build=Path(sys.argv[2]), work_root=Path(sys.argv[3]))))
"""


def _observe(tmp_path, *flags, env=None) -> dict:
    out = subprocess.run([sys.executable, *flags, "-c", _OBSERVE, str(INSTRUMENT / "freeze.py"),
                          str(native_build_dir()), str(tmp_path)], capture_output=True, text=True, check=True,
                         env=env, cwd=tmp_path)
    return json.loads(out.stdout)


@pytest.fixture(scope="module")
def observed(tmp_path_factory):
    require_runtime_library(testing=True)
    return _observe(tmp_path_factory.mktemp("observe"), "-I", "-B")


def _as_freeze(observed: dict) -> dict:
    return dict(copy.deepcopy(observed), schema=freeze.FREEZE_SCHEMA)


def test_an_isolated_interpreter_imports_the_checkouts_own_package_whatever_pythonpath_says(tmp_path, observed):
    assert observed["python"]["isolated"] == 1
    assert observed["python"]["package"] == observed["python"]["source_package"]
    assert observed["python"]["modules_outside_package"] == []
    # A decoy package on PYTHONPATH: isolated mode ignores it, so the measured package is still the real one ...
    decoy = tmp_path / "decoy" / "elpis"
    decoy.mkdir(parents=True)
    (decoy / "__init__.py").write_text("DECOY = True\n")
    env = dict(os.environ, PYTHONPATH=str(decoy.parent))
    assert _observe(tmp_path, "-I", "-B", env=env)["python"]["package_root"] == observed["python"]["package_root"]
    # ... while a non-isolated interpreter would import the decoy (it shadows the real package): such an observation
    # is refused, as not isolated and as not the checkout's package.
    shadow = subprocess.run([sys.executable, "-B", "-c", "import elpis; print(elpis.__file__)"], capture_output=True,
                            text=True, check=True, env=env, cwd=tmp_path).stdout.strip()
    assert Path(shadow).resolve().parent == decoy.resolve()
    wrong = copy.deepcopy(observed)
    wrong["python"].update(isolated=0, package_root=str(decoy), package=freeze.tree_digest({"__init__.py": "0"}))
    problems = freeze.compare(_as_freeze(observed), wrong)
    assert any("isolated" in p for p in problems) and any("not the checkout's src/elpis" in p for p in problems)
    assert any(p.startswith("python.package") for p in problems)


_FIELDS = [
    (("source", "commit"), "0" * 40), (("source", "tree"), "1" * 40),
    (("instrument", "files", "plan.json"), "2" * 64), (("instrument", "files", "runner.py"), "3" * 64),
    (("instrument", "files", "check.py"), "4" * 64), (("instrument", "files", "freeze.py"), "5" * 64),
    (("instrument", "specification"), "6" * 64), (("subject", "native/runtime/src/durable.rs"), "7" * 64),
    (("python", "executable"), "/usr/bin/python-other"), (("python", "version"), "3.0.0"),
    (("python", "package"), "8" * 64), (("native", "native/runtime/libelpis_runtime_testing.so"), "9" * 64),
    (("toolchain", "cc"), "cc (other) 1.0"), (("platform", "kernel"), "0.0.0"), (("filesystem", "type"), "fat"),
]


@pytest.mark.parametrize("path,value", _FIELDS, ids=[".".join(p) for p, _ in _FIELDS])
def test_every_bound_field_refuses_when_the_executing_environment_differs(observed, path, value):
    baseline = _as_freeze(observed)
    clean = freeze.compare(baseline, observed)
    mutated = copy.deepcopy(baseline)
    target = mutated
    for key in path[:-1]:
        target = target[key]
    assert target.get(path[-1]) != value
    target[path[-1]] = value
    problems = set(freeze.compare(mutated, observed)) - set(clean)
    top = ".".join(path[:2]) if path[0] in ("source", "python", "filesystem") else path[0]
    assert any(p.startswith(top) for p in problems), (path, problems)


def test_a_freeze_record_of_another_schema_is_refused(observed):
    assert freeze.compare(dict(_as_freeze(observed), schema="elpis.k1-recovery-r0-v4.freeze.v1"), observed) == \
        [f"not a {freeze.FREEZE_SCHEMA} record"]


@pytest.mark.parametrize("change,needle", [
    (lambda o: o["source"].update(clean=False, status=[" M src/elpis/runtime/core.py"]), "not clean"),
    (lambda o: o["python"].update(isolated=0), "isolated"),
    (lambda o: o["python"].update(package="a" * 64), "not the checkout's src/elpis"),
    (lambda o: o["python"].update(modules_outside_package=["elpis.runtime.core"]), "outside the package"),
    (lambda o: o["native"].update({"native/ECS_C/libelpis_ecs_memory.so": None}), "missing"),
])
def test_the_preconditions_refuse_whatever_freeze_record_is_offered(observed, change, needle):
    bad = copy.deepcopy(observed)
    change(bad)
    assert any(needle in p for p in freeze.compare(_as_freeze(bad), bad))


def _git(repo, *args):
    subprocess.run(["git", "-C", str(repo), *args], check=True, capture_output=True,
                   env=dict(os.environ, GIT_AUTHOR_NAME="t", GIT_AUTHOR_EMAIL="t@t", GIT_COMMITTER_NAME="t",
                            GIT_COMMITTER_EMAIL="t@t"))


def test_the_source_measure_sees_a_replaced_checkout_an_edit_and_an_untracked_file(tmp_path):
    repo = tmp_path / "repo"
    repo.mkdir()
    _git(repo, "init", "-q")
    (repo / "a.py").write_text("A = 1\n")
    _git(repo, "add", "a.py")
    _git(repo, "commit", "-q", "-m", "one")
    first = freeze.source(repo)
    assert first["clean"] is True
    (repo / "a.py").write_text("A = 2\n")
    assert freeze.source(repo)["clean"] is False                     # an uncommitted edit
    _git(repo, "checkout", "-q", "a.py")
    (repo / "b.py").write_text("B = 1\n")
    assert freeze.source(repo)["clean"] is False                     # an untracked file
    _git(repo, "add", "b.py")
    _git(repo, "commit", "-q", "-m", "two")
    second = freeze.source(repo)
    assert second["clean"] is True and (second["commit"], second["tree"]) != (first["commit"], first["tree"])


def test_the_instrument_measure_covers_every_file_including_the_freeze_producer(tmp_path):
    copy_root = tmp_path / "repo" / INSTRUMENT.relative_to(REPO)
    shutil.copytree(INSTRUMENT, copy_root, ignore=shutil.ignore_patterns("__pycache__", "evidence"))
    (tmp_path / "repo" / "docs" / "qualification").mkdir(parents=True)
    shutil.copyfile(REPO / freeze.SPECIFICATION, tmp_path / "repo" / freeze.SPECIFICATION)
    base = freeze.instrument(tmp_path / "repo")
    assert {"plan.json", "runner.py", "check.py", "freeze.py"} <= set(base["files"])
    for name in ("plan.json", "runner.py", "check.py", "freeze.py"):
        original = (copy_root / name).read_bytes()
        (copy_root / name).write_bytes(original + b"\n")
        assert freeze.instrument(tmp_path / "repo")["files"][name] != base["files"][name], name
        (copy_root / name).write_bytes(original)
    (copy_root / "extra.py").write_text("X = 1\n")                   # an added file is part of the instrument too
    assert "extra.py" in freeze.instrument(tmp_path / "repo")["files"]


# The deployment the frozen run measured at the qualifying commit (its sealed freeze record).
SEALED_FREEZE = json.loads((INSTRUMENT / "evidence" / "freeze.json").read_text())


def test_the_successor_runtime_cannot_be_deployed_by_the_frozen_instrument(tmp_path, observed):
    # The qualified subject is RuntimeCore ABI v5 over standalone continuity (the qualifying commit). This checkout's
    # RuntimeCore is ABI v7 (Stage F, docs/NATIVE_SPLIT_CENSUS.md, section 9): continuity retired into ECS_C, and the
    # runtime no longer carries continuity's C ABI. The frozen instrument cannot even measure a deployment of it, so
    # no freeze of this checkout can exist: the RUN PASS can never be re-derived from, or extended to, the successor.
    with pytest.raises(AttributeError, match="elpis_continuity_abi_version"):
        freeze.trial_deployment(native_build_dir(), tmp_path / "deployment")


def test_the_deployed_copies_their_pins_and_the_reported_abi_are_verified():
    # Over the deployment the qualifying run measured: the same copies, pins and ABI are accepted, and every check
    # still binds (the successor's own deployment cannot be measured at all: see above).
    frozen = copy.deepcopy(SEALED_FREEZE)
    trial = copy.deepcopy(frozen["deployed"])
    assert trial["deployment_abi"] == {"runtime": 5, "composite": 1} and trial["reported_abi"]["runtime"] == 5
    assert freeze.compare_deployed(frozen, copy.deepcopy(trial)) == []
    # A swapped deployment library: other bytes under the testing runtime's name.
    swapped = dict(copy.deepcopy(trial), lib=dict(trial["lib"], **{"libelpis_runtime_testing.so": "0" * 64}))
    assert any("home/lib" in p for p in freeze.compare_deployed(frozen, swapped))
    for key, value in (("catalog_sha256", "0" * 64), ("deployment_abi", {"runtime": 4, "composite": 1}),
                       ("reported_abi", dict(trial["reported_abi"], runtime=4))):
        stale = dict(copy.deepcopy(trial), **{key: value})
        assert any(f"deployed.{key}" in p for p in freeze.compare_deployed(frozen, stale)), key
    # A runtime of another ABI (the successor's, v7) is refused by name, whatever else matches.
    v7 = dict(copy.deepcopy(trial), deployment_abi={"runtime": 7, "composite": 1},
              reported_abi=dict(trial["reported_abi"], runtime=7))
    assert f"the deployment is not RuntimeCore ABI v5 (composite ABI 1): {v7['deployment_abi']}, " \
           f"reported {v7['reported_abi']}" in freeze.compare_deployed(dict(frozen, deployed=v7), v7)


def test_a_refused_qual_run_creates_nothing(tmp_path, observed):
    runner = INSTRUMENT / "runner.py"
    base = [sys.executable, "-I", "-B", str(runner), "run", "--native-build", str(native_build_dir()),
            "--inputs", "qual", "--out", str(tmp_path / "records.jsonl")]
    record = tmp_path / "FREEZE.json"
    record.write_text(json.dumps(dict(_as_freeze(observed), source=dict(observed["source"], commit="0" * 40))))
    for extra, needle in (([], "only under a verified freeze record"),
                          (["--freeze", str(record), "--only", "publish"], "every trial"),
                          (["--freeze", str(record)], "source.commit differs")):
        work = tmp_path / f"work-{len(extra)}"
        done = subprocess.run(base + ["--work", str(work)] + extra, capture_output=True, text=True, cwd=tmp_path)
        assert done.returncode != 0 and needle in done.stderr, done.stderr
        assert not work.exists() and not (tmp_path / "records.jsonl").exists()


premise = _load("premise")
_V4 = json.loads((INSTRUMENT / "plan.json").read_text())["premise"]["abi_v4_commit"]


def test_the_production_view_drops_exactly_the_testing_only_items():
    base = "pub fn law(a: u32, b: u32) -> u32 {\n    a + b\n}\nstruct S {\n    x: u32,\n}\n"
    testing = ('pub fn law(a: u32, b: u32) -> u32 {\n    #[cfg(feature = "testing")]\n    if a == 9 {\n        return 0;\n'
               '    }\n    a + b // the law\n}\n#[cfg(any(test, feature = "testing"))]\npub fn hook(\n    a: u32,\n    b: u32,\n'
               ') -> u32 {\n    a\n}\nstruct S {\n    x: u32,\n    #[cfg(feature = "testing")]\n    plan: Option<u32>,\n}\n')
    assert premise.production(testing) == premise.production(base)
    assert premise.production(base.replace("a + b", "a - b")) != premise.production(base)   # a law change remains


def test_the_unchanged_law_premise_held_at_the_qualifying_commit_and_breaks_here():
    # At the qualifying commit (the accepted source the evidence's RUN PASS names), the premise held: re-checked from
    # git, and equal to what the run recorded.
    observed = json.loads((INSTRUMENT / "evidence/run.json").read_text())["observed"]
    at_qualifying = premise.check(REPO, _V4, observed["source"]["commit"])
    assert at_qualifying["problems"] == [] and len(at_qualifying["abi_v4_recovery_tests"]) >= 21
    assert at_qualifying == observed["premise"]
    # This checkout is past Stage F (RuntimeCore ABI v7, docs/NATIVE_SPLIT_CENSUS.md, section 9): standalone
    # continuity retired into ECS_C, so P1 breaks, and the production durability law and its tests moved with the
    # ECS_C authority (P2-P4). The RUN PASS stays bound to its commit; it does not extend to this RuntimeCore.
    here = premise.check(REPO, _V4)
    assert here["problems"][0] == "P1: native/continuity differs from the ABI v4 commit", here["problems"]
    assert "P4: the production durability law in native/runtime/src/durable.rs differs from the ABI v4 commit" in \
        here["problems"]


@pytest.mark.parametrize("path,edit,needle", [
    ("native/runtime/src/durable.rs", lambda t: t.replace("Status::Acknowledged", "Status::Uncertain", 1), "P4"),
    ("native/runtime/src/tests.rs", lambda t: t.replace("fn provisioning_is_fixed_and_never_overwrites",
                                                        "fn provisioning_renamed", 1), "P3"),
    ("tests/integration/test_k1_recovery.py", lambda t: t + "\n# changed\n", "P2"),
])
def test_a_changed_law_or_law_test_breaks_the_premise(monkeypatch, path, edit, needle):
    # Edited at the qualifying commit, where the premise holds (this checkout already breaks it: see above).
    qualifying = json.loads((INSTRUMENT / "evidence/run.json").read_text())["observed"]["source"]["commit"]
    real = premise._show

    def show(repo, commit, p):
        text = real(repo, commit, p)
        return edit(text) if commit == qualifying and p == path else text

    monkeypatch.setattr(premise, "_show", show)
    assert [p for p in premise.check(REPO, _V4, qualifying)["problems"] if p.startswith(needle)]


def test_a_broken_premise_refuses_the_freeze(observed):
    broken = copy.deepcopy(observed)
    broken["premise"] = dict(broken["premise"], problems=["P1: native/continuity differs from the ABI v4 commit"])
    assert any("unchanged-law premise" in p for p in freeze.compare(_as_freeze(observed), broken))
