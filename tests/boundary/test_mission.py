"""Mission gate (docs/ELPIS_MISSION.md): DSV4 communicates, ECS computes, FMS materializes.

Mechanics tests prove modules meet their own specifications. These tests
prove the canonical architecture is the mission's topology, and that the gate
rejects the ECS-as-DSV-sidecar topology of f4e1f05/75313fb.
"""
from __future__ import annotations

import json
import shutil
import subprocess
import sys

import pytest

from . import _mission as M
from ._system import REPO

# Corrections still pending on this branch, keyed by check. Each corrective
# commit deletes the entries it fixes; strict xfail turns a fixed check that is
# still listed here into a failure, so this table cannot go stale.
PENDING = {
}


def pending(key):
    return pytest.mark.xfail(strict=True, reason=PENDING[key]) if key in PENDING else (lambda f: f)


STATIC = [pytest.param(c, marks=pytest.mark.xfail(strict=True, reason=PENDING[c])) if c in PENDING else c
          for c in sorted(M.STATIC_CHECKS)]


@pytest.mark.parametrize("check", STATIC)
def test_canonical_tree_passes_the_mission_gate(check):
    found = M.STATIC_CHECKS[check](REPO)
    assert not found, f"{check}: " + json.dumps(found, indent=1)


def _write(root, name, text):
    path = root / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def test_gate_rejects_the_sidecar_topology(tmp_path):
    """A minimal tree with the f4e1f05/75313fb shape fails the three topology checks."""
    _write(tmp_path, "src/elpis/__init__.py", "")
    _write(tmp_path, "src/elpis/runtime/__init__.py", "")
    _write(tmp_path, "src/elpis/runtime/world_model.py",
           "from elpis.ECS.native import WorldState\n"
           "from elpis.inference.conditioning import TurnConditioning, TurnObservation\n"
           "class WorldModelLoop:\n    pass\n")
    _write(tmp_path, "src/elpis/runtime/composition.py",
           "def run_principal(engine):\n    from elpis.inference.principal import PrincipalEngine\n")
    _write(tmp_path, "src/elpis/inference/__init__.py", "")
    _write(tmp_path, "src/elpis/inference/conditioning.py", "class TurnConditioning:\n    pass\n")
    _write(tmp_path, "src/elpis/inference/drivers/dsv41/target.py",
           "class ConditioningProjection:\n    pass\n# embedded = embed[token] + state.conditioning\n")
    _write(tmp_path, "native/inference/include/elpis/dsv41_stream.h",
           "enum { ELPIS_DSV41_STREAM_FEATURE_CONDITIONING = 1u << 2 };\n")
    found = M.violations(tmp_path)
    assert {"sidecar_conditioning", "runtime_model_execution", "canonical_tower"} <= set(found), found


@pytest.mark.parametrize("commit", ["f4e1f059987bd7392562a6eaaf03e6711a02f600",
                                    "75313fb7828ef04779deec1936fe124e2cfa1349"])
def test_gate_rejects_the_forensic_drift_commits(commit, tmp_path):
    """The gate would have rejected the sidecar commits as canonical integration."""
    git = shutil.which("git")
    if git is None or subprocess.run([git, "cat-file", "-e", commit + "^{commit}"], cwd=REPO,
                                     capture_output=True).returncode != 0:
        pytest.skip(f"forensic commit {commit[:7]} not in this clone (shallow checkout)")
    archive = subprocess.run([git, "archive", commit, "src", "native"], cwd=REPO, capture_output=True, check=True)
    subprocess.run(["tar", "-x", "-C", str(tmp_path)], input=archive.stdout, check=True)
    found = M.violations(tmp_path)
    assert {"sidecar_conditioning", "canonical_tower"} <= set(found), found
    if commit.startswith("75313fb"):
        assert "runtime_model_execution" in found


_FAIL_CLOSED_PROBE = r"""
import json, sys
from elpis.runtime.cognition import CODEC_UNQUALIFIED, run_turn
from elpis.runtime.composition import CompositionError
try:
    run_turn(None, "Hello, Elpis.", tokenizer=None)
except CompositionError as exc:
    code = exc.code
else:
    code = None
model = sorted(m for m in sys.modules if m.startswith(("elpis.inference.", "research"))
               and not m.startswith(tuple(sys.argv[1:])))
print(json.dumps({"code": code, "expected": CODEC_UNQUALIFIED, "model_modules": model}))
"""


@pending("fail_closed_turn")
def test_canonical_text_turn_fails_closed_without_a_qualified_codec():
    """No qualified ECS<->DSV codec exists: text generation refuses, and no DSV model machinery loads."""
    result = subprocess.run([sys.executable, "-c", _FAIL_CLOSED_PROBE, *M.CODEC_MODULES],
                            capture_output=True, text=True, cwd=REPO,
                            env={"PYTHONPATH": str(REPO / "src"), "PYTHONDONTWRITEBYTECODE": "1"})
    assert result.returncode == 0, result.stderr
    report = json.loads(result.stdout)
    assert report["code"] == report["expected"] == "ECS_CODEC_UNQUALIFIED"
    assert report["model_modules"] == [], report["model_modules"]


_RUNTIME_CLOSURE_PROBE = r"""
import importlib, json, pkgutil, sys
import elpis.runtime as root
for info in pkgutil.walk_packages(root.__path__, "elpis.runtime."):
    importlib.import_module(info.name)
print(json.dumps(sorted(sys.modules)))
"""


@pending("runtime_closure")
def test_runtime_import_closure_loads_no_dsv_model_machinery():
    """Importing every runtime module reaches elpis.inference only through the codec modules."""
    result = subprocess.run([sys.executable, "-c", _RUNTIME_CLOSURE_PROBE], capture_output=True, text=True,
                            cwd=REPO, env={"PYTHONPATH": str(REPO / "src"), "PYTHONDONTWRITEBYTECODE": "1"})
    assert result.returncode == 0, result.stderr
    loaded = json.loads(result.stdout)
    model = [m for m in loaded if m.startswith("elpis.inference.")
             and not any(m == c or m.startswith(c + ".") for c in M.CODEC_MODULES)]
    assert not model, model
    assert not [m for m in loaded if m == "research" or m.startswith("research.")]


# Runtime operations. A new operation is an architectural decision: adding one
# (for example a model decode or a model text path) must update this list,
# which is reviewed against docs/ELPIS_MISSION.md.
#
# ``anchor_cognition`` is the explicit, one-time continuity bootstrap of the
# canonical K1 turn: it reads the caller's K1 retained-state identity and
# publishes it as the expected lineage identity. It executes no model, mutates
# no K1 state and is never invoked implicitly (reconciliation never anchors).
# ``evolution_authority`` reads the current evolution authority an assertion
# must be bound to; it writes nothing.
RUNTIME_OPERATIONS = {"open", "close", "run_ingress", "admit_retrieval", "publish_canonical",
                      "evolution_authority", "evolve", "admit_context", "run_turn", "anchor_cognition"}


def test_runtime_composes_no_model_operation():
    import elpis.runtime.composition as composition
    public = {n for n in vars(composition.Runtime) if not n.startswith("_")}
    assert public == RUNTIME_OPERATIONS, sorted(public ^ RUNTIME_OPERATIONS)


def test_the_cognitive_turn_is_codec_then_ecs_then_codec():
    """The canonical turn module depends on ECS and on no inference module at all."""
    path = REPO / "src" / "elpis" / "runtime" / "cognition.py"
    names = M.imports_of(REPO, path)
    assert any(M._under(n, "elpis.ECS") for n in names), names
    assert not [n for n in names if M._under(n, "elpis.inference") or M._under(n, "research")], names


# Descriptions of the sidecar topology or of a DSV model as the cognitive
# center. The authority and the design documents must not reintroduce them
# (ELPIS_MISSION.md names them only to prohibit them).
DRIFT_PHRASES = ("turn conditioning", "worldmodelloop", "drivemap", "conditioningprojection",
                 "feature_conditioning", "principal cognitive", "cognitive center", "cognitive centre",
                 "below every model", "completed inference", "runtime.run_text", "runtime.run_principal")


@pytest.mark.parametrize("name", ["ELPIS_SYSTEM.json", "README.md", "docs/ARCHITECTURE.md",
                                  "native/ECS/README.md", "docs/NONCLAIMS.md",
                                  "docs/inference/DSV41_TEXT_BOUNDARY.md"])
def test_authority_and_design_docs_describe_the_corrected_architecture(name):
    text = (REPO / name).read_text(encoding="utf-8").lower()
    found = [p for p in DRIFT_PHRASES if p in text]
    assert not found, (name, found)


def test_system_authority_encodes_the_mission():
    system = json.loads((REPO / "ELPIS_SYSTEM.json").read_text(encoding="utf-8"))
    assert system["mission"]["authority"] == "docs/ELPIS_MISSION.md"
    assert system["mission"]["gate"] == "tests/boundary/test_mission.py"
    subs = {s["id"]: s for s in system["subsystems"]}
    assert tuple(subs["inference"]["canonical_modules"]) == M.CODEC_MODULES
    assert subs["ECS"]["depends_on"] == ["substrate"] and not subs["ECS"].get("uses_numpy")
    assert {"ECS", "continuity", "inference", "substrate"} <= set(subs["runtime"]["depends_on"])
    assert not subs["continuity"]["depends_on"] and "continuity" not in subs["ECS"]["depends_on"]
    assert "ECS" not in subs["inference"]["depends_on"] and "continuity" not in subs["inference"]["depends_on"]
    assert not [t for t in subs["inference"]["native_targets"] if "dsv41" in t]
    assert set(system["research"]["native_targets"]) == {"elpis_dsv41_native", "elpis_dsv41_clock",
                                                         "elpis_dsv41_materializer"}
