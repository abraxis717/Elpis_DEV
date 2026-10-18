"""The contextual (WARM) lane has no ECS mutation authority (docs/ARCHITECTURE.md, "Execution lanes").

A cognitive operation records the coordinate of the published context view it consumed, and nothing more: no
retrieval similarity, candidate relationship, binding axiom or context metadata reaches the K1 schedule, the ECS
memory or any learning instruction. Observations enter the memory only as explicit typed values under an explicit
ObservationAuthority (``Runtime.run_learn``); the lane can produce neither. Statically, the lane imports nothing that
could mutate ECS state and the facade passes ``context`` only to the coordinate read; dynamically, a LEARN consuming a
populated view leaves the composite byte-identical to the same LEARN without one. A contextual dynamics law (HCA)
would need its own preregistered specification and qualification; none exists or is implemented.
"""
from __future__ import annotations

import ast
from pathlib import Path

from elpis.runtime import Runtime
from elpis.runtime.context import BINDING, ContextAxiom, ContextLane, ContextLimits, Observation

from .test_composite_runtime import AUTHORITY, _learn, _store, fresh, library, obs  # noqa: F401 (fixture)
from .test_context_lane import _axiom, _hex, _relation, _settle
from .test_k1_recovery import _cfg

SRC = Path(__file__).resolve().parents[2] / "src" / "elpis" / "runtime"


def _imports(path):
    names = set()
    for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
        if isinstance(node, ast.Import):
            names |= {a.name for a in node.names}
        elif isinstance(node, ast.ImportFrom):
            names.add("." * node.level + (node.module or ""))
    return names


def test_the_context_lane_imports_nothing_that_can_mutate_ecs_state():
    imports = _imports(SRC / "context.py")
    # Structure's read-only retrieval modules supply the references; nothing that owns or mutates ECS, continuity
    # or the runtime's cognitive operations is reachable.
    forbidden = ("elpis.ECS", "elpis.continuity", "elpis.runtime", ".core", ".composition", ".cognition",
                 ".observation", ".recovery", "ctypes")
    assert not [n for n in imports if n.startswith(forbidden)], imports
    assert {n for n in imports if n.startswith("elpis.")} <= {"elpis.structure.retrieval.errors",
                                                              "elpis.structure.retrieval.hacf",
                                                              "elpis.structure.retrieval.objects"}, imports
    text = (SRC / "context.py").read_text(encoding="utf-8")
    for name in ("turn_observe", "turn_begin", "run_learn", "MemoryObservation", "ObservationAuthority",
                 "LearnAuthority", "Stimulus"):
        assert name not in text, name


def test_one_context_authority_the_runtime_never_uses_the_proposal_only_model_view_context():
    """The WARM lane (elpis.runtime.context) is the one definition of the active context's identity, version and
    predecessor the runtime uses. elpis.inference.context / global_context (the DSV model-view context snapshot and
    index, ProposalOnly) define their own generation and predecessor for a different object; nothing outside
    elpis.inference may import them, so two context identities can never meet in one operation."""
    root = SRC.parent
    duplicates = ("elpis.inference.context", "elpis.inference.global_context")
    users = []
    for path in root.rglob("*.py"):
        if path.is_relative_to(root / "inference"):
            continue
        for name in _imports(path):
            if name.startswith(duplicates) or (name.startswith(".") and path.parent.name != "inference"
                                               and name.lstrip(".").startswith(("inference.context",
                                                                                "inference.global_context"))):
                users.append((str(path.relative_to(root)), name))
    assert not users, users


def test_the_facade_hands_a_context_only_to_the_coordinate_read():
    """In every cognitive operation of the facade, the ``context`` argument is read exactly once, by
    ``_consumed_context`` (one reference read of the published view's coordinate), and goes nowhere else."""
    tree = ast.parse((SRC / "composition.py").read_text(encoding="utf-8"))
    runtime = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "Runtime")
    checked = 0
    for method in (n for n in runtime.body if isinstance(n, ast.FunctionDef)):
        if "context" not in {a.arg for a in method.args.kwonlyargs + method.args.args}:
            continue
        uses = [n for n in ast.walk(method) if isinstance(n, ast.Name) and n.id == "context"]
        calls = [n for n in ast.walk(method) if isinstance(n, ast.Call) and getattr(n.func, "id", None) ==
                 "_consumed_context" and [getattr(a, "id", None) for a in n.args] == ["context"]]
        assert len(uses) == 1 and len(calls) == 1, method.name
        checked += 1
    assert checked >= 2   # run_query and run_learn at least


def test_a_consumed_context_view_never_reaches_the_k1_schedule_or_the_memory(library, tmp_path):   # noqa: F811
    with ContextLane(ContextLimits(capacity=8)) as lane:
        for observation in (_axiom("a", 5), _axiom("b", 9), _relation("a", "b"),
                            Observation(axiom=ContextAxiom(_hex("binding x"), (), BINDING), required=True)):
            assert _settle(lane, lane.submit(observation)[0]) == "APPLIED"
        view = lane.view()
        assert view.axioms and view.relations
        with fresh(library) as x, fresh(library) as twin, \
                Runtime(_cfg(tmp_path, _store(library, tmp_path / "ck"))) as runtime, \
                Runtime(_cfg(tmp_path, _store(library, tmp_path / "ck-twin"), continuity="twin")) as other:
            runtime.anchor_cognition(x)
            other.anchor_cognition(twin)
            with_context = runtime.run_learn(x, _learn(), context=lane)
            without = other.run_learn(twin, _learn())
            assert with_context.context == view.coordinate and without.context is None
            assert with_context.observed == () and x.inspect()["memory_slots_used"] == 0
            assert x.snapshot() == twin.snapshot()          # K1 and memory: byte-identical
            # With explicit observations, exactly those (and nothing from the view) enter the memory.
            explicit = runtime.run_learn(x, _learn(), context=view, observations=(obs(1, 2, 10),),
                                         observation_authority=AUTHORITY)
            other.run_learn(twin, _learn(), observations=(obs(1, 2, 10),), observation_authority=AUTHORITY)
            assert len(explicit.observed) == 1 and x.inspect()["memory_slots_used"] == 1
            assert x.snapshot() == twin.snapshot()
