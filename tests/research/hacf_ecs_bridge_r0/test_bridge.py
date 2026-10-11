"""HACF -> ECS research bridge infrastructure (research/hacf_ecs_bridge_r0). RESEARCH_ONLY; NO_CLAIM.

Tiny synthetic fixtures only. The bridge observes validated structural evidence through an identified map and asks
a K1 state a read-only QUERY; every degraded evidence condition is refused or recorded; nothing learns, writes back
or is promoted to authority.
"""
from __future__ import annotations

import ast
from dataclasses import replace
from pathlib import Path
import struct

import numpy as np
import pytest

from elpis.ECS.k1 import K1Library, K1State
from elpis.runtime.cognition import QueryRequest, run_query
from elpis.runtime.composition import CompositionError

from research.hacf_ecs_bridge_r0 import harness as H
from research.hacf_ecs_bridge_r0.bridge import (
    BridgeRefusal,
    ECSObservation,
    StructuralObservationPacket,
    map_identity,
    observe,
    query,
)

from ...conftest import admit_k1_set

REPO = Path(__file__).resolve().parents[3]
BRIDGE = REPO / "research" / "hacf_ecs_bridge_r0"


@pytest.fixture
def state():
    k1 = K1Library(admit_k1_set()[1].lib)
    w = np.random.default_rng(7).normal(0.0, 0.18, size=(6, 36)).reshape(-1)
    with K1State.create(k1, 6, 36, memoryview(np.ascontiguousarray(w))) as s:
        yield s


def _code(call):
    with pytest.raises(BridgeRefusal) as info:
        call()
    return info.value.code


def test_valid_evidence_is_observed_and_queried_read_only(state):
    snapshot = state.snapshot()
    records = H.run(state)
    valid = records["VALID"]
    assert valid["outcome"] == "OBSERVED" and len(valid["values"]) == 3   # one row per primary item
    assert all(r["state_unchanged"] for r in records.values())
    assert state.snapshot() == snapshot                                   # (W, epoch, H, a) byte for byte
    assert H.run(state)["VALID"] == valid                                 # deterministic


def test_every_degraded_evidence_condition_is_refused_or_recorded(state):
    records = H.run(state)
    outcome = {name: (r["outcome"], r.get("code")) for name, r in records.items()}
    assert outcome == {
        "VALID": ("OBSERVED", None),
        "WITHHELD": ("REFUSED", "NO_EVIDENCE"),
        "PERMUTED": ("REFUSED", "EVIDENCE_REFUSED"),
        "PERMUTED_RERANKED": ("OBSERVED", None),
        "STALE": ("REFUSED", "STALE_EVIDENCE"),
        "MISLEADING": ("REFUSED", "EVIDENCE_REFUSED"),
        "MISLEADING_WELL_FORMED": ("OBSERVED", None),
    }
    assert "RANK_ORDER" in records["PERMUTED"]["detail"]
    assert "QUERY_DIGEST_MISMATCH" in records["MISLEADING"]["detail"]
    # Well-formed variants are observed as different observations; whether that is right is not judged here.
    assert len({records[n]["observation"] for n in ("VALID", "PERMUTED_RERANKED", "MISLEADING_WELL_FORMED")}) == 3


def test_the_claimed_bundle_digest_is_not_identity_and_text_never_crosses():
    bundle = H.bundle()
    a = StructuralObservationPacket.from_bundle(bundle, H.CONTEXT)
    b = StructuralObservationPacket.from_bundle(replace(bundle, bundle_digest=H.h("another claim")), H.CONTEXT)
    assert a.digest == b.digest and a.claimed_bundle_digest != b.claimed_bundle_digest
    assert not any(hasattr(item, "text") for item in a.items)
    stale_graph = replace(bundle, graph_snapshot_digest=H.h("an older graph"))
    assert _code(lambda: StructuralObservationPacket.from_bundle(stale_graph, H.CONTEXT)) == "STALE_EVIDENCE"
    other_corpus = replace(bundle, corpus_manifest_digest=H.h("another corpus"))
    assert _code(lambda: StructuralObservationPacket.from_bundle(other_corpus, H.CONTEXT)) == "EVIDENCE_REFUSED"


class _Retuned(H.StructuralFixtureMap):
    def parameter_bytes(self):
        return struct.pack("<3d", 9.0, 9.0, 9.0)


class _Impostor(H.StructuralFixtureMap):   # same map_id, other code
    def rows(self, packet):
        return [[1.0] * 6]


class _Unshaped(H.StructuralFixtureMap):
    def rows(self, packet):
        return [[float("nan")] * 6]


class _Short(H.StructuralFixtureMap):
    def rows(self, packet):
        return [[0.0] * 5]


def test_the_map_is_identified_by_measurement_not_by_its_own_claims(state):
    packet = StructuralObservationPacket.from_bundle(H.bundle(), H.CONTEXT)
    pin = map_identity(H.StructuralFixtureMap())
    assert observe(packet, H.StructuralFixtureMap(), pin).dim == 6
    for impostor in (_Retuned(), _Impostor()):
        assert impostor.map_id == pin.map_id
        assert _code(lambda: observe(packet, impostor, pin)) == "MAP_UNIDENTIFIED"
    assert _code(lambda: observe(packet, H.StructuralFixtureMap(),
                                 replace(pin, implementation_sha256=H.h("forged")))) == "MAP_UNIDENTIFIED"
    for broken in (_Unshaped(), _Short()):
        assert _code(lambda: observe(packet, broken, map_identity(broken))) == "OBSERVATION_SHAPE"
    observation = observe(packet, H.StructuralFixtureMap(), pin)
    narrow = replace(observation, dim=5, rows=tuple(r[:5] for r in observation.rows))
    assert _code(lambda: query(state, narrow)) == "OBSERVATION_SHAPE"


def test_an_observation_is_not_a_codec_and_the_runtime_refuses_it(state):
    from tests.integration._turn_fixtures import ByteTokens
    packet = StructuralObservationPacket.from_bundle(H.bundle(), H.CONTEXT)
    observation = observe(packet, H.StructuralFixtureMap(), map_identity(H.StructuralFixtureMap()))
    assert observation.classification.startswith("RESEARCH_ONLY")
    before = state.snapshot()
    with pytest.raises(CompositionError):
        run_query(state, QueryRequest("x", ByteTokens(), observation))
    assert state.snapshot() == before
    assert isinstance(observation, ECSObservation) and not hasattr(observation, "capabilities")


_FORBIDDEN = {
    # LEARN and every K1 mutation; continuity and runtime authority.
    "learn", "consolidate", "reset", "transaction", "txn_begin", "run_schedule", "run_learn", "run_turn",
    "LearnRequest", "LearnAuthority", "anchor_cognition", "Runtime", "RuntimeCore", "commit", "commit_identity",
    "reserve", "evolve",
    # HACF writeback and semantic publication.
    "project_semantic_edges", "build_semantic_context_epoch", "build_corpus_and_index", "publish_candidate",
    "write_bytes", "write_text", "mkdir", "unlink", "rename", "replace_file",
}


def test_the_bridge_has_no_learn_writeback_or_authority_path():
    for path in sorted(BRIDGE.glob("*.py")):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        names = {n.id for n in ast.walk(tree) if isinstance(n, ast.Name)}
        names |= {n.attr for n in ast.walk(tree) if isinstance(n, ast.Attribute)}
        imported = {a.name for n in ast.walk(tree) if isinstance(n, ast.Import) for a in n.names}
        imported |= {n.module or "" for n in ast.walk(tree) if isinstance(n, ast.ImportFrom) and not n.level}
        assert not names & _FORBIDDEN, (path.name, names & _FORBIDDEN)
        assert all(m.startswith("elpis.structure.retrieval.") or m in {
            "__future__", "dataclasses", "hashlib", "inspect", "json", "math", "struct"} for m in imported), imported
        opens = [n for n in ast.walk(tree) if isinstance(n, ast.Call) and getattr(n.func, "id", None) == "open"]
        assert all(len(c.args) > 1 and getattr(c.args[1], "value", "") == "rb" for c in opens), path.name
    for path in (REPO / "src").rglob("*.py"):
        assert "hacf_ecs_bridge" not in path.read_text(encoding="utf-8"), path
