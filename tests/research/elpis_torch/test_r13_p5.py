"""Elpis Torch R13 P5: the competence study's task, regions, controls, arms and frozen statistics. RESEARCH_ONLY.

Mechanics only: nothing here trains to competence or reads a confirmation result.
"""
from __future__ import annotations

import random

import pytest

from .conftest import SKIP_REASON

torch = pytest.importorskip("torch", reason=SKIP_REASON)

from research.elpis_torch import composition_tasks as ct  # noqa: E402
from research.elpis_torch.experiments import run_r13_p5 as p5  # noqa: E402
from research.elpis_torch.geodome_cognitive import arms as am  # noqa: E402
from research.elpis_torch.geodome_cognitive import controls as cg  # noqa: E402
from research.elpis_torch.geodome_cognitive import regions as rg  # noqa: E402
from research.elpis_torch.geodome_cognitive.binding import CognitiveSession  # noqa: E402
from research.elpis_torch.geodome_cognitive.kernels import CognitiveConfig  # noqa: E402
from research.elpis_torch.geodome_cognitive.topology import substrate  # noqa: E402

TINY = {"steps": 4, "backbone_steps": 2, "label": "TEST",
        "evaluation": {"primary": {"heldout_combination": {"2": 2, "3": 2}, "heldout_structure": {"2": 2, "3": 2}},
                       "iid": {"1": 1, "2": 1, "3": 1}}}


@pytest.fixture(scope="module")
def sub():
    return substrate()


@pytest.fixture(scope="module")
def world(sub):
    """A barely trained backbone (2 steps), its regions, plans and a few cached views."""
    model = p5.build_backbone(9001)
    p5.train_backbone(model, 9001, 2)
    regions = rg.draw_regions(sub, 9001, 4)
    plans, stats = p5.build_plans(sub, regions, 9001)
    observer = CognitiveSession(model, am.CognitiveKernels(CognitiveConfig(**p5._kernel_config(1)),
                                                           model.config.d_model, model.encoder.width),
                                capture_candidates=8)
    episodes = ct.episodes(9001, "iid", 6)
    views = [p5.view_of(observer, e, f"test:{i}") for i, e in enumerate(episodes)]
    observer.close()
    return {"model": model, "regions": regions, "plans": plans, "stats": stats, "views": views, "episodes": episodes}


# -- the task ---------------------------------------------------------------------------------------------------------

def test_every_episode_satisfies_the_task_invariants():
    for split in ct.SPLITS:
        hops = (2, 3) if split == "heldout_structure" else (1, 2, 3)
        for episode in ct.episodes(5, split, 90, hops):
            answer, required = ct.solve(episode)                # raises on any violated invariant
            assert answer == episode.answer and len(set(required)) == episode.hops
            assert max(required) < ct.QUERY_BLOCK
            query = episode.tokens[ct.QUERY_BLOCK * ct.BLOCK:]
            assert episode.answer not in query                   # the query never holds the answer
            assert query[-3] == ct.entity(episode.path[0]) and query[-2] == ct.HOP[episode.hops]


def test_heldout_pairs_never_reach_training_and_every_entity_is_trained():
    seen, entities = set(), set()
    for episode in ct.episodes(5, "train", 1500):
        for block in episode.fact_blocks:
            a, b = episode.tokens[block * ct.BLOCK + 1], episode.tokens[block * ct.BLOCK + 3]
            seen.add((a - ct.FIRST_ENTITY, b - ct.FIRST_ENTITY))
            entities |= {a, b}
    assert not seen & ct.HELDOUT_PAIRS and len(ct.HELDOUT_PAIRS) == 451
    assert entities == {ct.entity(i) for i in range(ct.ENTITIES)}
    for episode in ct.episodes(5, "heldout_combination", 60):
        assert any(p in ct.HELDOUT_PAIRS for p in zip(episode.path, episode.path[1:]))
    for episode in ct.episodes(5, "heldout_structure", 60, (2, 3)):
        assert episode.structure in ct.HELDOUT_STRUCTURES[episode.hops]
        assert episode.structure not in ct.TRAIN_STRUCTURES[episode.hops]


def test_a_corrupted_episode_is_refused():
    episode = ct.episodes(5, "train", 3, (2,))[0]
    tokens = list(episode.tokens)
    a = episode.path[0]
    spare = next(b for b in ct.CONTENT if b not in episode.fact_blocks)        # a NOTE block becomes a second edge
    tokens[spare * ct.BLOCK:spare * ct.BLOCK + ct.BLOCK] = [ct.FACT, ct.entity(a), ct.REL, ct.entity(episode.path[-1])] \
        + [ct.PAD] * (ct.BLOCK - 4)
    broken = ct.Episode(**{**episode.__dict__, "tokens": tuple(tokens)})
    with pytest.raises(ct.TaskError):
        ct.solve(broken)


# -- regions and controls ----------------------------------------------------------------------------------------------

def test_regions_are_disjoint_connected_and_bound_without_episodes(world, sub):
    regions = world["regions"]
    assert len({v for r in regions for v in r.vertices}) == sum(len(r.vertices) for r in regions)
    for r in regions:
        assert r.stats["connected"] and r.stats["evidence_to_readout"] == [2] * 8 and r.query == r.readout == r.center
        assert all(v in sub.neighbors(u) for u, v in (sub.endpoints(a) for a in r.arcs))     # physical arcs only
    assert rg.draw_regions(sub, 9001, 4) == regions                                      # fixed by the seed


def test_controls_match_the_region_and_say_what_they_are(world):
    physical = world["plans"]["GEODOME"]["main"][0][8]
    n, arcs, _ = cg.from_compiled(physical)
    edges = cg.edges_of(arcs)
    degrees = sorted(sum(v in e for e in edges) for v in range(n))
    for arm in ("COMPUTE_MATCHED", "REWIRED"):
        control = world["plans"][arm]["main"][0][8]
        c_edges = cg.edges_of(cg.from_compiled(control)[1])
        assert control.label == cg.LABEL and control.kind == arm
        assert sorted(sum(v in e for e in c_edges) for v in range(n)) == degrees
        assert len(c_edges) == len(edges) and control.stats["connected"]
        assert control.stats["evidence_distance_mismatch"] == 0
        assert int(control.ports.max()) < 6 and int(control.back_ports.max()) < 6
    rewired = world["plans"]["REWIRED"]["main"][0][8]
    old = [sum(v in e for e in edges) for v in range(n)]
    assert [sum(v in e for e in cg.edges_of(cg.from_compiled(rewired)[1])) for v in range(n)] == old
    star = world["plans"]["FLAT_RECURRENT"]["main"][0][8]
    assert star.vertices == 9 and int(star.ports.abs().sum()) == 0 and star.incoming.tolist() == [True] + [False] * 8


# -- arms ------------------------------------------------------------------------------------------------------------

def test_every_arm_starts_exactly_at_the_backbone_and_the_relabel_is_the_same_computation(world):
    model, views = world["model"], world["views"]
    out = {}
    for name in am.TRAINABLE:
        arm = am.Arm(name, model, CognitiveConfig(**p5._kernel_config(7)), world["plans"][name])
        with torch.no_grad():
            for view in views[:3]:
                source = "oracle" if name == "DIAGNOSTIC_ORACLE" else "deployable"
                assert torch.equal(arm(view, 0, source=source), view.observation.logits), name   # gate 0
            with torch.no_grad():
                arm.kernels.gate.fill_(0.5)
            out[name] = arm(views[0], 0)
        arm.session.close()
    assert torch.allclose(out["GEODOME"], out["RELABEL_ISOMORPHIC"], atol=1e-6, rtol=0)
    assert not torch.allclose(out["GEODOME"], out["FLAT_RECURRENT"], atol=1e-6, rtol=0)


def test_the_gate_opens_and_every_kernel_then_receives_gradient(world):
    arm = am.Arm("GEODOME", world["model"], CognitiveConfig(**p5._kernel_config(7)), world["plans"]["GEODOME"])
    record = p5.train_arm(arm, world["views"], 3, started=__import__("time").perf_counter(), log_steps=[0, 1, 2])
    first, second = record["log"][0], record["log"][1]
    assert first["gate"] == 0.0 and first["grad_norm"]["gate"] > 0
    assert all(first["grad_norm"][g] == 0.0 for g in ("ingress", "message", "update", "readout", "inject"))
    assert second["gate"] != 0.0
    assert all(second["grad_norm"][g] > 0 for g in ("ingress", "message", "update", "readout", "inject"))
    arm.session.close()


def test_equal_budget_is_parameter_matched_and_has_no_graph_kernels(world):
    graph = am.Arm("GEODOME", world["model"], CognitiveConfig(**p5._kernel_config(7)), world["plans"]["GEODOME"])
    direct = am.Arm("EQUAL_BUDGET", world["model"], CognitiveConfig(**p5._kernel_config(7)),
                    world["plans"]["EQUAL_BUDGET"])
    assert am.core_parameters(32, 64) == 9120 and am.reader_hidden(32, 64) == 72
    assert abs(am.parameter_count(direct.trainable()) - am.parameter_count(graph.trainable())) <= 16
    names = {id(p) for p in direct.trainable()}
    assert not any(id(p) in names for n, p in direct.kernels.named_parameters() if n.startswith(am.CORE))
    graph.session.close()
    direct.session.close()


def test_no_state_or_graph_crosses_episodes(world):
    arm = am.Arm("GEODOME", world["model"], CognitiveConfig(**p5._kernel_config(7)), world["plans"]["GEODOME"])
    with torch.no_grad():
        arm.kernels.gate.fill_(0.5)
        alone = arm(world["views"][1], 1)
        arm(world["views"][0], 1)
        after = arm(world["views"][1], 1)
    assert torch.equal(alone, after)
    loss = torch.nn.functional.cross_entropy(arm(world["views"][0], 0), torch.tensor([world["views"][0].answer]))
    loss.backward()
    assert all(v.observation.rows.grad_fn is None and v.deployable.encoded.grad_fn is None for v in world["views"])
    arm.session.close()


def test_views_hold_the_backbones_own_ranking_and_a_gold_diagnostic(world):
    for view, episode in zip(world["views"], world["episodes"]):
        assert len(view.candidates) == 8 and all(b < ct.QUERY_BLOCK for b in view.candidates)
        assert [a.block for a in view.deployable.addresses] == list(view.candidates)
        assert sorted(a.block for a in view.oracle.addresses) == list(episode.fact_blocks)
        assert view.complete == set(episode.required_blocks).issubset(view.candidates)


def test_interventions_run_and_the_disabled_injection_is_native(world):
    arm = am.Arm("GEODOME", world["model"], CognitiveConfig(**p5._kernel_config(7)), world["plans"]["GEODOME"])
    out = p5._interventions(arm, world["views"], lambda i: i % 3)
    assert out["disabled_injection"] == {"bit_exact_native": True, "episodes": len(world["views"])}
    assert set(out) >= {"full", "evidence_removal", "evidence_swap", "reduced_depth", "interrupted_communication",
                        "intermediate_state_erasure", "port_permutation"}
    assert all(world["plans"]["GEODOME"]["cut"][r][8].arc_ids.numel() < len(world["regions"][r].arcs)
               for r in range(4))
    arm.session.close()


def test_a_seed_runs_end_to_end_at_tiny_size():
    record = p5.run_seed(9001, TINY, plan_digest=None)
    assert set(record["arms"]) == set(am.TRAINABLE)
    assert record["interventions"]["disabled_injection"]["bit_exact_native"]
    assert record["episodes"] == {"train": 4, "primary": 8, "iid": 3}
    assert all(a["training"]["status"] == "COMPLETED" for a in record["arms"].values())


# -- the frozen statistics -------------------------------------------------------------------------------------------

def test_the_hierarchical_bootstrap_and_the_comparison_rule():
    stats = p5.PLAN["statistics"]
    rng = random.Random(3)
    strong = [[1 if rng.random() < 0.3 else 0 for _ in range(200)] for _ in range(5)]
    null = [[rng.choice((-1, 0, 0, 1)) for _ in range(200)] for _ in range(5)]
    win = p5.compare(strong, m=3, margin=0.05, stats=stats)
    assert win["pass"] and win["lower_bound"] > 0 and win["positive_seeds"] == 5
    lose = p5.compare(null, m=3, margin=0.05, stats=stats)
    assert not lose["pass"]
    boot = p5.hierarchical_bootstrap(strong, 2000, 1)
    assert boot.shape == (2000,) and float(boot.min()) > 0


def _records(gains: dict, recall: float = 1.0, n: int = 100) -> list:
    """Synthetic confirmation records: per arm, a primary accuracy (NATIVE at 0.1)."""
    out = []
    for seed in p5.PLAN["seeds"]["confirmation"]:
        rng = random.Random(seed)

        def correct(p):
            return "".join("1" if rng.random() < p else "0" for _ in range(n))
        base = {name: correct(gains.get(name, 0.1)) for name in am.TRAINABLE if gains.get(name) != "same"}
        base["RELABEL_ISOMORPHIC"] = base.get("RELABEL_ISOMORPHIC", base["GEODOME"])   # "same": identical outcomes
        record = {"seed": seed, "arms": {name: {"primary": {"correct": base[name]}} for name in am.TRAINABLE},
                  "native": {"primary": {"correct": correct(0.1)}}, "recall": {"primary": {"complete_set": recall}}}
        cut = correct(0.1)
        record["interventions"] = {name: {"correct": cut} for name in p5.PLAN["statistics"]["families"]["interventions"]}
        record["interventions"]["full"] = {"correct": base["GEODOME"]}
        record["interventions"]["disabled_injection"] = {"bit_exact_native": True}
        out.append(record)
    return out


def test_the_frozen_reading_rule():
    relabel = {"RELABEL_ISOMORPHIC": "same"}
    assert p5.reading(_records({"GEODOME": 0.9, **relabel}))["reading"] == "GEOMETRY_SPECIFIC_GAIN"
    assert p5.reading(_records({"GEODOME": 0.9, **relabel, "COMPUTE_MATCHED": 0.9, "REWIRED": 0.9}))["reading"] \
        == "SUBSTRATE_AGNOSTIC_GAIN"
    assert p5.reading(_records({}, recall=0.5))["reading"] == "RETRIEVAL_LIMITED"
    assert p5.reading(_records({}))["reading"] == "NOT_ESTABLISHED"
    unequal = p5.reading(_records({"GEODOME": 0.9, "RELABEL_ISOMORPHIC": 0.1}))     # a label artifact
    assert not unequal["relabel_ok"] and unequal["reading"] != "GEOMETRY_SPECIFIC_GAIN"
