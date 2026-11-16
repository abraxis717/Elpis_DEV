"""Offline training: the models really learn, held-out performance is reported apart, and runs reproduce."""
from __future__ import annotations

import hashlib
import math

import pytest

from .conftest import CHANNELS, SKIP_REASON, TRAIN, make_model

torch = pytest.importorskip("torch", reason=SKIP_REASON)

from research.elpis_torch.data import DOUBLE, HELDOUT, SINGLE, build_splits, generate, make_world, rewire  # noqa: E402
from research.elpis_torch.model import parameter_digest  # noqa: E402
from research.elpis_torch.training import (TrainConfig, evaluate, freeze, load_checkpoint, rollout,  # noqa: E402
                                           save_checkpoint, train)

NAMES = tuple(CHANNELS)
CHANCE = 1 / 6          # six states
# Held-out single-step accuracy after the fixture's 500 steps. A and B reach 1.00 here. C, started at gate 0
# (exactly the FFN block), is measurably slower in this small regime (0.57 at 500 steps; it reaches A's speed
# when started at gate 0.1): recorded in docs/research/elpis_torch/EVIDENCE_R0.md, not tuned away.
MINIMUM_HELD_OUT_ACCURACY = {"A": 0.9, "B": 0.9, "C": 2.5 * CHANCE}


# -- the data ----------------------------------------------------------------------------------------------

def test_the_split_is_disjoint_and_the_compositional_set_is_really_held_out(world, splits):
    report = splits.report
    assert report["train_val_overlap"] == 0 and report["train_val_sequences_containing_a_heldout_triple"] == 0
    assert report["comp_heldout_targets"] > 0 and 0 < report["heldout_triples"] < report["all_double_triples"]
    train_rows = {tuple(r.tolist()) for r in splits.train.tokens}
    assert not train_rows & {tuple(r.tolist()) for r in splits.val.tokens}
    assert not train_rows & {tuple(r.tolist()) for r in splits.comp.tokens}
    assert int((splits.train.kind == HELDOUT).sum()) == 0 and int((splits.val.kind == HELDOUT).sum()) == 0
    assert int((splits.comp.kind == DOUBLE).sum()) == 0
    # Every state target is what the world's table says, single or composed.
    for batch in (splits.train, splits.comp):
        rows, cols = torch.nonzero(batch.kind != 0, as_tuple=True)
        for r, c in list(zip(rows.tolist(), cols.tolist()))[:400]:
            expected = world.step(int(batch.state[r, c]), int(batch.op[r, c]))
            assert int(batch.tokens[r, c]) == world.state_token(expected)


def test_generation_of_data_is_deterministic_and_rewiring_changes_one_operation(world):
    a, b = generate(world, 8, 32, seed=5), generate(world, 8, 32, seed=5)
    assert torch.equal(a.tokens, b.tokens) and not torch.equal(a.tokens, generate(world, 8, 32, seed=6).tokens)
    assert make_world(6, 3, seed=1) == world != make_world(6, 3, seed=2)
    other = rewire(world, op=2, seed=0)
    assert other.changed_from(world) == frozenset((s, 2) for s in range(world.states))
    assert other.table[0] == world.table[0] and sorted(other.table[2]) == list(range(world.states))
    with pytest.raises(RuntimeError):
        build_splits(world, train=64, val=64, comp=8, length=4, seed=0)      # too few distinct sequences


# -- learning ----------------------------------------------------------------------------------------------

@pytest.mark.parametrize("name", NAMES)
def test_training_reduces_the_loss_and_generalizes_to_unseen_trajectories(name, trained, splits):
    model, report, before = trained[name]
    after = report.history[-1]["val"]
    assert report.status == "COMPLETED" and report.anomalies == [] and report.steps_done == TRAIN["steps"]
    assert report.tokens_seen == TRAIN["steps"] * TRAIN["batch_size"] * 64
    # Training loss: from ~log(259) to well under a third of it.
    assert report.first_loss > 5.0 and report.final_loss < 0.3 * report.first_loss
    # Held-out validation (sequences absent from training), reported separately from training.
    assert after["loss"] < 0.3 * before["loss"]
    assert before["single"]["accuracy"] < 2 * CHANCE
    assert after["single"]["loss"] < 0.3 * before["single"]["loss"]
    assert after["single"]["accuracy"] > MINIMUM_HELD_OUT_ACCURACY[name]
    assert math.isfinite(evaluate(model, splits.comp)["heldout"]["loss"])


@pytest.mark.parametrize("name", ["B", "C"])
def test_cgc_gradients_are_finite_and_nonzero_and_cgc_weights_move(name, trained):
    model, report, _ = trained[name]
    for group in ("cgc.core", "cgc.proj"):
        assert math.isfinite(report.grad_norm[group]["max"]) and report.grad_norm[group]["max"] > 0
        assert report.update_norm[group] > 0.01
    if name == "B":
        assert report.grad_norm["cgc.core"]["min"] > 0           # C's first step has gate 0: no CGC gradient yet
    else:
        assert report.grad_norm["cgc.core"]["min"] == 0.0 and report.update_norm["cgc_gate"] > 0
        assert all(float(b.cgc_gate) != 0.0 for b in model.blocks)
    fresh = make_model(name)
    for (key, trained_p), (_, initial_p) in zip(model.named_parameters(), fresh.named_parameters()):
        if "cgc.core" in key:
            assert not torch.equal(trained_p, initial_p), key


def test_free_running_generation_through_the_cache_follows_the_world(trained, splits):
    model, _, _ = trained["B"]
    result = rollout(model, splits.val)
    assert result["states"] == int((splits.val.kind != 0).sum()) and result["trajectories"] == 128
    assert result["state_accuracy"] > 2 * CHANCE
    untrained = rollout(freeze(make_model("B")), splits.val)
    assert untrained["state_accuracy"] < result["state_accuracy"] and untrained["exact_trajectories"] == 0.0


def test_training_reproduces_bit_for_bit_from_its_seeds(trained, splits):
    model, report, _ = trained["B"]
    again = make_model("B")
    second = train(again, splits.train, TrainConfig(**TRAIN), val=splits.val)
    assert parameter_digest(again) == parameter_digest(model)
    assert second.final_loss == report.final_loss and second.history[-1]["val"] == report.history[-1]["val"]
    other = make_model("B")
    train(other, splits.train, TrainConfig(**{**TRAIN, "steps": 20, "seed": 1}))
    short = make_model("B")
    train(short, splits.train, TrainConfig(**{**TRAIN, "steps": 20}))
    assert parameter_digest(other) != parameter_digest(short)


def test_a_non_finite_loss_stops_training_and_is_reported(splits):
    model = make_model("B", cgc_preact_bound=None)
    with torch.no_grad():
        model.blocks[0].cgc.core.weight[0, 0, 0] = float("nan")
    before = parameter_digest(model)
    report = train(model, splits.train, TrainConfig(**{**TRAIN, "steps": 5}))
    assert report.status == "DIVERGED" and report.steps_done == 0 and "non-finite" in report.anomalies[0]
    assert parameter_digest(model) == before            # nothing was applied


@pytest.mark.parametrize("bad", [dict(steps=0), dict(steps=1, lr=0.0), dict(steps=1, batch_size=0),
                                 dict(steps=1, min_lr_fraction=2.0), dict(steps=1, grad_clip=float("inf"))])
def test_invalid_training_settings_are_refused(bad):
    with pytest.raises(ValueError):
        TrainConfig(**bad)


def test_the_learning_rate_schedule_warms_up_then_decays():
    config = TrainConfig(steps=100, lr=1e-2, warmup_steps=10, min_lr_fraction=0.1)
    rates = [config.lr_at(s) for s in range(100)]
    assert rates[0] == pytest.approx(1e-3) and max(rates) == pytest.approx(1e-2)
    assert rates[10:] == sorted(rates[10:], reverse=True) and rates[-1] >= 1e-3


# -- explicit offline artifacts ----------------------------------------------------------------------------

def test_a_checkpoint_is_written_only_on_request_and_reloads_to_the_same_frozen_model(trained, splits, tmp_path):
    model, _, _ = trained["C"]
    assert list(tmp_path.iterdir()) == []               # training wrote nothing here or anywhere it was told to
    path = tmp_path / "elpis-torch-r0.checkpoint"
    digest = save_checkpoint(model, path)
    assert digest == parameter_digest(model)
    file_digest = hashlib.sha256(path.read_bytes()).hexdigest()
    first, second = load_checkpoint(path), load_checkpoint(path)
    assert first.config == model.config and parameter_digest(first) == parameter_digest(second) == digest
    assert not first.training and not any(p.requires_grad for p in first.parameters())
    tokens = splits.val.tokens[:4]
    assert torch.equal(first(tokens), model(tokens)) and torch.equal(first(tokens), second(tokens))
    assert hashlib.sha256(path.read_bytes()).hexdigest() == file_digest      # loading is read-only
    bogus = tmp_path / "bogus"
    torch.save({"format": "something else"}, bogus)
    with pytest.raises(ValueError):
        load_checkpoint(bogus)
