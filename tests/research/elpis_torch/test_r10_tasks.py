"""Elpis Torch R10: the adaptation-and-retention assay (``adaptation_tasks``). No PyTorch needed.

The assay must be answerable from the episode and from nothing else. These tests check determinism, the answer
positions, the conflicts and rules each family promises, train/test disjointness, the absence of a global static
shortcut, and the oracle ceiling, with an EXECUTABLE oracle that reads only the tokens before each answer (not the
generator's own metadata).
"""
from __future__ import annotations

import random

import pytest

from research.elpis_torch import adaptation_tasks as at

VALUE = {at.tok(c): i for i, c in enumerate(at.VALUES)}
KEY = {at.tok(c): i for i, c in enumerate(at.KEYS)}
REGIME = {at.tok(c) for c in at.REGIMES}


def oracle_predictions(e: at.Episode) -> list:
    """Per probe: the answer an exact-memory reader of the tokens BEFORE the answer can determine, else None.
    Binding families recall the regime's last value for the key; family B infers the regime's offset from any
    earlier example of that regime."""
    out = []
    for p in e.probes:
        context = e.tokens[:p.pos + 1]
        regime, key = context[-2], context[-1]
        bindings, offsets = {}, {}
        for i in range(1, len(context) - 3):
            r, k, v = context[i], context[i + 1], context[i + 2]
            if r in REGIME and k in KEY and v in VALUE and context[i + 3] == at.tok(at.SEP):
                bindings[(r, k)] = v
                offsets[r] = (VALUE[v] - KEY[k]) % at.CYCLE
        if e.family == "B":
            guess = None if regime not in offsets else at.tok(at.VALUES[(KEY[key] + offsets[regime]) % at.CYCLE])
        else:
            guess = bindings.get((regime, key))
        out.append(guess)
    return out


@pytest.fixture(scope="module")
def sets():
    return {v: at.eval_set(v, 40, split_seed=123) for v in at.TRAIN_VARIANTS + at.HELDOUT_VARIANTS}


def test_episodes_are_deterministic_and_seed_dependent():
    a = at.eval_set("A", 3, split_seed=5)
    assert [e.tokens for e in a] == [e.tokens for e in at.eval_set("A", 3, split_seed=5)]
    assert [e.tokens for e in a] != [e.tokens for e in at.eval_set("A", 3, split_seed=6)]
    assert at.train_episode(1, 2, 3).tokens == at.train_episode(1, 2, 3).tokens


def test_every_probe_scores_the_value_after_a_regime_and_key(sets):
    for variant, eps in sets.items():
        for e in eps:
            assert e.tokens[0] == at.BOS
            for p in e.probes:
                assert e.tokens[p.pos + 1] == p.answer and p.answer in VALUE
                assert e.tokens[p.pos] in KEY and e.tokens[p.pos - 1] in REGIME, variant


def test_family_a_binds_one_key_to_conflicting_values_across_regimes(sets):
    for e in sets["A"]:
        table = {}
        for p in e.probes:
            table.setdefault((p.regime, p.key), set()).add(p.answer)
        assert all(len(v) == 1 for v in table.values())                 # a binding never changes
        by_key = {}
        for (r, k), v in table.items():
            by_key.setdefault(k, {})[r] = next(iter(v))
        assert any(len(set(r.values())) == len(r) > 1 for r in by_key.values())   # the same key conflicts
        for r in {r for r, _ in table}:
            values = [next(iter(v)) for (rr, _), v in table.items() if rr == r]
            assert len(values) == len(set(values))                        # injective per regime


def test_family_b_asks_only_novel_inputs_and_follows_one_shift_per_regime(sets):
    for variant, offsets in (("B", at.TRAIN_OFFSETS), ("B_heldout", at.HELDOUT_OFFSETS)):
        for e in sets[variant]:
            shift, seen = {}, {}
            for p in e.probes:
                c = (VALUE[p.answer] - p.key) % at.CYCLE
                assert shift.setdefault(p.regime, c) == c and c in offsets
                assert p.key not in seen.setdefault(p.regime, set())       # never an input shown before
                seen[p.regime].add(p.key)
            assert len(set(shift.values())) == len(shift)                   # regimes conflict


def test_family_c_probes_need_retention_not_recency(sets):
    for variant in ("C_short", "C_long", "C_gap", "C_gap128", "C_4reg"):
        for e in sets[variant]:
            posts = [p for p in e.probes if p.kind == "post"]
            pres = [p for p in e.probes if p.kind == "pre"]
            assert posts and len(pres) == len(posts)
            for p in posts:
                assert p.latest != p.answer and p.latest in p.conflicts       # the most recent value is wrong
                assert p.regime == 0 and p.k >= 1
                if variant == "C_short":
                    assert p.distance <= at.TaskSpec().length // 4 - 4      # inside a 64-token window
                else:
                    assert p.distance > 64                                    # outside it
            if variant == "C_gap128":
                assert len(e.tokens) == at.TaskSpec().length + at.TaskSpec().c_gap_heldout + 1


def test_the_oracle_ceiling_is_verified_by_an_executable_reader(sets):
    """Wherever the generator says the answer is determined (k >= 1), an exact reader of the earlier tokens
    predicts it; it never predicts before evidence exists (k = 0 is unknowable)."""
    for variant, eps in sets.items():
        determined = correct = 0
        for e in eps:
            for p, guess in zip(e.probes, oracle_predictions(e)):
                if p.k >= 1:
                    determined += 1
                    correct += guess == p.answer
                    assert at.oracle_correct(p) == 1.0
                else:
                    assert guess is None
        assert determined and correct == determined, variant


def test_shuffled_support_makes_answers_unknowable(sets):
    rng = random.Random(0)
    hits = total = 0
    for e in sets["A"] + sets["B"]:
        s = at.with_shuffled_targets(e, rng)
        for p, guess in zip(s.probes, oracle_predictions(s)):
            if p.k >= 1:
                total += 1
                hits += guess == p.answer
    assert total and hits / total < 0.15                                  # chance is 0.05


def test_diagnostic_transforms_keep_answers_and_lengths(sets):
    rng = random.Random(1)
    for e in sets["C_long"]:
        for d in (at.without_regime_identity(e), at.with_irrelevant_demonstrations(e, rng),
                  at.with_conflicting_recent(e)):
            assert len(d.tokens) == len(e.tokens)
            assert all(d.tokens[p.pos + 1] == p.answer for p in d.probes)
        trap = at.with_conflicting_recent(e)
        for p in trap.probes:
            if p.kind == "post":                       # the example right before is the conflicting one
                assert trap.tokens[p.pos - 5:p.pos - 1] == [trap.tokens[p.pos - 5], at.tok(at.KEYS[p.key]), p.latest,
                                                             at.tok(at.SEP)]
                assert trap.tokens[p.pos - 5] != trap.tokens[p.pos - 1]        # another regime's symbol
        assert at.tok(at.NO_REGIME) in at.without_regime_identity(e).tokens


def test_no_global_static_mapping_predicts_the_answers():
    train = [at.train_episode(seed, step, row) for seed in (0, 1) for step in range(40) for row in range(16)]
    test = [e for v in at.TRAIN_VARIANTS for e in at.eval_set(v, 60, split_seed=7001)]
    base = at.static_baseline(train, test)
    assert base["key_only"] < 0.08 and base["regime_and_key"] < 0.08     # chance 0.05


def test_training_never_sees_held_out_variants_or_evaluation_episodes():
    """Disjoint by construction (reserved signatures), not by luck: family B's rule space is small enough that
    unreserved random draws did repeat across splits (the R10 pilot counted 5 in 24,000 training episodes)."""
    assert not set(at.TRAIN_MIX) & set(at.HELDOUT_VARIANTS)
    assert all(at.reserved(e.signature) for v in at.TRAIN_VARIANTS for e in at.eval_set(v, 32, split_seed=3))
    assert not any(at.reserved(at.train_episode(0, step, row).signature) for step in range(20) for row in range(16))
    train = {at.train_episode(seed, step, row).signature for seed in range(3) for step in range(30) for row in range(16)}
    for v in at.TRAIN_VARIANTS + at.HELDOUT_VARIANTS:
        assert not train & {e.signature for e in at.eval_set(v, 64, split_seed=7001)}


def test_the_spec_digest_names_every_generator_constant():
    spec = at.TaskSpec()
    assert spec.digest() == at.TaskSpec().digest()
    assert spec.digest() != at.TaskSpec(a_keys=5).digest()
