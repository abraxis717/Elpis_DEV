"""R6: R5 reconciliation (evaluation-only evidence-slot overrides, checkpoint reproducibility), the evidence-aware
context session, sparse FFN+CGC experts and their integration with R5, autoregressive training, scaling."""
from __future__ import annotations

import argparse

import pytest

from .conftest import SKIP_REASON

torch = pytest.importorskip("torch", reason=SKIP_REASON)

from research.elpis_torch import evidence_tasks as et  # noqa: E402
from research.elpis_torch.config import ConfigError, ModelConfig  # noqa: E402
from research.elpis_torch.ersa_model import ErsaConfig  # noqa: E402
from research.elpis_torch.evidence_model import EvidenceConfig, EvidenceLM, evaluation_state_dict  # noqa: E402
from research.elpis_torch.model import parameter_digest  # noqa: E402

CGC = dict(channel="cgc", cgc_bottleneck=8, cgc_channels=24, cgc_units=4)


def tiny_evidence(chunk=1024, dtype="float64", **changes):
    widths = ModelConfig(vocab_size=et.VOCAB, d_model=32, n_layers=2, n_heads=2, n_kv_heads=1, max_seq_len=64,
                         **CGC, seed=5, dtype=dtype)
    ersa = ErsaConfig(widths, ersa_heads=2, key_dim=8, value_dim=8, ersa_kernel="chunked", scan_block=16,
                      conv_kernel=4, chunk=chunk)
    evidence = dict(block=16, capacity=256, per_block=2, key_dim=16, coarse_top=6, evidence=3, fine_top=4, overlap=4,
                    encoder_dim=32, fine_dim=8, read_heads=2, read_head_dim=8, query_tile=7)
    evidence.update(changes)
    return EvidenceLM(ersa, EvidenceConfig(**evidence)).eval()


# -- P0: R5 reconciliation --------------------------------------------------------------------------------------

def test_slot_extension_keeps_learned_rows_and_reuses_the_oldest_trained_rank():
    model = tiny_evidence()
    state = model.state_dict()
    out = evaluation_state_dict(state, model.evidence, model.evidence.replace(evidence=4))
    old, new = state["lookup.key_rank"], out["lookup.key_rank"]
    assert new.shape == (5, old.shape[1]) and torch.equal(new[:4], old) and torch.equal(new[4], old[-1])
    assert all(torch.equal(out[k], v) for k, v in state.items() if k != "lookup.key_rank")
    same = evaluation_state_dict(state, model.evidence, model.evidence)
    assert all(torch.equal(same[k], v) for k, v in state.items())


def test_evaluation_overrides_are_validated():
    model = tiny_evidence()
    state = model.state_dict()
    with pytest.raises(ConfigError):                      # fewer slots would discard trained rank rows
        evaluation_state_dict(state, model.evidence, model.evidence.replace(evidence=2))
    with pytest.raises(ConfigError):                      # a trained shape is not an evaluation override
        evaluation_state_dict(state, model.evidence, model.evidence.replace(per_block=4))
    with pytest.raises(ConfigError):                      # the table must match the configuration it came from
        evaluation_state_dict(state, model.evidence.replace(evidence=2, fine_top=4), model.evidence)
    allowed = evaluation_state_dict(state, model.evidence, model.evidence.replace(coarse_top=8, supersession="sum",
                                                                                   capacity=512))
    assert torch.equal(allowed["lookup.key_rank"], state["lookup.key_rank"])


def test_more_slots_change_nothing_while_no_more_evidence_exists():
    """With at most ``trained.evidence`` historical candidates, the wider model selects the same blocks, so every
    logit equals the trained model's (the extension invents no behaviour)."""
    model = tiny_evidence()
    wide = tiny_evidence(evidence=4)
    wide.load_state_dict(evaluation_state_dict(model.state_dict(), model.evidence, wide.evidence))
    tokens = torch.randint(1, et.VOCAB, (2, 63), generator=torch.Generator().manual_seed(1))   # <= 3 complete blocks
    with torch.no_grad():
        assert torch.equal(model(tokens, model.new_state(2)), wide(tokens, wide.new_state(2)))


def _checkpoint(tmp_path, model, record_extra=None):
    from research.elpis_torch.experiments import run_r5
    path = tmp_path / "r5.pt"
    record = {"seed": model.config.seed, "evidence": {}, "parameter_digest": parameter_digest(model),
              **(record_extra or {})}
    torch.save({"record": record, "state_dict": model.state_dict(), "resume": {"phase": "done"}}, path)
    return run_r5, str(path)


def test_run_r5_load_requires_explicit_slot_extension_and_verifies_the_digest(tmp_path):
    from research.elpis_torch.experiments import run_r5
    model = run_r5.build(0)
    run_r5_mod, path = _checkpoint(tmp_path, model)
    loaded, _ = run_r5_mod.load(path)
    assert parameter_digest(loaded) == parameter_digest(model) and loaded.evaluation_override == {}
    with pytest.raises(ValueError):
        run_r5_mod.load(path, evidence=8)
    wide, _ = run_r5_mod.load(path, evidence=8, allow_slot_extension=True)
    assert wide.evaluation_override == {"evidence": 8} and wide.trained_evidence.evidence == 4
    state = {k: v.clone() for k, v in model.state_dict().items()}
    state["embed.weight"] += 1e-3                                                          # corrupt one tensor
    torch.save({"record": {"seed": 0, "evidence": {}, "parameter_digest": parameter_digest(model)},
                "state_dict": state}, tmp_path / "bad.pt")
    with pytest.raises(ValueError):
        run_r5.load(str(tmp_path / "bad.pt"))


def test_r5_training_smoke_is_reproducible_and_its_checkpoint_round_trips(tmp_path):
    from research.elpis_torch.experiments import run_r5
    digests = []
    for name in ("a", "b"):
        args = argparse.Namespace(seed=3, steps=2, eval_every=2, out=str(tmp_path / f"{name}.pt"), smoke=True,
                                  no_curriculum=True, batch=4, max_length=1024, resume=False, init=None, threads=1,
                                  per_block=None, block=None, coarse_top=None, evidence=None, supersession=None)
        result = run_r5.command_train(args)
        digests.append(result["checkpoint"]["parameter_digest"])
    assert digests[0] == digests[1]
    saved = torch.load(tmp_path / "a.pt", weights_only=True)
    repro = saved["record"]["reproducibility"]
    assert repro["schema"] == run_r5.CHECKPOINT_SCHEMA and repro["vocabulary"]["size"] == et.VOCAB
    model, record = run_r5.load(str(tmp_path / "a.pt"))
    assert parameter_digest(model) == record["parameter_digest"] == digests[0]
    assert repro["model_config_digest"] == model.config.digest()


# -- P1: R5 as a first-class context-session model -----------------------------------------------------------------

from research.elpis_torch.context_runtime import ContextSession, ModelRevisionError  # noqa: E402


def _frozen(model):
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    return model


@pytest.fixture(scope="module")
def r5_stream():
    return et.generate_records(et.RecordTask(queries=6, background=20.0), 2, 600, seed=7).tokens


def test_r5_session_equals_direct_execution_across_chunks_calls_steps_and_wraparound(r5_stream):
    model = _frozen(tiny_evidence())
    with torch.no_grad():
        direct = model(r5_stream, model.new_state(2))                       # one call, every position
    small = _frozen(tiny_evidence(chunk=37))
    small.load_state_dict(model.state_dict())
    probes = [0, 15, 16, 200, 255, 256, 257, 400, 511]
    with ContextSession(small, batch=2) as session:
        first = session.prefill(r5_stream[:, :300], probes=[p for p in probes if p < 300])
        second = session.prefill(r5_stream[:, 300:520], probes=[p for p in probes if 300 <= p < 520])
        for position, logits in {**first.probes, **second.probes}.items():
            assert float((logits - direct[:, position]).abs().max()) < 1e-9, position
        for position in range(520, 600):                                   # single-token decoding
            logits = session.step(r5_stream[:, position])
            assert float((logits - direct[:, position]).abs().max()) < 1e-9, position
        assert session.position == 600 and session.state.ring.oldest == 600 - 256      # the ring wrapped
        assert session.counters()["ring_stale_rejections"] == 0


def test_r5_sessions_reset_in_place_and_never_share_state(r5_stream):
    model = _frozen(tiny_evidence(chunk=64))
    with ContextSession(model, batch=2) as a, ContextSession(model, batch=2) as b:
        a.prefill(r5_stream[:, :300])
        b.prefill(r5_stream.flip(1)[:, :300])
        a.reset()
        fresh = a.prefill(r5_stream[:, :128]).last_logits
        with ContextSession(model, batch=2) as c:
            assert torch.equal(fresh, c.prefill(r5_stream[:, :128]).last_logits)
        assert a.state.ring.session != b.state.ring.session and a.state.ring.epoch == 1


def test_r5_session_accounting_is_exact_and_includes_the_ring():
    model = _frozen(tiny_evidence(chunk=64))
    with ContextSession(model, batch=2) as session:
        projected = session.projected_bytes()
        held = session.nbytes()
        assert projected["state"] == held["total"]
        assert projected["state_by_tier"] == {k: v for k, v in held.items() if k != "total"}
        assert {"history_tokens", "index_keys", "index_ids"} <= set(projected["state_by_tier"])
        assert projected["scratch"] > 0 and session.interface.kind == "r5-evidence"


def test_options_that_do_not_apply_are_refused_not_ignored():
    model = _frozen(tiny_evidence(chunk=64))
    with pytest.raises(ValueError):
        ContextSession(model, scorer="reference")
    with pytest.raises(ValueError):
        ContextSession(model, history_device="cpu", index_device="meta")
    from research.elpis_torch.ersa_model import ErsaLM
    core = _frozen(ErsaLM(tiny_evidence().ersa))
    with pytest.raises(ValueError):
        ContextSession(core, cache_bytes=1 << 20)


def test_r5_reread_cache_is_exact_bounded_and_identity_safe(r5_stream):
    model = _frozen(tiny_evidence(chunk=64))
    probes = list(range(64, 600, 7))
    with ContextSession(model, batch=2) as plain, ContextSession(model, batch=2, cache_bytes=1 << 16) as cached:
        reference = plain.prefill(r5_stream, probes=probes).probes
        warm = cached.prefill(r5_stream, probes=probes).probes
        for position in probes:
            assert float((reference[position] - warm[position]).abs().max()) < 1e-10
        stats = cached.counters()
        assert stats["cache_hits"] > 0 and stats["cache_bytes"] <= 1 << 16
        lo = cached.state.ring.live()[0]
        assert all(block >= lo for block in cached.state.cache.entries)      # nothing outside the live window
        assert stats["blocks_reencoded"] < plain.counters()["blocks_reencoded"]
        cached.reset()
        assert not cached.state.cache.entries
        with torch.no_grad():
            model.encoder.delta.weight.add_(1e-3)                             # the model changes under the session
        with pytest.raises(ModelRevisionError):
            cached.prefill(r5_stream[:, :10])
