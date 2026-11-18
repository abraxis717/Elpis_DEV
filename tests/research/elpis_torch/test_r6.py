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


def tiny_evidence(chunk=1024, dtype="float64", sparse=None, **changes):
    widths = ModelConfig(vocab_size=et.VOCAB, d_model=32, n_layers=2, n_heads=2, n_kv_heads=1, max_seq_len=64,
                         **CGC, seed=5, dtype=dtype)
    ersa = ErsaConfig(widths, ersa_heads=2, key_dim=8, value_dim=8, ersa_kernel="chunked", scan_block=16,
                      conv_kernel=4, chunk=chunk, sparse=sparse)
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


# -- P2 / P3: sparse FFN + CGC experts ---------------------------------------------------------------------------

from research.elpis_torch.ersa_model import ErsaLM  # noqa: E402
from research.elpis_torch.sparse_channel import (SparseChannel, SparseConfig, active_parameters,  # noqa: E402
                                                 aux_loss, channel_reports, upcycle)


def core_widths(channel="cgc", dtype="float64", **changes):
    base = dict(vocab_size=97, d_model=32, n_layers=2, n_heads=2, n_kv_heads=1, max_seq_len=64, channel=channel,
                seed=11, dtype=dtype)
    if "cgc" in channel:
        base.update(cgc_bottleneck=8, cgc_channels=24, cgc_units=4)
    if "ffn" in channel:
        base.update(ffn_hidden=48)
    base.update(changes)
    return ModelConfig(**base)


def sparse_core(sparse=None, channel="cgc", dtype="float64", chunk=64):
    return ErsaLM(ErsaConfig(core_widths(channel, dtype), conv_kernel=4, chunk=chunk, sparse=sparse))


def channel_alone(experts=8, top_k=2, dtype=torch.float64, **changes):
    m = core_widths(dtype="float64")
    config = SparseConfig(experts=experts, top_k=top_k, layers=(0,), **changes)
    channel = SparseChannel(m, config, dtype)
    generator = torch.Generator().manual_seed(3)
    with torch.no_grad():
        for parameter in channel.parameters():
            parameter.copy_(torch.randn(parameter.shape, generator=generator, dtype=dtype) * 0.3)
    return channel


def test_routing_ties_resolve_to_the_lower_expert_deterministically():
    channel = channel_alone()
    with torch.no_grad():
        channel.router.weight.zero_()                                   # every probability equal
    u = torch.randn(10, 32, dtype=torch.float64)
    _, probs, index, weight = channel.router(u)
    assert index.tolist() == [[0, 1]] * 10 and torch.allclose(weight, torch.full_like(weight, 0.5))
    assert torch.equal(channel.router(u)[2], index)


@pytest.mark.parametrize("experts,top_k,shape", [(8, 2, (2, 13)), (16, 2, (1, 1)), (4, 3, (3, 7)), (16, 1, (2, 5))])
def test_grouped_dispatch_equals_the_reference_in_value_and_gradient(experts, top_k, shape):
    grouped = channel_alone(experts, top_k, shared=True)
    reference = channel_alone(experts, top_k, shared=True, dispatch="reference")
    reference.load_state_dict(grouped.state_dict())
    u = torch.randn(*shape, 32, dtype=torch.float64)
    a, b = grouped(u), reference(u)
    assert float((a - b).abs().max()) < 1e-12
    a.square().sum().backward()
    b.square().sum().backward()
    for (name, p), q in zip(grouped.named_parameters(), reference.parameters()):
        if p.grad is None:
            assert q.grad is None or float(q.grad.abs().max()) == 0.0, name    # an unselected expert
        else:
            assert float((p.grad - q.grad).abs().max()) < 1e-10, name
    counts = grouped.report["stats"]["counts"]
    assert sum(counts) == shape[0] * shape[1] * top_k                         # no token dropped, none doubled
    strided = torch.randn(32, *shape, dtype=torch.float64).movedim(0, -1)    # a non-contiguous input
    assert float((grouped(strided) - reference(strided)).abs().max()) < 1e-12


def test_every_expert_owns_its_parameters_and_its_cgc_acts_alone():
    channel = channel_alone()
    pointers = [p.data_ptr() for p in channel.parameters()]
    assert len(set(pointers)) == len(pointers)                                # no aliased storage
    u = torch.randn(40, 32, dtype=torch.float64)
    before = channel(u)
    _, _, index, _ = channel.router(u)
    with torch.no_grad():
        channel.experts[0].cgc.core.weight.add_(0.5)                          # expert 0's CGC only
    after = channel(u)
    uses_0 = (index == 0).any(-1)
    changed = (after - before).abs().amax(-1) > 0
    assert torch.equal(changed, uses_0) and bool(uses_0.any()) and bool((~uses_0).any())


def test_gradients_reach_router_experts_local_cgc_shared_ersa_embeddings_and_head():
    model = sparse_core(SparseConfig(experts=4, top_k=2, layers=(0, 1), shared=True))
    tokens = torch.randint(0, 97, (2, 40), generator=torch.Generator().manual_seed(0))
    logits = model(tokens, model.new_state(2))
    torch.nn.functional.cross_entropy(logits[:, :-1].reshape(-1, 97), tokens[:, 1:].reshape(-1)).backward()
    grads = {n: p.grad for n, p in model.named_parameters()}
    nonzero = lambda n: grads[n] is not None and float(grads[n].abs().max()) > 0      # noqa: E731
    for block in (0, 1):
        assert nonzero(f"blocks.{block}.sparse.router.weight")
        active = [e for e, c in enumerate(channel_reports(model)[block]["stats"]["counts"]) if c]
        for e in active:
            for part in ("ffn.gate.weight", "ffn.out.weight", "cgc.core.weight", "cgc.proj_in.weight", "cgc_gate"):
                assert nonzero(f"blocks.{block}.sparse.experts.{e}.{part}"), (block, e, part)
        assert nonzero(f"blocks.{block}.sparse.shared.cgc.core.weight")
    assert nonzero("blocks.0.ersa.qkv.weight" if "blocks.0.ersa.qkv.weight" in grads else
                   next(n for n in grads if n.startswith("blocks.0.ersa.") and n.endswith("weight")))
    assert nonzero("embed.weight")                                            # the tied output head as well


def test_a_zero_gate_is_exactly_the_ffn_and_its_cgc_then_starts_learning():
    channel = channel_alone(gate_init=0.0, kind="ffn+cgc")
    with torch.no_grad():
        for expert in channel.experts:
            expert.cgc_gate.zero_()
    u = torch.randn(30, 32, dtype=torch.float64)
    plain = channel_alone(kind="ffn")
    plain.load_state_dict({k: v for k, v in channel.state_dict().items() if ".cgc" not in k})
    assert float((channel(u) - plain(u)).abs().max()) == 0.0                  # exact control
    optimizer = torch.optim.SGD(channel.parameters(), lr=0.1)
    channel(u).square().sum().backward()
    used = [e for e, c in enumerate(channel.report["stats"]["counts"]) if c]
    assert all(float(channel.experts[e].cgc_gate.grad.abs()) > 0 for e in used)
    assert all(float(channel.experts[e].cgc.core.weight.grad.abs().max()) == 0 for e in used)
    optimizer.step()
    optimizer.zero_grad()
    channel(u).square().sum().backward()
    assert any(float(channel.experts[e].cgc.core.weight.grad.abs().max()) > 0 for e in used)


def test_without_a_sparse_config_the_model_is_the_old_dense_model():
    old = ErsaLM(ErsaConfig(core_widths(), conv_kernel=4, chunk=64))
    new = sparse_core(None)
    empty = sparse_core(SparseConfig(layers=()))
    assert list(old.state_dict()) == list(new.state_dict()) == list(empty.state_dict())
    tokens = torch.randint(0, 97, (2, 30), generator=torch.Generator().manual_seed(1))
    with torch.no_grad():
        reference = old(tokens, old.new_state(2))
        assert torch.equal(reference, new(tokens, new.new_state(2)))
        assert torch.equal(reference, empty(tokens, empty.new_state(2)))


@pytest.mark.parametrize("channel", ["cgc", "ffn", "ffn+cgc"])
def test_upcycling_reproduces_the_dense_model(channel):
    dense = ErsaLM(ErsaConfig(core_widths(channel), conv_kernel=4, chunk=64)).eval()
    if channel == "ffn+cgc":
        with torch.no_grad():
            for block in dense.blocks:
                block.cgc_gate.fill_(0.7)
    kind = "ffn" if channel == "ffn" else "ffn+cgc"
    tokens = torch.randint(0, 97, (2, 30), generator=torch.Generator().manual_seed(2))
    with torch.no_grad():
        reference = dense(tokens, dense.new_state(2))
        clones = upcycle(dense, SparseConfig(experts=4, top_k=2, layers=(0, 1), kind=kind)).eval()
        assert float((clones(tokens, clones.new_state(2)) - reference).abs().max()) < 1e-12
        shared = upcycle(dense, SparseConfig(experts=4, top_k=2, layers=(0, 1), kind=kind, shared=True)).eval()
        assert torch.equal(shared(tokens, shared.new_state(2)), reference)          # bitwise: routed outputs are 0
        for block in shared.blocks:
            block.sparse.routed_enabled = False                                 # the disabled-routing control
        assert torch.equal(shared(tokens, shared.new_state(2)), reference)


def test_routing_statistics_and_auxiliary_losses():
    model = sparse_core(SparseConfig(experts=8, top_k=2, layers=(1,), balance_weight=0.01, z_weight=0.001))
    tokens = torch.randint(0, 97, (3, 50), generator=torch.Generator().manual_seed(4))
    model(tokens, model.new_state(3))
    report = channel_reports(model)[0]
    stats = report["stats"]
    assert sum(stats["counts"]) == 150 * 2 and len(stats["coactivation"]) == 8
    assert [stats["coactivation"][i][i] for i in range(8)] == stats["counts"]
    assert 0 <= stats["router_entropy"] <= stats["router_entropy_max"] + 1e-9
    assert stats["load_imbalance_max_over_mean"] >= 1.0 and 0 <= stats["inactive_fraction"] <= 1
    total = aux_loss(model)
    expected = 0.01 * report["balance_loss"] + 0.001 * report["z_loss"]
    assert torch.isfinite(total) and float((total - expected).abs()) < 1e-12 and total.requires_grad
    counts = active_parameters(model)
    assert counts["active_per_token"] < counts["total"] == counts["resident"]


def test_extreme_routing_stays_finite_and_mixed_precision_routes_in_float32():
    channel = channel_alone(experts=8, top_k=8)
    with torch.no_grad():
        channel.router.weight.mul_(1e4)                                         # saturated router
    u = torch.randn(20, 32, dtype=torch.float64, requires_grad=True)
    out = channel(u)
    out.sum().backward()
    assert bool(torch.isfinite(out).all()) and bool(torch.isfinite(u.grad).all())
    low = channel_alone(dtype=torch.float32)
    with torch.autocast("cpu", dtype=torch.bfloat16):
        result = low(torch.randn(16, 32))
    logits, probs, _, _ = low.router(torch.randn(4, 32))
    assert bool(torch.isfinite(result).all()) and logits.dtype == torch.float32 == probs.dtype


# -- P4: routed experts with the R5 evidence path ----------------------------------------------------------------

EXPERTS = SparseConfig(experts=4, top_k=2, layers=(0, 1), shared=True)


def test_experts_with_r5_are_causal_and_chunk_invariant(r5_stream):
    model = _frozen(tiny_evidence(sparse=EXPERTS))
    changed = r5_stream.clone()
    changed[:, 301:] = torch.randint(1, 30, changed[:, 301:].shape, generator=torch.Generator().manual_seed(0))
    with torch.no_grad():
        whole = model(r5_stream, model.new_state(2))
        assert torch.equal(whole[:, :301], model(changed, model.new_state(2))[:, :301])     # no future leakage
        small = _frozen(tiny_evidence(chunk=29, sparse=EXPERTS))
        small.load_state_dict(model.state_dict())
        state = small.new_state(2)
        parts = torch.cat([small(r5_stream[:, a:a + 29], state) for a in range(0, 600, 29)], dim=1)
    assert float((parts - whole).abs().max()) < 1e-9
    assert state.ring.stats["stale_rejections"] == 0


def test_r5_off_is_the_routed_core_and_no_label_reaches_the_routes(r5_stream):
    model = _frozen(tiny_evidence(sparse=EXPERTS))
    core = _frozen(ErsaLM(model.ersa))
    core.load_state_dict({k: v for k, v in model.state_dict().items() if k in core.state_dict()})
    tokens = r5_stream[:, :200]
    labels = et.EvidenceStreams(r5_stream, torch.zeros_like(r5_stream, dtype=torch.int8),
                                torch.full_like(r5_stream, -1)).query_source()[:, :200]
    with torch.no_grad():
        assert torch.equal(model(tokens, model.new_state(2), mode="off"), core(tokens, core.new_state(2)))
        plain = model(tokens, model.new_state(2))
        routes = [r["stats"]["counts"] for r in channel_reports(model)]
        labelled, _ = model(tokens, model.new_state(2), evidence=labels, diagnostics=True)
        assert torch.equal(plain, labelled) and routes == [r["stats"]["counts"] for r in channel_reports(model)]


def test_expert_selection_adds_no_persistent_state_and_revisions_are_refused(r5_stream):
    model = _frozen(tiny_evidence(chunk=64, sparse=EXPERTS))
    dense = _frozen(tiny_evidence(chunk=64))
    with ContextSession(model, batch=2) as session:
        assert session.nbytes() == ContextSession(dense, batch=2).nbytes()          # experts are weights, not state
        before = session.nbytes()
        session.prefill(r5_stream)
        assert session.nbytes() == before
        with torch.no_grad():
            model.blocks[0].sparse.router.weight.add_(1e-3)
        with pytest.raises(ModelRevisionError):
            session.step(r5_stream[:, 0])


def test_injected_and_exposed_evidence_counts_are_independent_and_all_ranks_train():
    model = tiny_evidence(evidence=8, fine_top=8, coarse_top=8, sparse=EXPERTS, dtype="float32")
    assert model.lookup.key_rank.shape[0] == 9
    stream = et.generate_records(et.RecordTask(queries=6, background=20.0), 2, 600, seed=7)
    logits = model(stream.tokens[:, :-1], model.new_state(2))
    torch.nn.functional.cross_entropy(logits.reshape(-1, logits.shape[-1]), stream.tokens[:, 1:].reshape(-1)).backward()
    grad = model.lookup.key_rank.grad
    assert grad is not None and int((grad.abs().sum(-1) > 0).sum()) >= 6          # ranks beyond 4 are trained
    narrow = tiny_evidence(evidence=4, fine_top=8, coarse_top=8)
    assert narrow.evidence.fine_top == 8 and narrow.lookup.key_rank.shape[0] == 5


# -- P5: autoregressive training of the expertized backbone ------------------------------------------------------

from research.elpis_torch import lm_training as lmt  # noqa: E402
from research.elpis_torch.tokenizer import ByteTokenizer, SubwordTokenizerProtocol, TokenizerProtocol  # noqa: E402
from research.elpis_torch.training import TrainConfig  # noqa: E402

TOK = ByteTokenizer()


def byte_model(sparse=EXPERTS, dtype="float32", chunk=32):
    widths = ModelConfig(vocab_size=259, d_model=32, n_layers=2, n_heads=2, n_kv_heads=1, max_seq_len=64, **CGC,
                         seed=13, dtype=dtype)
    return ErsaLM(ErsaConfig(widths, conv_kernel=4, chunk=chunk, sparse=sparse))


def test_rows_never_cross_documents_and_padding_is_ignored():
    docs = [b"abc", b"defghij"]
    rows = list(lmt.document_rows(docs, TOK, 6))
    assert [r for r, _ in rows][0][:5] == [TOK.BOS, 97, 98, 99, TOK.EOS]           # BOS doc EOS, then padding
    assert all(TOK.EOS not in r[1:-1] or r.index(TOK.EOS) == n for r, n in rows)
    (inputs, targets), = lmt.document_batches(rows[:2], 2, seed=0, ignore_pad=TOK.PAD)
    for i in range(2):
        valid = targets[i] != lmt.IGNORE
        assert torch.equal(targets[i][valid], torch.tensor([r for r, _ in rows[:2]][i])[1:][valid])
        assert bool((torch.tensor([r for r, _ in rows[:2]][i])[1:][~valid] == TOK.PAD).all())


def test_loss_is_normalized_over_valid_targets_and_chunking_does_not_change_it():
    torch.manual_seed(0)
    model = byte_model(sparse=None, dtype="float64", chunk=64)
    inputs = torch.randint(0, 256, (3, 40))
    targets = torch.randint(0, 256, (3, 40))
    targets[1, 30:] = lmt.IGNORE
    logits = model(inputs, model.new_state(3))
    manual = F_ce(logits, targets)
    for chunk in (40, 16, 7):
        loss, info = lmt.batch_loss(model, inputs, targets, lmt.LMConfig(seq=40, chunk=chunk), backward=False)
        assert abs(float(loss) - manual) < 1e-9 and info["targets"] == 3 * 40 - 10


def F_ce(logits, targets):
    return float(torch.nn.functional.cross_entropy(logits.reshape(-1, logits.shape[-1]), targets.reshape(-1),
                                                   ignore_index=lmt.IGNORE))


def test_truncated_backpropagation_detaches_between_chunks_and_full_does_not():
    torch.manual_seed(1)
    inputs = torch.randint(0, 256, (2, 32))
    targets = torch.randint(0, 256, (2, 32))
    grads = {}
    for tbptt in (False, True):
        model = byte_model(sparse=None, dtype="float64")
        lmt.batch_loss(model, inputs, targets, lmt.LMConfig(seq=32, chunk=16, tbptt=tbptt), backward=True)
        grads[tbptt] = model.embed.weight.grad.clone()
    assert float((grads[False] - grads[True]).abs().max()) > 1e-9          # the truncation is real
    single = {}
    for tbptt in (False, True):
        model = byte_model(sparse=None, dtype="float64")
        lmt.batch_loss(model, inputs, targets, lmt.LMConfig(seq=32, chunk=32, tbptt=tbptt), backward=True)
        single[tbptt] = model.embed.weight.grad.clone()
    assert float((single[False] - single[True]).abs().max()) < 1e-12       # one chunk: nothing to truncate


def test_router_auxiliary_losses_are_added_with_their_configured_weights():
    sparse = SparseConfig(experts=4, top_k=2, layers=(1,), balance_weight=0.5, z_weight=0.0)
    model = byte_model(sparse=sparse, dtype="float64")
    inputs, targets = torch.randint(0, 256, (2, 32)), torch.randint(0, 256, (2, 32))
    loss, info = lmt.batch_loss(model, inputs, targets, lmt.LMConfig(seq=32, chunk=32), backward=False)
    assert abs(float(loss) - info["cross_entropy"] - info["aux"]) < 1e-9 and info["aux"] > 0


def test_training_refuses_a_non_finite_loss_and_is_deterministic():
    rows = list(lmt.document_rows([b"hello world " * 20, b"elpis " * 30], TOK, 32))
    def run(poison=False):
        model = byte_model()
        if poison:
            with torch.no_grad():
                model.blocks[0].sparse.experts[0].ffn.up.weight.fill_(float("nan"))
        before = parameter_digest(model)
        result = lmt.train_lm(model, lmt.document_batches(rows, 2, seed=0, ignore_pad=TOK.PAD),
                              TrainConfig(steps=3, batch_size=2, warmup_steps=1), lmt.LMConfig(seq=32, chunk=16))
        return result, before, parameter_digest(model)
    bad, before, after = run(poison=True)
    assert bad["status"] == "DIVERGED" and before == after                  # nothing applied
    first, second = run(), run()
    assert first[0]["status"] == "COMPLETED" and first[2] == second[2] != first[1]


def test_a_small_code_fixture_is_learned_and_generation_mechanics_hold(tmp_path):
    """Trainability on real bytes (this repository's own source), not language competence."""
    from research.elpis_torch.config import ModelConfig as MC
    docs = list(lmt.iter_documents([str(__import__("pathlib").Path(lmt.__file__).parent)], suffixes=(".py",)))
    lm = lmt.LMConfig(seq=64, chunk=32)
    rows = list(lmt.document_rows(docs[2:], TOK, lm.seq))
    val = list(lmt.document_batches(list(lmt.document_rows(docs[:2], TOK, lm.seq)), 8, seed=1, ignore_pad=TOK.PAD))[:3]
    model = byte_model()
    lmt.check_vocabulary(model, TOK)
    start = lmt.evaluate_lm(model, val, lm)["nats_per_token"]
    result = lmt.train_lm(model, lmt.document_batches(rows, 8, seed=0, ignore_pad=TOK.PAD),
                          TrainConfig(steps=40, batch_size=8, lr=3e-3, warmup_steps=5), lm)
    end = lmt.evaluate_lm(model, val, lm)["nats_per_token"]
    assert result["status"] == "COMPLETED" and end < start - 1.5, (start, end)
    prompt = torch.tensor([TOK.encode("def train(model", bos=True)] * 2)
    one = lmt.generate(model, prompt, 12, chunk=8)
    three = lmt.generate(model, prompt, 12, chunk=8, prefill_calls=3)
    assert torch.equal(one, three)                                          # generation after several prefills
    with torch.no_grad():                                                   # teacher-forced equals free-running
        state = model.new_state(2)
        logits = model(torch.cat((prompt, one[:, :-1]), 1), state)
    assert torch.equal(logits[:, prompt.shape[1] - 1:].argmax(-1), one)
    path = tmp_path / "lm.pt"
    digest = lmt.save_lm_checkpoint(model, path, tokenizer=TOK, step=result["steps"], seed=13,
                                    optimizer=result["optimizer"])
    loaded, record = lmt.load_lm_checkpoint(path, tokenizer=TOK)
    assert parameter_digest(loaded) == digest == record["parameter_digest"]
    assert record["architecture"]["sparse_config"]["experts"] == 4 and record["step"] == 40
    assert torch.equal(lmt.generate(loaded, prompt, 12, chunk=8), one)
    with pytest.raises(ValueError):
        lmt.load_lm_checkpoint(path, tokenizer=type("T", (), {"vocab_size": 300})())
    del MC


def test_generation_with_r5_and_experts_and_the_tokenizer_protocols(tmp_path):
    model = tiny_evidence(sparse=EXPERTS, dtype="float32")
    prompt = et.generate_records(et.RecordTask(queries=4), 1, 300, seed=2).tokens
    one = lmt.generate(model, prompt, 10, chunk=64)
    two = lmt.generate(model, prompt, 10, chunk=64, prefill_calls=4)
    assert torch.equal(one, two)
    path = tmp_path / "ev.pt"
    lmt.save_lm_checkpoint(model, path, tokenizer=type("V", (), {"vocab_size": et.VOCAB})(), step=0, seed=5)
    loaded, record = lmt.load_lm_checkpoint(path)
    assert record["architecture"]["family"] == "evidence" and torch.equal(lmt.generate(loaded, prompt, 10, chunk=64), one)
    assert isinstance(TOK, TokenizerProtocol) and isinstance(TOK, SubwordTokenizerProtocol)
    data = "naïve — ünïcode ✓".encode()
    assert TOK.decode_bytes(TOK.encode_bytes(data)) == data
    with pytest.raises(ValueError):
        lmt.check_vocabulary(tiny_evidence(), TOK)                          # synthetic vocabulary != bytes


# -- P6: sparse scaling configurations and context accounting ----------------------------------------------------

def test_the_500m_and_1b_sparse_configurations_account_for_every_parameter():
    from research.elpis_torch.experiments import run_r6
    for name, target in (("r6_500m", 500e6), ("r6_1b", 1e9)):
        model = run_r6.scale_model(name)                                       # meta device: no memory
        counts = active_parameters(model)
        groups = run_r6._group_counts(model)
        assert sum(groups.values()) == counts["total"] == counts["resident"]
        assert abs(counts["total"] - target) / target < 0.06, (name, counts["total"])
        assert counts["active_per_token"] < counts["total"]
        assert {"routers", "routed_experts", "evidence_path", "ersa", "attention", "embeddings_and_head"} <= set(groups)
        assert (groups.get("shared_experts", 0) > 0) == run_r6.SCALES[name][4]
        macs = run_r6.per_token_macs(model, 1 << 20)
        assert macs["coarse_scan"] == 32768 * 4 * 64


def test_context_state_formula_is_exact_and_linear_in_the_ring():
    sizes = {}
    for capacity in (4096, 16384):
        model = tiny_evidence(capacity=capacity, sparse=EXPERTS)
        state = model.new_state(1)
        fixed = model.context_interface().fixed_bytes(1)
        assert fixed == state.nbytes()
        sizes[capacity] = fixed
    grow = {k: sizes[16384][k] - sizes[4096][k] for k in sizes[4096] if k != "total"}
    assert grow["ersa_cells"] == 0 and grow["conv_tail"] == 0                  # the core state is constant
    assert grow["history_tokens"] == (16384 - 4096) * 4                        # exact tokens: 4 B each
    assert grow["index_keys"] == (16384 - 4096) // 16 * 2 * 16 * 4             # per_block x key_dim x float32


def test_the_query_tile_is_an_execution_option_that_changes_no_logit(r5_stream):
    """The scaled configurations bound a read at every position with smaller query tiles; a tile only partitions
    independent queries, so it must change no logit."""
    reference = _frozen(tiny_evidence(sparse=EXPERTS, query_tile=128))
    with torch.no_grad():
        expected = reference(r5_stream, reference.new_state(2))
        for tile in (1, 5, 32):
            model = _frozen(tiny_evidence(sparse=EXPERTS, query_tile=tile))
            model.load_state_dict(reference.state_dict())
            assert float((model(r5_stream, model.new_state(2)) - expected).abs().max()) < 1e-12, tile


def test_the_scaled_configurations_stay_within_the_context_target_even_when_every_position_reads():
    """P6 budget per stream (target 1 GiB): EXACT state with a bfloat16 index plus the PROJECTED scratch of a decode
    step and of a 512-token prefill that reads at every position, at every reported length."""
    from research.elpis_torch.experiments import run_r6
    for name in run_r6.SCALES:
        for length in run_r6.CONTEXTS:
            face = run_r6.scale_model(name, capacity=length).context_interface()
            state = face.fixed_bytes(1, {"history_device": "cuda"})["total"]
            for chunk in (1, 512):
                scratch = face.scratch_bytes(batch=1, chunk=chunk, options={}, queries=chunk)["total"]
                assert state + scratch <= 1 << 30, (name, length, chunk, state + scratch)


def test_r5_with_window_attention_and_experts_runs_and_is_chunk_invariant(r5_stream):
    """Regression (found by the bounded 500M forward): the core's window-slot map and the R5 ring shared the
    attribute name ``ring``; an evidence model with window attention crashed."""
    def build(chunk):
        widths = ModelConfig(vocab_size=et.VOCAB, d_model=32, n_layers=2, n_heads=2, n_kv_heads=1, max_seq_len=64,
                             **CGC, seed=5, dtype="float64")
        ersa = ErsaConfig(widths, conv_kernel=4, attention="window", attn_heads=2, attn_kv_heads=1, attn_window=65,
                          attn_layers=(1,), chunk=chunk, sparse=EXPERTS)
        return _frozen(EvidenceLM(ersa, EvidenceConfig(block=16, capacity=256, per_block=2, key_dim=16, coarse_top=6,
                                                       evidence=3, fine_top=4, overlap=4, encoder_dim=32, fine_dim=8,
                                                       read_heads=2, read_head_dim=8)))
    whole_model, small = build(64), build(16)
    small.load_state_dict(whole_model.state_dict())
    with torch.no_grad():
        state = whole_model.new_state(2)
        whole = torch.cat([whole_model(r5_stream[:, a:a + 64], state) for a in range(0, 320, 64)], 1)
        state = small.new_state(2)
        parts = torch.cat([small(r5_stream[:, a:a + 16], state) for a in range(0, 320, 16)], 1)
    assert float((whole - parts).abs().max()) < 1e-9


def test_a_narrower_evidence_encoder_reads_projected_core_embeddings_and_stays_exact(r5_stream):
    """R6: encoder_dim < d_model (a learned down-projection, as in R3) keeps causality and chunk parity."""
    model = _frozen(tiny_evidence(encoder_dim=16, sparse=EXPERTS))
    assert model.encoder.down is not None and model.encoder.width == 16
    small = _frozen(tiny_evidence(chunk=37, encoder_dim=16, sparse=EXPERTS))
    small.load_state_dict(model.state_dict())
    with torch.no_grad():
        whole = model(r5_stream, model.new_state(2))
        state = small.new_state(2)
        parts = torch.cat([small(r5_stream[:, a:a + 37], state) for a in range(0, 600, 37)], 1)
    assert float((whole - parts).abs().max()) < 1e-9
    with pytest.raises(ConfigError):
        tiny_evidence(encoder_dim=64)                                          # wider than the model: refused
