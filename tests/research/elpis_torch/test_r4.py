"""R4: the context runtime. Exact selection, bounded streaming, rollover, block cache, gradient replay, isolation."""
from __future__ import annotations

import copy
import pickle

import pytest

from .conftest import SKIP_REASON

torch = pytest.importorskip("torch", reason=SKIP_REASON)

from research.elpis_torch.config import ModelConfig  # noqa: E402
from research.elpis_torch.context_runtime import (BlockCache, BudgetError, ContextSession,  # noqa: E402
                                                  RuntimeOptions, SessionClosedError, exact_topk)
from research.elpis_torch.ersa_model import ErsaConfig, ErsaLM  # noqa: E402
from research.elpis_torch.history import EvictedError  # noqa: E402
from research.elpis_torch.history_model import ErsaHistoryLM, HistoryConfig  # noqa: E402
from research.elpis_torch.scaling import fixed_state_bytes  # noqa: E402

WIDTHS = dict(vocab_size=64, d_model=32, n_layers=2, n_heads=4, n_kv_heads=2, max_seq_len=64, channel="cgc",
              cgc_bottleneck=8, cgc_channels=16, dtype="float64", seed=1)
HISTORY = dict(block=8, horizon=64, topk=2, index_dim=8, read_head_dim=8, overlap=3)
CUDA = torch.cuda.is_available()
# R4.1: the CUDA tests run only on the device named here (default cuda:0); nothing else on the GPU is touched.
CUDA_DEVICE = __import__("os").environ.get("ELPIS_TORCH_CUDA_DEVICE", "cuda:0")


def _model(window=False, frozen=True, dtype="float64", **h):
    extra = dict(attention="window", attn_window=80, attn_layers=(1,), chunk=64) if window else dict(chunk=256)
    config = ErsaConfig(ModelConfig(**{**WIDTHS, "dtype": dtype}), ersa_heads=2, key_dim=8, value_dim=8,
                        conv_kernel=3, ersa_kernel="chunked", **extra)
    model = ErsaHistoryLM(config, HistoryConfig(**{**HISTORY, **h})).eval()
    if frozen:
        for p in model.parameters():
            p.requires_grad_(False)
    return model


def _tokens(batch=2, length=400, seed=0):
    return torch.randint(0, 64, (batch, length), generator=torch.Generator().manual_seed(seed))


def _run(model, tokens, step, runtime=None, origin=0):
    state = model.new_state(tokens.shape[0], origin=origin)
    state.runtime = runtime
    outs, picks = [], []
    with torch.no_grad():
        for at in range(0, tokens.shape[1], step):
            logits, diag = model(tokens[:, at:at + step], state, diagnostics=True)
            outs.append(logits)
            picks.append(diag[model.read_layers[-1]]["selected"])
    return torch.cat(outs, 1), torch.cat(picks, 1), state


# -- P2: exact top-K and the tiled scorer ---------------------------------------------------------------------

def _brute_topk(scores, ids, k):
    """Reference by enumeration: sort by (score descending, id descending); -inf entries are not selected."""
    out_v, out_i = [], []
    for row_s, row_i in zip(scores.reshape(-1, scores.shape[-1]).tolist(), ids.expand_as(scores).reshape(
            -1, scores.shape[-1]).tolist()):
        order = sorted(range(len(row_s)), key=lambda j: (-row_s[j], -row_i[j]))
        picked = [j for j in order if row_s[j] != float("-inf")][:k]
        out_v.append([row_s[j] for j in picked] + [float("-inf")] * (k - len(picked)))
        out_i.append([row_i[j] for j in picked] + [-1] * (k - len(picked)))
    return out_v, out_i


@pytest.mark.parametrize("seed", range(6))
def test_exact_topk_resolves_adversarial_ties_newer_block_first(seed):
    g = torch.Generator().manual_seed(seed)
    scores = torch.randint(0, 4, (3, 5, 11), generator=g).double()        # many exact ties
    scores[0, 0] = float("-inf")                                          # nothing eligible
    scores[1, 1, 3:] = float("-inf")                                      # fewer eligible than k
    ids = torch.randperm(11, generator=g)
    values, positions = exact_topk(scores, ids, 4)
    want_v, want_i = _brute_topk(scores, ids[None, None], 4)
    got_i = torch.where(torch.isfinite(values), ids[positions], torch.full_like(positions, -1))
    assert values.reshape(-1, 4).tolist() == want_v and got_i.reshape(-1, 4).tolist() == want_i


@pytest.mark.parametrize("window", [False, True])
@pytest.mark.parametrize("step,tile", [(64, 3), (13, 5), (1, 64), (64, 1000)])
def test_tiled_scoring_selects_the_reference_blocks_and_gives_the_reference_logits(window, step, tile):
    """Partially populated index (early positions), full index, rollover past the horizon, several chunkings."""
    model = _model(window=window)
    tokens = _tokens()
    step = min(step, model.ersa.chunk)
    reference, ref_picks, _ = _run(model, tokens, 64 if not window else 64)
    tiled, picks, _ = _run(model, tokens, step, RuntimeOptions(scorer="tiled", tile=tile))
    assert torch.equal(picks, ref_picks)
    assert float((tiled - reference).abs().max()) < 1e-10


def test_adversarial_ties_between_identical_blocks_are_resolved_newer_first_and_agree_in_score():
    """Identical repeated blocks with the recency slope flattened: content scores tie. Within one evaluation, bitwise
    ties are resolved newer block first. Across different chunkings or tilings, the same block can be encoded in a
    different batch and its score can differ in the last bits, which may reorder such near-ties (this holds for the
    R3 reference itself); the selected SCORES then agree to rounding. That is the documented tolerance."""
    model = _model()
    with torch.no_grad():
        for reader in model.readers.values():
            reader.recency.fill_(-1e4)                    # softplus ~ 0: no recency separation
    tokens = torch.arange(8).repeat(2, 32) % 64           # every block identical, 256 tokens = one reference chunk
    with torch.no_grad():
        _, diag = model(tokens, model.new_state(2), diagnostics=True)
    record = diag[model.read_layers[-1]]
    scores, table = record["scores"], record["table_ids"]
    top = scores.topk(3, dim=-1).values
    tied = torch.isfinite(top[..., 1]) & (top[..., 0] == top[..., 1])
    assert bool(tied.any())                               # the fixture really produces bitwise ties
    chosen = record["selected"][tied]
    assert bool((chosen[:, 0] > chosen[:, 1]).all())      # ... resolved newer block first
    position = {int(b): i for i, b in enumerate(table)}
    for step, tile in ((256, 1000), (7, 2), (13, 64)):
        _, picks, _ = _run(model, tokens, step, RuntimeOptions(scorer="tiled", tile=tile))
        mine = torch.where(picks >= 0, torch.tensor([[[position.get(int(b), 0) for b in row] for row in batch]
                                                     for batch in picks.tolist()]), 0)
        got = torch.where(picks >= 0, scores.gather(-1, mine), torch.full_like(scores[..., :2], float("-inf")))
        want = torch.where(record["selected"] >= 0, scores.gather(-1, torch.where(
            record["selected"] >= 0, torch.tensor([[[position.get(int(b), 0) for b in row] for row in batch]
                                                    for batch in record["selected"].tolist()]), 0)),
            torch.full_like(scores[..., :2], float("-inf")))
        finite = torch.isfinite(want)
        assert torch.equal(finite, torch.isfinite(got))
        assert float((got[finite] - want[finite]).abs().max()) < 1e-12


def test_float32_tiled_scoring_matches_the_float32_reference():
    model = _model(dtype="float32")
    tokens = _tokens()
    reference, ref_picks, _ = _run(model, tokens, 64)
    tiled, picks, _ = _run(model, tokens, 64, RuntimeOptions(scorer="tiled", tile=5))
    assert torch.equal(picks, ref_picks)
    assert float((tiled - reference).abs().max()) < 1e-4


def test_the_tiled_scorer_refuses_the_oracle_diagnosis_modes():
    model = _model()
    state = model.new_state(2)
    state.runtime = RuntimeOptions(scorer="tiled")
    with pytest.raises(ValueError):
        model(_tokens(length=64), state, mode="oracle", address=torch.full((2, 64), -1))


def test_invalid_source_addresses_and_evicted_positions_are_refused():
    model = _model()
    state = model.new_state(2)
    with torch.no_grad():
        model(_tokens(length=200), state)
    with pytest.raises(EvictedError):
        state.tokens.gather(torch.tensor([0]))                            # evicted
    with pytest.raises(EvictedError):
        state.tokens.gather(torch.tensor([200]))                          # not yet written
    bad = torch.full((2, 8), -1)
    bad[:, 3] = 0                                                         # block 0 left the horizon
    with pytest.raises(ValueError):
        model(_tokens(length=8, seed=3), state, mode="oracle", address=bad)


# -- P1: the streaming session --------------------------------------------------------------------------------

def test_streaming_prefill_keeps_only_requested_logits_and_matches_the_reference():
    model = _model()
    tokens = _tokens()
    full = model.run(tokens, chunk=256)[0]
    with ContextSession(model, batch=2, chunk=37) as session:
        result = session.prefill(tokens[:, :300], probes=[0, 5, 150, 299])
        assert sorted(result.probes) == [0, 5, 150, 299]
        for position, logits in result.probes.items():
            assert float((logits - full[:, position]).abs().max()) < 1e-10
        assert float((result.last_logits - full[:, 299]).abs().max()) < 1e-10
        more = session.prefill(tokens[:, 300:390])                        # continuation over calls
        assert float((more.last_logits - full[:, 389]).abs().max()) < 1e-10
        for position in range(390, 400):                                  # incremental decoding
            assert float((session.step(tokens[:, position]) - full[:, position]).abs().max()) < 1e-10
        assert session.position == 400


def test_logit_positions_equal_slices_of_the_full_logits():
    model = _model()
    tokens = _tokens(length=100)
    full = model(tokens, model.new_state(2))
    some = model(tokens, model.new_state(2), logit_positions=torch.tensor([0, 7, 99]))
    # Different matrix shapes (3 rows vs 100): equal up to BLAS rounding, which depends on the CPU (CI run 435
    # measured 8e-17 where the development CPU gave 0.0). Bitwise equality is not the contract.
    assert float((some - full[:, [0, 7, 99]]).abs().max()) < 1e-12


def test_reset_is_in_place_and_reproduces_a_fresh_stream():
    model = _model(window=True)
    tokens = _tokens()
    with ContextSession(model, batch=2, chunk=50, cache_bytes=1 << 16) as session:
        first = session.prefill(tokens).last_logits.clone()
        pointers = [session.state.tokens.tokens.data_ptr(), session.state.cells[0].data_ptr()]
        session.reset()
        assert session.position == 0 and not session.state.runtime.cache.entries
        again = session.prefill(tokens).last_logits
        assert [session.state.tokens.tokens.data_ptr(), session.state.cells[0].data_ptr()] == pointers
    assert float((first - again).abs().max()) == 0.0


def test_sessions_are_isolated_frozen_only_unserializable_and_closable():
    model = _model()
    a, b = ContextSession(model, batch=1, cache_bytes=1 << 16), ContextSession(model, batch=1, cache_bytes=1 << 16)
    tokens = _tokens(batch=1)
    alone = ContextSession(model, batch=1).prefill(tokens[:, :200]).last_logits
    a.prefill(tokens[:, :200])
    b.prefill(_tokens(batch=1, seed=9)[:, :300])                           # interleaved, different stream
    a_out = a.prefill(tokens[:, 200:201]).last_logits
    ref = ContextSession(model, batch=1).prefill(tokens[:, :201]).last_logits
    assert float((a_out - ref).abs().max()) < 1e-12 and alone is not None
    assert a.state.runtime.cache is not b.state.runtime.cache
    for bad in (pickle.dumps, copy.deepcopy):
        with pytest.raises(TypeError):
            bad(a)
    a.close()
    with pytest.raises(SessionClosedError):
        a.step(torch.zeros(1, dtype=torch.long))
    with pytest.raises(ValueError):
        ContextSession(_model(frozen=False))


def test_budget_refusal_happens_before_allocation_and_the_state_formula_is_exact():
    model = _model(window=True)
    assert fixed_state_bytes(model, 3)["total"] == model.new_state(3).nbytes()["total"]
    session = ContextSession(model, batch=1, cache_bytes=1 << 16)
    need = session.projected_bytes()
    assert need["state"] == model.new_state(1).nbytes()["total"] and need["block_cache"] == 1 << 16
    with pytest.raises(BudgetError):
        ContextSession(model, batch=1, cache_bytes=1 << 16, budget_bytes=need["total"] - 1)
    ContextSession(model, batch=1, cache_bytes=1 << 16, budget_bytes=need["total"])


def test_state_memory_is_constant_and_buffers_are_never_reallocated_while_streaming():
    model = _model()
    with ContextSession(model, batch=1, chunk=23, cache_bytes=4096) as session:
        session.prefill(_tokens(batch=1, length=30))
        pointers = (session.state.tokens.tokens.data_ptr(), session.state.indexes[1].keys.data_ptr())
        size = session.state.nbytes()["total"]
        for seed in range(6):
            session.prefill(_tokens(batch=1, length=97, seed=seed))
            assert session.state.nbytes()["total"] == size
            assert session.state.runtime.cache.bytes <= 4096
        assert (session.state.tokens.tokens.data_ptr(), session.state.indexes[1].keys.data_ptr()) == pointers


# -- P1: rollover at the million-token boundary ---------------------------------------------------------------

def test_streams_continue_across_absolute_position_2_pow_20_with_unchanged_logits():
    """The logits depend only on relative positions: a stream crossing 1,048,576 equals the same stream at 0."""
    model = _model(horizon=1 << 20)
    tokens = _tokens(length=320)
    at_zero, picks_zero, _ = _run(model, tokens, 64, RuntimeOptions(scorer="tiled"))
    origin = (1 << 20) - 160
    crossing, picks_crossing, state = _run(model, tokens, 64, RuntimeOptions(scorer="tiled"), origin=origin)
    assert state.position == origin + 320 > 1 << 20
    # The index ring wraps at a different slot for another origin, so the scoring matrices have other shapes:
    # equal to rounding (CI run 435: 1.7e-16), with identical selections.
    assert float((at_zero - crossing).abs().max()) < 1e-12
    shift = origin // model.history.block
    assert torch.equal(torch.where(picks_crossing >= 0, picks_crossing - shift, picks_crossing), picks_zero)


def test_eviction_is_deterministic_and_selection_never_reaches_beyond_the_horizon():
    model = _model()
    tokens = _tokens(length=600)
    _, picks, state = _run(model, tokens, 41, RuntimeOptions(scorer="tiled", cache=BlockCache(1 << 14)))
    h = model.history
    position = torch.arange(600)[None, :, None].expand_as(picks)
    used = picks >= 0
    assert bool((picks[used] >= ((position + 1) // h.block - h.slots)[used]).all())
    assert bool(((picks[used] + 1) * h.block - 1 <= position[used]).all())
    held = sorted(state.indexes[1].ids.tolist())
    assert held == list(range(600 // h.block - h.slots, 600 // h.block))
    assert min(state.runtime.cache.entries) >= 600 // h.block - h.slots
    _, picks_again, _ = _run(model, tokens, 41, RuntimeOptions(scorer="tiled", cache=BlockCache(1 << 14)))
    assert torch.equal(picks, picks_again)


# -- P3: the encoded-block cache ------------------------------------------------------------------------------

@pytest.mark.parametrize("capacity", [1, 14000, 1 << 20])   # one entry here: 2 x 9 x 32 x 8 = 4,608 B
def test_cache_cold_or_warm_gives_the_uncached_output_within_its_byte_bound(capacity):
    model = _model(window=True)
    tokens = _tokens()
    plain, plain_picks, _ = _run(model, tokens, 9, RuntimeOptions(scorer="tiled"))
    runtime = RuntimeOptions(scorer="tiled", cache=BlockCache(capacity))
    cached, picks, _ = _run(model, tokens, 9, runtime)
    assert torch.equal(picks, plain_picks) and float((cached - plain).abs().max()) < 1e-12
    stats = runtime.cache.stats
    assert runtime.cache.bytes <= capacity
    if capacity == 1:
        assert stats["hits"] == 0 and stats["rejected_too_large"] > 0
    if capacity == 1 << 20:
        assert stats["hits"] > 0 and runtime.counters.get("blocks_reencoded", 0) == 0
    if capacity == 14000:
        assert stats["evictions"] > 0


def test_cache_is_invalidated_when_an_encoder_or_embedding_parameter_changes():
    model = _model()
    tokens = _tokens()
    runtime = RuntimeOptions(scorer="tiled", cache=BlockCache(1 << 20))
    state = model.new_state(2)
    state.runtime = runtime
    with torch.no_grad():
        model(tokens[:, :200], state)
        assert runtime.cache.entries
        model.encoder.qkv.weight.mul_(1.01)                               # in place: version counter advances
        continued = model(tokens[:, 200:260], state)
    assert runtime.cache.stats["invalidations"] == 1
    # The continuation equals a reread with the new encoder (no stale cached block was used).
    fresh_state = model.new_state(2)
    fresh_state.runtime = RuntimeOptions(scorer="tiled")
    with torch.no_grad():
        model.encoder.qkv.weight.div_(1.01)
        model(tokens[:, :200], fresh_state)                               # same index keys as the cached run
        model.encoder.qkv.weight.mul_(1.01)
        expected = model(tokens[:, 200:260], fresh_state)
    assert float((continued - expected).abs().max()) < 1e-12


def test_cache_is_bypassed_with_gradients_so_training_keeps_its_graph():
    model = _model(frozen=False).train()
    runtime = RuntimeOptions(scorer="tiled", cache=BlockCache(1 << 20))
    state = model.new_state(2)
    state.runtime = runtime
    logits = model(_tokens(length=200), state)
    logits.sum().backward()
    assert runtime.cache.stats["hits"] == 0 and runtime.cache.stats["bypassed"] > 0 and not runtime.cache.entries
    assert model.encoder.qkv.weight.grad is not None


# -- P4: cross-chunk gradient replay --------------------------------------------------------------------------

def _gradients(model, tokens, step, key_grad_blocks):
    model.zero_grad(set_to_none=True)
    state = model.new_state(tokens.shape[0])
    state.runtime = RuntimeOptions(key_grad_blocks=key_grad_blocks)
    weights = torch.randn(tokens.shape[1], 64, generator=torch.Generator().manual_seed(5), dtype=torch.float64)
    loss = 0.0
    for at in range(0, tokens.shape[1], step):
        logits = model(tokens[:, at:at + step], state)
        loss = loss + (torch.log_softmax(logits, -1) * weights[at:at + step]).sum()
    loss.backward()
    return {n: p.grad.clone() for n, p in model.named_parameters() if p.grad is not None}, state


def test_key_replay_makes_chunked_training_gradients_equal_the_unchunked_ones():
    model = _model(frozen=False, horizon=256).train()
    tokens = _tokens(length=200)
    whole, _ = _gradients(model, tokens, 200, 0)
    replayed, state = _gradients(model, tokens, 40, -1)
    detached, _ = _gradients(model, tokens, 40, 0)
    assert state.runtime.counters["replayed_key_blocks"] > 0
    encoder = [n for n in whole if n.startswith(("encoder.", "readers.", "embed"))]
    for name in whole:
        assert float((replayed[name] - whole[name]).abs().max()) < 1e-9, name
    # Without replay the stored keys are detached: the encoder misses gradient that crosses the chunk boundary.
    assert max(float((detached[n] - whole[n]).abs().max()) for n in encoder) > 1e-6


def test_a_bounded_replay_horizon_bounds_which_distant_keys_receive_gradient():
    model = _model(frozen=False, horizon=256).train()
    tokens = _tokens(length=200)
    whole, _ = _gradients(model, tokens, 200, 0)
    bounded, state = _gradients(model, tokens, 40, 2)
    full_replay, _ = _gradients(model, tokens, 40, -1)
    # A 2-block horizon replays at most 2 resident blocks per chunk (and less work than full replay).
    assert 0 < state.runtime.counters["replayed_key_blocks"] <= 2 * (200 // 40)
    name = "encoder.qkv.weight"
    assert float((bounded[name] - whole[name]).abs().max()) > float((full_replay[name] - whole[name]).abs().max())


@pytest.mark.skipif(not CUDA, reason="NOT_RUN: no CUDA device in this environment (not a PASS)")
def test_cuda_session_with_cpu_token_history_matches_cpu():   # pragma: no cover - needs the workstation
    model = _model()
    tokens = _tokens()
    reference = ContextSession(model, batch=2).prefill(tokens).last_logits
    gpu = copy.deepcopy(model).to(CUDA_DEVICE)
    with ContextSession(gpu, batch=2, history_device="cpu", cache_bytes=1 << 20) as session:
        out = session.prefill(tokens.to(CUDA_DEVICE)).last_logits.cpu()
        assert session.state.tokens.device.type == "cpu" and session.state.cells[0].is_cuda
    assert float((out - reference).abs().max()) < 1e-9


def test_plain_ersa_models_stream_through_a_session_too():
    config = ErsaConfig(ModelConfig(**WIDTHS), ersa_heads=2, key_dim=8, value_dim=8, conv_kernel=3, chunk=64,
                        ersa_kernel="chunked")
    model = ErsaLM(config).eval()
    for p in model.parameters():
        p.requires_grad_(False)
    tokens = _tokens(length=150)
    full = model.run(tokens)[0]
    with ContextSession(model, batch=2, chunk=30) as session:
        assert float((session.prefill(tokens).last_logits - full[:, -1]).abs().max()) < 1e-10
