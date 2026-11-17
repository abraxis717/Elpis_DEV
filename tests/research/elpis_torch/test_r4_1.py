"""R4.1: corrections and hardening of the R4 context runtime.

Each defect test was written before its fix and failed on R4 head 8b5de9b (recorded in EVIDENCE_R4_1.md)."""
from __future__ import annotations

import pytest

from .conftest import SKIP_REASON

torch = pytest.importorskip("torch", reason=SKIP_REASON)

from research.elpis_torch.config import ModelConfig  # noqa: E402
from research.elpis_torch.context_runtime import (BlockCache, ContextSession, RuntimeOptions,  # noqa: E402
                                                  SessionClosedError, ring_segments)
from research.elpis_torch.ersa_model import ErsaConfig  # noqa: E402
from research.elpis_torch.history import BlockIndex  # noqa: E402
from research.elpis_torch.history_model import ErsaHistoryLM, HistoryConfig  # noqa: E402

WIDTHS = dict(vocab_size=64, d_model=32, n_layers=2, n_heads=4, n_kv_heads=2, max_seq_len=64, channel="cgc",
              cgc_bottleneck=8, cgc_channels=16, dtype="float64", seed=1)


def _model(window=False, **h):
    extra = dict(attention="window", attn_window=80, attn_layers=(1,), chunk=64) if window else dict(chunk=256)
    config = ErsaConfig(ModelConfig(**WIDTHS), ersa_heads=2, key_dim=8, value_dim=8, conv_kernel=3,
                        ersa_kernel="chunked", **extra)
    model = ErsaHistoryLM(config, HistoryConfig(**{**dict(block=8, horizon=64, topk=2, index_dim=8, read_head_dim=8,
                                                         overlap=3), **h})).eval()
    for p in model.parameters():
        p.requires_grad_(False)
    return model


def _tokens(batch=1, length=400, seed=0):
    return torch.randint(0, 64, (batch, length), generator=torch.Generator().manual_seed(seed))


# -- P0: generation only from logits of the current position ----------------------------------------------------

def test_generate_refuses_logits_left_behind_by_a_keep_last_false_prefill():
    """R4 defect: prefill(keep_last=False) advanced the state but left the previous call's logits in place, so
    generate() continued from a prediction made for an earlier position."""
    model = _model()
    tokens = _tokens()
    with ContextSession(model, chunk=50) as session:
        session.prefill(tokens[:, :100])
        session.prefill(tokens[:, 100:150], keep_last=False)
        with pytest.raises(ValueError, match="current position"):
            session.generate(1)


# -- P1: occupancy of the index ring is independent of the stream's absolute origin -----------------------------

def test_an_index_opened_at_a_large_origin_is_empty_and_only_written_records_are_scored():
    """R4 defect: ring_segments counted min(next_block, slots) records as occupied, so a stream starting at a
    large origin scanned the whole (empty) capacity."""
    model = _model(horizon=4096)                      # 512 index slots
    origin = 1 << 20
    state = model.new_state(1, origin=origin)
    state.runtime = RuntimeOptions(scorer="tiled", tile=64)
    with torch.no_grad():
        model(_tokens(length=40), state)
    index = state.indexes[model.read_layers[-1]]
    held = int((index.ids >= 0).sum())
    assert held == 40 // 8
    assert sum(ids.shape[0] for _, ids, _ in ring_segments(index)) == held


def _greedy_reference(model, tokens, count):
    """Greedy continuation computed position by position through a fresh R3 reference state."""
    state = model.new_state(tokens.shape[0])
    with torch.no_grad():
        logits = model.run(tokens, state, chunk=64)[0][:, -1]
        out = []
        for _ in range(count):
            token = logits.argmax(-1)
            out.append(token)
            logits = model(token[:, None], state)[:, 0]
    return torch.stack(out, dim=1)


def test_generation_continues_correctly_across_prefills_steps_and_chunk_boundaries():
    model = _model()
    tokens = _tokens(batch=2)
    reference = _greedy_reference(model, tokens[:, :233], 12)
    with ContextSession(model, batch=2, chunk=37) as session:
        session.prefill(tokens[:, :100])
        session.prefill(tokens[:, 100:150], keep_last=False)         # no current prediction ...
        session.step(tokens[:, 150])                                    # ... re-established by a step
        session.prefill(tokens[:, 151:233])                             # several chunks, keep_last
        assert session.current_logits() is not None
        assert torch.equal(session.generate(12), reference)
        assert session.position == 233 + 12
        assert session.generate(0).shape == (2, 0) and session.position == 245      # zero: nothing changes
        for bad in (-1, 1.5, True, "3"):
            with pytest.raises(ValueError):
                session.generate(bad)


def test_empty_prefill_changes_nothing_and_keeps_or_hides_the_current_prediction():
    model = _model()
    tokens = _tokens()
    with ContextSession(model, chunk=50) as session:
        before = session.prefill(tokens[:, :60]).last_logits
        empty = session.prefill(tokens[:, :0])
        assert session.position == 60 and torch.equal(empty.last_logits, before) and empty.tokens == 0
        assert session.prefill(tokens[:, :0], keep_last=False).last_logits is None
        assert torch.equal(session.current_logits(), before)            # an empty call does not advance
        with pytest.raises(ValueError):
            session.prefill(tokens[:, :0], probes=[60])


def test_reset_and_close_invalidate_the_generation_state():
    model = _model()
    tokens = _tokens()
    session = ContextSession(model, chunk=50)
    session.prefill(tokens[:, :80])
    session.reset()
    assert session.current_logits() is None
    with pytest.raises(ValueError):
        session.generate(1)
    session.prefill(tokens[:, :80])
    session.close()
    for call in (lambda: session.generate(1), session.current_logits, lambda: session.prefill(tokens[:, :8])):
        with pytest.raises(SessionClosedError):
            call()


def test_a_direct_model_call_on_the_session_state_also_invalidates_the_kept_prediction():
    model = _model()
    tokens = _tokens()
    with ContextSession(model, chunk=50) as session:
        session.prefill(tokens[:, :80])
        with torch.no_grad():
            model(tokens[:, 80:90], session.state)                       # bypasses the session API
        assert session.current_logits() is None
        with pytest.raises(ValueError):
            session.generate(1)


# -- P1 matrix ----------------------------------------------------------------------------------------------------

def _invariant(index):
    held = int((index.ids >= 0).sum())
    segments = ring_segments(index)
    covered = [b for keys, ids, first in segments for b in range(first, first + ids.shape[0])]
    stored = sorted(int(b) for b in index.ids if b >= 0)
    assert index.count == held == len(covered) and covered == stored
    for keys, ids, first in segments:
        assert ids.tolist() == list(range(first, first + ids.shape[0]))          # views hold what they claim


@pytest.mark.parametrize("origin_block", [0, 1 << 17, (1 << 17) + 3])
def test_block_index_occupancy_through_empty_partial_full_wrapped_and_reset(origin_block):
    index = BlockIndex(2, 8, 2, 3)
    _invariant(index)                                                            # empty
    block = origin_block
    for count in (3, 5, 4, 9, 1, 16):                                            # partial, full, wrapped, overfull
        index.write(block, torch.randn(2, count, 2, 3))
        block += count
        _invariant(index)
    pointer = index.keys.data_ptr()
    index.reset()
    assert index.count == 0 and ring_segments(index) == [] and index.keys.data_ptr() == pointer
    index.write(5, torch.randn(2, 2, 2, 3))
    _invariant(index)


@pytest.mark.parametrize("origin", [0, 1 << 20, (1 << 20) - 64 * 8])
def test_tiled_scoring_at_any_origin_scores_only_held_records_and_matches_the_reference(origin):
    """Reference = the R3 path (it masks empty slots explicitly). Includes continuation beyond 2^20 and eviction."""
    model = _model(horizon=256)                                   # 32 slots of 8 tokens
    tokens = _tokens(batch=2, length=400)

    def run(runtime):
        state = model.new_state(2, origin=origin)
        state.runtime = runtime
        outs, picks, scored = [], [], []
        with torch.no_grad():
            for at in range(0, 400, 23):
                index = state.indexes[model.read_layers[-1]]
                held, before = index.count, (runtime.counters.get("records_scored", 0) if runtime else 0)
                logits, diag = model(tokens[:, at:at + 23], state, diagnostics=True)
                outs.append(logits)
                picks.append(diag[model.read_layers[-1]]["selected"])
                if runtime is not None:
                    length = tokens[:, at:at + 23].shape[1]
                    new = (origin + at + length) // 8 - (origin + at) // 8
                    scored.append((runtime.counters.get("records_scored", 0) - before, held, new, length))
                _invariant(state.indexes[model.read_layers[-1]])
        return torch.cat(outs, 1), torch.cat(picks, 1), scored, state

    reference, ref_picks, _, _ = run(None)
    tiled, picks, scored, state = run(RuntimeOptions(scorer="tiled", tile=5, tile_elements=0))
    assert torch.equal(picks, ref_picks) and float((tiled - reference).abs().max()) < 1e-10
    for work, held, new, length in scored:                       # held + in-chunk blocks, never the empty capacity
        assert work <= (held + new) * length
    assert state.position == origin + 400 and state.indexes[model.read_layers[-1]].count == 32
    position = origin + torch.arange(400)[None, :, None]
    used = picks >= 0
    assert bool(((picks[used] + 1) * 8 - 1 <= position.expand_as(picks)[used]).all())          # complete
    assert bool((picks[used] >= ((position + 1) // 8 - 32).expand_as(picks)[used]).all())      # not evicted


def test_sessions_at_different_origins_and_reuse_after_reset_keep_correct_occupancy():
    model = _model(horizon=256)
    tokens = _tokens(batch=1, length=300)
    a = ContextSession(model, origin=0, chunk=40, cache_bytes=1 << 16)
    b = ContextSession(model, origin=1 << 20, chunk=40, cache_bytes=1 << 16)
    out_a, out_b = a.prefill(tokens).last_logits, b.prefill(tokens).last_logits
    assert float((out_a - out_b).abs().max()) < 1e-12                     # relative positions only
    for session in (a, b):
        _invariant(session.state.indexes[model.read_layers[-1]])
        session.reset()
        assert session.state.indexes[model.read_layers[-1]].count == 0
        assert float((session.prefill(tokens).last_logits - out_a).abs().max()) < 1e-12
        _invariant(session.state.indexes[model.read_layers[-1]])


# -- P2: a running session refuses a changed model revision --------------------------------------------------------

from research.elpis_torch.context_runtime import ModelRevisionError  # noqa: E402

MUTATIONS = {
    "embedding": lambda m: m.embed.weight,
    "block_encoder": lambda m: m.encoder.qkv.weight,
    "index_key_projection": lambda m: m.readers[str(m.read_layers[-1])].index_key.weight,
    "reader": lambda m: m.readers[str(m.read_layers[-1])].q.weight,
    "ersa_upstream": lambda m: m.blocks[0].ersa.k.weight,
    "final_block_cgc": lambda m: m.blocks[-1].cgc.core.weight,
}


@pytest.mark.parametrize("target", sorted(MUTATIONS))
def test_in_place_mutation_of_any_parameter_refuses_continuation_until_reset(target):
    model = _model()
    tokens = _tokens()
    session = ContextSession(model, chunk=50, cache_bytes=1 << 16)
    session.prefill(tokens[:, :120])
    with torch.no_grad():
        MUTATIONS[target](model).mul_(1.01)
    for call in (lambda: session.prefill(tokens[:, 120:130]), lambda: session.step(tokens[:, 120]),
                 lambda: session.generate(1)):
        with pytest.raises(ModelRevisionError):
            call()
    assert session.position == 120                                         # nothing was computed
    session.reset()                                                        # rebinds to the model as it is now
    out = session.prefill(tokens[:, :120]).last_logits
    fresh = ContextSession(model, chunk=50).prefill(tokens[:, :120]).last_logits
    assert float((out - fresh).abs().max()) < 1e-12


def test_replacing_a_parameter_tensor_with_equal_values_is_still_a_new_revision():
    model = _model()
    tokens = _tokens()
    session = ContextSession(model, chunk=50)
    session.prefill(tokens[:, :60])
    block = model.blocks[0].ersa.k
    block.weight = torch.nn.Parameter(block.weight.detach().clone(), requires_grad=False)
    with pytest.raises(ModelRevisionError):
        session.step(tokens[:, 60])


def test_unfreezing_or_train_mode_is_refused_and_reset_requires_a_frozen_model():
    model = _model()
    tokens = _tokens()
    session = ContextSession(model, chunk=50)
    session.prefill(tokens[:, :60])
    model.train()
    with pytest.raises(ModelRevisionError):
        session.step(tokens[:, 60])
    with pytest.raises(ValueError):
        session.reset()
    model.eval()
    session.reset()
    session.prefill(tokens[:, :60])


def test_an_unchanged_model_never_triggers_the_revision_check_and_sessions_share_it_safely():
    model = _model()
    tokens = _tokens(batch=1)
    a, b = ContextSession(model, chunk=50, cache_bytes=1 << 16), ContextSession(model, chunk=50)
    a.prefill(tokens[:, :150])
    b.prefill(_tokens(batch=1, seed=4)[:, :90])
    assert a.generate(20).shape == (1, 20) and b.generate(5).shape == (1, 5)
    with torch.no_grad():
        model.embed.weight.add_(0.0)                                       # an in-place op: a new version
    for session in (a, b):                                                  # both sessions notice
        with pytest.raises(ModelRevisionError):
            session.generate(1)


def test_writes_that_bypass_version_tracking_are_caught_only_by_the_opt_in_digest():
    model = _model()
    tokens = _tokens()
    session = ContextSession(model, chunk=50, record_digest=True)
    session.prefill(tokens[:, :60])
    session.verify_digest()                                                 # unchanged: passes
    model.encoder.qkv.weight.data.mul_(1.01)                                # no version-counter change
    session.step(tokens[:, 60])                                             # the cheap check cannot see it
    with pytest.raises(ModelRevisionError):
        session.verify_digest()
    with pytest.raises(ModelRevisionError):                                 # and the session is now refused
        session.step(tokens[:, 61])
    with pytest.raises(ValueError):
        ContextSession(model).verify_digest()                              # opt-in only
