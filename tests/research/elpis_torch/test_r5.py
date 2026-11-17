"""R5: the exact ring's identity and bounds, the evidence path's streaming parity, causality, the absence of any
oracle at inference, reset / session isolation, the task generators' labels, and memory bounds."""
from __future__ import annotations

import json
import subprocess
import sys

import pytest

from .conftest import REPO, SKIP_REASON

torch = pytest.importorskip("torch", reason=SKIP_REASON)

from research.elpis_torch import evidence_tasks as et  # noqa: E402
from research.elpis_torch import longrange_tasks as r3  # noqa: E402
from research.elpis_torch.config import ModelConfig  # noqa: E402
from research.elpis_torch.context_runtime import exact_topk  # noqa: E402
from research.elpis_torch.ersa_model import ErsaConfig  # noqa: E402
from research.elpis_torch.evidence_model import EvidenceConfig, EvidenceLM  # noqa: E402
from research.elpis_torch.evidence_ring import EvidenceRing, StaleAddressError  # noqa: E402
from research.elpis_torch.history import EvictedError  # noqa: E402

CGC = dict(channel="cgc", cgc_bottleneck=8, cgc_channels=24, cgc_units=4)


def small(chunk=1024, capacity=256, block=16, dtype="float64", **changes):
    widths = ModelConfig(vocab_size=et.VOCAB, d_model=32, n_layers=2, n_heads=2, n_kv_heads=1, max_seq_len=64,
                         **CGC, seed=3, dtype=dtype)
    ersa = ErsaConfig(widths, ersa_heads=2, key_dim=8, value_dim=8, ersa_kernel="chunked", scan_block=16,
                      conv_kernel=4, chunk=chunk)
    evidence = dict(block=block, capacity=capacity, per_block=2, key_dim=16, coarse_top=6, evidence=3, fine_top=4,
                    overlap=4, encoder_dim=32, encoder_heads=2, encoder_layers=2, fine_dim=8, read_heads=2,
                    read_head_dim=8, query_tile=7)
    evidence.update(changes)
    return EvidenceLM(ersa, EvidenceConfig(**evidence)).eval()


@pytest.fixture(scope="module")
def streams():
    return et.generate_records(et.RecordTask(queries=6, background=20.0), 2, 700, seed=4)


def run(model, tokens, chunk, queries=None, **extra):
    """Logits ``{position: [B, V]}`` (and diagnostics) streaming ``tokens`` in chunks of ``chunk``."""
    state = model.new_state(tokens.shape[0])
    out, diags = {}, {}
    for at in range(0, tokens.shape[1], chunk):
        piece = tokens[:, at:at + chunk]
        q = None if queries is None else [i - at for i in queries if at <= i < at + piece.shape[1]]
        lp = None if q is None else torch.tensor(q, dtype=torch.long)
        result = model(piece, state, logit_positions=lp, **extra)
        logits, diag = result if isinstance(result, tuple) else (result, None)
        index = range(at, at + piece.shape[1]) if q is None else [at + i for i in q]
        for j, i in enumerate(index):
            out[i] = logits[:, j]
            if diag is not None:
                diags[i] = {k: v[:, j] for k, v in diag.items() if isinstance(v, torch.Tensor) and v.ndim >= 2}
    return out, diags, state


# -- the exact ring ---------------------------------------------------------------------------------------------

def test_the_ring_is_a_fifo_of_exactly_2_20_tokens_and_wraps_there():
    ring = EvidenceRing(1, capacity=1 << 20, block=32, per_block=1, dim=2, overlap=8)
    assert ring.slots == 32768 and ring.tokens.shape == (1, 1 << 20)
    data = torch.arange(1 << 20, dtype=torch.int64)[None] % 1000
    ring.append(data[:, :(1 << 20) - 1])
    ring.write(0, torch.zeros(1, ((1 << 20) - 1) // 32, 1, 2))
    assert ring.oldest == 0 and ring.position == (1 << 20) - 1
    ring.append(data[:, -1:])                                    # exactly 1,048,576 tokens held: nothing evicted
    ring.write(32767, torch.zeros(1, 1, 1, 2))
    assert ring.oldest == 0 and ring.count == 32768 and ring.live() == (0, 32768)
    assert int(ring.gather(torch.tensor([0]))[0, 0]) == 0
    ring.append(torch.tensor([[7]]))                            # token 1,048,576 overwrites token 0
    assert ring.oldest == 1 and ring.live() == (1, 32768)
    with pytest.raises(EvictedError):
        ring.gather(torch.tensor([0]))
    with pytest.raises(StaleAddressError):                     # block 0 lost its first token: no longer live
        ring.reread(torch.tensor([0]), 0)
    assert int(ring.gather(torch.tensor([1 << 20]))[0, 0]) == 7
    tokens, valid = ring.reread(torch.tensor([1]), 8)           # block 1 and its left context 24..31 are held
    assert bool(valid.all()) and tokens[0, 0].tolist() == list(range(24, 64))
    ring.append(torch.full((1, 24), 9))                         # oldest is now 25: token 24 (block 1's context) gone
    assert ring.live() == (2, 32768)
    with pytest.raises(StaleAddressError):                     # a live block always has its whole span held
        ring.reread(torch.tensor([1]), 8)
    assert ring.reread(torch.tensor([2]), 8)[1].all()


def test_stale_addresses_are_refused_never_read_and_counted():
    ring = EvidenceRing(1, capacity=64, block=16, per_block=1, dim=2)
    ring.append(torch.arange(64)[None])
    ring.write(0, torch.ones(1, 4, 1, 2))
    address = ring.address(1)
    ring.check(address)
    ring.check(address, at=40)
    with pytest.raises(StaleAddressError):
        ring.check(address, at=20)                              # block 1 is not complete at position 20
    ring.append(torch.arange(64, 96)[None])                     # positions 64..95 evict blocks 0 and 1
    ring.write(4, torch.ones(1, 2, 1, 2))
    with pytest.raises(StaleAddressError):
        ring.check(address)                                     # slot 1 now holds block 5 (generation 1)
    fresh = ring.address(5)
    assert (fresh.slot, fresh.generation) == (1, 1)
    other = EvidenceRing(1, capacity=64, block=16, per_block=1, dim=2)
    other.append(torch.arange(96)[None])
    other.write(0, torch.ones(1, 6, 1, 2))
    with pytest.raises(StaleAddressError):
        other.check(fresh)                                      # another session
    ring.reset()
    with pytest.raises(StaleAddressError):
        ring.check(fresh)                                       # another epoch
    assert ring.stats["stale_rejections"] == 3 and other.stats["stale_rejections"] == 1


def test_descriptors_are_written_in_order_only_after_their_tokens():
    ring = EvidenceRing(1, capacity=64, block=16, per_block=1, dim=2)
    ring.append(torch.arange(20)[None])
    with pytest.raises(ValueError):
        ring.write(0, torch.ones(1, 2, 1, 2))                  # block 1 is not complete
    ring.write(0, torch.ones(1, 1, 1, 2))
    with pytest.raises(ValueError):
        ring.write(2, torch.ones(1, 1, 1, 2))                  # a gap


# -- the evidence path ------------------------------------------------------------------------------------------

def test_monolithic_and_chunked_ingestion_agree_across_blocks_and_wraparound(streams):
    """Chunks of 1, 13, 37 and 100 tokens against one chunk: blocks straddle chunk boundaries, queries sit in
    partially filled tail blocks, and the 256-token ring wraps (700 tokens)."""
    model = small()
    tokens = streams.tokens
    with torch.no_grad():
        whole, wdiag, wstate = run(model, tokens, 1024, diagnostics=True)
        for chunk in (100, 37, 13, 1):
            other = small(chunk=chunk)
            other.load_state_dict(model.state_dict())
            positions = list(range(0, 700, 9)) if chunk == 1 else None
            part, pdiag, state = run(other, tokens, chunk, queries=positions, diagnostics=True)
            for i in (positions or range(700)):
                assert float((whole[i] - part[i]).abs().max()) < 1e-9, (chunk, i)
                assert torch.equal(wdiag[i]["fine_top"], pdiag[i]["fine_top"]), (chunk, i)
                assert torch.equal(wdiag[i]["selected"], pdiag[i]["selected"]), (chunk, i)
            assert torch.allclose(state.ring.keys, wstate.ring.keys, atol=1e-12)
            assert torch.equal(state.ring.ids, wstate.ring.ids)
            assert state.ring.stats["stale_rejections"] == 0
            assert state.ring.count == state.ring.slots and state.ring.position == 700


def test_the_read_is_computed_exactly_where_logits_are_requested(streams):
    model = small()
    tokens = streams.tokens
    queries = sorted(set((streams.probes["answer"] - 1).tolist()))
    with torch.no_grad():
        dense, _, _ = run(model, tokens, 1024)
        lazy, _, _ = run(model, tokens, 1024, queries=queries)
        chunked, _, _ = run(model, tokens, 64, queries=queries)
    for i in queries:
        assert float((dense[i] - lazy[i]).abs().max()) < 1e-12
        assert float((dense[i] - chunked[i]).abs().max()) < 1e-9


def test_labels_never_change_learned_outputs_and_teacher_is_training_only(streams):
    model = small()
    tokens = streams.tokens[:, :400]
    labels = streams.query_source()[:, :400]
    with torch.no_grad():
        plain = model(tokens, model.new_state(2))
        labelled, diag = model(tokens, model.new_state(2), evidence=labels, diagnostics=True)
        assert torch.equal(plain, labelled)
        with pytest.raises(ValueError):
            model(tokens, model.new_state(2), evidence=labels, teacher=True)
        with pytest.raises(ValueError):
            model(tokens, model.new_state(2), mode="oracle")
    assert int(diag["historical"].sum()) > 0


def test_future_tokens_change_no_earlier_output(streams):
    model = small()
    tokens = streams.tokens
    changed = tokens.clone()
    changed[:, 401:] = torch.randint(1, 30, changed[:, 401:].shape, generator=torch.Generator().manual_seed(0))
    with torch.no_grad():
        a = model(tokens, model.new_state(2))
        b = model(changed, model.new_state(2))
    assert torch.equal(a[:, :401], b[:, :401])


def test_reset_clears_everything_and_sessions_do_not_leak(streams):
    model = small()
    tokens = streams.tokens
    with torch.no_grad():
        fresh = model(tokens[:, 300:600], model.new_state(2))
        used = model.new_state(2)
        model(tokens[:, :300], used)
        epoch = used.ring.epoch
        used.reset()
        assert used.ring.epoch == epoch + 1 and used.ring.count == 0 and int(used.ring.tokens.abs().sum()) == 0
        assert torch.equal(model(tokens[:, 300:600], used), fresh)
        a, b = model.new_state(2), model.new_state(2)          # interleaved sessions
        model(tokens[:, :200], a)
        model(tokens.flip(1)[:, :200], b)
        assert torch.equal(model(tokens[:, 200:400], a), model(tokens[:, 200:400],
                                                               _fed(model, tokens[:, :200])))
    assert a.ring.session != b.ring.session


def _fed(model, prefix):
    state = model.new_state(prefix.shape[0])
    model(prefix, state)
    return state


def test_state_is_bounded_and_does_not_grow_over_a_lifetime(streams):
    model = small(chunk=64)
    state = model.new_state(2)
    sizes = state.nbytes()
    with torch.no_grad():
        for _ in range(3):
            for at in range(0, 700, 64):
                model(streams.tokens[:, at:at + 64], state, logit_positions=torch.tensor([0]))
            assert state.nbytes() == sizes
            assert state.ring.count <= state.ring.slots
            lo, hi = state.ring.live()
            assert hi - lo <= state.ring.slots
            state.reset()
    assert set(sizes) == {"ersa_cells", "conv_tail", "history_tokens", "index_keys", "index_ids", "total"}


def test_topk_is_deterministic_with_ties_to_the_newer_block():
    scores = torch.tensor([[[1.0, 3.0, 3.0, 2.0, 3.0]]])
    ids = torch.tensor([10, 11, 12, 13, 14])
    values, picks = exact_topk(scores, ids, 3)
    assert ids[picks[0, 0]].tolist() == [14, 12, 11]
    model = small(per_block=1)
    tokens = torch.full((1, 200), 5, dtype=torch.long)          # identical blocks: every coarse score ties
    with torch.no_grad():
        _, diag = model(tokens, model.new_state(1), logit_positions=torch.tensor([199]), diagnostics=True)
    chosen = diag["fine_top"][0, 0]
    assert chosen[chosen >= 0].tolist() == sorted(chosen[chosen >= 0].tolist(), reverse=True)


def test_the_working_context_is_bounded_and_ranges_are_exposed(streams):
    model = small()
    with torch.no_grad():
        _, diag = model(streams.tokens[:, :600], model.new_state(2), evidence=streams.query_source()[:, :600],
                        diagnostics=True)
    e = model.evidence
    assert diag["selected"].shape[-1] == e.evidence and diag["fine_top"].shape[-1] == e.fine_top
    assert int(diag["candidates"].max()) <= e.coarse_top
    positions = torch.arange(600)
    selected = diag["selected"]
    ok = selected >= 0
    # every selected block is complete before the query's own block and inside the ring's reach
    assert bool(((selected[ok] + 1) * e.block <= (positions[None, :, None].expand_as(selected)[ok] // e.block)
                 * e.block).all())
    assert bool((diag["local_block"] == positions[None] // e.block).all())


def test_bfloat16_index_storage_keeps_the_selection_close(streams):
    model = small(dtype="float32")
    low = small(dtype="float32", key_dtype="bfloat16")
    low.load_state_dict(model.state_dict())
    queries = torch.tensor(sorted(set((streams.probes["answer"] - 1).tolist())))
    with torch.no_grad():
        _, a = model(streams.tokens, model.new_state(2), logit_positions=queries, diagnostics=True)
        state = low.new_state(2)
        _, b = low(streams.tokens, state, logit_positions=queries, diagnostics=True)
    assert state.ring.keys.dtype == torch.bfloat16
    assert float((a["fine_top"] == b["fine_top"]).double().mean()) > 0.8


def test_default_persistent_retrieval_state_is_far_below_200_mib():
    widths = ModelConfig(vocab_size=et.VOCAB, d_model=64, n_layers=2, n_heads=4, n_kv_heads=2, max_seq_len=64,
                         **dict(CGC, cgc_bottleneck=32, cgc_channels=96))
    model = EvidenceLM(ErsaConfig(widths, conv_kernel=4, chunk=64), EvidenceConfig())
    e = model.evidence
    assert (e.capacity, e.block, e.slots, e.per_block, e.key_dim, e.coarse_top) == (1 << 20, 32, 32768, 4, 64, 64)
    state = model.new_state(1, ring_device="meta")
    sizes = state.ring.nbytes()
    assert sizes == {"history_tokens": 4 << 20, "index_keys": 32768 * 4 * 64 * 4, "index_ids": 32768 * 8}
    assert sum(sizes.values()) < 40 << 20 < 200 << 20


def test_the_evidence_stream_writes_nothing_to_storage(tmp_path):
    scenario = REPO / "tests" / "research" / "elpis_torch" / "zero_write_scenario.py"
    run = subprocess.run([sys.executable, "-B", str(scenario), "--evidence", "--audit"], capture_output=True, text=True,
                         cwd=REPO, env={"PYTHONPATH": str(REPO), "PYTHONDONTWRITEBYTECODE": "1"})
    assert run.returncode == 0, run.stderr
    report = json.loads(run.stdout.strip().splitlines()[-1])
    assert report["write_events"] == [] and report["position"] == 900
    control = subprocess.run([sys.executable, "-B", str(scenario), "--evidence", "--audit", "--negative-control",
                              str(tmp_path / "x")], capture_output=True, text=True, cwd=REPO,
                             env={"PYTHONPATH": str(REPO), "PYTHONDONTWRITEBYTECODE": "1"})
    assert json.loads(control.stdout.strip().splitlines()[-1])["write_events"]          # the detector works


# -- the task generators ----------------------------------------------------------------------------------------

def test_the_vocabulary_extends_r3_so_r3_streams_are_valid_inputs():
    assert et.VOCAB > r3.VOCAB and et.NOT == r3.VOCAB and et.ASK > et.NOT
    streams = et.generate_r3(2, 512, seed=1)
    assert int(streams.tokens.max()) < r3.VOCAB and streams.probes["row"].numel() > 0


def test_records_labels_point_at_the_latest_bind_and_negatives_are_what_they_claim():
    task = et.RecordTask(queries=12, revisions=(1, 3), negated=(1, 2), reference=(1, 1), overlap=(1, 1))
    data = et.generate_records(task, 3, 2048, seed=11)
    p, tokens = data.probes, data.tokens
    finals = 0
    for i in range(p["row"].numel()):
        row, answer, src = int(p["row"][i]), int(p["answer"][i]), int(p["source"][i])
        key = int(tokens[row, src - 1])
        assert r3.KEY0 <= key < r3.KEY0 + r3.MAX_KEYS and int(tokens[row, src]) == int(tokens[row, answer])
        assert int(tokens[row, src - 2]) not in (et.NOT, et.USE)
        head = int(tokens[row, answer - 1])
        k = key - r3.KEY0
        assert head == r3.ASSOC0 + r3._PAIRING[k] or (head == key and int(tokens[row, answer - 2]) == et.ASK)
        # no later bind of the same key before the question
        span = tokens[row, src + 1:answer - 1]
        later = ((span[:-1] == key) & (span[1:] >= r3.VAL0) & (span[1:] < r3.VAL0 + r3.VALUES))
        prior = torch.cat((tokens[row, src:answer - 2], ))
        negated = (prior[:-2] == et.NOT) if prior.numel() > 2 else torch.zeros(0, dtype=torch.bool)
        assert not bool(later.any()) or bool(negated.any())
        for old in p["stale"][i][p["stale"][i] >= 0].tolist():
            assert old < src and int(tokens[row, old - 1]) == key
        finals += int(p["negatives"][i] & et.NEG_BIT["negated"] != 0)
    assert finals > 0 and int(p["revised"].sum()) > 0


def test_scoped_labels_respect_scope_and_wrong_scope_declarations_are_marked():
    data = et.generate_scoped(et.ScopeTask(queries=12, wrong_scope=(1, 2)), 3, 2048, seed=5)
    p, tokens = data.probes, data.tokens
    wrong_seen = 0
    for i in range(p["row"].numel()):
        row, answer, src = int(p["row"][i]), int(p["answer"][i]), int(p["source"][i])
        assert int(tokens[row, src - 3]) == et.DEF and int(tokens[row, answer - 3]) == et.GET
        assert int(tokens[row, src - 2]) == int(tokens[row, answer - 2])          # same scope
        assert int(tokens[row, src - 1]) == int(tokens[row, answer - 1])          # same symbol
        assert int(tokens[row, src]) == int(tokens[row, answer])
        for w in p["wrong_scope"][i][p["wrong_scope"][i] >= 0].tolist():     # planted and background alike
            assert w < answer - 3 and int(tokens[row, w - 3]) == et.DEF
            assert int(tokens[row, w - 2]) != int(tokens[row, answer - 2])         # another scope
            assert int(tokens[row, w - 1]) == int(tokens[row, answer - 1])         # the same symbol
            wrong_seen += 1
    assert wrong_seen > 0


def test_generators_are_deterministic_and_seed_disjoint():
    a = et.generate_records(et.RecordTask(), 2, 1024, seed=3)
    b = et.generate_records(et.RecordTask(), 2, 1024, seed=3)
    c = et.generate_records(et.RecordTask(), 2, 1024, seed=4)
    assert torch.equal(a.tokens, b.tokens) and not torch.equal(a.tokens, c.tokens)
