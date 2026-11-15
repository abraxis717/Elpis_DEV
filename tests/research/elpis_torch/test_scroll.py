"""R1 scrolling context: the rings, eviction and positions against an independent per-token oracle.

The oracle below shares no code with ``scroll.py``'s ring arithmetic. It keeps every key, value and gate of the
whole stream in dictionaries keyed by absolute position, and decides what a query may see from the written
definition alone. If the preallocated, chunked, wrapping implementation agrees with it over a stream several
horizons long, the slot arithmetic, the eviction order and the position handling are right.
"""
from __future__ import annotations

import math

import pytest

from .conftest import SKIP_REASON

torch = pytest.importorskip("torch", reason=SKIP_REASON)

from research.elpis_torch import accounting  # noqa: E402
from research.elpis_torch.config import ConfigError, ModelConfig  # noqa: E402
from research.elpis_torch.model import ElpisTorchLM, next_token_loss  # noqa: E402
from research.elpis_torch.scroll import ChunkError, EvictedError, ScrollConfig, ScrollLM  # noqa: E402
from research.elpis_torch import scroll_data as task  # noqa: E402
from research.elpis_torch.scroll_data import (ASSOC_Q, DIRECT, HOP1, HOP2, SPAN_Q, generate,  # noqa: E402
                                               probe_loss, score)

WIDTHS = dict(vocab_size=259, d_model=32, n_layers=2, n_heads=4, n_kv_heads=2, max_seq_len=512, channel="cgc",
              cgc_bottleneck=8, cgc_channels=16, dtype="float64", seed=1)
SHAPE = dict(window=48, block=8, chunk=32, horizon=128, global_topk=4, fine_blocks=2, sinks=3)


def _model(global_layers=(0, 1), sharpen=True, **changes):
    shape = {**SHAPE, **{k: v for k, v in changes.items() if k in SHAPE or k == "memory_keys"}}
    widths = {**WIDTHS, **{k: v for k, v in changes.items() if k in WIDTHS}}
    model = ScrollLM(ScrollConfig(ModelConfig(**widths), global_layers=global_layers, **shape)).eval()
    if sharpen:      # make the memory logits large enough that what gets selected decides the output
        with torch.no_grad():
            for name, parameter in model.named_parameters():
                if ".attn.g" in name:
                    parameter.mul_(20.0)
                elif name.endswith((".attn.q.weight", ".attn.k.weight")) and shape.get("memory_keys") != "separate":
                    parameter.mul_(6.0)                     # shared maps: the window's own query and key
    return model


def _tokens(batch=2, length=300, seed=0):
    return torch.randint(0, 259, (batch, length), generator=torch.Generator().manual_seed(seed))


def _rotate(vector, steps, theta, e):
    angle = steps * theta ** (-torch.arange(0, e, 2, dtype=torch.float64) / e)
    even, odd = vector[..., 0::2], vector[..., 1::2]
    return torch.stack((even * angle.cos() - odd * angle.sin(), even * angle.sin() + odd * angle.cos()), -1).flatten(-2)


@torch.no_grad()
def oracle(model, tokens, origin=0):
    """Logits of one stream, one token at a time, from the definition of what a query may see."""
    c, m = model.scroll, model.config
    e, h, g, W, s, M = m.head_dim, m.n_heads, m.kv_heads, c.window, c.block, c.slots
    wide = lambda t: t.repeat_interleave(h // g, dim=0)     # noqa: E731
    stores = [dict(k={}, v={}, gk={}, gate={}) for _ in model.blocks]
    out = []
    for i, token in enumerate(tokens.tolist()):
        p, x = origin + i, model.embed.weight[token]
        for block, store in zip(model.blocks, stores):
            a, xn = block.attn, block.attn_norm(x)
            q, store["k"][p], store["v"][p] = a.q(xn).view(h, e), a.k(xn).view(g, e), a.v(xn).view(g, e)
            logits, values = [], []
            for key in range(max(origin, p - W + 1), p + 1):                       # the window
                logits.append((_rotate(q, p - key, m.rope_theta, e) * wide(store["k"][key])).sum(-1))
                values.append(wide(store["v"][key]))
            for j in range(c.sinks):                                              # pinned first tokens
                if origin + j <= p - W:
                    logits.append((_rotate(q, W + c.sinks - j, m.rope_theta, e) * wide(store["k"][origin + j])).sum(-1))
                    values.append(wide(store["v"][origin + j]))
            if a.is_global:
                gq, slope = (a.gq(xn).view(h, e) if a.separate else q), torch.nn.functional.softplus(a.recency)
                store["gk"][p], store["gate"][p] = (a.gk(xn).view(g, e) if a.separate else store["k"][p]), a.gate(xn)
                reachable = []
                for b in range(origin // s, p // s):
                    end = b * s + s - 1
                    if end <= p - W and b >= p // s - M:                          # behind the window, in the horizon
                        span = range(b * s, end + 1)
                        pool = torch.softmax(torch.stack([store["gate"][t] for t in span], -1), -1)   # [g, s]
                        ck = (pool[..., None] * torch.stack([store["gk"][t] for t in span], 1)).sum(1)
                        cv = (pool[..., None] * torch.stack([store["v"][t] for t in span], 1)).sum(1)
                        logit = (gq * wide(ck)).sum(-1) - slope * math.log(p - end) * e ** 0.5
                        reachable.append((b, logit, wide(cv)))
                if reachable:
                    order = torch.stack([logit.max() for _, logit, _ in reachable]).topk(
                        min(c.global_topk, len(reachable))).indices.tolist()
                    for rank, pick in enumerate(order):
                        b, logit, cv = reachable[pick]
                        if rank < c.fine_blocks:                                  # read as raw tokens
                            for t in range(b * s, b * s + s):
                                logits.append((gq * wide(store["gk"][t])).sum(-1) - slope * math.log(p - t) * e ** 0.5)
                                values.append(wide(store["v"][t]))
                        else:                                                     # read as one compressed token
                            logits.append(logit)
                            values.append(cv)
            weights = torch.softmax(torch.stack(logits, -1) * e ** -0.5, -1)        # [h, keys]
            x = x + a.out((weights[..., None] * torch.stack(values, 1)).sum(1).reshape(-1))
            x = x + block.channel(block.channel_norm(x))
        out.append(model.final_norm(x) @ model.embed.weight.T)
    return torch.stack(out)


# -- configuration -----------------------------------------------------------------------------------------

@pytest.mark.parametrize("change", [dict(window=30), dict(horizon=100), dict(horizon=40), dict(block=0),
                                    dict(fine_blocks=9), dict(sinks=-1), dict(chunk=0), dict(global_topk=0)])
def test_context_shapes_that_could_lose_tokens_are_refused(change):
    with pytest.raises(ConfigError):
        ScrollConfig(ModelConfig(**WIDTHS), global_layers=(1,), **{**SHAPE, **change})
    for layers in ((2,), (0, 0), (1.0,)):
        with pytest.raises(ConfigError):
            ScrollConfig(ModelConfig(**WIDTHS), global_layers=layers, **SHAPE)
    assert ScrollConfig(ModelConfig(**WIDTHS), **SHAPE).topology == "local"
    assert ScrollConfig(ModelConfig(**WIDTHS), global_layers=[1], **SHAPE).topology == "sparse_global"


# -- exactness where exactness is claimed ------------------------------------------------------------------

def test_with_everything_in_the_window_the_model_is_the_r0_dense_model():
    """Local topology, window >= sequence: same weights (same names, same seed), same logits as R0."""
    scroll = ScrollLM(ScrollConfig(ModelConfig(**WIDTHS), window=400, block=16, chunk=64, horizon=4096)).eval()
    dense = ElpisTorchLM(ModelConfig(**WIDTHS)).eval()
    shared = dict(dense.named_parameters())
    assert all(torch.equal(p, shared[name]) for name, p in scroll.named_parameters())
    tokens = _tokens()
    with torch.no_grad():
        assert float((scroll.run(tokens)[0] - dense(tokens)).abs().max()) < 1e-10


@pytest.mark.parametrize("keys", ["shared", "separate"])
@pytest.mark.parametrize("layers,sinks,origin", [((0, 1), 3, 0), ((1,), 0, 0), ((0, 1), 3, 1 << 20), ((), 2, 64)])
def test_the_chunked_ring_implementation_equals_the_per_token_oracle(layers, sinks, origin, keys):
    """300 tokens through a 128-token horizon: 2.3 horizons of wrap-around, eviction and re-selection."""
    model = _model(layers, sinks=sinks, memory_keys=keys)
    tokens = _tokens(1, 300, seed=3)
    with torch.no_grad():
        got = model.run(tokens, model.new_state(1, origin=origin))[0][0]
    assert float((got - oracle(model, tokens[0], origin)).abs().max()) < 1e-9


def test_the_oracle_itself_depends_on_what_the_memory_selects():
    """Guard against a vacuous comparison: without memory the same stream gives different logits."""
    tokens = _tokens(1, 200, seed=3)
    with_memory, without = _model((0, 1)), _model(())
    with torch.no_grad():
        assert float((with_memory.run(tokens)[0] - without.run(tokens)[0])[:, 100:].abs().max()) > 1e-3


@pytest.mark.parametrize("chunk", [1, 7, 31])
def test_the_chunk_size_is_an_execution_detail(chunk):
    model, tokens = _model(), _tokens()
    with torch.no_grad():
        assert float((model.run(tokens)[0] - model.run(tokens, chunk=chunk)[0]).abs().max()) < 1e-10


@pytest.mark.parametrize("origin", [8, (1 << 20) - 160, (1 << 20) - 8, 1 << 20, (1 << 40) + 8, (1 << 62) - 8 * 1000])
def test_logits_do_not_depend_on_where_in_the_stream_the_window_sits(origin):
    """Bitwise: across the 1M boundary and at positions far beyond what a float32 could represent exactly."""
    model, tokens = _model(), _tokens()
    with torch.no_grad():
        reference, here = model.run(tokens)[0], model.run(tokens, model.new_state(2, origin=origin))
    assert torch.equal(here[0], reference)
    assert here[1].position == origin + 300 and here[1].coordinates()["tokens_seen"] == 300
    with pytest.raises(ValueError):
        model.new_state(1, origin=3)                       # not on a block boundary


def test_float32_and_float64_agree():
    tokens = _tokens()
    with torch.no_grad():
        double, single = _model().run(tokens)[0], _model(dtype="float32").run(tokens)[0]
    assert float((double - single.double()).abs().max()) < 5e-4


# -- bounded state, deterministic eviction, explicit coordinates -------------------------------------------

def test_the_state_is_allocated_once_and_never_grows():
    model, tokens = _model(), _tokens(2, 700)
    state = model.new_state(2)
    sizes = state.nbytes()
    assert sizes == accounting.scroll_state_bytes(model.scroll, batch=2)
    pointers = [(name, tensor.data_ptr(), tuple(tensor.shape)) for layer in state.layers for name, tensor in layer.items()]
    with torch.no_grad():
        model.run(tokens, state)                           # 5.5 horizons
    assert state.nbytes() == sizes
    assert pointers == [(name, tensor.data_ptr(), tuple(tensor.shape)) for layer in state.layers
                        for name, tensor in layer.items()]
    assert sizes["fine"] > sizes["compressed"] > sizes["open_block"] > 0 and sizes["sinks"] > 0
    local = _model(()).new_state(2).nbytes()
    assert set(local) == {"token_history", "window", "sinks", "total"}


def test_the_four_coordinate_systems_are_reported_separately_and_eviction_is_fifo():
    model, tokens = _model(sinks=0), _tokens(1, 700)
    state = model.new_state(1)
    with torch.no_grad():
        model.run(tokens[:, :100], state)
    assert state.coordinates() == {"position": 100, "tokens_seen": 100, "window": (52, 100), "blocks_resident": 12,
                                   "oldest_resident_block": 0, "oldest_reachable_position": 0,
                                   "oldest_addressable_token": 0, "evicted_blocks": 0}
    with torch.no_grad():
        model.run(tokens[:, 100:], state)
    coordinates = state.coordinates()
    assert coordinates["position"] == 700 and coordinates["window"] == (652, 700)
    assert coordinates["blocks_resident"] == 16 == model.scroll.slots       # horizon / block
    assert coordinates["oldest_resident_block"] == 87 - 16 and coordinates["evicted_blocks"] == 87 - 16
    assert coordinates["oldest_reachable_position"] == (87 - 16) * 8
    assert coordinates["oldest_addressable_token"] == 700 - 136
    # The token history is addressable by absolute position, and refuses what it no longer holds.
    for position in (699, 652, 700 - 136):
        assert int(state.token_at(position)) == int(tokens[0, position])
    for position in (700 - 137, 0, 700, -1):
        with pytest.raises(EvictedError):
            state.token_at(position)
    # Slot b mod M holds exactly the newest block congruent to it: block 86 lives in slot 86 % 16, and the
    # compressed key there is that block's pooled key, not an older one.
    fresh = model.new_state(1)
    with torch.no_grad():
        model.run(tokens, fresh, chunk=1)
    live = 700 % 8                                          # tokens in the open block; the rest of it is dead
    for layer, other in zip(state.layers, fresh.layers):
        for name in layer:
            a, b = (layer[name][:, :, :live], other[name][:, :, :live]) if name[0] == "p" else (layer[name], other[name])
            assert torch.allclose(a, b, rtol=0, atol=1e-12), name


def test_a_query_never_reads_a_block_outside_its_horizon_or_inside_its_window():
    model, tokens = _model(sinks=0), _tokens(1, 700)
    state = model.new_state(1)
    s, W, M = 8, 48, 16
    reads = 0
    with torch.no_grad():
        for at in range(0, 700, 32):
            _, chosen = model(tokens[:, at:at + 32], state, return_selected=True)
            for picked in chosen.values():
                for offset in range(picked.shape[1]):
                    p = at + offset
                    for block in picked[0, offset].tolist():
                        if block >= 0:
                            reads += 1
                            assert block * s + s - 1 <= p - W and block >= p // s - M
    assert reads > 4000
    # With nothing reachable yet, nothing is selected.
    with torch.no_grad():
        _, chosen = model(tokens[:, :32], model.new_state(1), return_selected=True)
    assert chosen == {}


def test_local_only_forgets_and_sparse_global_does_not():
    """Change one token far behind the window: the local topology cannot tell, the sparse-global one can."""
    tokens = _tokens(1, 260)
    changed = tokens.clone()
    changed[0, 20] = (changed[0, 20] + 7) % 259
    local, sparse = _model((), sinks=0), _model((1,), sinks=0)
    with torch.no_grad():
        # Two layers of a 48-token window reach at most 94 tokens back.
        assert torch.equal(local.run(tokens)[0][:, 20 + 95:], local.run(changed)[0][:, 20 + 95:])
        difference = (sparse.run(tokens)[0] - sparse.run(changed)[0]).abs()
        assert float(difference[:, 20 + 95:200].max()) > 1e-6
        # ... and only within the horizon. Layer 0 is local, so the change reaches layer 1's keys and values up
        # to position 67 (block 8). A query at p reads blocks >= p // 8 - 16, so from p = 200 on it is gone.
        assert float(difference[:, 200:].max()) == 0.0
        # With memory in BOTH layers the reach is not one horizon: layer 1 remembers what layer 0 remembered.
        both = _model((0, 1), sinks=0)
        assert float((both.run(tokens)[0] - both.run(changed)[0])[:, 200:].abs().max()) > 0.0


def test_an_oversized_chunk_is_refused_whole_and_nothing_is_truncated():
    model, tokens = _model(), _tokens(2, 80)
    state = model.new_state(2)
    with torch.no_grad():
        model(tokens[:, :10], state)
        before = [{k: v.clone() for k, v in layer.items()} for layer in state.layers]
        for bad in (tokens[:, :33], tokens[:, :0], tokens[:1, :4], torch.full((2, 4), 259), tokens[:, :4].float()):
            with pytest.raises((ChunkError, ValueError)):
                model(bad, state)
        with pytest.raises(ValueError):
            model(tokens[:, :4], _model(horizon=256).new_state(2))
    assert state.position == 10
    assert all(torch.equal(layer[k], kept[k]) for layer, kept in zip(state.layers, before) for k in kept)


# -- training path -----------------------------------------------------------------------------------------

def test_the_training_path_is_the_inference_path_and_every_memory_parameter_learns():
    model, tokens = _model(sharpen=False, horizon=512), _tokens(2, 200)
    with torch.no_grad():
        inference = model.run(tokens)[0]
    logits = model.run(tokens)[0]                           # graph attached: buffers are written out of place
    assert float((logits - inference).abs().max()) < 1e-12
    next_token_loss(logits[:, :-1], tokens).backward()
    for name, parameter in model.named_parameters():
        assert parameter.grad is not None and bool(torch.isfinite(parameter.grad).all()), name
        if any(key in name for key in (".attn.q.", ".attn.k.", ".attn.gate", ".attn.recency", "cgc.core")):
            assert float(parameter.grad.abs().sum()) > 0, name
    assert float(torch.nn.functional.softplus(model.blocks[1].attn.recency).min()) > 0


@pytest.mark.parametrize("bound", [3.0, None])
def test_cgc_stays_finite_in_a_long_scrolling_stream(bound):
    """The CGC channel sees RMS-normalized input in this topology too: 4,000 tokens, wrapped 30 times."""
    model = _model(cgc_preact_bound=bound, dtype="float32")
    state = model.new_state(2)
    worst = 0.0
    with torch.no_grad():
        for seed in range(125):
            logits = model(_tokens(2, 32, seed=seed), state)
            assert bool(torch.isfinite(logits).all())
            worst = max(worst, float(logits.abs().max()))
    assert state.position == 4000 and worst < 50


# -- the task ----------------------------------------------------------------------------------------------

def test_the_long_stream_task_is_consistent_and_deterministic():
    streams = generate(3, 4096, events=96, seed=5, span=True)
    again = generate(3, 4096, events=96, seed=5, span=True)
    assert torch.equal(streams.tokens, again.tokens) and not torch.equal(
        streams.tokens, generate(3, 4096, events=96, seed=6, span=True).tokens)
    assert int(streams.tokens.max()) < task.VOCAB and int(streams.tokens[0, 0]) == task.BOS
    assert {int(k) for k in streams.kind.unique()} == {0, DIRECT, ASSOC_Q, HOP1, HOP2, SPAN_Q}

    def fact(token):                                        # a pair token -> (key, value)
        assert task.PAIR0 <= token < task.ASK0
        return divmod(token - task.PAIR0, task.VALUE_COUNT)

    rows, cols = torch.nonzero(streams.kind != 0, as_tuple=True)
    checked = contradicted = 0
    for row, col in zip(rows.tolist(), cols.tolist()):
        kind, source = int(streams.kind[row, col]), col - int(streams.distance[row, col])
        text = streams.tokens[row].tolist()
        answer, question = text[col], text[col - 1]
        if kind == HOP1:                                    # the alias fact names the key the model must emit
            symbol, key = divmod(text[source] - task.ALIAS0, task.KEYS)
            assert question == task.HOP0 + symbol and answer == task.ASK0 + key
            later = [t for t in range(source + 1, col) if task.ALIAS0 <= text[t] < task.HOP0
                     and (text[t] - task.ALIAS0) // task.KEYS == symbol]
            assert not later
            checked += 1
            continue
        key, value = fact(text[source])
        assert answer == task.VALUE0 + value
        if kind == ASSOC_Q:                                 # asked with a token of another family
            assert question == task.ASSOC_OF[key] and not task.ASK0 <= question < task.ASSOC0
        else:
            assert question == task.ASK0 + key
        if kind == HOP2:
            assert int(streams.kind[row, col - 1]) == HOP1
        # The latest fact for the key wins, and the flag marks keys that had held another value before.
        facts = [t for t in range(col) if task.PAIR0 <= text[t] < task.ASK0 and fact(text[t])[0] == key]
        assert facts[-1] == source
        if any(fact(text[t])[1] != value for t in facts[:-1]):
            assert bool(streams.contradicted[row, col])
        contradicted += bool(streams.contradicted[row, col])
        assert question != text[source]                     # never a lexical match
        checked += 1
    assert checked > 100 and contradicted > 10
    span = streams.distance[streams.kind == SPAN_Q]
    assert int(span.min()) > 0.9 * 4096                    # whole-stream probes
    with pytest.raises(ValueError):
        generate(1, 256, events=96, seed=0)


def test_scoring_reports_kinds_distances_and_contradictions():
    streams = generate(2, 2048, events=64, seed=1, span=True)
    perfect = torch.cat((streams.tokens[:, 1:], streams.tokens[:, :1]), dim=1)
    report = score(perfect, torch.zeros(2, 2048), streams, edges=(64, 512))
    assert set(report) == {"direct", "assoc", "hop1", "hop2", "span"}
    assert all(entry["accuracy"] == 1.0 and entry["loss"] == 0.0 for entry in report.values())
    assert sum(b["n"] for b in report["direct"]["by_distance"].values()) == report["direct"]["n"]
    assert report["direct"]["contradicted"]["n"] + report["direct"]["uncontradicted"]["n"] == report["direct"]["n"]
    wrong = score(torch.zeros_like(perfect), torch.ones(2, 2048), streams)
    assert all(entry["accuracy"] == 0.0 for entry in wrong.values())
    logits = torch.zeros(2, 2047, task.VOCAB)
    assert float(probe_loss(logits, streams.tokens, streams.kind)) == pytest.approx(1.1 * math.log(task.VOCAB), rel=1e-5)


# -- capacity model ----------------------------------------------------------------------------------------

def test_the_capacity_model_separates_gpu_cpu_and_immutable_resources():
    on_cpu, on_gpu = accounting.deployment_projection(), accounting.deployment_projection(fine_on_gpu=True)
    assert on_cpu["status"] == on_gpu["status"] == "PROJECTED"
    assert "fine" in on_cpu["cpu_ram"] and "fine" not in on_cpu["gpu_resident"] and "fine" in on_gpu["gpu_resident"]
    assert on_cpu["fits_gpu"] and on_gpu["fits_gpu"] and on_gpu["gpu_resident"]["total"] < 24 << 30
    assert on_cpu["immutable_persistent"]["checkpoint"] == on_cpu["gpu_resident"]["weights"]
    # A dense KV cache of the same model at the same horizon does not fit: that is the reason for the design.
    assert on_cpu["dense_kv_cache_at_horizon_for_comparison"] > 24 << 30
    assert not accounting.deployment_projection(fine_on_gpu=True, memory_layers=24)["fits_gpu"]
    compute = accounting.scroll_compute_per_token(_model().scroll)
    assert compute["dense_reference_at_horizon"] > compute["local_layer"] > compute["memory_selected"] > 0
    assert accounting.scroll_compute_per_token(_model().scroll, resident_blocks=4)["memory_scoring"] \
        < compute["memory_scoring"]
