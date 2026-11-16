"""The transformer: configuration, deterministic initialization, causality, cache parity, generation, gradients."""
from __future__ import annotations

import pytest

from .conftest import BASE, CHANNELS, SKIP_REASON, make_model

torch = pytest.importorskip("torch", reason=SKIP_REASON)

from research.elpis_torch import accounting  # noqa: E402
from research.elpis_torch.config import ConfigError, ModelConfig  # noqa: E402
from research.elpis_torch.model import (ContextLengthError, ElpisTorchLM, StaleCacheError,  # noqa: E402
                                        next_token_loss, parameter_digest, sample_next)
from research.elpis_torch.tokenizer import ByteTokenizer, TokenizerProtocol  # noqa: E402

NAMES = tuple(CHANNELS)
TOLERANCE = {"float64": 1e-10, "float32": 2e-5}     # declared prefill-versus-incremental tolerances


def _tokens(batch=3, length=40, seed=0):
    return torch.randint(0, BASE["vocab_size"], (batch, length), generator=torch.Generator().manual_seed(seed))


# -- configuration -----------------------------------------------------------------------------------------

@pytest.mark.parametrize("change", [
    dict(d_model=30, n_heads=4), dict(d_model=12, n_heads=4), dict(n_heads=4, n_kv_heads=3), dict(channel="moe"),
    dict(dtype="bfloat16"), dict(vocab_size=1), dict(n_layers=0), dict(max_seq_len=-1), dict(cgc_units=0),
    dict(cgc_preact_bound=0.0), dict(cgc_preact_bound=float("nan")), dict(cgc_init_gain=-1.0), dict(seed=-1),
    dict(d_model=32.0), dict(tie_embeddings=1), dict(ffn_hidden=0), dict(cgc_gate_init=float("inf")),
])
def test_inconsistent_configurations_are_refused_before_any_tensor_exists(change):
    with pytest.raises(ConfigError):
        ModelConfig(**{**BASE, **change})


def test_a_configuration_round_trips_and_has_a_stable_identity():
    config = ModelConfig(**BASE, **CHANNELS["C"])
    assert ModelConfig.from_dict(config.to_dict()) == config
    assert config.digest() == ModelConfig.from_dict(config.to_dict()).digest() != config.replace(seed=2).digest()
    with pytest.raises(ConfigError):
        ModelConfig.from_dict({**config.to_dict(), "expert_count": 4})
    assert (config.head_dim, config.kv_heads, config.has_ffn, config.has_cgc) == (16, 1, True, True)


# -- tokenizer ---------------------------------------------------------------------------------------------

def test_the_byte_tokenizer_is_reversible():
    tokenizer = ByteTokenizer()
    assert isinstance(tokenizer, TokenizerProtocol)
    every_byte = bytes(range(256)) * 2
    assert tokenizer.decode_bytes(tokenizer.encode_bytes(every_byte)) == every_byte
    text = "Elpis  ελπίς \U0001f331 a=b\n"
    ids = tokenizer.encode(text, bos=True, eos=True)
    assert ids[0] == tokenizer.BOS and ids[-1] == tokenizer.EOS and max(ids) < tokenizer.vocab_size
    assert tokenizer.decode(ids) == text
    assert tokenizer.decode(tokenizer.encode("ε")[:1]) == "�"      # a cut character is shown, not lost
    with pytest.raises(ValueError):
        tokenizer.decode_bytes([300])


# -- initialization ----------------------------------------------------------------------------------------

@pytest.mark.parametrize("name", NAMES)
def test_initialization_is_a_function_of_the_seed_alone(name):
    assert parameter_digest(make_model(name)) == parameter_digest(make_model(name))
    assert parameter_digest(make_model(name)) != parameter_digest(make_model(name, seed=2))
    double, single = make_model(name, dtype="float64"), make_model(name, dtype="float32")
    for (key, a), (_, b) in zip(double.named_parameters(), single.named_parameters()):
        assert torch.equal(a.float(), b), key


def test_the_augmented_block_starts_as_exactly_the_ffn_block():
    """C at gate 0 holds A's attention and FFN weights bit for bit and computes A's logits bit for bit."""
    a, c = make_model("A"), make_model("C")
    shared = dict(a.named_parameters())
    mine = dict(c.named_parameters())
    assert set(shared) < set(mine)
    assert all(torch.equal(shared[key], mine[key]) for key in shared)
    tokens = _tokens()
    assert torch.equal(a(tokens), c(tokens))
    assert not torch.equal(a(tokens), make_model("C", cgc_gate_init=0.5)(tokens))


@pytest.mark.parametrize("name", ["B", "C"])
def test_initial_pre_activations_have_the_configured_scale(name):
    model = make_model(name)
    x = torch.randn(64, BASE["d_model"], generator=torch.Generator().manual_seed(5))
    x = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True))            # unit RMS, as the block's norm delivers
    cgc = model.blocks[0].cgc
    stats = cgc.core.diagnostics(x if cgc.proj_in is None else cgc.proj_in(x))
    assert 0.3 < stats["preact_rms"] < 0.7 and stats["saturated_fraction"] == 0.0     # cgc_init_gain = 0.5


# -- forward -----------------------------------------------------------------------------------------------

@pytest.mark.parametrize("name", NAMES)
@pytest.mark.parametrize("heads,kv", [(2, 1), (2, 2), (4, 2), (4, 1)])
def test_logits_have_the_declared_shape_for_mha_and_gqa(name, heads, kv):
    model = make_model(name, n_heads=heads, n_kv_heads=kv)
    logits = model(_tokens())
    assert logits.shape == (3, 40, BASE["vocab_size"]) and bool(torch.isfinite(logits).all())
    assert model.blocks[0].attn.k.weight.shape[0] == kv * (BASE["d_model"] // heads)


def test_malformed_token_tensors_are_refused():
    model = make_model("B")
    for bad in (torch.zeros(3, dtype=torch.long), torch.zeros(1, 4), torch.full((1, 4), 259),
                torch.full((1, 4), -1), torch.zeros(1, 0, dtype=torch.long)):
        with pytest.raises(ValueError):
            model(bad)
    with pytest.raises(ContextLengthError):
        model(_tokens(1, BASE["max_seq_len"] + 1))


@pytest.mark.parametrize("name", NAMES)
def test_future_tokens_cannot_affect_earlier_logits(name):
    model = make_model(name, dtype="float64")
    tokens = _tokens(2, 24)
    reference = model(tokens)
    for cut in (1, 7, 23):
        changed = tokens.clone()
        changed[:, cut:] = (changed[:, cut:] + 1 + torch.arange(24 - cut)) % BASE["vocab_size"]
        out = model(changed)
        assert torch.equal(out[:, :cut], reference[:, :cut]), cut
        assert not torch.equal(out[:, cut:], reference[:, cut:])


@pytest.mark.parametrize("name", NAMES)
def test_the_jacobian_of_a_position_is_zero_on_every_later_position(name):
    model = make_model(name, dtype="float64")
    captured = {}
    handle = model.embed.register_forward_hook(lambda m, i, out: (out.retain_grad(), captured.update(e=out))[1])
    try:
        logits = model(_tokens(1, 16))
        for position in (0, 5, 14):
            model.zero_grad()
            if captured["e"].grad is not None:
                captured["e"].grad = None
            logits[0, position].sum().backward(retain_graph=True)
            grad = captured["e"].grad[0]
            assert float(grad[position + 1:].abs().max()) == 0.0
            assert float(grad[:position + 1].abs().sum()) > 0.0
    finally:
        handle.remove()


@pytest.mark.parametrize("name", NAMES)
def test_a_row_of_a_batch_equals_the_same_sequence_alone(name):
    model = make_model(name, dtype="float64")
    tokens = _tokens(4, 20)
    batched = model(tokens)
    for row in range(4):
        assert torch.allclose(batched[row], model(tokens[row:row + 1])[0], rtol=0, atol=1e-10)


# -- KV cache ----------------------------------------------------------------------------------------------

@pytest.mark.parametrize("name", NAMES)
@pytest.mark.parametrize("dtype", ["float64", "float32"])
@pytest.mark.parametrize("chunks", [(40,), (17,) + (1,) * 23, (1,) * 40, (5, 3, 9, 1, 22)])
def test_prefill_and_incremental_decoding_agree_with_the_batched_forward(name, dtype, chunks):
    model = make_model(name, dtype=dtype, n_heads=4, n_kv_heads=2)
    tokens = _tokens()
    full = model(tokens)
    cache, parts, at = model.new_cache(3), [], 0
    for size in chunks:
        parts.append(model(tokens[:, at:at + size], cache))
        at += size
    assert cache.length == 40 and len(cache.keys) == BASE["n_layers"] and cache.nbytes() > 0
    assert cache.keys[0].shape == (3, 2, 40, BASE["d_model"] // 4)
    worst = float((torch.cat(parts, dim=1) - full).detach().abs().max())
    assert worst <= TOLERANCE[dtype], worst


def test_a_cache_is_refused_for_other_weights_another_batch_or_an_overlong_sequence():
    model = make_model("B")
    tokens = _tokens(2, 10)
    cache = model.new_cache(2)
    model(tokens, cache)
    with pytest.raises(StaleCacheError):
        model(tokens[:, :1], cache, weights_tag="other weights")
    with pytest.raises(StaleCacheError):
        model(_tokens(3, 1), cache)
    with pytest.raises(StaleCacheError):
        make_model("B", n_layers=3)(tokens[:, :1], cache)
    keys = [k.clone() for k in cache.keys]
    with pytest.raises(ContextLengthError):
        model(_tokens(2, BASE["max_seq_len"] - 9), cache)
    # A refusal leaves the cache exactly as it was.
    assert cache.length == 10 and all(torch.equal(a, b) for a, b in zip(keys, cache.keys))
    assert not cache.keys[0].requires_grad
    cache.clear()
    assert cache.length == 0 and cache.nbytes() == 0


# -- generation --------------------------------------------------------------------------------------------

@pytest.mark.parametrize("name", NAMES)
def test_cached_generation_equals_recomputing_the_whole_prefix(name):
    model = make_model(name, dtype="float64").eval()
    prompt = _tokens(2, 9)
    cached = model.generate(prompt, 20)
    assert cached.shape == (2, 20) and cached.dtype == torch.long
    assert torch.equal(cached, model.generate(prompt, 20, use_cache=False))


def test_sampling_is_reproducible_from_its_generator():
    model, prompt = make_model("B").eval(), _tokens(2, 5)

    def draw(seed):
        return model.generate(prompt, 24, temperature=1.0, top_k=50, generator=torch.Generator().manual_seed(seed))

    assert torch.equal(draw(7), draw(7)) and not torch.equal(draw(7), draw(8))
    assert not torch.equal(draw(7), model.generate(prompt, 24))
    logits = torch.tensor([[0.0, 5.0, 1.0, 4.0]])
    assert int(sample_next(logits)) == 1
    picks = {int(sample_next(logits, temperature=1.0, top_k=2, generator=torch.Generator().manual_seed(s)))
             for s in range(40)}
    assert picks == {1, 3}
    with pytest.raises(FloatingPointError):
        sample_next(torch.tensor([[0.0, float("nan")]]))


def test_text_goes_in_and_text_comes_out():
    tokenizer, model = ByteTokenizer(), make_model("C").eval()
    prompt = torch.tensor([tokenizer.encode("Ka=", bos=True)])
    new = model.generate(prompt, 12)
    assert isinstance(tokenizer.decode(new[0].tolist()), str)
    with pytest.raises(ContextLengthError):
        model.generate(prompt, BASE["max_seq_len"])
    with pytest.raises(ValueError):
        model.generate(prompt, 0)


# -- backward ----------------------------------------------------------------------------------------------

@pytest.mark.parametrize("name", NAMES)
def test_every_parameter_receives_a_finite_gradient(name):
    model = make_model(name, cgc_gate_init=0.1)
    tokens = _tokens()
    next_token_loss(model(tokens[:, :-1]), tokens).backward()
    for key, parameter in model.named_parameters():
        assert parameter.grad is not None and bool(torch.isfinite(parameter.grad).all()), key
        assert float(parameter.grad.abs().sum()) > 0, key


def test_at_gate_zero_the_gate_learns_first_and_opens_the_cgc_path():
    model, tokens = make_model("C"), _tokens()
    next_token_loss(model(tokens[:, :-1]), tokens).backward()
    block = model.blocks[0]
    assert float(block.cgc_gate.grad.abs()) > 0 and float(block.cgc.core.weight.grad.abs().sum()) == 0
    with torch.no_grad():
        block.cgc_gate.fill_(1e-3)
    model.zero_grad()
    next_token_loss(model(tokens[:, :-1]), tokens).backward()
    assert float(block.cgc.core.weight.grad.abs().sum()) > 0


@pytest.mark.parametrize("variant", [dict(cgc_bottleneck=16), dict(cgc_bottleneck=None)])
@pytest.mark.parametrize("bound", [3.0, None])
def test_a_deep_stack_is_finite_and_its_gradients_do_not_explode_at_initialization(variant, bound):
    """Twelve stacked CGC blocks: the residual stream stays O(1) and every gradient is finite."""
    model = make_model("B", n_layers=12, cgc_preact_bound=bound, **variant)
    tokens = _tokens(2, 32)
    streams = []
    handles = [b.register_forward_hook(lambda m, i, out: streams.append(float(out[0].pow(2).mean().sqrt())))
               for b in model.blocks]
    loss = next_token_loss(model(tokens[:, :-1]), tokens)
    loss.backward()
    for handle in handles:
        handle.remove()
    assert all(0 < value < 1.0 for value in streams), streams         # embeddings start at RMS 0.02
    assert streams[-1] / streams[0] < 20
    norms = [float(p.grad.norm()) for p in model.parameters()]
    assert all(n == n and n < 1e3 for n in norms) and float(loss) < 2 * 5.56      # ~log(259) at initialization


def test_the_loss_counts_exactly_the_masked_targets():
    model, tokens = make_model("A"), _tokens(2, 12)
    logits = model(tokens[:, :-1])
    mask = torch.zeros(2, 11, dtype=torch.bool)
    mask[0, 3] = mask[1, 9] = True
    expected = -(torch.log_softmax(logits[0, 3], -1)[tokens[0, 4]] + torch.log_softmax(logits[1, 9], -1)[tokens[1, 10]]) / 2
    assert torch.allclose(next_token_loss(logits, tokens, mask), expected, rtol=1e-6)
    assert torch.allclose(next_token_loss(logits, tokens), next_token_loss(logits, tokens, torch.ones(2, 11, dtype=torch.bool)))
    with pytest.raises(ValueError):
        next_token_loss(logits, tokens, torch.zeros(2, 11, dtype=torch.bool))
    with pytest.raises(ValueError):
        next_token_loss(model(tokens), tokens)


# -- accounting --------------------------------------------------------------------------------------------

@pytest.mark.parametrize("name", NAMES)
def test_reported_parameter_counts_are_the_models(name):
    model = make_model(name)
    counts = accounting.parameter_counts(model)
    assert counts["total"] == sum(p.numel() for p in model.parameters() if p.requires_grad)
    per_block = accounting.channel_parameters(model.config)
    block = model.blocks[0]
    assert per_block["ffn"] == (sum(p.numel() for p in block.ffn.parameters()) if block.ffn else 0)
    assert per_block["cgc"] == (sum(p.numel() for p in block.cgc.parameters()) if block.cgc else 0)
    assert counts.get("ffn", 0) + counts.get("cgc.core", 0) + counts.get("cgc.proj", 0) + counts.get("cgc_gate", 0) \
        == BASE["n_layers"] * sum(per_block.values())


def test_the_reference_comparison_configurations_are_exactly_parameter_matched_but_not_compute_matched():
    """The experiment's A, B-dense and B-bottleneck: equal parameters; compute and memory still differ."""
    base = dict(vocab_size=259, d_model=64, n_layers=2, n_heads=4, max_seq_len=48)
    a = ModelConfig(**base, channel="ffn", ffn_hidden=172)
    dense = ModelConfig(**base, channel="cgc", cgc_channels=86, cgc_units=5)
    narrow = ModelConfig(**base, channel="cgc", cgc_channels=121, cgc_units=6, cgc_bottleneck=32)
    totals = {accounting.parameter_counts(ElpisTorchLM(c))["total"] for c in (a, dense, narrow)}
    assert totals == {115712}
    compute = [accounting.compute_per_token(c, 48) for c in (a, dense, narrow)]
    assert len({c["total"] for c in compute}) == 3
    assert all(c["per_block_attention"] == compute[0]["per_block_attention"] for c in compute)
    memory = [accounting.memory_estimate(c, batch=64, length=48) for c in (a, dense, narrow)]
    assert memory[0]["widest_channel_activation_per_token"] == 172
    assert memory[1]["widest_channel_activation_per_token"] == 430
    assert memory[2]["widest_channel_activation_per_token"] == 726
    assert accounting.session_memory(a)["fast_weight_scalars"] == 0
    assert accounting.session_memory(narrow)["fast_weight_scalars"] == 2 * 32 * 121 * 6
