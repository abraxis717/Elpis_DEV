from dataclasses import replace
from types import SimpleNamespace
import numpy as np
import pytest

from elpis.inference.associative import AddressScheme
from elpis.inference.drivers.dsv41.numerics import quant_dequant, hc_mixes, hc_post, rms, rotate, rope_frequencies
from elpis.inference.drivers.dsv41.moe import route, expert
from elpis.inference.drivers.dsv41.attention import candidate_mask, sparse_attention
from .donor_oracle import (load_model, load_engram, donor_args, TokenizerView, build_oracle,
                           stable_topk, torch_cache_quant, torch_hc)


def test_exact_donor_compressed_map_layout_and_hash(v41, tower_config, address_parameters):
    torch = pytest.importorskip("torch")
    donor, model = load_engram(), load_model()
    p = address_parameters
    args = donor_args(model, tower_config, p)
    layout = donor.EngramLayout.from_args(args)
    hasher = donor.NgramHashState(args, layout, TokenizerView(v41))
    assert tuple(hasher.token_map.tolist()) == p.token_map
    assert tuple(map(tuple, hasher.multipliers.tolist())) == p.multipliers
    assert tuple(map(tuple, hasher.offsets.tolist())) == p.offsets
    assert tuple(tuple(i for group in row for i in group) for row in layout.primes) == p.primes
    scheme = AddressScheme(p, expected_digest=p.digest, tokenizer=p.tokenizer, scheme=p.schema)
    tokens = v41.encode(" The the THE café CAFE 👩🏽‍💻 and Unicode test") * 2
    mask = tuple(i not in (0, 4, 7, 8, len(tokens)-1) for i in range(len(tokens)))
    expected = hasher(torch.tensor([tokens]), 0, torch.tensor([mask])).numpy()[0]
    streamed = scheme.stream_hash(scheme.initial(), tokens, mask)
    np.testing.assert_array_equal(expected, np.array(streamed.rows))
    history = scheme.initial()
    chunks = []
    for start, end in ((0, 3), (3, 11), (11, len(tokens))):
        result = scheme.stream_hash(history, tokens[start:end], mask[start:end])
        history = result.history
        chunks.extend(result.rows)
    np.testing.assert_array_equal(expected, chunks)


@pytest.mark.parametrize("mode", ("local", "compressed", "index"))
def test_cache_quantization_matches_independent_torch_kernel_equations(mode):
    torch = pytest.importorskip("torch")
    rng = np.random.default_rng(190)
    values = rng.normal(0, 3, (17, 64)).astype("<f4")
    values[0] = 0
    np.testing.assert_array_equal(quant_dequant(values, mode), torch_cache_quant(torch.tensor(values), mode).numpy())


@pytest.mark.parametrize("score", ("softmax", "sigmoid", "sqrtsoftplus"))
@pytest.mark.parametrize("k,normalize", ((1, True), (2, True), (2, False)))
def test_router_and_expert_match_original_donor(tower_config, score, k, normalize):
    torch = pytest.importorskip("torch")
    m = load_model()
    c = replace(tower_config, score_func=score, active_experts=k, norm_topk_prob=normalize, swiglu_limit=.4)
    rng = np.random.default_rng(310)
    x = rng.normal(size=c.dimension).astype("<f4")
    w = rng.normal(size=(c.expert_count, c.dimension)).astype("<f4")
    bias = np.array([-.5, .7, .1, -.3], dtype="<f4")
    gate = m.Gate.__new__(m.Gate)
    torch.nn.Module.__init__(gate)
    gate.weight, gate.bias = torch.nn.Parameter(torch.tensor(w)), torch.nn.Parameter(torch.tensor(bias))
    gate.gate_temp, gate.score_func, gate.topk = c.gate_temp, c.score_func, k
    gate.norm_topk_prob, gate.route_scale, gate.bias_vl = normalize, c.route_scale, None
    with torch.no_grad(), stable_topk():
        weights, chosen = gate(torch.tensor(x)[None])
    actual_ids, actual_weights = route(x, w, bias, c)
    np.testing.assert_array_equal(actual_ids, chosen.numpy()[0])
    np.testing.assert_allclose(actual_weights, weights.numpy()[0], rtol=2e-6, atol=2e-7)
    donor = m.Expert(c.dimension, c.expert_dim, dtype=torch.float32, swiglu_limit=c.swiglu_limit)
    tensors = [rng.normal(size=s).astype("<f4") for s in ((c.expert_dim,c.dimension),
               (c.expert_dim,c.dimension), (c.dimension,c.expert_dim))]
    with torch.no_grad():
        for module, array in zip((donor.w1,donor.w3,donor.w2), tensors):
            module.weight.copy_(torch.tensor(array))
        expected = donor(torch.tensor(x), torch.tensor(.37)).numpy()
    np.testing.assert_allclose(expert(x, tensors, .37, c.swiglu_limit), expected, rtol=2e-6, atol=2e-6)


def test_hyperconnection_orientation_and_norms(tower_config):
    torch = pytest.importorskip("torch")
    m = load_model()
    c = tower_config
    rng = np.random.default_rng(98)
    x = rng.normal(size=(c.hc_mult, c.dimension)).astype("<f4")
    fn = rng.normal(size=((2+c.hc_mult)*c.hc_mult, c.hc_mult*c.dimension)).astype("<f4")
    scale = np.array([.1,.2,.3], dtype="<f4")
    base = rng.normal(size=fn.shape[0]).astype("<f4")
    block = m.Block.__new__(m.Block)
    torch.nn.Module.__init__(block)
    block.norm_eps, block.hc_eps, block.hc_mult = c.norm_eps, c.hc_eps, c.hc_mult
    block.hc_sinkhorn_iters = c.hc_sinkhorn_iters
    expected = block.hc_mixes(torch.tensor(x)[None,None], torch.tensor(fn), torch.tensor(scale), torch.tensor(base))
    actual = hc_mixes(x, fn, scale, base, c)
    for a, b in zip(actual, expected):
        np.testing.assert_allclose(a, b.numpy()[0,0], rtol=3e-6, atol=1e-7)
    output = rng.normal(size=c.dimension).astype("<f4")
    donor = block.hc_post(torch.tensor(output)[None,None], torch.tensor(x)[None,None], expected[1], expected[2])
    np.testing.assert_allclose(hc_post(output, x, actual[1], actual[2]), donor.numpy()[0,0], rtol=3e-6, atol=1e-6)
    norm = m.RMSNorm(c.dimension, c.norm_eps)
    with torch.no_grad():
        np.testing.assert_allclose(rms(x, np.ones(c.dimension,dtype="<f4"), c.norm_eps),
                                   norm(torch.tensor(x)).numpy(), rtol=2e-6, atol=1e-7)


def test_full_seven_layer_donor_prefill_decode_and_per_layer_streams(tower, v41):
    torch = pytest.importorskip("torch")
    target, _ = tower
    model, module = build_oracle(target, v41)
    tokens = v41.encode("Hello, Elpis. Multi layer attention and memory exercise.") * 2
    # Cross partial/full compressor groups and wrap the local ring repeatedly.
    cut = 9
    state, experts = target.window_initial(), target.admit_stream()
    for token in tokens[:cut]:
        target.window_step(state, token, experts=experts)
    layer_outputs = []
    hooks = [layer.register_forward_hook(lambda _m, _a, out: layer_outputs.append(out[0].detach().numpy().copy()))
             for layer in model.layers]
    with torch.no_grad(), stable_topk():
        _, logits, _ = model(torch.tensor([tokens[:cut]]), 0)
        np.testing.assert_allclose(state.logits, logits.numpy()[0], rtol=1e-4, atol=1e-5)
        for i, output in enumerate(layer_outputs):
            np.testing.assert_allclose(state.layer_streams[i], output[0,-1], rtol=1e-4, atol=1e-5)
        for position, token in enumerate(tokens[cut:], cut):
            layer_outputs.clear()
            target.window_step(state, token, experts=experts)
            _, logits, _ = model(torch.tensor([[token]]), position)
            for i, output in enumerate(layer_outputs):
                np.testing.assert_allclose(state.layer_streams[i], output[0,0], rtol=1e-4, atol=1e-5,
                                           err_msg=f"position={position} layer={i}")
            np.testing.assert_allclose(state.logits, logits.numpy()[0], rtol=1e-4, atol=1e-5)
            assert int(np.argmax(state.logits)) == int(torch.argmax(logits))
    for hook in hooks:
        hook.remove()
