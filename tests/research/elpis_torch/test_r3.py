"""R3: exact chunked ERSA kernel, device/dtype placement, token history, learned addressing, isolation of labels."""
from __future__ import annotations

import pytest

from .conftest import SKIP_REASON

torch = pytest.importorskip("torch", reason=SKIP_REASON)

from torch.nn import functional as F  # noqa: E402

from research.elpis_torch import longrange_tasks as facts  # noqa: E402
from research.elpis_torch import program_tasks as programs  # noqa: E402
from research.elpis_torch.config import ConfigError, ModelConfig  # noqa: E402
from research.elpis_torch.ersa import ERSAMixer, chunked_scan, serial_scan  # noqa: E402
from research.elpis_torch.ersa_model import ErsaConfig, ErsaLM  # noqa: E402
from research.elpis_torch.history import BlockIndex, EvictedError, TokenHistory  # noqa: E402
from research.elpis_torch.history_model import ErsaHistoryLM, HistoryConfig, addressing_loss  # noqa: E402
from research.elpis_torch.lastblock import last_block_logits  # noqa: E402

F64 = torch.float64
WIDTHS = dict(vocab_size=64, d_model=32, n_layers=2, n_heads=4, n_kv_heads=2, max_seq_len=64, channel="cgc",
              cgc_bottleneck=8, cgc_channels=16, dtype="float64", seed=1)
HISTORY = dict(block=8, horizon=64, topk=2, index_dim=8, read_head_dim=8, overlap=3)
CUDA = torch.cuda.is_available()
# R4.1: the CUDA tests run only on the device named here (default cuda:0); nothing else on the GPU is touched.
CUDA_DEVICE = __import__("os").environ.get("ELPIS_TORCH_CUDA_DEVICE", "cuda:0")
NOT_RUN_CUDA = "NOT_RUN: no CUDA device in this environment (not a PASS)"


def _signals(batch=2, length=150, heads=3, key=8, value=5, seed=0, dtype=F64):
    g = torch.Generator().manual_seed(seed)
    q = F.normalize(torch.randn(batch, length, heads, key, generator=g, dtype=dtype), dim=-1)
    k = F.normalize(torch.randn(batch, length, heads, key, generator=g, dtype=dtype), dim=-1)
    v = torch.randn(batch, length, heads, value, generator=g, dtype=dtype)
    r, e, w = (torch.rand(batch, length, heads, generator=g, dtype=dtype) for _ in range(3))
    r[:, 10], r[:, 77], w[:, 5], e[:, 6] = 0.0, 0.0, 0.0, 1.0          # exact wipes, no write, full erase
    s0 = torch.randn(batch, heads, value, key, generator=g, dtype=dtype)
    return s0, q, k, v, r, e, w


def _model(history=True, window=False, **h):
    extra = dict(attention="window", attn_window=80, attn_layers=(1,), chunk=64) if window else dict(chunk=256)
    config = ErsaConfig(ModelConfig(**WIDTHS), ersa_heads=2, key_dim=8, value_dim=8, conv_kernel=3,
                        ersa_kernel="chunked", **extra)
    model = ErsaHistoryLM(config, HistoryConfig(**{**HISTORY, **h})) if history else ErsaLM(config)
    return model.eval()


def _tokens(batch=2, length=200, seed=0, vocab=64):
    return torch.randint(0, vocab, (batch, length), generator=torch.Generator().manual_seed(seed))


def _stream(model, tokens, step, **extra):
    state, outs, picks = model.new_state(tokens.shape[0]), [], []
    with torch.no_grad():
        for at in range(0, tokens.shape[1], step):
            piece = {k: (v[:, at:at + step] if k == "address" else v) for k, v in extra.items()}
            out = model(tokens[:, at:at + step], state, diagnostics=True, **piece)
            outs.append(out[0])
            if model.read_layers[-1] in out[1]:
                picks.append(out[1][model.read_layers[-1]]["selected"])
    return torch.cat(outs, 1), torch.cat(picks, 1) if picks else None, state


# -- P3: the chunked kernel is the serial recurrence ----------------------------------------------------------

@pytest.mark.parametrize("block", [1, 7, 64, 200])
def test_the_chunked_scan_equals_the_serial_reference_including_wipes(block):
    s0, *rest = _signals()
    reads, state = serial_scan(s0, *rest)
    got, got_state = chunked_scan(s0, *rest, block=block)
    assert float((reads - got).abs().max()) < 1e-12 and float((state - got_state).abs().max()) < 1e-12


def test_the_chunked_scan_has_the_serial_gradients():
    s0, *rest = _signals(length=96)
    grads = []
    for scan in (serial_scan, lambda *a: chunked_scan(*a, block=32)):
        inputs = [t.clone().requires_grad_(True) for t in (s0, *rest)]
        reads, state = scan(*inputs)
        (reads.sin().sum() + state.cos().sum()).backward()
        grads.append([t.grad for t in inputs])
    for a, b in zip(*grads):
        assert float((a - b).abs().max()) < 1e-11


def test_the_chunked_scan_in_float32_stays_within_the_declared_tolerance():
    s0, *rest = _signals(dtype=torch.float32, length=256)
    reads, state = serial_scan(s0, *rest)
    got, got_state = chunked_scan(s0, *rest, block=64)
    assert float((reads - got).abs().max()) < 1e-4 and float((state - got_state).abs().max()) < 1e-4


@pytest.mark.parametrize("window", [False, True])
def test_a_chunked_kernel_model_equals_the_serial_reference_model(window):
    chunked = _model(history=False, window=window)
    config = chunked.ersa.__class__(**{**chunked.ersa.__dict__, "ersa_kernel": "serial"})
    reference = ErsaLM(config).eval()
    reference.load_state_dict(chunked.state_dict())
    tokens = _tokens(length=128)
    with torch.no_grad():
        a = chunked.run(tokens)[0]
        b = reference.run(tokens)[0]
    assert float((a - b).abs().max()) < 1e-10


def test_ersa_kernel_and_attention_placement_are_validated():
    m = ModelConfig(**WIDTHS)
    with pytest.raises(ConfigError):
        ErsaConfig(m, ersa_kernel="fast")
    with pytest.raises(ConfigError):
        ErsaConfig(m, attn_layers=(0,))                       # placement without attention
    with pytest.raises(ConfigError):
        ErsaConfig(m, attention="window", attn_window=80, attn_layers=(2,), chunk=64)
    model = ErsaLM(ErsaConfig(m, attention="window", attn_window=80, attn_layers=(1,), chunk=64))
    assert model.blocks[0].attn is None and model.blocks[1].attn is not None
    assert len(model.new_state(1).window.layers) == 1           # one ring per attention-carrying layer only


# -- P1: the exact token history and the bounded index -------------------------------------------------------

def test_token_history_reads_exactly_evicts_deterministically_and_refuses_out_of_range():
    history = TokenHistory(2, 10, origin=4)
    stream = torch.arange(4, 40).view(1, -1).repeat(2, 1)
    stream[1] += 100                                            # two streams: different contents
    for at in range(0, 36, 7):
        history.append(stream[:, at:at + 7])
    assert history.position == 40 and history.oldest == 30
    assert torch.equal(history.read(30, 10), stream[:, 26:36])
    assert torch.equal(history.gather(torch.tensor([[31, 39]])), stream[:, [27, 35]][:, None])
    for bad in ((29, 2), (35, 6)):
        with pytest.raises(EvictedError):
            history.read(*bad)
    with pytest.raises(EvictedError):
        history.gather(torch.tensor([3]))
    with pytest.raises(ValueError):
        history.append(torch.zeros(3, 1, dtype=torch.long))  # another batch


def test_token_history_is_preallocated_and_stable_beyond_a_million_tokens():
    history = TokenHistory(1, 4096)
    buffer, size = history.tokens.data_ptr(), history.nbytes()
    chunk = torch.arange(4096).view(1, -1) % 997
    for _ in range(260):                                        # 1,064,960 tokens
        history.append(chunk)
    assert history.position == 1_064_960 and history.tokens.data_ptr() == buffer and history.nbytes() == size
    assert torch.equal(history.read(history.position - 4096, 4096), chunk)


def test_the_block_index_is_a_ring_of_consecutive_blocks():
    index = BlockIndex(2, 4, 2, 3)
    index.write(0, torch.ones(2, 3, 2, 3))
    index.write(3, 2 * torch.ones(2, 3, 2, 3))
    ids, keys = index.resident()
    assert ids.tolist() == [2, 3, 4, 5] and float(keys[:, 0].mean()) == 1.0 and float(keys[:, 1:].mean()) == 2.0
    with pytest.raises(ValueError):
        index.write(7, torch.ones(2, 1, 2, 3))                  # a gap
    index.write(6, torch.zeros(2, 9, 2, 3))                     # more than the ring: only the newest are kept
    assert index.resident()[0].tolist() == [11, 12, 13, 14] and index.next_block == 15


# -- P1: causality, chunk invariance, source identity -------------------------------------------------------

@pytest.mark.parametrize("window", [False, True])
def test_history_model_is_chunk_invariant_past_the_horizon(window):
    model = _model(window=window)
    tokens = _tokens()
    a, sa, state = _stream(model, tokens, 64)
    for step in (13, 1):
        b, sb, _ = _stream(model, tokens, step)
        assert float((a - b).abs().max()) < 1e-10 and torch.equal(sa, sb)
    assert state.position == 200 and state.tokens.oldest == 200 - model.history.history_capacity


def test_no_future_token_changes_an_earlier_prediction():
    model = _model()
    tokens = _tokens(length=120)
    changed = tokens.clone()
    changed[:, 70:] = (changed[:, 70:] + 5) % 64
    a, sa, _ = _stream(model, tokens, 120)
    b, sb, _ = _stream(model, changed, 120)
    assert torch.equal(a[:, :70], b[:, :70]) and torch.equal(sa[:, :70], sb[:, :70])
    assert not torch.equal(a[:, 70:], b[:, 70:])


def test_every_selected_block_is_complete_and_inside_the_horizon_of_its_query():
    model = _model()
    _, selected, _ = _stream(model, _tokens(length=300), 37)
    h, position = model.history, torch.arange(300)[None, :, None]
    used = selected >= 0
    assert bool(used.any())
    ends = (selected + 1) * h.block - 1
    assert bool((ends[used] <= position.expand_as(selected)[used]).all())
    assert bool((selected[used] >= ((position + 1) // h.block - h.slots).expand_as(selected)[used]).all())


def test_a_historical_block_is_read_with_its_own_source_identity():
    """The readout's keys carry the block's recency rank and its distance; the same tokens read from another
    distance give another read. History never enters ERSA, the window or the token stream."""
    model = _model()
    reader = model.readers[str(model.read_layers[-1])]
    with torch.no_grad():
        reader.read_recency.fill_(0.5)
        reader.key_rank.normal_()
    u = torch.randn(1, 1, 32, dtype=F64)
    states = model._encode(_tokens(1, 11)[:, None], torch.ones(1, 1, 11, dtype=torch.bool))     # [1, 1, s, d]
    keys, values = reader.project(states)
    keys, values = keys[:, None].expand(1, 1, 2, -1, -1), values[:, None].expand(1, 1, 2, -1, -1)
    valid, scores = torch.tensor([[[True, True]]]), torch.zeros(1, 1, 2, dtype=F64)
    near, _ = reader.read(u, keys, values, scores, valid, torch.tensor([[[5, 900]]]))
    far, _ = reader.read(u, keys, values, scores, valid, torch.tensor([[[5000, 900]]]))
    assert not torch.allclose(near, far)


def test_history_reads_change_no_recurrent_state_and_no_stored_token():
    """The read is a pure function of the state: ERSA cells are those of the same model with history off."""
    model = _model()
    tokens = _tokens(length=100)
    _, _, learned = _stream(model, tokens, 50)
    _, _, off = _stream(model, tokens, 50, mode="off")
    assert torch.equal(learned.tokens.tokens, off.tokens.tokens)
    for index in model.read_layers:
        assert torch.equal(learned.indexes[index].keys, off.indexes[index].keys)
    # The read is added beside the ERSA mixer of its block: no ERSA cell is written from it.
    assert all(torch.equal(a, b) for a, b in zip(learned.cells, off.cells))


def test_history_state_is_fixed_in_size_and_formula_exact():
    model = _model()
    state = model.new_state(2)
    before = state.nbytes()
    with torch.no_grad():
        for at in range(0, 400, 50):
            model(_tokens(length=50, seed=at), state)
    h = model.history
    assert state.nbytes() == before
    assert before["history_tokens"] == 2 * h.history_capacity * 4
    assert before["block_index"] == len(model.read_layers) * (2 * h.slots * h.per_block * h.index_dim * 8 + h.slots * 8)


# -- the oracle is diagnosis only; labels are training only ----------------------------------------------------

def test_oracle_modes_need_an_address_and_learned_mode_refuses_one():
    model = _model()
    tokens = _tokens(length=64)
    with pytest.raises(ValueError):
        model(tokens, model.new_state(2), mode="oracle")
    with pytest.raises(ValueError):
        model(tokens, model.new_state(2), address=torch.full((2, 64), -1))
    future = torch.full((2, 64), -1)
    future[:, 10] = 3                                             # block 3 ends at 31 > 10: not eligible
    with pytest.raises(ValueError):
        model(tokens, model.new_state(2), mode="oracle", address=future)


def test_the_oracle_supplies_an_address_not_an_answer():
    model = _model()
    tokens = _tokens(length=64)
    address = torch.full((2, 64), -1)
    address[:, 40], address[:, 63] = 2, 5
    with torch.no_grad():
        logits, diag = model(tokens, model.new_state(2), mode="oracle", address=address, diagnostics=True)
    selected = diag[model.read_layers[-1]]["selected"]
    assert selected[:, 40].tolist() == [[2, -1]] * 2 and selected[:, 63].tolist() == [[5, -1]] * 2
    assert bool((selected[:, :40] == -1).all())                     # nowhere else is anything read
    assert logits.shape == (2, 64, 64)                              # the model still has to produce the answer


def test_learned_mode_is_independent_of_the_labels_and_the_addressing_loss_reaches_every_parameter():
    model = _model().train()
    tokens = _tokens(length=64)
    logits, diag = model(tokens, model.new_state(2), diagnostics=True)
    with torch.no_grad():
        again = model(tokens, model.new_state(2))
    assert torch.equal(logits.detach(), again)                     # no label is an input of learned mode
    gold = torch.full((2, 64), -1)
    gold[:, 40] = 2
    (logits.logsumexp(-1).mean() + addressing_loss(diag, gold)).backward()
    assert not [n for n, p in model.named_parameters() if p.grad is None or not bool(p.grad.abs().sum())]


def test_training_validation_and_test_streams_are_disjoint_and_sources_precede_their_answers():
    task = facts.FactTask(keys=16, events=40, planted=(0.1, 0.5, 0.9))
    a = facts.generate(task, 3, 512, seed=0)
    b = facts.generate(task, 3, 512, seed=600_000_000)
    assert not torch.equal(a.tokens, b.tokens)
    position = torch.arange(512)[None]
    probes = a.kind != 0
    assert bool((a.source[probes] < position.expand_as(a.source)[probes]).all())
    assert bool((a.tokens[probes] == a.tokens.gather(1, a.source.clamp(min=0))[probes]).all())  # evidence = answer
    assert set(a.region[a.kind == facts.PLANTED].tolist()) == {1, 2, 3}


def test_the_program_checker_accepts_references_and_catches_each_violation():
    p = programs.generate(programs.ProgramTask(functions=6, variables=6, header_tokens=200, body_statements=40),
                          2, seed=3)
    tokens, header = p.tokens[0].tolist(), p.header[0]
    clean = programs.check(tokens[:header], tokens[header:])
    assert clean["valid_statement_fraction"] == 1.0 and clean["statements"] > 0
    fns, vrs = programs._defined_state(tokens[:header])
    f, j = sorted(fns)[0], sorted(vrs)[0]
    bad_arity = [programs.CALL, programs.FN0 + f] + [programs.VAR0 + j] * (fns[f] + 1) + [programs.END]
    bad_print = [programs.PRINT, programs.VAR0 + j, programs.EQ, programs.NUM0 + (vrs[j] + 1) % 16, programs.END]
    revised = [programs.LET, programs.VAR0 + j, programs.NUM0 + 3, programs.END,
               programs.PRINT, programs.VAR0 + j, programs.EQ, programs.NUM0 + 3, programs.END]
    assert programs.check(tokens[:header], bad_arity)["call_arity_wrong"] == 1
    assert programs.check(tokens[:header], bad_print)["print_wrong_value"] == 1
    assert programs.check(tokens[:header], revised)["valid_statement_fraction"] == 1.0
    assert programs.check(tokens[:header], [programs.EQ, programs.END])["grammar_errors"] == 1


# -- P3: device, dtype, mixed precision, the last-block invariant ---------------------------------------------

def test_every_state_tensor_is_allocated_on_the_models_device():
    """The meta device stands in for a second device: anything allocated on the default device instead of the
    model's would be a CPU tensor here."""
    with torch.device("meta"):
        model = _model(window=True)
    state = model.new_state(2)
    tensors = [*state.cells, *[t for t in state.tails if t is not None], state.tokens.tokens,
               *[t for layer in state.window.layers for t in layer.values()], state.window.tokens,
               *[i.keys for i in state.indexes.values()], *[i.ids for i in state.indexes.values()]]
    assert all(t.device.type == "meta" for t in tensors)


def test_bfloat16_autocast_keeps_the_recurrent_state_in_float32_and_stays_close():
    torch.manual_seed(0)                                        # nn.Linear's default initialization
    mixer = ERSAMixer(16, heads=2, key_dim=8, value_dim=8, conv_kernel=3, kernel="chunked")
    x = torch.randn(2, 96, 16, generator=torch.Generator().manual_seed(0))
    with torch.no_grad():
        reference, state, _ = mixer(x, mixer.new_state(2), mixer.new_tail(2))
        with torch.autocast("cpu", dtype=torch.bfloat16):
            mixed, low_state, _ = mixer(x, mixer.new_state(2), mixer.new_tail(2))
    assert low_state.dtype == torch.float32 and bool(torch.isfinite(mixed).all())
    assert float((mixed.float() - reference).abs().max()) < 0.05 * float(reference.abs().max())
    assert float((low_state - state).abs().max()) < 0.05 * float(state.abs().max())


def test_a_last_block_cgc_change_leaves_the_history_index_and_ersa_state_valid():
    """The last-block fast-weight invariant holds with history: index keys depend on token ids and encoder
    weights only, and the read happens before the last channel."""
    model = _model()
    tokens = _tokens(length=96)
    state = model.new_state(2)
    with torch.no_grad():
        logits, pre = model(tokens, state, return_pre=True)[:2]
        name = f"blocks.{len(model.blocks) - 1}.cgc.core.weight"
        weight = dict(model.named_parameters())[name]
        changed = weight + 0.1 * torch.randn_like(weight)
        recomputed = torch.func.functional_call(model, {name: changed}, (tokens, model.new_state(2)))
    assert float((last_block_logits(model, pre, changed) - recomputed).abs().max()) < 1e-10
    assert float((last_block_logits(model, pre) - logits).abs().max()) < 1e-10


@pytest.mark.skipif(not CUDA, reason=NOT_RUN_CUDA)
def test_cuda_matches_cpu_and_bf16_is_finite():   # pragma: no cover - needs the workstation
    model = _model(window=True)
    tokens = _tokens(length=128)
    cpu = _stream(model, tokens, 32)[0]
    gpu_model = _model(window=True).to(CUDA_DEVICE)
    gpu = _stream(gpu_model, tokens.to(CUDA_DEVICE), 32)[0].cpu()
    assert float((cpu - gpu).abs().max()) < 1e-8
    float_model = ErsaHistoryLM(gpu_model.ersa.__class__(**{**gpu_model.ersa.__dict__, "model":
                                gpu_model.config.replace(dtype="float32")}), gpu_model.history).to(CUDA_DEVICE).eval()
    with torch.autocast("cuda", dtype=torch.bfloat16), torch.no_grad():
        out = _stream(float_model, tokens.to(CUDA_DEVICE), 32)[0]
    assert bool(torch.isfinite(out).all())


def test_the_byte_tokenizer_satisfies_the_subword_interface_and_lengths_are_compared_in_bytes():
    from research.elpis_torch.tokenizer import ByteTokenizer, SubwordTokenizerProtocol, bytes_per_token
    tokenizer = ByteTokenizer()
    assert isinstance(tokenizer, SubwordTokenizerProtocol) and tokenizer.byte_fallback
    data = "Elpis — ε".encode()
    assert tokenizer.decode_bytes(tokenizer.encode_bytes(data)) == data
    assert b"".join(tokenizer.token_bytes(t) for t in tokenizer.encode_bytes(data)) == data
    assert bytes_per_token(tokenizer, data) == 1.0


def test_program_evidence_labels_point_back_at_the_deciding_token():
    p = programs.generate(programs.ProgramTask(functions=6, variables=6, header_tokens=200, body_statements=40),
                          2, seed=4)
    position = torch.arange(p.tokens.shape[1])[None].expand_as(p.tokens)
    labelled = p.source >= 0
    assert bool(labelled.any()) and bool((p.source[labelled] < position[labelled]).all())
    printed = p.role == programs.PRINT_VALUE
    assert bool((p.tokens[printed] == p.tokens.gather(1, p.source.clamp(min=0))[printed]).all())
    ends = p.role == programs.ARITY_END
    arity = p.tokens.gather(1, p.source.clamp(min=0))[ends]
    assert bool(((arity >= programs.AR0) & (arity < programs.AR0 + programs.ARITIES)).all())
