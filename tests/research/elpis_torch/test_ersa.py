"""R2 ERSA: the recurrent associative mixer, the exact addressable policy, and the ERSA language model."""
from __future__ import annotations

import copy
import random

import pytest

from .conftest import SKIP_REASON

torch = pytest.importorskip("torch", reason=SKIP_REASON)

from research.elpis_torch.config import ConfigError, ModelConfig  # noqa: E402
from research.elpis_torch.ersa import (ERSACells, ERSAMixer, cell_code, compile_cell_mixer, delta_step,  # noqa: E402
                                       hard_sigmoid, read_cells, recall_capacity)
from research.elpis_torch.ersa_model import ErsaConfig, ErsaLM  # noqa: E402
from research.elpis_torch.lastblock import last_block_logits  # noqa: E402
from research.elpis_torch.model import next_token_loss  # noqa: E402
from research.elpis_torch.scroll import ChunkError  # noqa: E402

F64 = torch.float64
WIDTHS = dict(vocab_size=64, d_model=32, n_layers=2, n_heads=4, n_kv_heads=2, max_seq_len=64, channel="cgc",
              cgc_bottleneck=8, cgc_channels=16, dtype="float64", seed=1)
VARIANTS = {"none": dict(), "conv": dict(conv_kernel=4), "untied": dict(tie_erase=False),
            "window": dict(attention="window", attn_window=16, chunk=15),
            "gqa": dict(attention="window", attn_window=128, chunk=64, conv_kernel=3)}


def _rand(*shape, seed=0):
    return torch.randn(*shape, generator=torch.Generator().manual_seed(seed), dtype=F64)


def _model(variant="none", **widths):
    return ErsaLM(ErsaConfig(ModelConfig(**{**WIDTHS, **widths}), ersa_heads=2, key_dim=8, value_dim=8,
                             **VARIANTS[variant])).eval()


def _tokens(batch=2, length=90, seed=0):
    return torch.randint(0, 64, (batch, length), generator=torch.Generator().manual_seed(seed))


# -- the kernel ----------------------------------------------------------------------------------------------

def test_one_transition_is_the_written_formula_element_by_element():
    state, key, value = _rand(5, 7), _rand(7, seed=1), _rand(5, seed=2)
    retain, erase, write = 0.9, 0.6, 0.3
    gate = lambda g: torch.tensor(g, dtype=F64)     # noqa: E731
    got = delta_step(state, key, value, retain=gate(retain), erase=gate(erase), write=gate(write))
    for v in range(5):
        stored = sum(float(state[v, j]) * float(key[j]) for j in range(7))
        for k in range(7):
            expected = retain * float(state[v, k]) - retain * erase * stored * float(key[k]) \
                + write * float(value[v]) * float(key[k])
            assert abs(float(got[v, k]) - expected) < 1e-12
    assert torch.equal(read_cells(got, key), got @ key)
    untouched = state.clone()
    delta_step(state, key, value, retain=gate(0.0), erase=gate(1.0), write=gate(1.0))
    assert torch.equal(state, untouched)                    # never in place


def test_the_gate_reaches_closed_half_and_open_exactly():
    z = torch.tensor([-100.0, -2.0, 0.0, 2.0, 100.0], dtype=F64)
    assert hard_sigmoid(z).tolist() == [0.0, 0.0, 0.5, 1.0, 1.0]
    inside = torch.tensor([-1.0, 1.0], dtype=F64, requires_grad=True)
    hard_sigmoid(inside).sum().backward()
    assert inside.grad.tolist() == [0.25, 0.25]


def test_with_unit_keys_the_erase_path_is_a_contraction():
    """Learning policy: ||S'|| <= retain ||S|| + write ||v||, whatever the gates in [0, 1]."""
    state = _rand(6, 8)
    for seed in range(50):
        key = torch.nn.functional.normalize(_rand(8, seed=seed), dim=-1)
        value, gates = _rand(6, seed=seed + 100), torch.rand(3, generator=torch.Generator().manual_seed(seed), dtype=F64)
        new = delta_step(state, key, value, retain=gates[0], erase=gates[1], write=gates[2])
        assert float(new.norm()) <= float(gates[0] * state.norm() + gates[2] * value.norm()) + 1e-12
        state = new


# -- the learning policy: a trainable causal mixer -------------------------------------------------------------

@pytest.mark.parametrize("options", [dict(), dict(conv_kernel=4), dict(tie_erase=False), dict(out_gate=False)])
def test_the_mixer_is_causal_and_chunking_does_not_change_it(options):
    mixer = ERSAMixer(12, heads=3, key_dim=5, value_dim=4, dtype=F64, **options)
    x = _rand(2, 40, 12)
    with torch.no_grad():
        whole, state, _ = mixer(x, mixer.new_state(2), mixer.new_tail(2))
        cells, tail, parts = mixer.new_state(2), mixer.new_tail(2), []
        for start, stop in ((0, 1), (1, 2), (2, 9), (9, 40)):
            piece, cells, tail = mixer(x[:, start:stop], cells, tail)
            parts.append(piece)
        assert float((torch.cat(parts, dim=1) - whole).abs().max()) < 1e-12
        assert float((cells - state).abs().max()) < 1e-12
        changed = x.clone()
        changed[:, 25:] += 1.0
        later, _, _ = mixer(changed, mixer.new_state(2), mixer.new_tail(2))
        assert torch.equal(later[:, :25], whole[:, :25]) and not torch.equal(later[:, 25:], whole[:, 25:])
    assert state.shape == (2, 3, 4, 5) and mixer.state_bytes(2) == (2 * 3 * 4 * 5 + (
        2 * 3 * 12 if options.get("conv_kernel") else 0)) * 8


def test_every_mixer_parameter_receives_a_gradient_through_the_recurrence():
    mixer = ERSAMixer(12, heads=3, key_dim=5, value_dim=4, conv_kernel=3, tie_erase=False, dtype=F64)
    x = _rand(2, 30, 12)
    out, state, _ = mixer(x, mixer.new_state(2), mixer.new_tail(2))
    (out[:, -1].pow(2).sum() + state.pow(2).sum()).backward()      # only the LAST position: needs the recurrence
    for name, parameter in mixer.named_parameters():
        assert parameter.grad is not None and bool(torch.isfinite(parameter.grad).all()), name
        assert float(parameter.grad.abs().sum()) > 0, name
    # The gradient really flows back through time: the first token's input influences the last output.
    x = x.clone().requires_grad_(True)
    out, _, _ = mixer(x, mixer.new_state(2), mixer.new_tail(2))
    out[:, -1].sum().backward()
    assert float(x.grad[:, 0].abs().sum()) > 0


def test_the_state_stays_finite_and_bounded_over_a_long_sequence():
    mixer = ERSAMixer(16, heads=2, key_dim=8, value_dim=8, dtype=torch.float32)
    state, worst = mixer.new_state(1), 0.0
    with torch.no_grad():
        mixer.gates.bias.view(2, -1)[:, 0] = -4.0           # never forget: the hardest case for boundedness
        for chunk in range(200):                            # 20,000 tokens
            x = torch.randn(1, 100, 16, generator=torch.Generator().manual_seed(chunk))
            out, state, _ = mixer(x, state)
            assert bool(torch.isfinite(out).all()) and bool(torch.isfinite(state).all())
            worst = max(worst, float(state.abs().max()))
    assert worst < 50 and state.shape == (1, 2, 8, 8)


def test_selective_retention_and_replacement_with_closed_and_open_gates():
    """A closed write gate leaves the state bit for bit alone; an open one replaces the old value for that key."""
    keys = torch.eye(4, dtype=F64)
    one, zero = torch.tensor(1.0, dtype=F64), torch.tensor(0.0, dtype=F64)
    old, new, other = _rand(3, seed=1), _rand(3, seed=2), _rand(3, seed=3)
    state = delta_step(torch.zeros(3, 4, dtype=F64), keys[0], old, retain=one, erase=one, write=one)
    state = delta_step(state, keys[2], other, retain=one, erase=one, write=one)
    kept = state.clone()
    for seed in range(1000):                               # a thousand tokens the gate ignores
        state = delta_step(state, torch.nn.functional.normalize(_rand(4, seed=seed), dim=-1), _rand(3, seed=seed),
                           retain=one, erase=zero, write=zero)
    assert torch.equal(state, kept)
    state = delta_step(state, keys[0], new, retain=one, erase=one, write=one)
    assert torch.equal(read_cells(state, keys[0]), new) and torch.equal(read_cells(state, keys[2]), other)
    # Without the erase term (no correction) the old value would still be there, mixed in.
    mixed = delta_step(kept, keys[0], new, retain=one, erase=zero, write=one)
    assert torch.allclose(read_cells(mixed, keys[0]), old + new)


def test_recall_degrades_with_load_which_is_why_the_state_is_not_a_million_token_memory():
    load = [recall_capacity(16, 16, pairs)["recalled_fraction"] for pairs in (2, 8, 16, 32, 64)]
    assert load[0] == 1.0 and load == sorted(load, reverse=True) and load[-1] < 0.5
    assert recall_capacity(64, 16, 16)["recalled_fraction"] > recall_capacity(16, 16, 16)["recalled_fraction"]


# -- the exact policy: addressable cells -----------------------------------------------------------------------

@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_cells_are_written_overwritten_accumulated_halved_and_cleared_exactly(dtype):
    bank = ERSACells(16, 2, dtype=dtype)
    values = {a: torch.tensor([0.1 * a + 0.3, -7.25 * a], dtype=dtype) for a in range(16)}
    for address, value in values.items():
        bank.write(address, value)
    for address, value in values.items():
        assert torch.equal(bank.read(address), value)                   # bit for bit, not approximately
    bank.write(5, [3.0, 4.0])
    assert torch.equal(bank.read(5), torch.tensor([3.0, 4.0], dtype=dtype))
    assert all(torch.equal(bank.read(a), values[a]) for a in range(16) if a != 5)     # nothing else moved
    bank.add(5, [1.0, 2.0])
    bank.add(5, [1.0, 2.0])
    assert bank.read(5).tolist() == [5.0, 8.0]
    assert torch.equal(bank.reduce(5, 0), bank.read(5) + bank.read(0))
    for _ in range(3):
        bank.halve()
    assert bank.read(5).tolist() == [0.625, 1.0]
    bank.clear()
    assert float(bank.state.abs().sum()) == 0.0 and bank.transitions == 16 + 1 + 2 + 3 + 1
    assert bank.nbytes() == 16 * 2 * (4 if dtype == torch.float32 else 8)


def test_cell_addresses_and_values_are_bounded_and_banks_are_independent():
    bank, other = ERSACells(4), ERSACells(4)
    for bad in (4, -1, 1.5, "0"):
        with pytest.raises(IndexError):
            bank.write(bad, 1.0)
    for bad in (float("nan"), float("inf"), [1.0, 2.0]):
        with pytest.raises(ValueError):
            bank.write(0, bad)
    bank.write(0, 1e308)
    before = bank.state.clone()
    with pytest.raises(FloatingPointError):
        bank.add(0, 1e308)                                  # would overflow: refused, state unchanged
    assert torch.equal(bank.state, before)
    assert float(other.state.abs().sum()) == 0.0
    bank.reset()
    assert float(bank.state.abs().sum()) == 0.0 and bank.transitions == 0
    with pytest.raises(ValueError):
        ERSACells(0)


def test_registers_counters_and_accumulators_compute_exactly_inside_the_state():
    """Every number below is produced by a read of the state tensor; the host only moves it between cells."""
    machine = ERSACells(8)
    for _ in range(1000):                                   # a counter
        machine.add(0, 1.0)
    assert float(machine.read(0)) == 1000.0
    for n in range(1, 101):                                 # an accumulator: 1 + 2 + ... + 100
        machine.add(1, float(n))
    assert float(machine.read(1)) == 5050.0
    machine.write(2, 0.0)                                   # Fibonacci in two registers and a temporary
    machine.write(3, 1.0)
    for _ in range(70):
        machine.write(4, machine.reduce(2, 3))              # the sum is a two-hot read
        machine.write(2, machine.read(3))
        machine.write(3, machine.read(4))
    assert float(machine.read(3)) == 308061521170129.0      # F(71), exact in float64
    machine.write(5, 1.0)                                   # a halving counter
    for _ in range(40):
        machine.halve()
    assert float(machine.read(5)) == 2.0 ** -40


def test_a_cell_survives_a_long_stream_of_other_traffic_bit_for_bit():
    """Long-range recovery in the exact policy: nothing addresses the cell, so nothing changes it."""
    machine, rng = ERSACells(32, dtype=torch.float32), random.Random(0)
    machine.write(7, 0.123456789)
    kept = machine.read(7).clone()
    for step in range(20_000):
        address = rng.choice([a for a in range(32) if a != 7])
        (machine.write if step % 2 else machine.add)(address, rng.uniform(-1, 1))
    assert torch.equal(machine.read(7), kept) and machine.transitions == 20_001


# -- the exact policy through the mixer's own forward pass -----------------------------------------------------

@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_a_compiled_mixer_is_exact_memory_in_its_ordinary_forward_pass(dtype):
    """No weight is learned. A random program is run by ``ERSAMixer.forward`` and replayed by a Python dict."""
    cells, width, rng = 12, 2, random.Random(3)
    mixer = compile_cell_mixer(cells, width, dtype=dtype)
    assert not any(p.requires_grad for p in mixer.parameters())
    reference, program, expected = {a: [0.0, 0.0] for a in range(cells)}, [], []
    for _ in range(600):
        kind, address = rng.choice(["store", "add", "read", "read", "halve", "clear"]), rng.randrange(cells)
        value = [float(rng.randrange(-50, 50)), float(rng.randrange(0, 9)) / 4]
        if kind == "store":
            reference[address] = list(value)
            program.append(cell_code(cells, width, address=address, value=value, store=True, dtype=dtype))
        elif kind == "add":
            reference[address] = [reference[address][0] + value[0], reference[address][1] + value[1]]
            program.append(cell_code(cells, width, address=address, value=value, add=True, dtype=dtype))
        elif kind == "halve" and rng.random() < 0.2:
            reference = {a: [v / 2 for v in vs] for a, vs in reference.items()}
            program.append(cell_code(cells, width, address=address, halve=True, dtype=dtype))
        elif kind == "clear" and rng.random() < 0.05:
            reference = {a: [0.0, 0.0] for a in range(cells)}
            program.append(cell_code(cells, width, address=address, clear=True, dtype=dtype))
        else:
            program.append(cell_code(cells, width, address=address, dtype=dtype))
        expected.append(list(reference[address]))
    x = torch.stack(program)[None]
    out, state, _ = mixer(x, mixer.new_state(1))
    got = out[0, :, cells:cells + width]
    assert torch.equal(got, torch.tensor(expected, dtype=dtype))            # every output, exactly
    assert torch.equal(state[0, 0].T, torch.tensor([reference[a] for a in range(cells)], dtype=dtype))
    cells_state, parts = mixer.new_state(1), []
    for token in range(x.shape[1]):                                         # one token at a time: the same bits
        piece, cells_state, _ = mixer(x[:, token:token + 1], cells_state)
        parts.append(piece)
    assert torch.equal(torch.cat(parts, dim=1), out) and torch.equal(cells_state, state)


# -- the ERSA language model -----------------------------------------------------------------------------------

@pytest.mark.parametrize("change", [dict(ersa_heads=0), dict(conv_kernel=1), dict(attention="dense"),
                                    dict(attention="window", attn_window=16, chunk=16),
                                    dict(attention="window", attn_heads=3)])
def test_invalid_ersa_configurations_are_refused(change):
    with pytest.raises(ConfigError):
        ErsaConfig(ModelConfig(**WIDTHS), **change)


@pytest.mark.parametrize("variant", list(VARIANTS))
def test_the_model_is_causal_trains_and_is_the_same_chunked_or_token_by_token(variant):
    model, tokens = _model(variant), _tokens()
    with torch.no_grad():
        whole, state = model.run(tokens)
        assert float((model.run(tokens, chunk=1)[0] - whole).abs().max()) < 1e-10
        assert float((model.run(tokens, chunk=7)[0] - whole).abs().max()) < 1e-10
        changed = tokens.clone()
        changed[:, 50:] = (changed[:, 50:] + 1) % 64
        later = model.run(changed)[0]
        assert torch.equal(later[:, :50], whole[:, :50]) and not torch.equal(later[:, 50:], whole[:, 50:])
    assert whole.shape == (2, 90, 64) and bool(torch.isfinite(whole).all()) and state.position == 90
    logits = model.run(tokens)[0]
    assert float((logits - whole).abs().max()) < 1e-10               # training path equals inference path
    next_token_loss(logits[:, :-1], tokens).backward()
    for name, parameter in model.named_parameters():
        assert parameter.grad is not None and bool(torch.isfinite(parameter.grad).all()), name
        if ".ersa." in name or "cgc.core" in name:
            assert float(parameter.grad.abs().sum()) > 0, name


@pytest.mark.parametrize("variant", ["none", "conv", "window"])
def test_the_state_is_fixed_in_size_whatever_the_stream_length(variant):
    model = _model(variant, dtype="float32")
    state = model.new_state(2)
    sizes, buffers = state.nbytes(), [c.data_ptr() for c in state.cells]
    with torch.no_grad():
        for seed in range(40):                              # 3,600 tokens
            logits = model.run(_tokens(2, 90, seed=seed), state)[0]
            assert bool(torch.isfinite(logits).all())
    assert state.nbytes() == sizes and buffers == [c.data_ptr() for c in state.cells] and state.position == 3600
    assert sizes["ersa_cells"] == 2 * 2 * 2 * 8 * 8 * 4                      # layers x batch x heads x V x K x bytes
    assert ("attention_window" in sizes) == (variant == "window")
    with pytest.raises(ChunkError):
        model(_tokens(2, 200), state)
    assert state.position == 3600


def test_generation_is_autoregressive_deterministic_and_unbounded_in_length():
    model, prompt = _model("conv"), _tokens(2, 12)
    new = model.generate(prompt, 150)                       # longer than any window or chunk
    assert new.shape == (2, 150) and torch.equal(new, model.generate(prompt, 150))
    with torch.no_grad():                                   # each generated token is the argmax given the prefix
        full = model.run(torch.cat((prompt, new), dim=1))[0]
    assert torch.equal(full[:, 11:-1].argmax(-1), new)
    sampled = model.generate(prompt, 20, temperature=1.0, generator=torch.Generator().manual_seed(4))
    assert torch.equal(sampled, model.generate(prompt, 20, temperature=1.0, generator=torch.Generator().manual_seed(4)))


@pytest.mark.parametrize("variant", ["none", "window"])
def test_a_last_block_cgc_change_touches_no_recurrent_state_and_no_window_cache(variant):
    """So volatile last-block adaptation never requires reconstructing history (the R1 argument, for ERSA)."""
    model, tokens = _model(variant), _tokens()
    adapted, earlier = copy.deepcopy(model), copy.deepcopy(model)
    with torch.no_grad():
        adapted.blocks[-1].cgc.core.weight.mul_(1.5)
        earlier.blocks[0].cgc.core.weight.mul_(1.5)
        state = model.new_state(2)
        logits, pre = model(tokens[:, :15], state, return_pre=True)
        changed_state, first_state = adapted.new_state(2), earlier.new_state(2)
        changed = adapted(tokens[:, :15], changed_state)
        earlier(tokens[:, :15], first_state)
    assert all(torch.equal(a, b) for a, b in zip(state.cells, changed_state.cells))
    assert not torch.equal(state.cells[1], first_state.cells[1])            # an earlier block's CGC does matter
    if variant == "window":
        assert all(torch.equal(a[k], b[k]) for a, b in zip(state.window.layers, changed_state.window.layers) for k in a)
    assert not torch.equal(logits, changed)
    assert float((last_block_logits(model, pre) - logits).abs().max()) < 1e-12
    assert float((last_block_logits(model, pre, adapted.blocks[-1].cgc.core.weight) - changed).abs().max()) < 1e-12


def test_a_small_ersa_model_learns_a_sequential_dependency(world, splits):
    """Offline training on R0's transduction task with no attention at all (ERSA plus a 4-tap causal convolution).

    Measured while writing this test: without the convolution the same model stays at chance for 600
    steps. The associative state alone carries no local order; EVIDENCE_R2 reports the comparison.
    """
    from research.elpis_torch.data import SINGLE
    torch.manual_seed(0)
    model = ErsaLM(ErsaConfig(ModelConfig(vocab_size=259, d_model=32, n_layers=2, n_heads=2, max_seq_len=64,
                                          channel="cgc", cgc_bottleneck=16, cgc_channels=32, seed=1),
                              ersa_heads=2, key_dim=8, value_dim=8, conv_kernel=4))
    tokens, kind = splits.train.tokens[:, :33], splits.val.kind[:, 1:33]
    optimizer = torch.optim.AdamW(model.parameters(), lr=4e-3, weight_decay=0.01)
    sampler, losses = torch.Generator().manual_seed(0), []
    held, accuracy = splits.val.tokens[:, :33], 0.0
    for step in range(1600):                         # the transition's step depends on the BLAS: 600 here, later in CI
        batch = tokens[torch.randint(len(tokens), (32,), generator=sampler)]
        loss = next_token_loss(model.run(batch[:, :-1])[0], batch)
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        losses.append(float(loss.detach()))
        if step % 200 == 199:
            with torch.no_grad():
                right = model.run(held[:, :-1])[0].argmax(-1) == held[:, 1:]
            accuracy = float(right[kind == SINGLE].double().mean())
            if accuracy > 0.9:
                break
    assert losses[0] > 5.0 and sum(losses[-10:]) / 10 < 0.5 * losses[0]
    assert accuracy > 0.6                                                      # chance is 1/6; measured 1.00
