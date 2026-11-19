"""R7: parametric lookup embeddings (PLE). Deterministic n-gram addressing, trainable tables with lazy updates, the
gated fusion and its exact zero-contribution control, bounded history, checkpoints, read-only stores, isolation."""
from __future__ import annotations

import dataclasses
import json
import os
import subprocess
import sys

import pytest

from .conftest import REPO, SKIP_REASON

torch = pytest.importorskip("torch", reason=SKIP_REASON)

from research.elpis_torch import lm_training as lmt  # noqa: E402
from research.elpis_torch.config import ConfigError, ModelConfig  # noqa: E402
from research.elpis_torch.ersa_model import ErsaConfig, ErsaLM  # noqa: E402
from research.elpis_torch.evidence_model import EvidenceConfig, EvidenceLM  # noqa: E402
from research.elpis_torch.experiments import run_r7  # noqa: E402
from research.elpis_torch.ple import INT64_LIMIT, NgramAddresser, PLEConfig, ple_parameters  # noqa: E402
from research.elpis_torch.ple_store import FileStore, TensorStore  # noqa: E402
from research.elpis_torch.scaling import fixed_state_bytes  # noqa: E402
from research.elpis_torch.sparse_channel import SparseConfig, active_parameters  # noqa: E402
from research.elpis_torch.tokenizer import ByteTokenizer  # noqa: E402
from research.elpis_torch.training import TrainConfig  # noqa: E402

BOS = ByteTokenizer.BOS
PLE = PLEConfig(orders=(2, 3), heads=2, rows=67, head_dim=4, layer=1, boundary=BOS)
EXPERTS = SparseConfig(experts=4, top_k=2, layers=(0, 1), shared=True)
ROUNDING = 1e-12          # as in test_r6: comparisons whose tensor shapes differ agree to rounding (EVIDENCE_R6 s. 10)
GOLDEN_DIGEST = "e245a2dd9daadf5efb7f91864881e2734376b39494ee85e6ebf0bd20b2201308"     # PLE above, vocabulary 259


def widths(dtype="float64", seed=5):
    return ModelConfig(vocab_size=259, d_model=32, n_layers=2, n_heads=2, n_kv_heads=1, max_seq_len=64,
                       channel="cgc", cgc_bottleneck=8, cgc_channels=24, cgc_units=4, seed=seed, dtype=dtype)


def core(ple=PLE, *, chunk=128, sparse=EXPERTS, dtype="float64", seed=5):
    return ErsaLM(ErsaConfig(widths(dtype, seed), ersa_heads=2, key_dim=8, value_dim=8, ersa_kernel="chunked",
                             scan_block=16, conv_kernel=4, chunk=chunk, sparse=sparse, ple=ple)).eval()


def evidence(ple=PLE, *, chunk=1024, dispatch="grouped"):
    ersa = ErsaConfig(widths(), ersa_heads=2, key_dim=8, value_dim=8, ersa_kernel="chunked", scan_block=16,
                      conv_kernel=4, chunk=chunk, sparse=dataclasses.replace(EXPERTS, dispatch=dispatch), ple=ple)
    return EvidenceLM(ersa, EvidenceConfig(block=16, capacity=256, per_block=2, key_dim=16, coarse_top=6,
                                           evidence=3, fine_top=4, overlap=4, encoder_dim=32, fine_dim=8,
                                           read_heads=2, read_head_dim=8, query_tile=7)).eval()


def frozen(model):
    for p in model.parameters():
        p.requires_grad_(False)
    return model.eval()


def tokens(batch=2, length=120, seed=0, high=259):
    return torch.randint(0, high, (batch, length), generator=torch.Generator().manual_seed(seed))


def reference_rows(addresser, tokens_, history):
    """Python integers (no overflow possible) following the module docstring's formula, position by position."""
    c, out = addresser.config, []
    for b in range(tokens_.shape[0]):
        stream, keep = history[b].tolist() + tokens_[b].tolist(), history.shape[1]
        rows = []
        for t in range(tokens_.shape[1]):
            at = keep + t
            latest = max([i for i in range(at + 1) if c.boundary is not None and stream[i] == c.boundary], default=-1)
            row, slot = [], 0
            for i, n in enumerate(c.orders):
                gram = [addresser.sentinel if at - k < latest else stream[at - k] for k in range(n)]
                for h in range(c.heads):
                    mix = 0
                    for k in range(n):
                        mix ^= gram[k] * addresser.multipliers[i][h][k]
                    row.append(addresser.offsets[slot] + mix % addresser.sizes[slot])
                    slot += 1
            rows.append(row)
        out.append(rows)
    return torch.tensor(out, dtype=torch.long)


# -- P2: deterministic addressing --------------------------------------------------------------------------------

def test_addresses_equal_a_big_integer_reference_and_the_addressing_is_pinned():
    addresser = NgramAddresser(PLE, 259)
    assert addresser.digest() == GOLDEN_DIGEST                   # seeded, no process-randomized hash(): stable
    seq = tokens(2, 40, seed=3)
    seq[0, 7], seq[1, 0], seq[1, 20], seq[0, 30] = BOS, BOS, BOS, 258    # boundaries; the largest token id
    history = torch.tensor([[addresser.sentinel, 5], [17, 4]])
    assert torch.equal(addresser.addresses(seq, history), reference_rows(addresser, seq, history))
    assert NgramAddresser(PLE.replace(seed=1), 259).digest() != GOLDEN_DIGEST
    assert NgramAddresser(PLE, 260).digest() != GOLDEN_DIGEST


def test_every_head_owns_a_distinct_prime_range_and_addresses_stay_inside_it():
    addresser = NgramAddresser(PLEConfig(orders=(2, 3, 4), heads=3, rows=100, head_dim=2), 259)
    sizes = addresser.sizes
    assert len(set(sizes)) == len(sizes) == 9 and all(s >= 100 for s in sizes)
    assert all(all(s % f for f in range(2, int(s ** 0.5) + 1)) for s in sizes)              # primes
    assert addresser.offsets == [sum(sizes[:i]) for i in range(9)] and addresser.total_rows == sum(sizes)
    rows = addresser.addresses(tokens(3, 200, seed=1), addresser.new_history(3))
    for slot, (offset, size) in enumerate(zip(addresser.offsets, sizes)):
        assert int(rows[..., slot].min()) >= offset and int(rows[..., slot].max()) < offset + size


def test_multipliers_are_odd_and_bounded_so_no_product_can_overflow():
    addresser = NgramAddresser(PLE, 259)
    flat = [m for order in addresser.multipliers for head in order for m in head]
    assert all(m % 2 == 1 and 0 < m < addresser.bound for m in flat)
    assert addresser.sentinel * max(flat) < INT64_LIMIT and (addresser.sentinel + 1) * addresser.bound <= INT64_LIMIT
    with pytest.raises(ConfigError):
        NgramAddresser(PLE, 1 << 62)                                       # no odd multiplier would fit


def test_addresses_use_only_tokens_at_or_before_each_position():
    addresser = NgramAddresser(PLE, 259)
    seq = tokens(2, 90, seed=4)
    changed = seq.clone()
    changed[:, 50:] = tokens(2, 40, seed=5)
    history = addresser.new_history(2)
    assert torch.equal(addresser.addresses(seq, history)[:, :50], addresser.addresses(changed, history)[:, :50])


def test_chunked_and_incremental_addressing_equal_one_call():
    addresser = NgramAddresser(PLE, 259)
    seq = tokens(2, 101, seed=6)
    seq[0, 33] = seq[1, 34] = BOS                                          # boundaries across chunk edges
    whole = addresser.addresses(seq, addresser.new_history(2))
    for step in (1, 2, 7):
        history, parts = addresser.new_history(2), []
        for at in range(0, 101, step):
            parts.append(addresser.addresses(seq[:, at:at + step], history))
            history = addresser.next_history(seq[:, at:at + step], history)
        assert torch.equal(torch.cat(parts, 1), whole), step


def test_a_boundary_token_isolates_documents_and_padding_needs_nothing_special():
    addresser = NgramAddresser(PLE, 259)
    first, second = tokens(1, 30, seed=7), tokens(1, 25, seed=8)
    second[0, 0] = BOS
    joined = torch.cat((first, second), dim=1)
    alone = addresser.addresses(second, addresser.new_history(1))
    assert torch.equal(addresser.addresses(joined, addresser.new_history(1))[:, 30:], alone)    # no cross-document n-gram
    grams = addresser.grams(joined, addresser.new_history(1))
    assert int(grams[3][0, 30, 1]) == int(grams[3][0, 30, 2]) == addresser.sentinel           # BOS starts afresh
    pad = torch.full((1, 6), ByteTokenizer.PAD)
    padded = addresser.addresses(torch.cat((second, pad), 1), addresser.new_history(1))
    assert torch.equal(padded[:, :25], alone)                              # trailing padding changes no address


def test_exact_byte_identity_is_kept_no_normalization():
    addresser = NgramAddresser(PLE, 259)
    tok = ByteTokenizer()
    upper = addresser.addresses(torch.tensor([tok.encode_bytes("Apple ✓".encode())]), addresser.new_history(1))
    lower = addresser.addresses(torch.tensor([tok.encode_bytes("apple ✓".encode())]), addresser.new_history(1))
    assert not torch.equal(upper[0, 1], lower[0, 1])                       # "Ap" and "ap" are different n-grams


def test_invalid_configurations_and_inputs_are_refused():
    for bad in (dict(orders=(1, 2)), dict(orders=(2, 2)), dict(orders=()), dict(heads=0), dict(rows=1),
                dict(head_dim=0), dict(layer=-1), dict(seed=-1), dict(boundary=-3), dict(lr_multiplier=0),
                dict(lr_multiplier=float("nan")), dict(addressing="nearest")):
        with pytest.raises(ConfigError):
            PLEConfig(**bad)
    with pytest.raises(ConfigError):
        core(PLE.replace(layer=2))                                         # the model has blocks 0 and 1
    with pytest.raises(ConfigError):
        NgramAddresser(PLE.replace(boundary=259), 259)                     # not a token id
    addresser = NgramAddresser(PLE, 259)
    with pytest.raises(ValueError):
        addresser.addresses(torch.tensor([[259]]), addresser.new_history(1))
    with pytest.raises(ValueError):
        addresser.addresses(torch.tensor([[3]]), torch.zeros(1, 1, dtype=torch.long))
    with pytest.raises(ValueError):
        addresser.addresses(torch.tensor([[3.0]]), addresser.new_history(1))
    with pytest.raises(ValueError):
        addresser.addresses(torch.tensor([[3]]), torch.tensor([[0, 260]]))


def test_colliding_ngrams_share_one_head_row_but_other_heads_tell_them_apart():
    addresser = NgramAddresser(PLE, 259)
    grams = torch.cartesian_prod(torch.arange(259), torch.arange(259)).flip(-1)       # every (x[t], x[t-1])
    rows = addresser.rows_of(0, grams)
    found = None
    for value in rows[:, 0].unique():
        group = (rows[:, 0] == value).nonzero().flatten()
        other = rows[group, 1]
        if other.unique().numel() > 1:
            a = int(group[0])
            b = int(group[(other != other[0]).nonzero()[0, 0]])
            found = (a, b)
            break
    assert found is not None
    a, b = found
    assert int(rows[a, 0]) == int(rows[b, 0]) and int(rows[a, 1]) != int(rows[b, 1])
    model = frozen(core())
    first = model.ple.table(rows[a]).flatten()
    second = model.ple.table(rows[b]).flatten()
    assert torch.equal(first[:4], second[:4]) and not torch.equal(first, second)       # shared head, distinct rows


def test_random_addressing_is_a_content_free_control():
    addresser = NgramAddresser(PLE.replace(addressing="random"), 259)
    seq = tokens(2, 50, seed=9)
    a = addresser.addresses(seq, addresser.new_history(2))
    b = addresser.addresses(seq, addresser.new_history(2))
    assert not torch.equal(a, b)                                           # redrawn: no stable content mapping
    for slot, (offset, size) in enumerate(zip(addresser.offsets, addresser.sizes)):
        assert int(a[..., slot].min()) >= offset and int(a[..., slot].max()) < offset + size
    again = NgramAddresser(PLE.replace(addressing="random"), 259)
    assert torch.equal(again.addresses(seq, again.new_history(2)), a)      # but reproducible from the seed


# -- P2: trainable memory ----------------------------------------------------------------------------------------

def test_addressed_rows_and_the_fusion_receive_gradients_and_no_other_row_does():
    model = core()
    model.train()
    seq = tokens(2, 60, seed=10)
    used = model.ple.addresser.addresses(seq, model.ple.addresser.new_history(2)).unique()
    logits = model(seq, model.new_state(2))
    torch.nn.functional.cross_entropy(logits.reshape(-1, 259), tokens(2, 60, seed=11).reshape(-1)).backward()
    grad = model.ple.table.weight.grad.coalesce()
    assert grad.is_sparse and torch.equal(grad.indices().flatten().unique(), used)
    assert int((grad.values().abs().sum(-1) > 0).sum()) == used.numel()
    for name in ("key", "value", "query_norm", "key_norm"):
        assert float(getattr(model.ple, name).weight.grad.abs().sum()) > 0, name


def test_lazy_updates_touch_only_the_rows_each_step_addresses():
    model = core(dtype="float32")
    seen, snapshots = [[]], [model.ple.table.weight.detach().clone()]
    hook = model.ple.table.register_forward_hook(lambda m, i, o: seen[-1].append(i[0].reshape(-1).clone()))

    def after_step(record):
        snapshots.append(model.ple.table.weight.detach().clone())
        seen.append([])

    rows = list(lmt.document_rows([bytes(range(32, 120)) * 3, b"lookup rows " * 9], ByteTokenizer(), 32))
    batches = lmt.document_batches(rows * 4, 2, seed=0, ignore_pad=ByteTokenizer.PAD)
    lmt.train_lm(model, batches, TrainConfig(steps=3, batch_size=2, lr=1e-2, warmup_steps=0, eval_every=1),
                 lmt.LMConfig(seq=32, chunk=16), log=after_step)
    hook.remove()
    assert len(snapshots) == 4
    for step in range(3):
        addressed = torch.cat(seen[step]).unique()
        moved = (snapshots[step + 1] != snapshots[step]).any(-1).nonzero().flatten()
        assert set(moved.tolist()) <= set(addressed.tolist()), step        # an unaddressed row never moves
        assert moved.numel() > 0.9 * addressed.numel(), step


def test_initialization_is_reproducible_and_only_the_seed_changes_it():
    a, b, c = core(seed=5), core(seed=5), core(seed=6)
    assert all(torch.equal(x, y) for x, y in zip(a.state_dict().values(), b.state_dict().values()))
    assert not torch.equal(a.ple.table.weight, c.ple.table.weight)
    assert a.ple.addresser.digest() == c.ple.addresser.digest()            # addressing seed is the PLE's own


def test_checkpoints_carry_the_configuration_and_addressing_and_refuse_mismatches(tmp_path):
    model, tok = frozen(core(dtype="float32")), ByteTokenizer()
    path = tmp_path / "ple.pt"
    lmt.save_lm_checkpoint(model, path, tokenizer=tok, step=3, seed=5)
    loaded, record = lmt.load_lm_checkpoint(path, tokenizer=tok)
    assert record["schema"] == "elpis-torch-r7-lm/1"
    assert record["architecture"]["ple_config"] == PLE.to_dict()
    assert record["architecture"]["ple_addressing_digest"] == model.ple.addresser.digest()
    seq = tokens(1, 50, seed=12)
    with torch.no_grad():
        assert torch.equal(loaded(seq, loaded.new_state(1)), model(seq, model.new_state(1)))
    saved = torch.load(path, weights_only=True)
    saved["record"]["architecture"]["ple_addressing_digest"] = "0" * 64
    torch.save(saved, tmp_path / "stale.pt")
    with pytest.raises(ValueError):
        lmt.load_lm_checkpoint(tmp_path / "stale.pt")
    saved = torch.load(path, weights_only=True)
    saved["record"]["architecture"]["ple_config"]["seed"] = 9            # other addresses, same shapes
    torch.save(saved, tmp_path / "reseeded.pt")
    with pytest.raises(ValueError):
        lmt.load_lm_checkpoint(tmp_path / "reseeded.pt")
    with pytest.raises(ValueError):
        lmt.load_lm_checkpoint(path, tokenizer=type("Other", (), {"vocab_size": 300})())


def test_an_r6_checkpoint_without_ple_still_loads(tmp_path):
    model = frozen(core(ple=None, dtype="float32"))
    path = tmp_path / "r6.pt"
    lmt.save_lm_checkpoint(model, path, tokenizer=ByteTokenizer(), step=0, seed=5)
    saved = torch.load(path, weights_only=True)
    saved["record"]["schema"] = "elpis-torch-r6-lm/1"
    for key in ("ple_config", "ple_addressing_digest"):
        del saved["record"]["architecture"][key]
    torch.save(saved, tmp_path / "old.pt")
    loaded, _ = lmt.load_lm_checkpoint(tmp_path / "old.pt")
    seq = tokens(1, 40, seed=13)
    with torch.no_grad():
        assert loaded.ple is None and torch.equal(loaded(seq, loaded.new_state(1)), model(seq, model.new_state(1)))


# -- P2: fusion -------------------------------------------------------------------------------------------------

@pytest.mark.parametrize("family", ["core", "evidence"])
def test_removing_the_contribution_is_exactly_the_model_without_ple(family):
    build = core if family == "core" else evidence
    with_ple, without = frozen(build()), frozen(build(ple=None))
    assert set(with_ple.load_state_dict(without.state_dict(), strict=False).missing_keys) == {
        "ple.table.weight", "ple.key.weight", "ple.value.weight", "ple.query_norm.weight", "ple.key_norm.weight"}
    seq = tokens(2, 70, seed=14)
    with torch.no_grad():
        plain = without(seq, without.new_state(2))
        with_ple.ple.enabled = False
        assert torch.equal(with_ple(seq, with_ple.new_state(2)), plain)
        with_ple.ple.enabled = True
        assert not torch.equal(with_ple(seq, with_ple.new_state(2)), plain)
        with_ple.ple.value.weight.zero_()                                  # zero value: x + 0 at every position
        assert torch.equal(with_ple(seq, with_ple.new_state(2)), plain)


def test_the_gate_reads_the_hidden_state_and_stays_in_the_open_interval():
    model = frozen(core())
    seq = tokens(1, 20, seed=15)
    addresses = model.ple.addresser.addresses(seq, model.ple.addresser.new_history(1))
    with torch.no_grad():
        model.ple(torch.randn(1, 20, 32, dtype=torch.float64, generator=torch.Generator().manual_seed(1)), addresses)
        first = model.ple.last_gate.clone()
        model.ple(torch.randn(1, 20, 32, dtype=torch.float64, generator=torch.Generator().manual_seed(2)), addresses)
    assert not torch.equal(first, model.ple.last_gate)
    assert float(first.min()) > 0 and float(first.max()) < 1


def test_fusion_happens_at_the_configured_block_and_addresses_are_computed_once_per_call(monkeypatch):
    model = frozen(core())
    calls, inputs = [], []
    original = NgramAddresser.addresses
    monkeypatch.setattr(NgramAddresser, "addresses", lambda self, t, h: calls.append(1) or original(self, t, h))
    hook = model.blocks[1].mix_norm.register_forward_hook(lambda m, i, o: inputs.append(i[0].clone()))
    seq = tokens(2, 40, seed=16)
    with torch.no_grad():
        state = model.new_state(2)
        model(seq[:, :30], state)
        model(seq[:, 30:], state)
        model.ple.enabled = False
        plain = model.new_state(2)
        model(seq[:, :30], plain)
    hook.remove()
    assert len(calls) == 2                                                 # one per call, none while disabled
    assert not torch.equal(inputs[0], inputs[2])                           # the rows enter block 1's input
    early = frozen(core(PLE.replace(layer=0)))
    early.load_state_dict(model.state_dict())
    with torch.no_grad():
        model.ple.enabled = True
        assert not torch.equal(early(seq, early.new_state(2)), model(seq, model.new_state(2)))


# -- invariants: state, causality, sessions, generation -----------------------------------------------------------

def test_state_is_bounded_reset_independent_and_accounted_exactly():
    model = frozen(core())
    state, other = model.new_state(2), model.new_state(2)
    assert state.nbytes()["ple_history"] == 2 * PLE.history * 8
    assert fixed_state_bytes(model, 2) == state.nbytes()
    size = state.nbytes()["total"]
    with torch.no_grad():
        for at in range(0, 300, 60):
            model(tokens(2, 60, seed=at), state)
    assert state.nbytes()["total"] == size                                  # nothing grows with the stream
    assert int((state.ple_history != model.ple.addresser.sentinel).sum()) == state.ple_history.numel()
    assert bool((other.ple_history == model.ple.addresser.sentinel).all())  # sessions never share history
    state.reset()
    assert bool((state.ple_history == model.ple.addresser.sentinel).all())
    ev = frozen(evidence())
    assert ev.context_interface().fixed_bytes(2) == ev.new_state(2).nbytes()


def test_ple_with_experts_and_r5_is_causal_and_chunk_invariant():
    seq = tokens(2, 600, seed=17)
    seq[:, 0] = BOS
    changed = seq.clone()
    changed[:, 301:] = tokens(2, 299, seed=18)
    reference = frozen(evidence(dispatch="reference"))
    model = frozen(evidence())
    model.load_state_dict(reference.state_dict())
    with torch.no_grad():
        r_a, r_b = reference(seq, reference.new_state(2)), reference(changed, reference.new_state(2))
        g_a, g_b = model(seq, model.new_state(2)), model(changed, model.new_state(2))
        assert torch.equal(r_a[:, :301], r_b[:, :301])                     # fixed shapes: bit for bit
        assert float((g_a[:, :301] - g_b[:, :301]).abs().max()) <= ROUNDING
        small = frozen(evidence(chunk=29))
        small.load_state_dict(model.state_dict())
        state = small.new_state(2)
        parts = torch.cat([small(seq[:, at:at + 29], state) for at in range(0, 600, 29)], dim=1)
    assert float((parts - g_a).abs().max()) < 1e-9


def test_generation_with_ple_is_free_running_and_independent_of_prefill_splits():
    model = frozen(core(dtype="float32"))
    prompt = torch.tensor([ByteTokenizer().encode_bytes(b"def lookup(rows):\n    return rows", bos=True)])
    one = lmt.generate(model, prompt, 12, chunk=16)
    assert torch.equal(one, lmt.generate(model, prompt, 12, chunk=16, prefill_calls=3))
    with torch.no_grad():
        full = torch.cat((prompt, one), dim=1)
        logits, _ = model.run(full[:, :-1], chunk=16)
    assert torch.equal(logits[:, prompt.shape[1] - 1:].argmax(-1), one)    # teacher forcing agrees


# -- P3: read-only stores ---------------------------------------------------------------------------------------

def test_a_tensor_store_serves_the_trained_rows_and_bf16_is_within_its_precision():
    model = frozen(core(dtype="float32"))
    seq = tokens(2, 80, seed=19)
    with torch.no_grad():
        trained = model(seq, model.new_state(2))
        exact = TensorStore(model.ple.table.weight, addressing_digest=model.ple.addresser.digest())
        model.ple.attach_store(exact)
        assert torch.equal(model(seq, model.new_state(2)), trained)
        half = TensorStore(model.ple.table.weight, addressing_digest=model.ple.addresser.digest(),
                           dtype=torch.bfloat16)
        model.ple.attach_store(half)
        rounded = model(seq, model.new_state(2))
    probe = torch.arange(model.ple.addresser.total_rows)
    assert torch.equal(half.read(probe), model.ple.table.weight.to(torch.bfloat16))
    assert 0 < float((rounded - trained).abs().max()) < 1e-2 * float(trained.abs().max())


@pytest.mark.parametrize("dtype", ["float32", "bfloat16"])
def test_a_file_store_returns_the_reference_rows_and_reads_only_what_it_needs(tmp_path, dtype):
    model = frozen(core(dtype="float32"))
    digest = model.ple.addresser.digest()
    path = tmp_path / "table.ple"
    run_r7.export_store(model.ple.table.weight, path, addressing_digest=digest, dtype=dtype)
    reference = TensorStore(model.ple.table.weight, addressing_digest=digest, dtype=getattr(torch, dtype))
    addresses = torch.tensor([[3, 4, 5, 200], [4, 4, 150, 5]])
    with FileStore(path) as store:
        assert torch.equal(store.read(addresses), reference.read(addresses))
        assert store.counters["rows_read"] == 5 and store.counters["reads"] == 3            # 3-5 is one run
        assert store.counters["bytes_read"] == 5 * store.row_bytes < store.rows * store.row_bytes
        assert store.verify()                                              # explicit offline check
        seq = tokens(2, 90, seed=20)
        with torch.no_grad():
            model.ple.attach_store(reference)
            expected = model(seq, model.new_state(2))
            model.ple.attach_store(store)
            assert torch.equal(model(seq, model.new_state(2)), expected)
        model.ple.attach_store(None)


def test_corrupt_or_mismatched_stores_are_refused(tmp_path):
    model = frozen(core(dtype="float32"))
    digest = model.ple.addresser.digest()
    good = tmp_path / "good.ple"
    run_r7.export_store(model.ple.table.weight, good, addressing_digest=digest)
    data = good.read_bytes()
    for name, blob in (("magic.ple", b"NOTAPLE!" + data[8:]), ("short.ple", data[:-4]), ("long.ple", data + b"\0")):
        (tmp_path / name).write_bytes(blob)
        with pytest.raises(ValueError):
            FileStore(tmp_path / name)
    flipped = bytearray(data)
    flipped[-1] ^= 0x40
    (tmp_path / "flipped.ple").write_bytes(bytes(flipped))
    with FileStore(tmp_path / "flipped.ple") as store:
        assert not store.verify()                                          # opening does not read the rows
    other = frozen(core(PLE.replace(seed=4), dtype="float32"))
    with FileStore(good) as store:
        with pytest.raises(ValueError):
            other.ple.attach_store(store)                                  # other addressing
        with pytest.raises(IndexError):
            store.read(torch.tensor([store.rows]))
    small = frozen(core(PLE.replace(rows=31), dtype="float32"))
    with FileStore(good) as store:
        with pytest.raises(ValueError):
            small.ple.attach_store(store)                                  # other shape


def test_an_immutable_store_cannot_be_trained():
    model = core(dtype="float32")
    model.ple.attach_store(TensorStore(model.ple.table.weight, addressing_digest=model.ple.addresser.digest()))
    model.train()
    with pytest.raises(RuntimeError):
        model(tokens(1, 10, seed=21), model.new_state(1))


def test_file_backed_inference_writes_nothing(tmp_path):
    scenario = REPO / "tests" / "research" / "elpis_torch" / "zero_write_scenario.py"
    env = {"PYTHONPATH": str(REPO), "PYTHONDONTWRITEBYTECODE": "1"}
    run = subprocess.run([sys.executable, "-B", str(scenario), "--ple", "--audit"], capture_output=True, text=True,
                         cwd=REPO, env=env)
    assert run.returncode == 0, run.stderr
    report = json.loads(run.stdout.strip().splitlines()[-1])
    assert report["write_events"] == [] and report["position"] == 305 and report["rows_read"] > 0
    control = subprocess.run([sys.executable, "-B", str(scenario), "--ple", "--audit", "--negative-control",
                              str(tmp_path / "x")], capture_output=True, text=True, cwd=REPO, env=env)
    assert json.loads(control.stdout.strip().splitlines()[-1])["write_events"]          # the detector works


# -- accounting -------------------------------------------------------------------------------------------------

def test_parameters_and_lookup_bytes_are_accounted():
    model = core(dtype="float32")
    sizes = ple_parameters(PLE, 32, 259)
    table = model.ple.table.weight.numel()
    assert table == sizes["table"] == sum(model.ple.addresser.sizes) * PLE.head_dim
    without = active_parameters(core(ple=None, dtype="float32"))
    counts = active_parameters(model)
    assert counts["total"] == without["total"] + table + sizes["fusion"]
    assert counts["active_per_token"] == without["active_per_token"] + PLE.lookups * PLE.head_dim + sizes["fusion"]
    assert model.ple.lookup_bytes_per_token() == PLE.lookups * PLE.head_dim * 4
    assert PLEConfig.from_dict(json.loads(json.dumps(PLE.to_dict()))) == PLE
