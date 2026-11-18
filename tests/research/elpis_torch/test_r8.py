"""R8: the R5 evidence read with stored fine keys (``EvidenceConfig.fine_keys = "stored"``).

Each block's fine keys are computed once, at block completion, from the same encoding as its coarse descriptors and
kept in the ring's index beside them. Candidates are reranked from the stored keys; only the blocks the working
context injects are reread and re-encoded. The R5 reference (``"reread"``) stays the default and the yardstick.

Contract (float64 models): the same candidates, rankings, selections and read positions as the reference, bit for
bit; scores and logits to ROUNDING (the stored keys and the injected encodings are computed in other batches than the
reference's single reread batch, and BLAS / vectorized reductions round sums of different lengths differently:
EVIDENCE_R6 section 10). Bit for bit wherever both runs execute identical shapes (causality with the fixed-shape
dispatch, reset, determinism). Every claim has a negative control that must fail it.
"""
from __future__ import annotations

import dataclasses
import json
import subprocess
import sys

import pytest

from .conftest import REPO, SKIP_REASON

torch = pytest.importorskip("torch", reason=SKIP_REASON)

from research.elpis_torch import evidence_model  # noqa: E402
from research.elpis_torch import evidence_tasks as et  # noqa: E402
from research.elpis_torch import lm_training as lmt  # noqa: E402
from research.elpis_torch.config import ConfigError, ModelConfig  # noqa: E402
from research.elpis_torch.context_runtime import ContextSession, ModelRevisionError  # noqa: E402
from research.elpis_torch.ersa_model import ErsaConfig  # noqa: E402
from research.elpis_torch.evidence_model import EvidenceConfig, EvidenceHead, EvidenceLM, evaluation_state_dict  # noqa: E402
from research.elpis_torch.evidence_ring import EvidenceRing, StaleAddressError  # noqa: E402
from research.elpis_torch.ple import PLEConfig  # noqa: E402
from research.elpis_torch.sparse_channel import SparseConfig  # noqa: E402

ROUNDING = 1e-12          # as in test_r6 / test_r7: comparisons whose tensor shapes differ agree to rounding
EXPERTS = SparseConfig(experts=4, top_k=2, layers=(0, 1), shared=True)
PLE = PLEConfig(orders=(2, 3), heads=2, rows=67, head_dim=4, layer=1)
PIECE = 70                # calls of 70 tokens: a multiple of the query tile (7), so CUT is a tile boundary
CUT = 301                 # inside the call [280, 350): the future starts there
EXPOSED = ("candidates", "eligible", "fine_top", "selected", "read_block", "read_position", "local_block")


def build(fine_keys="stored", *, chunk=PIECE, dtype="float64", sparse=EXPERTS, dispatch="grouped", ple=None,
          **changes):
    widths = ModelConfig(vocab_size=et.VOCAB, d_model=32, n_layers=2, n_heads=2, n_kv_heads=1, max_seq_len=64,
                         channel="cgc", cgc_bottleneck=8, cgc_channels=24, cgc_units=4, seed=5, dtype=dtype)
    routed = None if sparse is None else dataclasses.replace(sparse, dispatch=dispatch)
    ersa = ErsaConfig(widths, ersa_heads=2, key_dim=8, value_dim=8, ersa_kernel="chunked", scan_block=16,
                      conv_kernel=4, chunk=chunk, sparse=routed, ple=ple)
    evidence = dict(block=16, capacity=256, per_block=2, key_dim=16, coarse_top=6, evidence=3, fine_top=4, overlap=4,
                    encoder_dim=32, fine_dim=8, read_heads=2, read_head_dim=8, query_tile=7, fine_keys=fine_keys)
    evidence.update(changes)
    return EvidenceLM(ersa, EvidenceConfig(**evidence))


def frozen(model):
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    return model.eval()


def pair(**changes):
    """The R5 reference and the R8 stored-keys model with identical weights."""
    reference = frozen(build("reread", **changes))
    stored = frozen(build("stored", **changes))
    stored.load_state_dict(reference.state_dict())
    return reference, stored


def records(batch=2, length=600, seed=7, **task):
    return et.generate_records(et.RecordTask(**{"queries": 6, "background": 20.0, **task}), batch, length, seed=seed)


@pytest.fixture(scope="module")
def stream():
    return records().tokens                                   # 600 tokens through a 256-token ring: it wraps


def run(model, tokens, *, piece=PIECE, state=None, labels=None, **options):
    """Every position read, call by call (earlier calls' blocks come from the ring: rereads or stored keys)."""
    state = model.new_state(tokens.shape[0]) if state is None else state
    logits, seen = [], []
    with torch.no_grad():
        for at in range(0, tokens.shape[1], piece):
            if labels is not None:
                options["evidence"] = labels[:, at:at + piece]
            out, diag = model(tokens[:, at:at + piece], state, diagnostics=True, **options)
            logits.append(out)
            seen.append(diag)
    exposed = {k: torch.cat([d[k] for d in seen], dim=1) for k in seen[0] if k != "losses"}
    return torch.cat(logits, dim=1), exposed, state


def finite_gap(a, b):
    both = torch.isfinite(a) & torch.isfinite(b)
    assert torch.equal(torch.isfinite(a), torch.isfinite(b))
    return float((a[both] - b[both]).abs().max()) if bool(both.any()) else 0.0


# -- parity with the reference --------------------------------------------------------------------------------------

@pytest.mark.parametrize("batch", [1, 2, 3])
def test_stored_keys_select_the_same_evidence_and_agree_with_the_reread(batch):
    reference, stored = pair()
    tokens = records(batch=batch).tokens
    want, want_seen, want_state = run(reference, tokens)
    got, got_seen, got_state = run(stored, tokens)
    for key in EXPOSED:                                                          # identity: bit for bit
        assert torch.equal(want_seen[key], got_seen[key]), key
    assert finite_gap(want_seen["fine_scores"], got_seen["fine_scores"]) <= ROUNDING
    assert float((want - got).abs().max()) <= ROUNDING
    assert float((want_seen["null_weight"] - got_seen["null_weight"]).abs().max()) <= ROUNDING
    reads = got_state.counters["fine_tier_reads"]
    assert reads > 0 and got_state.ring.stats["fine_reads"] == reads
    assert got_state.counters["blocks_reencoded"] < want_state.counters["blocks_reencoded"]
    assert got_state.ring.stats["stale_rejections"] == want_state.ring.stats["stale_rejections"] == 0
    assert got_state.ring.oldest > 0                                             # the ring wrapped


@pytest.mark.parametrize("ple", [None, PLE], ids=["no_ple", "ple"])
@pytest.mark.parametrize("dispatch", ["grouped", "reference"])
def test_parity_holds_with_and_without_ple_under_either_expert_dispatch(stream, ple, dispatch):
    reference, stored = pair(ple=ple, dispatch=dispatch)
    want, want_seen, _ = run(reference, stream)
    got, got_seen, state = run(stored, stream)
    assert all(torch.equal(want_seen[k], got_seen[k]) for k in EXPOSED)
    assert float((want - got).abs().max()) <= ROUNDING and state.counters["fine_tier_reads"] > 0


@pytest.mark.parametrize("changes", [dict(rerank="coarse"), dict(rerank="all"), dict(supersession="sum"),
                                     dict(supersession="off"), dict(coarse_top=12, evidence=4, fine_top=6)],
                         ids=["rerank_coarse", "rerank_all", "supersession_sum", "supersession_off", "wider"])
def test_parity_holds_for_every_reranking_variant(stream, changes):
    reference, stored = pair(**changes)
    want, want_seen, _ = run(reference, stream)
    got, got_seen, _ = run(stored, stream)
    assert all(torch.equal(want_seen[k], got_seen[k]) for k in EXPOSED)
    assert float((want - got).abs().max()) <= ROUNDING


def test_revision_heavy_streams_keep_the_same_sources_ranks_and_supersession():
    """Many superseded binds and stale questions: the newest revision must still win in the same places, and the
    labelled gold's coarse and fine ranks must be identical (labels are metrics here, never a selection input)."""
    data = records(seed=11, revisions=(3, 5), p_stale_question=1.0, queries=8, background=4.0)
    labels = data.query_source()
    reference, stored = pair()
    want, want_seen, _ = run(reference, data.tokens, labels=labels)
    got, got_seen, _ = run(stored, data.tokens, labels=labels)
    for key in (*EXPOSED, "coarse_rank", "fine_rank", "gold_block", "historical"):
        assert torch.equal(want_seen[key], got_seen[key]), key
    assert int(want_seen["historical"].sum()) > 0 and float((want - got).abs().max()) <= ROUNDING
    assert int(data.probes["revised"].sum()) > 0


def test_oracle_diagnosis_rereads_the_forced_gold_and_still_agrees():
    data = records(seed=3)
    labels = data.query_source()
    reference, stored = pair()
    want, want_seen, _ = run(reference, data.tokens, labels=labels, mode="oracle")
    got, got_seen, _ = run(stored, data.tokens, labels=labels, mode="oracle")
    assert torch.equal(want_seen["selected"], got_seen["selected"])
    assert float((want - got).abs().max()) <= ROUNDING


def test_runs_are_deterministic_bit_for_bit(stream):
    _, stored = pair()
    a, a_seen, _ = run(stored, stream)
    b, b_seen, _ = run(stored, stream)
    assert torch.equal(a, b) and all(torch.equal(a_seen[k], b_seen[k]) for k in a_seen)


# -- chunking, single tokens, positions inside and across blocks ---------------------------------------------------

def test_one_call_chunks_sessions_and_single_tokens_agree(stream):
    whole_model = frozen(build("reread", chunk=1024))
    with torch.no_grad():
        whole = whole_model(stream, whole_model.new_state(2))                  # the R5 reference, one call
    small = frozen(build("stored", chunk=37))
    small.load_state_dict(whole_model.state_dict())
    probes = [0, 1, 15, 16, 17, 31, 32, 200, 255, 256, 257, 299]               # first, middle, last of blocks
    with ContextSession(small, batch=2) as session:
        first = session.prefill(stream[:, :300], probes=probes)
        for position, logits in first.probes.items():
            assert float((logits - whole[:, position]).abs().max()) < 1e-9, position
        for position in range(300, 600):                                       # decoding, token by token
            logits = session.step(stream[:, position])
            assert float((logits - whole[:, position]).abs().max()) < 1e-9, position
        counters = session.counters()
    assert counters["fine_tier_reads"] > 0 and counters["ring_stale_rejections"] == 0


def test_reading_only_some_positions_equals_reading_all_of_them(stream):
    """Logits at requested positions only (R4) equal the dense computation's at those positions, with stored keys."""
    _, stored = pair()
    dense, _, _ = run(stored, stream)
    state = stored.new_state(2)
    with torch.no_grad():
        for at in range(0, 600, PIECE):
            wanted = torch.tensor([p for p in (0, 1, 15, 16, 31, 47, 63, 69) if at + p < 600])
            sparse = stored(stream[:, at:at + PIECE], state, logit_positions=wanted)
            assert float((sparse - dense[:, at + wanted]).abs().max()) <= ROUNDING, at


# -- causality and its negative control ----------------------------------------------------------------------------

def _future(stream):
    changed = stream.clone()
    changed[:, CUT:] = torch.randint(1, 30, changed[:, CUT:].shape, generator=torch.Generator().manual_seed(0))
    return changed


@pytest.mark.parametrize("dispatch", ["reference", "grouped"])
def test_stored_keys_are_causal(stream, dispatch):
    """Changing every token from CUT on changes nothing before it: bit for bit with the fixed-shape expert dispatch,
    within ROUNDING with the grouped one (routes in a call depend on its later tokens: EVIDENCE_R6 section 10)."""
    model = frozen(build("stored", dispatch=dispatch))
    a, a_seen, _ = run(model, stream)
    b, b_seen, _ = run(model, _future(stream))
    assert all(torch.equal(a_seen[k][:, :CUT], b_seen[k][:, :CUT]) for k in EXPOSED)
    if dispatch == "reference":
        assert torch.equal(a[:, :CUT], b[:, :CUT])
    else:
        assert float((a[:, :CUT] - b[:, :CUT]).abs().max()) <= ROUNDING


@pytest.mark.parametrize("target", ["fine_keys", "read_keys"])
def test_the_causality_test_detects_a_planted_future_leak(stream, monkeypatch, target):
    """A key projection that adds 1e-7 of the mean over a call's blocks (the call's later blocks included) leaks
    the future into earlier positions; the test above must see it far beyond ROUNDING."""
    method = getattr(EvidenceHead, target)

    def leaky(self, states):
        out = method(self, states)
        if isinstance(out, tuple):
            return tuple(t + 1e-7 * t.mean(dim=1, keepdim=True) for t in out)
        return out + 1e-7 * out.mean(dim=1, keepdim=True)

    monkeypatch.setattr(EvidenceHead, target, leaky)
    model = frozen(build("stored", dispatch="reference"))
    a, _, _ = run(model, stream)
    b, _, _ = run(model, _future(stream))
    assert float((a[:, :CUT] - b[:, :CUT]).abs().max()) > 100 * ROUNDING


# -- identity: wrong blocks, wraparound, reset, sessions ------------------------------------------------------------

def test_an_incorrect_block_is_never_accepted(stream, monkeypatch):
    """The ring refuses every address it does not hold live; and keys served for the wrong blocks (rolled by one
    among the candidates) change the selections and logits, so the parity tests would reject them."""
    _, stored = pair()
    _, _, state = run(stored, stream)
    ring = state.ring
    lo, hi = ring.live()
    for bad in ([0], [lo - 1], [hi], [hi + 5]):              # evicted, at the edge, not written, never existed
        with pytest.raises(StaleAddressError):
            ring.read_fine(torch.tensor(bad))
    before = ring.stats["stale_rejections"]
    assert ring.read_fine(torch.tensor([lo, hi - 1])).shape == (2, 2, 16, 8) and before == 4
    reference, stored = pair()
    want, want_seen, _ = run(reference, stream)
    original = EvidenceRing.read_fine
    monkeypatch.setattr(EvidenceRing, "read_fine", lambda self, blocks: original(self, blocks).roll(1, dims=1))
    got, got_seen, _ = run(stored, stream)
    assert not torch.equal(want_seen["selected"], got_seen["selected"]) or \
        float((want - got).abs().max()) > 1e6 * ROUNDING


def test_the_stored_keys_are_the_keys_of_the_block_their_slot_names(stream):
    """Exact source identification: for every live block, the stored keys equal the keys recomputed from that
    block's own token ids, reread from the ring (to ROUNDING: another batch); after wraparound the slot of an
    overwritten block holds its successor's keys and the overwritten address is refused."""
    _, stored = pair()
    _, _, state = run(stored, stream)
    ring = state.ring
    lo, hi = ring.live()
    blocks = torch.arange(lo, hi)
    tokens, valid = ring.reread(blocks, stored.evidence.overlap)
    with torch.no_grad():
        recomputed = stored.lookup.fine_keys(stored._encode(tokens, valid))
    assert float((ring.read_fine(blocks) - recomputed).abs().max()) <= ROUNDING
    assert int(ring.ids[lo % ring.slots]) == lo and lo >= ring.slots           # the slot was reused
    with pytest.raises(StaleAddressError):
        ring.read_fine(torch.tensor([lo - ring.slots]))


def test_reset_clears_the_tier_and_sessions_never_share_one(stream):
    _, stored = pair()
    with ContextSession(stored, batch=2) as a, ContextSession(stored, batch=2) as b:
        a.prefill(stream[:, :300])
        b.prefill(stream.flip(1)[:, :300])
        assert a.state.ring.fine.data_ptr() != b.state.ring.fine.data_ptr()
        assert not torch.equal(a.state.ring.fine, b.state.ring.fine)
        a.reset()
        assert not bool(a.state.ring.fine.any()) and a.state.fine_revision is None and not a.state.fine_stale
        fresh = a.prefill(stream[:, :200]).last_logits
        with ContextSession(stored, batch=2) as c:
            assert torch.equal(fresh, c.prefill(stream[:, :200]).last_logits)
        assert a.state.ring.epoch == 1 and a.state.ring.session != b.state.ring.session


# -- weight changes, training ------------------------------------------------------------------------------------

def test_a_weight_change_retires_the_stored_keys_until_reset(stream, monkeypatch):
    """No stale features: once a parameter the keys depend on changes, the reference path runs (and agrees with the
    reference model under the same change) until reset. A blind fingerprint would serve stale keys: they differ."""
    def changed_run(blind):
        reference, stored = pair()
        if blind:
            monkeypatch.setattr(evidence_model, "fine_fingerprint", lambda model: ())
        states = [reference.new_state(2), stored.new_state(2)]
        outs = [[], []]
        with torch.no_grad():
            for at in range(0, 600, PIECE):
                if at == 280:
                    for model in (reference, stored):
                        model.encoder.delta.weight.add_(0.05)          # the same in-place change to both
                for i, model in enumerate((reference, stored)):
                    outs[i].append(model(stream[:, at:at + PIECE], states[i]))
        monkeypatch.undo()
        return torch.cat(outs[0], 1), torch.cat(outs[1], 1), states[1], stored

    want, got, state, stored = changed_run(blind=False)
    assert float((want - got).abs().max()) <= ROUNDING
    assert state.fine_stale and state.counters["fine_tier_invalidations"] == 1
    reads = state.counters["fine_tier_reads"]
    with torch.no_grad():
        stored(stream[:, :PIECE], state)
    assert state.counters["fine_tier_reads"] == reads                         # retired: no read after the change
    state.reset()
    _, _, again = run(stored, stream[:, :300], state=state)
    assert not again.fine_stale and again.counters["fine_tier_reads"] > 0       # reset: a fresh, valid tier
    want, got, _, _ = changed_run(blind=True)                                  # NEGATIVE CONTROL
    assert float((want - got).abs().max()) > 1e6 * ROUNDING
    with ContextSession(stored, batch=2) as session:                           # sessions refuse a changed model
        session.prefill(stream[:, :100])
        with torch.no_grad():
            stored.lookup.fine_k.weight.add_(1e-3)
        with pytest.raises(ModelRevisionError):
            session.step(stream[:, 100])


def test_training_never_uses_the_stored_keys_and_keeps_every_gradient():
    """With gradients the reference path runs: the losses and every gradient equal the reference model's bit for
    bit, and the tier is never read."""
    data = records(seed=5, length=280)
    tokens, labels = data.tokens, data.query_source()
    grads = []
    for fine_keys in ("reread", "stored"):
        model = build(fine_keys)
        model.train()
        state = model.new_state(2)
        total = 0.0
        for at in range(0, 280, PIECE):
            logits, diag = model(tokens[:, at:at + PIECE], state, evidence=labels[:, at:at + PIECE],
                                 teacher=True, diagnostics="losses")
            losses = diag.get("losses", {})
            total = total + logits.square().mean() + sum(v for k, v in losses.items() if not k.endswith("_n"))
        total.backward()
        grads.append({n: p.grad.clone() for n, p in model.named_parameters() if p.grad is not None})
        if fine_keys == "stored":
            assert "fine_tier_reads" not in state.counters and state.counters["fine_tier_bypassed"] > 0
    assert grads[0].keys() == grads[1].keys() and any(n.startswith("lookup.fine") for n in grads[0])
    assert all(torch.equal(grads[0][n], grads[1][n]) for n in grads[0])


# -- with the R6 cache -------------------------------------------------------------------------------------------

def test_the_reread_cache_still_serves_the_injected_blocks_and_changes_nothing(stream):
    _, stored = pair()
    probes = list(range(64, 600, 5))
    with ContextSession(stored, batch=2) as plain, ContextSession(stored, batch=2, cache_bytes=1 << 15) as cached:
        want = plain.prefill(stream, probes=probes).probes
        got = cached.prefill(stream, probes=probes).probes
        assert all(float((want[p] - got[p]).abs().max()) <= ROUNDING for p in probes)
        stats, base = cached.counters(), plain.counters()
        assert stats["cache_hits"] > 0 and stats["cache_bytes"] <= 1 << 15
        assert stats["blocks_reencoded"] < base["blocks_reencoded"]
        lo = cached.state.ring.live()[0]
        assert all(block >= lo for block in cached.state.cache.entries)


# -- accounting, configuration, checkpoints, zero writes ------------------------------------------------------------

def test_the_tier_is_accounted_exactly():
    for capacity in (256, 4096):
        model = build("stored", capacity=capacity)
        state = model.new_state(2)
        fixed = model.context_interface().fixed_bytes(2)
        assert fixed == state.nbytes()
        assert fixed["index_fine_keys"] == 2 * (capacity // 16) * 16 * 8 * 8   # B x slots x block x fine_dim x 8 B
        reference = build("reread", capacity=capacity)
        assert "index_fine_keys" not in reference.new_state(2).nbytes() and reference.new_state(2).ring.fine is None
    model = frozen(build("stored"))
    with ContextSession(model, batch=2) as session:
        projected, held = session.projected_bytes(), session.nbytes()
        assert projected["state"] == held["total"] and projected["state_by_tier"]["index_fine_keys"] > 0


def test_the_1m_configurations_stay_within_the_context_target_with_stored_keys():
    """P6 budget per stream (1 GiB): EXACT state (bfloat16 index, float32 fine-key tier) plus the PROJECTED scratch of
    a decode step and of a 512-token prefill that reads at every position, at 1M. Stored keys make the state larger
    and the prefill scratch smaller."""
    from research.elpis_torch.experiments import run_r6
    for name in run_r6.SCALES:
        model = run_r6.scale_model(name, capacity=1 << 20)
        before = model.context_interface()
        old_state = before.fixed_bytes(1, {"history_device": "cuda"})["total"]
        old_prefill = before.scratch_bytes(batch=1, chunk=512, options={}, queries=512)["total"]
        model.evidence = model.evidence.replace(fine_keys="stored")
        face = model.context_interface()
        fixed = face.fixed_bytes(1, {"history_device": "cuda"})
        assert fixed["index_fine_keys"] == (1 << 20) // 32 * 32 * 64 * 4      # 256 MiB in float32
        for chunk in (1, 512):
            scratch = face.scratch_bytes(batch=1, chunk=chunk, options={}, queries=chunk)["total"]
            assert fixed["total"] + scratch <= 1 << 30, (name, chunk, fixed["total"] + scratch)
        new_prefill = face.scratch_bytes(batch=1, chunk=512, options={}, queries=512)["total"]
        assert fixed["total"] > old_state and new_prefill < old_prefill


def test_the_option_is_validated_overridable_and_carried_by_checkpoints():
    with pytest.raises(ConfigError):
        EvidenceConfig(fine_keys="cached")
    assert EvidenceConfig().fine_keys == "reread"                               # the R5 reference stays the default
    model = build("reread")
    state = model.state_dict()
    out = evaluation_state_dict(state, model.evidence, model.evidence.replace(fine_keys="stored"))
    assert all(torch.equal(out[k], v) for k, v in state.items())                 # an execution option: no weight moves
    arch = lmt.architecture(build("stored"))
    assert arch["evidence_config"]["fine_keys"] == "stored"
    assert lmt.build_from(arch).evidence.fine_keys == "stored"
    legacy = json.loads(json.dumps(arch))
    del legacy["evidence_config"]["fine_keys"]                                   # an R5 / R6 / R7 record
    assert lmt.build_from(legacy).evidence.fine_keys == "reread"


def test_the_ring_refuses_inconsistent_writes():
    ring = EvidenceRing(1, capacity=64, block=16, per_block=2, dim=4, fine_dim=3)
    ring.append(torch.zeros(1, 32, dtype=torch.long))
    with pytest.raises(ValueError):
        ring.write(0, torch.zeros(1, 2, 2, 4))                                  # a tier needs its keys
    with pytest.raises(ValueError):
        ring.write(0, torch.zeros(1, 2, 2, 4), fine=torch.zeros(1, 2, 16, 2))  # wrong width
    plain = EvidenceRing(1, capacity=64, block=16, per_block=2, dim=4)
    plain.append(torch.zeros(1, 16, dtype=torch.long))
    with pytest.raises(ValueError):
        plain.write(0, torch.zeros(1, 1, 2, 4), fine=torch.zeros(1, 1, 16, 3))  # no tier: refused
    with pytest.raises(ValueError):
        plain.read_fine(torch.tensor([0]))
    ring.write(0, torch.zeros(1, 2, 2, 4), fine=torch.ones(1, 2, 16, 3))
    assert torch.equal(ring.read_fine(torch.tensor([1])), torch.ones(1, 1, 16, 3))
    ring.reset()
    assert not bool(ring.fine.any()) and ring.epoch == 1
    with pytest.raises(StaleAddressError):
        ring.read_fine(torch.tensor([1]))


def test_inference_with_stored_keys_writes_nothing(tmp_path):
    scenario = REPO / "tests" / "research" / "elpis_torch" / "zero_write_scenario.py"
    env = {"PYTHONPATH": str(REPO), "PYTHONDONTWRITEBYTECODE": "1"}
    run_ = subprocess.run([sys.executable, "-B", str(scenario), "--evidence", "--stored-fine-keys", "--audit"],
                          capture_output=True, text=True, cwd=REPO, env=env)
    assert run_.returncode == 0, run_.stderr
    report = json.loads(run_.stdout.strip().splitlines()[-1])
    assert report["write_events"] == [] and report["position"] == 900 and report["fine_tier_reads"] > 0
    control = subprocess.run([sys.executable, "-B", str(scenario), "--evidence", "--stored-fine-keys", "--audit",
                              "--negative-control", str(tmp_path / "x")], capture_output=True, text=True, cwd=REPO,
                             env=env)
    assert json.loads(control.stdout.strip().splitlines()[-1])["write_events"]           # the detector works
