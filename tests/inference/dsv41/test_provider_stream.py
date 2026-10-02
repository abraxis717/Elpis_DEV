"""YTS-R0 qualification: the DSV4.1 tower through a provider stream over the generic port.

The provider is the test-only reference provider (CPU DSV4.1 kernels behind the real
elpis_exec_backend). It must be bitwise identical to DSV41NativeBackend and within the
established tolerance of the NumPy reference, independent of transport chunking,
caching and completion style, with only YTS messages crossing the boundary.
"""
from __future__ import annotations

from hashlib import sha256
import math
from pathlib import Path
import re
import struct
import time

import numpy as np
import pytest

from elpis.inference.contracts import Code, ContractError
from elpis.inference.drivers.dsv41 import stream_protocol as P
from elpis.inference.drivers.dsv41 import target as target_mod
from elpis.inference.drivers.dsv41.config import TowerConfig
from elpis.inference.principal import PrincipalEngine, PrincipalRequest
from elpis.substrate import execution as X

from . import provider_harness as H
from .test_tower import CONTEXT, admission

ROOT = Path(__file__).resolve().parents[3]
IMAGE = 3 * 12 * 8 * 4  # fixture expert image: w1 || w3 || w2


@pytest.fixture
def rig(native_workspace, v41):
    rig = H.make_rig(native_workspace, v41)
    yield rig
    rig.close()


def run(target, tokens, state=None):
    state = state or target.window_initial()
    experts = target.admit_stream()
    snaps = []
    for token in tokens:
        target.window_step(state, token, experts=experts)
        snaps.append(dict(logits=state.logits.copy(), streams=state.layer_streams.copy(), step=state.step,
                          selected=state.selected, counts=tuple(a.count for a in state.attention),
                          history=state.history, metrics=dict(target.last_metrics)))
    return state, snaps


def assert_same_run(a, b, *, streams=True):
    assert len(a) == len(b)
    for position, (x, y) in enumerate(zip(a, b)):
        assert H.bitwise_equal(x["logits"], y["logits"]), position
        if streams:
            assert H.bitwise_equal(x["streams"], y["streams"]), position
        assert (x["step"], x["selected"], x["counts"], x["history"]) == (y["step"], y["selected"], y["counts"],
                                                                         y["history"]), position


def balanced(provider):
    counts = provider.exec.counts
    return (counts["allocated"] == counts["released"] + counts["consumed"]
            and counts["outputs"] == counts["outputs_released"])


def forbid_host_arithmetic(monkeypatch):
    """Any NumPy DSV4.1 arithmetic or synchronous native-backend call fails the test."""
    from elpis.inference.drivers.dsv41 import attention, engram, layer, moe, native_backend, numerics

    def trap(name):
        def fail(*_args, **_kwargs):
            raise AssertionError("host DSV4.1 arithmetic during a provider token: " + name)
        return fail

    monkeypatch.setattr(layer.Layer, "apply", trap("Layer.apply"))
    monkeypatch.setattr(moe.MoE, "apply", trap("MoE.apply"))
    monkeypatch.setattr(attention.Attention, "apply", trap("Attention.apply"))
    monkeypatch.setattr(engram.EngramLayer, "apply", trap("EngramLayer.apply"))
    for module in (numerics, attention, moe, engram, layer, target_mod):
        for name in ("linear", "rms", "hc_pre", "hc_post", "hc_mixes", "rotate", "quant_dequant", "softmax",
                     "sigmoid", "gated_write", "route", "expert", "sparse_attention"):
            if hasattr(module, name):
                monkeypatch.setattr(module, name, trap(f"{module.__name__}.{name}"))
    for name in ("apply_layer", "final_head", "_local_attention", "_compressed_attention",
                 "create_attention_state"):
        monkeypatch.setattr(native_backend.DSV41NativeBackend, name, trap("DSV41NativeBackend." + name))
    monkeypatch.setattr(native_backend.NativeAttentionState, "__init__", trap("NativeAttentionState"))


# ---------------------------------------------------------------------------
# Q1 + Q10: bitwise reference parity and every topology class
# ---------------------------------------------------------------------------

def test_q1_q10_provider_bitwise_native_and_within_numpy_tolerance(rig):
    tokens = rig.tokens(24)
    _, native = run(rig.native_target(rig.fms(), "native"), tokens)
    _, numpy_ref = run(rig.numpy_target(rig.fms(), "numpy"), tokens)
    fms = rig.fms()
    provider = rig.provider()
    target = rig.provider_target(fms, "provider", provider)
    state, ours = run(target, tokens)

    assert_same_run(native, ours)
    worst_streams = worst_logits = 0.0
    for position, (ref, mine) in enumerate(zip(numpy_ref, ours)):
        np.testing.assert_allclose(mine["streams"], ref["streams"], rtol=H.RTOL, atol=H.ATOL,
                                   err_msg=f"position={position}")
        np.testing.assert_allclose(mine["logits"], ref["logits"], rtol=H.RTOL, atol=H.ATOL,
                                   err_msg=f"position={position}")
        assert int(np.argmax(mine["logits"])) == int(np.argmax(ref["logits"]))
        assert (mine["step"], mine["selected"], mine["counts"]) == (ref["step"], ref["selected"], ref["counts"])
        worst_streams = max(worst_streams, float(np.max(np.abs(mine["streams"] - ref["streams"]))))
        worst_logits = max(worst_logits, float(np.max(np.abs(mine["logits"] - ref["logits"]))))
    # Topology: ratio-0 layer 0; ratio-2 owner 1 shared by 2 (index) and 3 (pure consumer);
    # ratio-1 owner 4 is the candidate source for masked 5 and pure consumer 6.
    assert rig.config.compress_ratios == (0, 2, 2, 2, 1, 1, 1)
    assert ours[-1]["counts"][1] > 0 and ours[-1]["counts"][4] > 0
    for snap in ours:
        assert snap["selected"][0] == ()
        assert snap["selected"][2] == snap["selected"][3] and snap["selected"][5] == snap["selected"][6]
    for snap in ours:
        m = snap["metrics"]
        assert m["submissions"] == 22 and m["needs"] == 21 and m["supplies"] == 21
        assert m["polls"] == m["submissions"]  # synchronous provider: one poll per submission
        assert m["staging_high_water"] == P.HEADER_BYTES + P.SUPPLY_PREFIX_BYTES + IMAGE
        assert m["provider_expert_slot_high_water"] == IMAGE
    assert target.store.staged_bytes == 0
    assert target.numerical_profile != rig.native_target(rig.fms(), "native2").numerical_profile
    target.release_window(state)
    provider.close()
    assert provider.state == "CLOSED" and balanced(provider)
    assert all(v == 0 for v in rig.live().values()), rig.live()
    print(f"PASS_YTS_Q1 bitwise==native tokens={len(tokens)} numpy_worst_streams={worst_streams:.3g} "
          f"numpy_worst_logits={worst_logits:.3g}")


# ---------------------------------------------------------------------------
# Q7: transport chunking is invisible; host and provider staging stay bounded
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("part,max_input", [(128, 4096), (384, 1 << 20), (None, 1 << 20)])
def test_q7_chunked_supply_is_bitwise_invariant_and_bounded(rig, part, max_input):
    tokens = rig.tokens(12)
    _, native = run(rig.native_target(rig.fms(), "native"), tokens)
    provider = rig.provider(part_bytes=part, max_input_bytes=max_input)
    target = rig.provider_target(rig.fms(), "provider", provider)
    state, ours = run(target, tokens)
    assert_same_run(native, ours)
    effective = part or IMAGE
    parts = math.ceil(IMAGE / effective)
    for snap in ours:
        m = snap["metrics"]
        assert m["supplies"] == 21 * parts and m["submissions"] == 1 + 21 * parts
        assert m["staging_high_water"] == P.HEADER_BYTES + P.SUPPLY_PREFIX_BYTES + min(effective, IMAGE)
        assert m["provider_expert_slot_high_water"] == IMAGE
    assert provider.totals["request_high_water"] <= max_input
    assert target.store.staged_bytes == 0 and target.store.high_water <= min(effective, IMAGE)
    target.release_window(state)
    provider.close()
    assert balanced(provider) and all(v == 0 for v in rig.live().values())


# ---------------------------------------------------------------------------
# Q8: the optional provider cache never changes results; no cross-model hits
# ---------------------------------------------------------------------------

def test_q8_cache_budgets_are_bitwise_invariant_and_isolated_per_model(rig):
    tokens = rig.tokens(16)
    _, native = run(rig.native_target(rig.fms(), "native"), tokens)
    all_experts = rig.config.layers * (rig.config.expert_count + 1) * IMAGE
    for budget in (0, IMAGE, all_experts):
        provider = rig.provider(expert_cache_bytes=budget)
        target = rig.provider_target(rig.fms(), f"provider-{budget}", provider)
        assert bool(provider.features & P.FEATURE_CACHE) == bool(budget)
        state, ours = run(target, tokens)
        assert_same_run(native, ours)
        # Analytic elision with a cache holding every expert: a layer needs no supply exactly
        # when all of its selected and shared experts were already used at that layer.
        seen, elided_expected = set(), 0
        for snap in ours:
            per_layer = {}
            for code in snap["step"].route:
                layer, expert = divmod(code, rig.config.expert_count + 1)
                per_layer.setdefault(layer, set()).add(expert)
            for layer, experts in per_layer.items():
                elided_expected += all((layer, e) in seen for e in experts)
                seen.update((layer, e) for e in experts)
        hits = sum(s["metrics"]["provider_cache_hits"] for s in ours)
        supplies = sum(s["metrics"]["supplies"] for s in ours)
        elided = sum(s["metrics"]["elided_layers"] for s in ours)
        if budget == all_experts:
            assert hits > 0 and supplies < 21 * len(tokens)
            assert elided == elided_expected and elided > 0
        elif budget == 0:
            assert hits == 0 and supplies == 21 * len(tokens) and elided == 0
        target.release_window(state)
        if budget == all_experts:
            cached = rig.live()["cache_entries"]
            assert cached > 0
            # Same runtime, second model differing in exactly one expert tensor.
            provider.release_model()
            assert rig.live()["cache_entries"] == 0 and rig.live()["models"] == 0
            role = "layers.3.ffn.experts.1.w2"
            native_b = H.variant_target(rig, rig.fms(), "native-b", mutate_role=role, native_backend=rig.native)
            _, ref_b = run(native_b, tokens)
            target_b = H.variant_target(rig, rig.fms(), "provider-b", mutate_role=role, provider_stream=provider)
            assert target_b.store.manifest.digest != target.store.manifest.digest
            state_b, ours_b = run(target_b, tokens)
            assert ours_b[0]["metrics"]["provider_cache_hits"] == 0
            assert_same_run(ref_b, ours_b)
            assert any(not H.bitwise_equal(a["logits"], b["logits"]) for a, b in zip(native, ref_b))
            target_b.release_window(state_b)
        provider.close()
        assert balanced(provider) and all(v == 0 for v in rig.live().values()), rig.live()


# ---------------------------------------------------------------------------
# Q2: FMS fulfilment slower than the whole poll budget costs no polls
# ---------------------------------------------------------------------------

def test_q2_delayed_fms_fulfilment_consumes_no_port_polls(rig):
    tokens = rig.tokens(6)
    provider = rig.provider(poll_limit=3)
    target = rig.provider_target(rig.fms(), "baseline", provider)
    state, baseline = run(target, tokens)
    target.release_window(state)
    provider.close()

    provider = rig.provider(poll_limit=3)  # budget: 3 polls >= 1 ms apart
    target = rig.provider_target(rig.fms(), "delayed", provider)
    state = target.window_initial()
    delays = []

    def observer(event, info):
        if event == "STAGING":
            begin = time.monotonic()
            time.sleep(0.02)  # far beyond the 3-poll budget; nothing is in flight
            delays.append(time.monotonic() - begin)

    state.provider.observer = observer
    _, delayed = run(target, tokens, state)
    assert_same_run(baseline, delayed)
    assert len(delays) == 21 * len(tokens) and min(delays) >= 0.02
    for snap in delayed:
        assert snap["metrics"]["polls"] == snap["metrics"]["submissions"]
    assert provider.state == "READY" and provider.totals["quarantines"] == 0
    target.release_window(state)
    provider.close()
    assert balanced(provider) and all(v == 0 for v in rig.live().values())


# ---------------------------------------------------------------------------
# Q14: completion notification avoids the 1 ms re-poll floor
# ---------------------------------------------------------------------------

def test_q14_notify_avoids_the_poll_floor_and_changes_nothing(rig):
    """Notify makes a device completion observable at once instead of on the >= 1 ms grid.

    The reference provider measures, per submission, the delay from device completion to
    the poll that observes it. Poll counts are reported but are not the criterion: any
    device step longer than the runtime's own >= 1 ms re-poll interval collects timer
    polls with or without notify (see docs/inference/DSV41_PROVIDER_STREAM.md).
    """
    tokens = rig.tokens(6)
    results = {}
    for name, (mode, notify) in dict(sync=(H.MODE_SYNC, 0), notify=(H.MODE_DEVICE, 1),
                                     floor=(H.MODE_DEVICE, 0)).items():
        rig.configure(mode=mode, notify=notify)
        provider = rig.provider()
        target = rig.provider_target(rig.fms(), name, provider)
        before = rig.counters()
        state, snaps = run(target, tokens)
        after = rig.counters()
        submissions = sum(s["metrics"]["submissions"] for s in snaps)
        polls = sum(s["metrics"]["polls"] for s in snaps)
        seconds = sum(s["metrics"]["provider_ns"] for s in snaps) / 1e9
        observed = after["observed"] - before["observed"]
        observe = (after["observe_ns"] - before["observe_ns"]) / observed / 1e9 if observed else 0.0
        need_polls = [p for s in snaps for p in s["metrics"]["polls_need"]]
        results[name] = dict(snaps=snaps, polls=polls / submissions, latency=seconds / submissions,
                             observe=observe, observed=observed, need_polls=need_polls,
                             complete_polls=[s["metrics"]["polls_complete"] for s in snaps])
        target.release_window(state)
        provider.close()
        assert balanced(provider) and all(v == 0 for v in rig.live().values())
    rig.configure()
    assert_same_run(results["sync"]["snaps"], results["notify"]["snaps"])
    assert_same_run(results["sync"]["snaps"], results["floor"]["snaps"])
    assert results["sync"]["polls"] == 1.0 and results["sync"]["observed"] == 0
    n, f = results["notify"], results["floor"]
    assert n["observed"] == f["observed"] == 1 + 22 * len(tokens)  # STREAM_OPEN + token messages
    assert f["observe"] >= 0.0002 and n["observe"] * 4 < f["observe"]
    assert n["latency"] < f["latency"]
    print(f"PASS_YTS_Q14 completion observed after notify={n['observe'] * 1e6:.0f}us floor={f['observe'] * 1e6:.0f}us; "
          f"polls/submission notify={n['polls']:.2f} floor={f['polls']:.2f}; NEED-step polls notify "
          f"max={max(n['need_polls'])} median={sorted(n['need_polls'])[len(n['need_polls']) // 2]}; COMPLETE-step "
          f"polls notify={n['complete_polls']}; latency/submission notify={n['latency'] * 1e6:.0f}us "
          f"floor={f['latency'] * 1e6:.0f}us")


# ---------------------------------------------------------------------------
# Q11: the whole tower through the principal path; trace, replay, profile binding
# ---------------------------------------------------------------------------

def test_q11_principal_sequence_trace_replay_and_numerical_profile(rig):
    v41 = rig.v41
    from elpis.inference.text import ChatMessage
    native_target = rig.native_target(rig.fms(), "native")
    provider = rig.provider()
    target = rig.provider_target(rig.fms(), "provider", provider)
    assert target.model_identity == native_target.model_identity
    assert target.numerical_profile != native_target.numerical_profile

    def generate(t):
        engine = PrincipalEngine(t)
        state = engine.initial(CONTEXT)
        prompt = v41.encode_chat((ChatMessage("user", "Hello, Elpis."),))
        request = PrincipalRequest("yts", prompt, 12, v41.stop_tokens)
        context = admission(t, v41)
        sequence = engine.begin(state, request, context, expected_state=state.digest)
        assert sequence.stop_reason is None
        outputs = [token for token in sequence]
        result = engine.finalize(sequence)
        return engine, state, request, context, outputs, result

    *_, native_outputs, native_result = generate(native_target)
    engine, state, request, context, outputs, result = generate(target)
    assert outputs == native_outputs and result.commit.outputs == native_result.commit.outputs
    assert result.commit.trace == native_result.commit.trace
    assert provider.stream is None and rig.live()["streams"] == 0  # finalize released the stream
    assert engine.replay(state, request, context, result.commit).commit == result.commit
    # A provider commit never replays on a backend with another numerical profile.
    cpu = PrincipalEngine(native_target)
    sequence = cpu.begin(state, request, context, expected_state=state.digest)
    assert sequence.done and sequence._failure == Code.UNSUPPORTED.value
    with pytest.raises(ContractError) as info:
        cpu.replay(state, request, context, result.commit)
    assert info.value.code == Code.IDENTITY
    provider.close()
    assert balanced(provider) and all(v == 0 for v in rig.live().values())


# ---------------------------------------------------------------------------
# Q12: only YTS messages cross; no host arithmetic; exact traffic per token
# ---------------------------------------------------------------------------

def test_q12_no_host_arithmetic_and_exact_yts_traffic(rig, monkeypatch):
    c = rig.config
    tokens = rig.tokens(10)
    _, native = run(rig.native_target(rig.fms(), "native"), tokens)
    provider = rig.provider()
    target = rig.provider_target(rig.fms(), "provider", provider)
    forbid_host_arithmetic(monkeypatch)
    submitted = []
    real_submit = X.Runtime.submit_backend_only

    def submit(self, operation, stage, tag, buffer):
        submitted.append((operation, stage, buffer.size))
        return real_submit(self, operation, stage, tag, buffer)

    monkeypatch.setattr(X.Runtime, "submit_backend_only", submit)
    state = target.window_initial()
    needs = []
    state.provider.observer = lambda event, info: needs.append(info) if event == "PARKED" else None
    experts = target.admit_stream()
    begin_bytes = P.HEADER_BYTES + 8 + len(c.engram_layers) * (16 + c.hash_columns * c.engram_dim * 4)
    need_reply = lambda n: P.HEADER_BYTES + 8 + 4 * c.active_experts + 4 + 4 * len(n.need) + 8 + 16
    complete_bytes = (P.HEADER_BYTES + 72 + 4 * c.vocab + c.layers * (4 * c.active_experts + 12 + 4 * c.index_topk)
                      + 4 * c.layers * c.hc_mult * c.dimension)
    for position, token in enumerate(tokens):
        needs.clear()
        submitted.clear()
        before = rig.counters()
        target.window_step(state, token, experts=experts)
        after = rig.counters()
        m = target.last_metrics
        assert H.bitwise_equal(state.logits, native[position]["logits"])
        assert all(op == P.OPERATION for op, _, _ in submitted)
        assert [stage for _, stage, _ in submitted] == [P.TOKEN_BEGIN] + [P.EXPERT_SUPPLY] * len(needs)
        expected_h2p = begin_bytes + len(needs) * (P.HEADER_BYTES + P.SUPPLY_PREFIX_BYTES + IMAGE)
        expected_p2h = sum(need_reply(n) for n in needs) + complete_bytes
        assert m["bytes_h2p"] == expected_h2p == sum(size for _, _, size in submitted)
        assert m["bytes_p2h"] == expected_p2h
        # The provider measured the same bytes on its side of the port.
        assert after["bytes_in"] - before["bytes_in"] == expected_h2p
        assert after["bytes_out"] - before["bytes_out"] == expected_p2h
        assert m["submissions"] == 1 + len(needs) == after["messages"] - before["messages"]
    target.release_window(state)
    provider.close()
    assert balanced(provider) and all(v == 0 for v in rig.live().values())
    print(f"PASS_YTS_Q12 per token: submissions={1 + 21} h2p={expected_h2p}B p2h={expected_p2h}B")


# ---------------------------------------------------------------------------
# Q9: Engram rows enter only with TOKEN_BEGIN, decoded on the host
# ---------------------------------------------------------------------------

def test_q9_engram_rows_travel_decoded_with_token_begin(rig, monkeypatch):
    from elpis.inference.contracts import RowIdentity
    from elpis.inference.drivers.dsv41 import provider_stream as PS
    provider = rig.provider()
    target = rig.provider_target(rig.fms(), "provider", provider)
    captured = []
    real = PS.P.encode_token_begin

    def encode(token, segments):
        captured.append((token, [(layer, rows.copy()) for layer, rows in segments]))
        return real(token, segments)

    monkeypatch.setattr(PS.P, "encode_token_begin", encode)
    state = target.window_initial()
    experts = target.admit_stream()
    for token in rig.tokens(5):
        history = state.history
        target.window_step(state, token, experts=experts)
        hashed = target.scheme.stream_hash(history, (token,))
        _, segments = captured[-1]
        assert [layer for layer, _ in segments] == list(rig.config.engram_layers) == [1, 5]
        for layer, rows in segments:
            engine = target.rows[layer]
            ids = hashed.rows[0][target._row_index[layer]]
            expected = engine.lookup(tuple(RowIdentity(engine.table.bank.digest, r) for r in ids))
            assert rows.dtype == np.dtype("<f4") and H.bitwise_equal(rows, np.ascontiguousarray(expected))
    target.release_window(state)
    provider.close()


# ---------------------------------------------------------------------------
# Q16: the generic execution port is byte-identical to the authority commit
# ---------------------------------------------------------------------------

PORT_AT_BA1E4F2 = {
    "native/substrate/EXECUTION.md": "921f87dbf54682f7dd4fd620cf1612209aeb28f41589f1c16f6ce785e5e5b3f6",
    "native/substrate/include/elpis/execution.h": "adde353bee9d765c107c8a922a5f9d521baa252552f96c9bcea4f4d47e055bbf",
    "native/substrate/src/execution.c": "148341a5daa5628c0c15a407e59d83de7a39f0a78878193817f7d076a854cdf7",
}


def test_q16_generic_execution_port_is_unchanged():
    for path, digest in PORT_AT_BA1E4F2.items():
        assert sha256((ROOT / path).read_bytes()).hexdigest() == digest, path


def test_yts_header_and_python_codec_agree():
    header = (ROOT / "native/inference/include/elpis/dsv41_stream.h").read_text()

    def value(name):
        match = re.search(rf"\b{name}\s*=\s*(0x[0-9A-Fa-f]+|\d+)u?\b", header)
        assert match, name
        return int(match.group(1), 0)

    pairs = dict(PROTOCOL_VERSION=P.VERSION, OPERATION=P.OPERATION, HEADER_BYTES=P.HEADER_BYTES, MAGIC=P.MAGIC,
                 MODEL_ADMIT_BEGIN=P.MODEL_ADMIT_BEGIN, MODEL_ADMIT_TENSOR=P.MODEL_ADMIT_TENSOR,
                 MODEL_ADMIT_END=P.MODEL_ADMIT_END, STREAM_OPEN=P.STREAM_OPEN, TOKEN_BEGIN=P.TOKEN_BEGIN,
                 EXPERT_SUPPLY=P.EXPERT_SUPPLY, STREAM_RELEASE=P.STREAM_RELEASE, MODEL_RELEASE=P.MODEL_RELEASE,
                 ADMIT_PART_ACK=P.ADMIT_PART_ACK, ADMIT_ACK=P.ADMIT_ACK, OPENED=P.OPENED, NEED=P.NEED,
                 COMPLETE=P.COMPLETE, RELEASED=P.RELEASED, MODEL_RELEASED=P.MODEL_RELEASED)
    for name, expected in pairs.items():
        assert value("ELPIS_DSV41_STREAM_" + name) == expected, name
    roles = {**{k: v for k, v in P.GLOBAL_ROLES.items()}, **P.LAYER_ROLES, **P.EXPERT_ROLES}
    c_names = dict(re.findall(r"ELPIS_DSV41_ROLE_(\w+)\s*=\s*(\d+)", header))
    assert sorted(int(v) for v in c_names.values()) == sorted(roles.values())
    for expr, expected in (("ADMIT_BEGIN_FIXED_BYTES", P.ADMIT_BEGIN_FIXED_BYTES),
                           ("ADMIT_TENSOR_PREFIX_BYTES", P.ADMIT_TENSOR_PREFIX_BYTES),
                           ("SUPPLY_PREFIX_BYTES", P.SUPPLY_PREFIX_BYTES)):
        text = re.search(rf"ELPIS_DSV41_STREAM_{expr}\s*=\s*([^,\n]+)", header).group(1)
        assert eval(text.replace("u", "")) == expected, expr  # arithmetic of integer literals only


# ---------------------------------------------------------------------------
# Pure codec / validator (no native library required)
# ---------------------------------------------------------------------------

def _config():
    return TowerConfig(model="codec", tokenizer="0" * 64, vocab=16, dimension=8, layers=7, heads=2, head_dim=32,
                       rope_dim=8, q_rank=8, o_groups=2, o_rank=4, local_window=4, max_tokens=32,
                       compress_ratios=(0, 2, 2, 2, 1, 1, 1), kv_sources=(1, 4), index_sources=(1, 2, 4, 5),
                       index_heads=2, index_dim=32, index_topk=2, expert_count=4, active_experts=2, expert_dim=12,
                       engram_layers=(1, 5), engram_order=4, engram_heads=2, engram_dim=32, engram_bucket=17,
                       candidate_source=4, candidate_blocks=2, candidate_block_size=2)


def _plan(cache=False, resident=()):
    return P.TokenPlan(_config(), resident=resident, cache=cache, image_bytes=IMAGE, part_bytes=500)


def test_codec_header_roundtrip_and_echo_checks():
    manifest = bytes(range(32))
    request = P.Header(P.TOKEN_BEGIN, 3, 99, 7, 5, P.NONE, 0, manifest)
    reply = P.Header(P.NEED, 3, 99, 7, 5, 2, 0, manifest)
    decoded = P.unpack_header(reply.pack())
    assert decoded == reply
    P.check_reply(decoded, request)
    for bad in (P.Header(P.NEED, 4, 99, 7, 5, 2, 0, manifest), P.Header(P.NEED, 3, 98, 7, 5, 2, 0, manifest),
                P.Header(P.NEED, 3, 99, 8, 5, 2, 0, manifest), P.Header(P.NEED, 3, 99, 7, 6, 2, 0, manifest),
                P.Header(P.OPENED, 3, 99, 7, 5, 2, 0, manifest),
                P.Header(P.NEED, 3, 99, 7, 5, 2, 0, bytes(32))):
        with pytest.raises(ContractError) as info:
            P.check_reply(bad, request)
        assert info.value.code == Code.INTEGRITY
    raw = bytearray(reply.pack())
    raw[100] = 1  # reserved byte
    with pytest.raises(ContractError):
        P.unpack_header(bytes(raw))
    with pytest.raises(ContractError):
        P.unpack_header(reply.pack() + b"x")  # body length mismatch


def test_codec_token_plan_rejects_untrusted_requests():
    need = P.Need(0, (3, 1), (1, 3, 4), 0, 0, 0)
    plan = _plan()
    assert plan.on_need(need) == (1, 0, 500)
    plan.on_supplied(500)
    assert plan.on_need(P.Need(0, (3, 1), (1, 3, 4), 0, 500, 0)) == (1, 500, 500)
    cases = [
        P.Need(0, (3, 9), (3, 4, 9), 0, 0, 0),        # expert out of range
        P.Need(0, (3, 3), (3, 4), 0, 0, 0),           # duplicate selection
        P.Need(0, (3, 1), (3, 1, 4), 0, 0, 0),        # need not ascending
        P.Need(0, (3, 1), (1, 2, 4), 0, 0, 0),        # need not ⊆ selected ∪ shared
        P.Need(0, (3, 1), (1, 4), 0, 0, 0),           # omission without a granted cache
        P.Need(0, (3, 1), (1, 3, 4), 1, 0, 0),        # cursor ahead
        P.Need(0, (3, 1), (1, 3, 4), 0, 0, 1),        # cache-hit map without a cache
        P.Need(9, (3, 1), (1, 3, 4), 0, 0, 0),        # layer out of range
    ]
    for bad in cases:
        with pytest.raises(ContractError) as info:
            _plan().on_need(bad)
        assert info.value.code == Code.INTEGRITY, bad
    plan = _plan()
    plan.on_need(P.Need(2, (3, 1), (1, 3, 4), 0, 0, 0))
    with pytest.raises(ContractError):
        plan.on_need(P.Need(1, (3, 1), (1, 3, 4), 0, 0, 0))  # layer regression
    plan = _plan()
    plan.on_need(P.Need(2, (3, 1), (1, 3, 4), 0, 0, 0))
    with pytest.raises(ContractError):
        plan.on_need(P.Need(3, (3, 1), (1, 3, 4), 0, 0, 0))  # previous layer not supplied
    cached = _plan(cache=True)
    assert cached.on_need(P.Need(0, (3, 1), (3,), 0, 0, 0b101)) == (3, 0, 500)  # 1 and shared cached
    with pytest.raises(ContractError):
        _plan(cache=True).on_need(P.Need(0, (3, 1), (3,), 0, 0, 0b001))  # wrong hit map
    resident = _plan(resident={(0, 1), (0, 3), (0, 4)})
    with pytest.raises(ContractError):
        resident.on_need(P.Need(0, (3, 1), (1, 3, 4), 0, 0, 0))  # a resident expert requested


def test_codec_complete_validation():
    c = _config()
    plan = _plan()
    positions = tuple(() if not c.compress_ratios[i] else (0,) for i in range(c.layers))
    good = P.Complete(np.zeros(c.vocab, "<f4"), tuple((1, 2) for _ in range(c.layers)), (True,) * c.layers,
                      (0, 3, 0, 0, 5, 0, 0), positions, None, {})
    with pytest.raises(ContractError) as info:  # elision without cache or residency
        _plan().on_complete(good)
    assert info.value.code == Code.INTEGRITY
    full = P.TokenPlan(c, resident={(L, e) for L in range(c.layers) for e in range(c.expert_count + 1)},
                       cache=False, image_bytes=IMAGE, part_bytes=500)
    full.on_complete(good)
    nan = P.Complete(np.full(c.vocab, np.nan, "<f4"), good.selected, good.elided, good.counts, good.positions,
                     None, {})
    with pytest.raises(ContractError) as info:
        P.TokenPlan(c, resident=full.resident, cache=False, image_bytes=IMAGE, part_bytes=500).on_complete(nan)
    assert info.value.code == Code.ENCODING
    local = P.Complete(good.logits, good.selected, good.elided, good.counts, ((0,),) + positions[1:], None, {})
    with pytest.raises(ContractError):
        P.TokenPlan(c, resident=full.resident, cache=False, image_bytes=IMAGE, part_bytes=500).on_complete(local)
    beyond = P.Complete(good.logits, good.selected, good.elided, good.counts,
                        positions[:2] + ((7,),) + positions[3:], None, {})
    with pytest.raises(ContractError):
        P.TokenPlan(c, resident=full.resident, cache=False, image_bytes=IMAGE, part_bytes=500).on_complete(beyond)
    del plan


def test_codec_role_codes_cover_every_tower_role():
    from elpis.inference.drivers.dsv41.config import tensor_shapes
    c = _config()
    codes = {P.role_code(name, c.expert_count) for name in tensor_shapes(c)}
    assert len(codes) == len(tensor_shapes(c))
    assert P.role_code("layers.3.ffn.experts.shared.w2", 4) == (66, 3, 4)
    assert P.role_code("rope.compressed", 4) == (5, P.NONE, P.NONE)
    with pytest.raises(ContractError):
        P.role_code("layers.3.ffn.experts.9.w2", 4)
    body = P.encode_admit_begin(c, features=3, part_bytes=500, cache_bytes=0, config_digest="1" * 64)
    assert len(body) == P.ADMIT_BEGIN_FIXED_BYTES + 8 * c.layers
    assert struct.unpack_from("<f", body, 27 * 4 + 4)[0] == np.float32(c.norm_eps)
