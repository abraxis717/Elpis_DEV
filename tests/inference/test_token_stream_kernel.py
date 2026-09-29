"""DSV4 stream kernel: bitwise parity with the legacy step, and no hashing on the hot path.

The no-hashing claim is measured with a profiler, not by patching names: every
Python call into ``elpis.identity``, ``elpis.substrate.digests`` or the JSON
encoder, and every C call into ``_hashlib``/``_sha2``/``_json``, made while
``stream_step`` runs is recorded. Canonical identity must never occur. Content
hashing is permitted only as substrate integrity on cold page faults, so its
count must equal the page misses, and it must be zero once pages are resident.
"""
from __future__ import annotations

from collections import Counter
import sys

import numpy as np
import pytest

from elpis.inference.context import initial_snapshot
from elpis.inference.contracts import ContractError
from elpis.inference.drivers.dsv4.fixtures import make_fixture
from elpis.inference.experts import ExpertAdmission
from elpis.inference.global_context import StreamCandidate
from elpis.inference.target import LatentInput
from elpis.substrate.synthetic import SyntheticFileAssets

TOKENS = tuple((7 * i + 3) % 16 for i in range(40))  # > 16 pool candidates: block pruning is exercised
WATCH_FILES = ("elpis/identity.py", "elpis/substrate/digests.py", "json/encoder.py", "json/__init__.py")
HASH_MODULES = ("_hashlib", "_sha2", "_sha256", "_json")


@pytest.fixture
def warm_provider(native_workspace, fms_file_library):
    provider = SyntheticFileAssets(root=native_workspace, library=fms_file_library,
                                   warm_bytes=1 << 20, staging_bytes=1 << 16)
    yield provider
    provider.close()


def _latents(target, context):
    packets = []
    for channel in ("G", "X", "R"):
        projection = target.projections[channel]
        rows = projection.weights.shape[0]
        packets.append(LatentInput(channel, projection.source_schema, "a" * 64, context, projection.digest,
                                   tuple(float((k % 5) - 2) / 8 for k in range(rows))))
    return tuple(packets)


def _same_numbers(stream, legacy):
    assert stream.tokens == legacy.tokens and stream.history == legacy.history
    assert stream.local_keys == legacy.local_keys and stream.local_values == legacy.local_values
    assert stream.pending == legacy.pending
    assert [(c.object_id, c.end_position, c.key, c.value) for c in stream.global_pool] == \
        [(c.object_id, c.end_position, c.key, c.value) for c in legacy.global_pool]
    assert np.asarray(stream.hidden, "<f4").tobytes() == np.asarray(legacy.hidden, "<f4").tobytes()
    assert np.asarray(stream.logits, "<f4").tobytes() == np.asarray(legacy.logits, "<f4").tobytes()


class Recorder:
    def __init__(self):
        self.calls = Counter()

    def __call__(self, frame, event, arg):
        if event == "call":
            name = frame.f_code.co_filename
            if any(name.endswith(w) for w in WATCH_FILES):
                self.calls["py:" + name.rsplit("/", 1)[-1] + ":" + frame.f_code.co_name] += 1
        elif event == "c_call" and (getattr(arg, "__module__", None) or "") in HASH_MODULES:
            self.calls["c:" + arg.__module__ + "." + arg.__name__] += 1

    def run(self, fn):
        sys.setprofile(self)
        try:
            return fn()
        finally:
            sys.setprofile(None)


def _stream(target, admission, context, latents):
    state = target.initial_stream(context)
    for token in TOKENS:
        state = target.stream_step(state, token, admission=admission, latents=latents)
    return state


@pytest.mark.parametrize("scheme", ["DSV41_ENGRAM", "QWEN38_PLE"])
@pytest.mark.parametrize("resident", [False, True], ids=["streamed-experts", "resident-experts"])
def test_stream_step_is_bitwise_equal_to_legacy_step(warm_provider, native_workspace, scheme, resident):
    target, resident_bytes, _ = make_fixture(warm_provider, native_workspace / scheme, scheme_name=scheme)
    context = initial_snapshot().digest
    latents = _latents(target, context)
    experts = resident_bytes if resident else None
    admission = target.admit_stream(resident_experts=experts)

    legacy = target.initial(context)
    stream = target.initial_stream(context)
    assert not hasattr(stream, "digest")
    for token in TOKENS:
        legacy, _ = target.step(legacy, token, expected_state=legacy.digest, latents=latents,
                                resident_experts=experts)
        stream = target.stream_step(stream, token, admission=admission, latents=latents)
        _same_numbers(stream, legacy)
    assert len(stream.global_pool) > 16
    assert all(type(c) is StreamCandidate and not hasattr(c, "source") for c in stream.global_pool)


@pytest.mark.parametrize("resident", [False, True], ids=["streamed-experts", "resident-experts"])
def test_resident_hot_path_does_no_hashing_or_canonical_encoding(warm_provider, native_workspace, resident):
    target, resident_bytes, _ = make_fixture(warm_provider, native_workspace / "t")
    context = initial_snapshot().digest
    admission = target.admit_stream(resident_experts=resident_bytes if resident else None)
    latents = _latents(target, context)
    _stream(target, admission, context, latents)  # fault every page in once

    recorder = Recorder()
    before = warm_provider.stats()
    recorder.run(lambda: _stream(target, admission, context, latents))
    assert warm_provider.stats()["misses"] == before["misses"]
    assert recorder.calls == Counter(), recorder.calls


def test_only_cold_page_integrity_is_hashed_under_memory_pressure(provider, native_workspace):
    f, *_ = provider  # 64-byte warm budget: pages fault continuously
    target, _, _ = make_fixture(f, native_workspace / "cold")
    context = initial_snapshot().digest
    admission = target.admit_stream()
    recorder = Recorder()
    before = f.stats()
    recorder.run(lambda: _stream(target, admission, context, ()))
    misses = f.stats()["misses"] - before["misses"]
    assert misses > 0
    assert recorder.calls == Counter({"py:digests.py:raw_digest": misses, "py:digests.py:raw_sha256": misses,
                                      "c:_hashlib.openssl_sha256": misses}), recorder.calls


def test_admission_is_verified_once_and_required(warm_provider, native_workspace, target):
    model, resident_bytes, _ = make_fixture(warm_provider, native_workspace / "adm")
    first = model.admit_stream()
    recorder = Recorder()
    second = recorder.run(model.admit_stream)
    assert type(second) is ExpertAdmission and recorder.calls == Counter()  # bindings verified once per bank
    state = model.initial_stream(initial_snapshot().digest)
    with pytest.raises(ContractError, match="expert admission"):
        model.stream_step(state, 1, admission=None)
    foreign = target[0].admit_stream()
    with pytest.raises(ContractError, match="expert admission"):
        model.stream_step(state, 1, admission=foreign)
    assert model.stream_step(state, 1, admission=first).tokens == (1,)

    tampered = dict(resident_bytes)
    gate, up, down = tampered[0, 0]
    tampered[0, 0] = (bytes([gate[0] ^ 1]) + gate[1:], up, down)
    with pytest.raises(ContractError, match="INTEGRITY"):
        model.admit_stream(resident_experts=tampered)
    partial = {k: v for k, v in resident_bytes.items() if k != (0, 3)}
    with pytest.raises(ContractError, match="MISSING"):
        model.admit_stream(resident_experts=partial)
