from __future__ import annotations

from dataclasses import replace
from pathlib import Path
import tempfile

import pytest

from elpis.inference.context import initial_snapshot
from elpis.substrate.synthetic import SyntheticFileAssets as FMSFileAssets
from elpis.inference.drivers.dsv4.fixtures import make_fixture
from elpis.inference.target import LatentInput
from elpis.inference.transaction import InferenceEngine, InferenceRequest


@pytest.fixture
def ready(native_workspace, fms_file_library):
    root = native_workspace
    directory = Path(tempfile.mkdtemp(prefix="transaction-", dir=root))
    provider = FMSFileAssets(
        root=root,
        library=fms_file_library,
        warm_bytes=64,
        staging_bytes=128,
    )
    target, _, _ = make_fixture(provider, directory / "target")
    runtime = InferenceEngine(target)
    state = runtime.initial(initial_snapshot())
    try:
        yield provider, target, runtime, state
    finally:
        provider.close()


@pytest.mark.parametrize(
    "latents",
    [
        (None,),
        (object(),),
        (LatentInput("G", "structural-latent.r0", "0" * 64, "0" * 64, "0" * 64, ("bad",)),),
        (LatentInput(7, "structural-latent.r0", "0" * 64, "0" * 64, "0" * 64, (0.0,)),),
    ],
)
def test_malformed_latents_are_typed_atomic_failures(ready, latents):
    _, _, runtime, state = ready
    request = InferenceRequest(
        "malformed-latent",
        state.context.digest,
        "PREFILL",
        (1,),
        latents=latents,
    )
    result = runtime.execute(state, request, expected_state=state.digest)
    assert result.state == state
    assert result.receipt.terminal == "FAILED"
    assert result.receipt.failure == "INVALID"
    assert result.receipt.target_steps == ()


@pytest.mark.parametrize(
    ("field", "mutator"),
    [
        ("scheme", lambda r: "tampered-scheme"),
        ("parameters", lambda r: "0" * 64),
        ("associative_result", lambda r: "0" * 64),
        ("bank", lambda r: "0" * 64),
        ("rows", lambda r: r.rows + (999999,)),
        ("global_index", lambda r: "0" * 64),
        ("latents", lambda r: r.latents + ("0" * 64,)),
        ("expert_route", lambda r: r.expert_route + (999999,)),
        ("expert_contents", lambda r: r.expert_contents + ("0" * 64,)),
        ("assets", lambda r: r.assets + ("0" * 64,)),
    ],
)
def test_unknown_tampered_state_replays_and_rejects_full_receipt_provenance(
    ready, field, mutator
):
    _, target, runtime, initial = ready
    prefill = InferenceRequest(
        "prefill-provenance",
        initial.context.digest,
        "PREFILL",
        (1, 2, 3, 4),
    )
    committed = runtime.execute(initial, prefill, expected_state=initial.digest)
    assert committed.receipt.terminal == "COMMITTED"

    first = committed.state.receipts[0]
    tampered_receipt = replace(first, **{field: mutator(first)})
    tampered = replace(
        committed.state,
        receipts=(tampered_receipt,) + committed.state.receipts[1:],
    )

    restarted = InferenceEngine(target)
    request = InferenceRequest(
        "continue-" + field,
        tampered.context.digest,
        "GREEDY",
        count=1,
    )
    result = restarted.execute(
        tampered,
        request,
        expected_state=tampered.digest,
    )
    assert result.state == tampered
    assert result.receipt.terminal == "FAILED"
    assert result.receipt.failure == "IDENTITY"


def test_fresh_runtime_revalidates_untampered_history_and_continues(ready):
    _, target, runtime, initial = ready
    prefill = InferenceRequest(
        "prefill-restart",
        initial.context.digest,
        "PREFILL",
        (1, 2, 3, 4),
    )
    committed = runtime.execute(initial, prefill, expected_state=initial.digest)

    restarted = InferenceEngine(target)
    request = InferenceRequest(
        "continue-restart",
        committed.state.context.digest,
        "GREEDY",
        count=1,
    )
    result = restarted.execute(
        committed.state,
        request,
        expected_state=committed.state.digest,
    )
    assert result.receipt.terminal == "COMMITTED"
    assert len(result.state.neural.tokens) == len(committed.state.neural.tokens) + 1


def test_same_runtime_uses_validated_state_cache_for_normal_continuation(ready, monkeypatch):
    _, target, runtime, initial = ready
    prefill = InferenceRequest(
        "prefill-cache",
        initial.context.digest,
        "PREFILL",
        (1, 2, 3, 4),
    )
    committed = runtime.execute(initial, prefill, expected_state=initial.digest)

    calls = 0
    real_step = target.step

    def counted(*args, **kwargs):
        nonlocal calls
        calls += 1
        return real_step(*args, **kwargs)

    monkeypatch.setattr(target, "step", counted)
    request = InferenceRequest(
        "continue-cache",
        committed.state.context.digest,
        "GREEDY",
        count=1,
    )
    result = runtime.execute(
        committed.state,
        request,
        expected_state=committed.state.digest,
    )
    assert result.receipt.terminal == "COMMITTED"
    assert calls == 1

def test_noncanonical_malformed_latent_failure_receipt_is_deterministic(ready):
    _, _, runtime, state = ready
    first = InferenceRequest(
        "malformed-deterministic",
        state.context.digest,
        "PREFILL",
        (1,),
        latents=(object(),),
    )
    second = InferenceRequest(
        "malformed-deterministic",
        state.context.digest,
        "PREFILL",
        (1,),
        latents=(object(),),
    )
    a = runtime.execute(state, first, expected_state=state.digest)
    b = runtime.execute(state, second, expected_state=state.digest)
    assert a.state == b.state == state
    assert a.receipt.terminal == b.receipt.terminal == "FAILED"
    assert a.receipt.failure == b.receipt.failure == "INVALID"
    assert a.receipt.request == b.receipt.request
    assert a.receipt.digest == b.receipt.digest

