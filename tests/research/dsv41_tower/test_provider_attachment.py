"""Attachment/lifetime proofs using the existing sealed YTS reference DSO.

No model, tokenizer, replacement provider ABI, or Python worker callback is
needed to exercise construction, ownership and independent execution contexts.
"""
import ctypes as C
import gc
from hashlib import sha256
import json
from concurrent.futures import ThreadPoolExecutor
from threading import Barrier
from types import SimpleNamespace
import weakref

import pytest

from elpis.substrate import execution as X
from elpis.substrate.authority import PinnedAuthority
from elpis.substrate.boundary import RootCapability, load_native, _LOADED_NATIVE_OBJECTS
from elpis.substrate.contracts import ContractError
from research.dsv41_tower import provider_stream as S
from research.dsv41_tower import stream_protocol as P
from .provider_harness import RefConfig, RefCounters, LIVE, MODE_DEVICE
from ...conftest import require_native_library


@pytest.fixture
def attachment(tmp_path):
    entries = []
    for stem in ("elpis_execution", "elpis_dsv41_stream_reference_provider"):
        data = require_native_library(stem).read_bytes()
        (tmp_path / (stem + ".so")).write_bytes(data)
        entries.append(dict(library_id=stem, size=len(data), sha256=sha256(data).hexdigest()))
    document = json.dumps(dict(schema="elpis.inference-authority.v1", source="attachment fixture",
                               provenance="synthetic-test", assets=[], libraries=entries)).encode()
    authority = PinnedAuthority(document, expected_sha256=sha256(document).hexdigest())
    provider_id = entries[1]["library_id"]
    with RootCapability(tmp_path) as root:
        lib = load_native(root, provider_id + ".so", authority.libraries[provider_id])
    lib.elpis_dsv41_reference_provider_configure.argtypes = [C.POINTER(RefConfig)]
    lib.elpis_dsv41_reference_provider_configure.restype = None
    lib.elpis_dsv41_reference_provider_counters.argtypes = [C.POINTER(RefCounters)]
    lib.elpis_dsv41_reference_provider_counters.restype = None
    lib.elpis_dsv41_stream_provider_bind_runtime.argtypes = [C.c_void_p, C.c_void_p]
    lib.elpis_dsv41_stream_provider_bind_runtime.restype = C.c_int
    lib.elpis_dsv41_stream_provider_detach.argtypes = [C.c_void_p]
    lib.elpis_dsv41_stream_provider_detach.restype = None
    lib.elpis_dsv41_reference_provider_configure(C.byref(RefConfig(MODE_DEVICE, 1, 0, 0, -1)))
    owners = []

    def create():
        p = S.DSV41StreamProvider(tmp_path, execution_library="elpis_execution.so",
                                 provider_library=provider_id + ".so", authority=authority,
                                 execution_id="elpis_execution", provider_id=provider_id)
        owners.append(p)
        return p

    def live():
        counts = RefCounters()
        lib.elpis_dsv41_reference_provider_counters(C.byref(counts))
        return {name: getattr(counts, name) for name in LIVE}

    yield SimpleNamespace(create=create, lib=lib, live=live, root=tmp_path, owners=owners, entries=entries)
    for p in owners:
        p.close()
    lib.elpis_dsv41_reference_provider_configure(None)
    assert not any(live().values())


@pytest.mark.parametrize("failure", ["status", "exception"])
def test_bind_failure_destroys_runtime_before_detach(attachment, monkeypatch, failure):
    runtimes, events, contexts = [], [], []
    real_runtime = X.Runtime
    real_bind = attachment.lib.elpis_dsv41_stream_provider_bind_runtime
    real_detach = attachment.lib.elpis_dsv41_stream_provider_detach

    def runtime(*args, **kwargs):
        r = real_runtime(*args, **kwargs)
        runtimes.append(r)
        return r

    def bind(context, handle):
        contexts.append(context)
        assert real_bind(context, handle) == 0  # provider may already retain runtime
        if failure == "exception":
            raise KeyboardInterrupt("injected bind interruption")
        return -1

    def detach(context):
        events.append(("detach", runtimes[0].closed))
        real_detach(context)

    monkeypatch.setattr(X, "Runtime", runtime)
    monkeypatch.setattr(attachment.lib, "elpis_dsv41_stream_provider_bind_runtime", bind)
    monkeypatch.setattr(attachment.lib, "elpis_dsv41_stream_provider_detach", detach)
    try:
        with pytest.raises(KeyboardInterrupt if failure == "exception" else ContractError):
            attachment.create()
        assert runtimes[0].closed and events == [("detach", True)]
        assert not any(attachment.live().values())
    finally:  # also keep the deliberate pre-fix regression run isolated
        if runtimes and not runtimes[0].closed:
            runtimes[0].destroy()
        if contexts and not events:
            real_detach(contexts[0])


@pytest.mark.parametrize("method", ["handle", "take", "metrics", "notify", "submit"])
def test_destroyed_runtime_never_reenters_native_code(attachment, monkeypatch, method):
    p = attachment.create()
    r = p.runtime
    p.close()
    b = p.exec.buffer(1)

    def forbidden(*args):
        pytest.fail("freed runtime pointer reached native entry point")

    for name in ("elpis_exec_take", "elpis_exec_get_metrics", "elpis_exec_notify", "elpis_exec_submit"):
        monkeypatch.setattr(p.exec._lib, name, forbidden)
    try:
        with pytest.raises(ContractError, match="CLOSED"):
            if method == "handle":
                r.handle
            elif method == "submit":
                r.submit_backend_only(P.OPERATION, 0, 0, b)
            elif method == "take":
                r.take(0)
            else:
                getattr(r, method)()
        r.destroy()
        r.shutdown()
        p.close()  # public destruction remains idempotent
    finally:
        b.release()


def test_runtime_create_failure_detaches_context(attachment, monkeypatch):
    def fail(*args, **kwargs):
        raise MemoryError("injected construction failure")
    monkeypatch.setattr(X, "Runtime", fail)
    with pytest.raises(MemoryError):
        attachment.create()
    assert not any(attachment.live().values())


@pytest.mark.parametrize("fault", ["version", "missing-symbol", "missing-callback"])
def test_attachment_abi_refuses_before_use_and_releases_context(attachment, monkeypatch, fault):
    lib = attachment.lib
    if fault == "version":
        monkeypatch.setattr(lib, "elpis_dsv41_stream_provider_abi_version", lambda: 999)
    elif fault == "missing-symbol":
        real_load = S.load_native

        class MissingEntry:
            def __getattr__(self, name):
                if name == "elpis_dsv41_stream_provider_bind_runtime":
                    raise AttributeError(name)
                return getattr(lib, name)

        monkeypatch.setattr(S, "load_native", lambda *a: (real_load(*a), MissingEntry())[1])
    else:
        attach = lib.elpis_dsv41_stream_provider_attach

        def malformed(host, backend):
            rc = attach(host, backend)
            C.cast(backend, C.POINTER(X.Backend)).contents.poll = None
            return rc
        monkeypatch.setattr(lib, "elpis_dsv41_stream_provider_attach", malformed)
    with pytest.raises(ContractError, match="UNSUPPORTED"):
        attachment.create()
    assert not any(attachment.live().values())


def test_sealed_code_outlives_owner_gc_and_pathnames(attachment):
    p = attachment.create()
    lib, identity, runtime = p._lib, p.identity, p.runtime
    ref = weakref.ref(p)
    for path in attachment.root.glob("*.so"):
        path.unlink()
    p.close()
    attachment.owners.clear()
    del p
    gc.collect()
    assert ref() is None and runtime.closed
    fd, cached = _LOADED_NATIVE_OBJECTS[(identity.size, identity.sha256)]
    assert cached is lib and lib.elpis_dsv41_stream_provider_abi_version() == 1
    import os
    assert os.fstat(fd).st_size == identity.size


@pytest.mark.parametrize("data", [b"not an ELF", b"\x7fELF\x02\x01\x01" + bytes(17)])
def test_pinned_malformed_or_truncated_dso_is_not_attached(attachment, data):
    # Authorize these exact synthetic bytes so rejection exercises dlopen,
    # rather than merely failing the already-qualified size/digest boundary.
    entry = attachment.entries[1]
    path = entry["library_id"] + ".so"
    (attachment.root / path).write_bytes(data)
    libraries = [attachment.entries[0], dict(entry, size=len(data), sha256=sha256(data).hexdigest())]
    document = json.dumps(dict(schema="elpis.inference-authority.v1", source="malformed fixture",
                               provenance="synthetic-test", assets=[], libraries=libraries)).encode()
    authority = PinnedAuthority(document, expected_sha256=sha256(document).hexdigest())
    with pytest.raises(ContractError, match="sealed native load"):
        S.DSV41StreamProvider(attachment.root, execution_library="elpis_execution.so",
                             provider_library=path, authority=authority, execution_id="elpis_execution",
                             provider_id=entry["library_id"])
    assert not any(attachment.live().values())


def test_separate_runtimes_attach_concurrently_and_close_independently(attachment):
    attached, retired = Barrier(4), Barrier(4)

    def run(index):
        p = attachment.create()
        context = p._backend.context
        attached.wait(timeout=10)
        b = p.exec.buffer(1)
        try:
            assert p.runtime.submit_backend_only(P.OPERATION - 1, 0, 0, b)[0] == X.BACKEND_UNAVAILABLE
            assert b.state == "OWNED"  # unsupported operation consumes nothing
            if index:
                # Deliberately invalid YTS payload: native provider token still
                # exercises submit/poll/notify and independent context teardown.
                assert p.runtime.submit_backend_only(P.OPERATION, 0, index, b) == (X.OK, 0)
            else:
                p.close()  # must not invalidate any other context or its token
            retired.wait(timeout=10)
            if index:
                rc, result = p.runtime.take(10000)
                assert rc == X.OK and result.status == X.INVALID
                assert p.runtime.metrics()["backend_fallback"] == 0
        finally:
            b.release()
            p.close()
        return context

    with ThreadPoolExecutor(max_workers=4) as pool:
        contexts = list(pool.map(run, range(4)))
    assert len(set(contexts)) == 4
    assert not any(attachment.live().values())
