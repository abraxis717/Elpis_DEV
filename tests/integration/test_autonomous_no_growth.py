"""The ordinary autonomous path persists nothing at all, and its volatile resources are released.

Real native libraries (ELPIS_REQUIRE_NATIVE=1 makes them mandatory). Python-level filesystem mutations are
observed through an armed audit hook; native mutations through the filesystem itself. RuntimeCore ABI v8 holds a
volatile ECS_C session (docs/ELPIS_MISSION.md, Amendment 3): no authority is provisioned, so nothing exists before the
trap and nothing may exist after it. One volatile HACF retrieval epoch, borrowed ingress, an explicit K1 anchor and
three canonical turns, and then three sessions of LEARN and QUERY, leave the directory exactly as empty as they found
it. (Until
ABI v8 this left exactly the durable authority's five provisioned files, and before ABI v7 standalone continuity's two
slots: elpis.runtime.persistence names those writers RETIRED.) Repeated epoch create/destroy holds file descriptors,
threads and live owning handles constant, including an epoch dropped without destroy().
"""
from __future__ import annotations

import gc
import os
import sys
import threading

from elpis.pipeline.ingress import QueryIngress
from elpis.runtime import Runtime
from elpis.runtime import persistence as P
from elpis.structure.retrieval.hacf import HacfHandle, build_corpus_and_index

from ..conftest import require_runtime_library, runtime_config
from ._turn_fixtures import LEARN, TEST_CODEC_PIN, ByteTokens, FixtureMap, admitted
from .conftest import DOCS, POSITIVE
from .test_codec_ecs_turn import RATE, k1, world  # noqa: F401  (k1 is a module pytest fixture)

_WRITE_EVENTS = {"os.rename", "os.replace", "os.remove", "os.rmdir", "os.mkdir", "os.link", "os.symlink",
                 "os.truncate", "os.chmod", "os.chown", "os.utime", "shutil.copyfile", "shutil.copytree",
                 "shutil.move", "shutil.rmtree", "sqlite3.connect", "tempfile.mkstemp", "tempfile.mkdtemp"}
_WRITE_FLAGS = os.O_WRONLY | os.O_RDWR | os.O_CREAT | os.O_TRUNC | os.O_APPEND


class _WriteTrap:
    """An audit hook (process-wide, armed only inside ``with``) recording Python-level persistent writes."""
    _installed = None

    def __init__(self):
        self.armed = False
        self.events: list[tuple] = []
        if _WriteTrap._installed is None:
            sys.addaudithook(_WriteTrap._dispatch)
        _WriteTrap._installed = self

    @staticmethod
    def _dispatch(event, args):
        trap = _WriteTrap._installed
        if trap is None or not trap.armed:
            return
        if event == "open":
            path, mode, flags = (tuple(args) + (None, None, None))[:3]
            writes = (isinstance(mode, str) and any(c in mode for c in "wax+")) or (
                isinstance(flags, int) and flags & _WRITE_FLAGS)
            if writes:
                trap.events.append((event, path))
        elif event in _WRITE_EVENTS:
            trap.events.append((event, args[0] if args else None))

    def __enter__(self):
        self.events.clear()
        self.armed = True
        return self

    def __exit__(self, *exc):
        self.armed = False


def _tree(root):
    return sorted((str(p.relative_to(root)), p.stat().st_size if p.is_file() else None)
                  for p in root.rglob("*"))


def test_ordinary_autonomous_path_writes_nothing_at_all(retrieval_library, ingress_library, k1, tmp_path):
    assert P.AUTONOMOUS_WRITERS == frozenset() and P.NATIVE_HOST_AUTONOMOUS_WRITERS == frozenset()
    state_root = tmp_path / "structural-memory"
    config = runtime_config(require_runtime_library(), TEST_CODEC_PIN)   # nothing to provision: a session is memory
    assert _tree(tmp_path) == []
    trap = _WriteTrap()
    with world(k1) as state, trap:
        handle = build_corpus_and_index(retrieval_library, state_root, DOCS)
        try:
            with QueryIngress(ingress_library, handle) as ingress:
                with Runtime(config) as runtime:
                    runtime.run_ingress(ingress, POSITIVE)
                    runtime.anchor_cognition(state)
                    for text in ("one", "two", "three"):
                        runtime.run_turn(state, text, tokenizer=ByteTokens(), codec=admitted(FixtureMap()),
                                         authority=LEARN)
        finally:
            handle.destroy()
    assert trap.events == [], trap.events
    assert not state_root.exists(), "a volatile retrieval epoch created a state root"
    assert _tree(tmp_path) == [], _tree(tmp_path)


def test_sessions_queries_and_learning_write_nothing_across_opens(k1, tmp_path):  # noqa: F811
    """Several sessions in a row (each a fresh genesis), with managed LEARN and QUERY: no file is created anywhere the
    test can see, and no Python-level write happens."""
    from elpis.runtime.cognition import LearnRequest, QueryRequest

    config = runtime_config(require_runtime_library(), TEST_CODEC_PIN)
    trap = _WriteTrap()
    with world(k1) as state, trap:
        for _ in range(3):
            with Runtime(config) as runtime:
                runtime.anchor_cognition(state)
                for text in ("one", "two", "three"):
                    runtime.run_learn(state, LearnRequest(text, ByteTokens(), admitted(FixtureMap()), LEARN))
                    runtime.run_query(state, QueryRequest(text, ByteTokens(), admitted(FixtureMap())))
    assert trap.events == [], trap.events
    assert _tree(tmp_path) == []


def _fds() -> int:
    return len(os.listdir("/proc/self/fd"))


def _live(kind) -> int:
    return sum(1 for o in gc.get_objects() if type(o) is kind)


def test_volatile_epochs_release_every_resource_across_repeated_create_destroy(retrieval_library,
                                                                               ingress_library, tmp_path):
    def cycle(destroy: bool):
        handle = build_corpus_and_index(retrieval_library, tmp_path / "epoch", DOCS)
        with QueryIngress(ingress_library, handle) as ingress:
            ingress.run(POSITIVE)
        assert handle._borrows == 0
        if destroy:
            handle.destroy()
            assert not handle._valid
        # Otherwise the epoch is simply dropped: its finalizer must release it.

    for _ in range(2):                      # warm up lazily created native and Python state
        cycle(True)
    gc.collect()
    fds, threads, handles, ingresses = _fds(), threading.active_count(), _live(HacfHandle), _live(QueryIngress)
    destroyed = []
    original = HacfHandle.destroy

    def counting_destroy(self):
        destroyed.append(self._valid)
        original(self)

    for i in range(16):
        cycle(destroy=i % 2 == 0)
    gc.collect()
    try:
        HacfHandle.destroy = counting_destroy
        cycle(destroy=False)
        gc.collect()
    finally:
        HacfHandle.destroy = original
    assert destroyed == [True], "a dropped epoch was not released by its finalizer"
    assert (_fds(), threading.active_count()) == (fds, threads)
    assert (_live(HacfHandle), _live(QueryIngress)) == (handles, ingresses)
    assert not (tmp_path / "epoch").exists()
