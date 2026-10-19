"""The ordinary autonomous path persists nothing but its durable authority's fixed files, and its volatile resources
are released.

Real native libraries (ELPIS_REQUIRE_NATIVE=1 makes them mandatory). Python-level filesystem mutations are
observed through an armed audit hook; native mutations through the filesystem itself. The operator provisions the
ECS_C authority (outside the trap). One volatile HACF retrieval epoch, borrowed ingress, an explicit K1 anchor and
three canonical turns may then leave exactly the authority's five provisioned files behind, at their provisioned sizes
and inodes (elpis.runtime.persistence: ecs_c_authority_slots), and nothing else. Until RuntimeCore ABI v7 this was
standalone continuity's two fixed 176-byte slots, which retired into the authority. Repeated epoch create/destroy
holds file descriptors, threads and live owning handles constant, including an epoch dropped without destroy().
"""
from __future__ import annotations

import gc
import os
import sys
import threading

from elpis.pipeline.ingress import QueryIngress
from elpis.runtime import Runtime, RuntimeConfig
from elpis.runtime import persistence as P
from elpis.structure.retrieval.hacf import HacfHandle, build_corpus_and_index

from ..conftest import TEST_AUTHORITY_CAPACITY, require_runtime_library, runtime_config
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


def _inodes(root):
    return sorted((str(p.relative_to(root)), p.stat().st_size, p.stat().st_ino) for p in root.rglob("*") if p.is_file())


_RECORD = 1344 + 160 * TEST_AUTHORITY_CAPACITY
_AUTHORITY_TREE = [("authority", None), ("authority/deployment.key", 32), ("authority/witness.a", 240),
                   ("authority/witness.b", 240), ("state", None), ("state/authority.a", _RECORD),
                   ("state/authority.b", _RECORD)]


def test_ordinary_autonomous_path_leaves_only_the_authoritys_fixed_files(retrieval_library, ingress_library,
                                                                          k1, tmp_path):
    assert P.AUTONOMOUS_WRITERS == {"ecs_c_authority_slots", "fms_posix_cold_store", "k1_checkpoint_slots"}
    state_root, authority = tmp_path / "structural-memory", tmp_path / "authority"
    config = runtime_config(authority, require_runtime_library(), TEST_CODEC_PIN)   # the operator provisions
    provisioned = _inodes(authority)
    assert _tree(authority) == _AUTHORITY_TREE, _tree(authority)
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
    assert _tree(tmp_path) == [("authority", None)] + [("authority/" + n, s) for n, s in _AUTHORITY_TREE], \
        _tree(tmp_path)
    assert _inodes(authority) == provisioned, "the authority's files are rewritten in place, never replaced"


def test_a_provisioned_k1_checkpoint_stays_two_fixed_slots_on_the_autonomous_path(k1, tmp_path):  # noqa: F811
    """K1 Recovery R0: the operator provisions the store (outside the trap); the autonomous path then rewrites
    its two slots in place and creates, grows or renames nothing (elpis.runtime.persistence: k1_checkpoint_slots)."""
    from dataclasses import replace
    from elpis.runtime.cognition import LearnRequest, QueryRequest
    from elpis.runtime.core import RuntimeLibrary
    from elpis.runtime.recovery import provision_k1_checkpoint
    from ..conftest import library_id_of, native_authority
    from .test_codec_ecs_turn import DIM, WIDTH

    path = require_runtime_library()
    library = RuntimeLibrary.admit(path.parent, path.name, native_authority(), library_id_of(path))
    checkpoint = tmp_path / "k1-checkpoint"
    provision_k1_checkpoint(library, checkpoint, k1.envelope_bytes(DIM, WIDTH))
    slots = sorted((p.name, p.stat().st_size, p.stat().st_ino) for p in checkpoint.iterdir())
    config = replace(runtime_config(tmp_path / "authority", path, TEST_CODEC_PIN), k1_checkpoint_dir=checkpoint)
    authority = _inodes(tmp_path / "authority")
    trap = _WriteTrap()
    with world(k1) as state, trap:
        with Runtime(config) as runtime:
            runtime.anchor_cognition(state)
            for text in ("one", "two", "three"):
                runtime.run_learn(state, LearnRequest(text, ByteTokens(), admitted(FixtureMap()), LEARN))
                runtime.run_query(state, QueryRequest(text, ByteTokens(), admitted(FixtureMap())))
            assert runtime.recover_k1().disposition == "RESUMABLE"
    assert trap.events == [], trap.events
    assert sorted((p.name, p.stat().st_size, p.stat().st_ino) for p in checkpoint.iterdir()) == slots
    assert _tree(tmp_path / "authority") == _AUTHORITY_TREE
    assert _inodes(tmp_path / "authority") == authority


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
