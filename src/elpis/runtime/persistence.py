"""Persistent-writer authority of the runtime: zero autonomously ever-expanding Elpis-owned persistence.

Elpis may hold persistent state; it may not accumulate it monotonically on its own. Every persistent writer
reachable from the canonical runtime (``elpis.runtime`` and the native libraries it loads) is classified here
exactly once, and every public :class:`~elpis.runtime.composition.Runtime` operation is classified as either
part of the ordinary autonomous path or as an explicit operator operation.
``tests/boundary/test_autonomous_persistence.py`` binds this table to the code, and
``tests/integration/test_autonomous_no_growth.py`` proves it on real native libraries under a write trap.

Classes:

* ``FIXED_CAPACITY_AUTONOMOUS``: may be written by the ordinary autonomous path, but its physical footprint is
  fixed or bounded by capacity fixed at construction. It never grows with turns, queries or restarts.
* ``OPERATOR_EXPLICIT_BOUNDED``: written only when an operator (or an explicit maintenance caller) supplies the
  authority for that one write: a one-use capability, an admitted assertion, a provisioning command, a
  preapproved update. Growth is bounded by the number of explicit operator acts, never by autonomous activity.
* ``OFFLINE_RESEARCH_ONLY``: generators and laboratories outside the runtime; never reachable from it.
* ``PROHIBITED_FROM_AUTONOMOUS_RUNTIME``: must not exist on, or be reconnected to, the autonomous path.

This module is declarative. It holds no state, opens nothing and grants nothing.
"""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class WriterClass(str, Enum):
    FIXED_CAPACITY_AUTONOMOUS = "FIXED_CAPACITY_AUTONOMOUS"
    OPERATOR_EXPLICIT_BOUNDED = "OPERATOR_EXPLICIT_BOUNDED"
    OFFLINE_RESEARCH_ONLY = "OFFLINE_RESEARCH_ONLY"
    PROHIBITED_FROM_AUTONOMOUS_RUNTIME = "PROHIBITED_FROM_AUTONOMOUS_RUNTIME"


@dataclass(frozen=True)
class PersistentWriter:
    id: str
    classification: WriterClass
    python: tuple[str, ...]           # "module:qualname" entry points (Python writers)
    native: tuple[str, ...]           # "repo/path.c:symbol" entry points (native writers)
    bound: str


F, O, R, P = (WriterClass.FIXED_CAPACITY_AUTONOMOUS, WriterClass.OPERATOR_EXPLICIT_BOUNDED,
              WriterClass.OFFLINE_RESEARCH_ONLY, WriterClass.PROHIBITED_FROM_AUTONOMOUS_RUNTIME)

WRITERS: tuple[PersistentWriter, ...] = (
    PersistentWriter(
        "continuity_slots", F, (),
        ("native/continuity/src/store.rs:pub struct Store", "native/runtime/src/core.rs:Store"),
        "Exactly two fixed 176-byte slot files in one explicit, exclusively owned directory; each publication "
        "replaces the non-current slot in place. No log, history, compaction or growth."),
    PersistentWriter(
        "fms_posix_cold_store", F, ("elpis.substrate.residency:Context",),
        ("native/substrate/src/fms_pal_posix.c:p_cold_put",),
        "Only behind a caller-constructed FMS Context with an explicit cold root, cold-tier budget and object "
        "capacity fixed at creation. One live owner per root (directory flock); orphans of a crashed owner "
        "are reclaimed on open; dropped objects unlink their backing file."),
    PersistentWriter(
        "canonical_grid81_publication", O, ("elpis.pipeline.canonical.publisher:publish_candidate",), (),
        "Requires the exact one-use promotion capability issued against an explicit operator approval digest. "
        "The canonical root is exchanged atomically and the previous snapshot is removed; one fixed recovery "
        "journal; one ledger row per approved publication."),
    PersistentWriter(
        "publication_ledger_v1", O,
        ("elpis.pipeline.application.durable_ledger:DurableApplicationLedger.append",), (),
        "One row per operator-approved publication or application, on a caller-supplied ledger path."),
    PersistentWriter(
        "application_ledger_v2", O,
        ("elpis.pipeline.application.durable_ledger_v2:DurableApplicationLedgerV2.append",
         "elpis.pipeline.application.application:apply_artifact"), (),
        "One row per explicitly applied artifact against caller-owned shadow state; never reached by a "
        "Runtime operation."),
    PersistentWriter(
        "canonical_candidate_construction", O, ("elpis.pipeline.canonical.candidate:construct_candidate",), (),
        "One candidate tree per explicit promotion, at a caller-chosen root, replaced atomically."),
    PersistentWriter(
        "evolution_child_materialization", O, ("elpis.evolution.promotion:atomic_materialize",), (),
        "Only inside an explicit evolution attempt whose assertion RuntimeCore has durably reserved; one child "
        "workspace per admitted attempt at a caller-chosen destination."),
    PersistentWriter(
        "hgram_fixed_capacity_store", O, (),
        ("native/structure/hacf/src/kernel/hgram_init.c:main",
         "native/structure/hacf/src/kernel/hgram_store.c:elpis_hgram_init_once",
         "native/structure/hacf/src/kernel/hgram_store.c:elpis_hgram_assoc_store_preapproved"),
        "One file provisioned only by the operator command elpis_hgram_init (full physical preallocation); "
        "association updates are explicit preapproved in-place writes inside the provisioned extent. Never "
        "implicitly provisioned or resized; no Python binding; not writable under elpis_autonomous_run."),
    PersistentWriter(
        "semantic_snapshot_publication", O, (),
        ("native/structure/semantic/src/graph/segment_writer.c:semantic_segment_write",
         "native/structure/semantic/src/graph/snapshot_manifest.c:semantic_snapshot_write",
         "native/structure/semantic/src/graph/snapshot_manifest.c:semantic_snapshot_publish_cas",
         "native/structure/semantic/src/graph/active_head_b2b.c:semantic_b2b_head_cas",
         "native/structure/semantic/src/evidence/admission_commit_b2c.c:semantic_b2c_publish_one"),
        "Immutable segments/manifests and a CAS active head, written only by an explicit caller with explicit "
        "admission and snapshot authority. No Python binding, no background or per-turn publisher."),
    PersistentWriter(
        "dsv4_synthetic_fixture_generator", R, ("elpis.inference.drivers.dsv4.fixtures:make_fixture",), (),
        "Deterministic synthetic fixtures written into a caller-supplied provider root; not reachable from "
        "the runtime."),
    PersistentWriter(
        "research_laboratories", R, (), (),
        "Write-once laboratory evidence under research/; never imported by src/."),
    PersistentWriter(
        "persistent_hacf_corpus", P, (),
        ("native/pipeline/ingress/bridge/ingress_bridge.c:elpis_ingress_env_open",),
        "Retired: the corpus is SQLite :memory: only (elpis_corpus_open_ephemeral) and persistent ingress "
        "refuses. A retrieval epoch is volatile and diskless."),
    PersistentWriter(
        "persistent_log_sink", P, (), (),
        "None exists. Stdout/stderr are not Elpis-owned storage; continuity is not a history."),
)

# Every public Runtime operation, exactly once. Ordinary autonomous operations may reach only
# FIXED_CAPACITY_AUTONOMOUS writers; operator operations require explicit authority from their caller.
AUTONOMOUS_OPERATIONS = frozenset({
    "open", "close", "fault", "run_ingress", "admit_retrieval", "admit_context",
    "anchor_cognition", "run_turn", "evolution_authority",
})
OPERATOR_OPERATIONS = frozenset({
    "publish_canonical",   # one-use promotion capability against an operator approval digest
    "evolve",              # an explicit assertion, durably reserved before execution
})
AUTONOMOUS_WRITERS = frozenset(w.id for w in WRITERS if w.classification is F)


def writer(writer_id: str) -> PersistentWriter:
    for w in WRITERS:
        if w.id == writer_id:
            return w
    raise KeyError(writer_id)
