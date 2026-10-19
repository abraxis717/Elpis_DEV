"""Persistent-writer authority of the runtime: zero autonomously ever-expanding Elpis-owned persistence.

Elpis may hold persistent state; it may not accumulate it monotonically on its own. Every persistent writer
reachable from either composer is classified here exactly once: the native host ``elpis`` (the production entry
point, docs/NATIVE_HOST.md) and the Python ``elpis.runtime`` (the offline reference interface) with the native
libraries it loads. Every public :class:`~elpis.runtime.composition.Runtime` operation is classified as either part
of the ordinary autonomous path or as an explicit operator operation.
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
_TOPOLOGY = "native/structure/semantic/src/topology/"
_GRID81 = "native/structure/semantic/src/grid81/"

WRITERS: tuple[PersistentWriter, ...] = (
    PersistentWriter(
        "k1_checkpoint_slots", F, (),
        ("native/runtime/src/checkpoint.rs:pub fn write", "native/runtime/src/checkpoint.rs:pub fn retract"),
        "K1 Recovery R0: exactly two slot files of one fixed size (header + the admitted K1 shape's envelope) in "
        "one operator-provisioned, exclusively locked directory. The anchor and each explicit durable publication "
        "rewrite the slot that does not hold the authorized state in place (a LEARN never does); the runtime never "
        "creates, grows, resizes, renames or multiplies them. No history; a slot never authorizes itself (the durable "
        "authority decides)."),
    PersistentWriter(
        "ecs_c_authority_slots", F, (),
        ("native/ECS_C/authority/src/store.rs:pub fn transition", "native/runtime/src/durable.rs:pub struct Stores"),
        "The ECS_C authority (docs/ECS_C_AUTHORITY.md): exactly two record slots of R(capacity) bytes in the state "
        "directory and two 240-byte witness slots in the authority directory, all created by provisioning. Each "
        "transition rewrites one record slot and one witness slot in place (R + 240 bytes, two fdatasync calls), "
        "whatever the length of the history; nothing is created, renamed, removed or grown, and the entity rows are "
        "bounded by the provisioned capacity (exhaustion refuses before any write). Reached from RuntimeCore (ABI "
        "v6; its only durable authority since ABI v7, when standalone continuity's two 176-byte slots retired into it) "
        "only by an explicit publication or operator event; a LEARN, OBSERVE or QUERY never writes it."),
    PersistentWriter(
        "fms_posix_cold_store", F, ("elpis.substrate.residency:Context",),
        ("native/substrate/src/fms_pal_posix.c:p_cold_put",),
        "Only behind a caller-constructed FMS Context with an explicit cold root, cold-tier budget and object "
        "capacity fixed at creation. One live owner per root (directory flock); orphans of a crashed owner "
        "are reclaimed on open; dropped objects unlink their backing file."),
    PersistentWriter(
        "canonical_grid81_publication", O, ("elpis.pipeline.canonical.publisher:publish_candidate",), (),
        "Requires the exact one-use promotion capability issued against an explicit operator approval digest. "
        "Each approved publication adds one append-only generation file and one ledger row; the canonical root is "
        "exchanged atomically and the previous snapshot removed; one fixed recovery journal. Growth is bounded by "
        "the number of operator approvals, never by autonomous activity."),
    PersistentWriter(
        "k1_checkpoint_provisioning", O, ("elpis.runtime.recovery:provision_k1_checkpoint",),
        ("native/runtime/src/checkpoint.rs:pub fn provision",),
        "One new directory with exactly two zeroed, synced slot files per explicit operator command; an existing "
        "directory is refused (never overwritten, resized or adopted). Never reached by a Runtime operation."),
    PersistentWriter(
        "ecs_c_authority_provisioning", O, ("elpis.runtime.authority:provision_authority",),
        ("native/ECS_C/authority/src/store.rs:pub fn provision",),
        "One state directory and one authority directory with their fixed file sets (two record slots; the "
        "deployment key and two witnesses) per explicit operator provisioning, with the sources the operator founds "
        "(each an in-place transition of those files); an existing deployment is refused (ALREADY_PROVISIONED, "
        "AUTHORITY_EXISTS). Never reached by a Runtime operation."),
    PersistentWriter(
        "native_host_deployment", O, (),
        ("native/runtime/host/src/host_deploy.c:host_init", "native/runtime/host/src/host_deploy.c:host_create_lineage"),
        "The native operator host (docs/NATIVE_HOST.md). init creates a home holding exactly host.cfg, catalog and "
        "the seven pinned libraries (write-once files, never replaced) and provisions the ECS_C authority; create "
        "writes lineage.cfg once and provisions the checkpoint store. Each is one explicit operator command, refused "
        "when its files exist. Afterwards the host writes nothing itself: every later write is the ECS_C authority's "
        "or the checkpoint store's fixed slots. No log, cache or history."),
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
        "evolution_child_materialization", O,
        ("elpis.evolution.promotion:atomic_materialize",
         "elpis.evolution.policy:EvolutionPolicyAuthority.materialize"), (),
        "One child workspace per operator-approved promotion grant of the pinned evolution policy "
        "(elpis.evolution.policy), at a caller-chosen destination that must not exist; never reached by a "
        "Runtime operation."),
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
        "semantic_context_artifacts", O, (),
        ("native/structure/semantic/src/context/context_writer.c:elpis_write_requirement_set",
         "native/structure/semantic/src/context/context_writer.c:elpis_write_deficit_policy",
         "native/structure/semantic/src/context/context_writer.c:elpis_write_deficit_report",
         "native/structure/semantic/src/context/context_writer.c:elpis_write_retrieval_requirement",
         "native/structure/semantic/src/context/context_writer.c:elpis_write_retrieval_bundle"),
        "One named file per explicit call, at a path the caller supplies (same-directory temp, fsync, atomic "
        "rename, directory fsync). Not reachable from retrieval, the bridges or any Runtime operation; no Python "
        "binding. The contextual lane (elpis.runtime.context) writes nothing."),
    PersistentWriter(
        "semantic_embedding_files", O, (),
        ("native/structure/semantic/src/embedding/embedding_writer.c:elpis_embedding_write_profile",
         "native/structure/semantic/src/embedding/embedding_writer.c:elpis_embedding_write_vector",
         "native/structure/semantic/src/embedding/embedding_writer.c:elpis_embedding_write_collection"),
        "One named file per explicit call (atomic no-replace publication) at a caller-supplied path. Retrieval "
        "embeds in memory only (the epoch's vector shard is an FMS object); no Python binding."),
    PersistentWriter(
        "semantic_evidence_records", O, (),
        ("native/structure/semantic/src/evidence/evidence_writer.c:elpis_evidence_write_record",),
        "One evidence record file per explicit call at a caller-supplied path; no Python binding; never reached "
        "by retrieval, the bridges or a Runtime operation."),
    PersistentWriter(
        "semantic_view_directory", O, (),
        ("native/structure/semantic/src/view/view_writer.c:elpis_view_write_all",),
        "The fixed file set of one bounded semantic view into a caller-supplied directory, per explicit call; "
        "no Python binding."),
    PersistentWriter(
        "semantic_topology_artifacts", O, (),
        tuple(_TOPOLOGY + f for f in (
            "topology_address.c:elpis_write_topology_addresses", "topology_anchor.c:elpis_write_topology_anchors",
            "topology_bridge.c:elpis_write_topology_bridges", "topology_conflict.c:elpis_write_topology_conflicts",
            "topology_constellation.c:elpis_write_topology_constellations",
            "topology_constraint.c:elpis_write_topology_constraints", "topology_graph.c:elpis_write_topology_graph",
            "topology_metric.c:elpis_write_topology_metric_hints", "topology_persist.c:elpis_write_topology_IR",
            "topology_persist.c:elpis_write_compile_receipt", "topology_persist.c:elpis_write_topology_handoff",
            "topology_policy.c:elpis_write_topology_policy", "topology_registry.c:elpis_write_topology_registry",
            "topology_roles.c:elpis_write_topology_roles")),
        "Offline topology compilation output: one named file per explicit call at a caller-supplied path. No "
        "Python binding; not reachable from retrieval or the runtime."),
    PersistentWriter(
        "semantic_grid81_artifacts", O, (),
        tuple(_GRID81 + f for f in (
            "grid81_capsule.c:elpis_write_grid81_capsule_manifest", "grid81_cell.c:elpis_write_grid81_masks",
            "grid81_codebook.c:elpis_write_grid81_codebook",
            "grid81_compile_receipt.c:elpis_write_grid81_compile_receipt",
            "grid81_constraint_projection.c:elpis_write_grid81_constraint_projections",
            "grid81_handoff.c:elpis_write_grid81_handoff", "grid81_packet.c:elpis_write_grid81_structural_packet",
            "grid81_policy.c:elpis_write_grid81_policy")),
        "Offline Grid81 compilation output: one named file per explicit call at a caller-supplied path (canonical "
        "Grid81 publication is the separate operator-approved publisher). No Python binding."),
    PersistentWriter(
        "dsv4_synthetic_fixture_generator", R, ("elpis.inference.drivers.dsv4.fixtures:make_fixture",), (),
        "Deterministic synthetic fixtures written into a caller-supplied provider root; not reachable from "
        "the runtime."),
    PersistentWriter(
        "native_host_testing_write_probe", R, (),
        ("native/runtime/host/src/elpis_host.c:op_testing_write_probe",),
        "The testing host's containment probe (compiled only with ELPIS_HOST_TESTING, run only over a testing "
        "deployment): one attempted file creation removed at once, an in-place rewrite of one byte with itself, or "
        "one byte past a slot's end, each at a path the test names, to show what containment refuses. Test support; "
        "never in the production host."),
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
        "None exists. Stdout/stderr are not Elpis-owned storage; the durable authority is not a history."),
)

# Every public Runtime operation, exactly once. Ordinary autonomous operations may reach only
# FIXED_CAPACITY_AUTONOMOUS writers; operator operations require explicit authority from their caller.
AUTONOMOUS_OPERATIONS = frozenset({
    "open", "close", "fault", "run_ingress", "admit_retrieval", "admit_context",
    "anchor_cognition", "evolution_authority",
    "release",             # gives a bound K1 state back (its managed lease); writes nothing
    "run_query",           # read-only: writes nothing at all
    "run_learn",           # explicit LearnAuthority; a volatile commit: writes nothing at all
    "run_observe",         # explicit ObservationAuthority; a volatile observe-only commit: writes nothing at all
    "run_numeric_query",   # read-only: writes nothing at all
    "run_numeric_learn",   # explicit LearnAuthority; a volatile commit: writes nothing at all
    "run_turn",            # LEGACY learned turn: a LEARN (explicit LearnAuthority); writes nothing at all
    "publish_recoverable",  # explicit recoverable publication: the fixed checkpoint and authority slots only
    "publish_identity",    # explicit identity-only publication: the fixed authority slots only
    "durability",          # read-only: the durability lane's state
    "live_identity",       # read-only: the live state's identity
    "recover_k1",          # read-only: what restart may do with the lineage
})
OPERATOR_OPERATIONS = frozenset({
    "publish_canonical",   # one-use promotion capability against an operator approval digest
    "evolve",              # an explicit assertion under the pinned evolution policy, reserved before execution
    "discard_k1_candidate",   # explicit reconciliation of exactly one named checkpoint candidate
    "adopt_k1_candidate",     # explicit reconciliation: the authority's authorized -> the named candidate
    "import_composite",       # explicit lineage format change (K1 -> ECS composite): fixed checkpoint, then authority
})
# The fixed-capacity writers each composer's ordinary path reaches, by composer. Every FIXED_CAPACITY_AUTONOMOUS
# writer belongs to at least one, and nothing else does (tests/boundary/test_autonomous_persistence.py).
# Both composers' durable authority is the ECS_C authority (RuntimeCore ABI v7 has no other; standalone continuity's
# slots retired into it).
# * AUTONOMOUS_WRITERS: the Python Runtime (elpis.runtime.composition, the offline reference interface) and the native
#   libraries it loads; tests/integration/test_autonomous_no_growth.py proves this set under a write trap.
# * NATIVE_HOST_AUTONOMOUS_WRITERS: the native host (docs/NATIVE_HOST.md), the production entry point. It builds no FMS
#   cold store; tests/integration/test_native_host.py proves its home a fixed file set under containment.
AUTONOMOUS_WRITERS = frozenset({"ecs_c_authority_slots", "fms_posix_cold_store", "k1_checkpoint_slots"})
NATIVE_HOST_AUTONOMOUS_WRITERS = frozenset({"ecs_c_authority_slots", "k1_checkpoint_slots"})


def writer(writer_id: str) -> PersistentWriter:
    for w in WRITERS:
        if w.id == writer_id:
            return w
    raise KeyError(writer_id)
