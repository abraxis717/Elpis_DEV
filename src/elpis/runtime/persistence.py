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

Superseded in direction (docs/ELPIS_MISSION.md, Amendment 3; docs/VOLATILE_RUNTIME_CENSUS.md). The volatile-native
directive retires the persistent runtime: no operational state may be written to persistent storage under any
circumstance, bounded or not, autonomous or operator-requested. The classes above describe the code as it still is;
:data:`VOLATILE_CENSUS` gives every writer its standing under the directive, and :data:`RUNTIME_CORE_PERSISTENT_ABI`
and :data:`NONCONFORMING_OPERATIONS` name the still-reachable durable paths. While any of them is non-empty
(:data:`VOLATILE_MIGRATION_COMPLETE` is False) no document may describe the runtime as diskless
(``tests/boundary/test_volatile_runtime.py``).

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
        "python_reference_deployment", O,
        ("elpis.runtime.deployment:init_deployment", "elpis.runtime.deployment:Deployment.create_lineage"), (),
        "The offline reference interface's deployment (elpis.runtime.deployment, docs/OPERATING_ELPIS.md): init "
        "creates a home holding elpis.json, native-catalog.json and the pinned libraries (write-once files, synced, "
        "never replaced); create writes lineage/lineage.json once and provisions the ECS_C authority and the "
        "checkpoint store beneath it. Each is one explicit operator command, refused when its files exist. Not "
        "reached by a Runtime operation (recorded by the volatile census of the V0 phase; it was unregistered)."),
    PersistentWriter(
        "acceptance_harness", R, ("elpis.runtime.acceptance:Acceptance",), (),
        "System acceptance of an installed deployment (python -m elpis.runtime.acceptance): fresh deployments, "
        "copies and one JSON report under a caller-supplied work directory that must not exist or be empty. A "
        "qualification instrument run by a developer; never imported by a composer (recorded by the volatile "
        "census of the V0 phase; it was unregistered)."),
    PersistentWriter(
        "developer_release_bundle", R, ("elpis.runtime.release:build_bundle",), (),
        "A developer installation bundle (python -m elpis.runtime.release): one new directory per explicit "
        "packaging command (source archive, libraries, manifest, sums). Packaging, never imported by a composer "
        "(recorded by the volatile census of the V0 phase; it was unregistered)."),
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


# ---------------------------------------------------------------------------------------------------------------
# The volatile-native census (V0; docs/VOLATILE_RUNTIME_CENSUS.md, docs/ELPIS_MISSION.md, Amendment 3).
# ---------------------------------------------------------------------------------------------------------------

class VolatileDisposition(str, Enum):
    """A writer's standing under the volatile-native directive.

    * ``NONCONFORMING_RUNTIME``: writes operational state (records, slots, checkpoints, journals, learned
      associations, cold-tier objects, deployment state) to persistent storage. To be retired from the active
      architecture; kept only as clearly labelled archival code once retired, never behind a flag.
    * ``OFFLINE_ARTIFACT_PRODUCER``: produces persistent artifacts outside the runtime; never reachable from a
      composer, and every run is a separately and explicitly authorized offline act (directive V7).
    * ``DEVELOPMENT_ONLY``: tests, qualification instruments, laboratories and packaging; never part of the runtime.
    * ``RETIRED``: no persistent path exists (the writer is absent or refuses).
    """
    NONCONFORMING_RUNTIME = "NONCONFORMING_RUNTIME"
    OFFLINE_ARTIFACT_PRODUCER = "OFFLINE_ARTIFACT_PRODUCER"
    DEVELOPMENT_ONLY = "DEVELOPMENT_ONLY"
    RETIRED = "RETIRED"


@dataclass(frozen=True)
class VolatileEntry:
    disposition: VolatileDisposition
    phase: str      # the migration phase that retires or isolates it (V3, V4), or "-" when nothing remains
    action: str


_N, _A, _D, _X = (VolatileDisposition.NONCONFORMING_RUNTIME, VolatileDisposition.OFFLINE_ARTIFACT_PRODUCER,
                  VolatileDisposition.DEVELOPMENT_ONLY, VolatileDisposition.RETIRED)

VOLATILE_CENSUS: dict[str, VolatileEntry] = {
    "k1_checkpoint_slots": VolatileEntry(_N, "V3",
        "Retire with RuntimeCore ABI v8: no checkpoint store, attach, recover, adopt or discard; restart is a new "
        "session (nothing is resumed)."),
    "ecs_c_authority_slots": VolatileEntry(_N, "V3",
        "Replaced by the volatile session (native/ECS_C/session, ECSC_VOLATILE_SESSION_R0; built and tested, not yet "
        "composed). RuntimeCore ABI v8 holds a session instead of the durable store; the durable crate becomes "
        "archival."),
    "fms_posix_cold_store": VolatileEntry(_N, "V4",
        "COLD must not write modified state to storage: every composer builds FMS RAM-only "
        "(fms_pal_posix_create_ram_only, as the retrieval epoch already does) and the posix cold store leaves the "
        "Python composer's reach."),
    "canonical_grid81_publication": VolatileEntry(_N, "V4",
        "Reachable from the runtime through Runtime.publish_canonical: remove that operation. The publisher may "
        "survive only as a separately authorized offline artifact tool (V7), never from a runtime."),
    "k1_checkpoint_provisioning": VolatileEntry(_N, "V3", "Retire with the checkpoint store."),
    "ecs_c_authority_provisioning": VolatileEntry(_N, "V3",
        "Retire with the durable authority: a session needs no provisioning, directory or key file."),
    "native_host_deployment": VolatileEntry(_N, "V3",
        "RAM-only host initialization: no writable home, authority, checkpoint or lineage files; pinned libraries "
        "read from a read-only location; containment grants no write."),
    "python_reference_deployment": VolatileEntry(_N, "V3",
        "As the native host: the reference interface becomes a volatile session over read-only pinned libraries."),
    "publication_ledger_v1": VolatileEntry(_A, "V4",
        "A persistent journal: allowed only inside separately authorized offline artifact production, never from "
        "a runtime."),
    "application_ledger_v2": VolatileEntry(_A, "V4",
        "A persistent journal: allowed only inside separately authorized offline artifact production, never from "
        "a runtime."),
    "canonical_candidate_construction": VolatileEntry(_A, "V4",
        "Offline promotion tooling; leaves the runtime's reach with Runtime.publish_canonical."),
    "evolution_child_materialization": VolatileEntry(_A, "V4",
        "Offline workspace materialization (a development artifact); the durable reservation Runtime.evolve makes "
        "retires with the durable authority (V3)."),
    "hgram_fixed_capacity_store": VolatileEntry(_N, "V4",
        "Persistent H-gram association writes are prohibited outright (no persistent H-gram learning). Not reachable "
        "from either composer (no built library carries it; only the operator executable elpis_hgram_init and "
        "tests). Retire the in-place association writer; a retained H-gram is a read-only static resource."),
    "semantic_snapshot_publication": VolatileEntry(_A, "V4",
        "Offline semantic artifact production; never called from a composer (gated)."),
    "semantic_context_artifacts": VolatileEntry(_A, "V4", "Offline artifact production; never called from a composer."),
    "semantic_embedding_files": VolatileEntry(_A, "V4", "Offline artifact production; never called from a composer."),
    "semantic_evidence_records": VolatileEntry(_A, "V4", "Offline artifact production; never called from a composer."),
    "semantic_view_directory": VolatileEntry(_A, "V4", "Offline artifact production; never called from a composer."),
    "semantic_topology_artifacts": VolatileEntry(_A, "V4", "Offline compilation output; never called from a composer."),
    "semantic_grid81_artifacts": VolatileEntry(_A, "V4", "Offline compilation output; never called from a composer."),
    "dsv4_synthetic_fixture_generator": VolatileEntry(_D, "-", "Test fixtures; never reachable from the runtime."),
    "native_host_testing_write_probe": VolatileEntry(_D, "V3",
        "Testing host only; rewritten with the host's RAM-only initialization (it probes the slots that retire)."),
    "research_laboratories": VolatileEntry(_D, "-", "Write-once laboratory evidence; never imported by src/."),
    "acceptance_harness": VolatileEntry(_D, "V3",
        "Qualifies the persistent product (restart, crash dispositions): historical once V3 lands; to be replaced "
        "by a volatile session acceptance."),
    "developer_release_bundle": VolatileEntry(_D, "-", "Packaging of a source commit and its libraries; not runtime."),
    "persistent_hacf_corpus": VolatileEntry(_X, "-",
        "SQLite :memory: only (temp_store=MEMORY, journal_mode=MEMORY); persistent ingress refuses."),
    "persistent_log_sink": VolatileEntry(_X, "-", "None exists; importing the runtime installs no handler (gated)."),
}

# The RuntimeCore C ABI v7 (native/runtime/include/elpis/runtime.h) functions whose effect writes persistent storage,
# opens it for writing, or resumes from it. RuntimeCore stays nonconforming while any is declared: the target is a
# runtime with no permitted persistent-state write path (ABI v8 over the volatile session).
RUNTIME_CORE_PERSISTENT_ABI = frozenset({
    "elpis_runtime_create_ecs_c",            # opens the durable authority's directories read-write (and locks them)
    "elpis_runtime_anchor",                  # publishes the anchor to the authority (and the checkpoint slot)
    "elpis_runtime_import_composite",        # a durable lineage format change
    "elpis_runtime_recoverable_request",     # durable publication (checkpoint slot + authority record), lane worker
    "elpis_runtime_identity_request",        # durable identity-only publication (authority record), lane worker
    "elpis_runtime_durable_wait",            # the durability lane
    "elpis_runtime_durable_status",          # the durability lane
    "elpis_runtime_checkpoint_provision",    # creates the checkpoint directory and its two slot files
    "elpis_runtime_checkpoint_attach",       # opens the checkpoint slots read-write
    "elpis_runtime_checkpoint_recover",      # crash recovery: resumes a persisted state
    "elpis_runtime_checkpoint_discard",      # rewrites a checkpoint slot
    "elpis_runtime_checkpoint_adopt",        # durable reconciliation
    "elpis_runtime_evolution_reserve",       # durable authority transitions of the evolution attempt
    "elpis_runtime_evolution_finalize",
    "elpis_runtime_evolution_abandon",
    "elpis_runtime_evolution_reconcile",
    "elpis_runtime_authority_transition",    # an operator event on the durable authority
})

# Public Runtime operations that write persistent state or resume from it. The composer itself is nonconforming as
# long as Runtime construction requires RuntimeConfig.authority_dir (RuntimeCore over the durable authority).
NONCONFORMING_OPERATIONS = frozenset({
    "open",                  # attaches a configured checkpoint store (read-write)
    "anchor_cognition", "publish_recoverable", "publish_identity", "durability", "recover_k1",
    "discard_k1_candidate", "adopt_k1_candidate", "import_composite", "evolve", "publish_canonical",
})

# The volatile replacements that exist (built and tested), and whether a composer uses them yet.
VOLATILE_REPLACEMENTS = {
    "ecs_c_authority_slots": ("native/ECS_C/session", "ECSC_VOLATILE_SESSION_R0", "not composed"),
}

VOLATILE_MIGRATION_COMPLETE = not (
    any(e.disposition is VolatileDisposition.NONCONFORMING_RUNTIME for e in VOLATILE_CENSUS.values())
    or RUNTIME_CORE_PERSISTENT_ABI or NONCONFORMING_OPERATIONS)
