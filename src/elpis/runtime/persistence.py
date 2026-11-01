"""The storage-write census of the runtime: Elpis SHALL NOT autonomously mutate persistent storage.

docs/ELPIS_MISSION.md, Amendment 3 and docs/VOLATILE_RUNTIME_CENSUS.md state the law:

* COLD is immutable and persistent, read/run only;
* WARM is bounded volatile RAM, and HOT is bounded active RAM or VRAM;
* all learning, adaptation, recurrence and mutable state stay volatile, and no runtime state is ever written back.

A bounded persistent writer is still a persistent writer. This module is the census of every writer that is, or ever
was, reachable from a composer:

* the native host ``elpis`` (the production entry point, docs/NATIVE_HOST.md);
* the Python ``elpis.runtime`` (the offline reference interface) and the native libraries it loads.

It also lists the persistent writers that live outside them. Each writer has exactly one disposition:

* ``RETIRED``: no persistent path exists. The entry points listed for it are absent from the code, or its surface
  refuses. The boundary tests check absence; a test merely never calling them proves nothing.
* ``REPLACED_WITH_VOLATILE``: the state it persisted lives in volatile memory instead (``replacement``), and its
  persistent entry points are absent.
* ``READ_ONLY_STATIC_RESOURCE``: persistent and immutable, opened read-only, never written by Elpis (COLD).
* ``OFFLINE_ARTIFACT_PRODUCER``: produces persistent artifacts outside the runtime. No composer imports or links it,
  and every run is a separately and explicitly human-authorized offline act.
* ``DEVELOPMENT_ONLY``: tests, qualification instruments, laboratories and packaging. Never part of the runtime.

Every active-runtime writer ends as one of the first three; no ``NONCONFORMING_RUNTIME`` remains.

Bindings to the code:

* tests/boundary/test_autonomous_persistence.py and tests/boundary/test_volatile_runtime.py bind this table to the
  code: the live entry points resolve, the retired ones are absent, no composer reaches a producer, and no binary the
  runtime loads imports a write-class C library function;
* tests/integration/test_autonomous_no_growth.py proves the table on real native libraries under a write trap;
* research/volatile_runtime_v1 proves it on the composed runtime under strace and read-only confinement.

:data:`CLOSURE` records the directive's fourteen closure conditions (section 10) with their evidence.
:data:`VOLATILE_MIGRATION_COMPLETE` holds only when every condition holds.

This module is declarative. It holds no state, opens nothing and grants nothing.
"""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class Disposition(str, Enum):
    RETIRED = "RETIRED"
    REPLACED_WITH_VOLATILE = "REPLACED_WITH_VOLATILE"
    READ_ONLY_STATIC_RESOURCE = "READ_ONLY_STATIC_RESOURCE"
    OFFLINE_ARTIFACT_PRODUCER = "OFFLINE_ARTIFACT_PRODUCER"
    DEVELOPMENT_ONLY = "DEVELOPMENT_ONLY"


# The dispositions an active-runtime writer may end with (directive, section 8).
RUNTIME_DISPOSITIONS = frozenset({Disposition.RETIRED, Disposition.REPLACED_WITH_VOLATILE,
                                  Disposition.READ_ONLY_STATIC_RESOURCE})


@dataclass(frozen=True)
class PersistentWriter:
    id: str
    disposition: Disposition
    # Entry points. For RETIRED and REPLACED_WITH_VOLATILE: the persistent entry points that existed, each of which must
    # now be absent ("module:qualname" not importable or not an attribute; "repo/path:symbol" not in that file, or the
    # file gone). For the other dispositions: the live entry points, each of which must resolve.
    python: tuple[str, ...]
    native: tuple[str, ...]
    runtime: bool                    # reachable from a composer before the migration (an active-runtime writer)
    replacement: str                 # what holds the state now / the read-only surface / the offline authority
    evidence: tuple[str, ...]        # tests ("path::name" or "ctest:label.name") that hold the disposition
    bound: str


R, V, S, A, D = (Disposition.RETIRED, Disposition.REPLACED_WITH_VOLATILE, Disposition.READ_ONLY_STATIC_RESOURCE,
                 Disposition.OFFLINE_ARTIFACT_PRODUCER, Disposition.DEVELOPMENT_ONLY)
_TOPOLOGY = "native/structure/semantic/src/topology/"
_GRID81 = "native/structure/semantic/src/grid81/"
_HOST = "tests/integration/test_native_host.py::"
_GROWTH = "tests/integration/test_autonomous_no_growth.py::"

WRITERS: tuple[PersistentWriter, ...] = (
    # ---- the persistent runtime, retired or replaced (phases V3, V4) ----------------------------------------------
    PersistentWriter(
        "ecs_c_authority_slots", V, (),
        ("native/ECS_C/authority/src/store.rs:pub fn transition", "native/runtime/src/durable.rs:pub struct Stores"),
        True, "the volatile ECS_C session (native/ECS_C/session): RuntimeCore ABI v8 holds one per open",
        ("tests/boundary/test_ecs_c_authority.py::test_the_durable_authority_has_no_source_target_or_library",
         "ctest:runtime.test_runtime_session", _HOST + "test_a_session_works_in_memory_and_nothing_survives_it"),
        "The durable authority's two record slots, two witnesses and deployment key are gone with its crate. The "
        "same transition law runs over RAM-only session state, and a restart is a new session."),
    PersistentWriter(
        "ecs_c_authority_provisioning", R, ("elpis.runtime.authority:provision_authority",),
        ("native/ECS_C/authority/src/store.rs:pub fn provision",),
        True, "",
        ("tests/boundary/test_ecs_c_authority.py::test_nothing_builds_links_includes_or_loads_the_durable_authority",),
        "No directory, key file or slot is provisioned: a session needs none."),
    PersistentWriter(
        "k1_checkpoint_slots", R,
        ("elpis.runtime.composition:Runtime.publish_recoverable", "elpis.runtime.composition:Runtime.recover_k1",
         "elpis.runtime.composition:Runtime.discard_k1_candidate",
         "elpis.runtime.composition:Runtime.adopt_k1_candidate"),
        ("native/runtime/src/checkpoint.rs:pub fn write", "native/runtime/src/checkpoint.rs:pub fn retract"),
        True, "",
        ("tests/boundary/test_volatile_runtime.py::test_the_runtime_core_abi_has_no_persistent_function",
         _HOST + "test_a_killed_process_leaves_nothing"),
        "K1 Recovery R0's two checkpoint slots: no store, attach, recover, adopt or discard exists. A K1 state is "
        "never persisted, so nothing is resumed."),
    PersistentWriter(
        "k1_checkpoint_provisioning", R, ("elpis.runtime.recovery:provision_k1_checkpoint",),
        ("native/runtime/src/checkpoint.rs:pub fn provision",),
        True, "",
        ("tests/boundary/test_volatile_runtime.py::test_the_runtime_core_abi_has_no_persistent_function",),
        "Retired with the checkpoint store."),
    PersistentWriter(
        "runtime_durability_lane", R,
        ("elpis.runtime.composition:Runtime.publish_identity", "elpis.runtime.composition:Runtime.durability",
         "elpis.runtime.composition:Runtime.import_composite"),
        ("native/runtime/src/durable.rs:fn lane_worker",),
        True, "",
        ("tests/boundary/test_volatile_runtime.py::test_the_runtime_core_abi_has_no_persistent_function",
         "tests/boundary/test_mission.py::test_runtime_composes_no_model_operation"),
        "The durable publications (recoverable and identity-only), their background lane, its waits and status, "
        "and the durable lineage format change are absent. RuntimeCore ABI v8 declares none of them."),
    PersistentWriter(
        "native_host_deployment", V, (),
        ("native/runtime/host/src/host_deploy.c:host_init", "native/runtime/host/src/host_deploy.c:host_create_lineage"),
        True, "a RAM-only installation (host_install.c): pinned libraries sealed read-only into memfds, the session "
              "in memory",
        (_HOST + "test_every_retired_persistent_command_is_refused",
         _HOST + "test_a_live_host_maps_no_interpreter_writes_nowhere_and_its_descriptors_never_grow"),
        "No home, init, create, publish or reconcile: those commands are refused OPERATION_RETIRED. The host writes "
        "nothing, and containment grants no write."),
    PersistentWriter(
        "python_reference_deployment", V,
        ("elpis.runtime.deployment:init_deployment", "elpis.runtime.deployment:Deployment.create_lineage"), (),
        True, "elpis.runtime.deployment.Installation and Session: a read-only catalog of pinned libraries and a "
              "lineage in memory",
        ("tests/integration/test_cli.py::test_every_retired_persistent_command_is_refused",
         "tests/integration/test_cli.py::test_a_session_works_in_memory_and_nothing_survives_it"),
        "The reference interface initializes, provisions and publishes nothing."),
    PersistentWriter(
        "runtime_canonical_publication", R, ("elpis.runtime.composition:Runtime.publish_canonical",), (),
        True, "",
        ("tests/integration/test_canonical_writer.py::test_the_runtime_exposes_no_canonical_publication",),
        "The Runtime no longer reaches the canonical publisher. The publisher itself is an offline producer "
        "(canonical_grid81_publication)."),
    PersistentWriter(
        "evolution_reservation", V, (), ("native/runtime/src/durable.rs:fn evolution_transition",),
        True, "the volatile session's evolution authority (reserve, finalize, abandon), gone at close",
        ("tests/integration/test_evolution.py::test_the_evolution_authority_lives_and_ends_with_the_session",),
        "Runtime.evolve's reservation and finalization are typed transitions of the session. Nothing is reconciled "
        "after a restart."),
    PersistentWriter(
        "operator_authority_events", V, (), ("native/runtime/src/durable.rs:fn authority_transition",),
        True, "typed events of the volatile session (elpis_runtime_authority_transition over the session)",
        (_HOST + "test_operator_events_are_typed_events_of_the_volatile_session",),
        "Founding and activating sources and operator lifecycle events happen in RAM. The persistence-only kinds "
        "(legacy import, reconciliations) are refused RUNTIME_RETIRED."),
    PersistentWriter(
        "fms_posix_cold_store", R, ("elpis.substrate.residency:Context.cold_root",),
        ("native/substrate/src/fms_pal_posix.c:p_cold_put", "native/substrate/src/fms_pal_posix.c:fms_pal_posix_create(",
         "native/substrate/src/fms_pal_posix.c:elpis_fms_create_posix"),
        True, "",
        ("ctest:substrate.test_fms_volatile_cold", "tests/ECS/test_mutable_fms.py::test_the_retired_cold_root_is_not_a_parameter",
         "tests/ECS/test_mutable_fms.py::test_capacity_bound_and_volatile_lifecycle"),
        "FMS ABI v3 has no COLD residency, no storage domain and no WARM -> COLD operation. A configuration that "
        "budgets COLD or storage is refused, as is a COLD registration or acquisition. The writable constructors "
        "are not exported, and a full WARM refuses a newcomer instead of spilling it."),
    PersistentWriter(
        "hgram_fixed_capacity_store", R, (),
        ("native/structure/hacf/src/kernel/hgram_init.c:main",
         "native/structure/hacf/src/kernel/hgram_store.c:elpis_hgram_init_once",
         "native/structure/hacf/src/kernel/hgram_store.c:elpis_hgram_assoc_store_preapproved"),
        True, "",
        ("ctest:structure.test_hgram_assoc", "ctest:structure.test_hgram_store"),
        "No H-gram is created, preallocated, initialized or written by Elpis: there is no persistent H-gram "
        "learning. A retained H-gram is a read-only static resource (hgram_retained_store)."),
    PersistentWriter(
        "persistent_hacf_corpus", R, (), ("native/structure/hacf/src/retrieval/corpus/corpus.cpp:elpis_corpus_open(",),
        True, "",
        ("tests/boundary/test_autonomous_persistence.py::test_hacf_has_no_persistent_corpus_or_ingress_path_and_the_epoch_fms_is_ram_only",),
        "The corpus is SQLite :memory: only (elpis_corpus_open_ephemeral; temp_store and journal_mode MEMORY), "
        "and persistent ingress refuses. A retrieval epoch is volatile."),
    PersistentWriter(
        "persistent_log_sink", R, (), (),
        True, "",
        ("tests/boundary/test_autonomous_persistence.py::test_importing_the_runtime_installs_no_persistent_log_sink",),
        "None exists: importing the runtime installs no handler, and the native host opens no socket (seccomp)."),
    # ---- read-only static resources (COLD) ---------------------------------------------------------------------
    PersistentWriter(
        "pinned_native_libraries", S, ("elpis.substrate.native_admission:admit_libraries",),
        ("native/runtime/host/src/host_admit.c:host_seal",),
        True, "the admitted libraries, pinned by SHA-256 and loaded from sealed read-only copies; the host holds no "
              "writable descriptor of any",
        (_HOST + "test_the_catalog_pins_the_host_and_every_library_read_only",
         "tests/integration/test_native_admission.py::test_a_modified_library_is_refused"),
        "Read and hashed, never written."),
    PersistentWriter(
        "fms_cold_backing", S, ("elpis.substrate.file_assets:FMSFileAssets.register",),
        ("native/substrate/src/fms_file_service.c:elpis_file_service_admit",),
        True, "immutable file assets read through an O_RDONLY descriptor (pread, page digests verified) into WARM; "
              "a writable descriptor is refused, and no replica is ever written",
        ("ctest:substrate.test_fms_volatile_cold", "ctest:substrate.test_fms_volatile_cold_rofs",
         "ctest:substrate.test_fms_file_service"),
        "COLD is the frozen backing: materialized into WARM on demand, dematerialized without a write."),
    PersistentWriter(
        "hgram_retained_store", S, (),
        ("native/structure/hacf/src/kernel/hgram_store.c:elpis_hgram_probe",
         "native/structure/hacf/src/kernel/hgram_store.c:elpis_hgram_assoc_get"),
        True, "probe and association reads, O_RDONLY under a shared lock",
        ("ctest:structure.test_hgram_assoc",),
        "Read only: every probe and read issues zero storage mutations (link-time trap)."),
    # ---- offline artifact producers: never reachable from a composer ---------------------------------------------
    PersistentWriter(
        "canonical_grid81_publication", A, ("elpis.pipeline.canonical.publisher:publish_candidate",), (),
        False, "a one-use promotion capability issued against an explicit operator approval digest",
        ("tests/integration/test_canonical_writer.py::test_the_runtime_exposes_no_canonical_publication",),
        "Each approved publication adds one append-only generation file and one ledger row."),
    PersistentWriter(
        "canonical_candidate_construction", A, ("elpis.pipeline.canonical.candidate:construct_candidate",), (),
        False, "an explicit promotion at a caller-chosen root", (), "One candidate tree per explicit promotion."),
    PersistentWriter(
        "publication_ledger_v1", A,
        ("elpis.pipeline.application.durable_ledger:DurableApplicationLedger.append",), (),
        False, "an operator-approved publication or application on a caller-supplied ledger path", (),
        "One row per explicit act."),
    PersistentWriter(
        "application_ledger_v2", A,
        ("elpis.pipeline.application.durable_ledger_v2:DurableApplicationLedgerV2.append",
         "elpis.pipeline.application.application:apply_artifact"), (),
        False, "an explicitly applied artifact against caller-owned shadow state", (), "One row per explicit act."),
    PersistentWriter(
        "evolution_child_materialization", A,
        ("elpis.evolution.materialize:atomic_materialize",),
        (), False, "an operator-approved promotion grant of the pinned evolution policy, at a destination that must "
                   "not exist (EvolutionPolicyAuthority.materialize imports the writer only when called)", (),
        "One child workspace per grant. elpis.evolution.materialize is its own module, so the promotion law the "
        "runtime's evolution gate imports carries no writer."),
    PersistentWriter(
        "semantic_snapshot_publication", A, (),
        ("native/structure/semantic/src/graph/segment_writer.c:semantic_segment_write",
         "native/structure/semantic/src/graph/snapshot_publish.c:semantic_snapshot_write",
         "native/structure/semantic/src/graph/snapshot_publish.c:semantic_snapshot_publish_cas",
         "native/structure/semantic/src/graph/active_head_b2b.c:semantic_b2b_head_cas",
         "native/structure/semantic/src/evidence/admission_commit_b2c.c:semantic_b2c_publish_one"),
        False, "an explicit caller with explicit admission and snapshot authority",
        ("tests/boundary/test_volatile_runtime.py::test_no_binary_the_runtime_loads_imports_a_write_function",),
        "Immutable segments and manifests and a CAS active head. snapshot_publish.c is its own translation unit, so "
        "the ingress bridge the runtime loads links the manifest mechanics without the writers."),
    PersistentWriter(
        "semantic_context_artifacts", A, (),
        ("native/structure/semantic/src/context/context_writer.c:elpis_write_requirement_set",
         "native/structure/semantic/src/context/context_writer.c:elpis_write_deficit_policy",
         "native/structure/semantic/src/context/context_writer.c:elpis_write_deficit_report",
         "native/structure/semantic/src/context/context_writer.c:elpis_write_retrieval_requirement",
         "native/structure/semantic/src/context/context_writer.c:elpis_write_retrieval_bundle"),
        False, "an explicit call at a caller-supplied path", (), "One named file per explicit call."),
    PersistentWriter(
        "semantic_embedding_files", A, (),
        ("native/structure/semantic/src/embedding/embedding_writer.c:elpis_embedding_write_profile",
         "native/structure/semantic/src/embedding/embedding_writer.c:elpis_embedding_write_vector",
         "native/structure/semantic/src/embedding/embedding_writer.c:elpis_embedding_write_collection"),
        False, "an explicit call at a caller-supplied path", (), "One named file per explicit call."),
    PersistentWriter(
        "semantic_evidence_records", A, (),
        ("native/structure/semantic/src/evidence/evidence_writer.c:elpis_evidence_write_record",),
        False, "an explicit call at a caller-supplied path", (), "One evidence record file per explicit call."),
    PersistentWriter(
        "semantic_view_directory", A, (),
        ("native/structure/semantic/src/view/view_writer.c:elpis_view_write_all",),
        False, "an explicit call into a caller-supplied directory", (), "The fixed file set of one bounded view."),
    PersistentWriter(
        "semantic_topology_artifacts", A, (),
        tuple(_TOPOLOGY + f for f in (
            "topology_address.c:elpis_write_topology_addresses", "topology_anchor.c:elpis_write_topology_anchors",
            "topology_bridge.c:elpis_write_topology_bridges", "topology_conflict.c:elpis_write_topology_conflicts",
            "topology_constellation.c:elpis_write_topology_constellations",
            "topology_constraint.c:elpis_write_topology_constraints", "topology_graph.c:elpis_write_topology_graph",
            "topology_metric.c:elpis_write_topology_metric_hints", "topology_persist.c:elpis_write_topology_IR",
            "topology_persist.c:elpis_write_compile_receipt", "topology_persist.c:elpis_write_topology_handoff",
            "topology_policy.c:elpis_write_topology_policy", "topology_registry.c:elpis_write_topology_registry",
            "topology_roles.c:elpis_write_topology_roles")),
        False, "offline topology compilation, an explicit call at a caller-supplied path", (),
        "One named file per explicit call."),
    PersistentWriter(
        "semantic_grid81_artifacts", A, (),
        tuple(_GRID81 + f for f in (
            "grid81_capsule.c:elpis_write_grid81_capsule_manifest", "grid81_cell.c:elpis_write_grid81_masks",
            "grid81_codebook.c:elpis_write_grid81_codebook",
            "grid81_compile_receipt.c:elpis_write_grid81_compile_receipt",
            "grid81_constraint_projection.c:elpis_write_grid81_constraint_projections",
            "grid81_handoff.c:elpis_write_grid81_handoff", "grid81_packet.c:elpis_write_grid81_structural_packet",
            "grid81_policy.c:elpis_write_grid81_policy")),
        False, "offline Grid81 compilation, an explicit call at a caller-supplied path", (),
        "One named file per explicit call."),
    # ---- development only ----------------------------------------------------------------------------------------
    PersistentWriter(
        "dsv4_synthetic_fixture_generator", D, ("elpis.inference.drivers.dsv4.fixtures:make_fixture",), (),
        False, "", (), "Synthetic fixtures written into a caller-supplied provider root."),
    PersistentWriter(
        "native_host_testing_write_probe", D, (),
        ("native/runtime/host/src/elpis_host.c:op_testing_write_probe",),
        False, "the testing host only (ELPIS_HOST_TESTING); never in the production host, never installed",
        (_HOST + "test_the_contained_hosts_deliberate_write_attempts_are_all_refused",),
        "The containment negative control: each attempt is refused under containment."),
    PersistentWriter(
        "research_laboratories", D, (), (), False, "", (), "Write-once laboratory evidence under research/."),
    PersistentWriter(
        "acceptance_harness", D, ("elpis.runtime.acceptance:Acceptance",), (),
        False, "", (), "A qualification instrument: copies and one JSON report under a caller-supplied work "
                       "directory. Never imported by a composer."),
    PersistentWriter(
        "developer_release_bundle", D, ("elpis.runtime.release:build_bundle",), (),
        False, "", (), "Packaging of a source commit and its libraries; never imported by a composer."),
)


def writer(writer_id: str) -> PersistentWriter:
    for w in WRITERS:
        if w.id == writer_id:
            return w
    raise KeyError(writer_id)


# Every public Runtime operation, exactly once. None of them writes persistent storage: RuntimeCore ABI v8 holds one
# volatile session per open. The operator operations need explicit authority from their caller.
AUTONOMOUS_OPERATIONS = frozenset({
    "open", "close", "fault", "run_ingress", "admit_retrieval", "admit_context",
    "anchor_cognition", "evolution_authority", "release", "run_query", "run_learn", "run_observe",
    "run_numeric_query", "run_numeric_learn", "run_turn", "live_identity",
})
OPERATOR_OPERATIONS = frozenset({
    "evolve",   # an explicit assertion under the pinned evolution policy, reserved in the session before execution
})

# Writers each composer's ordinary path reaches: none.
AUTONOMOUS_WRITERS: frozenset[str] = frozenset()
NATIVE_HOST_AUTONOMOUS_WRITERS: frozenset[str] = frozenset()

# RuntimeCore C ABI functions that write persistent storage, open it for writing, or resume from it: none (ABI v8).
# The ABI v7 ones, all absent from native/runtime/include/elpis/runtime.h and from libelpis_runtime.so:
RETIRED_RUNTIME_CORE_ABI = frozenset({
    "elpis_runtime_create_ecs_c", "elpis_runtime_import_composite", "elpis_runtime_recoverable_request",
    "elpis_runtime_identity_request", "elpis_runtime_durable_wait", "elpis_runtime_durable_status",
    "elpis_runtime_checkpoint_provision", "elpis_runtime_checkpoint_attach", "elpis_runtime_checkpoint_recover",
    "elpis_runtime_checkpoint_discard", "elpis_runtime_checkpoint_adopt", "elpis_runtime_evolution_reconcile",
})
RUNTIME_CORE_PERSISTENT_ABI: frozenset[str] = frozenset()
# Public Runtime operations that write persistent state or resume from it: none.
NONCONFORMING_OPERATIONS: frozenset[str] = frozenset()

# What holds the state the persistent writers held.
VOLATILE_REPLACEMENTS = {w.id: w.replacement for w in WRITERS if w.disposition is Disposition.REPLACED_WITH_VOLATILE}


# ---------------------------------------------------------------------------------------------------------------
# The directive's closure conditions (section 10), each with the evidence that holds it.
# ---------------------------------------------------------------------------------------------------------------

@dataclass(frozen=True)
class Condition:
    number: int
    statement: str
    holds: bool
    evidence: tuple[str, ...]


# The composed-runtime qualification (condition 11) is the recorded run of research/volatile_runtime_v1; its verdict is
# read back by tests/boundary/test_volatile_runtime.py.
COMPOSED_QUALIFICATION = "research/volatile_runtime_v1/results/zero_write_v1.json"

CLOSURE: tuple[Condition, ...] = (
    Condition(1, "RuntimeCore no longer depends on durable ECS_C authority.", True, (
        "tests/boundary/test_ecs_split.py::test_dependency_directions_keep_runtime_as_the_only_composer",
        "tests/boundary/test_ecs_c_authority.py::test_nothing_builds_links_includes_or_loads_the_durable_authority")),
    Condition(2, "The native host starts without a writable deployment home.", True, (
        _HOST + "test_a_session_works_in_memory_and_nothing_survives_it",
        _HOST + "test_the_contained_hosts_deliberate_write_attempts_are_all_refused")),
    Condition(3, "The Python runtime has no reachable persistent-state writer.", True, (
        "tests/boundary/test_volatile_runtime.py::test_every_python_source_that_writes_is_a_registered_writer",
        "tests/boundary/test_autonomous_persistence.py::test_no_composer_imports_an_offline_or_development_writer",
        _GROWTH + "test_ordinary_autonomous_path_writes_nothing_at_all")),
    Condition(4, "The checkpoint and durability lanes are retired.", True, (
        "tests/boundary/test_volatile_runtime.py::test_the_runtime_core_abi_has_no_persistent_function",)),
    Condition(5, "FMS COLD is immutable, read-only backing.", True, (
        "ctest:substrate.test_fms_volatile_cold", "ctest:substrate.test_fms_volatile_cold_rofs")),
    Condition(6, "WARM and HOT are volatile.", True, ("ctest:substrate.test_fms", "ctest:substrate.test_fms_volatile_cold")),
    Condition(7, "Persistent FMS writeback is retired.", True, (
        "ctest:substrate.test_fms_volatile_cold",
        "tests/ECS/test_mutable_fms.py::test_the_retired_cold_root_is_not_a_parameter")),
    Condition(8, "Runtime H-gram mutation is retired.", True, ("ctest:structure.test_hgram_assoc",)),
    Condition(9, "Runtime canonical publication is retired.", True, (
        "tests/integration/test_canonical_writer.py::test_the_runtime_exposes_no_canonical_publication",)),
    Condition(10, "All remaining runtime-connected storage writers have been eliminated or made read-only.", True, (
        "tests/boundary/test_volatile_runtime.py::test_every_writer_has_exactly_one_final_disposition",
        "tests/boundary/test_volatile_runtime.py::test_no_binary_the_runtime_loads_imports_a_write_function")),
    Condition(11, "The composed runtime passes independent zero-write verification.", True, (COMPOSED_QUALIFICATION,)),
    Condition(12, "Environmental prerequisites are enforced or fail closed.", True, (
        _HOST + "test_containment_refuses_what_it_cannot_contain",
        _HOST + "test_the_swap_policy_locks_the_process_memory_or_refuses",
        "tests/integration/test_environment_gate.py::test_a_runtime_opens_only_once_the_process_cannot_dump_or_write_bytecode")),
    Condition(13, "Bounded-memory and atomic-refusal invariants remain intact.", True, (
        _HOST + "test_entity_capacity_is_refused_and_reclamation_frees_a_row",
        _HOST + "test_refusals_are_reported_with_stable_codes_and_change_nothing", COMPOSED_QUALIFICATION)),
    Condition(14, "Architectural documents and ABI inventories match the implementation.", True, (
        "tests/boundary/test_abi_inventory.py::test_every_library_exports_exactly_its_recorded_symbols",
        "tests/boundary/test_volatile_runtime.py::test_the_documents_state_the_migration_as_it_is")),
)

# Remaining dependencies that keep the migration incomplete (empty when every condition holds).
REMAINING_DEPENDENCIES: tuple[str, ...] = tuple(f"condition {c.number}: {c.statement}" for c in CLOSURE if not c.holds)

VOLATILE_MIGRATION_COMPLETE = (
    all(c.holds for c in CLOSURE)
    and all(w.disposition in RUNTIME_DISPOSITIONS for w in WRITERS if w.runtime)
    and not (AUTONOMOUS_WRITERS or NATIVE_HOST_AUTONOMOUS_WRITERS or RUNTIME_CORE_PERSISTENT_ABI
             or NONCONFORMING_OPERATIONS))
