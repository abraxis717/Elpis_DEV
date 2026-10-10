# Structure: structural memory and representation.
#
#   hacf/      Hash-addressed cascade fabric: content-addressed cascade and graph
#              identity, deterministic chunking, bounded RAM-only corpus,
#              embedding providers, FMS-resident exact vector index,
#              context graph, deterministic hybrid retrieval, RetrievalBundle.
#   semantic/  Semantic hypergraph (typed nodes/hyperedges/incidences, segments,
#              snapshots, query overlays), embeddings, context-deficit control,
#              retrieval materialization, evidence typing and admission, bounded
#              semantic views, semantic topology IR and Grid81 structural packets.
#   bridge/    Narrow C ABI loaded by elpis.structure.retrieval.

find_package(SQLite3 REQUIRED)
if(NOT TARGET SQLite::SQLite3)
  add_library(SQLite::SQLite3 INTERFACE IMPORTED)
  set_target_properties(SQLite::SQLite3 PROPERTIES
    INTERFACE_INCLUDE_DIRECTORIES "${SQLite3_INCLUDE_DIRS}"
    INTERFACE_LINK_LIBRARIES "${SQLite3_LIBRARIES}")
endif()
find_library(ELPIS_LIBM m)

set(HACF "${CMAKE_CURRENT_SOURCE_DIR}/hacf")
set(SEM "${CMAKE_CURRENT_SOURCE_DIR}/semantic")

# ---------------------------------------------------------------------------
# HACF structural memory
# ---------------------------------------------------------------------------
elpis_add_library(structure elpis_cascade STATIC ${HACF}/src/kernel/cascade.c)
target_include_directories(elpis_cascade PUBLIC ${HACF}/include)
target_link_libraries(elpis_cascade PUBLIC elpis_hash Threads::Threads)

elpis_add_library(structure elpis_graph STATIC ${HACF}/src/graph/graph.c)
target_link_libraries(elpis_graph PUBLIC elpis_cascade elpis_hash)

elpis_add_library(structure elpis_chunking STATIC ${HACF}/src/retrieval/chunking/chunker.cpp)
target_include_directories(elpis_chunking PUBLIC ${HACF}/include)
target_link_libraries(elpis_chunking PUBLIC elpis_hash)

elpis_add_library(structure elpis_corpus STATIC ${HACF}/src/retrieval/corpus/corpus.cpp)
target_link_libraries(elpis_corpus PUBLIC elpis_chunking elpis_hash SQLite::SQLite3)

elpis_add_library(structure elpis_embedding STATIC
  ${HACF}/src/retrieval/embedding/fixture_embedder.cpp
  ${HACF}/src/retrieval/embedding/external_vector_provider.cpp)
target_include_directories(elpis_embedding PUBLIC ${HACF}/include PRIVATE ${HACF}/src/retrieval/embedding)
target_link_libraries(elpis_embedding PUBLIC elpis_hash)

# The exact CPU kernel is the qualification oracle: no fast-math and no FP
# contraction, so a fused multiply-add cannot change a reference score.
elpis_add_library(structure elpis_vector STATIC
  ${HACF}/src/retrieval/vector/vector_shard.cpp
  ${HACF}/src/retrieval/vector/cpu_exact.cpp
  ${HACF}/src/retrieval/vector/vector_index.cpp
  ${HACF}/src/retrieval/vector/vector_manifest.cpp)
target_link_libraries(elpis_vector PUBLIC elpis_embedding elpis_fms elpis_hash elpis_execution Threads::Threads)
target_compile_options(elpis_vector PRIVATE -ffp-contract=off)

elpis_add_library(structure elpis_hybrid STATIC
  ${HACF}/src/retrieval/hybrid/context_graph.cpp
  ${HACF}/src/retrieval/hybrid/hybrid_retrieval.cpp
  ${HACF}/src/retrieval/hybrid/retrieval_bundle.cpp)
target_include_directories(elpis_hybrid PRIVATE ${HACF}/src/retrieval/hybrid)
target_link_libraries(elpis_hybrid PUBLIC elpis_corpus elpis_vector elpis_cascade elpis_hash Threads::Threads)
target_compile_options(elpis_hybrid PRIVATE -ffp-contract=off)

set(VEC_LIBS elpis_vector elpis_embedding elpis_fms elpis_hash Threads::Threads)
set(HYB_LIBS elpis_hybrid elpis_vector elpis_embedding elpis_corpus elpis_fms elpis_hash Threads::Threads)
set(STATE ${CMAKE_CURRENT_BINARY_DIR})

elpis_add_test(structure test_cascade SOURCES ${HACF}/tests/kernel/test_cascade.c LIBS elpis_cascade)
elpis_add_test(structure test_graph SOURCES ${HACF}/tests/graph/test_graph.c LIBS elpis_graph)
elpis_add_test(structure test_corpus SOURCES ${HACF}/tests/retrieval/test_corpus.cpp
  LIBS elpis_corpus ARGS ${STATE}/corpus-state)
elpis_add_test(structure test_chunking_adversarial SOURCES ${HACF}/tests/retrieval/test_chunking_adversarial.cpp
  LIBS elpis_chunking)
elpis_add_test(structure test_embedding_provider SOURCES ${HACF}/tests/vector/test_embedding_provider.cpp
  LIBS elpis_embedding)
elpis_add_test(structure test_vector_shard SOURCES ${HACF}/tests/vector/test_vector_shard.cpp
  LIBS ${VEC_LIBS} INCLUDES ${HACF}/tests/vector ARGS ${STATE}/vector-shard-state)
elpis_add_test(structure test_vector_search SOURCES ${HACF}/tests/vector/test_vector_search.cpp
  LIBS ${VEC_LIBS} INCLUDES ${HACF}/tests/vector ARGS ${STATE}/vector-search-state)
elpis_add_test(structure test_vector_fms SOURCES ${HACF}/tests/vector/test_vector_fms.cpp
  LIBS ${VEC_LIBS} INCLUDES ${HACF}/tests/vector ARGS ${STATE}/vector-fms-state)
# Dominated by real cold I/O under deliberate shard thrashing; the timeout is
# finite so a liveness defect fails instead of hanging.
elpis_add_test(structure test_vector_concurrency SOURCES ${HACF}/tests/vector/test_vector_concurrency.cpp
  LIBS ${VEC_LIBS} INCLUDES ${HACF}/tests/vector ARGS ${STATE}/vector-conc-state TIMEOUT 600)
elpis_add_test(structure test_vector_crosshost_fixture SOURCES ${HACF}/tests/vector/test_vector_crosshost_fixture.cpp
  LIBS elpis_corpus ${VEC_LIBS} INCLUDES ${HACF}/tests/vector ARGS ${STATE}/vector-xhost-state
  DEFINES HACF_FIXTURE_DIR="${HACF}/tests/vector/fixture")
elpis_add_test(structure test_vector_adversarial SOURCES ${HACF}/tests/vector/test_vector_adversarial.cpp
  LIBS elpis_corpus ${VEC_LIBS} INCLUDES ${HACF}/tests/vector ARGS ${STATE}/vector-adv-state)
elpis_add_test(structure test_context_graph SOURCES ${HACF}/tests/hybrid/test_context_graph.cpp
  LIBS elpis_hybrid INCLUDES ${HACF}/tests/vector ${HACF}/tests/hybrid)
elpis_add_test(structure test_hybrid_fusion SOURCES ${HACF}/tests/hybrid/test_hybrid_fusion.cpp
  LIBS ${HYB_LIBS} INCLUDES ${HACF}/tests/vector ${HACF}/tests/hybrid ARGS ${STATE}/hybrid-fusion-state)
elpis_add_test(structure test_retrieval_bundle SOURCES ${HACF}/tests/hybrid/test_retrieval_bundle.cpp
  LIBS ${HYB_LIBS} INCLUDES ${HACF}/tests/vector ${HACF}/tests/hybrid ARGS ${STATE}/retrieval-bundle-state)
elpis_add_test(structure test_hybrid_adversarial SOURCES ${HACF}/tests/hybrid/test_hybrid_adversarial.cpp
  LIBS ${HYB_LIBS} INCLUDES ${HACF}/tests/vector ${HACF}/tests/hybrid ARGS ${STATE}/hybrid-adversarial-state)
elpis_add_test(structure test_hybrid_crosshost_fixture SOURCES ${HACF}/tests/hybrid/test_hybrid_crosshost_fixture.cpp
  LIBS ${HYB_LIBS} INCLUDES ${HACF}/tests/vector ${HACF}/tests/hybrid ARGS ${STATE}/hybrid-xhost-state
  DEFINES HACF_HYBRID_FIXTURE_DIR="${HACF}/tests/hybrid/fixture")
elpis_add_test(structure test_hybrid_concurrency SOURCES ${HACF}/tests/hybrid/test_hybrid_concurrency.cpp
  LIBS ${HYB_LIBS} INCLUDES ${HACF}/tests/vector ${HACF}/tests/hybrid ARGS ${STATE}/hybrid-concurrency-state
  TIMEOUT 180)

# ---------------------------------------------------------------------------
# Semantic structural core
# ---------------------------------------------------------------------------
elpis_add_library(structure elpis_semantic STATIC
  ${SEM}/src/graph/identity.c
  ${SEM}/src/graph/type_registry.c
  ${SEM}/src/graph/hypergraph_builder.c
  ${SEM}/src/graph/incidence_validate.c
  ${SEM}/src/graph/hacf_mapping.c
  ${SEM}/src/graph/semantic_context_projection.c
  ${SEM}/src/graph/segment_writer.c
  ${SEM}/src/graph/segment_reader.c
  ${SEM}/src/graph/snapshot_manifest.c
  ${SEM}/src/graph/active_head_b2b.c
  ${SEM}/src/graph/snapshot_view.c
  ${SEM}/src/graph/query_overlay.c
  ${SEM}/src/embedding/embedding_profile.c
  ${SEM}/src/embedding/embedding_vector.c
  ${SEM}/src/embedding/embedding_ref.c
  ${SEM}/src/embedding/embedding_collection.c
  ${SEM}/src/embedding/embedding_metric.c
  ${SEM}/src/embedding/embedding_neighborhood.c
  ${SEM}/src/embedding/embedding_writer.c
  ${SEM}/src/embedding/embedding_reader.c
  ${SEM}/src/embedding/embedding_view.c
  ${SEM}/src/embedding/embedding_coverage.c
  ${SEM}/src/context/context_requirement.c
  ${SEM}/src/context/context_requirement_set.c
  ${SEM}/src/context/context_deficit_policy.c
  ${SEM}/src/context/context_deficit.c
  ${SEM}/src/context/context_deficit_report.c
  ${SEM}/src/context/retrieval_requirement.c
  ${SEM}/src/context/retrieval_requirement_bundle.c
  ${SEM}/src/context/context_writer.c
  ${SEM}/src/context/context_reader.c
  ${SEM}/src/retrieval/retrieval_materialization.c
  ${SEM}/src/retrieval/retrieval_item_attachment.c
  ${SEM}/src/retrieval/retrieval_expansion.c
  ${SEM}/src/retrieval/retrieval_epoch.cpp
  ${SEM}/src/retrieval/hybrid_query_plan.cpp
  ${SEM}/src/retrieval/hybrid_bridge.cpp
  ${SEM}/src/retrieval/retrieval_bundle_verify.cpp
  ${SEM}/src/retrieval/retrieval_ingest.cpp
  ${SEM}/src/evidence/evidence_typer_profile.c
  ${SEM}/src/evidence/evidence_span.c
  ${SEM}/src/evidence/evidence_claim_candidate.c
  ${SEM}/src/evidence/evidence_relation_candidate.c
  ${SEM}/src/evidence/evidence_typing_bundle.c
  ${SEM}/src/evidence/evidence_candidate_validate.c
  ${SEM}/src/evidence/evidence_admission_policy.c
  ${SEM}/src/evidence/evidence_adjudicator.c
  ${SEM}/src/evidence/evidence_admission_receipt.c
  ${SEM}/src/evidence/evidence_admission_segment.c
  ${SEM}/src/evidence/evidence_admission_layer.c
  ${SEM}/src/evidence/admission_preflight_b1a.c
  ${SEM}/src/evidence/admission_source_b1b.c
  ${SEM}/src/evidence/admission_authority_b1d.c
  ${SEM}/src/evidence/admission_bundle_b1e.c
  ${SEM}/src/evidence/admission_item_b1f.c
  ${SEM}/src/evidence/admission_composition_b1g.c
  ${SEM}/src/evidence/admission_decision_b1h.c
  ${SEM}/src/evidence/admission_materialization_b1i.c
  ${SEM}/src/evidence/admission_base_b1j.c
  ${SEM}/src/evidence/admission_genesis_b1k.c
  ${SEM}/src/evidence/admission_populated_b1l.c
  ${SEM}/src/evidence/admission_populated_b1o.c
  ${SEM}/src/evidence/admission_successor_b2a.c
  ${SEM}/src/evidence/admission_commit_b2c.c
  ${SEM}/src/evidence/typed_evidence_view.c
  ${SEM}/src/evidence/evidence_writer.c
  ${SEM}/src/evidence/evidence_reader.c
  ${SEM}/src/context/context_rebind.c
  ${SEM}/src/context/context_iteration_policy.c
  ${SEM}/src/context/context_iteration_state.c
  ${SEM}/src/context/context_reevaluation.c
  ${SEM}/src/context/context_progress.c
  ${SEM}/src/view/bounded_view_policy.c
  ${SEM}/src/view/bounded_view_seed.c
  ${SEM}/src/view/bounded_view_candidate.c
  ${SEM}/src/view/bounded_view_select.c
  ${SEM}/src/view/bounded_view_closure.c
  ${SEM}/src/view/bounded_semantic_view.c
  ${SEM}/src/view/downstream_handoff.c
  ${SEM}/src/view/view_writer.c
  ${SEM}/src/view/view_reader.c
  ${SEM}/src/topology/topology_policy.c
  ${SEM}/src/topology/topology_registry.c
  ${SEM}/src/topology/topology_input.c
  ${SEM}/src/topology/topology_graph.c
  ${SEM}/src/topology/topology_anchor.c
  ${SEM}/src/topology/topology_distance.c
  ${SEM}/src/topology/topology_constellation.c
  ${SEM}/src/topology/topology_roles.c
  ${SEM}/src/topology/topology_conflict.c
  ${SEM}/src/topology/topology_bridge.c
  ${SEM}/src/topology/topology_metric.c
  ${SEM}/src/topology/topology_address.c
  ${SEM}/src/topology/topology_constraint.c
  ${SEM}/src/topology/topology_compile.c
  ${SEM}/src/topology/topology_persist.c
  ${SEM}/src/grid81/grid81_policy.c
  ${SEM}/src/grid81/grid81_codebook.c
  ${SEM}/src/grid81/grid81_input_validate.c
  ${SEM}/src/grid81/grid81_capsule.c
  ${SEM}/src/grid81/grid81_place.c
  ${SEM}/src/grid81/grid81_group_template.c
  ${SEM}/src/grid81/grid81_cell.c
  ${SEM}/src/grid81/grid81_constraint_projection.c
  ${SEM}/src/grid81/grid81_packet.c
  ${SEM}/src/grid81/grid81_compile_receipt.c
  ${SEM}/src/grid81/grid81_handoff.c
  ${SEM}/src/grid81/grid81_persist.c
  ${SEM}/src/observation/structural_observation.c)
target_include_directories(elpis_semantic PUBLIC ${SEM}/include)
target_link_libraries(elpis_semantic PUBLIC elpis_graph elpis_hash elpis_hybrid elpis_vector elpis_corpus
  elpis_embedding elpis_fms Threads::Threads)

function(semantic_test name source)
  elpis_add_test(structure ${name} SOURCES ${SEM}/tests/${source} LIBS elpis_semantic elpis_hash ${ARGN})
endfunction()

semantic_test(test_semantic_identity graph/test_identity.c)
semantic_test(test_type_registry graph/test_type_registry.c)
semantic_test(test_hypergraph_builder graph/test_hypergraph_builder.c)
semantic_test(test_hacf_mapping graph/test_hacf_mapping.c)
semantic_test(test_semantic_context_projection graph/test_semantic_context_projection.c)
semantic_test(test_segment_snapshot graph/test_segment_snapshot.c)
semantic_test(test_snapshot_publication_b0 graph/test_snapshot_publication_b0.c)
semantic_test(test_active_head_b2b graph/test_active_head_b2b.c)
if(CMAKE_SYSTEM_NAME STREQUAL "Linux")
  # Test-only syscall boundaries; never linked into elpis_semantic.
  semantic_test(test_active_head_crash_b2e graph/test_active_head_crash_b2e.c)
  target_link_options(test_active_head_crash_b2e PRIVATE "-Wl,--wrap=fsync" "-Wl,--wrap=renameat")
  set_tests_properties(structure.test_active_head_crash_b2e PROPERTIES TIMEOUT 30)
endif()
semantic_test(test_segment_payload graph/test_segment_payload.c)
set_tests_properties(structure.test_segment_payload PROPERTIES TIMEOUT 30)
semantic_test(test_snapshot_view graph/test_snapshot_view.c)
semantic_test(test_view_contract graph/test_view_contract.c)
if(CMAKE_SYSTEM_NAME STREQUAL "Linux")
  target_compile_definitions(test_view_contract PRIVATE VIEW_WRAP_ALLOC=1)
  target_link_options(test_view_contract PRIVATE "-Wl,--wrap=malloc")
endif()
semantic_test(test_embedding_view_contract embedding/test_embedding_view_contract.c)
semantic_test(test_query_overlay graph/test_query_overlay.c)
semantic_test(test_fresh_process_determinism graph/test_fresh_process_determinism.c)
semantic_test(test_embedding_profile embedding/test_embedding_profile.c)
semantic_test(test_embedding_vector embedding/test_embedding_vector.c ${ELPIS_LIBM})
semantic_test(test_embedding_ref embedding/test_embedding_ref.c)
semantic_test(test_embedding_collection embedding/test_embedding_collection.c)
semantic_test(test_embedding_metric embedding/test_embedding_metric.c ${ELPIS_LIBM})
target_compile_options(test_embedding_metric PRIVATE -fno-fast-math)
semantic_test(test_embedding_neighborhood embedding/test_embedding_neighborhood.c ${ELPIS_LIBM})
semantic_test(test_embedding_storage embedding/test_embedding_storage.c ${ELPIS_LIBM})
semantic_test(test_embedding_boundary embedding/test_embedding_boundary.c)
target_compile_definitions(test_embedding_boundary PRIVATE
  ELPIS_SEMANTIC_SOURCE_ROOT="${SEM}" ELPIS_HACF_SOURCE_ROOT="${HACF}")
semantic_test(test_context_requirement context/test_context_requirement.c)
semantic_test(test_context_requirement_set context/test_context_requirement_set.c)
semantic_test(test_context_deficit_policy context/test_context_deficit_policy.c)
semantic_test(test_context_evaluator context/test_context_evaluator.c)
semantic_test(test_context_persistence context/test_context_persistence.c)
semantic_test(test_retrieval_requirement context/test_retrieval_requirement.c)
semantic_test(test_retrieval_requirement_bundle context/test_retrieval_bundle.c)
semantic_test(test_context_boundaries context/test_context_boundaries.c)
target_compile_definitions(test_context_boundaries PRIVATE
  ELPIS_SEMANTIC_SOURCE_ROOT="${SEM}" ELPIS_HACF_SOURCE_ROOT="${HACF}")
semantic_test(test_typer_profile evidence/test_typer_profile.c)
semantic_test(test_evidence_span evidence/test_evidence_span.c)
semantic_test(test_claim_candidate evidence/test_claim_candidate.c)
semantic_test(test_relation_candidate evidence/test_relation_candidate.c)
semantic_test(test_typing_bundle evidence/test_typing_bundle.c)
semantic_test(test_admission_policy evidence/test_admission_policy.c)
semantic_test(test_adjudicator evidence/test_adjudicator.c)
semantic_test(test_admission_segment evidence/test_admission_segment.c)
semantic_test(test_admission_preflight_b1a evidence/test_admission_preflight_b1a.c)
semantic_test(test_admission_source_b1b evidence/test_admission_source_b1b.c)
semantic_test(test_admission_authority_b1d evidence/test_admission_authority_b1d.c)
semantic_test(test_admission_bundle_b1e evidence/test_admission_bundle_b1e.c)
semantic_test(test_admission_item_b1f evidence/test_admission_item_b1f.c)
semantic_test(test_admission_composition_b1g evidence/test_admission_composition_b1g.c)
semantic_test(test_admission_decision_b1h evidence/test_admission_decision_b1h.c)
semantic_test(test_admission_materialization_b1i evidence/test_admission_materialization_b1i.c)
semantic_test(test_admission_base_b1j evidence/test_admission_base_b1j.c)
semantic_test(test_admission_genesis_b1k evidence/test_admission_genesis_b1k.c)
semantic_test(test_admission_populated_b1l evidence/test_admission_populated_b1l.c)
semantic_test(test_admission_populated_chain_b1m evidence/test_admission_populated_chain_b1m.c)
semantic_test(test_admission_global_unique_b1n evidence/test_admission_global_unique_b1n.c)
semantic_test(test_admission_populated_novelty_b1o evidence/test_admission_populated_novelty_b1o.c)
if(CMAKE_SYSTEM_NAME STREQUAL "Linux")
  target_link_options(test_admission_populated_novelty_b1o PRIVATE "-Wl,--wrap=link" "-Wl,--wrap=renameat")
  set_tests_properties(structure.test_admission_populated_novelty_b1o PROPERTIES TIMEOUT 120)
endif()
semantic_test(test_typed_evidence_view evidence/test_typed_evidence_view.c)
semantic_test(test_evidence_persistence evidence/test_evidence_persistence.c)
semantic_test(test_evidence_determinism evidence/test_evidence_determinism.c)
semantic_test(test_evidence_boundaries evidence/test_evidence_boundaries.c)
semantic_test(test_context_rebind view/test_context_rebind.c)
semantic_test(test_iteration_policy view/test_iteration_policy.c)
semantic_test(test_iteration_state view/test_iteration_state.c)
semantic_test(test_context_reevaluation view/test_context_reevaluation.c)
semantic_test(test_context_progress view/test_context_progress.c)
semantic_test(test_bounded_view_policy view/test_bounded_view_policy.c)
semantic_test(test_seed_construction view/test_seed_construction.c)
semantic_test(test_candidate_enumeration view/test_candidate_enumeration.c)
semantic_test(test_bounded_selection view/test_bounded_selection.c)
semantic_test(test_conflict_closure view/test_conflict_closure.c)
semantic_test(test_provenance_closure view/test_provenance_closure.c)
semantic_test(test_metric_nonauthority view/test_metric_nonauthority.c)
semantic_test(test_bounded_view_identity view/test_bounded_view_identity.c)
semantic_test(test_downstream_handoff view/test_downstream_handoff.c)
semantic_test(test_view_persistence view/test_view_persistence.c)
semantic_test(test_view_determinism view/test_view_determinism.c)
semantic_test(test_view_boundaries view/test_view_boundaries.c)
semantic_test(test_topology_compile topology/test_topology_compile.c)
semantic_test(test_grid81_compiler grid81/test_grid81_compiler.c)
semantic_test(test_structural_observation observation/test_structural_observation.c)

# ---------------------------------------------------------------------------
# Python-facing retrieval bridge
# ---------------------------------------------------------------------------
elpis_add_library(structure elpis_retrieval_bridge SHARED ${CMAKE_CURRENT_SOURCE_DIR}/bridge/retrieval_bridge.c)
set_target_properties(elpis_retrieval_bridge PROPERTIES LINKER_LANGUAGE CXX)
# Python calls only the elpis_retrieval_env_* entry points; ordinary static
# linking pulls exactly the objects they reach.
target_link_libraries(elpis_retrieval_bridge PRIVATE
  elpis_hybrid elpis_vector elpis_embedding elpis_corpus elpis_chunking elpis_fms
  elpis_cascade elpis_hash SQLite::SQLite3 Threads::Threads ${ELPIS_LIBM})

# HACF normal retrieval is an ephemeral, finite RAM epoch: no directory,
# SQLite/WAL, content blob, cold FMS artifact or saved manifest is created.
elpis_add_test(structure test_hacf_ephemeral_diskless
  SOURCES ${CMAKE_CURRENT_SOURCE_DIR}/bridge/tests/test_hacf_ephemeral_diskless.c
  LIBS elpis_retrieval_bridge elpis_corpus
  INCLUDES ${CMAKE_CURRENT_SOURCE_DIR}/bridge ${HACF}/include
  ARGS ${STATE}/hacf-ephemeral-no-disk-state TIMEOUT 120)

# The bridge's context graph (retrieval_bridge.h, ABI v2): construction from explicit edges, refusal paths that
# release everything (leak-checked in the sanitizer jobs), one-hop retrieval with the graph identity, destruction.
elpis_add_test(structure test_retrieval_bridge_graph
  SOURCES ${CMAKE_CURRENT_SOURCE_DIR}/bridge/tests/test_retrieval_bridge_graph.c
  LIBS elpis_retrieval_bridge INCLUDES ${CMAKE_CURRENT_SOURCE_DIR}/bridge ${HACF}/include
  ARGS ${STATE}/retrieval-bridge-graph-state TIMEOUT 120)

# HACF's one user-provisioned, physically preallocated associative-state file.
# The initializer runs only on explicit operator invocation. No inference-time
# initialization, append, resize or persistent metadata sidecars.
target_sources(elpis_cascade PRIVATE ${HACF}/src/kernel/hgram_store.c)
elpis_add_test(structure test_hgram_store
  SOURCES ${HACF}/tests/kernel/test_hgram_store.c
  LIBS elpis_cascade TIMEOUT 45)
if(CMAKE_SYSTEM_NAME STREQUAL "Linux")
  add_executable(elpis_hgram_init ${HACF}/src/kernel/hgram_init.c)
  _elpis_register(structure elpis_hgram_init executable)
  target_link_libraries(elpis_hgram_init PRIVATE elpis_cascade)
endif()

# R1 direct, operator-gated fixed association mechanics. Never a runtime writer.
elpis_add_test(structure test_hgram_assoc
  SOURCES ${HACF}/tests/kernel/test_hgram_assoc.c
  LIBS elpis_cascade TIMEOUT 45)
