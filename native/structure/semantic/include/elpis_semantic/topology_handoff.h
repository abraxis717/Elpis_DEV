/* elpis_semantic/topology_handoff.h — Grid81-compiler topology handoff ABI.
 *
 * Creates SEMANTIC_TO_GRID81_COMPILER_INPUT handoff binding topology topology
 * IR for a future Grid81-compiler Grid81 structural compiler.
 *
 * Identity domain: "elpis.semantic.topology_handoff.v1"
 */
#ifndef ELPIS_SEMANTIC_TOPOLOGY_HANDOFF_H
#define ELPIS_SEMANTIC_TOPOLOGY_HANDOFF_H

#include "elpis_semantic/semantic_topology_ir.h"
#include "elpis/cascade.h"
#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

#define TOPOLOGY_HANDOFF_ABI_VERSION 1u

/* ──────────────────────────────────────────────────────────────────── */
/* Handoff kind — for Grid81-compiler                                                */
/* ──────────────────────────────────────────────────────────────────── */

typedef enum topology_handoff_kind {
    TOPOLOGY_HANDOFF_SEMANTIC_TO_GRID81_COMPILER_INPUT = 0,
} topology_handoff_kind;

/* ──────────────────────────────────────────────────────────────────── */
/* Grid81-compiler handoff packet                                                     */
/* ──────────────────────────────────────────────────────────────────── */

typedef struct elpis_semantic_topology_handoff_v1 {
    uint32_t                          abi_version;
    uint32_t                          handoff_kind; /* topology_handoff_kind */

    /* Root query overlay from bounded-view */
    hacf_digest                       root_query_overlay_digest;

    /* bounded-view bounded view */
    hacf_digest                       P5_bounded_view_digest;

    /* topology topology IR */
    hacf_digest                       topology_IR_digest;

    /* Compile receipt */
    hacf_digest                       compile_receipt_digest;

    /* Policy and registry */
    hacf_digest                       topology_policy_digest;
    hacf_digest                       relation_registry_digest;

    /* Type-registry chain and authority */
    hacf_digest                       type_registry_chain_digest;
    hacf_digest                       authority_registry_digest;

    /* Ordered topology addresses and constraints */
    hacf_digest                       ordered_addresses_digest;
    hacf_digest                       ordered_constraints_digest;

    /* Feature schema and dependency manifest */
    hacf_digest                       feature_schema_digest;
    hacf_digest                       dependency_manifest_digest;

    /* Handoff and package identities */
    hacf_digest                       handoff_digest;
    hacf_digest                       HACF_package_digest;

    /* Explicit Grid81-compiler boundaries */
    uint32_t                          P7_may_assign_discrete_placement;  /* 1 */
    uint32_t                          P7_may_not_alter_relation_types;   /* 1 */
    uint32_t                          P7_may_not_alter_authority;        /* 1 */
    uint32_t                          P7_may_not_remove_conflict_polarity; /* 1 */
    uint32_t                          P7_may_not_treat_metric_as_semantic; /* 1 */
    uint32_t                          local_ordinal_is_not_grid81_cell;  /* 1 */
    uint32_t                          one_vertex_not_one_cell;           /* 1 */

    uint8_t                           reserved[64];
} elpis_semantic_topology_handoff_v1;

/* ──────────────────────────────────────────────────────────────────── */
/* Operations                                                            */
/* ──────────────────────────────────────────────────────────────────── */

/* Initialize handoff. Sets abi_version and all Grid81-compiler boundary flags. */
void elpis_topology_handoff_init(elpis_semantic_topology_handoff_v1 *handoff);

/* Construct Grid81-compiler handoff from completed compilation context. */
int elpis_topology_handoff_construct(
    elpis_semantic_topology_handoff_v1 *handoff,
    const elpis_topology_compile_context *ctx,
    const elpis_semantic_downstream_handoff_v1 *P5_handoff);

/* Compute handoff identity. Domain: "elpis.semantic.topology_handoff.v1" */
int elpis_topology_handoff_identity(
    const elpis_semantic_topology_handoff_v1 *handoff, hacf_digest *out);

/* Validate handoff: ABI version, Grid81-compiler boundaries set, digests non-zero. */
int elpis_topology_handoff_validate(
    const elpis_semantic_topology_handoff_v1 *handoff);

/* Persistence */
int elpis_write_topology_handoff(const char *path,
    const elpis_semantic_topology_handoff_v1 *handoff);
int elpis_read_topology_handoff(const char *path,
    elpis_semantic_topology_handoff_v1 *out);

#ifdef __cplusplus
}
#endif
#endif
