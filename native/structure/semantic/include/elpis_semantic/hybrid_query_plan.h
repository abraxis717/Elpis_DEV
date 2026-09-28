/* elpis_semantic/hybrid_query_plan.h — hybrid-retrieval query plan.
 *
 * One logical hybrid query plan per context-deficit retrieval requirement. Derives an HACF hybrid
 * hybrid policy from a sealed template + requirement constraints.
 *
 * Identity domain: "elpis.semantic.r3_query_plan.v1"
 */
#ifndef ELPIS_SEMANTIC_HYBRID_QUERY_PLAN_H
#define ELPIS_SEMANTIC_HYBRID_QUERY_PLAN_H

#include "elpis_semantic/retrieval_requirement.h"
#include "elpis_semantic/retrieval_materialization.h"
#include "elpis_semantic/retrieval_epoch.h"
#include "elpis/hybrid_retrieval.h"
#include "elpis/cascade.h"
#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

#define HYBRID_QUERY_PLAN_ABI_VERSION 1u

/* ──────────────────────────────────────────────────────────────────── */
/* hybrid query plan record */
/* ──────────────────────────────────────────────────────────────────── */

typedef struct elpis_hybrid_query_plan_v1 {
    uint32_t                    abi_version;
    hacf_digest                 retrieval_requirement_digest;
    hacf_digest                 materialization_entry_digest;
    hacf_digest                 retrieval_epoch_binding_digest;
    elpis_hybrid_policy         derived_policy;
    hacf_digest                 namespace_digest; /* zero if no namespace filter */
    uint32_t                    authority_floor;   /* 0..3 numeric */
    int                         has_exact_authority; /* 1 = requirement carries exact-authority constraint */
    const char                 *exact_authority; /* non-null if has_exact_authority */
    uint32_t                    requested_result_limit;
    hacf_digest                 query_plan_digest;
    uint8_t                     reserved[32];
} elpis_hybrid_query_plan_v1;

/* Derive a query plan from a context-deficit requirement, materialization entry, and epoch
 * binding. Begins from a sealed HACF hybrid policy template and derives query-specific
 * limits deterministically. Returns SEMANTIC_OK or error.
 *
 * Derived policy rules:
 *   total_limit = requested result limit
 *   primary_limit = min(template.primary_limit, total_limit)
 *   graph_seed_limit = min(template.graph_seed_limit, primary_limit)
 *   graph_neighbors_per_seed = template (unchanged)
 *   lexical_limit = template (unchanged)
 *   dense_limit = template (unchanged)
 *   rrf_k = template (unchanged)
 *   lexical_weight, dense_weight = template (unchanged)
 *   min_graph_authority = max(template.min_graph_authority, requirement.authority_floor) */
int elpis_hybrid_query_plan_derive(
    elpis_hybrid_query_plan_v1 *plan,
    const elpis_semantic_retrieval_requirement_v1 *requirement,
    const elpis_materialization_entry_v1 *materialization,
    const elpis_retrieval_epoch_binding_v1 *epoch_binding,
    const elpis_hybrid_policy *template_policy);

/* Zero-initialize. Sets abi_version. */
void elpis_hybrid_query_plan_init(elpis_hybrid_query_plan_v1 *plan);

/* Compute query plan identity digest.
 * Domain: "elpis.semantic.r3_query_plan.v1"
 * Byte stream: domain_tag || abi_version(4 BE)
 *             || retrieval_requirement_digest(32)
 *             || materialization_entry_digest(32)
 *             || retrieval_epoch_binding_digest(32)
 *             || derived_policy fields (LE32 each, same as HACF hybrid policy digest)
 *             || namespace_digest(32)
 *             || authority_floor(4 BE)
 *             || has_exact_authority(4 BE)
 *             || requested_result_limit(4 BE). */
int elpis_hybrid_query_plan_digest(
    const elpis_hybrid_query_plan_v1 *plan, hacf_digest *out);

/* Validate: known ABI, zero reserved, valid derived policy (via HACF hybrid retrieval validator),
 * authority_floor 0..3, nonzero result limit. */
int elpis_hybrid_query_plan_validate(
    const elpis_hybrid_query_plan_v1 *plan);

#ifdef __cplusplus
}
#endif
#endif
