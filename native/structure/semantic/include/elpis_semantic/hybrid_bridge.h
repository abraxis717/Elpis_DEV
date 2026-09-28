/* elpis_semantic/hybrid_bridge.h — hybrid-retrieval bridge execution.
 *
 * Executes one bounded HACF hybrid retrieval per query plan and produces an
 * immutable bridge-execution receipt. Public ABI is C-compatible.
 *
 * Identity domain: "elpis.semantic.r3_bridge.v1"
 */
#ifndef ELPIS_SEMANTIC_HYBRID_BRIDGE_H
#define ELPIS_SEMANTIC_HYBRID_BRIDGE_H

#include "elpis_semantic/hybrid_query_plan.h"
#include "elpis_semantic/retrieval_materialization.h"
#include "elpis/cascade.h"
#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

#define HYBRID_BRIDGE_ABI_VERSION 1u

/* Forward declarations from HACF hybrid public ABI */
struct elpis_corpus;
struct elpis_vector_index;
struct elpis_context_graph;
struct elpis_hybrid_retriever;
struct elpis_retrieval_bundle;
typedef struct elpis_corpus elpis_corpus;
typedef struct elpis_vector_index elpis_vector_index;
typedef struct elpis_context_graph elpis_context_graph;
typedef struct elpis_hybrid_retriever elpis_hybrid_retriever;
typedef struct elpis_retrieval_bundle elpis_retrieval_bundle;

/* ──────────────────────────────────────────────────────────────────── */
/* Bridge execution disposition */
/* ──────────────────────────────────────────────────────────────────── */

typedef enum elpis_hybrid_bridge_disposition {
    HYBRID_RETRIEVAL_COMPLETE     = 0,
    HYBRID_RETRIEVAL_EMPTY        = 1,
    HYBRID_EPOCH_DRIFT            = 2,
    HYBRID_QUERY_REJECTED         = 3,
    HYBRID_CORPUS_FAILURE         = 4,
    HYBRID_VECTOR_FAILURE         = 5,
    HYBRID_GRAPH_FAILURE          = 6,
    HYBRID_INTEGRITY_FAILURE      = 7,
    HYBRID_LIMIT_FAILURE          = 8,
    HYBRID_INTERNAL_FAILURE       = 9
} elpis_hybrid_bridge_disposition;

/* ──────────────────────────────────────────────────────────────────── */
/* Bridge execution receipt */
/* ──────────────────────────────────────────────────────────────────── */

typedef struct elpis_hybrid_bridge_receipt_v1 {
    uint32_t                    abi_version;
    hacf_digest                 query_plan_digest;
    hacf_digest                 hybrid_query_digest;
    hacf_digest                 epoch_binding_digest;
    hacf_digest                 retrieval_bundle_digest;
    hacf_digest                 retrieval_bundle_package_digest;
    uint32_t                    item_count;
    elpis_hybrid_bridge_disposition disposition;
    hacf_digest                 bridge_execution_digest;
    uint8_t                     reserved[32];
} elpis_hybrid_bridge_receipt_v1;

/* Execute one HACF hybrid retrieval for a query plan.
 * Returns SEMANTIC_OK on success (receipt filled, bundle_out owned by caller).
 * On failure, disposition is set in receipt; bundle_out is NULL.
 * Caller must call elpis_retrieval_bundle_destroy(*bundle_out) on success.
 *
 * Steps:
 *  1. Verify context-deficit retrieval requirement
 *  2. Verify materialization entry
 *  3. Verify epoch binding
 *  4. Construct exact elpis_hybrid_query
 *  5. Preserve exact query-text bytes
 *  6. Preserve exact float32 vector bytes
 *  7. Apply exact namespace filter
 *  8. Apply exact authority filter only if requirement carries exact constraint
 *  9. Create HACF hybrid retriever
 * 10. Execute one bounded retrieval
 * 11. Receive immutable RetrievalBundle
 * 12. Verify post-search epoch identity
 * 13. Produce immutable bridge-execution receipt */
int elpis_hybrid_bridge_execute(
    elpis_hybrid_bridge_receipt_v1 *receipt,
    elpis_retrieval_bundle **bundle_out,
    const elpis_hybrid_query_plan_v1 *plan,
    const elpis_materialization_entry_v1 *materialization,
    elpis_corpus *corpus,
    elpis_vector_index *index,
    const elpis_context_graph *graph);

/* Zero-initialize a receipt. Sets abi_version. */
void elpis_hybrid_bridge_receipt_init(elpis_hybrid_bridge_receipt_v1 *receipt);

/* Compute bridge execution receipt identity digest.
 * Domain: "elpis.semantic.r3_bridge.v1"
 * Byte stream: domain_tag || abi_version(4 BE)
 *             || query_plan_digest(32)
 *             || hybrid_query_digest(32)
 *             || epoch_binding_digest(32)
 *             || retrieval_bundle_digest(32)
 *             || retrieval_bundle_package_digest(32)
 *             || item_count(4 BE)
 *             || disposition(4 BE). */
int elpis_hybrid_bridge_receipt_digest(
    const elpis_hybrid_bridge_receipt_v1 *receipt, hacf_digest *out);

/* Validate receipt: known ABI, zero reserved, valid disposition,
 * nonzero digests for complete disposition. */
int elpis_hybrid_bridge_receipt_validate(
    const elpis_hybrid_bridge_receipt_v1 *receipt);

/* Map HACF hybrid status code to bridge disposition. */
elpis_hybrid_bridge_disposition elpis_hybrid_bridge_disposition_from_status(int status);

#ifdef __cplusplus
}
#endif
#endif
