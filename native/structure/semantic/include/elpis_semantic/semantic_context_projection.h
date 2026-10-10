/* Track C v1: explicit admitted semantic relation -> native HACF chunk edge.
 * Read-only; no inference, corpus write, snapshot publish or ECS interface.
 * The trusted caller supplies independently authenticated admission-layer and
 * snapshot authority. A self-created layer is NOT an authorization grant.
 * The initial safe domain is admitted CLAIM_NODE -> CLAIM_NODE relations backed
 * by PRIMARY retrieval-item evidence on each endpoint. Other cases fail closed.
 */
#ifndef ELPIS_SEMANTIC_CONTEXT_PROJECTION_H
#define ELPIS_SEMANTIC_CONTEXT_PROJECTION_H
#include "elpis_semantic/evidence_admission.h"
#include "elpis_semantic/evidence_admission_decision.h"
#include "elpis_semantic/evidence_admission_receipt.h"
#include "elpis_semantic/evidence_relation_candidate.h"
#include "elpis_semantic/evidence_span.h"
#include "elpis_semantic/retrieval_item_attachment.h"
#include "elpis/context_graph.h"
#include "elpis/corpus.h"
#include <stdint.h>
#ifdef __cplusplus
extern "C" {
#endif
#define ELPIS_SEMANTIC_CGRAPH_PROJECTION_ABI_VERSION 1u
/* Versioned, explicit mapping; no accidental equality of semantic/context ids. */
#define ELPIS_CGRAPH_SEMANTIC_MENTIONS 0x53430101u
#define ELPIS_CGRAPH_SEMANTIC_DEFINES 0x53430102u
#define ELPIS_CGRAPH_SEMANTIC_SUPPORTS 0x53430103u
#define ELPIS_CGRAPH_SEMANTIC_CONTRADICTS 0x53430104u
#define ELPIS_CGRAPH_SEMANTIC_QUALIFIES 0x53430105u
#define ELPIS_CGRAPH_SEMANTIC_LIMITS_SCOPE 0x53430106u
#define ELPIS_CGRAPH_SEMANTIC_PROVIDES_CONTEXT 0x53430107u

typedef struct elpis_semantic_cgraph_endpoint_v1 {
    const elpis_evidence_admission_decision_v1 *decision;
    const elpis_evidence_admission_receipt_v1 *receipt;
    const elpis_evidence_span_v1 *span;
    const elpis_retrieval_item_attachment_v1 *attachment;
    const uint8_t *item_text;
    uint32_t item_text_bytes;
} elpis_semantic_cgraph_endpoint_v1;

typedef struct elpis_semantic_cgraph_projection_v1 {
    const elpis_evidence_admission_v1 *admission_layer;
    const hacf_digest *trusted_base_snapshot_digest;
    const elpis_evidence_relation_candidate_v1 *relation;
    const elpis_evidence_admission_decision_v1 *relation_decision;
    const elpis_evidence_admission_receipt_v1 *relation_receipt;
    elpis_semantic_cgraph_endpoint_v1 source;
    elpis_semantic_cgraph_endpoint_v1 target;
    elpis_corpus *corpus;  /* borrowed, this exact volatile retrieval epoch */
} elpis_semantic_cgraph_projection_v1;

/* Explicit audit binding accompanies the native graph edge. The existing graph
 * ABI remains unchanged. The provenance digest identifies this precise chain;
 * it cannot reconstruct omitted witnesses, which the caller must retain for
 * independent audit. Never treat the digest alone as an admission decision. */
typedef struct elpis_semantic_cgraph_audit_v1 {
    uint32_t abi_version;
    hacf_digest admission_layer_digest;
    hacf_digest snapshot_digest;
    hacf_digest relation_candidate_digest;
    hacf_digest relation_decision_digest;
    hacf_digest relation_receipt_digest;
    hacf_digest source_decision_digest;
    hacf_digest source_receipt_digest;
    hacf_digest source_span_digest;
    hacf_digest source_attachment_digest;
    hacf_digest target_decision_digest;
    hacf_digest target_receipt_digest;
    hacf_digest target_span_digest;
    hacf_digest target_attachment_digest;
    hacf_digest provenance_digest;
} elpis_semantic_cgraph_audit_v1;

/* All-or-nothing. No writes on refusal (including output pointers).
 * The output is one explicitly supported directed chunk edge. For multiple
 * admitted support pairs the caller invokes this per pair, then constructs the
 * existing immutable native graph, which already sorts and deduplicates. */
int elpis_semantic_cgraph_project_v1(const elpis_semantic_cgraph_projection_v1 *proof,
                                     elpis_context_edge_input *edge_out,
                                     elpis_semantic_cgraph_audit_v1 *audit_out);
#ifdef __cplusplus
}
#endif
#endif
