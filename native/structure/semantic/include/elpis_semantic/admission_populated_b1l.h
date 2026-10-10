/* Native read-only witness for one pinned, populated, genesis segment.
 * This proves membership/nonmembership of a NODE ID in that entire verified
 * segment. It does NOT prove semantic novelty across historical manifests,
 * payload equivalence, conflicts, or permission to publish.
 */
#ifndef ELPIS_SEMANTIC_ADMISSION_POPULATED_B1L_H
#define ELPIS_SEMANTIC_ADMISSION_POPULATED_B1L_H
#include "elpis_semantic/snapshot.h"
#ifdef __cplusplus
extern "C" {
#endif
/* Reuses the complete existing segment reader's single-file validation pass.
 * Presence is derived from the validated in-memory node array before free.
 * Outputs are written only on SEMANTIC_OK. */
int semantic_segment_read_node_presence(const char *path,
    const hacf_digest *target_node_identity,
    semantic_segment_record *segment_out,
    hacf_digest *segment_digest_out,
    uint32_t *match_count_out);

int semantic_b1l_populated_genesis_node_audit(
    const semantic_snapshot_manifest *manifest,
    const hacf_digest *trusted_manifest_digest,
    const char *segment_path,
    const hacf_digest *target_node_identity,
    uint32_t *node_present_out,
    hacf_digest *descriptive_report_out);
/* B1m: target identity membership over a fully verified bounded multi-segment
 * genesis manifest. Does not prove global object uniqueness or semantics. */
#define SEMANTIC_B1M_MAX_SEGMENTS 64u
int semantic_b1m_populated_chain_node_audit(
    const semantic_snapshot_manifest *manifest,
    const hacf_digest *trusted_manifest_digest,
    const char *const *segment_paths,
    uint32_t path_count,
    const hacf_digest *target_node_identity,
    uint32_t *node_present_out,
    hacf_digest *descriptive_report_out);
/* B1n: in one fully verified descriptor pass return owned node identities.
 * Caller frees *nodes_out on success; outputs are unchanged on failure. */
int semantic_segment_read_node_inventory(const char *path,
    semantic_segment_record *segment_out, hacf_digest *segment_digest_out,
    hacf_digest **nodes_out, uint32_t *node_count_out);

/* Genesis-only, 2..64 segment global node identity nonduplication.
 * No historical manifest claims, cross-type semantic conflict claims or writes. */
#define SEMANTIC_B1N_MAX_TOTAL_NODES 32768u
int semantic_b1n_global_unique_node_audit(
    const semantic_snapshot_manifest *manifest,
    const hacf_digest *trusted_manifest_digest,
    const char *const *segment_paths, uint32_t path_count,
    const hacf_digest *target_node_identity,
    uint32_t *node_present_out, hacf_digest *descriptive_report_out);
/* B1p: count exact (type, payload SHA-256) matches in one verified
 * segment read, ignoring identity-affecting flags. No writes. */
int semantic_segment_read_typed_payload_occurrences(const char *path,
    uint32_t node_type, const hacf_digest *payload_digest,
    semantic_segment_record *segment_out,
    hacf_digest *segment_digest_out, uint32_t *occurrences_out);
#ifdef __cplusplus
}
#endif
#endif
