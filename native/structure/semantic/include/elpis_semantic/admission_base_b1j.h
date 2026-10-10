/* B1j: authenticated genesis-only EMPTY base-chain audit.
 * A successful descriptive hash is NOT a publication permit, NOT a proof of
 * absence in nonempty snapshots, and NOT semantic adjudication. Paths are read
 * only; an independently trusted snapshot manifest digest is mandatory.
 */
#ifndef ELPIS_SEMANTIC_ADMISSION_BASE_B1J_H
#define ELPIS_SEMANTIC_ADMISSION_BASE_B1J_H
#include "elpis_semantic/snapshot.h"
#ifdef __cplusplus
extern "C" {
#endif
int semantic_b1j_empty_base_chain_audit(
    const semantic_snapshot_manifest *manifest,
    const hacf_digest *trusted_manifest_digest,
    const char *const *segment_paths,
    uint32_t path_count,
    hacf_digest *descriptive_report_out);
#ifdef __cplusplus
}
#endif
#endif
