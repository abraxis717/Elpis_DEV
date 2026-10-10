/* B1d: exact-byte catalog/source binding. Read-only descriptive preflight.
 * The catalog SHA-256 MUST be pinned by an independent trusted deployment
 * channel. Supplying SHA256(catalog) from the same untrusted candidate is
 * self-attestation and conveys NO authority. No admission or write permit.
 * In-memory API; no filesystem, Python, network, or runtime system calls.
 */
#ifndef ELPIS_SEMANTIC_ADMISSION_AUTHORITY_B1D_H
#define ELPIS_SEMANTIC_ADMISSION_AUTHORITY_B1D_H
#include "elpis_semantic/snapshot.h"
#include "elpis_semantic/evidence_admission_policy.h"
#include <stddef.h>
#include <stdint.h>
#ifdef __cplusplus
extern "C" {
#endif
#define SEMANTIC_B1D_CATALOG_BYTES 148u
#define SEMANTIC_B1D_MAX_ARTIFACT_BYTES (8u * 1024u * 1024u)
/* Strict canonical catalog, total 148 bytes:
 * [0:16] ASCII ELPIS-B1D-AUTH01
 * [16:20] big-endian u32 schema version 1
 * [20:52] expected native semantic admission-policy identity
 * [52:84] expected HACF retrieval bundle PACKAGE identity (not raw SHA)
 * [84:116] SHA256 of raw retrieval bundle artifact bytes
 * [116:148] expected base semantic snapshot manifest identity
 * The catalog SHA pin is independent deployment input, not obtained from the
 * candidate. The raw artifact is additionally structurally untrusted here;
 * parsing/authenticating the retrieval bundle's internal contents is B1e work.
 */
int semantic_b1d_authority_bytes_audit(
    const uint8_t *catalog, size_t catalog_bytes,
    const hacf_digest *trusted_catalog_sha256,
    const semantic_snapshot_manifest *base,
    const elpis_evidence_admission_policy_v1 *policy,
    const uint8_t *raw_retrieval_artifact, size_t raw_retrieval_artifact_bytes,
    hacf_digest *pinned_policy_identity_out,
    hacf_digest *pinned_bundle_package_identity_out,
    hacf_digest *descriptive_audit_out);
#ifdef __cplusplus
}
#endif
#endif
