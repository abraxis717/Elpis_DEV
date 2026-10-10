/* B1d: external SHA trust-root binding; zero I/O and zero write privileges. */
#include "elpis_semantic/admission_authority_b1d.h"
#include "elpis/sha256.h"
#include <string.h>

static const uint8_t MAGIC[16] = {
    'E','L','P','I','S','-','B','1','D','-','A','U','T','H','0','1'
};
static int match_digest(const uint8_t *expected, const hacf_digest *actual) {
    return memcmp(expected, actual->bytes, HACF_DIGEST_BYTES) == 0;
}
static int nonzero(const hacf_digest *d) {
    static const uint8_t ZERO[HACF_DIGEST_BYTES] = {0};
    return d && memcmp(d->bytes, ZERO, sizeof(ZERO)) != 0;
}
int semantic_b1d_authority_bytes_audit(
    const uint8_t *catalog, size_t catalog_bytes,
    const hacf_digest *trusted_catalog_sha256,
    const semantic_snapshot_manifest *base,
    const elpis_evidence_admission_policy_v1 *policy,
    const uint8_t *raw_retrieval_artifact, size_t raw_retrieval_artifact_bytes,
    hacf_digest *pinned_policy_identity_out,
    hacf_digest *pinned_bundle_package_identity_out,
    hacf_digest *descriptive_audit_out) {
    /* All three outputs are transactional: never mutate any on refusal. */
    if (!catalog || catalog_bytes != SEMANTIC_B1D_CATALOG_BYTES ||
        !trusted_catalog_sha256 || !nonzero(trusted_catalog_sha256) ||
        !base || !policy || !raw_retrieval_artifact ||
        raw_retrieval_artifact_bytes == 0 ||
        raw_retrieval_artifact_bytes > SEMANTIC_B1D_MAX_ARTIFACT_BYTES ||
        !pinned_policy_identity_out || !pinned_bundle_package_identity_out ||
        !descriptive_audit_out ||
        pinned_policy_identity_out == pinned_bundle_package_identity_out ||
        pinned_policy_identity_out == descriptive_audit_out ||
        pinned_bundle_package_identity_out == descriptive_audit_out)
        return SEMANTIC_E_INVAL;
    if (memcmp(catalog, MAGIC, sizeof(MAGIC)) != 0 ||
        catalog[16] != 0 || catalog[17] != 0 ||
        catalog[18] != 0 || catalog[19] != 1)
        return SEMANTIC_E_INVAL;
    hacf_digest catalog_hash, artifact_hash, policy_hash;
    elpis_sha256(catalog, catalog_bytes, catalog_hash.bytes);
    if (!match_digest(catalog_hash.bytes, trusted_catalog_sha256))
        return SEMANTIC_E_AUTHORITY;
    if (semantic_snapshot_validate(base) != SEMANTIC_OK ||
        elpis_admission_policy_validate(policy) != SEMANTIC_OK ||
        elpis_admission_policy_identity(policy, &policy_hash) != SEMANTIC_OK)
        return SEMANTIC_E_AUTHORITY;
    elpis_sha256(raw_retrieval_artifact, raw_retrieval_artifact_bytes,
                 artifact_hash.bytes);
    if (!match_digest(catalog + 20, &policy_hash) ||
        !match_digest(catalog + 84, &artifact_hash) ||
        !match_digest(catalog + 116, &base->manifest_digest))
        return SEMANTIC_E_AUTHORITY;
    hacf_digest package_hash;
    memcpy(package_hash.bytes, catalog + 52, HACF_DIGEST_BYTES);
    if (!nonzero(&package_hash)) return SEMANTIC_E_AUTHORITY;
    static const char DOMAIN[] = "elpis.semantic.b1d.authority-bytes-audit.v1";
    elpis_sha256_ctx ctx;
    elpis_sha256_init(&ctx);
    elpis_sha256_update(&ctx, DOMAIN, sizeof(DOMAIN) - 1);
    elpis_sha256_update(&ctx, catalog_hash.bytes, HACF_DIGEST_BYTES);
    elpis_sha256_update(&ctx, base->manifest_digest.bytes, HACF_DIGEST_BYTES);
    elpis_sha256_update(&ctx, policy_hash.bytes, HACF_DIGEST_BYTES);
    elpis_sha256_update(&ctx, package_hash.bytes, HACF_DIGEST_BYTES);
    elpis_sha256_update(&ctx, artifact_hash.bytes, HACF_DIGEST_BYTES);
    hacf_digest report;
    elpis_sha256_final(&ctx, report.bytes);
    *pinned_policy_identity_out = policy_hash;
    *pinned_bundle_package_identity_out = package_hash;
    *descriptive_audit_out = report;
    return SEMANTIC_OK;
}
