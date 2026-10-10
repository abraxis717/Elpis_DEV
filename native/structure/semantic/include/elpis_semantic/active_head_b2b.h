/* B2b storage transaction: locked, durable active-head CAS.
 * Does NOT authorize semantic admission; callers must independently establish
 * the trust root, provenance and B2a successor before invoking this API.
 * Trusted private directory, local POSIX filesystem with flock/rename/fsync.
 */
#ifndef ELPIS_SEMANTIC_ACTIVE_HEAD_B2B_H
#define ELPIS_SEMANTIC_ACTIVE_HEAD_B2B_H
#include "elpis_semantic/snapshot.h"
#ifdef __cplusplus
extern "C" {
#endif
#define SEMANTIC_B2B_OK 0
#define SEMANTIC_B2B_CONFLICT 1
#define SEMANTIC_B2B_INVALID 2
#define SEMANTIC_B2B_IO 3
#define SEMANTIC_B2B_UNCERTAIN 4
/* Reads and independently verifies active HEAD and its CAS manifest.
 * On any failure, out is unchanged. */
int semantic_b2b_head_read(const char *trusted_dir, hacf_digest *out);
/* Explicit trusted-root initialization: refuses existing HEAD, requires a
 * complete valid genesis manifest at <digest>.snapshot. Not callable by
 * untrusted candidate admission logic. */
int semantic_b2b_head_bootstrap(const char *trusted_dir,
                                const hacf_digest *trusted_genesis_manifest);
/* Atomic compare-and-swap from expected to successor. HEAD remains unchanged
 * on pre-rename failure. Post-rename fsync errors return UNCERTAIN, requiring
 * a read/reconcile; never blindly retry based on that return code.
 * Requires both immutable manifests and the appended verified segment already
 * staged in trusted_dir. Does NOT independently authorize the admission. */
int semantic_b2b_head_cas(const char *trusted_dir,
                          const hacf_digest *expected,
                          const hacf_digest *successor);
#ifdef __cplusplus
}
#endif
#endif
