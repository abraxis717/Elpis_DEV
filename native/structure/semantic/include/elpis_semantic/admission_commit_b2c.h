/* B2c: restricted native admission-to-durable-HEAD transaction.
 * The caller supplies independent, trusted pins in semantic_b2a_witness.
 * This API is privileged control-plane code: not a general-purpose CAS or an
 * authorization token. It supports only B2a's single canonical claim shape.
 * Storage root must be a private directory on a local POSIX filesystem.
 */
#ifndef ELPIS_SEMANTIC_ADMISSION_COMMIT_B2C_H
#define ELPIS_SEMANTIC_ADMISSION_COMMIT_B2C_H
#include "elpis_semantic/admission_successor_b2a.h"
#include "elpis_semantic/active_head_b2b.h"
#ifdef __cplusplus
extern "C" {
#endif
/* Return SEMANTIC_B2B_OK / CONFLICT / INVALID / IO / UNCERTAIN.
 * A success means active HEAD was durably switched to the freshly reaudited
 * successor. On UNCERTAIN caller MUST reconcile with head_read before any
 * attempt to retry; a conflict or error must never be interpreted as success.
 * Orphaned immutable objects after a failed CAS are safe and reusable if exact.
 * No implicit bootstrap and no output-based permission grant.
 */
int semantic_b2c_publish_one(const char *trusted_root,
                              const semantic_b2a_witness *w);
/* B2h restricted, witness-bound verification of the current published HEAD,
 * all base CAS segment bytes, and exact admitted successor segment/manifest.
 * Does not mutate HEAD or CAS and does not replace admission trust authority.
 * A bare B2b HEAD read is NOT an equivalent full-inventory verification. */
int semantic_b2h_verify_published_one(const char *trusted_root,
                                      const semantic_b2a_witness *trusted_witness);
#ifdef __cplusplus
}
#endif
#endif
