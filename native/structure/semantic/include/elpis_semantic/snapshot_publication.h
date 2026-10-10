/* snapshot_publication.h — Track B0 native storage extension.
 * This extension deliberately does not modify the release-sealed snapshot.h.
 * Filesystem publication is explicit, blocking, and not a per-token operation.
 * No query overlay or admission decision is promoted by this storage API.
 */
#ifndef ELPIS_SEMANTIC_SNAPSHOT_PUBLICATION_H
#define ELPIS_SEMANTIC_SNAPSHOT_PUBLICATION_H

#include "elpis_semantic/snapshot.h"

#ifdef __cplusplus
extern "C" {
#endif

/* Write immutable full manifest. EEXIST (including dangling symlinks) maps to
 * SEMANTIC_E_DUPLICATE. File and directory are fsynced before SEMANTIC_OK.
 * After an IO failure the destination may exist; callers must reconcile by
 * verified readback rather than assuming absence. Trusted destination dir. */

/* Legacy semantic_snapshot_write() is declared in the sealed snapshot.h; its
 * implementation now also honors immutable, no-replace publication. */

/* Content-addressed publication at directory/<manifest_digest>.snapshot.
 * Identity is the existing semantic manifest digest (no ABI or hash changes).
 * A storage helper only: it does not authorize semantic state transitions. */
int semantic_snapshot_publish_cas(const semantic_snapshot_manifest *m,
                                  const char *directory,
                                  char hex_out[65]);

#ifdef __cplusplus
}
#endif
#endif
