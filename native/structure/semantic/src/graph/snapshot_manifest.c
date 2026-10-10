/* snapshot_manifest.c — Append-only snapshot manifests with chain validation.
 *
 * Chain continuity: segment prior snapshot equals preceding result snapshot.
 * Final segment result equals manifest graph-snapshot digest.
 * Registry identity constant within chain. No omitted or duplicate segments.
 */
#define _DEFAULT_SOURCE
#include "elpis_semantic/snapshot_publication.h"
#include "elpis_semantic/identity.h"
#include "elpis/sha256.h"
#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <fcntl.h>
#include <unistd.h>
#include <sys/stat.h>
#include <errno.h>

/* Zero-comparison buffer — prevents global-buffer-overflow on string literal. */
static const uint8_t ZERO_64[64] = {0};
#include <arpa/inet.h>

semantic_snapshot_manifest *semantic_snapshot_create(void) {
    semantic_snapshot_manifest *m = calloc(1, sizeof(*m));
    if (m) m->abi_version = SEMANTIC_SNAPSHOT_ABI_VERSION;
    return m;
}

void semantic_snapshot_destroy(semantic_snapshot_manifest *m) {
    free(m);
}

int semantic_snapshot_add_segment(semantic_snapshot_manifest *m,
                                   const semantic_segment_record *segment) {
    if (!m || !segment) return SEMANTIC_E_INVAL;
    if (segment->abi_version != SEMANTIC_SEGMENT_ABI_VERSION) return SEMANTIC_E_INVAL;
    if (m->segment_count >= SEMANTIC_MAX_SEGMENTS) return SEMANTIC_E_NOMEM;

    if (m->abi_version != SEMANTIC_SNAPSHOT_ABI_VERSION) return SEMANTIC_E_INVAL;
    if (m->segment_count != 0 &&
        memcmp(segment->type_registry_digest.bytes, m->type_registry_digest.bytes,
               HACF_DIGEST_BYTES) != 0) return SEMANTIC_E_INVAL;

    /* Check for duplicate segment. */
    for (uint32_t i = 0; i < m->segment_count; i++) {
        if (memcmp(m->segment_digests[i].bytes, segment->segment_identity.bytes, HACF_DIGEST_BYTES) == 0)
            return SEMANTIC_E_DUPLICATE;
    }

    const hacf_digest *expected_prior = m->segment_count == 0
        ? &m->genesis_identity : &m->hacf_graph_snapshot_digest;
    if (memcmp(segment->prior_snapshot_digest.bytes, expected_prior->bytes,
               HACF_DIGEST_BYTES) != 0) return SEMANTIC_E_INVAL;

    /* Reject before changing any manifest state. */
    if (segment->node_count > UINT32_MAX - m->unique_node_count ||
        segment->hyperedge_count > UINT32_MAX - m->unique_hyperedge_count ||
        segment->assertion_count > UINT32_MAX - m->assertion_count ||
        segment->incidence_count > UINT32_MAX - m->incidence_count)
        return SEMANTIC_E_INVAL;
    if (m->segment_count == 0)
        m->type_registry_digest = segment->type_registry_digest;

    m->segment_digests[m->segment_count] = segment->segment_identity;
    m->segment_count++;
    m->unique_node_count += segment->node_count;
    m->unique_hyperedge_count += segment->hyperedge_count;
    m->assertion_count += segment->assertion_count;
    m->incidence_count += segment->incidence_count;

    /* Track the latest graph-snapshot digest. */
    m->hacf_graph_snapshot_digest = segment->hacf_next_snapshot;

    return SEMANTIC_OK;
}

int semantic_snapshot_finalize(semantic_snapshot_manifest *m) {
    if (!m || m->segment_count == 0 ||
        m->segment_count > SEMANTIC_MAX_SEGMENTS) return SEMANTIC_E_INVAL;

    elpis_sha256_ctx ctx;
    elpis_sha256_init(&ctx);

    const char *domain = "elpis.semantic.snapshot.v1";
    uint32_t be_len = htonl((uint32_t)strlen(domain));
    elpis_sha256_update(&ctx, &be_len, 4);
    elpis_sha256_update(&ctx, domain, strlen(domain));

    uint32_t be = htonl(m->abi_version);
    elpis_sha256_update(&ctx, &be, 4);
    elpis_sha256_update(&ctx, m->type_registry_digest.bytes, HACF_DIGEST_BYTES);
    elpis_sha256_update(&ctx, m->genesis_identity.bytes, HACF_DIGEST_BYTES);
    elpis_sha256_update(&ctx, m->prior_manifest_digest.bytes, HACF_DIGEST_BYTES);
    be = htonl(m->segment_count);
    elpis_sha256_update(&ctx, &be, 4);

    for (uint32_t i = 0; i < m->segment_count; i++) {
        elpis_sha256_update(&ctx, m->segment_digests[i].bytes, HACF_DIGEST_BYTES);
    }

    elpis_sha256_update(&ctx, m->hacf_graph_snapshot_digest.bytes, HACF_DIGEST_BYTES);
    be = htonl(m->unique_node_count);
    elpis_sha256_update(&ctx, &be, 4);
    be = htonl(m->unique_hyperedge_count);
    elpis_sha256_update(&ctx, &be, 4);
    be = htonl(m->assertion_count);
    elpis_sha256_update(&ctx, &be, 4);
    be = htonl(m->incidence_count);
    elpis_sha256_update(&ctx, &be, 4);

    elpis_sha256_final(&ctx, m->manifest_digest.bytes);
    m->hacf_package_digest = m->manifest_digest;

    return SEMANTIC_OK;
}

int semantic_snapshot_validate(const semantic_snapshot_manifest *m) {
    if (!m) return SEMANTIC_E_INVAL;
    if (m->abi_version != SEMANTIC_SNAPSHOT_ABI_VERSION) return SEMANTIC_E_INVAL;
    if (m->segment_count == 0 || m->segment_count > SEMANTIC_MAX_SEGMENTS)
        return SEMANTIC_E_INVAL;
    if (memcmp(m->reserved, ZERO_64, sizeof(m->reserved)) != 0) return SEMANTIC_E_RESERVATION;

    /* Verify manifest digest. */
    semantic_snapshot_manifest check = *m;
    memset(&check.manifest_digest, 0, sizeof(check.manifest_digest));
    if (semantic_snapshot_finalize(&check) != SEMANTIC_OK) return SEMANTIC_E_INVAL;
    if (memcmp(check.manifest_digest.bytes, m->manifest_digest.bytes, HACF_DIGEST_BYTES) != 0 ||
        memcmp(check.hacf_package_digest.bytes, m->hacf_package_digest.bytes,
               HACF_DIGEST_BYTES) != 0)
        return SEMANTIC_E_DIGEST;
    /* The serialized file contains all 4096 slots. Without canonical zero
     * unused slots, two byte-distinct files can claim the same manifest key. */
    for (uint32_t i = m->segment_count; i < SEMANTIC_MAX_SEGMENTS; ++i) {
        static const uint8_t zero_digest[HACF_DIGEST_BYTES] = {0};
        if (memcmp(m->segment_digests[i].bytes, zero_digest, HACF_DIGEST_BYTES) != 0)
            return SEMANTIC_E_RESERVATION;
    }

    return SEMANTIC_OK;
}

int semantic_snapshot_digest(const semantic_snapshot_manifest *m, hacf_digest *out) {
    if (!m || !out) return SEMANTIC_E_INVAL;
    *out = m->manifest_digest;
    return SEMANTIC_OK;
}

int semantic_snapshot_write(const semantic_snapshot_manifest *m,
                             const char *path,
                             char hex_out[65]) {
    if (!m || !path) return SEMANTIC_E_INVAL;
    int rc = semantic_snapshot_validate(m);
    if (rc != SEMANTIC_OK) return rc;

    /* The destination directory is trusted. Temp and destination must share
     * a filesystem. Never follow an existing destination, including a
     * dangling symlink; never replace another committed manifest. */
    size_t len = strlen(path);
    if (len == 0 || len > 4000 || path[len - 1] == '/') return SEMANTIC_E_INVAL;
    const char *base = strrchr(path, '/');
    base = base ? base + 1 : path;
    if (!*base || strcmp(base, ".") == 0 || strcmp(base, "..") == 0)
        return SEMANTIC_E_INVAL;
    char dir[4096];
    size_t dir_len = (size_t)(base - path);
    if (dir_len == 0) {
        strcpy(dir, ".");
    } else {
        if (dir_len >= sizeof(dir)) return SEMANTIC_E_INVAL;
        memcpy(dir, path, dir_len);
        dir[dir_len] = 0;
    }

    /* A same-directory random temporary name is exclusive and private. */
    char tmp_path[4096];
    const char suffix[] = ".tmp_snap_XXXXXX";
    size_t dir_size = strlen(dir);
    if (dir_size + sizeof(suffix) + 1 > sizeof(tmp_path)) return SEMANTIC_E_INVAL;
    memcpy(tmp_path, dir, dir_size);
    if (dir[dir_size - 1] != '/') tmp_path[dir_size++] = '/';
    memcpy(tmp_path + dir_size, suffix, sizeof(suffix));
    int fd = mkstemp(tmp_path);
    if (fd < 0) return SEMANTIC_E_IO;

    /* write() is allowed to complete partially, even for regular files. */
    const uint8_t *bytes = (const uint8_t *)m;
    size_t offset = 0;
    while (offset < sizeof(*m)) {
        ssize_t n = write(fd, bytes + offset, sizeof(*m) - offset);
        if (n < 0 && errno == EINTR) continue;
        if (n <= 0) { close(fd); unlink(tmp_path); return SEMANTIC_E_IO; }
        offset += (size_t)n;
    }
    if (fsync(fd) != 0) { close(fd); unlink(tmp_path); return SEMANTIC_E_IO; }
    if (close(fd) != 0) { unlink(tmp_path); return SEMANTIC_E_IO; }

    if (link(tmp_path, path) != 0) {
        int saved = errno;
        unlink(tmp_path);
        return saved == EEXIST ? SEMANTIC_E_DUPLICATE : SEMANTIC_E_IO;
    }
    /* Once link() succeeds, failure is potentially post-publication. Return
     * IO, preserve the destination, and require readback reconciliation. */
    if (unlink(tmp_path) != 0) return SEMANTIC_E_IO;
    int dfd = open(dir, O_RDONLY | O_DIRECTORY | O_CLOEXEC | O_NOFOLLOW);
    if (dfd < 0) return SEMANTIC_E_IO;
    int sync_rc = fsync(dfd);
    int close_rc = close(dfd);
    if (sync_rc != 0 || close_rc != 0) return SEMANTIC_E_IO;
    if (hex_out) elpis_hex32(m->manifest_digest.bytes, hex_out);
    return SEMANTIC_OK;
}

/* Immutable content-addressed publication: the path is derived exclusively
 * from the finalized manifest identity. An existing digest path is a
 * duplicate, never an overwrite, even when bytes happen to be identical. */
int semantic_snapshot_publish_cas(const semantic_snapshot_manifest *m,
                                  const char *directory,
                                  char hex_out[65]) {
    if (!m || !directory || !*directory) return SEMANTIC_E_INVAL;
    int rc = semantic_snapshot_validate(m);
    if (rc != SEMANTIC_OK) return rc;
    char digest[65];
    elpis_hex32(m->manifest_digest.bytes, digest);
    const size_t size = strlen(directory);
    const char extension[] = ".snapshot";
    if (size > 4000 || size + 1 + 64 + sizeof(extension) >= 4096)
        return SEMANTIC_E_INVAL;
    char path[4096];
    int n = snprintf(path, sizeof(path), "%s%s%s%s", directory,
                     directory[size - 1] == '/' ? "" : "/", digest, extension);
    if (n < 0 || (size_t)n >= sizeof(path)) return SEMANTIC_E_INVAL;
    return semantic_snapshot_write(m, path, hex_out);
}

int semantic_snapshot_read(const char *path, semantic_snapshot_manifest *m_out) {
    if (!path || !m_out) return SEMANTIC_E_INVAL;

    FILE *f = fopen(path, "rb");
    if (!f) return SEMANTIC_E_IO;

    semantic_snapshot_manifest candidate;
    if (fread(&candidate, sizeof(candidate), 1, f) != 1) {
        fclose(f);
        return SEMANTIC_E_IO;
    }
    int trailing = fgetc(f);
    int read_error = ferror(f);
    int close_error = fclose(f);
    if (read_error || close_error) return SEMANTIC_E_IO;
    if (trailing != EOF) return SEMANTIC_E_INVAL;

    int rc = semantic_snapshot_validate(&candidate);
    if (rc != SEMANTIC_OK) return rc;
    *m_out = candidate;
    return SEMANTIC_OK;
}
