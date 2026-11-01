/* snapshot_publish.c — publication of a finalized snapshot manifest: OFFLINE ARTIFACT PRODUCTION ONLY.
 *
 * Moved out of snapshot_manifest.c unchanged so that the manifest mechanics (create, add, finalize, validate, digest,
 * read) link without any writer: a library that only builds or reads manifests (the runtime's ingress bridge) carries
 * no mkstemp, link, unlink or fsync. Nothing in a composer calls these; an offline producer that does needs separate
 * human authorization (docs/VOLATILE_RUNTIME_CENSUS.md, OFFLINE_ARTIFACT_PRODUCER). */
#define _DEFAULT_SOURCE
#include "elpis_semantic/snapshot_publication.h"
#include "elpis_semantic/identity.h"
#include "elpis/sha256.h"
#include <errno.h>
#include <fcntl.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <unistd.h>

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
