/* The H-gram probe over fixture files (elpis/hgram_store.h): a retained H-gram is a read-only static resource.
 * Valid files are accepted whether owner-writable or not; absent, linked, sparse, modified, foreign-readable or
 * symlinked files are refused; probing changes nothing. The retired initializer is absent from the API. */
#define _GNU_SOURCE 1
#include "elpis/hgram_store.h"
#include "hgram_fixture.h"

#include <assert.h>
#include <dirent.h>
#include <errno.h>
#include <fcntl.h>
#include <limits.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <unistd.h>

static void pathcat(char *out, size_t cap, const char *dir, const char *leaf) {
    int n = snprintf(out, cap, "%s/%s", dir, leaf);
    assert(n > 0 && (size_t)n < cap);
}
static unsigned entries(const char *dir) {
    DIR *d = opendir(dir);
    assert(d);
    unsigned n = 0;
    struct dirent *e;
    while ((e = readdir(d)) != NULL) n += strcmp(e->d_name, ".") && strcmp(e->d_name, "..");
    assert(closedir(d) == 0);
    return n;
}
static int same_stat(const struct stat *a, const struct stat *b) {
    return a->st_ino == b->st_ino && a->st_size == b->st_size && a->st_mode == b->st_mode &&
           a->st_mtim.tv_sec == b->st_mtim.tv_sec && a->st_mtim.tv_nsec == b->st_mtim.tv_nsec &&
           a->st_ctim.tv_sec == b->st_ctim.tv_sec && a->st_ctim.tv_nsec == b->st_ctim.tv_nsec;
}

int main(void) {
    char cwd[PATH_MAX - 64], dir[PATH_MAX];
    assert(getcwd(cwd, sizeof cwd));
    pathcat(dir, sizeof dir, cwd, "hgram-XXXXXX");
    assert(mkdtemp(dir) != NULL);   /* 0700: the probe requires a private parent */
    char bin[PATH_MAX + 32], ro[PATH_MAX + 32], sym[PATH_MAX + 32], sparse[PATH_MAX + 32], hard[PATH_MAX + 32],
        foreign[PATH_MAX + 32];
    pathcat(bin, sizeof bin, dir, "hgram.bin");
    pathcat(ro, sizeof ro, dir, "readonly.bin");
    pathcat(sym, sizeof sym, dir, "symlink.bin");
    pathcat(sparse, sizeof sparse, dir, "sparse.bin");
    pathcat(hard, sizeof hard, dir, "hard.bin");
    pathcat(foreign, sizeof foreign, dir, "foreign.bin");

    elpis_hgram_info a = {0}, b = {0};
    assert(elpis_hgram_probe(bin, &a) == ELPIS_HGRAM_PATH);            /* absent: never created on read */
    assert(entries(dir) == 0);
    assert(elpis_hgram_probe("relative.bin", &a) == ELPIS_HGRAM_PATH);
    assert(elpis_hgram_probe(bin, NULL) == ELPIS_HGRAM_INVALID);

    /* A valid static resource, owner-writable (0600) or read-only (0400): accepted, and probing changes nothing. */
    assert(hgram_fixture_create(bin, 1, 0600) == 0 && hgram_fixture_create(ro, 2, 0400) == 0);
    struct stat before, after;
    assert(stat(bin, &before) == 0);
    assert(elpis_hgram_probe(bin, &a) == ELPIS_HGRAM_EXISTS);
    assert(a.total_bytes == 1048576ULL && a.slots == (1048576 - 4096) / 256 && a.capacity_mib == 1);
    assert(elpis_hgram_probe(ro, &b) == ELPIS_HGRAM_EXISTS && b.capacity_mib == 2);
    assert(stat(bin, &after) == 0 && same_stat(&before, &after));

    /* A second link, a sparse file of the right length, a modified header, a foreign-readable mode, a symlink:
     * each refused. */
    assert(link(bin, hard) == 0);
    assert(elpis_hgram_probe(bin, &b) == ELPIS_HGRAM_INVALID);
    assert(unlink(hard) == 0);
    assert(elpis_hgram_probe(bin, &b) == ELPIS_HGRAM_EXISTS);
    int original = open(bin, O_RDONLY);
    char header[4096];
    assert(original >= 0 && read(original, header, sizeof header) == sizeof header && close(original) == 0);
    int fake = open(sparse, O_CREAT | O_EXCL | O_WRONLY, 0600);
    assert(fake >= 0 && write(fake, header, sizeof header) == sizeof header);
    assert(ftruncate(fake, 1048576) == 0 && close(fake) == 0);
    assert(elpis_hgram_probe(sparse, &b) == ELPIS_HGRAM_INVALID);      /* equal length is not preallocation */
    assert(hgram_fixture_create(foreign, 1, 0640) == 0);
    assert(elpis_hgram_probe(foreign, &b) == ELPIS_HGRAM_INVALID);
    assert(symlink(bin, sym) == 0);
    assert(elpis_hgram_probe(sym, &b) == ELPIS_HGRAM_INVALID);
    assert(hgram_fixture_flip(bin, 100) == 0);
    assert(elpis_hgram_probe(bin, &b) == ELPIS_HGRAM_INVALID);
    assert(hgram_fixture_flip(bin, 100) == 0);                            /* restored */
    assert(elpis_hgram_probe(bin, &b) == ELPIS_HGRAM_EXISTS);

    assert(entries(dir) == 5);
    assert(unlink(sym) == 0 && unlink(foreign) == 0 && unlink(sparse) == 0 && unlink(ro) == 0 && unlink(bin) == 0);
    assert(rmdir(dir) == 0);
    puts("ELPIS_HGRAM_READ_ONLY_PROBE_PASS");
    return 0;
}
