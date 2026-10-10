/* test_fms_cold_reclaim.c - finite physical envelope of the POSIX cold store.
 * One live owner per cold root; a crashed owner's files (undropped blobs and
 * interrupted-commit temporaries) are reclaimed by the next owner; files the
 * PAL does not name are never touched; put/drop cycles cannot accumulate. */
#define _POSIX_C_SOURCE 200809L

#include "elpis/fms_pal.h"
#include "elpis/fms_pal_posix.h"

#include <dirent.h>
#include <fcntl.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <sys/wait.h>
#include <unistd.h>

#define CHECK(c) do { if (!(c)) { fprintf(stderr, "FAIL_FMS_COLD_RECLAIM line=%d %s\n", __LINE__, #c); exit(1); } } while (0)

static unsigned entries(const char *root, unsigned *blobs, unsigned *temps) {
    DIR *d = opendir(root); CHECK(d);
    unsigned n = 0; *blobs = 0; *temps = 0;
    struct dirent *e;
    while ((e = readdir(d)) != NULL) {
        if (!strcmp(e->d_name, ".") || !strcmp(e->d_name, "..")) continue;
        n++;
        size_t len = strlen(e->d_name);
        if (len > 5 && !strcmp(e->d_name + len - 5, ".blob")) (*blobs)++;
        if (!strncmp(e->d_name, ".fms-tmp-", 9)) (*temps)++;
    }
    closedir(d);
    return n;
}
static void touch(const char *root, const char *leaf, int dir) {
    char p[1024]; snprintf(p, sizeof p, "%s/%s", root, leaf);
    if (dir) { CHECK(mkdir(p, 0700) == 0); return; }
    int fd = open(p, O_WRONLY | O_CREAT | O_EXCL, 0600); CHECK(fd >= 0);
    CHECK(write(fd, "x", 1) == 1); CHECK(close(fd) == 0);
}

int main(int argc, char **argv) {
    CHECK(argc == 2);
    char root[900]; snprintf(root, sizeof root, "%s/cold", argv[1]);
    char cmd[2048]; snprintf(cmd, sizeof cmd, "rm -rf '%s'", argv[1]);
    CHECK(system(cmd) == 0);
    CHECK(mkdir(argv[1], 0700) == 0);
    unsigned blobs, temps;
    const char payload[] = "cold object";

    /* A crashed owner: two committed, never-dropped objects, then death. */
    pid_t child = fork(); CHECK(child >= 0);
    if (child == 0) {
        fms_pal *p = fms_pal_posix_create(root);
        if (!p) _exit(2);
        fms_cold_token *a = NULL, *b = NULL;
        if (p->cold_put(p->self, payload, sizeof payload, &a) != FMS_PAL_OK) _exit(3);
        if (p->cold_put(p->self, payload, sizeof payload, &b) != FMS_PAL_OK) _exit(4);
        _exit(0);
    }
    int status = 0; CHECK(waitpid(child, &status, 0) == child);
    CHECK(WIFEXITED(status) && WEXITSTATUS(status) == 0);
    touch(root, ".fms-tmp-AbC123", 0);                 /* interrupted commit */
    touch(root, "operator-note.txt", 0);               /* not PAL-named: kept */
    touch(root, "0000000000000000000000000000000000000000000000000000000000000000.7.blob.d", 1);
    CHECK(entries(root, &blobs, &temps) == 5 && blobs == 2 && temps == 1);

    /* The next owner reclaims exactly the orphans it names. */
    fms_pal *p = fms_pal_posix_create(root); CHECK(p);
    CHECK(entries(root, &blobs, &temps) == 2 && blobs == 0 && temps == 0);

    /* Exactly one live owner per cold root. */
    CHECK(fms_pal_posix_create(root) == NULL);
    CHECK(entries(root, &blobs, &temps) == 2);

    /* Put/drop cycles cannot accumulate files. */
    for (int i = 0; i < 64; ++i) {
        fms_cold_token *t = NULL;
        CHECK(p->cold_put(p->self, payload, sizeof payload, &t) == FMS_PAL_OK);
        CHECK(entries(root, &blobs, &temps) == 3 && blobs == 1 && temps == 0);
        p->cold_drop(p->self, t);
        CHECK(entries(root, &blobs, &temps) == 2 && blobs == 0);
    }
    p->destroy(p->self);

    /* Ownership is released with the PAL. */
    p = fms_pal_posix_create(root); CHECK(p);
    p->destroy(p->self);
    CHECK(entries(root, &blobs, &temps) == 2 && blobs == 0 && temps == 0);
    CHECK(system(cmd) == 0);
    puts("PASS_FMS_COLD_ROOT_FINITE_ENVELOPE");
    return 0;
}
