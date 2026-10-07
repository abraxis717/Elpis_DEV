/* The continuity C ABI against the frozen v2 vectors, through the production library only.
 *
 * Usage: test_continuity_abi <fixture-dir> <scratch-dir>
 * Deterministic: no threads, no sleeps. */
#define _POSIX_C_SOURCE 200809L
#include "elpis/continuity.h"

#include <assert.h>
#include <dirent.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <unistd.h>

#define OK(x) assert((x) == ELPIS_CONTINUITY_OK)

_Static_assert(sizeof(elpis_continuity_evolution) == 80, "evolution ABI size");
_Static_assert(sizeof(elpis_continuity_snapshot) == 192, "snapshot ABI size");

/* dir + "/" + name into out; the result must fit (checked, never silently truncated). */
static void join(char *out, size_t size, const char *dir, const char *name) {
    int n = snprintf(out, size, "%s/%s", dir, name);
    assert(n > 0 && (size_t)n < size);
}

static void unhex(const char *s, uint8_t *out, size_t n) {
    assert(strlen(s) == 2 * n);
    for (size_t i = 0; i < n; ++i) assert(sscanf(s + 2 * i, "%2hhx", &out[i]) == 1);
}

static void records(const char *fixtures) {
    char path[4096];
    join(path, sizeof(path), fixtures, "continuity_v2_vectors.txt");
    FILE *f = fopen(path, "r");
    assert(f);
    char line[4096];
    unsigned ok = 0, empty = 0, corrupt = 0;
    while (fgets(line, sizeof(line), f)) {
        if (line[0] == '#' || line[0] == '\n') continue;
        char name[128], expect[16], gen[32], anch[8], k1[80], rev[32], head[80], pend[8], asrt[80], evd[80], recd[80],
            raw[1024];
        assert(sscanf(line, "%127s %15s %31s %7s %79s %31s %79s %7s %79s %79s %79s %1023s", name, expect, gen, anch, k1,
                      rev, head, pend, asrt, evd, recd, raw) == 12);
        size_t len = strlen(raw) / 2;
        uint8_t bytes[512];
        unhex(raw, bytes, len);
        elpis_continuity_snapshot s;
        memset(&s, 0xa5, sizeof(s));
        int is_empty = -1;
        int rc = elpis_continuity_record_decode(bytes, len, &s, &is_empty);
        if (!strcmp(expect, "corrupt")) {
            assert(rc == ELPIS_CONTINUITY_CORRUPT);
            ++corrupt;
            continue;
        }
        OK(rc);
        if (!strcmp(expect, "empty")) {
            assert(is_empty == 1);
            ++empty;
            continue;
        }
        assert(is_empty == 0);
        uint8_t want[32];
        assert(s.generation == strtoull(gen, NULL, 10));
        assert(s.anchored == (uint8_t)atoi(anch));
        unhex(k1, want, 32), assert(!memcmp(s.k1_state_digest, want, 32));
        assert(s.evolution.revision == strtoull(rev, NULL, 10));
        unhex(head, want, 32), assert(!memcmp(s.evolution.head, want, 32));
        assert(s.evolution.pending == (uint8_t)atoi(pend));
        unhex(asrt, want, 32), assert(!memcmp(s.evolution.assertion, want, 32));
        unhex(evd, want, 32), assert(!memcmp(s.evolution_digest, want, 32));
        unhex(recd, want, 32), assert(!memcmp(s.record_digest, want, 32));
        uint8_t digest[32], encoded[ELPIS_CONTINUITY_RECORD_SIZE];
        OK(elpis_continuity_evolution_digest(&s.evolution, digest));
        assert(!memcmp(digest, s.evolution_digest, 32));
        OK(elpis_continuity_record_encode(&s, encoded));
        assert(len == ELPIS_CONTINUITY_RECORD_SIZE && !memcmp(encoded, bytes, len));
        ++ok;
    }
    fclose(f);
    assert(ok == 9 && empty == 1 && corrupt == 19);
    printf("frozen v2 records: %u ok, %u empty, %u corrupt PASS\n", ok, empty, corrupt);
}

static void remove_tree(const char *dir) {
    DIR *d = opendir(dir);
    if (!d) return;
    struct dirent *e;
    char p[4096];
    while ((e = readdir(d))) {
        if (!strcmp(e->d_name, ".") || !strcmp(e->d_name, "..")) continue;
        join(p, sizeof(p), dir, e->d_name);
        unlink(p);
    }
    closedir(d);
    rmdir(dir);
}

static void slot(const char *dir, const char *name, char *hex_out) {
    char p[4096];
    join(p, sizeof(p), dir, name);
    FILE *f = fopen(p, "rb");
    assert(f);
    uint8_t b[ELPIS_CONTINUITY_RECORD_SIZE];
    assert(fread(b, 1, sizeof(b), f) == sizeof(b) && fgetc(f) == EOF);
    fclose(f);
    for (size_t i = 0; i < sizeof(b); ++i) sprintf(hex_out + 2 * i, "%02x", b[i]);
}

static void session(const char *fixtures, const char *scratch) {
    char dir[4096], path[4096], line[4096];
    join(dir, sizeof(dir), scratch, "abi-session");
    remove_tree(dir);
    join(path, sizeof(path), fixtures, "continuity_v2_sequence.txt");
    FILE *f = fopen(path, "r");
    assert(f);
    elpis_continuity_store *s = NULL;
    OK(elpis_continuity_store_create((const uint8_t *)dir, strlen(dir), &s));
    unsigned steps = 0;
    while (fgets(line, sizeof(line), f)) {
        if (line[0] == '#' || line[0] == '\n') continue;
        char step[16], op[16], a1[80], a2[80], sa[512], sb[512];
        assert(sscanf(line, "%15s %15s %79s %79s %511s %511s", step, op, a1, a2, sa, sb) == 6);
        uint8_t x[32], y[32];
        elpis_continuity_snapshot cur;
        if (!strcmp(op, "open")) {
            OK(elpis_continuity_store_open(s, NULL));
        } else if (!strcmp(op, "reopen")) {
            elpis_continuity_store_close(s);
            OK(elpis_continuity_store_open(s, NULL));
        } else if (!strcmp(op, "anchor")) {
            unhex(a1, x, 32);
            OK(elpis_continuity_anchor_cognition(s, x, NULL));
        } else if (!strcmp(op, "commit")) {
            unhex(a1, x, 32), unhex(a2, y, 32);
            OK(elpis_continuity_commit_cognition(s, x, y, NULL));
        } else if (!strcmp(op, "reserve")) {
            unhex(a1, x, 32);
            OK(elpis_continuity_store_snapshot(s, &cur));
            OK(elpis_continuity_reserve_evolution(s, &cur.evolution, x, NULL));
        } else {
            assert(!strcmp(op, "finalize"));
            unhex(a1, x, 32);
            OK(elpis_continuity_store_snapshot(s, &cur));
            OK(elpis_continuity_finalize_evolution(s, &cur.evolution, x, NULL));
        }
        char ha[2 * ELPIS_CONTINUITY_RECORD_SIZE + 1], hb[sizeof(ha)];
        slot(dir, "continuity.a", ha), slot(dir, "continuity.b", hb);
        assert(!strcmp(ha, sa) && !strcmp(hb, sb));
        ++steps;
    }
    fclose(f);
    assert(steps == 10);
    elpis_continuity_store_destroy(&s);
    assert(s == NULL);
    remove_tree(dir);
    printf("frozen v2 store session: %u steps, both slots byte-identical PASS\n", steps);
}

static void contract(const char *scratch) {
    char dir[4096];
    join(dir, sizeof(dir), scratch, "abi-contract");
    remove_tree(dir);
    assert(elpis_continuity_abi_version() == ELPIS_CONTINUITY_ABI_V1);
    assert(elpis_continuity_record_size() == ELPIS_CONTINUITY_RECORD_SIZE);
    assert(!strcmp(elpis_continuity_code_name(0), "CONTINUITY_OK"));
    assert(!strcmp(elpis_continuity_code_name(ELPIS_CONTINUITY_PUBLICATION_UNCERTAIN),
                   "CONTINUITY_PUBLICATION_UNCERTAIN"));
    assert(elpis_continuity_code_name(99) == NULL);

    elpis_continuity_store *s = NULL, *other = NULL;
    assert(elpis_continuity_store_create((const uint8_t *)"relative", 8, &s) == ELPIS_CONTINUITY_PATH && !s);
    OK(elpis_continuity_store_create((const uint8_t *)dir, strlen(dir), &s));
    elpis_continuity_snapshot snap, before;
    assert(elpis_continuity_store_snapshot(s, &snap) == ELPIS_CONTINUITY_UNINITIALIZED);
    OK(elpis_continuity_store_open(s, &snap));
    assert(snap.generation == 1 && !snap.anchored && !snap.evolution.pending && !snap.evolution.revision);
    assert(elpis_continuity_store_open(s, NULL) == ELPIS_CONTINUITY_OPEN);
    OK(elpis_continuity_store_create((const uint8_t *)dir, strlen(dir), &other));
    assert(elpis_continuity_store_open(other, NULL) == ELPIS_CONTINUITY_LOCKED);

    uint8_t k0[32] = {1}, k1[32] = {2}, a[32] = {3}, r[32] = {4};
    assert(elpis_continuity_commit_cognition(s, k0, k1, NULL) == ELPIS_CONTINUITY_UNANCHORED);
    assert(elpis_continuity_anchor_cognition(s, NULL, NULL) == ELPIS_CONTINUITY_INVALID);
    OK(elpis_continuity_anchor_cognition(s, k0, &snap));
    assert(snap.anchored && !memcmp(snap.k1_state_digest, k0, 32) && snap.generation == 2);
    assert(elpis_continuity_commit_cognition(s, k1, k0, NULL) == ELPIS_CONTINUITY_STATE_MISMATCH);

    /* An idle expectation carrying an assertion is malformed: it can never be the authority. */
    elpis_continuity_evolution forged = snap.evolution;
    forged.assertion[0] = 9;
    assert(elpis_continuity_reserve_evolution(s, &forged, a, NULL) == ELPIS_CONTINUITY_AUTHORITY_MISMATCH);
    uint8_t digest[32];
    assert(elpis_continuity_evolution_digest(&forged, digest) == ELPIS_CONTINUITY_CORRUPT);
    assert(elpis_continuity_reserve_evolution(s, NULL, a, NULL) == ELPIS_CONTINUITY_AUTHORITY_MISMATCH);
    assert(elpis_continuity_finalize_evolution(s, &snap.evolution, r, NULL) == ELPIS_CONTINUITY_EVOLUTION_NOT_PENDING);

    before = snap;
    OK(elpis_continuity_reserve_evolution(s, &before.evolution, a, &snap));
    assert(snap.evolution.pending == 1 && !memcmp(snap.evolution.assertion, a, 32) && snap.evolution.revision == 0);
    assert(memcmp(snap.evolution_digest, before.evolution_digest, 32));
    assert(elpis_continuity_reserve_evolution(s, &snap.evolution, a, NULL) == ELPIS_CONTINUITY_EVOLUTION_PENDING);
    uint8_t zero[32] = {0};
    assert(elpis_continuity_finalize_evolution(s, &snap.evolution, zero, NULL) == ELPIS_CONTINUITY_INVALID);
    elpis_continuity_snapshot pending = snap;
    OK(elpis_continuity_finalize_evolution(s, &pending.evolution, r, &snap));
    assert(!snap.evolution.pending && snap.evolution.revision == 1 && !memcmp(snap.evolution.head, r, 32));
    assert(elpis_continuity_finalize_evolution(s, &pending.evolution, r, NULL) == ELPIS_CONTINUITY_AUTHORITY_MISMATCH);

    elpis_continuity_store_close(s);
    OK(elpis_continuity_store_open(other, &before)); /* the lock was released */
    assert(!memcmp(before.record_digest, snap.record_digest, 32));
    elpis_continuity_store_destroy(&other);
    elpis_continuity_store_destroy(&s);
    elpis_continuity_store_destroy(&s); /* idempotent on NULL */
    struct stat st;
    char p[4096];
    join(p, sizeof(p), dir, "continuity.a");
    assert(!stat(p, &st) && st.st_size == ELPIS_CONTINUITY_RECORD_SIZE);
    remove_tree(dir);
    puts("C ABI contract: codes, lifecycle, lock, cognition, reservation and finalization PASS");
}

int main(int argc, char **argv) {
    assert(argc == 3);
    mkdir(argv[2], 0700);
    records(argv[1]);
    session(argv[1], argv[2]);
    contract(argv[2]);
    return 0;
}
