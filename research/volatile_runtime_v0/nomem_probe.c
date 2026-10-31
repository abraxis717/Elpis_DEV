/*
 * research/volatile_runtime_v0: memory pressure never becomes storage (docs/ELPIS_MISSION.md, Amendment 3: bounded,
 * no spill; docs/VOLATILE_RUNTIME_CENSUS.md).
 *
 * Under an address-space limit of the probe's current mapping plus a fixed margin, it creates sessions of the
 * largest capacity (4096 rows, elpis_ecs_c_session_bytes(4096) bytes each) until creation is refused. It requires:
 *   - the refusal is ELPIS_ECS_C_NOMEM: never an abort, and never a partial session (the out handle stays NULL);
 *   - every session created before the refusal still accepts a transition (it founds one source) and still reports
 *     its exact, constant footprint;
 *   - after half of the sessions are destroyed, creation succeeds again.
 * The harness (zero_write.py) runs it under strace: memory exhaustion must cause no write-class system call on any
 * file. The probe itself opens no file; it reads its mapping size from /proc/self/statm, which is a read.
 */
#define _GNU_SOURCE 1
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/random.h>
#include <sys/resource.h>
#include <unistd.h>

#include "elpis/ecs_c_session.h"

#define CHECK(expr)                                                                                             \
    do {                                                                                                        \
        if (!(expr)) {                                                                                          \
            fprintf(stderr, "%s:%d: CHECK failed: %s\n", __FILE__, __LINE__, #expr);                            \
            exit(1);                                                                                            \
        }                                                                                                       \
    } while (0)
#define OK(expr) CHECK((expr) == ELPIS_ECS_C_OK)

enum { CAPACITY = 4096, MAX_SESSIONS = 4096 };
static const unsigned long MARGIN = 48ul << 20; /* bytes of address space above the probe's own mapping */

static void put_u64(uint8_t *p, uint64_t v) {
    for (int i = 0; i < 8; i++) p[i] = (uint8_t)(v >> (8 * i));
}
static void put_u32(uint8_t *p, uint32_t v) {
    for (int i = 0; i < 4; i++) p[i] = (uint8_t)(v >> (8 * i));
}

static int32_t create(elpis_ecs_c_session **out) {
    elpis_ecs_c_session_config c;
    memset(&c, 0, sizeof(c));
    c.abi_version = ELPIS_ECS_C_SESSION_ABI_VERSION;
    c.capacity = CAPACITY;
    CHECK(getrandom(c.nonce, sizeof(c.nonce), 0) == (ssize_t)sizeof(c.nonce));
    memset(c.operator_label, 1, 32);
    memset(c.runtime_label, 2, 32);
    return elpis_ecs_c_session_create(&c, out);
}

/* The founding operator founds one source (ENTITY_FOUND, kind 2): a real, accepted transition. */
static void found_source(elpis_ecs_c_session *s) {
    elpis_ecs_c_entity opr;
    OK(elpis_ecs_c_session_entity_at(s, 0, &opr));
    CHECK(opr.kind == ELPIS_ECS_C_OPERATOR);
    uint8_t label[32], id[32], e[512];
    memset(label, 7, 32);
    OK(elpis_ecs_c_session_next_entity_id(s, ELPIS_ECS_C_SOURCE, label, id));
    memset(e, 0, sizeof(e));
    e[0] = 2;
    memcpy(e + 8, opr.id, 32);
    put_u64(e + 40, opr.sender_watermark + 1);
    memcpy(e + 48, id, 32);
    e[80] = ELPIS_ECS_C_SOURCE;
    put_u32(e + 84, 8);
    put_u64(e + 88, 1);
    memcpy(e + 96, label, 32);
    memset(e + 128, 3, 32);
    put_u32(e + 160, 1);
    elpis_ecs_c_session_coordinate c;
    OK(elpis_ecs_c_session_read_coordinate(s, &c));
    elpis_ecs_c_session_receipt r;
    OK(elpis_ecs_c_session_transition(s, &c, e, &r));
    CHECK(r.generation == c.generation + 1 && r.founded_present);
}

static unsigned long mapped_bytes(void) {
    FILE *f = fopen("/proc/self/statm", "r"); /* a read of the probe's own mapping size */
    CHECK(f != NULL);
    unsigned long pages = 0;
    CHECK(fscanf(f, "%lu", &pages) == 1);
    fclose(f);
    return pages * (unsigned long)sysconf(_SC_PAGESIZE);
}

int main(void) {
    static elpis_ecs_c_session *held[MAX_SESSIONS];
    const uint64_t each = elpis_ecs_c_session_bytes(CAPACITY);
    CHECK(each > 0);
    const unsigned long limit = mapped_bytes() + MARGIN;
    const struct rlimit as = {limit, limit};
    CHECK(setrlimit(RLIMIT_AS, &as) == 0);

    size_t n = 0;
    int32_t refused = ELPIS_ECS_C_OK;
    for (;;) {
        CHECK(n < MAX_SESSIONS);
        elpis_ecs_c_session *s = NULL;
        refused = create(&s);
        if (refused != ELPIS_ECS_C_OK) {
            CHECK(s == NULL); /* nothing was created */
            break;
        }
        held[n++] = s;
    }
    CHECK(refused == ELPIS_ECS_C_NOMEM);
    CHECK(n >= 2);
    for (size_t i = 0; i < n; i++) {
        elpis_ecs_c_session_status st;
        OK(elpis_ecs_c_session_read_status(held[i], &st));
        CHECK(st.volatile_bytes == each && st.capacity == CAPACITY && st.generation == 1);
        found_source(held[i]);
        OK(elpis_ecs_c_session_read_status(held[i], &st));
        CHECK(st.volatile_bytes == each && st.generation == 2 && st.entity_count == 3);
    }
    const size_t kept = n / 2;
    for (size_t i = kept; i < n; i++) elpis_ecs_c_session_destroy(held[i]);
    elpis_ecs_c_session *again = NULL;
    OK(create(&again));
    found_source(again);
    elpis_ecs_c_session_destroy(again);
    for (size_t i = 0; i < kept; i++) elpis_ecs_c_session_destroy(held[i]);
    printf("nomem_probe: RLIMIT_AS = mapping + %lu MiB; %zu sessions of capacity %u (%llu bytes each) created, the "
           "next refused ECS_C_NOMEM with nothing created; all %zu then transitioned with a constant footprint; after "
           "destroying %zu, creation succeeded again\n",
           MARGIN >> 20, n, (unsigned)CAPACITY, (unsigned long long)each, n, n - kept);
    printf("nomem_probe: PASS\n");
    return 0;
}
