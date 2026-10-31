/* RuntimeCore's C tests over the volatile ECS_C session (ABI v8): no file, directory or provisioning exists.
 *
 * test_runtime_create(&rt) creates a closed runtime: sessions of capacity 8 with a fixed nonce seed (reproducible).
 * test_runtime_open(rt, &snap) opens a new session and, as its operator, founds one ACTIVE source for each memory
 * source authority 0, 1 and 2 (what these tests' typed observations record) at a per-turn bound of 256, through
 * elpis_runtime_authority_transition, so admission never decides what a test measures. Every open is a new session:
 * the sources are founded again each time. */
#ifndef ELPIS_RUNTIME_TESTS_SESSION_RUNTIME_H
#define ELPIS_RUNTIME_TESTS_SESSION_RUNTIME_H

#include "elpis/ecs_c_types.h"
#include "elpis/runtime.h"

#include <assert.h>
#include <string.h>

enum { TEST_RUNTIME_CAPACITY = 8, TEST_RUNTIME_SOURCES = 3, TEST_RUNTIME_SOURCE_BOUND = 256 };

static inline void test_runtime_put_u64(uint8_t *p, uint64_t v) {
    for (int i = 0; i < 8; i++) p[i] = (uint8_t)(v >> (8 * i));
}

static inline void test_runtime_put_u32(uint8_t *p, uint32_t v) {
    for (int i = 0; i < 4; i++) p[i] = (uint8_t)(v >> (8 * i));
}

static inline elpis_runtime_config test_runtime_config(uint8_t seed) {
    elpis_runtime_config c;
    memset(&c, 0, sizeof c);
    c.abi_version = ELPIS_RUNTIME_ABI_V8;
    c.capacity = TEST_RUNTIME_CAPACITY;
    memset(c.nonce, seed, 32);
    memset(c.operator_label, 1, 32);
    memset(c.runtime_label, 2, 32);
    return c;
}

static inline int test_runtime_create(elpis_runtime **out) {
    const elpis_runtime_config c = test_runtime_config(0x5e);
    return elpis_runtime_create(&c, out);
}

/* One operator event (elpis/ecs_c_types.h), from the session's founding operator at its next sequence. */
static inline int test_runtime_operator(elpis_runtime *rt, uint16_t kind, const uint8_t id[32], uint8_t e[512]) {
    elpis_runtime_authority_entity op;
    assert(elpis_runtime_authority_entity_at(rt, 0, &op) == ELPIS_RUNTIME_OK); /* row 0: the operator */
    e[0] = (uint8_t)kind;
    e[1] = (uint8_t)(kind >> 8);
    memcpy(e + 8, op.id, 32);
    test_runtime_put_u64(e + 40, op.sender_watermark + 1);
    memcpy(e + 48, id, 32);
    return elpis_runtime_authority_transition(rt, e, NULL);
}

/* Found an ACTIVE source of `source_authority` at `bound`; its identity into `id`. */
static inline void test_runtime_found_source(elpis_runtime *rt, uint32_t source_authority, uint32_t bound, uint8_t id[32]) {
    uint8_t label[32], scope[32], e[512];
    memset(label, (int)(0x30 + source_authority), 32);
    memset(scope, 4, 32);
    assert(elpis_runtime_authority_next_entity_id(rt, ELPIS_ECS_C_SOURCE, label, id) == ELPIS_RUNTIME_OK);
    memset(e, 0, sizeof e); /* ENTITY_FOUND: kind, per-turn bound, version, label, scope, source authority */
    e[80] = ELPIS_ECS_C_SOURCE;
    test_runtime_put_u32(e + 80 + 4, bound);
    test_runtime_put_u64(e + 80 + 8, 1);
    memcpy(e + 80 + 16, label, 32);
    memcpy(e + 80 + 48, scope, 32);
    test_runtime_put_u32(e + 80 + 80, source_authority);
    assert(test_runtime_operator(rt, 2, id, e) == ELPIS_RUNTIME_OK);
    memset(e, 0, sizeof e); /* ENTITY_LIFECYCLE: FOUNDED -> ACTIVE */
    e[80] = ELPIS_ECS_C_FOUNDED;
    e[81] = ELPIS_ECS_C_ACTIVE;
    assert(test_runtime_operator(rt, 3, id, e) == ELPIS_RUNTIME_OK);
}

static inline int test_runtime_open(elpis_runtime *rt, elpis_continuity_snapshot *out) {
    const int rc = elpis_runtime_open(rt, out);
    if (rc != ELPIS_RUNTIME_OK) return rc;
    for (uint32_t source = 0; source < TEST_RUNTIME_SOURCES; ++source) {
        uint8_t id[32];
        test_runtime_found_source(rt, source, TEST_RUNTIME_SOURCE_BOUND, id);
    }
    return out ? elpis_runtime_snapshot(rt, out) : ELPIS_RUNTIME_OK;
}

#endif
