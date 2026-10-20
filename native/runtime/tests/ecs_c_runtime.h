/* RuntimeCore's C tests over a provisioned ECS_C authority (elpis/ecs_c_authority.h), the only durable authority
 * since ABI v7.
 *
 * test_runtime_create(dir, &rt) creates a closed runtime over the authority under `dir` (`dir`/state and the
 * protected `dir`/authority), provisioned by the operator's own ABI on first use: capacity 8, and one ACTIVE source
 * for each memory source authority 0, 1 and 2 (what these tests' typed observations record) at a per-turn bound of
 * 256, so admission never decides what a test measures. A second call over the same `dir` is a restart: it opens the
 * same authority. */
#ifndef ELPIS_RUNTIME_TESTS_ECS_C_RUNTIME_H
#define ELPIS_RUNTIME_TESTS_ECS_C_RUNTIME_H

#include "elpis/ecs_c_authority.h"
#include "elpis/runtime.h"

#include <assert.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

enum { TEST_RUNTIME_CAPACITY = 8, TEST_RUNTIME_SOURCES = 3, TEST_RUNTIME_SOURCE_BOUND = 256 };

static void test_runtime_put_u64(uint8_t *p, uint64_t v) {
    for (int i = 0; i < 8; i++) p[i] = (uint8_t)(v >> (8 * i));
}

static void test_runtime_put_u32(uint8_t *p, uint32_t v) {
    for (int i = 0; i < 4; i++) p[i] = (uint8_t)(v >> (8 * i));
}

/* One operator event (elpis/ecs_c_authority.h, "Events"), sent at the authority's current generation and root. */
static void test_runtime_send(elpis_ecs_c_authority *a, uint16_t kind, const uint8_t op[32], const uint8_t id[32],
                              uint8_t e[512]) {
    elpis_ecs_c_entity sender;
    elpis_ecs_c_status s;
    assert(elpis_ecs_c_entity_find(a, op, &sender) == ELPIS_ECS_C_OK);
    e[0] = (uint8_t)kind;
    e[1] = (uint8_t)(kind >> 8);
    memcpy(e + 8, op, 32);
    test_runtime_put_u64(e + 40, sender.sender_watermark + 1);
    memcpy(e + 48, id, 32);
    assert(elpis_ecs_c_read_status(a, &s) == ELPIS_ECS_C_OK);
    assert(elpis_ecs_c_transition(a, s.generation, s.state_root, e, NULL) == ELPIS_ECS_C_OK);
}

static void test_runtime_provision(const char *state, const char *authority) {
    uint8_t op_label[32], rt_label[32], op[32];
    memset(op_label, 1, 32);
    memset(rt_label, 2, 32);
    assert(elpis_ecs_c_provision(state, strlen(state), authority, strlen(authority), TEST_RUNTIME_CAPACITY, op_label,
                                 rt_label, NULL) == ELPIS_ECS_C_OK);
    elpis_ecs_c_authority *a = NULL;
    assert(elpis_ecs_c_open(state, strlen(state), authority, strlen(authority), &a, NULL) == ELPIS_ECS_C_OK);
    elpis_ecs_c_entity row;
    assert(elpis_ecs_c_entity_at(a, 0, &row) == ELPIS_ECS_C_OK); /* row 0: the operator */
    memcpy(op, row.id, 32);
    for (uint32_t source = 0; source < TEST_RUNTIME_SOURCES; ++source) {
        uint8_t label[32], scope[32], id[32], e[512];
        memset(label, (int)(0x30 + source), 32);
        memset(scope, 4, 32);
        assert(elpis_ecs_c_next_entity_id(a, ELPIS_ECS_C_SOURCE, label, id) == ELPIS_ECS_C_OK);
        memset(e, 0, sizeof e); /* ENTITY_FOUND: kind, per-turn bound, version, label, scope, source authority */
        e[80] = ELPIS_ECS_C_SOURCE;
        test_runtime_put_u32(e + 80 + 4, TEST_RUNTIME_SOURCE_BOUND);
        test_runtime_put_u64(e + 80 + 8, 1);
        memcpy(e + 80 + 16, label, 32);
        memcpy(e + 80 + 48, scope, 32);
        test_runtime_put_u32(e + 80 + 80, source);
        test_runtime_send(a, 2, op, id, e);
        memset(e, 0, sizeof e); /* ENTITY_LIFECYCLE: FOUNDED -> ACTIVE */
        e[80] = ELPIS_ECS_C_FOUNDED;
        e[81] = ELPIS_ECS_C_ACTIVE;
        test_runtime_send(a, 3, op, id, e);
    }
    elpis_ecs_c_close(a);
}

static int test_runtime_create(const char *dir, elpis_runtime **out) {
    char state[4200], authority[4200], witness[4300], cmd[4400];
    assert(snprintf(state, sizeof state, "%s/state", dir) < (int)sizeof state);
    assert(snprintf(authority, sizeof authority, "%s/authority", dir) < (int)sizeof authority);
    assert(snprintf(witness, sizeof witness, "%s/witness.a", authority) < (int)sizeof witness);
    if (access(witness, F_OK) != 0) {
        assert(snprintf(cmd, sizeof cmd, "mkdir -p '%s'", dir) < (int)sizeof cmd);
        assert(system(cmd) == 0);
        test_runtime_provision(state, authority);
    }
    return elpis_runtime_create_ecs_c((const uint8_t *)state, strlen(state), (const uint8_t *)authority,
                                      strlen(authority), out);
}

#endif
