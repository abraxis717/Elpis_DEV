/* test_structural_observation.c — read-only structural observation contract.
 *
 * Extracted from the retired integrated-spine suite: the observation record
 * maps a topology vertex through a Grid81 capsule to a cell transition and is
 * inherently read-only.
 */
#include "elpis_semantic/structural_observation.h"
#include <stdio.h>
#include <string.h>

static int tests_passed = 0;
static int tests_failed = 0;

#define ASSERT_EQ(a, b) do { \
    if ((a) != (b)) { \
        fprintf(stderr, "FAIL %s:%d %s != %s\n", __FILE__, __LINE__, #a, #b); \
        tests_failed++; return 0; \
    } \
} while(0)

#define ASSERT_TRUE(x) do { \
    if (!(x)) { \
        fprintf(stderr, "FAIL %s:%d %s\n", __FILE__, __LINE__, #x); \
        tests_failed++; return 0; \
    } \
} while(0)

#define TEST_PASS do { tests_passed++; return 1; } while(0)

static int test_observation_init(void) {
    elpis_semantic_structural_observation_v1 obs;
    elpis_spine_observation_init(&obs);
    ASSERT_EQ(obs.abi_version, SPINE_OBSERVATION_ABI_VERSION);
    TEST_PASS;
}

static int test_observation_validate(void) {
    elpis_semantic_structural_observation_v1 obs;
    elpis_spine_observation_init(&obs);
    obs.P7_primary_cell_index = 5;
    obs.initial_grid81_digit = 0;
    obs.final_grid81_digit = 7;
    ASSERT_EQ(elpis_spine_observation_validate(&obs), SEMANTIC_OK);
    TEST_PASS;
}

static int test_observation_readonly(void) {
    elpis_semantic_structural_observation_v1 obs;
    elpis_spine_observation_init(&obs);
    ASSERT_TRUE(elpis_spine_observation_is_readonly(&obs));
    TEST_PASS;
}

static int test_observation_invalid_cell(void) {
    elpis_semantic_structural_observation_v1 obs;
    elpis_spine_observation_init(&obs);
    obs.P7_primary_cell_index = 81; /* out of range */
    ASSERT_EQ(elpis_spine_observation_validate(&obs), SEMANTIC_E_INVAL);
    TEST_PASS;
}

static int test_observation_reserved_bytes_rejected(void) {
    elpis_semantic_structural_observation_v1 obs;
    elpis_spine_observation_init(&obs);
    obs.reserved[3] = 1;
    ASSERT_EQ(elpis_spine_observation_validate(&obs), SEMANTIC_E_RESERVATION);
    TEST_PASS;
}

static int test_observation_identity_is_deterministic_and_binding(void) {
    elpis_semantic_structural_observation_v1 a, b;
    hacf_digest da, db;
    elpis_spine_observation_init(&a);
    elpis_spine_observation_init(&b);
    a.final_grid81_digit = b.final_grid81_digit = 4;
    ASSERT_EQ(elpis_spine_observation_identity(&a, &da), SEMANTIC_OK);
    ASSERT_EQ(elpis_spine_observation_identity(&b, &db), SEMANTIC_OK);
    ASSERT_TRUE(memcmp(da.bytes, db.bytes, sizeof da.bytes) == 0);
    b.final_grid81_digit = 5;
    ASSERT_EQ(elpis_spine_observation_identity(&b, &db), SEMANTIC_OK);
    ASSERT_TRUE(memcmp(da.bytes, db.bytes, sizeof da.bytes) != 0);
    TEST_PASS;
}

int main(void) {
    test_observation_init();
    test_observation_validate();
    test_observation_readonly();
    test_observation_invalid_cell();
    test_observation_reserved_bytes_rejected();
    test_observation_identity_is_deterministic_and_binding();
    printf("structural observation: %d passed, %d failed\n", tests_passed, tests_failed);
    return tests_failed > 0 ? 1 : 0;
}
