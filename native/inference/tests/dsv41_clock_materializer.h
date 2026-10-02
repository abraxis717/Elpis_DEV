#ifndef DSV41_CLOCK_TEST_MATERIALIZER_H
#define DSV41_CLOCK_TEST_MATERIALIZER_H
#include "elpis/dsv41_clock.h"
/* TEST ONLY. Cold-admitted decoded banks and canonical expert images. No FMS
 * adapter, host cache or production residency assumption is introduced. The
 * provider still receives experts in bounded parts and uses one image slot. */
typedef struct {
    uint32_t layer, dimension;
    uint64_t rows;
    uint8_t bank[32];
    const uint8_t *values;
} clock_test_bank;
typedef struct {
    uint32_t layer, expert;
    uint64_t bytes;
    uint8_t digests[96];
    const uint8_t *image;
} clock_test_expert;
typedef struct {
    const clock_test_bank *banks;
    const clock_test_expert *experts;
    uint32_t bank_count, expert_count;
    uint8_t *scratch;
    size_t scratch_bytes;
    uint64_t calls, acquires, releases, live, quiesces;
    uint64_t fault_call;
    uint32_t fault_code, fault_repeat;
    elpis_dsv41_clock cancel_clock;
    elpis_clock_code (*cancel)(elpis_dsv41_clock);
} clock_test_materializer;
void elpis_dsv41_clock_test_materializer(clock_test_materializer *, elpis_dsv41_materializer_v1 *);
#endif
