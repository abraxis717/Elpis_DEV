#include "dsv41_clock_materializer.h"
#include <string.h>
static elpis_clock_code start(clock_test_materializer *m) {
    m->calls++;
    if (m->cancel_clock && m->cancel && m->calls == m->fault_call) m->cancel(m->cancel_clock);
    if (m->calls == m->fault_call || (m->fault_repeat && m->calls >= m->fault_call))
        return (elpis_clock_code)m->fault_code;
    return m->live ? ELPIS_CLOCK_BUSY : ELPIS_CLOCK_OK;
}
static elpis_clock_code rows(void *ctx, uint32_t layer, const uint8_t bank[32], const uint64_t *ids,
                              size_t count, uint32_t dim, elpis_clock_span *out) {
    clock_test_materializer *m = ctx;
    elpis_clock_code rc = start(m);
    if (rc != ELPIS_CLOCK_OK) return rc;
    if (!dim || count > m->scratch_bytes / 4 / dim) return ELPIS_CLOCK_LIMIT;
    for (uint32_t i = 0; i < m->bank_count; ++i) if (m->banks[i].layer == layer) {
        const clock_test_bank *b = m->banks + i;
        if (b->dimension != dim || memcmp(b->bank, bank, 32)) return ELPIS_CLOCK_INTEGRITY;
        for (size_t j = 0; j < count; ++j) {
            if (ids[j] >= b->rows) return ELPIS_CLOCK_INTEGRITY;
            memcpy(m->scratch + j * dim * 4, b->values + ids[j] * dim * 4, dim * 4);
        }
        *out = (elpis_clock_span){.data=m->scratch, .bytes=count * dim * 4, .lease=m};
        m->live++; m->acquires++; return ELPIS_CLOCK_OK;
    }
    return ELPIS_CLOCK_INTEGRITY;
}
static elpis_clock_code expert(void *ctx, uint32_t layer, uint32_t expert_id, uint64_t offset,
                               size_t bytes, elpis_clock_span *out) {
    clock_test_materializer *m = ctx;
    elpis_clock_code rc = start(m);
    if (rc != ELPIS_CLOCK_OK) return rc;
    for (uint32_t i = 0; i < m->expert_count; ++i) {
        const clock_test_expert *e = m->experts + i;
        if (e->layer != layer || e->expert != expert_id) continue;
        if (offset > e->bytes || bytes > e->bytes - offset) return ELPIS_CLOCK_INTEGRITY;
        *out = (elpis_clock_span){.data=e->image + offset, .bytes=bytes, .lease=m};
        memcpy(out->digests, e->digests, 96);
        m->live++; m->acquires++; return ELPIS_CLOCK_OK;
    }
    return ELPIS_CLOCK_INTEGRITY;
}
static void release(void *ctx, elpis_clock_span *span) {
    clock_test_materializer *m = ctx;
    if (span->lease == m && m->live) { m->live--; m->releases++; }
    memset(span, 0, sizeof(*span));
}
static void quiesce(void *ctx) { ((clock_test_materializer *)ctx)->quiesces++; }
void elpis_dsv41_clock_test_materializer(clock_test_materializer *m, elpis_dsv41_materializer_v1 *out) {
    *out = (elpis_dsv41_materializer_v1){1, 0, m, rows, expert, release, quiesce};
}
