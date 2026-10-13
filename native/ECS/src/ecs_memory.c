/* ECS memory (canonical ECS capability): bounded model-free typed relational memory. Contract: elpis/ecs_memory.h
 * and docs/ECS_MEMORY.md. Relocated from research/ecs_memory_mf_r0 (the frozen reference, differential-tested
 * against this file) with capability gating added and the reference's slow-bank law left out: the slow banks and
 * their cursor stay in the ELPISEM1 layout and must be empty. Integer arithmetic only; one allocation per handle;
 * no I/O. */
#include "elpis/ecs_memory.h"
#include "elpis/sha256.h"

#include <stdatomic.h>
#include <stdlib.h>
#include <string.h>

#define NONE UINT32_MAX
#define Q16 ((int64_t)ELPIS_ECS_MEM_Q16_ONE)

typedef struct {
    elpis_ecs_mem_digest id;
    uint64_t first_tick;
    uint64_t last_tick;
} entity_rec;

typedef struct {
    uint32_t cap, count, head, tail, mask;
    uint32_t *prev, *next, *index;
} table;

/* A view of 32-byte keys at a fixed stride (in-memory records and serialized records alike). */
typedef struct {
    const uint8_t *base;
    size_t stride;
} keyview;

struct elpis_ecs_mem {
    elpis_ecs_mem_config cfg;
    _Atomic int guard; /* 0 idle, n > 0 readers, -1 one writer */
    uint64_t lease;
    uint64_t clock;
    uint32_t words; /* D / 64 */
    int32_t *fast, *slow;
    elpis_ecs_mem_slot *slots;
    entity_rec *ents;
    table st, et;
    uint32_t *scratch_slot_index, *scratch_ent_index;
    uint8_t *scratch_marks;
    void *arena;
    uint64_t arena_bytes;
    elpis_ecs_mem_stats tel; /* telemetry fields only */
};

/* ------------------------------------------------------------------------------------------------------ */
/* Small helpers                                                                                          */
/* ------------------------------------------------------------------------------------------------------ */

static void le32(uint8_t *p, uint32_t v) {
    for (int i = 0; i < 4; ++i) p[i] = (uint8_t)(v >> (8 * i));
}
static void le64(uint8_t *p, uint64_t v) {
    for (int i = 0; i < 8; ++i) p[i] = (uint8_t)(v >> (8 * i));
}
static uint32_t rd32(const uint8_t *p) {
    uint32_t v = 0;
    for (int i = 3; i >= 0; --i) v = (v << 8) | p[i];
    return v;
}
static uint64_t rd64(const uint8_t *p) {
    uint64_t v = 0;
    for (int i = 7; i >= 0; --i) v = (v << 8) | p[i];
    return v;
}
static uint64_t mix64(uint64_t x) {
    x ^= x >> 30;
    x *= UINT64_C(0xbf58476d1ce4e5b9);
    x ^= x >> 27;
    x *= UINT64_C(0x94d049bb133111eb);
    return x ^ (x >> 31);
}
/* Division rounding half away from zero (d > 0). */
static int64_t rdiv(int64_t n, int64_t d) {
    return n >= 0 ? (n + d / 2) / d : -((-n + d / 2) / d);
}
static uint64_t sha_blocks(size_t bytes) {
    return (uint64_t)((bytes + 9u + 63u) / 64u);
}
static int is_zero(const void *p, size_t n) {
    const uint8_t *b = (const uint8_t *)p;
    for (size_t i = 0; i < n; ++i)
        if (b[i]) return 0;
    return 1;
}
static void work_add(elpis_ecs_mem_work *into, const elpis_ecs_mem_work *w) {
    into->index_probes += w->index_probes;
    into->slot_scans += w->slot_scans;
    into->sha_blocks += w->sha_blocks;
    into->mac_ops += w->mac_ops;
    into->candidates_scored += w->candidates_scored;
}

/* ------------------------------------------------------------------------------------------------------ */
/* Guard, lease, refusal telemetry                                                                        */
/* ------------------------------------------------------------------------------------------------------ */

static int enter_read(const elpis_ecs_mem *m) {
    _Atomic int *g = (_Atomic int *)&m->guard; /* the handle is never a const object; only the API is */
    int v = atomic_load(g);
    while (v >= 0)
        if (atomic_compare_exchange_weak(g, &v, v + 1)) return 1;
    return 0;
}
static void leave_read(const elpis_ecs_mem *m) {
    atomic_fetch_sub((_Atomic int *)&m->guard, 1);
}
static int enter_write(elpis_ecs_mem *m) {
    int idle = 0;
    return atomic_compare_exchange_strong(&m->guard, &idle, -1);
}
static void leave_write(elpis_ecs_mem *m) {
    atomic_store(&m->guard, 0);
}
/* Called with the write guard held. */
static elpis_ecs_mem_status refuse(elpis_ecs_mem *m, elpis_ecs_mem_status s) {
    m->tel.refusals++;
    if (s < 0 && -(int)s < 12) m->tel.refusals_by_code[-(int)s]++;
    return s;
}

const char *elpis_ecs_mem_code_name(int status) {
    switch (status) {
    case ELPIS_ECS_MEM_OK: return "ELPIS_ECS_MEM_OK";
    case ELPIS_ECS_MEM_INVALID: return "ELPIS_ECS_MEM_INVALID";
    case ELPIS_ECS_MEM_CAPACITY: return "ELPIS_ECS_MEM_CAPACITY";
    case ELPIS_ECS_MEM_NUMERIC: return "ELPIS_ECS_MEM_NUMERIC";
    case ELPIS_ECS_MEM_CLOCK: return "ELPIS_ECS_MEM_CLOCK";
    case ELPIS_ECS_MEM_BUSY: return "ELPIS_ECS_MEM_BUSY";
    case ELPIS_ECS_MEM_LEASED: return "ELPIS_ECS_MEM_LEASED";
    case ELPIS_ECS_MEM_CORRUPT: return "ELPIS_ECS_MEM_CORRUPT";
    case ELPIS_ECS_MEM_NOMEM: return "ELPIS_ECS_MEM_NOMEM";
    case ELPIS_ECS_MEM_NOT_FOUND: return "ELPIS_ECS_MEM_NOT_FOUND";
    case ELPIS_ECS_MEM_NO_NET_EVIDENCE: return "ELPIS_ECS_MEM_NO_NET_EVIDENCE";
    case ELPIS_ECS_MEM_UNQUALIFIED: return "ELPIS_ECS_MEM_UNQUALIFIED";
    default: return "ELPIS_ECS_MEM_UNKNOWN";
    }
}

/* ------------------------------------------------------------------------------------------------------ */
/* Exact hash index (linear probing, backward-shift deletion) and intrusive LRU list                      */
/* ------------------------------------------------------------------------------------------------------ */

static const uint8_t *key_at(keyview kv, uint32_t i) {
    return kv.base + (size_t)i * kv.stride;
}
static uint32_t home_of(const uint8_t *key, uint32_t mask) {
    return (uint32_t)(mix64(rd64(key) ^ mix64(rd64(key + 8))) & mask);
}
static uint32_t index_find(const uint32_t *index, uint32_t mask, keyview kv, const uint8_t *key,
                           uint64_t *probes) {
    uint32_t pos = home_of(key, mask);
    for (;;) {
        ++*probes;
        const uint32_t v = index[pos];
        if (!v) return NONE;
        if (!memcmp(key_at(kv, v - 1u), key, ELPIS_ECS_MEM_DIGEST_BYTES)) return v - 1u;
        pos = (pos + 1u) & mask;
    }
}
static void index_insert(uint32_t *index, uint32_t mask, keyview kv, uint32_t i, uint64_t *probes,
                         uint64_t *collisions) {
    uint32_t pos = home_of(key_at(kv, i), mask);
    ++*probes;
    if (index[pos] && collisions) ++*collisions;
    while (index[pos]) {
        pos = (pos + 1u) & mask;
        ++*probes;
    }
    index[pos] = i + 1u;
}
static void index_remove(uint32_t *index, uint32_t mask, keyview kv, uint32_t i, uint64_t *probes) {
    uint32_t pos = home_of(key_at(kv, i), mask);
    while (index[pos] != i + 1u) {
        pos = (pos + 1u) & mask;
        ++*probes;
    }
    uint32_t hole = pos, j = pos;
    for (;;) {
        j = (j + 1u) & mask;
        ++*probes;
        if (!index[j]) break;
        const uint32_t h = home_of(key_at(kv, index[j] - 1u), mask);
        /* The entry at j may stay iff its home lies cyclically in (hole, j]. */
        const int stays = hole <= j ? (hole < h && h <= j) : (hole < h || h <= j);
        if (stays) continue;
        index[hole] = index[j];
        hole = j;
    }
    index[hole] = 0;
}
static void lru_unlink(table *t, uint32_t i) {
    const uint32_t p = t->prev[i], n = t->next[i];
    if (p != NONE) t->next[p] = n; else t->head = n;
    if (n != NONE) t->prev[n] = p; else t->tail = p;
    t->prev[i] = t->next[i] = NONE;
}
static void lru_push_tail(table *t, uint32_t i) {
    t->prev[i] = t->tail;
    t->next[i] = NONE;
    if (t->tail != NONE) t->next[t->tail] = i; else t->head = i;
    t->tail = i;
}
static void lru_touch(table *t, uint32_t i) {
    if (t->tail == i) return;
    lru_unlink(t, i);
    lru_push_tail(t, i);
}
static keyview slot_keys(const elpis_ecs_mem *m) {
    keyview kv = {(const uint8_t *)m->slots, sizeof(elpis_ecs_mem_slot)};
    return kv;
}
static keyview ent_keys(const elpis_ecs_mem *m) {
    keyview kv = {(const uint8_t *)m->ents, sizeof(entity_rec)};
    return kv;
}

/* ------------------------------------------------------------------------------------------------------ */
/* Deterministic encoding (SHA-256 counter mode). Bit 1 means -1, bit 0 means +1.                         */
/* ------------------------------------------------------------------------------------------------------ */

static void code_bits(const char *domain, size_t dlen, const uint8_t *data, size_t len, uint32_t words,
                      uint64_t *out, uint64_t *blocks) {
    const uint32_t chunks = (words + 3u) / 4u; /* 256 bits per SHA-256 output; the last may be truncated */
    for (uint32_t c = 0; c < chunks; ++c) {
        elpis_sha256_ctx ctx;
        uint8_t ctr[4], digest[32];
        le32(ctr, c);
        elpis_sha256_init(&ctx);
        elpis_sha256_update(&ctx, domain, dlen);
        elpis_sha256_update(&ctx, data, len);
        elpis_sha256_update(&ctx, ctr, 4);
        elpis_sha256_final(&ctx, digest);
        for (uint32_t w = 0; w < 4 && 4u * c + w < words; ++w) out[4u * c + w] = rd64(digest + 8u * w);
        *blocks += sha_blocks(dlen + len + 4u);
    }
}
#define DOMAIN(s) s, sizeof(s)
static void entity_bits(const elpis_ecs_mem_digest *e, uint32_t words, uint64_t *out, uint64_t *blocks) {
    code_bits(DOMAIN("elpis.ecs-mem.code.entity.v1"), e->b, 32, words, out, blocks);
}
static void pair_bits(const char *domain, size_t dlen, uint32_t a, uint32_t b, uint32_t words, uint64_t *out,
                      uint64_t *blocks) {
    uint8_t buf[8];
    le32(buf, a);
    le32(buf + 4, b);
    code_bits(domain, dlen, buf, 8, words, out, blocks);
}
static void one_bits(const char *domain, size_t dlen, uint32_t a, uint32_t words, uint64_t *out,
                     uint64_t *blocks) {
    uint8_t buf[4];
    le32(buf, a);
    code_bits(domain, dlen, buf, 4, words, out, blocks);
}
static int bit_at(const uint64_t *bits, uint32_t i) {
    return (int)((bits[i >> 6] >> (i & 63u)) & 1u);
}
static uint32_t bank_of(uint32_t type, uint32_t role, uint32_t ordinal, uint32_t banks, uint64_t *blocks) {
    static const char domain[] = "elpis.ecs-mem.bank.v1";
    uint8_t buf[12], digest[32];
    elpis_sha256_ctx ctx;
    le32(buf, type);
    le32(buf + 4, role);
    le32(buf + 8, ordinal);
    elpis_sha256_init(&ctx);
    elpis_sha256_update(&ctx, domain, sizeof(domain));
    elpis_sha256_update(&ctx, buf, sizeof(buf));
    elpis_sha256_final(&ctx, digest);
    *blocks += sha_blocks(sizeof(domain) + sizeof(buf));
    return (uint32_t)(rd64(digest) % banks);
}
/* cue = majority_k(role(r,o) * entity(e)) * type(T) * target(r*, o*). */
static void cue_bits(uint32_t type, uint32_t trole, uint32_t tord, const elpis_ecs_mem_participant *known,
                     uint32_t k, uint32_t dim, uint64_t *cue, uint64_t *blocks) {
    const uint32_t words = dim / 64u;
    uint64_t a[ELPIS_ECS_MEM_MAX_DIM / 64u], b[ELPIS_ECS_MEM_MAX_DIM / 64u];
    memset(cue, 0, words * sizeof(uint64_t));
    if (k == 1) {
        entity_bits(&known[0].entity, words, a, blocks);
        pair_bits(DOMAIN("elpis.ecs-mem.code.role.v1"), known[0].role, known[0].ordinal, words, b, blocks);
        for (uint32_t w = 0; w < words; ++w) cue[w] = a[w] ^ b[w];
    } else if (k > 1) {
        int8_t count[ELPIS_ECS_MEM_MAX_DIM];
        memset(count, 0, dim);
        for (uint32_t p = 0; p < k; ++p) {
            entity_bits(&known[p].entity, words, a, blocks);
            pair_bits(DOMAIN("elpis.ecs-mem.code.role.v1"), known[p].role, known[p].ordinal, words, b, blocks);
            for (uint32_t i = 0; i < dim; ++i) {
                const int neg = (int)(((a[i >> 6] ^ b[i >> 6]) >> (i & 63u)) & 1u);
                count[i] = (int8_t)(count[i] + (neg ? -1 : 1));
            }
        }
        if (!(k & 1u)) one_bits(DOMAIN("elpis.ecs-mem.code.tie.v1"), type, words, b, blocks);
        for (uint32_t i = 0; i < dim; ++i) {
            const int neg = count[i] < 0 ? 1 : count[i] > 0 ? 0 : bit_at(b, i);
            cue[i >> 6] |= (uint64_t)neg << (i & 63u);
        }
    }
    one_bits(DOMAIN("elpis.ecs-mem.code.type.v1"), type, words, a, blocks);
    pair_bits(DOMAIN("elpis.ecs-mem.code.target.v1"), trole, tord, words, b, blocks);
    for (uint32_t w = 0; w < words; ++w) cue[w] ^= a[w] ^ b[w];
}

/* ------------------------------------------------------------------------------------------------------ */
/* Bank arithmetic                                                                                        */
/* ------------------------------------------------------------------------------------------------------ */

static uint32_t bank_maxabs(const int32_t *bank, uint32_t dim, uint64_t *ops) {
    uint32_t best = 0;
    const size_t n = (size_t)dim * dim;
    for (size_t i = 0; i < n; ++i) {
        const int64_t v = bank[i];
        const uint32_t a = (uint32_t)(v < 0 ? -v : v);
        if (a > best) best = a;
    }
    *ops += n;
    return best;
}
/* One law step: row_i <- retain row_i, then row_i <- row_i + d_i c^T with d_i = rate (t_i - row_i.c / D) (DELTA)
 * or d_i = rate t_i (HEBBIAN). target NULL means the zero target. Bound (ECS_MEMORY_MF_R0.md, "Numerical
 * bound"): the DELTA step is non-expansive on each row up to rounding, and any step adds at most
 * (65536 + 1) sqrt(D) to a row's 2-norm; with every |w| <= 2^24 before at most 128 steps and D <= 1024, every
 * |w| stays below 32 (2^24 + 128 * 65537) < 2^30. Callers check that bound first. */
static void bank_step(int32_t *bank, uint32_t dim, const uint64_t *cue, const uint64_t *target, uint32_t rate,
                      uint32_t retain, uint32_t law, uint64_t *ops) {
    const int64_t D = dim;
    for (uint32_t i = 0; i < dim; ++i) {
        int32_t *row = bank + (size_t)i * dim;
        const int64_t t = target ? (bit_at(target, i) ? -Q16 : Q16) : 0;
        if (retain != ELPIS_ECS_MEM_Q16_ONE) {
            for (uint32_t j = 0; j < dim; ++j) row[j] = (int32_t)rdiv((int64_t)retain * row[j], Q16);
            *ops += dim;
        }
        int64_t d;
        if (law == ELPIS_ECS_MEM_LAW_DELTA) {
            int64_t s = 0;
            for (uint32_t j = 0; j < dim; ++j) s += bit_at(cue, j) ? -(int64_t)row[j] : (int64_t)row[j];
            *ops += dim;
            const int64_t y = rdiv(s, D);
            d = rdiv((int64_t)rate * (t - y), Q16);
        } else {
            d = rdiv((int64_t)rate * t, Q16);
        }
        if (d) {
            for (uint32_t j = 0; j < dim; ++j) row[j] = (int32_t)(row[j] + (bit_at(cue, j) ? -d : d));
            *ops += dim;
        }
    }
}
static int64_t readout(const int32_t *bank, uint32_t dim, const uint64_t *cue, int64_t *y, uint64_t *ops) {
    int64_t signal = 0;
    for (uint32_t i = 0; i < dim; ++i) {
        const int32_t *row = bank + (size_t)i * dim;
        int64_t s = 0;
        for (uint32_t j = 0; j < dim; ++j) s += bit_at(cue, j) ? -(int64_t)row[j] : (int64_t)row[j];
        y[i] = rdiv(s, (int64_t)dim);
        signal += y[i] < 0 ? -y[i] : y[i];
    }
    *ops += (uint64_t)dim * dim;
    return signal;
}
static int64_t score_of(const int64_t *y, uint32_t dim, const uint64_t *code) {
    int64_t s = 0;
    for (uint32_t i = 0; i < dim; ++i) s += bit_at(code, i) ? -y[i] : y[i];
    return s;
}

/* ------------------------------------------------------------------------------------------------------ */
/* Configuration and sizes                                                                                */
/* ------------------------------------------------------------------------------------------------------ */

void elpis_ecs_mem_config_default(elpis_ecs_mem_config *out) {
    if (!out) return;
    memset(out, 0, sizeof(*out));
    out->abi_version = ELPIS_ECS_MEM_ABI_V1;
    out->dim = 256;
    out->banks = 4;
    out->slot_capacity = 1024;
    out->codebook_capacity = 1024;
    out->slot_policy = ELPIS_ECS_MEM_EVICT_LRU;
    out->codebook_policy = ELPIS_ECS_MEM_EVICT_LRU;
    out->law = ELPIS_ECS_MEM_LAW_DELTA;
    out->learn_rate_q16 = ELPIS_ECS_MEM_Q16_ONE;
    out->retain_q16 = ELPIS_ECS_MEM_Q16_ONE;
    out->slow_rate_q16 = ELPIS_ECS_MEM_Q16_ONE / 2u;
    out->capabilities = 0; /* admitted operations only */
    out->byte_budget = (uint64_t)64 << 20;
}
static int config_shape_ok(const elpis_ecs_mem_config *c) {
    if (c->abi_version != ELPIS_ECS_MEM_ABI_V1) return 0;
    if (c->capabilities & ~(uint32_t)ELPIS_ECS_MEM_CAP_RECURRENT) return 0;
    if (c->dim < ELPIS_ECS_MEM_MIN_DIM || c->dim > ELPIS_ECS_MEM_MAX_DIM || (c->dim & (c->dim - 1u))) return 0;
    if (!c->banks || c->banks > ELPIS_ECS_MEM_MAX_BANKS) return 0;
    if (!c->slot_capacity || c->slot_capacity > ELPIS_ECS_MEM_MAX_SLOTS) return 0;
    if (c->codebook_capacity < ELPIS_ECS_MEM_MAX_PARTICIPANTS || c->codebook_capacity > ELPIS_ECS_MEM_MAX_CODEBOOK) return 0;
    if (c->slot_policy > ELPIS_ECS_MEM_EVICT_LRU || c->codebook_policy > ELPIS_ECS_MEM_EVICT_LRU) return 0;
    if (c->law > ELPIS_ECS_MEM_LAW_HEBBIAN) return 0;
    if (!c->learn_rate_q16 || c->learn_rate_q16 > ELPIS_ECS_MEM_Q16_ONE) return 0;
    if (!c->retain_q16 || c->retain_q16 > ELPIS_ECS_MEM_Q16_ONE) return 0;
    if (!c->slow_rate_q16 || c->slow_rate_q16 > ELPIS_ECS_MEM_Q16_ONE) return 0;
    return 1;
}
static uint32_t index_size(uint32_t cap) {
    uint32_t n = 16;
    while (n < 2u * cap) n <<= 1;
    return n;
}
typedef struct {
    uint64_t fast, slow, slots, sprev, snext, sindex, sscratch, ents, eprev, enext, eindex, escratch, marks, total;
} layout;
static int place(uint64_t *total, uint64_t bytes, uint64_t *at) {
    const uint64_t aligned = (*total + 63u) & ~(uint64_t)63u;
    if (aligned < *total || bytes > ELPIS_ECS_MEM_MAX_ARENA_BYTES || aligned > ELPIS_ECS_MEM_MAX_ARENA_BYTES - bytes) return 0;
    *at = aligned;
    *total = aligned + bytes;
    return 1;
}
static int compute_layout(const elpis_ecs_mem_config *c, layout *l) {
    const uint64_t bank = (uint64_t)c->banks * c->dim * c->dim * sizeof(int32_t);
    const uint64_t S = c->slot_capacity, E = c->codebook_capacity;
    const uint64_t si = index_size(c->slot_capacity), ei = index_size(c->codebook_capacity);
    uint64_t t = 0;
    memset(l, 0, sizeof(*l));
    if (!place(&t, bank, &l->fast) || !place(&t, bank, &l->slow) ||
        !place(&t, S * sizeof(elpis_ecs_mem_slot), &l->slots) || !place(&t, S * 4u, &l->sprev) ||
        !place(&t, S * 4u, &l->snext) || !place(&t, si * 4u, &l->sindex) || !place(&t, si * 4u, &l->sscratch) ||
        !place(&t, E * sizeof(entity_rec), &l->ents) || !place(&t, E * 4u, &l->eprev) ||
        !place(&t, E * 4u, &l->enext) || !place(&t, ei * 4u, &l->eindex) || !place(&t, ei * 4u, &l->escratch) ||
        !place(&t, S > E ? S : E, &l->marks))
        return 0;
    l->total = (t + 63u) & ~(uint64_t)63u;
    return 1;
}
static uint64_t state_bytes_for(const elpis_ecs_mem_config *c, uint64_t slots, uint64_t ents) {
    return ELPIS_ECS_MEM_HEADER_BYTES + 2u * (uint64_t)c->banks * c->dim * c->dim * 4u +
           slots * (ELPIS_ECS_MEM_SLOT_RECORD_BYTES + 4u) + ents * (ELPIS_ECS_MEM_CODEBOOK_RECORD_BYTES + 4u) +
           ELPIS_ECS_MEM_DIGEST_BYTES;
}
elpis_ecs_mem_status elpis_ecs_mem_required_bytes(const elpis_ecs_mem_config *config, uint64_t *arena_bytes,
                                      uint64_t *max_state_bytes) {
    layout l;
    if (!config || !arena_bytes || !max_state_bytes || !config_shape_ok(config)) return ELPIS_ECS_MEM_INVALID;
    if (!compute_layout(config, &l)) return ELPIS_ECS_MEM_CAPACITY;
    *arena_bytes = l.total;
    *max_state_bytes = state_bytes_for(config, config->slot_capacity, config->codebook_capacity);
    return ELPIS_ECS_MEM_OK;
}

static void table_init(table *t, uint32_t cap, uint32_t *prev, uint32_t *next, uint32_t *index) {
    t->cap = cap;
    t->count = 0;
    t->head = t->tail = NONE;
    t->mask = index_size(cap) - 1u;
    t->prev = prev;
    t->next = next;
    t->index = index;
    for (uint32_t i = 0; i < cap; ++i) prev[i] = next[i] = NONE;
    memset(index, 0, (size_t)(t->mask + 1u) * 4u);
}

elpis_ecs_mem_status elpis_ecs_mem_create(const elpis_ecs_mem_config *config, elpis_ecs_mem **out) {
    layout l;
    if (!config || !out || !config_shape_ok(config)) return ELPIS_ECS_MEM_INVALID;
    if (!compute_layout(config, &l) || l.total > config->byte_budget) return ELPIS_ECS_MEM_CAPACITY;
    if (l.total > (uint64_t)SIZE_MAX) return ELPIS_ECS_MEM_CAPACITY;
    elpis_ecs_mem *m = (elpis_ecs_mem *)calloc(1, sizeof(*m));
    if (!m) return ELPIS_ECS_MEM_NOMEM;
    uint8_t *arena = (uint8_t *)aligned_alloc(64, (size_t)l.total);
    if (!arena) {
        free(m);
        return ELPIS_ECS_MEM_NOMEM;
    }
    memset(arena, 0, (size_t)l.total);
    m->cfg = *config;
    atomic_init(&m->guard, 0);
    m->words = config->dim / 64u;
    m->arena = arena;
    m->arena_bytes = l.total;
    m->fast = (int32_t *)(void *)(arena + l.fast);
    m->slow = (int32_t *)(void *)(arena + l.slow);
    m->slots = (elpis_ecs_mem_slot *)(void *)(arena + l.slots);
    m->ents = (entity_rec *)(void *)(arena + l.ents);
    m->scratch_slot_index = (uint32_t *)(void *)(arena + l.sscratch);
    m->scratch_ent_index = (uint32_t *)(void *)(arena + l.escratch);
    m->scratch_marks = arena + l.marks;
    table_init(&m->st, config->slot_capacity, (uint32_t *)(void *)(arena + l.sprev),
               (uint32_t *)(void *)(arena + l.snext), (uint32_t *)(void *)(arena + l.sindex));
    table_init(&m->et, config->codebook_capacity, (uint32_t *)(void *)(arena + l.eprev),
               (uint32_t *)(void *)(arena + l.enext), (uint32_t *)(void *)(arena + l.eindex));
    *out = m;
    return ELPIS_ECS_MEM_OK;
}

void elpis_ecs_mem_destroy(elpis_ecs_mem *memory) {
    if (!memory) return;
    free(memory->arena);
    free(memory);
}

elpis_ecs_mem_status elpis_ecs_mem_lease_claim(elpis_ecs_mem *m, uint64_t lease) {
    if (!m || !lease) return ELPIS_ECS_MEM_INVALID;
    if (!enter_write(m)) return ELPIS_ECS_MEM_BUSY;
    m->lease = lease;
    leave_write(m);
    return ELPIS_ECS_MEM_OK;
}
elpis_ecs_mem_status elpis_ecs_mem_lease_release(elpis_ecs_mem *m, uint64_t lease) {
    if (!m || !lease) return ELPIS_ECS_MEM_INVALID;
    if (!enter_write(m)) return ELPIS_ECS_MEM_BUSY;
    elpis_ecs_mem_status s = ELPIS_ECS_MEM_OK;
    if (m->lease != lease) s = refuse(m, ELPIS_ECS_MEM_LEASED);
    else m->lease = 0;
    leave_write(m);
    return s;
}

/* ------------------------------------------------------------------------------------------------------ */
/* Observations and relation keys                                                                         */
/* ------------------------------------------------------------------------------------------------------ */

static int participant_less(const elpis_ecs_mem_participant *a, const elpis_ecs_mem_participant *b) {
    if (a->role != b->role) return a->role < b->role;
    if (a->ordinal != b->ordinal) return a->ordinal < b->ordinal;
    return memcmp(a->entity.b, b->entity.b, 32) < 0;
}
/* Copy, canonicalize and validate participants; (role, ordinal) pairs must be distinct. */
static int canonical_participants(const elpis_ecs_mem_participant *in, uint32_t n, elpis_ecs_mem_participant *out) {
    if (n < 1 || n > ELPIS_ECS_MEM_MAX_PARTICIPANTS) return 0;
    memset(out, 0, sizeof(elpis_ecs_mem_participant) * ELPIS_ECS_MEM_MAX_PARTICIPANTS);
    for (uint32_t i = 0; i < n; ++i) {
        elpis_ecs_mem_participant v = in[i];
        uint32_t j = i;
        while (j > 0 && participant_less(&v, &out[j - 1])) {
            out[j] = out[j - 1];
            --j;
        }
        out[j] = v;
    }
    for (uint32_t i = 1; i < n; ++i)
        if (out[i].role == out[i - 1].role && out[i].ordinal == out[i - 1].ordinal) return 0;
    for (uint32_t i = n; i < ELPIS_ECS_MEM_MAX_PARTICIPANTS; ++i)
        if (!is_zero(&in[i], sizeof(in[i]))) return 0;
    return 1;
}
static void relation_key_of(uint32_t type, const elpis_ecs_mem_digest *payload, const elpis_ecs_mem_participant *parts,
                            uint32_t n, uint8_t out[32], uint64_t *blocks) {
    static const char domain[] = "elpis.ecs-mem.relation.v1";
    elpis_sha256_ctx ctx;
    uint8_t buf[8];
    elpis_sha256_init(&ctx);
    elpis_sha256_update(&ctx, domain, sizeof(domain));
    le32(buf, type);
    elpis_sha256_update(&ctx, buf, 4);
    elpis_sha256_update(&ctx, payload->b, 32);
    le32(buf, n);
    elpis_sha256_update(&ctx, buf, 4);
    for (uint32_t i = 0; i < n; ++i) {
        le32(buf, parts[i].role);
        le32(buf + 4, parts[i].ordinal);
        elpis_sha256_update(&ctx, buf, 8);
        elpis_sha256_update(&ctx, parts[i].entity.b, 32);
    }
    elpis_sha256_final(&ctx, out);
    *blocks += sha_blocks(sizeof(domain) + 40u + 40u * n);
}
static int observation_ok(const elpis_ecs_mem_observation *o, elpis_ecs_mem_participant *parts, int check_polarity) {
    if (!o || o->abi_version != ELPIS_ECS_MEM_ABI_V1 || o->reserved) return 0;
    if (check_polarity && (o->polarity < ELPIS_ECS_MEM_ASSERT || o->polarity > ELPIS_ECS_MEM_UNCERTAIN)) return 0;
    return canonical_participants(o->participants, o->participant_count, parts);
}

elpis_ecs_mem_status elpis_ecs_mem_relation_key(const elpis_ecs_mem_observation *observation, elpis_ecs_mem_digest *out) {
    elpis_ecs_mem_participant parts[ELPIS_ECS_MEM_MAX_PARTICIPANTS];
    uint64_t blocks = 0;
    if (!out || !observation_ok(observation, parts, 0)) return ELPIS_ECS_MEM_INVALID;
    relation_key_of(observation->relation_type, &observation->payload, parts, observation->participant_count,
                    out->b, &blocks);
    return ELPIS_ECS_MEM_OK;
}

elpis_ecs_mem_status elpis_ecs_mem_observe(elpis_ecs_mem *m, uint64_t lease, const elpis_ecs_mem_observation *o, elpis_ecs_mem_digest *out_key,
                               elpis_ecs_mem_work *work) {
    elpis_ecs_mem_participant parts[ELPIS_ECS_MEM_MAX_PARTICIPANTS];
    elpis_ecs_mem_work w;
    memset(&w, 0, sizeof(w));
    if (!m) return ELPIS_ECS_MEM_INVALID;
    if (!enter_write(m)) return ELPIS_ECS_MEM_BUSY;
    if (m->lease != lease) {
        elpis_ecs_mem_status s = refuse(m, ELPIS_ECS_MEM_LEASED);
        leave_write(m);
        return s;
    }
    if (!observation_ok(o, parts, 1)) {
        elpis_ecs_mem_status s = refuse(m, ELPIS_ECS_MEM_INVALID);
        leave_write(m);
        return s;
    }
    elpis_ecs_mem_status status = ELPIS_ECS_MEM_OK;
    const uint32_t n = o->participant_count;
    uint8_t key[32];
    relation_key_of(o->relation_type, &o->payload, parts, n, key, &w.sha_blocks);
    const keyview sk = slot_keys(m), ek = ent_keys(m);

    /* Plan the slot. */
    uint32_t slot = index_find(m->st.index, m->st.mask, sk, key, &w.index_probes), victim = NONE;
    if (o->tick < m->clock) {
        status = ELPIS_ECS_MEM_CLOCK;
    } else if (slot != NONE) {
        const elpis_ecs_mem_slot *s = &m->slots[slot];
        const uint32_t c = o->polarity == ELPIS_ECS_MEM_ASSERT   ? s->assert_count
                           : o->polarity == ELPIS_ECS_MEM_NEGATE ? s->negate_count
                                                           : s->uncertain_count;
        if (c == UINT32_MAX) status = ELPIS_ECS_MEM_CLOCK;
    } else if (m->st.count == m->st.cap) {
        if (m->cfg.slot_policy == ELPIS_ECS_MEM_REFUSE) status = ELPIS_ECS_MEM_CAPACITY;
        else victim = m->st.head;
    }

    /* Plan the codebook: distinct entities, existing entries (touched, never victims), victims. */
    elpis_ecs_mem_digest uniq[ELPIS_ECS_MEM_MAX_PARTICIPANTS];
    uint32_t found[ELPIS_ECS_MEM_MAX_PARTICIPANTS], victims[ELPIS_ECS_MEM_MAX_PARTICIPANTS], nu = 0, missing = 0, nv = 0;
    if (status == ELPIS_ECS_MEM_OK) {
        for (uint32_t i = 0; i < n; ++i) {
            int dup = 0;
            for (uint32_t j = 0; j < nu; ++j)
                if (!memcmp(uniq[j].b, parts[i].entity.b, 32)) dup = 1;
            if (dup) continue;
            uniq[nu] = parts[i].entity;
            found[nu] = index_find(m->et.index, m->et.mask, ek, uniq[nu].b, &w.index_probes);
            if (found[nu] == NONE) ++missing;
            ++nu;
        }
        const uint32_t free_entries = m->et.cap - m->et.count;
        const uint32_t need = missing > free_entries ? missing - free_entries : 0;
        if (need && m->cfg.codebook_policy == ELPIS_ECS_MEM_REFUSE) {
            status = ELPIS_ECS_MEM_CAPACITY;
        } else {
            for (uint32_t e = m->et.head; e != NONE && nv < need; e = m->et.next[e]) {
                int present = 0;
                for (uint32_t j = 0; j < nu; ++j)
                    if (found[j] == e) present = 1;
                if (!present) victims[nv++] = e;
            }
            if (nv < need) status = ELPIS_ECS_MEM_CAPACITY;
        }
    }
    if (status != ELPIS_ECS_MEM_OK) {
        status = refuse(m, status);
        leave_write(m);
        return status;
    }

    /* Commit: nothing below can fail. */
    elpis_ecs_mem_slot *s;
    if (slot == NONE) {
        if (victim != NONE) {
            index_remove(m->st.index, m->st.mask, sk, victim, &w.index_probes);
            lru_unlink(&m->st, victim);
            slot = victim;
            m->tel.slot_evictions++;
        } else {
            slot = m->st.count++;
        }
        s = &m->slots[slot];
        memset(s, 0, sizeof(*s));
        memcpy(s->key.b, key, 32);
        s->payload = o->payload;
        s->relation_type = o->relation_type;
        s->participant_count = n;
        memcpy(s->participants, parts, sizeof(parts));
        s->first_tick = o->tick;
        s->first_provenance = o->provenance;
        index_insert(m->st.index, m->st.mask, sk, slot, &w.index_probes, &m->tel.index_collisions);
        lru_push_tail(&m->st, slot);
        m->tel.new_slots++;
    } else {
        s = &m->slots[slot];
        lru_touch(&m->st, slot);
        m->tel.repeated_observations++;
    }
    if (o->polarity == ELPIS_ECS_MEM_ASSERT) {
        s->assert_count++;
        s->last_assert_tick = o->tick;
    } else if (o->polarity == ELPIS_ECS_MEM_NEGATE) {
        s->negate_count++;
        s->last_negate_tick = o->tick;
    } else {
        s->uncertain_count++;
    }
    s->last_tick = o->tick;
    s->last_polarity = o->polarity;
    s->last_authority = o->source_authority;
    s->last_provenance = o->provenance;
    s->structure_ref = o->structure_ref;

    uint32_t next_victim = 0;
    for (uint32_t j = 0; j < nu; ++j) {
        uint32_t e = found[j];
        if (e != NONE) {
            lru_touch(&m->et, e);
            m->ents[e].last_tick = o->tick;
            continue;
        }
        if (m->et.count < m->et.cap) {
            e = m->et.count++;
        } else {
            e = victims[next_victim++];
            index_remove(m->et.index, m->et.mask, ek, e, &w.index_probes);
            lru_unlink(&m->et, e);
            m->tel.codebook_evictions++;
        }
        m->ents[e].id = uniq[j];
        m->ents[e].first_tick = m->ents[e].last_tick = o->tick;
        index_insert(m->et.index, m->et.mask, ek, e, &w.index_probes, &m->tel.index_collisions);
        lru_push_tail(&m->et, e);
        m->tel.codebook_admissions++;
    }
    m->clock = o->tick;
    m->tel.observes++;
    work_add(&m->tel.write_work, &w);
    if (out_key) memcpy(out_key->b, key, 32);
    if (work) *work = w;
    leave_write(m);
    return ELPIS_ECS_MEM_OK;
}

/* ------------------------------------------------------------------------------------------------------ */
/* LEARN                                                                                                  */
/* ------------------------------------------------------------------------------------------------------ */

typedef struct {
    uint32_t bank[ELPIS_ECS_MEM_MAX_PARTICIPANTS];
    uint64_t cue[ELPIS_ECS_MEM_MAX_PARTICIPANTS][ELPIS_ECS_MEM_MAX_DIM / 64u];
    uint64_t target[ELPIS_ECS_MEM_MAX_PARTICIPANTS][ELPIS_ECS_MEM_MAX_DIM / 64u];
} slot_plan;

/* Cues, banks and targets of every direction of one slot (each participant in turn is the target). */
static void plan_slot(const elpis_ecs_mem *m, const elpis_ecs_mem_slot *s, slot_plan *p, uint64_t *blocks) {
    const uint32_t n = s->participant_count;
    for (uint32_t t = 0; t < n; ++t) {
        elpis_ecs_mem_participant known[ELPIS_ECS_MEM_MAX_PARTICIPANTS];
        uint32_t k = 0;
        for (uint32_t i = 0; i < n; ++i)
            if (i != t) known[k++] = s->participants[i];
        const elpis_ecs_mem_participant *tg = &s->participants[t];
        p->bank[t] = bank_of(s->relation_type, tg->role, tg->ordinal, m->cfg.banks, blocks);
        cue_bits(s->relation_type, tg->role, tg->ordinal, known, k, m->cfg.dim, p->cue[t], blocks);
        entity_bits(&tg->entity, m->words, p->target[t], blocks);
    }
}
static int64_t net_of(const elpis_ecs_mem_slot *s) {
    return (int64_t)s->assert_count - (int64_t)s->negate_count;
}

elpis_ecs_mem_status elpis_ecs_mem_learn(elpis_ecs_mem *m, uint64_t lease, const elpis_ecs_mem_digest *key, uint64_t tick,
                             elpis_ecs_mem_work *work) {
    elpis_ecs_mem_work w;
    slot_plan plan;
    memset(&w, 0, sizeof(w));
    if (!m) return ELPIS_ECS_MEM_INVALID;
    if (!enter_write(m)) return ELPIS_ECS_MEM_BUSY;
    elpis_ecs_mem_status status = ELPIS_ECS_MEM_OK;
    uint32_t slot = NONE;
    if (m->lease != lease) status = ELPIS_ECS_MEM_LEASED;
    else if (!(m->cfg.capabilities & ELPIS_ECS_MEM_CAP_RECURRENT)) status = ELPIS_ECS_MEM_UNQUALIFIED;
    else if (!key) status = ELPIS_ECS_MEM_INVALID;
    else if (tick < m->clock) status = ELPIS_ECS_MEM_CLOCK;
    else if ((slot = index_find(m->st.index, m->st.mask, slot_keys(m), key->b, &w.index_probes)) == NONE)
        status = ELPIS_ECS_MEM_NOT_FOUND;
    else if (!net_of(&m->slots[slot])) status = ELPIS_ECS_MEM_NO_NET_EVIDENCE;
    else if (m->slots[slot].learn_count == UINT32_MAX) status = ELPIS_ECS_MEM_CLOCK;
    if (status == ELPIS_ECS_MEM_OK) {
        const elpis_ecs_mem_slot *s = &m->slots[slot];
        const size_t bank_floats = (size_t)m->cfg.dim * m->cfg.dim;
        plan_slot(m, s, &plan, &w.sha_blocks);
        for (uint32_t t = 0; t < s->participant_count && status == ELPIS_ECS_MEM_OK; ++t) {
            int seen = 0;
            for (uint32_t u = 0; u < t; ++u)
                if (plan.bank[u] == plan.bank[t]) seen = 1;
            if (!seen && bank_maxabs(m->fast + plan.bank[t] * bank_floats, m->cfg.dim, &w.mac_ops) >
                             ELPIS_ECS_MEM_LEARN_WEIGHT_LIMIT)
                status = ELPIS_ECS_MEM_NUMERIC;
        }
    }
    if (status != ELPIS_ECS_MEM_OK) {
        status = refuse(m, status);
        leave_write(m);
        return status;
    }
    elpis_ecs_mem_slot *s = &m->slots[slot];
    const int positive = net_of(s) > 0;
    const size_t bank_floats = (size_t)m->cfg.dim * m->cfg.dim;
    for (uint32_t t = 0; t < s->participant_count; ++t)
        bank_step(m->fast + plan.bank[t] * bank_floats, m->cfg.dim, plan.cue[t], positive ? plan.target[t] : NULL,
                  m->cfg.learn_rate_q16, m->cfg.retain_q16, m->cfg.law, &w.mac_ops);
    s->learn_count++;
    lru_touch(&m->st, slot);
    m->clock = tick;
    m->tel.learns++;
    m->tel.learn_bank_updates += s->participant_count;
    work_add(&m->tel.write_work, &w);
    if (work) *work = w;
    leave_write(m);
    return ELPIS_ECS_MEM_OK;
}

/* ------------------------------------------------------------------------------------------------------ */
/* QUERY                                                                                                  */
/* ------------------------------------------------------------------------------------------------------ */

static int query_shape_ok(const elpis_ecs_mem_request *q) {
    if (!q || q->abi_version != ELPIS_ECS_MEM_ABI_V1) return 0;
    if (q->cue_count >= ELPIS_ECS_MEM_MAX_PARTICIPANTS) return 0;
    for (uint32_t i = 0; i < q->cue_count; ++i) {
        if (q->cue[i].role == q->target_role && q->cue[i].ordinal == q->target_ordinal) return 0;
        for (uint32_t j = 0; j < i; ++j)
            if (q->cue[i].role == q->cue[j].role && q->cue[i].ordinal == q->cue[j].ordinal) return 0;
    }
    for (uint32_t i = q->cue_count; i < ELPIS_ECS_MEM_MAX_PARTICIPANTS; ++i)
        if (!is_zero(&q->cue[i], sizeof(q->cue[i]))) return 0;
    return 1;
}
static int query_ok(const elpis_ecs_mem_request *q) {
    if (!query_shape_ok(q)) return 0;
    if (q->channel == ELPIS_ECS_MEM_LOOKUP)
        return q->candidate_source == ELPIS_ECS_MEM_CANDIDATES_CALLER && !q->candidate_count && !q->candidates;
    if (q->channel != ELPIS_ECS_MEM_FAST && q->channel != ELPIS_ECS_MEM_SLOW) return 0;
    if (q->candidate_source == ELPIS_ECS_MEM_CANDIDATES_CALLER)
        return q->candidates && q->candidate_count >= 1 && q->candidate_count <= ELPIS_ECS_MEM_MAX_CANDIDATES;
    return q->candidate_source == ELPIS_ECS_MEM_CANDIDATES_CODEBOOK && !q->candidate_count && !q->candidates;
}
static const elpis_ecs_mem_participant *filler_of(const elpis_ecs_mem_slot *s, uint32_t role, uint32_t ordinal) {
    for (uint32_t i = 0; i < s->participant_count; ++i)
        if (s->participants[i].role == role && s->participants[i].ordinal == ordinal) return &s->participants[i];
    return NULL;
}
static int slot_matches(const elpis_ecs_mem_slot *s, const elpis_ecs_mem_request *q) {
    if (s->relation_type != q->relation_type) return 0;
    for (uint32_t c = 0; c < q->cue_count; ++c) {
        const elpis_ecs_mem_participant *p = filler_of(s, q->cue[c].role, q->cue[c].ordinal);
        if (!p || memcmp(p->entity.b, q->cue[c].entity.b, 32)) return 0;
    }
    return 1;
}
/* LOOKUP order: latest assert, then larger net evidence, then smaller filler identity, then smaller key. */
static int lookup_better(const elpis_ecs_mem_slot *a, const elpis_ecs_mem_participant *fa, const elpis_ecs_mem_slot *b,
                         const elpis_ecs_mem_participant *fb) {
    if (a->last_assert_tick != b->last_assert_tick) return a->last_assert_tick > b->last_assert_tick;
    if (net_of(a) != net_of(b)) return net_of(a) > net_of(b);
    const int c = memcmp(fa->entity.b, fb->entity.b, 32);
    if (c) return c < 0;
    return memcmp(a->key.b, b->key.b, 32) < 0;
}
static void lookup(const elpis_ecs_mem *m, const elpis_ecs_mem_request *q, elpis_ecs_mem_answer *a) {
    const elpis_ecs_mem_slot *best = NULL;
    const elpis_ecs_mem_participant *bf = NULL;
    for (uint32_t i = 0; i < m->st.count; ++i) {
        const elpis_ecs_mem_slot *s = &m->slots[i];
        ++a->work.slot_scans;
        if (!slot_matches(s, q)) continue;
        const elpis_ecs_mem_participant *f = filler_of(s, q->target_role, q->target_ordinal);
        if (!f) continue;
        ++a->matches;
        a->uncertain = (uint32_t)((uint64_t)a->uncertain + s->uncertain_count > UINT32_MAX
                                      ? UINT32_MAX
                                      : a->uncertain + s->uncertain_count); /* saturating */
        if (net_of(s) < 0) ++a->contradicted;
        if (net_of(s) > 0 && (!best || lookup_better(s, f, best, bf))) {
            best = s;
            bf = f;
        }
    }
    a->runner_up = INT64_MIN;
    if (!best) return;
    for (uint32_t i = 0; i < m->st.count; ++i) {
        const elpis_ecs_mem_slot *s = &m->slots[i];
        ++a->work.slot_scans;
        if (s == best || net_of(s) <= 0 || !slot_matches(s, q)) continue;
        const elpis_ecs_mem_participant *f = filler_of(s, q->target_role, q->target_ordinal);
        if (!f || !memcmp(f->entity.b, bf->entity.b, 32)) continue;
        ++a->conflicts;
        if (net_of(s) > a->runner_up) a->runner_up = net_of(s);
    }
    a->found = 1;
    a->answer = bf->entity;
    a->answer_key = best->key;
    a->score = net_of(best);
    a->answer_tick = best->last_assert_tick;
}
static void associative(const elpis_ecs_mem *m, const elpis_ecs_mem_request *q, elpis_ecs_mem_answer *a) {
    uint64_t cue[ELPIS_ECS_MEM_MAX_DIM / 64u], code[ELPIS_ECS_MEM_MAX_DIM / 64u];
    int64_t y[ELPIS_ECS_MEM_MAX_DIM];
    const uint32_t bank = bank_of(q->relation_type, q->target_role, q->target_ordinal, m->cfg.banks,
                                  &a->work.sha_blocks);
    cue_bits(q->relation_type, q->target_role, q->target_ordinal, q->cue, q->cue_count, m->cfg.dim, cue,
             &a->work.sha_blocks);
    const int32_t *M = m->fast + (size_t)bank * m->cfg.dim * m->cfg.dim; /* FAST: the only bank channel admitted */
    a->signal = readout(M, m->cfg.dim, cue, y, &a->work.mac_ops);
    const uint32_t n = q->candidate_source == ELPIS_ECS_MEM_CANDIDATES_CALLER ? q->candidate_count : m->et.count;
    int64_t best = INT64_MIN, second = INT64_MIN;
    for (uint32_t c = 0; c < n; ++c) {
        const elpis_ecs_mem_digest *e = q->candidate_source == ELPIS_ECS_MEM_CANDIDATES_CALLER ? &q->candidates[c] : &m->ents[c].id;
        entity_bits(e, m->words, code, &a->work.sha_blocks);
        const int64_t s = score_of(y, m->cfg.dim, code);
        a->work.mac_ops += m->cfg.dim;
        ++a->work.candidates_scored;
        if (!a->found || s > best || (s == best && memcmp(e->b, a->answer.b, 32) < 0)) {
            if (a->found) second = best; /* the displaced best is never below the current second */
            best = s;
            a->answer = *e;
            a->found = 1;
        } else if (s > second) {
            second = s;
        }
    }
    a->matches = n;
    a->score = a->found ? best : 0;
    a->runner_up = second;
}

elpis_ecs_mem_status elpis_ecs_mem_query(const elpis_ecs_mem *m, const elpis_ecs_mem_request *q, elpis_ecs_mem_answer *out) {
    if (!m || !out || !query_ok(q)) return ELPIS_ECS_MEM_INVALID;
    /* SLOW is never admitted here (its law is research-only); FAST needs the recurrent capability. */
    if (q->channel == ELPIS_ECS_MEM_SLOW || (q->channel == ELPIS_ECS_MEM_FAST && !(m->cfg.capabilities & ELPIS_ECS_MEM_CAP_RECURRENT)))
        return ELPIS_ECS_MEM_UNQUALIFIED;
    if (!enter_read(m)) return ELPIS_ECS_MEM_BUSY;
    elpis_ecs_mem_answer a;
    memset(&a, 0, sizeof(a));
    if (q->channel == ELPIS_ECS_MEM_LOOKUP) lookup(m, q, &a);
    else associative(m, q, &a);
    leave_read(m);
    *out = a;
    return ELPIS_ECS_MEM_OK;
}

/* ------------------------------------------------------------------------------------------------------ */
/* INSPECT                                                                                                */
/* ------------------------------------------------------------------------------------------------------ */

elpis_ecs_mem_status elpis_ecs_mem_inspect(const elpis_ecs_mem *m, elpis_ecs_mem_stats *out) {
    if (!m || !out) return ELPIS_ECS_MEM_INVALID;
    if (!enter_read(m)) return ELPIS_ECS_MEM_BUSY;
    elpis_ecs_mem_stats s = m->tel;
    s.config = m->cfg;
    s.clock = m->clock;
    s.slots_used = m->st.count;
    s.codebook_used = m->et.count;
    s.lease_held = m->lease != 0;
    s.state_bytes = state_bytes_for(&m->cfg, m->st.count, m->et.count);
    s.arena_bytes = m->arena_bytes;
    s.allocated_bytes = m->arena_bytes + sizeof(*m);
    leave_read(m);
    *out = s;
    return ELPIS_ECS_MEM_OK;
}
elpis_ecs_mem_status elpis_ecs_mem_slot_read(const elpis_ecs_mem *m, const elpis_ecs_mem_digest *key, elpis_ecs_mem_slot *out) {
    uint64_t probes = 0;
    if (!m || !key || !out) return ELPIS_ECS_MEM_INVALID;
    if (!enter_read(m)) return ELPIS_ECS_MEM_BUSY;
    const uint32_t i = index_find(m->st.index, m->st.mask, slot_keys(m), key->b, &probes);
    if (i != NONE) *out = m->slots[i];
    leave_read(m);
    return i == NONE ? ELPIS_ECS_MEM_NOT_FOUND : ELPIS_ECS_MEM_OK;
}
elpis_ecs_mem_status elpis_ecs_mem_codebook_contains(const elpis_ecs_mem *m, const elpis_ecs_mem_digest *entity, int *out) {
    uint64_t probes = 0;
    if (!m || !entity || !out) return ELPIS_ECS_MEM_INVALID;
    if (!enter_read(m)) return ELPIS_ECS_MEM_BUSY;
    *out = index_find(m->et.index, m->et.mask, ent_keys(m), entity->b, &probes) != NONE;
    leave_read(m);
    return ELPIS_ECS_MEM_OK;
}

/* ------------------------------------------------------------------------------------------------------ */
/* ELPISEM1 envelope                                                                                      */
/* ------------------------------------------------------------------------------------------------------ */

static const uint8_t MAGIC[8] = {'E', 'L', 'P', 'I', 'S', 'E', 'M', '1'};

typedef struct {
    uint8_t *out;
    size_t pos;
    elpis_sha256_ctx *sha;
} sink;
static void emit(sink *k, const void *p, size_t n) {
    if (k->out) memcpy(k->out + k->pos, p, n);
    if (k->sha) elpis_sha256_update(k->sha, p, n);
    k->pos += n;
}
static void config_fields(uint8_t *h, const elpis_ecs_mem_config *c) {
    le32(h + 16, c->dim);
    le32(h + 20, c->banks);
    le32(h + 24, c->slot_capacity);
    le32(h + 28, c->codebook_capacity);
    le32(h + 32, c->slot_policy);
    le32(h + 36, c->codebook_policy);
    le32(h + 40, c->law);
    le32(h + 44, c->learn_rate_q16);
    le32(h + 48, c->retain_q16);
    le32(h + 52, c->slow_rate_q16);
}
static void emit_bank(sink *k, const int32_t *v, size_t n) {
    uint8_t buf[1024];
    size_t at = 0;
    for (size_t i = 0; i < n; ++i) {
        le32(buf + at, (uint32_t)v[i]);
        at += 4;
        if (at == sizeof(buf)) {
            emit(k, buf, at);
            at = 0;
        }
    }
    if (at) emit(k, buf, at);
}
static void slot_encode(const elpis_ecs_mem_slot *s, uint8_t *r) {
    memcpy(r, s->key.b, 32);
    memcpy(r + 32, s->payload.b, 32);
    memcpy(r + 64, s->first_provenance.b, 32);
    memcpy(r + 96, s->last_provenance.b, 32);
    memcpy(r + 128, s->structure_ref.b, 32);
    le32(r + 160, s->relation_type);
    le32(r + 164, s->participant_count);
    le32(r + 168, s->assert_count);
    le32(r + 172, s->negate_count);
    le32(r + 176, s->uncertain_count);
    le32(r + 180, s->last_polarity);
    le32(r + 184, s->last_authority);
    le32(r + 188, s->learn_count);
    le64(r + 192, s->first_tick);
    le64(r + 200, s->last_tick);
    le64(r + 208, s->last_assert_tick);
    le64(r + 216, s->last_negate_tick);
    for (uint32_t i = 0; i < ELPIS_ECS_MEM_MAX_PARTICIPANTS; ++i) {
        uint8_t *p = r + 224 + 40u * i;
        le32(p, s->participants[i].role);
        le32(p + 4, s->participants[i].ordinal);
        memcpy(p + 8, s->participants[i].entity.b, 32);
    }
}
static void slot_decode(const uint8_t *r, elpis_ecs_mem_slot *s) {
    memcpy(s->key.b, r, 32);
    memcpy(s->payload.b, r + 32, 32);
    memcpy(s->first_provenance.b, r + 64, 32);
    memcpy(s->last_provenance.b, r + 96, 32);
    memcpy(s->structure_ref.b, r + 128, 32);
    s->relation_type = rd32(r + 160);
    s->participant_count = rd32(r + 164);
    s->assert_count = rd32(r + 168);
    s->negate_count = rd32(r + 172);
    s->uncertain_count = rd32(r + 176);
    s->last_polarity = rd32(r + 180);
    s->last_authority = rd32(r + 184);
    s->learn_count = rd32(r + 188);
    s->first_tick = rd64(r + 192);
    s->last_tick = rd64(r + 200);
    s->last_assert_tick = rd64(r + 208);
    s->last_negate_tick = rd64(r + 216);
    for (uint32_t i = 0; i < ELPIS_ECS_MEM_MAX_PARTICIPANTS; ++i) {
        const uint8_t *p = r + 224 + 40u * i;
        s->participants[i].role = rd32(p);
        s->participants[i].ordinal = rd32(p + 4);
        memcpy(s->participants[i].entity.b, p + 8, 32);
    }
}
static void serialize(const elpis_ecs_mem *m, sink *k) {
    uint8_t h[ELPIS_ECS_MEM_HEADER_BYTES];
    memset(h, 0, sizeof(h));
    memcpy(h, MAGIC, 8);
    le32(h + 8, ELPIS_ECS_MEM_FORMAT_VERSION);
    le32(h + 12, ELPIS_ECS_MEM_ABI_V1);
    config_fields(h, &m->cfg);
    le64(h + 64, m->clock);
    le32(h + 72, 0); /* the slow-bank cursor: always 0 here */
    le32(h + 76, m->st.count);
    le32(h + 80, m->et.count);
    emit(k, h, sizeof(h));
    const size_t n = (size_t)m->cfg.banks * m->cfg.dim * m->cfg.dim;
    emit_bank(k, m->fast, n);
    emit_bank(k, m->slow, n);
    uint8_t r[ELPIS_ECS_MEM_SLOT_RECORD_BYTES];
    for (uint32_t i = 0; i < m->st.count; ++i) {
        slot_encode(&m->slots[i], r);
        emit(k, r, sizeof(r));
    }
    for (uint32_t i = m->st.head; i != NONE; i = m->st.next[i]) {
        le32(r, i);
        emit(k, r, 4);
    }
    for (uint32_t i = 0; i < m->et.count; ++i) {
        memcpy(r, m->ents[i].id.b, 32);
        le64(r + 32, m->ents[i].first_tick);
        le64(r + 40, m->ents[i].last_tick);
        emit(k, r, ELPIS_ECS_MEM_CODEBOOK_RECORD_BYTES);
    }
    for (uint32_t i = m->et.head; i != NONE; i = m->et.next[i]) {
        le32(r, i);
        emit(k, r, 4);
    }
}

size_t elpis_ecs_mem_snapshot_size(const elpis_ecs_mem *m) {
    if (!m) return 0;
    if (!enter_read(m)) return 0;
    const uint64_t n = state_bytes_for(&m->cfg, m->st.count, m->et.count);
    leave_read(m);
    return (size_t)n;
}
elpis_ecs_mem_status elpis_ecs_mem_snapshot(const elpis_ecs_mem *m, uint8_t *out, size_t capacity, size_t *written) {
    if (!m || !out || !written) return ELPIS_ECS_MEM_INVALID;
    if (!enter_read(m)) return ELPIS_ECS_MEM_BUSY;
    const uint64_t n = state_bytes_for(&m->cfg, m->st.count, m->et.count);
    if (n > capacity) {
        leave_read(m);
        return ELPIS_ECS_MEM_CAPACITY;
    }
    elpis_sha256_ctx sha;
    elpis_sha256_init(&sha);
    sink k = {out, 0, &sha};
    serialize(m, &k);
    elpis_sha256_final(&sha, out + k.pos);
    leave_read(m);
    *written = k.pos + ELPIS_ECS_MEM_DIGEST_BYTES;
    return ELPIS_ECS_MEM_OK;
}
elpis_ecs_mem_status elpis_ecs_mem_identity(const elpis_ecs_mem *m, elpis_ecs_mem_digest *out) {
    if (!m || !out) return ELPIS_ECS_MEM_INVALID;
    if (!enter_read(m)) return ELPIS_ECS_MEM_BUSY;
    elpis_sha256_ctx sha;
    elpis_sha256_init(&sha);
    sink k = {NULL, 0, &sha};
    serialize(m, &k);
    elpis_sha256_final(&sha, out->b);
    leave_read(m);
    return ELPIS_ECS_MEM_OK;
}

static int header_config(const uint8_t *h, elpis_ecs_mem_config *c) {
    if (memcmp(h, MAGIC, 8) || rd32(h + 8) != ELPIS_ECS_MEM_FORMAT_VERSION || rd32(h + 12) != ELPIS_ECS_MEM_ABI_V1) return 0;
    memset(c, 0, sizeof(*c));
    c->abi_version = ELPIS_ECS_MEM_ABI_V1;
    c->dim = rd32(h + 16);
    c->banks = rd32(h + 20);
    c->slot_capacity = rd32(h + 24);
    c->codebook_capacity = rd32(h + 28);
    c->slot_policy = rd32(h + 32);
    c->codebook_policy = rd32(h + 36);
    c->law = rd32(h + 40);
    c->learn_rate_q16 = rd32(h + 44);
    c->retain_q16 = rd32(h + 48);
    c->slow_rate_q16 = rd32(h + 52);
    if (!is_zero(h + 56, 8) || !is_zero(h + 84, ELPIS_ECS_MEM_HEADER_BYTES - 84u)) return 0;
    return config_shape_ok(c);
}
static int same_config(const elpis_ecs_mem_config *a, const elpis_ecs_mem_config *b) {
    return a->dim == b->dim && a->banks == b->banks && a->slot_capacity == b->slot_capacity &&
           a->codebook_capacity == b->codebook_capacity && a->slot_policy == b->slot_policy &&
           a->codebook_policy == b->codebook_policy && a->law == b->law && a->learn_rate_q16 == b->learn_rate_q16 &&
           a->retain_q16 == b->retain_q16 && a->slow_rate_q16 == b->slow_rate_q16;
}
/* Validate an LRU order of `count` u32 indices: a permutation of 0..count-1. */
static int order_ok(const uint8_t *p, uint32_t count, uint8_t *marks) {
    memset(marks, 0, count);
    for (uint32_t k = 0; k < count; ++k) {
        const uint32_t i = rd32(p + 4u * k);
        if (i >= count || marks[i]) return 0;
        marks[i] = 1;
    }
    return 1;
}
static int slot_record_ok(const uint8_t *r, uint64_t clock) {
    elpis_ecs_mem_slot s;
    uint8_t key[32];
    uint64_t blocks = 0;
    slot_decode(r, &s);
    if (s.participant_count < 1 || s.participant_count > ELPIS_ECS_MEM_MAX_PARTICIPANTS) return 0;
    for (uint32_t i = 1; i < s.participant_count; ++i) {
        const elpis_ecs_mem_participant *a = &s.participants[i - 1], *b = &s.participants[i];
        if (a->role > b->role || (a->role == b->role && a->ordinal >= b->ordinal)) return 0;
    }
    for (uint32_t i = s.participant_count; i < ELPIS_ECS_MEM_MAX_PARTICIPANTS; ++i)
        if (!is_zero(r + 224 + 40u * i, 40)) return 0;
    if ((uint64_t)s.assert_count + s.negate_count + s.uncertain_count == 0) return 0;
    if (s.last_polarity < ELPIS_ECS_MEM_ASSERT || s.last_polarity > ELPIS_ECS_MEM_UNCERTAIN) return 0;
    const uint32_t lastc = s.last_polarity == ELPIS_ECS_MEM_ASSERT   ? s.assert_count
                           : s.last_polarity == ELPIS_ECS_MEM_NEGATE ? s.negate_count
                                                               : s.uncertain_count;
    if (!lastc) return 0;
    if (s.first_tick > s.last_tick || s.last_tick > clock) return 0;
    if (s.assert_count ? (s.last_assert_tick < s.first_tick || s.last_assert_tick > s.last_tick)
                       : s.last_assert_tick != 0)
        return 0;
    if (s.negate_count ? (s.last_negate_tick < s.first_tick || s.last_negate_tick > s.last_tick)
                       : s.last_negate_tick != 0)
        return 0;
    relation_key_of(s.relation_type, &s.payload, s.participants, s.participant_count, key, &blocks);
    return !memcmp(key, s.key.b, 32);
}

typedef struct {
    elpis_ecs_mem_config cfg;
    uint64_t clock;
    uint32_t cursor, slots, ents;
    size_t banks_at, slots_at, sorder_at, ents_at, eorder_at;
} envelope;

/* Full validation; touches only the scratch workspace. */
static elpis_ecs_mem_status validate(elpis_ecs_mem *m, const uint8_t *b, size_t len, envelope *v) {
    if (len < ELPIS_ECS_MEM_HEADER_BYTES + ELPIS_ECS_MEM_DIGEST_BYTES) return ELPIS_ECS_MEM_CORRUPT;
    if (!header_config(b, &v->cfg) || !same_config(&v->cfg, &m->cfg)) return ELPIS_ECS_MEM_CORRUPT;
    v->clock = rd64(b + 64);
    v->cursor = rd32(b + 72);
    v->slots = rd32(b + 76);
    v->ents = rd32(b + 80);
    if (v->slots > m->cfg.slot_capacity || v->ents > m->cfg.codebook_capacity) return ELPIS_ECS_MEM_CORRUPT;
    if (v->slots ? v->cursor >= v->slots : v->cursor != 0) return ELPIS_ECS_MEM_CORRUPT;
    if (state_bytes_for(&m->cfg, v->slots, v->ents) != (uint64_t)len) return ELPIS_ECS_MEM_CORRUPT;
    uint8_t digest[32];
    elpis_sha256(b, len - ELPIS_ECS_MEM_DIGEST_BYTES, digest);
    if (memcmp(digest, b + len - ELPIS_ECS_MEM_DIGEST_BYTES, 32)) return ELPIS_ECS_MEM_CORRUPT;
    const size_t bank_values = 2u * (size_t)m->cfg.banks * m->cfg.dim * m->cfg.dim;
    v->banks_at = ELPIS_ECS_MEM_HEADER_BYTES;
    v->slots_at = v->banks_at + 4u * bank_values;
    v->sorder_at = v->slots_at + (size_t)v->slots * ELPIS_ECS_MEM_SLOT_RECORD_BYTES;
    v->ents_at = v->sorder_at + 4u * (size_t)v->slots;
    v->eorder_at = v->ents_at + (size_t)v->ents * ELPIS_ECS_MEM_CODEBOOK_RECORD_BYTES;
    int fast_content = 0, slow_content = 0;
    for (size_t i = 0; i < bank_values; ++i) {
        const int64_t w = (int32_t)rd32(b + v->banks_at + 4u * i);
        if (w > (int64_t)ELPIS_ECS_MEM_WEIGHT_LIMIT || w < -(int64_t)ELPIS_ECS_MEM_WEIGHT_LIMIT) return ELPIS_ECS_MEM_CORRUPT;
        if (w) *(i < bank_values / 2u ? &fast_content : &slow_content) = 1;
    }
    /* Unqualified content never enters a handle that does not admit it (checked after integrity): fast-bank
     * content needs the recurrent capability; slow-bank content and a moved slow-bank cursor are never admitted. */
    if ((fast_content && !(m->cfg.capabilities & ELPIS_ECS_MEM_CAP_RECURRENT)) || slow_content || v->cursor)
        return ELPIS_ECS_MEM_UNQUALIFIED;
    uint64_t probes = 0;
    const uint32_t smask = m->st.mask, emask = m->et.mask;
    memset(m->scratch_slot_index, 0, (size_t)(smask + 1u) * 4u);
    keyview sk = {b + v->slots_at, ELPIS_ECS_MEM_SLOT_RECORD_BYTES};
    for (uint32_t i = 0; i < v->slots; ++i) {
        const uint8_t *r = b + v->slots_at + (size_t)i * ELPIS_ECS_MEM_SLOT_RECORD_BYTES;
        if (!slot_record_ok(r, v->clock)) return ELPIS_ECS_MEM_CORRUPT;
        if (index_find(m->scratch_slot_index, smask, sk, r, &probes) != NONE) return ELPIS_ECS_MEM_CORRUPT;
        index_insert(m->scratch_slot_index, smask, sk, i, &probes, NULL);
    }
    if (!order_ok(b + v->sorder_at, v->slots, m->scratch_marks)) return ELPIS_ECS_MEM_CORRUPT;
    memset(m->scratch_ent_index, 0, (size_t)(emask + 1u) * 4u);
    keyview ek = {b + v->ents_at, ELPIS_ECS_MEM_CODEBOOK_RECORD_BYTES};
    for (uint32_t i = 0; i < v->ents; ++i) {
        const uint8_t *r = b + v->ents_at + (size_t)i * ELPIS_ECS_MEM_CODEBOOK_RECORD_BYTES;
        const uint64_t first = rd64(r + 32), last = rd64(r + 40);
        if (first > last || last > v->clock) return ELPIS_ECS_MEM_CORRUPT;
        if (index_find(m->scratch_ent_index, emask, ek, r, &probes) != NONE) return ELPIS_ECS_MEM_CORRUPT;
        index_insert(m->scratch_ent_index, emask, ek, i, &probes, NULL);
    }
    if (!order_ok(b + v->eorder_at, v->ents, m->scratch_marks)) return ELPIS_ECS_MEM_CORRUPT;
    return ELPIS_ECS_MEM_OK;
}
static void install(elpis_ecs_mem *m, const uint8_t *b, const envelope *v) {
    const size_t bank_values = (size_t)m->cfg.banks * m->cfg.dim * m->cfg.dim;
    for (size_t i = 0; i < bank_values; ++i) m->fast[i] = (int32_t)rd32(b + v->banks_at + 4u * i);
    for (size_t i = 0; i < bank_values; ++i) m->slow[i] = (int32_t)rd32(b + v->banks_at + 4u * (bank_values + i));
    memset(m->slots, 0, sizeof(elpis_ecs_mem_slot) * m->cfg.slot_capacity);
    for (uint32_t i = 0; i < v->slots; ++i)
        slot_decode(b + v->slots_at + (size_t)i * ELPIS_ECS_MEM_SLOT_RECORD_BYTES, &m->slots[i]);
    memset(m->ents, 0, sizeof(entity_rec) * m->cfg.codebook_capacity);
    for (uint32_t i = 0; i < v->ents; ++i) {
        const uint8_t *r = b + v->ents_at + (size_t)i * ELPIS_ECS_MEM_CODEBOOK_RECORD_BYTES;
        memcpy(m->ents[i].id.b, r, 32);
        m->ents[i].first_tick = rd64(r + 32);
        m->ents[i].last_tick = rd64(r + 40);
    }
    uint64_t probes = 0;
    table_init(&m->st, m->cfg.slot_capacity, m->st.prev, m->st.next, m->st.index);
    m->st.count = v->slots;
    for (uint32_t k = 0; k < v->slots; ++k) lru_push_tail(&m->st, rd32(b + v->sorder_at + 4u * k));
    for (uint32_t i = 0; i < v->slots; ++i) index_insert(m->st.index, m->st.mask, slot_keys(m), i, &probes, NULL);
    table_init(&m->et, m->cfg.codebook_capacity, m->et.prev, m->et.next, m->et.index);
    m->et.count = v->ents;
    for (uint32_t k = 0; k < v->ents; ++k) lru_push_tail(&m->et, rd32(b + v->eorder_at + 4u * k));
    for (uint32_t i = 0; i < v->ents; ++i) index_insert(m->et.index, m->et.mask, ent_keys(m), i, &probes, NULL);
    m->clock = v->clock;
}
elpis_ecs_mem_status elpis_ecs_mem_restore_into(elpis_ecs_mem *m, uint64_t lease, const uint8_t *bytes, size_t length) {
    if (!m) return ELPIS_ECS_MEM_INVALID;
    if (!enter_write(m)) return ELPIS_ECS_MEM_BUSY;
    elpis_ecs_mem_status s = ELPIS_ECS_MEM_OK;
    envelope v;
    if (m->lease != lease) s = ELPIS_ECS_MEM_LEASED;
    else if (!bytes) s = ELPIS_ECS_MEM_INVALID;
    else s = validate(m, bytes, length, &v);
    if (s != ELPIS_ECS_MEM_OK) {
        s = refuse(m, s);
        leave_write(m);
        return s;
    }
    install(m, bytes, &v);
    m->tel.restores++;
    leave_write(m);
    return ELPIS_ECS_MEM_OK;
}
elpis_ecs_mem_status elpis_ecs_mem_restore(const uint8_t *bytes, size_t length, uint64_t byte_budget,
                                           uint32_t capabilities, elpis_ecs_mem **out) {
    elpis_ecs_mem_config c;
    if (!bytes || !out) return ELPIS_ECS_MEM_INVALID;
    if (length < ELPIS_ECS_MEM_HEADER_BYTES + ELPIS_ECS_MEM_DIGEST_BYTES || !header_config(bytes, &c)) return ELPIS_ECS_MEM_CORRUPT;
    c.byte_budget = byte_budget;
    c.capabilities = capabilities;
    elpis_ecs_mem *m = NULL;
    elpis_ecs_mem_status s = elpis_ecs_mem_create(&c, &m);
    if (s != ELPIS_ECS_MEM_OK) return s;
    s = elpis_ecs_mem_restore_into(m, 0, bytes, length);
    if (s != ELPIS_ECS_MEM_OK) {
        elpis_ecs_mem_destroy(m);
        return s;
    }
    *out = m;
    return ELPIS_ECS_MEM_OK;
}

/* ------------------------------------------------------------------------------------------------------ */
/* Mechanical access to the encoding                                                                      */
/* ------------------------------------------------------------------------------------------------------ */

static int dim_ok(uint32_t dim) {
    return dim >= ELPIS_ECS_MEM_MIN_DIM && dim <= ELPIS_ECS_MEM_MAX_DIM && !(dim & (dim - 1u));
}
elpis_ecs_mem_status elpis_ecs_mem_entity_code(const elpis_ecs_mem_digest *entity, uint32_t dim, int8_t *out) {
    uint64_t bits[ELPIS_ECS_MEM_MAX_DIM / 64u], blocks = 0;
    if (!entity || !out || !dim_ok(dim)) return ELPIS_ECS_MEM_INVALID;
    entity_bits(entity, dim / 64u, bits, &blocks);
    for (uint32_t i = 0; i < dim; ++i) out[i] = bit_at(bits, i) ? -1 : 1;
    return ELPIS_ECS_MEM_OK;
}
elpis_ecs_mem_status elpis_ecs_mem_cue(const elpis_ecs_mem_request *q, uint32_t dim, uint32_t banks, int8_t *out_cue,
                           uint32_t *out_bank) {
    uint64_t bits[ELPIS_ECS_MEM_MAX_DIM / 64u], blocks = 0;
    if (!query_shape_ok(q) || !out_cue || !out_bank || !dim_ok(dim) || !banks || banks > ELPIS_ECS_MEM_MAX_BANKS)
        return ELPIS_ECS_MEM_INVALID;
    cue_bits(q->relation_type, q->target_role, q->target_ordinal, q->cue, q->cue_count, dim, bits, &blocks);
    for (uint32_t i = 0; i < dim; ++i) out_cue[i] = bit_at(bits, i) ? -1 : 1;
    *out_bank = bank_of(q->relation_type, q->target_role, q->target_ordinal, banks, &blocks);
    return ELPIS_ECS_MEM_OK;
}
