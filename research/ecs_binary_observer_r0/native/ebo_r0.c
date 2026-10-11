/* ECS Binary Observation R0 observer (RESEARCH_ONLY). See ebo_r0.h and research/ecs_binary_observer_r0/ADR.md.
 *
 * Determinism: binary64 only, fixed operation order, compiled with -ffp-contract=off and without fast-math, so the
 * pure-Python reference (research/ecs_binary_observer_r0/law.py) reproduces every record bitwise. No libc call is
 * made on the observation path: copies are fixed-size builtins, and the library's only link dependency is the static
 * SHA-256 used once by ebo_law_init. */
#include "ebo_r0.h"

#include "elpis/sha256.h"

#define EBO_LAW_READY 0x45424f4cu /* "EBOL" */

static const uint64_t H64_K = 0x9e3779b97f4a7c15ull;
static const uint64_t H64_LANE[4] = {0x243f6a8885a308d3ull, 0x13198a2e03707344ull, 0xa4093822299f31d0ull,
                                     0x082efa98ec4e6c89ull};

static uint64_t get_u64(const uint8_t *p)
{
    uint64_t v = 0u;
    unsigned i;
    for (i = 0u; i < 8u; ++i) {
        v |= (uint64_t)p[i] << (8u * i);
    }
    return v;
}

static double get_f64(const uint8_t *p)
{
    uint64_t bits = get_u64(p);
    double v;
    __builtin_memcpy(&v, &bits, sizeof v);
    return v;
}

static uint64_t bits_of(double v)
{
    uint64_t bits;
    __builtin_memcpy(&bits, &v, sizeof bits);
    return bits;
}

static int finite_bits(uint64_t bits)
{
    return ((bits >> 52) & 0x7ffu) != 0x7ffu;
}

static void put_u16(uint8_t *p, uint64_t v)
{
    p[0] = (uint8_t)(v & 0xffu);
    p[1] = (uint8_t)((v >> 8) & 0xffu);
}

static void put_u64(uint8_t *p, uint64_t v)
{
    unsigned i;
    for (i = 0u; i < 8u; ++i) {
        p[i] = (uint8_t)((v >> (8u * i)) & 0xffu);
    }
}

static uint64_t rotl64(uint64_t v, unsigned r)
{
    return (v << r) | (v >> (64u - r));
}

static uint64_t mix64(uint64_t z)
{
    z = (z ^ (z >> 30)) * 0xbf58476d1ce4e5b9ull;
    z = (z ^ (z >> 27)) * 0x94d049bb133111ebull;
    return z ^ (z >> 31);
}

size_t ebo_law_blob_bytes(uint32_t encoder)
{
    if (encoder == EBO_ENCODER_GRAY4) {
        return 8u + 8u * EBO_GRAY4_THRESHOLDS;
    }
    if (encoder == EBO_ENCODER_SPROJ) {
        return 8u + 16u + 2u * EBO_PAYLOAD_BITS * EBO_SPROJ_TERMS + 8u * EBO_PAYLOAD_BITS;
    }
    return 0u;
}

size_t ebo_law_bytes(void)
{
    return sizeof(ebo_law);
}

size_t ebo_ring_bytes(void)
{
    return sizeof(ebo_ring);
}

/* Cold path: admit a canonical blob. The law is staged locally and installed only when every check passes. */
ebo_status ebo_law_init(ebo_law *law, const uint8_t *blob, size_t size)
{
    ebo_law staged;
    uint8_t digest[32];
    uint32_t encoder;
    size_t i;
    size_t k;
    if (law == NULL || blob == NULL || size < 8u) {
        return EBO_INVALID;
    }
    if (blob[0] != 'E' || blob[1] != 'B' || blob[2] != 'O' || blob[3] != 'L') {
        return EBO_LAW;
    }
    if ((uint32_t)blob[4] + 256u * (uint32_t)blob[5] != EBO_LAW_VERSION) {
        return EBO_VERSION;
    }
    encoder = blob[6];
    if (ebo_law_blob_bytes(encoder) == 0u || blob[7] != 0u) {
        return EBO_LAW;
    }
    if (size != ebo_law_blob_bytes(encoder)) {
        return EBO_INVALID;
    }
    __builtin_memset(&staged, 0, sizeof staged);
    staged.encoder = encoder;
    if (encoder == EBO_ENCODER_GRAY4) {
        for (i = 0u; i < EBO_GRAY4_THRESHOLDS; ++i) {
            staged.thresholds[i] = get_f64(blob + 8u + 8u * i);
            if (!finite_bits(bits_of(staged.thresholds[i])) || (i > 0u && !(staged.thresholds[i] > staged.thresholds[i - 1u]))) {
                return EBO_LAW;
            }
        }
    } else {
        const uint8_t *p = blob + 8u;
        staged.mu = get_f64(p);
        staged.inv_scale = get_f64(p + 8u);
        if (!finite_bits(bits_of(staged.mu)) || !finite_bits(bits_of(staged.inv_scale)) || !(staged.inv_scale > 0.0)) {
            return EBO_LAW;
        }
        p += 16u;
        for (k = 0u; k < EBO_PAYLOAD_BITS; ++k) {
            for (i = 0u; i < EBO_SPROJ_TERMS; ++i) {
                const uint8_t idx = p[k * EBO_SPROJ_TERMS + i];
                const uint8_t sg = p[EBO_PAYLOAD_BITS * EBO_SPROJ_TERMS + k * EBO_SPROJ_TERMS + i];
                if (idx >= EBO_WINDOW_ROWS || (sg != 0x01u && sg != 0xffu)) {
                    return EBO_LAW;
                }
                staged.index[k][i] = idx;
                staged.sign[k][i] = sg == 0x01u ? (int8_t)1 : (int8_t)-1;
            }
        }
        p += 2u * EBO_PAYLOAD_BITS * EBO_SPROJ_TERMS;
        for (k = 0u; k < EBO_PAYLOAD_BITS; ++k) {
            staged.tau[k] = get_f64(p + 8u * k);
            if (!finite_bits(bits_of(staged.tau[k]))) {
                return EBO_LAW;
            }
        }
    }
    elpis_sha256(blob, size, digest);
    for (i = 0u; i < 4u; ++i) {
        staged.tag[i] = digest[i];
    }
    staged.ready = EBO_LAW_READY;
    __builtin_memcpy(law, &staged, sizeof staged);
    return EBO_OK;
}

static uint64_t input_identity(size_t dim, size_t rows, const double *x, size_t window, int *finite)
{
    uint64_t lane[4];
    uint64_t bad = 0u;
    const size_t words = window * dim;
    size_t j;
    lane[0] = H64_LANE[0];
    lane[1] = H64_LANE[1];
    lane[2] = H64_LANE[2];
    lane[3] = H64_LANE[3];
    for (j = 0u; j < words; ++j) {
        const uint64_t w = bits_of(x[j]);
        bad |= (uint64_t)(((w >> 52) & 0x7ffu) == 0x7ffu);
        lane[j & 3u] = rotl64((lane[j & 3u] ^ w) * H64_K, 31u);
    }
    *finite = bad == 0u;
    {
        uint64_t h = mix64((uint64_t)dim ^ ((uint64_t)rows << 8) ^ ((uint64_t)window << 24));
        h = mix64(h ^ lane[0]);
        h = mix64(h ^ lane[1]);
        h = mix64(h ^ lane[2]);
        h = mix64(h ^ lane[3]);
        return h;
    }
}

static void encode_gray4(const ebo_law *law, const double *values, size_t window, uint8_t payload[32])
{
    size_t r;
    for (r = 0u; r < window; ++r) {
        const double v = values[r];
        unsigned b = 0u;
        unsigned k;
        for (k = 0u; k < EBO_GRAY4_THRESHOLDS; ++k) {
            b += (unsigned)(v >= law->thresholds[k]);   /* an exact tie goes to the upper level */
        }
        b ^= b >> 1u;                                    /* Gray code: adjacent levels differ in one bit */
        payload[r >> 1u] |= (uint8_t)(b << (4u * (r & 1u)));
    }
}

static void encode_sproj(const ebo_law *law, const double *values, size_t window, uint8_t payload[32])
{
    double u[EBO_WINDOW_ROWS];
    size_t r;
    size_t k;
    for (r = 0u; r < EBO_WINDOW_ROWS; ++r) {
        u[r] = r < window ? (values[r] - law->mu) * law->inv_scale : 0.0;   /* rows beyond the query: 0 */
    }
    for (k = 0u; k < EBO_PAYLOAD_BITS; ++k) {
        double s = 0.0;
        unsigned i;
        for (i = 0u; i < EBO_SPROJ_TERMS; ++i) {
            const double t = u[law->index[k][i]];
            s = s + (law->sign[k][i] > 0 ? t : -t);
        }
        payload[k >> 3u] |= (uint8_t)((unsigned)(s >= law->tau[k]) << (k & 7u));   /* tie: 1 */
    }
}

ebo_status ebo_encode(const ebo_law *law, size_t dim, size_t rows, const double *x, const double *values,
                      const uint8_t digest[EBO_DIGEST_BYTES], uint8_t record[EBO_RECORD_BYTES])
{
    uint8_t staged[EBO_RECORD_BYTES];
    size_t window;
    size_t r;
    int finite = 1;
    uint64_t ident;
    if (law == NULL || x == NULL || values == NULL || digest == NULL || record == NULL) {
        return EBO_INVALID;
    }
    if (law->ready != EBO_LAW_READY) {
        return EBO_LAW;
    }
    if (dim == 0u || dim > EBO_MAX_DIM || rows == 0u || rows > EBO_MAX_ROWS) {
        return EBO_SHAPE;
    }
    window = rows < EBO_WINDOW_ROWS ? rows : EBO_WINDOW_ROWS;
    for (r = 0u; r < window; ++r) {
        if (!finite_bits(bits_of(values[r]))) {
            return EBO_NONFINITE;
        }
    }
    ident = input_identity(dim, rows, x, window, &finite);
    if (!finite) {
        return EBO_NONFINITE;
    }
    __builtin_memset(staged, 0, sizeof staged);
    staged[0] = 'E';
    staged[1] = 'B';
    staged[2] = 'O';
    staged[3] = 'R';
    put_u16(staged + 4u, EBO_FORMAT_VERSION);
    staged[6] = (uint8_t)law->encoder;
    staged[7] = (uint8_t)((rows > EBO_WINDOW_ROWS ? EBO_FLAG_TRUNCATED : 0u) |
                          (rows < EBO_WINDOW_ROWS ? EBO_FLAG_PADDED : 0u));
    put_u16(staged + 8u, rows);
    staged[10] = (uint8_t)dim;
    staged[11] = (uint8_t)window;
    staged[12] = law->tag[0];
    staged[13] = law->tag[1];
    staged[14] = law->tag[2];
    staged[15] = law->tag[3];
    for (r = 0u; r < 8u; ++r) {
        staged[16u + r] = digest[r];
    }
    put_u64(staged + 24u, ident);
    if (law->encoder == EBO_ENCODER_GRAY4) {
        encode_gray4(law, values, window, staged + 32u);
    } else {
        encode_sproj(law, values, window, staged + 32u);
    }
    __builtin_memcpy(record, staged, EBO_RECORD_BYTES);
    return EBO_OK;
}

ebo_status ebo_record_check(const ebo_law *law, const uint8_t record[EBO_RECORD_BYTES])
{
    size_t rows;
    size_t window;
    unsigned flags;
    if (law == NULL || record == NULL) {
        return EBO_INVALID;
    }
    if (law->ready != EBO_LAW_READY) {
        return EBO_LAW;
    }
    if (record[0] != 'E' || record[1] != 'B' || record[2] != 'O' || record[3] != 'R') {
        return EBO_INVALID;
    }
    if ((unsigned)record[4] + 256u * (unsigned)record[5] != EBO_FORMAT_VERSION) {
        return EBO_VERSION;
    }
    if (record[6] != law->encoder || record[12] != law->tag[0] || record[13] != law->tag[1] ||
        record[14] != law->tag[2] || record[15] != law->tag[3]) {
        return EBO_LAW;
    }
    rows = (size_t)record[8] + 256u * (size_t)record[9];
    window = rows < EBO_WINDOW_ROWS ? rows : EBO_WINDOW_ROWS;
    flags = (rows > EBO_WINDOW_ROWS ? EBO_FLAG_TRUNCATED : 0u) | (rows < EBO_WINDOW_ROWS ? EBO_FLAG_PADDED : 0u);
    if (rows == 0u || rows > EBO_MAX_ROWS || record[10] == 0u || record[10] > EBO_MAX_DIM ||
        record[11] != window || record[7] != flags) {
        return EBO_INVALID;
    }
    if (law->encoder == EBO_ENCODER_GRAY4) {   /* padded nibbles are zero */
        size_t r;
        for (r = window; r < EBO_WINDOW_ROWS; ++r) {
            if (((record[32u + (r >> 1u)] >> (4u * (r & 1u))) & 0x0fu) != 0u) {
                return EBO_INVALID;
            }
        }
    }
    return EBO_OK;
}

/* --- bounded nonblocking handoff (single producer, single consumer) --------------------------------------------- */

ebo_status ebo_ring_init(ebo_ring *ring, uint8_t (*slots)[EBO_RECORD_BYTES], uint64_t capacity)
{
    if (ring == NULL || slots == NULL || capacity == 0u || capacity > 65536u || (capacity & (capacity - 1u)) != 0u) {
        return EBO_INVALID;
    }
    __builtin_memset(ring, 0, sizeof *ring);
    ring->slots = slots;
    ring->mask = capacity - 1u;
    return EBO_OK;
}

ebo_status ebo_ring_push(ebo_ring *ring, const uint8_t record[EBO_RECORD_BYTES])
{
    uint64_t tail;
    uint64_t head;
    if (ring == NULL || record == NULL || ring->slots == NULL) {
        return EBO_INVALID;
    }
    tail = __atomic_load_n(&ring->tail, __ATOMIC_RELAXED);
    head = __atomic_load_n(&ring->head, __ATOMIC_ACQUIRE);
    if (tail - head > ring->mask) {
        __atomic_store_n(&ring->dropped, __atomic_load_n(&ring->dropped, __ATOMIC_RELAXED) + 1u, __ATOMIC_RELAXED);
        return EBO_DROPPED;
    }
    __builtin_memcpy(ring->slots[tail & ring->mask], record, EBO_RECORD_BYTES);
    __atomic_store_n(&ring->tail, tail + 1u, __ATOMIC_RELEASE);
    __atomic_store_n(&ring->pushed, __atomic_load_n(&ring->pushed, __ATOMIC_RELAXED) + 1u, __ATOMIC_RELAXED);
    return EBO_OK;
}

ebo_status ebo_ring_pop(ebo_ring *ring, uint8_t record[EBO_RECORD_BYTES])
{
    uint64_t head;
    uint64_t tail;
    if (ring == NULL || record == NULL || ring->slots == NULL) {
        return EBO_INVALID;
    }
    head = __atomic_load_n(&ring->head, __ATOMIC_RELAXED);
    tail = __atomic_load_n(&ring->tail, __ATOMIC_ACQUIRE);
    if (head == tail) {
        return EBO_EMPTY;
    }
    __builtin_memcpy(record, ring->slots[head & ring->mask], EBO_RECORD_BYTES);
    __atomic_store_n(&ring->head, head + 1u, __ATOMIC_RELEASE);
    return EBO_OK;
}

uint64_t ebo_ring_pushed(const ebo_ring *ring)
{
    return ring == NULL ? 0u : __atomic_load_n(&ring->pushed, __ATOMIC_RELAXED);
}

uint64_t ebo_ring_dropped(const ebo_ring *ring)
{
    return ring == NULL ? 0u : __atomic_load_n(&ring->dropped, __ATOMIC_RELAXED);
}
