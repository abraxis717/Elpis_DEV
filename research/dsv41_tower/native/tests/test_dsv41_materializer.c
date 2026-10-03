/* DSV4.1 Native Materializer R1: codec, rows, expert ranges, lifecycle,
 * faults and races over real files (elpis/dsv41_materializer.h). */
#define _POSIX_C_SOURCE 200809L
#include "elpis/dsv41_materializer.h"
#include <assert.h>
#include <dirent.h>
#include <errno.h>
#include <fcntl.h>
#include <float.h>
#include <math.h>
#include <pthread.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <time.h>
#include <unistd.h>

#define FP8 ELPIS_DSV41_ROW_E4M3_E8M0_BF16
#define F32 ELPIS_DSV41_ROW_F32_LE
#define DIM 64u
#define STRIDE (DIM + DIM / 32u)
#define ROWS 200u
#define ROW_OFFSET 100u
#define FDIM 16u
#define FROWS 40u
#define LAYERS 2u
#define EXPERTS 3u            /* routed; index EXPERTS is shared */
#define ROLE 4000u
#define IMAGE (3u * ROLE)
#define EPAGE 1024u

/* ---- independent reference codec ----------------------------------------- */
/* Exact value in binary64, IEEE conversion, then BF16 RNE by explicit halves. */
static int reference_value(uint8_t code, uint8_t scale, uint32_t *bits) {
    if ((code & 127) == 127 || scale == 255) return 0;
    int e = (code >> 3) & 15, m = code & 7;
    double v = e ? ldexp(m + 8, e - 10) : ldexp(m, -9);
    v = ldexp(v, scale - 127);
    if (code & 128) v = -v;
    if (fabs(v) > FLT_MAX) return 0;
    float f = (float)v;
    assert((double)f == v); /* every E4M3 x E8M0 value is exactly representable */
    uint32_t b;
    memcpy(&b, &f, 4);
    uint32_t upper = b >> 16, lower = b & 0xffff;
    if (lower > 0x8000 || (lower == 0x8000 && (upper & 1))) upper++;
    *bits = upper << 16;
    float r; memcpy(&r, bits, 4);
    return isfinite(r);
}

static void test_codec_exhaustive(void) {
    uint8_t raw[32 + 1];
    float out[32];
    uint64_t accepted = 0, rejected = 0, flushed = 0;
    for (unsigned scale = 0; scale < 256; ++scale) for (unsigned code = 0; code < 256; ++code) {
        memset(raw, 0, sizeof raw);
        raw[7] = (uint8_t)code; raw[32] = (uint8_t)scale;
        uint32_t want;
        int ok = reference_value((uint8_t)code, (uint8_t)scale, &want);
        elpis_clock_code rc = elpis_dsv41_row_decode(FP8, raw, sizeof raw, 32, out);
        if (!ok) { assert(rc == ELPIS_CLOCK_ENCODING); ++rejected; continue; }
        assert(rc == ELPIS_CLOCK_OK);
        uint32_t got; memcpy(&got, &out[7], 4);
        assert(got == want);
        for (int i = 0; i < 32; ++i) if (i != 7) { uint32_t z; memcpy(&z, &out[i], 4); assert(z == 0); }
        flushed += want == 0 && code != 0 && code != 128;
        ++accepted;
    }
    assert(accepted + rejected == 65536);
    assert(rejected > 0 && flushed > 0); /* overflow rejections and BF16 subnormal flushes are exercised */
    /* All 256 codes in one row with distinct per-block scales (multi-block indexing). */
    uint8_t row[256 + 8];
    float values[256];
    for (unsigned i = 0; i < 256; ++i) row[i] = (uint8_t)(i == 127 || i == 255 ? 0 : i);
    for (unsigned j = 0; j < 8; ++j) row[256 + j] = (uint8_t)(100 + 5 * j);
    assert(elpis_dsv41_row_decode(FP8, row, sizeof row, 256, values) == ELPIS_CLOCK_OK);
    for (unsigned i = 0; i < 256; ++i) {
        uint32_t want, got; assert(reference_value(row[i], row[256 + i / 32], &want));
        memcpy(&got, &values[i], 4); assert(got == want);
    }
    /* Rejections and geometry. */
    row[255] = 0xff; assert(elpis_dsv41_row_decode(FP8, row, sizeof row, 256, values) == ELPIS_CLOCK_ENCODING);
    row[255] = 0x7f; assert(elpis_dsv41_row_decode(FP8, row, sizeof row, 256, values) == ELPIS_CLOCK_ENCODING);
    row[255] = 0; row[256 + 3] = 255; assert(elpis_dsv41_row_decode(FP8, row, sizeof row, 256, values) == ELPIS_CLOCK_ENCODING);
    assert(elpis_dsv41_row_decode(FP8, row, sizeof row - 1, 256, values) == ELPIS_CLOCK_ENCODING); /* row length */
    assert(elpis_dsv41_row_decode(FP8, row, 33, 31, values) == ELPIS_CLOCK_INVALID);              /* scale blocks */
    assert(elpis_dsv41_row_decode(7, row, 4, 1, values) == ELPIS_CLOCK_INVALID);
    assert(elpis_dsv41_row_bytes(FP8, 64) == 66 && elpis_dsv41_row_bytes(F32, 3) == 12 && !elpis_dsv41_row_bytes(FP8, 33));
    /* F32 LE: exact bits, subnormals/negative zero kept, NaN/Inf rejected. */
    uint32_t special[6] = {0x00000001u, 0x80000000u, 0x7f7fffffu, 0xff7fffffu, 0x3f800001u, 0x00800000u};
    uint8_t le[24]; float f[6];
    for (unsigned i = 0; i < 6; ++i) for (unsigned k = 0; k < 4; ++k) le[4 * i + k] = (uint8_t)(special[i] >> (8 * k));
    assert(elpis_dsv41_row_decode(F32, le, 24, 6, f) == ELPIS_CLOCK_OK);
    for (unsigned i = 0; i < 6; ++i) { uint32_t b; memcpy(&b, &f[i], 4); assert(b == special[i]); }
    uint32_t bad[3] = {0x7f800000u, 0xff800000u, 0x7fc00001u};
    for (unsigned i = 0; i < 3; ++i) {
        for (unsigned k = 0; k < 4; ++k) le[4 + k] = (uint8_t)(bad[i] >> (8 * k));
        assert(elpis_dsv41_row_decode(F32, le, 24, 6, f) == ELPIS_CLOCK_ENCODING);
    }
    printf("codec exhaustive: accepted=%llu rejected=%llu bf16_flushed=%llu\n",
           (unsigned long long)accepted, (unsigned long long)rejected, (unsigned long long)flushed);
}

/* ---- fixture files -------------------------------------------------------- */
static uint8_t rows_file[ROW_OFFSET + ROWS * STRIDE + 37];
static uint8_t frows_file[FROWS * FDIM * 4];
static uint8_t experts_a[LAYERS * (EXPERTS + 1) * 2 * ROLE + 512], experts_b[LAYERS * (EXPERTS + 1) * ROLE];
static uint8_t images[LAYERS][EXPERTS + 1][IMAGE];

static void write_file(const char *name, const uint8_t *data, size_t n) {
    FILE *f = fopen(name, "wb");
    assert(f && fwrite(data, 1, n, f) == n && fclose(f) == 0);
}
static void build_files(void) {
    uint32_t x = 12345;
    for (size_t i = 0; i < sizeof rows_file; ++i) { x = x * 1664525u + 1013904223u; rows_file[i] = (uint8_t)(x >> 24); }
    for (unsigned r = 0; r < ROWS; ++r) {
        uint8_t *p = rows_file + ROW_OFFSET + r * STRIDE;
        for (unsigned i = 0; i < DIM; ++i) { x = x * 1664525u + 1013904223u; p[i] = (uint8_t)((x >> 24) % 96) | (uint8_t)((x >> 8) & 128); }
        for (unsigned j = 0; j < DIM / 32; ++j) p[DIM + j] = (uint8_t)(110 + (r + j) % 20);
        if (r == 3) p[DIM] = 0; /* subnormal flush row */
    }
    rows_file[ROW_OFFSET + 150 * STRIDE + 9] = 0x7f;                       /* invalid FP8 code */
    rows_file[ROW_OFFSET + 151 * STRIDE + DIM + 1] = 255;                  /* invalid E8M0 scale */
    rows_file[ROW_OFFSET + 152 * STRIDE] = 0x7e; rows_file[ROW_OFFSET + 152 * STRIDE + DIM] = 254; /* overflow */
    for (unsigned i = 0; i < FROWS * FDIM; ++i) {
        float v = (float)i * 0.25f - 7.0f;
        uint32_t b; memcpy(&b, &v, 4);
        if (i == FDIM * 7 + 3) b = 0x7fc00000u;                              /* NaN in F32 row 7 */
        for (unsigned k = 0; k < 4; ++k) frows_file[4 * i + k] = (uint8_t)(b >> (8 * k));
    }
    /* w1, w2 in asset A (interleaved, offset 512), w3 in asset B. */
    for (unsigned l = 0; l < LAYERS; ++l) for (unsigned e = 0; e <= EXPERTS; ++e) {
        unsigned k = l * (EXPERTS + 1) + e;
        for (unsigned r = 0; r < 3; ++r) for (unsigned i = 0; i < ROLE; ++i)
            images[l][e][r * ROLE + i] = (uint8_t)(i * 7u + r * 31u + k * 101u + (i >> 8));
        memcpy(experts_a + 512 + (2 * k) * ROLE, images[l][e], ROLE);
        memcpy(experts_a + 512 + (2 * k + 1) * ROLE, images[l][e] + 2 * ROLE, ROLE);
        memcpy(experts_b + k * ROLE, images[l][e] + ROLE, ROLE);
    }
    write_file("rows.bin", rows_file, sizeof rows_file);
    write_file("frows.bin", frows_file, sizeof frows_file);
    write_file("experts_a.bin", experts_a, sizeof experts_a);
    write_file("experts_b.bin", experts_b, sizeof experts_b);
}
static int open_fds(void) {
    DIR *d = opendir("/proc/self/fd");
    int n = 0;
    assert(d);
    while (readdir(d)) ++n;
    closedir(d);
    return n;
}

typedef struct { elpis_dsv41_materializer id; elpis_dsv41_materializer_v1 t; uint8_t bank[32], fbank[32]; } rig;

static uint32_t admit(elpis_dsv41_materializer id, const char *name, uint32_t page, int corrupt_page) {
    int fd = open(name, O_RDONLY | O_CLOEXEC);
    assert(fd >= 0);
    struct stat st;
    assert(fstat(fd, &st) == 0);
    uint64_t size = (uint64_t)st.st_size, count = (size + page - 1) / page;
    uint8_t *map = malloc(count * 32), *buf = malloc(page);
    for (uint64_t p = 0; p < count; ++p) {
        size_t n = (size_t)(p + 1 == count ? size - p * page : page);
        assert(pread(fd, buf, n, (off_t)(p * page)) == (ssize_t)n);
        elpis_file_service_raw_digest(buf, n, map + p * 32);
    }
    if (corrupt_page >= 0) map[corrupt_page * 32] ^= 1; /* wrong pinned page digest */
    elpis_file_asset_v1 a = {ELPIS_FILE_SERVICE_ABI_V1, page, fd, 0, size, count, {0}, map};
    a.stamp = (elpis_file_stamp){(uint64_t)st.st_dev, (uint64_t)st.st_ino, size,
        (int64_t)st.st_mtim.tv_sec * 1000000000 + st.st_mtim.tv_nsec,
        (int64_t)st.st_ctim.tv_sec * 1000000000 + st.st_ctim.tv_nsec};
    uint32_t asset = 0;
    assert(elpis_dsv41_materializer_admit_asset(id, &a, &asset) == ELPIS_CLOCK_OK);
    close(fd); free(map); free(buf);
    return asset;
}
static elpis_dsv41_materializer_config_v1 config(void) {
    elpis_dsv41_materializer_config_v1 c = {ELPIS_DSV41_MATERIALIZER_ABI_V1, 0,
        {ELPIS_FILE_SERVICE_ABI_V1, 0, 16 * EPAGE, 4 * 4096, 1 << 20, 64, 8, 4, 0},
        LAYERS, EXPERTS, IMAGE, 6000, 2, 64, DIM, 0};
    return c;
}
static void setup(rig *r, int corrupt) {
    memset(r, 0, sizeof *r);
    elpis_dsv41_materializer_config_v1 c = config();
    assert(elpis_dsv41_materializer_create(&c, &r->id) == ELPIS_CLOCK_OK);
    uint32_t rows = admit(r->id, "rows.bin", 4096, -1), frows = admit(r->id, "frows.bin", 512, -1);
    uint32_t a = admit(r->id, "experts_a.bin", EPAGE, corrupt ? 9 : -1), b = admit(r->id, "experts_b.bin", EPAGE, -1);
    memset(r->bank, 0xab, 32); memset(r->fbank, 0xcd, 32);
    elpis_dsv41_bank_v1 bank = {1, DIM, FP8, rows, ROWS, ROW_OFFSET, {0}, 64, 0, 1 << 20};
    memcpy(bank.bank, r->bank, 32);
    assert(elpis_dsv41_materializer_add_bank(r->id, &bank) == ELPIS_CLOCK_OK);
    assert(elpis_dsv41_materializer_add_bank(r->id, &bank) == ELPIS_CLOCK_INVALID);       /* one per layer */
    elpis_dsv41_bank_v1 fb = {0, FDIM, F32, frows, FROWS, 0, {0}, 8, 0, 8 * FDIM * 4};
    memcpy(fb.bank, r->fbank, 32);
    elpis_dsv41_bank_v1 wrong = fb; wrong.rows = FROWS + 1;
    assert(elpis_dsv41_materializer_add_bank(r->id, &wrong) == ELPIS_CLOCK_INVALID);      /* table range */
    assert(elpis_dsv41_materializer_add_bank(r->id, &fb) == ELPIS_CLOCK_OK);
    for (unsigned l = 0; l < LAYERS; ++l) for (unsigned e = 0; e <= EXPERTS; ++e) {
        if (l == 1 && e == 1) continue; /* resident expert: never staged */
        unsigned k = l * (EXPERTS + 1) + e;
        elpis_dsv41_expert_v1 x = {l, e, {a, b, a}, 0, {512 + 2 * k * ROLE, k * ROLE, 512 + (2 * k + 1) * ROLE},
                                   {ROLE, ROLE, ROLE}, {0}};
        for (unsigned i = 0; i < 96; ++i) x.digests[i] = (uint8_t)(k * 3 + i);
        if (!l && !e) {
            elpis_dsv41_expert_v1 bad = x; bad.sizes[2] = ROLE - 1;
            assert(elpis_dsv41_materializer_add_expert(r->id, &bad) == ELPIS_CLOCK_INVALID); /* exact image geometry */
            bad = x; bad.offsets[1] = sizeof experts_b - ROLE + 1;
            assert(elpis_dsv41_materializer_add_expert(r->id, &bad) == ELPIS_CLOCK_INVALID); /* asset range */
        }
        assert(elpis_dsv41_materializer_add_expert(r->id, &x) == ELPIS_CLOCK_OK);
    }
    elpis_dsv41_materializer_bind((void *)(uintptr_t)r->id, &r->t);
    elpis_clock_span span = {0};
    uint64_t ids[1] = {0};
    assert(r->t.rows(r->t.context, 1, r->bank, ids, 1, DIM, &span) == ELPIS_CLOCK_INVALID); /* unsealed */
    assert(elpis_dsv41_materializer_seal(r->id) == ELPIS_CLOCK_OK);
    assert(elpis_dsv41_materializer_add_expert(r->id, NULL) == ELPIS_CLOCK_INVALID);
    elpis_dsv41_bank_v1 late = fb; late.layer = 2 % LAYERS;
    assert(elpis_dsv41_materializer_add_bank(r->id, &late) == ELPIS_CLOCK_CLOSED);           /* sealed */
    assert(elpis_dsv41_materializer_seal(r->id) == ELPIS_CLOCK_CLOSED);
}
static elpis_dsv41_materializer_stats_v1 stats(const rig *r) {
    elpis_dsv41_materializer_stats_v1 s;
    assert(elpis_dsv41_materializer_stats(r->id, &s) == ELPIS_CLOCK_OK);
    return s;
}
static void closed(const rig *r) {
    elpis_dsv41_materializer_stats_v1 s = stats(r);
    assert(s.live_spans == 0 && s.file.pinned_bytes == 0 && s.file.live_ranges == 0);
    assert(s.file.lease_acquires == s.file.lease_releases);
}
static void teardown(rig *r) {
    closed(r);
    assert(elpis_dsv41_materializer_destroy(r->id) == ELPIS_CLOCK_OK);
    assert(elpis_dsv41_materializer_destroy(r->id) == ELPIS_CLOCK_OK); /* idempotent */
    elpis_dsv41_materializer_stats_v1 s;
    assert(elpis_dsv41_materializer_stats(r->id, &s) == ELPIS_CLOCK_STALE);
    elpis_clock_span span = {0};
    uint64_t ids[1] = {0};
    assert(r->t.rows(r->t.context, 1, r->bank, ids, 1, DIM, &span) == ELPIS_CLOCK_STALE);
    assert(r->t.expert(r->t.context, 0, 0, 0, 1, &span) == ELPIS_CLOCK_STALE);
    span.lease = (void *)1; r->t.release(r->t.context, &span); assert(!span.lease);
    r->t.quiesce(r->t.context);
}

static void expect_row(uint32_t row, const float *got) {
    float want[DIM];
    assert(elpis_dsv41_row_decode(FP8, rows_file + ROW_OFFSET + row * STRIDE, STRIDE, DIM, want) == ELPIS_CLOCK_OK);
    assert(!memcmp(want, got, sizeof want));
}
static void test_rows(void) {
    rig r; setup(&r, 0);
    uint64_t ids[12] = {61, 3, 199, 61, 0, 62, 3, 3, 120, 0, 61, 1};
    elpis_clock_span span = {0};
    assert(r.t.rows(r.t.context, 1, r.bank, ids, 12, DIM, &span) == ELPIS_CLOCK_OK);
    assert(span.bytes == 12 * DIM * 4 && span.lease);
    for (unsigned i = 0; i < 12; ++i) expect_row((uint32_t)ids[i], (const float *)span.data + i * DIM); /* request order */
    elpis_dsv41_materializer_stats_v1 s = stats(&r);
    assert(s.unique_rows == 7 && s.row_requests == 12 && s.file.semantic_bytes == 7 * STRIDE && s.live_spans == 1);
    elpis_clock_span other = {0};
    assert(r.t.rows(r.t.context, 1, r.bank, ids, 1, DIM, &other) == ELPIS_CLOCK_BUSY);      /* one borrowed span */
    assert(r.t.expert(r.t.context, 0, 0, 0, 8, &other) == ELPIS_CLOCK_BUSY && !other.lease);
    elpis_clock_span copy = span;
    r.t.release(r.t.context, &span);
    assert(!span.data && !span.lease);
    r.t.release(r.t.context, &copy);                                                        /* exactly once */
    s = stats(&r);
    assert(s.span_acquires == 1 && s.span_releases == 1 && s.stale_releases == 1);
    uint8_t bank[32]; memcpy(bank, r.bank, 32); bank[31] ^= 1;
    assert(r.t.rows(r.t.context, 1, bank, ids, 2, DIM, &span) == ELPIS_CLOCK_INTEGRITY);    /* bank identity */
    assert(r.t.rows(r.t.context, 1, r.bank, ids, 2, DIM - 32, &span) == ELPIS_CLOCK_INTEGRITY);
    assert(r.t.rows(r.t.context, 1, r.bank, ids, 65, DIM, &span) == ELPIS_CLOCK_LIMIT);     /* row batch */
    uint64_t outside[2] = {5, ROWS};
    assert(r.t.rows(r.t.context, 1, r.bank, outside, 2, DIM, &span) == ELPIS_CLOCK_INVALID); /* row outside bank */
    assert(r.t.rows(r.t.context, 5, r.fbank, ids, 1, FDIM, &span) == ELPIS_CLOCK_INTEGRITY);   /* no bank */
    for (uint64_t bad = 150; bad <= 152; ++bad) {
        uint64_t req[3] = {4, bad, 5};
        assert(r.t.rows(r.t.context, 1, r.bank, req, 3, DIM, &span) == ELPIS_CLOCK_ENCODING); /* code/scale/overflow */
        closed(&r);
    }
    uint64_t z[1] = {3};
    assert(r.t.rows(r.t.context, 1, r.bank, z, 1, DIM, &span) == ELPIS_CLOCK_OK);           /* BF16 flush row */
    expect_row(3, (const float *)span.data);
    r.t.release(r.t.context, &span);
    /* F32_LE bank with per-engine bounds. */
    uint64_t f[3] = {39, 0, 39};
    assert(r.t.rows(r.t.context, 0, r.fbank, f, 3, FDIM, &span) == ELPIS_CLOCK_OK);
    assert(!memcmp(span.data, frows_file + 39 * FDIM * 4, FDIM * 4) && !memcmp((const uint8_t *)span.data + FDIM * 4, frows_file, FDIM * 4));
    r.t.release(r.t.context, &span);
    uint64_t nine[9] = {0};
    assert(r.t.rows(r.t.context, 0, r.fbank, nine, 9, FDIM, &span) == ELPIS_CLOCK_LIMIT);
    uint64_t nan_row[1] = {7};
    assert(r.t.rows(r.t.context, 0, r.fbank, nan_row, 1, FDIM, &span) == ELPIS_CLOCK_ENCODING); /* non-finite */
    assert(r.t.rows(r.t.context, 1, r.bank, ids, 0, DIM, &span) == ELPIS_CLOCK_OK && span.bytes == 0);
    r.t.release(r.t.context, &span);
    teardown(&r);
}

static void expect_part(const rig *r, uint32_t l, uint32_t e, uint64_t off, size_t len) {
    elpis_clock_span span = {0};
    assert(r->t.expert(r->t.context, l, e, off, len, &span) == ELPIS_CLOCK_OK);
    assert(span.bytes == len && !memcmp(span.data, images[l][e] + off, len));
    unsigned k = l * (EXPERTS + 1) + e;
    for (unsigned i = 0; i < 96; ++i) assert(span.digests[i] == (uint8_t)(k * 3 + i));
    r->t.release(r->t.context, &span);
}
static void test_experts(void) {
    rig r; setup(&r, 0);
    uint64_t cuts[] = {0, 1, EPAGE - 1, EPAGE, ROLE - 1, ROLE, ROLE + 1, 2 * ROLE - 1, 2 * ROLE, 2 * ROLE + 1, IMAGE - 1};
    for (unsigned l = 0; l < LAYERS; ++l) for (unsigned e = 0; e <= EXPERTS; ++e) {
        if (l == 1 && e == 1) continue;
        for (unsigned i = 0; i < sizeof cuts / sizeof *cuts; ++i) {
            expect_part(&r, l, e, cuts[i], 1);                                              /* smallest part */
            if (cuts[i] + 6000 <= IMAGE) expect_part(&r, l, e, cuts[i], 6000);              /* crosses w1/w3, w3/w2 */
        }
        for (uint64_t off = 0; off < IMAGE; off += 1152) expect_part(&r, l, e, off, off + 1152 > IMAGE ? IMAGE - off : 1152);
    }
    elpis_clock_span span = {0};
    assert(r.t.expert(r.t.context, 0, 2, 0, IMAGE, &span) == ELPIS_CLOCK_LIMIT);   /* full image > staging budget */
    assert(r.t.expert(r.t.context, 1, 1, 0, 8, &span) == ELPIS_CLOCK_INTEGRITY);   /* resident expert */
    assert(r.t.expert(r.t.context, LAYERS, 0, 0, 8, &span) == ELPIS_CLOCK_INVALID);
    assert(r.t.expert(r.t.context, 0, EXPERTS + 1, 0, 8, &span) == ELPIS_CLOCK_INVALID);
    assert(r.t.expert(r.t.context, 0, 0, IMAGE - 4, 8, &span) == ELPIS_CLOCK_INVALID);
    assert(r.t.expert(r.t.context, 0, 0, 0, 0, &span) == ELPIS_CLOCK_INVALID);
    assert(!span.lease);
    elpis_dsv41_materializer_stats_v1 s = stats(&r);
    printf("experts: hw=%llu staging=%llu refusals=%llu\n", (unsigned long long)s.file.resident_high_water, (unsigned long long)s.staging_high_water, (unsigned long long)s.refusals);
    assert(s.file.resident_high_water <= 16 * EPAGE && s.staging_high_water == 6000 && s.refusals == 7); /* + unsealed probe */
    assert(s.file.hits > 0 && s.file.evictions > 0 && s.expert_chunks > s.expert_calls - 6);
    closed(&r);
    teardown(&r);
    /* Full canonical image with a staging budget that admits it. */
    elpis_dsv41_materializer_config_v1 c = config();
    c.staging_bytes = IMAGE;
    rig big; memset(&big, 0, sizeof big);
    assert(elpis_dsv41_materializer_create(&c, &big.id) == ELPIS_CLOCK_OK);
    uint32_t a = admit(big.id, "experts_a.bin", EPAGE, -1), b = admit(big.id, "experts_b.bin", EPAGE, -1);
    elpis_dsv41_expert_v1 x = {0, 0, {a, b, a}, 0, {512, 0, 512 + ROLE}, {ROLE, ROLE, ROLE}, {0}};
    for (unsigned i = 0; i < 96; ++i) x.digests[i] = (uint8_t)i;
    assert(elpis_dsv41_materializer_add_expert(big.id, &x) == ELPIS_CLOCK_OK);
    assert(elpis_dsv41_materializer_seal(big.id) == ELPIS_CLOCK_OK);
    elpis_dsv41_materializer_bind((void *)(uintptr_t)big.id, &big.t);
    expect_part(&big, 0, 0, 0, IMAGE);
    teardown(&big);
}

static int pread_mode;
static long fault_pread(int fd, void *buf, size_t n, int64_t off) {
    if (pread_mode == 1) return 0;                               /* short read */
    if (pread_mode == 2) { static int k; if (k++ % 2 == 0) { errno = EINTR; return -1; } }
    return (long)pread(fd, buf, n, (off_t)off);
}
static void test_faults(void) {
    rig r; setup(&r, 1);                                         /* experts_a page 9 digest is wrong */
    elpis_clock_span span = {0};
    expect_part(&r, 0, 0, 0, 1000);                              /* pages 0-1: verified */
    assert(r.t.expert(r.t.context, 0, 1, 1000, 6000, &span) == ELPIS_CLOCK_INTEGRITY); /* reaches page 9 */
    assert(!span.lease); closed(&r);
    teardown(&r);
    /* pread faults through the test hooks. */
    elpis_file_service_test_hooks h = {fault_pread, 0};
    elpis_file_service_set_test_hooks(&h);
    setup(&r, 0);
    uint64_t ids[2] = {10, 11};
    pread_mode = 1;
    assert(r.t.rows(r.t.context, 1, r.bank, ids, 2, DIM, &span) == ELPIS_CLOCK_IO); closed(&r);
    assert(r.t.expert(r.t.context, 0, 0, 0, 100, &span) == ELPIS_CLOCK_IO); closed(&r);
    pread_mode = 2;
    assert(r.t.rows(r.t.context, 1, r.bank, ids, 2, DIM, &span) == ELPIS_CLOCK_OK);
    expect_row(10, (const float *)span.data); r.t.release(r.t.context, &span);
    assert(stats(&r).file.interrupted_reads >= 1);
    pread_mode = 0;
    teardown(&r);
    /* Allocation failure at create and at admission. */
    elpis_dsv41_materializer_config_v1 c = config();
    elpis_dsv41_materializer id = 0;
    h.pread = NULL; h.fail_alloc_after = 1; elpis_file_service_set_test_hooks(&h);
    assert(elpis_dsv41_materializer_create(&c, &id) == ELPIS_CLOCK_LIMIT && !id);
    elpis_file_service_set_test_hooks(NULL);
    assert(elpis_dsv41_materializer_create(&c, &id) == ELPIS_CLOCK_OK);
    int fd = open("rows.bin", O_RDONLY);
    struct stat st; assert(fstat(fd, &st) == 0);
    uint8_t map[4 * 32] = {0};
    elpis_file_asset_v1 a = {ELPIS_FILE_SERVICE_ABI_V1, 4096, fd, 0, (uint64_t)st.st_size, 4, {0}, map};
    a.stamp = (elpis_file_stamp){(uint64_t)st.st_dev, (uint64_t)st.st_ino, (uint64_t)st.st_size,
        (int64_t)st.st_mtim.tv_sec * 1000000000 + st.st_mtim.tv_nsec, (int64_t)st.st_ctim.tv_sec * 1000000000 + st.st_ctim.tv_nsec};
    h.fail_alloc_after = 1; elpis_file_service_set_test_hooks(&h);
    uint32_t asset;
    int before = open_fds();
    assert(elpis_dsv41_materializer_admit_asset(id, &a, &asset) == ELPIS_CLOCK_LIMIT);
    assert(open_fds() == before);
    elpis_file_service_set_test_hooks(NULL);
    a.fd = 999;
    assert(elpis_dsv41_materializer_admit_asset(id, &a, &asset) == ELPIS_CLOCK_INVALID);    /* bad cold descriptor */
    close(fd);
    assert(elpis_dsv41_materializer_destroy(id) == ELPIS_CLOCK_OK);
    /* Changed and truncated assets after admission. */
    write_file("victim.bin", experts_a, sizeof experts_a);
    elpis_dsv41_materializer_config_v1 vc = config();
    rig v; memset(&v, 0, sizeof v);
    assert(elpis_dsv41_materializer_create(&vc, &v.id) == ELPIS_CLOCK_OK);
    uint32_t va = admit(v.id, "victim.bin", EPAGE, -1), vb = admit(v.id, "experts_b.bin", EPAGE, -1);
    elpis_dsv41_expert_v1 x = {0, 0, {va, vb, va}, 0, {512, 0, 512 + ROLE}, {ROLE, ROLE, ROLE}, {0}};
    for (unsigned i = 0; i < 96; ++i) x.digests[i] = (uint8_t)i;
    assert(elpis_dsv41_materializer_add_expert(v.id, &x) == ELPIS_CLOCK_OK);
    assert(elpis_dsv41_materializer_seal(v.id) == ELPIS_CLOCK_OK);
    elpis_dsv41_materializer_bind((void *)(uintptr_t)v.id, &v.t);
    assert(v.t.expert(v.t.context, 0, 0, 0, 100, &span) == ELPIS_CLOCK_OK); v.t.release(v.t.context, &span);
    struct timespec pause = {0, 20000000}; nanosleep(&pause, NULL);
    write_file("victim.bin", experts_a, sizeof experts_a);                  /* same bytes, new stamp */
    assert(v.t.expert(v.t.context, 0, 0, 0, 100, &span) == ELPIS_CLOCK_INTEGRITY);
    assert(truncate("victim.bin", 100) == 0);
    assert(v.t.expert(v.t.context, 0, 0, 9000, 100, &span) == ELPIS_CLOCK_INTEGRITY);
    closed(&v);
    teardown(&v);
}

static void test_lifecycle(void) {
    rig r; setup(&r, 0);
    int base = open_fds();
    elpis_clock_span span = {0};
    assert(r.t.expert(r.t.context, 0, 0, 0, 100, &span) == ELPIS_CLOCK_OK);
    assert(elpis_dsv41_materializer_destroy(r.id) == ELPIS_CLOCK_BUSY);    /* borrowed span outstanding */
    r.t.quiesce(r.t.context);                                               /* forced: no live span or lease */
    r.t.quiesce(r.t.context);                                               /* idempotent */
    elpis_dsv41_materializer_stats_v1 s = stats(&r);
    assert(s.forced_releases == 1 && s.quiesces == 2 && s.live_spans == 0);
    r.t.release(r.t.context, &span);                                        /* after quiesce: counted no-op */
    s = stats(&r);
    assert(s.stale_releases == 1 && s.span_releases == 0);
    closed(&r);
    assert(elpis_dsv41_materializer_quiesce(r.id) == ELPIS_CLOCK_OK);
    assert(elpis_dsv41_materializer_evict(r.id) == ELPIS_CLOCK_OK);
    size_t n = 1;
    assert(elpis_dsv41_materializer_pages(r.id, NULL, NULL, 0, &n) == ELPIS_CLOCK_OK && n == 0);
    expect_part(&r, 0, 0, 0, 100);                                         /* usable after quiesce */
    uint32_t assets[64]; uint64_t pages[64];
    assert(elpis_dsv41_materializer_pages(r.id, assets, pages, 64, &n) == ELPIS_CLOCK_OK && n == 1 && pages[0] == 0);
    uint64_t lat[8], res[8]; uint32_t kinds[8];
    assert(elpis_dsv41_materializer_samples(r.id, lat, res, kinds, 8, &n) == ELPIS_CLOCK_OK && n == 2 && kinds[1] == 2);
    assert(elpis_dsv41_materializer_destroy(r.id + 1000) == ELPIS_CLOCK_STALE); /* never issued */
    teardown(&r);
    assert(open_fds() == base - 4);                                        /* every owned descriptor closed */
    /* Bounded identity table; identities are never reused. */
    elpis_dsv41_materializer ids[64];
    elpis_dsv41_materializer_config_v1 c = config();
    for (int i = 0; i < 64; ++i) { ids[i] = 0; assert(elpis_dsv41_materializer_create(&c, &ids[i]) == ELPIS_CLOCK_OK); }
    elpis_dsv41_materializer extra = 0;
    assert(elpis_dsv41_materializer_create(&c, &extra) == ELPIS_CLOCK_LIMIT);
    for (int i = 0; i < 64; ++i) assert(elpis_dsv41_materializer_destroy(ids[i]) == ELPIS_CLOCK_OK);
    assert(elpis_dsv41_materializer_create(&c, &extra) == ELPIS_CLOCK_OK && extra > ids[63]);
    assert(elpis_dsv41_materializer_destroy(extra) == ELPIS_CLOCK_OK);
    c.max_banks = LAYERS + 1;
    assert(elpis_dsv41_materializer_create(&c, &extra) == ELPIS_CLOCK_INVALID);
}

/* ---- races ---------------------------------------------------------------- */
typedef struct { rig *r; int stale; unsigned ok, deferred; } racer;
static void *acquire_loop(void *arg) {
    racer *x = arg;
    for (unsigned i = 0; i < 300; ++i) {
        elpis_clock_span span = {0};
        elpis_clock_code rc = x->r->t.expert(x->r->t.context, 0, i % 4, (i * 97) % (IMAGE - 3000), 3000, &span);
        assert(rc == ELPIS_CLOCK_OK || rc == ELPIS_CLOCK_DEFER || rc == ELPIS_CLOCK_BUSY || (x->stale && rc == ELPIS_CLOCK_STALE));
        if (rc == ELPIS_CLOCK_OK) {
            /* Bytes may be invalidated by a racing quiesce; release stays exactly-once safe. */
            x->ok++; x->r->t.release(x->r->t.context, &span);
        } else { x->deferred++; assert(!span.lease); }
        uint64_t ids[3] = {i % 140, 7, i % 140};
        rc = x->r->t.rows(x->r->t.context, 1, x->r->bank, ids, 3, DIM, &span);
        assert(rc == ELPIS_CLOCK_OK || rc == ELPIS_CLOCK_DEFER || rc == ELPIS_CLOCK_BUSY || (x->stale && rc == ELPIS_CLOCK_STALE));
        if (rc == ELPIS_CLOCK_OK) x->r->t.release(x->r->t.context, &span);
    }
    return NULL;
}
static void *quiesce_loop(void *arg) {
    racer *x = arg;
    for (unsigned i = 0; i < 300; ++i) { x->r->t.quiesce(x->r->t.context); sched_yield(); }
    return NULL;
}
static void *destroy_loop(void *arg) {
    racer *x = arg;
    for (;;) {
        elpis_clock_code rc = elpis_dsv41_materializer_destroy(x->r->id);
        assert(rc == ELPIS_CLOCK_OK || rc == ELPIS_CLOCK_BUSY);
        if (rc == ELPIS_CLOCK_OK) return NULL;
        sched_yield();
    }
}
static void *independent(void *arg) {
    rig *r = arg;
    for (unsigned i = 0; i < 200; ++i) {
        expect_part(r, i % 2, (i % 2) ? 0 : i % 4, (i * 131) % (IMAGE - 2000), 2000);
        uint64_t ids[2] = {i % 140, (i * 7) % 140};
        elpis_clock_span span = {0};
        assert(r->t.rows(r->t.context, 1, r->bank, ids, 2, DIM, &span) == ELPIS_CLOCK_OK);
        expect_row((uint32_t)ids[0], (const float *)span.data); expect_row((uint32_t)ids[1], (const float *)span.data + DIM);
        r->t.release(r->t.context, &span);
    }
    return NULL;
}
static void test_races(void) {
    /* Cancellation/quiescence racing acquisition on one instance. */
    rig r; setup(&r, 0);
    racer x = {&r, 0, 0, 0};
    pthread_t a, b;
    assert(pthread_create(&a, NULL, acquire_loop, &x) == 0 && pthread_create(&b, NULL, quiesce_loop, &x) == 0);
    assert(pthread_join(a, NULL) == 0 && pthread_join(b, NULL) == 0);
    r.t.quiesce(r.t.context);
    closed(&r);
    printf("quiesce race: ok=%u deferred=%u forced=%llu\n", x.ok, x.deferred,
           (unsigned long long)stats(&r).forced_releases);
    /* Destroy racing in-flight calls: BUSY until joined, never freed under a call. */
    x.stale = 1;
    assert(pthread_create(&a, NULL, acquire_loop, &x) == 0 && pthread_create(&b, NULL, destroy_loop, &x) == 0);
    assert(pthread_join(a, NULL) == 0 && pthread_join(b, NULL) == 0);
    elpis_dsv41_materializer_stats_v1 s;
    assert(elpis_dsv41_materializer_stats(r.id, &s) == ELPIS_CLOCK_STALE);
    /* Independent instances share no mutable state. */
    rig p, q; setup(&p, 0); setup(&q, 0);
    assert(pthread_create(&a, NULL, independent, &p) == 0 && pthread_create(&b, NULL, independent, &q) == 0);
    assert(pthread_join(a, NULL) == 0 && pthread_join(b, NULL) == 0);
    elpis_dsv41_materializer_stats_v1 sp = stats(&p), sq = stats(&q);
    assert(sp.span_acquires == 400 && sq.span_acquires == 400 && sp.file.misses == sq.file.misses);
    teardown(&p); teardown(&q);
}

int main(void) {
    setvbuf(stdout, NULL, _IONBF, 0);
    assert(elpis_dsv41_materializer_abi_version() == ELPIS_DSV41_MATERIALIZER_ABI_V1);
    test_codec_exhaustive();
    build_files();
    test_rows();
    test_experts();
    test_faults();
    test_lifecycle();
    test_races();
    puts("dsv41 materializer: ok");
    return 0;
}
