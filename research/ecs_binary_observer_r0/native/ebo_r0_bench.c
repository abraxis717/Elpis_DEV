/* ECS Binary Observation R0 direct-native performance driver (RESEARCH_ONLY, PERFORMANCE_ONLY).
 *
 * Matched native workloads on the real K1 library, no Python on the timed path:
 *   OFF = one canonical QUERY  (elpis_ecsg_k1_query_identity)
 *   ON  = the same QUERY + ebo_encode + ebo_ring_push   (the whole synchronous observer cost)
 * OFF and ON samples are interleaved pairwise in a fixed pseudo-random order. The ring is drained outside the
 * timed region (the consumer is not on the hot path; a push is always a real push, never the cheaper drop).
 * Also measured: the isolated encode / push / drop costs (tight loops) and the bare forward (disclosure only).
 *
 * usage: ebo_r0_bench <law blob> <raw samples out> <pairs> <rows>...
 * Raw per-call nanoseconds go to <raw samples out> (u32 rows, u32 pairs, then pairs x (off, on) as u64, per rows);
 * a JSON summary of fixed facts goes to stdout. */
#define _GNU_SOURCE
#include <math.h>
#include <sched.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#include "ebo_r0.h"
#include "elpis/ecsg_k1.h"

enum { DIM = 6, WIDTH = 36, MAX_ROWS = 4096, WARMUP = 2000 };

static uint64_t now_ns(void)
{
    struct timespec t;
    clock_gettime(CLOCK_MONOTONIC, &t);
    return (uint64_t)t.tv_sec * 1000000000ull + (uint64_t)t.tv_nsec;
}

static uint64_t lcg(uint64_t *z)
{
    *z = *z * 6364136223846793005ull + 1442695040888963407ull;
    return *z >> 33;
}

static int die(const char *what)
{
    fprintf(stderr, "ebo_r0_bench: %s\n", what);
    return 2;
}

int main(int argc, char **argv)
{
    static double x[MAX_ROWS * DIM];
    static double out[MAX_ROWS];
    static uint8_t slots[64][EBO_RECORD_BYTES];
    uint8_t blob[8192];
    uint8_t digest[32];
    uint8_t rec[EBO_RECORD_BYTES];
    double w[DIM * WIDTH];
    ebo_law law;
    ebo_ring ring;
    elpis_ecsg_k1 *s = NULL;
    FILE *f;
    FILE *raw;
    size_t blob_size;
    size_t pairs;
    size_t i;
    int a;
    uint64_t z = 0x45424f52u;
    uint64_t *samples;
    cpu_set_t cpus;

    if (argc < 5) {
        return die("usage: ebo_r0_bench <law blob> <raw out> <pairs> <rows>...");
    }
    f = fopen(argv[1], "rb");
    if (f == NULL) {
        return die("law blob unreadable");
    }
    blob_size = fread(blob, 1, sizeof blob, f);
    fclose(f);
    if (ebo_law_init(&law, blob, blob_size) != EBO_OK) {
        return die("law refused");
    }
    pairs = (size_t)strtoull(argv[3], NULL, 10);
    if (pairs == 0u || pairs > 10000000u) {
        return die("pairs");
    }
    samples = malloc(pairs * 2u * sizeof(uint64_t));
    if (samples == NULL) {
        return die("nomem");
    }
    raw = fopen(argv[2], "wb");
    if (raw == NULL) {
        return die("raw output");
    }
    CPU_ZERO(&cpus);
    CPU_SET(1, &cpus);
    (void)sched_setaffinity(0, sizeof cpus, &cpus); /* best effort; reported */

    for (i = 0u; i < DIM * WIDTH; ++i) {
        w[i] = 0.18 * sin(1.3 * (double)i + 0.7);
    }
    for (i = 0u; i < (size_t)MAX_ROWS * DIM; ++i) {
        x[i] = 0.5 * cos(0.77 * (double)i + 0.1);
    }
    if (elpis_ecsg_k1_create(DIM, WIDTH, MAX_ROWS, w, &s) != ELPIS_ECSG_K1_OK) {
        return die("K1 create");
    }
    if (ebo_ring_init(&ring, slots, 64u) != EBO_OK) {
        return die("ring");
    }

    printf("{\"dim\": %d, \"width\": %d, \"pairs\": %zu, \"encoder\": %u, \"law_tag\": \"%02x%02x%02x%02x\", "
           "\"affinity_cpu\": %d, \"rows\": [",
           DIM, WIDTH, pairs, (unsigned)law.encoder, law.tag[0], law.tag[1], law.tag[2], law.tag[3],
           sched_getcpu());
    for (a = 4; a < argc; ++a) {
        const size_t rows = (size_t)strtoull(argv[a], NULL, 10);
        elpis_ecsg_k1_counters c0;
        elpis_ecsg_k1_counters c1;
        uint64_t t0;
        uint64_t t1;
        uint64_t sink = 0u;
        size_t p;
        const size_t iso = 1000000u;
        double encode_ns;
        double push_ns;
        double drop_ns;
        double forward_ns;
        uint32_t hdr[2];
        size_t fwd_reps;
        if (rows == 0u || rows > MAX_ROWS) {
            return die("rows");
        }
        for (p = 0u; p < WARMUP; ++p) {
            (void)elpis_ecsg_k1_query_identity(s, DIM, x, rows, out, digest);
            (void)ebo_encode(&law, DIM, rows, x, out, digest, rec);
            (void)ebo_ring_push(&ring, rec);
            (void)ebo_ring_pop(&ring, rec);
        }
        (void)elpis_ecsg_k1_stats(s, &c0);
        for (p = 0u; p < pairs; ++p) {
            const int on_first = (int)(lcg(&z) & 1u);
            int arm;
            for (arm = 0; arm < 2; ++arm) {
                const int on = arm == 0 ? on_first : !on_first;
                t0 = now_ns();
                if (elpis_ecsg_k1_query_identity(s, DIM, x, rows, out, digest) != ELPIS_ECSG_K1_OK) {
                    return die("query refused");
                }
                if (on) {
                    if (ebo_encode(&law, DIM, rows, x, out, digest, rec) != EBO_OK ||
                        ebo_ring_push(&ring, rec) != EBO_OK) {
                        return die("observer refused");
                    }
                }
                t1 = now_ns();
                samples[2u * p + (on ? 1u : 0u)] = t1 - t0;
                if (on) {
                    (void)ebo_ring_pop(&ring, rec); /* the consumer, off the timed path */
                    sink += rec[40];
                }
            }
        }
        (void)elpis_ecsg_k1_stats(s, &c1);
        hdr[0] = (uint32_t)rows;
        hdr[1] = (uint32_t)pairs;
        if (fwrite(hdr, sizeof hdr, 1, raw) != 1u || fwrite(samples, sizeof(uint64_t), 2u * pairs, raw) != 2u * pairs) {
            return die("raw write");
        }
        /* isolated costs */
        (void)elpis_ecsg_k1_query_identity(s, DIM, x, rows, out, digest);
        t0 = now_ns();
        for (p = 0u; p < iso; ++p) {
            out[0] += 0.0 * (double)p; /* keep the loop honest without changing values */
            (void)ebo_encode(&law, DIM, rows, x, out, digest, rec);
            sink += rec[33];
        }
        t1 = now_ns();
        encode_ns = (double)(t1 - t0) / (double)iso;
        t0 = now_ns();
        for (p = 0u; p < iso; ++p) {
            (void)ebo_ring_push(&ring, rec);
            (void)ebo_ring_pop(&ring, rec);
        }
        t1 = now_ns();
        push_ns = (double)(t1 - t0) / (double)iso; /* push + pop together: an upper bound for push */
        while (ebo_ring_push(&ring, rec) == EBO_OK) {
        }
        t0 = now_ns();
        for (p = 0u; p < iso; ++p) {
            (void)ebo_ring_push(&ring, rec);
        }
        t1 = now_ns();
        drop_ns = (double)(t1 - t0) / (double)iso;
        while (ebo_ring_pop(&ring, rec) == EBO_OK) {
        }
        fwd_reps = rows >= 1024u ? 2000u : 200000u;
        t0 = now_ns();
        for (p = 0u; p < fwd_reps; ++p) {
            (void)elpis_ecsg_k1_forward(s, x, rows, out);
        }
        t1 = now_ns();
        forward_ns = (double)(t1 - t0) / (double)fwd_reps;
        printf("%s{\"rows\": %zu, \"k1_forward_calls_timed\": %llu, \"k1_learn_calls_timed\": %llu, "
               "\"encode_ns\": %.2f, \"push_pop_ns\": %.2f, \"drop_ns\": %.2f, \"bare_forward_ns\": %.2f, "
               "\"sink\": %llu}",
               a == 4 ? "" : ", ", rows, (unsigned long long)(c1.forward_calls - c0.forward_calls),
               (unsigned long long)(c1.learn_calls - c0.learn_calls), encode_ns, push_ns, drop_ns, forward_ns,
               (unsigned long long)(sink & 1u));
        fflush(stdout);
    }
    printf("], \"dropped\": %llu}\n", (unsigned long long)ebo_ring_dropped(&ring));
    fclose(raw);
    free(samples);
    elpis_ecsg_k1_destroy(&s);
    return 0;
}
