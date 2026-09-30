/* test_vector_concurrency.cpp - concurrent read-only dense search.
 * Run under ThreadSanitizer as well as the functional suite. */
#include "vector_test_support.h"

#include "elpis/vector_execution.h"
#include "elpis/vector_index.h"
#include "elpis/vector_result.h"

#include <atomic>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

static const uint32_t D = ELPIS_EMBEDDING_DIM;
static const char *kCorpusDg = "dddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddd";

static std::atomic<int> g_fail{0};
static std::mutex g_io;

static void report(const char *what, const std::string &detail) {
    std::lock_guard<std::mutex> lk(g_io);
    std::printf("  FAIL %s: %s\n", what, detail.c_str());
    g_fail.store(1);
}

int main(int argc, char **argv) {
    std::string base = argc > 1 ? argv[1] : "/tmp/elpis-vector-conc";
    vecfix::rmtree(base);
    vecfix::mkdirp(base);
    std::printf("R2 concurrent search suite, root=%s\n", base.c_str());

    /* Deliberately tight: four ~800 KiB shards against a 2 MiB ceiling, so
     * queries race each other through promotion and demotion. */
    const uint64_t warm = 2ull << 20;
    fms_ctx *fms = vecfix::make_fms(base + "/cold", warm);
    if (!fms) { std::printf("  FAIL fms\n"); return 1; }

    elpis_embedder *emb = nullptr;
    elpis_embedder_fixture_create(ELPIS_NORM_L2, &emb);
    elpis_embedding_profile prof;
    elpis_embedder_profile(emb, &prof);

    elpis_vector_index *ix = nullptr;
    if (elpis_vector_index_create(fms, &prof, kCorpusDg, &ix) != 0) { std::printf("  FAIL index\n"); return 1; }

    std::vector<vecfix::Corpus> shards(4);
    for (int i = 0; i < 4; i++) {
        char tag[32];
        std::snprintf(tag, sizeof tag, "conc-%d", i);
        shards[(size_t)i].build(tag, 384, emb);
        char sd[65];
        std::vector<uint8_t> img = vecfix::build_shard(shards[(size_t)i], kCorpusDg, sd);
        if (elpis_vector_index_add_shard_bytes(ix, img.data(), img.size(), nullptr) != 0) {
            std::printf("  FAIL admit %d: %s\n", i, elpis_vector_index_error(ix));
            return 1;
        }
    }

    /* Single-threaded reference results for four distinct queries. */
    struct Q { std::vector<float> v; std::string digest; };
    std::vector<Q> queries(4);
    for (int i = 0; i < 4; i++) {
        queries[(size_t)i].v = shards[(size_t)i].vec[(size_t)(i * 7 + 3)];
        elpis_vector_query q;
        std::memset(&q, 0, sizeof q);
        q.vector = queries[(size_t)i].v.data();
        q.dimensions = D;
        q.k = 8;
        std::vector<elpis_vector_hit> hits(q.k);
        uint32_t n = 0;
        if (elpis_vector_index_search(ix, &q, hits.data(), &n) != 0 || n != 8) {
            std::printf("  FAIL reference search %d\n", i);
            return 1;
        }
        char dg[65];
        elpis_vector_result_digest(hits.data(), n, dg);
        queries[(size_t)i].digest = dg;
    }

    /* Iteration count is overridable purely for runtime attribution: scaling it
     * and observing linear runtime distinguishes per-operation cost from a
     * fixed stall such as a residency deadline. The default is unchanged and is
     * what CTest runs. */
    const int threads = 4;
    int iters = 60;
    if (const char *e = std::getenv("ELPIS_VECTOR_CONC_ITERS")) {
        int v = std::atoi(e);
        if (v > 0 && v <= 10000) iters = v;
    }
    std::vector<std::thread> pool;
    for (int t = 0; t < threads; t++) {
        pool.emplace_back([&, t]() {
            for (int it = 0; it < iters; it++) {
                const Q &qq = queries[(size_t)((t + it) % 4)];
                elpis_vector_query q;
                std::memset(&q, 0, sizeof q);
                q.vector = qq.v.data();
                q.dimensions = D;
                q.k = 8;
                elpis_vector_hit hits[8];
                uint32_t n = 0;
                int rc = elpis_vector_index_search(ix, &q, hits, &n);
                if (rc != 0) { report("search", elpis_vector_index_error(ix)); return; }
                if (n != 8) { report("hit count", std::to_string(n)); return; }
                char dg[65];
                elpis_vector_result_digest(hits, n, dg);
                if (qq.digest != dg)
                    report("result digest diverged under concurrency", std::string(dg) + " != " + qq.digest);
                for (uint32_t i = 1; i < n; i++)
                    if (elpis_vector_hit_compare(&hits[i - 1], &hits[i]) > 0)
                        report("ordering", "hits out of order");
            }
        });
    }
    /* A pump thread demotes underneath the readers the whole time. */
    std::thread pumper([&]() {
        for (int i = 0; i < iters * threads; i++) {
            fms_pump(fms);
            fms_stats st;
            fms_get_stats(fms, &st);
            if (st.domain_bytes[FMS_DOM_RAM] > warm)
                report("ram ceiling", std::to_string(st.domain_bytes[FMS_DOM_RAM]));
            if (st.tier_bytes[FMS_HOT] != 0) report("hot residency", "HOT tier used");
        }
    });
    for (auto &th : pool) th.join();
    pumper.join();

    fms_stats st;
    fms_get_stats(fms, &st);
    if (st.pinned_bytes != 0) report("pin leak", std::to_string(st.pinned_bytes));
    if (st.inflight_ops != 0) report("inflight leak", std::to_string(st.inflight_ops));
    if (st.digest_failures != 0) report("digest failures", std::to_string(st.digest_failures));

    /* Parallel executor under residency pressure. A shard task whose WARM lease
     * is refused (LIMIT while other objects are pinned, BUSY while one moves)
     * returns ELPIS_EXEC_DEFER: it must not hold a pool thread while it waits,
     * and it must still produce the serial result. */
    auto digest_of = [&](elpis_vector_executor *vx, const Q &qq, std::string *dg_out) {
        elpis_vector_query q;
        std::memset(&q, 0, sizeof q);
        q.vector = qq.v.data();
        q.dimensions = D;
        q.k = 8;
        elpis_vector_hit hits[8];
        uint32_t n = 0;
        int rc = elpis_vector_executor_search(vx, &q, hits, &n);
        if (rc != 0) return rc;
        if (n != 8) return -1000 - static_cast<int>(n);
        char dg[65];
        elpis_vector_result_digest(hits, n, dg);
        *dg_out = dg;
        return 0;
    };
    {
        /* Deterministic: pin 1.5 MiB of the 2 MiB ceiling so no ~800 KiB shard
         * can become WARM, search, and hold the pin until every shard task has
         * deferred. Meanwhile unrelated CPU work must run on all four threads. */
        fms_id hog = 0;
        void *hog_ptr = nullptr;
        if (fms_register(fms, 0x7e57u, 3ull << 19, FMS_WARM, 0.0f, nullptr, &hog) < 0 ||
            static_cast<int>(fms_acquire(fms, hog, FMS_WARM, FMS_READ, &hog_ptr)) != FMS_WARM) {
            report("residency hog", "could not pin the WARM hog");
        } else {
            elpis_vector_executor *vx = nullptr;
            if (elpis_vector_executor_create(ix, 4, &vx) != 0) report("executor", "create");
            std::string got;
            int rc = -1;
            std::thread searcher([&]() { rc = digest_of(vx, queries[0], &got); });
            elpis_exec_metrics m{};
            for (int i = 0; i < 20000; i++) {
                elpis_vector_executor_metrics(vx, &m);
                if (m.deferred >= 4) break;
                std::this_thread::sleep_for(std::chrono::microseconds(100));
            }
            if (m.deferred < 4) report("deferral", "shard tasks did not defer under a pinned ceiling");
            if (m.running != 0) report("deferral", "a refused shard task occupies a pool thread");

            elpis_exec_config cfg{};
            cfg.workers = 4; cfg.capacity = 16; cfg.max_input_bytes = 8; cfg.max_output_bytes = 8;
            elpis_exec_runtime *other = nullptr;
            if (elpis_exec_create(&cfg, &other) != ELPIS_EXEC_OK) report("runtime", "create");
            auto echo = [](const elpis_exec_buffer *in, size_t, elpis_exec_buffer **out) {
                *out = elpis_exec_buffer_alloc(elpis_exec_buffer_size(in));
                if (!*out) return ELPIS_EXEC_INTERNAL;
                std::memcpy(elpis_exec_buffer_mutable_data(*out), elpis_exec_buffer_data(in),
                            elpis_exec_buffer_size(in));
                return ELPIS_EXEC_OK;
            };
            for (unsigned i = 0; other && i < 8; i++) {
                elpis_exec_task t{1, 0, i, 0, i, echo};
                elpis_exec_buffer *b = elpis_exec_buffer_alloc(8);
                std::memset(elpis_exec_buffer_mutable_data(b), static_cast<int>(i), 8);
                uint64_t seq = 0;
                if (elpis_exec_submit(other, &t, &b, &seq) != ELPIS_EXEC_OK) report("runtime", "submit");
            }
            for (unsigned i = 0; other && i < 8; i++) {
                elpis_exec_result res{};
                /* Well inside the 5 s residency deadline the hog still holds. */
                if (elpis_exec_take(other, 2000, &res) != ELPIS_EXEC_OK || res.status != ELPIS_EXEC_OK)
                    report("pool", "CPU work stalled behind deferred shard tasks");
                elpis_exec_buffer_release(res.output);
            }
            elpis_exec_destroy(other);

            fms_release(fms, hog);
            searcher.join();
            if (rc != 0) report("deferred search", std::to_string(rc) + " " + elpis_vector_index_error(ix));
            else if (got != queries[0].digest) report("deferred search digest", got + " != " + queries[0].digest);
            elpis_vector_executor_metrics(vx, &m);
            std::printf("executor under pinned ceiling: deferred=%llu completed=%llu\n",
                        (unsigned long long)m.deferred, (unsigned long long)m.completed);
            elpis_vector_executor_destroy(vx);
            fms_unregister(fms, hog);
        }
    }
    {
        /* Racing: one executor per querying thread on the shared pool, a pump
         * demoting underneath. Every result must equal the serial reference. */
        std::vector<elpis_vector_executor *> vx(static_cast<size_t>(threads), nullptr);
        for (auto &e : vx)
            if (elpis_vector_executor_create(ix, 4, &e) != 0) report("executor", "create");
        std::vector<std::thread> qs;
        for (int t = 0; t < threads; t++) {
            qs.emplace_back([&, t]() {
                for (int it = 0; it < iters; it++) {
                    const Q &qq = queries[(size_t)((t + it) % 4)];
                    std::string dg;
                    int rc = digest_of(vx[(size_t)t], qq, &dg);
                    if (rc != 0) { report("executor search", std::to_string(rc)); return; }
                    if (dg != qq.digest) report("executor digest diverged under concurrency", dg + " != " + qq.digest);
                }
            });
        }
        std::thread pump2([&]() {
            for (int i = 0; i < iters * threads; i++) {
                fms_pump(fms);
                fms_stats s2;
                fms_get_stats(fms, &s2);
                if (s2.domain_bytes[FMS_DOM_RAM] > warm) report("ram ceiling", std::to_string(s2.domain_bytes[FMS_DOM_RAM]));
                if (s2.tier_bytes[FMS_HOT] != 0) report("hot residency", "HOT tier used");
            }
        });
        for (auto &th : qs) th.join();
        pump2.join();
        uint64_t deferred = 0, completed = 0;
        for (auto *e : vx) {
            elpis_exec_metrics m{};
            elpis_vector_executor_metrics(e, &m);
            deferred += m.deferred; completed += m.completed;
            elpis_vector_executor_destroy(e);
        }
        std::printf("executor racing: shard tasks=%llu deferred=%llu\n",
                    (unsigned long long)completed, (unsigned long long)deferred);
        fms_get_stats(fms, &st);
        if (st.pinned_bytes != 0) report("pin leak", std::to_string(st.pinned_bytes));
    }

    uint32_t shard_n = 0;
    elpis_vector_index_list_shards(ix, nullptr, 0, &shard_n);
    if (shard_n != 4) report("shard count", std::to_string(shard_n));

    std::printf("threads=%d iters=%d promotions=%llu demotions=%llu bytes_promoted=%llu "
                "bytes_demoted=%llu ram_peak_ok=%s\n",
                threads, iters, (unsigned long long)st.promotions,
                (unsigned long long)st.demotions, (unsigned long long)st.bytes_promoted,
                (unsigned long long)st.bytes_demoted,
                st.domain_bytes[FMS_DOM_RAM] <= warm ? "yes" : "no");

    elpis_vector_index_destroy(ix);
    fms_get_stats(fms, &st);
    if (st.objects != 0) report("object leak", std::to_string(st.objects));
    fms_destroy(fms);
    elpis_embedder_destroy(emb);

    std::printf(g_fail.load() ? "RESULT: concurrency failures\n" : "RESULT: concurrent search clean\n");
    return g_fail.load();
}
