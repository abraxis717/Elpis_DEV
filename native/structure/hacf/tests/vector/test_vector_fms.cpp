/* test_vector_fms.cpp - Gate HACF vector.4 FMS residency suite.
 *
 * Every context here is built with hot_absent_policy = FMS_REJECT. If any HACF vector
 * code path ever requested FMS_HOT it would return FMS_E_UNSUPPORTED instead of
 * quietly folding down, so "no HOT request occurs in HACF vector" is enforced by the
 * configuration rather than asserted in prose.
 *
 * FMS ABI v3: residency is volatile (WARM in RAM, HOT never requested here). There is no COLD
 * tier: a shard beyond the WARM ceiling is refused, never spilled, and no case creates, writes or
 * reads a file. */
#include "vector_test_support.h"

#include "elpis/vector_index.h"
#include "elpis/vector_result.h"

#include <chrono>
#include <cstdio>
#include <cstring>
#include <string>
#include <thread>
#include <vector>

static int fails = 0, checks = 0;
static const char *cur = "?";
#define CHECK(cond, ...) do { checks++; if (!(cond)) { \
        std::printf("  FAIL [%s] %s:%d ", cur, __FILE__, __LINE__); std::printf(__VA_ARGS__); \
        std::putchar('\n'); fails++; } } while (0)
#define CASE(name) do { cur = (name); std::printf("- %s\n", cur); } while (0)

static const uint32_t D = ELPIS_EMBEDDING_DIM;

static const char *kCorpusDg = "cccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc";

/* One 512-vector shard is ~800 KiB of payload. */
static const uint32_t kShardVectors = 512;

static void case_registration_and_lifecycle() {
    CASE("shard is an FMS object, leased at WARM, released on completion");
    fms_ctx *fms = vecfix::make_fms("life", 8ull << 20);
    CHECK(fms != nullptr, "fms");
    if (!fms) return;
    CHECK(fms_hot_available(fms) == 0, "posix PAL must not offer a HOT tier");

    elpis_embedder *emb = nullptr;
    elpis_embedder_fixture_create(ELPIS_NORM_L2, &emb);
    elpis_embedding_profile prof;
    elpis_embedder_profile(emb, &prof);
    elpis_vector_index *ix = nullptr;
    CHECK(elpis_vector_index_create(fms, &prof, kCorpusDg, &ix) == 0, "index");

    vecfix::Corpus c;
    c.build("life", kShardVectors, emb);
    char sd[65];
    std::vector<uint8_t> img = vecfix::build_shard(c, kCorpusDg, sd);
    CHECK(elpis_vector_index_add_shard_bytes(ix, img.data(), img.size(), nullptr) == 0,
          "admit: %s", elpis_vector_index_error(ix));

    fms_id id = 0;
    CHECK(elpis_vector_index_shard_object(ix, sd, &id) == 0, "shard has no FMS object");
    fms_object_info info;
    CHECK(fms_query(fms, id, &info) == FMS_OK, "fms_query");
    CHECK(info.kind == ELPIS_FMS_KIND_VECTOR_SHARD, "wrong object kind 0x%x", info.kind);
    CHECK(info.size_bytes == img.size(), "object size %llu vs image %zu",
          (unsigned long long)info.size_bytes, img.size());
    CHECK(info.tier == FMS_WARM, "shard admitted at tier %u", info.tier);
    CHECK(info.pin_count == 0 && info.lease_count == 0, "shard pinned after admission");

    fms_stats st;
    fms_get_stats(fms, &st);
    CHECK(st.objects == 1, "objects %llu", (unsigned long long)st.objects);
    CHECK(st.tier_bytes[FMS_HOT] == 0, "bytes resident in HOT");

    elpis_vector_query q;
    std::memset(&q, 0, sizeof q);
    q.vector = c.vec[3].data();
    q.dimensions = D;
    q.k = 4;
    std::vector<elpis_vector_hit> hits(q.k);
    uint32_t n = 0;
    CHECK(elpis_vector_index_search(ix, &q, hits.data(), &n) == 0, "search: %s",
          elpis_vector_index_error(ix));
    CHECK(n == 4, "hits %u", n);

    CHECK(fms_query(fms, id, &info) == FMS_OK, "query after search");
    CHECK(info.lease_count == 0, "lease leak: %u leases still held", info.lease_count);
    CHECK(info.pin_count == 0, "pin leak: %u pins still held", info.pin_count);
    fms_get_stats(fms, &st);
    CHECK(st.pinned_bytes == 0, "pinned bytes leaked: %llu", (unsigned long long)st.pinned_bytes);
    CHECK(st.inflight_ops == 0, "in-flight ops leaked");
    CHECK(st.tier_bytes[FMS_HOT] == 0, "HOT residency after search");
    CHECK(st.forced_placements == 0, "a placement was forced down: HOT may have been requested");

    /* The pump has nowhere to demote a WARM shard to: it stays resident and intact. */
    for (int i = 0; i < 4; i++) fms_pump(fms);
    CHECK(fms_query(fms, id, &info) == FMS_OK && info.tier == FMS_WARM && info.state == FMS_ST_RESIDENT,
          "the pump moved or dropped a WARM shard");

    elpis_vector_index_destroy(ix);
    fms_get_stats(fms, &st);
    CHECK(st.objects == 0, "object leak after index destroy: %llu", (unsigned long long)st.objects);
    CHECK(st.tier_bytes[FMS_WARM] == 0, "tier bytes leaked");
    CHECK(st.tier_bytes[FMS_COLD] == 0 && st.domain_bytes[FMS_DOM_STORAGE] == 0 && st.cold_writes == 0 &&
          st.cold_reads == 0, "storage was charged or touched");
    CHECK(st.domain_bytes[FMS_DOM_RAM] == 0, "RAM charge leaked");

    fms_destroy(fms);
    elpis_embedder_destroy(emb);
}

static void case_ceiling_without_spill() {
    CASE("RAM ceiling is enforced: a shard beyond WARM is refused, never spilled; resident shards stay exact");
    /* Four shards of ~800 KiB against a 2 MiB ceiling: two fit, the third is refused at admission. */
    const uint64_t warm = 2ull << 20;
    fms_ctx *fms = vecfix::make_fms("ceiling", warm);
    CHECK(fms != nullptr, "fms");
    if (!fms) return;

    elpis_embedder *emb = nullptr;
    elpis_embedder_fixture_create(ELPIS_NORM_L2, &emb);
    elpis_embedding_profile prof;
    elpis_embedder_profile(emb, &prof);
    elpis_vector_index *ix = nullptr;
    elpis_vector_index_create(fms, &prof, kCorpusDg, &ix);

    std::vector<vecfix::Corpus> shards(4);
    std::vector<std::vector<uint8_t>> images(4);
    std::vector<std::string> ids;
    int admitted = 0, refused = 0;
    for (int i = 0; i < 4; i++) {
        char tag[32];
        std::snprintf(tag, sizeof tag, "ceil-%d", i);
        shards[(size_t)i].build(tag, kShardVectors, emb);
        char sd[65];
        images[(size_t)i] = vecfix::build_shard(shards[(size_t)i], kCorpusDg, sd);
        ids.emplace_back(sd);
        int rc = elpis_vector_index_add_shard_bytes(ix, images[(size_t)i].data(), images[(size_t)i].size(), nullptr);
        if (rc == ELPIS_VEC_OK) {
            admitted++;
        } else {
            refused++;
            CHECK(rc == ELPIS_VEC_E_RESIDENCY, "shard %d: expected ELPIS_VEC_E_RESIDENCY, got %d (%s)", i, rc,
                  elpis_vec_strerror(rc));
            const elpis_vec_error *err = elpis_vector_index_last_error(ix);
            CHECK(err && err->cause == FMS_E_LIMIT, "shard %d: cause %d, expected FMS_E_LIMIT", i,
                  err ? err->cause : 0);
        }
        fms_stats st;
        fms_get_stats(fms, &st);
        CHECK(st.domain_bytes[FMS_DOM_RAM] <= warm, "RAM ceiling breached after shard %d: %llu",
              i, (unsigned long long)st.domain_bytes[FMS_DOM_RAM]);
    }
    CHECK(admitted == 2 && refused == 2, "admitted %d, refused %d (expected 2 and 2)", admitted, refused);
    CHECK(elpis_vector_index_shard_count(ix) == 2, "a refused shard is listed");

    fms_stats st;
    fms_get_stats(fms, &st);
    CHECK(st.objects == 2, "objects %llu", (unsigned long long)st.objects);
    CHECK(st.demotions == 0 && st.bytes_demoted == 0, "a shard was demoted");
    CHECK(st.tier_bytes[FMS_COLD] == 0 && st.domain_bytes[FMS_DOM_STORAGE] == 0 && st.cold_writes == 0,
          "something spilled to storage");

    /* The resident shards answer exactly, and keep answering exactly across pumps. */
    elpis_vector_query q;
    std::memset(&q, 0, sizeof q);
    q.vector = shards[1].vec[9].data();
    q.dimensions = D;
    q.k = 5;
    std::vector<elpis_vector_hit> hits(q.k);
    uint32_t n = 0;
    CHECK(elpis_vector_index_search(ix, &q, hits.data(), &n) == 0, "search at the ceiling: %s",
          elpis_vector_index_error(ix));
    CHECK(n == 5, "hits %u", n);
    if (n) CHECK(hits[0].chunk_digest == shards[1].chunk[9], "wrong nearest at the ceiling");
    char first[65] = {0}, later[65] = {0};
    elpis_vector_result_digest(hits.data(), n, first);
    for (int i = 0; i < 6; i++) {
        elpis_vector_index_search(ix, &q, hits.data(), &n);
        fms_pump(fms);
        fms_get_stats(fms, &st);
        CHECK(st.domain_bytes[FMS_DOM_RAM] <= warm, "RAM ceiling breached in loop");
    }
    elpis_vector_result_digest(hits.data(), n, later);
    CHECK(std::strcmp(first, later) == 0, "results changed across searches and pumps");
    fms_get_stats(fms, &st);
    CHECK(st.promotions == 0 && st.demotions == 0, "residency moved: %llu promotions, %llu demotions",
          (unsigned long long)st.promotions, (unsigned long long)st.demotions);
    CHECK(st.pinned_bytes == 0, "pins leaked at the ceiling");
    CHECK(st.tier_bytes[FMS_HOT] == 0, "HOT used at the ceiling");

    /* Bounded, not stuck: closing a resident shard makes room, and a refused shard is then admitted. */
    CHECK(elpis_vector_index_close_shard(ix, ids[0].c_str()) == 0, "close");
    CHECK(elpis_vector_index_add_shard_bytes(ix, images[2].data(), images[2].size(), nullptr) == 0,
          "re-admission after close: %s", elpis_vector_index_error(ix));
    CHECK(elpis_vector_index_shard_count(ix) == 2, "shard count after re-admission");
    fms_get_stats(fms, &st);
    CHECK(st.domain_bytes[FMS_DOM_RAM] <= warm && st.tier_bytes[FMS_COLD] == 0, "re-admission breached a bound");

    elpis_vector_index_destroy(ix);
    fms_destroy(fms);
    elpis_embedder_destroy(emb);
}

static void case_no_headroom() {
    CASE("a shard that cannot be made WARM-resident fails structurally");
    /* Ceiling far below one shard: admission itself must fail, loudly. */
    fms_ctx *fms = vecfix::make_fms("nohead", 64ull << 10);
    CHECK(fms != nullptr, "fms");
    if (!fms) return;
    elpis_embedder *emb = nullptr;
    elpis_embedder_fixture_create(ELPIS_NORM_L2, &emb);
    elpis_embedding_profile prof;
    elpis_embedder_profile(emb, &prof);
    elpis_vector_index *ix = nullptr;
    elpis_vector_index_create(fms, &prof, kCorpusDg, &ix);

    vecfix::Corpus c;
    c.build("nohead", kShardVectors, emb);
    char sd[65];
    std::vector<uint8_t> img = vecfix::build_shard(c, kCorpusDg, sd);
    int rc = elpis_vector_index_add_shard_bytes(ix, img.data(), img.size(), nullptr);
    CHECK(rc == ELPIS_VEC_E_RESIDENCY, "expected ELPIS_VEC_E_RESIDENCY, got %d (%s)",
          rc, elpis_vec_strerror(rc));
    const elpis_vec_error *err = elpis_vector_index_last_error(ix);
    CHECK(err && err->cause == FMS_E_LIMIT, "underlying FMS cause not preserved: %d",
          err ? err->cause : 0);
    CHECK(std::strstr(elpis_vector_index_error(ix), "fms_register") != nullptr,
          "unexpected error: %s", elpis_vector_index_error(ix));
    CHECK(elpis_vector_index_shard_count(ix) == 0, "failed admission left a shard behind");

    fms_stats st;
    fms_get_stats(fms, &st);
    CHECK(st.objects == 0, "failed admission leaked an object");

    /* A search over an empty index is empty, not an error. */
    elpis_vector_query q;
    std::memset(&q, 0, sizeof q);
    q.vector = c.vec[0].data();
    q.dimensions = D;
    q.k = 3;
    std::vector<elpis_vector_hit> hits(3);
    uint32_t n = 99;
    CHECK(elpis_vector_index_search(ix, &q, hits.data(), &n) == ELPIS_VEC_OK,
          "empty index search failed");
    CHECK(n == 0, "empty index returned %u hits", n);

    elpis_vector_index_destroy(ix);
    fms_destroy(fms);
    elpis_embedder_destroy(emb);
}

static void case_resident_corruption() {
    CASE("a resident shard corrupted in RAM is an integrity failure, never not-found or empty");
    fms_ctx *fms = vecfix::make_fms("rot", 4ull << 20);
    CHECK(fms != nullptr, "fms");
    if (!fms) return;
    elpis_embedder *emb = nullptr;
    elpis_embedder_fixture_create(ELPIS_NORM_L2, &emb);
    elpis_embedding_profile prof;
    elpis_embedder_profile(emb, &prof);
    elpis_vector_index *ix = nullptr;
    elpis_vector_index_create(fms, &prof, kCorpusDg, &ix);

    vecfix::Corpus c;
    c.build("rot", kShardVectors, emb);
    char sd[65];
    std::vector<uint8_t> img = vecfix::build_shard(c, kCorpusDg, sd);
    CHECK(elpis_vector_index_add_shard_bytes(ix, img.data(), img.size(), nullptr) == 0, "admit");
    CHECK(elpis_vector_index_verify(ix, sd) == ELPIS_VEC_OK, "intact shard failed verify: %s",
          elpis_vector_index_error(ix));

    /* Flip one payload byte of the resident copy (a RAM fault): the only copy there is. */
    fms_id id = 0;
    CHECK(elpis_vector_index_shard_object(ix, sd, &id) == 0, "shard object");
    void *p = nullptr;
    CHECK((int)fms_acquire(fms, id, FMS_WARM, FMS_WRITE, &p) == (int)FMS_WARM && p, "acquire for the fault");
    if (p) ((unsigned char *)p)[img.size() / 2] ^= 0xA5;
    fms_release(fms, id);

    int rc = elpis_vector_index_verify(ix, sd);
    CHECK(rc == ELPIS_VEC_E_INTEGRITY, "expected ELPIS_VEC_E_INTEGRITY, got %d (%s)", rc, elpis_vec_strerror(rc));
    CHECK(rc != ELPIS_VEC_E_NOTFOUND && rc != ELPIS_VEC_E_RESIDENCY, "integrity failure downgraded");
    const elpis_vec_error *err = elpis_vector_index_last_error(ix);
    CHECK(err && err->status == ELPIS_VEC_E_INTEGRITY, "structured status %d", err ? err->status : 0);
    if (err) CHECK(std::strstr(err->detail, "verify") != nullptr, "detail: %s", err->detail);

    fms_stats st;
    fms_get_stats(fms, &st);
    CHECK(st.pinned_bytes == 0, "lease leaked on the failure path");
    CHECK(st.tier_bytes[FMS_COLD] == 0 && st.cold_reads == 0 && st.cold_writes == 0, "storage touched");

    elpis_vector_index_destroy(ix);
    fms_destroy(fms);
    elpis_embedder_destroy(emb);
}

static void case_close_shard() {
    CASE("closing a shard releases its FMS object");
    fms_ctx *fms = vecfix::make_fms("close", 8ull << 20);
    if (!fms) { CHECK(false, "fms"); return; }
    elpis_embedder *emb = nullptr;
    elpis_embedder_fixture_create(ELPIS_NORM_L2, &emb);
    elpis_embedding_profile prof;
    elpis_embedder_profile(emb, &prof);
    elpis_vector_index *ix = nullptr;
    elpis_vector_index_create(fms, &prof, kCorpusDg, &ix);

    vecfix::Corpus c;
    c.build("close", 128, emb);
    char sd[65];
    std::vector<uint8_t> img = vecfix::build_shard(c, kCorpusDg, sd);
    elpis_vector_index_add_shard_bytes(ix, img.data(), img.size(), nullptr);
    CHECK(elpis_vector_index_shard_count(ix) == 1, "admitted");

    CHECK(elpis_vector_index_verify(ix, sd) == 0, "verify admitted shard: %s",
          elpis_vector_index_error(ix));
    elpis_vshard_header h;
    CHECK(elpis_vector_index_inspect(ix, sd, &h) == 0, "inspect");
    CHECK(h.vector_count == 128, "inspect vector_count %llu", (unsigned long long)h.vector_count);

    CHECK(elpis_vector_index_close_shard(ix, sd) == 0, "close");
    CHECK(elpis_vector_index_shard_count(ix) == 0, "shard still listed");
    fms_stats st;
    fms_get_stats(fms, &st);
    CHECK(st.objects == 0, "closing left the FMS object behind");
    CHECK(elpis_vector_index_close_shard(ix, sd) != 0, "closing an unknown shard succeeded");

    elpis_vector_index_destroy(ix);
    fms_destroy(fms);
    elpis_embedder_destroy(emb);
}

static void case_nowait_never_waits() {
    CASE("every admitted shard is resident: the no-wait search is the exact search and never waits, even "
         "while another holder pins a shard");
    fms_ctx *fms = vecfix::make_fms("nowait", 2ull << 20);
    CHECK(fms != nullptr, "fms");
    if (!fms) return;
    elpis_embedder *emb = nullptr;
    elpis_embedder_fixture_create(ELPIS_NORM_L2, &emb);
    elpis_embedding_profile prof;
    elpis_embedder_profile(emb, &prof);
    elpis_vector_index *ix = nullptr;
    elpis_vector_index_create(fms, &prof, kCorpusDg, &ix);
    std::vector<vecfix::Corpus> shards(2);
    std::vector<std::string> ids;
    for (int i = 0; i < 2; i++) {
        char tag[32];
        std::snprintf(tag, sizeof tag, "nowait-%d", i);
        shards[(size_t)i].build(tag, kShardVectors, emb);
        char sd[65];
        std::vector<uint8_t> img = vecfix::build_shard(shards[(size_t)i], kCorpusDg, sd);
        CHECK(elpis_vector_index_add_shard_bytes(ix, img.data(), img.size(), nullptr) == 0, "admit %d: %s", i,
              elpis_vector_index_error(ix));
        ids.emplace_back(sd);
    }
    elpis_vector_query q;
    std::memset(&q, 0, sizeof q);
    q.vector = shards[0].vec[3].data();
    q.dimensions = D;
    q.k = 4;
    std::vector<elpis_vector_hit> hits(q.k);
    uint32_t n = 0;
    CHECK(elpis_vector_index_search_nowait(ix, &q, hits.data(), &n) == ELPIS_VEC_OK, "nowait search: %s",
          elpis_vector_index_error(ix));
    char a[65] = {0}, b[65] = {0};
    elpis_vector_result_digest(hits.data(), n, a);
    CHECK(elpis_vector_index_search(ix, &q, hits.data(), &n) == ELPIS_VEC_OK, "search");
    elpis_vector_result_digest(hits.data(), n, b);
    CHECK(std::strcmp(a, b) == 0, "the two entries disagree");

    /* Another holder pins a shard: a pin is shared, residency does not move, so neither entry waits. */
    fms_id held = 0;
    elpis_vector_index_shard_object(ix, ids[0].c_str(), &held);
    void *ptr = nullptr;
    CHECK((int)fms_acquire(fms, held, FMS_WARM, FMS_READ, &ptr) == (int)FMS_WARM, "external pin");
    for (int entry = 0; entry < 2; entry++) {
        const auto t0 = std::chrono::steady_clock::now();
        n = 99;
        int rc = entry ? elpis_vector_index_search(ix, &q, hits.data(), &n)
                       : elpis_vector_index_search_nowait(ix, &q, hits.data(), &n);
        const auto waited = std::chrono::steady_clock::now() - t0;
        CHECK(rc == ELPIS_VEC_OK && n == 4, "entry %d under an external pin: %d (%u hits)", entry, rc, n);
        CHECK(waited < std::chrono::milliseconds(1000), "entry %d waited %lld ms", entry,
              (long long)std::chrono::duration_cast<std::chrono::milliseconds>(waited).count());
        elpis_vector_result_digest(hits.data(), n, b);
        CHECK(std::strcmp(a, b) == 0, "entry %d differs under an external pin", entry);
    }
    fms_release(fms, held);
    fms_stats st;
    fms_get_stats(fms, &st);
    CHECK(st.pinned_bytes == 0, "pins leaked");
    CHECK(st.promotions == 0 && st.demotions == 0, "residency moved");

    elpis_vector_index_destroy(ix);
    fms_destroy(fms);
    elpis_embedder_destroy(emb);
}

int main() {
    std::printf("R2 FMS residency suite (volatile: FMS ABI v3, no COLD tier, no file)\n");

    case_registration_and_lifecycle();
    case_ceiling_without_spill();
    case_no_headroom();
    case_resident_corruption();
    case_close_shard();
    case_nowait_never_waits();

    std::printf("%d checks, %d failures\n", checks, fails);
    return fails != 0;
}
