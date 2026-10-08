/* The Python-facing retrieval bridge's context graph (retrieval_bridge.h, ABI v2), natively: construction from
 * explicit edges, refusal of malformed, self and unadmitted-endpoint edges (with everything built so far released;
 * the sanitizer jobs run this under leak detection), graph-enabled one-hop retrieval with the graph identity in
 * the bundle, the graph policy refused without a graph, and destruction.
 *
 * Usage: test_retrieval_bridge_graph <scratch-dir>. Deterministic: no threads. */
#define _POSIX_C_SOURCE 200809L
#include "retrieval_bridge.h"

#include <assert.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

enum { DIM = 384, JSON_CAP = 1 << 18, DOCS = 3 };

static const char *LABELS[DOCS] = {"alpha", "beta", "gamma"};
static const char *TEXTS[DOCS] = {"alpha engine exact retrieval anchor", "beta companion context bridge",
                                  "gamma vector semantic neighbor"};
static const char *NAMESPACES[DOCS] = {"elpis.docs", "elpis.docs", "elpis.code"};
static const char *AUTHORITIES[DOCS] = {"canonical", "reference", "canonical"};
static const char ZERO[65] = "0000000000000000000000000000000000000000000000000000000000000000";

static char ROOT[2048];
static char JSON[JSON_CAP];

typedef struct {
    int rc;
    int items;
    char bundle[65], query[65], corpus[65], vindex[65], graph[65], policy[65], package[65];
} result;

static elpis_retrieval_env_t *env_at(const char *leaf, int with_graph, const elpis_context_edge_input *edges,
                                     uint32_t n, char error[256]) {
    char root[2304];
    assert(snprintf(root, sizeof root, "%s/%s", ROOT, leaf) < (int)sizeof root);
    char cmd[2400];
    assert(snprintf(cmd, sizeof cmd, "rm -rf '%s'", root) < (int)sizeof cmd);
    assert(system(cmd) == 0);
    return elpis_retrieval_env_create(root, LABELS, TEXTS, NAMESPACES, AUTHORITIES, DOCS, with_graph, edges, n,
                                      error);
}

static result retrieve(elpis_retrieval_env_t *env, uint32_t lexical, uint32_t dense, uint32_t primary,
                       uint32_t total, const elpis_retrieval_graph_policy *gp) {
    static float vector[DIM];
    for (int i = 0; i < DIM; ++i) vector[i] = (float)((i * 7) % 11) / 11.0f;
    result r;
    memset(&r, 0, sizeof r);
    char error[256];
    r.rc = elpis_retrieval_env_retrieve(env, "alpha", vector, DIM, lexical, dense, primary, total, NULL, NULL, gp,
                                        JSON, JSON_CAP, r.bundle, r.query, r.corpus, r.vindex, r.graph, r.policy,
                                        r.package, &r.items, error);
    return r;
}

/* The chunk digests of the bundle's items, in rank order. */
static int chunks(char out[][65], int cap) {
    int n = 0;
    for (const char *p = JSON; n < cap && (p = strstr(p, "\"chunk_digest\":\"")) != NULL; ++n) {
        p += strlen("\"chunk_digest\":\"");
        memcpy(out[n], p, 64);
        out[n][64] = '\0';
    }
    return n;
}

static elpis_context_edge_input edge(const char *subject, const char *object, uint32_t type, uint32_t authority) {
    elpis_context_edge_input e;
    memset(&e, 0, sizeof e);
    assert(strlen(subject) == 64 && strlen(object) == 64);
    memcpy(e.subject_chunk_digest, subject, 65);
    memcpy(e.object_chunk_digest, object, 65);
    memset(e.provenance_digest, '1', 64);   /* the fact's provenance identity (fixture) */
    e.edge_type = type;
    e.authority = authority;
    return e;
}

static void refused(const char *leaf, int with_graph, const elpis_context_edge_input *edges, uint32_t n,
                    const char *prefix) {
    char error[256];
    assert(env_at(leaf, with_graph, edges, n, error) == NULL);
    assert(!strncmp(error, prefix, strlen(prefix)));
}

int main(int argc, char **argv) {
    assert(argc == 2 && strlen(argv[1]) < 1024);
    snprintf(ROOT, sizeof ROOT, "%s", argv[1]);
    assert(elpis_retrieval_bridge_abi_version() == ELPIS_RETRIEVAL_BRIDGE_ABI_VERSION);
    assert(ELPIS_RETRIEVAL_BRIDGE_ABI_VERSION == 2u);

    /* Graph disabled: the environment's and the bundle's graph identity are the native zero digest. */
    char error[256];
    elpis_retrieval_env_t *plain = env_at("plain", 0, NULL, 0, error);
    assert(plain && !strcmp(error, "ok"));
    assert(!strcmp(elpis_retrieval_env_graph_digest(plain), ZERO) && elpis_retrieval_env_graph_edge_count(plain) == 0);
    result base = retrieve(plain, 50, 50, 30, 60, NULL);
    assert(base.rc == 0 && base.items == DOCS && !strcmp(base.graph, ZERO) && strcmp(base.package, ZERO));
    char c[DOCS][65];
    assert(chunks(c, DOCS) == DOCS);
    elpis_retrieval_graph_policy gp = {1, 1, 0};
    assert(retrieve(plain, 50, 50, 30, 60, &gp).rc == -3);   /* a graph policy needs a graph */

    /* Refusals release everything built so far. */
    elpis_context_edge_input bad[2] = {edge(c[0], c[1], 1, 1), edge(c[0], c[1], 1, 1)};
    memset(bad[1].object_chunk_digest, '9', 64);   /* well formed, but not an admitted chunk */
    refused("missing-object", 1, bad, 2, "E_GRAPH");
    bad[1] = edge(c[0], c[1], 1, 1);
    memset(bad[1].subject_chunk_digest, '9', 64);
    refused("missing-subject", 1, bad, 2, "E_GRAPH");
    bad[1] = edge(c[2], c[2], 1, 1);
    refused("self", 1, bad, 2, "E_GRAPH");
    bad[1] = edge(c[0], c[1], 0, 1);
    refused("type", 1, bad, 2, "E_GRAPH");
    bad[1] = edge(c[0], c[1], 1, 4);
    refused("authority", 1, bad, 2, "E_GRAPH");
    bad[1] = edge(c[0], c[1], 1, 1);
    bad[1].subject_chunk_digest[3] = 'A';
    refused("case", 1, bad, 2, "E_GRAPH");
    refused("edges-without-graph", 0, bad, 1, "E_INVAL");
    refused("graph-flag", 2, NULL, 0, "E_INVAL");
    refused("count-without-edges", 1, NULL, 3, "E_INVAL");

    /* A complete graph over the three chunks (duplicates collapse) expands the single primary by one hop. */
    elpis_context_edge_input all[8];
    int n = 0;
    for (int s = 0; s < DOCS; ++s)
        for (int o = 0; o < DOCS; ++o)
            if (s != o) all[n++] = edge(c[s], c[o], 1 + (uint32_t)o, 2);
    all[n++] = all[0];
    all[n++] = all[3];
    elpis_retrieval_env_t *graph = env_at("graph", 1, all, (uint32_t)n, error);
    assert(graph && !strcmp(error, "ok"));
    assert(elpis_retrieval_env_graph_edge_count(graph) == 6);
    char digest[65];
    assert(strlen(elpis_retrieval_env_graph_digest(graph)) == 64);
    memcpy(digest, elpis_retrieval_env_graph_digest(graph), 65);
    assert(strcmp(digest, ZERO) != 0);
    result r = retrieve(graph, 1, 1, 1, 4, NULL);
    assert(r.rc == 0 && r.items == DOCS && !strcmp(r.graph, digest) && strstr(JSON, digest));
    assert(strstr(JSON, "\"graph_hop\":1") && strstr(JSON, "\"item_kind\":2"));
    assert(!strcmp(r.corpus, base.corpus) && !strcmp(r.vindex, base.vindex));
    result once = retrieve(graph, 1, 1, 1, 4, &gp);   /* one neighbor per seed */
    assert(once.rc == 0 && once.items == 2 && strcmp(once.policy, r.policy) != 0);
    gp.graph_seed_limit = 2;   /* more seeds than primaries: the native validator refuses */
    assert(retrieve(graph, 1, 1, 1, 4, &gp).rc != 0);

    /* The same edges in another order are the same graph. */
    elpis_context_edge_input reversed[8];
    for (int i = 0; i < n; ++i) reversed[i] = all[n - 1 - i];
    elpis_retrieval_env_t *again = env_at("again", 1, reversed, (uint32_t)n, error);
    assert(again && !strcmp(elpis_retrieval_env_graph_digest(again), digest));
    result r2 = retrieve(again, 1, 1, 1, 4, NULL);
    assert(r2.rc == 0 && !strcmp(r2.bundle, r.bundle) && !strcmp(r2.package, r.package));

    elpis_retrieval_env_destroy(again);
    elpis_retrieval_env_destroy(graph);
    elpis_retrieval_env_destroy(plain);
    elpis_retrieval_env_destroy(NULL);
    puts("retrieval bridge graph: PASS");
    return 0;
}
