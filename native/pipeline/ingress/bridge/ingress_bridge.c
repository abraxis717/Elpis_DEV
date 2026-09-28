/* ingress_bridge.c — narrow C ABI over the query ingress for Python. */

#define _POSIX_C_SOURCE 200809L

#include "ingress_bridge.h"

#include "elpis/corpus.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>

#define INGRESS_PATH_CAP 4096

struct elpis_ingress_env {
    elpis_corpus *corpus;
    elpis_context_graph *graph;
};

static int fail(char *error, int code, const char *detail) {
    if (error) snprintf(error, ELPIS_INGRESS_BRIDGE_ERROR_BYTES, "%s", detail);
    return code;
}

uint32_t elpis_ingress_bridge_abi_version(void) {
    return ELPIS_INGRESS_BRIDGE_ABI_VERSION;
}

static int existing_corpus(const char *root, char *error) {
    struct stat st;
    char db[INGRESS_PATH_CAP];
    size_t len = strlen(root);
    if (len == 0 || len >= INGRESS_PATH_CAP - 32)
        return fail(error, ELPIS_INGRESS_BRIDGE_E_INVAL, "E_INVAL: corpus root length");
    if (lstat(root, &st) != 0 || !S_ISDIR(st.st_mode))
        return fail(error, ELPIS_INGRESS_BRIDGE_E_CORPUS, "E_CORPUS: corpus root is not an existing directory");
    snprintf(db, sizeof db, "%s/metadata.sqlite", root);
    if (lstat(db, &st) != 0 || !S_ISREG(st.st_mode))
        return fail(error, ELPIS_INGRESS_BRIDGE_E_CORPUS, "E_CORPUS: no existing corpus at root");
    return ELPIS_INGRESS_BRIDGE_OK;
}

int elpis_ingress_env_open(const char *corpus_root,
                           const elpis_context_edge_input *edges,
                           uint32_t edge_count,
                           elpis_ingress_env **out,
                           char error[ELPIS_INGRESS_BRIDGE_ERROR_BYTES]) {
    if (error) error[0] = '\0';
    if (!out) return fail(error, ELPIS_INGRESS_BRIDGE_E_INVAL, "E_INVAL: output slot");
    *out = NULL;
    if (!corpus_root || (edge_count && !edges) || edge_count > ELPIS_CGRAPH_MAX_EDGES)
        return fail(error, ELPIS_INGRESS_BRIDGE_E_INVAL, "E_INVAL: open arguments");
    int rc = existing_corpus(corpus_root, error);
    if (rc != ELPIS_INGRESS_BRIDGE_OK) return rc;

    elpis_ingress_env *env = calloc(1, sizeof *env);
    if (!env) return fail(error, ELPIS_INGRESS_BRIDGE_E_NOMEM, "E_NOMEM: environment");
    if (elpis_context_graph_create(edge_count ? edges : NULL, edge_count, &env->graph) != 0) {
        elpis_ingress_env_close(env);
        return fail(error, ELPIS_INGRESS_BRIDGE_E_GRAPH, "E_GRAPH: context graph rejected");
    }
    if (elpis_corpus_open(corpus_root, &env->corpus) != 0) {
        elpis_ingress_env_close(env);
        return fail(error, ELPIS_INGRESS_BRIDGE_E_CORPUS, "E_CORPUS: corpus open failed");
    }
    *out = env;
    return fail(error, ELPIS_INGRESS_BRIDGE_OK, "ok");
}

void elpis_ingress_env_close(elpis_ingress_env *env) {
    if (!env) return;
    if (env->graph) elpis_context_graph_destroy(env->graph);
    if (env->corpus) elpis_corpus_close(env->corpus);
    free(env);
}

int elpis_ingress_env_run(elpis_ingress_env *env,
                          const uint8_t *task_bytes,
                          size_t task_len,
                          size_t regex_chunk_size,
                          elpis_regex_hacf_query_ingress_result_v1 **out) {
    if (!out) return ELPIS_REGEX_HACF_QUERY_INGRESS_E_INVAL;
    *out = NULL;
    if (!env || !env->corpus || !env->graph)
        return ELPIS_REGEX_HACF_QUERY_INGRESS_E_INVAL;
    return elpis_regex_hacf_query_ingress_run_v1(
        task_bytes, task_len, regex_chunk_size, env->corpus, env->graph, out);
}

int elpis_ingress_env_corpus_counts(elpis_ingress_env *env,
                                    uint64_t *documents,
                                    uint64_t *chunks) {
    if (!env || !env->corpus || !documents || !chunks) return ELPIS_INGRESS_BRIDGE_E_INVAL;
    return elpis_corpus_counts(env->corpus, documents, chunks) == 0
        ? ELPIS_INGRESS_BRIDGE_OK : ELPIS_INGRESS_BRIDGE_E_CORPUS;
}

uint32_t elpis_ingress_env_edge_count(const elpis_ingress_env *env) {
    return env && env->graph ? elpis_context_graph_edge_count(env->graph) : 0u;
}
