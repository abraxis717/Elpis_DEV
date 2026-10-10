/* ingress_bridge.c — narrow C ABI over the query ingress for Python. */

#define _POSIX_C_SOURCE 200809L

#include "ingress_bridge.h"

#include "elpis/corpus.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>


struct elpis_ingress_env {
    elpis_corpus *corpus;
    int owns_corpus;
    elpis_context_graph *graph;
};

static int fail(char *error, int code, const char *detail) {
    if (error) snprintf(error, ELPIS_INGRESS_BRIDGE_ERROR_BYTES, "%s", detail);
    return code;
}

uint32_t elpis_ingress_bridge_abi_version(void) {
    return ELPIS_INGRESS_BRIDGE_ABI_VERSION;
}

/* Retired path-backed ingress ABI. No disk corpus may be reopened or created.
 * Keep this narrow refusal symbol for existing ctypes clients. */
int elpis_ingress_env_open(const char *corpus_root,
                           const elpis_context_edge_input *edges,
                           uint32_t edge_count,
                           elpis_ingress_env **out,
                           char error[ELPIS_INGRESS_BRIDGE_ERROR_BYTES]) {
    (void)corpus_root; (void)edges; (void)edge_count;
    if (!out) return fail(error, ELPIS_INGRESS_BRIDGE_E_INVAL, "E_INVAL: output slot");
    *out=NULL;
    return fail(error, ELPIS_INGRESS_BRIDGE_E_CORPUS, "E_CORPUS: persistent ingress retired");
}

int elpis_ingress_env_open_borrowed(void *corpus,
                           const elpis_context_edge_input *edges,
                           uint32_t edge_count,
                           elpis_ingress_env **out,
                           char error[ELPIS_INGRESS_BRIDGE_ERROR_BYTES]) {
    if(error) error[0]='\0';
    if(!out) return fail(error, ELPIS_INGRESS_BRIDGE_E_INVAL, "E_INVAL: output slot");
    *out=NULL;
    if(!corpus || (edge_count && !edges) || edge_count>ELPIS_CGRAPH_MAX_EDGES)
        return fail(error, ELPIS_INGRESS_BRIDGE_E_INVAL, "E_INVAL: borrowed arguments");
    elpis_ingress_env *env=calloc(1,sizeof *env);
    if(!env) return fail(error, ELPIS_INGRESS_BRIDGE_E_NOMEM, "E_NOMEM: environment");
    env->corpus=(elpis_corpus *)corpus;
    env->owns_corpus=0;
    if(elpis_context_graph_create(edge_count ? edges : NULL,edge_count,&env->graph)!=0) {
        elpis_ingress_env_close(env);
        return fail(error, ELPIS_INGRESS_BRIDGE_E_GRAPH, "E_GRAPH: context graph rejected");
    }
    *out=env;
    return fail(error, ELPIS_INGRESS_BRIDGE_OK, "ok");
}

void elpis_ingress_env_close(elpis_ingress_env *env) {
    if (!env) return;
    if (env->graph) elpis_context_graph_destroy(env->graph);
    if (env->owns_corpus && env->corpus) elpis_corpus_close(env->corpus);
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
