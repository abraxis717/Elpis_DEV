/* ingress_bridge.h — narrow C ABI over the query ingress for Python.
 *
 * Loaded by elpis.pipeline.ingress through an explicit library path. The
 * shared library exports exactly this header, the streaming Regex ABI and the
 * Regex -> HACF query-ingress result ABI (see ingress_bridge.map).
 */
#ifndef ELPIS_INGRESS_BRIDGE_H
#define ELPIS_INGRESS_BRIDGE_H

#include <stddef.h>
#include <stdint.h>

#include "elpis/context_graph.h"
#include "regex_hacf_query_ingress.h"

#ifdef __cplusplus
extern "C" {
#endif

#define ELPIS_INGRESS_BRIDGE_ABI_VERSION 1u
#define ELPIS_INGRESS_BRIDGE_ERROR_BYTES 256u

#define ELPIS_INGRESS_BRIDGE_OK 0
#define ELPIS_INGRESS_BRIDGE_E_INVAL (-1)
#define ELPIS_INGRESS_BRIDGE_E_CORPUS (-2)
#define ELPIS_INGRESS_BRIDGE_E_GRAPH (-3)
#define ELPIS_INGRESS_BRIDGE_E_NOMEM (-4)

typedef struct elpis_ingress_env elpis_ingress_env;

uint32_t elpis_ingress_bridge_abi_version(void);

/* Opens an EXISTING HACF corpus: corpus_root must be a real directory holding
 * a regular metadata.sqlite (no symlinks). No corpus is ever created. The
 * context graph is built once from the supplied immutable edges (NULL + 0 for
 * an empty graph). */
int elpis_ingress_env_open(const char *corpus_root,
                           const elpis_context_edge_input *edges,
                           uint32_t edge_count,
                           elpis_ingress_env **out,
                           char error[ELPIS_INGRESS_BRIDGE_ERROR_BYTES]);

void elpis_ingress_env_close(elpis_ingress_env *env);

/* Runs Regex -> HACF -> query-local proposal batch over the opened corpus and
 * graph. Read-only with respect to both; the result uses the query-ingress
 * result ABI and is released with elpis_regex_hacf_query_ingress_result_destroy_v1. */
int elpis_ingress_env_run(elpis_ingress_env *env,
                          const uint8_t *task_bytes,
                          size_t task_len,
                          size_t regex_chunk_size,
                          elpis_regex_hacf_query_ingress_result_v1 **out);

int elpis_ingress_env_corpus_counts(elpis_ingress_env *env,
                                    uint64_t *documents,
                                    uint64_t *chunks);

uint32_t elpis_ingress_env_edge_count(const elpis_ingress_env *env);

#ifdef __cplusplus
}
#endif

#endif
