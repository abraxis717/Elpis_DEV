/* retrieval_bridge.h — the narrow C ABI over HACF structural retrieval that
 * elpis.structure.retrieval.hacf loads by explicit path (ABI v2).
 *
 * One environment is one immutable retrieval epoch: a content-addressed corpus
 * built from the supplied documents, its FMS-resident exact vector index and,
 * optionally, one immutable context graph built once from explicitly supplied
 * edges. The bridge never infers a relationship: no edge is derived from
 * embeddings, lexical overlap, rank, adjacency or any model output. Every edge
 * endpoint must be an admitted chunk of the environment's own corpus; an edge
 * naming any other chunk refuses construction (E_GRAPH), so a graph cannot
 * bring an unverified payload into a bundle (context text is always read from
 * the corpus by the native retriever).
 *
 * Retrieval is the native HACF hybrid retriever (lexical + dense RRF fusion and,
 * with a graph, bounded one-hop expansion) under one elpis_hybrid_policy. The
 * graph-disabled environment keeps graph_seed_limit = graph_neighbors_per_seed
 * = 0 and no graph, so its policy, query, bundle and package identities are
 * exactly those of ABI v1. v2 adds the graph, the query filters, the graph
 * policy selection and the graph snapshot / HACF package identity outputs. */
#ifndef ELPIS_RETRIEVAL_BRIDGE_H
#define ELPIS_RETRIEVAL_BRIDGE_H

#include <stdint.h>
#include <stddef.h>

#include "elpis/context_graph.h"

#ifdef __cplusplus
extern "C" {
#endif

#define ELPIS_RETRIEVAL_BRIDGE_ABI_VERSION 2u

/* A graph policy field left at this value takes the native default
 * (elpis_hybrid_policy_default); graph_seed_limit's default is then bounded by
 * primary_limit, as elpis_hybrid_query_plan_derive bounds it. */
#define ELPIS_RETRIEVAL_POLICY_DEFAULT UINT32_MAX

/* The graph fields of elpis_hybrid_policy a caller may select. Validated by
 * elpis_hybrid_policy_validate; covered by the fusion policy digest. */
typedef struct elpis_retrieval_graph_policy {
    uint32_t graph_seed_limit;
    uint32_t graph_neighbors_per_seed;
    uint32_t min_graph_authority;   /* hacf_authority 0..3 */
} elpis_retrieval_graph_policy;

typedef struct elpis_retrieval_env elpis_retrieval_env_t;

uint32_t elpis_retrieval_bridge_abi_version(void);

/* with_graph = 0: no context graph (edges NULL, edge_count 0).
 * with_graph = 1: one immutable graph from edges[0..edge_count) (zero edges is
 * an empty graph, which has an identity). Exact duplicates collapse; a
 * malformed digest, zero edge type, authority above 3, self-edge, or an
 * endpoint that is not an admitted chunk of this corpus refuses creation. */
elpis_retrieval_env_t *elpis_retrieval_env_create(const char *state_root,
                                                  const char **labels, const char **texts,
                                                  const char **namespaces, const char **authorities,
                                                  int n_docs,
                                                  int with_graph,
                                                  const elpis_context_edge_input *edges,
                                                  uint32_t edge_count,
                                                  char error_buf[256]);
void elpis_retrieval_env_destroy(elpis_retrieval_env_t *env);
/* Borrowed, unowned native pointer. Only valid while env remains alive.
 * The ingress bridge MUST NOT close it and Python owns the lifetime. */
void *elpis_retrieval_env_borrow_corpus(elpis_retrieval_env_t *env);
/* Bounded verified document read: copies into caller memory, never filesystem. */
int elpis_retrieval_env_copy_document(elpis_retrieval_env_t *env, const char *digest,
                                     void *dst, size_t capacity, size_t *actual);

/* The graph snapshot digest (64 zeros without a graph) and its edge count. */
const char *elpis_retrieval_env_graph_digest(elpis_retrieval_env_t *env);
uint32_t elpis_retrieval_env_graph_edge_count(elpis_retrieval_env_t *env);

/* namespace_filter / authority_filter: NULL = all (exact match otherwise).
 * graph_policy: NULL = the native default graph fields with a graph, and the
 * graph disabled (0, 0) without one; non-NULL without a graph is E_INVAL.
 * Every *_out digest buffer holds 65 bytes. */
int elpis_retrieval_env_retrieve(elpis_retrieval_env_t *env,
                                 const char *query_text,
                                 const float *query_vector, int query_dim,
                                 uint32_t lexical_limit, uint32_t dense_limit,
                                 uint32_t primary_limit, uint32_t total_limit,
                                 const char *namespace_filter,
                                 const char *authority_filter,
                                 const elpis_retrieval_graph_policy *graph_policy,
                                 char *bundle_json_out, int bundle_json_cap,
                                 char *bundle_digest_out,
                                 char *query_digest_out,
                                 char *corpus_manifest_digest_out,
                                 char *vindex_manifest_digest_out,
                                 char *graph_snapshot_digest_out,
                                 char *fusion_policy_digest_out,
                                 char *hacf_package_digest_out,
                                 int *item_count_out,
                                 char error_buf[256]);

#ifdef __cplusplus
}
#endif
#endif
