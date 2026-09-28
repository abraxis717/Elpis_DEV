/* grid81_input_validate.c — Grid81-compiler input validation from topology handoff. */
#include "elpis_semantic/grid81_policy.h"
#include "elpis_semantic/grid81_codebook.h"
#include "elpis_semantic/topology_handoff.h"
#include "elpis_semantic/topology_graph.h"
#include "elpis_semantic/topology_constraint.h"
#include "elpis_semantic/topology_address.h"
#include "elpis_semantic/topology_constellation.h"
#include "elpis_semantic/topology_anchor.h"
#include <string.h>

/* Validate topology handoff for Grid81-compiler consumption. */
int elpis_grid81_validate_handoff(
    const elpis_semantic_topology_handoff_v1 *handoff) {
    if (!handoff) return SEMANTIC_E_INVAL;
    if (handoff->abi_version != TOPOLOGY_HANDOFF_ABI_VERSION) return SEMANTIC_E_INVAL;
    if (handoff->handoff_kind != TOPOLOGY_HANDOFF_SEMANTIC_TO_GRID81_COMPILER_INPUT) {
        return SEMANTIC_E_INVAL;
    }
    /* Check Grid81-compiler boundaries are set */
    if (!handoff->P7_may_assign_discrete_placement) return SEMANTIC_E_INVAL;
    if (!handoff->P7_may_not_alter_relation_types) return SEMANTIC_E_INVAL;
    if (!handoff->P7_may_not_alter_authority) return SEMANTIC_E_INVAL;
    if (!handoff->P7_may_not_remove_conflict_polarity) return SEMANTIC_E_INVAL;
    if (!handoff->P7_may_not_treat_metric_as_semantic) return SEMANTIC_E_INVAL;
    if (!handoff->local_ordinal_is_not_grid81_cell) return SEMANTIC_E_INVAL;
    if (!handoff->one_vertex_not_one_cell) return SEMANTIC_E_INVAL;
    return SEMANTIC_OK;
}

/* Validate topology graph for Grid81-compiler placement: vertices and incidences present. */
int elpis_grid81_validate_topology_graph(
    const elpis_semantic_topology_graph_v1 *graph) {
    if (!graph) return SEMANTIC_E_INVAL;
    if (graph->abi_version != TOPOLOGY_GRAPH_ABI_VERSION) return SEMANTIC_E_INVAL;
    if (graph->vertex_count == 0) return SEMANTIC_E_INVAL;
    return SEMANTIC_OK;
}

/* Validate constraints collection for Grid81-compiler projection. */
int elpis_grid81_validate_constraints(
    const elpis_semantic_topology_constraints_v1 *constraints) {
    if (!constraints) return SEMANTIC_E_INVAL;
    if (constraints->abi_version != TOPOLOGY_CONSTRAINT_ABI_VERSION) return SEMANTIC_E_INVAL;
    return SEMANTIC_OK;
}

/* Validate addresses/roles collection for Grid81-compiler lane assignment. */
int elpis_grid81_validate_addresses(
    const elpis_semantic_topology_addresses_v1 *addresses) {
    if (!addresses) return SEMANTIC_E_INVAL;
    if (addresses->abi_version != TOPOLOGY_ADDRESS_ABI_VERSION) return SEMANTIC_E_INVAL;
    return SEMANTIC_OK;
}
