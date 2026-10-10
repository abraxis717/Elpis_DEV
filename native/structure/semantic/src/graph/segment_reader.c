/* segment_reader.c — Verify the writer's native header and record arrays.
 * The v1 identity binds the HACF projection, not a hash of the file bytes.
 * Registry definitions are not persisted; schema admission remains a builder
 * responsibility. Here we verify record fields, identities, references, order,
 * and the exact projection committed by the header, without changing v1 IDs.
 */
#define _POSIX_C_SOURCE 200809L
#include "elpis_semantic/segment.h"
#include "elpis_semantic/admission_populated_b1l.h"
#include "elpis_semantic/hacf_mapping.h"
#include "builder_internal.h"
#include <stdio.h>
#include <fcntl.h>
#include <unistd.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <stdint.h>

static int digest_equal(const hacf_digest *a, const hacf_digest *b) {
    return memcmp(a->bytes, b->bytes, HACF_DIGEST_BYTES) == 0;
}

static int add_record_bytes(size_t *total, uint32_t count, size_t record_size) {
    if (count > (SIZE_MAX - *total) / record_size) return SEMANTIC_E_INVAL;
    *total += (size_t)count * record_size;
    /* No allocation or pointer subtraction may exceed the addressable range. */
    return *total > PTRDIFF_MAX ? SEMANTIC_E_INVAL : SEMANTIC_OK;
}

/* Validate the same structural references required by the production builder.
 * The arrays are already in writer order; do not silently sort or deduplicate. */
static int validate_references(const semantic_hypergraph_builder *b) {
    for (uint32_t i = 0; i < b->assertion_count; ++i) {
        const elpis_semantic_assertion_v1 *a = &b->assertions[i];
        if (a->asserted_object_kind == SEMANTIC_OBJECT_KIND_NODE) {
            if (!semantic_builder_has_node(b, &a->asserted_object_digest))
                return SEMANTIC_E_INVAL;
        } else if (!semantic_builder_has_hyperedge(b, &a->asserted_object_digest)) {
            return SEMANTIC_E_INVAL;
        }
    }
    for (uint32_t i = 0; i < b->hyperedge_count; ++i) {
        const elpis_semantic_hyperedge_v1 *e = &b->hyperedges[i];
        for (uint32_t j = 0; j < e->participant_count; ++j) {
            if (!semantic_builder_has_node(b, &e->participants[j].node_identity))
                return SEMANTIC_E_INVAL;
        }
    }
    for (uint32_t i = 0; i < b->incidence_count; ++i) {
        const elpis_semantic_incidence_v1 *inc = &b->incidences[i];
        if (!semantic_builder_has_node(b, &inc->node_digest) ||
            !semantic_builder_has_hyperedge(b, &inc->hyperedge_digest))
            return SEMANTIC_E_INVAL;
    }
    return SEMANTIC_OK;
}

static int read_segment_checked(const char *path,
                           semantic_segment_record *segment_out,
                           hacf_digest *segment_digest_out,
                           const hacf_digest *target, uint32_t *matches,
                           hacf_digest **inventory, uint32_t *inventory_count,
                           const uint32_t *typed_node_type,
                           const hacf_digest *typed_payload,
                           uint32_t *typed_matches) {
    if (!path || !segment_out || ((target == NULL) != (matches == NULL)) ||
        ((inventory == NULL) != (inventory_count == NULL)) ||
        ((typed_node_type == NULL) != (typed_payload == NULL)) ||
        ((typed_node_type == NULL) != (typed_matches == NULL)) ||
        (target && inventory) || (target && typed_node_type) ||
        (inventory && typed_node_type)) return SEMANTIC_E_INVAL;
    /* Preserve the established stream contract of semantic_segment_read():
     * its FIFO tests require a blocking reader while a child opens the writer.
     * Membership verification is a stronger trust boundary: it must never
     * wait for an untrusted FIFO, nor accept nonregular input. Both paths use
     * one symlink-refusing descriptor for the entire validation. */
    int flags = O_RDONLY | O_CLOEXEC | O_NOFOLLOW;
    if (target || inventory || typed_node_type) flags |= O_NONBLOCK;
    int fd = open(path, flags);
    if (fd < 0) return SEMANTIC_E_IO;
    struct stat opened;
    if (fstat(fd, &opened) != 0 || ((target || inventory || typed_node_type) && !S_ISREG(opened.st_mode))) {
        close(fd);
        return SEMANTIC_E_IO;
    }
    FILE *f = fdopen(fd, "rb");
    if (!f) { close(fd); return SEMANTIC_E_IO; }

    int rc = SEMANTIC_E_IO;
    semantic_segment_record candidate;
    semantic_hypergraph_builder records = {0};
    hacf_graph_op *ops = NULL;
    hacf_digest computed, delta, next;
    uint32_t op_count = 0;
    hacf_digest *inventory_candidate = NULL;
    uint32_t inventory_length = 0;
    /* Declared before the first goto so every path to done sees them
     * initialized; they are published only on SEMANTIC_OK. */
    uint32_t target_matches = 0;
    uint32_t typed_count = 0;
    if (fread(&candidate, sizeof(candidate), 1, f) != 1) goto done;
    rc = SEMANTIC_E_INVAL;
    if (candidate.abi_version != SEMANTIC_SEGMENT_ABI_VERSION) goto done;
    const uint8_t zero[sizeof(candidate.reserved)] = {0};
    rc = SEMANTIC_E_RESERVATION;
    if (memcmp(candidate.reserved, zero, sizeof(zero)) != 0) goto done;
    rc = semantic_segment_identity(&candidate, &computed);
    if (rc != SEMANTIC_OK) goto done;
    rc = SEMANTIC_E_DIGEST;
    if (!digest_equal(&computed, &candidate.segment_identity) ||
        !digest_equal(&computed, &candidate.hacf_package_digest)) goto done;

    size_t expected = sizeof(candidate);
    rc = SEMANTIC_E_INVAL;
    if (add_record_bytes(&expected, candidate.node_count, sizeof(*records.nodes)) ||
        add_record_bytes(&expected, candidate.assertion_count, sizeof(*records.assertions)) ||
        add_record_bytes(&expected, candidate.hyperedge_count, sizeof(*records.hyperedges)) ||
        add_record_bytes(&expected, candidate.incidence_count, sizeof(*records.incidences)))
        goto done;
    /* Reject inconsistent regular-file lengths before allocating from counts.
     * Streams still require complete record reads and a final EOF check. */
    struct stat st;
    rc = SEMANTIC_E_IO;
    if (fstat(fileno(f), &st) != 0) goto done;
    if (S_ISREG(st.st_mode)) {
        if (st.st_size < 0 || (uintmax_t)st.st_size < expected) goto done;
        rc = SEMANTIC_E_INVAL;
        if ((uintmax_t)st.st_size != expected) goto done;
    }

    /* The writer emits these four arrays, in this order, as full native structs.
     * Each validator runs before identity hashing (notably participant bounds). */
#define READ_RECORDS(member, count_field, prefix, identity_field) do { \
    records.count_field = candidate.count_field; \
    if (records.count_field) { \
        records.member = malloc((size_t)records.count_field * sizeof(*records.member)); \
        rc = SEMANTIC_E_NOMEM; \
        if (!records.member) goto done; \
    } \
    for (uint32_t i = 0; i < records.count_field; ++i) { \
        rc = SEMANTIC_E_IO; \
        if (fread(&records.member[i], sizeof(*records.member), 1, f) != 1) goto done; \
        rc = prefix##_validate(&records.member[i]); \
        if (rc != SEMANTIC_OK) goto done; \
        hacf_digest identity; \
        rc = prefix##_identity(&records.member[i], &identity); \
        if (rc != SEMANTIC_OK) goto done; \
        rc = SEMANTIC_E_DIGEST; \
        if (!digest_equal(&identity, &records.member[i].identity_field)) goto done; \
        rc = SEMANTIC_E_INVAL; \
        if (i && prefix##_cmp(&records.member[i - 1], &records.member[i]) >= 0) goto done; \
    } \
} while (0)
    READ_RECORDS(nodes, node_count, elpis_semantic_node, node_identity);
    READ_RECORDS(assertions, assertion_count, elpis_semantic_assertion, assertion_identity);
    READ_RECORDS(hyperedges, hyperedge_count, elpis_semantic_hyperedge, hyperedge_identity);
    READ_RECORDS(incidences, incidence_count, elpis_semantic_incidence, incidence_identity);
#undef READ_RECORDS

    int trailing = fgetc(f);
    rc = SEMANTIC_E_IO;
    if (ferror(f)) goto done;
    rc = SEMANTIC_E_INVAL;
    if (trailing != EOF) goto done;
    rc = validate_references(&records);
    if (rc != SEMANTIC_OK) goto done;
    rc = semantic_map_to_hacf_ops(&records, &ops, &op_count);
    if (rc != SEMANTIC_OK) goto done;
    rc = SEMANTIC_E_DIGEST;
    if (op_count != candidate.hacf_op_count) goto done;
    rc = semantic_compute_hacf_delta(&candidate.prior_snapshot_digest, ops,
                                     op_count, &delta, &next);
    if (rc != SEMANTIC_OK) goto done;
    rc = SEMANTIC_E_DIGEST;
    if (!digest_equal(&delta, &candidate.hacf_delta_digest) ||
        !digest_equal(&next, &candidate.hacf_next_snapshot)) goto done;
    /* Count a target only after every record, reference and graph projection
     * has been verified. One opened FILE*, no path-based second read. */
    if (typed_node_type) {
        for (uint32_t i = 0; i < records.node_count; ++i) {
            if (records.nodes[i].node_type == *typed_node_type &&
                digest_equal(&records.nodes[i].payload_digest, typed_payload))
                ++typed_count;
        }
    }
    if (target) {
        for (uint32_t i = 0; i < records.node_count; ++i)
            if (digest_equal(&records.nodes[i].node_identity, target))
                ++target_matches;
    }
    /* Transfer only identities from already validated records and projection.
     * The inventory is never filled on a partial/error read. */
    if (inventory) {
        if (records.node_count > SEMANTIC_B1N_MAX_TOTAL_NODES) {
            rc = SEMANTIC_E_INVAL;
            goto done;
        }
        inventory_length = records.node_count;
        if (inventory_length) {
            inventory_candidate = malloc((size_t)inventory_length * sizeof(*inventory_candidate));
            if (!inventory_candidate) { rc = SEMANTIC_E_NOMEM; goto done; }
            for (uint32_t i = 0; i < inventory_length; ++i)
                inventory_candidate[i] = records.nodes[i].node_identity;
        }
    }
    rc = SEMANTIC_OK;

done:
    semantic_free_hacf_ops(ops);
    free(records.nodes);
    free(records.assertions);
    free(records.hyperedges);
    free(records.incidences);
    if (fclose(f) != 0 && rc == SEMANTIC_OK) rc = SEMANTIC_E_IO;
    if (rc == SEMANTIC_OK) {
        *segment_out = candidate;
        if (segment_digest_out) *segment_digest_out = computed;
        if (matches) *matches = target_matches;
        if (typed_matches) *typed_matches = typed_count;
        if (inventory) {
            *inventory = inventory_candidate;
            *inventory_count = inventory_length;
            inventory_candidate = NULL;
        }
    }
    free(inventory_candidate);
    return rc;
}

int semantic_segment_read(const char *path,
                          semantic_segment_record *segment_out,
                          hacf_digest *segment_digest_out) {
    return read_segment_checked(path, segment_out, segment_digest_out, NULL, NULL, NULL, NULL, NULL, NULL, NULL);
}

int semantic_segment_read_node_presence(const char *path,
    const hacf_digest *target_node_identity,
    semantic_segment_record *segment_out,
    hacf_digest *segment_digest_out,
    uint32_t *match_count_out) {
    if (!target_node_identity || !match_count_out) return SEMANTIC_E_INVAL;
    return read_segment_checked(path, segment_out, segment_digest_out,
                                target_node_identity, match_count_out, NULL, NULL, NULL, NULL, NULL);
}

int semantic_segment_read_node_inventory(const char *path,
    semantic_segment_record *segment_out, hacf_digest *segment_digest_out,
    hacf_digest **nodes_out, uint32_t *node_count_out) {
    if (!nodes_out || !node_count_out || !segment_out ||
        nodes_out == (hacf_digest **)(void *)node_count_out)
        return SEMANTIC_E_INVAL;
    return read_segment_checked(path, segment_out, segment_digest_out,
                                NULL, NULL, nodes_out, node_count_out, NULL, NULL, NULL);
}

/* B1p: same validated descriptor, semantic records and HACF projection.
 * Count a typed payload collision regardless of semantic identity flags.
 * The caller owns the trusted manifest and validates chain membership. */
int semantic_segment_read_typed_payload_occurrences(const char *path,
    uint32_t node_type, const hacf_digest *payload_digest,
    semantic_segment_record *segment_out,
    hacf_digest *segment_digest_out, uint32_t *occurrences_out) {
    static const hacf_digest zero = {{0}};
    if (!path || !segment_out || !payload_digest || !occurrences_out ||
        (node_type & 0xF0000000u) != SEMANTIC_NODE_NAMESPACE ||
        memcmp(payload_digest, &zero, sizeof(zero)) == 0 ||
        (void *)occurrences_out == (void *)segment_out ||
        (void *)occurrences_out == (void *)segment_digest_out)
        return SEMANTIC_E_INVAL;
    return read_segment_checked(path,segment_out,segment_digest_out,
                                NULL,NULL,NULL,NULL,&node_type,
                                payload_digest,occurrences_out);
}
