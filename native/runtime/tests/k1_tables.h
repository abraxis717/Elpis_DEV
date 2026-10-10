/* The K1 function tables the RuntimeCore C tests hand to RuntimeCore: typed wrappers over the K1 library's own
 * leased entry points (ecsg_k1.h / ecsg_k1_fms.h, "Managed ownership"). No casts of function pointers. */
#ifndef ELPIS_RUNTIME_TEST_K1_TABLES_H
#define ELPIS_RUNTIME_TEST_K1_TABLES_H

#include "elpis/ecsg_k1.h"
#include "elpis/runtime.h"

#ifdef ELPIS_TEST_K1_FMS
#include "elpis/ecsg_k1_fms.h"
#endif

_Static_assert(sizeof(elpis_runtime_experience) == sizeof(elpis_ecsg_k1_experience), "experience layout");
_Static_assert(sizeof(elpis_runtime_schedule_result) == sizeof(elpis_ecsg_k1_schedule_result), "schedule layout");
_Static_assert(sizeof(elpis_runtime_commit_identity) == sizeof(elpis_ecsg_k1_commit_identity), "identity layout");

static int k1_digest(void *s, uint8_t out[32]) { return elpis_ecsg_k1_state_digest(s, out); }
static int k1_shape(void *s, size_t *dim, size_t *width) { return elpis_ecsg_k1_shape(s, dim, width); }
static int k1_query(void *s, size_t dim, const double *x, size_t rows, double *out, uint8_t digest[32]) {
    return elpis_ecsg_k1_query_identity(s, dim, x, rows, out, digest);
}
static int k1_claim(void *s, uint64_t lease) { return elpis_ecsg_k1_lease_claim(s, lease); }
static int k1_release(void *s, uint64_t lease) { return elpis_ecsg_k1_lease_release(s, lease); }
static int k1_reserve(void *s, uint64_t lease, size_t rows) { return elpis_ecsg_k1_leased_reserve(s, lease, rows); }
static int k1_begin(void *s, uint64_t lease, uint8_t source[32], uint64_t *token) {
    return elpis_ecsg_k1_leased_txn_begin(s, lease, source, token);
}
static int k1_schedule(void *s, uint64_t lease, uint64_t token, const double *x, const double *y, size_t rows,
                       const elpis_runtime_experience *schedule, size_t n, double rate, double *s3, size_t s3n,
                       elpis_runtime_schedule_result *result) {
    return elpis_ecsg_k1_leased_txn_run_schedule(s, lease, token, x, y, rows,
                                                 (const elpis_ecsg_k1_experience *)schedule, n, rate, s3, s3n,
                                                 (elpis_ecsg_k1_schedule_result *)result);
}
static int k1_commit(void *s, uint64_t lease, uint64_t token, elpis_runtime_commit_identity *identity) {
    return elpis_ecsg_k1_leased_txn_commit_identity(s, lease, token, (elpis_ecsg_k1_commit_identity *)identity);
}
static int k1_abort(void *s, uint64_t token) { return elpis_ecsg_k1_txn_abort(s, token); }

static const elpis_runtime_k1_api K1_API = {k1_digest, k1_shape, k1_query, k1_claim, k1_release,
                                            k1_reserve, k1_begin, k1_schedule, k1_commit, k1_abort};

#ifdef ELPIS_TEST_K1_FMS
static int k1fms_digest(void *r, uint64_t id, uint8_t out[32]) { return elpis_ecsg_k1_fms_state_digest(r, id, out); }
static int k1fms_shape(void *r, uint64_t id, size_t *dim, size_t *width) {
    return elpis_ecsg_k1_fms_shape(r, id, dim, width);
}
static int k1fms_query(void *r, uint64_t id, size_t dim, const double *x, size_t rows, double *out,
                     uint8_t digest[32]) {
    return elpis_ecsg_k1_fms_query_identity(r, id, dim, x, rows, out, digest);
}
static int k1fms_claim(void *r, uint64_t id, uint64_t lease) { return elpis_ecsg_k1_fms_lease_claim(r, id, lease); }
static int k1fms_release(void *r, uint64_t id, uint64_t lease) { return elpis_ecsg_k1_fms_lease_release(r, id, lease); }
static int k1fms_reserve(void *r, uint64_t id, uint64_t lease, size_t rows) {
    return elpis_ecsg_k1_fms_leased_reserve(r, id, lease, rows);
}
static int k1fms_begin(void *r, uint64_t id, uint64_t lease, uint8_t source[32], uint64_t *token) {
    return elpis_ecsg_k1_fms_leased_txn_begin(r, id, lease, source, token);
}
static int k1fms_schedule(void *r, uint64_t id, uint64_t lease, uint64_t token, const double *x, const double *y,
                        size_t rows, const elpis_runtime_experience *schedule, size_t n, double rate, double *s3,
                        size_t s3n, elpis_runtime_schedule_result *result) {
    return elpis_ecsg_k1_fms_leased_txn_run_schedule(r, id, lease, token, x, y, rows,
                                                     (const elpis_ecsg_k1_experience *)schedule, n, rate, s3, s3n,
                                                     (elpis_ecsg_k1_schedule_result *)result);
}
static int k1fms_commit(void *r, uint64_t id, uint64_t lease, uint64_t token, elpis_runtime_commit_identity *identity) {
    return elpis_ecsg_k1_fms_leased_txn_commit_identity(r, id, lease, token,
                                                        (elpis_ecsg_k1_commit_identity *)identity);
}
static int k1fms_abort(void *r, uint64_t id, uint64_t token) { return elpis_ecsg_k1_fms_txn_abort(r, id, token); }

static const elpis_runtime_k1_fms_api FMS_API = {k1fms_digest, k1fms_shape, k1fms_query, k1fms_claim, k1fms_release,
                                                 k1fms_reserve, k1fms_begin, k1fms_schedule, k1fms_commit, k1fms_abort};
#endif

#endif
