#ifndef ELPIS_ECSG_MATH_H
#define ELPIS_ECSG_MATH_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

enum {
    ELPIS_ECSG_MATH_ABI_V1 = 1u
};

typedef enum {
    ELPIS_ECSG_MATH_OK = 0,
    ELPIS_ECSG_MATH_INVALID = -1,
    ELPIS_ECSG_MATH_NONFINITE = -2
} elpis_ecsg_math_status;

/*
 * ECS_G mathematical kernel R0.
 *
 * Microscopic state W is contiguous IEEE-754 binary64 with shape [dim,width]
 * in row-major order: W[a*width + i] is coordinate a of entity/column i.
 *
 * R0 inherits the frozen cubic ECS primitive:
 *
 *   phi(z) = 0.5*z + 0.5*z^2 + 0.5*z^3
 *   f_W(x) = sum_i phi(x^T w_i)
 *
 * and the RAW-SUM collective state:
 *
 *   mu_a      = sum_i W[a,i]
 *   M_ab      = sum_i W[a,i] W[b,i]
 *   T3_abc    = sum_i W[a,i] W[b,i] W[c,i]
 *
 * M is packed in lexicographic (a,b), a<=b order.
 * T3 is packed in lexicographic (a,b,c), a<=b<=c order.
 *
 * For dim=6 the packed S3 dimension is exactly 6+21+56 = 83.
 *
 * Every output buffer is caller-owned. The library allocates nothing, retains
 * no caller pointers, creates no threads, performs no I/O and owns no runtime
 * transition law.
 */

uint32_t elpis_ecsg_math_abi_version(void);

size_t elpis_ecsg_symmetric2_size(size_t dim);
size_t elpis_ecsg_symmetric3_size(size_t dim);
size_t elpis_ecsg_s3_size(size_t dim);

elpis_ecsg_math_status
elpis_ecsg_project_s3_f64(const double *w,
                          size_t dim,
                          size_t width,
                          double *mu,
                          double *m_packed,
                          double *t3_packed);

elpis_ecsg_math_status
elpis_ecsg_forward_f64(const double *w,
                       size_t dim,
                       size_t width,
                       const double *x,
                       size_t rows,
                       double *out);

elpis_ecsg_math_status
elpis_ecsg_forward_s3_f64(const double *mu,
                          const double *m_packed,
                          const double *t3_packed,
                          size_t dim,
                          const double *x,
                          size_t rows,
                          double *out);

#ifdef __cplusplus
}
#endif

#endif
