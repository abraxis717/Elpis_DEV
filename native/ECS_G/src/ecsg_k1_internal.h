#ifndef ELPIS_ECSG_K1_INTERNAL_H
#define ELPIS_ECSG_K1_INTERNAL_H

/* Internal interface between the K1 state and its FMS residency adapter (not an exported ABI). A bound state
 * owns its workspace but not its image: the adapter points it at FMS-resident bytes for each operation. */

#include "elpis/ecsg_k1.h"

elpis_ecsg_k1_status ecsg_k1_internal_create_bound(size_t dim, size_t width, size_t max_rows, elpis_ecsg_k1 **out);
void ecsg_k1_internal_bind(elpis_ecsg_k1 *state, uint8_t *image);
size_t ecsg_k1_internal_image_bytes(const elpis_ecsg_k1 *state);
const uint8_t *ecsg_k1_internal_image(const elpis_ecsg_k1 *state);
elpis_ecsg_k1_status ecsg_k1_internal_check_envelope(const uint8_t *envelope, size_t size, size_t *dim, size_t *width);
elpis_ecsg_k1_status ecsg_k1_internal_decode(const elpis_ecsg_k1 *state, const uint8_t *envelope, uint8_t *image);
void ecsg_k1_internal_encode(const elpis_ecsg_k1 *state, const uint8_t *image, uint8_t *out);

#endif
