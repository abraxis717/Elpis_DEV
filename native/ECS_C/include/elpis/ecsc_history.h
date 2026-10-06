#ifndef ELPIS_ECSC_HISTORY_H
#define ELPIS_ECSC_HISTORY_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

enum {
    ELPIS_ECSC_OK = 0,
    ELPIS_ECSC_INVALID = -1,
    ELPIS_ECSC_CAPACITY = -2,
    ELPIS_ECSC_UTF8 = -3
};

enum {
    ELPIS_ECSC_MAX_BINDINGS = 16,
    ELPIS_ECSC_MAX_NAME_BYTES = 128,
    ELPIS_ECSC_MAX_VALUE_BYTES = 256
};

typedef struct elpis_ecsc_binding_view {
    const char *name;
    size_t name_len;
    const char *value;
    size_t value_len;
} elpis_ecsc_binding_view;

/* Native runtime-history ABI, additive v1 foundation. */
uint32_t elpis_ecsc_history_abi_version(void);

/*
 * Exact native encoding of elpis.runtime.history.ReceiptRecord.payload().
 *
 * Canonical byte form:
 *   json.dumps(..., sort_keys=True, separators=(",", ":"),
 *              ensure_ascii=True, allow_nan=False).encode("ascii")
 *
 * Binding names must already be strictly sorted and unique, matching the
 * existing Python ReceiptRecord invariant.
 */
int elpis_ecsc_receipt_payload_size(
    const char *subsystem,
    size_t subsystem_len,
    const char *kind,
    size_t kind_len,
    const char *digest_hex,
    size_t digest_len,
    const elpis_ecsc_binding_view *bindings,
    size_t binding_count,
    size_t *out_size
);

int elpis_ecsc_receipt_payload_write(
    const char *subsystem,
    size_t subsystem_len,
    const char *kind,
    size_t kind_len,
    const char *digest_hex,
    size_t digest_len,
    const elpis_ecsc_binding_view *bindings,
    size_t binding_count,
    uint8_t *out,
    size_t out_capacity,
    size_t *out_written
);

/*
 * Exact native equivalent of elpis.ECS_C.canonical.digest_bytes(raw):
 *
 *   sha256(canonical_bytes({"bytes": raw.hex()}))
 *
 * Output is lowercase hex plus terminating NUL.
 */
int elpis_ecsc_digest_bytes(
    const void *data,
    size_t size,
    char out_hex[65]
);

#ifdef __cplusplus
}
#endif

#endif
