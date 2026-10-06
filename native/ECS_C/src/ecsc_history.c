#include "elpis/ecsc_history.h"
#include "elpis/sha256.h"

#include <limits.h>
#include <stdint.h>
#include <string.h>

#define RECEIPT_SCHEMA "elpis.runtime.receipt-record.v1"

typedef struct writer {
    uint8_t *out;
    size_t cap;
    size_t pos;
} writer;

static int checked_add(size_t a, size_t b, size_t *out)
{
    if (b > SIZE_MAX - a) {
        return ELPIS_ECSC_CAPACITY;
    }
    *out = a + b;
    return ELPIS_ECSC_OK;
}

static int put_bytes(writer *w, const void *data, size_t n)
{
    size_t next;
    int rc;

    if (w == NULL || (data == NULL && n != 0u)) {
        return ELPIS_ECSC_INVALID;
    }
    rc = checked_add(w->pos, n, &next);
    if (rc != ELPIS_ECSC_OK) {
        return rc;
    }
    if (w->out != NULL) {
        if (next > w->cap) {
            return ELPIS_ECSC_CAPACITY;
        }
        if (n != 0u) {
            memcpy(w->out + w->pos, data, n);
        }
    }
    w->pos = next;
    return ELPIS_ECSC_OK;
}

static int put_c(writer *w, char c)
{
    return put_bytes(w, &c, 1u);
}

static int put_literal(writer *w, const char *s)
{
    return s == NULL ? ELPIS_ECSC_INVALID : put_bytes(w, s, strlen(s));
}

static int is_name(const char *s, size_t n)
{
    size_t i;
    unsigned char c;

    if (s == NULL || n == 0u || n > ELPIS_ECSC_MAX_NAME_BYTES) {
        return 0;
    }
    c = (unsigned char)s[0];
    if (c < (unsigned char)'a' || c > (unsigned char)'z') {
        return 0;
    }
    for (i = 1u; i < n; ++i) {
        c = (unsigned char)s[i];
        if (!((c >= (unsigned char)'a' && c <= (unsigned char)'z') ||
              (c >= (unsigned char)'0' && c <= (unsigned char)'9') ||
              c == (unsigned char)'_' ||
              c == (unsigned char)'.' ||
              c == (unsigned char)'-')) {
            return 0;
        }
    }
    return 1;
}

static int is_digest(const char *s, size_t n)
{
    size_t i;
    unsigned char c;

    if (s == NULL || n != 64u) {
        return 0;
    }
    for (i = 0u; i < 64u; ++i) {
        c = (unsigned char)s[i];
        if (!((c >= (unsigned char)'0' && c <= (unsigned char)'9') ||
              (c >= (unsigned char)'a' && c <= (unsigned char)'f'))) {
            return 0;
        }
    }
    return 1;
}

static int continuation(unsigned char c)
{
    return (c & 0xc0u) == 0x80u;
}

/* Strict UTF-8 decoder: rejects overlongs, surrogates and > U+10FFFF. */
static int utf8_next(
    const unsigned char *s,
    size_t n,
    size_t *used,
    uint32_t *cp
)
{
    unsigned char a;
    unsigned char b;
    unsigned char c;
    unsigned char d;

    if (s == NULL || used == NULL || cp == NULL || n == 0u) {
        return ELPIS_ECSC_UTF8;
    }

    a = s[0];

    if (a <= 0x7fu) {
        *used = 1u;
        *cp = (uint32_t)a;
        return ELPIS_ECSC_OK;
    }

    if (a >= 0xc2u && a <= 0xdfu) {
        if (n < 2u || !continuation(s[1])) {
            return ELPIS_ECSC_UTF8;
        }
        *used = 2u;
        *cp = ((uint32_t)(a & 0x1fu) << 6) |
              (uint32_t)(s[1] & 0x3fu);
        return ELPIS_ECSC_OK;
    }

    if (a >= 0xe0u && a <= 0xefu) {
        if (n < 3u) {
            return ELPIS_ECSC_UTF8;
        }
        b = s[1];
        c = s[2];

        if (!continuation(b) || !continuation(c)) {
            return ELPIS_ECSC_UTF8;
        }
        if (a == 0xe0u && b < 0xa0u) {
            return ELPIS_ECSC_UTF8;
        }
        if (a == 0xedu && b >= 0xa0u) {
            return ELPIS_ECSC_UTF8;
        }

        *used = 3u;
        *cp = ((uint32_t)(a & 0x0fu) << 12) |
              ((uint32_t)(b & 0x3fu) << 6) |
              (uint32_t)(c & 0x3fu);
        return ELPIS_ECSC_OK;
    }

    if (a >= 0xf0u && a <= 0xf4u) {
        if (n < 4u) {
            return ELPIS_ECSC_UTF8;
        }
        b = s[1];
        c = s[2];
        d = s[3];

        if (!continuation(b) || !continuation(c) || !continuation(d)) {
            return ELPIS_ECSC_UTF8;
        }
        if (a == 0xf0u && b < 0x90u) {
            return ELPIS_ECSC_UTF8;
        }
        if (a == 0xf4u && b > 0x8fu) {
            return ELPIS_ECSC_UTF8;
        }

        *used = 4u;
        *cp = ((uint32_t)(a & 0x07u) << 18) |
              ((uint32_t)(b & 0x3fu) << 12) |
              ((uint32_t)(c & 0x3fu) << 6) |
              (uint32_t)(d & 0x3fu);
        return ELPIS_ECSC_OK;
    }

    return ELPIS_ECSC_UTF8;
}

static char hex_digit(unsigned v)
{
    static const char h[] = "0123456789abcdef";
    return h[v & 15u];
}

static int put_u_escape(writer *w, uint16_t v)
{
    char e[6];

    e[0] = '\\';
    e[1] = 'u';
    e[2] = hex_digit((unsigned)(v >> 12));
    e[3] = hex_digit((unsigned)(v >> 8));
    e[4] = hex_digit((unsigned)(v >> 4));
    e[5] = hex_digit((unsigned)v);
    return put_bytes(w, e, sizeof e);
}

static int json_string(writer *w, const char *s, size_t n)
{
    const unsigned char *p;
    size_t off = 0u;
    size_t used;
    uint32_t cp;
    int rc;

    if (w == NULL || (s == NULL && n != 0u)) {
        return ELPIS_ECSC_INVALID;
    }

    rc = put_c(w, '"');
    if (rc != ELPIS_ECSC_OK) {
        return rc;
    }

    p = (const unsigned char *)s;

    while (off < n) {
        rc = utf8_next(p + off, n - off, &used, &cp);
        if (rc != ELPIS_ECSC_OK) {
            return rc;
        }
        off += used;

        if (cp == (uint32_t)'"') {
            rc = put_literal(w, "\\\"");
        } else if (cp == (uint32_t)'\\') {
            rc = put_literal(w, "\\\\");
        } else if (cp == 0x08u) {
            rc = put_literal(w, "\\b");
        } else if (cp == 0x09u) {
            rc = put_literal(w, "\\t");
        } else if (cp == 0x0au) {
            rc = put_literal(w, "\\n");
        } else if (cp == 0x0cu) {
            rc = put_literal(w, "\\f");
        } else if (cp == 0x0du) {
            rc = put_literal(w, "\\r");
        } else if (cp < 0x20u) {
            rc = put_u_escape(w, (uint16_t)cp);
        } else if (cp <= 0x7eu) {
            rc = put_c(w, (char)cp);
        } else if (cp <= 0xffffu) {
            rc = put_u_escape(w, (uint16_t)cp);
        } else {
            uint32_t v = cp - 0x10000u;
            uint16_t hi = (uint16_t)(0xd800u + (v >> 10));
            uint16_t lo = (uint16_t)(0xdc00u + (v & 0x3ffu));

            rc = put_u_escape(w, hi);
            if (rc == ELPIS_ECSC_OK) {
                rc = put_u_escape(w, lo);
            }
        }

        if (rc != ELPIS_ECSC_OK) {
            return rc;
        }
    }

    return put_c(w, '"');
}

static int compare_names(
    const char *a,
    size_t an,
    const char *b,
    size_t bn
)
{
    size_t n = an < bn ? an : bn;
    int c = memcmp(a, b, n);

    if (c != 0) {
        return c;
    }
    return an < bn ? -1 : (an > bn ? 1 : 0);
}

static int validate_record(
    const char *subsystem,
    size_t subsystem_len,
    const char *kind,
    size_t kind_len,
    const char *digest_hex,
    size_t digest_len,
    const elpis_ecsc_binding_view *bindings,
    size_t binding_count
)
{
    size_t i;

    if (!is_name(subsystem, subsystem_len) ||
        !is_name(kind, kind_len) ||
        !is_digest(digest_hex, digest_len) ||
        binding_count > ELPIS_ECSC_MAX_BINDINGS ||
        (bindings == NULL && binding_count != 0u)) {
        return ELPIS_ECSC_INVALID;
    }

    for (i = 0u; i < binding_count; ++i) {
        if (!is_name(bindings[i].name, bindings[i].name_len) ||
            bindings[i].value == NULL ||
            bindings[i].value_len > ELPIS_ECSC_MAX_VALUE_BYTES) {
            return ELPIS_ECSC_INVALID;
        }
        if (i != 0u &&
            compare_names(
                bindings[i - 1u].name,
                bindings[i - 1u].name_len,
                bindings[i].name,
                bindings[i].name_len
            ) >= 0) {
            return ELPIS_ECSC_INVALID;
        }
    }

    return ELPIS_ECSC_OK;
}

static int encode_record(
    writer *w,
    const char *subsystem,
    size_t subsystem_len,
    const char *kind,
    size_t kind_len,
    const char *digest_hex,
    size_t digest_len,
    const elpis_ecsc_binding_view *bindings,
    size_t binding_count
)
{
    size_t i;
    int rc;

    rc = validate_record(
        subsystem,
        subsystem_len,
        kind,
        kind_len,
        digest_hex,
        digest_len,
        bindings,
        binding_count
    );
    if (rc != ELPIS_ECSC_OK) {
        return rc;
    }

    rc = put_literal(w, "{\"bindings\":{");
    if (rc != ELPIS_ECSC_OK) {
        return rc;
    }

    for (i = 0u; i < binding_count; ++i) {
        if (i != 0u) {
            rc = put_c(w, ',');
            if (rc != ELPIS_ECSC_OK) {
                return rc;
            }
        }

        rc = json_string(w, bindings[i].name, bindings[i].name_len);
        if (rc != ELPIS_ECSC_OK) {
            return rc;
        }
        rc = put_c(w, ':');
        if (rc != ELPIS_ECSC_OK) {
            return rc;
        }
        rc = json_string(w, bindings[i].value, bindings[i].value_len);
        if (rc != ELPIS_ECSC_OK) {
            return rc;
        }
    }

    rc = put_literal(w, "},\"digest\":");
    if (rc != ELPIS_ECSC_OK) {
        return rc;
    }
    rc = json_string(w, digest_hex, digest_len);
    if (rc != ELPIS_ECSC_OK) {
        return rc;
    }

    rc = put_literal(w, ",\"kind\":");
    if (rc != ELPIS_ECSC_OK) {
        return rc;
    }
    rc = json_string(w, kind, kind_len);
    if (rc != ELPIS_ECSC_OK) {
        return rc;
    }

    rc = put_literal(w, ",\"schema\":\"" RECEIPT_SCHEMA "\",\"subsystem\":");
    if (rc != ELPIS_ECSC_OK) {
        return rc;
    }
    rc = json_string(w, subsystem, subsystem_len);
    if (rc != ELPIS_ECSC_OK) {
        return rc;
    }

    return put_c(w, '}');
}

uint32_t elpis_ecsc_history_abi_version(void)
{
    return UINT32_C(1);
}

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
)
{
    writer w = {NULL, 0u, 0u};
    int rc;

    if (out_size == NULL) {
        return ELPIS_ECSC_INVALID;
    }
    *out_size = 0u;

    rc = encode_record(
        &w,
        subsystem,
        subsystem_len,
        kind,
        kind_len,
        digest_hex,
        digest_len,
        bindings,
        binding_count
    );
    if (rc == ELPIS_ECSC_OK) {
        *out_size = w.pos;
    }
    return rc;
}

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
)
{
    size_t need = 0u;
    writer w;
    int rc;

    if (out_written == NULL || (out == NULL && out_capacity != 0u)) {
        return ELPIS_ECSC_INVALID;
    }
    *out_written = 0u;

    rc = elpis_ecsc_receipt_payload_size(
        subsystem,
        subsystem_len,
        kind,
        kind_len,
        digest_hex,
        digest_len,
        bindings,
        binding_count,
        &need
    );
    if (rc != ELPIS_ECSC_OK) {
        return rc;
    }
    if (need > out_capacity || (need != 0u && out == NULL)) {
        return ELPIS_ECSC_CAPACITY;
    }

    w.out = out;
    w.cap = out_capacity;
    w.pos = 0u;

    rc = encode_record(
        &w,
        subsystem,
        subsystem_len,
        kind,
        kind_len,
        digest_hex,
        digest_len,
        bindings,
        binding_count
    );
    if (rc == ELPIS_ECSC_OK) {
        *out_written = w.pos;
    }
    return rc;
}

int elpis_ecsc_digest_bytes(
    const void *data,
    size_t size,
    char out_hex[65]
)
{
    static const char hex[] = "0123456789abcdef";
    static const char prefix[] = "{\"bytes\":\"";
    static const char suffix[] = "\"}";
    const uint8_t *p = (const uint8_t *)data;
    elpis_sha256_ctx ctx;
    uint8_t digest[32];
    char pair[2];
    size_t i;

    if (out_hex == NULL || (data == NULL && size != 0u)) {
        return ELPIS_ECSC_INVALID;
    }

    elpis_sha256_init(&ctx);
    elpis_sha256_update(&ctx, prefix, sizeof prefix - 1u);

    for (i = 0u; i < size; ++i) {
        pair[0] = hex[p[i] >> 4];
        pair[1] = hex[p[i] & 0x0fu];
        elpis_sha256_update(&ctx, pair, sizeof pair);
    }

    elpis_sha256_update(&ctx, suffix, sizeof suffix - 1u);
    elpis_sha256_final(&ctx, digest);
    elpis_hex32(digest, out_hex);

    return ELPIS_ECSC_OK;
}
