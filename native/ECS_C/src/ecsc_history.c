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

#define ELPIS_ECSC_MAX_INT UINT64_C(9223372036854775807)
#define ELPIS_ECSC_MAX_PAYLOAD_BYTES 65536u
#define ELPIS_ECSC_MAX_FRAME_BYTES 262144u

static size_t u64_decimal(uint64_t value, char out[20])
{
    char reverse[20];
    size_t n = 0u;
    size_t i;

    do {
        reverse[n++] = (char)('0' + (value % UINT64_C(10)));
        value /= UINT64_C(10);
    } while (value != 0u);

    for (i = 0u; i < n; ++i) {
        out[i] = reverse[n - i - 1u];
    }

    return n;
}

static int valid_entity_id(const char id[64])
{
    return is_digest(id, 64u);
}

static int valid_message_inputs(
    const char sender_entity_id[64],
    const char receiver_entity_id[64],
    uint64_t sequence,
    const void *payload,
    size_t payload_size
)
{
    if (!valid_entity_id(sender_entity_id) ||
        !valid_entity_id(receiver_entity_id) ||
        sequence == 0u ||
        sequence > ELPIS_ECSC_MAX_INT ||
        payload == NULL ||
        payload_size == 0u ||
        payload_size > ELPIS_ECSC_MAX_PAYLOAD_BYTES) {
        return ELPIS_ECSC_INVALID;
    }

    return ELPIS_ECSC_OK;
}

int elpis_ecsc_message_id(
    const char sender_entity_id[64],
    const char receiver_entity_id[64],
    uint64_t sequence,
    const void *payload,
    size_t payload_size,
    char out_message_id[65],
    char out_payload_digest[65]
)
{
    static const char prefix[] =
        "{\"domain\":\"ecs.message.v1\",\"payload\":{"
        "\"payload_digest\":\"";
    static const char receiver_key[] =
        "\",\"receiver_entity_id\":\"";
    static const char sender_key[] =
        "\",\"sender_entity_id\":\"";
    static const char sequence_key[] =
        "\",\"sequence\":";
    static const char suffix[] = "}}";

    elpis_sha256_ctx ctx;
    uint8_t digest[32];
    char sequence_ascii[20];
    size_t sequence_len;
    int rc;

    if (out_message_id == NULL || out_payload_digest == NULL) {
        return ELPIS_ECSC_INVALID;
    }

    out_message_id[0] = '\0';
    out_payload_digest[0] = '\0';

    rc = valid_message_inputs(
        sender_entity_id,
        receiver_entity_id,
        sequence,
        payload,
        payload_size
    );
    if (rc != ELPIS_ECSC_OK) {
        return rc;
    }

    rc = elpis_ecsc_digest_bytes(
        payload,
        payload_size,
        out_payload_digest
    );
    if (rc != ELPIS_ECSC_OK) {
        return rc;
    }

    sequence_len = u64_decimal(sequence, sequence_ascii);

    elpis_sha256_init(&ctx);
    elpis_sha256_update(&ctx, prefix, sizeof prefix - 1u);
    elpis_sha256_update(&ctx, out_payload_digest, 64u);
    elpis_sha256_update(&ctx, receiver_key, sizeof receiver_key - 1u);
    elpis_sha256_update(&ctx, receiver_entity_id, 64u);
    elpis_sha256_update(&ctx, sender_key, sizeof sender_key - 1u);
    elpis_sha256_update(&ctx, sender_entity_id, 64u);
    elpis_sha256_update(&ctx, sequence_key, sizeof sequence_key - 1u);
    elpis_sha256_update(&ctx, sequence_ascii, sequence_len);
    elpis_sha256_update(&ctx, suffix, sizeof suffix - 1u);
    elpis_sha256_final(&ctx, digest);

    elpis_hex32(digest, out_message_id);
    return ELPIS_ECSC_OK;
}

static int encode_envelope(
    writer *w,
    const char sender_entity_id[64],
    const char receiver_entity_id[64],
    uint64_t sequence,
    const void *payload,
    size_t payload_size,
    uint64_t logical_clock,
    char out_message_id[65],
    char out_payload_digest[65]
)
{
    static const char payload_hex_chars[] = "0123456789abcdef";
    const uint8_t *bytes = (const uint8_t *)payload;
    char sequence_ascii[20];
    char clock_ascii[20];
    size_t sequence_len;
    size_t clock_len;
    size_t i;
    int rc;

    if (w == NULL ||
        logical_clock > ELPIS_ECSC_MAX_INT ||
        out_message_id == NULL ||
        out_payload_digest == NULL) {
        return ELPIS_ECSC_INVALID;
    }

    rc = elpis_ecsc_message_id(
        sender_entity_id,
        receiver_entity_id,
        sequence,
        payload,
        payload_size,
        out_message_id,
        out_payload_digest
    );
    if (rc != ELPIS_ECSC_OK) {
        return rc;
    }

    sequence_len = u64_decimal(sequence, sequence_ascii);
    clock_len = u64_decimal(logical_clock, clock_ascii);

    rc = put_literal(w, "{\"logical_clock\":");
    if (rc != ELPIS_ECSC_OK) return rc;

    rc = put_bytes(w, clock_ascii, clock_len);
    if (rc != ELPIS_ECSC_OK) return rc;

    rc = put_literal(w, ",\"message_id\":\"");
    if (rc != ELPIS_ECSC_OK) return rc;

    rc = put_bytes(w, out_message_id, 64u);
    if (rc != ELPIS_ECSC_OK) return rc;

    rc = put_literal(w, "\",\"payload_digest\":\"");
    if (rc != ELPIS_ECSC_OK) return rc;

    rc = put_bytes(w, out_payload_digest, 64u);
    if (rc != ELPIS_ECSC_OK) return rc;

    rc = put_literal(w, "\",\"payload_hex\":\"");
    if (rc != ELPIS_ECSC_OK) return rc;

    for (i = 0u; i < payload_size; ++i) {
        char pair[2];

        pair[0] = payload_hex_chars[bytes[i] >> 4];
        pair[1] = payload_hex_chars[bytes[i] & 0x0fu];

        rc = put_bytes(w, pair, sizeof pair);
        if (rc != ELPIS_ECSC_OK) return rc;
    }

    rc = put_literal(w, "\",\"receiver_entity_id\":\"");
    if (rc != ELPIS_ECSC_OK) return rc;

    rc = put_bytes(w, receiver_entity_id, 64u);
    if (rc != ELPIS_ECSC_OK) return rc;

    rc = put_literal(w, "\",\"schema\":\"ecs.message.v1\","
                         "\"sender_entity_id\":\"");
    if (rc != ELPIS_ECSC_OK) return rc;

    rc = put_bytes(w, sender_entity_id, 64u);
    if (rc != ELPIS_ECSC_OK) return rc;

    rc = put_literal(w, "\",\"sequence\":");
    if (rc != ELPIS_ECSC_OK) return rc;

    rc = put_bytes(w, sequence_ascii, sequence_len);
    if (rc != ELPIS_ECSC_OK) return rc;

    return put_c(w, '}');
}

int elpis_ecsc_envelope_size(
    const char sender_entity_id[64],
    const char receiver_entity_id[64],
    uint64_t sequence,
    const void *payload,
    size_t payload_size,
    uint64_t logical_clock,
    size_t *out_size
)
{
    writer w = {NULL, 0u, 0u};
    char message_id[65];
    char payload_digest[65];
    int rc;

    if (out_size == NULL) {
        return ELPIS_ECSC_INVALID;
    }

    *out_size = 0u;

    rc = encode_envelope(
        &w,
        sender_entity_id,
        receiver_entity_id,
        sequence,
        payload,
        payload_size,
        logical_clock,
        message_id,
        payload_digest
    );

    if (rc == ELPIS_ECSC_OK) {
        *out_size = w.pos;
    }

    return rc;
}

int elpis_ecsc_envelope_write(
    const char sender_entity_id[64],
    const char receiver_entity_id[64],
    uint64_t sequence,
    const void *payload,
    size_t payload_size,
    uint64_t logical_clock,
    uint8_t *out,
    size_t out_capacity,
    size_t *out_written,
    char out_message_id[65],
    char out_payload_digest[65]
)
{
    size_t need = 0u;
    writer w;
    int rc;

    if (out_written == NULL ||
        out_message_id == NULL ||
        out_payload_digest == NULL ||
        (out == NULL && out_capacity != 0u)) {
        return ELPIS_ECSC_INVALID;
    }

    *out_written = 0u;
    out_message_id[0] = '\0';
    out_payload_digest[0] = '\0';

    rc = elpis_ecsc_envelope_size(
        sender_entity_id,
        receiver_entity_id,
        sequence,
        payload,
        payload_size,
        logical_clock,
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

    rc = encode_envelope(
        &w,
        sender_entity_id,
        receiver_entity_id,
        sequence,
        payload,
        payload_size,
        logical_clock,
        out_message_id,
        out_payload_digest
    );

    if (rc == ELPIS_ECSC_OK) {
        *out_written = w.pos;
    }

    return rc;
}

int elpis_ecsc_event_frame_size(
    size_t canonical_event_size,
    size_t *out_size
)
{
    if (out_size == NULL) {
        return ELPIS_ECSC_INVALID;
    }

    *out_size = 0u;

    if (canonical_event_size == 0u ||
        canonical_event_size > ELPIS_ECSC_MAX_FRAME_BYTES) {
        return ELPIS_ECSC_INVALID;
    }

    if (canonical_event_size > SIZE_MAX - 8u) {
        return ELPIS_ECSC_CAPACITY;
    }

    *out_size = canonical_event_size + 8u;
    return ELPIS_ECSC_OK;
}

int elpis_ecsc_event_frame_write(
    const void *canonical_event,
    size_t canonical_event_size,
    uint8_t *out,
    size_t out_capacity,
    size_t *out_written
)
{
    size_t need = 0u;
    uint64_t n;
    size_t i;
    int rc;

    if (out_written == NULL ||
        canonical_event == NULL ||
        out == NULL) {
        return ELPIS_ECSC_INVALID;
    }

    *out_written = 0u;

    rc = elpis_ecsc_event_frame_size(
        canonical_event_size,
        &need
    );
    if (rc != ELPIS_ECSC_OK) {
        return rc;
    }

    if (need > out_capacity) {
        return ELPIS_ECSC_CAPACITY;
    }

    n = (uint64_t)canonical_event_size;

    for (i = 0u; i < 8u; ++i) {
        out[7u - i] = (uint8_t)(n & UINT64_C(0xff));
        n >>= 8;
    }

    memcpy(
        out + 8u,
        canonical_event,
        canonical_event_size
    );

    *out_written = need;
    return ELPIS_ECSC_OK;
}

typedef struct event_sink {
    uint8_t *out;
    size_t cap;
    size_t pos;
    elpis_sha256_ctx *hash;
} event_sink;

static int es_put(event_sink *s, const void *data, size_t n)
{
    size_t next;
    int rc;

    if (s == NULL || (data == NULL && n != 0u)) {
        return ELPIS_ECSC_INVALID;
    }

    rc = checked_add(s->pos, n, &next);
    if (rc != ELPIS_ECSC_OK) {
        return rc;
    }

    if (s->out != NULL) {
        if (next > s->cap) {
            return ELPIS_ECSC_CAPACITY;
        }
        if (n != 0u) {
            memcpy(s->out + s->pos, data, n);
        }
    }

    if (s->hash != NULL && n != 0u) {
        elpis_sha256_update(s->hash, data, n);
    }

    s->pos = next;
    return ELPIS_ECSC_OK;
}

static int es_literal(event_sink *s, const char *text)
{
    if (text == NULL) {
        return ELPIS_ECSC_INVALID;
    }
    return es_put(s, text, strlen(text));
}

static int es_u64(event_sink *s, uint64_t value)
{
    char text[20];
    size_t n = u64_decimal(value, text);
    return es_put(s, text, n);
}

static int es_hex_payload(
    event_sink *s,
    const void *data,
    size_t size
)
{
    static const char h[] = "0123456789abcdef";
    const uint8_t *p = (const uint8_t *)data;
    size_t i;
    int rc;

    if (data == NULL && size != 0u) {
        return ELPIS_ECSC_INVALID;
    }

    for (i = 0u; i < size; ++i) {
        char pair[2];

        pair[0] = h[p[i] >> 4];
        pair[1] = h[p[i] & 0x0fu];

        rc = es_put(s, pair, sizeof pair);
        if (rc != ELPIS_ECSC_OK) {
            return rc;
        }
    }

    return ELPIS_ECSC_OK;
}

typedef struct enqueue_event_context {
    const char *sender;
    const char *receiver;
    uint64_t sequence;
    const void *payload;
    size_t payload_size;
    uint64_t logical_clock;
    char message_id[65];
    char payload_content_digest[65];
} enqueue_event_context;

typedef struct processed_event_context {
    const char *receiver;
    const char *message_id;
} processed_event_context;

static int encode_event_envelope(
    event_sink *s,
    const enqueue_event_context *ctx
)
{
    int rc;

    rc = es_literal(s, "{\"logical_clock\":");
    if (rc != ELPIS_ECSC_OK) return rc;

    rc = es_u64(s, ctx->logical_clock);
    if (rc != ELPIS_ECSC_OK) return rc;

    rc = es_literal(s, ",\"message_id\":\"");
    if (rc != ELPIS_ECSC_OK) return rc;

    rc = es_put(s, ctx->message_id, 64u);
    if (rc != ELPIS_ECSC_OK) return rc;

    rc = es_literal(s, "\",\"payload_digest\":\"");
    if (rc != ELPIS_ECSC_OK) return rc;

    rc = es_put(s, ctx->payload_content_digest, 64u);
    if (rc != ELPIS_ECSC_OK) return rc;

    rc = es_literal(s, "\",\"payload_hex\":\"");
    if (rc != ELPIS_ECSC_OK) return rc;

    rc = es_hex_payload(s, ctx->payload, ctx->payload_size);
    if (rc != ELPIS_ECSC_OK) return rc;

    rc = es_literal(s, "\",\"receiver_entity_id\":\"");
    if (rc != ELPIS_ECSC_OK) return rc;

    rc = es_put(s, ctx->receiver, 64u);
    if (rc != ELPIS_ECSC_OK) return rc;

    rc = es_literal(
        s,
        "\",\"schema\":\"ecs.message.v1\","
        "\"sender_entity_id\":\""
    );
    if (rc != ELPIS_ECSC_OK) return rc;

    rc = es_put(s, ctx->sender, 64u);
    if (rc != ELPIS_ECSC_OK) return rc;

    rc = es_literal(s, "\",\"sequence\":");
    if (rc != ELPIS_ECSC_OK) return rc;

    rc = es_u64(s, ctx->sequence);
    if (rc != ELPIS_ECSC_OK) return rc;

    return es_literal(s, "}");
}

static int encode_enqueue_payload(
    event_sink *s,
    const void *opaque
)
{
    const enqueue_event_context *ctx =
        (const enqueue_event_context *)opaque;
    int rc;

    rc = es_literal(s, "{\"envelope\":");
    if (rc != ELPIS_ECSC_OK) {
        return rc;
    }

    rc = encode_event_envelope(s, ctx);
    if (rc != ELPIS_ECSC_OK) {
        return rc;
    }

    return es_literal(s, "}");
}

static int encode_processed_payload(
    event_sink *s,
    const void *opaque
)
{
    const processed_event_context *ctx =
        (const processed_event_context *)opaque;
    int rc;

    rc = es_literal(s, "{\"message_id\":\"");
    if (rc != ELPIS_ECSC_OK) return rc;

    rc = es_put(s, ctx->message_id, 64u);
    if (rc != ELPIS_ECSC_OK) return rc;

    rc = es_literal(s, "\",\"receiver_entity_id\":\"");
    if (rc != ELPIS_ECSC_OK) return rc;

    rc = es_put(s, ctx->receiver, 64u);
    if (rc != ELPIS_ECSC_OK) return rc;

    return es_literal(s, "\"}");
}

typedef int (*payload_encoder_fn)(
    event_sink *,
    const void *
);

static int digest_payload(
    payload_encoder_fn encoder,
    const void *ctx,
    char out_digest[65]
)
{
    event_sink sink;
    elpis_sha256_ctx hash;
    uint8_t digest[32];
    int rc;

    if (encoder == NULL || out_digest == NULL) {
        return ELPIS_ECSC_INVALID;
    }

    elpis_sha256_init(&hash);

    sink.out = NULL;
    sink.cap = 0u;
    sink.pos = 0u;
    sink.hash = &hash;

    rc = encoder(&sink, ctx);
    if (rc != ELPIS_ECSC_OK) {
        return rc;
    }

    elpis_sha256_final(&hash, digest);
    elpis_hex32(digest, out_digest);

    return ELPIS_ECSC_OK;
}

static int event_transaction(
    event_sink *s,
    const char *prefix,
    const char message_id[64]
)
{
    int rc;

    rc = es_put(s, prefix, strlen(prefix));
    if (rc != ELPIS_ECSC_OK) {
        return rc;
    }

    return es_put(s, message_id, 64u);
}

static int encode_message_event_object(
    event_sink *s,
    int include_after_root,
    int include_event_digest,
    const char *event_kind,
    const char *transaction_prefix,
    const char entity_id[64],
    const char message_id[64],
    uint64_t event_index,
    uint64_t logical_clock,
    payload_encoder_fn payload_encoder,
    const void *payload_context,
    const char payload_digest[64],
    const char before_state_root[64],
    const char after_state_root[64],
    const char prev_event_digest[64],
    const char event_digest[64]
)
{
    int rc;

    rc = es_literal(s, "{");
    if (rc != ELPIS_ECSC_OK) return rc;

    if (include_after_root) {
        rc = es_literal(s, "\"after_state_root\":\"");
        if (rc != ELPIS_ECSC_OK) return rc;

        rc = es_put(s, after_state_root, 64u);
        if (rc != ELPIS_ECSC_OK) return rc;

        rc = es_literal(s, "\",");
        if (rc != ELPIS_ECSC_OK) return rc;
    }

    rc = es_literal(s, "\"before_state_root\":\"");
    if (rc != ELPIS_ECSC_OK) return rc;

    rc = es_put(s, before_state_root, 64u);
    if (rc != ELPIS_ECSC_OK) return rc;

    rc = es_literal(s, "\",\"entity_id\":\"");
    if (rc != ELPIS_ECSC_OK) return rc;

    rc = es_put(s, entity_id, 64u);
    if (rc != ELPIS_ECSC_OK) return rc;

    if (include_event_digest) {
        rc = es_literal(s, "\",\"event_digest\":\"");
        if (rc != ELPIS_ECSC_OK) return rc;

        rc = es_put(s, event_digest, 64u);
        if (rc != ELPIS_ECSC_OK) return rc;
    }

    rc = es_literal(s, "\",\"event_index\":");
    if (rc != ELPIS_ECSC_OK) return rc;

    rc = es_u64(s, event_index);
    if (rc != ELPIS_ECSC_OK) return rc;

    rc = es_literal(s, ",\"event_kind\":\"");
    if (rc != ELPIS_ECSC_OK) return rc;

    rc = es_put(s, event_kind, strlen(event_kind));
    if (rc != ELPIS_ECSC_OK) return rc;

    rc = es_literal(s, "\",\"logical_clock\":");
    if (rc != ELPIS_ECSC_OK) return rc;

    rc = es_u64(s, logical_clock);
    if (rc != ELPIS_ECSC_OK) return rc;

    rc = es_literal(s, ",\"payload\":");
    if (rc != ELPIS_ECSC_OK) return rc;

    rc = payload_encoder(s, payload_context);
    if (rc != ELPIS_ECSC_OK) return rc;

    rc = es_literal(s, ",\"payload_digest\":\"");
    if (rc != ELPIS_ECSC_OK) return rc;

    rc = es_put(s, payload_digest, 64u);
    if (rc != ELPIS_ECSC_OK) return rc;

    rc = es_literal(s, "\",\"prev_event_digest\":\"");
    if (rc != ELPIS_ECSC_OK) return rc;

    rc = es_put(s, prev_event_digest, 64u);
    if (rc != ELPIS_ECSC_OK) return rc;

    rc = es_literal(
        s,
        "\",\"schema\":\"ecs.event.v1\","
        "\"transaction_id\":\""
    );
    if (rc != ELPIS_ECSC_OK) return rc;

    rc = event_transaction(
        s,
        transaction_prefix,
        message_id
    );
    if (rc != ELPIS_ECSC_OK) return rc;

    return es_literal(s, "\"}");
}

static int digest_message_event(
    const char *domain,
    int include_after_root,
    const char *event_kind,
    const char *transaction_prefix,
    const char entity_id[64],
    const char message_id[64],
    uint64_t event_index,
    uint64_t logical_clock,
    payload_encoder_fn payload_encoder,
    const void *payload_context,
    const char payload_digest[64],
    const char before_state_root[64],
    const char after_state_root[64],
    const char prev_event_digest[64],
    char out_digest[65]
)
{
    event_sink sink;
    elpis_sha256_ctx hash;
    uint8_t digest[32];
    int rc;

    if (domain == NULL || out_digest == NULL) {
        return ELPIS_ECSC_INVALID;
    }

    elpis_sha256_init(&hash);

    sink.out = NULL;
    sink.cap = 0u;
    sink.pos = 0u;
    sink.hash = &hash;

    rc = es_literal(&sink, "{\"domain\":\"");
    if (rc != ELPIS_ECSC_OK) return rc;

    rc = es_put(&sink, domain, strlen(domain));
    if (rc != ELPIS_ECSC_OK) return rc;

    rc = es_literal(&sink, "\",\"payload\":");
    if (rc != ELPIS_ECSC_OK) return rc;

    rc = encode_message_event_object(
        &sink,
        include_after_root,
        0,
        event_kind,
        transaction_prefix,
        entity_id,
        message_id,
        event_index,
        logical_clock,
        payload_encoder,
        payload_context,
        payload_digest,
        before_state_root,
        after_state_root,
        prev_event_digest,
        NULL
    );
    if (rc != ELPIS_ECSC_OK) return rc;

    rc = es_literal(&sink, "}");
    if (rc != ELPIS_ECSC_OK) return rc;

    elpis_sha256_final(&hash, digest);
    elpis_hex32(digest, out_digest);

    return ELPIS_ECSC_OK;
}

static int validate_committed_event_inputs(
    const char entity_id[64],
    uint64_t event_index,
    const char before_state_root[64],
    const char after_state_root[64],
    const char prev_event_digest[64]
)
{
    if (!valid_entity_id(entity_id) ||
        !is_digest(before_state_root, 64u) ||
        !is_digest(after_state_root, 64u) ||
        !is_digest(prev_event_digest, 64u) ||
        event_index >= ELPIS_ECSC_MAX_INT) {
        return ELPIS_ECSC_INVALID;
    }

    return ELPIS_ECSC_OK;
}

static int build_message_event(
    int enqueue,
    const char *sender_entity_id,
    const char receiver_entity_id[64],
    uint64_t sequence,
    const void *payload,
    size_t payload_size,
    const char supplied_message_id[64],
    uint64_t event_index,
    const char before_state_root[64],
    const char after_state_root[64],
    const char prev_event_digest[64],
    uint8_t *out,
    size_t out_capacity,
    size_t *out_written,
    char out_event_digest[65],
    char out_intent_digest[65],
    char out_message_id[65]
)
{
    enqueue_event_context enqueue_ctx;
    processed_event_context processed_ctx;
    payload_encoder_fn encoder;
    const void *payload_ctx;
    const char *event_kind;
    const char *transaction_prefix;
    const char *message_id;
    uint64_t logical_clock;
    char payload_digest[65];
    char sender_copy[65];
    char receiver_copy[65];
    char supplied_message_id_copy[65];
    char before_state_root_copy[65];
    char after_state_root_copy[65];
    char prev_event_digest_copy[65];
    event_sink sink;
    int rc;

    if (out_written == NULL ||
        out_event_digest == NULL ||
        out_intent_digest == NULL) {
        return ELPIS_ECSC_INVALID;
    }

    /*
     * Validate and snapshot every fixed-size input before writing any output.
     *
     * This deliberately permits natural event chaining where the caller uses
     * the previous call's out_event_digest buffer as this call's
     * prev_event_digest while also reusing it for the new out_event_digest.
     */
    rc = validate_committed_event_inputs(
        receiver_entity_id,
        event_index,
        before_state_root,
        after_state_root,
        prev_event_digest
    );
    if (rc != ELPIS_ECSC_OK) {
        return rc;
    }

    memcpy(receiver_copy, receiver_entity_id, 64u);
    receiver_copy[64] = '\0';

    memcpy(before_state_root_copy, before_state_root, 64u);
    before_state_root_copy[64] = '\0';

    memcpy(after_state_root_copy, after_state_root, 64u);
    after_state_root_copy[64] = '\0';

    memcpy(prev_event_digest_copy, prev_event_digest, 64u);
    prev_event_digest_copy[64] = '\0';

    if (enqueue) {
        if (sender_entity_id == NULL ||
            out_message_id == NULL) {
            return ELPIS_ECSC_INVALID;
        }

        rc = valid_message_inputs(
            sender_entity_id,
            receiver_entity_id,
            sequence,
            payload,
            payload_size
        );
        if (rc != ELPIS_ECSC_OK) {
            return rc;
        }

        memcpy(sender_copy, sender_entity_id, 64u);
        sender_copy[64] = '\0';
    } else {
        if (!is_digest(supplied_message_id, 64u)) {
            return ELPIS_ECSC_INVALID;
        }

        memcpy(
            supplied_message_id_copy,
            supplied_message_id,
            64u
        );
        supplied_message_id_copy[64] = '\0';
    }

    *out_written = 0u;
    out_event_digest[0] = '\0';
    out_intent_digest[0] = '\0';

    if (out_message_id != NULL) {
        out_message_id[0] = '\0';
    }

    logical_clock = event_index + UINT64_C(1);

    if (enqueue) {
        enqueue_ctx.sender = sender_copy;
        enqueue_ctx.receiver = receiver_copy;
        enqueue_ctx.sequence = sequence;
        enqueue_ctx.payload = payload;
        enqueue_ctx.payload_size = payload_size;
        enqueue_ctx.logical_clock = logical_clock;

        rc = elpis_ecsc_message_id(
            sender_entity_id,
            receiver_entity_id,
            sequence,
            payload,
            payload_size,
            enqueue_ctx.message_id,
            enqueue_ctx.payload_content_digest
        );
        if (rc != ELPIS_ECSC_OK) {
            return rc;
        }

        memcpy(
            out_message_id,
            enqueue_ctx.message_id,
            65u
        );

        encoder = encode_enqueue_payload;
        payload_ctx = &enqueue_ctx;
        event_kind = "MESSAGE_ENQUEUED";
        transaction_prefix = "ENQ:";
        message_id = enqueue_ctx.message_id;
    } else {
        processed_ctx.receiver = receiver_copy;
        processed_ctx.message_id = supplied_message_id_copy;

        encoder = encode_processed_payload;
        payload_ctx = &processed_ctx;
        event_kind = "MESSAGE_PROCESSED";
        transaction_prefix = "PROC:";
        message_id = supplied_message_id_copy;
    }

    rc = digest_payload(
        encoder,
        payload_ctx,
        payload_digest
    );
    if (rc != ELPIS_ECSC_OK) {
        return rc;
    }

    rc = digest_message_event(
        "ecs.event.intent.v1",
        0,
        event_kind,
        transaction_prefix,
        receiver_copy,
        message_id,
        event_index,
        logical_clock,
        encoder,
        payload_ctx,
        payload_digest,
        before_state_root_copy,
        after_state_root_copy,
        prev_event_digest_copy,
        out_intent_digest
    );
    if (rc != ELPIS_ECSC_OK) {
        return rc;
    }

    rc = digest_message_event(
        "ecs.event.v1",
        1,
        event_kind,
        transaction_prefix,
        receiver_copy,
        message_id,
        event_index,
        logical_clock,
        encoder,
        payload_ctx,
        payload_digest,
        before_state_root_copy,
        after_state_root_copy,
        prev_event_digest_copy,
        out_event_digest
    );
    if (rc != ELPIS_ECSC_OK) {
        return rc;
    }

    sink.out = out;
    sink.cap = out_capacity;
    sink.pos = 0u;
    sink.hash = NULL;

    rc = encode_message_event_object(
        &sink,
        1,
        1,
        event_kind,
        transaction_prefix,
        receiver_copy,
        message_id,
        event_index,
        logical_clock,
        encoder,
        payload_ctx,
        payload_digest,
        before_state_root_copy,
        after_state_root_copy,
        prev_event_digest_copy,
        out_event_digest
    );
    if (rc != ELPIS_ECSC_OK) {
        return rc;
    }

    *out_written = sink.pos;
    return ELPIS_ECSC_OK;
}

static int message_event_size_common(
    int enqueue,
    const char *sender_entity_id,
    const char receiver_entity_id[64],
    uint64_t sequence,
    const void *payload,
    size_t payload_size,
    const char supplied_message_id[64],
    uint64_t event_index,
    const char before_state_root[64],
    const char after_state_root[64],
    const char prev_event_digest[64],
    size_t *out_size
)
{
    char event_digest[65];
    char intent_digest[65];
    char message_id[65];
    size_t written = 0u;
    int rc;

    if (out_size == NULL) {
        return ELPIS_ECSC_INVALID;
    }

    *out_size = 0u;

    rc = build_message_event(
        enqueue,
        sender_entity_id,
        receiver_entity_id,
        sequence,
        payload,
        payload_size,
        supplied_message_id,
        event_index,
        before_state_root,
        after_state_root,
        prev_event_digest,
        NULL,
        0u,
        &written,
        event_digest,
        intent_digest,
        enqueue ? message_id : NULL
    );

    if (rc == ELPIS_ECSC_OK) {
        *out_size = written;
    }

    return rc;
}

int elpis_ecsc_enqueue_event_size(
    const char sender_entity_id[64],
    const char receiver_entity_id[64],
    uint64_t sequence,
    const void *payload,
    size_t payload_size,
    uint64_t event_index,
    const char before_state_root[64],
    const char after_state_root[64],
    const char prev_event_digest[64],
    size_t *out_size
)
{
    return message_event_size_common(
        1,
        sender_entity_id,
        receiver_entity_id,
        sequence,
        payload,
        payload_size,
        NULL,
        event_index,
        before_state_root,
        after_state_root,
        prev_event_digest,
        out_size
    );
}

int elpis_ecsc_enqueue_event_write(
    const char sender_entity_id[64],
    const char receiver_entity_id[64],
    uint64_t sequence,
    const void *payload,
    size_t payload_size,
    uint64_t event_index,
    const char before_state_root[64],
    const char after_state_root[64],
    const char prev_event_digest[64],
    uint8_t *out,
    size_t out_capacity,
    size_t *out_written,
    char out_event_digest[65],
    char out_intent_digest[65],
    char out_message_id[65]
)
{
    size_t need = 0u;
    int rc;

    rc = elpis_ecsc_enqueue_event_size(
        sender_entity_id,
        receiver_entity_id,
        sequence,
        payload,
        payload_size,
        event_index,
        before_state_root,
        after_state_root,
        prev_event_digest,
        &need
    );
    if (rc != ELPIS_ECSC_OK) {
        return rc;
    }

    if (out == NULL || need > out_capacity) {
        return ELPIS_ECSC_CAPACITY;
    }

    return build_message_event(
        1,
        sender_entity_id,
        receiver_entity_id,
        sequence,
        payload,
        payload_size,
        NULL,
        event_index,
        before_state_root,
        after_state_root,
        prev_event_digest,
        out,
        out_capacity,
        out_written,
        out_event_digest,
        out_intent_digest,
        out_message_id
    );
}

int elpis_ecsc_processed_event_size(
    const char receiver_entity_id[64],
    const char message_id[64],
    uint64_t event_index,
    const char before_state_root[64],
    const char after_state_root[64],
    const char prev_event_digest[64],
    size_t *out_size
)
{
    return message_event_size_common(
        0,
        NULL,
        receiver_entity_id,
        0u,
        NULL,
        0u,
        message_id,
        event_index,
        before_state_root,
        after_state_root,
        prev_event_digest,
        out_size
    );
}

int elpis_ecsc_processed_event_write(
    const char receiver_entity_id[64],
    const char message_id[64],
    uint64_t event_index,
    const char before_state_root[64],
    const char after_state_root[64],
    const char prev_event_digest[64],
    uint8_t *out,
    size_t out_capacity,
    size_t *out_written,
    char out_event_digest[65],
    char out_intent_digest[65]
)
{
    size_t need = 0u;
    int rc;

    rc = elpis_ecsc_processed_event_size(
        receiver_entity_id,
        message_id,
        event_index,
        before_state_root,
        after_state_root,
        prev_event_digest,
        &need
    );
    if (rc != ELPIS_ECSC_OK) {
        return rc;
    }

    if (out == NULL || need > out_capacity) {
        return ELPIS_ECSC_CAPACITY;
    }

    return build_message_event(
        0,
        NULL,
        receiver_entity_id,
        0u,
        NULL,
        0u,
        message_id,
        event_index,
        before_state_root,
        after_state_root,
        prev_event_digest,
        out,
        out_capacity,
        out_written,
        out_event_digest,
        out_intent_digest,
        NULL
    );
}

#define ELPIS_ECSC_MAX_STRING_BYTES 1024u

#define ELPIS_ECSC_SCHED_V1 \
    "active-mailbox-fifo/entity-id-before-founded-activation.v1"

#define ELPIS_ECSC_SCHED_V2 \
    "active-mailbox-global-arrival/founding-index-activation.v2"

static int es_json_string_utf8(
    event_sink *s,
    const char *value,
    size_t size
)
{
    const unsigned char *p =
        (const unsigned char *)value;

    size_t offset = 0u;
    size_t used = 0u;
    uint32_t cp = 0u;
    int rc;

    if (s == NULL ||
        (value == NULL && size != 0u)) {
        return ELPIS_ECSC_INVALID;
    }

    rc = es_literal(s, "\"");
    if (rc != ELPIS_ECSC_OK) {
        return rc;
    }

    while (offset < size) {
        rc = utf8_next(
            p + offset,
            size - offset,
            &used,
            &cp
        );

        if (rc != ELPIS_ECSC_OK) {
            return rc;
        }

        if (cp == (uint32_t)'"') {
            rc = es_literal(s, "\\\"");
        } else if (cp == (uint32_t)'\\') {
            rc = es_literal(s, "\\\\");
        } else if (cp == UINT32_C(0x08)) {
            rc = es_literal(s, "\\b");
        } else if (cp == UINT32_C(0x09)) {
            rc = es_literal(s, "\\t");
        } else if (cp == UINT32_C(0x0a)) {
            rc = es_literal(s, "\\n");
        } else if (cp == UINT32_C(0x0c)) {
            rc = es_literal(s, "\\f");
        } else if (cp == UINT32_C(0x0d)) {
            rc = es_literal(s, "\\r");
        } else if (cp < UINT32_C(0x20)) {
            char escaped[6];

            escaped[0] = '\\';
            escaped[1] = 'u';
            escaped[2] = '0';
            escaped[3] = '0';
            escaped[4] =
                hex_digit((unsigned)(cp >> 4));
            escaped[5] =
                hex_digit((unsigned)cp);

            rc = es_put(
                s,
                escaped,
                sizeof escaped
            );
        } else {
            /*
             * canonical.canonical_bytes uses ensure_ascii=False.
             * Therefore valid non-ASCII UTF-8 is preserved byte-for-byte.
             */
            rc = es_put(
                s,
                p + offset,
                used
            );
        }

        if (rc != ELPIS_ECSC_OK) {
            return rc;
        }

        offset += used;
    }

    return es_literal(s, "\"");
}

static int finish_stream_digest(
    elpis_sha256_ctx *hash,
    char out_digest[65]
)
{
    uint8_t digest[32];

    if (hash == NULL || out_digest == NULL) {
        return ELPIS_ECSC_INVALID;
    }

    elpis_sha256_final(hash, digest);
    elpis_hex32(digest, out_digest);

    return ELPIS_ECSC_OK;
}

int elpis_ecsc_genesis_digest(
    const char *genesis_label,
    size_t genesis_label_len,
    uint32_t scheduler_protocol,
    char out_digest[65]
)
{
    const char *scheduler;
    event_sink sink;
    elpis_sha256_ctx hash;
    int rc;

    if (out_digest == NULL ||
        genesis_label == NULL ||
        genesis_label_len == 0u ||
        genesis_label_len > ELPIS_ECSC_MAX_STRING_BYTES) {
        return ELPIS_ECSC_INVALID;
    }

    out_digest[0] = '\0';

    if (scheduler_protocol ==
        ELPIS_ECSC_SCHEDULER_V1) {
        scheduler = ELPIS_ECSC_SCHED_V1;
    } else if (scheduler_protocol ==
               ELPIS_ECSC_SCHEDULER_V2) {
        scheduler = ELPIS_ECSC_SCHED_V2;
    } else {
        return ELPIS_ECSC_INVALID;
    }

    elpis_sha256_init(&hash);

    sink.out = NULL;
    sink.cap = 0u;
    sink.pos = 0u;
    sink.hash = &hash;

    rc = es_literal(
        &sink,
        "{\"domain\":\"ecs.genesis.v1\","
        "\"payload\":{\"genesis_label\":"
    );
    if (rc != ELPIS_ECSC_OK) {
        return rc;
    }

    rc = es_json_string_utf8(
        &sink,
        genesis_label,
        genesis_label_len
    );
    if (rc != ELPIS_ECSC_OK) {
        return rc;
    }

    /*
     * Canonical key ordering inside protocol:
     *
     * lifecycle
     * max_frame_bytes
     * max_int
     * max_payload_bytes
     * max_string_bytes
     * revision
     * scheduler
     */
    rc = es_literal(
        &sink,
        ",\"protocol\":{"
        "\"lifecycle\":"
        "\"founded-active-dormant-terminal.v1\","
        "\"max_frame_bytes\":262144,"
        "\"max_int\":9223372036854775807,"
        "\"max_payload_bytes\":65536,"
        "\"max_string_bytes\":1024,"
        "\"revision\":\"ecs.m1a.integration.v3\","
        "\"scheduler\":"
    );
    if (rc != ELPIS_ECSC_OK) {
        return rc;
    }

    rc = es_json_string_utf8(
        &sink,
        scheduler,
        strlen(scheduler)
    );
    if (rc != ELPIS_ECSC_OK) {
        return rc;
    }

    rc = es_literal(
        &sink,
        "},\"schema_version\":1}}"
    );
    if (rc != ELPIS_ECSC_OK) {
        return rc;
    }

    return finish_stream_digest(
        &hash,
        out_digest
    );
}

int elpis_ecsc_entity_id(
    uint64_t founding_index,
    const char *label,
    size_t label_len,
    const char genesis_digest[64],
    char out_entity_id[65]
)
{
    event_sink sink;
    elpis_sha256_ctx hash;
    int rc;

    if (out_entity_id == NULL ||
        label == NULL ||
        label_len == 0u ||
        label_len > ELPIS_ECSC_MAX_STRING_BYTES ||
        founding_index > ELPIS_ECSC_MAX_INT ||
        !is_digest(genesis_digest, 64u)) {
        return ELPIS_ECSC_INVALID;
    }

    out_entity_id[0] = '\0';

    elpis_sha256_init(&hash);

    sink.out = NULL;
    sink.cap = 0u;
    sink.pos = 0u;
    sink.hash = &hash;

    /*
     * domain_digest(
     *   "ecs.entity.founded.v1",
     *   {
     *     "schema": "ecs.entity.founding.v1",
     *     "founding_index": ...,
     *     "label": ...,
     *     "genesis_digest": ...
     *   }
     * )
     *
     * canonical JSON sorts the founding-record keys:
     *
     * founding_index
     * genesis_digest
     * label
     * schema
     */
    rc = es_literal(
        &sink,
        "{\"domain\":\"ecs.entity.founded.v1\","
        "\"payload\":{\"founding_index\":"
    );
    if (rc != ELPIS_ECSC_OK) {
        return rc;
    }

    rc = es_u64(
        &sink,
        founding_index
    );
    if (rc != ELPIS_ECSC_OK) {
        return rc;
    }

    rc = es_literal(
        &sink,
        ",\"genesis_digest\":\""
    );
    if (rc != ELPIS_ECSC_OK) {
        return rc;
    }

    rc = es_put(
        &sink,
        genesis_digest,
        64u
    );
    if (rc != ELPIS_ECSC_OK) {
        return rc;
    }

    rc = es_literal(
        &sink,
        "\",\"label\":"
    );
    if (rc != ELPIS_ECSC_OK) {
        return rc;
    }

    rc = es_json_string_utf8(
        &sink,
        label,
        label_len
    );
    if (rc != ELPIS_ECSC_OK) {
        return rc;
    }

    rc = es_literal(
        &sink,
        ",\"schema\":\"ecs.entity.founding.v1\"}}"
    );
    if (rc != ELPIS_ECSC_OK) {
        return rc;
    }

    return finish_stream_digest(
        &hash,
        out_entity_id
    );
}

int elpis_ecsc_initial_state_digest(
    const char entity_id[64],
    char out_state_digest[65]
)
{
    event_sink sink;
    elpis_sha256_ctx hash;
    int rc;

    if (out_state_digest == NULL ||
        !is_digest(entity_id, 64u)) {
        return ELPIS_ECSC_INVALID;
    }

    out_state_digest[0] = '\0';

    elpis_sha256_init(&hash);

    sink.out = NULL;
    sink.cap = 0u;
    sink.pos = 0u;
    sink.hash = &hash;

    /*
     * _state_record(entity_id, 0, {})
     *
     * canonical keys:
     * entity_id, payload, schema, version
     */
    rc = es_literal(
        &sink,
        "{\"domain\":\"ecs.entity.state.v1\","
        "\"payload\":{\"entity_id\":\""
    );
    if (rc != ELPIS_ECSC_OK) {
        return rc;
    }

    rc = es_put(
        &sink,
        entity_id,
        64u
    );
    if (rc != ELPIS_ECSC_OK) {
        return rc;
    }

    rc = es_literal(
        &sink,
        "\",\"payload\":{},"
        "\"schema\":\"ecs.entity.state.v1\","
        "\"version\":0}}"
    );
    if (rc != ELPIS_ECSC_OK) {
        return rc;
    }

    return finish_stream_digest(
        &hash,
        out_state_digest
    );
}

#define ELPIS_ECSC_RUNTIME_MAX_ENTITIES 64u
#define ELPIS_ECSC_RUNTIME_MAX_MAILBOXES 64u
#define ELPIS_ECSC_RUNTIME_MAX_WATERMARKS 64u
#define ELPIS_ECSC_RUNTIME_MAX_MAILBOX_CONTENTS 64u

static int digest_ptr_valid(const char *value)
{
    return value != NULL && is_digest(value, 64u);
}

static int ordered_digest_keys(
    const char *left,
    const char *right
)
{
    if (left == NULL || right == NULL) {
        return 0;
    }
    return memcmp(left, right, 64u) < 0;
}

static const char *lifecycle_name(uint32_t lifecycle)
{
    switch (lifecycle) {
        case ELPIS_ECSC_LIFECYCLE_FOUNDED:
            return "FOUNDED";
        case ELPIS_ECSC_LIFECYCLE_ACTIVE:
            return "ACTIVE";
        case ELPIS_ECSC_LIFECYCLE_DORMANT:
            return "DORMANT";
        case ELPIS_ECSC_LIFECYCLE_TERMINATED:
            return "TERMINATED";
        default:
            return NULL;
    }
}

static int validate_state_entity(
    const elpis_ecsc_state_entity_view *entity
)
{
    const char *lifecycle;

    if (entity == NULL ||
        !digest_ptr_valid(entity->registry_key) ||
        !digest_ptr_valid(entity->entity_id) ||
        !digest_ptr_valid(entity->founding_digest) ||
        !digest_ptr_valid(entity->state_entity_id) ||
        !digest_ptr_valid(entity->prev_state_digest) ||
        !digest_ptr_valid(entity->state_digest) ||
        entity->label == NULL ||
        entity->label_len == 0u ||
        entity->label_len > ELPIS_ECSC_MAX_STRING_BYTES ||
        entity->founding_index > ELPIS_ECSC_MAX_INT ||
        entity->state_version > ELPIS_ECSC_MAX_INT ||
        entity->delivered > ELPIS_ECSC_MAX_INT ||
        (entity->has_delivered != 0u &&
         entity->has_delivered != 1u)) {
        return ELPIS_ECSC_INVALID;
    }

    lifecycle = lifecycle_name(entity->lifecycle);
    if (lifecycle == NULL) {
        return ELPIS_ECSC_INVALID;
    }

    /*
     * Runtime-history projection invariants.
     * These are true for ECS_C EntityRecord.state_root_projection().
     */
    if (memcmp(
            entity->registry_key,
            entity->entity_id,
            64u
        ) != 0 ||
        memcmp(
            entity->entity_id,
            entity->founding_digest,
            64u
        ) != 0 ||
        memcmp(
            entity->entity_id,
            entity->state_entity_id,
            64u
        ) != 0) {
        return ELPIS_ECSC_INVALID;
    }

    return ELPIS_ECSC_OK;
}

static int validate_state_envelope(
    const elpis_ecsc_state_envelope_view *env
)
{
    char expected_mid[65];
    char expected_payload_digest[65];
    int rc;

    if (env == NULL ||
        !digest_ptr_valid(env->message_id) ||
        !digest_ptr_valid(env->payload_digest) ||
        !digest_ptr_valid(env->receiver_entity_id) ||
        !digest_ptr_valid(env->sender_entity_id) ||
        env->logical_clock == 0u ||
        env->logical_clock > ELPIS_ECSC_MAX_INT ||
        env->sequence == 0u ||
        env->sequence > ELPIS_ECSC_MAX_INT ||
        env->payload == NULL ||
        env->payload_size == 0u ||
        env->payload_size > ELPIS_ECSC_MAX_PAYLOAD_BYTES) {
        return ELPIS_ECSC_INVALID;
    }

    rc = elpis_ecsc_message_id(
        env->sender_entity_id,
        env->receiver_entity_id,
        env->sequence,
        env->payload,
        env->payload_size,
        expected_mid,
        expected_payload_digest
    );

    if (rc != ELPIS_ECSC_OK) {
        return rc;
    }

    if (memcmp(
            expected_mid,
            env->message_id,
            64u
        ) != 0 ||
        memcmp(
            expected_payload_digest,
            env->payload_digest,
            64u
        ) != 0) {
        return ELPIS_ECSC_INVALID;
    }

    return ELPIS_ECSC_OK;
}

static int validate_state_root_view(
    const elpis_ecsc_state_root_view *view
)
{
    size_t i;
    size_t j;

    if (view == NULL ||
        !digest_ptr_valid(view->genesis_digest) ||
        !digest_ptr_valid(view->history_digest) ||
        view->logical_clock > ELPIS_ECSC_MAX_INT ||
        view->mailbox_capacity == 0u ||
        view->mailbox_capacity > ELPIS_ECSC_MAX_INT ||
        view->mailbox_default_capacity == 0u ||
        view->mailbox_default_capacity > ELPIS_ECSC_MAX_INT ||
        view->next_founding_index > ELPIS_ECSC_MAX_INT ||
        view->entity_count > ELPIS_ECSC_RUNTIME_MAX_ENTITIES ||
        view->mailbox_count > ELPIS_ECSC_RUNTIME_MAX_MAILBOXES ||
        view->watermark_count > ELPIS_ECSC_RUNTIME_MAX_WATERMARKS ||
        (view->entities == NULL &&
         view->entity_count != 0u) ||
        (view->mailboxes == NULL &&
         view->mailbox_count != 0u) ||
        (view->watermarks == NULL &&
         view->watermark_count != 0u)) {
        return ELPIS_ECSC_INVALID;
    }

    for (i = 0u; i < view->entity_count; ++i) {
        if (validate_state_entity(
                &view->entities[i]
            ) != ELPIS_ECSC_OK) {
            return ELPIS_ECSC_INVALID;
        }

        if (i != 0u &&
            !ordered_digest_keys(
                view->entities[i - 1u].registry_key,
                view->entities[i].registry_key
            )) {
            return ELPIS_ECSC_INVALID;
        }
    }

    for (i = 0u; i < view->mailbox_count; ++i) {
        const elpis_ecsc_state_mailbox_view *box =
            &view->mailboxes[i];

        if (!digest_ptr_valid(box->mailbox_key) ||
            !digest_ptr_valid(box->receiver_entity_id) ||
            memcmp(
                box->mailbox_key,
                box->receiver_entity_id,
                64u
            ) != 0 ||
            box->capacity == 0u ||
            box->capacity > ELPIS_ECSC_MAX_INT ||
            box->content_count >
                ELPIS_ECSC_RUNTIME_MAX_MAILBOX_CONTENTS ||
            box->content_count > box->capacity ||
            (box->contents == NULL &&
             box->content_count != 0u)) {
            return ELPIS_ECSC_INVALID;
        }

        if (i != 0u &&
            !ordered_digest_keys(
                view->mailboxes[i - 1u].mailbox_key,
                box->mailbox_key
            )) {
            return ELPIS_ECSC_INVALID;
        }

        for (j = 0u; j < box->content_count; ++j) {
            if (validate_state_envelope(
                    &box->contents[j]
                ) != ELPIS_ECSC_OK ||
                memcmp(
                    box->contents[j].receiver_entity_id,
                    box->receiver_entity_id,
                    64u
                ) != 0) {
                return ELPIS_ECSC_INVALID;
            }
        }
    }

    for (i = 0u; i < view->watermark_count; ++i) {
        const elpis_ecsc_state_watermark_view *wm =
            &view->watermarks[i];

        if (!digest_ptr_valid(wm->sender_entity_id) ||
            wm->sequence == 0u ||
            wm->sequence > ELPIS_ECSC_MAX_INT) {
            return ELPIS_ECSC_INVALID;
        }

        if (i != 0u &&
            !ordered_digest_keys(
                view->watermarks[i - 1u].sender_entity_id,
                wm->sender_entity_id
            )) {
            return ELPIS_ECSC_INVALID;
        }
    }

    return ELPIS_ECSC_OK;
}

static int encode_state_envelope(
    event_sink *sink,
    const elpis_ecsc_state_envelope_view *env
)
{
    int rc;

    rc = es_literal(
        sink,
        "{\"logical_clock\":"
    );
    if (rc != ELPIS_ECSC_OK) return rc;

    rc = es_u64(
        sink,
        env->logical_clock
    );
    if (rc != ELPIS_ECSC_OK) return rc;

    rc = es_literal(
        sink,
        ",\"message_id\":\""
    );
    if (rc != ELPIS_ECSC_OK) return rc;

    rc = es_put(
        sink,
        env->message_id,
        64u
    );
    if (rc != ELPIS_ECSC_OK) return rc;

    rc = es_literal(
        sink,
        "\",\"payload_digest\":\""
    );
    if (rc != ELPIS_ECSC_OK) return rc;

    rc = es_put(
        sink,
        env->payload_digest,
        64u
    );
    if (rc != ELPIS_ECSC_OK) return rc;

    rc = es_literal(
        sink,
        "\",\"payload_hex\":\""
    );
    if (rc != ELPIS_ECSC_OK) return rc;

    rc = es_hex_payload(
        sink,
        env->payload,
        env->payload_size
    );
    if (rc != ELPIS_ECSC_OK) return rc;

    rc = es_literal(
        sink,
        "\",\"receiver_entity_id\":\""
    );
    if (rc != ELPIS_ECSC_OK) return rc;

    rc = es_put(
        sink,
        env->receiver_entity_id,
        64u
    );
    if (rc != ELPIS_ECSC_OK) return rc;

    rc = es_literal(
        sink,
        "\",\"schema\":\"ecs.message.v1\","
        "\"sender_entity_id\":\""
    );
    if (rc != ELPIS_ECSC_OK) return rc;

    rc = es_put(
        sink,
        env->sender_entity_id,
        64u
    );
    if (rc != ELPIS_ECSC_OK) return rc;

    rc = es_literal(
        sink,
        "\",\"sequence\":"
    );
    if (rc != ELPIS_ECSC_OK) return rc;

    rc = es_u64(
        sink,
        env->sequence
    );
    if (rc != ELPIS_ECSC_OK) return rc;

    return es_literal(
        sink,
        "}"
    );
}

static int encode_state_entity(
    event_sink *sink,
    const elpis_ecsc_state_entity_view *entity
)
{
    const char *lifecycle =
        lifecycle_name(entity->lifecycle);

    int rc;

    rc = es_literal(
        sink,
        "{\"entity_id\":\""
    );
    if (rc != ELPIS_ECSC_OK) return rc;

    rc = es_put(
        sink,
        entity->entity_id,
        64u
    );
    if (rc != ELPIS_ECSC_OK) return rc;

    rc = es_literal(
        sink,
        "\",\"founding_digest\":\""
    );
    if (rc != ELPIS_ECSC_OK) return rc;

    rc = es_put(
        sink,
        entity->founding_digest,
        64u
    );
    if (rc != ELPIS_ECSC_OK) return rc;

    rc = es_literal(
        sink,
        "\",\"founding_index\":"
    );
    if (rc != ELPIS_ECSC_OK) return rc;

    rc = es_u64(
        sink,
        entity->founding_index
    );
    if (rc != ELPIS_ECSC_OK) return rc;

    rc = es_literal(
        sink,
        ",\"label\":"
    );
    if (rc != ELPIS_ECSC_OK) return rc;

    rc = es_json_string_utf8(
        sink,
        entity->label,
        entity->label_len
    );
    if (rc != ELPIS_ECSC_OK) return rc;

    rc = es_literal(
        sink,
        ",\"lifecycle\":\""
    );
    if (rc != ELPIS_ECSC_OK) return rc;

    rc = es_put(
        sink,
        lifecycle,
        strlen(lifecycle)
    );
    if (rc != ELPIS_ECSC_OK) return rc;

    rc = es_literal(
        sink,
        "\",\"payload\":"
    );
    if (rc != ELPIS_ECSC_OK) return rc;

    if (entity->has_delivered != 0u) {
        rc = es_literal(
            sink,
            "{\"delivered\":"
        );
        if (rc != ELPIS_ECSC_OK) return rc;

        rc = es_u64(
            sink,
            entity->delivered
        );
        if (rc != ELPIS_ECSC_OK) return rc;

        rc = es_literal(
            sink,
            "}"
        );
    } else {
        rc = es_literal(
            sink,
            "{}"
        );
    }

    if (rc != ELPIS_ECSC_OK) return rc;

    rc = es_literal(
        sink,
        ",\"prev_state_digest\":\""
    );
    if (rc != ELPIS_ECSC_OK) return rc;

    rc = es_put(
        sink,
        entity->prev_state_digest,
        64u
    );
    if (rc != ELPIS_ECSC_OK) return rc;

    rc = es_literal(
        sink,
        "\",\"registry_key\":\""
    );
    if (rc != ELPIS_ECSC_OK) return rc;

    rc = es_put(
        sink,
        entity->registry_key,
        64u
    );
    if (rc != ELPIS_ECSC_OK) return rc;

    rc = es_literal(
        sink,
        "\",\"state_digest\":\""
    );
    if (rc != ELPIS_ECSC_OK) return rc;

    rc = es_put(
        sink,
        entity->state_digest,
        64u
    );
    if (rc != ELPIS_ECSC_OK) return rc;

    rc = es_literal(
        sink,
        "\",\"state_entity_id\":\""
    );
    if (rc != ELPIS_ECSC_OK) return rc;

    rc = es_put(
        sink,
        entity->state_entity_id,
        64u
    );
    if (rc != ELPIS_ECSC_OK) return rc;

    rc = es_literal(
        sink,
        "\",\"state_version\":"
    );
    if (rc != ELPIS_ECSC_OK) return rc;

    rc = es_u64(
        sink,
        entity->state_version
    );
    if (rc != ELPIS_ECSC_OK) return rc;

    return es_literal(
        sink,
        "}"
    );
}

static int encode_state_root_payload(
    event_sink *sink,
    const elpis_ecsc_state_root_view *view
)
{
    size_t i;
    size_t j;
    int rc;

    /*
     * canonical key order for ecs.state_root.v3:
     *
     * entities
     * genesis_digest
     * history_digest
     * logical_clock
     * mailbox_capacity
     * mailbox_default_capacity
     * mailboxes
     * next_founding_index
     * scheduler_state
     * schema
     * watermarks
     */
    rc = es_literal(
        sink,
        "{\"entities\":["
    );
    if (rc != ELPIS_ECSC_OK) return rc;

    for (i = 0u; i < view->entity_count; ++i) {
        if (i != 0u) {
            rc = es_literal(
                sink,
                ","
            );
            if (rc != ELPIS_ECSC_OK) return rc;
        }

        rc = encode_state_entity(
            sink,
            &view->entities[i]
        );
        if (rc != ELPIS_ECSC_OK) return rc;
    }

    rc = es_literal(
        sink,
        "],\"genesis_digest\":\""
    );
    if (rc != ELPIS_ECSC_OK) return rc;

    rc = es_put(
        sink,
        view->genesis_digest,
        64u
    );
    if (rc != ELPIS_ECSC_OK) return rc;

    rc = es_literal(
        sink,
        "\",\"history_digest\":\""
    );
    if (rc != ELPIS_ECSC_OK) return rc;

    rc = es_put(
        sink,
        view->history_digest,
        64u
    );
    if (rc != ELPIS_ECSC_OK) return rc;

    rc = es_literal(
        sink,
        "\",\"logical_clock\":"
    );
    if (rc != ELPIS_ECSC_OK) return rc;

    rc = es_u64(
        sink,
        view->logical_clock
    );
    if (rc != ELPIS_ECSC_OK) return rc;

    rc = es_literal(
        sink,
        ",\"mailbox_capacity\":"
    );
    if (rc != ELPIS_ECSC_OK) return rc;

    rc = es_u64(
        sink,
        view->mailbox_capacity
    );
    if (rc != ELPIS_ECSC_OK) return rc;

    rc = es_literal(
        sink,
        ",\"mailbox_default_capacity\":"
    );
    if (rc != ELPIS_ECSC_OK) return rc;

    rc = es_u64(
        sink,
        view->mailbox_default_capacity
    );
    if (rc != ELPIS_ECSC_OK) return rc;

    rc = es_literal(
        sink,
        ",\"mailboxes\":["
    );
    if (rc != ELPIS_ECSC_OK) return rc;

    for (i = 0u; i < view->mailbox_count; ++i) {
        const elpis_ecsc_state_mailbox_view *box =
            &view->mailboxes[i];

        if (i != 0u) {
            rc = es_literal(
                sink,
                ","
            );
            if (rc != ELPIS_ECSC_OK) return rc;
        }

        rc = es_literal(
            sink,
            "{\"capacity\":"
        );
        if (rc != ELPIS_ECSC_OK) return rc;

        rc = es_u64(
            sink,
            box->capacity
        );
        if (rc != ELPIS_ECSC_OK) return rc;

        rc = es_literal(
            sink,
            ",\"contents\":["
        );
        if (rc != ELPIS_ECSC_OK) return rc;

        for (j = 0u; j < box->content_count; ++j) {
            if (j != 0u) {
                rc = es_literal(
                    sink,
                    ","
                );
                if (rc != ELPIS_ECSC_OK) return rc;
            }

            rc = encode_state_envelope(
                sink,
                &box->contents[j]
            );
            if (rc != ELPIS_ECSC_OK) return rc;
        }

        rc = es_literal(
            sink,
            "],\"mailbox_key\":\""
        );
        if (rc != ELPIS_ECSC_OK) return rc;

        rc = es_put(
            sink,
            box->mailbox_key,
            64u
        );
        if (rc != ELPIS_ECSC_OK) return rc;

        rc = es_literal(
            sink,
            "\",\"receiver_entity_id\":\""
        );
        if (rc != ELPIS_ECSC_OK) return rc;

        rc = es_put(
            sink,
            box->receiver_entity_id,
            64u
        );
        if (rc != ELPIS_ECSC_OK) return rc;

        rc = es_literal(
            sink,
            "\"}"
        );
        if (rc != ELPIS_ECSC_OK) return rc;
    }

    rc = es_literal(
        sink,
        "],\"next_founding_index\":"
    );
    if (rc != ELPIS_ECSC_OK) return rc;

    rc = es_u64(
        sink,
        view->next_founding_index
    );
    if (rc != ELPIS_ECSC_OK) return rc;

    rc = es_literal(
        sink,
        ",\"scheduler_state\":{},"
        "\"schema\":\"ecs.state_root.v3\","
        "\"watermarks\":{"
    );
    if (rc != ELPIS_ECSC_OK) return rc;

    for (i = 0u; i < view->watermark_count; ++i) {
        if (i != 0u) {
            rc = es_literal(
                sink,
                ","
            );
            if (rc != ELPIS_ECSC_OK) return rc;
        }

        rc = es_literal(
            sink,
            "\""
        );
        if (rc != ELPIS_ECSC_OK) return rc;

        rc = es_put(
            sink,
            view->watermarks[i].sender_entity_id,
            64u
        );
        if (rc != ELPIS_ECSC_OK) return rc;

        rc = es_literal(
            sink,
            "\":"
        );
        if (rc != ELPIS_ECSC_OK) return rc;

        rc = es_u64(
            sink,
            view->watermarks[i].sequence
        );
        if (rc != ELPIS_ECSC_OK) return rc;
    }

    return es_literal(
        sink,
        "}}"
    );
}

int elpis_ecsc_state_root_digest(
    const elpis_ecsc_state_root_view *view,
    char out_digest[65]
)
{
    event_sink sink;
    elpis_sha256_ctx hash;
    int rc;

    if (out_digest == NULL) {
        return ELPIS_ECSC_INVALID;
    }

    out_digest[0] = '\0';

    rc = validate_state_root_view(view);
    if (rc != ELPIS_ECSC_OK) {
        return rc;
    }

    elpis_sha256_init(&hash);

    sink.out = NULL;
    sink.cap = 0u;
    sink.pos = 0u;
    sink.hash = &hash;

    rc = es_literal(
        &sink,
        "{\"domain\":\"ecs.state_root.v1\","
        "\"payload\":"
    );
    if (rc != ELPIS_ECSC_OK) {
        return rc;
    }

    rc = encode_state_root_payload(
        &sink,
        view
    );
    if (rc != ELPIS_ECSC_OK) {
        return rc;
    }

    rc = es_literal(
        &sink,
        "}"
    );
    if (rc != ELPIS_ECSC_OK) {
        return rc;
    }

    return finish_stream_digest(
        &hash,
        out_digest
    );
}

int elpis_ecsc_delivered_state_digest(
    const char entity_id[64],
    uint64_t version,
    uint64_t delivered,
    char out_state_digest[65]
)
{
    event_sink sink;
    elpis_sha256_ctx hash;
    int rc;

    if (out_state_digest == NULL ||
        !is_digest(entity_id, 64u) ||
        version == 0u ||
        version > ELPIS_ECSC_MAX_INT ||
        delivered == 0u ||
        delivered > ELPIS_ECSC_MAX_INT) {
        return ELPIS_ECSC_INVALID;
    }

    out_state_digest[0] = '\0';

    elpis_sha256_init(&hash);

    sink.out = NULL;
    sink.cap = 0u;
    sink.pos = 0u;
    sink.hash = &hash;

    /*
     * domain_digest(
     *   "ecs.entity.state.v1",
     *   {
     *     "entity_id": ...,
     *     "payload": {"delivered": N},
     *     "schema": "ecs.entity.state.v1",
     *     "version": V
     *   }
     * )
     */
    rc = es_literal(
        &sink,
        "{\"domain\":\"ecs.entity.state.v1\","
        "\"payload\":{\"entity_id\":\""
    );
    if (rc != ELPIS_ECSC_OK) return rc;

    rc = es_put(
        &sink,
        entity_id,
        64u
    );
    if (rc != ELPIS_ECSC_OK) return rc;

    rc = es_literal(
        &sink,
        "\",\"payload\":{\"delivered\":"
    );
    if (rc != ELPIS_ECSC_OK) return rc;

    rc = es_u64(
        &sink,
        delivered
    );
    if (rc != ELPIS_ECSC_OK) return rc;

    rc = es_literal(
        &sink,
        "},\"schema\":\"ecs.entity.state.v1\","
        "\"version\":"
    );
    if (rc != ELPIS_ECSC_OK) return rc;

    rc = es_u64(
        &sink,
        version
    );
    if (rc != ELPIS_ECSC_OK) return rc;

    rc = es_literal(
        &sink,
        "}}"
    );
    if (rc != ELPIS_ECSC_OK) return rc;

    return finish_stream_digest(
        &hash,
        out_state_digest
    );
}

static int runtime_find_entity(
    const elpis_ecsc_state_root_view *root,
    const char entity_id[64],
    size_t *out_index
)
{
    size_t i;

    if (root == NULL ||
        entity_id == NULL ||
        out_index == NULL) {
        return ELPIS_ECSC_INVALID;
    }

    for (i = 0u; i < root->entity_count; ++i) {
        if (memcmp(
                root->entities[i].entity_id,
                entity_id,
                64u
            ) == 0) {
            *out_index = i;
            return ELPIS_ECSC_OK;
        }
    }

    return ELPIS_ECSC_INVALID;
}

static int runtime_copy_entities(
    const elpis_ecsc_state_root_view *root,
    elpis_ecsc_state_entity_view *out
)
{
    if (root == NULL ||
        (root->entity_count != 0u &&
         out == NULL)) {
        return ELPIS_ECSC_INVALID;
    }

    if (root->entity_count != 0u) {
        memcpy(
            out,
            root->entities,
            root->entity_count * sizeof *out
        );
    }

    return ELPIS_ECSC_OK;
}

static int runtime_prepare_mailboxes(
    const elpis_ecsc_state_root_view *current,
    const char history_entity_id[64],
    const elpis_ecsc_state_envelope_view *envelope,
    int queued,
    elpis_ecsc_state_mailbox_view out[1],
    size_t *out_count
)
{
    uint64_t capacity;

    if (current == NULL ||
        history_entity_id == NULL ||
        out == NULL ||
        out_count == NULL) {
        return ELPIS_ECSC_INVALID;
    }

    /*
     * Runtime ReceiptHistory owns exactly one possible mailbox: history.
     * Before the first record it does not yet exist. Once created it remains
     * materialized and empty between records.
     */
    if (current->mailbox_count > 1u) {
        return ELPIS_ECSC_INVALID;
    }

    if (current->mailbox_count == 1u) {
        const elpis_ecsc_state_mailbox_view *existing =
            &current->mailboxes[0];

        if (memcmp(
                existing->mailbox_key,
                history_entity_id,
                64u
            ) != 0 ||
            memcmp(
                existing->receiver_entity_id,
                history_entity_id,
                64u
            ) != 0 ||
            existing->content_count != 0u) {
            return ELPIS_ECSC_INVALID;
        }

        capacity = existing->capacity;
    } else {
        capacity = current->mailbox_capacity;
    }

    out[0].mailbox_key = history_entity_id;
    out[0].receiver_entity_id = history_entity_id;
    out[0].capacity = capacity;

    if (queued) {
        if (envelope == NULL) {
            return ELPIS_ECSC_INVALID;
        }

        out[0].contents = envelope;
        out[0].content_count = 1u;
    } else {
        out[0].contents = NULL;
        out[0].content_count = 0u;
    }

    *out_count = 1u;
    return ELPIS_ECSC_OK;
}

static int runtime_prepare_watermarks(
    const elpis_ecsc_state_root_view *current,
    const char sender_entity_id[64],
    elpis_ecsc_state_watermark_view out[
        ELPIS_ECSC_RUNTIME_MAX_WATERMARKS
    ],
    size_t *out_count,
    uint64_t *out_sequence
)
{
    size_t i;
    size_t insert_at = 0u;
    int found = 0;
    uint64_t previous = 0u;

    if (current == NULL ||
        sender_entity_id == NULL ||
        out == NULL ||
        out_count == NULL ||
        out_sequence == NULL ||
        current->watermark_count >=
            ELPIS_ECSC_RUNTIME_MAX_WATERMARKS) {
        return ELPIS_ECSC_INVALID;
    }

    for (i = 0u; i < current->watermark_count; ++i) {
        int cmp = memcmp(
            current->watermarks[i].sender_entity_id,
            sender_entity_id,
            64u
        );

        if (cmp < 0) {
            insert_at = i + 1u;
        } else if (cmp == 0) {
            found = 1;
            insert_at = i;
            previous =
                current->watermarks[i].sequence;
            break;
        } else {
            break;
        }
    }

    if (previous >= ELPIS_ECSC_MAX_INT) {
        return ELPIS_ECSC_INVALID;
    }

    *out_sequence = previous + 1u;

    if (found) {
        memcpy(
            out,
            current->watermarks,
            current->watermark_count *
                sizeof *out
        );

        out[insert_at].sender_entity_id =
            sender_entity_id;

        out[insert_at].sequence =
            *out_sequence;

        *out_count =
            current->watermark_count;

        return ELPIS_ECSC_OK;
    }

    if (insert_at != 0u) {
        memcpy(
            out,
            current->watermarks,
            insert_at * sizeof *out
        );
    }

    out[insert_at].sender_entity_id =
        sender_entity_id;

    out[insert_at].sequence =
        *out_sequence;

    if (insert_at < current->watermark_count) {
        memcpy(
            out + insert_at + 1u,
            current->watermarks + insert_at,
            (current->watermark_count - insert_at) *
                sizeof *out
        );
    }

    *out_count =
        current->watermark_count + 1u;

    return ELPIS_ECSC_OK;
}

static const char ELPIS_ECSC_ZERO_DIGEST[65] =
    "00000000000000000000000000000000"
    "00000000000000000000000000000000";

int elpis_ecsc_runtime_record_plan(
    const elpis_ecsc_state_root_view *current,
    const char current_event_digest[64],
    const char sender_entity_id[64],
    const char history_entity_id[64],
    const void *receipt_payload,
    size_t receipt_payload_size,

    uint8_t *enqueue_event,
    size_t enqueue_capacity,

    uint8_t *processed_event,
    size_t processed_capacity,

    elpis_ecsc_runtime_record_plan_result *out
)
{
    elpis_ecsc_state_entity_view entities[
        ELPIS_ECSC_RUNTIME_MAX_ENTITIES
    ];

    elpis_ecsc_state_mailbox_view mailboxes[1];

    elpis_ecsc_state_watermark_view watermarks[
        ELPIS_ECSC_RUNTIME_MAX_WATERMARKS
    ];

    elpis_ecsc_state_envelope_view envelope;

    elpis_ecsc_state_root_view enqueue_root;
    elpis_ecsc_state_root_view processed_root;

    size_t sender_index;
    size_t history_index;
    size_t mailbox_count = 0u;
    size_t watermark_count = 0u;

    uint64_t sequence;
    uint64_t delivered_before;
    uint64_t delivered_after;
    uint64_t next_version;

    char before_root[65];
    char payload_digest[65];
    char message_id[65];
    char writer_message_id[65];

    char enqueue_intent[65];
    char enqueue_digest[65];
    char enqueue_after_root[65];

    char processed_intent[65];
    char processed_digest[65];
    char processed_after_root[65];

    char history_prev_digest[65];
    char history_next_digest[65];

    size_t enqueue_written = 0u;
    size_t processed_written = 0u;

    int rc;

    if (out == NULL) {
        return ELPIS_ECSC_INVALID;
    }

    memset(out, 0, sizeof *out);

    if (current == NULL ||
        !is_digest(current_event_digest, 64u) ||
        !is_digest(sender_entity_id, 64u) ||
        !is_digest(history_entity_id, 64u) ||
        receipt_payload == NULL ||
        receipt_payload_size == 0u ||
        receipt_payload_size >
            ELPIS_ECSC_MAX_PAYLOAD_BYTES ||
        enqueue_event == NULL ||
        processed_event == NULL ||
        current->entity_count == 0u ||
        current->entity_count >
            ELPIS_ECSC_RUNTIME_MAX_ENTITIES ||
        current->logical_clock >
            ELPIS_ECSC_MAX_INT - 2u) {
        return ELPIS_ECSC_INVALID;
    }

    rc = validate_state_root_view(current);
    if (rc != ELPIS_ECSC_OK) {
        return rc;
    }

    rc = runtime_find_entity(
        current,
        sender_entity_id,
        &sender_index
    );
    if (rc != ELPIS_ECSC_OK) {
        return rc;
    }

    rc = runtime_find_entity(
        current,
        history_entity_id,
        &history_index
    );
    if (rc != ELPIS_ECSC_OK) {
        return rc;
    }

    if (sender_index == history_index ||
        current->entities[sender_index].lifecycle !=
            ELPIS_ECSC_LIFECYCLE_ACTIVE ||
        current->entities[history_index].lifecycle !=
            ELPIS_ECSC_LIFECYCLE_ACTIVE) {
        return ELPIS_ECSC_INVALID;
    }

    rc = elpis_ecsc_state_root_digest(
        current,
        before_root
    );
    if (rc != ELPIS_ECSC_OK) {
        return rc;
    }

    rc = runtime_prepare_watermarks(
        current,
        sender_entity_id,
        watermarks,
        &watermark_count,
        &sequence
    );
    if (rc != ELPIS_ECSC_OK) {
        return rc;
    }

    rc = elpis_ecsc_message_id(
        sender_entity_id,
        history_entity_id,
        sequence,
        receipt_payload,
        receipt_payload_size,
        message_id,
        payload_digest
    );
    if (rc != ELPIS_ECSC_OK) {
        return rc;
    }

    memset(&envelope, 0, sizeof envelope);

    envelope.logical_clock =
        current->logical_clock + 1u;

    envelope.message_id = message_id;
    envelope.payload_digest = payload_digest;
    envelope.payload = receipt_payload;
    envelope.payload_size = receipt_payload_size;
    envelope.receiver_entity_id =
        history_entity_id;
    envelope.sender_entity_id =
        sender_entity_id;
    envelope.sequence = sequence;

    rc = runtime_prepare_mailboxes(
        current,
        history_entity_id,
        &envelope,
        1,
        mailboxes,
        &mailbox_count
    );
    if (rc != ELPIS_ECSC_OK) {
        return rc;
    }

    /*
     * First obtain the enqueue intent digest. after_state_root is excluded
     * from event-intent identity, so the zero sentinel is sufficient here.
     */
    rc = elpis_ecsc_enqueue_event_write(
        sender_entity_id,
        history_entity_id,
        sequence,
        receipt_payload,
        receipt_payload_size,
        current->logical_clock,
        before_root,
        ELPIS_ECSC_ZERO_DIGEST,
        current_event_digest,
        enqueue_event,
        enqueue_capacity,
        &enqueue_written,
        enqueue_digest,
        enqueue_intent,
        writer_message_id
    );
    if (rc != ELPIS_ECSC_OK) {
        return rc;
    }

    if (memcmp(
            writer_message_id,
            message_id,
            64u
        ) != 0) {
        return ELPIS_ECSC_INVALID;
    }

    enqueue_root = *current;
    enqueue_root.logical_clock =
        current->logical_clock + 1u;
    enqueue_root.history_digest =
        enqueue_intent;
    enqueue_root.mailboxes =
        mailboxes;
    enqueue_root.mailbox_count =
        mailbox_count;
    enqueue_root.watermarks =
        watermarks;
    enqueue_root.watermark_count =
        watermark_count;

    rc = elpis_ecsc_state_root_digest(
        &enqueue_root,
        enqueue_after_root
    );
    if (rc != ELPIS_ECSC_OK) {
        return rc;
    }

    /*
     * Rebuild the final canonical enqueue event now that its after-root is
     * known.
     */
    rc = elpis_ecsc_enqueue_event_write(
        sender_entity_id,
        history_entity_id,
        sequence,
        receipt_payload,
        receipt_payload_size,
        current->logical_clock,
        before_root,
        enqueue_after_root,
        current_event_digest,
        enqueue_event,
        enqueue_capacity,
        &enqueue_written,
        enqueue_digest,
        enqueue_intent,
        writer_message_id
    );
    if (rc != ELPIS_ECSC_OK) {
        return rc;
    }

    if (memcmp(
            writer_message_id,
            message_id,
            64u
        ) != 0) {
        return ELPIS_ECSC_INVALID;
    }

    /*
     * Prepare the processed projection:
     * - history mailbox remains materialized but becomes empty;
     * - watermark remains advanced;
     * - receiver state version/delivery counter advances once.
     */
    rc = runtime_copy_entities(
        current,
        entities
    );
    if (rc != ELPIS_ECSC_OK) {
        return rc;
    }

    if (entities[history_index].has_delivered != 0u) {
        delivered_before =
            entities[history_index].delivered;
    } else {
        delivered_before = 0u;
    }

    if (delivered_before >= ELPIS_ECSC_MAX_INT ||
        entities[history_index].state_version >=
            ELPIS_ECSC_MAX_INT) {
        return ELPIS_ECSC_INVALID;
    }

    delivered_after =
        delivered_before + 1u;

    next_version =
        entities[history_index].state_version + 1u;

    memcpy(
        history_prev_digest,
        entities[history_index].state_digest,
        64u
    );
    history_prev_digest[64] = '\0';

    rc = elpis_ecsc_delivered_state_digest(
        history_entity_id,
        next_version,
        delivered_after,
        history_next_digest
    );
    if (rc != ELPIS_ECSC_OK) {
        return rc;
    }

    entities[history_index].prev_state_digest =
        history_prev_digest;

    entities[history_index].state_version =
        next_version;

    entities[history_index].state_digest =
        history_next_digest;

    entities[history_index].has_delivered = 1u;
    entities[history_index].delivered =
        delivered_after;

    rc = runtime_prepare_mailboxes(
        current,
        history_entity_id,
        NULL,
        0,
        mailboxes,
        &mailbox_count
    );
    if (rc != ELPIS_ECSC_OK) {
        return rc;
    }

    /*
     * Obtain processed intent first, again with zero after-root.
     */
    rc = elpis_ecsc_processed_event_write(
        history_entity_id,
        message_id,
        current->logical_clock + 1u,
        enqueue_after_root,
        ELPIS_ECSC_ZERO_DIGEST,
        enqueue_digest,
        processed_event,
        processed_capacity,
        &processed_written,
        processed_digest,
        processed_intent
    );
    if (rc != ELPIS_ECSC_OK) {
        return rc;
    }

    processed_root = *current;
    processed_root.logical_clock =
        current->logical_clock + 2u;
    processed_root.history_digest =
        processed_intent;
    processed_root.entities =
        entities;
    processed_root.mailboxes =
        mailboxes;
    processed_root.mailbox_count =
        mailbox_count;
    processed_root.watermarks =
        watermarks;
    processed_root.watermark_count =
        watermark_count;

    rc = elpis_ecsc_state_root_digest(
        &processed_root,
        processed_after_root
    );
    if (rc != ELPIS_ECSC_OK) {
        return rc;
    }

    rc = elpis_ecsc_processed_event_write(
        history_entity_id,
        message_id,
        current->logical_clock + 1u,
        enqueue_after_root,
        processed_after_root,
        enqueue_digest,
        processed_event,
        processed_capacity,
        &processed_written,
        processed_digest,
        processed_intent
    );
    if (rc != ELPIS_ECSC_OK) {
        return rc;
    }

    out->sequence = sequence;

    out->enqueue_event_index =
        current->logical_clock;

    out->processed_event_index =
        current->logical_clock + 1u;

    out->final_logical_clock =
        current->logical_clock + 2u;

    out->final_history_state_version =
        next_version;

    out->final_delivered =
        delivered_after;

    out->enqueue_event_size =
        enqueue_written;

    out->processed_event_size =
        processed_written;

    memcpy(
        out->message_id,
        message_id,
        65u
    );

    memcpy(
        out->enqueue_state_root,
        enqueue_after_root,
        65u
    );

    memcpy(
        out->final_state_root,
        processed_after_root,
        65u
    );

    memcpy(
        out->enqueue_event_digest,
        enqueue_digest,
        65u
    );

    memcpy(
        out->processed_event_digest,
        processed_digest,
        65u
    );

    memcpy(
        out->final_history_state_digest,
        history_next_digest,
        65u
    );

    return ELPIS_ECSC_OK;
}
