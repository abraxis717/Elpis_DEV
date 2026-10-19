/* The native host's JSON (host_json.h). Strict RFC 8259 input within fixed bounds; nothing is allocated. */
#include "host_json.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/* ---- parser -------------------------------------------------------------------------------------------------- */

typedef struct {
    const char *p, *end;
    hj_arena *a;
    char *error;
    size_t error_len;
    int depth;
} parser;

static int fail(parser *s, const char *what) {
    if (s->error && s->error_len) snprintf(s->error, s->error_len, "%s", what);
    return -1;
}

static void skip(parser *s) {
    while (s->p < s->end && (*s->p == ' ' || *s->p == '\t' || *s->p == '\n' || *s->p == '\r')) ++s->p;
}

static char *new_bytes(parser *s, size_t n, size_t align) {
    size_t at = (s->a->used_bytes + align - 1) & ~(align - 1);
    if (at > s->a->cap_bytes || n > s->a->cap_bytes - at) return NULL;
    s->a->used_bytes = at + n;
    return s->a->bytes + at;
}

static int put_utf8(char *out, size_t *n, size_t cap, uint32_t c) {
    if (c < 0x80) {
        if (*n + 1 > cap) return -1;
        out[(*n)++] = (char)c;
    } else if (c < 0x800) {
        if (*n + 2 > cap) return -1;
        out[(*n)++] = (char)(0xC0 | (c >> 6));
        out[(*n)++] = (char)(0x80 | (c & 0x3F));
    } else if (c < 0x10000) {
        if (*n + 3 > cap) return -1;
        out[(*n)++] = (char)(0xE0 | (c >> 12));
        out[(*n)++] = (char)(0x80 | ((c >> 6) & 0x3F));
        out[(*n)++] = (char)(0x80 | (c & 0x3F));
    } else {
        if (*n + 4 > cap) return -1;
        out[(*n)++] = (char)(0xF0 | (c >> 18));
        out[(*n)++] = (char)(0x80 | ((c >> 12) & 0x3F));
        out[(*n)++] = (char)(0x80 | ((c >> 6) & 0x3F));
        out[(*n)++] = (char)(0x80 | (c & 0x3F));
    }
    return 0;
}

static int hex4(const char *p, uint32_t *out) {
    uint32_t v = 0;
    for (int i = 0; i < 4; ++i) {
        const char c = p[i];
        v <<= 4;
        if (c >= '0' && c <= '9') v |= (uint32_t)(c - '0');
        else if (c >= 'a' && c <= 'f') v |= (uint32_t)(c - 'a' + 10);
        else if (c >= 'A' && c <= 'F') v |= (uint32_t)(c - 'A' + 10);
        else return -1;
    }
    *out = v;
    return 0;
}

/* A string token at s->p (the opening quote): unescaped into the arena. */
static int parse_string(parser *s, const char **out, size_t *len) {
    ++s->p;
    const char *start = s->p;
    while (s->p < s->end && *s->p != '"') {
        if ((unsigned char)*s->p < 0x20) return fail(s, "control character in a string");
        if (*s->p == '\\') {
            if (s->p + 1 >= s->end) return fail(s, "unterminated escape");
            s->p += 2;
        } else {
            ++s->p;
        }
    }
    if (s->p >= s->end) return fail(s, "unterminated string");
    const char *stop = s->p++;
    const size_t cap = (size_t)(stop - start) + 1;
    char *buf = new_bytes(s, cap, 1);
    if (!buf) return fail(s, "input too large (string bytes)");
    size_t n = 0;
    for (const char *q = start; q < stop;) {
        if (*q != '\\') {
            buf[n++] = *q++;
            continue;
        }
        const char e = q[1];
        q += 2;
        uint32_t c;
        switch (e) {
        case '"': c = '"'; break;
        case '\\': c = '\\'; break;
        case '/': c = '/'; break;
        case 'b': c = '\b'; break;
        case 'f': c = '\f'; break;
        case 'n': c = '\n'; break;
        case 'r': c = '\r'; break;
        case 't': c = '\t'; break;
        case 'u':
            if (stop - q < 4 || hex4(q, &c)) return fail(s, "bad \\u escape");
            q += 4;
            if (c >= 0xD800 && c < 0xDC00) {
                uint32_t lo;
                if (stop - q < 6 || q[0] != '\\' || q[1] != 'u' || hex4(q + 2, &lo) || lo < 0xDC00 || lo > 0xDFFF)
                    return fail(s, "unpaired surrogate");
                q += 6;
                c = 0x10000 + ((c - 0xD800) << 10) + (lo - 0xDC00);
            } else if (c >= 0xDC00 && c < 0xE000) {
                return fail(s, "unpaired surrogate");
            }
            if (c == 0) return fail(s, "NUL in a string");
            break;
        default: return fail(s, "bad escape");
        }
        if (put_utf8(buf, &n, cap - 1, c)) return fail(s, "bad escape");
    }
    buf[n] = 0;
    *out = buf;
    *len = n;
    return 0;
}

static int parse_value(parser *s, hj_value *v);

static int parse_number(parser *s, hj_value *v) {
    const char *start = s->p;
    if (s->p < s->end && *s->p == '-') ++s->p;
    if (s->p >= s->end) return fail(s, "bad number");
    if (*s->p == '0') {
        ++s->p;
    } else if (*s->p >= '1' && *s->p <= '9') {
        while (s->p < s->end && *s->p >= '0' && *s->p <= '9') ++s->p;
    } else {
        return fail(s, "bad number");
    }
    int integral = 1;
    if (s->p < s->end && *s->p == '.') {
        integral = 0;
        ++s->p;
        if (s->p >= s->end || *s->p < '0' || *s->p > '9') return fail(s, "bad number");
        while (s->p < s->end && *s->p >= '0' && *s->p <= '9') ++s->p;
    }
    if (s->p < s->end && (*s->p == 'e' || *s->p == 'E')) {
        integral = 0;
        ++s->p;
        if (s->p < s->end && (*s->p == '+' || *s->p == '-')) ++s->p;
        if (s->p >= s->end || *s->p < '0' || *s->p > '9') return fail(s, "bad number");
        while (s->p < s->end && *s->p >= '0' && *s->p <= '9') ++s->p;
    }
    char tmp[64];
    const size_t n = (size_t)(s->p - start);
    if (n >= sizeof(tmp)) return fail(s, "number too long");
    memcpy(tmp, start, n);
    tmp[n] = 0;
    v->kind = HJ_NUMBER;
    v->number = strtod(tmp, NULL);
    if (!isfinite(v->number)) return fail(s, "number out of range");
    v->integral = integral && fabs(v->number) <= 9007199254740992.0;
    v->integer = v->integral ? (int64_t)v->number : 0;
    return 0;
}

static int literal(parser *s, const char *word) {
    const size_t n = strlen(word);
    if ((size_t)(s->end - s->p) < n || memcmp(s->p, word, n)) return fail(s, "bad literal");
    s->p += n;
    return 0;
}

/* Arrays and objects. Members are parsed into a stack chunk of 64 and flushed into an arena table that grows
 * geometrically (the waste is at most the final table's size); at most HJ_MAX_MEMBERS members per container. */
#define HJ_MAX_MEMBERS 4096
#define HJ_CHUNK 64

typedef struct {
    hj_value *items;
    const char **keys;
    size_t count, cap;
} members;

static int flush(parser *s, members *m, const hj_value *chunk, const char *const *chunk_keys, size_t n, int object) {
    if (m->count + n > HJ_MAX_MEMBERS) return fail(s, "too many members");
    if (m->count + n > m->cap) {
        size_t cap = m->cap ? 2 * m->cap : HJ_CHUNK;
        while (cap < m->count + n) cap *= 2;
        hj_value *items = (hj_value *)new_bytes(s, cap * sizeof(hj_value), sizeof(void *));
        const char **keys = object ? (const char **)new_bytes(s, cap * sizeof(char *), sizeof(void *)) : NULL;
        if (!items || (object && !keys)) return fail(s, "input too large");
        if (m->count) {
            memcpy(items, m->items, m->count * sizeof(hj_value));
            if (object) memcpy(keys, m->keys, m->count * sizeof(char *));
        }
        m->items = items, m->keys = keys, m->cap = cap;
    }
    memcpy(m->items + m->count, chunk, n * sizeof(hj_value));
    if (object) memcpy(m->keys + m->count, chunk_keys, n * sizeof(char *));
    m->count += n;
    return 0;
}

static int parse_container(parser *s, hj_value *v, int object) {
    if (++s->depth > 32) return fail(s, "nesting too deep");
    ++s->p;
    const char close = object ? '}' : ']';
    hj_value chunk[HJ_CHUNK];
    const char *chunk_keys[HJ_CHUNK];
    size_t n = 0;
    members m = {NULL, NULL, 0, 0};
    skip(s);
    if (s->p < s->end && *s->p == close) {
        ++s->p;
    } else {
        for (;;) {
            skip(s);
            const char *key = NULL;
            if (object) {
                size_t klen;
                if (s->p >= s->end || *s->p != '"') return fail(s, "expected a key");
                if (parse_string(s, &key, &klen)) return -1;
                skip(s);
                if (s->p >= s->end || *s->p != ':') return fail(s, "expected ':'");
                ++s->p;
            }
            if (n == HJ_CHUNK) {
                if (flush(s, &m, chunk, chunk_keys, n, object)) return -1;
                n = 0;
            }
            if (parse_value(s, &chunk[n])) return -1;
            chunk_keys[n++] = key;
            skip(s);
            if (s->p < s->end && *s->p == ',') {
                ++s->p;
                continue;
            }
            if (s->p < s->end && *s->p == close) {
                ++s->p;
                break;
            }
            return fail(s, object ? "expected ',' or '}'" : "expected ',' or ']'");
        }
        if (flush(s, &m, chunk, chunk_keys, n, object)) return -1;
    }
    if (object)
        for (size_t i = 0; i < m.count; ++i)
            for (size_t j = i + 1; j < m.count; ++j)
                if (!strcmp(m.keys[i], m.keys[j])) return fail(s, "duplicate key");
    v->kind = object ? HJ_OBJECT : HJ_ARRAY;
    v->items = m.items;
    v->keys = m.keys;
    v->count = m.count;
    --s->depth;
    return 0;
}

static int parse_value(parser *s, hj_value *v) {
    memset(v, 0, sizeof(*v));
    skip(s);
    if (s->p >= s->end) return fail(s, "unexpected end");
    if (s->a->used_values == s->a->max_values) return fail(s, "input too large (values)");
    s->a->used_values++; /* counted here, held in their container's table (the root in its own slot) */
    switch (*s->p) {
    case '{': return parse_container(s, v, 1);
    case '[': return parse_container(s, v, 0);
    case '"':
        v->kind = HJ_STRING;
        return parse_string(s, &v->string, &v->length);
    case 't': v->kind = HJ_BOOL, v->boolean = 1; return literal(s, "true");
    case 'f': v->kind = HJ_BOOL, v->boolean = 0; return literal(s, "false");
    case 'n': v->kind = HJ_NULL; return literal(s, "null");
    default: return parse_number(s, v);
    }
}

int hj_parse(const char *text, size_t len, hj_arena *arena, hj_value **out, char *error, size_t error_len) {
    parser s = {text, text + len, arena, error, error_len, 0};
    hj_value *root = (hj_value *)new_bytes(&s, sizeof(hj_value), sizeof(void *));
    if (!root) return fail(&s, "input too large");
    if (parse_value(&s, root)) return -1;
    skip(&s);
    if (s.p != s.end) return fail(&s, "trailing characters");
    *out = root;
    return 0;
}

const hj_value *hj_get(const hj_value *object, const char *key) {
    if (!object || object->kind != HJ_OBJECT) return NULL;
    for (size_t i = 0; i < object->count; ++i)
        if (!strcmp(object->keys[i], key)) return &object->items[i];
    return NULL;
}

/* ---- writer -------------------------------------------------------------------------------------------------- */

void hjw_init(hjw *w, char *buf, size_t cap) {
    memset(w, 0, sizeof(*w));
    w->buf = buf;
    w->cap = cap;
    w->first[0] = 1;
    if (cap) buf[0] = 0;
}

int hjw_ok(const hjw *w) { return !w->overflow; }

static void emit(hjw *w, const char *s, size_t n) {
    if (w->overflow) return;
    if (w->len + n + 1 > w->cap) {
        w->overflow = 1;
        return;
    }
    memcpy(w->buf + w->len, s, n);
    w->len += n;
    w->buf[w->len] = 0;
}

static void emits(hjw *w, const char *s) { emit(w, s, strlen(s)); }

/* A comma before every element except the first of its container (a value after a key takes none). */
static void element(hjw *w) {
    if (w->after_key) {
        w->after_key = 0;
        return;
    }
    if (!w->first[w->depth]) emits(w, ",");
    w->first[w->depth] = 0;
}

void hjw_object(hjw *w) {
    element(w);
    emits(w, "{");
    if (w->depth < 32) w->first[++w->depth] = 1;
}

void hjw_array(hjw *w) {
    element(w);
    emits(w, "[");
    if (w->depth < 32) w->first[++w->depth] = 1;
}

void hjw_end(hjw *w, char close) {
    char c[2] = {close, 0};
    emits(w, c);
    if (w->depth > 0) --w->depth;
}

static void quoted(hjw *w, const char *s) {
    emits(w, "\"");
    for (const unsigned char *p = (const unsigned char *)s; *p; ++p) {
        char esc[8];
        if (*p == '"') emits(w, "\\\"");
        else if (*p == '\\') emits(w, "\\\\");
        else if (*p == '\n') emits(w, "\\n");
        else if (*p == '\t') emits(w, "\\t");
        else if (*p == '\r') emits(w, "\\r");
        else if (*p < 0x20) snprintf(esc, sizeof esc, "\\u%04x", *p), emits(w, esc);
        else emit(w, (const char *)p, 1);
    }
    emits(w, "\"");
}

void hjw_key(hjw *w, const char *key) {
    element(w);
    quoted(w, key);
    emits(w, ":");
    w->after_key = 1;
}

void hjw_str(hjw *w, const char *value) {
    element(w);
    quoted(w, value);
}

void hjw_u64(hjw *w, uint64_t value) {
    char b[32];
    element(w);
    snprintf(b, sizeof b, "%llu", (unsigned long long)value);
    emits(w, b);
}

void hjw_i64(hjw *w, int64_t value) {
    char b[32];
    element(w);
    snprintf(b, sizeof b, "%lld", (long long)value);
    emits(w, b);
}

void hjw_double(hjw *w, double value) {
    char b[40];
    element(w);
    if (!isfinite(value)) {
        emits(w, "null");
        return;
    }
    snprintf(b, sizeof b, "%.17g", value); /* round-trips binary64 exactly */
    emits(w, b);
}

void hjw_bool(hjw *w, int value) {
    element(w);
    emits(w, value ? "true" : "false");
}

void hjw_null(hjw *w) {
    element(w);
    emits(w, "null");
}

void hjw_hex(hjw *w, const uint8_t *bytes, size_t n) {
    static const char digits[] = "0123456789abcdef";
    element(w);
    emits(w, "\"");
    for (size_t i = 0; i < n; ++i) {
        const char pair[2] = {digits[bytes[i] >> 4], digits[bytes[i] & 15]};
        emit(w, pair, 2);
    }
    emits(w, "\"");
}

void hjw_kstr(hjw *w, const char *key, const char *value) { hjw_key(w, key), hjw_str(w, value); }
void hjw_ku64(hjw *w, const char *key, uint64_t value) { hjw_key(w, key), hjw_u64(w, value); }
void hjw_ki64(hjw *w, const char *key, int64_t value) { hjw_key(w, key), hjw_i64(w, value); }
void hjw_kbool(hjw *w, const char *key, int value) { hjw_key(w, key), hjw_bool(w, value); }
void hjw_khex(hjw *w, const char *key, const uint8_t *bytes, size_t n) { hjw_key(w, key), hjw_hex(w, bytes, n); }
