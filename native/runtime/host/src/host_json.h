/* The native host's JSON (docs/NATIVE_HOST.md): a strict, bounded parser for operator input (evidence items, session
 * scripts) and a writer for its one-object-per-operation output. No allocation beyond the caller's arena. */
#ifndef ELPIS_HOST_JSON_H
#define ELPIS_HOST_JSON_H

#include <stddef.h>
#include <stdint.h>

typedef enum { HJ_NULL, HJ_BOOL, HJ_NUMBER, HJ_STRING, HJ_ARRAY, HJ_OBJECT } hj_kind;

typedef struct hj_value {
    hj_kind kind;
    int boolean;
    double number;
    int integral;           /* the number was written without fraction or exponent */
    int64_t integer;        /* its value when integral (within +-2^53) */
    const char *string;     /* NUL-terminated, unescaped (strings and object keys) */
    size_t length;
    struct hj_value *items; /* array items, or object values */
    const char **keys;      /* object keys (count of them) */
    size_t count;
} hj_value;

/* A fixed arena: at most `max_values` values, held with their strings and key tables in `cap_bytes` bytes (a value
 * costs sizeof(hj_value), its container's tables grow geometrically: allow 4 * sizeof(hj_value) per value). */
typedef struct {
    size_t max_values, used_values;
    char *bytes;
    size_t cap_bytes, used_bytes;
} hj_arena;

/* Parse exactly one JSON document of `len` bytes (depth <= 32). 0 on success; otherwise -1 with `error` set. */
int hj_parse(const char *text, size_t len, hj_arena *arena, hj_value **out, char *error, size_t error_len);
/* The member `key` of an object, or NULL. */
const hj_value *hj_get(const hj_value *object, const char *key);

/* The writer: a bounded buffer; overflow is sticky and reported by hjw_ok. */
typedef struct {
    char *buf;
    size_t cap, len;
    int overflow;
    int depth;
    int after_key;
    int first[34];
} hjw;

void hjw_init(hjw *w, char *buf, size_t cap);
int hjw_ok(const hjw *w);
void hjw_object(hjw *w);
void hjw_array(hjw *w);
void hjw_end(hjw *w, char close);
void hjw_key(hjw *w, const char *key);
void hjw_str(hjw *w, const char *value);
void hjw_u64(hjw *w, uint64_t value);
void hjw_i64(hjw *w, int64_t value);
void hjw_double(hjw *w, double value);
void hjw_bool(hjw *w, int value);
void hjw_null(hjw *w);
void hjw_hex(hjw *w, const uint8_t *bytes, size_t n);
/* key: value shorthands. */
void hjw_kstr(hjw *w, const char *key, const char *value);
void hjw_ku64(hjw *w, const char *key, uint64_t value);
void hjw_ki64(hjw *w, const char *key, int64_t value);
void hjw_kbool(hjw *w, const char *key, int value);
void hjw_khex(hjw *w, const char *key, const uint8_t *bytes, size_t n);

#endif
