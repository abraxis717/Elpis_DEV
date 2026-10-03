#include "elpis/dsv41_native.h"

#include <math.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

struct elpis_dsv41_attention_state {
    size_t local_window, head_dim, index_dim, max_tokens, ratio, capacity, count;
    size_t logical_bytes;
    int owner;
    float *window, *compressed, *index_keys, *pending, *scores;
};

static int checked_mul(size_t a, size_t b, size_t *out) {
    if (a != 0u && b > SIZE_MAX / a) return 0;
    *out = a * b;
    return 1;
}

static int checked_add(size_t a, size_t b, size_t *out) {
    if (b > SIZE_MAX - a) return 0;
    *out = a + b;
    return 1;
}

static float *alloc_f32(size_t n) {
    size_t bytes;
    if (!checked_mul(n, sizeof(float), &bytes) || bytes == 0u) return NULL;
    return (float *)calloc(1u, bytes);
}

static void free_state(elpis_dsv41_attention_state *s) {
    if (s == NULL) return;
    free(s->window);
    free(s->compressed);
    free(s->index_keys);
    free(s->pending);
    free(s->scores);
    free(s);
}

static int add_float_storage(size_t n, size_t *bytes) {
    size_t part, total;
    if (!checked_mul(n, sizeof(float), &part)) return 0;
    if (!checked_add(*bytes, part, &total)) return 0;
    *bytes = total;
    return 1;
}

elpis_dsv41_native_status
elpis_dsv41_attention_state_create(size_t local_window,
                                   size_t head_dim,
                                   size_t index_dim,
                                   size_t max_tokens,
                                   size_t ratio,
                                   int owner,
                                   elpis_dsv41_attention_state **out) {
    elpis_dsv41_attention_state *s;
    size_t n, bytes = 0u;

    if (out == NULL || *out != NULL || local_window == 0u || head_dim == 0u ||
        index_dim == 0u || max_tokens == 0u || (owner != 0 && owner != 1) ||
        (owner && ratio == 0u)) {
        return ELPIS_DSV41_NATIVE_INVALID;
    }

    s = (elpis_dsv41_attention_state *)calloc(1u, sizeof(*s));
    if (s == NULL) return ELPIS_DSV41_NATIVE_NOMEM;

    s->local_window = local_window;
    s->head_dim = head_dim;
    s->index_dim = index_dim;
    s->max_tokens = max_tokens;
    s->ratio = ratio;
    s->owner = owner;
    s->capacity = owner ? max_tokens / ratio : 0u;

    if (!checked_mul(local_window, head_dim, &n) || !add_float_storage(n, &bytes)) {
        free_state(s);
        return ELPIS_DSV41_NATIVE_INVALID;
    }
    s->window = alloc_f32(n);
    if (s->window == NULL) {
        free_state(s);
        return ELPIS_DSV41_NATIVE_NOMEM;
    }

    if (owner) {
        if (s->capacity == 0u) {
            free_state(s);
            return ELPIS_DSV41_NATIVE_INVALID;
        }

        if (!checked_mul(s->capacity, head_dim, &n) || !add_float_storage(n, &bytes)) {
            free_state(s);
            return ELPIS_DSV41_NATIVE_INVALID;
        }
        s->compressed = alloc_f32(n);
        if (s->compressed == NULL) {
            free_state(s);
            return ELPIS_DSV41_NATIVE_NOMEM;
        }

        if (!checked_mul(s->capacity, index_dim, &n) || !add_float_storage(n, &bytes)) {
            free_state(s);
            return ELPIS_DSV41_NATIVE_INVALID;
        }
        s->index_keys = alloc_f32(n);
        if (s->index_keys == NULL) {
            free_state(s);
            return ELPIS_DSV41_NATIVE_NOMEM;
        }

        if (ratio > 1u) {
            if (!checked_mul(ratio, head_dim, &n) ||
                !add_float_storage(n, &bytes) ||
                !add_float_storage(n, &bytes)) {
                free_state(s);
                return ELPIS_DSV41_NATIVE_INVALID;
            }
            s->pending = alloc_f32(n);
            s->scores = alloc_f32(n);
            if (s->pending == NULL || s->scores == NULL) {
                free_state(s);
                return ELPIS_DSV41_NATIVE_NOMEM;
            }
        }
    }

    s->logical_bytes = bytes;
    *out = s;
    return ELPIS_DSV41_NATIVE_OK;
}

elpis_dsv41_native_status
elpis_dsv41_attention_state_destroy(elpis_dsv41_attention_state **state) {
    if (state == NULL) return ELPIS_DSV41_NATIVE_INVALID;
    if (*state != NULL) {
        free_state(*state);
        *state = NULL;
    }
    return ELPIS_DSV41_NATIVE_OK;
}

size_t elpis_dsv41_attention_state_bytes(const elpis_dsv41_attention_state *s) {
    return s == NULL ? 0u : s->logical_bytes;
}

size_t elpis_dsv41_attention_state_count(const elpis_dsv41_attention_state *s) {
    return s == NULL ? 0u : s->count;
}

elpis_dsv41_native_status
elpis_dsv41_attention_local_store_f32(elpis_dsv41_attention_state *s,
                                      size_t position,
                                      const float *local) {
    size_t slot, i;
    if (s == NULL || local == NULL || position >= s->max_tokens)
        return ELPIS_DSV41_NATIVE_INVALID;

    for (i = 0u; i < s->head_dim; ++i)
        if (!isfinite(local[i]))
            return ELPIS_DSV41_NATIVE_NONFINITE;

    slot = position % s->local_window;
    memcpy(s->window + slot * s->head_dim,
           local,
           s->head_dim * sizeof(float));
    return ELPIS_DSV41_NATIVE_OK;
}

elpis_dsv41_native_status
elpis_dsv41_attention_local_copy_f32(const elpis_dsv41_attention_state *s,
                                     size_t position,
                                     float *out,
                                     size_t row_capacity,
                                     size_t *rows) {
    size_t n, first, r, slot;
    if (s == NULL || out == NULL || rows == NULL || position >= s->max_tokens)
        return ELPIS_DSV41_NATIVE_INVALID;

    n = position + 1u < s->local_window ? position + 1u : s->local_window;
    if (row_capacity < n)
        return ELPIS_DSV41_NATIVE_INVALID;

    first = position + 1u - n;
    for (r = 0u; r < n; ++r) {
        slot = (first + r) % s->local_window;
        memcpy(out + r * s->head_dim,
               s->window + slot * s->head_dim,
               s->head_dim * sizeof(float));
    }
    *rows = n;
    return ELPIS_DSV41_NATIVE_OK;
}

elpis_dsv41_native_status
elpis_dsv41_attention_compress_push_f32(elpis_dsv41_attention_state *s,
                                        size_t position,
                                        const float *kv,
                                        const float *gate,
                                        float *out_latent,
                                        int *ready) {
    size_t d, r, slot;
    if (s == NULL || !s->owner || kv == NULL || gate == NULL ||
        out_latent == NULL || ready == NULL || position >= s->max_tokens)
        return ELPIS_DSV41_NATIVE_INVALID;

    for (d = 0u; d < s->head_dim; ++d) {
        if (!isfinite(kv[d]) || !isfinite(gate[d]))
            return ELPIS_DSV41_NATIVE_NONFINITE;
    }

    if (s->ratio == 1u) {
        memcpy(out_latent, kv, s->head_dim * sizeof(float));
        *ready = 1;
        return ELPIS_DSV41_NATIVE_OK;
    }

    slot = position % s->ratio;
    memcpy(s->pending + slot * s->head_dim,
           kv,
           s->head_dim * sizeof(float));
    memcpy(s->scores + slot * s->head_dim,
           gate,
           s->head_dim * sizeof(float));

    if ((position + 1u) % s->ratio != 0u) {
        *ready = 0;
        return ELPIS_DSV41_NATIVE_OK;
    }

    for (d = 0u; d < s->head_dim; ++d) {
        float maximum = -INFINITY;
        float denom = 0.0f;
        float total = 0.0f;

        for (r = 0u; r < s->ratio; ++r) {
            float v = s->scores[r * s->head_dim + d];
            if (v > maximum) maximum = v;
        }

        for (r = 0u; r < s->ratio; ++r)
            denom += expf(s->scores[r * s->head_dim + d] - maximum);

        if (!(denom > 0.0f) || !isfinite(denom))
            return ELPIS_DSV41_NATIVE_NONFINITE;

        for (r = 0u; r < s->ratio; ++r) {
            float w = expf(s->scores[r * s->head_dim + d] - maximum) / denom;
            total += s->pending[r * s->head_dim + d] * w;
        }
        out_latent[d] = total;
    }

    *ready = 1;
    return ELPIS_DSV41_NATIVE_OK;
}

elpis_dsv41_native_status
elpis_dsv41_attention_publish_group_f32(elpis_dsv41_attention_state *s,
                                        size_t position,
                                        const float *compressed,
                                        const float *index_key) {
    size_t group, i;
    if (s == NULL || !s->owner || compressed == NULL || index_key == NULL ||
        position >= s->max_tokens || (position + 1u) % s->ratio != 0u)
        return ELPIS_DSV41_NATIVE_INVALID;

    group = (position + 1u) / s->ratio;
    if (group == 0u || group > s->capacity)
        return ELPIS_DSV41_NATIVE_INVALID;

    for (i = 0u; i < s->head_dim; ++i)
        if (!isfinite(compressed[i]))
            return ELPIS_DSV41_NATIVE_NONFINITE;

    for (i = 0u; i < s->index_dim; ++i)
        if (!isfinite(index_key[i]))
            return ELPIS_DSV41_NATIVE_NONFINITE;

    memcpy(s->compressed + (group - 1u) * s->head_dim,
           compressed,
           s->head_dim * sizeof(float));
    memcpy(s->index_keys + (group - 1u) * s->index_dim,
           index_key,
           s->index_dim * sizeof(float));
    s->count = group;
    return ELPIS_DSV41_NATIVE_OK;
}

elpis_dsv41_native_status
elpis_dsv41_attention_index_copy_f32(const elpis_dsv41_attention_state *s,
                                     float *out,
                                     size_t row_capacity,
                                     size_t *rows) {
    if (s == NULL || !s->owner || out == NULL || rows == NULL ||
        row_capacity < s->count)
        return ELPIS_DSV41_NATIVE_INVALID;

    if (s->count != 0u)
        memcpy(out,
               s->index_keys,
               s->count * s->index_dim * sizeof(float));

    *rows = s->count;
    return ELPIS_DSV41_NATIVE_OK;
}

elpis_dsv41_native_status
elpis_dsv41_attention_gather_compressed_f32(
    const elpis_dsv41_attention_state *s,
    const uint32_t *positions,
    size_t selected,
    float *out) {
    size_t r, pos;
    if (s == NULL || !s->owner || positions == NULL || out == NULL)
        return ELPIS_DSV41_NATIVE_INVALID;

    for (r = 0u; r < selected; ++r) {
        pos = (size_t)positions[r];
        if (pos >= s->count)
            return ELPIS_DSV41_NATIVE_INVALID;

        memcpy(out + r * s->head_dim,
               s->compressed + pos * s->head_dim,
               s->head_dim * sizeof(float));
    }
    return ELPIS_DSV41_NATIVE_OK;
}
