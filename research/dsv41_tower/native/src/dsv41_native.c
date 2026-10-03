#include "elpis/dsv41_native.h"

#include <math.h>

static int finite_vector(const float *x, size_t n) {
    size_t i;
    for (i = 0; i < n; ++i) {
        if (!isfinite(x[i])) {
            return 0;
        }
    }
    return 1;
}

uint32_t elpis_dsv41_native_abi_version(void) {
    return ELPIS_DSV41_NATIVE_ABI_V1;
}

uint64_t elpis_dsv41_native_capabilities(void) {
    return ELPIS_DSV41_CAP_LINEAR_F32 |
           ELPIS_DSV41_CAP_RMS_F32 |
           ELPIS_DSV41_CAP_HC_PRE_F32 |
           ELPIS_DSV41_CAP_HC_POST_F32 |
           ELPIS_DSV41_CAP_QUANT_DEQUANT_F32 |
           ELPIS_DSV41_CAP_ROPE_F32 |
           ELPIS_DSV41_CAP_CANDIDATE_MASK_F32 |
           ELPIS_DSV41_CAP_SELECT_POSITIONS_F32 |
           ELPIS_DSV41_CAP_SPARSE_ATTN_F32 |
           ELPIS_DSV41_CAP_HC_MIXES_F32 |
           ELPIS_DSV41_CAP_ROUTE_F32 |
           ELPIS_DSV41_CAP_EXPERT_F32 |
           ELPIS_DSV41_CAP_ATTN_STATE_V1 |
           ELPIS_DSV41_CAP_ENGRAM_GATED_WRITE_F32 |
           ELPIS_DSV41_CAP_INDEX_SCORES_F32 |
           ELPIS_DSV41_CAP_GROUPED_OUTPUT_F32 |
           ELPIS_DSV41_CAP_LOCAL_ATTN_APPLY_F32 |
           ELPIS_DSV41_CAP_COMPRESSED_ATTN_APPLY_F32 |
           ELPIS_DSV41_CAP_ROUTE_ORDER_U32 |
           ELPIS_DSV41_CAP_EXPERT_ACCUMULATE_F32 |
           ELPIS_DSV41_CAP_LAYER_BEGIN_F32 |
           ELPIS_DSV41_CAP_LAYER_AFTER_ATTN_ROUTE_F32 |
           ELPIS_DSV41_CAP_LAYER_FINISH_F32;
}

elpis_dsv41_native_status
elpis_dsv41_linear_f32(const float *x,
                       const float *weight,
                       float *out,
                       size_t in_dim,
                       size_t out_dim) {
    size_t o, i;
    if (x == NULL || weight == NULL || out == NULL || in_dim == 0 || out_dim == 0) {
        return ELPIS_DSV41_NATIVE_INVALID;
    }
    for (o = 0; o < out_dim; ++o) {
        float sum = 0.0f;
        const float *row = weight + o * in_dim;
        for (i = 0; i < in_dim; ++i) {
            sum += x[i] * row[i];
        }
        out[o] = sum;
    }
    return finite_vector(out, out_dim) ? ELPIS_DSV41_NATIVE_OK : ELPIS_DSV41_NATIVE_NONFINITE;
}

elpis_dsv41_native_status
elpis_dsv41_rms_f32(const float *x,
                    const float *weight,
                    float eps,
                    float *out,
                    size_t dim) {
    size_t i;
    float sum = 0.0f;
    float inv;
    if (x == NULL || weight == NULL || out == NULL || dim == 0 ||
        !isfinite(eps) || eps < 0.0f) {
        return ELPIS_DSV41_NATIVE_INVALID;
    }
    for (i = 0; i < dim; ++i) {
        sum += x[i] * x[i];
    }
    inv = 1.0f / sqrtf(sum / (float)dim + eps);
    if (!isfinite(inv)) {
        return ELPIS_DSV41_NATIVE_NONFINITE;
    }
    for (i = 0; i < dim; ++i) {
        out[i] = x[i] * inv * weight[i];
    }
    return finite_vector(out, dim) ? ELPIS_DSV41_NATIVE_OK : ELPIS_DSV41_NATIVE_NONFINITE;
}

elpis_dsv41_native_status
elpis_dsv41_hc_pre_f32(const float *stream,
                       const float *pre,
                       float *out,
                       size_t copies,
                       size_t dim) {
    size_t d, c;
    if (stream == NULL || pre == NULL || out == NULL || copies == 0 || dim == 0) {
        return ELPIS_DSV41_NATIVE_INVALID;
    }
    for (d = 0; d < dim; ++d) {
        float sum = 0.0f;
        for (c = 0; c < copies; ++c) {
            sum += pre[c] * stream[c * dim + d];
        }
        out[d] = sum;
    }
    return finite_vector(out, dim) ? ELPIS_DSV41_NATIVE_OK : ELPIS_DSV41_NATIVE_NONFINITE;
}

elpis_dsv41_native_status
elpis_dsv41_hc_post_f32(const float *x,
                        const float *residual,
                        const float *post,
                        const float *comb,
                        float *out,
                        size_t copies,
                        size_t dim) {
    size_t j, d, i;
    if (x == NULL || residual == NULL || post == NULL || comb == NULL ||
        out == NULL || copies == 0 || dim == 0) {
        return ELPIS_DSV41_NATIVE_INVALID;
    }
    for (j = 0; j < copies; ++j) {
        for (d = 0; d < dim; ++d) {
            float sum = post[j] * x[d];
            for (i = 0; i < copies; ++i) {
                sum += comb[i * copies + j] * residual[i * dim + d];
            }
            out[j * dim + d] = sum;
        }
    }
    return finite_vector(out, copies * dim) ? ELPIS_DSV41_NATIVE_OK : ELPIS_DSV41_NATIVE_NONFINITE;
}

static float e4m3_level(unsigned code) {
    unsigned exponent = code / 8u;
    unsigned mantissa = code % 8u;
    if (exponent == 0u) {
        return ldexpf((float)mantissa, -9);
    }
    return ldexpf((float)(8u + mantissa), (int)exponent - 10);
}

static float nearest_e4m3(float value) {
    float magnitude = fabsf(value);
    unsigned lo = 0u, hi = 126u, upper, lower;
    float a, b, dl, du, chosen;
    if (magnitude >= e4m3_level(126u)) {
        return copysignf(e4m3_level(126u), value);
    }
    while (lo < hi) {
        unsigned mid = lo + (hi - lo) / 2u;
        if (e4m3_level(mid) < magnitude) {
            lo = mid + 1u;
        } else {
            hi = mid;
        }
    }
    upper = lo;
    lower = upper == 0u ? 0u : upper - 1u;
    a = e4m3_level(lower);
    b = e4m3_level(upper);
    dl = magnitude - a;
    du = b - magnitude;
    chosen = (du < dl || (du == dl && (upper & 1u) == 0u)) ? b : a;
    return copysignf(chosen, value);
}

static float nearest_e2m1(float value) {
    static const float levels[8] = {0.0f, 0.5f, 1.0f, 1.5f, 2.0f, 3.0f, 4.0f, 6.0f};
    float magnitude = fabsf(value);
    unsigned upper = 0u, lower;
    float dl, du, chosen;
    if (magnitude >= 6.0f) {
        return copysignf(6.0f, value);
    }
    while (upper < 7u && levels[upper] < magnitude) {
        ++upper;
    }
    lower = upper == 0u ? 0u : upper - 1u;
    dl = magnitude - levels[lower];
    du = levels[upper] - magnitude;
    chosen = (du < dl || (du == dl && (upper & 1u) == 0u)) ?
             levels[upper] : levels[lower];
    return copysignf(chosen, value);
}

static float ceil_power_two(float value) {
    int exponent;
    float mantissa = frexpf(value, &exponent);
    if (mantissa == 0.5f) {
        return ldexpf(1.0f, exponent - 1);
    }
    return ldexpf(1.0f, exponent);
}

elpis_dsv41_native_status
elpis_dsv41_quant_dequant_f32(const float *x,
                              float *out,
                              size_t rows,
                              size_t width,
                              uint32_t mode) {
    size_t block, row, start, i;
    if (x == NULL || out == NULL || rows == 0 || width == 0) {
        return ELPIS_DSV41_NATIVE_INVALID;
    }
    if (mode == ELPIS_DSV41_QUANT_COMPRESSED) {
        block = 16u;
    } else if (mode == ELPIS_DSV41_QUANT_LOCAL || mode == ELPIS_DSV41_QUANT_INDEX) {
        block = 32u;
    } else {
        return ELPIS_DSV41_NATIVE_INVALID;
    }
    if (width % block != 0u) {
        return ELPIS_DSV41_NATIVE_INVALID;
    }

    for (row = 0; row < rows; ++row) {
        for (start = 0; start < width; start += block) {
            float amax = 0.0f;
            float scale;
            for (i = 0; i < block; ++i) {
                float value = x[row * width + start + i];
                float magnitude;
                if (!isfinite(value)) {
                    return ELPIS_DSV41_NATIVE_NONFINITE;
                }
                magnitude = fabsf(value);
                if (magnitude > amax) {
                    amax = magnitude;
                }
            }

            if (mode == ELPIS_DSV41_QUANT_LOCAL) {
                float floor_value = amax < 1e-4f ? 1e-4f : amax;
                scale = ceil_power_two(floor_value * (float)(1.0 / 448.0));
            } else if (mode == ELPIS_DSV41_QUANT_INDEX) {
                float minimum = ldexpf(6.0f, -126);
                float floor_value = amax < minimum ? minimum : amax;
                scale = ceil_power_two(floor_value * (float)(1.0 / 6.0));
            } else {
                float minimum = ldexpf(6.0f, -9);
                float floor_value = amax < minimum ? minimum : amax;
                scale = nearest_e4m3(floor_value / 6.0f);
            }
            if (!(scale > 0.0f) || !isfinite(scale)) {
                return ELPIS_DSV41_NATIVE_NONFINITE;
            }
            for (i = 0; i < block; ++i) {
                float normalized = x[row * width + start + i] / scale;
                float quantized = mode == ELPIS_DSV41_QUANT_LOCAL ?
                                  nearest_e4m3(normalized) : nearest_e2m1(normalized);
                out[row * width + start + i] = quantized * scale;
            }
        }
    }
    return finite_vector(out, rows * width) ? ELPIS_DSV41_NATIVE_OK : ELPIS_DSV41_NATIVE_NONFINITE;
}

elpis_dsv41_native_status
elpis_dsv41_rope_f32(const float *x,
                     const float *freq,
                     float *out,
                     size_t vectors,
                     size_t dim,
                     size_t pairs,
                     int inverse) {
    size_t v, i, tail;
    if (x == NULL || freq == NULL || out == NULL || vectors == 0 || dim == 0 ||
        pairs == 0 || pairs > dim / 2u || (inverse != 0 && inverse != 1)) {
        return ELPIS_DSV41_NATIVE_INVALID;
    }
    tail = dim - 2u * pairs;
    for (v = 0; v < vectors; ++v) {
        for (i = 0; i < tail; ++i) {
            out[v * dim + i] = x[v * dim + i];
        }
        for (i = 0; i < pairs; ++i) {
            float a = x[v * dim + tail + 2u * i];
            float b = x[v * dim + tail + 2u * i + 1u];
            float cosine = freq[2u * i];
            float sine = freq[2u * i + 1u];
            if (!isfinite(a) || !isfinite(b) || !isfinite(cosine) || !isfinite(sine)) {
                return ELPIS_DSV41_NATIVE_NONFINITE;
            }
            if (inverse) {
                sine = -sine;
            }
            out[v * dim + tail + 2u * i] = a * cosine - b * sine;
            out[v * dim + tail + 2u * i + 1u] = b * cosine + a * sine;
        }
    }
    return finite_vector(out, vectors * dim) ? ELPIS_DSV41_NATIVE_OK : ELPIS_DSV41_NATIVE_NONFINITE;
}

static int block_selected(const uint8_t *mask, size_t block, size_t block_size) {
    return mask[block * block_size] != 0u;
}

elpis_dsv41_native_status
elpis_dsv41_candidate_mask_f32(const float *logits,
                               uint8_t *mask,
                               size_t width,
                               size_t top_blocks,
                               size_t block_size) {
    size_t blocks, newest, take, rank, b, best, i;
    if (logits == NULL || mask == NULL || width == 0 || top_blocks == 0 || block_size == 0) {
        return ELPIS_DSV41_NATIVE_INVALID;
    }
    blocks = (width + block_size - 1u) / block_size;
    newest = (width - 1u) / block_size;
    take = top_blocks < blocks ? top_blocks : blocks;
    for (i = 0; i < width; ++i) {
        if (isnan(logits[i])) {
            return ELPIS_DSV41_NATIVE_NONFINITE;
        }
        mask[i] = 0u;
    }

    for (rank = 0; rank < take; ++rank) {
        float best_score = -INFINITY;
        best = blocks;
        for (b = 0; b < blocks; ++b) {
            size_t begin, end, j;
            float score = -INFINITY;
            if (block_selected(mask, b, block_size)) {
                continue;
            }
            if (b == newest) {
                score = INFINITY;
            } else {
                begin = b * block_size;
                end = begin + block_size < width ? begin + block_size : width;
                for (j = begin; j < end; ++j) {
                    if (logits[j] > score) {
                        score = logits[j];
                    }
                }
            }
            if (best == blocks || score > best_score) {
                best = b;
                best_score = score;
            }
        }
        if (best == blocks || !(best_score > -INFINITY)) {
            break;
        }
        {
            size_t begin = best * block_size;
            size_t end = begin + block_size < width ? begin + block_size : width;
            for (i = begin; i < end; ++i) {
                mask[i] = 1u;
            }
        }
    }
    return ELPIS_DSV41_NATIVE_OK;
}

static int position_already_selected(const uint32_t *positions, size_t count, size_t value) {
    size_t i;
    for (i = 0; i < count; ++i) {
        if ((size_t)positions[i] == value) {
            return 1;
        }
    }
    return 0;
}

elpis_dsv41_native_status
elpis_dsv41_select_positions_f32(const float *scores,
                                 uint32_t *positions,
                                 size_t width,
                                 size_t topk,
                                 size_t *selected) {
    size_t take, count = 0, rank, i;
    if (scores == NULL || positions == NULL || selected == NULL || width == 0 || topk == 0) {
        return ELPIS_DSV41_NATIVE_INVALID;
    }
    take = topk < width ? topk : width;
    for (i = 0; i < width; ++i) {
        if (isnan(scores[i])) {
            return ELPIS_DSV41_NATIVE_NONFINITE;
        }
    }
    for (rank = 0; rank < take; ++rank) {
        size_t best = width;
        float best_score = -INFINITY;
        for (i = 0; i < width; ++i) {
            if (position_already_selected(positions, count, i)) {
                continue;
            }
            if (best == width || scores[i] > best_score) {
                best = i;
                best_score = scores[i];
            }
        }
        if (best == width) {
            return ELPIS_DSV41_NATIVE_INVALID;
        }
        {
            size_t at = count;
            while (at > 0 && positions[at - 1] > best) {
                positions[at] = positions[at - 1];
                --at;
            }
            positions[at] = (uint32_t)best;
            ++count;
        }
    }
    *selected = count;
    return ELPIS_DSV41_NATIVE_OK;
}

elpis_dsv41_native_status
elpis_dsv41_sparse_attention_f32(const float *query,
                                 const float *kv,
                                 const float *sink,
                                 float *out,
                                 float *score_scratch,
                                 size_t heads,
                                 size_t kv_count,
                                 size_t dim) {
    size_t h, n, d;
    float scale;
    if (query == NULL || kv == NULL || sink == NULL || out == NULL ||
        score_scratch == NULL || heads == 0 || kv_count == 0 || dim == 0) {
        return ELPIS_DSV41_NATIVE_INVALID;
    }
    scale = 1.0f / sqrtf((float)dim);
    for (h = 0; h < heads; ++h) {
        float maximum = sink[h];
        float denominator = 0.0f;
        if (!isfinite(sink[h])) {
            return ELPIS_DSV41_NATIVE_NONFINITE;
        }
        for (n = 0; n < kv_count; ++n) {
            float score = 0.0f;
            for (d = 0; d < dim; ++d) {
                float q = query[h * dim + d];
                float k = kv[n * dim + d];
                if (!isfinite(q) || !isfinite(k)) {
                    return ELPIS_DSV41_NATIVE_NONFINITE;
                }
                score += q * k;
            }
            score *= scale;
            score_scratch[n] = score;
            if (score > maximum) {
                maximum = score;
            }
        }
        for (n = 0; n < kv_count; ++n) {
            score_scratch[n] = expf(score_scratch[n] - maximum);
            denominator += score_scratch[n];
        }
        denominator += expf(sink[h] - maximum);
        if (!(denominator > 0.0f) || !isfinite(denominator)) {
            return ELPIS_DSV41_NATIVE_NONFINITE;
        }
        for (d = 0; d < dim; ++d) {
            float sum = 0.0f;
            for (n = 0; n < kv_count; ++n) {
                sum += score_scratch[n] * kv[n * dim + d];
            }
            out[h * dim + d] = sum / denominator;
        }
    }
    return finite_vector(out, heads * dim) ? ELPIS_DSV41_NATIVE_OK : ELPIS_DSV41_NATIVE_NONFINITE;
}

static float elpis_native_sigmoid(float x) {
    if (x >= 0.0f) {
        float z = expf(-x);
        return 1.0f / (1.0f + z);
    }
    {
        float z = expf(x);
        return z / (1.0f + z);
    }
}

static float elpis_native_softplus(float x) {
    if (x > 20.0f) return x;
    if (x >= 0.0f) return x + log1pf(expf(-x));
    return log1pf(expf(x));
}

elpis_dsv41_native_status
elpis_dsv41_hc_mixes_f32(const float *stream, const float *fn, const float *scale,
                         const float *base, float *pre, float *post, float *comb,
                         float *mix_scratch, size_t copies, size_t dim,
                         size_t sinkhorn_iters, float norm_eps, float hc_eps) {
    size_t flat_dim, mix_dim, r, i, j, iteration;
    float sumsq = 0.0f, divisor;
    if (stream == NULL || fn == NULL || scale == NULL || base == NULL ||
        pre == NULL || post == NULL || comb == NULL || mix_scratch == NULL ||
        copies == 0 || dim == 0 || sinkhorn_iters == 0 ||
        !isfinite(norm_eps) || norm_eps < 0.0f || !isfinite(hc_eps) || hc_eps < 0.0f)
        return ELPIS_DSV41_NATIVE_INVALID;
    flat_dim = copies * dim;
    mix_dim = (2u + copies) * copies;
    for (i = 0; i < flat_dim; ++i) {
        float v = stream[i];
        if (!isfinite(v)) return ELPIS_DSV41_NATIVE_NONFINITE;
        sumsq += v * v;
    }
    divisor = sqrtf(sumsq / (float)flat_dim + norm_eps);
    if (!(divisor > 0.0f) || !isfinite(divisor)) return ELPIS_DSV41_NATIVE_NONFINITE;
    for (r = 0; r < mix_dim; ++r) {
        float total = 0.0f;
        const float *row = fn + r * flat_dim;
        for (i = 0; i < flat_dim; ++i) total += stream[i] * row[i];
        mix_scratch[r] = total / divisor;
    }
    for (i = 0; i < copies; ++i) {
        pre[i] = elpis_native_sigmoid(mix_scratch[i] * scale[0] + base[i]) + hc_eps;
        post[i] = 2.0f * elpis_native_sigmoid(mix_scratch[copies+i] * scale[1] + base[copies+i]);
    }
    for (i = 0; i < copies; ++i) {
        size_t off = 2u * copies + i * copies;
        float maximum = -INFINITY, denominator = 0.0f;
        for (j = 0; j < copies; ++j) {
            float z = mix_scratch[off+j] * scale[2] + base[off+j];
            comb[i*copies+j] = z;
            if (z > maximum) maximum = z;
        }
        for (j = 0; j < copies; ++j) {
            float e = expf(comb[i*copies+j] - maximum);
            comb[i*copies+j] = e;
            denominator += e;
        }
        for (j = 0; j < copies; ++j) comb[i*copies+j] = comb[i*copies+j] / denominator + hc_eps;
    }
    for (j = 0; j < copies; ++j) {
        float total = 0.0f;
        for (i = 0; i < copies; ++i) total += comb[i*copies+j];
        total += hc_eps;
        for (i = 0; i < copies; ++i) comb[i*copies+j] /= total;
    }
    for (iteration = 1; iteration < sinkhorn_iters; ++iteration) {
        for (i = 0; i < copies; ++i) {
            float total = 0.0f;
            for (j = 0; j < copies; ++j) total += comb[i*copies+j];
            total += hc_eps;
            for (j = 0; j < copies; ++j) comb[i*copies+j] /= total;
        }
        for (j = 0; j < copies; ++j) {
            float total = 0.0f;
            for (i = 0; i < copies; ++i) total += comb[i*copies+j];
            total += hc_eps;
            for (i = 0; i < copies; ++i) comb[i*copies+j] /= total;
        }
    }
    return finite_vector(pre,copies) && finite_vector(post,copies) && finite_vector(comb,copies*copies)
        ? ELPIS_DSV41_NATIVE_OK : ELPIS_DSV41_NATIVE_NONFINITE;
}

static int elpis_native_u32_contains(const uint32_t *values, size_t count, size_t value) {
    size_t i;
    for (i = 0; i < count; ++i) if ((size_t)values[i] == value) return 1;
    return 0;
}

elpis_dsv41_native_status
elpis_dsv41_route_f32(const float *x, const float *weight, const float *bias,
                      uint32_t *chosen, float *values, float *score_scratch,
                      size_t dim, size_t expert_count, size_t active,
                      uint32_t score_mode, float gate_temp, int normalize_topk,
                      float route_scale) {
    size_t e,d,rank;
    float maximum=-INFINITY, denominator=0.0f;
    if (x==NULL || weight==NULL || bias==NULL || chosen==NULL || values==NULL ||
        score_scratch==NULL || dim==0 || expert_count==0 || active==0 || active>expert_count ||
        !isfinite(gate_temp) || !(gate_temp>0.0f) || !isfinite(route_scale) ||
        (normalize_topk!=0 && normalize_topk!=1) || score_mode>ELPIS_DSV41_SCORE_SQRT_SOFTPLUS)
        return ELPIS_DSV41_NATIVE_INVALID;
    for (e=0;e<expert_count;++e) {
        float total=0.0f;
        for (d=0;d<dim;++d) total += x[d]*weight[e*dim+d];
        total /= gate_temp;
        if (!isfinite(total) || !isfinite(bias[e])) return ELPIS_DSV41_NATIVE_NONFINITE;
        score_scratch[e]=total;
        if (total>maximum) maximum=total;
    }
    if (score_mode==ELPIS_DSV41_SCORE_SOFTMAX) {
        for (e=0;e<expert_count;++e) { float v=expf(score_scratch[e]-maximum); score_scratch[e]=v; denominator+=v; }
        if (!(denominator>0.0f) || !isfinite(denominator)) return ELPIS_DSV41_NATIVE_NONFINITE;
        for (e=0;e<expert_count;++e) score_scratch[e] /= denominator;
    } else if (score_mode==ELPIS_DSV41_SCORE_SIGMOID) {
        for (e=0;e<expert_count;++e) score_scratch[e]=elpis_native_sigmoid(score_scratch[e]);
    } else {
        for (e=0;e<expert_count;++e) {
            float sp=elpis_native_softplus(score_scratch[e]);
            if (sp<0.0f || !isfinite(sp)) return ELPIS_DSV41_NATIVE_NONFINITE;
            score_scratch[e]=sqrtf(sp);
        }
    }
    for (rank=0;rank<active;++rank) {
        size_t best=expert_count;
        float best_selection=-INFINITY;
        for (e=0;e<expert_count;++e) {
            float sel;
            if (elpis_native_u32_contains(chosen,rank,e)) continue;
            sel=score_scratch[e]+bias[e];
            if (best==expert_count || sel>best_selection) { best=e; best_selection=sel; }
        }
        if (best==expert_count) return ELPIS_DSV41_NATIVE_INVALID;
        chosen[rank]=(uint32_t)best;
        values[rank]=score_scratch[best];
    }
    if (normalize_topk && active>1u) {
        float sum=0.0f;
        for (rank=0;rank<active;++rank) sum += values[rank];
        sum += 1e-20f;
        for (rank=0;rank<active;++rank) values[rank] /= sum;
    }
    for (rank=0;rank<active;++rank) values[rank] *= route_scale;
    return finite_vector(values,active) ? ELPIS_DSV41_NATIVE_OK : ELPIS_DSV41_NATIVE_NONFINITE;
}

elpis_dsv41_native_status
elpis_dsv41_expert_f32(const float *x, const float *w1, const float *w3, const float *w2,
                       float route_weight, int weighted, float swiglu_limit,
                       float *gate_scratch, float *up_scratch, float *out,
                       size_t dim, size_t expert_dim) {
    size_t f,d;
    if (x==NULL || w1==NULL || w3==NULL || w2==NULL || gate_scratch==NULL || up_scratch==NULL ||
        out==NULL || dim==0 || expert_dim==0 || (weighted!=0 && weighted!=1) ||
        !isfinite(swiglu_limit) || swiglu_limit<0.0f || (weighted && !isfinite(route_weight)))
        return ELPIS_DSV41_NATIVE_INVALID;
    for (f=0;f<expert_dim;++f) {
        float gate=0.0f, up=0.0f;
        for (d=0;d<dim;++d) { gate += x[d]*w1[f*dim+d]; up += x[d]*w3[f*dim+d]; }
        if (swiglu_limit>0.0f) {
            if (gate>swiglu_limit) gate=swiglu_limit;
            if (up>swiglu_limit) up=swiglu_limit; else if (up<-swiglu_limit) up=-swiglu_limit;
        }
        gate=(gate*elpis_native_sigmoid(gate))*up;
        if (weighted) gate *= route_weight;
        gate_scratch[f]=gate;
        up_scratch[f]=up;
    }
    for (d=0;d<dim;++d) {
        float total=0.0f;
        for (f=0;f<expert_dim;++f) total += gate_scratch[f]*w2[d*expert_dim+f];
        out[d]=total;
    }
    return finite_vector(out,dim) ? ELPIS_DSV41_NATIVE_OK : ELPIS_DSV41_NATIVE_NONFINITE;
}


elpis_dsv41_native_status
elpis_dsv41_engram_gated_write_f32(const float *stream,
                                   const float *rows,
                                   const float *wkv,
                                   const float *q_weight,
                                   const float *k_weight,
                                   float norm_eps,
                                   float *kv_scratch,
                                   float *out,
                                   size_t copies,
                                   size_t dim,
                                   size_t row_values) {
    size_t kv_dim, o, i, c, d;
    float inv_dim_scale;

    if (stream == NULL || rows == NULL || wkv == NULL ||
        q_weight == NULL || k_weight == NULL ||
        kv_scratch == NULL || out == NULL ||
        copies == 0u || dim == 0u || row_values == 0u ||
        !isfinite(norm_eps) || norm_eps < 0.0f) {
        return ELPIS_DSV41_NATIVE_INVALID;
    }

    kv_dim = (copies + 1u) * dim;
    inv_dim_scale = 1.0f / sqrtf((float)dim);

    for (i = 0u; i < copies * dim; ++i) {
        if (!isfinite(stream[i]) ||
            !isfinite(q_weight[i]) ||
            !isfinite(k_weight[i])) {
            return ELPIS_DSV41_NATIVE_NONFINITE;
        }
    }
    for (i = 0u; i < row_values; ++i) {
        if (!isfinite(rows[i])) {
            return ELPIS_DSV41_NATIVE_NONFINITE;
        }
    }

    for (o = 0u; o < kv_dim; ++o) {
        float total = 0.0f;
        const float *row = wkv + o * row_values;
        for (i = 0u; i < row_values; ++i) {
            float weight = row[i];
            if (!isfinite(weight)) {
                return ELPIS_DSV41_NATIVE_NONFINITE;
            }
            total += rows[i] * weight;
        }
        kv_scratch[o] = total;
    }

    for (c = 0u; c < copies; ++c) {
        const float *stream_row = stream + c * dim;
        const float *key_row = kv_scratch + c * dim;
        const float *q_row = q_weight + c * dim;
        const float *k_row = k_weight + c * dim;
        const float *value = kv_scratch + copies * dim;
        float stream_ss = 0.0f;
        float key_ss = 0.0f;
        float dot = 0.0f;
        float rstd;
        float transformed;
        float gate;

        for (d = 0u; d < dim; ++d) {
            stream_ss += stream_row[d] * stream_row[d];
            key_ss += key_row[d] * key_row[d];
            dot += stream_row[d] * (q_row[d] * k_row[d]) * key_row[d];
        }

        rstd =
            (1.0f / sqrtf(stream_ss / (float)dim + norm_eps)) *
            (1.0f / sqrtf(key_ss / (float)dim + norm_eps));

        dot *= rstd * inv_dim_scale;
        if (!isfinite(dot)) {
            return ELPIS_DSV41_NATIVE_NONFINITE;
        }

        transformed = sqrtf(fmaxf(fabsf(dot), 1e-6f));
        transformed = copysignf(transformed, dot);
        gate = elpis_native_sigmoid(transformed);

        for (d = 0u; d < dim; ++d) {
            out[c * dim + d] = stream_row[d] + gate * value[d];
        }
    }

    return finite_vector(out, copies * dim) ?
           ELPIS_DSV41_NATIVE_OK : ELPIS_DSV41_NATIVE_NONFINITE;
}


elpis_dsv41_native_status
elpis_dsv41_index_scores_f32(const float *iq,
                             const float *index_keys,
                             const float *iw,
                             float *out,
                             size_t index_heads,
                             size_t index_dim,
                             size_t count) {
    size_t n, hidx, d;

    if (iq == NULL || index_keys == NULL || iw == NULL || out == NULL ||
        index_heads == 0u || index_dim == 0u) {
        return ELPIS_DSV41_NATIVE_INVALID;
    }

    for (hidx = 0u; hidx < index_heads; ++hidx) {
        if (!isfinite(iw[hidx])) {
            return ELPIS_DSV41_NATIVE_NONFINITE;
        }
        for (d = 0u; d < index_dim; ++d) {
            if (!isfinite(iq[hidx * index_dim + d])) {
                return ELPIS_DSV41_NATIVE_NONFINITE;
            }
        }
    }

    for (n = 0u; n < count; ++n) {
        float total = 0.0f;
        const float *key = index_keys + n * index_dim;

        for (hidx = 0u; hidx < index_heads; ++hidx) {
            float dot = 0.0f;
            const float *query = iq + hidx * index_dim;

            for (d = 0u; d < index_dim; ++d) {
                if (!isfinite(key[d])) {
                    return ELPIS_DSV41_NATIVE_NONFINITE;
                }
                dot += query[d] * key[d];
            }
            if (dot < 0.0f) {
                dot = 0.0f;
            }
            total += dot * iw[hidx];
        }
        out[n] = total;
    }

    return finite_vector(out, count) ?
           ELPIS_DSV41_NATIVE_OK : ELPIS_DSV41_NATIVE_NONFINITE;
}

elpis_dsv41_native_status
elpis_dsv41_grouped_output_f32(const float *attn_out,
                               const float *wo_a,
                               const float *wo_b,
                               float *scratch,
                               float *out,
                               size_t heads,
                               size_t head_dim,
                               size_t o_groups,
                               size_t o_rank,
                               size_t dimension) {
    size_t total_attn, group_width, g, r, i, d, flat_rank;

    if (attn_out == NULL || wo_a == NULL || wo_b == NULL ||
        scratch == NULL || out == NULL ||
        heads == 0u || head_dim == 0u || o_groups == 0u ||
        o_rank == 0u || dimension == 0u) {
        return ELPIS_DSV41_NATIVE_INVALID;
    }

    total_attn = heads * head_dim;
    if (total_attn % o_groups != 0u) {
        return ELPIS_DSV41_NATIVE_INVALID;
    }
    group_width = total_attn / o_groups;
    flat_rank = o_groups * o_rank;

    for (g = 0u; g < o_groups; ++g) {
        const float *group = attn_out + g * group_width;
        for (r = 0u; r < o_rank; ++r) {
            const float *proj = wo_a + (g * o_rank + r) * group_width;
            double total = 0.0;
            for (i = 0u; i < group_width; ++i) {
                if (!isfinite(group[i]) || !isfinite(proj[i])) {
                    return ELPIS_DSV41_NATIVE_NONFINITE;
                }
                total += (double)group[i] * (double)proj[i];
            }
            scratch[g * o_rank + r] = (float)total;
        }
    }

    for (d = 0u; d < dimension; ++d) {
        const float *row = wo_b + d * flat_rank;
        double total = 0.0;
        for (i = 0u; i < flat_rank; ++i) {
            if (!isfinite(row[i])) {
                return ELPIS_DSV41_NATIVE_NONFINITE;
            }
            total += (double)scratch[i] * (double)row[i];
        }
        out[d] = (float)total;
    }

    return finite_vector(out, dimension) ?
           ELPIS_DSV41_NATIVE_OK : ELPIS_DSV41_NATIVE_NONFINITE;
}
