/* Test-only scalar operators. Include after xenolith.c; no model state or forward. */
#ifndef XE_TEST_SCALAR_OPS_H
#define XE_TEST_SCALAR_OPS_H
static const float ref_kq_scale = 1.0f;
static const float ref_moe_weight_sum_min = 6.103515625e-5f;
static const float ref_gelu_coef_a = 0.044715f;
static const float ref_gelu_sqrt_2_over_pi = 0.79788456080286535587989211986876f;

static inline float ref_fp16_to_fp32(uint16_t bits) {
    uint32_t sign = (uint32_t)(bits >> 15) & 1u;
    uint32_t exponent = (uint32_t)(bits >> 10) & 0x1Fu;
    uint32_t mantissa = (uint32_t)bits & 0x3FFu;
    float magnitude;

    if (exponent == 0)
        magnitude = ldexpf((float)mantissa, -24);
    else if (exponent == 31)
        magnitude = mantissa ? NAN : INFINITY;
    else
        magnitude = ldexpf((float)(mantissa + 1024), (int)exponent - 25);

    return sign ? -magnitude : magnitude;
}

static inline void ref_dequant_block(const xe_q4 *w, uint64_t row, int blocks,
                              int block, float *out) {
    uint8_t nibbles[16];
    xe_q4_copy_block(w, row, blocks, block, nibbles);
    float scale = ref_fp16_to_fp32(xe_q4_scale(w, row, blocks, block));

    for (int j = 0; j < 16; j++) {
        int low = nibbles[j] & 0x0F;
        int high = (nibbles[j] >> 4) & 0x0F;
        out[j] = (float)(low - 8) * scale;
        out[j + 16] = (float)(high - 8) * scale;
    }
}

static inline void ref_dequant_row(const xe_q4 *w, uint64_t row, int n_in, float *out) {
    int n_blocks = n_in / 32;

    for (int b = 0; b < n_blocks; b++)
        ref_dequant_block(w, row, n_blocks, b, out + 32 * b);
}

#define REF_Q8_MAX_IN 8192

static inline float ref_fp16_round(float x) {
    return (float)(_Float16)x;
}

static inline void ref_quantize_q8(const float *x, int n_in, int8_t *qs, float *ds) {
    int n_blocks = n_in / 32;

    for (int b = 0; b < n_blocks; b++) {
        float amax = 0.0f;

        for (int j = 0; j < 32; j++) {
            float v = fabsf(x[32 * b + j]);
            if (v > amax) amax = v;
        }

        float d = amax / 127.0f;
        float id = d ? 1.0f / d : 0.0f;

        ds[b] = ref_fp16_round(d);

        for (int j = 0; j < 32; j++)
            qs[32 * b + j] = (int8_t)roundf(x[32 * b + j] * id);
    }
}

static inline int ref_dot_q4_q8_integer(const xe_q4 *w, uint64_t row, int blocks,
                                 int block, const int8_t *q) {
    uint8_t nibbles[16];
    xe_q4_copy_block(w, row, blocks, block, nibbles);
    int sumi0 = 0;
    int sumi1 = 0;

    for (int j = 0; j < 16; j++) {
        int v0 = (nibbles[j] & 0x0F) - 8;
        int v1 = (nibbles[j] >> 4) - 8;
        sumi0 += v0 * q[j];
        sumi1 += v1 * q[j + 16];
    }
    return sumi0 + sumi1;
}

static inline float ref_dot_q4_q8_row(const xe_q4 *w, uint64_t row, int n_in, const int8_t *qs, const float *ds) {
    int n_blocks = n_in / 32;
    float sumf = 0.0f;

    for (int b = 0; b < n_blocks; b++) {
        int integer = ref_dot_q4_q8_integer(w, row, n_blocks, b, qs + 32 * b);
        sumf += (float)integer * ref_fp16_to_fp32(xe_q4_scale(w, row, n_blocks, b)) * ds[b];
    }
    return sumf;
}

static inline float ref_dot_q4_q8_row_v1(const xe_q4 *w, uint64_t row, int n_in,
                                   const int8_t *qs, const float *ds) {
    int n_blocks = n_in / 32;
    float sums[8] = { 0.0f };
    int b = 0;

    for (; b + 7 < n_blocks; b += 8) {
        for (int j = 0; j < 8; j++) {
            int integer = ref_dot_q4_q8_integer(w, row, n_blocks, b + j,
                                                qs + 32 * (b + j));
            float scale = ref_fp16_to_fp32(xe_q4_scale(w, row, n_blocks, b + j)) *
                          ds[b + j];
            sums[j] += (float)integer * scale;
        }
    }

    float lo0 = sums[0] + sums[4];
    float lo1 = sums[1] + sums[5];
    float lo2 = sums[2] + sums[6];
    float lo3 = sums[3] + sums[7];
    float result = (lo0 + lo1) + (lo2 + lo3);
    for (; b < n_blocks; b++) {
        int integer = ref_dot_q4_q8_integer(w, row, n_blocks, b, qs + 32 * b);
        result += (float)integer * ref_fp16_to_fp32(xe_q4_scale(w, row, n_blocks, b)) * ds[b];
    }
    return result;
}

static inline void ref_matvec_q4_q8_packed(const xe_q4 *w, uint64_t first_row, int n_in,
                                    int n_out, const int8_t *qs, const float *ds,
                                    int v1_order, float *y) {
    for (int r = 0; r < n_out; r++) {
        uint64_t row = first_row + (uint64_t)r;
        y[r] = v1_order ? ref_dot_q4_q8_row_v1(w, row, n_in, qs, ds) :
                          ref_dot_q4_q8_row(w, row, n_in, qs, ds);
    }
}

static inline void ref_matvec_q4_q8(const xe_q4 *w, uint64_t first_row, int n_in, int n_out,
                             const float *x, int v1_order, float *y) {
    int8_t qs[REF_Q8_MAX_IN];
    float ds[REF_Q8_MAX_IN / 32];

    if (n_in > REF_Q8_MAX_IN) xe_fatal("q8 matvec: n_in %d exceeds %d", n_in, REF_Q8_MAX_IN);

    ref_quantize_q8(x, n_in, qs, ds);
    ref_matvec_q4_q8_packed(w, first_row, n_in, n_out, qs, ds, v1_order, y);
}

static inline float ref_dot_f32(const float *a, const float *b, int n) {
    float sum = 0.0f;

    for (int i = 0; i < n; i++)
        sum += a[i] * b[i];
    return sum;
}

static inline void ref_matvec_f32(const float *w, int n_in, int n_out, const float *x, float *y) {
    for (int r = 0; r < n_out; r++)
        y[r] = ref_dot_f32(w + (size_t)r * (size_t)n_in, x, n_in);
}

static inline void ref_rmsnorm(const float *x, const float *weight, int n, float *y) {
    float sum = 0.0f;

    for (int i = 0; i < n; i++)
        sum += x[i] * x[i];

    float scale = 1.0f / sqrtf(sum / (float)n + XE_RMS_EPS);

    for (int i = 0; i < n; i++)
        y[i] = x[i] * scale * weight[i];
}

static inline void ref_rmsnorm_weightless(const float *x, int n, float *y) {
    float sum = 0.0f;

    for (int i = 0; i < n; i++)
        sum += x[i] * x[i];

    float scale = 1.0f / sqrtf(sum / (float)n + XE_RMS_EPS);

    for (int i = 0; i < n; i++)
        y[i] = x[i] * scale;
}

static inline void ref_softmax(float *x, int n) {
    float max = x[0];

    for (int i = 1; i < n; i++)
        if (x[i] > max) max = x[i];

    float sum = 0.0f;
    for (int i = 0; i < n; i++) {
        x[i] = expf(x[i] - max);
        sum += x[i];
    }

    for (int i = 0; i < n; i++)
        x[i] /= sum;
}

static inline void ref_top_k(const float *values, int n, int k, int *out_idx, float *out_val) {
    for (int i = 0; i < k; i++) {
        out_idx[i] = -1;
        out_val[i] = -INFINITY;
    }

    for (int i = 0; i < n; i++) {
        if (values[i] <= out_val[k - 1]) continue;

        int slot = k - 1;
        while (slot > 0 && out_val[slot - 1] < values[i]) {
            out_val[slot] = out_val[slot - 1];
            out_idx[slot] = out_idx[slot - 1];
            slot--;
        }
        out_val[slot] = values[i];
        out_idx[slot] = i;
    }
}

static inline float ref_gelu(float x) {
    float inner = ref_gelu_sqrt_2_over_pi * (x + ref_gelu_coef_a * x * x * x);
    return 0.5f * x * (1.0f + tanhf(inner));
}

static inline float ref_gelu_fp16(float x) {
    if (x <= -10.0f) return 0.0f;
    if (x >= 10.0f) return x;
    return (float)(_Float16)ref_gelu((float)(_Float16)x);
}

static inline void ref_attention(const float *q, const float *k_cache, const float *v_cache, int pos,
                          int n_kv_heads, int head_dim, int sliding_window, float *scores, float *out) {
    int group = XE_Q_HEADS / n_kv_heads;
    int n_keys = pos + 1;

    for (int h = 0; h < XE_Q_HEADS; h++) {
        const float *q_head = q + (size_t)h * (size_t)head_dim;
        int kv_head = h / group;

        for (int p = 0; p < n_keys; p++) {
            if (sliding_window > 0 && pos - p >= sliding_window) {
                scores[p] = -INFINITY;
                continue;
            }
            const float *k_head = k_cache + ((size_t)p * (size_t)n_kv_heads + (size_t)kv_head) * (size_t)head_dim;
            scores[p] = ref_dot_f32(q_head, k_head, head_dim) * ref_kq_scale;
        }

        ref_softmax(scores, n_keys);

        float *out_head = out + (size_t)h * (size_t)head_dim;
        for (int d = 0; d < head_dim; d++)
            out_head[d] = 0.0f;

        for (int p = 0; p < n_keys; p++) {
            const float *v_head = v_cache + ((size_t)p * (size_t)n_kv_heads + (size_t)kv_head) * (size_t)head_dim;
            for (int d = 0; d < head_dim; d++)
                out_head[d] += scores[p] * v_head[d];
        }
    }
}

#endif
