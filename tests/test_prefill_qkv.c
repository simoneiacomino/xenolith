#include "../xenolith.c"

static double qkv_rel_f32(const float *a, const float *b, size_t n) {
    double error = 0.0;
    double reference = 0.0;
    for (size_t i = 0; i < n; i++) {
        double difference = (double)a[i] - b[i];
        error += difference * difference;
        reference += (double)b[i] * b[i];
    }
    return sqrt(error / (reference + 1e-30));
}

static double qkv_rel_f16(const _Float16 *a, const _Float16 *b, size_t n,
                          int *mismatches, int *max_ulp) {
    double error = 0.0;
    double reference = 0.0;
    *mismatches = 0;
    *max_ulp = 0;
    for (size_t i = 0; i < n; i++) {
        double got = (double)a[i];
        double want = (double)b[i];
        double difference = got - want;
        error += difference * difference;
        reference += want * want;
        *mismatches += a[i] != b[i];
        uint16_t ua;
        uint16_t ub;
        memcpy(&ua, a + i, sizeof ua);
        memcpy(&ub, b + i, sizeof ub);
        int oa = ua & 0x8000 ? 0x8000 - (ua & 0x7fff) : 0x8000 + ua;
        int ob = ub & 0x8000 ? 0x8000 - (ub & 0x7fff) : 0x8000 + ub;
        int ulp = abs(oa - ob);
        if (ulp > *max_ulp) *max_ulp = ulp;
    }
    return sqrt(error / (reference + 1e-30));
}

static void qkv_reference(const xe_engine *e, const xe_layer *layer,
                          const float *q_projection,
                          const float *k_projection,
                          const float *v_projection,
                          const float *rope_cos, const float *rope_sin,
                          float *q, _Float16 *k, _Float16 *v,
                          int rows, int dimension, int kv_heads, int has_v) {
    int half = dimension / 2;
    for (int row = 0; row < rows; row++) {
        for (int head = 0; head < XE_Q_HEADS; head++) {
            const float *source = q_projection
                + ((size_t)row * XE_Q_HEADS + head) * dimension;
            float scale = xe_rms_scale_scalar(source, dimension);
            size_t base = ((size_t)head * rows + row) * dimension;
            for (int d = 0; d < half; d++) {
                float lo = source[d] * scale * layer->q_norm[d];
                float hi = source[d + half] * scale
                           * layer->q_norm[d + half];
                float cosine = rope_cos[(size_t)row * half + d];
                float sine = rope_sin[(size_t)row * half + d];
                q[base + d] = lo * cosine - hi * sine;
                q[base + d + half] = lo * sine + hi * cosine;
            }
        }
        for (int head = 0; head < kv_heads; head++) {
            const float *ks = k_projection
                + ((size_t)row * kv_heads + head) * dimension;
            const float *vs = has_v ? v_projection
                + ((size_t)row * kv_heads + head) * dimension : ks;
            float k_scale = xe_rms_scale_scalar(ks, dimension);
            float v_scale = xe_rms_scale_scalar(vs, dimension);
            size_t base = ((size_t)head * rows + row) * dimension;
            for (int d = 0; d < half; d++) {
                float lo = ks[d] * k_scale * layer->k_norm[d];
                float hi = ks[d + half] * k_scale
                           * layer->k_norm[d + half];
                float cosine = rope_cos[(size_t)row * half + d];
                float sine = rope_sin[(size_t)row * half + d];
                k[base + d] = (_Float16)(lo * cosine - hi * sine);
                k[base + d + half] = (_Float16)(lo * sine + hi * cosine);
            }
            for (int d = 0; d < dimension; d++)
                v[base + d] = (_Float16)(vs[d] * v_scale);
        }
    }
    (void)e;
}

static double qkv_attention_error(const float *q, const _Float16 *k,
                                  const _Float16 *v, const float *output,
                                  int rows, int dimension, int kv_heads) {
    int queries[4] = { 0, rows / 3, 2 * rows / 3, rows - 1 };
    double error = 0.0;
    double reference = 0.0;
    double *scores = xe_alloc(NULL, (size_t)rows * sizeof(*scores), XE_MEM_HOST);
    for (int sample = 0; sample < 4; sample++) {
        int query = queries[sample];
        for (int head = 0; head < XE_Q_HEADS; head++) {
            int kv_head = head * kv_heads / XE_Q_HEADS;
            const float *qh = q
                + ((size_t)head * rows + query) * dimension;
            double maximum = -INFINITY;
            for (int key = 0; key <= query; key++) {
                const _Float16 *kh = k
                    + ((size_t)kv_head * rows + key) * dimension;
                double score = 0.0;
                for (int d = 0; d < dimension; d++)
                    score += (double)qh[d] * (double)kh[d];
                scores[key] = score;
                if (score > maximum) maximum = score;
            }
            double denominator = 0.0;
            for (int key = 0; key <= query; key++) {
                scores[key] = exp(scores[key] - maximum);
                denominator += scores[key];
            }
            const float *got = output
                + ((size_t)head * rows + query) * dimension;
            for (int d = 0; d < dimension; d++) {
                double want = 0.0;
                for (int key = 0; key <= query; key++)
                    want += scores[key]
                            * (double)v[((size_t)kv_head * rows + key)
                                        * dimension + d];
                want /= denominator;
                double difference = (double)got[d] - want;
                error += difference * difference;
                reference += want * want;
            }
        }
    }
    xe_free(NULL, scores, XE_MEM_HOST);
    return sqrt(error / (reference + 1e-30));
}

static int qkv_run(xe_engine *e, int layer_index, int rows) {
    const xe_layer *layer = &e->layers[layer_index];
    int global = XE_IS_GLOBAL(layer_index);
    int dimension = global ? XE_GLOBAL_HEAD_DIM : XE_SWA_HEAD_DIM;
    int kv_heads = global ? XE_GLOBAL_KV_HEADS : XE_SWA_KV_HEADS;
    int q_width = XE_Q_HEADS * dimension;
    int kv_width = kv_heads * dimension;
    int blocks = XE_EMBD / 32;
    size_t input_n = (size_t)rows * XE_EMBD;
    size_t q_n = (size_t)rows * q_width;
    size_t kv_n = (size_t)rows * kv_width;
    size_t rope_n = (size_t)rows * dimension / 2;
    float *input = xe_alloc(e, input_n * sizeof(*input), XE_MEM_SHARED);
    float *scale = xe_alloc(e, rows * sizeof(*scale), XE_MEM_SHARED);
    xe_q8 input_q8 = {
        xe_alloc(e, input_n, XE_MEM_SHARED),
        xe_alloc(e, (size_t)rows * blocks * sizeof(*input_q8.d), XE_MEM_SHARED),
        xe_alloc(e, (size_t)rows * blocks * sizeof(*input_q8.sigma), XE_MEM_SHARED),
        (int)input_n
    };
    float *qp = xe_alloc(e, q_n * sizeof(*qp), XE_MEM_SHARED);
    float *kp = xe_alloc(e, kv_n * sizeof(*kp), XE_MEM_SHARED);
    float *vp = xe_alloc(e, kv_n * sizeof(*vp), XE_MEM_SHARED);
    float *rope_cos = xe_alloc(e, rope_n * sizeof(*rope_cos), XE_MEM_SHARED);
    float *rope_sin = xe_alloc(e, rope_n * sizeof(*rope_sin), XE_MEM_SHARED);
    float *q = xe_alloc(e, q_n * sizeof(*q), XE_MEM_SHARED);
    _Float16 *k = xe_alloc(e, kv_n * sizeof(*k), XE_MEM_SHARED);
    _Float16 *v = xe_alloc(e, kv_n * sizeof(*v), XE_MEM_SHARED);
    float *attention = xe_alloc(e, q_n * sizeof(*attention), XE_MEM_SHARED);
    int head_blocks = q_width / 32;
    xe_q8 attention_q8 = {
        xe_alloc(e, q_n, XE_MEM_SHARED),
        xe_alloc(e, (size_t)rows * head_blocks * sizeof(*attention_q8.d),
                 XE_MEM_SHARED),
        xe_alloc(e, (size_t)rows * head_blocks * sizeof(*attention_q8.sigma),
                 XE_MEM_SHARED),
        (int)q_n
    };
    float *attention_projection = xe_alloc(
        e, input_n * sizeof(*attention_projection), XE_MEM_SHARED);
    float *attention_output = xe_alloc(
        e, input_n * sizeof(*attention_output), XE_MEM_SHARED);
    float *q_reference = xe_alloc(NULL, q_n * sizeof(*q_reference), XE_MEM_HOST);
    _Float16 *k_reference = xe_alloc(NULL, kv_n * sizeof(*k_reference), XE_MEM_HOST);
    _Float16 *v_reference = xe_alloc(NULL, kv_n * sizeof(*v_reference), XE_MEM_HOST);
    int8_t *attention_reference_qs = xe_alloc(NULL, q_n, XE_MEM_HOST);
    _Float16 *attention_reference_d = xe_alloc(
        NULL, (size_t)rows * head_blocks * sizeof(*attention_reference_d),
        XE_MEM_HOST);
    int16_t *attention_reference_sigma = xe_alloc(
        NULL, (size_t)rows * head_blocks * sizeof(*attention_reference_sigma),
        XE_MEM_HOST);
    float *attention_row = xe_alloc(NULL, q_width * sizeof(*attention_row),
                                    XE_MEM_HOST);
    float *output_reference = xe_alloc(NULL, input_n * sizeof(*output_reference),
                                       XE_MEM_HOST);
    for (int row = 0; row < rows; row++) {
        xe_embed_decode(e, 2 + row, input + (size_t)row * XE_EMBD);
        for (int d = 0; d < dimension / 2; d++) {
            float inverse = global ? e->rope_global_inv[d] : e->rope_swa_inv[d];
            float theta = (float)row * inverse;
            size_t index = (size_t)row * dimension / 2 + d;
            rope_cos[index] = cosf(theta);
            rope_sin[index] = sinf(theta);
        }
    }
    xe_prefill_input_append(e, input, layer->attn_norm, scale, &input_q8,
                            rows, XE_EMBD);
    xe_prefill_projection_append(e, &layer->attn_q, &input_q8, qp,
                                 rows, q_width, rows > 32);
    xe_prefill_projection_append(e, &layer->attn_k, &input_q8, kp,
                                 rows, kv_width, global && rows > 32);
    if (!global)
        xe_prefill_projection_append(e, &layer->attn_v, &input_q8, vp,
                                     rows, kv_width, 0);
    xe_prefill_qkv_append(e, qp, kp, global ? kp : vp, layer->q_norm,
                          layer->k_norm, rope_cos, rope_sin, q, k, v,
                          rows, dimension, kv_heads, !global);
    xe_prefill_attention_online_append(e, q, k, v, attention, rows, rows,
                                        dimension, kv_heads, 0,
                                        global ? 0 : XE_SWA_WINDOW, XE_ATTENTION_CAUSAL);
    xe_prefill_heads_q8_append(e, attention, &attention_q8, rows, dimension);
    xe_prefill_projection_append(e, &layer->attn_o, &attention_q8,
                                 attention_projection, rows, XE_EMBD,
                                 rows > 32);
    xe_prefill_rms_residual_append(e, attention_projection,
                                    layer->post_attn_norm, input,
                                    attention_output, rows, XE_EMBD);
    xe_ze_check("zeCommandListHostSynchronize prefill qkv test",
                zeCommandListHostSynchronize(e->gpu.commands, UINT64_MAX));
    struct timespec start;
    struct timespec finish;
    const int repetitions = 30;
    clock_gettime(CLOCK_MONOTONIC, &start);
    for (int repetition = 0; repetition < repetitions; repetition++)
        xe_prefill_qkv_append(e, qp, kp, global ? kp : vp, layer->q_norm,
                              layer->k_norm, rope_cos, rope_sin, q, k, v,
                              rows, dimension, kv_heads, !global);
    xe_ze_check("zeCommandListHostSynchronize prefill qkv timing",
                zeCommandListHostSynchronize(e->gpu.commands, UINT64_MAX));
    clock_gettime(CLOCK_MONOTONIC, &finish);
    double elapsed = (double)(finish.tv_sec - start.tv_sec)
                     + (double)(finish.tv_nsec - start.tv_nsec) * 1e-9;
    const int attention_repetitions = 10;
    clock_gettime(CLOCK_MONOTONIC, &start);
    for (int repetition = 0; repetition < attention_repetitions; repetition++)
        xe_prefill_attention_online_append(e, q, k, v, attention, rows, rows,
                                            dimension, kv_heads, 0,
                                            global ? 0 : XE_SWA_WINDOW, XE_ATTENTION_CAUSAL);
    xe_ze_check("zeCommandListHostSynchronize prefill attention timing",
                zeCommandListHostSynchronize(e->gpu.commands, UINT64_MAX));
    clock_gettime(CLOCK_MONOTONIC, &finish);
    double attention_elapsed = (double)(finish.tv_sec - start.tv_sec)
                               + (double)(finish.tv_nsec - start.tv_nsec)
                                 * 1e-9;
    const int tail_repetitions = 10;
    clock_gettime(CLOCK_MONOTONIC, &start);
    for (int repetition = 0; repetition < tail_repetitions; repetition++) {
        xe_prefill_heads_q8_append(e, attention, &attention_q8, rows,
                                   dimension);
        xe_prefill_projection_append(e, &layer->attn_o, &attention_q8,
                                     attention_projection, rows, XE_EMBD,
                                     rows > 32);
        xe_prefill_rms_residual_append(e, attention_projection,
                                        layer->post_attn_norm, input,
                                        attention_output, rows, XE_EMBD);
    }
    xe_ze_check("zeCommandListHostSynchronize prefill attention tail timing",
                zeCommandListHostSynchronize(e->gpu.commands, UINT64_MAX));
    clock_gettime(CLOCK_MONOTONIC, &finish);
    double tail_elapsed = (double)(finish.tv_sec - start.tv_sec)
                          + (double)(finish.tv_nsec - start.tv_nsec) * 1e-9;
    qkv_reference(e, layer, qp, kp, global ? kp : vp, rope_cos, rope_sin,
                  q_reference, k_reference, v_reference,
                  rows, dimension, kv_heads, !global);
    int k_mismatches;
    int v_mismatches;
    int k_max_ulp;
    int v_max_ulp;
    double q_error = qkv_rel_f32(q, q_reference, q_n);
    double k_error = qkv_rel_f16(k, k_reference, kv_n, &k_mismatches,
                                 &k_max_ulp);
    double v_error = qkv_rel_f16(v, v_reference, kv_n, &v_mismatches,
                                 &v_max_ulp);
    double attention_error = qkv_attention_error(q, k, v, attention, rows,
                                                  dimension, kv_heads);
    int attention_q_mismatch = 0;
    int attention_d_mismatch = 0;
    int attention_sigma_mismatch = 0;
    int attention_max_q_delta = 0;
    for (int row = 0; row < rows; row++) {
        for (int head = 0; head < XE_Q_HEADS; head++)
            memcpy(attention_row + (size_t)head * dimension,
                   attention + ((size_t)head * rows + row) * dimension,
                   (size_t)dimension * sizeof(*attention_row));
        xe_q8 reference_q8 = {
            attention_reference_qs + (size_t)row * q_width,
            attention_reference_d + (size_t)row * head_blocks,
            attention_reference_sigma + (size_t)row * head_blocks,
            q_width
        };
        xe_quantize_q8(attention_row, q_width, &reference_q8);
        for (int column = 0; column < q_width; column++) {
            size_t index = (size_t)row * q_width + column;
            int delta = abs((int)attention_q8.qs[index]
                            - (int)reference_q8.qs[column]);
            attention_q_mismatch += delta != 0;
            if (delta > attention_max_q_delta) attention_max_q_delta = delta;
        }
        for (int block = 0; block < head_blocks; block++) {
            size_t index = (size_t)row * head_blocks + block;
            attention_d_mismatch += attention_q8.d[index]
                                    != reference_q8.d[block];
            attention_sigma_mismatch += attention_q8.sigma[index]
                                        != reference_q8.sigma[block];
        }
        float rms = xe_rms_scale_scalar(
            attention_projection + (size_t)row * XE_EMBD, XE_EMBD);
        for (int column = 0; column < XE_EMBD; column++) {
            size_t index = (size_t)row * XE_EMBD + column;
            output_reference[index] = attention_projection[index] * rms
                                      * layer->post_attn_norm[column]
                                      + input[index];
        }
    }
    float projection_gpu[176];
    float projection_cpu[176];
    int projection_samples = 0;
    int sample_rows = rows < 8 ? rows : 8;
    for (int row = 0; row < sample_rows; row++) {
        xe_q8 row_q8 = {
            attention_q8.qs + (size_t)row * q_width,
            attention_q8.d + (size_t)row * head_blocks,
            attention_q8.sigma + (size_t)row * head_blocks,
            q_width
        };
        for (int column = 0; column < XE_EMBD; column += 128) {
            projection_gpu[projection_samples] =
                attention_projection[(size_t)row * XE_EMBD + column];
            projection_cpu[projection_samples] = xe_q4_q8_dot(
                &layer->attn_o, column, q_width, &row_q8);
            projection_samples++;
        }
    }
    double output_projection_error = qkv_rel_f32(
        projection_gpu, projection_cpu, projection_samples);
    double output_error = qkv_rel_f32(attention_output, output_reference,
                                      input_n);
    int ok = q_error < 3e-6 && k_error < 2e-5 && v_error < 2e-5
             && k_max_ulp <= 1 && v_max_ulp <= 1
             && attention_error < 2e-5 && attention_max_q_delta <= 1
             && output_projection_error < 2e-5 && output_error < 3e-6;
    printf("prefill-qkv: layer %d %s M%d qkv %.6f ms attention %.6f ms tail %.6f ms q %.3e k %.3e (%d/%zu ulp %d) v %.3e (%d/%zu ulp %d) attention-error %.3e heads-q8 %d/%d/%d delta %d o-projection %.3e residual %.3e %s\n",
           layer_index, global ? "global" : "swa", rows,
           elapsed * 1000.0 / repetitions,
           attention_elapsed * 1000.0 / attention_repetitions,
           tail_elapsed * 1000.0 / tail_repetitions, q_error,
           k_error, k_mismatches, kv_n, k_max_ulp,
           v_error, v_mismatches, kv_n, v_max_ulp, attention_error,
           attention_q_mismatch, attention_d_mismatch,
           attention_sigma_mismatch, attention_max_q_delta,
           output_projection_error, output_error,
           ok ? "PASS" : "FAIL");
    xe_free(NULL, output_reference, XE_MEM_HOST);
    xe_free(NULL, attention_row, XE_MEM_HOST);
    xe_free(NULL, attention_reference_sigma, XE_MEM_HOST);
    xe_free(NULL, attention_reference_d, XE_MEM_HOST);
    xe_free(NULL, attention_reference_qs, XE_MEM_HOST);
    xe_free(NULL, v_reference, XE_MEM_HOST);
    xe_free(NULL, k_reference, XE_MEM_HOST);
    xe_free(NULL, q_reference, XE_MEM_HOST);
    xe_free(e, attention_output, XE_MEM_SHARED);
    xe_free(e, attention_projection, XE_MEM_SHARED);
    xe_free(e, attention_q8.sigma, XE_MEM_SHARED);
    xe_free(e, attention_q8.d, XE_MEM_SHARED);
    xe_free(e, attention_q8.qs, XE_MEM_SHARED);
    xe_free(e, attention, XE_MEM_SHARED);
    xe_free(e, v, XE_MEM_SHARED);
    xe_free(e, k, XE_MEM_SHARED);
    xe_free(e, q, XE_MEM_SHARED);
    xe_free(e, rope_sin, XE_MEM_SHARED);
    xe_free(e, rope_cos, XE_MEM_SHARED);
    xe_free(e, vp, XE_MEM_SHARED);
    xe_free(e, kp, XE_MEM_SHARED);
    xe_free(e, qp, XE_MEM_SHARED);
    xe_free(e, input_q8.sigma, XE_MEM_SHARED);
    xe_free(e, input_q8.d, XE_MEM_SHARED);
    xe_free(e, input_q8.qs, XE_MEM_SHARED);
    xe_free(e, scale, XE_MEM_SHARED);
    xe_free(e, input, XE_MEM_SHARED);
    return ok;
}

int main(int argc, char **argv) {
    if (argc < 2 || !argv[1][0]) {
        fprintf(stderr, "usage: %s <model.gguf> [rows]\n", argv[0]);
        return 2;
    }
    const char *model = argv[1];
    int rows = argc > 2 ? (int)strtol(argv[2], NULL, 10) : 32;
    if (rows < 1 || rows > 512) xe_fatal("prefill qkv rows out of range");
    xe_engine *e = xe_engine_open(model);
    int ok = qkv_run(e, 0, rows) & qkv_run(e, 5, rows);
    xe_engine_close(e);
    return ok ? 0 : 1;
}
