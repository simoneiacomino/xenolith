#include "../xenolith.c"

static double swa_now(void) {
    struct timespec time;
    clock_gettime(CLOCK_MONOTONIC, &time);
    return (double)time.tv_sec + (double)time.tv_nsec * 1e-9;
}

static double swa_attention_error(const float *q, const _Float16 *k,
                                  const _Float16 *v, const float *output,
                                  int rows, int keys, int query_offset) {
    int queries[4] = { 0, rows / 3, 2 * rows / 3, rows - 1 };
    double error = 0.0;
    double reference = 0.0;
    double *scores = xe_alloc(NULL, (size_t)keys * sizeof(*scores), XE_MEM_HOST);
    for (int sample = 0; sample < 4; sample++) {
        int query = queries[sample];
        int position = query_offset + query;
        int first = position - XE_SWA_WINDOW + 1;
        if (first < 0) first = 0;
        for (int head = 0; head < XE_Q_HEADS; head++) {
            int kv_head = head * XE_SWA_KV_HEADS / XE_Q_HEADS;
            const float *qh = q
                + ((size_t)head * rows + query) * XE_SWA_HEAD_DIM;
            double maximum = -INFINITY;
            for (int key = first; key <= position; key++) {
                const _Float16 *kh = k
                    + ((size_t)kv_head * keys + key) * XE_SWA_HEAD_DIM;
                double score = 0.0;
                for (int d = 0; d < XE_SWA_HEAD_DIM; d++)
                    score += (double)qh[d] * (double)kh[d];
                scores[key] = score;
                if (score > maximum) maximum = score;
            }
            double denominator = 0.0;
            for (int key = first; key <= position; key++) {
                scores[key] = exp(scores[key] - maximum);
                denominator += scores[key];
            }
            const float *got = output
                + ((size_t)head * rows + query) * XE_SWA_HEAD_DIM;
            for (int d = 0; d < XE_SWA_HEAD_DIM; d++) {
                double want = 0.0;
                for (int key = first; key <= position; key++)
                    want += scores[key]
                            * (double)v[((size_t)kv_head * keys + key)
                                        * XE_SWA_HEAD_DIM + d];
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

int main(int argc, char **argv) {
    if (argc < 2 || !argv[1][0]) {
        fprintf(stderr, "usage: %s <model.gguf>\n", argv[0]);
        return 2;
    }
    const char *model = argv[1];
    const int rows = 512;
    const int batch_start = 900;
    const int stage_base = 0;
    const int stage_count = batch_start + rows - stage_base;
    const size_t ring_elements = (size_t)XE_SWA_KV_HEADS * XE_SWA_WINDOW
                                 * XE_SWA_HEAD_DIM;
    const size_t batch_elements = (size_t)XE_SWA_KV_HEADS * rows
                                  * XE_SWA_HEAD_DIM;
    const size_t stage_elements = (size_t)XE_SWA_KV_HEADS * stage_count
                                  * XE_SWA_HEAD_DIM;
    const size_t q_elements = (size_t)XE_Q_HEADS * rows * XE_SWA_HEAD_DIM;
    xe_engine *e = xe_engine_open(model);
    _Float16 *ring_k = xe_alloc(e, ring_elements * sizeof(*ring_k), XE_MEM_SHARED);
    _Float16 *ring_v = xe_alloc(e, ring_elements * sizeof(*ring_v), XE_MEM_SHARED);
    _Float16 *batch_k = xe_alloc(e, batch_elements * sizeof(*batch_k), XE_MEM_SHARED);
    _Float16 *batch_v = xe_alloc(e, batch_elements * sizeof(*batch_v), XE_MEM_SHARED);
    _Float16 *stage_k = xe_alloc(e, stage_elements * sizeof(*stage_k), XE_MEM_SHARED);
    _Float16 *stage_v = xe_alloc(e, stage_elements * sizeof(*stage_v), XE_MEM_SHARED);
    float *q = xe_alloc(e, q_elements * sizeof(*q), XE_MEM_SHARED);
    float *output = xe_alloc(e, q_elements * sizeof(*output), XE_MEM_SHARED);
    for (size_t i = 0; i < ring_elements; i++) {
        ring_k[i] = (_Float16)(((int)(i * 17 % 1009) - 504) / 8192.0f);
        ring_v[i] = (_Float16)(((int)(i * 29 % 1013) - 506) / 512.0f);
    }
    for (size_t i = 0; i < batch_elements; i++) {
        batch_k[i] = (_Float16)(((int)(i * 31 % 1019) - 509) / 8192.0f);
        batch_v[i] = (_Float16)(((int)(i * 43 % 1021) - 510) / 512.0f);
    }
    for (size_t i = 0; i < q_elements; i++)
        q[i] = ((int)(i * 47 % 1021) - 510) / 8192.0f;
    xe_prefill_swa_stage_append(e, ring_k, ring_v, batch_k, batch_v,
                                stage_k, stage_v, rows, batch_start,
                                stage_base, stage_count);
    xe_prefill_attention_online_append(e, q, stage_k, stage_v, output,
                                        rows, stage_count, XE_SWA_HEAD_DIM,
                                        XE_SWA_KV_HEADS,
                                        batch_start - stage_base,
                                        XE_SWA_WINDOW, XE_ATTENTION_CAUSAL);
    xe_ze_check("zeCommandListHostSynchronize prefill swa stage attention",
                zeCommandListHostSynchronize(e->gpu.commands, UINT64_MAX));
    int stage_mismatches = 0;
    for (int head = 0; head < XE_SWA_KV_HEADS; head++)
        for (int key = stage_base; key < batch_start + rows; key++) {
            const _Float16 *expected_k = key < batch_start
                ? ring_k + ((size_t)head * XE_SWA_WINDOW
                            + (key & (XE_SWA_WINDOW - 1))) * XE_SWA_HEAD_DIM
                : batch_k + ((size_t)head * rows + key - batch_start)
                            * XE_SWA_HEAD_DIM;
            const _Float16 *expected_v = key < batch_start
                ? ring_v + ((size_t)head * XE_SWA_WINDOW
                            + (key & (XE_SWA_WINDOW - 1))) * XE_SWA_HEAD_DIM
                : batch_v + ((size_t)head * rows + key - batch_start)
                            * XE_SWA_HEAD_DIM;
            size_t target = ((size_t)head * stage_count + key - stage_base)
                            * XE_SWA_HEAD_DIM;
            stage_mismatches += memcmp(stage_k + target, expected_k,
                                       XE_SWA_HEAD_DIM * sizeof(*stage_k)) != 0;
            stage_mismatches += memcmp(stage_v + target, expected_v,
                                       XE_SWA_HEAD_DIM * sizeof(*stage_v)) != 0;
        }
    double attention_error = swa_attention_error(
        q, stage_k, stage_v, output, rows, stage_count,
        batch_start - stage_base);
    const int repetitions = 30;
    double start = swa_now();
    for (int repetition = 0; repetition < repetitions; repetition++)
        xe_prefill_swa_stage_append(e, ring_k, ring_v, batch_k, batch_v,
                                    stage_k, stage_v, rows, batch_start,
                                    stage_base, stage_count);
    xe_ze_check("zeCommandListHostSynchronize prefill swa stage timing",
                zeCommandListHostSynchronize(e->gpu.commands, UINT64_MAX));
    double stage_ms = (swa_now() - start) * 1000.0 / repetitions;
    start = swa_now();
    for (int repetition = 0; repetition < repetitions; repetition++)
        xe_prefill_swa_commit_append(e, ring_k, ring_v, batch_k, batch_v,
                                     rows, batch_start, XE_SWA_HEAD_DIM,
                                     XE_SWA_KV_HEADS, XE_SWA_WINDOW);
    xe_ze_check("zeCommandListHostSynchronize prefill swa commit timing",
                zeCommandListHostSynchronize(e->gpu.commands, UINT64_MAX));
    double commit_ms = (swa_now() - start) * 1000.0 / repetitions;
    int commit_mismatches = 0;
    for (int head = 0; head < XE_SWA_KV_HEADS; head++)
        for (int query = 0; query < rows; query++) {
            size_t source = ((size_t)head * rows + query) * XE_SWA_HEAD_DIM;
            size_t target = ((size_t)head * XE_SWA_WINDOW
                             + ((batch_start + query) & (XE_SWA_WINDOW - 1)))
                            * XE_SWA_HEAD_DIM;
            commit_mismatches += memcmp(ring_k + target, batch_k + source,
                                        XE_SWA_HEAD_DIM * sizeof(*ring_k)) != 0;
            commit_mismatches += memcmp(ring_v + target, batch_v + source,
                                        XE_SWA_HEAD_DIM * sizeof(*ring_v)) != 0;
        }
    int ok = stage_mismatches == 0 && commit_mismatches == 0
             && attention_error < 2e-5;
    printf("prefill-swa: start %d M%d stage-count %d stage %.6f ms commit %.6f ms stage-mismatch %d commit-mismatch %d attention %.3e %s\n",
           batch_start, rows, stage_count, stage_ms, commit_ms,
           stage_mismatches, commit_mismatches, attention_error,
           ok ? "PASS" : "FAIL");
    xe_free(e, output, XE_MEM_SHARED);
    xe_free(e, q, XE_MEM_SHARED);
    xe_free(e, stage_v, XE_MEM_SHARED);
    xe_free(e, stage_k, XE_MEM_SHARED);
    xe_free(e, batch_v, XE_MEM_SHARED);
    xe_free(e, batch_k, XE_MEM_SHARED);
    xe_free(e, ring_v, XE_MEM_SHARED);
    xe_free(e, ring_k, XE_MEM_SHARED);
    xe_engine_close(e);
    return ok ? 0 : 1;
}
