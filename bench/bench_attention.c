#include "../xenolith.c"

static double attention_now(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec + (double)ts.tv_nsec * 1e-9;
}

static double attention_rel(const float *a, const float *b, int n) {
    double error = 0.0;
    double reference = 0.0;
    for (int i = 0; i < n; i++) {
        double d = (double)a[i] - b[i];
        error += d * d;
        reference += (double)b[i] * b[i];
    }
    return sqrt(error / (reference + 1e-30));
}

static int attention_cmp(const void *a, const void *b) {
    double x = *(const double *)a;
    double y = *(const double *)b;
    return (x > y) - (x < y);
}

static _Float16 attention_value(size_t i, int salt) {
    uint64_t x = (uint64_t)i * UINT64_C(6364136223846793005) + (uint64_t)salt;
    int v = (int)((x >> 32) % 2047) - 1023;
    return (_Float16)((float)v * 0.0003f);
}

static void attention_fill(xe_session *s, int max_n) {
    int layer = 5;
    _Float16 *k = xe_kv_layer_ptr(s, layer, 0);
    _Float16 *v = xe_kv_layer_ptr(s, layer, 1);
    for (int h = 0; h < XE_GLOBAL_KV_HEADS; h++) {
        for (int p = 0; p < max_n; p++) {
            size_t base = ((size_t)h * s->engine->context + p) * XE_GLOBAL_HEAD_DIM;
            for (int d = 0; d < XE_GLOBAL_HEAD_DIM; d++) {
                k[base + d] = attention_value(((size_t)h * max_n + p) * XE_GLOBAL_HEAD_DIM + d, layer + 1);
                v[base + d] = attention_value(((size_t)h * max_n + p) * XE_GLOBAL_HEAD_DIM + d, layer + 97);
            }
        }
    }
    for (int i = 0; i < XE_Q_HEADS * XE_GLOBAL_HEAD_DIM; i++)
        s->q[i] = (float)attention_value((size_t)i, 211) * 0.03f;
}

static void attention_empty_phase(xe_session *s, const void *arg, int worker, int workers) {
    (void)s;
    (void)arg;
    (void)worker;
    (void)workers;
}

static double attention_pass(xe_session *s, int n, int sharded) {
    double start = attention_now();
    xe_attention_parallel_mode(s, 5, n - 1, sharded);
    return attention_now() - start;
}

int main(void) {
    xe_engine e;
    memset(&e, 0, sizeof e);
    e.context = XE_CONTEXT_DEFAULT;
    xe_session *s = xe_session_new(&e);
    int contexts[] = { 1, 8, 16, 24, 32, 64, 128, 512, 1024, 4096, 8192, 32768, 262144 };
    int n_contexts = (int)(sizeof contexts / sizeof contexts[0]);
    attention_fill(s, contexts[n_contexts - 1]);
    float *reference = xe_alloc(NULL, (size_t)XE_Q_HEADS * XE_GLOBAL_HEAD_DIM * sizeof(*reference), XE_MEM_HOST);
    int ok = 1;

    xe_workers_begin(&e);
    double empty_start = attention_now();
    for (int i = 0; i < 10000; i++)
        xe_dispatch(&e, s, attention_empty_phase, NULL, 1);
    printf("attention: empty phase %.3f us\n", (attention_now() - empty_start) * 1e6 / 10000.0);
    for (int ci = 0; ci < n_contexts; ci++) {
        int n = contexts[ci];
        xe_attention_parallel_mode(s, 5, n - 1, 0);
        memcpy(reference, s->attn_heads,
               (size_t)XE_Q_HEADS * XE_GLOBAL_HEAD_DIM * sizeof(*reference));
        xe_attention_parallel_mode(s, 5, n - 1, 1);
        double rel = attention_rel(s->attn_heads, reference, XE_Q_HEADS * XE_GLOBAL_HEAD_DIM);
        if (rel > 2e-5) ok = 0;

        double direct[7];
        double sharded[7];
        for (int round = 0; round < 7; round++) {
            if (round & 1) {
                sharded[round] = attention_pass(s, n, 1);
                direct[round] = attention_pass(s, n, 0);
            } else {
                direct[round] = attention_pass(s, n, 0);
                sharded[round] = attention_pass(s, n, 1);
            }
        }
        qsort(direct, 7, sizeof(double), attention_cmp);
        qsort(sharded, 7, sizeof(double), attention_cmp);
        printf("attention: n=%6d direct %.3f ms sharded %.3f ms ratio %.3f rel %.3e\n",
               n, direct[3] * 1000.0, sharded[3] * 1000.0,
               sharded[3] / direct[3], rel);
    }
    xe_workers_end(&e);

    xe_free(NULL, reference, XE_MEM_HOST);
    xe_session_free(s);
    xe_worker_pool_destroy(&e);
    xe_free(NULL, e.gelu_lut, XE_MEM_HOST);
    return ok ? 0 : 1;
}
