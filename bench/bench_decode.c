#include "../xenolith.c"

typedef struct {
    double value[12];
    int n;
} bench_samples;

static double bench_now(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec + (double)ts.tv_nsec * 1e-9;
}

static double bench_rel(const float *a, const float *b, int n) {
    double error = 0.0;
    double reference = 0.0;
    for (int i = 0; i < n; i++) {
        double d = (double)a[i] - b[i];
        error += d * d;
        reference += (double)b[i] * b[i];
    }
    return sqrt(error / (reference + 1e-30));
}

static int bench_double_cmp(const void *a, const void *b) {
    double x = *(const double *)a;
    double y = *(const double *)b;
    return (x > y) - (x < y);
}

static void bench_report(const char *name, bench_samples samples, double bytes) {
    qsort(samples.value, (size_t)samples.n, sizeof(samples.value[0]), bench_double_cmp);
    double median = samples.value[samples.n / 2];
    double best = samples.value[0];
    double worst = samples.value[samples.n - 1];
    if (bytes > 0.0)
        printf("bench: %-22s median %8.3f ms best %8.3f worst %8.3f GB/s %6.2f\n",
               name, median * 1000.0, best * 1000.0, worst * 1000.0, bytes / median / 1e9);
    else
        printf("bench: %-22s median %8.3f ms best %8.3f worst %8.3f\n",
               name, median * 1000.0, best * 1000.0, worst * 1000.0);
}

static void bench_cool(void) {
    struct timespec ts = { 0, 750000000 };
    nanosleep(&ts, NULL);
}

static void bench_restore_trace(xe_session *s, int layer) {
    for (int i = 0; i < XE_EXPERTS_USED; i++) {
        s->experts[i] = s->expert_trace[layer][i];
        s->expert_weights[i] = s->expert_weight_trace[layer][i];
    }
}

static double bench_moe_pass(xe_session *s, int mode) {
    xe_workers_begin(s->engine);
    double start = bench_now();
    for (int layer = 0; layer < XE_LAYERS; layer++) {
        bench_restore_trace(s, layer);
        xe_moe_run(s, &s->engine->layers[layer], mode);
    }
    double elapsed = bench_now() - start;
    xe_workers_end(s->engine);
    return elapsed;
}

static double bench_dense_router_pass(xe_session *s, int overlap) {
    xe_workers_begin(s->engine);
    double start = bench_now();
    for (int layer = 0; layer < XE_LAYERS; layer++) {
        xe_dense_gate_arg dense = { &s->engine->layers[layer] };
        if (overlap && xe_engine_worker_count(s->engine) > 1) {
            uint64_t epoch = xe_publish_phase(s->engine, s, xe_dense_gate_phase, &dense, 0);
            xe_router(s, &s->engine->layers[layer], layer);
            xe_wait_followers(s->engine, epoch);
        } else {
            xe_dispatch(s->engine, s, xe_dense_gate_phase, &dense, 1);
            xe_router(s, &s->engine->layers[layer], layer);
        }
    }
    double elapsed = bench_now() - start;
    xe_workers_end(s->engine);
    return elapsed;
}

static int bench_moe_correctness(xe_session *s) {
    float *reference = xe_alloc(NULL, (size_t)XE_LAYERS * XE_EMBD * sizeof(*reference), XE_MEM_HOST);
    int8_t *packed_qs = xe_alloc(NULL, (size_t)XE_EXPERTS_USED * XE_EXPERT_FFN, XE_MEM_HOST);
    _Float16 *packed_d = xe_alloc(NULL, (size_t)XE_EXPERTS_USED * (XE_EXPERT_FFN / 32) * sizeof(*packed_d), XE_MEM_HOST);
    int16_t *packed_sigma = xe_alloc(NULL, (size_t)XE_EXPERTS_USED * (XE_EXPERT_FFN / 32) * sizeof(*packed_sigma), XE_MEM_HOST);
    int ok = 1;
    int packed_exact = 1;
    xe_workers_begin(s->engine);
    for (int layer = 0; layer < XE_LAYERS; layer++) {
        bench_restore_trace(s, layer);
        xe_moe_run(s, &s->engine->layers[layer], XE_MOE_FUSED_BATCHED);
        memcpy(reference + (size_t)layer * XE_EMBD, s->moe_out, XE_EMBD * sizeof(*s->moe_out));
        for (int slot = 0; slot < XE_EXPERTS_USED; slot++) {
            memcpy(packed_qs + (size_t)slot * XE_EXPERT_FFN, s->q8_expert[slot].qs, XE_EXPERT_FFN);
            memcpy(packed_d + (size_t)slot * (XE_EXPERT_FFN / 32), s->q8_expert[slot].d,
                   (XE_EXPERT_FFN / 32) * sizeof(*packed_d));
            memcpy(packed_sigma + (size_t)slot * (XE_EXPERT_FFN / 32), s->q8_expert[slot].sigma,
                   (XE_EXPERT_FFN / 32) * sizeof(*packed_sigma));
        }
        xe_moe_run(s, &s->engine->layers[layer], XE_MOE_GENERIC_BATCHED);
        for (int slot = 0; slot < XE_EXPERTS_USED; slot++) {
            packed_exact = packed_exact && memcmp(packed_qs + (size_t)slot * XE_EXPERT_FFN,
                                                  s->q8_expert[slot].qs, XE_EXPERT_FFN) == 0;
            packed_exact = packed_exact && memcmp(packed_d + (size_t)slot * (XE_EXPERT_FFN / 32),
                                                  s->q8_expert[slot].d,
                                                  (XE_EXPERT_FFN / 32) * sizeof(*packed_d)) == 0;
            packed_exact = packed_exact && memcmp(packed_sigma + (size_t)slot * (XE_EXPERT_FFN / 32),
                                                  s->q8_expert[slot].sigma,
                                                  (XE_EXPERT_FFN / 32) * sizeof(*packed_sigma)) == 0;
        }
    }
    for (int mode = XE_MOE_EXPERT_PARALLEL; mode <= XE_MOE_GENERIC_BATCHED; mode++) {
        double worst = 0.0;
        for (int layer = 0; layer < XE_LAYERS; layer++) {
            bench_restore_trace(s, layer);
            xe_moe_run(s, &s->engine->layers[layer], mode);
            double rel = bench_rel(s->moe_out, reference + (size_t)layer * XE_EMBD, XE_EMBD);
            if (rel > worst) worst = rel;
        }
        printf("bench: MoE correctness mode %d worst rel_rms %.3e\n", mode, worst);
        if (worst > 2e-6) ok = 0;
    }
    printf("bench: MoE fused/generic packed Q8 %s\n", packed_exact ? "exact" : "DIFFERENT");
    if (!packed_exact) ok = 0;
    xe_workers_end(s->engine);
    xe_free(NULL, packed_sigma, XE_MEM_HOST);
    xe_free(NULL, packed_d, XE_MEM_HOST);
    xe_free(NULL, packed_qs, XE_MEM_HOST);
    xe_free(NULL, reference, XE_MEM_HOST);
    return ok;
}

static double bench_output(xe_session *s, int mode) {
    xe_workers_begin(s->engine);
    double start = bench_now();
    xe_output_decode(s, mode);
    double elapsed = bench_now() - start;
    xe_workers_end(s->engine);
    return elapsed;
}

static double bench_full_mode(xe_session *s, int moe_mode, int overlap) {
    s->n_tokens = 0;
    double start = bench_now();
    xe_decode_token_mode(s, 2, 0, moe_mode, overlap, XE_SOFTCAP_SECOND_LOOP, 1);
    return bench_now() - start;
}

static double bench_full(xe_session *s, int overlap) {
    return bench_full_mode(s, XE_MOE_GENERIC_BATCHED, overlap);
}

int main(int argc, char **argv) {
    if (argc < 2 || !argv[1][0]) {
        fprintf(stderr, "usage: %s <model.gguf>\n", argv[0]);
        return 2;
    }
    const char *model = argv[1];
    xe_engine *e = xe_engine_open(model);
    xe_session *s = xe_session_new(e);
    double warm = bench_full(s, 1);
    printf("bench: warm full token %.3f ms\n", warm * 1000.0);

    int ok = bench_moe_correctness(s);
    bench_samples moe[3] = {0};
    const int order[6][3] = {
        {0, 1, 2}, {2, 0, 1}, {1, 2, 0},
        {0, 2, 1}, {1, 0, 2}, {2, 1, 0}
    };
    for (int round = 0; round < 6; round++) {
        for (int j = 0; j < 3; j++) {
            int mode = order[round][j];
            moe[mode].value[moe[mode].n++] = bench_moe_pass(s, mode);
        }
    }
    double moe_bytes = 802897920.0;
    bench_report("MoE expert-parallel", moe[0], moe_bytes);
    bench_report("MoE generic-batched", moe[1], moe_bytes);
    bench_report("MoE fused-batched", moe[2], moe_bytes);
    bench_cool();

    bench_samples dense_router[2] = {0};
    for (int round = 0; round < 10; round++) {
        int first = round & 1;
        dense_router[first].value[dense_router[first].n++] = bench_dense_router_pass(s, first);
        dense_router[1 - first].value[dense_router[1 - first].n++] =
            bench_dense_router_pass(s, 1 - first);
    }
    bench_report("dense+router sequential", dense_router[0], 0.0);
    bench_report("dense+router overlap", dense_router[1], 0.0);
    bench_cool();

    float *raw_reference = xe_alloc(NULL, XE_VOCAB * sizeof(float), XE_MEM_HOST);
    float *softcap_reference = xe_alloc(NULL, XE_VOCAB * sizeof(float), XE_MEM_HOST);
    bench_output(s, XE_SOFTCAP_RAW);
    memcpy(raw_reference, s->logits, XE_VOCAB * sizeof(float));
    bench_output(s, XE_SOFTCAP_SECOND_LOOP);
    memcpy(softcap_reference, s->logits, XE_VOCAB * sizeof(float));
    int head_exact = 1;
    for (int i = 0; i < XE_VOCAB; i++) {
        float want = XE_LOGIT_SOFTCAP * tanhf(raw_reference[i] / XE_LOGIT_SOFTCAP);
        head_exact = head_exact && isfinite(raw_reference[i]) && softcap_reference[i] == want;
    }
    bench_output(s, XE_SOFTCAP_IMMEDIATE);
    double softcap_rel = bench_rel(s->logits, softcap_reference, XE_VOCAB);
    bench_output(s, XE_SOFTCAP_RAW_VECTOR);
    int raw_exact = memcmp(s->logits, raw_reference, XE_VOCAB * sizeof(float)) == 0;
    printf("bench: head raw=%s softcap=%s immediate/second rel_rms %.3e\n",
           raw_exact ? "exact" : "DIFFERENT", head_exact ? "exact" : "DIFFERENT", softcap_rel);
    if (softcap_rel != 0.0 || !raw_exact || !head_exact) ok = 0;
    xe_free(NULL, softcap_reference, XE_MEM_HOST);
    xe_free(NULL, raw_reference, XE_MEM_HOST);

    bench_samples output[3] = {0};
    for (int round = 0; round < 6; round++) {
        for (int j = 0; j < 3; j++) {
            int mode = order[round][j];
            output[mode].value[output[mode].n++] = bench_output(s, mode);
        }
    }
    double head_bytes = 415236096.0;
    bench_report("head raw", output[0], head_bytes);
    bench_report("head immediate tanh", output[1], head_bytes);
    bench_report("head second-loop tanh", output[2], head_bytes);
    bench_samples raw4 = {0};
    for (int round = 0; round < 6; round++)
        raw4.value[raw4.n++] = bench_output(s, XE_SOFTCAP_RAW4);
    bench_report("head raw dot4", raw4, head_bytes);
    bench_samples raw_vector = {0};
    for (int round = 0; round < 6; round++)
        raw_vector.value[raw_vector.n++] = bench_output(s, XE_SOFTCAP_RAW_VECTOR);
    bench_report("head raw vector-sum", raw_vector, head_bytes);
    bench_cool();

    bench_samples full_moe[3] = {0};
    for (int round = 0; round < 6; round++) {
        for (int j = 0; j < 3; j++) {
            int mode = order[round][j];
            full_moe[mode].value[full_moe[mode].n++] = bench_full_mode(s, mode, 1);
        }
    }
    bench_report("full expert-parallel", full_moe[0], 0.0);
    bench_report("full generic-batched", full_moe[1], 0.0);
    bench_report("full fused-batched", full_moe[2], 0.0);
    bench_cool();

    bench_samples full[2] = {0};
    for (int round = 0; round < 6; round++) {
        int first = round & 1;
        full[first].value[full[first].n++] = bench_full(s, first);
        full[1 - first].value[full[1 - first].n++] = bench_full(s, 1 - first);
    }
    bench_report("full router sequential", full[0], 0.0);
    bench_report("full router overlap", full[1], 0.0);

    bench_samples rms[2] = {0};
    for (int round = 0; round < 8; round++) {
        int first = round & 1;
        e->scalar_rms = first;
        rms[first].value[rms[first].n++] = bench_full(s, 1);
        e->scalar_rms = 1 - first;
        rms[1 - first].value[rms[1 - first].n++] = bench_full(s, 1);
    }
    e->scalar_rms = 1;
    bench_report("full AVX RMS", rms[0], 0.0);
    bench_report("full scalar RMS", rms[1], 0.0);

    xe_session_free(s);
    xe_engine_close(e);
    return ok ? 0 : 1;
}
