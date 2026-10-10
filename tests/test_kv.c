#include "../xenolith.c"

#include <sys/wait.h>

static float test_k(int p, int h, int d) {
    return (float)(((p * 17 + h * 29 + d * 13) % 257) - 128) * 0.002f;
}

static float test_v(int p, int h, int d) {
    return (float)(((p * 31 + h * 11 + d * 7) % 251) - 125) * 0.003f;
}

static float test_q(int h, int d) {
    return (float)(((h * 43 + d * 19) % 263) - 131) * 0.0005f;
}

static double test_rel_rms(const float *a, const float *b, size_t n) {
    double err = 0.0;
    double ref = 0.0;
    for (size_t i = 0; i < n; i++) {
        double d = (double)a[i] - b[i];
        err += d * d;
        ref += (double)b[i] * b[i];
    }
    return sqrt(err / (ref + 1e-30));
}

typedef struct {
    atomic_int hits[257];
    atomic_int lane_hits[XE_WORKERS];
    int n;
} test_phase_arg;

typedef struct {
    uint8_t *begin;
    size_t size;
} test_region;

static void test_phase_fn(xe_session *s, const void *opaque, int worker, int workers) {
    (void)s;
    test_phase_arg *a = (test_phase_arg *)opaque;
    int begin = a->n * worker / workers;
    int end = a->n * (worker + 1) / workers;
    atomic_fetch_add_explicit(&a->lane_hits[worker], end - begin, memory_order_relaxed);
    for (int i = begin; i < end; i++)
        atomic_fetch_add_explicit(&a->hits[i], 1, memory_order_relaxed);
}

static int test_scheduler_case(xe_session *s, int n, int rounds, int caller_participates) {
    test_phase_arg a;
    memset(&a, 0, sizeof a);
    a.n = n;
    xe_workers_begin(s->engine);
    for (int i = 0; i < rounds; i++)
        xe_dispatch(s->engine, s, test_phase_fn, &a, caller_participates);
    xe_workers_end(s->engine);

    int workers = caller_participates ? XE_WORKERS : XE_WORKERS - 1;
    int ok = 1;
    for (int i = 0; i < n; i++)
        if (atomic_load_explicit(&a.hits[i], memory_order_relaxed) != rounds) ok = 0;
    for (int lane = 0; lane < workers; lane++) {
        int want = (n * (lane + 1) / workers - n * lane / workers) * rounds;
        if (atomic_load_explicit(&a.lane_hits[lane], memory_order_relaxed) != want) ok = 0;
    }
    return ok;
}

static int test_scheduler(xe_session *s) {
    int ok = test_scheduler_case(s, 257, 200, 1);
    ok = ok && test_scheduler_case(s, 17, 200, 0);
    ok = ok && test_scheduler_case(s, 3, 200, 1);
    ok = ok && test_scheduler_case(s, 0, 100000, 1);
    ok = ok && test_scheduler_case(s, 0, 100000, 0);
    for (int lane = 0; lane < XE_WORKERS; lane++)
        ok = ok && ((uintptr_t)&s->engine->phase_done[lane] & 63u) == 0;
    printf("workers: phases %s\n", ok ? "PASS" : "FAIL");
    return ok;
}

static void test_add_q8_regions(test_region *r, int *n, xe_q8 *q) {
    r[(*n)++] = (test_region){ (uint8_t *)q->qs, (size_t)q->n };
    r[(*n)++] = (test_region){ (uint8_t *)q->d, (size_t)(q->n / 32) * sizeof(*q->d) };
    r[(*n)++] = (test_region){ (uint8_t *)q->sigma, (size_t)(q->n / 32) * sizeof(*q->sigma) };
}

static int test_workspace_regions(xe_session *s) {
    test_region r[64];
    int n = 0;
#define TEST_REGION(p, count) r[n++] = (test_region){ (uint8_t *)(p), (size_t)(count) * sizeof(*(p)) }
    TEST_REGION(s->hidden, XE_EMBD);
    TEST_REGION(s->q, XE_Q_HEADS * XE_GLOBAL_HEAD_DIM);
    TEST_REGION(s->k, XE_SWA_KV_HEADS * XE_SWA_HEAD_DIM);
    TEST_REGION(s->v, XE_SWA_KV_HEADS * XE_SWA_HEAD_DIM);
    TEST_REGION(s->attn_heads, XE_Q_HEADS * XE_GLOBAL_HEAD_DIM);
    TEST_REGION(s->attn_proj, XE_EMBD);
    TEST_REGION(s->attn_out, XE_EMBD);
    TEST_REGION(s->dense_out, XE_EMBD);
    TEST_REGION(s->moe_out, XE_EMBD);
    TEST_REGION(s->combined, XE_EMBD);
    TEST_REGION(s->router_in, XE_EMBD);
    TEST_REGION(s->router_logits, XE_EXPERTS);
    TEST_REGION(s->scores, XE_Q_HEADS * s->engine->context);
    TEST_REGION(s->attn_partial, XE_WORKERS * 8 * XE_GLOBAL_HEAD_DIM);
    TEST_REGION(s->rope_swa_cos, XE_SWA_HEAD_DIM / 2);
    TEST_REGION(s->rope_swa_sin, XE_SWA_HEAD_DIM / 2);
    TEST_REGION(s->rope_global_cos, XE_GLOBAL_HEAD_DIM / 2);
    TEST_REGION(s->rope_global_sin, XE_GLOBAL_HEAD_DIM / 2);
    TEST_REGION(s->logits, XE_VOCAB);
    TEST_REGION(s->sample_candidates, XE_VOCAB);
#undef TEST_REGION
    test_add_q8_regions(r, &n, &s->q8_main);
    test_add_q8_regions(r, &n, &s->q8_dense_in);
    test_add_q8_regions(r, &n, &s->q8_moe_in);
    test_add_q8_regions(r, &n, &s->q8_dense_act);
    for (int i = 0; i < XE_EXPERTS_USED; i++) test_add_q8_regions(r, &n, &s->q8_expert[i]);

    for (int i = 0; i < n; i++) {
        for (int j = i + 1; j < n; j++) {
            if (r[j].begin < r[i].begin) {
                test_region t = r[i];
                r[i] = r[j];
                r[j] = t;
            }
        }
    }

    uint8_t *base = s->workspace;
    uint8_t *limit = base + s->workspace_size;
    int ok = 1;
    for (int i = 0; i < n; i++) {
        ok = ok && r[i].size > 0 && r[i].begin >= base && r[i].begin + r[i].size <= limit;
        ok = ok && ((uintptr_t)r[i].begin & 63u) == 0;
        if (i) ok = ok && r[i - 1].begin + r[i - 1].size <= r[i].begin;
        r[i].begin[0] = (uint8_t)(0x40 + i);
        r[i].begin[r[i].size - 1] = (uint8_t)(0x80 + i);
    }
    for (int i = 0; i < n; i++) {
        ok = ok && r[i].begin[0] == (uint8_t)(0x40 + i);
        ok = ok && r[i].begin[r[i].size - 1] == (uint8_t)(0x80 + i);
    }
    printf("workspace: %d regions %s\n", n, ok ? "PASS" : "FAIL");
    return ok;
}

static int test_raw_append(void) {
    int n_kv_heads = 2;
    int head_dim = 512;
    int capacity = 4;
    int slot = 2;
    size_t cache_elems = (size_t)n_kv_heads * capacity * head_dim;
    size_t token_elems = (size_t)n_kv_heads * head_dim;
    _Float16 *kc = xe_alloc(NULL, cache_elems * sizeof(*kc), XE_MEM_HOST);
    _Float16 *vc = xe_alloc(NULL, cache_elems * sizeof(*vc), XE_MEM_HOST);
    float *k = xe_alloc(NULL, token_elems * sizeof(*k), XE_MEM_HOST);
    float *v = xe_alloc(NULL, token_elems * sizeof(*v), XE_MEM_HOST);
    for (size_t i = 0; i < cache_elems; i++) {
        kc[i] = (_Float16)-7.0f;
        vc[i] = (_Float16)7.0f;
    }
    for (int h = 0; h < n_kv_heads; h++) {
        for (int d = 0; d < head_dim; d++) {
            k[(size_t)h * head_dim + d] = test_k(19, h, d);
            v[(size_t)h * head_dim + d] = test_v(19, h, d);
        }
    }
    xe_kv_append_f16(kc, vc, n_kv_heads, head_dim, capacity, slot, k, v);
    int ok = 1;
    for (int h = 0; h < n_kv_heads; h++) {
        for (int p = 0; p < capacity; p++) {
            for (int d = 0; d < head_dim; d++) {
                size_t i = ((size_t)h * capacity + p) * head_dim + d;
                _Float16 want_k = p == slot ? (_Float16)test_k(19, h, d) : (_Float16)-7.0f;
                _Float16 want_v = p == slot ? (_Float16)test_v(19, h, d) : (_Float16)7.0f;
                if (kc[i] != want_k || vc[i] != want_v) ok = 0;
            }
        }
    }
    xe_free(NULL, v, XE_MEM_HOST);
    xe_free(NULL, k, XE_MEM_HOST);
    xe_free(NULL, vc, XE_MEM_HOST);
    xe_free(NULL, kc, XE_MEM_HOST);
    printf("kv: raw append %s\n", ok ? "PASS" : "FAIL");
    return ok;
}

static void *test_wrong_owner_main(void *opaque) {
    xe_workers_begin(opaque);
    return NULL;
}

static int test_owner_enforcement(void) {
    fflush(NULL);
    pid_t pid = fork();
    if (pid < 0) return 0;
    if (pid == 0) {
        xe_engine e;
        memset(&e, 0, sizeof e);
        xe_worker_pool_init(&e);
        pthread_t thread;
        if (pthread_create(&thread, NULL, test_wrong_owner_main, &e) != 0) _exit(2);
        pthread_join(thread, NULL);
        _exit(0);
    }
    int status;
    int ok = waitpid(pid, &status, 0) == pid && WIFEXITED(status) && WEXITSTATUS(status) == 1;
    printf("workers: owner enforcement %s\n", ok ? "PASS" : "FAIL");
    return ok;
}

static int test_shared_allocation(xe_engine *e, const void *p) {
    ze_memory_allocation_properties_t properties = {
        .stype = ZE_STRUCTURE_TYPE_MEMORY_ALLOCATION_PROPERTIES
    };
    ze_device_handle_t device = NULL;
    return zeMemGetAllocProperties(e->gpu.context, p, &properties, &device) ==
               ZE_RESULT_SUCCESS &&
           properties.type == ZE_MEMORY_TYPE_SHARED && device == e->gpu.device;
}

static int test_prefill_global_kv(xe_engine *e);

static int test_context_bounds(void) {
    int ok = 1;
    /* These failures must happen before file opening or any GPU access. */
    for (int operation = 0; operation < 5; operation++) {
        int fds[2];
        if (pipe(fds)) return 0;
        fflush(NULL);
        pid_t child = fork();
        if (child == 0) {
            close(fds[0]);
            dup2(fds[1], STDERR_FILENO);
            close(fds[1]);
            if (operation < 2) {
                xe_engine_open_with_context("/missing-model.gguf",
                                            operation ? XE_CONTEXT_MAX + 1 : XE_CONTEXT_MIN - 1);
            } else {
                xe_engine e = { .context = 64, .pool_initialized = 1,
                                .owner = pthread_self() };
                xe_session s = { .engine = &e, .n_tokens = 64 };
                int32_t ids[65] = {0};
                xe_tokens prefix = { ids, 65, 65 };
                if (operation == 2) xe_session_sync(&s, &prefix);
                else if (operation == 3) xe_session_kv_append(&s, 5, 64, NULL, NULL);
                else xe_decode_token_mode(&s, 0, 64, 0, 0, 0, 0);
            }
            _exit(0);
        }
        close(fds[1]);
        char message[512];
        size_t used = 0;
        ssize_t got;
        while ((got = read(fds[0], message + used, sizeof message - 1 - used)) > 0)
            used += (size_t)got;
        close(fds[0]);
        message[used] = '\0';
        int status;
        int passed = child > 0 && waitpid(child, &status, 0) == child &&
                     WIFEXITED(status) && WEXITSTATUS(status) == 1 &&
                     strstr(message, "out of range") != NULL;
        if (!passed) fprintf(stderr, "context bounds operation %d failed: %s\n",
                             operation, message);
        ok &= passed;
    }
    printf("kv: runtime context bounds %s\n", ok ? "PASS" : "FAIL");
    return ok;
}

static int test_session_slabs(int context) {
    xe_engine e;
    memset(&e, 0, sizeof e);
    e.context = context;
    cpu_set_t original_affinity;
    if (pthread_getaffinity_np(pthread_self(), sizeof original_affinity, &original_affinity) != 0)
        return 0;
    xe_gpu_init(&e);
    xe_session *s = xe_session_new(&e);
    int ok = s && s->engine == &e && s->n_tokens == 0 &&
             xe_context_size(&e) == context;
    int position = context - 1;
    ok = ok && e.worker_pinned == xe_worker_pin_available(&original_affinity);
    ok = ok && ((uintptr_t)s->swa_k & 63u) == 0;
    ok = ok && ((uintptr_t)s->swa_v & 63u) == 0;
    ok = ok && ((uintptr_t)s->global_k & 63u) == 0;
    ok = ok && ((uintptr_t)s->global_v & 63u) == 0;
    ok = ok && ((uintptr_t)s->workspace & 63u) == 0;
    ok = ok && test_shared_allocation(&e, s->swa_k);
    ok = ok && test_shared_allocation(&e, s->swa_v);
    ok = ok && test_shared_allocation(&e, s->global_k);
    ok = ok && test_shared_allocation(&e, s->global_v);
    ok = ok && test_shared_allocation(&e, s->workspace);
    ok = ok && ((uintptr_t)s->hidden & 63u) == 0;
    ok = ok && ((uintptr_t)s->scores & 63u) == 0;
    ok = ok && ((uintptr_t)s->logits & 63u) == 0;
    ok = ok && ((uintptr_t)s->q8_main.qs & 63u) == 0;
    ok = ok && ((uintptr_t)s->q8_main.d & 63u) == 0;
    ok = ok && ((uintptr_t)s->q8_main.sigma & 63u) == 0;
    ok = ok && xe_kv_layer_ptr(s, 0, 0) == s->swa_k;
    ok = ok && xe_kv_layer_ptr(s, 6, 0) == s->swa_k + 5 * XE_SWA_LAYER_ELEMS;
    ok = ok && xe_kv_layer_ptr(s, 5, 0) == s->global_k;
    ok = ok && xe_kv_layer_ptr(s, 29, 1) == s->global_v + 4 * xe_global_layer_elems(s->engine);
    s->n_tokens = context;
    xe_session_reset(s);
    ok = ok && s->n_tokens == 0;
    ok = ok && xe_engine_worker_count(&e) == XE_WORKERS;
    for (int lane = 0; lane < XE_WORKERS; lane++)
        ok = ok && xe_engine_worker_cpu(&e, lane) ==
                   (e.worker_pinned ? xe_worker_cpu(lane) : -1);

    float *swa_k = xe_alloc(NULL, XE_SWA_KV_HEADS * XE_SWA_HEAD_DIM * sizeof(*swa_k), XE_MEM_HOST);
    float *swa_v = xe_alloc(NULL, XE_SWA_KV_HEADS * XE_SWA_HEAD_DIM * sizeof(*swa_v), XE_MEM_HOST);
    for (int h = 0; h < XE_SWA_KV_HEADS; h++) {
        for (int d = 0; d < XE_SWA_HEAD_DIM; d++) {
            swa_k[(size_t)h * XE_SWA_HEAD_DIM + d] = test_k(position, h, d);
            swa_v[(size_t)h * XE_SWA_HEAD_DIM + d] = test_v(position, h, d);
        }
    }
    xe_session_kv_append(s, 6, position, swa_k, swa_v);
    _Float16 *swa_k_layer = xe_kv_layer_ptr(s, 6, 0);
    _Float16 *swa_v_layer = xe_kv_layer_ptr(s, 6, 1);
    for (int h = 0; h < XE_SWA_KV_HEADS; h++) {
        for (int d = 0; d < XE_SWA_HEAD_DIM; d++) {
            size_t i = ((size_t)h * XE_SWA_WINDOW + (position & (XE_SWA_WINDOW - 1))) * XE_SWA_HEAD_DIM + d;
            if (swa_k_layer[i] != (_Float16)test_k(position, h, d)) ok = 0;
            if (swa_v_layer[i] != (_Float16)test_v(position, h, d)) ok = 0;
        }
    }

    float *global_k = xe_alloc(NULL, XE_GLOBAL_KV_HEADS * XE_GLOBAL_HEAD_DIM * sizeof(*global_k), XE_MEM_HOST);
    float *global_v = xe_alloc(NULL, XE_GLOBAL_KV_HEADS * XE_GLOBAL_HEAD_DIM * sizeof(*global_v), XE_MEM_HOST);
    for (int h = 0; h < XE_GLOBAL_KV_HEADS; h++) {
        for (int d = 0; d < XE_GLOBAL_HEAD_DIM; d++) {
            global_k[(size_t)h * XE_GLOBAL_HEAD_DIM + d] = test_k(position, h, d);
            global_v[(size_t)h * XE_GLOBAL_HEAD_DIM + d] = test_v(position, h, d);
        }
    }
    xe_session_kv_append(s, 5, position, global_k, global_v);
    _Float16 *global_k_layer = xe_kv_layer_ptr(s, 5, 0);
    _Float16 *global_v_layer = xe_kv_layer_ptr(s, 5, 1);
    for (int h = 0; h < XE_GLOBAL_KV_HEADS; h++) {
        for (int d = 0; d < XE_GLOBAL_HEAD_DIM; d++) {
            size_t i = ((size_t)h * s->engine->context + position) * XE_GLOBAL_HEAD_DIM + d;
            if (global_k_layer[i] != (_Float16)test_k(position, h, d)) ok = 0;
            if (global_v_layer[i] != (_Float16)test_v(position, h, d)) ok = 0;
        }
    }

    xe_free(NULL, global_v, XE_MEM_HOST);
    xe_free(NULL, global_k, XE_MEM_HOST);
    xe_free(NULL, swa_v, XE_MEM_HOST);
    xe_free(NULL, swa_k, XE_MEM_HOST);
    ok = ok && test_scheduler(s);
    ok = ok && test_workspace_regions(s);
    xe_tokens prefix = {0};
    for (int i = 0; i < context; i++) xe_tokens_push(&prefix, 2);
    memcpy(s->tokens, prefix.v, (size_t)context * sizeof(*s->tokens));
    s->n_tokens = context;
    xe_sync_report report;
    xe_session_sync_report(s, &prefix, &report);
    ok &= report.reused == context && report.prefilled == 0 &&
          xe_context_size(&e) == context;
    xe_tokens_free(&prefix);
    if (context == 4097) ok &= test_prefill_global_kv(&e);
    xe_session_free(s);
    xe_worker_pool_destroy(&e);
    cpu_set_t restored_affinity;
    ok = ok && pthread_getaffinity_np(pthread_self(), sizeof restored_affinity,
                                      &restored_affinity) == 0 &&
               CPU_EQUAL(&original_affinity, &restored_affinity);
    xe_gpu_destroy(&e);
    printf("kv: session capacity %d slabs %s\n", context, ok ? "PASS" : "FAIL");
    return ok;
}

static int test_attention_case(const char *name, int n_kv_heads, int head_dim,
                               int capacity, int pos, int window) {
    size_t cache_elems = (size_t)n_kv_heads * capacity * head_dim;
    size_t token_elems = (size_t)n_kv_heads * head_dim;
    int n_keys = pos + 1;
    if (window && n_keys > window) n_keys = window;
    _Float16 *kc = xe_alloc(NULL, cache_elems * sizeof(*kc), XE_MEM_HOST);
    _Float16 *vc = xe_alloc(NULL, cache_elems * sizeof(*vc), XE_MEM_HOST);
    float *k = xe_alloc(NULL, token_elems * sizeof(*k), XE_MEM_HOST);
    float *v = xe_alloc(NULL, token_elems * sizeof(*v), XE_MEM_HOST);
    float *q = xe_alloc(NULL, (size_t)XE_Q_HEADS * head_dim * sizeof(*q), XE_MEM_HOST);
    float *scores = xe_alloc(NULL, (size_t)XE_Q_HEADS * n_keys * sizeof(*scores), XE_MEM_HOST);
    float *out = xe_alloc(NULL, (size_t)XE_Q_HEADS * head_dim * sizeof(*out), XE_MEM_HOST);
    float *ref = xe_alloc(NULL, (size_t)XE_Q_HEADS * head_dim * sizeof(*ref), XE_MEM_HOST);
    float *ref_k = xe_alloc(NULL, (size_t)(pos + 1) * n_kv_heads * head_dim * sizeof(*ref_k), XE_MEM_HOST);
    float *ref_v = xe_alloc(NULL, (size_t)(pos + 1) * n_kv_heads * head_dim * sizeof(*ref_v), XE_MEM_HOST);
    float *ref_scores = xe_alloc(NULL, (size_t)(pos + 1) * sizeof(*ref_scores), XE_MEM_HOST);
    memset(kc, 0, cache_elems * sizeof(*kc));
    memset(vc, 0, cache_elems * sizeof(*vc));
    for (int p = 0; p <= pos; p++) {
        for (int h = 0; h < n_kv_heads; h++) {
            for (int d = 0; d < head_dim; d++) {
                k[(size_t)h * head_dim + d] = test_k(p, h, d);
                v[(size_t)h * head_dim + d] = test_v(p, h, d);
                ref_k[((size_t)p * n_kv_heads + h) * head_dim + d] = (float)(_Float16)test_k(p, h, d);
                ref_v[((size_t)p * n_kv_heads + h) * head_dim + d] = (float)(_Float16)test_v(p, h, d);
            }
        }
        xe_kv_append_f16(kc, vc, n_kv_heads, head_dim, capacity,
                         window ? p & (capacity - 1) : p, k, v);
    }
    for (int h = 0; h < XE_Q_HEADS; h++)
        for (int d = 0; d < head_dim; d++)
            q[(size_t)h * head_dim + d] = test_q(h, d);
    xe_kv_attention_f16(q, kc, vc, n_kv_heads, head_dim, capacity, pos, window, scores, out);
    ref_attention(q, ref_k, ref_v, pos, n_kv_heads, head_dim, window, ref_scores, ref);
    double rel = test_rel_rms(out, ref, (size_t)XE_Q_HEADS * head_dim);
    int ok = rel < 2e-5;
    printf("kv: %-12s rel_rms=%.3e %s\n", name, rel, ok ? "PASS" : "FAIL");
    xe_free(NULL, ref_scores, XE_MEM_HOST);
    xe_free(NULL, ref_v, XE_MEM_HOST);
    xe_free(NULL, ref_k, XE_MEM_HOST);
    xe_free(NULL, ref, XE_MEM_HOST);
    xe_free(NULL, out, XE_MEM_HOST);
    xe_free(NULL, scores, XE_MEM_HOST);
    xe_free(NULL, q, XE_MEM_HOST);
    xe_free(NULL, v, XE_MEM_HOST);
    xe_free(NULL, k, XE_MEM_HOST);
    xe_free(NULL, vc, XE_MEM_HOST);
    xe_free(NULL, kc, XE_MEM_HOST);
    return ok;
}

/* Exercise runtime capacity arguments, including a partial final GPU tile. */
static int test_prefill_global_case(xe_engine *e, int capacity, int split) {
    const int heads = XE_GLOBAL_KV_HEADS, dimension = XE_GLOBAL_HEAD_DIM;
    const int batch_rows = 32, queries = 3;
    const size_t guard = 64;
    size_t elements = (size_t)heads * capacity * dimension;
    size_t q_elements = (size_t)XE_Q_HEADS * queries * dimension;
    _Float16 *cache[2], *batch[2], *tail[2];
    float *reference_cache[2];
    int tail_capacity = capacity - split;
    for (int value = 0; value < 2; value++) {
        cache[value] = xe_alloc(e, (elements + guard) * sizeof(_Float16), XE_MEM_SHARED);
        batch[value] = xe_alloc(e, (size_t)heads * batch_rows * dimension * sizeof(_Float16), XE_MEM_SHARED);
        tail[value] = xe_alloc(e, (size_t)heads * tail_capacity * dimension * sizeof(_Float16), XE_MEM_SHARED);
        reference_cache[value] = xe_alloc(NULL, elements * sizeof(float), XE_MEM_HOST);
        for (size_t i = 0; i < elements + guard; i++) cache[value][i] = (_Float16)7.0f;
        for (int p = 0; p < capacity; p++)
            for (int h = 0; h < heads; h++)
                for (int d = 0; d < dimension; d++)
                    reference_cache[value][((size_t)p * heads + h) * dimension + d] =
                        (float)(_Float16)(value ? test_v(p, h, d) : test_k(p, h, d));
    }
    int ok = 1;
    for (int start = 0; start < capacity; start += batch_rows) {
        int rows = capacity - start;
        if (rows > batch_rows) rows = batch_rows;
        for (int value = 0; value < 2; value++)
            for (int h = 0; h < heads; h++)
                for (int row = 0; row < rows; row++)
                    for (int d = 0; d < dimension; d++)
                        batch[value][((size_t)h * rows + row) * dimension + d] =
                            (_Float16)(value ? test_v(start + row, h, d) : test_k(start + row, h, d));
        xe_prefill_linear_commit_append(e, cache[0], cache[1], batch[0], batch[1],
                                        rows, start, dimension, heads, capacity);
        xe_ze_check("prefill KV commit test synchronize",
                    zeCommandListHostSynchronize(e->gpu.commands, UINT64_MAX));
        /* Check the prefix, the untouched suffix and every head's stride. */
        for (int value = 0; value < 2; value++)
            for (int h = 0; h < heads; h++)
                for (int p = 0; p < capacity; p++)
                    for (int d = 0; d < dimension; d++) {
                        _Float16 want = p < start + rows
                            ? (_Float16)reference_cache[value][((size_t)p * heads + h) * dimension + d]
                            : (_Float16)7.0f;
                        if (cache[value][((size_t)h * capacity + p) * dimension + d] != want) ok = 0;
                    }
    }
    for (int value = 0; value < 2; value++) {
        for (size_t i = elements; i < elements + guard; i++)
            if (cache[value][i] != (_Float16)7.0f) ok = 0;
        for (int h = 0; h < heads; h++)
            memcpy(tail[value] + (size_t)h * tail_capacity * dimension,
                   cache[value] + ((size_t)h * capacity + split) * dimension,
                   (size_t)tail_capacity * dimension * sizeof(_Float16));
    }
    float *q = xe_alloc(e, q_elements * sizeof(float), XE_MEM_SHARED);
    float *out = xe_alloc(e, q_elements * sizeof(float), XE_MEM_SHARED);
    float *reference_q = xe_alloc(NULL, (size_t)XE_Q_HEADS * dimension * sizeof(float), XE_MEM_HOST);
    float *reference_out = xe_alloc(NULL, (size_t)XE_Q_HEADS * dimension * sizeof(float), XE_MEM_HOST);
    float *row_out = xe_alloc(NULL, (size_t)XE_Q_HEADS * dimension * sizeof(float), XE_MEM_HOST);
    float *scores = xe_alloc(NULL, (size_t)capacity * sizeof(float), XE_MEM_HOST);
    for (int h = 0; h < XE_Q_HEADS; h++)
        for (int d = 0; d < dimension; d++) {
            reference_q[(size_t)h * dimension + d] = test_q(h, d);
            for (int row = 0; row < queries; row++)
                q[((size_t)h * queries + row) * dimension + d] = test_q(h, d);
        }
    double maximum_error = 0.0;
    for (int cow = 0; cow < 2; cow++) {
        if (cow)
            xe_prefill_attention_cow_append(e, q, cache[0], cache[1], tail[0], tail[1],
                                           out, queries, capacity, tail_capacity,
                                           capacity - queries, split);
        else
            xe_prefill_attention_online_append(e, q, cache[0], cache[1], out,
                                              queries, capacity, dimension, heads,
                                              capacity - queries, 0);
        xe_ze_check("prefill KV attention test synchronize",
                    zeCommandListHostSynchronize(e->gpu.commands, UINT64_MAX));
        for (int row = 0; row < queries; row++) {
            ref_attention(reference_q, reference_cache[0], reference_cache[1],
                          capacity - queries + row, heads, dimension, 0, scores, reference_out);
            for (int h = 0; h < XE_Q_HEADS; h++)
                memcpy(row_out + (size_t)h * dimension,
                       out + ((size_t)h * queries + row) * dimension,
                       (size_t)dimension * sizeof(float));
            double error = test_rel_rms(row_out, reference_out, (size_t)XE_Q_HEADS * dimension);
            if (!(error < 2e-5)) ok = 0;
            if (error > maximum_error) maximum_error = error;
        }
    }
    xe_free(NULL, scores, XE_MEM_HOST);
    xe_free(NULL, row_out, XE_MEM_HOST);
    xe_free(NULL, reference_out, XE_MEM_HOST);
    xe_free(NULL, reference_q, XE_MEM_HOST);
    xe_free(e, out, XE_MEM_SHARED);
    xe_free(e, q, XE_MEM_SHARED);
    for (int value = 0; value < 2; value++) {
        xe_free(NULL, reference_cache[value], XE_MEM_HOST);
        xe_free(e, tail[value], XE_MEM_SHARED);
        xe_free(e, batch[value], XE_MEM_SHARED);
        xe_free(e, cache[value], XE_MEM_SHARED);
    }
    printf("kv: prefill capacity %d split %d rel_rms=%.3e %s\n",
           capacity, split, maximum_error, ok ? "PASS" : "FAIL");
    return ok;
}

static int test_prefill_global_kv(xe_engine *e) {
    int ok = 1;
    ok &= test_prefill_global_case(e, 64, 8);
    ok &= test_prefill_global_case(e, 65, 7);
    ok &= test_prefill_global_case(e, 95, 61);
    ok &= test_prefill_global_case(e, 129, 8);
    ok &= test_prefill_global_case(e, 1025, 1000);
    return ok;
}

int main(void) {
    int ok = 1;
    ok &= test_context_bounds();
    ok &= test_raw_append();
    const int capacities[] = {64, 65, 129, 4097, 50000, 65536, XE_CONTEXT_MAX};
    for (size_t i = 0; i < sizeof capacities / sizeof capacities[0]; i++)
        ok &= test_session_slabs(capacities[i]);
    ok &= test_owner_enforcement();
    ok &= test_attention_case("global", XE_GLOBAL_KV_HEADS, XE_GLOBAL_HEAD_DIM, 128, 127, 0);
    ok &= test_attention_case("global-odd", XE_GLOBAL_KV_HEADS, XE_GLOBAL_HEAD_DIM, 129, 128, 0);
    ok &= test_attention_case("swa-1023", XE_SWA_KV_HEADS, XE_SWA_HEAD_DIM, XE_SWA_WINDOW, 1023, XE_SWA_WINDOW);
    ok &= test_attention_case("swa-1024", XE_SWA_KV_HEADS, XE_SWA_HEAD_DIM, XE_SWA_WINDOW, 1024, XE_SWA_WINDOW);
    ok &= test_attention_case("swa-1025", XE_SWA_KV_HEADS, XE_SWA_HEAD_DIM, XE_SWA_WINDOW, 1025, XE_SWA_WINDOW);
    return ok ? 0 : 1;
}
