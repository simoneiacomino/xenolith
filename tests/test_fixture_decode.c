#define XE_TEST_ALLOC
#define XE_TEST_OUTPUT_COUNT
#include "../xenolith.c"

static char *fixture_read(const char *path, size_t *size) {
    FILE *f = fopen(path, "rb");
    if (!f) xe_fatal("%s: %s", path, strerror(errno));
    if (fseek(f, 0, SEEK_END) != 0) xe_fatal("%s: seek failed", path);
    long n = ftell(f);
    if (n < 0 || fseek(f, 0, SEEK_SET) != 0) xe_fatal("%s: seek failed", path);
    char *p = xe_alloc(NULL, (size_t)n + 1, XE_MEM_HOST);
    if (fread(p, 1, (size_t)n, f) != (size_t)n) xe_fatal("%s: short read", path);
    fclose(f);
    p[n] = 0;
    *size = (size_t)n;
    return p;
}

static int fixture_ids(const char *path, int32_t *ids, int cap) {
    size_t size;
    char *text = fixture_read(path, &size);
    (void)size;
    char *p = text;
    int n = 0;
    while (*p) {
        while (*p == ' ' || *p == '\n' || *p == '\r' || *p == '\t' || *p == ',') p++;
        if (!*p) break;
        char *end;
        long value = strtol(p, &end, 10);
        if (end == p || n == cap) xe_fatal("%s: bad token list", path);
        ids[n++] = (int32_t)value;
        p = end;
    }
    xe_free(NULL, text, XE_MEM_HOST);
    return n;
}

static double fixture_rel(const float *a, const float *b, int n) {
    double error = 0.0;
    double reference = 0.0;
    for (int i = 0; i < n; i++) {
        double d = (double)a[i] - b[i];
        error += d * d;
        reference += (double)b[i] * b[i];
    }
    return sqrt(error / (reference + 1e-30));
}

static int fixture_argmax(const float *x, int n) {
    int best = 0;
    for (int i = 1; i < n; i++) if (x[i] > x[best]) best = i;
    return best;
}

static int fixture_logits_equal(const float *got, const float *want,
                                double *maximum) {
    double error = fixture_rel(got, want, XE_VOCAB);
    if (error > *maximum) *maximum = error;
    return memcmp(got, want, XE_VOCAB * sizeof(*got)) == 0;
}

static void fixture_session_sync(xe_session *s, int32_t *ids, int n) {
    xe_tokens prefix = { ids, n, n };
    xe_session_sync(s, &prefix);
}

static double fixture_now(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec + (double)ts.tv_nsec * 1e-9;
}

/* Independent schedule control: deliberately bypass sync's common-prefix,
 * rewind and backend-selection decisions. Reuse the real math implementation. */
static void fixture_control_materialize(xe_session *s, const int32_t *ids,
                                        int start, int end, int cpu) {
    s->n_tokens = start;
    if (cpu) {
        if (end != start + 1) xe_fatal("fixture: CPU control expects one token");
        xe_decode_token_mode(s, ids[start], start, XE_MOE_GENERIC_BATCHED, 1,
                             XE_SOFTCAP_SECOND_LOOP, 1);
        s->tokens[start] = ids[start];
    } else {
        while (s->n_tokens < end) {
            int position = s->n_tokens;
            int rows = end - position;
            if (rows > s->text_prefill_rows) rows = s->text_prefill_rows;
            xe_prefill_batch_run(s, ids + position, rows, position,
                                 position + rows == end);
        }
    }
}

static int fixture_state_equal(xe_session *s, xe_session *control,
                                const int32_t *ids, int n, const char *stage,
                                double *maximum) {
    int ok = s->n_tokens == n && control->n_tokens == n
             && !memcmp(s->tokens, ids, (size_t)n * sizeof(*ids))
             && !memcmp(control->tokens, ids, (size_t)n * sizeof(*ids));
    for (int i = 0; i < XE_VOCAB; i++)
        ok &= isfinite(s->logits[i]) && isfinite(control->logits[i]);
    ok = fixture_logits_equal(s->logits, control->logits, maximum) && ok;
    printf("fixture: explicit-schedule %s exact %s\n", stage, ok ? "PASS" : "FAIL");
    return ok;
}

static int fixture_sync(xe_engine *e, int32_t *ids, int n, const float *full_gpu_logits) {
    xe_session *control = xe_session_new(e);
    xe_session *s = xe_session_new(e);
    size_t allocations = xe_test_allocations;
    size_t outputs = xe_test_output_calls;
    double maximum = 0.0;
    int output_count_ok = 1, ok = 1;
#define CHECK_OUTPUTS() (output_count_ok &= xe_test_output_calls == outputs)
    fixture_control_materialize(control, ids, 0, n - 1, 0);
    outputs++; CHECK_OUTPUTS();
    fixture_control_materialize(control, ids, n - 1, n, 1);
    outputs++; CHECK_OUTPUTS();
    fixture_session_sync(s, ids, n - 1);
    outputs++; CHECK_OUTPUTS();
    fixture_session_sync(s, ids, n);
    outputs++; CHECK_OUTPUTS();
    ok &= fixture_state_equal(s, control, ids, n, "initial append", &maximum);
    printf("fixture: whole-GPU vs GPU-prefix/CPU-last-token rel %.6e top1 %s DIAGNOSTIC\n",
           fixture_rel(xe_session_logits(s), full_gpu_logits, XE_VOCAB),
           fixture_argmax(xe_session_logits(s), XE_VOCAB)
           == fixture_argmax(full_gpu_logits, XE_VOCAB) ? "same" : "different");
    fixture_session_sync(s, ids, n);
    CHECK_OUTPUTS();
    ok &= fixture_state_equal(s, control, ids, n, "identical reuse", &maximum);

    /* Shortening materializes the shortened final token on GPU, even for
     * one row. Reappend is CPU; this history differs from initial[n-1,1]. */
    fixture_session_sync(s, ids, n - 1);
    outputs++; CHECK_OUTPUTS();
    fixture_control_materialize(control, ids, n - 2, n - 1, 0);
    outputs++; CHECK_OUTPUTS();
    ok &= fixture_state_equal(s, control, ids, n - 1, "shorter GPU1", &maximum);
    fixture_session_sync(s, ids, n);
    outputs++; CHECK_OUTPUTS();
    fixture_control_materialize(control, ids, n - 1, n, 1);
    outputs++; CHECK_OUTPUTS();
    ok &= fixture_state_equal(s, control, ids, n, "reappend CPU1", &maximum);

    int32_t changed[8192];
    memcpy(changed, ids, (size_t)n * sizeof(*ids));
    changed[n - 1] = (changed[n - 1] + 1) % XE_VOCAB;
    fixture_session_sync(s, changed, n);
    outputs++; CHECK_OUTPUTS();
    fixture_control_materialize(control, changed, n - 1, n, 0);
    outputs++; CHECK_OUTPUTS();
    ok &= fixture_state_equal(s, control, changed, n, "diverge GPU1", &maximum);
    fixture_session_sync(s, ids, n);
    outputs++; CHECK_OUTPUTS();
    fixture_control_materialize(control, ids, n - 1, n, 0);
    outputs++; CHECK_OUTPUTS();
    ok &= fixture_state_equal(s, control, ids, n, "restore GPU1", &maximum);

    /* Continue from the independently materialized complete history rather
     * than reset a third session to a different numeric/backend schedule. */
    int32_t expected[8], generated[8], prompt[8200];
    memcpy(prompt, ids, (size_t)n * sizeof(*ids));
    xe_sampler control_greedy = { 0.0f, 0, 1.0f, 1 };
    xe_sampler greedy = { 0.0f, 0, 1.0f, 1 };
    int continuation_ok = 1;
    for (int i = 0; i < 8; i++) {
        expected[i] = xe_session_next(control, &control_greedy);
        generated[i] = xe_session_next(s, &greedy);
        CHECK_OUTPUTS();
        continuation_ok &= expected[i] == generated[i];
        prompt[n + i] = generated[i];
        if (i + 1 < 8) {
            fixture_session_sync(s, prompt, n + i + 1);
            outputs++; CHECK_OUTPUTS();
            fixture_control_materialize(control, prompt, n + i, n + i + 1, 1);
            outputs++; CHECK_OUTPUTS();
            ok &= fixture_state_equal(s, control, prompt, n + i + 1,
                                       "continuation CPU1", &maximum);
        }
    }
    if (!continuation_ok) {
        printf("fixture: greedy got");
        for (int i = 0; i < 8; i++) printf(" %d", generated[i]);
        printf("\n");
    }
    ok &= continuation_ok && output_count_ok && allocations == xe_test_allocations;
    printf("fixture: greedy continuation %s\n", continuation_ok ? "PASS" : "FAIL");
    printf("fixture: session output count %s\n", output_count_ok ? "PASS" : "FAIL");
    printf("fixture: session explicit-schedule append/reuse/shorter/diverge exact rel %.3e %s\n",
           maximum, ok ? "PASS" : "FAIL");
#undef CHECK_OUTPUTS
    xe_session_free(s);
    xe_session_free(control);
    return ok;
}

static int fixture_run(xe_engine *e, const char *name) {
    char ids_path[1024];
    snprintf(ids_path, sizeof ids_path, "tests/golden/%s.ids", name);
    int32_t ids[8192];
    int n = fixture_ids(ids_path, ids, 8192);
    if (n < 2) xe_fatal("fixture requires at least two tokens");

    xe_session *s = xe_session_new(e);
    size_t allocations = xe_test_allocations;
    double start = fixture_now();
    fixture_session_sync(s, ids, n);
    double seconds = fixture_now() - start;
    int finite = 1;
    for (int i = 0; i < XE_VOCAB; i++) finite = finite && isfinite(s->logits[i]);
    int no_alloc = allocations == xe_test_allocations;
    int ok = finite && no_alloc;
    printf("fixture: %s n=%d %.3f s %.3f tok/s finite %s alloc %s %s\n",
           name, n, seconds, (double)n / seconds, finite ? "yes" : "no",
           no_alloc ? "none" : "HOT-PATH", ok ? "PASS" : "FAIL");

    float *sync_logits = NULL;
    if (!strcmp(name, "short")) {
        sync_logits = malloc(XE_VOCAB * sizeof(float));
        if (!sync_logits) xe_fatal("fixture: allocation failed");
        memcpy(sync_logits, s->logits, XE_VOCAB * sizeof(float));
    }

    xe_session_free(s);
    if (sync_logits) {
        ok = fixture_sync(e, ids, n, sync_logits) && ok;
        free(sync_logits);
    }
    return ok;
}

int main(int argc, char **argv) {
    if (argc < 2 || argc > 3 || !argv[1][0] ||
        (argc == 3 && strcmp(argv[2], "short") && strcmp(argv[2], "long"))) {
        fprintf(stderr, "usage: %s <model.gguf> [short|long]\n", argv[0]);
        return 2;
    }
    const char *model = argv[1];
    int context = XE_CONTEXT_MIN;
    const char *names[] = { "short", "long" };
    for (int i = 0; i < 2; i++) {
        if (argc == 3 && strcmp(argv[2], names[i])) continue;
        char path[1024];
        int32_t ids[8192];
        snprintf(path, sizeof path, "tests/golden/%s.ids", names[i]);
        int needed = fixture_ids(path, ids, 8192) + 8;
        if (needed > context) context = needed;
    }
    xe_engine_options options = { .context = context };
    xe_engine *e = xe_engine_open_with_options(model, &options);
    int ok = 1;
    if (argc <= 2 || !strcmp(argv[2], "short")) ok &= fixture_run(e, "short");
    if (argc <= 2 || !strcmp(argv[2], "long")) ok &= fixture_run(e, "long");
    xe_engine_close(e);
    return ok ? 0 : 1;
}
