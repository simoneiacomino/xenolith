/* Built against both the historical and current engine by check_snapshot_compat.py.
 * Exercise the real serializer/loader on host-backed session buffers. No GPU or
 * model is opened; this checks exact persisted state, not inference equivalence. */
#ifndef SNAPSHOT_ENGINE_SOURCE
#define SNAPSHOT_ENGINE_SOURCE "../xenolith.c"
#endif
#include SNAPSHOT_ENGINE_SOURCE

#define CHECK(expr) do { if (!(expr)) { \
    fprintf(stderr, "%s:%d: %s\n", __FILE__, __LINE__, #expr); exit(1); \
} } while (0)

static uint16_t kv_bits(int layer, int head, int pos, int dim, int value, int salt) {
    return (uint16_t)(0x3c00 + (layer * 31 + head * 7 + pos * 11 + dim +
                               value * 43 + salt * 101) % 1024);
}

static void fixture_state(xe_session *s, int n, int salt, int check) {
    if (check) CHECK(s->n_tokens == n);
    else s->n_tokens = n;
    for (int i = 0; i < n; i++) {
        int32_t token = 100 + i + salt;
        if (check) CHECK(s->tokens[i] == token);
        else s->tokens[i] = token;
    }
    for (int i = 0; i < XE_VOCAB; i++) {
        float logit = (float)((i + salt) % 1024) / 1024.0f;
        if (check) CHECK(s->logits[i] == logit);
        else s->logits[i] = logit;
    }
    for (int layer = 0; layer < XE_LAYERS; layer++) {
        int global = XE_IS_GLOBAL(layer);
        int heads = global ? XE_GLOBAL_KV_HEADS : XE_SWA_KV_HEADS;
        int dim = global ? XE_GLOBAL_HEAD_DIM : XE_SWA_HEAD_DIM;
        int capacity = global ? s->engine->context : XE_SWA_WINDOW;
        int first = global || n < XE_SWA_WINDOW ? 0 : n - XE_SWA_WINDOW;
        for (int value = 0; value < 2; value++) {
            _Float16 *base = xe_kv_layer_ptr(s, layer, value);
            for (int head = 0; head < heads; head++) {
                for (int pos = first; pos < n; pos++) {
                    int slot = global ? pos : pos % XE_SWA_WINDOW;
                    for (int d = 0; d < dim; d++) {
                        _Float16 *at = base + ((size_t)head * capacity + slot) * dim + d;
                        uint16_t bits = kv_bits(layer, head, pos, d, value, salt);
                        if (check) CHECK(memcmp(at, &bits, sizeof bits) == 0);
                        else memcpy(at, &bits, sizeof bits);
                    }
                }
            }
        }
    }
}

#ifndef SNAPSHOT_FIXTURE_NO_MAIN
int main(int argc, char **argv) {
    CHECK(argc == 7);
    int capacity = atoi(argv[4]), n = atoi(argv[5]);
    CHECK(capacity >= 64 && capacity <= 4097 && n > 0 && n <= capacity);
    xe_engine engine = {.context = capacity, .pool_initialized = 1,
                        .owner = pthread_self(), .scalar_rms = 1};
    /* Synthetic header/payload bytes exercise model identity independently of
     * GPU inference and GGUF schema validation, also with historical writers. */
    const char *model_path = getenv("SNAPSHOT_MODEL_FILE");
    if (model_path) {
        FILE *model = fopen(model_path, "rb");
        uint64_t bytes;
        CHECK(model && format_file_size(model, &bytes) && bytes >= 64 && bytes <= SIZE_MAX);
        engine.map_len = (size_t)bytes;
        engine.header_len = 64;
        engine.map = mmap(NULL, engine.map_len, PROT_READ, MAP_PRIVATE, fileno(model), 0);
        CHECK(engine.map != MAP_FAILED && fclose(model) == 0);
        xe_snapshot_fingerprints_init(&engine);
        /* Save/load must use the cached identity, never revisit mapped weights. */
        CHECK(mprotect(engine.map, engine.map_len, PROT_NONE) == 0);
    }
    xe_session s = {.engine = &engine};
    s.tokens = calloc((size_t)capacity, sizeof *s.tokens);
    s.logits = calloc(XE_VOCAB, sizeof *s.logits);
    s.swa_k = calloc(XE_SWA_SLAB_ELEMS, sizeof *s.swa_k);
    s.swa_v = calloc(XE_SWA_SLAB_ELEMS, sizeof *s.swa_v);
    s.global_k = calloc(xe_global_slab_elems(&engine), sizeof *s.global_k);
    s.global_v = calloc(xe_global_slab_elems(&engine), sizeof *s.global_v);
    CHECK(s.tokens && s.logits && s.swa_k && s.swa_v && s.global_k && s.global_v);
    int write = !strcmp(argv[1], "write");
    CHECK(write || !strcmp(argv[1], "read"));
    FILE *file = fopen(argv[2], write ? "w+b" : "rb");
    CHECK(file);
    xe_snapshot_status status;
    if (write) {
        fixture_state(&s, n, 0, 0);
        status = xe_session_snapshot_save(&s, file);
    } else {
        fixture_state(&s, n, 7, 0);
        int32_t *ids = calloc((size_t)n, sizeof *ids);
        CHECK(ids);
        for (int i = 0; i < n; i++) ids[i] = 100 + i;
        xe_tokens expected = {ids, n, n};
        status = xe_session_snapshot_load(&s, file, &expected);
        free(ids);
        if (status == XE_SNAPSHOT_OK) fixture_state(&s, n, 0, 1);
        else if (!strcmp(argv[6], "empty")) CHECK(s.n_tokens == 0);
        else fixture_state(&s, n, 7, 1);
    }
    if (strcmp(xe_snapshot_status_name(status), argv[3])) {
        int saved_errno = errno;
        fprintf(stderr, "%s: expected %s, got %s (offset %lld, errno %d: %s)\n",
                argv[2], argv[3], xe_snapshot_status_name(status),
                (long long)ftello(file), saved_errno, strerror(saved_errno));
        return 1;
    }
    CHECK(fclose(file) == 0);
    free(s.tokens); free(s.logits);
    free(s.swa_k); free(s.swa_v); free(s.global_k); free(s.global_v);
    if (model_path) CHECK(munmap(engine.map, engine.map_len) == 0);
    return 0;
}
#endif
