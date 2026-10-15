/* Separate processes write with the historical engine and continue with the
 * current engine. The writer's live continuation supplies the exact oracle. */
#define XE_TEST_SESSION
#define XE_TEST_OUTPUT_COUNT
#ifndef SNAPSHOT_ENGINE_SOURCE
#define SNAPSHOT_ENGINE_SOURCE "../xenolith.c"
#endif
#include SNAPSHOT_ENGINE_SOURCE

size_t xe_test_prefill_batches;
size_t xe_test_decode_tokens;

#define CHECK(expr) do { if (!(expr)) { \
    fprintf(stderr, "%s:%d: %s\n", __FILE__, __LINE__, #expr); exit(1); \
} } while (0)

static void check_logits(xe_session *s, FILE *oracle, int writing) {
    const float *actual = xe_session_logits(s);
    for (int i = 0; i < XE_VOCAB; i++) CHECK(isfinite(actual[i]));
    if (writing) {
        CHECK(fwrite(actual, sizeof(float), XE_VOCAB, oracle) == XE_VOCAB);
    } else {
        float *expected = malloc(XE_VOCAB * sizeof(*expected));
        CHECK(expected);
        CHECK(fread(expected, sizeof(float), XE_VOCAB, oracle) == XE_VOCAB);
        CHECK(memcmp(actual, expected, XE_VOCAB * sizeof(*expected)) == 0);
        free(expected);
    }
}

int main(int argc, char **argv) {
    CHECK(argc == 4 || argc == 5);
    int writing = !strcmp(argv[1], "write");
    CHECK(writing || !strcmp(argv[1], "read"));
    int replay = argc == 5 && !strcmp(argv[4], "model-mismatch");
    CHECK(argc == 4 || (!writing && replay));
    xe_engine *engine = xe_engine_open_with_context(argv[2], 4097);
    xe_session *session = xe_session_new(engine);
    const int lengths[] = {64, 1057};
    for (size_t c = 0; c < sizeof lengths / sizeof *lengths; c++) {
        int n = lengths[c];
        int32_t ids[1080];
        for (int i = 0; i < 1080; i++) ids[i] = 2 + i;
        xe_tokens prefix = {ids, n, 1080};
        xe_session_reset(session);
        char path[4096];
        CHECK(snprintf(path, sizeof path, "%s/%d.xkv", argv[3], n) < (int)sizeof path);
        FILE *snapshot = fopen(path, writing ? "wb" : "rb");
        CHECK(snapshot);
        CHECK(snprintf(path, sizeof path, "%s/%d.logits", argv[3], n) < (int)sizeof path);
        FILE *oracle = fopen(path, writing ? "wb" : "rb");
        CHECK(oracle);
        if (writing) {
            xe_session_sync(session, &prefix);
            CHECK(xe_session_snapshot_save(session, snapshot) == XE_SNAPSHOT_OK);
        } else {
            xe_test_prefill_batches = xe_test_decode_tokens = xe_test_output_calls = 0;
            CHECK(xe_session_snapshot_load(session, snapshot, &prefix) ==
                  (replay ? XE_SNAPSHOT_MODEL_MISMATCH : XE_SNAPSHOT_OK));
            CHECK(xe_test_prefill_batches == 0 && xe_test_decode_tokens == 0 &&
                  xe_test_output_calls == 0);
            if (replay) {
                CHECK(xe_session_position(session) == 0);
                xe_session_sync(session, &prefix);
                CHECK(xe_test_prefill_batches > 0);
            }
            CHECK(xe_session_position(session) == n);
        }
        CHECK(fclose(snapshot) == 0);
        check_logits(session, oracle, writing);
        /* Greedy CPU decode, a GPU extension, then eight more decode steps. */
        for (int step = 0; step < 10; step++) {
            xe_sampler greedy = {0.0f, 1, 1.0f, 1};
            ids[prefix.len++] = xe_session_next(session, &greedy);
            if (step == 1) prefix.len += 6;
            xe_test_prefill_batches = xe_test_decode_tokens = 0;
            xe_session_sync(session, &prefix);
            CHECK(xe_test_prefill_batches == (size_t)(step == 1));
            CHECK(xe_test_decode_tokens == (size_t)(step != 1));
            check_logits(session, oracle, writing);
        }
        if (!writing) CHECK(fgetc(oracle) == EOF && !ferror(oracle));
        CHECK(fclose(oracle) == 0);
        printf("snapshot model compatibility: %s N=%d exact logits, CPU decode, GPU extension PASS\n",
               replay ? "replay" : argv[1], n);
        fflush(stdout);
    }
    xe_session_free(session);
    xe_engine_close(engine);
    return 0;
}
