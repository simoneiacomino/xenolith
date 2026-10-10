#define XE_TEST_SESSION
#define XE_TEST_OUTPUT_COUNT
#include "../xenolith.c"
#include "test_context.h"

size_t xe_test_prefill_batches;
size_t xe_test_decode_tokens;
size_t xe_test_output_calls;

int main(int argc, char **argv) {
    int context = test_context_capacity(&argc, argv);
    if (argc < 2 || !argv[1][0]) {
        fprintf(stderr, "usage: %s <model.gguf> [--ctx N]\n", argv[0]);
        return 2;
    }
    const char *model = argv[1];
    xe_engine *engine = xe_engine_open_with_context(model, context);
    xe_session *session = xe_session_new(engine);
    int32_t ids[72];
    for (int i = 0; i < 64; i++) ids[i] = 2 + i;
    xe_tokens prefix = { ids, 64, 72 };
    xe_session_sync(session, &prefix);

    float *checkpoint_logits = malloc(XE_VOCAB * sizeof(*checkpoint_logits));
    float *continuation_logits = malloc(XE_VOCAB * sizeof(*continuation_logits));
    int ok = checkpoint_logits != NULL && continuation_logits != NULL;
    if (ok) memcpy(checkpoint_logits, xe_session_logits(session),
                   XE_VOCAB * sizeof(*checkpoint_logits));

    FILE *file = tmpfile();
    ok &= file != NULL;
    uint64_t snapshot_size = 0;
    if (ok) ok &= xe_session_snapshot_size(session, &snapshot_size) ==
                  XE_SNAPSHOT_OK;
    /* 3.7 finding 1: the first checkpoint of a fresh process must cost
     * the same as the second (the MODEL fingerprint is no longer a lazy
     * SHA-256 of the whole GGUF). */
    struct timespec t0, t1, t2;
    clock_gettime(CLOCK_MONOTONIC, &t0);
    if (ok) ok &= xe_session_snapshot_save(session, file) == XE_SNAPSHOT_OK;
    clock_gettime(CLOCK_MONOTONIC, &t1);
    FILE *second = tmpfile();
    ok &= second != NULL;
    if (ok) ok &= xe_session_snapshot_save(session, second) == XE_SNAPSHOT_OK;
    clock_gettime(CLOCK_MONOTONIC, &t2);
    if (second) fclose(second);
    double first_ms = (double)(t1.tv_sec - t0.tv_sec) * 1e3 + (double)(t1.tv_nsec - t0.tv_nsec) / 1e6;
    double second_ms = (double)(t2.tv_sec - t1.tv_sec) * 1e3 + (double)(t2.tv_nsec - t1.tv_nsec) / 1e6;
    printf("snapshot model: first save %.1f ms, second save %.1f ms\n", first_ms, second_ms);
    ok &= first_ms < 1000.0;

    xe_sampler greedy = { 0.0f, 1, 1.0f, 1 };
    int32_t live_next = ok ? xe_session_next(session, &greedy) : -1;
    ids[64] = live_next;
    prefix.len = 65;
    if (ok) xe_session_sync(session, &prefix);
    if (ok) memcpy(continuation_logits, xe_session_logits(session),
                   XE_VOCAB * sizeof(*continuation_logits));

    prefix.len = 64;
    xe_test_prefill_batches = 0;
    xe_test_decode_tokens = 0;
    xe_test_output_calls = 0;
    xe_snapshot_status status = ok
        ? xe_session_snapshot_load(session, file, &prefix)
        : XE_SNAPSHOT_IO;
    ok &= status == XE_SNAPSHOT_OK;
    ok &= xe_test_prefill_batches == 0 && xe_test_decode_tokens == 0 &&
          xe_test_output_calls == 0;
    ok &= xe_session_position(session) == 64;
    ok &= memcmp(xe_session_logits(session), checkpoint_logits,
                 XE_VOCAB * sizeof(*checkpoint_logits)) == 0;
    xe_sampler restored_greedy = { 0.0f, 1, 1.0f, 999 };
    int32_t restored_next = ok ? xe_session_next(session, &restored_greedy) : -2;
    ok &= restored_next == live_next;
    prefix.len = 65;
    if (ok) xe_session_sync(session, &prefix);
    ok &= memcmp(xe_session_logits(session), continuation_logits,
                 XE_VOCAB * sizeof(*continuation_logits)) == 0;

    float *extension_logits = malloc(XE_VOCAB * sizeof(*extension_logits));
    ok &= extension_logits != NULL;
    for (int i = 65; i < 72; i++) ids[i] = 1000 + i;
    prefix.len = 72;
    xe_test_prefill_batches = 0;
    if (ok) xe_session_sync(session, &prefix);
    ok &= xe_test_prefill_batches == 1;
    if (ok) memcpy(extension_logits, xe_session_logits(session),
                   XE_VOCAB * sizeof(*extension_logits));

    xe_session_reset(session);
    prefix.len = 64;
    if (ok) xe_session_sync(session, &prefix);
    int deterministic = ok && memcmp(xe_session_logits(session),
                                     checkpoint_logits,
                                     XE_VOCAB * sizeof(*checkpoint_logits)) == 0;
    ok &= deterministic;
    prefix.len = 65;
    if (ok) xe_session_sync(session, &prefix);
    prefix.len = 72;
    if (ok) xe_session_sync(session, &prefix);
    ok &= memcmp(xe_session_logits(session), extension_logits,
                 XE_VOCAB * sizeof(*extension_logits)) == 0;

    printf("snapshot-model: %llu bytes next=%d deterministic=%d zero-prefill "
           "exact-continuation gpu-extension %s\n",
           (unsigned long long)snapshot_size, live_next, deterministic,
           ok ? "PASS" : "FAIL");
    if (file) fclose(file);
    free(extension_logits);
    free(continuation_logits);
    free(checkpoint_logits);
    xe_session_free(session);
    xe_engine_close(engine);
    return ok ? 0 : 1;
}
