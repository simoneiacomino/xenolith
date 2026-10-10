#define XE_TEST_SESSION
#define XE_TEST_OUTPUT_COUNT
#include "../xenolith.c"
#include "test_context.h"

#include "../conversation.h"
#include "../kvstore.h"

#include <dirent.h>

size_t xe_test_prefill_batches;
size_t xe_test_decode_tokens;
size_t xe_test_output_calls;

static void counters_reset(void) {
    xe_test_prefill_batches = 0;
    xe_test_decode_tokens = 0;
    xe_test_output_calls = 0;
}

static void remove_tree(const char *path) {
    DIR *directory = opendir(path);
    if (directory) {
        struct dirent *entry;
        while ((entry = readdir(directory)) != NULL) {
            if (!strcmp(entry->d_name, ".") || !strcmp(entry->d_name, ".."))
                continue;
            char child[1024];
            int n = snprintf(child, sizeof child, "%s/%s", path,
                             entry->d_name);
            if (n <= 0 || (size_t)n >= sizeof child) continue;
            struct stat st;
            if (lstat(child, &st) == 0 && S_ISDIR(st.st_mode))
                remove_tree(child);
            else
                unlink(child);
        }
        closedir(directory);
    }
    rmdir(path);
}

int main(int argc, char **argv) {
    int context = test_context_capacity(&argc, argv);
    if (argc < 2 || !argv[1][0]) {
        fprintf(stderr, "usage: %s <model.gguf> [--ctx N]\n", argv[0]);
        return 2;
    }
    const char *model = argv[1];
    xe_engine *engine = xe_engine_open_with_context(model, context);
    xe_session *session = xe_session_new(engine);

    char root[] = "/tmp/xenolith-conv-model-XXXXXX";
    if (!mkdtemp(root)) return 1;
    char state_dir[1080], cache_dir[1080];
    snprintf(state_dir, sizeof state_dir, "%s/state", root);
    snprintf(cache_dir, sizeof cache_dir, "%s/cache", root);

    conversation_store *store = NULL;
    kvstore *cache = NULL;
    int ok = conversation_store_open(&store, state_dir) == CONVERSATION_OK;
    ok &= kvstore_open(&cache, cache_dir, KVSTORE_DEFAULT_BUDGET) ==
          KVSTORE_OK;

    conversation *c = NULL;
    conversation_id id;
    ok &= conversation_create(store, &c, &id) == CONVERSATION_OK;
    conversation_settings settings = { 0.8f, 64, 0.95f, 512,
                                       CONVERSATION_SAMPLER_ABI, 1234, 1234,
                                       CONVERSATION_REASONING_OFF,
                                       CONVERSATION_REASONING_DISCARD, -1 };
    ok &= conversation_append_settings(c, &settings) == CONVERSATION_OK;
    ok &= conversation_append_title(c, "model acceptance") ==
          CONVERSATION_OK;
    ok &= conversation_commit(c) == CONVERSATION_OK;

    int32_t prompt[64];
    for (int i = 0; i < 64; i++) prompt[i] = 2 + i;
    ok &= conversation_append_message(c, CONVERSATION_ROLE_USER, NULL, 0,
                                      "prompt", 6, prompt, 64) ==
          CONVERSATION_OK;
    ok &= conversation_commit(c) == CONVERSATION_OK;

    uint64_t count = 0;
    const int32_t *tokens = conversation_tokens(c, &count);
    ok &= count == 64;
    xe_tokens prefix = { (int32_t *)tokens, 64, 64 };
    xe_session_sync(session, &prefix);

    float *logits64 = malloc(XE_VOCAB * sizeof(*logits64));
    float *logits65 = malloc(XE_VOCAB * sizeof(*logits65));
    ok &= logits64 && logits65;
    memcpy(logits64, xe_session_logits(session),
           XE_VOCAB * sizeof(*logits64));

    kvstore_save_options options;
    memset(&options, 0, sizeof options);
    kvstore_id snapshot64;
    xe_snapshot_status core;
    ok &= kvstore_save(cache, session, &options, &snapshot64, &core) ==
          KVSTORE_OK;
    ok &= conversation_append_snapshot_ref(c, &snapshot64, 64) ==
          CONVERSATION_OK;
    ok &= conversation_commit(c) == CONVERSATION_OK;

    xe_sampler live_sampler = { settings.temperature, settings.top_k,
                                settings.top_p, settings.rng_state };
    int32_t sampled = xe_session_next(session, &live_sampler);
    conversation_generation generation = { 1, settings };
    ok &= conversation_append_generation_started(c, &generation) ==
          CONVERSATION_OK;
    int32_t reply[1] = { sampled };
    ok &= conversation_append_generation_result(
              c, 1, CONVERSATION_STOP_LIMIT, live_sampler.rng_state, NULL, 0,
              "r", 1, reply, 1) == CONVERSATION_OK;
    ok &= conversation_commit(c) == CONVERSATION_OK;

    tokens = conversation_tokens(c, &count);
    ok &= count == 65 && tokens[64] == sampled;
    xe_tokens extended = { (int32_t *)tokens, 65, 65 };
    xe_session_sync(session, &extended);
    memcpy(logits65, xe_session_logits(session),
           XE_VOCAB * sizeof(*logits65));
    xe_sampler live_next = live_sampler;
    int32_t live_second = xe_session_next(session, &live_next);
    conversation_close(c);

    conversation *resumed = NULL;
    ok &= conversation_open(store, &id, &resumed) == CONVERSATION_OK;
    conversation_resume_report report;
    xe_session_reset(session);
    counters_reset();
    ok &= conversation_resume(resumed, cache, session, &report) ==
          CONVERSATION_OK;
    ok &= report.used_snapshot == 1 && report.zero_prefill == 0 &&
          report.boundary == 64;
    ok &= xe_test_prefill_batches == 0 && xe_test_decode_tokens == 1;
    ok &= xe_session_position(session) == 65;
    ok &= memcmp(xe_session_logits(session), logits65,
                 XE_VOCAB * sizeof(*logits65)) == 0;
    conversation_settings restored;
    ok &= conversation_get_settings(resumed, &restored) &&
          restored.rng_state == live_sampler.rng_state &&
          restored.rng_seed == 1234;
    xe_sampler resumed_sampler = { restored.temperature, restored.top_k,
                                   restored.top_p, restored.rng_state };
    int32_t resumed_second = xe_session_next(session, &resumed_sampler);
    ok &= resumed_second == live_second;
    int suffix_ok = ok;
    printf("conversation-model: older-kv suffix resume rng replay %s\n",
           suffix_ok ? "PASS" : "FAIL");

    kvstore_id snapshot65;
    ok &= kvstore_save(cache, session, &options, &snapshot65, &core) ==
          KVSTORE_OK;
    ok &= conversation_append_snapshot_ref(resumed, &snapshot65, 65) ==
          CONVERSATION_OK;
    ok &= conversation_commit(resumed) == CONVERSATION_OK;
    conversation_close(resumed);

    resumed = NULL;
    ok &= conversation_open(store, &id, &resumed) == CONVERSATION_OK;
    xe_session_reset(session);
    counters_reset();
    ok &= conversation_resume(resumed, cache, session, &report) ==
          CONVERSATION_OK;
    ok &= report.used_snapshot == 1 && report.zero_prefill == 1 &&
          report.boundary == 65;
    ok &= xe_test_prefill_batches == 0 && xe_test_decode_tokens == 0 &&
          xe_test_output_calls == 0;
    ok &= xe_session_position(session) == 65;
    ok &= memcmp(xe_session_logits(session), logits65,
                 XE_VOCAB * sizeof(*logits65)) == 0;
    int zero_ok = ok;
    printf("conversation-model: zero-prefill current-kv resume %s\n",
           zero_ok ? "PASS" : "FAIL");

    int32_t greedy_live = 0;
    {
        float best = -INFINITY;
        for (int32_t t = 0; t < XE_VOCAB; t++)
            if (logits65[t] > best) {
                best = logits65[t];
                greedy_live = t;
            }
    }

    ok &= conversation_append_cache_epoch(resumed) == CONVERSATION_OK;
    ok &= conversation_commit(resumed) == CONVERSATION_OK;
    ok &= conversation_snapshot_current(resumed, NULL, NULL) == 0;
    xe_session_reset(session);
    counters_reset();
    ok &= conversation_resume(resumed, cache, session, &report) ==
          CONVERSATION_OK;
    ok &= report.used_snapshot == 0;
    ok &= xe_test_prefill_batches == 1 && xe_test_decode_tokens == 0;
    ok &= xe_session_position(session) == 65;
    xe_sampler greedy = { 0.0f, 1, 1.0f, 1 };
    ok &= xe_session_next(session, &greedy) == greedy_live;
    int strip_ok = ok;
    printf("conversation-model: rebuild after cache epoch strip %s\n",
           strip_ok ? "PASS" : "FAIL");

    kvstore_id snapshot_again;
    ok &= kvstore_save(cache, session, &options, &snapshot_again, &core) ==
          KVSTORE_OK;
    ok &= conversation_append_snapshot_ref(resumed, &snapshot_again, 65) ==
          CONVERSATION_OK;
    ok &= conversation_commit(resumed) == CONVERSATION_OK;

    engine->snapshot_fingerprint[XE_SNAPSHOT_FP_MODEL][0] ^= 1;
    xe_session_reset(session);
    counters_reset();
    ok &= conversation_resume(resumed, cache, session, &report) ==
          CONVERSATION_OK;
    ok &= report.used_snapshot == 0 && xe_test_prefill_batches == 1;
    ok &= xe_session_position(session) == 65;
    ok &= xe_session_next(session, &greedy) == greedy_live;
    engine->snapshot_fingerprint[XE_SNAPSHOT_FP_MODEL][0] ^= 1;

    ok &= kvstore_remove(cache, &snapshot_again) == KVSTORE_OK;
    xe_session_reset(session);
    counters_reset();
    ok &= conversation_resume(resumed, cache, session, &report) ==
          CONVERSATION_OK;
    ok &= report.used_snapshot == 0 && xe_test_prefill_batches == 1;
    ok &= xe_session_next(session, &greedy) == greedy_live;
    int reject_ok = ok;
    printf("conversation-model: rejected and missing kv rebuild %s\n",
           reject_ok ? "PASS" : "FAIL");

    conversation_close(resumed);
    conversation_store_close(store);
    kvstore_close(cache);
    free(logits65);
    free(logits64);
    xe_session_free(session);
    xe_engine_close(engine);
    remove_tree(root);
    printf("conversation-model: %s\n", ok ? "PASS" : "FAIL");
    return ok ? 0 : 1;
}
