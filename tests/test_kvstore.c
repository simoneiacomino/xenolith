#define XE_TEST_SESSION
#include "../xenolith.c"

#include "kvstore.h"

#include <dirent.h>

size_t xe_test_prefill_batches;
size_t xe_test_decode_tokens;

static const char *store_fault_point;

int kvstore_fault(const char *point) {
    return store_fault_point && strcmp(store_fault_point, point) == 0;
}

static int32_t store_token(int position, int salt) {
    return 2 + (position * 8191 + salt * 131071) % (XE_VOCAB - 2);
}

static void store_fill(xe_session *session, int n, int salt) {
    session->n_tokens = n;
    for (int i = 0; i < n; i++) session->tokens[i] = store_token(i, salt);
    for (int i = 0; i < XE_VOCAB; i++)
        session->logits[i] = (float)((i + 97 * salt) % 1009) / 13.0f;
    for (int layer = 0; layer < XE_LAYERS; layer++) {
        int global = XE_IS_GLOBAL(layer);
        int heads = global ? XE_GLOBAL_KV_HEADS : XE_SWA_KV_HEADS;
        int dimension = global ? XE_GLOBAL_HEAD_DIM : XE_SWA_HEAD_DIM;
        int capacity = global ? session->engine->context : XE_SWA_WINDOW;
        for (int value = 0; value < 2; value++) {
            _Float16 *base = xe_kv_layer_ptr(session, layer, value);
            for (int head = 0; head < heads; head++) {
                _Float16 *head_base = base +
                    (size_t)head * capacity * dimension;
                for (int position = 0; position < n; position++) {
                    _Float16 v = (_Float16)(salt + value + layer * 0.01f +
                                            head * 0.001f +
                                            position * 0.0001f);
                    for (int d = 0; d < dimension; d++)
                        head_base[(size_t)position * dimension + d] = v;
                }
            }
        }
    }
}

static int store_state_matches(xe_session *session, const int32_t *tokens,
                               int n, int salt) {
    if (session->n_tokens != n ||
        memcmp(session->tokens, tokens, (size_t)n * sizeof(*tokens)) != 0)
        return 0;
    for (int i = 0; i < XE_VOCAB; i++) {
        float want = (float)((i + 97 * salt) % 1009) / 13.0f;
        if (memcmp(&session->logits[i], &want, sizeof want) != 0) return 0;
    }
    return 1;
}

static uint64_t store_state_crc(xe_session *session) {
    uint64_t crc = format_crc64(0, &session->n_tokens,
                                sizeof session->n_tokens);
    crc = format_crc64(crc, session->tokens,
                       (size_t)session->n_tokens * sizeof(*session->tokens));
    return format_crc64(crc, session->logits,
                        XE_VOCAB * sizeof(*session->logits));
}

static void store_key(uint8_t key[32], int value) {
    for (int i = 0; i < 32; i++) key[i] = (uint8_t)(value * 37 + i);
}

static int store_no_temps(const char *cache) {
    DIR *directory = opendir(cache);
    if (!directory) return 0;
    int ok = 1;
    struct dirent *entry;
    while ((entry = readdir(directory)) != NULL)
        if (strncmp(entry->d_name, ".tmp-", 5) == 0) ok = 0;
    closedir(directory);
    return ok;
}

static int store_snapshot_count(const char *cache) {
    DIR *directory = opendir(cache);
    if (!directory) return -1;
    int count = 0;
    struct dirent *entry;
    while ((entry = readdir(directory)) != NULL) {
        size_t length = strlen(entry->d_name);
        if (length == 36 && strcmp(entry->d_name + 32, ".xkv") == 0) count++;
    }
    closedir(directory);
    return count;
}

static int store_snapshot_mode(const char *cache, const kvstore_id *id) {
    char hex[33], path[512];
    kvstore_id_format(id, hex);
    int n = snprintf(path, sizeof path, "%s/%s.xkv", cache, hex);
    if (n < 0 || (size_t)n >= sizeof path) return 0;
    struct stat st;
    return stat(path, &st) == 0 && S_ISREG(st.st_mode) &&
           (st.st_mode & 0777) == 0600;
}

static void store_cleanup_directory(const char *cache) {
    DIR *directory = opendir(cache);
    if (directory) {
        struct dirent *entry;
        while ((entry = readdir(directory)) != NULL) {
            if (!strcmp(entry->d_name, ".") || !strcmp(entry->d_name, ".."))
                continue;
            char path[512];
            int n = snprintf(path, sizeof path, "%s/%s", cache, entry->d_name);
            if (n > 0 && (size_t)n < sizeof path) unlink(path);
        }
        closedir(directory);
    }
    rmdir(cache);
}

static void store_cleanup(const char *root, const char *cache) {
    store_cleanup_directory(cache);
    char transcript[512];
    int n = snprintf(transcript, sizeof transcript, "%s/transcript.bin", root);
    if (n > 0 && (size_t)n < sizeof transcript) unlink(transcript);
    rmdir(root);
}

int main(void) {
    xe_engine engine;
    memset(&engine, 0, sizeof engine);
    engine.context = 4097;
    xe_gpu_init(&engine);
    for (int i = 0; i < XE_SNAPSHOT_FP_COUNT; i++)
        for (int j = 0; j < 32; j++)
            engine.snapshot_fingerprint[i][j] = (uint8_t)(i * 43 + j);
    engine.snapshot_fingerprint_ready = 1;
    xe_session *session = xe_session_new(&engine);
    store_fill(session, 4, 0);
    int32_t expected_ids[4];
    memcpy(expected_ids, session->tokens, sizeof expected_ids);
    xe_tokens expected = { expected_ids, 4, 4 };
    uint64_t snapshot_size = 0;
    int ok = xe_session_snapshot_size(session, &snapshot_size) == XE_SNAPSHOT_OK;

    char root[] = "/tmp/xenolith-kvstore-XXXXXX";
    ok &= mkdtemp(root) != NULL;
    char cache[512], transcript[512];
    int cache_n = snprintf(cache, sizeof cache, "%s/cache", root);
    int transcript_n = snprintf(transcript, sizeof transcript,
                                "%s/transcript.bin", root);
    ok &= cache_n > 0 && (size_t)cache_n < sizeof cache;
    ok &= transcript_n > 0 && (size_t)transcript_n < sizeof transcript;
    int transcript_fd = open(transcript, O_WRONLY | O_CREAT | O_EXCL, 0600);
    ok &= transcript_fd >= 0;
    if (transcript_fd >= 0) {
        ok &= write(transcript_fd, "durable", 7) == 7;
        close(transcript_fd);
    }

    kvstore *store = NULL;
    ok &= kvstore_open(&store, cache, 2 * snapshot_size + 4096) == KVSTORE_OK;
    kvstore_save_options options;
    memset(&options, 0, sizeof options);
    options.has_key = 1;
    options.rebuild_cost = snapshot_size;
    store_key(options.key, 1);
    kvstore_id first;
    xe_snapshot_status core;
    ok &= kvstore_save(store, session, &options, &first, &core) == KVSTORE_OK;
    ok &= store_snapshot_mode(cache, &first) && store_no_temps(cache);
    char first_text[33];
    kvstore_id parsed_first;
    kvstore_id_format(&first, first_text);
    ok &= kvstore_id_parse(first_text, &parsed_first) &&
          memcmp(parsed_first.bytes, first.bytes, 16) == 0;
    kvstore_id immediate;
    kvstore_status immediate_find = kvstore_find(store, options.key, &immediate);
    ok &= immediate_find == KVSTORE_OK &&
          memcmp(immediate.bytes, first.bytes, 16) == 0;
    int publish_ok = ok;
    printf("kvstore: publish find=%s %s\n",
           kvstore_status_name(immediate_find),
           publish_ok ? "PASS" : "FAIL");

    store_fill(session, 4, 9);
    ok &= kvstore_load(store, &first, session, &expected, &core) == KVSTORE_OK;
    int load_ok = ok;
    printf("kvstore: load and hits %s\n",
           load_ok && publish_ok ? "PASS" : "FAIL");
    ok &= core == XE_SNAPSHOT_OK &&
          store_state_matches(session, expected_ids, 4, 0);
    ok &= kvstore_load(store, &first, session, &expected, &core) == KVSTORE_OK;

    store_fill(session, 4, 0);
    kvstore_id second, third;
    store_key(options.key, 2);
    kvstore_status save_second = kvstore_save(store, session, &options,
                                              &second, &core);
    ok &= save_second == KVSTORE_OK;
    store_key(options.key, 3);
    kvstore_status save_third = kvstore_save(store, session, &options,
                                             &third, &core);
    ok &= save_third == KVSTORE_OK;
    kvstore_id found;
    uint8_t key[32];
    store_key(key, 1);
    kvstore_status find_first = kvstore_find(store, key, &found);
    int first_same = find_first == KVSTORE_OK &&
                     memcmp(found.bytes, first.bytes, 16) == 0;
    ok &= first_same;
    store_key(key, 2);
    kvstore_status find_second = kvstore_find(store, key, &found);
    ok &= find_second == KVSTORE_MISS;
    store_key(key, 3);
    kvstore_status find_third = kvstore_find(store, key, &found);
    int third_same = find_third == KVSTORE_OK &&
                     memcmp(found.bytes, third.bytes, 16) == 0;
    ok &= third_same;
    int eviction_ok = ok;
    printf("kvstore: eviction save=%s/%s find=%s/%s/%s same=%d/%d %s\n",
           kvstore_status_name(save_second), kvstore_status_name(save_third),
           kvstore_status_name(find_first), kvstore_status_name(find_second),
           kvstore_status_name(find_third), first_same, third_same,
           eviction_ok && load_ok ? "PASS" : "FAIL");

    options.pinned = 1;
    store_key(options.key, 4);
    kvstore_id system;
    ok &= kvstore_save(store, session, &options, &system, &core) == KVSTORE_OK;
    ok &= kvstore_anchor_set(store, "system", &system) == KVSTORE_OK;
    ok &= kvstore_anchor_get(store, "system", &found) == KVSTORE_OK &&
          memcmp(found.bytes, system.bytes, 16) == 0;
    int pin_ok = ok;
    printf("kvstore: pin %s\n", pin_ok && eviction_ok ? "PASS" : "FAIL");
    ok &= kvstore_remove(store, &system) == KVSTORE_PINNED;

    options.pinned = 0;
    store_key(options.key, 5);
    kvstore_id ordinary;
    ok &= kvstore_save(store, session, &options, &ordinary, &core) == KVSTORE_OK;
    ok &= kvstore_find(store, key, &found) == KVSTORE_MISS;
    ok &= kvstore_anchor_get(store, "system", &found) == KVSTORE_OK &&
          memcmp(found.bytes, system.bytes, 16) == 0;

    uint64_t total = 0, pinned = 0;
    ok &= kvstore_usage(store, &total, &pinned) == KVSTORE_OK;
    ok &= total <= 2 * snapshot_size + 4096 && pinned == snapshot_size;
    engine.snapshot_fingerprint[XE_SNAPSHOT_FP_MODEL][0] ^= 1;
    uint64_t before = store_state_crc(session);
    ok &= kvstore_load(store, &system, session, &expected, &core) ==
          KVSTORE_REJECTED;
    ok &= core == XE_SNAPSHOT_MODEL_MISMATCH &&
          store_state_crc(session) == before;
    int mismatch_ok = ok;
    printf("kvstore: mismatch %s\n",
           mismatch_ok && pin_ok ? "PASS" : "FAIL");
    engine.snapshot_fingerprint[XE_SNAPSHOT_FP_MODEL][0] ^= 1;
    options.pinned = 1;
    store_key(options.key, 6);
    kvstore_id replacement;
    ok &= kvstore_save(store, session, &options, &replacement, &core) ==
          KVSTORE_OK;
    ok &= kvstore_anchor_set(store, "system", &replacement) == KVSTORE_OK;
    ok &= kvstore_remove(store, &system) == KVSTORE_OK;
    ok &= kvstore_anchor_get(store, "system", &found) == KVSTORE_OK &&
          memcmp(found.bytes, replacement.bytes, 16) == 0;

    struct stat transcript_st;
    ok &= stat(transcript, &transcript_st) == 0 &&
          transcript_st.st_size == 7 && store_no_temps(cache);
    kvstore_close(store);

    char fault_cache[512];
    int fault_n = snprintf(fault_cache, sizeof fault_cache, "%s/fault-cache", root);
    ok &= fault_n > 0 && (size_t)fault_n < sizeof fault_cache;
    store = NULL;
    ok &= kvstore_open(&store, fault_cache, 10 * snapshot_size) == KVSTORE_OK;
    store_fill(session, 4, 0);
    options.pinned = 0;
    store_key(options.key, 8);
    kvstore_id baseline;
    ok &= kvstore_save(store, session, &options, &baseline, &core) == KVSTORE_OK;
    static const char *faults[] = {
        "snapshot-written", "snapshot-synced", "snapshot-renamed",
        "snapshot-directory-synced"
    };
    for (size_t i = 0; i < sizeof faults / sizeof faults[0]; i++) {
        store_fault_point = faults[i];
        kvstore_id failed;
        ok &= kvstore_save(store, session, &options, &failed, &core) == KVSTORE_IO;
        ok &= store_snapshot_count(fault_cache) == 1 &&
              store_no_temps(fault_cache);
    }
    store_fault_point = NULL;
    ok &= kvstore_load(store, &baseline, session, &expected, &core) == KVSTORE_OK;
    kvstore_close(store);
    store_cleanup_directory(fault_cache);

    store = NULL;
    ok &= kvstore_open(&store, cache, 0) == KVSTORE_OK;
    ok &= kvstore_save(store, session, &options, &ordinary, &core) ==
          KVSTORE_DISABLED;
    ok &= stat(transcript, &transcript_st) == 0 && transcript_st.st_size == 7;
    kvstore_close(store);

    printf("kvstore: atomic ids hits eviction pin isolation %s\n",
           ok ? "PASS" : "FAIL");
    xe_session_free(session);
    xe_worker_pool_destroy(&engine);
    xe_gpu_destroy(&engine);
    store_cleanup(root, cache);
    return ok ? 0 : 1;
}
