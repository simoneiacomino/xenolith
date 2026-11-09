#define SNAPSHOT_FIXTURE_NO_MAIN
#include "snapshot_compat_fixture.c"
#include "../kvstore.h"
#include <dirent.h>
#include <sys/wait.h>

static const char *fault_point;
static int crash_at_fault;
static const char *cache_path;
static uint64_t budget;
static int large_mode;

static uint64_t disk_bytes(void) {
    DIR *dir = opendir(cache_path); CHECK(dir);
    uint64_t total = 0; struct dirent *entry;
    while ((entry = readdir(dir))) {
        if (!strcmp(entry->d_name, ".") || !strcmp(entry->d_name, "..")) continue;
        struct stat st;
        CHECK(fstatat(dirfd(dir), entry->d_name, &st, AT_SYMLINK_NOFOLLOW) == 0);
        CHECK(S_ISREG(st.st_mode)); total += (uint64_t)st.st_size;
    }
    CHECK(closedir(dir) == 0); return total;
}
int kvstore_fault(const char *point) {
    static unsigned chunks;
    /* Budget enforcement may start over a newly reduced limit. Save writes,
     * unlike eviction stages, must never exceed the configured budget. */
    if (strcmp(point, "checkpoint-unlinked") && strcmp(point, "reserve-carried") &&
        (!large_mode || strcmp(point, "chunk-written") || chunks++ % 128 == 0))
        CHECK(disk_bytes() <= budget);
    if (!fault_point || strcmp(point, fault_point)) return 0;
    if (crash_at_fault) _exit(99);
    return 1;
}
static kvstore *open_cache(uint64_t limit) {
    kvstore *store = NULL; budget = limit;
    CHECK(kvstore_open(&store, cache_path, limit) == KVSTORE_OK); return store;
}
static uint64_t usage(kvstore *store) {
    uint64_t total, pinned;
    CHECK(kvstore_usage(store, &total, &pinned) == KVSTORE_OK);
    CHECK(total == disk_bytes() && total <= budget); return total;
}
static kvstore_id save(kvstore *store, xe_session *s, int n, int salt, int group) {
    fixture_state(s, n, salt, 0);
    kvstore_save_options options = {.has_group = 1, .rebuild_cost = (uint64_t)n};
    options.group.bytes[0] = (uint8_t)group;
    kvstore_id id; xe_snapshot_status core;
    kvstore_status status = kvstore_save(store, s, &options, &id, &core);
    if (status != KVSTORE_OK) fprintf(stderr, "save n=%d: %s / %s\n", n,
                                     kvstore_status_name(status), xe_snapshot_status_name(core));
    CHECK(status == KVSTORE_OK); usage(store); return id;
}
static void load(kvstore *store, xe_session *s, kvstore_id id, int n, int salt) {
    int32_t *ids = calloc((size_t)n, sizeof *ids); CHECK(ids);
    for (int i = 0; i < n; i++) ids[i] = 100 + i + salt;
    xe_tokens expected = {ids, n, n};
    fixture_state(s, n, salt + 77, 0);
    CHECK(kvstore_load(store, &id, s, &expected, NULL) == KVSTORE_OK);
    fixture_state(s, n, salt, 1); free(ids);
}
static int present(kvstore *store, kvstore_id id) { return kvstore_contains(store, &id) == KVSTORE_OK; }
static void clear_cache(kvstore **store) {
    kvstore_close(*store); *store = open_cache(1);
    CHECK(kvstore_enforce_budget(*store) == KVSTORE_OK); CHECK(usage(*store) == 0);
    kvstore_close(*store); *store = open_cache(UINT64_C(1073741824));
}
static kvstore_id commit(kvstore *store, xe_session *s, int n, int salt, int group) {
    kvstore_id id = save(store, s, n, salt, group);
    CHECK(kvstore_committed(store, &id, s, NULL, NULL) == KVSTORE_OK);
    return id;
}
static int snapshots(void) {
    DIR *dir = opendir(cache_path); CHECK(dir);
    int count = 0; struct dirent *entry;
    while ((entry = readdir(dir))) {
        size_t n = strlen(entry->d_name);
        if (n == 36 && !strcmp(entry->d_name + 32, ".xkv")) count++;
    }
    CHECK(closedir(dir) == 0); return count;
}
/* Deterministic access order, independent of wall clock and filesystem speed.
 * Old hit counters deliberately disagree with recency: LRU must ignore them. */
static void timestamps(kvstore_id id, int64_t created, int64_t used, uint64_t hits) {
    char hex[33], path[4200]; kvstore_id_format(&id, hex);
    snprintf(path, sizeof path, "%s/%s.meta", cache_path, hex);
    int fd = open(path, O_RDWR); CHECK(fd >= 0);
    uint8_t bytes[128]; CHECK(pread(fd, bytes, sizeof bytes, 0) == sizeof bytes);
    format_put_u64le(bytes + 72, hits); format_put_u64le(bytes + 80, 0);
    format_put_u64le(bytes + 96, (uint64_t)created * 1000000000);
    format_put_u64le(bytes + 120, format_crc64(0, bytes, 120));
    CHECK(pwrite(fd, bytes, sizeof bytes, 0) == sizeof bytes); CHECK(close(fd) == 0);
    snprintf(path, sizeof path, "%s/%s.xkv", cache_path, hex);
    struct timespec times[2] = {{.tv_nsec = UTIME_OMIT}, {.tv_sec = used}};
    CHECK(utimensat(AT_FDCWD, path, times, 0) == 0);
}
static int only_eligible(const kvstore_id *id, void *context) {
    return !memcmp(id->bytes, ((kvstore_id *)context)->bytes, sizeof id->bytes);
}
static void retention_cases(kvstore **cache, xe_session *s) {
    kvstore *store = *cache;
    int steps[] = {32, 33, 48, 63, 64, 65, 96, 127, 128, 129, 192, 255, 256};
    kvstore_id milestone32 = {0}, milestone64 = {0}, milestone128 = {0}, previous = {0};
    for (size_t k = 0; k < sizeof steps / sizeof *steps; k++) {
        int n = steps[k]; kvstore_id id = commit(store, s, n, 0, 1);
        if (n == 32) milestone32 = id;
        if (n == 64) milestone64 = id;
        if (n == 128) milestone128 = id;
        CHECK(present(store, id)); CHECK(snapshots() <= 3);
        if (n > 32 && n < 128) CHECK(present(store, milestone32));
        if (n >= 64 && n < 256) CHECK(present(store, milestone64));
        if (n >= 128) CHECK(!present(store, milestone32) && present(store, milestone128));
        if (n == 256) CHECK(!present(store, milestone64));
        if (k && steps[k-1] != 32 && steps[k-1] != 64 && steps[k-1] != 128)
            CHECK(!present(store, previous));
        previous = id;
        /* Reserve selection must survive process/store lifetimes. */
        kvstore_close(store); store = open_cache(UINT64_C(1073741824));
    }
    load(store, s, milestone128, 128, 0);
    puts("cache: frequent saves preserve 32/64/128 milestones across reopen PASS");
    clear_cache(&store);
    kvstore_id a = commit(store, s, 33, 0, 1);
    kvstore_id b = commit(store, s, 70, 0, 1);
    kvstore_id c = commit(store, s, 139, 0, 1);
    CHECK(present(store, a) && present(store, b) && present(store, c));
    kvstore_id d = commit(store, s, 140, 0, 1);
    CHECK(!present(store, a) && present(store, b) && !present(store, c) && present(store, d));
    /* Rewind: longer states disappear only after the new commit. */
    c = commit(store, s, 80, 0, 1);
    CHECK(present(store, b) && present(store, c) && !present(store, d));
    /* Divergence, even at the same boundary: old tokens are not a reserve. */
    d = commit(store, s, 80, 3, 1);
    CHECK(!present(store, b) && !present(store, c) && present(store, d));
    CHECK(snapshots() == 1);
    /* Same token prefix, new epoch: the old snapshot is no longer reachable
     * through transcript restore, so it must not become the reserve. */
    c = save(store, s, 160, 3, 1);
    CHECK(kvstore_committed(store, &c, s, only_eligible, &c) == KVSTORE_OK);
    CHECK(present(store, c) && !present(store, d) && snapshots() == 1);
    puts("cache: sparse thresholds, rewind and diverged-prefix reserves PASS");
    clear_cache(&store);
    a = commit(store, s, 32, 0, 1);
    b = commit(store, s, 64, 0, 1);
    c = commit(store, s, 96, 0, 1);
    d = commit(store, s, 64, 7, 2);
    timestamps(a, 1, 1, UINT64_MAX); timestamps(b, 2, 2, UINT64_MAX);
    timestamps(c, 3, 20, UINT64_MAX); timestamps(d, 4, 30, 0);
    uint64_t before = usage(store);
    kvstore_close(store); store = open_cache(before - 1);
    CHECK(kvstore_enforce_budget(store) == KVSTORE_OK);
    CHECK(present(store, a) && !present(store, b) && !present(store, c) && present(store, d));
    /* Another conversation at time 10 must be older than the remaining
     * reserve: degrading the group must preserve its time 20. */
    timestamps(d, 4, 10, 0);
    before = usage(store);
    kvstore_close(store); store = open_cache(before - 1);
    CHECK(kvstore_enforce_budget(store) == KVSTORE_OK);
    CHECK(present(store, a) && !present(store, d));
    load(store, s, a, 32, 0);
    puts("cache: whole LRU stages preserve conversation recency across reopen PASS");
    clear_cache(&store);
    a = commit(store, s, 32, 0, 1); b = commit(store, s, 32, 7, 2);
    timestamps(a, 1, 1, 0); timestamps(b, 2, 2, UINT64_MAX);
    before = usage(store);
    kvstore_close(store); store = open_cache(before);
    /* A load at an exactly full budget still durably updates recency. */
    load(store, s, a, 32, 0); CHECK(usage(store) == before);
    kvstore_close(store); store = open_cache(before - 1);
    CHECK(kvstore_enforce_budget(store) == KVSTORE_OK);
    CHECK(present(store, a) && !present(store, b));
    puts("cache: successful access at full budget updates persistent LRU PASS");
    clear_cache(&store);
    a = commit(store, s, 32, 0, 1); b = commit(store, s, 64, 0, 1);
    c = commit(store, s, 32, 7, 2);
    timestamps(a, 1, 10, 0); timestamps(b, 2, 10, 0); timestamps(c, 1, 10, 0);
    before = usage(store); kvstore_close(store); store = open_cache(before - 1);
    CHECK(kvstore_enforce_budget(store) == KVSTORE_OK);
    CHECK(present(store, a) && !present(store, b) && present(store, c));
    before = usage(store); kvstore_close(store); store = open_cache(before - 1);
    CHECK(kvstore_enforce_budget(store) == KVSTORE_OK);
    CHECK(!present(store, a) && present(store, c));
    puts("cache: equal-recency ties stay in conversation order across stages PASS");
    clear_cache(&store);
    const char *eviction_faults[] = {"reserve-carried", "checkpoint-unlinked"};
    for (size_t stage = 0; stage < 2; stage++) {
        a = commit(store, s, 32, 0, 1); b = commit(store, s, 64, 0, 1);
        c = commit(store, s, 96, 0, 1); d = commit(store, s, 64, 7, 2);
        timestamps(a, 1, 1, 0); timestamps(b, 2, 2, 0);
        timestamps(c, 3, 20, 0); timestamps(d, 4, 30, 0);
        before = usage(store); kvstore_close(store); store = open_cache(before - 1);
        pid_t child = fork(); CHECK(child >= 0);
        if (!child) {
            fault_point = eviction_faults[stage]; crash_at_fault = 1;
            kvstore_enforce_budget(store); _exit(1);
        }
        int status; CHECK(waitpid(child, &status, 0) == child && WIFEXITED(status) && WEXITSTATUS(status) == 99);
        kvstore_close(store); store = open_cache(before - 1);
        CHECK(kvstore_enforce_budget(store) == KVSTORE_OK);
        CHECK(present(store, a) && !present(store, b) && present(store, d));
        /* A crash after removing future may leave latest alive when the
         * reduced budget already fits. Keeping extra valid state is safe;
         * eviction is not a multi-file transaction. */
        CHECK(stage ? present(store, c) : !present(store, c));
        timestamps(d, 4, 10, 0); before = usage(store);
        kvstore_close(store); store = open_cache(before - 1);
        CHECK(kvstore_enforce_budget(store) == KVSTORE_OK);
        CHECK(present(store, a) && !present(store, d));
        load(store, s, a, 32, 0); clear_cache(&store);
    }
    puts("cache: crashes during LRU reduction preserve reserve and recency PASS");
    clear_cache(&store); *cache = store;
}
static void large_global(xe_session *s, int n, int check) {
    for (int layer = 0; layer < XE_LAYERS; layer++) if (XE_IS_GLOBAL(layer))
        for (int value = 0; value < 2; value++)
            for (int head = 0; head < XE_GLOBAL_KV_HEADS; head++) {
                _Float16 *base = xe_kv_layer_ptr(s, layer, value) +
                    (size_t)head * s->engine->context * XE_GLOBAL_HEAD_DIM;
                for (int pos = 0; pos < n; pos++) for (int d = 0; d < XE_GLOBAL_HEAD_DIM; d++) {
                    uint32_t x = (uint32_t)pos * 0x9e3779b1u ^ (uint32_t)d * 0xc2b2ae35u ^
                                 (uint32_t)(layer * 23 + value * 7 + head);
                    x ^= x >> 16; x *= 0x85ebca6bu; x ^= x >> 13;
                    uint16_t bits = (uint16_t)(0x3c00u + (x & 1023));
                    _Float16 *at = base + (size_t)pos * XE_GLOBAL_HEAD_DIM + d;
                    if (check) CHECK(memcmp(at, &bits, sizeof bits) == 0);
                    else memcpy(at, &bits, sizeof bits);
                }
            }
}
static void large_case(kvstore *store, xe_session *s) {
    kvstore_save_options options = {.has_group = 1}; options.group.bytes[0] = 1;
    kvstore_id ids[2]; uint64_t standalone = 0;
    for (int k = 0; k < 2; k++) {
        int n = k ? 65536 : 32768;
        s->n_tokens = n;
        for (int i = 0; i < n; i++) s->tokens[i] = 100 + i;
        large_global(s, n, 0); options.rebuild_cost = (uint64_t)n;
        uint64_t size; CHECK(xe_session_snapshot_size(s, &size) == XE_SNAPSHOT_OK); standalone += size;
        CHECK(kvstore_save(store, s, &options, &ids[k], NULL) == KVSTORE_OK);
    }
    uint64_t shared = usage(store);
    CHECK(shared < UINT64_C(2) * 1024 * 1024 * 1024);
    CHECK(standalone > budget); /* A full intermediate snapshot would not fit. */
    CHECK(standalone - shared > UINT64_C(639) * 1024 * 1024);
    CHECK(kvstore_remove(store, &ids[0]) == KVSTORE_OK);
    memset(s->global_k, 0, xe_global_slab_elems(s->engine) * sizeof *s->global_k);
    memset(s->global_v, 0, xe_global_slab_elems(s->engine) * sizeof *s->global_v);
    int32_t *expected_ids = malloc(65536 * sizeof *expected_ids); CHECK(expected_ids);
    for (int i = 0; i < 65536; i++) expected_ids[i] = 100+i;
    xe_tokens expected = {expected_ids, 65536, 65536};
    CHECK(kvstore_load(store, &ids[1], s, &expected, NULL) == KVSTORE_OK);
    large_global(s, 65536, 1); free(expected_ids);
    CHECK(kvstore_remove(store, &ids[1]) == KVSTORE_OK); CHECK(usage(store) == 0);
    printf("cache: 32k/64k standalone=%llu shared=%llu saved=%llu peak-budget=%llu exact restore PASS\n",
           (unsigned long long)standalone, (unsigned long long)shared,
           (unsigned long long)(standalone-shared), (unsigned long long)budget);
}
int main(int argc, char **argv) {
    CHECK(argc == 1 || (argc == 2 && !strcmp(argv[1], "--large")));
    large_mode = argc == 2;

    xe_engine e = {.context = large_mode ? 65536 : 2049, .pool_initialized = 1, .owner = pthread_self(), .scalar_rms = 1};
    /* Match the snapshot fixture's fixed identity without probing host CPUs. */
    e.worker_count = XE_WORKERS;
    xe_session s = {.engine = &e};
    s.tokens = calloc((size_t)e.context, sizeof *s.tokens); s.logits = calloc(XE_VOCAB, sizeof *s.logits);
    s.swa_k = calloc(XE_SWA_SLAB_ELEMS, sizeof *s.swa_k); s.swa_v = calloc(XE_SWA_SLAB_ELEMS, sizeof *s.swa_v);
    s.global_k = calloc(xe_global_slab_elems(&e), sizeof *s.global_k); s.global_v = calloc(xe_global_slab_elems(&e), sizeof *s.global_v);
    CHECK(s.tokens && s.logits && s.swa_k && s.swa_v && s.global_k && s.global_v);
    char path[4096]; const char *tmp = getenv("TMPDIR"); if (!tmp) tmp = "/tmp";
    CHECK(snprintf(path, sizeof path, "%s/xenolith-cache-policy-XXXXXX", tmp) < (int)sizeof path);
    CHECK(mkdtemp(path)); cache_path = path;
    kvstore *store = open_cache(large_mode ? UINT64_C(2147483648) : UINT64_C(1073741824));
    if (large_mode) { large_case(store, &s); goto done; }
    retention_cases(&store, &s);
    kvstore_id a = save(store, &s, 256, 0, 1);
    uint64_t raw_a; CHECK(xe_session_snapshot_size(&s, &raw_a) == XE_SNAPSHOT_OK);
    kvstore_id b = save(store, &s, 512, 0, 1);
    uint64_t raw_b; CHECK(xe_session_snapshot_size(&s, &raw_b) == XE_SNAPSHOT_OK);
    uint64_t shared = usage(store);
    CHECK(shared < raw_a + raw_b - UINT64_C(4) * 1024 * 1024);
    printf("cache: 256/512 global sharing saves %llu bytes PASS\n", (unsigned long long)(raw_a + raw_b - shared));
    kvstore_id c = save(store, &s, 768, 0, 1);
    CHECK(kvstore_committed(store, &c, &s, NULL, NULL) == KVSTORE_OK);
    CHECK(present(store, a) && present(store, b) && present(store, c));
    CHECK(kvstore_remove(store, &a) == KVSTORE_OK);
    load(store, &s, c, 768, 0);
    kvstore_id branch = save(store, &s, 769, 3, 1);
    load(store, &s, c, 768, 0); load(store, &s, branch, 769, 3);
    CHECK(kvstore_committed(store, &branch, &s, NULL, NULL) == KVSTORE_OK);
    CHECK(!present(store, c)); load(store, &s, branch, 769, 3);
    puts("cache: independent manifests, branch isolation, post-commit reserve pruning PASS");
    uint64_t before = usage(store);
    kvstore_close(store); store = open_cache(before + 1024 * 1024);
    fixture_state(&s, 770, 9, 0);
    kvstore_save_options opts = {.has_group = 1, .rebuild_cost = 770}; opts.group.bytes[0] = 1;
    kvstore_id rejected;
    CHECK(kvstore_save(store, &s, &opts, &rejected, NULL) == KVSTORE_BUDGET);
    CHECK(usage(store) == before); load(store, &s, branch, 769, 3);
    kvstore_close(store); store = open_cache(0);
    CHECK(disk_bytes() == before);
    CHECK(kvstore_save(store, &s, &opts, &rejected, NULL) == KVSTORE_DISABLED);
    xe_tokens expected = {s.tokens, s.n_tokens, s.n_tokens};
    CHECK(kvstore_load(store, &branch, &s, &expected, NULL) == KVSTORE_DISABLED);
    CHECK(kvstore_enforce_budget(store) == KVSTORE_DISABLED && disk_bytes() == before);
    kvstore_close(store); store = open_cache(UINT64_C(1073741824));
    load(store, &s, branch, 769, 3);
    puts("cache: insufficient budget preserves current points; disabled preserves files PASS");
    clear_cache(&store);
    a = save(store, &s, 256, 0, 1); b = save(store, &s, 512, 0, 1);
    c = save(store, &s, 512, 3, 2);
    before = usage(store);
    kvstore_close(store); store = open_cache(before - raw_a / 2);
    CHECK(kvstore_enforce_budget(store) == KVSTORE_OK);
    CHECK(present(store, a) && !present(store, b) && present(store, c));
    load(store, &s, a, 256, 0); load(store, &s, c, 512, 3); usage(store);
    puts("cache: LRU eviction reduces the oldest conversation to its reserve; physical accounting PASS");
    clear_cache(&store);
    a = save(store, &s, 256, 0, 1);
    before = usage(store);
    const char *faults[] = {"chunk-written", "snapshot-written", "snapshot-synced", "snapshot-renamed", "snapshot-directory-synced"};
    for (size_t i = 0; i < sizeof faults / sizeof *faults; i++) {
        fixture_state(&s, 512, 0, 0); fault_point = faults[i];
        opts.rebuild_cost = 512;
        CHECK(kvstore_save(store, &s, &opts, &rejected, NULL) == KVSTORE_IO);
        fault_point = NULL; CHECK(usage(store) == before); load(store, &s, a, 256, 0);
    }
    for (int stage = 0; stage < 2; stage++) {
        fixture_state(&s, 512, 0, 0);
        pid_t child = fork(); CHECK(child >= 0);
        if (!child) { fault_point = faults[stage ? 3 : 0]; crash_at_fault = 1;
            kvstore_save(store, &s, &opts, &rejected, NULL); _exit(1); }
        int status; CHECK(waitpid(child, &status, 0) == child && WIFEXITED(status) && WEXITSTATUS(status) == 99);
        kvstore_close(store); store = open_cache(UINT64_C(1073741824));
        CHECK(kvstore_enforce_budget(store) == KVSTORE_OK); load(store, &s, a, 256, 0);
        if (!stage) CHECK(usage(store) == before);
    }
    puts("cache: failed writes and process crashes preserve committed checkpoints PASS");
    clear_cache(&store);
    a = save(store, &s, 256, 0, 1);
    DIR *dir = opendir(cache_path); CHECK(dir);
    struct dirent *entry; int corrupted = 0;
    while ((entry = readdir(dir))) {
        size_t len = strlen(entry->d_name);
        if (len < 4 || strcmp(entry->d_name + len - 4, ".xkc")) continue;
        int fd = openat(dirfd(dir), entry->d_name, O_RDWR | O_NOFOLLOW); CHECK(fd >= 0);
        uint8_t byte; CHECK(pread(fd, &byte, 1, 0) == 1); byte ^= 1;
        CHECK(pwrite(fd, &byte, 1, 0) == 1); CHECK(close(fd) == 0);
        corrupted = 1; break;
    }
    CHECK(closedir(dir) == 0 && corrupted);
    b = save(store, &s, 512, 0, 1);
    load(store, &s, a, 256, 0); load(store, &s, b, 512, 0);
    puts("cache: recomputed shared chunk repairs corruption atomically PASS");

    clear_cache(&store);
done:
    CHECK(e.worker_count == XE_WORKERS && e.worker_cpus == NULL);
    kvstore_close(store);
    char lock[4200]; snprintf(lock, sizeof lock, "%s/.lock", path); CHECK(unlink(lock) == 0); CHECK(rmdir(path) == 0);
    free(s.tokens); free(s.logits); free(s.swa_k); free(s.swa_v); free(s.global_k); free(s.global_v);
    return 0;
}
