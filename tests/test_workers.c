/* CPU-only regressions for the real pool, affinity lifecycle and six-task V phase. */
#include "../xenolith.c"
#include <assert.h>
#include <sys/wait.h>

#define CHECK(x) do { if (!(x)) { fprintf(stderr, "workers:%d: %s\n", __LINE__, #x); exit(2); } } while (0)

static int available[CPU_SETSIZE], available_count;
static cpu_set_t original;
static int smt_a = -1, smt_b = -1;

typedef struct { atomic_int hits[257]; atomic_int bad_pin; int rounds; } work;
static void work_phase(xe_session *s, const void *opaque, int lane, int lanes) {
    work *w = (work *)opaque;
    cpu_set_t actual;
    if (pthread_getaffinity_np(pthread_self(), sizeof actual, &actual) || CPU_COUNT(&actual) != 1)
        atomic_store(&w->bad_pin, 1);
    int cpu = sched_getcpu(), found = 0;
    for (int i = 0; i < xe_engine_worker_count(s->engine); i++)
        if (cpu == xe_engine_worker_cpu(s->engine, i)) found++;
    if (found != 1) atomic_store(&w->bad_pin, 1);
    for (int i = 257 * lane / lanes; i < 257 * (lane + 1) / lanes; i++)
        atomic_fetch_add(&w->hits[i], w->rounds);
}

static void pool_case(int count) {
    int cpus[CPU_SETSIZE];
    for (int i = 0; i < count; i++) cpus[i] = available[count - i - 1];
    xe_engine_options options = {.context = 64, .cpus = cpus, .cpu_count = count};
    xe_engine e = {0};
    xe_engine_set_options(&e, &options);
    /* The caller may reuse or release the options and CPU array immediately. */
    cpus[0] = -1;
    options.cpu_count = 0;
    xe_worker_config_init(&e);
    CHECK(xe_engine_worker_count(&e) == count);
    for (int i = 0; i < count; i++) CHECK(xe_engine_worker_cpu(&e, i) == available[count - i - 1]);
    cpu_set_t current;
    CHECK(pthread_getaffinity_np(pthread_self(), sizeof current, &current) == 0);
    CHECK(CPU_EQUAL(&current, &original)); /* discovery must not leave caller pinned */
    xe_worker_pool_init(&e);
    xe_session s = {.engine = &e};
    for (int caller = 0; caller <= 1; caller++) {
        work w = {0};
        for (int i = 0; i < 257; i++) atomic_init(&w.hits[i], 0);
        atomic_init(&w.bad_pin, 0);
        int expected = 0;
        for (int token = 0; token < 4; token++) {
            xe_workers_begin(&e);
            for (int phase = 1; phase <= 32; phase++) {
                w.rounds = phase; expected += phase;
                xe_dispatch(&e, &s, work_phase, &w, caller);
            }
            xe_workers_end(&e);
        }
        CHECK(!atomic_load(&w.bad_pin));
        for (int i = 0; i < 257; i++) CHECK(atomic_load(&w.hits[i]) == expected);
    }
    for (int i = 0; i < count; i++) CHECK(((uintptr_t)&e.phase_done[i] & 63) == 0);

    /* Run the actual six-task attention writer with fewer than six workers.
     * Equal V=1/2 and normalized scores have the analytic output 1/2. */
    size_t bytes = xe_workspace_layout(&s, NULL);
    void *workspace = xe_alloc(NULL, bytes, XE_MEM_HOST);
    memset(workspace, 0, bytes); xe_workspace_layout(&s, workspace);
    s.global_v = xe_alloc(NULL, xe_global_slab_elems(&e) * sizeof(_Float16), XE_MEM_HOST);
    for (size_t i = 0; i < xe_global_slab_elems(&e); i++) s.global_v[i] = (_Float16)0.5f;
    const int keys = 33;
    for (int i = 0; i < XE_Q_HEADS * keys; i++) s.scores[i] = 1.0f / keys;
    memset(s.rope_swa_cos, 0x5a, XE_SWA_HEAD_DIM / 2 * sizeof(float));
    xe_attention_arg a = {.layer = 5, .pos = keys - 1};
    xe_workers_begin(&e);
    xe_dispatch(&e, &s, xe_attention_global_v_phase, &a, 1);
    xe_dispatch(&e, &s, xe_attention_global_reduce_phase, &a, 1);
    xe_workers_end(&e);
    for (int i = 0; i < XE_Q_HEADS * XE_GLOBAL_HEAD_DIM; i++) CHECK(fabsf(s.attn_heads[i] - 0.5f) < 1e-6f);
    const unsigned char *sentinel = (const unsigned char *)s.rope_swa_cos;
    for (size_t i = 0; i < XE_SWA_HEAD_DIM / 2 * sizeof(float); i++) CHECK(sentinel[i] == 0x5a);
    free(s.global_v); free(workspace);
    xe_worker_pool_destroy(&e);
    CHECK(pthread_getaffinity_np(pthread_self(), sizeof current, &current) == 0);
    CHECK(CPU_EQUAL(&current, &original));
    free(e.worker_cpus); free(e.requested_cpus); free(e.gelu_lut);
    printf("workers: count %d, repeated phases, six-task scratch and affinity restore PASS\n", count);
}

static void invalid_config(xe_engine_options options) {
    fflush(NULL);
    pid_t pid = fork(); CHECK(pid >= 0);
    if (!pid) {
        xe_engine e = {0}; xe_engine_set_options(&e, &options);
        xe_worker_config_init(&e); _exit(0);
    }
    int status; CHECK(waitpid(pid, &status, 0) == pid);
    CHECK(WIFEXITED(status) && WEXITSTATUS(status) == 1);
}

static void automatic_case(void) {
    CHECK(setenv("XE_WORKERS", "invalid", 1) == 0);
    CHECK(setenv("XE_CPU_LIST", "invalid", 1) == 0);
    xe_engine e = {.context = 64};
    xe_worker_config_init(&e);
    int count = xe_engine_worker_count(&e);
    CHECK(count > 0 && count <= available_count);
    cpu_set_t after;
    CHECK(pthread_getaffinity_np(pthread_self(), sizeof after, &after) == 0);
    CHECK(CPU_EQUAL(&after, &original));
    for (int i = 0; i < count; i++) {
        int found = 0;
        for (int j = 0; j < available_count; j++) found += e.worker_cpus[i] == available[j];
        CHECK(found == 1);
        for (int j = 0; j < i; j++) CHECK(e.worker_cpus[i] != e.worker_cpus[j]);
    }
    xe_worker_pool_init(&e);
    xe_session s = {.engine = &e};
    work w = {.rounds = 1};
    for (int i = 0; i < 257; i++) atomic_init(&w.hits[i], 0);
    atomic_init(&w.bad_pin, 0);
    xe_workers_begin(&e); xe_dispatch(&e, &s, work_phase, &w, 1); xe_workers_end(&e);
    CHECK(!atomic_load(&w.bad_pin));
    for (int i = 0; i < 257; i++) CHECK(atomic_load(&w.hits[i]) == 1);
    xe_worker_pool_destroy(&e);
    CHECK(pthread_getaffinity_np(pthread_self(), sizeof after, &after) == 0);
    CHECK(CPU_EQUAL(&after, &original));
    free(e.worker_cpus); free(e.requested_cpus); free(e.gelu_lut);
    /* Count-only overrides retain ascending physical IDs, not catalog classes. */
    xe_engine explicit = {0};
    xe_engine_options options = {.workers = 1};
    xe_engine_set_options(&explicit, &options); xe_worker_config_init(&explicit);
    CHECK(explicit.worker_count == 1 && explicit.worker_cpus[0] == available[0]);
    free(explicit.worker_cpus);
    unsetenv("XE_WORKERS");
    printf("workers: automatic selection, live dispatch, affinity restore and count override PASS (%d workers)\n", count);
}

int main(void) {
    CHECK(pthread_getaffinity_np(pthread_self(), sizeof original, &original) == 0);
    int packages[CPU_SETSIZE], cores[CPU_SETSIZE];
    for (int cpu = 0; cpu < CPU_SETSIZE; cpu++) {
        if (!CPU_ISSET(cpu, &original)) continue;
        /* Independently read the physical identity used by the test inventory. */
        char path[160]; int package, core;
        snprintf(path, sizeof path, "/sys/devices/system/cpu/cpu%d/topology/physical_package_id", cpu);
        FILE *f = fopen(path, "r"); if (!f) continue;
        int ok = fscanf(f, "%d", &package) == 1; fclose(f); if (!ok) continue;
        snprintf(path, sizeof path, "/sys/devices/system/cpu/cpu%d/topology/core_id", cpu);
        f = fopen(path, "r"); if (!f) continue;
        ok = fscanf(f, "%d", &core) == 1; fclose(f); if (!ok) continue;
        int duplicate = 0;
        for (int i = 0; i < available_count; i++) if (packages[i] == package && cores[i] == core) {
            duplicate = 1; smt_a = available[i]; smt_b = cpu;
        }
        if (!duplicate) { packages[available_count] = package; cores[available_count] = core; available[available_count++] = cpu; }
    }
    CHECK(available_count > 0);
    int duplicate[] = {available[0], available[0]};
    int bad_cpu[] = {-1};
    invalid_config((xe_engine_options){.workers = -1});
    invalid_config((xe_engine_options){.workers = CPU_SETSIZE + 1});
    invalid_config((xe_engine_options){.cpu_count = -1});
    invalid_config((xe_engine_options){.cpu_count = 1});
    invalid_config((xe_engine_options){.cpus = available});
    invalid_config((xe_engine_options){.cpus = bad_cpu, .cpu_count = 1});
    invalid_config((xe_engine_options){.cpus = duplicate, .cpu_count = 2});
    invalid_config((xe_engine_options){.workers = 2, .cpus = available, .cpu_count = 1});
    if (smt_a >= 0) {
        int siblings[] = {smt_a, smt_b};
        invalid_config((xe_engine_options){.cpus = siblings, .cpu_count = 2});
        puts("workers: explicit SMT siblings rejected PASS");
    } else puts("workers: SMT sibling case SKIP (no allowed sibling pair)");
    /* Requests remain independent and own their lists before deferred resolution. */
    if (available_count > 1) {
        int cpu_a = available[0], cpu_b = available[1];
        xe_engine first = {0}, second = {0};
        xe_engine_options a = {.cpus = &cpu_a, .cpu_count = 1};
        xe_engine_options b = {.cpus = &cpu_b, .cpu_count = 1};
        xe_engine_set_options(&first, &a); xe_engine_set_options(&second, &b);
        cpu_a = cpu_b = -1;
        xe_worker_config_init(&first); xe_worker_config_init(&second);
        CHECK(first.worker_cpus[0] == available[0] && second.worker_cpus[0] == available[1]);
        free(first.requested_cpus); free(first.worker_cpus);
        free(second.requested_cpus); free(second.worker_cpus);
    }
    int counts[] = {1, 2, 4, 6, 12};
    for (size_t i = 0; i < sizeof counts / sizeof *counts; i++)
        if (counts[i] <= available_count) pool_case(counts[i]);
    automatic_case();
    /* An allowed set with a single high-numbered CPU must not demand CPU zero. */
    unsetenv("XE_WORKERS"); unsetenv("XE_CPU_LIST");
    cpu_set_t subset; CPU_ZERO(&subset); CPU_SET(available[available_count - 1], &subset);
    CHECK(pthread_setaffinity_np(pthread_self(), sizeof subset, &subset) == 0);
    xe_engine one = {0}; xe_worker_config_init(&one);
    CHECK(xe_engine_worker_count(&one) == 1);
    CHECK(xe_engine_worker_cpu(&one, 0) == available[available_count - 1]);
    free(one.worker_cpus);
    if (available_count > 1) {
        invalid_config((xe_engine_options){.cpus = available, .cpu_count = 1}); /* CPU exists but is outside this thread's restricted mask. */
    }
    CHECK(pthread_setaffinity_np(pthread_self(), sizeof original, &original) == 0);
    puts("workers: invalid configuration and restricted cpuset PASS");
    return 0;
}
