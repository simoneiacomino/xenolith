#define XE_BENCH_TG_PROFILE
#include "../xenolith.c"

#include <dirent.h>
#include <limits.h>
#include <linux/perf_event.h>
#include <sys/resource.h>
#include <sys/ioctl.h>
#include <sys/syscall.h>

#define TG_PMU_COUNT 5

enum {
    TG_IMC0_READ,
    TG_IMC0_WRITE,
    TG_IMC1_READ,
    TG_IMC1_WRITE,
    TG_ENERGY_PKG
};

typedef struct {
    int fd[TG_PMU_COUNT];
    double energy_scale;
    char error[128];
} tg_pmu;

typedef struct {
    double count[TG_PMU_COUNT];
} tg_pmu_result;

typedef struct {
    struct rusage usage;
    long core_count;
    long core_ms;
    long package_count;
    long package_ms;
    long rss_kb;
    long anon_kb;
    long swap_kb;
    long huge_kb;
} tg_snapshot;

typedef struct {
    pthread_t thread;
    const xe_engine *engine;
    cpu_set_t allowed_cpus;
    atomic_int stop;
    atomic_int rep;
    long freq_sum[20];
    long freq_min[20];
    long freq_max[20];
    long freq_samples[20];
    long temp_sum[20];
    long temp_max[20];
    long temp_samples[20];
    char temp_path[PATH_MAX];
} tg_sampler;

static double tg_now(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec + (double)ts.tv_nsec * 1e-9;
}

static int tg_compare(const void *a, const void *b) {
    double x = *(const double *)a;
    double y = *(const double *)b;
    return (x > y) - (x < y);
}

static long tg_read_long(const char *path) {
    int fd = open(path, O_RDONLY);
    if (fd < 0) return -1;
    char buf[64];
    ssize_t n = read(fd, buf, sizeof(buf) - 1);
    close(fd);
    if (n <= 0) return -1;
    buf[n] = '\0';
    char *end;
    long value = strtol(buf, &end, 10);
    return end == buf ? -1 : value;
}

static double tg_read_double(const char *path) {
    FILE *file = fopen(path, "r");
    if (!file) return -1.0;
    double value = -1.0;
    if (fscanf(file, "%lf", &value) != 1) value = -1.0;
    fclose(file);
    return value;
}

static int tg_perf_open(uint32_t type, uint64_t config) {
    struct perf_event_attr attr = {0};
    attr.type = type;
    attr.size = sizeof attr;
    attr.config = config;
    attr.disabled = 1;
    attr.read_format = PERF_FORMAT_TOTAL_TIME_ENABLED | PERF_FORMAT_TOTAL_TIME_RUNNING;
    return (int)syscall(SYS_perf_event_open, &attr, -1, 0, -1, 0);
}

static tg_pmu tg_pmu_open(void) {
    tg_pmu pmu = {0};
    for (int i = 0; i < TG_PMU_COUNT; i++) pmu.fd[i] = -1;
    long imc0 = tg_read_long("/sys/bus/event_source/devices/uncore_imc_free_running_0/type");
    long imc1 = tg_read_long("/sys/bus/event_source/devices/uncore_imc_free_running_1/type");
    long power = tg_read_long("/sys/bus/event_source/devices/power/type");
    pmu.energy_scale = tg_read_double("/sys/bus/event_source/devices/power/events/energy-pkg.scale");
    if (imc0 < 0 || imc1 < 0 || power < 0 || pmu.energy_scale <= 0.0) {
        snprintf(pmu.error, sizeof pmu.error, "PMU metadata unavailable");
        return pmu;
    }
    uint32_t types[TG_PMU_COUNT] = {
        (uint32_t)imc0, (uint32_t)imc0, (uint32_t)imc1, (uint32_t)imc1,
        (uint32_t)power
    };
    uint64_t configs[TG_PMU_COUNT] = { 0x20ff, 0x30ff, 0x20ff, 0x30ff, 0x02 };
    for (int i = 0; i < TG_PMU_COUNT; i++) {
        pmu.fd[i] = tg_perf_open(types[i], configs[i]);
        if (pmu.fd[i] >= 0) continue;
        snprintf(pmu.error, sizeof pmu.error, "perf_event_open: %s", strerror(errno));
        for (int j = 0; j < i; j++) {
            close(pmu.fd[j]);
            pmu.fd[j] = -1;
        }
        return pmu;
    }
    return pmu;
}

static int tg_pmu_available(const tg_pmu *pmu) {
    return pmu->fd[0] >= 0;
}

static void tg_pmu_start(tg_pmu *pmu) {
    if (!tg_pmu_available(pmu)) return;
    for (int i = 0; i < TG_PMU_COUNT; i++) {
        ioctl(pmu->fd[i], PERF_EVENT_IOC_RESET, 0);
        ioctl(pmu->fd[i], PERF_EVENT_IOC_ENABLE, 0);
    }
}

static tg_pmu_result tg_pmu_stop(tg_pmu *pmu) {
    tg_pmu_result result = {0};
    if (!tg_pmu_available(pmu)) return result;
    for (int i = 0; i < TG_PMU_COUNT; i++) {
        uint64_t values[3] = {0};
        ioctl(pmu->fd[i], PERF_EVENT_IOC_DISABLE, 0);
        if (read(pmu->fd[i], values, sizeof values) != sizeof values || !values[2]) {
            snprintf(pmu->error, sizeof pmu->error, "PMU counter read failed: %s", strerror(errno));
            for (int j = 0; j < TG_PMU_COUNT; j++) {
                close(pmu->fd[j]);
                pmu->fd[j] = -1;
            }
            return result;
        }
        result.count[i] = (double)values[0] * (double)values[1] / (double)values[2];
    }
    return result;
}

static void tg_pmu_close(tg_pmu *pmu) {
    for (int i = 0; i < TG_PMU_COUNT; i++)
        if (pmu->fd[i] >= 0) close(pmu->fd[i]);
}

static int tg_read_text(const char *path, char *out, size_t size) {
    int fd = open(path, O_RDONLY);
    if (fd < 0) return 0;
    ssize_t n = read(fd, out, size - 1);
    close(fd);
    if (n <= 0) return 0;
    while (n > 0 && (out[n - 1] == '\n' || out[n - 1] == '\r')) n--;
    out[n] = '\0';
    return 1;
}

static void tg_find_package_temp(char *out, size_t size) {
    out[0] = '\0';
    for (int i = 0; i < 64; i++) {
        char path[PATH_MAX];
        char name[64];
        snprintf(path, sizeof path, "/sys/class/hwmon/hwmon%d/name", i);
        if (!tg_read_text(path, name, sizeof name) || strcmp(name, "coretemp") != 0) continue;
        snprintf(path, sizeof path, "/sys/class/hwmon/hwmon%d/temp1_input", i);
        if (tg_read_long(path) >= 0) {
            snprintf(out, size, "%s", path);
            return;
        }
    }
}

static void *tg_sampler_main(void *opaque) {
    tg_sampler *sampler = opaque;
    cpu_set_t spare = sampler->allowed_cpus;
    int workers = xe_engine_worker_count(sampler->engine);
    for (int lane = 0; lane < workers; lane++) {
        int cpu = xe_engine_worker_cpu(sampler->engine, lane);
        if (cpu >= 0) CPU_CLR(cpu, &spare);
    }
    /* The new thread inherited the owner's pin. Prefer other allowed logical
     * CPUs for telemetry; otherwise restore the original allowed set. */
    const cpu_set_t *allowed = CPU_COUNT(&spare) ? &spare : &sampler->allowed_cpus;
    int rc = pthread_setaffinity_np(pthread_self(), sizeof *allowed, allowed);
    if (rc != 0) xe_fatal("tg128: sampler affinity: %s", strerror(rc));
    struct timespec delay = { 0, 50000000 };
    while (!atomic_load_explicit(&sampler->stop, memory_order_acquire)) {
        int rep = atomic_load_explicit(&sampler->rep, memory_order_acquire);
        if (rep >= 0) {
            long freq_total = 0;
            int freq_count = 0;
            for (int lane = 0; lane < workers; lane++) {
                int cpu = xe_engine_worker_cpu(sampler->engine, lane);
                if (cpu < 0) continue;
                char path[PATH_MAX];
                snprintf(path, sizeof path,
                         "/sys/devices/system/cpu/cpu%d/cpufreq/scaling_cur_freq",
                         cpu);
                long value = tg_read_long(path);
                if (value >= 0) {
                    freq_total += value;
                    freq_count++;
                }
            }
            if (freq_count) {
                long value = freq_total / freq_count;
                sampler->freq_sum[rep] += value;
                if (value < sampler->freq_min[rep]) sampler->freq_min[rep] = value;
                if (value > sampler->freq_max[rep]) sampler->freq_max[rep] = value;
                sampler->freq_samples[rep]++;
            }
            long temp = sampler->temp_path[0] ? tg_read_long(sampler->temp_path) : -1;
            if (temp >= 0) {
                sampler->temp_sum[rep] += temp;
                if (temp > sampler->temp_max[rep]) sampler->temp_max[rep] = temp;
                sampler->temp_samples[rep]++;
            }
        }
        nanosleep(&delay, NULL);
    }
    return NULL;
}

static void tg_read_memory(tg_snapshot *snapshot) {
    FILE *fp = fopen("/proc/self/status", "r");
    if (fp) {
        char line[256];
        while (fgets(line, sizeof line, fp)) {
            if (sscanf(line, "VmRSS: %ld kB", &snapshot->rss_kb) == 1) continue;
            if (sscanf(line, "RssAnon: %ld kB", &snapshot->anon_kb) == 1) continue;
            if (sscanf(line, "VmSwap: %ld kB", &snapshot->swap_kb) == 1) continue;
        }
        fclose(fp);
    }
    fp = fopen("/proc/self/smaps_rollup", "r");
    if (fp) {
        char line[256];
        while (fgets(line, sizeof line, fp))
            if (sscanf(line, "AnonHugePages: %ld kB", &snapshot->huge_kb) == 1) break;
        fclose(fp);
    }
}

static tg_snapshot tg_take_snapshot(const xe_engine *e) {
    tg_snapshot snapshot = {0};
    getrusage(RUSAGE_SELF, &snapshot.usage);
    for (int lane = 0; lane < xe_engine_worker_count(e); lane++) {
        int cpu = xe_engine_worker_cpu(e, lane);
        if (cpu < 0) continue;
        char path[PATH_MAX];
        snprintf(path, sizeof path,
                 "/sys/devices/system/cpu/cpu%d/thermal_throttle/core_throttle_count", cpu);
        long value = tg_read_long(path);
        if (value >= 0) snapshot.core_count += value;
        snprintf(path, sizeof path,
                 "/sys/devices/system/cpu/cpu%d/thermal_throttle/core_throttle_total_time_ms", cpu);
        value = tg_read_long(path);
        if (value >= 0) snapshot.core_ms += value;
    }
    int cpu = xe_engine_worker_cpu(e, 0);
    char path[PATH_MAX];
    snprintf(path, sizeof path,
             "/sys/devices/system/cpu/cpu%d/thermal_throttle/package_throttle_count", cpu);
    snapshot.package_count = tg_read_long(path);
    snprintf(path, sizeof path,
             "/sys/devices/system/cpu/cpu%d/thermal_throttle/package_throttle_total_time_ms", cpu);
    snapshot.package_ms = tg_read_long(path);
    tg_read_memory(&snapshot);
    return snapshot;
}

static void tg_print_environment(const xe_engine *e, int cooldown) {
    long ac = tg_read_long("/sys/class/power_supply/ACAD/online");
    long pl1 = tg_read_long("/sys/devices/virtual/powercap/intel-rapl-mmio/intel-rapl-mmio:0/constraint_0_power_limit_uw");
    long pl1_window = tg_read_long("/sys/devices/virtual/powercap/intel-rapl-mmio/intel-rapl-mmio:0/constraint_0_time_window_us");
    long pl2 = tg_read_long("/sys/devices/virtual/powercap/intel-rapl-mmio/intel-rapl-mmio:0/constraint_1_power_limit_uw");
    int workers = xe_engine_worker_count(e);
    printf("tg128: workers %d cpus", workers);
    for (int lane = 0; lane < workers; lane++) printf(" %d", xe_engine_worker_cpu(e, lane));
    printf("\n");
    printf("tg128: AC %ld PL1 %.1f W window %.3f s PL2 %.1f W repack %.3f s cooldown %d s\n",
           ac, pl1 / 1e6, pl1_window / 1e6, pl2 / 1e6, e->repack_seconds, cooldown);
#ifdef XE_REPACK_DROP_SOURCE
    printf("tg128: source-pages drop\n");
#else
    printf("tg128: source-pages normal\n");
#endif
#ifdef XE_REPACK_HUGEPAGE
    printf("tg128: repack hugepage advised\n");
#elif defined(XE_REPACK_NOHUGEPAGE)
    printf("tg128: repack hugepage disabled\n");
#else
    printf("tg128: repack hugepage default\n");
#endif
}

static int tg_router_ids(const char *path, int32_t *ids, int limit) {
    FILE *file = fopen(path, "r");
    if (!file) xe_fatal("%s: %s", path, strerror(errno));
    int count = 0;
    while (count < limit) {
        int value;
        if (fscanf(file, "%d", &value) != 1) break;
        ids[count++] = value;
        int separator = fgetc(file);
        if (separator == EOF) break;
        if (separator != ',') ungetc(separator, file);
    }
    fclose(file);
    return count;
}

static int tg_router_trace(const char *model, const char *ids_path, int limit) {
    if (limit < 1 || limit > 8192) return 2;
    int32_t *ids = xe_alloc(NULL, (size_t)limit * sizeof(*ids), XE_MEM_HOST);
    int count = tg_router_ids(ids_path, ids, limit);
    if (count != limit) xe_fatal("router trace requested %d tokens, found %d", limit, count);
    int routes[XE_LAYERS][XE_EXPERTS];
    memset(routes, 0, sizeof routes);
    xe_engine *engine = xe_engine_open(model);
    xe_session *session = xe_session_new(engine);
    double start = tg_now();
    for (int position = 0; position < count; position++) {
        xe_decode_token_mode(session, ids[position], position,
                             XE_MOE_GENERIC_BATCHED, 1,
                             XE_SOFTCAP_SECOND_LOOP, 0);
        for (int layer = 0; layer < XE_LAYERS; layer++)
            for (int slot = 0; slot < XE_EXPERTS_USED; slot++)
                routes[layer][session->expert_trace[layer][slot]]++;
    }
    double seconds = tg_now() - start;
    int route_count = count * XE_EXPERTS_USED;
    int tile_min = INT_MAX;
    int tile_max = 0;
    int tile64_min = INT_MAX;
    int tile64_max = 0;
    int count_min = INT_MAX;
    int count_max = 0;
    double tile_sum = 0.0;
    double tile64_sum = 0.0;
    double padding_sum = 0.0;
    printf("router-trace: tokens %d routes/layer %d time %.3f s %.3f tok/s\n",
           count, route_count, seconds, count / seconds);
    for (int layer = 0; layer < XE_LAYERS; layer++) {
        int tiles = 0;
        int tiles64 = 0;
        int nonzero = 0;
        int layer_min = route_count;
        int layer_max = 0;
        for (int expert = 0; expert < XE_EXPERTS; expert++) {
            int value = routes[layer][expert];
            if (value) {
                nonzero++;
                if (value < layer_min) layer_min = value;
                if (value > layer_max) layer_max = value;
                tiles += (value + 31) / 32;
                tiles64 += (value + 63) / 64;
            }
        }
        double padding = (double)tiles * 32 / route_count;
        printf("router-trace: L%02d %s experts %d count %d..%d tiles32 %d padding32 %.6fx tiles64 %d padding64 %.6fx\n",
               layer, XE_IS_GLOBAL(layer) ? "global" : "swa", nonzero,
               layer_min, layer_max, tiles, padding, tiles64,
               (double)tiles64 * 64 / route_count);
        if (tiles < tile_min) tile_min = tiles;
        if (tiles > tile_max) tile_max = tiles;
        if (tiles64 < tile64_min) tile64_min = tiles64;
        if (tiles64 > tile64_max) tile64_max = tiles64;
        if (layer_min < count_min) count_min = layer_min;
        if (layer_max > count_max) count_max = layer_max;
        tile_sum += tiles;
        tile64_sum += tiles64;
        padding_sum += padding;
    }
    printf("router-trace: aggregate count %d..%d tiles32 mean %.3f range %d..%d padding32 mean %.6fx tiles64 mean %.3f range %d..%d padding64 mean %.6fx\n",
           count_min, count_max, tile_sum / XE_LAYERS, tile_min, tile_max,
           padding_sum / XE_LAYERS, tile64_sum / XE_LAYERS, tile64_min,
           tile64_max, tile64_sum * 64 / (XE_LAYERS * route_count));
    xe_session_free(session);
    xe_engine_close(engine);
    xe_free(NULL, ids, XE_MEM_HOST);
    return 0;
}

int main(int argc, char **argv) {
    if (argc > 1 && !strcmp(argv[1], "--router-trace")) {
        if (argc < 3 || !argv[2][0]) {
            fprintf(stderr, "usage: %s --router-trace <model.gguf> [ids-file [limit]]\n", argv[0]);
            return 2;
        }
        const char *model = argv[2];
        const char *ids = argc > 3 ? argv[3] : "tests/golden/long.ids";
        int limit = argc > 4 ? (int)strtol(argv[4], NULL, 10) : 512;
        return tg_router_trace(model, ids, limit);
    }
    if (argc < 2 || !argv[1][0]) {
        fprintf(stderr, "usage: %s <model.gguf> [reps [cooldown [warmup]]]\n", argv[0]);
        return 2;
    }
    const char *model = argv[1];
    int reps = argc > 2 ? (int)strtol(argv[2], NULL, 10) : 5;
    int cooldown = argc > 3 ? (int)strtol(argv[3], NULL, 10) : 0;
    int warmup = argc > 4 ? (int)strtol(argv[4], NULL, 10) : 35;
    if (reps < 1 || reps > 20 || cooldown < 0 || cooldown > 600 ||
        warmup < 0 || warmup > 600) return 2;

    long ac = tg_read_long("/sys/class/power_supply/ACAD/online");
    if (ac != 1 && !getenv("XE_BENCH_ALLOW_BATTERY")) {
        fprintf(stderr, "tg128: AC power is offline; set XE_BENCH_ALLOW_BATTERY=1 to override\n");
        return 3;
    }

    cpu_set_t original_affinity;
    int affinity_rc = pthread_getaffinity_np(pthread_self(), sizeof original_affinity, &original_affinity);
    if (affinity_rc != 0) xe_fatal("tg128: original affinity: %s", strerror(affinity_rc));
    xe_engine *e = xe_engine_open(model);
    xe_session *s = xe_session_new(e);
    tg_print_environment(e, cooldown);
    if (cooldown) sleep((unsigned)cooldown);

    double warm_start = tg_now();
    int warm_sequences = 0;
    do {
        s->n_tokens = 0;
        uint32_t state = UINT32_C(0x9e3779b9) + (uint32_t)warm_sequences;
        for (int pos = 0; pos < 128; pos++) {
            int32_t token = 2;
            if (pos) {
                state = state * UINT32_C(1664525) + UINT32_C(1013904223);
                token = (int32_t)(state % XE_VOCAB);
            }
            xe_decode_token(s, token, pos);
        }
        warm_sequences++;
    } while (tg_now() - warm_start < warmup);
    double warm_seconds = tg_now() - warm_start;
    printf("tg128: active warmup %.3f s sequences %d %.3f tok/s\n",
           warm_seconds, warm_sequences, warm_sequences * 128.0 / warm_seconds);

    tg_pmu pmu = tg_pmu_open();
    if (tg_pmu_available(&pmu))
        printf("tg128: PMU IMC read/write and package energy available\n");
    else
        printf("tg128: PMU unavailable: %s\n", pmu.error);

    tg_sampler sampler = {0};
    sampler.engine = e;
    sampler.allowed_cpus = original_affinity;
    for (int rep = 0; rep < 20; rep++) sampler.freq_min[rep] = LONG_MAX;
    atomic_init(&sampler.stop, 0);
    atomic_init(&sampler.rep, -1);
    tg_find_package_temp(sampler.temp_path, sizeof sampler.temp_path);
    if (pthread_create(&sampler.thread, NULL, tg_sampler_main, &sampler) != 0)
        xe_fatal("tg128: sampler thread creation failed");

    double rep_seconds[20];
    double all_tokens[20 * 128];
    tg_snapshot before[20];
    tg_snapshot after[20];
    xe_tg_profile profiles[20];
    tg_pmu_result pmu_results[20];
    double model_bytes = (double)e->dense_bytes + (double)e->experts_bytes *
                         XE_EXPERTS_USED / XE_EXPERTS + (double)e->tok_embd_bytes;
    for (int rep = 0; rep < reps; rep++) {
        s->n_tokens = 0;
        uint32_t state = UINT32_C(0x12345678) + (uint32_t)rep;
        memset(&xe_tg_profile_data, 0, sizeof xe_tg_profile_data);
        before[rep] = tg_take_snapshot(e);
        atomic_store_explicit(&sampler.rep, rep, memory_order_release);
        tg_pmu_start(&pmu);
        double start = tg_now();
        for (int pos = 0; pos < 128; pos++) {
            int32_t token;
            if (pos == 0) {
                token = 2;
            } else {
                state = state * UINT32_C(1664525) + UINT32_C(1013904223);
                token = (int32_t)(state % XE_VOCAB);
            }
            double t0 = tg_now();
            xe_decode_token(s, token, pos);
            all_tokens[rep * 128 + pos] = tg_now() - t0;
        }
        rep_seconds[rep] = tg_now() - start;
        pmu_results[rep] = tg_pmu_stop(&pmu);
        atomic_store_explicit(&sampler.rep, -1, memory_order_release);
        after[rep] = tg_take_snapshot(e);
        profiles[rep] = xe_tg_profile_data;
        double first = 0.0;
        double last = 0.0;
        for (int i = 0; i < 16; i++) first += all_tokens[rep * 128 + i];
        for (int i = 112; i < 128; i++) last += all_tokens[rep * 128 + i];
        double sorted_tokens[128];
        memcpy(sorted_tokens, all_tokens + rep * 128, sizeof sorted_tokens);
        qsort(sorted_tokens, 128, sizeof sorted_tokens[0], tg_compare);
        printf("tg128: rep %d %.3f tok/s p50 %.3f p90 %.3f first16 %.3f last16 %.3f ms model %.2f GB/s\n",
               rep + 1, 128.0 / rep_seconds[rep],
               sorted_tokens[64] * 1000.0, sorted_tokens[115] * 1000.0,
               first * 1000.0 / 16.0, last * 1000.0 / 16.0,
               model_bytes * 128.0 / rep_seconds[rep] / 1e9);
        if (tg_pmu_available(&pmu)) {
            double read_gb = (pmu_results[rep].count[TG_IMC0_READ] +
                              pmu_results[rep].count[TG_IMC1_READ]) * 64.0 / 1e9;
            double write_gb = (pmu_results[rep].count[TG_IMC0_WRITE] +
                               pmu_results[rep].count[TG_IMC1_WRITE]) * 64.0 / 1e9;
            double joules = pmu_results[rep].count[TG_ENERGY_PKG] * pmu.energy_scale;
            printf("tg128: PMU rep %d DRAM read %.2f write %.2f GB/s package %.2f W %.3f J/token\n",
                   rep + 1, read_gb / rep_seconds[rep], write_gb / rep_seconds[rep],
                   joules / rep_seconds[rep], joules / 128.0);
        }
        fflush(stdout);
    }

    atomic_store_explicit(&sampler.stop, 1, memory_order_release);
    pthread_join(sampler.thread, NULL);

    for (int rep = 0; rep < reps; rep++) {
        xe_tg_profile *p = &profiles[rep];
        long freq_avg = sampler.freq_samples[rep]
            ? sampler.freq_sum[rep] / sampler.freq_samples[rep] : -1;
        long temp_avg = sampler.temp_samples[rep]
            ? sampler.temp_sum[rep] / sampler.temp_samples[rep] : -1;
        printf("tg128: diag rep %d freq avg/min/max %.0f/%.0f/%.0f MHz temp avg/max %.1f/%.1f C samples %ld\n",
               rep + 1, freq_avg / 1000.0, sampler.freq_min[rep] / 1000.0,
               sampler.freq_max[rep] / 1000.0, temp_avg / 1000.0,
               sampler.temp_max[rep] / 1000.0, sampler.freq_samples[rep]);
        printf("tg128: diag rep %d throttle core count/ms %ld/%ld package count/ms %ld/%ld faults minor/major %ld/%ld csw vol/invol %ld/%ld\n",
               rep + 1, after[rep].core_count - before[rep].core_count,
               after[rep].core_ms - before[rep].core_ms,
               after[rep].package_count - before[rep].package_count,
               after[rep].package_ms - before[rep].package_ms,
               after[rep].usage.ru_minflt - before[rep].usage.ru_minflt,
               after[rep].usage.ru_majflt - before[rep].usage.ru_majflt,
               after[rep].usage.ru_nvcsw - before[rep].usage.ru_nvcsw,
               after[rep].usage.ru_nivcsw - before[rep].usage.ru_nivcsw);
        printf("tg128: diag rep %d memory rss/anon/swap/huge %ld/%ld/%ld/%ld MiB\n",
               rep + 1, after[rep].rss_kb / 1024, after[rep].anon_kb / 1024,
               after[rep].swap_kb / 1024, after[rep].huge_kb / 1024);
        printf("tg128: profile rep %d embed %.3f attention %.3f dense %.3f moe %.3f glue %.3f output %.3f end %.3f ms/token\n",
               rep + 1, p->embed_rope * 1000.0 / 128.0,
               p->attention * 1000.0 / 128.0, p->dense * 1000.0 / 128.0,
               p->moe * 1000.0 / 128.0, p->glue * 1000.0 / 128.0,
               p->output * 1000.0 / 128.0, p->worker_end * 1000.0 / 128.0);
    }

    double sorted_reps[20];
    memcpy(sorted_reps, rep_seconds, (size_t)reps * sizeof(*sorted_reps));
    qsort(sorted_reps, (size_t)reps, sizeof(sorted_reps[0]), tg_compare);
    qsort(all_tokens, (size_t)reps * 128, sizeof(all_tokens[0]), tg_compare);
    double total = 0.0;
    for (int i = 0; i < reps; i++) total += rep_seconds[i];
    double mean = total / reps;
    printf("tg128: workers %d reps %d mean %.3f tok/s median-rep %.3f tok/s "
           "token-p50 %.3f ms token-p90 %.3f ms model %.2f GB/s bytes/token %.6f GB\n",
           xe_engine_worker_count(e), reps, 128.0 / mean, 128.0 / sorted_reps[reps / 2],
           all_tokens[reps * 64] * 1000.0, all_tokens[reps * 115] * 1000.0,
           model_bytes * 128.0 / mean / 1e9, model_bytes / 1e9);

    tg_pmu_close(&pmu);
    xe_session_free(s);
    xe_engine_close(e);
    return 0;
}
