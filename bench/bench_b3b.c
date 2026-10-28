#include "../xenolith.c"

#include <level_zero/ze_api.h>
#include <linux/perf_event.h>
#include <sys/ioctl.h>
#include <sys/syscall.h>

#define B3B_ROWS 4096
#define B3B_BLOCKS 88
#define B3B_N (B3B_BLOCKS * 32)
#ifndef B3B_REPLICAS
#define B3B_REPLICAS 160
#endif
#define B3B_TAIL_SECONDS 35.0
#define B3B_PMU_COUNT 5

enum {
    B3B_IMC0_READ,
    B3B_IMC0_WRITE,
    B3B_IMC1_READ,
    B3B_IMC1_WRITE,
    B3B_ENERGY_PKG
};

typedef struct {
    int fd[B3B_PMU_COUNT];
    double energy_scale;
    char error[128];
} b3b_pmu;

typedef struct {
    double count[B3B_PMU_COUNT];
} b3b_pmu_result;

typedef struct {
    uint8_t *weights;
    uint16_t *weight_scales;
    int8_t *activation;
    _Float16 *activation_scales;
    int16_t *activation_sigma;
    float *output;
    int x8_layout;
} b3b_data;

typedef struct {
    ze_driver_handle_t driver;
    ze_device_handle_t device;
    ze_context_handle_t context;
    ze_command_list_handle_t commands;
    ze_module_handle_t module;
    ze_kernel_handle_t matvec_kernel;
    ze_kernel_handle_t read_kernel;
} b3b_gpu;

typedef struct {
    pthread_t thread;
    atomic_int stop;
    double start;
    long samples[3];
    long requested_sum[3];
    long requested_min[3];
    long requested_max[3];
    long actual_sum[3];
    long actual_min[3];
    long actual_max[3];
    long pl1_samples[3];
    long pl2_samples[3];
    long thermal_samples[3];
} b3b_gpu_sampler;

typedef void (*b3b_pass_fn)(void *);

static double b3b_now(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec + (double)ts.tv_nsec * 1e-9;
}

static uint64_t b3b_rng(uint64_t *state) {
    uint64_t x = *state;
    x ^= x >> 12;
    x ^= x << 25;
    x ^= x >> 27;
    *state = x;
    return x * UINT64_C(0x2545f4914f6cdd1d);
}

static void *b3b_alloc(size_t bytes) {
    size_t rounded = (bytes + 63) & ~(size_t)63;
    void *p = aligned_alloc(64, rounded);
    if (!p) xe_fatal("b3b: out of memory allocating %zu bytes", rounded);
    return p;
}

static uint16_t b3b_half_bits(_Float16 value) {
    uint16_t bits;
    memcpy(&bits, &value, sizeof bits);
    return bits;
}

static void b3b_fill(b3b_data *data) {
    size_t matrix_blocks = (size_t)B3B_ROWS * B3B_BLOCKS;
    size_t weight_bytes = matrix_blocks * 16;
    size_t scale_bytes = matrix_blocks * sizeof(*data->weight_scales);
    uint64_t rng = UINT64_C(0x6a09e667f3bcc909);

    for (size_t i = 0; i < weight_bytes; i++)
        data->weights[i] = (uint8_t)b3b_rng(&rng);
    for (size_t i = 0; i < matrix_blocks; i++) {
        float scale = 0.0005f + (float)(b3b_rng(&rng) & 4095) * (0.025f / 4095.0f);
        data->weight_scales[i] = b3b_half_bits((_Float16)scale);
    }
    for (int rep = 1; rep < B3B_REPLICAS; rep++) {
        memcpy(data->weights + (size_t)rep * weight_bytes, data->weights, weight_bytes);
        memcpy(data->weight_scales + (size_t)rep * matrix_blocks,
               data->weight_scales, scale_bytes);
    }

    float values[B3B_N];
    for (int i = 0; i < B3B_N; i++)
        values[i] = (float)((int)(b3b_rng(&rng) >> 40) - 8388608) * (1.0f / 4194304.0f);
    xe_q8 q = { data->activation, data->activation_scales,
                data->activation_sigma, B3B_N };
    xe_quantize_q8(values, B3B_N, &q);
}

#ifndef B3B_LEGACY_LAYOUT
static void b3b_repack_cpu_x8(b3b_data *data) {
    size_t matrix_blocks = (size_t)B3B_ROWS * B3B_BLOCKS;
    size_t weight_bytes = matrix_blocks * 16;
    size_t scale_bytes = matrix_blocks * sizeof(*data->weight_scales);
    uint8_t *packed_weights = b3b_alloc(weight_bytes);
    uint16_t *packed_scales = b3b_alloc(scale_bytes);
    for (int group = 0; group < B3B_ROWS / 8; group++) {
        for (int block = 0; block < B3B_BLOCKS; block++) {
            uint8_t *dst = packed_weights + ((size_t)group * B3B_BLOCKS + block) * 128;
            uint16_t *ds = packed_scales + ((size_t)group * B3B_BLOCKS + block) * 8;
            for (int row = 0; row < 8; row++) {
                size_t native = ((size_t)group * 8 + row) * B3B_BLOCKS + block;
                for (int chunk = 0; chunk < 4; chunk++)
                    memcpy(dst + chunk * 32 + row * 4,
                           data->weights + native * 16 + chunk * 4, 4);
                ds[row] = data->weight_scales[native];
            }
        }
    }
    for (int rep = 0; rep < B3B_REPLICAS; rep++) {
        memcpy(data->weights + (size_t)rep * weight_bytes, packed_weights, weight_bytes);
        memcpy(data->weight_scales + (size_t)rep * matrix_blocks, packed_scales, scale_bytes);
    }
    free(packed_scales);
    free(packed_weights);
    data->x8_layout = 1;
}
#endif

static float b3b_reference_row(const b3b_data *data, int row) {
    float sum = 0.0f;
    for (int block = 0; block < B3B_BLOCKS; block++) {
        uint8_t row_weights[16];
        const uint8_t *weights;
        if (data->x8_layout) {
            const uint8_t *src = data->weights +
                (((size_t)row / 8 * B3B_BLOCKS + block) * 128 + (row & 7) * 4);
            for (int chunk = 0; chunk < 4; chunk++)
                memcpy(row_weights + chunk * 4, src + chunk * 32, 4);
            weights = row_weights;
        } else {
            weights = data->weights + ((size_t)row * B3B_BLOCKS + block) * 16;
        }
        const int8_t *activation = data->activation + block * 32;
        int integer = 0;
        for (int i = 0; i < 16; i++) {
            integer += ((weights[i] & 15) - 8) * activation[i];
            integer += ((weights[i] >> 4) - 8) * activation[i + 16];
        }
        size_t scale_index = data->x8_layout ?
            (((size_t)row / 8 * B3B_BLOCKS + block) * 8 + (row & 7)) :
            ((size_t)row * B3B_BLOCKS + block);
        float wd = _cvtsh_ss(data->weight_scales[scale_index]);
        float qd = (float)data->activation_scales[block];
        sum += (float)integer * wd * qd;
    }
    return sum;
}

static void b3b_verify(const char *mode, const b3b_data *data) {
    double error = 0.0;
    double reference = 0.0;
    double max_abs = 0.0;
    for (int row = 0; row < B3B_ROWS; row++) {
        double expected = b3b_reference_row(data, row);
        double difference = (double)data->output[row] - expected;
        error += difference * difference;
        reference += expected * expected;
        if (fabs(difference) > max_abs) max_abs = fabs(difference);
    }
    double rel_rms = sqrt(error / (reference + 1e-30));
    printf("b3b: %s correctness rel_rms %.9g max_abs %.9g\n", mode, rel_rms, max_abs);
    if (!isfinite(rel_rms) || rel_rms > 5e-5)
        xe_fatal("b3b: %s correctness failed", mode);
}

static long b3b_read_long(const char *path) {
    FILE *file = fopen(path, "r");
    if (!file) return -1;
    long value = -1;
    if (fscanf(file, "%ld", &value) != 1) value = -1;
    fclose(file);
    return value;
}

static double b3b_read_double(const char *path) {
    FILE *file = fopen(path, "r");
    if (!file) return -1.0;
    double value = -1.0;
    if (fscanf(file, "%lf", &value) != 1) value = -1.0;
    fclose(file);
    return value;
}

static int b3b_perf_open(uint32_t type, uint64_t config) {
    struct perf_event_attr attr = {0};
    attr.type = type;
    attr.size = sizeof attr;
    attr.config = config;
    attr.disabled = 1;
    attr.read_format = PERF_FORMAT_TOTAL_TIME_ENABLED | PERF_FORMAT_TOTAL_TIME_RUNNING;
    return (int)syscall(SYS_perf_event_open, &attr, -1, 0, -1, 0);
}

static b3b_pmu b3b_pmu_open(void) {
    b3b_pmu pmu = {0};
    for (int i = 0; i < B3B_PMU_COUNT; i++) pmu.fd[i] = -1;
    long imc0 = b3b_read_long("/sys/bus/event_source/devices/uncore_imc_free_running_0/type");
    long imc1 = b3b_read_long("/sys/bus/event_source/devices/uncore_imc_free_running_1/type");
    long power = b3b_read_long("/sys/bus/event_source/devices/power/type");
    pmu.energy_scale = b3b_read_double("/sys/bus/event_source/devices/power/events/energy-pkg.scale");
    if (imc0 < 0 || imc1 < 0 || power < 0 || pmu.energy_scale <= 0.0) {
        snprintf(pmu.error, sizeof pmu.error, "PMU metadata unavailable");
        return pmu;
    }
    uint32_t types[B3B_PMU_COUNT] = {
        (uint32_t)imc0, (uint32_t)imc0, (uint32_t)imc1, (uint32_t)imc1,
        (uint32_t)power
    };
    uint64_t configs[B3B_PMU_COUNT] = { 0x20ff, 0x30ff, 0x20ff, 0x30ff, 0x02 };
    for (int i = 0; i < B3B_PMU_COUNT; i++) {
        pmu.fd[i] = b3b_perf_open(types[i], configs[i]);
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

static int b3b_pmu_available(const b3b_pmu *pmu) {
    return pmu->fd[0] >= 0;
}

static void b3b_pmu_start(b3b_pmu *pmu) {
    if (!b3b_pmu_available(pmu)) return;
    for (int i = 0; i < B3B_PMU_COUNT; i++) {
        ioctl(pmu->fd[i], PERF_EVENT_IOC_RESET, 0);
        ioctl(pmu->fd[i], PERF_EVENT_IOC_ENABLE, 0);
    }
}

static b3b_pmu_result b3b_pmu_stop(b3b_pmu *pmu) {
    b3b_pmu_result result = {0};
    if (!b3b_pmu_available(pmu)) return result;
    for (int i = 0; i < B3B_PMU_COUNT; i++) {
        uint64_t values[3] = {0};
        ioctl(pmu->fd[i], PERF_EVENT_IOC_DISABLE, 0);
        if (read(pmu->fd[i], values, sizeof values) != sizeof values || !values[2]) {
            snprintf(pmu->error, sizeof pmu->error, "PMU counter read failed: %s", strerror(errno));
            for (int j = 0; j < B3B_PMU_COUNT; j++) {
                close(pmu->fd[j]);
                pmu->fd[j] = -1;
            }
            return result;
        }
        result.count[i] = (double)values[0] * (double)values[1] / (double)values[2];
    }
    return result;
}

static void b3b_pmu_close(b3b_pmu *pmu) {
    for (int i = 0; i < B3B_PMU_COUNT; i++)
        if (pmu->fd[i] >= 0) close(pmu->fd[i]);
}

static long b3b_read_fd(int fd) {
    char buffer[64];
    ssize_t count = pread(fd, buffer, sizeof buffer - 1, 0);
    if (count <= 0) return -1;
    buffer[count] = 0;
    char *end;
    long value = strtol(buffer, &end, 10);
    return end == buffer ? -1 : value;
}

static void b3b_sample_value(long value, long *sum, long *min, long *max) {
    if (value < 0) return;
    *sum += value;
    if (*min < 0 || value < *min) *min = value;
    if (*max < 0 || value > *max) *max = value;
}

static void *b3b_gpu_sampler_main(void *opaque) {
    b3b_gpu_sampler *sampler = opaque;
    const char *base = "/sys/class/drm/card1/gt/gt0/";
    char path[256];
    snprintf(path, sizeof path, "%spunit_req_freq_mhz", base);
    int requested_fd = open(path, O_RDONLY);
    snprintf(path, sizeof path, "%srps_act_freq_mhz", base);
    int actual_fd = open(path, O_RDONLY);
    snprintf(path, sizeof path, "%sthrottle_reason_pl1", base);
    int pl1_fd = open(path, O_RDONLY);
    snprintf(path, sizeof path, "%sthrottle_reason_pl2", base);
    int pl2_fd = open(path, O_RDONLY);
    snprintf(path, sizeof path, "%sthrottle_reason_thermal", base);
    int thermal_fd = open(path, O_RDONLY);
    if (requested_fd < 0 || actual_fd < 0 || pl1_fd < 0 || pl2_fd < 0 || thermal_fd < 0)
        xe_fatal("b3b: cannot open i915 frequency telemetry");

    xe_pin_thread(13);
    while (!atomic_load_explicit(&sampler->stop, memory_order_relaxed)) {
        double elapsed = b3b_now() - sampler->start;
        int band = elapsed < 20.0 ? 0 : elapsed < B3B_TAIL_SECONDS ? 1 : 2;
        long requested = b3b_read_fd(requested_fd);
        long actual = b3b_read_fd(actual_fd);
        long pl1 = b3b_read_fd(pl1_fd);
        long pl2 = b3b_read_fd(pl2_fd);
        long thermal = b3b_read_fd(thermal_fd);
        b3b_sample_value(requested, &sampler->requested_sum[band],
                         &sampler->requested_min[band], &sampler->requested_max[band]);
        b3b_sample_value(actual, &sampler->actual_sum[band],
                         &sampler->actual_min[band], &sampler->actual_max[band]);
        sampler->samples[band]++;
        if (pl1 > 0) sampler->pl1_samples[band]++;
        if (pl2 > 0) sampler->pl2_samples[band]++;
        if (thermal > 0) sampler->thermal_samples[band]++;
        struct timespec interval = { 0, 50000000 };
        nanosleep(&interval, NULL);
    }

    close(thermal_fd);
    close(pl2_fd);
    close(pl1_fd);
    close(actual_fd);
    close(requested_fd);
    return NULL;
}

static void b3b_gpu_sampler_start(b3b_gpu_sampler *sampler) {
    memset(sampler, 0, sizeof *sampler);
    for (int band = 0; band < 3; band++) {
        sampler->requested_min[band] = -1;
        sampler->requested_max[band] = -1;
        sampler->actual_min[band] = -1;
        sampler->actual_max[band] = -1;
    }
    atomic_init(&sampler->stop, 0);
    sampler->start = b3b_now();
    if (pthread_create(&sampler->thread, NULL, b3b_gpu_sampler_main, sampler) != 0)
        xe_fatal("b3b: cannot create GPU telemetry thread");
}

static void b3b_gpu_sampler_stop(b3b_gpu_sampler *sampler) {
    atomic_store_explicit(&sampler->stop, 1, memory_order_relaxed);
    pthread_join(sampler->thread, NULL);
    const char *names[3] = { "0-20", "20-35", "35-tail" };
    for (int band = 0; band < 3; band++) {
        long count = sampler->samples[band];
        if (!count) continue;
        printf("b3b: gpu telemetry %s s samples %ld requested %.1f/%ld/%ld MHz actual %.1f/%ld/%ld MHz throttle pl1/pl2/thermal %ld/%ld/%ld\n",
               names[band], count,
               (double)sampler->requested_sum[band] / count,
               sampler->requested_min[band], sampler->requested_max[band],
               (double)sampler->actual_sum[band] / count,
               sampler->actual_min[band], sampler->actual_max[band],
               sampler->pl1_samples[band], sampler->pl2_samples[band],
               sampler->thermal_samples[band]);
    }
}

static void b3b_print_power_policy(void) {
    const char *base = "/sys/devices/virtual/powercap/intel-rapl-mmio/intel-rapl-mmio:0/";
    char path[256];
    snprintf(path, sizeof path, "%sconstraint_0_power_limit_uw", base);
    long pl1 = b3b_read_long(path);
    snprintf(path, sizeof path, "%sconstraint_0_time_window_us", base);
    long tau = b3b_read_long(path);
    snprintf(path, sizeof path, "%sconstraint_1_power_limit_uw", base);
    long pl2 = b3b_read_long(path);
    printf("b3b: PL1 %.3f W tau %.6f s PL2 %.3f W\n",
           pl1 / 1e6, tau / 1e6, pl2 / 1e6);
}

static void b3b_delay(double seconds) {
    if (seconds <= 0.0) return;
    printf("b3b: cooldown %.3f s\n", seconds);
    fflush(stdout);
    struct timespec delay;
    delay.tv_sec = (time_t)seconds;
    delay.tv_nsec = (long)((seconds - (double)delay.tv_sec) * 1e9);
    while (nanosleep(&delay, &delay) != 0) {
        if (errno != EINTR) xe_fatal("b3b: cooldown: %s", strerror(errno));
    }
}

static void b3b_run(const char *mode, double seconds, b3b_pass_fn pass, void *state) {
    const double useful_bytes = (double)B3B_REPLICAS * B3B_ROWS * B3B_BLOCKS * 18.0;
    double start = b3b_now();
    double window_start = start;
    double window_bytes = 0.0;
    double total_bytes = 0.0;
    double tail_bytes = 0.0;
    double tail_start = 0.0;
    double tail_end = 0.0;
    int passes = 0;
    int window = 0;
    b3b_pmu pmu = b3b_pmu_open();
    if (b3b_pmu_available(&pmu))
        printf("b3b: PMU IMC read/write and package energy available\n");
    else
        printf("b3b: PMU unavailable: %s\n", pmu.error);

    while (b3b_now() - start < seconds) {
        double before = b3b_now();
        pass(state);
        double after = b3b_now();
        double elapsed = after - before;
        if (passes == 0)
            printf("b3b: %s cold pass %.6f s %.6f GB/s\n",
                   mode, elapsed, useful_bytes / elapsed / 1e9);
        if (before - start >= B3B_TAIL_SECONDS) {
            if (tail_start == 0.0) {
                tail_start = before;
                b3b_pmu_start(&pmu);
            }
            tail_bytes += useful_bytes;
            tail_end = after;
        }
        total_bytes += useful_bytes;
        window_bytes += useful_bytes;
        passes++;
        if (after - window_start >= 1.0) {
            window++;
            printf("b3b: %s window %d %.3f-%.3f s %.6f GB/s\n",
                   mode, window, window_start - start, after - start,
                   window_bytes / (after - window_start) / 1e9);
            fflush(stdout);
            window_start = after;
            window_bytes = 0.0;
        }
    }

    double total_time = b3b_now() - start;
    b3b_pmu_result pmu_result = b3b_pmu_stop(&pmu);
    printf("b3b: %s total passes %d time %.6f s %.6f GB/s\n",
           mode, passes, total_time, total_bytes / total_time / 1e9);
    if (tail_start > 0.0)
        printf("b3b: %s tail start %.1f s time %.6f s %.6f GB/s\n",
               mode, B3B_TAIL_SECONDS, tail_end - tail_start,
               tail_bytes / (tail_end - tail_start) / 1e9);
    else
        printf("b3b: %s tail unavailable\n", mode);
    if (tail_start > 0.0 && b3b_pmu_available(&pmu)) {
        double read_gb = (pmu_result.count[B3B_IMC0_READ] +
                          pmu_result.count[B3B_IMC1_READ]) * 64.0 / 1e9;
        double write_gb = (pmu_result.count[B3B_IMC0_WRITE] +
                           pmu_result.count[B3B_IMC1_WRITE]) * 64.0 / 1e9;
        double joules = pmu_result.count[B3B_ENERGY_PKG] * pmu.energy_scale;
        printf("b3b: %s tail PMU DRAM read %.6f write %.6f GB/s package %.3f W %.6f J/GB\n",
               mode, read_gb / (tail_end - tail_start),
               write_gb / (tail_end - tail_start), joules / (tail_end - tail_start),
               joules / (tail_bytes / 1e9));
    }
    b3b_pmu_close(&pmu);
}

typedef struct {
    xe_engine engine;
    xe_session session;
    b3b_data *data;
    _Alignas(64) uint32_t lane_checksum[CPU_SETSIZE];
    uint32_t checksum;
} b3b_cpu_state;

static uint32_t b14_sum_u32x8(__m256i value) {
    _Alignas(32) uint32_t lanes[8];
    _mm256_store_si256((__m256i *)lanes, value);
    uint32_t sum = 0;
    for (int i = 0; i < 8; i++) sum += lanes[i];
    return sum;
}

static __m256i b14_read_vectors(const void *data, size_t vectors,
                                int worker, int workers) {
    const __m256i *input = data;
    size_t begin = vectors * (size_t)worker / (size_t)workers;
    size_t end = vectors * (size_t)(worker + 1) / (size_t)workers;
    __m256i sum0 = _mm256_setzero_si256();
    __m256i sum1 = _mm256_setzero_si256();
    __m256i sum2 = _mm256_setzero_si256();
    __m256i sum3 = _mm256_setzero_si256();
    size_t i = begin;
    for (; i + 3 < end; i += 4) {
        sum0 = _mm256_add_epi32(sum0, _mm256_loadu_si256(input + i));
        sum1 = _mm256_add_epi32(sum1, _mm256_loadu_si256(input + i + 1));
        sum2 = _mm256_add_epi32(sum2, _mm256_loadu_si256(input + i + 2));
        sum3 = _mm256_add_epi32(sum3, _mm256_loadu_si256(input + i + 3));
    }
    sum0 = _mm256_add_epi32(sum0, sum1);
    sum2 = _mm256_add_epi32(sum2, sum3);
    sum0 = _mm256_add_epi32(sum0, sum2);
    for (; i < end; i++)
        sum0 = _mm256_add_epi32(sum0, _mm256_loadu_si256(input + i));
    return sum0;
}

static void b14_cpu_read_phase(xe_session *session, const void *opaque,
                               int worker, int workers) {
    (void)session;
    b3b_cpu_state *state = (b3b_cpu_state *)opaque;
    size_t blocks = (size_t)B3B_REPLICAS * B3B_ROWS * B3B_BLOCKS;
    __m256i weights = b14_read_vectors(state->data->weights,
                                       blocks * 16 / 32, worker, workers);
    __m256i scales = b14_read_vectors(state->data->weight_scales,
                                      blocks * 2 / 32, worker, workers);
    state->lane_checksum[worker] = b14_sum_u32x8(_mm256_add_epi32(weights, scales));
}

static void b14_cpu_read_pass(void *opaque) {
    b3b_cpu_state *state = opaque;
    xe_workers_begin(&state->engine);
    xe_dispatch(&state->engine, &state->session, b14_cpu_read_phase, state, 1);
    xe_workers_end(&state->engine);
    uint32_t sum = 0;
    for (int worker = 0; worker < xe_engine_worker_count(&state->engine); worker++)
        sum += state->lane_checksum[worker];
    state->checksum = sum;
}

static uint32_t b14_expected_checksum(const b3b_data *data) {
    size_t base_blocks = (size_t)B3B_ROWS * B3B_BLOCKS;
    uint32_t sum = 0;
    for (size_t i = 0; i < base_blocks * 16 / 4; i++) {
        uint32_t value;
        memcpy(&value, data->weights + 4 * i, sizeof value);
        sum += value;
    }
    const uint8_t *scales = (const uint8_t *)data->weight_scales;
    for (size_t i = 0; i < base_blocks * 2 / 4; i++) {
        uint32_t value;
        memcpy(&value, scales + 4 * i, sizeof value);
        sum += value;
    }
    return sum * (uint32_t)B3B_REPLICAS;
}

static void b14_verify_checksum(const char *mode, uint32_t actual,
                                const b3b_data *data) {
    uint32_t expected = b14_expected_checksum(data);
    printf("b3b: %s checksum %08x expected %08x %s\n",
           mode, actual, expected, actual == expected ? "PASS" : "FAIL");
    if (actual != expected) xe_fatal("b3b: %s checksum failed", mode);
}

static void b3b_cpu_pass(void *opaque) {
    b3b_cpu_state *state = opaque;
    size_t matrix_blocks = (size_t)B3B_ROWS * B3B_BLOCKS;
    size_t weight_bytes = matrix_blocks * 16;
    xe_workers_begin(&state->engine);
    for (int rep = 0; rep < B3B_REPLICAS; rep++) {
        xe_q4 weights = {
            state->data->weights + (size_t)rep * weight_bytes,
            state->data->weight_scales + (size_t)rep * matrix_blocks,
#ifdef B3B_LEGACY_LAYOUT
            0
#else
            B3B_BLOCKS
#endif
        };
        xe_matvec_arg arg = { &weights, 0, B3B_N, B3B_ROWS,
                              &(xe_q8){ state->data->activation,
                                        state->data->activation_scales,
                                        state->data->activation_sigma, B3B_N },
                              state->data->output };
        xe_dispatch(&state->engine, &state->session, xe_matvec_phase, &arg, 1);
    }
    xe_workers_end(&state->engine);
}

static void b3b_cpu_init(b3b_cpu_state *state, b3b_data *data) {
    memset(state, 0, sizeof *state);
    state->data = data;
    state->session.engine = &state->engine;
    state->engine.scalar_rms = 1;
    xe_worker_pool_init(&state->engine);
    int workers = xe_engine_worker_count(&state->engine);
    printf("b3b: cpu workers %d cpus", workers);
    for (int lane = 0; lane < workers; lane++)
        printf(" %d", xe_engine_worker_cpu(&state->engine, lane));
    printf("\n");
}

static void b3b_cpu_destroy(b3b_cpu_state *state) {
    xe_worker_pool_destroy(&state->engine);
    xe_free(NULL, state->engine.worker_cpus, XE_MEM_HOST);
    xe_free(NULL, state->engine.gelu_lut, XE_MEM_HOST);
}

static _Noreturn void b3b_ze_fatal(const char *operation, ze_result_t result) {
    xe_fatal("b3b: %s failed: 0x%x", operation, result);
}

static void b3b_ze_check(const char *operation, ze_result_t result) {
    if (result != ZE_RESULT_SUCCESS) b3b_ze_fatal(operation, result);
}

static void *b3b_read_file(const char *path, size_t *size) {
    FILE *file = fopen(path, "rb");
    if (!file) xe_fatal("b3b: %s: %s", path, strerror(errno));
    if (fseek(file, 0, SEEK_END) != 0) xe_fatal("b3b: %s: seek failed", path);
    long length = ftell(file);
    if (length <= 0) xe_fatal("b3b: %s: invalid size", path);
    if (fseek(file, 0, SEEK_SET) != 0) xe_fatal("b3b: %s: seek failed", path);
    void *data = b3b_alloc((size_t)length);
    if (fread(data, 1, (size_t)length, file) != (size_t)length)
        xe_fatal("b3b: %s: short read", path);
    fclose(file);
    *size = (size_t)length;
    return data;
}

static void *b3b_gpu_alloc(b3b_gpu *gpu, size_t bytes) {
    ze_device_mem_alloc_desc_t device_desc = {
        .stype = ZE_STRUCTURE_TYPE_DEVICE_MEM_ALLOC_DESC
    };
    ze_host_mem_alloc_desc_t host_desc = {
        .stype = ZE_STRUCTURE_TYPE_HOST_MEM_ALLOC_DESC
    };
    void *pointer = NULL;
    b3b_ze_check("zeMemAllocShared",
                 zeMemAllocShared(gpu->context, &device_desc, &host_desc,
                                  bytes, 64, gpu->device, &pointer));
    return pointer;
}

static void b3b_gpu_init(b3b_gpu *gpu, const char *spv_path) {
    memset(gpu, 0, sizeof *gpu);
    b3b_ze_check("zeInit", zeInit(ZE_INIT_FLAG_GPU_ONLY));
    uint32_t driver_count = 0;
    b3b_ze_check("zeDriverGet count", zeDriverGet(&driver_count, NULL));
    if (driver_count == 0) xe_fatal("b3b: no Level Zero GPU driver");
    ze_driver_handle_t *drivers = b3b_alloc(driver_count * sizeof(*drivers));
    b3b_ze_check("zeDriverGet", zeDriverGet(&driver_count, drivers));
    gpu->driver = drivers[0];
    free(drivers);

    uint32_t device_count = 0;
    b3b_ze_check("zeDeviceGet count", zeDeviceGet(gpu->driver, &device_count, NULL));
    if (device_count == 0) xe_fatal("b3b: no Level Zero GPU device");
    ze_device_handle_t *devices = b3b_alloc(device_count * sizeof(*devices));
    b3b_ze_check("zeDeviceGet", zeDeviceGet(gpu->driver, &device_count, devices));
    gpu->device = devices[0];
    free(devices);

    ze_device_properties_t properties = {
        .stype = ZE_STRUCTURE_TYPE_DEVICE_PROPERTIES
    };
    b3b_ze_check("zeDeviceGetProperties", zeDeviceGetProperties(gpu->device, &properties));
    printf("b3b: gpu %s clock %u MHz\n", properties.name, properties.coreClockRate);

    ze_context_desc_t context_desc = {
        .stype = ZE_STRUCTURE_TYPE_CONTEXT_DESC
    };
    b3b_ze_check("zeContextCreate", zeContextCreate(gpu->driver, &context_desc, &gpu->context));
    ze_command_queue_desc_t queue_desc = {
        .stype = ZE_STRUCTURE_TYPE_COMMAND_QUEUE_DESC,
        .ordinal = 0,
        .index = 0,
        .mode = ZE_COMMAND_QUEUE_MODE_ASYNCHRONOUS,
        .priority = ZE_COMMAND_QUEUE_PRIORITY_NORMAL
    };
    b3b_ze_check("zeCommandListCreateImmediate",
                 zeCommandListCreateImmediate(gpu->context, gpu->device,
                                              &queue_desc, &gpu->commands));

    size_t spv_size;
    void *spv = b3b_read_file(spv_path, &spv_size);
    ze_module_desc_t module_desc = {
        .stype = ZE_STRUCTURE_TYPE_MODULE_DESC,
        .format = ZE_MODULE_FORMAT_IL_SPIRV,
        .inputSize = spv_size,
        .pInputModule = spv,
        .pBuildFlags = ""
    };
    ze_module_build_log_handle_t log = NULL;
    ze_result_t result = zeModuleCreate(gpu->context, gpu->device, &module_desc,
                                        &gpu->module, &log);
    if (result != ZE_RESULT_SUCCESS) {
        if (log) {
            size_t log_size = 0;
            zeModuleBuildLogGetString(log, &log_size, NULL);
            char *message = b3b_alloc(log_size + 1);
            zeModuleBuildLogGetString(log, &log_size, message);
            message[log_size] = 0;
            fprintf(stderr, "%s\n", message);
            free(message);
        }
        b3b_ze_fatal("zeModuleCreate", result);
    }
    if (log) zeModuleBuildLogDestroy(log);
    free(spv);

    ze_kernel_desc_t kernel_desc = { .stype = ZE_STRUCTURE_TYPE_KERNEL_DESC };
    kernel_desc.pKernelName = "b3b_q4_q8";
    b3b_ze_check("zeKernelCreate matvec",
                 zeKernelCreate(gpu->module, &kernel_desc, &gpu->matvec_kernel));
    b3b_ze_check("zeKernelSetGroupSize matvec",
                 zeKernelSetGroupSize(gpu->matvec_kernel, 128, 1, 1));
    kernel_desc.pKernelName = "b14_read";
    b3b_ze_check("zeKernelCreate read",
                 zeKernelCreate(gpu->module, &kernel_desc, &gpu->read_kernel));
    b3b_ze_check("zeKernelSetGroupSize read",
                 zeKernelSetGroupSize(gpu->read_kernel, 256, 1, 1));
    xe_pin_thread(12);
    printf("b3b: gpu submission cpu 12\n");
}

typedef struct {
    b3b_gpu *gpu;
    b3b_data *data;
    uint32_t checksum;
} b3b_gpu_state;

static void b3b_gpu_set_pointer(ze_kernel_handle_t kernel, uint32_t index,
                                const void *pointer) {
    b3b_ze_check("zeKernelSetArgumentValue",
                 zeKernelSetArgumentValue(kernel, index, sizeof(pointer), &pointer));
}

static void b3b_gpu_pass(void *opaque) {
    b3b_gpu_state *state = opaque;
    b3b_gpu *gpu = state->gpu;
    b3b_data *data = state->data;
    size_t matrix_blocks = (size_t)B3B_ROWS * B3B_BLOCKS;
    size_t weight_bytes = matrix_blocks * 16;
    ze_group_count_t groups = { B3B_ROWS / 8, 1, 1 };

    b3b_gpu_set_pointer(gpu->matvec_kernel, 2, data->activation);
    b3b_gpu_set_pointer(gpu->matvec_kernel, 3, data->activation_scales);
    b3b_gpu_set_pointer(gpu->matvec_kernel, 4, data->activation_sigma);
    b3b_gpu_set_pointer(gpu->matvec_kernel, 5, data->output);
    for (int rep = 0; rep < B3B_REPLICAS; rep++) {
        b3b_gpu_set_pointer(gpu->matvec_kernel, 0,
                            data->weights + (size_t)rep * weight_bytes);
        b3b_gpu_set_pointer(gpu->matvec_kernel, 1,
                            data->weight_scales + (size_t)rep * matrix_blocks);
        b3b_ze_check("zeCommandListAppendLaunchKernel",
                     zeCommandListAppendLaunchKernel(gpu->commands, gpu->matvec_kernel,
                                                     &groups, NULL, 0, NULL));
    }
    b3b_ze_check("zeCommandListHostSynchronize",
                 zeCommandListHostSynchronize(gpu->commands, UINT64_MAX));
}

static void b14_gpu_read_pass(void *opaque) {
    b3b_gpu_state *state = opaque;
    b3b_gpu *gpu = state->gpu;
    b3b_data *data = state->data;
    uint64_t blocks = (uint64_t)B3B_REPLICAS * B3B_ROWS * B3B_BLOCKS;
    uint64_t weight_vectors = blocks * 16 / 16;
    uint64_t scale_vectors = blocks * 2 / 16;
    ze_group_count_t groups = { 64, 1, 1 };

    b3b_gpu_set_pointer(gpu->read_kernel, 0, data->weights);
    b3b_ze_check("zeKernelSetArgumentValue weight vectors",
                 zeKernelSetArgumentValue(gpu->read_kernel, 1,
                                          sizeof weight_vectors, &weight_vectors));
    b3b_gpu_set_pointer(gpu->read_kernel, 2, data->weight_scales);
    b3b_ze_check("zeKernelSetArgumentValue scale vectors",
                 zeKernelSetArgumentValue(gpu->read_kernel, 3,
                                          sizeof scale_vectors, &scale_vectors));
    b3b_gpu_set_pointer(gpu->read_kernel, 4, data->output);
    b3b_ze_check("zeCommandListAppendLaunchKernel read",
                 zeCommandListAppendLaunchKernel(gpu->commands, gpu->read_kernel,
                                                 &groups, NULL, 0, NULL));
    b3b_ze_check("zeCommandListHostSynchronize read",
                 zeCommandListHostSynchronize(gpu->commands, UINT64_MAX));

    uint32_t sum = 0;
    uint32_t *partial = (uint32_t *)data->output;
    for (int i = 0; i < 64 * 16; i++) sum += partial[i];
    state->checksum = sum;
}

static void b3b_gpu_destroy(b3b_gpu *gpu, b3b_data *data) {
    if (data->output) zeMemFree(gpu->context, data->output);
    if (data->activation_sigma) zeMemFree(gpu->context, data->activation_sigma);
    if (data->activation_scales) zeMemFree(gpu->context, data->activation_scales);
    if (data->activation) zeMemFree(gpu->context, data->activation);
    if (data->weight_scales) zeMemFree(gpu->context, data->weight_scales);
    if (data->weights) zeMemFree(gpu->context, data->weights);
    if (gpu->read_kernel) zeKernelDestroy(gpu->read_kernel);
    if (gpu->matvec_kernel) zeKernelDestroy(gpu->matvec_kernel);
    if (gpu->module) zeModuleDestroy(gpu->module);
    if (gpu->commands) zeCommandListDestroy(gpu->commands);
    if (gpu->context) zeContextDestroy(gpu->context);
}

static void b3b_allocate_cpu(b3b_data *data) {
    size_t blocks = (size_t)B3B_REPLICAS * B3B_ROWS * B3B_BLOCKS;
    data->weights = b3b_alloc(blocks * 16);
    data->weight_scales = b3b_alloc(blocks * sizeof(*data->weight_scales));
    data->activation = b3b_alloc(B3B_N * sizeof(*data->activation));
    data->activation_scales = b3b_alloc(B3B_BLOCKS * sizeof(*data->activation_scales));
    data->activation_sigma = b3b_alloc(B3B_BLOCKS * sizeof(*data->activation_sigma));
    data->output = b3b_alloc(B3B_ROWS * sizeof(*data->output));
}

static void b3b_allocate_gpu(b3b_gpu *gpu, b3b_data *data) {
    size_t blocks = (size_t)B3B_REPLICAS * B3B_ROWS * B3B_BLOCKS;
    data->weights = b3b_gpu_alloc(gpu, blocks * 16);
    data->weight_scales = b3b_gpu_alloc(gpu, blocks * sizeof(*data->weight_scales));
    data->activation = b3b_gpu_alloc(gpu, B3B_N * sizeof(*data->activation));
    data->activation_scales = b3b_gpu_alloc(gpu, B3B_BLOCKS * sizeof(*data->activation_scales));
    data->activation_sigma = b3b_gpu_alloc(gpu, B3B_BLOCKS * sizeof(*data->activation_sigma));
    data->output = b3b_gpu_alloc(gpu, B3B_ROWS * sizeof(*data->output));
}

static void b3b_free_cpu(b3b_data *data) {
    free(data->output);
    free(data->activation_sigma);
    free(data->activation_scales);
    free(data->activation);
    free(data->weight_scales);
    free(data->weights);
}

static double b3b_parse_nonnegative(const char *name, const char *value) {
    char *end;
    errno = 0;
    double parsed = strtod(value, &end);
    if (errno || *end || !isfinite(parsed) || parsed < 0.0)
        xe_fatal("b3b: invalid %s: %s", name, value);
    return parsed;
}

int main(int argc, char **argv) {
    if (argc < 2 || (strcmp(argv[1], "cpu") && strcmp(argv[1], "gpu") &&
                     strcmp(argv[1], "cpu-read") && strcmp(argv[1], "gpu-read"))) {
        fprintf(stderr, "usage: %s cpu|gpu|cpu-read|gpu-read [--seconds N] [--delay N] [--spv PATH]\n", argv[0]);
        return 1;
    }

    const char *mode = argv[1];
    const char *spv_path = "bench/bench_b3b.spv";
    double seconds = 60.0;
    double delay = 45.0;
    for (int i = 2; i < argc; i++) {
        if (!strcmp(argv[i], "--seconds") && i + 1 < argc)
            seconds = b3b_parse_nonnegative("seconds", argv[++i]);
        else if (!strcmp(argv[i], "--delay") && i + 1 < argc)
            delay = b3b_parse_nonnegative("delay", argv[++i]);
        else if (!strcmp(argv[i], "--spv") && i + 1 < argc)
            spv_path = argv[++i];
        else
            xe_fatal("b3b: unknown or incomplete option: %s", argv[i]);
    }
    if (seconds <= 0.0) xe_fatal("b3b: seconds must be positive");

    b3b_print_power_policy();
    printf("b3b: mode %s rows %d blocks %d replicas %d useful %.6f GB/pass\n",
           mode, B3B_ROWS, B3B_BLOCKS, B3B_REPLICAS,
           (double)B3B_REPLICAS * B3B_ROWS * B3B_BLOCKS * 18.0 / 1e9);

    b3b_data data = {0};
    if (!strcmp(mode, "cpu") || !strcmp(mode, "cpu-read")) {
        b3b_allocate_cpu(&data);
        b3b_fill(&data);
#ifndef B3B_LEGACY_LAYOUT
        b3b_repack_cpu_x8(&data);
#endif
        b3b_cpu_state state;
        b3b_cpu_init(&state, &data);
        b3b_pass_fn pass = !strcmp(mode, "cpu") ? b3b_cpu_pass : b14_cpu_read_pass;
        pass(&state);
        if (!strcmp(mode, "cpu")) b3b_verify(mode, &data);
        else b14_verify_checksum(mode, state.checksum, &data);
        b3b_delay(delay);
        b3b_run(mode, seconds, pass, &state);
        b3b_cpu_destroy(&state);
        b3b_free_cpu(&data);
        return 0;
    }

    b3b_gpu gpu;
    b3b_gpu_init(&gpu, spv_path);
    b3b_allocate_gpu(&gpu, &data);
    b3b_fill(&data);
    b3b_gpu_state state = { &gpu, &data, 0 };
    b3b_pass_fn pass = !strcmp(mode, "gpu") ? b3b_gpu_pass : b14_gpu_read_pass;
    pass(&state);
    if (!strcmp(mode, "gpu")) b3b_verify(mode, &data);
    else b14_verify_checksum(mode, state.checksum, &data);
    b3b_delay(delay);
    b3b_gpu_sampler sampler;
    b3b_gpu_sampler_start(&sampler);
    b3b_run(mode, seconds, pass, &state);
    b3b_gpu_sampler_stop(&sampler);
    b3b_gpu_destroy(&gpu, &data);
    return 0;
}
