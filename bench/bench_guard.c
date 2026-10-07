#define _GNU_SOURCE
#define _POSIX_C_SOURCE 200809L

#include <ctype.h>
#include <dirent.h>
#include <errno.h>
#include <fcntl.h>
#include <limits.h>
#include <signal.h>
#include <stdarg.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/file.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <sys/utsname.h>
#include <sys/wait.h>
#include <time.h>
#include <unistd.h>

typedef struct {
    uint64_t available_kib;
    uint64_t swap_free_kib;
} memory_state;

typedef struct {
    uint64_t rss_kib;
    uint64_t pss_kib;
    uint64_t swap_kib;
    int processes;
} group_state;

static volatile sig_atomic_t guard_signal;

static int64_t monotonic_ms(void) {
    struct timespec ts;
    if (clock_gettime(CLOCK_MONOTONIC, &ts) != 0) return 0;
    return (int64_t)ts.tv_sec * 1000 + ts.tv_nsec / 1000000;
}

static int64_t realtime_ms(void) {
    struct timespec ts;
    if (clock_gettime(CLOCK_REALTIME, &ts) != 0) return 0;
    return (int64_t)ts.tv_sec * 1000 + ts.tv_nsec / 1000000;
}

static void on_signal(int signal_number) {
    guard_signal = signal_number;
}

static int mkdirs(const char *path, mode_t mode) {
    char copy[PATH_MAX];
    int n = snprintf(copy, sizeof copy, "%s", path);
    if (n < 1 || (size_t)n >= sizeof copy) return 0;
    size_t length = strlen(copy);
    while (length > 1 && copy[length - 1] == '/') copy[--length] = '\0';
    for (char *p = copy + (copy[0] == '/'); *p; p++) {
        if (*p != '/') continue;
        *p = '\0';
        if (*copy && mkdir(copy, mode) != 0 && errno != EEXIST) return 0;
        *p = '/';
    }
    return mkdir(copy, mode) == 0 || errno == EEXIST;
}

static int path_make(char *out, size_t cap, const char *a, const char *b) {
    int n = snprintf(out, cap, "%s/%s", a, b);
    return n > 0 && (size_t)n < cap;
}

static int read_memory(memory_state *state) {
    FILE *file = fopen("/proc/meminfo", "r");
    if (!file) return 0;
    state->available_kib = 0;
    state->swap_free_kib = 0;
    char key[64];
    unsigned long long value;
    char unit[32];
    while (fscanf(file, "%63s %llu %31s", key, &value, unit) == 3) {
        if (!strcmp(key, "MemAvailable:")) state->available_kib = value;
        if (!strcmp(key, "SwapFree:")) state->swap_free_kib = value;
    }
    fclose(file);
    return state->available_kib != 0;
}

static int read_int_file(const char *path, int *value) {
    FILE *file = fopen(path, "r");
    if (!file) return 0;
    long parsed;
    int ok = fscanf(file, "%ld", &parsed) == 1 && parsed >= INT_MIN &&
             parsed <= INT_MAX;
    fclose(file);
    if (ok) *value = (int)parsed;
    return ok;
}

static int read_ac(void) {
    const char *paths[] = {
        "/sys/class/power_supply/ACAD/online",
        "/sys/class/power_supply/AC/online"
    };
    int value;
    for (size_t i = 0; i < sizeof paths / sizeof paths[0]; i++)
        if (read_int_file(paths[i], &value)) return value;
    return -1;
}

static int read_max_temperature(void) {
    DIR *dir = opendir("/sys/class/thermal");
    if (!dir) return -1;
    int maximum = -1;
    struct dirent *entry;
    while ((entry = readdir(dir))) {
        if (strncmp(entry->d_name, "thermal_zone", 12)) continue;
        char path[PATH_MAX];
        int n = snprintf(path, sizeof path, "/sys/class/thermal/%s/temp",
                         entry->d_name);
        int value;
        if (n > 0 && (size_t)n < sizeof path && read_int_file(path, &value) &&
            value >= 10000 && value <= 125000 && value > maximum)
            maximum = value;
    }
    closedir(dir);
    return maximum;
}

static double read_load(void) {
    FILE *file = fopen("/proc/loadavg", "r");
    if (!file) return -1.0;
    double value = -1.0;
    fscanf(file, "%lf", &value);
    fclose(file);
    return value;
}

static void read_pressure(double *some, double *full) {
    *some = -1.0;
    *full = -1.0;
    FILE *file = fopen("/proc/pressure/memory", "r");
    if (!file) return;
    char line[512];
    while (fgets(line, sizeof line, file)) {
        double value;
        if (!strncmp(line, "some ", 5) &&
            sscanf(line, "some avg10=%lf", &value) == 1) *some = value;
        if (!strncmp(line, "full ", 5) &&
            sscanf(line, "full avg10=%lf", &value) == 1) *full = value;
    }
    fclose(file);
}

static int read_parent(pid_t pid, pid_t *parent) {
    char path[64];
    snprintf(path, sizeof path, "/proc/%ld/status", (long)pid);
    FILE *file = fopen(path, "r");
    if (!file) return 0;
    char line[256];
    int ok = 0;
    while (fgets(line, sizeof line, file)) {
        long value;
        if (sscanf(line, "PPid: %ld", &value) == 1) {
            *parent = (pid_t)value;
            ok = 1;
            break;
        }
    }
    fclose(file);
    return ok;
}

static int is_ancestor(pid_t candidate) {
    pid_t current = getpid();
    for (int depth = 0; depth < 128 && current > 1; depth++) {
        pid_t parent;
        if (!read_parent(current, &parent)) return 0;
        if (parent == candidate) return 1;
        if (parent == current) return 0;
        current = parent;
    }
    return 0;
}

static int pid_from_name(const char *name, pid_t *pid) {
    if (!*name) return 0;
    long value = 0;
    for (const unsigned char *p = (const unsigned char *)name; *p; p++) {
        if (!isdigit(*p)) return 0;
        value = value * 10 + (*p - '0');
        if (value > INT_MAX) return 0;
    }
    if (value < 1) return 0;
    *pid = (pid_t)value;
    return 1;
}

static int read_comm(pid_t pid, char *out, size_t cap) {
    char path[64];
    snprintf(path, sizeof path, "/proc/%ld/comm", (long)pid);
    FILE *file = fopen(path, "r");
    if (!file) return 0;
    int ok = fgets(out, (int)cap, file) != NULL;
    fclose(file);
    if (!ok) return 0;
    out[strcspn(out, "\r\n")] = '\0';
    return 1;
}

static int known_heavy_comm(const char *comm) {
    if (!strcmp(comm, "xenolith")) return 1;
    if (!strncmp(comm, "llama-", 6)) return 1;
    if (!strncmp(comm, "test_prefill", 12)) return 1;
    if (!strncmp(comm, "test_runtime_mo", 15)) return 1;
    if (!strncmp(comm, "test_model_dec", 14)) return 1;
    if (!strncmp(comm, "test_fixture_d", 14)) return 1;
    if (!strncmp(comm, "test_snapshot_", 14)) return 1;
    if (!strncmp(comm, "bench_tg", 8)) return 1;
    if (!strncmp(comm, "bench_prefill", 13)) return 1;
    if (!strncmp(comm, "ollama", 6)) return 1;
    if (!strncmp(comm, "kobold", 6)) return 1;
    return 0;
}

static int maps_model(pid_t pid) {
    char path[64];
    snprintf(path, sizeof path, "/proc/%ld/maps", (long)pid);
    FILE *file = fopen(path, "r");
    if (!file) return 0;
    char line[4096];
    int found = 0;
    while (fgets(line, sizeof line, file)) {
        if (strstr(line, ".gguf") || strstr(line, "/models/")) {
            found = 1;
            break;
        }
    }
    fclose(file);
    return found;
}

static int cmdline_heavy(pid_t pid) {
    char path[64];
    snprintf(path, sizeof path, "/proc/%ld/cmdline", (long)pid);
    int fd = open(path, O_RDONLY | O_CLOEXEC);
    if (fd < 0) return 0;
    char data[8192];
    ssize_t got = read(fd, data, sizeof data - 1);
    close(fd);
    if (got <= 0) return 0;
    data[got] = '\0';
    for (ssize_t i = 0; i < got; i++) if (!data[i]) data[i] = ' ';
    return strstr(data, " vllm") || strstr(data, "text-generation-launcher") ||
           strstr(data, "koboldcpp") || strstr(data, "ollama_llama_server");
}

static pid_t external_heavy(pid_t owned_group, char *name, size_t name_cap) {
    DIR *dir = opendir("/proc");
    if (!dir) return -1;
    pid_t self = getpid();
    struct dirent *entry;
    pid_t found = 0;
    while ((entry = readdir(dir))) {
        pid_t pid;
        if (!pid_from_name(entry->d_name, &pid) || pid == self ||
            is_ancestor(pid)) continue;
        if (owned_group > 0 && getpgid(pid) == owned_group) continue;
        char comm[128];
        if (!read_comm(pid, comm, sizeof comm)) continue;
        if (!known_heavy_comm(comm) && !maps_model(pid) &&
            !cmdline_heavy(pid)) continue;
        found = pid;
        snprintf(name, name_cap, "%s", comm);
        break;
    }
    closedir(dir);
    return found;
}

static void read_status_values(pid_t pid, group_state *state) {
    char path[64];
    snprintf(path, sizeof path, "/proc/%ld/status", (long)pid);
    FILE *file = fopen(path, "r");
    if (!file) return;
    char line[256];
    while (fgets(line, sizeof line, file)) {
        unsigned long long value;
        if (sscanf(line, "VmRSS: %llu kB", &value) == 1)
            state->rss_kib += value;
        if (sscanf(line, "VmSwap: %llu kB", &value) == 1)
            state->swap_kib += value;
    }
    fclose(file);
    snprintf(path, sizeof path, "/proc/%ld/smaps_rollup", (long)pid);
    file = fopen(path, "r");
    if (!file) return;
    while (fgets(line, sizeof line, file)) {
        unsigned long long value;
        if (sscanf(line, "Pss: %llu kB", &value) == 1) {
            state->pss_kib += value;
            break;
        }
    }
    fclose(file);
}

static group_state read_group(pid_t group) {
    group_state state = { 0 };
    if (group <= 0) return state;
    DIR *dir = opendir("/proc");
    if (!dir) return state;
    struct dirent *entry;
    while ((entry = readdir(dir))) {
        pid_t pid;
        if (!pid_from_name(entry->d_name, &pid)) continue;
        if (getpgid(pid) != group) continue;
        state.processes++;
        read_status_values(pid, &state);
    }
    closedir(dir);
    return state;
}

static uint64_t threshold_kib(const char *name, uint64_t default_gib) {
    const char *text = getenv(name);
    if (!text || !*text) return default_gib * 1024 * 1024;
    char *end;
    errno = 0;
    unsigned long long value = strtoull(text, &end, 10);
    if (errno || *end || value > UINT64_MAX / (1024 * 1024))
        return default_gib * 1024 * 1024;
    return value * 1024 * 1024;
}

static uint64_t threshold_mib(const char *name, uint64_t default_mib) {
    const char *text = getenv(name);
    if (!text || !*text) return default_mib * 1024;
    char *end;
    errno = 0;
    unsigned long long value = strtoull(text, &end, 10);
    if (errno || *end || value > UINT64_MAX / 1024)
        return default_mib * 1024;
    return value * 1024;
}

static int threshold_temp_mc(const char *name, unsigned default_celsius) {
    const char *text = getenv(name);
    if (!text || !*text) return (int)default_celsius * 1000;
    char *end;
    errno = 0;
    unsigned long value = strtoul(text, &end, 10);
    if (errno || *end || value < 40 || value > 125)
        return (int)default_celsius * 1000;
    return (int)value * 1000;
}

static void event(FILE *file, const char *type, const char *format, ...) {
    fprintf(file, "{\"wall_ms\":%lld,\"monotonic_ms\":%lld,\"event\":\"%s\"",
            (long long)realtime_ms(), (long long)monotonic_ms(), type);
    if (format && *format) {
        fputc(',', file);
        va_list arguments;
        va_start(arguments, format);
        vfprintf(file, format, arguments);
        va_end(arguments);
    }
    fputs("}\n", file);
    fflush(file);
}

static void write_quoted(FILE *file, const char *text) {
    fputc('\'', file);
    for (const char *p = text; *p; p++) {
        if (*p == '\'') fputs("'\\''", file);
        else fputc(*p, file);
    }
    fputc('\'', file);
}

static void write_command(const char *path, int argc, char **argv,
                          const char *workdir) {
    FILE *file = fopen(path, "w");
    if (!file) return;
    fputs("cwd=", file);
    write_quoted(file, workdir);
    fputc('\n', file);
    for (int i = 0; i < argc; i++) {
        if (i) fputc(' ', file);
        write_quoted(file, argv[i]);
    }
    fputc('\n', file);
    fclose(file);
}

static void write_environment(const char *path) {
    static const char *names[] = {
        "PATH", "OMP_NUM_THREADS", "OMP_PROC_BIND", "OMP_PLACES",
        "GGML_VK_VISIBLE_DEVICES", "GGML_CUDA_VISIBLE_DEVICES",
        "ZES_ENABLE_SYSMAN", "ZE_AFFINITY_MASK", "XDG_STATE_HOME"
    };
    FILE *file = fopen(path, "w");
    if (!file) return;
    for (size_t i = 0; i < sizeof names / sizeof names[0]; i++) {
        const char *value = getenv(names[i]);
        if (value) fprintf(file, "%s=%s\n", names[i], value);
    }
    fclose(file);
}

static void copy_file(FILE *out, const char *path) {
    FILE *file = fopen(path, "r");
    if (!file) {
        fprintf(out, "%s: %s\n", path, strerror(errno));
        return;
    }
    fprintf(out, "[%s]\n", path);
    char data[4096];
    size_t got;
    while ((got = fread(data, 1, sizeof data, file))) fwrite(data, 1, got, out);
    fputc('\n', out);
    fclose(file);
}

static void write_snapshot(const char *path) {
    FILE *file = fopen(path, "w");
    if (!file) return;
    struct utsname uts;
    if (uname(&uts) == 0)
        fprintf(file, "uname=%s %s %s %s %s\n", uts.sysname, uts.nodename,
                uts.release, uts.version, uts.machine);
    fprintf(file, "pid=%ld ac=%d max_temp_mc=%d load1=%.3f\n",
            (long)getpid(), read_ac(), read_max_temperature(), read_load());
    copy_file(file, "/proc/meminfo");
    copy_file(file, "/proc/pressure/memory");
    copy_file(file, "/proc/loadavg");
    copy_file(file, "/sys/devices/system/cpu/intel_pstate/status");
    copy_file(file, "/sys/devices/system/cpu/cpu0/cpufreq/scaling_governor");
    copy_file(file, "/sys/devices/system/cpu/cpu0/cpufreq/energy_performance_preference");
    fclose(file);
}

static int engine_lock_path(char *out, size_t cap) {
    const char *state = getenv("XDG_STATE_HOME");
    int n;
    if (state && *state == '/') n = snprintf(out, cap, "%s/xenolith", state);
    else {
        const char *home = getenv("HOME");
        if (!home || *home != '/') return 0;
        n = snprintf(out, cap, "%s/.local/state/xenolith", home);
    }
    if (n < 1 || (size_t)n >= cap || !mkdirs(out, 0700)) return 0;
    size_t length = strlen(out);
    n = snprintf(out + length, cap - length, "/engine.lock");
    return n > 0 && (size_t)n < cap - length;
}

static int lock_file(const char *path, int write_holder) {
    int fd = open(path, O_RDWR | O_CREAT | O_CLOEXEC, 0600);
    if (fd < 0) return -1;
    if (flock(fd, LOCK_EX | LOCK_NB) != 0) {
        int saved_errno = errno;
        close(fd);
        errno = saved_errno;
        return -1;
    }
    if (write_holder) {
        char holder[256];
        int n = snprintf(holder, sizeof holder, "pid %ld benchmark guard\n",
                         (long)getpid());
        if (n > 0 && (size_t)n < sizeof holder) {
            ssize_t written = pwrite(fd, holder, (size_t)n, 0);
            if (written == n) ftruncate(fd, n);
        }
    }
    return fd;
}

static int terminate_group(pid_t group, pid_t child, int *status) {
    kill(-group, SIGTERM);
    int64_t deadline = monotonic_ms() + 5000;
    while (monotonic_ms() < deadline) {
        pid_t result = waitpid(child, status, WNOHANG);
        if (result == child) return 1;
        struct timespec pause = { .tv_sec = 0, .tv_nsec = 100000000 };
        nanosleep(&pause, NULL);
    }
    kill(-group, SIGKILL);
    while (waitpid(child, status, 0) < 0 && errno == EINTR) {}
    return 1;
}

int main(int argc, char **argv) {
    if (argc < 7) {
        fprintf(stderr, "usage: %s RUN_ROOT LABEL TIMEOUT_SECONDS WORKDIR -- COMMAND [ARG...]\n", argv[0]);
        return 2;
    }
    if (strcmp(argv[5], "--")) {
        fputs("missing -- before command\n", stderr);
        return 2;
    }
    char *end;
    errno = 0;
    long timeout_seconds = strtol(argv[3], &end, 10);
    if (errno || *end || timeout_seconds < 1 || timeout_seconds > 86400) {
        fputs("invalid timeout\n", stderr);
        return 2;
    }
    char run_dir[PATH_MAX];
    if (!path_make(run_dir, sizeof run_dir, argv[1], argv[2])) return 2;
    if (mkdir(run_dir, 0700) != 0) {
        fprintf(stderr, "%s: %s\n", run_dir, strerror(errno));
        return 2;
    }
    char path[PATH_MAX];
    path_make(path, sizeof path, run_dir, "events.jsonl");
    FILE *events = fopen(path, "w");
    if (!events) return 2;
    path_make(path, sizeof path, run_dir, "command.txt");
    write_command(path, argc - 6, argv + 6, argv[4]);
    path_make(path, sizeof path, run_dir, "environment.txt");
    write_environment(path);
    path_make(path, sizeof path, run_dir, "system-before.txt");
    write_snapshot(path);

    uint64_t min_start = threshold_kib("BENCH_MIN_START_GIB", 24);
    uint64_t min_run = threshold_kib("BENCH_MIN_RUN_GIB", 2);
    uint64_t swap_notice_growth = threshold_mib(
        "BENCH_SWAP_NOTICE_MIB", 512);
    uint64_t min_recovered = threshold_kib("BENCH_MIN_RECOVERED_GIB", 24);
    int max_temp_mc = threshold_temp_mc("BENCH_MAX_TEMP_C", 95);
    event(events, "guard_start",
          "\"label\":\"%s\",\"timeout_seconds\":%ld,\"min_start_kib\":%llu,\"min_run_kib\":%llu,\"swap_policy\":\"observe\",\"swap_notice_growth_kib\":%llu,\"min_recovered_kib\":%llu,\"max_temp_mc\":%d",
          argv[2], timeout_seconds, (unsigned long long)min_start,
          (unsigned long long)min_run, (unsigned long long)swap_notice_growth,
          (unsigned long long)min_recovered, max_temp_mc);

    char lock_path[PATH_MAX];
    int n = snprintf(lock_path, sizeof lock_path, "%s/model-heavy.lock", argv[1]);
    if (n < 1 || (size_t)n >= sizeof lock_path) return 2;
    int global_lock = lock_file(lock_path, 0);
    if (global_lock < 0) {
        int saved_errno = errno;
        event(events, "preflight_failed",
              "\"reason\":\"global_lock\",\"errno\":%d", saved_errno);
        fclose(events);
        return 125;
    }
    int engine_lock = -1;
    char actual_engine_lock[PATH_MAX];
    if (!engine_lock_path(actual_engine_lock, sizeof actual_engine_lock)) {
        event(events, "preflight_failed", "\"reason\":\"engine_lock_path\"");
        close(global_lock);
        fclose(events);
        return 125;
    }
    engine_lock = lock_file(actual_engine_lock, 1);
    if (engine_lock < 0) {
        int saved_errno = errno;
        event(events, "preflight_failed",
              "\"reason\":\"engine_lock\",\"errno\":%d", saved_errno);
        close(global_lock);
        fclose(events);
        return 125;
    }

    memory_state initial;
    char heavy_name[128] = "";
    pid_t heavy_pid = external_heavy(0, heavy_name, sizeof heavy_name);
    int ac = read_ac();
    int initial_temp_mc = read_max_temperature();
    if (!read_memory(&initial) || initial.available_kib < min_start ||
        ac != 1 || heavy_pid != 0 || initial_temp_mc > max_temp_mc) {
        event(events, "preflight_failed",
              "\"reason\":\"state\",\"mem_available_kib\":%llu,\"swap_free_kib\":%llu,\"ac\":%d,\"max_temp_mc\":%d,\"heavy_pid\":%ld,\"heavy_name\":\"%s\"",
              (unsigned long long)initial.available_kib,
              (unsigned long long)initial.swap_free_kib, ac, initial_temp_mc,
              (long)heavy_pid, heavy_name);
        if (engine_lock >= 0) close(engine_lock);
        close(global_lock);
        fclose(events);
        return 125;
    }
    event(events, "preflight_passed",
          "\"mem_available_kib\":%llu,\"swap_free_kib\":%llu,\"ac\":%d,\"engine_lock_mode\":\"%s\"",
          (unsigned long long)initial.available_kib,
          (unsigned long long)initial.swap_free_kib, ac,
          "guard_inherited");

    char stdout_path[PATH_MAX];
    char stderr_path[PATH_MAX];
    path_make(stdout_path, sizeof stdout_path, run_dir, "stdout.log");
    path_make(stderr_path, sizeof stderr_path, run_dir, "stderr.log");
    int stdout_fd = open(stdout_path, O_WRONLY | O_CREAT | O_EXCL | O_CLOEXEC, 0600);
    int stderr_fd = open(stderr_path, O_WRONLY | O_CREAT | O_EXCL | O_CLOEXEC, 0600);
    if (stdout_fd < 0 || stderr_fd < 0) {
        event(events, "launch_failed", "\"reason\":\"log_open\"");
        if (stdout_fd >= 0) close(stdout_fd);
        if (stderr_fd >= 0) close(stderr_fd);
        if (engine_lock >= 0) close(engine_lock);
        close(global_lock);
        fclose(events);
        return 125;
    }
    pid_t child = fork();
    if (child == 0) {
        setpgid(0, 0);
        char lock_fd_text[32];
        snprintf(lock_fd_text, sizeof lock_fd_text, "%d", engine_lock);
        fcntl(engine_lock, F_SETFD, 0);
        setenv("XENOLITH_ENGINE_LOCK_FD", lock_fd_text, 1);
        if (chdir(argv[4]) != 0) _exit(126);
        if (dup2(stdout_fd, STDOUT_FILENO) < 0 ||
            dup2(stderr_fd, STDERR_FILENO) < 0) _exit(126);
        close(stdout_fd);
        close(stderr_fd);
        execvp(argv[6], argv + 6);
        dprintf(STDERR_FILENO, "%s: %s\n", argv[6], strerror(errno));
        _exit(127);
    }
    close(stdout_fd);
    close(stderr_fd);
    if (child < 0) {
        event(events, "launch_failed", "\"reason\":\"fork\"");
        if (engine_lock >= 0) close(engine_lock);
        close(global_lock);
        fclose(events);
        return 125;
    }
    setpgid(child, child);
    event(events, "process_started", "\"pid\":%ld,\"pgid\":%ld",
          (long)child, (long)child);
    path_make(path, sizeof path, run_dir, "telemetry.tsv");
    FILE *telemetry = fopen(path, "w");
    if (telemetry) {
        fputs("wall_ms\tmonotonic_ms\tmem_available_kib\tswap_free_kib\tmemory_some_avg10\tmemory_full_avg10\tac\tmax_temp_mc\tload1\tgroup_processes\tgroup_rss_kib\tgroup_pss_kib\tgroup_swap_kib\texternal_heavy_pid\texternal_heavy_name\n", telemetry);
        fflush(telemetry);
    }

    struct sigaction action = { 0 };
    action.sa_handler = on_signal;
    sigemptyset(&action.sa_mask);
    sigaction(SIGINT, &action, NULL);
    sigaction(SIGTERM, &action, NULL);
    sigaction(SIGHUP, &action, NULL);

    int status = 0;
    int reaped = 0;
    int aborted = 0;
    int swap_notice_sent = 0;
    const char *abort_reason = "";
    int64_t start = monotonic_ms();
    while (!reaped) {
        pid_t waited = waitpid(child, &status, WNOHANG);
        if (waited == child) {
            reaped = 1;
            break;
        }
        memory_state current = { 0 };
        read_memory(&current);
        double pressure_some;
        double pressure_full;
        read_pressure(&pressure_some, &pressure_full);
        group_state group = read_group(child);
        heavy_name[0] = '\0';
        heavy_pid = external_heavy(child, heavy_name, sizeof heavy_name);
        ac = read_ac();
        int current_temp_mc = read_max_temperature();
        if (telemetry) {
            fprintf(telemetry,
                    "%lld\t%lld\t%llu\t%llu\t%.3f\t%.3f\t%d\t%d\t%.3f\t%d\t%llu\t%llu\t%llu\t%ld\t%s\n",
                    (long long)realtime_ms(), (long long)monotonic_ms(),
                    (unsigned long long)current.available_kib,
                    (unsigned long long)current.swap_free_kib,
                    pressure_some, pressure_full, ac,
                    current_temp_mc, read_load(), group.processes,
                    (unsigned long long)group.rss_kib,
                    (unsigned long long)group.pss_kib,
                    (unsigned long long)group.swap_kib,
                    (long)heavy_pid, heavy_name);
            fflush(telemetry);
        }
        if (!swap_notice_sent && initial.swap_free_kib > current.swap_free_kib &&
            initial.swap_free_kib - current.swap_free_kib > swap_notice_growth) {
            event(events, "swap_growth_notice",
                  "\"growth_kib\":%llu,\"mem_available_kib\":%llu",
                  (unsigned long long)(initial.swap_free_kib - current.swap_free_kib),
                  (unsigned long long)current.available_kib);
            swap_notice_sent = 1;
        }
        if (guard_signal) {
            aborted = 1;
            abort_reason = "guard_signal";
        } else if (monotonic_ms() - start >= timeout_seconds * 1000LL) {
            aborted = 1;
            abort_reason = "timeout";
        } else if (current.available_kib < min_run) {
            aborted = 1;
            abort_reason = "memory_floor";
        } else if (ac != 1) {
            aborted = 1;
            abort_reason = "ac_offline";
        } else if (current_temp_mc > max_temp_mc) {
            aborted = 1;
            abort_reason = "temperature";
        } else if (heavy_pid > 0) {
            aborted = 1;
            abort_reason = "external_heavy_process";
        }
        if (aborted) {
            event(events, "abort",
                  "\"reason\":\"%s\",\"mem_available_kib\":%llu,\"swap_free_kib\":%llu,\"ac\":%d,\"max_temp_mc\":%d,\"heavy_pid\":%ld,\"heavy_name\":\"%s\"",
                  abort_reason,
                  (unsigned long long)current.available_kib,
                  (unsigned long long)current.swap_free_kib, ac,
                  current_temp_mc,
                  (long)heavy_pid, heavy_name);
            terminate_group(child, child, &status);
            reaped = 1;
            break;
        }
        struct timespec pause = { .tv_sec = 1, .tv_nsec = 0 };
        nanosleep(&pause, NULL);
    }
    if (telemetry) fclose(telemetry);

    group_state leftover = read_group(child);
    if (leftover.processes) {
        kill(-child, SIGKILL);
        event(events, "cleanup_kill",
              "\"processes\":%d,\"rss_kib\":%llu,\"swap_kib\":%llu",
              leftover.processes, (unsigned long long)leftover.rss_kib,
              (unsigned long long)leftover.swap_kib);
        if (!aborted) {
            aborted = 1;
            abort_reason = "leftover_processes";
        }
    }

    int recovered = 0;
    int64_t recovery_deadline = monotonic_ms() + 180000;
    memory_state final_memory = { 0 };
    while (monotonic_ms() < recovery_deadline) {
        if (read_memory(&final_memory) &&
            final_memory.available_kib >= min_recovered) {
            recovered = 1;
            break;
        }
        struct timespec pause = { .tv_sec = 1, .tv_nsec = 0 };
        nanosleep(&pause, NULL);
    }
    if (!recovered) {
        aborted = 1;
        abort_reason = "memory_not_recovered";
        event(events, "cleanup_failed",
              "\"reason\":\"memory_not_recovered\",\"mem_available_kib\":%llu,\"swap_free_kib\":%llu",
              (unsigned long long)final_memory.available_kib,
              (unsigned long long)final_memory.swap_free_kib);
    }
    path_make(path, sizeof path, run_dir, "system-after.txt");
    write_snapshot(path);

    int exit_code = 125;
    int exit_signal = 0;
    if (!aborted && WIFEXITED(status)) exit_code = WEXITSTATUS(status);
    else if (!aborted && WIFSIGNALED(status)) {
        exit_signal = WTERMSIG(status);
        exit_code = 128 + exit_signal;
    }
    event(events, "guard_finish",
          "\"exit_code\":%d,\"exit_signal\":%d,\"aborted\":%s,\"abort_reason\":\"%s\",\"elapsed_ms\":%lld,\"mem_available_kib\":%llu,\"swap_free_kib\":%llu,\"recovered\":%s",
          exit_code, exit_signal, aborted ? "true" : "false", abort_reason,
          (long long)(monotonic_ms() - start),
          (unsigned long long)final_memory.available_kib,
          (unsigned long long)final_memory.swap_free_kib,
          recovered ? "true" : "false");
    if (engine_lock >= 0) close(engine_lock);
    close(global_lock);
    fclose(events);
    return exit_code;
}
