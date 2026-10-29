/* Deterministic sysfs parser tests; do not probe CPUID or change affinity. */
#define _GNU_SOURCE
#include <dirent.h>
#include <errno.h>
#include <ftw.h>
#include <limits.h>
#include <sched.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <unistd.h>

#define CHECK(x) do { if (!(x)) { \
    fprintf(stderr, "cpu-topology:%d: %s\n", __LINE__, #x); exit(2); \
} } while (0)

static char fixture_root[PATH_MAX];
static const char sys_cpu_prefix[] = "/sys/devices/system/cpu/";

static const char *fixture_path(const char *path, char mapped[PATH_MAX]) {
    size_t prefix = strlen(sys_cpu_prefix);
    if (strncmp(path, sys_cpu_prefix, prefix)) return path;
    int n = snprintf(mapped, PATH_MAX, "%s/%s", fixture_root, path + prefix);
    CHECK(n > 0 && n < PATH_MAX);
    return mapped;
}

static FILE *fixture_fopen(const char *path, const char *mode) {
    char mapped[PATH_MAX];
    return fopen(fixture_path(path, mapped), mode);
}

static DIR *fixture_opendir(const char *path) {
    char mapped[PATH_MAX];
    return opendir(fixture_path(path, mapped));
}

#define fopen fixture_fopen
#define opendir fixture_opendir
#include "../cpu_topology.c"
#undef fopen
#undef opendir

static void make_parents(const char *path) {
    char part[PATH_MAX];
    CHECK(strlen(path) < sizeof part);
    strcpy(part, path);
    for (char *p = part + 1; *p; p++) {
        if (*p != '/') continue;
        *p = 0;
        CHECK(mkdir(part, 0700) == 0 || errno == EEXIST);
        *p = '/';
    }
}

static void write_bytes(const char *relative, const char *value, size_t size) {
    char path[PATH_MAX];
    int n = snprintf(path, sizeof path, "%s/%s", fixture_root, relative);
    CHECK(n > 0 && n < (int)sizeof path);
    make_parents(path);
    FILE *f = fopen(path, "w"); CHECK(f != NULL);
    CHECK(fwrite(value, 1, size, f) == size);
    CHECK(fclose(f) == 0);
}

static void write_text(const char *relative, const char *value) {
    write_bytes(relative, value, strlen(value));
}

static void check_mask(const char *value, int expected, const int *cpus, size_t count) {
    write_text("list", value);
    char path[PATH_MAX];
    int n = snprintf(path, sizeof path, "%s/list", fixture_root);
    CHECK(n > 0 && n < (int)sizeof path);
    cpu_set_t mask;
    CPU_ZERO(&mask);
    CHECK(read_mask(path, &mask) == expected);
    if (!expected) return;
    CHECK(CPU_COUNT(&mask) == (int)count);
    for (size_t i = 0; i < count; i++) CHECK(CPU_ISSET(cpus[i], &mask));
}

static void read_mask_cases(void) {
    const int a[] = {0, 1, 2, 3, 7, 9, 10};
    check_mask("0-3,7,9-10\n", 1, a, sizeof a / sizeof a[0]);
    const int b[] = {2, 4, 6, 8};
    check_mask("2,4,6,8", 1, b, sizeof b / sizeof b[0]);
    check_mask("2,4,6,8\n \t\n", 1, b, sizeof b / sizeof b[0]);
    check_mask("2,4,6,8garbage\n", 0, NULL, 0);
    check_mask("2,4,6,8\n9\n", 0, NULL, 0);
    check_mask("1,1\n", 0, NULL, 0);
    check_mask("1-3,3\n", 0, NULL, 0);
    check_mask("4-2\n", 0, NULL, 0);
    check_mask("1,,2\n", 0, NULL, 0);
    check_mask("1,\n", 0, NULL, 0);
    check_mask("1, 2\n", 0, NULL, 0);
    check_mask("-1\n", 0, NULL, 0);
    char outside[40]; snprintf(outside, sizeof outside, "%d\n", CPU_SETSIZE);
    check_mask(outside, 0, NULL, 0);
    size_t long_len = (size_t)CPU_SETSIZE * 12 + 100;
    char *long_line = malloc(long_len + 2); CHECK(long_line != NULL);
    memset(long_line, '9', long_len); long_line[long_len] = '\n'; long_line[long_len + 1] = 0;
    check_mask(long_line, 0, NULL, 0);
    free(long_line);
}

static void read_number_cases(void) {
    const char *path = "/sys/devices/system/cpu/number";
    const char *invalid[] = {
        "3garbage\n", "3 4\n", "3\n4\n", "-1\n", "\n", "",
        "999999999999999999999999999999\n"
    };
    for (size_t i = 0; i < sizeof invalid / sizeof invalid[0]; i++) {
        write_text("number", invalid[i]);
        int value = 99;
        CHECK(!read_number(path, &value));
        CHECK(value == 99);
    }
    char over_int[64];
    snprintf(over_int, sizeof over_int, "%llu", (unsigned long long)INT_MAX + 1);
    write_text("number", over_int);
    int value;
    CHECK(!read_number(path, &value));
    write_text("number", "3\n \t\n");
    CHECK(read_number(path, &value) && value == 3);
    write_text("number", "0");
    CHECK(read_number(path, &value) && value == 0);
    write_bytes("number", "3\0garbage", 9);
    CHECK(!read_number(path, &value));
}

static void physical_core_cases(void) {
    int package = 99, core = 99;
    CHECK(!xe_cpu_topology_core(20, &package, &core));
    CHECK(package == 99 && core == 99);
    write_text("cpu20/topology/physical_package_id", "0\n");
    CHECK(!xe_cpu_topology_core(20, &package, &core));
    CHECK(package == 99 && core == 99);
    write_text("cpu20/topology/core_id", "3\n");
    CHECK(xe_cpu_topology_core(20, &package, &core));
    CHECK(package == 0 && core == 3);

    const char *fields[] = {"physical_package_id", "core_id"};
    const char *invalid[] = {"3garbage\n", "3\n4\n", "-1\n", "99999999999999999999"};
    for (size_t i = 0; i < 2; i++) {
        char path[80];
        snprintf(path, sizeof path, "cpu20/topology/%s", fields[i]);
        for (size_t j = 0; j < sizeof invalid / sizeof invalid[0]; j++) {
            write_text(path, invalid[j]);
            package = core = 99;
            CHECK(!xe_cpu_topology_core(20, &package, &core));
            CHECK(package == 99 && core == 99);
        }
        write_text(path, "0\n");
    }
    CHECK(!xe_cpu_topology_core(-1, &package, &core));
    CHECK(!xe_cpu_topology_core(CPU_SETSIZE, &package, &core));
}

static void cache_cases(void) {
    cpu_set_t mask;
    write_text("cpu7/cache/index0/level", "2\n");
    write_text("cpu7/cache/index0/type", "Unified\n");
    write_text("cpu7/cache/index0/shared_cpu_list", "7-8\n");
    CHECK(cache_mask(7, 2, 1 << 2, &mask) == 7);
    CHECK(CPU_ISSET(7, &mask) && CPU_ISSET(8, &mask));
    CHECK(cache_mask(7, 3, 1 << 2, &mask) == XE_CPU_CACHE_ABSENT);
    CHECK(cache_mask(7, 3, (1 << 2) | (1 << 3), &mask) == XE_CPU_CACHE_UNKNOWN);
    CHECK(cache_mask(9, 3, 1 << 2, &mask) == XE_CPU_CACHE_UNKNOWN);
    CHECK(cache_mask(7, 2, -1, &mask) == XE_CPU_CACHE_UNKNOWN);

    /* Malformed records must not become positive cache-sharing evidence. */
    const char *fields[] = {"level", "type", "shared_cpu_list"};
    const char *good[] = {"2\n", "Unified\n", "7-8\n"};
    const char *bad[][3] = {
        {"2garbage\n", "2 3\n", "2\n3\n"},
        {"UnifiedGarbage\n", "Unified garbage\n", "Unified\nData\n"},
        {"7-8garbage\n", "7-8 9\n", "7-8\n9\n"}
    };
    for (size_t i = 0; i < 3; i++) {
        char relative[80];
        snprintf(relative, sizeof relative, "cpu7/cache/index0/%s", fields[i]);
        for (size_t j = 0; j < 3; j++) {
            write_text(relative, bad[i][j]);
            CHECK(cache_mask(7, 2, 1 << 2, &mask) == XE_CPU_CACHE_UNKNOWN);
        }
        write_bytes(relative, good[i], strlen(good[i]) + 1);
        CHECK(cache_mask(7, 2, 1 << 2, &mask) == XE_CPU_CACHE_UNKNOWN);
        write_text(relative, good[i]);
        CHECK(cache_mask(7, 2, 1 << 2, &mask) == 7);
    }
    write_text("cpu7/cache/index0/type", "Unified\n \t\n");
    CHECK(cache_mask(7, 2, 1 << 2, &mask) == 7);

    write_text("cpu10/cache/index0/level", "2\n");
    write_text("cpu10/cache/index0/type", "Unified\n");
    write_text("cpu10/cache/index0/shared_cpu_list", "11\n");
    CHECK(cache_mask(10, 2, 1 << 2, &mask) == XE_CPU_CACHE_UNKNOWN);

    write_text("cpu11/cache/index0/level", "2\n");
    write_text("cpu11/cache/index0/type", "Data\n");
    write_text("cpu11/cache/index0/shared_cpu_list", "11\n");
    CHECK(cache_mask(11, 2, 1 << 2, &mask) == XE_CPU_CACHE_UNKNOWN);

    write_text("cpu12/cache/index0/level", "2\n");
    write_text("cpu12/cache/index0/type", "Unified\n");
    write_text("cpu12/cache/index0/shared_cpu_list", "12\n");
    write_text("cpu12/cache/index1/level", "2\n");
    write_text("cpu12/cache/index1/type", "Unified\n");
    write_text("cpu12/cache/index1/shared_cpu_list", "12\n");
    CHECK(cache_mask(12, 2, 1 << 2, &mask) == XE_CPU_CACHE_UNKNOWN);

    write_text("cpu13/cache/index0/level", "3\n");
    write_text("cpu13/cache/index0/type", "Unified\n");
    write_text("cpu13/cache/index0/shared_cpu_list", "13\n");
    CHECK(cache_mask(13, 3, 1 << 2, &mask) == XE_CPU_CACHE_UNKNOWN);
}

static int remove_fixture(const char *path, const struct stat *st, int type, struct FTW *ftw) {
    (void)st; (void)type; (void)ftw;
    return remove(path);
}

int main(void) {
    strcpy(fixture_root, "/tmp/xenolith-topology-XXXXXX");
    CHECK(mkdtemp(fixture_root) != NULL);
    read_mask_cases();
    read_number_cases();
    physical_core_cases();
    cache_cases();
    CHECK(nftw(fixture_root, remove_fixture, 16, FTW_DEPTH | FTW_PHYS) == 0);
    puts("cpu-topology: sysfs mask and cache parsing PASS");
    return 0;
}
