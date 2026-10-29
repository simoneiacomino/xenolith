#define _GNU_SOURCE
#include "cpu_topology.h"

#include <cpuid.h>
#include <ctype.h>
#include <dirent.h>
#include <errno.h>
#include <limits.h>
#include <pthread.h>
#include <sched.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/* Read the whole bounded record, not just its first token or line. Only
 * trailing whitespace may follow the value; embedded NULs are invalid. */
static int read_record(const char *path, char *buf, size_t capacity) {
    FILE *f = fopen(path, "r");
    if (!f) return 0;
    size_t n = fread(buf, 1, capacity - 1, f);
    int extra = fgetc(f);
    int ok = n > 0 && extra == EOF && !ferror(f) && !memchr(buf, 0, n);
    fclose(f);
    if (!ok) return 0;
    while (n && isspace((unsigned char)buf[n - 1])) n--;
    buf[n] = 0;
    return n > 0;
}

/* sysfs CPU lists can contain ranges and SMT siblings, unlike XE_CPU_LIST. */
static int read_mask(const char *path, cpu_set_t *mask) {
    char buf[CPU_SETSIZE * 12];
    CPU_ZERO(mask);
    if (!read_record(path, buf, sizeof buf)) return 0;
    const char *p = buf;
    while (*p) {
        if (!isdigit((unsigned char)*p)) return 0;
        char *end;
        errno = 0;
        long first = strtol(p, &end, 10), last = first;
        if (errno || first < 0 || first >= CPU_SETSIZE) return 0;
        p = end;
        if (*p == '-') {
            p++;
            if (!isdigit((unsigned char)*p)) return 0;
            errno = 0; last = strtol(p, &end, 10);
            if (errno || last < first || last >= CPU_SETSIZE) return 0;
            p = end;
        }
        for (long cpu = first; cpu <= last; cpu++) {
            if (CPU_ISSET(cpu, mask)) return 0;
            CPU_SET(cpu, mask);
        }
        if (*p == ',') { p++; if (!*p) return 0; continue; }
        while (isspace((unsigned char)*p)) p++;
        return !*p && CPU_COUNT(mask) > 0;
    }
    return 0;
}

static int read_number(const char *path, int *out) {
    char buf[64], *end;
    if (!read_record(path, buf, sizeof buf)) return 0;
    errno = 0;
    long value = strtol(buf, &end, 10);
    if (errno || end == buf || *end || value < 0 || value > INT_MAX) return 0;
    *out = (int)value;
    return 1;
}

int xe_cpu_topology_core(int cpu, int *package, int *core) {
    if (cpu < 0 || cpu >= CPU_SETSIZE) return 0;
    char path[128];
    int package_id, core_id;
    snprintf(path, sizeof path, "/sys/devices/system/cpu/cpu%d/topology/physical_package_id", cpu);
    if (!read_number(path, &package_id)) return 0;
    snprintf(path, sizeof path, "/sys/devices/system/cpu/cpu%d/topology/core_id", cpu);
    if (!read_number(path, &core_id)) return 0;
    *package = package_id;
    *core = core_id;
    return 1;
}

/* CPUID confirms which cache levels exist. A missing sysfs file is never
 * interpreted as proof that a core has no LLC. */
static int cache_levels(unsigned max_leaf) {
    if (max_leaf < 4) return -1;
    int levels = 0;
    for (unsigned sub = 0; sub < 32; sub++) {
        unsigned a, b, c, d;
        __cpuid_count(4, sub, a, b, c, d);
        if (!(a & 31)) return (levels & (1 << 2)) ? levels : -1;
        unsigned level = (a >> 5) & 7;
        if (level < 4) levels |= 1 << level;
    }
    return -1;
}

static int cache_mask(int cpu, int level, int levels, cpu_set_t *mask) {
    CPU_ZERO(mask);
    if (levels < 0) return XE_CPU_CACHE_UNKNOWN;
    char path[192];
    snprintf(path, sizeof path, "/sys/devices/system/cpu/cpu%d/cache", cpu);
    DIR *dir = opendir(path);
    if (!dir) return XE_CPU_CACHE_UNKNOWN;
    struct dirent *entry;
    int found = 0, valid = 1;
    while ((entry = readdir(dir))) {
        if (strncmp(entry->d_name, "index", 5)) continue;
        char *end;
        long index = strtol(entry->d_name + 5, &end, 10);
        if (end == entry->d_name + 5 || *end || index < 0 || index > 255) { valid = 0; break; }
        int actual_level;
        snprintf(path, sizeof path, "/sys/devices/system/cpu/cpu%d/cache/index%ld/level", cpu, index);
        if (!read_number(path, &actual_level)) { valid = 0; break; }
        if (actual_level != level) continue;
        char type[32];
        snprintf(path, sizeof path, "/sys/devices/system/cpu/cpu%d/cache/index%ld/type", cpu, index);
        if (!read_record(path, type, sizeof type) || strcmp(type, "Unified")) { valid = 0; break; }
        snprintf(path, sizeof path, "/sys/devices/system/cpu/cpu%d/cache/index%ld/shared_cpu_list", cpu, index);
        if (found++ || !read_mask(path, mask) || !CPU_ISSET(cpu, mask)) { valid = 0; break; }
    }
    closedir(dir);
    if (!valid || (!!found != !!(levels & (1 << level)))) return XE_CPU_CACHE_UNKNOWN;
    if (!found) return XE_CPU_CACHE_ABSENT;
    for (int id = 0; id < CPU_SETSIZE; id++) if (CPU_ISSET(id, mask)) return id;
    return XE_CPU_CACHE_UNKNOWN;
}

typedef struct { cpu_set_t siblings, l2, l3; } core_masks;

int xe_cpu_topology_probe(xe_cpu_inventory *inv, xe_cpu_core *cores,
                          size_t count, char brand[49]) {
    *inv = (xe_cpu_inventory){.brand = brand, .cores = cores, .core_count = count};
    brand[0] = 0;
    for (size_t i = 0; i < count; i++) {
        cores[i].class_id = XE_CPU_UNKNOWN;
        cores[i].l2_group = cores[i].l3_group = XE_CPU_CACHE_UNKNOWN;
    }
    if (!count || count > CPU_SETSIZE) return 0;
    cpu_set_t original, online_before, online_after;
    if (pthread_getaffinity_np(pthread_self(), sizeof original, &original)) return 0;
    if (!read_mask("/sys/devices/system/cpu/online", &online_before)) return 0;
    core_masks *masks = calloc(count, sizeof *masks);
    if (!masks) return 0;
    int valid = 1;
    unsigned signature = 0;
    for (size_t i = 0; i < count && valid; i++) {
        int cpu = cores[i].cpu;
        if (cpu < 0 || cpu >= CPU_SETSIZE || !CPU_ISSET(cpu, &original) || !CPU_ISSET(cpu, &online_before)) { valid = 0; break; }
        cpu_set_t pin, actual;
        CPU_ZERO(&pin); CPU_SET(cpu, &pin);
        if (pthread_setaffinity_np(pthread_self(), sizeof pin, &pin) ||
            pthread_getaffinity_np(pthread_self(), sizeof actual, &actual) || !CPU_EQUAL(&pin, &actual)) { valid = 0; break; }
        unsigned a, b, c, d;
        unsigned max = __get_cpuid_max(0, NULL);
        __cpuid(0, a, b, c, d);
        if (b != 0x756e6547 || d != 0x49656e69 || c != 0x6c65746e || max < 1 ||
            __get_cpuid_max(0x80000000, NULL) < 0x80000004) { valid = 0; break; }
        __cpuid(1, a, b, c, d);
        unsigned current_signature = a;
        char current_brand[49];
        for (unsigned leaf = 0; leaf < 3; leaf++) {
            unsigned words[4];
            __cpuid(0x80000002 + leaf, words[0], words[1], words[2], words[3]);
            memcpy(current_brand + leaf * 16, words, 16);
        }
        current_brand[48] = 0;
        if (!i) { signature = current_signature; memcpy(brand, current_brand, 49); }
        else if (signature != current_signature || strcmp(brand, current_brand)) { valid = 0; break; }
        int hybrid = 0;
        if (max >= 7) { __cpuid_count(7, 0, a, b, c, d); hybrid = (d >> 15) & 1; }
        unsigned type = 0;
        if (max >= 0x1a) { __cpuid_count(0x1a, 0, a, b, c, d); type = a >> 24; }
        cores[i].class_id = type == 0x40 ? XE_CPU_P : type == 0x20 ? XE_CPU_ATOM :
                           !type && !hybrid ? XE_CPU_HOMOGENEOUS : XE_CPU_UNKNOWN;
        int levels = cache_levels(max);
        cores[i].l2_group = cache_mask(cpu, 2, levels, &masks[i].l2);
        cores[i].l3_group = cache_mask(cpu, 3, levels, &masks[i].l3);
        char path[192];
        snprintf(path, sizeof path, "/sys/devices/system/cpu/cpu%d/topology/thread_siblings_list", cpu);
        if (!read_mask(path, &masks[i].siblings) || !CPU_ISSET(cpu, &masks[i].siblings)) { valid = 0; break; }
        /* Validate all online sibling identities, even when only one SMT
         * thread is allowed; never execute on a CPU outside original. */
        for (int sib = 0; sib < CPU_SETSIZE && valid; sib++) {
            if (!CPU_ISSET(sib, &masks[i].siblings) || !CPU_ISSET(sib, &online_before)) continue;
            cpu_set_t reciprocal;
            int package, core;
            snprintf(path, sizeof path, "/sys/devices/system/cpu/cpu%d/topology/thread_siblings_list", sib);
            if (!read_mask(path, &reciprocal) || !CPU_EQUAL(&reciprocal, &masks[i].siblings)) { valid = 0; break; }
            if (!xe_cpu_topology_core(sib, &package, &core) ||
                package != cores[i].package || core != cores[i].core) valid = 0;
        }
    }
    int restore = pthread_setaffinity_np(pthread_self(), sizeof original, &original);
    cpu_set_t actual;
    if (!restore) restore = pthread_getaffinity_np(pthread_self(), sizeof actual, &actual);
    if (!restore && !CPU_EQUAL(&actual, &original)) restore = EINVAL;
    if (valid && (!read_mask("/sys/devices/system/cpu/online", &online_after) || !CPU_EQUAL(&online_before, &online_after))) valid = 0;
    /* A restricted cpuset may omit SMT siblings, but must represent every
     * online physical core before an exact whole-SKU recipe can apply. */
    int complete = valid;
    for (int cpu = 0; cpu < CPU_SETSIZE && complete; cpu++) {
        if (!CPU_ISSET(cpu, &online_before)) continue;
        int found = 0;
        for (size_t i = 0; i < count; i++) found += CPU_ISSET(cpu, &masks[i].siblings);
        if (found != 1) complete = 0;
    }
    /* Cache membership must agree on every probed representative. */
    for (int level = 2; level <= 3; level++) {
        int consistent = valid;
        for (size_t i = 0; i < count && consistent; i++) {
            int gi = level == 2 ? cores[i].l2_group : cores[i].l3_group;
            const cpu_set_t *mi = level == 2 ? &masks[i].l2 : &masks[i].l3;
            if (gi < 0) continue;
            for (size_t j = 0; j < count; j++) {
                int gj = level == 2 ? cores[j].l2_group : cores[j].l3_group;
                const cpu_set_t *mj = level == 2 ? &masks[j].l2 : &masks[j].l3;
                /* Sharing must include all online SMT siblings together,
                 * not only whichever representative happens to be allowed. */
                for (int sib = 0; sib < CPU_SETSIZE; sib++) {
                    if (CPU_ISSET(sib, &online_before) && CPU_ISSET(sib, &masks[j].siblings) &&
                        !!CPU_ISSET(sib, mi) != !!CPU_ISSET(cores[j].cpu, mi)) consistent = 0;
                }
                if (!consistent) break;
                if (CPU_ISSET(cores[j].cpu, mi) && (gj != gi || !CPU_EQUAL(mi, mj))) { consistent = 0; break; }
            }
        }
        if (!consistent) for (size_t i = 0; i < count; i++) {
            if (level == 2) cores[i].l2_group = XE_CPU_CACHE_UNKNOWN;
            else cores[i].l3_group = XE_CPU_CACHE_UNKNOWN;
        }
    }
    inv->intel_vendor = inv->identity_verified = valid;
    inv->complete = complete;
    free(masks);
    return restore;
}
