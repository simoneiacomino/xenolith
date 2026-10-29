/* Pure CPU profile selection tests; no GPU, model, sysfs or affinity required. */
#include "../cpu_policy.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define CHECK(x) do { if (!(x)) { \
    fprintf(stderr, "cpu-policy:%d: %s\n", __LINE__, #x); exit(2); \
} } while (0)

static xe_cpu_core core(int cpu, int physical, xe_cpu_class kind, int l2, int l3) {
    return (xe_cpu_core){cpu, 0, physical, kind, l2, l3};
}

static xe_cpu_inventory inventory(const char *brand, const xe_cpu_core *cores, size_t count) {
    return (xe_cpu_inventory){brand, 1, 1, 1, cores, count};
}

static void expect_match(const xe_cpu_inventory *inv, const int *want, size_t count,
                         const char *profile, xe_cpu_policy_evidence evidence) {
    int selected[20];
    for (size_t i = 0; i < 20; i++) selected[i] = -77;
    xe_cpu_policy_result r = xe_cpu_policy_select(inv, selected, 20);
    CHECK(r.status == XE_CPU_POLICY_MATCH);
    CHECK(r.evidence == evidence);
    CHECK(r.profile && strcmp(r.profile, profile) == 0);
    CHECK(r.count == count);
    for (size_t i = 0; i < count; i++) CHECK(selected[i] == want[i]);
    for (size_t i = count; i < 20; i++) CHECK(selected[i] == -77);
}

static void expect_fallback(const xe_cpu_inventory *inv, xe_cpu_policy_status status,
                            size_t capacity) {
    int selected[20];
    for (size_t i = 0; i < 20; i++) selected[i] = -77;
    xe_cpu_policy_result r = xe_cpu_policy_select(inv, selected, capacity);
    CHECK(r.status == status);
    CHECK(r.evidence == XE_CPU_EVIDENCE_NONE);
    CHECK(r.profile == NULL && r.count == 0);
    for (size_t i = 0; i < 20; i++) CHECK(selected[i] == -77);
    CHECK(xe_cpu_policy_reason(status) != NULL);
}

static void laptop_profile(void) {
    xe_cpu_core cores[14];
    for (int i = 0; i < 6; i++) cores[5 - i] = core(2 + 2*i, i, XE_CPU_P, i, 1);
    for (int i = 0; i < 8; i++) cores[6 + i] = core(20 + i, 6 + i, XE_CPU_ATOM, 8, 1);
    xe_cpu_inventory inv = inventory("13th Gen Intel(R) Core(TM) i7-13700H", cores, 14);
    const int want[] = {2, 4, 6, 8, 10, 12};
    expect_match(&inv, want, 6, "i7-13700h", XE_CPU_EVIDENCE_MEASURED_PROFILE);

    inv.brand = "Intel(R) Core(TM) i7-13700HX";
    expect_fallback(&inv, XE_CPU_POLICY_UNKNOWN_SKU, 20);
    inv.brand = "Intel(R) Core(TM) i7-13700H Extra";
    expect_fallback(&inv, XE_CPU_POLICY_UNKNOWN_SKU, 20);
    inv.brand = "13th Gen Intel(R) Core(TM) i7-13700H";
    inv.intel_vendor = 0;
    expect_fallback(&inv, XE_CPU_POLICY_UNVERIFIED, 20);
    inv.intel_vendor = 1; inv.identity_verified = 0;
    expect_fallback(&inv, XE_CPU_POLICY_UNVERIFIED, 20);
    inv.identity_verified = 1; inv.complete = 0;
    expect_fallback(&inv, XE_CPU_POLICY_INCOMPLETE, 20);
    inv.complete = 1; inv.core_count = 13;
    expect_fallback(&inv, XE_CPU_POLICY_TOPOLOGY_MISMATCH, 20);
    inv.core_count = 14;
    cores[13].cpu = cores[12].cpu;
    expect_fallback(&inv, XE_CPU_POLICY_INVALID_INVENTORY, 20);
    cores[13].cpu = 27; cores[13].core = cores[12].core;
    expect_fallback(&inv, XE_CPU_POLICY_INVALID_INVENTORY, 20);
    cores[13].core = 13; cores[13].class_id = XE_CPU_UNKNOWN;
    expect_fallback(&inv, XE_CPU_POLICY_INVALID_INVENTORY, 20);
}

static void panther_profile(void) {
    xe_cpu_core cores[16];
    for (int i = 0; i < 4; i++) cores[i] = core(2 + 2*i, i, XE_CPU_P, i, 9);
    for (int i = 0; i < 8; i++) cores[4 + i] = core(10 + i, 4 + i, XE_CPU_ATOM,
                                                   i < 4 ? 3 : 4, 9);
    for (int i = 0; i < 4; i++) cores[12 + i] = core(18 + i, 12 + i, XE_CPU_ATOM,
                                                    5, XE_CPU_CACHE_ABSENT);
    xe_cpu_inventory inv = inventory("Intel(R) Core(TM) Ultra X7 358H", cores, 16);
    const int want[] = {2, 4, 6, 8, 10, 11, 12, 13, 14, 15, 16, 17};
    expect_match(&inv, want, 12, "358h", XE_CPU_EVIDENCE_MEASURED_PROFILE);
    expect_fallback(&inv, XE_CPU_POLICY_OUTPUT_TOO_SMALL, 11);

    inv.complete = 0;
    expect_fallback(&inv, XE_CPU_POLICY_INCOMPLETE, 20);
    inv.complete = 1; inv.core_count = 15;
    expect_fallback(&inv, XE_CPU_POLICY_TOPOLOGY_MISMATCH, 20);
    inv.core_count = 16; cores[15].l3_group = XE_CPU_CACHE_UNKNOWN;
    expect_fallback(&inv, XE_CPU_POLICY_CACHE_AMBIGUOUS, 20);
    cores[15].l3_group = XE_CPU_CACHE_ABSENT;
    cores[11].l3_group = XE_CPU_CACHE_ABSENT;
    expect_fallback(&inv, XE_CPU_POLICY_CACHE_AMBIGUOUS, 20);
    cores[11].l3_group = 9;
    inv.brand = "Intel(R) Core(TM) Ultra X7 358HX";
    expect_fallback(&inv, XE_CPU_POLICY_UNKNOWN_SKU, 20);
}

static void specification_profiles(void) {
    xe_cpu_core alder[6];
    for (int i = 0; i < 6; i++) alder[5 - i] = core(2*i, i, XE_CPU_HOMOGENEOUS, i, 4);
    xe_cpu_inventory inv = inventory("12th Gen Intel(R) Core(TM) i5-12400", alder, 6);
    const int want_alder[] = {0, 2, 4, 6, 8, 10};
    expect_match(&inv, want_alder, 6, "i5-12400", XE_CPU_EVIDENCE_SPEC);

    xe_cpu_core mobile[10];
    mobile[0] = core(20, 0, XE_CPU_P, 1, 5);
    mobile[1] = core(21, 1, XE_CPU_P, 2, 5);
    for (int i = 0; i < 4; i++) mobile[2 + i] = core(2 + i, 2 + i, XE_CPU_ATOM, 9, 5);
    for (int i = 0; i < 4; i++) mobile[6 + i] = core(10 + i, 6 + i, XE_CPU_ATOM, 3, 5);
    inv = inventory("Intel(R) Core(TM) i3-1220P", mobile, 10);
    const int want_mobile[] = {20, 21, 10, 11, 12, 13};
    expect_match(&inv, want_mobile, 6, "i3-1220p", XE_CPU_EVIDENCE_SPEC);

    xe_cpu_core atom[4];
    for (int i = 0; i < 4; i++) atom[i] = core(i + 4, i, XE_CPU_HOMOGENEOUS, 1, 2);
    inv = inventory("Intel(R) Processor N100", atom, 4);
    const int want_atom[] = {4, 5, 6, 7};
    expect_match(&inv, want_atom, 4, "n100", XE_CPU_EVIDENCE_SPEC);
    inv.brand = "Intel(R) Processor N1000";
    expect_fallback(&inv, XE_CPU_POLICY_UNKNOWN_SKU, 20);
}

static void cache_and_homogeneous_profiles(void) {
    xe_cpu_core lunar[8];
    for (int i = 0; i < 4; i++) lunar[i] = core(i, i, XE_CPU_P, i, XE_CPU_CACHE_UNKNOWN);
    for (int i = 4; i < 8; i++) lunar[i] = core(i, i, XE_CPU_ATOM, 4, XE_CPU_CACHE_UNKNOWN);
    xe_cpu_inventory inv = inventory("Intel(R) Core(TM) Ultra 7 268V", lunar, 8);
    const int want_lunar[] = {0, 1, 2, 3};
    expect_match(&inv, want_lunar, 4, "268v", XE_CPU_EVIDENCE_SPEC);

    xe_cpu_core meteor[12];
    for (int i = 0; i < 2; i++) meteor[i] = core(i, i, XE_CPU_P, i, 8);
    for (int i = 2; i < 10; i++) meteor[i] = core(i, i, XE_CPU_ATOM, i < 6 ? 2 : 6, 8);
    for (int i = 10; i < 12; i++) meteor[i] = core(i, i, XE_CPU_ATOM, 10, XE_CPU_CACHE_ABSENT);
    inv = inventory("Intel(R) Core(TM) Ultra 5 125U", meteor, 12);
    const int want_meteor[] = {0, 1, 2, 3, 4, 5};
    expect_match(&inv, want_meteor, 6, "125u", XE_CPU_EVIDENCE_SPEC);
    meteor[10].l3_group = meteor[11].l3_group = 8;
    expect_fallback(&inv, XE_CPU_POLICY_CACHE_AMBIGUOUS, 20);

}

int main(void) {
    laptop_profile();
    panther_profile();
    specification_profiles();
    cache_and_homogeneous_profiles();
    puts("cpu-policy: synthetic selection and conservative fallback PASS");
    return 0;
}
