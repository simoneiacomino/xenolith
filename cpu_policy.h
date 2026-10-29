#ifndef XE_CPU_POLICY_H
#define XE_CPU_POLICY_H

#include <stddef.h>

/* Internal, pure selection policy. IDs are chosen by the caller from its
 * allowed mask; each entry represents one distinct physical core. */
typedef enum {
    XE_CPU_UNKNOWN, XE_CPU_HOMOGENEOUS, XE_CPU_P, XE_CPU_ATOM
} xe_cpu_class;

enum { XE_CPU_CACHE_UNKNOWN = -1, XE_CPU_CACHE_ABSENT = -2 };
typedef struct {
    int cpu, package, core;
    xe_cpu_class class_id;
    /* Canonical group ID or the negative states above. */
    int l2_group, l3_group;
} xe_cpu_core;

typedef struct {
    const char *brand;
    int intel_vendor, identity_verified, complete;
    const xe_cpu_core *cores;
    size_t core_count;
} xe_cpu_inventory;

typedef enum {
    XE_CPU_POLICY_MATCH, XE_CPU_POLICY_UNVERIFIED, XE_CPU_POLICY_UNKNOWN_SKU,
    XE_CPU_POLICY_INCOMPLETE, XE_CPU_POLICY_INVALID_INVENTORY,
    XE_CPU_POLICY_TOPOLOGY_MISMATCH, XE_CPU_POLICY_CACHE_AMBIGUOUS,
    XE_CPU_POLICY_OUTPUT_TOO_SMALL
} xe_cpu_policy_status;

typedef enum { XE_CPU_EVIDENCE_NONE, XE_CPU_EVIDENCE_SPEC,
               XE_CPU_EVIDENCE_MEASURED_PROFILE } xe_cpu_policy_evidence;

typedef struct {
    xe_cpu_policy_status status;
    xe_cpu_policy_evidence evidence;
    const char *profile; /* Static SKU label; NULL without a match. */
    size_t count;
} xe_cpu_policy_result;

/* Returns a profile or a fallback reason; never produces a partial selection.
 * Output is ordered P cores, then Atom cores, with ascending IDs per group.
 * The caller implements fallback and validates/sets actual affinity and ISA. */
xe_cpu_policy_result xe_cpu_policy_select(const xe_cpu_inventory *inventory,
                                          int *selected, size_t capacity);
const char *xe_cpu_policy_reason(xe_cpu_policy_status status);

#endif
