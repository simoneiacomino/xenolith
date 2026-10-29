#ifndef XE_CPU_TOPOLOGY_H
#define XE_CPU_TOPOLOGY_H

#include "cpu_policy.h"

/* Read a logical CPU's physical identity. Returns one on success; missing or
 * malformed sysfs data returns zero without changing the outputs. */
int xe_cpu_topology_core(int cpu, int *package, int *core);

/* Probe one allowed representative per physical core supplied by the engine.
 * The caller supplies CPU/package/core IDs; this fills class/cache groups and
 * CPUID identity in inv (brand_storage must have 49 bytes).
 * Returns zero after restoring the calling thread's affinity, or a pthread
 * error if restoring it failed. Missing/ambiguous data disables profile use.
 * No pool, model, GPU or public configuration is owned by this module. */
int xe_cpu_topology_probe(xe_cpu_inventory *inv, xe_cpu_core *cores,
                          size_t count, char brand_storage[49]);

#endif
