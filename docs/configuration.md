# Engine configuration

Xenolith uses the GPU for prefill and a CPU worker pool for decode. Without
overrides, it selects a [catalog profile](tuning.md) when the CPU identity and
physical topology match. Otherwise it uses up to six allowed physical cores,
one logical CPU per core in ascending CPU ID order. A profile is a starting
configuration, not an automatic throughput or energy optimum.

## CPU workers

Use CLI options to choose the pool for an engine invocation:

```sh
# Select four distinct physical cores from the allowed CPU set.
./xenolith run /path/to/model.gguf --ctx 16384 --workers 4 -p "Hello" -n 128

# Use only after verifying the physical cores behind these IDs on this machine.
./xenolith run /path/to/model.gguf --ctx 16384 --cpu-list 0,2,4,6,8,10 -p "Hello" -n 128

# Select the automatic profile, ignoring inherited CPU environment settings.
./xenolith run /path/to/model.gguf --workers auto -p "Hello" -n 128
```

`--workers` accepts a positive integer or `auto`. `--cpu-list` accepts an
ordered, comma-separated list of decimal logical CPU IDs; ranges and spaces
are not accepted. The list sets the worker count; an accompanying numeric
`--workers` must equal its length. `--workers auto` cannot be combined with a
list. These options apply to inference commands and `vision-encode`;
`info` and `tokenize` do not accept CPU options.

The engine rejects unavailable CPUs, repeated IDs, SMT siblings and unknown
physical topology in an explicit list. An explicit worker count must fit the
available distinct physical cores. One worker is supported, including serial
router and dense work.

An explicit count or list bypasses the automatic catalog. A count alone keeps
the ascending-ID rule; it does not request a specific mix of P/E cores.
For example, `--workers 6` selects the generic six-core plan even on a machine
whose automatic profile would choose twelve workers. Use `--cpu-list` to
control membership and order precisely; `--workers auto` uses automatic selection.

Without an explicit list, selection respects the opening thread's allowed CPU
set, including noncontiguous sets and cpuset restrictions. CPUs with unavailable
physical topology are skipped. If none have readable topology, the default
falls back to one allowed CPU with a notice; an explicit count fails in
that case. The engine reports the selected CPU map and verifies affinity on
each selected CPU. Affinity must be permitted by the host or container, even
for automatic selection; there is no unpinned fallback.
Invalid configuration follows the engine's fatal initialization-error path:
it prints a diagnostic and exits the process, including when called through
the C API.

### Environment fallback

With neither CPU option present, the CLI reads `XE_WORKERS` (a positive integer)
and `XE_CPU_LIST` (the same list syntax). A list alone determines the count;
when both are set, the count must equal the list length. With either CPU option
present, the CLI ignores **both** environment variables, including malformed
values. For example, `--workers auto` restores automatic selection even if an
inherited `XE_CPU_LIST` is invalid. The C library never reads these variables.

## C API

Pass engine configuration explicitly through `xe_engine_options`:

```c
xe_engine_options options = { .context = 16384, .workers = 4 };
xe_engine *engine = xe_engine_open_with_options("/path/to/model.gguf", &options);
/* Create sessions, run inference and close on the pool's owner thread. */
xe_engine_close(engine);
```

Zero-initialize options, or pass `NULL`, for defaults. `context = 0` uses
`XE_CONTEXT_DEFAULT`; `workers = 0` selects automatically when no CPU list is
supplied. For a verified map, set `cpus` to an ordered `int` array and
`cpu_count` to its length;
the list determines the count unless a nonzero `workers` value is also set,
in which case they must match. `cpus` must be NULL exactly when `cpu_count` is
zero. Opening copies the options and CPU array; callers retain ownership and
may release or change their originals after the call returns.

`xe_engine_open_cached_with_options()` also prepares snapshot fingerprints;
`xe_engine_open_vocab_with_options()` opens without loading model weights.
The older opening functions retain their signatures and select CPU workers
automatically, ignoring CPU environment variables. The
[lifetime rules](#engine-and-thread-lifetime) apply to all opening functions.

## Automatic selection

The engine reads Intel CPUID identity and core types on the allowed physical
core representatives, and checks Linux sibling and cache-sharing topology.
CPU IDs and clock speeds are not used to infer core classes. Profiles require
the exact catalog SKU and its complete expected physical-core counts on one
package. A cpuset may exclude SMT siblings if every physical core still has
one allowed representative; a cpuset excluding entire cores uses the generic
fallback. Unknown SKUs, inconsistent data or ambiguous required caches also
use that fallback, without expanding the allowed set.

P-only profiles need no E/LP-E cache distinction. Mixed profiles on the 125U,
235U and 358H require exactly eight Atom-type cores sharing the P-cores' L3;
the other Atom cores must have a known separate or absent L3. This uses the
cache organization documented in Intel's [Meteor Lake datasheet](https://edc.intel.com/content/www/us/en/design/products/platforms/details/meteor-lake-u-p/core-ultra-processor-datasheet-volume-1-of-2/002/intel-smart-cache-technology/),
[200H/200U datasheet, section 12.7](https://cdrdv2-public.intel.com/842704/842704-002.pdf)
and [Series 3 datasheet, section 11.7](https://cdrdv2-public.intel.com/872188/872188-001.pdf).
Missing cache files are not evidence of LP-E cores: the probe checks CPUID
cache presence and reciprocal kernel sharing masks. An unavailable or
contradictory required cache view prevents that profile from matching.
Selection does not run a benchmark at startup. The catalog is in
[cpu_policy.c](../cpu_policy.c); Linux/CPUID discovery is in
[cpu_topology.c](../cpu_topology.c).

Startup diagnostics report the profile and its evidence, or the fallback
reason, followed by the actual worker count and ordered CPU map. For example:

```text
xenolith: CPU profile i7-13700h (based on measured configuration)
xenolith: decode workers 6; CPU map 0,2,4,6,8,10
```

The measured-profile label refers to the source of the recipe; it does not
certify the memory, power settings or performance of the current machine.
Physical-core and instruction checks still apply to explicit overrides.

## Inspecting a manual CPU map

Inspect topology from the same environment that will launch Xenolith:

```sh
awk '/^Cpus_allowed_list:/' /proc/self/status
lscpu -e=CPU,SOCKET,CORE,ONLINE
# Replace cpu0 with each candidate CPU when checking its SMT siblings.
cat /sys/devices/system/cpu/cpu0/topology/thread_siblings_list
```

Choose online CPUs from the allowed list, one per `(SOCKET, CORE)` pair;
CPUs in the same sibling list share a physical core. For an embedded engine,
inspect `/proc/PID/task/TID/status` for the thread that opens it instead.
Neither CPU IDs nor clock speeds identify P/E core classes.

On Intel hybrid systems where the kernel exposes these PMUs, their CPU lists
identify Core-type (P) and Atom-type (E or LP-E) CPUs respectively; see
[Linux's hybrid PMU documentation](https://github.com/torvalds/linux/blob/master/tools/perf/Documentation/intel-hybrid.txt):

```sh
cat /sys/bus/event_source/devices/cpu_core/cpus
cat /sys/bus/event_source/devices/cpu_atom/cpus
```

These are system-wide lists; intersect them with the allowed CPUs above.
They do not distinguish regular E from LP-E cores. On CPUs containing both,
check the SKU's cache organization against each candidate's
`/sys/devices/system/cpu/cpuN/cache/index*/level` and `shared_cpu_list`;
missing files do not establish an LP-E core. A matched automatic profile's
printed CPU map already passes the engine's checks and can supply a smaller
comparison pool by selecting a subset. Adding other core classes requires
verifying their membership first. If class information is unavailable, keep
the automatic plan or compare subsets without labeling them P/E profiles.

## Launching through XenoPi

When XenoPi starts Xenolith, the engine CLI inherits XenoPi's environment and
uses the environment fallback above. With its binary and model already
configured, launch `xenopi` without worker overrides to use automatic selection.
To choose the worker count explicitly:

```sh
XE_WORKERS=6 xenopi
```

To select specific physical cores, use `XE_CPU_LIST` instead, with IDs verified
on that machine. For the measured i7-13700H laptop:

```sh
XE_CPU_LIST=0,2,4,6,8,10 xenopi
```

This applies only when XenoPi starts a new engine. If the socket already has a
running service, XenoPi connects to that service with its existing CPU plan.
Restart Xenolith to apply a different plan; restarting only XenoPi is not enough.
XenoPi currently has no worker-count or CPU-map field in `xenolith.json`.
The affinity rules below apply to Xenolith's engine thread, not to XenoPi's
separate UI process.

## Engine and thread lifetime

Worker count is a runtime setting. A normal engine opening resolves the count
and CPU map before GPU initialization and weight repacking; creating the first
session or enabling vision then starts the pool. The same built executable can
run with different counts without recompilation. Native compiler flags affect
instruction compatibility, not the selected worker count. Vocab-only opening
copies the requested configuration immediately but defers machine topology,
the allowed CPU set and affinity checks until vision initializes its pool.
Vision initialization binds the owner even if loading the projector subsequently
fails.

The selection remains fixed until that engine closes. A new conversation or
wire request does not select a new pool, and changing the launching shell's
environment cannot reconfigure an already running service.

The thread creating a session or vision pool becomes its owner and is pinned to
the first selected CPU until engine close restores its previous affinity. Keep
session work and closing on that thread. Use a dedicated inference thread when
embedding Xenolith in a UI. The owner must still be allowed to use every CPU in
the selected map when a later pool starts. Nested pools on an owner already
pinned to one CPU are not supported transparently.

`xe_engine_worker_count()` and `xe_engine_worker_cpu()` expose the resolved
count and ordered map; vocab-only engines report zero workers before pool
configuration. The effective worker count contributes to
the snapshot numeric fingerprint, so changing it can make existing snapshots
incompatible.

For measured profiles and performance limits, see [performance tuning](tuning.md).
