# Performance tuning

The CPU worker count and map are configurable at engine startup. See
[engine configuration](configuration.md) for the exact selection and affinity rules.

## Measured starting profiles

The GPU compatibility matrix describes enabled devices and reported inference
runs; it is not a CPU tuning table. Core classes and counts, cache topology,
memory, power limits and the allowed CPU set matter within a processor family.
Use these measured configurations as starting points, with the physical core
map verified on the destination machine:

| Measured system | Starting profile | Evidence |
|---|---|---|
| Core i7-13700H / DDR4 target laptop | Six P-cores | Adding E-cores slowed the short-prompt screen. |
| Core Ultra X7 358H / Arc B390 system | Twelve cores: 4P + 8E | Improved decode throughput and package energy/token against the measured six-core map. |

See the [worker measurements](../bench/cpu-workers.md) for workloads and limits.
These recipes are automatic defaults for matching SKUs and complete verified
topologies, not rules for every Raptor Lake or Panther Lake processor. Their
measured status describes the source systems above, not a new performance
measurement on the current host. Explicit settings override automatic choice.

## Specification-derived defaults

The following catalog proposes starting profiles for which this project has
no worker measurements. Intel's linked specifications establish the
hardware configuration; the worker choices are our engineering hypotheses,
not Intel recommendations or measured performance results. Entries describe
the named SKU, not every processor in that family. Specifications were checked
on 2026-10-28. The starting pool is selected automatically when the
[matching requirements](configuration.md#automatic-selection) hold;
the comparison pool is a manual tuning option.

`P`, `E` and `LP-E` denote performance, efficient and low-power efficient
physical cores; `C` denotes the homogeneous cores in older designs. Each
selected physical core gets one worker, even if it supports two SMT threads.

| Platform / example SKU and specification | Physical cores | Automatic starting pool | One comparison pool |
|---|---|---|---|
| Tiger Lake / [i7-1185G7](https://www.intel.com/content/www/us/en/products/sku/208664/intel-core-i71185g7-processor-12m-cache-up-to-4-80-ghz-with-ipu/specifications.html) | 4C | 4C = 4 workers | 2C |
| Rocket Lake / [i7-11700](https://www.intel.com/content/www/us/en/products/compare.html?productIds=212252%2C212279%2C212274) | 8C | 6C = 6 workers | 8C |
| Alder Lake mobile / [i7-1260P](https://www.intel.com/content/www/us/en/products/sku/226254/intel-core-i71260p-processor-18m-cache-up-to-4-70-ghz/specifications.html) | 4P + 8E | 4P = 4 workers | 4P + 4E |
| Alder Lake mobile / [i3-1220P](https://download.intel.com/newsroom/2022/client-computing/12th-gen-intel-mobile-product-brief.pdf) | 2P + 8E | 2P + 4E = 6 workers | 2P |
| Alder Lake desktop / [i5-12400 and i5-12600K](https://www.intel.com/content/www/us/en/products/compare.html?productIds=134589%2C134586) | 6P; 6P + 4E respectively | 6P = 6 workers | 4P |
| Alder Lake-N / [N100](https://www.intel.com/content/www/us/en/products/sku/231803/intel-processor-n100-6m-cache-up-to-3-40-ghz/specifications.htm), Twin Lake / [N150](https://www.intel.com/content/www/us/en/products/sku/241636/intel-processor-n150-6m-cache-up-to-3-60-ghz/specifications.html) | 4E | 4E = 4 workers, conditional on model memory fit | 2E |
| Raptor Lake mobile / [i7-1355U](https://www.intel.com/content/www/us/en/products/sku/232160/intel-core-i71355u-processor-12m-cache-up-to-5-00-ghz/specifications.html) | 2P + 8E | 2P + 4E = 6 workers | 2P |
| Raptor Lake desktop / [i7-13700](https://www.intel.com/content/www/us/en/products/sku/230490/intel-core-i713700-processor-30m-cache-up-to-5-20-ghz/specifications.html) | 8P + 8E | 6P = 6 workers | 8P |
| Meteor Lake H / [Ultra 7 155H](https://www.intel.com/content/www/us/en/products/sku/236847/intel-core-ultra-7-processor-155h-24m-cache-up-to-4-80-ghz/specifications.html) | 6P + 8E + 2LP-E | 6P = 6 workers | 6P + 4E |
| Meteor Lake U / [Ultra 5 125U](https://www.intel.com/content/www/us/en/products/sku/237330/intel-core-ultra-5-processor-125u-12m-cache-up-to-4-30-ghz/specifications.html) | 2P + 8E + 2LP-E | 2P + 4E = 6 workers | 2P |
| Arrow Lake H / [Ultra 7 255H](https://www.intel.com/content/www/us/en/products/sku/241751/intel-core-ultra-7-processor-255h-24m-cache-up-to-5-10-ghz/specifications.html) | 6P + 8E + 2LP-E | 6P = 6 workers | 6P + 4E |
| Arrow Lake U / [Ultra 5 235U](https://www.intel.com/content/www/us/en/products/sku/241862/intel-core-ultra-5-processor-235u-12m-cache-up-to-4-90-ghz/specifications.html) | 2P + 8E + 2LP-E | 2P + 4E = 6 workers | 2P |
| Lunar Lake / [Ultra 7 268V](https://www.intel.com/content/www/us/en/products/sku/240958/intel-core-ultra-7-processor-268v-12m-cache-up-to-5-00-ghz/specifications.html) | 4P + 4LP-E | 4P = 4 workers | 4P + 4LP-E |
| Wildcat Lake / [Core 5 315](https://www.intel.com/content/www/us/en/products/sku/246127/intel-core-5-processor-315-6m-cache-up-to-4-40-ghz/specifications.html) | 2P + 4LP-E | 2P = 2 workers | 2P + 4LP-E |

The measured i7-13700H and Ultra X7 358H configurations are listed separately
above. Other Panther Lake SKUs need their own topology match: Intel's
[Series 3 datasheet](https://cdrdv2-public.intel.com/872188/872188-001.pdf)
includes configurations with no regular E-cores. The Nova Lake GPU entries in
the compatibility matrix do not establish a CPU topology; no sufficiently
verified SKU specification was identified for a worker recommendation here.
Unlisted SKUs remain outside this catalog.

### Why these starting points

With four or more P-cores, start with a homogeneous pool of four to six P-cores
to limit mixed-core synchronization and preserve the target laptop's starting
policy. With only two P-cores and eight regular E-cores, adding four E-cores is
a lower-confidence proposal for more parallel work; compare it against the
two-P-core control. When choosing four E-cores, prefer one shared-cache group
if the actual topology exposes it. This locality preference is also a
hypothesis, not an established optimum.

Keep LP-E cores out of the initial mixed pool, but include them in the bounded
comparison for Lunar and Wildcat configurations that have no regular E-cores.
The LP-E label does not imply identical cache placement or performance across
families. On homogeneous CPUs, use the available physical cores up to six as
the initial pool, with the listed smaller or larger control.

These choices bound the first comparison rather than exhaust every possible
map. A faster result must still be checked for correctness, tail latency and
the resource costs relevant to the intended workload.

For a manual comparison, use `--cpu-list` with a map verified on the destination
machine. See [inspecting a manual CPU map](configuration.md#inspecting-a-manual-cpu-map)
and the [CLI and C API settings](configuration.md#cpu-workers); a worker count
alone does not specify core classes.

### Memory and system limits

Memory specifications are eligibility information, not achieved bandwidth.
For example, the 155H supports DDR5-5600 or LPDDR5X-7467, while the 268V lists
32 GB LPDDR5X-8533; both facts come from their linked specifications. Installed
memory, power settings and cooling can change the useful pool size. Intel
lists only 16 GB maximum memory for N100/N150; the conditional CPU recipe is
not a recommendation that the target model and context fit those systems.
GPU enablement also does not establish instruction support or a successful
model run. Automatic worker selection does not validate model memory fit or
match the installed RAM to the measured system.

## Performance and portability

More workers can slow decode, especially across different core classes. Fewer
workers can reduce CPU occupancy without reducing joules per token. Idle cores
alone do not guarantee browser or desktop responsiveness because memory,
caches, GPU and power limits remain shared. Active waits use PAUSE in the
normal build; no low-power wait mode is selected automatically.

Build for the CPU on which the binary will run. The runtime checks selected
cores for AVX2, FMA, F16C, the compiled VNNI variant and OS vector state.
`-march=native` may emit other instructions, so these checks do not make a
native binary portable across CPU generations. GPU device compatibility does
not establish CPU compatibility.
