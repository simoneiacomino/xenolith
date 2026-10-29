# CPU worker measurements

These historical measurements illustrate why worker count is configurable rather than a universal optimum. They are small, workload-specific experiments, not a cross-machine ranking. They used Gemma 4 26B-A4B IT QAT UD-Q4_K_XL GGUF (SHA-256 `a7c5bc715f5ff8e99a3e8901ce7d2b42b402c669bf24f7c5250747633d0f5891`), fixed synthetic token IDs, context capacity 16384, and 128 timed decode steps after a prompt. Decode throughput excludes model loading and prefill.

## Worker scaling: Core Ultra X7 358H

The remote system had an Intel Arc B390 GPU (`8086:b080`), 64,263,780 KiB reported memory, an Ubuntu 7.0.0-38-generic Linux kernel and GCC 15.2.0. On 2026-10-26, two experimental PAUSE builds with the same scratch fix and harness compared CPU IDs `0,2,4,6,8,10` (2 performance and 4 efficiency cores) against `0..11` (4 performance and 8 efficiency cores). These are physical-core maps on this machine, not portable CPU ID prescriptions. The model was hashed before each case, warming the filesystem cache. Three pairs per prompt were run in AB/BA/AB order; full prefill and final output vectors matched bitwise. Rates below are medians of three samples per profile; the percentage is the median of the paired ratios.

| Prompt tokens | 6 workers tok/s | 12 workers tok/s | Paired throughput change | Package J/token, 6 → 12 | Paired energy change |
|---:|---:|---:|---:|---:|---:|
| 512 | 28.219 | 34.320 | +21.73% | 0.9273 → 0.8619 | -6.91% |
| 8192 | 21.761 | 25.170 | +15.52% | 1.0258 → 0.8925 | -13.00% |

The [12 individual samples](cpu-workers-remote-20261026.csv) include measured decode time, process CPU time and package energy. This six-core control is the old fixed map, not the new automatic six-core selection on this machine. Package RAPL is gross package-domain energy, not process-only or whole-system energy; CPU time includes waits and is not energy. CPU time rose by a paired median of 64.17% and 72.70% respectively. These data do not establish desktop responsiveness, sustained behavior, or the best pool among all maps. Three pairs do not establish statistical significance.

These measurements used experimental fixed-map builds, rather than the current runtime configuration. Their source transformations and harness are retained in the private campaign archives.

## Worker screen: Core i7-13700H laptop

On 2026-10-26, isolated experimental builds compared different maps with fresh six-P-core controls. At prompt 512, one pair each gave 17.919 → 15.209 tok/s for 12 workers (4P+8E) and 17.806 → 15.863 tok/s for 14 workers (6P+8E), changes of -15.121% and -10.912%. The 6P map was `0,2,4,6,8,10`; the 12-worker map was `0,2,4,6,12..19`; the 14-worker map was `0,2,4,6,8,10,12..19`. All six short-screen samples, including a separate wait-method comparison, passed the local numeric and affinity checks. These single-pair observations do not determine the best laptop pool.

The first 8192-token baseline attempt stopped at the temperature guard before prefill or decode measurements; subsequent cases were not run. Energy was disabled. There is therefore no long-prompt or laptop energy conclusion from this screen. The earlier [laptop competitor comparison](compare_pp_tg.md) uses a different protocol and should be read separately.

## Integrated worker configuration: regression control

On the Core Ultra X7 358H on 2026-10-27, a separately built integrated worker implementation was compared with an original fixed-map build with the scratch fix applied. Both used the same map, harness, toolchain and GPU kernel artifacts. The model was hashed before each numeric case, warming the filesystem cache. Two pairs per map and prompt were run in AB/BA order. The [16 performance samples](cpu-workers-integration-20261027.csv) include both builds; full prefill and final vectors matched bitwise against fresh controls. Qualification also covered 1, 4, 6 and 12 workers with short inputs.

| Workers | Prompt tokens | Median paired decode throughput change, integrated vs control | Range |
|---:|---:|---:|---:|
| 4 | 512 | -0.155% | -0.216% to -0.095% |
| 4 | 8192 | +2.152% | +1.374% to +2.931% |
| 12 | 512 | +1.511% | +0.091% to +2.931% |
| 12 | 8192 | -0.001% | -1.927% to +1.925% |

These two-pair controls do not prove equivalence or a general speedup. Tail latency was not uniformly preserved: for 4 workers at 512, maximum step time rose from about 33 ms to 39 ms in both pairs; the cause was not identified. Energy was not measured here. Functional, GPU KV and snapshot tests passed, but ASan was not qualified in that cloud environment, including for a minimal probe. Headless browser observations do not establish real desktop responsiveness.

The measured integrated `xenolith.c` had SHA-256 `d47d2f647272400f0ff910b131b8abeb6c3e8ff2325b02b85fbda03a06d45c8e`. It predates the explicit C options API and is retained with the historical campaign archives. The CSVs are selected extracts of audited measurements; the raw archives contain private system paths and are not included. The historical campaign harnesses are unpublished, so this is not a complete reproduction package.

For current controls and configuration, see [CLI reference](../docs/cli.md) and `make check-workers`. The [comparison procedure](compare_pp_tg.md) describes a different public benchmark; it does not reproduce these historical worker campaigns.

Run `python3 -B bench/cpu-workers-summary.py` from the repository root to calculate the remote and integration tables from the CSVs. For each prompt, the script pairs rows with the same repetition number and computes `(candidate / control - 1) * 100`, then takes the median and range. The remote table also reports medians of the individual profile rates. The integration CSV rates are rounded to six decimal places.
