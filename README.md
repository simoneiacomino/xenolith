# What is Xenolith
Xenolith is a pure C inference engine designed and optimized for Intel Xe-LP iGPUs (11th–13th gen mobile), 32 GB RAM, Linux, and a single target model (Gemma 4 26B-A4B QAT). Additional Intel Xe integrated GPUs are enabled as listed in the [GPU compatibility matrix](#gpu-compatibility), with tested and untested devices identified separately.
This project exists because I need it. I want to extract the most I can from my laptop, and so it's tuned for exactly that:
HP Envy 17, i7-13700H (6 P-core Raptor Lake-H), DDR4-3200 dual channel (51.2 GB/).
Systems like mine are very interesting for two reasons:

- First, because they are very common setups, broadly adopted for both work and personal use.
- Second reason is that they have unified memory but with a very restrained bus.

This work is inspired by antirez's DwarfStar.

# Why should you use Xenolith
If you have the target spec, Xenolith could be interesting because it is specifically designed to take the most out of this hardware.
In fact, I noticed that general engines like llama.cpp (honestly, the only one I tried) has this problem: either you run it on the CPU or on the GPU. I mean, there are some options that enable to offload some layer to one and some to the other, but that's not the point. The thing is that it makes more sense to separate the two based on the two phases: prefill and decode.
We know that prefill is mostly compute-bound and decode is bandwidth-bound, so it makes sense to exploit the GPU parallelization during the prefill and use CPU during decode. Why don't use GPU also for decode? Because, for some reason, on this hardware CPU can exploit more bandwidth.
So Xenolith is built to do exactly that, and we can see some non-extraordinary results in the benchmarks:

| Token/s | Xenolith | llama.cpp CPU | llama.cpp Vulkan |
|---|---:|---:|---:|
| pp512 | **204,27** | 70,59 | 189,07 |
| pp2048 | **171,08** | 51,11 | 160,67 |
| tg128 | **18,39** | 14,66 | 11,96 |

These are medians of three measurements per row on the target laptop. The llama.cpp CPU column uses one profile throughout: six generation threads, twenty batch threads, CPUs 0–19, mmap, and no BLAS. The input is the same fixed synthetic token sequence for all engines; `tg128` starts from an empty context. The measured samples and instructions for checking or repeating the comparison are in [bench/compare_pp_tg.md](bench/compare_pp_tg.md). Results from a new run can vary with system conditions.

So you can see that, while with llama.cpp you have to choose which one to prioritize between prefill and decode, with Xenolith you have a more balanced setup.

Again, if you look at the decode and prefill improvements separately, you can see that there is no incredible jump, but if you look combined, it can be very useful for a better local experience on this limited hardware.

I want to specify that I didn't write the kernels myself, but I used a mix of Fable and Sol to autoresearch and autoimprove them.

# GPU compatibility

Xenolith targets Intel integrated GPUs based on the Xe architecture, on Linux using Level Zero (`intel-compute-runtime`). Compatibility is tracked by GPU device ID. Discrete GPUs are not enabled.

- **Tested:** a successful Xenolith run has been reported on a system with this device ID. This does not imply validation on every system or driver version.
- **Untested:** enabled based on expected compatibility, but a successful Xenolith run has not yet been reported. The engine prints a notice and continues initialization.

All IDs below are hexadecimal and use Intel vendor ID `8086`. IDs not listed are not enabled. A compatible driver and sufficient system memory are required. All enabled devices currently use the existing kernels without XMX acceleration.

| GPU platform | GPU architecture | Tested device IDs | Untested device IDs |
|---|---|---|---|
| **Tiger Lake** | Xe-LP | — | `9a40`, `9a49`, `9a59`, `9a60`, `9a68`, `9a70`, `9a78` |
| **Rocket Lake** | Xe-LP | — | `4c80`, `4c8a`, `4c8b`, `4c8c`, `4c90`, `4c9a` |
| **Alder Lake mobile** | Xe-LP | — | `4626`, `4628`, `462a`, `46a0`, `46a1`, `46a3`, `46a6`, `46a8`, `46aa`, `46b0`, `46b1`, `46b3`, `46c0`, `46c1`, `46c3` |
| **Alder Lake desktop** | Xe-LP | — | `4680`, `4682`, `4688`, `468a`, `468b`, `4690`, `4692`, `4693` |
| **Alder Lake-N / Twin Lake** | Xe-LP | — | `46d0`, `46d1`, `46d2`, `46d3`, `46d4` |
| **Raptor Lake mobile / refresh** | Xe-LP | `a7a0` | `a720`, `a721`, `a7a1`, `a7a8`, `a7a9`, `a7aa`, `a7ab`, `a7ac`, `a7ad` |
| **Raptor Lake desktop / refresh** | Xe-LP | — | `a780`, `a781`, `a782`, `a783`, `a788`, `a789`, `a78a`, `a78b` |
| **Meteor Lake** | Xe-LPG | — | `7d40`, `7d45`, `7d55`, `7dd5` |
| **Arrow Lake** | Xe-LPG / Xe-LPG+ | `7d51` | `7d41`, `7d67`, `7dd1` |
| **Lunar Lake** | Xe2 | `64a0` | `6420`, `64b0` |
| **Panther Lake** | Xe3 | — | `b080`, `b081`, `b082`, `b083`, `b084`, `b085`, `b086`, `b087`, `b08f`, `b090`, `b0a0`, `b0b0` |
| **Wildcat Lake** | Xe3 | — | `fd80`, `fd81` |
| **Nova Lake (Xe3 GPU)** | Xe3 | — | `d740`, `d741`, `d742`, `d743`, `d744`, `d745` |
| **Nova Lake (Xe3P GPU)** | Xe3P | — | `d74a`, `d74b`, `d750`, `d751`, `d752`, `d753`, `d754`, `d755`, `d756`, `d757`, `d75f` |

Device IDs are based on [Intel Compute Runtime](https://github.com/intel/compute-runtime/blob/master/shared/source/dll/devices/devices_base.inl), with platform names cross-checked against Intel's [i915](https://dgpu-docs.intel.com/overview/supported-hardware/i915-driver-gpus.html) and [Xe](https://dgpu-docs.intel.com/overview/supported-hardware/xe-driver-gpus.html) hardware tables and the [Linux PCI ID definitions](https://github.com/torvalds/linux/blob/master/include/drm/intel/pciids.h). Tested status refers to Xenolith runs, not Intel's driver support status.

Tested configurations:

| GPU device ID | Processor / GPU | Test source |
|---|---|---|
| `8086:a7a0` | Core i7-13700H / Iris Xe, 32 GB RAM | Maintainer's HP Envy 17; [benchmark results](bench/compare_pp_tg.md) |
| `8086:7d51` | Core Ultra 7 255H / Arc 140T | [aziis98's report in PR #2](https://github.com/simoneiacomino/xenolith/pull/2) |
| `8086:64a0` | Core Ultra 7 268V / Arc 140V, 32 GB RAM | [raffaeleedidonna's report in PR #6](https://github.com/simoneiacomino/xenolith/pull/6); Windows 11, Ubuntu 24.04 under WSL2 |

If inference works on an untested device, please [open an issue](https://github.com/simoneiacomino/xenolith/issues) or submit a PR with your results so its ID can be marked tested in both the code and this table. If it fails, please open an issue describing the problem. Include the GPU device ID, processor, RAM, OS (including WSL if applicable), driver/runtime versions, Xenolith commit, model, command and relevant output. Device detection alone is not an inference test.

# Design choices
First of all, I decided to not implement an internal agent/harness because that's another whole piece of software that needs attention and care. Also, harnesses are the new IDEs, so very personal, and everyone should use the one he prefers.

So now we need a way to connect your harness to Xenolith. The standard way is a stateless chat/completions HTTP server that was designed to handle massive multi-user requests to a central server. This is a totally different use case than the one we are handling: single-machine, single-user inference.
I thought that, in this context, it would be better to create a [stateful protocol](PROTOCOL.md) so that the harness could talk to the engine in a session-aware way like an integrated agent would.
So now we don't have to send all the history, but we can just "append" the user message. Also, we can make the harness aware of such thing that are valuable in local inference, like the progress of a prefill or the cost of a rewind.

But this currently means that you have to fork your harness and make it talk the Xenolith protocol or write an adapter like I did.
In fact, I also made [Xenopi](https://github.com/simoneiacomino/xenopi), a pi distribution that ships an adapter with pi that translates its requests to Xenolith protocol. This is just a simple example I did to test the engine and the protocol.

# Limits
Current observed limits are right now in the prefill kernel that doesn't scale as well as the Vulkan one. In fact, we can start see a regression in pp8192 where Vulkan becomes faster. But nothing that isn't improvable by spending some tokens on it.

# Build and run

On Arch install the build tools, Level Zero, and Intel's GPU runtime:

```sh
sudo pacman -S --needed gcc make binutils coreutils level-zero-headers level-zero-loader intel-compute-runtime
```

Get Unsloth's [Gemma 4 26B-A4B IT QAT UD-Q4_K_XL GGUF](https://huggingface.co/unsloth/gemma-4-26B-A4B-it-qat-GGUF/blob/main/gemma-4-26B-A4B-it-qat-UD-Q4_K_XL.gguf) then build and run:

```sh
make
./xenolith run /path/to/gemma-4-26B-A4B-it-qat-UD-Q4_K_XL.gguf -p "Hello" -n 64
```

Use `--ctx N` to set the context capacity without rebuilding (default: 262144).
Any integer from 64 through 262144 is supported, including non-powers of two.
The capacity includes both prompt and generated tokens:

```sh
./xenolith run /path/to/model.gguf --ctx 50000 -p "Hello" -n 64
```

The capacity is fixed when the engine opens and shared by its sessions.
C callers can use `xe_engine_open_with_context(path, capacity)`;
`xe_engine_open(path)` retains the default. Snapshots require matching context
capacities. Global KV caches use linear positions, while sliding-window layers
retain their fixed 1024-token circular caches.

To check KV indexing with a non-power-of-two capacity on a supported GPU:

```sh
make check-kv tests/test_prefill_session
./tests/test_prefill_session /path/to/model.gguf kv-indexing --ctx 4097
```

`make check-kv-edges` runs only the synthetic GPU edge matrix, without
loading a model or allocating a full model session. It tests all 64 pairs of
capacity 64–71 and COW split 8–15, plus 24 split/tail combinations around
511/512/513 tail rows. Each case checks KV commits and ordinary/COW attention
against a CPU reference. `check-kv` and `check-gpu` also include this matrix.
The tail sizes here are synthetic kernel inputs; session tail growth and
promotion need separate integration tests.

When optimizing global attention kernels, preserve the bounds of the K loads
as well as the causal mask. Unused rows must not be read outside the buffers.
Run the edge matrix and `check-kv` for each candidate. The latter checks
session capacities 64, 65, 129, 4097, 50000, 65536 and 262144 in one build,
covering session indexing and circular wraparound. Retain the CPU
reference and the edge inputs when comparing optimization candidates.

The GPU matrix is a numerical regression test, not a memory sanitizer: discarded
out-of-range reads can leave logits and write canaries unchanged. Host ASan
does not instrument GPU execution.

`make check-kv-load-bounds` adds a separate host check (Python 3 and a C compiler
with AddressSanitizer required), also included in `check-unit`. It extracts the
K-load blocks from the current OpenCL source, translates the address-space and
zero-vector syntax to C, and executes the actual load loops on exact-size host
buffers. Volatile destinations keep even unused loads observable. Its 152 cases
cover the GPU matrix plus short tails of 1–8 rows. It also requires ASan to
detect out-of-bounds reads in three temporary mutations: removed ordinary
tile bound, removed COW tile bound, and removed COW fallback row bound.
Production sources are never mutated by the check.

This checks the extracted load footprint; it does not emulate GPU scheduling,
barriers, subgroup operations or compiler transformations. Unsupported refactors
(including K-pointer uses outside the extracted block) fail and require review
of the harness. Continue to run the numerical GPU matrix on the target device.

For automated optimization, keep the reference tests and acceptance criteria
outside the candidate's editable scope. A trusted copy of this script accepts
`--kernel /path/to/candidate/xenolith.cl` and prints the tested source hash.
Require these host checks and the target GPU checks before accepting a faster
candidate. Changes to the tests or their tolerances need a separate review;
test failure must not be converted into an accepted benchmark-only result.

For a targeted model integration check of actual COW tail growth:

```sh
make tests/test_kv_model
./tests/test_kv_model /path/to/model.gguf cow-growth 512 --ctx 4097
./tests/test_kv_model /path/to/model.gguf cow-growth 513 --ctx 4097
```

These cases grow a live shadow through tail sizes 511/512/513 and
1023/1024/1025, verify exact preservation of existing global KV rows across
reallocations, compare logits to an ordinary session using the same GPU chunks,
promote the shadow and compare eight subsequent CPU decode steps. They also
exercise a source that diverged after opening the shadow and SWA wraparound.
Logits must have relative RMS error below `1e-5`, matching the existing COW
test, and the same top token. No nonfinite values are accepted.

`./tests/test_kv_model /path/to/model.gguf decode-dump /existing/directory --ctx 4097`
writes full float32 logits after prefill and each of eight fixed-token decode
steps. Depths are 512 and 2048, plus capacity-8 for capacities up to 4104. Select a
capacity of at least 2056. Dumps can be compared across binaries with identical inputs and
capacity; successful execution alone does not establish cross-build equality.
These targeted checks do not run the full snapshot suite or measure throughput.
