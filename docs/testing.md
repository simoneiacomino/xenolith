# Testing

`make check-install check-devices` covers source installation and the lightweight
hardware diagnostic without accessing a GPU or model. Installation tests use
temporary user/staging prefixes and verify PATH invocation, unchanged unrelated
files and no configuration creation. Device tests run the actual CLI with a
temporary Level Zero preload fixture: tested/untested and unsupported devices,
runtime failures, denied access, device loss, config bypass and no GPU contexts
or allocations. These tests require Python 3, a C compiler and Level Zero headers.
They do not establish actual hardware compatibility. For a real runtime check,
run `./xenolith check-device` separately; it requires the installed GPU driver
but performs no model inference.

Run the commands below from the repository root.

## General checks

- `make check-installer`: shell installation, packaging, build recipe and download
  resume fixtures, without compiling or loading the engine. Requires Python 3,
  Node26+, npm, Git, binutils and standard Linux shell tools; tests use temporary
  homes, pseudo-terminals and loopback HTTP. Simulated hardware/engine responses
  do not establish inference. See [installation tests](../installer/README.md#tests)
  for opt-in checks using real packages and models.
- `make check-config`: pure configuration/argument resolution and actual CLI
  `config init`/`inspect` checks, without weights or GPU initialization. Covers
  atomic creation, concurrent initialization, existing files, custom paths,
  environment selection, malformed files, projector defaults and literal prompt
  values. Requires Python 3 for the CLI checks; the standalone
  `tests/test_cli_config` executable needs neither Python nor Level Zero.
- `make check-host`: host unit and CLI checks, including configuration,
  installation, mocked devices, synthetic cache/snapshot and vision checks,
  and ASan checks of kernel load loops. No GPU, model or VNNI execution is needed.
- `make check-vnni`: worker scheduling/affinity and CPU decode arithmetic checks.
  Requires a CPU compatible with the compiled engine ISA, including VNNI;
  no GPU or model is needed.
- `make check-unit check-context`: the complete unit suite plus CLI context and
  worker-option checks. `check-unit` combines `check-host-unit` and `check-vnni`.
  For focused CPU policy, topology, worker and CLI worker-option checks, use
  `make check-workers`.
- `make check-gpu`: numerical GPU regressions, requiring a supported GPU but no model.
- `make check-all MODEL=/path/to/model.gguf`: the standard suite, including model,
  persistence, prefill, runtime and server checks; requires a supported GPU and model.

The host suite requires Python 3, a C compiler with AddressSanitizer support,
Level Zero headers/loader and `ocloc`. See the [source build guide](build.md)
for build dependencies.

Benchmark methods and selected results are indexed in [bench/README.md](../bench/README.md).

Development utilities live in `tools/`, executable checks and fixtures in
`tests/`, and performance experiments in `bench/`. The text regression matrix
is `bench/text_regression.py`; its `--help` describes text matrix and paired ABBA
runs. It records executable identities and commands, and also supports
candidate-only measurements. Collecting measurements alone does not establish
numerical or performance equivalence. `bench/bench_prefill_gemm.c` and
its OpenCL companion are an experimental kernel lab pinned to device `a7a0`,
including candidate kernels that are not used by the engine. They are not a
portable regression suite. The end-to-end benchmark procedure remains in
[bench/compare_pp_tg.md](../bench/compare_pp_tg.md).

## KV regression checks and kernel optimization guidance

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

## Vision checks

Python 3 is used by the vision checks. Pillow and NumPy are development
dependencies for preprocessing and independent numerical references. Recorded
versions, source hashes and artifact identities are in
[the manifest](../tests/vision/manifest.json). See [vision](vision.md) for
supported inputs and persistence contracts.

[tests/vision/](../tests/vision) holds checks, fixtures and reference metadata.
`tools/` holds artifact verification, fixture generation, preprocessing and
numerical oracles. Keep raw execution reports outside the public repository and
use a fresh output directory for each run. Preserve source revisions, artifact
identities and environment details with those internal reports; filenames such
as `final` do not certify the current checkout. The Python integration checks
reject existing output directories.

### Input, GPU and preprocessing checks

Vision input checks are included in `check-unit`; vision attention/ring checks
are included in `check-gpu`. `make check-vision` runs those two focused checks
together, requiring a supported GPU but no model. Projector and projected-image
checks below are explicit opt-ins and are not part of `check-all`.

For preprocessing comparison, build `tools/vision_prepare`, obtain the pinned
Transformers reference files with `python3 tools/vision_artifacts.py --references`,
and install the manifest's Pillow version in a development environment. The
artifact utility verifies local files; add `--download` to download missing
projector/reference files into its `--directory` (default `.vision-cache`).

```sh
make tools/vision_prepare
python3 tests/vision/check_preprocess.py --output /tmp/xe-preprocess-new
```

This compares EXIF orientation, alpha compositing, resizing and all five image
budgets, with a separate JPEG pixel tolerance. `python3 tools/vision_fixtures.py`
regenerates reference fixtures; do not regenerate committed fixtures merely to
make a candidate pass.

### Model and session checks

Set `MODEL` and `MMPROJ` to the pinned GGUF files. The scripts create their own
state/cache under the supplied fresh output directory:

```sh
python3 tests/vision/check_wire.py --engine ./xenolith \
  --model "$MODEL" --mmproj "$MMPROJ" --output /tmp/xe-wire-new
python3 tests/vision/check_wire.py --engine ./xenolith \
  --model "$MODEL" --mmproj "$MMPROJ" --large --output /tmp/xe-wire-large-new
python3 tests/vision/check_chat.py --engine ./xenolith \
  --model "$MODEL" --mmproj "$MMPROJ" --output /tmp/xe-chat-new
make check-text-batch-model MODEL="$MODEL" TEST_CONTEXT=65536
```

The wire suite covers image transport, retained media, multiple images, rewind,
rebuild, ephemeral requests, checkpoints and restart. `--large` selects the
separate 131072-context/1120-row checkpoint and restart scenario. Chat uses the
included quadrant image and checks both answers, plus two text-only turns.
The text batch check compares prefill sizes 512 and 1024; repeat it with
`TEST_CONTEXT=131072` for the larger capacity.

For C session lifecycle checks, produce a matching `prompt.ids` and
`embedding.f32` using `xenolith run` with `--image`, `--mmproj` and
`--vision-dump DIR` (create `DIR` first). Then run:

```sh
make check-vision-session-model MODEL="$MODEL" TEST_CONTEXT=65536 \
  VISION_PROMPT_IDS=/path/to/dump/prompt.ids \
  VISION_EMBEDDING=/path/to/dump/embedding.f32
```

The prompt must contain the matching image span. This check exercises identity,
snapshot restore, shadow ownership, promotion, replay and rewind without
rerunning the encoder. It is distinct from encoder numerical validation.

`make check-weights-identity` takes `MODEL`, `MMPROJ`, `MMPROJ_VARIANT`
(a compatible projector with changed tensor bytes) and `MMPROJ_INVALID`
(invalid projector metadata). It needs the pinned originals and separate test
variants, but no GPU device. Never mutate the original weight files.

`python3 tests/vision/check_cache.py --help` describes the migration check. Its
`--previous-engine` must be an older build that writes
visual-v2 snapshots; an arbitrary second build is not a valid substitute.
The script checks replay into the current format, predecessor recovery and
disabled caches using disposable state.

### Independent numerical references

The C++ encoder and decoder oracles require two separate llama.cpp checkouts
at `references.llama_cpp_commit` from the manifest. Apply
`python3 tools/vision_reference_patch.py /path/to/encoder-checkout` only to the
encoder copy: its F32 GELU adapter is necessary for the pinned vision model.
Keep the decoder copy unmodified. Build shared CPU libraries in each checkout's
`build-cpu`, including `mtmd` for the encoder. Use the manifest's
`ggml_cpu_options` for reproducible reference builds. Then compile the wrappers:

```sh
make vision-reference-tools \
  LLAMA_VISION_DIR=/path/to/encoder-checkout \
  LLAMA_DECODE_DIR=/path/to/decoder-checkout
```

Override `LLAMA_VISION_BUILD` and `LLAMA_DECODE_BUILD` for other build directories.
The targets check the pinned revisions, GELU source variant and source/build
association. These checks do not replace rebuilding libraries after source
changes. Neither target downloads dependencies nor changes their sources.

`tools/vision_prepare IMAGE BUDGET prepared.rgb` produces the resized pixels
consumed by `tools/vision_reference MMPROJ prepared.rgb OUTPUT_DIR` and
`python3 tools/vision_numpy_reference.py MMPROJ prepared.rgb OUTPUT_DIR`. Compare encoder
dumps with `python3 tools/vision_compare.py REFERENCE_DIR CANDIDATE_DIR --output REPORT`;
its default checks all 30 selected stages with fixed RMS/cosine limits.
The decoder oracle accepts `MODEL IDS EMBEDDING START ROWS CONTEXT OUTPUT_PREFIX`;
use the same prompt/span and projected rows as the candidate.
