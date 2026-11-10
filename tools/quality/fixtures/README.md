# Shared numerical fixtures

`gemma4/` contains exact inputs, shared Google FP64 logits and a pinned
llama.cpp Vulkan baseline for each of Panther Lake (`panther-vulkan`) and
Raptor Lake (`raptor-vulkan`). The format supports separate named baselines for
additional hosts.
The arrays are ordinary Git files;
there is no separate download service. Model weights are not included.
Source text attribution and licenses are listed in [ATTRIBUTION.md](ATTRIBUTION.md).

The directory name stays stable. `manifest.json` carries a schema version and a
content hash covering the corpus, numerical contract, producer and baseline
identities, and the checksums of the full-vocabulary arrays. Git records earlier
versions. A change to Xenolith alone does not regenerate this data.

Install the lightweight compare profile described in the [package guide](../README.md),
then run these commands from the repository root:

```sh
/tmp/xe-quality/bin/xe-quality-fixtures verify \
  --fixtures tools/quality/fixtures/gemma4

/tmp/xe-quality/bin/xe-quality-backend --source . --model /data/model.gguf \
  --output /data/new-candidate/backend.json

/tmp/xe-quality/bin/xe-quality-fixtures compare \
  --fixtures tools/quality/fixtures/gemma4 --model /data/model.gguf \
  --baseline panther-vulkan \
  --backend-config /data/new-candidate/backend.json \
  --output /data/new-candidate/comparison
```

Comparison verifies the entire fixture before starting the candidate. It needs
NumPy, the exact GGUF and the current-build native driver; it imports neither
JAX nor Torch and performs no Google or llama model inference. The native
candidate runs three times from new processes, with recorded input orders.
Select `panther-vulkan` or `raptor-vulkan` explicitly. The report identifies the
chosen profile; selection does not automatically certify that the candidate
ran on matching hardware. Check its GPU/runtime provenance against that profile.
Verification checks every baseline in a fixture, including unselected profiles. Missing or
unknown selections fail before inference. Every output is retained in the chosen new comparison directory. Use
`--repetitions 1` for a faster diagnostic, which remains `INCONCLUSIVE`.

## What is accepted

All engines consume the same token IDs and declared chunk schedule. Each saved
row describes the next-token distribution after the specified input position.
Logits cover all 262,144 tokens, after the model softcap and before sampling,
at temperature one. A long document needs only a few saved rows, but the model
must still process its full prefix.

The [Gemma4 dataset card](gemma4/README.md) lists the five inputs and 14 saved
positions, with a maximum prefix of 5,348 tokens. This covers cached prefill;
it does not qualify 8,192-token inputs, single-token decode, cache restoration
or every intermediate position.

The primary quantity is `KL(Google || Xenolith) - KL(Google || selected Vulkan baseline)`;
negative values favor Xenolith. Average the observed positions within each
document group, then give groups equal weight within each family (the corpus
`stratum`), then give families equal weight in the overall mean. Thus a longer
document or a family containing more documents does not receive extra weight.
The acceptance margin on this overall mean is exactly zero.

The report retains each family's mean KL, maximum KL and maximum total variation,
and lists individual positions where Xenolith is farther from Google on KL or TV.
These are nonblocking diagnostics: a `PASS` can contain worse families or positions.
`diagnostics.worse_mean_kl_families`, `family_endpoint_regressions` and
`position_regressions` make those cases explicit; positive deltas mean worse.
Differences with an uncertain sign under the precision check are listed separately.

The decision is scoped to this **finite public suite**:

- `PASS` (exit 0): the equal-family mean KL is no worse than the baseline, and all three
  native runs have identical output bytes and recorded dispatch.
- `FAIL` (exit 1): a stable candidate has a worse equal-family mean KL.
- `INCONCLUSIVE` (exit 3): executions vary, fewer than three runs are available,
  or the overall mean difference changes sign between FP64 and the
  extended-precision metric check. Local diagnostic differences do not veto a pass.
- `ERROR` (exit 2): integrity, input, execution or completeness checks fail.
  No scientific acceptance is issued for incomplete output.

The reported resolution interval brackets the FP64 and `longdouble` calculations;
it is an empirical numerical sensitivity check, **not** a statistical confidence
interval or a rigorous floating-point error bound. The report identifies the
actual `longdouble` precision. A small, repeatedly used public corpus cannot
establish statistical equality on unseen texts. Longer prefixes from the same
document are not independent documents. `population_inference` therefore remains
`NOT_ESTABLISHED`, even when this suite passes. Task quality, vision and unmeasured
positions are separate questions.

The manifest explicitly identifies the `mean-family-kl` gate and its aggregation.
Older manifests with per-family endpoint gates retain their original behavior.
Changing this acceptance rule changes the fixture manifest hash, while the saved
logits can be reused unchanged. Historical reports remain tied to their original rule.

## Reference and baseline

Google is the trusted operational reference on the values represented by the
exact GGUF. New fixture generation does not require a second HF run and does
not claim new independent HF qualification. Independently qualified archives use the separate archive CLI and retain
their recorded qualification scope.

Both included baselines use official llama.cpp commit
`f0458bd05e698306ce423b1850164042112f59c4`, full layer offload to one explicitly
selected Vulkan GPU, eight regular and batch CPU threads, F16 K/V, native CPU
optimizations and production extra buffer types enabled.
Context capacity is explicit per case; requested batch and microbatch capacity
are 1,024. Flash attention is explicitly **OFF**. K/Q/V and supported operations
are offloaded. Each profile records its device, compiler, driver/runtime hashes
and observed dispatch. These profiles do not represent other GPUs, drivers or
llama.cpp configurations; a performance comparison must use the same profile
before claiming matching numerical quality and speed.

Historical v1 CPU fixtures remain readable under their original contract.
Their verdicts do not establish a result against either Vulkan baseline.

To reproduce a backend, supply a clean checkout of the exact revision, Vulkan
development headers/loader, SPIRV-Headers (including its CMake package) and
`glslc`, then build on the intended GPU host into
a new output directory. The builder records the actual compiler; `--cc` and
`--cxx` select explicit compiler executables. Runtime environment controls must
remain identical between building/probing and capture:

The maintained wrapper uses the same numerical configuration as the included
baselines. Exact sources used for the committed arrays are identified in the
[producer source record](producers/README.md).

```sh
OMP_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8 OMP_DYNAMIC=FALSE \
  /tmp/xe-quality/bin/xe-quality-llama-backend \
  --source /data/llama.cpp --profile vulkan --device Vulkan0 \
  --driver-source tools/quality/native/llama_driver.cpp \
  --model /data/model.gguf --output /data/new-llama-build
```

The receipt records the compiler, CMake options, source, libraries and executable
hashes, including the dynamically loaded Vulkan ICD. A hardware Vulkan device
and full layer offload are required; CPU fallback cannot become a Vulkan fixture.
Rebuilding on a different CPU/runtime produces a new baseline identity;
do not silently substitute its logits for the committed baseline.

## Reproduce the arrays

Run the following capture on each intended GPU host, using its own freshly
built receipt. This executes five cases in three independent passes: 15 native
processes per host. The repeated logits and dispatch must match exactly;
nonrepeatable outputs are retained and cannot be sealed. Stderr evidence is
hashed in the capture receipt. These are repeatability checks, not statistical
population samples.

```sh
OMP_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8 OMP_DYNAMIC=FALSE \
  /tmp/xe-quality/bin/xe-quality-fixtures capture \
  --corpus tools/quality/fixtures/gemma4/corpus/manifest.json \
  --model /data/model.gguf --backend-config /data/new-llama-build/backend.json \
  --output /data/new-llama-rows

```

Collect the complete capture directories, then seal a new
fixture. Existing Google/corpus bytes and producer identity are reused unchanged;
no Google or HF inference is needed:

```sh
/tmp/xe-quality/bin/xe-quality-fixtures seal-vulkan \
  --fixtures tools/quality/fixtures/gemma4 \
  --baseline panther-vulkan=/data/panther-capture \
  --baseline raptor-vulkan=/data/raptor-capture \
  --output /data/new-fixture
```

To include additional hosts, repeat `--baseline NAME=CAPTURE` with each verified
capture. All supplied captures must satisfy the same corpus and repeatability
contract. Generate a separate baseline before claiming a comparison against
llama.cpp Vulkan on hardware not represented by the included profile.

Generation is explicit and resource intensive. Use a host with sufficient memory
and its normal admission controls; it is not part of ordinary CI or comparison.
All output directories must be new. Preserve the production receipts separately;
`seal-vulkan` copies the corpus, verified arrays and reproduction metadata:
device/runtime identity, compiler and numerical settings, source hashes and
observed dispatch. Local
paths, desktop-session variables and hashes of unpublished build receipts or
logs remain in the capture directory. Verification of a published fixture
checks its arrays and declared profile; auditing the original repeated executions
requires the complete capture, including logs. Review the generated fixture and
its source attributions before replacing repository data. Changing texts,
tokenization, selected positions or state protocols requires
corresponding new reference rows; changing only the candidate does not.

To reproduce Google outputs independently, install the separate reference profile
and fetch the pinned upstream sources described in the package guide, then run
`python -m xe_quality.fixture_reference --corpus CORPUS_MANIFEST --model MODEL
--google-source GOOGLE_SOURCE --output NEW_DIRECTORY --max-forwards 40` in that
environment. This is optional reference generation, not a step in replacing a
Vulkan baseline on an unchanged corpus.

Keep the corpus, baseline and rule fixed while evaluating an optimization.
Do not remove failing cases or regenerate a more favorable baseline in response
to a candidate result. For broader validation, freeze the candidate before
selecting fresh documents and targeted state/length controls. Once used for
tuning, those cases become development data too; this public suite is never a
hidden confirmation set.
