# Xenolith numerical fidelity

This optional package compares the current engine with immutable full-vocabulary
reference results from **Google Gemma4 FP64**, using the exact represented GGUF
weights. HF/Torch is a separate independent qualifier of the reference producer.
Ordinary comparisons use only Python, NumPy, a compatible reference archive,
the model GGUF and a current-build native driver. No reference model is loaded
by a comparison, and an engine-only change does not invalidate reference results.
The GGUF reader supports F32, F16 and Q4_0 tensors. The Google adapter also
checks the Gemma4 26B tensor and metadata schema; unsupported layouts fail
explicitly. Each archive records its exact model identity.

Install explicitly from the repository root. On Linux x86_64 use Python 3.12:

```sh
python3.12 tools/quality/setup_environment.py --profile compare --output /tmp/xe-quality
/tmp/xe-quality/bin/xe-quality --help
/tmp/xe-quality/bin/xe-quality self-check
make check-quality-unit PYTHON=/tmp/xe-quality/bin/python
```

The helper creates a new venv, installs the hashed transitive profile lock and
the public package, runs pip check and records `QUALITY-INSTALL.json`. It refuses
existing destinations. The C build and tests never install anything. On other
platforms an explicit `python -m pip install ./tools/quality` installs the NumPy
package; the supplied tested locks cover Linux x86_64/Python 3.12 only. Platform
and runtime changes need their own qualification. No model is downloaded.

Build the driver and freeze its provenance:

```sh
/tmp/xe-quality/bin/xe-quality-backend --source . --model /data/model.gguf \
  --output /data/new-run/backend.json
/tmp/xe-quality/bin/xe-quality verify-archive --archive /data/reference
/tmp/xe-quality/bin/xe-quality diagnose --archive /data/reference \
  --model /data/model.gguf --backend-config /data/new-run/backend.json \
  --output /data/new-run/diagnostic --timeout 1800
```

The builder forces compilation of the driver and engine objects from the chosen
checkout with its normal flags/kernel. It records exact source and dirty diff,
commit, build log, compiler/flags, SPIR-V, executable and linked-library hashes.
Comparison rejects changed pinned inputs. Rebuild into a new receipt after any
candidate change. Backend receipts contain paths specific to the chosen build;
generate them at the execution location, rather than copying a receipt and
silently substituting another binary. The driver uses the public session API,
its own cache and eight CPU workers.
`--mode prefill` follows recorded chunk endpoints; `--mode decode` is an explicit
one-token diagnostic state variant and cannot pass the current prefill qualification gate. `--layers /data/new-layer-dir` enables real-engine hidden
state diagnostics and validates all layer/position rows; see the testing guide.

Every comparison uses a new output directory and retains full raw candidate
logits, driver output, input tokens, metric report, reference identity, grouping
and limitations. Logits are after model softcap and before sampling, temperature
1, over the entire 262144-token vocabulary. Metrics include stable KL(B||candidate),
TV, NLL aligned to observed successors, top-token disagreement and margins.
Positions are averaged within passage/session groups, then groups weighted equally
within each explicit stratum. There is no fabricated global average across
unknown usage weights or inference of free-response task scores.

`diagnose` always labels successful calculations **INCONCLUSIVE** and returns 0;
a technical failure returns 2. `compare` is the automation gate: PASS=0, FAIL=1,
ERROR=2, INCONCLUSIVE=3. A missing policy, insufficient groups, incompatible or
unqualified scope cannot pass. No candidate acceptance bounds are supplied.
A policy is a caller-owned JSON object with schema `xe-quality-policy-v1` and
`domains` mapping every measured stratum to `{min_groups, metrics}`. Each metric
maps to exactly one finite `max` or `min` bound. Supported keys are
`kl.mean_group_mean`, `kl.max_observed_position_value`, the same two suffixes for
`tv` and `top1_disagreement`, and `nll.delta` with either suffix. Choose bounds
and sample requirements outside this tool based on your actual requirement;
the software tests use synthetic bounds to exercise the gate, not to approve a
model. The current producer qualification covers one exact fixed-token input, selected
positions and declared state protocol. Its gate checks that normalized input hash
and the actual cache-control evidence and allowed chunk vectors; semantic domain labels do not extend this
calibration to other histories. Multi-history qualification is not implemented.
Archives without compatible qualification remain inconclusive even with a policy.

Archives carry a versioned, hashed contract: GGUF identity, exact token corpus,
template, selected positions, state/chunks, producer source/precision/runtime,
qualification and artifact schema/checksums. `import-archive` and `export-archive`
copy compatible results without changing their original producer or index bytes:

```sh
/tmp/xe-quality/bin/xe-quality import-archive --archive /data/original \
  --output /data/reference
```

Supply your reference archive directory explicitly. Keep its input provenance,
producer identity and qualification packet with the arrays. Review the source
dataset licenses and access terms before redistributing text or model weights.
Source acquisition records upstream revisions and hashes.
Corpus readers are offline. Optional acquisition/tokenization commands
are separate modules (`xe_quality.acquisition`, `xe_quality.tokenize_panel`) with
`--help`; acquisition requires `--output NEW_DIR` and refuses existing output.
They do not import reference or qualification frameworks.

The optional official RULER NIAH noise generator has its own locked profile.
It generates prompt data, without model inference or benchmark scores. Supply
an independently acquired RULER checkout at the pinned revision, the exact GGUF
and a tokenizer executable built from this checkout (`make quality-tokenizer`):

```sh
python3.12 tools/quality/setup_environment.py --profile corpus --output /tmp/xe-corpus
git clone https://github.com/NVIDIA/RULER.git /data/ruler
git -C /data/ruler checkout db43f92ff634bbce1c33d1fb15858aec4bba509a
/tmp/xe-corpus/bin/python -m xe_quality.run_official_ruler \
  --source /data/ruler --tokenizer "$PWD/tools/quality/native/xe_tokenizer" \
  --model /data/model.gguf --output /data/new-ruler --length 512 --num-samples 2
```

The generator verifies the executed upstream Python bytes and records its seed,
dependency/wordlist identities, model/tokenizer hashes and output JSONL hashes
in `GENERATION.json`. Keep that receipt with the JSONL when using
`xe_quality.tokenize_panel`. The pinned noise subset does not need downloaded
NLTK data. Frozen corpora need none of these generation dependencies.

For **new reference generation**, install separate environments:

```sh
python3.12 tools/quality/setup_environment.py --profile reference --output /tmp/xe-reference
python3.12 tools/quality/setup_environment.py --profile qualification --output /tmp/xe-hf
/tmp/xe-reference/bin/python -m xe_quality.reference.fetch_google --cache /data/google-cache
/tmp/xe-quality/bin/python -m xe_quality.qualification.driver --help
```

The Google profile pins JAX/jaxlib 0.8.2, Flax 0.12.2, NumPy 2.3.5; Google sources
are explicitly fetched at the immutable revision and verified against individual
executed-file and license hashes before execution. No full upstream clone is
vendored. The adapter's bounded expert dispatch, stable routing, streaming and
FP64 cast/RoPE changes are maintained source, part of the producer identity.
The qualifier launches both children with Python isolated mode (`-I`), removes
inherited Python search-path/home overrides, and verifies both installed package
source manifests before either model worker starts. Install this exact package
separately in both selected environments; source-tree fallback is not supported.

The HF profile uses a separate Torch CPU wheel from its official index, pinned
Transformers/tokenizers/safetensors and the independent precise adapter.
Installation is not numerical qualification. Run model-producing commands on
a system with the model and sufficient memory;
ordinary CI runs no model or GPU inference.

The qualification driver takes an exact fixed-token JSON input, separate Google
and HF Python executables, explicit upstream source and a new evidence directory.
It requires the final token among the selected positions, retains independent
process outputs and arithmetic/runtime identities, and declares only the actually
checked whole-history and midpoint-split chunk protocols. It
uses the existing producer validation protocol, not a candidate-quality policy.
Qualify the exact producer and runtime used to build an archive. Results from
a different producer identity do not qualify it.

Reference configuration requires `weights_sha256`, `reference_precision: "fp64"`,
finite positive `max_reference_forwards`, optional `record_keys` and optional
`positions` keyed by `source:record_id`. The frozen corpus records exact token
IDs, source revisions/hashes, tokenizer/template, groups and state. Use the CLI's
`prepare-corpus` on token-exact JSONL or the independent dual-tokenizer builder.
Only an explicit compatible qualification is accepted by `build-reference`:

```sh
JAX_PLATFORMS=cpu OPENBLAS_NUM_THREADS=8 OMP_NUM_THREADS=8 \
  /tmp/xe-reference/bin/xe-quality build-reference --model /data/model.gguf \
  --google-source /data/google-cache/gemma-9e1f74131ba9d45e5fe6ae7bb3941d9f2abb39c3 \
  --corpus /data/corpus.json --config /data/reference-config.json \
  --qualification /data/qualified/QUALIFICATION.json --output /data/new-reference
```

The producer is text-only. Qualification applies only to the measured inputs,
positions, state protocols and runtime identities. Successful execution at a
longer context does not establish numerical accuracy there. Small panels and
shared GGUF mapping limit independence and coverage; vision, state and operator
checks remain separate. See [testing](../../docs/testing.md) and the
[migration guide](../../docs/quality-migration.md).
