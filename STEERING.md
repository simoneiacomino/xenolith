# Experimental activation steering

Rebased onto upstream `d730fa9` on 2026-11-10. Portable steering checks,
configuration unit tests and CLI syntax checks pass; Linux/Intel engine and
model validation remains pending. The [previous rebase validation](steering/REBASE_VALIDATION.md)
and earlier measurements retain their recorded source/executable identities.

Xenolith can extract activation directions locally from its Gemma 4 26B-A4B IT
QAT GGUF and apply them during inference. The original model weights are used
for both operations. A downloaded control vector is not required.

This feature is experimental. Artifact compatibility and numerical correctness
are distinct from behavioral quality. An artifact can load correctly and still
produce poor answers at an unsuitable strength. No default strength is a quality
guarantee; compare against ordinary system prompting on your own tasks.
The bundled succinctness experiment did not yield a useful validated setting;
see [the evaluation report](steering/EVALUATION.md). Command strengths below
illustrate the interface and are not recommendations for answer quality.
The separate [English/Italian pilot](steering/LANGUAGE_EVALUATION.md) provides
the current language-steering experiment and a one-command native chat launcher.

The [ds4-style verbosity follow-up](steering/VERBOSITY_EVALUATION.md) uses
locally extracted FFN directions and demonstrates shorter validation answers,
with documented information loss and formatting limitations.

## Model identity and provenance

Use the exact same GGUF for capture and inference. Xenolith binds an artifact
to the SHA-256 of every byte of that file, not its filename or model family.
QAT describes quantization-aware training; it does not certify the provenance
of a particular download.

The reference GGUF already recorded in `tests/vision/manifest.json` is the
Unsloth Gemma 4 26B-A4B IT QAT UD-Q4_K_XL artifact at revision
`7b92b5b28818151e8669af2e45e88d6086f490dd`, with SHA-256
`a7c5bc715f5ff8e99a3e8901ce7d2b42b402c669bf24f7c5250747633d0f5891`.
That is a third-party GGUF distribution. Recording its source, revision and
hash establishes reproducibility of the chosen file.

Steering-enabled CLI commands compute the full model hash at startup, including
when the service uses `--cache-budget 0`. Hashing happens before repacking and
is reused for artifact checks. There is no per-token hashing or file loading.

## Extract a direction locally

Build Xenolith normally on supported Linux/Intel hardware. Python 3 is needed
for the builder; it uses the standard library. Start with the bundled 100-pair
English dataset:

```sh
python3 tools/build_steering.py \
  --xenolith ./xenolith \
  --model /path/to/model.gguf \
  --dataset steering/examples/succinct-train.jsonl \
  --out steering/local/succinct-projection \
  --mode projection \
  --layers 8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27 \
  --model-source https://huggingface.co/unsloth/gemma-4-26B-A4B-it-qat-GGUF \
  --model-revision 7b92b5b28818151e8669af2e45e88d6086f490dd
```

The output directory must not exist. This is an initial experimental layer
selection, not a measured universal optimum. Adjust the source and revision
when using another GGUF. The builder writes `manifest.json` and `vectors.f32`;
its stdout includes their artifact ID.

Each JSONL row contains nonempty `positive` and `negative` prompt strings and an
optional `system` string. The default system prompt is `You are a helpful
assistant.` The two examples should isolate one intended behavioral difference.
The included data changes the requested answer detail while keeping the topic
fixed. Do not use the validation/evaluation prompts as extraction examples.

The native `steering-capture` command loads the model once and uses fresh KV
state for each example. It captures the last token of the complete rendered
prompt, including the assistant opening, at every layer's post-layer residual.
It uses the production CPU/GPU paths rather than a separate framework. The
optional builder flag `--orthogonalize-control` removes the component of the
mean difference parallel to the negative-class activation mean before any unit
normalization. This can reduce interference with the common activation state;
it is a separately identified extraction method and still requires evaluation.
The builder subtracts negative activations from positive activations, averages
paired differences using compensated summation, and optionally normalizes each
selected row. There is no backpropagation or weight update.

To retain captures for separate mode/layer experiments:

```sh
./xenolith steering-capture /path/to/model.gguf \
  --dataset steering/examples/succinct-train.jsonl \
  --out /tmp/succinct-capture --ctx 512
```

Then pass `--capture-dir /tmp/succinct-capture` to the builder in place of
`--model`. Capture metadata binds the dataset bytes, model, template, reasoning
mode, engine revision and numerical build settings. Reusing a capture with a
different dataset is rejected. Add `--thinking` during capture when that is the
intended extraction setting. Extraction fails on context overflow instead of
truncating examples. Existing output files are never overwritten.

For language experiments, `--completions` captures the mean post-layer state over
teacher-forced assistant answer tokens. Each dataset row then requires a shared
`prompt` plus `positive` and `negative` answer strings. Answers are appended after
the rendered assistant opening, not placed in the user message. Reasoning must
be disabled for this capture recipe. `--cpu` processes the prefix one token at a
time for diagnostic comparisons; otherwise the prefix uses production prefill.
Both modes save `*.tokens.json` traces containing exact token IDs and observed
positions. The builder accepts the same flags and records the completion method
separately in artifact provenance. This extraction recipe still needs behavioral
validation and is not automatically a useful direction.

## Apply steering

```sh
./xenolith chat /path/to/model.gguf \
  --steering steering/local/succinct-projection/manifest.json \
  --steering-strength -0.25
```

`run` accepts the same options and retains its existing raw-prompt behavior.
Use `chat` or the wire service for the model's chat template. During interactive
chat, `/steer F` changes the loaded vector's strength and `/steer off` disables
new interventions. Loading a different artifact in a durable session is
available through the service. The oracle accepts the same steering options
for numerical reference work.

For instruction-following questions, use `chat`, which renders the conversational
template. Text-only `run -p` performs raw completion of the literal prompt.
Compare fresh chats when assessing a behavioral change.

The first supported site, `post_layer_residual_v1`, is after the FFN branch
normalizations, attention residual addition, and final layer output scaling.
Both CPU decode and GPU prefill intervene there. Steering applies to every
processed text position in selected layers, including prompt tokens and replay.
Visual input in a session with steering history is rejected in this version.

The additional experimental site `post_ffn_output_v1` operates on Gemma's
combined dense/routed FFN contribution after its branch/combined normalizations,
before attention-residual addition and `layer_out_scale`. This is the FFN-output
analogue used for the ds4-inspired verbosity comparison. Capture it explicitly:

```sh
./xenolith steering-capture /path/to/model.gguf --dataset pairs.jsonl \
  --out steering/local/ffn-capture --ctx 512 --capture-site post_ffn_output_v1
```

The builder reads the site from capture metadata. Captures without a site retain
the original residual-site meaning. Re-capture for a different site: relabeling
a manifest cannot turn residual activations into FFN-output activations.
Both modes are supported at either site. CPU/GPU zero-strength paths bypass the
intervention; FFN capture leaves the original unsteered hidden/logit path intact.
Snapshots already include the artifact ID, which binds site as well as mode.
Previous loaders explicitly reject artifacts declaring the new site.

Two operations have deliberately different semantics:

```text
additive:   h' = h + strength * v
projection: h' = h - strength * dot(h, v) * v
```

Projection rows have unit norm. Strength 1 removes the represented component,
zero skips the edit, and negative strength amplifies the component already
present. Negating a projection vector does not change that operation. Additive
rows preserve their extracted magnitude; changing the strength's sign changes
the direction of the displacement. Calibration values do not transfer between
these two modes. For either mode, absent/zero steering uses the unchanged
numerical path and launches no steering GPU kernels.

## Registered artifacts and durable sessions

Import an artifact into a local registry before using the service:

```sh
./xenolith steering-import /path/to/model.gguf \
  --steering steering/local/succinct-projection/manifest.json \
  --out /path/to/steering-registry

./xenolith wire /path/to/model.gguf \
  --state /path/to/state --steering-dir /path/to/steering-registry
```

The import prints an ID. The registry contains immutable copies under
`REGISTRY/ID/manifest.json` and `REGISTRY/ID/vectors.f32`. Keep historical
artifacts as long as their conversations may need replay. Missing/incompatible
artifacts cause an explicit error, including when a snapshot is available.
Network requests select IDs and cannot ask the server to load arbitrary paths.

After `create` or `open`, use the `steer` operation:

```json
{"op":"steer","artifact":"<64-character-artifact-id>","strength":-0.25}
{"op":"steer"}
{"op":"steer","artifact":null}
```

The reply reports the selected artifact, strength, effective context position
and number of transitions. A transition is durably recorded before the update
is acknowledged. Changes are accepted while idle, between generation requests.
Changing settings retains existing KV. Already-computed logits remain valid
historical output: the next sampling step may use them, and the new setting
applies when the next token is processed.

Artifacts and histories belong to sessions; changing one conversation does not
change another. An `ephemeral` request can explicitly select its own steering:

```json
{"op":"ephemeral","messages":[{"role":"user","text":"Explain DNS."}],"temperature":0,"reasoning":false,"max_tokens":128,"steering":{"artifact":"<64-character-artifact-id>","strength":-0.25}}
```

Omission or `null` starts an unsteered temporary request. Ephemeral settings do
not change the durable conversation.

### Historical effects and replay

Steering is prospective. Disabling it does not undo cached activations or text
already generated. Xenolith persists the ordered schedule, including transitions
that have not yet affected any token. Snapshots bind their numeric identity to
the schedule that produced their prefix. A pending change at the next context
position does not invalidate an otherwise compatible historical snapshot.

The first transition freezes the current projected context. Subsequent turns
retain their original generated tokens, including reasoning and control tokens,
so positional steering boundaries stay meaningful across restart and replay.
Consequently, `reasoning.history` filtering no longer removes historical
reasoning in that conversation, even after steering is disabled. This consumes
additional context compared with ordinary discard-mode conversations. Current
reasoning effort can still control future generation budgets, but it cannot
retroactively change the frozen system prefix. Gemma's global `<|think|>` marker
must already be present for its usual thinking template: enable reasoning before
the first steering transition, or rebuild after changing reasoning settings.
A model may close its reasoning channel immediately when that marker is absent.
Rebuild replaces the
transcript and starts a fresh schedule at zero using the selected setting;
rewind removes transitions on the discarded branch.

Steered conversations use synchronous history reconciliation rather than
background reasoning-removal shadows. On cache misses, replay uses the recorded
settings at each token position. GPU rows at a transition are handled separately
before the next layer consumes them. CPU/GPU numerical differences still exist;
replay does not promise cross-backend bitwise-identical continuations.

The conversation extension is a critical event: older readers reject its
committed history instead of silently ignoring it. Existing unsteered snapshots
retain the previous format and numeric identity. Older engines reject steered
snapshots through their incompatible numeric identity.

## Native artifact format v1

The UTF-8 JSON manifest is limited to 64 KiB. Required fields are:

| Field | Value / meaning |
| --- | --- |
| `version` | Integer `1`. |
| `layers`, `width` | Exact model shape; currently 30 and 2816. |
| `site` | `post_layer_residual_v1` or experimental `post_ffn_output_v1`. |
| `dtype`, `payload` | `f32le`, `vectors.f32` beside the manifest. |
| `model_sha256`, `payload_sha256` | Lowercase SHA-256 strings. |
| `mode` | `projection` or `additive`. |
| `normalization` | `unit` for projection; `none` for additive. |
| `active_layers` | Nonempty array of unique zero-based layer indices. |
| `token_policy` | `all_text`. |
| `provenance` | Object described below. |

The provenance object requires string fields `model_source`, `model_revision`,
`dataset_sha256`, `template_sha256`, `method`, `position`, `reasoning`,
`engine_commit`, `numeric_settings`, `positive_label`, `negative_label`, and
`evaluation`. Dataset and template identities must be lowercase SHA-256.
`evaluation` is `unevaluated; no recommended strength` until an actual evaluation
is recorded; the loader does not treat that string as a quality certificate.

The payload is a row-major `[layers,width]` matrix of little-endian float32.
Inactive layers contain zeros. Active rows must be finite and nonzero;
projection squared norm must be within `1e-4` of 1. Wrong hashes, dimensions,
unsupported contracts, invalid masks, non-finite values, and short/extra payload
bytes are rejected before configuration changes.

Artifact ID is SHA-256 of the following concatenation:

```text
UTF-8 bytes "xenolith-steering-v1" + one NUL byte
exact manifest bytes
exact payload bytes
```

The builder emits sorted-key compact JSON with a trailing newline. Import
preserves those bytes. Whitespace or metadata edits change identity; renaming
an unchanged manifest does not. No filesystem path participates in identity.

## Validation and evaluation

```sh
make check-steering-host
make check-steering
make check-unit check-context
make check-gpu
make check-all MODEL=/path/to/model.gguf
```

The focused tests cover analytical operations, artifact identity and validation,
GPU rows/masks/transitions, frozen context, restart, and rewind. The optional
model integration runner is `tests/check_steering_session.py`; its help describes
registry and artifact inputs.
`python3 tests/check_steering_capture.py --model /path/to/model.gguf` checks
native capture completion and explicit failure for invalid or oversized final
pairs, including datasets without a trailing newline.

`tools/evaluate_steering.py` compares baseline, a concise system prompt, and an
explicit strength sweep with greedy decoding, fixed seed, and fresh contexts.
Use `succinct-validation.jsonl` to select settings; reserve
`succinct-eval.jsonl` (including Italian prompts) for held-out evaluation. Record
full answers as well as length, truncation, repetition, first-text latency and
runtime throughput. General factual quality needs answer review; word counts
alone cannot establish correctness. Additional JSON/Python constraint prompts
can carry `kind` and expected-output fields for mechanical checks.

See `steering/EVALUATION.md` for the environment, commands, measured results,
remaining limitations, and release status of this implementation.

To measure enabled steering cost with fixed tokens rather than different-length
answers, build `make bench/bench_steering` and run:

```sh
./bench/bench_steering /path/to/model.gguf /path/to/manifest.json -0.25 3
```

It warms both configurations, then measures off/steering/steering/off order for
three repetitions, using 512 prefill tokens and 128 fixed decode tokens. JSONL
records separate startup, prefill and decode timings. Run without concurrent
inference; this utility holds the same engine lock as the CLI.
Use `bench/text_regression.py --paired` with a separately built baseline
binary to measure disabled-path regression against the original source.

The local builder is an independent implementation. The ds4 extraction workflow
and optional control-mean orthogonalization were inspected at upstream commit
[`ee7e8405232c8971000cc7ba2b945b00bf6d20d3`](https://github.com/antirez/ds4/blob/ee7e8405232c8971000cc7ba2b945b00bf6d20d3/dir-steering/tools/build_direction.py).
No ds4 model weights or generated direction files are included.

## C API integration

`steering.h` owns immutable reference-counted artifact handles; `xenolith.h`
exposes session configuration and native capture. Link `steering.o` and `json.o`
alongside the existing engine dependencies, as the Makefile targets do.

```c
xe_steering *vector = NULL;
char error[256];
if (!xe_steering_load(manifest_path, xe_engine_layer_count(engine),
                     xe_engine_hidden_width(engine),
                     xe_engine_model_fingerprint(engine),
                     &vector, error, sizeof error)) {
    /* Report error and leave the current session configuration unchanged. */
}
/* Configure only after a successful load, on the engine owner thread. */
if (vector) {
    int configured = xe_session_steering_set(session, vector, -0.25f);
    xe_steering_release(vector); /* A configured session retains its own reference. */
    /* Check configured before continuing. */
    (void)configured;
}
```

A session retains the full ordered `xe_steering_step` schedule. Use
`xe_session_steering_schedule` to replace a complete history when managing replay
or branches directly; `xe_session_steering_set` appends a prospective transition
at the current evaluated position. `xe_session_reset` resets KV evaluation while
retaining that schedule for replay. Clear it explicitly with an empty schedule
when starting unrelated work. Inspect the const schedule through
`xe_session_steering_steps` and `xe_session_steering_count`.

Lower-level snapshot readers must configure the expected schedule before loading
a steered snapshot. The snapshot binds its cached prefix to that history but does
not contain the vector payloads. The durable runtime performs history/artifact
restoration automatically. Capture is enabled on an empty unsteered session with
`xe_session_capture_enable`; read its session-owned rows only after a successful
nonempty synchronization. Artifact handles and session mutation follow the
engine's single owner-thread execution model.
