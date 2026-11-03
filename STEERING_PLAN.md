# Activation steering implementation plan

Status: implementation complete as an experimental feature. The bundled
succinctness experiment has a negative validation result and no recommended
production strength. See
[STEERING.md](STEERING.md) for user documentation.
The next behavioral research steps are specified in
[steering/NEXT_STEPS.md](steering/NEXT_STEPS.md); their success is not implied
by completion of the runtime integration.
The subsequent [English/Italian pilot](steering/LANGUAGE_EVALUATION.md) now
demonstrates a visible language change on held-out questions, with observed
quality defects documented separately. Anti-refusal is deferred.
The [ds4-style verbosity follow-up](steering/VERBOSITY_EVALUATION.md) adds
FFN-output capture/intervention and demonstrates shorter validation answers,
with required-fact omissions explicitly recorded. Earlier performance claims
refer to the original residual-site implementation.

## Implementation progress

| Milestone | Evidence |
| --- | --- |
| Artifact format and validation | Native loader/import implemented; portable analytical and identity tests pass, including UBSan on the development host. |
| CPU/GPU intervention | Both modes implemented; synthetic GPU comparisons and real-model zero-strength/snapshot checks pass. |
| Local extraction | 100 paired examples captured on the pinned QAT GGUF; projection and additive artifacts generated locally. |
| Durable runtime and protocol | Session transitions, restart, snapshot restore, eviction replay, rewind, rebuild and ephemeral isolation pass the model integration runner. |
| Evaluation | Projection/additive screens reject the bundled direction; JSON/Python diagnostics show an extra coding failure. Held-out prose remains reserved because no candidate passed. Disabled throughput changes are within 0.5%; enabled cost is within observed timing variation. |

The standard ASan bounds gate cannot start correctly with the validation host's
preloaded monitoring library. Unaffected suites were run separately; this is
not a clean `check-all`. The unchanged baseline reproduces the full-model
prefill numerical failure with the same errors. Both limitations are recorded
in [steering/EVALUATION.md](steering/EVALUATION.md).

## Objective and agreed behavior

Add locally generated activation steering to Xenolith's existing Gemma 4
26B-A4B IT QAT inference path. Use the same original GGUF for vector extraction
and steered inference, without modifying its weights. Downloaded steering
vectors are not required.

Steering changes are prospective: retain existing KV state and apply the new
configuration to subsequent forward passes. Disabling steering stops new
interventions; it does not remove their effects from cached states or from the
conversation text. This is intentional behavior.

All implementation documentation, command help, artifact specifications,
evaluation reports, and code comments added for this feature must be in English.
Each implementation milestone must update its documentation in the same change.

## Initial scope

- One loaded steering artifact active per session, with selectable strength.
- Text inference through `run`, `chat`, `wire`, and `serve`.
- Local extraction from paired prompt datasets using production inference.
- CPU decode and GPU prefill implementations of the same intervention.
- Runtime changes between generation operations, with durable steering history.
- Snapshot restore, replay, rewind, and session isolation.
- A reproducible succinct-versus-detailed experiment before other behaviors.

Defer multiple-vector composition, automatic model-selected steering, conditional
steering, external format import, and steering of visual sessions. Initially
reject steering for visual input explicitly; existing unsteered vision behavior
must continue to work. A vision-enabled server can still serve steered text-only
sessions if the runtime can enforce that distinction.

## Current code and integration points

| Area | Existing code | Planned work |
| --- | --- | --- |
| Model dimensions | `xenolith.c`: `XE_LAYERS`, `XE_EMBD` | Validate against 30 layers and width 2816; avoid independent shape constants in tools. |
| CPU decode | `xenolith.c`: `xe_ffn_half_decode` | Intervene after the final residual addition and layer output scaling. |
| GPU prefill | `xenolith.c`: `xe_prefill_ffn_append`; `xenolith.cl`: `xe_prefill_ffn_finish` | Apply the equivalent operation to each eligible row before the next layer consumes it. |
| Reference implementation | `xenolith.c`: `ref_layer`, `ref_ffn_half`, `xe_oracle` | Add an independent numerical reference and comparison hooks. |
| Public C interface | `xenolith.h`, session ownership in `xenolith.c` | Add immutable artifact handles and session-owned steering configuration. |
| User interfaces | `main.c`, `serve.c`, `runtime.h` | Add explicit selection, strength, disable, and inspection operations. |
| Durable history | `conversation.c`, `conversation.h`, `runtime.c` | Persist steering transitions and replay their effective boundaries. |
| Snapshots | Snapshot code in `xenolith.c`, `kvstore.c` | Identify the steering history that produced a cached prefix. |
| Documentation | `README.md`, `docs/protocol.md`, new `STEERING.md` | Explain extraction, application, provenance, persistence, and limitations. |

The existing `tools/dump_layers.c` is a llama.cpp-based reference utility, not
a native production-path extractor. The oracle can already dump layer outputs,
but extraction should ultimately capture the actual production path.

## Intervention contract

Use the post-layer residual state as the first supported site: after combining
the dense and routed FFN branches, their normalizations, the attention residual,
and `layer_out_scale`. Give it an unambiguous versioned identifier such as
`post_layer_residual_v1`. This is a proposed Gemma integration site, not a claim
that it is numerically identical to every ds4 `ffn_out` implementation.

Support two explicit modes, with the artifact declaring its intended mode:

```text
additive:   h' = h + strength * v
projection: h' = h - strength * dot(h, v) * v
```

Projection requires unit-length active vectors. Strength 1 removes the
component, 0 performs no intervention, and negative strength amplifies the
existing signed component. Reversing the vector's sign does not reverse a
projection intervention. Additive steering has different strength semantics;
do not transfer calibration values between modes or silently normalize its
vectors. Use projection for the initial ds4-inspired experiment and evaluate
additive steering as a separately calibrated comparison.

Start with application to all processed text positions in selected layers,
including prompt prefill, generation, and replay. This keeps the initial policy
independent of whether a token is processed through CPU decode or GPU prefill.
Generation-only and answer-only policies are deferred.

When steering is absent or strength is zero, bypass the new numerical operations
and GPU launches. This must preserve the existing unsteered numerical path.

## Artifact and provenance contract

Use a versioned JSON manifest plus an explicitly little-endian float32 payload
for the initial native format. A complete 30-by-2816 bank occupies 337,920 bytes
(330 KiB), excluding metadata. Layers can be selected using a declared mask;
inactive rows must have a documented representation.

The manifest must identify:

- Payload hash, shape, dtype, byte order, layer indices, and schema version.
- Exact source GGUF SHA-256, model name/revision when known, and source URL.
- Intervention site, mode, normalization, and supported token policy.
- Extraction method, dataset hash, template identity, reasoning setting,
  token-position selection, engine revision, and relevant numerical settings.
- Direction labels, recommended strength only if evaluated, and report location.

Reject wrong shapes, unsupported versions/sites/modes, truncated or oversized
payloads, non-finite values, invalid layer indices, and model-hash mismatches.
Handle zero-norm candidate directions explicitly during extraction; do not emit
NaNs or silently mark them as valid normalized active directions.

Artifact identity must cover the semantic manifest and payload, not the filename.
Define canonical serialization and hash boundaries in `STEERING.md`. Import
validated artifacts into a content-addressed local store so durable sessions do
not depend on a mutable original path. An identical artifact under a different
filename must resolve to the same identity.

QAT and a matching hash do not establish provenance by themselves. Record the
trusted source and revision of the chosen GGUF; a hash binds subsequent work to
those exact bytes. Do not describe the current third-party GGUF as an official
Google-distributed file unless that provenance has been verified.

Steering requires model hashing even when disk KV caching is disabled. Compute
and reuse the identity before repacking releases source pages; document the
startup-cost change for steering-enabled commands.

## Session, boundary, and cache semantics

Artifacts are immutable and may be shared by sessions. The selected artifact,
mode, strength, layer selection, and transition history belong to each session.
Do not implement steering as mutable engine-global state.

Accept configuration changes only while the affected session is idle. The
operation returns the effective context position and event order. A transition
at position P applies to forward passes for positions P and later, until the
next transition. Positions count actual model context elements, not display
characters or message numbers. Multiple changes at the same position need
deterministic event ordering.

Already-computed logits are also historical state. If logits are ready when a
change is accepted, the next sampling step can consume those logits unchanged;
the new configuration takes effect when the next token is processed. Document
and test this boundary explicitly rather than silently reprocessing a token.

Persist each transition before acknowledging it, including artifact identity,
configuration, effective position, and ordering relative to conversation events.
Persist the selected configuration even when no token has yet been evaluated
under it. Invalid or failed changes must leave session state unchanged.

Snapshots must bind the cached prefix to its steering schedule, not only the
currently selected vector. Include sufficient identity to reject a snapshot
from a different steering history while preserving valid prefix reuse. Account
for pending transitions and stored logits at a checkpoint boundary.

On replay, split prefill work at steering boundaries and reproduce the recorded
schedule. Rewind restores the applicable configuration and removes transitions
on the discarded branch, following existing transcript branching semantics.
For a rebuild that replaces the transcript, start a new schedule using an
explicit configuration; do not reuse positional transitions from unrelated text.

Audit reasoning-history filtering, synthetic delimiters, tool continuations,
shadow sessions, cancellation, and temporary generations. Persisted positions
must either remain valid in the replay representation or have an explicit
mapping. A replay that omits or changes text cannot be claimed numerically
equivalent to the original execution.

Missing historical artifacts must produce an actionable error, not silently
replay without steering. Existing unsteered histories retain their current
meaning. Version persisted extensions so older readers cannot silently load
steered state as unsteered state. CPU/GPU replay comparisons use justified
numerical tolerances; do not promise cross-backend bitwise identity.

## Implementation milestones

### 1. Define and validate artifacts

Implement a small artifact loader/validator, ownership API, identity computation,
and local artifact storage. Prefer dedicated `steering.c` / `steering.h` for
format handling over expanding unrelated inference code.

Deliver `STEERING.md` with exact formulas, site definition, artifact schema,
provenance requirements, and inactive-layer behavior. Add malformed-file and
identity tests, including failure without partial state changes.

Exit condition: a valid synthetic artifact loads deterministically, incompatible
artifacts are rejected, and unsteered engine behavior is unchanged.

### 2. Implement numerical intervention and CLI experiments

Add session configuration and the CPU, GPU, and oracle implementations. Begin
with a separate GPU operation where it simplifies numerical validation; fuse
it into the FFN completion kernel only if measurement justifies that change.
Keep vectors resident during inference and avoid per-token file I/O or GPU-host
round trips.

Expose explicit artifact and strength options for `run`. Validate configuration
before evaluation. Add small-vector mathematical tests, CPU/GPU comparisons,
batch-tail coverage, layer-mask checks, and no-op comparisons with the baseline.

Exit condition: controlled synthetic interventions agree across implementations,
including prefill/decode transitions, and disabled steering preserves the prior
path. The feature remains experimental until behavioral evaluation passes.

### 3. Build vectors locally

Add a native capture mode that loads the model once and processes many examples
with fresh session state for each example. Capture the unsteered post-layer
residual at the last token of the fully rendered prompt, including the declared
assistant-opening template. Record that convention precisely.

Add a small Python builder, following ds4's workflow, which consumes paired
JSONL examples, calculates per-layer mean differences, and writes native
artifacts. Reject unequal/missing pairs and context overflow; do not silently
truncate prompts. Use stable accumulation, explicit normalization, and recorded
method options. Dataset generation must not require a remote model or service.

Ship an initial English succinct/detailed dataset, a separate validation set for
strength/layer selection, and a held-out evaluation set. Target roughly 100
training pairs initially; this is an experimental starting point, not a quality
guarantee. Add Italian evaluation prompts for the intended usage.

Exit condition: documented local commands reproduce an artifact and its
provenance using the exact target GGUF, without changing model weights.

### 4. Add durable runtime and protocol controls

Implement steering transitions, replay scheduling, snapshot compatibility, and
artifact retention before exposing changes in persistent sessions.

Add steering capabilities to `describe`, selected configuration to inspection,
and an explicit session update operation. For the initial service interface,
select registered local artifact IDs rather than accepting arbitrary client
filesystem paths. Add equivalent interactive chat commands and explicit
configuration for ephemeral requests. Advertise supported modes and sites.

Specify wire additions in `docs/protocol.md`, including busy behavior, errors,
effective-position reporting, persistence, and the cached-logit boundary.
Determine whether protocol version negotiation needs changing from the actual
compatibility behavior; do not reuse the vision version number as a steering
capability flag.

Exit condition: A-to-B-to-off changes retain KV, survive restart, and replay
correctly after checkpoint eviction. Rewind, session switching, and temporary
requests cannot leak another session's steering state.

### 5. Evaluate and document release readiness

Run a fixed comparison of baseline, system-prompt-only control, and steering
across a small predeclared strength/layer sweep. Select settings on validation
data and report results on held-out data. Keep prompt template, reasoning mode,
sampling, and output budget controlled.

Measure response length together with task correctness, retained key facts,
instruction following, repetition, reasoning behavior, JSON/tool-call validity,
and a representative coding workload. Shorter incomplete or truncated answers
must not count as successful succinctness. Evaluate multi-turn transitions in
addition to fresh sessions. Refusal-related experiments are a later, separately
specified dataset and metric suite.

Measure prefill throughput, decode throughput, time to first token, startup
hashing cost, and memory on supported Intel hardware. Report enabled and
disabled overhead separately. Set acceptance thresholds before assessing the
candidate; do not invent performance or quality guarantees in advance.

Publish an English evaluation report containing the engine commit, exact model
and artifact identities, hardware/software environment, dataset splits, commands,
results, limitations, and selected settings. Update `README.md` with the tested
workflow and link the detailed guide.

Exit condition: the experiment demonstrates useful control without unacceptable
measured regressions, or the report explicitly records a negative result and
the implementation remains experimental without recommended production values.

## Validation and completion criteria

Use existing suites as integration gates: `make check-unit check-context`,
`make check-gpu`, and `make check-all MODEL=/path/to/model.gguf` on the required
Linux/Intel environment. Add focused steering checks to the relevant targets.
Model and GPU validation cannot be replaced by host-only checks on macOS.

Required coverage includes:

- Loader bounds, non-finite values, hash checks, and semantic compatibility.
- Analytical additive/projection cases and unchanged disabled-path results.
- CPU/GPU agreement, chunk boundaries, tail rows, and production/oracle checks.
- A-to-B-to-off history with restart, cache hit, cache miss, and missing artifacts.
- Cached-logit boundaries, rewind, rebuild, cancellation, reasoning policies,
  tools, session isolation, and ephemeral request cleanup.
- Rejection of steered visual sessions without regressing unsteered vision.
- Baseline snapshot compatibility and explicit rejection by incompatible readers.
- Reproducible held-out behavioral evaluation and target-hardware measurements.

The feature is complete only when users can generate a vector locally, apply
and inspect it through Xenolith, retain its historical effects across durable
session operations, and reproduce the documented evaluation. A loader alone is
not the intended deliverable.

## References

- [ds4 directional steering](https://github.com/antirez/ds4/blob/ee7e8405232c8971000cc7ba2b945b00bf6d20d3/dir-steering/README.md)
- [ds4 local vector builder](https://github.com/antirez/ds4/blob/ee7e8405232c8971000cc7ba2b945b00bf6d20d3/dir-steering/tools/build_direction.py)
- [Steering Llama 2 via Contrastive Activation Addition](https://arxiv.org/abs/2312.06681)
- [Refusal in Language Models Is Mediated by a Single Direction](https://arxiv.org/abs/2406.11717)

The ds4 links above are pinned to the inspected upstream commit. The builder
and integration are independent implementations; no upstream direction payloads
or model weights were incorporated.
