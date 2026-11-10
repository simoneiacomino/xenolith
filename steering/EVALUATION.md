# Steering validation and evaluation

Rebased onto upstream `362af7a` on 2026-11-03. See [rebase validation](REBASE_VALIDATION.md)
for current checks and pending native validation; earlier measurements retain
their recorded pre-rebase source/executable identities.

Status: experimental implementation, without a recommended production strength.
Local extraction, loading, inference and durable session controls are
implemented. The bundled succinctness direction failed validation; this is a
negative behavioral result, not a claim of useful style control.
See [NEXT_STEPS.md](NEXT_STEPS.md) for the proposed extraction audit, visible
behavior experiment and subsequent refusal evaluation.
The subsequent [English/Italian pilot](LANGUAGE_EVALUATION.md) is evaluated
separately; its results do not change this negative succinctness result.

## Reproducibility

- Date: 2026-10-24.
- Base source commit: `82ae168ce5fcba325af248865aa0dcb7690ff898`.
- Validation host: `xenolith-ptl`, native Ubuntu 26.04.1 LTS.
- CPU: Intel Core Ultra X7 358H, 16 logical CPUs, 64 GB installed RAM.
- GPU: Intel integrated GPU device `8086:b080`, existing non-XMX kernels.
- Kernel: `7.0.0-38-generic`.
- Compiler: GCC 15.2.0, `-O3 -march=native -std=c11 -Wall -Wextra -pthread`.
- Intel GPU runtime packages: `intel-opencl-icd` and `libze-intel-gpu1`
  `26.05.37020.3-1`.
- Exact model SHA-256:
  `a7c5bc715f5ff8e99a3e8901ce7d2b42b402c669bf24f7c5250747633d0f5891`.
  This matches the reference GGUF in `tests/vision/manifest.json`.

The source changes are a working-tree implementation on the base commit. Capture
metadata records its numerical source/flags hashes. Evaluations additionally
record executable SHA-256; the source commit alone does not identify this build.
The dedicated implementation branch is `feat/activation-steering` on both the
development and validation machines.

## Local extraction

The production-path capture completed all 100 pairs from
`examples/succinct-train.jsonl`, with context 512 and reasoning disabled. There
was no remote inference, backpropagation, weight edit, or downloaded direction.
All three artifacts use zero-based layers 8 through 27.

| Artifact | ID |
| --- | --- |
| Projection, per-layer unit norm | `da7746a72c5bcbc93697e2eb2c7c86d6e781787e2288ab4c55d3d5dfb4dbfb9a` |
| Additive, original mean-difference magnitude | `3ee4f37f13d7164b002f7fe40f8758402faaa03277c7937fd62e5d844ec8d3b3` |
| Projection, control-mean orthogonalization | `cd811e6ac529794375b71130b87c34cdcf437461051e60dc1d2e1aee8b157087` |

Capture dataset SHA-256 is
`5fd7d55083a34d5902c741954169fff80ce65bb988568c3c5495207b5da86c53`;
chat-template SHA-256 is
`845f1ee48e39fc942fe190da9df6a1c5db229e17a96ea08966ad1c9274e73d1b`.
The artifact manifests identify the source model by its local SHA-256 revision;
the repository also pins its distribution revision in the vision manifest.

A runtime bank contains 84,480 float32 values, or 337,920 bytes. Loading also
retains host data/metadata and a resident shared-memory bank for each distinct
nonzero artifact in a session's history. Additional transitions sharing one
artifact reuse that resident bank. Historical artifacts remain required for
replay after disabling steering.

## Numerical and persistence checks

| Check | Observed result |
| --- | --- |
| Portable artifact validation, import/identity and analytical operations | PASS on macOS ARM and native Linux. |
| Builder mean difference, normalization, masks and dataset identity | PASS. |
| Native capture final line, invalid final pair and context overflow without a trailing newline | PASS; failures do not publish completion metadata. |
| New loader/analytical tests under UBSan | PASS on macOS ARM. |
| Synthetic GPU additive/projection, masks, repeated-position boundaries, tails and guard rows | PASS; maximum absolute error `2.38e-7` against a double-precision reference. |
| Real-model zero strength | Byte-identical hidden states and logits across prefill and single-token decode. |
| Cached-logit boundary and pending-change snapshot | Existing logits unchanged; pending transition produces byte-identical historical snapshot. |
| Different steering history at snapshot load | Explicit numeric mismatch before changing destination state. |
| Matching steered snapshot continuation | Byte-identical continuation after restore in the tested CPU decode path. |
| Frozen context, reasoning retention, restart and rewind | PASS in the focused conversation test. |
| Real-model A-to-B-to-off, snapshot resume, eviction replay, rebuild and session isolation | PASS. |
| Tool calling, busy rejection, cancellation, and missing artifacts on restart | PASS in the extended integration run. |
| Generated reasoning with an existing thinking prefix and additive 0.01 | PASS; 64 reasoning tokens retained in visible history across restart despite discard policy. This is a persistence test, not a reasoning-quality result. |
| Mixed visual/steering input rejection | PASS before decoding or transcript mutation. |
| Interactive chat load, disable and re-enable | PASS in a native CLI smoke run. |

These checks validate the implementation contract. They do not establish that a
particular learned direction improves factuality, reasoning, or task success.

Reasoning settings cannot rewrite the frozen system prefix. A probe that enabled
thinking only after freezing a non-thinking prefix returned an immediately closed
thought channel. Tests that require generated reasoning must first establish the
global thinking marker, then enable steering. This is distinct from retaining
already generated reasoning, which the history tests check even in discard mode.

## Existing regression suites

The default `make check-all` is blocked at the existing ASan bounds prerequisite:
`ASan runtime does not come first in initial library list`. This host preloads
`/usr/lib/x86_64-linux-gnu/liboneagentproc.so` through `/etc/ld.so.preload`.
Static-ASan and explicit-preload attempts also terminated before useful test
output. No host monitoring settings or sanitizer pass/fail assertions were
changed. The independent UBSan and ordinary checks are reported above.

To collect unaffected suites, each leaf target was invoked separately with
`-o check-kv-load-bounds`. That explicitly omits the blocked prerequisite; it is
not a successful ASan run and is not reported as a clean `make check-all`.

| Suite | Result |
| --- | --- |
| Host/unit checks, excluding the ASan bounds prerequisite | PASS. |
| CLI context validation | PASS, 175 cases. |
| GPU checks, excluding the ASan bounds prerequisite | PASS. |
| Model persistence, including the new steering snapshot test | PASS. |
| Full-model prefill numerical suite | FAIL; the unchanged baseline reproduces exactly the same numerical errors. |
| Runtime and server suites | PASS. |

The unchanged baseline and candidate both report, for the first M32 prefill
comparison: hidden relative RMS `0.05982`, logits `0.1258`, K `0.08641`,
V `0.1151`, and decode-handoff logits `0.1272`; both have zero greedy mismatches
in the tested eight-token continuation. Both fail the existing numerical
thresholds. No thresholds were changed. Timings from those regression runs are
not used as a performance comparison because other build/test work overlapped.

## Behavioral evaluation protocol

The predeclared screen compares baseline, a concise system prompt, and projection
strengths `-0.5,-0.25,0.25,0.5` on the 10 validation prompts. Decoding is greedy,
seed 1, reasoning disabled, with fresh contexts and a 256-token output cap.
Additive strengths are calibrated separately. Evaluate chosen settings on the
15 held-out prose prompts (including Italian) and the separate JSON/Python
constraint battery. Preserve complete outputs for manual review.

Screening targets are at least 15% lower median word count, no additional
length-limited completions or mechanical constraint failures, and a repeated
4-gram fraction below 0.20. These targets do not replace factual/code review and
are too small a sample for broad statistical claims. Length alone is not success.

Latency, runtime prefill/decode throughput and observed process memory are
recorded alongside output text. Startup hashing is reported separately. Compare
the original and candidate binaries without steering for baseline overhead;
measure enabled steering separately on the same machine without concurrent
inference workloads. Performance targets are under 5% disabled-path regression
and under 10% enabled-path throughput cost in the selected workload; observed
variance and failures must be reported rather than hidden by changing thresholds.

### First screen: plain mean-difference projection

All 60 validation generations are archived in
[`results/validation-projection`](results/validation-projection). The 256-token
cap was too low for the unsteered model: all 10 baseline answers were truncated,
so this screen cannot estimate its complete-answer length. The initial concise
system prompt was also too weak (9/10 truncated). Later probes use an explicit
three-sentence control and report their different settings separately.

| Configuration | Median words | Length stops / 10 | Observation |
| --- | ---: | ---: | --- |
| Baseline | 182 | 10 | Coherent but capped explanations. |
| Initial concise system prompt | 174.5 | 9 | Insufficient control at this output cap. |
| Projection -0.5 | 12.5 | 10 | Degenerate character/numeric loops; rejected. |
| Projection -0.25 | 180.5 | 10 | Little useful reduction; answers remain capped. |
| Projection +0.25 | 185 | 10 | Capped answers; no succinctness result. |
| Projection +0.5 | 188 | 10 | Capped answers; no succinctness result. |

The apparent 93% reduction at -0.5 is not successful steering: token consumption
remained at the cap while word boundaries disappeared into repeated characters.
This is why whole responses and completion status are retained. All 10 outputs
at -0.5 contain character runs of at least 32 identical characters. Word-level
4-gram repetition alone missed that failure, so later probes also measure the
longest repeated-character run. This diagnostic was added after inspecting the
first screen; it was not a predeclared metric of that screen.

A follow-up extraction removes the component parallel to the negative-class
activation mean before normalization, following the inspected ds4 recipe. The
builder exposes this as `--orthogonalize-control` and records a distinct method
in artifact provenance. Its synthetic arithmetic test passes.

During the orthogonalized probe, byte-token output exposed an existing transport
problem: a text delta could end inside a UTF-8 character, producing an invalid
JSON frame. The service now buffers incomplete text/reasoning characters
separately and the JSON writer replaces malformed sequences with U+FFFD.
Focused tests cover split characters, channel isolation and terminal flushing.
The interrupted probe is retained as a failed run; subsequent measurements use
the corrected transport. Generation errors are recorded as failed samples,
never counted as successful concise answers.

### Follow-up projection and additive probes

The orthogonalized projection probe used the first five validation prompts,
strengths `-1,-0.5,0.5,1`, and a 384-token cap. The additive probe independently
used the first three prompts, strengths `-1,0.5,1,2`, and the same cap. These are
explicitly limited exploratory probes, not full validation or held-out results.
Both used the stronger system control: at most three sentences, essential facts,
no headings, lists or preamble. Full outputs and executable identities are in
[`results/validation-orthogonal`](results/validation-orthogonal) and
[`results/validation-additive`](results/validation-additive).

| Configuration | Median words | Length stops | Observation |
| --- | ---: | ---: | --- |
| Orthogonal probe baseline | 273 | 5/5 | Coherent, capped. |
| Orthogonal probe system control | 67 | 0/5 | Concise, complete answers in this small probe. |
| Orthogonal projection -1 | 3 | 5/5 | Repeated/invalid byte output; up to 383 replacement characters. |
| Orthogonal projection -0.5 | 2 | 5/5 | Repeated characters; longest run 321. |
| Orthogonal projection +0.5 | 277 | 5/5 | No useful reduction. |
| Orthogonal projection +1 | 289 | 5/5 | No useful reduction. |
| Additive probe baseline | 262 | 3/3 | Coherent, capped. |
| Additive probe system control | 67 | 0/3 | Concise answers. |
| Additive -1 | 384 | 3/3 | Repeated words; maximum repeated 4-gram fraction 0.997. |
| Additive +0.5 | 1 | 3/3 | Single-character loops, up to 384 identical characters. |
| Additive +1 | 1 | 3/3 | Single-character loops. |
| Additive +2 | 384 | 3/3 | Repeated words; maximum repeated 4-gram fraction 0.958. |

Manual inspection confirmed that the very short steering outputs were damaged
answers, not successful compression. The stronger system control was much more
effective on these prompts. None of these directions/settings passes the
screen, so no production strength is recommended. The 15 held-out prose prompts,
including Italian, remain unused: there was no successful candidate to submit
to a held-out comparison. This preserves that split for a future extraction
experiment. It also means there is no measured Italian quality claim.

The dataset, position, site and layers are one starting recipe. A negative result
does not establish that all activation steering on Gemma fails, and the
implementation does not automatically search for a successful direction.

### JSON, coding and tool diagnostics

The separate six-prompt constraint battery was run at raw projection strength
`-0.25` as a mild, nondegenerate diagnostic, not as a successful succinctness
candidate. Each request had a 512-token cap and the same greedy/reasoning-off
settings. Outputs are archived in [`results/constraints`](results/constraints).

| Configuration | Exact JSON cases | Python definition cases | Length stops |
| --- | ---: | ---: | ---: |
| Baseline | 3/3 | 3/3 | 0/6 |
| Three-sentence system control | 3/3 | 3/3 | 0/6 |
| Raw projection -0.25 | 3/3 | 2/3 | 0/6 |

The failed steered coding answer was only `max(low, min(n, high))`, omitting the
requested `def clamp(n, low, high):` wrapper. All other short definitions were
manually reviewed and express the requested arithmetic. The automated Python
check parses syntax and the required function name; it does not execute generated
code or establish broader software correctness. This extra failure independently
violates the no-additional-constraint-failures target. More substantial coding
and Italian quality evaluations remain future work with a direction that first
passes validation.

Median first-text latency on this small battery was 0.620 seconds for baseline,
0.730 for the system control, and 0.622 for steering. These observed latencies
exclude server startup and are not a service-level latency guarantee.
The integration runner separately checked an `add(2,3)` tool call under baseline
and projection -0.25, including the exact JSON arguments; both passed.

## Fixed-workload performance

The enabled comparison uses the raw projection artifact at strength `-0.25`,
layers 8–27, with 512 fixed prompt tokens and 128 fixed decode tokens. Both
configurations are warmed; three off/on/on/off groups give six measurements per
configuration. No other inference job or compilation ran concurrently.

| Configuration | Median prefill tokens/s | Median decode tokens/s |
| --- | ---: | ---: |
| Same binary, off | 460.44 | 27.79 |
| Projection enabled | 472.70 | 28.14 |

Measured throughput changes are +2.66% and +1.25%, respectively. These small
positive differences are timing variation, not evidence that steering speeds up
inference. Prefill ranges overlap (off 448.89–474.63, on 451.02–476.64 tokens/s),
as do decode ranges (off 27.21–28.22, on 27.18–28.22). There is no observed cost
above the 10% target in this workload. This does not cover every layer count,
context size, mode or device. Raw measurements are
[`results/xenolith-steering-performance-enabled.jsonl`](results/xenolith-steering-performance-enabled.jsonl).

Full model hashing took 5.37 seconds in that run; engine open including hashing
and setup took 9.40 seconds. Hashing is a startup cost when steering is enabled,
including with disk caching disabled. It is not included in the warmed token
throughput figures. Model pages may already be in the filesystem cache; these
observations are not a cold-storage startup guarantee.

For the disabled-path comparison, an unchanged export of base commit
`82ae168ce5fcba325af248865aa0dcb7690ff898` was compiled on the same machine.
The paired harness ran baseline/candidate/candidate/baseline separately at each
context capacity, with three timed repetitions per process after warm-up:
six samples per variant and phase. Prefill uses 2,048 prompt tokens in 512-row
batches; decode uses the harness's 128-token workload. Context capacity is not
the number of populated attention positions in these tests.

| Context capacity | Phase | Original tokens/s | Candidate off tokens/s | Change |
| --- | --- | ---: | ---: | ---: |
| 65,536 | Prefill 2,048 | 419.17 | 419.23 | +0.01% |
| 65,536 | Decode 128 | 31.07 | 31.21 | +0.43% |
| 131,072 | Prefill 2,048 | 417.69 | 417.82 | +0.03% |
| 131,072 | Decode 128 | 30.91 | 30.90 | -0.04% |

No disabled regression above the 5% target was observed in this workload.
Raw samples, executable SHA-256 values, exact commands and completion status
are archived in [`results/performance-off`](results/performance-off).
The subsequent final-line capture validation fix and explicit chat-disable error
reporting change CLI error handling in `main.c`; inference code and kernels are
unchanged from the measured build.
The measured candidate is retained on the host as
`/tmp/xenolith-steering-performance-candidate`. Source hashes for the initial
delivery are in [`results/implementation-sources.json`](results/implementation-sources.json).
The later language pilot has a separate source/provenance record.

On the additive probe, `/proc/PID/status` reported baseline RSS 184,768 KiB and
steered RSS 185,428–185,444 KiB, about 660–676 KiB higher. These are process RSS
observations, **not total model/GPU memory**: Level Zero allocations are not
fully represented there. Each distinct historical artifact additionally requires
a 337,920-byte GPU shared-memory vector bank, alongside the retained host rows,
payload and manifest. This implementation does not duplicate the model weights.

## Reproduce the native runs

From the validation checkout, set these paths (adjust them on another host):

```sh
MODEL=/home/devcloud/models/gemma-4-26B-A4B-it-qat/gemma-4-26B-A4B-it-qat-UD-Q4_K_XL.gguf
REGISTRY=/home/devcloud/dev/xenolith-steering-artifacts
PROJECTION=da7746a72c5bcbc93697e2eb2c7c86d6e781787e2288ab4c55d3d5dfb4dbfb9a
ADDITIVE=3ee4f37f13d7164b002f7fe40f8758402faaa03277c7937fd62e5d844ec8d3b3
ORTHOGONAL=cd811e6ac529794375b71130b87c34cdcf437461051e60dc1d2e1aee8b157087

make check-steering check-steering-model MODEL="$MODEL"
python3 tests/check_steering_capture.py --model "$MODEL"
python3 tests/check_steering_session.py --model "$MODEL" \
  --registry "$REGISTRY" --projection "$PROJECTION" --additive "$ADDITIVE" \
  --out /tmp/steering-session-new

python3 tools/evaluate_steering.py --model "$MODEL" --registry "$REGISTRY" \
  --artifact "$ORTHOGONAL" --prompts steering/examples/succinct-validation.jsonl \
  --limit 5 --strengths=-1,-0.5,0.5,1 --tokens 384 \
  --out /tmp/steering-orthogonal-new
python3 tools/evaluate_steering.py --model "$MODEL" --registry "$REGISTRY" \
  --artifact "$ADDITIVE" --prompts steering/examples/succinct-validation.jsonl \
  --limit 3 --strengths=-1,0.5,1,2 --tokens 384 \
  --out /tmp/steering-additive-new
python3 tools/evaluate_steering.py --model "$MODEL" --registry "$REGISTRY" \
  --artifact "$PROJECTION" --prompts steering/examples/constraints-eval.jsonl \
  --strengths=-0.25 --tokens 512 --out /tmp/steering-constraints-new

make bench/bench_steering
./bench/bench_steering "$MODEL" "$REGISTRY/$PROJECTION/manifest.json" -0.25 3
python3 bench/text_regression.py --model "$MODEL" --candidate ./xenolith \
  --baseline /tmp/xenolith-steering-baseline.gqxtgD/xenolith --paired \
  --repetitions 3 --output /tmp/steering-disabled-new
```

The archived first projection screen used the earlier, weaker system-control
wording and a 256-token cap; use its recorded arguments and outputs when
interpreting it. Re-running the current evaluator uses the stronger control.
