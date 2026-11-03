# ds4-style verbosity experiment

Rebased onto upstream `362af7a` on 2026-11-03. See [rebase validation](REBASE_VALIDATION.md)
for current checks and pending native validation; earlier measurements retain
their recorded pre-rebase source/executable identities.

Status: positive held-out length-control result, with failed required-fact
preservation. Scale -0.5 was locked before reserved generation. Native regression and enabled-cost measurements pass as detailed below. Work stays on `feat/activation-steering`.
Documentation and full outputs are retained in English. Anti-refusal is out of
scope.

The user subsequently switched the active experiment to the ds4 approach.
The additive screen is archived without a held-out claim. Its late-arriving
layer-14/60% result is promising on the two selection questions (median 41 words,
both complete, versus a capped baseline median of 525.5), but remains unvalidated.

## Active ds4-style phase

Use all 30 Gemma layers, projection rather than addition, unit vectors and
control-mean orthogonalization enabled. The algebra matches the inspected
upstream builder's defaults, apart from stable accumulation/float rounding.
The profile and architecture remain Gemma, not DeepSeek.

A preliminary run reused residual-site captures, explicitly marked
`post_layer_residual_v1`; its strong negative scales produced degenerate output.
This is not treated as the faithful FFN experiment. Inspection of pinned ds4
source shows the DeepSeek path projecting the shared+routed FFN output before
its residual/hyperconnection expansion. Xenolith therefore adds the explicit
`post_ffn_output_v1` site: normalized dense+routed FFN contribution, before
attention-residual addition and layer output scale. Capture and application use
this same site. Gemma's own normalization/residual architecture is retained.

The new GPU stage uses scratch FFN normalization and recomposes only rows with
an active FFN edit. Ordinary fused finish remains unchanged for the unsteered
path. Synthetic comparisons against an independent double reference pass
(maximum absolute error `9.54e-7`), including mixed-site schedules, boundaries,
zero-strength rows and guards. Real-model zero strength and FFN capture preserve
hidden states/logits byte for byte in the focused test; snapshot boundary and
restored-continuation checks pass for both sites.

Re-capture the 100 pairs at the FFN site and derive normalized/orthogonalized
directions. Screen upstream example scales `-1,-0.5,0.5,1,2` on the same two
selection questions, with a fresh baseline/system control under the new build.
Zero is represented by the baseline. Advance at most two coherent candidates
to the ten validation questions at 2048 tokens. The final fifteen questions stay
reserved until a setting is locked. Earlier additive controls are not silently
reused across the changed executable/site. Original quality gates below apply.

## Archived additive phase: inputs and intended comparison

The user requested the same approach as the English/Italian pilot and authorized
using ds4's example prompts. Training now uses the 100 corresponding rows of
`dir-steering/examples/succinct.txt` and `verbose.txt` from antirez/ds4 commit
`ee7e8405232c8971000cc7ba2b945b00bf6d20d3`. Files are copied without changing
their text under `examples/verbosity/ds4`, including the MIT license. Source
hashes and the revision are in `examples/verbosity/source.json`.

The JSONL conversion pairs each succinct row with its corresponding detailed
row. It does not generate answers or download vectors. Native Xenolith captures
all 30 layer states locally from the pinned Gemma GGUF. Capture uses the last
rendered prompt token, including the assistant opening, reasoning off and context
512. This is instruction-conditioned extraction, matching the recipe selected
in the language pilot. The initial additive phase used the original residual intervention site. The
active ds4 phase re-captures at the FFN site described above.

Each candidate is **additive**, at **one layer only**, with no control-mean
orthogonalization. Direction magnitudes are retained. This differs from ds4's
default normalized/orthogonalized projection recipe and from the prior
many-layer Xenolith experiment. The dataset also differs from the previously
authored succinctness dataset; outcomes cannot isolate a single causal factor.

## Selection protocol and acceptance targets

Ten separate validation questions are rendered in succinct/detailed forms for
activation separation. Shortlist two layers in 4–27, at least eight apart, by
validation classification accuracy and normalized margin, requiring accuracy
at least 0.8 and positive class margin. The numerical layer-selection rule is
the same as the language pilot. Classification alone is not behavioral success.

Test relative additive displacement estimates 10%, 30%, 60% and -30% at each
shortlisted layer, where strength is the requested fraction divided by the
training direction/activation norm ratio. These estimates apply to capture data,
not necessarily live generation. Both signs are included because extraction
labels alone do not establish the causal response to an intervention.

Initial generation screening uses the first two prompts from ds4's five example
evaluation prompts. These share topics with training; they are selection data,
not independent evidence. Use fresh sessions, greedy decoding, seed 1, reasoning
off and a 768-token cap. Baseline and candidates share `You are a helpful
assistant.` A separate explicit three-sentence system-prompt control is included.
If baseline is capped, its observed length is only a lower bound; screening
cannot establish full-answer length. Do not count repeated characters, missing
answers, generation errors or truncation as successful succinctness.

Advance at most two settings with coherent, shorter outputs to the original ten
validation questions, using a 2048-token cap to inspect complete answers. If
all coarse settings are capped, inspect at most two promising nondegenerate
settings at the larger budget before drawing a negative conclusion. Lock one
setting before opening the fifteen reserved prose questions, including five
Italian questions. The original six JSON/Python diagnostics are an independent
mechanical gate. No target brevity instruction is inserted into steered requests.

Pilot acceptance requires at least 15% lower median word count on the reserved
questions, more than half of paired answers shorter, no additional length stops,
generation errors or JSON/Python failures, no degenerate loops or replacement
characters, and manual review of essential facts and language. Repeated four-gram
fraction above 0.20 or identical-character runs of 32 or more rejects a candidate.
Word reduction is not a correctness score. If required facts are lost, record
that failure rather than claiming a quality-preserving verbosity control.

System-prompt-only control is reported alongside steering. A longer coefficient
sweep is not automatic if this bounded search fails. Record incomplete screening
and a negative result honestly; a good setting is not guaranteed.

The initial layer-6/26 sweep showed only a modest effect, with many capped
answers. Before larger-budget validation, add one bounded middle-layer check:
layer 14, the highest normalized separation margin among layers 12–19, at 30%,
60% and -30%. This is an exploratory extension made on selection data only;
the fifteen reserved questions remain unopened. It tests whether the initial
shortlist missed a causally useful middle layer without launching an exhaustive
layer search. The at-most-two-finalists rule remains unchanged.

## Reproducibility

Same model/host as [the language pilot](LANGUAGE_EVALUATION.md). Training and
validation captures remain in `steering/local/verbosity-captures` on
`xenolith-ptl`. These generated float banks are ignored by Git. The research
tools are `prepare_verbosity_steering.py`, `analyze_steering.py`,
`build_steering.py` and `evaluate_steering.py --configs`.

The known ASan preload limitation and baseline CPU/GPU prefill discrepancy
remain documented in the original evaluation. This follow-up adds the FFN
capture/intervention site and research tooling. Model weights remain unchanged.


## Interim selection-data results

The two original additive layers (6/26) gave little clear reduction. The explicit
middle-layer extension was more promising: at layer 14 the 60% estimate yielded
complete 28- and 54-word answers (median 41), versus a capped baseline median of
525.5 words. This late result arrived as the user requested switching to ds4;
it is archived, not silently discarded or promoted to a held-out claim.

Residual-site projection with ds4 normalization/control orthogonalization still
collapsed at negative scales. Re-capturing at the actual FFN contribution site
changed the observed outcome:

| FFN projection scale | Median words, two selection prompts | Length stops | Observation |
| --- | ---: | ---: | --- |
| Baseline | 525.5 | 2/2 | Capped; full length unknown. |
| Explicit concise system control | 65.5 | 0/2 | Complete short answers. |
| -1 | 200.5 | 0/2 | Coherent shorter answers; Markdown spacing defects. |
| -0.5 | 385.5 | 0/2 | Coherent, generally cleaner formatting. |
| +0.5 | 484 | 2/2 | Capped; not a succinctness result. |
| +1 | 538 | 2/2 | Capped; not a succinctness result. |
| +2 | 366.5 | 2/2 | Capped with elevated repetition; not a success. |

Both -1 and -0.5 passed all six JSON/Python diagnostic cases, as did baseline
and the system control. The real-model checks were extended to 600-token
prefill, covering multiple workspace tiles/chunks, zero strength and capture
isolation. A nonzero FFN snapshot/continuation check is also included.

## Complete-answer validation and frozen setting

All ten validation answers completed within 2048 output tokens:

| Configuration | Median words | Length stops | Generation errors |
| --- | ---: | ---: | ---: |
| Baseline | 549 | 0 | 0 |
| Explicit concise system control | 65 | 0 | 0 |
| FFN projection -1 | 98 | 0 | 0 |
| FFN projection -0.5 | 333.5 | 0 | 0 |

Scale -0.5 shortens all ten paired answers and reduces the median by 39.3%.
It is locked in `examples/verbosity/selected.json` before reserved generation,
with the decision recorded in `results/verbosity/validation-review.json`.
It is chosen over -1 for less formatting damage, not because it passes all
quality gates. Both omit the compiler's behavior-preservation constraint;
the condensation explanation omits the dew-point condition. The signature
explanation repeats the baseline's inaccurate generic encrypted-hash account.
Some Markdown spacing defects remain. This is an experimental demonstration
of length control, not a quality-preserving acceptance or production default.
The explicit system control remains shorter than either steering finalist.

The immutable artifact is
`e802259e2f1b4d9507381a2063a232a1c1eed0206c257c45ef16dd894ddda581`:
all 30 layers, `projection`, unit normalization, control orthogonalization,
`post_ffn_output_v1`. A negative coefficient amplifies the activation's
existing component along this axis; it is not additive displacement toward
the dataset label. The observed brevity direction is determined experimentally.

## Rebuild the locally extracted artifact

The trial below requires no Python. For research reproduction, the standard-library
builder can repeat extraction and derivation into a **new** output directory:

```sh
python3 tools/build_steering.py --xenolith ./xenolith --model "$MODEL" \
  --dataset steering/examples/verbosity/train-instructions.jsonl \
  --capture-site post_ffn_output_v1 --mode projection --orthogonalize-control \
  --model-source https://huggingface.co/unsloth/gemma-4-26B-A4B-it-qat-GGUF \
  --model-revision 7b92b5b28818151e8669af2e45e88d6086f490dd \
  --out steering/local/verbosity-rebuilt
```

No `--layers` means all 30 layers. The builder records the capture token policy,
model/template/source hashes, numeric settings and normalization in the manifest.
A different build may produce a different artifact identity; compare provenance
and payload rather than assuming filenames establish equivalence.

## Native trial

On `xenolith-ptl`, in `~/dev/xenolith-steering`, start a fresh native chat:

```sh
MODEL=~/models/gemma-4-26B-A4B-it-qat/gemma-4-26B-A4B-it-qat-UD-Q4_K_XL.gguf
./xenolith chat "$MODEL" --ctx 4096 --temp 0 --seed 1 -n 2048 \
  --steering steering/local/verbosity/manifest.json --steering-strength -0.5
```

Ask `Perché il pane lievita?`, then `/exit`. For a fresh baseline, run:

```sh
./xenolith chat "$MODEL" --ctx 4096 --temp 0 --seed 1 -n 2048
```

These commands use the native binary and insert no brevity instruction.
Use `chat` for the instruction model's conversational template. Text-only
`run -p` is raw completion (BOS plus literal text), not chat rendering; our raw
baseline smoke degenerated into repeated punctuation and is not a successful
verbosity comparison. That output is archived as a failed interface trial.
The research harness uses a fixed helpful-assistant system message; native
chat uses its ordinary system rendering. Exact word counts can differ. Start
fresh processes for clean comparisons; live changes retain existing KV effects.


## Reserved fifteen-question result

Fresh ephemeral contexts, greedy decoding, seed 1, reasoning off, context 4096,
2048 output-token cap. The fixed setting was not changed after this run.

| Configuration | Median words | Shorter than paired baseline | Length stops | Errors |
| --- | ---: | ---: | ---: | ---: |
| Baseline | 529 | — | 0/15 | 0/15 |
| Explicit concise system control | 64 | 15/15 | 0/15 | 0/15 |
| FFN projection -0.5 | 350 | 15/15 | 0/15 | 0/15 |

Steering reduces the median by **33.8%**. All five Italian requests remain in
Italian in both baseline and steering. Maximum repeated four-gram fraction is
0.0464, longest identical-character run is 4, and no replacement characters
occur. The six separate JSON/Python mechanical diagnostics pass 6/6 for all
four validation configurations; Python checks syntax/function presence, not
functional correctness of arbitrary programs.

The mechanical brevity targets pass. **Required-fact preservation fails.**
For example, the hash answer loses the trusted-reference caveat present in the
baseline. The transaction answer incorrectly restricts ARIES redo to committed
transactions; the baseline's ARIES section includes operations of subsequently
failing transactions. Other answers retain the main idea but contain
oversimplifications or Markdown defects. Review decisions and complete answers
are retained in `results/verbosity/heldout-review.json` and
`results/verbosity/ds4-ffn-heldout/responses.jsonl`.

An easily visible Italian example is “Perché il pane lievita?”: **323 to 122
words**, with yeast fermentation, carbon dioxide and gas retention preserved.
The explicit system prompt is still much shorter than steering. No claim is
made that steering is preferable to prompting, generally improves factuality,
or supplies a production-ready verbosity knob. No further setting is selected
on these reserved answers.

## Native regression and enabled cost

The final serial native run passes:

```sh
make -o check-kv-load-bounds check-unit check-context check-steering \
  check-steering-model check-runtime check-serve MODEL="$MODEL"
python3 tests/check_steering_capture.py --model "$MODEL"
```

The `-o` omission is deliberate: the previously diagnosed ASan preload gate
cannot start under the host monitoring preload. It is not a clean `check-all`.
The existing full-model CPU/GPU prefill discrepancy, reproduced on the unchanged
baseline, remains unresolved; its tolerances were not relaxed.

New coverage includes FFN GPU comparison with an independent double reference
(max error 9.54e-7), mixed residual/FFN/off/zero schedules, masks and guard rows,
600-token input in a 1024-row batch crossing the 512-row FFN workspace tile,
byte-exact zero/capture isolation, and nonzero snapshot restore/continuation.
Existing unit/context/runtime/server checks pass under the command above.
Full output is retained in `results/verbosity/xenolith-ffn-final-regression.log`.

Enabled versus disabled cost is measured on the same new executable, without
other inference jobs, using fixed token IDs (512 prefill + 128 decode), one
warm-up per setting, and three ABBA rounds (six samples per setting):

```sh
make bench/bench_steering
./bench/bench_steering "$MODEL" steering/local/verbosity/manifest.json -0.5 3
```

| Median elapsed time | Disabled | FFN -0.5 | Increase |
| --- | ---: | ---: | ---: |
| Prefill, 512 tokens | 1.10125 s | 1.10192 s | 0.06% |
| Decode, 128 tokens | 4.58387 s | 4.62387 s | 0.87% |

This is within the observed timing spread and below the original 10% enabled
throughput-cost target for this workload. It is not a new original-baseline
versus candidate disabled-path comparison, or a guarantee for other contexts,
layers or hardware. Earlier off-path measurements predate the FFN extension.
Raw timings and summary are retained under `results/verbosity/`.
Artifact and executable identities, input hashes, and report summaries are in
`results/verbosity/provenance.json`. The immutable manifest retains its original
unevaluated extraction label; the evaluation result lives outside the artifact
so that its content identity does not change.

## Verified native chat smoke

The exact chat commands above were tested with “Perché il pane lievita?”.
The baseline generated 725 tokens (367 words); FFN -0.5 generated 277
tokens (136 words). Both retain fermentation, carbon dioxide and gluten gas
retention in coherent Italian. This is a usability smoke, not another reserved
benchmark or a reason to change the frozen setting. Full stdout/stderr and
`native-chat-summary.json` are retained under `results/verbosity/`. The earlier
raw-completion baseline degenerated; its steered process was interrupted and
is not counted as an evaluation result.
