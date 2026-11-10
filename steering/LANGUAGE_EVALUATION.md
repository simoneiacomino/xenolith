# English-to-Italian steering pilot

Rebased onto upstream `362af7a` on 2026-11-03. See [rebase validation](REBASE_VALIDATION.md)
for current checks and pending native validation; earlier measurements retain
their recorded pre-rebase source/executable identities.

Status: positive held-out language result and verified native CLI demonstration,
with observed language/content imperfections. This is an experimental pilot,
not a general quality or factuality guarantee. Anti-refusal is deferred.

## Try the selected artifact

On the validation host, the artifact and model are already installed:

```sh
ssh xenolith-ptl
cd ~/dev/xenolith-steering
python3 tools/try_language_steering.py
```

Ask a question in English, for example `How does a parachute slow a fall?`.
The launcher uses the native interactive chat with the selected vector. For a
fresh unsteered comparison, exit with `/exit` and run:

```sh
python3 tools/try_language_steering.py --baseline
```

Inside the chat, `/steer off` disables new interventions and
`/steer 0.586135537` enables the pilot setting again. Existing cache/text effects
remain present after a toggle. Use separate fresh chats for controlled
baseline/steered comparisons. `--model` and `--manifest` override the launcher's
default paths on other hosts. The pilot artifact is generated locally and lives
in the ignored directory `steering/local/italian/`; it is not a downloaded vector.

## Frozen setting and scope

- Branch: `feat/activation-steering`, working tree based on
  `82ae168ce5fcba325af248865aa0dcb7690ff898`.
- Same native Intel host and pinned QAT GGUF as [the original evaluation](EVALUATION.md).
- Artifact ID: `f866d1f97d1ccd77262810c57db431a90d59a2329c4dc515c6a27604daaeb8dc`.
- Additive, zero-based layer 16 only (the seventeenth layer),
  `post_layer_residual_v1`, all processed text positions.
- Strength: `0.5861355370624906`.
- Positive class: Italian; negative class: English.
- Extraction: mean difference at the final rendered instruction token, including
  the non-thinking assistant opening. No completion text is used in this winning
  recipe; the completion-conditioned recipe is a separately evaluated comparison.
- Calibration: approximately 30% of the median captured activation norm at that
  layer. This is an estimate on capture data, not a fixed percentage of live
  activations or output words.
- `examples/language/selected.json` was frozen after validation and before the
  held-out generation. No setting is selected on the final twenty questions.

Source, dataset, selected-setting and executable identities are recorded in
[`results/language/provenance.json`](results/language/provenance.json).

The model weights remain unchanged. The intervention adds a displacement to one
layer's residual state; it does not insert an Italian instruction into the user
message or system prompt during steered evaluation.

## Data and extraction audit

The authored dataset contains 50 training topics, 10 validation topics and 20
held-out topics. Two extraction recipes share the training topics:

1. Instructions requesting an Italian versus English answer.
2. Equivalent teacher-forced Italian versus English assistant answers to the
   same English question, averaging captured states over answer tokens.

The second recipe required new `steering-capture --completions` support. The
native extractor records all token IDs, prefix length, exact observed token
range and final prefix token piece in `*.tokens.json`. The inspected opening
ends in `<channel|>`; completion capture begins at the next position. Capture
metadata distinguishes recipes and records an extractor source hash.
`--cpu` advances the entire prefix one token at a time for diagnosis. Existing
instruction datasets, manifests and builders remain supported.

Four training topics were captured through both CPU tokenwise execution and
production GPU prefill. Across layers and these examples:

| Recipe | Maximum activation relative L2 difference | Maximum contrast relative L2 difference | Minimum contrast cosine |
| --- | ---: | ---: | ---: |
| Instruction final token | 0.0634 | 0.1300 | 0.9916 |
| Mean completion state | 0.0532 | 0.0705 | 0.9975 |

The contrast directions are similar in this diagnostic. This does **not**
resolve the pre-existing full-model prefill numerical test failure or certify
all backend inputs. Existing thresholds were not relaxed. Audit JSON files are
in [`results/language/audit`](results/language/audit).

Layer selection uses held-out activation separation, with two distinct single
layers per recipe and at least eight layers between them. The shortlisted
layers were 26/5 for completion means and 5/16 for instructions. All classified
the ten validation capture pairs correctly. Classification is a diagnostic,
not evidence by itself that an intervention controls generation.

## Generation selection and validation

All generation comparisons use fresh ephemeral sessions, greedy decoding,
seed 1, reasoning off, context 4096 and a 384-token cap. Baseline and steering
share exactly this system prompt:

```text
You are a helpful assistant. Answer the question briefly in two or three sentences.
```

The separate system-prompt control appends `Answer in Italian.`. An English
question is the user message in every arm. The launcher uses the native chat's
ordinary template instead; its smoke run is a separate usability check.

The first sweep tested estimated relative displacements 1%, 3%, 10% and -10%
at the four selected recipe/layer combinations. None produced a consistently
Italian answer on the first three validation questions. One instruction-layer
setting produced a mixed English/Spanish answer. Spanish markers were added to
the screening language heuristic; such text is not counted as Italian.

A documented second tier tested 30% and 60% at the same layers. Several settings
produced Italian answers. Two finalists advanced to all ten validation topics:

| Setting | Predominantly Italian | Length stops | Degenerate loops / errors |
| --- | ---: | ---: | ---: |
| Baseline | 0/10 | 0 | 0 |
| Italian system control | 10/10 | 0 | 0 |
| Completion recipe, layer 5, 30% estimate | 10/10 | 0 | 0 |
| Instruction recipe, layer 16, 30% estimate | 10/10 | 0 | 0 |

Output review favored instruction/layer 16 for generally cleaner wording.
Imperfections remain: both finalists used the English word `purpose` in one
otherwise Italian answer; the completion recipe also used `Le metallo`.
Predominantly Italian does not mean flawless translation or grammar. The winning
recipe is still instruction-only, so this experiment does not show that adding
completion capture caused the improvement. Task, layer count and calibration
also differ from the failed succinctness experiment.

The selected setting then passed the mechanical JSON/Python diagnostics:

| Setting | Exact JSON | Python definitions | Length stops |
| --- | ---: | ---: | ---: |
| Baseline | 3/3 | 3/3 | 0/6 |
| Italian system control | 2/3 | 3/3 | 0/6 |
| Selected steering | 3/3 | 3/3 | 0/6 |

The system control's failed answer wrapped the requested JSON in Italian prose.
This is one small diagnostic, not a broad advantage over system prompting. The
Python check is syntax/function-name validation, with output review; it does not
execute generated code. Complete outputs for every stage are retained under
[`results/language`](results/language).

## Locked held-out result and quality limits

The frozen setting was evaluated once on the twenty reserved questions:

| Setting | Predominantly Italian | English | Length stops | Errors / loops |
| --- | ---: | ---: | ---: | ---: |
| Baseline | 0/20 | 20/20 | 0 | 0 |
| Italian system control | 20/20 | 0/20 | 0 | 0 |
| Selected steering | 20/20 | 0/20 | 0 | 0 |

This passes the predeclared language-count, truncation, degeneration and
mechanical diagnostic gates. Output review confirms predominantly Italian in
all twenty steered answers. It does **not** establish unchanged factual accuracy:
the outputs contain wording/content defects. Examples include `L'iron ruggine`,
`I crawler ... scansion` and `L'abstract serve a come`. The thermostat answer
confuses the desired setpoint with the sensed temperature, and the zipper answer
describes interlocking as permanent. The orbital answer also uses misleading
gravity-compensation wording. These are recorded in `heldout/manual-review.json`;
no configuration was retuned on these observed failures.

Consequently, the deliverable is a visible language-steering demonstration with
quality limits. It should not be presented as a production translation setting
or evidence that steering preserves every capability. System prompting already
achieves the language target on this dataset. The experiment demonstrates that
the activation intervention can change the language without a target-language
instruction at evaluation time.

The native launcher was separately checked with the ordinary chat template:
the metal-spoon question starts in Italian with steering, and in English in a
fresh zero-strength chat. In that same zero-strength conversation, enabling
`/steer 0.586135537` before the parachute question changes the new response to
Italian while keeping prior context. Those CLI smoke runs used a 192-token
budget to inspect the effect and were truncated; they are not the complete-answer
held-out measurements above. The launcher defaults to 512 output tokens.

The new native capture tests pass, including teacher-forced token positions and
CPU provenance. Portable builder/separation/language checks pass. The standard
ASan preload limitation and pre-existing full-model prefill numerical failure
from the original report remain unresolved and are not reported as passes.
The native `check-unit check-context` rerun passed with the blocked ASan bounds
prerequisite explicitly omitted (`make -o check-kv-load-bounds ...`), including
175 CLI cases. This is not a clean `check-all` result. Inference kernels and
session semantics were unchanged during this follow-up; new code covers capture,
research tools and the launcher. Complete diagnostic logs are retained in
[`results/language/checks`](results/language/checks).

## Reproduction

To reproduce the winning vector from the retained native captures:

```sh
python3 tools/build_steering.py \
  --capture-dir steering/local/language-captures/xenolith-language-train-instructions \
  --dataset steering/examples/language/train-instructions.jsonl \
  --out steering/local/italian-rebuilt --mode additive --layers 16 \
  --positive-label italian --negative-label english \
  --model-source https://huggingface.co/unsloth/gemma-4-26B-A4B-it-qat-GGUF \
  --model-revision 7b92b5b28818151e8669af2e45e88d6086f490dd
```

To collect fresh captures, use the same dataset with `xenolith steering-capture`
and the exact GGUF. For the comparison recipe, use `train-completions.jsonl` and
`--completions`. Re-run validation capture on its separate ten topics and use
`tools/analyze_steering.py` to produce separation reports. The bounded sweep
builder is `tools/prepare_language_steering.py`; generation and archives use
`tools/evaluate_language_steering.py`. Each tool has `--help`.

The language detector uses stopword markers only. Its labels require output
review and do not measure factuality. The initial split sizes and pilot gates
were declared before generation in [the dataset notes](examples/language/README.md).
This pilot does not establish behavior on arbitrary conversations, reasoning,
long contexts, vision or other GGUFs, and it makes no anti-refusal claim.
