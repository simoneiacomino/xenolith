# Behavioral steering: next experiments

Status: English/Italian pilot completed with a visible language result: 20/20
held-out answers predominantly Italian versus 0/20 baseline, with grammar and
content defects recorded in [LANGUAGE_EVALUATION.md](LANGUAGE_EVALUATION.md).
Anti-refusal is deferred at the user's request. Further work is needed before
claiming broad quality preservation. Keep work on
`feat/activation-steering` and use the same pinned GGUF on `xenolith-ptl`.

The subsequent [ds4-style verbosity experiment](VERBOSITY_EVALUATION.md) adds
an FFN-output site and a locally extracted all-layer projection direction.
The validation setting shortens answers but loses some required information;
quality preservation remains an open requirement.

## Outcome required

Deliver one reproducible command and a locally generated artifact that visibly
changes a specified behavior on unseen questions while preserving usable
answers. A nonzero numerical edit, repeated characters, or a shorter incomplete
answer is not a successful demonstration.

The first milestone is a diagnostic language experiment: English versus Italian
answers to English questions, without a language instruction at evaluation
time. This makes the intended change easy to inspect. It is an experiment, not
a claim that this direction will work on Gemma. The active objective is a
visible, useful language-steering demonstration.

## 1. Audit extraction and the numerical path

- Confirm the actual capture token IDs, rendered assistant opening, residual
  site, layer indices and reasoning setting with a small inspectable fixture.
- Reproduce the existing full-model CPU/GPU prefill discrepancy on both the
  baseline and candidate. Find its source or establish a separate validated
  diagnostic path before attributing behavioral failure solely to the data.
  Do not relax the existing test tolerances to declare success.
- Compare contrastive captures on identical inputs through GPU prefill and a
  diagnostic harness that advances one token at a time through CPU decode.
  Compare both absolute activation error and error relative to the much smaller
  contrastive difference. Passing an intervention-kernel test alone does not
  validate the extracted signal.
- Record per-layer activation norms, direction norms, difference-to-activation
  ratios and held-out class separation. A direction that does not discriminate
  held-out examples is a poor intervention candidate.

The current capture observes only the final rendered prompt token. Its data
contrast instructions requesting a behavior, rather than actual paired
assistant completions. This is a candidate explanation for the negative result,
not an established root cause. The current many-layer application and scale
selection are additional candidates that must be tested separately.

## 2. Compare extraction recipes on a visible behavior

Start with 50 paired training topics, 10 validation topics and 20 held-out topics.
These are proposed pilot sizes, not a statistical power guarantee. Split by
topic, not just wording, and keep all splits fixed before selecting settings.

Use equivalent short English/Italian answers to the same questions. Keep
content, formatting and approximate detail matched. Have a human review the
paired answers for meaning. Extend capture to consume structured prompt and
assistant-completion fields and record the exact observed completion token
positions. Do not insert a purported assistant answer as plain user text.

Compare:

1. The existing instruction-only final-prompt-token capture as a control.
2. Teacher-forced paired assistant completions, with declared aligned positions
   or a declared pooling rule. Retain token IDs so language/tokenization
   confounds can be inspected.

The CAA paper extracts a contrast at an appended assistant answer token; its
method is not equivalent to merely changing an instruction's wording. Our new
completion experiment would be an adaptation, not an exact CAA reproduction.
Record a distinct extraction method and position in artifact provenance, and
retain compatibility with existing artifacts.

## 3. Select intervention settings systematically

- For the language experiment, begin with additive steering at one layer at a
  time. Use held-out activation separation to shortlist layers before running
  a bounded generation sweep. Do not start by editing 20 layers simultaneously.
- Calibrate scale using measured vector and activation norms. Predeclare small
  relative perturbation levels; report the resulting per-layer ratios. Include
  zero and both signs. Existing additive and projection strength values are
  not interchangeable.
- Compare the existing all-text policy with a separately implemented,
  explicitly identified generation-only policy if prompt perturbations appear
  to be a source of instability. Define whether the final prompt position is
  included: it produces the first generated token. A new policy must have
  manifest validation and replay/snapshot semantics; never silently reinterpret
  an old artifact's policy.
- Reuse one set of captures for offline layer/method experiments. Load the model
  once per generation sweep, with fresh session state for each sample.
- Select on validation data only, then lock the artifact and configuration.

For the pilot language demonstration, propose acceptance before running it:
at least 16/20 held-out answers predominantly in the target language, at least
10 more target-language answers than baseline, no degenerate loops or malformed
text, no extra length stops, and no extra failures on the six JSON/Python
diagnostics. Manually review meaning and required facts. These are pilot gates,
not broad quality guarantees. If baseline already meets the language target,
the experiment cannot demonstrate this effect and must be redesigned before
opening the held-out split.

## 4. Build the refusal experiment

Deferred. No refusal dataset or refusal experiment is part of this iteration.
The methodology below is retained only as a future reference.

First record actual baseline completions and identify the intended refusal
behavior. For an initial false-refusal experiment, use legitimate requests that
the model unnecessarily refuses, with comparable answered controls and topic
diversity. Do not assume a request triggers refusal because of its subject, and
do not use an instruction to say "I refuse" as evidence of natural refusal.

The refusal-direction paper contrasts instruction classes and selects a
direction using validation interventions at post-instruction positions. This
is a separate recipe from completion-conditioned CAA; implement and evaluate
it explicitly rather than treating all extraction methods as interchangeable.
Prompt classification alone is insufficient: measure actual generated answers.

Keep extraction, validation and final evaluation sets separate. Evaluate refusal
rate together with useful-answer rate, correctness, uncertainty, repetition,
language, JSON/code and tool behavior. A response that drops a refusal phrase
but becomes wrong, empty or incoherent fails the useful-answer criterion.
Predeclare the required improvement once the baseline contains enough measured
refusals to make that comparison meaningful.

For projection, positive strength removes the represented component; negative
strength amplifies it. Sign reversal of the vector itself does not reverse
projection. Calibrate on this model rather than transferring another engine's
strengths or directions.

## Deliverables and stopping rules

Each stage produces an English report with exact dataset/artifact/source hashes,
commands, complete outputs and failed settings. The successful demonstration
must show the same unseen questions in fresh baseline and steered sessions,
then include a separate live-toggle demonstration that intentionally retains
historical cache effects.

Stop a candidate at the first gate it fails. If neither extraction recipe yields
a usable direction, report that result and investigate capture/site/architecture
compatibility before expanding into a large blind parameter search. Keep the
model weights unchanged throughout. The ASan host limitation and the baseline
prefill failure remain separately tracked; neither is a successful test run.

## Primary references

- [Contrastive Activation Addition, method section](https://arxiv.org/html/2312.06681v4#S3)
- [Refusal direction extraction and selection](https://arxiv.org/html/2406.11717v3#S2.SS3)
- [Current Xenolith results](EVALUATION.md)
