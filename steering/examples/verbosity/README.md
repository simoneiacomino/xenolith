# Verbosity follow-up data

`ds4/` contains unmodified text examples and their MIT license from
[antirez/ds4](https://github.com/antirez/ds4), pinned at
`ee7e8405232c8971000cc7ba2b945b00bf6d20d3`. Copyright remains with the
upstream authors. `source.json` records file SHA-256 values and provenance.

`train-instructions.jsonl` pairs the 100 succinct and detailed rows without
changing their content. `coarse.jsonl` wraps the five upstream evaluation prompts
in the Xenolith runner's JSONL format. `audit-instructions.jsonl` is the first
two training pairs for optional diagnostics.

`validation-instructions.jsonl` uses the ten original Xenolith validation
questions with fixed succinct/detailed requests. The generation questions are
`../succinct-validation.jsonl`. The final fifteen questions, including five in
Italian, remain in `../succinct-eval.jsonl` and are not used to select settings.
Topics can overlap across these small sets; this is a pilot, not a broad
independent benchmark. See [the experiment protocol](../../VERBOSITY_EVALUATION.md).

`selected.json` freezes the all-layer FFN projection setting at -0.5 before
reserved generation. The fifteen-question result shows 33.8% lower median word
count and 15/15 shorter answers, with required-fact omissions. See the report
for limits and native trial commands.
