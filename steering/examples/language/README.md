# English / Italian diagnostic dataset

Authored, topic-separated pilot: 50 training pairs, 10 validation pairs and 20
held-out questions. No downloaded vectors or remote model responses are used.
Positive answers are Italian; negative answers are equivalent English answers.
The paired completions are short factual examples, reviewed during authoring;
there has not been an independent human annotation study.

`*-instructions.jsonl` controls capture only a language request. The completion
recipe uses `*-completions.jsonl` with a shared English `prompt` and separately
teacher-forced assistant answers. `audit-*` are the first four training topics
and are used only to compare CPU and GPU capture paths. Held-out topics are not
used for extraction, layer selection or strength calibration.

Generation uses greedy decoding, seed 1, reasoning off, a fresh context, and the
system instruction `You are a helpful assistant. Answer the question briefly in
two or three sentences.` There is no target-language instruction in baseline or
steered evaluation. A separate Italian system-prompt control is included.

Before generation screening, shortlist two distinct single layers per recipe
from validation activation separation, with at least eight layers between them
and indices in 4–27. Rank by classification accuracy, then margin divided by
direction norm. Test additive displacements estimated at 1%, 3% and 10% of the
training activation norm, plus the negative 10% control. These ratios describe
the captured states and do not guarantee identical ratios during generation.
Run the first three validation questions initially; submit at most two promising
configurations to full ten-question validation before locking one candidate.

Pilot success requires at least 16/20 held-out answers predominantly in Italian,
at least ten more than baseline, no degenerate loops/malformed text, no extra
length stops and no extra failures on the existing JSON/Python diagnostic set.
Automatic stopword language labels are only a screening proxy. Complete outputs
must also be reviewed for language, meaning and required facts. Reject repeated
four-gram fractions over 0.20, identical-character runs of 32 or more, replacements
and generation errors. The held-out split is opened only after configuration
selection. Anti-refusal is deferred for this iteration.

The initial 1–10% sweep did not produce Italian answers. A bounded second tier
uses 30% and 60% estimated displacements at the same layers: this moves closer
to the observed English/Italian class separation while avoiding a blind layer
search. These larger perturbations are exploratory and can damage output.
The language heuristic now includes Spanish markers after the first sweep
exposed a mixed English/Spanish answer; no such answer counts as Italian.
