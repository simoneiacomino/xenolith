# Steering experiment datasets

These examples were authored for this implementation; they require no external
API, model-generated training corpus, or downloaded steering artifact.

- `succinct-train.jsonl`: 100 paired prompts. Each pair asks about the same topic,
  changing only the requested level of detail. The positive pole requests a
  concise essential explanation; the negative pole requests context and examples.
- `succinct-validation.jsonl`: 10 unseen prompts for selecting layers/strength.
- `succinct-eval.jsonl`: 15 held-out prompts, including 5 Italian prompts.
- `constraints-eval.jsonl`: JSON and Python-format checks, evaluated separately
  from prose length. Python syntax/function presence is a limited structural
  check; inspect the saved code to assess its correctness.

Keep extraction, calibration, and held-out evaluation separate. The initial
experiment is about response detail, not factuality or refusal behavior.
Capturing a direction from an instruction does not establish that the direction
causally controls the behavior; the inference comparisons are essential.
