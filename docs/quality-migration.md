# Migrating numerical checks

The text fidelity workflow uses [Google FP64 references](testing.md#numerical-fidelity).
The following API, CLI and test changes affect existing integrations.

| Previous entry point | Replacement |
| --- | --- |
| `xenolith oracle`, `xe_oracle` | Use the session API for inference and `xe-quality` for full-vocabulary reference comparisons. The removed C API has no compatibility wrapper. |
| `certify` numerical stage | Use a qualified reference archive and an explicit acceptance policy. The former RMS 0.6 bound is not a KL threshold. |
| `certify` tokenizer/template stages | Run `tests/test_tokenizer_model` with a model path or `XENOLITH_MODEL`; it checks vocabulary fixtures, prompt IDs, byte roundtrips and chat templates. |
| `make golden` | Prepare new corpora explicitly with `xe_quality.tokenize_panel`. Existing token fixtures remain unchanged; tests do not regenerate expectations. |
| `tools/compare_logits`, `tools/dump_logits`, external logits argument to `test_prefill_session` | Use the current-build native driver and the quality package's archive comparison commands. |
| `tools/dump_layers` | Register `xe_session_set_layer_observer` or use the native quality driver's layer output. |
| `tools/tokenize_prompt` | Use `make quality-tokenizer`; for independent parity checks, build `quality-llama-tokenizer` against an explicit llama.cpp checkout. |

## Test coverage

`test_fixture_decode` checks state transitions against an independently
materialized execution schedule, finite logits, token/position counts, allocation
behavior and deterministic continuation. Cross-path RMS/top1 is diagnostic;
the former RMS 0.1 bound and fixed greedy-token expectations are removed.
Full-model numerical acceptance belongs to the reference comparison workflow.

`test_model_decode` checks isolated layer operators, packing, routing and every
selected expert against scalar calculations. `test_prefill_layer` checks CPU/GPU
block integration and intermediate layouts. Neither traverses a second scalar
model. Minimal scalar helpers remain in `tests/support/scalar_ops.h`.
Tokenizer, KV, cache, snapshot, protocol, memory and vision checks remain separate;
passing text fidelity checks does not qualify vision inference.

## Observer and snapshot compatibility

The layer observer is disabled by default and receives borrowed hidden-state
rows at absolute token positions. Callbacks must not mutate the data or reenter
the engine. Registration survives reset/rewind, is not inherited by a shadow,
and is not serialized. GPU observations synchronize execution and must be
disabled for throughput measurements.

Removing the scalar model changes the numeric source hash. Text snapshot
compatibility continues to use the numeric ABI and worker/rounding contract;
visual snapshots include the source hash and may require media replay after
this update.
