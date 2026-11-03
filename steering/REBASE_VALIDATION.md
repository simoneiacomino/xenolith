# Upstream rebase validation — 2026-11-03

The steering change is rebased onto upstream commit
`362af7a817e17f7287d9586e113335b68219c830`. Only the steering commit is
reapplied; previously integrated fork commits are not replayed.

## Integration

- Preserve upstream CPU topology selection, worker options, and tested GPU IDs.
- Preserve `--workers`/`--cpus` when opening steered CLI and service engines,
  including the cached opening needed for full model identity checks.
- Link engine and steering tests with both steering and CPU policy/topology modules.
- Carry protocol changes into `docs/protocol.md`; link steering from the new
  documentation index and CLI guide.

## Checks performed after rebase

- `make check-steering-host`: PASS (native artifact/analytical tests and Python
  builder, capture analysis, evaluator tests on macOS).
- `clang -std=c11 -Wall -Wextra -fsyntax-only main.c`: PASS.
- Portable upstream CPU policy test: PASS.
- Make dry runs for the engine, worker, steering, context, runtime and server
  targets: PASS. These inspect build rules; they do not compile the Linux engine.
- Source conflict-marker and whitespace checks: PASS outside archived raw outputs.

## Pending target-hardware validation

The previous Intel host is no longer accessible, as confirmed by its owner.
Linux engine build, GPU/model tests and performance measurements have **not**
been rerun after the rebase. Attempts to check Linux-only runtime/server sources
with macOS headers encounter unavailable Linux APIs; they are not build validation.

All earlier GPU, behavior and performance results remain historical evidence for
their recorded executable/source identities. Upstream changed automatic CPU
worker selection, so those measurements must not be presented as measurements
of this rebased implementation. Run the documented native suites and behavior
smoke checks on a supported Intel host before treating the PR as fully validated.
