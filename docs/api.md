# C API

Include [xenolith.h](../xenolith.h) for declarations and the function-level
reference. An engine owns the model and worker configuration; a session holds
inference state and must be freed before its engine closes. For image input and
wire integration, see [vision](vision.md) and the [protocol](protocol.md).

## Opening and using an engine

Run the following on a dedicated inference thread. It opens an engine, evaluates
one token and samples the next; a real chat client must construct the model's
appropriate prompt and handle the generated tokens.

```c
#include "xenolith.h"
#include <stdint.h>

void example(const char *model_path) {
    xe_engine_options options = { .context = 16384, .workers = 4 };
    xe_engine *engine = xe_engine_open_with_options(model_path, &options);
    xe_session *session = xe_session_new(engine);
    xe_tokens prefix = {0};
    xe_tokens_push(&prefix, xe_bos_id(engine));
    xe_session_sync(session, &prefix);
    xe_sampler sampler = { .temperature = 0.0f, .top_p = 1.0f };
    int32_t next = xe_session_next(session, &sampler);
    (void)next;
    xe_tokens_free(&prefix);
    xe_session_free(session);
    xe_engine_close(engine);
}
```

`xe_session_sync` evaluates a nonempty token prefix, reusing existing session
state where possible. `xe_session_next` samples from the current logits; it
does not append the result. To continue, append the sampled token to the prefix
and sync again, keeping the prefix within the configured context. The optional
`xe_session_sync_report` reports reused and prefilled tokens and whether the
session restarted.

Zero-initialize `xe_engine_options`, or pass `NULL`, for defaults. A zero
`context` uses `XE_CONTEXT_DEFAULT`; otherwise it must be between
`XE_CONTEXT_MIN` and `XE_CONTEXT_MAX`, inclusive. Zero `workers` selects
automatically unless `cpus` is supplied. For a manual map, `cpus` is an ordered
array of allowed logical CPU IDs on distinct physical cores; `cpu_count` is
its length and determines the worker count. `cpus` must be NULL exactly when
`cpu_count` is zero. A nonzero `workers` must match the list length. Opening
copies the options and array, so the caller may change or release them after
opening returns. Explicit counts and lists bypass the automatic catalog; see
[tuning](tuning.md) for selection and topology inspection.

`xe_engine_open` and `xe_engine_open_with_context` are wrappers that
select workers automatically. `xe_engine_open_cached` and
`xe_engine_open_cached_with_options` also prepare snapshot fingerprints at
opening. `xe_engine_open_vocab` and `xe_engine_open_vocab_with_options` open
without model weights and cannot create inference sessions. None of these C
openings reads the CLI's CPU environment variables.

Normal opening resolves the worker count and map before GPU initialization and
weight repacking. Vocab-only opening copies and validates options, but defers
machine topology and affinity checks until vision initializes the pool.
`xe_engine_worker_count` and `xe_engine_worker_cpu` expose the resolved count
and ordered map; the count is zero before vocab-only pool configuration, and a
worker index must be in `[0, count)`. The effective count contributes to the
snapshot numeric fingerprint and can affect snapshot compatibility.

## Thread and resource lifetime

The thread that creates the first session or initializes vision becomes the
pool owner. It is pinned to the first selected CPU until engine close restores
its previous affinity. Keep session work, session destruction and engine close
on that owner thread. The owner must still be allowed to use every selected CPU
when a pool starts; nested pools on an already pinned owner are not transparently
supported. Vision initialization establishes ownership even if projector
loading subsequently fails. Use a dedicated inference thread in a UI.

Keep the model file unchanged throughout the engine lifetime. The model
fingerprint returned by `xe_engine_model_fingerprint` is engine-owned and valid
until close; call it on the owner thread once a pool exists. Free token vectors
with `xe_tokens_free` and sessions with `xe_session_free` before closing the
engine.

## Errors

Engine opening and invalid worker configuration use the fatal error path: they
print a diagnostic and terminate the process with exit status 1. Other calls
can also terminate on invalid arguments or runtime failures, so an embedder
must not assume every API failure is recoverable. Some functions instead
return a status: for example, `xe_engine_enable_vision` returns zero on failure
and can write an error message, and snapshot operations return
`xe_snapshot_status`. Consult the declarations and implementation for each
function's contract; a status-returning call can still encounter a fatal error
in underlying initialization or execution.
