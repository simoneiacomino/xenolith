#ifndef XENOLITH_H
#define XENOLITH_H

#include <stdint.h>
#include <stdio.h>
#include "vision.h"
#include "steering.h"

#ifndef XE_BUILD_COMMIT
#define XE_BUILD_COMMIT "unknown"
#endif

typedef struct xe_engine xe_engine;
typedef struct xe_session xe_session;

/* Check the same first Level Zero driver/device selected for inference, without
 * loading a model, creating a context or allocating GPU memory. Writes human
 * diagnostics to the supplied streams; returns 0 when the enabled device is
 * reachable, 1 otherwise. Does not establish kernel/model compatibility. */
int xe_device_check(FILE *report, FILE *errors);

typedef struct {
    int32_t *v;
    int len;
    int cap;
} xe_tokens;

typedef struct {
    float temperature;
    int top_k;
    float top_p;
    uint64_t rng_state;
} xe_sampler;

typedef enum {
    XE_SNAPSHOT_OK,
    XE_SNAPSHOT_INVALID_ARGUMENT,
    XE_SNAPSHOT_EMPTY,
    XE_SNAPSHOT_IO,
    XE_SNAPSHOT_FORMAT,
    XE_SNAPSHOT_VERSION,
    XE_SNAPSHOT_MODEL_MISMATCH,
    XE_SNAPSHOT_TOKENIZER_MISMATCH,
    XE_SNAPSHOT_TEMPLATE_MISMATCH,
    XE_SNAPSHOT_LAYOUT_MISMATCH,
    XE_SNAPSHOT_CONTEXT_MISMATCH,
    XE_SNAPSHOT_NUMERIC_MISMATCH,
    XE_SNAPSHOT_TOKEN_MISMATCH,
    XE_SNAPSHOT_BACKEND,
    XE_SNAPSHOT_NOMEM,
    XE_SNAPSHOT_CHANGED
} xe_snapshot_status;

enum {
    XE_CONTEXT_MIN = 64,
    XE_CONTEXT_MAX = 262144,
    XE_CONTEXT_DEFAULT = XE_CONTEXT_MAX
};

/* Zero-initialize options for defaults. context=0 uses XE_CONTEXT_DEFAULT;
 * workers=0 selects automatically unless cpus is supplied. cpus is an ordered
 * list of cpu_count allowed logical CPUs on distinct physical cores. cpus must
 * be NULL exactly when cpu_count is zero; a nonzero workers value must match
 * the list length. Explicit requests bypass the automatic SKU catalog.
 * Options and the CPU list are copied during opening. Invalid options terminate
 * as other engine opening errors do. Vocab-only opening defers machine topology
 * and affinity checks until its vision pool starts.
 * Selected CPUs must support the build's required instructions. Native compiler
 * flags may require more than the checked minimum. The pool pins its owner thread
 * until engine close restores its affinity; embedding UIs should use a dedicated
 * inference thread. See docs/api.md and docs/tuning.md. */
typedef struct {
    int context;
    int workers;
    const int *cpus;
    int cpu_count;
} xe_engine_options;

xe_engine *xe_engine_open_with_options(const char *gguf_path, const xe_engine_options *options);
xe_engine *xe_engine_open_cached_with_options(const char *gguf_path, const xe_engine_options *options);
xe_engine *xe_engine_open_vocab_with_options(const char *gguf_path, const xe_engine_options *options);
/* Legacy openings select workers automatically and ignore CPU environment variables. */
xe_engine *xe_engine_open_with_context(const char *gguf_path, int context);
xe_engine *xe_engine_open(const char *gguf_path);
xe_engine *xe_engine_open_vocab(const char *gguf_path);
/* Prepare snapshot fingerprints during opening, with the full model hash
 * computed before repack releases source pages. Used by wire/serve when
 * --cache-budget > 0 or --mmproj is set, so requests never pay that cost. */
xe_engine *xe_engine_open_cached(const char *gguf_path, int context);
/* SHA-256 of the complete original GGUF, owned by the engine until close.
 * Cached opening computes it eagerly. Other opening functions defer it until
 * this call or the first snapshot save/load/check. Subsequent calls reuse it.
 * Call on the engine owner thread. The model file must remain unchanged for
 * the engine's lifetime.
 * This identifies cache contents; it does not restrict compatible variants. */
const uint8_t *xe_engine_model_fingerprint(xe_engine *e);
void xe_engine_close(xe_engine *e);
void xe_engine_info(const xe_engine *e, FILE *out);
int xe_engine_enable_vision(xe_engine *e, const char *mmproj_path,
                             char *error, size_t error_cap);
int xe_engine_encode_image(xe_engine *e, const xe_image *image, float *output,
                            const char *dump_directory);
const uint8_t *xe_engine_vision_fingerprint(const xe_engine *e);
int xe_engine_vision_artifacts_match(const xe_engine *e,
                                    const uint8_t model_id[32],
                                    const uint8_t projector_id[32]);
int xe_engine_vision_numeric_fingerprint(xe_engine *e, uint8_t out[32]);
int xe_session_set_text_prefill(xe_session *s, int rows);
int xe_session_sync_input(xe_session *s, const xe_input *input);
int xe_session_common_input(const xe_session *s, const xe_input *input);
int xe_session_has_visual(const xe_session *s);

int xe_encode_text(const xe_engine *e, const char *text, int32_t *out, int cap);
/* Returns the token count; leaves out unchanged when the count exceeds cap. */
int xe_encode_text_bounded(const xe_engine *e, const char *text, int32_t *out, int cap);
int xe_detokenize(const xe_engine *e, int32_t tok, char *buf, int cap);
int32_t xe_token_id(const xe_engine *e, const char *piece);
const char *xe_token_piece(const xe_engine *e, int32_t tok, int *len);
int32_t xe_bos_id(const xe_engine *e);
int32_t xe_eos_id(const xe_engine *e);
int32_t xe_eot_id(const xe_engine *e);
int xe_context_size(const xe_engine *e);
int xe_vocab_size(const xe_engine *e);
uint64_t xe_engine_model_size(const xe_engine *e);
/* Effective count and ordered CPU map. Vocab-only engines report count zero
 * until vision configuration; worker_cpu requires an index in [0, count). */
int xe_engine_worker_count(const xe_engine *e);
int xe_engine_worker_cpu(const xe_engine *e, int worker);

void xe_tokens_push(xe_tokens *tokens, int32_t token);
void xe_tokens_free(xe_tokens *tokens);
const char *xe_chat_template(const xe_engine *e, uint64_t *length);

/* Optional diagnostic observation of the real execution path after each layer.
 * hidden contains count floats, borrowed until callback returns; position is
 * absolute. Callbacks must not mutate data or reenter engine/session APIs.
 * Prefill observations synchronize the GPU and can substantially affect timing.
 * Registration survives reset/rewind, is not inherited by new shadow sessions,
 * and is not snapshot state. Keep user valid until disabling or session free.
 * Set NULL to disable. */
typedef void (*xe_layer_observer)(void *user, int layer, int position,
                                  const float *hidden, int count);
void xe_session_set_layer_observer(xe_session *s, xe_layer_observer observer,
                                   void *user);

/* Ordered prospective transitions. The session retains artifact references.
 * Replacing history invalidates only a prefix computed under different settings. */
typedef struct {
    int position;
    xe_steering *vector; /* NULL disables steering. */
    float strength;
} xe_steering_step;
int xe_engine_steering_store(xe_engine *e, const char *directory);
const char *xe_engine_steering_directory(const xe_engine *e);
int xe_engine_layer_count(const xe_engine *e);
int xe_engine_hidden_width(const xe_engine *e);
int xe_session_steering_schedule(xe_session *s, const xe_steering_step *steps, size_t count);
int xe_session_steering_set(xe_session *s, xe_steering *vector, float strength);
size_t xe_session_steering_count(const xe_session *s);
const xe_steering_step *xe_session_steering_steps(const xe_session *s);
/* Enable unsteered last-token activation capture; rows are session-owned. */
int xe_session_capture_enable(xe_session *s);
int xe_session_capture_enable_at(xe_session *s, xe_steering_site site);
const float *xe_session_capture_rows(const xe_session *s);
xe_session *xe_session_new(xe_engine *e);
void xe_session_free(xe_session *s);
void xe_session_reset(xe_session *s);
void xe_session_rewind(xe_session *s, int position);
int xe_session_position(xe_session *s);
int xe_session_context_size(const xe_session *s);

typedef struct {
    int reused;
    int prefilled;
    int restarted;
} xe_sync_report;

int xe_session_sync_input_report(xe_session *s, const xe_input *input,
                                  xe_sync_report *report);
/* Append text to the session's already owned visual prefix; rejects edits. */
int xe_session_sync_continuation(xe_session *s, const xe_tokens *prefix,
                                  xe_sync_report *report);

void xe_session_sync(xe_session *s, const xe_tokens *prefix);
void xe_session_sync_report(xe_session *s, const xe_tokens *prefix,
                            xe_sync_report *report);
int xe_session_common(const xe_session *s, const xe_tokens *prefix);
const float *xe_session_logits(xe_session *s);
int32_t xe_session_next(xe_session *s, xe_sampler *sp);

xe_snapshot_status xe_session_snapshot_size(xe_session *s, uint64_t *size);
xe_snapshot_status xe_session_snapshot_save(xe_session *s, FILE *out);
xe_snapshot_status xe_session_snapshot_load(xe_session *s, FILE *in,
                                            const xe_tokens *expected);
/* Mixed snapshot v2 binds image, media and embedding identities. */
xe_snapshot_status xe_session_snapshot_load_input(xe_session *s, FILE *in,
                                                  const xe_input *expected);
const char *xe_snapshot_status_name(xe_snapshot_status status);

#endif
