#ifndef XENOLITH_H
#define XENOLITH_H

#include <stdint.h>
#include <stdio.h>

#ifndef XE_BUILD_COMMIT
#define XE_BUILD_COMMIT "unknown"
#endif

typedef struct xe_engine xe_engine;
typedef struct xe_session xe_session;

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

/* Capacity is fixed for the engine and inherited by all its sessions. */
xe_engine *xe_engine_open_with_context(const char *gguf_path, int context);
xe_engine *xe_engine_open(const char *gguf_path);
xe_engine *xe_engine_open_vocab(const char *gguf_path);
void xe_engine_close(xe_engine *e);
void xe_engine_info(const xe_engine *e, FILE *out);

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
int xe_engine_worker_count(const xe_engine *e);
int xe_engine_worker_cpu(const xe_engine *e, int worker);

void xe_tokens_push(xe_tokens *tokens, int32_t token);
void xe_tokens_free(xe_tokens *tokens);
const char *xe_chat_template(const xe_engine *e, uint64_t *length);

void xe_oracle(xe_engine *e, const int32_t *tokens, int n_tokens, const char *dump_path,
               const char *layers_dir, int q8_mode, FILE *out);

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
const char *xe_snapshot_status_name(xe_snapshot_status status);

#endif
