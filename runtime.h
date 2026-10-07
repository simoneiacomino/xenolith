#ifndef RUNTIME_H
#define RUNTIME_H

#include "xenolith.h"
#include "kvstore.h"
#include "conversation.h"
#include "profile.h"

#include <stddef.h>
#include <stdint.h>

typedef struct runtime runtime;
typedef uint64_t runtime_marker;

#define RUNTIME_MARKER_NONE UINT64_MAX

typedef enum {
    RUNTIME_OK,
    RUNTIME_CONTEXT_LENGTH_EXCEEDED,
    RUNTIME_SESSION_NOT_FOUND,
    RUNTIME_MARKER_UNAVAILABLE,
    RUNTIME_INVALID_ARGUMENT,
    RUNTIME_BUSY,
    RUNTIME_IO,
    RUNTIME_NOMEM
} runtime_status;

typedef struct {
    const char *model;
    int context_window;
    int max_output;
    int kvstore;            /* 1 when the snapshot store opened */
    int reasoning;
} runtime_info;

/* Why a checkpoint did not save. Checkpoints are never fatal (the
 * transcript is the source of truth and replay always works); the
 * report makes the outcome visible to the client, the log and tests. */
typedef enum {
    RUNTIME_CKPT_SAVED,
    RUNTIME_CKPT_NO_KVSTORE,    /* store never opened: permanent for the process */
    RUNTIME_CKPT_EMPTY,         /* nothing computed yet (no tokens, or KV not loaded) */
    RUNTIME_CKPT_NOTHING_NEW,   /* already saved up to the current boundary */
    RUNTIME_CKPT_KV_DIVERGED,   /* KV is not a prefix of the transcript (transient) */
    RUNTIME_CKPT_BUDGET,        /* snapshot does not fit the store budget: permanent for the session */
    RUNTIME_CKPT_IO,            /* store write failed */
    RUNTIME_CKPT_REJECTED,      /* engine refused the snapshot */
    RUNTIME_CKPT_RECORD_FAILED  /* snapshot written, transcript record failed */
} runtime_ckpt_reason;

typedef struct {
    int saved;
    runtime_ckpt_reason reason;
    uint64_t tokens;        /* boundary saved, or current size when not saved */
} runtime_checkpoint_report;

/* Why a resume did not load the snapshot it was expected to. */
typedef enum {
    RUNTIME_RESUME_LOADED,
    RUNTIME_RESUME_EVICTED,        /* snapshot file gone from the store */
    RUNTIME_RESUME_MODEL_MISMATCH,
    RUNTIME_RESUME_TOKEN_MISMATCH, /* transcript no longer matches the snapshot */
    RUNTIME_RESUME_IO,
    RUNTIME_RESUME_REJECTED        /* any other engine refusal */
} runtime_resume_reason;

typedef struct {
    int loaded;
    runtime_resume_reason reason;
    uint64_t tokens;        /* boundary loaded or attempted */
} runtime_resume_report;

const char *runtime_ckpt_reason_name(runtime_ckpt_reason reason);
const char *runtime_resume_reason_name(runtime_resume_reason reason);

typedef enum {
    RUNTIME_MESSAGE_USER = 1,
    RUNTIME_MESSAGE_TOOL_RESULT = 2,
    RUNTIME_MESSAGE_ASSISTANT = 3,
    RUNTIME_MESSAGE_SYSTEM = 4
} runtime_message_kind;

typedef struct {
    uint32_t kind;
    const char *text;
    const char *reasoning;
    uint64_t call_id;
    uint32_t tool_status;
    const char *tool_name;
    const profile_call *calls;
    size_t call_count;
} runtime_message;

typedef struct {
    float temperature;
    int32_t top_k;
    float top_p;
    int32_t max_tokens;
    uint64_t rng_seed;
    int reasoning_set;
    uint32_t reasoning_effort;
    int reasoning_history_set;
    uint32_t reasoning_history;
    int reasoning_budget_set;
    int32_t reasoning_budget;
} runtime_gen_params;

typedef struct {
    uint64_t input;
    uint64_t cache_read;
    uint64_t output;
    uint64_t total;
    uint64_t reasoning;
    uint64_t replayed;
    uint64_t shadow_prefilled;
    uint64_t shadow_background;
    uint64_t shadow_remaining;
    uint64_t shadow_kv_bytes;
    uint64_t shadow_wait_us;
} runtime_usage;

typedef enum {
    RUNTIME_EVENT_START = 1,
    RUNTIME_EVENT_INFERENCE_PROGRESS = 2,
    RUNTIME_EVENT_TEXT_DELTA = 3,
    RUNTIME_EVENT_TOOLCALL_START = 4,
    RUNTIME_EVENT_TOOLCALL_END = 5,
    RUNTIME_EVENT_DONE = 6,
    RUNTIME_EVENT_ERROR = 7,
    RUNTIME_EVENT_REASONING_DELTA = 8
} runtime_event_kind;

typedef enum {
    RUNTIME_STOP_STOP = 1,
    RUNTIME_STOP_TOOL_USE = 2,
    RUNTIME_STOP_LENGTH = 3,
    RUNTIME_STOP_ABORTED = 4
} runtime_stop;

typedef enum {
    RUNTIME_INFERENCE_PREFILL,
    RUNTIME_INFERENCE_DECODE
} runtime_inference_phase;

/* The first RUNNING opens a phase, before its work begins. FINISHED carries
 * its final measurements, including partial work on cancellation; it does not
 * imply request success. DONE/ERROR may close an open phase directly. No
 * progress follows FINISHED for that phase, and unentered phases emit nothing.
 * A fully cached prefill still emits RUNNING(0/0), then FINISHED(0/0). */
typedef enum {
    RUNTIME_INFERENCE_RUNNING,
    RUNTIME_INFERENCE_FINISHED
} runtime_inference_state;

/* Cumulative active runtime time, measured with CLOCK_MONOTONIC. Prefill
 * measures initial synchronization only, excluding shadow waits/setup. Includes
 * sampling, parsing and shadow work during decode; excludes client waits,
 * snapshot loads and terminal reconciliation/checkpoint writes. This is
 * interactive inference throughput, not a kernel benchmark. Prefill counts
 * initial computation only (no cached tokens or terminal replay). Decode
 * counts the same sampled/control tokens as usage.output. */
typedef struct {
    uint64_t tokens;
    uint64_t total;       /* prefill only; sync may revise initial cache estimate */
    uint64_t elapsed_ms;
} runtime_inference_measure;

typedef struct {
    runtime_inference_measure prefill;
    runtime_inference_measure decode;
} runtime_inference;

typedef struct {
    uint32_t kind;
    const uint8_t *text;
    size_t text_length;
    uint64_t call_id;
    const char *call_name;
    const char *arguments_json;
    uint32_t phase;      /* inference_progress: runtime_inference_phase */
    uint32_t state;      /* inference_progress: runtime_inference_state */
    runtime_inference_measure progress;
    runtime_inference inference; /* done/error: frozen before finalization */
    uint32_t stop;
    uint32_t reasoning_close;
    runtime_usage usage;
    runtime_marker marker;
    uint32_t error;
    const char *error_text;
    uint64_t error_tokens;     /* error: context_length_exceeded detail */
    uint64_t error_context;
    int checkpoint_attempted;          /* done: autosave ran at end of turn */
    runtime_checkpoint_report checkpoint;
    int resume_attempted;              /* done: a snapshot load was tried */
    runtime_resume_report resume;
} runtime_event;

typedef struct {
    uint64_t token_count;
    runtime_marker marker;
    int turn_open;
    int zero_prefill;
    int resume_stale;       /* a snapshot exists but behind the current boundary */
    size_t pending_calls;
} runtime_open_report;

typedef struct {
    uint32_t kind;
    uint32_t role;
    const char *text;
    uint64_t text_length;
    const char *reasoning;
    uint64_t reasoning_length;
    const char *extra_json;
    uint64_t extra_length;
    uint64_t call_id;
    uint32_t tool_status;
    const char *tool_name;
    uint32_t stop_reason;
    runtime_marker marker;
} runtime_history_entry;

runtime_status runtime_open(runtime **out, xe_engine *engine, const char *state_dir,
                      const char *cache_dir);
void runtime_close(runtime *w);

runtime_status runtime_describe(runtime *w, runtime_info *out);

runtime_status runtime_session_create(runtime *w, const char *system,
                                const profile_tool *tools, size_t tool_count,
                                conversation_id *id, runtime_marker *marker);
runtime_status runtime_session_open(runtime *w, const conversation_id *id,
                              runtime_open_report *out);
runtime_status runtime_session_list(runtime *w, conversation_summary **out,
                              size_t *count);
runtime_status runtime_session_stat(runtime *w, const conversation_id *id,
                              conversation_summary *out);
runtime_status runtime_session_delete(runtime *w, const conversation_id *id);

size_t runtime_pending_calls(runtime *w, uint64_t *call_ids, size_t cap);
uint64_t runtime_history_count(runtime *w);
runtime_status runtime_history_at(runtime *w, uint64_t position,
                            runtime_history_entry *out);

runtime_status runtime_append(runtime *w, const runtime_message *message,
                        runtime_marker *out);
runtime_status runtime_generate(runtime *w, const runtime_gen_params *params);
runtime_status runtime_ephemeral_generate(runtime *w, const char *system,
                                    const profile_tool *tools,
                                    size_t tool_count,
                                    const runtime_message *messages,
                                    size_t count,
                                    const runtime_gen_params *params);
runtime_status runtime_next_event(runtime *w, runtime_event *out);
/* Requests cooperative cancellation. OK confirms acceptance, not an aborted
 * outcome: a terminal result already decided is retained. Continue pulling
 * events through DONE/ERROR; cancellation does not roll back finalization. */
runtime_status runtime_cancel(runtime *w);

runtime_status runtime_rewind(runtime *w, runtime_marker marker);
runtime_status runtime_rewind_cost(runtime *w, runtime_marker marker,
                             uint64_t *prefill_tokens);
runtime_status runtime_rebuild(runtime *w, const char *system,
                         const profile_tool *tools, size_t tool_count,
                         const runtime_message *messages, size_t count,
                         runtime_marker *out);
runtime_status runtime_checkpoint(runtime *w, runtime_checkpoint_report *out);
/* Status of the snapshot store at open; KVSTORE_OK when it works. */
int runtime_kvstore_open_status(const runtime *w);

const char *runtime_status_code(runtime_status status);
const char *runtime_error_text(const runtime *w);
/* Structured detail of the last context_length_exceeded failure; 0 when
 * the last failure was something else. */
int runtime_error_detail(const runtime *w, uint64_t *tokens, uint64_t *context);

#endif
