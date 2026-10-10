#define _POSIX_C_SOURCE 200809L

#include "runtime.h"
#include "format.h"
#include "json.h"
#include "xenolith_internal.h"

#include <stdarg.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

enum {
    RUNTIME_GEN_NONE = 0,
    RUNTIME_GEN_DURABLE = 1,
    RUNTIME_GEN_EPHEMERAL = 2,
    RUNTIME_PHASE_PREFILL = 0,
    RUNTIME_PHASE_DECODE = 1,
    RUNTIME_PREFILL_CHUNK = 512
};

typedef struct {
    uint64_t id;
    char *name;
    char *arguments;
    int complete;
} runtime_gen_call;

typedef struct {
    uint8_t *render;
    uint64_t render_length;
    int32_t *tokens;
    uint32_t token_count;
} runtime_render_copy;

struct runtime {
    xe_engine *engine;
    profile *prof;
    conversation_store *cstore;
    kvstore *kv;
    int context;

    conversation *current;
    conversation_id current_id;
    int has_current;
    uint64_t next_call_id;
    uint64_t saved_tokens;
    int64_t saved_at;
    int ckpt_failures;        /* consecutive autosave failures (backoff) */
    int ckpt_autosave_off;    /* set on budget: the session cannot fit, stop trying */
    int kv_open_status;       /* kvstore_status of the open attempt */
    int autosave_attempted;   /* last turn ran an autosave */
    runtime_checkpoint_report autosave;
    int resume_attempted;     /* last generation tried a snapshot load */
    runtime_resume_report resume;

    xe_session *session;
    xe_session *ephemeral;
    xe_session *gen_session;
    xe_session *shadow;
    int shadow_inflight;
    int shadow_cooldown;
    uint64_t shadow_prefilled;
    uint64_t shadow_background;
    uint64_t shadow_peak_kv;

    int gen_kind;
    int gen_phase;
    int cancel_requested;
    int started_emitted;
    int first_sync_done;
    uint64_t generation_id;
    conversation_settings gen_settings;
    xe_sampler sampler;
    int32_t gen_max_tokens;
    int gen_thinking;
    int gen_after_tool;
    uint32_t gen_turn;
    int capture_anchor;
    profile_reasoning_policy reasoning_policy;
    int reasoning_tokens;
    uint32_t reasoning_close;
    int utf8_pending;
    int replay_from;

    int32_t *prompt;
    int prompt_length;
    int prompt_synced;
    int prompt_common;
    int framing_offset;
    int sampled_length;
    runtime_usage usage;
    runtime_inference inference;
    uint64_t inference_ns[2];
    uint64_t decode_progress_ns;
    int64_t inference_started;
    int inference_active;
    int inference_phase;
    int prefill_started_emitted;
    int prefill_complete_emitted;
    int decode_complete_emitted;
    int decode_started_emitted;
    int terminal_pending;
    uint32_t terminal_stop;
    uint32_t terminal_record;

    json_writer content;
    json_writer reasoning;
    json_writer render;
    runtime_gen_call *calls;
    size_t call_count;
    size_t call_capacity;
    int call_open;

    char error_text[512];
    uint64_t error_tokens;    /* context_length_exceeded: prompt size */
    uint64_t error_context;   /* context_length_exceeded: window */
    json_writer tools_scratch;
    json_writer calls_scratch;
};

static void runtime_settings_defaults(conversation_settings *settings);
static runtime_status runtime_project_mutation(runtime *w, conversation *c,
                                         int thinking, uint32_t history);

const char *runtime_status_code(runtime_status status) {
    switch (status) {
    case RUNTIME_OK: return "ok";
    case RUNTIME_CONTEXT_LENGTH_EXCEEDED: return "context_length_exceeded";
    case RUNTIME_SESSION_NOT_FOUND: return "session_not_found";
    case RUNTIME_MARKER_UNAVAILABLE: return "marker_unavailable";
    case RUNTIME_INVALID_ARGUMENT: return "invalid_request";
    case RUNTIME_BUSY: return "busy";
    case RUNTIME_IO: return "io_error";
    case RUNTIME_NOMEM: return "out_of_memory";
    }
    return "unknown";
}

const char *runtime_error_text(const runtime *w) {
    return w && w->error_text[0] ? w->error_text : "";
}

/* Error texts are short phrases from a closed set: no digits, no
 * client-supplied strings. Harnesses classify provider errors by regex on
 * the text (pi matches bare "429"/"5xx" substrings as retryable), so a
 * marker, call id or tool name inside the text can turn a deterministic
 * error into a retried one. Quantities travel as structured fields
 * (see runtime_error_detail). test_runtime enforces the rule on every failure
 * it provokes. */
static runtime_status runtime_fail(runtime *w, runtime_status status,
                             const char *fmt, ...) {
    va_list args;
    va_start(args, fmt);
    vsnprintf(w->error_text, sizeof w->error_text, fmt, args);
    va_end(args);
    w->error_tokens = 0;
    w->error_context = 0;
    return status;
}

int runtime_error_detail(const runtime *w, uint64_t *tokens, uint64_t *context) {
    if (!w || !w->error_context) return 0;
    if (tokens) *tokens = w->error_tokens;
    if (context) *context = w->error_context;
    return 1;
}

static runtime_status runtime_from_conversation(runtime *w, conversation_status s) {
    switch (s) {
    case CONVERSATION_OK:
        return RUNTIME_OK;
    case CONVERSATION_MISS:
        return runtime_fail(w, RUNTIME_SESSION_NOT_FOUND, "session not found");
    case CONVERSATION_LOCKED:
        return runtime_fail(w, RUNTIME_BUSY, "session record is locked");
    case CONVERSATION_IO:
        return runtime_fail(w, RUNTIME_IO,
                         "session record io failed, temporarily unavailable");
    case CONVERSATION_NOMEM:
        return runtime_fail(w, RUNTIME_NOMEM, "out of memory");
    case CONVERSATION_DAMAGED:
        return runtime_fail(w, RUNTIME_INVALID_ARGUMENT,
                         "session record is damaged");
    case CONVERSATION_VERSION:
        return runtime_fail(w, RUNTIME_INVALID_ARGUMENT,
                         "session record version is unsupported");
    default:
        return runtime_fail(w, RUNTIME_INVALID_ARGUMENT, "%s",
                         conversation_status_name(s));
    }
}

static runtime_status runtime_from_profile(runtime *w, profile_status s) {
    switch (s) {
    case PROFILE_OK:
        return RUNTIME_OK;
    case PROFILE_NOMEM:
        return runtime_fail(w, RUNTIME_NOMEM, "out of memory");
    default:
        return runtime_fail(w, RUNTIME_INVALID_ARGUMENT, "render failed");
    }
}

static int64_t runtime_now(void) {
    struct timespec now;
    if (clock_gettime(CLOCK_REALTIME, &now) != 0) return 0;
    return (int64_t)now.tv_sec * INT64_C(1000000000) + now.tv_nsec;
}

static int64_t runtime_monotonic(void) {
    struct timespec now;
    if (clock_gettime(CLOCK_MONOTONIC, &now) != 0) return 0;
    return (int64_t)now.tv_sec * INT64_C(1000000000) + now.tv_nsec;
}

static void runtime_inference_begin(runtime *w, int phase) {
    w->inference_started = runtime_monotonic();
    w->inference_phase = phase;
    w->inference_active = 1;
}

static void runtime_inference_end(runtime *w) {
    if (!w->inference_active) return;
    int64_t finished = runtime_monotonic();
    int phase = w->inference_phase;
    if (w->inference_started > 0 && finished > w->inference_started)
        w->inference_ns[phase] += (uint64_t)(finished - w->inference_started);
    w->inference_active = 0;
    w->inference.prefill.elapsed_ms = w->inference_ns[0] / 1000000;
    w->inference.decode.elapsed_ms = w->inference_ns[1] / 1000000;
    w->inference.decode.tokens = (uint64_t)w->sampled_length;
}

static runtime_status runtime_inference_progress(runtime *w, int phase, int state,
                                                 runtime_event *out) {
    runtime_inference_end(w);
    memset(out, 0, sizeof *out);
    out->kind = RUNTIME_EVENT_INFERENCE_PROGRESS;
    out->phase = (uint32_t)phase;
    out->state = (uint32_t)state;
    out->progress = phase == RUNTIME_INFERENCE_PREFILL
                    ? w->inference.prefill : w->inference.decode;
    if (phase == RUNTIME_INFERENCE_DECODE)
        w->decode_progress_ns = w->inference_ns[1];
    return RUNTIME_OK;
}

static int runtime_default_cache_dir(char *out, size_t cap) {
    const char *cache = getenv("XDG_CACHE_HOME");
    int n;
    if (cache && *cache == '/') {
        n = snprintf(out, cap, "%s/xenolith/kv", cache);
    } else {
        const char *home = getenv("HOME");
        if (!home || *home != '/') return 0;
        n = snprintf(out, cap, "%s/.cache/xenolith/kv", home);
    }
    return n > 0 && (size_t)n < cap;
}

runtime_status runtime_open(runtime **out, xe_engine *engine, const char *state_dir,
                      const char *cache_dir) {
    if (!out || !engine) return RUNTIME_INVALID_ARGUMENT;
    *out = NULL;
    runtime *w = calloc(1, sizeof *w);
    if (!w) return RUNTIME_NOMEM;
    w->engine = engine;
    w->context = xe_context_size(engine);
    w->call_open = -1;
    if (profile_open(&w->prof, engine) != PROFILE_OK) {
        free(w);
        return RUNTIME_INVALID_ARGUMENT;
    }
    char state_path[4096];
    if (!state_dir) {
        if (!conversation_default_state_dir(state_path, sizeof state_path)) {
            profile_close(w->prof);
            free(w);
            return RUNTIME_IO;
        }
        state_dir = state_path;
    }
    conversation_status opened = conversation_store_open(&w->cstore,
                                                         state_dir);
    if (opened != CONVERSATION_OK) {
        runtime_status status = runtime_from_conversation(w, opened);
        profile_close(w->prof);
        free(w);
        return status;
    }
    char cache_path[4096];
    if (!cache_dir && runtime_default_cache_dir(cache_path, sizeof cache_path))
        cache_dir = cache_path;
    w->kv_open_status = KVSTORE_OK;
    if (cache_dir) {
        w->kv_open_status = (int)kvstore_open(&w->kv, cache_dir,
                                              KVSTORE_DEFAULT_BUDGET);
        if (w->kv_open_status != KVSTORE_OK) w->kv = NULL;
    }
    *out = w;
    return RUNTIME_OK;
}

int runtime_kvstore_open_status(const runtime *w) {
    return w ? w->kv_open_status : (int)KVSTORE_INVALID_ARGUMENT;
}

const char *runtime_ckpt_reason_name(runtime_ckpt_reason reason) {
    switch (reason) {
    case RUNTIME_CKPT_SAVED: return "saved";
    case RUNTIME_CKPT_NO_KVSTORE: return "no_kvstore";
    case RUNTIME_CKPT_EMPTY: return "empty";
    case RUNTIME_CKPT_NOTHING_NEW: return "nothing_new";
    case RUNTIME_CKPT_KV_DIVERGED: return "kv_diverged";
    case RUNTIME_CKPT_BUDGET: return "budget";
    case RUNTIME_CKPT_IO: return "io";
    case RUNTIME_CKPT_REJECTED: return "rejected";
    case RUNTIME_CKPT_RECORD_FAILED: return "record_failed";
    }
    return "unknown";
}

const char *runtime_resume_reason_name(runtime_resume_reason reason) {
    switch (reason) {
    case RUNTIME_RESUME_LOADED: return "loaded";
    case RUNTIME_RESUME_EVICTED: return "evicted";
    case RUNTIME_RESUME_MODEL_MISMATCH: return "model_mismatch";
    case RUNTIME_RESUME_TOKEN_MISMATCH: return "token_mismatch";
    case RUNTIME_RESUME_IO: return "io";
    case RUNTIME_RESUME_REJECTED: return "rejected";
    }
    return "unknown";
}

static void runtime_calls_reset(runtime *w) {
    for (size_t i = 0; i < w->call_count; i++) {
        free(w->calls[i].name);
        free(w->calls[i].arguments);
    }
    w->call_count = 0;
    w->call_open = -1;
}

static void runtime_shadow_record(runtime *w, int rows, int background) {
    if (rows <= 0) return;
    w->shadow_prefilled += (uint64_t)rows;
    if (background) w->shadow_background += (uint64_t)rows;
    if (w->gen_kind != RUNTIME_GEN_NONE) {
        w->usage.input += (uint64_t)rows;
        w->usage.replayed += (uint64_t)rows;
    }
}

static int runtime_shadow_poll(runtime *w, int background) {
    if (!w->shadow || !w->shadow_inflight) return 1;
    int rows = xe_session_shadow_poll(w->shadow);
    if (!rows) return 0;
    w->shadow_inflight = 0;
    runtime_shadow_record(w, rows, background);
    if (background) w->shadow_cooldown = 32;
    return 1;
}

static void runtime_shadow_wait(runtime *w, int background) {
    if (!w->shadow || !w->shadow_inflight) return;
    int rows = xe_session_shadow_wait(w->shadow);
    w->shadow_inflight = 0;
    runtime_shadow_record(w, rows, background);
    if (background) w->shadow_cooldown = 32;
}

static void runtime_shadow_drop(runtime *w) {
    if (!w->shadow) return;
    runtime_shadow_wait(w, 1);
    xe_session_free(w->shadow);
    w->shadow = NULL;
    w->shadow_inflight = 0;
    w->shadow_cooldown = 0;
    w->shadow_prefilled = 0;
    w->shadow_background = 0;
    w->shadow_peak_kv = 0;
}

static runtime_status runtime_shadow_target(runtime *w, int32_t **tokens,
                                      uint64_t *count) {
    conversation_settings settings;
    if (!conversation_get_settings(w->current, &settings))
        runtime_settings_defaults(&settings);
    return runtime_from_conversation(
        w, conversation_project_copy(
               w->current,
               settings.reasoning_effort != CONVERSATION_REASONING_OFF,
               settings.reasoning_history, 1, tokens, count));
}

static runtime_status runtime_shadow_tick(runtime *w) {
    if (!w->shadow || !w->has_current) return RUNTIME_OK;
    if (!runtime_shadow_poll(w, 1)) return RUNTIME_OK;
    if (w->shadow_cooldown > 0) {
        w->shadow_cooldown--;
        return RUNTIME_OK;
    }
    int32_t *tokens;
    uint64_t count;
    runtime_status status = runtime_shadow_target(w, &tokens, &count);
    if (status != RUNTIME_OK) return status;
    if (count < (uint64_t)xe_session_shadow_split(w->shadow)) {
        free(tokens);
        return RUNTIME_OK;
    }
    xe_tokens target = { tokens, (int)count, (int)count };
    int rows = xe_session_shadow_start(w->shadow, &target, 64);
    free(tokens);
    if (rows < 0) {
        runtime_shadow_drop(w);
        return RUNTIME_OK;
    }
    if (rows > 0) w->shadow_inflight = rows;
    uint64_t bytes = xe_session_shadow_kv_bytes(w->shadow);
    if (bytes > w->shadow_peak_kv) w->shadow_peak_kv = bytes;
    return RUNTIME_OK;
}

static void runtime_shadow_begin(runtime *w) {
    runtime_shadow_drop(w);
    w->shadow = xe_session_shadow_new(w->session);
    if (!w->shadow) return;
    w->shadow_cooldown = 0;
    w->shadow_peak_kv = xe_session_shadow_kv_bytes(w->shadow);
}

static int runtime_shadow_promote_resume(runtime *w) {
    if (!w->shadow || !w->session || !w->has_current) return 0;
    runtime_shadow_wait(w, 1);
    uint64_t count;
    const int32_t *tokens = conversation_tokens(w->current, &count);
    xe_tokens prefix = { (int32_t *)tokens, (int)count, (int)count };
    int position = xe_session_position(w->shadow);
    int split = xe_session_shadow_split(w->shadow);
    if (position < split || xe_session_common(w->shadow, &prefix) != position)
        return 0;
    if (!xe_session_shadow_promote(w->session, w->shadow)) return 0;
    xe_session_free(w->shadow);
    w->shadow = NULL;
    w->shadow_inflight = 0;
    w->shadow_cooldown = 0;
    w->shadow_prefilled = 0;
    w->shadow_background = 0;
    w->shadow_peak_kv = 0;
    return 1;
}

void runtime_close(runtime *w) {
    if (!w) return;
    runtime_shadow_drop(w);
    if (w->current) conversation_close(w->current);
    if (w->session) xe_session_free(w->session);
    if (w->ephemeral) xe_session_free(w->ephemeral);
    conversation_store_close(w->cstore);
    kvstore_close(w->kv);
    profile_close(w->prof);
    runtime_calls_reset(w);
    free(w->calls);
    free(w->prompt);
    json_writer_free(&w->content);
    json_writer_free(&w->reasoning);
    json_writer_free(&w->render);
    json_writer_free(&w->tools_scratch);
    json_writer_free(&w->calls_scratch);
    free(w);
}

runtime_status runtime_describe(runtime *w, runtime_info *out) {
    if (!w || !out) return RUNTIME_INVALID_ARGUMENT;
    out->model = profile_model(w->prof);
    out->context_window = w->context;
    out->max_output = w->context;
    out->kvstore = w->kv != NULL;
    out->reasoning = 1;
    return RUNTIME_OK;
}

static int runtime_render_copy_set(runtime_render_copy *copy,
                                const profile_render *render) {
    memset(copy, 0, sizeof *copy);
    copy->render = malloc(render->render_length ? render->render_length : 1);
    copy->tokens = malloc((size_t)(render->token_count ?
                                  render->token_count : 1) *
                          sizeof(*copy->tokens));
    if (!copy->render || !copy->tokens) {
        free(copy->render);
        free(copy->tokens);
        memset(copy, 0, sizeof *copy);
        return 0;
    }
    if (render->render_length)
        memcpy(copy->render, render->render, render->render_length);
    if (render->token_count)
        memcpy(copy->tokens, render->tokens,
               (size_t)render->token_count * sizeof(*copy->tokens));
    copy->render_length = render->render_length;
    copy->token_count = render->token_count;
    return 1;
}

static void runtime_render_copy_free(runtime_render_copy *copy) {
    free(copy->render);
    free(copy->tokens);
    memset(copy, 0, sizeof *copy);
}

static uint32_t runtime_turn(const conversation *c) {
    uint64_t count = conversation_visible_count(c);
    if (!count) return PROFILE_TURN_PADDED;
    const conversation_event *last = conversation_event_at(
        c, conversation_visible_index(c, count - 1));
    if (!last) return PROFILE_TURN_PADDED;
    switch (last->type) {
    case CONVERSATION_EVENT_MESSAGE:
        if (last->role != CONVERSATION_ROLE_ASSISTANT)
            return PROFILE_TURN_PADDED;
        if (last->token_count &&
            last->tokens[last->token_count - 1] == 106)
            return PROFILE_TURN_BARE;
        return PROFILE_TURN_OPEN;
    case CONVERSATION_EVENT_TOOL_RESULT:
        return PROFILE_TURN_OPEN;
    case CONVERSATION_EVENT_GENERATION_RESULT:
        switch (last->stop_reason) {
        case CONVERSATION_STOP_EOT_SAMPLED:
        case CONVERSATION_STOP_EOT_SYNTHETIC:
        case CONVERSATION_STOP_EOS:
            return PROFILE_TURN_BARE;
        default:
            return PROFILE_TURN_OPEN;
        }
    default:
        return PROFILE_TURN_PADDED;
    }
}

static const conversation_event *runtime_last_visible(const conversation *c) {
    uint64_t count = conversation_visible_count(c);
    if (!count) return NULL;
    return conversation_event_at(c,
        conversation_visible_index(c, count - 1));
}

static int runtime_after_tool(const conversation *c) {
    const conversation_event *last = runtime_last_visible(c);
    return last && last->type == CONVERSATION_EVENT_TOOL_RESULT;
}

static int runtime_reasoning_continues(const conversation *c) {
    const conversation_event *last = runtime_last_visible(c);
    return last &&
           last->type == CONVERSATION_EVENT_GENERATION_RESULT &&
           (last->reasoning_close == CONVERSATION_REASONING_LENGTH ||
            last->reasoning_close == CONVERSATION_REASONING_ABORTED);
}

static runtime_status runtime_tools_json(runtime *w, const profile_tool *tools,
                                   size_t tool_count) {
    json_writer_reset(&w->tools_scratch);
    json_raw(&w->tools_scratch, "[");
    for (size_t i = 0; i < tool_count; i++) {
        if (i) json_raw(&w->tools_scratch, ",");
        json_raw(&w->tools_scratch, "{\"name\":");
        json_string(&w->tools_scratch, tools[i].name,
                    strlen(tools[i].name));
        json_raw(&w->tools_scratch, ",\"description\":");
        json_string(&w->tools_scratch, tools[i].description,
                    strlen(tools[i].description));
        json_raw(&w->tools_scratch, ",\"parameters\":");
        if (tools[i].parameters_json && *tools[i].parameters_json)
            json_raw(&w->tools_scratch, tools[i].parameters_json);
        else
            json_raw(&w->tools_scratch, "null");
        json_raw(&w->tools_scratch, "}");
    }
    json_raw(&w->tools_scratch, "]");
    if (w->tools_scratch.failed)
        return runtime_fail(w, RUNTIME_NOMEM, "out of memory");
    return RUNTIME_OK;
}

static runtime_status runtime_context_error(runtime *w, uint64_t total) {
    runtime_status status = runtime_fail(w, RUNTIME_CONTEXT_LENGTH_EXCEEDED,
                                   "prompt does not fit the context window");
    w->error_tokens = total;
    w->error_context = (uint64_t)w->context;
    return status;
}

static runtime_status runtime_budget_check(runtime *w, uint64_t total) {
    if (total < (uint64_t)w->context - 1) return RUNTIME_OK;
    return runtime_context_error(w, total);
}

static runtime_status runtime_append_system_event(runtime *w, conversation *c,
                                            const char *system,
                                            const profile_tool *tools,
                                            size_t tool_count,
                                            runtime_marker *marker) {
    profile_render render;
    runtime_status status = runtime_from_profile(
        w, profile_render_system(w->prof, system, tools, tool_count, 0,
                                 &render));
    if (status != RUNTIME_OK) return status;
    runtime_render_copy primary;
    if (!runtime_render_copy_set(&primary, &render))
        return runtime_fail(w, RUNTIME_NOMEM, "out of memory");
    status = runtime_from_profile(
        w, profile_render_system(w->prof, system, tools, tool_count, 1,
                                 &render));
    if (status != RUNTIME_OK) {
        runtime_render_copy_free(&primary);
        return status;
    }
    conversation_block blocks[2];
    uint32_t block_count = 1;
    blocks[0].format = CONVERSATION_BLOCK_TEXT;
    blocks[0].data = system ? system : "";
    blocks[0].length = system ? strlen(system) : 0;
    if (tool_count) {
        status = runtime_tools_json(w, tools, tool_count);
        if (status != RUNTIME_OK) {
            runtime_render_copy_free(&primary);
            return status;
        }
        blocks[1].format = CONVERSATION_BLOCK_JSON;
        blocks[1].data = w->tools_scratch.data;
        blocks[1].length = w->tools_scratch.length;
        block_count = 2;
    }
    status = runtime_from_conversation(
        w, conversation_append_message_variants(
               c, CONVERSATION_ROLE_SYSTEM, blocks, block_count,
               primary.render, primary.render_length,
               primary.tokens, primary.token_count,
               render.render, render.render_length,
               render.tokens, render.token_count,
               NULL, 0, CONVERSATION_REASONING_NONE));
    runtime_render_copy_free(&primary);
    if (status != RUNTIME_OK) return status;
    if (marker) *marker = conversation_event_count(c) - 1;
    return RUNTIME_OK;
}

/* Consecutive transient failures push the next autosave out: 30 s, then
 * +30, +90, +210, +450 s on top of the 30 s window, capped. The explicit
 * checkpoint op bypasses this and a success resets it. */
static void runtime_ckpt_backoff(runtime *w, int64_t now) {
    if (w->ckpt_failures < 5) w->ckpt_failures++;
    int64_t window = INT64_C(30000000000);
    int64_t extra = window * ((INT64_C(1) << w->ckpt_failures) - 2);
    w->saved_at = now + extra;
}

static void runtime_ckpt_reset(runtime *w) {
    w->ckpt_failures = 0;
    w->ckpt_autosave_off = 0;
}

static void runtime_checkpoint_now(runtime *w, runtime_checkpoint_report *out) {
    runtime_checkpoint_report report;
    memset(&report, 0, sizeof report);
    if (!out) out = &report;
    memset(out, 0, sizeof *out);
    if (!w->kv) { out->reason = RUNTIME_CKPT_NO_KVSTORE; return; }
    if (!w->session || !w->has_current) { out->reason = RUNTIME_CKPT_EMPTY; return; }
    conversation *c = w->current;
    uint64_t total;
    const int32_t *tokens = conversation_tokens(c, &total);
    out->tokens = total;
    if (!total) { out->reason = RUNTIME_CKPT_EMPTY; return; }
    /* Checkpointing must never sync a projection beyond the engine capacity. */
    if (total > (uint64_t)w->context) {
        out->reason = RUNTIME_CKPT_REJECTED;
        return;
    }
    int position = xe_session_position(w->session);
    if (position <= 0) { out->reason = RUNTIME_CKPT_EMPTY; return; }
    if ((uint64_t)position > total) { out->reason = RUNTIME_CKPT_KV_DIVERGED; return; }
    xe_tokens full = { (int32_t *)tokens, (int)total, (int)total };
    if (xe_session_common(w->session, &full) < position) {
        out->reason = RUNTIME_CKPT_KV_DIVERGED;
        return;
    }
    if ((uint64_t)position < total) {
        xe_session_sync(w->session, &full);
        position = (int)total;
    }
    if ((uint64_t)position <= w->saved_tokens) {
        out->reason = RUNTIME_CKPT_NOTHING_NEW;
        return;
    }
    kvstore_save_options options;
    memset(&options, 0, sizeof options);
    options.rebuild_cost = (uint64_t)position;
    kvstore_id id;
    xe_snapshot_status snapshot_status;
    kvstore_status ks = kvstore_save(w->kv, w->session, &options, &id,
                                     &snapshot_status);
    int64_t now = runtime_now();
    if (ks != KVSTORE_OK) {
        if (ks == KVSTORE_BUDGET) {
            out->reason = RUNTIME_CKPT_BUDGET;
            w->ckpt_autosave_off = 1;
        } else {
            out->reason = ks == KVSTORE_REJECTED ? RUNTIME_CKPT_REJECTED
                                                 : RUNTIME_CKPT_IO;
            runtime_ckpt_backoff(w, now);
        }
        return;
    }
    if (conversation_append_snapshot_ref(c, &id, (uint64_t)position)
            != CONVERSATION_OK ||
        conversation_commit(c) != CONVERSATION_OK) {
        out->reason = RUNTIME_CKPT_RECORD_FAILED;
        runtime_ckpt_backoff(w, now);
        return;
    }
    w->saved_tokens = (uint64_t)position;
    w->saved_at = now;
    runtime_ckpt_reset(w);
    out->saved = 1;
    out->reason = RUNTIME_CKPT_SAVED;
    out->tokens = (uint64_t)position;
}

static void runtime_park(runtime *w, runtime_checkpoint_report *out) {
    if (out) memset(out, 0, sizeof *out);
    if (!w->has_current) return;
    runtime_shadow_drop(w);
    runtime_checkpoint_now(w, out);
    conversation_close(w->current);
    w->current = NULL;
    w->has_current = 0;
    if (w->session) xe_session_anchor_clear(w->session);
}

runtime_status runtime_session_create(runtime *w, const char *system,
                                const profile_tool *tools, size_t tool_count,
                                conversation_id *id, runtime_marker *marker) {
    if (!w || !id) return RUNTIME_INVALID_ARGUMENT;
    if (w->gen_kind != RUNTIME_GEN_NONE)
        return runtime_fail(w, RUNTIME_BUSY, "generation in progress");
    conversation *c = NULL;
    runtime_status status = runtime_from_conversation(
        w, conversation_create(w->cstore, &c, id));
    if (status != RUNTIME_OK) return status;
    runtime_marker system_marker = 0;
    status = runtime_append_system_event(w, c, system, tools, tool_count,
                                      &system_marker);
    if (status == RUNTIME_OK)
        status = runtime_from_conversation(w, conversation_commit(c));
    if (status != RUNTIME_OK) {
        conversation_close(c);
        conversation_delete(w->cstore, id);
        return status;
    }
    runtime_park(w, NULL);
    w->current = c;
    w->current_id = *id;
    w->has_current = 1;
    w->next_call_id = 1;
    w->saved_tokens = 0;
    w->saved_at = 0;
    runtime_ckpt_reset(w);
    if (marker) *marker = system_marker;
    return RUNTIME_OK;
}

static void runtime_scan_call_ids(runtime *w, conversation *c) {
    uint64_t next = 1;
    uint64_t count = conversation_event_count(c);
    for (uint64_t i = 0; i < count; i++) {
        const conversation_event *ev = conversation_event_at(c, i);
        if (ev->type == CONVERSATION_EVENT_TOOL_STARTED &&
            ev->call_id >= next)
            next = ev->call_id + 1;
    }
    w->next_call_id = next;
}

static void runtime_open_report_fill(runtime *w, runtime_open_report *out) {
    conversation *c = w->current;
    uint64_t tokens;
    conversation_tokens(c, &tokens);
    uint64_t visible = conversation_visible_count(c);
    out->token_count = tokens;
    out->marker = visible ? conversation_visible_index(c, visible - 1)
                          : RUNTIME_MARKER_NONE;
    out->turn_open = runtime_turn(c) == PROFILE_TURN_OPEN;
    kvstore_id snapshot;
    uint64_t boundary = 0;
    int has_snapshot = conversation_snapshot_current(c, &snapshot, &boundary);
    out->zero_prefill = has_snapshot && boundary == tokens && tokens > 0;
    out->resume_stale = has_snapshot && !out->zero_prefill && boundary > 0;
    out->pending_calls = conversation_unknown_tool_calls(c, NULL, 0);
}

runtime_status runtime_session_open(runtime *w, const conversation_id *id,
                              runtime_open_report *out) {
    if (!w || !id) return RUNTIME_INVALID_ARGUMENT;
    if (w->gen_kind != RUNTIME_GEN_NONE)
        return runtime_fail(w, RUNTIME_BUSY, "generation in progress");
    if (w->has_current &&
        memcmp(w->current_id.bytes, id->bytes, 16) == 0) {
        uint64_t tokens;
        conversation_tokens(w->current, &tokens);
        if (tokens > (uint64_t)w->context)
            return runtime_context_error(w, tokens);
        if (out) runtime_open_report_fill(w, out);
        return RUNTIME_OK;
    }
    conversation *c = NULL;
    runtime_status status = runtime_from_conversation(
        w, conversation_open(w->cstore, id, &c));
    if (status != RUNTIME_OK) return status;
    conversation_settings settings;
    if (!conversation_get_settings(c, &settings))
        runtime_settings_defaults(&settings);
    status = runtime_from_conversation(
        w, conversation_project(
               c, settings.reasoning_effort != CONVERSATION_REASONING_OFF,
               settings.reasoning_history));
    if (status != RUNTIME_OK) {
        conversation_close(c);
        return status;
    }
    uint64_t tokens;
    conversation_tokens(c, &tokens);
    if (tokens > (uint64_t)w->context) {
        conversation_close(c);
        return runtime_context_error(w, tokens);
    }
    runtime_park(w, NULL);
    w->current = c;
    w->current_id = *id;
    w->has_current = 1;
    runtime_scan_call_ids(w, c);
    kvstore_id snapshot;
    uint64_t boundary = 0;
    w->saved_tokens = conversation_snapshot_current(c, &snapshot, &boundary)
                      ? boundary : 0;
    w->saved_at = 0;
    runtime_ckpt_reset(w);
    if (out) runtime_open_report_fill(w, out);
    return RUNTIME_OK;
}

runtime_status runtime_session_list(runtime *w, conversation_summary **out,
                              size_t *count) {
    if (!w || !out || !count) return RUNTIME_INVALID_ARGUMENT;
    return runtime_from_conversation(
        w, conversation_list(w->cstore, out, count));
}

runtime_status runtime_session_stat(runtime *w, const conversation_id *id,
                              conversation_summary *out) {
    if (!w || !id || !out) return RUNTIME_INVALID_ARGUMENT;
    conversation_summary *all = NULL;
    size_t count = 0;
    runtime_status status = runtime_from_conversation(
        w, conversation_list(w->cstore, &all, &count));
    if (status != RUNTIME_OK) return status;
    status = runtime_fail(w, RUNTIME_SESSION_NOT_FOUND, "session not found");
    for (size_t i = 0; i < count; i++)
        if (memcmp(all[i].id.bytes, id->bytes, 16) == 0) {
            *out = all[i];
            status = RUNTIME_OK;
            break;
        }
    free(all);
    return status;
}

runtime_status runtime_session_delete(runtime *w, const conversation_id *id) {
    if (!w || !id) return RUNTIME_INVALID_ARGUMENT;
    if (w->gen_kind != RUNTIME_GEN_NONE)
        return runtime_fail(w, RUNTIME_BUSY, "generation in progress");
    if (w->has_current &&
        memcmp(w->current_id.bytes, id->bytes, 16) == 0) {
        runtime_shadow_drop(w);
        conversation_close(w->current);
        w->current = NULL;
        w->has_current = 0;
        if (w->session) xe_session_anchor_clear(w->session);
    }
    return runtime_from_conversation(w,
                                  conversation_delete(w->cstore, id));
}

size_t runtime_pending_calls(runtime *w, uint64_t *call_ids, size_t cap) {
    if (!w || !w->has_current) return 0;
    return conversation_unknown_tool_calls(w->current, call_ids, cap);
}

uint64_t runtime_history_count(runtime *w) {
    if (!w || !w->has_current) return 0;
    return conversation_visible_count(w->current);
}

static const char *runtime_tool_name_of(conversation *c, uint64_t call_id) {
    uint64_t count = conversation_event_count(c);
    for (uint64_t i = count; i > 0; i--) {
        const conversation_event *ev = conversation_event_at(c, i - 1);
        if (ev->type == CONVERSATION_EVENT_TOOL_STARTED &&
            ev->call_id == call_id)
            return ev->tool;
    }
    return NULL;
}

runtime_status runtime_history_at(runtime *w, uint64_t position,
                            runtime_history_entry *out) {
    if (!w || !out) return RUNTIME_INVALID_ARGUMENT;
    if (!w->has_current)
        return runtime_fail(w, RUNTIME_SESSION_NOT_FOUND, "no open session");
    conversation *c = w->current;
    if (position >= conversation_visible_count(c))
        return runtime_fail(w, RUNTIME_INVALID_ARGUMENT,
                         "history position out of range");
    uint64_t index = conversation_visible_index(c, position);
    const conversation_event *ev = conversation_event_at(c, index);
    memset(out, 0, sizeof *out);
    out->marker = index;
    if (ev->block_count) {
        out->text = ev->blocks[0].data;
        out->text_length = ev->blocks[0].length;
    }
    if (ev->block_count > 1) {
        out->extra_json = ev->blocks[1].data;
        out->extra_length = ev->blocks[1].length;
    }
    out->reasoning = (const char *)ev->reasoning;
    out->reasoning_length = ev->reasoning_length;
    switch (ev->type) {
    case CONVERSATION_EVENT_MESSAGE:
        out->role = ev->role;
        out->kind = ev->role == CONVERSATION_ROLE_SYSTEM
                    ? RUNTIME_MESSAGE_SYSTEM
                    : ev->role == CONVERSATION_ROLE_ASSISTANT
                    ? RUNTIME_MESSAGE_ASSISTANT : RUNTIME_MESSAGE_USER;
        break;
    case CONVERSATION_EVENT_GENERATION_RESULT:
        out->role = CONVERSATION_ROLE_ASSISTANT;
        out->kind = RUNTIME_MESSAGE_ASSISTANT;
        out->stop_reason = ev->stop_reason;
        break;
    case CONVERSATION_EVENT_TOOL_RESULT:
        out->role = CONVERSATION_ROLE_TOOL;
        out->kind = RUNTIME_MESSAGE_TOOL_RESULT;
        out->call_id = ev->call_id;
        out->tool_status = ev->tool_status;
        out->tool_name = runtime_tool_name_of(c, ev->call_id);
        break;
    default:
        return runtime_fail(w, RUNTIME_INVALID_ARGUMENT, "unexpected event");
    }
    return RUNTIME_OK;
}

runtime_status runtime_append(runtime *w, const runtime_message *message,
                        runtime_marker *out) {
    if (!w || !message) return RUNTIME_INVALID_ARGUMENT;
    if (w->gen_kind != RUNTIME_GEN_NONE)
        return runtime_fail(w, RUNTIME_BUSY, "generation in progress");
    if (!w->has_current)
        return runtime_fail(w, RUNTIME_SESSION_NOT_FOUND, "no open session");
    conversation *c = w->current;
    uint64_t tokens;
    conversation_tokens(c, &tokens);
    profile_render render;
    runtime_status status;
    conversation_block block;
    block.format = CONVERSATION_BLOCK_TEXT;
    block.data = message->text ? message->text : "";
    block.length = message->text ? strlen(message->text) : 0;

    if (message->kind == RUNTIME_MESSAGE_USER) {
        if (!message->text)
            return runtime_fail(w, RUNTIME_INVALID_ARGUMENT, "missing text");
        status = runtime_from_profile(
            w, profile_render_user(w->prof, message->text, runtime_turn(c),
                                   &render));
        if (status != RUNTIME_OK) return status;
        status = runtime_budget_check(w, tokens + render.token_count);
        if (status != RUNTIME_OK) return status;
        status = runtime_from_conversation(
            w, conversation_append_message(c, CONVERSATION_ROLE_USER,
                                           &block, 1, render.render,
                                           render.render_length,
                                           render.tokens,
                                           render.token_count));
    } else if (message->kind == RUNTIME_MESSAGE_TOOL_RESULT) {
        runtime_shadow_poll(w, 1);
        if (!message->text)
            return runtime_fail(w, RUNTIME_INVALID_ARGUMENT, "missing text");
        if (runtime_turn(c) != PROFILE_TURN_OPEN)
            return runtime_fail(w, RUNTIME_INVALID_ARGUMENT,
                             "no open model turn for a tool result");
        const char *name = runtime_tool_name_of(c, message->call_id);
        if (!name)
            return runtime_fail(w, RUNTIME_INVALID_ARGUMENT,
                             "unknown tool call id");
        status = runtime_from_profile(
            w, profile_render_tool_result(w->prof, name, message->text,
                                          &render));
        if (status != RUNTIME_OK) return status;
        status = runtime_budget_check(w, tokens + render.token_count);
        if (status != RUNTIME_OK) return status;
        uint32_t tool_status = message->tool_status
                               ? message->tool_status
                               : CONVERSATION_TOOL_OK;
        status = runtime_from_conversation(
            w, conversation_append_tool_result(c, message->call_id,
                                               tool_status, &block, 1,
                                               render.render,
                                               render.render_length,
                                               render.tokens,
                                               render.token_count));
    } else {
        return runtime_fail(w, RUNTIME_INVALID_ARGUMENT,
                         "unsupported message kind for append");
    }
    if (status != RUNTIME_OK) return status;
    runtime_marker marker = conversation_event_count(c) - 1;
    conversation_settings settings;
    if (!conversation_get_settings(c, &settings))
        runtime_settings_defaults(&settings);
    status = runtime_project_mutation(
        w, c, settings.reasoning_effort != CONVERSATION_REASONING_OFF,
        settings.reasoning_history);
    if (status != RUNTIME_OK) return status;
    status = runtime_from_conversation(w, conversation_commit(c));
    if (status != RUNTIME_OK) return status;
    if (message->kind == RUNTIME_MESSAGE_TOOL_RESULT)
        runtime_shadow_promote_resume(w);
    if (out) *out = marker;
    return RUNTIME_OK;
}

static void runtime_settings_defaults(conversation_settings *settings) {
    settings->temperature = 1.0f;
    settings->top_k = 64;
    settings->top_p = 0.95f;
    settings->max_tokens = 0;
    settings->sampler_abi = CONVERSATION_SAMPLER_ABI;
    settings->rng_seed = 1;
    settings->rng_state = 1;
    settings->reasoning_effort = CONVERSATION_REASONING_OFF;
    settings->reasoning_history = CONVERSATION_REASONING_DISCARD;
    settings->reasoning_budget = -1;
}

static int runtime_params_valid(const runtime_gen_params *params) {
    if (!params) return 1;
    if (params->reasoning_set &&
        params->reasoning_effort > CONVERSATION_REASONING_MAX)
        return 0;
    if (params->reasoning_history_set &&
        params->reasoning_history >
            CONVERSATION_REASONING_PRESERVE_TOOL_CALLS)
        return 0;
    if (params->reasoning_budget_set && params->reasoning_budget < 0)
        return 0;
    return 1;
}

static int runtime_settings_apply(conversation_settings *settings,
                               const runtime_gen_params *params) {
    int changed = 0;
    int effort_changed = 0;
    if (!params) return 0;
    if (params->temperature >= 0.0f &&
        params->temperature != settings->temperature) {
        settings->temperature = params->temperature;
        changed = 1;
    }
    if (params->top_k >= 0 && params->top_k != settings->top_k) {
        settings->top_k = params->top_k;
        changed = 1;
    }
    if (params->top_p > 0.0f && params->top_p != settings->top_p) {
        settings->top_p = params->top_p;
        changed = 1;
    }
    if (params->max_tokens >= 0 &&
        params->max_tokens != settings->max_tokens) {
        settings->max_tokens = params->max_tokens;
        changed = 1;
    }
    if (params->rng_seed) {
        settings->rng_seed = params->rng_seed;
        settings->rng_state = params->rng_seed;
        changed = 1;
    }
    if (params->reasoning_set &&
        params->reasoning_effort != settings->reasoning_effort) {
        settings->reasoning_effort = params->reasoning_effort;
        changed = 1;
        effort_changed = 1;
    }
    if (params->reasoning_history_set &&
        params->reasoning_history != settings->reasoning_history) {
        settings->reasoning_history = params->reasoning_history;
        changed = 1;
    }
    if (params->reasoning_budget_set &&
        params->reasoning_budget != settings->reasoning_budget) {
        settings->reasoning_budget = params->reasoning_budget;
        changed = 1;
    } else if (effort_changed && settings->reasoning_budget != -1) {
        settings->reasoning_budget = -1;
        changed = 1;
    }
    return changed;
}

static runtime_status runtime_prompt_reserve(runtime *w) {
    if (w->prompt) return RUNTIME_OK;
    w->prompt = malloc((size_t)w->context * sizeof(*w->prompt));
    if (!w->prompt) return runtime_fail(w, RUNTIME_NOMEM, "out of memory");
    return RUNTIME_OK;
}

static void runtime_gen_reset(runtime *w, int reasoning_open) {
    w->cancel_requested = 0;
    w->started_emitted = 0;
    w->first_sync_done = 0;
    w->sampled_length = 0;
    w->reasoning_tokens = 0;
    w->reasoning_close = CONVERSATION_REASONING_NONE;
    w->utf8_pending = 0;
    w->replay_from = 0;
    memset(&w->usage, 0, sizeof w->usage);
    memset(&w->inference, 0, sizeof w->inference);
    memset(w->inference_ns, 0, sizeof w->inference_ns);
    w->decode_progress_ns = 0;
    w->inference_active = 0;
    w->prefill_started_emitted = 0;
    w->prefill_complete_emitted = 0;
    w->decode_complete_emitted = 0;
    w->decode_started_emitted = 0;
    w->terminal_pending = 0;
    json_writer_reset(&w->content);
    json_writer_reset(&w->reasoning);
    json_writer_reset(&w->render);
    runtime_calls_reset(w);
    profile_parser_reset(w->prof, reasoning_open);
}

static runtime_status runtime_reasoning_policy_load(runtime *w) {
    uint32_t effort;
    switch (w->gen_settings.reasoning_effort) {
    case CONVERSATION_REASONING_LOW:
        effort = PROFILE_REASONING_LOW;
        break;
    case CONVERSATION_REASONING_MEDIUM:
        effort = PROFILE_REASONING_MEDIUM;
        break;
    case CONVERSATION_REASONING_HIGH:
        effort = PROFILE_REASONING_HIGH;
        break;
    case CONVERSATION_REASONING_MAX:
        effort = PROFILE_REASONING_MAX;
        break;
    default:
        effort = PROFILE_REASONING_OFF;
        break;
    }
    return runtime_from_profile(
        w, profile_get_reasoning_policy(
               w->prof, effort, w->gen_settings.reasoning_budget,
               w->gen_max_tokens, w->context - w->prompt_length,
               &w->reasoning_policy));
}

static void runtime_utf8_feed(runtime *w, const uint8_t *data, size_t length) {
    for (size_t i = 0; i < length; i++) {
        uint8_t byte = data[i];
        if (w->utf8_pending) {
            if ((byte & 0xc0) == 0x80) w->utf8_pending--;
            else w->utf8_pending = 0;
        } else if ((byte & 0xe0) == 0xc0) {
            w->utf8_pending = 1;
        } else if ((byte & 0xf0) == 0xe0) {
            w->utf8_pending = 2;
        } else if ((byte & 0xf8) == 0xf0) {
            w->utf8_pending = 3;
        }
    }
}

static runtime_status runtime_best_snapshot(conversation *c, uint64_t limit,
                                      kvstore_id *id, uint64_t *boundary) {
    uint64_t best = 0;
    int found = 0;
    uint64_t epoch = conversation_epoch_current(c);
    uint64_t count = conversation_event_count(c);
    for (uint64_t i = 0; i < count; i++) {
        const conversation_event *ev = conversation_event_at(c, i);
        if (ev->type != CONVERSATION_EVENT_SNAPSHOT_REF) continue;
        if (ev->epoch != epoch) continue;
        if (ev->snapshot_boundary > limit) continue;
        if (ev->snapshot_boundary < best) continue;
        best = ev->snapshot_boundary;
        *id = ev->snapshot;
        found = 1;
    }
    if (!found) return RUNTIME_SESSION_NOT_FOUND;
    *boundary = best;
    return RUNTIME_OK;
}

static runtime_status runtime_gen_prepare(runtime *w, xe_session *session,
                                    conversation *c) {
    w->gen_session = session;
    w->autosave_attempted = 0;
    w->resume_attempted = 0;
    xe_tokens prompt = { w->prompt, w->prompt_length, w->context };
    int common = xe_session_common(session, &prompt);
    if (c && w->kv) {
        kvstore_id id;
        uint64_t boundary = 0;
        if (runtime_best_snapshot(c, (uint64_t)w->prompt_length, &id,
                               &boundary) == RUNTIME_OK &&
            boundary > (uint64_t)common) {
            xe_tokens expected = { w->prompt, (int)boundary,
                                   (int)boundary };
            xe_snapshot_status snapshot_status;
            kvstore_status ks = kvstore_load(w->kv, &id, session, &expected,
                                             &snapshot_status);
            w->resume_attempted = 1;
            memset(&w->resume, 0, sizeof w->resume);
            w->resume.tokens = boundary;
            if (ks == KVSTORE_OK) {
                common = xe_session_common(session, &prompt);
                w->resume.loaded = 1;
                w->resume.reason = RUNTIME_RESUME_LOADED;
            } else if (ks == KVSTORE_MISS) {
                w->resume.reason = RUNTIME_RESUME_EVICTED;
            } else if (ks == KVSTORE_REJECTED) {
                switch (snapshot_status) {
                case XE_SNAPSHOT_MODEL_MISMATCH:
                    w->resume.reason = RUNTIME_RESUME_MODEL_MISMATCH; break;
                case XE_SNAPSHOT_TOKEN_MISMATCH:
                    w->resume.reason = RUNTIME_RESUME_TOKEN_MISMATCH; break;
                case XE_SNAPSHOT_IO:
                    w->resume.reason = RUNTIME_RESUME_IO; break;
                default:
                    w->resume.reason = RUNTIME_RESUME_REJECTED; break;
                }
            } else {
                w->resume.reason = RUNTIME_RESUME_IO;
            }
        }
    }
    w->inference.prefill.total = (uint64_t)(w->prompt_length - common);
    w->prompt_common = common;
    w->prompt_synced = common;
    w->gen_phase = RUNTIME_PHASE_PREFILL;
    return RUNTIME_OK;
}

runtime_status runtime_generate(runtime *w, const runtime_gen_params *params) {
    if (!w) return RUNTIME_INVALID_ARGUMENT;
    if (w->gen_kind != RUNTIME_GEN_NONE)
        return runtime_fail(w, RUNTIME_BUSY, "generation in progress");
    if (!w->has_current)
        return runtime_fail(w, RUNTIME_SESSION_NOT_FOUND, "no open session");
    if (!runtime_params_valid(params))
        return runtime_fail(w, RUNTIME_INVALID_ARGUMENT,
                         "invalid reasoning settings");
    conversation *c = w->current;

    conversation_settings settings;
    int stored = conversation_get_settings(c, &settings);
    if (!stored) runtime_settings_defaults(&settings);
    if (settings.sampler_abi != CONVERSATION_SAMPLER_ABI)
        return runtime_fail(w, RUNTIME_INVALID_ARGUMENT,
                         "sampler abi mismatch");
    conversation_settings previous = settings;
    int changed = runtime_settings_apply(&settings, params);
    int was_thinking = previous.reasoning_effort !=
                       CONVERSATION_REASONING_OFF;
    int is_thinking = settings.reasoning_effort !=
                      CONVERSATION_REASONING_OFF;
    int projection_changed = stored &&
        (was_thinking != is_thinking ||
         (is_thinking &&
          previous.reasoning_history != settings.reasoning_history));
    if (projection_changed) {
        runtime_shadow_drop(w);
        runtime_status status = runtime_from_conversation(
            w, conversation_append_cache_epoch(c));
        if (status != RUNTIME_OK) return status;
        w->saved_tokens = 0;
        w->saved_at = 0;
        runtime_ckpt_reset(w);
        if (w->session) xe_session_anchor_clear(w->session);
    }
    if (!stored || changed) {
        runtime_status status = runtime_from_conversation(
            w, conversation_append_settings(c, &settings));
        if (status != RUNTIME_OK) return status;
    }

    runtime_status status = runtime_from_conversation(
        w, conversation_project(
               c, settings.reasoning_effort != CONVERSATION_REASONING_OFF,
               settings.reasoning_history));
    if (status != RUNTIME_OK) return status;

    uint64_t tokens;
    const int32_t *projection = conversation_tokens(c, &tokens);
    profile_render render;
    uint32_t turn = runtime_turn(c);
    int after_tool = runtime_after_tool(c);
    int reasoning_open =
        settings.reasoning_effort != CONVERSATION_REASONING_OFF &&
        (runtime_reasoning_continues(c) || after_tool);
    status = runtime_from_profile(
        w, profile_render_reply_open(
               w->prof, turn,
               settings.reasoning_effort != CONVERSATION_REASONING_OFF,
               after_tool, &render));
    if (status != RUNTIME_OK) return status;
    status = runtime_budget_check(w, tokens + render.token_count);
    if (status != RUNTIME_OK) return status;
    status = runtime_prompt_reserve(w);
    if (status != RUNTIME_OK) return status;
    if (tokens) memcpy(w->prompt, projection,
                       (size_t)tokens * sizeof(*w->prompt));
    if (render.token_count)
        memcpy(w->prompt + tokens, render.tokens,
               (size_t)render.token_count * sizeof(*w->prompt));
    w->framing_offset = (int)tokens;
    w->prompt_length = (int)(tokens + render.token_count);

    w->generation_id = conversation_event_count(c);
    conversation_generation generation = { w->generation_id, settings };
    status = runtime_from_conversation(
        w, conversation_append_generation_started(c, &generation));
    if (status == RUNTIME_OK)
        status = runtime_from_conversation(w, conversation_commit(c));
    if (status != RUNTIME_OK) return status;

    w->gen_thinking = settings.reasoning_effort !=
                      CONVERSATION_REASONING_OFF;
    w->gen_after_tool = after_tool;
    w->gen_turn = turn;
    w->capture_anchor = w->gen_thinking && turn == PROFILE_TURN_PADDED;
    runtime_gen_reset(w, reasoning_open);
    if (json_rawn(&w->render, render.render,
                  render.render_length) == 0)
        return runtime_fail(w, RUNTIME_NOMEM, "out of memory");
    w->gen_settings = settings;
    w->sampler.temperature = settings.temperature;
    w->sampler.top_k = settings.top_k;
    w->sampler.top_p = settings.top_p;
    w->sampler.rng_state = settings.rng_state;
    w->gen_max_tokens = settings.max_tokens;
    status = runtime_reasoning_policy_load(w);
    if (status != RUNTIME_OK) return status;
    if (!w->session) w->session = xe_session_new(w->engine);
    w->gen_kind = RUNTIME_GEN_DURABLE;
    if (w->session && xe_session_anchor_valid(w->session))
        w->replay_from = xe_session_anchor_restore(w->session);
    return runtime_gen_prepare(w, w->session, c);
}

runtime_status runtime_ephemeral_generate(runtime *w, const char *system,
                                    const profile_tool *tools,
                                    size_t tool_count,
                                    const runtime_message *messages,
                                    size_t count,
                                    const runtime_gen_params *params) {
    if (!w) return RUNTIME_INVALID_ARGUMENT;
    if (w->gen_kind != RUNTIME_GEN_NONE)
        return runtime_fail(w, RUNTIME_BUSY, "generation in progress");
    if (!runtime_params_valid(params))
        return runtime_fail(w, RUNTIME_INVALID_ARGUMENT,
                         "invalid reasoning settings");
    runtime_status status = runtime_prompt_reserve(w);
    if (status != RUNTIME_OK) return status;

    conversation_settings settings;
    runtime_settings_defaults(&settings);
    runtime_settings_apply(&settings, params);
    int thinking = settings.reasoning_effort !=
                   CONVERSATION_REASONING_OFF;
    profile_render render;
    status = runtime_from_profile(
        w, profile_render_system(w->prof, system, tools, tool_count,
                                 thinking,
                                 &render));
    if (status != RUNTIME_OK) return status;
    int length = 0;
    if ((int)render.token_count >= w->context)
        return runtime_budget_check(w, render.token_count);
    memcpy(w->prompt, render.tokens,
           (size_t)render.token_count * sizeof(*w->prompt));
    length = (int)render.token_count;

    size_t last_user = SIZE_MAX;
    for (size_t i = 0; i < count; i++)
        if (messages[i].kind == RUNTIME_MESSAGE_USER) last_user = i;
    int current_open = count &&
        (messages[count - 1].kind == RUNTIME_MESSAGE_TOOL_RESULT ||
         (messages[count - 1].kind == RUNTIME_MESSAGE_ASSISTANT &&
          messages[count - 1].call_count));
    uint32_t turn = PROFILE_TURN_PADDED;
    for (size_t i = 0; i < count; i++) {
        const runtime_message *m = &messages[i];
        switch (m->kind) {
        case RUNTIME_MESSAGE_USER:
            if (!m->text)
                return runtime_fail(w, RUNTIME_INVALID_ARGUMENT, "missing text");
            status = runtime_from_profile(
                w, profile_render_user(w->prof, m->text, turn, &render));
            turn = PROFILE_TURN_PADDED;
            break;
        case RUNTIME_MESSAGE_ASSISTANT: {
            int current = current_open &&
                          (last_user == SIZE_MAX || i > last_user);
            int historical =
                settings.reasoning_history ==
                    CONVERSATION_REASONING_PRESERVE_TOOL_CALLS &&
                m->call_count;
            status = runtime_from_profile(
                w, profile_render_assistant(
                       w->prof, current || historical ? m->reasoning : NULL,
                       1,
                                            m->text,
                                            m->calls, m->call_count, turn,
                                            1, &render));
            turn = m->call_count ? PROFILE_TURN_OPEN : PROFILE_TURN_BARE;
            break;
        }
        case RUNTIME_MESSAGE_TOOL_RESULT:
            if (turn != PROFILE_TURN_OPEN)
                return runtime_fail(w, RUNTIME_INVALID_ARGUMENT,
                                 "no open model turn for a tool result");
            if (!m->tool_name || !m->text)
                return runtime_fail(w, RUNTIME_INVALID_ARGUMENT,
                                 "missing tool name or text");
            status = runtime_from_profile(
                w, profile_render_tool_result(w->prof, m->tool_name,
                                              m->text, &render));
            break;
        default:
            return runtime_fail(w, RUNTIME_INVALID_ARGUMENT,
                             "unsupported message kind");
        }
        if (status != RUNTIME_OK) return status;
        status = runtime_budget_check(w,
                                   (uint64_t)length + render.token_count);
        if (status != RUNTIME_OK) return status;
        memcpy(w->prompt + length, render.tokens,
               (size_t)render.token_count * sizeof(*w->prompt));
        length += (int)render.token_count;
    }
    status = runtime_from_profile(
        w, profile_render_reply_open(w->prof, turn, thinking,
                                     count && messages[count - 1].kind ==
                                              RUNTIME_MESSAGE_TOOL_RESULT,
                                     &render));
    if (status != RUNTIME_OK) return status;
    status = runtime_budget_check(w, (uint64_t)length + render.token_count);
    if (status != RUNTIME_OK) return status;
    memcpy(w->prompt + length, render.tokens,
           (size_t)render.token_count * sizeof(*w->prompt));
    length += (int)render.token_count;

    w->framing_offset = length;
    w->prompt_length = length;
    w->gen_thinking = thinking;
    w->gen_after_tool = count && messages[count - 1].kind ==
                                 RUNTIME_MESSAGE_TOOL_RESULT;
    w->gen_turn = turn;
    w->capture_anchor = 0;
    runtime_gen_reset(w, w->gen_after_tool && thinking);
    w->gen_settings = settings;
    w->sampler.temperature = settings.temperature;
    w->sampler.top_k = settings.top_k;
    w->sampler.top_p = settings.top_p;
    w->sampler.rng_state = settings.rng_state;
    w->gen_max_tokens = settings.max_tokens;
    status = runtime_reasoning_policy_load(w);
    if (status != RUNTIME_OK) return status;
    if (!w->ephemeral) w->ephemeral = xe_session_new(w->engine);
    w->gen_kind = RUNTIME_GEN_EPHEMERAL;
    return runtime_gen_prepare(w, w->ephemeral, NULL);
}

runtime_status runtime_cancel(runtime *w) {
    if (!w) return RUNTIME_INVALID_ARGUMENT;
    if (w->gen_kind == RUNTIME_GEN_NONE)
        return runtime_fail(w, RUNTIME_INVALID_ARGUMENT,
                         "no generation in progress");
    w->cancel_requested = 1;
    return RUNTIME_OK;
}

static runtime_status runtime_call_register(runtime *w, const char *name) {
    if (w->call_count == w->call_capacity) {
        size_t capacity = w->call_capacity ? w->call_capacity * 2 : 4;
        runtime_gen_call *grown = realloc(w->calls,
                                       capacity * sizeof(*grown));
        if (!grown) return runtime_fail(w, RUNTIME_NOMEM, "out of memory");
        w->calls = grown;
        w->call_capacity = capacity;
    }
    runtime_gen_call *call = &w->calls[w->call_count];
    call->id = w->next_call_id++;
    call->name = strdup(name);
    call->arguments = NULL;
    call->complete = 0;
    if (!call->name) return runtime_fail(w, RUNTIME_NOMEM, "out of memory");
    w->call_open = (int)w->call_count;
    w->call_count++;
    return RUNTIME_OK;
}

static int runtime_render_token(runtime *w, int32_t token) {
    char buffer[512];
    int bytes = xe_detokenize(w->engine, token, buffer, (int)sizeof buffer);
    if (bytes > 0) return json_rawn(&w->render, buffer, (size_t)bytes);
    int piece_length = 0;
    const char *piece = xe_token_piece(w->engine, token, &piece_length);
    if (piece && piece_length > 0)
        return json_rawn(&w->render, piece, (size_t)piece_length);
    return 1;
}

static runtime_status runtime_calls_json(runtime *w) {
    json_writer_reset(&w->calls_scratch);
    json_raw(&w->calls_scratch, "[");
    for (size_t i = 0; i < w->call_count; i++) {
        if (i) json_raw(&w->calls_scratch, ",");
        json_raw(&w->calls_scratch, "{\"id\":");
        json_u64(&w->calls_scratch, w->calls[i].id);
        json_raw(&w->calls_scratch, ",\"name\":");
        json_string(&w->calls_scratch, w->calls[i].name,
                    strlen(w->calls[i].name));
        json_raw(&w->calls_scratch, ",\"arguments\":");
        json_raw(&w->calls_scratch, w->calls[i].arguments
                                    ? w->calls[i].arguments : "{}");
        json_raw(&w->calls_scratch, "}");
    }
    json_raw(&w->calls_scratch, "]");
    if (w->calls_scratch.failed)
        return runtime_fail(w, RUNTIME_NOMEM, "out of memory");
    return RUNTIME_OK;
}

static void runtime_gen_teardown(runtime *w) {
    if (w->gen_kind == RUNTIME_GEN_EPHEMERAL && w->ephemeral) {
        xe_session_free(w->ephemeral);
        w->ephemeral = NULL;
    }
    w->gen_kind = RUNTIME_GEN_NONE;
    w->gen_session = NULL;
}

static runtime_status runtime_gen_error(runtime *w, runtime_status status,
                                  runtime_event *out) {
    runtime_inference_end(w);
    runtime_gen_teardown(w);
    memset(out, 0, sizeof *out);
    out->inference = w->inference;
    out->kind = RUNTIME_EVENT_ERROR;
    out->error = status;
    out->error_text = w->error_text;
    out->error_tokens = w->error_tokens;
    out->error_context = w->error_context;
    return RUNTIME_OK;
}

static runtime_status runtime_generation_render(runtime *w, const char *reasoning,
                                          int reasoning_complete,
                                          int turn_complete,
                                          runtime_render_copy *out) {
    profile_call *calls = NULL;
    if (w->call_count) {
        calls = calloc(w->call_count, sizeof(*calls));
        if (!calls) return runtime_fail(w, RUNTIME_NOMEM, "out of memory");
        for (size_t i = 0; i < w->call_count; i++) {
            calls[i].name = w->calls[i].name;
            calls[i].arguments_json = w->calls[i].arguments
                                      ? w->calls[i].arguments : "{}";
        }
    }
    profile_render render;
    profile_status ps = profile_render_assistant(
        w->prof, reasoning, reasoning_complete,
        w->content.data ? w->content.data : NULL,
        calls, w->call_count, w->gen_turn, turn_complete, &render);
    free(calls);
    runtime_status status = runtime_from_profile(w, ps);
    if (status != RUNTIME_OK) return status;
    if (!runtime_render_copy_set(out, &render))
        return runtime_fail(w, RUNTIME_NOMEM, "out of memory");
    return RUNTIME_OK;
}

static runtime_status runtime_project_mutation(runtime *w, conversation *c,
                                         int thinking, uint32_t history) {
    int had_snapshot = conversation_snapshot_current(c, NULL, NULL);
    runtime_status status = runtime_from_conversation(
        w, conversation_project(c, thinking, history));
    if (status != RUNTIME_OK) return status;
    if (had_snapshot && !conversation_snapshot_current(c, NULL, NULL)) {
        status = runtime_from_conversation(
            w, conversation_append_cache_epoch(c));
        if (status == RUNTIME_OK) {
            w->saved_tokens = 0;
            w->saved_at = 0;
            runtime_ckpt_reset(w);
        }
        return status;
    }
    return RUNTIME_OK;
}

static runtime_status runtime_finalize(runtime *w, uint32_t runtime_stop_reason,
                                 uint32_t record_stop, runtime_event *out) {
    if (w->reasoning_close == CONVERSATION_REASONING_NONE &&
        profile_parser_reasoning(w->prof)) {
        if (record_stop == CONVERSATION_STOP_LIMIT)
            w->reasoning_close = CONVERSATION_REASONING_LENGTH;
        else if (record_stop == CONVERSATION_STOP_CANCELLED)
            w->reasoning_close = CONVERSATION_REASONING_ABORTED;
        else if (record_stop == CONVERSATION_STOP_EOS)
            w->reasoning_close = CONVERSATION_REASONING_EOS;
    }
    if (w->call_open >= 0 && !w->calls[w->call_open].complete) {
        free(w->calls[w->call_count - 1].name);
        free(w->calls[w->call_count - 1].arguments);
        w->call_count--;
        w->call_open = -1;
    }
    w->usage.output = (uint64_t)w->sampled_length;
    w->usage.reasoning = (uint64_t)w->reasoning_tokens;
    runtime_marker marker = RUNTIME_MARKER_NONE;
    int shadow_promoted = 0;
    int main_reconciled = 0;

    if (w->gen_kind == RUNTIME_GEN_DURABLE) {
        conversation *c = w->current;
        conversation_block blocks[2];
        uint32_t block_count = 1;
        blocks[0].format = CONVERSATION_BLOCK_TEXT;
        blocks[0].data = w->content.data ? w->content.data : "";
        blocks[0].length = w->content.length;
        if (w->call_count) {
            runtime_status status = runtime_calls_json(w);
            if (status != RUNTIME_OK) return runtime_gen_error(w, status, out);
            blocks[1].format = CONVERSATION_BLOCK_JSON;
            blocks[1].data = w->calls_scratch.data;
            blocks[1].length = w->calls_scratch.length;
            block_count = 2;
        }
        int turn_complete = record_stop == CONVERSATION_STOP_EOT_SAMPLED ||
                            record_stop == CONVERSATION_STOP_EOT_SYNTHETIC ||
                            record_stop == CONVERSATION_STOP_EOS;
        int reasoning_complete =
            w->reasoning_close == CONVERSATION_REASONING_NATURAL ||
            w->reasoning_close == CONVERSATION_REASONING_SOFT ||
            w->reasoning_close == CONVERSATION_REASONING_HARD;
        runtime_render_copy primary;
        runtime_status status = runtime_generation_render(
            w, NULL, 1, turn_complete, &primary);
        if (status != RUNTIME_OK) return runtime_gen_error(w, status, out);
        runtime_render_copy alternate;
        memset(&alternate, 0, sizeof alternate);
        if (w->gen_thinking) {
            status = runtime_generation_render(
                w, w->reasoning.length ? w->reasoning.data : NULL,
                reasoning_complete, turn_complete, &alternate);
            if (status != RUNTIME_OK) {
                runtime_render_copy_free(&primary);
                return runtime_gen_error(w, status, out);
            }
        }
        const int32_t *raw_tokens = w->prompt + w->framing_offset;
        uint32_t raw_count = (uint32_t)(w->prompt_length -
                                        w->framing_offset +
                                        w->sampled_length);
        status = runtime_from_conversation(
            w, conversation_append_generation_result_variants(
                   c, w->generation_id, record_stop, w->sampler.rng_state,
                   blocks, block_count,
                   primary.render, primary.render_length,
                   primary.tokens, primary.token_count,
                   alternate.render, alternate.render_length,
                   alternate.tokens, alternate.token_count,
                   w->reasoning.length ? w->reasoning.data : NULL,
                   w->reasoning.length, w->reasoning_close,
                   w->render.data ? w->render.data : "", w->render.length,
                   raw_tokens, raw_count));
        runtime_render_copy_free(&alternate);
        runtime_render_copy_free(&primary);
        if (status != RUNTIME_OK) return runtime_gen_error(w, status, out);
        marker = conversation_event_count(c) - 1;
        for (size_t i = 0; i < w->call_count; i++) {
            conversation_tool_call call;
            memset(&call, 0, sizeof call);
            call.call_id = w->calls[i].id;
            call.server = "";
            call.tool = w->calls[i].name;
            call.arguments = (const uint8_t *)(w->calls[i].arguments
                                               ? w->calls[i].arguments
                                               : "{}");
            call.arguments_length = strlen((const char *)call.arguments);
            format_sha256 hasher;
            format_sha256_init(&hasher);
            format_sha256_update(&hasher, call.tool, strlen(call.tool));
            format_sha256_update(&hasher, call.arguments,
                                 call.arguments_length);
            format_sha256_final(&hasher, call.fingerprint);
            status = runtime_from_conversation(
                w, conversation_append_tool_started(c, &call));
            if (status != RUNTIME_OK) return runtime_gen_error(w, status, out);
        }
        status = runtime_project_mutation(
            w, c, w->gen_thinking,
            w->gen_settings.reasoning_history);
        if (status != RUNTIME_OK) return runtime_gen_error(w, status, out);
        status = runtime_from_conversation(w, conversation_commit(c));
        if (status != RUNTIME_OK) return runtime_gen_error(w, status, out);
        uint64_t total;
        const int32_t *projected = conversation_tokens(c, &total);
        w->usage.total = total;
        int open = record_stop == CONVERSATION_STOP_TOOL_CALLS ||
                   record_stop == CONVERSATION_STOP_LIMIT ||
                   record_stop == CONVERSATION_STOP_CANCELLED;
        if (!open && w->session && w->gen_after_tool) {
            xe_tokens prefix = { (int32_t *)projected, (int)total,
                                 (int)total };
            int common = xe_session_common(w->session, &prefix);
            int shadow_position = w->shadow
                                  ? xe_session_position(w->shadow) : 0;
            if (common > shadow_position) {
                xe_sync_report report;
                xe_session_sync_report(w->session, &prefix, &report);
                w->usage.input += (uint64_t)report.prefilled;
                w->usage.replayed += (uint64_t)report.prefilled;
                runtime_shadow_drop(w);
                xe_session_anchor_clear(w->session);
                main_reconciled = 1;
            }
        }
        if (open && record_stop == CONVERSATION_STOP_TOOL_CALLS &&
            w->shadow) {
            runtime_shadow_poll(w, 1);
            w->shadow_cooldown = 0;
            if (runtime_shadow_tick(w) != RUNTIME_OK)
                runtime_shadow_drop(w);
        }
        if (open && record_stop != CONVERSATION_STOP_TOOL_CALLS)
            runtime_shadow_drop(w);
        if (!open && w->session && w->shadow) {
            int64_t started = runtime_monotonic();
            runtime_shadow_poll(w, 1);
            int position = xe_session_position(w->shadow);
            if (position <= (int)total)
                w->usage.shadow_remaining = total - (uint64_t)position;
            runtime_shadow_wait(w, 0);
            xe_tokens prefix = { (int32_t *)projected, (int)total,
                                 (int)total };
            int prefilled = xe_session_shadow_sync(w->shadow, &prefix);
            if (prefilled >= 0) {
                runtime_shadow_record(w, prefilled, 0);
                xe_session_shadow_refresh_logits(w->shadow);
                uint64_t bytes = xe_session_shadow_kv_bytes(w->shadow);
                if (bytes > w->shadow_peak_kv)
                    w->shadow_peak_kv = bytes;
                shadow_promoted = xe_session_shadow_promote(
                    w->session, w->shadow);
            }
            int64_t finished = runtime_monotonic();
            if (shadow_promoted) {
                if (finished > started)
                    w->usage.shadow_wait_us =
                        (uint64_t)(finished - started) / 1000;
                w->usage.shadow_prefilled = w->shadow_prefilled;
                w->usage.shadow_background = w->shadow_background;
                w->usage.shadow_kv_bytes = w->shadow_peak_kv;
                xe_session_free(w->shadow);
                w->shadow = NULL;
                w->shadow_inflight = 0;
            } else {
                runtime_shadow_drop(w);
            }
        }
        if (!open && w->session && !shadow_promoted && !main_reconciled) {
            int restored = xe_session_anchor_restore(w->session);
            if (restored) {
                xe_tokens prefix = { (int32_t *)projected, (int)total,
                                     (int)total };
                xe_sync_report report;
                xe_session_sync_report(w->session, &prefix, &report);
                w->usage.input += (uint64_t)report.prefilled;
                w->usage.replayed += (uint64_t)report.prefilled;
                xe_session_anchor_clear(w->session);
            }
        }
        if (!open && w->kv && !w->ckpt_autosave_off &&
            conversation_autosave_due(total, w->saved_tokens,
                                      w->saved_at, runtime_now())) {
            w->autosave_attempted = 1;
            runtime_checkpoint_now(w, &w->autosave);
        }
        if (open) {
            w->usage.shadow_prefilled = w->shadow_prefilled;
            w->usage.shadow_background = w->shadow_background;
            w->usage.shadow_kv_bytes = w->shadow_peak_kv;
        }
    } else {
        w->usage.total = (uint64_t)(w->prompt_length + w->sampled_length);
    }

    runtime_usage usage = w->usage;
    if (shadow_promoted) {
        w->shadow_prefilled = 0;
        w->shadow_background = 0;
        w->shadow_peak_kv = 0;
    }
    runtime_gen_teardown(w);
    memset(out, 0, sizeof *out);
    out->inference = w->inference;
    out->kind = RUNTIME_EVENT_DONE;
    out->stop = runtime_stop_reason;
    out->reasoning_close = w->reasoning_close;
    out->usage = usage;
    out->marker = marker;
    out->checkpoint_attempted = w->autosave_attempted;
    out->checkpoint = w->autosave;
    out->resume_attempted = w->resume_attempted;
    out->resume = w->resume;
    return RUNTIME_OK;
}

/* Yield terminal phase measurements before expensive persistence. The stop
 * token has already been consumed, so subsequent calls never sample it twice. */
static runtime_status runtime_complete(runtime *w, uint32_t stop,
                                       uint32_t record, runtime_event *out) {
    runtime_inference_end(w);
    w->terminal_pending = 1;
    w->terminal_stop = stop;
    w->terminal_record = record;
    if (w->prefill_started_emitted && !w->prefill_complete_emitted) {
        w->prefill_complete_emitted = 1;
        return runtime_inference_progress(w, RUNTIME_INFERENCE_PREFILL,
                                          RUNTIME_INFERENCE_FINISHED, out);
    }
    if (w->decode_started_emitted && !w->decode_complete_emitted) {
        w->decode_complete_emitted = 1;
        return runtime_inference_progress(w, RUNTIME_INFERENCE_DECODE,
                                          RUNTIME_INFERENCE_FINISHED, out);
    }
    return runtime_finalize(w, stop, record, out);
}

static runtime_status runtime_next_event_impl(runtime *w, runtime_event *out) {
    if (!w || !out) return RUNTIME_INVALID_ARGUMENT;
    if (w->gen_kind == RUNTIME_GEN_NONE)
        return runtime_fail(w, RUNTIME_INVALID_ARGUMENT,
                         "no generation in progress");
    memset(out, 0, sizeof *out);

    if (!w->started_emitted) {
        w->started_emitted = 1;
        out->kind = RUNTIME_EVENT_START;
        return RUNTIME_OK;
    }

    for (;;) {
        if (w->terminal_pending)
            return runtime_complete(w, w->terminal_stop, w->terminal_record, out);
        if (w->cancel_requested)
            return runtime_complete(w, RUNTIME_STOP_ABORTED,
                                 CONVERSATION_STOP_CANCELLED, out);

        if (w->inference_ns[1] - w->decode_progress_ns >= UINT64_C(250000000))
            return runtime_inference_progress(w, RUNTIME_INFERENCE_DECODE,
                                              RUNTIME_INFERENCE_RUNNING, out);

        if (w->gen_phase == RUNTIME_PHASE_PREFILL) {
            if (!w->prefill_started_emitted) {
                runtime_shadow_wait(w, 1);
                w->prefill_started_emitted = 1;
                return runtime_inference_progress(w, RUNTIME_INFERENCE_PREFILL,
                                                  RUNTIME_INFERENCE_RUNNING, out);
            }
            int target = w->prompt_synced + RUNTIME_PREFILL_CHUNK;
            if (target > w->prompt_length) target = w->prompt_length;
            if (!w->first_sync_done || w->prompt_synced < w->prompt_length) {
                runtime_shadow_wait(w, 1);
                xe_tokens prefix = { w->prompt, target, w->context };
                xe_sync_report report;
                runtime_inference_begin(w, RUNTIME_INFERENCE_PREFILL);
                xe_session_sync_report(w->gen_session, &prefix, &report);
                runtime_inference_end(w);
                if (!w->first_sync_done) {
                    w->usage.cache_read = (uint64_t)report.reused;
                    w->first_sync_done = 1;
                }
                w->inference.prefill.tokens += (uint64_t)report.prefilled;
                w->inference.prefill.total = w->inference.prefill.tokens +
                    (uint64_t)(w->prompt_length - target);
                w->usage.input += (uint64_t)report.prefilled;
                if (w->replay_from)
                    w->usage.replayed += (uint64_t)report.prefilled;
                w->prompt_synced = target;
                if (w->prompt_synced < w->prompt_length)
                    return runtime_inference_progress(w, RUNTIME_INFERENCE_PREFILL,
                                                      RUNTIME_INFERENCE_RUNNING, out);
            }
            w->gen_phase = RUNTIME_PHASE_DECODE;
            w->replay_from = 0;
            w->prefill_complete_emitted = 1;
            return runtime_inference_progress(w, RUNTIME_INFERENCE_PREFILL,
                                              RUNTIME_INFERENCE_FINISHED, out);
        }

        if (!w->decode_started_emitted) {
            /* Between the closed prefill and the not-yet-open decode. These
             * setup costs belong to neither phase's active measurements. */
            if (w->capture_anchor) {
                xe_session_anchor_capture(w->gen_session);
                w->capture_anchor = 0;
                runtime_shadow_begin(w);
            }
            runtime_status shadow_status = runtime_shadow_tick(w);
            if (shadow_status != RUNTIME_OK)
                return runtime_gen_error(w, shadow_status, out);
            w->decode_started_emitted = 1;
            return runtime_inference_progress(w, RUNTIME_INFERENCE_DECODE,
                                              RUNTIME_INFERENCE_RUNNING, out);
        }

        runtime_inference_begin(w, RUNTIME_INFERENCE_DECODE);
        runtime_status shadow_status = runtime_shadow_tick(w);
        if (shadow_status != RUNTIME_OK)
            return runtime_gen_error(w, shadow_status, out);

        if (w->gen_max_tokens > 0 &&
            w->sampled_length >= w->gen_max_tokens)
            return runtime_complete(w, RUNTIME_STOP_LENGTH,
                                 CONVERSATION_STOP_LIMIT, out);
        if (w->prompt_length + w->sampled_length >= w->context - 1)
            return runtime_complete(w, RUNTIME_STOP_LENGTH,
                                 CONVERSATION_STOP_LIMIT, out);

        uint32_t forced_close = CONVERSATION_REASONING_NONE;
        if (w->reasoning_policy.hard_tokens >= 0 &&
            profile_parser_reasoning(w->prof) && !w->utf8_pending) {
            if (w->reasoning_tokens >=
                w->reasoning_policy.hard_tokens) {
                forced_close = CONVERSATION_REASONING_HARD;
            } else if (w->reasoning_tokens >=
                           w->reasoning_policy.soft_tokens &&
                       xe_session_token_near_top(
                           w->gen_session,
                           profile_reasoning_end_token(w->prof),
                           w->reasoning_policy.delimiter_rank,
                           w->reasoning_policy.delimiter_margin)) {
                forced_close = CONVERSATION_REASONING_SOFT;
            }
        }
        int32_t token = forced_close != CONVERSATION_REASONING_NONE
                        ? profile_reasoning_end_token(w->prof)
                        : xe_session_next(w->gen_session, &w->sampler);
        int was_reasoning = profile_parser_reasoning(w->prof);
        profile_parse_event event;
        profile_status parsed = profile_parser_feed(w->prof, token, &event);
        if (parsed != PROFILE_OK)
            return runtime_gen_error(
                w, runtime_fail(w, RUNTIME_NOMEM, "out of memory"), out);
        if (was_reasoning && !profile_parser_reasoning(w->prof))
            w->reasoning_close = forced_close !=
                                 CONVERSATION_REASONING_NONE
                                 ? forced_close
                                 : CONVERSATION_REASONING_NATURAL;

        if (event.kind == PROFILE_PARSE_STOP) {
            if (event.include_token) {
                w->prompt[w->prompt_length + w->sampled_length] = token;
                w->sampled_length++;
                if (!runtime_render_token(w, token))
                    return runtime_gen_error(
                        w, runtime_fail(w, RUNTIME_NOMEM, "out of memory"), out);
            }
            switch (event.stop_reason) {
            case PROFILE_STOP_TOOL_CALLS:
                if (w->call_count)
                    return runtime_complete(w, RUNTIME_STOP_TOOL_USE,
                                         CONVERSATION_STOP_TOOL_CALLS, out);
                return runtime_complete(w, RUNTIME_STOP_STOP,
                                     CONVERSATION_STOP_TOOL_CALLS, out);
            case PROFILE_STOP_EOS:
                return runtime_complete(w, RUNTIME_STOP_STOP,
                                     CONVERSATION_STOP_EOS, out);
            default:
                return runtime_complete(w, RUNTIME_STOP_STOP,
                                     CONVERSATION_STOP_EOT_SAMPLED, out);
            }
        }

        w->prompt[w->prompt_length + w->sampled_length] = token;
        w->sampled_length++;
        if (!runtime_render_token(w, token))
            return runtime_gen_error(
                w, runtime_fail(w, RUNTIME_NOMEM, "out of memory"), out);
        xe_tokens prefix = { w->prompt,
                             w->prompt_length + w->sampled_length,
                             w->context };
        xe_session_sync(w->gen_session, &prefix);

        switch (event.kind) {
        case PROFILE_PARSE_REASONING:
            if (!json_rawn(&w->reasoning, event.text, event.text_length))
                return runtime_gen_error(
                    w, runtime_fail(w, RUNTIME_NOMEM, "out of memory"), out);
            runtime_utf8_feed(w, event.text, event.text_length);
            w->reasoning_tokens++;
            out->kind = RUNTIME_EVENT_REASONING_DELTA;
            out->text = event.text;
            out->text_length = event.text_length;
            return RUNTIME_OK;
        case PROFILE_PARSE_TEXT:
            if (!json_rawn(&w->content, event.text, event.text_length))
                return runtime_gen_error(
                    w, runtime_fail(w, RUNTIME_NOMEM, "out of memory"), out);
            out->kind = RUNTIME_EVENT_TEXT_DELTA;
            out->text = event.text;
            out->text_length = event.text_length;
            return RUNTIME_OK;
        case PROFILE_PARSE_CALL_START: {
            runtime_status status = runtime_call_register(w, event.call_name);
            if (status != RUNTIME_OK) return runtime_gen_error(w, status, out);
            out->kind = RUNTIME_EVENT_TOOLCALL_START;
            out->call_id = w->calls[w->call_count - 1].id;
            out->call_name = w->calls[w->call_count - 1].name;
            return RUNTIME_OK;
        }
        case PROFILE_PARSE_CALL_END: {
            if (w->call_open < 0)
                return runtime_gen_error(
                    w, runtime_fail(w, RUNTIME_INVALID_ARGUMENT,
                                 "call end without start"), out);
            runtime_gen_call *call = &w->calls[w->call_open];
            call->arguments = strdup(event.arguments_json
                                     ? event.arguments_json : "{}");
            call->complete = 1;
            w->call_open = -1;
            if (!call->arguments)
                return runtime_gen_error(
                    w, runtime_fail(w, RUNTIME_NOMEM, "out of memory"), out);
            out->kind = RUNTIME_EVENT_TOOLCALL_END;
            out->call_id = call->id;
            out->call_name = call->name;
            out->arguments_json = call->arguments;
            return RUNTIME_OK;
        }
        default:
            runtime_inference_end(w);
            continue;
        }
    }
}

runtime_status runtime_next_event(runtime *w, runtime_event *out) {
    runtime_status status = runtime_next_event_impl(w, out);
    if (w) runtime_inference_end(w);
    return status;
}

runtime_status runtime_rewind(runtime *w, runtime_marker marker) {
    if (!w) return RUNTIME_INVALID_ARGUMENT;
    if (w->gen_kind != RUNTIME_GEN_NONE)
        return runtime_fail(w, RUNTIME_BUSY, "generation in progress");
    if (!w->has_current)
        return runtime_fail(w, RUNTIME_SESSION_NOT_FOUND, "no open session");
    conversation *c = w->current;
    uint64_t position;
    if (!conversation_visible_position(c, marker, &position))
        return runtime_fail(w, RUNTIME_MARKER_UNAVAILABLE,
                         "marker is not addressable");
    if (position + 1 == conversation_visible_count(c)) return RUNTIME_OK;
    runtime_shadow_drop(w);
    runtime_status status = runtime_from_conversation(
        w, conversation_append_rewind(c, marker));
    if (status != RUNTIME_OK) return status;
    status = runtime_from_conversation(w, conversation_commit(c));
    if (status != RUNTIME_OK) return status;
    if (w->session) xe_session_anchor_clear(w->session);
    kvstore_id snapshot;
    uint64_t boundary = 0;
    w->saved_tokens = conversation_snapshot_current(c, &snapshot, &boundary)
                      ? boundary : 0;
    runtime_ckpt_reset(w);
    return RUNTIME_OK;
}

runtime_status runtime_rewind_cost(runtime *w, runtime_marker marker,
                             uint64_t *prefill_tokens) {
    if (!w || !prefill_tokens) return RUNTIME_INVALID_ARGUMENT;
    if (!w->has_current)
        return runtime_fail(w, RUNTIME_SESSION_NOT_FOUND, "no open session");
    conversation *c = w->current;
    uint64_t position;
    if (!conversation_visible_position(c, marker, &position))
        return runtime_fail(w, RUNTIME_MARKER_UNAVAILABLE,
                         "marker is not addressable");
    uint64_t boundary = conversation_visible_boundary(c, position);
    if (w->session && boundary) {
        uint64_t total;
        const int32_t *tokens = conversation_tokens(c, &total);
        if (boundary <= total) {
            xe_tokens prefix = { (int32_t *)tokens, (int)boundary,
                                 (int)boundary };
            int common = xe_session_common(w->session, &prefix);
            int frontier = xe_session_position(w->session);
            if ((uint64_t)common == boundary &&
                (uint64_t)frontier >= boundary &&
                (uint64_t)frontier - boundary <= 1024) {
                *prefill_tokens = 0;
                return RUNTIME_OK;
            }
        }
    }
    kvstore_id id;
    uint64_t snapshot_boundary = 0;
    if (runtime_best_snapshot(c, boundary, &id, &snapshot_boundary)
            == RUNTIME_OK) {
        *prefill_tokens = boundary - snapshot_boundary;
        return RUNTIME_OK;
    }
    *prefill_tokens = boundary;
    return RUNTIME_OK;
}

runtime_status runtime_rebuild(runtime *w, const char *system,
                         const profile_tool *tools, size_t tool_count,
                         const runtime_message *messages, size_t count,
                         runtime_marker *out) {
    if (!w) return RUNTIME_INVALID_ARGUMENT;
    if (w->gen_kind != RUNTIME_GEN_NONE)
        return runtime_fail(w, RUNTIME_BUSY, "generation in progress");
    if (!w->has_current)
        return runtime_fail(w, RUNTIME_SESSION_NOT_FOUND, "no open session");
    conversation *c = w->current;
    runtime_shadow_drop(w);
    if (w->session) xe_session_anchor_clear(w->session);
    runtime_status status = runtime_from_conversation(
        w, conversation_append_rewind(c, CONVERSATION_REWIND_ALL));
    if (status != RUNTIME_OK) return status;
    status = runtime_from_conversation(w, conversation_append_cache_epoch(c));
    if (status != RUNTIME_OK) goto abort;
    status = runtime_append_system_event(w, c, system, tools, tool_count,
                                      NULL);
    if (status != RUNTIME_OK) goto abort;

    uint32_t turn = PROFILE_TURN_PADDED;
    profile_render render;
    for (size_t i = 0; i < count; i++) {
        const runtime_message *m = &messages[i];
        uint64_t tokens;
        conversation_tokens(c, &tokens);
        conversation_block blocks[2];
        uint32_t block_count = 1;
        blocks[0].format = CONVERSATION_BLOCK_TEXT;
        blocks[0].data = m->text ? m->text : "";
        blocks[0].length = m->text ? strlen(m->text) : 0;
        switch (m->kind) {
        case RUNTIME_MESSAGE_USER:
            if (!m->text) {
                status = runtime_fail(w, RUNTIME_INVALID_ARGUMENT,
                                   "missing text");
                goto abort;
            }
            status = runtime_from_profile(
                w, profile_render_user(w->prof, m->text, turn, &render));
            if (status != RUNTIME_OK) goto abort;
            status = runtime_budget_check(w, tokens + render.token_count);
            if (status != RUNTIME_OK) goto abort;
            status = runtime_from_conversation(
                w, conversation_append_message(c, CONVERSATION_ROLE_USER,
                                               blocks, 1, render.render,
                                               render.render_length,
                                               render.tokens,
                                               render.token_count));
            if (status != RUNTIME_OK) goto abort;
            turn = PROFILE_TURN_PADDED;
            break;
        case RUNTIME_MESSAGE_ASSISTANT: {
            status = runtime_from_profile(
                w, profile_render_assistant(w->prof, NULL, 1, m->text,
                                            m->calls, m->call_count, turn,
                                            1, &render));
            if (status != RUNTIME_OK) goto abort;
            runtime_render_copy primary;
            if (!runtime_render_copy_set(&primary, &render)) {
                status = runtime_fail(w, RUNTIME_NOMEM, "out of memory");
                goto abort;
            }
            runtime_render_copy alternate;
            memset(&alternate, 0, sizeof alternate);
            if (m->reasoning && *m->reasoning) {
                status = runtime_from_profile(
                    w, profile_render_assistant(
                           w->prof, m->reasoning, 1, m->text,
                           m->calls, m->call_count, turn, 1, &render));
                if (status != RUNTIME_OK ||
                    !runtime_render_copy_set(&alternate, &render)) {
                    runtime_render_copy_free(&primary);
                    if (status == RUNTIME_OK)
                        status = runtime_fail(w, RUNTIME_NOMEM, "out of memory");
                    goto abort;
                }
            }
            status = runtime_budget_check(w, tokens + primary.token_count);
            if (status != RUNTIME_OK) {
                runtime_render_copy_free(&alternate);
                runtime_render_copy_free(&primary);
                goto abort;
            }
            runtime_calls_reset(w);
            for (size_t j = 0; j < m->call_count; j++) {
                status = runtime_call_register(w, m->calls[j].name);
                if (status != RUNTIME_OK) {
                    runtime_render_copy_free(&alternate);
                    runtime_render_copy_free(&primary);
                    goto abort;
                }
                w->calls[j].arguments = strdup(
                    m->calls[j].arguments_json
                    ? m->calls[j].arguments_json : "{}");
                w->calls[j].complete = 1;
                if (!w->calls[j].arguments) {
                    runtime_render_copy_free(&alternate);
                    runtime_render_copy_free(&primary);
                    status = runtime_fail(w, RUNTIME_NOMEM, "out of memory");
                    goto abort;
                }
            }
            w->call_open = -1;
            if (m->call_count) {
                status = runtime_calls_json(w);
                if (status != RUNTIME_OK) {
                    runtime_render_copy_free(&alternate);
                    runtime_render_copy_free(&primary);
                    goto abort;
                }
                blocks[1].format = CONVERSATION_BLOCK_JSON;
                blocks[1].data = w->calls_scratch.data;
                blocks[1].length = w->calls_scratch.length;
                block_count = 2;
            }
            status = runtime_from_conversation(
                w, conversation_append_message_variants(
                       c, CONVERSATION_ROLE_ASSISTANT, blocks, block_count,
                       primary.render, primary.render_length,
                       primary.tokens, primary.token_count,
                       alternate.render, alternate.render_length,
                       alternate.tokens, alternate.token_count,
                       m->reasoning && *m->reasoning ? m->reasoning : NULL,
                       m->reasoning ? strlen(m->reasoning) : 0,
                       m->reasoning && *m->reasoning
                       ? CONVERSATION_REASONING_NATURAL
                       : CONVERSATION_REASONING_NONE));
            runtime_render_copy_free(&alternate);
            runtime_render_copy_free(&primary);
            if (status != RUNTIME_OK) goto abort;
            for (size_t j = 0; j < w->call_count; j++) {
                conversation_tool_call call;
                memset(&call, 0, sizeof call);
                call.call_id = w->calls[j].id;
                call.server = "";
                call.tool = w->calls[j].name;
                call.arguments = (const uint8_t *)w->calls[j].arguments;
                call.arguments_length = strlen(w->calls[j].arguments);
                format_sha256 hasher;
                format_sha256_init(&hasher);
                format_sha256_update(&hasher, call.tool,
                                     strlen(call.tool));
                format_sha256_update(&hasher, call.arguments,
                                     call.arguments_length);
                format_sha256_final(&hasher, call.fingerprint);
                status = runtime_from_conversation(
                    w, conversation_append_tool_started(c, &call));
                if (status != RUNTIME_OK) goto abort;
            }
            turn = m->call_count ? PROFILE_TURN_OPEN : PROFILE_TURN_BARE;
            break;
        }
        case RUNTIME_MESSAGE_TOOL_RESULT: {
            if (turn != PROFILE_TURN_OPEN) {
                status = runtime_fail(w, RUNTIME_INVALID_ARGUMENT,
                                   "no open model turn for a tool result");
                goto abort;
            }
            if (!m->text) {
                status = runtime_fail(w, RUNTIME_INVALID_ARGUMENT,
                                   "missing text");
                goto abort;
            }
            uint64_t pending[64];
            size_t pending_count = conversation_unknown_tool_calls(
                c, pending, sizeof pending / sizeof pending[0]);
            if (!pending_count) {
                status = runtime_fail(w, RUNTIME_INVALID_ARGUMENT,
                                   "no pending tool call");
                goto abort;
            }
            uint64_t call_id = pending[0];
            const char *name = runtime_tool_name_of(c, call_id);
            if (m->tool_name) {
                size_t k = 0;
                for (; k < pending_count; k++) {
                    const char *candidate = runtime_tool_name_of(c,
                                                              pending[k]);
                    if (candidate && strcmp(candidate,
                                            m->tool_name) == 0) {
                        call_id = pending[k];
                        name = candidate;
                        break;
                    }
                }
                if (k == pending_count) {
                    status = runtime_fail(w, RUNTIME_INVALID_ARGUMENT,
                                       "no pending call for that tool");
                    goto abort;
                }
            }
            if (!name) {
                status = runtime_fail(w, RUNTIME_INVALID_ARGUMENT,
                                   "unknown tool call");
                goto abort;
            }
            status = runtime_from_profile(
                w, profile_render_tool_result(w->prof, name, m->text,
                                              &render));
            if (status != RUNTIME_OK) goto abort;
            status = runtime_budget_check(w, tokens + render.token_count);
            if (status != RUNTIME_OK) goto abort;
            uint32_t tool_status = m->tool_status ? m->tool_status
                                                  : CONVERSATION_TOOL_OK;
            status = runtime_from_conversation(
                w, conversation_append_tool_result(
                       c, call_id, tool_status, blocks, 1, render.render,
                       render.render_length, render.tokens,
                       render.token_count));
            if (status != RUNTIME_OK) goto abort;
            break;
        }
        default:
            status = runtime_fail(w, RUNTIME_INVALID_ARGUMENT,
                               "unsupported message kind");
            goto abort;
        }
    }
    conversation_settings settings;
    if (!conversation_get_settings(c, &settings))
        runtime_settings_defaults(&settings);
    status = runtime_from_conversation(
        w, conversation_project(
               c, settings.reasoning_effort != CONVERSATION_REASONING_OFF,
               settings.reasoning_history));
    if (status != RUNTIME_OK) goto abort;
    status = runtime_from_conversation(w, conversation_commit(c));
    if (status != RUNTIME_OK) goto abort;
    w->saved_tokens = 0;
    w->saved_at = 0;
    runtime_ckpt_reset(w);
    if (out) *out = conversation_event_count(c) - 1;
    return RUNTIME_OK;

abort:
    {
        conversation_status rollback = conversation_rollback(c);
        if (rollback != CONVERSATION_OK)
            return runtime_from_conversation(w, rollback);
    }
    return status;
}

runtime_status runtime_checkpoint(runtime *w, runtime_checkpoint_report *out) {
    if (!w) return RUNTIME_INVALID_ARGUMENT;
    if (w->gen_kind != RUNTIME_GEN_NONE)
        return runtime_fail(w, RUNTIME_BUSY, "generation in progress");
    if (!w->has_current)
        return runtime_fail(w, RUNTIME_SESSION_NOT_FOUND, "no open session");
    runtime_shadow_wait(w, 1);
    /* Explicit: bypass the autosave backoff and re-arm it on success. */
    w->ckpt_autosave_off = 0;
    int64_t now = runtime_now();
    if (w->saved_at > now) w->saved_at = now;
    runtime_checkpoint_now(w, out);
    return RUNTIME_OK;
}
