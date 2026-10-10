#ifndef CONVERSATION_H
#define CONVERSATION_H

#include "xenolith.h"
#include "kvstore.h"

#include <stdint.h>

#define CONVERSATION_SAMPLER_ABI 1
#define CONVERSATION_TITLE_MAX 4096
#define CONVERSATION_WORKSPACE_MAX 4096
#define CONVERSATION_FRAME_MAX 67108864

typedef struct conversation conversation;
typedef struct conversation_store conversation_store;

typedef struct {
    uint8_t bytes[16];
} conversation_id;

typedef enum {
    CONVERSATION_OK,
    CONVERSATION_MISS,
    CONVERSATION_INVALID_ARGUMENT,
    CONVERSATION_IO,
    CONVERSATION_FORMAT,
    CONVERSATION_VERSION,
    CONVERSATION_DAMAGED,
    CONVERSATION_LOCKED,
    CONVERSATION_LIMIT,
    CONVERSATION_NOMEM
} conversation_status;

typedef enum {
    CONVERSATION_EVENT_TITLE = 1,
    CONVERSATION_EVENT_WORKSPACE = 2,
    CONVERSATION_EVENT_SETTINGS = 3,
    CONVERSATION_EVENT_MESSAGE = 4,
    CONVERSATION_EVENT_GENERATION_STARTED = 5,
    CONVERSATION_EVENT_GENERATION_RESULT = 6,
    CONVERSATION_EVENT_TOOL_STARTED = 7,
    CONVERSATION_EVENT_TOOL_RESULT = 8,
    CONVERSATION_EVENT_SNAPSHOT_REF = 9,
    CONVERSATION_EVENT_CACHE_EPOCH = 10,
    CONVERSATION_EVENT_COMMIT = 11,
    CONVERSATION_EVENT_REWIND = 12
} conversation_event_type;

#define CONVERSATION_REWIND_ALL UINT64_MAX

typedef enum {
    CONVERSATION_ROLE_SYSTEM = 1,
    CONVERSATION_ROLE_USER = 2,
    CONVERSATION_ROLE_ASSISTANT = 3,
    CONVERSATION_ROLE_TOOL = 4
} conversation_role;

typedef enum {
    CONVERSATION_BLOCK_TEXT = 1,
    CONVERSATION_BLOCK_JSON = 2,
    CONVERSATION_BLOCK_BINARY = 3
} conversation_block_format;

typedef enum {
    CONVERSATION_STOP_EOT_SAMPLED = 1,
    CONVERSATION_STOP_EOT_SYNTHETIC = 2,
    CONVERSATION_STOP_EOS = 3,
    CONVERSATION_STOP_LIMIT = 4,
    CONVERSATION_STOP_CANCELLED = 5,
    CONVERSATION_STOP_TOOL_CALLS = 6
} conversation_stop_reason;

typedef enum {
    CONVERSATION_TOOL_OK = 1,
    CONVERSATION_TOOL_ERROR = 2
} conversation_tool_status;

typedef enum {
    CONVERSATION_REASONING_OFF = 0,
    CONVERSATION_REASONING_LOW = 1,
    CONVERSATION_REASONING_MEDIUM = 2,
    CONVERSATION_REASONING_HIGH = 3,
    CONVERSATION_REASONING_MAX = 4
} conversation_reasoning_effort;

typedef enum {
    CONVERSATION_REASONING_DISCARD = 0,
    CONVERSATION_REASONING_PRESERVE_TOOL_CALLS = 1
} conversation_reasoning_history;

typedef enum {
    CONVERSATION_REASONING_NONE = 0,
    CONVERSATION_REASONING_NATURAL = 1,
    CONVERSATION_REASONING_SOFT = 2,
    CONVERSATION_REASONING_HARD = 3,
    CONVERSATION_REASONING_LENGTH = 4,
    CONVERSATION_REASONING_ABORTED = 5,
    CONVERSATION_REASONING_EOS = 6
} conversation_reasoning_close;

typedef struct {
    uint32_t format;
    const void *data;
    uint64_t length;
} conversation_block;

typedef struct {
    float temperature;
    int32_t top_k;
    float top_p;
    int32_t max_tokens;
    uint32_t sampler_abi;
    uint64_t rng_seed;
    uint64_t rng_state;
    uint32_t reasoning_effort;
    uint32_t reasoning_history;
    int32_t reasoning_budget;
} conversation_settings;

typedef struct {
    uint64_t generation_id;
    conversation_settings settings;
} conversation_generation;

typedef struct {
    uint64_t call_id;
    const char *server;
    const char *tool;
    const uint8_t *arguments;
    uint64_t arguments_length;
    uint8_t fingerprint[32];
} conversation_tool_call;

typedef struct {
    uint32_t type;
    int64_t timestamp;
    uint32_t role;
    const conversation_block *blocks;
    uint32_t block_count;
    const uint8_t *render;
    uint64_t render_length;
    const int32_t *tokens;
    uint32_t token_count;
    const uint8_t *alternate_render;
    uint64_t alternate_render_length;
    const int32_t *alternate_tokens;
    uint32_t alternate_token_count;
    const uint8_t *raw_render;
    uint64_t raw_render_length;
    const int32_t *raw_tokens;
    uint32_t raw_token_count;
    const uint8_t *reasoning;
    uint64_t reasoning_length;
    uint32_t reasoning_close;
    conversation_settings settings;
    uint64_t generation_id;
    uint32_t stop_reason;
    uint64_t rng_after;
    uint64_t call_id;
    const char *server;
    const char *tool;
    const uint8_t *arguments;
    uint64_t arguments_length;
    const uint8_t *fingerprint;
    uint32_t tool_status;
    kvstore_id snapshot;
    uint64_t snapshot_boundary;
    uint64_t epoch;
    uint64_t rewind_target;
} conversation_event;

typedef struct {
    conversation_id id;
    int64_t created;
    int64_t updated;
    char title[128];
    uint64_t token_count;
    int resumable;
} conversation_summary;

int conversation_default_state_dir(char *out, size_t cap);

conversation_status conversation_store_open(conversation_store **out,
                                            const char *state_dir);
void conversation_store_close(conversation_store *store);

conversation_status conversation_create(conversation_store *store,
                                        conversation **out,
                                        conversation_id *id);
conversation_status conversation_open(conversation_store *store,
                                      const conversation_id *id,
                                      conversation **out);
void conversation_close(conversation *c);
conversation_status conversation_delete(conversation_store *store,
                                        const conversation_id *id);
conversation_status conversation_list(conversation_store *store,
                                      conversation_summary **out,
                                      size_t *count);

conversation_status conversation_append_title(conversation *c,
                                              const char *title);
conversation_status conversation_append_workspace(conversation *c,
                                                  const char *path);
conversation_status conversation_append_settings(
    conversation *c, const conversation_settings *settings);
conversation_status conversation_append_message(
    conversation *c, uint32_t role,
    const conversation_block *blocks, uint32_t block_count,
    const void *render, uint64_t render_length,
    const int32_t *tokens, uint32_t token_count);
conversation_status conversation_append_message_variants(
    conversation *c, uint32_t role,
    const conversation_block *blocks, uint32_t block_count,
    const void *render, uint64_t render_length,
    const int32_t *tokens, uint32_t token_count,
    const void *alternate_render, uint64_t alternate_render_length,
    const int32_t *alternate_tokens, uint32_t alternate_token_count,
    const void *reasoning, uint64_t reasoning_length,
    uint32_t reasoning_close);
conversation_status conversation_append_generation_started(
    conversation *c, const conversation_generation *generation);
conversation_status conversation_append_generation_result(
    conversation *c, uint64_t generation_id, uint32_t stop_reason,
    uint64_t rng_after,
    const conversation_block *blocks, uint32_t block_count,
    const void *render, uint64_t render_length,
    const int32_t *tokens, uint32_t token_count);
conversation_status conversation_append_generation_result_variants(
    conversation *c, uint64_t generation_id, uint32_t stop_reason,
    uint64_t rng_after,
    const conversation_block *blocks, uint32_t block_count,
    const void *render, uint64_t render_length,
    const int32_t *tokens, uint32_t token_count,
    const void *alternate_render, uint64_t alternate_render_length,
    const int32_t *alternate_tokens, uint32_t alternate_token_count,
    const void *reasoning, uint64_t reasoning_length,
    uint32_t reasoning_close,
    const void *raw_render, uint64_t raw_render_length,
    const int32_t *raw_tokens, uint32_t raw_token_count);
conversation_status conversation_append_tool_started(
    conversation *c, const conversation_tool_call *call);
conversation_status conversation_append_tool_result(
    conversation *c, uint64_t call_id, uint32_t tool_status,
    const conversation_block *blocks, uint32_t block_count,
    const void *render, uint64_t render_length,
    const int32_t *tokens, uint32_t token_count);
conversation_status conversation_append_snapshot_ref(
    conversation *c, const kvstore_id *snapshot, uint64_t token_boundary);
conversation_status conversation_append_cache_epoch(conversation *c);
conversation_status conversation_append_rewind(conversation *c,
                                               uint64_t target_event);
conversation_status conversation_commit(conversation *c);
conversation_status conversation_rollback(conversation *c);

const conversation_id *conversation_get_id(const conversation *c);
int64_t conversation_created(const conversation *c);
int64_t conversation_updated(const conversation *c);
const char *conversation_get_title(const conversation *c);
const char *conversation_get_workspace(const conversation *c);
int conversation_get_settings(const conversation *c,
                              conversation_settings *out);
uint64_t conversation_epoch_current(const conversation *c);
const int32_t *conversation_tokens(const conversation *c, uint64_t *count);
conversation_status conversation_project(conversation *c, int thinking,
                                         uint32_t reasoning_history);
conversation_status conversation_project_copy(const conversation *c,
                                               int thinking,
                                               uint32_t reasoning_history,
                                               int close_current,
                                               int32_t **tokens,
                                               uint64_t *count);
int conversation_snapshot_current(const conversation *c, kvstore_id *id,
                                  uint64_t *boundary);
uint64_t conversation_event_count(const conversation *c);
const conversation_event *conversation_event_at(const conversation *c,
                                                uint64_t index);
uint64_t conversation_visible_count(const conversation *c);
uint64_t conversation_visible_index(const conversation *c,
                                    uint64_t position);
uint64_t conversation_visible_boundary(const conversation *c,
                                       uint64_t position);
int conversation_visible_position(const conversation *c,
                                  uint64_t event_index, uint64_t *position);
int conversation_generation_interrupted(const conversation *c);
size_t conversation_unknown_tool_calls(const conversation *c,
                                       uint64_t *call_ids, size_t cap);
uint64_t conversation_uncommitted(const conversation *c);

typedef struct {
    int used_snapshot;
    int zero_prefill;
    uint64_t boundary;
} conversation_resume_report;

/* Returns CONVERSATION_LIMIT without changing session state if the record
 * exceeds the session's engine capacity. */
conversation_status conversation_resume(conversation *c, kvstore *store,
                                        xe_session *session,
                                        conversation_resume_report *report);

int conversation_autosave_due(uint64_t tokens, uint64_t saved_tokens,
                              int64_t saved_at, int64_t now);

const char *conversation_status_name(conversation_status status);

#endif
