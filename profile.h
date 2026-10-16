#ifndef PROFILE_H
#define PROFILE_H

#include "xenolith.h"

#include <stddef.h>
#include <stdint.h>

typedef struct profile profile;

typedef enum {
    PROFILE_OK,
    PROFILE_TEMPLATE_MISMATCH,
    PROFILE_INVALID_ARGUMENT,
    PROFILE_LIMIT,
    PROFILE_NOMEM
} profile_status;

typedef enum {
    PROFILE_TURN_PADDED,
    PROFILE_TURN_BARE,
    PROFILE_TURN_OPEN
} profile_turn;

typedef enum {
    PROFILE_REASONING_OFF,
    PROFILE_REASONING_LOW,
    PROFILE_REASONING_MEDIUM,
    PROFILE_REASONING_HIGH,
    PROFILE_REASONING_MAX
} profile_reasoning_effort;

typedef struct {
    int32_t hard_tokens;
    int32_t soft_tokens;
    int32_t delimiter_rank;
    float delimiter_margin;
} profile_reasoning_policy;

typedef struct {
    const char *name;
    const char *description;
    const char *parameters_json;
} profile_tool;

typedef struct {
    const char *name;
    const char *arguments_json;
} profile_call;

typedef struct {
    const uint8_t *render;
    uint64_t render_length;
    const int32_t *tokens;
    uint32_t token_count;
} profile_render;

profile_status profile_open(profile **out, const xe_engine *engine);
void profile_close(profile *p);
const char *profile_model(const profile *p);

profile_status profile_render_system(profile *p, const char *text,
                                     const profile_tool *tools,
                                     size_t tool_count, int thinking,
                                     profile_render *out);
profile_status profile_render_user(profile *p, const char *text,
                                   uint32_t turn, profile_render *out);
profile_status profile_render_user_image(profile *p, const char *text,
                                         uint32_t image_rows, uint32_t turn,
                                         profile_render *out, uint32_t *image_offset);
profile_status profile_render_tool_image(profile *p, const char *name, const char *text,
                                          uint32_t rows, profile_render *out,
                                          uint32_t *image_offset);
profile_status profile_render_tool_result(profile *p, const char *name,
                                          const char *value,
                                          profile_render *out);
profile_status profile_render_assistant(profile *p, const char *reasoning,
                                        int reasoning_complete,
                                        const char *text,
                                        const profile_call *calls,
                                        size_t call_count, uint32_t turn,
                                        int turn_complete,
                                        profile_render *out);
profile_status profile_render_reply_open(profile *p, uint32_t turn,
                                         int thinking, int after_tool,
                                         profile_render *out);

typedef enum {
    PROFILE_PARSE_NONE,
    PROFILE_PARSE_REASONING,
    PROFILE_PARSE_TEXT,
    PROFILE_PARSE_CALL_START,
    PROFILE_PARSE_CALL_END,
    PROFILE_PARSE_STOP
} profile_parse_kind;

typedef enum {
    PROFILE_STOP_EOT,
    PROFILE_STOP_EOS,
    PROFILE_STOP_TOOL_CALLS
} profile_stop_reason;

typedef struct {
    uint32_t kind;
    const uint8_t *text;
    size_t text_length;
    const char *call_name;
    const char *arguments_json;
    uint32_t stop_reason;
    int include_token;
} profile_parse_event;

void profile_parser_reset(profile *p, int reasoning_open);
profile_status profile_parser_feed(profile *p, int32_t token,
                                   profile_parse_event *out);
size_t profile_parser_calls(const profile *p);
int profile_parser_reasoning(const profile *p);
int32_t profile_reasoning_end_token(const profile *p);
profile_status profile_get_reasoning_policy(
    const profile *p, uint32_t effort, int32_t budget_override,
    int32_t max_tokens, int32_t context_remaining,
    profile_reasoning_policy *out);

#endif
