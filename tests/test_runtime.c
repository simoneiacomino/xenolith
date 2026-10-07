#define _GNU_SOURCE
#define _POSIX_C_SOURCE 200809L

#include "../runtime.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static int failures;

#define CHECK(condition) do { \
    if (!(condition)) { \
        failures++; \
        fprintf(stderr, "FAIL %s:%d: %s\n", __FILE__, __LINE__, \
                #condition); \
    } \
} while (0)

/* 3.7 finding 4: runtime error texts carry no digits and no client strings;
 * quantities travel as structured fields. */
static int text_is_digit_free(const char *text) {
    for (; *text; text++) if (*text >= '0' && *text <= '9') return 0;
    return 1;
}
#define CHECK_FAIL(expr, expected) do { \
    CHECK((expr) == (expected)); \
    CHECK(text_is_digit_free(runtime_error_text(w))); \
} while (0)

int main(int argc, char **argv) {
    if (argc < 2 || !argv[1][0]) {
        fprintf(stderr, "usage: %s <model.gguf>\n", argv[0]);
        return 2;
    }
    const char *model = argv[1];
    char state_dir[] = "/tmp/xenolith-test-runtime-state-XXXXXX";
    char cache_dir[] = "/tmp/xenolith-test-runtime-cache-XXXXXX";
    CHECK(mkdtemp(state_dir) != NULL);
    CHECK(mkdtemp(cache_dir) != NULL);

    xe_engine *e = xe_engine_open_vocab(model);
    runtime *w = NULL;
    CHECK(runtime_open(&w, e, state_dir, cache_dir) == RUNTIME_OK);
    if (!w) return 1;

    runtime_info info;
    CHECK(runtime_describe(w, &info) == RUNTIME_OK);
    CHECK(info.context_window == xe_context_size(e));
    CHECK(strcmp(info.model, "gemma-4-26B-A4B-it-qat") == 0);
    CHECK(info.kvstore == 1);
    CHECK(info.reasoning == 1);
    CHECK(runtime_kvstore_open_status(w) == KVSTORE_OK);

    profile_tool tool = {
        "read",
        "Read a file.",
        "{\"type\":\"object\",\"properties\":{\"path\":{\"type\":"
        "\"string\",\"description\":\"Path\"}},\"required\":[\"path\"]}"
    };
    conversation_id first_id;
    runtime_marker marker = RUNTIME_MARKER_NONE;
    CHECK(runtime_session_create(w, "You are a coding agent.", &tool, 1,
                              &first_id, &marker) == RUNTIME_OK);
    CHECK(marker == 0);
    CHECK(runtime_history_count(w) == 1);

    runtime_message user;
    memset(&user, 0, sizeof user);
    user.kind = RUNTIME_MESSAGE_USER;
    user.text = "fix the makefile";
    CHECK(runtime_append(w, &user, &marker) == RUNTIME_OK);
    CHECK(marker == 1);
    CHECK(runtime_history_count(w) == 2);

    runtime_history_entry entry;
    CHECK(runtime_history_at(w, 0, &entry) == RUNTIME_OK);
    CHECK(entry.kind == RUNTIME_MESSAGE_SYSTEM);
    CHECK(entry.text_length == strlen("You are a coding agent.") &&
          memcmp(entry.text, "You are a coding agent.",
                 entry.text_length) == 0);
    CHECK(entry.extra_json != NULL &&
          memmem(entry.extra_json, entry.extra_length, "\"read\"", 6)
          != NULL);
    CHECK(runtime_history_at(w, 1, &entry) == RUNTIME_OK);
    CHECK(entry.kind == RUNTIME_MESSAGE_USER && entry.marker == 1);

    runtime_message bad_result;
    memset(&bad_result, 0, sizeof bad_result);
    bad_result.kind = RUNTIME_MESSAGE_TOOL_RESULT;
    bad_result.call_id = 7;
    bad_result.text = "output";
    CHECK_FAIL(runtime_append(w, &bad_result, NULL), RUNTIME_INVALID_ARGUMENT);

    CHECK(runtime_rewind(w, 1) == RUNTIME_OK);
    CHECK(runtime_history_count(w) == 2);
    CHECK(runtime_rewind(w, 0) == RUNTIME_OK);
    CHECK(runtime_history_count(w) == 1);
    CHECK_FAIL(runtime_rewind(w, 1), RUNTIME_MARKER_UNAVAILABLE);
    uint64_t cost = 0;
    CHECK(runtime_rewind_cost(w, 0, &cost) == RUNTIME_OK);
    CHECK(cost > 0 && cost < 10000);

    user.text = "second attempt";
    CHECK(runtime_append(w, &user, &marker) == RUNTIME_OK);
    CHECK(marker == 3);
    CHECK(runtime_history_count(w) == 2);
    CHECK(runtime_history_at(w, 1, &entry) == RUNTIME_OK);
    CHECK(entry.marker == 3);
    CHECK(entry.text_length == strlen("second attempt"));

    profile_call call = { "read", "{\"path\":\"Makefile\"}" };
    runtime_message messages[4];
    memset(messages, 0, sizeof messages);
    messages[0].kind = RUNTIME_MESSAGE_USER;
    messages[0].text = "summary of history";
    messages[1].kind = RUNTIME_MESSAGE_ASSISTANT;
    messages[1].text = "Let me look.";
    messages[1].reasoning = "I should inspect it.";
    messages[1].calls = &call;
    messages[1].call_count = 1;
    messages[2].kind = RUNTIME_MESSAGE_TOOL_RESULT;
    messages[2].tool_name = "read";
    messages[2].text = "CC=gcc";
    messages[3].kind = RUNTIME_MESSAGE_ASSISTANT;
    messages[3].text = "It uses gcc.";
    messages[3].reasoning = "The answer is clear.";
    CHECK(runtime_rebuild(w, "You are a coding agent.", &tool, 1, messages, 4,
                       &marker) == RUNTIME_OK);
    CHECK(runtime_history_count(w) == 5);
    CHECK_FAIL(runtime_rewind(w, 3), RUNTIME_MARKER_UNAVAILABLE);
    CHECK(runtime_pending_calls(w, NULL, 0) == 0);
    CHECK(runtime_history_at(w, 2, &entry) == RUNTIME_OK);
    CHECK(entry.kind == RUNTIME_MESSAGE_ASSISTANT);
    CHECK(entry.reasoning_length == strlen("I should inspect it.") &&
          memcmp(entry.reasoning, "I should inspect it.",
                 entry.reasoning_length) == 0);
    CHECK(entry.extra_json != NULL);
    CHECK(runtime_history_at(w, 3, &entry) == RUNTIME_OK);
    CHECK(entry.kind == RUNTIME_MESSAGE_TOOL_RESULT);
    CHECK(entry.tool_name && strcmp(entry.tool_name, "read") == 0);
    CHECK(entry.tool_status == CONVERSATION_TOOL_OK);

    CHECK(runtime_rebuild(w, "You are a coding agent.", &tool, 1, messages, 2,
                       &marker) == RUNTIME_OK);
    uint64_t pending[4];
    CHECK(runtime_pending_calls(w, pending, 4) == 1);

    runtime_message result;
    memset(&result, 0, sizeof result);
    result.kind = RUNTIME_MESSAGE_TOOL_RESULT;
    result.call_id = pending[0];
    result.text = "CC=gcc";
    CHECK(runtime_append(w, &result, &marker) == RUNTIME_OK);
    CHECK(runtime_pending_calls(w, NULL, 0) == 0);
    uint64_t history_count = runtime_history_count(w);
    CHECK(runtime_history_at(w, history_count - 1, &entry) == RUNTIME_OK);
    CHECK(entry.kind == RUNTIME_MESSAGE_TOOL_RESULT &&
          entry.call_id == pending[0]);

    conversation_id second_id;
    CHECK(runtime_session_create(w, NULL, NULL, 0, &second_id, &marker)
          == RUNTIME_OK);
    CHECK(runtime_history_count(w) == 1);

    conversation_summary *summaries = NULL;
    size_t summary_count = 0;
    CHECK(runtime_session_list(w, &summaries, &summary_count) == RUNTIME_OK);
    CHECK(summary_count == 2);
    free(summaries);

    conversation_summary stat;
    CHECK(runtime_session_stat(w, &first_id, &stat) == RUNTIME_OK);
    CHECK(stat.token_count > 0);
    conversation_id missing;
    memset(&missing, 0x5a, sizeof missing);
    CHECK_FAIL(runtime_session_stat(w, &missing, &stat), RUNTIME_SESSION_NOT_FOUND);

    runtime_open_report report;
    CHECK(runtime_session_open(w, &first_id, &report) == RUNTIME_OK);
    CHECK(report.token_count == stat.token_count);
    CHECK(report.turn_open == 1);
    CHECK(report.zero_prefill == 0);
    CHECK(report.pending_calls == 0);

    CHECK(runtime_session_delete(w, &second_id) == RUNTIME_OK);
    CHECK(runtime_session_open(w, &second_id, &report)
          == RUNTIME_SESSION_NOT_FOUND);

    uint64_t before_close = runtime_history_count(w);
    CHECK(runtime_session_open(w, &first_id, &report) == RUNTIME_OK);
    runtime_close(w);
    w = NULL;
    CHECK(runtime_open(&w, e, state_dir, cache_dir) == RUNTIME_OK);
    CHECK(runtime_session_open(w, &first_id, &report) == RUNTIME_OK);
    CHECK(runtime_history_count(w) == before_close);
    CHECK(report.turn_open == 1);
    CHECK_FAIL(runtime_rewind(w, 3), RUNTIME_MARKER_UNAVAILABLE);
    CHECK(runtime_history_at(w, 1, &entry) == RUNTIME_OK);
    CHECK(entry.kind == RUNTIME_MESSAGE_USER &&
          entry.text_length == strlen("summary of history"));

    runtime_gen_params invalid_params;
    memset(&invalid_params, 0, sizeof invalid_params);
    invalid_params.reasoning_set = 1;
    invalid_params.reasoning_effort = CONVERSATION_REASONING_MAX + 1;
    CHECK_FAIL(runtime_generate(w, &invalid_params), RUNTIME_INVALID_ARGUMENT);

    runtime_message oversized;
    memset(&oversized, 0, sizeof oversized);
    oversized.kind = RUNTIME_MESSAGE_USER;
    size_t big = 2400000;
    char *big_text = malloc(big + 1);
    CHECK(big_text != NULL);
    if (big_text) {
        for (size_t i = 0; i < big; i++)
            big_text[i] = "qwe rty uio zxc vbn "[i % 20];
        big_text[big] = '\0';
        oversized.text = big_text;
        CHECK_FAIL(runtime_append(w, &oversized, NULL),
                   RUNTIME_CONTEXT_LENGTH_EXCEEDED);
        CHECK(strcmp(runtime_error_text(w),
                     "prompt does not fit the context window") == 0);
        uint64_t over_tokens = 0, over_context = 0;
        CHECK(runtime_error_detail(w, &over_tokens, &over_context) == 1);
        CHECK(over_context == (uint64_t)xe_context_size(e));
        CHECK(over_tokens + 1 >= over_context);
        /* the detail belongs to that failure only */
        CHECK_FAIL(runtime_rewind(w, 999), RUNTIME_MARKER_UNAVAILABLE);
        CHECK(runtime_error_detail(w, NULL, NULL) == 0);

        runtime_open_report before_rebuild;
        CHECK(runtime_session_open(w, &first_id, &before_rebuild) == RUNTIME_OK);
        uint64_t history_before_rebuild = runtime_history_count(w);
        size_t pending_before_rebuild = runtime_pending_calls(w, NULL, 0);
        CHECK(runtime_history_at(w, history_before_rebuild - 1, &entry) ==
              RUNTIME_OK);
        runtime_marker last_marker = entry.marker;
        uint32_t last_kind = entry.kind;
        runtime_message failed_rebuild[2];
        memset(failed_rebuild, 0, sizeof failed_rebuild);
        failed_rebuild[0].kind = RUNTIME_MESSAGE_USER;
        failed_rebuild[0].text = "replacement prefix";
        failed_rebuild[1] = oversized;
        CHECK_FAIL(runtime_rebuild(w, "A different system prompt.", &tool, 1,
                                failed_rebuild, 2, NULL),
                   RUNTIME_CONTEXT_LENGTH_EXCEEDED);
        CHECK(runtime_error_detail(w, &over_tokens, &over_context) == 1);
        CHECK(over_context == (uint64_t)xe_context_size(e));
        CHECK(over_tokens + 1 >= over_context);
        runtime_open_report after_rebuild;
        CHECK(runtime_session_open(w, &first_id, &after_rebuild) == RUNTIME_OK);
        CHECK(after_rebuild.token_count == before_rebuild.token_count);
        CHECK(after_rebuild.marker == before_rebuild.marker);
        CHECK(after_rebuild.turn_open == before_rebuild.turn_open);
        CHECK(runtime_history_count(w) == history_before_rebuild);
        CHECK(runtime_pending_calls(w, NULL, 0) == pending_before_rebuild);
        CHECK(runtime_history_at(w, history_before_rebuild - 1, &entry) ==
              RUNTIME_OK);
        CHECK(entry.marker == last_marker && entry.kind == last_kind);
        free(big_text);
    }

    runtime_close(w);

    /* 3.7 finding 2: a store that cannot open is reported, not hidden. */
    {
        char blocked[4096];
        snprintf(blocked, sizeof blocked, "%s/file/kv", cache_dir);
        char file_path[4096];
        snprintf(file_path, sizeof file_path, "%s/file", cache_dir);
        FILE *f = fopen(file_path, "w");
        CHECK(f != NULL);
        if (f) fclose(f);
        runtime *nokv = NULL;
        CHECK(runtime_open(&nokv, e, state_dir, blocked) == RUNTIME_OK);
        if (nokv) {
            runtime_info info2;
            CHECK(runtime_describe(nokv, &info2) == RUNTIME_OK);
            CHECK(info2.kvstore == 0);
            CHECK(runtime_kvstore_open_status(nokv) != KVSTORE_OK);
            conversation_id cid;
            runtime_marker m;
            CHECK(runtime_session_create(nokv, "S", NULL, 0, &cid, &m) == RUNTIME_OK);
            runtime_checkpoint_report r;
            CHECK(runtime_checkpoint(nokv, &r) == RUNTIME_OK);
            CHECK(r.saved == 0 && r.reason == RUNTIME_CKPT_NO_KVSTORE);
            CHECK(strcmp(runtime_ckpt_reason_name(r.reason), "no_kvstore") == 0);
            runtime_close(nokv);
        }
    }
    xe_engine_close(e);

    if (failures) {
        fprintf(stderr, "test_runtime: %d failures\n", failures);
        return 1;
    }
    printf("test_runtime: all checks passed\n");
    return 0;
}
