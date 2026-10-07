#define _GNU_SOURCE
#define _POSIX_C_SOURCE 200809L

#include "../runtime.h"
#include "../json.h"

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

static const char *store_fault_point;
int kvstore_fault(const char *point) {
    return store_fault_point && strcmp(store_fault_point, point) == 0;
}

typedef struct {
    char text[16384];
    size_t text_length;
    char reasoning[16384];
    size_t reasoning_length;
    uint64_t call_ids[8];
    char call_names[8][64];
    char call_args[8][4096];
    size_t calls;
    uint32_t stop;
    uint32_t reasoning_close;
    runtime_usage usage;
    runtime_marker marker;
    int error;
    int progress_events;
    runtime_inference inference;
    int checkpoint_attempted;
    runtime_checkpoint_report checkpoint;
    int resume_attempted;
    runtime_resume_report resume;
} run_result;

static void run_generation(runtime *w, run_result *out, int cancel_after_text) {
    memset(out, 0, sizeof *out);
    out->marker = RUNTIME_MARKER_NONE;
    runtime_inference_measure phases[2] = {{0}, {0}};
    int phase_seen[2] = {0};
    int phase_finished[2] = {0};
    for (;;) {
        runtime_event event;
        runtime_status status = runtime_next_event(w, &event);
        if (status != RUNTIME_OK) {
            out->error = 1;
            fprintf(stderr, "stream failed: %s\n", runtime_error_text(w));
            return;
        }
        switch (event.kind) {
        case RUNTIME_EVENT_INFERENCE_PROGRESS: {
            CHECK(event.phase <= RUNTIME_INFERENCE_DECODE);
            if (event.phase > RUNTIME_INFERENCE_DECODE) return;
            CHECK(!phase_finished[event.phase]);
            CHECK(event.state == RUNTIME_INFERENCE_RUNNING ||
                  event.state == RUNTIME_INFERENCE_FINISHED);
            if (!phase_seen[event.phase]) {
                CHECK(event.state == RUNTIME_INFERENCE_RUNNING);
                CHECK(event.progress.tokens == 0 && event.progress.elapsed_ms == 0);
                if (event.phase == RUNTIME_INFERENCE_DECODE)
                    CHECK(phase_finished[RUNTIME_INFERENCE_PREFILL]);
            }
            phase_finished[event.phase] = event.state == RUNTIME_INFERENCE_FINISHED;
            runtime_inference_measure *previous = &phases[event.phase];
            CHECK(event.progress.tokens >= previous->tokens);
            CHECK(event.progress.elapsed_ms >= previous->elapsed_ms);
            if (event.phase == RUNTIME_INFERENCE_PREFILL)
                CHECK(event.progress.tokens <= event.progress.total);
            *previous = event.progress;
            phase_seen[event.phase] = 1;
            out->progress_events++;
            break;
        }
        case RUNTIME_EVENT_TEXT_DELTA:
            if (out->text_length + event.text_length <
                sizeof out->text - 1) {
                memcpy(out->text + out->text_length, event.text,
                       event.text_length);
                out->text_length += event.text_length;
                out->text[out->text_length] = '\0';
            }
            if (cancel_after_text) {
                runtime_cancel(w);
                cancel_after_text = 0;
            }
            break;
        case RUNTIME_EVENT_REASONING_DELTA:
            if (out->reasoning_length + event.text_length <
                sizeof out->reasoning - 1) {
                memcpy(out->reasoning + out->reasoning_length, event.text,
                       event.text_length);
                out->reasoning_length += event.text_length;
                out->reasoning[out->reasoning_length] = '\0';
            }
            break;
        case RUNTIME_EVENT_TOOLCALL_END:
            if (out->calls < 8) {
                out->call_ids[out->calls] = event.call_id;
                snprintf(out->call_names[out->calls], 64, "%s",
                         event.call_name);
                snprintf(out->call_args[out->calls], 4096, "%s",
                         event.arguments_json);
            }
            out->calls++;
            break;
        case RUNTIME_EVENT_DONE:
            out->inference = event.inference;
            CHECK(!phase_seen[0] || phase_finished[0]);
            CHECK(!phase_seen[1] || phase_finished[1]);
            if (event.stop != RUNTIME_STOP_ABORTED)
                CHECK(phase_seen[0] && phase_seen[1]);
            CHECK(event.inference.prefill.tokens == phases[0].tokens);
            CHECK(event.inference.prefill.elapsed_ms == phases[0].elapsed_ms);
            CHECK(event.inference.decode.tokens == phases[1].tokens);
            CHECK(event.inference.decode.elapsed_ms == phases[1].elapsed_ms);
            CHECK(event.inference.decode.tokens == event.usage.output);
            CHECK(event.inference.prefill.tokens <= event.usage.input);
            out->stop = event.stop;
            out->reasoning_close = event.reasoning_close;
            out->usage = event.usage;
            out->marker = event.marker;
            out->checkpoint_attempted = event.checkpoint_attempted;
            out->checkpoint = event.checkpoint;
            out->resume_attempted = event.resume_attempted;
            out->resume = event.resume;
            return;
        case RUNTIME_EVENT_ERROR:
            out->error = 1;
            fprintf(stderr, "stream error: %s\n",
                    event.error_text ? event.error_text : "");
            return;
        default:
            break;
        }
    }
}

static int file_contains(const char *path, const char *needle) {
    FILE *f = fopen(path, "rb");
    if (!f) return 0;
    fseek(f, 0, SEEK_END);
    long size = ftell(f);
    fseek(f, 0, SEEK_SET);
    if (size <= 0) {
        fclose(f);
        return 0;
    }
    char *data = malloc((size_t)size + 1);
    if (!data || fread(data, 1, (size_t)size, f) != (size_t)size) {
        free(data);
        fclose(f);
        return 0;
    }
    data[size] = '\0';
    fclose(f);
    int found = strstr(data, needle) != NULL;
    free(data);
    return found;
}

int main(int argc, char **argv) {
    if (argc < 2 || !argv[1][0]) {
        fprintf(stderr, "usage: %s <model.gguf>\n", argv[0]);
        return 2;
    }
    const char *model = argv[1];
    char state_dir[] = "/tmp/xenolith-test-wm-state-XXXXXX";
    char cache_dir[] = "/tmp/xenolith-test-wm-cache-XXXXXX";
    char ndjson_dir[] = "/tmp/xenolith-test-wm-ndjson-XXXXXX";
    CHECK(mkdtemp(state_dir) != NULL);
    CHECK(mkdtemp(cache_dir) != NULL);
    CHECK(mkdtemp(ndjson_dir) != NULL);

    xe_engine *e = xe_engine_open(model);
    runtime *w = NULL;
    CHECK(runtime_open(&w, e, state_dir, cache_dir) == RUNTIME_OK);
    if (!w) return 1;

    profile_tool weather = {
        "get_weather",
        "Get the current weather for a city.",
        "{\"type\":\"object\",\"properties\":{\"city\":{\"type\":"
        "\"string\",\"description\":\"The city name, e.g. Kyoto\"}},"
        "\"required\":[\"city\"]}"
    };
    conversation_id id;
    runtime_marker marker;
    CHECK(runtime_session_create(w, "You are a helpful assistant with tools.",
                              &weather, 1, &id, &marker) == RUNTIME_OK);

    runtime_gen_params params = {
        .temperature = 1.0f,
        .top_k = 1,
        .top_p = 1.0f,
        .max_tokens = 300,
        .rng_seed = 42
    };
    runtime_message user;
    memset(&user, 0, sizeof user);
    user.kind = RUNTIME_MESSAGE_USER;
    user.text = "What's the weather in Kyoto right now?";
    CHECK(runtime_append(w, &user, &marker) == RUNTIME_OK);
    CHECK(runtime_generate(w, &params) == RUNTIME_OK);

    run_result turn1;
    run_generation(w, &turn1, 0);
    CHECK(!turn1.error);
    CHECK(turn1.stop == RUNTIME_STOP_TOOL_USE);
    CHECK(turn1.calls == 1);
    CHECK(strcmp(turn1.call_names[0], "get_weather") == 0);
    CHECK(strstr(turn1.call_args[0], "Kyoto") != NULL);
    CHECK(turn1.usage.output > 0);
    CHECK(turn1.usage.input > 0);
    CHECK(turn1.usage.cache_read == 0);
    CHECK(runtime_pending_calls(w, NULL, 0) == 1);

    runtime_message result;
    memset(&result, 0, sizeof result);
    result.kind = RUNTIME_MESSAGE_TOOL_RESULT;
    result.call_id = turn1.call_ids[0];
    result.text = "21C, light rain, humidity 81%";
    runtime_marker result_marker;
    CHECK(runtime_append(w, &result, &result_marker) == RUNTIME_OK);
    CHECK(runtime_pending_calls(w, NULL, 0) == 0);

    runtime_gen_params greedy = {
        .temperature = 1.0f,
        .top_k = 1,
        .top_p = 1.0f,
        .max_tokens = 300
    };
    CHECK(runtime_generate(w, &greedy) == RUNTIME_OK);
    run_result turn2;
    run_generation(w, &turn2, 0);
    CHECK(!turn2.error);
    CHECK(turn2.stop == RUNTIME_STOP_STOP);
    CHECK(turn2.text_length > 0);
    CHECK(strstr(turn2.text, "21") != NULL);
    CHECK(turn2.usage.cache_read + 5 >= turn1.usage.total);
    CHECK(turn2.usage.input < 60);

    CHECK(runtime_rewind(w, result_marker) == RUNTIME_OK);
    uint64_t cost = 12345;
    CHECK(runtime_rewind_cost(w, result_marker, &cost) == RUNTIME_OK);
    CHECK(cost <= 1);
    CHECK(runtime_generate(w, &greedy) == RUNTIME_OK);
    run_result turn3;
    run_generation(w, &turn3, 0);
    CHECK(!turn3.error);
    CHECK(turn3.stop == RUNTIME_STOP_STOP);
    CHECK(turn3.text_length == turn2.text_length &&
          memcmp(turn3.text, turn2.text, turn2.text_length) == 0);
    CHECK(turn3.usage.input <= 2);
    CHECK(turn3.usage.cache_read > 0);

    size_t too_big_length = 2400000;
    char *too_big = malloc(too_big_length + 1);
    CHECK(too_big != NULL);
    if (too_big) {
        for (size_t i = 0; i < too_big_length; i++)
            too_big[i] = "qwe rty uio zxc vbn "[i % 20];
        too_big[too_big_length] = '\0';
        runtime_message failed_rebuild[2];
        memset(failed_rebuild, 0, sizeof failed_rebuild);
        failed_rebuild[0].kind = RUNTIME_MESSAGE_USER;
        failed_rebuild[0].text = "replacement prefix";
        failed_rebuild[1].kind = RUNTIME_MESSAGE_USER;
        failed_rebuild[1].text = too_big;
        uint64_t history_before_rebuild = runtime_history_count(w);
        runtime_open_report before_rebuild;
        CHECK(runtime_session_open(w, &id, &before_rebuild) == RUNTIME_OK);
        CHECK(runtime_rebuild(w, "A different system prompt.", &weather, 1,
                           failed_rebuild, 2, NULL) ==
              RUNTIME_CONTEXT_LENGTH_EXCEEDED);
        runtime_open_report after_rebuild;
        CHECK(runtime_session_open(w, &id, &after_rebuild) == RUNTIME_OK);
        CHECK(after_rebuild.token_count == before_rebuild.token_count);
        CHECK(after_rebuild.marker == before_rebuild.marker);
        CHECK(runtime_history_count(w) == history_before_rebuild);
        free(too_big);
    }

    /* 3.7 finding 2: a failed save is reported, never silent. */
    runtime_checkpoint_report ckpt;
    store_fault_point = "snapshot-synced";
    CHECK(runtime_checkpoint(w, &ckpt) == RUNTIME_OK);
    CHECK(ckpt.saved == 0 && ckpt.reason == RUNTIME_CKPT_IO);
    CHECK(ckpt.tokens == turn3.usage.total);
    runtime_open_report failed_report;
    CHECK(runtime_session_open(w, &id, &failed_report) == RUNTIME_OK);
    CHECK(failed_report.zero_prefill == 0);
    store_fault_point = NULL;
    CHECK(runtime_checkpoint(w, &ckpt) == RUNTIME_OK);
    CHECK(ckpt.saved == 1 && ckpt.reason == RUNTIME_CKPT_SAVED);
    CHECK(ckpt.tokens == turn3.usage.total);
    CHECK(runtime_checkpoint(w, &ckpt) == RUNTIME_OK);
    CHECK(ckpt.saved == 0 && ckpt.reason == RUNTIME_CKPT_NOTHING_NEW);
    uint64_t tokens_before;
    tokens_before = turn3.usage.total;
    runtime_close(w);
    w = NULL;
    CHECK(runtime_open(&w, e, state_dir, cache_dir) == RUNTIME_OK);
    runtime_open_report report;
    CHECK(runtime_session_open(w, &id, &report) == RUNTIME_OK);
    CHECK(report.token_count == tokens_before);
    CHECK(report.zero_prefill == 1);
    CHECK(report.resume_stale == 0);
    CHECK(report.turn_open == 0);

    user.text = "Thanks. Reply with one short sentence.";
    CHECK(runtime_append(w, &user, &marker) == RUNTIME_OK);
    runtime_open_report stale_report;
    CHECK(runtime_session_open(w, &id, &stale_report) == RUNTIME_OK);
    CHECK(stale_report.zero_prefill == 0 && stale_report.resume_stale == 1);
    CHECK(runtime_generate(w, &greedy) == RUNTIME_OK);
    run_result turn4;
    run_generation(w, &turn4, 0);
    CHECK(!turn4.error);
    CHECK(turn4.usage.cache_read >= tokens_before);
    CHECK(turn4.usage.input < 40);
    CHECK(turn4.resume_attempted == 1);
    CHECK(turn4.resume.loaded == 1);
    CHECK(turn4.resume.tokens == tokens_before);

    user.text = "Tell me a very long story about the sea.";
    CHECK(runtime_append(w, &user, &marker) == RUNTIME_OK);
    CHECK(runtime_generate(w, &greedy) == RUNTIME_OK);
    run_result cancelled;
    run_generation(w, &cancelled, 1);
    CHECK(!cancelled.error);
    CHECK(cancelled.stop == RUNTIME_STOP_ABORTED);
    CHECK(cancelled.usage.output >= 1);
    runtime_open_report after_cancel;
    CHECK(runtime_session_open(w, &id, &after_cancel) == RUNTIME_OK);
    CHECK(after_cancel.turn_open == 1);

    runtime_gen_params tiny = {
        .temperature = 1.0f,
        .top_k = 1,
        .top_p = 1.0f,
        .max_tokens = 30
    };
    user.text = "Never mind, just say bye.";
    CHECK(runtime_append(w, &user, &marker) == RUNTIME_OK);
    CHECK(runtime_generate(w, &tiny) == RUNTIME_OK);
    run_result resumed;
    run_generation(w, &resumed, 0);
    CHECK(!resumed.error);
    CHECK(resumed.stop == RUNTIME_STOP_STOP ||
          resumed.stop == RUNTIME_STOP_LENGTH);

    runtime_message hello;
    memset(&hello, 0, sizeof hello);
    hello.kind = RUNTIME_MESSAGE_USER;
    hello.text = "Say hello.";
    runtime_gen_params eph_params = {
        .temperature = 1.0f,
        .top_k = 1,
        .top_p = 1.0f,
        .max_tokens = 16
    };
    CHECK(runtime_ephemeral_generate(w, "You answer with one short word.",
                                  NULL, 0, &hello, 1, &eph_params)
          == RUNTIME_OK);
    run_result ephemeral;
    run_generation(w, &ephemeral, 0);
    CHECK(!ephemeral.error);
    CHECK(ephemeral.marker == RUNTIME_MARKER_NONE);
    CHECK(ephemeral.usage.total > 0);
    CHECK(runtime_session_open(w, &id, &report) == RUNTIME_OK);
    CHECK(report.token_count > tokens_before);

    conversation_id thinking_id;
    CHECK(runtime_session_create(w, "Solve carefully and answer clearly.",
                              NULL, 0, &thinking_id, &marker) == RUNTIME_OK);
    user.text = "Calculate 137 times 29. Think step by step, then give the "
                "number in the final answer.";
    CHECK(runtime_append(w, &user, &marker) == RUNTIME_OK);
    runtime_gen_params thinking = {
        .temperature = 1.0f,
        .top_k = 1,
        .top_p = 1.0f,
        .max_tokens = 300,
        .reasoning_set = 1,
        .reasoning_effort = CONVERSATION_REASONING_LOW,
        .reasoning_budget_set = 1,
        .reasoning_budget = 12
    };
    CHECK(runtime_generate(w, &thinking) == RUNTIME_OK);
    run_result thought;
    run_generation(w, &thought, 0);
    CHECK(!thought.error);
    CHECK(thought.stop == RUNTIME_STOP_STOP ||
          thought.stop == RUNTIME_STOP_LENGTH);
    CHECK(thought.reasoning_length > 0);
    CHECK(thought.text_length > 0);
    CHECK(thought.usage.reasoning > 0 && thought.usage.reasoning <= 12);
    CHECK(thought.usage.output > thought.usage.reasoning);
    CHECK(thought.reasoning_close == CONVERSATION_REASONING_NATURAL ||
          thought.reasoning_close == CONVERSATION_REASONING_SOFT ||
          thought.reasoning_close == CONVERSATION_REASONING_HARD);
    uint64_t history_count = runtime_history_count(w);
    runtime_history_entry entry;
    CHECK(runtime_history_at(w, history_count - 1, &entry) == RUNTIME_OK);
    CHECK(entry.reasoning_length == thought.reasoning_length &&
          memcmp(entry.reasoning, thought.reasoning,
                 thought.reasoning_length) == 0);

    user.text = "What is 2 plus 2? Answer with just the number.";
    CHECK(runtime_append(w, &user, &marker) == RUNTIME_OK);
    runtime_gen_params constrained = {
        .temperature = 1.0f,
        .top_k = 1,
        .top_p = 1.0f,
        .max_tokens = 24,
        .reasoning_set = 1,
        .reasoning_effort = CONVERSATION_REASONING_MEDIUM
    };
    CHECK(runtime_generate(w, &constrained) == RUNTIME_OK);
    run_result short_answer;
    run_generation(w, &short_answer, 0);
    CHECK(!short_answer.error);
    CHECK(short_answer.text_length > 0);
    CHECK(short_answer.usage.reasoning == 0);
    CHECK(short_answer.reasoning_close == CONVERSATION_REASONING_HARD);
    CHECK(thought.usage.replayed + short_answer.usage.replayed > 0);

    runtime_close(w);
    xe_engine_close(e);

    char requests_path[256], output_path[256], command[2048];
    snprintf(requests_path, sizeof requests_path, "%s/requests.ndjson",
             ndjson_dir);
    snprintf(output_path, sizeof output_path, "%s/output.ndjson",
             ndjson_dir);
    FILE *requests = fopen(requests_path, "w");
    CHECK(requests != NULL);
    if (requests) {
        fprintf(requests, "{\"op\":\"describe\"}\n");
        fprintf(requests,
                "{\"op\":\"create\",\"system\":\"You are a helpful "
                "assistant with tools.\",\"tools\":[{\"name\":"
                "\"get_weather\",\"description\":\"Get the current "
                "weather for a city.\",\"parameters\":{\"type\":"
                "\"object\",\"properties\":{\"city\":{\"type\":"
                "\"string\",\"description\":\"The city name, e.g. "
                "Kyoto\"}},\"required\":[\"city\"]}}]}\n");
        fprintf(requests,
                "{\"op\":\"append\",\"role\":\"user\",\"text\":"
                "\"What's the weather in Kyoto right now?\"}\n");
        fprintf(requests,
                "{\"op\":\"generate\",\"temperature\":1,\"top_k\":1,"
                "\"max_tokens\":200,\"seed\":42}\n");
        fprintf(requests, "{\"op\":\"pending\"}\n");
        fprintf(requests, "{\"op\":\"history\"}\n");
        fprintf(requests, "{\"op\":\"list\"}\n");
        fclose(requests);
    }
    snprintf(command, sizeof command,
             "XDG_STATE_HOME=%s ./xenolith wire %s --state %s/state "
             "--cache %s/cache < %s > %s 2>%s/stderr.log",
             ndjson_dir, model, ndjson_dir, ndjson_dir, requests_path,
             output_path, ndjson_dir);
    CHECK(system(command) == 0);
    CHECK(file_contains(output_path, "\"ok\":true"));
    CHECK(file_contains(output_path, "\"reasoning\":{\"efforts\""));
    CHECK(file_contains(output_path, "\"event\":\"start\""));
    CHECK(file_contains(output_path, "\"event\":\"toolcall_end\""));
    CHECK(file_contains(output_path, "\"name\":\"get_weather\""));
    CHECK(file_contains(output_path, "\"stop\":\"tool_use\""));
    CHECK(file_contains(output_path, "\"kind\":\"assistant\""));
    CHECK(file_contains(output_path, "\"calls\":[1]"));

    if (failures) {
        fprintf(stderr, "test_runtime_model: %d failures\n", failures);
        return 1;
    }
    printf("test_runtime_model: all checks passed\n");
    return 0;
}
