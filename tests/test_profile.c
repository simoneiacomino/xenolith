#include "../profile.h"
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

static void check_render(const profile_render *render,
                         const char *expected, int line) {
    size_t length = strlen(expected);
    if (render->render_length != length ||
        memcmp(render->render, expected, length) != 0) {
        failures++;
        fprintf(stderr, "FAIL %s:%d: render mismatch\n  expected: %s\n"
                        "  actual:   %.*s\n",
                __FILE__, line, expected, (int)render->render_length,
                (const char *)render->render);
    }
}

typedef struct {
    int32_t v[4096];
    int n;
} sequence;

static void seq_id(sequence *seq, int32_t id) {
    seq->v[seq->n++] = id;
}

static void seq_text(const xe_engine *e, sequence *seq, const char *text) {
    seq->n += xe_encode_text(e, text, seq->v + seq->n,
                             (int)(sizeof seq->v / sizeof seq->v[0]) -
                             seq->n);
}

typedef struct {
    char text[4096];
    size_t text_length;
    char reasoning[4096];
    size_t reasoning_length;
    char call_names[8][64];
    char call_args[8][1024];
    size_t starts;
    size_t ends;
    int stopped;
    uint32_t stop_reason;
    int include_token;
} collected;

static void collect_open(profile *p, const sequence *seq, collected *out,
                         int reasoning_open) {
    memset(out, 0, sizeof *out);
    profile_parser_reset(p, reasoning_open);
    for (int i = 0; i < seq->n && !out->stopped; i++) {
        profile_parse_event event;
        CHECK(profile_parser_feed(p, seq->v[i], &event) == PROFILE_OK);
        switch (event.kind) {
        case PROFILE_PARSE_REASONING:
            memcpy(out->reasoning + out->reasoning_length, event.text,
                   event.text_length);
            out->reasoning_length += event.text_length;
            out->reasoning[out->reasoning_length] = '\0';
            break;
        case PROFILE_PARSE_TEXT:
            memcpy(out->text + out->text_length, event.text,
                   event.text_length);
            out->text_length += event.text_length;
            out->text[out->text_length] = '\0';
            break;
        case PROFILE_PARSE_CALL_START:
            if (out->starts < 8)
                snprintf(out->call_names[out->starts], 64, "%s",
                         event.call_name);
            out->starts++;
            break;
        case PROFILE_PARSE_CALL_END:
            if (out->ends < 8)
                snprintf(out->call_args[out->ends], 1024, "%s",
                         event.arguments_json);
            out->ends++;
            break;
        case PROFILE_PARSE_STOP:
            out->stopped = 1;
            out->stop_reason = event.stop_reason;
            out->include_token = event.include_token;
            break;
        default:
            break;
        }
    }
}

static void collect(profile *p, const sequence *seq, collected *out) {
    collect_open(p, seq, out, 0);
}

int main(int argc, char **argv) {
    if (argc < 2 || !argv[1][0]) {
        fprintf(stderr, "usage: %s <model.gguf>\n", argv[0]);
        return 2;
    }
    const char *model = argv[1];
    xe_engine *e = xe_engine_open_vocab(model);
    profile *p = NULL;
    CHECK(profile_open(&p, e) == PROFILE_OK);
    if (!p) return 1;
    CHECK(strcmp(profile_model(p), "gemma-4-26B-A4B-it-qat") == 0);

    CHECK(profile_reasoning_prefix_valid("  È un prefisso.\n"));
    CHECK(!profile_reasoning_prefix_valid("<channel|>"));
    CHECK(!profile_reasoning_prefix_valid("\xc0\x80"));
    profile_render prefixed;
    CHECK(profile_render_reasoning_prefix(p,"  È un prefisso.\n",PROFILE_TURN_PADDED,&prefixed) == PROFILE_OK);
    CHECK(prefixed.reasoning_prefix_tokens > 0);
    CHECK(prefixed.render_length >= strlen("  È un prefisso.\n"));
    CHECK(!memcmp(prefixed.render + prefixed.render_length - strlen("  È un prefisso.\n"),
                  "  È un prefisso.\n",strlen("  È un prefisso.\n")));

    profile_reasoning_policy policy;
    CHECK(profile_get_reasoning_policy(
              p, PROFILE_REASONING_LOW, -1, 1000, 2000, &policy) ==
          PROFILE_OK);
    CHECK(policy.hard_tokens == 128 && policy.soft_tokens == 96 &&
          policy.delimiter_rank == 3 && policy.delimiter_margin == 3.0f);
    CHECK(profile_get_reasoning_policy(
              p, PROFILE_REASONING_MEDIUM, -1, 300, 2000, &policy) ==
          PROFILE_OK);
    CHECK(policy.hard_tokens == 44 && policy.soft_tokens == 0);
    CHECK(profile_get_reasoning_policy(
              p, PROFILE_REASONING_HIGH, 17, 1000, 2000, &policy) ==
          PROFILE_OK);
    CHECK(policy.hard_tokens == 17 && policy.soft_tokens == 0);
    CHECK(profile_get_reasoning_policy(
              p, PROFILE_REASONING_MAX, -1, 0, 1000, &policy) ==
          PROFILE_OK);
    CHECK(policy.hard_tokens == 744 && policy.soft_tokens == 232);
    CHECK(profile_get_reasoning_policy(
              p, PROFILE_REASONING_LOW, -1, 24, 2000, &policy) ==
          PROFILE_OK);
    CHECK(policy.hard_tokens == 0 && policy.soft_tokens == 0);

    profile_render render;

    CHECK(profile_render_system(p, NULL, NULL, 0, 0, &render) == PROFILE_OK);
    check_render(&render, "<bos>", __LINE__);
    CHECK(render.token_count == 1 && render.tokens[0] == 2);

    CHECK(profile_render_user(p, " hi \n", PROFILE_TURN_PADDED, &render)
          == PROFILE_OK);
    check_render(&render, "<|turn>user\nhi<turn|>\n", __LINE__);
    CHECK(render.tokens[0] == 105);

    CHECK(profile_render_reply_open(p, PROFILE_TURN_PADDED, 0, 0, &render)
          == PROFILE_OK);
    check_render(&render, "<|turn>model\n<|channel>thought\n<channel|>",
                 __LINE__);
    CHECK(render.token_count >= 5 &&
          render.tokens[render.token_count - 1] == 101);

    CHECK(profile_render_reply_open(p, PROFILE_TURN_OPEN, 0, 0, &render)
          == PROFILE_OK);
    CHECK(render.token_count == 0 && render.render_length == 0);

    CHECK(profile_render_system(p, NULL, NULL, 0, 1, &render) == PROFILE_OK);
    check_render(&render,
                 "<bos><|turn>system\n<|think|>\n<turn|>\n", __LINE__);

    CHECK(profile_render_reply_open(p, PROFILE_TURN_PADDED, 1, 0,
                                    &render) == PROFILE_OK);
    check_render(&render, "<|turn>model\n", __LINE__);

    CHECK(profile_render_reply_open(p, PROFILE_TURN_OPEN, 1, 1,
                                    &render) == PROFILE_OK);
    check_render(&render, "<|channel>thought\n", __LINE__);

    CHECK(profile_render_user(p, "next", PROFILE_TURN_OPEN, &render)
          == PROFILE_OK);
    check_render(&render, "<turn|>\n<|turn>user\nnext<turn|>\n", __LINE__);
    CHECK(render.tokens[0] == 106);

    CHECK(profile_render_user(p, "again", PROFILE_TURN_BARE, &render)
          == PROFILE_OK);
    check_render(&render, "\n<|turn>user\nagain<turn|>\n", __LINE__);

    profile_tool weather = {
        "get_weather",
        "Get the current weather for a city.",
        "{\"type\":\"object\",\"properties\":{\"city\":{\"type\":"
        "\"string\",\"description\":\"The city name, e.g. Kyoto\"}},"
        "\"required\":[\"city\"]}"
    };
    profile_tool stock = {
        "get_stock_price",
        "Get the latest stock price for a ticker symbol.",
        "{\"type\":\"object\",\"properties\":{\"ticker\":{\"type\":"
        "\"string\",\"description\":\"Ticker symbol, e.g. AAPL\"}},"
        "\"required\":[\"ticker\"]}"
    };
    profile_tool both[2];
    both[0] = weather;
    both[1] = stock;
    CHECK(profile_render_system(p, "You are a helpful assistant with tools.",
                                both, 2, 0, &render) == PROFILE_OK);
    check_render(&render,
        "<bos><|turn>system\nYou are a helpful assistant with tools."
        "<|tool>declaration:get_weather{description:<|\"|>Get the current "
        "weather for a city.<|\"|>,parameters:{properties:{city:"
        "{description:<|\"|>The city name, e.g. Kyoto<|\"|>,type:<|\"|>"
        "STRING<|\"|>}},required:[<|\"|>city<|\"|>],type:<|\"|>OBJECT"
        "<|\"|>}}<tool|><|tool>declaration:get_stock_price{description:"
        "<|\"|>Get the latest stock price for a ticker symbol.<|\"|>,"
        "parameters:{properties:{ticker:{description:<|\"|>Ticker symbol, "
        "e.g. AAPL<|\"|>,type:<|\"|>STRING<|\"|>}},required:[<|\"|>ticker"
        "<|\"|>],type:<|\"|>OBJECT<|\"|>}}<tool|><turn|>\n", __LINE__);

    profile_tool rich = {
        "get_weather",
        "Get current weather for a city.",
        "{\"type\":\"object\",\"properties\":{"
        "\"city\":{\"type\":\"string\",\"description\":\"City name\"},"
        "\"days\":{\"type\":\"integer\",\"description\":\"Forecast days\"},"
        "\"units\":{\"type\":\"string\",\"description\":\"Unit system\","
        "\"enum\":[\"celsius\",\"fahrenheit\"]}},"
        "\"required\":[\"city\"]}"
    };
    CHECK(profile_render_system(p, NULL, &rich, 1, 0, &render) == PROFILE_OK);
    check_render(&render,
        "<bos><|turn>system\n"
        "<|tool>declaration:get_weather{description:<|\"|>Get current "
        "weather for a city.<|\"|>,parameters:{properties:{city:"
        "{description:<|\"|>City name<|\"|>,type:<|\"|>STRING<|\"|>},"
        "days:{description:<|\"|>Forecast days<|\"|>,type:<|\"|>INTEGER"
        "<|\"|>},units:{description:<|\"|>Unit system<|\"|>,enum:[<|\"|>"
        "celsius<|\"|>,<|\"|>fahrenheit<|\"|>],type:<|\"|>STRING<|\"|>}},"
        "required:[<|\"|>city<|\"|>],type:<|\"|>OBJECT<|\"|>}}<tool|>"
        "<turn|>\n", __LINE__);

    CHECK(profile_render_tool_result(p, "get_weather", "21C", &render)
          == PROFILE_OK);
    check_render(&render,
        "<|tool_response>response:get_weather{value:<|\"|>21C<|\"|>}"
        "<tool_response|>", __LINE__);
    CHECK(render.tokens[0] == 50 &&
          render.tokens[render.token_count - 1] == 51);

    profile_call call = { "get_weather", "{\"city\":\"Kyoto\"}" };
    CHECK(profile_render_assistant(p, NULL, 1, NULL, &call, 1,
                                   PROFILE_TURN_PADDED, 1, &render)
          == PROFILE_OK);
    check_render(&render,
        "<|turn>model\n"
        "<|tool_call>call:get_weather{city:<|\"|>Kyoto<|\"|>}<tool_call|>",
        __LINE__);
    CHECK(render.tokens[render.token_count - 1] == 49);

    CHECK(profile_render_assistant(p, NULL, 1, "done", NULL, 0,
                                   PROFILE_TURN_OPEN, 1, &render)
          == PROFILE_OK);
    check_render(&render, "done<turn|>", __LINE__);
    CHECK(render.tokens[render.token_count - 1] == 106);

    profile_call sorted_call = { "set", "{\"b\":2,\"a\":\"x\"}" };
    CHECK(profile_render_assistant(p, NULL, 1, NULL, &sorted_call, 1,
                                   PROFILE_TURN_OPEN, 1, &render)
          == PROFILE_OK);
    check_render(&render,
        "<|tool_call>call:set{a:<|\"|>x<|\"|>,b:2}<tool_call|>", __LINE__);

    CHECK(profile_render_assistant(p, " plan ", 1, "done", NULL, 0,
                                   PROFILE_TURN_PADDED, 1, &render)
          == PROFILE_OK);
    check_render(&render,
        "<|turn>model\n<|channel>thought\nplan\n<channel|>done<turn|>",
        __LINE__);

    sequence seq;
    collected got;

    seq.n = 0;
    seq_text(e, &seq, "Hello there");
    seq_id(&seq, 106);
    collect(p, &seq, &got);
    CHECK(strcmp(got.text, "Hello there") == 0);
    CHECK(got.stopped && got.stop_reason == PROFILE_STOP_EOT &&
          got.include_token == 1);

    seq.n = 0;
    seq_id(&seq, 1);
    collect(p, &seq, &got);
    CHECK(got.stopped && got.stop_reason == PROFILE_STOP_EOS);

    seq.n = 0;
    seq_text(e, &seq, "Check.");
    seq_id(&seq, 48);
    seq_text(e, &seq, "call:get_weather{city:");
    seq_id(&seq, 52);
    seq_text(e, &seq, "Kyoto");
    seq_id(&seq, 52);
    seq_text(e, &seq, "}");
    seq_id(&seq, 49);
    seq_id(&seq, 50);
    collect(p, &seq, &got);
    CHECK(strcmp(got.text, "Check.") == 0);
    CHECK(got.starts == 1 && got.ends == 1);
    CHECK(strcmp(got.call_names[0], "get_weather") == 0);
    CHECK(strcmp(got.call_args[0], "{\"city\":\"Kyoto\"}") == 0);
    CHECK(got.stopped && got.stop_reason == PROFILE_STOP_TOOL_CALLS &&
          got.include_token == 0);
    CHECK(profile_parser_calls(p) == 1);

    seq.n = 0;
    seq_id(&seq, 48);
    seq_text(e, &seq, "call:get_weather{city:");
    seq_id(&seq, 52);
    seq_text(e, &seq, "Kyoto");
    seq_id(&seq, 52);
    seq_text(e, &seq, "}");
    seq_id(&seq, 49);
    seq_id(&seq, 48);
    seq_text(e, &seq, "call:get_weather{city:");
    seq_id(&seq, 52);
    seq_text(e, &seq, "Osaka");
    seq_id(&seq, 52);
    seq_text(e, &seq, "}");
    seq_id(&seq, 49);
    seq_id(&seq, 50);
    collect(p, &seq, &got);
    CHECK(got.starts == 2 && got.ends == 2);
    CHECK(strcmp(got.call_args[1], "{\"city\":\"Osaka\"}") == 0);
    CHECK(profile_parser_calls(p) == 2);

    seq.n = 0;
    seq_id(&seq, 48);
    seq_text(e, &seq, "call:noop{}");
    seq_id(&seq, 49);
    seq_id(&seq, 50);
    collect(p, &seq, &got);
    CHECK(got.starts == 1 && got.ends == 1);
    CHECK(strcmp(got.call_args[0], "{}") == 0);

    seq.n = 0;
    seq_id(&seq, 48);
    seq_text(e, &seq,
             "call:mix{flag:true,n:-7,x:3.14,list:[1,2],obj:{a:null}}");
    seq_id(&seq, 49);
    seq_id(&seq, 50);
    collect(p, &seq, &got);
    CHECK(got.ends == 1);
    CHECK(strcmp(got.call_args[0],
                 "{\"flag\":true,\"n\":-7,\"x\":3.14,\"list\":[1,2],"
                 "\"obj\":{\"a\":null}}") == 0);

    seq.n = 0;
    seq_id(&seq, 48);
    seq_text(e, &seq, "call:write_file{content:");
    seq_id(&seq, 52);
    seq_text(e, &seq, "line1 has a brace } and a comma , and \"quotes\"\n"
                      "line2 ends here");
    seq_id(&seq, 52);
    seq_text(e, &seq, ",path:");
    seq_id(&seq, 52);
    seq_text(e, &seq, "/tmp/x.txt");
    seq_id(&seq, 52);
    seq_text(e, &seq, "}");
    seq_id(&seq, 49);
    seq_id(&seq, 50);
    collect(p, &seq, &got);
    CHECK(got.ends == 1);
    json_value *arguments = json_parse(got.call_args[0],
                                       strlen(got.call_args[0]));
    CHECK(arguments != NULL);
    if (arguments) {
        const json_value *content = json_member(arguments, "content");
        CHECK(content && strcmp(content->text,
              "line1 has a brace } and a comma , and \"quotes\"\n"
              "line2 ends here") == 0);
        const json_value *path = json_member(arguments, "path");
        CHECK(path && strcmp(path->text, "/tmp/x.txt") == 0);
        json_free(arguments);
    }

    seq.n = 0;
    seq_id(&seq, 48);
    seq_text(e, &seq, "call:rm{path:");
    seq_id(&seq, 106);
    collect(p, &seq, &got);
    CHECK(got.stopped && got.stop_reason == PROFILE_STOP_EOT);
    CHECK(got.ends == 0 && profile_parser_calls(p) == 0);

    seq.n = 0;
    seq_id(&seq, 100);
    seq_text(e, &seq, "thought\n");
    seq_text(e, &seq, "I am thinking about {braces} here.");
    seq_id(&seq, 101);
    seq_text(e, &seq, "answer");
    seq_id(&seq, 106);
    collect(p, &seq, &got);
    CHECK(strcmp(got.text, "answer") == 0);
    CHECK(strcmp(got.reasoning,
                 "I am thinking about {braces} here.") == 0);

    seq.n = 0;
    seq_text(e, &seq, "continuing");
    seq_id(&seq, 101);
    seq_text(e, &seq, "answer");
    seq_id(&seq, 106);
    collect_open(p, &seq, &got, 1);
    CHECK(strcmp(got.reasoning, "continuing") == 0);
    CHECK(strcmp(got.text, "answer") == 0);

    seq.n = 0;
    seq_id(&seq, 101);
    seq_text(e, &seq, "text");
    seq_id(&seq, 106);
    collect(p, &seq, &got);
    CHECK(strcmp(got.text, "text") == 0);

    seq.n = 0;
    seq_id(&seq, 100);
    seq_text(e, &seq, "thought\n");
    seq_id(&seq, 48);
    seq_text(e, &seq, "call:evil{}");
    seq_id(&seq, 49);
    seq_id(&seq, 101);
    seq_text(e, &seq, "ok");
    seq_id(&seq, 106);
    collect(p, &seq, &got);
    CHECK(got.starts == 0 && got.ends == 0);
    CHECK(strcmp(got.text, "ok") == 0);

    profile_close(p);
    xe_engine_close(e);

    if (failures) {
        fprintf(stderr, "test_profile: %d failures\n", failures);
        return 1;
    }
    printf("test_profile: all checks passed\n");
    return 0;
}
