#define _GNU_SOURCE
#define _POSIX_C_SOURCE 200809L

#include "../runtime.h"
#include <time.h>

/* Deterministic engine/parser/clock fixtures exercise the real runtime event
 * loop and socket serializer without loading weights or initializing a GPU. */
static int fixture_clock_gettime(clockid_t id, struct timespec *out);
static void fixture_sync_report(xe_session *s, const xe_tokens *p, xe_sync_report *r);
static void fixture_sync(xe_session *s, const xe_tokens *p);
static int32_t fixture_next(xe_session *s, xe_sampler *sp);
static int fixture_detokenize(const xe_engine *e, int32_t t, char *b, int cap);
static int fixture_reasoning(const profile *p);
static profile_status fixture_feed(profile *p, int32_t t, profile_parse_event *out);
static int fixture_anchor_capture(xe_session *s);
static xe_session *fixture_shadow_new(xe_session *s);
static int fixture_shadow_wait(xe_session *s);
#define clock_gettime fixture_clock_gettime
#define xe_session_sync_report fixture_sync_report
#define xe_session_sync fixture_sync
#define xe_session_next fixture_next
#define xe_detokenize fixture_detokenize
#define profile_parser_reasoning fixture_reasoning
#define profile_parser_feed fixture_feed
#define xe_session_anchor_capture fixture_anchor_capture
#define xe_session_shadow_new fixture_shadow_new
#define xe_session_shadow_wait fixture_shadow_wait
#undef _POSIX_C_SOURCE
#include "../runtime.c"
#include "../serve.c"
#undef clock_gettime

static int failures;
#define CHECK(c) do { if (!(c)) { failures++; fprintf(stderr, "FAIL %s:%d: %s\n", __FILE__, __LINE__, #c); } } while (0)
static int64_t fixture_ns = INT64_C(1000000000);
static int fixture_position;
static int fixture_step;
static int fixture_error;
static int fixture_slow_finalize;
static int fixture_anchor_calls;

static int fixture_anchor_capture(xe_session *s) {
    (void)s;
    fixture_anchor_calls++;
    fixture_ns += INT64_C(5000000000);
    return 1;
}
static xe_session *fixture_shadow_new(xe_session *s) {
    (void)s;
    return NULL;
}
static int fixture_shadow_wait(xe_session *s) {
    (void)s;
    fixture_ns += INT64_C(3000000000);
    return 0;
}

static int fixture_clock_gettime(clockid_t id, struct timespec *out) {
    (void)id;
    out->tv_sec = fixture_ns / 1000000000;
    out->tv_nsec = fixture_ns % 1000000000;
    return 0;
}
static void fixture_sync_report(xe_session *s, const xe_tokens *p, xe_sync_report *r) {
    (void)s;
    r->reused = fixture_position;
    r->prefilled = p->len - fixture_position;
    r->restarted = 0;
    if (r->prefilled) fixture_ns += INT64_C(400000000);
    fixture_position = p->len;
}
static void fixture_sync(xe_session *s, const xe_tokens *p) {
    (void)s;
    fixture_position = p->len;
    fixture_ns += INT64_C(100000000);
}
static int32_t fixture_next(xe_session *s, xe_sampler *sp) {
    (void)s; (void)sp;
    fixture_ns += INT64_C(100000000);
    return ++fixture_step;
}
static int fixture_detokenize(const xe_engine *e, int32_t t, char *b, int cap) {
    (void)e; (void)t; (void)cap;
    *b = 'x';
    return 1;
}
static int fixture_reasoning(const profile *p) {
    (void)p;
    if (fixture_slow_finalize) {
        fixture_ns += INT64_C(5000000000);
        fixture_slow_finalize = 0;
    }
    return 0;
}
static profile_status fixture_feed(profile *p, int32_t t, profile_parse_event *out) {
    (void)p;
    memset(out, 0, sizeof *out);
    if (fixture_error) return PROFILE_NOMEM;
    if (t == 1) {
        out->kind = PROFILE_PARSE_TEXT;
        out->text = (const uint8_t *)"hello";
        out->text_length = 5;
    } else if (t == 2) {
        out->kind = PROFILE_PARSE_CALL_START;
        out->call_name = "read";
    } else if (t == 6) {
        out->kind = PROFILE_PARSE_CALL_END;
        out->arguments_json = "{\"path\":\"main.c\"}";
    } else if (t == 7) {
        out->kind = PROFILE_PARSE_STOP;
        out->include_token = 1;
        out->stop_reason = PROFILE_STOP_TOOL_CALLS;
    }
    return PROFILE_OK;
}

static runtime make_runtime(int cached) {
    runtime w = {0};
    w.gen_kind = RUNTIME_GEN_EPHEMERAL;
    w.context = 2048;
    w.prompt_length = 1024;
    w.prompt_synced = cached;
    w.prompt = calloc((size_t)w.context, sizeof *w.prompt);
    w.inference.prefill.total = (uint64_t)(1024 - cached);
    w.reasoning_policy.hard_tokens = -1;
    w.call_open = -1;
    fixture_position = cached;
    fixture_step = 0;
    fixture_error = 0;
    fixture_slow_finalize = 0;
    fixture_anchor_calls = 0;
    return w;
}
static void free_runtime(runtime *w) {
    runtime_calls_reset(w);
    free(w->calls);
    free(w->prompt);
    json_writer_free(&w->content);
    json_writer_free(&w->reasoning);
    json_writer_free(&w->render);
}
static runtime_event next_progress(runtime *w, uint32_t phase, uint32_t state) {
    runtime_event event;
    CHECK(runtime_next_event(w, &event) == RUNTIME_OK);
    CHECK(event.kind == RUNTIME_EVENT_INFERENCE_PROGRESS);
    CHECK(event.phase == phase);
    CHECK(event.state == state);
    return event;
}
static void start_cached_decode(runtime *w) {
    runtime_event event;
    CHECK(runtime_next_event(w, &event) == RUNTIME_OK);
    CHECK(event.kind == RUNTIME_EVENT_START);
    event = next_progress(w, RUNTIME_INFERENCE_PREFILL, RUNTIME_INFERENCE_RUNNING);
    CHECK(event.progress.tokens == 0 && event.progress.total == 0);
    CHECK(event.progress.elapsed_ms == 0);
    event = next_progress(w, RUNTIME_INFERENCE_PREFILL, RUNTIME_INFERENCE_FINISHED);
    CHECK(event.progress.tokens == 0 && event.progress.total == 0);
    event = next_progress(w, RUNTIME_INFERENCE_DECODE, RUNTIME_INFERENCE_RUNNING);
    CHECK(event.progress.tokens == 0 && event.progress.elapsed_ms == 0);
}
static void test_stream(void) {
    runtime w = make_runtime(0);
    runtime_event event;
    CHECK(runtime_next_event(&w, &event) == RUNTIME_OK);
    CHECK(event.kind == RUNTIME_EVENT_START);
    int64_t before = fixture_ns;
    event = next_progress(&w, RUNTIME_INFERENCE_PREFILL, RUNTIME_INFERENCE_RUNNING);
    CHECK(fixture_position == 0 && fixture_ns == before);
    CHECK(event.progress.tokens == 0 && event.progress.total == 1024);
    CHECK(event.progress.elapsed_ms == 0);
    for (int i = 1; i <= 2; i++) {
        CHECK(runtime_next_event(&w, &event) == RUNTIME_OK);
        CHECK(event.kind == RUNTIME_EVENT_INFERENCE_PROGRESS);
        CHECK(event.phase == RUNTIME_INFERENCE_PREFILL);
        CHECK(event.state == (i == 2 ? RUNTIME_INFERENCE_FINISHED : RUNTIME_INFERENCE_RUNNING));
        CHECK(event.progress.tokens == (uint64_t)(i * 512));
        CHECK(event.progress.total == 1024);
        CHECK(event.progress.elapsed_ms == (uint64_t)(i * 400));
    }
    CHECK(runtime_next_event(&w, &event) == RUNTIME_OK);
    CHECK(event.kind == RUNTIME_EVENT_INFERENCE_PROGRESS);
    CHECK(event.phase == RUNTIME_INFERENCE_DECODE);
    CHECK(event.state == RUNTIME_INFERENCE_RUNNING);
    CHECK(event.progress.tokens == 0 && event.progress.elapsed_ms == 0);
    fixture_ns += INT64_C(10000000000); /* client pause excluded */
    CHECK(runtime_next_event(&w, &event) == RUNTIME_OK);
    CHECK(event.kind == RUNTIME_EVENT_TEXT_DELTA);
    CHECK(event.text_length == 5 && !memcmp(event.text, "hello", 5));
    CHECK(runtime_next_event(&w, &event) == RUNTIME_OK);
    CHECK(event.kind == RUNTIME_EVENT_TOOLCALL_START);
    int progress_during_arguments = 0;
    int tool_end = 0;
    int decode_finished = 0;
    runtime_inference_measure last_decode = {0};
    for (int i = 0; i < 20; i++) {
        CHECK(runtime_next_event(&w, &event) == RUNTIME_OK);
        if (event.kind == RUNTIME_EVENT_INFERENCE_PROGRESS) {
            CHECK(!decode_finished);
            CHECK(event.phase == RUNTIME_INFERENCE_DECODE);
            CHECK(event.progress.tokens >= last_decode.tokens);
            CHECK(event.progress.elapsed_ms >= last_decode.elapsed_ms);
            CHECK(event.progress.elapsed_ms <= 1300);
            last_decode = event.progress;
            if (!tool_end && event.progress.tokens >= 4)
                progress_during_arguments = 1;
            if (event.state == RUNTIME_INFERENCE_FINISHED) {
                decode_finished++;
                CHECK(event.progress.tokens == 7);
                fixture_slow_finalize = 1;
            }
        } else if (event.kind == RUNTIME_EVENT_TOOLCALL_END) {
            tool_end++;
            CHECK(!strcmp(event.arguments_json, "{\"path\":\"main.c\"}"));
        } else if (event.kind == RUNTIME_EVENT_DONE) {
            CHECK(event.inference.prefill.tokens == 1024);
            CHECK(event.inference.prefill.elapsed_ms == 800);
            CHECK(event.inference.decode.tokens == event.usage.output);
            CHECK(event.inference.decode.tokens == 7);
            CHECK(event.inference.decode.elapsed_ms == 1300);
            CHECK(event.inference.decode.elapsed_ms == last_decode.elapsed_ms);
            CHECK(event.stop == RUNTIME_STOP_TOOL_USE);
            break;
        } else CHECK(0);
    }
    CHECK(event.kind == RUNTIME_EVENT_DONE);
    CHECK(progress_during_arguments && tool_end == 1 && fixture_step == 7);
    CHECK(decode_finished == 1 && fixture_slow_finalize == 0);
    free_runtime(&w);
}
static void test_cancel_and_error(void) {
    runtime w = make_runtime(1024);
    runtime_event event;
    CHECK(runtime_next_event(&w, &event) == RUNTIME_OK);
    event = next_progress(&w, RUNTIME_INFERENCE_PREFILL, RUNTIME_INFERENCE_RUNNING);
    CHECK(event.progress.tokens == 0 && event.progress.total == 0);
    CHECK(runtime_cancel(&w) == RUNTIME_OK);
    event = next_progress(&w, RUNTIME_INFERENCE_PREFILL, RUNTIME_INFERENCE_FINISHED);
    CHECK(event.progress.tokens == 0);
    CHECK(event.progress.elapsed_ms == 0);
    CHECK(runtime_next_event(&w, &event) == RUNTIME_OK);
    CHECK(event.kind == RUNTIME_EVENT_DONE && event.stop == RUNTIME_STOP_ABORTED);
    CHECK(event.inference.decode.tokens == 0 && event.inference.decode.elapsed_ms == 0);
    free_runtime(&w);

    w = make_runtime(0);
    CHECK(runtime_next_event(&w, &event) == RUNTIME_OK);
    CHECK(runtime_cancel(&w) == RUNTIME_OK);
    CHECK(runtime_next_event(&w, &event) == RUNTIME_OK);
    CHECK(event.kind == RUNTIME_EVENT_DONE && event.inference.prefill.elapsed_ms == 0);
    CHECK(event.stop == RUNTIME_STOP_ABORTED && fixture_position == 0);
    free_runtime(&w);

    w = make_runtime(1024);
    start_cached_decode(&w);
    fixture_error = 1;
    CHECK(runtime_next_event(&w, &event) == RUNTIME_OK);
    CHECK(event.kind == RUNTIME_EVENT_ERROR);
    CHECK(event.inference.decode.tokens == 0 && event.inference.decode.elapsed_ms == 100);
    CHECK(runtime_next_event(&w, &event) == RUNTIME_INVALID_ARGUMENT);
    free_runtime(&w);

    w = make_runtime(1024);
    start_cached_decode(&w);
    CHECK(runtime_next_event(&w, &event) == RUNTIME_OK);
    CHECK(event.kind == RUNTIME_EVENT_TEXT_DELTA);
    CHECK(runtime_cancel(&w) == RUNTIME_OK);
    event = next_progress(&w, RUNTIME_INFERENCE_DECODE, RUNTIME_INFERENCE_FINISHED);
    CHECK(event.progress.tokens == 1 && event.progress.elapsed_ms == 200);
    CHECK(runtime_next_event(&w, &event) == RUNTIME_OK);
    CHECK(event.kind == RUNTIME_EVENT_DONE && event.stop == RUNTIME_STOP_ABORTED);
    CHECK(event.inference.decode.tokens == 1);
    free_runtime(&w);
}

static void test_partial_and_short(void) {
    runtime w = make_runtime(0);
    runtime_event event;
    CHECK(runtime_next_event(&w, &event) == RUNTIME_OK);
    next_progress(&w, RUNTIME_INFERENCE_PREFILL, RUNTIME_INFERENCE_RUNNING);
    CHECK(runtime_next_event(&w, &event) == RUNTIME_OK);
    CHECK(event.progress.tokens == 512 && event.progress.total == 1024);
    CHECK(runtime_cancel(&w) == RUNTIME_OK);
    event = next_progress(&w, RUNTIME_INFERENCE_PREFILL, RUNTIME_INFERENCE_FINISHED);
    CHECK(event.progress.tokens == 512);
    CHECK(event.progress.elapsed_ms == 400 && event.progress.total == 1024);
    fixture_slow_finalize = 1;
    CHECK(runtime_next_event(&w, &event) == RUNTIME_OK);
    CHECK(event.kind == RUNTIME_EVENT_DONE && event.inference.prefill.tokens == 512);
    CHECK(event.inference.decode.elapsed_ms == 0);
    CHECK(fixture_slow_finalize == 0 && fixture_step == 0);
    free_runtime(&w);

    w = make_runtime(1024);
    w.gen_max_tokens = 1;
    start_cached_decode(&w);
    CHECK(runtime_next_event(&w, &event) == RUNTIME_OK);
    CHECK(event.kind == RUNTIME_EVENT_TEXT_DELTA);
    CHECK(runtime_next_event(&w, &event) == RUNTIME_OK);
    CHECK(event.kind == RUNTIME_EVENT_INFERENCE_PROGRESS);
    CHECK(event.phase == RUNTIME_INFERENCE_DECODE && event.progress.tokens == 1);
    CHECK(event.state == RUNTIME_INFERENCE_FINISHED);
    CHECK(event.progress.elapsed_ms == 200); /* shorter than throttle */
    CHECK(runtime_next_event(&w, &event) == RUNTIME_OK);
    CHECK(event.kind == RUNTIME_EVENT_DONE && event.stop == RUNTIME_STOP_LENGTH);
    CHECK(event.inference.decode.tokens == event.usage.output);
    CHECK(fixture_step == 1);
    free_runtime(&w);
}

static void test_late_cancel(void) {
    runtime w = make_runtime(1024);
    w.gen_max_tokens = 1;
    start_cached_decode(&w);
    runtime_event event;
    CHECK(runtime_next_event(&w, &event) == RUNTIME_OK);
    CHECK(event.kind == RUNTIME_EVENT_TEXT_DELTA);
    event = next_progress(&w, RUNTIME_INFERENCE_DECODE, RUNTIME_INFERENCE_FINISHED);
    runtime_inference frozen = w.inference;
    int steps = fixture_step;
    CHECK(event.progress.tokens == 1 && event.progress.elapsed_ms == 200);
    CHECK(runtime_cancel(&w) == RUNTIME_OK);
    fixture_slow_finalize = 1;
    /* Cancellation accepted after the stop decision does not change it. */
    CHECK(runtime_next_event(&w, &event) == RUNTIME_OK);
    CHECK(event.kind == RUNTIME_EVENT_DONE && event.stop == RUNTIME_STOP_LENGTH);
    CHECK(event.usage.output == frozen.decode.tokens);
    CHECK(event.inference.decode.tokens == frozen.decode.tokens);
    CHECK(event.inference.decode.elapsed_ms == frozen.decode.elapsed_ms);
    CHECK(event.inference.prefill.tokens == frozen.prefill.tokens);
    CHECK(event.inference.prefill.total == frozen.prefill.total);
    CHECK(event.inference.prefill.elapsed_ms == frozen.prefill.elapsed_ms);
    CHECK(fixture_step == steps && fixture_slow_finalize == 0);
    CHECK(runtime_next_event(&w, &event) == RUNTIME_INVALID_ARGUMENT);
    free_runtime(&w);
}

static void test_preparation_boundaries(void) {
    runtime w = make_runtime(0);
    runtime_event event;
    w.capture_anchor = 1;
    CHECK(runtime_next_event(&w, &event) == RUNTIME_OK);
    /* The initial shadow wait precedes the phase opening, not the first sync. */
    w.shadow = (xe_session *)&w;
    w.shadow_inflight = 1;
    int64_t before = fixture_ns;
    event = next_progress(&w, RUNTIME_INFERENCE_PREFILL, RUNTIME_INFERENCE_RUNNING);
    w.shadow = NULL;
    CHECK(fixture_ns == before + INT64_C(3000000000));
    CHECK(fixture_position == 0 && event.progress.elapsed_ms == 0);
    next_progress(&w, RUNTIME_INFERENCE_PREFILL, RUNTIME_INFERENCE_RUNNING);
    event = next_progress(&w, RUNTIME_INFERENCE_PREFILL, RUNTIME_INFERENCE_FINISHED);
    CHECK(fixture_anchor_calls == 0 && event.progress.elapsed_ms == 800);
    before = fixture_ns;
    event = next_progress(&w, RUNTIME_INFERENCE_DECODE, RUNTIME_INFERENCE_RUNNING);
    CHECK(fixture_anchor_calls == 1 && fixture_ns == before + INT64_C(5000000000));
    CHECK(event.progress.tokens == 0 && event.progress.elapsed_ms == 0);
    CHECK(w.inference.prefill.elapsed_ms == 800);
    free_runtime(&w);

    /* Cancellation between phases neither runs setup nor opens decode. */
    w = make_runtime(1024);
    w.capture_anchor = 1;
    CHECK(runtime_next_event(&w, &event) == RUNTIME_OK);
    next_progress(&w, RUNTIME_INFERENCE_PREFILL, RUNTIME_INFERENCE_RUNNING);
    next_progress(&w, RUNTIME_INFERENCE_PREFILL, RUNTIME_INFERENCE_FINISHED);
    CHECK(runtime_cancel(&w) == RUNTIME_OK);
    CHECK(runtime_next_event(&w, &event) == RUNTIME_OK);
    CHECK(event.kind == RUNTIME_EVENT_DONE && event.stop == RUNTIME_STOP_ABORTED);
    CHECK(fixture_anchor_calls == 0 && fixture_step == 0);
    free_runtime(&w);
}

static json_value *serialize(runtime_event *event) {
    serve s = {0};
    serve_conn c = {0}; /* inactive: inspect actual event serializer output */
    serve_emit_event(&s, &c, event);
    json_value *value = json_parse(s.out.data, s.out.length);
    CHECK(value != NULL);
    json_writer_free(&s.out);
    return value;
}
static void test_serialization(void) {
    runtime_event event = {0};
    event.kind = RUNTIME_EVENT_INFERENCE_PROGRESS;
    event.phase = RUNTIME_INFERENCE_PREFILL;
    event.progress = (runtime_inference_measure){512, 1024, 250};
    json_value *v = serialize(&event);
    CHECK(!strcmp(json_member(v, "event")->text, "inference_progress"));
    CHECK(!strcmp(json_member(v, "phase")->text, "prefill"));
    CHECK(!strcmp(json_member(v, "state")->text, "running"));
    CHECK(json_member(v, "tokens")->number == 512);
    CHECK(json_member(v, "total")->number == 1024);
    CHECK(json_member(v, "elapsed_ms")->number == 250);
    json_free(v);
    event.phase = RUNTIME_INFERENCE_DECODE;
    event.state = RUNTIME_INFERENCE_FINISHED;
    v = serialize(&event);
    CHECK(!strcmp(json_member(v, "phase")->text, "decode"));
    CHECK(!strcmp(json_member(v, "state")->text, "finished"));
    CHECK(json_member(v, "total") == NULL);
    json_free(v);
    event.kind = RUNTIME_EVENT_DONE;
    event.stop = RUNTIME_STOP_LENGTH;
    event.marker = RUNTIME_MARKER_NONE;
    event.usage.output = 9;
    event.inference.prefill = (runtime_inference_measure){0, 0, 0};
    event.inference.decode = (runtime_inference_measure){9, 0, 300};
    v = serialize(&event);
    const json_value *inference = json_member(v, "inference");
    CHECK(json_member(json_member(inference, "prefill"), "total")->number == 0);
    CHECK(json_member(json_member(inference, "decode"), "tokens")->number == 9);
    CHECK(json_member(json_member(inference, "decode"), "total") == NULL);
    CHECK(json_member(json_member(v, "usage"), "output")->number == 9);
    json_free(v);
    event.kind = RUNTIME_EVENT_ERROR;
    event.error = RUNTIME_NOMEM;
    v = serialize(&event);
    CHECK(json_member(v, "inference") != NULL);
    CHECK(!strcmp(json_member(v, "code")->text, "out_of_memory"));
    json_free(v);
}
int main(void) {
    test_stream();
    test_cancel_and_error();
    test_partial_and_short();
    test_late_cancel();
    test_preparation_boundaries();
    test_serialization();
    printf("inference: %s\n", failures ? "FAIL" : "ok");
    return failures ? 1 : 0;
}
