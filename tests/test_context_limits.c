/* Real tokenizer, CLI validation, conversation store and wire dispatch; only
 * model loading/session allocation are replaced to keep these checks CPU-only. */
#include "../xenolith.c"
#undef _POSIX_C_SOURCE
#include "../runtime.c"
#include "../serve.c"
#include <assert.h>
#include <sys/wait.h>

static xe_engine fixture_engine;
static xe_engine *fixture_open(const char *path, int context) {
    (void)path;
    fixture_engine.context = context;
    return &fixture_engine;
}
static void fixture_close(xe_engine *e) { (void)e; }
static xe_session *fixture_session(xe_engine *e) { (void)e; _exit(0); }
static profile_status fixture_profile(profile **out, xe_engine *e) {
    (void)out; (void)e; _exit(0);
}
#define main cli_main
#define xe_engine_open_with_context fixture_open
#define xe_engine_close fixture_close
#define xe_session_new fixture_session
#define profile_open fixture_profile
#include "../main.c"
#undef main
#undef xe_engine_open_with_context
#undef xe_engine_close
#undef xe_session_new
#undef profile_open

static int failures;
#define CHECK(c) do { if (!(c)) { failures++; fprintf(stderr, "FAIL %d: %s\n", __LINE__, #c); } } while (0)

static void cli_case(const char *command, const char *prompt, const char *top_k,
                     const char *max_tokens, int expected, const char *message) {
    char *args[] = { "xenolith", (char *)command, "fixture.gguf", "--ctx", "64",
                     "--top-k", (char *)top_k, "-n", (char *)max_tokens, "-p", (char *)prompt, NULL };
    int argc = !strcmp(command, "run") ? 11 : 9;
    int fds[2];
    assert(pipe(fds) == 0);
    fflush(NULL);
    pid_t child = fork();
    assert(child >= 0);
    if (!child) {
        close(fds[0]);
        assert(dup2(fds[1], STDERR_FILENO) >= 0);
        close(fds[1]);
        _exit(cli_main(argc, args));
    }
    close(fds[1]);
    char output[4096];
    size_t used = 0;
    ssize_t n;
    while ((n = read(fds[0], output + used, sizeof output - 1 - used)) > 0)
        used += (size_t)n;
    output[used] = 0;
    close(fds[0]);
    int status;
    assert(waitpid(child, &status, 0) == child);
    CHECK(WIFEXITED(status) && WEXITSTATUS(status) == expected);
    CHECK(!message || strstr(output, message));
    if ((message && !strstr(output, message)) || !WIFEXITED(status) ||
        WEXITSTATUS(status) != expected) fprintf(stderr, "%s: %s\n", command, output);
}

static void tokenizer_and_cli(void) {
    char long_text[81], boundary[65];
    memset(long_text, '\n', 80); long_text[80] = 0;
    for (int i = 0; i < 64; i++) boundary[i] = i % 2 ? 'A' : '\n';
    boundary[64] = 0;
    xe_str pieces[] = {{ .p = "\n", .len = 1 }, { .p = "A", .len = 1 },
                       { .p = long_text, .len = 80 }};
    fixture_engine = (xe_engine){ .context = 64, .tok_tokens_count = XE_VOCAB,
                                  .tok_piece = pieces, .bos_id = 1,
                                  .pool_initialized = 1, .owner = pthread_self() };
    fixture_engine.tok_hash = malloc(XE_TOK_HASH * sizeof(int32_t));
    assert(fixture_engine.tok_hash);
    for (int i = 0; i < XE_TOK_HASH; i++) fixture_engine.tok_hash[i] = -1;
    for (int id = 0; id < 3; id++) {
        uint32_t slot = xe_fnv1a(pieces[id].p, pieces[id].len) & (XE_TOK_HASH - 1);
        while (fixture_engine.tok_hash[slot] >= 0) slot = (slot + 1) & (XE_TOK_HASH - 1);
        fixture_engine.tok_hash[slot] = id;
    }
    int32_t out[64];
    for (int i = 0; i < 64; i++) out[i] = -123;
    CHECK(xe_vocab_size(&fixture_engine) == XE_VOCAB);
    CHECK(xe_encode_text_bounded(&fixture_engine, long_text, out, 1) == 1);
    CHECK(out[0] == 2 && out[1] == -123);
    for (int i = 0; i < 64; i++) out[i] = -123;
    CHECK(xe_encode_text_bounded(&fixture_engine, boundary, out, 63) == 64);
    for (int i = 0; i < 64; i++) CHECK(out[i] == -123);
    CHECK(xe_encode_text_bounded(&fixture_engine, boundary, out, 64) == 64);
    for (int i = 0; i < 64; i++) CHECK(out[i] == i % 2);
    CHECK(xe_encode_text(&fixture_engine, long_text, out, 1) == 1);
    CHECK(xe_encode_text_bounded(&fixture_engine, "", out, 0) == 0);
    for (int c = 0; c < 2; c++) {
        const char *command = c ? "chat" : "run";
        cli_case(command, long_text, "128", "0", 0, NULL);
        cli_case(command, long_text, "262144", "0", 0, NULL);
        cli_case(command, long_text, "0", "0", 0, NULL);
        cli_case(command, long_text, "262145", "0", 1, "top-k 262145 out of range [0, 262144]");
    }
    cli_case("run", boundary, "128", "0", 1, "prompt has 65 tokens including BOS; context capacity is 64");
    boundary[63] = 0;
    cli_case("run", boundary, "128", "0", 0, NULL); /* BOS + 63, no generation */
    cli_case("run", boundary, "128", "1", 1, "prompt leaves no room for generation");
    free(fixture_engine.tok_hash);
    fixture_engine.tok_hash = NULL;
}

static conversation_id record(conversation_store *store, int count) {
    conversation *c = NULL;
    conversation_id id;
    int32_t tokens[65];
    for (int i = 0; i < count; i++) tokens[i] = 2;
    assert(conversation_create(store, &c, &id) == CONVERSATION_OK);
    assert(conversation_append_message(c, CONVERSATION_ROLE_USER, NULL, 0,
                                      "prompt", 6, tokens, count) == CONVERSATION_OK);
    assert(conversation_commit(c) == CONVERSATION_OK);
    conversation_close(c);
    return id;
}

static uint64_t record_crc(const char *root, const conversation_id *id) {
    char hex[33], path[1024];
    serve_id_hex(id, hex);
    snprintf(path, sizeof path, "%s/sessions/%s/record.xcr", root, hex);
    FILE *f = fopen(path, "rb");
    assert(f);
    uint64_t crc = 0;
    unsigned char buf[4096];
    size_t n;
    while ((n = fread(buf, 1, sizeof buf, f))) crc = format_crc64(crc, buf, n);
    assert(!ferror(f));
    fclose(f);
    return crc;
}

static json_value *open_request(serve *server, serve_conn *conn, const conversation_id *id) {
    char hex[33], text[128];
    serve_id_hex(id, hex);
    snprintf(text, sizeof text, "{\"op\":\"open\",\"session\":\"%s\"}", hex);
    json_value *request = json_parse(text, strlen(text));
    assert(request);
    serve_execute(server, conn, request);
    json_free(request);
    json_value *reply = json_parse(server->out.data, server->out.length);
    assert(reply);
    return reply;
}

static void resume_and_protocol(const char *root) {
    conversation_store *store = NULL;
    assert(conversation_store_open(&store, root) == CONVERSATION_OK);
    conversation_id small = record(store, 64), large = record(store, 65);
    uint64_t before = record_crc(root, &large);
    runtime w = { .context = 64, .cstore = store };
    serve server = { .w = &w };
    serve_conn *conn = &server.conns[0]; /* inactive; inspect real JSON output */
    json_value *reply = open_request(&server, conn, &small);
    CHECK(json_member(reply, "ok")->boolean);
    json_free(reply);
    CHECK(w.has_current && conn->has_session);
    reply = open_request(&server, conn, &large);
    CHECK(!json_member(reply, "ok")->boolean);
    CHECK(!strcmp(json_member(reply, "code")->text, "context_length_exceeded"));
    CHECK(json_member(reply, "tokens")->number == 65);
    CHECK(json_member(reply, "context")->number == 64);
    json_free(reply);
    CHECK(!memcmp(w.current_id.bytes, small.bytes, 16));
    CHECK(!memcmp(conn->session.bytes, small.bytes, 16));
    CHECK(runtime_history_count(&w) == 1);
    CHECK(record_crc(root, &large) == before);
    /* A failed first open must not bind the client either. */
    serve_conn *unbound = &server.conns[1];
    reply = open_request(&server, unbound, &large);
    CHECK(!json_member(reply, "ok")->boolean && !unbound->has_session);
    json_free(reply);
    CHECK(runtime_budget_check(&w, UINT64_MAX) == RUNTIME_CONTEXT_LENGTH_EXCEEDED);
    CHECK(w.error_tokens == UINT64_MAX && w.error_context == 64);
    conversation *oversized = NULL;
    assert(conversation_open(store, &large, &oversized) == CONVERSATION_OK);
    int32_t tokens[64];
    for (int i = 0; i < 64; i++) tokens[i] = 2;
    xe_engine engine = { .context = 64, .pool_initialized = 1, .owner = pthread_self() };
    xe_session session = { .engine = &engine, .tokens = tokens, .n_tokens = 64 };
    CHECK(conversation_resume(oversized, NULL, &session, NULL) == CONVERSATION_LIMIT);
    CHECK(session.n_tokens == 64);
    CHECK(conversation_resume(w.current, NULL, &session, NULL) == CONVERSATION_OK);
    CHECK(session.n_tokens == 64);
    conversation *previous = w.current;
    w.current = oversized;
    w.kv = (kvstore *)&w; /* Rejection must precede any cache access. */
    w.session = &session;
    runtime_checkpoint_report checkpoint;
    runtime_checkpoint_now(&w, &checkpoint);
    CHECK(!checkpoint.saved && checkpoint.reason == RUNTIME_CKPT_REJECTED);
    CHECK(checkpoint.tokens == 65 && session.n_tokens == 64);
    w.current = previous;
    w.kv = NULL;
    w.session = NULL;
    conversation_close(oversized);
    runtime_park(&w, NULL);
    /* Reopening with sufficient capacity works, and does not alter the record. */
    w.context = 65;
    reply = open_request(&server, conn, &large);
    CHECK(json_member(reply, "ok")->boolean);
    CHECK(json_member(reply, "tokens")->number == 65);
    json_free(reply);
    runtime_park(&w, NULL);
    CHECK(record_crc(root, &large) == before);
    json_writer_free(&server.out);
    conversation_delete(store, &small);
    conversation_delete(store, &large);
    conversation_store_close(store);
}

int main(void) {
    char root[] = "/tmp/xenolith-context-limits-XXXXXX";
    assert(mkdtemp(root));
    assert(setenv("XDG_RUNTIME_DIR", root, 1) == 0);
    assert(setenv("XDG_STATE_HOME", root, 1) == 0);
    tokenizer_and_cli();
    char state[1024];
    snprintf(state, sizeof state, "%s/state", root);
    resume_and_protocol(state);
    char path[1100];
    const char *dirs[] = {"state/sessions", "state/trash", "state", "xenolith"};
    snprintf(path, sizeof path, "%s/xenolith/engine.lock", root); unlink(path);
    for (size_t i = 0; i < sizeof dirs / sizeof *dirs; i++) {
        snprintf(path, sizeof path, "%s/%s", root, dirs[i]); rmdir(path);
    }
    rmdir(root);
    printf("context limits: %s\n", failures ? "FAIL" : "PASS");
    return failures ? 1 : 0;
}
