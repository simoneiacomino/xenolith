#define _GNU_SOURCE
#define _POSIX_C_SOURCE 200809L

#include "serve.h"
#include "runtime.h"
#include "json.h"

#include <errno.h>
#include <fcntl.h>
#include <limits.h>
#include <poll.h>
#include <signal.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/file.h>
#include <sys/socket.h>
#include <sys/stat.h>
#include <sys/un.h>
#include <time.h>
#include <unistd.h>

enum {
    SERVE_MAX_CONNECTIONS = 64,
    /* Frame cap per request line and per response, derived from the
     * advertised context window: the worst transcript measured is ~4.2
     * JSON bytes per token (Italian prose), 16 leaves headroom for
     * multibyte text, double-escaped tool results and call arguments.
     * Single-line frames are a deliberate limit of protocol 1; see
     * NOTES "3.7 finding 3" for the bound and when chunking replaces it. */
    SERVE_FRAME_BYTES_PER_TOKEN = 16,
    SERVE_READ_CHUNK = 4096,
    SERVE_BACKLOG = 64,
    SERVE_ACCEPT_RETRY_MS = 1000,
    SERVE_PROTOCOL = 1
};

enum {
    SERVE_ROUTE_DONE = 0,
    SERVE_ROUTE_INLINE = 1,
    SERVE_ROUTE_QUEUE = 2
};

typedef struct {
    int active;
    int in_fd;
    int out_fd;
    int is_socket;
    int sequential;
    int eof;
    int queued;
    int head_checked;
    char *in;
    size_t in_length;
    size_t in_capacity;
    char *out;
    size_t out_length;
    size_t out_sent;
    size_t out_capacity;
    int has_session;
    conversation_id session;
    json_value *request;
} serve_conn;

typedef struct {
    runtime *w;
    int listen_fd;
    const char *socket_path;
    serve_conn conns[SERVE_MAX_CONNECTIONS];
    int queue[SERVE_MAX_CONNECTIONS];
    int queue_head;
    int queue_count;
    int generating;
    int gen_owner;
    int stop;
    int accept_paused;
    int64_t idle_ms;
    size_t max_frame;        /* context_window * SERVE_FRAME_BYTES_PER_TOKEN */
    int64_t active_at;
    json_writer out;
} serve;

static int serve_signal_fd[2] = { -1, -1 };

static void serve_on_signal(int sig) {
    int saved = errno;
    char byte = (char)sig;
    ssize_t n = write(serve_signal_fd[1], &byte, 1);
    (void)n;
    errno = saved;
}

static int64_t serve_now_ms(void) {
    struct timespec ts;
    if (clock_gettime(CLOCK_MONOTONIC, &ts) != 0) return 0;
    return (int64_t)ts.tv_sec * 1000 + ts.tv_nsec / 1000000;
}

static int serve_mkdir_parents(const char *path) {
    char copy[4096];
    int n = snprintf(copy, sizeof copy, "%s", path);
    if (n < 0 || (size_t)n >= sizeof copy) return 0;
    size_t length = strlen(copy);
    while (length > 1 && copy[length - 1] == '/') copy[--length] = '\0';
    for (char *p = copy + (copy[0] == '/'); *p; p++) {
        if (*p != '/') continue;
        *p = '\0';
        if (*copy && mkdir(copy, 0755) != 0 && errno != EEXIST) return 0;
        *p = '/';
    }
    return mkdir(copy, 0700) == 0 || errno == EEXIST;
}

int serve_socket_path(const char *override, const char *state_dir, char *out,
                      size_t cap) {
    int n;
    if (override) {
        n = snprintf(out, cap, "%s", override);
        return n > 0 && (size_t)n < cap;
    }
    const char *runtime = getenv("XDG_RUNTIME_DIR");
    if (runtime && *runtime == '/') {
        char dir[4096];
        n = snprintf(dir, sizeof dir, "%s/xenolith", runtime);
        if (n > 0 && (size_t)n < sizeof dir &&
            (mkdir(dir, 0700) == 0 || errno == EEXIST)) {
            n = snprintf(out, cap, "%s/wire.sock", dir);
            if (n > 0 && (size_t)n < cap) return 1;
        }
    }
    if (!state_dir) return 0;
    n = snprintf(out, cap, "%s/wire.sock", state_dir);
    return n > 0 && (size_t)n < cap;
}

static int serve_lock_dir(char *out, size_t cap) {
    if (!conversation_default_state_dir(out, cap)) return 0;
    return serve_mkdir_parents(out);
}

int serve_lock_path(char *out, size_t cap) {
    char dir[4096];
    if (!serve_lock_dir(dir, sizeof dir)) return 0;
    int n = snprintf(out, cap, "%s/engine.lock", dir);
    return n > 0 && (size_t)n < cap;
}

static int serve_lock_inherited(const char *path) {
    const char *text = getenv("XENOLITH_ENGINE_LOCK_FD");
    if (!text || !*text) return -1;
    char *end;
    errno = 0;
    long value = strtol(text, &end, 10);
    if (errno || *end || value < 0 || value > INT_MAX) return -1;
    struct stat inherited_stat;
    struct stat path_stat;
    if (fstat((int)value, &inherited_stat) != 0 ||
        stat(path, &path_stat) != 0 ||
        inherited_stat.st_dev != path_stat.st_dev ||
        inherited_stat.st_ino != path_stat.st_ino) return -1;
    int fd = fcntl((int)value, F_DUPFD_CLOEXEC, 3);
    if (fd < 0) return -1;
    if (flock(fd, LOCK_EX | LOCK_NB) != 0) {
        close(fd);
        return -1;
    }
    unsetenv("XENOLITH_ENGINE_LOCK_FD");
    return fd;
}

int serve_lock(const char *socket_path, char *holder, size_t holder_cap) {
    if (holder && holder_cap) holder[0] = '\0';
    char path[4096];
    if (!serve_lock_path(path, sizeof path)) return -1;
    int fd = serve_lock_inherited(path);
    if (fd < 0) fd = open(path, O_RDWR | O_CREAT | O_CLOEXEC, 0600);
    if (fd < 0) return -1;
    if (flock(fd, LOCK_EX | LOCK_NB) != 0) {
        if (holder && holder_cap > 1) {
            ssize_t got = read(fd, holder, holder_cap - 1);
            holder[got > 0 ? (size_t)got : 0] = '\0';
            for (char *p = holder; *p; p++)
                if (*p == '\n' || *p == '\r') *p = ' ';
            size_t length = strlen(holder);
            while (length && holder[length - 1] == ' ')
                holder[--length] = '\0';
        }
        close(fd);
        return -1;
    }
    char text[4600];
    int n = snprintf(text, sizeof text, "pid %ld socket %s\n", (long)getpid(),
                     socket_path ? socket_path : "");
    if (n > 0 && (size_t)n < sizeof text &&
        pwrite(fd, text, (size_t)n, 0) == n) {
        int trimmed = ftruncate(fd, n);
        (void)trimmed;
    }
    return fd;
}

static const char *serve_hex_digits = "0123456789abcdef";

static void serve_id_hex(const conversation_id *id, char out[33]) {
    for (int i = 0; i < 16; i++) {
        out[2 * i] = serve_hex_digits[id->bytes[i] >> 4];
        out[2 * i + 1] = serve_hex_digits[id->bytes[i] & 15];
    }
    out[32] = '\0';
}

static int serve_id_parse(const char *text, conversation_id *id) {
    if (!text || strlen(text) != 32) return 0;
    for (int i = 0; i < 32; i++) {
        const char *p = strchr(serve_hex_digits, text[i]);
        if (!p || !text[i]) return 0;
        int v = (int)(p - serve_hex_digits);
        if (i % 2 == 0) id->bytes[i / 2] = (uint8_t)(v << 4);
        else id->bytes[i / 2] |= (uint8_t)v;
    }
    return 1;
}

static const char *serve_str(const json_value *object, const char *key) {
    const json_value *v = json_member(object, key);
    return v && v->type == JSON_STRING ? v->text : NULL;
}

static int serve_u64(const json_value *object, const char *key,
                     uint64_t *out) {
    const json_value *v = json_member(object, key);
    if (!v || v->type != JSON_NUMBER || v->number < 0) return 0;
    *out = (uint64_t)v->number;
    return 1;
}

static int serve_conn_index(const serve *s, const serve_conn *c) {
    return (int)(c - s->conns);
}

static void serve_dequeue(serve *s, int index) {
    int kept[SERVE_MAX_CONNECTIONS];
    int count = 0;
    for (int i = 0; i < s->queue_count; i++) {
        int slot = s->queue[(s->queue_head + i) % SERVE_MAX_CONNECTIONS];
        if (slot != index) kept[count++] = slot;
    }
    for (int i = 0; i < count; i++) s->queue[i] = kept[i];
    s->queue_head = 0;
    s->queue_count = count;
}

static void serve_drop(serve *s, serve_conn *c) {
    if (!c->active) return;
    int index = serve_conn_index(s, c);
    if (s->generating && s->gen_owner == index) {
        runtime_cancel(s->w);
        s->gen_owner = -1;
    }
    if (c->queued) serve_dequeue(s, index);
    c->active = 0;
    c->queued = 0;
    c->head_checked = 0;
    c->has_session = 0;
    json_free(c->request);
    c->request = NULL;
    free(c->in);
    c->in = NULL;
    c->in_length = 0;
    c->in_capacity = 0;
    free(c->out);
    c->out = NULL;
    c->out_length = 0;
    c->out_sent = 0;
    c->out_capacity = 0;
    if (c->is_socket && c->in_fd >= 0) close(c->in_fd);
    c->in_fd = -1;
    c->out_fd = -1;
    s->accept_paused = 0;
}

static void serve_flush(serve *s, serve_conn *c) {
    while (c->active && c->out_sent < c->out_length) {
        ssize_t n = write(c->out_fd, c->out + c->out_sent,
                          c->out_length - c->out_sent);
        if (n > 0) {
            c->out_sent += (size_t)n;
            continue;
        }
        if (n < 0 && errno == EINTR) continue;
        if (n < 0 && (errno == EAGAIN || errno == EWOULDBLOCK)) break;
        serve_drop(s, c);
        return;
    }
    if (c->out_sent >= c->out_length) {
        c->out_sent = 0;
        c->out_length = 0;
    } else if (c->out_sent) {
        memmove(c->out, c->out + c->out_sent, c->out_length - c->out_sent);
        c->out_length -= c->out_sent;
        c->out_sent = 0;
    }
}

static void serve_push(serve *s, serve_conn *c, const void *data,
                       size_t length) {
    if (!c->active) return;
    if (c->out_length + length > s->max_frame) {
        serve_drop(s, c);
        return;
    }
    if (c->out_length + length > c->out_capacity) {
        size_t capacity = c->out_capacity ? c->out_capacity : 4096;
        while (capacity < c->out_length + length) capacity *= 2;
        char *grown = realloc(c->out, capacity);
        if (!grown) {
            serve_drop(s, c);
            return;
        }
        c->out = grown;
        c->out_capacity = capacity;
    }
    memcpy(c->out + c->out_length, data, length);
    c->out_length += length;
}

static const char serve_oom_line[] =
    "{\"ok\":false,\"code\":\"out_of_memory\",\"error\":\"out of memory\"}\n";

static void serve_emit(serve *s, serve_conn *c) {
    if (s->out.failed) {
        serve_push(s, c, serve_oom_line, sizeof serve_oom_line - 1);
        serve_flush(s, c);
        return;
    }
    serve_push(s, c, s->out.data ? s->out.data : "", s->out.length);
    serve_push(s, c, "\n", 1);
    serve_flush(s, c);
}

static void serve_ok_begin(serve *s) {
    json_writer_reset(&s->out);
    json_raw(&s->out, "{\"ok\":true");
}

static void serve_error_detail(json_writer *out, runtime_status status,
                               uint64_t tokens, uint64_t context) {
    if (status != RUNTIME_CONTEXT_LENGTH_EXCEEDED || !context) return;
    json_raw(out, ",\"tokens\":");
    json_u64(out, tokens);
    json_raw(out, ",\"context\":");
    json_u64(out, context);
}

static void serve_error_with(serve *s, serve_conn *c, runtime_status status,
                             const char *text, uint64_t tokens,
                             uint64_t context) {
    json_writer *out = &s->out;
    json_writer_reset(out);
    json_raw(out, "{\"ok\":false,\"code\":");
    json_string(out, runtime_status_code(status),
                strlen(runtime_status_code(status)));
    json_raw(out, ",\"error\":");
    json_string(out, text, strlen(text));
    serve_error_detail(out, status, tokens, context);
    json_raw(out, "}");
    serve_emit(s, c);
}

static void serve_error(serve *s, serve_conn *c, runtime_status status,
                        const char *text) {
    serve_error_with(s, c, status, text, 0, 0);
}

static void serve_runtime_error(serve *s, serve_conn *c, runtime_status status) {
    const char *text = runtime_error_text(s->w);
    uint64_t tokens = 0, context = 0;
    runtime_error_detail(s->w, &tokens, &context);
    serve_error_with(s, c, status, *text ? text : runtime_status_code(status),
                     tokens, context);
}

static const char *serve_stop_name(uint32_t stop) {
    switch (stop) {
    case RUNTIME_STOP_STOP: return "stop";
    case RUNTIME_STOP_TOOL_USE: return "tool_use";
    case RUNTIME_STOP_LENGTH: return "length";
    case RUNTIME_STOP_ABORTED: return "aborted";
    }
    return "unknown";
}

static const char *serve_record_stop_name(uint32_t stop) {
    switch (stop) {
    case CONVERSATION_STOP_EOT_SAMPLED: return "eot";
    case CONVERSATION_STOP_EOT_SYNTHETIC: return "eot_synthetic";
    case CONVERSATION_STOP_EOS: return "eos";
    case CONVERSATION_STOP_LIMIT: return "limit";
    case CONVERSATION_STOP_CANCELLED: return "cancelled";
    case CONVERSATION_STOP_TOOL_CALLS: return "tool_calls";
    }
    return "unknown";
}

static const char *serve_reasoning_close_name(uint32_t close) {
    switch (close) {
    case CONVERSATION_REASONING_NATURAL: return "natural";
    case CONVERSATION_REASONING_SOFT: return "soft";
    case CONVERSATION_REASONING_HARD: return "hard";
    case CONVERSATION_REASONING_LENGTH: return "length";
    case CONVERSATION_REASONING_ABORTED: return "aborted";
    case CONVERSATION_REASONING_EOS: return "eos";
    }
    return "none";
}

typedef struct {
    profile_tool *tools;
    size_t count;
    char **owned;
    size_t owned_count;
} serve_tools;

typedef struct {
    runtime_message *items;
    size_t count;
    profile_call *calls;
    char **owned;
    size_t owned_count;
} serve_messages;

static void serve_tools_free(serve_tools *tools) {
    for (size_t i = 0; i < tools->owned_count; i++) free(tools->owned[i]);
    free(tools->owned);
    free(tools->tools);
    memset(tools, 0, sizeof *tools);
}

static void serve_messages_free(serve_messages *messages) {
    for (size_t i = 0; i < messages->owned_count; i++)
        free(messages->owned[i]);
    free(messages->owned);
    free(messages->calls);
    free(messages->items);
    memset(messages, 0, sizeof *messages);
}

static char *serve_serialize(const json_value *value) {
    json_writer w = {0};
    if (!json_write_value(&w, value)) {
        json_writer_free(&w);
        return NULL;
    }
    return w.data;
}

static int serve_own(char ***owned, size_t *count, char *text) {
    char **grown = realloc(*owned, (*count + 1) * sizeof(*grown));
    if (!grown) {
        free(text);
        return 0;
    }
    *owned = grown;
    grown[(*count)++] = text;
    return 1;
}

static int serve_tools_parse(const json_value *request, serve_tools *out) {
    memset(out, 0, sizeof *out);
    const json_value *array = json_member(request, "tools");
    if (!array) return 1;
    if (array->type != JSON_ARRAY) return 0;
    size_t n = json_length(array);
    if (!n) return 1;
    out->tools = calloc(n, sizeof *out->tools);
    if (!out->tools) return 0;
    size_t i = 0;
    for (const json_value *v = array->child; v; v = v->next, i++) {
        const char *name = serve_str(v, "name");
        const char *description = serve_str(v, "description");
        if (!name || !description) return 0;
        out->tools[i].name = name;
        out->tools[i].description = description;
        const json_value *parameters = json_member(v, "parameters");
        if (parameters && parameters->type == JSON_OBJECT) {
            char *text = serve_serialize(parameters);
            if (!text || !serve_own(&out->owned, &out->owned_count, text))
                return 0;
            out->tools[i].parameters_json = text;
        }
    }
    out->count = n;
    return 1;
}

static int serve_messages_parse(const json_value *request,
                                serve_messages *out) {
    memset(out, 0, sizeof *out);
    const json_value *array = json_member(request, "messages");
    if (!array) return 1;
    if (array->type != JSON_ARRAY) return 0;
    size_t n = json_length(array);
    if (!n) return 1;
    out->items = calloc(n, sizeof *out->items);
    if (!out->items) return 0;
    size_t total_calls = 0;
    for (const json_value *v = array->child; v; v = v->next)
        total_calls += json_length(json_member(v, "calls"));
    if (total_calls) {
        out->calls = calloc(total_calls, sizeof *out->calls);
        if (!out->calls) return 0;
    }
    size_t i = 0, call_at = 0;
    for (const json_value *v = array->child; v; v = v->next, i++) {
        runtime_message *m = &out->items[i];
        const char *role = serve_str(v, "role");
        if (!role) return 0;
        m->text = serve_str(v, "text");
        if (!strcmp(role, "user")) {
            m->kind = RUNTIME_MESSAGE_USER;
        } else if (!strcmp(role, "assistant")) {
            m->kind = RUNTIME_MESSAGE_ASSISTANT;
            m->reasoning = serve_str(v, "reasoning");
            const json_value *calls = json_member(v, "calls");
            if (calls && calls->type == JSON_ARRAY && calls->child) {
                m->calls = &out->calls[call_at];
                for (const json_value *cv = calls->child; cv;
                     cv = cv->next) {
                    const char *name = serve_str(cv, "name");
                    if (!name) return 0;
                    out->calls[call_at].name = name;
                    const json_value *arguments = json_member(cv,
                                                              "arguments");
                    if (arguments) {
                        char *text = serve_serialize(arguments);
                        if (!text || !serve_own(&out->owned,
                                                &out->owned_count, text))
                            return 0;
                        out->calls[call_at].arguments_json = text;
                    }
                    call_at++;
                    m->call_count++;
                }
            }
        } else if (!strcmp(role, "tool")) {
            m->kind = RUNTIME_MESSAGE_TOOL_RESULT;
            m->tool_name = serve_str(v, "tool");
            serve_u64(v, "call_id", &m->call_id);
            const char *status = serve_str(v, "status");
            m->tool_status = status && !strcmp(status, "error")
                             ? CONVERSATION_TOOL_ERROR
                             : CONVERSATION_TOOL_OK;
        } else {
            return 0;
        }
    }
    out->count = n;
    return 1;
}

static int serve_params_parse(const json_value *request,
                              runtime_gen_params *params) {
    memset(params, 0, sizeof *params);
    params->temperature = -1.0f;
    params->top_k = -1;
    params->top_p = -1.0f;
    params->max_tokens = -1;
    params->rng_seed = 0;
    const json_value *v;
    if ((v = json_member(request, "temperature")) != NULL &&
        v->type == JSON_NUMBER && v->number >= 0)
        params->temperature = (float)v->number;
    if ((v = json_member(request, "top_k")) != NULL &&
        v->type == JSON_NUMBER && v->number >= 0)
        params->top_k = (int32_t)v->number;
    if ((v = json_member(request, "top_p")) != NULL &&
        v->type == JSON_NUMBER && v->number > 0)
        params->top_p = (float)v->number;
    if ((v = json_member(request, "max_tokens")) != NULL &&
        v->type == JSON_NUMBER && v->number >= 0)
        params->max_tokens = (int32_t)v->number;
    uint64_t seed;
    if (serve_u64(request, "seed", &seed)) params->rng_seed = seed;
    v = json_member(request, "reasoning");
    if (!v) return 1;
    params->reasoning_set = 1;
    if (v->type == JSON_BOOL) {
        if (v->boolean) return 0;
        params->reasoning_effort = CONVERSATION_REASONING_OFF;
        return 1;
    }
    if (v->type != JSON_OBJECT) return 0;
    const char *effort = serve_str(v, "effort");
    if (!effort) return 0;
    if (!strcmp(effort, "low"))
        params->reasoning_effort = CONVERSATION_REASONING_LOW;
    else if (!strcmp(effort, "medium"))
        params->reasoning_effort = CONVERSATION_REASONING_MEDIUM;
    else if (!strcmp(effort, "high"))
        params->reasoning_effort = CONVERSATION_REASONING_HIGH;
    else if (!strcmp(effort, "max"))
        params->reasoning_effort = CONVERSATION_REASONING_MAX;
    else
        return 0;
    const json_value *budget = json_member(v, "budget_tokens");
    if (budget) {
        if (budget->type != JSON_NUMBER || budget->number < 0 ||
            budget->number > INT32_MAX ||
            (double)(int32_t)budget->number != budget->number)
            return 0;
        params->reasoning_budget_set = 1;
        params->reasoning_budget = (int32_t)budget->number;
    }
    const char *history = serve_str(v, "history");
    if (history) {
        params->reasoning_history_set = 1;
        if (!strcmp(history, "discard"))
            params->reasoning_history = CONVERSATION_REASONING_DISCARD;
        else if (!strcmp(history, "preserve_tool_calls"))
            params->reasoning_history =
                CONVERSATION_REASONING_PRESERVE_TOOL_CALLS;
        else
            return 0;
    }
    return 1;
}

static int serve_ckpt_noteworthy(const runtime_checkpoint_report *r) {
    return !r->saved && r->reason != RUNTIME_CKPT_NOTHING_NEW &&
           r->reason != RUNTIME_CKPT_EMPTY;
}

static void serve_log_checkpoint(const char *where,
                                 const runtime_checkpoint_report *r) {
    if (!serve_ckpt_noteworthy(r)) return;
    fprintf(stderr, "xenolith: checkpoint (%s): not saved, %s, %llu tokens\n",
            where, runtime_ckpt_reason_name(r->reason),
            (unsigned long long)r->tokens);
}

static void serve_log_resume(const runtime_resume_report *r) {
    if (r->loaded) return;
    fprintf(stderr, "xenolith: resume: snapshot not loaded, %s, %llu tokens\n",
            runtime_resume_reason_name(r->reason), (unsigned long long)r->tokens);
}

static void serve_log_kvstore(const runtime *w, const char *cache_dir) {
    int status = runtime_kvstore_open_status(w);
    if (status == KVSTORE_OK) return;
    fprintf(stderr, "xenolith: kvstore disabled: open failed (%s) at %s; "
            "sessions will not resume from cache\n",
            kvstore_status_name((kvstore_status)status),
            cache_dir ? cache_dir : "default cache dir");
}

static void serve_inference_measure(json_writer *out,
                                    const runtime_inference_measure *measure,
                                    int prefill) {
    json_raw(out, "{\"tokens\":");
    json_u64(out, measure->tokens);
    if (prefill) {
        json_raw(out, ",\"total\":");
        json_u64(out, measure->total);
    }
    json_raw(out, ",\"elapsed_ms\":");
    json_u64(out, measure->elapsed_ms);
    json_raw(out, "}");
}

static void serve_inference(json_writer *out,
                            const runtime_inference *inference) {
    json_raw(out, ",\"inference\":{\"prefill\":");
    serve_inference_measure(out, &inference->prefill, 1);
    json_raw(out, ",\"decode\":");
    serve_inference_measure(out, &inference->decode, 0);
    json_raw(out, "}");
}

static void serve_emit_event(serve *s, serve_conn *c,
                             const runtime_event *event) {
    json_writer *out = &s->out;
    json_writer_reset(out);
    switch (event->kind) {
    case RUNTIME_EVENT_START:
        json_raw(out, "{\"event\":\"start\"}");
        break;
    case RUNTIME_EVENT_INFERENCE_PROGRESS:
        json_raw(out, "{\"event\":\"inference_progress\",\"phase\":\"");
        json_raw(out, event->phase == RUNTIME_INFERENCE_PREFILL
                      ? "prefill" : "decode");
        json_raw(out, "\",\"state\":\"");
        json_raw(out, event->state == RUNTIME_INFERENCE_FINISHED
                      ? "finished" : "running");
        json_raw(out, "\",\"tokens\":");
        json_u64(out, event->progress.tokens);
        if (event->phase == RUNTIME_INFERENCE_PREFILL) {
            json_raw(out, ",\"total\":");
            json_u64(out, event->progress.total);
        }
        json_raw(out, ",\"elapsed_ms\":");
        json_u64(out, event->progress.elapsed_ms);
        json_raw(out, "}");
        break;
    case RUNTIME_EVENT_TEXT_DELTA:
        json_raw(out, "{\"event\":\"text_delta\",\"text\":");
        json_string(out, (const char *)event->text, event->text_length);
        json_raw(out, "}");
        break;
    case RUNTIME_EVENT_REASONING_DELTA:
        json_raw(out, "{\"event\":\"reasoning_delta\",\"text\":");
        json_string(out, (const char *)event->text, event->text_length);
        json_raw(out, "}");
        break;
    case RUNTIME_EVENT_TOOLCALL_START:
        json_raw(out, "{\"event\":\"toolcall_start\",\"id\":");
        json_u64(out, event->call_id);
        json_raw(out, ",\"name\":");
        json_string(out, event->call_name, strlen(event->call_name));
        json_raw(out, "}");
        break;
    case RUNTIME_EVENT_TOOLCALL_END:
        json_raw(out, "{\"event\":\"toolcall_end\",\"id\":");
        json_u64(out, event->call_id);
        json_raw(out, ",\"name\":");
        json_string(out, event->call_name, strlen(event->call_name));
        json_raw(out, ",\"arguments\":");
        json_raw(out, event->arguments_json);
        json_raw(out, "}");
        break;
    case RUNTIME_EVENT_DONE:
        json_raw(out, "{\"event\":\"done\",\"stop\":\"");
        json_raw(out, serve_stop_name(event->stop));
        json_raw(out, "\",\"usage\":{\"input\":");
        json_u64(out, event->usage.input);
        json_raw(out, ",\"cache_read\":");
        json_u64(out, event->usage.cache_read);
        json_raw(out, ",\"output\":");
        json_u64(out, event->usage.output);
        json_raw(out, ",\"reasoning\":");
        json_u64(out, event->usage.reasoning);
        json_raw(out, ",\"replayed\":");
        json_u64(out, event->usage.replayed);
        json_raw(out, ",\"shadow_prefilled\":");
        json_u64(out, event->usage.shadow_prefilled);
        json_raw(out, ",\"shadow_background\":");
        json_u64(out, event->usage.shadow_background);
        json_raw(out, ",\"shadow_remaining\":");
        json_u64(out, event->usage.shadow_remaining);
        json_raw(out, ",\"shadow_kv_bytes\":");
        json_u64(out, event->usage.shadow_kv_bytes);
        json_raw(out, ",\"shadow_wait_us\":");
        json_u64(out, event->usage.shadow_wait_us);
        json_raw(out, ",\"total\":");
        json_u64(out, event->usage.total);
        json_raw(out, "}");
        serve_inference(out, &event->inference);
        json_raw(out, ",\"marker\":");
        if (event->marker == RUNTIME_MARKER_NONE) json_raw(out, "null");
        else json_u64(out, event->marker);
        if (event->reasoning_close != CONVERSATION_REASONING_NONE) {
            const char *close = serve_reasoning_close_name(
                event->reasoning_close);
            json_raw(out, ",\"reasoning_close\":");
            json_string(out, close, strlen(close));
        }
        if (event->checkpoint_attempted) {
            const char *reason = runtime_ckpt_reason_name(event->checkpoint.reason);
            serve_log_checkpoint("autosave", &event->checkpoint);
            json_raw(out, ",\"checkpoint\":{\"saved\":");
            json_raw(out, event->checkpoint.saved ? "true" : "false");
            if (!event->checkpoint.saved) {
                json_raw(out, ",\"reason\":");
                json_string(out, reason, strlen(reason));
            }
            json_raw(out, ",\"tokens\":");
            json_u64(out, event->checkpoint.tokens);
            json_raw(out, "}");
        }
        if (event->resume_attempted) {
            const char *reason = runtime_resume_reason_name(event->resume.reason);
            serve_log_resume(&event->resume);
            json_raw(out, ",\"resume\":{\"loaded\":");
            json_raw(out, event->resume.loaded ? "true" : "false");
            if (!event->resume.loaded) {
                json_raw(out, ",\"reason\":");
                json_string(out, reason, strlen(reason));
            }
            json_raw(out, ",\"tokens\":");
            json_u64(out, event->resume.tokens);
            json_raw(out, "}");
        }
        json_raw(out, "}");
        break;
    case RUNTIME_EVENT_ERROR:
        json_raw(out, "{\"event\":\"error\",\"code\":");
        json_string(out, runtime_status_code(event->error),
                    strlen(runtime_status_code(event->error)));
        json_raw(out, ",\"error\":");
        json_string(out, event->error_text ? event->error_text : "",
                    event->error_text ? strlen(event->error_text) : 0);
        serve_error_detail(out, (runtime_status)event->error,
                           event->error_tokens, event->error_context);
        serve_inference(out, &event->inference);
        json_raw(out, "}");
        break;
    }
    serve_emit(s, c);
}

static int serve_dispatch(serve *s, serve_conn *c,
                          const json_value *request) {
    json_writer *out = &s->out;
    const char *op = serve_str(request, "op");
    if (!op) {
        serve_error(s, c, RUNTIME_INVALID_ARGUMENT, "missing op");
        return 0;
    }
    runtime_status status;
    if (!strcmp(op, "describe")) {
        runtime_info info;
        status = runtime_describe(s->w, &info);
        if (status != RUNTIME_OK) {
            serve_runtime_error(s, c, status);
            return 0;
        }
        serve_ok_begin(s);
        json_raw(out, ",\"protocol\":");
        json_i64(out, SERVE_PROTOCOL);
        json_raw(out, ",\"model\":");
        json_string(out, info.model, strlen(info.model));
        json_raw(out, ",\"context_window\":");
        json_i64(out, info.context_window);
        json_raw(out, ",\"max_output\":");
        json_i64(out, info.max_output);
        json_raw(out, ",\"max_frame\":");
        json_u64(out, (uint64_t)s->max_frame);
        json_raw(out, ",\"kvstore\":");
        json_raw(out, info.kvstore ? "true" : "false");
        json_raw(out, ",\"reasoning\":{\"efforts\":[\"low\","
                      "\"medium\",\"high\",\"max\"],\"history\":["
                      "\"discard\",\"preserve_tool_calls\"],"
                      "\"budget_tokens\":true}");
        json_raw(out, "}");
        serve_emit(s, c);
        return 0;
    }
    if (!strcmp(op, "create")) {
        serve_tools tools;
        if (!serve_tools_parse(request, &tools)) {
            serve_tools_free(&tools);
            serve_error(s, c, RUNTIME_INVALID_ARGUMENT, "invalid tools");
            return 0;
        }
        conversation_id id;
        runtime_marker marker;
        status = runtime_session_create(s->w, serve_str(request, "system"),
                                     tools.tools, tools.count, &id,
                                     &marker);
        serve_tools_free(&tools);
        if (status != RUNTIME_OK) {
            serve_runtime_error(s, c, status);
            return 0;
        }
        c->session = id;
        c->has_session = 1;
        char hex[33];
        serve_id_hex(&id, hex);
        serve_ok_begin(s);
        json_raw(out, ",\"session\":\"");
        json_raw(out, hex);
        json_raw(out, "\",\"marker\":");
        json_u64(out, marker);
        json_raw(out, "}");
        serve_emit(s, c);
        return 0;
    }
    if (!strcmp(op, "open") || !strcmp(op, "stat") ||
        !strcmp(op, "delete")) {
        conversation_id id;
        if (!serve_id_parse(serve_str(request, "session"), &id)) {
            serve_error(s, c, RUNTIME_INVALID_ARGUMENT, "invalid session id");
            return 0;
        }
        if (!strcmp(op, "delete")) {
            int bound = c->has_session &&
                        memcmp(c->session.bytes, id.bytes, 16) == 0;
            status = runtime_session_delete(s->w, &id);
            if (status != RUNTIME_OK) serve_runtime_error(s, c, status);
            else {
                if (bound) c->has_session = 0;
                serve_ok_begin(s);
                json_raw(out, "}");
                serve_emit(s, c);
            }
            return 0;
        }
        if (!strcmp(op, "stat")) {
            conversation_summary summary;
            status = runtime_session_stat(s->w, &id, &summary);
            if (status != RUNTIME_OK) {
                serve_runtime_error(s, c, status);
                return 0;
            }
            char hex[33];
            serve_id_hex(&summary.id, hex);
            serve_ok_begin(s);
            json_raw(out, ",\"session\":\"");
            json_raw(out, hex);
            json_raw(out, "\",\"title\":");
            json_string(out, summary.title, strlen(summary.title));
            json_raw(out, ",\"tokens\":");
            json_u64(out, summary.token_count);
            json_raw(out, ",\"resumable\":");
            json_raw(out, summary.resumable ? "true" : "false");
            json_raw(out, ",\"created\":");
            json_i64(out, summary.created);
            json_raw(out, ",\"updated\":");
            json_i64(out, summary.updated);
            json_raw(out, "}");
            serve_emit(s, c);
            return 0;
        }
        runtime_open_report report;
        status = runtime_session_open(s->w, &id, &report);
        if (status != RUNTIME_OK) {
            serve_runtime_error(s, c, status);
            return 0;
        }
        c->session = id;
        c->has_session = 1;
        serve_ok_begin(s);
        json_raw(out, ",\"tokens\":");
        json_u64(out, report.token_count);
        json_raw(out, ",\"marker\":");
        if (report.marker == RUNTIME_MARKER_NONE) json_raw(out, "null");
        else json_u64(out, report.marker);
        json_raw(out, ",\"turn_open\":");
        json_raw(out, report.turn_open ? "true" : "false");
        json_raw(out, ",\"zero_prefill\":");
        json_raw(out, report.zero_prefill ? "true" : "false");
        json_raw(out, ",\"resume\":");
        json_raw(out, report.zero_prefill ? "\"snapshot\""
                      : report.resume_stale ? "\"stale\"" : "\"none\"");
        json_raw(out, ",\"pending\":[");
        uint64_t pending[64];
        size_t pending_count = runtime_pending_calls(
            s->w, pending, sizeof pending / sizeof pending[0]);
        if (pending_count > sizeof pending / sizeof pending[0])
            pending_count = sizeof pending / sizeof pending[0];
        for (size_t i = 0; i < pending_count; i++) {
            if (i) json_raw(out, ",");
            json_u64(out, pending[i]);
        }
        json_raw(out, "]}");
        serve_emit(s, c);
        return 0;
    }
    if (!strcmp(op, "list")) {
        conversation_summary *summaries = NULL;
        size_t count = 0;
        status = runtime_session_list(s->w, &summaries, &count);
        if (status != RUNTIME_OK) {
            serve_runtime_error(s, c, status);
            return 0;
        }
        serve_ok_begin(s);
        json_raw(out, ",\"sessions\":[");
        for (size_t i = 0; i < count; i++) {
            char hex[33];
            serve_id_hex(&summaries[i].id, hex);
            if (i) json_raw(out, ",");
            json_raw(out, "{\"session\":\"");
            json_raw(out, hex);
            json_raw(out, "\",\"title\":");
            json_string(out, summaries[i].title,
                        strlen(summaries[i].title));
            json_raw(out, ",\"tokens\":");
            json_u64(out, summaries[i].token_count);
            json_raw(out, ",\"resumable\":");
            json_raw(out, summaries[i].resumable ? "true" : "false");
            json_raw(out, "}");
        }
        json_raw(out, "]}");
        serve_emit(s, c);
        free(summaries);
        return 0;
    }
    if (!strcmp(op, "append")) {
        runtime_message message;
        memset(&message, 0, sizeof message);
        const char *role = serve_str(request, "role");
        message.text = serve_str(request, "text");
        if (role && !strcmp(role, "user")) {
            message.kind = RUNTIME_MESSAGE_USER;
        } else if (role && !strcmp(role, "tool")) {
            message.kind = RUNTIME_MESSAGE_TOOL_RESULT;
            if (!serve_u64(request, "call_id", &message.call_id)) {
                serve_error(s, c, RUNTIME_INVALID_ARGUMENT, "missing call_id");
                return 0;
            }
            const char *tool_status = serve_str(request, "status");
            message.tool_status = tool_status &&
                                  !strcmp(tool_status, "error")
                                  ? CONVERSATION_TOOL_ERROR
                                  : CONVERSATION_TOOL_OK;
        } else {
            serve_error(s, c, RUNTIME_INVALID_ARGUMENT, "invalid role");
            return 0;
        }
        runtime_marker marker;
        status = runtime_append(s->w, &message, &marker);
        if (status != RUNTIME_OK) {
            serve_runtime_error(s, c, status);
            return 0;
        }
        serve_ok_begin(s);
        json_raw(out, ",\"marker\":");
        json_u64(out, marker);
        json_raw(out, "}");
        serve_emit(s, c);
        return 0;
    }
    if (!strcmp(op, "generate")) {
        runtime_gen_params params;
        if (!serve_params_parse(request, &params)) {
            serve_error(s, c, RUNTIME_INVALID_ARGUMENT,
                        "invalid reasoning settings");
            return 0;
        }
        status = runtime_generate(s->w, &params);
        if (status != RUNTIME_OK) {
            serve_runtime_error(s, c, status);
            return 0;
        }
        return 1;
    }
    if (!strcmp(op, "ephemeral")) {
        serve_tools tools;
        serve_messages messages;
        if (!serve_tools_parse(request, &tools)) {
            serve_tools_free(&tools);
            serve_error(s, c, RUNTIME_INVALID_ARGUMENT, "invalid tools");
            return 0;
        }
        if (!serve_messages_parse(request, &messages)) {
            serve_tools_free(&tools);
            serve_messages_free(&messages);
            serve_error(s, c, RUNTIME_INVALID_ARGUMENT, "invalid messages");
            return 0;
        }
        runtime_gen_params params;
        if (!serve_params_parse(request, &params)) {
            serve_tools_free(&tools);
            serve_messages_free(&messages);
            serve_error(s, c, RUNTIME_INVALID_ARGUMENT,
                        "invalid reasoning settings");
            return 0;
        }
        status = runtime_ephemeral_generate(s->w, serve_str(request, "system"),
                                         tools.tools, tools.count,
                                         messages.items, messages.count,
                                         &params);
        serve_tools_free(&tools);
        serve_messages_free(&messages);
        if (status != RUNTIME_OK) {
            serve_runtime_error(s, c, status);
            return 0;
        }
        return 1;
    }
    if (!strcmp(op, "rebuild")) {
        serve_tools tools;
        serve_messages messages;
        if (!serve_tools_parse(request, &tools)) {
            serve_tools_free(&tools);
            serve_error(s, c, RUNTIME_INVALID_ARGUMENT, "invalid tools");
            return 0;
        }
        if (!serve_messages_parse(request, &messages)) {
            serve_tools_free(&tools);
            serve_messages_free(&messages);
            serve_error(s, c, RUNTIME_INVALID_ARGUMENT, "invalid messages");
            return 0;
        }
        runtime_marker marker;
        status = runtime_rebuild(s->w, serve_str(request, "system"),
                              tools.tools, tools.count, messages.items,
                              messages.count, &marker);
        serve_tools_free(&tools);
        serve_messages_free(&messages);
        if (status != RUNTIME_OK) {
            serve_runtime_error(s, c, status);
            return 0;
        }
        serve_ok_begin(s);
        json_raw(out, ",\"marker\":");
        json_u64(out, marker);
        json_raw(out, "}");
        serve_emit(s, c);
        return 0;
    }
    if (!strcmp(op, "rewind") || !strcmp(op, "rewind_cost")) {
        uint64_t marker;
        if (!serve_u64(request, "marker", &marker)) {
            serve_error(s, c, RUNTIME_INVALID_ARGUMENT, "missing marker");
            return 0;
        }
        if (!strcmp(op, "rewind")) {
            status = runtime_rewind(s->w, marker);
            if (status != RUNTIME_OK) serve_runtime_error(s, c, status);
            else {
                serve_ok_begin(s);
                json_raw(out, "}");
                serve_emit(s, c);
            }
            return 0;
        }
        uint64_t prefill;
        status = runtime_rewind_cost(s->w, marker, &prefill);
        if (status != RUNTIME_OK) {
            serve_runtime_error(s, c, status);
            return 0;
        }
        serve_ok_begin(s);
        json_raw(out, ",\"prefill\":");
        json_u64(out, prefill);
        json_raw(out, "}");
        serve_emit(s, c);
        return 0;
    }
    if (!strcmp(op, "checkpoint")) {
        runtime_checkpoint_report report;
        status = runtime_checkpoint(s->w, &report);
        if (status != RUNTIME_OK) serve_runtime_error(s, c, status);
        else {
            serve_log_checkpoint("checkpoint", &report);
            serve_ok_begin(s);
            json_raw(out, ",\"saved\":");
            json_raw(out, report.saved ? "true" : "false");
            if (!report.saved) {
                const char *reason = runtime_ckpt_reason_name(report.reason);
                json_raw(out, ",\"reason\":");
                json_string(out, reason, strlen(reason));
            }
            json_raw(out, ",\"tokens\":");
            json_u64(out, report.tokens);
            json_raw(out, "}");
            serve_emit(s, c);
        }
        return 0;
    }
    if (!strcmp(op, "pending")) {
        uint64_t pending[256];
        size_t count = runtime_pending_calls(
            s->w, pending, sizeof pending / sizeof pending[0]);
        if (count > sizeof pending / sizeof pending[0])
            count = sizeof pending / sizeof pending[0];
        serve_ok_begin(s);
        json_raw(out, ",\"calls\":[");
        for (size_t i = 0; i < count; i++) {
            if (i) json_raw(out, ",");
            json_u64(out, pending[i]);
        }
        json_raw(out, "]}");
        serve_emit(s, c);
        return 0;
    }
    if (!strcmp(op, "history")) {
        uint64_t count = runtime_history_count(s->w);
        serve_ok_begin(s);
        json_raw(out, ",\"entries\":[");
        for (uint64_t i = 0; i < count; i++) {
            runtime_history_entry entry;
            if (runtime_history_at(s->w, i, &entry) != RUNTIME_OK) break;
            if (i) json_raw(out, ",");
            json_raw(out, "{\"kind\":\"");
            json_raw(out, entry.kind == RUNTIME_MESSAGE_SYSTEM ? "system"
                          : entry.kind == RUNTIME_MESSAGE_USER ? "user"
                          : entry.kind == RUNTIME_MESSAGE_ASSISTANT
                          ? "assistant" : "tool_result");
            json_raw(out, "\",\"marker\":");
            json_u64(out, entry.marker);
            json_raw(out, ",\"text\":");
            json_string(out, entry.text ? entry.text : "",
                        entry.text_length);
            if (entry.reasoning) {
                json_raw(out, ",\"reasoning\":");
                json_string(out, entry.reasoning,
                            entry.reasoning_length);
            }
            if (entry.extra_json) {
                json_raw(out, ",\"extra\":");
                json_rawn(out, entry.extra_json, entry.extra_length);
            }
            if (entry.kind == RUNTIME_MESSAGE_TOOL_RESULT) {
                json_raw(out, ",\"call_id\":");
                json_u64(out, entry.call_id);
                json_raw(out, ",\"status\":\"");
                json_raw(out, entry.tool_status == CONVERSATION_TOOL_ERROR
                              ? "error" : "ok");
                json_raw(out, "\"");
                if (entry.tool_name) {
                    json_raw(out, ",\"tool\":");
                    json_string(out, entry.tool_name,
                                strlen(entry.tool_name));
                }
            }
            if (entry.stop_reason) {
                json_raw(out, ",\"stop\":\"");
                json_raw(out, serve_record_stop_name(entry.stop_reason));
                json_raw(out, "\"");
            }
            json_raw(out, "}");
        }
        json_raw(out, "]}");
        serve_emit(s, c);
        return 0;
    }
    if (!strcmp(op, "cancel")) {
        serve_error(s, c, RUNTIME_INVALID_ARGUMENT, "no generation in progress");
        return 0;
    }
    serve_error(s, c, RUNTIME_INVALID_ARGUMENT, "unknown op");
    return 0;
}

static void serve_cancel(serve *s, serve_conn *c) {
    int index = serve_conn_index(s, c);
    if (s->generating && s->gen_owner == index) {
        runtime_cancel(s->w);
        serve_ok_begin(s);
        json_raw(&s->out, "}");
        serve_emit(s, c);
        return;
    }
    if (c->queued && c->request) {
        const char *op = serve_str(c->request, "op");
        if (op && (!strcmp(op, "generate") || !strcmp(op, "ephemeral"))) {
            serve_dequeue(s, index);
            json_free(c->request);
            c->request = NULL;
            c->queued = 0;
            serve_ok_begin(s);
            json_raw(&s->out, "}");
            serve_emit(s, c);
            if (!c->active) return;
            runtime_event event;
            memset(&event, 0, sizeof event);
            event.kind = RUNTIME_EVENT_DONE;
            event.stop = RUNTIME_STOP_ABORTED;
            event.marker = RUNTIME_MARKER_NONE;
            serve_emit_event(s, c, &event);
            return;
        }
    }
    serve_error(s, c, RUNTIME_INVALID_ARGUMENT, "no generation in progress");
}

static int serve_binding_other(const serve *s, const conversation_id *id,
                               int except) {
    for (int i = 0; i < SERVE_MAX_CONNECTIONS; i++) {
        const serve_conn *c = &s->conns[i];
        if (i == except) continue;
        if (c->active && c->has_session &&
            memcmp(c->session.bytes, id->bytes, 16) == 0)
            return i;
    }
    return -1;
}

static int serve_route(serve *s, serve_conn *c, const json_value *request) {
    const char *op = serve_str(request, "op");
    if (!op) {
        serve_error(s, c, RUNTIME_INVALID_ARGUMENT, "missing op");
        return SERVE_ROUTE_DONE;
    }
    if (!strcmp(op, "describe") || !strcmp(op, "list") ||
        !strcmp(op, "stat"))
        return SERVE_ROUTE_INLINE;
    if (!strcmp(op, "cancel")) {
        serve_cancel(s, c);
        return SERVE_ROUTE_DONE;
    }
    if (!strcmp(op, "delete")) {
        conversation_id id;
        if (!serve_id_parse(serve_str(request, "session"), &id)) {
            serve_error(s, c, RUNTIME_INVALID_ARGUMENT, "invalid session id");
            return SERVE_ROUTE_DONE;
        }
        if (serve_binding_other(s, &id, serve_conn_index(s, c)) >= 0) {
            serve_error(s, c, RUNTIME_BUSY,
                        "session is bound by another connection");
            return SERVE_ROUTE_DONE;
        }
        int bound = c->has_session &&
                    memcmp(c->session.bytes, id.bytes, 16) == 0;
        if (bound || s->generating) return SERVE_ROUTE_QUEUE;
        return SERVE_ROUTE_INLINE;
    }
    return SERVE_ROUTE_QUEUE;
}

static int serve_needs_session(const char *op) {
    return !strcmp(op, "append") || !strcmp(op, "generate") ||
           !strcmp(op, "rewind") || !strcmp(op, "rewind_cost") ||
           !strcmp(op, "rebuild") || !strcmp(op, "checkpoint") ||
           !strcmp(op, "history") || !strcmp(op, "pending");
}

static int serve_execute(serve *s, serve_conn *c,
                         const json_value *request) {
    const char *op = serve_str(request, "op");
    if (op && serve_needs_session(op)) {
        if (!c->has_session) {
            serve_error(s, c, RUNTIME_SESSION_NOT_FOUND, "no open session");
            return 0;
        }
        runtime_status status = runtime_session_open(s->w, &c->session, NULL);
        if (status != RUNTIME_OK) {
            serve_runtime_error(s, c, status);
            return 0;
        }
    }
    return serve_dispatch(s, c, request);
}

static void serve_run_queue(serve *s) {
    while (!s->generating && s->queue_count > 0) {
        int index = s->queue[s->queue_head];
        s->queue_head = (s->queue_head + 1) % SERVE_MAX_CONNECTIONS;
        s->queue_count--;
        serve_conn *c = &s->conns[index];
        json_value *request = c->request;
        c->request = NULL;
        c->queued = 0;
        if (!c->active || !request) {
            json_free(request);
            continue;
        }
        s->active_at = serve_now_ms();
        int started = serve_execute(s, c, request);
        json_free(request);
        if (started) {
            s->generating = 1;
            s->gen_owner = index;
        }
        s->active_at = serve_now_ms();
    }
}

static void serve_stream_pull(serve *s) {
    serve_conn *owner = s->gen_owner >= 0 ? &s->conns[s->gen_owner] : NULL;
    if (owner && !owner->active) owner = NULL;
    runtime_event event = {0};
    runtime_status status = runtime_next_event(s->w, &event);
    if (status != RUNTIME_OK) {
        event.kind = RUNTIME_EVENT_ERROR;
        event.error = status;
        event.error_text = runtime_error_text(s->w);
        if (owner) serve_emit_event(s, owner, &event);
        s->generating = 0;
        s->gen_owner = -1;
        return;
    }
    if (owner) serve_emit_event(s, owner, &event);
    if (event.kind == RUNTIME_EVENT_DONE || event.kind == RUNTIME_EVENT_ERROR) {
        s->generating = 0;
        s->gen_owner = -1;
    }
}

static int serve_take_line(serve_conn *c, size_t *length) {
    if (!c->in_length) return 0;
    const char *nl = memchr(c->in, '\n', c->in_length);
    if (!nl) return 0;
    *length = (size_t)(nl - c->in) + 1;
    return 1;
}

static void serve_consume(serve_conn *c, size_t length) {
    c->head_checked = 0;
    if (length >= c->in_length) {
        c->in_length = 0;
        return;
    }
    memmove(c->in, c->in + length, c->in_length - length);
    c->in_length -= length;
}

static void serve_handle_request(serve *s, serve_conn *c,
                                 json_value *request) {
    if (!request || request->type != JSON_OBJECT) {
        serve_error(s, c, RUNTIME_INVALID_ARGUMENT, "invalid request");
        json_free(request);
        return;
    }
    int route = serve_route(s, c, request);
    if (route == SERVE_ROUTE_QUEUE) {
        c->request = request;
        c->queued = 1;
        s->queue[(s->queue_head + s->queue_count) % SERVE_MAX_CONNECTIONS] =
            serve_conn_index(s, c);
        s->queue_count++;
        return;
    }
    if (route == SERVE_ROUTE_INLINE) {
        s->active_at = serve_now_ms();
        serve_dispatch(s, c, request);
    }
    json_free(request);
}

static void serve_input(serve *s, serve_conn *c) {
    while (c->active) {
        size_t length;
        if (c->sequential && s->generating) break;
        if (!serve_take_line(c, &length)) break;
        if (length == 1) {
            serve_consume(c, length);
            continue;
        }
        int index = serve_conn_index(s, c);
        if (s->generating && s->gen_owner == index) {
            json_value *request = json_parse(c->in, length);
            serve_consume(c, length);
            const char *op = request ? serve_str(request, "op") : NULL;
            if (op && !strcmp(op, "cancel")) {
                runtime_cancel(s->w);
                serve_ok_begin(s);
                json_raw(&s->out, "}");
                serve_emit(s, c);
            } else {
                serve_error(s, c, RUNTIME_BUSY, "generation in progress");
            }
            json_free(request);
            continue;
        }
        if (c->queued) {
            if (c->head_checked) break;
            json_value *request = json_parse(c->in, length);
            const char *op = request ? serve_str(request, "op") : NULL;
            int is_cancel = op && !strcmp(op, "cancel");
            json_free(request);
            if (!is_cancel) {
                c->head_checked = 1;
                break;
            }
            serve_consume(c, length);
            serve_cancel(s, c);
            continue;
        }
        json_value *request = json_parse(c->in, length);
        serve_consume(c, length);
        serve_handle_request(s, c, request);
    }
    if (c->active && c->in_length >= s->max_frame &&
        !memchr(c->in, '\n', c->in_length)) {
        serve_error(s, c, RUNTIME_INVALID_ARGUMENT, "request line is too long");
        serve_drop(s, c);
    }
}

static void serve_residue(serve *s, serve_conn *c) {
    if (!c->active || !c->in_length) return;
    if (memchr(c->in, '\n', c->in_length)) return;
    c->in_length = 0;
    serve_error(s, c, RUNTIME_INVALID_ARGUMENT, "invalid request");
}

static void serve_read(serve *s, serve_conn *c) {
    for (;;) {
        if (c->in_length + SERVE_READ_CHUNK > c->in_capacity) {
            size_t capacity = c->in_capacity ? c->in_capacity
                                             : SERVE_READ_CHUNK;
            while (capacity < c->in_length + SERVE_READ_CHUNK) capacity *= 2;
            char *grown = realloc(c->in, capacity);
            if (!grown) {
                serve_drop(s, c);
                return;
            }
            c->in = grown;
            c->in_capacity = capacity;
        }
        size_t room = c->in_capacity - c->in_length;
        if (c->in_length <= s->max_frame &&
            room > s->max_frame - c->in_length + 1)
            room = s->max_frame - c->in_length + 1;
        if (!room) return;
        ssize_t n = read(c->in_fd, c->in + c->in_length, room);
        if (n > 0) {
            for (ssize_t i = 0; i < n; i++)
                if ((unsigned char)c->in[c->in_length + (size_t)i] > ' ') {
                    s->active_at = serve_now_ms();
                    break;
                }
            c->in_length += (size_t)n;
            c->head_checked = 0;
            if (c->in_length >= s->max_frame &&
                !memchr(c->in, '\n', c->in_length)) {
                serve_error(s, c, RUNTIME_INVALID_ARGUMENT,
                            "request line is too long");
                serve_drop(s, c);
                return;
            }
            if (c->in_length >= s->max_frame || !c->is_socket) return;
            continue;
        }
        if (n == 0) {
            if (s->generating && s->gen_owner == serve_conn_index(s, c))
                runtime_cancel(s->w);
            c->eof = 1;
            return;
        }
        if (errno == EINTR) continue;
        if (errno == EAGAIN || errno == EWOULDBLOCK) return;
        serve_drop(s, c);
        return;
    }
}

static int serve_can_read(const serve *s, const serve_conn *c) {
    if (c->eof) return 0;
    if (c->sequential && s->generating) return 0;
    return c->in_length < s->max_frame;
}

static void serve_reap(serve *s) {
    for (int i = 0; i < SERVE_MAX_CONNECTIONS; i++) {
        serve_conn *c = &s->conns[i];
        if (!c->active || !c->eof || c->queued) continue;
        if (s->generating && s->gen_owner == i) continue;
        if (c->in_length && memchr(c->in, '\n', c->in_length)) continue;
        serve_residue(s, c);
        if (!c->active) continue;
        serve_flush(s, c);
        if (!c->active || c->out_length > c->out_sent) continue;
        serve_drop(s, c);
    }
}

static void serve_accept(serve *s) {
    for (;;) {
        int fd = accept4(s->listen_fd, NULL, NULL,
                         SOCK_NONBLOCK | SOCK_CLOEXEC);
        if (fd < 0) {
            if (errno == EMFILE || errno == ENFILE) s->accept_paused = 1;
            return;
        }
        s->accept_paused = 0;
        s->active_at = serve_now_ms();
        struct ucred cred;
        socklen_t length = sizeof cred;
        if (getsockopt(fd, SOL_SOCKET, SO_PEERCRED, &cred, &length) != 0 ||
            cred.uid != getuid()) {
            close(fd);
            continue;
        }
        int slot = -1;
        for (int i = 0; i < SERVE_MAX_CONNECTIONS; i++)
            if (!s->conns[i].active) {
                slot = i;
                break;
            }
        if (slot < 0) {
            close(fd);
            continue;
        }
        serve_conn *c = &s->conns[slot];
        memset(c, 0, sizeof *c);
        c->active = 1;
        c->is_socket = 1;
        c->in_fd = fd;
        c->out_fd = fd;
    }
}

static void serve_poll(serve *s, int timeout) {
    struct pollfd fds[SERVE_MAX_CONNECTIONS + 2];
    int slots[SERVE_MAX_CONNECTIONS + 2];
    nfds_t n = 0;
    if (s->listen_fd >= 0 && !s->accept_paused) {
        fds[n].fd = s->listen_fd;
        fds[n].events = POLLIN;
        fds[n].revents = 0;
        slots[n] = -1;
        n++;
    }
    if (serve_signal_fd[0] >= 0) {
        fds[n].fd = serve_signal_fd[0];
        fds[n].events = POLLIN;
        fds[n].revents = 0;
        slots[n] = -2;
        n++;
    }
    for (int i = 0; i < SERVE_MAX_CONNECTIONS; i++) {
        serve_conn *c = &s->conns[i];
        if (!c->active) continue;
        short events = 0;
        if (serve_can_read(s, c)) events |= POLLIN;
        if (c->is_socket && c->out_length > c->out_sent) events |= POLLOUT;
        if (!events) continue;
        fds[n].fd = c->in_fd;
        fds[n].events = events;
        fds[n].revents = 0;
        slots[n] = i;
        n++;
    }
    if (!n) {
        s->accept_paused = 0;
        return;
    }
    if (s->accept_paused && (timeout < 0 || timeout > SERVE_ACCEPT_RETRY_MS))
        timeout = SERVE_ACCEPT_RETRY_MS;
    if (poll(fds, n, timeout) <= 0) {
        s->accept_paused = 0;
        return;
    }
    for (nfds_t i = 0; i < n; i++) {
        if (!fds[i].revents) continue;
        if (slots[i] == -1) {
            serve_accept(s);
            continue;
        }
        if (slots[i] == -2) {
            char drain[64];
            ssize_t got = read(serve_signal_fd[0], drain, sizeof drain);
            (void)got;
            s->stop = 1;
            continue;
        }
        serve_conn *c = &s->conns[slots[i]];
        if (!c->active) continue;
        if (fds[i].revents & POLLOUT) serve_flush(s, c);
        if (!c->active) continue;
        if (fds[i].revents & (POLLIN | POLLHUP | POLLERR)) {
            serve_read(s, c);
            if (c->active) serve_input(s, c);
        }
    }
}

static void serve_drain(serve *s) {
    for (int i = 0; i < SERVE_MAX_CONNECTIONS; i++) {
        serve_conn *c = &s->conns[i];
        if (c->active && c->in_length) serve_input(s, c);
    }
}

static int serve_pending_input(const serve *s) {
    for (int i = 0; i < SERVE_MAX_CONNECTIONS; i++) {
        const serve_conn *c = &s->conns[i];
        if (!c->active || c->queued || !c->in_length) continue;
        if (memchr(c->in, '\n', c->in_length)) return 1;
    }
    return 0;
}

static int serve_finished(const serve *s) {
    if (s->listen_fd >= 0) return 0;
    if (s->generating || s->queue_count) return 0;
    for (int i = 0; i < SERVE_MAX_CONNECTIONS; i++) {
        const serve_conn *c = &s->conns[i];
        if (!c->active) continue;
        if (!c->eof) return 0;
        if (c->in_length && memchr(c->in, '\n', c->in_length)) return 0;
    }
    return 1;
}

static int serve_timeout(const serve *s) {
    if (s->idle_ms <= 0) return -1;
    int64_t remaining = s->idle_ms - (serve_now_ms() - s->active_at);
    if (remaining <= 0) return 0;
    if (remaining > INT_MAX) return INT_MAX;
    return (int)remaining;
}

static int serve_idle_expired(const serve *s) {
    if (s->idle_ms <= 0 || s->generating || s->queue_count) return 0;
    return serve_now_ms() - s->active_at >= s->idle_ms;
}

static void serve_loop(serve *s) {
    for (;;) {
        serve_drain(s);
        serve_run_queue(s);
        serve_reap(s);
        if (s->stop || serve_finished(s)) return;
        if (s->generating) {
            serve_stream_pull(s);
            s->active_at = serve_now_ms();
            serve_poll(s, 0);
        } else {
            serve_poll(s, serve_pending_input(s) ? 0 : serve_timeout(s));
            if (serve_idle_expired(s)) {
                s->stop = 1;
                return;
            }
        }
        if (s->stop) return;
    }
}

static void serve_shutdown_checkpoint(serve *s) {
    runtime_checkpoint_report r;
    if (runtime_checkpoint(s->w, &r) == RUNTIME_OK) serve_log_checkpoint("shutdown", &r);
}

static void serve_shutdown(serve *s) {
    if (s->generating) {
        runtime_cancel(s->w);
        while (s->generating) serve_stream_pull(s);
    }
    serve_shutdown_checkpoint(s);
    for (int i = 0; i < SERVE_MAX_CONNECTIONS; i++) {
        serve_conn *c = &s->conns[i];
        if (!c->active) continue;
        serve_flush(s, c);
        serve_drop(s, c);
    }
    if (s->listen_fd >= 0) {
        close(s->listen_fd);
        s->listen_fd = -1;
        if (s->socket_path) unlink(s->socket_path);
    }
}

static void serve_init(serve *s, const xe_engine *engine) {
    memset(s, 0, sizeof *s);
    s->listen_fd = -1;
    s->max_frame = (size_t)xe_context_size(engine) *
                   SERVE_FRAME_BYTES_PER_TOKEN;
    s->gen_owner = -1;
    for (int i = 0; i < SERVE_MAX_CONNECTIONS; i++) {
        s->conns[i].in_fd = -1;
        s->conns[i].out_fd = -1;
    }
}

static int serve_stdin_interactive(void) {
    struct stat st;
    if (fstat(0, &st) != 0) return 0;
    return !S_ISREG(st.st_mode);
}

int serve_stdio(xe_engine *engine, const char *state_dir,
                const char *cache_dir) {
    serve s;
    serve_init(&s, engine);
    runtime_status status = runtime_open(&s.w, engine, state_dir, cache_dir);
    if (status != RUNTIME_OK) {
        fprintf(stderr, "xenolith: wire: %s\n", runtime_status_code(status));
        return 1;
    }
    serve_log_kvstore(s.w, cache_dir);
    serve_conn *c = &s.conns[0];
    c->active = 1;
    c->is_socket = 0;
    c->in_fd = 0;
    c->out_fd = 1;
    c->sequential = !serve_stdin_interactive();
    s.active_at = serve_now_ms();
    serve_loop(&s);
    serve_shutdown_checkpoint(&s);
    for (int i = 0; i < SERVE_MAX_CONNECTIONS; i++) {
        if (s.conns[i].active) {
            serve_residue(&s, &s.conns[i]);
            serve_flush(&s, &s.conns[i]);
            serve_drop(&s, &s.conns[i]);
        }
    }
    json_writer_free(&s.out);
    runtime_close(s.w);
    return 0;
}

static void serve_signal_reset(void) {
    if (serve_signal_fd[0] < 0) return;
    signal(SIGTERM, SIG_DFL);
    signal(SIGINT, SIG_DFL);
    close(serve_signal_fd[0]);
    close(serve_signal_fd[1]);
    serve_signal_fd[0] = -1;
    serve_signal_fd[1] = -1;
}

static int serve_listen(serve *s, const char *path) {
    struct sockaddr_un addr;
    memset(&addr, 0, sizeof addr);
    if (!path || strlen(path) >= sizeof addr.sun_path) return 0;
    int fd = socket(AF_UNIX, SOCK_STREAM | SOCK_NONBLOCK | SOCK_CLOEXEC, 0);
    if (fd < 0) return 0;
    addr.sun_family = AF_UNIX;
    memcpy(addr.sun_path, path, strlen(path));
    unlink(path);
    if (bind(fd, (struct sockaddr *)&addr, sizeof addr) != 0 ||
        listen(fd, SERVE_BACKLOG) != 0) {
        close(fd);
        return 0;
    }
    s->listen_fd = fd;
    s->socket_path = path;
    return 1;
}

int serve_run(xe_engine *engine, const char *state_dir, const char *cache_dir,
              const char *socket_path, double idle_minutes) {
    serve s;
    serve_init(&s, engine);
    if (idle_minutes > 0.0) {
        double ms = idle_minutes * 60000.0;
        if (ms > 9.0e15) ms = 9.0e15;
        s.idle_ms = ms < 1.0 ? 1 : (int64_t)ms;
    }
    runtime_status status = runtime_open(&s.w, engine, state_dir, cache_dir);
    if (status != RUNTIME_OK) {
        fprintf(stderr, "xenolith: serve: %s\n", runtime_status_code(status));
        return 1;
    }
    serve_log_kvstore(s.w, cache_dir);
    signal(SIGPIPE, SIG_IGN);
    if (pipe(serve_signal_fd) == 0) {
        fcntl(serve_signal_fd[0], F_SETFL, O_NONBLOCK);
        fcntl(serve_signal_fd[1], F_SETFL, O_NONBLOCK);
        struct sigaction action;
        memset(&action, 0, sizeof action);
        action.sa_handler = serve_on_signal;
        sigemptyset(&action.sa_mask);
        action.sa_flags = SA_RESTART;
        sigaction(SIGTERM, &action, NULL);
        sigaction(SIGINT, &action, NULL);
    }
    if (!serve_listen(&s, socket_path)) {
        fprintf(stderr, "xenolith: serve: cannot listen on %s\n",
                socket_path ? socket_path : "");
        json_writer_free(&s.out);
        runtime_close(s.w);
        serve_signal_reset();
        return 1;
    }
    s.active_at = serve_now_ms();
    serve_loop(&s);
    serve_shutdown(&s);
    json_writer_free(&s.out);
    runtime_close(s.w);
    serve_signal_reset();
    return 0;
}
