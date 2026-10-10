#define _GNU_SOURCE
#define _POSIX_C_SOURCE 200809L

#include <errno.h>
#include "test_context.h"
#include "../json.h"
#include <fcntl.h>
#include <poll.h>
#include <signal.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/file.h>
#include <sys/socket.h>
#include <sys/stat.h>
#include <sys/un.h>
#include <sys/wait.h>
#include <time.h>
#include <unistd.h>

static int failures;
static char context_arg[16];

#define CHECK(condition) do { \
    if (!(condition)) { \
        failures++; \
        fprintf(stderr, "FAIL %s:%d: %s\n", __FILE__, __LINE__, \
                #condition); \
    } \
} while (0)

typedef struct {
    int fd;
    char buf[262144];
    size_t length;
    const char *name;
} client;

static void test_sleep_ms(int ms) {
    struct timespec ts = { ms / 1000, (long)(ms % 1000) * 1000000L };
    nanosleep(&ts, NULL);
}

static int client_connect(client *c, const char *path, const char *name,
                          int attempts) {
    memset(c, 0, sizeof *c);
    c->fd = -1;
    c->name = name;
    for (int attempt = 0; attempt < attempts; attempt++) {
        int fd = socket(AF_UNIX, SOCK_STREAM, 0);
        if (fd < 0) return 0;
        struct sockaddr_un addr;
        memset(&addr, 0, sizeof addr);
        addr.sun_family = AF_UNIX;
        memcpy(addr.sun_path, path, strlen(path));
        if (connect(fd, (struct sockaddr *)&addr, sizeof addr) == 0) {
            c->fd = fd;
            return 1;
        }
        close(fd);
        test_sleep_ms(200);
    }
    return 0;
}

static void client_close(client *c) {
    if (c->fd >= 0) close(c->fd);
    c->fd = -1;
}

static int client_send(client *c, const char *line) {
    size_t length = strlen(line);
    size_t sent = 0;
    while (sent < length) {
        ssize_t n = write(c->fd, line + sent, length - sent);
        if (n <= 0) return 0;
        sent += (size_t)n;
    }
    return 1;
}

static int client_buffered(client *c, char *out, size_t cap) {
    char *nl = memchr(c->buf, '\n', c->length);
    if (!nl) return 0;
    size_t length = (size_t)(nl - c->buf);
    if (length >= cap) length = cap - 1;
    memcpy(out, c->buf, length);
    out[length] = '\0';
    size_t used = (size_t)(nl - c->buf) + 1;
    memmove(c->buf, c->buf + used, c->length - used);
    c->length -= used;
    return 1;
}

static int client_fill(client **clients, int count, int timeout_ms) {
    struct pollfd fds[8];
    int n = 0;
    for (int i = 0; i < count; i++) {
        if (clients[i]->fd < 0) continue;
        fds[n].fd = clients[i]->fd;
        fds[n].events = POLLIN;
        fds[n].revents = 0;
        n++;
    }
    if (!n) return 0;
    if (poll(fds, (nfds_t)n, timeout_ms) <= 0) return 0;
    int got = 0;
    n = 0;
    for (int i = 0; i < count; i++) {
        client *c = clients[i];
        if (c->fd < 0) continue;
        int index = n++;
        if (!fds[index].revents) continue;
        if (c->length >= sizeof c->buf) continue;
        ssize_t r = read(c->fd, c->buf + c->length,
                         sizeof c->buf - c->length);
        if (r <= 0) continue;
        c->length += (size_t)r;
        got = 1;
    }
    return got;
}

static int client_next(client **clients, int count, int *who, char *out,
                       size_t cap, int timeout_ms) {
    int waited = 0;
    for (;;) {
        for (int i = 0; i < count; i++)
            if (clients[i]->fd >= 0 && client_buffered(clients[i], out, cap)) {
                *who = i;
                return 1;
            }
        if (waited >= timeout_ms) return 0;
        client_fill(clients, count, 200);
        waited += 200;
    }
}

static int client_recv(client *c, char *out, size_t cap, int timeout_ms) {
    client *one = c;
    int who;
    return client_next(&one, 1, &who, out, cap, timeout_ms);
}

static int client_call(client *c, const char *request, char *out,
                       size_t cap) {
    if (!client_send(c, request)) return 0;
    return client_recv(c, out, cap, 300000);
}

static int session_of(const char *line, char *out) {
    const char *p = strstr(line, "\"session\":\"");
    if (!p) return 0;
    p += strlen("\"session\":\"");
    if (strlen(p) < 32) return 0;
    memcpy(out, p, 32);
    out[32] = '\0';
    return 1;
}

static int marker_of(const char *line, unsigned long long *out) {
    const char *p = strstr(line, "\"marker\":");
    if (!p) return 0;
    p += strlen("\"marker\":");
    if (*p < '0' || *p > '9') return 0;
    *out = strtoull(p, NULL, 10);
    return 1;
}

static const char *last_field(const char *line, const char *tag) {
    const char *found = NULL;
    for (const char *p = strstr(line, tag); p; p = strstr(p + 1, tag))
        found = p + strlen(tag);
    return found;
}

static void delta_append(const char *line, char *text, size_t cap) {
    const char *p = strstr(line, "\"text\":\"");
    if (!p) return;
    p += strlen("\"text\":\"");
    const char *end = strrchr(p, '"');
    if (!end || end < p) return;
    size_t used = strlen(text);
    size_t length = (size_t)(end - p);
    if (used + length + 1 >= cap) length = cap - used - 1;
    memcpy(text + used, p, length);
    text[used + length] = '\0';
}

static pid_t spawn_server(const char *model, const char *state_dir,
                          const char *cache_dir, const char *socket_path,
                          const char *idle, const char *log_path) {
    fflush(NULL);
    pid_t pid = fork();
    if (pid != 0) return pid;
    int fd = open(log_path, O_WRONLY | O_CREAT | O_APPEND, 0600);
    if (fd >= 0) {
        dup2(fd, 1);
        dup2(fd, 2);
        close(fd);
    }
    if (idle)
        execl("./xenolith", "xenolith", "serve", model, "--ctx", context_arg, "--state", state_dir,
              "--cache", cache_dir, "--socket", socket_path,
              "--idle-shutdown", idle, (char *)NULL);
    else
        execl("./xenolith", "xenolith", "serve", model, "--ctx", context_arg, "--state", state_dir,
              "--cache", cache_dir, "--socket", socket_path, (char *)NULL);
    _exit(127);
    return -1;
}

static int wait_exit(pid_t pid, int seconds) {
    for (int i = 0; i < seconds * 10; i++) {
        int status = 0;
        pid_t got = waitpid(pid, &status, WNOHANG);
        if (got == pid)
            return WIFEXITED(status) ? WEXITSTATUS(status) : -1;
        test_sleep_ms(100);
    }
    return -2;
}

int main(int argc, char **argv) {
    int context = test_context_capacity(&argc, argv);
    if (argc < 2 || !argv[1][0]) {
        fprintf(stderr, "usage: %s <model.gguf> [--ctx N]\n", argv[0]);
        return 2;
    }
    const char *model = argv[1];
    snprintf(context_arg, sizeof context_arg, "%d", context);
    char root[] = "/tmp/xenolith-test-sm-XXXXXX";
    CHECK(mkdtemp(root) != NULL);
    signal(SIGPIPE, SIG_IGN);

    char state_dir[256], cache_dir[256], socket_path[256], log_path[256];
    char runtime_dir[256], state_home[256];
    char other_state[256], other_socket[256];
    snprintf(state_dir, sizeof state_dir, "%s/state", root);
    snprintf(cache_dir, sizeof cache_dir, "%s/cache", root);
    snprintf(socket_path, sizeof socket_path, "%s/wire.sock", root);
    snprintf(log_path, sizeof log_path, "%s/serve.log", root);
    snprintf(runtime_dir, sizeof runtime_dir, "%s/run", root);
    snprintf(state_home, sizeof state_home, "%s/home", root);
    snprintf(other_state, sizeof other_state, "%s/other-state", root);
    snprintf(other_socket, sizeof other_socket, "%s/other.sock", root);
    CHECK(mkdir(runtime_dir, 0700) == 0);
    CHECK(mkdir(state_home, 0700) == 0);
    CHECK(setenv("XDG_RUNTIME_DIR", runtime_dir, 1) == 0);
    CHECK(setenv("XDG_STATE_HOME", state_home, 1) == 0);

    pid_t server = spawn_server(model, state_dir, cache_dir, socket_path,
                                "0.2", log_path);
    CHECK(server > 0);

    char line[16384];
    client a, b, c;
    if (!client_connect(&a, socket_path, "A", 900)) {
        kill(server, SIGKILL);
        fprintf(stderr, "test_serve_model: server never accepted\n");
        return 1;
    }
    CHECK(client_connect(&b, socket_path, "B", 50));
    CHECK(client_connect(&c, socket_path, "C", 50));

    CHECK(client_call(&a, "{\"op\":\"describe\"}\n", line, sizeof line));
    CHECK(strstr(line, "\"protocol\":1") != NULL);
    char context_field[64];
    snprintf(context_field, sizeof context_field, "\"context_window\":%d", context);
    CHECK(strstr(line, context_field) != NULL);

    pid_t intruder = spawn_server(model, other_state, cache_dir, other_socket,
                                  NULL, log_path);
    CHECK(intruder > 0);
    CHECK(wait_exit(intruder, 60) == 1);
    struct stat intruder_st;
    CHECK(stat(other_socket, &intruder_st) != 0);

    char session_a[64], session_b[64];
    CHECK(client_call(&a, "{\"op\":\"create\",\"system\":\"You are a "
                          "helpful assistant.\"}\n", line, sizeof line));
    CHECK(session_of(line, session_a));
    CHECK(client_call(&b, "{\"op\":\"create\",\"system\":\"You are a "
                          "helpful assistant.\"}\n", line, sizeof line));
    CHECK(session_of(line, session_b));

    CHECK(client_call(&a, "{\"op\":\"append\",\"role\":\"user\",\"text\":"
                          "\"Count from one to five in words.\"}\n",
                      line, sizeof line));
    CHECK(strstr(line, "\"ok\":true") != NULL);
    CHECK(client_send(&a, "{\"op\":\"generate\",\"temperature\":1,"
                          "\"top_k\":1,\"max_tokens\":40,\"seed\":0}\n"));

    client *all[3] = { &a, &b, &c };
    int who = 0;
    int started = 0, list_seen = 0, b_lines = 0, a_errors = 0;
    int a_done = 0, b_done = 0;
    for (int i = 0; i < 4096 && !a_done; i++) {
        if (!client_next(all, 3, &who, line, sizeof line, 300000)) {
            failures++;
            break;
        }
        if (who == 0) {
            if (strstr(line, "\"ok\":false")) a_errors++;
            if (strstr(line, "\"event\":\"text_delta\"") && !started) {
                started = 1;
                CHECK(client_send(&b,
                    "{\"op\":\"append\",\"role\":\"user\",\"text\":"
                    "\"Name one color.\"}\n"
                    "{\"op\":\"generate\",\"temperature\":1,\"top_k\":1,"
                    "\"max_tokens\":16,\"seed\":0}\n"));
                CHECK(client_send(&c, "{\"op\":\"list\"}\n"));
            }
            if (strstr(line, "\"event\":\"done\"")) a_done = 1;
        } else if (who == 1) {
            b_lines++;
        } else if (strstr(line, "\"sessions\":[")) {
            list_seen = 1;
        }
    }
    CHECK(started == 1);
    CHECK(a_errors == 0);
    CHECK(list_seen == 1);
    CHECK(b_lines == 0);
    for (int i = 0; i < 4096 && !b_done; i++) {
        if (!client_next(all, 3, &who, line, sizeof line, 300000)) {
            failures++;
            break;
        }
        if (who != 1) continue;
        if (strstr(line, "\"event\":\"done\"")) b_done = 1;
        if (strstr(line, "\"ok\":false")) failures++;
    }

    CHECK(client_call(&a, "{\"op\":\"append\",\"role\":\"user\",\"text\":"
                          "\"Say one short sentence about the sea.\"}\n",
                      line, sizeof line));
    unsigned long long marker = 0;
    CHECK(marker_of(line, &marker));
    static char first[16384], second[16384];
    first[0] = '\0';
    second[0] = '\0';
    CHECK(client_send(&a, "{\"op\":\"generate\",\"temperature\":1,"
                          "\"top_k\":1,\"max_tokens\":32,\"seed\":0}\n"));
    for (int i = 0; i < 4096; i++) {
        if (!client_recv(&a, line, sizeof line, 300000)) {
            failures++;
            break;
        }
        if (strstr(line, "\"event\":\"text_delta\""))
            delta_append(line, first, sizeof first);
        if (strstr(line, "\"event\":\"done\"")) break;
        if (strstr(line, "\"ok\":false")) {
            failures++;
            break;
        }
    }
    CHECK(strlen(first) > 0);

    CHECK(client_call(&b, "{\"op\":\"append\",\"role\":\"user\",\"text\":"
                          "\"Name one fruit.\"}\n", line, sizeof line));
    CHECK(client_send(&b, "{\"op\":\"generate\",\"temperature\":1,"
                          "\"top_k\":1,\"max_tokens\":16,\"seed\":0}\n"));
    for (int i = 0; i < 4096; i++) {
        if (!client_recv(&b, line, sizeof line, 300000)) {
            failures++;
            break;
        }
        if (strstr(line, "\"event\":\"done\"")) break;
        if (strstr(line, "\"ok\":false")) {
            failures++;
            break;
        }
    }

    char request[512];
    snprintf(request, sizeof request,
             "{\"op\":\"rewind\",\"marker\":%llu}\n", marker);
    CHECK(client_call(&a, request, line, sizeof line));
    CHECK(strstr(line, "\"ok\":true") != NULL);
    CHECK(client_send(&a, "{\"op\":\"generate\",\"temperature\":1,"
                          "\"top_k\":1,\"max_tokens\":32,\"seed\":0}\n"));
    for (int i = 0; i < 4096; i++) {
        if (!client_recv(&a, line, sizeof line, 300000)) {
            failures++;
            break;
        }
        if (strstr(line, "\"event\":\"text_delta\""))
            delta_append(line, second, sizeof second);
        if (strstr(line, "\"event\":\"done\"")) break;
        if (strstr(line, "\"ok\":false")) {
            failures++;
            break;
        }
    }
    CHECK(strcmp(first, second) == 0);

    client f1, f2, f3;
    char session_f[64];
    CHECK(client_connect(&f1, socket_path, "F1", 50));
    CHECK(client_connect(&f2, socket_path, "F2", 50));
    CHECK(client_connect(&f3, socket_path, "F3", 50));
    CHECK(client_call(&f1, "{\"op\":\"create\",\"system\":\"Queue.\"}\n", line,
                      sizeof line));
    CHECK(session_of(line, session_f));
    snprintf(request, sizeof request,
             "{\"op\":\"open\",\"session\":\"%s\"}\n", session_f);
    CHECK(client_call(&f2, request, line, sizeof line));
    CHECK(strstr(line, "\"ok\":true") != NULL);
    CHECK(client_call(&f3, request, line, sizeof line));
    CHECK(strstr(line, "\"ok\":true") != NULL);
    CHECK(client_call(&a, "{\"op\":\"append\",\"role\":\"user\",\"text\":"
                          "\"Tell me a long story about a mountain.\"}\n",
                      line, sizeof line));
    CHECK(strstr(line, "\"ok\":true") != NULL);
    CHECK(client_send(&a, "{\"op\":\"generate\",\"temperature\":1,"
                          "\"top_k\":1,\"max_tokens\":120,\"seed\":0}\n"));
    int fifo_sent = 0;
    for (int i = 0; i < 4096; i++) {
        if (!client_recv(&a, line, sizeof line, 300000)) {
            failures++;
            break;
        }
        if (strstr(line, "\"event\":\"text_delta\"") && !fifo_sent) {
            fifo_sent = 1;
            CHECK(client_send(&f1, "{\"op\":\"append\",\"role\":\"user\","
                                   "\"text\":\"queued-one\"}\n"));
            test_sleep_ms(300);
            CHECK(client_send(&f2, "{\"op\":\"append\",\"role\":\"user\","
                                   "\"text\":\"queued-two\"}\n"));
            test_sleep_ms(300);
            CHECK(client_send(&f3, "{\"op\":\"append\",\"role\":\"user\","
                                   "\"text\":\"queued-three\"}\n"));
        }
        if (strstr(line, "\"event\":\"done\"")) break;
        if (strstr(line, "\"ok\":false")) {
            failures++;
            break;
        }
    }
    CHECK(fifo_sent == 1);
    client *fifo[3] = { &f1, &f2, &f3 };
    for (int i = 0; i < 3; i++) {
        int served = -1;
        CHECK(client_next(fifo, 3, &served, line, sizeof line, 300000));
        CHECK(served == i);
        CHECK(strstr(line, "\"ok\":true") != NULL);
    }
    CHECK(client_call(&f1, "{\"op\":\"history\"}\n", line, sizeof line));
    const char *one = strstr(line, "queued-one");
    const char *two = strstr(line, "queued-two");
    const char *three = strstr(line, "queued-three");
    CHECK(one != NULL && two != NULL && three != NULL);
    if (one && two && three) {
        CHECK(one < two);
        CHECK(two < three);
    }
    client_close(&f1);
    client_close(&f2);
    client_close(&f3);

    CHECK(client_call(&a, "{\"op\":\"append\",\"role\":\"user\",\"text\":"
                          "\"Tell me a very long story about the sea.\"}\n",
                      line, sizeof line));
    CHECK(client_send(&a, "{\"op\":\"generate\",\"temperature\":1,"
                          "\"top_k\":1,\"max_tokens\":300,\"seed\":0}\n"));
    int cancel_sent = 0, cancel_ok = 0, aborted = 0;
    for (int i = 0; i < 4096; i++) {
        if (!client_recv(&a, line, sizeof line, 300000)) {
            failures++;
            break;
        }
        if (strstr(line, "\"event\":\"text_delta\"") && !cancel_sent) {
            cancel_sent = 1;
            CHECK(client_send(&a, "{\"op\":\"cancel\"}\n"));
        }
        if (strstr(line, "\"ok\":true")) cancel_ok = 1;
        if (strstr(line, "\"event\":\"done\"")) {
            aborted = strstr(line, "\"stop\":\"aborted\"") != NULL;
            break;
        }
    }
    CHECK(cancel_ok == 1);
    CHECK(aborted == 1);

    CHECK(client_call(&a, "{\"op\":\"append\",\"role\":\"user\",\"text\":"
                          "\"Never mind, just say bye.\"}\n", line,
                      sizeof line));
    CHECK(strstr(line, "\"ok\":true") != NULL);
    CHECK(client_send(&a, "{\"op\":\"generate\",\"temperature\":1,"
                          "\"top_k\":1,\"max_tokens\":16,\"seed\":0}\n"));
    for (int i = 0; i < 4096; i++) {
        if (!client_recv(&a, line, sizeof line, 300000)) {
            failures++;
            break;
        }
        if (strstr(line, "\"event\":\"done\"")) break;
        if (strstr(line, "\"ok\":false")) {
            failures++;
            break;
        }
    }

    CHECK(client_call(&a, "{\"op\":\"append\",\"role\":\"user\",\"text\":"
                          "\"Tell me a long story about a forest.\"}\n", line,
                      sizeof line));
    CHECK(strstr(line, "\"ok\":true") != NULL);
    CHECK(client_send(&a, "{\"op\":\"generate\",\"temperature\":1,"
                          "\"top_k\":1,\"max_tokens\":64,\"seed\":0}\n"));
    int queued_cancel = 0;
    for (int i = 0; i < 4096; i++) {
        if (!client_recv(&a, line, sizeof line, 300000)) {
            failures++;
            break;
        }
        if (strstr(line, "\"event\":\"text_delta\"") && !queued_cancel) {
            queued_cancel = 1;
            CHECK(client_send(&b,
                "{\"op\":\"append\",\"role\":\"user\",\"text\":"
                "\"Name one animal.\"}\n"
                "{\"op\":\"generate\",\"temperature\":1,\"top_k\":1,"
                "\"max_tokens\":16,\"seed\":0}\n"));
            test_sleep_ms(200);
            CHECK(client_send(&b, "{\"op\":\"cancel\"}\n"));
        }
        if (strstr(line, "\"event\":\"done\"")) break;
        if (strstr(line, "\"ok\":false")) {
            failures++;
            break;
        }
    }
    CHECK(queued_cancel == 1);
    CHECK(client_recv(&b, line, sizeof line, 300000));
    CHECK(strstr(line, "\"ok\":true") != NULL);
    CHECK(strstr(line, "\"marker\":") != NULL);
    CHECK(client_recv(&b, line, sizeof line, 300000));
    CHECK(strcmp(line, "{\"ok\":true}") == 0);
    CHECK(client_recv(&b, line, sizeof line, 300000));
    CHECK(strstr(line, "\"event\":\"done\"") != NULL);
    CHECK(strstr(line, "\"stop\":\"aborted\"") != NULL);
    CHECK(strstr(line, "\"marker\":null") != NULL);
    json_value *done = json_parse(line, strlen(line));
    const json_value *usage = json_member(done, "usage");
    const char *zero_fields[] = { "input", "cache_read", "output", "total" };
    for (size_t i = 0; i < sizeof zero_fields / sizeof zero_fields[0]; i++) {
        const json_value *field = json_member(usage, zero_fields[i]);
        CHECK(field && field->type == JSON_NUMBER && field->number == 0);
    }
    json_free(done);
    CHECK(client_call(&b, "{\"op\":\"history\"}\n", line, sizeof line));
    const char *tail_kind = last_field(line, "\"kind\":\"");
    CHECK(tail_kind != NULL && strncmp(tail_kind, "user", 4) == 0);
    CHECK(strstr(line, "Name one animal.") != NULL);

    CHECK(client_call(&b, "{\"op\":\"append\",\"role\":\"user\",\"text\":"
                          "\"Tell me a very long story about a river.\"}\n",
                      line, sizeof line));
    CHECK(client_send(&b, "{\"op\":\"generate\",\"temperature\":1,"
                          "\"top_k\":1,\"max_tokens\":300,\"seed\":0}\n"));
    for (int i = 0; i < 4096; i++) {
        if (!client_recv(&b, line, sizeof line, 300000)) {
            failures++;
            break;
        }
        if (strstr(line, "\"event\":\"text_delta\"")) break;
        if (strstr(line, "\"event\":\"done\"")) break;
    }
    client_close(&b);

    client d;
    CHECK(client_connect(&d, socket_path, "D", 50));
    snprintf(request, sizeof request,
             "{\"op\":\"open\",\"session\":\"%s\"}\n", session_b);
    CHECK(client_call(&d, request, line, sizeof line));
    CHECK(strstr(line, "\"ok\":true") != NULL);
    CHECK(strstr(line, "\"turn_open\":true") != NULL);
    CHECK(client_call(&d, "{\"op\":\"history\"}\n", line, sizeof line));
    const char *dropped_kind = last_field(line, "\"kind\":\"");
    const char *dropped_stop = last_field(line, "\"stop\":\"");
    CHECK(dropped_kind != NULL &&
          strncmp(dropped_kind, "assistant", 9) == 0);
    CHECK(dropped_stop != NULL &&
          strncmp(dropped_stop, "cancelled", 9) == 0);
    CHECK(client_call(&d, "{\"op\":\"append\",\"role\":\"user\",\"text\":"
                          "\"Never mind, just say bye.\"}\n", line,
                      sizeof line));
    CHECK(strstr(line, "\"ok\":true") != NULL);
    CHECK(client_send(&d, "{\"op\":\"generate\",\"temperature\":1,"
                          "\"top_k\":1,\"max_tokens\":16,\"seed\":0}\n"));
    for (int i = 0; i < 4096; i++) {
        if (!client_recv(&d, line, sizeof line, 300000)) {
            failures++;
            break;
        }
        if (strstr(line, "\"event\":\"done\"")) break;
        if (strstr(line, "\"ok\":false")) {
            failures++;
            break;
        }
    }
    client_close(&d);

    snprintf(request, sizeof request,
             "{\"op\":\"open\",\"session\":\"%s\"}\n", session_a);
    CHECK(client_call(&a, request, line, sizeof line));
    CHECK(strstr(line, "\"ok\":true") != NULL);
    CHECK(client_call(&a, "{\"op\":\"append\",\"role\":\"user\",\"text\":"
                          "\"Name one star.\"}\n", line, sizeof line));
    CHECK(strstr(line, "\"ok\":true") != NULL);
    CHECK(client_send(&a, "{\"op\":\"generate\",\"temperature\":1,"
                          "\"top_k\":1,\"max_tokens\":16,\"seed\":0}\n"));
    for (int i = 0; i < 4096; i++) {
        if (!client_recv(&a, line, sizeof line, 300000)) {
            failures++;
            break;
        }
        if (strstr(line, "\"event\":\"done\"")) break;
        if (strstr(line, "\"ok\":false")) {
            failures++;
            break;
        }
    }
    client_close(&a);
    client_close(&c);

    CHECK(wait_exit(server, 180) == 0);
    struct stat st;
    CHECK(stat(socket_path, &st) != 0);

    char lock_path[300];
    snprintf(lock_path, sizeof lock_path, "%s/xenolith/engine.lock",
             state_home);
    int lock_fd = open(lock_path, O_RDWR);
    CHECK(lock_fd >= 0);
    if (lock_fd >= 0) {
        CHECK(flock(lock_fd, LOCK_EX | LOCK_NB) == 0);
        close(lock_fd);
    }

    server = spawn_server(model, state_dir, cache_dir, socket_path, NULL,
                          log_path);
    CHECK(server > 0);
    client r;
    if (!client_connect(&r, socket_path, "R", 900)) {
        kill(server, SIGKILL);
        fprintf(stderr, "test_serve_model: restarted server never accepted\n");
        return 1;
    }
    CHECK(client_call(&r, request, line, sizeof line));
    CHECK(strstr(line, "\"ok\":true") != NULL);
    CHECK(strstr(line, "\"zero_prefill\":true") != NULL);
    client_close(&r);
    kill(server, SIGTERM);
    CHECK(wait_exit(server, 180) == 0);

    if (failures) {
        fprintf(stderr, "test_serve_model: %d failures (log %s)\n", failures,
                log_path);
        return 1;
    }
    printf("test_serve_model: all checks passed\n");
    return 0;
}
