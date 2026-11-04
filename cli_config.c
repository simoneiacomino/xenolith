#define _POSIX_C_SOURCE 200809L
#include "cli_config.h"
#include "json.h"

#include <errno.h>
#include <fcntl.h>
#include <inttypes.h>
#include <math.h>
#include <stdarg.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <sys/un.h>
#include <unistd.h>

enum { PATH_CAP = 4096, CONFIG_MAX = 65536 };
typedef struct {
    char path[PATH_CAP], model[PATH_CAP], mmproj[PATH_CAP];
    char socket[PATH_CAP], state[PATH_CAP], cache[PATH_CAP];
    char budget[32], context[16], idle[64];
    int present;
} native_config;

static int fail(char *error, size_t cap, const char *fmt, ...) {
    va_list ap;
    va_start(ap, fmt);
    vsnprintf(error, cap, fmt, ap);
    va_end(ap);
    return 0;
}

static int copy_path(char *out, size_t cap, const char *value,
                     const char *field, char *error, size_t error_cap) {
    if (!value || value[0] != '/')
        return fail(error, error_cap, "%s must be an absolute path", field);
    size_t n = strlen(value);
    if (n >= cap) return fail(error, error_cap, "%s path is too long", field);
    memcpy(out, value, n + 1);
    return 1;
}

static int join_path(char *out, size_t cap, const char *base, const char *tail) {
    if (!base || base[0] != '/') return 0;
    int n = snprintf(out, cap, "%s%s%s", base,
                     base[strlen(base) - 1] == '/' ? "" : "/", tail);
    return n > 0 && (size_t)n < cap;
}

static int xdg_path(char *out, size_t cap, const char *variable,
                    const char *suffix, const char *fallback) {
    const char *xdg = getenv(variable);
    return xdg && xdg[0] == '/' ? join_path(out, cap, xdg, suffix)
                                : join_path(out, cap, getenv("HOME"), fallback);
}

static int select_config(native_config *cfg, const char *override,
                         char *error, size_t cap) {
    const char *path = override ? override : getenv("XENOLITH_CONFIG");
    if (path) return copy_path(cfg->path, sizeof cfg->path, path,
                               "configuration", error, cap);
    if (xdg_path(cfg->path, sizeof cfg->path, "XDG_CONFIG_HOME",
                 "xenolith/config.json", ".config/xenolith/config.json")) return 1;
    return fail(error, cap, "set an absolute HOME, XDG_CONFIG_HOME or --config path");
}

static int unsigned_decimal(const char *text, uint64_t *out) {
    uint64_t value = 0;
    if (!text || !*text) return 0;
    for (const unsigned char *p = (const unsigned char *)text; *p; p++) {
        if (*p < '0' || *p > '9' || value > (UINT64_MAX - (*p - '0')) / 10) return 0;
        value = value * 10 + (*p - '0');
    }
    *out = value;
    return 1;
}

static int budget_bytes(const char *text, uint64_t *out) {
    if (!text) return 0;
    size_t n = strspn(text, "0123456789");
    if (!n || n >= 32) return 0;
    char number[32];
    memcpy(number, text, n); number[n] = 0;
    uint64_t value, scale = 0;
    if (!unsigned_decimal(number, &value)) return 0;
    const char *unit = text + n;
    const char *units[] = {"", "B", "kB", "MB", "GB", "KiB", "MiB", "GiB"};
    const uint64_t scales[] = {1, 1, 1000, 1000000, 1000000000, 1024, 1048576, 1073741824};
    for (size_t i = 0; i < sizeof scales / sizeof *scales; i++)
        if (!strcmp(unit, units[i])) scale = scales[i];
    if (!scale || value > UINT64_MAX / scale) return 0;
    *out = value * scale;
    return 1;
}

static int set_field(native_config *cfg, const char *key, const char *value,
                      int numeric, char *error, size_t cap) {
    char *path = NULL;
    if (!strcmp(key, "model")) path = cfg->model;
    else if (!strcmp(key, "mmproj")) path = cfg->mmproj;
    else if (!strcmp(key, "socket")) path = cfg->socket;
    else if (!strcmp(key, "state_dir")) path = cfg->state;
    else if (!strcmp(key, "cache_dir")) path = cfg->cache;
    if (path) {
        if (numeric) return fail(error, cap, "%s must be a path string", key);
        if (!copy_path(path, PATH_CAP, value, key, error, cap)) return 0;
        if (path == cfg->socket && strlen(path) >= sizeof(((struct sockaddr_un *)0)->sun_path))
            return fail(error, cap, "socket exceeds the Unix socket path limit");
        return 1;
    }
    if (!strcmp(key, "context")) {
        uint64_t n;
        if (!numeric || !unsigned_decimal(value, &n) || n < 64 || n > 262144)
            return fail(error, cap, "context must be an integer in [64, 262144]");
        snprintf(cfg->context, sizeof cfg->context, "%" PRIu64, n);
        return 1;
    }
    if (!strcmp(key, "cache_budget")) {
        uint64_t n;
        if (!budget_bytes(value, &n) || (numeric && strspn(value, "0123456789") != strlen(value)))
            return fail(error, cap, "cache_budget must be a uint64 byte count or size such as 4GiB");
        snprintf(cfg->budget, sizeof cfg->budget, "%" PRIu64, n);
        return 1;
    }
    if (!strcmp(key, "idle_shutdown_minutes")) {
        char *end = NULL;
        errno = 0;
        double n = value && *value ? strtod(value, &end) : 0;
        if (!numeric || !end || *end || errno || !isfinite(n) || n <= 0)
            return fail(error, cap, "idle_shutdown_minutes must be a positive finite number");
        snprintf(cfg->idle, sizeof cfg->idle, "%.17g", n);
        return 1;
    }
    return fail(error, cap, "unknown configuration field: %s", key);
}

static int read_config(native_config *cfg, char *error, size_t cap) {
    int fd = open(cfg->path, O_RDONLY | O_CLOEXEC | O_NONBLOCK);
    if (fd < 0) {
        if (errno == ENOENT) return 1;
        return fail(error, cap, "cannot read configuration %s: %s", cfg->path, strerror(errno));
    }
    struct stat st;
    if (fstat(fd, &st) || !S_ISREG(st.st_mode) || st.st_size < 0 || st.st_size > CONFIG_MAX) {
        close(fd);
        return fail(error, cap, "configuration must be a regular file of at most %d bytes: %s", CONFIG_MAX, cfg->path);
    }
    char *data = malloc(CONFIG_MAX + 1);
    if (!data) { close(fd); return fail(error, cap, "out of memory reading configuration"); }
    size_t n = 0;
    ssize_t got;
    while ((got = read(fd, data + n, CONFIG_MAX + 1 - n)) > 0) {
        n += (size_t)got;
        if (n > CONFIG_MAX) break;
    }
    int saved = errno;
    close(fd);
    if (got < 0 || n > CONFIG_MAX) {
        free(data);
        return fail(error, cap, "cannot read configuration %s: %s", cfg->path,
                    n > CONFIG_MAX ? "file too large" : strerror(saved));
    }
    json_value *root = json_parse(data, n);
    free(data);
    if (!root || root->type != JSON_OBJECT) {
        json_free(root);
        return fail(error, cap, "invalid JSON configuration: %s", cfg->path);
    }
    const json_value *version = json_member(root, "schema_version");
    uint64_t schema;
    if (!version || version->type != JSON_NUMBER ||
        !unsigned_decimal(version->text, &schema) || schema != 1) {
        json_free(root);
        return fail(error, cap, "configuration %s requires schema_version 1", cfg->path);
    }
    int ok = 1;
    for (const json_value *v = root->child; v && ok; v = v->next) {
        for (const json_value *prev = root->child; prev != v; prev = prev->next)
            if (!strcmp(prev->key, v->key)) {
                ok = fail(error, cap, "duplicate configuration field: %s", v->key);
                break;
            }
        if (!ok || !strcmp(v->key, "schema_version")) continue;
        if (v->type != JSON_STRING && v->type != JSON_NUMBER) {
            ok = fail(error, cap, "invalid type for configuration field: %s", v->key);
            break;
        }
        ok = set_field(cfg, v->key, v->text, v->type == JSON_NUMBER, error, cap);
    }
    json_free(root);
    if (ok) cfg->present = 1;
    else {
        char detail[512];
        snprintf(detail, sizeof detail, "%s", error);
        fail(error, cap, "configuration %s: %s", cfg->path, detail);
    }
    return ok;
}

static int writable_directory(const char *path) {
    struct stat st;
    return !stat(path, &st) && S_ISDIR(st.st_mode) && !access(path, W_OK | X_OK);
}

int cli_config_socket_path(const char *override, const char *state_dir,
                           char *out, size_t cap) {
    if (override) {
        size_t n = strlen(override);
        if (!n || n >= cap) return 0;
        memcpy(out, override, n + 1);
        return 1;
    }
    const char *runtime = getenv("XDG_RUNTIME_DIR");
    char dir[PATH_CAP];
    if (runtime && runtime[0] == '/' && writable_directory(runtime) &&
        join_path(dir, sizeof dir, runtime, "xenolith")) {
        struct stat st;
        int missing = stat(dir, &st) != 0 && errno == ENOENT;
        if (missing || writable_directory(dir))
            return join_path(out, cap, dir, "wire.sock");
    }
    return join_path(out, cap, state_dir, "wire.sock");
}

static int mkdir_parents(const char *path) {
    char copy[PATH_CAP];
    if (!path || path[0] != '/' || strlen(path) >= sizeof copy) { errno = EINVAL; return 0; }
    strcpy(copy, path);
    for (char *p = copy + 1; ; p++) {
        if (*p && *p != '/') continue;
        char saved = *p; *p = 0;
        if (mkdir(copy, 0700) && errno != EEXIST) return 0;
        struct stat st;
        if (stat(copy, &st) || !S_ISDIR(st.st_mode)) { errno = ENOTDIR; return 0; }
        *p = saved;
        if (!saved) return 1;
    }
}

int cli_config_socket_parent(const char *socket_path) {
    char parent[PATH_CAP];
    if (!socket_path || strlen(socket_path) >= sizeof parent) { errno = EINVAL; return 0; }
    strcpy(parent, socket_path);
    char *slash = strrchr(parent, '/');
    if (!slash) { errno = EINVAL; return 0; }
    if (slash == parent) return 1;
    *slash = 0;
    return mkdir_parents(parent);
}

static int resolve_paths(native_config *cfg, char *socket, char *lock,
                         char *log, char *error, size_t cap) {
    char global_state[PATH_CAP];
    if (!xdg_path(global_state, sizeof global_state, "XDG_STATE_HOME",
                   "xenolith", ".local/state/xenolith") ||
        !join_path(lock, PATH_CAP, global_state, "engine.lock"))
        return fail(error, cap, "cannot resolve the default engine lock directory");
    if (!cfg->state[0]) strcpy(cfg->state, global_state);
    if (!cfg->cache[0] && !xdg_path(cfg->cache, sizeof cfg->cache, "XDG_CACHE_HOME",
                                   "xenolith/kv", ".cache/xenolith/kv"))
        return fail(error, cap, "cannot resolve the default cache directory");
    if (!cli_config_socket_path(cfg->socket[0] ? cfg->socket : NULL, cfg->state, socket, PATH_CAP) ||
        strlen(socket) >= sizeof(((struct sockaddr_un *)0)->sun_path))
        return fail(error, cap, "resolved socket exceeds the Unix socket path limit; set --socket");
    if (!join_path(log, PATH_CAP, cfg->state, "service.log"))
        return fail(error, cap, "service log path is too long");
    return 1;
}

static void string_field(json_writer *w, const char *key, const char *value) {
    json_raw(w, ","); json_string(w, key, strlen(key)); json_raw(w, ":");
    json_string(w, value, strlen(value));
}

static int print_inspect(native_config *cfg, char *error, size_t cap) {
    char socket[PATH_CAP], lock[PATH_CAP], log[PATH_CAP];
    if (!resolve_paths(cfg, socket, lock, log, error, cap)) return 0;
    json_writer w = {0};
    json_raw(&w, "{\"schema_version\":1");
    string_field(&w, "config_path", cfg->path);
    json_raw(&w, cfg->present ? ",\"config_present\":true" : ",\"config_present\":false");
    json_raw(&w, ",\"endpoint\":{\"transport\":\"unix\"");
    string_field(&w, "path", socket); json_raw(&w, "}");
    string_field(&w, "engine_lock", lock);
    string_field(&w, "state_dir", cfg->state);
    string_field(&w, "cache_dir", cfg->cache);
    string_field(&w, "log_path", log);
    json_raw(&w, "}\n");
    int ok = !w.failed && fwrite(w.data, 1, w.length, stdout) == w.length && !fflush(stdout);
    json_writer_free(&w);
    return ok || fail(error, cap, "cannot write configuration discovery output");
}

static int write_config(native_config *cfg, char *error, size_t cap) {
    struct stat st;
    if (!lstat(cfg->path, &st) || errno != ENOENT)
        return fail(error, cap, "configuration already exists or cannot be inspected: %s; left unchanged", cfg->path);
    if (!cfg->model[0]) return fail(error, cap, "config init requires --model /absolute/model.gguf");
    if (cfg->mmproj[0] && cfg->context[0] && strcmp(cfg->context, "65536") && strcmp(cfg->context, "131072"))
        return fail(error, cap, "vision configuration requires --ctx 65536 or 131072");
    if (!cli_config_socket_parent(cfg->path))
        return fail(error, cap, "cannot create configuration directory: %s", strerror(errno));
    char temp[PATH_CAP];
    int n = snprintf(temp, sizeof temp, "%s.tmp.XXXXXX", cfg->path);
    if (n < 0 || (size_t)n >= sizeof temp) return fail(error, cap, "configuration path is too long");
    int fd = mkstemp(temp);
    if (fd < 0) return fail(error, cap, "cannot create temporary configuration: %s", strerror(errno));
    json_writer w = {0};
    json_raw(&w, "{\"schema_version\":1");
    string_field(&w, "model", cfg->model);
    if (cfg->mmproj[0]) string_field(&w, "mmproj", cfg->mmproj);
    if (cfg->socket[0]) string_field(&w, "socket", cfg->socket);
    if (cfg->state[0]) string_field(&w, "state_dir", cfg->state);
    if (cfg->cache[0]) string_field(&w, "cache_dir", cfg->cache);
    if (cfg->context[0]) { json_raw(&w, ",\"context\":"); json_raw(&w, cfg->context); }
    if (cfg->budget[0]) { json_raw(&w, ",\"cache_budget\":"); json_raw(&w, cfg->budget); }
    if (cfg->idle[0]) { json_raw(&w, ",\"idle_shutdown_minutes\":"); json_raw(&w, cfg->idle); }
    json_raw(&w, "}\n");
    int ok = !w.failed;
    size_t written = 0;
    while (ok && written < w.length) {
        ssize_t count = write(fd, w.data + written, w.length - written);
        if (count < 0 && errno == EINTR) continue;
        if (count <= 0) ok = 0; else written += (size_t)count;
    }
    if (ok && fsync(fd)) ok = 0;
    if (close(fd)) ok = 0;
    /* link publishes atomically and refuses to replace even a dangling symlink. */
    if (ok && link(temp, cfg->path)) ok = 0;
    int saved = errno;
    unlink(temp);
    json_writer_free(&w);
    if (!ok) return fail(error, cap, "cannot create configuration %s: %s; existing file left unchanged", cfg->path, strerror(saved));
    fprintf(stderr, "xenolith: created %s\n", cfg->path);
    return 1;
}

static const char *field_for_option(const char *option) {
    const char *options[] = {"--model", "--mmproj", "--ctx", "--socket", "--state", "--cache", "--cache-budget", "--idle-shutdown"};
    const char *fields[] = {"model", "mmproj", "context", "socket", "state_dir", "cache_dir", "cache_budget", "idle_shutdown_minutes"};
    for (size_t i = 0; i < sizeof options / sizeof *options; i++)
        if (!strcmp(option, options[i])) return fields[i];
    return NULL;
}

int cli_config_command(int argc, char **argv) {
    char error[8192] = {0};
    native_config cfg = {0};
    const char *path = NULL;
    int init = argc >= 1 && !strcmp(argv[0], "init");
    int inspect = argc >= 1 && !strcmp(argv[0], "inspect");
    int json = 0;
    if (!init && !inspect) {
        fprintf(stderr, "Usage: xenolith config init --model /path/model.gguf [options]\n"
                        "       xenolith config inspect --json [--config FILE] [--socket PATH] [--state DIR] [--cache DIR]\n");
        return 1;
    }
    /* Select the file before loading; option values are never rescanned. */
    for (int i = 1; i < argc; i++) {
        const char *option = argv[i];
        if (!strcmp(option, "--json")) { json++; continue; }
        if (++i >= argc) { fail(error, sizeof error, "%s requires a value", option); goto bad; }
        if (!strcmp(option, "--config")) {
            if (path) { fail(error, sizeof error, "--config may appear only once"); goto bad; }
            path = argv[i];
        } else if (!field_for_option(option)) {
            fail(error, sizeof error, "unknown config option: %s", option); goto bad;
        }
    }
    if ((inspect && json != 1) || (init && json)) {
        fail(error, sizeof error, "inspect requires --json once; init does not accept --json"); goto bad;
    }
    if (!select_config(&cfg, path, error, sizeof error)) goto bad;
    if (inspect && !read_config(&cfg, error, sizeof error)) goto bad;
    if (init) {
        struct stat st;
        if (!lstat(cfg.path, &st) || errno != ENOENT) {
            fail(error, sizeof error, "configuration already exists or cannot be inspected: %s; left unchanged", cfg.path);
            goto bad;
        }
    }
    unsigned seen = 0;
    for (int i = 1; i < argc; i++) {
        const char *option = argv[i];
        if (!strcmp(option, "--json")) continue;
        const char *value = argv[++i];
        if (!strcmp(option, "--config")) continue;
        const char *field = field_for_option(option);
        unsigned bit = !strcmp(field, "model") ? 1 : !strcmp(field, "mmproj") ? 2 :
                       !strcmp(field, "context") ? 4 : !strcmp(field, "socket") ? 8 :
                       !strcmp(field, "state_dir") ? 16 : !strcmp(field, "cache_dir") ? 32 :
                       !strcmp(field, "cache_budget") ? 64 : 128;
        if (seen & bit) { fail(error, sizeof error, "%s may appear only once", option); goto bad; }
        seen |= bit;
        if (inspect && bit != 8 && bit != 16 && bit != 32) {
            fail(error, sizeof error, "%s is not an operational inspect override", option); goto bad;
        }
        int numeric = bit == 4 || bit == 128;
        if (!set_field(&cfg, field, value, numeric, error, sizeof error)) goto bad;
    }
    if (init && !cfg.model[0]) {
        FILE *tty = fopen("/dev/tty", "r+");
        if (tty) {
            fprintf(tty, "Model GGUF (absolute path): "); fflush(tty);
            char line[PATH_CAP];
            if (fgets(line, sizeof line, tty)) {
                size_t len = strlen(line);
                if (len && line[len - 1] == '\n') line[len - 1] = 0;
                if (!set_field(&cfg, "model", line, 0, error, sizeof error)) { fclose(tty); goto bad; }
            }
            fclose(tty);
        }
    }
    if (init ? !write_config(&cfg, error, sizeof error) : !print_inspect(&cfg, error, sizeof error)) goto bad;
    return 0;
bad:
    fprintf(stderr, "xenolith: %s\n", error);
    return 1;
}

int cli_config_load(const char *path, cli_config *out,
                    char *error, size_t cap) {
    native_config cfg = {0};
    memset(out, 0, sizeof *out);
    if (!select_config(&cfg, path, error, cap) || !read_config(&cfg, error, cap))
        return 0;
    memcpy(out->model, cfg.model, sizeof out->model);
    memcpy(out->mmproj, cfg.mmproj, sizeof out->mmproj);
    memcpy(out->socket, cfg.socket, sizeof out->socket);
    memcpy(out->state_dir, cfg.state, sizeof out->state_dir);
    memcpy(out->cache_dir, cfg.cache, sizeof out->cache_dir);
    /* These numeric strings have already been validated and normalized. */
    if (cfg.context[0]) out->context = (int)strtol(cfg.context, NULL, 10);
    if (cfg.budget[0]) {
        out->cache_budget = strtoull(cfg.budget, NULL, 10);
        out->has_cache_budget = 1;
    }
    if (cfg.idle[0]) out->idle_shutdown_minutes = strtod(cfg.idle, NULL);
    return 1;
}
