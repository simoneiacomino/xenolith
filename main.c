#define _POSIX_C_SOURCE 200809L

#include "xenolith.h"
#include "profile.h"
#include "conversation.h"
#include "serve.h"

#include <errno.h>
#include <inttypes.h>
#include <limits.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <unistd.h>
#include <sys/ioctl.h>

static void usage(void) {
    fprintf(stderr,
"Usage: xenolith <command> <model.gguf> [options]\n"
"  --ctx N  context capacity for inference commands (64..262144; default 262144)\n"
"  info     print model metadata\n"
"  tokenize -p \"text\" | -f file [--pieces]\n"
"  run      -p \"prompt\" [-n max] [--temp F] [--top-k N] [--top-p F] [--seed S]\n"
"  chat     [-n max] [--temp F] [--top-k N] [--top-p F] [--seed S]\n"
"  oracle   -t \"id,id,id\" [-o logits.bin] [--layers dir] [--q8]\n"
"  bench    [-p N] [-n N] [-d N] [-r N] [--delay S] [--no-warmup] [--progress] [-o md|jsonl]\n"
"  wire     [--state DIR] [--cache DIR]\n"
"  serve    [--state DIR] [--cache DIR] [--socket PATH] [--idle-shutdown MIN]\n");
    exit(1);
}

static int cli_lock_fd = -1;

static void cli_state_dir(const char *override, char *out, size_t cap) {
    if (override) {
        int n = snprintf(out, cap, "%s", override);
        if (n > 0 && (size_t)n < cap) return;
        fprintf(stderr, "xenolith: state directory path is too long\n");
        exit(1);
    }
    if (conversation_default_state_dir(out, cap)) return;
    fprintf(stderr, "xenolith: cannot resolve the default state directory\n");
    exit(1);
}

static void cli_lock(const char *socket_path) {
    char holder[512];
    int fd = serve_lock(socket_path, holder, sizeof holder);
    if (fd < 0) {
        char path[4096];
        if (!serve_lock_path(path, sizeof path))
            snprintf(path, sizeof path, "engine.lock");
        if (holder[0])
            fprintf(stderr,
                    "xenolith: another engine already owns the weights "
                    "(%s held by %s)\n", path, holder);
        else
            fprintf(stderr, "xenolith: cannot acquire %s\n", path);
        exit(1);
    }
    cli_lock_fd = fd;
}

static void cli_unlock(void) {
    if (cli_lock_fd >= 0) close(cli_lock_fd);
    cli_lock_fd = -1;
}

/* Each command parses its own options so option values remain literal. */
static int cli_context_option(int argc, char **argv, int *index,
                               int *context, int *seen) {
    if (strcmp(argv[*index], "--ctx")) return 0;
    if (*seen || ++*index >= argc) {
        fprintf(stderr, "xenolith: --ctx requires one capacity and may appear only once\n");
        exit(1);
    }
    const char *value = argv[*index];
    int digits = value[0] != '\0';
    for (const char *p = value; *p; p++)
        if (*p < '0' || *p > '9') digits = 0;
    errno = 0;
    char *end;
    long n = strtol(value, &end, 10);
    if (!digits || errno || *end || n < XE_CONTEXT_MIN || n > XE_CONTEXT_MAX) {
        fprintf(stderr, "xenolith: --ctx must be an integer in [%d, %d]\n",
                XE_CONTEXT_MIN, XE_CONTEXT_MAX);
        exit(1);
    }
    *context = (int)n;
    *seen = 1;
    return 1;
}

static int cmd_info(const char *model, int argc, char **argv) {
    int context = XE_CONTEXT_DEFAULT, context_seen = 0;
    for (int i = 3; i < argc; i++)
        if (!cli_context_option(argc, argv, &i, &context, &context_seen)) usage();
    cli_lock(NULL);
    xe_engine *e = xe_engine_open_with_context(model, context);
    xe_engine_info(e, stdout);
    xe_engine_close(e);
    cli_unlock();
    return 0;
}

static char *read_file(const char *path, size_t *len_out) {
    FILE *f = fopen(path, "rb");
    if (!f) {
        fprintf(stderr, "xenolith: cannot open %s\n", path);
        exit(1);
    }
    if (fseek(f, 0, SEEK_END) != 0) {
        fprintf(stderr, "xenolith: cannot seek %s\n", path);
        exit(1);
    }
    long n = ftell(f);
    if (n < 0 || fseek(f, 0, SEEK_SET) != 0) {
        fprintf(stderr, "xenolith: cannot seek %s\n", path);
        exit(1);
    }
    char *buf = malloc((size_t)n + 1);
    if (!buf) {
        fprintf(stderr, "xenolith: out of memory reading %s\n", path);
        exit(1);
    }
    if (fread(buf, 1, (size_t)n, f) != (size_t)n) {
        fprintf(stderr, "xenolith: short read on %s\n", path);
        exit(1);
    }
    fclose(f);
    buf[n] = '\0';
    *len_out = (size_t)n;
    return buf;
}

static int cmd_detokenize(const char *model) {
    xe_engine *e = xe_engine_open_vocab(model);
    char buf[512];
    long v;
    int c;

    for (;;) {
        do { c = getchar(); } while (c == ',' || (c >= 0 && c <= ' '));
        if (c == EOF) break;
        ungetc(c, stdin);
        if (scanf("%ld", &v) != 1) {
            fprintf(stderr, "xenolith: detok: bad token list\n");
            exit(1);
        }
        if (v < 0 || v > 262143) {
            fprintf(stderr, "xenolith: detok: token id %ld out of range [0, 262143]\n", v);
            exit(1);
        }
        int n = xe_detokenize(e, (int32_t)v, buf, (int)sizeof buf);
        fwrite(buf, 1, (size_t)n, stdout);
    }

    fflush(stdout);
    xe_engine_close(e);
    return 0;
}

static int cmd_tokenize(const char *model, int argc, char **argv) {
    const char *prompt = NULL;
    const char *file = NULL;
    int pieces = 0;

    for (int i = 3; i < argc; i++) {
        if (!strcmp(argv[i], "-p")) {
            if (++i >= argc) usage();
            prompt = argv[i];
        } else if (!strcmp(argv[i], "-f")) {
            if (++i >= argc) usage();
            file = argv[i];
        } else if (!strcmp(argv[i], "--pieces")) {
            pieces = 1;
        } else if (!strcmp(argv[i], "--detok")) {
            return cmd_detokenize(model);
        } else {
            usage();
        }
    }
    if ((!prompt && !file) || (prompt && file)) usage();

    char *text;
    size_t len;
    if (file) {
        text = read_file(file, &len);
    } else {
        len = strlen(prompt);
        text = malloc(len + 1);
        if (!text) {
            fprintf(stderr, "xenolith: out of memory\n");
            exit(1);
        }
        memcpy(text, prompt, len + 1);
    }

    xe_engine *e = xe_engine_open_vocab(model);

    if (len > (SIZE_MAX - 1) / 3 || len > (size_t)(INT_MAX - 1) / 3) {
        fprintf(stderr, "xenolith: input is too large\n");
        exit(1);
    }
    int cap = (int)(3 * len + 1);
    int32_t *toks = malloc((size_t)cap * sizeof *toks);
    if (!toks) {
        fprintf(stderr, "xenolith: out of memory\n");
        exit(1);
    }
    toks[0] = xe_bos_id(e);
    int n = 1 + xe_encode_text(e, text, toks + 1, cap - 1);

    if (pieces) {
        for (int i = 0; i < n; i++) {
            int plen;
            const char *p = xe_token_piece(e, toks[i], &plen);
            printf("%d\t", toks[i]);
            fwrite(p, 1, (size_t)plen, stdout);
            putchar('\n');
        }
    } else {
        for (int i = 0; i < n; i++) printf("%s%d", i ? "," : "", toks[i]);
        printf("\n");
    }
    fflush(stdout);
    fprintf(stderr, "n_tokens %d\n", n);

    free(toks);
    free(text);
    xe_engine_close(e);
    return 0;
}

static int cmd_run(const char *model, int argc, char **argv) {
    int context = XE_CONTEXT_DEFAULT, context_seen = 0;
    const char *prompt = NULL;
    int max_n = 512;
    float temp = 1.0f;
    int top_k = 64;
    float top_p = 0.95f;
    uint64_t seed = (uint64_t)time(NULL);

    for (int i = 3; i < argc; i++) {
        if (cli_context_option(argc, argv, &i, &context, &context_seen)) continue;
        if (!strcmp(argv[i], "-p")) {
            if (++i >= argc) usage();
            prompt = argv[i];
        } else if (!strcmp(argv[i], "-n")) {
            if (++i >= argc) usage();
            errno = 0;
            char *end;
            long value = strtol(argv[i], &end, 10);
            if (errno || *end || value < 0 || value > INT_MAX) usage();
            max_n = (int)value;
        } else if (!strcmp(argv[i], "--temp")) {
            if (++i >= argc) usage();
            errno = 0;
            char *end;
            float value = strtof(argv[i], &end);
            if (errno || *end || !isfinite(value) || value < 0.0f) usage();
            temp = value;
        } else if (!strcmp(argv[i], "--top-k")) {
            if (++i >= argc) usage();
            errno = 0;
            char *end;
            long value = strtol(argv[i], &end, 10);
            if (errno || *end || value < 0 || value > INT_MAX) usage();
            top_k = (int)value;
        } else if (!strcmp(argv[i], "--top-p")) {
            if (++i >= argc) usage();
            errno = 0;
            char *end;
            float value = strtof(argv[i], &end);
            if (errno || *end || !isfinite(value) || value <= 0.0f || value > 1.0f) usage();
            top_p = value;
        } else if (!strcmp(argv[i], "--seed")) {
            if (++i >= argc) usage();
            if (argv[i][0] == '-') usage();
            errno = 0;
            char *end;
            unsigned long long value = strtoull(argv[i], &end, 10);
            if (errno || *end) usage();
            seed = (uint64_t)value;
        } else {
            usage();
        }
    }
    if (!prompt) usage();

    cli_lock(NULL);

    xe_engine *e = xe_engine_open_with_context(model, context);
    int ctx = xe_context_size(e);
    if (top_k > ctx) {
        fprintf(stderr, "xenolith: run: top-k %d out of range [0, %d]\n", top_k, ctx);
        exit(1);
    }
    if (strlen(prompt) + 1 > (size_t)ctx) {
        fprintf(stderr, "xenolith: run: prompt exceeds context capacity\n");
        exit(1);
    }
    int32_t *toks = malloc((size_t)ctx * sizeof(*toks));
    if (!toks) {
        fprintf(stderr, "xenolith: out of memory\n");
        exit(1);
    }
    toks[0] = xe_bos_id(e);
    int n = 1 + xe_encode_text(e, prompt, toks + 1, ctx - 1);
    if (max_n > 0 && n >= ctx) {
        fprintf(stderr, "xenolith: run: prompt leaves no room for generation\n");
        exit(1);
    }

    xe_tokens tokens = { toks, n, ctx };
    xe_session *s = xe_session_new(e);

    int32_t eot = xe_eot_id(e);
    int32_t eos = xe_eos_id(e);

    xe_sampler sp = { temp, top_k, top_p, seed };

    for (int i = 0; i < max_n && tokens.len < ctx; i++) {
        xe_session_sync(s, &tokens);
        int32_t t = xe_session_next(s, &sp);
        if (t == eot || t == eos) break;
        char buf[256];
        int nb = xe_detokenize(e, t, buf, sizeof buf);
        fwrite(buf, 1, (size_t)nb, stdout);
        fflush(stdout);
        xe_tokens_push(&tokens, t);
    }
    putchar('\n');

    xe_tokens_free(&tokens);
    xe_session_free(s);
    xe_engine_close(e);
    cli_unlock();
    return 0;
}

static uint64_t bench_now_ns(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (uint64_t)ts.tv_sec * UINT64_C(1000000000) + (uint64_t)ts.tv_nsec;
}

static int term_width(void) {
    struct winsize ws;
    if (ioctl(fileno(stdout), TIOCGWINSZ, &ws) == 0 && ws.ws_col > 0)
        return ws.ws_col;
    return 80;
}

static void chat_put_token(const char *buf, int nb, int *col, int term_cols) {
    for (int b = 0; b < nb; b++) {
        char c = buf[b];
        if (c == '\n') {
            putchar('\n');
            *col = 0;
        } else {
            int is_codepoint = (c & 0xc0) != 0x80;
            if (is_codepoint && *col >= term_cols) {
                putchar('\n');
                *col = 0;
            }
            putchar(c);
            if (is_codepoint) (*col)++;
        }
    }
}

static void chat_print_stats_streaming(int n_gen, double cur_tps, double avg_tps) {
    printf("\n[%d tokens | %.1f tps | avg %.1f tps]", n_gen, cur_tps, avg_tps);
}

static void chat_print_stats(int n_gen, double avg_tps) {
    printf("\n[%d tokens | avg %.1f tps]", n_gen, avg_tps);
}

static int cmd_chat(const char *model, int argc, char **argv) {
    int context = XE_CONTEXT_DEFAULT, context_seen = 0;
    int max_n = 512;
    float temp = 1.0f;
    int top_k = 64;
    float top_p = 0.95f;
    uint64_t seed = (uint64_t)time(NULL);

    for (int i = 3; i < argc; i++) {
        if (cli_context_option(argc, argv, &i, &context, &context_seen)) continue;
        if (!strcmp(argv[i], "-n")) {
            if (++i >= argc) usage();
            errno = 0;
            char *end;
            long value = strtol(argv[i], &end, 10);
            if (errno || *end || value < 0 || value > INT_MAX) usage();
            max_n = (int)value;
        } else if (!strcmp(argv[i], "--temp")) {
            if (++i >= argc) usage();
            errno = 0;
            char *end;
            float value = strtof(argv[i], &end);
            if (errno || *end || !isfinite(value) || value < 0.0f) usage();
            temp = value;
        } else if (!strcmp(argv[i], "--top-k")) {
            if (++i >= argc) usage();
            errno = 0;
            char *end;
            long value = strtol(argv[i], &end, 10);
            if (errno || *end || value < 0 || value > INT_MAX) usage();
            top_k = (int)value;
        } else if (!strcmp(argv[i], "--top-p")) {
            if (++i >= argc) usage();
            errno = 0;
            char *end;
            float value = strtof(argv[i], &end);
            if (errno || *end || !isfinite(value) || value <= 0.0f || value > 1.0f) usage();
            top_p = value;
        } else if (!strcmp(argv[i], "--seed")) {
            if (++i >= argc || argv[i][0] == '-') usage();
            errno = 0;
            char *end;
            unsigned long long value = strtoull(argv[i], &end, 10);
            if (errno || *end) usage();
            seed = (uint64_t)value;
        } else {
            usage();
        }
    }

    cli_lock(NULL);

    xe_engine *e = xe_engine_open_with_context(model, context);
    int ctx = xe_context_size(e);
    if (top_k > ctx) {
        fprintf(stderr, "xenolith: chat: top-k %d out of range [0, %d]\n", top_k, ctx);
        exit(1);
    }

    profile *pf = NULL;
    if (profile_open(&pf, e) != PROFILE_OK) {
        fprintf(stderr, "xenolith: chat: model profile mismatch\n");
        exit(1);
    }

    xe_session *s = xe_session_new(e);
    xe_tokens transcript = {0};
    xe_sampler sp = { temp, top_k, top_p, seed };
    int32_t eot = xe_eot_id(e);
    int32_t eos = xe_eos_id(e);
    char *line = NULL;
    size_t line_cap = 0;
    int stop = 0;
    uint32_t turn = PROFILE_TURN_PADDED;
    profile_render render;

    if (profile_render_system(pf, NULL, NULL, 0, 0, &render) != PROFILE_OK) {
        fprintf(stderr, "xenolith: chat: render failed\n");
        exit(1);
    }
    for (uint32_t i = 0; i < render.token_count; i++)
        xe_tokens_push(&transcript, render.tokens[i]);

    while (!stop) {
        fputs("you> ", stdout);
        fflush(stdout);
        ssize_t nread = getline(&line, &line_cap, stdin);
        if (nread < 0) break;
        while (nread > 0 && (line[nread - 1] == '\n' || line[nread - 1] == '\r'))
            line[--nread] = '\0';
        if (!strcmp(line, "/exit")) break;
        if (profile_render_user(pf, line, turn, &render) != PROFILE_OK) {
            fprintf(stderr, "xenolith: chat: render failed\n");
            break;
        }
        if (transcript.len + (int)render.token_count >= ctx - 1) {
            fprintf(stderr, "xenolith: chat: conversation exceeds context capacity\n");
            break;
        }
        for (uint32_t i = 0; i < render.token_count; i++)
            xe_tokens_push(&transcript, render.tokens[i]);
        if (profile_render_reply_open(pf, PROFILE_TURN_PADDED, 0, 0,
                                      &render)
                != PROFILE_OK ||
            transcript.len + (int)render.token_count >= ctx - 1) {
            fprintf(stderr, "xenolith: chat: conversation exceeds context capacity\n");
            break;
        }
        for (uint32_t i = 0; i < render.token_count; i++)
            xe_tokens_push(&transcript, render.tokens[i]);
        turn = PROFILE_TURN_OPEN;

        fputs("assistant> ", stdout);
        fflush(stdout);
        uint64_t t_start = 0, t_prev = 0;
        int n_gen = 0, col = 11, live = 0;
        int tty = isatty(fileno(stdout));
        int term_cols = tty ? term_width() : INT_MAX;
        for (int i = 0; i < max_n && transcript.len < ctx - 1; i++) {
            xe_session_sync(s, &transcript);
            if (i == 0) {
                t_start = bench_now_ns();
                t_prev = t_start;
            }
            int32_t t = xe_session_next(s, &sp);
            if (t == eos) {
                stop = 1;
                break;
            }
            xe_tokens_push(&transcript, t);
            if (t == eot) {
                turn = PROFILE_TURN_BARE;
                break;
            }
            char buf[256];
            int nb = xe_detokenize(e, t, buf, sizeof buf);
            if (nb > 0) {
                if (tty && live) {
                    if (col > 0) printf("\r\033[J\033[A\r\033[%dC", col);
                    else printf("\r\033[J\033[A\r");
                }
                if (tty) {
                    chat_put_token(buf, nb, &col, term_cols);
                } else {
                    fwrite(buf, 1, (size_t)nb, stdout);
                }
                n_gen++;
                uint64_t now = bench_now_ns();
                double cur_tps = (now > t_prev) ? 1e9 / (double)(now - t_prev) : 0.0;
                double avg_tps = (now > t_start) ? 1e9 * (double)n_gen / (double)(now - t_start) : 0.0;
                t_prev = now;
                if (tty) {
                    chat_print_stats_streaming(n_gen, cur_tps, avg_tps);
                    fflush(stdout);
                    live = 1;
                } else {
                    fflush(stdout);
                }
            }
        }
        if (!tty && n_gen > 0) {
            uint64_t now = bench_now_ns();
            double avg_tps = (now > t_start) ? 1e9 * (double)n_gen / (double)(now - t_start) : 0.0;
            chat_print_stats(n_gen, avg_tps);
        }
        putchar('\n');
    }

    free(line);
    xe_tokens_free(&transcript);
    xe_session_free(s);
    profile_close(pf);
    xe_engine_close(e);
    cli_unlock();
    return 0;
}

static int cmd_oracle(const char *model, int argc, char **argv) {
    int context = XE_CONTEXT_DEFAULT, context_seen = 0;
    const char *ids = NULL;
    const char *dump_path = NULL;
    const char *layers_dir = NULL;
    int q8_mode = 0;

    for (int i = 3; i < argc; i++) {
        if (cli_context_option(argc, argv, &i, &context, &context_seen)) continue;
        if (!strcmp(argv[i], "-t")) {
            if (++i >= argc) usage();
            ids = argv[i];
        } else if (!strcmp(argv[i], "-o")) {
            if (++i >= argc) usage();
            dump_path = argv[i];
        } else if (!strcmp(argv[i], "--layers")) {
            if (++i >= argc) usage();
            layers_dir = argv[i];
        } else if (!strcmp(argv[i], "--q8")) {
            q8_mode = 1;
        } else {
            usage();
        }
    }
    if (!ids) usage();

    static int32_t toks[8192];
    int n = 0;
    const char *p = ids;

    while (*p) {
        char *end;
        long v = strtol(p, &end, 10);
        if (end == p) {
            fprintf(stderr, "xenolith: oracle: bad token list \"%s\"\n", ids);
            exit(1);
        }
        if (v < 0 || v > 262143) {
            fprintf(stderr, "xenolith: oracle: token id %ld out of range [0, 262143]\n", v);
            exit(1);
        }
        if (n >= 8192) {
            fprintf(stderr, "xenolith: oracle: too many token ids (max 8192)\n");
            exit(1);
        }
        toks[n++] = (int32_t)v;
        p = end;
        if (*p == ',') {
            p++;
        } else if (*p) {
            fprintf(stderr, "xenolith: oracle: bad token list \"%s\"\n", ids);
            exit(1);
        }
    }
    if (n == 0) usage();

    cli_lock(NULL);

    xe_engine *e = xe_engine_open_with_context(model, context);
    xe_oracle(e, toks, n, dump_path, layers_dir, q8_mode, stdout);
    xe_engine_close(e);
    cli_unlock();
    return 0;
}

typedef struct {
    int n_prompt;
    int n_gen;
    int n_depth;
    int reps;
    int delay;
    int warmup;
    int progress;
    int jsonl;
} bench_params;

typedef struct {
    int n_prompt;
    int n_gen;
    int n_depth;
    const char *backend;
    char test_time[32];
    uint64_t *samples_ns;
    int reps;
} bench_result;

static int bench_int(const char *arg, int min, int max) {
    errno = 0;
    char *end;
    long value = strtol(arg, &end, 10);
    if (errno || *end || value < min || value > max) usage();
    return (int)value;
}

static void bench_time(char out[32]) {
    time_t now = time(NULL);
    struct tm tm;
    gmtime_r(&now, &tm);
    strftime(out, 32, "%FT%TZ", &tm);
}

static void bench_prompt(xe_engine *e, xe_session *s, int32_t *tokens,
                         int offset, int n) {
    int first = 0;
    if (!offset) {
        tokens[0] = xe_bos_id(e);
        first = 1;
    }
    for (int i = first; i < n; i++) tokens[offset + i] = rand() % 262144;
    xe_tokens prefix = { tokens, offset + n, offset + n };
    xe_session_sync(s, &prefix);
}

static void bench_gen(xe_engine *e, xe_session *s, int32_t *tokens,
                      int offset, int n) {
    xe_tokens prefix = { tokens, offset, offset + n };
    int32_t token = offset ? rand() % 262144 : xe_bos_id(e);
    for (int i = 0; i < n; i++) {
        tokens[offset + i] = token;
        prefix.len = offset + i + 1;
        xe_session_sync(s, &prefix);
        token = rand() % 262144;
    }
}

static bench_result bench_run(xe_engine *e, const bench_params *params,
                              int n_prompt, int n_gen, const char *backend) {
    bench_result result = { n_prompt, n_gen, params->n_depth, backend,
                            {0}, NULL, params->reps };
    int measured = n_prompt ? n_prompt : n_gen;
    int total = params->n_depth + measured;
    int32_t *tokens = malloc((size_t)total * sizeof(*tokens));
    result.samples_ns = malloc((size_t)params->reps * sizeof(*result.samples_ns));
    if (!tokens || !result.samples_ns) {
        fprintf(stderr, "xenolith: bench: out of memory\n");
        exit(1);
    }

    xe_session *s = xe_session_new(e);
    if (params->delay) sleep((unsigned)params->delay);
    if (params->warmup) {
        if (params->progress)
            fprintf(stderr, "xenolith: bench: %s%d warmup\n",
                    n_prompt ? "pp" : "tg", measured);
        xe_session_reset(s);
        if (n_prompt)
            bench_prompt(e, s, tokens, 0, measured);
        else
            bench_gen(e, s, tokens, 0, 1);
    }

    bench_time(result.test_time);
    for (int rep = 0; rep < params->reps; rep++) {
        if (params->progress)
            fprintf(stderr, "xenolith: bench: %s%d depth %d run %d/%d\n",
                    n_prompt ? "pp" : "tg", measured, params->n_depth,
                    rep + 1, params->reps);
        xe_session_reset(s);
        if (params->n_depth)
            bench_prompt(e, s, tokens, 0, params->n_depth);
        uint64_t start = bench_now_ns();
        if (n_prompt)
            bench_prompt(e, s, tokens, params->n_depth, measured);
        else
            bench_gen(e, s, tokens, params->n_depth, measured);
        result.samples_ns[rep] = bench_now_ns() - start;
    }

    xe_session_free(s);
    free(tokens);
    return result;
}

static double bench_sample_ts(const bench_result *result, int rep) {
    int n = result->n_prompt + result->n_gen;
    return 1e9 * (double)n / (double)result->samples_ns[rep];
}

static double bench_avg_ns(const bench_result *result) {
    double sum = 0.0;
    for (int i = 0; i < result->reps; i++) sum += (double)result->samples_ns[i];
    return sum / result->reps;
}

static double bench_avg_ts(const bench_result *result) {
    double sum = 0.0;
    for (int i = 0; i < result->reps; i++) sum += bench_sample_ts(result, i);
    return sum / result->reps;
}

static double bench_stddev_ns(const bench_result *result) {
    if (result->reps <= 1) return 0.0;
    double mean = bench_avg_ns(result);
    double sum = 0.0;
    for (int i = 0; i < result->reps; i++) {
        double d = (double)result->samples_ns[i] - mean;
        sum += d * d;
    }
    return sqrt(sum / (result->reps - 1));
}

static double bench_stddev_ts(const bench_result *result) {
    if (result->reps <= 1) return 0.0;
    double mean = bench_avg_ts(result);
    double sum = 0.0;
    for (int i = 0; i < result->reps; i++) {
        double d = bench_sample_ts(result, i) - mean;
        sum += d * d;
    }
    return sqrt(sum / (result->reps - 1));
}

static void bench_json_string(const char *value) {
    putchar('"');
    for (const unsigned char *p = (const unsigned char *)value; *p; p++) {
        if (*p == '"' || *p == '\\') {
            putchar('\\');
            putchar(*p);
        } else if (*p <= 0x1f) {
            printf("\\u%04x", *p);
        } else {
            putchar(*p);
        }
    }
    putchar('"');
}

static void bench_cpu_mask(const xe_engine *e, char *out, size_t cap) {
    size_t used = 0;
    out[0] = '\0';
    if (xe_engine_worker_cpu(e, 0) < 0) {
        snprintf(out, cap, "unbound");
        return;
    }
    int workers = xe_engine_worker_count(e);
    for (int i = 0; i < workers; i++) {
        int n = snprintf(out + used, cap - used, "%s%d", i ? "," : "",
                         xe_engine_worker_cpu(e, i));
        if (n < 0 || (size_t)n >= cap - used) {
            fprintf(stderr, "xenolith: bench: worker mask overflow\n");
            exit(1);
        }
        used += (size_t)n;
    }
}

static void bench_print_jsonl(const char *model, const xe_engine *e,
                              const bench_result *result) {
    char mask[128];
    bench_cpu_mask(e, mask, sizeof mask);
    printf("{\"engine\":\"xenolith\",\"build_commit\":\"%s\",\"build_number\":0,\"model_filename\":",
           XE_BUILD_COMMIT);
    bench_json_string(model);
    const char *backends = !strcmp(result->backend, "CPU")
                           || !strcmp(result->backend, "CPU serial")
                           ? "CPU" : "Level Zero,CPU";
    printf(",\"model_size\":%" PRIu64 ",\"backends\":",
           xe_engine_model_size(e));
    bench_json_string(backends);
    printf(",\"implementation\":");
    bench_json_string(result->backend);
    printf(",\"n_threads\":%d,\"cpu_mask\":", xe_engine_worker_count(e));
    bench_json_string(mask);
    printf(",\"n_prompt\":%d,\"n_gen\":%d,\"n_depth\":%d,\"test_time\":",
           result->n_prompt, result->n_gen, result->n_depth);
    bench_json_string(result->test_time);
    printf(",\"avg_ns\":%.0f,\"stddev_ns\":%.0f,\"avg_ts\":%.6f,\"stddev_ts\":%.6f,\"samples_ns\":[",
           bench_avg_ns(result), bench_stddev_ns(result),
           bench_avg_ts(result), bench_stddev_ts(result));
    for (int i = 0; i < result->reps; i++)
        printf("%s%" PRIu64, i ? "," : "", result->samples_ns[i]);
    printf("],\"samples_ts\":[");
    for (int i = 0; i < result->reps; i++)
        printf("%s%.6f", i ? "," : "", bench_sample_ts(result, i));
    printf("]}\n");
    fflush(stdout);
}

static void bench_print_md(const char *model, const xe_engine *e,
                           const bench_result *result) {
    const char *base = strrchr(model, '/');
    base = base ? base + 1 : model;
    char test[64];
    if (result->n_depth)
        snprintf(test, sizeof test, "%s%d@d%d",
                 result->n_prompt ? "pp" : "tg",
                 result->n_prompt + result->n_gen, result->n_depth);
    else
        snprintf(test, sizeof test, "%s%d",
                 result->n_prompt ? "pp" : "tg",
                 result->n_prompt + result->n_gen);
    printf("| %-42s | %-10s | %7d | %8s | %9.2f ± %-9.2f |\n",
           base, result->backend, xe_engine_worker_count(e), test,
           bench_avg_ts(result), bench_stddev_ts(result));
    fflush(stdout);
}

static int cmd_bench(const char *model, int argc, char **argv) {
    int context = XE_CONTEXT_DEFAULT, context_seen = 0;
    bench_params params = { 512, 128, 0, 5, 0, 1, 0, 0 };
    for (int i = 3; i < argc; i++) {
        if (cli_context_option(argc, argv, &i, &context, &context_seen)) continue;
        if (!strcmp(argv[i], "-p") || !strcmp(argv[i], "--n-prompt")) {
            if (++i >= argc) usage();
            params.n_prompt = bench_int(argv[i], 0, INT_MAX);
        } else if (!strcmp(argv[i], "-n") || !strcmp(argv[i], "--n-gen")) {
            if (++i >= argc) usage();
            params.n_gen = bench_int(argv[i], 0, INT_MAX);
        } else if (!strcmp(argv[i], "-d") || !strcmp(argv[i], "--n-depth")) {
            if (++i >= argc) usage();
            params.n_depth = bench_int(argv[i], 0, INT_MAX);
        } else if (!strcmp(argv[i], "-r") || !strcmp(argv[i], "--repetitions")) {
            if (++i >= argc) usage();
            params.reps = bench_int(argv[i], 1, 1000);
        } else if (!strcmp(argv[i], "--delay")) {
            if (++i >= argc) usage();
            params.delay = bench_int(argv[i], 0, 3600);
        } else if (!strcmp(argv[i], "--no-warmup")) {
            params.warmup = 0;
        } else if (!strcmp(argv[i], "--progress")) {
            params.progress = 1;
        } else if (!strcmp(argv[i], "-o") || !strcmp(argv[i], "--output")) {
            if (++i >= argc) usage();
            if (!strcmp(argv[i], "md"))
                params.jsonl = 0;
            else if (!strcmp(argv[i], "jsonl"))
                params.jsonl = 1;
            else
                usage();
        } else {
            usage();
        }
    }
    if (params.n_prompt == 0 && params.n_gen == 0) usage();

    cli_lock(NULL);

    xe_engine *e = xe_engine_open_with_context(model, context);
    int ctx = xe_context_size(e);
    if (params.n_depth > ctx || params.n_prompt > ctx - params.n_depth ||
        params.n_gen > ctx - params.n_depth) {
        fprintf(stderr, "xenolith: bench: token count exceeds context capacity %d\n", ctx);
        exit(1);
    }

    srand(1);
    if (!params.jsonl) {
        printf("| %-42s | %-10s | %7s | %8s | %21s |\n",
               "model", "backend", "threads", "test", "t/s");
        printf("|--------------------------------------------|------------|--------:|---------:|----------------------:|\n");
    }

    if (params.n_prompt) {
        bench_result result = bench_run(e, &params, params.n_prompt, 0,
                                        "Level Zero+CPU");
        if (params.jsonl)
            bench_print_jsonl(model, e, &result);
        else
            bench_print_md(model, e, &result);
        free(result.samples_ns);
    }
    if (params.n_gen) {
        bench_result result = bench_run(e, &params, 0, params.n_gen, "CPU");
        if (params.jsonl)
            bench_print_jsonl(model, e, &result);
        else
            bench_print_md(model, e, &result);
        free(result.samples_ns);
    }

    if (!params.jsonl) printf("\nbuild: xenolith\n");
    xe_engine_close(e);
    cli_unlock();
    return 0;
}

static int cmd_wire(const char *model, int argc, char **argv) {
    int context = XE_CONTEXT_DEFAULT, context_seen = 0;
    const char *state_option = NULL;
    const char *cache_dir = NULL;
    for (int i = 3; i < argc; i++) {
        if (cli_context_option(argc, argv, &i, &context, &context_seen)) continue;
        if (!strcmp(argv[i], "--state")) {
            if (++i >= argc) usage();
            state_option = argv[i];
        } else if (!strcmp(argv[i], "--cache")) {
            if (++i >= argc) usage();
            cache_dir = argv[i];
        } else {
            usage();
        }
    }
    char state_dir[4096];
    cli_state_dir(state_option, state_dir, sizeof state_dir);
    cli_lock(NULL);

    xe_engine *e = xe_engine_open_with_context(model, context);
    int status = serve_stdio(e, state_dir, cache_dir);
    xe_engine_close(e);
    cli_unlock();
    return status;
}

static int cmd_serve(const char *model, int argc, char **argv) {
    int context = XE_CONTEXT_DEFAULT, context_seen = 0;
    const char *state_option = NULL;
    const char *cache_dir = NULL;
    const char *socket_option = NULL;
    double idle_minutes = 0.0;
    for (int i = 3; i < argc; i++) {
        if (cli_context_option(argc, argv, &i, &context, &context_seen)) continue;
        if (!strcmp(argv[i], "--state")) {
            if (++i >= argc) usage();
            state_option = argv[i];
        } else if (!strcmp(argv[i], "--cache")) {
            if (++i >= argc) usage();
            cache_dir = argv[i];
        } else if (!strcmp(argv[i], "--socket")) {
            if (++i >= argc) usage();
            socket_option = argv[i];
        } else if (!strcmp(argv[i], "--idle-shutdown")) {
            if (++i >= argc) usage();
            errno = 0;
            char *end;
            double value = strtod(argv[i], &end);
            if (errno || *end || !isfinite(value) || value <= 0.0) usage();
            idle_minutes = value;
        } else {
            usage();
        }
    }
    char state_dir[4096];
    cli_state_dir(state_option, state_dir, sizeof state_dir);
    char socket_path[4096];
    if (!serve_socket_path(socket_option, state_dir, socket_path,
                           sizeof socket_path)) {
        fprintf(stderr, "xenolith: serve: socket path is too long\n");
        exit(1);
    }
    cli_lock(socket_path);

    xe_engine *e = xe_engine_open_with_context(model, context);
    int status = serve_run(e, state_dir, cache_dir, socket_path,
                           idle_minutes);
    xe_engine_close(e);
    cli_unlock();
    return status;
}

int main(int argc, char **argv) {
    if (argc < 2) usage();
    const char *cmd = argv[1];
    if (argc < 3) usage();
    const char *model = argv[2];

    if (!strcmp(cmd, "info")) return cmd_info(model, argc, argv);
    if (!strcmp(cmd, "tokenize")) return cmd_tokenize(model, argc, argv);
    if (!strcmp(cmd, "run")) return cmd_run(model, argc, argv);
    if (!strcmp(cmd, "chat")) return cmd_chat(model, argc, argv);
    if (!strcmp(cmd, "oracle")) return cmd_oracle(model, argc, argv);
    if (!strcmp(cmd, "bench")) return cmd_bench(model, argc, argv);
    if (!strcmp(cmd, "wire")) return cmd_wire(model, argc, argv);
    if (!strcmp(cmd, "serve")) return cmd_serve(model, argc, argv);

    usage();
    return 0;
}
