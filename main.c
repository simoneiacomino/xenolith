#define _POSIX_C_SOURCE 200809L

#include "xenolith.h"
#include "format.h"
#include "profile.h"
#include "conversation.h"
#include "serve.h"
#include "kvstore.h"
#include "json.h"
#include <sys/stat.h>

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
"  --workers N|auto  --cpu-list id,id,...  (one worker per physical core)\n"
"  Without CPU options, XE_WORKERS and XE_CPU_LIST supply the CPU plan.\n"
"  info     print model metadata\n"
"  tokenize -p \"text\" | -f file [--pieces]\n"
"  run      -p \"prompt\" [-n max] [--temp F] [--top-k N] [--top-p F] [--seed S]\n"
"  vision-encode --mmproj FILE --image FILE [--image-tokens N] --vision-dump DIR [--final-only]\n"
"  run      image options: --mmproj FILE --image FILE [--image-tokens N] [--vision-dump DIR]\n"
"  chat     image options: --mmproj FILE --image FILE [--image-tokens N] (first message)\n"
"  run/bench --prefill 512|1024 (default 512)\n"
"  chat     [-n max] [--temp F] [--top-k N] [--top-p F] [--seed S]\n"
"  run/chat --steering FILE [--steering-strength F] (experimental, text only)\n"
"  steering-capture --dataset FILE --out DIR [--ctx N] [--thinking] [--completions] [--cpu]\n"
"                   [--capture-site post_layer_residual_v1|post_ffn_output_v1]\n"
"  steering-import --steering FILE --out DIR (prints artifact ID)\n"
"  wire/serve --steering-dir DIR enables registered steering artifacts\n"
"  oracle   -t \"id,id,id\" [-o logits.bin] [--layers dir] [--q8]\n"
"  bench    [-p N] [-n N] [-d N] [-r N] [--delay S] [--no-warmup] [--progress] [-o md|jsonl]\n"
"  wire     [--state DIR] [--cache DIR] [--cache-budget SIZE] [--mmproj FILE]\n"
"  serve    [--state DIR] [--cache DIR] [--cache-budget SIZE] [--socket PATH] [--idle-shutdown MIN] [--mmproj FILE]\n"
"           SIZE: bytes, kB/MB/GB or KiB/MiB/GiB; default 16GiB; 0 disables disk cache\n");
    exit(1);
}

static int cli_lock_fd = -1;

static int cli_parse_cache_budget(const char *text, uint64_t *out) {
    if (!text || !*text || !out) return 0;
    const unsigned char *p = (const unsigned char *)text;
    uint64_t value = 0;
    if (*p < '0' || *p > '9') return 0;
    do {
        unsigned digit = (unsigned)(*p - '0');
        if (value > (UINT64_MAX - digit) / 10) return 0;
        value = value * 10 + digit;
        p++;
    } while (*p >= '0' && *p <= '9');

    uint64_t multiplier = 1;
    if (!*p || strcmp((const char *)p, "B") == 0) multiplier = 1;
    else if (strcmp((const char *)p, "kB") == 0) multiplier = UINT64_C(1000);
    else if (strcmp((const char *)p, "MB") == 0) multiplier = UINT64_C(1000000);
    else if (strcmp((const char *)p, "GB") == 0) multiplier = UINT64_C(1000000000);
    else if (strcmp((const char *)p, "KiB") == 0) multiplier = UINT64_C(1024);
    else if (strcmp((const char *)p, "MiB") == 0) multiplier = UINT64_C(1048576);
    else if (strcmp((const char *)p, "GiB") == 0) multiplier = UINT64_C(1073741824);
    else return 0;
    if (value > UINT64_MAX / multiplier) return 0;
    *out = value * multiplier;
    return 1;
}

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

typedef struct {
    xe_engine_options options;
    int workers_seen, list_seen, automatic;
    int *cpus;
} cli_cpu_plan;

static int cli_decimal(const char *s, int allow_zero, int *out) {
    if (!s || !*s) return 0;
    int n = 0;
    for (; *s; s++) {
        if (*s < '0' || *s > '9' || n > (INT_MAX - (*s - '0')) / 10) return 0;
        n = n * 10 + (*s - '0');
    }
    if (!allow_zero && !n) return 0;
    *out = n;
    return 1;
}

static void cli_cpu_workers(cli_cpu_plan *plan, const char *value, const char *name,
                            int allow_auto) {
    if (allow_auto && !strcmp(value, "auto")) { plan->automatic = 1; return; }
    if (!cli_decimal(value, 0, &plan->options.workers)) {
        fprintf(stderr, "xenolith: %s must be a positive integer%s\n", name,
                allow_auto ? " or auto" : "");
        exit(1);
    }
}

static void cli_cpu_list(cli_cpu_plan *plan, const char *value, const char *name) {
    if (!value || !*value) goto invalid;
    const char *p = value;
    int count = 0;
    while (*p) {
        int n = 0;
        if (*p < '0' || *p > '9') goto invalid;
        do {
            int digit = *p++ - '0';
            if (n > (INT_MAX - digit) / 10) goto invalid;
            n = n * 10 + digit;
        } while (*p >= '0' && *p <= '9');
        if (count == INT_MAX || (size_t)count >= SIZE_MAX / sizeof(int) - 1) goto invalid;
        int *next = realloc(plan->cpus, ((size_t)count + 1) * sizeof(int));
        if (!next) { fprintf(stderr, "xenolith: out of memory parsing %s\n", name); exit(1); }
        plan->cpus = next;
        plan->cpus[count++] = n;
        if (!*p) break;
        if (*p++ != ',' || !*p) goto invalid;
    }
    plan->options.cpus = plan->cpus;
    plan->options.cpu_count = count;
    return;
invalid:
    fprintf(stderr, "xenolith: %s requires comma-separated decimal CPU IDs\n", name);
    exit(1);
}

static int cli_cpu_option(int argc, char **argv, int *index, cli_cpu_plan *plan) {
    int workers = !strcmp(argv[*index], "--workers");
    int list = !strcmp(argv[*index], "--cpu-list");
    if (!workers && !list) return 0;
    const char *name = argv[*index];
    int *seen = workers ? &plan->workers_seen : &plan->list_seen;
    if (*seen || ++*index >= argc) {
        fprintf(stderr, "xenolith: %s requires one value and may appear only once\n", name);
        exit(1);
    }
    *seen = 1;
    if (workers) cli_cpu_workers(plan, argv[*index], name, 1);
    else cli_cpu_list(plan, argv[*index], name);
    return 1;
}

static void cli_cpu_finish(cli_cpu_plan *plan, int context) {
    if (!plan->workers_seen && !plan->list_seen) {
        const char *workers = getenv("XE_WORKERS"), *list = getenv("XE_CPU_LIST");
        if (workers) cli_cpu_workers(plan, workers, "XE_WORKERS", 0);
        if (list) cli_cpu_list(plan, list, "XE_CPU_LIST");
    }
    if (plan->automatic && plan->list_seen) {
        fprintf(stderr, "xenolith: --workers auto cannot be combined with --cpu-list\n");
        exit(1);
    }
    if (plan->options.workers && plan->options.cpu_count &&
        plan->options.workers != plan->options.cpu_count) {
        fprintf(stderr, "xenolith: worker count does not match CPU list length\n");
        exit(1);
    }
    plan->options.context = context;
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

static uint64_t bench_now_ns(void);

static int cli_visual_budget(const char *arg) {
    char *end;
    errno = 0;
    long n = strtol(arg, &end, 10);
    if (errno || *end || !(n == 70 || n == 140 || n == 280 || n == 560 || n == 1120)) usage();
    return (int)n;
}

static float *cli_encode_image(const char *model, const char *mmproj, const char *path,
                               int budget, const char *dump, xe_image *image, uint8_t hash[32], uint8_t model_hash[32],
                               const xe_engine_options *options) {
    char error[256];uint64_t started=bench_now_ns();
    if (!xe_image_load(image, path, budget, error, sizeof error)) {
        fprintf(stderr, "xenolith: %s\n", error); exit(1);
    }
    uint64_t prepared=bench_now_ns();
    xe_engine *encoder = xe_engine_open_vocab_with_options(model, options);
    memcpy(model_hash,xe_engine_model_fingerprint(encoder),32);
    uint64_t hashed=bench_now_ns();
    if (!xe_engine_enable_vision(encoder, mmproj, error, sizeof error)) {
        fprintf(stderr, "xenolith: %s\n", error); exit(1);
    }
    uint64_t loaded=bench_now_ns();
    float *out = malloc((size_t)image->rows * XE_VISUAL_WIDTH * sizeof(float));
    if (!out || !xe_engine_encode_image(encoder, image, out, dump)) {
        fprintf(stderr, "xenolith: image encoding failed\n"); exit(1);
    }
    uint64_t encoded=bench_now_ns();
    fprintf(stderr,"vision: budget=%d rows=%d preprocess_ms=%.3f model_hash_ms=%.3f mmproj_load_ms=%.3f encoder_ms=%.3f\n",
            image->budget,image->rows,(prepared-started)/1e6,(hashed-prepared)/1e6,(loaded-hashed)/1e6,(encoded-loaded)/1e6);
    memcpy(hash, xe_engine_vision_fingerprint(encoder), 32);
    xe_engine_close(encoder);
    return out;
}

static int cmd_vision_encode(const char *model, int argc, char **argv) {
    const char *mmproj = NULL, *path = NULL, *dump = NULL;
    int budget = 280;
    int final_only = 0;
    cli_cpu_plan cpu = {0};
    for (int i = 3; i < argc; i++) {
        if (cli_cpu_option(argc, argv, &i, &cpu)) continue;
        if (!strcmp(argv[i], "--final-only")) { final_only = 1; continue; }
        if (++i >= argc) usage();
        if (!strcmp(argv[i-1], "--mmproj")) mmproj = argv[i];
        else if (!strcmp(argv[i-1], "--image")) path = argv[i];
        else if (!strcmp(argv[i-1], "--vision-dump")) dump = argv[i];
        else if (!strcmp(argv[i-1], "--image-tokens")) budget = cli_visual_budget(argv[i]);
        else usage();
    }
    if (!mmproj || !path || !dump) usage();
    cli_cpu_finish(&cpu, 0);
    cli_lock(NULL);
    xe_image image; uint8_t hash[32], model_hash[32];
    float *out = cli_encode_image(model, mmproj, path, budget, final_only ? NULL : dump, &image, hash, model_hash, &cpu.options);
    free(cpu.cpus);
    if (final_only) {
        char output_path[4096];
        int length = snprintf(output_path, sizeof output_path, "%s/embedding.f32", dump);
        if (length < 0 || (size_t)length >= sizeof output_path) { fprintf(stderr,"xenolith: vision dump path too long\n"); exit(1); }
        FILE *file = fopen(output_path, "wb");
        if (!file) { fprintf(stderr,"xenolith: cannot open final vision dump\n"); exit(1); }
        int ok = format_write(file, out, (size_t)image.rows * XE_VISUAL_WIDTH * sizeof(float));
        if (fclose(file)) ok = 0;
        if (!ok) { fprintf(stderr,"xenolith: cannot write final vision dump\n"); exit(1); }
    }
    printf("vision rows %d width %d\n", image.rows, XE_VISUAL_WIDTH);
    free(out); xe_image_free(&image); cli_unlock(); return 0;
}

static float cli_steering_strength(const char *text) {
    errno = 0; char *end; float v = strtof(text, &end);
    if (!*text || errno || *end || !isfinite(v)) {
        fprintf(stderr, "xenolith: steering strength must be finite\n"); exit(1);
    }
    return v;
}
static xe_steering *cli_steering_load(xe_engine *e, const char *path) {
    if (!path) return NULL;
    xe_steering *v = NULL; char error[256];
    if (!xe_steering_load(path, xe_engine_layer_count(e), xe_engine_hidden_width(e),
                           xe_engine_model_fingerprint(e), &v, error, sizeof error)) {
        fprintf(stderr, "xenolith: %s\n", error); exit(1);
    }
    return v;
}
static void cli_steering_apply(xe_session *s, xe_steering *v, float strength) {
    if (v && !xe_session_steering_set(s, v, strength)) {
        fprintf(stderr, "xenolith: cannot configure steering\n"); exit(1);
    }
}

static int cmd_run(const char *model, int argc, char **argv) {
    int context = XE_CONTEXT_DEFAULT, context_seen = 0;
    cli_cpu_plan cpu = {0};
    const char *steering_path = NULL;
    float steering_strength = 1;
    xe_steering *steering_vector = NULL;
    const char *prompt = NULL;
    const char *mmproj = NULL, *image_path = NULL, *vision_dump = NULL;
    int image_budget = 280, prefill_rows = 512;
    int max_n = 512;
    float temp = 1.0f;
    int top_k = 64;
    float top_p = 0.95f;
    uint64_t seed = (uint64_t)time(NULL);

    for (int i = 3; i < argc; i++) {
        if (cli_context_option(argc, argv, &i, &context, &context_seen) ||
            cli_cpu_option(argc, argv, &i, &cpu)) continue;
        if (!strcmp(argv[i], "--steering")) {
            if (++i >= argc || steering_path) usage();
            steering_path = argv[i];
        } else if (!strcmp(argv[i], "--steering-strength")) {
            if (++i >= argc) usage();
            steering_strength = cli_steering_strength(argv[i]);
        } else if (!strcmp(argv[i], "--image") || !strcmp(argv[i], "--mmproj") || !strcmp(argv[i], "--vision-dump")) {
            const char *key = argv[i]; if (++i >= argc) usage();
            if (!strcmp(key, "--image")) image_path = argv[i];
            else if (!strcmp(key, "--mmproj")) mmproj = argv[i];
            else vision_dump = argv[i];
        } else if (!strcmp(argv[i], "--image-tokens")) {
            if (++i >= argc) usage();
            image_budget = cli_visual_budget(argv[i]);
        } else if (!strcmp(argv[i], "--prefill")) {
            if (++i >= argc || (strcmp(argv[i], "512") && strcmp(argv[i], "1024"))) usage();
            prefill_rows = atoi(argv[i]);
        } else if (!strcmp(argv[i], "-p")) {
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
    if (!prompt || (!!image_path != !!mmproj) || (vision_dump && !image_path)) usage();
    if (image_path && !context_seen) context = 65536;
    if (image_path && context!=65536 && context!=131072) {
        fprintf(stderr,"xenolith: initial vision support requires --ctx 65536 or 131072\n");return 1;
    }

    if (steering_path && image_path) {
        fprintf(stderr, "xenolith: steering with visual input is not supported\n"); return 1;
    }
    cli_cpu_finish(&cpu, context);
    cli_lock(NULL);

    xe_image image = {0}; uint8_t projector_hash[32], model_hash[32]; float *image_embeddings = NULL;
    if (image_path) image_embeddings = cli_encode_image(model, mmproj, image_path, image_budget, vision_dump, &image, projector_hash, model_hash, &cpu.options);
    xe_engine *e = (steering_path ? xe_engine_open_cached_with_options(model, &cpu.options) : xe_engine_open_with_options(model, &cpu.options));
    free(cpu.cpus);
    int ctx = xe_context_size(e);
    int vocab = xe_vocab_size(e);
    if (top_k > vocab) {
        fprintf(stderr, "xenolith: run: top-k %d out of range [0, %d]\n", top_k, vocab);
        exit(1);
    }
    int32_t *toks = malloc((size_t)ctx * sizeof(*toks));
    if (!toks) {
        fprintf(stderr, "xenolith: out of memory\n");
        exit(1);
    }
    toks[0] = xe_bos_id(e);
    int encoded = xe_encode_text_bounded(e, prompt, toks + 1, ctx - 1);
    if (encoded > ctx - 1) {
        fprintf(stderr, "xenolith: run: prompt has %lld tokens including BOS; context capacity is %d\n",
                (long long)encoded + 1, ctx);
        free(toks);
        xe_steering_release(steering_vector);
        xe_engine_close(e);
        cli_unlock();
        return 1;
    }
    int n = 1 + encoded;
    xe_visual_span visual = {0};
    profile *visual_profile = NULL;
    if (image_embeddings) {
        if (profile_open(&visual_profile,e) != PROFILE_OK) {
            fprintf(stderr,"xenolith: visual profile mismatch\n"); exit(1);
        }
        profile_render render;
        if (profile_render_system(visual_profile,NULL,NULL,0,0,&render) != PROFILE_OK ||
            render.token_count >= (uint32_t)ctx) exit(1);
        n = (int)render.token_count;
        memcpy(toks,render.tokens,(size_t)n*sizeof(*toks));
        uint32_t offset;
        if (profile_render_user_image(visual_profile,prompt,image.rows,PROFILE_TURN_PADDED,&render,&offset) != PROFILE_OK ||
            render.token_count > (uint32_t)(ctx-n)) exit(1);
        visual.start=n+(int)offset; visual.rows=image.rows; visual.embeddings=image_embeddings;
        visual.media=image.encoded;visual.media_size=image.encoded_size;visual.budget=image.budget;
        memcpy(toks+n,render.tokens,(size_t)render.token_count*sizeof(*toks));n+=(int)render.token_count;
        if (profile_render_reply_open(visual_profile,PROFILE_TURN_PADDED,0,0,&render) != PROFILE_OK ||
            render.token_count > (uint32_t)(ctx-n)) exit(1);
        memcpy(toks+n,render.tokens,(size_t)render.token_count*sizeof(*toks));n+=(int)render.token_count;
        profile_parser_reset(visual_profile,0);
        xe_visual_identity(&image, model_hash, projector_hash,
                           image_embeddings, XE_VISION_NUMERIC_REVISION, visual.identity);
    }
    if (max_n > 0 && n >= ctx) {
        fprintf(stderr, "xenolith: run: prompt leaves no room for generation\n");
        exit(1);
    }

    xe_tokens tokens = { toks, n, ctx };
    steering_vector = cli_steering_load(e, steering_path);
    xe_session *s = xe_session_new(e);
    cli_steering_apply(s, steering_vector, steering_strength);
    if (!xe_session_set_text_prefill(s, prefill_rows)) { fprintf(stderr,"xenolith: cannot set prefill size\n"); exit(1); }
    xe_input input = {toks,n,image_embeddings ? &visual : NULL,image_embeddings ? 1 : 0};
    uint64_t prefill_started=bench_now_ns();
    if (image_embeddings && !xe_session_sync_input(s, &input)) { fprintf(stderr,"xenolith: invalid visual prompt\n"); exit(1); }

    if (image_embeddings) fprintf(stderr,"vision: llm_prefill_ms=%.3f context=%d text_batch=%d\n",(bench_now_ns()-prefill_started)/1e6,ctx,prefill_rows);

    if (vision_dump && image_embeddings) {
        char path[4096];
        if (strlen(vision_dump)>sizeof path-32) { fprintf(stderr,"xenolith: vision dump path too long\n");exit(1); }
        snprintf(path,sizeof path,"%s/logits.f32",vision_dump);
        FILE *f=fopen(path,"wb");
        if (!f || fwrite(xe_session_logits(s),sizeof(float),(size_t)vocab,f)!=(size_t)vocab || fclose(f)) {
            fprintf(stderr,"xenolith: logits dump failed\n"); exit(1);
        }
        snprintf(path,sizeof path,"%s/prompt.ids",vision_dump);
        f=fopen(path,"w");if(!f)exit(1);
        for(int j=0;j<n;j++)fprintf(f,"%s%d",j?",":"",toks[j]);
        fputs("\n",f);fclose(f);
        snprintf(path,sizeof path,"%s/span.json",vision_dump);
        f=fopen(path,"w");if(!f)exit(1);
        fprintf(f,"{\"start\":%d,\"rows\":%d,\"count\":%d}\n",visual.start,visual.rows,n);fclose(f);
    }
    int32_t eot = xe_eot_id(e);
    int32_t eos = xe_eos_id(e);

    xe_sampler sp = { temp, top_k, top_p, seed };

    for (int i = 0; i < max_n && tokens.len < ctx; i++) {
        if (image_embeddings) {
            input.count = tokens.len;
            if (!xe_session_sync_continuation(s, &tokens, NULL)) { fprintf(stderr,"xenolith: visual sync failed\n"); exit(1); }
        } else xe_session_sync(s, &tokens);
        int32_t t = xe_session_next(s, &sp);
        if (t == eot || t == eos) break;
        if (visual_profile) {
            profile_parse_event event;
            if (profile_parser_feed(visual_profile,t,&event) != PROFILE_OK) exit(1);
            if (event.kind == PROFILE_PARSE_TEXT) fwrite(event.text,1,event.text_length,stdout);
        } else {
            char buf[256];
            int nb = xe_detokenize(e, t, buf, sizeof buf);
            fwrite(buf, 1, (size_t)nb, stdout);
        }
        fflush(stdout);
        xe_tokens_push(&tokens, t);
    }
    putchar('\n');

    profile_close(visual_profile);
    free(image_embeddings); xe_image_free(&image);
    xe_tokens_free(&tokens);
    xe_session_free(s);
    xe_steering_release(steering_vector);
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
    cli_cpu_plan cpu = {0};
    const char *steering_path = NULL;
    float steering_strength = 1;
    xe_steering *steering_vector = NULL;
    const char *mmproj = NULL, *image_path = NULL;
    int image_budget = 280;
    int max_n = 512;
    float temp = 1.0f;
    int top_k = 64;
    float top_p = 0.95f;
    uint64_t seed = (uint64_t)time(NULL);

    for (int i = 3; i < argc; i++) {
        if (cli_context_option(argc, argv, &i, &context, &context_seen) ||
            cli_cpu_option(argc, argv, &i, &cpu)) continue;
        if (!strcmp(argv[i], "--steering")) {
            if (++i >= argc || steering_path) usage();
            steering_path = argv[i];
        } else if (!strcmp(argv[i], "--steering-strength")) {
            if (++i >= argc) usage();
            steering_strength = cli_steering_strength(argv[i]);
        } else if (!strcmp(argv[i], "--image") || !strcmp(argv[i], "--mmproj")) {
            const char *key = argv[i];
            if (++i >= argc) usage();
            if (!strcmp(key, "--image")) image_path = argv[i];
            else mmproj = argv[i];
        } else if (!strcmp(argv[i], "--image-tokens")) {
            if (++i >= argc) usage();
            image_budget = cli_visual_budget(argv[i]);
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

    if (!!image_path != !!mmproj) usage();
    if (image_path && !context_seen) context = 65536;
    if (image_path && context != 65536 && context != 131072) {
        fprintf(stderr, "xenolith: vision chat requires --ctx 65536 or 131072\n");
        return 1;
    }
    cli_cpu_finish(&cpu, context);
    if (steering_path && image_path) {
        fprintf(stderr, "xenolith: steering with visual input is not supported\n"); return 1;
    }
    cli_lock(NULL);

    xe_image image = {0};
    float *image_embeddings = NULL;
    uint8_t projector_hash[32], model_hash[32];
    xe_visual_span visual = {0};
    int visual_added = 0, visual_synced = 0;
    if (image_path) {
        image_embeddings = cli_encode_image(model, mmproj, image_path, image_budget,
                                             NULL, &image, projector_hash, model_hash, &cpu.options);
        visual.rows = image.rows;
        visual.embeddings = image_embeddings;
        visual.media = image.encoded;
        visual.media_size = image.encoded_size;
        visual.budget = image.budget;
        xe_visual_identity(&image, model_hash, projector_hash, image_embeddings,
                           XE_VISION_NUMERIC_REVISION, visual.identity);
    }

    xe_engine *e = (steering_path ? xe_engine_open_cached_with_options(model, &cpu.options) : xe_engine_open_with_options(model, &cpu.options));
    free(cpu.cpus);
    int ctx = xe_context_size(e);
    int vocab = xe_vocab_size(e);
    if (top_k > vocab) {
        fprintf(stderr, "xenolith: chat: top-k %d out of range [0, %d]\n", top_k, vocab);
        exit(1);
    }

    profile *pf = NULL;
    if (profile_open(&pf, e) != PROFILE_OK) {
        fprintf(stderr, "xenolith: chat: model profile mismatch\n");
        exit(1);
    }

    steering_vector = cli_steering_load(e, steering_path);
    xe_session *s = xe_session_new(e);
    cli_steering_apply(s, steering_vector, steering_strength);
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

    if (image_embeddings) fputs("Image ready. Ask your first question.\n", stdout);

    while (!stop) {
        fputs("you> ", stdout);
        fflush(stdout);
        ssize_t nread = getline(&line, &line_cap, stdin);
        if (nread < 0) break;
        while (nread > 0 && (line[nread - 1] == '\n' || line[nread - 1] == '\r'))
            line[--nread] = '\0';
        if (!strncmp(line, "/steer ", 7)) {
            const char *arg = line + 7;
            if (!strcmp(arg, "off")) {
                if (!xe_session_steering_set(s, NULL, 0))
                    fprintf(stderr, "xenolith: cannot disable steering\n");
            } else if (steering_vector) {
                steering_strength = cli_steering_strength(arg);
                cli_steering_apply(s, steering_vector, steering_strength);
            } else fprintf(stderr, "xenolith: start chat with --steering FILE\n");
            continue;
        }
        if (!strcmp(line, "/exit")) break;
        profile_status user_status;
        if (image_embeddings && !visual_added) {
            uint32_t offset;
            user_status = profile_render_user_image(pf, line, image.rows, turn,
                                                    &render, &offset);
            if (user_status == PROFILE_OK) {
                visual.start = transcript.len + (int)offset;
                visual_added = 1;
            }
        } else user_status = profile_render_user(pf, line, turn, &render);
        if (user_status != PROFILE_OK) {
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

        if (image_embeddings) profile_parser_reset(pf, 0);
        fputs("assistant> ", stdout);
        fflush(stdout);
        uint64_t t_start = 0, t_prev = 0;
        int n_gen = 0, col = 11, live = 0;
        int tty = isatty(fileno(stdout));
        int term_cols = tty ? term_width() : INT_MAX;
        for (int i = 0; i < max_n && transcript.len < ctx - 1; i++) {
            if (image_embeddings) {
                int synced;
                if (!visual_synced) {
                    xe_input input = {transcript.v, transcript.len, &visual, 1};
                    synced = xe_session_sync_input(s, &input);
                    if (synced) visual_synced = 1;
                } else synced = xe_session_sync_continuation(s, &transcript, NULL);
                if (!synced) {
                    fprintf(stderr, "xenolith: visual chat sync failed\n");
                    stop = 1;
                    break;
                }
            } else xe_session_sync(s, &transcript);
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
            const char *visible = buf;
            int nb;
            if (image_embeddings) {
                profile_parse_event event;
                if (profile_parser_feed(pf, t, &event) != PROFILE_OK) {
                    fprintf(stderr, "xenolith: visual chat parse failed\n");
                    stop = 1;
                    break;
                }
                nb = event.kind == PROFILE_PARSE_TEXT ? (int)event.text_length : 0;
                visible = (const char *)event.text;
            } else nb = xe_detokenize(e, t, buf, sizeof buf);
            if (nb > 0) {
                if (tty && live) {
                    if (col > 0) printf("\r\033[J\033[A\r\033[%dC", col);
                    else printf("\r\033[J\033[A\r");
                }
                if (tty) {
                    chat_put_token(visible, nb, &col, term_cols);
                } else {
                    fwrite(visible, 1, (size_t)nb, stdout);
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

    free(image_embeddings);
    xe_image_free(&image);
    free(line);
    xe_tokens_free(&transcript);
    xe_session_free(s);
    profile_close(pf);
    xe_steering_release(steering_vector);
    xe_engine_close(e);
    cli_unlock();
    return 0;
}

static int cmd_oracle(const char *model, int argc, char **argv) {
    int context = XE_CONTEXT_DEFAULT, context_seen = 0;
    cli_cpu_plan cpu = {0};
    const char *ids = NULL;
    const char *dump_path = NULL;
    const char *layers_dir = NULL;
    int q8_mode = 0;
    const char *steering_path = NULL;
    float steering_strength = 1;

    for (int i = 3; i < argc; i++) {
        if (cli_context_option(argc, argv, &i, &context, &context_seen) ||
            cli_cpu_option(argc, argv, &i, &cpu)) continue;
        if (!strcmp(argv[i], "--steering")) {if (++i >= argc) usage(); steering_path = argv[i];}
        else if (!strcmp(argv[i], "--steering-strength")) {if (++i >= argc) usage(); steering_strength = cli_steering_strength(argv[i]);}
        else if (!strcmp(argv[i], "-t")) {
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

    cli_cpu_finish(&cpu, context);
    cli_lock(NULL);

    xe_engine *e = (steering_path ? xe_engine_open_cached_with_options(model, &cpu.options) : xe_engine_open_with_options(model, &cpu.options));
    free(cpu.cpus);
    xe_steering *vector = cli_steering_load(e, steering_path);
    xe_oracle_steered(e, toks, n, dump_path, layers_dir, q8_mode, stdout, vector, steering_strength);
    xe_steering_release(vector);
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
    int prefill_rows;
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
    if (!xe_session_set_text_prefill(s, params->prefill_rows)) exit(1);
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
                              const bench_result *result, int prefill_rows) {
    char mask[128];
    bench_cpu_mask(e, mask, sizeof mask);
    printf("{\"engine\":\"xenolith\",\"build_commit\":\"%s\",\"build_number\":0,\"model_filename\":",
           XE_BUILD_COMMIT);
    bench_json_string(model);
    const char *backends = !strcmp(result->backend, "CPU")
                           || !strcmp(result->backend, "CPU serial")
                           ? "CPU" : "Level Zero,CPU";
    printf(",\"context_window\":%d,\"text_prefill_rows\":%d",xe_context_size(e),prefill_rows);
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
    cli_cpu_plan cpu = {0};
    bench_params params = { 512, 128, 0, 5, 0, 1, 0, 0, 512 };
    for (int i = 3; i < argc; i++) {
        if (cli_context_option(argc, argv, &i, &context, &context_seen) ||
            cli_cpu_option(argc, argv, &i, &cpu)) continue;
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
        } else if (!strcmp(argv[i], "--prefill")) {
            if (++i >= argc || (strcmp(argv[i], "512") && strcmp(argv[i], "1024"))) usage();
            params.prefill_rows = atoi(argv[i]);
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

    cli_cpu_finish(&cpu, context);
    cli_lock(NULL);

    xe_engine *e = xe_engine_open_with_options(model, &cpu.options);
    free(cpu.cpus);
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
            bench_print_jsonl(model, e, &result, params.prefill_rows);
        else
            bench_print_md(model, e, &result);
        free(result.samples_ns);
    }
    if (params.n_gen) {
        bench_result result = bench_run(e, &params, 0, params.n_gen, "CPU");
        if (params.jsonl)
            bench_print_jsonl(model, e, &result, params.prefill_rows);
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
    cli_cpu_plan cpu = {0};
    const char *state_option = NULL;
    const char *steering_dir = NULL;
    const char *mmproj = NULL;
    const char *cache_dir = NULL;
    uint64_t cache_budget = KVSTORE_DEFAULT_BUDGET;
    for (int i = 3; i < argc; i++) {
        if (cli_context_option(argc, argv, &i, &context, &context_seen) ||
            cli_cpu_option(argc, argv, &i, &cpu)) continue;
        if (!strcmp(argv[i], "--steering-dir")) {
            if (++i >= argc || steering_dir) usage();
            steering_dir = argv[i];
        } else if (!strcmp(argv[i],"--mmproj")) {
            if(++i>=argc)usage();
            mmproj=argv[i];
        } else if (!strcmp(argv[i], "--state")) {
            if (++i >= argc) usage();
            state_option = argv[i];
        } else if (!strcmp(argv[i], "--cache")) {
            if (++i >= argc) usage();
            cache_dir = argv[i];
        } else if (!strcmp(argv[i], "--cache-budget")) {
            if (++i >= argc || !cli_parse_cache_budget(argv[i], &cache_budget)) usage();
        } else {
            usage();
        }
    }
    if(mmproj&&!context_seen)context=65536;
    if(mmproj&&context!=65536&&context!=131072){fprintf(stderr,"xenolith: vision requires --ctx 65536 or 131072\n");return 1;}
    char state_dir[4096];
    cli_state_dir(state_option, state_dir, sizeof state_dir);
    cli_cpu_finish(&cpu, context);
    cli_lock(NULL);

    xe_engine *e = (cache_budget || mmproj || steering_dir) ? xe_engine_open_cached_with_options(model, &cpu.options)
                                          : xe_engine_open_with_options(model, &cpu.options);
    free(cpu.cpus);
    if (steering_dir && !xe_engine_steering_store(e, steering_dir)) { fprintf(stderr,"xenolith: invalid steering directory\n"); return 1; }
    if(mmproj){char error[256];if(!xe_engine_enable_vision(e,mmproj,error,sizeof error)){fprintf(stderr,"xenolith: %s\n",error);xe_engine_close(e);cli_unlock();return 1;}}
    int status = serve_stdio_with_budget(e, state_dir, cache_dir, cache_budget);
    xe_engine_close(e);
    cli_unlock();
    return status;
}

static int cmd_serve(const char *model, int argc, char **argv) {
    int context = XE_CONTEXT_DEFAULT, context_seen = 0;
    cli_cpu_plan cpu = {0};
    const char *state_option = NULL;
    const char *steering_dir = NULL;
    const char *mmproj = NULL;
    const char *cache_dir = NULL;
    uint64_t cache_budget = KVSTORE_DEFAULT_BUDGET;
    const char *socket_option = NULL;
    double idle_minutes = 0.0;
    for (int i = 3; i < argc; i++) {
        if (cli_context_option(argc, argv, &i, &context, &context_seen) ||
            cli_cpu_option(argc, argv, &i, &cpu)) continue;
        if (!strcmp(argv[i], "--steering-dir")) {
            if (++i >= argc || steering_dir) usage();
            steering_dir = argv[i];
        } else if (!strcmp(argv[i],"--mmproj")) {
            if(++i>=argc)usage();
            mmproj=argv[i];
        } else if (!strcmp(argv[i], "--state")) {
            if (++i >= argc) usage();
            state_option = argv[i];
        } else if (!strcmp(argv[i], "--cache")) {
            if (++i >= argc) usage();
            cache_dir = argv[i];
        } else if (!strcmp(argv[i], "--cache-budget")) {
            if (++i >= argc || !cli_parse_cache_budget(argv[i], &cache_budget)) usage();
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
    if(mmproj&&!context_seen)context=65536;
    if(mmproj&&context!=65536&&context!=131072){fprintf(stderr,"xenolith: vision requires --ctx 65536 or 131072\n");return 1;}
    char state_dir[4096];
    cli_state_dir(state_option, state_dir, sizeof state_dir);
    char socket_path[4096];
    if (!serve_socket_path(socket_option, state_dir, socket_path,
                           sizeof socket_path)) {
        fprintf(stderr, "xenolith: serve: socket path is too long\n");
        exit(1);
    }
    cli_cpu_finish(&cpu, context);
    cli_lock(socket_path);

    xe_engine *e = (cache_budget || mmproj || steering_dir) ? xe_engine_open_cached_with_options(model, &cpu.options)
                                          : xe_engine_open_with_options(model, &cpu.options);
    free(cpu.cpus);
    if (steering_dir && !xe_engine_steering_store(e, steering_dir)) { fprintf(stderr,"xenolith: invalid steering directory\n"); return 1; }
    if(mmproj){char error[256];if(!xe_engine_enable_vision(e,mmproj,error,sizeof error)){fprintf(stderr,"xenolith: %s\n",error);xe_engine_close(e);cli_unlock();return 1;}}
    int status = serve_run_with_budget(e, state_dir, cache_dir, socket_path,
                           idle_minutes, cache_budget);
    xe_engine_close(e);
    cli_unlock();
    return status;
}

static int cmd_steering(const char *model, int argc, char **argv, int capture) {
    const char *dataset = NULL, *out = NULL, *manifest = NULL;
    int context = 512, seen = 0, thinking = 0, completions = 0, cpu = 0;
    xe_steering_site capture_site=XE_STEERING_RESIDUAL;
    for (int i = 3; i < argc; i++) {
        if (cli_context_option(argc, argv, &i, &context, &seen)) continue;
        if (!strcmp(argv[i], "--thinking")) thinking = 1;
        else if (!strcmp(argv[i], "--completions")) completions = 1;
        else if (!strcmp(argv[i], "--cpu")) cpu = 1;
        else if (!strcmp(argv[i],"--capture-site")){
            if(++i>=argc)usage();
            if(!strcmp(argv[i],XE_STEERING_SITE))capture_site=XE_STEERING_RESIDUAL;
            else if(!strcmp(argv[i],XE_STEERING_FFN_SITE))capture_site=XE_STEERING_FFN;
            else usage();
        }
        else if (!strcmp(argv[i], "--dataset")) {if (++i >= argc) usage(); dataset = argv[i];}
        else if (!strcmp(argv[i], "--out")) {if (++i >= argc) usage(); out = argv[i];}
        else if (!strcmp(argv[i], "--steering")) {if (++i >= argc) usage(); manifest = argv[i];}
        else usage();
    }
    if (!out || (capture ? !dataset || manifest : !manifest || dataset)) usage();
    if ((!capture && (completions || cpu)) || (completions && thinking)) usage();
    if (mkdir(out, 0700) && errno != EEXIST) {perror(out); return 1;}
    cli_lock(NULL);
    xe_engine *e = xe_engine_open_cached(model, context);
    int layers = xe_engine_layer_count(e), width = xe_engine_hidden_width(e), result = 1;
    if (!capture) {
        char id[65], error[256];
        if (!xe_steering_import(out, manifest, layers, width, xe_engine_model_fingerprint(e), id, error, sizeof error))
            fprintf(stderr, "xenolith: %s\n", error);
        else {puts(id); result = 0;}
        goto done;
    }
    size_t dataset_size; char *dataset_bytes = read_file(dataset,&dataset_size);
    uint8_t dataset_hash[32]; char dataset_hex[65];
    format_sha256_bytes(dataset_bytes,dataset_size,dataset_hash); free(dataset_bytes);
    xe_steering_hex(dataset_hash,dataset_hex);
    FILE *input = fopen(dataset, "rb");
    if (!input) {perror(dataset); goto done;}
    profile *pf = NULL;
    if (profile_open(&pf, e) != PROFILE_OK) {fclose(input); goto done;}
    xe_session *session = xe_session_new(e);
    if(!xe_session_capture_enable_at(session,capture_site)){xe_session_free(session);profile_close(pf);fclose(input);goto done;}
    size_t elements = (size_t)layers * width;
    double *sum = calloc(elements, sizeof *sum);
    float *pooled = malloc(elements * sizeof *pooled);
    int32_t *answer_tokens = malloc((size_t)context * sizeof *answer_tokens);
    if (!sum || !pooled || !answer_tokens) {
        free(sum); free(pooled); free(answer_tokens);
        xe_session_free(session); profile_close(pf); fclose(input); goto done;
    }
    char *line = NULL; size_t capacity = 0; ssize_t length; int count = 0, capture_failed = 0;
    while ((length = getline(&line, &capacity, input)) >= 0) {
        if ((size_t)length > 1048576) {capture_failed = 1; break;}
        json_value *pair = json_parse(line, (size_t)length);
        const json_value *positive = json_member(pair,"positive"), *negative = json_member(pair,"negative");
        const json_value *system = json_member(pair,"system");
        const json_value *question = json_member(pair,"prompt");
        if (!positive || positive->type != JSON_STRING || !negative || negative->type != JSON_STRING ||
            !positive->text_length || !negative->text_length ||
            (completions ? !question || question->type != JSON_STRING || !question->text_length : question != NULL) ||
            (system && system->type != JSON_STRING)) {json_free(pair); capture_failed = 1; break;}
        int ok = 1;
        for (int pole = 0; pole < 2 && ok; pole++) {
            xe_tokens tokens = {0}; profile_render render;
            xe_session_reset(session);
            const char *text = pole ? negative->text : positive->text;
            profile_status ps = profile_render_system(pf, system ? system->text : "You are a helpful assistant.", NULL, 0, thinking, &render);
            if (ps == PROFILE_OK) for (uint32_t k = 0; k < render.token_count; k++) xe_tokens_push(&tokens,render.tokens[k]);
            if (ps == PROFILE_OK) ps = profile_render_user(pf,completions ? question->text : text,PROFILE_TURN_PADDED,&render);
            if (ps == PROFILE_OK) for (uint32_t k = 0; k < render.token_count; k++) xe_tokens_push(&tokens,render.tokens[k]);
            if (ps == PROFILE_OK) ps = profile_render_reply_open(pf,PROFILE_TURN_PADDED,thinking,0,&render);
            if (ps == PROFILE_OK) for (uint32_t k = 0; k < render.token_count; k++) xe_tokens_push(&tokens,render.tokens[k]);
            if (ps != PROFILE_OK || tokens.len > context || !tokens.len) ok = 0;
            int prefix = tokens.len, answer_count = 0;
            if (ok && completions) {
                answer_count = xe_encode_text_bounded(e, text, answer_tokens, context - prefix);
                if (answer_count <= 0 || answer_count > context - prefix) ok = 0;
            }
            if (ok) {
                if (cpu) {
                    for (int end = 1; end <= prefix; end++) {
                        tokens.len = end; xe_session_sync(session,&tokens);
                    }
                } else xe_session_sync(session,&tokens);
                if (completions) {
                    memset(sum, 0, elements * sizeof *sum);
                    for (int a = 0; a < answer_count; a++) {
                        xe_tokens_push(&tokens, answer_tokens[a]);
                        xe_session_sync(session, &tokens);
                        const float *rows = xe_session_capture_rows(session);
                        for (size_t k = 0; k < elements; k++) sum[k] += rows[k];
                    }
                    for (size_t k = 0; k < elements; k++) pooled[k] = (float)(sum[k] / answer_count);
                }
                char path[4096];
                int n = snprintf(path,sizeof path,"%s/%04d-%s.f32",out,count,pole?"negative":"positive");
                FILE *f = n > 0 && (size_t)n < sizeof path ? fopen(path,"wbx") : NULL;
                if (!f) ok = 0;
                else {
                    const float *rows = completions ? pooled : xe_session_capture_rows(session);
                    for (int k = 0; k < layers * width && ok; k++) {
                        uint32_t bits; uint8_t bytes[4]; memcpy(&bits, rows+k, 4); format_put_u32le(bytes,bits);
                        if (fwrite(bytes,1,4,f) != 4) ok = 0;
                    }
                    if (fclose(f)) ok = 0;
                }
                /* Preserve token positions so extraction can be independently audited. */
                json_writer trace = {0}; json_raw(&trace,"{\"prefix_tokens\":"); json_i64(&trace,prefix);
                json_raw(&trace,",\"capture_first\":"); json_i64(&trace,completions?prefix:prefix-1);
                json_raw(&trace,",\"capture_count\":"); json_i64(&trace,completions?answer_count:1);
                int piece_length; const char *piece=xe_token_piece(e,tokens.v[prefix-1],&piece_length);
                json_raw(&trace,",\"prefix_last_piece\":"); json_string(&trace,piece,(size_t)piece_length);
                json_raw(&trace,",\"tokens\":[");
                for (int k = 0; k < tokens.len; k++) {if(k)json_raw(&trace,",");json_i64(&trace,tokens.v[k]);}
                json_raw(&trace,"]}\n");
                n = snprintf(path,sizeof path,"%s/%04d-%s.tokens.json",out,count,pole?"negative":"positive");
                f = n > 0 && (size_t)n < sizeof path ? fopen(path,"wbx") : NULL;
                if (!f) ok = 0;
                else {if(trace.failed || fwrite(trace.data,1,trace.length,f)!=trace.length)ok=0;if(fclose(f))ok=0;}
                json_writer_free(&trace);
            }
            xe_tokens_free(&tokens);
        }
        json_free(pair);
        if (!ok) {capture_failed = 1; break;}
        count++; fprintf(stderr,"steering capture: pair %d\n",count);
    }
    if (!capture_failed && feof(input) && !ferror(input) && count > 0) {
        char model_hash[65],template_hash[65]; uint8_t hash[32]; uint64_t n;
        xe_steering_hex(xe_engine_model_fingerprint(e),model_hash);
        const char *tmpl = xe_chat_template(e,&n); format_sha256_bytes(tmpl,(size_t)n,hash); xe_steering_hex(hash,template_hash);
        json_writer j = {0}; json_raw(&j,"{\"version\":1,\"layers\":"); json_i64(&j,layers);
        json_raw(&j,",\"width\":"); json_i64(&j,width); json_raw(&j,",\"pairs\":"); json_i64(&j,count);
        json_raw(&j,",\"model_sha256\":"); json_string(&j,model_hash,64);
        json_raw(&j,",\"dataset_sha256\":"); json_string(&j,dataset_hex,64);
        json_raw(&j,",\"template_sha256\":"); json_string(&j,template_hash,64);
        json_raw(&j,",\"engine_commit\":"); json_string(&j,XE_BUILD_COMMIT,strlen(XE_BUILD_COMMIT));
        json_raw(&j,",\"reasoning\":"); json_string(&j,thinking?"on":"off",thinking?2:3);
        json_raw(&j,",\"capture_kind\":");
        const char *kind = completions ? "paired_completions_v1" : "paired_instructions_v1";
        json_string(&j,kind,strlen(kind));
        json_raw(&j,",\"site\":");
        const char *site=capture_site==XE_STEERING_FFN?XE_STEERING_FFN_SITE:XE_STEERING_SITE;
        json_string(&j,site,strlen(site));
        json_raw(&j,",\"position\":");
        const char *position = completions ? "mean_assistant_completion_tokens" : "last_rendered_prompt_token_including_assistant_opening";
        json_string(&j,position,strlen(position));
        json_raw(&j,",\"backend\":");
        const char *backend = cpu ? "cpu_tokenwise" : "production_prefill_cpu_completion";
        json_string(&j,backend,strlen(backend));
        json_raw(&j,",\"numeric_settings\":");
        const char *numeric = "production_cpu_decode_gpu_prefill; prefill=512; compiler=" __VERSION__
#ifdef XE_NUMERIC_SOURCE_HASH
            "; source=" XE_NUMERIC_SOURCE_HASH
#endif
#ifdef XE_NUMERIC_CFLAGS_HASH
            "; flags=" XE_NUMERIC_CFLAGS_HASH
#endif
            ;
        json_writer settings = {0}; json_raw(&settings,numeric);
        json_raw(&settings,"; capture_site=");json_raw(&settings,site);
        if(cpu)json_raw(&settings,"; capture_prefix=cpu_tokenwise");
#ifdef XE_CAPTURE_SOURCE_HASH
        json_raw(&settings,"; extractor=" XE_CAPTURE_SOURCE_HASH);
#endif
        if(settings.failed)j.failed=1;
        json_string(&j,settings.data,settings.length); json_writer_free(&settings);
        json_raw(&j,"}\n");
        char path[4096]; int npath = snprintf(path,sizeof path,"%s/capture.json",out);
        FILE *f = npath > 0 && (size_t)npath < sizeof path ? fopen(path,"wbx") : NULL;
        if (f) {int ok = !j.failed && fwrite(j.data,1,j.length,f)==j.length; if (fclose(f)) ok=0; if(ok)result=0;}
        json_writer_free(&j);
    }
    if (result) fprintf(stderr,"xenolith: capture failed; check dataset, context, and output directory\n");
    free(sum); free(pooled); free(answer_tokens);
    free(line); fclose(input); xe_session_free(session); profile_close(pf);
done: xe_engine_close(e); cli_unlock(); return result;
}

int main(int argc, char **argv) {
    if (argc < 2) usage();
    const char *cmd = argv[1];
    if (argc < 3) usage();
    const char *model = argv[2];

    if (!strcmp(cmd, "info")) return cmd_info(model, argc, argv);
    if (!strcmp(cmd, "tokenize")) return cmd_tokenize(model, argc, argv);
    if (!strcmp(cmd, "steering-capture")) return cmd_steering(model, argc, argv, 1);
    if (!strcmp(cmd, "steering-import")) return cmd_steering(model, argc, argv, 0);
    if (!strcmp(cmd, "run")) return cmd_run(model, argc, argv);
    if (!strcmp(cmd, "vision-encode")) return cmd_vision_encode(model, argc, argv);
    if (!strcmp(cmd, "chat")) return cmd_chat(model, argc, argv);
    if (!strcmp(cmd, "oracle")) return cmd_oracle(model, argc, argv);
    if (!strcmp(cmd, "bench")) return cmd_bench(model, argc, argv);
    if (!strcmp(cmd, "wire")) return cmd_wire(model, argc, argv);
    if (!strcmp(cmd, "serve")) return cmd_serve(model, argc, argv);

    usage();
    return 0;
}
