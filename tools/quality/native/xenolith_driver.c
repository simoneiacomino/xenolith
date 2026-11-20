#define _POSIX_C_SOURCE 200809L
#include "xenolith.h"
#include <errno.h>
#include <limits.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

/* TSV: safe case key, exact token IDs, selected positions, explicit chunks.
 * Output: little-endian float32, selected rows in order, full vocabulary.
 * Optional layer stream: int32(layer, absolute position, width), float32[width].
 */
static double seconds(void) {
    struct timespec t;
    clock_gettime(CLOCK_MONOTONIC, &t);
    return t.tv_sec + t.tv_nsec * 1e-9;
}
static int integer(const char *s, int maximum, int *result) {
    char *end;
    errno = 0;
    long value = strtol(s, &end, 10);
    if (!*s || *end || errno || value < 0 || value > maximum) return 0;
    *result = (int)value;
    return 1;
}
static int integers(char *s, int32_t *out, int capacity, int maximum) {
    int n = 0;
    for (char *p = s; ; ) {
        char *comma = strchr(p, ',');
        if (comma) *comma = 0;
        int value;
        if (n == capacity || !integer(p, maximum, &value)) return -1;
        out[n++] = value;
        if (!comma) return n;
        p = comma + 1;
    }
}
struct layer_sink { FILE *stream; int failed; int tokens; int count; unsigned char *seen; };
static void layer(void *user, int index, int position, const float *hidden, int width) {
    struct layer_sink *sink = user;
    int32_t header[] = { index, position, width };
    if (sink->failed) return;
    if (index < 0 || index >= 30 || position < 0 || position >= sink->tokens || width != 2816) { sink->failed = 1; return; }
    size_t key = (size_t)position * 30 + (size_t)index;
    if (sink->seen[key]) { sink->failed = 1; return; }
    sink->seen[key] = 1; ++sink->count;
    for (int i = 0; i < width; ++i) if (!isfinite(hidden[i])) sink->failed = 1;
    if (fwrite(header, sizeof(header), 1, sink->stream) != 1 ||
        fwrite(hidden, sizeof(float), (size_t)width, sink->stream) != (size_t)width)
        sink->failed = 1;
}
int main(int argc, char **argv) {
    uint32_t endian = 1;
    int context;
    if ((argc != 6 && argc != 7) || *(unsigned char *)&endian != 1 ||
        sizeof(float) != 4 || !integer(argv[4], 262144, &context) || context < 64 ||
        (strcmp(argv[5], "prefill") && strcmp(argv[5], "decode"))) {
        fprintf(stderr, "usage: xenolith_driver MODEL INPUT.tsv OUTPUT_DIR CONTEXT prefill|decode [LAYER_DIR]\n");
        return 2;
    }
    FILE *input = fopen(argv[2], "r");
    if (!input) { perror(argv[2]); return 2; }
    int32_t *ids = malloc((size_t)context * sizeof(*ids));
    int32_t *positions = malloc((size_t)context * sizeof(*positions));
    int32_t *chunks = malloc((size_t)context * sizeof(*chunks));
    if (!ids || !positions || !chunks) return 2;
    xe_engine *engine = NULL;
    xe_session *session = NULL;
    char *line = NULL;
    size_t capacity = 0;
    int cases = 0, result = 2;
    while (getline(&line, &capacity, input) >= 0) {
        char *fields[4];
        fields[0] = line;
        for (int i = 1; i < 4; ++i) {
            char *tab = strchr(fields[i-1], '\t');
            if (!tab) goto cleanup;
            *tab = 0; fields[i] = tab + 1;
        }
        fields[3][strcspn(fields[3], "\r\n")] = 0;
        const char *key = fields[0];
        if (!*key || strlen(key) > 128 ||
            strspn(key, "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-_") != strlen(key)) goto cleanup;
        int n = integers(fields[1], ids, context, 262143);
        int np = integers(fields[2], positions, context, context-1);
        int nc = integers(fields[3], chunks, context, context);
        if (n < 1 || np < 1 || nc < 1) goto cleanup;
        for (int j = 0; j < np; ++j)
            if (positions[j] >= n || (j && positions[j] <= positions[j-1])) goto cleanup;
        if (!strcmp(argv[5], "decode")) {
            nc = n;
            for (int j = 0; j < n; ++j) chunks[j] = 1;
        }
        int end = 0, selected = 0;
        for (int j = 0; j < nc; ++j) {
            if (chunks[j] < 1 || chunks[j] > n-end) goto cleanup;
            end += chunks[j];
            if (selected < np && positions[selected] < end-1) goto cleanup;
            if (selected < np && positions[selected] == end-1) ++selected;
        }
        if (end != n || selected != np) goto cleanup;
        if (!engine) {
            double start = seconds();
            xe_engine_options options = { .context = context, .workers = 8 };
            engine = xe_engine_open_with_options(argv[1], &options);
            if (!engine) goto cleanup;
            session = xe_session_new(engine);
            if (!session || xe_vocab_size(engine) != 262144) goto cleanup;
            printf("{\"event\":\"loaded\",\"seconds\":%.9g,\"vocab\":%d}\n", seconds()-start, xe_vocab_size(engine));
            fflush(stdout);
        }
        char path[4096];
        int used = snprintf(path, sizeof(path), "%s/%s.bin", argv[3], key);
        if (used < 0 || (size_t)used >= sizeof(path)) goto cleanup;
        FILE *output = fopen(path, "wbx");
        if (!output) goto cleanup;
        struct layer_sink sink = {0};
        if (argc == 7) {
            used = snprintf(path, sizeof(path), "%s/%s.layers", argv[6], key);
            if (used < 0 || (size_t)used >= sizeof(path)) { fclose(output); goto cleanup; }
            sink.stream = fopen(path, "wbx");
            if (!sink.stream) { fclose(output); goto cleanup; }
            sink.tokens = n;
            sink.seen = calloc((size_t)n * 30, 1);
            if (!sink.seen) { fclose(sink.stream); fclose(output); goto cleanup; }
            xe_session_set_layer_observer(session, layer, &sink);
        }
        xe_session_reset(session);
        xe_tokens tokens = {0};
        double start = seconds();
        end = selected = 0;
        int failed = 0;
        for (int j = 0; j < nc; ++j) {
            end += chunks[j];
            while (tokens.len < end) xe_tokens_push(&tokens, ids[tokens.len]);
            xe_session_sync(session, &tokens);
            if (selected < np && positions[selected] == end-1) {
                const float *z = xe_session_logits(session);
                if (!z) { failed = 1; break; }
                for (int v = 0; v < 262144; ++v) if (!isfinite(z[v])) failed = 1;
                if (fwrite(z, sizeof(float), 262144, output) != 262144) failed = 1;
                ++selected;
            }
            if (failed || sink.failed) break;
        }
        xe_session_set_layer_observer(session, NULL, NULL);
        if (sink.stream && sink.count != n * 30) failed = 1;
        free(sink.seen);
        if (sink.stream && fclose(sink.stream)) failed = 1;
        if (fclose(output)) failed = 1;
        xe_tokens_free(&tokens);
        if (failed || sink.failed || selected != np) goto cleanup;
        ++cases;
        printf("{\"event\":\"case\",\"id\":\"%s\",\"seconds\":%.9g,\"tokens\":%d,\"positions\":%d}\n", key, seconds()-start, n, np);
        fflush(stdout);
    }
    if (!ferror(input) && cases) result = 0;
cleanup:
    if (result) fprintf(stderr, "quality driver ERROR: invalid input, incomplete output or engine failure\n");
    free(line); free(ids); free(positions); free(chunks);
    fclose(input);
    if (session) xe_session_free(session);
    if (engine) xe_engine_close(engine);
    return result;
}
