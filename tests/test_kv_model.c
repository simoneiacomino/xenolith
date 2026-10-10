/* Targeted model integration checks. No timing or sampling decisions are involved. */
#include "../xenolith.c"
#include "test_context.h"

static void require(int ok, const char *what) {
    if (!ok) xe_fatal("kv-model: FAIL %s", what);
}

static int top(const float *p) {
    int best = 0;
    for (int i = 0; i < XE_VOCAB; i++) {
        require(isfinite(p[i]), "nonfinite logits");
        if (p[i] > p[best]) best = i;
    }
    return best;
}

static void compare(const float *got, const float *want, const char *stage) {
    double err = 0, norm = 0;
    for (int i = 0; i < XE_VOCAB; i++) {
        double d = (double)got[i] - want[i];
        err += d * d;
        norm += (double)want[i] * want[i];
    }
    double rel = sqrt(err / (norm + 1e-30));
    int same_top = top(got) == top(want);
    printf("kv-model: %s relative_rms=%.9g top=%s\n",
           stage, rel, same_top ? "same" : "different");
    require(isfinite(rel) && rel < 1e-5 && same_top, stage);
}

static void dump(FILE *f, xe_session *s) {
    (void)top(s->logits);
    require(fwrite(s->logits, sizeof(float), XE_VOCAB, f) == XE_VOCAB,
            "write logits");
}

static void decode_dump(xe_engine *e, int32_t *tokens, const char *directory) {
    int depths[] = {512, 2048, e->context - 8};
    int count = e->context <= 4104 ? 3 : 2;
    for (int i = 0; i < count; i++) {
        int depth = depths[i];
        require(depth > 0 && depth + 8 <= e->context, "decode shape");
        char path[4096];
        int n = snprintf(path, sizeof path, "%s/depth%d.f32", directory, depth);
        require(n > 0 && (size_t)n < sizeof path, "dump path");
        FILE *f = fopen(path, "wb");
        require(f != NULL, "open logits dump");
        xe_session *s = xe_session_new(e);
        xe_tokens prefix = {tokens, depth, e->context};
        xe_session_sync(s, &prefix);
        dump(f, s);
        for (int step = 0; step < 8; step++) {
            prefix.len++;
            xe_session_sync(s, &prefix); /* A one-token append uses CPU decode. */
            require(s->n_tokens == depth + step + 1, "decode position");
            dump(f, s);
        }
        require(fclose(f) == 0, "close logits dump");
        xe_session_free(s);
        printf("kv-model: decode depth=%d steps=8 vocab=%d PASS\n", depth, XE_VOCAB);
        fflush(stdout);
    }
}

/* Capture/compare populated rows across every global layer, K/V and head.
 * The dense snapshot deliberately has no dependency on the tail's stride. */
static void tail_copy(xe_session *s, int rows, _Float16 *copy, int check) {
    size_t offset = 0, bytes = (size_t)rows * XE_GLOBAL_HEAD_DIM * sizeof(*copy);
    for (int layer = 0; layer < XE_GLOBAL_LAYERS; layer++)
        for (int value = 0; value < 2; value++)
            for (int head = 0; head < XE_GLOBAL_KV_HEADS; head++) {
                _Float16 *p = (value ? s->cow_global_v[layer] : s->cow_global_k[layer])
                    + (size_t)head * s->cow_capacity * XE_GLOBAL_HEAD_DIM;
                if (check) require(memcmp(copy + offset, p, bytes) == 0,
                                   "old tail rows changed during growth");
                else memcpy(copy + offset, p, bytes);
                offset += (size_t)rows * XE_GLOBAL_HEAD_DIM;
            }
}

static void promoted_kv(xe_session *s, xe_session *shadow, int rows) {
    size_t bytes = (size_t)rows * XE_GLOBAL_HEAD_DIM * sizeof(_Float16);
    for (int layer = 0; layer < XE_GLOBAL_LAYERS; layer++)
        for (int value = 0; value < 2; value++)
            for (int head = 0; head < XE_GLOBAL_KV_HEADS; head++) {
                const _Float16 *source = (value ? shadow->cow_global_v[layer]
                    : shadow->cow_global_k[layer])
                    + (size_t)head * shadow->cow_capacity * XE_GLOBAL_HEAD_DIM;
                const _Float16 *target = (value ? s->global_v : s->global_k)
                    + (size_t)layer * xe_global_layer_elems(s->engine)
                    + ((size_t)head * s->engine->context + shadow->cow_split) * XE_GLOBAL_HEAD_DIM;
                require(memcmp(source, target, bytes) == 0, "promoted global KV");
            }
}

static void cow_growth(xe_engine *e, int32_t *tokens, int split) {
    const int tails[] = {511, 512, 513, 1023, 1024, 1025};
    require(split > 0 && split + 1025 + 8 <= e->context, "growth shape");
    xe_tokens prefix = {tokens, split, e->context};
    xe_session *reference = xe_session_new(e);
    xe_session *source = xe_session_new(e);
    xe_session_sync(reference, &prefix);
    xe_session_sync(source, &prefix);
    xe_session *shadow = xe_session_shadow_new(source);
    require(shadow != NULL, "open shadow");
    int32_t *raw = malloc((size_t)e->context * sizeof(*raw));
    require(raw != NULL, "allocate divergent tokens");
    memcpy(raw, tokens, (size_t)e->context * sizeof(*raw));
    for (int i = split; i < split + 64; i++) raw[i] += 1024;
    xe_tokens divergent = {raw, split + 64, e->context};
    xe_session_sync(source, &divergent);
    free(raw);

    int previous = 0, growths = 0;
    for (size_t i = 0; i < sizeof tails / sizeof *tails; i++) {
        int rows = tails[i], old_capacity = shadow->cow_capacity;
        size_t elements = (size_t)2 * XE_GLOBAL_LAYERS * XE_GLOBAL_KV_HEADS
                          * previous * XE_GLOBAL_HEAD_DIM;
        _Float16 *saved = elements ? malloc(elements * sizeof(*saved)) : NULL;
        require(!elements || saved != NULL, "allocate tail snapshot");
        if (previous) tail_copy(shadow, previous, saved, 0);
        prefix.len = split + rows;
        require(xe_session_shadow_sync(shadow, &prefix) == rows - previous,
                "shadow appended rows");
        xe_session_shadow_refresh_logits(shadow);
        if (previous) tail_copy(shadow, previous, saved, 1);
        free(saved);
        require(shadow->cow_capacity == ((rows + 511) & ~511), "tail capacity");
        if (old_capacity && old_capacity != shadow->cow_capacity) growths++;
        /* Use the same GPU chunks in the ordinary reference, including the
         * one-row checkpoints. Public sync would choose CPU for those rows. */
        xe_prefill_batch_run(reference, tokens + split + previous,
                             rows - previous, split + previous, 1);
        compare(shadow->logits, reference->logits, "cow checkpoint");
        require(shadow->n_tokens == prefix.len, "shadow position");
        printf("kv-model: cow split=%d tail=%d capacity=%d preserved=exact PASS\n",
               split, rows, shadow->cow_capacity);
        fflush(stdout);
        previous = rows;
    }
    require(growths == 2, "two actual reallocations");
    require(xe_session_shadow_promote(source, shadow), "promote");
    require(source->n_tokens == prefix.len, "promoted position");
    require(memcmp(source->tokens, tokens, (size_t)prefix.len * sizeof(*tokens)) == 0,
            "promoted token history");
    promoted_kv(source, shadow, previous);
    compare(source->logits, reference->logits, "promoted logits");
    xe_session_free(shadow);
    for (int step = 0; step < 8; step++) {
        prefix.len++;
        xe_session_sync(reference, &prefix);
        xe_session_sync(source, &prefix);
        compare(source->logits, reference->logits, "post-promotion decode");
    }
    xe_session_free(source);
    xe_session_free(reference);
    printf("kv-model: cow-growth split=%d checkpoints=6 reallocations=2 decode=8 PASS\n", split);
}

int main(int argc, char **argv) {
    int context = test_context_capacity(&argc, argv);
    if (argc != 4 || (strcmp(argv[2], "decode-dump") && strcmp(argv[2], "cow-growth"))) {
        fprintf(stderr, "usage: %s model.gguf decode-dump DIRECTORY | cow-growth SPLIT [--ctx N]\n", argv[0]);
        return 2;
    }
    require(context >= 2056, "context >= 2056");
    int32_t *tokens = malloc((size_t)context * sizeof(*tokens));
    require(tokens != NULL, "allocate tokens");
    for (int i = 0; i < context; i++) tokens[i] = 2 + (i * 17 + 11) % 8192;
    xe_engine *e = xe_engine_open_with_context(argv[1], context);
    if (!strcmp(argv[2], "decode-dump")) decode_dump(e, tokens, argv[3]);
    else cow_growth(e, tokens, atoi(argv[3]));
    xe_engine_close(e);
    free(tokens);
    puts("kv-model: complete PASS");
    return 0;
}
