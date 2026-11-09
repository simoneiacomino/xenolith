#define _POSIX_C_SOURCE 200809L

#include "xenolith.h"
#include "profile.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <unistd.h>

#define XE_VOCAB 262144
#define XE_BOS 2

#define XE_DELIM "\n__ggml_vocab_test__\n"

static const char *CORPUS[] = { "short", "long", "mixed", "ws", "nl", "nlonly", "json" };
#define N_CORPUS ((int)(sizeof CORPUS / sizeof CORPUS[0]))

static const char *ROUNDTRIP_PROMPTS[] = { "short", "long" };
#define N_ROUNDTRIP ((int)(sizeof ROUNDTRIP_PROMPTS / sizeof ROUNDTRIP_PROMPTS[0]))

static void usage(const char *prog) {
    fprintf(stderr, "usage: %s [model.gguf]\n", prog);
    fprintf(stderr, "env: XENOLITH_MODEL\n");
    exit(2);
}

static char *read_whole(const char *path, size_t *len_out) {
    FILE *f = fopen(path, "rb");
    if (!f) {
        fprintf(stderr, "tokenizer: cannot open %s\n", path);
        return NULL;
    }
    if (fseek(f, 0, SEEK_END) != 0) { fclose(f); return NULL; }
    long n = ftell(f);
    if (n < 0 || fseek(f, 0, SEEK_SET) != 0) { fclose(f); return NULL; }
    char *buf = malloc((size_t)n + 1);
    if (!buf) { fclose(f); return NULL; }
    if (fread(buf, 1, (size_t)n, f) != (size_t)n) {
        fprintf(stderr, "tokenizer: short read on %s\n", path);
        free(buf);
        fclose(f);
        return NULL;
    }
    fclose(f);
    buf[n] = '\0';
    if (len_out) *len_out = (size_t)n;
    return buf;
}

static int parse_ids(const char *s, int32_t *out, int cap) {
    int n = 0;
    const char *p = s;
    while (*p) {
        while (*p == ' ' || *p == '\t' || *p == '\n' || *p == '\r' ||
               *p == ',' || *p == '[' || *p == ']') {
            p++;
        }
        if (!*p) break;
        char *e;
        long v = strtol(p, &e, 10);
        if (e == p) return -1;
        if (n >= cap) return -1;
        out[n++] = (int32_t)v;
        p = e;
    }
    return n;
}

static int ids_equal(const int32_t *a, int na, const int32_t *b, int nb) {
    if (na != nb) return 0;
    for (int i = 0; i < na; i++)
        if (a[i] != b[i]) return 0;
    return 1;
}

static int xenolith_tokenize(const xe_engine *e, const char *text, int32_t *out, int cap) {
    if (cap < 1) return -1;
    out[0] = xe_bos_id(e);
    return 1 + xe_encode_text(e, text, out + 1, cap - 1);
}

static void here_dir(const char *argv0, char *out, size_t cap) {
    const char *slash = strrchr(argv0, '/');
    if (!slash) {
        snprintf(out, cap, ".");
        return;
    }
    size_t n = (size_t)(slash - argv0);
    if (n >= cap) n = cap - 1;
    memcpy(out, argv0, n);
    out[n] = '\0';
}

static int stage_fixtures(const xe_engine *e, const char *here) {
    char inp_path[2048], out_path[2048];
    snprintf(inp_path, sizeof inp_path, "%s/fixtures/ggml-vocab-gemma-4.gguf.inp", here);
    snprintf(out_path, sizeof out_path, "%s/fixtures/ggml-vocab-gemma-4.gguf.out", here);

    size_t inp_len = 0;
    char *inp = read_whole(inp_path, &inp_len);
    if (!inp) return 0;
    char *out = read_whole(out_path, NULL);
    if (!out) { free(inp); return 0; }

    char *seg[512];
    int ncases = 0;
    char *cur = inp;
    seg[ncases++] = cur;
    char *d;
    size_t dl = strlen(XE_DELIM);
    while ((d = strstr(cur, XE_DELIM)) != NULL) {
        *d = '\0';
        cur = d + dl;
        if (ncases >= (int)(sizeof seg / sizeof seg[0])) break;
        seg[ncases++] = cur;
    }

    char *line[512];
    int nlines = 0;
    char *lp = out;
    line[nlines++] = lp;
    for (char *q = out; *q; q++) {
        if (*q == '\n') {
            *q = '\0';
            char *nxt = q + 1;
            if (*nxt && nlines < (int)(sizeof line / sizeof line[0]))
                line[nlines++] = nxt;
        }
    }

    if (ncases != nlines) {
        printf("tokenizer: stage a FAIL (%d input cases but %d output lines)\n", ncases, nlines);
        free(inp);
        free(out);
        return 0;
    }

    int pass = 0, fail = 0;
    int32_t got[8192], want[8192];
    for (int i = 0; i < ncases; i++) {
        int n_got = xenolith_tokenize(e, seg[i], got, (int)(sizeof got / sizeof got[0]));
        if (n_got < 1 || got[0] != XE_BOS) {
            printf("tokenizer:   case %d FAIL (no BOS)\n", i);
            fail++;
            continue;
        }
        int n_want = parse_ids(line[i], want, (int)(sizeof want / sizeof want[0]));
        if (n_want < 0) {
            printf("tokenizer:   case %d FAIL (bad expected ids)\n", i);
            fail++;
            continue;
        }
        if (ids_equal(got + 1, n_got - 1, want, n_want)) {
            pass++;
        } else {
            fail++;
            printf("tokenizer:   case %d FAIL (want %d ids, got %d)\n", i, n_want, n_got - 1);
        }
    }

    free(inp);
    free(out);
    if (fail == 0)
        printf("tokenizer: stage a PASS  %d/%d fixtures\n", pass, ncases);
    else
        printf("tokenizer: stage a FAIL  %d/%d fixtures passed, %d failed\n", pass, ncases, fail);
    return fail == 0;
}

static int stage_corpus(const xe_engine *e, const char *here) {
    int pass = 0, fail = 0;
    for (int i = 0; i < N_CORPUS; i++) {
        char src[2048], gold[2048];
        snprintf(src, sizeof src, "%s/prompts/%s.txt", here, CORPUS[i]);
        snprintf(gold, sizeof gold, "%s/golden/%s.ids", here, CORPUS[i]);

        size_t len = 0;
        char *text = read_whole(src, &len);
        if (!text) { printf("tokenizer:   %s FAIL (missing %s)\n", CORPUS[i], src); fail++; continue; }
        char *gtext = read_whole(gold, NULL);
        if (!gtext) { printf("tokenizer:   %s FAIL (missing %s)\n", CORPUS[i], gold); free(text); fail++; continue; }

        int cap = (int)len + 16;
        int32_t *got = malloc((size_t)cap * sizeof *got);
        int32_t *want = malloc((size_t)cap * sizeof *want);
        int n_got = xenolith_tokenize(e, text, got, cap);
        int n_want = parse_ids(gtext, want, cap);

        int n = n_got;
        if (ids_equal(got, n_got, want, n_want)) {
            printf("tokenizer:   %s PASS  n_tokens=%d\n", CORPUS[i], n);
            pass++;
        } else {
            printf("tokenizer:   %s FAIL  n_tokens=%d (golden %d)\n", CORPUS[i], n_got, n_want);
            fail++;
        }
        free(got);
        free(want);
        free(text);
        free(gtext);
    }
    if (fail == 0)
        printf("tokenizer: stage b PASS  %d/%d corpora\n", pass, N_CORPUS);
    else
        printf("tokenizer: stage b FAIL  %d passed, %d failed\n", pass, fail);
    return fail == 0;
}

static int stage_roundtrip(const xe_engine *e, const char *here) {
    int pass = 0, fail = 0;
    for (int i = 0; i < N_ROUNDTRIP; i++) {
        char src[2048];
        snprintf(src, sizeof src, "%s/prompts/%s.txt", here, ROUNDTRIP_PROMPTS[i]);
        size_t len = 0;
        char *text = read_whole(src, &len);
        if (!text) { printf("tokenizer:   %s FAIL (missing %s)\n", ROUNDTRIP_PROMPTS[i], src); fail++; continue; }

        int cap = (int)len + 16;
        int32_t *ids = malloc((size_t)cap * sizeof *ids);
        int n = xenolith_tokenize(e, text, ids, cap);

        char *rt = malloc(len + 1);
        size_t rn = 0;
        int overflow = 0;
        for (int k = 1; k < n; k++) {
            char buf[512];
            int nb = xe_detokenize(e, ids[k], buf, (int)sizeof buf);
            if (rn + (size_t)nb > len) { overflow = 1; break; }
            memcpy(rt + rn, buf, (size_t)nb);
            rn += (size_t)nb;
        }
        if (!overflow && rn == len && memcmp(rt, text, len) == 0) {
            printf("tokenizer:   %s PASS  %zu bytes reconstructed\n", ROUNDTRIP_PROMPTS[i], len);
            pass++;
        } else {
            printf("tokenizer:   %s FAIL  reconstructed %zu of %zu bytes\n", ROUNDTRIP_PROMPTS[i], rn, len);
            fail++;
        }
        free(ids);
        free(rt);
        free(text);
    }
    if (fail == 0)
        printf("tokenizer: stage d PASS  %d/%d roundtrips\n", pass, N_ROUNDTRIP);
    else
        printf("tokenizer: stage d FAIL  %d passed, %d failed\n", pass, fail);
    return fail == 0;
}

static int stage_chat_prompt(xe_engine *e) {
    static const int32_t first[] = {
        2, 105, 2364, 107, 5379, 236748, 236764, 2229, 203360, 236881,
        106, 107, 105, 4368, 107, 100, 45518, 107, 101,
    };
    static const int32_t second[] = {
        107, 105, 2364, 107, 3900, 152039, 655, 236909, 887,
        236813, 6064, 514, 4775, 106, 107, 105, 4368, 107, 100, 45518,
        107, 101,
    };

    profile *p = NULL;
    int ok = profile_open(&p, e) == PROFILE_OK;
    if (!ok) {
        printf("tokenizer: stage e FAIL  profile mismatch\n");
        return 0;
    }

    xe_tokens transcript = {0};
    profile_render render;
    ok = ok && profile_render_system(p, NULL, NULL, 0, 0, &render)
               == PROFILE_OK;
    for (uint32_t i = 0; ok && i < render.token_count; i++)
        xe_tokens_push(&transcript, render.tokens[i]);
    ok = ok && profile_render_user(p, "  ciao, come stai?  ",
                                   PROFILE_TURN_PADDED, &render)
               == PROFILE_OK;
    for (uint32_t i = 0; ok && i < render.token_count; i++)
        xe_tokens_push(&transcript, render.tokens[i]);
    ok = ok && profile_render_reply_open(p, PROFILE_TURN_PADDED, 0, 0,
                                         &render)
               == PROFILE_OK;
    for (uint32_t i = 0; ok && i < render.token_count; i++)
        xe_tokens_push(&transcript, render.tokens[i]);
    ok = ok && ids_equal(transcript.v, transcript.len,
                         first, (int)(sizeof first / sizeof first[0]));

    int32_t answer[64];
    int na = xe_encode_text(e, "Ciao!", answer, (int)(sizeof answer / sizeof answer[0]));
    for (int i = 0; i < na; i++) xe_tokens_push(&transcript, answer[i]);
    xe_tokens_push(&transcript, xe_eot_id(e));
    int before = transcript.len;
    ok = ok && profile_render_user(p, "Scrivi <|turn> letteralmente",
                                   PROFILE_TURN_BARE, &render)
               == PROFILE_OK;
    for (uint32_t i = 0; ok && i < render.token_count; i++)
        xe_tokens_push(&transcript, render.tokens[i]);
    ok = ok && profile_render_reply_open(p, PROFILE_TURN_PADDED, 0, 0,
                                         &render)
               == PROFILE_OK;
    for (uint32_t i = 0; ok && i < render.token_count; i++)
        xe_tokens_push(&transcript, render.tokens[i]);
    ok = ok && ids_equal(transcript.v + before, transcript.len - before,
                         second, (int)(sizeof second / sizeof second[0]));

    int32_t literal[32];
    int nl = xe_encode_text(e, "<|turn>", literal, (int)(sizeof literal / sizeof literal[0]));
    for (int i = 0; i < nl; i++) if (literal[i] == 105) ok = 0;
    if (xe_encode_text(e, "", literal, (int)(sizeof literal / sizeof literal[0])) != 0) ok = 0;

    xe_tokens_free(&transcript);
    profile_close(p);
    printf("tokenizer: stage e %s  template parity, live append, special-token isolation\n",
           ok ? "PASS" : "FAIL");
    return ok;
}

int main(int argc, char **argv) {
    if (argc > 2) usage(argv[0]);
    const char *model = argc == 2 ? argv[1] : getenv("XENOLITH_MODEL");
    if (!model || !*model) usage(argv[0]);
    char here[1024];
    here_dir(argv[0], here, sizeof here);
    xe_engine *e = xe_engine_open_vocab(model);
    int ok = stage_fixtures(e, here);
    ok &= stage_corpus(e, here);
    ok &= stage_roundtrip(e, here);
    ok &= stage_chat_prompt(e);
    xe_engine_close(e);
    printf("tokenizer: %s\n", ok ? "PASS" : "FAIL");
    return ok ? 0 : 1;
}
