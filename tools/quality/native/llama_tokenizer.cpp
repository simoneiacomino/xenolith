#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "llama.h"

static char *read_file(const char *path, size_t *len_out) {
    FILE *f = fopen(path, "rb");
    if (!f) {
        fprintf(stderr, "error: cannot open %s\n", path);
        return NULL;
    }
    if (fseek(f, 0, SEEK_END) != 0) {
        fclose(f);
        return NULL;
    }
    long n = ftell(f);
    if (n < 0 || fseek(f, 0, SEEK_SET) != 0) {
        fclose(f);
        return NULL;
    }
    char *buf = (char *)malloc((size_t)n + 1);
    if (!buf) {
        fclose(f);
        return NULL;
    }
    if (fread(buf, 1, (size_t)n, f) != (size_t)n) {
        fprintf(stderr, "error: short read on %s\n", path);
        free(buf);
        fclose(f);
        return NULL;
    }
    fclose(f);
    buf[n] = '\0';
    *len_out = (size_t)n;
    return buf;
}

static void quiet_log(enum ggml_log_level level, const char *text, void *user) {
    (void)level;
    (void)text;
    (void)user;
}

int main(int argc, char **argv) {
    if (argc != 3) {
        fprintf(stderr, "usage: %s <model.gguf> <prompt-file>\n", argv[0]);
        return 2;
    }

    size_t len = 0;
    char *text = read_file(argv[2], &len);
    if (!text) return 2;

    llama_log_set(quiet_log, NULL);
    llama_backend_init();

    struct llama_model_params mparams = llama_model_default_params();
    mparams.n_gpu_layers = 0;
    mparams.vocab_only   = true;

    struct llama_model *model = llama_model_load_from_file(argv[1], mparams);
    if (!model) {
        fprintf(stderr, "error: failed to load model %s\n", argv[1]);
        free(text);
        llama_backend_free();
        return 1;
    }

    const struct llama_vocab *vocab = llama_model_get_vocab(model);
    const bool add_special  = false; /* Explicit raw IDs; Gemma4 loader otherwise forces BOS. */
    const bool parse_specl  = false;

    int32_t n = -llama_tokenize(vocab, text, (int32_t)len, NULL, 0, add_special, parse_specl);
    if (n <= 0) {
        fprintf(stderr, "error: tokenization produced %d tokens\n", n);
        llama_model_free(model);
        free(text);
        llama_backend_free();
        return 1;
    }

    llama_token *ids = (llama_token *)malloc((size_t)n * sizeof *ids);
    if (!ids) {
        llama_model_free(model);
        free(text);
        llama_backend_free();
        return 1;
    }
    if (llama_tokenize(vocab, text, (int32_t)len, ids, n, add_special, parse_specl) < 0) {
        fprintf(stderr, "error: tokenization failed\n");
        free(ids);
        llama_model_free(model);
        free(text);
        llama_backend_free();
        return 1;
    }

    for (int32_t i = 0; i < n; i++) {
        printf("%s%d", i ? "," : "", ids[i]);
    }
    printf("\n");
    fflush(stdout);
    fprintf(stderr, "n_tokens %d\n", n);

    free(ids);
    llama_model_free(model);
    free(text);
    llama_backend_free();
    return 0;
}
