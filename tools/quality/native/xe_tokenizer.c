#define _POSIX_C_SOURCE 200809L
#include "xenolith.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
int main(int argc, char **argv) {
    if (argc != 3) return 2;
    FILE *f = fopen(argv[2], "rb");
    if (!f || fseek(f, 0, SEEK_END)) return 2;
    long size = ftell(f);
    if (size < 0 || size > 1048576 || fseek(f, 0, SEEK_SET)) return 2;
    char *text = calloc((size_t)size + 1, 1);
    int32_t *ids = malloc(262144 * sizeof(*ids));
    if (!text || !ids || fread(text, 1, (size_t)size, f) != (size_t)size) return 2;
    fclose(f);
    if (memchr(text, 0, (size_t)size)) return 2;
    xe_engine *engine = xe_engine_open_vocab(argv[1]);
    if (!engine) return 2;
    int count = xe_encode_text_bounded(engine, text, ids, 262144);
    if (count < 1 || count > 262144) return 2;
    for (int i = 0; i < count; ++i) printf("%s%d", i ? "," : "", ids[i]);
    puts("");
    xe_engine_close(engine); free(text); free(ids);
    return 0;
}
