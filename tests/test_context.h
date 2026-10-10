#ifndef XE_TEST_CONTEXT_H
#define XE_TEST_CONTEXT_H

#include "../xenolith.h"
#include <errno.h>
#include <stdlib.h>
#include <string.h>

/* Model tests use a smaller default; an optional trailing --ctx selects another. */
static int test_context_capacity(int *argc, char **argv) {
    int context = 65536;
    if (*argc >= 3 && !strcmp(argv[*argc - 2], "--ctx")) {
        const char *value = argv[*argc - 1];
        int digits = value[0] != '\0';
        for (const char *p = value; *p; p++)
            if (*p < '0' || *p > '9') digits = 0;
        char *end;
        errno = 0;
        long n = strtol(value, &end, 10);
        if (!digits || errno || *end || n < XE_CONTEXT_MIN || n > XE_CONTEXT_MAX) {
            fprintf(stderr, "test context must be an integer in [%d, %d]\n",
                    XE_CONTEXT_MIN, XE_CONTEXT_MAX);
            exit(2);
        }
        context = (int)n;
        *argc -= 2;
    }
    return context;
}

#endif
