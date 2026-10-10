#define _GNU_SOURCE
#include "../xenolith.h"

#include <stdlib.h>
#include <string.h>
#include <sys/wait.h>
#include <unistd.h>

static int cases;

/* Missing model: accepted options reach file opening without needing a GPU. */
static int check(const char *command, const char *model, char **options,
                 const char *message) {
    char *args[24] = { "./xenolith", (char *)command, (char *)model };
    int n = 3;
    for (int i = 0; options[i]; i++) args[n++] = options[i];
    if (!strcmp(command, "run")) {
        args[n++] = "-p";
        args[n++] = "--ctx"; /* A prompt value must remain literal. */
    } else if (!strcmp(command, "oracle")) {
        args[n++] = "-t";
        args[n++] = "2";
    }
    /* Keep a missing --ctx value at the end, after required command options. */
    if (n > 3 && options[0] && !strcmp(options[0], "--ctx") && !options[1]) {
        memmove(args + 3, args + 4, (size_t)(n - 4) * sizeof(*args));
        args[n - 1] = "--ctx";
    }
    args[n] = NULL;
    int fds[2];
    if (pipe(fds)) return 0;
    pid_t child = fork();
    if (child == 0) {
        close(fds[0]);
        dup2(fds[1], STDERR_FILENO);
        close(fds[1]);
        execv(args[0], args);
        _exit(127);
    }
    close(fds[1]);
    char output[8192];
    size_t used = 0;
    ssize_t got;
    while ((got = read(fds[0], output + used, sizeof output - 1 - used)) > 0)
        used += (size_t)got;
    output[used] = '\0';
    close(fds[0]);
    int status = 0;
    int ok = child > 0 && waitpid(child, &status, 0) == child &&
             WIFEXITED(status) && WEXITSTATUS(status) == 1 &&
             strstr(output, message) != NULL;
    if (!ok) fprintf(stderr, "cli context: %s case %d failed: %s\n",
                     command, cases, output);
    cases++;
    return ok;
}

int main(void) {
    char root[] = "/tmp/xenolith-cli-context-XXXXXX";
    if (!mkdtemp(root) || setenv("XDG_RUNTIME_DIR", root, 1) ||
        setenv("XDG_STATE_HOME", root, 1)) return 2;
    char model[256];
    snprintf(model, sizeof model, "%s/missing.gguf", root);
    const char *commands[] = { "info", "run", "chat", "oracle", "bench", "wire", "serve" };
    char *invalid[] = { "", "0", "63", "262145", "-65", "+65", " 65", "65 ",
                        "65x", "65.0", "999999999999999999999999999999" };
    char *valid[] = { "64", "65", "4097", "50000", "65536", "262144" };
    int ok = 1;
    for (size_t c = 0; c < sizeof commands / sizeof commands[0]; c++) {
        for (size_t i = 0; i < sizeof invalid / sizeof invalid[0]; i++) {
            char *args[] = { "--ctx", invalid[i], NULL };
            ok &= check(commands[c], model, args, "--ctx must be an integer");
        }
        char *missing[] = { "--ctx", NULL };
        ok &= check(commands[c], model, missing, "--ctx");
        char *duplicate[] = { "--ctx", "65", "--ctx", "129", NULL };
        ok &= check(commands[c], model, duplicate, "may appear only once");
        for (size_t i = 0; i < sizeof valid / sizeof valid[0]; i++) {
            char *args[] = { "--ctx", valid[i], NULL };
            ok &= check(commands[c], model, args, model);
        }
        char *defaults[] = { NULL };
        ok &= check(commands[c], model, defaults, model);
    }
    char lock[256];
    snprintf(lock, sizeof lock, "%s/xenolith/engine.lock", root);
    unlink(lock);
    snprintf(lock, sizeof lock, "%s/xenolith", root);
    rmdir(lock);
    rmdir(root);
    printf("cli context: %d cases %s\n", cases, ok ? "PASS" : "FAIL");
    return ok ? 0 : 1;
}
