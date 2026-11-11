#define _POSIX_C_SOURCE 200809L
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/wait.h>
#include <unistd.h>

static int cases;

/* No model or image is opened successfully; these cases exercise parsing only. */
static int check(const char *command, char *const *opts, const char *workers,
                 const char *list, const char *expected) {
    char *args[32] = {"./xenolith", (char *)command, "/missing-xenolith-model.gguf"};
    int n = 3;
    for (int i = 0; opts[i]; i++) args[n++] = opts[i];
    if (!strcmp(command, "run")) { args[n++] = "-p"; args[n++] = "--workers"; }
    if (!strcmp(command, "vision-encode")) {
        args[n++] = "--mmproj"; args[n++] = "--workers";
        args[n++] = "--image"; args[n++] = "/missing-xenolith-image";
        args[n++] = "--vision-dump"; args[n++] = "/missing-xenolith-dump";
    }
    args[n] = NULL;
    int fds[2];
    if (pipe(fds)) return 0;
    pid_t pid = fork();
    if (!pid) {
        close(fds[0]); dup2(fds[1], STDERR_FILENO); close(fds[1]);
        if (workers) setenv("XE_WORKERS", workers, 1); else unsetenv("XE_WORKERS");
        if (list) setenv("XE_CPU_LIST", list, 1); else unsetenv("XE_CPU_LIST");
        execv(args[0], args); _exit(127);
    }
    close(fds[1]);
    char output[16384]; size_t used = 0; ssize_t got;
    while (used < sizeof output - 1 && (got = read(fds[0], output + used, sizeof output - 1 - used)) > 0)
        used += (size_t)got;
    output[used] = 0; close(fds[0]);
    int status = 0;
    int ok = pid > 0 && waitpid(pid, &status, 0) == pid && WIFEXITED(status) &&
             WEXITSTATUS(status) == 1 && strstr(output, expected);
    if (!ok) fprintf(stderr, "cli workers: %s case %d failed: %s\n", command, cases, output);
    cases++;
    return ok;
}

int main(void) {
    char root[] = "/tmp/xenolith-cli-workers-XXXXXX";
    if (!mkdtemp(root) || setenv("XDG_RUNTIME_DIR", root, 1) || setenv("XDG_STATE_HOME", root, 1) ||
        unsetenv("XENOLITH_CONFIG") || setenv("XDG_CONFIG_HOME", root, 1)) return 2;
    const char *commands[] = {"run", "chat", "bench", "wire", "serve", "vision-encode"};
    char *none[] = {NULL};
    char *auto_opt[] = {"--workers", "auto", NULL};
    char *numeric[] = {"--workers", "2", "--cpu-list", "0,2", NULL};
    char *list_only[] = {"--cpu-list", "0,2", NULL};
    char *bad_auto[] = {"--workers", "auto", "--cpu-list", "0", NULL};
    char *mismatch[] = {"--workers", "2", "--cpu-list", "0", NULL};
    char *duplicate[] = {"--workers", "2", "--workers", "1", NULL};
    char *invalid[] = {"--cpu-list", "0,", NULL};
    int ok = 1;
    for (size_t i = 0; i < sizeof commands / sizeof *commands; i++) {
        const char *target = !strcmp(commands[i], "vision-encode") ? "cannot open image" : "/missing-xenolith-model.gguf";
        ok &= check(commands[i], none, "bad", NULL, "XE_WORKERS");
        ok &= check(commands[i], none, "2", "0", "does not match");
        ok &= check(commands[i], auto_opt, "bad", "bad", target);
        ok &= check(commands[i], numeric, "bad", "bad", target);
        ok &= check(commands[i], list_only, "bad", "bad", target);
        ok &= check(commands[i], bad_auto, NULL, NULL, "cannot be combined");
        ok &= check(commands[i], mismatch, NULL, NULL, "does not match");
        ok &= check(commands[i], duplicate, NULL, NULL, "may appear only once");
        ok &= check(commands[i], invalid, NULL, NULL, "comma-separated");
    }
    char *bad_values[] = {"", "0", "-1", "+1", " 1", "1 ", "1x", "2147483648", NULL};
    for (int i = 0; bad_values[i]; i++) {
        char *args[] = {"--workers", bad_values[i], NULL};
        ok &= check("run", args, NULL, NULL, "positive integer");
    }
    char *bad_lists[] = {"", ",0", "0,,1", "0, 1", "-1", "2147483648", NULL};
    for (int i = 0; bad_lists[i]; i++) {
        char *args[] = {"--cpu-list", bad_lists[i], NULL};
        ok &= check("run", args, NULL, NULL, "comma-separated");
    }
    char *missing[] = {"--workers", NULL};
    ok &= check("chat", missing, NULL, NULL, "requires one value");
    ok &= check("info", none, "bad", "bad", "/missing-xenolith-model.gguf");
    ok &= check("run", none, "auto", NULL, "XE_WORKERS");
    ok &= check("run", none, "0", NULL, "XE_WORKERS");
    ok &= check("info", auto_opt, "bad", "bad", "Usage:");
    char *tokenize[] = {"-p", "test", "--workers", "1", NULL};
    ok &= check("tokenize", tokenize, "bad", "bad", "Usage:");
    char lock[256]; snprintf(lock, sizeof lock, "%s/xenolith/engine.lock", root); unlink(lock);
    snprintf(lock, sizeof lock, "%s/xenolith", root); rmdir(lock); rmdir(root);
    printf("cli workers: %d cases %s\n", cases, ok ? "PASS" : "FAIL");
    return ok ? 0 : 1;
}
