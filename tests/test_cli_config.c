#define _POSIX_C_SOURCE 200809L
#include "../cli_config.h"

#include <assert.h>
#include <errno.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <unistd.h>

static int cases;
static char root[64], config[4096];

static void write_settings(const char *text) {
    FILE *f = fopen(config, "w");
    assert(f);
    assert(fputs(text, f) >= 0);
    assert(!fclose(f));
}

static cli_config load(const char *path) {
    cli_config cfg;
    char error[8192];
    int ok = cli_config_load(path, &cfg, error, sizeof error);
    if (!ok) fprintf(stderr, "load failed: %s\n", error);
    assert(ok);
    cases++;
    return cfg;
}

static void rejects(void) {
    cli_config cfg;
    memset(&cfg, 0xff, sizeof cfg);
    char error[8192] = {0};
    assert(!cli_config_load(NULL, &cfg, error, sizeof error));
    assert(error[0]);
    assert(!cfg.model[0] && !cfg.mmproj[0] && !cfg.socket[0]);
    assert(!cfg.state_dir[0] && !cfg.cache_dir[0]);
    assert(!cfg.context && !cfg.has_cache_budget && !cfg.cache_budget);
    assert(!cfg.idle_shutdown_minutes);
    cases++;
}

int main(void) {
    strcpy(root, "/tmp/xe-config-XXXXXX");
    assert(mkdtemp(root));
    assert(snprintf(config, sizeof config, "%s/config.json", root) > 0);
    assert(!setenv("HOME", root, 1));
    assert(!setenv("XENOLITH_CONFIG", config, 1));
    unsetenv("XDG_CONFIG_HOME"); unsetenv("XDG_RUNTIME_DIR");
    unsetenv("XDG_STATE_HOME"); unsetenv("XDG_CACHE_HOME");
    cli_config cfg = load(NULL); /* Missing configuration is allowed. */
    assert(!cfg.model[0] && !cfg.context && !cfg.has_cache_budget);
    write_settings("{\"schema_version\":1,\"model\":\"/models/model.gguf\","
        "\"mmproj\":\"/models/projector.gguf\",\"context\":65536,"
        "\"socket\":\"/tmp/configured.sock\",\"state_dir\":\"/data/state\","
        "\"cache_dir\":\"/data/cache\",\"cache_budget\":\"4GiB\",\"idle_shutdown_minutes\":12.5}");
    cfg = load(NULL);
    assert(!strcmp(cfg.model, "/models/model.gguf"));
    assert(!strcmp(cfg.mmproj, "/models/projector.gguf"));
    assert(cfg.context == 65536);
    assert(!strcmp(cfg.socket, "/tmp/configured.sock"));
    assert(!strcmp(cfg.state_dir, "/data/state"));
    assert(!strcmp(cfg.cache_dir, "/data/cache"));
    assert(cfg.has_cache_budget && cfg.cache_budget == UINT64_C(4294967296));
    assert(cfg.idle_shutdown_minutes == 12.5);
    char missing[4096];
    snprintf(missing, sizeof missing, "%s/missing.json", root);
    cfg = load(missing);
    assert(!cfg.model[0]); /* Explicit path takes precedence over the environment. */
    write_settings("{\"schema_version\":1,\"cache_budget\":0}");
    cfg = load(NULL);
    assert(cfg.has_cache_budget && cfg.cache_budget == 0);
    write_settings("{\"schema_version\":1,\"cache_budget\":18446744073709551615}");
    cfg = load(NULL);
    assert(cfg.has_cache_budget && cfg.cache_budget == UINT64_MAX);
    const char *invalid[] = {
        "broken", "[]", "{}", "{\"schema_version\":2}",
        "{\"schema_version\":1,\"schema_version\":1}",
        "{\"schema_version\":1,\"model\":\"relative.gguf\"}",
        "{\"schema_version\":1,\"context\":63}",
        "{\"schema_version\":1,\"context\":65536.5}",
        "{\"schema_version\":1,\"context\":\"65536\"}",
        "{\"schema_version\":1,\"cache_budget\":18446744073709551616}",
        "{\"schema_version\":1,\"cache_budget\":\"18446744073709551615GiB\"}",
        "{\"schema_version\":1,\"idle_shutdown_minutes\":0}",
        "{\"schema_version\":1,\"idle_shutdown_minutes\":1e999}",
        "{\"schema_version\":1,\"unknown\":true}",
        "{\"schema_version\":1,\"model\":null}"
    };
    for (size_t i = 0; i < sizeof invalid / sizeof *invalid; i++) {
        write_settings(invalid[i]); rejects();
    }
    char selected[4096], expected[4096];
    assert(!setenv("XDG_RUNTIME_DIR", root, 1));
    assert(cli_config_socket_path(NULL, "/fallback", selected, sizeof selected));
    snprintf(expected, sizeof expected, "%s/xenolith/wire.sock", root);
    assert(!strcmp(selected, expected));
    snprintf(expected, sizeof expected, "%s/xenolith", root);
    struct stat st;
    assert(stat(expected, &st) && errno == ENOENT); /* Selection never mkdirs. */
    FILE *blocker = fopen(expected, "w"); assert(blocker); fclose(blocker);
    assert(cli_config_socket_path(NULL, "/fallback", selected, sizeof selected));
    assert(!strcmp(selected, "/fallback/wire.sock"));
    unlink(expected);
    unlink(config); rmdir(root);
    printf("native config: %d cases PASS\n", cases);
    return 0;
}
