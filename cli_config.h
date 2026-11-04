#ifndef CLI_CONFIG_H
#define CLI_CONFIG_H

#include <stddef.h>
#include <stdint.h>

/* CLI configuration is separate from the inference API and never opens weights. */
typedef struct {
    char model[4096], mmproj[4096], socket[4096], state_dir[4096], cache_dir[4096];
    int context; /* Zero means unspecified. */
    uint64_t cache_budget;
    int has_cache_budget; /* A configured zero disables disk cache. */
    double idle_shutdown_minutes; /* Zero means unspecified. */
} cli_config;

int cli_config_command(int argc, char **argv);
/* Read validated defaults. NULL selects the environment/XDG path. An absent
 * file yields zeroed defaults; failure also leaves out zeroed. No CLI parsing,
 * directory creation or model loading is performed. */
int cli_config_load(const char *path, cli_config *out,
                    char *error, size_t error_cap);

/* Read-only selection shared by inspect and serve. Creating the selected parent
 * is a separate operation; a later failure must not select another endpoint. */
int cli_config_socket_path(const char *override, const char *state_dir,
                          char *out, size_t cap);
int cli_config_socket_parent(const char *socket_path);

#endif
