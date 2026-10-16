#ifndef KVSTORE_H
#define KVSTORE_H

#include "xenolith.h"

#include <stdint.h>

#define KVSTORE_DEFAULT_BUDGET UINT64_C(17179869184)

typedef struct kvstore kvstore;

typedef struct {
    uint8_t bytes[16];
} kvstore_id;

typedef struct {
    uint8_t key[32];
    uint64_t rebuild_cost;
    kvstore_id group;
    int has_group;
    int has_key;
    int pinned;
} kvstore_save_options;

typedef enum {
    KVSTORE_OK,
    KVSTORE_MISS,
    KVSTORE_DISABLED,
    KVSTORE_INVALID_ARGUMENT,
    KVSTORE_IO,
    KVSTORE_BUDGET,
    KVSTORE_REJECTED,
    KVSTORE_PINNED
} kvstore_status;

/* Open does not evict. Restore legacy membership before enforcing a budget. */
kvstore_status kvstore_open(kvstore **out, const char *directory,
                            uint64_t budget);
void kvstore_close(kvstore *store);
kvstore_status kvstore_set_group(kvstore *store, const kvstore_id *id,
                                 const kvstore_id *group);
/* Called only after the transcript durably references the new checkpoint.
 * Optional eligibility filter excludes refs outside the current cache epoch. */
typedef int (*kvstore_eligible_fn)(const kvstore_id *id, void *context);
kvstore_status kvstore_committed(kvstore *store, const kvstore_id *id,
                                  xe_session *session,
                                  kvstore_eligible_fn eligible, void *context);

kvstore_status kvstore_save(kvstore *store, xe_session *session,
                            const kvstore_save_options *options,
                            kvstore_id *id,
                            xe_snapshot_status *snapshot_status);
kvstore_status kvstore_load(kvstore *store, const kvstore_id *id,
                            xe_session *session, const xe_tokens *expected,
                            xe_snapshot_status *snapshot_status);
kvstore_status kvstore_load_input(kvstore *store, const kvstore_id *id,
                                  xe_session *session, const xe_input *expected,
                                  xe_snapshot_status *snapshot_status);
kvstore_status kvstore_find(kvstore *store, const uint8_t key[32],
                            kvstore_id *id);
kvstore_status kvstore_contains(kvstore *store, const kvstore_id *id);
kvstore_status kvstore_remove(kvstore *store, const kvstore_id *id);

kvstore_status kvstore_anchor_get(kvstore *store, const char *name,
                                  kvstore_id *id);
kvstore_status kvstore_anchor_set(kvstore *store, const char *name,
                                  const kvstore_id *id);
kvstore_status kvstore_anchor_clear(kvstore *store, const char *name);

kvstore_status kvstore_enforce_budget(kvstore *store);
kvstore_status kvstore_usage(kvstore *store, uint64_t *total,
                             uint64_t *pinned);

void kvstore_id_format(const kvstore_id *id, char out[33]);
int kvstore_id_parse(const char *text, kvstore_id *id);
const char *kvstore_status_name(kvstore_status status);

#endif
