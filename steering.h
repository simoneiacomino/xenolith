#ifndef XE_STEERING_H
#define XE_STEERING_H

#include <stddef.h>
#include <stdint.h>

#define XE_STEERING_SITE "post_layer_residual_v1"
#define XE_STEERING_FFN_SITE "post_ffn_output_v1"
typedef struct xe_steering xe_steering;
typedef enum { XE_STEERING_ADDITIVE = 1, XE_STEERING_PROJECTION = 2 } xe_steering_mode;
typedef enum { XE_STEERING_RESIDUAL = 1, XE_STEERING_FFN = 2 } xe_steering_site;

/* Immutable, reference-counted artifacts. Outputs change only on success. */
int xe_steering_load(const char *manifest, int layers, int width,
                     const uint8_t model[32], xe_steering **out,
                     char *error, size_t cap);
void xe_steering_retain(xe_steering *v);
void xe_steering_release(xe_steering *v);
const char *xe_steering_id(const xe_steering *v);
xe_steering_mode xe_steering_get_mode(const xe_steering *v);
xe_steering_site xe_steering_get_site(const xe_steering *v);
int xe_steering_layers(const xe_steering *v);
int xe_steering_width(const xe_steering *v);
const float *xe_steering_row(const xe_steering *v, int layer);
const uint8_t *xe_steering_model(const xe_steering *v);
int xe_steering_active(const xe_steering *v, int layer);
void xe_steering_apply(float *h, const float *v, int width,
                       xe_steering_mode mode, float strength);
/* root must exist. Imported artifacts live under root/<id>/. */
int xe_steering_import(const char *root, const char *manifest, int layers, int width,
                       const uint8_t model[32], char id[65], char *error, size_t cap);
int xe_steering_open_id(const char *root, const char *id, int layers, int width,
                        const uint8_t model[32], xe_steering **out,
                        char *error, size_t cap);
int xe_steering_valid_id(const char *id);
void xe_steering_hex(const uint8_t bytes[32], char out[65]);
int xe_steering_unhex(const char *hex, uint8_t out[32]);
#endif
