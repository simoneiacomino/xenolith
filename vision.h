#ifndef XE_VISION_H
#define XE_VISION_H

#include <stddef.h>
#include <stdint.h>
#include <stdio.h>

#define XE_VISION_NUMERIC_REVISION "xe-gemma4v-f16gpu/v2-striped32"

enum { XE_VISUAL_MAX = 1120, XE_VISUAL_WIDTH = 2816,
       XE_IMAGE_RECORD_HEADER = 288, XE_IMAGE_RECORD_LIMIT = 48*1024*1024 };

typedef struct {
    unsigned char *encoded;
    size_t encoded_size;
    unsigned char *rgb;
    unsigned char *resized;
    int width, height, resized_width, resized_height;
    int budget, rows;
    uint8_t encoded_hash[32], rgb_hash[32], resized_hash[32], identity[32];
} xe_image;

typedef struct {
    int start, rows;
    const float *embeddings;
    uint8_t identity[32];
    const unsigned char *media;
    size_t media_size;
    int budget;
} xe_visual_span;

typedef struct {
    const int32_t *tokens;
    int count;
    const xe_visual_span *visual;
    size_t visual_count;
} xe_input;

/* Owned copy for replay; callers can release their image and embeddings. */
typedef struct {
    xe_input input;
    void *storage;
} xe_owned_input;

/* Reference status is diagnostic, not a compatibility or cache allowlist. */
int xe_vision_weights_reference(const uint8_t id[32], int projector);

int xe_image_size(int width, int height, int budget, int *out_width, int *out_height);
int xe_image_load(xe_image *out, const char *path, int budget,
                  char *error, size_t error_cap);
int xe_image_load_memory(xe_image *out, const unsigned char *bytes, size_t size,
                         int budget, char *error, size_t error_cap);
void xe_image_free(xe_image *image);
void xe_image_refresh_identity(xe_image *image);
int xe_image_record_parse(const void *data, size_t bytes, xe_image *metadata,
                           int *offset, const uint8_t **media, size_t *media_size);
int xe_input_layout_valid(const xe_input *input, int context);
int xe_input_valid(const xe_input *input, int context);
int xe_input_boundary(const xe_input *input, int position);
void xe_input_hash(const xe_input *input, uint8_t out[32]);
int xe_input_copy(xe_owned_input *out, const xe_input *input);
void xe_input_free(xe_owned_input *input);
/* Portable input archive, including original media; never path references. */
int xe_input_save(FILE *file, const xe_input *input);
int xe_input_load(FILE *file, int context, xe_owned_input *out);
void xe_visual_identity(const xe_image *image, const uint8_t model_hash[32],
                         const uint8_t projector_hash[32], const float *embeddings,
                         const char *numeric_revision, uint8_t out[32]);

#endif
