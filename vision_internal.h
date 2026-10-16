#ifndef XE_VISION_INTERNAL_H
#define XE_VISION_INTERNAL_H

#include "vision.h"

typedef struct {
    char name[128];
    uint64_t ne[4], offset, bytes;
    uint32_t dimensions, type;
    const void *data;
} xe_vision_tensor;

typedef struct {
    void *mapping;
    size_t mapping_size, data_offset;
    xe_vision_tensor *tensors;
    size_t tensor_count;
    uint8_t sha256[32];
} xe_vision_model;

int xe_vision_model_open(xe_vision_model *out, const char *path,
                         char *error, size_t error_cap);
void xe_vision_model_close(xe_vision_model *model);
const xe_vision_tensor *xe_vision_tensor_get(const xe_vision_model *model,
                                            const char *name);

#endif
