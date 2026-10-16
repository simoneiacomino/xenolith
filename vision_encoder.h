#ifndef XE_VISION_ENCODER_H
#define XE_VISION_ENCODER_H

#include "vision_internal.h"
#include <level_zero/ze_api.h>

/* The engine owns the shared module and ordered command list. Only the seven
 * encoder kernel handles live here; the engine destroys them before its module.
 * Encoding and allocation run on the engine owner thread. */
typedef struct {
    ze_module_handle_t module;
    ze_command_list_handle_t commands;
    ze_kernel_handle_t kernels[7];
} xe_vision_gpu;

typedef struct xe_engine xe_engine;

/* Reuse the engine allocator and its device-limit policy. */
void *xe_vision_alloc_shared(xe_engine *engine, size_t bytes);
void xe_vision_free_shared(xe_engine *engine, void *pointer);

/* Inputs, model storage and GPU handles remain owned by the caller. The output
 * is copied back before return and no encoder work remains in flight. */
int xe_vision_encode(xe_engine *owner, xe_vision_gpu *gpu,
                     const xe_vision_model *model, const xe_image *image,
                     float *output, const char *dump_directory);

#endif
