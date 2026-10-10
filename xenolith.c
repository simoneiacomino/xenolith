#define _GNU_SOURCE
#define _POSIX_C_SOURCE 200809L
#define _DEFAULT_SOURCE

#include "xenolith.h"
#include "xenolith_internal.h"
#include "format.h"

#include <errno.h>
#include <fcntl.h>
#include <immintrin.h>
#include <level_zero/ze_api.h>
#include <limits.h>
#include <math.h>
#include <pthread.h>
#include <sched.h>
#include <stdarg.h>
#include <stdatomic.h>
#include <stddef.h>
#include <stdlib.h>
#include <string.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <time.h>
#include <unistd.h>

#ifndef XE_NUMERIC_SOURCE_HASH
#define XE_NUMERIC_SOURCE_HASH "unknown"
#endif

#ifndef XE_NUMERIC_CFLAGS_HASH
#define XE_NUMERIC_CFLAGS_HASH "unknown"
#endif

static _Noreturn void xe_fatal(const char *fmt, ...) {
    va_list ap;
    fprintf(stderr, "xenolith: ");
    va_start(ap, fmt);
    vfprintf(stderr, fmt, ap);
    va_end(ap);
    fprintf(stderr, "\n");
    exit(1);
}

typedef enum { XE_MEM_HOST, XE_MEM_SHARED } xe_mem_kind;
typedef enum { XE_RELAXED_NONE, XE_RELAXED_EXP, XE_RELAXED_EXT } xe_relaxed_limits;

/* We centralize the allocation here because when implementing GPU prefill
 * the buffer touched by GPU should be allocated with zeMemAllocShared().
 * Memory allocated with plain malloc would not be visible by GPU. */
static void *xe_alloc(const xe_engine *e, size_t n, xe_mem_kind kind);
static void xe_free(const xe_engine *e, void *p, xe_mem_kind kind);

#ifdef XE_TEST_ALLOC
size_t xe_test_allocations;
#endif

#ifdef XE_TEST_OUTPUT_COUNT
size_t xe_test_output_calls;
#endif

#define XE_LAYERS 30
#define XE_EMBD 2816
#define XE_VOCAB 262144
#define XE_MODEL_CTX XE_CONTEXT_MAX
#define XE_Q_HEADS 16
#define XE_DENSE_FFN 2112
#define XE_EXPERTS 128
#define XE_EXPERTS_USED 8
#define XE_EXPERT_FFN 704
#define XE_SWA_WINDOW 1024
#define XE_SWA_KV_HEADS 8
#define XE_SWA_HEAD_DIM 256
#define XE_SWA_ROPE_BASE 10000.0f
#define XE_REPACK_THREADS 6
#ifndef XE_WORKERS
#define XE_WORKERS 6
#endif
#define XE_GLOBAL_KV_HEADS 2
#define XE_GLOBAL_HEAD_DIM 512
#define XE_GLOBAL_ROPE_BASE 1000000.0f
#define XE_GLOBAL_SHARD_CROSSOVER 32
#define XE_REPACK_PARTS 64
#define XE_REPACK_PART_LIMIT (UINT64_C(512) * 1024 * 1024)
#define XE_RMS_EPS 1e-6f
#define XE_LOGIT_SOFTCAP 30.0f
#define XE_TENSOR_COUNT 658
#define XE_MERGES 514906
#define XE_BYTE0_ID 238
#define XE_TOK_HASH 524288
#define XE_MERGE_HASH 1048576
#define XE_TT_NORMAL 1
#define XE_TT_BYTE 6
#define XE_CHANNEL_BEGIN_ID 100
#define XE_CHANNEL_END_ID 101
#define XE_TURN_BEGIN_ID 105
#define XE_TURN_END_ID 106
#define XE_CHAT_TEMPLATE_LEN 18924

#define XE_CHAT_TEMPLATE_HASH UINT64_C(0xe9f262823e5bda06)

#define XE_SWA_LAYER_ELEMS ((size_t)XE_SWA_KV_HEADS * XE_SWA_WINDOW * XE_SWA_HEAD_DIM)
#define XE_SWA_SLAB_ELEMS ((size_t)25 * XE_SWA_LAYER_ELEMS)
#define XE_GLOBAL_LAYERS 5

#define XE_IS_GLOBAL(i) ((i) % 6 == 5)

#define XE_GGUF_MAGIC 0x46554747u
#define XE_GGUF_VERSION 3

enum { XE_T_U8, XE_T_I8, XE_T_U16, XE_T_I16, XE_T_U32, XE_T_I32, XE_T_F32,
       XE_T_BOOL, XE_T_STRING, XE_T_ARRAY, XE_T_U64, XE_T_I64, XE_T_F64 };

#define XE_GGML_TYPE_F32 0
#define XE_GGML_TYPE_Q4_0 2

/* The binding loop in xe_bind_tensors() relies on this exact field order
 * to write the mmap pointer to this struct with a simple plain memcpy().
 * Do not reorder the fields. */
typedef struct {
    const uint8_t *qs;
    const uint16_t *d;
    int blocks;
} xe_q4;

typedef struct {
    int8_t *qs;
    _Float16 *d;
    int16_t *sigma;
    int n;
} xe_q8;

typedef struct {
    int32_t id;
    float logit;
    float prob;
} xe_sample_candidate;

typedef void (*xe_phase_fn)(xe_session *, const void *, int, int);

typedef struct {
    xe_phase_fn fn;
    xe_session *session;
    const void *arg;
    int caller_participates;
} xe_phase;

typedef struct {
    _Alignas(64) atomic_uint_fast64_t value;
} xe_phase_signal;

typedef struct {
    xe_engine *engine;
    int lane;
} xe_worker_arg;

typedef struct xe_layer {
    xe_q4 attn_q, attn_k, attn_v, attn_o, ffn_gate, ffn_up, ffn_down, gate_up_exps, down_exps;
    const float *attn_norm, *q_norm, *k_norm, *post_attn_norm, *ffn_norm, *post_ffw_norm, *pre_ffw_norm2, *post_ffw_norm1, *post_ffw_norm2, *router_w, *router_scale, *down_exps_scale, *layer_out_scale;
} xe_layer;

typedef struct {
    const char *p;
    uint64_t len;
} xe_str;

typedef struct {
    uint64_t key;
    uint32_t rank;
    int32_t res;
} xe_merge;

typedef struct {
    ze_driver_handle_t driver;
    ze_device_handle_t device;
    ze_context_handle_t context;
    uint64_t max_mem_alloc_size;
    xe_relaxed_limits relaxed_limits;
    ze_command_list_handle_t commands;
    ze_module_handle_t module;
    ze_kernel_handle_t prefill_rms_scale;
    ze_kernel_handle_t prefill_norm_q8;
    ze_kernel_handle_t prefill_q4q8_n32;
    ze_kernel_handle_t prefill_q4q8_n64;
    ze_kernel_handle_t prefill_q4q8_n128;
    ze_kernel_handle_t prefill_q4q8_kv_n128;
    ze_kernel_handle_t prefill_q4q8_swa_q_n128;
    ze_kernel_handle_t prefill_q4q8_swa_o_n128;
    ze_kernel_handle_t prefill_q4q8_global_q_n128;
    ze_kernel_handle_t prefill_q4q8_global_k_n128;
    ze_kernel_handle_t prefill_q4q8_global_o_n128;
    ze_kernel_handle_t prefill_q4q8_dense_down_n128;
    ze_kernel_handle_t prefill_q4q8_n128_tail;
    ze_kernel_handle_t prefill_qkv_post;
    ze_kernel_handle_t prefill_attn_online_b8;
    ze_kernel_handle_t prefill_attn_online_b8_global_shared;
    ze_kernel_handle_t prefill_attn_online_b8_global_cow;
    ze_kernel_handle_t prefill_attn_online_b8_swa;
    ze_kernel_handle_t prefill_heads_q8;
    ze_kernel_handle_t prefill_rms_residual;
    ze_kernel_handle_t prefill_swa_stage;
    ze_kernel_handle_t prefill_swa_commit;
    ze_kernel_handle_t prefill_ffn_input_q8;
    ze_kernel_handle_t prefill_geglu_q8;
    ze_kernel_handle_t prefill_router_gemm;
    ze_kernel_handle_t prefill_router_top8;
    ze_kernel_handle_t prefill_route_reset;
    ze_kernel_handle_t prefill_route_count;
    ze_kernel_handle_t prefill_route_prefix;
    ze_kernel_handle_t prefill_route_scatter;
    ze_kernel_handle_t prefill_route_pack;
    ze_kernel_handle_t prefill_q4q8_grouped_n128;
    ze_kernel_handle_t prefill_q4q8_grouped_gate_n128;
    ze_kernel_handle_t prefill_q4q8_grouped_down_n128;
    ze_kernel_handle_t prefill_q4q8_grouped_m16_n64;
    ze_kernel_handle_t prefill_q4q8_grouped_m16_n128;
    ze_kernel_handle_t prefill_q4q8_grouped_m8_n128;
    ze_kernel_handle_t prefill_expert_geglu_q8;
    ze_kernel_handle_t prefill_route_reduce;
    ze_kernel_handle_t prefill_ffn_finish;
} xe_gpu;

struct xe_engine {
    int context;
    void *map;
    size_t map_len;
    int map_fd;
    const void *data;
    size_t header_len;   /* bytes before the tensor data section: metadata + tensor directory */
    uint32_t gguf_version;
    char *name;

    uint64_t n_kv;
    uint64_t alignment;

    xe_q4 tok_embd;
    const float *out_norm;
    const float *rope_freqs;

    xe_layer layers[XE_LAYERS];

    const void *tok_tokens; uint64_t tok_tokens_count;
    const void *tok_scores; uint64_t tok_scores_count;
    const void *tok_token_type; uint64_t tok_token_type_count;
    const void *tok_merges; uint64_t tok_merges_count;
    xe_str chat_template;

    int32_t bos_id, eos_id, eot_id, unk_id, pad_id;

    xe_str *tok_piece;
    int32_t *tok_hash;
    xe_merge *tok_merge;
    uint32_t tok_max_piece;
    double tok_build_seconds;
    int vocab_only;

    uint64_t q4_0_count, q4_0_bytes;
    uint64_t f32_count, f32_bytes;
    uint64_t dense_bytes, experts_bytes, tok_embd_bytes;

    void *repack_parts[XE_REPACK_PARTS];
    uint64_t repack_part_sizes[XE_REPACK_PARTS];
    int repack_part_count;
    uint64_t repack_slab_size;
    uint64_t repack_nibble_bytes, repack_scale_bytes;
    double repack_seconds;
    uint64_t repack_verified_blocks;
    void *f32_slab;
    uint64_t f32_slab_size;

    xe_gpu gpu;

    pthread_t followers[XE_WORKERS - 1];
    xe_worker_arg worker_args[XE_WORKERS - 1];
    pthread_mutex_t worker_mutex;
    pthread_cond_t worker_cv;
    pthread_cond_t ready_cv;
    pthread_t owner;
    cpu_set_t owner_affinity;
    uint64_t worker_ticket;
    int worker_ready;
    int worker_stop;
    int pool_initialized;
    int worker_pinned;
    int token_active;
    xe_phase phase;
    atomic_uint_fast64_t phase_epoch;
    xe_phase_signal phase_done[XE_WORKERS];
    _Float16 *gelu_lut;
    float rope_swa_inv[XE_SWA_HEAD_DIM / 2];
    float rope_global_inv[XE_GLOBAL_HEAD_DIM / 2];
    int scalar_rms;
    uint8_t snapshot_fingerprint[6][32];
    int snapshot_fingerprint_ready;
};

struct xe_session {
    xe_engine *engine;
    _Float16 *swa_k;
    _Float16 *swa_v;
    _Float16 *swa_spare_k;
    _Float16 *swa_spare_v;
    uint8_t swa_dirty[XE_SWA_WINDOW];
    _Float16 *global_k;
    _Float16 *global_v;
    _Float16 *cow_global_k[XE_GLOBAL_LAYERS];
    _Float16 *cow_global_v[XE_GLOBAL_LAYERS];
    void *workspace;
    size_t workspace_size;
    void *prefill_workspace;
    size_t prefill_workspace_size;
    float *hidden;
    float *q;
    float *k;
    float *v;
    float *attn_heads;
    float *attn_proj;
    float *attn_out;
    float *dense_out;
    float *moe_out;
    float *combined;
    float *router_in;
    float *router_logits;
    float *scores;
    float *attn_partial;
    float *rope_swa_cos;
    float *rope_swa_sin;
    float *rope_global_cos;
    float *rope_global_sin;
    float *logits;
    xe_sample_candidate *sample_candidates;
    xe_q8 q8_main;
    xe_q8 q8_dense_in;
    xe_q8 q8_moe_in;
    xe_q8 q8_dense_act;
    xe_q8 q8_expert[XE_EXPERTS_USED];
    int experts[XE_EXPERTS_USED];
    float expert_weights[XE_EXPERTS_USED];
    int expert_trace[XE_LAYERS][XE_EXPERTS_USED];
    float expert_weight_trace[XE_LAYERS][XE_EXPERTS_USED];
    int32_t *tokens;
    int n_tokens;
    _Float16 *anchor_swa_k;
    _Float16 *anchor_swa_v;
    float *anchor_logits;
    int anchor_position;
    int anchor_valid;
    xe_session *cow_source;
    int cow;
    int cow_split;
    int cow_capacity;
    int prefill_pending;
    int prefill_pending_start;
    int prefill_pending_rows;
    float *prefill_pending_hidden;
};

static size_t xe_global_layer_elems(const xe_engine *e) {
    return (size_t)XE_GLOBAL_KV_HEADS * e->context * XE_GLOBAL_HEAD_DIM;
}

static size_t xe_global_slab_elems(const xe_engine *e) {
    return (size_t)XE_GLOBAL_LAYERS * xe_global_layer_elems(e);
}

extern const unsigned char _binary_xenolith_gpu_spv_start[];
extern const unsigned char _binary_xenolith_gpu_spv_end[];

static _Noreturn void xe_ze_fatal(const char *operation, ze_result_t result) {
    xe_fatal("%s failed: 0x%x", operation, result);
}

static void xe_ze_check(const char *operation, ze_result_t result) {
    if (result != ZE_RESULT_SUCCESS) xe_ze_fatal(operation, result);
}

static void *xe_alloc(const xe_engine *e, size_t n, xe_mem_kind kind) {
#ifdef XE_TEST_ALLOC
    extern size_t xe_test_allocations;
    xe_test_allocations++;
#endif
    if (n > SIZE_MAX - 63) xe_fatal("allocation size overflow: %zu", n);
    size_t rounded = (n + 63) / 64 * 64;
    if (kind == XE_MEM_HOST) {
        void *p = aligned_alloc(64, rounded);
        if (!p) xe_fatal("out of memory allocating %zu bytes", rounded);
        return p;
    }
    if (kind != XE_MEM_SHARED) xe_fatal("invalid memory kind %d", kind);
    if (!e || !e->gpu.context) xe_fatal("shared allocation without a GPU context");
    ze_device_mem_alloc_desc_t device_desc = {
        .stype = ZE_STRUCTURE_TYPE_DEVICE_MEM_ALLOC_DESC
    };
    ze_host_mem_alloc_desc_t host_desc = {
        .stype = ZE_STRUCTURE_TYPE_HOST_MEM_ALLOC_DESC
    };
    ze_relaxed_allocation_limits_exp_desc_t relaxed_exp = {
        .stype = ZE_STRUCTURE_TYPE_RELAXED_ALLOCATION_LIMITS_EXP_DESC,
        .flags = ZE_RELAXED_ALLOCATION_LIMITS_EXP_FLAG_MAX_SIZE
    };
#ifdef ZE_RELAXED_ALLOCATION_LIMITS_EXT_NAME
    ze_relaxed_allocation_limits_ext_desc_t relaxed_ext = {
        .stype = ZE_STRUCTURE_TYPE_RELAXED_ALLOCATION_LIMITS_EXT_DESC,
        .flags = ZE_RELAXED_ALLOCATION_LIMITS_EXT_FLAG_MAX_SIZE
    };
#endif
    if (rounded > e->gpu.max_mem_alloc_size) {
        if (e->gpu.relaxed_limits == XE_RELAXED_EXP)
            device_desc.pNext = &relaxed_exp;
#ifdef ZE_RELAXED_ALLOCATION_LIMITS_EXT_NAME
        else if (e->gpu.relaxed_limits == XE_RELAXED_EXT)
            device_desc.pNext = &relaxed_ext;
#endif
        else
            xe_fatal("shared allocation of %zu bytes exceeds device limit %llu; "
                     "driver does not support relaxed allocation limits",
                     rounded, (unsigned long long)e->gpu.max_mem_alloc_size);
    }
    void *p = NULL;
    ze_result_t result = zeMemAllocShared(e->gpu.context, &device_desc, &host_desc,
                                        rounded, 64, e->gpu.device, &p);
    if (result != ZE_RESULT_SUCCESS)
        xe_fatal("zeMemAllocShared failed: 0x%x allocating %zu bytes "
                 "(device limit %llu, relaxed limits %s)", result, rounded,
                 (unsigned long long)e->gpu.max_mem_alloc_size,
                 device_desc.pNext ? "enabled" : "disabled");
    return p;
}

static void xe_free(const xe_engine *e, void *p, xe_mem_kind kind) {
    if (!p) return;
    if (kind == XE_MEM_HOST) {
        free(p);
        return;
    }
    if (kind != XE_MEM_SHARED) xe_fatal("invalid memory kind %d", kind);
    if (!e || !e->gpu.context) xe_fatal("shared free without a GPU context");
    xe_ze_check("zeMemFree", zeMemFree(e->gpu.context, p));
}

typedef enum {
    XE_GPU_NOT_ENABLED,
    XE_GPU_TESTED,
    XE_GPU_UNTESTED
} xe_gpu_status;

static xe_gpu_status xe_gpu_classify(uint32_t id) {
    /* Integrated GPUs only. Keep these IDs and test status in sync with README.md.
     * IDs: intel/compute-runtime shared/source/dll/devices/devices_base.inl;
     * Wildcat Lake grouping: Linux include/drm/intel/pciids.h. */
    switch (id) {
        case 0xa7a0: /* Maintainer's Core i7-13700H. */
        case 0x7d51: /* Core Ultra 7 255H, reported by aziis98 in PR #2. */
        case 0x64a0: /* Core Ultra 7 268V; Raffaele's WSL report in PR #6, 64k capacity. */
            return XE_GPU_TESTED;
        /* Tiger Lake. */
        case 0x9a40: case 0x9a49: case 0x9a59: case 0x9a60:
        case 0x9a68: case 0x9a70: case 0x9a78:
        /* Rocket Lake. */
        case 0x4c80: case 0x4c8a: case 0x4c8b: case 0x4c8c:
        case 0x4c90: case 0x4c9a:
        /* Alder Lake mobile. */
        case 0x4626: case 0x4628: case 0x462a: case 0x46a0:
        case 0x46a1: case 0x46a3: case 0x46a6: case 0x46a8:
        case 0x46aa: case 0x46b0: case 0x46b1: case 0x46b3:
        case 0x46c0: case 0x46c1: case 0x46c3:
        /* Alder Lake desktop. */
        case 0x4680: case 0x4682: case 0x4688: case 0x468a:
        case 0x468b: case 0x4690: case 0x4692: case 0x4693:
        /* Alder Lake-N / Twin Lake. */
        case 0x46d0: case 0x46d1: case 0x46d2: case 0x46d3: case 0x46d4:
        /* Raptor Lake mobile / refresh. */
        case 0xa720: case 0xa721: case 0xa7a1: case 0xa7a8:
        case 0xa7a9: case 0xa7aa: case 0xa7ab: case 0xa7ac: case 0xa7ad:
        /* Raptor Lake desktop / refresh. */
        case 0xa780: case 0xa781: case 0xa782: case 0xa783:
        case 0xa788: case 0xa789: case 0xa78a: case 0xa78b:
        /* Meteor Lake and Arrow Lake. */
        case 0x7d40: case 0x7d45: case 0x7d55: case 0x7dd5:
        case 0x7d41: case 0x7d67: case 0x7dd1:
        /* Lunar Lake. */
        case 0x6420: case 0x64b0:
        /* Panther Lake. */
        case 0xb080: case 0xb081: case 0xb082: case 0xb083:
        case 0xb084: case 0xb085: case 0xb086: case 0xb087:
        case 0xb08f: case 0xb090: case 0xb0a0: case 0xb0b0:
        /* Wildcat Lake. */
        case 0xfd80: case 0xfd81:
        /* Nova Lake, Xe3 (IGFX_NVL_XE3G). */
        case 0xd740: case 0xd741: case 0xd742: case 0xd743:
        case 0xd744: case 0xd745:
        /* Nova Lake, Xe3P (IGFX_NVL). */
        case 0xd74a: case 0xd74b:
        case 0xd750: case 0xd751: case 0xd752: case 0xd753:
        case 0xd754: case 0xd755: case 0xd756: case 0xd757: case 0xd75f:
            return XE_GPU_UNTESTED;
        default:
            return XE_GPU_NOT_ENABLED;
    }
}

static xe_relaxed_limits xe_gpu_get_relaxed_limits(ze_driver_handle_t driver) {
    uint32_t count = 0;
    xe_ze_check("zeDriverGetExtensionProperties count",
                zeDriverGetExtensionProperties(driver, &count, NULL));
    if (!count) return XE_RELAXED_NONE;
    ze_driver_extension_properties_t *extensions =
        xe_alloc(NULL, (size_t)count * sizeof(*extensions), XE_MEM_HOST);
    xe_ze_check("zeDriverGetExtensionProperties",
                zeDriverGetExtensionProperties(driver, &count, extensions));
    xe_relaxed_limits limits = XE_RELAXED_NONE;
    for (uint32_t i = 0; i < count; i++) {
        if (extensions[i].version < ZE_MAKE_VERSION(1, 0)) continue;
        if (limits == XE_RELAXED_NONE &&
            strcmp(extensions[i].name, ZE_RELAXED_ALLOCATION_LIMITS_EXP_NAME) == 0)
            limits = XE_RELAXED_EXP;
#ifdef ZE_RELAXED_ALLOCATION_LIMITS_EXT_NAME
        if (strcmp(extensions[i].name, ZE_RELAXED_ALLOCATION_LIMITS_EXT_NAME) == 0)
            limits = XE_RELAXED_EXT;
#endif
    }
    xe_free(NULL, extensions, XE_MEM_HOST);
    return limits;
}

static void xe_gpu_init(xe_engine *e) {
    xe_ze_check("zeInit", zeInit(ZE_INIT_FLAG_GPU_ONLY));

    uint32_t driver_count = 0;
    xe_ze_check("zeDriverGet count", zeDriverGet(&driver_count, NULL));
    if (!driver_count) xe_fatal("no Level Zero GPU driver");
    driver_count = 1;
    xe_ze_check("zeDriverGet", zeDriverGet(&driver_count, &e->gpu.driver));

    uint32_t device_count = 0;
    xe_ze_check("zeDeviceGet count", zeDeviceGet(e->gpu.driver, &device_count, NULL));
    if (!device_count) xe_fatal("no Level Zero GPU device");
    device_count = 1;
    xe_ze_check("zeDeviceGet", zeDeviceGet(e->gpu.driver, &device_count, &e->gpu.device));

    ze_device_properties_t properties = {
        .stype = ZE_STRUCTURE_TYPE_DEVICE_PROPERTIES
    };
    xe_ze_check("zeDeviceGetProperties", zeDeviceGetProperties(e->gpu.device, &properties));
    e->gpu.max_mem_alloc_size = properties.maxMemAllocSize;
    e->gpu.relaxed_limits = xe_gpu_get_relaxed_limits(e->gpu.driver);
    xe_gpu_status status = xe_gpu_classify(properties.deviceId);
    if (properties.vendorId != 0x8086 || status == XE_GPU_NOT_ENABLED)
        xe_fatal("GPU %04x:%04x is not enabled; see README.md for Intel Xe integrated GPU device IDs",
                 properties.vendorId, properties.deviceId);
    if (status == XE_GPU_UNTESTED)
        fprintf(stderr, "xenolith: %s (%04x:%04x) is enabled but has not yet been tested with Xenolith; "
                        "using existing kernels without XMX.\n"
                        "xenolith: if inference works, please report your results so this device can be marked tested; "
                        "if it fails, please report the problem. Include your device ID, system/driver versions, "
                        "Xenolith commit, command and output: https://github.com/simoneiacomino/xenolith/issues\n",
                properties.name, properties.vendorId, properties.deviceId);

    ze_context_desc_t context_desc = {
        .stype = ZE_STRUCTURE_TYPE_CONTEXT_DESC
    };
    xe_ze_check("zeContextCreate",
                zeContextCreate(e->gpu.driver, &context_desc, &e->gpu.context));

    ze_command_queue_desc_t queue_desc = {
        .stype = ZE_STRUCTURE_TYPE_COMMAND_QUEUE_DESC,
        .ordinal = 0,
        .index = 0,
        .flags = ZE_COMMAND_QUEUE_FLAG_IN_ORDER,
        .mode = ZE_COMMAND_QUEUE_MODE_ASYNCHRONOUS,
        .priority = ZE_COMMAND_QUEUE_PRIORITY_NORMAL
    };
    xe_ze_check("zeCommandListCreateImmediate",
                zeCommandListCreateImmediate(e->gpu.context, e->gpu.device,
                                             &queue_desc, &e->gpu.commands));
    size_t spv_size = (size_t)(_binary_xenolith_gpu_spv_end -
                               _binary_xenolith_gpu_spv_start);
    ze_module_desc_t module_desc = {
        .stype = ZE_STRUCTURE_TYPE_MODULE_DESC,
        .format = ZE_MODULE_FORMAT_IL_SPIRV,
        .inputSize = spv_size,
        .pInputModule = _binary_xenolith_gpu_spv_start,
        .pBuildFlags = ""
    };
    ze_module_build_log_handle_t log = NULL;
    ze_result_t result = zeModuleCreate(e->gpu.context, e->gpu.device, &module_desc,
                                        &e->gpu.module, &log);
    if (result != ZE_RESULT_SUCCESS) {
        if (log) {
            size_t log_size = 0;
            zeModuleBuildLogGetString(log, &log_size, NULL);
            char *message = xe_alloc(e, log_size + 1, XE_MEM_HOST);
            zeModuleBuildLogGetString(log, &log_size, message);
            message[log_size] = 0;
            fprintf(stderr, "%s\n", message);
            xe_free(e, message, XE_MEM_HOST);
        }
        xe_ze_fatal("zeModuleCreate", result);
    }
    if (log) xe_ze_check("zeModuleBuildLogDestroy", zeModuleBuildLogDestroy(log));

    const char *prefill_names[40] = {
        "xe_prefill_rms_scale",
        "xe_prefill_norm_q8",
        "xe_prefill_q4q8_n32",
        "xe_prefill_q4q8_n64",
        "xe_prefill_q4q8_n128",
        "xe_prefill_q4q8_kv_n128",
        "xe_prefill_q4q8_swa_q_n128",
        "xe_prefill_q4q8_swa_o_n128",
        "xe_prefill_q4q8_global_q_n128",
        "xe_prefill_q4q8_global_k_n128",
        "xe_prefill_q4q8_global_o_n128",
        "xe_prefill_q4q8_dense_down_n128",
        "xe_prefill_q4q8_n128_tail",
        "xe_prefill_qkv_post",
        "xe_prefill_attn_online_b8",
        "xe_prefill_attn_online_b8_global_shared",
        "xe_prefill_attn_online_b8_global_cow",
        "xe_prefill_attn_online_b8_swa",
        "xe_prefill_heads_q8",
        "xe_prefill_rms_residual",
        "xe_prefill_swa_stage",
        "xe_prefill_swa_commit",
        "xe_prefill_ffn_input_q8",
        "xe_prefill_geglu_q8",
        "xe_prefill_router_gemm",
        "xe_prefill_router_top8",
        "xe_prefill_route_reset",
        "xe_prefill_route_count",
        "xe_prefill_route_prefix",
        "xe_prefill_route_scatter",
        "xe_prefill_route_pack",
        "xe_prefill_q4q8_grouped_n128",
        "xe_prefill_q4q8_grouped_gate_n128",
        "xe_prefill_q4q8_grouped_down_n128",
        "xe_prefill_q4q8_grouped_m16_n64",
        "xe_prefill_q4q8_grouped_m16_n128",
        "xe_prefill_q4q8_grouped_m8_n128",
        "xe_prefill_expert_geglu_q8",
        "xe_prefill_route_reduce",
        "xe_prefill_ffn_finish"
    };
    ze_kernel_handle_t *prefill_handles[40] = {
        &e->gpu.prefill_rms_scale,
        &e->gpu.prefill_norm_q8,
        &e->gpu.prefill_q4q8_n32,
        &e->gpu.prefill_q4q8_n64,
        &e->gpu.prefill_q4q8_n128,
        &e->gpu.prefill_q4q8_kv_n128,
        &e->gpu.prefill_q4q8_swa_q_n128,
        &e->gpu.prefill_q4q8_swa_o_n128,
        &e->gpu.prefill_q4q8_global_q_n128,
        &e->gpu.prefill_q4q8_global_k_n128,
        &e->gpu.prefill_q4q8_global_o_n128,
        &e->gpu.prefill_q4q8_dense_down_n128,
        &e->gpu.prefill_q4q8_n128_tail,
        &e->gpu.prefill_qkv_post,
        &e->gpu.prefill_attn_online_b8,
        &e->gpu.prefill_attn_online_b8_global_shared,
        &e->gpu.prefill_attn_online_b8_global_cow,
        &e->gpu.prefill_attn_online_b8_swa,
        &e->gpu.prefill_heads_q8,
        &e->gpu.prefill_rms_residual,
        &e->gpu.prefill_swa_stage,
        &e->gpu.prefill_swa_commit,
        &e->gpu.prefill_ffn_input_q8,
        &e->gpu.prefill_geglu_q8,
        &e->gpu.prefill_router_gemm,
        &e->gpu.prefill_router_top8,
        &e->gpu.prefill_route_reset,
        &e->gpu.prefill_route_count,
        &e->gpu.prefill_route_prefix,
        &e->gpu.prefill_route_scatter,
        &e->gpu.prefill_route_pack,
        &e->gpu.prefill_q4q8_grouped_n128,
        &e->gpu.prefill_q4q8_grouped_gate_n128,
        &e->gpu.prefill_q4q8_grouped_down_n128,
        &e->gpu.prefill_q4q8_grouped_m16_n64,
        &e->gpu.prefill_q4q8_grouped_m16_n128,
        &e->gpu.prefill_q4q8_grouped_m8_n128,
        &e->gpu.prefill_expert_geglu_q8,
        &e->gpu.prefill_route_reduce,
        &e->gpu.prefill_ffn_finish
    };
    uint32_t prefill_group_sizes[40] = {
        128, 128, 128, 128, 256, 256, 256, 256, 256, 256, 256, 256, 256, 128, 128, 128, 128, 128, 128, 128, 256, 256, 128, 128,
        128, 128, 128, 128, 128, 128, 128, 256, 256, 256, 128, 128, 128, 128, 128,
        128
    };
    for (int i = 0; i < 40; i++) {
        ze_kernel_desc_t desc = {
            .stype = ZE_STRUCTURE_TYPE_KERNEL_DESC,
            .pKernelName = prefill_names[i]
        };
        xe_ze_check("zeKernelCreate prefill",
                    zeKernelCreate(e->gpu.module, &desc, prefill_handles[i]));
        xe_ze_check("zeKernelSetGroupSize prefill",
                    zeKernelSetGroupSize(*prefill_handles[i],
                                         prefill_group_sizes[i], 1, 1));
    }
    ze_kernel_desc_t kernel_desc = {
        .stype = ZE_STRUCTURE_TYPE_KERNEL_DESC,
        .pKernelName = "xe_probe"
    };
    ze_kernel_handle_t probe;
    xe_ze_check("zeKernelCreate probe",
                zeKernelCreate(e->gpu.module, &kernel_desc, &probe));
    xe_ze_check("zeKernelSetGroupSize probe", zeKernelSetGroupSize(probe, 1, 1, 1));
    uint32_t *value = xe_alloc(e, sizeof(*value), XE_MEM_SHARED);
    *value = 0;
    xe_ze_check("zeKernelSetArgumentValue probe",
                zeKernelSetArgumentValue(probe, 0, sizeof(value), &value));
    ze_group_count_t groups = { 1, 1, 1 };
    xe_ze_check("zeCommandListAppendLaunchKernel probe",
                zeCommandListAppendLaunchKernel(e->gpu.commands, probe, &groups,
                                                NULL, 0, NULL));
    xe_ze_check("zeCommandListHostSynchronize probe",
                zeCommandListHostSynchronize(e->gpu.commands, UINT64_MAX));
    if (*value != UINT32_C(0x78656e6f))
        xe_fatal("GPU probe returned 0x%08x", *value);
    xe_free(e, value, XE_MEM_SHARED);
    xe_ze_check("zeKernelDestroy probe", zeKernelDestroy(probe));
}

static void xe_gpu_destroy(xe_engine *e) {
    if (e->gpu.commands)
        xe_ze_check("zeCommandListHostSynchronize close",
                    zeCommandListHostSynchronize(e->gpu.commands, UINT64_MAX));
    if (e->gpu.prefill_ffn_finish)
        xe_ze_check("zeKernelDestroy prefill ffn finish",
                    zeKernelDestroy(e->gpu.prefill_ffn_finish));
    if (e->gpu.prefill_route_reduce)
        xe_ze_check("zeKernelDestroy prefill route reduce",
                    zeKernelDestroy(e->gpu.prefill_route_reduce));
    if (e->gpu.prefill_expert_geglu_q8)
        xe_ze_check("zeKernelDestroy prefill expert geglu q8",
                    zeKernelDestroy(e->gpu.prefill_expert_geglu_q8));
    if (e->gpu.prefill_q4q8_grouped_n128)
        xe_ze_check("zeKernelDestroy prefill grouped q4q8 n64",
                    zeKernelDestroy(e->gpu.prefill_q4q8_grouped_n128));
    if (e->gpu.prefill_q4q8_grouped_gate_n128)
        xe_ze_check("zeKernelDestroy prefill grouped gate q4q8 n128",
                    zeKernelDestroy(e->gpu.prefill_q4q8_grouped_gate_n128));
    if (e->gpu.prefill_q4q8_grouped_down_n128)
        xe_ze_check("zeKernelDestroy prefill grouped down q4q8 n128",
                    zeKernelDestroy(e->gpu.prefill_q4q8_grouped_down_n128));
    if (e->gpu.prefill_q4q8_grouped_m16_n64)
        xe_ze_check("zeKernelDestroy prefill grouped q4q8 m16 n64",
                    zeKernelDestroy(e->gpu.prefill_q4q8_grouped_m16_n64));
    if (e->gpu.prefill_q4q8_grouped_m16_n128)
        xe_ze_check("zeKernelDestroy prefill grouped q4q8 m16 n128",
                    zeKernelDestroy(e->gpu.prefill_q4q8_grouped_m16_n128));
    if (e->gpu.prefill_q4q8_grouped_m8_n128)
        xe_ze_check("zeKernelDestroy prefill grouped q4q8 m8 n128",
                    zeKernelDestroy(e->gpu.prefill_q4q8_grouped_m8_n128));
    if (e->gpu.prefill_route_pack)
        xe_ze_check("zeKernelDestroy prefill route pack",
                    zeKernelDestroy(e->gpu.prefill_route_pack));
    if (e->gpu.prefill_route_scatter)
        xe_ze_check("zeKernelDestroy prefill route scatter",
                    zeKernelDestroy(e->gpu.prefill_route_scatter));
    if (e->gpu.prefill_route_prefix)
        xe_ze_check("zeKernelDestroy prefill route prefix",
                    zeKernelDestroy(e->gpu.prefill_route_prefix));
    if (e->gpu.prefill_route_count)
        xe_ze_check("zeKernelDestroy prefill route count",
                    zeKernelDestroy(e->gpu.prefill_route_count));
    if (e->gpu.prefill_route_reset)
        xe_ze_check("zeKernelDestroy prefill route reset",
                    zeKernelDestroy(e->gpu.prefill_route_reset));
    if (e->gpu.prefill_router_top8)
        xe_ze_check("zeKernelDestroy prefill router top8",
                    zeKernelDestroy(e->gpu.prefill_router_top8));
    if (e->gpu.prefill_router_gemm)
        xe_ze_check("zeKernelDestroy prefill router gemm",
                    zeKernelDestroy(e->gpu.prefill_router_gemm));
    if (e->gpu.prefill_geglu_q8)
        xe_ze_check("zeKernelDestroy prefill geglu q8",
                    zeKernelDestroy(e->gpu.prefill_geglu_q8));
    if (e->gpu.prefill_ffn_input_q8)
        xe_ze_check("zeKernelDestroy prefill ffn input q8",
                    zeKernelDestroy(e->gpu.prefill_ffn_input_q8));
    if (e->gpu.prefill_swa_commit)
        xe_ze_check("zeKernelDestroy prefill swa commit",
                    zeKernelDestroy(e->gpu.prefill_swa_commit));
    if (e->gpu.prefill_swa_stage)
        xe_ze_check("zeKernelDestroy prefill swa stage",
                    zeKernelDestroy(e->gpu.prefill_swa_stage));
    if (e->gpu.prefill_rms_residual)
        xe_ze_check("zeKernelDestroy prefill rms residual",
                    zeKernelDestroy(e->gpu.prefill_rms_residual));
    if (e->gpu.prefill_heads_q8)
        xe_ze_check("zeKernelDestroy prefill heads q8",
                    zeKernelDestroy(e->gpu.prefill_heads_q8));
    if (e->gpu.prefill_attn_online_b8)
        xe_ze_check("zeKernelDestroy prefill attention online b8",
                    zeKernelDestroy(e->gpu.prefill_attn_online_b8));
    if (e->gpu.prefill_attn_online_b8_global_shared)
        xe_ze_check("zeKernelDestroy prefill attention online b8 global shared",
                    zeKernelDestroy(e->gpu.prefill_attn_online_b8_global_shared));
    if (e->gpu.prefill_attn_online_b8_global_cow)
        xe_ze_check("zeKernelDestroy prefill attention online b8 global cow",
                    zeKernelDestroy(e->gpu.prefill_attn_online_b8_global_cow));
    if (e->gpu.prefill_attn_online_b8_swa)
        xe_ze_check("zeKernelDestroy prefill attention online b8 swa",
                    zeKernelDestroy(e->gpu.prefill_attn_online_b8_swa));
    if (e->gpu.prefill_qkv_post)
        xe_ze_check("zeKernelDestroy prefill qkv post",
                    zeKernelDestroy(e->gpu.prefill_qkv_post));
    if (e->gpu.prefill_q4q8_n64)
        xe_ze_check("zeKernelDestroy prefill q4q8 n64",
                    zeKernelDestroy(e->gpu.prefill_q4q8_n64));
    if (e->gpu.prefill_q4q8_n128)
        xe_ze_check("zeKernelDestroy prefill q4q8 n128",
                    zeKernelDestroy(e->gpu.prefill_q4q8_n128));
    if (e->gpu.prefill_q4q8_kv_n128)
        xe_ze_check("zeKernelDestroy prefill q4q8 kv n128",
                    zeKernelDestroy(e->gpu.prefill_q4q8_kv_n128));
    if (e->gpu.prefill_q4q8_swa_q_n128)
        xe_ze_check("zeKernelDestroy prefill q4q8 swa q n128",
                    zeKernelDestroy(e->gpu.prefill_q4q8_swa_q_n128));
    if (e->gpu.prefill_q4q8_swa_o_n128)
        xe_ze_check("zeKernelDestroy prefill q4q8 swa o n128",
                    zeKernelDestroy(e->gpu.prefill_q4q8_swa_o_n128));
    if (e->gpu.prefill_q4q8_global_q_n128)
        xe_ze_check("zeKernelDestroy prefill q4q8 global q n128",
                    zeKernelDestroy(e->gpu.prefill_q4q8_global_q_n128));
    if (e->gpu.prefill_q4q8_global_k_n128)
        xe_ze_check("zeKernelDestroy prefill q4q8 global k n128",
                    zeKernelDestroy(e->gpu.prefill_q4q8_global_k_n128));
    if (e->gpu.prefill_q4q8_global_o_n128)
        xe_ze_check("zeKernelDestroy prefill q4q8 global o n128",
                    zeKernelDestroy(e->gpu.prefill_q4q8_global_o_n128));
    if (e->gpu.prefill_q4q8_dense_down_n128)
        xe_ze_check("zeKernelDestroy prefill q4q8 dense down n128",
                    zeKernelDestroy(e->gpu.prefill_q4q8_dense_down_n128));
    if (e->gpu.prefill_q4q8_n128_tail)
        xe_ze_check("zeKernelDestroy prefill q4q8 n128 tail",
                    zeKernelDestroy(e->gpu.prefill_q4q8_n128_tail));
    if (e->gpu.prefill_q4q8_n32)
        xe_ze_check("zeKernelDestroy prefill q4q8 n32",
                    zeKernelDestroy(e->gpu.prefill_q4q8_n32));
    if (e->gpu.prefill_norm_q8)
        xe_ze_check("zeKernelDestroy prefill norm q8",
                    zeKernelDestroy(e->gpu.prefill_norm_q8));
    if (e->gpu.prefill_rms_scale)
        xe_ze_check("zeKernelDestroy prefill rms scale",
                    zeKernelDestroy(e->gpu.prefill_rms_scale));
    if (e->gpu.module) xe_ze_check("zeModuleDestroy", zeModuleDestroy(e->gpu.module));
    if (e->gpu.commands)
        xe_ze_check("zeCommandListDestroy", zeCommandListDestroy(e->gpu.commands));
    if (e->gpu.context) xe_ze_check("zeContextDestroy", zeContextDestroy(e->gpu.context));
    memset(&e->gpu, 0, sizeof e->gpu);
}

static void xe_gpu_pointer_arg(ze_kernel_handle_t kernel, uint32_t index,
                               const void *pointer) {
    xe_ze_check("zeKernelSetArgumentValue pointer",
                zeKernelSetArgumentValue(kernel, index, sizeof(pointer),
                                         &pointer));
}

static void xe_gpu_int_arg(ze_kernel_handle_t kernel, uint32_t index,
                           int value) {
    xe_ze_check("zeKernelSetArgumentValue int",
                zeKernelSetArgumentValue(kernel, index, sizeof(value), &value));
}

static void xe_gpu_float_arg(ze_kernel_handle_t kernel, uint32_t index,
                             float value) {
    xe_ze_check("zeKernelSetArgumentValue float",
                zeKernelSetArgumentValue(kernel, index, sizeof(value), &value));
}

static void __attribute__((unused)) xe_prefill_rms_append(
        xe_engine *e, const float *input, float *row_scale,
        int rows, int width) {
    ze_kernel_handle_t rms = e->gpu.prefill_rms_scale;
    xe_gpu_pointer_arg(rms, 0, input);
    xe_gpu_pointer_arg(rms, 1, row_scale);
    xe_gpu_int_arg(rms, 2, rows);
    xe_gpu_int_arg(rms, 3, width);
    xe_gpu_float_arg(rms, 4, XE_RMS_EPS);
    ze_group_count_t groups = { (uint32_t)rows, 1, 1 };
    xe_ze_check("zeCommandListAppendLaunchKernel prefill rms",
                zeCommandListAppendLaunchKernel(e->gpu.commands, rms,
                                                &groups, NULL, 0, NULL));
}

static void __attribute__((unused)) xe_prefill_input_append(xe_engine *e, const float *input,
                                    const float *weight, float *row_scale,
                                    xe_q8 *q8, int rows, int width) {
    xe_prefill_rms_append(e, input, row_scale, rows, width);

    ze_kernel_handle_t norm = e->gpu.prefill_norm_q8;
    xe_gpu_pointer_arg(norm, 0, input);
    xe_gpu_pointer_arg(norm, 1, weight);
    xe_gpu_pointer_arg(norm, 2, row_scale);
    xe_gpu_pointer_arg(norm, 3, q8->qs);
    xe_gpu_pointer_arg(norm, 4, q8->d);
    xe_gpu_pointer_arg(norm, 5, q8->sigma);
    xe_gpu_int_arg(norm, 6, rows);
    xe_gpu_int_arg(norm, 7, width);
    ze_group_count_t norm_groups = {
        (uint32_t)(((size_t)rows * (width / 32) + 7) / 8), 1, 1
    };
    xe_ze_check("zeCommandListAppendLaunchKernel prefill norm q8",
                zeCommandListAppendLaunchKernel(e->gpu.commands, norm,
                                                &norm_groups, NULL, 0, NULL));
}

static void __attribute__((unused)) xe_prefill_projection_append(xe_engine *e, const xe_q4 *weight,
                                         const xe_q8 *input, float *output,
                                         int rows, int columns, int n64) {
    int n128_tail = n64 && columns == XE_DENSE_FFN;
    int n128 = n64 && !(columns & 127);
    ze_kernel_handle_t kernel = n128 && rows == 512 && columns == 2048
                                         && weight->blocks == 88
                              ? e->gpu.prefill_q4q8_kv_n128
                              : n128 && rows == 512 && columns == 4096
                                         && weight->blocks == 88
                              ? e->gpu.prefill_q4q8_swa_q_n128
                              : n128 && rows == 512 && columns == 2816
                                         && weight->blocks == 128
                              ? e->gpu.prefill_q4q8_swa_o_n128
                              : n128 && rows == 512 && columns == 8192
                                         && weight->blocks == 88
                              ? e->gpu.prefill_q4q8_global_q_n128
                              : n128 && rows == 512 && columns == 1024
                                         && weight->blocks == 88
                              ? e->gpu.prefill_q4q8_global_k_n128
                              : n128 && rows == 512 && columns == 2816
                                         && weight->blocks == 256
                              ? e->gpu.prefill_q4q8_global_o_n128
                              : n128 && rows == 512 && columns == 2816
                                         && weight->blocks == 66
                              ? e->gpu.prefill_q4q8_dense_down_n128
                              : n128_tail ? e->gpu.prefill_q4q8_n128_tail
                              : n128 ? e->gpu.prefill_q4q8_n128
                              : n64 ? e->gpu.prefill_q4q8_n64
                                    : e->gpu.prefill_q4q8_n32;
    xe_gpu_pointer_arg(kernel, 0, weight->qs);
    xe_gpu_pointer_arg(kernel, 1, weight->d);
    xe_gpu_pointer_arg(kernel, 2, input->qs);
    xe_gpu_pointer_arg(kernel, 3, input->d);
    xe_gpu_pointer_arg(kernel, 4, input->sigma);
    xe_gpu_pointer_arg(kernel, 5, output);
    xe_gpu_int_arg(kernel, 6, rows);
    xe_gpu_int_arg(kernel, 7, columns);
    xe_gpu_int_arg(kernel, 8, weight->blocks);
    ze_group_count_t groups = {
        (uint32_t)((rows + 31) / 32),
        (uint32_t)(n128_tail ? (columns + 127) / 128
                             : columns / (n128 ? 128 : n64 ? 64 : 32)), 1
    };
    xe_ze_check("zeCommandListAppendLaunchKernel prefill projection",
                zeCommandListAppendLaunchKernel(e->gpu.commands, kernel,
                                                &groups, NULL, 0, NULL));
}

static void __attribute__((unused)) xe_prefill_qkv_append(
        xe_engine *e, const float *q_projection, const float *k_projection,
        const float *v_projection, const float *q_weight,
        const float *k_weight, const float *rope_cos, const float *rope_sin,
        float *q, _Float16 *k, _Float16 *v, int rows, int dimension,
        int kv_heads, int has_v) {
    ze_kernel_handle_t kernel = e->gpu.prefill_qkv_post;
    xe_gpu_pointer_arg(kernel, 0, q_projection);
    xe_gpu_pointer_arg(kernel, 1, k_projection);
    xe_gpu_pointer_arg(kernel, 2, v_projection);
    xe_gpu_pointer_arg(kernel, 3, q_weight);
    xe_gpu_pointer_arg(kernel, 4, k_weight);
    xe_gpu_pointer_arg(kernel, 5, rope_cos);
    xe_gpu_pointer_arg(kernel, 6, rope_sin);
    xe_gpu_pointer_arg(kernel, 7, q);
    xe_gpu_pointer_arg(kernel, 8, k);
    xe_gpu_pointer_arg(kernel, 9, v);
    xe_gpu_int_arg(kernel, 10, rows);
    xe_gpu_int_arg(kernel, 11, dimension);
    xe_gpu_int_arg(kernel, 12, kv_heads);
    xe_gpu_int_arg(kernel, 13, has_v);
    xe_gpu_float_arg(kernel, 14, XE_RMS_EPS);
    ze_group_count_t groups = {
        (uint32_t)((16 + 2 * kv_heads) * rows), 1, 1
    };
    xe_ze_check("zeCommandListAppendLaunchKernel prefill qkv post",
                zeCommandListAppendLaunchKernel(e->gpu.commands, kernel,
                                                &groups, NULL, 0, NULL));
}

static void __attribute__((unused)) xe_prefill_attention_online_append(
        xe_engine *e, const float *q, const _Float16 *k, const _Float16 *v,
        float *output, int rows, int keys, int dimension, int kv_heads,
        int query_offset, int window) {
    ze_kernel_handle_t kernel = dimension == 256
                                ? e->gpu.prefill_attn_online_b8_swa
                                : e->gpu.prefill_attn_online_b8_global_shared;
    xe_gpu_pointer_arg(kernel, 0, q);
    xe_gpu_pointer_arg(kernel, 1, k);
    xe_gpu_pointer_arg(kernel, 2, v);
    xe_gpu_pointer_arg(kernel, 3, output);
    xe_gpu_int_arg(kernel, 4, rows);
    xe_gpu_int_arg(kernel, 5, keys);
    xe_gpu_int_arg(kernel, 6, dimension);
    xe_gpu_int_arg(kernel, 7, XE_Q_HEADS);
    xe_gpu_int_arg(kernel, 8, kv_heads);
    xe_gpu_int_arg(kernel, 9, query_offset);
    xe_gpu_int_arg(kernel, 10, window);
    if (dimension == 256) {
        ze_group_count_t groups = {
            (uint32_t)(((size_t)XE_Q_HEADS * rows + 7) / 8), 1, 1
        };
        xe_ze_check("zeCommandListAppendLaunchKernel prefill attention online",
                    zeCommandListAppendLaunchKernel(e->gpu.commands, kernel,
                                                    &groups, NULL, 0, NULL));
    } else {
        int slice_rows = query_offset >= 8192 ? 256 : rows;
        for (int query_base = 0; query_base < rows;
             query_base += slice_rows) {
            int query_count = rows - query_base;
            if (query_count > slice_rows) query_count = slice_rows;
            xe_gpu_int_arg(kernel, 6, query_count);
            xe_gpu_int_arg(kernel, 10, query_base);
            ze_group_count_t groups = {
                (uint32_t)(kv_heads * query_count), 1, 1
            };
            xe_ze_check("zeCommandListAppendLaunchKernel prefill attention online",
                        zeCommandListAppendLaunchKernel(e->gpu.commands, kernel,
                                                        &groups, NULL, 0, NULL));
            if (slice_rows < rows)
                xe_ze_check("zeCommandListHostSynchronize prefill attention slice",
                            zeCommandListHostSynchronize(e->gpu.commands,
                                                         UINT64_MAX));
        }
    }
}

static void xe_prefill_attention_cow_append(
        xe_engine *e, const float *q, const _Float16 *prefix_k,
        const _Float16 *prefix_v, const _Float16 *tail_k,
        const _Float16 *tail_v, float *output, int rows,
        int prefix_capacity, int tail_capacity, int query_offset, int split) {
    ze_kernel_handle_t kernel = e->gpu.prefill_attn_online_b8_global_cow;
    xe_gpu_pointer_arg(kernel, 0, q);
    xe_gpu_pointer_arg(kernel, 1, prefix_k);
    xe_gpu_pointer_arg(kernel, 2, prefix_v);
    xe_gpu_pointer_arg(kernel, 3, tail_k);
    xe_gpu_pointer_arg(kernel, 4, tail_v);
    xe_gpu_pointer_arg(kernel, 5, output);
    xe_gpu_int_arg(kernel, 6, rows);
    xe_gpu_int_arg(kernel, 7, prefix_capacity);
    xe_gpu_int_arg(kernel, 8, tail_capacity);
    xe_gpu_int_arg(kernel, 10, XE_Q_HEADS);
    xe_gpu_int_arg(kernel, 11, XE_GLOBAL_KV_HEADS);
    xe_gpu_int_arg(kernel, 12, query_offset);
    xe_gpu_int_arg(kernel, 14, split);
    int slice_rows = query_offset >= 8192 ? 256 : rows;
    for (int query_base = 0; query_base < rows;
         query_base += slice_rows) {
        int query_count = rows - query_base;
        if (query_count > slice_rows) query_count = slice_rows;
        xe_gpu_int_arg(kernel, 9, query_count);
        xe_gpu_int_arg(kernel, 13, query_base);
        ze_group_count_t groups = {
            (uint32_t)(XE_GLOBAL_KV_HEADS * query_count), 1, 1
        };
        xe_ze_check("zeCommandListAppendLaunchKernel prefill attention cow",
                    zeCommandListAppendLaunchKernel(e->gpu.commands, kernel,
                                                    &groups, NULL, 0, NULL));
        if (slice_rows < rows)
            xe_ze_check("zeCommandListHostSynchronize prefill attention cow slice",
                        zeCommandListHostSynchronize(e->gpu.commands,
                                                     UINT64_MAX));
    }
}

static void __attribute__((unused)) xe_prefill_heads_q8_append(
        xe_engine *e, const float *heads, xe_q8 *output, int rows,
        int dimension) {
    ze_kernel_handle_t kernel = e->gpu.prefill_heads_q8;
    xe_gpu_pointer_arg(kernel, 0, heads);
    xe_gpu_pointer_arg(kernel, 1, output->qs);
    xe_gpu_pointer_arg(kernel, 2, output->d);
    xe_gpu_pointer_arg(kernel, 3, output->sigma);
    xe_gpu_int_arg(kernel, 4, rows);
    xe_gpu_int_arg(kernel, 5, XE_Q_HEADS);
    xe_gpu_int_arg(kernel, 6, dimension);
    int blocks = XE_Q_HEADS * dimension / 32;
    ze_group_count_t groups = {
        (uint32_t)(((size_t)rows * blocks + 7) / 8), 1, 1
    };
    xe_ze_check("zeCommandListAppendLaunchKernel prefill heads q8",
                zeCommandListAppendLaunchKernel(e->gpu.commands, kernel,
                                                &groups, NULL, 0, NULL));
}

static void __attribute__((unused)) xe_prefill_rms_residual_append(
        xe_engine *e, const float *input, const float *weight,
        const float *residual, float *output, int rows, int width) {
    ze_kernel_handle_t kernel = e->gpu.prefill_rms_residual;
    xe_gpu_pointer_arg(kernel, 0, input);
    xe_gpu_pointer_arg(kernel, 1, weight);
    xe_gpu_pointer_arg(kernel, 2, residual);
    xe_gpu_pointer_arg(kernel, 3, output);
    xe_gpu_int_arg(kernel, 4, rows);
    xe_gpu_int_arg(kernel, 5, width);
    xe_gpu_float_arg(kernel, 6, XE_RMS_EPS);
    ze_group_count_t groups = { (uint32_t)rows, 1, 1 };
    xe_ze_check("zeCommandListAppendLaunchKernel prefill rms residual",
                zeCommandListAppendLaunchKernel(e->gpu.commands, kernel,
                                                &groups, NULL, 0, NULL));
}

static void __attribute__((unused)) xe_prefill_swa_stage_append(
        xe_engine *e, const _Float16 *ring_k, const _Float16 *ring_v,
        const _Float16 *batch_k, const _Float16 *batch_v,
        _Float16 *stage_k, _Float16 *stage_v, int rows, int batch_start,
        int stage_base, int stage_count) {
    ze_kernel_handle_t kernel = e->gpu.prefill_swa_stage;
    xe_gpu_pointer_arg(kernel, 0, ring_k);
    xe_gpu_pointer_arg(kernel, 1, ring_v);
    xe_gpu_pointer_arg(kernel, 2, batch_k);
    xe_gpu_pointer_arg(kernel, 3, batch_v);
    xe_gpu_pointer_arg(kernel, 4, stage_k);
    xe_gpu_pointer_arg(kernel, 5, stage_v);
    xe_gpu_int_arg(kernel, 6, rows);
    xe_gpu_int_arg(kernel, 7, XE_SWA_HEAD_DIM);
    xe_gpu_int_arg(kernel, 8, XE_SWA_KV_HEADS);
    xe_gpu_int_arg(kernel, 9, batch_start);
    xe_gpu_int_arg(kernel, 10, stage_base);
    xe_gpu_int_arg(kernel, 11, stage_count);
    xe_gpu_int_arg(kernel, 12, XE_SWA_WINDOW);
    size_t elements = (size_t)XE_SWA_KV_HEADS * stage_count
                      * XE_SWA_HEAD_DIM;
    ze_group_count_t groups = { (uint32_t)((elements + 255) / 256), 1, 1 };
    xe_ze_check("zeCommandListAppendLaunchKernel prefill swa stage",
                zeCommandListAppendLaunchKernel(e->gpu.commands, kernel,
                                                &groups, NULL, 0, NULL));
}

static void __attribute__((unused)) xe_prefill_swa_commit_append(
        xe_engine *e, _Float16 *ring_k, _Float16 *ring_v,
        const _Float16 *batch_k, const _Float16 *batch_v,
        int rows, int batch_start, int dimension, int kv_heads,
        int capacity) {
    ze_kernel_handle_t kernel = e->gpu.prefill_swa_commit;
    xe_gpu_pointer_arg(kernel, 0, ring_k);
    xe_gpu_pointer_arg(kernel, 1, ring_v);
    xe_gpu_pointer_arg(kernel, 2, batch_k);
    xe_gpu_pointer_arg(kernel, 3, batch_v);
    xe_gpu_int_arg(kernel, 4, rows);
    xe_gpu_int_arg(kernel, 5, dimension);
    xe_gpu_int_arg(kernel, 6, kv_heads);
    xe_gpu_int_arg(kernel, 7, batch_start);
    xe_gpu_int_arg(kernel, 8, capacity);
    size_t elements = (size_t)kv_heads * rows * dimension;
    ze_group_count_t groups = { (uint32_t)((elements + 255) / 256), 1, 1 };
    xe_ze_check("zeCommandListAppendLaunchKernel prefill swa commit",
                zeCommandListAppendLaunchKernel(e->gpu.commands, kernel,
                                                &groups, NULL, 0, NULL));
}

static void xe_prefill_linear_commit_append(
        xe_engine *e, _Float16 *cache_k, _Float16 *cache_v,
        const _Float16 *batch_k, const _Float16 *batch_v,
        int rows, int batch_start, int dimension, int kv_heads,
        int capacity) {
    if (rows < 1 || batch_start < 0 || batch_start > capacity ||
        rows > capacity - batch_start)
        xe_fatal("prefill linear KV commit: batch exceeds capacity %d", capacity);
    size_t bytes = (size_t)rows * dimension * sizeof(*cache_k);
    for (int head = 0; head < kv_heads; head++) {
        _Float16 *target_k = cache_k +
            ((size_t)head * capacity + batch_start) * dimension;
        _Float16 *target_v = cache_v +
            ((size_t)head * capacity + batch_start) * dimension;
        const _Float16 *source_k = batch_k +
            (size_t)head * rows * dimension;
        const _Float16 *source_v = batch_v +
            (size_t)head * rows * dimension;
        xe_ze_check("zeCommandListAppendMemoryCopy prefill linear K",
                    zeCommandListAppendMemoryCopy(e->gpu.commands, target_k,
                                                  source_k, bytes, NULL,
                                                  0, NULL));
        xe_ze_check("zeCommandListAppendMemoryCopy prefill linear V",
                    zeCommandListAppendMemoryCopy(e->gpu.commands, target_v,
                                                  source_v, bytes, NULL,
                                                  0, NULL));
    }
}

static void __attribute__((unused)) xe_prefill_ffn_input_append(
        xe_engine *e, const xe_layer *layer, const float *input,
        const float *row_scale, xe_q8 *dense, xe_q8 *moe,
        float *router_input, int rows) {
    ze_kernel_handle_t kernel = e->gpu.prefill_ffn_input_q8;
    xe_gpu_pointer_arg(kernel, 0, input);
    xe_gpu_pointer_arg(kernel, 1, layer->ffn_norm);
    xe_gpu_pointer_arg(kernel, 2, layer->pre_ffw_norm2);
    xe_gpu_pointer_arg(kernel, 3, layer->router_scale);
    xe_gpu_pointer_arg(kernel, 4, row_scale);
    xe_gpu_pointer_arg(kernel, 5, dense->qs);
    xe_gpu_pointer_arg(kernel, 6, dense->d);
    xe_gpu_pointer_arg(kernel, 7, dense->sigma);
    xe_gpu_pointer_arg(kernel, 8, moe->qs);
    xe_gpu_pointer_arg(kernel, 9, moe->d);
    xe_gpu_pointer_arg(kernel, 10, moe->sigma);
    xe_gpu_pointer_arg(kernel, 11, router_input);
    xe_gpu_float_arg(kernel, 12, 1.0f / sqrtf((float)XE_EMBD));
    xe_gpu_int_arg(kernel, 13, rows);
    xe_gpu_int_arg(kernel, 14, XE_EMBD);
    ze_group_count_t groups = {
        (uint32_t)(((size_t)rows * (XE_EMBD / 32) + 7) / 8), 1, 1
    };
    xe_ze_check("zeCommandListAppendLaunchKernel prefill ffn input",
                zeCommandListAppendLaunchKernel(e->gpu.commands, kernel,
                                                &groups, NULL, 0, NULL));
}

static void __attribute__((unused)) xe_prefill_geglu_q8_append(
        xe_engine *e, const float *gate, const float *up, xe_q8 *output,
        int rows, int width) {
    ze_kernel_handle_t kernel = e->gpu.prefill_geglu_q8;
    xe_gpu_pointer_arg(kernel, 0, gate);
    xe_gpu_pointer_arg(kernel, 1, up);
    xe_gpu_pointer_arg(kernel, 2, output->qs);
    xe_gpu_pointer_arg(kernel, 3, output->d);
    xe_gpu_pointer_arg(kernel, 4, output->sigma);
    xe_gpu_int_arg(kernel, 5, rows);
    xe_gpu_int_arg(kernel, 6, width);
    ze_group_count_t groups = {
        (uint32_t)(((size_t)rows * (width / 32) + 7) / 8), 1, 1
    };
    xe_ze_check("zeCommandListAppendLaunchKernel prefill geglu q8",
                zeCommandListAppendLaunchKernel(e->gpu.commands, kernel,
                                                &groups, NULL, 0, NULL));
}

static void __attribute__((unused)) xe_prefill_router_append(
        xe_engine *e, const xe_layer *layer, const float *input,
        const float *row_scale,
        float *logits, int *route_expert, float *route_weight, int rows) {
    ze_kernel_handle_t gemm = e->gpu.prefill_router_gemm;
    xe_gpu_pointer_arg(gemm, 0, input);
    xe_gpu_pointer_arg(gemm, 1, layer->router_scale);
    xe_gpu_pointer_arg(gemm, 2, row_scale);
    xe_gpu_pointer_arg(gemm, 3, layer->router_w);
    xe_gpu_pointer_arg(gemm, 4, logits);
    xe_gpu_float_arg(gemm, 5, 1.0f / sqrtf((float)XE_EMBD));
    xe_gpu_int_arg(gemm, 6, rows);
    xe_gpu_int_arg(gemm, 7, XE_EMBD);
    xe_gpu_int_arg(gemm, 8, XE_EXPERTS);
    ze_group_count_t gemm_groups = {
        (uint32_t)((rows + 15) / 16), XE_EXPERTS / 16, 1
    };
    xe_ze_check("zeCommandListAppendLaunchKernel prefill router gemm",
                zeCommandListAppendLaunchKernel(e->gpu.commands, gemm,
                                                &gemm_groups, NULL, 0, NULL));
    ze_kernel_handle_t top8 = e->gpu.prefill_router_top8;
    xe_gpu_pointer_arg(top8, 0, logits);
    xe_gpu_pointer_arg(top8, 1, route_expert);
    xe_gpu_pointer_arg(top8, 2, route_weight);
    xe_gpu_int_arg(top8, 3, rows);
    xe_gpu_int_arg(top8, 4, XE_EXPERTS);
    ze_group_count_t top8_groups = { (uint32_t)((rows + 7) / 8), 1, 1 };
    xe_ze_check("zeCommandListAppendLaunchKernel prefill router top8",
                zeCommandListAppendLaunchKernel(e->gpu.commands, top8,
                                                &top8_groups, NULL, 0, NULL));
}

typedef struct {
    int *expert_count;
    int *token_offset;
    int *cursor;
    int *tile_expert;
    int *tile_m0;
    int *packed_route;
    int *route_packed;
} xe_prefill_routes;

static void __attribute__((unused)) xe_prefill_route_append(
        xe_engine *e, const xe_q8 *source, xe_q8 *packed,
        const int *route_expert, xe_prefill_routes *route, int rows) {
    int routes = rows * XE_EXPERTS_USED;
#ifdef XE_TEST_PREFILL_TILE
    int tile_rows = xe_test_prefill_tile_rows
                    ? xe_test_prefill_tile_rows
                    : rows >= 32 && rows <= 96 ? 16 : 0;
#elif defined(XE_PREFILL_FORCE_M32)
    int tile_rows = 32;
#else
    int tile_rows = rows >= 32 && rows <= 96 ? 16 : 0;
#endif
    ze_kernel_handle_t reset = e->gpu.prefill_route_reset;
    xe_gpu_pointer_arg(reset, 0, route->expert_count);
    xe_gpu_pointer_arg(reset, 1, route->cursor);
    xe_gpu_pointer_arg(reset, 2, route->tile_expert);
    xe_gpu_pointer_arg(reset, 3, route->tile_m0);
    ze_group_count_t reset_groups = {
        (uint32_t)(!tile_rows || (rows > 96 && tile_rows != 32) ? 4 : 2), 1, 1
    };
    xe_ze_check("zeCommandListAppendLaunchKernel prefill route reset",
                zeCommandListAppendLaunchKernel(e->gpu.commands, reset,
                                                &reset_groups, NULL, 0, NULL));
    ze_kernel_handle_t count = e->gpu.prefill_route_count;
    xe_gpu_pointer_arg(count, 0, route_expert);
    xe_gpu_pointer_arg(count, 1, route->expert_count);
    xe_gpu_int_arg(count, 2, routes);
    ze_group_count_t route_groups = { (uint32_t)((routes + 127) / 128), 1, 1 };
    xe_ze_check("zeCommandListAppendLaunchKernel prefill route count",
                zeCommandListAppendLaunchKernel(e->gpu.commands, count,
                                                &route_groups, NULL, 0, NULL));
    ze_kernel_handle_t prefix = e->gpu.prefill_route_prefix;
    xe_gpu_pointer_arg(prefix, 0, route->expert_count);
    xe_gpu_pointer_arg(prefix, 1, route->token_offset);
    xe_gpu_pointer_arg(prefix, 2, route->cursor);
    xe_gpu_pointer_arg(prefix, 3, route->tile_expert);
    xe_gpu_pointer_arg(prefix, 4, route->tile_m0);
    xe_gpu_int_arg(prefix, 5, tile_rows);
    ze_group_count_t one_group = { 1, 1, 1 };
    xe_ze_check("zeCommandListAppendLaunchKernel prefill route prefix",
                zeCommandListAppendLaunchKernel(e->gpu.commands, prefix,
                                                &one_group, NULL, 0, NULL));
    ze_kernel_handle_t scatter = e->gpu.prefill_route_scatter;
    xe_gpu_pointer_arg(scatter, 0, route_expert);
    xe_gpu_pointer_arg(scatter, 1, route->cursor);
    xe_gpu_pointer_arg(scatter, 2, route->packed_route);
    xe_gpu_pointer_arg(scatter, 3, route->route_packed);
    xe_gpu_int_arg(scatter, 4, routes);
    xe_ze_check("zeCommandListAppendLaunchKernel prefill route scatter",
                zeCommandListAppendLaunchKernel(e->gpu.commands, scatter,
                                                &route_groups, NULL, 0, NULL));
    ze_kernel_handle_t pack = e->gpu.prefill_route_pack;
    xe_gpu_pointer_arg(pack, 0, source->qs);
    xe_gpu_pointer_arg(pack, 1, source->d);
    xe_gpu_pointer_arg(pack, 2, source->sigma);
    xe_gpu_pointer_arg(pack, 3, packed->qs);
    xe_gpu_pointer_arg(pack, 4, packed->d);
    xe_gpu_pointer_arg(pack, 5, packed->sigma);
    xe_gpu_pointer_arg(pack, 6, route->packed_route);
    xe_gpu_int_arg(pack, 7, XE_EMBD / 32);
    xe_gpu_int_arg(pack, 8, routes);
    ze_group_count_t pack_groups = { (uint32_t)routes, 1, 1 };
    xe_ze_check("zeCommandListAppendLaunchKernel prefill route pack",
                zeCommandListAppendLaunchKernel(e->gpu.commands, pack,
                                                &pack_groups, NULL, 0, NULL));
}

static void __attribute__((unused)) xe_prefill_grouped_projection_append(
        xe_engine *e, const xe_q4 *weight, const xe_q8 *input,
        float *output, const xe_prefill_routes *route, int columns,
        int rows) {
#ifdef XE_TEST_PREFILL_TILE
    int tile_rows = xe_test_prefill_tile_rows
                    ? xe_test_prefill_tile_rows
                    : rows >= 32 && rows <= 96 ? 16 : 0;
#elif defined(XE_PREFILL_FORCE_M32)
    int tile_rows = 32;
#else
    int tile_rows = rows >= 32 && rows <= 96 ? 16 : 0;
#endif
    ze_kernel_handle_t kernels[3] = {
        tile_rows == 16
        ? rows <= 96 ? e->gpu.prefill_q4q8_grouped_m16_n64
                     : e->gpu.prefill_q4q8_grouped_m16_n128
        : columns == 2 * XE_EXPERT_FFN
          ? e->gpu.prefill_q4q8_grouped_gate_n128
          : columns == XE_EMBD
            ? e->gpu.prefill_q4q8_grouped_down_n128
            : e->gpu.prefill_q4q8_grouped_n128,
        e->gpu.prefill_q4q8_grouped_m16_n128,
        e->gpu.prefill_q4q8_grouped_m8_n128
    };
    const int *tile_expert[3] = {
        route->tile_expert, route->tile_expert + 256,
        route->tile_expert + 384
    };
    const int *tile_m0[3] = {
        route->tile_m0, route->tile_m0 + 256, route->tile_m0 + 384
    };
    int group_x[3] = {
        tile_rows == 16 && rows > 96 ? 384 : 256, 256, 128
    };
    int passes = tile_rows ? 1 : 3;
    if (!tile_rows) {
        group_x[0] = 256;
        group_x[1] = 128;
    }
    int tile_columns = tile_rows == 16 && rows <= 96 ? 32 : 128;
    for (int pass = 0; pass < passes; pass++) {
        ze_kernel_handle_t kernel = kernels[pass];
        xe_gpu_pointer_arg(kernel, 0, weight->qs);
        xe_gpu_pointer_arg(kernel, 1, weight->d);
        xe_gpu_pointer_arg(kernel, 2, input->qs);
        xe_gpu_pointer_arg(kernel, 3, input->d);
        xe_gpu_pointer_arg(kernel, 4, input->sigma);
        xe_gpu_pointer_arg(kernel, 5, output);
        xe_gpu_pointer_arg(kernel, 6, route->expert_count);
        xe_gpu_pointer_arg(kernel, 7, route->token_offset);
        xe_gpu_pointer_arg(kernel, 8, tile_expert[pass]);
        xe_gpu_pointer_arg(kernel, 9, tile_m0[pass]);
        xe_gpu_int_arg(kernel, 10, columns);
        xe_gpu_int_arg(kernel, 11, weight->blocks);
        ze_group_count_t groups = {
            (uint32_t)group_x[pass],
            (uint32_t)(columns / tile_columns), 1
        };
        xe_ze_check("zeCommandListAppendLaunchKernel prefill grouped projection",
                    zeCommandListAppendLaunchKernel(e->gpu.commands, kernel,
                                                    &groups, NULL, 0, NULL));
    }
}

static void __attribute__((unused)) xe_prefill_expert_geglu_append(
        xe_engine *e, const float *gate_up, xe_q8 *output, int rows) {
    ze_kernel_handle_t kernel = e->gpu.prefill_expert_geglu_q8;
    xe_gpu_pointer_arg(kernel, 0, gate_up);
    xe_gpu_pointer_arg(kernel, 1, output->qs);
    xe_gpu_pointer_arg(kernel, 2, output->d);
    xe_gpu_pointer_arg(kernel, 3, output->sigma);
    xe_gpu_int_arg(kernel, 4, rows);
    xe_gpu_int_arg(kernel, 5, XE_EXPERT_FFN);
    ze_group_count_t groups = {
        (uint32_t)(((size_t)rows * (XE_EXPERT_FFN / 32) + 7) / 8), 1, 1
    };
    xe_ze_check("zeCommandListAppendLaunchKernel prefill expert geglu",
                zeCommandListAppendLaunchKernel(e->gpu.commands, kernel,
                                                &groups, NULL, 0, NULL));
}

static void __attribute__((unused)) xe_prefill_route_reduce_append(
        xe_engine *e, const xe_layer *layer, const float *routed,
        const float *route_weight, const int *route_expert,
        const xe_prefill_routes *route, float *output, int rows) {
    ze_kernel_handle_t kernel = e->gpu.prefill_route_reduce;
    xe_gpu_pointer_arg(kernel, 0, routed);
    xe_gpu_pointer_arg(kernel, 1, route_weight);
    xe_gpu_pointer_arg(kernel, 2, route_expert);
    xe_gpu_pointer_arg(kernel, 3, route->route_packed);
    xe_gpu_pointer_arg(kernel, 4, layer->down_exps_scale);
    xe_gpu_pointer_arg(kernel, 5, output);
    xe_gpu_int_arg(kernel, 6, XE_EMBD);
    ze_group_count_t groups = { (uint32_t)rows, 1, 1 };
    xe_ze_check("zeCommandListAppendLaunchKernel prefill route reduce",
                zeCommandListAppendLaunchKernel(e->gpu.commands, kernel,
                                                &groups, NULL, 0, NULL));
}

static void __attribute__((unused)) xe_prefill_ffn_finish_append(
        xe_engine *e, const xe_layer *layer, const float *dense,
        const float *moe, const float *residual, float *output, int rows) {
    ze_kernel_handle_t kernel = e->gpu.prefill_ffn_finish;
    xe_gpu_pointer_arg(kernel, 0, dense);
    xe_gpu_pointer_arg(kernel, 1, moe);
    xe_gpu_pointer_arg(kernel, 2, layer->post_ffw_norm1);
    xe_gpu_pointer_arg(kernel, 3, layer->post_ffw_norm2);
    xe_gpu_pointer_arg(kernel, 4, layer->post_ffw_norm);
    xe_gpu_pointer_arg(kernel, 5, residual);
    xe_gpu_pointer_arg(kernel, 6, layer->layer_out_scale);
    xe_gpu_pointer_arg(kernel, 7, output);
    xe_gpu_int_arg(kernel, 8, rows);
    xe_gpu_int_arg(kernel, 9, XE_EMBD);
    xe_gpu_float_arg(kernel, 10, XE_RMS_EPS);
    ze_group_count_t groups = { (uint32_t)rows, 1, 1 };
    xe_ze_check("zeCommandListAppendLaunchKernel prefill ffn finish",
                zeCommandListAppendLaunchKernel(e->gpu.commands, kernel,
                                                &groups, NULL, 0, NULL));
}

#ifdef XE_BENCH_TG_PROFILE
typedef struct {
    double embed_rope;
    double attention;
    double dense;
    double moe;
    double glue;
    double output;
    double worker_end;
} xe_tg_profile;

static xe_tg_profile xe_tg_profile_data;

static double xe_tg_profile_now(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec + (double)ts.tv_nsec * 1e-9;
}
#endif

typedef struct {
    uint8_t *base;
    size_t pos;
} xe_workspace_arena;

static float xe_gelu_fp16_value(float x) {
    if (x <= -10.0f) return 0.0f;
    if (x >= 10.0f) return x;
    float inner = 0.79788456080286535588f * (x + 0.044715f * x * x * x);
    return (float)(_Float16)(0.5f * x * (1.0f + tanhf(inner)));
}

static void xe_constants_init(xe_engine *e) {
    if (e->gelu_lut) return;
    e->gelu_lut = xe_alloc(e, 65536 * sizeof(*e->gelu_lut), XE_MEM_HOST);
    for (uint32_t bits = 0; bits < 65536; bits++) {
        uint16_t hbits = (uint16_t)bits;
        _Float16 h;
        memcpy(&h, &hbits, sizeof h);
        e->gelu_lut[bits] = (_Float16)xe_gelu_fp16_value((float)h);
    }
    for (int i = 0; i < XE_SWA_HEAD_DIM / 2; i++)
        e->rope_swa_inv[i] = powf(XE_SWA_ROPE_BASE, -2.0f * (float)i / (float)XE_SWA_HEAD_DIM);
    for (int i = 0; i < XE_GLOBAL_HEAD_DIM / 2; i++) {
        float inv = powf(XE_GLOBAL_ROPE_BASE, -2.0f * (float)i / (float)XE_GLOBAL_HEAD_DIM);
        e->rope_global_inv[i] = e->rope_freqs ? inv / e->rope_freqs[i] : inv;
    }
}

static void xe_pin_thread(int cpu) {
    cpu_set_t set;
    CPU_ZERO(&set);
    CPU_SET(cpu, &set);
    int rc = pthread_setaffinity_np(pthread_self(), sizeof set, &set);
    if (rc != 0) xe_fatal("pthread_setaffinity_np CPU %d: %s", cpu, strerror(rc));
}

static int xe_worker_cpu(int lane) {
#ifdef XE_WORKER_ECORES
    return 12 + lane;
#else
    return 2 * lane;
#endif
}

static int xe_worker_pin_available(const cpu_set_t *allowed) {
    int packages[XE_WORKERS];
    int cores[XE_WORKERS];
    for (int lane = 0; lane < XE_WORKERS; lane++) {
        int cpu = xe_worker_cpu(lane);
        if (!CPU_ISSET(cpu, allowed)) return 0;
        char path[128];
        snprintf(path, sizeof path, "/sys/devices/system/cpu/cpu%d/topology/physical_package_id", cpu);
        FILE *f = fopen(path, "r");
        if (!f) return 0;
        int package;
        int read = fscanf(f, "%d", &package);
        fclose(f);
        if (read != 1) return 0;
        snprintf(path, sizeof path, "/sys/devices/system/cpu/cpu%d/topology/core_id", cpu);
        f = fopen(path, "r");
        if (!f) return 0;
        int core;
        read = fscanf(f, "%d", &core);
        fclose(f);
        if (read != 1) return 0;
        for (int other = 0; other < lane; other++)
            if (packages[other] == package && cores[other] == core) return 0;
        packages[lane] = package;
        cores[lane] = core;
    }
    return 1;
}

static void xe_require_owner(const xe_engine *e) {
    if (!e->pool_initialized || !pthread_equal(e->owner, pthread_self()))
        xe_fatal("decode called from a thread other than the engine owner");
}

static uint64_t xe_wait_changed(atomic_uint_fast64_t *value, uint64_t seen) {
    uint64_t current;
    do {
        current = atomic_load_explicit(value, memory_order_acquire);
        if (current == seen) {
#ifdef XE_LOW_POWER_WAIT
            _umonitor((void *)value);
            if (atomic_load_explicit(value, memory_order_acquire) == seen)
                _umwait(0, __rdtsc() + UINT64_C(100000));
#else
            _mm_pause();
#endif
        }
    } while (current == seen);
    return current;
}

static void xe_wait_equal(atomic_uint_fast64_t *value, uint64_t expected) {
    for (;;) {
        if (atomic_load_explicit(value, memory_order_acquire) == expected) return;
#ifdef XE_LOW_POWER_WAIT
        _umonitor((void *)value);
        if (atomic_load_explicit(value, memory_order_acquire) != expected)
            _umwait(0, __rdtsc() + UINT64_C(100000));
#else
        _mm_pause();
#endif
    }
}

static void *xe_worker_main(void *opaque) {
    xe_worker_arg *wa = opaque;
    xe_engine *e = wa->engine;
    int lane = wa->lane;
    uint64_t seen = 0;

    if (e->worker_pinned) xe_pin_thread(xe_worker_cpu(lane));

    pthread_mutex_lock(&e->worker_mutex);
    e->worker_ready++;
    pthread_cond_signal(&e->ready_cv);
    uint64_t ticket = e->worker_ticket;

    for (;;) {
        while (!e->worker_stop && ticket == e->worker_ticket)
            pthread_cond_wait(&e->worker_cv, &e->worker_mutex);
        if (e->worker_stop) break;
        ticket = e->worker_ticket;
        pthread_mutex_unlock(&e->worker_mutex);

        for (;;) {
            uint64_t epoch = xe_wait_changed(&e->phase_epoch, seen);

            xe_phase phase = e->phase;
            if (phase.fn) {
                int worker = phase.caller_participates ? lane : lane - 1;
                int workers = phase.caller_participates ? XE_WORKERS : XE_WORKERS - 1;
                phase.fn(phase.session, phase.arg, worker, workers);
            }
            atomic_store_explicit(&e->phase_done[lane].value, epoch, memory_order_release);
            seen = epoch;
            if (!phase.fn) break;
        }

        pthread_mutex_lock(&e->worker_mutex);
    }

    pthread_mutex_unlock(&e->worker_mutex);
    return NULL;
}

static void xe_worker_pool_init(xe_engine *e) {
    if (e->pool_initialized) {
        xe_require_owner(e);
        return;
    }

    e->owner = pthread_self();
    e->worker_pinned = pthread_getaffinity_np(e->owner, sizeof e->owner_affinity,
                                                &e->owner_affinity) == 0 &&
                       xe_worker_pin_available(&e->owner_affinity);
    if (pthread_mutex_init(&e->worker_mutex, NULL) != 0 ||
        pthread_cond_init(&e->worker_cv, NULL) != 0 ||
        pthread_cond_init(&e->ready_cv, NULL) != 0)
        xe_fatal("worker synchronization initialization failed");

    e->worker_ticket = 0;
    e->worker_ready = 0;
    e->worker_stop = 0;
    e->token_active = 0;
    e->phase.fn = NULL;
    e->phase.session = NULL;
    e->phase.arg = NULL;
    e->phase.caller_participates = 0;
    atomic_init(&e->phase_epoch, 0);
    for (int lane = 0; lane < XE_WORKERS; lane++)
        atomic_init(&e->phase_done[lane].value, 0);
    e->pool_initialized = 1;

    xe_constants_init(e);

    if (e->worker_pinned) xe_pin_thread(xe_worker_cpu(0));

    pthread_mutex_lock(&e->worker_mutex);
    for (int lane = 1; lane < XE_WORKERS; lane++) {
        xe_worker_arg *wa = &e->worker_args[lane - 1];
        wa->engine = e;
        wa->lane = lane;
        int rc = pthread_create(&e->followers[lane - 1], NULL, xe_worker_main, wa);
        if (rc != 0) xe_fatal("pthread_create worker %d: %s", lane, strerror(rc));
    }
    while (e->worker_ready != XE_WORKERS - 1)
        pthread_cond_wait(&e->ready_cv, &e->worker_mutex);
    pthread_mutex_unlock(&e->worker_mutex);
}

static void xe_worker_pool_destroy(xe_engine *e) {
    if (!e->pool_initialized) return;
    xe_require_owner(e);
    if (e->token_active) xe_fatal("engine closed during an active decode token");

    pthread_mutex_lock(&e->worker_mutex);
    e->worker_stop = 1;
    pthread_cond_broadcast(&e->worker_cv);
    pthread_mutex_unlock(&e->worker_mutex);

    for (int i = 0; i < XE_WORKERS - 1; i++)
        pthread_join(e->followers[i], NULL);

    pthread_cond_destroy(&e->ready_cv);
    pthread_cond_destroy(&e->worker_cv);
    pthread_mutex_destroy(&e->worker_mutex);
    e->pool_initialized = 0;
    if (e->worker_pinned) {
        int rc = pthread_setaffinity_np(e->owner, sizeof e->owner_affinity, &e->owner_affinity);
        if (rc != 0) xe_fatal("pthread_setaffinity_np restore: %s", strerror(rc));
    }
}

static void xe_workers_begin(xe_engine *e) {
    xe_require_owner(e);
    if (e->token_active) xe_fatal("nested decode token");
    e->token_active = 1;
    pthread_mutex_lock(&e->worker_mutex);
    e->worker_ticket++;
    pthread_cond_broadcast(&e->worker_cv);
    pthread_mutex_unlock(&e->worker_mutex);
}

static uint64_t xe_publish_phase(xe_engine *e, xe_session *s, xe_phase_fn fn,
                                 const void *arg, int caller_participates) {
    e->phase.fn = fn;
    e->phase.session = s;
    e->phase.arg = arg;
    e->phase.caller_participates = caller_participates;
    uint64_t epoch = atomic_load_explicit(&e->phase_epoch, memory_order_relaxed) + 1;
    atomic_store_explicit(&e->phase_epoch, epoch, memory_order_release);
    return epoch;
}

static void xe_wait_followers(xe_engine *e, uint64_t epoch) {
    for (int lane = 1; lane < XE_WORKERS; lane++)
        xe_wait_equal(&e->phase_done[lane].value, epoch);
}

static void xe_dispatch(xe_engine *e, xe_session *s, xe_phase_fn fn,
                        const void *arg, int caller_participates) {
    xe_require_owner(e);
    if (!e->token_active || !fn) xe_fatal("invalid worker phase");
    uint64_t epoch = xe_publish_phase(e, s, fn, arg, caller_participates);
    if (caller_participates) fn(s, arg, 0, XE_WORKERS);
    xe_wait_followers(e, epoch);
}

static void xe_workers_end(xe_engine *e) {
    xe_require_owner(e);
    if (!e->token_active) xe_fatal("decode token is not active");
    uint64_t epoch = xe_publish_phase(e, NULL, NULL, NULL, 0);
    xe_wait_followers(e, epoch);
    e->token_active = 0;
}

static void *xe_workspace_take(xe_workspace_arena *a, size_t bytes) {
    a->pos = (a->pos + 63) & ~(size_t)63;
    void *p = a->base ? a->base + a->pos : NULL;
    a->pos += bytes;
    return p;
}

static void xe_workspace_q8(xe_workspace_arena *a, xe_q8 *q, int n) {
    q->qs = xe_workspace_take(a, (size_t)n * sizeof(*q->qs));
    q->d = xe_workspace_take(a, (size_t)(n / 32) * sizeof(*q->d));
    q->sigma = xe_workspace_take(a, (size_t)(n / 32) * sizeof(*q->sigma));
    q->n = n;
}

typedef struct {
    float *hidden[2];
    float *row_scale;
    xe_q8 attention_input;
    float *q_projection;
    float *k_projection;
    float *v_projection;
    float *rope_swa_cos;
    float *rope_swa_sin;
    float *rope_global_cos;
    float *rope_global_sin;
    float *q_heads;
    _Float16 *k_batch;
    _Float16 *v_batch;
    _Float16 *k_stage;
    _Float16 *v_stage;
    float *attention_heads;
    xe_q8 heads_q8;
    float *attention_projection;
    float *attention_output;
    xe_q8 dense_input;
    xe_q8 moe_input;
    float *router_input;
    float *router_logits;
    int *route_expert;
    float *route_weight;
    xe_prefill_routes routes;
    xe_q8 packed_moe;
    float *dense_gate;
    float *dense_up;
    xe_q8 dense_activation;
    float *dense_down;
    float *expert_gate_up;
    xe_q8 expert_activation;
    float *expert_down;
    float *moe_output;
} xe_prefill_workspace;

static _Float16 *xe_kv_layer_ptr(xe_session *s, int layer, int value);

static size_t __attribute__((unused)) xe_prefill_workspace_layout(
        xe_prefill_workspace *w, void *base) {
    const int rows = 512;
    const int routes = rows * XE_EXPERTS_USED;
    const int q_width = XE_Q_HEADS * XE_GLOBAL_HEAD_DIM;
    const int kv_width = XE_SWA_KV_HEADS * XE_SWA_HEAD_DIM;
    const int stage_keys = XE_SWA_WINDOW + rows - 1;
    xe_workspace_arena a = { base, 0 };
    w->hidden[0] = xe_workspace_take(
        &a, (size_t)rows * XE_EMBD * sizeof(float));
    w->hidden[1] = xe_workspace_take(
        &a, (size_t)rows * XE_EMBD * sizeof(float));
    w->row_scale = xe_workspace_take(&a, rows * sizeof(float));
    xe_workspace_q8(&a, &w->attention_input, rows * XE_EMBD);
    w->q_projection = xe_workspace_take(
        &a, (size_t)rows * q_width * sizeof(float));
    w->k_projection = xe_workspace_take(
        &a, (size_t)rows * kv_width * sizeof(float));
    w->v_projection = xe_workspace_take(
        &a, (size_t)rows * kv_width * sizeof(float));
    w->rope_swa_cos = xe_workspace_take(
        &a, (size_t)rows * (XE_SWA_HEAD_DIM / 2) * sizeof(float));
    w->rope_swa_sin = xe_workspace_take(
        &a, (size_t)rows * (XE_SWA_HEAD_DIM / 2) * sizeof(float));
    w->rope_global_cos = xe_workspace_take(
        &a, (size_t)rows * (XE_GLOBAL_HEAD_DIM / 2) * sizeof(float));
    w->rope_global_sin = xe_workspace_take(
        &a, (size_t)rows * (XE_GLOBAL_HEAD_DIM / 2) * sizeof(float));
    w->q_heads = xe_workspace_take(
        &a, (size_t)rows * q_width * sizeof(float));
    w->k_batch = xe_workspace_take(
        &a, (size_t)rows * kv_width * sizeof(_Float16));
    w->v_batch = xe_workspace_take(
        &a, (size_t)rows * kv_width * sizeof(_Float16));
    w->k_stage = xe_workspace_take(
        &a, (size_t)stage_keys * kv_width * sizeof(_Float16));
    w->v_stage = xe_workspace_take(
        &a, (size_t)stage_keys * kv_width * sizeof(_Float16));
    w->attention_heads = xe_workspace_take(
        &a, (size_t)rows * q_width * sizeof(float));
    xe_workspace_q8(&a, &w->heads_q8, rows * q_width);
    w->attention_projection = xe_workspace_take(
        &a, (size_t)rows * XE_EMBD * sizeof(float));
    w->attention_output = xe_workspace_take(
        &a, (size_t)rows * XE_EMBD * sizeof(float));
    xe_workspace_q8(&a, &w->dense_input, rows * XE_EMBD);
    xe_workspace_q8(&a, &w->moe_input, rows * XE_EMBD);
    w->router_input = xe_workspace_take(
        &a, (size_t)rows * XE_EMBD * sizeof(float));
    w->router_logits = xe_workspace_take(
        &a, (size_t)rows * XE_EXPERTS * sizeof(float));
    w->route_expert = xe_workspace_take(
        &a, (size_t)routes * sizeof(int));
    w->route_weight = xe_workspace_take(
        &a, (size_t)routes * sizeof(float));
    w->routes.expert_count = xe_workspace_take(
        &a, XE_EXPERTS * sizeof(int));
    w->routes.token_offset = xe_workspace_take(
        &a, (XE_EXPERTS + 1) * sizeof(int));
    w->routes.cursor = xe_workspace_take(&a, XE_EXPERTS * sizeof(int));
    w->routes.tile_expert = xe_workspace_take(&a, 512 * sizeof(int));
    w->routes.tile_m0 = xe_workspace_take(&a, 512 * sizeof(int));
    w->routes.packed_route = xe_workspace_take(
        &a, (size_t)routes * sizeof(int));
    w->routes.route_packed = xe_workspace_take(
        &a, (size_t)routes * sizeof(int));
    xe_workspace_q8(&a, &w->packed_moe, routes * XE_EMBD);
    w->dense_gate = xe_workspace_take(
        &a, (size_t)rows * XE_DENSE_FFN * sizeof(float));
    w->dense_up = xe_workspace_take(
        &a, (size_t)rows * XE_DENSE_FFN * sizeof(float));
    xe_workspace_q8(&a, &w->dense_activation, rows * XE_DENSE_FFN);
    w->dense_down = xe_workspace_take(
        &a, (size_t)rows * XE_EMBD * sizeof(float));
    w->expert_gate_up = xe_workspace_take(
        &a, (size_t)routes * 2 * XE_EXPERT_FFN * sizeof(float));
    xe_workspace_q8(&a, &w->expert_activation, routes * XE_EXPERT_FFN);
    w->expert_down = xe_workspace_take(
        &a, (size_t)routes * XE_EMBD * sizeof(float));
    w->moe_output = xe_workspace_take(
        &a, (size_t)rows * XE_EMBD * sizeof(float));
    return (a.pos + 63) & ~(size_t)63;
}

static void __attribute__((unused)) xe_prefill_rope_prepare_batch(
        const xe_engine *e, xe_prefill_workspace *w, int rows,
        int batch_start, int global) {
    int dimension = global ? XE_GLOBAL_HEAD_DIM : XE_SWA_HEAD_DIM;
    const float *inverse = global ? e->rope_global_inv : e->rope_swa_inv;
    float *cosine = global ? w->rope_global_cos : w->rope_swa_cos;
    float *sine = global ? w->rope_global_sin : w->rope_swa_sin;
    for (int row = 0; row < rows; row++)
        for (int d = 0; d < dimension / 2; d++) {
            float theta = (float)(batch_start + row) * inverse[d];
            size_t index = (size_t)row * (dimension / 2) + d;
            cosine[index] = cosf(theta);
            sine[index] = sinf(theta);
        }
}

static void xe_prefill_attention_qkv_append(
        xe_engine *e, int layer_index, xe_prefill_workspace *w, int rows) {
    if (rows < 1 || rows > 512)
        xe_fatal("prefill layer initial supports M1 through M512");
    const xe_layer *layer = &e->layers[layer_index];
    int global = XE_IS_GLOBAL(layer_index);
    int dimension = global ? XE_GLOBAL_HEAD_DIM : XE_SWA_HEAD_DIM;
    int kv_heads = global ? XE_GLOBAL_KV_HEADS : XE_SWA_KV_HEADS;
    int q_width = XE_Q_HEADS * dimension;
    int kv_width = kv_heads * dimension;
    int wide = rows == 512;
    xe_prefill_input_append(e, w->hidden[0], layer->attn_norm, w->row_scale,
                            &w->attention_input, rows, XE_EMBD);
    xe_prefill_projection_append(e, &layer->attn_q, &w->attention_input,
                                 w->q_projection, rows, q_width, wide);
    xe_prefill_projection_append(e, &layer->attn_k, &w->attention_input,
                                 w->k_projection, rows, kv_width,
                                 wide);
    if (!global)
        xe_prefill_projection_append(e, &layer->attn_v, &w->attention_input,
                                     w->v_projection, rows, kv_width, wide);
    xe_prefill_qkv_append(e, w->q_projection, w->k_projection,
                          global ? w->k_projection : w->v_projection,
                          layer->q_norm, layer->k_norm,
                          global ? w->rope_global_cos : w->rope_swa_cos,
                          global ? w->rope_global_sin : w->rope_swa_sin,
                          w->q_heads,
                          w->k_batch, w->v_batch, rows, dimension, kv_heads,
                          !global);
}

static void xe_prefill_attention_project_append(
        xe_engine *e, int layer_index, xe_prefill_workspace *w, int rows) {
    int dimension = XE_IS_GLOBAL(layer_index) ? XE_GLOBAL_HEAD_DIM
                                               : XE_SWA_HEAD_DIM;
    int wide = rows == 512;
    const xe_layer *layer = &e->layers[layer_index];
    xe_prefill_heads_q8_append(e, w->attention_heads, &w->heads_q8,
                               rows, dimension);
    xe_prefill_projection_append(e, &layer->attn_o, &w->heads_q8,
                                 w->attention_projection, rows, XE_EMBD,
                                 wide);
    xe_prefill_rms_residual_append(e, w->attention_projection,
                                    layer->post_attn_norm, w->hidden[0],
                                    w->attention_output, rows, XE_EMBD);
}

static void xe_prefill_attention_output_append(
        xe_engine *e, int layer_index, xe_prefill_workspace *w, int rows,
        const _Float16 *k, const _Float16 *v, int keys, int query_offset,
        int window) {
    int global = XE_IS_GLOBAL(layer_index);
    int dimension = global ? XE_GLOBAL_HEAD_DIM : XE_SWA_HEAD_DIM;
    int kv_heads = global ? XE_GLOBAL_KV_HEADS : XE_SWA_KV_HEADS;
    xe_prefill_attention_online_append(e, w->q_heads, k, v,
                                        w->attention_heads, rows, keys,
                                        dimension, kv_heads, query_offset,
                                        window);
    xe_prefill_attention_project_append(e, layer_index, w, rows);
}

static void xe_prefill_attention_prepared_append(
        xe_engine *e, int layer_index, xe_prefill_workspace *w, int rows) {
    xe_prefill_attention_qkv_append(e, layer_index, w, rows);
    xe_prefill_attention_output_append(
        e, layer_index, w, rows, w->k_batch, w->v_batch, rows, 0,
        XE_IS_GLOBAL(layer_index) ? 0 : XE_SWA_WINDOW);
}

static void xe_prefill_attention_batch_append(
        xe_session *s, int layer_index, xe_prefill_workspace *w, int rows,
        int batch_start) {
    xe_engine *e = s->engine;
    int global = XE_IS_GLOBAL(layer_index);
    int dimension = global ? XE_GLOBAL_HEAD_DIM : XE_SWA_HEAD_DIM;
    int kv_heads = global ? XE_GLOBAL_KV_HEADS : XE_SWA_KV_HEADS;
    int capacity = global ? s->engine->context : XE_SWA_WINDOW;
    _Float16 *cache_k = xe_kv_layer_ptr(s, layer_index, 0);
    _Float16 *cache_v = xe_kv_layer_ptr(s, layer_index, 1);
    xe_prefill_attention_qkv_append(e, layer_index, w, rows);
    if (global && s->cow) {
        int slot = layer_index / 6;
        int local_start = batch_start - s->cow_split;
        const _Float16 *prefix_k = s->global_k +
            (size_t)slot * xe_global_layer_elems(s->engine);
        const _Float16 *prefix_v = s->global_v +
            (size_t)slot * xe_global_layer_elems(s->engine);
        xe_prefill_linear_commit_append(
            e, cache_k, cache_v, w->k_batch, w->v_batch, rows,
            local_start, dimension, kv_heads, s->cow_capacity);
        xe_prefill_attention_cow_append(
            e, w->q_heads, prefix_k, prefix_v, cache_k, cache_v,
            w->attention_heads, rows, capacity, s->cow_capacity, batch_start,
            s->cow_split);
        xe_prefill_attention_project_append(e, layer_index, w, rows);
    } else if (batch_start == 0) {
        xe_prefill_attention_output_append(
            e, layer_index, w, rows, w->k_batch, w->v_batch, rows, 0,
            global ? 0 : XE_SWA_WINDOW);
        if (global)
            xe_prefill_linear_commit_append(e, cache_k, cache_v, w->k_batch,
                                            w->v_batch, rows, batch_start,
                                            dimension, kv_heads, capacity);
        else
            xe_prefill_swa_commit_append(e, cache_k, cache_v, w->k_batch,
                                         w->v_batch, rows, batch_start,
                                         dimension, kv_heads, capacity);
    } else if (global) {
        xe_prefill_linear_commit_append(e, cache_k, cache_v, w->k_batch,
                                        w->v_batch, rows, batch_start,
                                        dimension, kv_heads, capacity);
        xe_prefill_attention_output_append(
            e, layer_index, w, rows, cache_k, cache_v, capacity,
            batch_start, 0);
    } else {
        int stage_base = batch_start >= XE_SWA_WINDOW - 1
                         ? batch_start - (XE_SWA_WINDOW - 1) : 0;
        int stage_count = batch_start + rows - stage_base;
        xe_prefill_swa_stage_append(e, cache_k, cache_v, w->k_batch,
                                    w->v_batch, w->k_stage, w->v_stage, rows,
                                    batch_start, stage_base, stage_count);
        xe_prefill_attention_output_append(
            e, layer_index, w, rows, w->k_stage, w->v_stage, stage_count,
            batch_start - stage_base, XE_SWA_WINDOW);
        xe_prefill_swa_commit_append(e, cache_k, cache_v, w->k_batch,
                                     w->v_batch, rows, batch_start,
                                     dimension, kv_heads, capacity);
    }
}

static void __attribute__((unused)) xe_prefill_attention_initial_append(
        xe_engine *e, int layer_index, xe_prefill_workspace *w, int rows) {
    xe_prefill_rope_prepare_batch(e, w, rows, 0,
                                  XE_IS_GLOBAL(layer_index));
    xe_prefill_attention_prepared_append(e, layer_index, w, rows);
}

static void __attribute__((unused)) xe_prefill_ffn_append(
        xe_engine *e, int layer_index, xe_prefill_workspace *w, int rows) {
    if (rows < 1 || rows > 512)
        xe_fatal("prefill FFN supports M1 through M512");
    const xe_layer *layer = &e->layers[layer_index];
    int wide = rows == 512;
    xe_prefill_rms_append(e, w->attention_output, w->row_scale,
                          rows, XE_EMBD);
    xe_prefill_ffn_input_append(e, layer, w->attention_output, w->row_scale,
                                &w->dense_input, &w->moe_input,
                                w->router_input, rows);
    xe_prefill_projection_append(e, &layer->ffn_gate, &w->dense_input,
                                 w->dense_gate, rows, XE_DENSE_FFN, wide);
    xe_prefill_projection_append(e, &layer->ffn_up, &w->dense_input,
                                 w->dense_up, rows, XE_DENSE_FFN, wide);
    xe_prefill_geglu_q8_append(e, w->dense_gate, w->dense_up,
                               &w->dense_activation, rows, XE_DENSE_FFN);
    xe_prefill_projection_append(e, &layer->ffn_down, &w->dense_activation,
                                 w->dense_down, rows, XE_EMBD, wide);
    xe_prefill_router_append(e, layer, w->attention_output, w->row_scale,
                             w->router_logits,
                             w->route_expert, w->route_weight, rows);
    xe_prefill_route_append(e, &w->moe_input, &w->packed_moe,
                            w->route_expert, &w->routes, rows);
    xe_prefill_grouped_projection_append(e, &layer->gate_up_exps,
                                          &w->packed_moe, w->expert_gate_up,
                                          &w->routes, 2 * XE_EXPERT_FFN,
                                          rows);
    xe_prefill_expert_geglu_append(e, w->expert_gate_up,
                                    &w->expert_activation,
                                    rows * XE_EXPERTS_USED);
    xe_prefill_grouped_projection_append(e, &layer->down_exps,
                                          &w->expert_activation,
                                          w->expert_down, &w->routes,
                                          XE_EMBD, rows);
    xe_prefill_route_reduce_append(e, layer, w->expert_down, w->route_weight,
                                    w->route_expert, &w->routes,
                                    w->moe_output, rows);
    xe_prefill_ffn_finish_append(e, layer, w->dense_down, w->moe_output,
                                 w->attention_output, w->hidden[1], rows);
}

static void __attribute__((unused)) xe_prefill_layer_initial_append(
        xe_engine *e, int layer_index, xe_prefill_workspace *w, int rows) {
    xe_prefill_attention_initial_append(e, layer_index, w, rows);
    xe_prefill_ffn_append(e, layer_index, w, rows);
}

static size_t xe_workspace_layout(xe_session *s, void *base) {
    xe_workspace_arena a = { base, 0 };
    s->hidden = xe_workspace_take(&a, XE_EMBD * sizeof(*s->hidden));
    s->q = xe_workspace_take(&a, (size_t)XE_Q_HEADS * XE_GLOBAL_HEAD_DIM * sizeof(*s->q));
    s->k = xe_workspace_take(&a, (size_t)XE_SWA_KV_HEADS * XE_SWA_HEAD_DIM * sizeof(*s->k));
    s->v = xe_workspace_take(&a, (size_t)XE_SWA_KV_HEADS * XE_SWA_HEAD_DIM * sizeof(*s->v));
    s->attn_heads = xe_workspace_take(&a, (size_t)XE_Q_HEADS * XE_GLOBAL_HEAD_DIM * sizeof(*s->attn_heads));
    s->attn_proj = xe_workspace_take(&a, XE_EMBD * sizeof(*s->attn_proj));
    s->attn_out = xe_workspace_take(&a, XE_EMBD * sizeof(*s->attn_out));
    s->dense_out = xe_workspace_take(&a, XE_EMBD * sizeof(*s->dense_out));
    s->moe_out = xe_workspace_take(&a, XE_EMBD * sizeof(*s->moe_out));
    s->combined = xe_workspace_take(&a, XE_EMBD * sizeof(*s->combined));
    s->router_in = xe_workspace_take(&a, XE_EMBD * sizeof(*s->router_in));
    s->router_logits = xe_workspace_take(&a, XE_EXPERTS * sizeof(*s->router_logits));
    s->scores = xe_workspace_take(&a, (size_t)XE_Q_HEADS * s->engine->context * sizeof(*s->scores));
    s->attn_partial = xe_workspace_take(&a, (size_t)XE_WORKERS * 8 * XE_GLOBAL_HEAD_DIM * sizeof(*s->attn_partial));
    s->rope_swa_cos = xe_workspace_take(&a, (XE_SWA_HEAD_DIM / 2) * sizeof(*s->rope_swa_cos));
    s->rope_swa_sin = xe_workspace_take(&a, (XE_SWA_HEAD_DIM / 2) * sizeof(*s->rope_swa_sin));
    s->rope_global_cos = xe_workspace_take(&a, (XE_GLOBAL_HEAD_DIM / 2) * sizeof(*s->rope_global_cos));
    s->rope_global_sin = xe_workspace_take(&a, (XE_GLOBAL_HEAD_DIM / 2) * sizeof(*s->rope_global_sin));
    s->logits = xe_workspace_take(&a, XE_VOCAB * sizeof(*s->logits));
    s->sample_candidates = xe_workspace_take(&a, XE_VOCAB * sizeof(*s->sample_candidates));
    xe_workspace_q8(&a, &s->q8_main, XE_Q_HEADS * XE_GLOBAL_HEAD_DIM);
    xe_workspace_q8(&a, &s->q8_dense_in, XE_EMBD);
    xe_workspace_q8(&a, &s->q8_moe_in, XE_EMBD);
    xe_workspace_q8(&a, &s->q8_dense_act, XE_DENSE_FFN);
    for (int i = 0; i < XE_EXPERTS_USED; i++)
        xe_workspace_q8(&a, &s->q8_expert[i], XE_EXPERT_FFN);
    return (a.pos + 63) & ~(size_t)63;
}

static float xe_hsum8_ps(__m256 x) {
    __m128 lo = _mm256_castps256_ps128(x);
    __m128 hi = _mm256_extractf128_ps(x, 1);
    __m128 sum = _mm_add_ps(lo, hi);
    sum = _mm_hadd_ps(sum, sum);
    sum = _mm_hadd_ps(sum, sum);
    return _mm_cvtss_f32(sum);
}

static int xe_hsum8_i32(__m256i x) {
    __m128i lo = _mm256_castsi256_si128(x);
    __m128i hi = _mm256_extracti128_si256(x, 1);
    __m128i sum = _mm_add_epi32(lo, hi);
    sum = _mm_hadd_epi32(sum, sum);
    sum = _mm_hadd_epi32(sum, sum);
    return _mm_cvtsi128_si32(sum);
}

static uint16_t xe_half_bits(_Float16 x);

static float xe_absmax32(const float *x) {
    __m256 mask = _mm256_castsi256_ps(_mm256_set1_epi32(0x7fffffff));
    __m256 m0 = _mm256_and_ps(_mm256_loadu_ps(x), mask);
    __m256 m1 = _mm256_and_ps(_mm256_loadu_ps(x + 8), mask);
    __m256 m2 = _mm256_and_ps(_mm256_loadu_ps(x + 16), mask);
    __m256 m3 = _mm256_and_ps(_mm256_loadu_ps(x + 24), mask);
    __m256 m = _mm256_max_ps(_mm256_max_ps(m0, m1), _mm256_max_ps(m2, m3));
    __m128 lo = _mm256_castps256_ps128(m);
    __m128 hi = _mm256_extractf128_ps(m, 1);
    __m128 r = _mm_max_ps(lo, hi);
    r = _mm_max_ps(r, _mm_movehl_ps(r, r));
    r = _mm_max_ss(r, _mm_shuffle_ps(r, r, 1));
    return _mm_cvtss_f32(r);
}

static void xe_quantize_q8_blocks(const float *x, xe_q8 *q, int first, int end) {
    for (int b = first; b < end; b++) {
        const float *xb = x + 32 * b;
        int8_t *qb = q->qs + 32 * b;
        float amax = xe_absmax32(xb);
        float d = amax / 127.0f;
        float id = d ? 1.0f / d : 0.0f;
        int sigma = 0;
        q->d[b] = (_Float16)d;
        for (int i = 0; i < 32; i++) {
            int v = (int)roundf(xb[i] * id);
            qb[i] = (int8_t)v;
            sigma += v;
        }
        q->sigma[b] = (int16_t)sigma;
    }
}

static void xe_quantize_q8_block_at(const float *x, xe_q8 *q, int b) {
    float amax = xe_absmax32(x);
    float d = amax / 127.0f;
    float id = d ? 1.0f / d : 0.0f;
    int sigma = 0;
    q->d[b] = (_Float16)d;
    for (int i = 0; i < 32; i++) {
        int v = (int)roundf(x[i] * id);
        q->qs[32 * b + i] = (int8_t)v;
        sigma += v;
    }
    q->sigma[b] = (int16_t)sigma;
}

static void xe_quantize_q8(const float *x, int n, xe_q8 *q) {
    if ((n & 31) || n > q->n) xe_fatal("q8: invalid length %d for capacity %d", n, q->n);
    xe_quantize_q8_blocks(x, q, 0, n / 32);
}

static inline int xe_q4_q8_raw_integer_q(const uint8_t *packed_weights, __m256i q) {
    __m128i packed = _mm_loadu_si128((const __m128i *)packed_weights);
    __m128i mask = _mm_set1_epi8(0x0f);
    __m128i low = _mm_and_si128(packed, mask);
    __m128i high = _mm_and_si128(_mm_srli_epi16(packed, 4), mask);
    __m256i nibbles = _mm256_set_m128i(high, low);
    __m256i dot = _mm256_dpbusd_epi32(_mm256_setzero_si256(), nibbles, q);
    return xe_hsum8_i32(dot);
}

static inline int xe_q4_q8_raw_integer(const uint8_t *packed_weights, const int8_t *activation) {
    return xe_q4_q8_raw_integer_q(packed_weights,
                                  _mm256_load_si256((const __m256i *)activation));
}

static inline uint16_t xe_q4_scale(const xe_q4 *w, uint64_t row, int blocks, int block) {
    if (!w->blocks) return w->d[row * (uint64_t)blocks + block];
    if (w->blocks != blocks) xe_fatal("q4: layout has %d blocks, matvec expects %d",
                                      w->blocks, blocks);
    uint64_t group = row >> 3;
    int lane = (int)(row & 7);
    return w->d[(group * (uint64_t)blocks + block) * 8 + lane];
}

static inline void xe_q4_copy_block(const xe_q4 *w, uint64_t row, int blocks,
                                    int block, uint8_t packed[16]) {
    if (!w->blocks) {
        memcpy(packed, w->qs + 16 * (row * (uint64_t)blocks + block), 16);
        return;
    }
    if (w->blocks != blocks) xe_fatal("q4: layout has %d blocks, matvec expects %d",
                                      w->blocks, blocks);
    uint64_t group = row >> 3;
    int lane = (int)(row & 7);
    const uint8_t *src = w->qs + (group * (uint64_t)blocks + block) * 128 + lane * 4;
    for (int chunk = 0; chunk < 4; chunk++)
        memcpy(packed + chunk * 4, src + chunk * 32, 4);
}

static float xe_q4_q8_dot_vector(const xe_q4 *w, uint64_t row, int n, const xe_q8 *q) {
    int blocks = n / 32;
    __m256 sum = _mm256_setzero_ps();
    int b = 0;

    for (; b + 7 < blocks; b += 8) {
        _Alignas(32) int integers[8];
        _Alignas(16) uint8_t packed[16];
        _Alignas(16) uint16_t wd_bits[8];
        for (int j = 0; j < 8; j++) {
            xe_q4_copy_block(w, row, blocks, b + j, packed);
            integers[j] = xe_q4_q8_raw_integer(packed, q->qs + 32 * (b + j));
            wd_bits[j] = xe_q4_scale(w, row, blocks, b + j);
        }
        __m256i vi = _mm256_load_si256((const __m256i *)integers);
        __m128i sigma16 = _mm_loadu_si128((const __m128i *)(q->sigma + b));
        __m256i correction = _mm256_slli_epi32(_mm256_cvtepi16_epi32(sigma16), 3);
        vi = _mm256_sub_epi32(vi, correction);
        __m256 wd = _mm256_cvtph_ps(_mm_load_si128((const __m128i *)wd_bits));
        __m256 qd = _mm256_cvtph_ps(_mm_load_si128((const __m128i *)(q->d + b)));
        __m256 scaled = _mm256_mul_ps(_mm256_cvtepi32_ps(vi), _mm256_mul_ps(wd, qd));
        sum = _mm256_add_ps(sum, scaled);
    }

    float result = xe_hsum8_ps(sum);
    for (; b < blocks; b++) {
        _Alignas(16) uint8_t packed[16];
        xe_q4_copy_block(w, row, blocks, b, packed);
        int integer = xe_q4_q8_raw_integer(packed, q->qs + 32 * b) -
                      8 * (int)q->sigma[b];
        result += (float)integer * _cvtsh_ss(xe_q4_scale(w, row, blocks, b)) *
                  _cvtsh_ss(xe_half_bits(q->d[b]));
    }
    return result;
}

static float xe_q4_q8_dot(const xe_q4 *w, uint64_t row, int n, const xe_q8 *q) {
    return xe_q4_q8_dot_vector(w, row, n, q);
}

static void xe_q4_q8_dot4(const xe_q4 *w, uint64_t first_row, int n,
                          const xe_q8 *q, float *out) {
    for (int r = 0; r < 4; r++)
        out[r] = xe_q4_q8_dot_vector(w, first_row + (uint64_t)r, n, q);
}

static inline uint32_t xe_load_u32(const void *p) {
    uint32_t value;
    memcpy(&value, p, sizeof(value));
    return value;
}

static inline __attribute__((always_inline)) __m256 xe_q4_q8_block8(
        const uint8_t *packed, const uint16_t *wd_bits, const xe_q8 *q, int block) {
    const __m256i mask = _mm256_set1_epi8(0x0f);
    __m256i integer = _mm256_setzero_si256();
    for (int chunk = 0; chunk < 4; chunk++) {
        __m256i weights = _mm256_load_si256((const __m256i *)(packed + chunk * 32));
        __m256i low = _mm256_and_si256(weights, mask);
        __m256i high = _mm256_and_si256(_mm256_srli_epi16(weights, 4), mask);
        __m256i qlow = _mm256_set1_epi32((int)xe_load_u32(q->qs + block * 32 + chunk * 4));
        __m256i qhigh = _mm256_set1_epi32((int)xe_load_u32(q->qs + block * 32 + 16 + chunk * 4));
        integer = _mm256_dpbusd_epi32(integer, low, qlow);
        integer = _mm256_dpbusd_epi32(integer, high, qhigh);
    }
    integer = _mm256_sub_epi32(integer,
        _mm256_set1_epi32(8 * (int)q->sigma[block]));
    __m256 wd = _mm256_cvtph_ps(_mm_load_si128((const __m128i *)wd_bits));
    __m256 qd = _mm256_set1_ps(_cvtsh_ss(xe_half_bits(q->d[block])));
    return _mm256_mul_ps(_mm256_cvtepi32_ps(integer), _mm256_mul_ps(wd, qd));
}

#define XE_DEFINE_Q4_Q8_DOT8(NAME, BLOCKS)                                      \
static __attribute__((noinline)) void NAME(const xe_q4 *w, uint64_t first_row,  \
                                            const xe_q8 *q, float out[8]) {      \
    const uint8_t *packed = w->qs + (first_row >> 3) * (uint64_t)(BLOCKS) * 128;\
    const uint16_t *wd = w->d + (first_row >> 3) * (uint64_t)(BLOCKS) * 8;       \
    __m256 s0 = _mm256_setzero_ps();                                             \
    __m256 s1 = _mm256_setzero_ps();                                             \
    __m256 s2 = _mm256_setzero_ps();                                             \
    __m256 s3 = _mm256_setzero_ps();                                             \
    __m256 s4 = _mm256_setzero_ps();                                             \
    __m256 s5 = _mm256_setzero_ps();                                             \
    __m256 s6 = _mm256_setzero_ps();                                             \
    __m256 s7 = _mm256_setzero_ps();                                             \
    int block = 0;                                                               \
    for (; block + 7 < (BLOCKS); block += 8) {                                   \
        s0 = _mm256_add_ps(s0, xe_q4_q8_block8(packed + (block + 0) * 128,       \
                                                wd + (block + 0) * 8, q, block + 0));\
        s1 = _mm256_add_ps(s1, xe_q4_q8_block8(packed + (block + 1) * 128,       \
                                                wd + (block + 1) * 8, q, block + 1));\
        s2 = _mm256_add_ps(s2, xe_q4_q8_block8(packed + (block + 2) * 128,       \
                                                wd + (block + 2) * 8, q, block + 2));\
        s3 = _mm256_add_ps(s3, xe_q4_q8_block8(packed + (block + 3) * 128,       \
                                                wd + (block + 3) * 8, q, block + 3));\
        s4 = _mm256_add_ps(s4, xe_q4_q8_block8(packed + (block + 4) * 128,       \
                                                wd + (block + 4) * 8, q, block + 4));\
        s5 = _mm256_add_ps(s5, xe_q4_q8_block8(packed + (block + 5) * 128,       \
                                                wd + (block + 5) * 8, q, block + 5));\
        s6 = _mm256_add_ps(s6, xe_q4_q8_block8(packed + (block + 6) * 128,       \
                                                wd + (block + 6) * 8, q, block + 6));\
        s7 = _mm256_add_ps(s7, xe_q4_q8_block8(packed + (block + 7) * 128,       \
                                                wd + (block + 7) * 8, q, block + 7));\
    }                                                                            \
    __m256 result = _mm256_add_ps(_mm256_add_ps(s0, s4), _mm256_add_ps(s1, s5));\
    result = _mm256_add_ps(result,                                                \
                           _mm256_add_ps(_mm256_add_ps(s2, s6),                  \
                                         _mm256_add_ps(s3, s7)));                \
    for (; block < (BLOCKS); block++)                                             \
        result = _mm256_add_ps(result, xe_q4_q8_block8(packed + block * 128,     \
                                                        wd + block * 8, q, block));\
    _mm256_storeu_ps(out, result);                                                \
}

XE_DEFINE_Q4_Q8_DOT8(xe_q4_q8_dot8_b22, 22)
XE_DEFINE_Q4_Q8_DOT8(xe_q4_q8_dot8_b66, 66)
XE_DEFINE_Q4_Q8_DOT8(xe_q4_q8_dot8_b88, 88)
XE_DEFINE_Q4_Q8_DOT8(xe_q4_q8_dot8_b128, 128)
XE_DEFINE_Q4_Q8_DOT8(xe_q4_q8_dot8_b256, 256)

#undef XE_DEFINE_Q4_Q8_DOT8

static inline void xe_q4_q8_dot8(const xe_q4 *w, uint64_t first_row, int n,
                                 const xe_q8 *q, float out[8]) {
    int blocks = n / 32;
    if (w->blocks == blocks && !(first_row & 7)) {
        switch (blocks) {
            case 22:  xe_q4_q8_dot8_b22(w, first_row, q, out); return;
            case 66:  xe_q4_q8_dot8_b66(w, first_row, q, out); return;
            case 88:  xe_q4_q8_dot8_b88(w, first_row, q, out); return;
            case 128: xe_q4_q8_dot8_b128(w, first_row, q, out); return;
            case 256: xe_q4_q8_dot8_b256(w, first_row, q, out); return;
        }
    }
    for (int row = 0; row < 8; row++)
        out[row] = xe_q4_q8_dot_vector(w, first_row + (uint64_t)row, n, q);
}

static void xe_matvec_q4_q8_rows4(const xe_q4 *w, uint64_t first_row, int n_in,
                                  int first_out, int end_out, const xe_q8 *q, float *y) {
    int row = first_out;
    while (row < end_out && ((first_row + (uint64_t)row) & 3)) {
        y[row] = xe_q4_q8_dot(w, first_row + (uint64_t)row, n_in, q);
        row++;
    }
    for (; row + 3 < end_out; row += 4)
        xe_q4_q8_dot4(w, first_row + (uint64_t)row, n_in, q, y + row);
    for (; row < end_out; row++)
        y[row] = xe_q4_q8_dot(w, first_row + (uint64_t)row, n_in, q);
}

static void xe_matvec_q4_q8_rows(const xe_q4 *w, uint64_t first_row, int n_in,
                                 int first_out, int end_out, const xe_q8 *q, float *y) {
    int row = first_out;
    while (row < end_out && ((first_row + (uint64_t)row) & 7)) {
        y[row] = xe_q4_q8_dot(w, first_row + (uint64_t)row, n_in, q);
        row++;
    }
    for (; row + 7 < end_out; row += 8)
        xe_q4_q8_dot8(w, first_row + (uint64_t)row, n_in, q, y + row);
    for (; row < end_out; row++)
        y[row] = xe_q4_q8_dot(w, first_row + (uint64_t)row, n_in, q);
}

static void xe_matvec_q4_q8_rows_vector(const xe_q4 *w, uint64_t first_row, int n_in,
                                        int first_out, int end_out, const xe_q8 *q, float *y) {
    for (int row = first_out; row < end_out; row++)
        y[row] = xe_q4_q8_dot_vector(w, first_row + (uint64_t)row, n_in, q);
}

static float xe_sumsq(const float *x, int n) {
    __m256 s0 = _mm256_setzero_ps();
    __m256 s1 = _mm256_setzero_ps();
    __m256 s2 = _mm256_setzero_ps();
    __m256 s3 = _mm256_setzero_ps();
    int i = 0;
    for (; i + 31 < n; i += 32) {
        __m256 x0 = _mm256_loadu_ps(x + i);
        __m256 x1 = _mm256_loadu_ps(x + i + 8);
        __m256 x2 = _mm256_loadu_ps(x + i + 16);
        __m256 x3 = _mm256_loadu_ps(x + i + 24);
        s0 = _mm256_fmadd_ps(x0, x0, s0);
        s1 = _mm256_fmadd_ps(x1, x1, s1);
        s2 = _mm256_fmadd_ps(x2, x2, s2);
        s3 = _mm256_fmadd_ps(x3, x3, s3);
    }
    float sum = xe_hsum8_ps(_mm256_add_ps(_mm256_add_ps(s0, s1), _mm256_add_ps(s2, s3)));
    for (; i < n; i++) sum += x[i] * x[i];
    return sum;
}

static float xe_rms_scale(const float *x, int n) {
    return 1.0f / sqrtf(xe_sumsq(x, n) / (float)n + XE_RMS_EPS);
}

static float xe_rms_scale_scalar(const float *x, int n) {
    float sum = 0.0f;
    for (int i = 0; i < n; i++) sum += x[i] * x[i];
    return 1.0f / sqrtf(sum / (float)n + XE_RMS_EPS);
}

static float xe_rms_scale_engine(const xe_engine *e, const float *x, int n) {
    return e->scalar_rms ? xe_rms_scale_scalar(x, n) : xe_rms_scale(x, n);
}

static void xe_rmsnorm_scale(const float *x, const float *weight, int n, float scale, float *y) {
    int i = 0;
    __m256 vs = _mm256_set1_ps(scale);
    if (weight) {
        for (; i + 7 < n; i += 8) {
            __m256 vx = _mm256_loadu_ps(x + i);
            __m256 vw = _mm256_loadu_ps(weight + i);
            _mm256_storeu_ps(y + i, _mm256_mul_ps(_mm256_mul_ps(vx, vs), vw));
        }
    } else {
        for (; i + 7 < n; i += 8)
            _mm256_storeu_ps(y + i, _mm256_mul_ps(_mm256_loadu_ps(x + i), vs));
    }
    for (; i < n; i++) y[i] = x[i] * scale * (weight ? weight[i] : 1.0f);
}

static void __attribute__((unused)) xe_rmsnorm(const float *x, const float *weight, int n, float *y) {
    xe_rmsnorm_scale(x, weight, n, xe_rms_scale(x, n), y);
}

static void xe_rmsnorm_engine(const xe_engine *e, const float *x, const float *weight,
                              int n, float *y) {
    xe_rmsnorm_scale(x, weight, n, xe_rms_scale_engine(e, x, n), y);
}

static void xe_rope_prepare(xe_session *s, int pos) {
    xe_engine *e = s->engine;
    for (int i = 0; i < XE_SWA_HEAD_DIM / 2; i++) {
        float theta = (float)pos * e->rope_swa_inv[i];
        s->rope_swa_cos[i] = cosf(theta);
        s->rope_swa_sin[i] = sinf(theta);
    }
    for (int i = 0; i < XE_GLOBAL_HEAD_DIM / 2; i++) {
        float theta = (float)pos * e->rope_global_inv[i];
        s->rope_global_cos[i] = cosf(theta);
        s->rope_global_sin[i] = sinf(theta);
    }
}

static void xe_rope_apply(float *heads, int n_heads, int head_dim,
                          const float *cosines, const float *sines) {
    int half = head_dim / 2;
    for (int h = 0; h < n_heads; h++) {
        float *head = heads + (size_t)h * head_dim;
        for (int i = 0; i < half; i++) {
            float lo = head[i];
            float hi = head[i + half];
            head[i] = lo * cosines[i] - hi * sines[i];
            head[i + half] = lo * sines[i] + hi * cosines[i];
        }
    }
}

typedef struct {
    const xe_q4 *w;
    uint64_t first_row;
    int n_in;
    int n_out;
    const xe_q8 *q;
    float *y;
} xe_matvec_arg;

typedef struct {
    xe_matvec_arg op[3];
    int n_ops;
} xe_matvec_batch_arg;

static void xe_matvec_phase(xe_session *s, const void *opaque, int worker, int workers) {
    (void)s;
    const xe_matvec_arg *a = opaque;
    int begin, end;
    if (!(a->n_out & 7) && !(a->first_row & 7)) {
        int groups = a->n_out / 8;
        begin = 8 * (groups * worker / workers);
        end = 8 * (groups * (worker + 1) / workers);
    } else {
        begin = a->n_out * worker / workers;
        end = a->n_out * (worker + 1) / workers;
    }
    xe_matvec_q4_q8_rows(a->w, a->first_row, a->n_in, begin, end, a->q, a->y);
}

static void xe_matvec_batch_phase(xe_session *s, const void *opaque, int worker, int workers) {
    (void)s;
    const xe_matvec_batch_arg *a = opaque;
    int total = 0;
    for (int i = 0; i < a->n_ops; i++) total += a->op[i].n_out;
    int begin, end;
    if (!(total & 7)) {
        int groups = total / 8;
        begin = 8 * (groups * worker / workers);
        end = 8 * (groups * (worker + 1) / workers);
    } else {
        begin = total * worker / workers;
        end = total * (worker + 1) / workers;
    }
    int offset = 0;
    for (int i = 0; i < a->n_ops; i++) {
        const xe_matvec_arg *op = &a->op[i];
        int lo = begin > offset ? begin - offset : 0;
        int hi = end < offset + op->n_out ? end - offset : op->n_out;
        if (lo < hi)
            xe_matvec_q4_q8_rows(op->w, op->first_row, op->n_in, lo, hi, op->q, op->y);
        offset += op->n_out;
    }
}

typedef struct {
    const float *x;
    const float *weight;
    float scale;
    int n;
    xe_q8 *q;
} xe_norm_q8_arg;

static void xe_norm_q8_phase(xe_session *s, const void *opaque, int worker, int workers) {
    (void)s;
    const xe_norm_q8_arg *a = opaque;
    int blocks = a->n / 32;
    int begin = blocks * worker / workers;
    int end = blocks * (worker + 1) / workers;
    for (int b = begin; b < end; b++) {
        float values[32];
        for (int i = 0; i < 32; i++) {
            int j = 32 * b + i;
            values[i] = a->x[j] * a->scale * a->weight[j];
        }
        xe_quantize_q8_block_at(values, a->q, b);
    }
}

typedef struct {
    const xe_layer *layer;
    float scale;
} xe_ffn_input_arg;

static void xe_ffn_input_phase(xe_session *s, const void *opaque, int worker, int workers) {
    const xe_ffn_input_arg *a = opaque;
    int blocks = XE_EMBD / 32;
    int begin = blocks * worker / workers;
    int end = blocks * (worker + 1) / workers;
    float router_scale = a->scale / sqrtf((float)XE_EMBD);
    for (int b = begin; b < end; b++) {
        float dense[32];
        float moe[32];
        for (int i = 0; i < 32; i++) {
            int j = 32 * b + i;
            float x = s->attn_out[j];
            dense[i] = x * a->scale * a->layer->ffn_norm[j];
            moe[i] = x * a->scale * a->layer->pre_ffw_norm2[j];
            s->router_in[j] = x * router_scale * a->layer->router_scale[j];
        }
        xe_quantize_q8_block_at(dense, &s->q8_dense_in, b);
        xe_quantize_q8_block_at(moe, &s->q8_moe_in, b);
    }
}

static uint16_t xe_half_bits(_Float16 x) {
    uint16_t bits;
    memcpy(&bits, &x, sizeof bits);
    return bits;
}

static float xe_gelu_lookup(const xe_engine *e, float x) {
    if (x <= -10.0f) return 0.0f;
    if (x >= 10.0f) return x;
    _Float16 h = (_Float16)x;
    return (float)e->gelu_lut[xe_half_bits(h)];
}

typedef struct {
    const xe_layer *layer;
} xe_dense_gate_arg;

static void xe_dense_gate_phase(xe_session *s, const void *opaque, int worker, int workers) {
    const xe_dense_gate_arg *a = opaque;
    int blocks = XE_DENSE_FFN / 32;
    int begin = blocks * worker / workers;
    int end = blocks * (worker + 1) / workers;
    for (int b = begin; b < end; b++) {
        float values[32];
        for (int i = 0; i < 32; i += 8) {
            int row = 32 * b + i;
            float gate[8], up[8];
            xe_q4_q8_dot8(&a->layer->ffn_gate, (uint64_t)row, XE_EMBD,
                           &s->q8_dense_in, gate);
            xe_q4_q8_dot8(&a->layer->ffn_up, (uint64_t)row, XE_EMBD,
                           &s->q8_dense_in, up);
            for (int lane = 0; lane < 8; lane++)
                values[i + lane] = xe_gelu_lookup(s->engine, gate[lane]) * up[lane];
        }
        xe_quantize_q8_block_at(values, &s->q8_dense_act, b);
    }
}

static float xe_dot_f32_avx(const float *a, const float *b, int n) {
    __m256 s0 = _mm256_setzero_ps();
    __m256 s1 = _mm256_setzero_ps();
    __m256 s2 = _mm256_setzero_ps();
    __m256 s3 = _mm256_setzero_ps();
    int i = 0;
    for (; i + 31 < n; i += 32) {
        s0 = _mm256_fmadd_ps(_mm256_loadu_ps(a + i), _mm256_loadu_ps(b + i), s0);
        s1 = _mm256_fmadd_ps(_mm256_loadu_ps(a + i + 8), _mm256_loadu_ps(b + i + 8), s1);
        s2 = _mm256_fmadd_ps(_mm256_loadu_ps(a + i + 16), _mm256_loadu_ps(b + i + 16), s2);
        s3 = _mm256_fmadd_ps(_mm256_loadu_ps(a + i + 24), _mm256_loadu_ps(b + i + 24), s3);
    }
    float sum = xe_hsum8_ps(_mm256_add_ps(_mm256_add_ps(s0, s1), _mm256_add_ps(s2, s3)));
    for (; i < n; i++) sum += a[i] * b[i];
    return sum;
}

static void xe_softmax_f32(float *x, int n) {
    float max = x[0];
    for (int i = 1; i < n; i++) if (x[i] > max) max = x[i];
    float sum = 0.0f;
    for (int i = 0; i < n; i++) {
        x[i] = expf(x[i] - max);
        sum += x[i];
    }
    float inv = 1.0f / sum;
    for (int i = 0; i < n; i++) x[i] *= inv;
}

static void xe_router(xe_session *s, const xe_layer *l, int layer) {
    for (int row = 0; row < XE_EXPERTS; row++)
        s->router_logits[row] = xe_dot_f32_avx(l->router_w + (size_t)row * XE_EMBD,
                                               s->router_in, XE_EMBD);
    xe_softmax_f32(s->router_logits, XE_EXPERTS);

    for (int i = 0; i < XE_EXPERTS_USED; i++) {
        s->experts[i] = -1;
        s->expert_weights[i] = -INFINITY;
    }
    for (int expert = 0; expert < XE_EXPERTS; expert++) {
        float value = s->router_logits[expert];
        if (value <= s->expert_weights[XE_EXPERTS_USED - 1]) continue;
        int slot = XE_EXPERTS_USED - 1;
        while (slot > 0 && s->expert_weights[slot - 1] < value) {
            s->expert_weights[slot] = s->expert_weights[slot - 1];
            s->experts[slot] = s->experts[slot - 1];
            slot--;
        }
        s->expert_weights[slot] = value;
        s->experts[slot] = expert;
    }
    float sum = 0.0f;
    for (int i = 0; i < XE_EXPERTS_USED; i++) sum += s->expert_weights[i];
    if (sum < 6.103515625e-5f) sum = 6.103515625e-5f;
    for (int i = 0; i < XE_EXPERTS_USED; i++) {
        s->expert_weights[i] /= sum;
        s->expert_trace[layer][i] = s->experts[i];
        s->expert_weight_trace[layer][i] = s->expert_weights[i];
    }
}

typedef struct {
    const xe_layer *layer;
} xe_moe_arg;

static void xe_moe_fused_gate_phase(xe_session *s, const void *opaque, int worker, int workers) {
    const xe_moe_arg *a = opaque;
    int tasks = XE_EXPERTS_USED * (XE_EXPERT_FFN / 32);
    int begin = tasks * worker / workers;
    int end = tasks * (worker + 1) / workers;
    for (int task = begin; task < end; task++) {
        int slot = task / (XE_EXPERT_FFN / 32);
        int block = task % (XE_EXPERT_FFN / 32);
        int expert = s->experts[slot];
        uint64_t base = (uint64_t)expert * (2 * XE_EXPERT_FFN);
        float values[32];
        for (int i = 0; i < 32; i += 8) {
            int row = 32 * block + i;
            float gate[8], up[8];
            xe_q4_q8_dot8(&a->layer->gate_up_exps, base + (uint64_t)row,
                           XE_EMBD, &s->q8_moe_in, gate);
            xe_q4_q8_dot8(&a->layer->gate_up_exps,
                           base + XE_EXPERT_FFN + (uint64_t)row,
                           XE_EMBD, &s->q8_moe_in, up);
            for (int lane = 0; lane < 8; lane++)
                values[i + lane] = xe_gelu_lookup(s->engine, gate[lane]) * up[lane];
        }
        xe_quantize_q8_block_at(values, &s->q8_expert[slot], block);
    }
}

static void xe_moe_down_phase(xe_session *s, const void *opaque, int worker, int workers) {
    const xe_moe_arg *a = opaque;
    int groups = XE_EMBD / 8;
    int begin = 8 * (groups * worker / workers);
    int end = 8 * (groups * (worker + 1) / workers);
    for (int row = begin; row < end; row += 8) {
        float sum[8] = { 0 };
        for (int slot = 0; slot < XE_EXPERTS_USED; slot++) {
            int expert = s->experts[slot];
            uint64_t wrow = (uint64_t)expert * XE_EMBD + (uint64_t)row;
            float value[8];
            xe_q4_q8_dot8(&a->layer->down_exps, wrow, XE_EXPERT_FFN,
                           &s->q8_expert[slot], value);
            for (int lane = 0; lane < 8; lane++)
                sum[lane] += s->expert_weights[slot] *
                             (value[lane] * a->layer->down_exps_scale[expert]);
        }
        memcpy(s->moe_out + row, sum, sizeof sum);
    }
}

static void xe_moe_generic_gate_phase(xe_session *s, const void *opaque, int worker, int workers) {
    const xe_moe_arg *a = opaque;
    float *temporary = s->attn_partial;
    int rows = XE_EXPERTS_USED * 2 * XE_EXPERT_FFN;
    int groups = rows / 8;
    int begin = 8 * (groups * worker / workers);
    int end = 8 * (groups * (worker + 1) / workers);
    for (int row = begin; row < end; row += 8) {
        int slot = row / (2 * XE_EXPERT_FFN);
        int local = row % (2 * XE_EXPERT_FFN);
        uint64_t wrow = (uint64_t)s->experts[slot] * (2 * XE_EXPERT_FFN) + (uint64_t)local;
        xe_q4_q8_dot8(&a->layer->gate_up_exps, wrow, XE_EMBD, &s->q8_moe_in,
                       temporary + row);
    }
}

static void xe_moe_generic_act_phase(xe_session *s, const void *opaque, int worker, int workers) {
    (void)opaque;
    float *temporary = s->attn_partial;
    int tasks = XE_EXPERTS_USED * (XE_EXPERT_FFN / 32);
    int begin = tasks * worker / workers;
    int end = tasks * (worker + 1) / workers;
    for (int task = begin; task < end; task++) {
        int slot = task / (XE_EXPERT_FFN / 32);
        int block = task % (XE_EXPERT_FFN / 32);
        float *base = temporary + slot * (2 * XE_EXPERT_FFN);
        float values[32];
        for (int i = 0; i < 32; i++) {
            int j = 32 * block + i;
            values[i] = xe_gelu_lookup(s->engine, base[j]) * base[XE_EXPERT_FFN + j];
        }
        xe_quantize_q8_block_at(values, &s->q8_expert[slot], block);
    }
}

static void xe_moe_expert_phase(xe_session *s, const void *opaque, int worker, int workers) {
    const xe_moe_arg *a = opaque;
    float *partials = s->attn_partial;
    int begin = XE_EXPERTS_USED * worker / workers;
    int end = XE_EXPERTS_USED * (worker + 1) / workers;
    for (int slot = begin; slot < end; slot++) {
        int expert = s->experts[slot];
        uint64_t base = (uint64_t)expert * (2 * XE_EXPERT_FFN);
        for (int block = 0; block < XE_EXPERT_FFN / 32; block++) {
            float values[32];
            for (int i = 0; i < 32; i += 8) {
                int row = 32 * block + i;
                float gate[8], up[8];
                xe_q4_q8_dot8(&a->layer->gate_up_exps, base + (uint64_t)row,
                               XE_EMBD, &s->q8_moe_in, gate);
                xe_q4_q8_dot8(&a->layer->gate_up_exps,
                               base + XE_EXPERT_FFN + (uint64_t)row,
                               XE_EMBD, &s->q8_moe_in, up);
                for (int lane = 0; lane < 8; lane++)
                    values[i + lane] = xe_gelu_lookup(s->engine, gate[lane]) * up[lane];
            }
            xe_quantize_q8_block_at(values, &s->q8_expert[slot], block);
        }
        float factor = s->expert_weights[slot] * a->layer->down_exps_scale[expert];
        float *out = partials + (size_t)slot * XE_EMBD;
        for (int row = 0; row < XE_EMBD; row += 8) {
            uint64_t wrow = (uint64_t)expert * XE_EMBD + (uint64_t)row;
            float value[8];
            xe_q4_q8_dot8(&a->layer->down_exps, wrow, XE_EXPERT_FFN,
                           &s->q8_expert[slot], value);
            for (int lane = 0; lane < 8; lane++) out[row + lane] = factor * value[lane];
        }
    }
}

static void xe_moe_reduce_phase(xe_session *s, const void *opaque, int worker, int workers) {
    (void)opaque;
    int begin = XE_EMBD * worker / workers;
    int end = XE_EMBD * (worker + 1) / workers;
    for (int row = begin; row < end; row++) {
        float sum = 0.0f;
        for (int slot = 0; slot < XE_EXPERTS_USED; slot++)
            sum += s->attn_partial[(size_t)slot * XE_EMBD + row];
        s->moe_out[row] = sum;
    }
}

typedef struct {
    const uint8_t *base;
    size_t size;
    size_t pos;
} xe_cur;

static void xe_need(xe_cur *c, size_t n) {
    if (n > c->size - c->pos)
        xe_fatal("truncated GGUF at byte %zu (need %zu bytes, %zu remain)", c->pos, n, c->size - c->pos);
}

static uint8_t xe_u8(xe_cur *c) {
    xe_need(c, 1);
    return c->base[c->pos++];
}

static uint32_t xe_u32(xe_cur *c) {
    xe_need(c, 4);
    uint32_t v;
    memcpy(&v, c->base + c->pos, 4);
    c->pos += 4;
    return v;
}

static uint64_t xe_u64(xe_cur *c) {
    xe_need(c, 8);
    uint64_t v;
    memcpy(&v, c->base + c->pos, 8);
    c->pos += 8;
    return v;
}

static float xe_f32(xe_cur *c) {
    xe_need(c, 4);
    float v;
    memcpy(&v, c->base + c->pos, 4);
    c->pos += 4;
    return v;
}

static xe_str xe_read_str(xe_cur *c) {
    uint64_t len = xe_u64(c);
    xe_need(c, len);
    xe_str s = { (const char *)(c->base + c->pos), len };
    c->pos += len;
    return s;
}

static int xe_str_eq(xe_str s, const char *lit) {
    size_t n = strlen(lit);
    return s.len == n && memcmp(s.p, lit, n) == 0;
}

static size_t xe_type_size(uint32_t t) {
    switch (t) {
        case XE_T_U8: case XE_T_I8: case XE_T_BOOL: return 1;
        case XE_T_U16: case XE_T_I16: return 2;
        case XE_T_U32: case XE_T_I32: case XE_T_F32: return 4;
        case XE_T_U64: case XE_T_I64: case XE_T_F64: return 8;
        default: xe_fatal("gguf: bad scalar value type %u", t);
    }
}

static void xe_skip_value(xe_cur *c, uint32_t type) {
    if (type == XE_T_STRING) { xe_read_str(c); return; }
    if (type == XE_T_ARRAY) {
        uint32_t etype = xe_u32(c);
        uint64_t n = xe_u64(c);
        if (etype == XE_T_STRING) {
            for (uint64_t i = 0; i < n; i++) xe_read_str(c);
        } else if (etype == XE_T_ARRAY) {
            xe_fatal("gguf: nested arrays not supported");
        } else {
            size_t sz = xe_type_size(etype);
            xe_need(c, sz * n);
            c->pos += sz * n;
        }
        return;
    }
    size_t sz = xe_type_size(type);
    xe_need(c, sz);
    c->pos += sz;
}

static void xe_capture_array(xe_cur *c, uint32_t vtype, const char *key, const void **ptr, uint64_t *count) {
    if (vtype != XE_T_ARRAY) xe_fatal("%s: expected array, found value type %u", key, vtype);
    uint32_t etype = xe_u32(c);
    uint64_t n = xe_u64(c);
    *ptr = c->base + c->pos;
    *count = n;
    if (etype == XE_T_STRING) {
        for (uint64_t i = 0; i < n; i++) xe_read_str(c);
    } else {
        size_t sz = xe_type_size(etype);
        xe_need(c, sz * n);
        c->pos += sz * n;
    }
}

static uint32_t xe_expect_u32(xe_cur *c, uint32_t type, const char *key) {
    if (type != XE_T_U32) xe_fatal("%s: expected u32, found value type %u", key, type);
    return xe_u32(c);
}

static float xe_expect_f32(xe_cur *c, uint32_t type, const char *key) {
    if (type != XE_T_F32) xe_fatal("%s: expected f32, found value type %u", key, type);
    return xe_f32(c);
}

static int xe_expect_bool(xe_cur *c, uint32_t type, const char *key) {
    if (type != XE_T_BOOL) xe_fatal("%s: expected bool, found value type %u", key, type);
    return xe_u8(c) != 0;
}

static xe_str xe_expect_string(xe_cur *c, uint32_t type, const char *key) {
    if (type != XE_T_STRING) xe_fatal("%s: expected string, found value type %u", key, type);
    return xe_read_str(c);
}

static uint64_t xe_fnv1a(const char *p, uint32_t n);

static void xe_expect_array_header(xe_cur *c, uint32_t vtype, uint32_t want_etype, uint64_t want_n, const char *key) {
    if (vtype != XE_T_ARRAY) xe_fatal("%s: expected array, found value type %u", key, vtype);
    uint32_t etype = xe_u32(c);
    uint64_t n = xe_u64(c);
    if (etype != want_etype || n != want_n)
        xe_fatal("%s: expected array[%u] of length %llu, found type %u length %llu",
                 key, want_etype, (unsigned long long)want_n, etype, (unsigned long long)n);
}

enum {
    K_ARCH, K_BLOCK_COUNT, K_CTX_LEN, K_EMBD_LEN, K_FFN_LEN, K_HEAD_COUNT,
    K_EXPERT_COUNT, K_EXPERT_USED, K_EXPERT_FFN, K_SWA_WIN, K_KEY_LEN, K_VAL_LEN,
    K_KEY_LEN_SWA, K_VAL_LEN_SWA, K_ROPE_BASE, K_ROPE_BASE_SWA, K_SOFTCAP, K_RMS_EPS,
    K_HEAD_COUNT_KV, K_SWA_PATTERN, K_TOK_MODEL, K_BOS, K_EOS, K_EOT, K_UNK, K_PAD,
    K_ADD_SPACE, K_ADD_BOS, K_CHAT_TEMPLATE, K_REQUIRED_COUNT
};

static const char *xe_req_keys[K_REQUIRED_COUNT] = {
    "general.architecture",
    "gemma4.block_count",
    "gemma4.context_length",
    "gemma4.embedding_length",
    "gemma4.feed_forward_length",
    "gemma4.attention.head_count",
    "gemma4.expert_count",
    "gemma4.expert_used_count",
    "gemma4.expert_feed_forward_length",
    "gemma4.attention.sliding_window",
    "gemma4.attention.key_length",
    "gemma4.attention.value_length",
    "gemma4.attention.key_length_swa",
    "gemma4.attention.value_length_swa",
    "gemma4.rope.freq_base",
    "gemma4.rope.freq_base_swa",
    "gemma4.final_logit_softcapping",
    "gemma4.attention.layer_norm_rms_epsilon",
    "gemma4.attention.head_count_kv",
    "gemma4.attention.sliding_window_pattern",
    "tokenizer.ggml.model",
    "tokenizer.ggml.bos_token_id",
    "tokenizer.ggml.eos_token_id",
    "tokenizer.ggml.eot_token_id",
    "tokenizer.ggml.unknown_token_id",
    "tokenizer.ggml.padding_token_id",
    "tokenizer.ggml.add_space_prefix",
    "tokenizer.ggml.add_bos_token",
    "tokenizer.chat_template",
};

#define XE_KEY(lit) xe_str_eq(key, lit)

#define XE_REQ_U32(lit, kidx, expect) \
    if (XE_KEY(lit)) { \
        uint32_t v = xe_expect_u32(c, vtype, lit); \
        if (v != (uint32_t)(expect)) xe_fatal(lit ": expected %u, found %u", (unsigned)(expect), v); \
        seen |= 1u << (kidx); \
        continue; \
    }

#define XE_REQ_F32(lit, kidx, expect) \
    if (XE_KEY(lit)) { \
        float v = xe_expect_f32(c, vtype, lit); \
        if (v != (float)(expect)) xe_fatal(lit ": expected %g, found %g", (double)(expect), (double)v); \
        seen |= 1u << (kidx); \
        continue; \
    }

#define XE_REQ_BOOL(lit, kidx, expect) \
    if (XE_KEY(lit)) { \
        int v = xe_expect_bool(c, vtype, lit); \
        if (v != (expect)) xe_fatal(lit ": expected %d, found %d", (int)(expect), v); \
        seen |= 1u << (kidx); \
        continue; \
    }

#define XE_REQ_STR(lit, kidx, expect) \
    if (XE_KEY(lit)) { \
        xe_str v = xe_expect_string(c, vtype, lit); \
        size_t n = strlen(expect); \
        if (v.len != n || memcmp(v.p, expect, n)) xe_fatal(lit ": expected \"%s\", found \"%.*s\"", expect, (int)v.len, v.p); \
        seen |= 1u << (kidx); \
        continue; \
    }

#define XE_REQ_U32_STORE(lit, kidx, expect, dest) \
    if (XE_KEY(lit)) { \
        uint32_t v = xe_expect_u32(c, vtype, lit); \
        if (v != (uint32_t)(expect)) xe_fatal(lit ": expected %u, found %u", (unsigned)(expect), v); \
        (dest) = (int32_t)v; \
        seen |= 1u << (kidx); \
        continue; \
    }

enum { XE_TK_SINGLETON, XE_TK_ALL, XE_TK_SWA, XE_TK_GLOBAL };
enum { XE_CAT_DENSE, XE_CAT_EXPERT, XE_CAT_EMBD };

typedef struct {
    const char *suffix;
    int kind;
    int qtype;
    uint64_t ne0, ne1, ne2;
    size_t off;
    int cat;
} xe_tspec;

static const xe_tspec xe_specs[] = {
    { "token_embd.weight", XE_TK_SINGLETON, XE_GGML_TYPE_Q4_0, XE_EMBD, XE_VOCAB, 1, offsetof(struct xe_engine, tok_embd), XE_CAT_EMBD },
    { "output_norm.weight", XE_TK_SINGLETON, XE_GGML_TYPE_F32, XE_EMBD, 1, 1, offsetof(struct xe_engine, out_norm), XE_CAT_DENSE },
    { "rope_freqs.weight", XE_TK_SINGLETON, XE_GGML_TYPE_F32, XE_SWA_HEAD_DIM, 1, 1, offsetof(struct xe_engine, rope_freqs), XE_CAT_DENSE },

    { "attn_norm.weight", XE_TK_ALL, XE_GGML_TYPE_F32, XE_EMBD, 1, 1, offsetof(xe_layer, attn_norm), XE_CAT_DENSE },
    { "post_attention_norm.weight", XE_TK_ALL, XE_GGML_TYPE_F32, XE_EMBD, 1, 1, offsetof(xe_layer, post_attn_norm), XE_CAT_DENSE },
    { "ffn_norm.weight", XE_TK_ALL, XE_GGML_TYPE_F32, XE_EMBD, 1, 1, offsetof(xe_layer, ffn_norm), XE_CAT_DENSE },
    { "post_ffw_norm.weight", XE_TK_ALL, XE_GGML_TYPE_F32, XE_EMBD, 1, 1, offsetof(xe_layer, post_ffw_norm), XE_CAT_DENSE },
    { "post_ffw_norm_1.weight", XE_TK_ALL, XE_GGML_TYPE_F32, XE_EMBD, 1, 1, offsetof(xe_layer, post_ffw_norm1), XE_CAT_DENSE },
    { "post_ffw_norm_2.weight", XE_TK_ALL, XE_GGML_TYPE_F32, XE_EMBD, 1, 1, offsetof(xe_layer, post_ffw_norm2), XE_CAT_DENSE },
    { "pre_ffw_norm_2.weight", XE_TK_ALL, XE_GGML_TYPE_F32, XE_EMBD, 1, 1, offsetof(xe_layer, pre_ffw_norm2), XE_CAT_DENSE },
    { "layer_output_scale.weight", XE_TK_ALL, XE_GGML_TYPE_F32, 1, 1, 1, offsetof(xe_layer, layer_out_scale), XE_CAT_DENSE },
    { "ffn_gate.weight", XE_TK_ALL, XE_GGML_TYPE_Q4_0, XE_EMBD, XE_DENSE_FFN, 1, offsetof(xe_layer, ffn_gate), XE_CAT_DENSE },
    { "ffn_up.weight", XE_TK_ALL, XE_GGML_TYPE_Q4_0, XE_EMBD, XE_DENSE_FFN, 1, offsetof(xe_layer, ffn_up), XE_CAT_DENSE },
    { "ffn_down.weight", XE_TK_ALL, XE_GGML_TYPE_Q4_0, XE_DENSE_FFN, XE_EMBD, 1, offsetof(xe_layer, ffn_down), XE_CAT_DENSE },
    { "ffn_gate_inp.weight", XE_TK_ALL, XE_GGML_TYPE_F32, XE_EMBD, XE_EXPERTS, 1, offsetof(xe_layer, router_w), XE_CAT_DENSE },
    { "ffn_gate_inp.scale", XE_TK_ALL, XE_GGML_TYPE_F32, XE_EMBD, 1, 1, offsetof(xe_layer, router_scale), XE_CAT_DENSE },
    { "ffn_gate_up_exps.weight", XE_TK_ALL, XE_GGML_TYPE_Q4_0, XE_EMBD, XE_EXPERT_FFN * 2, XE_EXPERTS, offsetof(xe_layer, gate_up_exps), XE_CAT_EXPERT },
    { "ffn_down_exps.weight", XE_TK_ALL, XE_GGML_TYPE_Q4_0, XE_EXPERT_FFN, XE_EMBD, XE_EXPERTS, offsetof(xe_layer, down_exps), XE_CAT_EXPERT },
    { "ffn_down_exps.scale", XE_TK_ALL, XE_GGML_TYPE_F32, XE_EXPERTS, 1, 1, offsetof(xe_layer, down_exps_scale), XE_CAT_DENSE },

    { "attn_q.weight", XE_TK_SWA, XE_GGML_TYPE_Q4_0, XE_EMBD, XE_Q_HEADS * XE_SWA_HEAD_DIM, 1, offsetof(xe_layer, attn_q), XE_CAT_DENSE },
    { "attn_k.weight", XE_TK_SWA, XE_GGML_TYPE_Q4_0, XE_EMBD, XE_SWA_KV_HEADS * XE_SWA_HEAD_DIM, 1, offsetof(xe_layer, attn_k), XE_CAT_DENSE },
    { "attn_v.weight", XE_TK_SWA, XE_GGML_TYPE_Q4_0, XE_EMBD, XE_SWA_KV_HEADS * XE_SWA_HEAD_DIM, 1, offsetof(xe_layer, attn_v), XE_CAT_DENSE },
    { "attn_output.weight", XE_TK_SWA, XE_GGML_TYPE_Q4_0, XE_Q_HEADS * XE_SWA_HEAD_DIM, XE_EMBD, 1, offsetof(xe_layer, attn_o), XE_CAT_DENSE },
    { "attn_q_norm.weight", XE_TK_SWA, XE_GGML_TYPE_F32, XE_SWA_HEAD_DIM, 1, 1, offsetof(xe_layer, q_norm), XE_CAT_DENSE },
    { "attn_k_norm.weight", XE_TK_SWA, XE_GGML_TYPE_F32, XE_SWA_HEAD_DIM, 1, 1, offsetof(xe_layer, k_norm), XE_CAT_DENSE },

    { "attn_q.weight", XE_TK_GLOBAL, XE_GGML_TYPE_Q4_0, XE_EMBD, XE_Q_HEADS * XE_GLOBAL_HEAD_DIM, 1, offsetof(xe_layer, attn_q), XE_CAT_DENSE },
    { "attn_k.weight", XE_TK_GLOBAL, XE_GGML_TYPE_Q4_0, XE_EMBD, XE_GLOBAL_KV_HEADS * XE_GLOBAL_HEAD_DIM, 1, offsetof(xe_layer, attn_k), XE_CAT_DENSE },
    { "attn_output.weight", XE_TK_GLOBAL, XE_GGML_TYPE_Q4_0, XE_Q_HEADS * XE_GLOBAL_HEAD_DIM, XE_EMBD, 1, offsetof(xe_layer, attn_o), XE_CAT_DENSE },
    { "attn_q_norm.weight", XE_TK_GLOBAL, XE_GGML_TYPE_F32, XE_GLOBAL_HEAD_DIM, 1, 1, offsetof(xe_layer, q_norm), XE_CAT_DENSE },
    { "attn_k_norm.weight", XE_TK_GLOBAL, XE_GGML_TYPE_F32, XE_GLOBAL_HEAD_DIM, 1, 1, offsetof(xe_layer, k_norm), XE_CAT_DENSE },
};

#define XE_NSPECS (sizeof(xe_specs) / sizeof(xe_specs[0]))

static const xe_tspec *xe_find_spec(const char *suffix, int kind, int *suffix_known) {
    *suffix_known = 0;
    for (size_t i = 0; i < XE_NSPECS; i++) {
        const xe_tspec *s = &xe_specs[i];
        if (strcmp(s->suffix, suffix)) continue;
        *suffix_known = 1;
        if (s->kind == XE_TK_ALL || s->kind == kind) return s;
    }
    return NULL;
}

static uint64_t xe_align_up(uint64_t v, uint64_t a) {
    return (v + a - 1) / a * a;
}

typedef struct {
    void *target;
    uint64_t offset;
    uint64_t size;
} xe_bind_rec;

typedef struct {
    xe_q4 *field;
    uint64_t nblocks;
    int blocks;
    uint64_t rows;
    const char *suffix;
    int layer_idx;
    uint8_t *qs_seg;
    uint16_t *d_seg;
    int part;
    uint64_t part_offset;
    uint64_t source_offset;
} xe_repack_job;

static int xe_repack_source_order(const void *a, const void *b) {
    const xe_repack_job *x = a, *y = b;
    return (x->source_offset > y->source_offset) -
           (x->source_offset < y->source_offset);
}

static void xe_repack_release(xe_engine *e, uint64_t *released,
                              uint64_t consumed, const char *gguf_path) {
    uint64_t end = consumed & ~UINT64_C(4095);
    if (end <= *released) return;
    size_t bytes = (size_t)(end - *released);
    if (madvise((uint8_t *)e->map + *released, bytes, MADV_DONTNEED) != 0)
        xe_fatal("%s: madvise consumed source: %s", gguf_path, strerror(errno));
    int error = posix_fadvise(e->map_fd, (off_t)*released, (off_t)bytes,
                             POSIX_FADV_DONTNEED);
    if (error)
        xe_fatal("%s: fadvise consumed source: %s", gguf_path, strerror(error));
    *released = end;
}

typedef struct {
    xe_repack_job *jobs;
    int njobs;
    atomic_int *next;
    const char *gguf_path;
    uint64_t verified;
} xe_repack_worker_arg;

static void *xe_repack_worker(void *arg_) {
    xe_repack_worker_arg *arg = arg_;
    for (;;) {
        int i = atomic_fetch_add(arg->next, 1);
        if (i >= arg->njobs) break;
        xe_repack_job *j = &arg->jobs[i];

        const uint8_t *src = j->field->qs;
        if (j->rows & 7) xe_fatal("%s: q4 repack rows not divisible by eight: %s",
                                  arg->gguf_path, j->suffix);
        for (uint64_t group = 0; group < j->rows / 8; group++) {
            for (int block = 0; block < j->blocks; block++) {
                uint8_t *dst_qs = j->qs_seg + (group * j->blocks + block) * 128;
                uint16_t *dst_d = j->d_seg + (group * j->blocks + block) * 8;
                for (int row = 0; row < 8; row++) {
                    uint64_t native = (group * 8 + row) * j->blocks + block;
                    const uint8_t *blk = src + (size_t)native * 18;
                    for (int chunk = 0; chunk < 4; chunk++)
                        memcpy(dst_qs + chunk * 32 + row * 4, blk + 2 + chunk * 4, 4);
                    memcpy(&dst_d[row], blk, 2);
                }
            }
        }

        j->field->qs = j->qs_seg;
        j->field->d = j->d_seg;
        j->field->blocks = j->blocks;

        uint64_t cand[3] = { 0, j->nblocks / 2, j->nblocks - 1 };
        uint64_t uniq[3];
        int nuniq = 0;
        for (int k = 0; k < 3; k++) {
            int dup = 0;
            for (int m = 0; m < nuniq; m++) if (uniq[m] == cand[k]) dup = 1;
            if (!dup) uniq[nuniq++] = cand[k];
        }
#ifdef XE_REPACK_VERIFY_ALL
        uint64_t verify_count = j->nblocks;
#else
        uint64_t verify_count = (uint64_t)nuniq;
#endif
        for (uint64_t k = 0; k < verify_count; k++) {
#ifdef XE_REPACK_VERIFY_ALL
            uint64_t b = k;
#else
            uint64_t b = uniq[k];
#endif
            uint64_t row = b / j->blocks;
            int block = (int)(b % j->blocks);
            uint8_t recon[18];
            memcpy(recon, &j->d_seg[((row >> 3) * j->blocks + block) * 8 + (row & 7)], 2);
            const uint8_t *packed = j->qs_seg + ((row >> 3) * j->blocks + block) * 128 +
                                    (row & 7) * 4;
            for (int chunk = 0; chunk < 4; chunk++)
                memcpy(recon + 2 + chunk * 4, packed + chunk * 32, 4);
            if (memcmp(recon, src + (size_t)b * 18, 18) != 0) {
                if (j->layer_idx < 0)
                    xe_fatal("%s: repack verify failed: %s block %llu", arg->gguf_path, j->suffix, (unsigned long long)b);
                else
                    xe_fatal("%s: repack verify failed: blk.%d.%s block %llu", arg->gguf_path, j->layer_idx, j->suffix, (unsigned long long)b);
            }
            arg->verified++;
        }
    }
    return NULL;
}

static void xe_copy_f32(xe_engine *e) {
    uint64_t total = 0;
    for (size_t i = 0; i < XE_NSPECS; i++) {
        const xe_tspec *s = &xe_specs[i];
        if (s->qtype != XE_GGML_TYPE_F32) continue;
        uint64_t bytes = 4 * s->ne0 * s->ne1 * s->ne2;
        int count = s->kind == XE_TK_SINGLETON ? 1 :
                    s->kind == XE_TK_ALL ? XE_LAYERS :
                    s->kind == XE_TK_SWA ? 25 : 5;
        total += xe_align_up(bytes, 64) * (uint64_t)count;
    }
    if (total > XE_REPACK_PART_LIMIT)
        xe_fatal("F32 weight slab exceeds shared allocation limit: %llu",
                 (unsigned long long)total);
    e->f32_slab = xe_alloc(e, total, XE_MEM_SHARED);
    e->f32_slab_size = total;
    uint8_t *cursor = e->f32_slab;

    for (size_t i = 0; i < XE_NSPECS; i++) {
        const xe_tspec *s = &xe_specs[i];
        if (s->qtype != XE_GGML_TYPE_F32) continue;
        uint64_t bytes = 4 * s->ne0 * s->ne1 * s->ne2;
        uint64_t padded = xe_align_up(bytes, 64);
        if (s->kind == XE_TK_SINGLETON) {
            const float **field = (const float **)((char *)e + s->off);
            memcpy(cursor, *field, bytes);
#ifdef XE_REPACK_VERIFY_ALL
            if (memcmp(cursor, *field, bytes) != 0)
                xe_fatal("F32 copy verification failed: %s", s->suffix);
#endif
            *field = (const float *)cursor;
            cursor += padded;
            continue;
        }
        for (int l = 0; l < XE_LAYERS; l++) {
            if (s->kind == XE_TK_SWA && XE_IS_GLOBAL(l)) continue;
            if (s->kind == XE_TK_GLOBAL && !XE_IS_GLOBAL(l)) continue;
            const float **field = (const float **)((char *)&e->layers[l] + s->off);
            memcpy(cursor, *field, bytes);
#ifdef XE_REPACK_VERIFY_ALL
            if (memcmp(cursor, *field, bytes) != 0)
                xe_fatal("F32 copy verification failed: blk.%d.%s", l, s->suffix);
#endif
            *field = (const float *)cursor;
            cursor += padded;
        }
    }
    if (cursor != (uint8_t *)e->f32_slab + total)
        xe_fatal("F32 weight slab layout mismatch");
}

static xe_engine *xe_map_file(const char *gguf_path) {
    int fd = open(gguf_path, O_RDONLY);
    if (fd < 0) xe_fatal("%s: %s", gguf_path, strerror(errno));

    struct stat st;
    if (fstat(fd, &st) < 0) xe_fatal("%s: fstat: %s", gguf_path, strerror(errno));
    if (st.st_size <= 0) xe_fatal("%s: empty file", gguf_path);

    void *map = mmap(NULL, (size_t)st.st_size, PROT_READ, MAP_PRIVATE, fd, 0);
    if (map == MAP_FAILED) xe_fatal("%s: mmap: %s", gguf_path, strerror(errno));
    xe_engine *e = calloc(1, sizeof *e);
    if (!e) xe_fatal("out of memory allocating engine");
    e->map = map;
    e->map_len = (size_t)st.st_size;
    e->map_fd = fd;
    e->bos_id = e->eos_id = e->eot_id = e->unk_id = e->pad_id = -1;

    return e;
}

static xe_cur xe_parse_header(xe_engine *e, const char *gguf_path) {
    xe_cur c = { (const uint8_t *)e->map, e->map_len, 0 };

    uint32_t magic = xe_u32(&c);
    if (magic != XE_GGUF_MAGIC) xe_fatal("%s: bad GGUF magic 0x%08x", gguf_path, magic);

    uint32_t version = xe_u32(&c);
    if (version != XE_GGUF_VERSION) xe_fatal("%s: unsupported GGUF version %u (expected %u)", gguf_path, version, (unsigned)XE_GGUF_VERSION);
    e->gguf_version = version;

    uint64_t n_tensors = xe_u64(&c);
    if (!e->vocab_only && n_tensors != XE_TENSOR_COUNT)
        xe_fatal("%s: expected %d tensors, found %llu", gguf_path, XE_TENSOR_COUNT, (unsigned long long)n_tensors);

    e->n_kv = xe_u64(&c);

    return c;
}

static void xe_parse_metadata(xe_engine *e, xe_cur *c, const char *gguf_path) {
    uint32_t seen = 0;
    uint64_t alignment = 32;

    for (uint64_t i = 0; i < e->n_kv; i++) {
        xe_str key = xe_read_str(c);
        uint32_t vtype = xe_u32(c);

        XE_REQ_STR("general.architecture", K_ARCH, "gemma4")
        XE_REQ_U32("gemma4.block_count", K_BLOCK_COUNT, XE_LAYERS)
        XE_REQ_U32("gemma4.context_length", K_CTX_LEN, XE_MODEL_CTX)
        XE_REQ_U32("gemma4.embedding_length", K_EMBD_LEN, XE_EMBD)
        XE_REQ_U32("gemma4.feed_forward_length", K_FFN_LEN, XE_DENSE_FFN)
        XE_REQ_U32("gemma4.attention.head_count", K_HEAD_COUNT, XE_Q_HEADS)
        XE_REQ_U32("gemma4.expert_count", K_EXPERT_COUNT, XE_EXPERTS)
        XE_REQ_U32("gemma4.expert_used_count", K_EXPERT_USED, XE_EXPERTS_USED)
        XE_REQ_U32("gemma4.expert_feed_forward_length", K_EXPERT_FFN, XE_EXPERT_FFN)
        XE_REQ_U32("gemma4.attention.sliding_window", K_SWA_WIN, XE_SWA_WINDOW)
        XE_REQ_U32("gemma4.attention.key_length", K_KEY_LEN, XE_GLOBAL_HEAD_DIM)
        XE_REQ_U32("gemma4.attention.value_length", K_VAL_LEN, XE_GLOBAL_HEAD_DIM)
        XE_REQ_U32("gemma4.attention.key_length_swa", K_KEY_LEN_SWA, XE_SWA_HEAD_DIM)
        XE_REQ_U32("gemma4.attention.value_length_swa", K_VAL_LEN_SWA, XE_SWA_HEAD_DIM)
        XE_REQ_F32("gemma4.rope.freq_base", K_ROPE_BASE, XE_GLOBAL_ROPE_BASE)
        XE_REQ_F32("gemma4.rope.freq_base_swa", K_ROPE_BASE_SWA, XE_SWA_ROPE_BASE)
        XE_REQ_F32("gemma4.final_logit_softcapping", K_SOFTCAP, XE_LOGIT_SOFTCAP)
        XE_REQ_STR("tokenizer.ggml.model", K_TOK_MODEL, "gemma4")
        XE_REQ_U32_STORE("tokenizer.ggml.bos_token_id", K_BOS, 2, e->bos_id)
        XE_REQ_U32_STORE("tokenizer.ggml.eos_token_id", K_EOS, 1, e->eos_id)
        XE_REQ_U32_STORE("tokenizer.ggml.eot_token_id", K_EOT, 106, e->eot_id)
        XE_REQ_U32_STORE("tokenizer.ggml.unknown_token_id", K_UNK, 3, e->unk_id)
        XE_REQ_U32_STORE("tokenizer.ggml.padding_token_id", K_PAD, 0, e->pad_id)
        XE_REQ_BOOL("tokenizer.ggml.add_space_prefix", K_ADD_SPACE, 0)
        XE_REQ_BOOL("tokenizer.ggml.add_bos_token", K_ADD_BOS, 0)

        if (XE_KEY("tokenizer.chat_template")) {
            xe_str v = xe_expect_string(c, vtype, "tokenizer.chat_template");
            if (v.len != XE_CHAT_TEMPLATE_LEN ||
                xe_fnv1a(v.p, (uint32_t)v.len) != XE_CHAT_TEMPLATE_HASH)
                xe_fatal("tokenizer.chat_template: unsupported Gemma 4 template");
            e->chat_template = v;
            seen |= 1u << K_CHAT_TEMPLATE;
            continue;
        }

        if (XE_KEY("gemma4.attention.layer_norm_rms_epsilon")) {
            float v = xe_expect_f32(c, vtype, "gemma4.attention.layer_norm_rms_epsilon");
            if (fabsf(v - XE_RMS_EPS) >= 1e-9f)
                xe_fatal("gemma4.attention.layer_norm_rms_epsilon: expected ~%g, found %g", (double)XE_RMS_EPS, (double)v);
            seen |= 1u << K_RMS_EPS;
            continue;
        }

        if (XE_KEY("gemma4.attention.head_count_kv")) {
            xe_expect_array_header(c, vtype, XE_T_I32, XE_LAYERS, "gemma4.attention.head_count_kv");
            for (int j = 0; j < XE_LAYERS; j++) {
                int32_t v = (int32_t)xe_u32(c);
                int32_t want = XE_IS_GLOBAL(j) ? XE_GLOBAL_KV_HEADS : XE_SWA_KV_HEADS;
                if (v != want) xe_fatal("gemma4.attention.head_count_kv[%d]: expected %d, found %d", j, want, v);
            }
            seen |= 1u << K_HEAD_COUNT_KV;
            continue;
        }

        if (XE_KEY("gemma4.attention.sliding_window_pattern")) {
            xe_expect_array_header(c, vtype, XE_T_BOOL, XE_LAYERS, "gemma4.attention.sliding_window_pattern");
            for (int j = 0; j < XE_LAYERS; j++) {
                uint8_t v = xe_u8(c);
                int want = !XE_IS_GLOBAL(j);
                if ((v != 0) != want) xe_fatal("gemma4.attention.sliding_window_pattern[%d]: expected %d, found %u", j, want, v);
            }
            seen |= 1u << K_SWA_PATTERN;
            continue;
        }

        if (XE_KEY("general.alignment")) {
            alignment = xe_expect_u32(c, vtype, "general.alignment");
            if (alignment == 0 || (alignment & (alignment - 1)))
                xe_fatal("general.alignment: bad value %llu", (unsigned long long)alignment);
            continue;
        }

        if (XE_KEY("general.name")) {
            xe_str v = xe_expect_string(c, vtype, "general.name");
            e->name = malloc(v.len + 1);
            if (!e->name) xe_fatal("out of memory copying general.name");
            memcpy(e->name, v.p, v.len);
            e->name[v.len] = 0;
            continue;
        }

        if (XE_KEY("tokenizer.ggml.tokens")) {
            xe_capture_array(c, vtype, "tokenizer.ggml.tokens", &e->tok_tokens, &e->tok_tokens_count);
            continue;
        }
        if (XE_KEY("tokenizer.ggml.scores")) {
            xe_capture_array(c, vtype, "tokenizer.ggml.scores", &e->tok_scores, &e->tok_scores_count);
            continue;
        }
        if (XE_KEY("tokenizer.ggml.token_type")) {
            xe_capture_array(c, vtype, "tokenizer.ggml.token_type", &e->tok_token_type, &e->tok_token_type_count);
            continue;
        }
        if (XE_KEY("tokenizer.ggml.merges")) {
            xe_capture_array(c, vtype, "tokenizer.ggml.merges", &e->tok_merges, &e->tok_merges_count);
            continue;
        }

        xe_skip_value(c, vtype);
    }

    for (int i = 0; i < K_REQUIRED_COUNT; i++)
        if (!(seen & (1u << i))) xe_fatal("%s: missing required key %s", gguf_path, xe_req_keys[i]);

    e->alignment = alignment;
}

static uint64_t xe_fnv1a(const char *p, uint32_t n) {
    uint64_t h = 14695981039346656037ull;
    for (uint32_t i = 0; i < n; i++) {
        h ^= (uint8_t)p[i];
        h *= 1099511628211ull;
    }
    return h;
}

static uint64_t xe_mix64(uint64_t x) {
    x += 0x9e3779b97f4a7c15ull;
    x = (x ^ (x >> 30)) * 0xbf58476d1ce4e5b9ull;
    x = (x ^ (x >> 27)) * 0x94d049bb133111ebull;
    return x ^ (x >> 31);
}

static int32_t xe_vocab_find(const xe_engine *e, const char *p, uint32_t n) {
    uint32_t i = (uint32_t)xe_fnv1a(p, n) & (XE_TOK_HASH - 1);
    for (;;) {
        int32_t id = e->tok_hash[i];
        if (id < 0) return -1;
        const xe_str *s = &e->tok_piece[id];
        if (s->len == n && memcmp(s->p, p, n) == 0) return id;
        i = (i + 1) & (XE_TOK_HASH - 1);
    }
}

static int xe_merge_find(const xe_engine *e, int32_t l, int32_t r, uint32_t *rank, int32_t *res) {
    uint64_t key = ((uint64_t)(uint32_t)l << 32) | (uint32_t)r;
    uint32_t i = (uint32_t)xe_mix64(key) & (XE_MERGE_HASH - 1);
    for (;;) {
        const xe_merge *m = &e->tok_merge[i];
        if (m->res < 0) return 0;
        if (m->key == key) {
            *rank = m->rank;
            *res = m->res;
            return 1;
        }
        i = (i + 1) & (XE_MERGE_HASH - 1);
    }
}

static xe_cur xe_cur_at(const xe_engine *e, const void *p) {
    xe_cur c = { (const uint8_t *)e->map, e->map_len, (size_t)((const uint8_t *)p - (const uint8_t *)e->map) };
    return c;
}

static void xe_tok_build(xe_engine *e, const char *gguf_path) {
    if (!e->tok_tokens) xe_fatal("%s: missing required key tokenizer.ggml.tokens", gguf_path);
    if (!e->tok_scores) xe_fatal("%s: missing required key tokenizer.ggml.scores", gguf_path);
    if (!e->tok_token_type) xe_fatal("%s: missing required key tokenizer.ggml.token_type", gguf_path);
    if (!e->tok_merges) xe_fatal("%s: missing required key tokenizer.ggml.merges", gguf_path);

    if (e->tok_tokens_count != XE_VOCAB)
        xe_fatal("tokenizer.ggml.tokens: expected %d entries, found %llu", XE_VOCAB, (unsigned long long)e->tok_tokens_count);
    if (e->tok_scores_count != XE_VOCAB)
        xe_fatal("tokenizer.ggml.scores: expected %d entries, found %llu", XE_VOCAB, (unsigned long long)e->tok_scores_count);
    if (e->tok_token_type_count != XE_VOCAB)
        xe_fatal("tokenizer.ggml.token_type: expected %d entries, found %llu", XE_VOCAB, (unsigned long long)e->tok_token_type_count);
    if (e->tok_merges_count != XE_MERGES)
        xe_fatal("tokenizer.ggml.merges: expected %d entries, found %llu", XE_MERGES, (unsigned long long)e->tok_merges_count);

    struct timespec t0, t1;
    clock_gettime(CLOCK_MONOTONIC, &t0);

    e->tok_piece = xe_alloc(e, XE_VOCAB * sizeof *e->tok_piece, XE_MEM_HOST);
    e->tok_hash = xe_alloc(e, XE_TOK_HASH * sizeof *e->tok_hash, XE_MEM_HOST);
    e->tok_merge = xe_alloc(e, XE_MERGE_HASH * sizeof *e->tok_merge, XE_MEM_HOST);

    for (uint32_t i = 0; i < XE_TOK_HASH; i++) e->tok_hash[i] = -1;
    for (uint32_t i = 0; i < XE_MERGE_HASH; i++) e->tok_merge[i].res = -1;

    xe_cur c = xe_cur_at(e, e->tok_tokens);
    uint32_t max_piece = 0;
    for (int32_t id = 0; id < XE_VOCAB; id++) {
        xe_str s = xe_read_str(&c);
        if (s.len > 0xffffffffull) xe_fatal("tokenizer.ggml.tokens: piece %d too long", id);
        e->tok_piece[id] = s;
        if ((uint32_t)s.len > max_piece) max_piece = (uint32_t)s.len;
    }
    e->tok_max_piece = max_piece;

    for (int32_t id = 0; id < XE_VOCAB; id++) {
        const xe_str *s = &e->tok_piece[id];
        uint32_t i = (uint32_t)xe_fnv1a(s->p, (uint32_t)s->len) & (XE_TOK_HASH - 1);
        for (;;) {
            int32_t cur = e->tok_hash[i];
            if (cur < 0) break;
            const xe_str *o = &e->tok_piece[cur];
            if (o->len == s->len && memcmp(o->p, s->p, s->len) == 0)
                xe_fatal("%s: tokenizer.ggml.tokens: ids %d and %d share the piece \"%.*s\"",
                         gguf_path, cur, id, (int)s->len, s->p);
            i = (i + 1) & (XE_TOK_HASH - 1);
        }
        e->tok_hash[i] = id;
    }

    static const char xe_hex[] = "0123456789ABCDEF";
    for (int b = 0; b < 256; b++) {
        char lit[6] = { '<', '0', 'x', xe_hex[b >> 4], xe_hex[b & 15], '>' };
        int32_t id = xe_vocab_find(e, lit, 6);
        if (id != XE_BYTE0_ID + b)
            xe_fatal("%s: tokenizer.ggml.tokens: expected \"%.*s\" at id %d, found %d",
                     gguf_path, 6, lit, XE_BYTE0_ID + b, id);
    }

    static const struct { const char *piece; int32_t id; } controls[] = {
        { "<|tool>", 46 },
        { "<tool|>", 47 },
        { "<|tool_call>", 48 },
        { "<tool_call|>", 49 },
        { "<|tool_response>", 50 },
        { "<tool_response|>", 51 },
        { "<|\"|>", 52 },
        { "<|think|>", 98 },
        { "<|channel>", XE_CHANNEL_BEGIN_ID },
        { "<channel|>", XE_CHANNEL_END_ID },
        { "<|turn>", XE_TURN_BEGIN_ID },
        { "<turn|>", XE_TURN_END_ID },
    };
    for (size_t i = 0; i < sizeof controls / sizeof controls[0]; i++) {
        int32_t id = xe_vocab_find(e, controls[i].piece, (uint32_t)strlen(controls[i].piece));
        if (id != controls[i].id)
            xe_fatal("%s: tokenizer.ggml.tokens: expected \"%s\" at id %d, found %d",
                     gguf_path, controls[i].piece, controls[i].id, id);
    }

    char *concat = xe_alloc(e, max_piece + 1, XE_MEM_HOST);
    xe_cur mc = xe_cur_at(e, e->tok_merges);
    for (uint32_t rank = 0; rank < XE_MERGES; rank++) {
        xe_str s = xe_read_str(&mc);
        uint64_t sp = 0;
        for (uint64_t k = 1; k < s.len; k++)
            if (s.p[k] == ' ') { sp = k; break; }
        if (sp == 0)
            xe_fatal("%s: tokenizer.ggml.merges[%u]: no separating space in \"%.*s\"",
                     gguf_path, rank, (int)s.len, s.p);

        uint32_t lnn = (uint32_t)sp, rnn = (uint32_t)(s.len - sp - 1);
        int32_t lid = xe_vocab_find(e, s.p, lnn);
        int32_t rid = xe_vocab_find(e, s.p + sp + 1, rnn);
        int32_t cid = -1;
        if (lnn + rnn <= max_piece) {
            memcpy(concat, s.p, lnn);
            memcpy(concat + lnn, s.p + sp + 1, rnn);
            cid = xe_vocab_find(e, concat, lnn + rnn);
        }
        if (lid < 0 || rid < 0 || cid < 0)
            xe_fatal("%s: tokenizer.ggml.merges[%u]: \"%.*s\" is not closed over the vocabulary (left %d right %d result %d)",
                     gguf_path, rank, (int)s.len, s.p, lid, rid, cid);

        uint64_t key = ((uint64_t)(uint32_t)lid << 32) | (uint32_t)rid;
        uint32_t i = (uint32_t)xe_mix64(key) & (XE_MERGE_HASH - 1);
        while (e->tok_merge[i].res >= 0) {
            if (e->tok_merge[i].key == key) break;
            i = (i + 1) & (XE_MERGE_HASH - 1);
        }
        if (e->tok_merge[i].res < 0)
            e->tok_merge[i] = (xe_merge){ key, rank, cid };
    }
    xe_free(e, concat, XE_MEM_HOST);

    clock_gettime(CLOCK_MONOTONIC, &t1);
    e->tok_build_seconds = (double)(t1.tv_sec - t0.tv_sec) + (double)(t1.tv_nsec - t0.tv_nsec) / 1e9;
}

static void xe_bind_tensors(xe_engine *e, xe_cur *c, const char *gguf_path) {
    static xe_bind_rec recs[XE_TENSOR_COUNT];

    for (uint64_t i = 0; i < XE_TENSOR_COUNT; i++) {
        xe_str name = xe_read_str(c);
        char namebuf[192];
        if (name.len >= sizeof namebuf) xe_fatal("%s: tensor name too long (%llu bytes)", gguf_path, (unsigned long long)name.len);
        memcpy(namebuf, name.p, name.len);
        namebuf[name.len] = 0;

        uint32_t n_dims = xe_u32(c);
        if (n_dims < 1 || n_dims > 3) xe_fatal("tensor %s: unsupported n_dims %u", namebuf, n_dims);
        uint64_t ne[3] = { 1, 1, 1 };
        for (uint32_t d = 0; d < n_dims; d++) ne[d] = xe_u64(c);
        uint32_t ttype = xe_u32(c);
        uint64_t offset = xe_u64(c);

        int kind, layer_idx = -1;
        const char *suffix;
        if (!strncmp(namebuf, "blk.", 4)) {
            char *end;
            long idx = strtol(namebuf + 4, &end, 10);
            if (end == namebuf + 4 || *end != '.' || idx < 0 || idx >= XE_LAYERS)
                xe_fatal("tensor %s: bad blk index", namebuf);
            layer_idx = (int)idx;
            suffix = end + 1;
            kind = XE_IS_GLOBAL(layer_idx) ? XE_TK_GLOBAL : XE_TK_SWA;
        } else {
            suffix = namebuf;
            kind = XE_TK_SINGLETON;
        }

        int suffix_known;
        const xe_tspec *spec = xe_find_spec(suffix, kind, &suffix_known);
        if (!spec) {
            if (suffix_known) xe_fatal("tensor %s: not valid for this layer's kind", namebuf);
            xe_fatal("tensor %s: unknown tensor name", namebuf);
        }

        if (ttype != (uint32_t)spec->qtype)
            xe_fatal("tensor %s: expected type %d, found %u", namebuf, spec->qtype, ttype);
        if (ne[0] != spec->ne0 || ne[1] != spec->ne1 || ne[2] != spec->ne2)
            xe_fatal("tensor %s: expected shape [%llu,%llu,%llu], found [%llu,%llu,%llu]",
                     namebuf,
                     (unsigned long long)spec->ne0, (unsigned long long)spec->ne1, (unsigned long long)spec->ne2,
                     (unsigned long long)ne[0], (unsigned long long)ne[1], (unsigned long long)ne[2]);

        uint64_t size;
        if (spec->qtype == XE_GGML_TYPE_Q4_0) {
            if (ne[0] % 32 != 0) xe_fatal("tensor %s: ne0 %llu not a multiple of 32", namebuf, (unsigned long long)ne[0]);
            size = (ne[0] / 32) * 18 * ne[1] * ne[2];
        } else {
            size = 4ull * ne[0] * ne[1] * ne[2];
        }

        if (offset % e->alignment != 0) xe_fatal("tensor %s: offset %llu not aligned to %llu", namebuf, (unsigned long long)offset, (unsigned long long)e->alignment);

        void *target = layer_idx >= 0 ? (void *)((char *)&e->layers[layer_idx] + spec->off) : (void *)((char *)e + spec->off);
        const void *existing;
        memcpy(&existing, target, sizeof existing);
        if (existing) xe_fatal("tensor %s: duplicate", namebuf);
        const void *claimed = (const void *)1;
        memcpy(target, &claimed, sizeof claimed);

        recs[i].target = target;
        recs[i].offset = offset;
        recs[i].size = size;

        if (spec->cat == XE_CAT_EXPERT) e->experts_bytes += size;
        else if (spec->cat == XE_CAT_EMBD) e->tok_embd_bytes += size;
        else e->dense_bytes += size;

        if (ttype == XE_GGML_TYPE_Q4_0) { e->q4_0_count++; e->q4_0_bytes += size; }
        else { e->f32_count++; e->f32_bytes += size; }
    }

    uint64_t data_off = xe_align_up((uint64_t)c->pos, e->alignment);
    if (data_off > e->map_len) xe_fatal("%s: data section start %llu beyond file size %zu", gguf_path, (unsigned long long)data_off, e->map_len);
    const uint8_t *data_base = (const uint8_t *)e->map + data_off;
    e->data = data_base;
    e->header_len = (size_t)data_off;

    uint64_t avail = e->map_len - data_off;
    for (uint64_t i = 0; i < XE_TENSOR_COUNT; i++) {
        xe_bind_rec *r = &recs[i];
        if (r->offset > avail || r->size > avail - r->offset)
            xe_fatal("%s: tensor #%llu offset %llu size %llu exceeds file bounds", gguf_path, (unsigned long long)i, (unsigned long long)r->offset, (unsigned long long)r->size);
        const void *ptr = data_base + r->offset;
        memcpy(r->target, &ptr, sizeof ptr);
    }
}

static void xe_check_complete(xe_engine *e, const char *gguf_path) {
    if (!e->tok_embd.qs) xe_fatal("%s: missing tensor token_embd.weight", gguf_path);
    if (!e->out_norm) xe_fatal("%s: missing tensor output_norm.weight", gguf_path);
    if (!e->rope_freqs) xe_fatal("%s: missing tensor rope_freqs.weight", gguf_path);

    for (int i = 0; i < XE_LAYERS; i++) {
        xe_layer *l = &e->layers[i];
        if (!l->attn_norm) xe_fatal("%s: blk.%d: missing attn_norm.weight", gguf_path, i);
        if (!l->post_attn_norm) xe_fatal("%s: blk.%d: missing post_attention_norm.weight", gguf_path, i);
        if (!l->ffn_norm) xe_fatal("%s: blk.%d: missing ffn_norm.weight", gguf_path, i);
        if (!l->post_ffw_norm) xe_fatal("%s: blk.%d: missing post_ffw_norm.weight", gguf_path, i);
        if (!l->post_ffw_norm1) xe_fatal("%s: blk.%d: missing post_ffw_norm_1.weight", gguf_path, i);
        if (!l->post_ffw_norm2) xe_fatal("%s: blk.%d: missing post_ffw_norm_2.weight", gguf_path, i);
        if (!l->pre_ffw_norm2) xe_fatal("%s: blk.%d: missing pre_ffw_norm_2.weight", gguf_path, i);
        if (!l->layer_out_scale) xe_fatal("%s: blk.%d: missing layer_output_scale.weight", gguf_path, i);
        if (!l->ffn_gate.qs) xe_fatal("%s: blk.%d: missing ffn_gate.weight", gguf_path, i);
        if (!l->ffn_up.qs) xe_fatal("%s: blk.%d: missing ffn_up.weight", gguf_path, i);
        if (!l->ffn_down.qs) xe_fatal("%s: blk.%d: missing ffn_down.weight", gguf_path, i);
        if (!l->router_w) xe_fatal("%s: blk.%d: missing ffn_gate_inp.weight", gguf_path, i);
        if (!l->router_scale) xe_fatal("%s: blk.%d: missing ffn_gate_inp.scale", gguf_path, i);
        if (!l->gate_up_exps.qs) xe_fatal("%s: blk.%d: missing ffn_gate_up_exps.weight", gguf_path, i);
        if (!l->down_exps.qs) xe_fatal("%s: blk.%d: missing ffn_down_exps.weight", gguf_path, i);
        if (!l->down_exps_scale) xe_fatal("%s: blk.%d: missing ffn_down_exps.scale", gguf_path, i);

        if (!l->attn_q.qs) xe_fatal("%s: blk.%d: missing attn_q.weight", gguf_path, i);
        if (!l->attn_k.qs) xe_fatal("%s: blk.%d: missing attn_k.weight", gguf_path, i);
        if (!l->attn_o.qs) xe_fatal("%s: blk.%d: missing attn_output.weight", gguf_path, i);
        if (!l->q_norm) xe_fatal("%s: blk.%d: missing attn_q_norm.weight", gguf_path, i);
        if (!l->k_norm) xe_fatal("%s: blk.%d: missing attn_k_norm.weight", gguf_path, i);

        if (XE_IS_GLOBAL(i)) {
            if (l->attn_v.qs) xe_fatal("%s: blk.%d: unexpected attn_v.weight on GLOBAL layer", gguf_path, i);
        } else {
            if (!l->attn_v.qs) xe_fatal("%s: blk.%d: missing attn_v.weight", gguf_path, i);
        }
    }
}

/* Q4_0 blocks are 18 bytes total, 2 of fp16 scale and 16 of weight nibbles.
 * We split scales from nibbles because the 18 byte stride defeats coalesced loads.
 * We measured approximately 3.1x on GPU and 1.6x on CPU. */
static void xe_repack(xe_engine *e, const char *gguf_path) {
    madvise(e->map, e->map_len, MADV_SEQUENTIAL);

    static xe_repack_job jobs[XE_TENSOR_COUNT];
    int njobs = 0;
    uint64_t slab_size = 0, nibble_bytes = 0, scale_bytes = 0;

    for (size_t i = 0; i < XE_NSPECS; i++) {
        const xe_tspec *s = &xe_specs[i];
        if (s->qtype != XE_GGML_TYPE_Q4_0) continue;
        int blocks = (int)(s->ne0 / 32);
        uint64_t rows = s->ne1 * s->ne2;
        uint64_t nblocks = (uint64_t)blocks * rows;
        uint64_t nb64 = xe_align_up(nblocks * 16, 64);
        uint64_t sb64 = xe_align_up(nblocks * 2, 64);

        if (s->kind == XE_TK_SINGLETON) {
            jobs[njobs++] = (xe_repack_job){
                .field = (xe_q4 *)((char *)e + s->off), .nblocks = nblocks,
                .blocks = blocks, .rows = rows, .suffix = s->suffix, .layer_idx = -1 };
            nibble_bytes += nblocks * 16;
            scale_bytes += nblocks * 2;
            slab_size += nb64 + sb64;
            continue;
        }
        for (int l = 0; l < XE_LAYERS; l++) {
            if (s->kind == XE_TK_SWA && XE_IS_GLOBAL(l)) continue;
            if (s->kind == XE_TK_GLOBAL && !XE_IS_GLOBAL(l)) continue;
            jobs[njobs++] = (xe_repack_job){
                .field = (xe_q4 *)((char *)&e->layers[l] + s->off), .nblocks = nblocks,
                .blocks = blocks, .rows = rows, .suffix = s->suffix, .layer_idx = l };
            nibble_bytes += nblocks * 16;
            scale_bytes += nblocks * 2;
            slab_size += nb64 + sb64;
        }
    }

    if ((uint64_t)njobs != e->q4_0_count)
        xe_fatal("%s: repack job count %d mismatches Q4_0 tensor count %llu", gguf_path, njobs, (unsigned long long)e->q4_0_count);

    for (int i = 0; i < njobs; i++)
        jobs[i].source_offset = (uint64_t)(jobs[i].field->qs -
                                          (const uint8_t *)e->map);
    qsort(jobs, (size_t)njobs, sizeof *jobs, xe_repack_source_order);

    int part = 0;
    uint64_t part_used = 0;
    for (int i = 0; i < njobs; i++) {
        uint64_t nb64 = xe_align_up(jobs[i].nblocks * 16, 64);
        uint64_t sb64 = xe_align_up(jobs[i].nblocks * 2, 64);
        uint64_t bytes = nb64 + sb64;
        if (bytes > XE_REPACK_PART_LIMIT)
            xe_fatal("%s: repack tensor %s exceeds shared allocation limit",
                     gguf_path, jobs[i].suffix);
        if (part_used && part_used + bytes > XE_REPACK_PART_LIMIT) {
            e->repack_part_sizes[part++] = part_used;
            part_used = 0;
        }
        if (part >= XE_REPACK_PARTS)
            xe_fatal("%s: repack needs more than %d shared allocations",
                     gguf_path, XE_REPACK_PARTS);
        jobs[i].part = part;
        jobs[i].part_offset = part_used;
        part_used += bytes;
    }
    e->repack_part_sizes[part++] = part_used;
    e->repack_part_count = part;
    e->repack_slab_size = slab_size;
    e->repack_nibble_bytes = nibble_bytes;
    e->repack_scale_bytes = scale_bytes;

    struct timespec t0, t1;
    clock_gettime(CLOCK_MONOTONIC, &t0);
    xe_copy_f32(e);

    uint64_t released = xe_align_up(e->header_len, 4096);
    uint64_t verified = 0;
    int first = 0;
    for (int i = 0; i < e->repack_part_count; i++) {
        e->repack_parts[i] = xe_alloc(e, e->repack_part_sizes[i], XE_MEM_SHARED);
#if defined(XE_REPACK_HUGEPAGE) || defined(XE_REPACK_NOHUGEPAGE)
        size_t page = 4096;
        int huge_advice =
#ifdef XE_REPACK_HUGEPAGE
            MADV_HUGEPAGE;
#else
            MADV_NOHUGEPAGE;
#endif
        uintptr_t huge_begin = ((uintptr_t)e->repack_parts[i] + page - 1) &
                               ~(uintptr_t)(page - 1);
        uintptr_t huge_end = ((uintptr_t)e->repack_parts[i] +
                              e->repack_part_sizes[i]) & ~(uintptr_t)(page - 1);
        if (huge_end > huge_begin &&
            madvise((void *)huge_begin, huge_end - huge_begin, huge_advice) != 0)
            xe_fatal("%s: madvise repack slab: %s", gguf_path, strerror(errno));
#endif
        int end = first;
        while (end < njobs && jobs[end].part == i) {
            uint64_t nb64 = xe_align_up(jobs[end].nblocks * 16, 64);
            uint8_t *cursor = (uint8_t *)e->repack_parts[i] +
                              jobs[end].part_offset;
            jobs[end].qs_seg = cursor;
            jobs[end].d_seg = (uint16_t *)(cursor + nb64);
            end++;
        }

        atomic_int next_job;
        atomic_init(&next_job, 0);
        pthread_t threads[XE_REPACK_THREADS];
        xe_repack_worker_arg args[XE_REPACK_THREADS];
        for (int t = 0; t < XE_REPACK_THREADS; t++) {
            args[t] = (xe_repack_worker_arg){ jobs + first, end - first,
                                             &next_job, gguf_path, 0 };
            if (pthread_create(&threads[t], NULL, xe_repack_worker, &args[t]) != 0)
                xe_fatal("%s: pthread_create failed for repack worker %d", gguf_path, t);
        }
        for (int t = 0; t < XE_REPACK_THREADS; t++) {
            pthread_join(threads[t], NULL);
            verified += args[t].verified;
        }
        uint64_t consumed = end < njobs ? jobs[end].source_offset : e->map_len;
        xe_repack_release(e, &released, consumed, gguf_path);
        first = end;
    }
    close(e->map_fd);
    e->map_fd = -1;

    clock_gettime(CLOCK_MONOTONIC, &t1);
    e->repack_seconds = (double)(t1.tv_sec - t0.tv_sec) + (double)(t1.tv_nsec - t0.tv_nsec) / 1e9;
    e->repack_verified_blocks = verified;

#ifdef XE_REPACK_DROP_SOURCE
    if (madvise(e->map, e->map_len, MADV_DONTNEED) != 0)
        xe_fatal("%s: madvise source map: %s", gguf_path, strerror(errno));
#else
    madvise(e->map, e->map_len, MADV_NORMAL);
#endif
}

static xe_engine *xe_open_common(const char *gguf_path, int vocab_only, xe_cur *c) {
    xe_engine *e = xe_map_file(gguf_path);
    e->vocab_only = vocab_only;
    e->context = XE_CONTEXT_DEFAULT;
    e->scalar_rms = 1;
    *c = xe_parse_header(e, gguf_path);
    xe_parse_metadata(e, c, gguf_path);
    xe_tok_build(e, gguf_path);
    if (vocab_only) {
        close(e->map_fd);
        e->map_fd = -1;
    }
    return e;
}

static void xe_snapshot_fingerprints_init(xe_engine *e);

xe_engine *xe_engine_open_with_context(const char *gguf_path, int context) {
    if (context < XE_CONTEXT_MIN || context > XE_CONTEXT_MAX)
        xe_fatal("context: capacity %d out of range [%d, %d]",
                 context, XE_CONTEXT_MIN, XE_CONTEXT_MAX);
    xe_cur c;
    xe_engine *e = xe_open_common(gguf_path, 0, &c);
    e->context = context;
    xe_bind_tensors(e, &c, gguf_path);
    xe_check_complete(e, gguf_path);
    xe_constants_init(e);
    xe_gpu_init(e);
    xe_repack(e, gguf_path);
    xe_snapshot_fingerprints_init(e);
    return e;
}

xe_engine *xe_engine_open(const char *gguf_path) {
    return xe_engine_open_with_context(gguf_path, XE_CONTEXT_DEFAULT);
}

xe_engine *xe_engine_open_vocab(const char *gguf_path) {
    xe_cur c;
    return xe_open_common(gguf_path, 1, &c);
}

void xe_engine_close(xe_engine *e) {
    if (!e) return;
    xe_worker_pool_destroy(e);
    xe_free(e, e->gelu_lut, XE_MEM_HOST);
    xe_free(e, e->f32_slab, XE_MEM_SHARED);
    for (int i = 0; i < e->repack_part_count; i++)
        xe_free(e, e->repack_parts[i], XE_MEM_SHARED);
    xe_gpu_destroy(e);
    xe_free(e, e->tok_merge, XE_MEM_HOST);
    xe_free(e, e->tok_hash, XE_MEM_HOST);
    xe_free(e, e->tok_piece, XE_MEM_HOST);
    if (e->map && e->map_fd >= 0) close(e->map_fd);
    munmap(e->map, e->map_len);
    free(e->name);
    free(e);
}

void xe_engine_info(const xe_engine *e, FILE *out) {
    uint64_t total_bytes = e->q4_0_bytes + e->f32_bytes;
    double q4_0_gib = e->q4_0_bytes / 1073741824.0;
    double f32_mib = e->f32_bytes / 1048576.0;
    double q4_0_pct = total_bytes ? 100.0 * (double)e->q4_0_bytes / (double)total_bytes : 0.0;
    double f32_pct = total_bytes ? 100.0 * (double)e->f32_bytes / (double)total_bytes : 0.0;

    fprintf(out, "name             %s\n", e->name ? e->name : "(unset)");
    fprintf(out, "architecture     gemma4\n");
    fprintf(out, "gguf version     %u\n", e->gguf_version);
    fprintf(out, "file size        %.2f GiB\n", e->map_len / 1073741824.0);
    fprintf(out, "tensors          %llu\n", (unsigned long long)(e->q4_0_count + e->f32_count));
    fprintf(out, "  Q4_0   %6llu tensors  %9.2f GiB  %5.1f%%\n", (unsigned long long)e->q4_0_count, q4_0_gib, q4_0_pct);
    fprintf(out, "  F32    %6llu tensors  %9.2f MiB  %5.1f%%\n", (unsigned long long)e->f32_count, f32_mib, f32_pct);
    fprintf(out, "\n");

    double nib_gib = e->repack_nibble_bytes / 1073741824.0;
    double scale_gib = e->repack_scale_bytes / 1073741824.0;
    double slab_gib = e->repack_slab_size / 1073741824.0;
    fprintf(out, "repack           nibbles %6.2f GiB   scales %5.2f GiB   slab %6.2f GiB\n", nib_gib, scale_gib, slab_gib);
    fprintf(out, "repack time      %.2f s   (sampled verify: %llu blocks ok)\n", e->repack_seconds, (unsigned long long)e->repack_verified_blocks);
    fprintf(out, "tokenizer        %d pieces  %d merges  max piece %u bytes\n", XE_VOCAB, XE_MERGES, e->tok_max_piece);
    fprintf(out, "tokenizer build  %.3f s\n", e->tok_build_seconds);
    fprintf(out, "\n");

    fprintf(out, "layers           %d\n", XE_LAYERS);
    fprintf(out, "context capacity %d\n", e->context);
    fprintf(out, "embedding dim    %d\n", XE_EMBD);
    fprintf(out, "vocab            %d\n", XE_VOCAB);
    fprintf(out, "query heads      %d\n", XE_Q_HEADS);
    fprintf(out, "kv heads (swa)   %d   head dim %d   window %d\n", XE_SWA_KV_HEADS, XE_SWA_HEAD_DIM, XE_SWA_WINDOW);
    fprintf(out, "kv heads (glob)  %d   head dim %d\n", XE_GLOBAL_KV_HEADS, XE_GLOBAL_HEAD_DIM);
    fprintf(out, "experts          %d used of %d, ffn %d\n", XE_EXPERTS_USED, XE_EXPERTS, XE_EXPERT_FFN);
    fprintf(out, "dense ffn        %d\n", XE_DENSE_FFN);
    fprintf(out, "\n");

    double active_experts_bytes = (double)e->experts_bytes * XE_EXPERTS_USED / XE_EXPERTS;
    double dense_gb = (double)e->dense_bytes / 1e9;
    double active_gb = active_experts_bytes / 1e9;
    double head_gb = (double)e->tok_embd_bytes / 1e9;
    double total_gb = dense_gb + active_gb + head_gb;

    fprintf(out, "decode read budget per token:\n");
    fprintf(out, "  dense weights    %7.3f GB   (attn + dense ffn + router + norms)\n", dense_gb);
    fprintf(out, "  active experts   %7.3f GB   (%d of %d experts, %d layers)\n", active_gb, XE_EXPERTS_USED, XE_EXPERTS, XE_LAYERS);
    fprintf(out, "  output head      %7.3f GB   (tied embeddings)\n", head_gb);
    fprintf(out, "  total            %7.3f GB/token\n", total_gb);
    fprintf(out, "\n");

    fprintf(out, "projected throughput:\n");
    fprintf(out, "  at 51.2 GB/s (theoretical)  %6.1f tok/s\n", total_gb > 0 ? 51.2 / total_gb : 0.0);
    fprintf(out, "  at 44.0 GB/s (realistic)    %6.1f tok/s\n", total_gb > 0 ? 44.0 / total_gb : 0.0);
}

static uint32_t xe_utf8_len(uint8_t c) {
    static const uint8_t lookup[16] = { 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 3, 4 };
    return lookup[c >> 4];
}

typedef struct {
    int32_t id;
    int32_t prev, next;
    uint32_t off, n;
} xe_sym;

typedef struct {
    uint32_t rank;
    uint32_t lpos, rpos;
    int32_t lid, rid, res;
} xe_big;

static int xe_big_less(const xe_big *a, const xe_big *b) {
    return a->rank < b->rank || (a->rank == b->rank && a->lpos < b->lpos);
}

static void xe_heap_push(xe_big *h, int *n, xe_big v) {
    int i = (*n)++;
    h[i] = v;
    while (i > 0) {
        int p = (i - 1) / 2;
        if (!xe_big_less(&h[i], &h[p])) break;
        xe_big t = h[p]; h[p] = h[i]; h[i] = t;
        i = p;
    }
}

static xe_big xe_heap_pop(xe_big *h, int *n) {
    xe_big top = h[0];
    h[0] = h[--(*n)];
    for (int i = 0;;) {
        int l = 2 * i + 1, r = l + 1, m = i;
        if (l < *n && xe_big_less(&h[l], &h[m])) m = l;
        if (r < *n && xe_big_less(&h[r], &h[m])) m = r;
        if (m == i) break;
        xe_big t = h[m]; h[m] = h[i]; h[i] = t;
        i = m;
    }
    return top;
}

static void xe_try_bigram(const xe_engine *e, xe_sym *sym, int l, int r,
                          xe_big *heap, int *hn) {
    if (l < 0 || r < 0) return;
    if (sym[l].id < 0 || sym[r].id < 0) return;
    uint32_t rank; int32_t res;
    if (!xe_merge_find(e, sym[l].id, sym[r].id, &rank, &res)) return;
    xe_heap_push(heap, hn, (xe_big){ rank, (uint32_t)l, (uint32_t)r,
                                     sym[l].id, sym[r].id, res });
}

static int xe_bpe_chunk(const xe_engine *e, const char *s, uint32_t len,
                        int32_t *out) {
    if (len == 0) return 0;

    xe_sym *sym = xe_alloc(e, len * sizeof *sym, XE_MEM_HOST);
    xe_big *heap = xe_alloc(e, 3 * len * sizeof *heap, XE_MEM_HOST);
    int nsym = 0, hn = 0;

    for (uint32_t off = 0; off < len; ) {
        uint32_t n = xe_utf8_len((uint8_t)s[off]);
        sym[nsym] = (xe_sym){ xe_vocab_find(e, s + off, n),
                              nsym - 1, nsym + 1, off, n };
        nsym++; off += n;
    }
    sym[nsym - 1].next = -1;

    for (int i = 0; i + 1 < nsym; i++)
        xe_try_bigram(e, sym, i, i + 1, heap, &hn);

    while (hn > 0) {
        xe_big b = xe_heap_pop(heap, &hn);
        xe_sym *l = &sym[b.lpos], *r = &sym[b.rpos];
        if (l->n == 0 || r->n == 0) continue;
        if (l->id != b.lid || r->id != b.rid) continue;
        l->id = b.res;
        l->n += r->n;
        l->next = r->next;
        if (r->next >= 0) sym[r->next].prev = (int32_t)b.lpos;
        r->n = 0;
        xe_try_bigram(e, sym, l->prev, (int)b.lpos, heap, &hn);
        xe_try_bigram(e, sym, (int)b.lpos, l->next, heap, &hn);
    }

    int nout = 0;
    for (int i = 0; i >= 0; i = sym[i].next) {
        if (sym[i].id >= 0) { out[nout++] = sym[i].id; continue; }
        for (uint32_t k = 0; k < sym[i].n; k++)
            out[nout++] = XE_BYTE0_ID + (uint8_t)s[sym[i].off + k];
    }

    xe_free(e, heap, XE_MEM_HOST);
    xe_free(e, sym, XE_MEM_HOST);
    return nout;
}

static uint32_t xe_cpt_to_utf8(uint32_t cpt, char *dst) {
    if (cpt <= 0x7f) {
        dst[0] = (char)cpt;
        return 1;
    }
    if (cpt <= 0x7ff) {
        dst[0] = (char)(0xc0 | ((cpt >> 6) & 0x1f));
        dst[1] = (char)(0x80 | (cpt & 0x3f));
        return 2;
    }
    if (cpt <= 0xffff) {
        dst[0] = (char)(0xe0 | ((cpt >> 12) & 0x0f));
        dst[1] = (char)(0x80 | ((cpt >> 6) & 0x3f));
        dst[2] = (char)(0x80 | (cpt & 0x3f));
        return 3;
    }
    dst[0] = (char)(0xf0 | ((cpt >> 18) & 0x07));
    dst[1] = (char)(0x80 | ((cpt >> 12) & 0x3f));
    dst[2] = (char)(0x80 | ((cpt >> 6) & 0x3f));
    dst[3] = (char)(0x80 | (cpt & 0x3f));
    return 4;
}

static uint32_t xe_escape(const char *text, size_t n, char *dst) {
    const uint8_t *s = (const uint8_t *)text;
    uint32_t o = 0;
    size_t i = 0;
    while (i < n) {
        uint8_t c = s[i];
        if (!(c & 0x80)) {
            if (c == ' ') {
                dst[o++] = (char)0xe2; dst[o++] = (char)0x96; dst[o++] = (char)0x81;
            } else {
                dst[o++] = (char)c;
            }
            i++;
            continue;
        }
        uint32_t need = 0;
        if (!(c & 0x40)) need = 0;
        else if (!(c & 0x20)) need = 2;
        else if (!(c & 0x10)) need = 3;
        else if (!(c & 0x08)) need = 4;
        int ok = need != 0 && i + need <= n;
        for (uint32_t k = 1; ok && k < need; k++)
            if ((s[i + k] & 0xc0) != 0x80) ok = 0;
        if (!ok) {
            dst[o++] = (char)0xef; dst[o++] = (char)0xbf; dst[o++] = (char)0xbd;
            i++;
            continue;
        }
        uint32_t cpt = c & (0x7fu >> need);
        for (uint32_t k = 1; k < need; k++) cpt = (cpt << 6) | (s[i + k] & 0x3f);
        i += need;
        o += xe_cpt_to_utf8(cpt, dst + o);
    }
    return o;
}

int xe_encode_text_bounded(const xe_engine *e, const char *text, int32_t *out, int cap) {
    if (!e || !text || !out) xe_fatal("encode_text: missing input");
    if (cap < 0) xe_fatal("encode_text: negative output capacity %d", cap);
    size_t n = strlen(text);
    if (n == 0) return 0;
    if (n > UINT32_MAX / 3u) xe_fatal("encode_text: input too large");

    char *buf = xe_alloc(e, 3 * n, XE_MEM_HOST);
    int32_t *tmp = xe_alloc(e, 3 * n * sizeof(*tmp), XE_MEM_HOST);
    uint32_t m = xe_escape(text, n, buf);
    int nout = 0;

    for (uint32_t i = 0; i < m; ) {
        uint32_t j = i;
        if (buf[i] == '\n') {
            while (j < m && buf[j] == '\n') j++;
            int32_t id = xe_vocab_find(e, buf + i, j - i);
            if (id >= 0) tmp[nout++] = id;
            else nout += xe_bpe_chunk(e, buf + i, j - i, tmp + nout);
        } else {
            while (j < m && buf[j] != '\n') j++;
            nout += xe_bpe_chunk(e, buf + i, j - i, tmp + nout);
        }
        i = j;
    }

    xe_free(e, buf, XE_MEM_HOST);
    if (nout <= cap) memcpy(out, tmp, (size_t)nout * sizeof(*out));
    xe_free(e, tmp, XE_MEM_HOST);
    return nout;
}

int xe_encode_text(const xe_engine *e, const char *text, int32_t *out, int cap) {
    int n = xe_encode_text_bounded(e, text, out, cap);
    if (n > cap) xe_fatal("encode_text: output capacity %d too small (need %d)", cap, n);
    return n;
}

static int32_t xe_token_type(const xe_engine *e, int32_t tok) {
    int32_t t;
    memcpy(&t, (const uint8_t *)e->tok_token_type + 4 * (size_t)tok, 4);
    return t;
}

int xe_detokenize(const xe_engine *e, int32_t tok, char *buf, int cap) {
    if (tok < 0 || tok >= XE_VOCAB)
        xe_fatal("detokenize: token id %d out of range [0, %d]", tok, XE_VOCAB - 1);

    int32_t type = xe_token_type(e, tok);
    if (type == XE_TT_BYTE) {
        if (cap < 1) xe_fatal("detokenize: output capacity %d too small for token %d", cap, tok);
        buf[0] = (char)(uint8_t)(tok - XE_BYTE0_ID);
        return 1;
    }
    if (type != XE_TT_NORMAL) return 0;

    const xe_str *p = &e->tok_piece[tok];
    int nout = 0;
    for (uint64_t i = 0; i < p->len; ) {
        if (nout >= cap) xe_fatal("detokenize: output capacity %d too small for token %d", cap, tok);
        if (p->len - i >= 3 && (uint8_t)p->p[i] == 0xe2 && (uint8_t)p->p[i + 1] == 0x96 && (uint8_t)p->p[i + 2] == 0x81) {
            buf[nout++] = ' ';
            i += 3;
        } else {
            buf[nout++] = p->p[i++];
        }
    }
    return nout;
}

int32_t xe_token_id(const xe_engine *e, const char *piece) {
    return xe_vocab_find(e, piece, (uint32_t)strlen(piece));
}

const char *xe_token_piece(const xe_engine *e, int32_t tok, int *len) {
    if (tok < 0 || tok >= XE_VOCAB)
        xe_fatal("token_piece: token id %d out of range [0, %d]", tok, XE_VOCAB - 1);
    *len = (int)e->tok_piece[tok].len;
    return e->tok_piece[tok].p;
}

int32_t xe_bos_id(const xe_engine *e) {
    return e->bos_id;
}

int32_t xe_eos_id(const xe_engine *e) {
    return e->eos_id;
}

int32_t xe_eot_id(const xe_engine *e) {
    return e->eot_id;
}

int xe_context_size(const xe_engine *e) {
    if (!e) xe_fatal("context_size: missing engine");
    return e->context;
}

int xe_vocab_size(const xe_engine *e) {
    if (!e) xe_fatal("vocab_size: missing engine");
    return (int)e->tok_tokens_count;
}

uint64_t xe_engine_model_size(const xe_engine *e) {
    if (!e) xe_fatal("model_size: missing engine");
    return e->q4_0_bytes + e->f32_bytes;
}

int xe_engine_worker_count(const xe_engine *e) {
    if (!e) xe_fatal("worker_count: missing engine");
    return XE_WORKERS;
}

int xe_engine_worker_cpu(const xe_engine *e, int worker) {
    if (!e) xe_fatal("worker_cpu: missing engine");
    if (worker < 0 || worker >= XE_WORKERS)
        xe_fatal("worker_cpu: worker %d out of range [0, %d]", worker, XE_WORKERS - 1);
    return e->worker_pinned ? xe_worker_cpu(worker) : -1;
}

static int xe_tokens_valid(const xe_tokens *tokens) {
    return tokens && tokens->len >= 0 && tokens->cap >= 0 && tokens->cap <= XE_MODEL_CTX &&
        tokens->len <= tokens->cap && ((tokens->v != NULL) == (tokens->cap != 0));
}

static void xe_tokens_reserve(xe_tokens *tokens, int need) {
    if (!xe_tokens_valid(tokens)) xe_fatal("tokens: invalid vector");
    if (need < tokens->len || need > XE_MODEL_CTX)
        xe_fatal("tokens: length %d out of range [0, %d]", need, XE_MODEL_CTX);
    if (need <= tokens->cap) return;

    int cap = tokens->cap ? tokens->cap : 64;
    while (cap < need) {
        if (cap > XE_MODEL_CTX / 2) {
            cap = XE_MODEL_CTX;
            break;
        }
        cap *= 2;
    }
    int32_t *v = xe_alloc(NULL, (size_t)cap * sizeof(*v), XE_MEM_HOST);
    if (tokens->len)
        memcpy(v, tokens->v, (size_t)tokens->len * sizeof(*v));
    xe_free(NULL, tokens->v, XE_MEM_HOST);
    tokens->v = v;
    tokens->cap = cap;
}

void xe_tokens_push(xe_tokens *tokens, int32_t token) {
    if (!xe_tokens_valid(tokens)) xe_fatal("tokens_push: invalid vector");
    if (tokens->len == XE_MODEL_CTX) xe_fatal("tokens_push: context is full");
    xe_tokens_reserve(tokens, tokens->len + 1);
    tokens->v[tokens->len++] = token;
}

void xe_tokens_free(xe_tokens *tokens) {
    if (!tokens) return;
    xe_free(NULL, tokens->v, XE_MEM_HOST);
    memset(tokens, 0, sizeof(*tokens));
}

const char *xe_chat_template(const xe_engine *e, uint64_t *length) {
    if (!e) xe_fatal("chat_template: missing engine");
    if (length) *length = e->chat_template.len;
    return e->chat_template.p;
}

static inline __m256 xe_load_f16x8(const _Float16 *p) {
    return _mm256_cvtph_ps(_mm_loadu_si128((const __m128i *)p));
}

static inline float xe_sum_f32x8(__m256 x) {
    __m128 lo = _mm256_castps256_ps128(x);
    __m128 hi = _mm256_extractf128_ps(x, 1);
    __m128 sum = _mm_add_ps(lo, hi);
    sum = _mm_hadd_ps(sum, sum);
    sum = _mm_hadd_ps(sum, sum);
    return _mm_cvtss_f32(sum);
}

static _Float16 *xe_kv_layer_ptr(xe_session *s, int layer, int value) {
    if (layer < 0 || layer >= XE_LAYERS) xe_fatal("kv: layer %d out of range", layer);
    if (XE_IS_GLOBAL(layer)) {
        if (s->cow) {
            _Float16 *base = value ? s->cow_global_v[layer / 6]
                                   : s->cow_global_k[layer / 6];
            if (!base) xe_fatal("kv: unallocated COW layer %d", layer);
            return base;
        }
        _Float16 *base = value ? s->global_v : s->global_k;
        return base + (size_t)(layer / 6) * xe_global_layer_elems(s->engine);
    }
    _Float16 *base = value ? s->swa_v : s->swa_k;
    return base + (size_t)(layer - layer / 6) * XE_SWA_LAYER_ELEMS;
}

void xe_kv_append_f16(_Float16 *k_cache, _Float16 *v_cache, int n_kv_heads,
                      int head_dim, int capacity, int slot, const float *k, const float *v) {
    if (slot < 0 || slot >= capacity) xe_fatal("kv: slot %d out of range [0, %d]", slot, capacity - 1);
    for (int h = 0; h < n_kv_heads; h++) {
        _Float16 *kd = k_cache + ((size_t)h * capacity + slot) * head_dim;
        _Float16 *vd = v_cache + ((size_t)h * capacity + slot) * head_dim;
        const float *ks = k + (size_t)h * head_dim;
        const float *vs = v + (size_t)h * head_dim;
        for (int d = 0; d < head_dim; d += 8) {
            __m128i kh = _mm256_cvtps_ph(_mm256_loadu_ps(ks + d), _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC);
            __m128i vh = _mm256_cvtps_ph(_mm256_loadu_ps(vs + d), _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC);
            _mm_storeu_si128((__m128i *)(kd + d), kh);
            _mm_storeu_si128((__m128i *)(vd + d), vh);
        }
    }
}

void xe_session_kv_append(xe_session *s, int layer, int pos, const float *k, const float *v) {
    if (pos < 0 || pos >= s->engine->context) xe_fatal("kv: position %d out of range [0, %d]", pos, s->engine->context - 1);
    int global = XE_IS_GLOBAL(layer);
    int n_kv_heads = global ? XE_GLOBAL_KV_HEADS : XE_SWA_KV_HEADS;
    int head_dim = global ? XE_GLOBAL_HEAD_DIM : XE_SWA_HEAD_DIM;
    int capacity = global ? s->engine->context : XE_SWA_WINDOW;
    int slot = global ? pos : pos & (XE_SWA_WINDOW - 1);
    xe_kv_append_f16(xe_kv_layer_ptr(s, layer, 0), xe_kv_layer_ptr(s, layer, 1),
                     n_kv_heads, head_dim, capacity, slot, k, v);
}

static void xe_kv_qk_g2(float *scores, const float *q, const _Float16 *k_cache,
                        int head_dim, int capacity, int first, int n_keys, int ring, int kv_head,
                        int key_begin, int key_end) {
    const float *q0 = q + (size_t)(kv_head * 2) * head_dim;
    const float *q1 = q0 + head_dim;
    for (int i = key_begin; i < key_end; i++) {
        int p = ring ? (first + i) & (capacity - 1) : first + i;
        const _Float16 *kp = k_cache + ((size_t)kv_head * capacity + p) * head_dim;
        __m256 a0 = _mm256_setzero_ps();
        __m256 a1 = _mm256_setzero_ps();
        for (int d = 0; d < head_dim; d += 8) {
            __m256 x = xe_load_f16x8(kp + d);
            a0 = _mm256_fmadd_ps(x, _mm256_loadu_ps(q0 + d), a0);
            a1 = _mm256_fmadd_ps(x, _mm256_loadu_ps(q1 + d), a1);
        }
        scores[(size_t)(kv_head * 2) * n_keys + i] = xe_sum_f32x8(a0);
        scores[(size_t)(kv_head * 2 + 1) * n_keys + i] = xe_sum_f32x8(a1);
    }
}

static void xe_kv_qk_g8(float *scores, const float *q, const _Float16 *k_cache,
                        int head_dim, int capacity, int first, int n_keys, int ring, int kv_head,
                        int key_begin, int key_end) {
    const float *q0 = q + (size_t)(kv_head * 8) * head_dim;
    const float *q1 = q0 + head_dim;
    const float *q2 = q1 + head_dim;
    const float *q3 = q2 + head_dim;
    const float *q4 = q3 + head_dim;
    const float *q5 = q4 + head_dim;
    const float *q6 = q5 + head_dim;
    const float *q7 = q6 + head_dim;
    for (int i = key_begin; i < key_end; i++) {
        int p = ring ? (first + i) & (capacity - 1) : first + i;
        const _Float16 *kp = k_cache + ((size_t)kv_head * capacity + p) * head_dim;
        __m256 a0 = _mm256_setzero_ps();
        __m256 a1 = _mm256_setzero_ps();
        __m256 a2 = _mm256_setzero_ps();
        __m256 a3 = _mm256_setzero_ps();
        __m256 a4 = _mm256_setzero_ps();
        __m256 a5 = _mm256_setzero_ps();
        __m256 a6 = _mm256_setzero_ps();
        __m256 a7 = _mm256_setzero_ps();
        for (int d = 0; d < head_dim; d += 8) {
            __m256 x = xe_load_f16x8(kp + d);
            a0 = _mm256_fmadd_ps(x, _mm256_loadu_ps(q0 + d), a0);
            a1 = _mm256_fmadd_ps(x, _mm256_loadu_ps(q1 + d), a1);
            a2 = _mm256_fmadd_ps(x, _mm256_loadu_ps(q2 + d), a2);
            a3 = _mm256_fmadd_ps(x, _mm256_loadu_ps(q3 + d), a3);
            a4 = _mm256_fmadd_ps(x, _mm256_loadu_ps(q4 + d), a4);
            a5 = _mm256_fmadd_ps(x, _mm256_loadu_ps(q5 + d), a5);
            a6 = _mm256_fmadd_ps(x, _mm256_loadu_ps(q6 + d), a6);
            a7 = _mm256_fmadd_ps(x, _mm256_loadu_ps(q7 + d), a7);
        }
        int h = kv_head * 8;
        scores[(size_t)(h + 0) * n_keys + i] = xe_sum_f32x8(a0);
        scores[(size_t)(h + 1) * n_keys + i] = xe_sum_f32x8(a1);
        scores[(size_t)(h + 2) * n_keys + i] = xe_sum_f32x8(a2);
        scores[(size_t)(h + 3) * n_keys + i] = xe_sum_f32x8(a3);
        scores[(size_t)(h + 4) * n_keys + i] = xe_sum_f32x8(a4);
        scores[(size_t)(h + 5) * n_keys + i] = xe_sum_f32x8(a5);
        scores[(size_t)(h + 6) * n_keys + i] = xe_sum_f32x8(a6);
        scores[(size_t)(h + 7) * n_keys + i] = xe_sum_f32x8(a7);
    }
}

static void xe_kv_softmax(float *scores, int q_heads, int n_keys) {
    for (int h = 0; h < q_heads; h++) {
        float *row = scores + (size_t)h * n_keys;
        float max = row[0];
        for (int i = 1; i < n_keys; i++)
            if (row[i] > max) max = row[i];
        float sum = 0.0f;
        for (int i = 0; i < n_keys; i++) {
            row[i] = expf(row[i] - max);
            sum += row[i];
        }
        float inv = 1.0f / sum;
        for (int i = 0; i < n_keys; i++) row[i] *= inv;
    }
}

static void xe_kv_v_g2(float *out, const float *scores, const _Float16 *v_cache,
                       int head_dim, int capacity, int first, int n_keys, int ring, int kv_head,
                       int out_group, int key_begin, int key_end) {
    float *o0 = out + (size_t)(out_group * 2) * head_dim;
    float *o1 = o0 + head_dim;
    for (int ib = key_begin; ib < key_end; ib += 32) {
        int ie = ib + 32 < key_end ? ib + 32 : key_end;
        for (int d = 0; d < head_dim; d += 8) {
            __m256 a0 = _mm256_loadu_ps(o0 + d);
            __m256 a1 = _mm256_loadu_ps(o1 + d);
            for (int i = ib; i < ie; i++) {
                int p = ring ? (first + i) & (capacity - 1) : first + i;
                __m256 x = xe_load_f16x8(v_cache + ((size_t)kv_head * capacity + p) * head_dim + d);
                a0 = _mm256_fmadd_ps(x, _mm256_set1_ps(scores[(size_t)(kv_head * 2) * n_keys + i]), a0);
                a1 = _mm256_fmadd_ps(x, _mm256_set1_ps(scores[(size_t)(kv_head * 2 + 1) * n_keys + i]), a1);
            }
            _mm256_storeu_ps(o0 + d, a0);
            _mm256_storeu_ps(o1 + d, a1);
        }
    }
}

static void xe_kv_v_g8(float *out, const float *scores, const _Float16 *v_cache,
                       int head_dim, int capacity, int first, int n_keys, int ring, int kv_head,
                       int out_group, int key_begin, int key_end) {
    float *o0 = out + (size_t)(out_group * 8) * head_dim;
    float *o1 = o0 + head_dim;
    float *o2 = o1 + head_dim;
    float *o3 = o2 + head_dim;
    float *o4 = o3 + head_dim;
    float *o5 = o4 + head_dim;
    float *o6 = o5 + head_dim;
    float *o7 = o6 + head_dim;
    for (int ib = key_begin; ib < key_end; ib += 32) {
        int ie = ib + 32 < key_end ? ib + 32 : key_end;
        for (int d = 0; d < head_dim; d += 8) {
            __m256 a0 = _mm256_loadu_ps(o0 + d);
            __m256 a1 = _mm256_loadu_ps(o1 + d);
            __m256 a2 = _mm256_loadu_ps(o2 + d);
            __m256 a3 = _mm256_loadu_ps(o3 + d);
            __m256 a4 = _mm256_loadu_ps(o4 + d);
            __m256 a5 = _mm256_loadu_ps(o5 + d);
            __m256 a6 = _mm256_loadu_ps(o6 + d);
            __m256 a7 = _mm256_loadu_ps(o7 + d);
            for (int i = ib; i < ie; i++) {
                int p = ring ? (first + i) & (capacity - 1) : first + i;
                __m256 x = xe_load_f16x8(v_cache + ((size_t)kv_head * capacity + p) * head_dim + d);
                int h = kv_head * 8;
                a0 = _mm256_fmadd_ps(x, _mm256_set1_ps(scores[(size_t)(h + 0) * n_keys + i]), a0);
                a1 = _mm256_fmadd_ps(x, _mm256_set1_ps(scores[(size_t)(h + 1) * n_keys + i]), a1);
                a2 = _mm256_fmadd_ps(x, _mm256_set1_ps(scores[(size_t)(h + 2) * n_keys + i]), a2);
                a3 = _mm256_fmadd_ps(x, _mm256_set1_ps(scores[(size_t)(h + 3) * n_keys + i]), a3);
                a4 = _mm256_fmadd_ps(x, _mm256_set1_ps(scores[(size_t)(h + 4) * n_keys + i]), a4);
                a5 = _mm256_fmadd_ps(x, _mm256_set1_ps(scores[(size_t)(h + 5) * n_keys + i]), a5);
                a6 = _mm256_fmadd_ps(x, _mm256_set1_ps(scores[(size_t)(h + 6) * n_keys + i]), a6);
                a7 = _mm256_fmadd_ps(x, _mm256_set1_ps(scores[(size_t)(h + 7) * n_keys + i]), a7);
            }
            _mm256_storeu_ps(o0 + d, a0);
            _mm256_storeu_ps(o1 + d, a1);
            _mm256_storeu_ps(o2 + d, a2);
            _mm256_storeu_ps(o3 + d, a3);
            _mm256_storeu_ps(o4 + d, a4);
            _mm256_storeu_ps(o5 + d, a5);
            _mm256_storeu_ps(o6 + d, a6);
            _mm256_storeu_ps(o7 + d, a7);
        }
    }
}

void xe_kv_attention_f16(const float *q, const _Float16 *k_cache, const _Float16 *v_cache,
                         int n_kv_heads, int head_dim, int capacity, int pos, int sliding_window,
                         float *scores, float *out) {
    int group = XE_Q_HEADS / n_kv_heads;
    if (pos < 0 || (pos >= capacity && sliding_window == 0)) xe_fatal("kv: invalid attention position %d", pos);
    if (!((n_kv_heads == XE_SWA_KV_HEADS && head_dim == XE_SWA_HEAD_DIM && group == 2) ||
          (n_kv_heads == XE_GLOBAL_KV_HEADS && head_dim == XE_GLOBAL_HEAD_DIM && group == 8)))
        xe_fatal("kv: unsupported attention shape");
    if (sliding_window && (capacity != sliding_window || (capacity & (capacity - 1)) != 0))
        xe_fatal("kv: invalid ring capacity %d", capacity);
    int n_keys = pos + 1;
    if (sliding_window && n_keys > sliding_window) n_keys = sliding_window;
    int first = pos + 1 - n_keys;
    int ring = sliding_window != 0;
    for (int h = 0; h < n_kv_heads; h++) {
        if (group == 8) xe_kv_qk_g8(scores, q, k_cache, head_dim, capacity, first, n_keys, ring, h, 0, n_keys);
        else xe_kv_qk_g2(scores, q, k_cache, head_dim, capacity, first, n_keys, ring, h, 0, n_keys);
    }
    xe_kv_softmax(scores, XE_Q_HEADS, n_keys);
    memset(out, 0, (size_t)XE_Q_HEADS * head_dim * sizeof(*out));
    for (int h = 0; h < n_kv_heads; h++) {
        if (group == 8) xe_kv_v_g8(out, scores, v_cache, head_dim, capacity, first, n_keys, ring, h, h, 0, n_keys);
        else xe_kv_v_g2(out, scores, v_cache, head_dim, capacity, first, n_keys, ring, h, h, 0, n_keys);
    }
}

void xe_session_kv_attention(xe_session *s, int layer, int pos, const float *q,
                             float *scores, float *out) {
    if (pos < 0 || pos >= s->engine->context) xe_fatal("kv: position %d out of range [0, %d]", pos, s->engine->context - 1);
    int global = XE_IS_GLOBAL(layer);
    xe_kv_attention_f16(q, xe_kv_layer_ptr(s, layer, 0), xe_kv_layer_ptr(s, layer, 1),
                        global ? XE_GLOBAL_KV_HEADS : XE_SWA_KV_HEADS,
                        global ? XE_GLOBAL_HEAD_DIM : XE_SWA_HEAD_DIM,
                        global ? s->engine->context : XE_SWA_WINDOW, pos,
                        global ? 0 : XE_SWA_WINDOW, scores, out);
}

typedef struct {
    const xe_layer *layer;
    int global;
    int pos;
} xe_head_prepare_arg;

static void xe_store_f16_vector(_Float16 *dst, const float *src, int n) {
    for (int i = 0; i < n; i += 8) {
        __m128i h = _mm256_cvtps_ph(_mm256_loadu_ps(src + i),
                                    _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC);
        _mm_storeu_si128((__m128i *)(dst + i), h);
    }
}

static void xe_head_prepare_phase(xe_session *s, const void *opaque, int worker, int workers) {
    const xe_head_prepare_arg *a = opaque;
    int head_dim = a->global ? XE_GLOBAL_HEAD_DIM : XE_SWA_HEAD_DIM;
    int kv_heads = a->global ? XE_GLOBAL_KV_HEADS : XE_SWA_KV_HEADS;
    int capacity = a->global ? s->engine->context : XE_SWA_WINDOW;
    int slot = a->global ? a->pos : a->pos & (XE_SWA_WINDOW - 1);
    int tasks = XE_Q_HEADS + 2 * kv_heads;
    int begin = tasks * worker / workers;
    int end = tasks * (worker + 1) / workers;
    const float *cosines = a->global ? s->rope_global_cos : s->rope_swa_cos;
    const float *sines = a->global ? s->rope_global_sin : s->rope_swa_sin;
    _Float16 *kc = xe_kv_layer_ptr(s, (int)(a->layer - s->engine->layers), 0);
    _Float16 *vc = xe_kv_layer_ptr(s, (int)(a->layer - s->engine->layers), 1);

    for (int task = begin; task < end; task++) {
        if (task < XE_Q_HEADS) {
            float *head = s->q + (size_t)task * head_dim;
            xe_rmsnorm_engine(s->engine, head, a->layer->q_norm, head_dim, head);
            xe_rope_apply(head, 1, head_dim, cosines, sines);
        } else if (task < XE_Q_HEADS + kv_heads) {
            int h = task - XE_Q_HEADS;
            float *head = s->k + (size_t)h * head_dim;
            xe_rmsnorm_engine(s->engine, head, a->layer->k_norm, head_dim, head);
            xe_rope_apply(head, 1, head_dim, cosines, sines);
            xe_store_f16_vector(kc + ((size_t)h * capacity + slot) * head_dim, head, head_dim);
        } else {
            int h = task - XE_Q_HEADS - kv_heads;
            float *head = s->v + (size_t)h * head_dim;
            xe_rmsnorm_engine(s->engine, head, NULL, head_dim, head);
            xe_store_f16_vector(vc + ((size_t)h * capacity + slot) * head_dim, head, head_dim);
        }
    }
}

typedef struct {
    int layer;
    int pos;
} xe_attention_arg;

static void xe_attention_direct_phase(xe_session *s, const void *opaque, int worker, int workers) {
    const xe_attention_arg *a = opaque;
    int global = XE_IS_GLOBAL(a->layer);
    int kv_heads = global ? XE_GLOBAL_KV_HEADS : XE_SWA_KV_HEADS;
    int head_dim = global ? XE_GLOBAL_HEAD_DIM : XE_SWA_HEAD_DIM;
    int capacity = global ? s->engine->context : XE_SWA_WINDOW;
    int n_keys = a->pos + 1;
    if (!global && n_keys > XE_SWA_WINDOW) n_keys = XE_SWA_WINDOW;
    int first = a->pos + 1 - n_keys;
    int ring = !global;
    int group = XE_Q_HEADS / kv_heads;
    const _Float16 *kc = xe_kv_layer_ptr(s, a->layer, 0);
    const _Float16 *vc = xe_kv_layer_ptr(s, a->layer, 1);
    int begin = kv_heads * worker / workers;
    int end = kv_heads * (worker + 1) / workers;
    for (int h = begin; h < end; h++) {
        if (group == 8)
            xe_kv_qk_g8(s->scores, s->q, kc, head_dim, capacity, first, n_keys, ring, h, 0, n_keys);
        else
            xe_kv_qk_g2(s->scores, s->q, kc, head_dim, capacity, first, n_keys, ring, h, 0, n_keys);
        xe_kv_softmax(s->scores + (size_t)(h * group) * n_keys, group, n_keys);
        memset(s->attn_heads + (size_t)(h * group) * head_dim, 0,
               (size_t)group * head_dim * sizeof(*s->attn_heads));
        if (group == 8)
            xe_kv_v_g8(s->attn_heads, s->scores, vc, head_dim, capacity, first,
                        n_keys, ring, h, h, 0, n_keys);
        else
            xe_kv_v_g2(s->attn_heads, s->scores, vc, head_dim, capacity, first,
                        n_keys, ring, h, h, 0, n_keys);
    }
}

static void xe_attention_global_qk_phase(xe_session *s, const void *opaque, int worker, int workers) {
    const xe_attention_arg *a = opaque;
    int n_keys = a->pos + 1;
    const _Float16 *kc = xe_kv_layer_ptr(s, a->layer, 0);
    int tasks = XE_GLOBAL_KV_HEADS * 3;
    int begin = tasks * worker / workers;
    int end = tasks * (worker + 1) / workers;
    for (int task = begin; task < end; task++) {
        int h = task / 3;
        int shard = task % 3;
        int ib = n_keys * shard / 3;
        int ie = n_keys * (shard + 1) / 3;
        xe_kv_qk_g8(s->scores, s->q, kc, XE_GLOBAL_HEAD_DIM, s->engine->context, 0,
                    n_keys, 0, h, ib, ie);
    }
}

static void xe_attention_global_softmax_phase(xe_session *s, const void *opaque, int worker, int workers) {
    const xe_attention_arg *a = opaque;
    int n_keys = a->pos + 1;
    int begin = XE_Q_HEADS * worker / workers;
    int end = XE_Q_HEADS * (worker + 1) / workers;
    if (begin < end)
        xe_kv_softmax(s->scores + (size_t)begin * n_keys, end - begin, n_keys);
}

static void xe_attention_global_v_phase(xe_session *s, const void *opaque, int worker, int workers) {
    const xe_attention_arg *a = opaque;
    int n_keys = a->pos + 1;
    const _Float16 *vc = xe_kv_layer_ptr(s, a->layer, 1);
    int tasks = XE_GLOBAL_KV_HEADS * 3;
    int begin = tasks * worker / workers;
    int end = tasks * (worker + 1) / workers;
    for (int task = begin; task < end; task++) {
        int h = task / 3;
        int shard = task % 3;
        int ib = n_keys * shard / 3;
        int ie = n_keys * (shard + 1) / 3;
        float *partial = s->attn_partial + (size_t)task * 8 * XE_GLOBAL_HEAD_DIM;
        memset(partial, 0, (size_t)8 * XE_GLOBAL_HEAD_DIM * sizeof(*partial));
        xe_kv_v_g8(partial, s->scores, vc, XE_GLOBAL_HEAD_DIM, s->engine->context, 0,
                    n_keys, 0, h, 0, ib, ie);
    }
}

static void xe_attention_global_reduce_phase(xe_session *s, const void *opaque, int worker, int workers) {
    (void)opaque;
    int begin = XE_Q_HEADS * worker / workers;
    int end = XE_Q_HEADS * (worker + 1) / workers;
    for (int h = begin; h < end; h++) {
        int kv = h / 8;
        int local = h % 8;
        float *out = s->attn_heads + (size_t)h * XE_GLOBAL_HEAD_DIM;
        const float *p0 = s->attn_partial + (size_t)(kv * 3) * 8 * XE_GLOBAL_HEAD_DIM + (size_t)local * XE_GLOBAL_HEAD_DIM;
        const float *p1 = p0 + (size_t)8 * XE_GLOBAL_HEAD_DIM;
        const float *p2 = p1 + (size_t)8 * XE_GLOBAL_HEAD_DIM;
        for (int d = 0; d < XE_GLOBAL_HEAD_DIM; d += 8) {
            __m256 sum = _mm256_add_ps(_mm256_loadu_ps(p0 + d), _mm256_loadu_ps(p1 + d));
            sum = _mm256_add_ps(sum, _mm256_loadu_ps(p2 + d));
            _mm256_storeu_ps(out + d, sum);
        }
    }
}

static void xe_attention_parallel_mode(xe_session *s, int layer, int pos, int sharded) {
    xe_attention_arg a = { layer, pos };
    if (!sharded) {
        xe_dispatch(s->engine, s, xe_attention_direct_phase, &a, 1);
        return;
    }
    xe_dispatch(s->engine, s, xe_attention_global_qk_phase, &a, 1);
    xe_dispatch(s->engine, s, xe_attention_global_softmax_phase, &a, 1);
    xe_dispatch(s->engine, s, xe_attention_global_v_phase, &a, 1);
    xe_dispatch(s->engine, s, xe_attention_global_reduce_phase, &a, 1);
}

static void xe_attention_parallel(xe_session *s, int layer, int pos) {
    int sharded = XE_IS_GLOBAL(layer) && pos + 1 >= XE_GLOBAL_SHARD_CROSSOVER;
    xe_attention_parallel_mode(s, layer, pos, sharded);
}

enum {
    XE_MOE_EXPERT_PARALLEL,
    XE_MOE_GENERIC_BATCHED,
    XE_MOE_FUSED_BATCHED
};

enum {
    XE_SOFTCAP_RAW,
    XE_SOFTCAP_IMMEDIATE,
    XE_SOFTCAP_SECOND_LOOP,
    XE_SOFTCAP_RAW4,
    XE_SOFTCAP_RAW_VECTOR
};

static void xe_embed_decode(const xe_engine *e, int32_t token, float *x) {
    int blocks = XE_EMBD / 32;
    float factor = sqrtf((float)XE_EMBD);
    for (int b = 0; b < blocks; b++) {
        _Alignas(16) uint8_t packed[16];
        xe_q4_copy_block(&e->tok_embd, (uint64_t)token, blocks, b, packed);
        float d = _cvtsh_ss(xe_q4_scale(&e->tok_embd, (uint64_t)token, blocks, b)) * factor;
        for (int i = 0; i < 16; i++) {
            x[32 * b + i] = (float)((packed[i] & 15) - 8) * d;
            x[32 * b + i + 16] = (float)((packed[i] >> 4) - 8) * d;
        }
    }
}

static void xe_dispatch_matvec(xe_session *s, const xe_q4 *w, uint64_t first_row,
                               int n_in, int n_out, const xe_q8 *q, float *y) {
    xe_matvec_arg a = { w, first_row, n_in, n_out, q, y };
    xe_dispatch(s->engine, s, xe_matvec_phase, &a, 1);
}

static void xe_moe_run(xe_session *s, const xe_layer *l, int mode) {
    xe_moe_arg a = { l };
    if (mode == XE_MOE_EXPERT_PARALLEL) {
        xe_dispatch(s->engine, s, xe_moe_expert_phase, &a, 1);
        xe_dispatch(s->engine, s, xe_moe_reduce_phase, &a, 1);
    } else if (mode == XE_MOE_GENERIC_BATCHED) {
        xe_dispatch(s->engine, s, xe_moe_generic_gate_phase, &a, 1);
        xe_dispatch(s->engine, s, xe_moe_generic_act_phase, &a, 1);
        xe_dispatch(s->engine, s, xe_moe_down_phase, &a, 1);
    } else {
        xe_dispatch(s->engine, s, xe_moe_fused_gate_phase, &a, 1);
        xe_dispatch(s->engine, s, xe_moe_down_phase, &a, 1);
    }
}

static void xe_attention_half_decode(xe_session *s, int layer, int pos) {
#ifdef XE_BENCH_TG_PROFILE
    double profile_start = xe_tg_profile_now();
#endif
    const xe_layer *l = &s->engine->layers[layer];
    int global = XE_IS_GLOBAL(layer);
    int head_dim = global ? XE_GLOBAL_HEAD_DIM : XE_SWA_HEAD_DIM;
    int kv_heads = global ? XE_GLOBAL_KV_HEADS : XE_SWA_KV_HEADS;
    int q_dim = XE_Q_HEADS * head_dim;
    int kv_dim = kv_heads * head_dim;

    xe_norm_q8_arg norm = { s->hidden, l->attn_norm, xe_rms_scale_engine(s->engine, s->hidden, XE_EMBD),
                            XE_EMBD, &s->q8_main };
    xe_dispatch(s->engine, s, xe_norm_q8_phase, &norm, 1);

    xe_matvec_batch_arg projections;
    projections.n_ops = global ? 2 : 3;
    projections.op[0] = (xe_matvec_arg){ &l->attn_q, 0, XE_EMBD, q_dim, &s->q8_main, s->q };
    projections.op[1] = (xe_matvec_arg){ &l->attn_k, 0, XE_EMBD, kv_dim, &s->q8_main, s->k };
    if (!global)
        projections.op[2] = (xe_matvec_arg){ &l->attn_v, 0, XE_EMBD, kv_dim, &s->q8_main, s->v };
    xe_dispatch(s->engine, s, xe_matvec_batch_phase, &projections, 1);
    if (global) memcpy(s->v, s->k, (size_t)kv_dim * sizeof(*s->v));

    xe_head_prepare_arg heads = { l, global, pos };
    xe_dispatch(s->engine, s, xe_head_prepare_phase, &heads, 1);
    xe_attention_parallel(s, layer, pos);

    xe_quantize_q8(s->attn_heads, q_dim, &s->q8_main);
    xe_dispatch_matvec(s, &l->attn_o, 0, q_dim, XE_EMBD, &s->q8_main, s->attn_proj);
    xe_rmsnorm_engine(s->engine, s->attn_proj, l->post_attn_norm, XE_EMBD, s->attn_proj);
    for (int i = 0; i < XE_EMBD; i++)
        s->attn_out[i] = s->attn_proj[i] + s->hidden[i];
#ifdef XE_BENCH_TG_PROFILE
    xe_tg_profile_data.attention += xe_tg_profile_now() - profile_start;
#endif
}

static void xe_ffn_half_decode(xe_session *s, int layer, int moe_mode, int overlap_router) {
#ifdef XE_BENCH_TG_PROFILE
    double profile_start = xe_tg_profile_now();
#endif
    const xe_layer *l = &s->engine->layers[layer];
    float common_scale = xe_rms_scale_engine(s->engine, s->attn_out, XE_EMBD);
    xe_ffn_input_arg input = { l, common_scale };
    xe_dispatch(s->engine, s, xe_ffn_input_phase, &input, 1);

    xe_dense_gate_arg dense = { l };
    if (overlap_router) {
        uint64_t epoch = xe_publish_phase(s->engine, s, xe_dense_gate_phase, &dense, 0);
        xe_router(s, l, layer);
        xe_wait_followers(s->engine, epoch);
    } else {
        xe_dispatch(s->engine, s, xe_dense_gate_phase, &dense, 1);
        xe_router(s, l, layer);
    }

    xe_dispatch_matvec(s, &l->ffn_down, 0, XE_DENSE_FFN, XE_EMBD,
                       &s->q8_dense_act, s->dense_out);
#ifdef XE_BENCH_TG_PROFILE
    xe_tg_profile_data.dense += xe_tg_profile_now() - profile_start;
    profile_start = xe_tg_profile_now();
#endif
    xe_moe_run(s, l, moe_mode);
#ifdef XE_BENCH_TG_PROFILE
    xe_tg_profile_data.moe += xe_tg_profile_now() - profile_start;
    profile_start = xe_tg_profile_now();
#endif

    xe_rmsnorm_engine(s->engine, s->dense_out, l->post_ffw_norm1, XE_EMBD, s->dense_out);
    xe_rmsnorm_engine(s->engine, s->moe_out, l->post_ffw_norm2, XE_EMBD, s->moe_out);
    for (int i = 0; i < XE_EMBD; i++) s->combined[i] = s->dense_out[i] + s->moe_out[i];
    xe_rmsnorm_engine(s->engine, s->combined, l->post_ffw_norm, XE_EMBD, s->combined);
    float out_scale = l->layer_out_scale[0];
    for (int i = 0; i < XE_EMBD; i++)
        s->hidden[i] = (s->combined[i] + s->attn_out[i]) * out_scale;
#ifdef XE_BENCH_TG_PROFILE
    xe_tg_profile_data.glue += xe_tg_profile_now() - profile_start;
#endif
}

typedef struct {
    int mode;
} xe_output_arg;

static void xe_output_phase(xe_session *s, const void *opaque, int worker, int workers) {
    const xe_output_arg *a = opaque;
    int groups = XE_VOCAB / 8;
    int begin = 8 * (groups * worker / workers);
    int end = 8 * (groups * (worker + 1) / workers);
    if (a->mode == XE_SOFTCAP_RAW4) {
        xe_matvec_q4_q8_rows4(&s->engine->tok_embd, 0, XE_EMBD, begin, end,
                              &s->q8_main, s->logits);
        return;
    }
    if (a->mode == XE_SOFTCAP_RAW_VECTOR) {
        xe_matvec_q4_q8_rows_vector(&s->engine->tok_embd, 0, XE_EMBD, begin, end,
                                    &s->q8_main, s->logits);
        return;
    }
    if (a->mode == XE_SOFTCAP_IMMEDIATE) {
        for (int row = begin; row < end; row++) {
            float raw = xe_q4_q8_dot(&s->engine->tok_embd, (uint64_t)row,
                                     XE_EMBD, &s->q8_main);
            s->logits[row] = XE_LOGIT_SOFTCAP * tanhf(raw / XE_LOGIT_SOFTCAP);
        }
        return;
    }
    xe_matvec_q4_q8_rows(&s->engine->tok_embd, 0, XE_EMBD, begin, end,
                         &s->q8_main, s->logits);
    if (a->mode == XE_SOFTCAP_SECOND_LOOP) {
        for (int row = begin; row < end; row++)
            s->logits[row] = XE_LOGIT_SOFTCAP * tanhf(s->logits[row] / XE_LOGIT_SOFTCAP);
    }
}

static void xe_output_decode(xe_session *s, int softcap_mode) {
#ifdef XE_TEST_OUTPUT_COUNT
    xe_test_output_calls++;
#endif
#ifdef XE_BENCH_TG_PROFILE
    double profile_start = xe_tg_profile_now();
#endif
    xe_norm_q8_arg norm = { s->hidden, s->engine->out_norm,
                            xe_rms_scale_engine(s->engine, s->hidden, XE_EMBD), XE_EMBD, &s->q8_main };
    xe_dispatch(s->engine, s, xe_norm_q8_phase, &norm, 1);
    xe_output_arg output = { softcap_mode };
    xe_dispatch(s->engine, s, xe_output_phase, &output, 1);
#ifdef XE_BENCH_TG_PROFILE
    xe_tg_profile_data.output += xe_tg_profile_now() - profile_start;
#endif
}

static void xe_prefill_batch_append_run(xe_session *s,
                                        const int32_t *tokens,
                                        int rows, int batch_start) {
    if (rows < 1 || rows > 512)
        xe_fatal("prefill batch supports M1 through M512");
    if (s->prefill_pending)
        xe_fatal("prefill batch already pending");
    if (batch_start != s->n_tokens)
        xe_fatal("prefill batch expected position %d, received %d",
                 s->n_tokens, batch_start);
#ifdef XE_TEST_SESSION
    extern size_t xe_test_prefill_batches;
    xe_test_prefill_batches++;
#endif
    xe_engine *e = s->engine;
    xe_prefill_workspace w;
    xe_prefill_workspace_layout(&w, s->prefill_workspace);
    for (int row = 0; row < rows; row++)
        xe_embed_decode(e, tokens[row],
                        w.hidden[0] + (size_t)row * XE_EMBD);
    xe_prefill_rope_prepare_batch(e, &w, rows, batch_start, 0);
    xe_prefill_rope_prepare_batch(e, &w, rows, batch_start, 1);
    for (int layer = 0; layer < XE_LAYERS; layer++) {
        xe_prefill_attention_batch_append(s, layer, &w, rows, batch_start);
        xe_prefill_ffn_append(e, layer, &w, rows);
        float *hidden = w.hidden[0];
        w.hidden[0] = w.hidden[1];
        w.hidden[1] = hidden;
    }
    memcpy(s->tokens + batch_start, tokens, (size_t)rows * sizeof(*tokens));
    s->prefill_pending = 1;
    s->prefill_pending_start = batch_start;
    s->prefill_pending_rows = rows;
    s->prefill_pending_hidden = w.hidden[0] +
                                (size_t)(rows - 1) * XE_EMBD;
}

static void xe_prefill_batch_complete(xe_session *s, int output_logits) {
    if (!s->prefill_pending) xe_fatal("prefill batch is not pending");
    memcpy(s->hidden, s->prefill_pending_hidden,
           XE_EMBD * sizeof(*s->hidden));
    s->n_tokens = s->prefill_pending_start + s->prefill_pending_rows;
    s->prefill_pending = 0;
    s->prefill_pending_hidden = NULL;
    if (output_logits) {
        xe_workers_begin(s->engine);
        xe_output_decode(s, XE_SOFTCAP_SECOND_LOOP);
        xe_workers_end(s->engine);
    }
}

static void xe_prefill_batch_run(xe_session *s, const int32_t *tokens,
                                 int rows, int batch_start,
                                 int output_logits) {
    xe_prefill_batch_append_run(s, tokens, rows, batch_start);
    xe_ze_check("zeCommandListHostSynchronize prefill batch",
                zeCommandListHostSynchronize(s->engine->gpu.commands,
                                             UINT64_MAX));
    xe_prefill_batch_complete(s, output_logits);
}

static void __attribute__((unused)) xe_prefill_initial_run(
        xe_session *s, const int32_t *tokens, int rows, int output_logits) {
    if (s->n_tokens != 0) xe_fatal("prefill initial requires an empty session");
    xe_prefill_batch_run(s, tokens, rows, 0, output_logits);
}

static void xe_decode_token_mode(xe_session *s, int32_t token, int pos,
                                 int moe_mode, int overlap_router, int softcap_mode,
                                 int output_logits) {
    if (token < 0 || token >= XE_VOCAB) xe_fatal("decode: token %d out of range", token);
    if (pos < 0 || pos >= s->engine->context) xe_fatal("decode: position %d out of range", pos);
    if (pos != s->n_tokens) xe_fatal("decode: expected position %d, received %d", s->n_tokens, pos);
#ifdef XE_TEST_SESSION
    extern size_t xe_test_decode_tokens;
    xe_test_decode_tokens++;
#endif

#ifdef XE_BENCH_TG_PROFILE
    double profile_start = xe_tg_profile_now();
#endif
    xe_embed_decode(s->engine, token, s->hidden);
    xe_rope_prepare(s, pos);
#ifdef XE_BENCH_TG_PROFILE
    xe_tg_profile_data.embed_rope += xe_tg_profile_now() - profile_start;
#endif
    xe_workers_begin(s->engine);
    for (int layer = 0; layer < XE_LAYERS; layer++) {
        xe_attention_half_decode(s, layer, pos);
        xe_ffn_half_decode(s, layer, moe_mode, overlap_router);
    }
    if (output_logits) xe_output_decode(s, softcap_mode);
#ifdef XE_BENCH_TG_PROFILE
    profile_start = xe_tg_profile_now();
#endif
    xe_workers_end(s->engine);
#ifdef XE_BENCH_TG_PROFILE
    xe_tg_profile_data.worker_end += xe_tg_profile_now() - profile_start;
#endif
    s->n_tokens = pos + 1;
}

static void __attribute__((unused)) xe_decode_token(
        xe_session *s, int32_t token, int pos) {
    xe_decode_token_mode(s, token, pos, XE_MOE_GENERIC_BATCHED, 1,
                         XE_SOFTCAP_SECOND_LOOP, 1);
}

static int xe_sample_better(xe_sample_candidate a, xe_sample_candidate b) {
    if (a.logit != b.logit) return a.logit > b.logit;
    return a.id < b.id;
}

static void xe_sample_heap_up(xe_sample_candidate *heap, int at) {
    while (at > 0) {
        int parent = (at - 1) / 2;
        if (!xe_sample_better(heap[parent], heap[at])) break;
        xe_sample_candidate tmp = heap[parent];
        heap[parent] = heap[at];
        heap[at] = tmp;
        at = parent;
    }
}

static void xe_sample_heap_down(xe_sample_candidate *heap, int n, int at) {
    for (;;) {
        int left = 2 * at + 1;
        int right = left + 1;
        int worst = at;
        if (left < n && xe_sample_better(heap[worst], heap[left])) worst = left;
        if (right < n && xe_sample_better(heap[worst], heap[right])) worst = right;
        if (worst == at) break;
        xe_sample_candidate tmp = heap[at];
        heap[at] = heap[worst];
        heap[worst] = tmp;
        at = worst;
    }
}

static int xe_sample_compare(const void *pa, const void *pb) {
    const xe_sample_candidate *a = pa;
    const xe_sample_candidate *b = pb;
    if (a->logit < b->logit) return 1;
    if (a->logit > b->logit) return -1;
    return (a->id > b->id) - (a->id < b->id);
}

static uint64_t xe_sample_rng_next(uint64_t *state) {
    uint64_t x = *state;
    if (x == 0) x = UINT64_C(0x9e3779b97f4a7c15);
    x ^= x >> 12;
    x ^= x << 25;
    x ^= x >> 27;
    *state = x;
    return x * UINT64_C(0x2545f4914f6cdd1d);
}

static float xe_sample_rng_f32(uint64_t *state) {
    uint64_t x = xe_sample_rng_next(state);
    return (float)((x >> 40) & UINT64_C(0xffffff)) / 16777216.0f;
}

static int32_t xe_sample_logits(const float *logits, int n_vocab,
                                xe_sample_candidate *candidates, xe_sampler *sp) {
    if (!logits || !candidates || !sp) xe_fatal("sample: missing input");
    if (!isfinite(sp->temperature) || sp->temperature < 0.0f)
        xe_fatal("sample: temperature must be finite and non-negative");
    if (sp->top_k < 0 || sp->top_k > n_vocab)
        xe_fatal("sample: top-k %d out of range [0, %d]", sp->top_k, n_vocab);
    if (!isfinite(sp->top_p) || sp->top_p <= 0.0f || sp->top_p > 1.0f)
        xe_fatal("sample: top-p must be in (0, 1]");

    int32_t best = 0;
    float best_logit = -INFINITY;
    for (int32_t id = 0; id < n_vocab; id++) {
        float v = logits[id];
        if (!isfinite(v)) xe_fatal("sample: logit %d is not finite (%g)", id, (double)v);
        if (v > best_logit) {
            best_logit = v;
            best = id;
        }
    }
    if (sp->temperature == 0.0f || sp->top_k == 1) return best;

    int n = 0;
    if (sp->top_k == 0) {
        for (int32_t id = 0; id < n_vocab; id++)
            candidates[n++] = (xe_sample_candidate){ id, logits[id], 0.0f };
    } else {
        for (int32_t id = 0; id < n_vocab; id++) {
            xe_sample_candidate candidate = { id, logits[id], 0.0f };
            if (n < sp->top_k) {
                candidates[n] = candidate;
                xe_sample_heap_up(candidates, n);
                n++;
            } else if (xe_sample_better(candidate, candidates[0])) {
                candidates[0] = candidate;
                xe_sample_heap_down(candidates, n, 0);
            }
        }
    }

    if (sp->top_k != 0 || sp->top_p < 1.0f)
        qsort(candidates, (size_t)n, sizeof(*candidates), xe_sample_compare);

    float sum = 0.0f;
    for (int i = 0; i < n; i++) {
        float p = expf((candidates[i].logit - best_logit) / sp->temperature);
        candidates[i].prob = p;
        sum += p;
    }
    if (!(sum > 0.0f) || !isfinite(sum)) xe_fatal("sample: invalid probability sum %g", (double)sum);

    int filtered = n;
    float filtered_sum = sum;
    if (sp->top_p < 1.0f) {
        filtered = 0;
        filtered_sum = 0.0f;
        float target = sp->top_p * sum;
        do {
            filtered_sum += candidates[filtered].prob;
            filtered++;
        } while (filtered < n && filtered_sum < target);
    }

    float draw = xe_sample_rng_f32(&sp->rng_state) * filtered_sum;
    for (int i = 0; i < filtered; i++) {
        draw -= candidates[i].prob;
        if (draw <= 0.0f) return candidates[i].id;
    }
    return candidates[filtered - 1].id;
}

static void xe_session_swa_copy_slot(xe_session *s, int slot) {
    size_t bytes = XE_SWA_HEAD_DIM * sizeof(*s->swa_k);
    for (int layer = 0; layer < XE_LAYERS - XE_GLOBAL_LAYERS; layer++) {
        size_t layer_offset = (size_t)layer * XE_SWA_LAYER_ELEMS;
        for (int head = 0; head < XE_SWA_KV_HEADS; head++) {
            size_t offset = layer_offset +
                ((size_t)head * XE_SWA_WINDOW + slot) * XE_SWA_HEAD_DIM;
            memcpy(s->swa_spare_k + offset, s->swa_k + offset, bytes);
            memcpy(s->swa_spare_v + offset, s->swa_v + offset, bytes);
        }
    }
    s->swa_dirty[slot] = 0;
}

static void xe_session_swa_mark(xe_session *s, int first, int end) {
    int rows = end - first;
    if (rows <= 0) return;
    if (rows >= XE_SWA_WINDOW) {
        memset(s->swa_dirty, 1, sizeof s->swa_dirty);
        return;
    }
    for (int pos = first; pos < end; pos++)
        s->swa_dirty[pos & (XE_SWA_WINDOW - 1)] = 1;
}

static void xe_session_swa_mirror(xe_session *s, int first, int end) {
    if (!s->swa_spare_k || !s->swa_spare_v) return;
    if (end - first > XE_SWA_WINDOW) first = end - XE_SWA_WINDOW;
    for (int pos = first; pos < end; pos++)
        xe_session_swa_copy_slot(s, pos & (XE_SWA_WINDOW - 1));
}

static void xe_session_swa_repair(xe_session *s) {
    int first = s->n_tokens > XE_SWA_WINDOW
                ? s->n_tokens - XE_SWA_WINDOW : 0;
    for (int pos = first; pos < s->n_tokens; pos++) {
        int slot = pos & (XE_SWA_WINDOW - 1);
        if (s->swa_dirty[slot]) xe_session_swa_copy_slot(s, slot);
    }
}

xe_session *xe_session_new(xe_engine *e) {
    if (!e || e->context < XE_CONTEXT_MIN || e->context > XE_CONTEXT_MAX)
        xe_fatal("session: invalid engine context capacity");
    if (e->vocab_only) xe_fatal("session: engine opened vocab-only, no weights available");
    xe_worker_pool_init(e);
    xe_session *s = xe_alloc(e, sizeof *s, XE_MEM_HOST);
    memset(s, 0, sizeof *s);
    s->engine = e;
    s->swa_k = xe_alloc(e, XE_SWA_SLAB_ELEMS * sizeof(*s->swa_k), XE_MEM_SHARED);
    s->swa_v = xe_alloc(e, XE_SWA_SLAB_ELEMS * sizeof(*s->swa_v), XE_MEM_SHARED);
    s->swa_spare_k = xe_alloc(e, XE_SWA_SLAB_ELEMS * sizeof(*s->swa_spare_k),
                              XE_MEM_SHARED);
    s->swa_spare_v = xe_alloc(e, XE_SWA_SLAB_ELEMS * sizeof(*s->swa_spare_v),
                              XE_MEM_SHARED);
    memset(s->swa_dirty, 1, sizeof s->swa_dirty);
    s->global_k = xe_alloc(e, xe_global_slab_elems(e) * sizeof(*s->global_k), XE_MEM_SHARED);
    s->global_v = xe_alloc(e, xe_global_slab_elems(e) * sizeof(*s->global_v), XE_MEM_SHARED);
    s->tokens = xe_alloc(e, (size_t)e->context * sizeof(*s->tokens), XE_MEM_HOST);
    s->workspace_size = xe_workspace_layout(s, NULL);
    s->workspace = xe_alloc(e, s->workspace_size, XE_MEM_SHARED);
    xe_workspace_layout(s, s->workspace);
    xe_prefill_workspace prefill;
    s->prefill_workspace_size = xe_prefill_workspace_layout(&prefill, NULL);
    s->prefill_workspace = xe_alloc(e, s->prefill_workspace_size,
                                    XE_MEM_SHARED);
    return s;
}

xe_session *xe_session_shadow_new(xe_session *source) {
    if (!source || source->cow || source->n_tokens <= 0 ||
        !source->swa_spare_k || !source->swa_spare_v)
        return NULL;
    xe_require_owner(source->engine);
    xe_session_swa_repair(source);
    xe_engine *e = source->engine;
    xe_session *s = xe_alloc(e, sizeof *s, XE_MEM_HOST);
    memset(s, 0, sizeof *s);
    s->engine = e;
    s->cow = 1;
    s->cow_source = source;
    s->cow_split = source->n_tokens;
    s->global_k = source->global_k;
    s->global_v = source->global_v;
    s->swa_k = source->swa_spare_k;
    s->swa_v = source->swa_spare_v;
    source->swa_spare_k = NULL;
    source->swa_spare_v = NULL;
    s->tokens = xe_alloc(e, (size_t)e->context * sizeof(*s->tokens), XE_MEM_HOST);
    s->workspace_size = xe_workspace_layout(s, NULL);
    s->workspace = xe_alloc(e, s->workspace_size, XE_MEM_SHARED);
    xe_workspace_layout(s, s->workspace);
    s->prefill_workspace = source->prefill_workspace;
    s->prefill_workspace_size = source->prefill_workspace_size;
    memcpy(s->tokens, source->tokens,
           (size_t)source->n_tokens * sizeof(*s->tokens));
    memcpy(s->hidden, source->hidden, XE_EMBD * sizeof(*s->hidden));
    memcpy(s->logits, source->logits, XE_VOCAB * sizeof(*s->logits));
    s->n_tokens = source->n_tokens;
    return s;
}

void xe_session_free(xe_session *s) {
    if (!s) return;
    xe_require_owner(s->engine);
    xe_engine *e = s->engine;
    if (s->prefill_pending)
        xe_ze_check("zeCommandListHostSynchronize free pending prefill",
                    zeCommandListHostSynchronize(e->gpu.commands,
                                                 UINT64_MAX));
    if (s->cow && s->cow_source && !s->cow_source->swa_spare_k &&
        !s->cow_source->swa_spare_v) {
        xe_session *source = s->cow_source;
        source->swa_spare_k = s->swa_k;
        source->swa_spare_v = s->swa_v;
        s->swa_k = NULL;
        s->swa_v = NULL;
        int end = source->n_tokens > s->n_tokens
                  ? source->n_tokens : s->n_tokens;
        xe_session_swa_mark(source, s->cow_split, end);
    }
    xe_free(e, s->anchor_logits, XE_MEM_HOST);
    xe_free(e, s->anchor_swa_v, XE_MEM_HOST);
    xe_free(e, s->anchor_swa_k, XE_MEM_HOST);
    if (!s->cow)
        xe_free(e, s->prefill_workspace, XE_MEM_SHARED);
    xe_free(e, s->workspace, XE_MEM_SHARED);
    xe_free(e, s->tokens, XE_MEM_HOST);
    if (s->cow) {
        for (int i = 0; i < XE_GLOBAL_LAYERS; i++) {
            xe_free(e, s->cow_global_v[i], XE_MEM_SHARED);
            xe_free(e, s->cow_global_k[i], XE_MEM_SHARED);
        }
    } else {
        xe_free(e, s->global_v, XE_MEM_SHARED);
        xe_free(e, s->global_k, XE_MEM_SHARED);
    }
    xe_free(e, s->swa_v, XE_MEM_SHARED);
    xe_free(e, s->swa_k, XE_MEM_SHARED);
    xe_free(e, s->swa_spare_v, XE_MEM_SHARED);
    xe_free(e, s->swa_spare_k, XE_MEM_SHARED);
    xe_free(e, s, XE_MEM_HOST);
}

static void xe_session_shadow_reserve(xe_session *s, int end) {
    if (!s->cow || end < s->cow_split || end > s->engine->context)
        xe_fatal("shadow reserve: invalid end %d", end);
    int needed = end - s->cow_split;
    if (needed <= s->cow_capacity) return;
    int capacity = (needed + 511) & ~511;
    int rows = s->n_tokens - s->cow_split;
    for (int layer = 0; layer < XE_GLOBAL_LAYERS; layer++) {
        size_t elements = (size_t)XE_GLOBAL_KV_HEADS * capacity *
                          XE_GLOBAL_HEAD_DIM;
        _Float16 *next_k = xe_alloc(s->engine,
                                    elements * sizeof(*next_k),
                                    XE_MEM_SHARED);
        _Float16 *next_v = xe_alloc(s->engine,
                                    elements * sizeof(*next_v),
                                    XE_MEM_SHARED);
        if (s->cow_capacity) {
            size_t bytes = (size_t)rows * XE_GLOBAL_HEAD_DIM *
                           sizeof(*next_k);
            for (int head = 0; head < XE_GLOBAL_KV_HEADS; head++) {
                memcpy(next_k + (size_t)head * capacity *
                                XE_GLOBAL_HEAD_DIM,
                       s->cow_global_k[layer] +
                           (size_t)head * s->cow_capacity *
                           XE_GLOBAL_HEAD_DIM,
                       bytes);
                memcpy(next_v + (size_t)head * capacity *
                                XE_GLOBAL_HEAD_DIM,
                       s->cow_global_v[layer] +
                           (size_t)head * s->cow_capacity *
                           XE_GLOBAL_HEAD_DIM,
                       bytes);
            }
        }
        xe_free(s->engine, s->cow_global_k[layer], XE_MEM_SHARED);
        xe_free(s->engine, s->cow_global_v[layer], XE_MEM_SHARED);
        s->cow_global_k[layer] = next_k;
        s->cow_global_v[layer] = next_v;
    }
    s->cow_capacity = capacity;
}

uint64_t xe_session_shadow_kv_bytes(const xe_session *s) {
    if (!s || !s->cow) return 0;
    uint64_t swa = UINT64_C(2) * XE_SWA_SLAB_ELEMS * sizeof(_Float16);
    uint64_t global = UINT64_C(2) * XE_GLOBAL_LAYERS *
                      XE_GLOBAL_KV_HEADS * (uint64_t)s->cow_capacity *
                      XE_GLOBAL_HEAD_DIM * sizeof(_Float16);
    return swa + global;
}

int xe_session_shadow_split(const xe_session *s) {
    return s && s->cow ? s->cow_split : 0;
}

void xe_session_reset(xe_session *s) {
    if (!s) xe_fatal("session_reset: missing session");
    xe_require_owner(s->engine);
    s->n_tokens = 0;
}

int xe_session_position(xe_session *s) {
    if (!s) xe_fatal("session_position: missing session");
    xe_require_owner(s->engine);
    return s->n_tokens;
}

int xe_session_context_size(const xe_session *s) {
    if (!s) xe_fatal("session_context_size: missing session");
    return xe_context_size(s->engine);
}

static int xe_session_swa_can_resume(int current, int resume) {
    if (resume <= 0) return 1;
    int oldest = current > XE_SWA_WINDOW ? current - XE_SWA_WINDOW : 0;
    int needed = resume >= XE_SWA_WINDOW
                 ? resume - XE_SWA_WINDOW + 1 : 0;
    return needed >= oldest;
}

static void xe_session_extend(xe_session *s, const int32_t *tokens, int end,
                              int cpu_single) {
    int mirror_start = s->n_tokens;
    while (s->n_tokens < end) {
        int start = s->n_tokens;
        int rows = end - start;
        if (rows > 512) rows = 512;
        if (rows == 1 && cpu_single) {
            int pos = s->n_tokens;
            xe_decode_token_mode(s, tokens[pos], pos,
                                 XE_MOE_GENERIC_BATCHED, 1,
                                 XE_SOFTCAP_SECOND_LOOP, 1);
            s->tokens[pos] = tokens[pos];
            continue;
        }
        xe_prefill_batch_run(s, tokens + start, rows, start,
                             start + rows == end);
    }
    if (!s->cow) xe_session_swa_mirror(s, mirror_start, s->n_tokens);
}

void xe_session_rewind(xe_session *s, int position) {
    if (!s) xe_fatal("session_rewind: missing session");
    xe_require_owner(s->engine);
    if (position < 0 || position > s->n_tokens)
        xe_fatal("session_rewind: position %d out of range [0, %d]",
                 position, s->n_tokens);
    if (position == 0) {
        s->n_tokens = 0;
        return;
    }
    xe_tokens prefix = { s->tokens, position, s->engine->context };
    xe_session_sync(s, &prefix);
}

void xe_session_sync_report(xe_session *s, const xe_tokens *prefix,
                            xe_sync_report *report) {
    if (!s || !prefix) xe_fatal("session_sync: missing session or prefix");
    if (!xe_tokens_valid(prefix)) xe_fatal("session_sync: invalid prefix");
    xe_require_owner(s->engine);
    int n = prefix->len;
    if (n <= 0 || n > s->engine->context)
        xe_fatal("session_sync: prefix length %d out of range [1, %d]", n, s->engine->context);
    for (int i = 0; i < n; i++)
        if (prefix->v[i] < 0 || prefix->v[i] >= XE_VOCAB)
            xe_fatal("session_sync: token %d at position %d out of range [0, %d]",
                     prefix->v[i], i, XE_VOCAB - 1);

    int current = s->n_tokens;
    int common_limit = n < current ? n : current;
    int common = 0;
    while (common < common_limit && prefix->v[common] == s->tokens[common])
        common++;
    if (report) {
        report->reused = 0;
        report->prefilled = 0;
        report->restarted = 0;
    }
    if (common == n && common == current) {
        if (report) report->reused = n;
        return;
    }

    if (common < current) {
        int resume = common < n ? common : n - 1;
        if (xe_session_swa_can_resume(current, resume)) {
            s->n_tokens = resume;
        } else {
            s->n_tokens = 0;
            if (report) report->restarted = 1;
        }
    }
    if (report) {
        report->reused = s->n_tokens;
        report->prefilled = n - s->n_tokens;
    }
    xe_session_extend(s, prefix->v, n,
                      common == current && n == current + 1);
}

void xe_session_sync(xe_session *s, const xe_tokens *prefix) {
    xe_session_sync_report(s, prefix, NULL);
}

int xe_session_shadow_start(xe_session *s, const xe_tokens *prefix,
                            int max_rows) {
    if (!s || !s->cow || !prefix || !xe_tokens_valid(prefix) ||
        max_rows < 1 || max_rows > 512 || s->prefill_pending)
        return -1;
    xe_require_owner(s->engine);
    if (prefix->len < s->cow_split || prefix->len > s->engine->context)
        return -1;
    int limit = prefix->len < s->n_tokens ? prefix->len : s->n_tokens;
    int common = 0;
    while (common < limit && prefix->v[common] == s->tokens[common])
        common++;
    if (common < s->cow_split) return -1;
    if (common < s->n_tokens) {
        if (!xe_session_swa_can_resume(s->n_tokens, common)) return -1;
        s->n_tokens = common;
    }
    if (s->n_tokens >= prefix->len) return 0;
    xe_session_shadow_reserve(s, prefix->len);
    int rows = prefix->len - s->n_tokens;
    if (rows > max_rows) rows = max_rows;
    xe_prefill_batch_append_run(s, prefix->v + s->n_tokens,
                                rows, s->n_tokens);
    return rows;
}

int xe_session_shadow_poll(xe_session *s) {
    if (!s || !s->cow || !s->prefill_pending) return 0;
    xe_require_owner(s->engine);
    ze_result_t result = zeCommandListHostSynchronize(s->engine->gpu.commands,
                                                      0);
    if (result == ZE_RESULT_NOT_READY) return 0;
    xe_ze_check("zeCommandListHostSynchronize shadow poll", result);
    int rows = s->prefill_pending_rows;
    xe_prefill_batch_complete(s, 0);
    return rows;
}

int xe_session_shadow_wait(xe_session *s) {
    if (!s || !s->cow || !s->prefill_pending) return 0;
    xe_require_owner(s->engine);
    xe_ze_check("zeCommandListHostSynchronize shadow wait",
                zeCommandListHostSynchronize(s->engine->gpu.commands,
                                             UINT64_MAX));
    int rows = s->prefill_pending_rows;
    xe_prefill_batch_complete(s, 0);
    return rows;
}

int xe_session_shadow_sync(xe_session *s, const xe_tokens *prefix) {
    if (!s || !s->cow || !prefix) return -1;
    int prefilled = 0;
    int completed = xe_session_shadow_wait(s);
    if (completed > 0) prefilled += completed;
    for (;;) {
        int rows = xe_session_shadow_start(s, prefix, 512);
        if (rows < 0) return -1;
        if (rows == 0) return prefilled;
        completed = xe_session_shadow_wait(s);
        if (completed != rows) return -1;
        prefilled += completed;
    }
}

void xe_session_shadow_refresh_logits(xe_session *s) {
    if (!s || !s->cow || s->prefill_pending || s->n_tokens <= 0)
        xe_fatal("shadow logits: invalid session");
    xe_require_owner(s->engine);
    xe_workers_begin(s->engine);
    xe_output_decode(s, XE_SOFTCAP_SECOND_LOOP);
    xe_workers_end(s->engine);
}

int xe_session_shadow_promote(xe_session *source, xe_session *shadow) {
    if (!source || !shadow || source->cow || !shadow->cow ||
        source->engine != shadow->engine || shadow->prefill_pending ||
        shadow->cow_source != source || source->swa_spare_k ||
        source->swa_spare_v ||
        shadow->n_tokens < shadow->cow_split ||
        source->global_k != shadow->global_k ||
        source->global_v != shadow->global_v)
        return 0;
    xe_require_owner(source->engine);
    xe_engine *e = source->engine;
    int source_position = source->n_tokens;
    int rows = shadow->n_tokens - shadow->cow_split;
    size_t row_bytes = (size_t)rows * XE_GLOBAL_HEAD_DIM *
                       sizeof(_Float16);
    if (rows > 0) {
        for (int layer = 0; layer < XE_GLOBAL_LAYERS; layer++) {
            _Float16 *target_k = source->global_k +
                (size_t)layer * xe_global_layer_elems(e);
            _Float16 *target_v = source->global_v +
                (size_t)layer * xe_global_layer_elems(e);
            for (int head = 0; head < XE_GLOBAL_KV_HEADS; head++) {
                size_t target_offset =
                    ((size_t)head * e->context + shadow->cow_split) *
                    XE_GLOBAL_HEAD_DIM;
                size_t source_offset =
                    (size_t)head * shadow->cow_capacity * XE_GLOBAL_HEAD_DIM;
                xe_ze_check("zeCommandListAppendMemoryCopy promote K",
                            zeCommandListAppendMemoryCopy(
                                e->gpu.commands, target_k + target_offset,
                                shadow->cow_global_k[layer] + source_offset,
                                row_bytes, NULL, 0, NULL));
                xe_ze_check("zeCommandListAppendMemoryCopy promote V",
                            zeCommandListAppendMemoryCopy(
                                e->gpu.commands, target_v + target_offset,
                                shadow->cow_global_v[layer] + source_offset,
                                row_bytes, NULL, 0, NULL));
            }
        }
    }
    xe_ze_check("zeCommandListAppendMemoryCopy promote hidden",
                zeCommandListAppendMemoryCopy(
                    e->gpu.commands, source->hidden, shadow->hidden,
                    XE_EMBD * sizeof(*source->hidden), NULL, 0, NULL));
    xe_ze_check("zeCommandListAppendMemoryCopy promote logits",
                zeCommandListAppendMemoryCopy(
                    e->gpu.commands, source->logits, shadow->logits,
                    XE_VOCAB * sizeof(*source->logits), NULL, 0, NULL));
    xe_ze_check("zeCommandListHostSynchronize promote",
                zeCommandListHostSynchronize(e->gpu.commands, UINT64_MAX));
    source->swa_spare_k = source->swa_k;
    source->swa_spare_v = source->swa_v;
    source->swa_k = shadow->swa_k;
    source->swa_v = shadow->swa_v;
    shadow->swa_k = NULL;
    shadow->swa_v = NULL;
    shadow->cow_source = NULL;
    int end = source_position > shadow->n_tokens
              ? source_position : shadow->n_tokens;
    xe_session_swa_mark(source, shadow->cow_split, end);
    memcpy(source->tokens, shadow->tokens,
           (size_t)shadow->n_tokens * sizeof(*source->tokens));
    source->n_tokens = shadow->n_tokens;
    source->anchor_valid = 0;
    return 1;
}

int xe_session_common(const xe_session *s, const xe_tokens *prefix) {
    if (!s || !prefix) xe_fatal("session_common: missing session or prefix");
    if (!xe_tokens_valid(prefix)) xe_fatal("session_common: invalid prefix");
    int limit = prefix->len < s->n_tokens ? prefix->len : s->n_tokens;
    int common = 0;
    while (common < limit && prefix->v[common] == s->tokens[common])
        common++;
    return common;
}

const float *xe_session_logits(xe_session *s) {
    if (!s) xe_fatal("session_logits: missing session");
    xe_require_owner(s->engine);
    if (s->n_tokens == 0) xe_fatal("session_logits: session has no evaluated tokens");
    return s->logits;
}

int32_t xe_session_next(xe_session *s, xe_sampler *sp) {
    const float *logits = xe_session_logits(s);
    return xe_sample_logits(logits, XE_VOCAB, s->sample_candidates, sp);
}

int xe_session_anchor_capture(xe_session *s) {
    if (!s || s->n_tokens <= 0) return 0;
    xe_require_owner(s->engine);
    if (!s->anchor_swa_k) {
        s->anchor_swa_k = xe_alloc(s->engine,
            XE_SWA_SLAB_ELEMS * sizeof(*s->anchor_swa_k), XE_MEM_HOST);
        s->anchor_swa_v = xe_alloc(s->engine,
            XE_SWA_SLAB_ELEMS * sizeof(*s->anchor_swa_v), XE_MEM_HOST);
        s->anchor_logits = xe_alloc(s->engine,
            XE_VOCAB * sizeof(*s->anchor_logits), XE_MEM_HOST);
    }
    memcpy(s->anchor_swa_k, s->swa_k,
           XE_SWA_SLAB_ELEMS * sizeof(*s->swa_k));
    memcpy(s->anchor_swa_v, s->swa_v,
           XE_SWA_SLAB_ELEMS * sizeof(*s->swa_v));
    memcpy(s->anchor_logits, s->logits,
           XE_VOCAB * sizeof(*s->logits));
    s->anchor_position = s->n_tokens;
    s->anchor_valid = 1;
    return 1;
}

int xe_session_anchor_restore(xe_session *s) {
    if (!s || !s->anchor_valid) return 0;
    xe_require_owner(s->engine);
    memcpy(s->swa_k, s->anchor_swa_k,
           XE_SWA_SLAB_ELEMS * sizeof(*s->swa_k));
    memcpy(s->swa_v, s->anchor_swa_v,
           XE_SWA_SLAB_ELEMS * sizeof(*s->swa_v));
    memcpy(s->logits, s->anchor_logits,
           XE_VOCAB * sizeof(*s->logits));
    s->n_tokens = s->anchor_position;
    if (s->swa_spare_k && s->swa_spare_v)
        memset(s->swa_dirty, 1, sizeof s->swa_dirty);
    return s->anchor_position;
}

void xe_session_anchor_clear(xe_session *s) {
    if (s) s->anchor_valid = 0;
}

int xe_session_anchor_valid(const xe_session *s) {
    return s && s->anchor_valid;
}

int xe_session_token_near_top(xe_session *s, int32_t token,
                              int max_rank, float max_margin) {
    const float *logits = xe_session_logits(s);
    if (token < 0 || token >= XE_VOCAB || max_rank < 1) return 0;
    float value = logits[token];
    float best = value;
    int rank = 1;
    for (int i = 0; i < XE_VOCAB; i++) {
        if (logits[i] > best) best = logits[i];
        if (logits[i] > value) rank++;
    }
    return rank <= max_rank && best - value <= max_margin;
}

enum {
    XE_SNAPSHOT_HEADER_SIZE = 384,
    XE_SNAPSHOT_ENTRY_SIZE = 40,
    XE_SNAPSHOT_REQUIRED_SECTIONS = 6,
    XE_SNAPSHOT_MAX_SECTIONS = 64,
    XE_SNAPSHOT_BUFFER_SIZE = 8 * 1024 * 1024,
    XE_SNAPSHOT_SECTION_REQUIRED = 1
};

enum {
    XE_SNAPSHOT_FP_MODEL,
    XE_SNAPSHOT_FP_TOKENIZER,
    XE_SNAPSHOT_FP_TEMPLATE,
    XE_SNAPSHOT_FP_LAYOUT,
    XE_SNAPSHOT_FP_CONTEXT,
    XE_SNAPSHOT_FP_NUMERIC,
    XE_SNAPSHOT_FP_COUNT
};

enum {
    XE_SNAPSHOT_TOKENS = 1,
    XE_SNAPSHOT_LOGITS,
    XE_SNAPSHOT_SWA_K,
    XE_SNAPSHOT_SWA_V,
    XE_SNAPSHOT_GLOBAL_K,
    XE_SNAPSHOT_GLOBAL_V
};

typedef struct {
    uint32_t type;
    uint32_t flags;
    uint64_t offset;
    uint64_t length;
    uint64_t count;
    uint64_t crc;
} xe_snapshot_section;

typedef struct {
    uint64_t file_size;
    uint64_t position;
    uint64_t swa_rows;
    uint64_t global_rows;
    uint64_t directory_offset;
    uint64_t directory_length;
    uint64_t payload_offset;
    uint32_t section_count;
    xe_snapshot_section sections[XE_SNAPSHOT_MAX_SECTIONS];
    int known[XE_SNAPSHOT_REQUIRED_SECTIONS];
} xe_snapshot_layout;

static const uint8_t xe_snapshot_magic[8] = {
    'X', 'E', 'N', 'X', 'K', 'V', '\r', '\n'
};

static void xe_snapshot_hash_u32(format_sha256 *hash, uint32_t value) {
    uint8_t bytes[4];
    format_put_u32le(bytes, value);
    format_sha256_update(hash, bytes, sizeof bytes);
}

static void xe_snapshot_hash_u64(format_sha256 *hash, uint64_t value) {
    uint8_t bytes[8];
    format_put_u64le(bytes, value);
    format_sha256_update(hash, bytes, sizeof bytes);
}

static void xe_snapshot_hash_string(format_sha256 *hash, const char *s) {
    size_t len = strlen(s);
    xe_snapshot_hash_u64(hash, len);
    format_sha256_update(hash, s, len);
}

static void xe_snapshot_fingerprints_init(xe_engine *e) {
    if (e->snapshot_fingerprint_ready) return;
    format_sha256 hash;

    /* MODEL covers the GGUF header (every metadata key plus the full
     * tensor directory: names, shapes, types, offsets) and the file size,
     * never the tensor bytes. The v1 fingerprint hashed the whole 14 GB
     * file, lazily, on the first checkpoint or resume (55 s). Same-shape
     * requants and fine-tunes with identical metadata are deliberately
     * not detected; see NOTES "3.7 finding 1". */
    format_sha256_init(&hash);
    format_sha256_update(&hash, "xenolith-model-v2",
                         sizeof("xenolith-model-v2") - 1);
    xe_snapshot_hash_u64(&hash, e->map_len);
    xe_snapshot_hash_u64(&hash, e->header_len);
    if (e->header_len) format_sha256_update(&hash, e->map, e->header_len);
    format_sha256_final(&hash, e->snapshot_fingerprint[XE_SNAPSHOT_FP_MODEL]);

    format_sha256_init(&hash);
    format_sha256_update(&hash, "xenolith-tokenizer-v1",
                         sizeof("xenolith-tokenizer-v1") - 1);
    xe_snapshot_hash_u32(&hash, (uint32_t)e->bos_id);
    xe_snapshot_hash_u32(&hash, (uint32_t)e->eos_id);
    xe_snapshot_hash_u32(&hash, (uint32_t)e->eot_id);
    xe_snapshot_hash_u32(&hash, (uint32_t)e->unk_id);
    xe_snapshot_hash_u32(&hash, (uint32_t)e->pad_id);
    xe_snapshot_hash_u64(&hash, e->tok_tokens_count);
    for (uint64_t i = 0; i < e->tok_tokens_count; i++) {
        const xe_str *piece = &e->tok_piece[i];
        xe_snapshot_hash_u64(&hash, piece->len);
        format_sha256_update(&hash, piece->p, (size_t)piece->len);
    }
    xe_snapshot_hash_u64(&hash, e->tok_scores_count);
    if (e->tok_scores_count)
        format_sha256_update(&hash, e->tok_scores,
                             (size_t)e->tok_scores_count * sizeof(float));
    xe_snapshot_hash_u64(&hash, e->tok_token_type_count);
    if (e->tok_token_type_count)
        format_sha256_update(&hash, e->tok_token_type,
                             (size_t)e->tok_token_type_count * sizeof(int32_t));
    xe_snapshot_hash_u64(&hash, e->tok_merges_count);
    if (e->tok_merges_count) {
        xe_cur cursor = xe_cur_at(e, e->tok_merges);
        for (uint64_t i = 0; i < e->tok_merges_count; i++) {
            xe_str merge = xe_read_str(&cursor);
            xe_snapshot_hash_u64(&hash, merge.len);
            format_sha256_update(&hash, merge.p, (size_t)merge.len);
        }
    }
    format_sha256_final(&hash,
                        e->snapshot_fingerprint[XE_SNAPSHOT_FP_TOKENIZER]);

    format_sha256_init(&hash);
    format_sha256_update(&hash, "xenolith-template-v1",
                         sizeof("xenolith-template-v1") - 1);
    xe_snapshot_hash_u64(&hash, e->chat_template.len);
    if (e->chat_template.len)
        format_sha256_update(&hash, e->chat_template.p,
                             (size_t)e->chat_template.len);
    xe_snapshot_hash_string(&hash, "xe_profile_render/v1");
    xe_snapshot_hash_u32(&hash, XE_CHANNEL_BEGIN_ID);
    xe_snapshot_hash_u32(&hash, XE_CHANNEL_END_ID);
    xe_snapshot_hash_u32(&hash, XE_TURN_BEGIN_ID);
    xe_snapshot_hash_u32(&hash, XE_TURN_END_ID);
    format_sha256_final(&hash,
                        e->snapshot_fingerprint[XE_SNAPSHOT_FP_TEMPLATE]);

    format_sha256_init(&hash);
    format_sha256_update(&hash, "xenolith-kv-layout-v1",
                         sizeof("xenolith-kv-layout-v1") - 1);
    xe_snapshot_hash_u32(&hash, XE_LAYERS);
    xe_snapshot_hash_u32(&hash, XE_SWA_WINDOW);
    xe_snapshot_hash_u32(&hash, XE_SWA_KV_HEADS);
    xe_snapshot_hash_u32(&hash, XE_SWA_HEAD_DIM);
    xe_snapshot_hash_u32(&hash, XE_GLOBAL_KV_HEADS);
    xe_snapshot_hash_u32(&hash, XE_GLOBAL_HEAD_DIM);
    for (int layer = 0; layer < XE_LAYERS; layer++)
        xe_snapshot_hash_u32(&hash, XE_IS_GLOBAL(layer));
    xe_snapshot_hash_u32(&hash, sizeof(_Float16));
    format_sha256_final(&hash,
                        e->snapshot_fingerprint[XE_SNAPSHOT_FP_LAYOUT]);

    format_sha256_init(&hash);
    format_sha256_update(&hash, "xenolith-context-v1",
                         sizeof("xenolith-context-v1") - 1);
    xe_snapshot_hash_u32(&hash, e->context);
    xe_snapshot_hash_u32(&hash, XE_EMBD);
    xe_snapshot_hash_u32(&hash, XE_VOCAB);
    xe_snapshot_hash_u32(&hash, XE_Q_HEADS);
    xe_snapshot_hash_u32(&hash, XE_EXPERTS);
    xe_snapshot_hash_u32(&hash, XE_EXPERTS_USED);
    xe_snapshot_hash_u32(&hash, XE_DENSE_FFN);
    xe_snapshot_hash_u32(&hash, XE_EXPERT_FFN);
    uint32_t bits;
    memcpy(&bits, &(float){XE_SWA_ROPE_BASE}, sizeof bits);
    xe_snapshot_hash_u32(&hash, bits);
    memcpy(&bits, &(float){XE_GLOBAL_ROPE_BASE}, sizeof bits);
    xe_snapshot_hash_u32(&hash, bits);
    memcpy(&bits, &(float){XE_RMS_EPS}, sizeof bits);
    xe_snapshot_hash_u32(&hash, bits);
    memcpy(&bits, &(float){XE_LOGIT_SOFTCAP}, sizeof bits);
    xe_snapshot_hash_u32(&hash, bits);
    format_sha256_final(&hash,
                        e->snapshot_fingerprint[XE_SNAPSHOT_FP_CONTEXT]);

    format_sha256_init(&hash);
    format_sha256_update(&hash, "xenolith-numeric-v1",
                         sizeof("xenolith-numeric-v1") - 1);
    xe_snapshot_hash_u32(&hash, 1);
    xe_snapshot_hash_string(&hash, XE_BUILD_COMMIT);
    xe_snapshot_hash_string(&hash, XE_NUMERIC_SOURCE_HASH);
    xe_snapshot_hash_string(&hash, XE_NUMERIC_CFLAGS_HASH);
    xe_snapshot_hash_string(&hash, __VERSION__);
    xe_snapshot_hash_u32(&hash, XE_WORKERS);
    xe_snapshot_hash_u32(&hash, e->scalar_rms);
    size_t spv_size = (size_t)(_binary_xenolith_gpu_spv_end -
                               _binary_xenolith_gpu_spv_start);
    xe_snapshot_hash_u64(&hash, spv_size);
    format_sha256_update(&hash, _binary_xenolith_gpu_spv_start, spv_size);
    format_sha256_final(&hash,
                        e->snapshot_fingerprint[XE_SNAPSHOT_FP_NUMERIC]);
    e->snapshot_fingerprint_ready = 1;
}

static void xe_snapshot_boundary_hash(const int32_t *tokens, int n,
                                      uint8_t out[32]) {
    format_sha256 hash;
    format_sha256_init(&hash);
    format_sha256_update(&hash, "xenolith-boundary-v1",
                         sizeof("xenolith-boundary-v1") - 1);
    xe_snapshot_hash_u64(&hash, (uint64_t)n);
    for (int i = 0; i < n; i++) xe_snapshot_hash_u32(&hash, (uint32_t)tokens[i]);
    format_sha256_final(&hash, out);
}

static int xe_snapshot_expected_valid(const xe_tokens *tokens) {
    if (!xe_tokens_valid(tokens) || tokens->len <= 0) return 0;
    for (int i = 0; i < tokens->len; i++)
        if (tokens->v[i] < 0 || tokens->v[i] >= XE_VOCAB) return 0;
    return 1;
}

static int xe_snapshot_layout_build(int capacity, uint64_t position,
                                    xe_snapshot_layout *layout) {
    if (!position || position > (uint64_t)capacity) return 0;
    memset(layout, 0, sizeof *layout);
    for (int i = 0; i < XE_SNAPSHOT_REQUIRED_SECTIONS; i++)
        layout->known[i] = i;
    layout->section_count = XE_SNAPSHOT_REQUIRED_SECTIONS;
    layout->position = position;
    layout->swa_rows = position < XE_SWA_WINDOW ? position : XE_SWA_WINDOW;
    layout->global_rows = position;
    layout->directory_offset = XE_SNAPSHOT_HEADER_SIZE;
    layout->directory_length = (uint64_t)XE_SNAPSHOT_REQUIRED_SECTIONS *
                               XE_SNAPSHOT_ENTRY_SIZE;
    uint64_t end;
    if (!format_add_u64(layout->directory_offset, layout->directory_length,
                        &end) ||
        !format_align_u64(end, 64, &layout->payload_offset)) return 0;

    uint64_t lengths[XE_SNAPSHOT_REQUIRED_SECTIONS];
    uint64_t counts[XE_SNAPSHOT_REQUIRED_SECTIONS];
    counts[0] = position;
    counts[1] = XE_VOCAB;
    if (!format_mul_u64(position, 4, &lengths[0]) ||
        !format_mul_u64(XE_VOCAB, 4, &lengths[1]) ||
        !format_mul_u64(25 * XE_SWA_KV_HEADS * XE_SWA_HEAD_DIM,
                        layout->swa_rows, &counts[2]) ||
        !format_mul_u64(counts[2], sizeof(_Float16), &lengths[2]) ||
        !format_mul_u64(5 * XE_GLOBAL_KV_HEADS * XE_GLOBAL_HEAD_DIM,
                        position, &counts[4]) ||
        !format_mul_u64(counts[4], sizeof(_Float16), &lengths[4])) return 0;
    counts[3] = counts[2];
    lengths[3] = lengths[2];
    counts[5] = counts[4];
    lengths[5] = lengths[4];

    uint64_t offset = layout->payload_offset;
    for (int i = 0; i < XE_SNAPSHOT_REQUIRED_SECTIONS; i++) {
        if (!format_align_u64(offset, 64, &offset)) return 0;
        layout->sections[i].type = (uint32_t)i + 1;
        layout->sections[i].flags = XE_SNAPSHOT_SECTION_REQUIRED;
        layout->sections[i].offset = offset;
        layout->sections[i].length = lengths[i];
        layout->sections[i].count = counts[i];
        if (!format_add_u64(offset, lengths[i], &offset)) return 0;
    }
    layout->file_size = offset;
    return 1;
}

xe_snapshot_status xe_session_snapshot_size(xe_session *s, uint64_t *size) {
    if (!s || !size) return XE_SNAPSHOT_INVALID_ARGUMENT;
    xe_require_owner(s->engine);
    if (!s->n_tokens) return XE_SNAPSHOT_EMPTY;
    xe_snapshot_layout layout;
    if (!xe_snapshot_layout_build(s->engine->context, (uint64_t)s->n_tokens, &layout))
        return XE_SNAPSHOT_FORMAT;
    *size = layout.file_size;
    return XE_SNAPSHOT_OK;
}

static xe_snapshot_status xe_snapshot_gpu_sync(xe_engine *e) {
    if (!e->gpu.commands) return XE_SNAPSHOT_OK;
    ze_result_t result = zeCommandListHostSynchronize(e->gpu.commands,
                                                       UINT64_MAX);
    return result == ZE_RESULT_SUCCESS ? XE_SNAPSHOT_OK : XE_SNAPSHOT_BACKEND;
}

static int xe_snapshot_stream_write(FILE *out, const void *data,
                                    uint64_t length, uint64_t *crc) {
    const uint8_t *p = data;
    while (length) {
        size_t chunk = length > XE_SNAPSHOT_BUFFER_SIZE
                       ? XE_SNAPSHOT_BUFFER_SIZE : (size_t)length;
        if (!format_write(out, p, chunk)) return 0;
        *crc = format_crc64(*crc, p, chunk);
        p += chunk;
        length -= chunk;
    }
    return 1;
}

static int xe_snapshot_write_tokens(FILE *out, const int32_t *tokens, int n,
                                    uint8_t *buffer, uint64_t *crc) {
    int at = 0;
    int cap = XE_SNAPSHOT_BUFFER_SIZE / 4;
    while (at < n) {
        int count = n - at < cap ? n - at : cap;
        for (int i = 0; i < count; i++)
            format_put_u32le(buffer + 4 * i, (uint32_t)tokens[at + i]);
        size_t bytes = (size_t)count * 4;
        if (!format_write(out, buffer, bytes)) return 0;
        *crc = format_crc64(*crc, buffer, bytes);
        at += count;
    }
    return 1;
}

static int xe_snapshot_write_kv(FILE *out, xe_session *s, int global,
                                int value, uint64_t rows, uint64_t *crc) {
    int heads = global ? XE_GLOBAL_KV_HEADS : XE_SWA_KV_HEADS;
    int dimension = global ? XE_GLOBAL_HEAD_DIM : XE_SWA_HEAD_DIM;
    int capacity = global ? s->engine->context : XE_SWA_WINDOW;
    int first = global ? 0 : s->n_tokens - (int)rows;
    for (int layer = 0; layer < XE_LAYERS; layer++) {
        if (XE_IS_GLOBAL(layer) != global) continue;
        _Float16 *base = xe_kv_layer_ptr(s, layer, value);
        for (int head = 0; head < heads; head++) {
            _Float16 *head_base = base + (size_t)head * capacity * dimension;
            if (global) {
                uint64_t bytes = rows * (uint64_t)dimension * sizeof(_Float16);
                if (!xe_snapshot_stream_write(out, head_base, bytes, crc))
                    return 0;
            } else {
                int slot = first & (XE_SWA_WINDOW - 1);
                uint64_t first_rows = rows;
                if (first_rows > (uint64_t)(capacity - slot))
                    first_rows = (uint64_t)(capacity - slot);
                uint64_t first_bytes = first_rows * (uint64_t)dimension *
                                       sizeof(_Float16);
                if (!xe_snapshot_stream_write(out,
                        head_base + (size_t)slot * dimension,
                        first_bytes, crc)) return 0;
                uint64_t second_rows = rows - first_rows;
                uint64_t second_bytes = second_rows * (uint64_t)dimension *
                                        sizeof(_Float16);
                if (second_bytes && !xe_snapshot_stream_write(out, head_base,
                                                               second_bytes,
                                                               crc)) return 0;
            }
        }
    }
    return 1;
}

static void xe_snapshot_encode_directory(const xe_snapshot_layout *layout,
                                         uint8_t *directory) {
    for (uint32_t i = 0; i < layout->section_count; i++) {
        uint8_t *entry = directory + (size_t)i * XE_SNAPSHOT_ENTRY_SIZE;
        const xe_snapshot_section *section = &layout->sections[i];
        format_put_u32le(entry, section->type);
        format_put_u32le(entry + 4, section->flags);
        format_put_u64le(entry + 8, section->offset);
        format_put_u64le(entry + 16, section->length);
        format_put_u64le(entry + 24, section->count);
        format_put_u64le(entry + 32, section->crc);
    }
}

static void xe_snapshot_encode_header(xe_engine *e,
                                      const xe_snapshot_layout *layout,
                                      const uint8_t boundary[32],
                                      const uint8_t *directory,
                                      uint8_t header[XE_SNAPSHOT_HEADER_SIZE]) {
    memset(header, 0, XE_SNAPSHOT_HEADER_SIZE);
    memcpy(header, xe_snapshot_magic, sizeof xe_snapshot_magic);
    format_put_u16le(header + 8, 1);
    format_put_u16le(header + 10, 0);
    format_put_u32le(header + 12, XE_SNAPSHOT_HEADER_SIZE);
    format_put_u32le(header + 16, XE_SNAPSHOT_ENTRY_SIZE);
    format_put_u32le(header + 20, layout->section_count);
    format_put_u64le(header + 32, layout->file_size);
    format_put_u64le(header + 40, layout->position);
    format_put_u64le(header + 48, layout->swa_rows);
    format_put_u64le(header + 56, layout->global_rows);
    format_put_u64le(header + 64, layout->directory_offset);
    format_put_u64le(header + 72, layout->directory_length);
    format_put_u64le(header + 80, layout->payload_offset);
    format_put_u64le(header + 88,
                     format_crc64(0, directory,
                                  (size_t)layout->directory_length));
    for (int i = 0; i < XE_SNAPSHOT_FP_COUNT; i++)
        memcpy(header + 96 + 32 * i, e->snapshot_fingerprint[i], 32);
    memcpy(header + 288, boundary, 32);
    format_put_u64le(header + 376, format_crc64(0, header, 376));
}

xe_snapshot_status xe_session_snapshot_save(xe_session *s, FILE *out) {
    if (!s || !out) return XE_SNAPSHOT_INVALID_ARGUMENT;
    xe_require_owner(s->engine);
    if (!s->n_tokens) return XE_SNAPSHOT_EMPTY;
    xe_snapshot_status status = xe_snapshot_gpu_sync(s->engine);
    if (status != XE_SNAPSHOT_OK) return status;
    xe_snapshot_fingerprints_init(s->engine);

    xe_snapshot_layout layout;
    if (!xe_snapshot_layout_build(s->engine->context, (uint64_t)s->n_tokens, &layout))
        return XE_SNAPSHOT_FORMAT;
    uint8_t *buffer = malloc(XE_SNAPSHOT_BUFFER_SIZE);
    if (!buffer) return XE_SNAPSHOT_NOMEM;
    uint8_t directory[XE_SNAPSHOT_REQUIRED_SECTIONS * XE_SNAPSHOT_ENTRY_SIZE];
    uint8_t header[XE_SNAPSHOT_HEADER_SIZE];
    uint8_t boundary[32];
    memset(buffer, 0, (size_t)layout.payload_offset);
    if (!format_seek(out, 0) ||
        !format_write(out, buffer, (size_t)layout.payload_offset)) {
        free(buffer);
        return XE_SNAPSHOT_IO;
    }

    for (int i = 0; i < XE_SNAPSHOT_REQUIRED_SECTIONS; i++) {
        xe_snapshot_section *section = &layout.sections[i];
        if (!format_seek(out, section->offset)) {
            free(buffer);
            return XE_SNAPSHOT_IO;
        }
        uint64_t crc = 0;
        int ok = 0;
        if (section->type == XE_SNAPSHOT_TOKENS)
            ok = xe_snapshot_write_tokens(out, s->tokens, s->n_tokens,
                                          buffer, &crc);
        else if (section->type == XE_SNAPSHOT_LOGITS)
            ok = xe_snapshot_stream_write(out, s->logits, section->length,
                                          &crc);
        else if (section->type == XE_SNAPSHOT_SWA_K)
            ok = xe_snapshot_write_kv(out, s, 0, 0, layout.swa_rows, &crc);
        else if (section->type == XE_SNAPSHOT_SWA_V)
            ok = xe_snapshot_write_kv(out, s, 0, 1, layout.swa_rows, &crc);
        else if (section->type == XE_SNAPSHOT_GLOBAL_K)
            ok = xe_snapshot_write_kv(out, s, 1, 0, layout.global_rows, &crc);
        else if (section->type == XE_SNAPSHOT_GLOBAL_V)
            ok = xe_snapshot_write_kv(out, s, 1, 1, layout.global_rows, &crc);
        if (!ok) {
            free(buffer);
            return XE_SNAPSHOT_IO;
        }
        section->crc = crc;
    }

    xe_snapshot_boundary_hash(s->tokens, s->n_tokens, boundary);
    xe_snapshot_encode_directory(&layout, directory);
    xe_snapshot_encode_header(s->engine, &layout, boundary, directory, header);
    int ok = format_seek(out, 0) &&
             format_write(out, header, sizeof header) &&
             format_write(out, directory, sizeof directory) &&
             fflush(out) == 0 && ftruncate(fileno(out), (off_t)layout.file_size) == 0;
    free(buffer);
    return ok ? XE_SNAPSHOT_OK : XE_SNAPSHOT_IO;
}

static xe_snapshot_status xe_snapshot_fingerprint_status(int index) {
    static const xe_snapshot_status status[XE_SNAPSHOT_FP_COUNT] = {
        XE_SNAPSHOT_MODEL_MISMATCH,
        XE_SNAPSHOT_TOKENIZER_MISMATCH,
        XE_SNAPSHOT_TEMPLATE_MISMATCH,
        XE_SNAPSHOT_LAYOUT_MISMATCH,
        XE_SNAPSHOT_CONTEXT_MISMATCH,
        XE_SNAPSHOT_NUMERIC_MISMATCH
    };
    return status[index];
}

static int xe_snapshot_all_zero(const uint8_t *p, size_t n) {
    for (size_t i = 0; i < n; i++) if (p[i]) return 0;
    return 1;
}

static xe_snapshot_status xe_snapshot_parse_header(
        xe_engine *e, FILE *in, const xe_tokens *expected,
        xe_snapshot_layout *layout,
        uint8_t saved_header[XE_SNAPSHOT_HEADER_SIZE],
        uint8_t saved_directory[XE_SNAPSHOT_MAX_SECTIONS *
                                XE_SNAPSHOT_ENTRY_SIZE]) {
    uint64_t actual_size;
    if (!format_file_size(in, &actual_size) || !format_seek(in, 0) ||
        !format_read(in, saved_header, XE_SNAPSHOT_HEADER_SIZE))
        return XE_SNAPSHOT_IO;
    if (memcmp(saved_header, xe_snapshot_magic, sizeof xe_snapshot_magic) != 0)
        return XE_SNAPSHOT_FORMAT;
    if (format_get_u16le(saved_header + 8) != 1)
        return XE_SNAPSHOT_VERSION;
    if (format_get_u32le(saved_header + 12) != XE_SNAPSHOT_HEADER_SIZE ||
        format_get_u32le(saved_header + 16) != XE_SNAPSHOT_ENTRY_SIZE)
        return XE_SNAPSHOT_VERSION;
    if (format_get_u32le(saved_header + 24) != 0 ||
        format_get_u32le(saved_header + 28) != 0 ||
        !xe_snapshot_all_zero(saved_header + 320, 56))
        return XE_SNAPSHOT_FORMAT;
    if (format_get_u64le(saved_header + 376) !=
        format_crc64(0, saved_header, 376)) return XE_SNAPSHOT_FORMAT;

    memset(layout, 0, sizeof *layout);
    for (int i = 0; i < XE_SNAPSHOT_REQUIRED_SECTIONS; i++)
        layout->known[i] = -1;
    layout->section_count = format_get_u32le(saved_header + 20);
    layout->file_size = format_get_u64le(saved_header + 32);
    layout->position = format_get_u64le(saved_header + 40);
    layout->swa_rows = format_get_u64le(saved_header + 48);
    layout->global_rows = format_get_u64le(saved_header + 56);
    layout->directory_offset = format_get_u64le(saved_header + 64);
    layout->directory_length = format_get_u64le(saved_header + 72);
    layout->payload_offset = format_get_u64le(saved_header + 80);
    if (layout->position != (uint64_t)expected->len)
        return XE_SNAPSHOT_TOKEN_MISMATCH;
    if (layout->section_count < XE_SNAPSHOT_REQUIRED_SECTIONS ||
        layout->section_count > XE_SNAPSHOT_MAX_SECTIONS ||
        layout->file_size != actual_size ||
        layout->global_rows != layout->position ||
        layout->swa_rows != (layout->position < XE_SWA_WINDOW
                            ? layout->position : XE_SWA_WINDOW) ||
        layout->directory_offset != XE_SNAPSHOT_HEADER_SIZE ||
        !format_mul_u64(layout->section_count, XE_SNAPSHOT_ENTRY_SIZE,
                        &layout->directory_length) ||
        format_get_u64le(saved_header + 72) != layout->directory_length)
        return XE_SNAPSHOT_FORMAT;
    uint64_t directory_end;
    if (!format_add_u64(layout->directory_offset, layout->directory_length,
                        &directory_end) ||
        layout->payload_offset < directory_end ||
        (layout->payload_offset & 63) || layout->payload_offset > actual_size)
        return XE_SNAPSHOT_FORMAT;
    if (!format_seek(in, layout->directory_offset) ||
        !format_read(in, saved_directory, (size_t)layout->directory_length))
        return XE_SNAPSHOT_IO;
    if (format_get_u64le(saved_header + 88) !=
        format_crc64(0, saved_directory, (size_t)layout->directory_length))
        return XE_SNAPSHOT_FORMAT;

    for (uint32_t i = 0; i < layout->section_count; i++) {
        const uint8_t *entry = saved_directory +
                               (size_t)i * XE_SNAPSHOT_ENTRY_SIZE;
        xe_snapshot_section *section = &layout->sections[i];
        section->type = format_get_u32le(entry);
        section->flags = format_get_u32le(entry + 4);
        section->offset = format_get_u64le(entry + 8);
        section->length = format_get_u64le(entry + 16);
        section->count = format_get_u64le(entry + 24);
        section->crc = format_get_u64le(entry + 32);
        uint64_t section_end;
        if (!section->type || (section->flags & ~XE_SNAPSHOT_SECTION_REQUIRED) ||
            section->offset < layout->payload_offset ||
            !format_add_u64(section->offset, section->length, &section_end) ||
            section_end > layout->file_size) return XE_SNAPSHOT_FORMAT;
        for (uint32_t j = 0; j < i; j++) {
            const xe_snapshot_section *other = &layout->sections[j];
            uint64_t other_end = other->offset + other->length;
            if (section->type == other->type ||
                (section->offset < other_end && other->offset < section_end))
                return XE_SNAPSHOT_FORMAT;
        }
        if (section->type <= XE_SNAPSHOT_REQUIRED_SECTIONS) {
            if (section->flags != XE_SNAPSHOT_SECTION_REQUIRED)
                return XE_SNAPSHOT_FORMAT;
            layout->known[section->type - 1] = (int)i;
        } else if (section->flags & XE_SNAPSHOT_SECTION_REQUIRED) {
            return XE_SNAPSHOT_VERSION;
        }
    }

    if (memcmp(saved_header + 96 + 32 * XE_SNAPSHOT_FP_CONTEXT,
               e->snapshot_fingerprint[XE_SNAPSHOT_FP_CONTEXT], 32))
        return XE_SNAPSHOT_CONTEXT_MISMATCH;
    xe_snapshot_layout expected_layout;
    if (!xe_snapshot_layout_build(e->context, layout->position, &expected_layout))
        return XE_SNAPSHOT_FORMAT;
    for (int type = 0; type < XE_SNAPSHOT_REQUIRED_SECTIONS; type++) {
        int index = layout->known[type];
        if (index < 0) return XE_SNAPSHOT_FORMAT;
        const xe_snapshot_section *got = &layout->sections[index];
        const xe_snapshot_section *want = &expected_layout.sections[type];
        if (got->length != want->length || got->count != want->count)
            return XE_SNAPSHOT_FORMAT;
    }

    for (int i = 0; i < XE_SNAPSHOT_FP_COUNT; i++)
        if (memcmp(saved_header + 96 + 32 * i,
                   e->snapshot_fingerprint[i], 32) != 0)
            return xe_snapshot_fingerprint_status(i);
    uint8_t boundary[32];
    xe_snapshot_boundary_hash(expected->v, expected->len, boundary);
    if (memcmp(saved_header + 288, boundary, 32) != 0)
        return XE_SNAPSHOT_TOKEN_MISMATCH;
    return XE_SNAPSHOT_OK;
}

static xe_snapshot_status xe_snapshot_verify_section(
        FILE *in, const xe_snapshot_section *section, uint8_t *buffer,
        const xe_tokens *expected) {
    if (!format_seek(in, section->offset)) return XE_SNAPSHOT_IO;
    uint64_t remaining = section->length;
    uint64_t crc = 0;
    int token_at = 0;
    while (remaining) {
        size_t chunk = remaining > XE_SNAPSHOT_BUFFER_SIZE
                       ? XE_SNAPSHOT_BUFFER_SIZE : (size_t)remaining;
        if (!format_read(in, buffer, chunk)) return XE_SNAPSHOT_IO;
        crc = format_crc64(crc, buffer, chunk);
        if (section->type == XE_SNAPSHOT_TOKENS) {
            if (chunk & 3) return XE_SNAPSHOT_FORMAT;
            for (size_t i = 0; i < chunk; i += 4, token_at++)
                if (token_at >= expected->len ||
                    format_get_u32le(buffer + i) !=
                    (uint32_t)expected->v[token_at])
                    return XE_SNAPSHOT_TOKEN_MISMATCH;
        }
        remaining -= chunk;
    }
    if (section->type == XE_SNAPSHOT_TOKENS && token_at != expected->len)
        return XE_SNAPSHOT_TOKEN_MISMATCH;
    return crc == section->crc ? XE_SNAPSHOT_OK : XE_SNAPSHOT_FORMAT;
}

static xe_snapshot_status xe_snapshot_read_raw(
        FILE *in, const xe_snapshot_section *section, void *destination) {
    if (!format_seek(in, section->offset)) return XE_SNAPSHOT_IO;
    uint8_t *p = destination;
    uint64_t remaining = section->length;
    uint64_t crc = 0;
    while (remaining) {
        size_t chunk = remaining > XE_SNAPSHOT_BUFFER_SIZE
                       ? XE_SNAPSHOT_BUFFER_SIZE : (size_t)remaining;
        if (!format_read(in, p, chunk)) return XE_SNAPSHOT_IO;
        crc = format_crc64(crc, p, chunk);
        p += chunk;
        remaining -= chunk;
    }
    return crc == section->crc ? XE_SNAPSHOT_OK : XE_SNAPSHOT_FORMAT;
}

static int xe_snapshot_stream_read(FILE *in, void *data, uint64_t length,
                                   uint64_t *crc) {
    uint8_t *p = data;
    while (length) {
        size_t chunk = length > XE_SNAPSHOT_BUFFER_SIZE
                       ? XE_SNAPSHOT_BUFFER_SIZE : (size_t)length;
        if (!format_read(in, p, chunk)) return 0;
        *crc = format_crc64(*crc, p, chunk);
        p += chunk;
        length -= chunk;
    }
    return 1;
}

static xe_snapshot_status xe_snapshot_read_kv(
        FILE *in, const xe_snapshot_section *section, xe_session *s,
        int global, int value, uint64_t rows, uint64_t position) {
    if (!format_seek(in, section->offset)) return XE_SNAPSHOT_IO;
    int heads = global ? XE_GLOBAL_KV_HEADS : XE_SWA_KV_HEADS;
    int dimension = global ? XE_GLOBAL_HEAD_DIM : XE_SWA_HEAD_DIM;
    int capacity = global ? s->engine->context : XE_SWA_WINDOW;
    int logical_first = global ? 0 : (int)(position - rows);
    uint64_t crc = 0;
    for (int layer = 0; layer < XE_LAYERS; layer++) {
        if (XE_IS_GLOBAL(layer) != global) continue;
        _Float16 *base = xe_kv_layer_ptr(s, layer, value);
        for (int head = 0; head < heads; head++) {
            _Float16 *head_base = base + (size_t)head * capacity * dimension;
            if (global) {
                uint64_t bytes = rows * (uint64_t)dimension * sizeof(_Float16);
                if (!xe_snapshot_stream_read(in, head_base, bytes, &crc))
                    return XE_SNAPSHOT_IO;
            } else {
                int slot = logical_first & (XE_SWA_WINDOW - 1);
                uint64_t first_rows = rows;
                if (first_rows > (uint64_t)(capacity - slot))
                    first_rows = (uint64_t)(capacity - slot);
                uint64_t first_bytes = first_rows * (uint64_t)dimension *
                                       sizeof(_Float16);
                if (!xe_snapshot_stream_read(in,
                        head_base + (size_t)slot * dimension,
                        first_bytes, &crc)) return XE_SNAPSHOT_IO;
                uint64_t second_rows = rows - first_rows;
                uint64_t second_bytes = second_rows * (uint64_t)dimension *
                                        sizeof(_Float16);
                if (second_bytes && !xe_snapshot_stream_read(in, head_base,
                                                              second_bytes,
                                                              &crc))
                    return XE_SNAPSHOT_IO;
            }
        }
    }
    return crc == section->crc ? XE_SNAPSHOT_OK : XE_SNAPSHOT_FORMAT;
}

xe_snapshot_status xe_session_snapshot_load(xe_session *s, FILE *in,
                                            const xe_tokens *expected) {
    if (!s || !in || !expected) return XE_SNAPSHOT_INVALID_ARGUMENT;
    xe_require_owner(s->engine);
    if (!xe_snapshot_expected_valid(expected) || expected->len > s->engine->context)
        return expected->len == 0 ? XE_SNAPSHOT_EMPTY
                                  : XE_SNAPSHOT_INVALID_ARGUMENT;
    xe_snapshot_fingerprints_init(s->engine);
    uint8_t header[XE_SNAPSHOT_HEADER_SIZE];
    uint8_t directory[XE_SNAPSHOT_MAX_SECTIONS * XE_SNAPSHOT_ENTRY_SIZE];
    xe_snapshot_layout layout;
    xe_snapshot_status status = xe_snapshot_parse_header(
        s->engine, in, expected, &layout, header, directory);
    if (status != XE_SNAPSHOT_OK) return status;
    uint8_t *buffer = malloc(XE_SNAPSHOT_BUFFER_SIZE);
    if (!buffer) return XE_SNAPSHOT_NOMEM;
    status = xe_snapshot_verify_section(
        in, &layout.sections[layout.known[XE_SNAPSHOT_TOKENS - 1]], buffer,
        expected);
    if (status != XE_SNAPSHOT_OK) {
        free(buffer);
        return status;
    }
    for (uint32_t i = 0; i < layout.section_count; i++) {
        if (layout.sections[i].type <= XE_SNAPSHOT_REQUIRED_SECTIONS) continue;
        status = xe_snapshot_verify_section(in, &layout.sections[i], buffer,
                                            expected);
        if (status != XE_SNAPSHOT_OK) {
            free(buffer);
            return status;
        }
    }
    free(buffer);
    status = xe_snapshot_gpu_sync(s->engine);
    if (status != XE_SNAPSHOT_OK) return status;
    s->n_tokens = 0;
    status = xe_snapshot_read_raw(
        in, &layout.sections[layout.known[XE_SNAPSHOT_LOGITS - 1]], s->logits);
    if (status == XE_SNAPSHOT_OK)
        status = xe_snapshot_read_kv(
            in, &layout.sections[layout.known[XE_SNAPSHOT_SWA_K - 1]], s,
            0, 0, layout.swa_rows, layout.position);
    if (status == XE_SNAPSHOT_OK)
        status = xe_snapshot_read_kv(
            in, &layout.sections[layout.known[XE_SNAPSHOT_SWA_V - 1]], s,
            0, 1, layout.swa_rows, layout.position);
    if (status == XE_SNAPSHOT_OK)
        status = xe_snapshot_read_kv(
            in, &layout.sections[layout.known[XE_SNAPSHOT_GLOBAL_K - 1]], s,
            1, 0, layout.global_rows, layout.position);
    if (status == XE_SNAPSHOT_OK)
        status = xe_snapshot_read_kv(
            in, &layout.sections[layout.known[XE_SNAPSHOT_GLOBAL_V - 1]], s,
            1, 1, layout.global_rows, layout.position);
    if (status != XE_SNAPSHOT_OK) return status;
    memcpy(s->tokens, expected->v,
           (size_t)expected->len * sizeof(*s->tokens));
    s->n_tokens = expected->len;
    if (s->swa_spare_k && s->swa_spare_v)
        memset(s->swa_dirty, 1, sizeof s->swa_dirty);
    return XE_SNAPSHOT_OK;
}

const char *xe_snapshot_status_name(xe_snapshot_status status) {
    static const char *names[] = {
        "ok", "invalid-argument", "empty", "io", "format", "version",
        "model-mismatch", "tokenizer-mismatch", "template-mismatch",
        "layout-mismatch", "context-mismatch", "numeric-mismatch",
        "token-mismatch", "backend", "no-memory", "changed"
    };
    return (unsigned)status < sizeof names / sizeof names[0]
           ? names[status] : "unknown";
}

static const float ref_kq_scale = 1.0f;
static const float ref_moe_weight_sum_min = 6.103515625e-5f;
static const float ref_gelu_coef_a = 0.044715f;
static const float ref_gelu_sqrt_2_over_pi = 0.79788456080286535587989211986876f;

static float ref_fp16_to_fp32(uint16_t bits) {
    uint32_t sign = (uint32_t)(bits >> 15) & 1u;
    uint32_t exponent = (uint32_t)(bits >> 10) & 0x1Fu;
    uint32_t mantissa = (uint32_t)bits & 0x3FFu;
    float magnitude;

    if (exponent == 0)
        magnitude = ldexpf((float)mantissa, -24);
    else if (exponent == 31)
        magnitude = mantissa ? NAN : INFINITY;
    else
        magnitude = ldexpf((float)(mantissa + 1024), (int)exponent - 25);

    return sign ? -magnitude : magnitude;
}

static void ref_dequant_block(const xe_q4 *w, uint64_t row, int blocks,
                              int block, float *out) {
    uint8_t nibbles[16];
    xe_q4_copy_block(w, row, blocks, block, nibbles);
    float scale = ref_fp16_to_fp32(xe_q4_scale(w, row, blocks, block));

    for (int j = 0; j < 16; j++) {
        int low = nibbles[j] & 0x0F;
        int high = (nibbles[j] >> 4) & 0x0F;
        out[j] = (float)(low - 8) * scale;
        out[j + 16] = (float)(high - 8) * scale;
    }
}

static void ref_dequant_row(const xe_q4 *w, uint64_t row, int n_in, float *out) {
    int n_blocks = n_in / 32;

    for (int b = 0; b < n_blocks; b++)
        ref_dequant_block(w, row, n_blocks, b, out + 32 * b);
}

static float ref_dot_q4_row(const xe_q4 *w, uint64_t row, int n_in, const float *x) {
    int n_blocks = n_in / 32;
    float sum = 0.0f;

    for (int b = 0; b < n_blocks; b++) {
        float weights[32];
        ref_dequant_block(w, row, n_blocks, b, weights);
        for (int j = 0; j < 32; j++)
            sum += weights[j] * x[32 * b + j];
    }
    return sum;
}

static void ref_matvec_q4(const xe_q4 *w, uint64_t first_row, int n_in, int n_out, const float *x, float *y) {
    for (int r = 0; r < n_out; r++)
        y[r] = ref_dot_q4_row(w, first_row + (uint64_t)r, n_in, x);
}

#define REF_Q8_MAX_IN 8192

static float ref_fp16_round(float x) {
    return (float)(_Float16)x;
}

static void ref_quantize_q8(const float *x, int n_in, int8_t *qs, float *ds) {
    int n_blocks = n_in / 32;

    for (int b = 0; b < n_blocks; b++) {
        float amax = 0.0f;

        for (int j = 0; j < 32; j++) {
            float v = fabsf(x[32 * b + j]);
            if (v > amax) amax = v;
        }

        float d = amax / 127.0f;
        float id = d ? 1.0f / d : 0.0f;

        ds[b] = ref_fp16_round(d);

        for (int j = 0; j < 32; j++)
            qs[32 * b + j] = (int8_t)roundf(x[32 * b + j] * id);
    }
}

static int ref_dot_q4_q8_integer(const xe_q4 *w, uint64_t row, int blocks,
                                 int block, const int8_t *q) {
    uint8_t nibbles[16];
    xe_q4_copy_block(w, row, blocks, block, nibbles);
    int sumi0 = 0;
    int sumi1 = 0;

    for (int j = 0; j < 16; j++) {
        int v0 = (nibbles[j] & 0x0F) - 8;
        int v1 = (nibbles[j] >> 4) - 8;
        sumi0 += v0 * q[j];
        sumi1 += v1 * q[j + 16];
    }
    return sumi0 + sumi1;
}

static float ref_dot_q4_q8_row(const xe_q4 *w, uint64_t row, int n_in, const int8_t *qs, const float *ds) {
    int n_blocks = n_in / 32;
    float sumf = 0.0f;

    for (int b = 0; b < n_blocks; b++) {
        int integer = ref_dot_q4_q8_integer(w, row, n_blocks, b, qs + 32 * b);
        sumf += (float)integer * ref_fp16_to_fp32(xe_q4_scale(w, row, n_blocks, b)) * ds[b];
    }
    return sumf;
}

static float ref_dot_q4_q8_row_v1(const xe_q4 *w, uint64_t row, int n_in,
                                   const int8_t *qs, const float *ds) {
    int n_blocks = n_in / 32;
    float sums[8] = { 0.0f };
    int b = 0;

    for (; b + 7 < n_blocks; b += 8) {
        for (int j = 0; j < 8; j++) {
            int integer = ref_dot_q4_q8_integer(w, row, n_blocks, b + j,
                                                qs + 32 * (b + j));
            float scale = ref_fp16_to_fp32(xe_q4_scale(w, row, n_blocks, b + j)) *
                          ds[b + j];
            sums[j] += (float)integer * scale;
        }
    }

    float lo0 = sums[0] + sums[4];
    float lo1 = sums[1] + sums[5];
    float lo2 = sums[2] + sums[6];
    float lo3 = sums[3] + sums[7];
    float result = (lo0 + lo1) + (lo2 + lo3);
    for (; b < n_blocks; b++) {
        int integer = ref_dot_q4_q8_integer(w, row, n_blocks, b, qs + 32 * b);
        result += (float)integer * ref_fp16_to_fp32(xe_q4_scale(w, row, n_blocks, b)) * ds[b];
    }
    return result;
}

static void ref_matvec_q4_q8_packed(const xe_q4 *w, uint64_t first_row, int n_in,
                                    int n_out, const int8_t *qs, const float *ds,
                                    int v1_order, float *y) {
    for (int r = 0; r < n_out; r++) {
        uint64_t row = first_row + (uint64_t)r;
        y[r] = v1_order ? ref_dot_q4_q8_row_v1(w, row, n_in, qs, ds) :
                          ref_dot_q4_q8_row(w, row, n_in, qs, ds);
    }
}

static void ref_matvec_q4_q8(const xe_q4 *w, uint64_t first_row, int n_in, int n_out,
                             const float *x, int v1_order, float *y) {
    int8_t qs[REF_Q8_MAX_IN];
    float ds[REF_Q8_MAX_IN / 32];

    if (n_in > REF_Q8_MAX_IN) xe_fatal("q8 matvec: n_in %d exceeds %d", n_in, REF_Q8_MAX_IN);

    ref_quantize_q8(x, n_in, qs, ds);
    ref_matvec_q4_q8_packed(w, first_row, n_in, n_out, qs, ds, v1_order, y);
}

static float ref_dot_f32(const float *a, const float *b, int n) {
    float sum = 0.0f;

    for (int i = 0; i < n; i++)
        sum += a[i] * b[i];
    return sum;
}

static void ref_matvec_f32(const float *w, int n_in, int n_out, const float *x, float *y) {
    for (int r = 0; r < n_out; r++)
        y[r] = ref_dot_f32(w + (size_t)r * (size_t)n_in, x, n_in);
}

static void ref_rmsnorm(const float *x, const float *weight, int n, float *y) {
    float sum = 0.0f;

    for (int i = 0; i < n; i++)
        sum += x[i] * x[i];

    float scale = 1.0f / sqrtf(sum / (float)n + XE_RMS_EPS);

    for (int i = 0; i < n; i++)
        y[i] = x[i] * scale * weight[i];
}

static void ref_rmsnorm_weightless(const float *x, int n, float *y) {
    float sum = 0.0f;

    for (int i = 0; i < n; i++)
        sum += x[i] * x[i];

    float scale = 1.0f / sqrtf(sum / (float)n + XE_RMS_EPS);

    for (int i = 0; i < n; i++)
        y[i] = x[i] * scale;
}

static void ref_softmax(float *x, int n) {
    float max = x[0];

    for (int i = 1; i < n; i++)
        if (x[i] > max) max = x[i];

    float sum = 0.0f;
    for (int i = 0; i < n; i++) {
        x[i] = expf(x[i] - max);
        sum += x[i];
    }

    for (int i = 0; i < n; i++)
        x[i] /= sum;
}

static void ref_top_k(const float *values, int n, int k, int *out_idx, float *out_val) {
    for (int i = 0; i < k; i++) {
        out_idx[i] = -1;
        out_val[i] = -INFINITY;
    }

    for (int i = 0; i < n; i++) {
        if (values[i] <= out_val[k - 1]) continue;

        int slot = k - 1;
        while (slot > 0 && out_val[slot - 1] < values[i]) {
            out_val[slot] = out_val[slot - 1];
            out_idx[slot] = out_idx[slot - 1];
            slot--;
        }
        out_val[slot] = values[i];
        out_idx[slot] = i;
    }
}

static float ref_gelu(float x) {
    float inner = ref_gelu_sqrt_2_over_pi * (x + ref_gelu_coef_a * x * x * x);
    return 0.5f * x * (1.0f + tanhf(inner));
}

static void ref_geglu(const float *gate, const float *up, int n, float *y) {
    for (int i = 0; i < n; i++)
        y[i] = ref_gelu(gate[i]) * up[i];
}

static float ref_gelu_fp16(float x) {
    if (x <= -10.0f) return 0.0f;
    if (x >= 10.0f) return x;
    return (float)(_Float16)ref_gelu((float)(_Float16)x);
}

static void ref_geglu_fp16(const float *gate, const float *up, int n, float *y) {
    for (int i = 0; i < n; i++)
        y[i] = ref_gelu_fp16(gate[i]) * up[i];
}

static void ref_rope_neox(float *heads, int n_heads, int head_dim, int pos, float base, const float *freq_factors) {
    int half = head_dim / 2;

    for (int h = 0; h < n_heads; h++) {
        float *head = heads + (size_t)h * (size_t)head_dim;

        for (int i = 0; i < half; i++) {
            float freq = powf(base, -2.0f * (float)i / (float)head_dim);
            if (freq_factors) freq = freq / freq_factors[i];

            float theta = (float)pos * freq;
            float cos_theta = cosf(theta);
            float sin_theta = sinf(theta);
            float lo = head[i];
            float hi = head[i + half];

            head[i] = lo * cos_theta - hi * sin_theta;
            head[i + half] = lo * sin_theta + hi * cos_theta;
        }
    }
}

static void ref_attention(const float *q, const float *k_cache, const float *v_cache, int pos,
                          int n_kv_heads, int head_dim, int sliding_window, float *scores, float *out) {
    int group = XE_Q_HEADS / n_kv_heads;
    int n_keys = pos + 1;

    for (int h = 0; h < XE_Q_HEADS; h++) {
        const float *q_head = q + (size_t)h * (size_t)head_dim;
        int kv_head = h / group;

        for (int p = 0; p < n_keys; p++) {
            if (sliding_window > 0 && pos - p >= sliding_window) {
                scores[p] = -INFINITY;
                continue;
            }
            const float *k_head = k_cache + ((size_t)p * (size_t)n_kv_heads + (size_t)kv_head) * (size_t)head_dim;
            scores[p] = ref_dot_f32(q_head, k_head, head_dim) * ref_kq_scale;
        }

        ref_softmax(scores, n_keys);

        float *out_head = out + (size_t)h * (size_t)head_dim;
        for (int d = 0; d < head_dim; d++)
            out_head[d] = 0.0f;

        for (int p = 0; p < n_keys; p++) {
            const float *v_head = v_cache + ((size_t)p * (size_t)n_kv_heads + (size_t)kv_head) * (size_t)head_dim;
            for (int d = 0; d < head_dim; d++)
                out_head[d] += scores[p] * v_head[d];
        }
    }
}

typedef struct {
    float *hidden;
    float *normed;
    float *q;
    float *k;
    float *v;
    float *attn_heads;
    float *attn_proj;
    float *attn_out;
    float *scores;
    float *dense_in;
    float *dense_gate;
    float *dense_up;
    float *dense_act;
    float *dense_out;
    float *router_in;
    float *router_probs;
    float *moe_in;
    float *expert_gate_up;
    float *expert_act;
    float *expert_out;
    float *moe_out;
    float *combined;
    float *logits;
    float *k_cache[XE_LAYERS];
    float *v_cache[XE_LAYERS];
    int q8;
    const char *layers_dir;
    int dump;
} ref_state;

static int ref_kv_dim(int layer) {
    if (XE_IS_GLOBAL(layer)) return XE_GLOBAL_KV_HEADS * XE_GLOBAL_HEAD_DIM;
    return XE_SWA_KV_HEADS * XE_SWA_HEAD_DIM;
}

static void ref_state_init(ref_state *s, int n_tokens) {
    int max_q_dim = XE_Q_HEADS * XE_GLOBAL_HEAD_DIM;
    int max_kv_dim = XE_SWA_KV_HEADS * XE_SWA_HEAD_DIM;

    s->hidden = xe_alloc(NULL, XE_EMBD * sizeof(float), XE_MEM_HOST);
    s->normed = xe_alloc(NULL, XE_EMBD * sizeof(float), XE_MEM_HOST);
    s->q = xe_alloc(NULL, (size_t)max_q_dim * sizeof(float), XE_MEM_HOST);
    s->k = xe_alloc(NULL, (size_t)max_kv_dim * sizeof(float), XE_MEM_HOST);
    s->v = xe_alloc(NULL, (size_t)max_kv_dim * sizeof(float), XE_MEM_HOST);
    s->attn_heads = xe_alloc(NULL, (size_t)max_q_dim * sizeof(float), XE_MEM_HOST);
    s->attn_proj = xe_alloc(NULL, XE_EMBD * sizeof(float), XE_MEM_HOST);
    s->attn_out = xe_alloc(NULL, XE_EMBD * sizeof(float), XE_MEM_HOST);
    s->scores = xe_alloc(NULL, (size_t)n_tokens * sizeof(float), XE_MEM_HOST);
    s->dense_in = xe_alloc(NULL, XE_EMBD * sizeof(float), XE_MEM_HOST);
    s->dense_gate = xe_alloc(NULL, XE_DENSE_FFN * sizeof(float), XE_MEM_HOST);
    s->dense_up = xe_alloc(NULL, XE_DENSE_FFN * sizeof(float), XE_MEM_HOST);
    s->dense_act = xe_alloc(NULL, XE_DENSE_FFN * sizeof(float), XE_MEM_HOST);
    s->dense_out = xe_alloc(NULL, XE_EMBD * sizeof(float), XE_MEM_HOST);
    s->router_in = xe_alloc(NULL, XE_EMBD * sizeof(float), XE_MEM_HOST);
    s->router_probs = xe_alloc(NULL, XE_EXPERTS * sizeof(float), XE_MEM_HOST);
    s->moe_in = xe_alloc(NULL, XE_EMBD * sizeof(float), XE_MEM_HOST);
    s->expert_gate_up = xe_alloc(NULL, 2 * XE_EXPERT_FFN * sizeof(float), XE_MEM_HOST);
    s->expert_act = xe_alloc(NULL, XE_EXPERT_FFN * sizeof(float), XE_MEM_HOST);
    s->expert_out = xe_alloc(NULL, XE_EMBD * sizeof(float), XE_MEM_HOST);
    s->moe_out = xe_alloc(NULL, XE_EMBD * sizeof(float), XE_MEM_HOST);
    s->combined = xe_alloc(NULL, XE_EMBD * sizeof(float), XE_MEM_HOST);
    s->logits = xe_alloc(NULL, XE_VOCAB * sizeof(float), XE_MEM_HOST);

    for (int il = 0; il < XE_LAYERS; il++) {
        size_t bytes = (size_t)n_tokens * (size_t)ref_kv_dim(il) * sizeof(float);
        s->k_cache[il] = xe_alloc(NULL, bytes, XE_MEM_HOST);
        s->v_cache[il] = xe_alloc(NULL, bytes, XE_MEM_HOST);
    }
}

static void ref_state_free(ref_state *s) {
    xe_free(NULL, s->hidden, XE_MEM_HOST);
    xe_free(NULL, s->normed, XE_MEM_HOST);
    xe_free(NULL, s->q, XE_MEM_HOST);
    xe_free(NULL, s->k, XE_MEM_HOST);
    xe_free(NULL, s->v, XE_MEM_HOST);
    xe_free(NULL, s->attn_heads, XE_MEM_HOST);
    xe_free(NULL, s->attn_proj, XE_MEM_HOST);
    xe_free(NULL, s->attn_out, XE_MEM_HOST);
    xe_free(NULL, s->scores, XE_MEM_HOST);
    xe_free(NULL, s->dense_in, XE_MEM_HOST);
    xe_free(NULL, s->dense_gate, XE_MEM_HOST);
    xe_free(NULL, s->dense_up, XE_MEM_HOST);
    xe_free(NULL, s->dense_act, XE_MEM_HOST);
    xe_free(NULL, s->dense_out, XE_MEM_HOST);
    xe_free(NULL, s->router_in, XE_MEM_HOST);
    xe_free(NULL, s->router_probs, XE_MEM_HOST);
    xe_free(NULL, s->moe_in, XE_MEM_HOST);
    xe_free(NULL, s->expert_gate_up, XE_MEM_HOST);
    xe_free(NULL, s->expert_act, XE_MEM_HOST);
    xe_free(NULL, s->expert_out, XE_MEM_HOST);
    xe_free(NULL, s->moe_out, XE_MEM_HOST);
    xe_free(NULL, s->combined, XE_MEM_HOST);
    xe_free(NULL, s->logits, XE_MEM_HOST);

    for (int il = 0; il < XE_LAYERS; il++) {
        xe_free(NULL, s->k_cache[il], XE_MEM_HOST);
        xe_free(NULL, s->v_cache[il], XE_MEM_HOST);
    }
}

static void ref_matvec(const ref_state *s, const xe_q4 *w, uint64_t first_row, int n_in, int n_out,
                       const float *x, float *y) {
    if (s->q8)
        ref_matvec_q4_q8(w, first_row, n_in, n_out, x, s->q8 == 2, y);
    else
        ref_matvec_q4(w, first_row, n_in, n_out, x, y);
}

static void ref_activate(const ref_state *s, const float *gate, const float *up, int n, float *y) {
    if (s->q8)
        ref_geglu_fp16(gate, up, n, y);
    else
        ref_geglu(gate, up, n, y);
}

static void ref_dump(const ref_state *s, const char *name, int il, const float *v) {
    char path[4096];

    if (il < 0)
        snprintf(path, sizeof path, "%s/%s.bin", s->layers_dir, name);
    else
        snprintf(path, sizeof path, "%s/%s_%d.bin", s->layers_dir, name, il);

    FILE *f = fopen(path, "wb");
    if (!f) xe_fatal("%s: %s", path, strerror(errno));
    if (fwrite(v, sizeof(float), XE_EMBD, f) != (size_t)XE_EMBD)
        xe_fatal("%s: short write", path);
    if (fclose(f) != 0) xe_fatal("%s: %s", path, strerror(errno));
}

static void ref_embed(const xe_engine *e, int32_t token, float *x) {
    float scale = sqrtf((float)XE_EMBD);

    ref_dequant_row(&e->tok_embd, (uint64_t)token, XE_EMBD, x);

    for (int i = 0; i < XE_EMBD; i++)
        x[i] = x[i] * scale;
}

static void ref_attention_half(const xe_engine *e, int il, int pos, ref_state *s) {
    const xe_layer *l = &e->layers[il];
    int is_global = XE_IS_GLOBAL(il);
    int head_dim = is_global ? XE_GLOBAL_HEAD_DIM : XE_SWA_HEAD_DIM;
    int n_kv_heads = is_global ? XE_GLOBAL_KV_HEADS : XE_SWA_KV_HEADS;
    int q_dim = XE_Q_HEADS * head_dim;
    int kv_dim = n_kv_heads * head_dim;
    float rope_base = is_global ? XE_GLOBAL_ROPE_BASE : XE_SWA_ROPE_BASE;
    const float *freq_factors = is_global ? e->rope_freqs : NULL;
    int sliding_window = is_global ? 0 : XE_SWA_WINDOW;

    ref_rmsnorm(s->hidden, l->attn_norm, XE_EMBD, s->normed);

    ref_matvec(s, &l->attn_q, 0, XE_EMBD, q_dim, s->normed, s->q);
    ref_matvec(s, &l->attn_k, 0, XE_EMBD, kv_dim, s->normed, s->k);

    if (is_global) {
        for (int i = 0; i < kv_dim; i++)
            s->v[i] = s->k[i];
    } else {
        ref_matvec(s, &l->attn_v, 0, XE_EMBD, kv_dim, s->normed, s->v);
    }

    for (int h = 0; h < XE_Q_HEADS; h++) {
        float *head = s->q + (size_t)h * (size_t)head_dim;
        ref_rmsnorm(head, l->q_norm, head_dim, head);
    }
    for (int h = 0; h < n_kv_heads; h++) {
        float *head = s->k + (size_t)h * (size_t)head_dim;
        ref_rmsnorm(head, l->k_norm, head_dim, head);
    }
    for (int h = 0; h < n_kv_heads; h++) {
        float *head = s->v + (size_t)h * (size_t)head_dim;
        ref_rmsnorm_weightless(head, head_dim, head);
    }

    ref_rope_neox(s->q, XE_Q_HEADS, head_dim, pos, rope_base, freq_factors);
    ref_rope_neox(s->k, n_kv_heads, head_dim, pos, rope_base, freq_factors);

    float *k_slot = s->k_cache[il] + (size_t)pos * (size_t)kv_dim;
    float *v_slot = s->v_cache[il] + (size_t)pos * (size_t)kv_dim;
    for (int i = 0; i < kv_dim; i++) {
        k_slot[i] = s->k[i];
        v_slot[i] = s->v[i];
    }

    if (s->q8) {
        for (int i = 0; i < kv_dim; i++) {
            k_slot[i] = ref_fp16_round(k_slot[i]);
            v_slot[i] = ref_fp16_round(v_slot[i]);
        }
    }

    ref_attention(s->q, s->k_cache[il], s->v_cache[il], pos, n_kv_heads, head_dim,
                  sliding_window, s->scores, s->attn_heads);

    ref_matvec(s, &l->attn_o, 0, q_dim, XE_EMBD, s->attn_heads, s->attn_proj);
    ref_rmsnorm(s->attn_proj, l->post_attn_norm, XE_EMBD, s->attn_proj);

    for (int i = 0; i < XE_EMBD; i++)
        s->attn_out[i] = s->attn_proj[i] + s->hidden[i];
}

static void ref_dense_branch(const xe_layer *l, ref_state *s) {
    ref_rmsnorm(s->attn_out, l->ffn_norm, XE_EMBD, s->dense_in);

    ref_matvec(s, &l->ffn_gate, 0, XE_EMBD, XE_DENSE_FFN, s->dense_in, s->dense_gate);
    ref_matvec(s, &l->ffn_up, 0, XE_EMBD, XE_DENSE_FFN, s->dense_in, s->dense_up);
    ref_activate(s, s->dense_gate, s->dense_up, XE_DENSE_FFN, s->dense_act);
    ref_matvec(s, &l->ffn_down, 0, XE_DENSE_FFN, XE_EMBD, s->dense_act, s->dense_out);

    ref_rmsnorm(s->dense_out, l->post_ffw_norm1, XE_EMBD, s->dense_out);
}

static void ref_router(const xe_layer *l, ref_state *s, int *experts, float *weights) {
    float input_scale = 1.0f / sqrtf((float)XE_EMBD);

    ref_rmsnorm_weightless(s->attn_out, XE_EMBD, s->router_in);
    for (int i = 0; i < XE_EMBD; i++)
        s->router_in[i] = s->router_in[i] * input_scale * l->router_scale[i];

    ref_matvec_f32(l->router_w, XE_EMBD, XE_EXPERTS, s->router_in, s->router_probs);
    ref_softmax(s->router_probs, XE_EXPERTS);

    ref_top_k(s->router_probs, XE_EXPERTS, XE_EXPERTS_USED, experts, weights);

    float sum = 0.0f;
    for (int i = 0; i < XE_EXPERTS_USED; i++)
        sum += weights[i];
    if (sum < ref_moe_weight_sum_min) sum = ref_moe_weight_sum_min;

    for (int i = 0; i < XE_EXPERTS_USED; i++)
        weights[i] = weights[i] / sum;
}

static void ref_moe_branch(const xe_layer *l, ref_state *s, const int *experts, const float *weights) {
    ref_rmsnorm(s->attn_out, l->pre_ffw_norm2, XE_EMBD, s->moe_in);

    for (int i = 0; i < XE_EMBD; i++)
        s->moe_out[i] = 0.0f;

    for (int i = 0; i < XE_EXPERTS_USED; i++) {
        int expert = experts[i];
        uint64_t gate_up_row = (uint64_t)expert * (2 * XE_EXPERT_FFN);
        uint64_t down_row = (uint64_t)expert * XE_EMBD;
        float down_scale = l->down_exps_scale[expert];

        ref_matvec(s, &l->gate_up_exps, gate_up_row, XE_EMBD, 2 * XE_EXPERT_FFN, s->moe_in, s->expert_gate_up);
        ref_activate(s, s->expert_gate_up, s->expert_gate_up + XE_EXPERT_FFN, XE_EXPERT_FFN, s->expert_act);
        ref_matvec(s, &l->down_exps, down_row, XE_EXPERT_FFN, XE_EMBD, s->expert_act, s->expert_out);

        for (int j = 0; j < XE_EMBD; j++)
            s->moe_out[j] += weights[i] * (s->expert_out[j] * down_scale);
    }

    ref_rmsnorm(s->moe_out, l->post_ffw_norm2, XE_EMBD, s->moe_out);
}

static void ref_ffn_half(const xe_engine *e, int il, ref_state *s) {
    const xe_layer *l = &e->layers[il];
    int experts[XE_EXPERTS_USED];
    float weights[XE_EXPERTS_USED];
    float out_scale = l->layer_out_scale[0];

    ref_dense_branch(l, s);
    ref_router(l, s, experts, weights);
    ref_moe_branch(l, s, experts, weights);

    for (int i = 0; i < XE_EMBD; i++)
        s->combined[i] = s->dense_out[i] + s->moe_out[i];

    ref_rmsnorm(s->combined, l->post_ffw_norm, XE_EMBD, s->combined);

    for (int i = 0; i < XE_EMBD; i++)
        s->hidden[i] = (s->combined[i] + s->attn_out[i]) * out_scale;
}

static void ref_layer(const xe_engine *e, int il, int pos, ref_state *s) {
    ref_attention_half(e, il, pos, s);
    if (s->dump) ref_dump(s, "attn_out", il, s->attn_out);

    ref_ffn_half(e, il, s);
    if (s->dump) ref_dump(s, "l_out", il, s->hidden);
}

static void ref_output(const xe_engine *e, ref_state *s) {
    ref_rmsnorm(s->hidden, e->out_norm, XE_EMBD, s->normed);
    ref_matvec(s, &e->tok_embd, 0, XE_EMBD, XE_VOCAB, s->normed, s->logits);

    for (int i = 0; i < XE_VOCAB; i++)
        s->logits[i] = XE_LOGIT_SOFTCAP * tanhf(s->logits[i] / XE_LOGIT_SOFTCAP);
}

static void ref_forward(const xe_engine *e, const int32_t *tokens, int n_tokens, ref_state *s) {
    for (int pos = 0; pos < n_tokens; pos++) {
        s->dump = s->layers_dir && pos == n_tokens - 1;

        ref_embed(e, tokens[pos], s->hidden);
        if (s->dump) ref_dump(s, "inp_scaled", -1, s->hidden);

        for (int il = 0; il < XE_LAYERS; il++)
            ref_layer(e, il, pos, s);
    }
    ref_output(e, s);
}

void xe_oracle(xe_engine *e, const int32_t *tokens, int n_tokens, const char *dump_path,
               const char *layers_dir, int q8_mode, FILE *out) {
    if (e->vocab_only) xe_fatal("oracle: engine opened vocab-only, no weights available");
    if (n_tokens <= 0 || n_tokens > e->context)
        xe_fatal("oracle: token count %d out of range [1, %d]", n_tokens, e->context);
    for (int i = 0; i < n_tokens; i++)
        if (tokens[i] < 0 || tokens[i] >= XE_VOCAB)
            xe_fatal("oracle: token id %d out of range [0, %d]", tokens[i], XE_VOCAB - 1);

    if (layers_dir && mkdir(layers_dir, 0777) != 0 && errno != EEXIST)
        xe_fatal("%s: %s", layers_dir, strerror(errno));

    ref_state s;
    ref_state_init(&s, n_tokens);
    s.q8 = q8_mode;
    s.layers_dir = layers_dir;
    s.dump = 0;
    ref_forward(e, tokens, n_tokens, &s);

    for (int i = 0; i < XE_VOCAB; i++)
        if (!isfinite(s.logits[i]))
            xe_fatal("oracle: logit %d is not finite (%g)", i, (double)s.logits[i]);

    int top_idx[8];
    float top_val[8];
    ref_top_k(s.logits, XE_VOCAB, 8, top_idx, top_val);

    for (int i = 0; i < 8; i++)
        fprintf(out, "%8d  %12.6f\n", top_idx[i], (double)top_val[i]);

    if (dump_path) {
        FILE *f = fopen(dump_path, "wb");
        if (!f) xe_fatal("%s: %s", dump_path, strerror(errno));
        if (fwrite(s.logits, sizeof(float), XE_VOCAB, f) != (size_t)XE_VOCAB)
            xe_fatal("%s: short write", dump_path);
        if (fclose(f) != 0) xe_fatal("%s: %s", dump_path, strerror(errno));
    }

    ref_state_free(&s);
}
