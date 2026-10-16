/* Exercise the real prefill dispatch with a recording Level Zero driver. */
#define zeKernelCreate test_zeKernelCreate
#define zeKernelSetGroupSize test_zeKernelSetGroupSize
#define zeKernelSetArgumentValue test_zeKernelSetArgumentValue
#define zeCommandListAppendLaunchKernel test_zeCommandListAppendLaunchKernel
#define zeCommandListHostSynchronize test_zeCommandListHostSynchronize
#define zeCommandListAppendMemoryCopy test_zeCommandListAppendMemoryCopy
#include "../xenolith.c"
#ifdef NDEBUG
#undef NDEBUG
#endif
#include <assert.h>

typedef struct {
    uint64_t arguments[16];
} test_kernel;

static test_kernel causal_kernel, visual_kernel, global_kernel;
static test_kernel generic_kernel, input_kernel, finish_kernel;
static ze_kernel_handle_t last_kernel;
static unsigned creates, group_sizes, launches;
static xe_prefill_workspace *active_workspace;
static int batch_rows;
static xe_attention_mode batch_mode;
static unsigned attention_count, input_count, finish_count, copy_count;

static void batch_launch(test_kernel *kernel, const ze_group_count_t *groups) {
    if (!active_workspace) return;
    xe_prefill_workspace *w = active_workspace;
    unsigned tiles = (batch_rows + 511) / 512;
    unsigned layer = finish_count / tiles;
    int start = (finish_count % tiles) * 512;
    int rows = batch_rows - start;
    if (rows > 512) rows = 512;
    if (kernel == &causal_kernel || kernel == &visual_kernel || kernel == &global_kernel) {
        assert(attention_count < XE_LAYERS);
        assert(finish_count == attention_count * tiles);
        test_kernel *expected = XE_IS_GLOBAL(attention_count) ? &global_kernel
                              : batch_mode == XE_ATTENTION_VISUAL ? &visual_kernel
                              : &causal_kernel;
        assert(kernel == expected && kernel->arguments[4] == (unsigned)batch_rows);
        attention_count++;
    } else if (kernel == &input_kernel) {
        assert(layer < XE_LAYERS && attention_count == layer + 1);
        assert(input_count == finish_count);
        assert(kernel->arguments[0] == (uintptr_t)(w->attention_output + (size_t)start * XE_EMBD));
        assert(kernel->arguments[5] == (uintptr_t)w->dense_input.qs);
        assert(kernel->arguments[8] == (uintptr_t)w->moe_input.qs);
        assert(kernel->arguments[11] == (uintptr_t)w->router_input);
        assert(kernel->arguments[13] == (unsigned)rows && rows <= w->ffn_capacity);
        input_count++;
    } else if (kernel == &finish_kernel) {
        assert(input_count == finish_count + 1);
        float *output = w->hidden[(layer + 1) % 2] + (size_t)start * XE_EMBD;
        assert(kernel->arguments[0] == (uintptr_t)w->dense_down);
        assert(kernel->arguments[1] == (uintptr_t)w->moe_output);
        assert(kernel->arguments[5] == (uintptr_t)(w->attention_output + (size_t)start * XE_EMBD));
        assert(kernel->arguments[7] == (uintptr_t)output);
        assert(kernel->arguments[8] == (unsigned)rows && groups->groupCountX == (unsigned)rows);
        /* Marker writes exercise the real tile output offsets and hidden swap.
         * They do not simulate attention, quantization or GPU synchronization. */
        for (int row = 0; row < rows; row++) {
            float marker = (float)(layer * 2048 + start + row);
            output[(size_t)row * XE_EMBD] = marker;
            output[(size_t)(row + 1) * XE_EMBD - 1] = marker;
        }
        finish_count++;
    }
}

ze_result_t ZE_APICALL test_zeKernelCreate(ze_module_handle_t module,
        const ze_kernel_desc_t *desc, ze_kernel_handle_t *out) {
    assert(module);
    assert(desc->stype == ZE_STRUCTURE_TYPE_KERNEL_DESC);
    assert(!strcmp(desc->pKernelName, "xe_prefill_attn_online_b8_swa_visual"));
    *out = (ze_kernel_handle_t)&visual_kernel;
    creates++;
    return ZE_RESULT_SUCCESS;
}

ze_result_t ZE_APICALL test_zeKernelSetGroupSize(ze_kernel_handle_t kernel,
        uint32_t x, uint32_t y, uint32_t z) {
    assert(kernel == (ze_kernel_handle_t)&visual_kernel);
    assert(x == 128 && y == 1 && z == 1);
    group_sizes++;
    return ZE_RESULT_SUCCESS;
}

ze_result_t ZE_APICALL test_zeKernelSetArgumentValue(ze_kernel_handle_t kernel,
        uint32_t index, size_t size, const void *value) {
    test_kernel *record = (test_kernel *)kernel;
    assert(record && index < 16 && size <= sizeof record->arguments[index]);
    record->arguments[index] = 0;
    memcpy(&record->arguments[index], value, size);
    return ZE_RESULT_SUCCESS;
}

ze_result_t ZE_APICALL test_zeCommandListAppendLaunchKernel(
        ze_command_list_handle_t commands, ze_kernel_handle_t kernel,
        const ze_group_count_t *groups, ze_event_handle_t event,
        uint32_t wait_count, ze_event_handle_t *waits) {
    assert(commands && kernel && groups->groupCountX > 0);
    assert(groups->groupCountY > 0 && groups->groupCountZ == 1);
    assert(!event && !wait_count && !waits);
    last_kernel = kernel;
    launches++;
    batch_launch((test_kernel *)kernel, groups);
    return ZE_RESULT_SUCCESS;
}

ze_result_t ZE_APICALL test_zeCommandListAppendMemoryCopy(
        ze_command_list_handle_t commands, void *dest, const void *source,
        size_t size, ze_event_handle_t event, uint32_t wait_count,
        ze_event_handle_t *waits) {
    assert(active_workspace && commands && dest && source);
    assert(!event && !wait_count && !waits);
    assert(size == (size_t)batch_rows * XE_GLOBAL_HEAD_DIM * sizeof(_Float16));
    copy_count++;
    return ZE_RESULT_SUCCESS;
}

ze_result_t ZE_APICALL test_zeCommandListHostSynchronize(
        ze_command_list_handle_t commands, uint64_t timeout) {
    assert(commands && timeout == UINT64_MAX);
    return ZE_RESULT_SUCCESS;
}

static void attention_case(xe_engine *e, xe_attention_mode mode,
                           int dimension, test_kernel *expected) {
    float q = 0, output = 0;
    _Float16 k = 0, v = 0;
    int heads = dimension == 256 ? 8 : 2;
    unsigned before = launches;
    xe_prefill_attention_online_append(e, &q, &k, &v, &output,
                                        17, 24, dimension, heads, 7, 1024, mode);
    assert(launches == before + 1 && last_kernel == (ze_kernel_handle_t)expected);
    assert(expected->arguments[0] == (uintptr_t)&q);
    assert(expected->arguments[1] == (uintptr_t)&k);
    assert(expected->arguments[2] == (uintptr_t)&v);
    assert(expected->arguments[3] == (uintptr_t)&output);
    assert(expected->arguments[4] == 17 && expected->arguments[5] == 24);
    assert(expected->arguments[8] == (unsigned)heads && expected->arguments[9] == 7);
    if (dimension == 256) {
        assert(expected->arguments[6] == 256 && expected->arguments[10] == 1024);
    } else {
        assert(expected->arguments[6] == 17 && expected->arguments[10] == 0);
    }
}

static void span(unsigned char *base, size_t bytes, unsigned char **end,
                  void *ptr, size_t size) {
    unsigned char *p = ptr;
    assert(size && (uintptr_t)p % 64 == 0);
    assert(p >= *end && p + size <= base + bytes);
    p[0] = 0x5a;
    p[size - 1] = 0x5a;
    *end = p + size;
}

static void q8_span(unsigned char *base, size_t bytes, unsigned char **end,
                     xe_q8 *q, int elements) {
    assert(q->n == elements);
    span(base, bytes, end, q->qs, elements);
    span(base, bytes, end, q->d, (size_t)elements / 32 * sizeof *q->d);
    span(base, bytes, end, q->sigma, (size_t)elements / 32 * sizeof *q->sigma);
}

static void layout_case(int rows, size_t expected_bytes) {
    xe_prefill_workspace w;
    size_t bytes = xe_prefill_workspace_layout_rows(&w, NULL, rows);
    assert(bytes == expected_bytes && bytes % 64 == 0);
    unsigned char *memory = aligned_alloc(64, bytes + 128);
    assert(memory);
    memset(memory, 0xa5, 64);
    memset(memory + bytes + 64, 0xa5, 64);
    unsigned char *base = memory + 64, *end = base;
    assert(xe_prefill_workspace_layout_rows(&w, base, rows) == bytes);
    int ffn = rows < 512 ? rows : 512, routes = ffn * 8;
    assert(w.capacity == rows && w.ffn_capacity == ffn);
#define SPAN(member, count) span(base, bytes, &end, w.member, (size_t)(count) * sizeof *w.member)
#define Q8(member, count) q8_span(base, bytes, &end, &w.member, (count))
    SPAN(hidden[0], rows * 2816);
    SPAN(hidden[1], rows * 2816);
    SPAN(row_scale, rows);
    Q8(attention_input, rows * 2816);
    SPAN(q_projection, rows * 8192);
    SPAN(k_projection, rows * 2048);
    SPAN(v_projection, rows * 2048);
    SPAN(rope_swa_cos, rows * 128);
    SPAN(rope_swa_sin, rows * 128);
    SPAN(rope_global_cos, rows * 256);
    SPAN(rope_global_sin, rows * 256);
    SPAN(q_heads, rows * 8192);
    SPAN(k_batch, rows * 2048);
    SPAN(v_batch, rows * 2048);
    SPAN(k_stage, (1023 + rows) * 2048);
    SPAN(v_stage, (1023 + rows) * 2048);
    SPAN(attention_heads, rows * 8192);
    Q8(heads_q8, rows * 8192);
    SPAN(attention_projection, rows * 2816);
    SPAN(attention_output, rows * 2816);
    Q8(dense_input, ffn * 2816);
    Q8(moe_input, ffn * 2816);
    SPAN(router_input, ffn * 2816);
    SPAN(router_logits, ffn * 128);
    SPAN(route_expert, routes);
    SPAN(route_weight, routes);
    SPAN(routes.expert_count, 128);
    SPAN(routes.token_offset, 129);
    SPAN(routes.cursor, 128);
    SPAN(routes.tile_expert, 512);
    SPAN(routes.tile_m0, 512);
    SPAN(routes.packed_route, routes);
    SPAN(routes.route_packed, routes);
    Q8(packed_moe, routes * 2816);
    SPAN(dense_gate, ffn * 2112);
    SPAN(dense_up, ffn * 2112);
    Q8(dense_activation, ffn * 2112);
    SPAN(dense_down, ffn * 2816);
    SPAN(expert_gate_up, routes * 1408);
    Q8(expert_activation, routes * 704);
    SPAN(expert_down, routes * 2816);
    SPAN(moe_output, ffn * 2816);
#undef SPAN
#undef Q8
    assert((size_t)(base + bytes - end) < 64);
    for (int i = 0; i < 64; i++)
        assert(memory[i] == 0xa5 && memory[bytes + 64 + i] == 0xa5);
    free(memory);
}

static void batch_cases(xe_engine *e) {
    /* These handles record launch arguments; no kernel numerics run here. */
    ze_kernel_handle_t *generic[] = {
        &e->gpu.prefill_rms_scale, &e->gpu.prefill_norm_q8,
        &e->gpu.prefill_q4q8_n32, &e->gpu.prefill_q4q8_n64,
        &e->gpu.prefill_q4q8_n128, &e->gpu.prefill_q4q8_n128_tail,
        &e->gpu.prefill_qkv_post, &e->gpu.prefill_heads_q8,
        &e->gpu.prefill_rms_residual, &e->gpu.prefill_swa_stage,
        &e->gpu.prefill_swa_commit, &e->gpu.prefill_geglu_q8,
        &e->gpu.prefill_router_gemm, &e->gpu.prefill_router_top8,
        &e->gpu.prefill_route_reset, &e->gpu.prefill_route_count,
        &e->gpu.prefill_route_prefix, &e->gpu.prefill_route_scatter,
        &e->gpu.prefill_route_pack, &e->gpu.prefill_q4q8_grouped_n128,
        &e->gpu.prefill_q4q8_grouped_gate_n128, &e->gpu.prefill_q4q8_grouped_down_n128,
        &e->gpu.prefill_q4q8_grouped_m16_n64, &e->gpu.prefill_q4q8_grouped_m16_n128,
        &e->gpu.prefill_q4q8_grouped_m8_n128, &e->gpu.prefill_expert_geglu_q8,
        &e->gpu.prefill_route_reduce
    };
    for (size_t i = 0; i < sizeof generic / sizeof *generic; i++)
        *generic[i] = (ze_kernel_handle_t)&generic_kernel;
    e->gpu.prefill_ffn_input_q8 = (ze_kernel_handle_t)&input_kernel;
    e->gpu.prefill_ffn_finish = (ze_kernel_handle_t)&finish_kernel;
    e->context = 4096;
    e->pool_initialized = 1;
    e->owner = pthread_self();
    xe_prefill_workspace w;
    size_t bytes = xe_prefill_workspace_layout_rows(&w, NULL, 1120);
    unsigned char *memory = aligned_alloc(64, bytes + 128);
    assert(memory);
    memset(memory, 0xa5, 64);
    memset(memory + bytes + 64, 0xa5, 64);
    xe_prefill_workspace_layout_rows(&w, memory + 64, 1120);
    float *embedding = calloc((size_t)1120 * XE_EMBD, sizeof(float));
    int32_t *tokens = calloc(e->context, sizeof(int32_t));
    float hidden[XE_EMBD];
    assert(embedding && tokens);
    xe_session s = {
        .engine = e, .prefill_capacity = 1120, .prefill_workspace_size = bytes,
        .prefill_workspace = memory + 64, .tokens = tokens, .hidden = hidden,
        /* Recording commands never read/write KV, so share the large arena. */
        .swa_k = (_Float16 *)(memory + 64), .swa_v = (_Float16 *)(memory + 64),
        .global_k = (_Float16 *)(memory + 64), .global_v = (_Float16 *)(memory + 64)
    };
    const int counts[] = {511, 512, 513, 1024, 1120, 1, 31, 32, 96, 97, 513};
    for (size_t c = 0; c < sizeof counts / sizeof *counts; c++) {
        batch_rows = counts[c];
        batch_mode = c % 2 ? XE_ATTENTION_CAUSAL : XE_ATTENTION_VISUAL;
        attention_count = input_count = finish_count = copy_count = 0;
        s.n_tokens = c % 2 ? 1023 : 0;
        int start = s.n_tokens;
        for (int bank = 0; bank < 2; bank++)
            for (int row = 0; row < 1120; row++) {
                w.hidden[bank][(size_t)row * XE_EMBD] = -1;
                w.hidden[bank][(size_t)(row + 1) * XE_EMBD - 1] = -1;
            }
        active_workspace = &w;
        xe_prefill_batch_append_input(&s, tokens + start, batch_rows, start,
                                     embedding, batch_mode);
        assert(attention_count == XE_LAYERS);
        assert(finish_count == XE_LAYERS * (unsigned)((batch_rows + 511) / 512));
        assert(input_count == finish_count && copy_count == XE_GLOBAL_LAYERS * 4);
        assert(s.prefill_pending && s.prefill_pending_rows == batch_rows);
        assert(s.prefill_pending_hidden == w.hidden[XE_LAYERS % 2] +
                                           (size_t)(batch_rows - 1) * XE_EMBD);
        xe_prefill_batch_complete(&s, 0);
        float marker = (float)((XE_LAYERS - 1) * 2048 + batch_rows - 1);
        assert(hidden[0] == marker && hidden[XE_EMBD - 1] == marker);
        assert(!s.prefill_pending && s.n_tokens == start + batch_rows);
        assert(s.prefill_capacity == 1120 && s.prefill_workspace_size == bytes);
        for (int bank = 0; bank < 2; bank++)
            for (int row = batch_rows; row < 1120; row++) {
                assert(w.hidden[bank][(size_t)row * XE_EMBD] == -1);
                assert(w.hidden[bank][(size_t)(row + 1) * XE_EMBD - 1] == -1);
            }
        for (int i = 0; i < 64; i++)
            assert(memory[i] == 0xa5 && memory[bytes + 64 + i] == 0xa5);
        active_workspace = NULL;
        xe_session_reset(&s);
        assert(!s.n_tokens && s.prefill_capacity == 1120);
        assert(s.prefill_workspace == memory + 64 && s.prefill_workspace_size == bytes);
    }
    free(tokens);
    free(embedding);
    free(memory);
}

int main(void) {
    xe_engine e = {0};
    e.gpu.module = (ze_module_handle_t)(uintptr_t)1;
    e.gpu.commands = (ze_command_list_handle_t)(uintptr_t)1;
    e.gpu.prefill_attn_online_b8_swa = (ze_kernel_handle_t)&causal_kernel;
    e.gpu.prefill_attn_online_b8_global_shared = (ze_kernel_handle_t)&global_kernel;
    attention_case(&e, XE_ATTENTION_VISUAL, 512, &global_kernel);
    attention_case(&e, XE_ATTENTION_CAUSAL, 256, &causal_kernel);
    assert(creates == 0 && group_sizes == 0);
    attention_case(&e, XE_ATTENTION_VISUAL, 256, &visual_kernel);
    assert(creates == 1 && group_sizes == 1);
    attention_case(&e, XE_ATTENTION_CAUSAL, 256, &causal_kernel);
    attention_case(&e, XE_ATTENTION_VISUAL, 256, &visual_kernel);
    attention_case(&e, XE_ATTENTION_CAUSAL, 512, &global_kernel);
    assert(creates == 1 && group_sizes == 1);
    assert(e.gpu.prefill_attn_online_b8_swa_visual == (ze_kernel_handle_t)&visual_kernel);
    for (size_t i = 0; i < sizeof e.gpu.vision.kernels / sizeof *e.gpu.vision.kernels; i++)
        assert(!e.gpu.vision.kernels[i]);
    puts("Prefill attention dispatch: PASS");
    layout_case(511, 222227840);
    layout_case(512, 222645824);
    layout_case(513, 222837504);
    layout_case(1024, 320738880);
    layout_case(1120, 339131328);
    batch_cases(&e);
    puts("Prefill workspace layout and partial batches: PASS");
    return 0;
}
