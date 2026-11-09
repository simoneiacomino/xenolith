#include "../xenolith.c"
#include "support/scalar_ops.h"

static double layer_now(void) {
    struct timespec time;
    clock_gettime(CLOCK_MONOTONIC, &time);
    return (double)time.tv_sec + (double)time.tv_nsec * 1e-9;
}

static double layer_rel(const float *a, const float *b, size_t n) {
    double error = 0.0;
    double reference = 0.0;
    for (size_t i = 0; i < n; i++) {
        double difference = (double)a[i] - b[i];
        error += difference * difference;
        reference += (double)b[i] * b[i];
    }
    return sqrt(error / (reference + 1e-30));
}

/* Atomic route scatter may permute rows within an expert's packed segment.
 * Validate the complete inverse mapping before using it to compare payloads. */
static int layer_route_map_valid(const int *experts, const int *route_packed,
                                 const int *packed_route, const int *counts,
                                 const int *offsets, int routes) {
    int observed[XE_EXPERTS] = {0};
    if (offsets[0] != 0 || offsets[XE_EXPERTS] != routes) return 0;
    for (int expert = 0; expert < XE_EXPERTS; expert++)
        if (counts[expert] < 0 || offsets[expert] < 0 ||
            offsets[expert+1] > routes ||
            offsets[expert+1] - offsets[expert] != counts[expert]) return 0;
    for (int route = 0; route < routes; route++) {
        int expert = experts[route], packed = route_packed[route];
        if (expert < 0 || expert >= XE_EXPERTS || packed < 0 || packed >= routes)
            return 0;
        if (packed_route[packed] != route || packed < offsets[expert]
            || packed >= offsets[expert+1]) return 0;
        observed[expert]++;
    }
    for (int expert = 0; expert < XE_EXPERTS; expert++)
        if (observed[expert] != counts[expert]) return 0;
    return 1;
}

static size_t layer_remapped_rows(const void *before, const void *after,
                                  const int *before_map, const int *after_map,
                                  int routes, size_t stride) {
    const unsigned char *a = before, *b = after;
    size_t mismatches = 0;
    for (int route = 0; route < routes; route++)
        mismatches += memcmp(a + (size_t)before_map[route] * stride,
                             b + (size_t)after_map[route] * stride, stride) != 0;
    return mismatches;
}

static int layer_mapping_self_test(void) {
    const int experts[] = {0, 1, 0, 1};
    const int before_map[] = {0, 2, 1, 3}, before_inverse[] = {0, 2, 1, 3};
    const int after_map[] = {1, 3, 0, 2}, after_inverse[] = {2, 0, 3, 1};
    int counts[XE_EXPERTS] = {2, 2}, offsets[XE_EXPERTS+1] = {0, 2};
    for (int i = 2; i <= XE_EXPERTS; i++) offsets[i] = 4;
    int before[] = {11, 33, 22, 44}, after[] = {33, 11, 44, 22};
    int ok = layer_route_map_valid(experts, before_map, before_inverse, counts, offsets, 4)
             && layer_route_map_valid(experts, after_map, after_inverse, counts, offsets, 4)
             && !layer_remapped_rows(before, after, before_map, after_map, 4, sizeof(int));
    after[0]++;
    ok &= layer_remapped_rows(before, after, before_map, after_map, 4, sizeof(int)) == 1;
    int bad_inverse[] = {2, 0, 1, 3};
    ok &= !layer_route_map_valid(experts, after_map, bad_inverse, counts, offsets, 4);
    int wrong_experts[] = {1, 1, 0, 0};
    ok &= !layer_route_map_valid(wrong_experts, after_map, after_inverse, counts, offsets, 4);
    int out_of_range[] = {4, 3, 0, 2};
    ok &= !layer_route_map_valid(experts, out_of_range, after_inverse, counts, offsets, 4);
    printf("prefill-layer: semantic packed-route permutation/corruption %s\n", ok ? "PASS" : "FAIL");
    return ok;
}

static int layer_run(xe_engine *e, int layer_index, int rows) {
    xe_prefill_workspace workspace;
    size_t workspace_size = xe_prefill_workspace_layout(&workspace, NULL);
    void *memory = xe_alloc(e, workspace_size, XE_MEM_SHARED);
    xe_prefill_workspace_layout(&workspace, memory);
    for (int row = 0; row < rows; row++)
        xe_embed_decode(e, 2 + row,
                        workspace.hidden[0] + (size_t)row * XE_EMBD);
    int dimension = XE_IS_GLOBAL(layer_index)
                    ? XE_GLOBAL_HEAD_DIM : XE_SWA_HEAD_DIM;
    int q_width = XE_Q_HEADS * dimension;
    float *q_snapshot = xe_alloc(
        NULL, (size_t)rows * q_width * sizeof(*q_snapshot), XE_MEM_HOST);
    float *heads_snapshot = xe_alloc(
        NULL, (size_t)rows * q_width * sizeof(*heads_snapshot), XE_MEM_HOST);
    float *projection_snapshot = xe_alloc(
        NULL, (size_t)rows * XE_EMBD * sizeof(*projection_snapshot),
        XE_MEM_HOST);
    float *attention_snapshot = xe_alloc(
        NULL, (size_t)rows * XE_EMBD * sizeof(*attention_snapshot),
        XE_MEM_HOST);
    float *hidden_snapshot = xe_alloc(
        NULL, (size_t)rows * XE_EMBD * sizeof(*hidden_snapshot), XE_MEM_HOST);
    float *dense_snapshot = xe_alloc(
        NULL, (size_t)rows * XE_EMBD * sizeof(*dense_snapshot), XE_MEM_HOST);
    float *moe_snapshot = xe_alloc(
        NULL, (size_t)rows * XE_EMBD * sizeof(*moe_snapshot), XE_MEM_HOST);
    float *expert_snapshot = xe_alloc(
        NULL, (size_t)rows * XE_EXPERTS_USED * XE_EMBD
              * sizeof(*expert_snapshot), XE_MEM_HOST);
    float *expert_gate_snapshot = xe_alloc(
        NULL, (size_t)rows * XE_EXPERTS_USED * 2 * XE_EXPERT_FFN
              * sizeof(*expert_gate_snapshot), XE_MEM_HOST);
    int8_t *expert_activation_q_snapshot = xe_alloc(
        NULL, (size_t)rows * XE_EXPERTS_USED * XE_EXPERT_FFN,
        XE_MEM_HOST);
    _Float16 *expert_activation_d_snapshot = xe_alloc(
        NULL, (size_t)rows * XE_EXPERTS_USED * (XE_EXPERT_FFN / 32)
              * sizeof(*expert_activation_d_snapshot), XE_MEM_HOST);
    int16_t *expert_activation_s_snapshot = xe_alloc(
        NULL, (size_t)rows * XE_EXPERTS_USED * (XE_EXPERT_FFN / 32)
              * sizeof(*expert_activation_s_snapshot), XE_MEM_HOST);
    int *route_packed_snapshot = xe_alloc(
        NULL, (size_t)rows * XE_EXPERTS_USED
              * sizeof(*route_packed_snapshot), XE_MEM_HOST);
    int *packed_route_snapshot = xe_alloc(
        NULL, (size_t)rows * XE_EXPERTS_USED
              * sizeof(*packed_route_snapshot), XE_MEM_HOST);
    int *route_expert_snapshot = xe_alloc(
        NULL, (size_t)rows * XE_EXPERTS_USED
              * sizeof(*route_expert_snapshot), XE_MEM_HOST);
    float *route_weight_snapshot = xe_alloc(
        NULL, (size_t)rows * XE_EXPERTS_USED
              * sizeof(*route_weight_snapshot), XE_MEM_HOST);
    int8_t *packed_moe_q_snapshot = xe_alloc(
        NULL, (size_t)rows * XE_EXPERTS_USED * XE_EMBD, XE_MEM_HOST);
    _Float16 *packed_moe_d_snapshot = xe_alloc(
        NULL, (size_t)rows * XE_EXPERTS_USED * (XE_EMBD / 32)
              * sizeof(*packed_moe_d_snapshot), XE_MEM_HOST);
    int16_t *packed_moe_s_snapshot = xe_alloc(
        NULL, (size_t)rows * XE_EXPERTS_USED * (XE_EMBD / 32)
              * sizeof(*packed_moe_s_snapshot), XE_MEM_HOST);
    int *expert_count_snapshot = xe_alloc(
        NULL, XE_EXPERTS * sizeof(*expert_count_snapshot), XE_MEM_HOST);
    int *token_offset_snapshot = xe_alloc(
        NULL, (XE_EXPERTS + 1) * sizeof(*token_offset_snapshot), XE_MEM_HOST);
    int *tile_expert_snapshot = xe_alloc(
        NULL, 256 * sizeof(*tile_expert_snapshot), XE_MEM_HOST);
    int *tile_m0_snapshot = xe_alloc(
        NULL, 256 * sizeof(*tile_m0_snapshot), XE_MEM_HOST);
    double start = layer_now();
    xe_prefill_attention_initial_append(e, layer_index, &workspace, rows);
    xe_ze_check("zeCommandListHostSynchronize prefill layer attention",
                zeCommandListHostSynchronize(e->gpu.commands, UINT64_MAX));
    memcpy(q_snapshot, workspace.q_heads,
           (size_t)rows * q_width * sizeof(*q_snapshot));
    memcpy(heads_snapshot, workspace.attention_heads,
           (size_t)rows * q_width * sizeof(*heads_snapshot));
    memcpy(projection_snapshot, workspace.attention_projection,
           (size_t)rows * XE_EMBD * sizeof(*projection_snapshot));
    memcpy(attention_snapshot, workspace.attention_output,
           (size_t)rows * XE_EMBD * sizeof(*attention_snapshot));
    xe_prefill_ffn_append(e, layer_index, &workspace, rows);
    xe_ze_check("zeCommandListHostSynchronize prefill layer FFN",
                zeCommandListHostSynchronize(e->gpu.commands, UINT64_MAX));
    memcpy(hidden_snapshot, workspace.hidden[1],
           (size_t)rows * XE_EMBD * sizeof(*hidden_snapshot));
    memcpy(dense_snapshot, workspace.dense_down,
           (size_t)rows * XE_EMBD * sizeof(*dense_snapshot));
    memcpy(moe_snapshot, workspace.moe_output,
           (size_t)rows * XE_EMBD * sizeof(*moe_snapshot));
    memcpy(expert_snapshot, workspace.expert_down,
           (size_t)rows * XE_EXPERTS_USED * XE_EMBD
           * sizeof(*expert_snapshot));
    memcpy(expert_gate_snapshot, workspace.expert_gate_up,
           (size_t)rows * XE_EXPERTS_USED * 2 * XE_EXPERT_FFN
           * sizeof(*expert_gate_snapshot));
    memcpy(expert_activation_q_snapshot, workspace.expert_activation.qs,
           (size_t)rows * XE_EXPERTS_USED * XE_EXPERT_FFN);
    memcpy(expert_activation_d_snapshot, workspace.expert_activation.d,
           (size_t)rows * XE_EXPERTS_USED * (XE_EXPERT_FFN / 32)
           * sizeof(*expert_activation_d_snapshot));
    memcpy(expert_activation_s_snapshot, workspace.expert_activation.sigma,
           (size_t)rows * XE_EXPERTS_USED * (XE_EXPERT_FFN / 32)
           * sizeof(*expert_activation_s_snapshot));
    memcpy(route_packed_snapshot, workspace.routes.route_packed,
           (size_t)rows * XE_EXPERTS_USED * sizeof(*route_packed_snapshot));
    memcpy(packed_route_snapshot, workspace.routes.packed_route,
           (size_t)rows * XE_EXPERTS_USED * sizeof(*packed_route_snapshot));
    memcpy(route_expert_snapshot, workspace.route_expert,
           (size_t)rows * XE_EXPERTS_USED * sizeof(*route_expert_snapshot));
    memcpy(route_weight_snapshot, workspace.route_weight,
           (size_t)rows * XE_EXPERTS_USED * sizeof(*route_weight_snapshot));
    memcpy(packed_moe_q_snapshot, workspace.packed_moe.qs,
           (size_t)rows * XE_EXPERTS_USED * XE_EMBD);
    memcpy(packed_moe_d_snapshot, workspace.packed_moe.d,
           (size_t)rows * XE_EXPERTS_USED * (XE_EMBD / 32)
           * sizeof(*packed_moe_d_snapshot));
    memcpy(packed_moe_s_snapshot, workspace.packed_moe.sigma,
           (size_t)rows * XE_EXPERTS_USED * (XE_EMBD / 32)
           * sizeof(*packed_moe_s_snapshot));
    memcpy(expert_count_snapshot, workspace.routes.expert_count,
           XE_EXPERTS * sizeof(*expert_count_snapshot));
    memcpy(token_offset_snapshot, workspace.routes.token_offset,
           (XE_EXPERTS + 1) * sizeof(*token_offset_snapshot));
    memcpy(tile_expert_snapshot, workspace.routes.tile_expert,
           256 * sizeof(*tile_expert_snapshot));
    memcpy(tile_m0_snapshot, workspace.routes.tile_m0,
           256 * sizeof(*tile_m0_snapshot));
    double cold_ms = (layer_now() - start) * 1000.0;
    const int repetitions = rows == 512 ? 5 : 10;
    start = layer_now();
    for (int repetition = 0; repetition < repetitions; repetition++)
        xe_prefill_layer_initial_append(e, layer_index, &workspace, rows);
    xe_ze_check("zeCommandListHostSynchronize prefill layer timing",
                zeCommandListHostSynchronize(e->gpu.commands, UINT64_MAX));
    double warm_ms = (layer_now() - start) * 1000.0 / repetitions;
    int finite = 1;
    size_t repeat_mismatches = 0;
    size_t q_repeat_mismatches = 0;
    size_t heads_repeat_mismatches = 0;
    size_t projection_repeat_mismatches = 0;
    size_t attention_repeat_mismatches = 0;
    size_t dense_repeat_mismatches = 0;
    size_t moe_repeat_mismatches = 0;
    size_t expert_repeat_mismatches = 0;
    size_t expert_gate_repeat_mismatches = 0;
    size_t expert_activation_q_repeat_mismatches = 0;
    size_t expert_activation_d_repeat_mismatches = 0;
    size_t expert_activation_s_repeat_mismatches = 0;
    size_t route_packed_repeat_mismatches = 0;
    size_t packed_route_repeat_mismatches = 0;
    size_t route_expert_repeat_mismatches = 0;
    size_t route_weight_repeat_mismatches = 0;
    size_t packed_moe_q_repeat_mismatches = 0;
    size_t packed_moe_d_repeat_mismatches = 0;
    size_t packed_moe_s_repeat_mismatches = 0;
    size_t expert_count_repeat_mismatches = 0;
    size_t token_offset_repeat_mismatches = 0;
    size_t tile_expert_repeat_mismatches = 0;
    size_t tile_m0_repeat_mismatches = 0;
    for (size_t i = 0; i < (size_t)rows * q_width; i++) {
        q_repeat_mismatches += workspace.q_heads[i] != q_snapshot[i];
        heads_repeat_mismatches +=
            workspace.attention_heads[i] != heads_snapshot[i];
    }
    for (size_t i = 0; i < (size_t)rows * XE_EMBD; i++) {
        finite &= isfinite(workspace.hidden[1][i]);
        projection_repeat_mismatches +=
            workspace.attention_projection[i] != projection_snapshot[i];
        attention_repeat_mismatches +=
            workspace.attention_output[i] != attention_snapshot[i];
        dense_repeat_mismatches +=
            workspace.dense_down[i] != dense_snapshot[i];
        moe_repeat_mismatches +=
            workspace.moe_output[i] != moe_snapshot[i];
        repeat_mismatches += workspace.hidden[1][i] != hidden_snapshot[i];
    }
    for (size_t i = 0; i < (size_t)rows * XE_EXPERTS_USED * XE_EMBD; i++)
        expert_repeat_mismatches +=
            workspace.expert_down[i] != expert_snapshot[i];
    for (size_t i = 0;
         i < (size_t)rows * XE_EXPERTS_USED * 2 * XE_EXPERT_FFN; i++)
        expert_gate_repeat_mismatches +=
            workspace.expert_gate_up[i] != expert_gate_snapshot[i];
    for (size_t i = 0;
         i < (size_t)rows * XE_EXPERTS_USED * XE_EXPERT_FFN; i++)
        expert_activation_q_repeat_mismatches +=
            workspace.expert_activation.qs[i]
            != expert_activation_q_snapshot[i];
    for (size_t i = 0;
         i < (size_t)rows * XE_EXPERTS_USED * (XE_EXPERT_FFN / 32); i++)
        expert_activation_d_repeat_mismatches +=
            workspace.expert_activation.d[i]
            != expert_activation_d_snapshot[i];
    for (size_t i = 0;
         i < (size_t)rows * XE_EXPERTS_USED * (XE_EXPERT_FFN / 32); i++)
        expert_activation_s_repeat_mismatches +=
            workspace.expert_activation.sigma[i]
            != expert_activation_s_snapshot[i];
    for (size_t i = 0; i < (size_t)rows * XE_EXPERTS_USED; i++)
        route_packed_repeat_mismatches +=
            workspace.routes.route_packed[i] != route_packed_snapshot[i];
    for (size_t i = 0; i < (size_t)rows * XE_EXPERTS_USED; i++) {
        packed_route_repeat_mismatches +=
            workspace.routes.packed_route[i] != packed_route_snapshot[i];
        route_expert_repeat_mismatches +=
            workspace.route_expert[i] != route_expert_snapshot[i];
        route_weight_repeat_mismatches +=
            workspace.route_weight[i] != route_weight_snapshot[i];
    }
    for (size_t i = 0; i < (size_t)rows * XE_EXPERTS_USED * XE_EMBD; i++)
        packed_moe_q_repeat_mismatches +=
            workspace.packed_moe.qs[i] != packed_moe_q_snapshot[i];
    for (size_t i = 0;
         i < (size_t)rows * XE_EXPERTS_USED * (XE_EMBD / 32); i++)
        packed_moe_d_repeat_mismatches +=
            workspace.packed_moe.d[i] != packed_moe_d_snapshot[i];
    for (size_t i = 0;
         i < (size_t)rows * XE_EXPERTS_USED * (XE_EMBD / 32); i++)
        packed_moe_s_repeat_mismatches +=
            workspace.packed_moe.sigma[i] != packed_moe_s_snapshot[i];
    for (size_t i = 0; i < XE_EXPERTS; i++)
        expert_count_repeat_mismatches +=
            workspace.routes.expert_count[i] != expert_count_snapshot[i];
    for (size_t i = 0; i < XE_EXPERTS + 1; i++)
        token_offset_repeat_mismatches +=
            workspace.routes.token_offset[i] != token_offset_snapshot[i];
    for (size_t i = 0; i < 256; i++) {
        tile_expert_repeat_mismatches +=
            workspace.routes.tile_expert[i] != tile_expert_snapshot[i];
        tile_m0_repeat_mismatches +=
            workspace.routes.tile_m0[i] != tile_m0_snapshot[i];
    }
    int routes = rows * XE_EXPERTS_USED;
    int mapping_ok = layer_route_map_valid(route_expert_snapshot,
        route_packed_snapshot, packed_route_snapshot, expert_count_snapshot,
        token_offset_snapshot, routes)
        && layer_route_map_valid(workspace.route_expert,
        workspace.routes.route_packed, workspace.routes.packed_route,
        workspace.routes.expert_count, workspace.routes.token_offset, routes);
    size_t semantic_mismatches = 0;
    if (mapping_ok) {
#define CHECK_PACKED(before, after, width) \
        semantic_mismatches += layer_remapped_rows((before), (after), \
            route_packed_snapshot, workspace.routes.route_packed, routes, \
            (size_t)(width) * sizeof(*(before)))
        CHECK_PACKED(expert_snapshot, workspace.expert_down, XE_EMBD);
        CHECK_PACKED(expert_gate_snapshot, workspace.expert_gate_up, 2 * XE_EXPERT_FFN);
        CHECK_PACKED(expert_activation_q_snapshot, workspace.expert_activation.qs, XE_EXPERT_FFN);
        CHECK_PACKED(expert_activation_d_snapshot, workspace.expert_activation.d, XE_EXPERT_FFN/32);
        CHECK_PACKED(expert_activation_s_snapshot, workspace.expert_activation.sigma, XE_EXPERT_FFN/32);
        CHECK_PACKED(packed_moe_q_snapshot, workspace.packed_moe.qs, XE_EMBD);
        CHECK_PACKED(packed_moe_d_snapshot, workspace.packed_moe.d, XE_EMBD/32);
        CHECK_PACKED(packed_moe_s_snapshot, workspace.packed_moe.sigma, XE_EMBD/32);
#undef CHECK_PACKED
    }
    printf("prefill-layer: packed mapping %s semantic-row-mismatches %zu\n",
           mapping_ok ? "PASS" : "FAIL", semantic_mismatches);
    double error = 0.0;
    double attention_error = 0.0;
    double router_input_error = 0.0;
    double q_error = 0.0;
    double heads_error = 0.0;
    double projection_error = 0.0;
    double q_projection_error = 0.0;
    double q_post_error = 0.0;
    double q_reference_error = 0.0;
    double q_reference_row0_error = 0.0;
    double reference_projection_error = 0.0;
    int input_q8_mismatches = 0;
    int input_q8_delta = 0;
    int route_mismatches = 0;
    int route_set_mismatches = 0;
    float mismatched_margin = 0.0f;
    if (rows == 32) {
        xe_session *reference_state = xe_session_new(e);
        xe_workers_begin(e);
        float *reference_output = xe_alloc(
            NULL, (size_t)rows * XE_EMBD * sizeof(*reference_output),
            XE_MEM_HOST);
        float *reference_attention = xe_alloc(
            NULL, (size_t)rows * XE_EMBD * sizeof(*reference_attention),
            XE_MEM_HOST);
        float *reference_router = xe_alloc(
            NULL, (size_t)rows * XE_EMBD * sizeof(*reference_router),
            XE_MEM_HOST);
        float *reference_q = xe_alloc(
            NULL, (size_t)rows * q_width * sizeof(*reference_q), XE_MEM_HOST);
        float *q_post_reference = xe_alloc(
            NULL, (size_t)rows * q_width * sizeof(*q_post_reference),
            XE_MEM_HOST);
        float *reference_heads = xe_alloc(
            NULL, (size_t)rows * q_width * sizeof(*reference_heads),
            XE_MEM_HOST);
        float *reference_projection = xe_alloc(
            NULL, (size_t)rows * XE_EMBD * sizeof(*reference_projection),
            XE_MEM_HOST);
        int8_t reference_qs[XE_EMBD];
        _Float16 reference_d[XE_EMBD / 32];
        int16_t reference_sigma[XE_EMBD / 32];
        float normalized[XE_EMBD];
        float q_gpu_samples[256];
        float q_cpu_samples[256];
        float q_ref_samples[256];
        int q_samples = 0;
        const int q_sample_rows[8] = { 0, 1, 7, 8, 15, 16, 24, 31 };
        for (int sample_row = 0; sample_row < 8; sample_row++) {
            int row = q_sample_rows[sample_row];
            const float *input = workspace.hidden[0]
                                 + (size_t)row * XE_EMBD;
            float scale = xe_rms_scale_scalar(input, XE_EMBD);
            xe_rmsnorm_scale(input, e->layers[layer_index].attn_norm,
                             XE_EMBD, scale, normalized);
            xe_q8 reference_q8 = {
                reference_qs, reference_d, reference_sigma, XE_EMBD
            };
            xe_quantize_q8(normalized, XE_EMBD, &reference_q8);
            float raw_reference[XE_Q_HEADS * XE_GLOBAL_HEAD_DIM];
            ref_matvec_q4_q8(&e->layers[layer_index].attn_q, 0, XE_EMBD,
                             q_width, normalized, 1, raw_reference);
            xe_q8 gpu_q8 = {
                workspace.attention_input.qs + (size_t)row * XE_EMBD,
                workspace.attention_input.d
                    + (size_t)row * (XE_EMBD / 32),
                workspace.attention_input.sigma
                    + (size_t)row * (XE_EMBD / 32), XE_EMBD
            };
            for (int column = 0; column < XE_EMBD; column++) {
                int delta = abs((int)gpu_q8.qs[column]
                                - (int)reference_qs[column]);
                input_q8_mismatches += delta != 0;
                if (delta > input_q8_delta) input_q8_delta = delta;
            }
            for (int column = 0; column < q_width; column += 256) {
                q_gpu_samples[q_samples] = workspace.q_projection[
                    (size_t)row * q_width + column];
                q_cpu_samples[q_samples++] = xe_q4_q8_dot(
                    &e->layers[layer_index].attn_q, column, XE_EMBD,
                    &gpu_q8);
                q_ref_samples[q_samples - 1] = raw_reference[column];
            }
        }
        q_projection_error = layer_rel(q_gpu_samples, q_cpu_samples,
                                       q_samples);
        reference_projection_error = layer_rel(q_gpu_samples, q_ref_samples,
                                               q_samples);
        for (int row = 0; row < rows; row++)
            for (int head = 0; head < XE_Q_HEADS; head++) {
                const float *source = workspace.q_projection
                    + ((size_t)row * XE_Q_HEADS + head) * dimension;
                float scale = xe_rms_scale_scalar(source, dimension);
                size_t base = ((size_t)head * rows + row) * dimension;
                for (int d = 0; d < dimension / 2; d++) {
                    float lo = source[d] * scale
                               * e->layers[layer_index].q_norm[d];
                    float hi = source[d + dimension / 2] * scale
                               * e->layers[layer_index]
                                  .q_norm[d + dimension / 2];
                    const float *rope_cos = XE_IS_GLOBAL(layer_index)
                        ? workspace.rope_global_cos : workspace.rope_swa_cos;
                    const float *rope_sin = XE_IS_GLOBAL(layer_index)
                        ? workspace.rope_global_sin : workspace.rope_swa_sin;
                    float cosine = rope_cos[
                        (size_t)row * (dimension / 2) + d];
                    float sine = rope_sin[
                        (size_t)row * (dimension / 2) + d];
                    q_post_reference[base + d] = lo * cosine - hi * sine;
                    q_post_reference[base + d + dimension / 2] =
                        lo * sine + hi * cosine;
                }
            }
        for (int row = 0; row < rows; row++) {
            xe_embed_decode(e, 2 + row, reference_state->hidden);
            xe_rope_prepare(reference_state, row);
            xe_attention_half_decode(reference_state, layer_index, row);
            for (int head = 0; head < XE_Q_HEADS; head++) {
                memcpy(reference_q + ((size_t)head * rows + row) * dimension,
                       reference_state->q + (size_t)head * dimension,
                       (size_t)dimension * sizeof(float));
                memcpy(reference_heads
                           + ((size_t)head * rows + row) * dimension,
                       reference_state->attn_heads + (size_t)head * dimension,
                       (size_t)dimension * sizeof(float));
            }
            memcpy(reference_projection + (size_t)row * XE_EMBD,
                   reference_state->attn_proj, XE_EMBD * sizeof(float));
            memcpy(reference_attention + (size_t)row * XE_EMBD,
                   reference_state->attn_out, XE_EMBD * sizeof(float));
            xe_ffn_half_decode(reference_state, layer_index, XE_MOE_GENERIC_BATCHED, 1);
            memcpy(reference_output + (size_t)row * XE_EMBD,
                   reference_state->hidden, XE_EMBD * sizeof(float));
            memcpy(reference_router + (size_t)row * XE_EMBD,
                   reference_state->router_in, XE_EMBD * sizeof(float));
            float best_value[XE_EXPERTS_USED + 1];
            int best_index[XE_EXPERTS_USED + 1];
            for (int slot = 0; slot <= XE_EXPERTS_USED; slot++) {
                best_value[slot] = -INFINITY;
                best_index[slot] = -1;
            }
            for (int expert = 0; expert < XE_EXPERTS; expert++) {
                float value = reference_state->router_logits[expert];
                if (value <= best_value[XE_EXPERTS_USED]) continue;
                int slot = XE_EXPERTS_USED;
                while (slot > 0 && best_value[slot - 1] < value) {
                    best_value[slot] = best_value[slot - 1];
                    best_index[slot] = best_index[slot - 1];
                    slot--;
                }
                best_value[slot] = value;
                best_index[slot] = expert;
            }
            int row_mismatch = 0;
            int row_set_mismatch = 0;
            for (int slot = 0; slot < XE_EXPERTS_USED; slot++) {
                int gpu_expert = workspace.route_expert[
                    row * XE_EXPERTS_USED + slot];
                int mismatch = gpu_expert != best_index[slot];
                route_mismatches += mismatch;
                row_mismatch |= mismatch;
                int found = 0;
                for (int reference_slot = 0;
                     reference_slot < XE_EXPERTS_USED; reference_slot++)
                    found |= gpu_expert == best_index[reference_slot];
                route_set_mismatches += !found;
                row_set_mismatch |= !found;
            }
            if (row_mismatch && row_set_mismatch) {
                float margin = best_value[XE_EXPERTS_USED - 1]
                               - best_value[XE_EXPERTS_USED];
                if (margin > mismatched_margin) mismatched_margin = margin;
            }
        }
        error = layer_rel(workspace.hidden[1], reference_output,
                          (size_t)rows * XE_EMBD);
        attention_error = layer_rel(attention_snapshot,
                                    reference_attention,
                                    (size_t)rows * XE_EMBD);
        router_input_error = layer_rel(workspace.router_input,
                                       reference_router,
                                       (size_t)rows * XE_EMBD);
        q_error = layer_rel(q_snapshot, reference_q,
                            (size_t)rows * q_width);
        q_post_error = layer_rel(q_snapshot, q_post_reference,
                                 (size_t)rows * q_width);
        q_reference_error = layer_rel(q_post_reference, reference_q,
                                      (size_t)rows * q_width);
        float q_post_row0[XE_Q_HEADS * XE_GLOBAL_HEAD_DIM];
        float q_reference_row0[XE_Q_HEADS * XE_GLOBAL_HEAD_DIM];
        for (int head = 0; head < XE_Q_HEADS; head++) {
            memcpy(q_post_row0 + (size_t)head * dimension,
                   q_post_reference + (size_t)head * rows * dimension,
                   (size_t)dimension * sizeof(float));
            memcpy(q_reference_row0 + (size_t)head * dimension,
                   reference_q + (size_t)head * rows * dimension,
                   (size_t)dimension * sizeof(float));
        }
        q_reference_row0_error = layer_rel(q_post_row0, q_reference_row0,
                                           q_width);
        heads_error = layer_rel(heads_snapshot, reference_heads,
                                (size_t)rows * q_width);
        double projection_difference = 0.0;
        double projection_reference = 0.0;
        for (size_t i = 0; i < (size_t)rows * XE_EMBD; i++) {
            double got = (double)attention_snapshot[i]
                         - workspace.hidden[0][i];
            double want = reference_projection[i];
            double difference = got - want;
            projection_difference += difference * difference;
            projection_reference += want * want;
        }
        projection_error = sqrt(projection_difference
                                / (projection_reference + 1e-30));
        xe_free(NULL, reference_projection, XE_MEM_HOST);
        xe_free(NULL, reference_heads, XE_MEM_HOST);
        xe_free(NULL, reference_q, XE_MEM_HOST);
        xe_free(NULL, q_post_reference, XE_MEM_HOST);
        xe_free(NULL, reference_router, XE_MEM_HOST);
        xe_free(NULL, reference_attention, XE_MEM_HOST);
        xe_free(NULL, reference_output, XE_MEM_HOST);
        xe_workers_end(e);
        xe_session_free(reference_state);
    }
    int routes_ok = route_set_mismatches == 0 || mismatched_margin < 5e-4f;
    int ok = finite && repeat_mismatches == 0
             && q_repeat_mismatches == 0 && heads_repeat_mismatches == 0
             && projection_repeat_mismatches == 0 && attention_repeat_mismatches == 0
             && dense_repeat_mismatches == 0 && moe_repeat_mismatches == 0
             && mapping_ok && semantic_mismatches == 0
             && route_expert_repeat_mismatches == 0 && route_weight_repeat_mismatches == 0
             && expert_count_repeat_mismatches == 0
             && token_offset_repeat_mismatches == 0 && tile_expert_repeat_mismatches == 0
             && tile_m0_repeat_mismatches == 0
             && (rows != 32 || (routes_ok && error < 5e-3));
    printf("prefill-layer-sigma: expert %zu packed %zu\n",
           expert_activation_s_repeat_mismatches,
           packed_moe_s_repeat_mismatches);
    printf("prefill-layer: L%d %s M%d arena %.3f MiB cold %.6f ms warm %.6f ms repeat %zu q-repeat %zu heads-repeat %zu projection-repeat %zu attention-repeat %zu dense-repeat %zu expert-gate-repeat %zu expert-act-q-repeat %zu expert-act-d-repeat %zu expert-repeat %zu route-expert-repeat %zu route-weight-repeat %zu packed-route-repeat %zu route-packed-repeat %zu packed-moe-q-repeat %zu packed-moe-d-repeat %zu expert-count-repeat %zu token-offset-repeat %zu tile-expert-repeat %zu tile-m0-repeat %zu moe-repeat %zu input-q8 %d delta %d q-projection %.3e ref-projection %.3e q-post %.3e q-reference %.3e/%.3e q %.3e heads %.3e projection %.3e attention %.3e router-input %.3e routes %d/%d margin %.3e rel %.3e %s\n",
           layer_index, XE_IS_GLOBAL(layer_index) ? "global" : "swa", rows,
           (double)workspace_size / (1024.0 * 1024.0), cold_ms, warm_ms,
           repeat_mismatches, q_repeat_mismatches, heads_repeat_mismatches,
           projection_repeat_mismatches, attention_repeat_mismatches,
           dense_repeat_mismatches, expert_gate_repeat_mismatches,
           expert_activation_q_repeat_mismatches,
           expert_activation_d_repeat_mismatches,
           expert_repeat_mismatches,
           route_expert_repeat_mismatches, route_weight_repeat_mismatches,
           packed_route_repeat_mismatches, route_packed_repeat_mismatches,
           packed_moe_q_repeat_mismatches, packed_moe_d_repeat_mismatches,
           expert_count_repeat_mismatches, token_offset_repeat_mismatches,
           tile_expert_repeat_mismatches, tile_m0_repeat_mismatches,
           moe_repeat_mismatches,
           input_q8_mismatches, input_q8_delta, q_projection_error,
           reference_projection_error, q_post_error, q_reference_error,
           q_reference_row0_error, q_error, heads_error,
           projection_error, attention_error,
           router_input_error, route_mismatches, route_set_mismatches,
           (double)mismatched_margin,
           error,
           ok ? "PASS" : "FAIL");
    xe_free(NULL, attention_snapshot, XE_MEM_HOST);
    xe_free(NULL, route_packed_snapshot, XE_MEM_HOST);
    xe_free(NULL, packed_moe_d_snapshot, XE_MEM_HOST);
    xe_free(NULL, packed_moe_s_snapshot, XE_MEM_HOST);
    xe_free(NULL, tile_m0_snapshot, XE_MEM_HOST);
    xe_free(NULL, tile_expert_snapshot, XE_MEM_HOST);
    xe_free(NULL, token_offset_snapshot, XE_MEM_HOST);
    xe_free(NULL, expert_count_snapshot, XE_MEM_HOST);
    xe_free(NULL, packed_moe_q_snapshot, XE_MEM_HOST);
    xe_free(NULL, route_weight_snapshot, XE_MEM_HOST);
    xe_free(NULL, route_expert_snapshot, XE_MEM_HOST);
    xe_free(NULL, packed_route_snapshot, XE_MEM_HOST);
    xe_free(NULL, expert_activation_d_snapshot, XE_MEM_HOST);
    xe_free(NULL, expert_activation_s_snapshot, XE_MEM_HOST);
    xe_free(NULL, expert_activation_q_snapshot, XE_MEM_HOST);
    xe_free(NULL, expert_gate_snapshot, XE_MEM_HOST);
    xe_free(NULL, expert_snapshot, XE_MEM_HOST);
    xe_free(NULL, moe_snapshot, XE_MEM_HOST);
    xe_free(NULL, dense_snapshot, XE_MEM_HOST);
    xe_free(NULL, hidden_snapshot, XE_MEM_HOST);
    xe_free(NULL, projection_snapshot, XE_MEM_HOST);
    xe_free(NULL, heads_snapshot, XE_MEM_HOST);
    xe_free(NULL, q_snapshot, XE_MEM_HOST);
    xe_free(e, memory, XE_MEM_SHARED);
    return ok;
}

int main(int argc, char **argv) {
    if (argc == 2 && !strcmp(argv[1], "--mapping-self-test"))
        return layer_mapping_self_test() ? 0 : 1;
    if (argc < 2 || !argv[1][0]) {
        fprintf(stderr, "usage: %s <model.gguf> [rows]\n", argv[0]);
        return 2;
    }
    const char *model = argv[1];
    int rows = argc > 2 ? (int)strtol(argv[2], NULL, 10) : 32;
    if (rows < 1 || rows > 512)
        xe_fatal("prefill layer test supports M1 through M512");
    /* This isolated block uses only positions [0, rows). Bound the decode
     * comparison cache instead of allocating the CLI's default 64K context. */
    xe_engine_options options = { .context = rows < XE_CONTEXT_MIN ? XE_CONTEXT_MIN : rows };
    xe_engine *e = xe_engine_open_with_options(model, &options);
    ze_kernel_properties_t properties = {
        .stype = ZE_STRUCTURE_TYPE_KERNEL_PROPERTIES
    };
    xe_ze_check("zeKernelGetProperties prefill grouped m16",
                zeKernelGetProperties(e->gpu.prefill_q4q8_grouped_m16_n64,
                                      &properties));
    printf("prefill-layer: m16 local %u private %u spill %u\n",
           properties.localMemSize, properties.privateMemSize,
           properties.spillMemSize);
    int ok = layer_run(e, 0, rows) & layer_run(e, 5, rows);
    xe_engine_close(e);
    return ok ? 0 : 1;
}
