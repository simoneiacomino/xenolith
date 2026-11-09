/* Real-model packing, shared allocations and isolated FFN operator contracts.
 * Full-model numerical fidelity is checked by tools/quality. */
#include "../xenolith.c"
#include "support/scalar_ops.h"

static int model_shared_allocation(const xe_engine *e, const void *p) {
    ze_memory_allocation_properties_t properties = {
        .stype = ZE_STRUCTURE_TYPE_MEMORY_ALLOCATION_PROPERTIES
    };
    ze_device_handle_t device = NULL;
    return zeMemGetAllocProperties(e->gpu.context, p, &properties, &device) ==
               ZE_RESULT_SUCCESS &&
           properties.type == ZE_MEMORY_TYPE_SHARED && device == e->gpu.device;
}

static int model_gpu_memory(const xe_engine *e) {
    uint64_t total = 0;
    int ok = e->repack_part_count > 0 && e->repack_part_count <= XE_REPACK_PARTS;
    for (int i = 0; i < e->repack_part_count; i++) {
        ok = ok && e->repack_part_sizes[i] > 0;
        ok = ok && e->repack_part_sizes[i] <= XE_REPACK_PART_LIMIT;
        ok = ok && model_shared_allocation(e, e->repack_parts[i]);
        total += e->repack_part_sizes[i];
    }
    ok = ok && total == e->repack_slab_size;
    ok = ok && e->f32_slab_size > 0;
    ok = ok && e->f32_slab_size <= XE_REPACK_PART_LIMIT;
    ok = ok && model_shared_allocation(e, e->f32_slab);
    printf("model: GPU memory Q4 parts %d F32 %.2f MiB %s\n",
           e->repack_part_count, e->f32_slab_size / 1048576.0,
           ok ? "PASS" : "FAIL");
    return ok;
}


static double model_rel(const float *got, const float *want, int n) {
    double delta = 0.0, energy = 0.0;
    for (int i = 0; i < n; i++) {
        double d = (double)got[i] - want[i];
        delta += d*d;
        energy += (double)want[i]*want[i];
    }
    return sqrt(delta / (energy + 1e-30));
}

static int model_q8_equal(const xe_q8 *got, const float *x, int n) {
    int8_t qs[REF_Q8_MAX_IN];
    float ds[REF_Q8_MAX_IN / 32];
    ref_quantize_q8(x, n, qs, ds);
    int ok = 1;
    for (int i = 0; i < n; i++) ok &= got->qs[i] == qs[i];
    for (int b = 0; b < n / 32; b++) {
        int sigma = 0;
        for (int j = 0; j < 32; j++) sigma += qs[32*b+j];
        ok &= (float)got->d[b] == ds[b] && got->sigma[b] == sigma;
    }
    return ok;
}

static int model_layer_operators(xe_engine *e, xe_session *s, int layer) {
    const xe_layer *l = &e->layers[layer];
    /* Every layer sees the same independent input: there is no reference
     * hidden-state chain, KV model, embedding forward or output head here. */
    for (int i = 0; i < XE_EMBD; i++)
        s->attn_out[i] = (float)((i * 17 % 257) - 128) / 128.0f;
    float dense[XE_EMBD], moe[XE_EMBD], router[XE_EMBD];
    ref_rmsnorm(s->attn_out, l->ffn_norm, XE_EMBD, dense);
    ref_rmsnorm(s->attn_out, l->pre_ffw_norm2, XE_EMBD, moe);
    ref_rmsnorm_weightless(s->attn_out, XE_EMBD, router);
    for (int i = 0; i < XE_EMBD; i++)
        router[i] = router[i] / sqrtf((float)XE_EMBD) * l->router_scale[i];
    xe_ffn_half_decode(s, layer, XE_MOE_GENERIC_BATCHED, 1);
    int ok = model_q8_equal(&s->q8_dense_in, dense, XE_EMBD)
             && model_q8_equal(&s->q8_moe_in, moe, XE_EMBD);
    float probabilities[XE_EXPERTS];
    ref_matvec_f32(l->router_w, XE_EMBD, XE_EXPERTS, router, probabilities);
    ref_softmax(probabilities, XE_EXPERTS);
    int experts[XE_EXPERTS_USED + 1];
    float weights[XE_EXPERTS_USED + 1];
    ref_top_k(probabilities, XE_EXPERTS, XE_EXPERTS_USED + 1, experts, weights);
    float max_delta = 0.0f;
    for (int i = 0; i < XE_EXPERTS; i++) {
        float d = fabsf(probabilities[i] - s->router_logits[i]);
        if (d > max_delta) max_delta = d;
        ok &= isfinite(s->router_logits[i]);
    }
    /* A changed selected set is valid only inside the directly measured
     * probability perturbation; preserve attribution without a whole model. */
    for (int slot = 0; slot < XE_EXPERTS_USED; slot++) {
        int selected = s->experts[slot], found = 0;
        if (selected < 0 || selected >= XE_EXPERTS) { ok = 0; continue; }
        for (int j = 0; j < XE_EXPERTS_USED; j++) found |= selected == experts[j];
        if (!found) ok &= weights[XE_EXPERTS_USED-1] - probabilities[selected]
                          <= 2.0f * max_delta + 1e-9f;
        ok &= s->expert_trace[layer][slot] == selected;
        ok &= s->expert_weight_trace[layer][slot] == s->expert_weights[slot];
        if (slot) ok &= s->expert_weights[slot-1] >= s->expert_weights[slot];
        for (int j = 0; j < slot; j++) ok &= selected != s->experts[j];
    }
    float sum = 0.0f;
    for (int slot = 0; slot < XE_EXPERTS_USED; slot++)
        sum += s->router_logits[s->experts[slot]];
    if (sum < ref_moe_weight_sum_min) sum = ref_moe_weight_sum_min;
    for (int slot = 0; slot < XE_EXPERTS_USED; slot++)
        ok &= fabsf(s->expert_weights[slot]
                    - s->router_logits[s->experts[slot]] / sum) < 2e-7f;

    int8_t input_qs[XE_EMBD];
    float input_ds[XE_EMBD/32];
    ref_quantize_q8(dense, XE_EMBD, input_qs, input_ds);
    float activation[XE_DENSE_FFN];
    for (int row = 0; row < XE_DENSE_FFN; row++) {
        float gate = ref_dot_q4_q8_row_v1(&l->ffn_gate, row, XE_EMBD, input_qs, input_ds);
        float up = ref_dot_q4_q8_row_v1(&l->ffn_up, row, XE_EMBD, input_qs, input_ds);
        activation[row] = ref_gelu_fp16(gate) * up;
    }
    ok &= model_q8_equal(&s->q8_dense_act, activation, XE_DENSE_FFN);
    int8_t activation_qs[XE_DENSE_FFN];
    float activation_ds[XE_DENSE_FFN/32];
    float dense_output[XE_EMBD], moe_output[XE_EMBD] = {0};
    ref_quantize_q8(activation, XE_DENSE_FFN, activation_qs, activation_ds);
    for (int row = 0; row < XE_EMBD; row++)
        dense_output[row] = ref_dot_q4_q8_row_v1(&l->ffn_down, row,
                             XE_DENSE_FFN, activation_qs, activation_ds);
    ref_rmsnorm(dense_output, l->post_ffw_norm1, XE_EMBD, dense_output);
    double dense_error = model_rel(s->dense_out, dense_output, XE_EMBD);
    ok &= dense_error < 1e-4;
    /* Independently reconstruct every selected expert activation. */
    ref_quantize_q8(moe, XE_EMBD, input_qs, input_ds);
    for (int slot = 0; slot < XE_EXPERTS_USED; slot++) {
        uint64_t first = (uint64_t)s->experts[slot] * 2 * XE_EXPERT_FFN;
        for (int row = 0; row < XE_EXPERT_FFN; row++) {
            float gate = ref_dot_q4_q8_row_v1(&l->gate_up_exps, first + row,
                                              XE_EMBD, input_qs, input_ds);
            float up = ref_dot_q4_q8_row_v1(&l->gate_up_exps, first + XE_EXPERT_FFN + row,
                                            XE_EMBD, input_qs, input_ds);
            activation[row] = ref_gelu_fp16(gate) * up;
        }
        ok &= model_q8_equal(&s->q8_expert[slot], activation, XE_EXPERT_FFN);
        ref_quantize_q8(activation, XE_EXPERT_FFN, activation_qs, activation_ds);
        int expert = s->experts[slot];
        for (int row = 0; row < XE_EMBD; row++) {
            float value = ref_dot_q4_q8_row_v1(&l->down_exps,
                (uint64_t)expert * XE_EMBD + row, XE_EXPERT_FFN,
                activation_qs, activation_ds);
            moe_output[row] += s->expert_weights[slot]
                               * (value * l->down_exps_scale[expert]);
        }
    }
    ref_rmsnorm(moe_output, l->post_ffw_norm2, XE_EMBD, moe_output);
    double moe_error = model_rel(s->moe_out, moe_output, XE_EMBD);
    ok &= moe_error < 1e-4;
    float combined[XE_EMBD];
    for (int i = 0; i < XE_EMBD; i++) combined[i] = dense_output[i] + moe_output[i];
    ref_rmsnorm(combined, l->post_ffw_norm, XE_EMBD, combined);
    for (int i = 0; i < XE_EMBD; i++)
        combined[i] = (combined[i] + s->attn_out[i]) * l->layer_out_scale[0];
    double hidden_error = model_rel(s->hidden, combined, XE_EMBD);
    ok &= hidden_error < 1e-4;
    for (int i = 0; i < XE_EMBD; i++)
        ok &= isfinite(s->dense_out[i]) && isfinite(s->moe_out[i]) && isfinite(s->hidden[i]);
    printf("model: L%02d scalar packing/router/activation dense %.2e moe %.2e residual %.2e %s\n",
           layer, dense_error, moe_error, hidden_error, ok ? "PASS" : "FAIL");
    return ok;
}

int main(int argc, char **argv) {
    if (argc != 2 || !argv[1][0]) {
        fprintf(stderr, "usage: %s <model.gguf>\n", argv[0]);
        return 2;
    }
    xe_engine_options options = { .context = XE_CONTEXT_MIN };
    xe_engine *e = xe_engine_open_with_options(argv[1], &options);
    xe_session *s = xe_session_new(e);
    int ok = model_gpu_memory(e);
    float embedding[XE_EMBD];
    ref_dequant_row(&e->tok_embd, 2, XE_EMBD, embedding);
    xe_embed_decode(e, 2, s->hidden);
    for (int i = 0; i < XE_EMBD; i++) {
        float expected = embedding[i] * sqrtf((float)XE_EMBD);
        ok &= fabsf(s->hidden[i] - expected) <= 2e-6f * fmaxf(1.0f, fabsf(expected));
    }
    xe_workers_begin(e);
    for (int layer = 0; layer < XE_LAYERS; layer++)
        ok &= model_layer_operators(e, s, layer);
    xe_workers_end(e);
    xe_session_free(s);
    xe_engine_close(e);
    return ok ? 0 : 1;
}
