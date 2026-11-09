#include "../xenolith.c"
#include "support/scalar_ops.h"

static uint16_t test_half_bits(_Float16 x) {
    uint16_t bits;
    memcpy(&bits, &x, sizeof bits);
    return bits;
}

static double test_rel_rms_decode(const float *a, const float *b, int n) {
    double error = 0.0;
    double reference = 0.0;
    for (int i = 0; i < n; i++) {
        double d = (double)a[i] - b[i];
        error += d * d;
        reference += (double)b[i] * b[i];
    }
    return sqrt(error / (reference + 1e-30));
}

static float test_value(int i) {
    return (float)(((i * 104729 + 8191) % 65521) - 32760) * 0.00037f;
}

static void test_q8_ref(const float *x, int n, xe_q8 *q) {
    for (int b = 0; b < n / 32; b++) {
        float amax = 0.0f;
        for (int i = 0; i < 32; i++) {
            float a = fabsf(x[32 * b + i]);
            if (a > amax) amax = a;
        }
        float d = amax / 127.0f;
        float id = d ? 1.0f / d : 0.0f;
        int sum = 0;
        q->d[b] = (_Float16)d;
        for (int i = 0; i < 32; i++) {
            int v = (int)roundf(x[32 * b + i] * id);
            q->qs[32 * b + i] = (int8_t)v;
            sum += v;
        }
        q->sigma[b] = (int16_t)sum;
    }
}

static xe_q8 test_q8_alloc(int n) {
    xe_q8 q;
    q.qs = xe_alloc(NULL, (size_t)n, XE_MEM_HOST);
    q.d = xe_alloc(NULL, (size_t)(n / 32) * sizeof(*q.d), XE_MEM_HOST);
    q.sigma = xe_alloc(NULL, (size_t)(n / 32) * sizeof(*q.sigma), XE_MEM_HOST);
    q.n = n;
    return q;
}

static void test_q8_free(xe_q8 *q) {
    xe_free(NULL, q->sigma, XE_MEM_HOST);
    xe_free(NULL, q->d, XE_MEM_HOST);
    xe_free(NULL, q->qs, XE_MEM_HOST);
}

static int test_q8_contract(void) {
    int n = XE_EMBD;
    float *x = xe_alloc(NULL, (size_t)n * sizeof(*x), XE_MEM_HOST);
    xe_q8 got = test_q8_alloc(n);
    xe_q8 ref = test_q8_alloc(n);
    for (int i = 0; i < n; i++) x[i] = test_value(i);
    for (int i = 0; i < 32; i++) x[32 * 17 + i] = 0.0f;
    xe_quantize_q8(x, n, &got);
    test_q8_ref(x, n, &ref);
    int ok = memcmp(got.qs, ref.qs, (size_t)n) == 0;
    ok = ok && memcmp(got.d, ref.d, (size_t)(n / 32) * sizeof(*got.d)) == 0;
    ok = ok && memcmp(got.sigma, ref.sigma, (size_t)(n / 32) * sizeof(*got.sigma)) == 0;
    for (int i = 0; i < 32; i++) ok = ok && got.qs[32 * 17 + i] == 0;
    ok = ok && test_half_bits(got.d[17]) == 0 && got.sigma[17] == 0;
    printf("decode: q8 contract %s\n", ok ? "PASS" : "FAIL");
    test_q8_free(&ref);
    test_q8_free(&got);
    xe_free(NULL, x, XE_MEM_HOST);
    return ok;
}

static float test_q4_q8_ref(const xe_q4 *w, int row, int n, const xe_q8 *q) {
    int blocks = n / 32;
    uint64_t first = (uint64_t)row * blocks;
    float sum = 0.0f;
    for (int b = 0; b < blocks; b++) {
        uint64_t block = first + (uint64_t)b;
        int integer = 0;
        for (int i = 0; i < 16; i++) {
            uint8_t packed = w->qs[16 * block + i];
            integer += ((packed & 15) - 8) * q->qs[32 * b + i];
            integer += ((packed >> 4) - 8) * q->qs[32 * b + i + 16];
        }
        sum += (float)integer * _cvtsh_ss(w->d[block]) * (float)q->d[b];
    }
    return sum;
}

static int test_v1_contract(void) {
    int n = XE_EMBD;
    int rows = 40;
    int blocks = n / 32;
    float *x = xe_alloc(NULL, (size_t)n * sizeof(*x), XE_MEM_HOST);
    uint8_t *nibbles = xe_alloc(NULL, (size_t)rows * blocks * 16, XE_MEM_HOST);
    uint16_t *scales = xe_alloc(NULL, (size_t)rows * blocks * sizeof(*scales), XE_MEM_HOST);
    uint8_t *packed_nibbles = xe_alloc(NULL, (size_t)rows * blocks * 16, XE_MEM_HOST);
    uint16_t *packed_scales = xe_alloc(NULL, (size_t)rows * blocks * sizeof(*packed_scales), XE_MEM_HOST);
    uint8_t *native = xe_alloc(NULL, (size_t)rows * blocks * 18, XE_MEM_HOST);
    float *got = xe_alloc(NULL, (size_t)rows * sizeof(*got), XE_MEM_HOST);
    float *ref = xe_alloc(NULL, (size_t)rows * sizeof(*ref), XE_MEM_HOST);
    float *ordered = xe_alloc(NULL, (size_t)rows * sizeof(*ordered), XE_MEM_HOST);
    float *qds = xe_alloc(NULL, (size_t)blocks * sizeof(*qds), XE_MEM_HOST);
    xe_q8 q = test_q8_alloc(n);
    for (int i = 0; i < n; i++) x[i] = test_value(i + 71);
    for (int i = 0; i < rows * blocks * 16; i++)
        nibbles[i] = (uint8_t)(((i * 13 + 7) & 15) | (((i * 29 + 3) & 15) << 4));
    for (int i = 0; i < rows * blocks; i++) {
        _Float16 d = (_Float16)(0.0005f + (float)(i % 97) * 0.00003f);
        scales[i] = test_half_bits(d);
    }
    for (int block = 0; block < rows * blocks; block++) {
        memcpy(native + (size_t)block * 18, &scales[block], 2);
        memcpy(native + (size_t)block * 18 + 2, nibbles + (size_t)block * 16, 16);
    }
    xe_q4 w = { .qs = native };
    xe_repack_job job = {
        .field = &w, .nblocks = (uint64_t)rows * blocks, .blocks = blocks,
        .rows = (uint64_t)rows, .suffix = "test.weight", .layer_idx = -1,
        .qs_seg = packed_nibbles, .d_seg = packed_scales
    };
    atomic_int next;
    atomic_init(&next, 0);
    xe_repack_worker_arg repack = { &job, 1, &next, "test", 0 };
    xe_repack_worker(&repack);
    xe_q4 row_major = { nibbles, scales, 0 };
    xe_quantize_q8(x, n, &q);
    xe_matvec_q4_q8_rows(&w, 0, n, 0, rows, &q, got);
    for (int b = 0; b < blocks; b++) qds[b] = (float)q.d[b];
    for (int row = 0; row < rows; row++) ref[row] = test_q4_q8_ref(&row_major, row, n, &q);
    for (int row = 0; row < rows; row++)
        ordered[row] = ref_dot_q4_q8_row_v1(&w, (uint64_t)row, n, q.qs, qds);
    double rel = test_rel_rms_decode(got, ref, rows);
    int exact = memcmp(got, ordered, (size_t)rows * sizeof(*got)) == 0;
    int repack_ok = repack.verified == 3 && w.blocks == blocks;
    int ok = rel < 2e-6 && exact && repack_ok;
    printf("decode: V1 rel_rms=%.3e ordered=%s repack=%s %s\n", rel,
           exact ? "exact" : "different", repack_ok ? "verified" : "FAIL",
           ok ? "PASS" : "FAIL");
    test_q8_free(&q);
    xe_free(NULL, qds, XE_MEM_HOST);
    xe_free(NULL, ordered, XE_MEM_HOST);
    xe_free(NULL, ref, XE_MEM_HOST);
    xe_free(NULL, got, XE_MEM_HOST);
    xe_free(NULL, packed_scales, XE_MEM_HOST);
    xe_free(NULL, packed_nibbles, XE_MEM_HOST);
    xe_free(NULL, native, XE_MEM_HOST);
    xe_free(NULL, scales, XE_MEM_HOST);
    xe_free(NULL, nibbles, XE_MEM_HOST);
    xe_free(NULL, x, XE_MEM_HOST);
    return ok;
}

static int test_x8_shape(int n, int seed) {
    const int rows = 8;
    int blocks = n / 32;
    float *x = xe_alloc(NULL, (size_t)n * sizeof(*x), XE_MEM_HOST);
    uint8_t *native = xe_alloc(NULL, (size_t)rows * blocks * 18, XE_MEM_HOST);
    uint8_t *packed = xe_alloc(NULL, (size_t)rows * blocks * 16, XE_MEM_HOST);
    uint16_t *scales = xe_alloc(NULL, (size_t)rows * blocks * sizeof(*scales), XE_MEM_HOST);
    float qds[256];
    float got[8], ordered[8];
    xe_q8 q = test_q8_alloc(n);
    for (int i = 0; i < n; i++) x[i] = test_value(i + seed);
    for (int block = 0; block < rows * blocks; block++) {
        _Float16 d = (_Float16)(0.0007f + (float)((block + seed) % 113) * 0.00002f);
        uint16_t bits = test_half_bits(d);
        memcpy(native + (size_t)block * 18, &bits, 2);
        for (int i = 0; i < 16; i++)
            native[(size_t)block * 18 + 2 + i] =
                (uint8_t)((((block * 17 + i * 7 + seed) & 15)) |
                          (((block * 11 + i * 13 + seed) & 15) << 4));
    }
    xe_q4 w = { .qs = native };
    xe_repack_job job = {
        .field = &w, .nblocks = (uint64_t)rows * blocks, .blocks = blocks,
        .rows = rows, .suffix = "shape.weight", .layer_idx = -1,
        .qs_seg = packed, .d_seg = scales
    };
    atomic_int next;
    atomic_init(&next, 0);
    xe_repack_worker_arg repack = { &job, 1, &next, "shape", 0 };
    xe_repack_worker(&repack);
    xe_quantize_q8(x, n, &q);
    for (int b = 0; b < blocks; b++) qds[b] = (float)q.d[b];
    xe_q4_q8_dot8(&w, 0, n, &q, got);
    for (int row = 0; row < rows; row++)
        ordered[row] = ref_dot_q4_q8_row_v1(&w, (uint64_t)row, n, q.qs, qds);
    int ok = repack.verified == 3 && memcmp(got, ordered, sizeof got) == 0;
    test_q8_free(&q);
    xe_free(NULL, scales, XE_MEM_HOST);
    xe_free(NULL, packed, XE_MEM_HOST);
    xe_free(NULL, native, XE_MEM_HOST);
    xe_free(NULL, x, XE_MEM_HOST);
    return ok;
}

static int test_x8_shapes(void) {
    static const int shapes[] = { 704, 2112, 2816, 4096, 8192 };
    int ok = 1;
    for (size_t i = 0; i < sizeof shapes / sizeof shapes[0]; i++)
        ok = test_x8_shape(shapes[i], 101 + (int)i * 29) && ok;
    printf("decode: x8 specialized shapes ordered=exact %s\n", ok ? "PASS" : "FAIL");
    return ok;
}

static float test_gelu(float x) {
    if (x <= -10.0f) return 0.0f;
    if (x >= 10.0f) return x;
    float inner = 0.79788456080286535588f * (x + 0.044715f * x * x * x);
    return (float)(_Float16)(0.5f * x * (1.0f + tanhf(inner)));
}

static int test_gelu_contract(void) {
    xe_engine e;
    memset(&e, 0, sizeof e);
    xe_constants_init(&e);
    int ok = 1;
    for (uint32_t bits = 0; bits < 65536; bits++) {
        uint16_t hbits = (uint16_t)bits;
        _Float16 h;
        memcpy(&h, &hbits, sizeof h);
        _Float16 want = (_Float16)test_gelu((float)h);
        if (test_half_bits(e.gelu_lut[bits]) != test_half_bits(want)) ok = 0;
    }
    ok = ok && xe_gelu_lookup(&e, 12.34567f) == 12.34567f;
    ok = ok && xe_gelu_lookup(&e, -12.34567f) == 0.0f;
    printf("decode: GELU table %s\n", ok ? "PASS" : "FAIL");
    xe_free(NULL, e.gelu_lut, XE_MEM_HOST);
    return ok;
}

static int test_norm_rope(void) {
    float x[XE_EMBD];
    float weight[XE_EMBD];
    float got[XE_EMBD];
    float ref[XE_EMBD];
    for (int i = 0; i < XE_EMBD; i++) {
        x[i] = test_value(i + 113);
        weight[i] = 0.5f + (float)(i % 101) * 0.002f;
    }
    xe_rmsnorm(x, weight, XE_EMBD, got);
    float sum = 0.0f;
    for (int i = 0; i < XE_EMBD; i++) sum += x[i] * x[i];
    float scale = 1.0f / sqrtf(sum / XE_EMBD + XE_RMS_EPS);
    for (int i = 0; i < XE_EMBD; i++) ref[i] = x[i] * scale * weight[i];
    double norm_rel = test_rel_rms_decode(got, ref, XE_EMBD);

    xe_engine e;
    xe_session s;
    memset(&e, 0, sizeof e);
    memset(&s, 0, sizeof s);
    s.engine = &e;
    xe_constants_init(&e);
    s.rope_swa_cos = xe_alloc(NULL, (XE_SWA_HEAD_DIM / 2) * sizeof(float), XE_MEM_HOST);
    s.rope_swa_sin = xe_alloc(NULL, (XE_SWA_HEAD_DIM / 2) * sizeof(float), XE_MEM_HOST);
    s.rope_global_cos = xe_alloc(NULL, (XE_GLOBAL_HEAD_DIM / 2) * sizeof(float), XE_MEM_HOST);
    s.rope_global_sin = xe_alloc(NULL, (XE_GLOBAL_HEAD_DIM / 2) * sizeof(float), XE_MEM_HOST);
    float *heads = xe_alloc(NULL, (size_t)XE_Q_HEADS * XE_GLOBAL_HEAD_DIM * sizeof(float), XE_MEM_HOST);
    float *heads_ref = xe_alloc(NULL, (size_t)XE_Q_HEADS * XE_GLOBAL_HEAD_DIM * sizeof(float), XE_MEM_HOST);
    for (int i = 0; i < XE_Q_HEADS * XE_GLOBAL_HEAD_DIM; i++) heads[i] = heads_ref[i] = test_value(i + 211);
    int pos = 1773;
    xe_rope_prepare(&s, pos);
    xe_rope_apply(heads, XE_Q_HEADS, XE_GLOBAL_HEAD_DIM, s.rope_global_cos, s.rope_global_sin);
    int half = XE_GLOBAL_HEAD_DIM / 2;
    for (int h = 0; h < XE_Q_HEADS; h++) {
        float *head = heads_ref + (size_t)h * XE_GLOBAL_HEAD_DIM;
        for (int i = 0; i < half; i++) {
            float inv = powf(XE_GLOBAL_ROPE_BASE, -2.0f * (float)i / XE_GLOBAL_HEAD_DIM);
            float theta = (float)pos * inv;
            float c = cosf(theta);
            float sn = sinf(theta);
            float lo = head[i];
            float hi = head[i + half];
            head[i] = lo * c - hi * sn;
            head[i + half] = lo * sn + hi * c;
        }
    }
    double rope_rel = test_rel_rms_decode(heads, heads_ref, XE_Q_HEADS * XE_GLOBAL_HEAD_DIM);
    int ok = norm_rel < 2e-6 && rope_rel < 2e-6;
    printf("decode: norm %.3e rope %.3e %s\n", norm_rel, rope_rel, ok ? "PASS" : "FAIL");
    xe_free(NULL, heads_ref, XE_MEM_HOST);
    xe_free(NULL, heads, XE_MEM_HOST);
    xe_free(NULL, s.rope_global_sin, XE_MEM_HOST);
    xe_free(NULL, s.rope_global_cos, XE_MEM_HOST);
    xe_free(NULL, s.rope_swa_sin, XE_MEM_HOST);
    xe_free(NULL, s.rope_swa_cos, XE_MEM_HOST);
    xe_free(NULL, e.gelu_lut, XE_MEM_HOST);
    return ok;
}

static int test_sampler(void) {
    float logits[] = { 2.0f, 2.0f, 1.0f, 0.0f, -1.0f, -2.0f, -3.0f, -4.0f };
    xe_sample_candidate candidates[8];
    xe_sampler greedy = { 0.0f, 0, 1.0f, 17 };
    int ok = xe_sample_logits(logits, 8, candidates, &greedy) == 0;
    ok = ok && greedy.rng_state == 17;

    xe_sampler one = { 1.0f, 1, 0.1f, 23 };
    ok = ok && xe_sample_logits(logits, 8, candidates, &one) == 0;
    ok = ok && one.rng_state == 23;

    xe_sampler a = { 0.8f, 3, 0.9f, 0 };
    xe_sampler b = a;
    for (int i = 0; i < 128; i++) {
        int32_t ta = xe_sample_logits(logits, 8, candidates, &a);
        int32_t tb = xe_sample_logits(logits, 8, candidates, &b);
        ok = ok && ta == tb && ta >= 0 && ta <= 2;
    }
    ok = ok && a.rng_state == b.rng_state && a.rng_state != 0;

    float equal[] = { 0.0f, 0.0f, 0.0f, 0.0f };
    xe_sample_candidate equal_candidates[4];
    xe_sampler nucleus = { 1.0f, 0, 0.5f, 91 };
    for (int i = 0; i < 128; i++) {
        int32_t t = xe_sample_logits(equal, 4, equal_candidates, &nucleus);
        ok = ok && t >= 0 && t <= 1;
    }

    printf("decode: sampler %s\n", ok ? "PASS" : "FAIL");
    return ok;
}

/* CPU router contract with analytic distributions, independent of the model
 * and of the engine's returned probabilities/selected IDs/weights. Sparse
 * basis products are exact. All denominators are powers of two until the
 * final selected-weight division, so no fidelity epsilon is needed. */
static int test_router_contract(void) {
    float *weights = calloc((size_t)XE_EXPERTS * XE_EMBD, sizeof(*weights));
    if (!weights) return 0;
    float input[XE_EMBD], probabilities[XE_EXPERTS];
    for (int i = 0; i < XE_EMBD; i++) input[i] = 1.0f;
    xe_layer layer = { .router_w = weights };
    xe_session s = { .router_in = input, .router_logits = probabilities };
    memset(s.expert_trace, -1, sizeof s.expert_trace);
    xe_router(&s, &layer, 3);
    int ok = 1;
    /* All logits zero: exact uniform distribution, stable ascending tie
     * order and renormalization over exactly eight selected experts. */
    for (int expert = 0; expert < XE_EXPERTS; expert++)
        ok &= probabilities[expert] == 1.0f / XE_EXPERTS;
    for (int slot = 0; slot < XE_EXPERTS_USED; slot++) {
        ok &= s.experts[slot] == slot && s.expert_weights[slot] == 0.125f;
        ok &= s.expert_trace[3][slot] == slot
              && s.expert_weight_trace[3][slot] == 0.125f;
        ok &= s.expert_trace[2][slot] == -1;
    }
    static const int high[] = {17, 29, 63, 127};
    static const int low[] = {0, 1, 3, 4, 6, 9, 12, 15};
    static const int selected[] = {17, 29, 63, 127, 0, 1, 3, 4};
    /* Four logits0, eight logits-log(2), all others -infinity: unnormalized
     * masses4*1 + 8*0.5=8, selected mass4*1 + 4*0.5=6.
     * This float argument's correctly rounded expf is exactly0.5. */
    const float half_logit = -0x1.62e43p-1f;
    ok &= expf(half_logit) == 0.5f;
    for (int expert = 0; expert < XE_EXPERTS; expert++) {
        int coordinate = (expert * 19 + 37) % XE_EMBD;
        float logit = -INFINITY;
        for (size_t j = 0; j < sizeof high / sizeof high[0]; j++)
            if (expert == high[j]) logit = 0.0f;
        for (size_t j = 0; j < sizeof low / sizeof low[0]; j++)
            if (expert == low[j]) logit = half_logit;
        weights[(size_t)expert * XE_EMBD + coordinate] = logit;
    }
    xe_router(&s, &layer, 5);
    for (int expert = 0; expert < XE_EXPERTS; expert++) {
        float expected = 0.0f;
        for (size_t j = 0; j < sizeof high / sizeof high[0]; j++)
            if (expert == high[j]) expected = 0.125f;
        for (size_t j = 0; j < sizeof low / sizeof low[0]; j++)
            if (expert == low[j]) expected = 0.0625f;
        ok &= probabilities[expert] == expected;
    }
    for (int slot = 0; slot < XE_EXPERTS_USED; slot++) {
        float expected_weight = slot < 4 ? 1.0f/6.0f : 1.0f/12.0f;
        ok &= s.experts[slot] == selected[slot]
              && s.expert_weights[slot] == expected_weight;
        ok &= s.expert_trace[5][slot] == selected[slot]
              && s.expert_weight_trace[5][slot] == expected_weight;
        ok &= s.expert_trace[3][slot] == slot
              && s.expert_weight_trace[3][slot] == 0.125f;
    }
    free(weights);
    printf("decode: CPU router analytic uniform/nonuniform/ties/weights/traces %s\n",
           ok ? "PASS" : "FAIL");
    return ok;
}

static void observer_fixture(void *user, int layer, int position,
                             const float *hidden, int count) {
    (void)user; (void)layer; (void)position; (void)hidden; (void)count;
}

static int test_observer_registration(void) {
    xe_engine e = { .pool_initialized = 1, .owner = pthread_self() };
    xe_session s = { .engine = &e };
    int user = 17;
    xe_session_set_layer_observer(&s, observer_fixture, &user);
    int ok = s.layer_observer == observer_fixture && s.layer_observer_user == &user;
    s.n_tokens = 7;
    xe_session_reset(&s);
    ok &= s.n_tokens == 0 && s.layer_observer == observer_fixture
          && s.layer_observer_user == &user;
    xe_session_set_layer_observer(&s, NULL, NULL);
    ok &= !s.layer_observer && !s.layer_observer_user;
    printf("decode: observer registration/reset/disable %s\n", ok ? "PASS" : "FAIL");
    return ok;
}

int main(void) {
    int ok = 1;
    ok = test_q8_contract() && ok;
    ok = test_v1_contract() && ok;
    ok = test_x8_shapes() && ok;
    ok = test_gelu_contract() && ok;
    ok = test_norm_rope() && ok;
    ok = test_sampler() && ok;
    ok = test_router_contract() && ok;
    ok = test_observer_registration() && ok;
    return ok ? 0 : 1;
}
