__kernel void xe_probe(__global uint *value) {
    if (get_global_id(0) == 0) value[0] = 0x78656e6fu;
}

#pragma OPENCL EXTENSION cl_khr_fp16 : enable
#pragma OPENCL EXTENSION cl_intel_subgroups : enable

#define XE_PREFILL_TM 32
#define XE_PREFILL_KB 64
#define XE_PREFILL_N128_KB 32

#define XE_PREFILL_GEMM_ARGS __global const uchar *wq, \
                             __global const half *wd, \
                             __global const char *aq, \
                             __global const half *ad, \
                             __global const short *as, \
                             __global float *out, int m_count, \
                             int n_count, int blocks

__attribute__((intel_reqd_sub_group_size(16)))
__kernel void xe_prefill_q4q8_n32(XE_PREFILL_GEMM_ARGS) {
    __local char la[XE_PREFILL_TM * XE_PREFILL_KB];
    __local uchar lw[32 * (XE_PREFILL_KB / 2)];
    __local half lad[XE_PREFILL_TM * 2];
    __local short las[XE_PREFILL_TM * 2];
    __local half lwd[32 * 2];
    int lid = get_local_id(0);
    int wm = lid >> 4;
    int wn = lid & 15;
    int m0 = get_group_id(0) * XE_PREFILL_TM;
    int n0 = get_group_id(1) * 32;
    float acc[4][2] = {{0.0f}};
    for (int kb = 0; kb < blocks * 32; kb += XE_PREFILL_KB) {
        for (int x = lid; x < XE_PREFILL_TM * XE_PREFILL_KB; x += 128) {
            int lm = x / XE_PREFILL_KB;
            int lk = x - lm * XE_PREFILL_KB;
            int gm = m0 + lm;
            la[x] = gm < m_count
                    ? aq[(size_t)gm * blocks * 32 + kb + lk] : 0;
        }
        #pragma unroll
        for (int seg = 0; seg < 8; seg++) {
            int lng = seg >> 1;
            int lb = seg & 1;
            int chunk = lid >> 5;
            int rem = lid & 31;
            int row8 = rem >> 2;
            int j = rem & 3;
            int ln = lng * 8 + row8;
            lw[ln * 32 + lb * 16 + chunk * 4 + j] =
                wq[((size_t)((n0 >> 3) + lng) * blocks
                    + kb / 32 + lb) * 128 + lid];
        }
        if (lid < XE_PREFILL_TM * 2) {
            int lm = lid >> 1;
            int lb = lid & 1;
            int gm = m0 + lm;
            lad[lid] = gm < m_count
                       ? ad[(size_t)gm * blocks + kb / 32 + lb] : (half)0;
            las[lid] = gm < m_count
                       ? as[(size_t)gm * blocks + kb / 32 + lb] : (short)0;
        }
        if (lid < 32 * 2) {
            int seg = lid >> 3;
            int row8 = lid & 7;
            int lng = seg >> 1;
            int lb = seg & 1;
            lwd[(lng * 8 + row8) * 2 + lb] =
                wd[((size_t)((n0 >> 3) + lng) * blocks
                    + kb / 32 + lb) * 8 + row8];
        }
        barrier(CLK_LOCAL_MEM_FENCE);
        #pragma unroll
        for (int lb = 0; lb < 2; lb++) {
            #pragma unroll
            for (int im = 0; im < 4; im++) {
                int lm = wm + im * 8;
                int correction = 8 * (int)las[lm * 2 + lb];
                float da = (float)lad[lm * 2 + lb];
                #pragma unroll
                for (int in = 0; in < 2; in++) {
                    int ln = wn + in * 16;
                    int integer = 0;
                    #pragma unroll
                    for (int c = 0; c < 4; c++) {
                        uchar4 packed = vload4(
                            0, lw + ln * 32 + lb * 16 + c * 4);
                        char4 lo = vload4(
                            0, la + lm * XE_PREFILL_KB + lb * 32 + c * 4);
                        char4 hi = vload4(
                            0, la + lm * XE_PREFILL_KB + lb * 32 + 16 + c * 4);
                        integer += dot(packed & (uchar4)(15), lo)
                                   + dot(packed >> (uchar4)(4), hi);
                    }
                    acc[im][in] += (float)(integer - correction) * da
                                   * (float)lwd[ln * 2 + lb];
                }
            }
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    #pragma unroll
    for (int im = 0; im < 4; im++) {
        int gm = m0 + wm + im * 8;
        if (gm >= m_count) continue;
        #pragma unroll
        for (int in = 0; in < 2; in++)
            out[(size_t)gm * n_count + n0 + wn + in * 16] = acc[im][in];
    }
}

__attribute__((intel_reqd_sub_group_size(16)))
__kernel void xe_prefill_q4q8_n64(XE_PREFILL_GEMM_ARGS) {
    __local char la[XE_PREFILL_TM * XE_PREFILL_KB];
    __local uint lw[64 * 8];
    __local half lad[XE_PREFILL_TM * 2];
    __local short las[XE_PREFILL_TM * 2];
    __local half lwd[64 * 2];
    int lid = get_local_id(0);
    int subgroup = lid >> 4;
    int lane = lid & 15;
    int wm = (subgroup & 3) * 8;
    int wn = (subgroup >> 2) * 32 + lane;
    int m0 = get_group_id(0) * XE_PREFILL_TM;
    int n0 = get_group_id(1) * 64;
    float acc[8][2] = {{0.0f}};
    for (int kb = 0; kb < blocks * 32; kb += XE_PREFILL_KB) {
        for (int x = lid; x < XE_PREFILL_TM * XE_PREFILL_KB; x += 128) {
            int lm = x / XE_PREFILL_KB;
            int lk = x - lm * XE_PREFILL_KB;
            int gm = m0 + lm;
            la[x] = gm < m_count
                    ? aq[(size_t)gm * blocks * 32 + kb + lk] : 0;
        }
        for (int x = lid; x < 64 * 8; x += 128) {
            int ln = x >> 3;
            int q = x & 7;
            int lb = q >> 2;
            int c = q & 3;
            int gn = n0 + ln;
            __global const uint *source = (__global const uint *)(
                wq + ((size_t)(gn >> 3) * blocks + kb / 32 + lb) * 128
                + c * 32 + (gn & 7) * 4);
            lw[(lb * 4 + c) * 64 + ln] = *source;
        }
        if (lid < XE_PREFILL_TM * 2) {
            int lm = lid >> 1;
            int lb = lid & 1;
            int gm = m0 + lm;
            lad[lid] = gm < m_count
                       ? ad[(size_t)gm * blocks + kb / 32 + lb] : (half)0;
            las[lid] = gm < m_count
                       ? as[(size_t)gm * blocks + kb / 32 + lb] : (short)0;
        }
        {
            int ln = lid >> 1;
            int lb = lid & 1;
            int gn = n0 + ln;
            lwd[lb * 64 + ln] =
                wd[((size_t)(gn >> 3) * blocks + kb / 32 + lb) * 8
                   + (gn & 7)];
        }
        barrier(CLK_LOCAL_MEM_FENCE);
        #pragma unroll
        for (int lb = 0; lb < 2; lb++) {
            uchar4 packed00 = as_uchar4(lw[(lb * 4) * 64 + wn]);
            uchar4 packed01 = as_uchar4(lw[(lb * 4 + 1) * 64 + wn]);
            uchar4 packed02 = as_uchar4(lw[(lb * 4 + 2) * 64 + wn]);
            uchar4 packed03 = as_uchar4(lw[(lb * 4 + 3) * 64 + wn]);
            uchar4 packed10 = as_uchar4(lw[(lb * 4) * 64 + wn + 16]);
            uchar4 packed11 = as_uchar4(lw[(lb * 4 + 1) * 64 + wn + 16]);
            uchar4 packed12 = as_uchar4(lw[(lb * 4 + 2) * 64 + wn + 16]);
            uchar4 packed13 = as_uchar4(lw[(lb * 4 + 3) * 64 + wn + 16]);
            #pragma unroll
            for (int im = 0; im < 8; im++) {
                int lm = wm + im;
                int correction = 8 * (int)las[lm * 2 + lb];
                float da = (float)lad[lm * 2 + lb];
                char4 lo0 = vload4(
                    0, la + lm * XE_PREFILL_KB + lb * 32);
                char4 lo1 = vload4(
                    0, la + lm * XE_PREFILL_KB + lb * 32 + 4);
                char4 lo2 = vload4(
                    0, la + lm * XE_PREFILL_KB + lb * 32 + 8);
                char4 lo3 = vload4(
                    0, la + lm * XE_PREFILL_KB + lb * 32 + 12);
                char4 hi0 = vload4(
                    0, la + lm * XE_PREFILL_KB + lb * 32 + 16);
                char4 hi1 = vload4(
                    0, la + lm * XE_PREFILL_KB + lb * 32 + 20);
                char4 hi2 = vload4(
                    0, la + lm * XE_PREFILL_KB + lb * 32 + 24);
                char4 hi3 = vload4(
                    0, la + lm * XE_PREFILL_KB + lb * 32 + 28);
                int integer0 = dot(packed00 & (uchar4)(15), lo0)
                               + dot(packed00 >> (uchar4)(4), hi0)
                               + dot(packed01 & (uchar4)(15), lo1)
                               + dot(packed01 >> (uchar4)(4), hi1)
                               + dot(packed02 & (uchar4)(15), lo2)
                               + dot(packed02 >> (uchar4)(4), hi2)
                               + dot(packed03 & (uchar4)(15), lo3)
                               + dot(packed03 >> (uchar4)(4), hi3);
                int integer1 = dot(packed10 & (uchar4)(15), lo0)
                               + dot(packed10 >> (uchar4)(4), hi0)
                               + dot(packed11 & (uchar4)(15), lo1)
                               + dot(packed11 >> (uchar4)(4), hi1)
                               + dot(packed12 & (uchar4)(15), lo2)
                               + dot(packed12 >> (uchar4)(4), hi2)
                               + dot(packed13 & (uchar4)(15), lo3)
                               + dot(packed13 >> (uchar4)(4), hi3);
                acc[im][0] += (float)(integer0 - correction) * da
                              * (float)lwd[lb * 64 + wn];
                acc[im][1] += (float)(integer1 - correction) * da
                              * (float)lwd[lb * 64 + wn + 16];
            }
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    #pragma unroll
    for (int im = 0; im < 8; im++) {
        int gm = m0 + wm + im;
        if (gm >= m_count) continue;
        out[(size_t)gm * n_count + n0 + wn] = acc[im][0];
        out[(size_t)gm * n_count + n0 + wn + 16] = acc[im][1];
    }
}

static inline void xe_prefill_q4q8_n128_body(
                             __global const uchar *wq,
                             __global const half *wd,
                             __global const char *aq,
                             __global const half *ad,
                             __global float *out, int m_count,
                             int n_count, int blocks,
                             __local char *la,
                             __local char *lw,
                             __local half *lad,
                             __local half *lwd) {
    int lid = get_local_id(0);
    int subgroup = lid >> 4;
    int lane = lid & 15;
    int wm = (subgroup & 3) * 8;
    int wn = (subgroup >> 2) * 32 + lane;
    int m0 = get_group_id(0) * XE_PREFILL_TM;
    int n0 = get_group_id(1) * 128;
    float acc[8][2] = {{0.0f}};
    for (int kb = 0; kb < blocks * 32; kb += XE_PREFILL_N128_KB) {
        for (int x = lid; x < XE_PREFILL_TM * XE_PREFILL_N128_KB; x += 256) {
            int lm = x / XE_PREFILL_N128_KB;
            int lk = x - lm * XE_PREFILL_N128_KB;
            int gm = m0 + lm;
            la[x] = gm < m_count
                    ? aq[(size_t)gm * blocks * 32 + kb + lk] : 0;
        }
        for (int x = lid; x < 128 * 4; x += 256) {
            int ln = x >> 2;
            int c = x & 3;
            int gn = n0 + ln;
            __global const uint *source = (__global const uint *)(
                wq + ((size_t)(gn >> 3) * blocks + kb / 32) * 128
                + c * 32 + (gn & 7) * 4);
            uchar4 packed = as_uchar4(*source);
            int base = ln * XE_PREFILL_N128_KB + c * 4;
            vstore4(convert_char4(packed & (uchar4)(15)) - (char4)(8),
                    0, lw + base);
            vstore4(convert_char4(packed >> (uchar4)(4)) - (char4)(8),
                    0, lw + base + 16);
        }
        if (lid < XE_PREFILL_TM) {
            int lm = lid;
            int gm = m0 + lm;
            lad[lm] = gm < m_count
                      ? ad[(size_t)gm * blocks + kb / 32] : (half)0;
        }
        if (lid < 128) {
            int ln = lid;
            int gn = n0 + ln;
            lwd[ln] =
                wd[((size_t)(gn >> 3) * blocks + kb / 32) * 8
                   + (gn & 7)];
        }
        barrier(CLK_LOCAL_MEM_FENCE);
        #pragma unroll
        for (int im = 0; im < 8; im++) {
            int lm = wm + im;
            float da = (float)lad[lm];
            int integer0 = 0;
            int integer1 = 0;
            #pragma unroll
            for (int c = 0; c < 8; c++) {
                char4 activation = vload4(
                    0, la + lm * XE_PREFILL_N128_KB + c * 4);
                integer0 += dot(vload4(
                    0, lw + wn * XE_PREFILL_N128_KB + c * 4),
                    activation);
                integer1 += dot(vload4(
                    0, lw + (wn + 16) * XE_PREFILL_N128_KB + c * 4),
                    activation);
            }
            acc[im][0] += (float)integer0 * da * (float)lwd[wn];
            acc[im][1] += (float)integer1 * da * (float)lwd[wn + 16];
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    #pragma unroll
    for (int im = 0; im < 8; im++) {
        int gm = m0 + wm + im;
        if (gm >= m_count) continue;
        out[(size_t)gm * n_count + n0 + wn] = acc[im][0];
        out[(size_t)gm * n_count + n0 + wn + 16] = acc[im][1];
    }
}

__attribute__((intel_reqd_sub_group_size(16)))
__kernel void xe_prefill_q4q8_n128(XE_PREFILL_GEMM_ARGS) {
    __local char la[XE_PREFILL_TM * XE_PREFILL_N128_KB];
    __local char lw[128 * XE_PREFILL_N128_KB];
    __local half lad[XE_PREFILL_TM];
    __local half lwd[128];
    xe_prefill_q4q8_n128_body(
        wq, wd, aq, ad, out, m_count, n_count, blocks, la, lw, lad, lwd);
}

__attribute__((intel_reqd_sub_group_size(16)))
__kernel void xe_prefill_q4q8_kv_n128(XE_PREFILL_GEMM_ARGS) {
    __local char la[XE_PREFILL_TM * XE_PREFILL_N128_KB];
    __local char lw[128 * XE_PREFILL_N128_KB];
    __local half lad[XE_PREFILL_TM];
    __local half lwd[128];
    xe_prefill_q4q8_n128_body(
        wq, wd, aq, ad, out, m_count, 2048, 88, la, lw, lad, lwd);
}

__attribute__((intel_reqd_sub_group_size(16)))
__kernel void xe_prefill_q4q8_swa_q_n128(XE_PREFILL_GEMM_ARGS) {
    __local char la[XE_PREFILL_TM * XE_PREFILL_N128_KB];
    __local char lw[128 * XE_PREFILL_N128_KB];
    __local half lad[XE_PREFILL_TM];
    __local half lwd[128];
    xe_prefill_q4q8_n128_body(
        wq, wd, aq, ad, out, m_count, 4096, 88, la, lw, lad, lwd);
}

__attribute__((intel_reqd_sub_group_size(16)))
__kernel void xe_prefill_q4q8_swa_o_n128(XE_PREFILL_GEMM_ARGS) {
    __local char la[XE_PREFILL_TM * XE_PREFILL_N128_KB];
    __local char lw[128 * XE_PREFILL_N128_KB];
    __local half lad[XE_PREFILL_TM];
    __local half lwd[128];
    xe_prefill_q4q8_n128_body(
        wq, wd, aq, ad, out, m_count, 2816, 128, la, lw, lad, lwd);
}

__attribute__((intel_reqd_sub_group_size(16)))
__kernel void xe_prefill_q4q8_global_q_n128(XE_PREFILL_GEMM_ARGS) {
    __local char la[XE_PREFILL_TM * XE_PREFILL_N128_KB];
    __local char lw[128 * XE_PREFILL_N128_KB];
    __local half lad[XE_PREFILL_TM];
    __local half lwd[128];
    xe_prefill_q4q8_n128_body(
        wq, wd, aq, ad, out, m_count, 8192, 88, la, lw, lad, lwd);
}

__attribute__((intel_reqd_sub_group_size(16)))
__kernel void xe_prefill_q4q8_global_k_n128(XE_PREFILL_GEMM_ARGS) {
    __local char la[XE_PREFILL_TM * XE_PREFILL_N128_KB];
    __local char lw[128 * XE_PREFILL_N128_KB];
    __local half lad[XE_PREFILL_TM];
    __local half lwd[128];
    xe_prefill_q4q8_n128_body(
        wq, wd, aq, ad, out, m_count, 1024, 88, la, lw, lad, lwd);
}

__attribute__((intel_reqd_sub_group_size(16)))
__kernel void xe_prefill_q4q8_global_o_n128(XE_PREFILL_GEMM_ARGS) {
    __local char la[XE_PREFILL_TM * XE_PREFILL_N128_KB];
    __local char lw[128 * XE_PREFILL_N128_KB];
    __local half lad[XE_PREFILL_TM];
    __local half lwd[128];
    xe_prefill_q4q8_n128_body(
        wq, wd, aq, ad, out, m_count, 2816, 256, la, lw, lad, lwd);
}

__attribute__((intel_reqd_sub_group_size(16)))
__kernel void xe_prefill_q4q8_dense_down_n128(XE_PREFILL_GEMM_ARGS) {
    __local char la[XE_PREFILL_TM * XE_PREFILL_N128_KB];
    __local char lw[128 * XE_PREFILL_N128_KB];
    __local half lad[XE_PREFILL_TM];
    __local half lwd[128];
    xe_prefill_q4q8_n128_body(
        wq, wd, aq, ad, out, m_count, 2816, 66, la, lw, lad, lwd);
}

__attribute__((intel_reqd_sub_group_size(16)))
__kernel void xe_prefill_q4q8_n128_tail(XE_PREFILL_GEMM_ARGS) {
    __local char la[XE_PREFILL_TM * XE_PREFILL_KB];
    __local char lw[128 * XE_PREFILL_KB];
    __local half lad[XE_PREFILL_TM * 2];
    __local half lwd[128 * 2];
    int lid = get_local_id(0);
    int subgroup = lid >> 4;
    int lane = lid & 15;
    int wm = (subgroup & 3) * 8;
    int wn = (subgroup >> 2) * 32 + lane;
    int m0 = get_group_id(0) * XE_PREFILL_TM;
    int n0 = get_group_id(1) * 128;
    float acc[8][2] = {{0.0f}};
    for (int kb = 0; kb < blocks * 32; kb += XE_PREFILL_KB) {
        for (int x = lid; x < XE_PREFILL_TM * XE_PREFILL_KB; x += 256) {
            int lm = x / XE_PREFILL_KB;
            int lk = x - lm * XE_PREFILL_KB;
            int gm = m0 + lm;
            la[x] = gm < m_count
                    ? aq[(size_t)gm * blocks * 32 + kb + lk] : 0;
        }
        for (int x = lid; x < 128 * 8; x += 256) {
            int ln = x >> 3;
            int q = x & 7;
            int lb = q >> 2;
            int c = q & 3;
            int gn = n0 + ln;
            __global const uint *source = (__global const uint *)(
                wq + ((size_t)(gn >> 3) * blocks + kb / 32 + lb) * 128
                + c * 32 + (gn & 7) * 4);
            uchar4 packed = as_uchar4(gn < n_count ? *source : 0);
            int base = ln * XE_PREFILL_KB + lb * 32 + c * 4;
            vstore4(convert_char4(packed & (uchar4)(15)) - (char4)(8),
                    0, lw + base);
            vstore4(convert_char4(packed >> (uchar4)(4)) - (char4)(8),
                    0, lw + base + 16);
        }
        if (lid < XE_PREFILL_TM * 2) {
            int lm = lid >> 1;
            int lb = lid & 1;
            int gm = m0 + lm;
            lad[lid] = gm < m_count
                       ? ad[(size_t)gm * blocks + kb / 32 + lb] : (half)0;
        }
        {
            int ln = lid >> 1;
            int lb = lid & 1;
            int gn = n0 + ln;
            lwd[lb * 128 + ln] = gn < n_count ?
                wd[((size_t)(gn >> 3) * blocks + kb / 32 + lb) * 8
                   + (gn & 7)] : (half)0;
        }
        barrier(CLK_LOCAL_MEM_FENCE);
        #pragma unroll
        for (int lb = 0; lb < 2; lb++) {
            #pragma unroll
            for (int im = 0; im < 8; im++) {
                int lm = wm + im;
                float da = (float)lad[lm * 2 + lb];
                int integer0 = 0;
                int integer1 = 0;
                #pragma unroll
                for (int c = 0; c < 8; c++) {
                    char4 activation = vload4(
                        0, la + lm * XE_PREFILL_KB + lb * 32 + c * 4);
                    integer0 += dot(vload4(
                        0, lw + wn * XE_PREFILL_KB + lb * 32 + c * 4),
                        activation);
                    integer1 += dot(vload4(
                        0, lw + (wn + 16) * XE_PREFILL_KB
                           + lb * 32 + c * 4), activation);
                }
                acc[im][0] += (float)integer0 * da
                              * (float)lwd[lb * 128 + wn];
                acc[im][1] += (float)integer1 * da
                              * (float)lwd[lb * 128 + wn + 16];
            }
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    #pragma unroll
    for (int im = 0; im < 8; im++) {
        int gm = m0 + wm + im;
        if (gm >= m_count) continue;
        int gn = n0 + wn;
        if (gn < n_count)
            out[(size_t)gm * n_count + gn] = acc[im][0];
        if (gn + 16 < n_count)
            out[(size_t)gm * n_count + gn + 16] = acc[im][1];
    }
}

__kernel void xe_prefill_rms_scale(__global const float *input,
                                   __global float *scale, int rows,
                                   int width, float epsilon) {
    __local float partial[128];
    int row = get_group_id(0);
    int lid = get_local_id(0);
    if (row >= rows) return;
    float sum = 0.0f;
    for (int column = lid; column < width; column += 128) {
        float value = input[(size_t)row * width + column];
        sum += value * value;
    }
    partial[lid] = sum;
    barrier(CLK_LOCAL_MEM_FENCE);
    for (int stride = 64; stride > 0; stride >>= 1) {
        if (lid < stride) partial[lid] += partial[lid + stride];
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    if (lid == 0)
        scale[row] = rsqrt(partial[0] / (float)width + epsilon);
}

__attribute__((intel_reqd_sub_group_size(16)))
__kernel void xe_prefill_norm_q8(__global const float *input,
                                 __global const float *weight,
                                 __global const float *row_scale,
                                 __global char *quantized,
                                 __global half *scale,
                                 __global short *sigma,
                                 int rows, int width) {
    int block_count = width / 32;
    int task = get_group_id(0) * 8 + get_sub_group_id();
    int lane = get_sub_group_local_id();
    int row = task / block_count;
    int block = task - row * block_count;
    if (row >= rows) return;
    float values[2];
    #pragma unroll
    for (int half_index = 0; half_index < 2; half_index++) {
        int column = block * 32 + lane + half_index * 16;
        values[half_index] = input[(size_t)row * width + column]
                             * row_scale[row] * weight[column];
    }
    float maximum = sub_group_reduce_max(
        fmax(fabs(values[0]), fabs(values[1])));
    float d = maximum / 127.0f;
    float inverse = d == 0.0f ? 0.0f : 1.0f / d;
    int q0 = convert_int(round(values[0] * inverse));
    int q1 = convert_int(round(values[1] * inverse));
    size_t base = ((size_t)row * block_count + block) * 32;
    quantized[base + lane] = (char)q0;
    quantized[base + lane + 16] = (char)q1;
    int sum = sub_group_reduce_add(q0 + q1);
    if (lane == 0) {
        size_t index = (size_t)row * block_count + block;
        scale[index] = (half)d;
        sigma[index] = (short)sum;
    }
}

__kernel void xe_prefill_qkv_post(__global const float *q_projection,
                                  __global const float *k_projection,
                                  __global const float *v_projection,
                                  __global const float *q_weight,
                                  __global const float *k_weight,
                                  __global const float *rope_cos,
                                  __global const float *rope_sin,
                                  __global float *q,
                                  __global half *k,
                                  __global half *v,
                                  int rows,
                                  int dimension,
                                  int kv_heads,
                                  int has_v,
                                  float epsilon) {
    __local float partial[128];
    int task = get_group_id(0);
    int lid = get_local_id(0);
    int per_row = 16 + 2 * kv_heads;
    int row = task / per_row;
    int local_task = task - row * per_row;
    if (row >= rows) return;
    int kind = local_task < 16 ? 0
               : local_task < 16 + kv_heads ? 1 : 2;
    int head = kind == 0 ? local_task
               : kind == 1 ? local_task - 16
               : local_task - 16 - kv_heads;
    __global const float *source = kind == 0 ? q_projection
                                   : kind == 1 || !has_v ? k_projection
                                   : v_projection;
    int source_heads = kind == 0 ? 16 : kv_heads;
    size_t source_base = ((size_t)row * source_heads + head) * dimension;
    float sum = 0.0f;
    for (int d = lid; d < dimension; d += 128) {
        float value = source[source_base + d];
        sum += value * value;
    }
    partial[lid] = sum;
    barrier(CLK_LOCAL_MEM_FENCE);
    for (int stride = 64; stride > 0; stride >>= 1) {
        if (lid < stride) partial[lid] += partial[lid + stride];
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    float scale = rsqrt(partial[0] / (float)dimension + epsilon);
    if (kind == 2) {
        for (int d = lid; d < dimension; d += 128)
            v[((size_t)head * rows + row) * dimension + d] =
                (half)(source[source_base + d] * scale);
        return;
    }
    __global const float *weight = kind == 0 ? q_weight : k_weight;
    int half_dimension = dimension / 2;
    for (int d = lid; d < half_dimension; d += 128) {
        float lo = source[source_base + d] * scale * weight[d];
        float hi = source[source_base + d + half_dimension] * scale
                   * weight[d + half_dimension];
        float cosine = rope_cos[(size_t)row * half_dimension + d];
        float sine = rope_sin[(size_t)row * half_dimension + d];
        float rotated_lo = lo * cosine - hi * sine;
        float rotated_hi = lo * sine + hi * cosine;
        if (kind == 0) {
            size_t base = ((size_t)head * rows + row) * dimension;
            q[base + d] = rotated_lo;
            q[base + d + half_dimension] = rotated_hi;
        } else {
            size_t base = ((size_t)head * rows + row) * dimension;
            k[base + d] = (half)rotated_lo;
            k[base + d + half_dimension] = (half)rotated_hi;
        }
    }
}

__attribute__((intel_reqd_sub_group_size(16)))
__kernel void xe_prefill_attn_online_b8(__global const float *q,
                                        __global const half *k,
                                        __global const half *v,
                                        __global float *out,
                                        int m_count,
                                        int n_count,
                                        int dimension,
                                        int heads,
                                        int kv_heads,
                                        int query_offset,
                                        int window) {
    int query_head = get_group_id(0) * 8 + get_sub_group_id();
    int lane = get_sub_group_local_id();
    int query = query_head % m_count;
    int head = query_head / m_count;
    if (head >= heads) return;
    int kv_head = head * kv_heads / heads;
    int position = query_offset + query;
    int first = window > 0 ? max(0, position - window + 1) : 0;
    float maximum = -INFINITY;
    float denominator = 0.0f;
    float acc[32] = {0.0f};
    for (int key0 = first; key0 <= position; key0 += 8) {
        int keys = min(8, position - key0 + 1);
        float score[8];
        float block_maximum = -INFINITY;
        for (int local_key = 0; local_key < keys; local_key++) {
            float value = 0.0f;
            int key = key0 + local_key;
            for (int d = lane; d < dimension; d += 16)
                value += q[((size_t)head * m_count + query) * dimension + d]
                         * (float)k[((size_t)kv_head * n_count + key)
                                    * dimension + d];
            score[local_key] = sub_group_reduce_add(value);
            block_maximum = fmax(block_maximum, score[local_key]);
        }
        float next_maximum = fmax(maximum, block_maximum);
        float alpha = maximum == -INFINITY ? 0.0f
                      : exp(maximum - next_maximum);
        float beta[8];
        float block_denominator = 0.0f;
        for (int local_key = 0; local_key < keys; local_key++) {
            beta[local_key] = exp(score[local_key] - next_maximum);
            block_denominator += beta[local_key];
        }
        denominator = denominator * alpha + block_denominator;
        int owned = 0;
        for (int d = lane; d < dimension; d += 16) {
            float value = acc[owned] * alpha;
            for (int local_key = 0; local_key < keys; local_key++)
                value += beta[local_key]
                         * (float)v[((size_t)kv_head * n_count
                                     + key0 + local_key) * dimension + d];
            acc[owned++] = value;
        }
        maximum = next_maximum;
    }
    int owned = 0;
    for (int d = lane; d < dimension; d += 16)
        out[((size_t)head * m_count + query) * dimension + d] =
            acc[owned++] / denominator;
}

static inline float xe_prefill_attn_float8_accumulate(float value,
                                                      float8 a, float8 b) {
    value += a.s0 * b.s0;
    value += a.s1 * b.s1;
    value += a.s2 * b.s2;
    value += a.s3 * b.s3;
    value += a.s4 * b.s4;
    value += a.s5 * b.s5;
    value += a.s6 * b.s6;
    value += a.s7 * b.s7;
    return value;
}

__attribute__((intel_reqd_sub_group_size(16)))
__kernel void xe_prefill_attn_online_b8_global_shared(
                                        __global const float *q,
                                        __global const half *k,
                                        __global const half *v,
                                        __global float *out,
                                        int m_count,
                                        int n_count,
                                        int query_count,
                                        int heads,
                                        int kv_heads,
                                        int query_offset,
                                        int query_base) {
    __local half lk[8 * 512];
    __local float lalpha[8];
    __local float lbeta[8 * 8];
    __local float ldenominator[8];
    int lid = get_local_id(0);
    int subgroup = get_sub_group_id();
    int lane = get_sub_group_local_id();
    int query_kv = get_group_id(0);
    int query = query_base + query_kv % query_count;
    int kv_head = query_kv / query_count;
    if (kv_head >= kv_heads) return;
    int head = kv_head * heads / kv_heads + subgroup;
    int position = query_offset + query;
    int maximum_position = min(n_count - 1, position);
    __global const uint *q_source = (__global const uint *)(
        q + ((size_t)head * m_count + query) * 512);
    float maximum = -INFINITY;
    float denominator = 0.0f;
    float8 acc0 = (float8)(0.0f);
    float8 acc1 = (float8)(0.0f);
    float8 acc2 = (float8)(0.0f);
    float8 acc3 = (float8)(0.0f);
    for (int key0 = 0; key0 <= maximum_position; key0 += 8) {
        __global const uint4 *source = (__global const uint4 *)(
            k + ((size_t)kv_head * n_count + key0) * 512);
        __local uint4 *target = (__local uint4 *)lk;
        /* Read bounds are required even for rows excluded from the scores.
         * Global KV allocations do not promise an extra tile of padding. */
        if (key0 + 8 <= n_count) {
            for (int x = lid; x < 512; x += 128) target[x] = source[x];
        } else {
            /* The final tile may extend past an arbitrary cache capacity. */
            for (int x = lid; x < 512; x += 128)
                target[x] = key0 + x / 64 < n_count
                            ? source[x] : (uint4)(0);
        }
        barrier(CLK_LOCAL_MEM_FENCE);
        int keys = clamp(maximum_position - key0 + 1, 0, 8);
        float score[8];
        float block_maximum = -INFINITY;
        for (int local_key = 0; local_key < keys; local_key++) {
            __local const ushort *source_k = (__local const ushort *)(
                lk + local_key * 512);
            float value = 0.0f;
            float8 q_value = as_float8(
                intel_sub_group_block_read8(q_source));
            float8 k_value = convert_float8(as_half8(
                intel_sub_group_block_read_us8(source_k)));
            value = xe_prefill_attn_float8_accumulate(
                value, q_value, k_value);
            q_value = as_float8(
                intel_sub_group_block_read8(q_source + 128));
            k_value = convert_float8(as_half8(
                intel_sub_group_block_read_us8(source_k + 128)));
            value = xe_prefill_attn_float8_accumulate(
                value, q_value, k_value);
            q_value = as_float8(
                intel_sub_group_block_read8(q_source + 256));
            k_value = convert_float8(as_half8(
                intel_sub_group_block_read_us8(source_k + 256)));
            value = xe_prefill_attn_float8_accumulate(
                value, q_value, k_value);
            q_value = as_float8(
                intel_sub_group_block_read8(q_source + 384));
            k_value = convert_float8(as_half8(
                intel_sub_group_block_read_us8(source_k + 384)));
            value = xe_prefill_attn_float8_accumulate(
                value, q_value, k_value);
            score[local_key] = sub_group_reduce_add(value);
            block_maximum = fmax(block_maximum, score[local_key]);
        }
        float next_maximum = fmax(maximum, block_maximum);
        float alpha = maximum == -INFINITY ? 0.0f
                      : exp(maximum - next_maximum);
        float beta[8];
        float block_denominator = 0.0f;
        for (int local_key = 0; local_key < keys; local_key++) {
            beta[local_key] = exp(score[local_key] - next_maximum);
            block_denominator += beta[local_key];
        }
        denominator = denominator * alpha + block_denominator;
        if (lane == 0) {
            lalpha[subgroup] = alpha;
            ldenominator[subgroup] = denominator;
            for (int local_key = 0; local_key < keys; local_key++)
                lbeta[local_key * 8 + subgroup] = beta[local_key];
        }
        barrier(CLK_LOCAL_MEM_FENCE);
        float8 shared_alpha = vload8(0, lalpha);
        acc0 *= shared_alpha;
        acc1 *= shared_alpha;
        acc2 *= shared_alpha;
        acc3 *= shared_alpha;
        for (int local_key = 0; local_key < keys; local_key++) {
            __global const ushort *source = (__global const ushort *)(
                v + ((size_t)kv_head * n_count + key0 + local_key) * 512
                  + subgroup * 64);
            float8 weight = vload8(0, lbeta + local_key * 8);
            float4 value = convert_float4(as_half4(
                intel_sub_group_block_read_us4(source)));
            acc0 += weight * value.s0;
            acc1 += weight * value.s1;
            acc2 += weight * value.s2;
            acc3 += weight * value.s3;
        }
        barrier(CLK_LOCAL_MEM_FENCE);
        maximum = next_maximum;
    }
    float8 shared_denominator = vload8(0, ldenominator);
    acc0 /= shared_denominator;
    acc1 /= shared_denominator;
    acc2 /= shared_denominator;
    acc3 /= shared_denominator;
#define XE_PREFILL_GLOBAL_WRITE(local_head, component) do {                   \
        __global uint *target = (__global uint *)(                            \
            out + ((size_t)(kv_head * 8 + local_head) * m_count + query)      \
                      * 512 + subgroup * 64);                                 \
        intel_sub_group_block_write4(                                         \
            target, as_uint4((float4)(acc0.component, acc1.component,         \
                                      acc2.component, acc3.component)));      \
    } while (0)
    XE_PREFILL_GLOBAL_WRITE(0, s0);
    XE_PREFILL_GLOBAL_WRITE(1, s1);
    XE_PREFILL_GLOBAL_WRITE(2, s2);
    XE_PREFILL_GLOBAL_WRITE(3, s3);
    XE_PREFILL_GLOBAL_WRITE(4, s4);
    XE_PREFILL_GLOBAL_WRITE(5, s5);
    XE_PREFILL_GLOBAL_WRITE(6, s6);
    XE_PREFILL_GLOBAL_WRITE(7, s7);
#undef XE_PREFILL_GLOBAL_WRITE
}

__attribute__((intel_reqd_sub_group_size(16)))
__kernel void xe_prefill_attn_online_b8_global_cow(
                                        __global const float *q,
                                        __global const half *prefix_k,
                                        __global const half *prefix_v,
                                        __global const half *tail_k,
                                        __global const half *tail_v,
                                        __global float *out,
                                        int m_count,
                                        int prefix_capacity,
                                        int tail_capacity,
                                        int query_count,
                                        int heads,
                                        int kv_heads,
                                        int query_offset,
                                        int query_base,
                                        int split) {
    __local half lk[8 * 512];
    __local float lalpha[8];
    __local float lbeta[8 * 8];
    __local float ldenominator[8];
    int lid = get_local_id(0);
    int subgroup = get_sub_group_id();
    int lane = get_sub_group_local_id();
    int query_kv = get_group_id(0);
    int query = query_base + query_kv % query_count;
    int kv_head = query_kv / query_count;
    if (kv_head >= kv_heads) return;
    int head = kv_head * heads / kv_heads + subgroup;
    int position = query_offset + query;
    __global const uint *q_source = (__global const uint *)(
        q + ((size_t)head * m_count + query) * 512);
    float maximum = -INFINITY;
    float denominator = 0.0f;
    float8 acc0 = (float8)(0.0f);
    float8 acc1 = (float8)(0.0f);
    float8 acc2 = (float8)(0.0f);
    float8 acc3 = (float8)(0.0f);
    for (int key0 = 0; key0 <= position; key0 += 8) {
        __local uint4 *target = (__local uint4 *)lk;
        int capacity = key0 < split ? prefix_capacity : tail_capacity;
        int source_key = key0 < split ? key0 : key0 - split;
        /* A tail can end in the middle of a globally aligned tile, even at
         * power-of-two CTX. Masking scores does not make extra reads safe. */
        if ((key0 + 7 < split || key0 >= split) &&
            source_key + 8 <= capacity) {
            __global const half *base = key0 < split ? prefix_k : tail_k;
            __global const uint4 *source = (__global const uint4 *)(
                base + ((size_t)kv_head * capacity + source_key) * 512);
            for (int x = lid; x < 512; x += 128) target[x] = source[x];
        } else {
            for (int local_key = 0; local_key < 8; local_key++) {
                int key = key0 + local_key;
                if (key <= position) {
                    int capacity = key < split ? prefix_capacity : tail_capacity;
                    int source_key = key < split ? key : key - split;
                    __global const half *base = key < split ? prefix_k : tail_k;
                    __global const uint4 *source = (__global const uint4 *)(
                        base + ((size_t)kv_head * capacity + source_key) * 512);
                    for (int x = lid; x < 64; x += 128)
                        target[local_key * 64 + x] = source[x];
                } else {
                    for (int x = lid; x < 64; x += 128)
                        target[local_key * 64 + x] = (uint4)(0);
                }
            }
        }
        barrier(CLK_LOCAL_MEM_FENCE);
        int keys = min(8, position - key0 + 1);
        float score[8];
        float block_maximum = -INFINITY;
        for (int local_key = 0; local_key < keys; local_key++) {
            __local const ushort *source_k = (__local const ushort *)(
                lk + local_key * 512);
            float value = 0.0f;
            float8 q_value = as_float8(
                intel_sub_group_block_read8(q_source));
            float8 k_value = convert_float8(as_half8(
                intel_sub_group_block_read_us8(source_k)));
            value = xe_prefill_attn_float8_accumulate(
                value, q_value, k_value);
            q_value = as_float8(
                intel_sub_group_block_read8(q_source + 128));
            k_value = convert_float8(as_half8(
                intel_sub_group_block_read_us8(source_k + 128)));
            value = xe_prefill_attn_float8_accumulate(
                value, q_value, k_value);
            q_value = as_float8(
                intel_sub_group_block_read8(q_source + 256));
            k_value = convert_float8(as_half8(
                intel_sub_group_block_read_us8(source_k + 256)));
            value = xe_prefill_attn_float8_accumulate(
                value, q_value, k_value);
            q_value = as_float8(
                intel_sub_group_block_read8(q_source + 384));
            k_value = convert_float8(as_half8(
                intel_sub_group_block_read_us8(source_k + 384)));
            value = xe_prefill_attn_float8_accumulate(
                value, q_value, k_value);
            score[local_key] = sub_group_reduce_add(value);
            block_maximum = fmax(block_maximum, score[local_key]);
        }
        float next_maximum = fmax(maximum, block_maximum);
        float alpha = maximum == -INFINITY ? 0.0f
                      : exp(maximum - next_maximum);
        float beta[8];
        float block_denominator = 0.0f;
        for (int local_key = 0; local_key < keys; local_key++) {
            beta[local_key] = exp(score[local_key] - next_maximum);
            block_denominator += beta[local_key];
        }
        denominator = denominator * alpha + block_denominator;
        if (lane == 0) {
            lalpha[subgroup] = alpha;
            ldenominator[subgroup] = denominator;
            for (int local_key = 0; local_key < keys; local_key++)
                lbeta[local_key * 8 + subgroup] = beta[local_key];
        }
        barrier(CLK_LOCAL_MEM_FENCE);
        float8 shared_alpha = vload8(0, lalpha);
        acc0 *= shared_alpha;
        acc1 *= shared_alpha;
        acc2 *= shared_alpha;
        acc3 *= shared_alpha;
        for (int local_key = 0; local_key < keys; local_key++) {
            int key = key0 + local_key;
            int capacity = key < split ? prefix_capacity : tail_capacity;
            int source_key = key < split ? key : key - split;
            __global const half *base = key < split ? prefix_v : tail_v;
            __global const ushort *source = (__global const ushort *)(
                base + ((size_t)kv_head * capacity + source_key) * 512
                  + subgroup * 64);
            float8 weight = vload8(0, lbeta + local_key * 8);
            float4 value = convert_float4(as_half4(
                intel_sub_group_block_read_us4(source)));
            acc0 += weight * value.s0;
            acc1 += weight * value.s1;
            acc2 += weight * value.s2;
            acc3 += weight * value.s3;
        }
        barrier(CLK_LOCAL_MEM_FENCE);
        maximum = next_maximum;
    }
    float8 shared_denominator = vload8(0, ldenominator);
    acc0 /= shared_denominator;
    acc1 /= shared_denominator;
    acc2 /= shared_denominator;
    acc3 /= shared_denominator;
#define XE_PREFILL_GLOBAL_COW_WRITE(local_head, component) do {               \
        __global uint *target = (__global uint *)(                            \
            out + ((size_t)(kv_head * 8 + local_head) * m_count + query)      \
                      * 512 + subgroup * 64);                                 \
        intel_sub_group_block_write4(                                         \
            target, as_uint4((float4)(acc0.component, acc1.component,         \
                                      acc2.component, acc3.component)));      \
    } while (0)
    XE_PREFILL_GLOBAL_COW_WRITE(0, s0);
    XE_PREFILL_GLOBAL_COW_WRITE(1, s1);
    XE_PREFILL_GLOBAL_COW_WRITE(2, s2);
    XE_PREFILL_GLOBAL_COW_WRITE(3, s3);
    XE_PREFILL_GLOBAL_COW_WRITE(4, s4);
    XE_PREFILL_GLOBAL_COW_WRITE(5, s5);
    XE_PREFILL_GLOBAL_COW_WRITE(6, s6);
    XE_PREFILL_GLOBAL_COW_WRITE(7, s7);
#undef XE_PREFILL_GLOBAL_COW_WRITE
}

static inline float xe_prefill_attn_swa_block_score(
        float8 q0, float8 q1, __global const half *k) {
    __global const ushort *k_source = (__global const ushort *)k;
    float value = 0.0f;
    float8 k0 = convert_float8(as_half8(
        intel_sub_group_block_read_us8(k_source)));
    value += q0.s0 * k0.s0;
    value += q0.s1 * k0.s1;
    value += q0.s2 * k0.s2;
    value += q0.s3 * k0.s3;
    value += q0.s4 * k0.s4;
    value += q0.s5 * k0.s5;
    value += q0.s6 * k0.s6;
    value += q0.s7 * k0.s7;
    float8 k1 = convert_float8(as_half8(
        intel_sub_group_block_read_us8(k_source + 128)));
    value += q1.s0 * k1.s0;
    value += q1.s1 * k1.s1;
    value += q1.s2 * k1.s2;
    value += q1.s3 * k1.s3;
    value += q1.s4 * k1.s4;
    value += q1.s5 * k1.s5;
    value += q1.s6 * k1.s6;
    value += q1.s7 * k1.s7;
    return sub_group_reduce_add(value);
}

__attribute__((intel_reqd_sub_group_size(16)))
__kernel void xe_prefill_attn_online_b8_swa(__global const float *q,
                                            __global const half *k,
                                            __global const half *v,
                                            __global float *out,
                                            int m_count,
                                            int n_count,
                                            int dimension,
                                            int heads,
                                            int kv_heads,
                                            int query_offset,
                                            int window) {
    int query_head = get_group_id(0) * 8 + get_sub_group_id();
    int lane = get_sub_group_local_id();
    int query = query_head % m_count;
    int head = query_head / m_count;
    if (head >= heads) return;
    int kv_head = head * kv_heads / heads;
    int position = query_offset + query;
    int first = window > 0 ? max(0, position - window + 1) : 0;
    float maximum = -INFINITY;
    float denominator = 0.0f;
    float8 acc0 = (float8)(0.0f);
    float8 acc1 = (float8)(0.0f);
    __global const uint *q_source = (__global const uint *)(
        q + ((size_t)head * m_count + query) * 256);
    float8 q0 = as_float8(intel_sub_group_block_read8(q_source));
    float8 q1 = as_float8(intel_sub_group_block_read8(q_source + 128));
    for (int key0 = first; key0 <= position; key0 += 8) {
        int keys = min(8, position - key0 + 1);
        float score[8];
        float block_maximum = -INFINITY;
        for (int local_key = 0; local_key < keys; local_key++) {
            int key = key0 + local_key;
            score[local_key] = xe_prefill_attn_swa_block_score(
                q0, q1,
                k + ((size_t)kv_head * n_count + key) * 256);
            block_maximum = fmax(block_maximum, score[local_key]);
        }
        float next_maximum = fmax(maximum, block_maximum);
        float alpha = maximum == -INFINITY ? 0.0f
                      : exp(maximum - next_maximum);
        float beta[8];
        float block_denominator = 0.0f;
        for (int local_key = 0; local_key < keys; local_key++) {
            beta[local_key] = exp(score[local_key] - next_maximum);
            block_denominator += beta[local_key];
        }
        denominator = denominator * alpha + block_denominator;
        acc0 *= alpha;
        acc1 *= alpha;
        for (int local_key = 0; local_key < keys; local_key++) {
            __global const ushort *source = (__global const ushort *)(
                v + ((size_t)kv_head * n_count + key0 + local_key) * 256);
            float weight = beta[local_key];
            acc0 += weight * convert_float8(as_half8(
                intel_sub_group_block_read_us8(source)));
            acc1 += weight * convert_float8(as_half8(
                intel_sub_group_block_read_us8(source + 128)));
        }
        maximum = next_maximum;
    }
    __global uint *target = (__global uint *)(
        out + ((size_t)head * m_count + query) * 256);
    intel_sub_group_block_write8(target,
                                 as_uint8(acc0 / denominator));
    intel_sub_group_block_write8(target + 128,
                                 as_uint8(acc1 / denominator));
}

__attribute__((intel_reqd_sub_group_size(16)))
__kernel void xe_prefill_heads_q8(__global const float *heads,
                                  __global char *quantized,
                                  __global half *scale,
                                  __global short *sigma,
                                  int rows,
                                  int head_count,
                                  int dimension) {
    int width = head_count * dimension;
    int block_count = width / 32;
    int task = get_group_id(0) * 8 + get_sub_group_id();
    int lane = get_sub_group_local_id();
    int row = task / block_count;
    int block = task - row * block_count;
    if (row >= rows) return;
    int column0 = block * 32 + lane;
    int column1 = column0 + 16;
    int head0 = column0 / dimension;
    int d0 = column0 - head0 * dimension;
    int head1 = column1 / dimension;
    int d1 = column1 - head1 * dimension;
    float value0 = heads[((size_t)head0 * rows + row) * dimension + d0];
    float value1 = heads[((size_t)head1 * rows + row) * dimension + d1];
    float maximum = sub_group_reduce_max(fmax(fabs(value0), fabs(value1)));
    volatile float d = maximum / 127.0f;
    volatile float inverse = d == 0.0f ? 0.0f : 1.0f / d;
    volatile float scaled0 = value0 * inverse;
    volatile float scaled1 = value1 * inverse;
    int q0 = convert_int(round(scaled0));
    int q1 = convert_int(round(scaled1));
    size_t base = ((size_t)row * block_count + block) * 32;
    quantized[base + lane] = (char)q0;
    quantized[base + lane + 16] = (char)q1;
    int sum = sub_group_reduce_add(q0 + q1);
    if (lane == 0) {
        size_t index = (size_t)row * block_count + block;
        scale[index] = (half)d;
        sigma[index] = (short)sum;
    }
}

__kernel void xe_prefill_rms_residual(__global const float *input,
                                      __global const float *weight,
                                      __global const float *residual,
                                      __global float *output,
                                      int rows,
                                      int width,
                                      float epsilon) {
    __local float partial[128];
    int row = get_group_id(0);
    int lid = get_local_id(0);
    if (row >= rows) return;
    float sum = 0.0f;
    for (int column = lid; column < width; column += 128) {
        float value = input[(size_t)row * width + column];
        sum += value * value;
    }
    partial[lid] = sum;
    barrier(CLK_LOCAL_MEM_FENCE);
    for (int stride = 64; stride > 0; stride >>= 1) {
        if (lid < stride) partial[lid] += partial[lid + stride];
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    float scale = rsqrt(partial[0] / (float)width + epsilon);
    for (int column = lid; column < width; column += 128) {
        size_t index = (size_t)row * width + column;
        output[index] = input[index] * scale * weight[column] + residual[index];
    }
}

__kernel void xe_prefill_swa_stage(__global const half *ring_k,
                                   __global const half *ring_v,
                                   __global const half *batch_k,
                                   __global const half *batch_v,
                                   __global half *stage_k,
                                   __global half *stage_v,
                                   int m_count,
                                   int dimension,
                                   int kv_heads,
                                   int batch_start,
                                   int stage_base,
                                   int stage_count,
                                   int capacity) {
    size_t index = get_global_id(0);
    size_t elements = (size_t)kv_heads * stage_count * dimension;
    if (index >= elements) return;
    int d = index % dimension;
    size_t row = index / dimension;
    int local_key = row % stage_count;
    int kv_head = row / stage_count;
    int key = stage_base + local_key;
    size_t source = key >= batch_start
        ? ((size_t)kv_head * m_count + key - batch_start) * dimension + d
        : ((size_t)kv_head * capacity + (key & (capacity - 1))) * dimension + d;
    stage_k[index] = key >= batch_start ? batch_k[source] : ring_k[source];
    stage_v[index] = key >= batch_start ? batch_v[source] : ring_v[source];
}

__kernel void xe_prefill_swa_commit(__global half *ring_k,
                                    __global half *ring_v,
                                    __global const half *batch_k,
                                    __global const half *batch_v,
                                    int m_count,
                                    int dimension,
                                    int kv_heads,
                                    int batch_start,
                                    int capacity) {
    size_t index = get_global_id(0);
    size_t elements = (size_t)kv_heads * m_count * dimension;
    if (index >= elements) return;
    int d = index % dimension;
    size_t row = index / dimension;
    int query = row % m_count;
    int kv_head = row / m_count;
    size_t target = ((size_t)kv_head * capacity
                     + ((batch_start + query) & (capacity - 1))) * dimension + d;
    ring_k[target] = batch_k[index];
    ring_v[target] = batch_v[index];
}

__attribute__((intel_reqd_sub_group_size(16)))
__kernel void xe_prefill_ffn_input_q8(
                                      __global const float *input,
                                      __global const float *dense_weight,
                                      __global const float *moe_weight,
                                      __global const float *router_weight,
                                      __global const float *row_scale,
                                      __global char *dense_q,
                                      __global half *dense_d,
                                      __global short *dense_s,
                                      __global char *moe_q,
                                      __global half *moe_d,
                                      __global short *moe_s,
                                      __global float *router_input,
                                      float router_scale,
                                      int rows,
                                      int width) {
    int block_count = width / 32;
    int task = get_group_id(0) * 8 + get_sub_group_id();
    int lane = get_sub_group_local_id();
    int row = task / block_count;
    int block = task - row * block_count;
    if (row >= rows) return;
    float dense[2];
    float moe[2];
    float rms = row_scale[row];
    #pragma unroll
    for (int half_index = 0; half_index < 2; half_index++) {
        int column = block * 32 + lane + half_index * 16;
        float x = input[(size_t)row * width + column];
        dense[half_index] = x * rms * dense_weight[column];
        moe[half_index] = x * rms * moe_weight[column];
    }
    float dense_max = sub_group_reduce_max(
        fmax(fabs(dense[0]), fabs(dense[1])));
    float moe_max = sub_group_reduce_max(fmax(fabs(moe[0]), fabs(moe[1])));
    float dd = dense_max / 127.0f;
    float md = moe_max / 127.0f;
    float dense_inverse = dd == 0.0f ? 0.0f : 1.0f / dd;
    float moe_inverse = md == 0.0f ? 0.0f : 1.0f / md;
    int dense_q0 = convert_int(round(dense[0] * dense_inverse));
    int dense_q1 = convert_int(round(dense[1] * dense_inverse));
    int moe_q0 = convert_int(round(moe[0] * moe_inverse));
    int moe_q1 = convert_int(round(moe[1] * moe_inverse));
    size_t base = ((size_t)row * block_count + block) * 32;
    dense_q[base + lane] = (char)dense_q0;
    dense_q[base + lane + 16] = (char)dense_q1;
    moe_q[base + lane] = (char)moe_q0;
    moe_q[base + lane + 16] = (char)moe_q1;
    int dense_sum = sub_group_reduce_add(dense_q0 + dense_q1);
    int moe_sum = sub_group_reduce_add(moe_q0 + moe_q1);
    if (lane == 0) {
        size_t index = (size_t)row * block_count + block;
        dense_d[index] = (half)dd;
        dense_s[index] = (short)dense_sum;
        moe_d[index] = (half)md;
        moe_s[index] = (short)moe_sum;
    }
}

__attribute__((intel_reqd_sub_group_size(16)))
__kernel void xe_prefill_geglu_q8(__global const float *gate,
                                  __global const float *up,
                                  __global char *quantized,
                                  __global half *scale,
                                  __global short *sigma,
                                  int rows,
                                  int width) {
    int block_count = width / 32;
    int task = get_group_id(0) * 8 + get_sub_group_id();
    int lane = get_sub_group_local_id();
    int row = task / block_count;
    int block = task - row * block_count;
    if (row >= rows) return;
    int column = block * 32 + lane;
    float values[2];
    #pragma unroll
    for (int half_index = 0; half_index < 2; half_index++) {
        int j = column + half_index * 16;
        float x = gate[(size_t)row * width + j];
        float activated;
        if (x <= -10.0f) activated = 0.0f;
        else if (x >= 10.0f) activated = x;
        else {
            float h = (float)convert_half(x);
            float inner = 0.7978845608028654f
                          * (h + 0.044715f * h * h * h);
            activated = (float)convert_half(
                0.5f * h * (1.0f + tanh(inner)));
        }
        values[half_index] = activated * up[(size_t)row * width + j];
    }
    float maximum = sub_group_reduce_max(
        fmax(fabs(values[0]), fabs(values[1])));
    float d = maximum / 127.0f;
    float inverse = d == 0.0f ? 0.0f : 1.0f / d;
    int q0 = convert_int(round(values[0] * inverse));
    int q1 = convert_int(round(values[1] * inverse));
    size_t base = ((size_t)row * block_count + block) * 32;
    quantized[base + lane] = (char)q0;
    quantized[base + lane + 16] = (char)q1;
    int sum = sub_group_reduce_add(q0 + q1);
    if (lane == 0) {
        size_t index = (size_t)row * block_count + block;
        scale[index] = (half)d;
        sigma[index] = (short)sum;
    }
}

__kernel void xe_prefill_router_gemm(__global const float *input,
                                     __global const float *input_weight,
                                     __global const float *row_scale,
                                     __global const float *weight,
                                     __global float *logits,
                                     float input_scale,
                                     int rows,
                                     int width,
                                     int experts) {
    __local float local_input[16 * 32];
    __local float local_weight[16 * 32];
    int lid = get_local_id(0);
    int local_row = lid >> 3;
    int local_column = (lid & 7) * 2;
    int row = get_group_id(0) * 16 + local_row;
    int column0 = get_group_id(1) * 16 + local_column;
    float sum0 = 0.0f;
    float sum1 = 0.0f;
    for (int k0 = 0; k0 < width; k0 += 32) {
        for (int x = lid; x < 16 * 32; x += 128) {
            int tile_row = x >> 5;
            int k = x & 31;
            int input_row = get_group_id(0) * 16 + tile_row;
            local_input[x] = input_row < rows
                ? input[(size_t)input_row * width + k0 + k]
                  * row_scale[input_row] * input_scale
                  * input_weight[k0 + k] : 0.0f;
            local_weight[x] = weight[(size_t)(get_group_id(1) * 16 + tile_row)
                                     * width + k0 + k];
        }
        barrier(CLK_LOCAL_MEM_FENCE);
        #pragma unroll
        for (int k = 0; k < 32; k++) {
            float value = local_input[local_row * 32 + k];
            sum0 += value * local_weight[local_column * 32 + k];
            sum1 += value * local_weight[(local_column + 1) * 32 + k];
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    if (row < rows) {
        logits[(size_t)row * experts + column0] = sum0;
        logits[(size_t)row * experts + column0 + 1] = sum1;
    }
}

__attribute__((intel_reqd_sub_group_size(16)))
__kernel void xe_prefill_router_top8(__global const float *logits,
                                     __global int *route_expert,
                                     __global float *route_weight,
                                     int rows,
                                     int experts) {
    int row = get_group_id(0) * 8 + get_sub_group_id();
    int lane = get_sub_group_local_id();
    if (row >= rows) return;
    float values[8];
    uchar used[8] = {0};
    #pragma unroll
    for (int i = 0; i < 8; i++)
        values[i] = logits[(size_t)row * experts + lane * 8 + i];
    float selected_value[8];
    int selected_index[8];
    #pragma unroll
    for (int slot = 0; slot < 8; slot++) {
        float best = -INFINITY;
        int index = 0x7fffffff;
        #pragma unroll
        for (int i = 0; i < 8; i++) {
            int expert = lane * 8 + i;
            if (!used[i] && (values[i] > best
                             || (values[i] == best && expert < index))) {
                best = values[i];
                index = expert;
            }
        }
        float global_best = sub_group_reduce_max(best);
        int candidate = best == global_best ? index : 0x7fffffff;
        int global_index = sub_group_reduce_min(candidate);
        if (global_index / 8 == lane) used[global_index & 7] = 1;
        selected_value[slot] = global_best;
        selected_index[slot] = global_index;
    }
    if (lane == 0) {
        float sum = 0.0f;
        float weights[8];
        #pragma unroll
        for (int slot = 0; slot < 8; slot++) {
            weights[slot] = exp(selected_value[slot] - selected_value[0]);
            sum += weights[slot];
        }
        #pragma unroll
        for (int slot = 0; slot < 8; slot++) {
            route_expert[row * 8 + slot] = selected_index[slot];
            route_weight[row * 8 + slot] = weights[slot] / sum;
        }
    }
}

__kernel void xe_prefill_route_reset(__global int *expert_count,
                                     __global int *cursor,
                                     __global int *tile_expert,
                                     __global int *tile_m0) {
    int index = get_global_id(0);
    if (index < 128) {
        expert_count[index] = 0;
        cursor[index] = 0;
    }
    if (index < 512) {
        tile_expert[index] = -1;
        tile_m0[index] = 0;
    }
}

__kernel void xe_prefill_route_count(__global const int *route_expert,
                                     __global int *expert_count,
                                     int routes) {
    int route = get_global_id(0);
    if (route < routes)
        atomic_inc((volatile __global unsigned int *)
                   &expert_count[route_expert[route]]);
}

__kernel void xe_prefill_route_prefix(__global const int *expert_count,
                                      __global int *token_offset,
                                      __global int *cursor,
                                      __global int *tile_expert,
                                      __global int *tile_m0,
                                      int tile_rows) {
    if (get_global_id(0) != 0) return;
    int offset = 0;
    int tile = 0;
    token_offset[0] = 0;
    for (int expert = 0; expert < 128; expert++) {
        int count = expert_count[expert];
        cursor[expert] = offset;
        offset += count;
        token_offset[expert + 1] = offset;
        if (tile_rows) {
            for (int m0 = 0; m0 < count; m0 += tile_rows) {
                tile_expert[tile] = expert;
                tile_m0[tile] = m0;
                tile++;
            }
        } else {
            int full_rows = count / 32 * 32;
            if (count - full_rows > 16) full_rows += 32;
            for (int m0 = 0; m0 < full_rows; m0 += 32) {
                tile_expert[tile] = expert;
                tile_m0[tile] = m0;
                tile++;
            }
        }
    }
    if (!tile_rows) {
        int tile16 = 256;
        int tile8 = 384;
        for (int expert = 0; expert < 128; expert++) {
            int count = expert_count[expert];
            int full_rows = count / 32 * 32;
            if (count - full_rows > 16) full_rows += 32;
            if (full_rows < count) {
                int tile = count - full_rows <= 8 ? tile8++ : tile16++;
                tile_expert[tile] = expert;
                tile_m0[tile] = full_rows;
            }
        }
    }
}

__kernel void xe_prefill_route_scatter(__global const int *route_expert,
                                       __global int *cursor,
                                       __global int *packed_route,
                                       __global int *route_packed,
                                       int routes) {
    int route = get_global_id(0);
    if (route >= routes) return;
    int expert = route_expert[route];
    unsigned int packed = atomic_inc(
        (volatile __global unsigned int *)&cursor[expert]);
    packed_route[packed] = route;
    route_packed[route] = packed;
}

__kernel void xe_prefill_route_pack(__global const char *source_q,
                                    __global const half *source_d,
                                    __global const short *source_s,
                                    __global char *packed_q,
                                    __global half *packed_d,
                                    __global short *packed_s,
                                    __global const int *packed_route,
                                    int blocks,
                                    int routes) {
    int packed = get_group_id(0);
    int lid = get_local_id(0);
    if (packed >= routes) return;
    int token = packed_route[packed] >> 3;
    for (int k = lid; k < blocks * 32; k += 128)
        packed_q[(size_t)packed * blocks * 32 + k] =
            source_q[(size_t)token * blocks * 32 + k];
    for (int block = lid; block < blocks; block += 128) {
        packed_d[(size_t)packed * blocks + block] =
            source_d[(size_t)token * blocks + block];
        packed_s[(size_t)packed * blocks + block] =
            source_s[(size_t)token * blocks + block];
    }
}

static inline void xe_prefill_q4q8_grouped_n128_body(
                                        __global const uchar *wq,
                                        __global const half *wd,
                                        __global const char *aq,
                                        __global const half *ad,
                                        __global const short *as,
                                        __global float *out,
                                        __global const int *expert_count,
                                        __global const int *token_offset,
                                        __global const int *tile_expert,
                                        __global const int *tile_m0,
    int n_count,
    int blocks,
    __local char *la,
    __local char *lw,
    __local half *lad,
    __local half *lwd) {
    int lid = get_local_id(0);
    int tile = get_group_id(0);
    int expert = tile_expert[tile];
    if (expert < 0) return;
    int local_m0 = tile_m0[tile];
    int packed_m0 = token_offset[expert] + local_m0;
    int count = expert_count[expert];
    int subgroup = lid >> 4;
    int lane = lid & 15;
    int wm = (subgroup & 3) * 8;
    int wn = (subgroup >> 2) * 32 + lane;
    int n0 = get_group_id(1) * 128;
    float acc[8][2] = {{0.0f}};
    for (int kb = 0; kb < blocks * 32; kb += 32) {
        for (int x = lid; x < XE_PREFILL_TM * 32; x += 256) {
            int lm = x >> 5;
            int lk = x & 31;
            int gm = packed_m0 + lm;
            la[x] = local_m0 + lm < count
                    ? aq[(size_t)gm * blocks * 32 + kb + lk] : 0;
        }
        for (int x = lid; x < 128 * 4; x += 256) {
            int ln = x >> 2;
            int c = x & 3;
            int gn = n0 + ln;
            __global const uint *source = (__global const uint *)(
                wq + (((size_t)expert * (n_count >> 3) + (gn >> 3))
                      * blocks + kb / 32) * 128
                + c * 32 + (gn & 7) * 4);
            uchar4 packed = as_uchar4(*source);
            int base = ln * 32 + c * 4;
            vstore4(convert_char4(packed & (uchar4)(15)) - (char4)(8),
                    0, lw + base);
            vstore4(convert_char4(packed >> (uchar4)(4)) - (char4)(8),
                    0, lw + base + 16);
        }
        if (lid < XE_PREFILL_TM) {
            int lm = lid;
            int gm = packed_m0 + lm;
            int valid = local_m0 + lm < count;
            lad[lid] = valid
                       ? ad[(size_t)gm * blocks + kb / 32] : (half)0;
        }
        if (lid < 128) {
            int ln = lid;
            int gn = n0 + ln;
            lwd[ln] =
                wd[(((size_t)expert * (n_count >> 3) + (gn >> 3))
                    * blocks + kb / 32) * 8 + (gn & 7)];
        }
        barrier(CLK_LOCAL_MEM_FENCE);
        #pragma unroll
        for (int im = 0; im < 8; im++) {
            int lm = wm + im;
            float da = (float)lad[lm];
            int integer0 = 0;
            int integer1 = 0;
            #pragma unroll
            for (int c = 0; c < 8; c++) {
                char4 activation = vload4(0, la + lm * 32 + c * 4);
                integer0 += dot(vload4(
                    0, lw + wn * 32 + c * 4), activation);
                integer1 += dot(vload4(
                    0, lw + (wn + 16) * 32 + c * 4), activation);
            }
            acc[im][0] += (float)integer0 * da * (float)lwd[wn];
            acc[im][1] += (float)integer1 * da * (float)lwd[wn + 16];
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    #pragma unroll
    for (int im = 0; im < 8; im++) {
        int lm = wm + im;
        if (local_m0 + lm >= count) continue;
        int gm = packed_m0 + lm;
        out[(size_t)gm * n_count + n0 + wn] = acc[im][0];
        out[(size_t)gm * n_count + n0 + wn + 16] = acc[im][1];
    }
}

__attribute__((intel_reqd_sub_group_size(16)))
__kernel void xe_prefill_q4q8_grouped_n128(
                                        __global const uchar *wq,
                                        __global const half *wd,
                                        __global const char *aq,
                                        __global const half *ad,
                                        __global const short *as,
                                        __global float *out,
                                        __global const int *expert_count,
                                        __global const int *token_offset,
                                        __global const int *tile_expert,
                                        __global const int *tile_m0,
                                        int n_count,
                                        int blocks) {
    __local char la[XE_PREFILL_TM * 32];
    __local char lw[128 * 32];
    __local half lad[XE_PREFILL_TM];
    __local half lwd[128];
    xe_prefill_q4q8_grouped_n128_body(
        wq, wd, aq, ad, as, out, expert_count, token_offset,
        tile_expert, tile_m0, n_count, blocks, la, lw, lad, lwd);
}

__attribute__((intel_reqd_sub_group_size(16)))
__kernel void xe_prefill_q4q8_grouped_gate_n128(
                                        __global const uchar *wq,
                                        __global const half *wd,
                                        __global const char *aq,
                                        __global const half *ad,
                                        __global const short *as,
                                        __global float *out,
                                        __global const int *expert_count,
                                        __global const int *token_offset,
                                        __global const int *tile_expert,
                                        __global const int *tile_m0,
                                        int n_count,
                                        int blocks) {
    __local char la[XE_PREFILL_TM * 32];
    __local char lw[128 * 32];
    __local half lad[XE_PREFILL_TM];
    __local half lwd[128];
    xe_prefill_q4q8_grouped_n128_body(
        wq, wd, aq, ad, as, out, expert_count, token_offset,
        tile_expert, tile_m0, 1408, 88, la, lw, lad, lwd);
}

__attribute__((intel_reqd_sub_group_size(16)))
__kernel void xe_prefill_q4q8_grouped_down_n128(
                                        __global const uchar *wq,
                                        __global const half *wd,
                                        __global const char *aq,
                                        __global const half *ad,
                                        __global const short *as,
                                        __global float *out,
                                        __global const int *expert_count,
                                        __global const int *token_offset,
                                        __global const int *tile_expert,
                                        __global const int *tile_m0,
                                        int n_count,
                                        int blocks) {
    __local char la[XE_PREFILL_TM * 32];
    __local char lw[128 * 32];
    __local half lad[XE_PREFILL_TM];
    __local half lwd[128];
    xe_prefill_q4q8_grouped_n128_body(
        wq, wd, aq, ad, as, out, expert_count, token_offset,
        tile_expert, tile_m0, 2816, 22, la, lw, lad, lwd);
}

__attribute__((intel_reqd_sub_group_size(16)))
__kernel void xe_prefill_q4q8_grouped_m16_n64(
                                        __global const uchar *wq,
                                        __global const half *wd,
                                        __global const char *aq,
                                        __global const half *ad,
                                        __global const short *as,
                                        __global float *out,
                                        __global const int *expert_count,
                                        __global const int *token_offset,
                                        __global const int *tile_expert,
                                        __global const int *tile_m0,
                                        int n_count,
                                        int blocks) {
    __local char la[16 * XE_PREFILL_KB];
    __local uchar lw[32 * (XE_PREFILL_KB / 2)];
    __local half lad[16 * 2];
    __local short las[16 * 2];
    __local half lwd[32 * 2];
    int lid = get_local_id(0);
    int tile = get_group_id(0);
    int expert = tile_expert[tile];
    if (expert < 0) return;
    int local_m0 = tile_m0[tile];
    int packed_m0 = token_offset[expert] + local_m0;
    int count = expert_count[expert];
    int wm = lid >> 4;
    int wn = lid & 15;
    int n0 = get_group_id(1) * 32;
    float acc[2][2] = {{0.0f}};

    for (int kb = 0; kb < blocks * 32; kb += XE_PREFILL_KB) {
        for (int x = lid; x < 16 * XE_PREFILL_KB; x += 128) {
            int lm = x / XE_PREFILL_KB;
            int lk = x - lm * XE_PREFILL_KB;
            int gm = packed_m0 + lm;
            la[x] = local_m0 + lm < count
                    ? aq[(size_t)gm * blocks * 32 + kb + lk] : 0;
        }
        #pragma unroll
        for (int seg = 0; seg < 8; seg++) {
            int lng = seg >> 1;
            int lb = seg & 1;
            int chunk = lid >> 5;
            int rem = lid & 31;
            int row8 = rem >> 2;
            int j = rem & 3;
            int ln = lng * 8 + row8;
            lw[ln * 32 + lb * 16 + chunk * 4 + j] =
                wq[(((size_t)expert * (n_count >> 3) + (n0 >> 3) + lng)
                     * blocks + kb / 32 + lb) * 128 + lid];
        }
        if (lid < 16 * 2) {
            int lm = lid >> 1;
            int lb = lid & 1;
            int gm = packed_m0 + lm;
            int valid = local_m0 + lm < count;
            lad[lid] = valid
                       ? ad[(size_t)gm * blocks + kb / 32 + lb] : (half)0;
            las[lid] = valid
                       ? as[(size_t)gm * blocks + kb / 32 + lb] : (short)0;
        }
        if (lid < 64) {
            int seg = lid >> 3;
            int row8 = lid & 7;
            int lng = seg >> 1;
            int lb = seg & 1;
            lwd[(lng * 8 + row8) * 2 + lb] =
                wd[(((size_t)expert * (n_count >> 3) + (n0 >> 3) + lng)
                    * blocks + kb / 32 + lb) * 8 + row8];
        }
        barrier(CLK_LOCAL_MEM_FENCE);
        #pragma unroll
        for (int lb = 0; lb < 2; lb++) {
            #pragma unroll
            for (int im = 0; im < 2; im++) {
                int lm = wm + im * 8;
                int correction = 8 * (int)las[lm * 2 + lb];
                float da = (float)lad[lm * 2 + lb];
                #pragma unroll
                for (int in = 0; in < 2; in++) {
                    int ln = wn + in * 16;
                    int integer = 0;
                    #pragma unroll
                    for (int c = 0; c < 4; c++) {
                        uchar4 packed = vload4(
                            0, lw + ln * 32 + lb * 16 + c * 4);
                        char4 lo = vload4(
                            0, la + lm * XE_PREFILL_KB + lb * 32 + c * 4);
                        char4 hi = vload4(
                            0, la + lm * XE_PREFILL_KB + lb * 32 + 16 + c * 4);
                        integer += dot(packed & (uchar4)(15), lo)
                                   + dot(packed >> (uchar4)(4), hi);
                    }
                    acc[im][in] += (float)(integer - correction) * da
                                   * (float)lwd[ln * 2 + lb];
                }
            }
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    #pragma unroll
    for (int im = 0; im < 2; im++) {
        int lm = wm + im * 8;
        if (local_m0 + lm >= count) continue;
        int gm = packed_m0 + lm;
        #pragma unroll
        for (int in = 0; in < 2; in++)
            out[(size_t)gm * n_count + n0 + wn + in * 16] = acc[im][in];
    }
}

__attribute__((intel_reqd_sub_group_size(16)))
__kernel void xe_prefill_q4q8_grouped_m16_n128(
                                        __global const uchar *wq,
                                        __global const half *wd,
                                        __global const char *aq,
                                        __global const half *ad,
                                        __global const short *as,
                                        __global float *out,
                                        __global const int *expert_count,
                                        __global const int *token_offset,
                                        __global const int *tile_expert,
                                        __global const int *tile_m0,
    int n_count,
    int blocks) {
    __local char la[16 * XE_PREFILL_KB];
    __local uint lw[128 * 8];
    __local half lad[16 * 2];
    __local short las[16 * 2];
    __local half lwd[128 * 2];
    int lid = get_local_id(0);
    int tile = get_group_id(0);
    int expert = tile_expert[tile];
    if (expert < 0) return;
    int local_m0 = tile_m0[tile];
    int packed_m0 = token_offset[expert] + local_m0;
    int count = expert_count[expert];
    int subgroup = lid >> 4;
    int lane = lid & 15;
    int wm = (subgroup & 1) * 8;
    int wn = (subgroup >> 1) * 32 + lane;
    int n0 = get_group_id(1) * 128;
    float acc[8][2] = {{0.0f}};
    for (int kb = 0; kb < blocks * 32; kb += XE_PREFILL_KB) {
        for (int x = lid; x < 16 * XE_PREFILL_KB; x += 128) {
            int lm = x / XE_PREFILL_KB;
            int lk = x - lm * XE_PREFILL_KB;
            int gm = packed_m0 + lm;
            la[x] = local_m0 + lm < count
                    ? aq[(size_t)gm * blocks * 32 + kb + lk] : 0;
        }
        for (int x = lid; x < 128 * 8; x += 128) {
            int ln = x >> 3;
            int q = x & 7;
            int lb = q >> 2;
            int c = q & 3;
            int gn = n0 + ln;
            __global const uint *source = (__global const uint *)(
                wq + (((size_t)expert * (n_count >> 3) + (gn >> 3))
                      * blocks + kb / 32 + lb) * 128
                + c * 32 + (gn & 7) * 4);
            lw[(lb * 4 + c) * 128 + ln] = *source;
        }
        if (lid < 16 * 2) {
            int lm = lid >> 1;
            int lb = lid & 1;
            int gm = packed_m0 + lm;
            int valid = local_m0 + lm < count;
            lad[lid] = valid
                       ? ad[(size_t)gm * blocks + kb / 32 + lb] : (half)0;
            las[lid] = valid
                       ? as[(size_t)gm * blocks + kb / 32 + lb] : (short)0;
        }
        for (int x = lid; x < 128 * 2; x += 128) {
            int ln = x >> 1;
            int lb = x & 1;
            int gn = n0 + ln;
            lwd[lb * 128 + ln] =
                wd[(((size_t)expert * (n_count >> 3) + (gn >> 3))
                    * blocks + kb / 32 + lb) * 8 + (gn & 7)];
        }
        barrier(CLK_LOCAL_MEM_FENCE);
        #pragma unroll
        for (int lb = 0; lb < 2; lb++) {
            uchar4 packed00 = as_uchar4(lw[(lb * 4) * 128 + wn]);
            uchar4 packed01 = as_uchar4(lw[(lb * 4 + 1) * 128 + wn]);
            uchar4 packed02 = as_uchar4(lw[(lb * 4 + 2) * 128 + wn]);
            uchar4 packed03 = as_uchar4(lw[(lb * 4 + 3) * 128 + wn]);
            uchar4 packed10 = as_uchar4(lw[(lb * 4) * 128 + wn + 16]);
            uchar4 packed11 = as_uchar4(lw[(lb * 4 + 1) * 128 + wn + 16]);
            uchar4 packed12 = as_uchar4(lw[(lb * 4 + 2) * 128 + wn + 16]);
            uchar4 packed13 = as_uchar4(lw[(lb * 4 + 3) * 128 + wn + 16]);
            #pragma unroll
            for (int im = 0; im < 8; im++) {
                int lm = wm + im;
                int correction = 8 * (int)las[lm * 2 + lb];
                float da = (float)lad[lm * 2 + lb];
                char4 lo0 = vload4(
                    0, la + lm * XE_PREFILL_KB + lb * 32);
                char4 lo1 = vload4(
                    0, la + lm * XE_PREFILL_KB + lb * 32 + 4);
                char4 lo2 = vload4(
                    0, la + lm * XE_PREFILL_KB + lb * 32 + 8);
                char4 lo3 = vload4(
                    0, la + lm * XE_PREFILL_KB + lb * 32 + 12);
                char4 hi0 = vload4(
                    0, la + lm * XE_PREFILL_KB + lb * 32 + 16);
                char4 hi1 = vload4(
                    0, la + lm * XE_PREFILL_KB + lb * 32 + 20);
                char4 hi2 = vload4(
                    0, la + lm * XE_PREFILL_KB + lb * 32 + 24);
                char4 hi3 = vload4(
                    0, la + lm * XE_PREFILL_KB + lb * 32 + 28);
                int integer0 = dot(packed00 & (uchar4)(15), lo0)
                               + dot(packed00 >> (uchar4)(4), hi0)
                               + dot(packed01 & (uchar4)(15), lo1)
                               + dot(packed01 >> (uchar4)(4), hi1)
                               + dot(packed02 & (uchar4)(15), lo2)
                               + dot(packed02 >> (uchar4)(4), hi2)
                               + dot(packed03 & (uchar4)(15), lo3)
                               + dot(packed03 >> (uchar4)(4), hi3);
                int integer1 = dot(packed10 & (uchar4)(15), lo0)
                               + dot(packed10 >> (uchar4)(4), hi0)
                               + dot(packed11 & (uchar4)(15), lo1)
                               + dot(packed11 >> (uchar4)(4), hi1)
                               + dot(packed12 & (uchar4)(15), lo2)
                               + dot(packed12 >> (uchar4)(4), hi2)
                               + dot(packed13 & (uchar4)(15), lo3)
                               + dot(packed13 >> (uchar4)(4), hi3);
                acc[im][0] += (float)(integer0 - correction) * da
                              * (float)lwd[lb * 128 + wn];
                acc[im][1] += (float)(integer1 - correction) * da
                              * (float)lwd[lb * 128 + wn + 16];
            }
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    #pragma unroll
    for (int im = 0; im < 8; im++) {
        int lm = wm + im;
        if (local_m0 + lm >= count) continue;
        int gm = packed_m0 + lm;
        out[(size_t)gm * n_count + n0 + wn] = acc[im][0];
        out[(size_t)gm * n_count + n0 + wn + 16] = acc[im][1];
    }
}

static inline int xe_prefill_grouped_m8_integer(
        uchar4 packed0, uchar4 packed1, uchar4 packed2, uchar4 packed3,
        __local const char *activation, int correction) {
    char4 lo0 = vload4(0, activation);
    char4 lo1 = vload4(0, activation + 4);
    char4 lo2 = vload4(0, activation + 8);
    char4 lo3 = vload4(0, activation + 12);
    char4 hi0 = vload4(0, activation + 16);
    char4 hi1 = vload4(0, activation + 20);
    char4 hi2 = vload4(0, activation + 24);
    char4 hi3 = vload4(0, activation + 28);
    return dot(packed0 & (uchar4)(15), lo0)
           + dot(packed0 >> (uchar4)(4), hi0)
           + dot(packed1 & (uchar4)(15), lo1)
           + dot(packed1 >> (uchar4)(4), hi1)
           + dot(packed2 & (uchar4)(15), lo2)
           + dot(packed2 >> (uchar4)(4), hi2)
           + dot(packed3 & (uchar4)(15), lo3)
           + dot(packed3 >> (uchar4)(4), hi3) - correction;
}

__attribute__((intel_reqd_sub_group_size(16)))
__kernel void xe_prefill_q4q8_grouped_m8_n128(
                                        __global const uchar *wq,
                                        __global const half *wd,
                                        __global const char *aq,
                                        __global const half *ad,
                                        __global const short *as,
                                        __global float *out,
                                        __global const int *expert_count,
                                        __global const int *token_offset,
                                        __global const int *tile_expert,
                                        __global const int *tile_m0,
    int n_count,
    int blocks) {
    __local char la[8 * XE_PREFILL_KB];
    __local uint lw[128 * 8];
    __local half lad[8 * 2];
    __local short las[8 * 2];
    __local half lwd[128 * 2];
    int lid = get_local_id(0);
    int tile = get_group_id(0);
    int expert = tile_expert[tile];
    if (expert < 0) return;
    int local_m0 = tile_m0[tile];
    int packed_m0 = token_offset[expert] + local_m0;
    int count = expert_count[expert];
    int subgroup = lid >> 4;
    int lane = lid & 15;
    int wn = subgroup * 16 + lane;
    int n0 = get_group_id(1) * 128;
    float8 acc = (float8)(0.0f);
    for (int kb = 0; kb < blocks * 32; kb += XE_PREFILL_KB) {
        for (int x = lid; x < 8 * XE_PREFILL_KB; x += 128) {
            int lm = x / XE_PREFILL_KB;
            int lk = x - lm * XE_PREFILL_KB;
            int gm = packed_m0 + lm;
            la[x] = local_m0 + lm < count
                    ? aq[(size_t)gm * blocks * 32 + kb + lk] : 0;
        }
        for (int x = lid; x < 128 * 8; x += 128) {
            int ln = x >> 3;
            int q = x & 7;
            int lb = q >> 2;
            int c = q & 3;
            int gn = n0 + ln;
            __global const uint *source = (__global const uint *)(
                wq + (((size_t)expert * (n_count >> 3) + (gn >> 3))
                      * blocks + kb / 32 + lb) * 128
                + c * 32 + (gn & 7) * 4);
            lw[(lb * 4 + c) * 128 + ln] = *source;
        }
        if (lid < 8 * 2) {
            int lm = lid >> 1;
            int lb = lid & 1;
            int gm = packed_m0 + lm;
            int valid = local_m0 + lm < count;
            lad[lid] = valid
                       ? ad[(size_t)gm * blocks + kb / 32 + lb] : (half)0;
            las[lid] = valid
                       ? as[(size_t)gm * blocks + kb / 32 + lb] : (short)0;
        }
        for (int x = lid; x < 128 * 2; x += 128) {
            int ln = x >> 1;
            int lb = x & 1;
            int gn = n0 + ln;
            lwd[lb * 128 + ln] =
                wd[(((size_t)expert * (n_count >> 3) + (gn >> 3))
                    * blocks + kb / 32 + lb) * 8 + (gn & 7)];
        }
        barrier(CLK_LOCAL_MEM_FENCE);
        #pragma unroll
        for (int lb = 0; lb < 2; lb++) {
            uchar4 packed0 = as_uchar4(lw[(lb * 4) * 128 + wn]);
            uchar4 packed1 = as_uchar4(lw[(lb * 4 + 1) * 128 + wn]);
            uchar4 packed2 = as_uchar4(lw[(lb * 4 + 2) * 128 + wn]);
            uchar4 packed3 = as_uchar4(lw[(lb * 4 + 3) * 128 + wn]);
            float8 contribution = (float8)(
                (float)xe_prefill_grouped_m8_integer(
                    packed0, packed1, packed2, packed3,
                    la + lb * 32, 8 * (int)las[lb]) * (float)lad[lb],
                (float)xe_prefill_grouped_m8_integer(
                    packed0, packed1, packed2, packed3,
                    la + XE_PREFILL_KB + lb * 32,
                    8 * (int)las[2 + lb]) * (float)lad[2 + lb],
                (float)xe_prefill_grouped_m8_integer(
                    packed0, packed1, packed2, packed3,
                    la + 2 * XE_PREFILL_KB + lb * 32,
                    8 * (int)las[4 + lb]) * (float)lad[4 + lb],
                (float)xe_prefill_grouped_m8_integer(
                    packed0, packed1, packed2, packed3,
                    la + 3 * XE_PREFILL_KB + lb * 32,
                    8 * (int)las[6 + lb]) * (float)lad[6 + lb],
                (float)xe_prefill_grouped_m8_integer(
                    packed0, packed1, packed2, packed3,
                    la + 4 * XE_PREFILL_KB + lb * 32,
                    8 * (int)las[8 + lb]) * (float)lad[8 + lb],
                (float)xe_prefill_grouped_m8_integer(
                    packed0, packed1, packed2, packed3,
                    la + 5 * XE_PREFILL_KB + lb * 32,
                    8 * (int)las[10 + lb]) * (float)lad[10 + lb],
                (float)xe_prefill_grouped_m8_integer(
                    packed0, packed1, packed2, packed3,
                    la + 6 * XE_PREFILL_KB + lb * 32,
                    8 * (int)las[12 + lb]) * (float)lad[12 + lb],
                (float)xe_prefill_grouped_m8_integer(
                    packed0, packed1, packed2, packed3,
                    la + 7 * XE_PREFILL_KB + lb * 32,
                    8 * (int)las[14 + lb]) * (float)lad[14 + lb]);
            acc += contribution * (float)lwd[lb * 128 + wn];
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    #pragma unroll
    for (int lm = 0; lm < 8; lm++) {
        if (local_m0 + lm >= count) continue;
        int gm = packed_m0 + lm;
        out[(size_t)gm * n_count + n0 + wn] = acc[lm];
    }
}

__attribute__((intel_reqd_sub_group_size(16)))
__kernel void xe_prefill_expert_geglu_q8(__global const float *gate_up,
                                         __global char *quantized,
                                         __global half *scale,
                                         __global short *sigma,
                                         int rows,
                                         int width) {
    int block_count = width / 32;
    int task = get_group_id(0) * 8 + get_sub_group_id();
    int lane = get_sub_group_local_id();
    int row = task / block_count;
    int block = task - row * block_count;
    if (row >= rows) return;
    int column = block * 32 + lane;
    float values[2];
    #pragma unroll
    for (int half_index = 0; half_index < 2; half_index++) {
        int j = column + half_index * 16;
        float gate = gate_up[(size_t)row * width * 2 + j];
        float up = gate_up[(size_t)row * width * 2 + width + j];
        float activated;
        if (gate <= -10.0f) activated = 0.0f;
        else if (gate >= 10.0f) activated = gate;
        else {
            float x = (float)convert_half(gate);
            float inner = 0.7978845608028654f
                          * (x + 0.044715f * x * x * x);
            activated = (float)convert_half(
                0.5f * x * (1.0f + tanh(inner)));
        }
        values[half_index] = activated * up;
    }
    float maximum = sub_group_reduce_max(
        fmax(fabs(values[0]), fabs(values[1])));
    float d = maximum / 127.0f;
    float inverse = d == 0.0f ? 0.0f : 1.0f / d;
    int q0 = convert_int(round(values[0] * inverse));
    int q1 = convert_int(round(values[1] * inverse));
    size_t base = ((size_t)row * block_count + block) * 32;
    quantized[base + lane] = (char)q0;
    quantized[base + lane + 16] = (char)q1;
    int sum = sub_group_reduce_add(q0 + q1);
    if (lane == 0) {
        size_t index = (size_t)row * block_count + block;
        scale[index] = (half)d;
        sigma[index] = (short)sum;
    }
}

__kernel void xe_prefill_route_reduce(__global const float *routed,
                                      __global const float *route_weight,
                                      __global const int *route_expert,
                                      __global const int *route_packed,
                                      __global const float *expert_scale,
                                      __global float *reduced,
                                      int dimension) {
    int token = get_group_id(0);
    int lid = get_local_id(0);
    for (int d = lid; d < dimension; d += 128) {
        float sum = 0.0f;
        for (int slot = 0; slot < 8; slot++) {
            int route = token * 8 + slot;
            int packed = route_packed[route];
            sum += route_weight[route] * expert_scale[route_expert[route]]
                   * routed[(size_t)packed * dimension + d];
        }
        reduced[(size_t)token * dimension + d] = sum;
    }
}

__kernel void xe_prefill_ffn_finish(__global const float *dense,
                                    __global const float *moe,
                                    __global const float *dense_weight,
                                    __global const float *moe_weight,
                                    __global const float *combine_weight,
                                    __global const float *residual,
                                    __global const float *layer_scale,
                                    __global float *output,
                                    int rows,
                                    int width,
                                    float epsilon) {
    __local float dense_partial[128];
    __local float moe_partial[128];
    int row = get_group_id(0);
    int lid = get_local_id(0);
    if (row >= rows) return;
    float dense_sum = 0.0f;
    float moe_sum = 0.0f;
    for (int column = lid; column < width; column += 128) {
        size_t index = (size_t)row * width + column;
        float dense_value = dense[index];
        float moe_value = moe[index];
        dense_sum += dense_value * dense_value;
        moe_sum += moe_value * moe_value;
    }
    dense_partial[lid] = dense_sum;
    moe_partial[lid] = moe_sum;
    barrier(CLK_LOCAL_MEM_FENCE);
    for (int stride = 64; stride > 0; stride >>= 1) {
        if (lid < stride) {
            dense_partial[lid] += dense_partial[lid + stride];
            moe_partial[lid] += moe_partial[lid + stride];
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    float dense_scale = rsqrt(dense_partial[0] / (float)width + epsilon);
    float moe_scale = rsqrt(moe_partial[0] / (float)width + epsilon);
    float combine_sum = 0.0f;
    for (int column = lid; column < width; column += 128) {
        size_t index = (size_t)row * width + column;
        float value = dense[index] * dense_scale * dense_weight[column]
                      + moe[index] * moe_scale * moe_weight[column];
        output[index] = value;
        combine_sum += value * value;
    }
    dense_partial[lid] = combine_sum;
    barrier(CLK_LOCAL_MEM_FENCE);
    for (int stride = 64; stride > 0; stride >>= 1) {
        if (lid < stride) dense_partial[lid] += dense_partial[lid + stride];
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    float combine_scale = rsqrt(dense_partial[0] / (float)width + epsilon);
    for (int column = lid; column < width; column += 128) {
        size_t index = (size_t)row * width + column;
        output[index] = (output[index] * combine_scale * combine_weight[column]
                         + residual[index]) * layer_scale[0];
    }
}
