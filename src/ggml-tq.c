/**
 * ggml-tq.c — TurboQuant quantize/dequantize + Hadamard transform
 *
 * TQ3_0: 3-bit Lloyd-Max quantization (8 non-uniform levels)
 * TQ4_0: 4-bit Lloyd-Max quantization (16 non-uniform levels)
 *
 * Both use per-block scaling (ggml_half d = max|x|) and pre-computed
 * optimal codebooks for N(0,1). The Hadamard rotation (applied before
 * quantization) ensures the input distribution matches the codebook.
 */

#include "ggml-tq.h"

#include <assert.h>
#include <math.h>

/* ── 3-bit packing helpers ─────────────────────────────────────────
 *
 * Pack 8 3-bit values (0-7) into 3 bytes.  Repeated 4× for 32 elements.
 *
 *   byte[0] = i0 | (i1 << 3) | (i2_lo << 6)
 *   byte[1] = i2_hi | (i3 << 1) | (i4 << 4) | (i5_lo << 7)
 *   byte[2] = i5_hi | (i6 << 2) | (i7 << 5)
 * ────────────────────────────────────────────────────────────────── */

static inline void pack_3bit_group(const uint8_t idx[8], uint8_t out[3]) {
    out[0] = (uint8_t)(
        (idx[0] & 7)       |
        ((idx[1] & 7) << 3) |
        ((idx[2] & 3) << 6)
    );
    out[1] = (uint8_t)(
        ((idx[2] >> 2) & 1) |
        ((idx[3] & 7) << 1)  |
        ((idx[4] & 7) << 4)  |
        ((idx[5] & 1) << 7)
    );
    out[2] = (uint8_t)(
        ((idx[5] >> 1) & 3) |
        ((idx[6] & 7) << 2)  |
        ((idx[7] & 7) << 5)
    );
}

static inline void unpack_3bit_group(const uint8_t in[3], uint8_t idx[8]) {
    idx[0] =  in[0]       & 7;
    idx[1] = (in[0] >> 3) & 7;
    idx[2] = ((in[0] >> 6) & 3) | ((in[1] & 1) << 2);
    idx[3] = (in[1] >> 1) & 7;
    idx[4] = (in[1] >> 4) & 7;
    idx[5] = ((in[1] >> 7) & 1) | ((in[2] & 3) << 1);
    idx[6] = (in[2] >> 2) & 7;
    idx[7] = (in[2] >> 5) & 7;
}

/* Pack 32 3-bit indices into 12 bytes (4 groups of 8) */
static void pack_3bit_32(const uint8_t indices[32], uint8_t qs[12]) {
    for (int g = 0; g < 4; g++) {
        pack_3bit_group(indices + g * 8, qs + g * 3);
    }
}

/* Unpack 12 bytes into 32 3-bit indices */
static void unpack_3bit_32(const uint8_t qs[12], uint8_t indices[32]) {
    for (int g = 0; g < 4; g++) {
        unpack_3bit_group(qs + g * 3, indices + g * 8);
    }
}

/* ── 2-bit packing helpers ────────────────────────────────────────
 *
 * Pack 32 2-bit values (0-3) into 8 bytes.  4 values per byte.
 * Simplest packing: no cross-byte straddling.
 *
 *   byte = i0 | (i1 << 2) | (i2 << 4) | (i3 << 6)
 * ────────────────────────────────────────────────────────────────── */

static inline void pack_2bit_32(const uint8_t indices[32], uint8_t qs[8]) {
    for (int i = 0; i < 8; i++) {
        qs[i] = (uint8_t)(
            (indices[4*i+0] & 3)        |
            ((indices[4*i+1] & 3) << 2) |
            ((indices[4*i+2] & 3) << 4) |
            ((indices[4*i+3] & 3) << 6)
        );
    }
}

static inline void unpack_2bit_32(const uint8_t qs[8], uint8_t indices[32]) {
    for (int i = 0; i < 8; i++) {
        indices[4*i+0] =  qs[i]       & 3;
        indices[4*i+1] = (qs[i] >> 2) & 3;
        indices[4*i+2] = (qs[i] >> 4) & 3;
        indices[4*i+3] = (qs[i] >> 6) & 3;
    }
}

/* ── Nearest-level lookup ──────────────────────────────────────────── */

/* Find the nearest codebook index for a normalized value in [-1, 1].
 * Linear scan on boundaries — 3 for 2-bit, 7 for 3-bit, 15 for 4-bit.
 * (Binary search possible but not worth it for ≤15 boundaries.) */

static inline uint8_t find_nearest_tq2(float xn) {
    uint8_t idx = 0;
    for (int b = 0; b < 3; b++) {
        if (xn > TQ2_BOUNDARIES[b]) idx = (uint8_t)(b + 1);
    }
    return idx;
}

static inline uint8_t find_nearest_tq3(float xn) {
    uint8_t idx = 0;
    for (int b = 0; b < 7; b++) {
        if (xn > TQ3_BOUNDARIES[b]) idx = (uint8_t)(b + 1);
    }
    return idx;
}

static inline uint8_t find_nearest_tq4(float xn) {
    // Binary search: 4 comparisons instead of 15
    // TQ4_BOUNDARIES is sorted, symmetric around 0
    uint8_t idx;
    if (xn <= TQ4_BOUNDARIES[7]) {             // <= 0.0
        if (xn <= TQ4_BOUNDARIES[3]) {         // <= -0.4025
            if (xn <= TQ4_BOUNDARIES[1]) {     // <= -0.6748
                idx = (xn <= TQ4_BOUNDARIES[0]) ? 0 : 1;
            } else {
                idx = (xn <= TQ4_BOUNDARIES[2]) ? 2 : 3;
            }
        } else {
            if (xn <= TQ4_BOUNDARIES[5]) {     // <= -0.1913
                idx = (xn <= TQ4_BOUNDARIES[4]) ? 4 : 5;
            } else {
                idx = (xn <= TQ4_BOUNDARIES[6]) ? 6 : 7;
            }
        }
    } else {
        if (xn <= TQ4_BOUNDARIES[11]) {        // <= +0.4025
            if (xn <= TQ4_BOUNDARIES[9]) {     // <= +0.1913
                idx = (xn <= TQ4_BOUNDARIES[8]) ? 8 : 9;
            } else {
                idx = (xn <= TQ4_BOUNDARIES[10]) ? 10 : 11;
            }
        } else {
            if (xn <= TQ4_BOUNDARIES[13]) {    // <= +0.6748
                idx = (xn <= TQ4_BOUNDARIES[12]) ? 12 : 13;
            } else {
                idx = (xn <= TQ4_BOUNDARIES[14]) ? 14 : 15;
            }
        }
    }
    return idx;
}

/* ── TQ2_0 quantize / dequantize ───────────────────────────────────── */

void quantize_row_tq2_0_ref(const float * restrict x, block_tq2_0 * restrict y, int64_t k) {
    assert(k % QK_TQ2 == 0);
    const int nb = (int)(k / QK_TQ2);

    for (int i = 0; i < nb; i++) {
        const float * block = x + i * QK_TQ2;

        /* find max absolute value in block */
        float amax = 0.0f;
        for (int j = 0; j < QK_TQ2; j++) {
            float v = fabsf(block[j]);
            if (v > amax) amax = v;
        }

        /* d = amax;  id = 1/d;  normalized x in [-1, 1] */
        const float d  = amax;
        const float id = (d > 0.0f) ? 1.0f / d : 0.0f;
        y[i].d = GGML_FP32_TO_FP16(d);

        /* quantize: find nearest Lloyd-Max level for each normalized value */
        uint8_t indices[QK_TQ2];
        for (int j = 0; j < QK_TQ2; j++) {
            float xn = block[j] * id;
            if (xn < -1.0f) xn = -1.0f;
            if (xn >  1.0f) xn =  1.0f;
            indices[j] = find_nearest_tq2(xn);
        }

        /* pack 32 × 2-bit indices into 8 bytes */
        pack_2bit_32(indices, y[i].qs);
    }
}

void dequantize_row_tq2_0(const block_tq2_0 * restrict x, float * restrict y, int64_t k) {
    assert(k % QK_TQ2 == 0);
    const int nb = (int)(k / QK_TQ2);

    for (int i = 0; i < nb; i++) {
        const float d = GGML_FP16_TO_FP32(x[i].d);

        /* unpack 2-bit indices */
        uint8_t indices[QK_TQ2];
        unpack_2bit_32(x[i].qs, indices);

        /* dequantize: level[index] * d */
        for (int j = 0; j < QK_TQ2; j++) {
            y[i * QK_TQ2 + j] = TQ2_LEVELS[indices[j]] * d;
        }
    }
}

void quantize_row_tq2_0(const float * restrict x, void * restrict y, int64_t k) {
    quantize_row_tq2_0_ref(x, (block_tq2_0 *)y, k);
}

/* ── TQ3_0 quantize / dequantize ───────────────────────────────────── */

void quantize_row_tq3_0_ref(const float * restrict x, block_tq3_0 * restrict y, int64_t k) {
    assert(k % QK_TQ3 == 0);
    const int nb = (int)(k / QK_TQ3);

    for (int i = 0; i < nb; i++) {
        const float * block = x + i * QK_TQ3;

        /* find max absolute value in block */
        float amax = 0.0f;
        for (int j = 0; j < QK_TQ3; j++) {
            float v = fabsf(block[j]);
            if (v > amax) amax = v;
        }

        /* d = amax;  id = 1/d;  normalized x in [-1, 1] */
        const float d  = amax;
        const float id = (d > 0.0f) ? 1.0f / d : 0.0f;
        y[i].d = GGML_FP32_TO_FP16(d);

        /* quantize: find nearest Lloyd-Max level for each normalized value */
        uint8_t indices[QK_TQ3];
        for (int j = 0; j < QK_TQ3; j++) {
            float xn = block[j] * id;  /* normalize to [-1, 1] */
            /* clamp to [-1, 1] for safety (FP16 rounding) */
            if (xn < -1.0f) xn = -1.0f;
            if (xn >  1.0f) xn =  1.0f;
            indices[j] = find_nearest_tq3(xn);
        }

        /* pack 32 × 3-bit indices into 12 bytes */
        pack_3bit_32(indices, y[i].qs);
    }
}

void dequantize_row_tq3_0(const block_tq3_0 * restrict x, float * restrict y, int64_t k) {
    assert(k % QK_TQ3 == 0);
    const int nb = (int)(k / QK_TQ3);

    for (int i = 0; i < nb; i++) {
        const float d = GGML_FP16_TO_FP32(x[i].d);

        /* unpack 3-bit indices */
        uint8_t indices[QK_TQ3];
        unpack_3bit_32(x[i].qs, indices);

        /* dequantize: level[index] * d */
        for (int j = 0; j < QK_TQ3; j++) {
            y[i * QK_TQ3 + j] = TQ3_LEVELS[indices[j]] * d;
        }
    }
}

void quantize_row_tq3_0(const float * restrict x, void * restrict y, int64_t k) {
    quantize_row_tq3_0_ref(x, (block_tq3_0 *)y, k);
}

/* ── TQ4_0 quantize / dequantize ───────────────────────────────────── */

void quantize_row_tq4_0_ref(const float * restrict x, block_tq4_0 * restrict y, int64_t k) {
    assert(k % QK_TQ4 == 0);
    const int nb = (int)(k / QK_TQ4);

    for (int i = 0; i < nb; i++) {
        const float * block = x + i * QK_TQ4;

        float amax = 0.0f;
        for (int j = 0; j < QK_TQ4; j++) {
            float v = fabsf(block[j]);
            if (v > amax) amax = v;
        }

        const float d  = amax;
        const float id = (d > 0.0f) ? 1.0f / d : 0.0f;
        y[i].d = GGML_FP32_TO_FP16(d);

        /* 4-bit: pack two indices per byte (same nibble layout as Q4_0) */
        for (int j = 0; j < QK_TQ4 / 2; j++) {
            float xn0 = block[j]              * id;
            float xn1 = block[j + QK_TQ4 / 2] * id;
            if (xn0 < -1.0f) xn0 = -1.0f;
            if (xn0 >  1.0f) xn0 =  1.0f;
            if (xn1 < -1.0f) xn1 = -1.0f;
            if (xn1 >  1.0f) xn1 =  1.0f;

            uint8_t i0 = find_nearest_tq4(xn0);
            uint8_t i1 = find_nearest_tq4(xn1);
            y[i].qs[j] = i0 | (i1 << 4);
        }
    }
}

void dequantize_row_tq4_0(const block_tq4_0 * restrict x, float * restrict y, int64_t k) {
    assert(k % QK_TQ4 == 0);
    const int nb = (int)(k / QK_TQ4);

    for (int i = 0; i < nb; i++) {
        const float d = GGML_FP16_TO_FP32(x[i].d);

        for (int j = 0; j < QK_TQ4 / 2; j++) {
            const uint8_t i0 = x[i].qs[j] & 0x0F;
            const uint8_t i1 = x[i].qs[j] >> 4;

            y[i * QK_TQ4 + j]              = TQ4_LEVELS[i0] * d;
            y[i * QK_TQ4 + j + QK_TQ4 / 2] = TQ4_LEVELS[i1] * d;
        }
    }
}

void quantize_row_tq4_0(const float * restrict x, void * restrict y, int64_t k) {
    quantize_row_tq4_0_ref(x, (block_tq4_0 *)y, k);
}

/* ── Requantize (for tiered cache migration) ─────────────────────── */

void requantize_tq4_to_tq3(const block_tq4_0 * restrict src, block_tq3_0 * restrict dst, int64_t k) {
    assert(k % QK_TQ4 == 0);
    const int nb = (int)(k / QK_TQ4);
    float tmp[QK_TQ4];
    for (int i = 0; i < nb; i++) {
        dequantize_row_tq4_0(src + i, tmp, QK_TQ4);
        quantize_row_tq3_0_ref(tmp, dst + i, QK_TQ3);
    }
}

void requantize_tq3_to_tq2(const block_tq3_0 * restrict src, block_tq2_0 * restrict dst, int64_t k) {
    assert(k % QK_TQ3 == 0);
    const int nb = (int)(k / QK_TQ3);
    float tmp[QK_TQ3];
    for (int i = 0; i < nb; i++) {
        dequantize_row_tq3_0(src + i, tmp, QK_TQ3);
        quantize_row_tq2_0_ref(tmp, dst + i, QK_TQ2);
    }
}

void requantize_tq4_to_tq2(const block_tq4_0 * restrict src, block_tq2_0 * restrict dst, int64_t k) {
    assert(k % QK_TQ4 == 0);
    const int nb = (int)(k / QK_TQ4);
    float tmp[QK_TQ4];
    for (int i = 0; i < nb; i++) {
        dequantize_row_tq4_0(src + i, tmp, QK_TQ4);
        quantize_row_tq2_0_ref(tmp, dst + i, QK_TQ2);
    }
}

/* ── Hadamard transform ────────────────────────────────────────────── */

void hadamard_transform(float * x, int d) {
    assert(d > 0 && (d & (d - 1)) == 0);  /* d must be power of 2 */

    /* Butterfly algorithm: O(d log d) */
    for (int h = 1; h < d; h <<= 1) {
        for (int i = 0; i < d; i += h * 2) {
            for (int j = i; j < i + h; j++) {
                float a = x[j];
                float b = x[j + h];
                x[j]     = a + b;
                x[j + h] = a - b;
            }
        }
    }

    /* Normalize so H^T H = I */
    float norm = 1.0f / sqrtf((float)d);
    for (int i = 0; i < d; i++) {
        x[i] *= norm;
    }
}

void randomized_hadamard_transform(float * x, int d, uint32_t seed) {
    /* Apply random ±1 diagonal first (breaks input-Hadamard alignment) */
    uint32_t s = seed;
    for (int i = 0; i < d; i++) {
        /* Simple LCG: s = s * 1664525 + 1013904223 */
        s = s * 1664525u + 1013904223u;
        if (s & 0x80000000u) {
            x[i] = -x[i];
        }
    }

    /* Then Hadamard */
    hadamard_transform(x, d);
}

void hadamard_transform_row(float * x, int n, int head_dim) {
    assert(n % head_dim == 0);
    int n_heads = n / head_dim;
    for (int h = 0; h < n_heads; h++) {
        hadamard_transform(x + h * head_dim, head_dim);
    }
}

void randomized_hadamard_transform_row(float * x, int n, int head_dim, uint32_t seed) {
    assert(n % head_dim == 0);
    int n_heads = n / head_dim;
    for (int h = 0; h < n_heads; h++) {
        randomized_hadamard_transform(x + h * head_dim, head_dim, seed + (uint32_t)h);
    }
}
