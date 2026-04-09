/**
 * ggml-tq.h — TurboQuant KV cache quantization types for ggml
 *
 * Implements TQ3_0 (3-bit) and TQ4_0 (4-bit) quantization using:
 *   1. Pre-rotation via Hadamard transform (distributes outlier energy)
 *   2. Lloyd-Max optimal scalar quantization (non-uniform codebook)
 *
 * Block layouts:
 *   block_tq3_0: 14 bytes / 32 elements = 3.5 bits/elem
 *     - ggml_half d (2 bytes): per-block scale = max(|x|)
 *     - uint8_t qs[12]: 32 × 3-bit packed indices
 *
 *   block_tq4_0: 18 bytes / 32 elements = 4.5 bits/elem
 *     - ggml_half d (2 bytes): per-block scale = max(|x|)
 *     - uint8_t qs[16]: 32 × 4-bit packed indices (same layout as Q4_0)
 *
 * The codebook levels are optimal (Lloyd-Max) for N(0,1), normalized to [-1,1].
 * Per-block scale d maps data range to codebook range.
 *
 * Reference: Zandieh et al. "TurboQuant" (2025), Google Research
 */

#ifndef GGML_TQ_H
#define GGML_TQ_H

#include <stdint.h>
#include <stddef.h>
#include <string.h>

#ifdef __cplusplus
extern "C" {
#endif

/* ── FP16 conversion (standalone build) ────────────────────────────── */

#ifdef GGML_TQ_STANDALONE
typedef uint16_t ggml_half;

static inline ggml_half ggml_fp32_to_fp16(float f) {
    uint32_t b;
    memcpy(&b, &f, sizeof(b));
    uint32_t sign = (b >> 16) & 0x8000;
    int32_t  expo = ((b >> 23) & 0xFF) - 127 + 15;
    uint32_t mant = (b >> 13) & 0x3FF;
    if (expo <= 0)      return (ggml_half)(sign);           /* flush to zero */
    if (expo >= 0x1F)   return (ggml_half)(sign | 0x7C00);  /* infinity      */
    return (ggml_half)(sign | (expo << 10) | mant);
}

static inline float ggml_fp16_to_fp32(ggml_half h) {
    uint32_t sign = ((uint32_t)(h & 0x8000)) << 16;
    uint32_t expo = (h >> 10) & 0x1F;
    uint32_t mant = h & 0x3FF;
    if (expo == 0) {
        if (mant == 0) { float f; uint32_t z = sign; memcpy(&f, &z, 4); return f; }
        /* denormal: normalize */
        while (!(mant & 0x400)) { mant <<= 1; expo--; }
        expo++; mant &= 0x3FF;
    }
    uint32_t b = sign | (((expo - 15 + 127) & 0xFF) << 23) | (mant << 13);
    float f;
    memcpy(&f, &b, sizeof(f));
    return f;
}

#define GGML_FP32_TO_FP16(x) ggml_fp32_to_fp16(x)
#define GGML_FP16_TO_FP32(x) ggml_fp16_to_fp32(x)
#else
#include "ggml.h"
#endif

/* ── Block sizes ───────────────────────────────────────────────────── */

#define QK_TQ3 32   /* elements per TQ3 block */
#define QK_TQ4 32   /* elements per TQ4 block */

/* ── Lloyd-Max codebooks for N(0,1), normalized to [-1, 1] ─────────
 *
 * Computed via Lloyd-Max algorithm on N(0,1) with support [-6, 6].
 * Divided by max_level so outer levels = ±1.0.
 * Per-block scale d = max(|block|) maps data to this range.
 *
 * The non-uniform spacing (denser near zero) is provably optimal
 * for Gaussian distributions (minimizes MSE).
 * ──────────────────────────────────────────────────────────────────── */

/* 3-bit: 8 levels, 7 boundaries */
static const float TQ3_LEVELS[8] = {
    -1.0000000f, -0.6245203f, -0.3513239f, -0.1138989f,
    +0.1138989f, +0.3513239f, +0.6245203f, +1.0000000f,
};

static const float TQ3_BOUNDARIES[7] = {
    -0.8122602f, -0.4879221f, -0.2326114f, +0.0000000f,
    +0.2326114f, +0.4879221f, +0.8122602f,
};

/* 4-bit: 16 levels, 15 boundaries */
static const float TQ4_LEVELS[16] = {
    -1.0000000f, -0.7573038f, -0.5923403f, -0.4599576f,
    -0.3450764f, -0.2405254f, -0.1421261f, -0.0470277f,
    +0.0470277f, +0.1421261f, +0.2405254f, +0.3450764f,
    +0.4599576f, +0.5923403f, +0.7573038f, +1.0000000f,
};

static const float TQ4_BOUNDARIES[15] = {
    -0.8786519f, -0.6748221f, -0.5261490f, -0.4025170f,
    -0.2928009f, -0.1913257f, -0.0945769f, +0.0000000f,
    +0.0945769f, +0.1913257f, +0.2928009f, +0.4025170f,
    +0.5261490f, +0.6748221f, +0.8786519f,
};

/* ── Block structures ──────────────────────────────────────────────── */

typedef struct {
    ggml_half d;            /* per-block scale factor (2 bytes)  */
    uint8_t  qs[12];        /* 32 × 3-bit packed indices (12 bytes) */
} block_tq3_0;
/* 14 bytes per 32 elements = 3.5 bits/elem */

typedef struct {
    ggml_half d;            /* per-block scale factor (2 bytes)  */
    uint8_t  qs[QK_TQ4/2];  /* 32 × 4-bit packed nibbles (16 bytes) */
} block_tq4_0;
/* 18 bytes per 32 elements = 4.5 bits/elem */

/* ── Quantize / dequantize ─────────────────────────────────────────── */

void quantize_row_tq3_0_ref(const float * restrict x, block_tq3_0 * restrict y, int64_t k);
void dequantize_row_tq3_0  (const block_tq3_0 * restrict x, float * restrict y, int64_t k);

void quantize_row_tq4_0_ref(const float * restrict x, block_tq4_0 * restrict y, int64_t k);
void dequantize_row_tq4_0  (const block_tq4_0 * restrict x, float * restrict y, int64_t k);

/* Wrappers matching ggml from_float signature: void (*)(const float *, void *, int64_t) */
void quantize_row_tq3_0(const float * restrict x, void * restrict y, int64_t k);
void quantize_row_tq4_0(const float * restrict x, void * restrict y, int64_t k);

/* ── Hadamard transform ────────────────────────────────────────────── */

/**
 * In-place Walsh-Hadamard transform via butterfly algorithm.
 * O(d log d) time, normalized by 1/sqrt(d).
 * d must be a power of 2.
 */
void hadamard_transform(float * x, int d);

/**
 * In-place randomized Hadamard: apply random ±1 diagonal, then Hadamard.
 * Signs generated from seed via simple LCG PRNG.
 */
void randomized_hadamard_transform(float * x, int d, uint32_t seed);

/**
 * Apply Hadamard to each head_dim-sized chunk in a row of length n.
 * n must be divisible by head_dim.
 */
void hadamard_transform_row(float * x, int n, int head_dim);

/**
 * Apply randomized Hadamard to each head_dim-sized chunk.
 * Uses seed + chunk_index for per-chunk randomization.
 */
void randomized_hadamard_transform_row(float * x, int n, int head_dim, uint32_t seed);

#ifdef __cplusplus
}
#endif

#endif /* GGML_TQ_H */
