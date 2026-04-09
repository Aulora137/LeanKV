/**
 * test-tq.c — Correctness tests for TurboQuant (TQ3_0 / TQ4_0)
 *
 * Tests:
 *   1. 3-bit packing roundtrip
 *   2. TQ3_0 quantize/dequantize roundtrip + MSE
 *   3. TQ4_0 quantize/dequantize roundtrip + MSE
 *   4. Hadamard transform orthogonality
 *   5. Randomized Hadamard preserves norms
 *   6. Full pipeline: Hadamard → TQ3 → dequantize → compare
 *   7. TQ3 vs uniform Q3 MSE comparison
 *   8. Codebook symmetry validation
 */

#ifndef GGML_TQ_STANDALONE
#define GGML_TQ_STANDALONE
#endif
#include "ggml-tq.h"

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>

/* ── Helpers ───────────────────────────────────────────────────────── */

static int g_pass = 0, g_fail = 0;

#define CHECK(cond, fmt, ...) do { \
    if (cond) { g_pass++; printf("  PASS: " fmt "\n", ##__VA_ARGS__); } \
    else      { g_fail++; printf("  FAIL: " fmt "\n", ##__VA_ARGS__); } \
} while(0)

/* Simple PRNG for test data */
static uint32_t xorshift(uint32_t * state) {
    uint32_t x = *state;
    x ^= x << 13; x ^= x >> 17; x ^= x << 5;
    *state = x;
    return x;
}

static float rand_normal(uint32_t * state) {
    /* Box-Muller: two uniform → one normal */
    float u1 = (float)(xorshift(state) & 0xFFFF) / 65535.0f;
    float u2 = (float)(xorshift(state) & 0xFFFF) / 65535.0f;
    if (u1 < 1e-10f) u1 = 1e-10f;
    return sqrtf(-2.0f * logf(u1)) * cosf(6.2831853f * u2);
}

static float compute_mse(const float * a, const float * b, int n) {
    double mse = 0.0;
    for (int i = 0; i < n; i++) {
        double d = (double)a[i] - (double)b[i];
        mse += d * d;
    }
    return (float)(mse / n);
}

static float compute_cosine_sim(const float * a, const float * b, int n) {
    double dot = 0.0, na = 0.0, nb = 0.0;
    for (int i = 0; i < n; i++) {
        dot += (double)a[i] * (double)b[i];
        na  += (double)a[i] * (double)a[i];
        nb  += (double)b[i] * (double)b[i];
    }
    if (na < 1e-30 || nb < 1e-30) return 0.0f;
    return (float)(dot / (sqrt(na) * sqrt(nb)));
}

/* ── Test 1: 3-bit pack/unpack roundtrip ───────────────────────────── */

static void test_3bit_packing(void) {
    printf("\n=== Test 1: 3-bit pack/unpack roundtrip ===\n");

    uint8_t indices[32];
    uint8_t packed[12];

    /* Fill with all valid values 0-7 */
    for (int i = 0; i < 32; i++) {
        indices[i] = (uint8_t)(i % 8);
    }

    /* Pack */
    for (int g = 0; g < 4; g++) {
        uint8_t out[3];
        /* inline the pack function for testing */
        out[0] = (uint8_t)(
            (indices[g*8+0] & 7) |
            ((indices[g*8+1] & 7) << 3) |
            ((indices[g*8+2] & 3) << 6)
        );
        out[1] = (uint8_t)(
            ((indices[g*8+2] >> 2) & 1) |
            ((indices[g*8+3] & 7) << 1) |
            ((indices[g*8+4] & 7) << 4) |
            ((indices[g*8+5] & 1) << 7)
        );
        out[2] = (uint8_t)(
            ((indices[g*8+5] >> 1) & 3) |
            ((indices[g*8+6] & 7) << 2) |
            ((indices[g*8+7] & 7) << 5)
        );
        packed[g*3+0] = out[0];
        packed[g*3+1] = out[1];
        packed[g*3+2] = out[2];
    }

    /* Unpack using the library function */
    /* Use a temporary block to test via the quantize/dequantize path */
    block_tq3_0 blk;
    memcpy(blk.qs, packed, 12);
    blk.d = GGML_FP32_TO_FP16(1.0f);

    float output[32];
    dequantize_row_tq3_0(&blk, output, 32);

    /* Check that each recovered index matches */
    int all_ok = 1;
    for (int i = 0; i < 32; i++) {
        /* Find which level the output corresponds to */
        int idx_orig = indices[i];
        float expected = TQ3_LEVELS[idx_orig] * 1.0f;
        if (fabsf(output[i] - expected) > 1e-4f) {
            printf("  Mismatch at [%d]: expected %.4f (idx=%d), got %.4f\n",
                   i, expected, idx_orig, output[i]);
            all_ok = 0;
        }
    }
    CHECK(all_ok, "3-bit pack/unpack: all 32 values roundtrip correctly");

    /* Test all possible 3-bit values in each position */
    int all_values_ok = 1;
    for (int val = 0; val < 8; val++) {
        for (int i = 0; i < 32; i++) indices[i] = (uint8_t)val;
        float input[32];
        for (int i = 0; i < 32; i++) input[i] = TQ3_LEVELS[val] * 2.0f; /* scale=2 */

        block_tq3_0 b2;
        quantize_row_tq3_0_ref(input, &b2, 32);
        float out2[32];
        dequantize_row_tq3_0(&b2, out2, 32);

        for (int i = 0; i < 32; i++) {
            if (fabsf(out2[i] - input[i]) > 0.01f) {
                printf("  Mismatch: val=%d pos=%d in=%.4f out=%.4f\n",
                       val, i, input[i], out2[i]);
                all_values_ok = 0;
            }
        }
    }
    CHECK(all_values_ok, "3-bit pack/unpack: all values 0-7 in all positions");
}

/* ── Test 2: TQ3_0 roundtrip quality ───────────────────────────────── */

static void test_tq3_roundtrip(void) {
    printf("\n=== Test 2: TQ3_0 quantize/dequantize quality ===\n");

    const int n = 1024;  /* 32 blocks */
    float * original = (float *)malloc(n * sizeof(float));
    float * restored = (float *)malloc(n * sizeof(float));
    block_tq3_0 * blocks = (block_tq3_0 *)malloc((n / QK_TQ3) * sizeof(block_tq3_0));

    /* Generate Gaussian data (simulating post-Hadamard KV cache values) */
    uint32_t rng = 12345;
    float sigma = 0.125f;  /* ~ 1/sqrt(64) for head_dim=64 */
    for (int i = 0; i < n; i++) {
        original[i] = rand_normal(&rng) * sigma;
    }

    quantize_row_tq3_0_ref(original, blocks, n);
    dequantize_row_tq3_0(blocks, restored, n);

    float mse = compute_mse(original, restored, n);
    float cosine = compute_cosine_sim(original, restored, n);
    float max_err = 0.0f;
    for (int i = 0; i < n; i++) {
        float err = fabsf(original[i] - restored[i]);
        if (err > max_err) max_err = err;
    }

    printf("  MSE: %.8f  Cosine: %.6f  MaxErr: %.6f\n", mse, cosine, max_err);
    CHECK(cosine > 0.95f, "TQ3_0 cosine similarity > 0.95 (got %.6f)", cosine);
    CHECK(mse < 0.001f,   "TQ3_0 MSE < 0.001 (got %.8f)", mse);

    /* Memory stats */
    size_t compressed = (n / QK_TQ3) * sizeof(block_tq3_0);
    size_t original_sz = n * sizeof(float);
    printf("  Memory: %zu bytes quantized vs %zu bytes FP32 (%.1fx compression)\n",
           compressed, original_sz, (float)original_sz / compressed);
    printf("  Bits per element: %.1f\n", (float)compressed * 8.0f / n);

    free(original); free(restored); free(blocks);
}

/* ── Test 3: TQ4_0 roundtrip quality ───────────────────────────────── */

static void test_tq4_roundtrip(void) {
    printf("\n=== Test 3: TQ4_0 quantize/dequantize quality ===\n");

    const int n = 1024;
    float * original = (float *)malloc(n * sizeof(float));
    float * restored = (float *)malloc(n * sizeof(float));
    block_tq4_0 * blocks = (block_tq4_0 *)malloc((n / QK_TQ4) * sizeof(block_tq4_0));

    uint32_t rng = 54321;
    float sigma = 0.125f;
    for (int i = 0; i < n; i++) {
        original[i] = rand_normal(&rng) * sigma;
    }

    quantize_row_tq4_0_ref(original, blocks, n);
    dequantize_row_tq4_0(blocks, restored, n);

    float mse = compute_mse(original, restored, n);
    float cosine = compute_cosine_sim(original, restored, n);

    printf("  MSE: %.8f  Cosine: %.6f\n", mse, cosine);
    CHECK(cosine > 0.98f, "TQ4_0 cosine similarity > 0.98 (got %.6f)", cosine);
    CHECK(mse < 0.0005f,  "TQ4_0 MSE < 0.0005 (got %.8f)", mse);

    size_t compressed = (n / QK_TQ4) * sizeof(block_tq4_0);
    printf("  Bits per element: %.1f\n", (float)compressed * 8.0f / n);

    free(original); free(restored); free(blocks);
}

/* ── Test 4: Hadamard orthogonality ────────────────────────────────── */

static void test_hadamard_orthogonality(void) {
    printf("\n=== Test 4: Hadamard transform orthogonality ===\n");

    for (int d = 32; d <= 128; d *= 2) {
        /* H^T H = I means applying H twice gives back the original (up to scale) */
        /* Actually: H^T = H for Walsh-Hadamard, so H * H = I (since H is symmetric)
         * and our normalized H satisfies H^T H = I */
        float * x = (float *)malloc(d * sizeof(float));
        float * y = (float *)malloc(d * sizeof(float));

        uint32_t rng = 42 + (uint32_t)d;
        for (int i = 0; i < d; i++) x[i] = rand_normal(&rng);
        memcpy(y, x, d * sizeof(float));

        /* Apply H twice — should get back original */
        hadamard_transform(y, d);
        hadamard_transform(y, d);

        float max_diff = 0.0f;
        for (int i = 0; i < d; i++) {
            float diff = fabsf(x[i] - y[i]);
            if (diff > max_diff) max_diff = diff;
        }

        CHECK(max_diff < 1e-5f, "Hadamard d=%d: H(H(x))=x, max_diff=%.2e", d, max_diff);

        /* Verify norm preservation: ||H x|| = ||x|| */
        float norm_orig = 0.0f, norm_rot = 0.0f;
        memcpy(y, x, d * sizeof(float));
        for (int i = 0; i < d; i++) norm_orig += x[i] * x[i];
        hadamard_transform(y, d);
        for (int i = 0; i < d; i++) norm_rot += y[i] * y[i];

        float norm_diff = fabsf(sqrtf(norm_orig) - sqrtf(norm_rot));
        CHECK(norm_diff < 1e-4f, "Hadamard d=%d: norm preserved, diff=%.2e", d, norm_diff);

        free(x); free(y);
    }
}

/* ── Test 5: Randomized Hadamard preserves norms ───────────────────── */

static void test_randomized_hadamard(void) {
    printf("\n=== Test 5: Randomized Hadamard norm preservation ===\n");

    const int d = 64;
    float * x = (float *)malloc(d * sizeof(float));
    float * y = (float *)malloc(d * sizeof(float));

    uint32_t rng = 99;
    for (int i = 0; i < d; i++) x[i] = rand_normal(&rng) * 0.5f;
    memcpy(y, x, d * sizeof(float));

    float norm_orig = 0.0f;
    for (int i = 0; i < d; i++) norm_orig += x[i] * x[i];

    randomized_hadamard_transform(y, d, 42);

    float norm_rot = 0.0f;
    for (int i = 0; i < d; i++) norm_rot += y[i] * y[i];

    float norm_diff = fabsf(sqrtf(norm_orig) - sqrtf(norm_rot));
    CHECK(norm_diff < 1e-4f, "RandHadamard d=%d: norm preserved, diff=%.2e", d, norm_diff);

    /* Check that values are more uniform after rotation */
    float var_orig = 0.0f, var_rot = 0.0f;
    float mean_orig = 0.0f, mean_rot = 0.0f;
    for (int i = 0; i < d; i++) { mean_orig += x[i]; mean_rot += y[i]; }
    mean_orig /= d; mean_rot /= d;
    for (int i = 0; i < d; i++) {
        var_orig += (x[i] - mean_orig) * (x[i] - mean_orig);
        var_rot  += (y[i] - mean_rot)  * (y[i] - mean_rot);
    }
    var_orig /= d; var_rot /= d;
    printf("  Variance: original=%.6f rotated=%.6f\n", var_orig, var_rot);

    free(x); free(y);
}

/* ── Test 6: Full pipeline (Hadamard → TQ3 → dequantize) ──────────── */

static void test_full_pipeline(void) {
    printf("\n=== Test 6: Full pipeline (Hadamard + TQ3_0) ===\n");

    const int head_dim = 64;
    const int n_heads = 4;
    const int n = head_dim * n_heads;  /* 256 elements = one token's KV */

    float * original = (float *)malloc(n * sizeof(float));
    float * rotated  = (float *)malloc(n * sizeof(float));
    float * restored = (float *)malloc(n * sizeof(float));

    /* Simulate real KV cache data: some dimensions are outliers */
    uint32_t rng = 777;
    for (int i = 0; i < n; i++) {
        original[i] = rand_normal(&rng) * 0.1f;
    }
    /* Inject outliers (the problem TurboQuant solves) */
    for (int h = 0; h < n_heads; h++) {
        int outlier_dim = (h * 17 + 3) % head_dim;  /* arbitrary outlier dimensions */
        original[h * head_dim + outlier_dim] = rand_normal(&rng) * 2.0f;
    }

    /* Step 1: Apply Hadamard rotation (distributes outlier energy) */
    memcpy(rotated, original, n * sizeof(float));
    hadamard_transform_row(rotated, n, head_dim);

    /* Step 2: Quantize rotated values with TQ3 */
    int nb = n / QK_TQ3;
    block_tq3_0 * blocks = (block_tq3_0 *)malloc(nb * sizeof(block_tq3_0));
    quantize_row_tq3_0_ref(rotated, blocks, n);

    /* Step 3: Dequantize */
    float * dequantized = (float *)malloc(n * sizeof(float));
    dequantize_row_tq3_0(blocks, dequantized, n);

    /* Step 4: Inverse Hadamard to get back to original space */
    memcpy(restored, dequantized, n * sizeof(float));
    hadamard_transform_row(restored, n, head_dim);  /* H^T = H for Walsh-Hadamard */

    float mse_with_rot = compute_mse(original, restored, n);
    float cos_with_rot = compute_cosine_sim(original, restored, n);

    /* Compare: quantize WITHOUT rotation */
    float * no_rot_restored = (float *)malloc(n * sizeof(float));
    block_tq3_0 * blocks_no_rot = (block_tq3_0 *)malloc(nb * sizeof(block_tq3_0));
    quantize_row_tq3_0_ref(original, blocks_no_rot, n);
    dequantize_row_tq3_0(blocks_no_rot, no_rot_restored, n);

    float mse_no_rot = compute_mse(original, no_rot_restored, n);
    float cos_no_rot = compute_cosine_sim(original, no_rot_restored, n);

    printf("  With Hadamard:    MSE=%.8f  Cosine=%.6f\n", mse_with_rot, cos_with_rot);
    printf("  Without Hadamard: MSE=%.8f  Cosine=%.6f\n", mse_no_rot, cos_no_rot);
    printf("  Improvement: %.1fx lower MSE with rotation\n",
           mse_no_rot > 0 ? mse_no_rot / mse_with_rot : 0.0f);

    CHECK(mse_with_rot < mse_no_rot,
          "Hadamard rotation reduces MSE (%.2e < %.2e)",
          mse_with_rot, mse_no_rot);
    CHECK(cos_with_rot > 0.95f,
          "Full pipeline cosine > 0.95 (got %.6f)", cos_with_rot);

    free(original); free(rotated); free(restored); free(dequantized);
    free(no_rot_restored); free(blocks); free(blocks_no_rot);
}

/* ── Test 7: TQ3 vs uniform Q3 MSE ────────────────────────────────── */

static void test_tq3_vs_uniform(void) {
    printf("\n=== Test 7: TQ3 (Lloyd-Max) vs uniform 3-bit quantization ===\n");

    const int n = 4096;
    float * data = (float *)malloc(n * sizeof(float));
    float * tq3_out = (float *)malloc(n * sizeof(float));
    float * uni_out = (float *)malloc(n * sizeof(float));

    /* Gaussian data */
    uint32_t rng = 333;
    for (int i = 0; i < n; i++) {
        data[i] = rand_normal(&rng) * 0.125f;
    }

    /* TQ3 quantize */
    int nb = n / QK_TQ3;
    block_tq3_0 * tq3_blocks = (block_tq3_0 *)malloc(nb * sizeof(block_tq3_0));
    quantize_row_tq3_0_ref(data, tq3_blocks, n);
    dequantize_row_tq3_0(tq3_blocks, tq3_out, n);

    /* Uniform 3-bit quantize (for comparison) */
    for (int b = 0; b < nb; b++) {
        float amax = 0.0f;
        for (int j = 0; j < 32; j++) {
            float v = fabsf(data[b * 32 + j]);
            if (v > amax) amax = v;
        }
        float d = amax / 3.5f;  /* 8 levels: [-3.5, -2.5, ..., +3.5] * d */
        float id = d > 0 ? 1.0f / d : 0.0f;
        for (int j = 0; j < 32; j++) {
            float xn = data[b * 32 + j] * id;
            /* Uniform 8 levels: round to nearest of [-3.5, -2.5, ..., +3.5] */
            int level = (int)roundf(xn + 3.5f);
            if (level < 0) level = 0;
            if (level > 7) level = 7;
            uni_out[b * 32 + j] = ((float)level - 3.5f) * d;
        }
    }

    float mse_tq3 = compute_mse(data, tq3_out, n);
    float mse_uni = compute_mse(data, uni_out, n);
    float cos_tq3 = compute_cosine_sim(data, tq3_out, n);
    float cos_uni = compute_cosine_sim(data, uni_out, n);

    printf("  TQ3 (Lloyd-Max):  MSE=%.8f  Cosine=%.6f\n", mse_tq3, cos_tq3);
    printf("  Uniform 3-bit:    MSE=%.8f  Cosine=%.6f\n", mse_uni, cos_uni);
    printf("  TQ3 advantage: %.1f%% lower MSE\n",
           mse_uni > 0 ? (1.0f - mse_tq3 / mse_uni) * 100.0f : 0.0f);

    CHECK(mse_tq3 < mse_uni,
          "Lloyd-Max beats uniform quantization (%.2e < %.2e)",
          mse_tq3, mse_uni);

    free(data); free(tq3_out); free(uni_out); free(tq3_blocks);
}

/* ── Test 8: Codebook symmetry ─────────────────────────────────────── */

static void test_codebook_symmetry(void) {
    printf("\n=== Test 8: Codebook symmetry validation ===\n");

    /* 3-bit levels should be symmetric: level[i] = -level[7-i] */
    int sym_ok = 1;
    for (int i = 0; i < 4; i++) {
        if (fabsf(TQ3_LEVELS[i] + TQ3_LEVELS[7 - i]) > 1e-6f) {
            printf("  TQ3 asymmetry at %d: %.7f vs %.7f\n",
                   i, TQ3_LEVELS[i], TQ3_LEVELS[7 - i]);
            sym_ok = 0;
        }
    }
    CHECK(sym_ok, "TQ3 codebook is symmetric around zero");

    /* 4-bit levels should be symmetric: level[i] = -level[15-i] */
    sym_ok = 1;
    for (int i = 0; i < 8; i++) {
        if (fabsf(TQ4_LEVELS[i] + TQ4_LEVELS[15 - i]) > 1e-6f) {
            printf("  TQ4 asymmetry at %d: %.7f vs %.7f\n",
                   i, TQ4_LEVELS[i], TQ4_LEVELS[15 - i]);
            sym_ok = 0;
        }
    }
    CHECK(sym_ok, "TQ4 codebook is symmetric around zero");

    /* Boundaries: midpoints of consecutive levels */
    int bounds_ok = 1;
    for (int i = 0; i < 7; i++) {
        float expected = (TQ3_LEVELS[i] + TQ3_LEVELS[i + 1]) / 2.0f;
        if (fabsf(TQ3_BOUNDARIES[i] - expected) > 1e-5f) {
            printf("  TQ3 boundary[%d]: expected %.7f, got %.7f\n",
                   i, expected, TQ3_BOUNDARIES[i]);
            bounds_ok = 0;
        }
    }
    CHECK(bounds_ok, "TQ3 boundaries are midpoints of levels");

    /* Outer levels should be ±1.0 (normalized codebook) */
    CHECK(fabsf(TQ3_LEVELS[0] - (-1.0f)) < 1e-6f &&
          fabsf(TQ3_LEVELS[7] - ( 1.0f)) < 1e-6f,
          "TQ3 outer levels = +/-1.0");
    CHECK(fabsf(TQ4_LEVELS[0] - (-1.0f)) < 1e-6f &&
          fabsf(TQ4_LEVELS[15] - ( 1.0f)) < 1e-6f,
          "TQ4 outer levels = +/-1.0");
}

/* ── Test 9: Attention score preservation (the metric that matters) ── */

static void test_attention_preservation(void) {
    printf("\n=== Test 9: Attention score preservation ===\n");

    const int head_dim = 64;
    const int seq_len  = 128;  /* 128 tokens in KV cache */

    float * K = (float *)malloc(seq_len * head_dim * sizeof(float));
    float * Q = (float *)malloc(head_dim * sizeof(float));

    /* Generate random K cache and Q vector */
    uint32_t rng = 555;
    for (int i = 0; i < seq_len * head_dim; i++)
        K[i] = rand_normal(&rng) * 0.1f;
    for (int i = 0; i < head_dim; i++)
        Q[i] = rand_normal(&rng) * 0.1f;

    /* Inject outliers in K */
    for (int t = 0; t < seq_len; t++) {
        int outlier = (t * 13 + 7) % head_dim;
        K[t * head_dim + outlier] *= 20.0f;
    }

    /* Compute original attention scores: score[t] = Q . K[t] */
    float * scores_orig = (float *)malloc(seq_len * sizeof(float));
    for (int t = 0; t < seq_len; t++) {
        float dot = 0.0f;
        for (int d = 0; d < head_dim; d++)
            dot += Q[d] * K[t * head_dim + d];
        scores_orig[t] = dot;
    }

    /* Pipeline: rotate K and Q, quantize K, compute scores in rotated space */
    float * K_rot = (float *)malloc(seq_len * head_dim * sizeof(float));
    float * Q_rot = (float *)malloc(head_dim * sizeof(float));
    memcpy(K_rot, K, seq_len * head_dim * sizeof(float));
    memcpy(Q_rot, Q, head_dim * sizeof(float));

    /* Rotate all K vectors and Q */
    for (int t = 0; t < seq_len; t++)
        hadamard_transform(K_rot + t * head_dim, head_dim);
    hadamard_transform(Q_rot, head_dim);

    /* Quantize rotated K with TQ3 */
    int n_elems = seq_len * head_dim;
    int nb = n_elems / QK_TQ3;
    block_tq3_0 * K_quant = (block_tq3_0 *)malloc(nb * sizeof(block_tq3_0));
    quantize_row_tq3_0_ref(K_rot, K_quant, n_elems);

    float * K_dequant = (float *)malloc(n_elems * sizeof(float));
    dequantize_row_tq3_0(K_quant, K_dequant, n_elems);

    /* Compute attention in rotated space: Q_rot . K_dequant[t] */
    float * scores_tq3 = (float *)malloc(seq_len * sizeof(float));
    for (int t = 0; t < seq_len; t++) {
        float dot = 0.0f;
        for (int d = 0; d < head_dim; d++)
            dot += Q_rot[d] * K_dequant[t * head_dim + d];
        scores_tq3[t] = dot;
    }

    float score_cosine = compute_cosine_sim(scores_orig, scores_tq3, seq_len);
    float score_mse = compute_mse(scores_orig, scores_tq3, seq_len);

    printf("  Attention score cosine: %.6f\n", score_cosine);
    printf("  Attention score MSE:    %.8f\n", score_mse);
    CHECK(score_cosine > 0.99f,
          "Attention cosine > 0.99 (got %.6f)", score_cosine);

    /* Also test without rotation for comparison */
    int nb2 = n_elems / QK_TQ3;
    block_tq3_0 * K_quant_norot = (block_tq3_0 *)malloc(nb2 * sizeof(block_tq3_0));
    float * K_dequant_norot = (float *)malloc(n_elems * sizeof(float));
    quantize_row_tq3_0_ref(K, K_quant_norot, n_elems);
    dequantize_row_tq3_0(K_quant_norot, K_dequant_norot, n_elems);

    float * scores_norot = (float *)malloc(seq_len * sizeof(float));
    for (int t = 0; t < seq_len; t++) {
        float dot = 0.0f;
        for (int d = 0; d < head_dim; d++)
            dot += Q[d] * K_dequant_norot[t * head_dim + d];
        scores_norot[t] = dot;
    }

    float norot_cosine = compute_cosine_sim(scores_orig, scores_norot, seq_len);
    printf("  Without rotation:       %.6f\n", norot_cosine);
    printf("  Rotation improvement:   +%.4f\n", score_cosine - norot_cosine);

    CHECK(score_cosine > norot_cosine,
          "Rotation improves attention preservation (%.4f > %.4f)",
          score_cosine, norot_cosine);

    free(K); free(Q); free(K_rot); free(Q_rot);
    free(K_quant); free(K_dequant); free(scores_orig); free(scores_tq3);
    free(K_quant_norot); free(K_dequant_norot); free(scores_norot);
}

/* ── Main ──────────────────────────────────────────────────────────── */

int main(void) {
    printf("TurboQuant TQ3_0 / TQ4_0 — Correctness Tests\n");
    printf("=============================================\n");

    test_codebook_symmetry();
    test_3bit_packing();
    test_tq3_roundtrip();
    test_tq4_roundtrip();
    test_hadamard_orthogonality();
    test_randomized_hadamard();
    test_full_pipeline();
    test_tq3_vs_uniform();
    test_attention_preservation();

    printf("\n=============================================\n");
    printf("Results: %d passed, %d failed\n", g_pass, g_fail);
    return g_fail > 0 ? 1 : 0;
}
