/**
 * test-tq.c — Correctness tests for TurboQuant (TQ2_0 / TQ3_0 / TQ4_0)
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
 *   9. Attention score preservation
 *  10. 2-bit pack/unpack roundtrip
 *  11. TQ2_0 quantize/dequantize roundtrip + MSE
 *  12. TQ2 codebook symmetry
 *  13. Full pipeline: Hadamard + TQ2
 *  14. TQ2 attention score preservation
 *  15. TQ2 vs uniform 2-bit
 *  16. Requantize TQ4→TQ3→TQ2 chain
 *  17. TQ2 memory layout
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

/* ── Test 10: 2-bit pack/unpack roundtrip ──────────────────────────── */

static void test_2bit_packing(void) {
    printf("\n=== Test 10: 2-bit pack/unpack roundtrip ===\n");

    /* Test all 4 values (0-3) in all 32 positions via quantize/dequantize */
    int all_ok = 1;
    for (int val = 0; val < 4; val++) {
        float input[32];
        for (int i = 0; i < 32; i++) input[i] = TQ2_LEVELS[val] * 2.0f; /* scale=2 */

        block_tq2_0 blk;
        quantize_row_tq2_0_ref(input, &blk, 32);
        float output[32];
        dequantize_row_tq2_0(&blk, output, 32);

        for (int i = 0; i < 32; i++) {
            if (fabsf(output[i] - input[i]) > 0.05f) {
                printf("  Mismatch: val=%d pos=%d in=%.4f out=%.4f\n",
                       val, i, input[i], output[i]);
                all_ok = 0;
            }
        }
    }
    CHECK(all_ok, "2-bit pack/unpack: all values 0-3 in all positions");

    /* Test mixed values: sequential pattern 0,1,2,3,0,1,2,3... */
    float mixed_in[32];
    for (int i = 0; i < 32; i++) mixed_in[i] = TQ2_LEVELS[i % 4] * 3.0f;
    block_tq2_0 blk2;
    quantize_row_tq2_0_ref(mixed_in, &blk2, 32);
    float mixed_out[32];
    dequantize_row_tq2_0(&blk2, mixed_out, 32);

    int mixed_ok = 1;
    for (int i = 0; i < 32; i++) {
        if (fabsf(mixed_out[i] - mixed_in[i]) > 0.1f) {
            printf("  Mixed mismatch at [%d]: in=%.4f out=%.4f\n",
                   i, mixed_in[i], mixed_out[i]);
            mixed_ok = 0;
        }
    }
    CHECK(mixed_ok, "2-bit pack/unpack: mixed pattern roundtrips correctly");

    /* Verify block size: 10 bytes = 2 (fp16 d) + 8 (packed qs) */
    CHECK(sizeof(block_tq2_0) == 10,
          "block_tq2_0 size = 10 bytes (got %zu)", sizeof(block_tq2_0));
}

/* ── Test 11: TQ2_0 roundtrip quality ─────────────────────────────── */

static void test_tq2_roundtrip(void) {
    printf("\n=== Test 11: TQ2_0 quantize/dequantize quality ===\n");

    const int n = 1024;  /* 32 blocks */
    float * original = (float *)malloc(n * sizeof(float));
    float * restored = (float *)malloc(n * sizeof(float));
    block_tq2_0 * blocks = (block_tq2_0 *)malloc((n / QK_TQ2) * sizeof(block_tq2_0));

    /* Generate Gaussian data (simulating post-Hadamard KV cache values) */
    uint32_t rng = 22222;
    float sigma = 0.125f;
    for (int i = 0; i < n; i++) {
        original[i] = rand_normal(&rng) * sigma;
    }

    quantize_row_tq2_0_ref(original, blocks, n);
    dequantize_row_tq2_0(blocks, restored, n);

    float mse = compute_mse(original, restored, n);
    float cosine = compute_cosine_sim(original, restored, n);
    float max_err = 0.0f;
    for (int i = 0; i < n; i++) {
        float err = fabsf(original[i] - restored[i]);
        if (err > max_err) max_err = err;
    }

    printf("  MSE: %.8f  Cosine: %.6f  MaxErr: %.6f\n", mse, cosine, max_err);
    CHECK(cosine > 0.85f, "TQ2_0 cosine similarity > 0.85 (got %.6f)", cosine);
    CHECK(mse < 0.003f,   "TQ2_0 MSE < 0.003 (got %.8f)", mse);

    /* Memory stats */
    size_t compressed = (n / QK_TQ2) * sizeof(block_tq2_0);
    size_t original_sz = n * sizeof(float);
    printf("  Memory: %zu bytes quantized vs %zu bytes FP32 (%.1fx compression)\n",
           compressed, original_sz, (float)original_sz / compressed);
    printf("  Bits per element: %.1f\n", (float)compressed * 8.0f / n);

    free(original); free(restored); free(blocks);
}

/* ── Test 12: TQ2 codebook symmetry ───────────────────────────────── */

static void test_tq2_codebook_symmetry(void) {
    printf("\n=== Test 12: TQ2 codebook symmetry ===\n");

    /* TQ2: 4 levels, level[i] = -level[3-i] */
    int sym_ok = 1;
    for (int i = 0; i < 2; i++) {
        if (fabsf(TQ2_LEVELS[i] + TQ2_LEVELS[3 - i]) > 1e-6f) {
            printf("  TQ2 asymmetry at %d: %.7f vs %.7f\n",
                   i, TQ2_LEVELS[i], TQ2_LEVELS[3 - i]);
            sym_ok = 0;
        }
    }
    CHECK(sym_ok, "TQ2 codebook is symmetric around zero");

    /* Boundaries should be midpoints */
    int bounds_ok = 1;
    for (int i = 0; i < 3; i++) {
        float expected = (TQ2_LEVELS[i] + TQ2_LEVELS[i + 1]) / 2.0f;
        if (fabsf(TQ2_BOUNDARIES[i] - expected) > 1e-5f) {
            printf("  TQ2 boundary[%d]: expected %.7f, got %.7f\n",
                   i, expected, TQ2_BOUNDARIES[i]);
            bounds_ok = 0;
        }
    }
    CHECK(bounds_ok, "TQ2 boundaries are midpoints of levels");

    /* Outer levels = ±1.0 */
    CHECK(fabsf(TQ2_LEVELS[0] - (-1.0f)) < 1e-6f &&
          fabsf(TQ2_LEVELS[3] - ( 1.0f)) < 1e-6f,
          "TQ2 outer levels = +/-1.0");

    /* Middle boundary should be exactly 0.0 (symmetric codebook) */
    CHECK(fabsf(TQ2_BOUNDARIES[1]) < 1e-6f,
          "TQ2 middle boundary = 0.0 (got %.7f)", TQ2_BOUNDARIES[1]);
}

/* ── Test 13: Full pipeline Hadamard + TQ2 ────────────────────────── */

static void test_full_pipeline_tq2(void) {
    printf("\n=== Test 13: Full pipeline (Hadamard + TQ2_0) ===\n");

    const int head_dim = 64;
    const int n_heads = 4;
    const int n = head_dim * n_heads;

    float * original = (float *)malloc(n * sizeof(float));
    float * rotated  = (float *)malloc(n * sizeof(float));
    float * restored = (float *)malloc(n * sizeof(float));

    /* Simulate KV cache with outliers */
    uint32_t rng = 888;
    for (int i = 0; i < n; i++) {
        original[i] = rand_normal(&rng) * 0.1f;
    }
    for (int h = 0; h < n_heads; h++) {
        int outlier_dim = (h * 17 + 3) % head_dim;
        original[h * head_dim + outlier_dim] = rand_normal(&rng) * 2.0f;
    }

    /* Hadamard → TQ2 → dequant → inverse Hadamard */
    memcpy(rotated, original, n * sizeof(float));
    hadamard_transform_row(rotated, n, head_dim);

    int nb = n / QK_TQ2;
    block_tq2_0 * blocks = (block_tq2_0 *)malloc(nb * sizeof(block_tq2_0));
    quantize_row_tq2_0_ref(rotated, blocks, n);

    float * dequantized = (float *)malloc(n * sizeof(float));
    dequantize_row_tq2_0(blocks, dequantized, n);

    memcpy(restored, dequantized, n * sizeof(float));
    hadamard_transform_row(restored, n, head_dim);

    float mse_with_rot = compute_mse(original, restored, n);
    float cos_with_rot = compute_cosine_sim(original, restored, n);

    /* Compare without rotation */
    float * no_rot_restored = (float *)malloc(n * sizeof(float));
    block_tq2_0 * blocks_no_rot = (block_tq2_0 *)malloc(nb * sizeof(block_tq2_0));
    quantize_row_tq2_0_ref(original, blocks_no_rot, n);
    dequantize_row_tq2_0(blocks_no_rot, no_rot_restored, n);

    float mse_no_rot = compute_mse(original, no_rot_restored, n);
    float cos_no_rot = compute_cosine_sim(original, no_rot_restored, n);

    printf("  With Hadamard:    MSE=%.8f  Cosine=%.6f\n", mse_with_rot, cos_with_rot);
    printf("  Without Hadamard: MSE=%.8f  Cosine=%.6f\n", mse_no_rot, cos_no_rot);

    CHECK(mse_with_rot < mse_no_rot,
          "Hadamard reduces TQ2 MSE (%.2e < %.2e)", mse_with_rot, mse_no_rot);
    CHECK(cos_with_rot > 0.90f,
          "Full pipeline TQ2 cosine > 0.90 (got %.6f)", cos_with_rot);

    free(original); free(rotated); free(restored); free(dequantized);
    free(no_rot_restored); free(blocks); free(blocks_no_rot);
}

/* ── Test 14: TQ2 attention score preservation ────────────────────── */

static void test_tq2_attention_preservation(void) {
    printf("\n=== Test 14: TQ2 attention score preservation ===\n");

    const int head_dim = 64;
    const int seq_len  = 128;

    float * K = (float *)malloc(seq_len * head_dim * sizeof(float));
    float * Q = (float *)malloc(head_dim * sizeof(float));

    uint32_t rng = 666;
    for (int i = 0; i < seq_len * head_dim; i++)
        K[i] = rand_normal(&rng) * 0.1f;
    for (int i = 0; i < head_dim; i++)
        Q[i] = rand_normal(&rng) * 0.1f;

    /* Inject outliers */
    for (int t = 0; t < seq_len; t++) {
        int outlier = (t * 13 + 7) % head_dim;
        K[t * head_dim + outlier] *= 20.0f;
    }

    /* Original attention scores */
    float * scores_orig = (float *)malloc(seq_len * sizeof(float));
    for (int t = 0; t < seq_len; t++) {
        float dot = 0.0f;
        for (int d = 0; d < head_dim; d++)
            dot += Q[d] * K[t * head_dim + d];
        scores_orig[t] = dot;
    }

    /* Rotate K and Q, quantize K with TQ2 */
    float * K_rot = (float *)malloc(seq_len * head_dim * sizeof(float));
    float * Q_rot = (float *)malloc(head_dim * sizeof(float));
    memcpy(K_rot, K, seq_len * head_dim * sizeof(float));
    memcpy(Q_rot, Q, head_dim * sizeof(float));

    for (int t = 0; t < seq_len; t++)
        hadamard_transform(K_rot + t * head_dim, head_dim);
    hadamard_transform(Q_rot, head_dim);

    int n_elems = seq_len * head_dim;
    int nb = n_elems / QK_TQ2;
    block_tq2_0 * K_quant = (block_tq2_0 *)malloc(nb * sizeof(block_tq2_0));
    quantize_row_tq2_0_ref(K_rot, K_quant, n_elems);

    float * K_dequant = (float *)malloc(n_elems * sizeof(float));
    dequantize_row_tq2_0(K_quant, K_dequant, n_elems);

    float * scores_tq2 = (float *)malloc(seq_len * sizeof(float));
    for (int t = 0; t < seq_len; t++) {
        float dot = 0.0f;
        for (int d = 0; d < head_dim; d++)
            dot += Q_rot[d] * K_dequant[t * head_dim + d];
        scores_tq2[t] = dot;
    }

    float score_cosine = compute_cosine_sim(scores_orig, scores_tq2, seq_len);
    float score_mse = compute_mse(scores_orig, scores_tq2, seq_len);

    printf("  Attention score cosine: %.6f\n", score_cosine);
    printf("  Attention score MSE:    %.8f\n", score_mse);
    CHECK(score_cosine > 0.92f,
          "TQ2 attention cosine > 0.92 (got %.6f)", score_cosine);

    free(K); free(Q); free(K_rot); free(Q_rot);
    free(K_quant); free(K_dequant); free(scores_orig); free(scores_tq2);
}

/* ── Test 15: TQ2 vs uniform 2-bit ───────────────────────────────── */

static void test_tq2_vs_uniform(void) {
    printf("\n=== Test 15: TQ2 (Lloyd-Max) vs uniform 2-bit quantization ===\n");

    const int n = 4096;
    float * data = (float *)malloc(n * sizeof(float));
    float * tq2_out = (float *)malloc(n * sizeof(float));
    float * uni_out = (float *)malloc(n * sizeof(float));

    /* Gaussian data */
    uint32_t rng = 444;
    for (int i = 0; i < n; i++) {
        data[i] = rand_normal(&rng) * 0.125f;
    }

    /* TQ2 quantize */
    int nb = n / QK_TQ2;
    block_tq2_0 * tq2_blocks = (block_tq2_0 *)malloc(nb * sizeof(block_tq2_0));
    quantize_row_tq2_0_ref(data, tq2_blocks, n);
    dequantize_row_tq2_0(tq2_blocks, tq2_out, n);

    /* Uniform 2-bit quantize (4 levels: [-1.5, -0.5, +0.5, +1.5] * d) */
    for (int b = 0; b < nb; b++) {
        float amax = 0.0f;
        for (int j = 0; j < 32; j++) {
            float v = fabsf(data[b * 32 + j]);
            if (v > amax) amax = v;
        }
        float d = amax / 1.5f;  /* 4 levels: [-1.5, -0.5, +0.5, +1.5] * d */
        float id = d > 0 ? 1.0f / d : 0.0f;
        for (int j = 0; j < 32; j++) {
            float xn = data[b * 32 + j] * id;
            /* Round to nearest of [-1.5, -0.5, +0.5, +1.5] */
            int level = (int)roundf(xn + 1.5f);
            if (level < 0) level = 0;
            if (level > 3) level = 3;
            uni_out[b * 32 + j] = ((float)level - 1.5f) * d;
        }
    }

    float mse_tq2 = compute_mse(data, tq2_out, n);
    float mse_uni = compute_mse(data, uni_out, n);
    float cos_tq2 = compute_cosine_sim(data, tq2_out, n);
    float cos_uni = compute_cosine_sim(data, uni_out, n);

    printf("  TQ2 (Lloyd-Max):  MSE=%.8f  Cosine=%.6f\n", mse_tq2, cos_tq2);
    printf("  Uniform 2-bit:    MSE=%.8f  Cosine=%.6f\n", mse_uni, cos_uni);
    printf("  TQ2 advantage: %.1f%% lower MSE\n",
           mse_uni > 0 ? (1.0f - mse_tq2 / mse_uni) * 100.0f : 0.0f);

    CHECK(mse_tq2 < mse_uni,
          "Lloyd-Max TQ2 beats uniform 2-bit (%.2e < %.2e)",
          mse_tq2, mse_uni);

    free(data); free(tq2_out); free(uni_out); free(tq2_blocks);
}

/* ── Test 16: Requantize TQ4→TQ3→TQ2 chain ───────────────────────── */

static void test_requantize_chain(void) {
    printf("\n=== Test 16: Requantize TQ4→TQ3→TQ2 chain ===\n");

    const int n = 1024;
    float * original = (float *)malloc(n * sizeof(float));

    uint32_t rng = 77777;
    float sigma = 0.125f;
    for (int i = 0; i < n; i++) {
        original[i] = rand_normal(&rng) * sigma;
    }

    int nb = n / 32;

    /* Quantize original to TQ4 */
    block_tq4_0 * tq4 = (block_tq4_0 *)malloc(nb * sizeof(block_tq4_0));
    quantize_row_tq4_0_ref(original, tq4, n);

    /* Requantize TQ4 → TQ3 */
    block_tq3_0 * tq3 = (block_tq3_0 *)malloc(nb * sizeof(block_tq3_0));
    requantize_tq4_to_tq3(tq4, tq3, n);

    /* Requantize TQ3 → TQ2 */
    block_tq2_0 * tq2_chain = (block_tq2_0 *)malloc(nb * sizeof(block_tq2_0));
    requantize_tq3_to_tq2(tq3, tq2_chain, n);

    /* Direct TQ4 → TQ2 */
    block_tq2_0 * tq2_direct = (block_tq2_0 *)malloc(nb * sizeof(block_tq2_0));
    requantize_tq4_to_tq2(tq4, tq2_direct, n);

    /* Dequantize all for comparison */
    float * from_tq4 = (float *)malloc(n * sizeof(float));
    float * from_tq3 = (float *)malloc(n * sizeof(float));
    float * from_chain = (float *)malloc(n * sizeof(float));
    float * from_direct = (float *)malloc(n * sizeof(float));

    dequantize_row_tq4_0(tq4, from_tq4, n);
    dequantize_row_tq3_0(tq3, from_tq3, n);
    dequantize_row_tq2_0(tq2_chain, from_chain, n);
    dequantize_row_tq2_0(tq2_direct, from_direct, n);

    float cos_tq4 = compute_cosine_sim(original, from_tq4, n);
    float cos_tq3 = compute_cosine_sim(original, from_tq3, n);
    float cos_chain = compute_cosine_sim(original, from_chain, n);
    float cos_direct = compute_cosine_sim(original, from_direct, n);

    printf("  TQ4 cosine:           %.6f\n", cos_tq4);
    printf("  TQ4→TQ3 cosine:      %.6f\n", cos_tq3);
    printf("  TQ4→TQ3→TQ2 cosine:  %.6f\n", cos_chain);
    printf("  TQ4→TQ2 direct:      %.6f\n", cos_direct);

    /* Quality should degrade gracefully: TQ4 > TQ3 > TQ2 */
    CHECK(cos_tq4 > cos_tq3,
          "Quality ordering: TQ4 > TQ3 (%.4f > %.4f)", cos_tq4, cos_tq3);
    CHECK(cos_tq3 > cos_chain,
          "Quality ordering: TQ3 > TQ2 (%.4f > %.4f)", cos_tq3, cos_chain);

    /* Chain and direct should produce similar quality */
    float chain_vs_direct_diff = fabsf(cos_chain - cos_direct);
    printf("  Chain vs direct diff: %.6f\n", chain_vs_direct_diff);
    CHECK(chain_vs_direct_diff < 0.05f,
          "Chain vs direct TQ2 similar quality (diff=%.4f)", chain_vs_direct_diff);

    /* TQ2 (even via chain) should still be usable */
    CHECK(cos_chain > 0.80f,
          "TQ4→TQ3→TQ2 chain still usable (cosine=%.4f > 0.80)", cos_chain);

    free(original); free(tq4); free(tq3); free(tq2_chain); free(tq2_direct);
    free(from_tq4); free(from_tq3); free(from_chain); free(from_direct);
}

/* ── Test 17: TQ2 memory layout ───────────────────────────────────── */

static void test_tq2_memory_layout(void) {
    printf("\n=== Test 17: TQ2 memory layout verification ===\n");

    /* block_tq2_0: 2 bytes (d) + 8 bytes (qs) = 10 bytes per 32 elements */
    CHECK(sizeof(block_tq2_0) == 10,
          "block_tq2_0 = 10 bytes (got %zu)", sizeof(block_tq2_0));
    CHECK(sizeof(block_tq3_0) == 14,
          "block_tq3_0 = 14 bytes (got %zu)", sizeof(block_tq3_0));
    CHECK(sizeof(block_tq4_0) == 18,
          "block_tq4_0 = 18 bytes (got %zu)", sizeof(block_tq4_0));

    /* Bits per element */
    float bpe_tq2 = (float)sizeof(block_tq2_0) * 8.0f / QK_TQ2;
    float bpe_tq3 = (float)sizeof(block_tq3_0) * 8.0f / QK_TQ3;
    float bpe_tq4 = (float)sizeof(block_tq4_0) * 8.0f / QK_TQ4;

    CHECK(fabsf(bpe_tq2 - 2.5f) < 0.01f,
          "TQ2_0 bits/elem = 2.5 (got %.1f)", bpe_tq2);
    CHECK(fabsf(bpe_tq3 - 3.5f) < 0.01f,
          "TQ3_0 bits/elem = 3.5 (got %.1f)", bpe_tq3);
    CHECK(fabsf(bpe_tq4 - 4.5f) < 0.01f,
          "TQ4_0 bits/elem = 4.5 (got %.1f)", bpe_tq4);

    /* Compression ratios vs FP16 (16 bits/elem) */
    printf("  TQ2: %.1fx compression vs FP16\n", 16.0f / bpe_tq2);
    printf("  TQ3: %.1fx compression vs FP16\n", 16.0f / bpe_tq3);
    printf("  TQ4: %.1fx compression vs FP16\n", 16.0f / bpe_tq4);

    /* Verify QK defines */
    CHECK(QK_TQ2 == 32, "QK_TQ2 = 32 (got %d)", QK_TQ2);
    CHECK(QK_TQ3 == 32, "QK_TQ3 = 32 (got %d)", QK_TQ3);
    CHECK(QK_TQ4 == 32, "QK_TQ4 = 32 (got %d)", QK_TQ4);
}

/* ── Main ──────────────────────────────────────────────────────────── */

int main(void) {
    printf("TurboQuant TQ2_0 / TQ3_0 / TQ4_0 — Correctness Tests\n");
    printf("=====================================================\n");

    /* TQ3/TQ4 tests (1-9) */
    test_codebook_symmetry();
    test_3bit_packing();
    test_tq3_roundtrip();
    test_tq4_roundtrip();
    test_hadamard_orthogonality();
    test_randomized_hadamard();
    test_full_pipeline();
    test_tq3_vs_uniform();
    test_attention_preservation();

    /* TQ2 tests (10-17) */
    test_2bit_packing();
    test_tq2_roundtrip();
    test_tq2_codebook_symmetry();
    test_full_pipeline_tq2();
    test_tq2_attention_preservation();
    test_tq2_vs_uniform();
    test_requantize_chain();
    test_tq2_memory_layout();

    printf("\n=====================================================\n");
    printf("Results: %d passed, %d failed\n", g_pass, g_fail);
    return g_fail > 0 ? 1 : 0;
}
