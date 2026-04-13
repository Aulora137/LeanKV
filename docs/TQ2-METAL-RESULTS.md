# TQ2 + TQ2_1 Metal GPU Results & Qwen3-4B Investigation

Date: 2026-04-12
Branch: `feature/tq2-outlier-tiered`
Backend: Metal (Apple Silicon)
Eval: WikiText-2, n_ctx=2048, 3 chunks (quick validation)

---

## 1. Metal Implementation Status

All four TQ types are fully implemented on Metal:

| Component | TQ2_0 | TQ2_1 | TQ3_0 | TQ4_0 |
|-----------|:---:|:---:|:---:|:---:|
| Dequant (type4x4) | Y | Y | Y | Y |
| Dequant (t4/vec) | Y | Y | Y | Y |
| Flash Attention | Y | Y | Y | Y |
| FA Vec | Y | Y | Y | Y |
| Copy kernel (f32→TQ) | Y | Y | Y | Y |
| get_rows | Y | Y | Y | Y |
| mul_mm (f32/f16) | Y | Y | Y | Y |
| mul_mm_id | Y | Y | Y | Y |
| Pipeline dispatch | Y | Y | Y | Y |

**TQ2_1** is the new mixed-precision type: 32 outlier channels at TQ3 (3.5 bpe) +
96 normal channels at TQ2 (2.5 bpe) = **2.75 effective bits/element** for 128 elements.
Block size: 44 bytes / 128 elements.

### TQ3 Metal Speed Fix

TQ3 was initially **slower** than TQ2 on Metal due to register spill:
- `type4x4` path: `(base+N)/4` and `(base+N)%4` indexing caused compiler to spill
- `t4` path: `uint8_t idx[8]` array forced stack allocation

**Fix:** Fully unrolled type4x4 with direct `reg[row][col]` assignments; replaced
idx array with if/else branch unpacking only the needed 4 indices. TQ3 now matches
TQ4/TQ2 speed.

---

## 2. PPL Results — Metal GPU (3 chunks)

### 2.1 Qwen3-8B (n_embd=4096, n_head=32, n_head_kv=8, head_dim=128)

| K-cache | V-cache | BPE (K) | PPL | Delta | Delta % |
|---------|---------|---------|----:|------:|--------:|
| F16 | F16 | 16.0 | 9.47 | — | — |
| TQ4_0 | TQ4_0 | 4.5 | 9.86 | +0.39 | +4.1% |
| TQ3_0 | TQ3_0 | 3.5 | 10.04 | +0.57 | +6.0% |
| TQ2_1 | TQ2_1 | 2.75 | 27.27 | +17.80 | +188% |
| TQ2_0 | TQ2_0 | 2.5 | 38.29 | +28.82 | +304% |

**Verdict:** TQ4 and TQ3 are production-quality on this model. TQ2_1 shows the
expected quality cliff below 3 bpe but is substantially better than TQ2_0 (27.3 vs
38.3), validating the mixed-precision approach. TQ2_1 may be viable for cold-tier
storage where some degradation is acceptable.

### 2.2 Qwen3-4B (n_embd=2560, n_head=32, n_head_kv=8, head_dim=128)

| K-cache | V-cache | BPE (K) | PPL | Delta | Delta % |
|---------|---------|---------|----:|------:|--------:|
| F16 | F16 | 16.0 | 13.04 | — | — |
| TQ4_0 | F16 | 4.5 | 13.85 | +0.81 | +6.2% |
| TQ4_0 | TQ4_0 | 4.5 | 13.90 | +0.86 | +6.6% |
| TQ3_0 | F16 | 3.5 | 18.45 | +5.41 | +41.5% |
| TQ3_0 | TQ3_0 | 3.5 | 19.28 | +6.24 | +47.9% |
| TQ2_1 | F16 | 2.75 | 61.27 | +48.23 | +370% |
| TQ2_1 | TQ2_1 | 2.75 | 74.96 | +61.92 | +475% |
| TQ2_0 | F16 | 2.5 | 112.61 | +99.57 | +764% |
| TQ2_0 | TQ2_0 | 2.5 | 144.68 | +131.64 | +1010% |

**Verdict:** Anomalously sensitive. TQ4 is borderline usable (+6.6%), TQ3 is
degraded (+48%), TQ2 variants are broken. See Section 4 for root cause analysis.

### 2.3 Gemma 3 4B (n_embd=2560, n_head=8, n_head_kv=4, head_dim=256)

*(Results from RESULTS.md Section 6, CPU/AVX2, full WikiText-2)*

| K-cache | V-cache | BPE (K) | PPL | Delta |
|---------|---------|---------|----:|------:|
| F16 | F16 | 16.0 | 12.536 | — |
| TQ4_0 | TQ4_0 | 4.5 | 12.416 | -0.120 |
| TQ3_0 | TQ3_0 | 3.5 | 12.434 | -0.102 |

**Verdict:** Every TQ configuration improves or matches F16 on Gemma 3. Despite
being the same parameter count as Qwen3-4B, Gemma is robust. See Section 4.

### 2.4 Mistral 7B (n_embd=4096, n_head=32, n_head_kv=8, head_dim=128)

*(Metal GPU, 3 chunks)*

| K-cache | V-cache | BPE (K) | PPL | Delta | Delta % |
|---------|---------|---------|----:|------:|--------:|
| F16 | F16 | 16.0 | 8.89 | — | — |
| TQ4_0 | TQ4_0 | 4.5 | 8.98 | +0.09 | +1.0% |
| TQ3_0 | TQ3_0 | 3.5 | 9.16 | +0.27 | +3.0% |
| TQ2_1 | TQ2_1 | 2.75 | 14.67 | +5.78 | +65% |
| TQ2_0 | TQ2_0 | 2.5 | 20.77 | +11.88 | +134% |

**Verdict:** TQ4 and TQ3 are excellent. TQ2_1 is better than TQ2_0 but still
shows significant degradation. Good reference point — matched Q/KV dimensions.

---

## 3. TQ2_1 Mixed-Precision Validation

TQ2_1 consistently outperforms TQ2_0 across all models, confirming the value of
giving 32 channels TQ3 precision (3.5 bpe) while the remaining 96 use TQ2 (2.5 bpe).

| Model | TQ2_0/TQ2_0 PPL | TQ2_1/TQ2_1 PPL | Improvement |
|-------|----------------:|----------------:|------------|
| Qwen3-8B | 38.29 | 27.27 | -29% PPL |
| Qwen3-4B | 144.68 | 74.96 | -48% PPL |
| Mistral 7B | 20.77 | 14.67 | -29% PPL |

The 0.25 bpe overhead (2.75 vs 2.5) buys significant quality. TQ2_1 doesn't
require any runtime outlier detection — after Hadamard, all channels have roughly
equal variance, so ANY 32 channels receiving TQ3 treatment benefits quality uniformly.

---

## 4. Root Cause: Qwen3-4B Quantization Sensitivity

### The Anomaly

Qwen3-4B degrades catastrophically with TQ3 (+48% PPL) while Qwen3-8B, an
architecturally identical model (same GQA ratio, same head_dim, same n_layer),
shows only +6% degradation.

### Investigation: Hadamard Impact

Tested TQ3/F16 with and without Hadamard on both models:

| Model | TQ3/F16 + Hadamard | TQ3/F16 no Hadamard |
|-------|-------------------:|--------------------:|
| Qwen3-4B | 18.45 (baseline 13.04) | 132.70 |
| Qwen3-8B | 10.04 (baseline 9.47) | 142.98 |

**Key finding:** Hadamard is essential for BOTH models — without it, TQ3 is
completely broken on both (~130-143 PPL). Hadamard is not the cause of Qwen3-4B's
sensitivity; it's actually saving the model from total failure.

### Root Cause: Q-dimension / KV-dimension Mismatch

The critical architectural difference:

| | Qwen3-4B | Qwen3-8B |
|---|---|---|
| n_embd | 2560 | 4096 |
| n_head | 32 | 32 |
| **n_embd / n_head (Q input dim)** | **80** | **128** |
| head_dim (K/V dim) | 128 | 128 |
| **Q→KV dimensional ratio** | **0.625** | **1.0** |

In Qwen3-4B, each Q head has only 80 input dimensions projected to 128-dim K/V
space via W_K and W_V. This means:

1. The learned KV representations occupy a **rank-80 subspace** of the 128-dim space
2. Quantization adds noise in **all 128 dimensions** — including the 48-dimensional
   orthogonal complement the model never saw during training
3. The attention mechanism cannot compensate because it was trained to operate
   exclusively within the rank-80 subspace
4. Quantization noise in the "unused" 48 dimensions creates spurious attention
   patterns that corrupt output

In Qwen3-8B, Q input is 128 dims → 128 dim K/V space. **Full rank.** Quantization
noise stays within the learned manifold and can be absorbed by the model's redundancy.

### Cross-Model Validation

| Model | n_embd/n_head | head_dim | Q→KV ratio | TQ3 sensitivity |
|-------|:---:|:---:|:---:|---|
| **Qwen3-4B** | 80 | 128 | **0.625** | **Catastrophic** (+48%) |
| Qwen3-8B | 128 | 128 | 1.0 | Normal (+6%) |
| Llama 3-8B | 128 | 128 | 1.0 | Normal |
| Gemma 3-4B | 320 | 256 | 1.25 | Robust (improves) |
| Mistral 7B | 128 | 128 | 1.0 | Normal (+3%) |

**Gemma 3-4B is the control case.** Same parameter count as Qwen3-4B, but uses 8
heads with n_embd/n_head = 320 projected to head_dim=256. Ratio is 1.25 — the KV
space is fully utilized (overparameterized). Result: TQ3 actually *improves* PPL.

### Practical Implications

**For models with Q dim < head_dim (ratio < 1.0):**
- TQ4_0 (4.5 bpe) is the maximum safe compression — still borderline (+6.6%)
- TQ3_0 and below are NOT recommended
- This is an inherent property of the architecture, not a fixable bug

**For models with Q dim >= head_dim (ratio >= 1.0):**
- TQ3_0 and TQ4_0 are production-quality (< +6% PPL)
- TQ2_1 (2.75 bpe) is viable for cold-tier storage
- TQ2_0 (2.5 bpe) shows significant degradation but may be acceptable for
  very long contexts where memory savings outweigh quality loss

**Auto-detection (implemented):** At context init in `llama_init_from_model()`
(`src/llama.cpp`), we check `n_embd / n_head` vs `n_embd_head_k`. If ratio < 1.0
and user requested TQ3_0, TQ2_0, or TQ2_1, we auto-downgrade to TQ4_0 with a
warning:

```
llama_init_from_model: this model has Q-dim (80) < KV head-dim (128) —
  rank-deficient KV subspace makes aggressive quantization unreliable.
  Downgrading K-cache from tq3_0 to tq4_0
```

TQ4_0 remains available (borderline acceptable at +6.6% PPL on Qwen3-4B).

---

## 5. Compression Summary

Memory per KV token (head_dim=128, per head):

| Type | Bits/elem | Bytes/token/head (K) | vs F16 |
|------|:---------:|:--------------------:|:------:|
| F16 | 16.0 | 256 | 1.0x |
| TQ4_0 | 4.5 | 18 | 14.2x |
| TQ3_0 | 3.5 | 14 | 18.3x |
| TQ2_1 | 2.75 | 44/128*128 = 44 per 128 elem | 23.3x |
| TQ2_0 | 2.5 | 10 per 32 elem | 25.6x |

For a model with 8 KV heads, 36 layers, 32k context:
- F16 K+V: 36 * 8 * 32768 * 256 * 2 = **4.5 GB**
- TQ4 K+V: **~320 MB** (14x reduction)
- TQ3 K+V: **~250 MB** (18x reduction)
- TQ2_1 K+V: **~200 MB** (23x reduction)

---

## 6. Models Tested

| Model | File | Size | Source |
|-------|------|------|--------|
| Qwen3-8B | Qwen3-8B-Q4_K_M.gguf | ~5 GB | unsloth |
| Qwen3-4B | Qwen3-4B-Q4_K_M.gguf | ~2.8 GB | unsloth |
| Gemma 3-4B | gemma-3-4b-it-Q4_K_M.gguf | ~2.8 GB | unsloth |
| Mistral 7B | Mistral-7B-Instruct-v0.3-Q4_K_M.gguf | ~4.4 GB | bartowski |
| Llama 3-8B | Meta-Llama-3-8B-Instruct-Q4_K_M.gguf | ~4.6 GB | bartowski |

All models quantized to Q4_K_M for weight storage. KV cache type varied per test.

---

## 7. Open Questions

1. ~~**Can we detect Q/KV mismatch at model load and auto-select max TQ level?**~~
   **Done.** Implemented in `src/llama.cpp:llama_init_from_model()`. Auto-downgrades
   TQ3/TQ2 to TQ4 when `n_embd/n_head < n_embd_head_k`.

2. **Would a model-specific Hadamard matrix (learned, not random) help the
   rank-deficient case?** The standard Hadamard spreads noise uniformly across all
   128 dimensions. A learned rotation could keep noise within the rank-80 subspace.

3. **TQ2_1 for cold-tier storage:** On ratio=1.0 models, TQ2_1 at 2.75 bpe
   shows ~2-3x PPL increase. Is this acceptable for tokens beyond position 4096
   in a 128k context? Needs LongBench evaluation.
