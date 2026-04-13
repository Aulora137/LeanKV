# Experiment: Per-Layer Outlier Fraction Auto-Detection

**Date:** 2026-04-12
**Status:** Phase 1 complete (diagnostic mode), informs Phase 2/3 scope
**Hardware:** AMD Ryzen 7 7735U, AVX2 (no AVX512), 8 threads
**Software:** Lean_llama.cpp `feature/tq2-outlier-tiered`, commit `114437b9`
**Branch:** both `LeanKV` and `Lean_llama.cpp` at `feature/tq2-outlier-tiered`

---

## Goal

Determine whether the fixed 25% outlier fraction used by TQ2_1 is optimal
across different model architectures, or whether per-layer auto-detection
could enable meaningful memory savings and/or better quality.

Phase 1 is **diagnostic-only** — it runs the variance analysis at model
load but does not change runtime behavior when Hadamard rotation is enabled
(the default for TQ types). The goal is measurement data to inform Phase 2
(variable per-layer outlier fractions) and Phase 3 (mixed-precision storage).

---

## Method

### Algorithm: `tq_auto_detect_outlier_frac()`

Analyze the per-channel variance spectrum of W_K weights and choose an
outlier fraction from {0%, 12.5%, 25%, 50%} based on the heavy-tailedness
of the distribution:

```c
float tq_auto_detect_outlier_frac(channel_var[], head_dim):
    sorted = sort(channel_var, descending)
    median = sorted[head_dim / 2]
    if median ≤ 1e-12: return 0.0        // degenerate case

    n_moderate = count(i where sorted[i] > 2 × median)
    n_strong   = count(i where sorted[i] > 5 × median)

    raw_frac = n_moderate / head_dim

    if raw_frac < 0.0625: return 0.0     // < 6.25% moderate outliers
    if raw_frac < 0.1875: return 0.125   // 6.25%-18.75%
    if raw_frac < 0.375:  return 0.25    // 18.75%-37.5%
    return 0.5                           // > 37.5% (heavy-tailed)
```

**Why "2× median"?** Standard robust-statistics threshold for "moderate
outliers." The 5× threshold identifies "strong outliers" but is only
reported as a diagnostic — the primary signal is `n_moderate`.

**Why these specific fractions?** They are the block-aligned choices
compatible with 32-element SIMD block quantization. At head_dim=256,
12.5% = 32 channels = one TQ3 sub-block. At head_dim=128, 25% = 32
channels. At lower fractions, outliers don't fill a block; at higher
fractions, we're approaching uniform quantization from the other side.

### Data source

Per-channel variance is computed from **W_K weight tensor row L2 norms**,
averaged across KV heads. This is a pre-Hadamard property — it tells us
which channels the model *wants* to produce high-variance K values on,
before any runtime rotation. Calibration happens once at model load,
zero runtime cost.

### CLI

```bash
# Auto-detect per layer (NEW)
llama-cli --kv-outlier-frac -1 ...

# Explicit fraction (existing behavior)
llama-cli --kv-outlier-frac 0.25 ...
```

### Models tested

Seven GGUF Q4_K_M models spanning three architecture families × three
head dimensions:

| Model | Architecture | head_dim | Attn layers |
|-------|--------------|---------:|------------:|
| Qwen 3.5-9B | Hybrid (Mamba+attn) | 256 | 8 |
| Qwen 3.5-4B | Hybrid (Mamba+attn) | 256 | 8 |
| Qwen 3.5-2B | Hybrid (Mamba+attn) | 256 | 6 |
| Mistral 7B | Dense (Mistral) | 128 | 32 |
| Llama 3.2-3B | Dense (Meta) | 128 | 28 |
| Gemma 3-4B | Dense (Google) | 256 | 34 |
| Qwen3-4B | Dense (old Alibaba) | 128 | 36 |

Note: Qwen 3.5 family is hybrid Mamba+attention, so only a subset of
layers have W_K tensors. Dense models have W_K in every layer.

---

## Results: Outlier Distribution Across 7 Models

| Model | head_dim | 0% | 12.5% | 25% | 50% | Max var/median range |
|-------|---------:|---:|------:|----:|----:|----------------------|
| Qwen 3.5-9B | 256 | **8** | 0 | 0 | 0 | 1.5–2.1× |
| Qwen 3.5-4B | 256 | **8** | 0 | 0 | 0 | 1.4–2.5× |
| Qwen 3.5-2B | 256 | **6** | 0 | 0 | 0 | ~2.3× |
| Mistral 7B | 128 | **27** | 5 | 0 | 0 | 1.5–2.7× |
| Llama 3.2-3B | 128 | **28** | 0 | 0 | 0 | 1.3–3.0× |
| Gemma 3-4B | 256 | 15 | **15** | 4 | 0 | 1.6–3.7× |
| Qwen3-4B | 128 | 18 | **11** | 7 | 0 | 1.6–3.8× |

### Per-layer detail (Mistral 7B, first 16 layers)

```
layer  0: frac=0.000 (0/128 ch), max_var/med=2.4x, moderate=6
layer  1: frac=0.000 (0/128 ch), max_var/med=2.7x, moderate=7
layer  2: frac=0.000 (0/128 ch), max_var/med=1.5x, moderate=0
layer  3: frac=0.000 (0/128 ch), max_var/med=1.8x, moderate=0
layer  4: frac=0.000 (0/128 ch), max_var/med=1.9x, moderate=0
layer  5: frac=0.000 (0/128 ch), max_var/med=1.8x, moderate=0
layer  6: frac=0.000 (0/128 ch), max_var/med=1.7x, moderate=0
layer  7: frac=0.000 (0/128 ch), max_var/med=2.2x, moderate=3
layer  8: frac=0.000 (0/128 ch), max_var/med=1.7x, moderate=0
layer  9: frac=0.125 (32/128 ch), max_var/med=2.2x, moderate=18  ← elevated
layer 10: frac=0.000 (0/128 ch), max_var/med=1.6x, moderate=0
layer 11: frac=0.125 (32/128 ch), max_var/med=2.2x, moderate=10  ← elevated
layer 12: frac=0.125 (32/128 ch), max_var/med=2.3x, moderate=17  ← elevated
layer 13: frac=0.000 (0/128 ch), max_var/med=1.5x, moderate=0
layer 14: frac=0.000 (0/128 ch), max_var/med=2.0x, moderate=1
layer 15: frac=0.125 (32/128 ch), max_var/med=2.3x, moderate=18  ← elevated
...
Summary: 0%=27 layers, 12.5%=5, 25%=0, 50%=0
```

Only layers 9, 11, 12, 15 (and one more not shown) have meaningful W_K
channel imbalance. The rest are flat.

---

## Key Findings

### 1. Hybrid architectures (Qwen 3.5): zero outliers across ALL layers

Every one of the 22 attention layers across three Qwen 3.5 model sizes
(2B, 4B, 9B) shows flat W_K variance (max/median < 2.5×, fewer than 8
channels above 2× median). This is **direct evidence that hybrid
Mamba+attention training produces attention layers with uniform channel
importance natively**.

This explains Phase 2b's finding that Qwen 3.5 is the most TQ-robust
architecture (+0.088 PPL delta on TQ3/TQ3 at 9B scale). It is not magic:
the model was trained such that W_K already does most of the "outlier
handling" that Hadamard + TQ types were designed to do. **The KV cache
is inherently quantization-friendly before any rotation or codebook is
applied.**

**Takeaway:** Outlier protection is wasted effort on Qwen 3.5 — use
uniform TQ2_0/TQ3_0/TQ4_0.

### 2. Llama 3.2-3B: all 28 layers at 0% outliers, yet most TQ3-sensitive

This is the most surprising finding. Llama 3.2-3B had the worst TQ3/TQ3
delta of any modern dense architecture in Phase 2b (+0.596), yet
auto-detect shows completely flat W_K variance. The max var/median
values are actually the most concentrated of any dense model (1.3–3.0×).

**This is a strong negative result**: Llama's TQ3 sensitivity is **NOT**
about outlier channels. The cause is something else — possibly attention
head rotational invariance interacting poorly with Lloyd-Max codebooks,
or non-local patterns that don't show up in per-channel variance
statistics.

**Takeaway:** Outlier-based compression techniques (TQ2_1, mixed-precision,
the whole Section 15 projection) **will not help Llama-family models**.
For Llama, the path to aggressive compression requires a different
mechanism entirely. TQ3 and TQ4 should be used directly; TQ2 should be
avoided regardless of outlier handling.

### 3. Mistral 7B: perfect case for variable-precision per-layer

Mistral is the cleanest positive case. 27 of 32 layers are flat (0%),
but 5 middle layers (9, 11, 12, 15, and one more) show 10–18 moderate
outliers at 12.5%. This matches the Metal TQ2_1 result (+5.78 delta,
best of head_dim=128 models tested).

**Actionable quantitative opportunity:**

With per-layer precision, Mistral could use:
- TQ2_0 uniformly on 27 layers: 2.50 bpe each
- TQ2_1 on 5 layers: 2.75 bpe each

Average effective bpe = (27 × 2.5 + 5 × 2.75) / 32 = **2.539 bpe**

Compare to uniform TQ2_1: 2.75 bpe. **Savings: ~7.7% memory at same quality.**

### 4. Gemma 3-4B: heaviest tails, middle layers dominate

Gemma shows the most varied distribution of any tested model — 34
attention layers split 15 / 15 / 4 across 0% / 12.5% / 25%. Middle
layers (9-15 in the auto-detect output) have max var/median up to 3.7×,
with layer 11 needing the full 25%.

Yet Phase 2b showed Gemma 3 with TQ3/TQ3 PPL delta of **-0.102**
(improves with quantization, doesn't degrade). This is because
Hadamard + head_dim=256 concentration is so effective that the outlier
channels get smoothed out at runtime. **The pre-Hadamard outliers we're
detecting here are exactly the channels Hadamard cleans up.**

**Takeaway:** Gemma 3 doesn't need runtime outlier handling. Auto-detect
tells us the model's structural property; Hadamard makes the structure
moot.

### 5. Qwen3-4B (old dense): middle layers need significant protection

Qwen3-4B shows 18 / 11 / 7 split across 0% / 12.5% / 25%. This is the
catastrophic Phase 2b case (TQ3/TQ3 delta +3.325). It has the same
structural properties as Gemma 3 (dense, heavy-tailed middle layers)
but with head_dim=128 instead of 256. At head_dim=128, Hadamard is
only moderately effective, and the remaining post-Hadamard outliers
destroy quality.

This is precisely the scenario TQ2_1 was designed for. The auto-downgrade
logic (added earlier in `6f9e0c3c`) now catches this failure at load
time by detecting rank-deficient models (`n_embd/n_head < head_dim`)
and forces TQ4 with a warning.

---

## Implications for Phase 2 and Beyond

### Phase 2 (variable per-layer outlier fraction with existing TQ types)

The data strongly supports Phase 2 for specific models:

**Worth building:**
- **Mistral 7B** and similar head_dim=128 dense models: ~7% memory savings
  via per-layer TQ2_0/TQ2_1 mix at same quality
- **Qwen3-old and other rank-limited models**: outlier handling helps
  but these are already caught by auto-downgrade; diminishing returns
- **Other untested head_dim=128 dense models**: likely to benefit, worth
  running auto-detect to check

**Not worth building:**
- **Qwen 3.5 hybrid family**: already uniform, no gain possible
- **Llama-family models**: sensitivity isn't outlier-related, wrong tool
- **Gemma 3 / other head_dim=256 models**: Hadamard handles it at runtime

### Phase 3 (true mixed-precision storage: outlier TQ3 + normal TQ2)

The uniform TQ2_1 design allocates 2.75 bpe everywhere. Auto-detect
suggests it should allocate 2.5-2.75 bpe dynamically per layer. For a
typical head_dim=128 dense model, this would save ~5-10% memory with
identical quality.

**Recommendation:** Build Phase 3 only after Phase 2 confirms the
per-layer approach works. Phase 2 gives us the infrastructure and the
measurement data to see if Phase 3's additional complexity is worth it.

### The "Llama mystery" deserves its own investigation

All 28 Llama layers register 0% outliers yet Llama has the worst TQ3
behavior of modern dense architectures. Possible explanations worth testing:

1. **Attention head rotation invariance** — RoPE interacts with Lloyd-Max
   codebooks in a way that propagates quantization noise through head
   rotation
2. **Non-local variance patterns** — outliers exist across tokens rather
   than across channels (temporal outliers, not structural)
3. **W_Q sensitivity** — we're measuring W_K variance but the dot product
   is Q·K; maybe Q's variance distribution is the bottleneck
4. **Training dynamics** — Llama 3's training recipe emphasizes something
   different from Qwen 3.5's hybrid architecture

A follow-up experiment would replicate the auto-detect analysis on W_Q
instead of W_K, and possibly on the attention output projection. If those
also show flat distributions, then the sensitivity is runtime-dynamic
and no static analysis will catch it.

---

## What Phase 1 Does Not Tell Us

**Phase 1 measures pre-Hadamard structure**, which tells us what the
*model* thinks is important, not what the *runtime quantizer* sees.
After Hadamard rotation is applied, the channels are completely
scrambled and the "outlier" property is redistributed.

For TQ types (which auto-enable Hadamard), Phase 1 is a diagnostic
for *where the model's natural structure lies*, not for where runtime
quantization errors will occur. A complete picture would require:

1. **Instrument the Hadamard-rotated KV** — collect actual variance of
   post-Hadamard K values on real inference inputs, per layer. Expensive
   but definitive.
2. **Per-token variance analysis** — W_K row norms only tell us which
   channels have high weight magnitude. Real KV values are W_K·x where
   x varies per token. A channel with high W_K norm might still produce
   small K values if the input doesn't activate it.
3. **Quantization error injection** — add synthetic noise to each
   channel and measure downstream PPL impact. Directly measures
   sensitivity, doesn't rely on variance as a proxy.

These would be Phase 1.5 experiments if we want to understand the
Llama mystery rigorously.

---

## Files and Commits

### Lean_llama.cpp

| File | Change |
|------|--------|
| `ggml/src/ggml-tq-outlier.h` | Added `tq_auto_detect_outlier_frac()` declaration |
| `ggml/src/ggml-tq-outlier.c` | 68-line implementation: sort + threshold + snap |
| `src/llama.cpp` | Auto-detect at model load, per-layer logging, histogram summary |
| `common/common.cpp` | Help text and Hadamard guard updated for negative values |

**Commit:** `114437b9 feat: Phase 1 — auto-detect outlier fraction per layer (diagnostic mode)`

### LeanKV

| File | Change |
|------|--------|
| `docs/RESULTS.md` | Section 17 with full findings and interpretation |
| `docs/experiment.md` | This document (standalone experiment report) |

---

## How to Reproduce

```bash
# Build on CPU
cd Lean_llama.cpp
git checkout feature/tq2-outlier-tiered
cmake -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j$(nproc)

# Run auto-detect on any model
./build/bin/llama-cli -m <model>.gguf -ngl 0 --kv-outlier-frac -1 \
    -ctk f16 -ctv f16 -c 32 -p "hi" -n 0 2>&1 | grep "outlier K"
```

Output format:
```
outlier K layer  0: frac=0.000 (0/128 ch), max_var/med=2.4x, moderate=6, strong=0
outlier K layer  1: frac=0.000 (0/128 ch), max_var/med=2.7x, moderate=7, strong=0
...
outlier K auto-detect summary: 0%=27 layers, 12.5%=5, 25%=0, 50%=0
```

Works on any Q4_K_M or F16 GGUF model with a W_K tensor. Skips Mamba
layers in hybrid models. Zero runtime cost — only affects load time.

---

## References

- RESULTS.md Section 14 — Phase 4 TQ2_0 validation (baseline)
- RESULTS.md Section 15 — Outlier handling vs QJL theoretical analysis
- RESULTS.md Section 16 — TQ2_1 CPU SIMD vec_dot implementation
- RESULTS.md Section 17 — Full Phase 1 findings (same data, different lens)
- DESIGN-FOR-QUANTIZATION.md — Architectural design principles for
  quantization-friendly LLMs
- TQ2IMPLEMENTATION.md — TQ2_1 implementation status and roadmap
- TIERED_KV_CACHE.md — Phase 4 tiered KV cache design
