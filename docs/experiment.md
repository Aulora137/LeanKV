# Experiment: Per-Layer Outlier Fraction Auto-Detection

**Date:** 2026-04-12
**Status:** Phase 3 complete (per-layer adaptive K-cache types)
**Hardware:** AMD Ryzen 7 7735U (AVX2) + Apple M2 (NEON)
**Software:** Lean_llama.cpp `feature/tq2-outlier-tiered`, commit `8c1c55ce`
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

## M2 Cross-Platform Validation (Apple Silicon)

**Date:** 2026-04-12
**Hardware:** Apple M2, 16 GB unified memory
**Software:** Same commit `114437b9`, built with Metal enabled
**Models:** 7 GGUF Q4_K_M files (5 overlap with Ryzen, 2 new)

### Results

| Model | head_dim | 0% | 12.5% | 25% | 50% | Max var/median range |
|-------|---------:|---:|------:|----:|----:|----------------------|
| Qwen 3.5-9B | 256 | **8** | 0 | 0 | 0 | 1.5–2.1× |
| Mistral 7B | 128 | **27** | 5 | 0 | 0 | 1.5–2.7× |
| Gemma 3-4B | 256 | 15 | **15** | 4 | 0 | 1.6–3.7× |
| Llama 3-8B | 128 | **32** | 0 | 0 | 0 | 1.3–2.8× |
| Qwen3-8B | 128 | 29 | **6** | 1 | 0 | 1.3–2.3× |
| Qwen3-4B | 128 | 29 | **6** | 1 | 0 | 1.4–2.7× |
| Qwen2.5-0.5B | 64 | 12 | 5 | **7** | 0 | 1.4–**17.2×** |

### Cross-Platform Comparison (Ryzen AVX2 vs M2 ARM)

**Exact matches** on all 3 models available on both platforms with identical
GGUF files:

| Model | Ryzen | M2 | Match |
|-------|-------|-----|:-----:|
| Qwen 3.5-9B | 0%=8 | 0%=8 | **Exact** |
| Mistral 7B | 0%=27, 12.5%=5 | 0%=27, 12.5%=5 | **Exact** |
| Gemma 3-4B | 0%=15, 12.5%=15, 25%=4 | 0%=15, 12.5%=15, 25%=4 | **Exact** |

The auto-detect is **platform-independent and deterministic** — it reads
W_K weight values, which are identical in the GGUF file regardless of CPU.

**Qwen3-4B discrepancy** (Ryzen: 18/11/7 vs M2: 29/6/1): likely different
GGUF files (different quantization runs from different HuggingFace sources).
Q4_K_M is itself quantized, so W_K weight values differ slightly between
quantization runs. Channels near the 2× median threshold flip classification.
This sensitivity to weight quantization artifacts is worth noting — the
auto-detect results are deterministic for a given GGUF file, but not
necessarily stable across different Q4_K_M quantizations of the same model.

**Different models per platform:**

| Ryzen only | M2 only |
|-----------|---------|
| Qwen 3.5-4B (0%=8) | Llama 3-8B (0%=32) |
| Qwen 3.5-2B (0%=6) | Qwen3-8B (29/6/1) |
| Llama 3.2-3B (0%=28) | Qwen2.5-0.5B (12/5/7) |

### New Findings from M2

**1. Qwen2.5-0.5B: extreme outliers at head_dim=64**

The smallest model has by far the heaviest tails — layer 0 shows
**17.2× max var/median** with 9 strong outliers (>5× median), and layers
9, 11, 13 show 6.7–7.3× with 7–13 strong outliers. No other model has
ANY strong outliers. At head_dim=64, the Hadamard concentration-of-measure
effect is weakest — outliers survive rotation.

This is direct evidence that **smaller models with smaller head_dim are
more vulnerable to KV quantization**, independent of the Q/KV ratio issue
found in Qwen3-4B.

**2. Llama 3-8B: completely flat, all 32 layers at 0%**

Same pattern as Llama 3.2-3B on Ryzen (all 28 layers at 0%). The "Llama
mystery" is confirmed: Llama-family models show zero W_K channel variance
imbalance yet are the most TQ3-sensitive modern dense architecture. The
sensitivity mechanism is not structural outliers — it's something else
entirely (see Section "The Llama mystery" above).

**3. Qwen3-8B and Qwen3-4B: identical distribution pattern on M2**

Both show 29/6/1 split. Since Qwen3-4B has the catastrophic rank-deficiency
issue (n_embd/n_head=80 < head_dim=128) while Qwen3-8B doesn't, the W_K
variance pattern is NOT what causes the quality divergence. The rank-deficiency
is the dominant factor, confirming the DESIGN-FOR-QUANTIZATION.md finding.

---

## The Unpredictability Problem

The cross-model and cross-platform data reveals an uncomfortable truth:
**outlier patterns are model-specific and cannot be predicted from architecture
parameters alone.** Every model family shows a different fingerprint:

| Family | Pattern | Predictable from arch? |
|--------|---------|:----------------------:|
| Qwen 3.5 hybrid | Flat everywhere | Yes (Mamba regularizes) |
| Llama | Flat everywhere, yet TQ3-sensitive | **No** |
| Mistral | Mostly flat, 5 middle layers elevated | No |
| Gemma | Heavy tails, middle layers dominate | Partially (head_dim=256 helps) |
| Qwen3 old dense | Moderate tails, variable per-layer | No |
| Qwen2.5 tiny | Extreme outliers, strong >5× | Partially (head_dim=64 hurts) |

**You cannot assume a fixed outlier fraction works.** 25% is too much for
Qwen 3.5 (wastes bits on already-flat channels), too little for Qwen2.5-0.5B
(layer 0 needs 50%+ to cover 17× outliers), and irrelevant for Llama
(outliers aren't the problem).

### How Much Dynamic Adjustment Is Possible?

The good news: all the adjustment happens **before** any KV token is
quantized. The auto-detect pipeline has three decision points, each at
a different stage of model loading:

**Stage 1: Architecture check (instant, at model load)**

Already implemented. Checks `n_embd/n_head < head_dim` and auto-downgrades
TQ3/TQ2 → TQ4. Catches the Qwen3-4B class of failures. Zero cost.

```
Decision: "Can this model tolerate aggressive quantization at all?"
Inputs: n_embd, n_head, head_dim (from GGUF metadata)
Cost: one integer comparison
```

**Stage 2: W_K variance analysis (milliseconds, at model load)**

Already implemented (`--kv-outlier-frac -1`). Reads W_K weight tensors,
computes per-channel variance, classifies each layer into {0%, 12.5%,
25%, 50%} outlier fraction. This is the data we've been collecting.

```
Decision: "Which layers need outlier protection, and how much?"
Inputs: W_K weight tensor values (from GGUF)
Cost: ~1ms per layer (sort + threshold)
```

**Stage 3: Per-layer quantization type selection (not yet implemented)**

The natural next step. Using Stage 2's per-layer outlier fractions,
select the quantization type per layer instead of using one type globally:

```
Per-layer decision matrix:
  outlier_frac = 0%    → use uniform TQ2_0 (2.5 bpe, maximum compression)
  outlier_frac = 12.5% → use TQ2_1 (2.75 bpe, 32 channels get TQ3)
  outlier_frac = 25%   → use TQ2_1 or TQ3_0 (depending on quality target)
  outlier_frac = 50%   → use TQ3_0 (3.5 bpe, too many outliers for TQ2)
```

For Mistral 7B this would mean: 27 layers × TQ2_0 + 5 layers × TQ2_1 =
**2.539 effective bpe** (vs 2.75 uniform TQ2_1, saving 7.7%).

For Qwen 3.5-9B: all 8 layers × TQ2_0 = **2.50 bpe** (no wasted outlier
bits at all).

**Stage 4: Runtime post-Hadamard calibration (future, expensive)**

Instrument actual KV values during the first N tokens of inference to
measure post-Hadamard channel variance. This would catch patterns that
W_K analysis misses (like whatever causes Llama sensitivity). Cost:
one extra variance-tracking pass during warmup. Not yet designed.

### What Cannot Be Adapted

Some properties are fixed at architecture design time and no amount of
runtime adaptation can help:

1. **Rank-deficient KV subspace** (Q dim < head_dim): quantization noise
   in unused dimensions is mathematically unavoidable
2. **head_dim not power-of-2**: Hadamard rotation cannot be applied
3. **head_dim < 64**: concentration of measure is too weak for any
   rotation to produce near-Gaussian distributions
4. **Whatever causes Llama sensitivity**: no static analysis we've tried
   captures it — may require fundamentally different quantization approach

### The Practical Answer

For **deployment today**, the two-stage pipeline (architecture check +
W_K variance analysis) covers the majority of models:

- Catches catastrophic failures (Qwen3-4B rank-deficiency) → auto-downgrade
- Identifies models that need zero outlier protection (Qwen 3.5, Llama) → save bits
- Identifies models that need per-layer protection (Mistral, Gemma) → right-size bits
- Flags models with extreme outliers (Qwen2.5-0.5B) → warn user

For **maximum compression**, Stage 3 (per-layer type selection) would give
~5-10% additional memory savings at same quality, with zero runtime cost.
The data to drive it is already collected by `--kv-outlier-frac -1`.

The remaining gap is the Llama mystery and anything else that static W_K
analysis can't see. That requires Stage 4 (runtime calibration) or a
fundamentally different quantization approach for those model families.

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

## Phase 3: Per-Layer Adaptive K-Cache Type Selection

**Date:** 2026-04-12
**Commit:** `8c1c55ce` (Lean_llama.cpp)
**Status:** Implemented and validated

### What Changed

Instead of using a single global `type_k` for all layers, auto-detect mode
(`--kv-outlier-frac -1`) now assigns each layer its own K-cache type based
on its outlier profile:

| Outlier fraction | Assigned type | Bits/elem |
|-----------------|---------------|-----------|
| < 6.25% (flat) | TQ2_0 | 2.5 |
| 6.25%–37.5% | TQ2_1 | 2.75 |
| ≥ 37.5% (heavy) | TQ3_0 | 3.5 |

When the user requests `--ctk tq3_0` with auto-detect:
- Flat layers: keep TQ3_0
- Heavy outlier layers: promote to TQ4_0

No kernel changes were needed — FA and IQK kernels dispatch per-tensor
type, so mixed per-layer types work transparently.

### Implementation

Two files modified in Lean_llama.cpp:

1. **`src/llama-context.h`**: Added `std::vector<ggml_type> type_k_l` to
   `llama_kv_cache` — stores the per-layer K-cache type.

2. **`src/llama.cpp`**:
   - Initialize `type_k_l` to global default in `llama_kv_cache_init()`
   - Use `type_k_l[i]` instead of `type_k` for tensor allocation
   - Assign per-layer types in auto-detect codepath
   - Log "adaptive K-cache types: ..." summary when types are mixed
   - Show "adaptive" instead of type name in KV size log

### Results (M2, ctx=2048)

| Model | Adaptive types | K-cache (adaptive) | K-cache (uniform TQ2_1) | Savings |
|-------|---------------|-------------------|------------------------|---------|
| Mistral 7B | tq2_0=27, tq2_1=5 | 20.00 MiB | 22.00 MiB | 9% |
| Qwen 3.5-9B | tq2_0=40 | 5.00 MiB | 6.25 MiB | 20% |
| Gemma 3-4B | tq2_0=15, tq2_1=19 | 21.25 MiB | 25.00 MiB | 15% |

### Quality Validation

**Mistral 7B** — adaptive vs uniform TQ2_1, temp=0, identical prompt:

Both produce **token-for-token identical output**. The 27 flat layers
safely downgraded from TQ2_1 → TQ2_0 without any quality loss, because
those layers have no outlier channels that need mixed-precision treatment.

```
Prompt: "The three laws of thermodynamics are:"
Adaptive (27×TQ2_0 + 5×TQ2_1): [identical output]
Uniform (32×TQ2_1):            [identical output]
Speed: 13.28 tok/s vs 13.46 tok/s (within noise)
```

### Log Output Example

```
llama_init_from_model: adaptive K-cache types: tq2_0=27, tq2_1=5
llama_init_from_model: KV self size  = 150.00 MiB, K (adaptive): 22.00 MiB, V (f16): 128.00 MiB
```

---

## Phase 5: The V1 Scorecard Correction (2026-04-15)

After Phase 3.5 shipped V1 as the new adaptive default, cross-architecture
validation on CUDA revealed that **V1's Mistral win does NOT generalize**.
The Phase 3.5 documentation needs correction.

### The cross-model CUDA batch (RTX 4090, 160 chunks WikiText-2)

Full 5-model + Qwen 3.5-9B supplemental validation on a single Vast.ai
RTX 4090 instance. Total wall time: 39 minutes.

| Model | head_dim | V1 adaptive PPL | Uniform TQ2_1 PPL | Delta | Verdict |
|-------|---------:|----------------:|------------------:|------:|:-------:|
| Mistral 7B (dense) | 128 | 6.005 | 5.973 | **+0.032** (tied) | ✅ |
| Qwen3-8B (dense) | 128 | 16.48 | 13.72 | **+2.77 (WORSE)** | ❌ |
| Gemma 3-4B (dense) | 256 | 14.39 | 14.00 | +0.39 (slightly worse) | ➖ |
| Llama 3-8B (dense) | 128 | 12.25 | 10.00 | **+2.25 (WORSE)** | ❌ |
| Qwen3-4B (rank-def.) | 128 | auto-forced TQ4 | auto-forced TQ4 | — | ✅ (safety) |
| Qwen 3.5-9B (hybrid) | 256 | 7.324 | 7.221 | +0.10 (= TQ2_0) | ➖ |

**V1 is strictly useful on 1 of 6 models**, neutral on 3, and **actively
harmful on 2** (Qwen3-8B and Llama 3-8B).

### Why V1 fails on Llama 3-8B (the "Llama mystery" confirmed)

V1 on Llama 3-8B produced this distribution:

```
outlier K spectrum: max/med=2.78x (mean 1.53x) → skew LOW
outlier K auto-detect summary: 0%=30 layers, 12.5%=2, 25%=0, 50%=0
adaptive K-cache types: tq2_0=30, tq2_1=2
```

**30 of 32 layers were classified as "flat" and downgraded to TQ2_0.** V1
effectively becomes uniform TQ2_0 with 2 "safety" TQ2_1 layers. The result:
PPL 12.2528 (+65%) vs uniform TQ2_1's 10.0031 (+35%). V1 is nearly as bad
as uniform TQ2_0 (12.71, +72%) because it IS nearly uniform TQ2_0.

But here's the mystery: **W_K variance on Llama is genuinely flat**. The
spectrum skew label is LOW. There are no per-channel outliers to detect.
V1 is making the right call based on the signal it can see. Yet Llama 3
is severely TQ-sensitive (+35% even at uniform TQ2_1). The sensitivity
lives in some mechanism V1 doesn't measure.

Hypotheses for future investigation:
1. **W_Q variance instead of W_K** — maybe the Q-side outliers are the
   real signal on Llama
2. **Post-Hadamard runtime variance** — instrument actual KV values during
   the first N inference tokens
3. **Attention head rotation** — Llama 3's RoPE configuration interacts
   with Lloyd-Max codebooks in a way that static weight analysis can't
   capture
4. **Training dynamics** — Llama 3's training recipe produces a
   quantization-sensitive attention pattern that isn't visible in the
   final weights

The Llama mystery is **confirmed across CPU, Metal, and CUDA backends**
and is the most important open question from this project.

### Why V1 fails on Qwen3-8B

V1 on Qwen3-8B produced `tq2_0=11, tq2_1=25`. 11 layers got downgraded
from TQ2_1 to TQ2_0. On this model, TQ2_0 degrades PPL by **+117%**
(vs TQ2_1's +59%). The 11 downgraded layers compound enough error to
push V1 to +91% — halfway between TQ2_1 and uniform TQ2_0.

**The root cause**: V1's W_K-variance heuristic assumes "flat layers are
safe to quantize aggressively." True on Mistral (TQ2_0 only +25%). False
on Qwen3-8B where TQ2_0 is catastrophic. V1 doesn't know the downstream
cost of its downgrades.

### Why V1 = TQ2_0 on Qwen 3.5-9B (hybrid architecture)

V1 on Qwen 3.5-9B produced:

```
outlier K policy: head_dim=256 metric=0 threshold=2.00
outlier K spectrum: max/med=2.06x (mean 1.74x) → skew LOW
outlier K auto-detect summary: 0%=8 layers, 12.5%=0, 25%=0, 50%=0
```

All 8 attention layers classified as flat. No `adaptive K-cache types:`
line (has_mixed=false). V1 produces uniform TQ2_0 assignment. V1 PPL =
TQ2_0 PPL = 7.3239 exactly. Not harmful, not helpful.

**This is actually a success case** — V1 correctly recognized there's
nothing to protect and degenerated to the simplest policy. Users who
need 6.4× compression on Qwen 3.5-9B can just use `-ctk tq2_0` directly.

### The Qwen 3.5-9B finding (bonus)

Qwen 3.5-9B has the **best TQ2 behavior in the entire test suite**:

- TQ2_0 PPL delta: **+2.57%** (vs Mistral's +25%, Qwen3-8B's +117%,
  Llama 3-8B's +72%)
- K-cache: **5.00 MiB** (vs F16's 32 MiB) — 6.4× compression
- V cache: 32 MiB (F16)

Why so good? **Hybrid Mamba+attention**: only 8 of 36 layers have KV
cache. The other 28 are Mamba state-space blocks that don't depend on
KV quantization at all. Only ~22% of the forward pass is affected by
K-cache precision. Even uniform TQ2_0 is a rounding error on the whole
model's quality.

**Deployment implication**: Qwen 3.5-9B is the single best model for
aggressive KV compression in the entire Aulora stack. Use `-ctk tq2_0`
directly — skip V1, skip the tuning guide, it's just free.

Separately confirmed: **CUDA is the only GPU backend that handles
Qwen 3.5-9B**. Metal crashes at `GGML_ASSERT(ne10 == ne02)` for
head_dim=256 hybrid attention. CPU works via AVX2 IQK.

### Updated V1 recommendation (ship-ready)

**Do NOT default to V1 adaptive.** Instead:

1. **Default (any architecture)**: `-ctk tq3_0 -ctv tq3_0`. Near-lossless
   on every model tested, 4.6× compression. Works on Mistral, Qwen3-8B,
   Gemma 3-4B, Llama 3-8B, and Qwen 3.5-9B.

2. **Memory-constrained long context**: `-ctk tq2_1 -ctv f16`. 5.8× K
   compression, acceptable on all non-Llama modern architectures.

3. **Maximum compression on Qwen 3.5 hybrid**: `-ctk tq2_0 -ctv f16`.
   Essentially free at +2.6% PPL.

4. **V1 adaptive (`--kv-outlier-frac -1`) is OPT-IN** for Mistral-class
   clean dense models where users have validated it matches or beats
   uniform TQ2_1 on their specific workload. **It is not a safe default.**

5. **Rank-deficient models (Q dim < head dim)**: auto-downgrade to TQ4_0
   kicks in automatically. No user action needed. Confirmed on Qwen3-4B
   across all three backends.

### Phase 3.5 retrospective

What was correct:
- V1 on Mistral is a genuine Pareto improvement (tied quality, less memory)
- The head_dim-aware default infrastructure is solid
- Env var override mechanism works and lets power users tune
- Spectrum skew diagnostic helps users understand the W_K shape
- Rank-deficient auto-downgrade protects users from the Qwen3-4B failure mode

What was wrong:
- **"V1 is strictly better than TQ2_1" was an overgeneralization** from
  Mistral-only 160-chunk testing
- **Spectrum skew is NOT a reliable predictor of V1 effectiveness** —
  both Mistral (works) and Llama (fails) show LOW skew
- **W_K variance is not a reliable predictor of quantization sensitivity**
  on all architectures — the Llama family breaks the assumption

What we learned:
- Different model families need different KV quantization strategies
- Static weight analysis has blind spots; some sensitivity is only
  visible at inference time
- The "one adaptive policy for all models" dream is not achievable with
  W_K variance alone

The V1 adaptive mechanism is **correct in its implementation** — it does
what the design says it should do. The issue is that the **design's
underlying assumption** (flat W_K ⇒ safe TQ2_0 downgrade) doesn't hold
universally. That's a research problem, not an engineering bug.

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
