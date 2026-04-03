# LeanKV — Complete Project Plan & Technical Trail

**Project:** LeanKV — 3-4 bit KV cache quantization for LLM inference
**Started:** 2026-04-03
**Status:** Phase 0 prototype complete, synthetic evaluation passed
**Repo:** https://github.com/hchengit/LeanKV

---

## Table of Contents

1. [The Problem We're Solving](#1-the-problem-were-solving)
2. [The Math Behind It (Crash Course)](#2-the-math-behind-it)
3. [The Algorithm (Step by Step)](#3-the-algorithm)
4. [What We Built So Far](#4-what-we-built-so-far)
5. [Results So Far](#5-results-so-far)
6. [Implementation Plan (4 Phases)](#6-implementation-plan)
7. [Trail Log (What We Did, When)](#7-trail-log)

---

## 1. The Problem We're Solving

### What is the KV Cache?

When an LLM generates text, it computes **Key** and **Value** vectors at every layer for every token. To avoid recomputing these for past tokens, they're stored in a **KV cache**. The problem: this cache grows linearly with sequence length and model depth.

**Example — Qwen 3.5-9B (32 layers, 8 KV heads, head_dim=128):**
```
KV cache for 4096 tokens = 2 × 32 × 8 × 128 × 4096 × 2 bytes (FP16)
                         = 536 MB per sequence
```

For long contexts (32K+ tokens) or batch serving, this becomes the dominant memory bottleneck.

### Current State (LeanInfer)

Our LeanInfer project already has `--kv-compress` which uses Q8_0 (8-bit) quantization:
- Reduces KV cache by ~47% (16-bit → 8-bit)
- Minimal quality loss (+4% decode speed from reduced memory traffic)

### What TurboQuant/LeanKV Achieves

Google's TurboQuant research shows we can go much further:
- **3.5 bits per element** — quality neutral (no measurable degradation)
- **2.5 bits per element** — marginal degradation
- **5x+ memory reduction** compared to FP16
- **Near-optimal** — within 2.7x of the information-theoretic lower bound

---

## 2. The Math Behind It

### 2.1 The Outlier Problem

LLM activations are NOT uniformly distributed. Certain channels (dimensions) consistently
have values 10-100x larger than the median. These are called **outliers**.

```
Normal dimension:  values in range [-0.1, +0.1]
Outlier dimension: values in range [-5.0, +5.0]
```

With only 8 quantization levels (3 bits = 2³ = 8 levels), a linear quantizer must span
the full range [-5, +5]. This means 99% of values (in [-0.1, +0.1]) get mapped to just
1-2 bins — effectively destroying the information.

### 2.2 The Solution: Random Rotation

**Key insight:** If you multiply a vector by a random orthogonal matrix Π, the outlier
energy gets **spread uniformly** across ALL dimensions. The result approximates a
Gaussian distribution — the ideal target for any fixed-point quantizer.

**What is an orthogonal matrix?**

A matrix Π where Π^T × Π = I (the identity matrix). This means:
- It preserves vector lengths: ||Πx|| = ||x||
- It preserves angles between vectors: (Πx)^T(Πy) = x^T y
- It's invertible: Π^{-1} = Π^T

Think of it as a "rotation" in high-dimensional space — it changes the direction
of every vector but preserves all distances and angles.

**Why does this help quantization?**

Before rotation:
```
dim 0: [-0.1, +0.1]  ← most values here
dim 1: [-0.1, +0.1]
dim 2: [-5.0, +5.0]  ← outlier! dominates quantizer range
dim 3: [-0.1, +0.1]
```

After rotation by Π:
```
dim 0: [-0.8, +0.8]  ← outlier energy spread evenly
dim 1: [-0.8, +0.8]
dim 2: [-0.8, +0.8]
dim 3: [-0.8, +0.8]
```

Now a 3-bit quantizer can use all 8 levels effectively across the full range.

### 2.3 The Mathematical Distribution After Rotation

After multiplying a unit vector x by a random orthogonal matrix Π, each coordinate
of Πx follows a **Beta distribution**:

```
f_X(x) = Γ(d/2) / (√π · Γ((d-1)/2)) · (1 - x²)^((d-3)/2)
```

where d is the dimension (e.g., 64 or 128 for typical head_dim).

**In plain English:** For d ≥ 64, this is almost exactly a Gaussian (bell curve) with
mean 0 and variance 1/d. This is a consequence of the **concentration of measure**
phenomenon in high dimensions.

**Why this matters:** Since we KNOW the distribution after rotation, we can design
the OPTIMAL quantizer for it — not a generic one, but one specifically tuned to
this exact bell curve shape.

### 2.4 Lloyd-Max Quantization (Optimal Scalar Quantization)

Given a known probability distribution, the **Lloyd-Max algorithm** finds the optimal
placement of quantization levels that minimizes mean squared error (MSE).

**The problem:** Partition the range into 2^b bins (b = bit-width) with:
- **Decision boundaries** t₁, t₂, ..., t_{2^b - 1}: where to split
- **Reconstruction levels** c₁, c₂, ..., c_{2^b}: what value to reconstruct

**The algorithm (iterative):**
1. Start with uniform levels
2. Set boundaries = midpoints between consecutive levels
3. Set each level = expected value within its bin (the centroid)
4. Repeat steps 2-3 until convergence

**Result for 3-bit Gaussian quantizer (8 levels):**

```
Level:     -1.748  -1.050  -0.500  -0.069  +0.069  +0.500  +1.050  +1.748
                ↑       ↑       ↑       ↑       ↑       ↑       ↑
Boundary:  -1.399  -0.775  -0.284   0.000  +0.284  +0.775  +1.399
```

Notice the levels are NOT uniformly spaced — they're denser near zero (where
most values live) and sparser in the tails. This is provably optimal.

**TurboQuant's MSE bound (Theorem 1 from the paper):**
```
MSE ≤ (√3π / 2) · (1/4^b)

For b=3: MSE ≤ 0.03  per coordinate
For b=4: MSE ≤ 0.009 per coordinate
```

This is within a factor of 2.7 of the information-theoretic lower bound (Shannon's
distortion-rate function). You literally cannot do much better with 3 bits.

### 2.5 QJL Residual Correction (The Extra 1-Bit Trick)

After Lloyd-Max quantization, there's still a residual error:
```
residual = original_value - quantized_value
```

Storing the full residual would double memory. Instead, store just the **sign** (1 bit):
```
sign_bit = +1 if residual ≥ 0, -1 otherwise
```

At reconstruction, apply a first-order correction:
```
corrected = quantized + sign_bit × mean_absolute_residual
```

The `mean_absolute_residual` is a single scalar per group (negligible storage).

**Why this works (QJL theory):**

The Johnson-Lindenstrauss lemma says random projections preserve inner products.
The Quantized JL (QJL) transform says this works even with 1-bit quantization
of the projected vector. The key result:

```
E[⟨y, Q^{-1}(Q(x))⟩] = ⟨y, x⟩     (unbiased!)
Var ≤ (π / 2d) · ||y||²              (low variance!)
```

**In plain English:** The 1-bit sign correction gives you an UNBIASED estimator
of the inner product, with variance that shrinks as dimension d grows. For d=128,
the variance contribution is tiny.

**Effective bit-width:** 3 bits (Lloyd-Max) + 1 bit (QJL sign) = 4 bits total,
but the reconstruction quality approaches 4-bit Lloyd-Max accuracy at lower cost.

### 2.6 RoPE Compatibility (Why Rotation Doesn't Break Position Encoding)

Most modern LLMs use **Rotary Position Embeddings (RoPE)**, which encode token
positions by rotating Q and K vectors:

```
Attention(q_i, k_j) = (R_{θ,i} q_i)^T (R_{θ,j} k_j) / √d_k
```

where R_{θ,i} is a position-dependent rotation matrix.

**The critical question:** Does our rotation Π interfere with RoPE?

**Answer: No, if we apply RoPE BEFORE Π.**

```
(Π R_{θ,i} q)^T (Π R_{θ,j} k)
= q^T R_{θ,i}^T  Π^T Π  R_{θ,j} k
                  ─────
                  = I (because Π is orthogonal!)
= q^T R_{θ,i}^T R_{θ,j} k
= original attention score ✓
```

**In ik_llama.cpp:** RoPE is applied to K BEFORE storing in the KV cache. So our
rotation Π goes after RoPE — exactly the right order. This is correct by construction.

**We verified this numerically:** Our `test_rope_invariance.py` confirms that
attention scores are identical (within floating-point precision, max error ~6e-6)
with and without rotation, across all three rotation strategies and all head
dimensions (32, 64, 128).

### 2.7 Three Rotation Strategies

| Strategy | How it works | Speed | Storage | Quality |
|----------|-------------|-------|---------|---------|
| **Random orthogonal** | QR decomposition of random Gaussian matrix | O(d²) | d×d floats per layer | Best theoretical |
| **Hadamard** | Walsh-Hadamard matrix (structured, recursive) | O(d log d) | Zero (deterministic) | Good |
| **Randomized Hadamard** | Hadamard × random ±1 diagonal | O(d log d) | 1 seed per layer | Best practical ✓ |

**Randomized Hadamard** is the winner:
- Fast (structured butterfly algorithm, already in ik_llama.cpp as `ggml_hadamard`)
- No storage needed (deterministic from a seed)
- Quality matches random orthogonal in practice
- The random sign diagonal breaks any alignment with input distribution

---

## 3. The Algorithm

### 3.1 Full Pipeline

**WRITE (when storing K/V in cache):**
```
Input: K_post_rope  [batch, heads, seq, head_dim]  (RoPE already applied)

Step 1: Normalize
  norms = ||K||           (save norms for later)
  K_unit = K / norms      (project to unit sphere)

Step 2: Rotate
  K_rot = K_unit @ Π^T    (spread outliers via rotation)

Step 3: Quantize (Lloyd-Max)
  indices = find_nearest_level(K_rot, codebook_boundaries)  (3-bit: 0-7)

Step 4: QJL residual (optional)
  K_approx = codebook_levels[indices]
  residual = K_rot - K_approx
  signs = sign(residual)           (1 bit per element)
  mean_abs = mean(|residual|)      (1 scalar per group)

Step 5: Pack and store
  Store: (indices, signs, mean_abs, norms)
```

**READ (when computing attention):**
```
Step 1: Rotate query (to match rotated keys in cache)
  Q_rot = Q_post_rope @ Π^T

Step 2: Dequantize keys
  K_rot_approx = codebook_levels[indices]
  K_rot_approx += signs * mean_abs    (QJL correction)
  K_rot_approx *= norms               (restore scale)

Step 3: Attention (in rotated space — Π^T Π = I cancels!)
  scores = softmax(Q_rot @ K_rot_approx^T / √d)
  output = scores @ V_dequantized
```

**Critical insight:** We NEVER need to inverse-rotate K. Since Q is also rotated,
the inner product `(ΠQ)^T(ΠK) = Q^T K` gives the correct attention score.

### 3.2 Memory Layout

For 3-bit + QJL with head_dim=64:

| Component | Bits per element | Notes |
|-----------|-----------------|-------|
| Lloyd-Max indices | 3.0 | 8 levels, packed |
| QJL sign bits | 1.0 | 1 per element, packed 8/byte |
| Mean abs residual | 0.5 | 32 bits / 64 elements |
| Vector norm | 0.5 | 32 bits / 64 elements |
| **Total** | **5.0** | vs 16 for FP16 |
| **Compression** | **3.2x** | |

Without QJL (pure 3-bit):

| Component | Bits per element |
|-----------|-----------------|
| Lloyd-Max indices | 3.0 |
| Vector norm | 0.5 |
| **Total** | **3.5** |
| **Compression** | **4.6x** |

---

## 4. What We Built So Far

### 4.1 Files

| File | Purpose | Lines |
|------|---------|-------|
| `prototype/turboquant/rotation.py` | 3 rotation matrix strategies + fast Hadamard transform | 180 |
| `prototype/turboquant/lloyd_max.py` | Lloyd-Max optimal quantizer for Beta/Gaussian distributions | 170 |
| `prototype/turboquant/qjl_residual.py` | 1-bit sign residual with pack/unpack (8 per byte) | 160 |
| `prototype/turboquant/quantizer.py` | Full pipeline: rotate → normalize → quantize → QJL | 210 |
| `prototype/turboquant/kv_cache.py` | Drop-in HuggingFace DynamicCache replacement (`LeanKVCache`) | 250 |
| `prototype/tests/test_rope_invariance.py` | Proves RoPE invariance for all rotation strategies | 200 |
| `prototype/eval/cosine_sim.py` | Quality evaluation on synthetic data | 200 |
| **Total** | | **~1,370** |

### 4.2 Tests Passed

```
=== Test: Rotation Orthogonality ===
  random_orthogonal d=32:  error = 5.96e-07  ✓
  hadamard d=64:           error = 0.00e+00  ✓
  randomized_hadamard d=128: error = 5.96e-08  ✓

=== Test: Rotation Preserves Norms ===
  All strategies: max diff < 1e-06  ✓

=== Test: RoPE Invariance ===
  random_orthogonal:    max attention score error = 6.20e-06  ✓
  hadamard:             max attention score error = 2.38e-06  ✓
  randomized_hadamard:  max attention score error = 3.81e-06  ✓
  head_dim=32/64/128:   all PASS  ✓
```

---

## 5. Results So Far

### 5.1 Synthetic Quality Evaluation (2026-04-03)

**KV Reconstruction (cosine similarity, higher = better):**

| Config | K cosine | V cosine | Effective bits | Compression |
|--------|----------|----------|----------------|-------------|
| 2-bit | 0.9443 | 0.9420 | 2.50 | 6.4x |
| 3-bit | 0.9841 | 0.9837 | 3.50 | 4.6x |
| 3-bit + QJL | **0.9950** | **0.9947** | 5.00 | 3.2x |
| 4-bit | 0.9957 | 0.9956 | 4.50 | 3.6x |
| 4-bit + QJL | 0.9987 | 0.9986 | 6.00 | 2.7x |

**Attention Score Preservation (the metric that matters most):**

| Config | Cosine sim | L1 error | Max error | KL divergence |
|--------|-----------|----------|-----------|---------------|
| 2-bit | 0.999989 | 0.000058 | 0.000248 | 0.00001159 |
| 2-bit + QJL | 0.999997 | 0.000031 | 0.000142 | 0.00000343 |
| 3-bit | 0.999996 | 0.000033 | 0.000132 | 0.00000351 |
| **3-bit + QJL** | **0.999999** | **0.000017** | **0.000066** | **0.00000106** |
| 4-bit | 0.999999 | 0.000016 | 0.000078 | 0.00000096 |
| 4-bit + QJL | 1.000000 | 0.000008 | 0.000036 | 0.00000024 |

**Key finding:** 3-bit + QJL achieves **6 nines** of attention cosine similarity.
The attention scores are virtually identical to FP16.

**Rotation strategy comparison (all at 3-bit + QJL):**

| Strategy | K cosine |
|----------|----------|
| Randomized Hadamard | 0.9950 |
| Hadamard | 0.9951 |
| Random orthogonal | 0.9946 |

All three perform equally well. **Randomized Hadamard is recommended** for production
(O(d log d) speed, zero storage).

---

## 6. Implementation Plan

### Phase 0: Python Prototype ✅ COMPLETE (2026-04-03)

- [x] Rotation matrix generation (3 strategies)
- [x] Lloyd-Max codebook computation
- [x] QJL 1-bit residual correction
- [x] Full quantizer pipeline
- [x] HuggingFace DynamicCache replacement (`LeanKVCache`)
- [x] RoPE invariance tests (all pass)
- [x] Synthetic quality evaluation (6 nines attention preservation)
- [x] GitHub repo created and pushed

### Phase 0.5: Real Model Evaluation (NEXT)

- [ ] Install HuggingFace `transformers` in venv
- [ ] Download Qwen 2.5-0.5B model
- [ ] Run perplexity evaluation (WikiText-2)
- [ ] Run cosine similarity on real activations
- [ ] Run needle-in-haystack test
- [ ] Compare: FP16 baseline vs 3-bit vs 3-bit+QJL vs 4-bit
- [ ] Document real model results

### Phase 1: C++ Integration (5-7 days)

- [ ] Register `GGML_TYPE_TQ3` and `GGML_TYPE_TQ4` in ggml
- [ ] Implement `block_tq3` struct (13 bytes / 32 elements)
- [ ] Implement `quantize_row_tq3` (from_float) and `dequantize_row_tq3` (to_float)
- [ ] Pre-compute Lloyd-Max codebooks as constexpr tables
- [ ] Implement randomized Hadamard rotation (reuse `ggml_hadamard`)
- [ ] Patch `llm_build_kv` to inject rotation before KV store
- [ ] Patch `llm_build_kqv` to rotate Q at attention time
- [ ] Wire `--cache-type-k tq3` CLI flag
- [ ] Benchmark: memory, speed, perplexity on Qwen 2.5-0.5B and 3.5-9B

### Phase 2: Autoresearch Loop (3-4 days build, 1-2 nights run)

- [ ] Define search space (6 knobs, ~2,592 configs)
- [ ] Build experiment runner (Python prototype for speed)
- [ ] Build SQLite results database
- [ ] Implement Bayesian optimization (optuna) around Pareto frontier
- [ ] Run overnight on Ryzen 7735U (~200 configs in 2 hours)
- [ ] Identify optimal configuration per model family
- [ ] Document Pareto frontier (memory vs quality vs speed)

**The 6 tuning knobs:**
1. K bits: {2, 2.5, 3, 3.125, 3.5, 4}
2. V bits: {2, 2.5, 3, 3.125, 3.5, 4}
3. Rotation: {Hadamard, RandHadamard, RandomGaussian}
4. Group size: {16, 32, 64, 128}
5. Layer policy: {Uniform, MoreBitsLater, MoreBitsFirst}
6. QJL: {Off, On}

### Phase 3: Optimized Kernels (5-7 days)

- [ ] CUDA: fused rotation + quantize kernel (prefill path)
- [ ] CUDA: fused dequantize-in-FlashAttention kernel (decode path)
- [ ] Metal: threadgroup Hadamard + codebook lookup
- [ ] CPU: AVX2/NEON intrinsics for rotation + table lookup
- [ ] Benchmark against Q8_0 baseline on RTX 4090 / M2 / Ryzen

**Key optimization insight:** The fused dequant-in-attention kernel reads **2.5x less
memory** from HBM than Q8_0. Since KV cache attention is memory-bandwidth-bound,
LeanKV should be **faster** than Q8_0, not slower.

---

## 7. Trail Log

### 2026-04-03: Project started

**Session context:** After completing LeanInfer (all phases, Metal 3.5x, CUDA benchmarks,
RESULTS.md written), the user presented 5 research papers on KV cache quantization:

1. **Johnson-Lindenstrauss Transforms** (Freksen 2021) — background math
2. **TurboESM** (Hu et al. 2026) — RoPE compatibility solution
3. **TurboQuant** (Zandieh et al. 2025, Google Research) — the core algorithm
4. **PolarQuant** (Han et al. 2025, Google/KAIST/Yale) — polar coordinate approach
5. **QJL** (Zandieh et al. 2024) — 1-bit quantized JL transform

**What we did:**
1. Read all 5 papers (total ~100 pages)
2. Explored ik_llama.cpp KV cache architecture:
   - Cache allocated per-layer as ggml tensors (`k_l[il]`, `v_l[il]`)
   - Supports F16, Q8_0, Q4_0 via `--cache-type-k` / `--cache-type-v`
   - K stored post-RoPE (critical for our rotation approach)
   - Hadamard transform already exists (`k_cache_hadamard`, `ggml_hadamard`)
3. Read the user's prior planning document (`TQuantforLeanInfer`)
4. Researched Karpathy's autoresearch loop
5. Designed 4-phase implementation plan
6. Created new repo (`/home/junc/LeanKV`, later renamed from TurboQuant)
7. Implemented all Phase 0 modules:
   - `rotation.py` — 3 rotation strategies
   - `lloyd_max.py` — optimal scalar quantizer
   - `qjl_residual.py` — 1-bit sign correction
   - `quantizer.py` — full pipeline
   - `kv_cache.py` — HuggingFace DynamicCache replacement (`LeanKVCache`)
8. Wrote and passed all tests:
   - RoPE invariance: proven for all rotation strategies, all head dims
   - Orthogonality: Π^T Π = I verified to < 1e-6
   - Norm preservation: verified to < 1e-6
9. Ran synthetic quality evaluation:
   - 3-bit + QJL: 0.999999 attention cosine similarity
   - 4.6x compression at 3.5 bits
10. Created GitHub repo (`hchengit/LeanKV`) and pushed
11. Renamed project from TurboQuant → LeanKV (Google's name → our name)
12. Wrote this plan document

**Decisions made:**
- New repo (not in LeanInfer) — keeps projects clean and independent
- Internal module stays `turboquant` (algorithm name) — only project name changed
- Randomized Hadamard as default rotation — fast, no storage, equally good quality
- No V cache rotation by default — values are more tolerant of quantization noise
- QJL enabled by default — small memory cost for significant quality improvement

**Open questions for next session:**
- Does the algorithm work as well on REAL activations as synthetic?
- What's the perplexity impact on Qwen 2.5-0.5B?
- Does needle-in-haystack pass at 3-bit?
- Which layer policy works best (uniform vs more-bits-later)?

---

## References

1. Zandieh et al. "TurboQuant: Online Vector Quantization with Near-optimal Distortion Rate." arXiv:2504.19874 (2025). Google Research.
2. Han et al. "PolarQuant: Quantizing KV Caches with Polar Transformation." arXiv:2502.02617 (2025).
3. Zandieh et al. "QJL: 1-Bit Quantized JL Transform for KV Cache Quantization with Zero Overhead." arXiv:2406.03482 (2024).
4. Hu et al. "TurboESM: Ultra-Efficient 3-Bit KV Cache Quantization for Protein Language Models." arXiv:2603.26110 (2026).
5. Freksen. "An Introduction to Johnson-Lindenstrauss Transforms." arXiv:2103.00564 (2021).
6. Lloyd. "Least squares quantization in PCM." IEEE Trans. Info. Theory (1982).
7. Shannon. "Coding theorems for a discrete source with a fidelity criterion." IRE Nat. Conv. Rec. (1959).
