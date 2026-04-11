# TQ3 Plan: Make TQ3 Usable for Dense Models (Without QJL)

## Context

TQ4_0 works great on ARM (PPL delta ~+0.02 from F16 -- essentially lossless). Now we want to push to **3-bit** (TQ3) for even more KV cache compression. The problem: 3-bit has only 8 codebook levels vs 16 for 4-bit, so quantization error roughly doubles. QJL (random projection) could help but requires head_dim >= 512 -- most dense models use 64-128, so it's ruled out.

**Goal:** Make TQ3 quality acceptable (PPL delta < 0.3 from F16) while keeping decode speed within 15% of TQ4.

**Strategy:** Prioritize improvements that cost NOTHING at decode time (encode-only), then add at most ONE small runtime addition if needed. Measure after each phase and stop when quality is acceptable.

**Platform:** AVX2/x86 first (faster iteration), then port to ARM/NEON.

---

## What TQ3 Looks Like Today

- **Block:** 32 values -> 14 bytes (2-byte fp16 scale + 12 bytes packed 3-bit indices)
- **Codebook:** 8 Lloyd-Max levels: {-127, -79, -45, -14, +14, +45, +79, +127} (int8)
- **Memory savings vs TQ4:** 22% (14 vs 18 bytes/block)
- **IQK support:** NONE -- falls back to slow generic path
- **Key files (in Lean_llama.cpp):**
  - Block struct: `ggml/src/ggml-common.h:234`
  - Codebook + quantize/dequant: `ggml/src/ggml-tq.c:51-186`
  - Type traits: `ggml/src/ggml.c:872-884`

---

## Phase 0: IQK Flash Attention Kernel for TQ3 (Speed Foundation)

- [ ] Complete

**Why first:** Without this, TQ3 uses the generic `to_float()` fallback path, which is far slower than TQ4's IQK path. No quality improvement matters if TQ3 is too slow to use.

**What:** Build a `HelperTQ30` struct for the IQK FA inner loop, modeled on `HelperTQ40`.

**The 3-bit unpacking challenge:** TQ4 packs 2 values per byte (nibbles -- trivial mask+shift). TQ3 packs 8 values into 3 bytes, requiring ~4-5 SIMD instructions to unpack vs TQ4's 2-3. Expected decode overhead: ~10-15% slower than TQ4.

**Files to modify:**
- `ggml/src/iqk/fa/iqk_fa_templates.h` -- Add `HelperTQ30` with SIMD 3-bit unpack + LUT lookup + scale
- `ggml/src/ggml-common.h` -- Add `tq3_values` LUT table (8 int8 values + 8 zero padding)
- `ggml/src/iqk/iqk_flash_attn.cpp` -- Add `GGML_TYPE_TQ3_0` to `supported_kv_types()`

**Expected result:** TQ3 decode speed ~85-90% of TQ4.

**Verification:**
- Build and run `llama-cli -m <model> -ngl 0 -ctk tq3_0 -ctv tq3_0 -c 2048 -p "The capital of France is" -n 16`
- Coherent output + reasonable tok/s

---

## Phase 1: Smarter Quantization (Zero Decode Cost)

- [ ] Complete

The biggest quality win. Only the ENCODE path changes -- block format and dequantization are untouched.

### 1a: Optimal Scale Computation

Current code uses `d = max|x|` as the block scale. Instead, after assigning codebook indices, compute the least-squares optimal scale:

```
d_opt = sum(x[j] * Level[idx[j]]) / sum(Level[idx[j]]^2)
```

This minimizes block MSE. One pass over 32 elements, negligible cost.

### 1b: Coordinate Descent Rounding

After initial nearest-level assignment, refine by trying adjacent codebook levels for each element. Keep changes that reduce total block MSE. 2-3 passes over the block:

```
for each pass (2-3x):
    for each element j in block:
        try index[j] +/- 1
        recompute optimal scale d
        keep if block MSE decreases
```

**Why it helps:** Nearest-level per-element is NOT the global optimum when all 32 elements share one scale. Moving one element to a "worse" individual level can improve the whole block.

**Encode cost:** ~3x slower quantization per block (microseconds -- negligible vs attention cost).

**File:** `ggml/src/ggml-tq.c`, function `quantize_row_tq3_0_ref()` (lines 138-168)

**Expected impact:** ~0.3-0.5 PPL improvement (the single biggest gain).

### Karpathy Loop (Python prototype first)

**Script:** `scripts/tq3_rounding.py`

1. **Data collection:** Run model once, dump post-Hadamard KV values per layer to `.npy` files
2. **Iterate on rounding strategy:**
   - Number of coordinate descent passes (2 vs 3 vs 5)
   - Search scope (adjacent levels only vs all 8 levels)
   - Element ordering (sequential vs worst-error-first)
   - Scale formula (least-squares vs max|x|)
3. **Measure:** Block MSE, angular error (cosine similarity), simulated attention score error
4. **Output:** Best strategy + expected quality numbers -> port winning approach to C++

**Verification:**
- Quick PPL (3 chunks): compare TQ3 before/after rounding improvement
- Full PPL (145 chunks) once strategy is finalized

---

## Phase 2: Per-Layer Calibrated Codebooks (Zero Decode Cost)

- [ ] Complete

**Only if Phase 1 doesn't hit PPL delta < 0.3.**

**Idea:** Different layers may have different value distributions even after Hadamard rotation. Instead of one global 8-level codebook, optimize 8 levels per layer on calibration data.

**Key insight:** The block format doesn't change. 3-bit indices still mean "level 0-7." Only the float values those indices map to are different per layer. In the IQK kernel, this is just loading a different 16-byte lookup table -- same PSHUFB instruction, zero extra cost.

**Implementation:**
- Offline calibration script: run small dataset through model, collect post-Hadamard value distributions per layer, run Lloyd-Max per layer
- Store 8 x fp16 per layer as metadata (negligible: 16 bytes x n_layers)
- Quantizer uses per-layer codebook + boundaries
- Dequantizer loads per-layer LUT instead of global

**Files:**
- `ggml/src/ggml-tq.c` -- Add codebook-parameterized quantize/dequant variants
- `ggml/src/iqk/fa/iqk_fa_templates.h` -- `HelperTQ30` takes per-layer LUT pointer

### Karpathy Loop (Python prototype first)

**Script:** `scripts/tq3_codebooks.py`

1. Analyze per-layer value distributions from the dumped data
2. Run Lloyd-Max per layer, compare to global codebook
3. **Key question to answer:** Is inter-layer variance large enough to matter? If all layers look similar after Hadamard, skip Phase 2 entirely.
4. **Output:** Per-layer codebook tables (or "not worth it" conclusion)

**Expected impact:** ~0.1-0.2 additional PPL improvement.

---

## Phase 3: Per-Block Bias Byte -- "TQ3_1" (Small Decode Cost)

- [ ] Complete

**Only if Phases 1-2 aren't enough.** Adds 1 byte per block for mean-error correction.

- New block: 15 bytes (still 17% smaller than TQ4's 18 bytes)
- During quantize: compute mean reconstruction error, store as int8 bias
- During dequant: add `bias * d / 128` to all elements (one SIMD add per block)
- Decode overhead: ~3-5%

This requires registering a new GGML type (`GGML_TYPE_TQ3_1`), which is more integration work.

---

## What We're NOT Doing (and Why)

| Technique | Why skip |
|-----------|----------|
| **Low-rank correction** | Extra matmul in attention inner loop -> slower than TQ4 |
| **Residual quantization** | 3-bit + residual ~ 4-bit storage, defeats the point |
| **QJL / random projection** | Needs head_dim >= 512, dense models have 64-128 |
| **Orthogonal projection at decode** | Iterative algorithm in inner loop, too slow |
| **Error diffusion across blocks** | Creates sequential dependency, breaks SIMD parallelism |

---

## Expected Quality Progression

| After Phase | Est. PPL delta from F16 | Decode overhead vs TQ4 |
|-------------|------------------------|----------------------|
| Current TQ3 (generic path) | ~0.8-1.2 | Much slower (no IQK) |
| Phase 0 (IQK kernel) | ~0.8-1.2 | ~10-15% slower |
| Phase 1 (optimal rounding) | ~0.4-0.7 | Same as Phase 0 (encode-only) |
| Phase 2 (per-layer codebooks) | ~0.3-0.6 | Same (zero decode cost) |
| Phase 3 (bias byte / TQ3_1) | ~0.25-0.5 | ~13-18% slower |

---

## Implementation Order

1. **Phase 0** -- IQK kernel (AVX2 first, then NEON). Prerequisite for everything.
2. **Phase 1** -- Optimal rounding (Python prototype -> C++). Biggest quality gain, zero decode cost.
3. **Benchmark** TQ3 PPL on Qwen3.5-9B (AVX2). If delta < 0.3, port to ARM and stop.
4. **Phase 2** -- Per-layer codebooks. Only if Phase 1 isn't enough.
5. **Phase 3** -- TQ3_1 with bias byte. Last resort (unlikely needed for < 0.3 target).
