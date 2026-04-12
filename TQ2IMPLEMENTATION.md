# TQ2 Implementation — Complete Plan, Status, and Remaining Work

## Project Overview

TurboQuant KV cache quantization for LeanKV and Lean_llama.cpp: three
precision tiers (TQ2/TQ3/TQ4), outlier channel treatment, and tiered
KV cache architecture for near-million-token contexts.

**Branch:** `feature/tq2-outlier-tiered` (both repos)
**Repos:** `hchengit/LeanKV`, `hchengit/Lean_llama.cpp`

---

## Phase 1: TQ2_0 Core (LeanKV Standalone) — COMPLETE

2-bit Lloyd-Max optimal scalar quantization at 2.5 bits/elem.

### What was implemented

- [x] Compute 2-bit Lloyd-Max codebook for N(0,1) distribution
  - `TQ2_LEVELS[4] = {-1.0, -0.2998013, +0.2998013, +1.0}`
  - `TQ2_BOUNDARIES[3] = {-0.6499007, 0.0, +0.6499007}`
- [x] `block_tq2_0` struct: `{ ggml_half d; uint8_t qs[8]; }` = 10 bytes / 32 elem = 2.5 bits/elem
- [x] 2-bit packing: 4 values per byte, `byte = (i0) | (i1<<2) | (i2<<4) | (i3<<6)`
- [x] `pack_2bit_32()` / `unpack_2bit_32()` — pack/unpack 32 indices into 8 bytes
- [x] `find_nearest_tq2()` — 3 boundary comparisons
- [x] `quantize_row_tq2_0_ref()` — find amax, normalize, nearest-level, pack
- [x] `dequantize_row_tq2_0()` — unpack, lookup `TQ2_LEVELS[idx] * d`
- [x] `requantize_tq4_to_tq3()` — dequant TQ4 blocks, requant as TQ3
- [x] `requantize_tq3_to_tq2()` — dequant TQ3 blocks, requant as TQ2
- [x] `requantize_tq4_to_tq2()` — dequant TQ4 blocks, requant as TQ2
- [x] 8 new unit tests (tests 10-17)
- [x] Makefile updated

### Files modified (LeanKV)

| File | Change |
|------|--------|
| `src/ggml-tq.h` | Added TQ2 codebook, `block_tq2_0` struct, function declarations, requantize declarations |
| `src/ggml-tq.c` | Added TQ2 pack/unpack, quantize/dequantize, requantize functions |
| `src/test-tq.c` | Added 8 TQ2 tests (10-17) |
| `src/Makefile` | Build targets for TQ2 |

### Commit

```
52a1b5e feat: add TQ2_0 2-bit Lloyd-Max quantization (2.5 bits/elem)
```

---

## Phase 2: TQ2_0 Integration (Lean_llama.cpp) — COMPLETE

Full integration into the llama.cpp inference stack with optimized SIMD
kernels for both AVX2 (x86) and NEON (ARM).

### What was implemented

- [x] **Type registration:** `GGML_TYPE_TQ2_0 = 44` in ggml.h enum
- [x] **Block struct:** `block_tq2_0` in ggml-common.h with `static_assert(sizeof == 10)`
- [x] **Int8 codebook LUT:** `tq2_values[16] = {-127, -38, 38, 127, 0,...}` for SIMD lookup
- [x] **Type traits:** blck_size=32, type_size=10, to_float, from_float, vec_dot in ggml.c
- [x] **Scalar reference:** `quantize_row_tq2_0_ref`, `dequantize_row_tq2_0` in ggml-tq.c
- [x] **Vec dot:** `ggml_vec_dot_tq2_0_q8_0` with ARM NEON and AVX2 SIMD paths
- [x] **IQK AVX2 kernel:** `TQ2_0_DequantizerS` — PSHUFB codebook lookup, mask 0x03 + shifts
- [x] **IQK AVX2 unpacker:** `TQ2_0_UnpackerS` with `ScaleHelperTQ2_0_S` (d/127.0f scale)
- [x] **IQK NEON kernel:** `DequantizerTQ2_0` — VTBL codebook lookup variant
- [x] **Flash attention helper:** `HelperTQ20` in iqk_fa_templates.h
  - 2-bit unpack logic for both NEON and AVX2/AVX512
  - Registered in V-cache and K-cache dispatch switches
- [x] **FA type support:** TQ2_0 added to both `supported_kv_types` sets in iqk_flash_attn.cpp
- [x] **Mul-mat dispatch:** TQ2_0 added to legacy quants in iqk_mul_mat.cpp
- [x] **CLI:** `"tq2_0"` in `kv_cache_type_from_str()` in common.cpp
- [x] **Auto-Hadamard:** TQ2_0 added to auto-enable conditions alongside TQ3/TQ4

### Files modified (Lean_llama.cpp)

| File | Change |
|------|--------|
| `ggml/include/ggml.h` | Added `GGML_TYPE_TQ2_0 = 44` |
| `ggml/src/ggml-common.h` | Added `block_tq2_0` struct, `tq2_values` int8 LUT |
| `ggml/src/ggml.c` | Added TQ2_0 type traits entry |
| `ggml/src/ggml-quants.h` | Added TQ2 function declarations |
| `ggml/src/ggml-tq.c` | Added TQ2 codebook, pack/unpack, quantize/dequantize, vec_dot (NEON+AVX2) |
| `ggml/src/iqk/iqk_mul_mat.cpp` | Added TQ2_0 to legacy quants dispatch |
| `ggml/src/iqk/iqk_gemm_legacy_quants.cpp` | Added TQ2_0_DequantizerS (AVX2), NEON dequantizer, all dispatch points |
| `ggml/src/iqk/fa/iqk_fa_templates.h` | Added HelperTQ20 FA class, V-cache and K-cache dispatch |
| `ggml/src/iqk/iqk_flash_attn.cpp` | Added TQ2_0 to supported_kv_types |
| `common/common.cpp` | Added `"tq2_0"` parsing + TQ2_0 auto-Hadamard |

### Commit

```
dbf40a36 feat: integrate TQ2_0 (2-bit KV cache) into Lean_llama.cpp
```

---

## Phase 3: Outlier Channel Treatment — COMPLETE

Mixed-precision quantization: outlier channels (high-variance) get more
bits, normal channels get fewer. Channel permutation makes both regions
contiguous for SIMD-friendly block quantization.

### Phase 3.1-3.3: Standalone Implementation (LeanKV)

- [x] `tq_tier_t` enum: `TQ_TIER_TQ2`, `TQ_TIER_TQ3`, `TQ_TIER_TQ4`
- [x] `tq_outlier_config` struct: head_dim, n_outlier, n_normal, outlier_frac, tiers, perm[256], inv_perm[256], channel_var[256]
- [x] `tq_identify_outliers()` — per-channel variance, argsort descending, round n_outlier to block-aligned multiple of 32, build perm/inv_perm
- [x] `tq_outlier_config_init_uniform()` — identity permutation, no outlier split
- [x] `tq_mixed_buffer_sizes()` — compute bytes needed for outlier and normal buffers
- [x] `tq_mixed_effective_bpe()` — weighted average bits per element
- [x] `tq_permute_channels()` — `y[i] = x[perm[i]]` (gather from perm positions)
- [x] `tq_unpermute_channels()` — `y[i] = x[inv_perm[i]]` (gather from inv_perm positions)
- [x] `tq_mixed_quantize()` — permute, quantize outlier channels with high tier, quantize normal with low tier
- [x] `tq_mixed_dequantize()` — dequantize both tiers, unpermute back to original order
- [x] `tier_quantize()` / `tier_dequantize()` — dispatch to TQ2/TQ3/TQ4 by tier
- [x] Bug fix: `tq_unpermute_channels` was doing scatter (`y[inv_perm[i]] = x[i]`), corrected to gather (`y[i] = x[inv_perm[i]]`)
- [x] 6 new unit tests (tests 18-23):
  - Test 18: Outlier detection — permutation validity, inverse correctness, variance ordering
  - Test 19: Mixed-precision quality — TQ3+TQ2 vs uniform TQ2 (95.4% lower MSE)
  - Test 20: Multi-sample MSE — 100-sample average (95.5% improvement)
  - Test 21: Effective bits — 2.75, 3.75, 3.00 bpe calculations and buffer sizes
  - Test 22: Hadamard + outlier pipeline — full end-to-end (cosine 0.957)
  - Test 23: Attention score preservation — mixed vs uniform (0.972 vs 0.954)
- [x] Makefile updated to include `ggml-tq-outlier.c`
- [x] **67/67 tests pass**

### Files created/modified (LeanKV)

| File | Change |
|------|--------|
| `src/ggml-tq-outlier.h` | **NEW** — outlier config struct, tier enum, API declarations |
| `src/ggml-tq-outlier.c` | **NEW** — detection, permutation, mixed-precision quantize/dequantize |
| `src/test-tq.c` | Added 6 outlier tests (18-23), updated main() |
| `src/Makefile` | Added ggml-tq-outlier.c to build |

### Commit

```
3eb3133 feat: add outlier channel treatment for mixed-precision TQ quantization
```

### Phase 3.4: Integration into Lean_llama.cpp

- [x] Copy `ggml-tq-outlier.h` and `ggml-tq-outlier.c` to `ggml/src/` (adapted includes: `ggml-common.h` instead of `ggml-tq.h`, added `ggml-quants.h`)
- [x] `--kv-outlier-frac N` CLI parameter (float, default 0.0 = disabled)
- [x] `kv_outlier_frac` field in `gpt_params`, `llama_context_params`, `llama_cparams`
- [x] `tq_channel_perm_data` struct in `llama-context.h` — per-layer permutation table
- [x] `outlier_perm_k` vector in `llama_kv_cache` — per-layer storage
- [x] **Weight-based calibration** at model load time:
  - Accesses W_K weight tensor per layer via `ggml_backend_tensor_get`
  - Dequantizes to FP32 using `ggml_internal_get_type_traits(type).to_float`
  - Computes per-head-channel row L2 norms (averaged across heads)
  - Feeds to `tq_identify_outliers()` as variance proxy
  - Zero runtime calibration cost — computed once at init
- [x] **Channel permutation custom op** (`tq_channel_perm_op`):
  - Registered via `ggml_map_custom1` in the attention graph
  - Applied to both K and Q after Hadamard rotation
  - Reorders elements along dimension 0 (head_dim) per permutation table
  - Single-threaded (`n_tasks=1`) for simplicity
  - K and Q get identical permutation — dot products preserved exactly
- [x] Logging: `kv_outlier = 0.25` and `outlier K: 32/64 channels` at init
- [x] CMakeLists.txt updated
- [x] **Build succeeds clean, tested on Qwen 0.5B and 9B models**

### Files created/modified (Lean_llama.cpp)

| File | Change |
|------|--------|
| `ggml/src/ggml-tq-outlier.h` | **NEW** — adapted from LeanKV (include ggml-common.h) |
| `ggml/src/ggml-tq-outlier.c` | **NEW** — adapted from LeanKV (include ggml-quants.h) |
| `ggml/src/CMakeLists.txt` | Added ggml-tq-outlier.c/.h to build |
| `common/common.h` | Added `float kv_outlier_frac` to gpt_params |
| `common/common.cpp` | Parse `--kv-outlier-frac`, pass to cparams, help text |
| `include/llama.h` | Added `float kv_outlier_frac` to llama_context_params |
| `src/llama-cparams.h` | Added `float kv_outlier_frac` |
| `src/llama-context.h` | Added `tq_channel_perm_data` struct, `outlier_perm_k` vector |
| `src/llama-build-context.cpp` | Added `tq_channel_perm_op`, wired into `llm_build_kv` after Hadamard |
| `src/llama.cpp` | Include outlier header, default param, cparams copy, weight-based calibration at init, logging |

### Commit

```
a7268a27 feat: integrate outlier channel treatment for mixed-precision KV cache
```

---

## Phase 4: Tiered KV Cache — DESIGN COMPLETE, IMPLEMENTATION FUTURE

Automatic hot/warm/cold token tiers with migration between precision
levels. Design documented; requantize primitives implemented.

### What was implemented

- [x] `requantize_tq4_to_tq3()` — TQ4 block to TQ3 block (via FP32)
- [x] `requantize_tq3_to_tq2()` — TQ3 block to TQ2 block (via FP32)
- [x] `requantize_tq4_to_tq2()` — TQ4 block to TQ2 block (via FP32)
- [x] Architecture design documented in `TIERED_KV_CACHE.md`
- [x] Three implementation options defined (user-selected / split tensors / single buffer with metadata)

### What remains

- [ ] **Auto-migration logic:** Track per-token age, trigger requantize when threshold exceeded
- [ ] **Split tensor KV cache:** Separate tensors per tier per layer, or per-token type metadata
- [ ] **FA dispatch per tier:** Flash attention kernel reads from mixed-type cache
- [ ] **CLI:** `--kv-tier-hot N --kv-tier-warm M` or similar age thresholds
- [ ] **Memory management:** Compact cold tiers, handle defragmentation across tiers

### Commit

```
4607cb6 docs: add tiered KV cache architecture design (Phase 4)
```

---

## Current User-Facing Capabilities

### Three manually selectable KV cache tiers

```bash
# Near-lossless (4.5 bits/elem, 3.6x compression)
llama-cli -ctk tq4_0 -ctv f16 ...

# Balanced (3.5 bits/elem, 4.6x compression)
llama-cli -ctk tq3_0 -ctv f16 ...

# Maximum compression (2.5 bits/elem, 6.4x compression)
llama-cli -ctk tq2_0 -ctv f16 ...
```

### Outlier channel permutation (improves any tier)

```bash
# TQ3 with 25% outlier channels getting better block alignment
llama-cli -ctk tq3_0 -ctv f16 --kv-outlier-frac 0.25 ...
```

### All configs auto-enable

- Hadamard rotation (distributes outlier energy, makes distribution Gaussian)
- IQK flash attention SIMD kernels (AVX2 on x86, NEON on ARM)
- Lloyd-Max optimal codebook lookup (branchless, table-based)

---

## Test Results Summary

### LeanKV standalone: 67/67 tests pass

| Test | Description | Result |
|------|-------------|--------|
| 1 | 3-bit pack/unpack roundtrip | PASS |
| 2 | TQ3_0 quantize/dequantize quality (cosine 0.985) | PASS |
| 3 | TQ4_0 quantize/dequantize quality (cosine 0.997) | PASS |
| 4 | Hadamard transform orthogonality | PASS |
| 5 | Randomized Hadamard norm preservation | PASS |
| 6 | Full pipeline Hadamard + TQ3 (cosine 0.989) | PASS |
| 7 | TQ3 Lloyd-Max vs uniform 3-bit (17.4% lower MSE) | PASS |
| 8 | Codebook symmetry validation | PASS |
| 9 | Attention score preservation (cosine 0.992) | PASS |
| 10 | 2-bit pack/unpack roundtrip | PASS |
| 11 | TQ2_0 quantize/dequantize quality (cosine 0.921) | PASS |
| 12 | TQ2 codebook symmetry | PASS |
| 13 | Full pipeline Hadamard + TQ2 (cosine 0.942) | PASS |
| 14 | TQ2 attention preservation (cosine 0.933) | PASS |
| 15 | TQ2 Lloyd-Max vs uniform 2-bit (10.3% lower MSE) | PASS |
| 16 | Requantize TQ4 to TQ3 to TQ2 chain | PASS |
| 17 | TQ2 memory layout verification (10 bytes / 32 elem) | PASS |
| 18 | Outlier detection and permutation validity | PASS |
| 19 | Mixed-precision quality (95.4% lower MSE than uniform TQ2) | PASS |
| 20 | Multi-sample MSE (95.5% improvement over 100 samples) | PASS |
| 21 | Effective bits (2.75, 3.75, 3.00 verified) | PASS |
| 22 | Hadamard + outlier pipeline (cosine 0.957) | PASS |
| 23 | Attention preservation mixed vs uniform (0.972 vs 0.954) | PASS |

### Lean_llama.cpp: builds clean, tested on 0.5B and 9B models

- TQ4_0: coherent output on both models
- TQ3_0: coherent output on 9B, degraded on 0.5B (expected)
- TQ2_0: coherent output on 9B, degraded on 0.5B (expected)
- Outlier permutation: produces different but coherent output
- KV cache compression verified (e.g., TQ3: 0.66 MiB vs F16: 3.00 MiB)

---

## Remaining Work Checklist

### Immediate: CPU Perplexity Validation (Ryzen)

- [ ] Build on Ryzen from `feature/tq2-outlier-tiered`
- [ ] Download `wiki.test.raw` (WikiText-2 test set)
- [ ] Run `./docs/run-tq-tests.sh` with Qwen3.5-9B model
- [ ] Record PPL for all 6 configs (F16, TQ4, TQ3, TQ2, TQ3+outlier, TQ2+outlier)
- [ ] Record KV cache memory sizes
- [ ] Record speed (tok/s) — confirm AVX2 kernels active
- [ ] Verify no configs produce garbage output on 9B model
- [ ] Test V-cache quantization (TQ4+TQ4, TQ3+TQ3)

### Next: CUDA Kernels

- [ ] `ggml/src/ggml-cuda/dequantize.cuh` — Add `dequantize_block_tq2_0` kernel
- [ ] `ggml/src/ggml-cuda/common.cuh` — Add `block_tq2_0` device struct if needed
- [ ] `ggml/src/ggml-cuda/vecdotq.cuh` — Add `vec_dot_tq2_0_q8_0` CUDA kernel
- [ ] `ggml/src/ggml-cuda/mmvq.cu` — Add TQ2_0 to mul_mat_vec dispatch
- [ ] `ggml/src/ggml-cuda/template-instances/` — Generate TQ2 template instances
- [ ] Test on NVIDIA GPU (RTX 4090 or similar)
- [ ] Repeat for TQ3_0 and TQ4_0 if not already present

### Next: Metal Kernels

- [ ] `ggml/src/ggml-metal/ggml-metal.metal` — Add TQ2 dequantize kernel template
- [ ] Register `host_name` for TQ2_0 in FA dispatch
- [ ] Add TQ2 to Metal kernel library compilation
- [ ] Test on Apple Silicon (M1/M2/M3)

### Future: Auto-Migration (Tiered KV Cache)

- [ ] Implement per-token age tracking in `llama_kv_cell`
- [ ] Add tier migration trigger (age threshold check after each decode)
- [ ] Implement in-place requantization of cache rows
- [ ] Handle mixed-type reads in FA (either split tensors or type metadata)
- [ ] CLI: `--kv-tier-policy` or similar configuration
- [ ] Benchmark memory savings at 32K/64K/128K contexts

### Future: Mixed-Precision FA Kernels

- [ ] FA kernel that reads outlier channels as TQ3 and normal channels as TQ2
- [ ] Requires either new GGML type or split tensor approach
- [ ] Would unlock 2.75 effective bits/elem (vs 2.5 for uniform TQ2)

### Future: V-Cache Outlier Treatment

- [ ] Apply channel permutation to V-cache (not just K)
- [ ] Requires unpermuting attention output after FA
- [ ] Add `ggml_map_custom1` unpermute op after `ggml_flash_attn_ext`

---

## Architecture Quick Reference

### Block Layout (all types: 32 elements per block)

```
TQ2_0:  [d:f16][qs:8B]  = 10 bytes = 2.5 bits/elem
TQ3_0:  [d:f16][qs:12B] = 14 bytes = 3.5 bits/elem
TQ4_0:  [d:f16][qs:16B] = 18 bytes = 4.5 bits/elem
```

### Quantization Pipeline

```
Input FP32 → Hadamard rotation → Channel permutation (if outlier enabled)
→ Find block max (d = amax) → Normalize (x/d) → Nearest codebook level
→ Pack indices → Store block [d, qs...]
```

### Dequantization Pipeline (IQK FA)

```
Load block [d, qs...] → Unpack indices (shift+mask)
→ PSHUFB/VTBL codebook lookup → Scale by d/127 → FP32 output
```

### Key Design Decisions

1. **Lloyd-Max codebooks** — optimal for post-Hadamard Gaussian distribution
2. **Hadamard rotation** — spreads outlier energy, auto-enabled for all TQ types
3. **Channel permutation** — groups similar-variance channels for better per-block scale
4. **Int8 codebook LUT** — enables branchless SIMD lookup (PSHUFB on x86, VTBL on ARM)
5. **32-element blocks** — matches SIMD register widths (256-bit AVX2 = 32 bytes)
6. **Same codebook for GPU** — `tq2_values`/`tq3_values`/`tq4_values` in ggml-common.h usable as device constant memory

---

## Commit History

### LeanKV

```
4607cb6 docs: add tiered KV cache architecture design (Phase 4)
3eb3133 feat: add outlier channel treatment for mixed-precision TQ quantization
52a1b5e feat: add TQ2_0 2-bit Lloyd-Max quantization (2.5 bits/elem)
```

### Lean_llama.cpp

```
f439e14c docs: add TQ KV cache test procedures and automated test runner
a7268a27 feat: integrate outlier channel treatment for mixed-precision KV cache
dbf40a36 feat: integrate TQ2_0 (2-bit KV cache) into Lean_llama.cpp
```
