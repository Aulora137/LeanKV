# Tiered KV Cache — Architecture Design

## Status: Design Hooks Only (Phase 4)

The requantize primitives (`requantize_tq4_to_tq3`, `requantize_tq3_to_tq2`, `requantize_tq4_to_tq2`) are implemented in `ggml-tq.c`. Full tiered migration is future work.

## Concept

```
Hot tier:   FP16 or TQ4_0   recent N tokens (e.g., 512)     — near-lossless
Warm tier:  TQ3_0            medium-age tokens                �� 3.5 bits/elem
Cold tier:  TQ2_0 + outlier  oldest tokens                    — 2.5–2.75 bits/elem
```

Migration trigger: when a token's age exceeds a threshold, requantize its KV data to the next lower tier. This happens in-place using the `requantize_*` functions.

## Effective Compression

| Tier | Bits/elem | Compression vs FP16 | Quality (cosine) |
|------|-----------|---------------------|-------------------|
| FP16 | 16.0 | 1.0x | 1.000 |
| TQ4_0 | 4.5 | 3.6x | 0.997 |
| TQ3_0 | 3.5 | 4.6x | 0.985 |
| TQ2_0 | 2.5 | 6.4x | 0.921 |
| TQ3+TQ2 (outlier) | 2.75 | 5.8x | 0.982 |

## Available Primitives

All primitives are implemented and tested in LeanKV standalone:

1. **Quantize**: `quantize_row_tq{2,3,4}_0_ref()` — FP32 → TQ block
2. **Dequantize**: `dequantize_row_tq{2,3,4}_0()` — TQ block → FP32
3. **Requantize**: `requantize_tq{4→3, 3→2, 4→2}()` — TQ block → TQ block (via FP32 intermediate)
4. **Outlier detection**: `tq_identify_outliers()` — per-channel variance analysis
5. **Mixed-precision**: `tq_mixed_quantize/dequantize()` — split outlier + normal channels
6. **Hadamard transform**: `hadamard_transform()` — orthogonal rotation for outlier spread

## Implementation Options

### Option A: User-selected cache type (SHIPPED)
Users manually choose `--cache-type-k tq2_0` for maximum compression.
No automatic migration. Simplest to implement and maintain.

### Option B: Separate tensors per tier (future)
Each layer has separate K/V tensors for each tier. FA dispatches per tier.
Cleanest architecture but requires significant KV cache restructuring.

### Option C: Single buffer with type metadata (future)
Single tensor per layer with per-token type tags. FA kernel checks type.
Complex kernel logic but no data movement during migration.

## Integration Notes

- The outlier channel permutation is now integrated (`--kv-outlier-frac`)
- Channel permutation tables are fixed per layer (computed at model load from W_K norms)
- Both K and Q are permuted identically → dot products preserved exactly
- V permutation (for V-cache outlier treatment) is deferred — would need output unpermute
- Requantize functions work on aligned 32-element blocks — compatible with SIMD

## Test Coverage

67/67 tests pass in LeanKV standalone:
- TQ2/TQ3/TQ4 quantize/dequantize quality
- Requantize chain (TQ4→TQ3→TQ2)
- Outlier detection and permutation validity
- Mixed-precision vs uniform quality comparison
- Hadamard + outlier full pipeline
- Attention score preservation
