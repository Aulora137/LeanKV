#!/usr/bin/env python3
"""
TQ3 Optimal Rounding Prototype (Karpathy Loop)

Compare rounding strategies for TQ3_0 (3-bit Lloyd-Max quantization):
  1. Baseline: nearest-level + max|x| scale (current C++ implementation)
  2. Optimal scale: nearest-level + least-squares optimal scale
  3. Coordinate descent: iterative rounding refinement with optimal scale

Operates on synthetic post-Hadamard Gaussian data (which is what the
real KV cache values look like after Hadamard rotation).
"""

import numpy as np
from dataclasses import dataclass

# Lloyd-Max codebook for N(0,1), 8 levels, normalized to [-1, 1]
TQ3_LEVELS = np.array([
    -1.0000000, -0.6245203, -0.3513239, -0.1138989,
    +0.1138989, +0.3513239, +0.6245203, +1.0000000,
], dtype=np.float64)

TQ3_BOUNDARIES = np.array([
    -0.8122602, -0.4879221, -0.2326114, +0.0000000,
    +0.2326114, +0.4879221, +0.8122602,
], dtype=np.float64)

BLOCK_SIZE = 32


def find_nearest_tq3(xn: np.ndarray) -> np.ndarray:
    """Nearest-level lookup via boundaries (matches C++ implementation)."""
    indices = np.zeros(len(xn), dtype=np.int32)
    for b in range(7):
        indices[xn > TQ3_BOUNDARIES[b]] = b + 1
    return indices


def quantize_baseline(block: np.ndarray):
    """Current C++ implementation: nearest-level + max|x| scale."""
    amax = np.max(np.abs(block))
    d = amax
    if d == 0:
        return np.zeros_like(block), d, np.zeros(BLOCK_SIZE, dtype=np.int32)
    xn = np.clip(block / d, -1.0, 1.0)
    indices = find_nearest_tq3(xn)
    recon = TQ3_LEVELS[indices] * d
    return recon, d, indices


def optimal_scale(block: np.ndarray, indices: np.ndarray) -> float:
    """Least-squares optimal scale: minimizes MSE for given index assignment."""
    levels = TQ3_LEVELS[indices]
    denom = np.dot(levels, levels)
    if denom == 0:
        return 0.0
    return np.dot(block, levels) / denom


def quantize_optimal_scale(block: np.ndarray):
    """Nearest-level assignment + least-squares optimal scale."""
    amax = np.max(np.abs(block))
    if amax == 0:
        return np.zeros_like(block), 0.0, np.zeros(BLOCK_SIZE, dtype=np.int32)
    # Initial assignment using max|x| scale
    xn = np.clip(block / amax, -1.0, 1.0)
    indices = find_nearest_tq3(xn)
    # Recompute scale optimally
    d = optimal_scale(block, indices)
    recon = TQ3_LEVELS[indices] * d
    return recon, d, indices


def quantize_coord_descent(block: np.ndarray, n_passes: int = 3,
                           try_all_levels: bool = False):
    """Coordinate descent: refine index assignments to minimize block MSE."""
    amax = np.max(np.abs(block))
    if amax == 0:
        return np.zeros_like(block), 0.0, np.zeros(BLOCK_SIZE, dtype=np.int32)

    # Start with nearest-level assignment
    xn = np.clip(block / amax, -1.0, 1.0)
    indices = find_nearest_tq3(xn)
    d = optimal_scale(block, indices)

    def block_mse(idx, scale):
        recon = TQ3_LEVELS[idx] * scale
        return np.mean((block - recon) ** 2)

    best_mse = block_mse(indices, d)

    for _ in range(n_passes):
        improved = False
        for j in range(BLOCK_SIZE):
            orig_idx = indices[j]

            if try_all_levels:
                candidates = range(8)
            else:
                # Try adjacent levels only
                candidates = []
                if orig_idx > 0:
                    candidates.append(orig_idx - 1)
                if orig_idx < 7:
                    candidates.append(orig_idx + 1)

            for new_idx in candidates:
                if new_idx == orig_idx:
                    continue
                indices[j] = new_idx
                new_d = optimal_scale(block, indices)
                new_mse = block_mse(indices, new_d)
                if new_mse < best_mse:
                    best_mse = new_mse
                    d = new_d
                    improved = True
                    break  # accept first improvement for this element
                else:
                    indices[j] = orig_idx  # revert

        if not improved:
            break  # converged

    recon = TQ3_LEVELS[indices] * d
    return recon, d, indices


def evaluate_strategy(name: str, quantize_fn, blocks: np.ndarray, **kwargs):
    """Evaluate a quantization strategy on many blocks."""
    n_blocks = len(blocks)
    mses = np.zeros(n_blocks)
    cosine_errors = np.zeros(n_blocks)

    for i in range(n_blocks):
        block = blocks[i]
        recon, d, indices = quantize_fn(block, **kwargs)
        mses[i] = np.mean((block - recon) ** 2)

        # Cosine similarity (relevant for attention dot products)
        norm_orig = np.linalg.norm(block)
        norm_recon = np.linalg.norm(recon)
        if norm_orig > 0 and norm_recon > 0:
            cosine_errors[i] = 1.0 - np.dot(block, recon) / (norm_orig * norm_recon)
        else:
            cosine_errors[i] = 0.0

    snr = -10 * np.log10(np.mean(mses) / np.mean(blocks ** 2))
    print(f"  {name:40s}  MSE={np.mean(mses):.6f}  SNR={snr:.2f}dB  "
          f"cos_err={np.mean(cosine_errors):.2e}  "
          f"max_cos_err={np.max(cosine_errors):.2e}")
    return np.mean(mses), snr


def main():
    np.random.seed(42)

    # Generate synthetic post-Hadamard data (approximately Gaussian)
    n_blocks = 10000
    blocks = np.random.randn(n_blocks, BLOCK_SIZE).astype(np.float64)

    # Also test with heavier tails (some layers may have this)
    blocks_heavy = np.random.standard_t(df=5, size=(n_blocks, BLOCK_SIZE))

    print("=" * 90)
    print("TQ3 Rounding Strategy Comparison")
    print("=" * 90)

    for label, data in [("Gaussian N(0,1)", blocks), ("Heavy-tail t(df=5)", blocks_heavy)]:
        print(f"\n--- {label} ({n_blocks} blocks x {BLOCK_SIZE} elements) ---\n")

        # Strategy 1: Baseline (current C++)
        evaluate_strategy("Baseline (nearest + max|x|)", quantize_baseline, data)

        # Strategy 2: Optimal scale only
        evaluate_strategy("Optimal scale (nearest + LS)", quantize_optimal_scale, data)

        # Strategy 3: Coordinate descent (adjacent, 2 passes)
        evaluate_strategy("Coord descent (adj, 2 pass)", quantize_coord_descent,
                          data, n_passes=2, try_all_levels=False)

        # Strategy 4: Coordinate descent (adjacent, 3 passes)
        evaluate_strategy("Coord descent (adj, 3 pass)", quantize_coord_descent,
                          data, n_passes=3, try_all_levels=False)

        # Strategy 5: Coordinate descent (adjacent, 5 passes)
        evaluate_strategy("Coord descent (adj, 5 pass)", quantize_coord_descent,
                          data, n_passes=5, try_all_levels=False)

        # Strategy 6: Coordinate descent (all levels, 3 passes)
        evaluate_strategy("Coord descent (all lvl, 3 pass)", quantize_coord_descent,
                          data, n_passes=3, try_all_levels=True)

    # Compare TQ3 vs TQ4 baseline for reference
    TQ4_LEVELS = np.array([
        -1.0000000, -0.7573038, -0.5923403, -0.4599576,
        -0.3450764, -0.2405254, -0.1421261, -0.0470277,
        +0.0470277, +0.1421261, +0.2405254, +0.3450764,
        +0.4599576, +0.5923403, +0.7573038, +1.0000000,
    ])
    TQ4_BOUNDARIES = np.array([
        -0.8786519, -0.6748221, -0.5261490, -0.4025170,
        -0.2928009, -0.1913257, -0.0945769, +0.0000000,
        +0.0945769, +0.1913257, +0.2928009, +0.4025170,
        +0.5261490, +0.6748221, +0.8786519,
    ])

    def quantize_tq4_baseline(block):
        amax = np.max(np.abs(block))
        d = amax
        if d == 0:
            return np.zeros_like(block), d, np.zeros(BLOCK_SIZE, dtype=np.int32)
        xn = np.clip(block / d, -1.0, 1.0)
        indices = np.zeros(BLOCK_SIZE, dtype=np.int32)
        for b in range(15):
            indices[xn > TQ4_BOUNDARIES[b]] = b + 1
        recon = TQ4_LEVELS[indices] * d
        return recon, d, indices

    print(f"\n--- TQ4 baseline for reference (Gaussian) ---\n")
    evaluate_strategy("TQ4 Baseline (nearest + max|x|)", quantize_tq4_baseline, blocks)


if __name__ == "__main__":
    main()
