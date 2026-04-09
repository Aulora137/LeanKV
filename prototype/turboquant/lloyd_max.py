"""
Lloyd-Max optimal scalar quantization for TurboQuant.

After random rotation, each coordinate of a unit vector on S^{d-1} follows
a Beta distribution: f_X(x) = Γ(d/2) / (√π · Γ((d-1)/2)) · (1-x²)^((d-3)/2)
which converges to N(0, 1/d) in high dimensions.

The Lloyd-Max algorithm finds optimal quantization levels that minimize
mean squared error (MSE) for a given distribution — equivalent to solving
a continuous 1D k-means problem.

For b=3 bits: 8 optimal levels for Gaussian distribution
For b=4 bits: 16 optimal levels
For b=2 bits: 4 optimal levels

TurboQuant's key result (Theorem 1):
  MSE distortion ≤ (√3π/2) · 1/4^b  for any bit-width b
  At b=3: MSE ≈ 0.03 per coordinate (vs 0.25 for linear quantization)

Reference: Lloyd (1982), Max (1960), TurboQuant Section 3.1
"""

import torch
import numpy as np
from scipy import integrate, optimize, stats
from typing import Tuple, Optional, Union

# Mapping from fractional bits to number of quantization levels
BITS_TO_LEVELS = {
    2: 4, 2.5: 6, 3: 8, 3.125: 9, 3.5: 12, 4: 16,
}
SUPPORTED_BITS = sorted(BITS_TO_LEVELS.keys())

# Module-level codebook cache: (n_levels, dim) -> (levels, boundaries)
_codebook_cache: dict = {}


def beta_pdf(x: float, d: int) -> float:
    """PDF of a coordinate of a uniformly random point on S^{d-1}.

    f_X(x) = Γ(d/2) / (√π · Γ((d-1)/2)) · (1-x²)^((d-3)/2)

    For d ≥ 64 (typical head_dim), this is nearly Gaussian N(0, 1/d).
    """
    from scipy.special import gamma
    if abs(x) >= 1.0:
        return 0.0
    coeff = gamma(d / 2) / (np.sqrt(np.pi) * gamma((d - 1) / 2))
    return coeff * (1 - x**2) ** ((d - 3) / 2)


def gaussian_pdf(x: float, variance: float) -> float:
    """PDF of N(0, variance)."""
    return np.exp(-x**2 / (2 * variance)) / np.sqrt(2 * np.pi * variance)


def lloyd_max_1d(
    pdf_func,
    support: Tuple[float, float],
    n_levels: int,
    max_iter: int = 200,
    tol: float = 1e-10,
) -> Tuple[np.ndarray, np.ndarray, float]:
    """Lloyd-Max algorithm for optimal scalar quantization.

    Finds n_levels quantization levels (centroids) and n_levels-1 decision
    boundaries that minimize E[(X - Q(X))²] for a random variable X with
    the given PDF.

    This is a continuous 1D k-means problem solved iteratively:
    1. Given centroids, boundaries are midpoints between consecutive centroids
    2. Given boundaries, centroids are conditional expectations within each bin

    Args:
        pdf_func: Probability density function f(x)
        support: (a, b) interval containing the distribution
        n_levels: Number of quantization levels (2^b for b-bit quantization)
        max_iter: Maximum Lloyd iterations
        tol: Convergence tolerance on MSE change

    Returns:
        levels: Array of n_levels reconstruction values (sorted)
        boundaries: Array of n_levels-1 decision boundaries
        mse: Final mean squared error
    """
    a, b = support

    # Initialize levels uniformly
    levels = np.linspace(a + (b - a) / (2 * n_levels),
                         b - (b - a) / (2 * n_levels),
                         n_levels)

    prev_mse = float('inf')

    for iteration in range(max_iter):
        # Step 1: Update boundaries as midpoints between consecutive levels
        boundaries = (levels[:-1] + levels[1:]) / 2

        # Step 2: Update levels as conditional expectations within each bin
        # For bin i: level_i = E[X | boundary_{i-1} < X < boundary_i]
        bin_edges = np.concatenate([[a], boundaries, [b]])
        new_levels = np.zeros(n_levels)

        for i in range(n_levels):
            lo, hi = bin_edges[i], bin_edges[i + 1]

            # Numerator: ∫ x · f(x) dx over [lo, hi]
            num, _ = integrate.quad(lambda x: x * pdf_func(x), lo, hi)
            # Denominator: ∫ f(x) dx over [lo, hi] (probability of bin)
            den, _ = integrate.quad(pdf_func, lo, hi)

            if den > 1e-15:
                new_levels[i] = num / den
            else:
                # Empty bin — keep previous level
                new_levels[i] = levels[i]

        levels = new_levels

        # Compute MSE
        mse = 0.0
        for i in range(n_levels):
            lo, hi = bin_edges[i], bin_edges[i + 1]
            mse_i, _ = integrate.quad(
                lambda x: (x - levels[i])**2 * pdf_func(x), lo, hi
            )
            mse += mse_i

        if abs(prev_mse - mse) < tol:
            break
        prev_mse = mse

    boundaries = (levels[:-1] + levels[1:]) / 2
    return levels, boundaries, mse


def compute_codebook(
    bits: Union[int, float],
    dim: int = 128,
    use_gaussian_approx: bool = True,
) -> Tuple[np.ndarray, np.ndarray, float]:
    """Compute optimal Lloyd-Max codebook for post-rotation distribution.

    Args:
        bits: Quantization bits (supports fractional: 2, 2.5, 3, 3.125, 3.5, 4)
        dim: Head dimension d (affects Beta distribution shape)
        use_gaussian_approx: If True, use N(0, 1/d) approximation (faster, accurate for d≥64)

    Returns:
        levels: Optimal reconstruction levels
        boundaries: Decision boundaries
        mse: Mean squared error per coordinate
    """
    if bits in BITS_TO_LEVELS:
        n_levels = BITS_TO_LEVELS[bits]
    else:
        n_levels = round(2 ** bits)

    if use_gaussian_approx and dim >= 32:
        # N(0, 1/d) approximation — accurate for d ≥ 32
        variance = 1.0 / dim
        std = np.sqrt(variance)
        pdf = lambda x: gaussian_pdf(x, variance)
        # Support: ±4σ covers 99.99% of the distribution
        support = (-4 * std, 4 * std)
    else:
        # Exact Beta distribution
        pdf = lambda x: beta_pdf(x, dim)
        support = (-1.0, 1.0)

    levels, boundaries, mse = lloyd_max_1d(pdf, support, n_levels)
    return levels, boundaries, mse


# Pre-computed codebooks for common configurations
# These are computed once and hardcoded for efficiency (matching TurboQuant paper)
def get_precomputed_codebook(bits: Union[int, float], dim: int = 128) -> Tuple[torch.Tensor, torch.Tensor]:
    """Get Lloyd-Max codebook with caching.

    Supports fractional bits (2, 2.5, 3, 3.125, 3.5, 4).
    Codebooks are cached so Lloyd-Max only runs once per (bits, dim) pair.

    Args:
        bits: Quantization bits (integer or fractional)
        dim: Head dimension

    Returns:
        levels: Tensor of reconstruction values
        boundaries: Tensor of decision boundaries
    """
    if bits in BITS_TO_LEVELS:
        n_levels = BITS_TO_LEVELS[bits]
    else:
        n_levels = round(2 ** bits)

    cache_key = (n_levels, dim)
    if cache_key not in _codebook_cache:
        levels, boundaries, mse = compute_codebook(bits, dim)
        _codebook_cache[cache_key] = (
            torch.tensor(levels, dtype=torch.float32),
            torch.tensor(boundaries, dtype=torch.float32),
        )
    return _codebook_cache[cache_key]


def quantize_scalar(
    x: torch.Tensor,
    boundaries: torch.Tensor,
) -> torch.Tensor:
    """Quantize values to codebook indices via binary search on boundaries.

    Args:
        x: Input values of any shape
        boundaries: Sorted decision boundaries of shape (2^b - 1,)

    Returns:
        Integer indices of shape matching x, values in [0, 2^b - 1]
    """
    # torch.bucketize does binary search — O(log(2^b)) = O(b) per element
    return torch.bucketize(x, boundaries)


def dequantize_scalar(
    indices: torch.Tensor,
    levels: torch.Tensor,
) -> torch.Tensor:
    """Dequantize indices back to reconstruction values via table lookup.

    Args:
        indices: Integer indices of any shape, values in [0, 2^b - 1]
        levels: Reconstruction values of shape (2^b,)

    Returns:
        Reconstructed values of shape matching indices
    """
    return levels[indices]
