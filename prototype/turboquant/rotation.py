"""
Rotation matrix generation for TurboQuant.

The key insight from TurboQuant: multiplying a vector by a random orthogonal
matrix Π spreads outlier energy uniformly across all dimensions, converting
heavy-tailed activation distributions into approximate Gaussians — the ideal
target for any fixed-point quantizer.

Three strategies implemented:
1. Random orthogonal: QR decomposition of random Gaussian matrix (paper default)
2. Hadamard: Walsh-Hadamard matrix, O(d log d), no storage needed
3. Randomized Hadamard: Hadamard × random sign diagonal (best practical choice)

RoPE compatibility:
  (Π R_{θ,i} q)^T (Π R_{θ,j} k) = q^T R_{θ,i}^T Π^T Π R_{θ,j} k = q^T R_{θ,i}^T R_{θ,j} k
  Since Π^T Π = I, attention scores are exactly preserved.
  Requirement: apply RoPE BEFORE rotation. In ik_llama.cpp, K is stored post-RoPE,
  so rotation goes after RoPE — correct by construction.
"""

import torch
import numpy as np
from typing import Optional


def random_orthogonal(d: int, seed: Optional[int] = None) -> torch.Tensor:
    """Generate a random d×d orthogonal matrix via QR decomposition.

    This is the paper's default approach. Each coordinate of Πx follows a
    Beta distribution that converges to N(0, 1/d) in high dimensions.

    Args:
        d: Dimension (typically head_dim, e.g. 64 or 128)
        seed: Random seed for reproducibility

    Returns:
        Orthogonal matrix Π of shape (d, d), satisfying Π^T Π = I
    """
    if seed is not None:
        torch.manual_seed(seed)
    # QR decomposition of random Gaussian matrix
    G = torch.randn(d, d)
    Q, R = torch.linalg.qr(G)
    # Ensure proper rotation (det = +1) by adjusting signs
    # This makes Q uniformly distributed over the orthogonal group O(d)
    diag_sign = torch.sign(torch.diag(R))
    Q = Q * diag_sign.unsqueeze(0)
    return Q


def hadamard_matrix(d: int) -> torch.Tensor:
    """Generate a normalized d×d Walsh-Hadamard matrix.

    Requires d to be a power of 2. The Hadamard transform can be applied
    in O(d log d) time via the butterfly algorithm, but here we construct
    the full matrix for the prototype. C++ implementation will use the
    fast transform (ggml_hadamard already exists in ik_llama.cpp).

    Args:
        d: Dimension (must be power of 2)

    Returns:
        Normalized Hadamard matrix H of shape (d, d), satisfying H^T H = I
    """
    assert d > 0 and (d & (d - 1)) == 0, f"d must be power of 2, got {d}"
    # Build recursively: H_d = H_2 ⊗ H_{d/2}
    H = torch.tensor([[1.0]])
    while H.shape[0] < d:
        H = torch.cat([
            torch.cat([H, H], dim=1),
            torch.cat([H, -H], dim=1),
        ], dim=0)
    # Normalize so H^T H = I
    return H / np.sqrt(d)


def randomized_hadamard(d: int, seed: Optional[int] = None) -> torch.Tensor:
    """Generate a randomized Hadamard matrix: H × diag(±1).

    This is the recommended approach for production:
    - Structured (O(d log d) fast transform available)
    - Randomized (spreads outliers as well as random orthogonal)
    - Deterministic from seed (no matrix storage needed)
    - Already validated in ik_llama.cpp's k_cache_hadamard feature

    The random sign diagonal breaks any alignment between the input
    distribution and the Hadamard basis, ensuring uniform variance
    redistribution regardless of the input.

    Args:
        d: Dimension (must be power of 2)
        seed: Random seed for reproducibility

    Returns:
        Randomized Hadamard matrix of shape (d, d), satisfying Π^T Π = I
    """
    if seed is not None:
        torch.manual_seed(seed)
    H = hadamard_matrix(d)
    # Random ±1 diagonal
    signs = torch.randint(0, 2, (d,)) * 2 - 1  # {-1, +1}
    signs = signs.float()
    # H × diag(signs) = multiply each column of H by the sign
    return H * signs.unsqueeze(0)


def fast_hadamard_transform(x: torch.Tensor) -> torch.Tensor:
    """Apply Walsh-Hadamard transform in O(d log d) time.

    This is the butterfly algorithm — much faster than matrix multiply
    for the C++ implementation. Used here for validation.

    Args:
        x: Input tensor of shape (..., d) where d is power of 2

    Returns:
        Transformed tensor of shape (..., d), normalized by 1/sqrt(d)
    """
    d = x.shape[-1]
    assert d > 0 and (d & (d - 1)) == 0, f"d must be power of 2, got {d}"

    result = x.clone()
    h = 1
    while h < d:
        # Butterfly operation: pairs at distance h
        for i in range(0, d, h * 2):
            for j in range(i, i + h):
                a = result[..., j].clone()
                b = result[..., j + h].clone()
                result[..., j] = a + b
                result[..., j + h] = a - b
        h *= 2

    return result / np.sqrt(d)


def apply_rotation(x: torch.Tensor, rotation: torch.Tensor) -> torch.Tensor:
    """Apply rotation matrix to input tensor.

    Args:
        x: Input of shape (..., d) — e.g. (batch, heads, seq, head_dim)
        rotation: Orthogonal matrix of shape (d, d)

    Returns:
        Rotated tensor of shape (..., d)
    """
    return x @ rotation.T


def apply_inverse_rotation(x: torch.Tensor, rotation: torch.Tensor) -> torch.Tensor:
    """Apply inverse rotation (Π^T) to recover original space.

    Since Π is orthogonal, Π^{-1} = Π^T.

    Args:
        x: Rotated input of shape (..., d)
        rotation: Orthogonal matrix of shape (d, d)

    Returns:
        De-rotated tensor of shape (..., d)
    """
    return x @ rotation  # Π^T x = (x^T Π)^T, but for batch: x @ Π


def generate_per_layer_rotations(
    n_layers: int,
    head_dim: int,
    strategy: str = "randomized_hadamard",
    base_seed: int = 42,
) -> list[torch.Tensor]:
    """Generate rotation matrices for all layers.

    Args:
        n_layers: Number of transformer layers
        head_dim: Attention head dimension
        strategy: One of "random_orthogonal", "hadamard", "randomized_hadamard"
        base_seed: Base random seed (per-layer seed = base_seed + layer_idx)

    Returns:
        List of n_layers rotation matrices, each (head_dim, head_dim)
    """
    generators = {
        "random_orthogonal": lambda s: random_orthogonal(head_dim, seed=s),
        "hadamard": lambda s: hadamard_matrix(head_dim),  # deterministic, ignores seed
        "randomized_hadamard": lambda s: randomized_hadamard(head_dim, seed=s),
    }

    if strategy not in generators:
        raise ValueError(f"Unknown rotation strategy: {strategy}. "
                        f"Choose from {list(generators.keys())}")

    gen = generators[strategy]
    return [gen(base_seed + il) for il in range(n_layers)]
