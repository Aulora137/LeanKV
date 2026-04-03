"""
Test RoPE invariance of the TurboQuant rotation.

The fundamental theorem:
  (Π R_{θ,i} q)^T (Π R_{θ,j} k) = q^T R_{θ,i}^T Π^T Π R_{θ,j} k = q^T R_{θ,i}^T R_{θ,j} k

Since Π^T Π = I for any orthogonal matrix Π, attention scores are EXACTLY
preserved regardless of the choice of Π.

This test verifies this property numerically for all three rotation strategies.
"""

import torch
import numpy as np
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from turboquant.rotation import (
    random_orthogonal,
    hadamard_matrix,
    randomized_hadamard,
    apply_rotation,
)


def apply_rope(x: torch.Tensor, positions: torch.Tensor, theta: float = 10000.0) -> torch.Tensor:
    """Apply Rotary Position Embedding (RoPE) to input tensor.

    Standard RoPE: for each pair (x_{2i}, x_{2i+1}), apply 2D rotation
    by angle position * theta_i where theta_i = theta^{-2i/d}.

    Args:
        x: Input of shape (batch, heads, seq, head_dim)
        positions: Position indices of shape (seq,)
        theta: RoPE base frequency

    Returns:
        RoPE-applied tensor of same shape
    """
    d = x.shape[-1]
    freqs = 1.0 / (theta ** (torch.arange(0, d, 2, dtype=torch.float32) / d))
    # (seq, d//2)
    angles = positions.unsqueeze(-1) * freqs.unsqueeze(0)

    cos_angles = torch.cos(angles)  # (seq, d//2)
    sin_angles = torch.sin(angles)  # (seq, d//2)

    # Reshape for broadcasting: (1, 1, seq, d//2)
    cos_angles = cos_angles.unsqueeze(0).unsqueeze(0)
    sin_angles = sin_angles.unsqueeze(0).unsqueeze(0)

    # Split into even/odd pairs
    x_even = x[..., 0::2]  # (..., d//2)
    x_odd = x[..., 1::2]   # (..., d//2)

    # Apply 2D rotation to each pair
    out_even = x_even * cos_angles - x_odd * sin_angles
    out_odd = x_even * sin_angles + x_odd * cos_angles

    # Interleave back
    out = torch.zeros_like(x)
    out[..., 0::2] = out_even
    out[..., 1::2] = out_odd

    return out


def test_rope_invariance_basic():
    """Test that Π preserves attention scores with RoPE."""
    torch.manual_seed(42)

    batch, heads, seq_q, seq_k, d = 1, 2, 1, 8, 64

    Q = torch.randn(batch, heads, seq_q, d)
    K = torch.randn(batch, heads, seq_k, d)

    pos_q = torch.tensor([seq_k - 1])  # query at last position
    pos_k = torch.arange(seq_k)

    # Standard attention: apply RoPE, then compute Q^T K
    Q_rope = apply_rope(Q, pos_q)
    K_rope = apply_rope(K, pos_k)
    scores_standard = torch.matmul(Q_rope, K_rope.transpose(-2, -1))

    # TurboQuant attention: apply RoPE, then rotate BOTH Q and K by Π
    for name, rotation_fn in [
        ("random_orthogonal", lambda: random_orthogonal(d, seed=123)),
        ("hadamard", lambda: hadamard_matrix(d)),
        ("randomized_hadamard", lambda: randomized_hadamard(d, seed=456)),
    ]:
        Pi = rotation_fn()

        # Verify orthogonality: Π^T Π = I
        identity = Pi.T @ Pi
        assert torch.allclose(identity, torch.eye(d), atol=1e-5), \
            f"{name}: Π^T Π ≠ I, max error = {(identity - torch.eye(d)).abs().max()}"

        # Apply rotation AFTER RoPE (matching ik_llama.cpp's pipeline)
        Q_rotated = apply_rotation(Q_rope, Pi)
        K_rotated = apply_rotation(K_rope, Pi)

        # Attention in rotated space
        scores_rotated = torch.matmul(Q_rotated, K_rotated.transpose(-2, -1))

        # These should be EXACTLY equal (up to floating-point precision)
        max_error = (scores_standard - scores_rotated).abs().max().item()
        assert max_error < 1e-4, \
            f"{name}: Attention scores differ! max_error = {max_error}"
        print(f"  {name}: PASS (max attention score error = {max_error:.2e})")


def test_rope_invariance_different_dims():
    """Test RoPE invariance across different head dimensions."""
    torch.manual_seed(42)

    for d in [32, 64, 128]:
        Q = torch.randn(1, 1, 1, d)
        K = torch.randn(1, 1, 4, d)

        pos_q = torch.tensor([3])
        pos_k = torch.arange(4)

        Q_rope = apply_rope(Q, pos_q)
        K_rope = apply_rope(K, pos_k)
        scores_ref = torch.matmul(Q_rope, K_rope.transpose(-2, -1))

        Pi = randomized_hadamard(d, seed=99)
        Q_rot = apply_rotation(Q_rope, Pi)
        K_rot = apply_rotation(K_rope, Pi)
        scores_rot = torch.matmul(Q_rot, K_rot.transpose(-2, -1))

        max_error = (scores_ref - scores_rot).abs().max().item()
        assert max_error < 1e-4, f"d={d}: max_error = {max_error}"
        print(f"  head_dim={d}: PASS (max error = {max_error:.2e})")


def test_rotation_orthogonality():
    """Verify all rotation matrices satisfy Π^T Π = I."""
    for d in [32, 64, 128]:
        for name, fn in [
            ("random_orthogonal", lambda: random_orthogonal(d, seed=42)),
            ("hadamard", lambda: hadamard_matrix(d)),
            ("randomized_hadamard", lambda: randomized_hadamard(d, seed=42)),
        ]:
            Pi = fn()
            identity = Pi.T @ Pi
            max_error = (identity - torch.eye(d)).abs().max().item()
            assert max_error < 1e-5, f"{name} d={d}: Π^T Π ≠ I, error = {max_error}"
            print(f"  {name} d={d}: orthogonal (error = {max_error:.2e})")


def test_rotation_preserves_norms():
    """Verify rotation preserves vector norms."""
    torch.manual_seed(42)
    d = 64
    x = torch.randn(10, d)

    for name, fn in [
        ("random_orthogonal", lambda: random_orthogonal(d, seed=42)),
        ("hadamard", lambda: hadamard_matrix(d)),
        ("randomized_hadamard", lambda: randomized_hadamard(d, seed=42)),
    ]:
        Pi = fn()
        x_rot = apply_rotation(x, Pi)
        norm_diff = (x.norm(dim=-1) - x_rot.norm(dim=-1)).abs().max().item()
        assert norm_diff < 1e-5, f"{name}: norm not preserved, diff = {norm_diff}"
        print(f"  {name}: norms preserved (max diff = {norm_diff:.2e})")


if __name__ == "__main__":
    print("=== Test: Rotation Orthogonality ===")
    test_rotation_orthogonality()

    print("\n=== Test: Rotation Preserves Norms ===")
    test_rotation_preserves_norms()

    print("\n=== Test: RoPE Invariance (basic) ===")
    test_rope_invariance_basic()

    print("\n=== Test: RoPE Invariance (different dims) ===")
    test_rope_invariance_different_dims()

    print("\n=== ALL TESTS PASSED ===")
