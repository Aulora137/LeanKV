"""
TurboQuant full quantization pipeline.

Combines: Random rotation → Lloyd-Max scalar quantization → QJL residual correction

Pipeline for KV cache:
  WRITE (prefill/encode):
    1. K already has RoPE applied (by the model)
    2. Rotate: K_rot = K @ Π^T
    3. Quantize: indices = lloyd_max_quantize(K_rot)
    4. QJL: signs, mean_abs = qjl_encode(K_rot - dequantize(indices))
    5. Store: (indices, signs, mean_abs, ‖K‖) in cache

  READ (decode/generate):
    1. Load: (indices, signs, mean_abs, norms) from cache
    2. Dequantize: K_rot_approx = lloyd_max_dequantize(indices)
    3. QJL correct: K_rot_corrected = K_rot_approx + decode(signs, mean_abs)
    4. For attention: Q_rot = Q @ Π^T, then score = Q_rot · K_rot_corrected^T
       (No inverse rotation needed! Π^T Π = I cancels in the dot product)

Memory per KV element:
  - 3-bit Lloyd-Max index: 3 bits
  - 1-bit QJL sign: 1 bit
  - Mean abs residual: 32 bits / d elements ≈ 0.25-0.5 bits
  - Norm: 32 bits / d elements ≈ 0.25-0.5 bits
  Total: ~4.5-5 bits per element (vs 16 for FP16)
  Compression: 3.2-3.6x

  Without QJL (pure 3-bit):
  Total: ~3.5 bits per element
  Compression: ~4.6x
"""

import torch
from typing import Tuple, Optional, NamedTuple
from .rotation import generate_per_layer_rotations, apply_rotation
from .lloyd_max import get_precomputed_codebook, quantize_scalar, dequantize_scalar
from .qjl_residual import QJLResidual


class QuantizedKV(NamedTuple):
    """Packed quantized KV representation."""
    indices: torch.Tensor      # uint8: Lloyd-Max indices, shape (..., d)
    qjl_signs: Optional[torch.Tensor]  # uint8 packed: QJL signs, shape (..., d//8)
    qjl_mean_abs: Optional[torch.Tensor]  # float32: mean |residual|, shape (...)
    norms: torch.Tensor        # float32: original vector norms, shape (...)


class TurboQuantizer:
    """TurboQuant KV cache quantizer.

    Implements the full rotation + Lloyd-Max + QJL pipeline.

    Args:
        n_layers: Number of transformer layers
        head_dim: Attention head dimension (must be power of 2)
        bits: Quantization bit-width (2, 3, or 4)
        rotation_strategy: "random_orthogonal", "hadamard", or "randomized_hadamard"
        use_qjl: Whether to apply QJL residual correction
        seed: Random seed for reproducibility
        device: torch device
    """

    def __init__(
        self,
        n_layers: int,
        head_dim: int,
        bits: int = 3,
        rotation_strategy: str = "randomized_hadamard",
        use_qjl: bool = True,
        seed: int = 42,
        device: str = "cpu",
    ):
        self.n_layers = n_layers
        self.head_dim = head_dim
        self.bits = bits
        self.use_qjl = use_qjl
        self.device = device

        # Generate per-layer rotation matrices
        self.rotations = generate_per_layer_rotations(
            n_layers, head_dim, rotation_strategy, seed
        )
        self.rotations = [R.to(device) for R in self.rotations]

        # Pre-compute Lloyd-Max codebook
        self.levels, self.boundaries = get_precomputed_codebook(bits, head_dim)
        self.levels = self.levels.to(device)
        self.boundaries = self.boundaries.to(device)

        # QJL module
        self.qjl = QJLResidual() if use_qjl else None

    def quantize(
        self,
        x: torch.Tensor,
        layer_idx: int,
    ) -> QuantizedKV:
        """Quantize a KV embedding for cache storage.

        Args:
            x: Input tensor of shape (batch, n_heads, seq_len, head_dim)
               K should already have RoPE applied.
            layer_idx: Transformer layer index (for selecting rotation matrix)

        Returns:
            QuantizedKV with packed indices, optional QJL data, and norms
        """
        # Save norms for rescaling at dequantization
        norms = x.norm(dim=-1)  # (batch, n_heads, seq_len)

        # Normalize to unit sphere (TurboQuant assumes ‖x‖=1)
        x_normalized = x / (norms.unsqueeze(-1) + 1e-10)

        # Apply rotation: x_rot = x_norm @ Π^T
        rotation = self.rotations[layer_idx]
        x_rotated = apply_rotation(x_normalized, rotation)

        # Lloyd-Max quantize
        indices = quantize_scalar(x_rotated, self.boundaries)
        reconstructed = dequantize_scalar(indices, self.levels)

        # QJL residual correction
        qjl_signs = None
        qjl_mean_abs = None
        if self.qjl is not None:
            residual = x_rotated - reconstructed
            qjl_signs, qjl_mean_abs = self.qjl.encode(residual, pack=True)

        return QuantizedKV(
            indices=indices.to(torch.uint8),
            qjl_signs=qjl_signs,
            qjl_mean_abs=qjl_mean_abs,
            norms=norms,
        )

    def dequantize(
        self,
        qkv: QuantizedKV,
        layer_idx: int,
        apply_inverse_rot: bool = False,
    ) -> torch.Tensor:
        """Dequantize a cached KV embedding.

        For attention computation, inverse rotation is NOT needed if Q is
        also rotated (which it is). Set apply_inverse_rot=True only for
        debugging or when computing values (V cache doesn't use rotation).

        Args:
            qkv: QuantizedKV struct
            layer_idx: Transformer layer index
            apply_inverse_rot: If True, apply Π^T to recover original space

        Returns:
            Reconstructed tensor of shape (batch, n_heads, seq_len, head_dim)
        """
        # Lloyd-Max dequantize
        reconstructed = dequantize_scalar(qkv.indices.long(), self.levels)

        # QJL correction
        if self.qjl is not None and qkv.qjl_signs is not None:
            correction = self.qjl.decode(
                qkv.qjl_signs, qkv.qjl_mean_abs,
                d=self.head_dim, packed=True
            )
            reconstructed = reconstructed + correction

        # Rescale by original norms
        reconstructed = reconstructed * qkv.norms.unsqueeze(-1)

        # Optionally apply inverse rotation
        if apply_inverse_rot:
            rotation = self.rotations[layer_idx]
            reconstructed = reconstructed @ rotation  # x @ Π (inverse = transpose)

        return reconstructed

    def quantize_for_attention(
        self,
        q: torch.Tensor,
        layer_idx: int,
    ) -> torch.Tensor:
        """Rotate query for attention with quantized keys.

        Since cached keys are in rotated space, the query must also be rotated
        for the dot product to be correct:
          (Π q)^T (Π k) = q^T Π^T Π k = q^T k ✓

        Args:
            q: Query tensor of shape (batch, n_heads, seq_len, head_dim)
            layer_idx: Transformer layer index

        Returns:
            Rotated query tensor
        """
        rotation = self.rotations[layer_idx]
        return apply_rotation(q, rotation)

    def memory_bits_per_element(self) -> float:
        """Compute total memory cost in bits per KV element.

        Components:
        - Lloyd-Max indices: self.bits per element
        - QJL signs: 1 bit per element (if enabled)
        - QJL mean_abs: 32 bits / head_dim per element (if enabled)
        - Norms: 32 bits / head_dim per element
        """
        total = float(self.bits)
        if self.use_qjl:
            total += 1.0  # sign bits
            total += 32.0 / self.head_dim  # mean_abs overhead
        total += 32.0 / self.head_dim  # norm overhead
        return total

    def compression_ratio(self, baseline_bits: int = 16) -> float:
        """Compression ratio vs FP16 baseline."""
        return baseline_bits / self.memory_bits_per_element()

    def __repr__(self) -> str:
        return (f"TurboQuantizer(layers={self.n_layers}, head_dim={self.head_dim}, "
                f"bits={self.bits}, qjl={self.use_qjl}, "
                f"effective_bits={self.memory_bits_per_element():.2f}, "
                f"compression={self.compression_ratio():.1f}x)")
