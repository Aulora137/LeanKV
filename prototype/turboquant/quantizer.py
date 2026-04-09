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

import math
import torch
from typing import Tuple, Optional, Union, NamedTuple
from .rotation import generate_per_layer_rotations, apply_rotation
from .lloyd_max import get_precomputed_codebook, quantize_scalar, dequantize_scalar, BITS_TO_LEVELS
from .qjl_residual import QJLResidual


class QuantizedKV(NamedTuple):
    """Packed quantized KV representation."""
    indices: torch.Tensor      # uint8: Lloyd-Max indices, shape (..., d)
    qjl_signs: Optional[torch.Tensor]  # uint8 packed: QJL signs, shape (..., d//8)
    qjl_mean_abs: Optional[torch.Tensor]  # float32: mean |residual|, shape (...)
    norms: torch.Tensor        # float32: per-vector or per-group norms


class TurboQuantizer:
    """TurboQuant KV cache quantizer.

    Implements the full rotation + Lloyd-Max + QJL pipeline.

    Args:
        n_layers: Number of transformer layers
        head_dim: Attention head dimension (must be power of 2)
        bits: Quantization bit-width (2, 2.5, 3, 3.125, 3.5, or 4)
        group_size: Quantization group size (default=None → head_dim).
                    Smaller groups = finer per-group scale = better quality but more overhead.
        rotation_strategy: "random_orthogonal", "hadamard", or "randomized_hadamard"
        use_qjl: Whether to apply QJL residual correction
        seed: Random seed for reproducibility
        device: torch device
    """

    def __init__(
        self,
        n_layers: int,
        head_dim: int,
        bits: Union[int, float] = 3,
        group_size: Optional[int] = None,
        rotation_strategy: str = "randomized_hadamard",
        use_qjl: bool = True,
        seed: int = 42,
        device: str = "cpu",
    ):
        self.n_layers = n_layers
        self.head_dim = head_dim
        self.bits = bits
        self.group_size = min(group_size, head_dim) if group_size is not None else head_dim
        self.use_qjl = use_qjl
        self.device = device

        assert head_dim % self.group_size == 0, \
            f"head_dim ({head_dim}) must be divisible by group_size ({self.group_size})"
        self.n_groups = head_dim // self.group_size

        # Generate per-layer rotation matrices (always full head_dim)
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
        # Apply rotation first (full head_dim)
        rotation = self.rotations[layer_idx]
        x_rotated = apply_rotation(x, rotation)

        if self.group_size < self.head_dim:
            # Per-group quantization: reshape, compute per-group scale
            orig_shape = x_rotated.shape  # (batch, n_heads, seq, head_dim)
            x_grouped = x_rotated.reshape(*orig_shape[:-1], self.n_groups, self.group_size)
            # Per-group amax scale
            norms = x_grouped.abs().amax(dim=-1)  # (batch, n_heads, seq, n_groups)
            x_normalized = x_grouped / (norms.unsqueeze(-1) + 1e-10)
            # Quantize in [-1, 1] range
            indices = quantize_scalar(x_normalized, self.boundaries)
            reconstructed = dequantize_scalar(indices, self.levels)
            # Flatten back for QJL
            x_normalized_flat = x_normalized.reshape(orig_shape)
            reconstructed_flat = reconstructed.reshape(orig_shape)
        else:
            # Original per-vector quantization
            norms = x.norm(dim=-1)  # (batch, n_heads, seq_len)
            x_normalized = x_rotated / (norms.unsqueeze(-1) + 1e-10)
            indices = quantize_scalar(x_normalized, self.boundaries)
            reconstructed = dequantize_scalar(indices, self.levels)
            x_normalized_flat = x_normalized
            reconstructed_flat = reconstructed

        # QJL residual correction
        qjl_signs = None
        qjl_mean_abs = None
        if self.qjl is not None:
            residual = x_normalized_flat - reconstructed_flat
            qjl_signs, qjl_mean_abs = self.qjl.encode(residual, pack=True)

        return QuantizedKV(
            indices=indices.to(torch.uint8).reshape(*x.shape[:-1], self.head_dim),
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

        if self.group_size < self.head_dim:
            # Rescale by per-group norms
            orig_shape = reconstructed.shape
            reconstructed = reconstructed.reshape(*orig_shape[:-1], self.n_groups, self.group_size)
            reconstructed = reconstructed * qkv.norms.unsqueeze(-1)
            reconstructed = reconstructed.reshape(orig_shape)
        else:
            # Rescale by per-vector norms
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
        - Lloyd-Max indices: log2(n_levels) per element
        - QJL signs: 1 bit per element (if enabled)
        - QJL mean_abs: 32 bits / head_dim per element (if enabled)
        - Group scale: 32 bits / group_size per element
        """
        total = math.log2(BITS_TO_LEVELS.get(self.bits, round(2 ** self.bits)))
        if self.use_qjl:
            total += 1.0  # sign bits
            total += 32.0 / self.head_dim  # mean_abs overhead
        total += 32.0 / self.group_size  # per-group scale overhead
        return total

    def compression_ratio(self, baseline_bits: int = 16) -> float:
        """Compression ratio vs FP16 baseline."""
        return baseline_bits / self.memory_bits_per_element()

    def __repr__(self) -> str:
        return (f"TurboQuantizer(layers={self.n_layers}, head_dim={self.head_dim}, "
                f"bits={self.bits}, group_size={self.group_size}, qjl={self.use_qjl}, "
                f"effective_bits={self.memory_bits_per_element():.2f}, "
                f"compression={self.compression_ratio():.1f}x)")
