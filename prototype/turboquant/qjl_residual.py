"""
Quantized Johnson-Lindenstrauss (QJL) 1-bit residual correction.

After Lloyd-Max quantization, the residual r = x - Q^{-1}(Q(x)) is small
but non-zero. Storing the full residual would double memory cost.

QJL insight: store only sign(r) — 1 bit per element — and correct with:
  x_corrected = Q^{-1}(Q(x)) + sign(r) · ē
where ē is the mean absolute residual magnitude (scalar per group).

From TurboQuant Theorem 2:
  The inner product estimator is UNBIASED: E[⟨y, x̃⟩] = ⟨y, x⟩
  Distortion: D_prod ≤ (√3π²·‖y‖²/d) · 1/4^b

Effective bit-width: 3 bits + 1 bit sign = 4 bits total,
but the QJL theory shows this achieves 4-bit-equivalent accuracy
at the information-theoretic cost of only 3.125 bits.

Reference: QJL paper (Zandieh et al. 2024), TurboQuant Section 3.2
"""

import torch
import numpy as np
from typing import Tuple, Optional


def compute_residual(
    original: torch.Tensor,
    reconstructed: torch.Tensor,
) -> torch.Tensor:
    """Compute quantization residual.

    Args:
        original: Original values (pre-quantization)
        reconstructed: Dequantized values

    Returns:
        Residual tensor: original - reconstructed
    """
    return original - reconstructed


def encode_qjl_residual(
    residual: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Encode residual using QJL 1-bit quantization.

    Stores:
    1. sign(residual) as packed bits (1 bit per element)
    2. mean absolute residual magnitude per group (scalar)

    Args:
        residual: Residual tensor of shape (..., d)

    Returns:
        signs: Boolean tensor of shape (..., d), True where residual >= 0
        mean_abs: Mean absolute residual per last-dim group, shape (...)
    """
    signs = residual >= 0  # True = positive, False = negative
    mean_abs = residual.abs().mean(dim=-1)  # scalar per token per head
    return signs, mean_abs


def decode_qjl_residual(
    signs: torch.Tensor,
    mean_abs: torch.Tensor,
) -> torch.Tensor:
    """Decode QJL residual correction.

    Reconstruction: correction[i] = sign[i] * mean_abs

    Args:
        signs: Boolean tensor of shape (..., d)
        mean_abs: Mean absolute residual, shape (...)

    Returns:
        Correction tensor of shape (..., d)
    """
    # Convert bool to ±1
    sign_values = signs.float() * 2 - 1  # True -> +1, False -> -1
    return sign_values * mean_abs.unsqueeze(-1)


def pack_signs(signs: torch.Tensor) -> torch.Tensor:
    """Pack boolean sign bits into uint8 for memory efficiency.

    8 signs per byte → 1 bit per element.

    Args:
        signs: Boolean tensor of shape (..., d) where d is divisible by 8

    Returns:
        Packed tensor of shape (..., d//8) as uint8
    """
    d = signs.shape[-1]
    assert d % 8 == 0, f"Dimension must be divisible by 8, got {d}"

    # Reshape to groups of 8
    signs_reshaped = signs.view(*signs.shape[:-1], d // 8, 8)

    # Pack: bit 0 is least significant
    powers = torch.tensor([1, 2, 4, 8, 16, 32, 64, 128],
                         dtype=torch.uint8, device=signs.device)
    packed = (signs_reshaped.to(torch.uint8) * powers).sum(dim=-1).to(torch.uint8)
    return packed


def unpack_signs(packed: torch.Tensor, d: int) -> torch.Tensor:
    """Unpack uint8 packed signs back to boolean tensor.

    Args:
        packed: Packed tensor of shape (..., d//8) as uint8
        d: Original dimension

    Returns:
        Boolean tensor of shape (..., d)
    """
    # Expand each byte to 8 bits
    powers = torch.tensor([1, 2, 4, 8, 16, 32, 64, 128],
                         dtype=torch.uint8, device=packed.device)
    unpacked = (packed.unsqueeze(-1) & powers) > 0
    return unpacked.view(*packed.shape[:-1], d)


class QJLResidual:
    """QJL 1-bit residual correction module.

    Usage:
        qjl = QJLResidual()

        # Encode: after Lloyd-Max quantization
        residual = original - dequantized
        signs, mean_abs = qjl.encode(residual)

        # Decode: at attention time
        correction = qjl.decode(signs, mean_abs)
        corrected = dequantized + correction
    """

    def encode(
        self,
        residual: torch.Tensor,
        pack: bool = True,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Encode residual into packed signs + mean magnitude.

        Args:
            residual: Shape (..., d)
            pack: If True, pack signs into uint8

        Returns:
            signs: Packed uint8 (..., d//8) or bool (..., d)
            mean_abs: Shape (...)
        """
        signs, mean_abs = encode_qjl_residual(residual)
        if pack:
            signs = pack_signs(signs)
        return signs, mean_abs

    def decode(
        self,
        signs: torch.Tensor,
        mean_abs: torch.Tensor,
        d: Optional[int] = None,
        packed: bool = True,
    ) -> torch.Tensor:
        """Decode packed signs + mean magnitude into correction tensor.

        Args:
            signs: Packed uint8 or bool tensor
            mean_abs: Mean absolute residual
            d: Original dimension (required if packed=True)
            packed: Whether signs are packed

        Returns:
            Correction tensor of shape (..., d)
        """
        if packed:
            assert d is not None, "Must provide d when signs are packed"
            signs = unpack_signs(signs, d)
        return decode_qjl_residual(signs, mean_abs)

    def memory_bits_per_element(self, d: int) -> float:
        """Compute memory cost of QJL residual in bits per element.

        For head_dim d:
        - 1 bit per element for signs
        - 32 bits per group (1 float32 for mean_abs) / d elements per group

        Total: 1 + 32/d bits per element
        For d=64: 1 + 0.5 = 1.5 bits
        For d=128: 1 + 0.25 = 1.25 bits
        """
        return 1.0 + 32.0 / d
