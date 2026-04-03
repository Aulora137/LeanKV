"""
Drop-in HuggingFace DynamicCache replacement with TurboQuant compression.

This module provides LeanKVCache, a subclass of transformers.DynamicCache
that transparently quantizes K/V embeddings as they're stored and dequantizes
them when read for attention computation.

Usage with HuggingFace models:
    from turboquant.kv_cache import LeanKVCache

    cache = LeanKVCache(
        n_layers=24, head_dim=64, n_kv_heads=2,
        k_bits=3, v_bits=4, use_qjl=True,
    )
    outputs = model.generate(input_ids, past_key_values=cache, ...)

Key design decisions:
1. K cache: rotate + Lloyd-Max + optional QJL (keys participate in softmax via dot product)
2. V cache: Lloyd-Max only, no rotation (values are weighted averages, more tolerant of noise)
3. Q rotation: applied at attention time, not cached
4. RoPE: already applied by the model before cache.update() is called
"""

import torch
from typing import Optional, Tuple, List, Dict, Any
from .quantizer import TurboQuantizer, QuantizedKV


class LeanKVCache:
    """LeanKV-compressed KV cache for HuggingFace transformers.

    Implements the same interface as transformers.DynamicCache so it can
    be passed as past_key_values to model.generate() or model.forward().

    Memory layout per layer:
      K cache: List of QuantizedKV (indices + QJL signs + norms)
      V cache: List of QuantizedKV (indices + norms, no rotation)

    Args:
        n_layers: Number of transformer layers
        head_dim: Attention head dimension
        n_kv_heads: Number of KV heads (for GQA models)
        k_bits: Quantization bits for keys (2, 3, or 4)
        v_bits: Quantization bits for values (2, 3, or 4)
        k_rotation: Rotation strategy for keys
        v_rotation: Rotation strategy for values ("none" = no rotation)
        use_qjl: Enable QJL residual correction for keys
        seed: Random seed
        device: torch device
    """

    def __init__(
        self,
        n_layers: int,
        head_dim: int,
        n_kv_heads: int = 1,
        k_bits: int = 3,
        v_bits: int = 4,
        k_rotation: str = "randomized_hadamard",
        v_rotation: str = "none",
        use_qjl: bool = True,
        seed: int = 42,
        device: str = "cpu",
    ):
        self.n_layers = n_layers
        self.head_dim = head_dim
        self.n_kv_heads = n_kv_heads

        # Key quantizer (with rotation + optional QJL)
        self.k_quantizer = TurboQuantizer(
            n_layers=n_layers,
            head_dim=head_dim,
            bits=k_bits,
            rotation_strategy=k_rotation,
            use_qjl=use_qjl,
            seed=seed,
            device=device,
        )

        # Value quantizer (typically no rotation, no QJL — values are more tolerant)
        if v_rotation == "none":
            self.v_quantizer = TurboQuantizer(
                n_layers=n_layers,
                head_dim=head_dim,
                bits=v_bits,
                rotation_strategy="hadamard",  # placeholder, won't be used
                use_qjl=False,
                seed=seed + 1000,
                device=device,
            )
            self.v_rotate = False
        else:
            self.v_quantizer = TurboQuantizer(
                n_layers=n_layers,
                head_dim=head_dim,
                bits=v_bits,
                rotation_strategy=v_rotation,
                use_qjl=False,
                seed=seed + 1000,
                device=device,
            )
            self.v_rotate = True

        # Storage: per-layer lists of QuantizedKV for K and V
        self.k_cache: List[List[QuantizedKV]] = [[] for _ in range(n_layers)]
        self.v_cache: List[List[QuantizedKV]] = [[] for _ in range(n_layers)]
        self._seen_tokens = 0

    def update(
        self,
        key_states: torch.Tensor,
        value_states: torch.Tensor,
        layer_idx: int,
        cache_kwargs: Optional[Dict[str, Any]] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Store new KV states and return full KV for attention.

        This is the main interface called by HuggingFace attention layers.

        Args:
            key_states: New keys, shape (batch, n_kv_heads, new_seq, head_dim)
                       Already has RoPE applied by the model.
            value_states: New values, shape (batch, n_kv_heads, new_seq, head_dim)
            layer_idx: Current layer index
            cache_kwargs: Additional kwargs (unused)

        Returns:
            Tuple of (all_keys, all_values) for attention computation.
            Keys are in rotated space; caller must rotate Q to match.
        """
        if layer_idx == 0:
            self._seen_tokens += key_states.shape[2]

        # Quantize and store new KV
        k_quantized = self.k_quantizer.quantize(key_states, layer_idx)
        self.k_cache[layer_idx].append(k_quantized)

        # For V: quantize without rotation
        v_quantized = self._quantize_values(value_states, layer_idx)
        self.v_cache[layer_idx].append(v_quantized)

        # Dequantize ALL cached KV for attention
        all_keys = self._get_all_keys(layer_idx)
        all_values = self._get_all_values(layer_idx)

        return all_keys, all_values

    def _quantize_values(
        self,
        v: torch.Tensor,
        layer_idx: int,
    ) -> QuantizedKV:
        """Quantize values (simpler than keys — no rotation needed typically)."""
        if self.v_rotate:
            return self.v_quantizer.quantize(v, layer_idx)
        else:
            # Direct quantization without rotation
            norms = v.norm(dim=-1)
            v_normalized = v / (norms.unsqueeze(-1) + 1e-10)
            indices = self.v_quantizer.levels  # use same codebook
            from .lloyd_max import quantize_scalar, dequantize_scalar
            idx = quantize_scalar(v_normalized, self.v_quantizer.boundaries)
            return QuantizedKV(
                indices=idx.to(torch.uint8),
                qjl_signs=None,
                qjl_mean_abs=None,
                norms=norms,
            )

    def _get_all_keys(self, layer_idx: int) -> torch.Tensor:
        """Dequantize and concatenate all cached keys for a layer.

        Keys remain in rotated space — Q must be rotated to match.
        """
        if not self.k_cache[layer_idx]:
            return None

        parts = []
        for qkv in self.k_cache[layer_idx]:
            # Dequantize in rotated space (no inverse rotation)
            k_approx = self.k_quantizer.dequantize(qkv, layer_idx, apply_inverse_rot=False)
            parts.append(k_approx)

        return torch.cat(parts, dim=2)  # concat along seq dim

    def _get_all_values(self, layer_idx: int) -> torch.Tensor:
        """Dequantize and concatenate all cached values for a layer.

        Values are returned in original space (inverse rotation applied if used).
        """
        if not self.v_cache[layer_idx]:
            return None

        parts = []
        for qkv in self.v_cache[layer_idx]:
            if self.v_rotate:
                v_approx = self.v_quantizer.dequantize(qkv, layer_idx, apply_inverse_rot=True)
            else:
                from .lloyd_max import dequantize_scalar
                v_approx = dequantize_scalar(qkv.indices.long(), self.v_quantizer.levels)
                v_approx = v_approx * qkv.norms.unsqueeze(-1)
            parts.append(v_approx)

        return torch.cat(parts, dim=2)

    def get_seq_length(self, layer_idx: int = 0) -> int:
        """Return total cached sequence length."""
        if not self.k_cache[layer_idx]:
            return 0
        return sum(qkv.indices.shape[2] for qkv in self.k_cache[layer_idx])

    def get_max_length(self) -> Optional[int]:
        """No max length constraint."""
        return None

    def get_usable_length(self, new_seq_length: int, layer_idx: int = 0) -> int:
        """Return usable length for attention."""
        return self.get_seq_length(layer_idx)

    @property
    def seen_tokens(self) -> int:
        return self._seen_tokens

    def memory_summary(self) -> dict:
        """Return memory usage summary."""
        k_bits = self.k_quantizer.memory_bits_per_element()
        v_bits = self.v_quantizer.memory_bits_per_element()
        total_tokens = self.get_seq_length()
        k_bytes = total_tokens * self.n_kv_heads * self.head_dim * k_bits / 8
        v_bytes = total_tokens * self.n_kv_heads * self.head_dim * v_bits / 8
        fp16_bytes = total_tokens * self.n_kv_heads * self.head_dim * 2 * 2  # K+V, 2 bytes each

        return {
            "k_bits_per_elem": k_bits,
            "v_bits_per_elem": v_bits,
            "total_tokens": total_tokens,
            "compressed_bytes": (k_bytes + v_bytes) * self.n_layers,
            "fp16_bytes": fp16_bytes * self.n_layers,
            "compression_ratio": fp16_bytes / max(k_bytes + v_bytes, 1),
        }

    def __repr__(self) -> str:
        return (f"LeanKVCache(layers={self.n_layers}, head_dim={self.head_dim}, "
                f"k={self.k_quantizer}, v_rotate={self.v_rotate}, "
                f"tokens={self.get_seq_length()})")
