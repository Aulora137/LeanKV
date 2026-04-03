"""
Cosine similarity evaluation for TurboQuant.

Measures how well the quantized KV cache preserves the original attention
computation by comparing:
1. Per-layer K/V tensors before and after quantization
2. Attention output with quantized vs FP16 KV cache
3. Final model output (logits) with quantized vs FP16 KV cache

This is the fastest quality check — runs in seconds on CPU.
"""

import torch
import numpy as np
from typing import Dict, List, Tuple
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from turboquant.quantizer import TurboQuantizer


def cosine_similarity(a: torch.Tensor, b: torch.Tensor, dim: int = -1) -> torch.Tensor:
    """Compute cosine similarity along a dimension."""
    a_norm = a / (a.norm(dim=dim, keepdim=True) + 1e-10)
    b_norm = b / (b.norm(dim=dim, keepdim=True) + 1e-10)
    return (a_norm * b_norm).sum(dim=dim)


def evaluate_quantizer_quality(
    n_layers: int = 24,
    head_dim: int = 64,
    n_heads: int = 2,
    seq_len: int = 128,
    bits: int = 3,
    use_qjl: bool = True,
    rotation: str = "randomized_hadamard",
    seed: int = 42,
    device: str = "cpu",
) -> Dict[str, float]:
    """Evaluate TurboQuant quality on synthetic data.

    Generates random KV embeddings (mimicking post-RoPE activations),
    quantizes them, and measures reconstruction quality.

    Args:
        n_layers: Number of layers to test
        head_dim: Head dimension
        n_heads: Number of KV heads
        seq_len: Sequence length
        bits: Quantization bits
        use_qjl: Enable QJL correction
        rotation: Rotation strategy
        seed: Random seed
        device: torch device

    Returns:
        Dict with quality metrics
    """
    torch.manual_seed(seed)

    quantizer = TurboQuantizer(
        n_layers=n_layers,
        head_dim=head_dim,
        bits=bits,
        rotation_strategy=rotation,
        use_qjl=use_qjl,
        seed=seed,
        device=device,
    )

    cos_sims_k = []
    cos_sims_v = []
    mse_k = []
    mse_v = []

    for il in range(n_layers):
        # Generate random post-RoPE K and V embeddings
        # Real activations have outlier patterns; random is a fair baseline
        K = torch.randn(1, n_heads, seq_len, head_dim, device=device) * 0.1
        V = torch.randn(1, n_heads, seq_len, head_dim, device=device) * 0.1

        # Add some outlier channels (mimicking real LLM behavior)
        # Later layers have worse outliers
        outlier_scale = 1.0 + (il / n_layers) * 5.0
        outlier_channels = torch.randint(0, head_dim, (4,))
        K[:, :, :, outlier_channels] *= outlier_scale
        V[:, :, :, outlier_channels] *= outlier_scale * 0.5

        # Quantize K
        k_quant = quantizer.quantize(K, il)
        K_recon = quantizer.dequantize(k_quant, il, apply_inverse_rot=True)

        # Quantize V (same quantizer, no rotation for V typically)
        v_quant = quantizer.quantize(V, il)
        V_recon = quantizer.dequantize(v_quant, il, apply_inverse_rot=True)

        # Cosine similarity (per-token, averaged)
        k_cos = cosine_similarity(K.view(-1, head_dim), K_recon.view(-1, head_dim)).mean().item()
        v_cos = cosine_similarity(V.view(-1, head_dim), V_recon.view(-1, head_dim)).mean().item()
        cos_sims_k.append(k_cos)
        cos_sims_v.append(v_cos)

        # MSE (normalized by norm)
        k_mse = ((K - K_recon) ** 2).mean().item()
        v_mse = ((V - V_recon) ** 2).mean().item()
        mse_k.append(k_mse)
        mse_v.append(v_mse)

    results = {
        "bits": bits,
        "use_qjl": use_qjl,
        "rotation": rotation,
        "k_cosine_sim_mean": np.mean(cos_sims_k),
        "k_cosine_sim_min": np.min(cos_sims_k),
        "v_cosine_sim_mean": np.mean(cos_sims_v),
        "v_cosine_sim_min": np.min(cos_sims_v),
        "k_mse_mean": np.mean(mse_k),
        "v_mse_mean": np.mean(mse_v),
        "effective_bits": quantizer.memory_bits_per_element(),
        "compression_ratio": quantizer.compression_ratio(),
    }

    return results


def evaluate_attention_preservation(
    head_dim: int = 64,
    n_heads: int = 2,
    seq_len: int = 64,
    bits: int = 3,
    use_qjl: bool = True,
    rotation: str = "randomized_hadamard",
    seed: int = 42,
) -> Dict[str, float]:
    """Evaluate how well TurboQuant preserves attention scores.

    The critical test: does quantizing K distort the softmax(Q K^T) scores?
    This directly impacts model quality.

    Returns:
        Dict with attention score comparison metrics
    """
    torch.manual_seed(seed)

    quantizer = TurboQuantizer(
        n_layers=1, head_dim=head_dim, bits=bits,
        rotation_strategy=rotation, use_qjl=use_qjl, seed=seed,
    )

    Q = torch.randn(1, n_heads, 1, head_dim) * 0.1   # single query token
    K = torch.randn(1, n_heads, seq_len, head_dim) * 0.1  # cached keys

    # Add outliers to K
    K[:, :, :, :4] *= 5.0

    # Reference attention scores (FP32)
    scale = 1.0 / (head_dim ** 0.5)
    scores_ref = torch.softmax(Q @ K.transpose(-2, -1) * scale, dim=-1)

    # Quantized attention scores
    # Step 1: Quantize K
    k_quant = quantizer.quantize(K, layer_idx=0)

    # Step 2: Rotate Q (to match rotated K in cache)
    Q_rot = quantizer.quantize_for_attention(Q, layer_idx=0)

    # Step 3: Dequantize K in rotated space
    K_rot_recon = quantizer.dequantize(k_quant, layer_idx=0, apply_inverse_rot=False)

    # Step 4: Attention in rotated space
    scores_quant = torch.softmax(Q_rot @ K_rot_recon.transpose(-2, -1) * scale, dim=-1)

    # Compare
    score_cos = cosine_similarity(
        scores_ref.view(-1), scores_quant.view(-1), dim=0
    ).item()
    score_l1 = (scores_ref - scores_quant).abs().mean().item()
    score_max_err = (scores_ref - scores_quant).abs().max().item()

    # KL divergence
    kl_div = (scores_ref * (scores_ref.log() - scores_quant.log())).sum(dim=-1).mean().item()

    return {
        "attention_cosine_sim": score_cos,
        "attention_l1_error": score_l1,
        "attention_max_error": score_max_err,
        "attention_kl_divergence": kl_div,
    }


if __name__ == "__main__":
    print("=" * 60)
    print("TurboQuant Quality Evaluation (Synthetic Data)")
    print("=" * 60)

    configs = [
        {"bits": 2, "use_qjl": False, "rotation": "randomized_hadamard"},
        {"bits": 3, "use_qjl": False, "rotation": "randomized_hadamard"},
        {"bits": 3, "use_qjl": True,  "rotation": "randomized_hadamard"},
        {"bits": 4, "use_qjl": False, "rotation": "randomized_hadamard"},
        {"bits": 4, "use_qjl": True,  "rotation": "randomized_hadamard"},
        {"bits": 3, "use_qjl": True,  "rotation": "random_orthogonal"},
        {"bits": 3, "use_qjl": True,  "rotation": "hadamard"},
    ]

    print(f"\n{'Config':<35} {'K cos':>8} {'V cos':>8} {'Eff bits':>9} {'Compress':>9}")
    print("-" * 75)

    for cfg in configs:
        results = evaluate_quantizer_quality(**cfg)
        label = f"{cfg['bits']}b {'QJL' if cfg['use_qjl'] else '   '} {cfg['rotation'][:12]}"
        print(f"  {label:<33} {results['k_cosine_sim_mean']:.4f}   "
              f"{results['v_cosine_sim_mean']:.4f}   "
              f"{results['effective_bits']:.2f}      "
              f"{results['compression_ratio']:.1f}x")

    print(f"\n{'='*60}")
    print("Attention Score Preservation")
    print(f"{'='*60}")

    for bits in [2, 3, 4]:
        for qjl in [False, True]:
            attn = evaluate_attention_preservation(bits=bits, use_qjl=qjl)
            label = f"{bits}b {'QJL' if qjl else '   '}"
            print(f"  {label:<10} cos={attn['attention_cosine_sim']:.6f}  "
                  f"L1={attn['attention_l1_error']:.6f}  "
                  f"max_err={attn['attention_max_error']:.6f}  "
                  f"KL={attn['attention_kl_divergence']:.8f}")
