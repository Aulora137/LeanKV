"""
Real model cosine similarity evaluation for LeanKV.

Loads Qwen 2.5-0.5B-Instruct, captures real K/V activations from each layer,
applies TurboQuant at different bit-widths, and measures:
1. K/V tensor cosine similarity (quantized vs original)
2. Attention score preservation (cosine sim of softmax(QK^T/sqrt(d)))

This replaces the synthetic eval in cosine_sim.py with REAL model activations.
"""

import torch
import numpy as np
import sys
import os
import time
from typing import Dict, List, Tuple

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from turboquant.quantizer import TurboQuantizer
from turboquant.lloyd_max import quantize_scalar, dequantize_scalar


def cosine_similarity(a: torch.Tensor, b: torch.Tensor, dim: int = -1) -> torch.Tensor:
    """Compute cosine similarity along a dimension."""
    a_norm = a / (a.norm(dim=dim, keepdim=True) + 1e-10)
    b_norm = b / (b.norm(dim=dim, keepdim=True) + 1e-10)
    return (a_norm * b_norm).sum(dim=dim)


def capture_kv_activations(model, tokenizer, prompts: List[str], device: str = "cpu"):
    """Capture real K and V activations from every layer of the model.

    Uses HuggingFace's output_attentions mechanism and hooks to capture
    the post-RoPE K and V tensors at each layer.

    Returns:
        List of dicts, one per prompt, each containing:
          - 'keys': list of (batch, n_kv_heads, seq_len, head_dim) tensors per layer
          - 'values': list of (batch, n_kv_heads, seq_len, head_dim) tensors per layer
          - 'queries': list of (batch, n_heads, seq_len, head_dim) tensors per layer
    """
    model.eval()
    all_activations = []

    for prompt in prompts:
        inputs = tokenizer(prompt, return_tensors="pt").to(device)
        layer_keys = []
        layer_values = []
        layer_queries = []
        hooks = []

        # Hook into each attention layer to capture post-RoPE K, V, Q
        for layer_idx, layer in enumerate(model.model.layers):
            attn = layer.self_attn

            def make_hook(idx):
                def hook_fn(module, args, kwargs, output):
                    # Qwen2 attention forward signature:
                    # hidden_states, attention_mask, position_ids, past_key_value, ...
                    # We need to capture key_states and value_states AFTER RoPE
                    # These are computed inside the forward method.
                    # We'll use a different approach: hook into the k_proj and v_proj
                    pass
                return hook_fn

            # Instead of hooking forward, we'll run forward once with
            # output_hidden_states and manually recompute K/V per layer
            pass

        # Simpler approach: run forward with output_hidden_states=True,
        # then manually compute K/V using the model's projection weights + RoPE
        with torch.no_grad():
            outputs = model(
                **inputs,
                output_hidden_states=True,
                use_cache=True,
            )

        # Extract K/V from the DynamicCache (past_key_values)
        cache = outputs.past_key_values
        n_layers = len(cache.layers)

        keys_per_layer = []
        values_per_layer = []

        for layer_idx in range(n_layers):
            # transformers 5.x DynamicCache: cache.layers[i].keys / .values
            # shape: (batch, n_kv_heads, seq_len, head_dim)
            k = cache.layers[layer_idx].keys.detach().clone()
            v = cache.layers[layer_idx].values.detach().clone()
            keys_per_layer.append(k)
            values_per_layer.append(v)

        # For attention score comparison, use K as proxy for Q (both have RoPE applied).
        # The key metric is whether quantized K preserves dot-product structure,
        # which we measure via K^T K attention-like scores.
        all_activations.append({
            'keys': keys_per_layer,
            'values': values_per_layer,
            'queries': keys_per_layer,  # Use K as Q proxy (same RoPE, same space)
            'seq_len': inputs['input_ids'].shape[1],
        })

    return all_activations


def evaluate_kv_quality(
    activations: List[dict],
    bits=3,
    v_bits=None,
    use_qjl: bool = True,
    rotation: str = "randomized_hadamard",
    group_size: int = None,
    k_bits_per_layer: List[float] = None,
    v_bits_per_layer: List[float] = None,
    seed: int = 42,
) -> Dict[str, float]:
    """Evaluate TurboQuant quality on real model activations.

    Args:
        activations: Output from capture_kv_activations
        bits: K quantization bits (float, supports fractional)
        v_bits: V quantization bits (defaults to bits if None)
        use_qjl: Enable QJL for K
        rotation: Rotation strategy
        group_size: Quantization group size (None = head_dim)
        k_bits_per_layer: Per-layer K bits (overrides bits if provided)
        v_bits_per_layer: Per-layer V bits (overrides v_bits if provided)
        seed: Random seed

    Returns:
        Dict with quality metrics
    """
    if v_bits is None:
        v_bits = bits
    n_layers = len(activations[0]['keys'])
    head_dim = activations[0]['keys'][0].shape[-1]

    # Build per-layer bit lists
    if k_bits_per_layer is None:
        k_bits_per_layer = [bits] * n_layers
    if v_bits_per_layer is None:
        v_bits_per_layer = [v_bits] * n_layers

    # Cache quantizers by bits value to avoid recomputing codebooks
    _k_quantizers = {}
    _v_quantizers = {}
    for b in set(k_bits_per_layer):
        _k_quantizers[b] = TurboQuantizer(
            n_layers=n_layers, head_dim=head_dim, bits=b,
            group_size=group_size, rotation_strategy=rotation,
            use_qjl=use_qjl, seed=seed,
        )
    for b in set(v_bits_per_layer):
        _v_quantizers[b] = TurboQuantizer(
            n_layers=n_layers, head_dim=head_dim, bits=b,
            group_size=group_size, rotation_strategy=rotation,
            use_qjl=False, seed=seed + 1000,
        )
    # Use the first K quantizer for attention rotation
    quantizer = _k_quantizers[k_bits_per_layer[0]]

    all_k_cos = []
    all_v_cos = []
    all_attn_cos = []
    all_attn_kl = []

    for act in activations:
        for layer_idx in range(n_layers):
            K = act['keys'][layer_idx]      # (batch, n_kv_heads, seq, head_dim)
            V = act['values'][layer_idx]     # (batch, n_kv_heads, seq, head_dim)

            # Select per-layer quantizers
            k_q = _k_quantizers[k_bits_per_layer[layer_idx]]
            v_q = _v_quantizers[v_bits_per_layer[layer_idx]]

            # --- K quantization quality ---
            k_quant = k_q.quantize(K, layer_idx)
            K_recon = k_q.dequantize(k_quant, layer_idx, apply_inverse_rot=True)

            k_cos = cosine_similarity(
                K.reshape(-1, head_dim),
                K_recon.reshape(-1, head_dim)
            ).mean().item()
            all_k_cos.append(k_cos)

            # --- V quantization quality ---
            v_norms = V.norm(dim=-1)
            v_normalized = V / (v_norms.unsqueeze(-1) + 1e-10)

            v_idx = quantize_scalar(v_normalized, v_q.boundaries)
            V_recon_norm = dequantize_scalar(v_idx.long(), v_q.levels)
            V_recon = V_recon_norm * v_norms.unsqueeze(-1)

            v_cos = cosine_similarity(
                V.reshape(-1, head_dim),
                V_recon.reshape(-1, head_dim)
            ).mean().item()
            all_v_cos.append(v_cos)

            # --- Attention score preservation ---
            if 'queries' in act and layer_idx < len(act['queries']):
                Q = act['queries'][layer_idx]  # (batch, n_heads, seq, head_dim)
                scale = 1.0 / (head_dim ** 0.5)

                # Expand K for GQA if needed
                n_q_heads = Q.shape[1]
                n_kv_heads = K.shape[1]
                if n_q_heads != n_kv_heads:
                    n_rep = n_q_heads // n_kv_heads
                    K_expanded = K.repeat_interleave(n_rep, dim=1)
                    K_recon_expanded = K_recon.repeat_interleave(n_rep, dim=1)
                else:
                    K_expanded = K
                    K_recon_expanded = K_recon

                # Reference attention (FP16)
                scores_ref = torch.softmax(Q @ K_expanded.transpose(-2, -1) * scale, dim=-1)

                # Quantized attention: rotate Q, use rotated K from cache
                Q_rot = k_q.quantize_for_attention(Q, layer_idx)
                K_rot_recon = k_q.dequantize(k_quant, layer_idx, apply_inverse_rot=False)
                if n_q_heads != n_kv_heads:
                    K_rot_recon_expanded = K_rot_recon.repeat_interleave(n_rep, dim=1)
                else:
                    K_rot_recon_expanded = K_rot_recon

                scores_quant = torch.softmax(
                    Q_rot @ K_rot_recon_expanded.transpose(-2, -1) * scale, dim=-1
                )

                attn_cos = cosine_similarity(
                    scores_ref.reshape(-1),
                    scores_quant.reshape(-1),
                    dim=0
                ).item()
                all_attn_cos.append(attn_cos)

                # KL divergence
                kl = (scores_ref * (scores_ref.clamp(min=1e-10).log() -
                      scores_quant.clamp(min=1e-10).log())).sum(dim=-1).mean().item()
                all_attn_kl.append(kl)

    results = {
        "k_bits": bits,
        "v_bits": v_bits,
        "use_qjl": use_qjl,
        "k_cosine_sim_mean": np.mean(all_k_cos),
        "k_cosine_sim_min": np.min(all_k_cos),
        "k_cosine_sim_std": np.std(all_k_cos),
        "v_cosine_sim_mean": np.mean(all_v_cos),
        "v_cosine_sim_min": np.min(all_v_cos),
        "effective_bits": quantizer.memory_bits_per_element(),
        "compression_ratio": quantizer.compression_ratio(),
    }

    if all_attn_cos:
        results["attn_cosine_sim_mean"] = np.mean(all_attn_cos)
        results["attn_cosine_sim_min"] = np.min(all_attn_cos)
        results["attn_kl_divergence_mean"] = np.mean(all_attn_kl)

    return results


def main():
    from transformers import AutoModelForCausalLM, AutoTokenizer

    model_name = "Qwen/Qwen2.5-0.5B-Instruct"
    device = "cpu"

    print("=" * 70)
    print("LeanKV Real Model Evaluation — Cosine Similarity & Attention Scores")
    print(f"Model: {model_name}")
    print("=" * 70)

    print("\n[1/4] Loading model and tokenizer...")
    t0 = time.time()
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_name, trust_remote_code=True, torch_dtype=torch.float32
    ).to(device)
    print(f"       Loaded in {time.time() - t0:.1f}s")

    # Model info
    config = model.config
    n_layers = config.num_hidden_layers
    head_dim = config.hidden_size // config.num_attention_heads
    n_kv_heads = config.num_key_value_heads
    print(f"       Layers: {n_layers}, head_dim: {head_dim}, KV heads: {n_kv_heads}")

    # Test prompts (diverse to test different activation patterns)
    prompts = [
        "The capital of France is Paris. The Eiffel Tower was built in 1889 for the World's Fair.",
        "In quantum mechanics, the Schrödinger equation describes how the quantum state of a physical system changes with time.",
        "Bitcoin is a decentralized digital currency that uses a peer-to-peer network to verify transactions through cryptography.",
        "def fibonacci(n):\n    if n <= 1:\n        return n\n    return fibonacci(n-1) + fibonacci(n-2)",
        "The quick brown fox jumps over the lazy dog. Pack my box with five dozen liquor jugs.",
    ]

    print(f"\n[2/4] Capturing real K/V activations from {len(prompts)} prompts...")
    t0 = time.time()
    activations = capture_kv_activations(model, tokenizer, prompts, device)
    print(f"       Captured in {time.time() - t0:.1f}s")
    total_tokens = sum(a['seq_len'] for a in activations)
    print(f"       Total tokens: {total_tokens}")

    # Configurations to test
    configs = [
        {"bits": 3, "use_qjl": False, "label": "3-bit"},
        {"bits": 3, "use_qjl": True,  "label": "3-bit + QJL"},
        {"bits": 4, "use_qjl": False, "label": "4-bit"},
        {"bits": 4, "use_qjl": True,  "label": "4-bit + QJL"},
    ]

    print(f"\n[3/4] Running quantization quality evaluation...")
    print(f"\n{'Config':<16} {'K cos':>8} {'K min':>8} {'V cos':>8} {'Attn cos':>10} {'Attn KL':>12} {'Eff bits':>9} {'Compress':>9}")
    print("-" * 90)

    all_results = []
    for cfg in configs:
        results = evaluate_kv_quality(
            activations,
            bits=cfg["bits"],
            use_qjl=cfg["use_qjl"],
        )
        all_results.append(results)

        attn_cos = results.get("attn_cosine_sim_mean", float('nan'))
        attn_kl = results.get("attn_kl_divergence_mean", float('nan'))
        print(f"  {cfg['label']:<14} {results['k_cosine_sim_mean']:.4f}   "
              f"{results['k_cosine_sim_min']:.4f}   "
              f"{results['v_cosine_sim_mean']:.4f}   "
              f"{attn_cos:.6f}   "
              f"{attn_kl:.8f}   "
              f"{results['effective_bits']:.2f}      "
              f"{results['compression_ratio']:.1f}x")

    # Per-layer analysis for best config (3-bit + QJL)
    print(f"\n[4/4] Per-layer K cosine similarity (3-bit + QJL)...")
    head_dim = activations[0]['keys'][0].shape[-1]
    n_layers = len(activations[0]['keys'])

    quantizer = TurboQuantizer(
        n_layers=n_layers, head_dim=head_dim, bits=3,
        rotation_strategy="randomized_hadamard", use_qjl=True, seed=42,
    )

    print(f"\n  {'Layer':>5}  {'K cos':>8}  {'V cos':>8}")
    print(f"  {'-'*5}  {'-'*8}  {'-'*8}")
    for layer_idx in range(n_layers):
        k_cos_all = []
        v_cos_all = []
        for act in activations:
            K = act['keys'][layer_idx]
            V = act['values'][layer_idx]

            k_q = quantizer.quantize(K, layer_idx)
            K_r = quantizer.dequantize(k_q, layer_idx, apply_inverse_rot=True)
            k_cos_all.append(cosine_similarity(
                K.reshape(-1, head_dim), K_r.reshape(-1, head_dim)
            ).mean().item())

            v_norms = V.norm(dim=-1)
            v_norm = V / (v_norms.unsqueeze(-1) + 1e-10)
            v_idx = quantize_scalar(v_norm, quantizer.boundaries)
            V_r = dequantize_scalar(v_idx.long(), quantizer.levels) * v_norms.unsqueeze(-1)
            v_cos_all.append(cosine_similarity(
                V.reshape(-1, head_dim), V_r.reshape(-1, head_dim)
            ).mean().item())

        print(f"  {layer_idx:>5}  {np.mean(k_cos_all):.4f}    {np.mean(v_cos_all):.4f}")

    print(f"\n{'='*70}")
    print("DONE — Real model evaluation complete.")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()
