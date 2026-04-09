"""
Needle-in-a-haystack test for LeanKV.

Inserts a specific fact ("needle") into a long context of padding text
("haystack"), then asks the model to retrieve it. Tests whether KV cache
quantization destroys the model's ability to attend to specific tokens
in long contexts.

Test setup:
  - Needle: "The secret code is DIAMOND-7742."
  - Haystack: Wikipedia-style filler text repeated to fill context
  - Query: "What is the secret code?"
  - Pass criteria: model output contains "DIAMOND-7742"

Tests FP16 baseline vs 3-bit + QJL quantized K/V.

Since we can't easily swap the internal cache in this prototype,
we test the attention mechanism directly:
  1. Run model to get K/V cache
  2. Quantize the K/V
  3. Check if the attention scores still peak at the needle position
  4. Also run generation to see if the model retrieves the needle
"""

import torch
import torch.nn.functional as F
import numpy as np
import sys
import os
import time
from typing import Dict, List, Tuple

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from turboquant.quantizer import TurboQuantizer
from turboquant.lloyd_max import quantize_scalar, dequantize_scalar


# Filler text to pad the context
HAYSTACK_FILLER = """The history of mathematics can be seen as an ever-increasing series of abstractions. Evolutionarily speaking, the first abstraction to take place, which is shared by many animals, was probably that of numbers: the realization that a collection of two apples and a collection of two oranges have something in common, namely the quantity of their members. As evidenced by tallies found on bone, in addition to recognizing how to count physical objects, prehistoric peoples may have also known how to count abstract quantities, like time, days, seasons, or years. Evidence for more complex mathematics does not appear until around 3000 BC, when the Babylonians and Egyptians began using arithmetic, algebra, and geometry for taxation, commerce, and astronomical observations. The earliest mathematical texts available are from Mesopotamia and Egypt. The Rhind Mathematical Papyrus is one of the best known examples of Egyptian mathematics. """

NEEDLE = "The secret code is DIAMOND-7742."
QUERY = "What is the secret code mentioned in the text above?"


def build_haystack(tokenizer, target_tokens: int = 200, needle_position: float = 0.5) -> Tuple[str, int]:
    """Build a haystack text with a needle inserted at the specified position.

    Args:
        tokenizer: Tokenizer for measuring token count
        target_tokens: Approximate total token count
        needle_position: Where to insert needle (0.0 = beginning, 1.0 = end)

    Returns:
        Tuple of (full_text, needle_token_position)
    """
    # Build enough filler text
    filler_tokens = tokenizer(HAYSTACK_FILLER, return_tensors="pt")["input_ids"].shape[1]
    repetitions = max(1, (target_tokens // filler_tokens) + 1)
    full_filler = (HAYSTACK_FILLER + "\n\n") * repetitions

    # Split at needle position
    filler_enc = tokenizer(full_filler, return_tensors="pt")["input_ids"]
    total_filler_tokens = min(filler_enc.shape[1], target_tokens - 20)  # leave room for needle

    # Decode back to get clean text at right length
    filler_text = tokenizer.decode(filler_enc[0, :total_filler_tokens], skip_special_tokens=True)

    # Insert needle
    insert_char = int(len(filler_text) * needle_position)
    # Find a good insertion point (paragraph break)
    search_start = max(0, insert_char - 100)
    search_end = min(len(filler_text), insert_char + 100)
    paragraph_break = filler_text.find("\n\n", search_start, search_end)
    if paragraph_break == -1:
        paragraph_break = insert_char

    text_before = filler_text[:paragraph_break]
    text_after = filler_text[paragraph_break:]
    full_text = text_before + "\n\n" + NEEDLE + "\n\n" + text_after

    # Find needle token position
    before_tokens = tokenizer(text_before, return_tensors="pt")["input_ids"].shape[1]

    return full_text, before_tokens


def test_attention_to_needle(
    model,
    tokenizer,
    bits: int = 3,
    use_qjl: bool = True,
    context_tokens: int = 200,
    needle_position: float = 0.5,
    device: str = "cpu",
) -> Dict[str, any]:
    """Test whether attention still focuses on the needle after KV quantization.

    Args:
        model: HuggingFace causal LM
        tokenizer: Tokenizer
        bits: Quantization bits
        use_qjl: Enable QJL
        context_tokens: Total context length in tokens
        needle_position: Needle position (0-1)
        device: torch device

    Returns:
        Dict with test results
    """
    config = model.config
    n_layers = config.num_hidden_layers
    head_dim = config.hidden_size // config.num_attention_heads
    n_kv_heads = config.num_key_value_heads
    n_heads = config.num_attention_heads

    # Build context with needle
    context, needle_start_tok = build_haystack(tokenizer, context_tokens, needle_position)

    # Add the query
    full_prompt = context + "\n\n" + QUERY
    inputs = tokenizer(full_prompt, return_tensors="pt").to(device)
    seq_len = inputs["input_ids"].shape[1]

    # Tokenize needle alone to know its token span
    needle_tokens = tokenizer(NEEDLE, return_tensors="pt")["input_ids"]
    needle_len = needle_tokens.shape[1]
    needle_range = range(needle_start_tok, needle_start_tok + needle_len)

    model.eval()
    with torch.no_grad():
        outputs = model(
            **inputs,
            output_hidden_states=True,
            output_attentions=True,
            use_cache=True,
        )

    cache = outputs.past_key_values
    attentions = outputs.attentions  # tuple of (batch, n_heads, seq, seq) per layer

    # For each layer, check attention from query tokens to needle tokens
    # Query tokens are the last few tokens (the question)
    query_start = seq_len - tokenizer(QUERY, return_tensors="pt")["input_ids"].shape[1]

    results = {
        "context_tokens": seq_len,
        "needle_position": needle_position,
        "needle_token_range": (needle_start_tok, needle_start_tok + needle_len),
        "query_token_range": (query_start, seq_len),
    }

    # FP16 attention to needle
    fp16_needle_attention = []
    for layer_idx in range(n_layers):
        attn = attentions[layer_idx]  # (batch, n_heads, seq, seq)
        # Attention from query tokens to needle tokens
        query_to_needle = attn[0, :, query_start:, needle_start_tok:needle_start_tok + needle_len]
        avg_attention = query_to_needle.mean().item()
        fp16_needle_attention.append(avg_attention)

    results["fp16_needle_attention_by_layer"] = fp16_needle_attention
    results["fp16_needle_attention_mean"] = np.mean(fp16_needle_attention)

    # Now simulate quantized attention
    k_quantizer = TurboQuantizer(
        n_layers=n_layers, head_dim=head_dim, bits=bits,
        rotation_strategy="randomized_hadamard", use_qjl=use_qjl, seed=42,
    )

    quant_needle_attention = []
    attn_score_cos_sims = []

    for layer_idx in range(n_layers):
        K = cache.layers[layer_idx].keys  # (batch, n_kv_heads, seq, head_dim)
        V = cache.layers[layer_idx].values

        # Get Q from hidden states
        attn_module = model.model.layers[layer_idx].self_attn
        # Use K as Q proxy (both have RoPE applied, same space)
        # This measures whether quantized K preserves the dot-product structure
        Q = K  # (batch, n_kv_heads, seq, head_dim)

        scale = 1.0 / (head_dim ** 0.5)

        # FP16 reference attention scores (K^T K self-similarity)
        K_expanded = K
        scores_ref = torch.softmax(Q @ K_expanded.transpose(-2, -1) * scale, dim=-1)

        # Quantized attention scores
        k_q = k_quantizer.quantize(K, layer_idx)
        Q_rot = k_quantizer.quantize_for_attention(Q, layer_idx)
        K_rot_recon = k_quantizer.dequantize(k_q, layer_idx, apply_inverse_rot=False)
        K_rot_expanded = K_rot_recon
        scores_quant = torch.softmax(Q_rot @ K_rot_expanded.transpose(-2, -1) * scale, dim=-1)

        # Attention from query tokens to needle tokens
        query_to_needle_quant = scores_quant[0, :, query_start:, needle_start_tok:needle_start_tok + needle_len]
        avg_attn_quant = query_to_needle_quant.mean().item()
        quant_needle_attention.append(avg_attn_quant)

        # Cosine similarity of full attention score matrices
        cos_sim = F.cosine_similarity(
            scores_ref[0, :, query_start:, :].reshape(-1),
            scores_quant[0, :, query_start:, :].reshape(-1),
            dim=0
        ).item()
        attn_score_cos_sims.append(cos_sim)

    results["quant_needle_attention_by_layer"] = quant_needle_attention
    results["quant_needle_attention_mean"] = np.mean(quant_needle_attention)
    results["attn_score_cosine_sim_mean"] = np.mean(attn_score_cos_sims)
    results["attn_score_cosine_sim_min"] = np.min(attn_score_cos_sims)

    # Needle attention preservation ratio
    ratio = np.mean(quant_needle_attention) / max(np.mean(fp16_needle_attention), 1e-10)
    results["needle_attention_preservation"] = ratio

    return results


def test_generation(
    model,
    tokenizer,
    context_tokens: int = 200,
    needle_position: float = 0.5,
    max_new_tokens: int = 50,
    device: str = "cpu",
) -> Dict[str, any]:
    """Test if model can retrieve the needle via generation.

    This tests FP16 only (no cache swap). The attention analysis above
    tests the quantization impact.
    """
    context, _ = build_haystack(tokenizer, context_tokens, needle_position)
    full_prompt = context + "\n\n" + QUERY + "\nAnswer:"

    inputs = tokenizer(full_prompt, return_tensors="pt").to(device)

    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            temperature=1.0,
        )

    generated = tokenizer.decode(output_ids[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
    found = "DIAMOND-7742" in generated

    return {
        "generated_text": generated.strip(),
        "needle_found": found,
    }


def main():
    from transformers import AutoModelForCausalLM, AutoTokenizer

    model_name = "Qwen/Qwen2.5-0.5B-Instruct"
    device = "cpu"

    print("=" * 70)
    print("LeanKV Needle-in-a-Haystack Test")
    print(f"Model: {model_name}")
    print(f"Needle: \"{NEEDLE}\"")
    print("=" * 70)

    print("\n[1/4] Loading model and tokenizer...")
    t0 = time.time()
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_name, trust_remote_code=True, torch_dtype=torch.float32,
        attn_implementation="eager",
    ).to(device)
    print(f"       Loaded in {time.time() - t0:.1f}s")

    # Test 1: FP16 generation baseline
    print("\n[2/4] FP16 generation baseline...")
    t0 = time.time()
    gen_result = test_generation(
        model, tokenizer,
        context_tokens=200,
        needle_position=0.5,
        device=device,
    )
    print(f"       Generated: \"{gen_result['generated_text'][:100]}\"")
    print(f"       Needle found: {'PASS' if gen_result['needle_found'] else 'FAIL'}")
    print(f"       ({time.time() - t0:.1f}s)")

    # Test 2: Attention analysis at different bit-widths
    print("\n[3/4] Attention-to-needle analysis...")

    configs = [
        {"bits": 3, "use_qjl": False, "label": "3-bit"},
        {"bits": 3, "use_qjl": True,  "label": "3-bit + QJL"},
        {"bits": 4, "use_qjl": False, "label": "4-bit"},
        {"bits": 4, "use_qjl": True,  "label": "4-bit + QJL"},
    ]

    print(f"\n  {'Config':<16} {'FP16 attn':>10} {'Quant attn':>11} {'Preserve':>9} {'Attn cos':>9}")
    print(f"  {'-'*58}")

    for cfg in configs:
        t0 = time.time()
        result = test_attention_to_needle(
            model, tokenizer,
            bits=cfg["bits"],
            use_qjl=cfg["use_qjl"],
            context_tokens=200,
            needle_position=0.5,
            device=device,
        )
        fp16_attn = result["fp16_needle_attention_mean"]
        quant_attn = result["quant_needle_attention_mean"]
        preserve = result["needle_attention_preservation"]
        attn_cos = result["attn_score_cosine_sim_mean"]
        print(f"  {cfg['label']:<16} {fp16_attn:.6f}   {quant_attn:.6f}   "
              f"{preserve:.4f}    {attn_cos:.6f}  ({time.time() - t0:.1f}s)")

    # Test 3: Different needle positions
    print(f"\n[4/4] Needle position sweep (3-bit + QJL)...")
    positions = [0.1, 0.3, 0.5, 0.7, 0.9]

    print(f"\n  {'Position':>8} {'FP16 attn':>10} {'Quant attn':>11} {'Preserve':>9} {'Attn cos':>9}")
    print(f"  {'-'*52}")

    for pos in positions:
        result = test_attention_to_needle(
            model, tokenizer,
            bits=3, use_qjl=True,
            context_tokens=200,
            needle_position=pos,
            device=device,
        )
        fp16_attn = result["fp16_needle_attention_mean"]
        quant_attn = result["quant_needle_attention_mean"]
        preserve = result["needle_attention_preservation"]
        attn_cos = result["attn_score_cosine_sim_mean"]
        print(f"  {pos:>8.1f} {fp16_attn:.6f}   {quant_attn:.6f}   "
              f"{preserve:.4f}    {attn_cos:.6f}")

    print(f"\n{'='*70}")
    print("DONE — Needle-in-a-haystack test complete.")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()
