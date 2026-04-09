"""Experiment runner — wraps evaluate_kv_quality with all 6 knobs."""

import sys
import os
import time
import torch
from typing import List, Dict

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from eval.real_model_eval import capture_kv_activations, evaluate_kv_quality
from autoresearch.config import QuantConfig


class ExperimentRunner:
    def __init__(self, model_name: str = "Qwen/Qwen2.5-0.5B-Instruct", device: str = "cpu"):
        from transformers import AutoModelForCausalLM, AutoTokenizer

        print(f"[Runner] Loading {model_name}...")
        t0 = time.time()
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name, trust_remote_code=True, torch_dtype=torch.float32
        ).to(device)
        self.device = device
        print(f"[Runner] Model loaded in {time.time() - t0:.1f}s")

        config = self.model.config
        self.n_layers = config.num_hidden_layers
        self.head_dim = config.hidden_size // config.num_attention_heads

        prompts = [
            "The capital of France is Paris. The Eiffel Tower was built in 1889 for the World's Fair.",
            "In quantum mechanics, the Schrödinger equation describes how the quantum state changes with time.",
            "Bitcoin is a decentralized digital currency that uses a peer-to-peer network.",
            "def fibonacci(n):\n    if n <= 1:\n        return n\n    return fibonacci(n-1) + fibonacci(n-2)",
            "The quick brown fox jumps over the lazy dog. Pack my box with five dozen liquor jugs.",
        ]

        print(f"[Runner] Capturing activations from {len(prompts)} prompts...")
        t0 = time.time()
        self.activations = capture_kv_activations(self.model, self.tokenizer, prompts, device)
        print(f"[Runner] Captured in {time.time() - t0:.1f}s")

    def run_config(self, config: QuantConfig) -> Dict:
        k_bits_per_layer = config.get_k_bits_per_layer(self.n_layers)
        v_bits_per_layer = config.get_v_bits_per_layer(self.n_layers)

        results = evaluate_kv_quality(
            self.activations,
            bits=config.k_bits,
            v_bits=config.v_bits,
            use_qjl=config.use_qjl,
            rotation=config.rotation,
            group_size=config.group_size,
            k_bits_per_layer=k_bits_per_layer,
            v_bits_per_layer=v_bits_per_layer,
            seed=config.seed,
        )
        results["config_name"] = config.name
        results["rotation"] = config.rotation
        results["group_size"] = config.group_size
        results["layer_policy"] = config.layer_policy
        results["effective_k_bits"] = config.effective_k_bits(self.head_dim)
        results["effective_v_bits"] = config.effective_v_bits(self.head_dim)
        results["total_bits_per_kv_pair"] = config.total_bits_per_kv_pair(self.head_dim)
        return results

    def run_all(self, configs: List[QuantConfig]) -> List[Dict]:
        results = []
        t_start = time.time()
        for i, config in enumerate(configs):
            t0 = time.time()
            r = self.run_config(config)
            elapsed = time.time() - t0

            # Progress and ETA
            done = i + 1
            total_elapsed = time.time() - t_start
            avg_per = total_elapsed / done
            eta = avg_per * (len(configs) - done)
            eta_str = f"{eta/60:.0f}m" if eta > 60 else f"{eta:.0f}s"

            print(f"  [{done}/{len(configs)}] {config.name:<30} "
                  f"K={r['k_cosine_sim_mean']:.4f} V={r['v_cosine_sim_mean']:.4f} "
                  f"bits={r['total_bits_per_kv_pair']:.1f} ({elapsed:.1f}s, ETA {eta_str})")
            results.append(r)
        return results
