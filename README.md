# LeanKV

**3-4 bit KV cache quantization for LLM inference via random rotation + Lloyd-Max quantization + QJL residual correction.**

Based on Google's TurboQuant (arXiv:2504.19874), PolarQuant, and QJL research.

## What This Does

Standard KV cache uses 16 bits per element (FP16). LeanKV compresses to 3.5 bits — a **4.6x memory reduction** — with quality-neutral results.

The algorithm:
1. **Random rotation** spreads activation outliers uniformly → approximate Gaussian distribution
2. **Lloyd-Max quantization** optimally maps to 3-4 bit codebook (MSE-optimal for Gaussian)
3. **QJL residual correction** stores 1-bit sign of residual → first-order error correction

Key property: **RoPE-invariant.** `(ΠRq)^T(ΠRk) = q^T R^T R k` since `Π^TΠ = I`.

## Quick Start

```bash
cd prototype
pip install -r requirements.txt

# Run core algorithm tests
python tests/test_rope_invariance.py

# Evaluate on Qwen 2.5-0.5B
python eval/run_eval.py --model Qwen/Qwen2.5-0.5B-Instruct --bits 3
```

## Project Structure

```
prototype/     Phase 0: Python validation (rotation, Lloyd-Max, QJL, HF cache)
cpp/           Phase 1: C++ integration with ik_llama.cpp
autoresearch/  Phase 2: Automated tuning (6 knobs, ~3000 configs)
cuda/          Phase 3: Fused CUDA kernels
metal/         Phase 3: Metal shaders for Apple Silicon
patches/       Patches against ik_llama.cpp upstream
```

## Related

- [LeanInfer](https://github.com/hchengit/LeanInfer) — our inference runtime (CPU/Metal/CUDA optimizations)
- [TurboQuant paper](https://arxiv.org/abs/2504.19874) — Google Research
- [QJL paper](https://arxiv.org/abs/2406.03482) — 1-bit quantized JL transform
- [PolarQuant paper](https://arxiv.org/abs/2502.02617) — polar coordinate quantization
