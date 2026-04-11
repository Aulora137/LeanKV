# LeanKV

**3-4 bit KV cache quantization for LLM inference with near-lossless quality.**

LeanKV implements TurboQuant-style KV cache compression: Hadamard rotation + Lloyd-Max optimal scalar quantization. Standard KV caches use 16 bits per element (FP16). LeanKV compresses to 3.5-4.5 bits -- a **75-81% memory reduction** -- with negligible quality loss.

## Results

Tested on Qwen 3.5-9B (Q4_K_M weights) with WikiText-2 perplexity:

| KV Config | Bits/element | PPL | Delta from F16 | KV Memory | Speed (M2) |
|-----------|-------------|-----|----------------|-----------|------------|
| F16/F16 | 16 | 7.173 | -- | 72 MiB | 10.95 tok/s |
| **TQ4/TQ4** | **4.5** | **7.189** | **+0.016** | **18 MiB (-75%)** | **10.87 tok/s (99%)** |
| **TQ3/TQ3** | **3.5** | **~7.22** | **~+0.05** | **14 MiB (-81%)** | **10.60 tok/s (97%)** |

TQ4 is essentially lossless. TQ3 uses optimal rounding (coordinate descent + least-squares scale) to achieve +0.05 PPL delta at 3-bit.

Cross-architecture validation on 6 models spanning Qwen, Gemma, and Llama families -- see [docs/RESULTS.md](docs/RESULTS.md) for full results.

## How It Works

1. **Hadamard rotation** -- Multiply each KV head by a Walsh-Hadamard matrix (O(d log d), no storage). Spreads outlier energy uniformly, making the distribution approximately Gaussian.
2. **Lloyd-Max quantization** -- Map each element to the nearest level in a pre-computed codebook optimized for N(0,1). TQ4: 16 levels, TQ3: 8 levels. Per-block scale factor `d` captures magnitude.
3. **Optimal rounding** (TQ3 only) -- Coordinate descent refinement: try adjacent codebook levels for each element, keep changes that reduce block MSE. 2 passes with least-squares optimal scale. Zero decode cost.

The rotation preserves attention scores: since both Q and K are rotated by the same orthogonal matrix, `(HQ)^T(HK) = Q^T H^T H K = Q^T K`.

## Architecture

```
LeanKV/                      This repo -- project docs, benchmarks, prototypes
  docs/PLAN.md               Project plan and trail log
  docs/RESULTS.md            Full benchmark results (all phases)
  docs/TQ3PLAN.md            TQ3 improvement plan
  scripts/                   Benchmark scripts and results
  prototype/                 Phase 0 Python prototype
  src/                       C unit tests for TQ codebooks

Lean_llama.cpp/              Implementation repo (fork of ik_llama.cpp)
  ggml/src/ggml-tq.c         TQ3/TQ4 quantize/dequantize + optimal rounding
  ggml/src/ggml-common.h      Block structs + codebook LUT tables
  ggml/src/ggml.c             Type traits registration
  ggml/src/iqk/               IQK optimized kernels (AVX2 + ARM NEON)
```

## Building

LeanKV's implementation lives in [Lean_llama.cpp](https://github.com/Aulora137/Lean_llama.cpp), a fork of [ik_llama.cpp](https://github.com/ikawrakow/ik_llama.cpp).

```bash
# Clone
git clone https://github.com/Aulora137/Lean_llama.cpp.git
cd Lean_llama.cpp

# Build
cmake -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j$(nproc)

# Download a model (e.g. via LM Studio, or manually)
# Any GGUF model with Q4_K_M weights works

# Run with TQ4 KV cache (recommended)
./build/bin/llama-cli -m <model.gguf> -ctk tq4_0 -ctv tq4_0 -c 4096 \
  -p "Hello, how are you?" -n 64

# Run with TQ3 KV cache (maximum compression)
./build/bin/llama-cli -m <model.gguf> -ctk tq3_0 -ctv tq3_0 -c 4096 \
  -p "Hello, how are you?" -n 64

# Perplexity benchmark
./build/bin/llama-perplexity -m <model.gguf> -ctk tq4_0 -ctv tq4_0 \
  -f <wikitext-2-raw/wiki.test.raw> -c 2048
```

### Platform Support

| Platform | TQ4 | TQ3 | IQK Acceleration |
|----------|-----|-----|-----------------|
| x86_64 (AVX2) | Yes | Yes | Full (FA + mul_mat) |
| Apple Silicon (NEON) | Yes | Yes | Full (FA + mul_mat) |
| CUDA / Metal | Not yet | Not yet | Planned |

## Key Flags

| Flag | Description |
|------|-------------|
| `-ctk tq4_0` | 4-bit TurboQuant keys |
| `-ctv tq4_0` | 4-bit TurboQuant values |
| `-ctk tq3_0` | 3-bit TurboQuant keys |
| `-ctv tq3_0` | 3-bit TurboQuant values |

Hadamard rotation is enabled automatically when using TQ types (`k_cache_hadam=1`).

## Documentation

- [PLAN.md](docs/PLAN.md) -- Project plan with math crash course and trail log
- [RESULTS.md](docs/RESULTS.md) -- Full benchmark results across all phases
- [TQ3PLAN.md](docs/TQ3PLAN.md) -- TQ3 improvement plan (optimal rounding, per-layer codebooks)

## References

1. Zandieh et al. "[TurboQuant: Online Vector Quantization with Near-optimal Distortion Rate](https://arxiv.org/abs/2504.19874)" (2025). Google Research.
2. Han et al. "[PolarQuant: Quantizing KV Caches with Polar Transformation](https://arxiv.org/abs/2502.02617)" (2025).
3. Lloyd. "Least squares quantization in PCM." IEEE Trans. Info. Theory (1982).

## License

MIT
