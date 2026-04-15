# Phase 7 Calibration — Initial K-Vector SVD Analysis

First data from the LeanKV Phase 7 calibration hook (see
`src/leankv-calib.{h,cpp}` in Lean_llama.cpp and `scripts/analyze_k_calib.py`
in LeanKV).

## How the data was collected

```bash
cd Lean_llama.cpp
LEANKV_CALIBRATION_DUMP=1 \
LEANKV_CALIBRATION_DUMP_PATH=/tmp/qwen3_4b_calib.bin \
./build/bin/llama-cli -m ~/models/Qwen3-4B-Q4_K_M.gguf \
    -ngl 0 -c 1024 -n 4 \
    -p "<75-token calibration prompt>"
```

Identical command for Mistral 7B. CPU backend (`-ngl 0`) so K-vectors are
captured in native fp32. 75-token prompt + 4 generated tokens = 608–688
K vectors per layer.

## How to analyze

```bash
/tmp/leankv-venv/bin/python3 scripts/analyze_k_calib.py \
    docs/phase7-calibration/qwen3_4b_calib.bin
```

## Findings (Apr 15, 2026)

### Side-by-side: Qwen3-4B vs Mistral 7B

| Metric                        | Qwen3-4B       | Mistral 7B     |
|-------------------------------|----------------|----------------|
| head_dim                      | 128            | 128            |
| n_layers                      | 36             | 32             |
| **Layer 0 r99**               | **3** / 128    | 46  / 128      |
| Median r99 (all layers)       | 100 / 128      | 110 / 128      |
| Median r95 (all layers)       | 60  / 128      | 75  / 128      |
| σ_min / σ_0 (typical)         | ~1×10⁻³        | ~5×10⁻²        |
| Layers flagged r99 < 112      | 36 / 36        | 25 / 32        |

Where `r99` = smallest rank capturing 99% of singular-value energy, etc.

### Observations

1. **Layer 0 is anomalously collapsed in Qwen3-4B (rank 3 at 99% energy).**
   Mistral's layer 0 is also rank-reduced (46) but nothing like this
   extreme. This likely reflects how Qwen3's first attention layer is
   doing something positional-encoding-like rather than exercising the
   full K-space.

2. **Qwen3-4B has a sharper spectral tail.** Median σ_min/σ_0 is ~1×10⁻³
   vs Mistral's ~5×10⁻² — a 50× difference. Even though Qwen3's "effective
   rank" is only modestly lower (100 vs 110), the bottom dims are much
   closer to zero relative to the top dims.

3. **Rank deficit alone does not break TQ.** Mistral is also rank-deficient
   on 25 of 32 layers by the r99 < 112 criterion, yet TQ3/TQ4 are
   near-lossless on Mistral. The combination of **rank deficit + steep
   spectral tail** is what hurts Qwen3: when Hadamard mixes zero-energy
   dims into high-energy dims, the quantization SNR collapses for the
   signal dims because there's ~0 signal diluting them.

4. **The knee is real.** For most Qwen3 layers, the spectrum has a visible
   drop around rank 60 (r95) — roughly half of head_dim carries 95% of
   the energy. A rank-aware rotation that places the top-60 principal
   components at the front of the vector would let TQ quantize those
   bits densely and zero the rest.

### Consequences for Phase 7

- **Target the spectral knee, not a fixed rank.** Zero out everything
  below r_eff(99%) per layer rather than using one global rank.
- **Per-layer rotation matrices (not universal Hadamard).** The rotation
  axis is the SVD of the calibration K-vectors per layer.
- **Layer 0 is a special case.** Its near-zero effective rank suggests
  Phase 7's rotation + zeroing should produce a very small effective
  quantizer for L0 — or it may be better to keep L0 in fp16 entirely
  and only apply Phase 7 to L1..L_{n-1}.
- **Phase 6 regularization dial is still needed.** Even after rank-aware
  rotation, different backends (Metal/CUDA) inject different amounts of
  noise into the signal-carrying dims. The `reg_bias` dial complements
  Phase 7's geometric fix.

## Raw data

- `qwen3_4b_calib.bin` — 180 records × 36 layers, ~11 MiB
- `mistral_7b_calib.bin` — 160 records × 32 layers, ~11 MiB

Both files use the LeanKV calibration binary format documented in
`src/leankv-calib.h`.

## Next step

Implement a prototype per-layer rotation in Python on these cached files:

1. For each layer L, SVD the K-vectors → get V_L (right singular vectors)
2. Simulate quantization: `K_rot = K @ V_L`, quantize to TQ3 on first
   r_L dims and zero after r_L, dequantize, inverse rotate
3. Compare reconstruction MSE to the Hadamard+TQ3 baseline on the same
   calibration vectors
4. Predict PPL improvement on Qwen3-4B based on MSE ratio

This is a pure-Python / pure-numpy exercise — no C code changes needed
until we validate the approach offline.
