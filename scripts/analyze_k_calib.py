#!/usr/bin/env python3
"""LeanKV Phase 7: analyze K-vector calibration dumps for rank deficiency.

Reads a `.bin` file produced by Lean_llama.cpp when run with
`LEANKV_CALIBRATION_DUMP=1`, stacks all K vectors per layer, and runs SVD
on the resulting [n_vectors, head_dim] matrix to expose the effective rank.

Usage:
    python3 analyze_k_calib.py /tmp/qwen3_4b_calib.bin
    python3 analyze_k_calib.py /tmp/qwen3_4b_calib.bin --layers 0,5,35
    python3 analyze_k_calib.py /tmp/qwen3_4b_calib.bin --energy 0.99

File format (see src/leankv-calib.h):
    file header: u32 magic='KCAL' (0x4C41434B), u32 version=1
    each record: u32 rec_magic='LKCR' (0x52434B4C), u32 il, u32 dtype,
                 u32 ndims, u32 ne[4], u32 nb[4], u32 n_bytes, u8 data[]

Tensor layout for K (post-RoPE, pre-cache):
    2D view: ne[0] = head_dim, ne[1] = n_tokens * n_head_kv
    (ggml packs heads contiguously in the second dimension)

A rank-deficient layer will have a sharp drop in its singular-value spectrum
somewhere well before r = head_dim. That drop position is the target rank
for Phase 7 rank-aware rotation.
"""

from __future__ import annotations

import argparse
import struct
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

FILE_MAGIC = 0x4C41434B   # 'KCAL'
REC_MAGIC  = 0x52434B4C   # 'LKCR'

# ggml_type codes we might see for calibration tensors.
GGML_TYPE_F32 = 0
GGML_TYPE_F16 = 1


def read_records(path: Path):
    """Yield (il, np.ndarray shape [n_vecs, head_dim]) per record."""
    with path.open("rb") as f:
        magic, version = struct.unpack("<II", f.read(8))
        if magic != FILE_MAGIC:
            raise ValueError(f"bad file magic 0x{magic:08x} (expected 0x{FILE_MAGIC:08x})")
        if version != 1:
            raise ValueError(f"unsupported calib version {version}")

        while True:
            hdr = f.read(4 * 12)
            if not hdr:
                return
            if len(hdr) < 4 * 12:
                raise EOFError("truncated record header")
            rec_magic, il, dtype, ndims = struct.unpack("<IIII", hdr[:16])
            ne = struct.unpack("<IIII", hdr[16:32])
            nb = struct.unpack("<IIII", hdr[32:48])
            (n_bytes,) = struct.unpack("<I", f.read(4))
            data = f.read(n_bytes)
            if len(data) < n_bytes:
                raise EOFError("truncated record data")
            if rec_magic != REC_MAGIC:
                raise ValueError(f"bad record magic 0x{rec_magic:08x}")

            head_dim = ne[0]
            n_rows   = ne[1] if ndims >= 2 else 1
            # For ndims >= 3, flatten trailing dims into n_rows (heads × tokens pack into dim 1+)
            if ndims >= 3:
                n_rows *= ne[2]
            if ndims >= 4:
                n_rows *= ne[3]

            if dtype == GGML_TYPE_F32:
                arr = np.frombuffer(data, dtype=np.float32)
            elif dtype == GGML_TYPE_F16:
                arr = np.frombuffer(data, dtype=np.float16).astype(np.float32)
            else:
                raise ValueError(f"unsupported dtype {dtype}")

            expected = head_dim * n_rows
            if arr.size < expected:
                raise ValueError(f"data size {arr.size} < expected {expected}")
            arr = arr[:expected].reshape(n_rows, head_dim)
            yield il, arr


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("path", type=Path)
    ap.add_argument("--layers", type=str, default=None,
                    help="comma-separated layer indices (default: all)")
    ap.add_argument("--energy", type=float, default=0.99,
                    help="energy threshold for effective rank (default 0.99)")
    ap.add_argument("--plot", action="store_true",
                    help="write per-layer spectrum PNG next to the input file")
    args = ap.parse_args()

    per_layer: dict[int, list[np.ndarray]] = defaultdict(list)
    total_records = 0
    for il, mat in read_records(args.path):
        per_layer[il].append(mat)
        total_records += 1

    layers_sorted = sorted(per_layer.keys())
    if args.layers:
        requested = {int(x) for x in args.layers.split(",") if x.strip()}
        layers_sorted = [il for il in layers_sorted if il in requested]

    print(f"loaded {total_records} records across {len(per_layer)} layers")
    print()
    print(f"{'layer':>5} {'n_vecs':>8} {'dim':>5} "
          f"{'r_eff':>6} {'r95':>5} {'r99':>5} {'sv[0]':>9} {'sv[-1]':>9} "
          f"{'decay':>8}")
    print("-" * 70)

    summary = {}
    for il in layers_sorted:
        stacked = np.concatenate(per_layer[il], axis=0)  # [N, head_dim]
        # Center? For K vectors post-RoPE we keep them as-is; SVD of raw is
        # what Hadamard would see. A centered variant is also instructive but
        # we want the pre-rotation geometry faithfully.
        n_vecs, dim = stacked.shape
        # Use thin SVD via gesdd
        try:
            sv = np.linalg.svd(stacked, compute_uv=False)
        except np.linalg.LinAlgError as e:
            print(f"{il:>5} SVD failed: {e}")
            continue

        cum = np.cumsum(sv ** 2)
        total_energy = cum[-1] if cum[-1] > 0 else 1.0
        r95 = int(np.searchsorted(cum / total_energy, 0.95) + 1)
        r99 = int(np.searchsorted(cum / total_energy, 0.99) + 1)
        r_eff = int(np.searchsorted(cum / total_energy, args.energy) + 1)
        decay = float(sv[-1] / sv[0]) if sv[0] > 0 else 0.0

        summary[il] = {
            "n_vecs": n_vecs, "dim": dim,
            "r_eff": r_eff, "r95": r95, "r99": r99,
            "sv_max": float(sv[0]), "sv_min": float(sv[-1]),
            "decay": decay,
            "spectrum": sv,
        }

        print(f"{il:>5} {n_vecs:>8} {dim:>5} "
              f"{r_eff:>6} {r95:>5} {r99:>5} "
              f"{sv[0]:>9.3f} {sv[-1]:>9.3f} {decay:>8.1e}")

    # Global summary
    if summary:
        r_effs = [s["r_eff"] for s in summary.values()]
        r99s   = [s["r99"]   for s in summary.values()]
        dim    = next(iter(summary.values()))["dim"]
        print()
        print(f"head_dim               : {dim}")
        print(f"effective rank @{args.energy:.0%}  : "
              f"min={min(r_effs)} max={max(r_effs)} median={int(np.median(r_effs))}")
        print(f"effective rank @99%    : "
              f"min={min(r99s)} max={max(r99s)} median={int(np.median(r99s))}")

        deficient = [il for il, s in summary.items() if s["r99"] < dim - dim // 8]
        if deficient:
            print(f"rank-deficient layers  : {len(deficient)}/{len(summary)} "
                  f"(r99 < {dim - dim // 8})")
            print(f"  layers: {deficient}")
        else:
            print(f"no strongly rank-deficient layers at r99 < {dim - dim // 8}")

    if args.plot:
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
        except ImportError:
            print("matplotlib not installed; skipping --plot", file=sys.stderr)
            return
        out = args.path.with_suffix(".svd.png")
        fig, ax = plt.subplots(figsize=(10, 6))
        for il in layers_sorted:
            s = summary[il]
            sv = s["spectrum"]
            ax.semilogy(sv / sv[0], label=f"L{il}", alpha=0.5, linewidth=0.8)
        ax.set_xlabel("singular value index")
        ax.set_ylabel("σ_i / σ_0 (log scale)")
        ax.set_title(f"K-vector singular spectrum — {args.path.name}")
        ax.axhline(1e-2, color="red", linestyle="--", alpha=0.3, label="1% threshold")
        ax.grid(True, alpha=0.3)
        if len(layers_sorted) <= 12:
            ax.legend(ncol=2, fontsize=8)
        fig.tight_layout()
        fig.savefig(out, dpi=120)
        print(f"\nplot: {out}")


if __name__ == "__main__":
    main()
