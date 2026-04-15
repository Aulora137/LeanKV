#!/usr/bin/env python3
"""
Phase 6.5: Rank-deficiency analysis for GQA attention projections.

Motivation
----------
Qwen3-4B has n_embd=2560, n_head=32, n_head_kv=8, head_dim=128.
Per-head W_K maps [2560 -> 128] and COULD have full rank 128, but
n_embd/n_head_kv = 320 suggests each KV head may only use a subspace
smaller than 128. If rank(W_K_h) << 128 on average, then after projection
onto the effective subspace the K cache has fewer dimensions to quantize
and the same TQ budget buys higher per-dimension precision.

This script measures the empirical singular-value spectrum of W_K and W_Q
per head and per layer, and reports:
  - singular values up to various energy fractions (90/95/99/99.9%)
  - how small a projection can be before residual energy exceeds 1%
  - block-aligned candidates {64, 96, 128} for compatibility with existing
    TQ kernels (32-block-aligned head dims)

For GQA we also compute the "union rank" across the 4 Q heads that share
a single KV head: the minimum subspace that preserves 99% of combined
K-space information, since the KV cache sits on that shared subspace.

Outputs: a markdown-style report to stdout and CSV summary to
docs/phase6.5-rank-deficiency-<model>.csv.

Usage
-----
    python3 scripts/rank_deficiency_analysis.py <path-to.gguf>
"""

import argparse
import csv
import os
import sys
from pathlib import Path

import numpy as np

GGUF_PY = os.environ.get("GGUF_PY", "/home/junc/Lean_llama.cpp/gguf-py")
if GGUF_PY not in sys.path:
    sys.path.insert(0, GGUF_PY)

from gguf import GGUFReader  # noqa: E402
from gguf.quants import dequantize  # noqa: E402


def scalar_field(reader, name):
    f = reader.fields.get(name)
    if f is None:
        return None
    return int(f.parts[f.data[0]][0])


def string_field(reader, name):
    f = reader.fields.get(name)
    if f is None:
        return None
    return bytes(f.parts[f.data[-1]]).decode("utf-8", "replace")


def load_weight(reader, tensor_name):
    t = next((x for x in reader.tensors if x.name == tensor_name), None)
    if t is None:
        return None
    # GGUF stores weight as [out, in] — dequantize returns same shape.
    arr = dequantize(t.data, t.tensor_type)
    out_dim, in_dim = int(t.shape[1]), int(t.shape[0])
    return arr.reshape(out_dim, in_dim).astype(np.float32)


def rank_at_energy(singular_values, target_fraction):
    """Smallest r such that sum(s[:r]^2) / sum(s^2) >= target_fraction."""
    sq = singular_values ** 2
    total = sq.sum()
    if total <= 0:
        return 0
    cum = np.cumsum(sq)
    return int(np.searchsorted(cum, target_fraction * total) + 1)


def analyse_head(W_head):
    """W_head: [in_dim, head_dim]. Returns singular values (descending)."""
    # np.linalg.svd on [in, out]: min(in, out) singular values.
    _, s, _ = np.linalg.svd(W_head, full_matrices=False)
    return s


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("gguf", help="path to GGUF model")
    ap.add_argument("--layers", type=int, default=0,
                    help="limit to first N layers (0 = all)")
    ap.add_argument("--csv", default=None,
                    help="output CSV path (default: docs/phase6.5-rank-<model>.csv)")
    args = ap.parse_args()

    gguf_path = Path(args.gguf).expanduser().resolve()
    print(f"# Phase 6.5 rank-deficiency analysis — {gguf_path.name}\n")

    reader = GGUFReader(str(gguf_path))

    arch = string_field(reader, "general.architecture") or "unknown"
    n_embd = scalar_field(reader, f"{arch}.embedding_length")
    n_layer = scalar_field(reader, f"{arch}.block_count")
    n_head = scalar_field(reader, f"{arch}.attention.head_count")
    n_head_kv = scalar_field(reader, f"{arch}.attention.head_count_kv") or n_head
    if None in (n_embd, n_layer, n_head):
        print("ERROR: missing required architecture fields", file=sys.stderr)
        sys.exit(2)

    # head_dim: look for attn_k tensor shape to derive
    t_k0 = next((x for x in reader.tensors if x.name == "blk.0.attn_k.weight"), None)
    if t_k0 is None:
        print("ERROR: missing blk.0.attn_k.weight", file=sys.stderr)
        sys.exit(2)
    kv_out = int(t_k0.shape[1])
    head_dim = kv_out // n_head_kv
    q_group = n_head // n_head_kv  # GQA sharing factor

    print(f"- arch: **{arch}**")
    print(f"- n_embd = {n_embd}, n_layer = {n_layer}")
    print(f"- n_head = {n_head}, n_head_kv = {n_head_kv} (GQA {q_group}:1)")
    print(f"- head_dim = {head_dim}")
    print(f"- n_embd / n_head = {n_embd / n_head:g} "
          f"(informal per-Q-head rank bound if the hidden state were evenly split)")
    print(f"- n_embd / n_head_kv = {n_embd / n_head_kv:g} "
          f"(same for KV heads)\n")

    layers_to_scan = n_layer if args.layers == 0 else min(args.layers, n_layer)

    energies = (0.90, 0.95, 0.99, 0.999)
    block_candidates = [d for d in (64, 96, 128, 160) if d <= head_dim * 2]

    # accumulators
    csv_rows = []
    summary_k = {e: [] for e in energies}      # per-layer median per-KV-head rank
    summary_q = {e: [] for e in energies}      # per-layer median per-Q-head rank
    summary_union = {e: [] for e in energies}  # per-layer median union-of-q_group rank
    residual_at_block = {d: [] for d in block_candidates}

    print("## Per-layer summary\n")
    print("| layer | K head rank @95% (median/max) | Q head rank @95% (median/max) |"
          " union@99% (median/max) | residual at dim=96 |")
    print("|------:|:-----------------------------:|:-----------------------------:"
          "|:----------------------:|:-------------------:|")

    for L in range(layers_to_scan):
        W_k = load_weight(reader, f"blk.{L}.attn_k.weight")
        W_q = load_weight(reader, f"blk.{L}.attn_q.weight")
        if W_k is None or W_q is None:
            continue

        # per-KV-head K: shape [n_embd, head_dim]
        k_head_ranks = {e: [] for e in energies}
        residual_at_block_layer = {d: [] for d in block_candidates}
        for h in range(n_head_kv):
            Wk_h = W_k[h * head_dim:(h + 1) * head_dim, :].T  # [n_embd, head_dim]
            s = analyse_head(Wk_h)
            for e in energies:
                k_head_ranks[e].append(rank_at_energy(s, e))
            sq = s ** 2
            total = sq.sum()
            for d in block_candidates:
                if d >= len(s):
                    residual_at_block_layer[d].append(0.0)
                else:
                    residual_at_block_layer[d].append(
                        float(1.0 - sq[:d].sum() / total))

        # per-Q-head Q
        q_head_ranks = {e: [] for e in energies}
        q_head_s = []  # keep s to compute union rank per group
        for h in range(n_head):
            Wq_h = W_q[h * head_dim:(h + 1) * head_dim, :].T  # [n_embd, head_dim]
            s = analyse_head(Wq_h)
            q_head_s.append(s)
            for e in energies:
                q_head_ranks[e].append(rank_at_energy(s, e))

        # union rank: stack q_group Q heads that share a KV head, SVD the concat
        union_ranks = {e: [] for e in energies}
        for g in range(n_head_kv):
            # heads g*q_group .. (g+1)*q_group share KV head g
            stacked_cols = []
            for h in range(g * q_group, (g + 1) * q_group):
                Wq_h = W_q[h * head_dim:(h + 1) * head_dim, :].T  # [n_embd, head_dim]
                stacked_cols.append(Wq_h)
            stacked = np.concatenate(stacked_cols, axis=1)  # [n_embd, q_group*head_dim]
            _, s, _ = np.linalg.svd(stacked, full_matrices=False)
            for e in energies:
                union_ranks[e].append(rank_at_energy(s, e))

        for e in energies:
            summary_k[e].append(np.median(k_head_ranks[e]))
            summary_q[e].append(np.median(q_head_ranks[e]))
            summary_union[e].append(np.median(union_ranks[e]))
        for d in block_candidates:
            residual_at_block[d].append(float(np.mean(residual_at_block_layer[d])))

        k95 = k_head_ranks[0.95]
        q95 = q_head_ranks[0.95]
        u99 = union_ranks[0.99]
        res96 = (float(np.mean(residual_at_block_layer[96]))
                 if 96 in residual_at_block_layer else float("nan"))
        print(f"| {L:3d} | {int(np.median(k95))}/{int(np.max(k95))} "
              f"| {int(np.median(q95))}/{int(np.max(q95))} "
              f"| {int(np.median(u99))}/{int(np.max(u99))} "
              f"| {res96*100:.2f}% |")

        csv_rows.append({
            "layer": L,
            "k_head_median_r95": int(np.median(k95)),
            "k_head_max_r95": int(np.max(k95)),
            "q_head_median_r95": int(np.median(q95)),
            "q_head_max_r95": int(np.max(q95)),
            "union_median_r99": int(np.median(u99)),
            "union_max_r99": int(np.max(u99)),
            "residual_at_64": float(np.mean(residual_at_block_layer.get(64, [0]))),
            "residual_at_96": res96,
            "residual_at_128": float(np.mean(residual_at_block_layer.get(128, [0]))),
        })

    print("\n## Aggregated (median across all layers)\n")
    print("| metric | 90% | 95% | 99% | 99.9% |")
    print("|:-------|----:|----:|----:|------:|")
    print("| K per-head rank (median)   | "
          f"{np.median(summary_k[0.90]):.0f} | {np.median(summary_k[0.95]):.0f} | "
          f"{np.median(summary_k[0.99]):.0f} | {np.median(summary_k[0.999]):.0f} |")
    print("| Q per-head rank (median)   | "
          f"{np.median(summary_q[0.90]):.0f} | {np.median(summary_q[0.95]):.0f} | "
          f"{np.median(summary_q[0.99]):.0f} | {np.median(summary_q[0.999]):.0f} |")
    print(f"| Q union-of-{q_group} rank (median) | "
          f"{np.median(summary_union[0.90]):.0f} | {np.median(summary_union[0.95]):.0f} | "
          f"{np.median(summary_union[0.99]):.0f} | {np.median(summary_union[0.999]):.0f} |")

    print("\n## Residual energy lost when truncating W_K to block-aligned dim\n")
    print("| truncation dim | mean residual (lower = more compressible) |")
    print("|:---------------|:------------------------------------------|")
    for d in block_candidates:
        if not residual_at_block[d]:
            continue
        m = float(np.mean(residual_at_block[d]))
        print(f"| {d:3d} | {m*100:.3f}% |")

    out_csv = args.csv
    if out_csv is None:
        out_csv = f"/home/junc/LeanKV/docs/phase6.5-rank-{gguf_path.stem}.csv"
    with open(out_csv, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(csv_rows[0].keys()))
        w.writeheader()
        w.writerows(csv_rows)
    print(f"\nCSV: {out_csv}")

    # Verdict
    print("\n## Verdict\n")
    median_union_99 = int(np.median(summary_union[0.99]))
    median_k_99 = int(np.median(summary_k[0.99]))
    if median_union_99 <= 96:
        verdict = (f"**Promising** — union rank at 99% energy is {median_union_99} "
                   "≤ 96. Truncating K cache to 96 dims (block-aligned, Hadamard-"
                   "friendly) loses <1% of projection energy. TQ2 at 96 dims ≈ "
                   f"same bytes as TQ3 at 128, potentially matching TQ3 quality.")
    elif median_union_99 <= 128:
        verdict = (f"**Marginal** — union rank is {median_union_99}, close to full "
                   "128. Projection savings are small; pursue only if runtime "
                   "measurements confirm 99%-energy rank is actually lower than W_K "
                   "suggests.")
    else:
        verdict = (f"**No** — union rank {median_union_99} exceeds head_dim=128, "
                   "so the full head space is used. Rank projection is not a win.")
    print(verdict)
    print(f"\nPer-K-head rank at 99%: {median_k_99}. If this is "
          f"much less than {head_dim} the K cache itself is rank-deficient even "
          f"without the union-over-GQA argument.")


if __name__ == "__main__":
    main()
