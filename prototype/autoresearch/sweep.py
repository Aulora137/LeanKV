"""Main entry point for LeanKV autoresearch grid sweep (6 knobs)."""

import argparse
import json
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from autoresearch.config import generate_search_space
from autoresearch.runner import ExperimentRunner
from autoresearch.database import ResultsDB


def main():
    parser = argparse.ArgumentParser(description="LeanKV autoresearch sweep (6 knobs)")
    parser.add_argument("--db-path", default="prototype/autoresearch/results/sweep_6knob.db")
    parser.add_argument("--model", default="Qwen/Qwen2.5-0.5B-Instruct")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()

    print("=" * 80)
    print("LeanKV Autoresearch Sweep (6 Knobs)")
    print(f"Model: {args.model}")
    print("=" * 80)

    configs = generate_search_space(seed=args.seed)
    print(f"\nSearch space: {len(configs)} configurations")
    print("Knobs: K_bits × V_bits × rotation × group_size × layer_policy × QJL")

    runner = ExperimentRunner(model_name=args.model, device=args.device)
    db = ResultsDB(db_path=args.db_path)

    print(f"\n{'='*80}")
    print("Running sweep...")
    print(f"{'='*80}\n")

    t_start = time.time()
    all_results = runner.run_all(configs)

    for r in all_results:
        db.insert_result(r)

    elapsed = time.time() - t_start
    elapsed_str = f"{elapsed/60:.1f}min" if elapsed > 60 else f"{elapsed:.0f}s"
    print(f"\nSweep complete: {len(all_results)} configs in {elapsed_str}")

    # Save raw JSON
    json_path = os.path.join(os.path.dirname(args.db_path), "sweep_6knob_results.json")
    with open(json_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"Results saved to {json_path}")

    # Print top 20 by quality
    print(f"\n{'='*80}")
    print("TOP 20 CONFIGS (by min(K_cos, V_cos))")
    print(f"{'='*80}")
    print(f"{'Config':<32} {'K cos':>7} {'V cos':>7} {'Total':>6} {'GS':>4} {'Policy':>8}")
    print("-" * 72)

    by_quality = sorted(all_results, key=lambda r: min(r["k_cosine_sim_mean"], r["v_cosine_sim_mean"]), reverse=True)
    for r in by_quality[:20]:
        print(f"  {r['config_name']:<30} {r['k_cosine_sim_mean']:.4f}  {r['v_cosine_sim_mean']:.4f}  "
              f"{r['total_bits_per_kv_pair']:>5.1f}  {r.get('group_size',''):>4}  {r.get('layer_policy',''):>8}")

    # Pareto frontier
    pareto = db.get_pareto_frontier()
    print(f"\n{'='*80}")
    print(f"PARETO FRONTIER ({len(pareto)} configs)")
    print(f"{'='*80}")
    print(f"{'Config':<32} {'K cos':>7} {'V cos':>7} {'Total':>6} {'Compress':>9}")
    print("-" * 65)
    for r in pareto:
        compress = 32.0 / r["total_bits_per_kv_pair"]
        print(f"  {r['config_name']:<30} {r['k_cos_mean']:.4f}  {r['v_cos_mean']:.4f}  "
              f"{r['total_bits_per_kv_pair']:>5.1f}  {compress:>8.1f}x")

    db.close()
    print(f"\nDatabase: {args.db_path}")


if __name__ == "__main__":
    main()
