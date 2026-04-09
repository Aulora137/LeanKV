#!/bin/bash
# Phase 2b: Perplexity benchmark for TQ KV cache types
# Runs llama-perplexity across 3 models x 6 KV configs = 18 runs

set -euo pipefail

LLAMA_PPL="/home/junc/LeanInfer/upstream/build/bin/llama-perplexity"
DATASET="/home/junc/LeanKV/prototype/eval/wikitext-2-raw/wiki.test.raw"
RESULTS_DIR="/home/junc/LeanKV/prototype/eval/results"
CSV_FILE="${RESULTS_DIR}/ppl_benchmark.csv"
LOG_DIR="${RESULTS_DIR}/logs"

mkdir -p "$RESULTS_DIR" "$LOG_DIR"

# Models
declare -A MODELS
MODELS["Qwen3.5-2B"]="/home/junc/Aulora/bitcoin-node-stack/models/Qwen3.5-2B-Q4_K_M.gguf"
MODELS["Qwen3-4B"]="/home/junc/Aulora/bitcoin-node-stack/models/Qwen3-4B-Q4_K_M.gguf"
MODELS["Qwen3.5-9B"]="/home/junc/Aulora/bitcoin-node-stack/models/Qwen3.5-9B-Q4_K_M.gguf"

# KV configs: "label ctk ctv"
CONFIGS=(
    "F16_F16 f16 f16"
    "Q8_F16 q8_0 f16"
    "TQ4_F16 tq4_0 f16"
    "TQ4_TQ4 tq4_0 tq4_0"
    "TQ3_F16 tq3_0 f16"
    "TQ3_TQ3 tq3_0 tq3_0"
)

# Model order (smallest first)
MODEL_ORDER=("Qwen3.5-2B" "Qwen3-4B" "Qwen3.5-9B")

# Write CSV header
echo "model,config,ctk,ctv,ppl,ppl_stderr,elapsed_sec" > "$CSV_FILE"

TOTAL_RUNS=$(( ${#MODEL_ORDER[@]} * ${#CONFIGS[@]} ))
RUN=0

for model_name in "${MODEL_ORDER[@]}"; do
    model_path="${MODELS[$model_name]}"
    echo ""
    echo "=========================================="
    echo "Model: $model_name"
    echo "=========================================="

    for config in "${CONFIGS[@]}"; do
        read -r label ctk ctv <<< "$config"
        RUN=$((RUN + 1))
        log_file="${LOG_DIR}/${model_name}_${label}.log"

        echo ""
        echo "[$RUN/$TOTAL_RUNS] $model_name / $label (ctk=$ctk, ctv=$ctv)"
        echo "  Log: $log_file"

        START_TIME=$(date +%s)

        # Run perplexity benchmark
        # -c 2048: context size
        # -t $(nproc): use all cores
        # --no-display-prompt: cleaner output
        "$LLAMA_PPL" \
            -m "$model_path" \
            -f "$DATASET" \
            -c 2048 \
            -t "$(nproc)" \
            -ctk "$ctk" \
            -ctv "$ctv" \
            --no-display-prompt \
            2>&1 | tee "$log_file"

        END_TIME=$(date +%s)
        ELAPSED=$((END_TIME - START_TIME))

        # Extract final perplexity from output
        # Format: "Final estimate: PPL over N chunks for n_ctx=2048 = 12.1842 +/- 0.76320"
        ppl=$(grep -oP 'Final estimate: PPL .* = \K[0-9.]+' "$log_file" || echo "FAILED")
        ppl_stderr=$(grep -oP 'Final estimate: PPL .* = [0-9.]+ \+/- \K[0-9.]+' "$log_file" || echo "N/A")

        echo "  PPL = $ppl +/- $ppl_stderr  (${ELAPSED}s)"
        echo "$model_name,$label,$ctk,$ctv,$ppl,$ppl_stderr,$ELAPSED" >> "$CSV_FILE"
    done
done

echo ""
echo "=========================================="
echo "BENCHMARK COMPLETE"
echo "=========================================="
echo "Results: $CSV_FILE"
echo ""

# Print summary table
echo "Model          | Config    | PPL        | +/-     | Time"
echo "---------------|-----------|------------|---------|------"
tail -n +2 "$CSV_FILE" | while IFS=, read -r model config ctk ctv ppl stderr elapsed; do
    printf "%-14s | %-9s | %-10s | %-7s | %ss\n" "$model" "$config" "$ppl" "$stderr" "$elapsed"
done
