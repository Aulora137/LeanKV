#!/bin/bash
# Phase 2b addendum: Perplexity benchmark for Qwen3.5-4B
set -euo pipefail

LLAMA_PPL="/home/junc/LeanInfer/upstream/build/bin/llama-perplexity"
DATASET="/home/junc/LeanKV/prototype/eval/wikitext-2-raw/wiki.test.raw"
RESULTS_DIR="/home/junc/LeanKV/prototype/eval/results"
CSV_FILE="${RESULTS_DIR}/ppl_benchmark_qwen35_4b.csv"
LOG_DIR="${RESULTS_DIR}/logs"

mkdir -p "$RESULTS_DIR" "$LOG_DIR"

MODEL_NAME="Qwen3.5-4B"
MODEL_PATH="/home/junc/Aulora/bitcoin-node-stack/models/Qwen3.5-4B-Q4_K_M.gguf"

CONFIGS=(
    "F16_F16 f16 f16"
    "Q8_F16 q8_0 f16"
    "TQ4_F16 tq4_0 f16"
    "TQ4_TQ4 tq4_0 tq4_0"
    "TQ3_F16 tq3_0 f16"
    "TQ3_TQ3 tq3_0 tq3_0"
)

echo "model,config,ctk,ctv,ppl,ppl_stderr,elapsed_sec" > "$CSV_FILE"

RUN=0
for config in "${CONFIGS[@]}"; do
    read -r label ctk ctv <<< "$config"
    RUN=$((RUN + 1))
    log_file="${LOG_DIR}/${MODEL_NAME}_${label}.log"

    echo ""
    echo "[$RUN/${#CONFIGS[@]}] $MODEL_NAME / $label (ctk=$ctk, ctv=$ctv)"

    START_TIME=$(date +%s)

    "$LLAMA_PPL" \
        -m "$MODEL_PATH" \
        -f "$DATASET" \
        -c 2048 \
        -t "$(nproc)" \
        -ctk "$ctk" \
        -ctv "$ctv" \
        --no-display-prompt \
        2>&1 | tee "$log_file"

    END_TIME=$(date +%s)
    ELAPSED=$((END_TIME - START_TIME))

    ppl=$(grep -oP 'Final estimate: PPL .* = \K[0-9.]+' "$log_file" || echo "FAILED")
    ppl_stderr=$(grep -oP 'Final estimate: PPL .* = [0-9.]+ \+/- \K[0-9.]+' "$log_file" || echo "N/A")

    echo "  PPL = $ppl +/- $ppl_stderr  (${ELAPSED}s)"
    echo "$MODEL_NAME,$label,$ctk,$ctv,$ppl,$ppl_stderr,$ELAPSED" >> "$CSV_FILE"
done

echo ""
echo "=========================================="
echo "BENCHMARK COMPLETE"
echo "=========================================="
cat "$CSV_FILE"
