#!/usr/bin/env bash
# LeanKV Phase 3b — Apple M2 Perplexity Benchmark (standalone)
#
# Runs PPL evaluation for each KV config independently so one crash
# doesn't kill the whole run. Designed for overnight execution.
#
# Usage:
#   nohup ./bench_m2_ppl.sh > ppl_run.log 2>&1 &

set -uo pipefail  # no -e: we handle errors per-config

LLAMA_PPL="/Users/hchome/Lean_llama.cpp/build/bin/llama-perplexity"
MODEL="/Users/hchome/.lmstudio/models/lmstudio-community/Qwen3.5-9B-GGUF/Qwen3.5-9B-Q4_K_M.gguf"
DATASET="/Users/hchome/LeanKV/prototype/eval/wikitext-2-raw/wikitext-2-raw/wiki.test.raw"
OUTPUT_DIR="/Users/hchome/LeanKV/scripts/results-m2"
PPL_CTX=2048
NCPU=$(sysctl -n hw.ncpu 2>/dev/null || echo 8)

mkdir -p "$OUTPUT_DIR/logs"

CSV_PPL="${OUTPUT_DIR}/ppl_overnight.csv"
echo "label,ctk,ctv,n_ctx,ppl,ppl_stderr,elapsed_sec" > "$CSV_PPL"

# KV cache configs: "label ctk ctv"
CONFIGS=(
    "F16_F16   f16   f16"
    "Q8_F16    q8_0  f16"
    "TQ4_F16   tq4_0 f16"
    "TQ4_TQ4   tq4_0 tq4_0"
)

echo ""
echo "=== LeanKV M2 PPL Benchmark (overnight) ==="
echo "Started: $(date)"
echo "Model  : $(basename "$MODEL")"
echo "Dataset: $(basename "$DATASET")"
echo "n_ctx  : $PPL_CTX"
echo "Threads: $NCPU"
echo "Configs: ${#CONFIGS[@]}"
echo "Est. time: ~2h per config, ~8h total"
echo ""

TOTAL=${#CONFIGS[@]}
RUN=0

for config in "${CONFIGS[@]}"; do
    read -r label ctk ctv <<< "$config"
    RUN=$((RUN + 1))
    log_file="${OUTPUT_DIR}/logs/ppl_${label}_c${PPL_CTX}.log"

    echo "──────────────────────────────────────────"
    echo "[$RUN/$TOTAL] $label (ctk=$ctk, ctv=$ctv)"
    echo "  Started: $(date)"
    echo "  Log: $log_file"

    START_TIME=$(date +%s)

    # Run in subshell so crash doesn't kill the loop
    (
        "$LLAMA_PPL" \
            -m "$MODEL" \
            -f "$DATASET" \
            -c "$PPL_CTX" \
            -ngl 0 \
            -ctk "$ctk" \
            -ctv "$ctv" \
            -t "$NCPU" \
            --no-display-prompt \
            2>&1
    ) | tee "$log_file"
    EXIT_CODE=${PIPESTATUS[0]}

    END_TIME=$(date +%s)
    ELAPSED=$((END_TIME - START_TIME))

    if [[ $EXIT_CODE -ne 0 ]]; then
        echo "  *** CRASHED (exit $EXIT_CODE) after ${ELAPSED}s ***"
        echo "$label,$ctk,$ctv,$PPL_CTX,CRASHED,N/A,$ELAPSED" >> "$CSV_PPL"
        echo ""
        continue
    fi

    # Extract PPL: "Final estimate: PPL over 145 chunks ... = 7.2591 +/- 0.06320"
    ppl=$(grep 'Final estimate: PPL' "$log_file" | sed -n 's/.*= *\([0-9][0-9]*\.[0-9][0-9]*\).*/\1/p')
    ppl_stderr=$(grep 'Final estimate: PPL' "$log_file" | sed -n 's/.*+\/- *\([0-9][0-9]*\.[0-9][0-9]*\).*/\1/p')
    ppl=${ppl:-FAILED}
    ppl_stderr=${ppl_stderr:-N/A}

    echo "  PPL = $ppl +/- $ppl_stderr  (${ELAPSED}s)"
    echo "$label,$ctk,$ctv,$PPL_CTX,$ppl,$ppl_stderr,$ELAPSED" >> "$CSV_PPL"
    echo ""
done

echo "══════════════════════════════════════════"
echo "BENCHMARK COMPLETE: $(date)"
echo "Results: $CSV_PPL"
echo ""
cat "$CSV_PPL"
echo ""

# Print summary table
echo ""
echo "Config     | PPL       | +/-     | Time"
echo "-----------|-----------|---------|--------"
tail -n +2 "$CSV_PPL" | while IFS=, read -r label ctk ctv n_ctx ppl stderr elapsed; do
    mins=$((elapsed / 60))
    printf "%-10s | %-9s | %-7s | %dm\n" "$label" "$ppl" "$stderr" "$mins"
done
