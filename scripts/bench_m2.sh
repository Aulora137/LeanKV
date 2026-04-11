#!/usr/bin/env bash
# LeanKV Phase 3b — Apple M2 Benchmark
#
# Benchmarks TQ4_0 KV cache quantization on Apple Silicon (ARM NEON path).
# Measures decode tok/s, prefill tok/s, and perplexity.
#
# Usage:
#   ./bench_m2.sh --model=/path/to/model.gguf --dataset=/path/to/wiki.test.raw
#
# Flags:
#   --model PATH       Path to GGUF model (required)
#   --llama-bench PATH Path to llama-bench binary (default: auto-detect)
#   --llama-ppl PATH   Path to llama-perplexity binary (default: auto-detect)
#   --dataset PATH     Path to WikiText-2 raw file (optional, for PPL runs)
#   --output DIR       Output directory (default: ./results-m2)
#   --skip-ppl         Skip perplexity runs (faster, throughput only)
#   --skip-throughput  Skip throughput runs (PPL only)

set -euo pipefail

# ─── Defaults ───────────────────────────────────────────────────────────────

MODEL=""
LLAMA_BENCH=""
LLAMA_PPL=""
DATASET=""
OUTPUT_DIR="./results-m2"
SKIP_PPL=false
SKIP_THROUGHPUT=false

# ─── Parse args ─────────────────────────────────────────────────────────────

for arg in "$@"; do
    case "$arg" in
        --model=*)       MODEL="${arg#--model=}" ;;
        --llama-bench=*) LLAMA_BENCH="${arg#--llama-bench=}" ;;
        --llama-ppl=*)   LLAMA_PPL="${arg#--llama-ppl=}" ;;
        --dataset=*)     DATASET="${arg#--dataset=}" ;;
        --output=*)      OUTPUT_DIR="${arg#--output=}" ;;
        --skip-ppl)      SKIP_PPL=true ;;
        --skip-throughput) SKIP_THROUGHPUT=true ;;
        --help|-h)
            head -20 "$0" | tail -16
            exit 0
            ;;
    esac
done

if [[ -z "$MODEL" ]]; then
    echo "ERROR: --model is required"
    echo "Usage: $0 --model=path/to/model.gguf"
    exit 1
fi

if [[ ! -f "$MODEL" ]]; then
    echo "ERROR: Model not found: $MODEL"
    exit 1
fi

# ─── Auto-detect binaries ──────────────────────────────────────────────────

find_binary() {
    local name="$1"
    for dir in /Users/hchome/Lean_llama.cpp/build/bin ./build/bin ../build/bin ../Lean_llama.cpp/build/bin; do
        if [[ -f "$dir/$name" ]]; then
            echo "$dir/$name"
            return
        fi
    done
    if command -v "$name" &>/dev/null; then
        command -v "$name"
        return
    fi
    echo ""
}

if [[ -z "$LLAMA_BENCH" ]]; then
    LLAMA_BENCH=$(find_binary "llama-bench")
fi
if [[ -z "$LLAMA_PPL" ]]; then
    LLAMA_PPL=$(find_binary "llama-perplexity")
fi

echo "llama-bench : ${LLAMA_BENCH:-not found}"
echo "llama-ppl   : ${LLAMA_PPL:-not found}"

# ─── Setup output ───────────────────────────────────────────────────────────

mkdir -p "$OUTPUT_DIR/logs"

TIMESTAMP=$(date +%Y%m%d_%H%M%S)
CSV_THROUGHPUT="${OUTPUT_DIR}/throughput_${TIMESTAMP}.csv"
CSV_PPL="${OUTPUT_DIR}/ppl_${TIMESTAMP}.csv"
SUMMARY="${OUTPUT_DIR}/summary_${TIMESTAMP}.txt"

# ─── System info ────────────────────────────────────────────────────────────

echo ""
echo "=== System Info ==="
CHIP=$(sysctl -n machdep.cpu.brand_string 2>/dev/null || echo "Unknown")
MEM_BYTES=$(sysctl -n hw.memsize 2>/dev/null || echo "0")
MEM_GB=$(( MEM_BYTES / 1073741824 ))
NCPU=$(sysctl -n hw.ncpu 2>/dev/null || echo "4")
PCPU=$(sysctl -n hw.perflevel0.physicalcpu 2>/dev/null || echo "?")
ECPU=$(sysctl -n hw.perflevel1.physicalcpu 2>/dev/null || echo "?")

echo "Chip    : $CHIP"
echo "Memory  : ${MEM_GB} GB"
echo "Cores   : $NCPU total (${PCPU}P + ${ECPU}E)"
echo "Model   : $(basename "$MODEL")"
echo ""

{
    echo "chip=$CHIP"
    echo "memory_gb=$MEM_GB"
    echo "cores=$NCPU"
    echo "p_cores=$PCPU"
    echo "e_cores=$ECPU"
} > "$OUTPUT_DIR/system_info.txt"

# ─── Benchmark matrix ──────────────────────────────────────────────────────

# KV cache configs: "label ctk ctv"
KV_CONFIGS=(
    "F16_F16   f16   f16"
    "Q8_F16    q8_0  f16"
    "TQ4_F16   tq4_0 f16"
    "TQ4_TQ4   tq4_0 tq4_0"
)

# ─── Throughput benchmark ──────────────────────────────────────────────────

if [[ "$SKIP_THROUGHPUT" == true ]]; then
    echo "Skipping throughput runs (--skip-throughput)"
elif [[ -z "$LLAMA_BENCH" || ! -f "$LLAMA_BENCH" ]]; then
    echo "Skipping throughput runs (llama-bench not found)"
else
    CONTEXT_LENGTHS=(512 2048 4096)
    GEN_TOKENS=128

    echo "=== Throughput Benchmark ==="
    echo "  Configs  : ${#KV_CONFIGS[@]} KV types"
    echo "  Contexts : ${CONTEXT_LENGTHS[*]}"
    echo "  Generate : $GEN_TOKENS tokens per run"
    echo "  Threads  : $NCPU"
    echo ""

    echo "label,ctk,ctv,n_prompt,n_gen,ngl,pp_tok_s,tg_tok_s,elapsed_sec" > "$CSV_THROUGHPUT"

    TOTAL_RUNS=$(( ${#KV_CONFIGS[@]} * ${#CONTEXT_LENGTHS[@]} ))
    RUN=0

    for config in "${KV_CONFIGS[@]}"; do
        read -r label ctk ctv <<< "$config"

        for n_prompt in "${CONTEXT_LENGTHS[@]}"; do
            RUN=$((RUN + 1))
            log_file="${OUTPUT_DIR}/logs/bench_${label}_p${n_prompt}.log"

            echo "[$RUN/$TOTAL_RUNS] $label | prompt=$n_prompt gen=$GEN_TOKENS"

            START_TIME=$(date +%s)

            # ngl=0: CPU-only (NEON path). Metal FA crashes on Qwen3.5
            # head_dim=256 (ne10 != ne02 assertion in ggml-metal.m:3282).
            "$LLAMA_BENCH" \
                -m "$MODEL" \
                -p "$n_prompt" \
                -n "$GEN_TOKENS" \
                -ngl 0 \
                -ctk "$ctk" \
                -ctv "$ctv" \
                -t "$NCPU" \
                2>&1 | tee "$log_file"

            END_TIME=$(date +%s)
            ELAPSED=$((END_TIME - START_TIME))

            # Parse llama-bench table: "| ... | pp512 | 53.37 ± 0.23 |" — extract number before ±
            pp_toks=$(grep "pp" "$log_file" | sed -n 's/.*| *\([0-9][0-9]*\.[0-9][0-9]*\) ±.*/\1/p' | tail -1)
            tg_toks=$(grep "tg" "$log_file" | sed -n 's/.*| *\([0-9][0-9]*\.[0-9][0-9]*\) ±.*/\1/p' | tail -1)
            pp_toks=${pp_toks:-N/A}
            tg_toks=${tg_toks:-N/A}

            echo "  => pp=${pp_toks} tok/s, tg=${tg_toks} tok/s (${ELAPSED}s)"
            echo "$label,$ctk,$ctv,$n_prompt,$GEN_TOKENS,0,$pp_toks,$tg_toks,$ELAPSED" >> "$CSV_THROUGHPUT"
            echo ""
        done
    done

    echo ""
    echo "=== Throughput results: $CSV_THROUGHPUT ==="
    cat "$CSV_THROUGHPUT"
fi

# ─── Perplexity benchmark ──────────────────────────────────────────────────

if [[ "$SKIP_PPL" == true ]]; then
    echo ""
    echo "Skipping perplexity runs (--skip-ppl)"
elif [[ -z "$LLAMA_PPL" || ! -f "$LLAMA_PPL" ]]; then
    echo ""
    echo "Skipping perplexity runs (llama-perplexity not found)"
elif [[ -z "$DATASET" || ! -f "$DATASET" ]]; then
    echo ""
    echo "Skipping perplexity runs (no --dataset provided)"
    echo "To run PPL: pass --dataset=path/to/wiki.test.raw"
else
    echo ""
    echo "=== Perplexity Benchmark ==="
    echo "label,ctk,ctv,n_ctx,ppl,ppl_stderr,elapsed_sec" > "$CSV_PPL"

    PPL_CTX=2048
    PPL_TOTAL=${#KV_CONFIGS[@]}
    PPL_RUN=0

    for config in "${KV_CONFIGS[@]}"; do
        read -r label ctk ctv <<< "$config"
        PPL_RUN=$((PPL_RUN + 1))
        log_file="${OUTPUT_DIR}/logs/ppl_${label}_c${PPL_CTX}.log"

        echo "[$PPL_RUN/$PPL_TOTAL] PPL: $label | n_ctx=$PPL_CTX"

        START_TIME=$(date +%s)

        # ngl=0: CPU-only (NEON). See throughput section for Metal crash note.
        "$LLAMA_PPL" \
            -m "$MODEL" \
            -f "$DATASET" \
            -c "$PPL_CTX" \
            -ngl 0 \
            -ctk "$ctk" \
            -ctv "$ctv" \
            -t "$NCPU" \
            --no-display-prompt \
            2>&1 | tee "$log_file"

        END_TIME=$(date +%s)
        ELAPSED=$((END_TIME - START_TIME))

        # macOS-compatible extraction (no grep -P)
        ppl=$(grep 'Final estimate: PPL' "$log_file" | grep -oE '[0-9]+\.[0-9]+' | head -1 || echo "FAILED")
        ppl_stderr=$(grep 'Final estimate: PPL' "$log_file" | grep -oE '[0-9]+\.[0-9]+' | tail -1 || echo "N/A")

        echo "  => PPL=$ppl +/- $ppl_stderr (${ELAPSED}s)"
        echo "$label,$ctk,$ctv,$PPL_CTX,$ppl,$ppl_stderr,$ELAPSED" >> "$CSV_PPL"
        echo ""
    done

    echo ""
    echo "=== PPL results: $CSV_PPL ==="
    cat "$CSV_PPL"
fi

# ─── Summary ────────────────────────────────────────────────────────────────

{
    echo "LeanKV Phase 3b — Apple M2 Benchmark Summary"
    echo "Date: $(date -u '+%Y-%m-%d %H:%M:%S UTC')"
    echo "Chip: $CHIP ($NCPU cores, ${MEM_GB} GB)"
    echo "Model: $(basename "$MODEL")"
    echo ""

    if [[ -f "$CSV_THROUGHPUT" ]]; then
        echo "=== Throughput (tok/s) ==="
        echo ""
        printf "%-10s | %-10s | %-10s | %-10s\n" "Config" "pp tok/s" "tg tok/s" "Context"
        printf "%-10s | %-10s | %-10s | %-10s\n" "----------" "----------" "----------" "----------"
        tail -n +2 "$CSV_THROUGHPUT" | while IFS=, read -r label ctk ctv n_prompt n_gen ngl pp tg elapsed; do
            printf "%-10s | %-10s | %-10s | %-10s\n" "$label" "$pp" "$tg" "$n_prompt"
        done
        echo ""
    fi

    if [[ -f "$CSV_PPL" ]]; then
        echo "=== Perplexity (WikiText-2) ==="
        echo ""
        printf "%-10s | %-7s | %-10s\n" "Config" "PPL" "+/- stderr"
        printf "%-10s | %-7s | %-10s\n" "----------" "-------" "----------"
        tail -n +2 "$CSV_PPL" | while IFS=, read -r label ctk ctv n_ctx ppl stderr elapsed; do
            printf "%-10s | %-7s | %s\n" "$label" "$ppl" "$stderr"
        done
    fi
} | tee "$SUMMARY"

echo ""
echo "=== All output in: $OUTPUT_DIR ==="
[[ -f "$CSV_THROUGHPUT" ]] && echo "  Throughput CSV : $CSV_THROUGHPUT"
[[ -f "$CSV_PPL" ]]        && echo "  Perplexity CSV : $CSV_PPL"
echo "  Summary        : $SUMMARY"
echo "  Logs           : $OUTPUT_DIR/logs/"
echo ""
echo "Done."
