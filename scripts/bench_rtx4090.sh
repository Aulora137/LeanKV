#!/usr/bin/env bash
# LeanKV Phase 3b — RTX 4090 GPU Benchmark
#
# Benchmarks TQ4_0 KV cache quantization on CUDA across multiple context
# lengths. Measures decode tok/s, prefill tok/s, and KV memory usage.
#
# Usage:
#   # 1. Set up on Vast.ai instance:
#   git clone https://github.com/hchengit/Lean_llama.cpp.git -b leanKV-tq-integration
#   cd Lean_llama.cpp
#   cmake -B build -DGGML_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=89 -DCMAKE_BUILD_TYPE=Release
#   cmake --build build --config Release -j$(nproc)
#
#   # 2. Download model:
#   pip install huggingface-hub
#   huggingface-cli download Qwen/Qwen3.5-9B-Q4_K_M-GGUF --local-dir models/
#   # (or whatever repo hosts the Q4_K_M GGUF)
#
#   # 3. Run benchmark:
#   ./bench_rtx4090.sh --model models/Qwen3.5-9B-Q4_K_M.gguf
#
# Flags:
#   --model PATH       Path to GGUF model (required)
#   --llama-bench PATH Path to llama-bench binary (default: auto-detect)
#   --llama-ppl PATH   Path to llama-perplexity binary (default: auto-detect)
#   --dataset PATH     Path to WikiText-2 raw file (optional, for PPL runs)
#   --output DIR       Output directory (default: ./results-rtx4090)
#   --skip-ppl         Skip perplexity runs (faster, throughput only)
#   --warmup N         Warmup tokens before measurement (default: 32)

set -euo pipefail

# ─── Defaults ───────────────────────────────────────────────────────────────

MODEL=""
LLAMA_BENCH=""
LLAMA_PPL=""
DATASET=""
OUTPUT_DIR="./results-rtx4090"
SKIP_PPL=false
WARMUP=32

# ─── Parse args ─────────────────────────────────────────────────────────────

for arg in "$@"; do
    case "$arg" in
        --model=*)       MODEL="${arg#--model=}" ;;
        --llama-bench=*) LLAMA_BENCH="${arg#--llama-bench=}" ;;
        --llama-ppl=*)   LLAMA_PPL="${arg#--llama-ppl=}" ;;
        --dataset=*)     DATASET="${arg#--dataset=}" ;;
        --output=*)      OUTPUT_DIR="${arg#--output=}" ;;
        --skip-ppl)      SKIP_PPL=true ;;
        --warmup=*)      WARMUP="${arg#--warmup=}" ;;
        --help|-h)
            head -30 "$0" | tail -25
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
    for dir in ./build/bin ../build/bin ./build .; do
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

if [[ -z "$LLAMA_BENCH" || ! -f "$LLAMA_BENCH" ]]; then
    echo "ERROR: llama-bench not found. Specify --llama-bench=PATH"
    exit 1
fi
echo "llama-bench : $LLAMA_BENCH"
echo "llama-ppl   : ${LLAMA_PPL:-not found (PPL runs will be skipped)}"

# ─── Setup output ───────────────────────────────────────────────────────────

mkdir -p "$OUTPUT_DIR/logs"

TIMESTAMP=$(date +%Y%m%d_%H%M%S)
CSV_THROUGHPUT="${OUTPUT_DIR}/throughput_${TIMESTAMP}.csv"
CSV_PPL="${OUTPUT_DIR}/ppl_${TIMESTAMP}.csv"
SUMMARY="${OUTPUT_DIR}/summary_${TIMESTAMP}.txt"

# ─── GPU info ───────────────────────────────────────────────────────────────

echo ""
echo "=== GPU Info ==="
if command -v nvidia-smi &>/dev/null; then
    nvidia-smi --query-gpu=name,memory.total,driver_version,compute_cap \
        --format=csv,noheader | tee "$OUTPUT_DIR/gpu_info.txt"
    echo ""
    nvidia-smi
else
    echo "WARNING: nvidia-smi not found"
fi

# ─── Benchmark matrix ──────────────────────────────────────────────────────

# KV cache configs: "label ctk ctv"
KV_CONFIGS=(
    "F16_F16   f16   f16"
    "Q8_F16    q8_0  f16"
    "TQ4_F16   tq4_0 f16"
    "TQ4_TQ4   tq4_0 tq4_0"
)

# Context lengths: prompt tokens (prefill), then generate N tokens (decode)
# Short context tests decode speed; long context tests KV memory pressure
CONTEXT_LENGTHS=(512 2048 8192 16384 32768)
GEN_TOKENS=128

echo ""
echo "=== Throughput Benchmark Matrix ==="
echo "  Model    : $(basename "$MODEL")"
echo "  Configs  : ${#KV_CONFIGS[@]} KV types"
echo "  Contexts : ${CONTEXT_LENGTHS[*]}"
echo "  Generate : $GEN_TOKENS tokens per run"
echo "  Total    : $(( ${#KV_CONFIGS[@]} * ${#CONTEXT_LENGTHS[@]} )) runs"
echo ""

# CSV header
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

        # llama-bench outputs a markdown table with pp (prefill) and tg (decode) tok/s
        "$LLAMA_BENCH" \
            -m "$MODEL" \
            -p "$n_prompt" \
            -n "$GEN_TOKENS" \
            -ngl 99 \
            -ctk "$ctk" \
            -ctv "$ctv" \
            -t "$(nproc)" \
            2>&1 | tee "$log_file"

        END_TIME=$(date +%s)
        ELAPSED=$((END_TIME - START_TIME))

        # Extract pp and tg from llama-bench output
        # llama-bench outputs lines like: "| model | ... | pp ... | tg ... |"
        # The last two numeric columns before the final "|" are pp tok/s and tg tok/s
        pp_toks=$(grep -oP 'pp512|pp\s+\K[0-9.]+' "$log_file" 2>/dev/null | tail -1 || echo "")
        tg_toks=$(grep -oP 'tg128|tg\s+\K[0-9.]+' "$log_file" 2>/dev/null | tail -1 || echo "")

        # Fallback: parse the table format that llama-bench actually outputs
        # Format: | model | size | params | backend | ngl | ... | test | t/s |
        if [[ -z "$pp_toks" || -z "$tg_toks" ]]; then
            pp_toks=$(grep "pp" "$log_file" | grep -oP '[0-9]+\.[0-9]+\s*$' | tr -d ' ' || echo "N/A")
            tg_toks=$(grep "tg" "$log_file" | grep -oP '[0-9]+\.[0-9]+\s*$' | tr -d ' ' || echo "N/A")
        fi

        echo "  => pp=${pp_toks} tok/s, tg=${tg_toks} tok/s (${ELAPSED}s)"
        echo "$label,$ctk,$ctv,$n_prompt,$GEN_TOKENS,99,$pp_toks,$tg_toks,$ELAPSED" >> "$CSV_THROUGHPUT"
        echo ""
    done
done

echo ""
echo "=== Throughput results: $CSV_THROUGHPUT ==="
cat "$CSV_THROUGHPUT"

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
    echo "To run PPL: download WikiText-2 and pass --dataset=path/to/wiki.test.raw"
else
    echo ""
    echo "=== Perplexity Benchmark ==="
    echo "label,ctk,ctv,n_ctx,ppl,ppl_stderr,elapsed_sec" > "$CSV_PPL"

    PPL_CONTEXTS=(2048 8192)
    PPL_TOTAL=$(( ${#KV_CONFIGS[@]} * ${#PPL_CONTEXTS[@]} ))
    PPL_RUN=0

    for config in "${KV_CONFIGS[@]}"; do
        read -r label ctk ctv <<< "$config"

        for n_ctx in "${PPL_CONTEXTS[@]}"; do
            PPL_RUN=$((PPL_RUN + 1))
            log_file="${OUTPUT_DIR}/logs/ppl_${label}_c${n_ctx}.log"

            echo "[$PPL_RUN/$PPL_TOTAL] PPL: $label | n_ctx=$n_ctx"

            START_TIME=$(date +%s)

            "$LLAMA_PPL" \
                -m "$MODEL" \
                -f "$DATASET" \
                -c "$n_ctx" \
                -ngl 99 \
                -ctk "$ctk" \
                -ctv "$ctv" \
                -t "$(nproc)" \
                --no-display-prompt \
                2>&1 | tee "$log_file"

            END_TIME=$(date +%s)
            ELAPSED=$((END_TIME - START_TIME))

            ppl=$(grep -oP 'Final estimate: PPL .* = \K[0-9.]+' "$log_file" || echo "FAILED")
            ppl_stderr=$(grep -oP 'Final estimate: PPL .* = [0-9.]+ \+/- \K[0-9.]+' "$log_file" || echo "N/A")

            echo "  => PPL=$ppl +/- $ppl_stderr (${ELAPSED}s)"
            echo "$label,$ctk,$ctv,$n_ctx,$ppl,$ppl_stderr,$ELAPSED" >> "$CSV_PPL"
            echo ""
        done
    done

    echo ""
    echo "=== PPL results: $CSV_PPL ==="
    cat "$CSV_PPL"
fi

# ─── Summary ────────────────────────────────────────────────────────────────

{
    echo "LeanKV Phase 3b — RTX 4090 Benchmark Summary"
    echo "Date: $(date -u '+%Y-%m-%d %H:%M:%S UTC')"
    echo "Model: $(basename "$MODEL")"
    echo ""

    if command -v nvidia-smi &>/dev/null; then
        echo "GPU: $(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader | head -1)"
        echo ""
    fi

    echo "=== Throughput (tok/s) ==="
    echo ""
    printf "%-10s" "Config"
    for ctx in "${CONTEXT_LENGTHS[@]}"; do
        printf " | pp%-5s tg%-5s" "$ctx" "$ctx"
    done
    echo ""
    printf "%-10s" "----------"
    for ctx in "${CONTEXT_LENGTHS[@]}"; do
        printf " | %-13s" "-------------"
    done
    echo ""

    for config in "${KV_CONFIGS[@]}"; do
        read -r label _ _ <<< "$config"
        printf "%-10s" "$label"
        for ctx in "${CONTEXT_LENGTHS[@]}"; do
            pp=$(grep "^${label}," "$CSV_THROUGHPUT" | grep ",$ctx," | cut -d, -f7)
            tg=$(grep "^${label}," "$CSV_THROUGHPUT" | grep ",$ctx," | cut -d, -f8)
            printf " | %-6s %-6s" "${pp:-N/A}" "${tg:-N/A}"
        done
        echo ""
    done

    if [[ -f "$CSV_PPL" ]]; then
        echo ""
        echo "=== Perplexity ==="
        echo ""
        echo "Config     | n_ctx | PPL     | +/- stderr"
        echo "-----------|-------|---------|----------"
        tail -n +2 "$CSV_PPL" | while IFS=, read -r label ctk ctv n_ctx ppl stderr elapsed; do
            printf "%-10s | %-5s | %-7s | %s\n" "$label" "$n_ctx" "$ppl" "$stderr"
        done
    fi
} | tee "$SUMMARY"

echo ""
echo "=== All output in: $OUTPUT_DIR ==="
echo "  Throughput CSV : $CSV_THROUGHPUT"
[[ -f "$CSV_PPL" ]] && echo "  Perplexity CSV : $CSV_PPL"
echo "  Summary        : $SUMMARY"
echo "  Logs           : $OUTPUT_DIR/logs/"
echo ""
echo "Done."
