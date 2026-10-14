#!/usr/bin/env bash
# benchmark.sh — Qwen3.8-Flash-Next-Uncensored perf harness on the 2-node GX10 cluster.
# Measures aggregate + per-stream decode tok/s and ACPI thermals across a
# concurrency ladder, for the current serving config. Reuses the DS4 harness
# method: completion_tokens / wall_time, non-streaming, standardized mixed-prose
# prompt (~short), max_tokens=320. ACPI thermal zones recorded via ssh.
#
# Call: benchmark.sh [c1 c2 c4 c8]   (default ladder: 1 2 4 6 8)
# Results append a dated table to tuning/RESULTS.md.
set -euo pipefail

API="http://127.0.0.1:8000/v1/chat/completions"
MODEL="${MODEL:-qwen3.8-flash-next-uncensored}"
MAX_TOKENS=320
PROMPT="${PROMPT:-Write a clear, well-structured 2-paragraph explanation of the Transformer attention mechanism, with an example.}"

log() { echo "[benchmark] $*"; }

n_tokens() { python3 -c "import sys,json; print(json.load(sys.stdin)['usage']['completion_tokens'])"; }

run_arm() {
  local c="$1"
  local results="" total_tok=0 wall_start wall_end
  # time the batch of c concurrent requests
  wall_start=$(date +%s.%N)
  for i in $(seq 1 "$c"); do
    ( curl -s "$API" -H 'Content-Type: application/json' \
        -d "{\"model\":\"$MODEL\",\"messages\":[{\"role\":\"user\",\"content\":\"$PROMPT\"}],\"max_tokens\":$MAX_TOKENS}" \
        > "/tmp/qwen_bench_${c}_${i}.json" ) &
  done
  wait
  wall_end=$(date +%s.%N)
  # sum completion tokens across all responses
  for i in $(seq 1 "$c"); do
    if [ -f "/tmp/qwen_bench_${c}_${i}.json" ]; then
      t=$(n_tokens < "/tmp/qwen_bench_${c}_${i}.json" 2>/dev/null || echo 0)
      total_tok=$((total_tok + t))
      rm -f "/tmp/qwen_bench_${c}_${i}.json"
    fi
  done
  local dt=$(echo "$wall_end - $wall_start" | bc -l)
  local agg=$(echo "$total_tok / $dt" | bc -l)
  local per=$(echo "$agg / $c" | bc -l)
  local temp=$(cat /sys/class/thermal/thermal_zone*/temp 2>/dev/null | sort -nr | head -1 | awk '{print $1/1000}')
  echo "$c $agg $per $temp" | awk '{printf "c=%s  agg=%.2f tok/s  per=%.2f tok/s  ACPI_max=%.1fC\n",$1,$2,$3,$4}'
  printf "| %s | %.2f | %.2f | %.1f C |\n" "$c" "$agg" "$per" "$temp" >> "$RESULTS"
}

RESULTS="$(dirname "$(readlink -f "$0")")/RESULTS.md"
[ -f "$RESULTS" ] || { printf '# Qwen3.8-Flash-Next-Uncensored — perf results\n\n| Concurrency | Aggregate tok/s | Per-stream tok/s | Max ACPI temp |\n|---|---|---|---|\n' > "$RESULTS"; }

log "benchmarking against $MODEL @ $API (max_tokens=$MAX_TOKENS)"
log "ladder: ${*:-1 2 4 6 8}"

DATE=$(date +%Y-%m-%d)
{
  echo ""
  echo "### Run $DATE (config: $MODEL)"
  echo "| Concurrency | Aggregate tok/s | Per-stream tok/s | Max ACPI temp |"
  echo "|---|---|---|---|"
} >> "$RESULTS"

for c in ${*:-1 2 4 6 8}; do
  run_arm "$c"
done
log "results appended to $RESULTS"
