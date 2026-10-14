#!/usr/bin/env bash
# Stage SM121 QSA + token-0 guard patches from the SGLang image, podman-aware.
# Produces build/ files that sglang-qwen38.sh bind-mounts into the container.
set -euo pipefail
SELF="$(readlink -f "${BASH_SOURCE[0]}")"
REPO_ROOT="$(cd "$(dirname "$SELF")/../.." && pwd)"
IMAGE_REF="${IMAGE_REF:-docker.io/lmsysorg/sglang@sha256:12d3392bdc8be8d35e9a95f191df6aef99c5114bdbefd41bfdc7e760e6d25ec1}"
PATCH_DIR="${REPO_ROOT}/sglang/patches"
OUTPUT_DIR="${REPO_ROOT}/build"
SRT="/sgl-workspace/sglang/python/sglang/srt"
mkdir -p "$OUTPUT_DIR"

podman_cat() { sudo -u wfoster podman run --rm --entrypoint cat "$IMAGE_REF" "$1"; }

entries=(
  "qwen_sparse_attn_backend.py|${PATCH_DIR}/sglang-qsa-sm121-fallback.patch|${SRT}/layers/attention/qwen_sparse_attn_backend.py"
  "schedule_batch.py|${PATCH_DIR}/sglang-token0-guard-schedule_batch.patch|${SRT}/managers/schedule_batch.py"
  "batch_result_processor.py|${PATCH_DIR}/sglang-token0-guard-batch-result-processor.patch|${SRT}/managers/scheduler_components/batch_result_processor.py"
  "scheduler.py|${PATCH_DIR}/sglang-token0-guard-scheduler.patch|${SRT}/managers/scheduler.py"
  "sm121_varlen.py|${PATCH_DIR}/sm121_varlen.py|-"
)

for e in "${entries[@]}"; do
  IFS='|' read -r out patch imgpath <<< "$e"
  tmp="$OUTPUT_DIR/$out.tmp"
  if [ "$imgpath" = "-" ]; then
    cp "$patch" "$tmp"
  else
    podman_cat "$imgpath" > "$tmp"
    patch --quiet --unified "$tmp" "$patch" || { echo "PATCH FAIL: $out" >&2; rm -f "$tmp"; exit 1; }
  fi
  python3 -m py_compile "$tmp" || { echo "COMPILE FAIL: $out" >&2; rm -f "$tmp"; exit 1; }
  mv "$tmp" "$OUTPUT_DIR/$out"
  echo "prepared $out"
done
echo "all SM121/token0 patches staged in $OUTPUT_DIR"
