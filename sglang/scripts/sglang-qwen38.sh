#!/usr/bin/env bash
# Qwen3.8-Flash-Next Uncensored (jpezzulli OrcaRouter ModelOpt-NVFP4) on TWO DGX Sparks via SGLang, TP2.
# Podman (rootless, sudo -u wfoster) integration of PixelML's 2x-DGX-Spark SGLang recipe.
# Applies the SM121 QSA/token0-guard patches (required on GB10), then launches.
# Usage: sglang-qwen38.sh <0|1>   (run rank 1 = worker FIRST, then rank 0 = head)
set -euo pipefail
NODE_RANK="${1:?usage: sglang-qwen38.sh <0|1>}"

# ---- our cluster fabric / model (substituted by swap-model.sh or real values) ----
IMAGE_REF="${IMAGE_REF:-docker.io/lmsysorg/sglang@sha256:12d3392bdc8be8d35e9a95f191df6aef99c5114bdbefd41bfdc7e760e6d25ec1}"
MODEL_DIR="${MODEL_DIR:-/models/orca-qwen3.8-modelopt-nvfp4}"
HEAD_IP="${HEAD_IP:-192.168.177.11}"; WORKER_IP="${WORKER_IP:-192.168.177.12}"
DIST_PORT="${DIST_PORT:-29522}"
API_PORT="${API_PORT:-8000}"
CONTEXT_LENGTH="${CONTEXT_LENGTH:-262144}"
YARN="${YARN:-}"   # empty = auto (on when >262K or ==1M); 1 = force; 0 = off
MAX_RUNNING_REQUESTS="${MAX_RUNNING_REQUESTS:-6}"
MEM_FRACTION_STATIC="${MEM_FRACTION_STATIC:-0.80}"
NCCL_SOCKET_IFNAME="${NCCL_SOCKET_IFNAME:-enp1s0f0np0}"
NCCL_IB_HCA="${NCCL_IB_HCA:-rocep1s0f0}"
NCCL_IB_GID_INDEX="${NCCL_IB_GID_INDEX:-3}"
NCCL_IB_ADDR_RANGE="${NCCL_IB_ADDR_RANGE:-192.168.177.0/24}"
NCCL_HOME="${NCCL_HOME:-$HOME/nccl230/build}"
# Resolve the symlink to the real .so so LD_PRELOAD doesn't hit a dangling pointer.
_NCCL_REAL="$(readlink -f "$NCCL_HOME/lib/libnccl.so.2" 2>/dev/null)"
[ -n "$_NCCL_REAL" ] || _NCCL_REAL="$NCCL_HOME/lib/libnccl.so.2"
NCCL_PRELOAD="${NCCL_PRELOAD:-$_NCCL_REAL}"
[ -f "$NCCL_PRELOAD" ] || NCCL_PRELOAD=""
NAME="${NAME:-qwen38fn-sglang}"

case "$NODE_RANK" in
  0) DIST_INIT_ADDR="$HEAD_IP:$DIST_PORT" ;;
  1) DIST_INIT_ADDR="$HEAD_IP:$DIST_PORT" ;;
  *) echo "rank must be 0 or 1" >&2; exit 2 ;;
esac

SELF="$(readlink -f "${BASH_SOURCE[0]}")"
REPO_ROOT="$(cd "$(dirname "$SELF")/../.." && pwd)"
BUILD_DIR="$REPO_ROOT/build"
SECRET_FILE="$REPO_ROOT/.sglang-api-key"

# ---- ensure SM121 patches are staged (podman-aware) ----
APPLY_SCRIPT="$REPO_ROOT/sglang/scripts/apply-sm121-podman.sh"
if [[ ! -f "$BUILD_DIR/sm121_varlen.py" ]]; then
  IMAGE_REF="$IMAGE_REF" bash "$APPLY_SCRIPT"
fi

# ---- API key secret (must exist) ----
[[ -s "$SECRET_FILE" ]] || { echo "API secret missing: $SECRET_FILE (run: openssl rand -hex 32)" >&2; exit 1; }
chmod 600 "$SECRET_FILE"

# ---- YaRN: enable when context >262K (or forced). Factor = round(ctx/262144). ----
YARN_ENABLED=0
if [ -n "$YARN" ]; then
  [ "$YARN" != "0" ] && YARN_ENABLED=1
elif [ "$CONTEXT_LENGTH" -gt 262144 ]; then
  YARN_ENABLED=1
fi
OVERRIDE_JSON=""
if [ "$YARN_ENABLED" = "1" ]; then
  FACTOR=$(( (CONTEXT_LENGTH + 131071) / 262144 ))  # round(ctx/262144)
  OVERRIDE_JSON="{\"text_config\":{\"rope_scaling\":{\"type\":\"yarn\",\"factor\":$FACTOR,\"mrope_section\":[11,11,10],\"partial_rotary_factor\":0.25}}}"
  OVER_EXTRA="--json-model-override-args '$OVERRIDE_JSON'"
  echo "YaRN ON: ctx=$CONTEXT_LENGTH factor=$FACTOR"
else
  OVER_EXTRA=""
fi

# ---- podman cleanup + run ----
sudo -u wfoster podman rm -f "$NAME" 2>/dev/null || true
sudo -u wfoster podman run --detach \
  --name "$NAME" --restart unless-stopped \
  --network host --ipc host --device nvidia.com/gpu=all \
  --ulimit memlock=-1:-1 --cap-add IPC_LOCK \
  --device /dev/infiniband \
  --log-opt max-size=100m --log-opt max-file=5 \
  --env NODE_RANK="$NODE_RANK" \
  --env DIST_INIT_ADDR="$DIST_INIT_ADDR" \
  --env API_PORT="$API_PORT" \
  --env CONTEXT_LENGTH="$CONTEXT_LENGTH" \
  --env MAX_RUNNING_REQUESTS="$MAX_RUNNING_REQUESTS" \
  --env MEM_FRACTION_STATIC="$MEM_FRACTION_STATIC" \
  --env SGLANG_API_KEY_FILE=/run/secrets/sglang-api-key \
  --env HF_HUB_OFFLINE=1 --env TRANSFORMERS_OFFLINE=1 \
  --env NCCL_SOCKET_IFNAME="$NCCL_SOCKET_IFNAME" \
  --env GLOO_SOCKET_IFNAME="$NCCL_SOCKET_IFNAME" \
  --env NCCL_NET=IB --env NCCL_IB_DISABLE=0 \
  --env NCCL_IB_HCA="$NCCL_IB_HCA" \
  --env NCCL_IB_GID_INDEX="$NCCL_IB_GID_INDEX" \
  --env NCCL_IB_ADDR_RANGE="$NCCL_IB_ADDR_RANGE" \
  --env NCCL_IB_ADDR_FAMILY=AF_INET --env NCCL_IB_ROCE_VERSION_NUM=2 \
  --env NCCL_IB_TIMEOUT=22 --env NCCL_IB_RETRY_CNT=7 \
  --env NCCL_IB_SUBNET_AWARE_ROUTING=1 --env NCCL_IB_MERGE_NICS=0 \
  --env NCCL_MAX_NCHANNELS=4 --env NCCL_MIN_NCHANNELS=4 --env NCCL_CROSS_NIC=1 \
  --env NCCL_CUMEM_ENABLE=0 --env NCCL_NVLS_ENABLE=0 --env NCCL_NET_PLUGIN=none \
  --env TORCH_CUDA_ARCH_LIST=12.1a --env FLASHINFER_CUDA_ARCH_LIST=12.1a \
  --env TORCH_NCCL_ASYNC_ERROR_HANDLING=1 \
  --env TP_SOCKET_IFNAME="$NCCL_SOCKET_IFNAME" --env MN_IF_NAME="$NCCL_SOCKET_IFNAME" \
  --env NCCL_IGNORE_CPU_AFFINITY=1 --env NCCL_DEBUG=INFO --env NCCL_DEBUG_SUBSYS=COLL \
  --env NCCL_ASYNC_ERROR_HANDLING=1 --env NCCL_BLOCKING_WAIT=1 \
  --env TORCH_DISTRIBUTED_TIMEOUT=1800 \
  --env TORCH_NCCL_DUMP_ON_TIMEOUT=1 --env TORCH_NCCL_ENABLE_MONITORING=1 \
  --env NCCL_HOME="/usr/local/lib" \
  ${NCCL_PRELOAD:+--env "LD_PRELOAD=/usr/local/lib/libnccl.so.2"} \
  ${NCCL_PRELOAD:+--mount "type=bind,src=$NCCL_PRELOAD,dst=/usr/local/lib/libnccl.so.2,readonly"} \
  ${NCCL_PRELOAD:+--mount "type=bind,src=$NCCL_PRELOAD,dst=/usr/local/lib/python3.12/dist-packages/nvidia/nccl/lib/libnccl.so.2,readonly"} \
  --env TORCHINDUCTOR_CACHE_DIR=/cache/torch --env TRITON_CACHE_DIR=/cache/triton \
  --env SGLANG_ALLOW_OVERWRITE_LONGER_CONTEXT_LEN=1 \
  ${OVERRIDE_JSON:+--env "OVERRIDE_JSON=$OVERRIDE_JSON"} \
  --mount type=bind,src="$MODEL_DIR",dst=/models/qwen38-flash-next,readonly \
  --mount type=bind,src="$SECRET_FILE",dst=/run/secrets/sglang-api-key,readonly \
  --mount type=bind,src="$REPO_ROOT/sglang/scripts/launch_secure.py",dst=/opt/qwen/launch_secure.py,readonly \
  --mount type=bind,src="$REPO_ROOT/sglang/scripts/redact_stream.py",dst=/opt/qwen/redact_stream.py,readonly \
  --mount type=bind,src="$BUILD_DIR/qwen_sparse_attn_backend.py",dst=/sgl-workspace/sglang/python/sglang/srt/layers/attention/qwen_sparse_attn_backend.py,readonly \
  --mount type=bind,src="$BUILD_DIR/sm121_varlen.py",dst=/sgl-workspace/sglang/python/sglang/srt/layers/attention/qsa/sm121_varlen.py,readonly \
  --mount type=bind,src="$BUILD_DIR/schedule_batch.py",dst=/sgl-workspace/sglang/python/sglang/srt/managers/schedule_batch.py,readonly \
  --mount type=bind,src="$BUILD_DIR/batch_result_processor.py",dst=/sgl-workspace/sglang/python/sglang/srt/managers/scheduler_components/batch_result_processor.py,readonly \
  --mount type=bind,src="$BUILD_DIR/scheduler.py",dst=/sgl-workspace/sglang/python/sglang/srt/managers/scheduler.py,readonly \
  --mount type=bind,src="$REPO_ROOT/cache/huggingface",dst=/root/.cache/huggingface \
  --entrypoint bash \
  "$IMAGE_REF" \
  -lc "set -o pipefail; python3 -u /opt/qwen/launch_secure.py \
    --model-path /models/qwen38-flash-next \
    --served-model-name qwen3.8-flash-next-uncensored \
    --tp 2 --nnodes 2 --node-rank \"$NODE_RANK\" \
    --dist-init-addr \"$DIST_INIT_ADDR\" \
    --quantization modelopt_fp4 --fp4-gemm-backend flashinfer_cutlass \
    --page-size 64 \
    --linear-attn-prefill-backend triton --linear-attn-decode-backend flashinfer \
    --mamba-ssm-dtype bfloat16 --mamba-scheduler-strategy extra_buffer --mamba-track-interval 64 --max-mamba-cache-size 97 \
    --speculative-algorithm NEXTN --speculative-num-steps 3 --speculative-eagle-topk 1 --speculative-num-draft-tokens 4 \
    --enable-linear-replayssm-spec --speculative-attention-mode decode \
    --ple-offload-embedding \
    --trust-remote-code \
    --stream-interval 32 \
    --chunked-prefill-size 4096 \
    --max-total-tokens 600000 \
    --cuda-graph-max-bs 8 --disable-cuda-graph-padding --disable-prefill-cuda-graph \
    --disable-radix-cache \
    --max-running-requests \"$MAX_RUNNING_REQUESTS\" \
    --context-length \"$CONTEXT_LENGTH\" \
    $OVER_EXTRA \
    --mem-fraction-static \"$MEM_FRACTION_STATIC\" \
    --allow-auto-truncate --reasoning-parser auto --tool-call-parser auto --sampling-defaults model \
    --sampling-backend pytorch \
    --watchdog-timeout 1200 --enable-metrics --host 0.0.0.0 --port \"$API_PORT\" 2>&1 | python3 -u /opt/qwen/redact_stream.py"

echo "started $NAME rank=$NODE_RANK dist=$DIST_INIT_ADDR tp=2 ctx=$CONTEXT_LENGTH gpu=$MEM_FRACTION_STATIC"
