#!/usr/bin/env bash
# Qwen3.8-Flash-Next-Uncensored-NVFP4 (orcarouter), vLLM TP=2 on 2x ASUS GX10 (GB10/sm_121).
# No YaRN: native 262K context (community: quality degrades hard past ~390K; MTP
# draft head stays at 262K -> geometry failures past 524K). 262K is the vetted max.
# Direct podman-run lane (separate stack from DS4 b12x recipe + GLM tonyd2wild).
# Adapted from glm53-vllm.sh for this cluster fabric (verified 2026-10-14).
# Worker (rank 1) FIRST, then head (rank 0). HEAD-first would hit the TCPStore race.
set -euo pipefail

NODE_RANK="${1:?usage: qwen38-vllm.sh <0|1>}"
[[ "$NODE_RANK" == "0" || "$NODE_RANK" == "1" ]] || { echo "rank must be 0 or 1" >&2; exit 2; }

IMAGE="vllm/vllm-openai:qwen38-flash-next"
NAME="vllm_qwen38"
MODEL_PATH="/models/orcarouter-qwen3.8-flash-next-uncensored-nvfp4"
CACHE_HOST_PATH="/var/tmp/qwen38-vllm-cache"
# cluster fabric: use the ConnectX RoCE x0 link (internal cross-node fabric only).
# Fill HEAD_IP / HOST_IP with your RoCE x0 IPs (rank0=head). No management LAN IPs.
HEAD_IP="<HEAD_ROCE_IP>"
MPORT="29522"
PORT="8000"

case "$NODE_RANK" in
  0) HOST_IP="<HEAD_ROCE_IP>"; HEADLESS="" ;;
  1) HOST_IP="<WORKER_ROCE_IP>"; HEADLESS="--headless" ;;
esac

test -f "$MODEL_PATH/config.json" || { echo "MODEL MISSING: $MODEL_PATH" >&2; exit 4; }
mkdir -p "$CACHE_HOST_PATH"

podman rm -f "$NAME" 2>/dev/null || true

podman run -d \
  --name "$NAME" --restart no \
  --device nvidia.com/gpu=all \
  --network host --ipc host \
  --ulimit memlock=-1:-1 --cap-add IPC_LOCK \
  --device /dev/infiniband:/dev/infiniband \
  -v "$MODEL_PATH:$MODEL_PATH:ro" \
  -v "$CACHE_HOST_PATH:/cache" \
  -e VLLM_HOST_IP="$HOST_IP" \
  -e HF_HOME=/cache/huggingface \
  -e HF_HUB_OFFLINE=1 -e TRANSFORMERS_OFFLINE=1 \
  -e VLLM_ENGINE_READY_TIMEOUT_S=3600 \
  -e PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  -e TORCH_CUDA_ARCH_LIST=12.1a -e FLASHINFER_CUDA_ARCH_LIST=12.1a \
  -e FLASHINFER_DISABLE_VERSION_CHECK=1 \
  -e NCCL_NET=IB -e NCCL_IB_DISABLE=0 \
  -e NCCL_IB_HCA=rocep1s0f0 -e NCCL_IB_GID_INDEX=3 \
  -e NCCL_IB_ROCE_VERSION_NUM=2 -e NCCL_IB_ADDR_FAMILY=AF_INET \
  -e NCCL_IB_ADDR_RANGE="${NCCL_IB_ADDR_RANGE:-<ROCE_SUBNET>}" \
  -e NCCL_SOCKET_IFNAME=enp1s0f0np0 -e GLOO_SOCKET_IFNAME=enp1s0f0np0 \
  -e TP_SOCKET_IFNAME=enp1s0f0np0 -e MN_IF_NAME=enp1s0f0np0 \
  -e NCCL_NVLS_ENABLE=0 -e NCCL_CROSS_NIC=0 -e NCCL_IB_MERGE_NICS=0 \
  -e NCCL_CUMEM_ENABLE=0 -e NCCL_IGNORE_CPU_AFFINITY=1 -e NCCL_DEBUG=WARN \
  -e TORCH_NCCL_ASYNC_ERROR_HANDLING=1 \
  "$IMAGE" \
    "$MODEL_PATH" \
    --served-model-name qwen3.8-flash-next-uncensored \
    --host 0.0.0.0 --port "$PORT" \
    --trust-remote-code \
    --tensor-parallel-size 2 \
    --enable-expert-parallel \
    --all2all-backend allgather_reducescatter \
    --gpu-memory-utilization 0.80 \
    --max-model-len 262144 \
    --max-num-seqs 8 \
    --block-size 16 \
    --max-num-batched-tokens 8192 \
    --kv-cache-dtype fp8 \
    --mamba-cache-mode align \
    --enable-prefix-caching \
    --enable-chunked-prefill \
    --quantization compressed-tensors \
    --load-format safetensors \
    --mm-encoder-tp-mode data \
    --speculative-config '{"method":"mtp","num_speculative_tokens":3}' \
    --gdn-decode-kernel b12x \
    --linear-backend b12x \
    --moe-backend b12x \
    --reasoning-parser qwen3 \
    --tool-call-parser qwen3_coder \
    --enable-auto-tool-choice \
    --compilation-config '{"pass_config":{"fuse_act_quant":true}}' \
    --distributed-executor-backend mp \
    --nnodes 2 --node-rank "$NODE_RANK" \
    --master-addr "$HEAD_IP" --master-port "$MPORT" \
    $HEADLESS

echo "launched $NAME rank=$NODE_RANK host=$HOST_IP"
sleep 2
podman ps --format '{{.Names}} {{.Status}}' | grep "$NAME" || { echo "$NAME exited; inspect with: podman logs $NAME" >&2; exit 1; }
