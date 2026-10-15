#!/usr/bin/env bash
# swap-model.sh — bring up the last-used model on boot, or --swap to another.
# Reuses the vetted 2-node orchestration (head-first for SGLang Qwen, idempotent
# ensure, health poll, bounded port-free wait, RoCE pinning) and just points to
# the right container + launcher per lane.
#
# Call: swap-model.sh            # ensure current_model is serving (idempotent) + motd
#       swap-model.sh --health   # report serving status / health
#       swap-model.sh --swap <deepseek|glm|qwen38>  # stop current, launch <m>, update motd
#       swap-model.sh --stop     # cleanly stop ALL lanes + free :8000 (default = last persists)
#       swap-model.sh restart    # clean stop all lanes, relaunch the default (last-used) model
set -euo pipefail

STATE=/etc/vllm/current_model
# Management/LAN reachability for the swap ssh. These are only used to reach the
# peer for the podman rm (the container launch itself uses the RoCE CX7 fabric).
LEO="192.168.0.226"
RAPH="192.168.0.143"

# ---- lane launcher dispatch (queue order per lane: GLM/deepseek worker-first;
#       SGLang Qwen head-first) ----
launch_lane() {
  local m="$1"
  case "$m" in
    deepseek) $LAUNCHER ;;
    glm)
      # GLM is served by vLLM (tp2, 2-node, distributed-executor-backend mp).
      # The launcher (glm53-vllm.sh) is rank-aware ($1 = 0|1) and uses
      # --nnodes 2 --master-addr 192.168.177.11:29521. The mp rendezvous needs
      # BOTH ranks up, but swap-model.sh previously launched ONLY rank 0 so the
      # 2-node job never joined (rendezvous blocked -> /health never 200). The
      # launcher's own comment says rank 1 (worker) FIRST, then rank 0 (head) -
      # the opposite of the SGLang Qwen lane (head-first). Launch the worker on
      # raph first, then the head on leo. No port wait here: rank 0 (head) binds
      # :29521 itself; a wait would be circular.
      echo "   launching worker (rank 1) on raph... (worker-first, per vLLM mp)"
      sudo -u wfoster ssh -o BatchMode=yes -o ConnectTimeout=10 "$RAPH" \
        "bash /home/wfoster/glm53-vllm.sh 1" 2>/dev/null || true
      sleep 3
      echo "   head (rank 0) launched on leo..."
      sudo -u wfoster bash /home/wfoster/glm53-vllm.sh 0
      ;;
    qwen38)
      # Qwen is served by vLLM (MiaAI native stack, tp2/2-node via the runbook).
      # The runbook's start.sh orchestrates head+worker itself (patch, launch,
      # health); --launch skips download/sync (weights cached on both nodes).
      # MODEL_ID / SERVED_MODEL_NAME / ABLIT come from the qwen model.conf +
      # flags .env that the lane writes first. SERVED_MODEL_NAME stays
      # qwen3.8-flash-next-uncensored so litellm/clients are unchanged.
      # Detach (setsid nohup) because start.sh --launch blocks on its own
      # UNBOUNDED /health loop, which would defeat the bounded readiness wait
      # below (a capture hang would hang --swap forever). Backgrounding lets
      # swap-model.sh's own 20-min poll own readiness.
      setsid nohup sudo -u wfoster env SPARK_VLLM_MODEL_CONF=/home/wfoster/.config/spark-vllm/qwen-model.conf \
        bash /home/wfoster/.local/bin/miaai_launch.sh >/home/wfoster/qwen38_launch.log 2>&1 < /dev/null &
      ;;
  esac
}
# Each lane is its own bake. Swap logic lives here; lane specifics live in the
# lane's launcher/script. The DeepSeek lane uses the b12x recipe stack (run-recipe.py).
case "${2:-${CURRENT_MODEL:-deepseek}}" in
  deepseek)
    CONTAINER=b12x-vllm-node
    # Detach the b12x launch (nohup setsid) so the engine survives swap-model.sh
    # exiting after the readiness wait / a model-tui process-timeout. run-recipe.py
    # already background-drops per-node, but wrapping in setsid orphans the whole
    # launch from the caller's process tree so a killed subprocess can't take it
    # down (the "deepseek cleared but never relaunched" bug).
    LAUNCHER="sudo -u wfoster setsid nohup /opt/vllm-recipe/.build/spark-vllm-docker/run-recipe.py /opt/vllm-recipe/recipes/orcarouter-eugr-1m.yaml -d >/home/wfoster/deepseek_launch.log 2>&1 < /dev/null &"
    MODEL_DIR=/models/orcarouter-deepseek-v4-flash-vision-uncensored
    ;;
  glm)
    CONTAINER=vllm_glm53
    LAUNCHER="sudo -u wfoster bash /home/wfoster/glm53-vllm.sh 0"
    MODEL_DIR=/models/glm-5.3-flash-nvfp4
    ;;
  qwen38)
    CONTAINER=vllm-fn
    LAUNCHER="sudo -u wfoster env SPARK_VLLM_MODEL_CONF=/home/wfoster/.config/spark-vllm/qwen-model.conf bash /home/wfoster/.local/bin/miaai_launch.sh"
    MODEL_DIR=/home/wfoster/.cache/huggingface/hub/models--drowzeys--keys-Qwen3.8-Flash-Next-NVFP4-dual-ablit-house-qsa-L3-47
    ;;
  *) echo "unknown model '${2:-}' (use deepseek|glm|qwen38)" >&2; exit 2 ;;
esac

log() { echo "[swap-model] $*"; }

write_state() { sudo mkdir -p "$(dirname "$STATE")"; echo "$1" | sudo tee "$STATE" >/dev/null; }
read_state() { [ -f "$STATE" ] && cat "$STATE" || echo deepseek; }

write_motd() {
  local m="$1"
  sudo tee /etc/motd >/dev/null <<EOF

  GX10 2-node cluster -- HEAD node (rank 0 / front door)

  LIVE MODEL: $m
    operate: swap-model.sh [--health] [--swap <deepseek|glm|qwen38>]
    TUI:     sudo model-tui   (arrow-select, Enter action, Esc cancel, q quit)

EOF
}

# Map a lane key (deepseek/glm/qwen38) to the served-model-name it exposes on
# /v1/models, so model_serving() can detect "already up" idempotently. Without
# this, the boot/ensure path greps for e.g. "qwen38" but the server exposes
# "qwen3.8-flash-next-uncensored" -> false negative -> relaunch every boot.
served_name() {
  case "$1" in
    qwen38)           echo "qwen3.8-flash-next-uncensored" ;;
    glm)              echo "dgx_hobo_default" ;;
    deepseek)         echo "deepseek-v4-flash-vision-exp" ;;
    *)                echo "$1" ;;
  esac
}
# Auth header for a lane's /v1/models probe. All lanes now answer open on
# :8000 (deepseek/glm vLLM, and qwen on the vLLM MiaAI stack). The old SGLang
# qwen lane was bearer-gated; vLLM is not, so we send no header.
lane_key() {
  return 0
}
model_serving() {
  local want; want=$(served_name "$1")
  local hdr; hdr=$(lane_key "$1")
  curl -fsS -H "$hdr" http://127.0.0.1:8000/v1/models 2>/dev/null | grep -qi "$want"
}

stop_stack() {
  # Stop ALL known lane containers (not just CONTAINER), because the currently
  # serving lane may differ from the target on a swap. Leaving an orphaned
  # container running on :8000 is a regression (swap X->Y->X double-binds).
  echo "==> stopping all lane containers on head + worker (deepseek/glm/qwen)"
  for c in b12x-vllm-node vllm_glm53 qwen38fn-sglang vllm-fn; do
    sudo -u wfoster podman rm -f "$c" 2>/dev/null || true
    sudo -u wfoster ssh -o BatchMode=yes -o ConnectTimeout=10 "$RAPH" \
      "podman rm -f $c 2>/dev/null || true" 2>/dev/null || true
  done
  echo "==> waiting for :8000 to free (bounded 60s)"
  for i in $(seq 1 30); do
    ! sudo ss -ltn 2>/dev/null | grep -q ":8000 " && { echo "   :8000 free"; return 0; }
    sleep 2
    [ $((i % 6)) -eq 0 ] && echo "   still waiting ($((i*2))s)..."
  done
  sudo ss -ltn 2>/dev/null | grep -q ":8000 " && { echo ":8000 still busy" >&2; exit 1; }
}

case "${1:-}" in
  --health)
    if curl -fsS http://127.0.0.1:8000/v1/models >/dev/null 2>&1; then
      echo "serving: $(curl -s http://127.0.0.1:8000/v1/models | tr -d '\n')"
      echo "current_model: $(read_state)"
    else
      hdr=$(lane_key "$(read_state)")
      if [ -n "$hdr" ] && curl -fsS -H "$hdr" http://127.0.0.1:8000/v1/models >/dev/null 2>&1; then
        echo "serving (auth): $(curl -s -H "$hdr" http://127.0.0.1:8000/v1/models | tr -d '\n')"
        echo "current_model: $(read_state)"
      else
        echo "NOT serving on :8000"
      fi
    fi
    exit 0
    ;;
  --swap)
    TARGET="${2:?usage: --swap <deepseek|glm|qwen38>}"
    stop_stack
    write_state "$TARGET"
    # launch_lane dispatches to the right launcher, worker-first where required.
    launch_lane "$TARGET"
    # Up to 20 min readiness wait. DeepSeek/b12x weight load + CUDA-graph
    # capture can exceed 10 min, especially under concurrent disk IO (e.g. a
    # model download on the same NVMe). 600s was too tight.
    for i in $(seq 1 120); do
      sleep 10
      c=$(curl -s -o /dev/null -w '%{http_code}' http://127.0.0.1:8000/health 2>/dev/null) || c=000
      [ "$c" = "200" ] && { log "READY after ~$((i*10))s ($TARGET)"; write_motd "$TARGET"; exit 0; }
      [ $((i % 6)) -eq 0 ] && log "  still loading ($((i*10))s)..."
    done
    log "TIMEOUT. Check: podman logs $CONTAINER"
    exit 1
    ;;
  --stop)
    # Cleanly stop the currently-serving lane on BOTH nodes and free :8000.
    # Leaves current_model (the default) intact so boot will bring it back up.
    log "stopping all lane containers..."
    stop_stack
    write_motd "stopped"
    log "model stopped. :8000 free. (boot will re-bring default: $(read_state))"
    exit 0
    ;;
  restart|--restart)
    # Clean stop ALL lanes, then relaunch the last-used model (default = last).
    # Equivalent to --swap <current> but self-targeting the boot default.
    CUR=$(read_state)
    log "restarting default model '$CUR' (clean stop + relaunch)"
    stop_stack
    launch_lane "$CUR"
    # Up to 20 min readiness wait (same rationale as --swap).
    for i in $(seq 1 120); do
      sleep 10
      c=$(curl -s -o /dev/null -w '%{http_code}' http://127.0.0.1:8000/health 2>/dev/null) || c=000
      [ "$c" = "200" ] && { log "READY after ~$((i*10))s ($CUR)"; write_motd "$CUR"; exit 0; }
      [ $((i % 6)) -eq 0 ] && log "  still loading ($((i*10))s)..."
    done
    log "TIMEOUT. Check: podman logs $CONTAINER"
    exit 1
    ;;
  "")
    # Boot / ensure path: read last-used model, ensure it's serving.
    CUR=$(read_state)
    if model_serving "$CUR"; then
      log "model '$CUR' already serving. Updating motd only."
    else
      log "ensuring model '$CUR' (from $STATE)"
      stop_stack
      launch_lane "$CUR"
    fi
    write_motd "$CUR"
    ;;
  *) echo "unknown arg '${1:-}'" >&2; exit 2 ;;
esac
