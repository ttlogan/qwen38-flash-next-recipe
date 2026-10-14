#!/usr/bin/env bash
# swap-model.sh — bring up the last-used model on boot, or --swap to another.
# Reuses the vetted 2-node orchestration (worker-first/head-first, idempotent
# ensure, health poll, bounded port-free wait, RoCE pinning) and just points to
# the right container + launcher per lane.
#
# Call: swap-model.sh            # ensure current_model is serving (idempotent) + motd
#       swap-model.sh --health   # report serving status / health
#       swap-model.sh --swap <deepseek|glm|qwen38>  # stop current, launch <m>, update motd
set -euo pipefail

STATE=/etc/vllm/current_model
# Management/LAN reachability for the swap ssh. These are only used to reach the
# peer for the podman rm (the container launch itself uses the RoCE CX7 fabric).
LEO="<HEAD_IP>"
RAPH="<WORKER_IP>"

# ---- lane launcher dispatch (handles worker-first where required) ----
launch_lane() {
  local m="$1"
  case "$m" in
    deepseek) sudo -u wfoster "$LAUNCHER" ;;
    glm)      sudo -u wfoster bash /home/wfoster/glm53-vllm.sh 0 ;;
    qwen38)
      # Qwen is served by SGLang (tp2, 2-node). WORKER (rank 1, raph) must come
      # up FIRST, then head (rank 0). HEAD-first hits the TCPStore rendezvous race.
      sudo -u wfoster ssh -o BatchMode=yes -o ConnectTimeout=10 "$RAPH" \
        "bash /home/wfoster/sglang-qwen38.sh 1" 2>/dev/null || true
      sudo -u wfoster bash /home/wfoster/sglang-qwen38.sh 0
      ;;
  esac
}
# Each lane is its own bake. Swap logic lives here; lane specifics live in the
# lane's launcher/script. The DeepSeek lane uses the b12x recipe stack (run-recipe.py).
case "${2:-${CURRENT_MODEL:-deepseek}}" in
  deepseek)
    CONTAINER=b12x-vllm-node
    LAUNCHER="sudo -u wfoster /opt/vllm-recipe/.build/spark-vllm-docker/run-recipe.py /opt/vllm-recipe/recipes/orcarouter-eugr-1m.yaml -d"
    MODEL_DIR=/models/orcarouter-deepseek-v4-flash-vision-uncensored
    ;;
  glm)
    CONTAINER=vllm_glm53
    LAUNCHER="sudo -u wfoster bash /home/wfoster/glm53-vllm.sh 0"
    MODEL_DIR=/models/glm-5.3-flash-nvfp4
    ;;
  qwen38)
    CONTAINER=qwen38fn-sglang
    LAUNCHER="sudo -u wfoster bash /home/wfoster/sglang-qwen38.sh 0"
    MODEL_DIR=/models/orca-qwen3.8-modelopt-nvfp4
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

EOF
}

# Map a lane key (deepseek/glm/qwen38) to the served-model-name it exposes on
# /v1/models, so model_serving() can detect "already up" idempotently. Without
# this, the boot/ensure path greps for e.g. "qwen38" but the server exposes
# "qwen3.8-flash-next-uncensored" -> false negative -> relaunch every boot.
served_name() {
  case "$1" in
    qwen38)           echo "qwen3.8-flash-next-uncensored" ;;
    glm)              echo "glm" ;;
    deepseek)         echo "deepseek" ;;
    *)                echo "$1" ;;
  esac
}
model_serving() {
  local want; want=$(served_name "$1")
  curl -fsS http://127.0.0.1:8000/v1/models 2>/dev/null | grep -qi "$want"
}

stop_stack() {
  # Stop ALL known lane containers (not just CONTAINER), because the currently
  # serving lane may differ from the target on a swap. Leaving an orphaned
  # container running on :8000 is a regression (swap X->Y->X double-binds).
  echo "==> stopping all lane containers on head + worker (deepseek/glm/qwen)"
  for c in b12x-vllm-node vllm_glm53 vllm_qwen38 qwen38fn-sglang; do
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
      echo "NOT serving on :8000"
    fi
    exit 0
    ;;
  --swap)
    TARGET="${2:?usage: --swap <deepseek|glm|qwen38>}"
    stop_stack
    write_state "$TARGET"
    # launch_lane dispatches to the right launcher, worker-first where required.
    launch_lane "$TARGET"
    for i in $(seq 1 60); do
      sleep 10
      c=$(curl -s -o /dev/null -w '%{http_code}' http://127.0.0.1:8000/health 2>/dev/null) || c=000
      [ "$c" = "200" ] && { log "READY after ~$((i*10))s ($TARGET)"; write_motd "$TARGET"; exit 0; }
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
