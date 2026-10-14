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

# ---- lane launcher dispatch (handles worker-first where required) ----
launch_lane() {
  local m="$1"
  case "$m" in
    deepseek) sudo -u wfoster "$LAUNCHER" ;;
    glm)      sudo -u wfoster bash /home/wfoster/glm53-vllm.sh 0 ;;
    qwen38)
      # Qwen is served by SGLang (tp2, 2-node). HEAD (rank 0, leo) must come up
      # FIRST as the TCPStore master, then the worker (rank 1, raph) joins after
      # a short delay. RANK-1-first deadlocks the worker at `Init torch
      # distributed begin` (it blocks on rank 0's dist_init port not yet bound).
      # With --weight-loader-disable-mmap each rank frees its shards as it reads
      # them, so head+worker loading together does NOT deadlock the unified pool.
      sudo -u wfoster bash /home/wfoster/sglang-qwen38.sh 0
      echo "   head (rank 0) launched; waiting for TCPStore :29522 to bind, then worker"
      # Wait for the head's TCPStore (dist-init-addr) to actually bind before starting
      # the worker, instead of a fixed sleep. A fixed 25s races a cold-boot where JIT/
      # cache warmup delays the bind -> worker starts too early and the rendezvous fails.
      for _ in $(seq 1 60); do
        sudo ss -ltn 2>/dev/null | grep -q ":29522 " && break
        sleep 1
      done
      sudo -u wfoster ssh -o BatchMode=yes -o ConnectTimeout=10 "$RAPH" \
        "bash /home/wfoster/sglang-qwen38.sh 1" 2>/dev/null || true
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
    glm)              echo "dgx_hobo_default" ;;
    deepseek)         echo "deepseek-v4-flash-vision-exp" ;;
    *)                echo "$1" ;;
  esac
}
# Auth header for a lane's /v1/models probe. SGLang (qwen) is bearer-gated; the
# vLLM lanes (deepseek/glm) answer open. Read the lane's key file if present,
# else send no header. The deployed Qwen launcher reads the same
# $REPO_ROOT/.sglang-api-key.
lane_key() {
  local f
  case "$1" in
    qwen38) f="${Qwen_KEY_FILE:-/home/wfoster/qwen38-flash-next-recipe/.sglang-api-key}" ;;
    *)      return 0 ;;
  esac
  [ -r "$f" ] && { echo "Authorization: Bearer $(tr -d '\n' < "$f")"; }
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
  for c in b12x-vllm-node vllm_glm53 qwen38fn-sglang; do
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
    for i in $(seq 1 60); do
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
    for i in $(seq 1 60); do
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
