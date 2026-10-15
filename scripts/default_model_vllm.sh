#!/usr/bin/env bash
# default_model_vllm.sh — now a THIN WRAPPER over swap-model.sh.
#
# This was the old DeepSeek-only recipe launcher (hardcoded DEFAULT_MODEL=deepseek,
# no qwen38 lane, stale "Qwen is untested" motd). It became a split-brain trap:
# it only knew deepseek/glm/b12x, would clobber /etc/motd with stale text, and
# would only ever bring up deepseek if run. The real orchestrator is now
# swap-model.sh (unified deepseek/glm/qwen38 lanes, last-used-model-on-boot).
#
# Keep this name so any legacy reference (systemd unit, cron, doc) still works,
# but delegate 100% to swap-model.sh so there is exactly ONE source of truth for
# which model is up and how it is launched. Never add model-specific logic here.
set -euo pipefail

SWAP=/usr/local/bin/swap-model.sh
[ -x "$SWAP" ] || { echo "error: $SWAP not found" >&2; exit 1; }

# Same call surface as the old script: no-arg (ensure default), --health, --swap <m>.
# swap-model.sh's boot/ensure path uses current_model (last-used) so the box always
# comes back up on the model the user last selected.
exec "$SWAP" "$@"
