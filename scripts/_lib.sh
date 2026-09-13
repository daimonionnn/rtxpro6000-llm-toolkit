# Shared by the scripts in this directory; source it, do not run it.
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export MODEL_DIR="${MODEL_DIR:-$ROOT/models/Qwen3.8-Flash-Next-NVFP4}"
V3_PIDFILE="$ROOT/sglang/v3-pennyroyal/server.pid"
CONTAINER=flashnext

# Print the variant(s) currently serving, one per line: v3-pennyroyal for the
# native fork, the flashnext.variant label (or "docker") for a running container.
running_variants() {
  if [ -f "$V3_PIDFILE" ] && kill -0 "$(cat "$V3_PIDFILE")" 2>/dev/null; then
    echo v3-pennyroyal
  fi
  if docker ps --format '{{.Names}}' 2>/dev/null | grep -qx "$CONTAINER"; then
    local v
    v=$(docker inspect "$CONTAINER" --format '{{index .Config.Labels "flashnext.variant"}}' 2>/dev/null)
    echo "${v:-docker}"
  fi
}

# Refuse to start a second variant: they all need the whole GPU and port 8090.
require_idle() {
  local running
  running=$(running_variants | tr '\n' ' ')
  if [ -n "$running" ]; then
    echo "already running: $running— stop it first with scripts/stop.sh" >&2
    exit 2
  fi
}
