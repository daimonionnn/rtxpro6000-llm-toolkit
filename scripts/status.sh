#!/usr/bin/env bash
# Show which launcher variant is serving, where it lives, and how to stop it.
#
#   scripts/status.sh
#
# Detects the native Variant 3 through its PID file and the Docker variants through
# the "flashnext" container. Docker variants are identified by their
# flashnext.variant label; containers started before that label existed are
# identified from their image and environment instead.
set -uo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
V3_PIDFILE="$ROOT/sglang/v3-pennyroyal/server.pid"
V3_LOG="$ROOT/logs/pennyroyal-serve.log"
CONTAINER=flashnext

describe() {
  case "$1" in
    v0-nvme)           echo "NVMe baseline — Docker, local image, PLE streamed from NVMe" ;;
    v1-ram)            echo "Variant 1 — Docker, local image, PLE in RAM" ;;
    v2-official-image) echo "Variant 2 — Docker, official lmsysorg image, PLE in RAM" ;;
    v3-pennyroyal)     echo "Variant 3 — native jpezzulli fork, PLE in RAM" ;;
  esac
}
launcher() {
  case "$1" in
    v0-nvme) echo "sglang/v0-nvme/serve-nvfp4-nvme.sh" ;;
    *)       echo "sglang/$1/serve-nvfp4-ram.sh" ;;
  esac
}

report() {  # variant port log-command hicache kv-dtype started
  local variant=$1 port=$2 logcmd=$3 hicache=$4 kvdtype=$5 started=$6
  echo "RUNNING   $(describe "$variant")"
  echo "  directory  sglang/$variant/"
  echo "  launcher   $(launcher "$variant")"
  echo "  stop       scripts/stop.sh"
  echo "  started    $started"
  echo "  endpoint   http://127.0.0.1:$port/v1"
  local models
  models=$(curl -s --max-time 5 "http://127.0.0.1:$port/v1/models" 2>/dev/null)
  if [ -n "$models" ]; then
    echo "$models" | python3 -c "
import sys, json
for m in json.load(sys.stdin)['data']:
    print('  model      {}, context window {:,}'.format(m['id'], m.get('max_model_len') or 0))"
  else
    echo "  model      (not answering yet — still starting?)"
  fi
  local kv
  kv=$(eval "$logcmd" 2>/dev/null | grep -ao "max_total_num_tokens=[0-9]*" | tail -1 | cut -d= -f2)
  [ -n "$kv" ] && printf "  KV cache   %'d tokens, %s\n" "$kv" "$kvdtype"
  echo "  HiCache    $hicache"
  local vram
  vram=$(nvidia-smi --query-gpu=memory.used,memory.total --format=csv,noheader 2>/dev/null | head -1)
  [ -n "$vram" ] && echo "  VRAM       $vram"
  echo
}

found=0

# ── Variant 3: native process ───────────────────────────────────────────────
if [ -f "$V3_PIDFILE" ] && kill -0 "$(cat "$V3_PIDFILE")" 2>/dev/null; then
  pid=$(cat "$V3_PIDFILE")
  cmd=$(tr '\0' ' ' < "/proc/$pid/cmdline" 2>/dev/null)
  port=$(grep -oE -- "--port [0-9]+" <<<"$cmd" | awk '{print $2}')
  kvd=$(grep -oE -- "--kv-cache-dtype [a-z0-9_]+" <<<"$cmd" | awk '{print $2}')
  hc="off"; grep -q -- "--enable-hierarchical-cache" <<<"$cmd" && hc="on (NIXL persistence)"
  report v3-pennyroyal "${port:-8090}" "cat '$V3_LOG'" "$hc" "${kvd:-?}" "$(ps -o lstart= -p "$pid") (PID $pid)"
  found=1
fi

# ── Docker variants ─────────────────────────────────────────────────────────
if docker ps --format '{{.Names}}' 2>/dev/null | grep -qx "$CONTAINER"; then
  variant=$(docker inspect "$CONTAINER" --format '{{index .Config.Labels "flashnext.variant"}}' 2>/dev/null)
  if [ -z "$variant" ]; then
    image=$(docker inspect "$CONTAINER" --format '{{.Config.Image}}')
    if [[ "$image" == lmsysorg/* ]]; then variant=v2-official-image
    elif docker inspect "$CONTAINER" --format '{{range .Config.Env}}{{println .}}{{end}}' | grep -q "^SGLANG_QWEN4_PLE_NVME_PATH="; then variant=v0-nvme
    else variant=v1-ram; fi
  fi
  port=$(docker port "$CONTAINER" 8000/tcp 2>/dev/null | head -1 | awk -F: '{print $NF}')
  kvd=$(docker inspect "$CONTAINER" --format '{{index .Config.Labels "flashnext.quant.kv_cache"}}' 2>/dev/null)
  started=$(docker inspect "$CONTAINER" --format '{{.State.StartedAt}}' | cut -c1-19 | tr T ' ')
  report "$variant" "${port:-8090}" "docker logs $CONTAINER" "off" "${kvd:-?}" "$started UTC (container $CONTAINER)"
  found=1
elif docker ps -a --format '{{.Names}}' 2>/dev/null | grep -qx "$CONTAINER"; then
  echo "STOPPED   container $CONTAINER exists but is not running"
  echo
fi

if [ "$found" = 0 ]; then
  echo "NOT RUNNING   no variant is serving"
  holder=$(ss -ltnp 2>/dev/null | grep ":8090 " | grep -oE 'users:\(\("[^"]+",pid=[0-9]+' | sed 's/users:(("//; s/",pid=/ PID /')
  [ -n "$holder" ] && echo "  port 8090 is held by: $holder"
  gpu=$(nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv,noheader 2>/dev/null)
  [ -n "$gpu" ] && { echo "  on the GPU:"; echo "$gpu" | sed 's/^/    /'; }
  echo
fi

[ "$found" = 1 ] && [ -f "$V3_PIDFILE" ] && kill -0 "$(cat "$V3_PIDFILE")" 2>/dev/null \
  && docker ps --format '{{.Names}}' 2>/dev/null | grep -qx "$CONTAINER" \
  && echo "WARNING   Variant 3 and a Docker variant are both running; only one can use the GPU and port." && echo

cat <<'EOF'
Profiles (one at a time; stop any of them with scripts/stop.sh):
  scripts/start-v0-nvme.sh               NVMe baseline   sglang/v0-nvme/
  scripts/start-v1-ram.sh                Variant 1       sglang/v1-ram/
  scripts/start-v2-official-image.sh     Variant 2       sglang/v2-official-image/
  scripts/start-v3-pennyroyal.sh         Variant 3       sglang/v3-pennyroyal/
  scripts/start-v3-pennyroyal-hicache.sh Variant 3 + HiCache
  scripts/start-ui.sh / stop-ui.sh       chat UI on http://127.0.0.1:5173/
EOF
