#!/usr/bin/env bash
# Show which profile is serving, where it lives, and how to stop it.
#
#   scripts/status.sh
#
# The native pennyroyal profile is found through its PID file, Docker profiles
# through the "rtxpro6000-llm" container and its rtxpro6000-llm.profile label.
set -uo pipefail
source "$(dirname "$0")/_lib.sh"
PENNYROYAL_LOG="$ROOT/logs/pennyroyal-serve.log"

report() {  # profile port log-command hicache kv-dtype started
  local profile=$1 port=$2 logcmd=$3 hicache=$4 kvdtype=$5 started=$6
  echo "RUNNING   $profile"
  echo "  about      $(profile_field "$profile" desc || echo '?')"
  echo "  directory  $(profile_field "$profile" dir || echo '?')/"
  echo "  model      models/$(profile_field "$profile" model || echo '?')/"
  echo "  start      scripts/start-$profile.sh"
  echo "  stop       scripts/stop.sh"
  echo "  started    $started"
  echo "  endpoint   http://127.0.0.1:$port/v1"
  local models
  models=$(curl -s --max-time 5 "http://127.0.0.1:$port/v1/models" 2>/dev/null)
  if [ -n "$models" ]; then
    # vLLM and SGLang report max_model_len; TabbyAPI reports it on /v1/model instead
    local tabby_ctx
    tabby_ctx=$(curl -s --max-time 5 "http://127.0.0.1:$port/v1/model" 2>/dev/null \
      | python3 -c "import sys, json; print(json.load(sys.stdin)['parameters']['max_seq_len'])" 2>/dev/null)
    echo "$models" | TABBY_CTX="$tabby_ctx" python3 -c "
import os, sys, json
for m in json.load(sys.stdin)['data']:
    ctx = m.get('max_model_len') or int(os.environ.get('TABBY_CTX') or 0)
    print('  served as  {}, context window {}'.format(m['id'], f'{ctx:,}' if ctx else '?'))"
  else
    echo "  served as  (not answering yet — still starting?)"
  fi
  local kv
  kv=$(eval "$logcmd" 2>/dev/null | grep -aoE "max_total_num_tokens=[0-9]+|GPU KV cache size: [0-9,]+ tokens" | tail -1 | grep -oE "[0-9][0-9,]*" | tail -1 | tr -d ,)
  [ -n "$kv" ] && printf "  KV cache   %'d tokens, %s\n" "$kv" "$kvdtype"
  [ -n "$hicache" ] && echo "  HiCache    $hicache"
  local vram
  vram=$(nvidia-smi --query-gpu=memory.used,memory.total --format=csv,noheader 2>/dev/null | head -1)
  [ -n "$vram" ] && echo "  VRAM       $vram"
  echo
}

found=0

if [ -f "$NATIVE_PIDFILE" ] && kill -0 "$(cat "$NATIVE_PIDFILE")" 2>/dev/null; then
  pid=$(cat "$NATIVE_PIDFILE")
  cmd=$(tr '\0' ' ' < "/proc/$pid/cmdline" 2>/dev/null)
  port=$(grep -oE -- "--port [0-9]+" <<<"$cmd" | awk '{print $2}')
  kvd=$(grep -oE -- "--kv-cache-dtype [a-z0-9_]+" <<<"$cmd" | awk '{print $2}')
  hc="off"; grep -q -- "--enable-hierarchical-cache" <<<"$cmd" && hc="on (NIXL persistence)"
  report qwen3.8-flash-next-sglang-nvfp4-ram-pennyroyal "${port:-8090}" "cat '$PENNYROYAL_LOG'" "$hc" "${kvd:-?}" "$(ps -o lstart= -p "$pid") (PID $pid)"
  found=1
fi

if docker ps --format '{{.Names}}' 2>/dev/null | grep -qx "$CONTAINER"; then
  profile=$(docker inspect "$CONTAINER" --format '{{index .Config.Labels "rtxpro6000-llm.profile"}}' 2>/dev/null)
  port=$(docker port "$CONTAINER" 8000/tcp 2>/dev/null | head -1 | awk -F: '{print $NF}')
  kvd=$(docker inspect "$CONTAINER" --format '{{index .Config.Labels "rtxpro6000-llm.quant.kv_cache"}}' 2>/dev/null)
  started=$(docker inspect "$CONTAINER" --format '{{.State.StartedAt}}' | cut -c1-19 | tr T ' ')
  report "${profile:-unknown-docker-profile}" "${port:-8090}" "docker logs $CONTAINER" "" "${kvd:-?}" "$started UTC (container $CONTAINER)"
  found=1
elif docker ps -a --format '{{.Names}}' 2>/dev/null | grep -qx "$CONTAINER"; then
  echo "STOPPED   container $CONTAINER exists but is not running"
  echo
fi

if [ "$found" = 0 ]; then
  echo "NOT RUNNING   no profile is serving"
  holder=$(ss -ltnp 2>/dev/null | grep ":8090 " | grep -oE 'users:\(\("[^"]+",pid=[0-9]+' | sed 's/users:(("//; s/",pid=/ PID /')
  [ -n "$holder" ] && echo "  port 8090 is held by: $holder"
  gpu=$(nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv,noheader 2>/dev/null)
  [ -n "$gpu" ] && { echo "  on the GPU:"; echo "$gpu" | sed 's/^/    /'; }
  echo
fi

[ "$(running_profiles | wc -l)" -gt 1 ] \
  && echo "WARNING   two profiles are running; only one can use the GPU and port 8090." && echo

echo "Profiles (one at a time; stop any of them with scripts/stop.sh):"
for line in "${PROFILES[@]}"; do
  IFS='|' read -r id dir launcher model desc <<<"$line"
  printf "  scripts/start-%-50s %s\n" "$id.sh" "$desc"
done
printf "  scripts/start-%-50s %s\n" "qwen3.8-flash-next-sglang-nvfp4-ram-pennyroyal-hicache.sh" "the pennyroyal profile with HiCache/NIXL persistence"
printf "  scripts/%-56s %s\n" "start-ui.sh / stop-ui.sh" "chat UI on http://127.0.0.1:5173/"
