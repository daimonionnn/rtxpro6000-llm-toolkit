#!/usr/bin/env bash
# Stop the Qwen3.8-Flash-Next server and wait until its VRAM is released.
#
#   ./stop.sh          # stop the container, keep it (the launcher recreates it anyway)
#   ./stop.sh --rm     # stop and remove the container
#
# Shared by every Docker profile; each profile directory has a
# stop.sh that calls this. They all run as the one container "rtxpro6000-llm".
#
# The Docker launchers use --restart unless-stopped. A container stopped by hand
# stays stopped across reboots, so this is also how you keep the model from
# grabbing the GPU at the next boot.
set -euo pipefail

NAME="${NAME:-rtxpro6000-llm}"
TIMEOUT="${TIMEOUT:-60}"   # seconds to wait for a graceful shutdown before SIGKILL
REMOVE=0
[ "${1:-}" = "--rm" ] && REMOVE=1

vram() { nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits 2>/dev/null | head -1; }

if ! docker ps -a --format '{{.Names}}' | grep -qx "$NAME"; then
  echo "no container named '$NAME'"
  exit 0
fi

if docker ps --format '{{.Names}}' | grep -qx "$NAME"; then
  before=$(vram)
  echo "stopping $NAME (VRAM in use: ${before:-?} MiB)"
  docker stop -t "$TIMEOUT" "$NAME" >/dev/null

  # The scheduler process can outlive `docker stop` returning by a moment; wait
  # until no sglang process is left on the GPU rather than trusting the exit.
  for _ in $(seq 1 30); do
    nvidia-smi --query-compute-apps=process_name --format=csv,noheader 2>/dev/null \
      | grep -q sglang || break
    sleep 1
  done
  echo "stopped   (VRAM in use: $(vram) MiB)"
else
  echo "$NAME is already stopped"
fi

if [ "$REMOVE" = 1 ]; then
  docker rm "$NAME" >/dev/null
  echo "removed   $NAME"
fi

left=$(nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv,noheader 2>/dev/null)
[ -n "$left" ] && { echo "still on the GPU:"; echo "$left" | sed 's/^/  /'; }
exit 0
