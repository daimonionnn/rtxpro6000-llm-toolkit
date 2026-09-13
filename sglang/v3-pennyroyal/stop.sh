#!/usr/bin/env bash
# Stop the natively running Pennyroyal server and wait until its VRAM is released.
#
#   ./stop.sh
#
# Unlike the Docker variants there is no restart policy: once stopped it stays
# stopped, and it does not come back after a reboot.
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
PIDFILE="$HERE/server.pid"
TIMEOUT="${TIMEOUT:-60}"

vram() { nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits 2>/dev/null | head -1; }

if [ ! -f "$PIDFILE" ] || ! kill -0 "$(cat "$PIDFILE")" 2>/dev/null; then
  echo "no running Pennyroyal server"
  rm -f "$PIDFILE"
  exit 0
fi

pid=$(cat "$PIDFILE")
echo "stopping PID $pid (VRAM in use: $(vram) MiB)"
# The launcher started it with setsid, so its PID is also the process-group ID;
# signalling the group reaches the scheduler and detokenizer children too.
kill -TERM -- "-$pid" 2>/dev/null || kill -TERM "$pid"

for _ in $(seq 1 "$TIMEOUT"); do
  kill -0 "$pid" 2>/dev/null || break
  sleep 1
done
if kill -0 "$pid" 2>/dev/null; then
  echo "still alive after ${TIMEOUT}s, sending SIGKILL"
  kill -KILL -- "-$pid" 2>/dev/null || kill -KILL "$pid"
fi

for _ in $(seq 1 30); do
  nvidia-smi --query-compute-apps=process_name --format=csv,noheader 2>/dev/null | grep -q . || break
  sleep 1
done
rm -f "$PIDFILE"
echo "stopped   (VRAM in use: $(vram) MiB)"

left=$(nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv,noheader 2>/dev/null)
[ -n "$left" ] && { echo "still on the GPU:"; echo "$left" | sed 's/^/  /'; }
exit 0
