#!/usr/bin/env bash
# Stop the native DeepSeek-V4.1 server, unlock its experts and wait until the VRAM is released.
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
PIDFILE="$HERE/server.pid"
LOCKPID="$HERE/lock.pid"
vram() { nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits 2>/dev/null | head -1; }

if [ -f "$LOCKPID" ]; then
  kill "$(cat "$LOCKPID")" 2>/dev/null || true
  rm -f "$LOCKPID"
fi
if [ ! -f "$PIDFILE" ] || ! kill -0 "$(cat "$PIDFILE")" 2>/dev/null; then
  echo "no running DeepSeek-V4.1 server"
  rm -f "$PIDFILE" "$HERE/server.profile"
  exit 0
fi
pid=$(cat "$PIDFILE")
echo "stopping PID $pid (VRAM in use: $(vram) MiB)"
kill -TERM -- "-$pid" 2>/dev/null || kill -TERM "$pid"
for _ in $(seq 1 60); do kill -0 "$pid" 2>/dev/null || break; sleep 1; done
if kill -0 "$pid" 2>/dev/null; then
  echo "still alive after 60 s, sending SIGKILL"
  kill -KILL -- "-$pid" 2>/dev/null || kill -KILL "$pid"
fi
for _ in $(seq 1 30); do
  nvidia-smi --query-compute-apps=process_name --format=csv,noheader 2>/dev/null | grep -q . || break
  sleep 1
done
rm -f "$PIDFILE" "$HERE/server.profile"
echo "stopped   (VRAM in use: $(vram) MiB)"
