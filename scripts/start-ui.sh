#!/usr/bin/env bash
# Start the chat UI in the background on http://127.0.0.1:5173 (PORT=... to change).
set -euo pipefail
source "$(dirname "$0")/_lib.sh"
PORT="${PORT:-5173}"
PIDFILE="$ROOT/logs/ui.pid"
mkdir -p "$ROOT/logs"
if ss -ltn 2>/dev/null | grep -q ":$PORT "; then
  echo "something already listens on $PORT — the UI is probably running: http://127.0.0.1:$PORT/"
  exit 0
fi
nohup python3 "$ROOT/ui/serve.py" "$PORT" > "$ROOT/logs/ui.log" 2>&1 < /dev/null &
echo $! > "$PIDFILE"
for _ in $(seq 1 20); do
  ss -ltn 2>/dev/null | grep -q ":$PORT " && { echo "chat UI on http://127.0.0.1:$PORT/  (PID $(cat "$PIDFILE"), stop with scripts/stop-ui.sh)"; exit 0; }
  sleep 0.25
done
echo "the UI did not come up; see $ROOT/logs/ui.log" >&2
exit 1
