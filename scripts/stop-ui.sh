#!/usr/bin/env bash
# Stop the chat UI started by scripts/start-ui.sh (or any ui/serve.py on the port).
set -euo pipefail
source "$(dirname "$0")/_lib.sh"
PORT="${PORT:-5173}"
PIDFILE="$ROOT/logs/ui.pid"
pid=""
[ -f "$PIDFILE" ] && kill -0 "$(cat "$PIDFILE")" 2>/dev/null && pid=$(cat "$PIDFILE")
if [ -z "$pid" ]; then
  # Started by hand: find the ui/serve.py process holding the port.
  pid=$(ss -ltnp 2>/dev/null | grep ":$PORT " | grep -oE 'pid=[0-9]+' | head -1 | cut -d= -f2)
  if [ -n "$pid" ] && ! tr '\0' ' ' < "/proc/$pid/cmdline" 2>/dev/null | grep -q "ui/serve.py"; then
    echo "port $PORT is held by something other than the chat UI (PID $pid); not touching it" >&2
    exit 1
  fi
fi
if [ -z "$pid" ]; then
  echo "the chat UI is not running"
  rm -f "$PIDFILE"
  exit 0
fi
kill "$pid"
rm -f "$PIDFILE"
echo "chat UI stopped (PID $pid)"
