#!/usr/bin/env bash
# Start Qwen3.8-Flash-Next Q8_0 on ik_llama.cpp through ik-llama-toolkit, in the background.
#
#   ./serve.sh q8-vision        plain Q8_0, vision projector on the GPU (CTX 131072 or 262144)
#   ./serve.sh q8-mtp-vision    the same weights plus the MTP draft head (CTX 131072 only)
#   ./stop.sh                   stop the server
#
# This is a thin integration: the placement, threads and flags live in the external
# ik-llama-toolkit (https://github.com/daimonionnn/ik-llama-toolkit), whose
# config/models/qwen38-flash-next-*.env profiles were measured on this machine. This
# script only maps a profile id of this toolkit to an ik-llama-toolkit profile, runs its
# serve.sh in a systemd scope and keeps the PID and profile so scripts/stop.sh and
# scripts/status.sh can find it. The Q8_0 GGUF is the same file the Strata Q8 profiles
# load, which makes this the reference engine for comparing Strata's numerics.
#
# Environment (all optional):
#   HOST PORT        bind address and port (default 127.0.0.1 8090)
#   CTX              context tokens (default 131072)
#   IK_TOOLKIT_DIR   ik-llama-toolkit checkout (default ~/development/ik-llama-toolkit)
#   IK_*             passed through to ik-llama-toolkit, e.g. IK_NCMOE=18 or IK_GDB=0
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
ROOT="$(cd "$HERE/../.." && pwd)"
VARIANT="${1:?usage: serve.sh q8-vision|q8-mtp-vision}"
PIDFILE="$HERE/server.pid"
STATE="$HERE/server.profile"
HOST="${HOST:-127.0.0.1}"
PORT="${PORT:-8090}"
CTX="${CTX:-131072}"
IK_TOOLKIT_DIR="${IK_TOOLKIT_DIR:-$HOME/development/ik-llama-toolkit}"
# The ik-llama-toolkit and ik_llama.cpp commits these profiles were checked against.
# A different checkout still starts, with a warning; update the pin and README.md
# after re-checking it.
PINNED_TOOLKIT=e7226e7
PINNED_ENGINE=13cffc00
mkdir -p "$ROOT/logs"

case "$VARIANT/$CTX" in
  q8-vision/131072)     IK_PROFILE=qwen38-flash-next-q8-128k ;;
  q8-vision/262144)     IK_PROFILE=qwen38-flash-next-q8-256k ;;
  q8-mtp-vision/131072) IK_PROFILE=qwen38-flash-next-q8-128k-mtp ;;
  *) echo "no ik-llama-toolkit profile for $VARIANT at CTX=$CTX" >&2; exit 2 ;;
esac
PROFILE=qwen3.8-flash-next-ik-$VARIANT-$((CTX / 1024))k

[ -x "$IK_TOOLKIT_DIR/serve.sh" ] || { echo "missing $IK_TOOLKIT_DIR/serve.sh; clone ik-llama-toolkit as in README.md" >&2; exit 2; }
[ -x "$IK_TOOLKIT_DIR/ik_llama.cpp/build/bin/llama-server" ] \
  || { echo "ik_llama.cpp is not built; run $IK_TOOLKIT_DIR/build.sh" >&2; exit 2; }
toolkit_head=$(git -C "$IK_TOOLKIT_DIR" rev-parse --short=7 HEAD 2>/dev/null || echo '?')
engine_head=$(git -C "$IK_TOOLKIT_DIR/ik_llama.cpp" rev-parse --short=8 HEAD 2>/dev/null || echo '?')
[ "$toolkit_head" = "$PINNED_TOOLKIT" ] || echo "warning: ik-llama-toolkit is at $toolkit_head, pinned $PINNED_TOOLKIT" >&2
[ "$engine_head" = "$PINNED_ENGINE" ] || echo "warning: ik_llama.cpp is at $engine_head, pinned $PINNED_ENGINE" >&2

if [ -f "$PIDFILE" ] && kill -0 "$(cat "$PIDFILE")" 2>/dev/null; then
  echo "already running (PID $(cat "$PIDFILE")); ./stop.sh first" >&2; exit 2
fi
# ik_llama does not set SO_REUSEADDR: wait for sockets left in TIME_WAIT
for _ in $(seq 60); do ss -tan | grep -q ":$PORT " || break; sleep 2; done
if ss -ltn | grep -q ":$PORT "; then echo "port $PORT is in use; stop the other server first" >&2; exit 2; fi

LOG="$ROOT/logs/$PROFILE-server.log"
echo "$PROFILE: ik-llama-toolkit $toolkit_head profile $IK_PROFILE, ik_llama.cpp $engine_head"
echo "  context $CTX, bind $HOST:$PORT"
# Own systemd scope, as in deepseek-v4.1-flash/native/serve.sh: systemd-oomd then cannot
# take the terminal with the server. With --scope, systemd-run execs, so $! is the PID.
SCOPE=()
if systemd-run --user --scope --quiet --collect true 2>/dev/null; then
  SCOPE=(systemd-run --user --scope --quiet --collect --unit="$PROFILE-$$")
fi
# IK_KILL_SQUATTERS=0: never let ik-llama-toolkit evict other GPU processes from here.
# setsid: own process group, so stop.sh reaches serve.sh, gdb, llama-server and tee.
IK_HOST="$HOST" IK_PORT="$PORT" IK_CTX="$CTX" IK_ALIAS="$PROFILE" IK_KILL_SQUATTERS=0 \
  setsid nohup "${SCOPE[@]}" "$IK_TOOLKIT_DIR/serve.sh" "$IK_PROFILE" > "$LOG" 2>&1 < /dev/null &
echo $! > "$PIDFILE"
echo "$PROFILE" > "$STATE"
echo "started PID $(cat "$PIDFILE"), log: $LOG; loading takes 1-3 minutes"

for _ in $(seq 1 180); do
  curl -sf -m 3 "http://127.0.0.1:$PORT/health" > /dev/null && break
  if ! kill -0 "$(cat "$PIDFILE")" 2>/dev/null; then
    echo "server exited:"; tr -d '\r' < "$LOG" | tail -20; rm -f "$PIDFILE" "$STATE"; exit 1
  fi
  sleep 5
done
curl -sf -m 3 "http://127.0.0.1:$PORT/health" > /dev/null || { echo "not healthy after 15 minutes; see $LOG" >&2; exit 1; }
echo "ready: http://$HOST:$PORT/v1 (model $PROFILE)"
