#!/usr/bin/env bash
# Start DeepSeek-V4.1-Flash-1017 natively on one RTX PRO 6000, in the background.
#
#   ./serve.sh ud-q3kxl     Unsloth UD-Q3_K_XL on llama.cpp PR #30817 (default profile)
#   ./serve.sh mxfp4        the original MXFP4 experts (mrfakename "Q8_0") on ik_llama.cpp
#   ./stop.sh               stop the server and unlock its experts
#
# The experts of the first NCMOE layers are read from RAM. A helper locks exactly
# those file ranges in the page cache (mlock_experts.py), because the Engram table
# is read from the same files and would otherwise push the experts out of the
# cache, costing 250-700 MB/s of re-reads. Both servers are pinned to the P-cores:
# left to the scheduler, the expert threads land on E-cores.
#
# Environment (all optional):
#   MODEL_DIR    checkpoint directory (default: ../../models/<checkpoint>)
#   HOST PORT    bind address and port (default 127.0.0.1 8090)
#   CTX          context tokens (default 131072)
#   NCMOE        expert layers kept in RAM (default 29 ud-q3kxl, 30 mxfp4)
#   CPUSET       CPUs for the server (default 0-7, the P-cores of a 270K Plus)
#   LOCK_EXPERTS 1 = lock the RAM experts (default), 0 = leave them to the page cache
#   LLAMACPP_DIR IK_DIR   engine checkouts (default ~/development/llama.cpp-ds41, ik_llama-ds41)
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
ROOT="$(cd "$HERE/../.." && pwd)"
VARIANT="${1:?usage: serve.sh ud-q3kxl|mxfp4}"
PIDFILE="$HERE/server.pid"
LOCKPID="$HERE/lock.pid"
STATE="$HERE/server.profile"
HOST="${HOST:-127.0.0.1}"
PORT="${PORT:-8090}"
CTX="${CTX:-131072}"
CPUSET="${CPUSET:-0-7}"
LOCK_EXPERTS="${LOCK_EXPERTS:-1}"
mkdir -p "$ROOT/logs"

case "$VARIANT" in
  ud-q3kxl)
    PROFILE=deepseek-v4.1-flash-llamacpp-ud-q3kxl-$((CTX / 1024))k
    MODEL_DIR="${MODEL_DIR:-$ROOT/models/DeepSeek-V4.1-Flash-1017-GGUF}"
    MODEL="$MODEL_DIR/UD-Q3_K_XL/DeepSeek-V4.1-Flash-1017-UD-Q3_K_XL-00001-of-00008.gguf"
    EXE="${LLAMACPP_DIR:-$HOME/development/llama.cpp-ds41}/build/bin/llama-server"
    NCMOE="${NCMOE:-29}"
    # --fit cannot split this architecture (it tries to put every expert on the GPU);
    # the Engram table (per_layer_token_embd) is read lazily from the file.
    ARGS=(--fit off --load-mode mmap --lazy-mode on)
    ENVS=()
    ;;
  mxfp4)
    PROFILE=deepseek-v4.1-flash-ik-mxfp4-$((CTX / 1024))k
    MODEL_DIR="${MODEL_DIR:-$ROOT/models/DeepSeek-V4.1-Flash-1017-ik-GGUF}"
    MODEL="$MODEL_DIR/Q8_0/DeepSeek-V4.1-Flash-1017-Q8_0-00001-of-00010.gguf"
    EXE="${IK_DIR:-$HOME/development/ik_llama-ds41}/build/bin/llama-server"
    NCMOE="${NCMOE:-30}"
    ARGS=(-ctk f16 -ctv f16 -thp --reasoning-format deepseek)
    # ik_llama otherwise copies the RAM experts into 247 GiB of pinned memory (OOM)
    ENVS=(GGML_CUDA_NO_PINNED=1)
    ;;
  *) echo "unknown variant $VARIANT (ud-q3kxl or mxfp4)" >&2; exit 2 ;;
esac

[ -x "$EXE" ] || { echo "missing $EXE; build it as in README.md" >&2; exit 2; }
[ -f "$MODEL" ] || { echo "missing $MODEL; download it as in README.md" >&2; exit 2; }
if [ -f "$PIDFILE" ] && kill -0 "$(cat "$PIDFILE")" 2>/dev/null; then
  echo "already running (PID $(cat "$PIDFILE")); ./stop.sh first" >&2; exit 2
fi
# ik_llama does not set SO_REUSEADDR: wait for sockets left in TIME_WAIT
for _ in $(seq 60); do ss -tan | grep -q ":$PORT " || break; sleep 2; done
if ss -ltn | grep -q ":$PORT "; then echo "port $PORT is in use; stop the other server first" >&2; exit 2; fi

LOG="$ROOT/logs/$PROFILE-server.log"
echo "$PROFILE: $MODEL"
echo "  experts of layers 0..$((NCMOE - 1)) in RAM, context $CTX, CPUs $CPUSET, bind $HOST:$PORT"
# Run the server in its own systemd scope: systemd-oomd and memory accounting then treat it
# separately, so a kill during the heavy load phase cannot take the terminal (or VS Code) with it.
# With --scope, systemd-run execs the command, so $! is still the server PID.
SCOPE=()
if systemd-run --user --scope --quiet --collect true 2>/dev/null; then
  SCOPE=(systemd-run --user --scope --quiet --collect --unit="$PROFILE-$$")
fi
# setsid: own process group, so stop.sh reaches every thread of the server
env "${ENVS[@]}" setsid nohup "${SCOPE[@]}" taskset -c "$CPUSET" "$EXE" -m "$MODEL" --alias "$PROFILE" \
    --host "$HOST" --port "$PORT" -c "$CTX" -ngl 99 --n-cpu-moe "$NCMOE" -fa on \
    -b 2048 -ub 2048 -t 8 -tb 8 --jinja "${ARGS[@]}" > "$LOG" 2>&1 < /dev/null &
echo $! > "$PIDFILE"
echo "$PROFILE" > "$STATE"
echo "started PID $(cat "$PIDFILE"), log: $LOG; loading takes 2-5 minutes"

for _ in $(seq 1 180); do
  curl -sf -m 3 "http://127.0.0.1:$PORT/health" > /dev/null && break
  if ! kill -0 "$(cat "$PIDFILE")" 2>/dev/null; then
    echo "server exited:"; tr -d '\r' < "$LOG" | grep -av "huge page" | tail -20; rm -f "$PIDFILE" "$STATE"; exit 1
  fi
  sleep 5
done
curl -sf -m 3 "http://127.0.0.1:$PORT/health" > /dev/null || { echo "not healthy after 15 minutes; see $LOG" >&2; exit 1; }
echo "ready: http://$HOST:$PORT/v1 (model $PROFILE)"

if [ "$LOCK_EXPERTS" = 1 ]; then
  setsid nohup "${SCOPE[@]/%$PROFILE-$$/$PROFILE-mlock-$$}" python3 -I "$HERE/mlock_experts.py" "$MODEL" "$NCMOE" > "$ROOT/logs/$PROFILE-mlock.log" 2>&1 < /dev/null &
  echo $! > "$LOCKPID"
  for _ in $(seq 1 120); do
    grep -q -E "holding|failed" "$ROOT/logs/$PROFILE-mlock.log" 2>/dev/null && break
    kill -0 "$(cat "$LOCKPID")" 2>/dev/null || break
    sleep 5
  done
  tail -1 "$ROOT/logs/$PROFILE-mlock.log"
fi
