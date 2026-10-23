#!/usr/bin/env bash
# Start DeepSeek-V4.1-Flash-1017 in an Xolotl XQ quant (https://github.com/Indras-Mirror/Xolotl) on
# one RTX PRO 6000, in the background, behind Xolotl's OpenAI/Anthropic API server.
#
#   ./serve.sh xq-v2      IndrasMirror's V41-XQ-1017 v2 (152 GB; refusal direction removed)
#   ../native/stop.sh     stops it (shared with the llama.cpp profiles)
#
# The XQ file holds the dense weights (Q8_0) and per-expert PTQ1_0 / Q2_0 / Q3_K experts. The engine
# keeps hot experts in VRAM, a pinned RAM arena for the next tier and reads the rest from the file.
# The Engram n-gram tables (~203 GB) are read straight from shards 47/48 of the original checkpoint:
# the engine follows its model.safetensors.index.json, so no extraction is needed on ext4.
#
# Environment (all optional):
#   MODEL_DIR   XQ repository: v2/, tokenizer/, the MTP sidecar (default ../../models/V41-XQ-1017)
#   ENGRAM_DIR  original checkpoint with shards 47/48 + index (default ../../models/DeepSeek-V4.1-Flash-1017-orig)
#   HOST PORT   bind address and port (default 127.0.0.1 8090)
#   CTX         context tokens (default 131072)
#   ARENA_GIB   pinned RAM expert arena (default 72)
#   ROUTE_BIAS  decode route bias, file-tier bias (default "0 0" = exact routing; "0.5 0.3" = Xolotl's
#               fast routing: about +40 % decode, but clearly worse Slovak)
#   DSPARK      1 = speculative decode with the MTP sidecar (default), 0 = off
#   CPUSET      CPUs for the server (default 0-7, the P-cores of a 270K Plus)
#   XOLOTL_DIR  Xolotl checkout (default ~/development/xolotl)
#   PYTHON      Python with Xolotl's requirements.txt (default $XOLOTL_DIR/.venv/bin/python)
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
ROOT="$(cd "$HERE/../.." && pwd)"
VARIANT="${1:?usage: serve.sh xq-v2}"
STATE_DIR="$ROOT/deepseek-v4.1-flash/native"   # one pid file for every DeepSeek profile (stop.sh, status.sh)
PIDFILE="$STATE_DIR/server.pid"
STATE="$STATE_DIR/server.profile"
HOST="${HOST:-127.0.0.1}"
PORT="${PORT:-8090}"
CTX="${CTX:-131072}"
ARENA_GIB="${ARENA_GIB:-72}"
read -r RB RBF <<<"${ROUTE_BIAS:-0 0}"
DSPARK="${DSPARK:-1}"
CPUSET="${CPUSET:-0-7}"
XOLOTL_DIR="${XOLOTL_DIR:-$HOME/development/xolotl}"
SIDE_DIR="${MODEL_DIR:-$ROOT/models/V41-XQ-1017}"
ENGRAM_DIR="${ENGRAM_DIR:-$ROOT/models/DeepSeek-V4.1-Flash-1017-orig}"
mkdir -p "$ROOT/logs"

case "$VARIANT" in
  xq-v2)
    PROFILE=deepseek-v4.1-flash-xolotl-xq-v2
    XQ_DIR="$SIDE_DIR/v2"
    ;;
  *) echo "unknown variant $VARIANT (xq-v2)" >&2; exit 2 ;;
esac
MODEL="$XQ_DIR/V41-XQ-1017.gguf"
RANK="$XQ_DIR/rank-1017-085.bin"
SIDECAR="$SIDE_DIR/V41-MTP-Vision-1017.Q2_K.gguf"
TOKENIZER="$SIDE_DIR/tokenizer"

[ -x "$XOLOTL_DIR/build-xolotl/ds41_generate" ] || { echo "missing $XOLOTL_DIR/build-xolotl/ds41_generate; build Xolotl as in README.md" >&2; exit 2; }
for f in "$MODEL" "$RANK" "$TOKENIZER/vocab.json" "$ENGRAM_DIR/model.safetensors.index.json" \
         "$ENGRAM_DIR/model-00047-of-00048.safetensors" "$ENGRAM_DIR/model-00048-of-00048.safetensors"; do
  [ -e "$f" ] || { echo "missing $f; download it as in README.md" >&2; exit 2; }
done
if [ -f "$PIDFILE" ] && kill -0 "$(cat "$PIDFILE")" 2>/dev/null; then
  echo "a DeepSeek server is already running (PID $(cat "$PIDFILE")); ../native/stop.sh first" >&2; exit 2
fi
if ss -ltn | grep -q ":$PORT "; then echo "port $PORT is in use; stop the other server first" >&2; exit 2; fi

LOG="$ROOT/logs/$PROFILE-server.log"
CONFIG="$ROOT/logs/$PROFILE-config.json"
ENGINE_LOG="$ROOT/logs/$PROFILE-engine.log"
PROFILE="$PROFILE" MODEL="$MODEL" RANK="$RANK" ENGRAM_DIR="$ENGRAM_DIR" CTX="$CTX" ARENA_GIB="$ARENA_GIB" \
RB="$RB" RBF="$RBF" PORT="$PORT" TOKENIZER="$TOKENIZER" ENGINE_LOG="$ENGINE_LOG" XOLOTL_DIR="$XOLOTL_DIR" \
ENGINE="$HERE/engine.sh" SIDECAR="$([ "$DSPARK" = 1 ] && echo "$SIDECAR")" python3 -I - > "$CONFIG" <<'EOF'
import json, os
e = os.environ
print(json.dumps({
    "exe": e["ENGINE"],
    "args": ["-m", e["MODEL"], "--backend", "cuda", "--experts", "gpu", "--engram", e["ENGRAM_DIR"],
             "--rank", e["RANK"], "--ctx", e["CTX"], "--stop", "1", "--arena-adapt", "--kv-fp4",
             "--route-bias", e["RB"], "--route-bias-file", e["RBF"], "--arena-gib", e["ARENA_GIB"],
             "--prompt-batch", "12288"],
    "cwd": e["XOLOTL_DIR"],
    "tokenizer": e["TOKENIZER"],
    "model_name": e["PROFILE"],
    "log": e["ENGINE_LOG"],
    "port": int(e["PORT"]),
    "gpu": 0,
    "env": {"STRATA_SERVE_DSPARK": e["SIDECAR"]},
    "allowed_hosts": "*",
    # the toolkit's chat UI (scripts/start-ui.sh) calls /v1 from this origin in the browser
    "cors_origins": ["http://127.0.0.1:5173", "http://localhost:5173"],
    "sampling": {"temperature": 1, "top_p": 1.0},
    "api_monitor": True,
    # agents send max_tokens 65536 whatever the prompt: shorten it to the room left instead of a 400
    "fit_max_tokens": True,
    "reasoning_budget_tokens": 32768,
}, indent=2))
EOF

echo "$PROFILE: $MODEL"
echo "  Engram $ENGRAM_DIR, context $CTX, arena $ARENA_GIB GiB, route bias $RB/$RBF, DSpark $DSPARK, CPUs $CPUSET, bind $HOST:$PORT"
# setsid: own process group, so stop.sh reaches the server and its engine
setsid nohup taskset -c "$CPUSET" "${PYTHON:-$XOLOTL_DIR/.venv/bin/python}" "$XOLOTL_DIR/tools/v41/serve/serve_ds41.py" --engine strata \
    --config "$CONFIG" --host "$HOST" --port "$PORT" > "$LOG" 2>&1 < /dev/null &
echo $! > "$PIDFILE"
echo "$PROFILE" > "$STATE"
echo "started PID $(cat "$PIDFILE"), log: $LOG (engine: $ENGINE_LOG); loading takes a few minutes"

for _ in $(seq 1 240); do
  curl -sf -m 3 "http://127.0.0.1:$PORT/health" 2>/dev/null | grep -q '"ok"\|ready' && break
  if ! kill -0 "$(cat "$PIDFILE")" 2>/dev/null; then
    echo "server exited:"; tail -20 "$LOG"; rm -f "$PIDFILE" "$STATE"; exit 1
  fi
  sleep 5
done
curl -s -m 3 "http://127.0.0.1:$PORT/health" || true; echo
echo "endpoint: http://$HOST:$PORT/v1 (model $PROFILE)"
