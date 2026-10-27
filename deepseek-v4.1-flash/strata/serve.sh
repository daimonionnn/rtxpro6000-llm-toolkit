#!/usr/bin/env bash
# Start DeepSeek-V4.1-Flash-1017 on Strata's DeepSeek engine (strata-dsv41, https://github.com/Niko1221/Strata)
# on one RTX PRO 6000, in the background, behind Strata's OpenAI/Anthropic API server.
#
#   ./serve.sh ud-q8kxl          Unsloth UD-Q8_K_XL (508 GB: MXFP4 experts, BF16 dense weights, Engram)
#                                (serve-ud-q8kxl-{128k,256k,512k}.sh set CTX)
#   ./serve.sh ud-q8kxl-vision   the same with image input: the engine's own vision tower (DeepSeek's
#                                BF16 tower in the pack, ~0.3 GiB of VRAM), Strata 0.2.7+
#   ../native/stop.sh     stops it (shared with the other DeepSeek profiles)
#
# The engine keeps the dense weights and the most-routed experts in VRAM (~95 GiB) and computes the
# rest (~208 GB) on the CPU. The Engram n-gram table is read from the model files on the SSD.
#
# Environment (all optional):
#   MODEL_DIR     folder with the UD-Q8_K_XL shards (default ../../models/DeepSeek-V4.1-Flash-1017-GGUF)
#   HOST PORT     bind address and port (default 127.0.0.1 8090)
#   CTX           context tokens: 131072 (default), 262144 or 524288; the profile id gets -256k / -512k
#   MMAP_EXPERTS  on = read the CPU experts in place from the files (page cache, can be evicted),
#                 off = copy them into the engine's memory at load (~208 GB of the 244 GiB),
#                 auto = the engine decides (copies them when they fit). Default: see below.
#   DSPARK        DSpark drafter: on (default; +3 % Slovak, +9 % English, +47 % code here), auto (the
#                 engine leaves it off when the experts do not fit the GPU), none
#   CPUSET        CPUs for the server and engine (default 0-7, the P-cores of a 270K Plus)
#   CPU_THREADS   CPU-lane expert threads (default 0 = the engine's choice)
#   STRATA_DIR    Strata checkout (0.2.7+) with build-dsv41/ and packs/ (default ~/development/Strata)
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
ROOT="$(cd "$HERE/../.." && pwd)"
VARIANT="${1:?usage: serve.sh ud-q8kxl|ud-q8kxl-vision}"
STATE_DIR="$ROOT/deepseek-v4.1-flash/native"   # one pid file for every DeepSeek profile (stop.sh, status.sh)
PIDFILE="$STATE_DIR/server.pid"
STATE="$STATE_DIR/server.profile"
HOST="${HOST:-127.0.0.1}"
PORT="${PORT:-8090}"
CTX="${CTX:-131072}"
MMAP_EXPERTS="${MMAP_EXPERTS:-on}"
DSPARK="${DSPARK:-on}"
CPUSET="${CPUSET:-0-7}"
CPU_THREADS="${CPU_THREADS:-0}"
STRATA_DIR="${STRATA_DIR:-$HOME/development/Strata}"
GGUF_DIR="${MODEL_DIR:-$ROOT/models/DeepSeek-V4.1-Flash-1017-GGUF}"
mkdir -p "$ROOT/logs"

case "$VARIANT" in
  ud-q8kxl|ud-q8kxl-vision)
    PROFILE=deepseek-v4.1-flash-strata-$VARIANT
    PACK="$STRATA_DIR/packs/dsv41-ud-q8kxl"   # one pack with the vision tower; --vision picks
    SHARDS="$GGUF_DIR/UD-Q8_K_XL"
    ;;
  *) echo "unknown variant $VARIANT (ud-q8kxl, ud-q8kxl-vision)" >&2; exit 2 ;;
esac
# the 256K / 512K profiles are the same server with a larger window (serve-*-256k.sh, serve-*-512k.sh)
case "$CTX" in
  131072) ;;
  262144|524288) PROFILE="$PROFILE-$((CTX / 1024))k" ;;
  *) echo "CTX must be 131072, 262144 or 524288" >&2; exit 2 ;;
esac
VISION=off; [[ "$VARIANT" == *-vision ]] && VISION=on
EXE="$STRATA_DIR/build-dsv41/strata-dsv41"
PYTHON="$STRATA_DIR/.venv/bin/python"

for f in "$EXE" "$PYTHON" "$PACK/ds4.txt" "$PACK/tensors.txt" "$PACK/tokenizer"; do
  [ -e "$f" ] || { echo "missing $f; build Strata and pack the model as in README.md" >&2; exit 2; }
done
# the pack names the folder it reads the shards from: it must be this one
grep -qx "model_dir $SHARDS" "$PACK/ds4.txt" || { echo "$PACK was packed from another folder than $SHARDS" >&2; exit 2; }
if [ "$VISION" = on ]; then
  grep -q "^image_token_id " "$PACK/ds4.txt" || { echo "$PACK has no vision tower; pack it as in README.md" >&2; exit 2; }
  [ -f "$STRATA_DIR/serve/ds_pictures.py" ] || { echo "$STRATA_DIR is older than Strata 0.2.7 (no DeepSeek pictures)" >&2; exit 2; }
fi
for i in $(seq 1 9); do
  f="$SHARDS/DeepSeek-V4.1-Flash-1017-UD-Q8_K_XL-0000$i-of-00009.gguf"
  [ -f "$f" ] || { echo "missing $f; download it as in README.md" >&2; exit 2; }
done
if [ -f "$PIDFILE" ] && kill -0 "$(cat "$PIDFILE")" 2>/dev/null; then
  echo "a DeepSeek server is already running (PID $(cat "$PIDFILE")); ../native/stop.sh first" >&2; exit 2
fi
if ss -ltn | grep -q ":$PORT "; then echo "port $PORT is in use; stop the other server first" >&2; exit 2; fi

LOG="$ROOT/logs/$PROFILE-server.log"
CONFIG="$ROOT/logs/$PROFILE-config.json"
ENGINE_LOG="$ROOT/logs/$PROFILE-engine.log"
PROFILE="$PROFILE" EXE="$EXE" PACK="$PACK" CTX="$CTX" MMAP_EXPERTS="$MMAP_EXPERTS" DSPARK="$DSPARK" \
CPU_THREADS="$CPU_THREADS" VISION="$VISION" PORT="$PORT" ENGINE_LOG="$ENGINE_LOG" STRATA_DIR="$STRATA_DIR" python3 -I - > "$CONFIG" <<'EOF'
import json, os
e = os.environ
print(json.dumps({
    "exe": e["EXE"],
    "args": ["--pack", e["PACK"], "--max-context", e["CTX"], "--eos", "1",
             "--dspark", e["DSPARK"], "--mmap-experts", e["MMAP_EXPERTS"], "--vision", e["VISION"],
             # all six routed experts a word (the engine drops to five on cards of 16 GiB or less)
             "--experts", "6"]
            + (["--cpu-threads", e["CPU_THREADS"]] if e["CPU_THREADS"] != "0" else []),
    "cwd": e["STRATA_DIR"],
    "tokenizer": e["PACK"] + "/tokenizer",
    "model_name": e["PROFILE"],
    "arch": "deepseek41",
    "log": e["ENGINE_LOG"],
    "port": int(e["PORT"]),
    "gpu": 0,
    # a 500 GB model: the first start reads the shards from the SSD
    "engine_ready_s": 3600,
    "allowed_hosts": "*",
    # the toolkit's chat UI (scripts/start-ui.sh) calls /v1 from this origin in the browser
    "cors_origins": ["http://127.0.0.1:5173", "http://localhost:5173"],
    "sampling": {"temperature": 1, "top_p": 1.0},
    "api_monitor": True,
    # agents send max_tokens 65536 whatever the prompt: shorten it to the room left instead of a 400
    "fit_max_tokens": True,
}, indent=2))
EOF

echo "$PROFILE: $SHARDS"
echo "  pack $PACK, context $CTX, mmap experts $MMAP_EXPERTS, DSpark $DSPARK, vision $VISION, CPUs $CPUSET, bind $HOST:$PORT"
# Run the server in its own systemd scope: systemd-oomd and memory accounting then treat it
# separately, so a kill during the heavy load phase cannot take the terminal (or VS Code) with it.
# With --scope, systemd-run execs the command, so $! is still the server PID.
SCOPE=()
if systemd-run --user --scope --quiet --collect true 2>/dev/null; then
  SCOPE=(systemd-run --user --scope --quiet --collect --unit="$PROFILE-$$")
fi
# setsid: own process group, so stop.sh reaches the server and its engine.
# coredump_filter 0: a crashed engine holds ~200 GB of host memory, partly mapped by the NVIDIA
# driver; dumping it can leave the process in state D for hours (see qwen3.8-flash-next/strata).
cd "$STRATA_DIR"
echo 0 > /proc/self/coredump_filter
setsid nohup "${SCOPE[@]}" taskset -c "$CPUSET" "$PYTHON" -m serve.server --engine strata \
    --config "$CONFIG" --host "$HOST" --port "$PORT" > "$LOG" 2>&1 < /dev/null &
echo $! > "$PIDFILE"
echo "$PROFILE" > "$STATE"
echo "started PID $(cat "$PIDFILE"), log: $LOG (engine: $ENGINE_LOG); loading takes several minutes"

for _ in $(seq 1 720); do
  curl -sf -m 3 "http://127.0.0.1:$PORT/health" 2>/dev/null | grep -q '"loaded": *true' && break
  if ! kill -0 "$(cat "$PIDFILE")" 2>/dev/null; then
    echo "server exited:"; tail -20 "$LOG"; tail -20 "$ENGINE_LOG"; rm -f "$PIDFILE" "$STATE"; exit 1
  fi
  sleep 5
done
curl -s -m 3 "http://127.0.0.1:$PORT/health" || true; echo
echo "endpoint: http://$HOST:$PORT/v1 (model $PROFILE)"
