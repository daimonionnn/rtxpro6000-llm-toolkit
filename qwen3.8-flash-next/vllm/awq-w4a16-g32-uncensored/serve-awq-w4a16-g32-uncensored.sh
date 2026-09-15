#!/usr/bin/env bash
# Profile vllm-awq-w4a16-g32-uncensored: an uncensored Qwen3.8-Flash-Next, AWQ INT4
# group 32, on vLLM with a one-function patch, PLE table in RAM, one RTX PRO 6000
# Blackwell (96 GB).
#
#   MODEL_DIR=/models/Qwen3.8-Flash-Next-Uncensored-AWQ-g32 ./serve-awq-w4a16-g32-uncensored.sh
#   ./stop.sh
#
# Checkpoint leoncca/Qwen3.8-Flash-Next-Uncensored-AWQ-g32, quantized from the
# abliterated orcarouter/Qwen3.8-Flash-Next-Uncensored:
#
#   routed experts   INT4 AWQ (AutoAWQ GEMM layout), group 32, zero points
#   everything else  BF16
#   PLE n-gram table the official FP8 table with its global scale (47.7 GiB)
#
# The preview image loads the AWQ experts through its Marlin MoE path, but it only
# recognises an FP8 PLE table when the whole checkpoint is FP8. Here the table
# would be created in BF16 and the FP8 bytes copied in without their scale.
# patches/0001-fp8-ple-with-non-fp8-quantization.patch selects the FP8 PLE method
# from text_config.ple_embedding_dtype as well; this launcher applies it to the
# image's own ple_layer.py at every start and mounts the result read-only.
#
# Two of the reused official FP8 PLE shard files also hold tensors the index does
# not list (1.0 GiB of FP8 experts and duplicate PLE buffers); vLLM reads every
# tensor in a file and fails on them. index_filter.py builds MODEL_DIR.indexed/ —
# hard links plus two rewritten files — and that directory is what gets served.
#
# --distributed-executor-backend mp is required on a single GPU (see
# docs/profiles/vllm-awq-w4a16.md).
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"

IMAGE="${IMAGE:-vllm/vllm-openai:qwen38-flash-next}"
NAME="${NAME:-rtxpro6000-llm}"
PORT="${PORT:-8090}"
MODEL_DIR="${MODEL_DIR:?set MODEL_DIR to the local checkpoint directory}"

CTX="${CTX:-262144}"
GPU_UTIL="${GPU_UTIL:-0.92}"         # fraction of VRAM vLLM may use for weights + KV cache
MAX_SEQS="${MAX_SEQS:-4}"            # concurrent requests; sized for one agent
KV_DTYPE="${KV_DTYPE:-auto}"         # auto = BF16; this image's QSA attention requires BF16
SPEC_TOKENS="${SPEC_TOKENS:-0}"      # >0 enables MTP speculative decoding with that many draft tokens
EXTRA_ARGS="${EXTRA_ARGS:-}"

PLE_LAYER=/usr/local/lib/python3.12/dist-packages/vllm/models/qwen3_8_flash_next/nvidia/ple_layer.py
PATCH="$HERE/patches/0001-fp8-ple-with-non-fp8-quantization.patch"
BUILD="$HERE/patches/.build"

python3 "$HERE/../../quant_info.py" "$MODEL_DIR" \
  --require awq --kv-dtype "$KV_DTYPE" --ple-mode ram --engine vllm || exit 2

free_gib=$(awk '/MemAvailable/{printf "%d", $2/1048576}' /proc/meminfo)
if [ "$free_gib" -lt 60 ]; then
  echo "only ${free_gib} GiB of host RAM available; the FP8 PLE table needs ~48 GiB" >&2
  exit 2
fi

SERVE_DIR="$(python3 "$HERE/index_filter.py" "$MODEL_DIR")" || exit 2

mkdir -p "$BUILD"
docker run --rm --network none --entrypoint cat "$IMAGE" "$PLE_LAYER" > "$BUILD/ple_layer.orig.py"
patch --quiet -o "$BUILD/ple_layer.py" "$BUILD/ple_layer.orig.py" < "$PATCH" \
  || { echo "the PLE patch does not apply to $IMAGE's ple_layer.py" >&2; exit 2; }

SPEC_ARGS=""
[ "$SPEC_TOKENS" -gt 0 ] && SPEC_ARGS="--speculative-config {\"method\":\"mtp\",\"num_speculative_tokens\":$SPEC_TOKENS}"
echo "image: $IMAGE (+ FP8 PLE patch)   context $CTX, max-num-seqs $MAX_SEQS, gpu-memory-utilization $GPU_UTIL, MTP draft tokens $SPEC_TOKENS"

docker rm -f "$NAME" >/dev/null 2>&1 || true

# shellcheck disable=SC2086
docker run -d --name "$NAME" --restart unless-stopped --gpus '"device=0"' \
  --ipc host --shm-size 32g \
  --ulimit memlock=-1 \
  --label "rtxpro6000-llm.profile=qwen3.8-flash-next-vllm-awq-w4a16-g32-uncensored" \
  --label "rtxpro6000-llm.quant.experts=INT4-AWQ-g32-zp" \
  --label "rtxpro6000-llm.quant.rest=BF16" \
  --label "rtxpro6000-llm.quant.kv_cache=$KV_DTYPE" \
  --label "rtxpro6000-llm.quant.ple=FP8/ram" \
  -p "127.0.0.1:${PORT}:8000" \
  -v "$SERVE_DIR:/model:ro" \
  -v "$BUILD/ple_layer.py:$PLE_LAYER:ro" \
  -e VLLM_PLE_CPU_OFFLOAD=1 \
  "$IMAGE" \
  --model /model \
  --served-model-name Qwen3.8-Flash-Next \
  --host 0.0.0.0 --port 8000 \
  --max-model-len "$CTX" \
  --max-num-seqs "$MAX_SEQS" \
  --gpu-memory-utilization "$GPU_UTIL" \
  --distributed-executor-backend mp \
  --kv-cache-dtype "$KV_DTYPE" \
  --enable-prefix-caching \
  --no-enable-flashinfer-autotune \
  --enable-auto-tool-choice --tool-call-parser qwen3_xml \
  --reasoning-parser qwen3 \
  $SPEC_ARGS \
  $EXTRA_ARGS

echo "waiting for the server (first start compiles and loads 129 GiB — allow 10-20 min)"
for _ in $(seq 1 180); do
  if curl -sf --max-time 3 "http://127.0.0.1:${PORT}/health" >/dev/null 2>&1; then
    echo "ready on http://127.0.0.1:${PORT}  ·  vLLM, uncensored AWQ g32 experts, BF16 rest, PLE FP8 in RAM, KV $KV_DTYPE"; exit 0
  fi
  if ! docker ps --format '{{.Names}}' | grep -qx "$NAME"; then
    echo "container exited:"; docker logs --tail 60 "$NAME"; exit 1
  fi
  sleep 10
done
echo "timed out; last log lines:"; docker logs --tail 60 "$NAME"; exit 1
