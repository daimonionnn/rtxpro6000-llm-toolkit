#!/usr/bin/env bash
# Profile vllm-fp8-offload: the official Qwen3.8-Flash-Next FP8 checkpoint on vLLM,
# one RTX PRO 6000 Blackwell (96 GB), with part of the routed experts in host RAM.
#
#   MODEL_DIR=/models/Qwen3.8-Flash-Next-FP8 ./serve-fp8-offload.sh
#   ./stop.sh
#
# Checkpoint Qwen/Qwen3.8-Flash-Next-FP8: routed experts in block FP8 (128x128,
# dynamic activation scales, ~112 GiB); attention, linear attention, shared
# experts, router, embeddings, MTP and vision stay BF16. The experts alone do not
# fit in 96 GB, so:
#
#   PLE n-gram table   host RAM, vLLM's PLE offload worker (VLLM_PLE_CPU_OFFLOAD=1)
#   OFFLOAD_GIB of     pinned host RAM through vLLM's UVA offloader
#   routed experts     (--offload-backend uva --cpu-offload-params experts): the GPU
#                      reads them in place over PCIe, so each forward pass only
#                      moves the expert weights it touches
#
# There is no published single-GPU recipe for this checkpoint; this is the
# closest path the preview image supports. Measured with the defaults: decode
# 15.6 tok/s, prefill ~700-800 tok/s — both bound by PCIe (~23 GB/s host-to-GPU
# during decode, Gen5 x16), since every forward pass reads the offloaded experts it
# touches. Offloading less is the only lever that helps. See docs/vllm-fp8-offload.md.
#
# --cpu-offload-params experts matters: without it vLLM offloads parameters in
# declaration order and moves the hot dense weights to RAM first.
# --distributed-executor-backend mp is required on one GPU (see vllm-awq.md).
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"

IMAGE="${IMAGE:-vllm/vllm-openai:qwen38-flash-next}"
NAME="${NAME:-rtxpro6000-llm}"
PORT="${PORT:-8090}"
MODEL_DIR="${MODEL_DIR:?set MODEL_DIR to the local checkpoint directory}"

CTX="${CTX:-262144}"
GPU_UTIL="${GPU_UTIL:-0.92}"         # fraction of VRAM vLLM may use for weights + KV cache
MAX_SEQS="${MAX_SEQS:-4}"            # concurrent requests; sized for one agent
OFFLOAD_GIB="${OFFLOAD_GIB:-50}"     # GiB of routed experts kept in pinned host RAM; 50 leaves a
                                     # 313K-token KV pool, the least offload that keeps 262K context
BATCH_TOKENS="${BATCH_TOKENS:-16384}" # --max-num-batched-tokens, the prefill chunk size
KV_DTYPE="${KV_DTYPE:-auto}"         # auto = BF16; this image's QSA attention requires BF16
MOE_BACKEND="${MOE_BACKEND:-auto}"   # auto picks Triton for block FP8 on SM120; marlin runs it as W8A16
SPEC_TOKENS="${SPEC_TOKENS:-0}"      # >0 enables MTP speculative decoding with that many draft tokens
EXTRA_ARGS="${EXTRA_ARGS:-}"

python3 "$HERE/../../quant_info.py" "$MODEL_DIR" \
  --require fp8 --kv-dtype "$KV_DTYPE" --ple-mode ram --experts-ram-gib "$OFFLOAD_GIB" \
  --engine vllm || exit 2

# PLE table (FP8, ~48 GiB) + offloaded experts, both pinned, plus headroom
need_gib=$(( 48 + ${OFFLOAD_GIB%.*} + 8 ))
free_gib=$(awk '/MemAvailable/{printf "%d", $2/1048576}' /proc/meminfo)
if [ "$free_gib" -lt "$need_gib" ]; then
  echo "only ${free_gib} GiB of host RAM available; PLE table + ${OFFLOAD_GIB} GiB of experts need ~${need_gib} GiB" >&2
  exit 2
fi

SPEC_ARGS=""
[ "$SPEC_TOKENS" -gt 0 ] && SPEC_ARGS="--speculative-config {\"method\":\"mtp\",\"num_speculative_tokens\":$SPEC_TOKENS}"
echo "image: $IMAGE   context $CTX, max-num-seqs $MAX_SEQS, experts offloaded ${OFFLOAD_GIB} GiB, chunk $BATCH_TOKENS, MoE backend $MOE_BACKEND, MTP draft tokens $SPEC_TOKENS"

docker rm -f "$NAME" >/dev/null 2>&1 || true

docker run -d --name "$NAME" --restart unless-stopped --gpus '"device=0"' \
  --ipc host --shm-size 32g \
  --ulimit memlock=-1 \
  --label "rtxpro6000-llm.profile=qwen3.8-flash-next-vllm-fp8-offload" \
  --label "rtxpro6000-llm.quant.experts=FP8-block128 (${OFFLOAD_GIB} GiB in RAM)" \
  --label "rtxpro6000-llm.quant.rest=BF16" \
  --label "rtxpro6000-llm.quant.kv_cache=$KV_DTYPE" \
  --label "rtxpro6000-llm.quant.ple=FP8/ram" \
  -p "127.0.0.1:${PORT}:8000" \
  -v "$MODEL_DIR:/model:ro" \
  -e VLLM_PLE_CPU_OFFLOAD=1 \
  "$IMAGE" \
  --model /model \
  --served-model-name Qwen3.8-Flash-Next \
  --host 0.0.0.0 --port 8000 \
  --max-model-len "$CTX" \
  --max-num-seqs "$MAX_SEQS" \
  --max-num-batched-tokens "$BATCH_TOKENS" \
  --gpu-memory-utilization "$GPU_UTIL" \
  --distributed-executor-backend mp \
  --offload-backend uva --cpu-offload-gb "$OFFLOAD_GIB" --cpu-offload-params experts \
  --moe-backend "$MOE_BACKEND" \
  --kv-cache-dtype "$KV_DTYPE" \
  --enable-prefix-caching \
  --no-enable-flashinfer-autotune \
  --enable-auto-tool-choice --tool-call-parser qwen3_xml \
  --reasoning-parser qwen3 \
  $SPEC_ARGS \
  $EXTRA_ARGS

echo "waiting for the server (loads 186 GiB and pins ~$(( 48 + ${OFFLOAD_GIB%.*} )) GiB of RAM — allow 10-25 min)"
for _ in $(seq 1 180); do
  if curl -sf --max-time 3 "http://127.0.0.1:${PORT}/health" >/dev/null 2>&1; then
    echo "ready on http://127.0.0.1:${PORT}  ·  vLLM, FP8 experts (${OFFLOAD_GIB} GiB in RAM), BF16 rest, PLE FP8 in RAM, KV $KV_DTYPE"; exit 0
  fi
  if ! docker ps --format '{{.Names}}' | grep -qx "$NAME"; then
    echo "container exited:"; docker logs --tail 60 "$NAME"; exit 1
  fi
  sleep 10
done
echo "timed out; last log lines:"; docker logs --tail 60 "$NAME"; exit 1
