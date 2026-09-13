#!/usr/bin/env bash
# Profile vllm-awq-w4a16: Qwen3.8-Flash-Next AWQ W4A16 on vLLM, PLE table in RAM,
# one RTX PRO 6000 Blackwell (96 GB).
#
#   MODEL_DIR=/models/Qwen3.8-Flash-Next-AWQ-W4A16 ./serve-awq-w4a16-ram.sh
#   ./stop.sh
#
# Checkpoint wtdcode/Qwen3.8-Flash-Next-AWQ-W4A16 (compressed-tensors). Unlike the
# NVFP4 checkpoint used by the SGLang profiles:
#
#   routed experts   INT4 weight-only, group 128, symmetric (4.13 bits/param) —
#                    activations stay BF16; runs on Marlin kernels
#   everything else  BF16, including the 51B PLE n-gram table (95.4 GiB)
#
# ~73 GiB of weights go to the GPU. The PLE table is held in host RAM by vLLM's
# PLE offload worker (VLLM_PLE_CPU_OFFLOAD=1), which needs ~96 GiB of free RAM.
#
# Image vllm/vllm-openai:qwen38-flash-next is the vLLM recipe's preview build for
# this model; PyPI vLLM does not support it.
#
# --distributed-executor-backend mp is required on a single GPU. The PLE offload
# worker is spawned and awaited only by the multiprocess executor; with TP=1 vLLM
# otherwise runs the model in-process (uniproc executor), which never starts the
# worker, and startup hangs after CUDA graph capture with EngineCore spinning.
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"

IMAGE="${IMAGE:-vllm/vllm-openai:qwen38-flash-next}"
NAME="${NAME:-rtxpro6000-llm}"
PORT="${PORT:-8090}"
MODEL_DIR="${MODEL_DIR:?set MODEL_DIR to the local checkpoint directory}"

CTX="${CTX:-262144}"
GPU_UTIL="${GPU_UTIL:-0.92}"         # fraction of VRAM vLLM may use for weights + KV cache
MAX_SEQS="${MAX_SEQS:-4}"            # concurrent requests; sized for one agent
KV_DTYPE="${KV_DTYPE:-auto}"         # auto = BF16; fp8 roughly doubles the KV pool
SPEC_TOKENS="${SPEC_TOKENS:-0}"      # >0 enables MTP speculative decoding with that many draft tokens
EXTRA_ARGS="${EXTRA_ARGS:-}"

python3 "$HERE/../../quant_info.py" "$MODEL_DIR" \
  --require w4a16 --kv-dtype "$KV_DTYPE" --ple-mode ram --engine vllm || exit 2

free_gib=$(awk '/MemAvailable/{printf "%d", $2/1048576}' /proc/meminfo)
if [ "$free_gib" -lt 100 ]; then
  echo "only ${free_gib} GiB of host RAM available; the BF16 PLE table needs ~96 GiB" >&2
  exit 2
fi

SPEC_ARGS=""
[ "$SPEC_TOKENS" -gt 0 ] && SPEC_ARGS="--speculative-config {\"method\":\"mtp\",\"num_speculative_tokens\":$SPEC_TOKENS}"
echo "image: $IMAGE   context $CTX, max-num-seqs $MAX_SEQS, gpu-memory-utilization $GPU_UTIL, MTP draft tokens $SPEC_TOKENS"

docker rm -f "$NAME" >/dev/null 2>&1 || true

docker run -d --name "$NAME" --restart unless-stopped --gpus '"device=0"' \
  --ipc host --shm-size 32g \
  --ulimit memlock=-1 \
  --label "rtxpro6000-llm.profile=qwen3.8-flash-next-vllm-awq-w4a16" \
  --label "rtxpro6000-llm.quant.experts=INT4-W4A16-g128" \
  --label "rtxpro6000-llm.quant.rest=BF16" \
  --label "rtxpro6000-llm.quant.kv_cache=$KV_DTYPE" \
  --label "rtxpro6000-llm.quant.ple=BF16/ram" \
  -p "127.0.0.1:${PORT}:8000" \
  -v "$MODEL_DIR:/model:ro" \
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

echo "waiting for the server (first start compiles and loads 168 GiB — allow 10-20 min)"
for _ in $(seq 1 180); do
  if curl -sf --max-time 3 "http://127.0.0.1:${PORT}/health" >/dev/null 2>&1; then
    echo "ready on http://127.0.0.1:${PORT}  ·  vLLM, AWQ W4A16 experts, BF16 rest, PLE BF16 in RAM, KV $KV_DTYPE"; exit 0
  fi
  if ! docker ps --format '{{.Names}}' | grep -qx "$NAME"; then
    echo "container exited:"; docker logs --tail 60 "$NAME"; exit 1
  fi
  sleep 10
done
echo "timed out; last log lines:"; docker logs --tail 60 "$NAME"; exit 1
