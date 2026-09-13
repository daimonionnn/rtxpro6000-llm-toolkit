#!/usr/bin/env bash
# Profile sglang-nvfp4-nvme: SGLang, NVFP4, PLE table streamed from NVMe, one RTX PRO 6000 (96 GB).
#
# Uses the Docker image built by ../build-local-image/build.sh.
#
#   MODEL_DIR=/models/Qwen3.8-Flash-Next-NVFP4 ./serve-nvfp4-nvme.sh
#
# The 51B-parameter N-gram (PLE) table is NOT loaded into VRAM or host RAM — it
# is read straight off NVMe with io_uring as each token needs it. That is what
# makes a 176B model fit on one 96 GB card with a 233K-token KV cache.
#
# ── Quantization: what runs at which precision ──────────────────────────────
# Checkpoint RadixArk/Qwen3.8-Flash-Next-NVFP4 (modelopt 0.46.0). It is NOT
# uniformly 4-bit:
#
#   routed MoE experts   NVFP4 W4A4 — 4-bit float weights and activations, FP8
#                        E4M3 scale per block of 16. The bulk of the parameters.
#   everything else      BF16 — attention, linear attention, router, shared
#                        experts, hyper-connections, MTP draft, vision, lm_head
#                        (the checkpoint's exclude_modules).
#   PLE n-gram table     FP8 E4M3, 47.7 GiB — streamed from NVMe, never resident.
#   KV cache             fp8_e4m3 — a launch choice (KV_DTYPE), not the checkpoint's.
#
# quant_info.py reads this from MODEL_DIR at launch, prints it, and aborts the
# launch if the checkpoint is not NVFP4.
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"

IMAGE="${IMAGE:-sglang-flashnext-sm120:local}"
NAME="${NAME:-rtxpro6000-llm}"
PORT="${PORT:-8090}"
MODEL_DIR="${MODEL_DIR:?set MODEL_DIR to the local checkpoint directory}"
SECCOMP="${SECCOMP:-$HERE/../build-local-image/seccomp-iouring.json}"

# ── Tuning (measured on this card; see BENCHMARKS.md) ───────────────────────
CTX="${CTX:-262144}"           # native window
MAX_TOTAL_TOKENS="${MAX_TOTAL_TOKENS:-393216}"  # request; engine clamps to ~233,856 with fp8 KV
MEMFRAC="${MEMFRAC:-0.95}"
MAXRUN="${MAXRUN:-8}"
MAMBA_SLOTS="${MAMBA_SLOTS:-25}"   # recurrent-state slots; the real concurrency limit (~5)
CHUNKED="${CHUNKED:-8192}"

# ── Quantization knobs ──────────────────────────────────────────────────────
WEIGHT_QUANT="${WEIGHT_QUANT:-modelopt_fp4}"           # --quantization; must match the checkpoint
FP4_GEMM_BACKEND="${FP4_GEMM_BACKEND:-flashinfer_cudnn}"
KV_DTYPE="${KV_DTYPE:-fp8_e4m3}"                       # 'auto' = BF16 KV, roughly half the tokens

# ── PLE table: NVMe only ────────────────────────────────────────────────────
# The 47.68 GiB PLE table is read off NVMe with io_uring, opened O_DIRECT, so it
# bypasses the page cache and host RAM stays free. That is what the "nvme" in this
# file's name means.
#
# For the table in pinned host RAM use ../nvfp4-ram/serve-nvfp4-ram.sh instead. It needs a
# smaller mamba state cache to fit, and on this card it gives more KV cache and
# faster prefill than this launcher. See docs/ple-ram-experiment.md.

# Extra launch flags, e.g. EXTRA_ARGS="--mamba-ssm-dtype bfloat16"
EXTRA_ARGS="${EXTRA_ARGS:-}"

# --enable-strict-thinking is only needed if you send per-request
# max_thinking_tokens (PR #36750). It requires xgrammar, which is already the
# default grammar backend, and does not change tool-call grammars.
STRICT_ARG=""
[ "${STRICT_THINKING:-1}" = "1" ] && STRICT_ARG="--enable-strict-thinking"

python3 "$HERE/../../quant_info.py" "$MODEL_DIR" \
  --weight-quant "$WEIGHT_QUANT" --kv-dtype "$KV_DTYPE" \
  --fp4-gemm-backend "$FP4_GEMM_BACKEND" --ple-mode nvme || exit 2

docker rm -f "$NAME" >/dev/null 2>&1 || true

# --security-opt seccomp: Docker's default profile blocks io_uring_setup /
# io_uring_enter / io_uring_register. Without this profile the NVMe PLE reader
# cannot start. This is the single most likely reason a first run fails.
docker run -d --name "$NAME" --restart unless-stopped --gpus '"device=0"' \
  --ipc host --shm-size 32g \
  --security-opt label=disable \
  --security-opt seccomp="$SECCOMP" \
  --label "rtxpro6000-llm.profile=qwen3.8-flash-next-sglang-nvfp4-nvme" \
  --label "rtxpro6000-llm.quant.experts=NVFP4-W4A4" \
  --label "rtxpro6000-llm.quant.rest=BF16" \
  --label "rtxpro6000-llm.quant.kv_cache=$KV_DTYPE" \
  --label "rtxpro6000-llm.quant.ple=FP8_E4M3/nvme" \
  -p "127.0.0.1:${PORT}:8000" \
  -v "$MODEL_DIR:/model:ro" \
  -e SGLANG_QWEN4_PLE_NVME_PATH=/model \
  -e SGLANG_QWEN4_PLE_NVME_BACKEND=io_uring \
  -e SGLANG_QWEN4_PLE_NVME_QUEUE_DEPTH=512 \
  -e SGLANG_QWEN4_PLE_NVME_LOG_INTERVAL=10000 \
  "$IMAGE" \
  python -m sglang.launch_server \
    --model-path /model \
    --served-model-name Qwen3.8-Flash-Next \
    --host 0.0.0.0 --port 8000 \
    --quantization "$WEIGHT_QUANT" \
    --fp4-gemm-backend "$FP4_GEMM_BACKEND" \
    --kv-cache-dtype "$KV_DTYPE" \
    --page-size 64 \
    --mamba-radix-cache-strategy extra_buffer \
    --mamba-track-interval 64 \
    --chunked-prefill-size "$CHUNKED" \
    --max-running-requests "$MAXRUN" \
    --context-length "$CTX" \
    --max-total-tokens "$MAX_TOTAL_TOKENS" \
    --max-mamba-cache-size "$MAMBA_SLOTS" \
    --mem-fraction-static "$MEMFRAC" \
    --speculative-algorithm NEXTN \
    --speculative-num-steps 3 \
    --speculative-eagle-topk 1 \
    --speculative-num-draft-tokens 4 \
    --cuda-graph-backend-decode breakable \
    --cuda-graph-backend-prefill disabled \
    --reasoning-parser auto --tool-call-parser auto \
    $STRICT_ARG \
    --enable-metrics \
    $EXTRA_ARGS

echo "waiting for the server (cold start ~3 min, ~1.5 min with a warm page cache)"
for _ in $(seq 1 60); do
  if docker logs "$NAME" 2>&1 | grep -q "The server is fired up"; then
    echo "ready on http://127.0.0.1:${PORT}  ·  NVFP4 experts, BF16 rest, KV $KV_DTYPE, PLE FP8 streamed from NVMe"; exit 0
  fi
  if ! docker ps --format '{{.Names}}' | grep -qx "$NAME"; then
    echo "container exited:"; docker logs --tail 40 "$NAME"; exit 1
  fi
  sleep 10
done
echo "timed out; last log lines:"; docker logs --tail 40 "$NAME"; exit 1
