#!/usr/bin/env bash
# Profile exllamav3-exl3-5.05bpw: Qwen3.8-Flash-Next EXL3 5.05 bpw on ExLlamaV3 via
# TabbyAPI, n-gram table in RAM, MTP drafting, one RTX PRO 6000 Blackwell (96 GB).
#
#   MODEL_DIR=/models/Qwen3.8-Flash-Next-EXL3-5.05bpw ./serve-exl3-5.05bpw.sh
#   ./stop.sh
#
# Checkpoint turboderp/Qwen3.8-Flash-Next-exl3, branch 5.05bpw_h6_ng6: ExLlamaV3's
# trellis quantization at 5.05 bits per weight for the linear layers, 6-bit output
# head and n-gram table. In the quantizer's own KL-divergence chart it sits much
# closer to BF16 than the 4-bit formats: 0.0040, against 0.0100 for NVFP4 W4A16
# and 0.0241 for NVFP4 W4A4 (in-domain text, not a multilingual measurement).
#
# The weights (~77 GB) go to the GPU; ngram_ram keeps the n-gram table in host
# RAM instead of reading it from disk per token (~45 GB of RAM).
#
# Image: ghcr.io/theroyallab/tabbyapi:cu13, pinned by digest (built 2026-09-13,
# ExLlamaV3 1.5.0, torch 2.11 cu130). Earlier cu13 builds lacked Python headers,
# and Triton failed on the first Gated DeltaNet kernel ("Python.h: No such file or
# directory"); this build has them. The config is generated from
# config.template.yml into config.generated.yml.
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"

IMAGE="${IMAGE:-ghcr.io/theroyallab/tabbyapi@sha256:a0befeadd9b4609e5a39334aa587b5bd8da33f4eeb6c68c482f2d1751fad79d3}"
NAME="${NAME:-rtxpro6000-llm}"
PORT="${PORT:-8090}"
MODEL_DIR="${MODEL_DIR:?set MODEL_DIR to the local checkpoint directory}"

CTX="${CTX:-262144}"                  # max_seq_len and cache_size; a multiple of 256
CACHE_MODE="${CACHE_MODE:-FP16}"      # FP16, or k_bits,v_bits (e.g. 8,8) to halve the KV cache
MAX_SEQS="${MAX_SEQS:-4}"             # max_batch_size
CHUNK="${CHUNK:-4096}"                # prefill chunk size
DRAFT_MODE="${DRAFT_MODE:-mtp}"       # mtp or disabled
EXTRA_ENV="${EXTRA_ENV:-}"            # extra -e VAR=value arguments for docker run

python3 "$HERE/../../quant_info.py" "$MODEL_DIR" \
  --require exl3 --kv-dtype "$CACHE_MODE" --ple-mode ram --engine exllamav3 || exit 2

free_gib=$(awk '/MemAvailable/{printf "%d", $2/1048576}' /proc/meminfo)
if [ "$free_gib" -lt 50 ]; then
  echo "only ${free_gib} GiB of host RAM available; the n-gram table needs ~45 GiB" >&2
  exit 2
fi

sed -e "s|__CTX__|$CTX|g" -e "s|__CACHE_MODE__|$CACHE_MODE|" -e "s|__MAX_SEQS__|$MAX_SEQS|" \
    -e "s|__CHUNK__|$CHUNK|" -e "s|__DRAFT_MODE__|$DRAFT_MODE|" \
    "$HERE/config.template.yml" > "$HERE/config.generated.yml"
mkdir -p "$HERE/cache"
echo "image: $IMAGE   context $CTX, cache $CACHE_MODE, max batch $MAX_SEQS, chunk $CHUNK, draft $DRAFT_MODE"

docker rm -f "$NAME" >/dev/null 2>&1 || true

# shellcheck disable=SC2086
docker run -d --name "$NAME" --restart unless-stopped --gpus '"device=0"' \
  --ipc host --shm-size 8g \
  --ulimit memlock=-1 \
  --label "rtxpro6000-llm.profile=qwen3.8-flash-next-exllamav3-exl3-5.05bpw" \
  --label "rtxpro6000-llm.quant.experts=EXL3-5.05bpw" \
  --label "rtxpro6000-llm.quant.rest=EXL3-5.05bpw/head-6bpw" \
  --label "rtxpro6000-llm.quant.kv_cache=$CACHE_MODE" \
  --label "rtxpro6000-llm.quant.ple=EXL3-6bpw/ram" \
  -p "127.0.0.1:${PORT}:8000" \
  -v "$MODEL_DIR:/app/models/Qwen3.8-Flash-Next:ro" \
  -v "$HERE/config.generated.yml:/app/config.yml:ro" \
  -v "$HERE/sampler-qwen38-flash-next.yml:/app/sampler_overrides/qwen38_flash_next.yml:ro" \
  -v "$HERE/cache:/root/.cache" \
  $EXTRA_ENV \
  "$IMAGE"

echo "waiting for the server (loads ~123 GB; first start compiles kernels — allow 5-15 min)"
for _ in $(seq 1 180); do
  if curl -sf --max-time 3 "http://127.0.0.1:${PORT}/v1/models" >/dev/null 2>&1; then
    echo "ready on http://127.0.0.1:${PORT}  ·  ExLlamaV3, EXL3 5.05 bpw, n-gram table in RAM, cache $CACHE_MODE, draft $DRAFT_MODE"; exit 0
  fi
  if ! docker ps --format '{{.Names}}' | grep -qx "$NAME"; then
    echo "container exited:"; docker logs --tail 60 "$NAME"; exit 1
  fi
  sleep 10
done
echo "timed out; last log lines:"; docker logs --tail 60 "$NAME"; exit 1
