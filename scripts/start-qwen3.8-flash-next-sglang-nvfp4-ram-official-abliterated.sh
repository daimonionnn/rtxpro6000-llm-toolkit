#!/usr/bin/env bash
# Start SGLang, official lmsysorg image, abliterated NVFP4 checkpoint (dealignai), PLE table in RAM.
# Same launcher and settings as sglang-nvfp4-ram-official; only the checkpoint differs.
# MODEL_DIR defaults to the profile's checkpoint under models/; launcher variables pass through.
set -euo pipefail
source "$(dirname "$0")/_lib.sh"
# Tuned for one agent. Keep BF16 KV: FP8 KV crashes this image on long prompts.
export MAXRUN="${MAXRUN:-4}" MAMBA_SLOTS="${MAMBA_SLOTS:-12}" KV_DTYPE="${KV_DTYPE:-auto}"
start_profile qwen3.8-flash-next-sglang-nvfp4-ram-official-abliterated
