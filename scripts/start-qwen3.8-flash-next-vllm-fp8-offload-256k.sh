#!/usr/bin/env bash
# Start vLLM, official image, official FP8 checkpoint, part of the experts and the PLE table in RAM.
# MODEL_DIR defaults to the profile's checkpoint under models/; launcher variables pass through.
set -euo pipefail
source "$(dirname "$0")/_lib.sh"

start_profile qwen3.8-flash-next-vllm-fp8-offload
