#!/usr/bin/env bash
# Start SGLang, local Docker image, NVFP4, PLE table in RAM.
# MODEL_DIR defaults to the profile's checkpoint under models/; launcher variables pass through.
set -euo pipefail
source "$(dirname "$0")/_lib.sh"

start_profile qwen3.8-flash-next-sglang-nvfp4-ram
