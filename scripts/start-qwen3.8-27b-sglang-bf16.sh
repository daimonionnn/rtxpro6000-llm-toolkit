#!/usr/bin/env bash
# Start Qwen3.8-27B, SGLang, official image, BF16, NEXTN speculation.
# MODEL_DIR defaults to the profile's checkpoint under models/; launcher variables pass through.
set -euo pipefail
source "$(dirname "$0")/_lib.sh"

start_profile qwen3.8-27b-sglang-bf16
