#!/usr/bin/env bash
# Start the SGLang pennyroyal fork with HiCache: prefixes persist to disk through NIXL and survive restarts.
# MODEL_DIR defaults to the profile's checkpoint under models/; launcher variables pass through.
set -euo pipefail
source "$(dirname "$0")/_lib.sh"
export HICACHE=1
start_profile qwen3.8-flash-next-sglang-nvfp4-ram-pennyroyal
