#!/usr/bin/env bash
# Start ExLlamaV3 + TabbyAPI, EXL3 5.05 bpw, n-gram table in RAM, MTP.
# MODEL_DIR defaults to the profile's checkpoint under models/; launcher variables pass through.
set -euo pipefail
source "$(dirname "$0")/_lib.sh"

start_profile qwen3.8-flash-next-exllamav3-exl3-5.05bpw
