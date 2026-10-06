#!/usr/bin/env bash
set -euo pipefail
source "$(dirname "$0")/_lib.sh"
start_profile qwen3.8-flash-next-strata-q8-vision-256k "$@"
