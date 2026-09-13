#!/usr/bin/env bash
# Stop whichever profile is serving and wait until its VRAM is released.
#   scripts/stop.sh          stop it (a Docker container is kept, stopped)
#   scripts/stop.sh --rm     also remove the Docker container
set -euo pipefail
source "$(dirname "$0")/_lib.sh"
running=$(running_profiles)
if [ -z "$running" ]; then
  echo "nothing is running"
  exit 0
fi
for p in $running; do
  echo "== $p"
  if [ "$p" = qwen3.8-flash-next-sglang-nvfp4-ram-pennyroyal ]; then
    "$ROOT/qwen3.8-flash-next/sglang/nvfp4-ram-pennyroyal/stop.sh"
  else
    "$ROOT/common/stop-docker.sh" "$@"
  fi
done
