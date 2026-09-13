#!/usr/bin/env bash
# Stop whichever variant is serving and wait until its VRAM is released.
#   scripts/stop.sh          stop it (a Docker container is kept, stopped)
#   scripts/stop.sh --rm     also remove the Docker container
set -euo pipefail
source "$(dirname "$0")/_lib.sh"
running=$(running_variants)
if [ -z "$running" ]; then
  echo "nothing is running"
  exit 0
fi
for v in $running; do
  echo "== $v"
  if [ "$v" = v3-pennyroyal ]; then
    "$ROOT/sglang/v3-pennyroyal/stop.sh"
  else
    "$ROOT/sglang/common/stop-docker.sh" "$@"
  fi
done
