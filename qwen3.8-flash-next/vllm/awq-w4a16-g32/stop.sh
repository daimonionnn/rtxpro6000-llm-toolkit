#!/usr/bin/env bash
# Stop this Docker profile and wait for its VRAM to be released.
#   ./stop.sh          stop the container
#   ./stop.sh --rm     stop and remove it
# All Docker profiles run as the container "rtxpro6000-llm"; see ../../../common/stop-docker.sh.
exec "$(cd "$(dirname "$0")" && pwd)/../../../common/stop-docker.sh" "$@"
