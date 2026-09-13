#!/usr/bin/env bash
# Stop this Docker variant and wait for its VRAM to be released.
#   ./stop.sh          stop the container
#   ./stop.sh --rm     stop and remove it
# All Docker variants run as the container "flashnext"; see ../common/stop-docker.sh.
exec "$(cd "$(dirname "$0")" && pwd)/../common/stop-docker.sh" "$@"
