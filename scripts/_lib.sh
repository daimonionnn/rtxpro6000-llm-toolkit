# Shared by the scripts in this directory; source it, do not run it.
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CONTAINER=rtxpro6000-llm   # every Docker profile runs as this one container
NATIVE_PIDFILE="$ROOT/qwen3.8-flash-next/sglang/nvfp4-ram-pennyroyal/server.pid"

# The profile registry: id | directory | launcher | checkpoint (under models/) | description.
# Profile ids are <model>-<engine>-<checkpoint/placement>; the directory is
# <model>/<engine>/<checkpoint/placement>. A new profile needs a line here and a
# scripts/start-<id>.sh wrapper.
PROFILES=(
  "qwen3.8-flash-next-sglang-nvfp4-nvme|qwen3.8-flash-next/sglang/nvfp4-nvme|serve-nvfp4-nvme.sh|Qwen3.8-Flash-Next-NVFP4|Qwen3.8-Flash-Next · SGLang, local Docker image, NVFP4, PLE table streamed from NVMe"
  "qwen3.8-flash-next-sglang-nvfp4-ram|qwen3.8-flash-next/sglang/nvfp4-ram|serve-nvfp4-ram.sh|Qwen3.8-Flash-Next-NVFP4|Qwen3.8-Flash-Next · SGLang, local Docker image, NVFP4, PLE table in RAM"
  "qwen3.8-flash-next-sglang-nvfp4-ram-official|qwen3.8-flash-next/sglang/nvfp4-ram-official|serve-nvfp4-ram.sh|Qwen3.8-Flash-Next-NVFP4|Qwen3.8-Flash-Next · SGLang, official lmsysorg image, NVFP4, PLE table in RAM"
  "qwen3.8-flash-next-sglang-nvfp4-ram-official-abliterated|qwen3.8-flash-next/sglang/nvfp4-ram-official|serve-nvfp4-ram.sh|Qwen3.8-Flash-Next-ABLITERATED-NVFP4|Qwen3.8-Flash-Next · SGLang, official image, abliterated NVFP4 (dealignai), PLE table in RAM"
  "qwen3.8-flash-next-sglang-nvfp4-ram-pennyroyal|qwen3.8-flash-next/sglang/nvfp4-ram-pennyroyal|serve-nvfp4-ram.sh|Qwen3.8-Flash-Next-NVFP4|Qwen3.8-Flash-Next · SGLang pennyroyal fork (native), NVFP4, PLE table in RAM, 524K context"
  "qwen3.8-flash-next-vllm-awq-w4a16|qwen3.8-flash-next/vllm/awq-w4a16|serve-awq-w4a16-ram.sh|Qwen3.8-Flash-Next-AWQ-W4A16|Qwen3.8-Flash-Next · vLLM, official image, AWQ W4A16, PLE table in RAM"
  "qwen3.8-flash-next-vllm-awq-w4a16-g32|qwen3.8-flash-next/vllm/awq-w4a16-g32|serve-awq-w4a16-g32-ram.sh|Qwen3.8-Flash-Next-AWQ-INT4-g32|Qwen3.8-Flash-Next · vLLM, official image, AWQ W4A16 group 32 (cyankiwi), PLE table in RAM"
  "qwen3.8-flash-next-vllm-awq-w4a16-g32-uncensored|qwen3.8-flash-next/vllm/awq-w4a16-g32-uncensored|serve-awq-w4a16-g32-uncensored.sh|Qwen3.8-Flash-Next-Uncensored-AWQ-g32|Qwen3.8-Flash-Next · vLLM, preview image + FP8 PLE patch, uncensored AWQ W4A16 g32 (leoncca), PLE table in RAM"
  "qwen3.8-flash-next-exllamav3-exl3-5.05bpw|qwen3.8-flash-next/exllamav3/exl3-5.05bpw|serve-exl3-5.05bpw.sh|Qwen3.8-Flash-Next-EXL3-5.05bpw|Qwen3.8-Flash-Next · ExLlamaV3 + TabbyAPI, EXL3 5.05 bpw, n-gram table in RAM, MTP"
  "qwen3.8-flash-next-vllm-fp8-offload|qwen3.8-flash-next/vllm/fp8-offload|serve-fp8-offload.sh|Qwen3.8-Flash-Next-FP8|Qwen3.8-Flash-Next · vLLM, official image, official FP8, part of the experts and PLE table in RAM"
  "qwen3.6-27b-sglang-bf16|qwen3.6-27b/sglang/bf16|serve-bf16.sh|Qwen3.6-27B|Qwen3.6-27B · SGLang, official image, BF16, NEXTN speculation"
  "qwen3.8-27b-sglang-bf16|qwen3.8-27b/sglang/bf16|serve-bf16.sh|Qwen3.8-27B|Qwen3.8-27B · SGLang, official image, BF16, NEXTN speculation"
)

# profile_field ID FIELD — FIELD is dir, launcher, model or desc
profile_field() {
  local line id dir launcher model desc
  for line in "${PROFILES[@]}"; do
    IFS='|' read -r id dir launcher model desc <<<"$line"
    [ "$id" = "$1" ] || continue
    case "$2" in
      dir) echo "$dir" ;; launcher) echo "$launcher" ;; model) echo "$model" ;; desc) echo "$desc" ;;
    esac
    return 0
  done
  return 1
}

# Print the profile(s) currently serving, one per line: the native pennyroyal
# profile from its PID file, a Docker profile from the container's
# rtxpro6000-llm.profile label.
running_profiles() {
  if [ -f "$NATIVE_PIDFILE" ] && kill -0 "$(cat "$NATIVE_PIDFILE")" 2>/dev/null; then
    echo qwen3.8-flash-next-sglang-nvfp4-ram-pennyroyal
  fi
  if docker ps --format '{{.Names}}' 2>/dev/null | grep -qx "$CONTAINER"; then
    local p
    p=$(docker inspect "$CONTAINER" --format '{{index .Config.Labels "rtxpro6000-llm.profile"}}' 2>/dev/null)
    echo "${p:-unknown-docker-profile}"
  fi
}

# Refuse to start a second profile: they all need the whole GPU and port 8090.
require_idle() {
  local running
  running=$(running_profiles | tr '\n' ' ')
  if [ -n "$running" ]; then
    echo "already running: $running— stop it first with scripts/stop.sh" >&2
    exit 2
  fi
}

# start_profile ID — run a profile's launcher, with MODEL_DIR defaulting to the
# profile's checkpoint under models/. Launcher variables pass through the environment.
start_profile() {
  require_idle
  local dir launcher model
  dir=$(profile_field "$1" dir) || { echo "unknown profile $1" >&2; exit 2; }
  launcher=$(profile_field "$1" launcher); model=$(profile_field "$1" model)
  export MODEL_DIR="${MODEL_DIR:-$ROOT/models/$model}"
  export PROFILE_ID="$1"   # launchers shared by several profiles label the container with it
  exec "$ROOT/$dir/$launcher"
}
