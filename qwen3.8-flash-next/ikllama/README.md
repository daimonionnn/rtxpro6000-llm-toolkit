# Qwen3.8-Flash-Next on ik_llama.cpp (ik-llama-toolkit)

A thin integration of the external
[ik-llama-toolkit](https://github.com/daimonionnn/ik-llama-toolkit). That project
keeps its own build, placement (`-ncmoe`, `-ub`, threads), benchmarks and results;
this directory only gives three of its profiles an id in this toolkit's registry,
so `scripts/start-*.sh`, `scripts/stop.sh` and `scripts/status.sh` work for them
like for every other engine.

They load the same Q8_0 GGUF as the Strata Q8 profiles
(`~/.lmstudio/models/lmstudio-community/Qwen3.8-Flash-Next-GGUF/`), so they are the
reference for checking Strata's numerics: same weights, independent kernels.

| Profile id | ik-llama-toolkit profile | Notes |
|---|---|---|
| `qwen3.8-flash-next-ik-q8-vision-128k` | `qwen38-flash-next-q8-128k` | experts of 17 layers on the CPU |
| `qwen3.8-flash-next-ik-q8-vision-256k` | `qwen38-flash-next-q8-256k` | |
| `qwen3.8-flash-next-ik-q8-mtp-vision-128k` | `qwen38-flash-next-q8-128k-mtp` | MTP draft head (dzannotti Q4_K_M), 19 layers on the CPU; the ik-llama-toolkit default |

Speeds, MTP acceptance and the reasons behind each setting are in the profile files
`config/models/qwen38-flash-next-*.env` and `docs/RESULTS.md` of ik-llama-toolkit.
The Q4_K_M profiles there are not registered here because that checkpoint is not
downloaded.

## Pinned versions

| | Commit | |
|---|---|---|
| ik-llama-toolkit | `e7226e7` | 2026-10-31 |
| ik_llama.cpp | `13cffc00` | 2026-10-30, built by ik-llama-toolkit's `build.sh` |

`serve.sh` warns when either checkout is on a different commit; it still starts.
After updating ik-llama-toolkit, re-check a profile and move the pins in `serve.sh`
and this table.

## Setup

```bash
git clone https://github.com/daimonionnn/ik-llama-toolkit ~/development/ik-llama-toolkit
cd ~/development/ik-llama-toolkit
git checkout e7226e7
git clone https://github.com/ikawrakow/ik_llama.cpp.git ik_llama.cpp
git -C ik_llama.cpp checkout 13cffc00
./build.sh
```

Downloads (Q8_0 GGUF, mmproj, the MTP head) are listed in the ik-llama-toolkit
profile headers. `IK_TOOLKIT_DIR` points `serve.sh` at a checkout elsewhere.

## Running

```bash
scripts/start-qwen3.8-flash-next-ik-q8-vision-128k.sh
scripts/status.sh
scripts/stop.sh
```

`serve.sh` passes `HOST`, `PORT` and `CTX` to ik-llama-toolkit as `IK_HOST`,
`IK_PORT` and `IK_CTX`, sets the served model name to the profile id, and forbids
ik-llama-toolkit from evicting other GPU processes (`IK_KILL_SQUATTERS=0`). Any
other `IK_*` variable passes straight through, e.g. `IK_GDB=0` to run without the
gdb wrapper. ik-llama-toolkit keeps its own log under its `logs/`; the console
output also goes to `logs/<profile>-server.log` here.

Do not start ik-llama-toolkit's systemd unit (`ik-llama-server.service`) at the
same time: it uses the same port 8090 and the whole GPU.
