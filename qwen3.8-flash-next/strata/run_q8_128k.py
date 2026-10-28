#!/usr/bin/env python3
"""Run the experimental Strata Q8 profile after the preparation in README.md.

Runs in the foreground, bound to localhost by default. Stop the other GPU server first.
"""
import argparse
import json
import os
from pathlib import Path
import re
import socket
import fcntl

from lifecycle import active, process, STATE

ROOT = Path(__file__).resolve().parents[2]


def main(quant="q8", description=None, context=131072, vision=False):
    ap = argparse.ArgumentParser(description=description or __doc__)
    ap.add_argument("--strata-dir", type=Path, default=ROOT.parent / "Strata")
    if quant == "q8":
        default_model = Path.home() / ".lmstudio/models/lmstudio-community/Qwen3.8-Flash-Next-GGUF/Qwen3.8-Flash-Next-Q8_0-00001-of-00006.gguf"
        pack_name = "q8_0"
    elif quant == "q6":
        default_model = ROOT / "models/Qwen3.8-Flash-Next-Q6_K-GGUF/Qwen3.8-Flash-Next-Q6_K-00001-of-00005.gguf"
        pack_name = "q6_k"
    elif quant == "q8-abliterated":
        default_model = ROOT / "models/Qwen3.8-Flash-Next-Abliterated-Q8_0-GGUF/Qwen3.8-Flash-Next-Abliterated-Q8_0.gguf"
        pack_name = "q8_0-abliterated"
    else:
        raise ValueError(f"unknown profile {quant}")
    ap.add_argument("--model", type=Path, default=default_model)
    ap.add_argument("--port", type=int, default=8090)
    ap.add_argument("--host", default="127.0.0.1", help="IPv4 bind address (use 0.0.0.0 for LAN access)")
    ap.add_argument("--context", type=int, default=context)
    ap.add_argument("--resident-gib", type=float, default=135)
    ap.add_argument("--prefill", default="auto")
    ap.add_argument("--vram-reserve-mib", type=int, default=1536)
    ap.add_argument("--pool-workers", type=int, default=2,
                    help="CPU expert pool threads (default 2: decode within a few percent of 7 at about half the "
                         "CPU power, see README.md); 0 = engine default. The workers spin while a request runs, "
                         "so each one keeps a core busy")
    ap.add_argument("--pool-affinity", choices=("all", "auto", "p-cores"), default="p-cores",
                    help="CPU pool placement on hybrid CPUs (default p-cores: P-cores minus the host's; "
                         "ignored on CPUs without E-cores)")
    if vision:
        ap.add_argument("--mmproj", type=Path, default=Path.home() / ".lmstudio/models/lmstudio-community/Qwen3.8-Flash-Next-GGUF/mmproj-Qwen3.8-Flash-Next-BF16.gguf")
        ap.add_argument("--vision-exe", type=Path, help="default: STRATA_DIR/build-vision/bin/strata-vision")
        ap.add_argument("--vision-device", choices=("gpu", "cpu"), default="gpu")
        ap.add_argument("--vision-tokens", type=int, default=4096, help="maximum context tokens per image (default: 4096)")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()
    if args.pool_workers < 0:
        ap.error("--pool-workers must be 0 (engine default) or positive")
    if args.context not in (131072, 262144):
        ap.error("supported profiles have 131072 or 262144 tokens")
    strata = args.strata_dir.resolve()
    model = args.model.resolve()
    pack = strata / "packs" / pack_name
    vision_config = None
    if vision:
        if args.vision_tokens <= 0:
            ap.error("--vision-tokens must be positive")
        vision_exe = (args.vision_exe or strata / "build-vision/bin/strata-vision").resolve()
        mmproj = args.mmproj.resolve()
        for path in (vision_exe, mmproj):
            if not path.is_file():
                ap.error(f"missing {path}; follow README.md's vision preparation")
        vision_config = dict(exe=str(vision_exe), mmproj=str(mmproj), model=str(model),
                             gpu=args.vision_device == "gpu", max_tokens=args.vision_tokens,
                             threads=os.cpu_count() or 1)
    for path in (strata / "build/strata", strata / ".venv/bin/python", model,
                 pack / "native_experts.txt", pack / "tokenizer", strata / "mtp/rt/draft_vocab.bin"):
        if not path.exists():
            ap.error(f"missing {path}; follow README.md first")
    split = re.fullmatch(r"(.*)-00001-of-(\d+)\.gguf", model.name)
    if split:
        prefix, count = split.groups()
        for shard in range(1, int(count) + 1):
            path = model.parent / f"{prefix}-{shard:05d}-of-{count}.gguf"
            if not path.exists():
                ap.error(f"missing shard {path}")
    logs = ROOT / "logs/strata-comparison"
    suffix = ("-vision" if vision else "") + ("-256k" if args.context == 262144 else "")   # log and config names
    profile = f"qwen3.8-flash-next-strata-{quant}{'-vision' if vision else ''}-{args.context // 1024}k"
    config = dict(
        exe=str(strata / "build/strata"), cwd=str(strata),
        args=["--pack", str(pack), "--native", str(model),
              "--resident-budget-gib", str(args.resident_gib),
              "--expert-profile", str(strata / "data/expert-profile.bin"),
              "--expert-cache", "auto", "--prefill", args.prefill,
              "--vram-reserve-mib", str(args.vram_reserve_mib),
              "--spec", "4", "--spec-min-p", "0.5", "--mtp", str(strata / "mtp/rt"),
              "--max-context", str(args.context), "--kv", "int8", "--ple-io", "mmap"],
        tokenizer=str(pack / "tokenizer"),
        model_name=profile,
        fit_max_tokens=True,
        log=str(ROOT / f"logs/strata-{quant}{suffix}-engine.log"), host=args.host, port=args.port)
    config["args"] += ["--pool-affinity", args.pool_affinity]
    if args.pool_workers:
        config["args"] += ["--pool-workers", str(args.pool_workers)]
    if vision_config:
        config["args"].append("--vision")
        config["vision"] = vision_config
    if args.dry_run:
        print(json.dumps(config, indent=2))
        return
    logs.mkdir(parents=True, exist_ok=True)
    target = logs / f"strata-{quant}{suffix}-{args.port}.json"
    python = str(strata / ".venv/bin/python")
    with (STATE.parent / "strata-server.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        running = active()
        if running:
            ap.error(f"already running: {running['profile']}; stop it with scripts/stop.sh")
        with socket.socket() as sock:
            # Match HTTPServer's reuse policy: closed client connections can
            # leave TIME_WAIT sockets after a profile has already stopped.
            sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            try:
                sock.bind((args.host, args.port))
            except OSError:
                ap.error(f"port {args.port} is occupied; stop the existing server first")
        target.write_text(json.dumps(config, indent=2) + "\n")
        STATE.write_text(json.dumps(dict(pid=os.getpid(), start_ticks=process(os.getpid())[1],
                         profile=profile, host=args.host, port=args.port, context=args.context, kv="int8",
                         prefill=args.prefill, model=str(model), log=config['log'], vision=vision_config,
                         engine=str(strata / "build/strata"), config=str(target)), indent=2) + "\n")
    # A crashed engine holds ~200 GiB of host memory, much of it mapped by the
    # NVIDIA driver. Dumping it can block the process in the kernel (state D)
    # for hours with the VRAM still held, where even SIGKILL cannot end it.
    # Dump no memory mappings; fork and exec inherit the filter.
    Path("/proc/self/coredump_filter").write_text("0")
    # Strata 0.1.42+ skips low-ranked missed experts by default on CUDA; its own
    # measurements show the largest drift on non-English text. Compute every
    # expert unless the caller set the variable (see NON-ENGLISH.md).
    os.environ.setdefault("STRATA_ROUTE_TAIL_SKIP", "0")
    os.chdir(strata)
    os.execv(python, [python, "-m", "serve.server", "--engine", "strata",
                     "--config", str(target), "--host", args.host, "--port", str(args.port)])


if __name__ == "__main__":
    main()
