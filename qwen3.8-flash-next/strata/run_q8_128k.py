#!/usr/bin/env python3
"""Run the experimental Strata Q8 profile after the preparation in README.md.

Runs in the foreground, bound to localhost. Stop the other GPU server first.
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


def main(quant="q8", description=None, context=131072):
    ap = argparse.ArgumentParser(description=description or __doc__)
    ap.add_argument("--strata-dir", type=Path, default=ROOT.parent / "Strata")
    if quant == "q8":
        default_model = Path.home() / ".lmstudio/models/lmstudio-community/Qwen3.8-Flash-Next-GGUF/Qwen3.8-Flash-Next-Q8_0-00001-of-00006.gguf"
        pack_name = "q8_0"
    elif quant == "q6":
        default_model = ROOT / "models/Qwen3.8-Flash-Next-Q6_K-GGUF/Qwen3.8-Flash-Next-Q6_K-00001-of-00005.gguf"
        pack_name = "q6_k"
    else:
        raise ValueError(f"unknown profile {quant}")
    ap.add_argument("--model", type=Path, default=default_model)
    ap.add_argument("--port", type=int, default=8090)
    ap.add_argument("--context", type=int, default=context)
    ap.add_argument("--resident-gib", type=float, default=135)
    ap.add_argument("--prefill", default="auto")
    ap.add_argument("--vram-reserve-mib", type=int, default=1536)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()
    if args.context not in (131072, 262144):
        ap.error("supported profiles have 131072 or 262144 tokens")
    strata = args.strata_dir.resolve()
    model = args.model.resolve()
    pack = strata / "packs" / pack_name
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
    suffix = "-256k" if args.context == 262144 else ""
    profile = f"qwen3.8-flash-next-strata-{quant}{suffix}"
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
        log=str(ROOT / f"logs/strata-{quant}{suffix}-engine.log"), host="127.0.0.1", port=args.port)
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
            try:
                sock.bind(("127.0.0.1", args.port))
            except OSError:
                ap.error(f"port {args.port} is occupied; stop the existing server first")
        target.write_text(json.dumps(config, indent=2) + "\n")
        STATE.write_text(json.dumps(dict(pid=os.getpid(), start_ticks=process(os.getpid())[1],
                         profile=profile, port=args.port, context=args.context, kv="int8",
                         prefill=args.prefill, model=str(model), log=config['log'],
                         engine=str(strata / "build/strata"), config=str(target)), indent=2) + "\n")
    os.chdir(strata)
    os.execv(python, [python, "-m", "serve.server", "--engine", "strata",
                     "--config", str(target), "--port", str(args.port)])


if __name__ == "__main__":
    main()
