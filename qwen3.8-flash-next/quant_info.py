#!/usr/bin/env python3
"""Print what precision each part of a Qwen3.8-Flash-Next checkpoint has, and where it runs.

    python3 quant_info.py MODEL_DIR [--require nvfp4|w4a16|awq|fp8|exl3] [--kv-dtype D]
                          [--ple-mode nvme|ram] [--experts-ram-gib N] [--engine NAME]

Detects the checkpoint format from its own metadata:

  nvfp4   ModelOpt NVFP4 (hf_quant_config.json, quant_algo NVFP4) — routed experts
          FP4 W4A4, the rest BF16, PLE table FP8
  w4a16   compressed-tensors INT4 weight-only (config.json quantization_config) —
          routed experts INT4, activations and the rest BF16, PLE table BF16
  awq     AutoAWQ / GPTQModel INT4 (config.json quantization_config, quant_method awq)
          — routed experts INT4 with zero points, the rest as the checkpoint stores it
  fp8     block FP8 (config.json quantization_config, quant_method fp8) — routed
          experts FP8 W8A8 with dynamic activation scales, the rest BF16
  exl3    ExLlamaV3 trellis quantization (quant_method exl3) — every linear layer at
          the bit rate in its config; sizes only, since packed tensors do not
          reveal parameter counts

Parameter counts, bits per parameter and sizes are read from the safetensors
headers, not assumed. With --require, exits 2 if the checkpoint is a different
format — the launchers use that to refuse a mismatched model.

Standard library only.
"""
import argparse
import json
import os
import struct
import sys

DTYPE_BITS = {"F64": 64, "F32": 32, "BF16": 16, "F16": 16, "F8_E4M3": 8, "F8_E5M2": 8,
              "I64": 64, "I32": 32, "I16": 16, "I8": 8, "U8": 8, "BOOL": 8}
AUX_SUFFIXES = ("weight_scale", "weight_scale_2", "weight_scale_inv", "input_scale", "weight_zero_point",
                "qzeros", "scales",
                "weight_shape", "weight_global_scale", "k_scale", "v_scale")


def detect(model_dir):
    cfg = json.load(open(os.path.join(model_dir, "config.json")))
    hq = os.path.join(model_dir, "hf_quant_config.json")
    if os.path.isfile(hq):
        q = json.load(open(hq))
        if q.get("quantization", {}).get("quant_algo") == "NVFP4":
            return "nvfp4", cfg, q
    qc = cfg.get("quantization_config") or (cfg.get("text_config") or {}).get("quantization_config") or {}
    if qc.get("quant_method") in ("fp8", "exl3", "awq"):
        return qc["quant_method"], cfg, qc
    if qc.get("quant_method") == "compressed-tensors":
        for group in (qc.get("config_groups") or {}).values():
            w = group.get("weights") or {}
            if w.get("num_bits") == 4 and w.get("type") == "int" and not group.get("input_activations"):
                return "w4a16", cfg, qc
    return None, cfg, None


def headers(model_dir):
    """Yield (tensor_name, dtype, numel) for every tensor in the checkpoint."""
    for name in sorted(os.listdir(model_dir)):
        if not name.endswith(".safetensors"):
            continue
        with open(os.path.join(model_dir, name), "rb") as fh:
            n = struct.unpack("<Q", fh.read(8))[0]
            h = json.loads(fh.read(n))
        for key, v in h.items():
            if key == "__metadata__":
                continue
            numel = 1
            for d in v["shape"]:
                numel *= d
            yield key, v["dtype"], numel


def component(name):
    if name.startswith("mtp.") or ".mtp." in name:
        return "rest"                      # the MTP head keeps its own BF16 experts
    if ".mlp.experts." in name:
        return "experts"
    if ".ple." in name and "ngram_embedding" in name:
        return "ple"
    return "rest"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("model_dir")
    ap.add_argument("--require", choices=("nvfp4", "w4a16", "awq", "fp8", "exl3"))
    ap.add_argument("--kv-dtype", default="auto")
    ap.add_argument("--ple-mode", default="ram", choices=("nvme", "ram"))
    ap.add_argument("--experts-ram-gib", type=float, default=0,
                    help="GiB of routed experts the engine keeps in host RAM")
    ap.add_argument("--engine", default="")
    # accepted for compatibility with older launchers
    ap.add_argument("--weight-quant")
    ap.add_argument("--fp4-gemm-backend")
    a = ap.parse_args()

    if not os.path.isfile(os.path.join(a.model_dir, "config.json")):
        print(f"no config.json in {a.model_dir} — is MODEL_DIR a checkpoint directory?", file=sys.stderr)
        return 2
    fmt, cfg, qmeta = detect(a.model_dir)
    require = a.require or ("nvfp4" if a.weight_quant == "modelopt_fp4" else None)
    if require and fmt != require:
        print(f"quantization mismatch: checkpoint is {fmt or 'unrecognised'}, launcher needs {require}.",
              file=sys.stderr)
        return 2

    params = {"experts": 0, "ple": 0, "rest": 0}
    nbytes = {"experts": 0, "ple": 0, "rest": 0}
    dtypes = {"experts": set(), "ple": set(), "rest": set()}
    for name, dtype, numel in headers(a.model_dir):
        c = component(name)
        nbytes[c] += numel * DTYPE_BITS.get(dtype, 8) // 8
        if name.endswith(AUX_SUFFIXES) or dtype == "I64":
            continue
        if fmt == "nvfp4" and c == "experts" and dtype == "U8":
            params[c] += 2 * numel          # two FP4 values per byte
        elif fmt in ("w4a16", "awq") and c == "experts" and dtype == "I32":
            params[c] += 8 * numel          # eight INT4 values per int32
        else:
            params[c] += numel
        dtypes[c].add(dtype)

    if fmt == "nvfp4":
        q = qmeta["quantization"]
        label = f"NVFP4 W4A4, group {q.get('group_size')}, FP8 block scales"
        producer = qmeta.get("producer", {})
        origin = f"{producer.get('name', '?')} {producer.get('version', '?')}"
    elif fmt == "w4a16":
        w = next(iter(qmeta["config_groups"].values()))["weights"]
        label = f"INT4 W4A16, group {w.get('group_size')}, {'symmetric' if w.get('symmetric') else 'asymmetric'}"
        origin = f"compressed-tensors {qmeta.get('version', '')}".strip()
    elif fmt == "awq":
        label = f"INT4 AWQ W4A16, group {qmeta.get('group_size')}, {'zero point' if qmeta.get('zero_point') else 'symmetric'}"
        origin = f"awq {qmeta.get('version', '')}".strip()
    elif fmt == "fp8":
        block = qmeta.get("weight_block_size")
        label = f"FP8 W8A8, block {'x'.join(map(str, block)) if block else 'per-tensor'}, {qmeta.get('activation_scheme', '?')} act."
        origin = "fp8"
    elif fmt == "exl3":
        label = f"EXL3 {qmeta.get('bits')} bpw (head {qmeta.get('head_bits')} bpw)"
        origin = f"exllamav3 {qmeta.get('version', '?')}"
    else:
        label, origin = "unrecognised", "?"

    def dt(c):
        s = sorted(dtypes[c] - {"I64"})
        return "/".join(s) if s else "?"

    where_ple = {"nvme": "NVMe (streamed)", "ram": "host RAM"}[a.ple_mode]
    rows = [
        ("routed experts", label, "experts",
         f"VRAM + {a.experts_ram_gib:g} GiB host RAM" if a.experts_ram_gib else "VRAM"),
        ("attention, router, shared expert, MTP, vision, embeddings", dt("rest"), "rest", "VRAM"),
        ("PLE n-gram table", dt("ple"), "ple", where_ple),
    ]
    print(f"\nQwen3.8-Flash-Next — {os.path.basename(os.path.normpath(a.model_dir))}  ({origin})"
          + (f"  ·  engine {a.engine}" if a.engine else ""))
    print(f"  {'component':<58}{'precision':<40}{'params':>9}{'bits/p':>8}{'size':>11}  where")
    tot_p = tot_b = 0
    for title, prec, c, where in rows:
        p, b = params[c], nbytes[c]
        tot_p += p; tot_b += b
        if fmt == "exl3":
            prec = label if c == "experts" else "EXL3 (packed)" if c == "ple" else "EXL3 / BF16"
            print(f"  {title:<58}{prec:<40}{'—':>9}{'—':>8}{b/2**30:>7.1f} GiB  {where}")
            continue
        bits = b * 8 / p if p else 0
        print(f"  {title:<58}{prec:<40}{p/1e9:>8.1f}B{bits:>8.2f}{b/2**30:>7.1f} GiB  {where}")
    if fmt == "exl3":
        print(f"  {'model total':<58}{'':<40}{'180.0B':>9}{tot_b*8/180.0e9:>8.2f}{tot_b/2**30:>7.1f} GiB")
    else:
        print(f"  {'model total':<58}{'':<40}{tot_p/1e9:>8.1f}B{tot_b*8/tot_p:>8.2f}{tot_b/2**30:>7.1f} GiB")
    kv = a.kv_dtype if a.kv_dtype != "auto" else "auto (BF16)"
    print(f"  {'KV cache':<58}{kv}\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
