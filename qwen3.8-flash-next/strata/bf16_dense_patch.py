#!/usr/bin/env python3
"""Replace a Strata pack's rounded BF16 tensors with the model's original BF16 bytes.

`iq_pack.py --compat-bf16` dequantizes small projections that a Q8_0 GGUF stores as
Q8_0 (hyper-connections, ssm_alpha/beta, ple_value) and rounds them to BF16, the form
the engine reads. The model was trained in BF16, so those tensors exist exactly in a
BF16 GGUF of the same checkpoint. This tool copies them from there into dense.bin:

    ../Strata/.venv/bin/python qwen3.8-flash-next/strata/bf16_dense_patch.py \
        --pack ../Strata/packs/q8_0-abliterated --source <BF16 .gguf path or https URL>

Only tensors that conversions.json lists as rounded from a quantized type are
replaced. Each must be BF16 with the same element count in the source. Before
writing, every tensor is compared with the current values: the difference must look
like Q8_0 quantization error (relative RMS below --max-rel-rms), which also proves
that names, shapes and element order match. A URL is read with HTTP range requests,
so only the header and the selected tensors are downloaded.

dense.bin is written to a new file and renamed into place; the previous one is kept
as dense.bin.before-bf16 and conversions.json records each replacement.
"""
import argparse
import hashlib
import json
import pathlib
import sys
import tempfile
import time
import urllib.request

import numpy as np

QUANTIZED_SOURCE = {"Q8_0", "Q6_K", "Q5_K", "Q4_K", "Q5_0", "Q5_1", "Q4_0", "Q4_1", "F16"}


class Source:
    """A local file or an https URL read by byte ranges."""

    def __init__(self, where):
        self.where = where
        self.url = where if where.startswith("https://") else None

    def read(self, offset, length):
        if not self.url:
            with open(self.where, "rb") as f:
                f.seek(offset)
                data = f.read(length)
        else:
            for attempt in range(6):
                try:
                    req = urllib.request.Request(self.url, headers={"Range": f"bytes={offset}-{offset + length - 1}"})
                    with urllib.request.urlopen(req, timeout=300) as r:
                        data = r.read()
                    break
                except Exception:
                    if attempt == 5:
                        raise
                    time.sleep(2 ** attempt)
        if len(data) != length:
            sys.exit(f"short read at {offset}: {len(data)} of {length} bytes")
        return data


def bf16_to_f32(b):
    return (np.frombuffer(b, dtype="<u2").astype(np.uint32) << 16).view(np.float32)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--pack", type=pathlib.Path, required=True)
    ap.add_argument("--source", required=True, help="BF16 GGUF of the same checkpoint: path or https URL")
    ap.add_argument("--strata-dir", type=pathlib.Path,
                    default=pathlib.Path(__file__).resolve().parents[3] / "Strata")
    ap.add_argument("--header-bytes", type=int, default=64 << 20)
    ap.add_argument("--max-rel-rms", type=float, default=0.02,
                    help="refuse a tensor whose values differ more than this (relative RMS)")
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()

    sys.path.insert(0, str(a.strata_dir / "tools"))
    import gguf_reader as G

    pack = a.pack.resolve()
    conv_path = pack / "conversions.json"
    conv = json.loads(conv_path.read_text(encoding="utf-8"))
    targets = {t["name"]: t for t in conv["tensors"]
               if t.get("dst_type") == "BF16" and t.get("src_type") in QUANTIZED_SOURCE and not t.get("exact")}
    rows = {}
    for line in (pack / "index.txt").read_text(encoding="utf-8").splitlines():
        if line and not line.startswith("#"):
            f = line.split()
            rows[f[0]] = f
    src = Source(a.source)
    with tempfile.NamedTemporaryFile(suffix=".gguf") as tmp:
        tmp.write(src.read(0, a.header_bytes))
        tmp.flush()
        g = G.GGUFFile(pathlib.Path(tmp.name))
    tensors = {t.name: t for t in g.tensors}

    dense = (pack / "dense.bin").read_bytes()
    out = bytearray(dense)
    report, total = [], 0
    for name in sorted(targets):
        row, t = rows.get(name), tensors.get(name)
        if row is None or row[2] != "4":
            sys.exit(f"{name}: not a BF16 row in index.txt")
        if t is None or t.type_name != "BF16":
            sys.exit(f"{name}: missing or not BF16 in the source ({t.type_name if t else 'missing'})")
        off, nbytes = int(row[3]), int(row[4])
        if t.elements * 2 != nbytes:
            sys.exit(f"{name}: {t.elements} source elements, {nbytes // 2} in the pack")
        new = src.read(g.data_start + t.offset, nbytes)
        old_v, new_v = bf16_to_f32(dense[off:off + nbytes]), bf16_to_f32(new)
        rms = float(np.sqrt(np.mean(new_v.astype(np.float64) ** 2))) or 1.0
        rel = float(np.sqrt(np.mean((old_v.astype(np.float64) - new_v) ** 2))) / rms
        if not np.isfinite(new_v).all() or rel > a.max_rel_rms:
            sys.exit(f"{name}: relative RMS difference {rel:.4f} - not the same tensor; nothing written")
        out[off:off + nbytes] = new
        total += nbytes
        report.append((name, rel, float(np.max(np.abs(old_v.astype(np.float64) - new_v)))))
        targets[name].update(src_type="BF16", method=f"original BF16 bytes from {pathlib.PurePath(a.source).name} "
                             f"(bf16_dense_patch.py), replacing the Q8_0 dequantization", exact=True,
                             max_abs_err=0.0, source_sha256=hashlib.sha256(new).hexdigest(),
                             replaced_rel_rms=round(rel, 6))
    worst = max(report, key=lambda r: r[1])
    print(f"{len(report)} tensors, {total / 2**30:.2f} GiB from the BF16 source; relative RMS change "
          f"median {np.median([r[1] for r in report]):.4f}, largest {worst[1]:.4f} ({worst[0]}); "
          f"largest absolute change {max(r[2] for r in report):.4f}")
    if a.dry_run:
        print("dry run: nothing written")
        return
    tmp = pack / "dense.bin.tmp"
    tmp.write_bytes(out)
    if not (pack / "dense.bin.before-bf16").exists():
        (pack / "dense.bin").replace(pack / "dense.bin.before-bf16")
    tmp.replace(pack / "dense.bin")
    conv["bf16_source"] = a.source
    conv_path.write_text(json.dumps(conv, indent=1) + "\n", encoding="utf-8")
    print(f"wrote {pack / 'dense.bin'}; previous kept as dense.bin.before-bf16")


if __name__ == "__main__":
    main()
