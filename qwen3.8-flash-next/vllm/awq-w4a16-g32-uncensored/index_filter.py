#!/usr/bin/env python3
"""Give vLLM a view of a checkpoint that contains only the tensors its index lists.

    python3 index_filter.py MODEL_DIR        # prints the directory to serve

leoncca/Qwen3.8-Flash-Next-Uncensored-AWQ-g32 reuses complete shard files from the
official FP8 checkpoint for its PLE table. Two of them also hold tensors that
model.safetensors.index.json does not list — FP8 expert weights and duplicate PLE
buffers, 1.0 GiB in all. The checkpoint's loader is expected to ignore them; vLLM
reads every tensor in every file and fails on the first one
("has no parameter 'w2_weight' for checkpoint weight ...down_proj.weight").

This builds MODEL_DIR.indexed/ next to the checkpoint: every file hard-linked
(no extra space; same filesystem), except safetensors files with unlisted tensors,
which are rewritten with only the listed ones. The downloaded checkpoint is not
modified and still verifies against its SHA256SUMS. Re-running is a no-op while
the index is unchanged. Standard library only.
"""
import hashlib
import json
import os
import shutil
import struct
import sys


def read_header(path):
    with open(path, "rb") as fh:
        n = struct.unpack("<Q", fh.read(8))[0]
        return json.loads(fh.read(n)), 8 + n


def write_filtered(src, dst, keep):
    header, data_start = read_header(src)
    names = sorted((k for k in header if k != "__metadata__" and k in keep),
                   key=lambda k: header[k]["data_offsets"][0])
    new_header, offset = {}, 0
    if "__metadata__" in header:
        new_header["__metadata__"] = header["__metadata__"]
    for k in names:
        begin, end = header[k]["data_offsets"]
        new_header[k] = dict(header[k], data_offsets=[offset, offset + end - begin])
        offset += end - begin
    blob = json.dumps(new_header, separators=(",", ":")).encode()
    blob += b" " * (-len(blob) % 8)                    # safetensors aligns the header to 8 bytes
    tmp = dst + ".tmp"
    with open(src, "rb") as fin, open(tmp, "wb") as fout:
        fout.write(struct.pack("<Q", len(blob)))
        fout.write(blob)
        for k in names:
            begin, end = header[k]["data_offsets"]
            fin.seek(data_start + begin)
            remaining = end - begin
            while remaining:
                chunk = fin.read(min(remaining, 64 << 20))
                fout.write(chunk)
                remaining -= len(chunk)
    os.replace(tmp, dst)
    return len(header) - (1 if "__metadata__" in header else 0) - len(names)


def main():
    src = os.path.normpath(sys.argv[1])
    dst = src + ".indexed"
    index_path = os.path.join(src, "model.safetensors.index.json")
    index_bytes = open(index_path, "rb").read()
    stamp = hashlib.sha256(index_bytes).hexdigest()
    marker = os.path.join(dst, ".index-filter")
    if os.path.isfile(marker) and open(marker).read().strip() == stamp:
        print(dst)
        return 0

    weight_map = json.loads(index_bytes)["weight_map"]
    by_file = {}
    for name, fname in weight_map.items():
        by_file.setdefault(fname, set()).add(name)

    os.makedirs(dst, exist_ok=True)
    dropped_total = 0
    for root, _, files in os.walk(src):
        rel = os.path.relpath(root, src)
        if rel.startswith(".cache") or "/.cache" in rel:
            continue
        os.makedirs(os.path.join(dst, rel), exist_ok=True)
        for f in files:
            s, d = os.path.join(root, f), os.path.normpath(os.path.join(dst, rel, f))
            relname = os.path.normpath(os.path.join(rel, f))
            if os.path.lexists(d):
                os.remove(d)
            if f.endswith(".safetensors") and relname in by_file:
                header, _ = read_header(s)
                listed = by_file[relname]
                if any(k != "__metadata__" and k not in listed for k in header):
                    dropped = write_filtered(s, d, listed)
                    dropped_total += dropped
                    print(f"  {relname}: dropped {dropped} unlisted tensors", file=sys.stderr)
                    continue
            try:
                os.link(s, d)
            except OSError:
                shutil.copy2(s, d)
    with open(marker, "w") as fh:
        fh.write(stamp + "\n")
    print(f"  {dst}: {dropped_total} unlisted tensors dropped in total", file=sys.stderr)
    print(dst)
    return 0


if __name__ == "__main__":
    sys.exit(main())
