#!/usr/bin/env python3
"""Keep the routed experts of the CPU-resident layers in RAM while a server runs.

    python3 -I mlock_experts.py FIRST_SHARD.gguf N_CPU_LAYERS [--dry-run]

llama.cpp and ik_llama.cpp map the GGUF with mmap. The experts of the first
N_CPU_LAYERS layers (the --n-cpu-moe count) are read from that mapping for every
generated token. DeepSeek V4.1's Engram table is read from the same files at
random, and the page cache cannot hold both: the kernel's LRU evicts expert pages
and the server re-reads them from disk. This maps the same file ranges and
mlock()s them. Page-cache pages are shared between mappings of a file, so the
server's reads hit locked pages while the Engram stays evictable.

Needs an unlimited (or large enough) RLIMIT_MEMLOCK. Runs until killed; killing
it unlocks the pages. Standard library only.
"""
import ctypes
import mmap
import os
import pathlib
import re
import signal
import struct
import sys
import time

# Value sizes of GGUF metadata types (8 = string, 9 = array are variable).
FIXED = {0: 1, 1: 1, 2: 2, 3: 2, 4: 4, 5: 4, 6: 4, 7: 1, 10: 8, 11: 8, 12: 8}


def read_tensors(path):
    """Return (data_start, [(name, offset)]) from a GGUF v3 header."""
    with open(path, "rb") as f:
        def u32():
            return struct.unpack("<I", f.read(4))[0]

        def u64():
            return struct.unpack("<Q", f.read(8))[0]

        def string():
            return f.read(u64()).decode("utf-8", "replace")

        def skip(t):
            if t == 8:
                f.seek(u64(), 1)
            elif t == 9:
                et, n = u32(), u64()
                if et == 8:
                    for _ in range(n):
                        f.seek(u64(), 1)
                else:
                    f.seek(FIXED[et] * n, 1)
            else:
                f.seek(FIXED[t], 1)

        if f.read(4) != b"GGUF" or u32() != 3:
            sys.exit(f"{path}: not a GGUF v3 file")
        n_tensors, n_kv = u64(), u64()
        align = 32
        for _ in range(n_kv):
            key, t = string(), u32()
            if key == "general.alignment" and t == 4:
                align = u32()
            else:
                skip(t)
        tensors = []
        for _ in range(n_tensors):
            name = string()
            f.seek(8 * u32(), 1)            # dimensions
            f.seek(4, 1)                    # type
            tensors.append((name, u64()))
        pos = f.tell()
    return (pos + align - 1) // align * align, tensors


def main():
    first = pathlib.Path(sys.argv[1])
    n_cpu = int(sys.argv[2])
    m = re.fullmatch(r"(.*)-00001-of-(\d+)\.gguf", first.name)
    shards = ([first.parent / f"{m.group(1)}-{i:05d}-of-{m.group(2)}.gguf" for i in range(1, int(m.group(2)) + 1)]
              if m else [first])
    ranges = []
    for p in shards:
        start, ts = read_tensors(p)
        ts.sort(key=lambda t: t[1])
        end = p.stat().st_size - start
        for (name, off), nxt in zip(ts, ts[1:] + [(None, end)]):
            mm = re.match(r"blk\.(\d+)\.ffn_(gate|up|down)_exps\.", name)
            if mm and int(mm.group(1)) < n_cpu:
                ranges.append((p, start + off, nxt[1] - off))
    total = sum(r[2] for r in ranges)
    print(f"{len(ranges)} expert tensors in layers 0..{n_cpu - 1}, {total / 1e9:.1f} GB to lock", flush=True)
    if "--dry-run" in sys.argv:
        return

    libc = ctypes.CDLL("libc.so.6", use_errno=True)
    libc.mmap.restype = ctypes.c_void_p
    libc.mmap.argtypes = [ctypes.c_void_p, ctypes.c_size_t, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_long]
    libc.mlock.argtypes = [ctypes.c_void_p, ctypes.c_size_t]
    locked = 0
    for p, start, length in ranges:
        fd = os.open(p, os.O_RDONLY)
        a0 = start // mmap.PAGESIZE * mmap.PAGESIZE
        ln = start + length - a0
        addr = libc.mmap(None, ln, mmap.PROT_READ, mmap.MAP_SHARED, fd, a0)
        os.close(fd)
        if addr in (None, ctypes.c_void_p(-1).value):
            sys.exit(f"mmap failed for {p.name}: errno {ctypes.get_errno()}")
        if libc.mlock(addr, ln) != 0:
            sys.exit(f"mlock failed after {locked / 1e9:.1f} GB (errno {ctypes.get_errno()}); "
                     "check ulimit -l and free RAM")
        locked += ln
    print(f"locked {locked / 1e9:.1f} GB; holding", flush=True)
    signal.signal(signal.SIGTERM, lambda *_: sys.exit(0))
    while True:
        time.sleep(3600)


if __name__ == "__main__":
    main()
