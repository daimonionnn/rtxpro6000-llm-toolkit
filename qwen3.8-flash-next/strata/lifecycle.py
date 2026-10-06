#!/usr/bin/env python3
"""Linux process tracking for foreground Strata launchers and toolkit scripts."""
import argparse
import fcntl
import json
import os
from pathlib import Path
import signal
import time

ROOT = Path(__file__).resolve().parents[2]
STATE = ROOT / "logs/strata-server.json"


def process(pid):
    try:
        # comm may contain spaces or parentheses; fields after its final ')' start at field 3.
        fields = Path(f"/proc/{pid}/stat").read_text().rsplit(")", 1)[1].split()
        return fields[0], int(fields[19]), int(fields[1])
    except (OSError, ValueError, IndexError):
        return None


def alive(pid, ticks):
    p = process(pid)
    return bool(p and p[0] != "Z" and p[1] == ticks)


def active():
    try:
        state = json.loads(STATE.read_text())
        return state if alive(state["pid"], state["start_ticks"]) else None
    except (OSError, ValueError, KeyError, TypeError):
        return None


def stop(timeout):
    STATE.parent.mkdir(parents=True, exist_ok=True)
    with (STATE.parent / "strata-server.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        _stop(timeout)


def _stop(timeout):
    state = active()
    if not state:
        STATE.unlink(missing_ok=True)
        print("No running Strata server")
        return
    pid = state["pid"]
    targets = [(pid, state["start_ticks"])]
    for p in Path("/proc").iterdir():
        if not p.name.isdigit():
            continue
        info = process(int(p.name))
        try:
            argv = (p / "cmdline").read_bytes().split(b"\0")
        except OSError:
            continue
        if info and info[2] == pid and argv[0] == os.fsencode(state["engine"]):
            targets.append((int(p.name), info[1]))
    print(f"Stopping {state['profile']} (PID {pid})")
    try:
        os.kill(pid, signal.SIGTERM)
    except ProcessLookupError:
        pass
    deadline = time.monotonic() + timeout
    while any(alive(p, ticks) for p, ticks in targets) and time.monotonic() < deadline:
        time.sleep(.2)
    for p, ticks in reversed(targets):
        if alive(p, ticks):
            try:
                os.kill(p, signal.SIGKILL)
            except ProcessLookupError:
                pass
    deadline = time.monotonic() + 10
    while any(alive(p, ticks) for p, ticks in targets) and time.monotonic() < deadline:
        time.sleep(.2)
    if any(alive(p, ticks) for p, ticks in targets):
        raise RuntimeError("Strata process did not exit; keeping its state")
    try:
        current = json.loads(STATE.read_text())
        if (current.get("pid"), current.get("start_ticks")) == (pid, state["start_ticks"]):
            STATE.unlink()
    except (OSError, ValueError):
        pass


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("command", choices=("running", "status", "stop"))
    ap.add_argument("--timeout", type=float, default=60)
    args = ap.parse_args()
    if args.command == "stop":
        stop(args.timeout)
    else:
        state = active()
        if state:
            print(state["profile"] if args.command == "running" else json.dumps(state))
