#!/usr/bin/env python3
"""Step 1c: Self-relocating refractory check.

Instead of trusting fixed ROI pixel coordinates across the whole sweep
(which broke when the rig moved), this version re-detects the hottest
pixel cluster inside a SEARCH window around each expected location at
EVERY bias_refr setting, then reports that cluster's live event count.
This is robust to a few pixels of drift, though NOT to large rig movement
-- keep the camera and LEDs physically fixed for the whole run.

Usage:
    python3 step1c_check_refractory_v2.py --serial 00050946 \
        --search 740,225,60 898,411,60 875,420,60 \
        --seconds 1.0
Each --search is "x,y,half_size": a generous window to search within.
"""
from __future__ import annotations
import argparse
import sys
import time

import numpy as np
from metavision_core.event_io import EventsIterator


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--serial", default="00050946")
    p.add_argument("--seconds", type=float, default=1.0)
    p.add_argument("--diff-off", type=int, default=190)
    p.add_argument("--diff-on", type=int, default=140)
    p.add_argument("--search", nargs="+", required=True)
    p.add_argument("--refr-candidates", nargs="+", type=int,
                    default=[-20, 0, 20, 55, 90, 140, 190, 235])
    return p.parse_args()


def parse_windows(args_list):
    out = []
    for spec in args_list:
        x, y, half = (int(v) for v in spec.split(","))
        out.append((x, y, half))
    return out


def main() -> int:
    args = parse_args()
    windows = parse_windows(args.search)

    stream = EventsIterator(
        input_path=args.serial, mode="delta_t", delta_t=20_000,
        relative_timestamps=False,
    )
    height, width = stream.get_size()
    device = stream.reader.device
    biases = device.get_i_ll_biases()

    for name, val in (("bias_diff_off", args.diff_off), ("bias_diff_on", args.diff_on)):
        try:
            biases.set(name, val)
        except Exception as exc:
            print(f"FAILED set {name}={val}: {exc}")

    event_iter = iter(stream)

    for refr in args.refr_candidates:
        try:
            biases.set("bias_refr", refr)
        except Exception as exc:
            print(f"bias_refr={refr:4d} -> INVALID: {exc}")
            continue
        time.sleep(0.3)

        count_map = np.zeros((height, width), dtype=np.uint32)
        t0 = time.monotonic()
        while time.monotonic() - t0 < args.seconds:
            try:
                events = next(event_iter)
            except StopIteration:
                break
            if events is None or not events.size:
                continue
            np.add.at(count_map, (events["y"], events["x"]), 1)

        line = [f"bias_refr={refr:4d}"]
        for i, (x, y, half) in enumerate(windows):
            y0, y1 = max(0, y - half), min(height, y + half + 1)
            x0, x1 = max(0, x - half), min(width, x + half + 1)
            sub = count_map[y0:y1, x0:x1]
            peak = int(sub.max())
            py, px = np.unravel_index(int(sub.argmax()), sub.shape)
            abs_x, abs_y = x0 + px, y0 + py
            total_in_window = int(sub.sum())
            line.append(
                f"W{i} search=({x},{y},r{half}) peak={peak:5d} at "
                f"({abs_x},{abs_y}) window_total={total_in_window:6d}"
            )
        print("  ".join(line))

    try:
        biases.set("bias_refr", 0)
    except Exception:
        pass

    print("\nCheck 'peak' and 'at (x,y)' per window across refr values.")
    print("If the peak LOCATION jumps around a lot, the rig is still moving "
          "or that window has no real LED -- widen --search or re-locate.")
    print("If peak count for one window collapses sharply at HIGH refr while "
          "others stay stable, that window's LED period is being clipped.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
