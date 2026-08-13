#!/usr/bin/env python3
"""Replay a dynamic .raw recording and visualize event-rate gating per window.

Green: pixels surviving the count gate.
Red: pixels with events removed by the count gate.

No LED-region boxes, coordinates, or region arguments are used.

Usage:
    python3 step2d_visualize_gating_dynamic_no_boxes.py --input raw_dynamic.raw \
        --detection-window-us 20000 --out-dir gating_frames
"""
from __future__ import annotations

import argparse
import csv
import math
import os
import sys

import cv2
import numpy as np
from metavision_core.event_io import EventsIterator

FREQUENCIES_HZ = (165.0, 365.0, 596.0)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="Path to a .raw recording")
    parser.add_argument("--detection-window-us", type=int, default=20_000)
    parser.add_argument("--count-scale", type=float, default=1.0)
    parser.add_argument("--polarity", choices=("on", "off", "both"), default="on")
    parser.add_argument("--out-dir", default="gating_frames")
    parser.add_argument("--max-frames", type=int, default=200)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.detection_window_us <= 0 or args.count_scale <= 0:
        raise ValueError("detection-window-us and count-scale must be positive")

    os.makedirs(args.out_dir, exist_ok=True)
    f_min = min(FREQUENCIES_HZ)
    threshold = max(1, int(math.ceil(
        args.detection_window_us * f_min * args.count_scale / 1_000_000.0
    )))

    stream = EventsIterator(
        input_path=args.input,
        mode="delta_t",
        delta_t=args.detection_window_us,
        relative_timestamps=False,
    )
    height, width = stream.get_size()
    print(f"sensor={width}x{height}; Td={args.detection_window_us} us; "
          f"count threshold={threshold}; polarity={args.polarity}")

    rows = []
    frame_idx = 0
    for events in stream:
        if args.polarity != "both" and events.size:
            polarity = 1 if args.polarity == "on" else 0
            events = events[events["p"] == polarity]

        count_map = np.zeros((height, width), dtype=np.uint16)
        if events.size:
            np.add.at(count_map, (events["y"], events["x"]), 1)

        survived = count_map >= threshold
        removed = (count_map > 0) & ~survived
        rows.append({
            "frame": frame_idx,
            "events_after_polarity_selection": int(events.size),
            "pixels_survived_gate": int(survived.sum()),
            "pixels_removed_by_gate": int(removed.sum()),
        })

        if frame_idx < args.max_frames:
            overlay = np.zeros((height, width, 3), dtype=np.uint8)
            overlay[removed] = (0, 0, 255)
            overlay[survived] = (0, 255, 0)
            cv2.imwrite(os.path.join(args.out_dir, f"frame_{frame_idx:04d}.png"), overlay)

        frame_idx += 1

    csv_path = os.path.join(args.out_dir, "gating_log.csv")
    with open(csv_path, "w", newline="") as out_file:
        writer = csv.DictWriter(out_file, fieldnames=rows[0].keys() if rows else [])
        if rows:
            writer.writeheader()
            writer.writerows(rows)

    print(f"Processed {frame_idx} windows.")
    print(f"Saved overlays to: {args.out_dir}/")
    print(f"Saved metrics to: {csv_path}")
    print("Overlay colors: green=survived gate; red=removed by gate.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
