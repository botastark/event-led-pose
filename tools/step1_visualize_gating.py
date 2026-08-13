#!/usr/bin/env python3
"""Step 2b: Visualize what the event-rate gating step removes, replayed on
a saved .raw recording (no live camera needed).

For a chosen detection window Td, this reproduces the exact gating logic
from detect_leds_alm_style.py:
    pixel_threshold = ceil(Td_seconds * f_min * count_scale)
and saves three images:
  1. raw_count_heatmap.png     - all accumulated event counts (log-scaled)
  2. gated_mask.png            - binary mask of pixels that SURVIVED gating
  3. removed_overlay.png       - red = removed by gating, green = survived

Usage:
    python3 step2b_visualize_gating.py --input raw_tuned.raw \
        --detection-window-us 20000 --count-scale 1.0
"""
from __future__ import annotations
import argparse
import math
import sys

import numpy as np
import cv2
from metavision_core.event_io import EventsIterator

FREQUENCIES_HZ = (165.0, 365.0, 596.0)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--input", required=True, help="Path to .raw file.")
    p.add_argument("--detection-window-us", type=int, default=20_000)
    p.add_argument("--count-scale", type=float, default=1.0)
    p.add_argument("--polarity", choices=("on", "off", "both"), default="on")
    p.add_argument("--out-prefix", default="gating_")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    f_min = min(FREQUENCIES_HZ)

    stream = EventsIterator(
        input_path=args.input, mode="delta_t",
        delta_t=args.detection_window_us, relative_timestamps=False,
    )
    height, width = stream.get_size()

    pixel_threshold = max(1, int(math.ceil(
        (args.detection_window_us / 1_000_000.0) * f_min * args.count_scale
    )))
    print(f"sensor={width}x{height}  Td={args.detection_window_us}us  "
          f"f_min={f_min}Hz  pixel_threshold={pixel_threshold}")

    count_map = np.zeros((height, width), dtype=np.uint32)
    total_events = 0
    for events in stream:
        if not events.size:
            continue
        if args.polarity != "both":
            target = 1 if args.polarity == "on" else 0
            events = events[events["p"] == target]
        if events.size:
            np.add.at(count_map, (events["y"], events["x"]), 1)
            total_events += int(events.size)

    print(f"Total events processed: {total_events:,}")
    print(f"Max pixel count: {count_map.max()}")
    print(f"Pixels with count >= threshold ({pixel_threshold}): "
          f"{int(np.count_nonzero(count_map >= pixel_threshold))}")
    print(f"Pixels with count > 0 but < threshold (REMOVED by gating): "
          f"{int(np.count_nonzero((count_map > 0) & (count_map < pixel_threshold)))}")

    # 1. Raw count heatmap (log scale)
    heat = np.log1p(count_map.astype(np.float32))
    heat = (heat / max(heat.max(), 1e-6) * 255.0).astype(np.uint8)
    heat_color = cv2.applyColorMap(heat, cv2.COLORMAP_JET)
    cv2.imwrite(f"{args.out_prefix}raw_count_heatmap.png", heat_color)

    # 2. Binary mask: survived gating
    survived = (count_map >= pixel_threshold).astype(np.uint8) * 255
    cv2.imwrite(f"{args.out_prefix}gated_mask.png", survived)

    # 3. Overlay: green = survived, red = removed (had events but below threshold)
    overlay = np.zeros((height, width, 3), dtype=np.uint8)
    removed_mask = (count_map > 0) & (count_map < pixel_threshold)
    survived_mask = count_map >= pixel_threshold
    overlay[removed_mask] = (0, 0, 255)    # BGR red
    overlay[survived_mask] = (0, 255, 0)   # BGR green
    cv2.imwrite(f"{args.out_prefix}removed_overlay.png", overlay)

    print(f"\nSaved:")
    print(f"  {args.out_prefix}raw_count_heatmap.png  (all activity, log-scaled)")
    print(f"  {args.out_prefix}gated_mask.png          (white = passed threshold)")
    print(f"  {args.out_prefix}removed_overlay.png     (green=kept, red=discarded)")
    print("\nInspect removed_overlay.png: if your 3 LED blobs show partial red "
          "edges/cores, the pixel_threshold or Td is discarding real LED pixels, "
          "not just noise.")

    return 0


if __name__ == "__main__":
    sys.exit(main())
