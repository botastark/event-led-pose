#!/usr/bin/env python3
"""Inspect one replay window of the ALM detector.

This diagnostic makes the three stages visible:
  1. Event-count gate -> connected components.
  2. Local temporal binning for each retained component.
  3. Known-period normalized autocorrelation for 165/365/596 Hz.

It writes PNG and CSV outputs; no camera hardware is opened.

Example:
  python3 inspect_component_frequency_estimation.py \
      --input raw_dynamic.raw --window-index 20 --polarity on \
      --detection-window-us 20000 --count-scale 0.75 \
      --time-bin-us 50 --out-dir frequency_inspection

Outputs:
  connected_components.png       Count heatmap + colour-coded component labels.
  candidates.csv                 One row per connected component and its scores.
  candidate_XX_rate.csv          Binned component event-rate signal.
  candidate_XX_autocorr.csv      Normalized autocorrelation near each known lag.
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
    parser.add_argument("--input", required=True, help="Input .raw recording")
    parser.add_argument("--window-index", type=int, default=0,
                        help="Zero-based Td window to inspect")
    parser.add_argument("--detection-window-us", type=int, default=20_000)
    parser.add_argument("--count-scale", type=float, default=0.75)
    parser.add_argument("--polarity", choices=("on", "off", "both"), default="on")
    parser.add_argument("--time-bin-us", type=int, default=50)
    parser.add_argument("--min-area-px", type=int, default=5)
    parser.add_argument("--max-area-px", type=int, default=20_000)
    parser.add_argument("--max-candidates", type=int, default=12,
                        help="Export timing data for at most this many strongest components")
    parser.add_argument("--out-dir", default="frequency_inspection")
    return parser.parse_args()


def select_polarity(events: np.ndarray, mode: str) -> np.ndarray:
    if mode == "both":
        return events
    target = 1 if mode == "on" else 0
    return events[events["p"] == target]


def normalized_autocorrelation(rate: np.ndarray, lag: int) -> float:
    if lag <= 0 or lag >= rate.size - 1:
        return 0.0
    a = rate[:-lag]
    b = rate[lag:]
    denom = math.sqrt(float(np.dot(a, a) * np.dot(b, b)))
    return float(np.dot(a, b) / denom) if denom > 1e-12 else 0.0


def temporal_signal(events: np.ndarray, start_us: int, end_us: int, bin_us: int) -> np.ndarray:
    n_bins = max(2, math.ceil((end_us - start_us) / bin_us))
    indices = (events["t"].astype(np.int64) - start_us) // bin_us
    indices = np.clip(indices, 0, n_bins - 1)
    rate = np.bincount(indices, minlength=n_bins).astype(np.float64)
    return rate - rate.mean()


def save_csv(path: str, rows: list[dict]) -> None:
    if not rows:
        return
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    args = parse_args()
    if args.window_index < 0:
        raise ValueError("window-index must be non-negative")
    if args.detection_window_us <= 0 or args.time_bin_us <= 0:
        raise ValueError("window and bin durations must be positive")

    os.makedirs(args.out_dir, exist_ok=True)
    f_min = min(FREQUENCIES_HZ)
    threshold = max(1, int(math.ceil(
        args.detection_window_us * f_min * args.count_scale / 1_000_000.0
    )))
    print(f"Td={args.detection_window_us} us, f_min={f_min} Hz, "
          f"count_scale={args.count_scale}, pixel threshold={threshold}")

    stream = EventsIterator(
        input_path=args.input,
        mode="delta_t",
        delta_t=args.detection_window_us,
        relative_timestamps=False,
    )
    height, width = stream.get_size()
    events = None
    for index, chunk in enumerate(stream):
        if index == args.window_index:
            events = chunk
            break
    if events is None:
        raise RuntimeError("window-index is beyond the end of the recording")
    if not events.size:
        raise RuntimeError("selected window has no events")

    selected = select_polarity(events, args.polarity)
    start_us = int(events["t"].min())
    end_us = int(events["t"].max()) + 1
    counts = np.zeros((height, width), dtype=np.uint16)
    if selected.size:
        np.add.at(counts, (selected["y"], selected["x"]), 1)

    binary = (counts >= threshold).astype(np.uint8)
    n_labels, labels, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)
    print(f"Window {args.window_index}: raw={events.size:,}, selected={selected.size:,}, "
          f"surviving pixels={int(binary.sum()):,}, labels={n_labels - 1}")

    heat = np.log1p(counts.astype(np.float32))
    heat = (255.0 * heat / max(float(heat.max()), 1e-9)).astype(np.uint8)
    visualization = cv2.applyColorMap(heat, cv2.COLORMAP_TURBO)
    rng = np.random.default_rng(7)
    palette = rng.integers(70, 256, size=(max(n_labels, 2), 3), dtype=np.uint8)
    palette[0] = 0

    candidates = []
    for label in range(1, n_labels):
        x, y, w, h, area = (int(v) for v in stats[label])
        if area < args.min_area_px or area > args.max_area_px:
            continue
        region_mask = labels == label
        region_events = selected[region_mask[selected["y"], selected["x"]]]
        if not region_events.size:
            continue
        ys, xs = np.nonzero(region_mask)
        weights = counts[ys, xs].astype(np.float64)
        center_x = float(np.dot(xs, weights) / weights.sum())
        center_y = float(np.dot(ys, weights) / weights.sum())
        rate = temporal_signal(region_events, start_us, end_us, args.time_bin_us)

        scores = []
        for frequency in FREQUENCIES_HZ:
            lag = int(round(1_000_000.0 / frequency / args.time_bin_us))
            scores.append(normalized_autocorrelation(rate, lag))
        order = np.argsort(scores)[::-1]
        best_idx = int(order[0])
        second_score = float(scores[order[1]])
        best_score = float(scores[best_idx])
        ratio = best_score / max(second_score, 1e-9)
        candidates.append({
            "label": label,
            "area_px": area,
            "event_count": int(region_events.size),
            "center_x": center_x,
            "center_y": center_y,
            "score_165": float(scores[0]),
            "score_365": float(scores[1]),
            "score_596": float(scores[2]),
            "best_frequency_hz": float(FREQUENCIES_HZ[best_idx]),
            "best_score": best_score,
            "best_second_ratio": ratio,
            "rate": rate,
            "events": region_events,
            "mask": region_mask,
        })

    candidates.sort(key=lambda c: c["event_count"], reverse=True)
    visible = visualization.copy()
    for rank, candidate in enumerate(candidates):
        color = tuple(int(v) for v in palette[(rank % (len(palette) - 1)) + 1])
        visible[candidate["mask"]] = color
        cx, cy = int(round(candidate["center_x"])), int(round(candidate["center_y"]))
        cv2.drawMarker(visible, (cx, cy), (255, 255, 255), cv2.MARKER_CROSS, 12, 1)
        cv2.putText(visible, f"{rank}: {candidate['best_frequency_hz']:.0f}",
                    (cx + 5, cy - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.4,
                    (255, 255, 255), 1, cv2.LINE_AA)

    image_path = os.path.join(args.out_dir, "connected_components.png")
    cv2.imwrite(image_path, visible)

    summary_rows = []
    for rank, candidate in enumerate(candidates):
        summary_rows.append({key: value for key, value in candidate.items()
                             if key not in ("rate", "events", "mask")} | {"rank": rank})
    summary_path = os.path.join(args.out_dir, "candidates.csv")
    save_csv(summary_path, summary_rows)

    for rank, candidate in enumerate(candidates[:args.max_candidates]):
        rate_rows = []
        raw_rate = candidate["rate"] + candidate["rate"].mean()
        for n, value in enumerate(raw_rate):
            rate_rows.append({
                "bin": n,
                "bin_start_us": start_us + n * args.time_bin_us,
                "event_count": float(value),
                "zero_mean_event_rate": float(candidate["rate"][n]),
            })
        save_csv(os.path.join(args.out_dir, f"candidate_{rank:02d}_rate.csv"), rate_rows)

        corr_rows = []
        for frequency in FREQUENCIES_HZ:
            nominal_lag = int(round(1_000_000.0 / frequency / args.time_bin_us))
            for lag in range(max(1, nominal_lag - 5), nominal_lag + 6):
                corr_rows.append({
                    "frequency_hz": frequency,
                    "lag_bins": lag,
                    "lag_us": lag * args.time_bin_us,
                    "normalized_autocorrelation": normalized_autocorrelation(candidate["rate"], lag),
                })
        save_csv(os.path.join(args.out_dir, f"candidate_{rank:02d}_autocorr.csv"), corr_rows)

    print(f"Kept {len(candidates)} components after area filtering.")
    print(f"Saved component visualisation: {image_path}")
    print(f"Saved candidate summary: {summary_path}")
    print("Candidate CSV has the three known-period scores. Per-candidate CSV files "
          "show the temporal bins and autocorrelation neighbourhoods.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
