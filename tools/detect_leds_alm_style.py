#!/usr/bin/env python3
"""ALM-style three-LED *detection* for the EVK4/IMX636.

This is intentionally closer to the detection architecture in:
  Real-Time 6-DoF Pose Estimation by an Event-Based Camera Using
  Active LED Markers (2023), Section 3.1.

Pipeline per detection window Td:
  1. Keep one selected event polarity (default ON), as in the paper's
     bias-tuned, one-event-per-pixel-per-blink operating regime.
  2. Accumulate a pixel event-count frame over Td.
  3. Mark pixels with counts >= ceil(Td * f_min * count_scale).
  4. Extract connected regions and reject regions below min-area-px.
  5. Compute the event-count weighted 2D center of mass per region.
  6. Identify each candidate from a local temporal event-rate signal.
     The implementation uses an autocorrelation score at each known LED
     period; this is a practical local timing test for the current hardware.
  7. Assign at most one connected component to each known frequency.

This file is a DETECTOR and REACQUIRER only. It deliberately does not pretend
that a 20 ms accumulated frame is an event-level tracker. A later per-LED local
tracker should consume its detections and update center positions from current
local events, as in the 2023 paper.

Important differences from the paper:
- Your current LED frequencies are approximately 165, 366, and 596 Hz, rather
  than the paper's preferred multi-kHz/integer-microsecond frequencies.
- The paper uses a histogram of time differences for frequency recognition.
  Here we use local binned-rate autocorrelation at known periods because it is
  vectorizable in Python and robust enough for a first replayable baseline.
- Correct bias/refractory tuning is still required. Without a sparse LED-only
  event stream, the paper's count threshold has less discrimination.
"""

from __future__ import annotations

import argparse
import math
import sys
import time
from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np
from metavision_core.event_io import EventsIterator
from metavision_sdk_core import BaseFrameGenerationAlgorithm
from metavision_sdk_ui import BaseWindow, EventLoop, MTWindow, UIKeyEvent

try:
    import cv2
except ImportError as exc:  # pragma: no cover
    raise RuntimeError(
        "This ALM-style detector needs python3-opencv (cv2) for connected components."
    ) from exc


# Provisional values from analyze_led_frequencies.py. Replace after electrical
# or firmware-side confirmation. The mapping below must match your physical plate.
FREQUENCIES_HZ: Tuple[float, float, float] = (165.0, 366.0, 596.0)
COLORS: Tuple[Tuple[int, int, int], ...] = (
    (0, 255, 0),    # 165 Hz: green
    (0, 220, 255),  # 366 Hz: yellow
    (255, 100, 0),  # 596 Hz: blue
)


@dataclass
class Candidate:
    label: int
    area_px: int
    event_count: int
    center_xy: np.ndarray
    bbox_xywh: Tuple[int, int, int, int]
    frequency_scores: np.ndarray
    best_frequency_index: int
    best_score: float
    best_ratio: float


@dataclass
class LedDetection:
    frequency: float
    candidate: Candidate | None
    locked: bool


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="ALM-style short-window detection of three known blinking LEDs."
    )
    parser.add_argument("--serial", default="00050946")
    parser.add_argument(
        "--detection-window-us", type=int, default=20_000,
        help="Td accumulation window. Must be > 2/f_min; default: 20 ms.",
    )
    parser.add_argument(
        "--polarity", choices=("on", "off", "both"), default="on",
        help="Event polarity used for detection. 'on' best matches the paper's intended mode.",
    )
    parser.add_argument(
        "--count-scale", type=float, default=1.0,
        help="Multiply Td*f_min pixel count threshold; lower only if bias tuning is not yet ideal.",
    )
    parser.add_argument(
        "--min-area-px", type=int, default=5,
        help="Minimum connected-component area in thresholded pixel frame.",
    )
    parser.add_argument(
        "--max-area-px", type=int, default=20_000,
        help="Reject connected components larger than this area.",
    )
    parser.add_argument(
        "--time-bin-us", type=int, default=50,
        help="Local ROI event-rate bin width for frequency recognition.",
    )
    parser.add_argument(
        "--min-frequency-score", type=float, default=0.10,
        help="Minimum normalized local period-autocorrelation score.",
    )
    parser.add_argument(
        "--min-frequency-ratio", type=float, default=1.10,
        help="Best frequency score divided by second-best known-frequency score.",
    )
    parser.add_argument(
        "--display-fps", type=float, default=30.0)
    parser.add_argument("--print-status", action="store_true")
    return parser.parse_args()


def selected_events(events: np.ndarray, polarity: str) -> np.ndarray:
    if polarity == "both":
        return events
    target = 1 if polarity == "on" else 0
    return events[events["p"] == target]


def count_frame(events: np.ndarray, height: int, width: int) -> np.ndarray:
    frame = np.zeros((height, width), dtype=np.uint16)
    if events.size:
        np.add.at(frame, (events["y"], events["x"]), 1)
    return frame


def candidate_frequency_scores(
    candidate_events: np.ndarray,
    window_start_us: int,
    window_end_us: int,
    bin_us: int,
) -> np.ndarray:
    """Score each known frequency using local event-rate autocorrelation.

    For each expected LED period P_i, calculate correlation of the zero-mean
    binned rate signal with itself shifted by round(P_i/bin_us). A periodic
    signal at f_i gives positive correlation at its expected period.
    """
    if candidate_events.size < 4:
        return np.zeros(len(FREQUENCIES_HZ), dtype=np.float64)

    duration_us = int(window_end_us - window_start_us)
    n_bins = max(2, math.ceil(duration_us / bin_us))
    bins = (candidate_events["t"].astype(np.int64) - window_start_us) // bin_us
    bins = np.clip(bins, 0, n_bins - 1)
    rate = np.bincount(bins, minlength=n_bins).astype(np.float64)
    rate -= rate.mean()
    energy = float(np.dot(rate, rate))
    if energy <= 1e-12:
        return np.zeros(len(FREQUENCIES_HZ), dtype=np.float64)

    scores = np.zeros(len(FREQUENCIES_HZ), dtype=np.float64)
    for i, frequency in enumerate(FREQUENCIES_HZ):
        lag = max(1, int(round((1_000_000.0 / frequency) / bin_us)))
        if lag >= n_bins - 1:
            continue
        # Normalize using energies of overlapping segments.
        a = rate[:-lag]
        b = rate[lag:]
        denom = math.sqrt(float(np.dot(a, a) * np.dot(b, b)))
        scores[i] = float(np.dot(a, b) / denom) if denom > 1e-12 else 0.0
    return scores


def detect_candidates(
    events: np.ndarray,
    height: int,
    width: int,
    window_start_us: int,
    window_end_us: int,
    pixel_threshold: int,
    min_area_px: int,
    max_area_px: int,
    time_bin_us: int,
) -> List[Candidate]:
    """Implement ALM-like pixel count threshold -> components -> local ID."""
    counts = count_frame(events, height, width)
    binary = (counts >= pixel_threshold).astype(np.uint8)

    n_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
        binary, connectivity=8
    )

    candidates: List[Candidate] = []
    for label in range(1, n_labels):
        x, y, w, h, area = (int(v) for v in stats[label])
        if area < min_area_px or area > max_area_px:
            continue

        # Events inside the actual connected component, rather than merely its bbox.
        event_labels = labels[events["y"], events["x"]]
        mask = event_labels == label
        region_events = events[mask]
        if not region_events.size:
            continue

        # Count-weighted COM of active pixels in the connected component.
        ys, xs = np.nonzero(labels == label)
        weights = counts[ys, xs].astype(np.float64)
        total = float(weights.sum())
        center_xy = np.array([
            float(np.dot(xs, weights) / total),
            float(np.dot(ys, weights) / total),
        ], dtype=np.float64)

        freq_scores = candidate_frequency_scores(
            region_events, window_start_us, window_end_us, time_bin_us
        )
        order = np.argsort(freq_scores)[::-1]
        best = int(order[0])
        best_score = float(freq_scores[best])
        second = float(freq_scores[order[1]]) if len(order) > 1 else 0.0
        best_ratio = best_score / max(second, 1e-9)

        candidates.append(Candidate(
            label=label,
            area_px=area,
            event_count=int(region_events.size),
            center_xy=center_xy,
            bbox_xywh=(x, y, w, h),
            frequency_scores=freq_scores,
            best_frequency_index=best,
            best_score=best_score,
            best_ratio=best_ratio,
        ))
    return candidates


def assign_candidates(
    candidates: List[Candidate],
    min_score: float,
    min_ratio: float,
) -> List[LedDetection]:
    """One-to-one greedy assignment of components to known frequency IDs."""
    detections = [LedDetection(frequency=f, candidate=None, locked=False)
                  for f in FREQUENCIES_HZ]
    used_labels: set[int] = set()

    # Evaluate all (frequency, candidate) pairs and assign strongest first.
    proposals = []
    for candidate in candidates:
        for frequency_index, score in enumerate(candidate.frequency_scores):
            other = np.delete(candidate.frequency_scores, frequency_index)
            ratio = float(score / max(float(other.max()), 1e-9)) if other.size else float("inf")
            proposals.append((float(score), ratio, frequency_index, candidate))

    for score, ratio, f_idx, candidate in sorted(proposals, reverse=True, key=lambda x: x[0]):
        if detections[f_idx].locked or candidate.label in used_labels:
            continue
        if score < min_score or ratio < min_ratio:
            continue
        detections[f_idx] = LedDetection(
            frequency=FREQUENCIES_HZ[f_idx], candidate=candidate, locked=True
        )
        used_labels.add(candidate.label)
    return detections


def draw_detection(frame: np.ndarray, detection: LedDetection, color: Tuple[int, int, int]) -> None:
    if not detection.locked or detection.candidate is None:
        return
    candidate = detection.candidate
    x, y, w, h = candidate.bbox_xywh
    center = tuple(int(round(v)) for v in candidate.center_xy)
    cv2.rectangle(frame, (x, y), (x + w - 1, y + h - 1), color, 2)
    cv2.drawMarker(frame, center, color, cv2.MARKER_CROSS, 20, 2)
    text = (
        f"{detection.frequency:.0f}Hz "
        f"ac={candidate.best_score:.2f} r={candidate.best_ratio:.2f} "
        f"n={candidate.event_count}"
    )
    cv2.putText(
        frame, text, (max(4, x), max(18, y - 6)),
        cv2.FONT_HERSHEY_SIMPLEX, 0.48, color, 1, cv2.LINE_AA
    )


def main() -> int:
    args = parse_args()
    if args.detection_window_us <= 0:
        raise ValueError("--detection-window-us must be positive")
    if args.time_bin_us <= 0:
        raise ValueError("--time-bin-us must be positive")
    if args.count_scale <= 0:
        raise ValueError("--count-scale must be positive")

    f_min = min(FREQUENCIES_HZ)
    min_td_us = math.ceil(2_000_000.0 / f_min)
    if args.detection_window_us < min_td_us:
        raise ValueError(
            f"Td={args.detection_window_us} us is too short; need at least {min_td_us} us for 2/f_min"
        )

    # Paper threshold: Td*f_min events per pixel. Integer count threshold.
    pixel_threshold = max(
        1,
        int(math.ceil((args.detection_window_us / 1_000_000.0) * f_min * args.count_scale)),
    )

    stream = EventsIterator(
        input_path=args.serial,
        mode="delta_t",
        delta_t=args.detection_window_us,
        relative_timestamps=False,
    )
    height, width = stream.get_size()
    frame = np.zeros((height, width, 3), dtype=np.uint8)

    display_period = 1.0 / max(args.display_fps, 1.0)
    last_display = 0.0

    print(
        f"sensor={width}x{height}; Td={args.detection_window_us / 1000:.1f} ms; "
        f"f_min={f_min:.1f} Hz; per-pixel threshold={pixel_threshold}; "
        f"polarity={args.polarity}"
    )

    with MTWindow(
        "ALM-style 3-LED detection (candidate components + local frequency ID)",
        width, height, BaseWindow.RenderMode.BGR,
    ) as window:
        def keyboard_callback(key, scancode, action, modifiers):
            del scancode, action, modifiers
            if key in (UIKeyEvent.KEY_Q, UIKeyEvent.KEY_ESCAPE):
                window.set_close_flag()

        window.set_keyboard_callback(keyboard_callback)

        for events in stream:
            EventLoop.poll_and_dispatch()
            if window.should_close():
                break
            if not events.size:
                continue

            selected = selected_events(events, args.polarity)
            window_start = int(events["t"].min())
            window_end = int(events["t"].max()) + 1

            candidates = detect_candidates(
                selected, height, width, window_start, window_end,
                pixel_threshold, args.min_area_px, args.max_area_px, args.time_bin_us
            )
            detections = assign_candidates(
                candidates, args.min_frequency_score, args.min_frequency_ratio
            )

            now = time.monotonic()
            if now - last_display >= display_period:
                BaseFrameGenerationAlgorithm.generate_frame(
                    events, frame, accumulation_time_us=args.detection_window_us
                )
                for detection, color in zip(detections, COLORS):
                    draw_detection(frame, detection, color)
                window.show_async(frame, auto_poll=False)
                last_display = now

            if args.print_status:
                result = []
                for detection in detections:
                    if not detection.locked or detection.candidate is None:
                        result.append(f"{detection.frequency:.0f}Hz SEARCH")
                        continue
                    c = detection.candidate
                    result.append(
                        f"{detection.frequency:.0f}Hz LOCK "
                        f"xy=({c.center_xy[0]:.1f},{c.center_xy[1]:.1f}) "
                        f"area={c.area_px} ev={c.event_count} "
                        f"ac={c.best_score:.3f} ratio={c.best_ratio:.2f}"
                    )
                print("\r" + " | ".join(result) + " " * 8, end="", flush=True)

    print("\nStopped.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
