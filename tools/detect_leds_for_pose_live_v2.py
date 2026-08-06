#!/usr/bin/env python3
"""Short-window LED detection + live visualization (v2).

This version adds:
- Detection cadence gating (to reduce latency).
- Frequency-specific thresholds (stricter for blue, slightly relaxed for yellow).
- Temporal consistency via a small lock counter per LED.
- Only draws blobs when a LED is stably locked.

It is still ALM-style: short-window Td detection per slice, per-frequency
cos/sin templates, and block-wise accumulation.
"""

from __future__ import annotations

import argparse
import math
import sys
import time
from dataclasses import dataclass
from typing import List, Tuple

import numpy as np
from metavision_core.event_io import EventsIterator
from metavision_sdk_core import BaseFrameGenerationAlgorithm
from metavision_sdk_ui import BaseWindow, EventLoop, MTWindow, UIKeyEvent

try:
    import cv2
except ImportError:  # pragma: no cover
    cv2 = None


FREQUENCIES_HZ: Tuple[float, float, float] = (165.0, 366.0, 596.0)
COLORS: Tuple[Tuple[int, int, int], ...] = (
    (0, 255, 0),    # ~165 Hz: green
    (0, 220, 255),  # ~366 Hz: yellow
    (255, 100, 0),  # ~596 Hz: blue
)


@dataclass
class LedObservation:
    frequency: float
    center_xy: np.ndarray
    coherence: float
    support: float
    peak_ratio: float
    locked: bool


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Short-window detection + live visualization of three blinking LEDs "
            "for pose estimation (v2)."
        )
    )

    parser.add_argument("--serial", default="00050946")

    parser.add_argument(
        "--slice-us",
        type=int,
        default=20_000,
        help="Event slice length for the iterator (microseconds).",
    )

    parser.add_argument(
        "--detection-window-us",
        type=int,
        default=20_000,
        help=(
            "Short accumulation window Td for detection. "
            "Must be >= 2/f_min; default 20 ms."
        ),
    )

    parser.add_argument(
        "--phase-bin-us",
        type=int,
        default=100,
        help=(
            "Timestamp bin size for phase tables [microseconds]. "
            "Should divide 1e6 for simplicity."
        ),
    )

    parser.add_argument(
        "--block-size",
        type=int,
        default=12,
        help="Spatial block size (pixels) for frequency accumulation.",
    )

    parser.add_argument(
        "--event-stride",
        type=int,
        default=2,
        help="Use every Nth event for frequency processing (performance trade-off).",
    )

    parser.add_argument(
        "--spatial-radius-blocks",
        type=int,
        default=2,
        help="Box-filter radius (in blocks) for spatial smoothing of responses.",
    )

    parser.add_argument(
        "--minimum-support-base",
        type=float,
        default=100.0,
        help="Base minimum support for green/yellow; blue uses a stricter value.",
    )

    parser.add_argument(
        "--minimum-coherence",
        type=float,
        default=0.12,
        help="Minimum coherence for a valid LED (0-1).",
    )

    parser.add_argument(
        "--minimum-peak-ratio-base",
        type=float,
        default=1.30,
        help="Base minimum peak-vs-second-best score ratio; blue uses stricter.",
    )

    parser.add_argument(
        "--display-fps",
        type=float,
        default=30.0,
        help="Maximum display refresh rate (frames per second).",
    )

    parser.add_argument(
        "--display-observations",
        action="store_true",
        help="Print LED observations as text for debugging.",
    )

    return parser.parse_args()


def box_sum(values: np.ndarray, radius: int) -> np.ndarray:
    if radius <= 0:
        return values.copy()
    size = 2 * radius + 1
    padded = np.pad(values, ((radius, radius), (radius, radius)), mode="constant")
    integral = np.pad(padded, ((1, 0), (1, 0)), mode="constant").cumsum(axis=0).cumsum(axis=1)
    return (
        integral[size:, size:]
        - integral[:-size, size:]
        - integral[size:, :-size]
        + integral[:-size, :-size]
    )


def build_phase_tables(phase_bin_us: int) -> tuple[np.ndarray, np.ndarray, int]:
    if 1_000_000 % phase_bin_us != 0:
        raise ValueError("--phase-bin-us must divide exactly into one second (1,000,000)")
    bins_per_second = 1_000_000 // phase_bin_us
    time_seconds = (
        np.arange(bins_per_second, dtype=np.float64) * phase_bin_us / 1_000_000.0
    )
    cosine = np.empty((len(FREQUENCIES_HZ), bins_per_second), dtype=np.float32)
    sine = np.empty_like(cosine)
    for index, frequency in enumerate(FREQUENCIES_HZ):
        phase = 2.0 * math.pi * frequency * time_seconds
        cosine[index] = np.cos(phase)
        sine[index] = np.sin(phase)
    return cosine, sine, bins_per_second


def detect_frequency_short_window(
    real_state: np.ndarray,
    imag_state: np.ndarray,
    support_state: np.ndarray,
    radius_blocks: int,
    minimum_support: float,
    minimum_coherence: float,
    minimum_peak_ratio: float,
    block_size: int,
) -> tuple[np.ndarray, float, float, float, float, bool]:
    summed_real = box_sum(real_state, radius_blocks)
    summed_imag = box_sum(imag_state, radius_blocks)
    summed_support = box_sum(support_state, radius_blocks)

    amplitude = np.hypot(summed_real, summed_imag)
    coherence = amplitude / np.maximum(summed_support, 1e-6)

    score = coherence * np.sqrt(np.maximum(summed_support, 0.0))

    peak_flat_index = int(np.argmax(score))
    peak_y, peak_x = np.unravel_index(peak_flat_index, score.shape)

    peak_score = float(score[peak_y, peak_x])
    peak_support = float(summed_support[peak_y, peak_x])
    peak_coherence = float(coherence[peak_y, peak_x])

    suppressed = score.copy()
    suppression_radius = max(2, 2 * radius_blocks + 1)

    y0 = max(0, peak_y - suppression_radius)
    y1 = min(score.shape[0], peak_y + suppression_radius + 1)
    x0 = max(0, peak_x - suppression_radius)
    x1 = min(score.shape[1], peak_x + suppression_radius + 1)

    suppressed[y0:y1, x0:x1] = 0.0
    second_score = float(suppressed.max())
    peak_ratio = peak_score / max(second_score, 1e-6)

    # Adaptive centroid patch: shrink when support is very high (close range).
    if peak_support > 5.0 * minimum_support:
        r = max(1, radius_blocks - 1)
    else:
        r = radius_blocks + 1

    cy0 = max(0, peak_y - r)
    cy1 = min(score.shape[0], peak_y + r + 1)
    cx0 = max(0, peak_x - r)
    cx1 = min(score.shape[1], peak_x + r + 1)

    support_patch = summed_support[cy0:cy1, cx0:cx1].astype(np.float64)
    w_sum = support_patch.sum()
    if w_sum <= 0.0:
        center_block_x = peak_x + 0.5
        center_block_y = peak_y + 0.5
    else:
        yy, xx = np.indices(support_patch.shape, dtype=np.float64)
        center_block_x = (support_patch * (xx + cx0 + 0.5)).sum() / w_sum
        center_block_y = (support_patch * (yy + cy0 + 0.5)).sum() / w_sum

    center_xy = np.array([
        center_block_x * block_size,
        center_block_y * block_size,
    ], dtype=np.float64)

    locked = (
        peak_support >= minimum_support
        and peak_coherence >= minimum_coherence
        and peak_ratio >= minimum_peak_ratio
    )

    return center_xy, peak_score, peak_support, peak_coherence, peak_ratio, locked


def draw_led_blob(
    frame: np.ndarray,
    obs: LedObservation,
    color: tuple[int, int, int],
    block_size: int,
    radius_blocks: int,
) -> None:
    if obs.center_xy is None:
        return

    height, width = frame.shape[:2]
    x = int(round(obs.center_xy[0]))
    y = int(round(obs.center_xy[1]))

    radius_px = max(20, (radius_blocks + 2) * block_size)
    state = "LOCKED" if obs.locked else "SEARCH"

    x = max(0, min(width - 1, x))
    y = max(0, min(height - 1, y))

    if cv2 is None:
        x0 = max(0, x - radius_px)
        x1 = min(width - 1, x + radius_px)
        y0 = max(0, y - radius_px)
        y1 = min(height - 1, y + radius_px)
        col = np.asarray(color, dtype=np.uint8)
        frame[y0:y0 + 2, x0:x1 + 1] = col
        frame[y1 - 1:y1 + 1, x0:x1 + 1] = col
        frame[y0:y1 + 1, x0:x0 + 2] = col
        frame[y0:y1 + 1, x1 - 1:x1 + 1] = col
        frame[y, max(0, x - 10):min(width, x + 11)] = col
        frame[max(0, y - 10):min(height, y + 11), x] = col
        return

    thickness = 3 if obs.locked else 1
    cv2.rectangle(
        frame,
        (x - radius_px, y - radius_px),
        (x + radius_px, y + radius_px),
        color,
        thickness,
    )
    cv2.drawMarker(
        frame,
        (x, y),
        color,
        markerType=cv2.MARKER_CROSS,
        markerSize=24,
        thickness=2,
    )

    label = (
        f"{obs.frequency:.0f} Hz {state} "
        f"coh={obs.coherence:.2f} "
        f"sup={obs.support:.0f} "
        f"ratio={obs.peak_ratio:.2f}"
    )
    cv2.putText(
        frame,
        label,
        (max(5, x - radius_px), max(22, y - radius_px - 8)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.50,
        color,
        1,
        cv2.LINE_AA,
    )


def main() -> int:
    args = parse_args()

    if args.slice_us <= 0:
        raise ValueError("--slice-us must be positive")
    if args.detection_window_us <= 0:
        raise ValueError("--detection-window-us must be positive")
    if args.block_size <= 0:
        raise ValueError("--block-size must be positive")
    if args.event_stride <= 0:
        raise ValueError("--event-stride must be positive")

    cosine, sine, bins_per_second = build_phase_tables(args.phase_bin_us)

    stream = EventsIterator(
        input_path=args.serial,
        mode="delta_t",
        delta_t=args.slice_us,
        relative_timestamps=False,
    )

    sensor_height, sensor_width = stream.get_size()
    grid_width = math.ceil(sensor_width / args.block_size)
    grid_height = math.ceil(sensor_height / args.block_size)
    grid_size = grid_width * grid_height

    print(
        f"sensor={sensor_width}x{sensor_height} "
        f"grid={grid_width}x{grid_height} "
        f"Td={args.detection_window_us/1000.0:.1f} ms"
    )

    det_support = np.zeros((grid_height, grid_width), dtype=np.float32)
    det_real = np.zeros((len(FREQUENCIES_HZ), grid_height, grid_width), dtype=np.float32)
    det_imag = np.zeros_like(det_real)

    frame = np.zeros((sensor_height, sensor_width, 3), dtype=np.uint8)

    display_interval = 1.0 / max(args.display_fps, 1.0)
    last_display = time.monotonic()

    # Detection cadence: run full detection roughly every Td.
    detection_interval = args.detection_window_us / 1e6
    last_detection = time.monotonic()

    # Frequency-specific thresholds (blue stricter, yellow slightly relaxed).
    base_support = args.minimum_support_base
    base_ratio = args.minimum_peak_ratio_base
    MIN_SUPPORT = {
        0: base_support,          # green
        1: base_support * 0.9,    # yellow: slightly relaxed
        2: base_support * 2.0,    # blue: much stricter
    }
    MIN_PEAK_RATIO = {
        0: base_ratio,            # green
        1: base_ratio * 0.95,     # yellow
        2: base_ratio * 1.4,      # blue
    }

    # Temporal lock counters per frequency.
    lock_count = [0, 0, 0]
    stable_locked_flags = [False, False, False]
    observations: List[LedObservation] = [
        LedObservation(
            frequency=float(freq),
            center_xy=np.array([np.nan, np.nan], dtype=np.float64),
            coherence=0.0,
            support=0.0,
            peak_ratio=0.0,
            locked=False,
        )
        for freq in FREQUENCIES_HZ
    ]

    with MTWindow(
        "Short-window LED detection for pose (v2)",
        sensor_width,
        sensor_height,
        BaseWindow.RenderMode.BGR,
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

            now = time.monotonic()

            # Run detection only every detection_interval to reduce latency.
            if now - last_detection >= detection_interval:
                det_support.fill(0.0)
                det_real.fill(0.0)
                det_imag.fill(0.0)

                processing_events = events[::args.event_stride]
                block_x = (
                    processing_events["x"].astype(np.int64, copy=False)
                    // args.block_size
                )
                block_y = (
                    processing_events["y"].astype(
                        np.int64,
                        copy=False,
                    )
                    // args.block_size
                )
                flat_blocks = block_y * grid_width + block_x

                event_support = np.bincount(flat_blocks, minlength=grid_size).reshape(
                    grid_height,
                    grid_width,
                )
                det_support += event_support.astype(np.float32, copy=False)

                polarity_sign = (
                    2.0
                    * processing_events["p"].astype(np.float32, copy=False)
                    - 1.0
                )
                phase_indices = (
                    processing_events["t"].astype(np.int64, copy=False)
                    // args.phase_bin_us
                    % bins_per_second
                )

                for f_idx in range(len(FREQUENCIES_HZ)):
                    real_weights = polarity_sign * cosine[f_idx, phase_indices]
                    imag_weights = polarity_sign * sine[f_idx, phase_indices]

                    real_slice = np.bincount(
                        flat_blocks,
                        weights=real_weights,
                        minlength=grid_size,
                    ).reshape(grid_height, grid_width)
                    imag_slice = np.bincount(
                        flat_blocks,
                        weights=imag_weights,
                        minlength=grid_size,
                    ).reshape(grid_height, grid_width)

                    det_real[f_idx] += real_slice.astype(np.float32, copy=False)
                    det_imag[f_idx] += imag_slice.astype(np.float32, copy=False)

                # Update observations and temporal lock state.
                for f_idx, frequency in enumerate(FREQUENCIES_HZ):
                    center_xy, score, support, coh, peak_ratio, locked = detect_frequency_short_window(
                        det_real[f_idx],
                        det_imag[f_idx],
                        det_support,
                        args.spatial_radius_blocks,
                        minimum_support=MIN_SUPPORT[f_idx] / args.event_stride,
                        minimum_coherence=args.minimum_coherence,
                        minimum_peak_ratio=MIN_PEAK_RATIO[f_idx],
                        block_size=args.block_size,
                    )

                    # Temporal lock: require two consecutive locked slices.
                    if locked:
                        lock_count[f_idx] = min(lock_count[f_idx] + 1, 3)
                    else:
                        lock_count[f_idx] = max(lock_count[f_idx] - 1, 0)

                    stable_locked = lock_count[f_idx] >= 2
                    stable_locked_flags[f_idx] = stable_locked

                    observations[f_idx] = LedObservation(
                        frequency=float(frequency),
                        center_xy=center_xy,
                        coherence=coh,
                        support=support,
                        peak_ratio=peak_ratio,
                        locked=stable_locked,
                    )

                last_detection = now

            # Visualization at display_fps.
            if now - last_display >= display_interval:
                BaseFrameGenerationAlgorithm.generate_frame(
                    events,
                    frame,
                    accumulation_time_us=args.detection_window_us,
                )

                for f_idx, (obs, color) in enumerate(zip(observations, COLORS)):
                    if not obs.locked:
                        continue  # do not draw blobs for weak / absent LEDs
                    draw_led_blob(
                        frame,
                        obs,
                        color,
                        args.block_size,
                        args.spatial_radius_blocks,
                    )

                window.show_async(frame, auto_poll=False)
                last_display = now

            if args.display_observations:
                parts = []
                for f_idx, obs in enumerate(observations):
                    state = "LOCK" if obs.locked else "SEARCH"
                    parts.append(
                        f"{obs.frequency:.0f}Hz {state} "
                        f"xy=({obs.center_xy[0]:.0f},{obs.center_xy[1]:.0f}) "
                        f"coh={obs.coherence:.2f} "
                        f"sup={obs.support:.0f} "
                        f"ratio={obs.peak_ratio:.2f} "
                        f"cnt={lock_count[f_idx]}"
                    )
                print("\r" + " | ".join(parts) + " " * 5, end="", flush=True)

    print("\nStopped.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
