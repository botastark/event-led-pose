#!/usr/bin/env python3
"""Local event-driven tracker for the 3 frequency-coded LEDs.

Architecture follows the 2023 active-LED-marker paper more closely than the
previous global frequency maps:

  global ALM-style detector (startup / recovery only)
      -> spawn one local tracker for each frequency-labelled LED
      -> update centre from CURRENT local events in short slices
      -> adapt local radius from observed spatial dispersion
      -> declare LOST after missing expected blink periods
      -> global detector reacquires only LOST tracks

The local tracker intentionally accepts all selected-polarity events inside its
local radius. This follows the paper's intended bias-tuned operating regime:
a LED blink produces a dense local event cloud, while background events between
blinks are suppressed. If that bias condition is not met, local tracking will
not be robust and must be augmented with local temporal frequency validation.

This is a tracking/reacquisition visual diagnostic. It does not publish pose.
"""

from __future__ import annotations

import argparse
import math
import sys
import time
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Deque, List, Tuple

import numpy as np
from metavision_core.event_io import EventsIterator
from metavision_sdk_core import BaseFrameGenerationAlgorithm
from metavision_sdk_ui import BaseWindow, EventLoop, MTWindow, UIKeyEvent

try:
    import cv2
except ImportError as exc:  # pragma: no cover
    raise RuntimeError("python3-opencv is required for local-tracker visualization.") from exc

# The companion detector implements: count frame -> connected components ->
# local temporal identification -> one-to-one frequency assignment.
try:
    from detect_leds_alm_style import (
        COLORS,
        FREQUENCIES_HZ,
        assign_candidates,
        detect_candidates,
        selected_events,
    )
except ImportError as exc:  # pragma: no cover
    raise RuntimeError(
        "Put detect_leds_alm_style.py in the same tools/ directory before "
        "running this tracker."
    ) from exc


@dataclass
class LocalTrack:
    frequency_hz: float
    center_xy: np.ndarray | None = None
    radius_px: float = 20.0
    covariance_xy: np.ndarray = field(default_factory=lambda: np.eye(2, dtype=np.float64) * np.inf)
    state: str = "SEARCHING"  # SEARCHING | TRACKING | LOST
    last_update_sensor_us: int | None = None
    accepted_events: int = 0
    update_count: int = 0
    trail: Deque[Tuple[float, float]] = field(default_factory=lambda: deque(maxlen=100))

    def spawn(self, center_xy: np.ndarray, radius_px: float, sensor_us: int) -> None:
        self.center_xy = np.asarray(center_xy, dtype=np.float64).copy()
        self.radius_px = float(radius_px)
        self.covariance_xy = np.eye(2, dtype=np.float64) * (radius_px * radius_px / 4.0)
        self.state = "TRACKING"
        self.last_update_sensor_us = int(sensor_us)
        self.accepted_events = 0
        self.update_count = 0
        self.trail.clear()
        self.trail.append((float(self.center_xy[0]), float(self.center_xy[1])))

    def mark_lost(self) -> None:
        self.state = "LOST"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="ALM-style global acquisition plus local event-driven LED tracking."
    )
    parser.add_argument("--serial", default="00050946")

    # Local tracker: short slices; this is the latency-critical path.
    parser.add_argument("--slice-us", type=int, default=1_000,
                        help="Short local-tracker input slice; default 1 ms.")
    parser.add_argument("--polarity", choices=("on", "off", "both"), default="on")
    parser.add_argument("--center-alpha", type=float, default=0.20,
                        help="EWMA gain for accepted local batch centres.")
    parser.add_argument("--radius-alpha", type=float, default=0.20,
                        help="EWMA gain for adaptive tracking radius.")
    parser.add_argument("--initial-radius-px", type=float, default=26.0)
    parser.add_argument("--min-radius-px", type=float, default=8.0)
    parser.add_argument("--max-radius-px", type=float, default=80.0)
    parser.add_argument("--min-update-events", type=int, default=4,
                        help="Minimum accepted local events before a centre update.")
    parser.add_argument("--lost-periods", type=float, default=4.0,
                        help="Mark track LOST after this many expected LED periods without update.")

    # Global detector: startup/recovery only. It needs Td > 2/f_min.
    parser.add_argument("--acquire-window-us", type=int, default=20_000)
    parser.add_argument("--acquire-period-ms", type=float, default=100.0,
                        help="How often to run full-image acquisition while any track is missing.")
    parser.add_argument("--count-scale", type=float, default=1.0)
    parser.add_argument("--min-area-px", type=int, default=5)
    parser.add_argument("--max-area-px", type=int, default=20_000)
    parser.add_argument("--time-bin-us", type=int, default=50)
    parser.add_argument("--min-frequency-score", type=float, default=0.10)
    parser.add_argument("--min-frequency-ratio", type=float, default=1.10)

    parser.add_argument("--display-fps", type=float, default=30.0)
    parser.add_argument("--display-accumulation-us", type=int, default=5_000)
    parser.add_argument("--print-status", action="store_true")
    return parser.parse_args()


def concat_recent_events(
    buffer: Deque[np.ndarray], current_time_us: int, window_us: int
) -> np.ndarray:
    """Return exactly the recent acquisition window from a bounded event deque."""
    earliest = current_time_us - window_us
    chunks = []
    for events in buffer:
        if events.size and int(events["t"].max()) >= earliest:
            chunks.append(events[events["t"] >= earliest])
    if not chunks:
        return np.empty(0, dtype=np.dtype([("x", np.uint16), ("y", np.uint16), ("p", np.bool_), ("t", np.int64)]))
    return np.concatenate(chunks)


def update_local_track(track: LocalTrack, events: np.ndarray, args: argparse.Namespace) -> bool:
    """Update one tracker from selected local events in the current short slice."""
    if track.state != "TRACKING" or track.center_xy is None or not events.size:
        return False

    xy = np.column_stack((events["x"], events["y"])).astype(np.float64, copy=False)
    delta = xy - track.center_xy
    inside = np.einsum("ij,ij->i", delta, delta) <= track.radius_px * track.radius_px
    local = xy[inside]
    if local.shape[0] < args.min_update_events:
        return False

    measured_center = local.mean(axis=0)
    alpha = args.center_alpha
    track.center_xy = (1.0 - alpha) * track.center_xy + alpha * measured_center

    residual = local - measured_center
    if local.shape[0] >= 2:
        # Covariance of the centre estimate, with a small numerical floor.
        scatter = (residual.T @ residual) / max(local.shape[0] - 1, 1)
        track.covariance_xy = scatter / local.shape[0] + np.eye(2) * 0.04
        mean_distance = float(np.linalg.norm(residual, axis=1).mean())
        target_radius = float(np.clip(
            2.0 * mean_distance,
            args.min_radius_px,
            args.max_radius_px,
        ))
        track.radius_px = (
            (1.0 - args.radius_alpha) * track.radius_px
            + args.radius_alpha * target_radius
        )

    track.last_update_sensor_us = int(events["t"].max())
    track.accepted_events += int(local.shape[0])
    track.update_count += 1
    track.trail.append((float(track.center_xy[0]), float(track.center_xy[1])))
    return True


def update_losses(tracks: List[LocalTrack], sensor_now_us: int, lost_periods: float) -> None:
    for track in tracks:
        if track.state != "TRACKING" or track.last_update_sensor_us is None:
            continue
        timeout_us = lost_periods * 1_000_000.0 / track.frequency_hz
        if sensor_now_us - track.last_update_sensor_us > timeout_us:
            track.mark_lost()


def acquire_missing_tracks(
    tracks: List[LocalTrack], acquisition_events: np.ndarray,
    height: int, width: int, args: argparse.Namespace,
) -> None:
    """Run global paper-like detector, but spawn only SEARCHING/LOST tracks."""
    if not acquisition_events.size:
        return

    selected = selected_events(acquisition_events, args.polarity)
    if not selected.size:
        return

    f_min = min(FREQUENCIES_HZ)
    pixel_threshold = max(1, int(math.ceil(
        (args.acquire_window_us / 1_000_000.0) * f_min * args.count_scale
    )))
    start_us = int(acquisition_events["t"].min())
    end_us = int(acquisition_events["t"].max()) + 1

    candidates = detect_candidates(
        selected, height, width, start_us, end_us,
        pixel_threshold, args.min_area_px, args.max_area_px, args.time_bin_us,
    )
    detections = assign_candidates(
        candidates, args.min_frequency_score, args.min_frequency_ratio,
    )

    for i, detection in enumerate(detections):
        if tracks[i].state == "TRACKING":
            continue
        if detection.locked and detection.candidate is not None:
            tracks[i].spawn(
                detection.candidate.center_xy,
                args.initial_radius_px,
                end_us,
            )


def draw_track(frame: np.ndarray, track: LocalTrack, color: Tuple[int, int, int]) -> None:
    if track.center_xy is None or track.state != "TRACKING":
        return
    x, y = (int(round(v)) for v in track.center_xy)
    radius = max(4, int(round(track.radius_px)))
    h, w = frame.shape[:2]
    x = min(max(x, 0), w - 1)
    y = min(max(y, 0), h - 1)

    cv2.circle(frame, (x, y), radius, color, 2)
    cv2.drawMarker(frame, (x, y), color, cv2.MARKER_CROSS, 18, 2)
    if len(track.trail) >= 2:
        points = np.asarray(track.trail, dtype=np.int32).reshape((-1, 1, 2))
        cv2.polylines(frame, [points], False, color, 1)

    sigma = math.sqrt(max(float(np.trace(track.covariance_xy) / 2.0), 0.0))
    label = (
        f"{track.frequency_hz:.0f}Hz TRACK "
        f"r={track.radius_px:.1f} sig={sigma:.2f} n={track.accepted_events}"
    )
    cv2.putText(frame, label, (max(3, x - radius), max(18, y - radius - 5)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.42, color, 1, cv2.LINE_AA)


def main() -> int:
    args = parse_args()
    if not 0.0 < args.center_alpha <= 1.0:
        raise ValueError("--center-alpha must be in (0, 1]")
    if not 0.0 < args.radius_alpha <= 1.0:
        raise ValueError("--radius-alpha must be in (0, 1]")
    if args.acquire_window_us < math.ceil(2_000_000.0 / min(FREQUENCIES_HZ)):
        raise ValueError("--acquire-window-us must satisfy Td > 2/f_min")

    stream = EventsIterator(
        input_path=args.serial,
        mode="delta_t",
        delta_t=args.slice_us,
        relative_timestamps=False,
    )
    height, width = stream.get_size()

    tracks = [LocalTrack(frequency_hz=f) for f in FREQUENCIES_HZ]
    acquisition_buffer: Deque[np.ndarray] = deque()
    frame = np.zeros((height, width, 3), dtype=np.uint8)

    acquire_period_s = args.acquire_period_ms / 1000.0
    display_period_s = 1.0 / max(args.display_fps, 1.0)
    last_acquire_wall = -float("inf")
    last_display_wall = -float("inf")
    last_status_wall = -float("inf")

    print(
        f"sensor={width}x{height}; local slice={args.slice_us} us; "
        f"acquire Td={args.acquire_window_us / 1000.0:.1f} ms; "
        f"polarity={args.polarity}"
    )

    with MTWindow(
        "ALM local LED tracker: detect to spawn, local events to track",
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

            sensor_now = int(events["t"].max())
            now = time.monotonic()

            # Keep only enough event history for acquisition/recovery.
            acquisition_buffer.append(events.copy())
            earliest = sensor_now - args.acquire_window_us
            while acquisition_buffer and int(acquisition_buffer[0]["t"].max()) < earliest:
                acquisition_buffer.popleft()

            local_events = selected_events(events, args.polarity)
            for track in tracks:
                update_local_track(track, local_events, args)
            update_losses(tracks, sensor_now, args.lost_periods)

            # Full-image detection happens only while at least one track is missing.
            any_missing = any(track.state != "TRACKING" for track in tracks)
            if any_missing and now - last_acquire_wall >= acquire_period_s:
                acquisition = concat_recent_events(
                    acquisition_buffer, sensor_now, args.acquire_window_us
                )
                acquire_missing_tracks(tracks, acquisition, height, width, args)
                last_acquire_wall = now

            if now - last_display_wall >= display_period_s:
                # UI is diagnostic only. It does not influence local tracking.
                BaseFrameGenerationAlgorithm.generate_frame(
                    events, frame, accumulation_time_us=args.display_accumulation_us
                )
                for track, color in zip(tracks, COLORS):
                    draw_track(frame, track, color)
                window.show_async(frame, auto_poll=False)
                last_display_wall = now

            if args.print_status and now - last_status_wall >= 0.25:
                parts = []
                for track in tracks:
                    if track.center_xy is None:
                        parts.append(f"{track.frequency_hz:.0f}Hz {track.state}")
                    else:
                        parts.append(
                            f"{track.frequency_hz:.0f}Hz {track.state} "
                            f"xy=({track.center_xy[0]:.1f},{track.center_xy[1]:.1f}) "
                            f"r={track.radius_px:.1f} updates={track.update_count}"
                        )
                print("\r" + " | ".join(parts) + " " * 10, end="", flush=True)
                last_status_wall = now

    print("\nStopped.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
