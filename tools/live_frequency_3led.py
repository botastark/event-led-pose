#!/usr/bin/env python3

import argparse
import math
import time
from collections import deque

import numpy as np
from metavision_core.event_io import EventsIterator
from metavision_sdk_core import BaseFrameGenerationAlgorithm
from metavision_sdk_ui import BaseWindow, EventLoop, MTWindow, UIKeyEvent

try:
    import cv2
except ImportError:
    cv2 = None


FREQUENCIES_HZ = (165.0, 366.0, 596.0)

# BGR
COLORS = (
    (0, 255, 0),      # 165 Hz: green
    (0, 220, 255),    # 366 Hz: yellow
    (255, 100, 0),    # 596 Hz: blue
)


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Global frequency-filtered detection and tracking "
            "of three blinking LEDs."
        )
    )

    parser.add_argument("--serial", default="00050946")

    parser.add_argument(
        "--slice-us",
        type=int,
        default=40_000,
    )
    parser.add_argument(
        "--phase-bin-us",
        type=int,
        default=100,
    )
    parser.add_argument(
        "--block-size",
        type=int,
        default=8,
    )
    parser.add_argument(
        "--event-stride",
        type=int,
        default=4,
        help="Use every Nth event for frequency processing.",
    )
    parser.add_argument(
        "--memory-ms",
        type=float,
        default=700.0,
    )
    parser.add_argument(
        "--spatial-radius-blocks",
        type=int,
        default=3,
    )

    parser.add_argument(
        "--minimum-support",
        type=float,
        default=100.0,
    )
    parser.add_argument(
        "--minimum-coherence",
        type=float,
        default=0.08,
    )
    parser.add_argument(
        "--minimum-peak-ratio",
        type=float,
        default=1.08,
    )

    parser.add_argument(
        "--center-alpha",
        type=float,
        default=0.35,
    )
    parser.add_argument(
        "--snap-distance-px",
        type=float,
        default=100.0,
    )

    parser.add_argument(
        "--detection-update-ms",
        type=float,
        default=80.0,
    )
    parser.add_argument(
        "--display-fps",
        type=float,
        default=15.0,
    )
    parser.add_argument(
        "--display-accumulation-us",
        type=int,
        default=20_000,
    )

    return parser.parse_args()


def box_sum(values, radius):
    if radius <= 0:
        return values.copy()

    size = 2 * radius + 1

    padded = np.pad(
        values,
        ((radius, radius), (radius, radius)),
        mode="constant",
    )

    integral = np.pad(
        padded,
        ((1, 0), (1, 0)),
        mode="constant",
    ).cumsum(axis=0).cumsum(axis=1)

    return (
        integral[size:, size:]
        - integral[:-size, size:]
        - integral[size:, :-size]
        + integral[:-size, :-size]
    )


def build_phase_tables(phase_bin_us):
    if 1_000_000 % phase_bin_us != 0:
        raise ValueError(
            "--phase-bin-us must divide exactly into one second"
        )

    bins_per_second = 1_000_000 // phase_bin_us

    time_seconds = (
        np.arange(bins_per_second, dtype=np.float64)
        * phase_bin_us
        / 1_000_000.0
    )

    cosine = np.empty(
        (len(FREQUENCIES_HZ), bins_per_second),
        dtype=np.float32,
    )
    sine = np.empty_like(cosine)

    for index, frequency in enumerate(FREQUENCIES_HZ):
        phase = 2.0 * math.pi * frequency * time_seconds
        cosine[index] = np.cos(phase)
        sine[index] = np.sin(phase)

    return cosine, sine, bins_per_second


class FrequencyTrack:
    def __init__(self, frequency):
        self.frequency = frequency
        self.center = None
        self.measured_center = None
        self.trail = deque(maxlen=80)

        self.coherence = 0.0
        self.support = 0.0
        self.peak_ratio = 0.0
        self.score = 0.0
        self.locked = False


def detect_frequency(
    real_state,
    imaginary_state,
    support_state,
    radius,
    minimum_support,
    minimum_coherence,
    minimum_peak_ratio,
    block_size,
):
    summed_real = box_sum(real_state, radius)
    summed_imaginary = box_sum(imaginary_state, radius)
    summed_support = box_sum(support_state, radius)

    amplitude = np.hypot(
        summed_real,
        summed_imaginary,
    )

    coherence = amplitude / np.maximum(
        summed_support,
        1e-6,
    )

    # Require both coherent periodicity and enough event support.
    score = coherence * np.sqrt(
        np.maximum(summed_support, 0.0)
    )

    peak_flat_index = int(np.argmax(score))
    peak_y, peak_x = np.unravel_index(
        peak_flat_index,
        score.shape,
    )

    peak_score = float(score[peak_y, peak_x])
    peak_support = float(
        summed_support[peak_y, peak_x]
    )
    peak_coherence = float(
        coherence[peak_y, peak_x]
    )

    # Compare with the strongest spatially separate response.
    suppressed = score.copy()

    suppression_radius = max(
        2,
        2 * radius + 1,
    )

    y0 = max(0, peak_y - suppression_radius)
    y1 = min(
        score.shape[0],
        peak_y + suppression_radius + 1,
    )
    x0 = max(0, peak_x - suppression_radius)
    x1 = min(
        score.shape[1],
        peak_x + suppression_radius + 1,
    )

    suppressed[y0:y1, x0:x1] = 0.0

    second_score = float(suppressed.max())
    peak_ratio = peak_score / max(second_score, 1e-6)

    centroid_radius = radius + 1

    centroid_y0 = max(
        0,
        peak_y - centroid_radius,
    )
    centroid_y1 = min(
        score.shape[0],
        peak_y + centroid_radius + 1,
    )
    centroid_x0 = max(
        0,
        peak_x - centroid_radius,
    )
    centroid_x1 = min(
        score.shape[1],
        peak_x + centroid_radius + 1,
    )

    patch = score[
        centroid_y0:centroid_y1,
        centroid_x0:centroid_x1,
    ]

    # Remove diffuse background before calculating the center.
    weights = np.maximum(
        patch - 0.30 * peak_score,
        0.0,
    )

    weight_sum = float(weights.sum())

    if weight_sum <= 0:
        center_block_x = peak_x + 0.5
        center_block_y = peak_y + 0.5
    else:
        local_y, local_x = np.indices(
            patch.shape,
            dtype=np.float64,
        )

        center_block_x = float(
            np.sum(
                (
                    local_x
                    + centroid_x0
                    + 0.5
                )
                * weights
            )
            / weight_sum
        )

        center_block_y = float(
            np.sum(
                (
                    local_y
                    + centroid_y0
                    + 0.5
                )
                * weights
            )
            / weight_sum
        )

    center = np.array(
        [
            center_block_x * block_size,
            center_block_y * block_size,
        ],
        dtype=np.float64,
    )

    locked = (
        peak_support >= minimum_support
        and peak_coherence >= minimum_coherence
        and peak_ratio >= minimum_peak_ratio
    )

    return {
        "center": center,
        "score": peak_score,
        "support": peak_support,
        "coherence": peak_coherence,
        "peak_ratio": peak_ratio,
        "locked": locked,
    }


def update_track(track, detection, args, width, height):
    measured = detection["center"]

    measured[0] = np.clip(
        measured[0],
        0,
        width - 1,
    )
    measured[1] = np.clip(
        measured[1],
        0,
        height - 1,
    )

    track.measured_center = measured
    track.coherence = detection["coherence"]
    track.support = detection["support"]
    track.peak_ratio = detection["peak_ratio"]
    track.score = detection["score"]
    track.locked = detection["locked"]

    if not track.locked:
        return

    if track.center is None:
        track.center = measured.copy()
    else:
        distance = float(
            np.linalg.norm(measured - track.center)
        )

        if distance >= args.snap_distance_px:
            # Global reacquisition after a large movement.
            track.center = measured.copy()
        else:
            track.center = (
                (1.0 - args.center_alpha)
                * track.center
                + args.center_alpha
                * measured
            )

    track.trail.append(
        (
            float(track.center[0]),
            float(track.center[1]),
        )
    )


def draw_numpy_marker(frame, center, radius, color):
    height, width = frame.shape[:2]

    x = int(round(center[0]))
    y = int(round(center[1]))

    x0 = max(0, x - radius)
    x1 = min(width - 1, x + radius)
    y0 = max(0, y - radius)
    y1 = min(height - 1, y + radius)

    color_array = np.asarray(
        color,
        dtype=np.uint8,
    )

    frame[y0:y0 + 2, x0:x1 + 1] = color_array
    frame[y1 - 1:y1 + 1, x0:x1 + 1] = color_array
    frame[y0:y1 + 1, x0:x0 + 2] = color_array
    frame[y0:y1 + 1, x1 - 1:x1 + 1] = color_array

    frame[
        y,
        max(0, x - 10):min(width, x + 11),
    ] = color_array

    frame[
        max(0, y - 10):min(height, y + 11),
        x,
    ] = color_array


def draw_track(frame, track, color, block_size, radius_blocks):
    center = (
        track.center
        if track.center is not None
        else track.measured_center
    )

    if center is None:
        return

    radius_px = max(
        20,
        (radius_blocks + 2) * block_size,
    )

    state = "LOCKED" if track.locked else "SEARCHING"

    if cv2 is None:
        draw_numpy_marker(
            frame,
            center,
            radius_px,
            color,
        )
        return

    x = int(round(center[0]))
    y = int(round(center[1]))

    thickness = 3 if track.locked else 1

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

    if len(track.trail) >= 2:
        points = np.asarray(
            track.trail,
            dtype=np.int32,
        ).reshape((-1, 1, 2))

        cv2.polylines(
            frame,
            [points],
            False,
            color,
            2,
        )

    label = (
        f"{track.frequency:.0f} Hz {state} "
        f"coh={track.coherence:.2f} "
        f"ratio={track.peak_ratio:.2f}"
    )

    cv2.putText(
        frame,
        label,
        (
            max(5, x - radius_px),
            max(22, y - radius_px - 8),
        ),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.50,
        color,
        1,
        cv2.LINE_AA,
    )


def print_status(tracks):
    sections = []

    for track in tracks:
        if track.center is None:
            position = "(?,?)"
        else:
            position = (
                f"({track.center[0]:.0f},"
                f"{track.center[1]:.0f})"
            )

        state = "LOCKED" if track.locked else "SEARCH"

        sections.append(
            f"{track.frequency:.0f}Hz "
            f"{state} "
            f"xy={position} "
            f"coh={track.coherence:.3f} "
            f"support={track.support:.0f} "
            f"ratio={track.peak_ratio:.2f}"
        )

    print(
        "\r" + " | ".join(sections) + " " * 5,
        end="",
        flush=True,
    )


def main():
    args = parse_args()

    if args.slice_us <= 0:
        raise ValueError("--slice-us must be positive")

    if args.block_size <= 0:
        raise ValueError("--block-size must be positive")

    if args.event_stride <= 0:
        raise ValueError("--event-stride must be positive")

    if args.memory_ms <= 0:
        raise ValueError("--memory-ms must be positive")

    cosine, sine, bins_per_second = build_phase_tables(
        args.phase_bin_us
    )

    stream = EventsIterator(
        input_path=args.serial,
        mode="delta_t",
        delta_t=args.slice_us,
        relative_timestamps=False,
    )

    sensor_height, sensor_width = stream.get_size()

    grid_width = math.ceil(
        sensor_width / args.block_size
    )
    grid_height = math.ceil(
        sensor_height / args.block_size
    )
    grid_size = grid_width * grid_height

    support_state = np.zeros(
        (grid_height, grid_width),
        dtype=np.float32,
    )

    real_state = np.zeros(
        (
            len(FREQUENCIES_HZ),
            grid_height,
            grid_width,
        ),
        dtype=np.float32,
    )

    imaginary_state = np.zeros_like(real_state)

    tracks = [
        FrequencyTrack(frequency)
        for frequency in FREQUENCIES_HZ
    ]

    decay = math.exp(
        -args.slice_us
        / (args.memory_ms * 1000.0)
    )

    frame = np.zeros(
        (sensor_height, sensor_width, 3),
        dtype=np.uint8,
    )

    last_detection = 0.0
    last_display = 0.0
    last_status = 0.0

    detection_interval = (
        args.detection_update_ms / 1000.0
    )
    display_interval = 1.0 / args.display_fps

    print(
        f"sensor={sensor_width}x{sensor_height} "
        f"grid={grid_width}x{grid_height} "
        f"memory={args.memory_ms:.0f}ms"
    )
    print(
        "165 Hz=green, 366 Hz=yellow, 596 Hz=blue. "
        "Press Q or Esc to stop."
    )

    with MTWindow(
        "Frequency-filtered three-LED tracking",
        sensor_width,
        sensor_height,
        BaseWindow.RenderMode.BGR,
    ) as window:

        def keyboard_callback(key, scancode, action, modifiers):
            del scancode, action, modifiers

            if key in (
                UIKeyEvent.KEY_Q,
                UIKeyEvent.KEY_ESCAPE,
            ):
                window.set_close_flag()

        window.set_keyboard_callback(
            keyboard_callback
        )

        for events in stream:
            EventLoop.poll_and_dispatch()

            if window.should_close():
                break

            support_state *= decay
            real_state *= decay
            imaginary_state *= decay

            if events.size:
                # Frequency processing is the expensive part. The display still
                # uses every event from the newest slice.
                processing_events = events[::args.event_stride]

                block_x = (
                    processing_events["x"].astype(
                        np.int64,
                        copy=False,
                    )
                    // args.block_size
                )

                block_y = (
                    processing_events["y"].astype(
                        np.int64,
                        copy=False,
                    )
                    // args.block_size
                )

                flat_blocks = (
                    block_y * grid_width + block_x
                )

                event_support = np.bincount(
                    flat_blocks,
                    minlength=grid_size,
                ).reshape(
                    grid_height,
                    grid_width,
                )

                support_state += event_support.astype(
                    np.float32,
                    copy=False,
                )

                polarity_sign = (
                    2.0
                    * processing_events["p"].astype(
                        np.float32,
                        copy=False,
                    )
                    - 1.0
                )

                phase_indices = (
                    (
                        processing_events["t"].astype(
                            np.int64,
                            copy=False,
                        )
                        // args.phase_bin_us
                    )
                    % bins_per_second
                )

                for frequency_index in range(
                    len(FREQUENCIES_HZ)
                ):
                    real_weights = (
                        polarity_sign
                        * cosine[
                            frequency_index,
                            phase_indices,
                        ]
                    )

                    imaginary_weights = (
                        polarity_sign
                        * sine[
                            frequency_index,
                            phase_indices,
                        ]
                    )

                    real_slice = np.bincount(
                        flat_blocks,
                        weights=real_weights,
                        minlength=grid_size,
                    ).reshape(
                        grid_height,
                        grid_width,
                    )

                    imaginary_slice = np.bincount(
                        flat_blocks,
                        weights=imaginary_weights,
                        minlength=grid_size,
                    ).reshape(
                        grid_height,
                        grid_width,
                    )

                    real_state[frequency_index] += (
                        real_slice.astype(
                            np.float32,
                            copy=False,
                        )
                    )

                    imaginary_state[frequency_index] += (
                        imaginary_slice.astype(
                            np.float32,
                            copy=False,
                        )
                    )

            now = time.monotonic()

            if now - last_detection >= detection_interval:
                for index, track in enumerate(tracks):
                    detection = detect_frequency(
                        real_state[index],
                        imaginary_state[index],
                        support_state,
                        args.spatial_radius_blocks,
                        args.minimum_support / args.event_stride,
                        args.minimum_coherence,
                        args.minimum_peak_ratio,
                        args.block_size,
                    )

                    update_track(
                        track,
                        detection,
                        args,
                        sensor_width,
                        sensor_height,
                    )

                last_detection = now

            if now - last_display >= display_interval:
                # Generate the image only from the newest event buffer.
                # This prevents old display events from accumulating.
                BaseFrameGenerationAlgorithm.generate_frame(
                    events,
                    frame,
                    accumulation_time_us=args.display_accumulation_us,
                )

                for track, color in zip(
                    tracks,
                    COLORS,
                ):
                    draw_track(
                        frame,
                        track,
                        color,
                        args.block_size,
                        args.spatial_radius_blocks,
                    )

                window.show_async(frame, auto_poll=False)
                last_display = now

            if now - last_status >= 0.5:
                print_status(tracks)
                last_status = now

    print("\nStopped.")


if __name__ == "__main__":
    main()
