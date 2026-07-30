#!/usr/bin/env python3

import argparse
import json
import time
from pathlib import Path

import numpy as np
from metavision_core.event_io import EventsIterator


def parse_args():
    parser = argparse.ArgumentParser(
        description="Generate deterministic spatial and temporal statistics from a RAW event file."
    )
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output-prefix", required=True, type=Path)
    parser.add_argument("--slice-us", type=int, default=100_000)
    parser.add_argument("--time-bin-us", type=int, default=100)
    parser.add_argument("--top-pixels", type=int, default=20)
    return parser.parse_args()


def output_path(prefix, suffix):
    return Path(f"{prefix}{suffix}")


def write_log_pgm(path, counts):
    log_counts = np.log1p(counts.astype(np.float64))
    nonzero = log_counts[log_counts > 0]

    if nonzero.size == 0:
        image = np.zeros(counts.shape, dtype=np.uint8)
        clipping_value = 0.0
    else:
        clipping_value = float(np.percentile(nonzero, 99.9))
        if clipping_value <= 0:
            clipping_value = float(nonzero.max())
        image = np.clip(log_counts / clipping_value, 0.0, 1.0)
        image = np.rint(image * 255.0).astype(np.uint8)

    height, width = counts.shape
    header = f"P5\n# log1p counts; 99.9 percentile clip={clipping_value}\n{width} {height}\n255\n"

    with path.open("wb") as stream:
        stream.write(header.encode("ascii"))
        stream.write(image.tobytes(order="C"))


def main():
    args = parse_args()

    if args.slice_us <= 0:
        raise ValueError("--slice-us must be positive")
    if args.time_bin_us <= 0:
        raise ValueError("--time-bin-us must be positive")
    if args.top_pixels <= 0:
        raise ValueError("--top-pixels must be positive")

    input_path = args.input.resolve()
    prefix = args.output_prefix.resolve()

    if not input_path.is_file():
        raise FileNotFoundError(input_path)

    prefix.parent.mkdir(parents=True, exist_ok=True)

    iterator = EventsIterator(
        input_path=str(input_path),
        mode="delta_t",
        delta_t=args.slice_us,
        relative_timestamps=False,
    )

    height, width = iterator.get_size()
    if height is None or width is None:
        raise RuntimeError("Recording does not contain sensor geometry")

    total_map = np.zeros((height, width), dtype=np.int64)
    on_map = np.zeros((height, width), dtype=np.int64)
    temporal_total = np.zeros(1024, dtype=np.int64)
    temporal_on = np.zeros(1024, dtype=np.int64)

    total_events = 0
    on_events = 0
    first_t = None
    last_t = None
    next_progress = 10_000_000
    wall_start = time.monotonic()

    for events in iterator:
        if events.size == 0:
            continue

        if events.dtype.names != ("x", "y", "p", "t"):
            raise RuntimeError(f"Unexpected event dtype: {events.dtype}")

        x = events["x"].astype(np.int64, copy=False)
        y = events["y"].astype(np.int64, copy=False)

        if x.max() >= width or y.max() >= height:
            raise RuntimeError("Event coordinates exceed recording geometry")

        flat_pixels = y * width + x
        total_map += np.bincount(
            flat_pixels, minlength=height * width
        ).reshape(height, width)

        on_mask = events["p"] != 0
        slice_on_events = int(np.count_nonzero(on_mask))
        if slice_on_events:
            on_map += np.bincount(
                flat_pixels[on_mask], minlength=height * width
            ).reshape(height, width)

        time_bins = (events["t"] // args.time_bin_us).astype(np.int64, copy=False)
        slice_temporal_total = np.bincount(time_bins)
        required_bins = slice_temporal_total.size

        if required_bins > temporal_total.size:
            new_size = max(required_bins, temporal_total.size * 2)
            temporal_total.resize(new_size, refcheck=False)
            temporal_on.resize(new_size, refcheck=False)

        temporal_total[:required_bins] += slice_temporal_total

        if slice_on_events:
            slice_temporal_on = np.bincount(time_bins[on_mask])
            temporal_on[: slice_temporal_on.size] += slice_temporal_on

        if first_t is None:
            first_t = int(events["t"][0])
        last_t = int(events["t"][-1])
        total_events += int(events.size)
        on_events += slice_on_events

        if total_events >= next_progress:
            print(f"progress_events={total_events}", flush=True)
            while total_events >= next_progress:
                next_progress += 10_000_000

    if total_events == 0 or first_t is None or last_t is None:
        raise RuntimeError("Recording contains no CD events")

    off_map = total_map - on_map
    off_events = total_events - on_events
    duration_us = max(1, last_t - first_t)
    duration_s = duration_us / 1_000_000.0
    event_rate_eps = total_events / duration_s
    last_time_bin = last_t // args.time_bin_us

    temporal_total = temporal_total[: last_time_bin + 1]
    temporal_on = temporal_on[: last_time_bin + 1]
    temporal_off = temporal_total - temporal_on

    flat_total = total_map.ravel()
    number_of_top_pixels = min(args.top_pixels, flat_total.size)
    top_indices = np.argpartition(
        flat_total, -number_of_top_pixels
    )[-number_of_top_pixels:]
    top_indices = top_indices[np.argsort(flat_total[top_indices])[::-1]]

    top_pixels = []
    for flat_index in top_indices:
        y_value, x_value = divmod(int(flat_index), width)
        count = int(total_map[y_value, x_value])
        top_pixels.append(
            {
                "x": x_value,
                "y": y_value,
                "events": count,
                "on_events": int(on_map[y_value, x_value]),
                "off_events": int(off_map[y_value, x_value]),
                "event_rate_eps": count / duration_s,
            }
        )

    report = {
        "input": str(input_path),
        "geometry": {"width": width, "height": height},
        "slice_us": args.slice_us,
        "time_bin_us": args.time_bin_us,
        "first_t_us": first_t,
        "last_t_us": last_t,
        "duration_us": duration_us,
        "events": total_events,
        "on_events": on_events,
        "off_events": off_events,
        "event_rate_eps": event_rate_eps,
        "active_pixels": int(np.count_nonzero(total_map)),
        "active_pixel_fraction": float(np.count_nonzero(total_map) / total_map.size),
        "processing_wall_s": time.monotonic() - wall_start,
        "top_pixels": top_pixels,
    }

    npz_path = output_path(prefix, ".npz")
    json_path = output_path(prefix, ".json")
    total_pgm_path = output_path(prefix, "_total_log.pgm")
    on_pgm_path = output_path(prefix, "_on_log.pgm")
    off_pgm_path = output_path(prefix, "_off_log.pgm")

    np.savez_compressed(
        npz_path,
        total_map=total_map,
        on_map=on_map,
        off_map=off_map,
        temporal_total=temporal_total,
        temporal_on=temporal_on,
        temporal_off=temporal_off,
        time_bin_us=np.int64(args.time_bin_us),
        first_t_us=np.int64(first_t),
        last_t_us=np.int64(last_t),
    )

    with json_path.open("w", encoding="utf-8") as stream:
        json.dump(report, stream, indent=2, sort_keys=True)
        stream.write("\n")

    write_log_pgm(total_pgm_path, total_map)
    write_log_pgm(on_pgm_path, on_map)
    write_log_pgm(off_pgm_path, off_map)

    print(f"input={input_path}")
    print(f"geometry={width}x{height}")
    print(f"events={total_events}")
    print(f"on_events={on_events}")
    print(f"off_events={off_events}")
    print(f"duration_s={duration_s:.6f}")
    print(f"event_rate_eps={event_rate_eps:.3f}")
    print(f"active_pixels={report['active_pixels']}")
    print(f"active_pixel_fraction={report['active_pixel_fraction']:.9f}")
    print(f"processing_wall_s={report['processing_wall_s']:.6f}")
    print(f"npz={npz_path}")
    print(f"json={json_path}")
    print(f"total_pgm={total_pgm_path}")
    print("RAW analysis: PASS")


if __name__ == "__main__":
    main()
