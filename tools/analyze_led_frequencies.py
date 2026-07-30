#!/usr/bin/env python3

import argparse
import json
import math
import time
from pathlib import Path

import numpy as np
from metavision_core.event_io import EventsIterator


def parse_args():
    parser = argparse.ArgumentParser(
        description="Estimate blinking-LED frequency candidates from detected marker regions."
    )
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--detections", required=True, type=Path)
    parser.add_argument("--output-prefix", required=True, type=Path)
    parser.add_argument("--bin-us", type=int, default=50)
    parser.add_argument("--slice-us", type=int, default=100_000)
    parser.add_argument("--roi-radius-px", type=float, default=30.0)
    parser.add_argument("--skip-start-ms", type=float, default=500.0)
    parser.add_argument("--skip-end-ms", type=float, default=500.0)
    parser.add_argument("--min-frequency-hz", type=float, default=5.0)
    parser.add_argument("--max-frequency-hz", type=float, default=3000.0)
    parser.add_argument("--top-candidates", type=int, default=12)
    return parser.parse_args()


def ensure_length(array, required_length):
    if required_length <= array.size:
        return
    new_length = max(required_length, array.size * 2)
    array.resize(new_length, refcheck=False)


def local_peak_indices(values):
    if values.size < 3:
        return np.empty(0, dtype=np.int64)
    return np.nonzero(
        (values[1:-1] > values[:-2]) & (values[1:-1] >= values[2:])
    )[0] + 1


def select_separated_peaks(frequencies, scores, count, minimum_separation_hz):
    peak_indices = local_peak_indices(scores)
    if peak_indices.size == 0:
        return []

    ordered = peak_indices[np.argsort(scores[peak_indices])[::-1]]
    selected = []

    for index in ordered:
        frequency = float(frequencies[index])
        if all(
            abs(frequency - item["frequency_hz"]) >= minimum_separation_hz
            for item in selected
        ):
            selected.append(
                {
                    "frequency_hz": frequency,
                    "score": float(scores[index]),
                }
            )
        if len(selected) >= count:
            break

    return selected


def fft_candidates(series, sample_rate_hz, min_frequency_hz, max_frequency_hz, count):
    centered = series.astype(np.float64) - float(np.mean(series))
    if not np.any(centered):
        return [], np.empty(0), np.empty(0)

    windowed = centered * np.hanning(centered.size)
    spectrum = np.abs(np.fft.rfft(windowed)) ** 2
    frequencies = np.fft.rfftfreq(centered.size, d=1.0 / sample_rate_hz)

    band = (
        (frequencies >= min_frequency_hz)
        & (frequencies <= max_frequency_hz)
    )
    band_frequencies = frequencies[band]
    band_spectrum = spectrum[band]

    if band_spectrum.size == 0:
        return [], band_frequencies, band_spectrum

    maximum = float(band_spectrum.max())
    normalized = band_spectrum / maximum if maximum > 0 else band_spectrum
    resolution_hz = sample_rate_hz / centered.size

    candidates = select_separated_peaks(
        band_frequencies,
        normalized,
        count,
        minimum_separation_hz=max(1.0, 3.0 * resolution_hz),
    )
    return candidates, band_frequencies, normalized


def autocorrelation_candidates(
    series,
    sample_rate_hz,
    min_frequency_hz,
    max_frequency_hz,
    count,
):
    centered = series.astype(np.float64) - float(np.mean(series))
    if not np.any(centered):
        return [], np.empty(0), np.empty(0)

    transform_size = 1 << (2 * centered.size - 1).bit_length()
    spectrum = np.fft.rfft(centered, n=transform_size)
    autocorrelation = np.fft.irfft(
        spectrum * np.conjugate(spectrum),
        n=transform_size,
    )[: centered.size]

    overlap = np.arange(centered.size, 0, -1, dtype=np.float64)
    autocorrelation /= overlap

    zero_lag = float(autocorrelation[0])
    if zero_lag > 0:
        autocorrelation /= zero_lag

    minimum_lag = max(1, int(math.ceil(sample_rate_hz / max_frequency_hz)))
    maximum_lag = min(
        centered.size - 1,
        int(math.floor(sample_rate_hz / min_frequency_hz)),
    )

    if maximum_lag <= minimum_lag:
        return [], np.empty(0), np.empty(0)

    lags = np.arange(minimum_lag, maximum_lag + 1)
    scores = autocorrelation[lags]
    frequencies = sample_rate_hz / lags

    peak_indices = local_peak_indices(scores)
    if peak_indices.size == 0:
        return [], frequencies, scores

    ordered = peak_indices[np.argsort(scores[peak_indices])[::-1]]
    candidates = []

    for index in ordered:
        frequency = float(frequencies[index])
        if all(
            abs(frequency - item["frequency_hz"]) >= 1.0
            for item in candidates
        ):
            candidates.append(
                {
                    "frequency_hz": frequency,
                    "score": float(scores[index]),
                    "period_us": float(1_000_000.0 / frequency),
                }
            )
        if len(candidates) >= count:
            break

    return candidates, frequencies, scores


def analyze_series(
    series,
    sample_rate_hz,
    min_frequency_hz,
    max_frequency_hz,
    candidate_count,
):
    fft_result, _, _ = fft_candidates(
        series,
        sample_rate_hz,
        min_frequency_hz,
        max_frequency_hz,
        candidate_count,
    )
    autocorrelation_result, _, _ = autocorrelation_candidates(
        series,
        sample_rate_hz,
        min_frequency_hz,
        max_frequency_hz,
        candidate_count,
    )
    return {
        "events": int(series.sum()),
        "nonzero_bins": int(np.count_nonzero(series)),
        "mean_events_per_bin": float(np.mean(series)),
        "fft_candidates": fft_result,
        "autocorrelation_candidates": autocorrelation_result,
    }


def main():
    args = parse_args()

    if args.bin_us <= 0 or args.slice_us <= 0:
        raise ValueError("Time bins and slices must be positive")
    if args.roi_radius_px <= 0:
        raise ValueError("--roi-radius-px must be positive")
    if args.skip_start_ms < 0 or args.skip_end_ms < 0:
        raise ValueError("Skip durations cannot be negative")
    if args.min_frequency_hz <= 0:
        raise ValueError("--min-frequency-hz must be positive")
    if args.max_frequency_hz <= args.min_frequency_hz:
        raise ValueError("Maximum frequency must exceed minimum frequency")
    if args.top_candidates <= 0:
        raise ValueError("--top-candidates must be positive")

    input_path = args.input.resolve()
    detections_path = args.detections.resolve()
    output_prefix = args.output_prefix.resolve()
    output_prefix.parent.mkdir(parents=True, exist_ok=True)

    with detections_path.open("r", encoding="utf-8") as stream:
        detection_report = json.load(stream)

    detections = detection_report["detections"]
    if len(detections) != 3:
        raise ValueError(f"Expected three detections, found {len(detections)}")

    iterator = EventsIterator(
        input_path=str(input_path),
        mode="delta_t",
        delta_t=args.slice_us,
        relative_timestamps=False,
    )

    height, width = iterator.get_size()
    expected_geometry = detection_report["geometry"]
    if (
        width != expected_geometry["width"]
        or height != expected_geometry["height"]
    ):
        raise ValueError("RAW file and detection report have different geometries")

    initial_bins = 4096
    on_series = [
        np.zeros(initial_bins, dtype=np.int64) for _ in detections
    ]
    off_series = [
        np.zeros(initial_bins, dtype=np.int64) for _ in detections
    ]

    total_decoded_events = 0
    first_t = None
    last_t = None
    next_progress = 10_000_000
    radius_squared = args.roi_radius_px ** 2
    wall_start = time.monotonic()

    for events in iterator:
        if events.size == 0:
            continue

        if first_t is None:
            first_t = int(events["t"][0])
        last_t = int(events["t"][-1])
        total_decoded_events += int(events.size)

        event_x = events["x"].astype(np.float64, copy=False)
        event_y = events["y"].astype(np.float64, copy=False)
        event_p = events["p"] != 0
        event_bins = (events["t"] // args.bin_us).astype(np.int64, copy=False)

        for detection_index, detection in enumerate(detections):
            delta_x = event_x - float(detection["center_x"])
            delta_y = event_y - float(detection["center_y"])
            inside = delta_x * delta_x + delta_y * delta_y <= radius_squared

            if not np.any(inside):
                continue

            roi_bins = event_bins[inside]
            roi_polarity = event_p[inside]
            required_length = int(roi_bins.max()) + 1
            ensure_length(on_series[detection_index], required_length)
            ensure_length(off_series[detection_index], required_length)

            if np.any(roi_polarity):
                positive_bins = roi_bins[roi_polarity]
                positive_counts = np.bincount(positive_bins)
                on_series[detection_index][: positive_counts.size] += positive_counts

            if np.any(~roi_polarity):
                negative_bins = roi_bins[~roi_polarity]
                negative_counts = np.bincount(negative_bins)
                off_series[detection_index][: negative_counts.size] += negative_counts

        if total_decoded_events >= next_progress:
            print(f"progress_events={total_decoded_events}", flush=True)
            while total_decoded_events >= next_progress:
                next_progress += 10_000_000

    if first_t is None or last_t is None:
        raise RuntimeError("RAW file contains no CD events")

    start_t = first_t + int(round(args.skip_start_ms * 1000.0))
    end_t = last_t - int(round(args.skip_end_ms * 1000.0))
    if end_t <= start_t:
        raise ValueError("Skip intervals remove the entire recording")

    first_bin = int(math.ceil(start_t / args.bin_us))
    last_bin_exclusive = int(math.floor(end_t / args.bin_us))
    if last_bin_exclusive <= first_bin:
        raise ValueError("No complete analysis bins remain")

    sample_rate_hz = 1_000_000.0 / args.bin_us
    nyquist_hz = sample_rate_hz / 2.0
    if args.max_frequency_hz >= nyquist_hz:
        raise ValueError(
            f"--max-frequency-hz must be below Nyquist frequency {nyquist_hz}"
        )

    report_detections = []
    output_arrays = {}

    for detection_index, detection in enumerate(detections):
        ensure_length(on_series[detection_index], last_bin_exclusive)
        ensure_length(off_series[detection_index], last_bin_exclusive)
        on = on_series[detection_index][first_bin:last_bin_exclusive]
        off = off_series[detection_index][first_bin:last_bin_exclusive]

        result = {
            "component": detection_index,
            "center_x": detection["center_x"],
            "center_y": detection["center_y"],
            "roi_radius_px": args.roi_radius_px,
            "on": analyze_series(
                on,
                sample_rate_hz,
                args.min_frequency_hz,
                args.max_frequency_hz,
                args.top_candidates,
            ),
            "off": analyze_series(
                off,
                sample_rate_hz,
                args.min_frequency_hz,
                args.max_frequency_hz,
                args.top_candidates,
            ),
        }
        report_detections.append(result)
        output_arrays[f"component_{detection_index}_on"] = on
        output_arrays[f"component_{detection_index}_off"] = off

    report = {
        "input": str(input_path),
        "detections": str(detections_path),
        "geometry": {"width": width, "height": height},
        "decoded_events": total_decoded_events,
        "first_t_us": first_t,
        "last_t_us": last_t,
        "analysis_start_t_us": first_bin * args.bin_us,
        "analysis_end_t_us": last_bin_exclusive * args.bin_us,
        "analysis_duration_s": (
            (last_bin_exclusive - first_bin) * args.bin_us / 1_000_000.0
        ),
        "bin_us": args.bin_us,
        "sample_rate_hz": sample_rate_hz,
        "min_frequency_hz": args.min_frequency_hz,
        "max_frequency_hz": args.max_frequency_hz,
        "processing_wall_s": time.monotonic() - wall_start,
        "components": report_detections,
    }

    json_path = Path(f"{output_prefix}.json")
    npz_path = Path(f"{output_prefix}.npz")

    with json_path.open("w", encoding="utf-8") as stream:
        json.dump(report, stream, indent=2, sort_keys=True)
        stream.write("\n")

    np.savez_compressed(
        npz_path,
        bin_us=np.int64(args.bin_us),
        sample_rate_hz=np.float64(sample_rate_hz),
        **output_arrays,
    )

    print(f"decoded_events={total_decoded_events}")
    print(f"analysis_duration_s={report['analysis_duration_s']:.6f}")
    for component in report_detections:
        on_fft = component["on"]["fft_candidates"]
        off_fft = component["off"]["fft_candidates"]
        print(
            f"component={component['component']} "
            f"center=({component['center_x']:.3f},{component['center_y']:.3f})"
        )
        print(f"component={component['component']} on_fft={on_fft[:5]}")
        print(f"component={component['component']} off_fft={off_fft[:5]}")
    print(f"json={json_path}")
    print(f"npz={npz_path}")
    print("LED frequency analysis: PASS")


if __name__ == "__main__":
    main()
