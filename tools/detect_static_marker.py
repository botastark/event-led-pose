#!/usr/bin/env python3

import argparse
import json
import math
from collections import deque
from pathlib import Path

import numpy as np


def parse_args():
    parser = argparse.ArgumentParser(
        description="Detect static blinking-marker regions using marker-minus-ambient event-rate maps."
    )
    parser.add_argument("--ambient", required=True, type=Path)
    parser.add_argument("--marker", required=True, type=Path)
    parser.add_argument("--output-prefix", required=True, type=Path)
    parser.add_argument("--block-size", type=int, default=4)
    parser.add_argument("--block-threshold-eps", type=float, default=800.0)
    parser.add_argument("--min-component-blocks", type=int, default=20)
    parser.add_argument("--expected-components", type=int, default=3)
    return parser.parse_args()


def scalar(data, key):
    return int(np.asarray(data[key]).item())


def recording_duration_s(data):
    duration_us = scalar(data, "last_t_us") - scalar(data, "first_t_us")
    if duration_us <= 0:
        raise ValueError("Recording duration must be positive")
    return duration_us / 1_000_000.0


def connected_components(mask):
    height, width = mask.shape
    visited = np.zeros(mask.shape, dtype=bool)
    components = []

    for start_y, start_x in zip(*np.nonzero(mask)):
        if visited[start_y, start_x]:
            continue

        queue = deque([(int(start_y), int(start_x))])
        visited[start_y, start_x] = True
        component = []

        while queue:
            y, x = queue.popleft()
            component.append((y, x))

            for dy in (-1, 0, 1):
                for dx in (-1, 0, 1):
                    if dx == 0 and dy == 0:
                        continue

                    neighbor_y = y + dy
                    neighbor_x = x + dx

                    if (
                        0 <= neighbor_y < height
                        and 0 <= neighbor_x < width
                        and mask[neighbor_y, neighbor_x]
                        and not visited[neighbor_y, neighbor_x]
                    ):
                        visited[neighbor_y, neighbor_x] = True
                        queue.append((neighbor_y, neighbor_x))

        components.append(component)

    return components


def component_statistics(component, positive_rate, block_size):
    weighted_x = 0.0
    weighted_y = 0.0
    signal_rate_eps = 0.0

    min_block_y = min(y for y, _ in component)
    max_block_y = max(y for y, _ in component)
    min_block_x = min(x for _, x in component)
    max_block_x = max(x for _, x in component)

    for block_y, block_x in component:
        y0 = block_y * block_size
        x0 = block_x * block_size
        block = positive_rate[
            y0 : y0 + block_size,
            x0 : x0 + block_size,
        ]

        ys, xs = np.indices(block.shape)
        block_signal = float(block.sum())
        signal_rate_eps += block_signal
        weighted_x += float(((xs + x0) * block).sum())
        weighted_y += float(((ys + y0) * block).sum())

    if signal_rate_eps <= 0:
        raise RuntimeError("Detected component has no positive signal")

    return {
        "block_count": len(component),
        "signal_rate_eps": signal_rate_eps,
        "center_x": weighted_x / signal_rate_eps,
        "center_y": weighted_y / signal_rate_eps,
        "bbox": {
            "min_x": min_block_x * block_size,
            "min_y": min_block_y * block_size,
            "max_x": (max_block_x + 1) * block_size - 1,
            "max_y": (max_block_y + 1) * block_size - 1,
        },
    }


def grayscale_log_image(values):
    log_values = np.log1p(np.maximum(values, 0.0))
    nonzero = log_values[log_values > 0]

    if nonzero.size == 0:
        return np.zeros(values.shape, dtype=np.uint8)

    clipping_value = float(np.percentile(nonzero, 99.9))
    if clipping_value <= 0:
        clipping_value = float(nonzero.max())

    return np.rint(
        np.clip(log_values / clipping_value, 0.0, 1.0) * 255.0
    ).astype(np.uint8)


def draw_detection_image(path, positive_rate, detections):
    gray = grayscale_log_image(positive_rate)
    image = np.repeat(gray[:, :, None], 3, axis=2)
    height, width = gray.shape

    colors = (
        np.array([255, 0, 0], dtype=np.uint8),
        np.array([0, 255, 0], dtype=np.uint8),
        np.array([0, 128, 255], dtype=np.uint8),
    )

    for index, detection in enumerate(detections):
        color = colors[index % len(colors)]
        center_x = int(round(detection["center_x"]))
        center_y = int(round(detection["center_y"]))
        bbox = detection["bbox"]

        min_x = max(0, bbox["min_x"])
        max_x = min(width - 1, bbox["max_x"])
        min_y = max(0, bbox["min_y"])
        max_y = min(height - 1, bbox["max_y"])

        image[min_y, min_x : max_x + 1] = color
        image[max_y, min_x : max_x + 1] = color
        image[min_y : max_y + 1, min_x] = color
        image[min_y : max_y + 1, max_x] = color

        cross_radius = 8
        image[
            center_y,
            max(0, center_x - cross_radius) : min(width, center_x + cross_radius + 1),
        ] = color
        image[
            max(0, center_y - cross_radius) : min(height, center_y + cross_radius + 1),
            center_x,
        ] = color

    header = f"P6\n{width} {height}\n255\n"
    with path.open("wb") as stream:
        stream.write(header.encode("ascii"))
        stream.write(image.tobytes(order="C"))


def main():
    args = parse_args()

    if args.block_size <= 0:
        raise ValueError("--block-size must be positive")
    if args.block_threshold_eps <= 0:
        raise ValueError("--block-threshold-eps must be positive")
    if args.min_component_blocks <= 0:
        raise ValueError("--min-component-blocks must be positive")
    if args.expected_components <= 0:
        raise ValueError("--expected-components must be positive")

    ambient_path = args.ambient.resolve()
    marker_path = args.marker.resolve()
    output_prefix = args.output_prefix.resolve()
    output_prefix.parent.mkdir(parents=True, exist_ok=True)

    with np.load(ambient_path) as ambient, np.load(marker_path) as marker:
        ambient_map = ambient["total_map"].astype(np.float64)
        marker_map = marker["total_map"].astype(np.float64)

        if ambient_map.shape != marker_map.shape:
            raise ValueError("Ambient and marker maps have different geometries")

        height, width = marker_map.shape
        if height % args.block_size or width % args.block_size:
            raise ValueError("Image dimensions must be divisible by --block-size")

        ambient_duration_s = recording_duration_s(ambient)
        marker_duration_s = recording_duration_s(marker)

    signed_rate_difference = (
        marker_map / marker_duration_s - ambient_map / ambient_duration_s
    )
    positive_rate = np.maximum(signed_rate_difference, 0.0)

    block_rate = positive_rate.reshape(
        height // args.block_size,
        args.block_size,
        width // args.block_size,
        args.block_size,
    ).sum(axis=(1, 3))

    block_mask = block_rate >= args.block_threshold_eps
    raw_components = connected_components(block_mask)

    detections = []
    for component in raw_components:
        if len(component) < args.min_component_blocks:
            continue
        detections.append(
            component_statistics(component, positive_rate, args.block_size)
        )

    detections.sort(key=lambda item: item["signal_rate_eps"], reverse=True)

    if len(detections) != args.expected_components:
        raise RuntimeError(
            f"Expected {args.expected_components} components, detected {len(detections)}. "
            "Review --block-threshold-eps and --min-component-blocks."
        )

    center_x = sum(item["center_x"] for item in detections) / len(detections)
    center_y = sum(item["center_y"] for item in detections) / len(detections)

    pairwise_distances = []
    for first_index in range(len(detections)):
        for second_index in range(first_index + 1, len(detections)):
            first = detections[first_index]
            second = detections[second_index]
            pairwise_distances.append(
                {
                    "first": first_index,
                    "second": second_index,
                    "distance_px": math.hypot(
                        first["center_x"] - second["center_x"],
                        first["center_y"] - second["center_y"],
                    ),
                }
            )

    report = {
        "ambient": str(ambient_path),
        "marker": str(marker_path),
        "geometry": {"width": width, "height": height},
        "ambient_duration_s": ambient_duration_s,
        "marker_duration_s": marker_duration_s,
        "block_size": args.block_size,
        "block_threshold_eps": args.block_threshold_eps,
        "min_component_blocks": args.min_component_blocks,
        "positive_difference_rate_eps": float(positive_rate.sum()),
        "marker_center_px": {"x": center_x, "y": center_y},
        "detections": detections,
        "pairwise_distances": pairwise_distances,
    }

    json_path = Path(f"{output_prefix}.json")
    npz_path = Path(f"{output_prefix}.npz")
    image_path = Path(f"{output_prefix}_detections.ppm")

    with json_path.open("w", encoding="utf-8") as stream:
        json.dump(report, stream, indent=2, sort_keys=True)
        stream.write("\n")

    np.savez_compressed(
        npz_path,
        signed_rate_difference=signed_rate_difference,
        positive_rate=positive_rate,
        block_rate=block_rate,
        block_mask=block_mask,
    )

    draw_detection_image(image_path, positive_rate, detections)

    print(f"positive_difference_rate_eps={positive_rate.sum():.3f}")
    for index, detection in enumerate(detections):
        print(
            f"component={index} "
            f"center=({detection['center_x']:.3f},{detection['center_y']:.3f}) "
            f"signal_rate_eps={detection['signal_rate_eps']:.3f} "
            f"blocks={detection['block_count']}"
        )
    print(f"marker_center=({center_x:.3f},{center_y:.3f})")
    print(f"json={json_path}")
    print(f"npz={npz_path}")
    print(f"detection_image={image_path}")
    print("Static marker detection: PASS")


if __name__ == "__main__":
    main()
