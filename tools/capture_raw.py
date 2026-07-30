#!/usr/bin/env python3

import argparse
import time
from pathlib import Path

from metavision_hal import DeviceDiscovery


def parse_args():
    parser = argparse.ArgumentParser(
        description="Headless, fixed-duration EVK4 RAW recorder."
    )
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--duration-s", required=True, type=float)
    parser.add_argument("--serial", required=True)
    return parser.parse_args()


def main():
    args = parse_args()

    if args.duration_s <= 0:
        raise ValueError("--duration-s must be positive")

    output = args.output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)

    if output.exists():
        raise FileExistsError(
            f"Refusing to overwrite existing recording: {output}"
        )

    device = DeviceDiscovery.open(args.serial)
    if device is None:
        raise RuntimeError(f"Could not open camera serial {args.serial}")

    geometry = device.get_i_geometry()
    cd_decoder = device.get_i_event_cd_decoder()
    stream_decoder = device.get_i_events_stream_decoder()
    event_stream = device.get_i_events_stream()

    if geometry is None:
        raise RuntimeError("Camera has no geometry facility")
    if cd_decoder is None:
        raise RuntimeError("Camera has no CD-event decoder")
    if stream_decoder is None:
        raise RuntimeError("Camera has no event-stream decoder")
    if event_stream is None:
        raise RuntimeError("Camera has no event-stream facility")

    statistics = {
        "events": 0,
        "on_events": 0,
        "off_events": 0,
        "first_t": None,
        "last_t": None,
        "min_x": None,
        "max_x": None,
        "min_y": None,
        "max_y": None,
    }

    def process_cd_events(events):
        if events.size == 0:
            return

        statistics["events"] += int(events.size)
        statistics["on_events"] += int(events["p"].sum())
        statistics["off_events"] += int(events.size - events["p"].sum())

        first_t = int(events["t"][0])
        last_t = int(events["t"][-1])

        if statistics["first_t"] is None:
            statistics["first_t"] = first_t
        statistics["last_t"] = last_t

        min_x = int(events["x"].min())
        max_x = int(events["x"].max())
        min_y = int(events["y"].min())
        max_y = int(events["y"].max())

        statistics["min_x"] = (
            min_x
            if statistics["min_x"] is None
            else min(statistics["min_x"], min_x)
        )
        statistics["max_x"] = (
            max_x
            if statistics["max_x"] is None
            else max(statistics["max_x"], max_x)
        )
        statistics["min_y"] = (
            min_y
            if statistics["min_y"] is None
            else min(statistics["min_y"], min_y)
        )
        statistics["max_y"] = (
            max_y
            if statistics["max_y"] is None
            else max(statistics["max_y"], max_y)
        )

    cd_decoder.add_event_buffer_callback(process_cd_events)

    if not event_stream.log_raw_data(str(output)):
        raise RuntimeError(f"Could not start RAW recording: {output}")

    started = False
    wall_start = time.monotonic()

    try:
        event_stream.start()
        started = True
        deadline = wall_start + args.duration_s

        while time.monotonic() < deadline:
            status = event_stream.poll_buffer()

            if status < 0:
                raise RuntimeError("Camera event stream stopped unexpectedly")

            if status == 0:
                time.sleep(0.001)
                continue

            raw_data = event_stream.get_latest_raw_data()
            if raw_data is not None:
                stream_decoder.decode(raw_data)
    finally:
        event_stream.stop_log_raw_data()
        if started:
            event_stream.stop()

    wall_duration = time.monotonic() - wall_start
    event_rate = statistics["events"] / wall_duration

    print(f"output={output}")
    print(f"camera_serial={args.serial}")
    print(f"geometry={geometry.get_width()}x{geometry.get_height()}")
    print(f"wall_duration_s={wall_duration:.6f}")
    print(f"events={statistics['events']}")
    print(f"event_rate_eps={event_rate:.3f}")
    print(f"on_events={statistics['on_events']}")
    print(f"off_events={statistics['off_events']}")
    print(f"first_t_us={statistics['first_t']}")
    print(f"last_t_us={statistics['last_t']}")
    print(
        "event_bounds="
        f"x[{statistics['min_x']},{statistics['max_x']}] "
        f"y[{statistics['min_y']},{statistics['max_y']}]"
    )


if __name__ == "__main__":
    main()
