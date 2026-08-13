#!/usr/bin/env python3
"""Step 0b (fixed): Check bias state and measure real background event rate.

Avoids the double-open device lock conflict by reading biases from the SAME
device handle that EventsIterator uses internally (via stream.reader.device),
instead of opening a second independent handle with DeviceDiscovery.

Usage:
    python3 step0b_check_bias_file_and_rate.py --serial 00050946 --seconds 3
"""
from __future__ import annotations
import argparse
import sys
import time

import numpy as np
from metavision_core.event_io import EventsIterator


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--serial", default="00050946")
    p.add_argument("--seconds", type=float, default=3.0)
    p.add_argument("--bias-file", default=None,
                    help="Optional path to a .bias file to load before measuring.")
    return p.parse_args()


def main() -> int:
    args = parse_args()

    print(f"Opening stream on serial={args.serial} ...")
    stream = EventsIterator(
        input_path=args.serial,
        mode="delta_t",
        delta_t=50_000,
        relative_timestamps=False,
    )
    height, width = stream.get_size()
    print(f"sensor={width}x{height}")

    # Reuse the SAME device handle the iterator already opened.
    device = getattr(stream.reader, "device", None)
    if device is None:
        print("Could not access underlying device handle from EventsIterator; "
              "skipping bias inspection, proceeding to rate measurement only.")
    else:
        biases = device.get_i_ll_biases()
        if biases is not None:
            if args.bias_file:
                try:
                    biases.load_from_file(args.bias_file)
                    print(f"Loaded bias file: {args.bias_file}")
                except Exception as exc:
                    print(f"Could not load bias file: {exc}")
            print("Biases (from live streaming device handle):")
            for name, value in biases.get_all_biases().items():
                print(f"  {name:15s} = {value}")
        else:
            print("No I_LL_Biases facility on this device handle.")

    print(f"\nStreaming raw events for {args.seconds:.1f}s to measure real background rate...")
    total_events = 0
    on_events = 0
    off_events = 0
    t_start = time.monotonic()
    n_slices = 0

    for events in stream:
        n_slices += 1
        if events.size:
            total_events += int(events.size)
            on_events += int(np.count_nonzero(events["p"] == 1))
            off_events += int(np.count_nonzero(events["p"] == 0))
        if time.monotonic() - t_start >= args.seconds:
            break

    elapsed = time.monotonic() - t_start
    rate = total_events / max(elapsed, 1e-9)
    print(f"\n== Background event rate over {elapsed:.2f}s ({n_slices} slices) ==")
    print(f"  total_events = {total_events:,}")
    print(f"  ON events    = {on_events:,}")
    print(f"  OFF events   = {off_events:,}")
    print(f"  rate         = {rate:,.0f} events/sec (whole sensor)")
    print(f"  rate/pixel   = {rate / (width*height):.4f} events/sec/pixel")

    return 0


if __name__ == "__main__":
    sys.exit(main())
