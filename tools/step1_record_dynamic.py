#!/usr/bin/env python3
"""Step 2c: Record a dynamic scene (LEDs stationary + lots of background
motion, e.g. waving hands/objects in front of camera, moving the rig
mount, etc.) at the current tuned biases, for offline gating analysis.

Usage:
    python3 step2c_record_dynamic.py --serial 00050946 --seconds 8 \
        --diff-off 190 --diff-on 140
Produces: raw_dynamic.raw
"""
from __future__ import annotations
import argparse
import sys
import time

from metavision_core.event_io import EventsIterator


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--serial", default="00050946")
    p.add_argument("--seconds", type=float, default=8.0)
    p.add_argument("--diff-off", type=int, default=190)
    p.add_argument("--diff-on", type=int, default=140)
    p.add_argument("--bias-refr", type=int, default=55)
    p.add_argument("--out", default="raw_dynamic.raw")
    return p.parse_args()


def main() -> int:
    args = parse_args()

    stream = EventsIterator(
        input_path=args.serial, mode="delta_t", delta_t=50_000,
        relative_timestamps=False,
    )
    device = stream.reader.device
    biases = device.get_i_ll_biases()
    for name, val in (("bias_diff_off", args.diff_off),
                      ("bias_diff_on", args.diff_on),
                      ("bias_refr", args.bias_refr)):
        biases.set(name, val)
    time.sleep(0.3)

    i_events_stream = device.get_i_events_stream()
    if i_events_stream is None:
        raise RuntimeError("I_EventsStream facility not available; cannot record raw.")

    i_events_stream.log_raw_data(args.out)
    print(f"Recording DYNAMIC scene to {args.out} for {args.seconds:.1f}s ...")
    print("Move objects / hands / camera-adjacent clutter through the FOV now, "
          "while keeping the 3 LEDs themselves stationary and visible.")
    t0 = time.monotonic()
    for _ in stream:
        if time.monotonic() - t0 >= args.seconds:
            break
    i_events_stream.stop_log_raw_data()
    print(f"Saved {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
