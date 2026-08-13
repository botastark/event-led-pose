#!/usr/bin/env python3
"""Step 2a: Record two short raw files -- one at baseline (all-zero) biases,
one at the tuned biases -- so we can visually diff what biases removed.

Usage:
    python3 step2a_record_before_after.py --serial 00050946 --seconds 5
Produces: raw_baseline.raw, raw_tuned.raw
"""
from __future__ import annotations
import argparse
import sys
import time

from metavision_core.event_io import EventsIterator


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--serial", default="00050946")
    p.add_argument("--seconds", type=float, default=5.0)
    p.add_argument("--diff-off", type=int, default=190)
    p.add_argument("--diff-on", type=int, default=140)
    p.add_argument("--baseline-out", default="raw_baseline.raw")
    p.add_argument("--tuned-out", default="raw_tuned.raw")
    return p.parse_args()


def record(serial: str, seconds: float, out_path: str, set_biases=None) -> None:
    stream = EventsIterator(
        input_path=serial, mode="delta_t", delta_t=50_000,
        relative_timestamps=False,
    )
    device = stream.reader.device
    i_events_stream = device.get_i_events_stream()
    if i_events_stream is None:
        raise RuntimeError("I_EventsStream facility not available; cannot record raw.")

    if set_biases:
        biases = device.get_i_ll_biases()
        for name, val in set_biases.items():
            biases.set(name, val)
        time.sleep(0.3)

    i_events_stream.log_raw_data(out_path)
    print(f"Recording to {out_path} for {seconds:.1f}s ...")
    t0 = time.monotonic()
    for events in stream:
        if time.monotonic() - t0 >= seconds:
            break
    i_events_stream.stop_log_raw_data()
    print(f"Saved {out_path}")


def main() -> int:
    args = parse_args()

    print("=== Recording BASELINE (all biases = 0) ===")
    record(args.serial, args.seconds, args.baseline_out,
           set_biases={"bias_diff_off": 0, "bias_diff_on": 0, "bias_refr": 0})

    time.sleep(1.0)

    print("\n=== Recording TUNED biases ===")
    record(args.serial, args.seconds, args.tuned_out,
           set_biases={"bias_diff_off": args.diff_off, "bias_diff_on": args.diff_on})

    return 0


if __name__ == "__main__":
    sys.exit(main())
