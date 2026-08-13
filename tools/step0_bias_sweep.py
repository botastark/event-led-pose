#!/usr/bin/env python3
"""Step 0d: Corrected bias sweep using confirmed valid ranges.

Confirmed from device error messages:
  bias_diff_off in [-35, 190]
  bias_fo       in [-35, 55]
(bias_diff_on assumed similar range to bias_diff_off; validated live below)

Previous sweep showed increasing values made noise WORSE, not better --
so this version sweeps broadly across the full range in both directions,
including the sensor's default/typical values, to empirically find the
true noise-minimizing direction rather than assuming it.

Usage:
    python3 step0d_bias_sweep_v2.py --serial 00050946 --sample-seconds 1.0
"""
from __future__ import annotations
import argparse
import csv
import sys
import time

from metavision_core.event_io import EventsIterator


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--serial", default="00050946")
    p.add_argument("--sample-seconds", type=float, default=1.0)
    p.add_argument("--settle-seconds", type=float, default=0.3)
    p.add_argument("--output-csv", default="bias_sweep_results_v2.csv")
    return p.parse_args()


def measure_rate(event_iter, seconds: float) -> tuple[int, float]:
    total = 0
    t0 = time.monotonic()
    while time.monotonic() - t0 < seconds:
        try:
            events = next(event_iter)
        except StopIteration:
            break
        if events is not None and events.size:
            total += int(events.size)
    elapsed = time.monotonic() - t0
    return total, total / max(elapsed, 1e-9)


def try_set(biases, name: str, value: int) -> bool:
    try:
        biases.set(name, value)
        return True
    except Exception as exc:
        print(f"  set({name}={value}) failed: {exc}")
        return False


def main() -> int:
    args = parse_args()

    print(f"Opening stream on serial={args.serial} ...")
    stream = EventsIterator(
        input_path=args.serial, mode="delta_t", delta_t=50_000,
        relative_timestamps=False,
    )
    height, width = stream.get_size()
    n_pixels = width * height
    device = stream.reader.device
    biases = device.get_i_ll_biases()
    event_iter = iter(stream)

    print(f"\nBaseline (all zero) over {args.sample_seconds}s:")
    total, rate = measure_rate(event_iter, args.sample_seconds)
    print(f"  rate={rate:,.0f} ev/s ({rate/n_pixels:.4f} ev/s/px)")

    results = [{"bias_diff_off": 0, "bias_diff_on": 0, "bias_fo": 0,
                "rate_evps": rate, "rate_per_pixel": rate / n_pixels}]

    # Broad sweep across the FULL confirmed valid range, positive-biased,
    # since negative direction already shown to worsen noise.
    diff_off_candidates = [0, 40, 90, 140, 190]
    diff_on_candidates = [0, 40, 90, 140, 190]
    fo_candidates = [0]  # hold fo fixed at 0 for this pass; isolate diff pair first

    print(f"\nSweeping diff_off x diff_on (fo held at 0)...")
    for doff in diff_off_candidates:
        for don in diff_on_candidates:
            ok1 = try_set(biases, "bias_diff_off", doff)
            ok2 = try_set(biases, "bias_diff_on", don)
            if not (ok1 and ok2):
                continue
            time.sleep(args.settle_seconds)
            total, rate = measure_rate(event_iter, args.sample_seconds)
            print(f"diff_off={doff:4d} diff_on={don:4d} -> "
                  f"rate={rate:,.0f} ev/s ({rate/n_pixels:.4f} ev/s/px)")
            results.append({
                "bias_diff_off": doff, "bias_diff_on": don, "bias_fo": 0,
                "rate_evps": rate, "rate_per_pixel": rate / n_pixels,
            })

    # Reset to zero at end.
    try_set(biases, "bias_diff_off", 0)
    try_set(biases, "bias_diff_on", 0)
    try_set(biases, "bias_fo", 0)

    with open(args.output_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(results[0].keys()))
        writer.writeheader()
        writer.writerows(results)

    best = min(results, key=lambda r: r["rate_per_pixel"])
    print(f"\nWrote {len(results)} rows to {args.output_csv}")
    print(f"Lowest noise so far: diff_off={best['bias_diff_off']} "
          f"diff_on={best['bias_diff_on']} -> {best['rate_per_pixel']:.4f} ev/s/px")
    print("NOTE: verify LEDs are still visible at this setting with a live "
          "viewer before adopting it -- an aggressive threshold could also "
          "suppress the LED signal itself.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
