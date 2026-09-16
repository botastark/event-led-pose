# Event LED Pose

Low-latency frequency-based detection of a rigid three-LED marker using a Prophesee EVK4 / Sony IMX636 event camera.

Each LED flashes at a different nominal frequency:

| LED ID    | Frequency | Display color |
| --------- | --------: | ------------- |
| `led_165` |    165 Hz | Green         |
| `led_366` |    366 Hz | Yellow        |
| `led_596` |    596 Hz | Blue          |

The current implementation is C++ only and focuses on low-latency event-by-event frequency filtering using event polarity and per-pixel timestamp history.

Metric 6-DoF pose estimation is not implemented yet.

## Requirements

The C++ targets use:

- C++17;
- CMake;
- Metavision SDK / OpenEB 5.3;
- Metavision SDK components:
  - `core`
  - `stream`
  - `ui`

## Build and run

using Makerfiles

```bash

make build
make rebuild
make run
make probe
make shell
make clean

```

The visualization colors accepted events according to their detected frequency:

- 165 Hz: green
- 366 Hz: yellow
- 596 Hz: blue

For latency and throughput measurements, run without visualization.
otherwise, use:

```bash

rm -rf $(BUILD_DIR)
cmake -S . -B $(BUILD_DIR) -DCMAKE_BUILD_TYPE=$(BUILD_TYPE)
cmake --build $(BUILD_DIR) -j$(JOBS)
./$(BUILD_DIR)/live_frequency_ring --visualize --display-fps 120

```

## Frequency-filter design

The pipeline uses two different kinds of rings. A transport ring moves packed EVK4 events from the camera callback to the filtering thread without blocking the camera; if the filter falls behind, stale transport events can be skipped to keep latency low.

```text

EVK4 callback
     |
     | packed events
     v
+------------------+
| transport ring   |
+------------------+
     |
     v
frequency worker

```

Inside the filter, each pixel keeps two tiny polarity-specific timing rings: one for OFF→ON transitions and one for ON→OFF transitions. Each ring stores the latest two valid same-polarity intervals.

```text

OFF→ON ---- T ---- OFF→ON
   +                  +

ON→OFF ---- T ---- ON→OFF
   -                  -

```

These intervals are compared against the known LED periods for 165, 366, and 596 Hz. Missed-cycle multiples are allowed where configured; 165 Hz currently uses a stricter set to reduce fast-motion false positives.

```text

per pixel

rise ring: [dt1][dt2]  -> OFF→ON periods
fall ring: [dt1][dt2]  -> ON→OFF periods
                 |
                 v
        compare with known T
                 |
        165 / 366 / 596 Hz

```

A frequency is accepted only after coherent evidence from both polarities. Each polarity keeps two matching intervals; a mismatch clears that polarity's evidence, and stale candidates are expired with a short timeout. This prevents isolated background matches from accumulating over time.
