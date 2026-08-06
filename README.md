# Event LED Pose

Low-latency detection and tracking of a rigid three-LED marker using a Prophesee EVK4 / Sony IMX636 event camera.

Each LED flashes at a different nominal frequency:

| LED ID    | Frequency | Display color |
| --------- | --------: | ------------- |
| `led_165` |    165 Hz | Green         |
| `led_366` |    366 Hz | Yellow        |
| `led_596` |    596 Hz | Blue          |

The current implementation detects all three frequencies over the full sensor, estimates their image positions, and visualizes the labelled tracks in real time.

The project does **not** estimate metric 6-DoF pose yet. Pose estimation will require calibrated camera intrinsics and measured three-dimensional LED coordinates.

## Current capabilities

- Rootless Podman runtime with OpenEB 5.3.0.
- Narrow access to one EVK4 USB device.
- Live event visualization.
- Global detection of the 165, 366, and 596 Hz LEDs.
- Frequency-labelled LED centers and tracking trails.
- Automatic reacquisition without a static initialization file.
- Low-latency display using only the newest event buffer.
- Decimated frequency processing to prevent event-stream backlog.
- RAW recording and offline event analysis.
- Offline static marker detection and frequency analysis.

## How the live tracker works

The tracker uses a frequency-selective event response for each known LED frequency.

For every incoming event slice:

1. Event coordinates are reduced to a coarse spatial grid.
2. Event polarity is multiplied by sine and cosine references at 165, 366, and 596 Hz.
3. Exponentially decayed frequency-response maps are updated.
4. A spatial energy peak is found independently for each frequency.
5. The local weighted center of each peak becomes the LED image position.
6. Position estimates are smoothed and displayed with frequency labels.
7. If a lock is lost, the full image remains available for global reacquisition.

This means frequency identity is determined before spatial tracking. The tracker does not simply follow the brightest event clusters.

### Lock criteria

Each LED reports either `LOCKED` or `SEARCH`.

A candidate becomes `LOCKED` when it has:

- enough event support;
- sufficient frequency coherence;
- adequate separation from competing spatial peaks.

Example terminal output:

```text
165Hz LOCKED xy=(612,481) coh=0.214 support=1720 ratio=1.48
366Hz LOCKED xy=(734,492) coh=0.187 support=1395 ratio=1.31
596Hz LOCKED xy=(667,375) coh=0.243 support=1841 ratio=1.62
```

## Low-latency design

The program avoids building a queue of old event data.

- The display is generated directly from the newest event buffer.
- Old event buffers are not accumulated for visualization.
- Window updates are submitted asynchronously.
- The display uses all events from the newest slice.
- Frequency processing uses every Nth event through `--event-stride`.
- Detection and display run at independently configurable rates.

Under temporary CPU load, the visualization may skip an intermediate state, but it should stay close to current camera time rather than showing an increasingly old stream.

## Host requirements

The host should provide:

- amd64 Linux;
- AVX2-capable CPU;
- rootless Podman;
- `crun`;
- USB access to the EVK4;
- an X11 or XWayland display for GUI tools.

Recommended Ubuntu packages:

```bash
sudo apt update
sudo apt install -y \
  git make podman crun uidmap fuse-overlayfs slirp4netns \
  usbutils acl ca-certificates curl x11-xserver-utils
```

OpenEB and its Python dependencies are installed only inside the container. Do not install a second OpenEB environment on the host.

## Build the runtime

Build the pinned runtime image:

```bash
podman build \
  --file container/Containerfile.runtime \
  --tag localhost/event-led-pose/openeb-runtime:5.3.0-r2-ubuntu24 \
  .
```

Query available OpenEB packages when needed:

```bash
make query-openeb
```

The query result is informational. Do not silently change the pinned OpenEB version.

## Check the camera

Connect the EVK4 directly to a USB 3 port and run:

```bash
make check-camera
```

Continue only when:

- exactly one `04b4:00f5` device is found;
- the current user has read and write access;
- the USB topology reports 5000 Mbps;
- the resolved device node exists.

The USB device number may change whenever the camera is disconnected or reset. Reopen the camera container after reconnecting it.

### Optional udev rule

When current-user access is missing, a system administrator may install this narrow rule:

```udev
SUBSYSTEM=="usb", ATTR{idVendor}=="04b4", ATTR{idProduct}=="00f5", MODE="0660", TAG+="uaccess"
```

Do not use a world-writable `0666` rule and do not run the camera software as root.

## Open a camera-enabled container

From the repository root:

```bash
./tools/host/open_camera_container.sh
```

The launcher:

- resolves the current EVK4 USB node;
- passes only that USB device to the container;
- mounts the repository at `/workspace`;
- configures X11 display access;
- opens an interactive shell by default.

Inside the container, the prompt should show the OpenEB environment and `/workspace` as the working directory.

Only one process may own the EVK4 at a time. Close other camera applications before starting another one.

## Test the normal event viewer

From the host:

```bash
./tools/host/live_view.sh
```

Or from inside the camera-enabled container:

```bash
metavision_viewer
```

`metavision_viewer` is useful for checking camera responsiveness independently of the Python tracker.

## Run live three-LED tracking

Use the validated launcher:

```bash
./tools/host/live_frequency_3led.sh
```

The launcher applies the current low-latency defaults and starts the tracker inside the camera-enabled container.

Press `Q` or `Esc` in the tracking window to stop.

### Run directly inside the container

```bash
python3 /workspace/tools/live_frequency_3led.py \
  --slice-us 20000 \
  --event-stride 4 \
  --memory-ms 250 \
  --detection-update-ms 40 \
  --display-fps 30 \
  --display-accumulation-us 10000 \
  --center-alpha 0.75
```

## Important tracking options

| Option                      | Effect                                                    |
| --------------------------- | --------------------------------------------------------- |
| `--slice-us`                | Duration of each incoming event slice                     |
| `--event-stride`            | Uses every Nth event for frequency processing             |
| `--memory-ms`               | Temporal memory of the frequency-response maps            |
| `--detection-update-ms`     | Interval between spatial detection updates                |
| `--display-fps`             | Maximum display update rate                               |
| `--display-accumulation-us` | Visible event persistence in the displayed frame          |
| `--center-alpha`            | Position smoothing; higher follows measurements faster    |
| `--snap-distance-px`        | Distance at which a detection is treated as reacquisition |
| `--minimum-support`         | Minimum local event support for lock                      |
| `--minimum-coherence`       | Minimum normalized frequency response                     |
| `--minimum-peak-ratio`      | Required separation from the next-best spatial peak       |

## Faster motion preset

For lower tracking latency:

```bash
./tools/host/live_frequency_3led.sh \
  --slice-us 10000 \
  --memory-ms 120 \
  --detection-update-ms 10 \
  --display-fps 40 \
  --display-accumulation-us 5000 \
  --center-alpha 0.95 \
  --snap-distance-px 35 \
  --minimum-support 50 \
  --minimum-coherence 0.05 \
  --minimum-peak-ratio 1.03
```

A shorter memory reacts faster because less historical frequency evidence remains behind a moving LED. It also provides less robustness to noise and temporary occlusion.

At 165 Hz:

- 250 ms contains about 41 cycles;
- 120 ms contains about 20 cycles;
- 80 ms contains about 13 cycles.

If locks become unstable, increase `--memory-ms` before changing the frequency thresholds.

## Troubleshooting

### Camera cannot be opened

Example:

```text
OSError: Failed to open camera 00050946
```

Check that no other process owns the camera:

```bash
pkill -f metavision_viewer || true
pkill -f live_frequency_3led.py || true
pkill -f capture_raw.py || true
```

Then test:

```bash
metavision_platform_info
```

If the camera was reconnected, exit the container, run `make check-camera`, and open a new camera-enabled container.

### Display is responsive but tracking lags

Reduce frequency memory and increase position responsiveness:

```bash
./tools/host/live_frequency_3led.sh \
  --memory-ms 120 \
  --detection-update-ms 10 \
  --center-alpha 0.95
```

### Display latency grows continuously

The processing loop is falling behind. Increase event decimation:

```bash
./tools/host/live_frequency_3led.sh --event-stride 8
```

The newest full event slice is still used for display.

### LEDs remain in `SEARCH`

Relax the lock thresholds:

```bash
./tools/host/live_frequency_3led.sh \
  --minimum-support 40 \
  --minimum-coherence 0.04 \
  --minimum-peak-ratio 1.02
```

### Tracker locks to background responses

Use stricter thresholds or a longer memory:

```bash
./tools/host/live_frequency_3led.sh \
  --memory-ms 350 \
  --minimum-coherence 0.12 \
  --minimum-peak-ratio 1.12
```

## Record RAW events

Open the camera-enabled container and create output directories:

```bash
mkdir -p data/raw data/reports
```

Record a sample:

```bash
python3 tools/capture_raw.py \
  --serial 00050946 \
  --duration-s 10 \
  --output data/raw/marker_test_10s.raw
```

Do not run the live viewer or tracker while recording.

### Offline frequency analysis

```bash
python3 tools/analyze_led_frequencies.py \
  --input data/raw/marker_test_10s.raw \
  --detections data/reports/static_marker_test.json \
  --output-prefix data/reports/frequency_test \
  --bin-us 50 \
  --roi-radius-px 30 \
  --min-frequency-hz 5 \
  --max-frequency-hz 3000
```

Check the leading ON and OFF FFT candidates for agreement near 165, 366, and 596 Hz.

## Current limitations

- Position estimation uses the same decayed frequency map used for identity.
- Fast motion creates a spatial frequency trail and can delay the estimated center.
- Center smoothing adds additional tracking delay.
- Temporary occlusion can weaken the frequency lock.
- The tracker estimates relative image motion only.
- It cannot distinguish camera motion from marker motion.
- Metric position and orientation are not yet calculated.
- Exactly three points lead to multiple possible P3P pose hypotheses.

## Next implementation checkpoint

The next tracker should separate frequency identity from instantaneous position:

1. Use a longer global memory for frequency identity.
2. Use a short local event window for current position.
3. Predict each LED with a constant-velocity model.
4. Expand the local search region according to image velocity.
5. Use the global detector only for acquisition and recovery.
6. Report position confidence and covariance.
7. Feed three frequency-labelled image points into P3P.
8. Rank pose hypotheses using positive depth, marker normal, valid range, and temporal continuity.
