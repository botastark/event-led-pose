# Event LED Pose

Explainable, offline-first tracking of a rigid triangular marker made from three
frequency-coded LEDs, observed by a Prophesee EVK4 / Sony IMX636 event camera.

The final output will be the six-degree-of-freedom transform `T_camera_plate`:
metric position `(x, y, z)` and orientation of the plate relative to the camera
optical frame. The repository is intentionally being built in small,
replayable checkpoints.

## Current checkpoint

The current code does **not** estimate 6-DoF pose yet. It has validated the
lower layers on real hardware:

- rootless access to one EVK4 USB device;
- a pinned Ubuntu 24.04 + OpenEB 5.3.0 runtime;
- deterministic EVT3 RAW recording and replay;
- event-rate maps and diagnostic images;
- detection of exactly three stationary LED regions;
- distinct temporal-frequency signatures for all three LEDs.

That boundary is deliberate. Camera calibration, surveyed target geometry,
P3P hypothesis handling, uncertainty, and temporal tracking must be validated
before a pose is called reliable.

## Why this environment

The host is kept clean. OpenEB and its Python dependencies are installed only
inside a rootless Podman image:

- no Docker daemon or `docker` group;
- no Conda;
- no host virtual environment;
- no host OpenEB/Metavision installation;
- no `sudo podman`;
- no `--privileged` container;
- only the resolved EVK4 USB node is passed to a live-camera container;
- replay containers run without network access.

The verified development host is Ubuntu 26.04 LTS. The controlled user space
inside the image is Ubuntu 24.04 amd64, which is supported by the packaged
OpenEB release. The host CPU must support AVX2.

## Verified hardware and software

| Item | Verified value |
|---|---|
| Camera | Prophesee EVK4, Sony IMX636 HD |
| USB ID | `04b4:00f5` |
| Camera serial | `00050946` |
| EVK4 firmware/release | `3.9.0` |
| Sensor geometry | `1280 x 720` |
| USB link | `5000M` |
| Encoding | `EVT3` |
| Container OS | Ubuntu 24.04 amd64 |
| OpenEB | `5.3.0` |
| Container Python | Ubuntu Python 3.12 |
| NumPy | `1:1.26.4+ds-6ubuntu1` |
| h5py / h5py-serial | `3.10.0-1ubuntu3` |
| Runtime image tag | `localhost/event-led-pose/openeb-runtime:5.3.0-r2-ubuntu24` |

The runtime base is pinned to:

```text
docker.io/library/ubuntu:24.04@sha256:52df9b1ee71626e0088f7d400d5c6b5f7bb916f8f0c82b474289a4ece6cf3faf
```

The tag is convenient; the digest and package versions are the reproducibility
controls.

## Repository layout

```text
event-led-pose/
├── .gitignore
├── Makefile
├── README.md
├── container/
│   ├── Containerfile.package-query
│   └── Containerfile.runtime
├── config/
│   └── hardware/
│       └── evk4_imx636.yaml
├── docs/
│   ├── GIT_COMMIT_GUIDE.md
│   └── NEXT_CHECKPOINT.md
├── tools/
│   ├── analyze_led_frequencies.py
│   ├── analyze_raw.py
│   ├── capture_raw.py
│   ├── detect_static_marker.py
│   ├── query_openeb_version.sh
│   └── host/
│       └── check_evk4.sh
└── data/                       # local and ignored by Git
    ├── raw/
    └── reports/
```

### Tool summary

| Tool | Purpose | Inputs | Outputs |
|---|---|---|---|
| `check_evk4.sh` | Resolves the EVK4 USB node and checks identity, topology, ACL, and current-user read/write access | Connected EVK4 | Terminal report |
| `query_openeb_version.sh` | Queries Prophesee packages in an expendable Ubuntu 24.04 image | Network during build | Available versions and base-image identity |
| `capture_raw.py` | Records CD events from the selected camera using sensor timestamps | Live EVK4 | EVT3 `.raw` and terminal statistics |
| `analyze_raw.py` | Decodes all events into spatial polarity maps and temporal counts | One `.raw` | `.json`, `.npz`, three log-scaled `.pgm` files |
| `detect_static_marker.py` | Subtracts the ambient event-rate map and finds the three connected LED regions | Ambient and marker `.npz` | Detection `.json`, `.npz`, annotated `.ppm` |
| `analyze_led_frequencies.py` | Builds per-LED ON/OFF time series and calculates FFT and autocorrelation candidates | Marker `.raw`, detections `.json` | Frequency `.json` and `.npz` |

All analysis outputs are evidence, not source. Regenerate them from RAW data
and the committed code.

## New-computer setup

These instructions target a normal amd64 Ubuntu 24.04 or newer lab computer.
Run ordinary development commands as the logged-in user. Use `sudo` only for
the reviewed host package and udev steps.

### 1. Install minimal host packages

```bash
sudo apt update
sudo apt install -y \
  git make podman crun uidmap fuse-overlayfs slirp4netns \
  usbutils acl ca-certificates curl
```

Do not install OpenEB, Python packages, Docker, Conda, or a project virtual
environment on the host.

### 2. Clone and inspect

```bash
git clone REPLACE_WITH_PRIVATE_REPOSITORY_URL event-led-pose
cd event-led-pose

git status
git log -1 --oneline
```

Use a private remote unless the lab has explicitly approved publishing the
camera configuration and project design. RAW recordings must remain outside
Git even when the code repository is private.

### 3. Check platform requirements and rootless Podman

```bash
uname -m
grep -m1 -o '\bavx2\b' /proc/cpuinfo
podman --version
crun --version
podman info --format '{{.Host.Security.Rootless}}'
```

Expected:

```text
x86_64
avx2
true
```

If rootless Podman is not configured, stop and have the lab administrator
review `/etc/subuid` and `/etc/subgid`. Do not work around it with
`sudo podman`.

### 4. Connect and check the camera

Connect the EVK4 directly to a USB 3 port using a known-good USB 3 cable:

```bash
make check-camera
```

Continue only when:

- exactly one `04b4:00f5` device is found;
- `lsusb -t` reports `5000M` or faster;
- the current user has both read and write access.

The USB device number changes after reconnecting the camera. Never hard-code
`/dev/bus/usb/002/005` or any previously observed node.

#### If access fails

First ask the lab administrator to review this narrow rule:

```udev
SUBSYSTEM=="usb", ATTR{idVendor}=="04b4", ATTR{idProduct}=="00f5", MODE="0660", TAG+="uaccess"
```

If approved, install it as:

```text
/etc/udev/rules.d/70-prophesee-evk4.rules
```

Then reload the rules and physically reconnect the camera:

```bash
sudo udevadm control --reload-rules
sudo udevadm trigger \
  --subsystem-match=usb \
  --attr-match=idVendor=04b4 \
  --attr-match=idProduct=00f5
```

Run `make check-camera` again. Do not use a `MODE="0666"` rule and do not run
the camera as root. `uaccess` normally depends on an active local login; an
SSH-only/shared-lab deployment may need a separately reviewed dedicated-group
rule.

### 5. Build the pinned runtime

This is the only step that needs network access after cloning:

```bash
make build-runtime
make verify-runtime
```

Expected verification includes:

```text
5.3.0
Runtime Python imports: PASS
```

`NumPy` and `h5py` are explicit runtime dependencies. If
`metavision_core.event_io.EventsIterator` fails because either is missing, the
wrong/older image was used; rebuild and use the exact `r2` tag.

Optional package-repository diagnostic:

```bash
make query-openeb
```

This query is informational. Do not silently update the pinned version because
a newer candidate appears.

### 6. Probe the live camera through the isolated runtime

Resolve the current device node:

```bash
read -r EVK_BUS EVK_DEVICE < <(
  lsusb -d 04b4:00f5 |
  awk 'NR == 1 {gsub(":", "", $4); print $2, $4}'
)
EVK_NODE="/dev/bus/usb/${EVK_BUS}/${EVK_DEVICE}"

printf 'EVK node: %s\n' "$EVK_NODE"
test -r "$EVK_NODE" && test -w "$EVK_NODE"
```

Probe it:

```bash
podman run --rm \
  --userns=keep-id \
  --network=none \
  --cap-drop=all \
  --security-opt=no-new-privileges \
  --read-only \
  --tmpfs=/tmp:rw,nosuid,nodev,size=64m \
  --device="$EVK_NODE" \
  localhost/event-led-pose/openeb-runtime:5.3.0-r2-ubuntu24 \
  metavision_platform_info
```

Confirm the IMX636, serial, firmware, EVT3, and USB speed. A headless container
may report that OpenGL information is unavailable; that does not invalidate a
successful HAL camera enumeration.

## Recording a baseline dataset

Create local output directories:

```bash
mkdir -p data/raw data/reports
chmod 0750 data data/raw data/reports
```

The following helper keeps the live container narrow:

```bash
capture_raw()
{
  local output_name="$1"
  local duration_s="$2"

  read -r EVK_BUS EVK_DEVICE < <(
    lsusb -d 04b4:00f5 |
    awk 'NR == 1 {gsub(":", "", $4); print $2, $4}'
  )
  local evk_node="/dev/bus/usb/${EVK_BUS}/${EVK_DEVICE}"

  test -r "$evk_node" && test -w "$evk_node"

  podman run --rm \
    --userns=keep-id \
    --network=none \
    --cap-drop=all \
    --security-opt=no-new-privileges \
    --read-only \
    --tmpfs=/tmp:rw,nosuid,nodev,size=64m \
    --device="$evk_node" \
    --volume="$PWD/tools:/tools:ro" \
    --volume="$PWD/data/raw:/data:rw" \
    localhost/event-led-pose/openeb-runtime:5.3.0-r2-ubuntu24 \
    python3 /tools/capture_raw.py \
      --serial 00050946 \
      --duration-s "$duration_s" \
      --output "/data/$output_name"
}
```

Capture three controlled conditions:

```bash
# Fit a genuinely opaque lens/sensor cover, wait for the scene to settle,
# then record. A printed card is not a valid dark cover.
capture_raw dark_opaque_settled_default_evt3_10s.raw 10

# Marker off, ordinary lab illumination, camera and scene stationary.
capture_raw ambient_default_evt3_5s.raw 5

# All three frequency-coded LEDs on, camera and plate stationary.
capture_raw marker_static_default_evt3_10s.raw 10
```

Record conditions in a sidecar text file: camera/plate distance, angle,
lighting, lens focus/aperture if adjustable, LED supply state, and anything
that moved. The current recorder starts saving immediately; manual settling is
required until a configurable pre-roll is implemented.

Inspect each RAW file using the path as seen **inside** the container:

```bash
for raw_name in data/raw/*.raw
do
  podman run --rm \
    --userns=keep-id \
    --network=none \
    --read-only \
    --tmpfs=/tmp:rw,nosuid,nodev,size=64m \
    --volume="$PWD/data/raw:/data:ro" \
    localhost/event-led-pose/openeb-runtime:5.3.0-r2-ubuntu24 \
    metavision_file_info -i "/data/$(basename "$raw_name")"
done
```

If the host directory `data/raw` is mounted at `/data`, the correct path is
`/data/file.raw`, not `/data/raw/file.raw`. Preserve underscores exactly.

## Offline analysis

All replay commands disable networking and mount the source tools read-only.

### 1. Generate event maps

```bash
for stem in \
  ambient_default_evt3_5s \
  marker_static_default_evt3_10s
do
  podman run --rm \
    --userns=keep-id \
    --network=none \
    --cap-drop=all \
    --security-opt=no-new-privileges \
    --read-only \
    --tmpfs=/tmp:rw,nosuid,nodev,size=64m \
    --volume="$PWD/tools:/tools:ro" \
    --volume="$PWD/data/raw:/data:ro" \
    --volume="$PWD/data/reports:/reports:rw" \
    localhost/event-led-pose/openeb-runtime:5.3.0-r2-ubuntu24 \
    python3 /tools/analyze_raw.py \
      --input "/data/${stem}.raw" \
      --output-prefix "/reports/${stem}"
done
```

### 2. Detect the stationary three-LED marker

```bash
podman run --rm \
  --userns=keep-id \
  --network=none \
  --cap-drop=all \
  --security-opt=no-new-privileges \
  --read-only \
  --tmpfs=/tmp:rw,nosuid,nodev,size=64m \
  --volume="$PWD/tools:/tools:ro" \
  --volume="$PWD/data/reports:/reports:rw" \
  localhost/event-led-pose/openeb-runtime:5.3.0-r2-ubuntu24 \
  python3 /tools/detect_static_marker.py \
    --ambient /reports/ambient_default_evt3_5s.npz \
    --marker /reports/marker_static_default_evt3_10s.npz \
    --output-prefix /reports/static_marker_default
```

The command must find exactly three components. Inspect
`data/reports/static_marker_default_detections.ppm`; a numerical PASS without a
sensible overlay is insufficient.

### 3. Estimate the three temporal frequencies

```bash
podman run --rm \
  --userns=keep-id \
  --network=none \
  --cap-drop=all \
  --security-opt=no-new-privileges \
  --read-only \
  --tmpfs=/tmp:rw,nosuid,nodev,size=64m \
  --volume="$PWD/tools:/tools:ro" \
  --volume="$PWD/data/raw:/data:ro" \
  --volume="$PWD/data/reports:/reports:rw" \
  localhost/event-led-pose/openeb-runtime:5.3.0-r2-ubuntu24 \
  python3 /tools/analyze_led_frequencies.py \
    --input /data/marker_static_default_evt3_10s.raw \
    --detections /reports/static_marker_default.json \
    --output-prefix /reports/marker_frequency_default
```

Interpret the FFT peak together with its harmonic series and ON/OFF agreement.
Autocorrelation can rank subharmonic periods highly for pulse trains; it is a
cross-check, not the sole ID decision.

## Verified dataset checkpoint

The first static-marker recording contained `102,155,063` CD events over
approximately 9.992 s (`10.2 Mev/s`). Ambient was approximately `1.8 Mev/s`.

Detected image coordinates:

| Component | Informal position | Center `(x, y)` px | Signal rate |
|---:|---|---:|---:|
| 0 | right | `(800.690, 321.130)` | `3.055 Mev/s` |
| 1 | top | `(666.767, 162.272)` | `2.880 Mev/s` |
| 2 | left | `(537.591, 325.240)` | `2.346 Mev/s` |

The 2D centroid was `(668.349, 269.547)` px. Pairwise image distances were
approximately `207.777`, `263.131`, and `207.954` px.

The strongest FFT fundamentals in the same recording were:

| Component | Informal position | Observed frequency |
|---:|---|---:|
| 0 | right | `364.813 Hz` |
| 1 | top | `596.381 Hz` |
| 2 | left | `164.388 Hz` |

These are provisional identifiers observed in one dataset. Confirm the intended
frequencies from the LED firmware or an electrical measurement and repeat the
analysis at multiple ranges, angles, lighting levels, and supply states before
freezing them in target calibration.

The 2D centroid is a diagnostic only. It is not metric `(x, y, z)`, and it is
not the projection of a surveyed 3D target-frame origin under all poses.

## Data integrity and Git policy

Commit:

- source code and tests;
- Containerfiles and exact version pins;
- Makefile and safe wrapper scripts;
- hardware and algorithm configuration without secrets;
- documentation;
- small synthetic test fixtures that are explicitly approved.

Do not commit:

- RAW/HDF5 recordings;
- `.npz`, `.pgm`, `.ppm`, or generated JSON reports under `data/`;
- build output, caches, core dumps, logs, virtual environments;
- tokens, credentials, private repository URLs, or personal paths;
- camera firmware images.

Before each capture transfer or archival step:

```bash
stat --format='%n  size=%s bytes' data/raw/*.raw
sha256sum data/raw/*.raw
```

Store large recordings in approved lab storage with a manifest containing the
hash, capture conditions, code commit, runtime image identity, camera serial,
firmware, and sensor settings.

Use the exact staging procedure in `docs/GIT_COMMIT_GUIDE.md`. In particular,
stage named source directories rather than `git add .`.

## Next engineering milestone

Before implementing pose:

1. add recorder pre-roll and a run manifest;
2. survey the three optical centers, emitter diameters, coplanarity, and
   measurement uncertainty in a declared plate frame;
3. electrically confirm the three nominal blink frequencies;
4. calibrate camera intrinsics and distortion for the installed lens and focus;
5. create static recordings at surveyed distances and tilts;
6. implement frequency-labelled subpixel centers with covariance;
7. enumerate and score all P3P/AP3P pose hypotheses;
8. report ambiguity and uncertainty instead of forcing a pose;
9. only then add asynchronous temporal tracking.

Exactly three coplanar points are a minimal pose target. P3P can produce
multiple hypotheses, one bad LED cannot be rejected with redundancy, and loss
of any LED prevents an unconstrained instantaneous 6-DoF solution. A fourth
uniquely coded LED—preferably slightly out of plane—is the best hardware
upgrade if reliability requirements become strict.

## Upstream references

- [Prophesee OpenEB package installation for Ubuntu](https://docs.prophesee.ai/stable/installation/linux_openeb_with_packages.html)
- [Prophesee RAW file format](https://docs.prophesee.ai/stable/data/file_formats/raw.html)
- [Prophesee `metavision_file_info`](https://docs.prophesee.ai/stable/samples/modules/stream/file_info.html)
- [Podman rootless-mode documentation](https://docs.podman.io/en/latest/markdown/podman.1.html#rootless-mode)
