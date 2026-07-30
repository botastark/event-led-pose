# Next Checkpoint: Freeze the Physical Target

## Validated baseline

- EVK4 / IMX636 enumerates through the pinned rootless Podman runtime.
- Current user has narrow read/write ACL access to USB ID `04b4:00f5`.
- OpenEB 5.3.0 records and replays EVT3 RAW data at `1280 x 720`.
- NumPy, h5py, HAL, and `EventsIterator` imports pass in the `r2` image.
- Static marker-minus-ambient analysis finds exactly three LED components.
- The three components have separable FFT fundamentals near 164.388 Hz,
  364.813 Hz, and 596.381 Hz in the validated recording.

## Objective

Replace provisional image labels with a surveyed, immutable target definition
that can support metric pose.

## Physical preparation

1. Mount all three optical emitters to one stiff plate.
2. Immobilize batteries, driver boards, wires, tape, and connectors.
3. Use a matte, non-reflective front surface around the emitters.
4. Add permanent marks defining the target origin, positive axes, and front
   normal.
5. Confirm that the optical centers do not move when the target is handled.

## Measurements to return

Use millimetres while measuring, retain the raw readings, then convert the
calibration model to metres.

For every LED:

- permanent ID and informal plate location;
- intended/firmware frequency;
- independently measured electrical frequency and uncertainty;
- optical-center coordinates in the declared target frame;
- visible emitting diameter and uncertainty;
- emitting-surface height relative to the plate datum.

For the target:

- all three pairwise optical-center distances;
- at least three repeated measurements of every distance;
- measurement instrument and its resolution;
- coplanarity measurement and residual;
- definition of the requested triangle center;
- definition of the positive plate normal.

Do not infer metric geometry from the prototype photograph or handwritten
marks.

## Recorder improvement

Before collecting calibration data, add:

- configurable pre-roll that is not saved;
- a sidecar run manifest;
- refusal to overwrite existing RAW or manifest files;
- camera serial, firmware, encoding, resolution, image identity, Git commit,
  sensor settings, wall-clock start, and sensor-time range in the manifest;
- final RAW SHA-256 after recording.

## Calibration dataset

After target survey and camera intrinsic calibration, record stationary
sequences at surveyed:

- distances spanning the operating range;
- fronto-parallel and tilted orientations;
- image center, edges, and corners;
- low, normal, and bright ambient illumination.

Each sequence needs an independent ground-truth pose or fixture measurement.
Do not tune and evaluate on the same poses.

## Gate to start pose implementation

Proceed only when:

- camera intrinsics and distortion are stored with reprojection diagnostics;
- target geometry and uncertainties are committed;
- frequency-to-physical-LED mapping is independently confirmed;
- replay produces stable labelled centers with a reported covariance;
- every calibration recording has a manifest and SHA-256;
- acceptance limits for translation, rotation, update rate, latency, range,
  dropout, and relock time are written down.

