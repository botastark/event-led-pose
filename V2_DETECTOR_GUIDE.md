# V2 Frequency-Selective LED Detector

This document describes `tools/detect_leds_for_pose_live_v2.py`, the current live baseline for frequency-labelled detection of the three blinking LEDs.

> **Scope:** v2 is a short-window global detector with temporal confirmation. It is not yet a full event-level local tracker and it does not estimate 6-DoF pose.

## Purpose

The script detects the image location of each known LED frequency over the full IMX636 sensor and returns one observation per nominal frequency:

| Provisional LED channel | Nominal frequency | Display colour |
| ----------------------- | ----------------: | -------------- |
| `led_165`               |            165 Hz | Green          |
| `led_365`               |            365 Hz | Yellow         |
| `led_596`               |            596 Hz | Blue           |

These values are provisional values derived from `analyze_led_frequencies.py`. The measured FFT fundamentals were approximately 164.39 Hz, 364.81 Hz, and 596.38 Hz. Confirm the actual firmware/electrical frequencies before freezing target IDs or relying on phase-sensitive frequency matching.

## Algorithm overview

For a detection window, v2 performs the following operations.

```text
Current event slice
  -> select every Nth event for frequency processing
  -> reduce x/y positions to a spatial block grid
  -> accumulate support per block
  -> correlate event timestamps with sin/cos reference for every known frequency
  -> spatially box-sum support and complex responses
  -> find one full-sensor peak per frequency
  -> calculate support, coherence, score, and peak ratio
  -> apply per-frequency lock gates
  -> require two consecutive locked detections
  -> display only stable locked blobs
```

The display uses all events from the newest slice; frequency processing may use a decimated subset. This keeps the visual stream current even if frequency processing is costly.

## Frequency response

For an event

\[
e_k = (x_k, y_k, t_k, p_k), \qquad p_k \in \{-1,+1\},
\]

and a known target frequency \(f_i\), the detector accumulates a complex response in each spatial block:

\[
R_i = \sum_k p_k\cos(2\pi f_i t_k),
\]

\[
I_i = \sum_k p_k\sin(2\pi f_i t_k).
\]

Its amplitude is:

\[
A_i = \sqrt{R_i^2 + I_i^2}.
\]

A true periodically blinking LED tends to add constructively at its own frequency. Background motion and unrelated events tend to cancel because their event times are not consistently phase-aligned to the target frequency.

The detector does not retain a long history in v2. Its `det_support`, `det_real`, and `det_imag` states are reset before each detection update. This reduces spatial trails during camera or target motion compared with the original long-memory tracker.

## Detection metrics

### `sup` / support

Console form:

```text
sup=438
```

`support` is the spatially box-summed count of frequency-processing events near the candidate peak.

\[
S = \sum\_{k \in \text{local spatial neighbourhood}} 1.
\]

It answers: **is there enough local event evidence to consider this a candidate LED?**

It is not frequency-specific: support can include background edges, LED halo, reflections, or events from another LED. It must therefore be combined with coherence and peak ratio.

Because v2 uses `--event-stride`, support is measured in **sampled-event units**. The code compensates by dividing the configured support threshold by `event_stride` before gating.

### `coh` / coherence

Console form:

```text
coh=0.18
```

The coherence is the magnitude of the complex frequency response divided by support:

\[
\operatorname{coherence}\_i = \frac{A_i}{\max(S,\epsilon)}.
\]

It answers: **do local event timestamps behave periodically at this LED's expected frequency?**

- High support + low coherence: bright or moving structure, but not periodic at \(f_i\).
- Low support + high coherence: potentially a random coincidence; support gate should reject it.
- High support + high coherence: plausible frequency-labelled LED.

In the idealized formulation it lies approximately in \([0,1]\), although implementation details, spatial summation, polarity imbalance, and sensor behavior mean its practical distribution should be measured from RAW replay rather than assumed.

### `ratio` / peak ratio

Console form:

```text
ratio=1.34
```

V2 computes a combined score:

\[
\operatorname{score}\_i = \operatorname{coherence}\_i\sqrt{S}.
\]

It finds the best score over the full sensor, suppresses a neighbourhood around that best peak, then finds the strongest remaining spatially separate score:

\[
\operatorname{peak\ ratio} =
\frac{\operatorname{score}_{\text{best}}}
{\max(\operatorname{score}_{\text{second}},\epsilon)}.
\]

It answers: **is this one clearly dominant spatial answer for frequency \(f_i\), or are there competing locations?**

- `ratio` close to 1: ambiguous; multiple similar candidates, diffuse background, LED halo, reflection, or cross-frequency response.
- Larger `ratio`: one spatial peak dominates.

A close LED can create multiple nearby maxima through halo and saturation. This can lower `ratio` even when the LED is genuine, so tune this threshold separately for near and far conditions if necessary.

### `cnt` / temporal lock counter

Console form:

```text
cnt=2
```

`cnt` is not an event count. It is a temporal persistence counter.

- Every gated detector success increments it, capped at 3.
- Every gated failure decrements it, floored at 0.
- A blob is displayed and treated as `LOCK` only when:

\[
\texttt{cnt} \ge 2.
\]

This rejects one-window false positives. It adds at least one additional detection interval before a new LED becomes usable.

## Main parameters

### Input and temporal parameters

| Parameter               |                  Default | Meaning                                       | Main trade-off                                                       |
| ----------------------- | -----------------------: | --------------------------------------------- | -------------------------------------------------------------------- |
| `--slice-us`            |                20,000 µs | Event iterator slice duration                 | Shorter reduces data latency but provides fewer LED cycles per slice |
| `--detection-window-us` |                20,000 µs | Short detection window \(T_d\)                | Longer improves frequency evidence but causes more motion smear      |
| `--detection-update-ms` | derived from `T_d` in v2 | Wall-time cadence for full detection          | Lower means more compute; higher increases reacquisition delay       |
| `--display-fps`         |                    30 Hz | Maximum GUI refresh rate                      | GUI only; not a sensor measurement rate                              |
| `--phase-bin-us`        |                   100 µs | Timestamp quantization for sine/cosine lookup | Smaller improves phase resolution, increases lookup table size       |

For the slowest 165 Hz LED:

\[
\frac{2}{f\_{\min}} = \frac{2}{165} \approx 12.1\text{ ms}.
\]

Therefore `--detection-window-us 20000` contains approximately 3.3 cycles of the slowest LED and is a reasonable starting point. A frequency-ID detector cannot observe two complete 165 Hz blinks in less than approximately 12.1 ms.

### Spatial and performance parameters

| Parameter                 |       Default | Meaning                                      | Tuning notes                                                           |
| ------------------------- | ------------: | -------------------------------------------- | ---------------------------------------------------------------------- |
| `--block-size`            |         12 px | Spatial resolution of frequency processing   | Smaller gives more accurate peak localization but costs more CPU       |
| `--spatial-radius-blocks` | 2 or 3 blocks | Box-sum radius for candidate evidence        | Larger is robust to small blobs but can merge halo/background evidence |
| `--event-stride`          |             2 | Use every Nth event for frequency processing | Higher reduces compute but weakens support and temporal evidence       |

The displayed event frame is generated from all events in the newest slice. `event_stride` affects frequency processing, not the visual event image.

### Base lock thresholds

| Parameter                   | Default | Meaning                                    |
| --------------------------- | ------: | ------------------------------------------ |
| `--minimum-support-base`    |     100 | Base required local sampled-event support  |
| `--minimum-coherence`       |    0.12 | Required periodic timing coherence         |
| `--minimum-peak-ratio-base` |    1.30 | Required best-vs-second spatial separation |

V2 applies channel-specific multipliers because blue was initially used to reject false locks while the physical blue LED was off:

```python
MIN_SUPPORT = {
    0: base_support,          # green
    1: base_support * 0.9,    # yellow, slightly relaxed
    2: base_support * 2.0,    # blue, conservative
}

MIN_PEAK_RATIO = {
    0: base_ratio,
    1: base_ratio * 0.95,     # yellow, slightly relaxed
    2: base_ratio * 1.4,      # blue, conservative
}
```

If blue is physically active and gets lost at close range, its conservative thresholds need to be relaxed and revalidated against recordings where blue is off. Do not relax blue blindly: that was the channel that previously produced false locks from another LED's activity.

## Reading console output

Example:

```text
165Hz LOCK xy=(614,481) coh=0.214 sup=1720 ratio=1.48 cnt=3 |
366Hz LOCK xy=(734,492) coh=0.187 sup=1395 ratio=1.31 cnt=3 |
596Hz SEARCH xy=(667,375) coh=0.061 sup=241 ratio=1.03 cnt=0
```

Interpretation:

- Green and yellow satisfy their support, coherence, and ratio gates on at least two consecutive windows.
- Blue has a location estimate because every frequency map has a maximum, but it is not trusted: low coherence, weak support, low peak ratio, and no temporal persistence.
- **Never use the position of a `SEARCH` observation as a P3P input.**

## Practical tuning workflow

Always tune from repeatable RAW recordings, not only by watching the GUI.

### 1. Establish false-positive baseline

Record an experiment with one LED physically off. For that frequency, log `coh`, `sup`, `ratio`, and `cnt`.

Desired result:

```text
inactive LED -> SEARCH consistently
```

### 2. Characterize active LED

Record the same LED active at:

- far distance;
- nominal distance;
- close distance;
- static;
- translation;
- rotation;
- partial stick occlusion.

For each, identify which threshold fails first.

| Failing metric | Likely condition                                             | First response                                                                |
| -------------- | ------------------------------------------------------------ | ----------------------------------------------------------------------------- |
| `sup` low      | LED dim, far, excessive `event-stride`                       | Reduce stride, improve illumination/bias, reduce support threshold cautiously |
| `coh` low      | Wrong nominal frequency, phase mismatch, background activity | Re-measure frequency, reduce phase bin, tune sensor bias                      |
| `ratio` low    | Close LED halo, reflection, multiple candidates              | Reduce spatial radius, use a conditional near-blob gate                       |
| `cnt` unstable | Detection intermittently passes                              | Improve the root metric; do not simply set lock requirement to one            |

### 3. Change one parameter at a time

Keep the RAW fixture and command line in the run manifest. Never tune and evaluate on the same undocumented live stream.

## Relation to the 2023 ALM paper

V2 takes inspiration from the paper's short-window approach but differs materially:

| 2023 ALM method                                                       | V2 implementation                                                              |
| --------------------------------------------------------------------- | ------------------------------------------------------------------------------ |
| LED candidates from thresholded event frame plus connected components | Frequency map searches full sensor independently per known frequency           |
| Frequency determined from local timing-difference histogram           | Frequency determined by sin/cos timestamp correlation                          |
| Typical frequencies in kHz range; integer-µs periods preferred        | Current LEDs at 165/366/596 Hz, provisional and not yet electrically confirmed |
| Detector spawns event-level local trackers                            | V2 only applies a two-window lock confirmation                                 |
| Local tracker updates centre/radius from current events               | V2 re-detects globally at each detection cycle                                 |
| Tracker loss checked by reprojection error                            | P3P/reprojection integration not implemented yet                               |

The paper's event-level tracker should not be copied into the current system until bias tuning produces a local ROI dominated by LED events. Your earlier local-tracker experiment showed that current background activity is too high: tracker radii expanded to the maximum and centres drifted to background structure.

## Current limits

- A 165 Hz minimum frequency places a hard lower bound of about 12.1 ms on any two-blink frequency-ID detection window.
- V2 is a detector, not a high-rate event-level tracker.
- LED centres are diagnostic measurements until intrinsics/distortion and target geometry are calibrated.
- Three planar LEDs form a minimal P3P target; multiple pose hypotheses are expected.
- Do not send v2 centres or provisional P3P output to the Panda controller.

## Recommended baseline command

```bash
python3 tools/detect_leds_for_pose_live_v2.py \
  --serial 00050946 \
  --spatial-radius-blocks 3 \
  --block-size 12 \
  --phase-bin-us 25 \
  --display-observations
```

Use `--phase-bin-us 25` only after checking CPU load and GUI responsiveness. It improves temporal phase resolution for the 596 Hz channel, whose period is only about 1.68 ms.
