#!/usr/bin/env python3
"""
Perception-side UDP pose publisher for event-led-pose → panda_tracker.

Wire protocol: panda_pbvs_project/common/protocol.py :: pack_task_pose
               Format "<4sBBHQf16d", 156 bytes, little-endian.

Frame convention (inherited from panda_tracker/panda_pbvs_project/
                  perception/task_pose_adapter.py):
  T_TS = inv(T_EC @ T_CT) @ T_ES
  where T_CT is what this module receives from the P3P solver.

For the current baseline (no hand-eye calibration), T_EC and T_ES
are identity — set PLACEHOLDER_EXTRINSICS=True and log a warning.
"""
from __future__ import annotations

import math
import socket
import struct
import threading
import time
from dataclasses import dataclass, field
from typing import Sequence

import numpy as np

# ── Wire format (must match panda_pbvs_project/common/protocol.py) ────────────
TASK_POSE_MAGIC   = b"PTP2"
TASK_POSE_VERSION = 2
TASK_POSE_FORMAT  = "<4sBBHQf16d"
TASK_POSE_SIZE    = struct.calcsize(TASK_POSE_FORMAT)   # 156 bytes


def pack_task_pose(
    T_TS: np.ndarray,
    *,
    sequence_id: int,
    confidence: float,
    valid: bool,
) -> bytes:
    """Encode a task-pose datagram. Validates all fields before packing."""
    T_TS = np.asarray(T_TS, dtype=float)
    if T_TS.shape != (4, 4):
        raise ValueError(f"T_TS must be (4,4), got {T_TS.shape}")
    if not np.all(np.isfinite(T_TS)):
        raise ValueError("T_TS contains non-finite values")
    if not isinstance(sequence_id, int) or not 0 <= sequence_id < (1 << 64):
        raise ValueError("sequence_id must be uint64")
    confidence = float(confidence)
    if not math.isfinite(confidence) or not 0.0 <= confidence <= 1.0:
        raise ValueError("confidence must be finite and in [0, 1]")
    return struct.pack(
        TASK_POSE_FORMAT,
        TASK_POSE_MAGIC,
        TASK_POSE_VERSION,
        int(bool(valid)),
        0,                      # reserved
        sequence_id,
        confidence,
        *T_TS.reshape(-1),
    )


# ── Extrinsic placeholders ─────────────────────────────────────────────────────
# Replace these with surveyed values after hand-eye calibration (gate 5+6).
PLACEHOLDER_EXTRINSICS = True
_I4 = np.eye(4, dtype=float)


def compose_T_TS(
    T_CT: np.ndarray,
    T_EC: np.ndarray = _I4,
    T_ES: np.ndarray = _I4,
) -> np.ndarray:
    """
    T_TS = inv(T_EC @ T_CT) @ T_ES
    Matches task_pose_adapter.task_pose_from_camera_target().
    """
    T_ET = T_EC @ T_CT
    T_TE = np.linalg.inv(T_ET)
    return T_TE @ T_ES


# ── Publisher ──────────────────────────────────────────────────────────────────
@dataclass
class PoseSample:
    """A single pose estimate from the P3P solver."""
    T_CT: np.ndarray          # camera-to-target SE(3), shape (4,4)
    sensor_timestamp_us: int  # event-camera hardware timestamp [µs]
    confidence: float         # hypothesis score in [0, 1]
    valid: bool               # False → AMBIGUOUS / INSUFFICIENT_OBSERVATIONS
    latency_s: float = 0.0    # processing latency for diagnostics


class EventLedPosePublisher:
    """
    Thread-safe UDP publisher that sends 156-byte task-pose datagrams
    to one or more (host, port) destinations.

    Usage (offline replay):
        pub = EventLedPosePublisher(destinations=[("127.0.0.1", 6601)])
        for sample in replay_sequence:
            pub.publish(sample)
        pub.close()

    Usage (live):
        pub = EventLedPosePublisher(destinations=[("192.168.1.10", 6601)])
        # call pub.publish(sample) from your P3P callback thread
    """

    def __init__(
        self,
        destinations: Sequence[tuple[str, int]],
        *,
        warn_placeholder_extrinsics: bool = True,
    ) -> None:
        self._destinations = list(destinations)
        self._sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self._lock = threading.Lock()
        self._seq = 0
        self._stats = {"sent": 0, "invalid": 0, "errors": 0}

        if PLACEHOLDER_EXTRINSICS and warn_placeholder_extrinsics:
            import warnings
            warnings.warn(
                "PLACEHOLDER_EXTRINSICS=True: T_EC and T_ES are identity. "
                "T_TS equals inv(T_CT). Do not use for metric control.",
                stacklevel=2,
            )

    def publish(self, sample: PoseSample) -> None:
        """
        Pack and send one pose sample.  Must be called from a single thread
        or protected externally; the internal lock only guards _seq.
        """
        with self._lock:
            self._seq += 1
            seq = self._seq

        T_TS = compose_T_TS(sample.T_CT)          # identity extrinsics for now

        try:
            payload = pack_task_pose(
                T_TS,
                sequence_id=seq,
                confidence=sample.confidence if sample.valid else 0.0,
                valid=sample.valid,
            )
        except (ValueError, struct.error) as exc:
            self._stats["errors"] += 1
            print(f"[pose_publisher] pack error seq={seq}: {exc}")
            return

        with self._lock:
            for dest in self._destinations:
                try:
                    self._sock.sendto(payload, dest)
                    self._stats["sent"] += 1
                    if not sample.valid:
                        self._stats["invalid"] += 1
                except OSError as exc:
                    self._stats["errors"] += 1
                    print(f"[pose_publisher] send error → {dest}: {exc}")

    def stats(self) -> dict:
        with self._lock:
            return dict(self._stats)

    def close(self) -> None:
        self._sock.close()