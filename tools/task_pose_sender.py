#!/usr/bin/env python3
"""Publish a tracker target pose T_CT over UDP.

Frame convention
----------------
T_XY is the pose of frame Y expressed in frame X and maps coordinates
from frame Y into frame X.

For the current Panda position-tracking interface this script publishes T_CT:
the target/triangle frame T expressed in camera frame C.

The existing PTP2 C++ field is historically named T_CT, but the receiver
interprets the payload as T_CT.

Wire format
-----------
PTP2 version 2:

    magic       4 bytes   b"PTP2"
    version     1 byte    2
    valid       1 byte
    reserved    2 bytes   0
    sequence    8 bytes
    confidence  4 bytes
    T_CT      128 bytes   16 little-endian float64, row-major

Total: 148 bytes.
"""

from __future__ import annotations

import argparse
import math
import signal
import socket
import struct
import time
from dataclasses import dataclass
from typing import Sequence

import numpy as np


TASK_POSE_FORMAT = "<4sBBHQf16d"
TASK_POSE_SIZE = struct.calcsize(TASK_POSE_FORMAT)
TASK_POSE_MAGIC = b"PTP2"
TASK_POSE_VERSION = 2


def rotation_from_rpy(roll: float, pitch: float, yaw: float) -> np.ndarray:
    """Return R = Rz(yaw) @ Ry(pitch) @ Rx(roll)."""
    cr, sr = math.cos(roll), math.sin(roll)
    cp, sp = math.cos(pitch), math.sin(pitch)
    cy, sy = math.cos(yaw), math.sin(yaw)

    return np.array(
        [
            [cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr],
            [sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr],
            [-sp, cp * sr, cp * cr],
        ],
        dtype=np.float64,
    )


def pose6_to_transform(pose6: Sequence[float] | np.ndarray) -> np.ndarray:
    """Convert [x, y, z, roll, pitch, yaw] into the 4x4 transform T_CT."""
    pose = np.asarray(pose6, dtype=np.float64).reshape(-1)

    if pose.size != 6:
        raise ValueError(f"pose6 must contain exactly 6 values; got {pose.size}.")
    if not np.all(np.isfinite(pose)):
        raise ValueError("pose6 contains a non-finite value.")

    x_t_s, y_t_s, z_t_s, roll_t_s, pitch_t_s, yaw_t_s = pose

    transform = np.eye(4, dtype=np.float64)
    transform[:3, :3] = rotation_from_rpy(
        float(roll_t_s),
        float(pitch_t_s),
        float(yaw_t_s),
    )
    transform[:3, 3] = [x_t_s, y_t_s, z_t_s]
    return transform


def validate_transform(transform: np.ndarray) -> np.ndarray:
    """Validate a finite, approximately rigid 4x4 homogeneous transform."""
    matrix = np.asarray(transform, dtype=np.float64)

    if matrix.shape != (4, 4):
        raise ValueError(f"T_CT must have shape (4, 4); got {matrix.shape}.")
    if not np.all(np.isfinite(matrix)):
        raise ValueError("T_CT contains a non-finite value.")
    if not np.allclose(matrix[3], [0.0, 0.0, 0.0, 1.0], atol=1e-9):
        raise ValueError("T_CT has an invalid homogeneous bottom row.")

    rotation = matrix[:3, :3]
    if not np.allclose(rotation.T @ rotation, np.eye(3), atol=1e-6):
        raise ValueError("T_CT rotation is not orthonormal.")
    if not math.isclose(float(np.linalg.det(rotation)), 1.0, abs_tol=1e-6):
        raise ValueError("T_CT rotation determinant is not +1.")

    return matrix


def pack_task_pose(
    transform: np.ndarray,
    *,
    valid: bool,
    sequence_id: int,
    confidence: float,
) -> bytes:
    """Pack one 148-byte PTP2 task-pose packet."""
    matrix = validate_transform(transform)

    if not 0 <= sequence_id <= 0xFFFFFFFFFFFFFFFF:
        raise ValueError("sequence_id must fit uint64.")
    if not math.isfinite(confidence) or not 0.0 <= confidence <= 1.0:
        raise ValueError("confidence must be finite and in [0, 1].")

    packet = struct.pack(
        TASK_POSE_FORMAT,
        TASK_POSE_MAGIC,
        TASK_POSE_VERSION,
        1 if valid else 0,
        0,
        sequence_id,
        confidence,
        *matrix.reshape(-1, order="C"),
    )

    if len(packet) != TASK_POSE_SIZE:
        raise RuntimeError(
            f"Unexpected packet size {len(packet)}; expected {TASK_POSE_SIZE}."
        )

    return packet


@dataclass
class TaskPosePublisher:
    destination_ip: str = "127.0.0.1"
    destination_port: int = 5000

    def __post_init__(self) -> None:
        if not 1 <= self.destination_port <= 65535:
            raise ValueError("destination_port must be between 1 and 65535.")
        self._socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self._destination = (self.destination_ip, self.destination_port)
        self._sequence_id = 0

    def publish_matrix(
        self,
        T_CT: np.ndarray,
        *,
        valid: bool = True,
        confidence: float = 1.0,
    ) -> None:
        """Publish one PTP2 T_CT packet."""
        packet = pack_task_pose(
            T_CT,
            valid=valid,
            sequence_id=self._sequence_id,
            confidence=confidence,
        )
        self._sequence_id = (self._sequence_id + 1) & 0xFFFFFFFFFFFFFFFF
        self._socket.sendto(packet, self._destination)

    def publish_pose6(self, pose6: Sequence[float] | np.ndarray) -> np.ndarray:
        """Convert a six-value vision pose to T_CT, publish it, and return it."""
        T_CT = pose6_to_transform(pose6)
        self.publish_matrix(T_CT)
        return T_CT

    def close(self) -> None:
        self._socket.close()

    def __enter__(self) -> "TaskPosePublisher":
        return self

    def __exit__(self, *_exc_info: object) -> None:
        self.close()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Publish a constant test T_CT tracker pose over UDP."
    )
    parser.add_argument("--x", type=float, required=True, help="x_T_S [m]")
    parser.add_argument("--y", type=float, required=True, help="y_T_S [m]")
    parser.add_argument("--z", type=float, required=True, help="z_T_S [m]")
    parser.add_argument("--roll", type=float, default=0.0, help="roll_T_S [rad]")
    parser.add_argument("--pitch", type=float, default=0.0, help="pitch_T_S [rad]")
    parser.add_argument("--yaw", type=float, default=0.0, help="yaw_T_S [rad]")
    parser.add_argument("--rate", type=float, default=30.0, help="Publish rate [Hz]")
    parser.add_argument("--destination-ip", default="127.0.0.1")
    parser.add_argument("--destination-port", type=int, default=5000)
    parser.add_argument(
        "--once",
        action="store_true",
        help="Publish one packet and exit instead of streaming.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    if args.rate <= 0.0:
        raise ValueError("--rate must be positive.")

    pose6 = np.array(
        [args.x, args.y, args.z, args.roll, args.pitch, args.yaw],
        dtype=np.float64,
    )
    T_CT = pose6_to_transform(pose6)

    print("pose6 = [x_T_S, y_T_S, z_T_S, roll_T_S, pitch_T_S, yaw_T_S]")
    print("pose6 =", np.array2string(pose6, precision=6))
    print("T_CT =\n", np.array2string(T_CT, precision=6, suppress_small=True))
    print(
        f"Packet: {TASK_POSE_SIZE} bytes (PTP2 v{TASK_POSE_VERSION}), "
        f"destination={args.destination_ip}:{args.destination_port}"
    )

    running = True

    def request_stop(_signum: int, _frame: object) -> None:
        nonlocal running
        running = False

    signal.signal(signal.SIGINT, request_stop)
    signal.signal(signal.SIGTERM, request_stop)

    with TaskPosePublisher(
        destination_ip=args.destination_ip,
        destination_port=args.destination_port,
    ) as publisher:
        if args.once:
            publisher.publish_matrix(T_CT)
            return 0

        period = 1.0 / args.rate
        next_send = time.monotonic()
        sent = 0
        last_status = next_send

        while running:
            now = time.monotonic()
            if now < next_send:
                time.sleep(next_send - now)
                continue

            next_send += period
            if next_send < now - period:
                next_send = now + period

            publisher.publish_matrix(T_CT)
            sent += 1

            if now - last_status >= 1.0:
                print(f"Published {sent} T_CT packets.")
                last_status = now

    return 0


if __name__ == "__main__":
    raise SystemExit(main())