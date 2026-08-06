#!/usr/bin/env python3
"""P3P/AP3P pose estimation for the 3-LED plate.

Consumes frequency-labelled LED center observations and camera intrinsics
and returns a 6-DoF camera-to-target pose T_CT with gating.

Target geometry: isosceles triangle with base 0.16 m and height 0.10 m.
Plate-frame origin is at the centroid of the three LEDs.

LED IDs are assumed to be 'left', 'right', 'apex'. Mapping from frequencies
is done by the caller.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import numpy as np
try:
    import cv2
except ImportError as exc:  # pragma: no cover
    raise RuntimeError("OpenCV (cv2) is required for pose estimation.") from exc


@dataclass
class LedObservation:
    """Minimal observation structure used for pose estimation."""

    center_xy: np.ndarray  # (2,) sensor pixels
    coherence: float
    support: float
    peak_ratio: float
    locked: bool


@dataclass
class PoseResult:
    status: str                     # 'VALID' | 'AMBIGUOUS' | 'INSUFFICIENT_OBSERVATIONS'
    T_CT: Optional[np.ndarray]      # (4,4) or None
    xyz_m: Optional[np.ndarray]     # (3,) or None
    quat_xyzw: Optional[np.ndarray] # (4,) or None
    euler_deg: Optional[np.ndarray] # (3,) or None
    reprojection_error_px: Optional[float]
    margin_px: Optional[float]


# Plate-frame LED coordinates (origin at centroid), metres.
# Base 0.16 m (L-R), height 0.10 m (apex above).
_base = 0.16
_height = 0.10
_B_left_raw = np.array([-_base / 2.0, 0.0, 0.0])
_B_right_raw = np.array([_base / 2.0, 0.0, 0.0])
_Apex_raw = np.array([0.0, _height, 0.0])
_centroid = (_B_left_raw + _B_right_raw + _Apex_raw) / 3.0
P_LEFT = _B_left_raw - _centroid
P_RIGHT = _B_right_raw - _centroid
P_APEX = _Apex_raw - _centroid
OBJECT_POINTS_PLATE = np.array([P_LEFT, P_RIGHT, P_APEX], dtype=np.float64)  # (3,3)


def p3p_hypotheses(
    object_points: np.ndarray,
    image_points_px: np.ndarray,
    K: np.ndarray,
    dist: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """Run AP3P and return list of candidate T_CT and reprojection errors.

    object_points: (3,3) in plate frame [m].
    image_points_px: (3,2) [px].
    K: (3,3) camera intrinsics.
    dist: distortion coeffs.
    Returns: (T_list, err_list).
    """

    obj = object_points.reshape(3, 1, 3).astype(np.float64)
    img = image_points_px.reshape(3, 1, 2).astype(np.float64)

    retval, rvecs, tvecs = cv2.solveP3P(
        obj, img, K, dist, flags=cv2.SOLVEPNP_AP3P
    )
    T_list = []
    err_list = []

    for i in range(retval):
        R, _ = cv2.Rodrigues(rvecs[i])
        t = tvecs[i].reshape(3)
        T_CT = np.eye(4)
        T_CT[:3, :3] = R
        T_CT[:3, 3] = t

        # Reprojection error
        proj, _ = cv2.projectPoints(obj, rvecs[i], tvecs[i], K, dist)
        proj = proj.reshape(3, 2)
        err = float(np.linalg.norm(proj - image_points_px, axis=1).mean())

        T_list.append(T_CT)
        err_list.append(err)

    return np.array(T_list), np.array(err_list)


def gate_hypothesis(T_CT: np.ndarray, err_px: float) -> Dict[str, bool]:
    """Apply geometric gates to a candidate pose.

    Returns dict of gate_name -> bool plus 'all_passed'.
    """

    R = T_CT[:3, :3]
    t = T_CT[:3, 3]

    gates: Dict[str, bool] = {}

    # Cheirality: target in front of camera.
    gates["cheirality"] = float(t[2]) > 0.05

    # Plate-facing: plate normal (z-axis of plate) in camera frame.
    plate_normal_cam = R[:, 2]
    gates["plate_facing"] = plate_normal_cam[2] < 0.0

    # Operating volume: plausible distance.
    dist = float(np.linalg.norm(t))
    gates["volume"] = 0.2 < dist < 2.0

    # Reprojection error threshold.
    gates["reprojection"] = err_px < 1.5

    # Rotation validity.
    gates["rotation_valid"] = abs(float(np.linalg.det(R)) - 1.0) < 1e-4

    gates["all_passed"] = all(gates.values())
    return gates


def estimate_pose_from_leds(
    observations: Dict[str, LedObservation],
    K: np.ndarray,
    dist: np.ndarray,
) -> PoseResult:
    """Estimate T_CT from three LED observations.

    observations: dict with keys 'left','right','apex' mapping to LedObservation.
    K: (3,3), dist: distortion coeffs.
    """

    # Require all three LEDs and they must be locked.
    required = ("left", "right", "apex")
    for key in required:
        obs = observations.get(key)
        if obs is None or not obs.locked:
            return PoseResult(
                status="INSUFFICIENT_OBSERVATIONS",
                T_CT=None,
                xyz_m=None,
                quat_xyzw=None,
                euler_deg=None,
                reprojection_error_px=None,
                margin_px=None,
            )

    # Assemble image points in plate order: [left, right, apex].
    img_pts = np.vstack([
        observations["left"].center_xy,
        observations["right"].center_xy,
        observations["apex"].center_xy,
    ]).astype(np.float64)

    T_list, err_list = p3p_hypotheses(OBJECT_POINTS_PLATE, img_pts, K, dist)

    # Apply gates.
    gated = []
    for T_CT, err_px in zip(T_list, err_list):
        gates = gate_hypothesis(T_CT, float(err_px))
        gated.append((T_CT, float(err_px), gates))

    passing = [g for g in gated if g[2]["all_passed"]]
    if not passing:
        return PoseResult(
            status="INSUFFICIENT_OBSERVATIONS",
            T_CT=None,
            xyz_m=None,
            quat_xyzw=None,
            euler_deg=None,
            reprojection_error_px=None,
            margin_px=None,
        )

    # Sort by reprojection error.
    passing.sort(key=lambda item: item[1])

    if len(passing) == 1:
        T_best, err_best, _ = passing[0]
        status = "VALID" if err_best < 1.0 else "AMBIGUOUS"
        return _compose_pose_result(T_best, err_best, None, status)

    # Multiple passing hypotheses.
    T_best, err_best, _ = passing[0]
    _, err_second, _ = passing[1]
    margin = float(err_second - err_best)

    status = "VALID" if margin > 0.5 else "AMBIGUOUS"
    return _compose_pose_result(T_best, err_best, margin, status)


def _compose_pose_result(
    T_CT: np.ndarray,
    err_best: float,
    margin: Optional[float],
    status: str,
) -> PoseResult:
    from scipy.spatial.transform import Rotation

    R = T_CT[:3, :3]
    t = T_CT[:3, 3]

    rot = Rotation.from_matrix(R)
    quat_xyzw = rot.as_quat()            # authoritative
    euler_deg = rot.as_euler("xyz", degrees=True)  # display-only

    return PoseResult(
        status=status,
        T_CT=T_CT,
        xyz_m=t.copy(),
        quat_xyzw=quat_xyzw.copy(),
        euler_deg=euler_deg.copy(),
        reprojection_error_px=float(err_best),
        margin_px=margin,
    )


# ---- CLI for offline testing ----------------------------------------------


def parse_cli() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Offline P3P pose estimation from three LED centers."
    )

    parser.add_argument("--K", type=str, required=False,
                        help="Intrinsic matrix as 9 comma-separated values (row-major).")
    parser.add_argument("--dist", type=str, required=False,
                        help="Distortion coeffs as comma-separated values.")

    parser.add_argument("--left", type=str, help="left LED center u,v (e.g. 800,320)")
    parser.add_argument("--right", type=str, help="right LED center u,v")
    parser.add_argument("--apex", type=str, help="apex LED center u,v")

    return parser.parse_args()


def _parse_uv(text: str) -> np.ndarray:
    u_str, v_str = text.split(",")
    return np.array([float(u_str), float(v_str)], dtype=np.float64)


def main() -> int:
    args = parse_cli()

    if args.K:
        K_vals = [float(v) for v in args.K.split(",")]
        if len(K_vals) != 9:
            raise ValueError("--K must have 9 values")
        K = np.array(K_vals, dtype=np.float64).reshape(3, 3)
    else:
        # Placeholder intrinsics; replace with evk4_imx636.yaml values.
        K = np.array([
            [900.0, 0.0, 640.0],
            [0.0, 900.0, 360.0],
            [0.0, 0.0, 1.0],
        ], dtype=np.float64)

    if args.dist:
        dist = np.array([float(v) for v in args.dist.split(",")], dtype=np.float64)
    else:
        dist = np.zeros(5, dtype=np.float64)

    if not (args.left and args.right and args.apex):
        raise ValueError("Must provide --left, --right, and --apex LED centers")

    obs = {
        "left": LedObservation(center_xy=_parse_uv(args.left), coherence=1.0, support=1.0, peak_ratio=1.0, locked=True),
        "right": LedObservation(center_xy=_parse_uv(args.right), coherence=1.0, support=1.0, peak_ratio=1.0, locked=True),
        "apex": LedObservation(center_xy=_parse_uv(args.apex), coherence=1.0, support=1.0, peak_ratio=1.0, locked=True),
    }

    result = estimate_pose_from_leds(obs, K, dist)

    print(f"status={result.status}")
    if result.T_CT is not None:
        print(f"T_CT=\n{result.T_CT}")
        print(f"xyz_m={result.xyz_m}")
        print(f"quat_xyzw={result.quat_xyzw}")
        print(f"euler_deg={result.euler_deg}")
        print(f"reproj_err_px={result.reprojection_error_px:.3f}")
        print(f"margin_px={result.margin_px}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
