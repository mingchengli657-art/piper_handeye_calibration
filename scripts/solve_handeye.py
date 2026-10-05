#!/usr/bin/env python3
"""Solve Piper X eye-in-hand calibration from collected JSON samples.

The collector stores base_T_tcp and camera_T_target. OpenCV expects exactly
these gripper-to-base and target-to-camera transforms and returns
camera-to-gripper (T_tcp_camera for this project).
"""

from pathlib import Path
import argparse
import json
import math
from common import validate_sample, validate_transform

import cv2
import numpy as np
import yaml
from scipy.spatial.transform import Rotation


METHODS = {
    "tsai": cv2.CALIB_HAND_EYE_TSAI,
    "park": cv2.CALIB_HAND_EYE_PARK,
    "horaud": cv2.CALIB_HAND_EYE_HORAUD,
    "andreff": cv2.CALIB_HAND_EYE_ANDREFF,
    "daniilidis": cv2.CALIB_HAND_EYE_DANIILIDIS,
}


def load_samples(metadata_dir, min_corners, max_dt):
    rows = []
    excluded = []
    for path in sorted(Path(metadata_dir).glob("sample_*.json")):
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            validate_sample(data)
            reasons = []
            if data.get("charuco_corner_count", 0) < min_corners:
                reasons.append("too_few_corners")
            if data.get("timestamp_difference_s", 999.0) > max_dt:
                reasons.append("timestamp_mismatch")
            if data.get("camera_T_target_rvec") is None:
                reasons.append("missing_target_pose")
            if data.get("base_T_tcp") is None:
                reasons.append("missing_robot_pose")
            if reasons:
                excluded.append({"file": str(path), "reasons": reasons})
            else:
                rows.append((path, data))
        except Exception as exc:
            excluded.append({"file": str(path), "reasons": [f"read_error:{exc}"]})
    return rows, excluded


def transforms(rows):
    r_gripper2base, t_gripper2base = [], []
    r_target2cam, t_target2cam = [], []
    base_tcp, cam_target = [], []
    for _, data in rows:
        b_t = np.asarray(data["base_T_tcp"], dtype=np.float64)
        r_gripper2base.append(b_t[:3, :3])
        t_gripper2base.append(b_t[:3, 3].reshape(3, 1))
        rvec = np.asarray(data["camera_T_target_rvec"], dtype=np.float64).reshape(3, 1)
        tvec = np.asarray(data["camera_T_target_tvec_m"], dtype=np.float64).reshape(3, 1)
        rmat, _ = cv2.Rodrigues(rvec)
        r_target2cam.append(rmat)
        t_target2cam.append(tvec)
        base_tcp.append(b_t)
        c_t = np.eye(4, dtype=np.float64)
        c_t[:3, :3] = rmat
        c_t[:3, 3] = tvec.reshape(3)
        cam_target.append(c_t)
    return r_gripper2base, t_gripper2base, r_target2cam, t_target2cam, base_tcp, cam_target


def make_transform(rotation, translation):
    t = np.eye(4, dtype=np.float64)
    t[:3, :3] = rotation
    t[:3, 3] = np.asarray(translation, dtype=np.float64).reshape(3)
    return t


def rotation_error_deg(a, b):
    relative = a[:3, :3].T @ b[:3, :3]
    value = max(-1.0, min(1.0, (np.trace(relative) - 1.0) / 2.0))
    return math.degrees(math.acos(value))


def consistency_metrics(base_tcp, cam_target, tcp_camera):
    """For fixed target, base_T_tcp * tcp_T_camera * camera_T_target is constant."""
    base_target = [b @ tcp_camera @ c for b, c in zip(base_tcp, cam_target)]
    translations = np.asarray([t[:3, 3] for t in base_target])
    mean_translation = translations.mean(axis=0)
    translation_errors = np.linalg.norm(translations - mean_translation, axis=1)
    ref = base_target[0]
    rotation_errors = np.asarray([rotation_error_deg(ref, t) for t in base_target])
    return {
        "base_target_translation_mean_m": mean_translation.tolist(),
        "base_target_translation_error_mean_m": float(translation_errors.mean()),
        "base_target_translation_error_rms_m": float(np.sqrt(np.mean(translation_errors**2))),
        "base_target_translation_error_max_m": float(translation_errors.max()),
        "base_target_rotation_error_mean_deg": float(rotation_errors.mean()),
        "base_target_rotation_error_rms_deg": float(np.sqrt(np.mean(rotation_errors**2))),
        "base_target_rotation_error_max_deg": float(rotation_errors.max()),
    }


def solve(rows, method_id):
    r_gb, t_gb, r_tc, t_tc, base_tcp, cam_target = transforms(rows)
    # At least two nonparallel relative rotation axes are required.
    axes = np.asarray([Rotation.from_matrix(r_gb[0].T @ r).as_rotvec() for r in r_gb[1:]])
    if len(rows) < 3 or np.linalg.matrix_rank(axes, tol=1e-3) < 2:
        raise ValueError("degenerate robot poses: rotate around at least two distinct axes")
    r_cg, t_cg = cv2.calibrateHandEye(r_gb, t_gb, r_tc, t_tc, method=method_id)
    tcp_camera = validate_transform(make_transform(r_cg, t_cg))
    metrics = consistency_metrics(base_tcp, cam_target, tcp_camera)
    metrics["method"] = next(name for name, value in METHODS.items() if value == method_id)
    return tcp_camera, metrics


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--metadata-dir", default="data/samples/metadata")
    parser.add_argument("--output", default="results/handeye_result.yaml")
    parser.add_argument("--min-corners", type=int, default=88)
    parser.add_argument("--max-dt", type=float, default=0.05)
    parser.add_argument(
        "--exclude-sample-ids",
        type=int,
        nargs="*",
        default=[],
        help="sample IDs to leave out (for example, a held-out validation set)",
    )
    args = parser.parse_args()

    rows, excluded = load_samples(args.metadata_dir, args.min_corners, args.max_dt)
    if args.exclude_sample_ids:
        held_out = set(args.exclude_sample_ids)
        kept = []
        for path, data in rows:
            if int(data["sample_id"]) in held_out:
                excluded.append({"file": str(path), "reasons": ["explicitly_held_out"]})
            else:
                kept.append((path, data))
        rows = kept
    if len(rows) < 3:
        raise SystemExit(f"Only {len(rows)} usable samples; at least 3 are required")

    results = {}
    matrices = {}
    for name, method in METHODS.items():
        try:
            matrix, metrics = solve(rows, method)
            results[name] = metrics
            matrices[name] = matrix
        except (cv2.error, ValueError, np.linalg.LinAlgError) as exc:
            results[name] = {"method": name, "error": str(exc)}

    if not matrices:
        raise SystemExit(f"All calibration methods failed: {results}")

    # Retain Park as the historical default; report every method for comparison.
    selected_name = "park" if "park" in matrices else next(iter(matrices))
    selected = matrices[selected_name]
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    report = {
        "configuration": {
            "type": "eye_in_hand",
            "robot_pose": "base_T_tcp",
            "camera_target_pose": "camera_T_target",
            "output_transform": "tcp_T_camera",
        },
        "selection": {
            "selected_method": selected_name,
            "min_charuco_corners": args.min_corners,
            "max_timestamp_difference_s": args.max_dt,
            "used_sample_count": len(rows),
            "excluded_sample_count": len(excluded),
            "explicitly_excluded_sample_ids": args.exclude_sample_ids,
            "used_samples": [data["sample_id"] for _, data in rows],
            "excluded_samples": excluded,
        },
        "tcp_T_camera": {
            "matrix": selected.tolist(),
            "translation_m": selected[:3, 3].tolist(),
            "quaternion_xyzw": Rotation.from_matrix(selected[:3, :3]).as_quat().tolist(),
            "rvec_rad": cv2.Rodrigues(selected[:3, :3])[0].reshape(3).tolist(),
        },
        "methods": results,
    }
    output.write_text(yaml.safe_dump(report, sort_keys=False), encoding="utf-8")

    print(f"usable samples: {len(rows)}")
    print(f"excluded samples: {len(excluded)}")
    print(f"selected method: {selected_name}")
    print("tcp_T_camera matrix:")
    print(np.array2string(selected, precision=8, suppress_small=True))
    for name, metrics in results.items():
        if "error" in metrics:
            print(f"{name}: ERROR {metrics['error']}")
        else:
            print(
                f"{name}: translation_rms={metrics['base_target_translation_error_rms_m']*1000:.2f} mm, "
                f"rotation_rms={metrics['base_target_rotation_error_rms_deg']:.3f} deg"
            )
    print(f"saved report: {output}")


if __name__ == "__main__":
    main()
