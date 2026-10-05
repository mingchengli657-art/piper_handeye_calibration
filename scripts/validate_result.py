#!/usr/bin/env python3
"""Validate an eye-in-hand result using base_T_tcp * tcp_T_camera * camera_T_target."""

from pathlib import Path
import argparse
import csv
import json
import math

import numpy as np
import yaml
from scipy.spatial.transform import Rotation
from solve_handeye import load_samples
from common import validate_transform


def rotation_error_deg(a, b):
    relative = a[:3, :3].T @ b[:3, :3]
    value = np.clip((np.trace(relative) - 1.0) / 2.0, -1.0, 1.0)
    return math.degrees(math.acos(float(value)))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--metadata-dir", default="data/samples/metadata")
    parser.add_argument("--result", default="results/handeye_result.yaml")
    parser.add_argument("--min-corners", type=int, default=88)
    parser.add_argument("--min-sample-id", type=int, default=None)
    parser.add_argument("--max-sample-id", type=int, default=None)
    parser.add_argument("--output", default="results/validation_report.yaml")
    parser.add_argument("--max-dt", type=float, default=0.05)
    args = parser.parse_args()

    result = yaml.safe_load(Path(args.result).read_text(encoding="utf-8"))
    tcp_camera = validate_transform(result["tcp_T_camera"]["matrix"])
    rows = []
    excluded = []
    loaded, rejected = load_samples(args.metadata_dir, args.min_corners, args.max_dt)
    for path, data in loaded:
        sample_id = int(data["sample_id"])
        if args.min_sample_id is not None and sample_id < args.min_sample_id:
            continue
        if args.max_sample_id is not None and sample_id > args.max_sample_id:
            continue
        if data.get("charuco_corner_count", 0) < args.min_corners:
            excluded.append(sample_id)
            continue
        base_tcp = np.asarray(data["base_T_tcp"], dtype=float)
        rvec = np.asarray(data["camera_T_target_rvec"], dtype=float).reshape(3, 1)
        tvec = np.asarray(data["camera_T_target_tvec_m"], dtype=float).reshape(3)
        target_rotation, _ = __import__("cv2").Rodrigues(rvec)
        camera_target = np.eye(4, dtype=float)
        camera_target[:3, :3] = target_rotation
        camera_target[:3, 3] = tvec
        base_target = base_tcp @ tcp_camera @ camera_target
        rows.append((data, base_target))

    if len(rows) < 3:
        raise SystemExit(f"Only {len(rows)} usable samples")

    target_rotations = Rotation.from_matrix(np.stack([t[:3, :3] for _, t in rows]))
    mean_rotation = target_rotations.mean().as_matrix()
    mean_translation = np.mean([t[:3, 3] for _, t in rows], axis=0)
    reference = np.eye(4, dtype=float)
    reference[:3, :3] = mean_rotation
    reference[:3, 3] = mean_translation

    details = []
    for data, base_target in rows:
        translation_error = float(np.linalg.norm(base_target[:3, 3] - mean_translation))
        rotation_error = float(rotation_error_deg(reference, base_target))
        details.append(
            {
                "sample_id": data["sample_id"],
                "charuco_corner_count": data["charuco_corner_count"],
                "translation_error_m": translation_error,
                "translation_error_mm": translation_error * 1000.0,
                "rotation_error_deg": rotation_error,
            }
        )

    t_errors = np.asarray([d["translation_error_m"] for d in details])
    r_errors = np.asarray([d["rotation_error_deg"] for d in details])
    summary = {
        "result_file": str(Path(args.result).resolve()),
        "min_charuco_corners": args.min_corners,
        "sample_count": len(details),
        "excluded_sample_ids": excluded,
        "rejected_samples": rejected,
        "max_timestamp_difference_s": args.max_dt,
        "overlap_with_training_ids": sorted(set(result.get("selection", {}).get("used_samples", [])) & {d["sample_id"] for d in details}),
        "base_T_target_translation_mean_m": mean_translation.tolist(),
        "translation_error_mean_mm": float(t_errors.mean() * 1000.0),
        "translation_error_rms_mm": float(np.sqrt(np.mean(t_errors**2)) * 1000.0),
        "translation_error_max_mm": float(t_errors.max() * 1000.0),
        "rotation_error_mean_deg": float(r_errors.mean()),
        "rotation_error_rms_deg": float(np.sqrt(np.mean(r_errors**2))),
        "rotation_error_max_deg": float(r_errors.max()),
        "worst_translation_samples": sorted(details, key=lambda d: d["translation_error_m"], reverse=True)[:10],
        "worst_rotation_samples": sorted(details, key=lambda d: d["rotation_error_deg"], reverse=True)[:10],
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(yaml.safe_dump(summary, sort_keys=False), encoding="utf-8")

    csv_path = output.with_suffix(".csv")
    with csv_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=details[0].keys())
        writer.writeheader()
        writer.writerows(details)

    if summary["overlap_with_training_ids"]:
        print("NOTE: sample IDs overlap with training; verify data provenance before claiming independent validation.")
    print(f"samples: {len(details)} (excluded: {len(excluded) + len(rejected)})")
    print(f"translation error: mean={summary['translation_error_mean_mm']:.3f} mm, "
          f"rms={summary['translation_error_rms_mm']:.3f} mm, "
          f"max={summary['translation_error_max_mm']:.3f} mm")
    print(f"rotation error: mean={summary['rotation_error_mean_deg']:.3f} deg, "
          f"rms={summary['rotation_error_rms_deg']:.3f} deg, "
          f"max={summary['rotation_error_max_deg']:.3f} deg")
    print(f"saved report: {output}")
    print(f"saved per-sample CSV: {csv_path}")


if __name__ == "__main__":
    main()
