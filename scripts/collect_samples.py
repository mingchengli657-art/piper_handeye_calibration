#!/usr/bin/env python3
"""Collect eye-in-hand samples from a live D405 and Piper X.

The operator moves the robot to a stationary pose, waits for it to settle,
then presses 'c'. The node saves the color image, ChArUco pose and the
nearest Piper TCP pose in one sample directory. This program does not move
the robot.
"""

from collections import deque
from pathlib import Path
import argparse
import json
import time

from board_config import DEFAULT_BOARD, load_board

import cv2
import numpy as np
import rclpy
from cv_bridge import CvBridge
from geometry_msgs.msg import PoseStamped
from rclpy.node import Node
from sensor_msgs.msg import CameraInfo, Image


def stamp_seconds(stamp):
    return float(stamp.sec) + float(stamp.nanosec) * 1e-9


def pose_to_matrix(msg):
    q = msg.pose.orientation
    t = msg.pose.position
    quat = np.array([q.x, q.y, q.z, q.w], dtype=np.float64)
    norm = np.linalg.norm(quat)
    if norm < 1e-12:
        raise ValueError("robot pose has a zero quaternion")
    quat /= norm
    x, y, z, w = quat
    rotation = np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ],
        dtype=np.float64,
    )
    matrix = np.eye(4, dtype=np.float64)
    matrix[:3, :3] = rotation
    matrix[:3, 3] = [t.x, t.y, t.z]
    return matrix


class SampleCollector(Node):
    def __init__(self, image_topic, info_topic, pose_topic, output_dir, sync_limit, target_count, board_path=DEFAULT_BOARD):
        super().__init__("eyehand_sample_collector")
        self.bridge = CvBridge()
        self.output_dir = Path(output_dir).expanduser()
        self.image_dir = self.output_dir / "images"
        self.overlay_dir = self.output_dir / "overlays"
        self.meta_dir = self.output_dir / "metadata"
        for directory in (self.image_dir, self.overlay_dir, self.meta_dir):
            directory.mkdir(parents=True, exist_ok=True)
        self.sync_limit = float(sync_limit)
        self.target_count = int(target_count)
        self.captured_count = 0

        self.camera_matrix = None
        self.dist_coeffs = None
        self.latest_frame = None
        self.latest_overlay = None
        self.latest_result = None
        self.image_stamp = None
        self.last_image_received = None
        self.last_captured_stamp = None
        self.pose_buffer = deque(maxlen=400)
        self.sample_index = self._next_index()

        self.board_config, self.dictionary, self.board = load_board(board_path)
        self.corner_total = (int(self.board_config["squares_x"]) - 1) * (int(self.board_config["squares_y"]) - 1)
        self.detector_params = (cv2.aruco.DetectorParameters_create() if hasattr(cv2.aruco, "DetectorParameters_create") else cv2.aruco.DetectorParameters())

        self.create_subscription(CameraInfo, info_topic, self.info_callback, 10)
        self.create_subscription(Image, image_topic, self.image_callback, 10)
        self.create_subscription(PoseStamped, pose_topic, self.pose_callback, 100)
        self.get_logger().info(f"image topic: {image_topic}")
        self.get_logger().info(f"camera info topic: {info_topic}")
        self.get_logger().info(f"robot pose topic: {pose_topic}")
        self.get_logger().info(f"saving samples under: {self.output_dir}")

    def _next_index(self):
        existing = list(self.meta_dir.glob("sample_*.json"))
        if not existing:
            return 1
        numbers = []
        for path in existing:
            try:
                numbers.append(int(path.stem.split("_")[-1]))
            except ValueError:
                pass
        return max(numbers, default=0) + 1

    def info_callback(self, msg):
        self.camera_matrix = np.asarray(msg.k, dtype=np.float64).reshape(3, 3)
        self.dist_coeffs = np.asarray(msg.d, dtype=np.float64)

    def pose_callback(self, msg):
        try:
            matrix = pose_to_matrix(msg)
        except ValueError as exc:
            self.get_logger().warn(str(exc))
            return
        stamp = stamp_seconds(msg.header.stamp)
        if stamp <= 0 or not np.isfinite(matrix).all():
            return
        self.pose_buffer.append((stamp, matrix))

    def image_callback(self, msg):
        try:
            frame = self.bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
        except Exception as exc:
            self.get_logger().error(f"image conversion failed: {exc}")
            return

        self.last_image_received = time.monotonic()
        self.latest_frame = frame.copy()
        self.image_stamp = stamp_seconds(msg.header.stamp)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        corners, ids, _ = cv2.aruco.detectMarkers(
            gray, self.dictionary, parameters=self.detector_params
        )
        marker_count = 0 if ids is None else int(len(ids))
        charuco_count = 0
        camera_target = None
        pose_ok = False
        overlay = frame.copy()

        if ids is not None:
            cv2.aruco.drawDetectedMarkers(overlay, corners, ids)
            if self.camera_matrix is not None and self.dist_coeffs is not None:
                try:
                    retval, charuco_corners, charuco_ids = cv2.aruco.interpolateCornersCharuco(
                        corners,
                        ids,
                        gray,
                        self.board,
                        cameraMatrix=self.camera_matrix,
                        distCoeffs=self.dist_coeffs,
                    )
                    if charuco_ids is not None:
                        charuco_count = int(retval)
                        cv2.aruco.drawDetectedCornersCharuco(
                            overlay, charuco_corners, charuco_ids
                        )
                        if charuco_count >= 4:
                            ok, rvec, tvec = cv2.aruco.estimatePoseCharucoBoard(
                                charuco_corners,
                                charuco_ids,
                                self.board,
                                self.camera_matrix,
                                self.dist_coeffs,
                                np.zeros((3, 1), dtype=np.float64),
                                np.zeros((3, 1), dtype=np.float64),
                            )
                            if ok:
                                pose_ok = True
                                camera_target = {
                                    "rvec": np.asarray(rvec, dtype=float).reshape(3).tolist(),
                                    "tvec": np.asarray(tvec, dtype=float).reshape(3).tolist(),
                                }
                                cv2.drawFrameAxes(
                                    overlay,
                                    self.camera_matrix,
                                    self.dist_coeffs,
                                    rvec,
                                    tvec,
                                    0.05,
                                    2,
                                )
                except cv2.error as exc:
                    self.get_logger().warn(f"ChArUco processing failed: {exc}")

        self.latest_overlay = overlay
        self.latest_result = {
            "marker_count": marker_count,
            "charuco_count": charuco_count,
            "pose_ok": pose_ok,
            "camera_target": camera_target,
        }

    def nearest_robot_pose(self):
        if self.image_stamp is None or not self.pose_buffer:
            return None, None
        stamp, matrix = min(
            self.pose_buffer, key=lambda item: abs(item[0] - self.image_stamp)
        )
        return stamp, matrix

    def try_capture(self):
        if self.latest_frame is None or self.latest_result is None:
            self.get_logger().warn("no image received yet")
            return
        if self.image_stamp <= 0 or time.monotonic() - self.last_image_received > 1.0:
            self.get_logger().warn("capture rejected: zero timestamp or stale image")
            return
        if self.image_stamp == self.last_captured_stamp:
            self.get_logger().warn("capture rejected: this image was already saved")
            return
        if not self.latest_result["pose_ok"] or self.latest_result["charuco_count"] < 20:
            self.get_logger().warn("capture rejected: need a valid pose and at least 20 ChArUco corners")
            return
        robot_stamp, base_tcp = self.nearest_robot_pose()
        if robot_stamp is None:
            self.get_logger().warn("capture rejected: no robot pose received yet")
            return
        time_delta = abs(robot_stamp - self.image_stamp)
        if time_delta > self.sync_limit:
            self.get_logger().warn(
                f"capture rejected: nearest robot/image timestamp difference is {time_delta:.3f}s "
                f"(limit {self.sync_limit:.3f}s)"
            )
            return

        number = self.sample_index
        stem = f"sample_{number:03d}"
        image_path = self.image_dir / f"{stem}.png"
        overlay_path = self.overlay_dir / f"{stem}.png"
        metadata_path = self.meta_dir / f"{stem}.json"
        if not cv2.imwrite(str(image_path), self.latest_frame) or not cv2.imwrite(str(overlay_path), self.latest_overlay):
            raise OSError("failed to write sample images")

        metadata = {
            "sample_id": number,
            "image_file": str(image_path),
            "overlay_file": str(overlay_path),
            "image_stamp": self.image_stamp,
            "robot_pose_stamp": robot_stamp,
            "timestamp_difference_s": time_delta,
            "marker_count": self.latest_result["marker_count"],
            "charuco_corner_count": self.latest_result["charuco_count"],
            "camera_matrix": self.camera_matrix.tolist(),
            "dist_coeffs": self.dist_coeffs.tolist(),
            "base_T_tcp": base_tcp.tolist(),
            "camera_T_target_rvec": self.latest_result["camera_target"]["rvec"],
            "camera_T_target_tvec_m": self.latest_result["camera_target"]["tvec"],
            "board": self.board_config,
        }
        metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
        self.get_logger().info(
            f"saved {stem}: corners={metadata['charuco_corner_count']}, "
            f"image/robot dt={time_delta:.3f}s"
        )
        self.last_captured_stamp = self.image_stamp
        self.sample_index += 1
        self.captured_count += 1
        if self.target_count > 0 and self.captured_count >= self.target_count:
            self.get_logger().info(f"target of {self.target_count} samples reached")
            rclpy.shutdown()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--image-topic", default="/camera/d405/color/image_raw")
    parser.add_argument("--camera-info-topic", default="/camera/d405/color/camera_info")
    parser.add_argument("--pose-topic", default="/feedback/tcp_pose")
    parser.add_argument("--output-dir", default="data/samples")
    parser.add_argument("--sync-limit", type=float, default=0.05)
    parser.add_argument("--count", type=int, default=0, help="stop after this many captures; 0 means unlimited")
    parser.add_argument("--board", default=str(DEFAULT_BOARD))
    args = parser.parse_args()

    rclpy.init()
    node = SampleCollector(
        args.image_topic,
        args.camera_info_topic,
        args.pose_topic,
        args.output_dir,
        args.sync_limit,
        args.count,
        args.board,
    )
    try:
        while rclpy.ok():
            rclpy.spin_once(node, timeout_sec=0.01)
            if node.latest_overlay is not None and node.latest_result is not None:
                display = node.latest_overlay.copy()
                result = node.latest_result
                ok = result["pose_ok"] and result["charuco_count"] >= 20
                color = (0, 255, 0) if ok else (0, 165, 255)
                cv2.putText(
                    display,
                    f"markers: {result['marker_count']}  corners: {result['charuco_count']}/{node.corner_total}",
                    (20, 35),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.75,
                    color,
                    2,
                    cv2.LINE_AA,
                )
                cv2.putText(
                    display,
                    f"c: capture   q: quit   saved: {node.captured_count}"
                    + (f"/{node.target_count}" if node.target_count > 0 else ""),
                    (20, 68),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.65,
                    (255, 255, 255),
                    2,
                    cv2.LINE_AA,
                )
                cv2.imshow("Eye-hand sample collector", display)
                key = cv2.waitKey(1) & 0xFF
                if key == ord("c"):
                    node.try_capture()
                elif key == ord("q"):
                    break
    except KeyboardInterrupt:
        pass
    finally:
        cv2.destroyAllWindows()
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
