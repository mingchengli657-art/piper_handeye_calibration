#!/usr/bin/env python3
"""Live ChArUco detection for the D405 color stream.

This is a camera/target check only. It does not perform hand-eye calibration.
Press 's' to save an annotated image and 'q' to quit.
"""

from pathlib import Path
import argparse
import time

from board_config import DEFAULT_BOARD, load_board

import cv2
import numpy as np
import rclpy
from cv_bridge import CvBridge
from rclpy.node import Node
from sensor_msgs.msg import CameraInfo, Image


class Detector(Node):
    def __init__(self, image_topic, info_topic, save_dir, board_path=DEFAULT_BOARD):
        super().__init__("charuco_detector")
        self.bridge = CvBridge()
        self.camera_matrix = None
        self.dist_coeffs = None
        self.latest_frame = None
        self.save_dir = Path(save_dir).expanduser()
        self.save_dir.mkdir(parents=True, exist_ok=True)

        self.board_config, self.dictionary, self.board = load_board(board_path)
        self.corner_total = (int(self.board_config["squares_x"]) - 1) * (int(self.board_config["squares_y"]) - 1)
        self.detector_params = (cv2.aruco.DetectorParameters_create() if hasattr(cv2.aruco, "DetectorParameters_create") else cv2.aruco.DetectorParameters())

        self.create_subscription(CameraInfo, info_topic, self.info_callback, 10)
        self.create_subscription(Image, image_topic, self.image_callback, 10)
        self.get_logger().info(f"image: {image_topic}")
        self.get_logger().info(f"camera_info: {info_topic}")

    def info_callback(self, msg):
        self.camera_matrix = np.asarray(msg.k, dtype=np.float64).reshape(3, 3)
        self.dist_coeffs = np.asarray(msg.d, dtype=np.float64)

    def image_callback(self, msg):
        try:
            frame = self.bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
        except Exception as exc:
            self.get_logger().error(f"image conversion failed: {exc}")
            return

        self.latest_frame = frame.copy()
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        corners, ids, _ = cv2.aruco.detectMarkers(
            gray, self.dictionary, parameters=self.detector_params
        )
        marker_count = 0 if ids is None else len(ids)
        charuco_count = 0
        board_pose_ok = False

        if ids is not None:
            cv2.aruco.drawDetectedMarkers(frame, corners, ids)
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
                            frame, charuco_corners, charuco_ids
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
                                board_pose_ok = True
                                cv2.drawFrameAxes(
                                    frame,
                                    self.camera_matrix,
                                    self.dist_coeffs,
                                    rvec,
                                    tvec,
                                    0.05,
                                    2,
                                )
                except cv2.error as exc:
                    self.get_logger().warn(f"ChArUco processing failed: {exc}")

        status = "POSE OK" if board_pose_ok else "DETECTING"
        color = (0, 255, 0) if board_pose_ok else (0, 165, 255)
        cv2.putText(
            frame,
            f"markers: {marker_count}  corners: {charuco_count}/{self.corner_total}  {status}",
            (20, 35),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.75,
            color,
            2,
            cv2.LINE_AA,
        )
        cv2.putText(
            frame,
            "s: save image   q: quit",
            (20, 68),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )
        cv2.imshow("D405 ChArUco detector", frame)
        key = cv2.waitKey(1) & 0xFF
        if key == ord("s"):
            stamp = time.strftime("%Y%m%d_%H%M%S")
            path = self.save_dir / f"charuco_{stamp}.png"
            cv2.imwrite(str(path), frame)
            self.get_logger().info(f"saved {path}")
        elif key == ord("q"):
            rclpy.shutdown()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--image-topic", default="/camera/d405/color/image_raw")
    parser.add_argument("--camera-info-topic", default="/camera/d405/color/camera_info")
    parser.add_argument("--save-dir", default="data/images")
    parser.add_argument("--board", default=str(DEFAULT_BOARD))
    args = parser.parse_args()

    rclpy.init()
    node = Detector(args.image_topic, args.camera_info_topic, args.save_dir, args.board)
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        cv2.destroyAllWindows()
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
