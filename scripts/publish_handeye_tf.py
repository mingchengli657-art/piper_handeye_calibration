#!/usr/bin/env python3
"""Publish the final eye-in-hand transform as a ROS 2 static TF."""

from pathlib import Path
import argparse

import rclpy
from geometry_msgs.msg import TransformStamped
from rclpy.node import Node
from tf2_ros.static_transform_broadcaster import StaticTransformBroadcaster
import yaml
from common import validate_transform
from scipy.spatial.transform import Rotation


class HandEyeTfPublisher(Node):
    def __init__(self, result_path, parent_frame, child_frame):
        super().__init__("handeye_static_tf_publisher")
        data = yaml.safe_load(Path(result_path).read_text(encoding="utf-8"))
        if not parent_frame or not child_frame or parent_frame == child_frame:
            raise ValueError("parent and child frames must be nonempty and distinct")
        matrix = validate_transform(data["tcp_T_camera"]["matrix"])
        q = Rotation.from_matrix(matrix[:3, :3]).as_quat()
        t = matrix[:3, 3]
        msg = TransformStamped()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = parent_frame
        msg.child_frame_id = child_frame
        msg.transform.translation.x = float(t[0])
        msg.transform.translation.y = float(t[1])
        msg.transform.translation.z = float(t[2])
        msg.transform.rotation.x = float(q[0])
        msg.transform.rotation.y = float(q[1])
        msg.transform.rotation.z = float(q[2])
        msg.transform.rotation.w = float(q[3])
        self.broadcaster = StaticTransformBroadcaster(self)
        self.broadcaster.sendTransform(msg)
        self.get_logger().info(f"published {parent_frame} -> {child_frame}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--result", required=True, help="path to the result from solve_handeye.py")
    parser.add_argument("--parent-frame", default="flange_link")
    parser.add_argument("--child-frame", required=True)
    args = parser.parse_args()
    rclpy.init()
    node = HandEyeTfPublisher(args.result, args.parent_frame, args.child_frame)
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()

