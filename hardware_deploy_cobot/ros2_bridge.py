"""Optional ROS 2 sensor bridge that emits signed SensorSentry telemetry.

Run this on the robot only after setting ``SENSORSENTRY_COBOT_HMAC_KEY``. It
subscribes to standard Nav2/robot topics and sends the minimum required data to
the local secure UDP listener with adaptive QoS and thread-safe session tracking.
"""

from __future__ import annotations

import argparse
import base64
import hashlib
import hmac
import json
import math
import os
import socket
import threading
import time
import uuid
from typing import Any, Dict, Optional

from telemetry import canonical_packet, load_hmac_key


class Ros2TelemetryBridge:
    def __init__(
        self,
        robot_id: str,
        key: bytes,
        host: str,
        port: int,
        namespace: str = "",
        odom_topic: str = "odom",
        scan_topic: str = "scan",
        amcl_topic: str = "amcl_pose",
        odom_frame: str = "map",
    ) -> None:
        try:
            import rclpy
            from geometry_msgs.msg import PoseWithCovarianceStamped, Twist
            from nav_msgs.msg import Odometry
            from rclpy.qos import (DurabilityPolicy, HistoryPolicy, QoSProfile,
                                   ReliabilityPolicy, qos_profile_sensor_data)
            from sensor_msgs.msg import LaserScan
        except ImportError as exc:
            raise RuntimeError("ROS 2 sensor packages are required for ros2_bridge.py") from exc

        self.rclpy = rclpy
        self.Odometry = Odometry
        self.LaserScan = LaserScan
        self.PoseWithCovarianceStamped = PoseWithCovarianceStamped
        self.robot_id = robot_id
        self.key = key
        self.destination = (host, port)
        self.odom_frame = odom_frame
        self.sequence = 0
        self.session_id = uuid.uuid4().hex[:12]
        self._lock = threading.Lock()

        rclpy.init()
        self.node = rclpy.create_node("sensorsentry_cobot_telemetry", namespace=namespace or None)
        self.socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

        # Resilient QoS for sensor topics: Best Effort with Keep Last history
        sensor_qos = qos_profile_sensor_data

        # Subscription to raw odometry
        self.node.create_subscription(Odometry, odom_topic, self._on_odom, sensor_qos)

        # Subscription to LiDAR scan with sensor data QoS (best-effort)
        self.node.create_subscription(LaserScan, scan_topic, self._on_scan, sensor_qos)

        # Subscription to AMCL map-frame pose if available
        self.node.create_subscription(
            PoseWithCovarianceStamped, amcl_topic, self._on_amcl_pose, sensor_qos
        )

    def _send(self, packet: Dict[str, Any]) -> None:
        with self._lock:
            self.sequence += 1
            packet.update({
                "schema_version": 1,
                "robot_id": self.robot_id,
                "sequence": self.sequence,
                "session_id": self.session_id,
                "timestamp": time.time(),
            })
            packet["signature"] = hmac.new(self.key, canonical_packet(packet), hashlib.sha256).hexdigest()
            self.socket.sendto(json.dumps(packet, separators=(",", ":")).encode("utf-8"), self.destination)

    @staticmethod
    def _quaternion_to_yaw(orientation: Any) -> float:
        x, y, z, w = orientation.x, orientation.y, orientation.z, orientation.w
        siny_cosp = 2.0 * (w * z + x * y)
        cosy_cosp = 1.0 - 2.0 * (y * y + z * z)
        return math.atan2(siny_cosp, cosy_cosp)

    def _on_odom(self, message: Any) -> None:
        yaw = self._quaternion_to_yaw(message.pose.pose.orientation)
        self._send({
            "sensor": "ODOMETRY",
            "frame": self.odom_frame,
            "position_m": [float(message.pose.pose.position.x), float(message.pose.pose.position.y)],
            "yaw_rad": yaw,
            "linear_velocity_m_s": abs(float(message.twist.twist.linear.x)),
            "angular_velocity_rad_s": float(message.twist.twist.angular.z),
        })

    def _on_amcl_pose(self, message: Any) -> None:
        yaw = self._quaternion_to_yaw(message.pose.pose.orientation)
        self._send({
            "sensor": "LOCALIZATION",
            "frame": "map",
            "position_m": [float(message.pose.pose.position.x), float(message.pose.pose.position.y)],
            "yaw_rad": yaw,
            "linear_velocity_m_s": 0.0,
            "angular_velocity_rad_s": 0.0,
        })

    def _on_scan(self, message: Any) -> None:
        valid = [range_m for range_m in message.ranges if message.range_min <= range_m <= message.range_max]
        if valid:
            self._send({"sensor": "LIDAR", "frame": "base_link", "min_range_m": float(min(valid))})

    def spin(self) -> None:
        try:
            self.rclpy.spin(self.node)
        finally:
            self.node.destroy_node()
            self.rclpy.shutdown()
            self.socket.close()


def main() -> None:
    parser = argparse.ArgumentParser(description="ROS 2 to authenticated SensorSentry telemetry bridge")
    parser.add_argument("--robot-id", default="COBOT-ALPHA-01")
    parser.add_argument("--udp-host", default="127.0.0.1")
    parser.add_argument("--udp-port", type=int, default=5010)
    parser.add_argument("--ros-namespace", default="")
    parser.add_argument("--odom-topic", default="odom")
    parser.add_argument("--scan-topic", default="scan")
    parser.add_argument("--amcl-topic", default="amcl_pose")
    parser.add_argument("--odom-frame", default="map", choices=("map", "odom"))
    parser.add_argument("--hmac-key-env", default="SENSORSENTRY_COBOT_HMAC_KEY")
    args = parser.parse_args()
    encoded_key = os.getenv(args.hmac_key_env)
    if not encoded_key:
        parser.error(f"{args.hmac_key_env} must contain a base64 32-byte HMAC key")
    Ros2TelemetryBridge(
        robot_id=args.robot_id,
        key=load_hmac_key(encoded_key),
        host=args.udp_host,
        port=args.udp_port,
        namespace=args.ros_namespace,
        odom_topic=args.odom_topic,
        scan_topic=args.scan_topic,
        amcl_topic=args.amcl_topic,
        odom_frame=args.odom_frame,
    ).spin()


if __name__ == "__main__":
    main()
