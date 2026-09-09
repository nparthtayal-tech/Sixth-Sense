"""Safe containment interface for a Nav2 mobile cobot.

The interface intentionally provides no drive, arm, or resume method. The only
physical action it can request is to cancel navigation and publish zero twist.
"""

from __future__ import annotations

import time
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional


@dataclass(frozen=True)
class StopRequest:
    timestamp: float
    reason: str


class CobotControl(ABC):
    @abstractmethod
    def request_safe_stop(self, reason: str) -> bool:
        """Contain the robot. True means the controller accepted the request."""

    def close(self) -> None:
        """Release optional controller resources."""


class AdvisoryCobotControl(CobotControl):
    """Default safe mode: record containment requests without robot I/O."""

    def __init__(self) -> None:
        self.last_request: Optional[StopRequest] = None

    def request_safe_stop(self, reason: str) -> bool:
        self.last_request = StopRequest(time.time(), reason)
        return False


class Ros2CobotControl(CobotControl):
    """ROS 2 containment adapter, constructed only after explicit opt-in.

    It publishes zero velocity repeatedly and requests cancellation of Nav2's
    NavigateToPose action. Motor safety hardware must independently enforce the
    stop; this adapter is an additional software layer, not a safety PLC.
    """

    def __init__(
        self,
        namespace: str = "",
        cmd_vel_topic: str = "cmd_vel",
        navigate_action: str = "navigate_to_pose",
        zero_publish_count: int = 3,
    ) -> None:
        try:
            import rclpy
            from action_msgs.srv import CancelGoal
            from geometry_msgs.msg import Twist
        except ImportError as exc:
            raise RuntimeError("ROS 2 Python packages are required for the ROS 2 controller") from exc
        self._rclpy = rclpy
        self._cancel_goal_type = CancelGoal
        self._twist_type = Twist
        self._owns_rclpy_context = not rclpy.ok()
        if self._owns_rclpy_context:
            rclpy.init()
        self.node = rclpy.create_node("sensorsentry_cobot_stop", namespace=namespace or "")
        self._publisher = self.node.create_publisher(Twist, cmd_vel_topic, 10)
        self._cancel = self.node.create_client(CancelGoal, navigate_action + "/_action/cancel_goal")
        self._zero_publish_count = max(1, int(zero_publish_count))

    def request_safe_stop(self, reason: str) -> bool:
        # Publish stop first: action cancellation can take longer or be absent.
        zero_twist = self._twist_type()
        for _ in range(self._zero_publish_count):
            self._publisher.publish(zero_twist)
            self._rclpy.spin_once(self.node, timeout_sec=0.0)

        # A zero GoalInfo means "all goals" according to the ROS action cancel
        # protocol. It is used solely during a confirmed containment event.
        if self._cancel.wait_for_service(timeout_sec=0.5):
            future = self._cancel.call_async(self._cancel_goal_type.Request())
            self._rclpy.spin_until_future_complete(self.node, future, timeout_sec=1.0)
        return True

    def close(self) -> None:
        self.node.destroy_node()
        if self._owns_rclpy_context:
            self._rclpy.shutdown()
