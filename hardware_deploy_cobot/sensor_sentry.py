"""Cobot safety supervisor for authenticated indoor-navigation telemetry."""

from __future__ import annotations

import time
from dataclasses import dataclass
from math import hypot
from typing import Dict, Iterable, Optional, Tuple

from cobot_control import CobotControl
from cobot_hal import HardwareBus
from cobot_response import FleetAlertBroadcaster, SafeStopController
from telemetry import SensorReading
from cobot_waypoint_security import MapBoundary, RouteCorridor


Point = Tuple[float, float]


@dataclass(frozen=True)
class CobotSafetyLimits:
    map_boundary: MapBoundary
    corridor_half_width_m: float
    max_linear_speed_m_s: float
    max_angular_speed_rad_s: float
    minimum_obstacle_distance_m: float
    max_localization_jump_m: float
    sensor_timeout_s: float


class CobotSentry:
    """Evaluates safety conditions and latches containment on a violation."""

    def __init__(
        self,
        robot_id: str,
        limits: CobotSafetyLimits,
        bus: HardwareBus,
        control: CobotControl,
        require_clearance: bool = True,
        supervisor_key: Optional[bytes] = None,
        persistence_path: Optional[Path | str] = None,
    ) -> None:
        self.robot_id = robot_id
        self.limits = limits
        self.bus = bus
        self.control = control
        self.safe_stop = SafeStopController(
            robot_id=robot_id,
            require_clearance=require_clearance,
            supervisor_key=supervisor_key,
            persistence_path=persistence_path,
        )
        self.alerts = FleetAlertBroadcaster(robot_id)
        self.corridor: Optional[RouteCorridor] = None
        self.last_sensor_at: Dict[str, float] = {}
        self.last_odometry: Optional[Point] = None
        self.last_localization: Optional[Point] = None

    def load_trusted_route(self, points: Iterable[Point]) -> None:
        """Load a route only from an already-authorised mission planner."""
        self.corridor = RouteCorridor(points, self.limits.corridor_half_width_m)

    def process_sensor_reading(self, reading: SensorReading) -> bool:
        """Process one validated reading. Returns false when containment occurs."""
        self.last_sensor_at[reading.sensor] = reading.timestamp
        values = reading.values
        if reading.sensor == "ODOMETRY":
            point = values["position_m"]
            self.last_odometry = point
            self.safe_stop.update_position(*point)
            if not self.limits.map_boundary.contains(point):
                return self._contain("GEOFENCE_BREACH", "odometry is outside the approved map boundary")
            if self.corridor is not None and not self.corridor.contains(point):
                distance = self.corridor.distance_to(point)
                return self._contain("CORRIDOR_BREACH", f"robot is {distance:.2f} m outside the route corridor")
            if values["linear_velocity_m_s"] > self.limits.max_linear_speed_m_s:
                return self._contain("SPEED_LIMIT", "linear velocity exceeds configured safety limit")
            if abs(values["angular_velocity_rad_s"]) > self.limits.max_angular_speed_rad_s:
                return self._contain("SPEED_LIMIT", "angular velocity exceeds configured safety limit")
        elif reading.sensor == "LOCALIZATION":
            point = values["position_m"]
            self.last_localization = point
            if self.last_odometry is not None:
                disagreement = hypot(point[0] - self.last_odometry[0], point[1] - self.last_odometry[1])
                if disagreement > self.limits.max_localization_jump_m:
                    return self._contain(
                        "LOCALIZATION_ATTACK",
                        f"localization disagreement is {disagreement:.2f} m",
                    )
        elif reading.sensor == "LIDAR":
            if values["min_range_m"] < self.limits.minimum_obstacle_distance_m:
                return self._contain(
                    "OBSTACLE",
                    f"nearest obstacle is {values['min_range_m']:.2f} m away",
                )
        elif reading.sensor == "ESTOP" and values["pressed"]:
            return self._contain("E_STOP", "physical e-stop telemetry is active", lockdown=True)
        return not self.safe_stop.is_stopped

    def check_health(self, now: Optional[float] = None) -> bool:
        """Stop if a robot that began moving loses odometry freshness."""
        now = time.time() if now is None else now
        last_odometry_at = self.last_sensor_at.get("ODOMETRY")
        if last_odometry_at is not None and now - last_odometry_at > self.limits.sensor_timeout_s:
            return self._contain("TELEMETRY_LOSS", "odometry telemetry timed out")
        return not self.safe_stop.is_stopped

    def _contain(self, category: str, reason: str, lockdown: bool = False) -> bool:
        already_stopped = self.safe_stop.is_stopped
        signal = self.safe_stop.stop(reason, lockdown=lockdown)
        if already_stopped:
            return False
        self.bus.send(signal.to_bytes(), "SAFE_STOP")
        action_accepted = self.control.request_safe_stop(reason)
        alert = self.alerts.create(
            category,
            reason,
            signal.position_m,
            evidence={"controller_request_accepted": str(action_accepted).lower()},
            severity="EMERGENCY" if lockdown else "CRITICAL",
        )
        self.bus.send(alert.to_bytes(), "FLEET_ALERT")
        return False
