"""HMAC-authenticated navigation goal validation for a cobot mission planner."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Optional, Tuple

from telemetry import PacketAuthenticator, TelemetryError, _finite_number, _point

from .geofence_validator import MapBoundary, RouteCorridor


@dataclass(frozen=True)
class NavigationCommand:
    robot_id: str
    map_id: str
    sequence: int
    timestamp: float
    target_position_m: Tuple[float, float]
    target_yaw_rad: float
    max_linear_speed_m_s: float


class CommandValidator:
    """Validates a goal but never dispatches it to the navigation controller."""

    def __init__(
        self,
        authenticator: PacketAuthenticator,
        map_id: str,
        map_boundary: MapBoundary,
        max_linear_speed_m_s: float,
    ) -> None:
        self.authenticator = authenticator
        self.map_id = map_id
        self.map_boundary = map_boundary
        self.max_linear_speed_m_s = max_linear_speed_m_s

    def validate(self, packet: Mapping[str, object], corridor: Optional[RouteCorridor] = None) -> NavigationCommand:
        if packet.get("packet_type") != "navigation_command":
            raise TelemetryError("packet is not a navigation_command")
        self.authenticator.verify(packet)
        if packet.get("map_id") != self.map_id:
            raise TelemetryError("navigation command map_id does not match configured map")
        target = _point(packet.get("target_position_m"))
        if not self.map_boundary.contains(target):
            raise TelemetryError("navigation target is outside the approved map boundary")
        if corridor is not None and not corridor.contains(target):
            raise TelemetryError("navigation target is outside the active route corridor")
        requested_speed = _finite_number(packet.get("max_linear_speed_m_s"), "max_linear_speed_m_s")
        if not 0.0 < requested_speed <= self.max_linear_speed_m_s:
            raise TelemetryError("navigation target exceeds configured speed limit")
        return NavigationCommand(
            robot_id=str(packet["robot_id"]),
            map_id=self.map_id,
            sequence=int(packet["sequence"]),
            timestamp=float(packet["timestamp"]),
            target_position_m=target,
            target_yaw_rad=_finite_number(packet.get("target_yaw_rad"), "target_yaw_rad"),
            max_linear_speed_m_s=requested_speed,
        )
