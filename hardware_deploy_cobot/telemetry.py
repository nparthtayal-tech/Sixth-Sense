"""Authenticated cobot telemetry parsing and replay protection."""

from __future__ import annotations

import base64
import hashlib
import hmac
import json
import math
import time
from dataclasses import dataclass
from typing import Any, Dict, Mapping, Optional, Tuple


class TelemetryError(ValueError):
    """A packet is malformed, unauthenticated, stale, or replayed."""


@dataclass(frozen=True)
class SensorReading:
    timestamp: float
    sensor: str
    values: Mapping[str, Any]


@dataclass(frozen=True)
class ClearanceCommand:
    timestamp: float
    robot_id: str
    token: str
    allow_lockdown_clear: bool = False


def canonical_packet(packet: Mapping[str, Any]) -> bytes:
    """Stable packet representation used for the HMAC input."""
    unsigned = {key: value for key, value in packet.items() if key != "signature"}
    return json.dumps(unsigned, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")


def load_hmac_key(base64_key: str) -> bytes:
    try:
        key = base64.b64decode(base64_key, validate=True)
    except (ValueError, TypeError) as exc:
        raise TelemetryError("HMAC key must be valid base64") from exc
    if len(key) != 32:
        raise TelemetryError("HMAC key must decode to exactly 32 bytes")
    return key


class PacketAuthenticator:
    """Rejects changed, stale, cross-robot, and replayed telemetry using a sliding window."""

    def __init__(
        self,
        key: bytes,
        robot_id: str,
        max_age_s: float = 2.0,
        window_size: int = 128,
    ) -> None:
        if len(key) != 32:
            raise ValueError("HMAC key must be exactly 32 bytes")
        self.key = key
        self.robot_id = robot_id
        self.max_age_s = max_age_s
        self.window_size = max(16, int(window_size))
        self._max_sequence = -1
        self._seen_sequences: set[int] = set()
        self._current_session_id: Optional[str] = None

    def verify(self, packet: Mapping[str, Any]) -> None:
        if packet.get("schema_version") != 1:
            raise TelemetryError("unsupported schema version")
        if packet.get("robot_id") != self.robot_id:
            raise TelemetryError("unexpected robot_id")
        timestamp = _finite_number(packet.get("timestamp"), "timestamp")
        if abs(time.time() - timestamp) > self.max_age_s:
            raise TelemetryError("stale timestamp")

        sequence = packet.get("sequence")
        if not isinstance(sequence, int) or isinstance(sequence, bool) or sequence < 0:
            raise TelemetryError("sequence must be a non-negative integer")

        session_id = packet.get("session_id")
        if session_id is not None and not isinstance(session_id, str):
            raise TelemetryError("session_id must be a string")

        # Verify HMAC signature BEFORE mutating sequence tracking state to prevent desync DoS
        signature = packet.get("signature")
        if not isinstance(signature, str) or len(signature) != 64:
            raise TelemetryError("missing or malformed signature")
        expected = hmac.new(self.key, canonical_packet(packet), hashlib.sha256).hexdigest()
        if not hmac.compare_digest(signature, expected):
            raise TelemetryError("signature verification failed")

        # Session tracking & reboot recovery
        if session_id is not None:
            if self._current_session_id is None:
                self._current_session_id = session_id
            elif session_id != self._current_session_id:
                # Legitimate producer restart with new session ID
                self._current_session_id = session_id
                self._max_sequence = -1
                self._seen_sequences.clear()

        # Sliding replay window algorithm (RFC 4303 style)
        if self._max_sequence == -1:
            self._max_sequence = sequence
            self._seen_sequences.add(sequence)
        elif sequence > self._max_sequence:
            self._max_sequence = sequence
            self._seen_sequences.add(sequence)
            # Evict sequences that have fallen outside the sliding window
            cutoff = self._max_sequence - self.window_size
            self._seen_sequences = {s for s in self._seen_sequences if s > cutoff}
        elif sequence > self._max_sequence - self.window_size:
            if sequence in self._seen_sequences:
                raise TelemetryError("replayed sequence")
            self._seen_sequences.add(sequence)
        else:
            raise TelemetryError("replayed or stale sequence outside replay window")


def parse_json(data: bytes) -> Dict[str, Any]:
    try:
        value = json.loads(data.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise TelemetryError("packet is not UTF-8 JSON") from exc
    if not isinstance(value, dict):
        raise TelemetryError("packet root must be an object")
    return value


def parse_clearance_packet(packet: Mapping[str, Any]) -> ClearanceCommand:
    if packet.get("action") != "CLEAR_SAFE_STOP":
        raise TelemetryError("packet is not a CLEAR_SAFE_STOP action")
    timestamp = _finite_number(packet.get("timestamp"), "timestamp")
    robot_id = str(packet.get("robot_id", ""))
    token = packet.get("token")
    if not isinstance(token, str) or not token:
        raise TelemetryError("clearance packet requires a valid non-empty token string")
    allow_lockdown = bool(packet.get("allow_lockdown_clear", False))
    return ClearanceCommand(
        timestamp=timestamp,
        robot_id=robot_id,
        token=token,
        allow_lockdown_clear=allow_lockdown,
    )


def parse_packet(packet: Mapping[str, Any]) -> SensorReading:
    sensor = packet.get("sensor")
    if sensor not in {"ODOMETRY", "LOCALIZATION", "IMU", "LIDAR", "BATTERY", "ESTOP"}:
        raise TelemetryError("unsupported sensor")
    timestamp = _finite_number(packet.get("timestamp"), "timestamp")
    values: Dict[str, Any]
    if sensor == "ODOMETRY":
        frame = packet.get("frame")
        if frame not in {"map", "odom"}:
            raise TelemetryError("ODOMETRY frame must be 'map' or 'odom'")
        values = {
            "position_m": _point(packet.get("position_m")),
            "yaw_rad": _finite_number(packet.get("yaw_rad"), "yaw_rad"),
            "linear_velocity_m_s": _non_negative(packet.get("linear_velocity_m_s"), "linear_velocity_m_s"),
            "angular_velocity_rad_s": _finite_number(packet.get("angular_velocity_rad_s"), "angular_velocity_rad_s"),
        }
    elif sensor == "LOCALIZATION":
        _require_frame(packet, "map")
        values = {
            "position_m": _point(packet.get("position_m")),
            "yaw_rad": _finite_number(packet.get("yaw_rad"), "yaw_rad"),
            "linear_velocity_m_s": _non_negative(packet.get("linear_velocity_m_s"), "linear_velocity_m_s"),
            "angular_velocity_rad_s": _finite_number(packet.get("angular_velocity_rad_s"), "angular_velocity_rad_s"),
        }
    elif sensor == "IMU":
        _require_frame(packet, "base_link")
        values = {
            "yaw_rate_rad_s": _finite_number(packet.get("yaw_rate_rad_s"), "yaw_rate_rad_s"),
            "forward_accel_m_s2": _finite_number(packet.get("forward_accel_m_s2"), "forward_accel_m_s2"),
        }
    elif sensor == "LIDAR":
        frame = packet.get("frame", "base_link")
        if frame != "base_link":
            raise TelemetryError("LIDAR frame must be base_link")
        values = {"min_range_m": _non_negative(packet.get("min_range_m"), "min_range_m")}
    elif sensor == "BATTERY":
        charge = _finite_number(packet.get("charge_percent"), "charge_percent")
        if not 0.0 <= charge <= 100.0:
            raise TelemetryError("charge_percent must be between 0 and 100")
        values = {"charge_percent": charge, "voltage_v": _positive(packet.get("voltage_v"), "voltage_v")}
    else:
        pressed = packet.get("pressed")
        if not isinstance(pressed, bool):
            raise TelemetryError("ESTOP pressed must be boolean")
        values = {"pressed": pressed}
    return SensorReading(timestamp, sensor, values)


def _require_frame(packet: Mapping[str, Any], expected: str) -> None:
    if packet.get("frame") != expected:
        raise TelemetryError(f"frame must be {expected}")


def _finite_number(value: Any, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise TelemetryError(f"{name} must be a finite number")
    return float(value)


def _non_negative(value: Any, name: str) -> float:
    result = _finite_number(value, name)
    if result < 0:
        raise TelemetryError(f"{name} must be non-negative")
    return result


def _positive(value: Any, name: str) -> float:
    result = _finite_number(value, name)
    if result <= 0:
        raise TelemetryError(f"{name} must be positive")
    return result


def _point(value: Any) -> Tuple[float, float]:
    if not isinstance(value, list) or len(value) != 2:
        raise TelemetryError("position_m must be a two-value array")
    return (_finite_number(value[0], "position_m[0]"), _finite_number(value[1], "position_m[1]"))
