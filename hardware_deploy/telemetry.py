"""Validated, replay-protected telemetry input for SensorSentry.

The EKF operates exclusively on two compact, documented measurements:

* IMU: ``[yaw_rate_rad_s, forward_accel_m_s2]`` in BODY_FRD.
* GNSS: ``[north, east, down, vn, ve, vd]`` in local NED metres/metres per
  second.

Network packets are JSON dictionaries.  In deployed UDP mode every packet is
HMAC-SHA256 authenticated and sequence checked before it reaches this module.
"""

from __future__ import annotations

import base64
import hashlib
import hmac
import json
import math
import time
from dataclasses import dataclass
from typing import Any, Dict, Mapping, MutableMapping, Optional, Sequence

import numpy as np

from hal import SensorReading


class TelemetryError(ValueError):
    """Raised when a packet is malformed, stale, unauthenticated, or unsafe."""


def _finite(value: Any, name: str) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise TelemetryError(f"{name} must be numeric") from exc
    if not math.isfinite(number):
        raise TelemetryError(f"{name} must be finite")
    return number


def _vector(value: Any, name: str, length: int) -> np.ndarray:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)) or len(value) != length:
        raise TelemetryError(f"{name} must contain exactly {length} numeric values")
    return np.asarray([_finite(item, f"{name}[{index}]") for index, item in enumerate(value)], dtype=float)


@dataclass(frozen=True)
class HomeLocation:
    """WGS-84 home point used to convert a GNSS fix into the EKF NED frame."""

    latitude_deg: float
    longitude_deg: float
    altitude_m: float

    def __post_init__(self) -> None:
        if not -90.0 <= self.latitude_deg <= 90.0:
            raise TelemetryError("home latitude must be between -90 and 90 degrees")
        if not -180.0 <= self.longitude_deg <= 180.0:
            raise TelemetryError("home longitude must be between -180 and 180 degrees")
        if not math.isfinite(self.altitude_m):
            raise TelemetryError("home altitude must be finite")

    def to_ned(self, latitude_deg: float, longitude_deg: float, altitude_m: float) -> np.ndarray:
        """Convert WGS-84 latitude/longitude/ellipsoid altitude to local NED.

        This uses an ECEF-to-NED rotation instead of a flat-earth approximation.
        The resulting frame remains local; operations must configure a new home
        point for every mission and must not use it as a global navigation frame.
        """
        point = _wgs84_to_ecef(latitude_deg, longitude_deg, altitude_m)
        home = _wgs84_to_ecef(self.latitude_deg, self.longitude_deg, self.altitude_m)
        lat = math.radians(self.latitude_deg)
        lon = math.radians(self.longitude_deg)
        sin_lat, cos_lat = math.sin(lat), math.cos(lat)
        sin_lon, cos_lon = math.sin(lon), math.cos(lon)
        rotation = np.array([
            [-sin_lat * cos_lon, -sin_lat * sin_lon, cos_lat],
            [-sin_lon, cos_lon, 0.0],
            [-cos_lat * cos_lon, -cos_lat * sin_lon, -sin_lat],
        ])
        return rotation @ (point - home)


def _wgs84_to_ecef(latitude_deg: float, longitude_deg: float, altitude_m: float) -> np.ndarray:
    lat = math.radians(_finite(latitude_deg, "latitude_deg"))
    lon = math.radians(_finite(longitude_deg, "longitude_deg"))
    alt = _finite(altitude_m, "altitude_m")
    if not -math.pi / 2 <= lat <= math.pi / 2 or not -math.pi <= lon <= math.pi:
        raise TelemetryError("GNSS latitude/longitude are out of range")
    semi_major = 6378137.0
    flattening = 1.0 / 298.257223563
    eccentricity_sq = flattening * (2.0 - flattening)
    normal_radius = semi_major / math.sqrt(1.0 - eccentricity_sq * math.sin(lat) ** 2)
    return np.array([
        (normal_radius + alt) * math.cos(lat) * math.cos(lon),
        (normal_radius + alt) * math.cos(lat) * math.sin(lon),
        (normal_radius * (1.0 - eccentricity_sq) + alt) * math.sin(lat),
    ])


def canonical_packet(packet: Mapping[str, Any]) -> bytes:
    """Canonicalize an unsigned packet so sender and receiver sign identical bytes."""
    unsigned = {key: value for key, value in packet.items() if key != "signature"}
    return json.dumps(unsigned, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode("utf-8")


class PacketAuthenticator:
    """HMAC, freshness, and monotonically increasing sequence enforcement."""

    def __init__(self, key: bytes, max_age_s: float = 2.0, max_future_s: float = 0.5):
        if len(key) < 32:
            raise TelemetryError("UDP HMAC key must contain at least 32 bytes")
        if max_age_s <= 0 or max_future_s < 0:
            raise TelemetryError("packet freshness limits are invalid")
        self._key = key
        self._max_age_s = max_age_s
        self._max_future_s = max_future_s
        self._last_sequence: MutableMapping[str, int] = {}

    @staticmethod
    def decode_key(value: str) -> bytes:
        """Accept a base64 key (preferred) or a raw 32+ character secret."""
        try:
            decoded = base64.b64decode(value, validate=True)
            if len(decoded) >= 32:
                return decoded
        except Exception:
            pass
        return value.encode("utf-8")

    def verify(self, packet: Mapping[str, Any], now: Optional[float] = None) -> None:
        signature = packet.get("signature")
        if not isinstance(signature, str):
            raise TelemetryError("packet signature is required")
        expected = hmac.new(self._key, canonical_packet(packet), hashlib.sha256).hexdigest()
        if not hmac.compare_digest(signature.lower(), expected):
            raise TelemetryError("packet signature is invalid")

        vehicle_id = packet.get("vehicle_id")
        if not isinstance(vehicle_id, str) or not vehicle_id:
            raise TelemetryError("vehicle_id is required")
        sequence = packet.get("sequence")
        if not isinstance(sequence, int) or isinstance(sequence, bool) or sequence < 0:
            raise TelemetryError("sequence must be a non-negative integer")
        timestamp = _finite(packet.get("timestamp"), "timestamp")
        current = time.time() if now is None else now
        if timestamp < current - self._max_age_s or timestamp > current + self._max_future_s:
            raise TelemetryError("packet timestamp is outside the accepted freshness window")
        previous = self._last_sequence.get(vehicle_id)
        if previous is not None and sequence <= previous:
            raise TelemetryError("replayed or out-of-order packet")
        self._last_sequence[vehicle_id] = sequence


def parse_packet(packet: Mapping[str, Any], home: HomeLocation, expected_vehicle_id: str) -> SensorReading:
    """Validate one schema-v1 packet and normalize it to an EKF measurement."""
    if packet.get("schema_version") != 1:
        raise TelemetryError("schema_version must be 1")
    if packet.get("vehicle_id") != expected_vehicle_id:
        raise TelemetryError("packet vehicle_id does not match this sentry instance")
    timestamp = _finite(packet.get("timestamp"), "timestamp")
    sensor = packet.get("sensor")
    if not isinstance(sensor, str):
        raise TelemetryError("sensor is required")
    sensor = sensor.upper()

    if sensor == "IMU":
        if packet.get("frame") != "BODY_FRD":
            raise TelemetryError("IMU frame must be BODY_FRD")
        data = np.array([
            _finite(packet.get("yaw_rate_rad_s"), "yaw_rate_rad_s"),
            _finite(packet.get("forward_accel_m_s2"), "forward_accel_m_s2"),
        ])
        return SensorReading(timestamp, data, "IMU", True, {"frame": "BODY_FRD"})

    if sensor != "GNSS":
        raise TelemetryError(f"unsupported sensor type: {sensor}")
    if packet.get("frame") != "NED":
        raise TelemetryError("GNSS frame must be NED")
    fix_type = packet.get("fix_type")
    if not isinstance(fix_type, int) or fix_type < 3:
        raise TelemetryError("GNSS requires a 3D fix (fix_type >= 3)")
    satellites = packet.get("satellites")
    if not isinstance(satellites, int) or satellites < 6:
        raise TelemetryError("GNSS requires at least six tracked satellites")
    velocity = _vector(packet.get("velocity_ned_m_s"), "velocity_ned_m_s", 3)
    if "position_ned_m" in packet:
        position = _vector(packet["position_ned_m"], "position_ned_m", 3)
    else:
        position = home.to_ned(
            _finite(packet.get("latitude_deg"), "latitude_deg"),
            _finite(packet.get("longitude_deg"), "longitude_deg"),
            _finite(packet.get("altitude_m"), "altitude_m"),
        )
    return SensorReading(
        timestamp,
        np.concatenate((position, velocity)),
        "GNSS",
        True,
        {"frame": "NED", "fix_type": fix_type, "satellites": satellites},
    )
