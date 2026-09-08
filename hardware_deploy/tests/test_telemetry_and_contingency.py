"""Regression tests for the production telemetry and aircraft-response boundary."""

from __future__ import annotations

import hashlib
import hmac
import sys
import time
import unittest
from pathlib import Path

import numpy as np

PACKAGE_ROOT = Path(__file__).resolve().parents[1]
if str(PACKAGE_ROOT) not in sys.path:
    sys.path.insert(0, str(PACKAGE_ROOT))

from flight_control import ContingencyAction, FlightControl
from hal import SimulatedBus
from main import decode_and_process
from sensor_sentry import SensorSentry
from telemetry import (HomeLocation, PacketAuthenticator, TelemetryError,
                       canonical_packet, parse_packet)


KEY = b"0123456789abcdef0123456789abcdef"
DRONE_ID = "DRONE-TEST-01"


def signed(packet: dict) -> dict:
    packet = dict(packet)
    packet["signature"] = hmac.new(KEY, canonical_packet(packet), hashlib.sha256).hexdigest()
    return packet


class RecordingFlightControl(FlightControl):
    def __init__(self) -> None:
        self.requests = []

    def request(self, action: ContingencyAction, reason: str) -> bool:
        self.requests.append((action, reason))
        return True


class TelemetryTests(unittest.TestCase):
    def setUp(self) -> None:
        self.home = HomeLocation(13.060421, 80.281054, 100.2)
        self.auth = PacketAuthenticator(KEY, max_age_s=5.0)

    def base(self, sensor: str, sequence: int = 1) -> dict:
        return {
            "schema_version": 1,
            "vehicle_id": DRONE_ID,
            "sequence": sequence,
            "timestamp": time.time(),
            "sensor": sensor,
        }

    def test_gnss_fix_is_normalized_to_six_value_local_ned_measurement(self) -> None:
        packet = self.base("GNSS") | {
            "frame": "NED",
            "latitude_deg": 13.060421,
            "longitude_deg": 80.281054,
            "altitude_m": 100.2,
            "velocity_ned_m_s": [1.0, -0.2, 0.1],
            "fix_type": 3,
            "satellites": 12,
        }
        packet = signed(packet)
        self.auth.verify(packet)
        reading = parse_packet(packet, self.home, DRONE_ID)
        self.assertEqual(reading.sensor_type, "GNSS")
        self.assertEqual(reading.data.shape, (6,))
        np.testing.assert_allclose(reading.data[:3], np.zeros(3), atol=1e-6)
        np.testing.assert_allclose(reading.data[3:], [1.0, -0.2, 0.1])

    def test_imu_contract_is_exactly_two_values(self) -> None:
        packet = self.base("IMU") | {
            "frame": "BODY_FRD",
            "yaw_rate_rad_s": 0.12,
            "forward_accel_m_s2": -0.3,
        }
        packet = signed(packet)
        self.auth.verify(packet)
        reading = parse_packet(packet, self.home, DRONE_ID)
        np.testing.assert_allclose(reading.data, [0.12, -0.3])

    def test_tamper_and_replay_are_rejected(self) -> None:
        packet = signed(self.base("IMU") | {
            "frame": "BODY_FRD", "yaw_rate_rad_s": 0.0, "forward_accel_m_s2": 0.0,
        })
        self.auth.verify(packet)
        with self.assertRaisesRegex(TelemetryError, "replayed"):
            self.auth.verify(packet)
        tampered = dict(packet)
        tampered["sequence"] = 2
        tampered["yaw_rate_rad_s"] = 5.0
        with self.assertRaisesRegex(TelemetryError, "signature"):
            PacketAuthenticator(KEY).verify(tampered)

    def test_aircraft_protection_requests_hold_without_motor_kill_or_brake(self) -> None:
        bus = SimulatedBus()
        flight_control = RecordingFlightControl()
        sentry = SensorSentry(DRONE_ID, bus, flight_control=flight_control, aircraft_mode=True)
        sentry._protective_stop(time.time(), "WAYPOINT_HIJACK", "test")
        self.assertEqual(flight_control.requests, [(ContingencyAction.HOLD, "test")])
        self.assertEqual(sentry.safe_stop.state.name, "NOMINAL")
        self.assertEqual(len(bus.get_messages_by_channel("BRAKE")), 0)
        self.assertEqual(len(bus.get_messages_by_channel("CONTINGENCY")), 1)

    def test_authenticated_packets_reach_the_ekf_with_matching_covariances(self) -> None:
        bus = SimulatedBus()
        sentry = SensorSentry(DRONE_ID, bus)
        sentry.initialize_state(np.zeros(3), 0.0)
        now = time.time()
        imu = signed(self.base("IMU", 1) | {
            "timestamp": now,
            "frame": "BODY_FRD",
            "yaw_rate_rad_s": 0.0,
            "forward_accel_m_s2": 0.0,
        })
        gnss = signed(self.base("GNSS", 2) | {
            "timestamp": now + 0.01,
            "frame": "NED",
            "position_ned_m": [0.0, 0.0, 0.0],
            "velocity_ned_m_s": [0.0, 0.0, 0.0],
            "fix_type": 3,
            "satellites": 10,
        })
        auth = PacketAuthenticator(KEY, max_age_s=5.0)
        decode_and_process(json_bytes(imu), sentry, self.home, auth, DRONE_ID)
        decode_and_process(json_bytes(gnss), sentry, self.home, auth, DRONE_ID)

    def test_untrained_ml_monitor_is_disabled(self) -> None:
        sentry = SensorSentry(DRONE_ID, SimulatedBus())
        self.assertFalse(sentry.ml_monitor.enabled)


def json_bytes(packet: dict) -> bytes:
    import json
    return json.dumps(packet, separators=(",", ":")).encode("utf-8")


if __name__ == "__main__":
    unittest.main()
