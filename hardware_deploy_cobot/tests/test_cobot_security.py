"""Regression tests for authenticated cobot containment boundaries."""

from __future__ import annotations

import hashlib
import hmac
import json
import os
import sys
import tempfile
import time
import unittest
from pathlib import Path

PACKAGE_ROOT = Path(__file__).resolve().parents[1]
if str(PACKAGE_ROOT) not in sys.path:
    sys.path.insert(0, str(PACKAGE_ROOT))

from cobot_control import AdvisoryCobotControl
from cobot_hal import SimulatedBus
from cobot_response.safe_stop import SafeStopController, SafeStopState
from sensor_sentry import CobotSafetyLimits, CobotSentry
from telemetry import (PacketAuthenticator, TelemetryError, canonical_packet,
                       parse_packet)
from cobot_waypoint_security import CommandValidator, MapBoundary

KEY = b"0123456789abcdef0123456789abcdef"
ROBOT_ID = "COBOT-TEST-01"


def signed(packet: dict) -> dict:
    packet = dict(packet)
    packet["signature"] = hmac.new(KEY, canonical_packet(packet), hashlib.sha256).hexdigest()
    return packet


def sensor_packet(sensor: str, sequence: int, session_id: str | None = None, **values: object) -> dict:
    pkt: dict = {
        "schema_version": 1,
        "robot_id": ROBOT_ID,
        "sequence": sequence,
        "timestamp": time.time(),
        "sensor": sensor,
        **values,
    }
    if session_id is not None:
        pkt["session_id"] = session_id
    return signed(pkt)


def sentry() -> tuple[CobotSentry, SimulatedBus, AdvisoryCobotControl]:
    bus = SimulatedBus()
    control = AdvisoryCobotControl()
    limits = CobotSafetyLimits(
        map_boundary=MapBoundary(0.0, 0.0, 10.0, 10.0),
        corridor_half_width_m=0.5,
        max_linear_speed_m_s=1.0,
        max_angular_speed_rad_s=1.5,
        minimum_obstacle_distance_m=0.4,
        max_localization_jump_m=1.0,
        sensor_timeout_s=1.0,
    )
    return CobotSentry(ROBOT_ID, limits, bus, control), bus, control


class CobotSecurityTests(unittest.TestCase):
    def test_authentication_rejects_replay_and_tampering(self) -> None:
        authenticator = PacketAuthenticator(KEY, ROBOT_ID, max_age_s=5.0)
        packet = sensor_packet(
            "IMU", 1, frame="base_link", yaw_rate_rad_s=0.1, forward_accel_m_s2=0.0,
        )
        authenticator.verify(packet)
        with self.assertRaisesRegex(TelemetryError, "replayed"):
            authenticator.verify(packet)
        altered = dict(packet)
        altered["sequence"] = 2
        altered["yaw_rate_rad_s"] = 4.0
        with self.assertRaisesRegex(TelemetryError, "signature"):
            PacketAuthenticator(KEY, ROBOT_ID).verify(altered)

    def test_sliding_window_allows_out_of_order_and_rejects_duplicates(self) -> None:
        authenticator = PacketAuthenticator(KEY, ROBOT_ID, max_age_s=5.0, window_size=64)
        # Advance to sequence 10
        p10 = sensor_packet("IMU", 10, frame="base_link", yaw_rate_rad_s=0.0, forward_accel_m_s2=0.0)
        authenticator.verify(p10)

        # Out-of-order sequence 8 arrives within window
        p8 = sensor_packet("IMU", 8, frame="base_link", yaw_rate_rad_s=0.0, forward_accel_m_s2=0.0)
        authenticator.verify(p8)

        # Replayed sequence 8 must be rejected
        with self.assertRaisesRegex(TelemetryError, "replayed sequence"):
            authenticator.verify(p8)

        # Large jump beyond window size
        p200 = sensor_packet("IMU", 200, frame="base_link", yaw_rate_rad_s=0.0, forward_accel_m_s2=0.0)
        authenticator.verify(p200)

        # Sequence 8 is now outside the sliding window (200 - 64 = 136)
        with self.assertRaisesRegex(TelemetryError, "outside replay window"):
            authenticator.verify(p8)

    def test_session_id_enables_producer_reboot_recovery(self) -> None:
        authenticator = PacketAuthenticator(KEY, ROBOT_ID, max_age_s=5.0)
        # Producer runs with session A up to sequence 50
        p_session_a = sensor_packet(
            "IMU", 50, session_id="session_A", frame="base_link", yaw_rate_rad_s=0.0, forward_accel_m_s2=0.0
        )
        authenticator.verify(p_session_a)

        # Producer crashes and restarts with sequence 1 under new session B
        p_session_b = sensor_packet(
            "IMU", 1, session_id="session_B", frame="base_link", yaw_rate_rad_s=0.0, forward_accel_m_s2=0.0
        )
        # Should NOT raise "replayed sequence" or "outside window"; it should accept the new session
        authenticator.verify(p_session_b)

    def test_obstacle_latches_stop_and_only_requests_safe_containment(self) -> None:
        supervisor, bus, control = sentry()
        reading = parse_packet(sensor_packet("LIDAR", 1, frame="base_link", min_range_m=0.2))
        self.assertFalse(supervisor.process_sensor_reading(reading))
        self.assertTrue(supervisor.safe_stop.is_stopped)
        self.assertIsNotNone(control.last_request)
        self.assertEqual(len(bus.messages_for("SAFE_STOP")), 1)
        self.assertEqual(len(bus.messages_for("FLEET_ALERT")), 1)

    def test_localization_disagreement_stops_cobot(self) -> None:
        supervisor, _, _ = sentry()
        odom = parse_packet(sensor_packet(
            "ODOMETRY", 1, frame="map", position_m=[2.0, 2.0], yaw_rad=0.0,
            linear_velocity_m_s=0.2, angular_velocity_rad_s=0.0,
        ))
        localization = parse_packet(sensor_packet(
            "LOCALIZATION", 2, frame="map", position_m=[4.5, 2.0], yaw_rad=0.0,
            linear_velocity_m_s=0.2, angular_velocity_rad_s=0.0,
        ))
        self.assertTrue(supervisor.process_sensor_reading(odom))
        self.assertFalse(supervisor.process_sensor_reading(localization))
        self.assertTrue(supervisor.safe_stop.is_stopped)

    def test_goal_must_be_signed_and_inside_map(self) -> None:
        validator = CommandValidator(PacketAuthenticator(KEY, ROBOT_ID, max_age_s=5.0), "factory-floor-v1", MapBoundary(0, 0, 10, 10), 1.0)
        goal = signed({
            "schema_version": 1,
            "packet_type": "navigation_command",
            "robot_id": ROBOT_ID,
            "sequence": 1,
            "timestamp": time.time(),
            "map_id": "factory-floor-v1",
            "target_position_m": [8.0, 4.0],
            "target_yaw_rad": 0.0,
            "max_linear_speed_m_s": 0.5,
        })
        self.assertEqual(validator.validate(goal).target_position_m, (8.0, 4.0))

    def test_state_persistence_across_reboots(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            persist_file = Path(tmpdir) / "state.json"
            controller1 = SafeStopController(
                robot_id=ROBOT_ID, require_clearance=True, persistence_path=persist_file
            )
            self.assertFalse(controller1.is_stopped)
            controller1.update_position(3.5, 4.2)
            controller1.stop("Simulated breach", lockdown=True)
            self.assertTrue(controller1.is_stopped)
            self.assertEqual(controller1.state, SafeStopState.LOCKDOWN)

            # Re-initialize controller (simulating power cycle / process restart)
            controller2 = SafeStopController(
                robot_id=ROBOT_ID, require_clearance=True, persistence_path=persist_file
            )
            self.assertTrue(controller2.is_stopped, "Controller must resume in stopped state across reboot")
            self.assertEqual(controller2.state, SafeStopState.LOCKDOWN)
            self.assertEqual(controller2.status().position_m, (3.5, 4.2))

    def test_cryptographic_clearance_verification(self) -> None:
        supervisor_key = b"supervisorkey1234567890abcdef12"
        controller = SafeStopController(
            robot_id=ROBOT_ID,
            require_clearance=True,
            supervisor_key=supervisor_key,
        )
        controller.stop("Temporary obstacle", lockdown=False)
        self.assertTrue(controller.is_stopped)

        # Invalid token fails
        controller.clear("unauthorized-token")
        self.assertTrue(controller.is_stopped)

        # Valid direct token succeeds
        valid_token = hmac.new(supervisor_key, f"{ROBOT_ID}:{SafeStopState.STOPPED.value}".encode("utf-8"), hashlib.sha256).hexdigest()
        controller.clear(valid_token)
        self.assertFalse(controller.is_stopped)
        self.assertEqual(controller.state, SafeStopState.NOMINAL)

        # Lockdown requires allow_lockdown_clear=True
        controller.stop("Physical emergency switch", lockdown=True)
        self.assertEqual(controller.state, SafeStopState.LOCKDOWN)
        lockdown_token = hmac.new(supervisor_key, f"{ROBOT_ID}:{SafeStopState.LOCKDOWN.value}".encode("utf-8"), hashlib.sha256).hexdigest()

        # Regular clear fails on lockdown
        controller.clear(lockdown_token, allow_lockdown_clear=False)
        self.assertEqual(controller.state, SafeStopState.LOCKDOWN)

        # Privileged clear succeeds on lockdown
        controller.clear(lockdown_token, allow_lockdown_clear=True)
        self.assertEqual(controller.state, SafeStopState.NOMINAL)

    def test_chi_square_nis_rejects_sudden_localization_spike(self) -> None:
        supervisor, bus, control = sentry()
        # Initialize nominal odometry and state at [2.0, 2.0]
        odom1 = parse_packet(sensor_packet(
            "ODOMETRY", 1, frame="map", position_m=[2.0, 2.0], yaw_rad=0.0,
            linear_velocity_m_s=0.2, angular_velocity_rad_s=0.0,
        ))
        self.assertTrue(supervisor.process_sensor_reading(odom1))
        self.assertTrue(supervisor.ekf_initialized)
        self.assertAlmostEqual(supervisor.filtered_position[0], 2.0, delta=0.1)

        # Inject sudden localization spike (e.g. multipath or spoofing: jump to [7.0, 2.0])
        loc_spike = parse_packet(sensor_packet(
            "LOCALIZATION", 2, frame="map", position_m=[7.0, 2.0], yaw_rad=0.0,
            linear_velocity_m_s=0.2, angular_velocity_rad_s=0.0,
        ))
        result = supervisor.process_sensor_reading(loc_spike)
        self.assertFalse(result)
        self.assertTrue(supervisor.safe_stop.is_stopped)
        self.assertEqual(len(bus.messages_for("SAFE_STOP")), 1)

    def test_cusum_detects_slow_creeping_drift(self) -> None:
        supervisor, bus, control = sentry()
        supervisor.cusum.alarm_threshold = 3.0
        supervisor.cusum.warning_threshold = 1.5
        supervisor.cusum.slack = 0.1

        # Initialize at [1.0, 1.0]
        init_odom = parse_packet(sensor_packet(
            "ODOMETRY", 1, frame="map", position_m=[1.0, 1.0], yaw_rad=0.0,
            linear_velocity_m_s=0.1, angular_velocity_rad_s=0.0,
        ))
        supervisor.process_sensor_reading(init_odom)

        # Feed interleaved odometry (grounding robot at [1.0, 1.0]) and creeping localization [1.35, 1.0]
        stopped = False
        for seq in range(2, 25):
            t = time.time() + (seq * 0.1)
            odom_pkt = parse_packet(sensor_packet(
                "ODOMETRY", seq * 2, frame="map", position_m=[1.0, 1.0], yaw_rad=0.0,
                linear_velocity_m_s=0.0, angular_velocity_rad_s=0.0, timestamp=t,
            ))
            supervisor.process_sensor_reading(odom_pkt)

            drift_loc = parse_packet(sensor_packet(
                "LOCALIZATION", seq * 2 + 1, frame="map", position_m=[1.35, 1.0], yaw_rad=0.0,
                linear_velocity_m_s=0.0, angular_velocity_rad_s=0.0, timestamp=t + 0.05,
            ))
            if not supervisor.process_sensor_reading(drift_loc):
                stopped = True
                break

        self.assertTrue(stopped, "CUSUM cumulative drift detector must trigger containment on persistent offset")
        self.assertTrue(supervisor.safe_stop.is_stopped)

    def test_ekf_smooths_odometry_fusion(self) -> None:
        supervisor, _, _ = sentry()
        for seq in range(1, 10):
            t = time.time() + (seq * 0.1)
            base_x = 1.0 + (seq * 0.1)
            noisy_x = base_x + (0.02 if seq % 2 == 0 else -0.02)
            odom = parse_packet(sensor_packet(
                "ODOMETRY", seq, frame="map", position_m=[noisy_x, 1.0], yaw_rad=0.0,
                linear_velocity_m_s=1.0, angular_velocity_rad_s=0.0, timestamp=t,
            ))
            supervisor.process_sensor_reading(odom)

        filtered_x, filtered_y = supervisor.filtered_position
        self.assertAlmostEqual(filtered_y, 1.0, delta=0.05)
        self.assertAlmostEqual(filtered_x, 1.9, delta=0.2)


if __name__ == "__main__":
    unittest.main()
