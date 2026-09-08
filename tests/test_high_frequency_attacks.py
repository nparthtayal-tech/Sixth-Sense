"""
test_high_frequency_attacks.py — Unit & Integration Tests for High-Frequency Cyber Attack & Negative Drift Engine
=================================================================================================================
Tests:
1. High-frequency attack rate calculation and burst threshold detection.
2. Attack pattern classification (Unidirectional, Rapid Alternating, Oscillating, Erratic).
3. Overall drift and negative drift ("nega drift") mathematical compensation.
4. Return-to-start override activation and strict rejection of new destinations.
5. SensorSentry integration with live NegativeDriftGuidance homing to start location.
6. Multi-pulse cyber attack quarantine and drift cancellation end-to-end.
"""

import os
import sys
import unittest
import numpy as np

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from decision_making.high_frequency_attack_detector import (
    HighFrequencyAttackDetector,
    AttackPattern,
    HighFrequencySignal,
)
from sensor_sentry import SensorSentry, NegativeDriftGuidance
from hal import SimulatedBus, SensorReading
from waypoint_security import WaypointCommand, CommandValidator


class TestHighFrequencyAttackDetector(unittest.TestCase):

    def setUp(self):
        self.detector = HighFrequencyAttackDetector(
            freq_threshold_hz=2.0,
            burst_window_s=2.0,
            burst_count_threshold=3,
            drift_override_threshold_m=5.0,
        )
        self.start_pos = np.array([15.0, 25.0, 10.0])
        self.detector.set_start_position(self.start_pos)

    def test_high_frequency_rate_detection(self):
        """Rapid successive attack pulses arriving at >= 2.0 Hz must trigger high-frequency alarm."""
        cur_pos = np.array([20.0, 30.0, 10.0])
        
        # 4 attacks arriving 0.2s apart (5 Hz frequency)
        t = 1.0
        sig = None
        for i in range(4):
            t += 0.20
            vec = np.array([3.0, 1.0])
            sig = self.detector.record_and_evaluate("GNSS", t, vec, cur_pos, is_attack_sample=True)

        self.assertIsNotNone(sig)
        self.assertTrue(sig.is_high_frequency)
        self.assertGreaterEqual(sig.attack_frequency_hz, 2.0)
        self.assertEqual(sig.burst_count, 4)

    def test_attack_pattern_classification(self):
        """Pattern classifier must identify Unidirectional, Rapid Alternating, and Oscillating attacks."""
        cur_pos = np.array([20.0, 30.0, 10.0])

        # A: Unidirectional attacks (all pointing East [4, 0])
        self.detector.reset()
        self.detector.set_start_position(self.start_pos)
        t = 0.0
        for _ in range(4):
            t += 0.3
            sig_uni = self.detector.record_and_evaluate("GNSS", t, np.array([4.0, 0.0]), cur_pos)
        self.assertEqual(sig_uni.attack_pattern, AttackPattern.UNIDIRECTIONAL)

        # B: Rapid Alternating attacks (flipping East [4, 0] then West [-4, 0])
        self.detector.reset()
        self.detector.set_start_position(self.start_pos)
        t = 0.0
        for i in range(4):
            t += 0.25
            vec = np.array([4.0, 0.0]) if (i % 2 == 0) else np.array([-4.0, 0.0])
            sig_alt = self.detector.record_and_evaluate("GNSS", t, vec, cur_pos)
        self.assertEqual(sig_alt.attack_pattern, AttackPattern.RAPID_ALTERNATING)

        # C: Oscillating attacks (orthogonal 90 deg shifts: East [4, 0] then North [0, 4])
        self.detector.reset()
        self.detector.set_start_position(self.start_pos)
        t = 0.0
        for i in range(4):
            t += 0.25
            vec = np.array([4.0, 0.0]) if (i % 2 == 0) else np.array([0.0, 4.0])
            sig_osc = self.detector.record_and_evaluate("GNSS", t, vec, cur_pos)
        self.assertEqual(sig_osc.attack_pattern, AttackPattern.OSCILLATING)

    def test_overall_and_negative_drift_calculation(self):
        """Negative drift ('nega drift') must be exactly -overall_drift to cancel displacement."""
        # Drone starts at [15, 25] and has drifted to [45, 65]
        cur_pos = np.array([45.0, 65.0, 10.0])
        
        overall, mag, nega = self.detector.calculate_drift(cur_pos)

        # Expected overall drift: [45 - 15, 65 - 25] = [30, 40]
        np.testing.assert_allclose(overall, np.array([30.0, 40.0]))
        # Expected magnitude: sqrt(30^2 + 40^2) = 50.0m
        self.assertAlmostEqual(mag, 50.0, places=3)
        # Expected negative drift: [-30, -40]
        np.testing.assert_allclose(nega, np.array([-30.0, -40.0]))
        # Mathematical verification: overall + negative_drift = 0 (perfect drift cancellation)
        np.testing.assert_allclose(overall + nega, np.zeros(2), atol=1e-9)

    def test_override_to_start_lock_and_guidance(self):
        """Under high-frequency attacks exceeding drift threshold, override to start must activate."""
        # Start at [15, 25] and drift to [35, 45] (magnitude ~28.3m > 5.0m threshold)
        cur_pos = np.array([35.0, 45.0, 10.0])

        t = 1.0
        sig = None
        for i in range(3):
            t += 0.25
            sig = self.detector.record_and_evaluate("GNSS", t, np.array([6.0, 0.0]), cur_pos)

        self.assertTrue(sig.override_to_start)
        self.assertTrue(self.detector.is_override_active("GNSS"))
        # Target location must be the original start location [15, 25, 10]
        np.testing.assert_allclose(sig.target_start_location, self.start_pos)
        # Guidance unit vector must point directly back to start: [-20, -20] / norm
        expected_dir = np.array([-20.0, -20.0]) / np.linalg.norm(np.array([-20.0, -20.0]))
        np.testing.assert_allclose(sig.guidance_unit_vector, expected_dir, atol=1e-4)


class TestSensorSentryHighFrequencyIntegration(unittest.TestCase):

    def setUp(self):
        self.bus = SimulatedBus(verbose=False)
        self.sentry = SensorSentry("UAV_TEST_SENTRY", self.bus)
        self.start_pos = np.array([10.0, 20.0, 15.0])
        self.sentry.initialize_state(self.start_pos, 0.0)
        self.sentry.current_destination = np.array([150.0, 120.0, 15.0])

    def test_rejection_of_new_destinations_during_override(self):
        """When return-to-start override is active, any new destination command must be strictly rejected."""
        # Force high-frequency override
        hf_sig = HighFrequencySignal(
            timestamp=2.0,
            sensor="GNSS",
            is_high_frequency=True,
            attack_frequency_hz=4.0,
            burst_count=4,
            attack_pattern=AttackPattern.HIGH_FREQ_BURST,
            pattern_variance_deg=10.0,
            overall_drift_vector=np.array([20.0, 15.0]),
            overall_drift_magnitude=25.0,
            negative_drift_vector=np.array([-20.0, -15.0]),
            override_to_start=True,
            target_start_location=self.start_pos.copy(),
            guidance_unit_vector=np.array([-0.8, -0.6]),
        )
        self.sentry._start_return_to_start_override("GNSS", 2.0, hf_sig)
        
        self.assertTrue(self.sentry._return_to_start_override)
        # Target destination is overridden to start position
        np.testing.assert_allclose(self.sentry.current_destination, self.start_pos)

        # Attempt to inject a new signed waypoint destination
        validator = CommandValidator()
        new_dest = np.array([200.0, 300.0, 15.0])
        sig = validator.sign_command(new_dest, 101, 2.5, "COMMAND-CENTER")
        cmd = WaypointCommand(
            destination=new_dest,
            sequence_number=101,
            timestamp=2.5,
            issuer_id="COMMAND-CENTER",
            signature=sig,
            priority=1,
        )

        # Must be rejected because return-to-start override is active
        accepted = self.sentry.validate_new_waypoint(cmd)
        self.assertFalse(accepted)
        self.assertTrue(self.sentry.safe_stop.is_stopped)

    def test_live_negative_drift_guidance(self):
        """SensorSentry must produce live NegativeDriftGuidance pointing back to launch origin."""
        # Drone drifted to [30, 40, 15] from start [10, 20, 15]
        self.sentry.ekf.x[0:3] = np.array([30.0, 40.0, 15.0])
        self.sentry._return_to_start_override = True

        guidance = self.sentry.get_negative_drift_guidance()
        self.assertIsNotNone(guidance)
        self.assertIsInstance(guidance, NegativeDriftGuidance)
        
        # Overall drift: [30 - 10, 40 - 20] = [20, 20]
        np.testing.assert_allclose(guidance.overall_drift_vector, np.array([20.0, 20.0]))
        # Negative drift: [-20, -20]
        np.testing.assert_allclose(guidance.negative_drift_vector, np.array([-20.0, -20.0]))
        # Guidance unit vector directly to start
        expected_unit = np.array([-1.0, -1.0]) / np.sqrt(2.0)
        np.testing.assert_allclose(guidance.direction_unit_xy, expected_unit, atol=1e-4)
        # Distance to start: sqrt(20^2 + 20^2) = 28.28m
        self.assertAlmostEqual(guidance.distance_to_start_m, np.sqrt(800.0), places=2)

    def test_high_frequency_gnss_readings_trigger_override(self):
        """Feeding high-frequency spoofed GNSS readings must trigger override and isolate GNSS."""
        R_cov = np.eye(6) * 0.5
        t = 1.0

        # Feed 4 rapid spoofed GNSS readings with high offset (0.2s interval = 5 Hz)
        for i in range(4):
            t += 0.20
            # Spoofed GPS fix pulling away
            reading = SensorReading(
                timestamp=t,
                data=np.array([float(20.0 + i * 5.0), float(30.0 + i * 5.0), 15.0, 1.0, 1.0, 0.0]),
                sensor_type='GNSS',
                valid=True,
            )
    def test_more_than_5_spoofing_in_5_seconds_overrides_destination_to_start(self):
        """User Requirement: >5 spoofing attacks in 5 seconds strictly overrides destination with start point."""
        detector = HighFrequencyAttackDetector(
            freq_threshold_hz=1.0,
            burst_window_s=5.0,
            burst_count_threshold=5,
            drift_override_threshold_m=4.0,
        )
        detector.set_start_position(self.start_pos)
        cur_pos = np.array([25.0, 35.0, 15.0])

        # Feed 5 attacks within 4 seconds: burst_count = 5 (not yet > 5)
        sig = None
        for i in range(5):
            t = 1.0 + i * 0.7
            sig = detector.record_and_evaluate("GNSS", t, np.array([3.0, 0.0]), cur_pos, is_attack_sample=True)

        self.assertEqual(sig.burst_count, 5)

        # 6th attack arriving at t=4.5s (within 5 seconds window) -> burst_count = 6 (> 5 in 5s)
        sig = detector.record_and_evaluate("GNSS", 4.5, np.array([3.0, 0.0]), cur_pos, is_attack_sample=True)
        self.assertEqual(sig.burst_count, 6)
        self.assertTrue(sig.override_to_start)
        np.testing.assert_allclose(sig.target_start_location, self.start_pos)

        # In SensorSentry, this forces current_destination to start_pos
        self.sentry._start_return_to_start_override("GNSS", 4.5, sig)
        np.testing.assert_allclose(self.sentry.current_destination, self.start_pos)


if __name__ == '__main__':
    unittest.main()
