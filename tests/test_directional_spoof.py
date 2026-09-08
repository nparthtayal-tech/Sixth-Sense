"""
test_directional_spoof.py — Unit & Integration Tests for Directional Spoof Vector Monitor
========================================================================================
Tests:
1. Zero-mean noise cancellation (vectors destructively interfere, low coherence, healthy).
2. Unidirectional spoof accumulation (consecutive vectors constructively add up, high coherence, alarm).
3. Location distrust flag activation upon reaching cumulative threshold.
4. Vector ignoring via orthogonal subspace projection (removes attack vector drift).
5. Destination alignment check (catches waypoint commands pulling along attack direction).
6. SensorSentry orchestrator integration.
"""

import unittest
import numpy as np

from decision_making.directional_spoof_detector import (
    DirectionalSpoofDetector,
    DirectionalStatus,
    DirectionalSpoofSignal,
)
from sensor_sentry import SensorSentry
from hal import SimulatedBus, SensorReading


class TestDirectionalSpoofDetector(unittest.TestCase):

    def setUp(self):
        self.detector = DirectionalSpoofDetector(
            window_size=5,
            cumulative_threshold=15.0,
            coherence_threshold=0.80,
            min_vector_norm=0.2,
        )

    def test_zero_mean_noise_cancellation(self):
        """Random zero-mean noise vectors must cancel each other out and remain HEALTHY."""
        # 5 vectors pointing in opposing directions around origin
        noise_vectors = [
            np.array([1.2, 0.8]),
            np.array([-1.1, -0.9]),
            np.array([0.9, -1.0]),
            np.array([-0.8, 1.1]),
            np.array([0.1, -0.2]),
        ]

        t = 0.0
        sig = None
        for v in noise_vectors:
            t += 0.2
            sig = self.detector.evaluate('GNSS', t, v)

        self.assertIsNotNone(sig)
        self.assertEqual(sig.status, DirectionalStatus.HEALTHY)
        self.assertFalse(sig.stop_trusting_location)
        # Vector sum should be small due to cancellation
        self.assertLess(sig.cumulative_magnitude, 5.0)
        # Coherence should be low (vectors do not point in same direction)
        self.assertLess(sig.coherence, 0.60)

    def test_unidirectional_spoof_accumulation(self):
        """5 consecutive vectors in the same direction must constructively add up and trigger ALARM."""
        # 5 spoof vectors consistently pulling East-Northeast
        spoof_vectors = [
            np.array([3.8, 0.9]),
            np.array([4.1, 1.0]),
            np.array([3.9, 0.8]),
            np.array([4.2, 1.1]),
            np.array([4.0, 0.9]),
        ]

        t = 10.0
        sig = None
        for i, v in enumerate(spoof_vectors):
            t += 0.2
            sig = self.detector.evaluate('GNSS', t, v)

        self.assertIsNotNone(sig)
        self.assertEqual(sig.window_count, 5)
        # Vector sum magnitude should be ~20m (exceeding 15m threshold)
        self.assertGreaterEqual(sig.cumulative_magnitude, 15.0)
        # Coherence should be very high (> 0.95 because all point ENE)
        self.assertGreater(sig.coherence, 0.95)
        # Status must transition to DIRECTIONAL_ATTACK
        self.assertEqual(sig.status, DirectionalStatus.DIRECTIONAL_ATTACK)
        # Must flag to stop trusting the final location
        self.assertTrue(sig.stop_trusting_location)

    def test_stop_trusting_final_location_threshold(self):
        """stop_trusting_location must only trigger when both cumulative magnitude and coherence cross limits."""
        # Below threshold vectors (small magnitude)
        small_vectors = [
            np.array([1.0, 0.0]),
            np.array([1.0, 0.0]),
            np.array([1.0, 0.0]),
        ]
        t = 0.0
        for v in small_vectors:
            t += 0.1
            sig = self.detector.evaluate('GNSS', t, v)
        # Coherence is 1.0, but cumulative magnitude is only 3.0m < 15.0m
        self.assertFalse(sig.stop_trusting_location)
        self.assertNotEqual(sig.status, DirectionalStatus.DIRECTIONAL_ATTACK)

        # Now add large vectors pushing cumulative magnitude past 15m
        large_vectors = [
            np.array([6.0, 0.0]),
            np.array([7.0, 0.0]),
        ]
        for v in large_vectors:
            t += 0.1
            sig = self.detector.evaluate('GNSS', t, v)

        self.assertTrue(sig.stop_trusting_location)
        self.assertEqual(sig.status, DirectionalStatus.DIRECTIONAL_ATTACK)

    def test_vector_ignoring_orthogonal_projection(self):
        """Orthogonal projection must mathematically remove 100% of the attack direction."""
        # Train detector on Eastward spoof attack: u_hat = [1.0, 0.0]
        east_vectors = [
            np.array([4.0, 0.0]),
            np.array([4.0, 0.0]),
            np.array([4.0, 0.0]),
            np.array([4.0, 0.0]),
            np.array([4.0, 0.0]),
        ]
        t = 0.0
        for v in east_vectors:
            t += 0.1
            sig = self.detector.evaluate('GNSS', t, v)

        self.assertEqual(sig.status, DirectionalStatus.DIRECTIONAL_ATTACK)

        # Incoming raw measurement has 10m East drift and 5m North true motion: [10.0, 5.0]
        incoming_y = np.array([10.0, 5.0])
        sanitized_y = self.detector.filter_vector('GNSS', incoming_y)

        # East component (attack direction) must be eliminated (≈ 0)
        self.assertAlmostEqual(sanitized_y[0], 0.0, places=4)
        # North component (orthogonal, legitimate motion) must be 100% preserved (5.0)
        self.assertAlmostEqual(sanitized_y[1], 5.0, places=4)

    def test_destination_alignment_check(self):
        """is_direction_aligned must flag waypoint shifts pointing along the attack vector."""
        # Set up attack vector pointing East: [1, 0]
        for _ in range(5):
            self.detector.evaluate('GNSS', 1.0, np.array([5.0, 0.0]))

        # Destination shift pointing East: should be aligned (compromised)
        east_shift = np.array([20.0, 2.0])
        self.assertTrue(self.detector.is_direction_aligned('GNSS', east_shift, angular_tolerance_deg=45.0))

        # Destination shift pointing North: orthogonal, should NOT be flagged
        north_shift = np.array([0.0, 20.0])
        self.assertFalse(self.detector.is_direction_aligned('GNSS', north_shift, angular_tolerance_deg=45.0))

        # Destination shift pointing West (opposite): should NOT be flagged
        west_shift = np.array([-20.0, 0.0])
        self.assertFalse(self.detector.is_direction_aligned('GNSS', west_shift, angular_tolerance_deg=45.0))

    def test_confirmed_attack_stays_latched_until_explicit_reset(self):
        """A changed spoof direction must not clear a previously confirmed attack."""
        for i in range(5):
            sig = self.detector.evaluate('GNSS', float(i), np.array([4.0, 0.0]))
        self.assertEqual(sig.status, DirectionalStatus.DIRECTIONAL_ATTACK)

        # The attacker changes to North.  GNSS is still compromised and must
        # remain blocked until a verified recovery explicitly resets it.
        for i in range(5, 10):
            sig = self.detector.evaluate('GNSS', float(i), np.array([0.0, 4.0]))
        self.assertEqual(sig.status, DirectionalStatus.DIRECTIONAL_ATTACK)
        self.assertTrue(sig.stop_trusting_location)
        self.assertTrue(self.detector.is_attack_latched('GNSS'))
        np.testing.assert_allclose(self.detector.attack_direction('GNSS'), np.array([1.0, 0.0]))

        self.detector.reset('GNSS')
        sig = self.detector.evaluate('GNSS', 10.0, np.array([0.0, 0.0]))
        self.assertEqual(sig.status, DirectionalStatus.HEALTHY)
        self.assertFalse(self.detector.is_attack_latched('GNSS'))

    def test_sensor_sentry_integration(self):
        """SensorSentry orchestrator must isolate GNSS when directional spoof threshold is breached."""
        bus = SimulatedBus(verbose=False)
        sentry = SensorSentry("TEST_UAV_01", bus)
        sentry.directional_detector.cumulative_threshold = 12.0
        sentry.initialize_state(np.array([0.0, 0.0, 10.0]), 0.0)
        # This represents the destination accepted before the spoof began.
        # A real caller sets it through validate_new_waypoint().
        sentry.current_destination = np.array([100.0, 0.0, 10.0])

        # Send 5 GNSS readings with strong directional bias in East direction
        R_dummy = np.eye(6) * 0.5
        t = 1.0
        for i in range(5):
            t += 0.2
            # 6D GNSS reading [px, py, pz, vx, vy, vz] with growing eastward offset
            reading = SensorReading(
                timestamp=t,
                data=np.array([float(4.0 * (i + 1)), 0.0, 10.0, 1.0, 0.0, 0.0]),
                sensor_type='GNSS',
                valid=True,
            )
            sentry.process_sensor_reading(reading, R_dummy)

        # Directional detector must have registered the attack
        self.assertTrue(sentry.directional_detector.is_attack_latched('GNSS'))
        self.assertTrue(sentry.saarm.is_quarantined('GNSS'))
        guidance = sentry.get_directional_recovery_guidance()
        self.assertIsNotNone(guidance)
        np.testing.assert_allclose(guidance.target_position, np.array([100.0, 0.0, 10.0]))
        np.testing.assert_allclose(guidance.direction_unit_xy, np.array([1.0, 0.0]))


if __name__ == '__main__':
    unittest.main()
