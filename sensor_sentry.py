"""
sensor_sentry.py — Top-Level Orchestrator
===========================================
Ties together all SensorSentry subsystems into a single real-time loop.

Handles:
1. Routing sensor readings into the EKF.
2. Checking fast (Chi-Square) and slow (CUSUM) monitors.
3. Managing SAARM filter bank quarantine and dead-reckoning.
4. Validating incoming waypoint commands (Scenario B defense).
5. Triggering Safe Stop and Fleet Alerts when attacks are confirmed.
"""

import time
import numpy as np
from dataclasses import dataclass
from typing import Optional, List, Tuple

import sys
import os
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, BASE_DIR)
sys.path.insert(0, os.path.join(BASE_DIR, 'filterpy-master'))

from ekf_fusion import MultiSensorEKF, SensorType
from ekf_fusion.sensor_models import (
    gnss_h, gnss_H, imu_h, imu_H, wheel_encoder_h, wheel_encoder_H,
    ultrasonic_h, ultrasonic_H, camera_h, camera_H, camera_residual,
    lidar_h, lidar_H, lidar_residual, five_g_h, five_g_H,
    atomic_clock_h, atomic_clock_H
)
from decision_making import (
    ChiSquareGate, CusumMonitor, LSTMAutoEncoderMonitor, MLStatus,
    DirectionalSpoofDetector, DirectionalStatus, DirectionalSpoofSignal,
    HighFrequencyAttackDetector, AttackPattern, HighFrequencySignal,
)
from saarm import SaarmFilterBank, SaarmVerdict
from waypoint_security import (
    CommandValidator, WaypointCommand, CommandVerdict,
    GeofenceValidator, GeofenceVerdict,
    EnvironmentMatcher, EnvironmentVerdict, DetectedFeature
)
from response import SafeStopController, SafeStopState, FleetAlertBroadcaster
from hal import HardwareBus, SensorReading


@dataclass(frozen=True)
class DirectionalRecoveryGuidance:
    """Trusted navigation command produced after directional GNSS spoofing.

    An autopilot adapter should use ``direction_unit_xy`` to steer from the
    independent navigation estimate toward the already-authorized mission
    target.  It must never derive this target from a newly received GNSS fix.
    """
    timestamp: float
    source_sensor: str
    trusted_position: np.ndarray
    target_position: np.ndarray
    direction_unit_xy: np.ndarray
    distance_to_target_m: float
    attack_heading_deg: float


@dataclass(frozen=True)
class NegativeDriftGuidance:
    """Guidance command calculated to cancel continuous cyber attack drift and return to start location.

    When high-frequency attacks occur, this guidance provides the negative drift
    vector and unit heading vector directly back to the verified launch origin.
    It strictly forbids proceeding toward newly injected destinations.
    """
    timestamp: float
    source_sensor: str
    trusted_position: np.ndarray
    start_position: np.ndarray
    overall_drift_vector: np.ndarray
    overall_drift_magnitude_m: float
    negative_drift_vector: np.ndarray
    direction_unit_xy: np.ndarray
    distance_to_start_m: float
    attack_frequency_hz: float
    attack_pattern: AttackPattern
    override_active: bool


class SensorSentry:
    """
    The main brain of the SensorSentry defense system.
    """
    def __init__(self, vehicle_id: str, hardware_bus: HardwareBus):
        self.vehicle_id = vehicle_id
        self.bus = hardware_bus
        
        # 1. Core State
        self.ekf = MultiSensorEKF()
        self.last_update_time = 0.0
        self.latest_omega = 0.0
        self.latest_accel = 0.0
        
        # 2. Monitors & SAARM (Scenario A Defense: Fast, Slow, ML, and Directional Vector)
        self.chi_square = ChiSquareGate(confidence=0.999)
        self.cusum = CusumMonitor(alarm_threshold=10.0, warning_threshold=5.0, slack=0.5)
        self.ml_monitor = LSTMAutoEncoderMonitor()
        self.directional_detector = DirectionalSpoofDetector(
            window_size=5,
            cumulative_threshold=15.0,
            coherence_threshold=0.80
        )
        self.saarm = SaarmFilterBank(isolation_threshold=2.0, min_alarm_count=3)
        
        # 3. Waypoint Security (Scenario B Defense)
        self.cmd_validator = CommandValidator()
        self.geofence = GeofenceValidator()
        self.env_matcher = EnvironmentMatcher()
        
        # 4. Response Layer
        self.safe_stop = SafeStopController(alert_to_stop_delay_s=0.0)
        self.fleet_alert = FleetAlertBroadcaster(vehicle_id=vehicle_id)
        
        self.current_destination: Optional[np.ndarray] = None
        self.start_position: Optional[np.ndarray] = None
        self.hf_detector = HighFrequencyAttackDetector(
            freq_threshold_hz=1.0,
            burst_window_s=5.0,
            burst_count_threshold=5,
            drift_override_threshold_m=4.0,
        )
        self._return_to_start_override: bool = False
        self._negative_drift_guidance: Optional[NegativeDriftGuidance] = None
        # Position fixed only by healthy non-GNSS measurements.  This becomes
        # the dead-reckoning origin if GNSS is later confirmed compromised.
        self._last_independent_position = self.ekf.x[0:3].copy()
        self._directional_recovery: Optional[DirectionalRecoveryGuidance] = None
        
    def initialize_state(self, initial_position: np.ndarray, initial_heading: float):
        """Set starting position before taking off/driving."""
        self.ekf.x[0:3] = initial_position
        self.ekf.x[6] = initial_heading
        self.last_update_time = time.time()
        self.start_position = np.asarray(initial_position, dtype=float).copy()
        self.hf_detector.set_start_position(self.start_position)
        self._last_independent_position = self.ekf.x[0:3].copy()
        self._directional_recovery = None
        self._return_to_start_override = False
        self._negative_drift_guidance = None
        self.safe_stop.update_good_position(initial_position)

    def get_negative_drift_guidance(self) -> Optional[NegativeDriftGuidance]:
        """Return live negative-drift compensation and homing guidance back to start location.

        Used when high-frequency continuous cyber attacks require an emergency override
        back to the initial launch point rather than proceeding to forward or new destinations.
        """
        if not self._return_to_start_override or self.start_position is None:
            return None

        trusted_position = self.get_current_position().copy()
        overall_drift, drift_mag, negative_drift = self.hf_detector.calculate_drift(trusted_position)
        
        delta_to_start = self.start_position[:2] - trusted_position[:2]
        dist_to_start = float(np.linalg.norm(delta_to_start))
        direction = delta_to_start / dist_to_start if dist_to_start > 1e-6 else np.zeros(2)
        
        sig = self.hf_detector.record_and_evaluate(
            "GNSS", self.last_update_time, np.zeros(2), trusted_position, is_attack_sample=False
        )

        return NegativeDriftGuidance(
            timestamp=self.last_update_time,
            source_sensor="GNSS",
            trusted_position=trusted_position,
            start_position=self.start_position.copy(),
            overall_drift_vector=overall_drift,
            overall_drift_magnitude_m=drift_mag,
            negative_drift_vector=negative_drift,
            direction_unit_xy=direction,
            distance_to_start_m=dist_to_start,
            attack_frequency_hz=sig.attack_frequency_hz,
            attack_pattern=sig.attack_pattern,
            override_active=True,
        )

    def _start_return_to_start_override(self, sensor: str, timestamp: float,
                                        hf_sig: HighFrequencySignal) -> None:
        """Quarantine sensor, lock override to return to start location, and forbid new destinations."""
        self._return_to_start_override = True
        
        # Override target to start location (no new destinations allowed)
        if self.start_position is not None:
            self.current_destination = self.start_position.copy()
            if self._directional_recovery is not None:
                self._directional_recovery.target_position = self.start_position.copy()

        # Isolate GNSS source
        trusted_state = self.ekf.x.copy()
        trusted_state[0:3] = self._last_independent_position
        saarm_sig = self.saarm.force_isolate(
            sensor, timestamp, trusted_state, self.ekf.x[6],
            np.linalg.norm(self.ekf.x[3:5]),
        )
        self.bus.send(saarm_sig.to_bytes(), 'SAARM')

        # Broadcast emergency high-frequency spoof alert
        alert = self.fleet_alert.create_gps_spoof_alert(
            timestamp=timestamp,
            position=self._last_independent_position,
            heading=self.get_current_heading(),
            speed=np.linalg.norm(self.ekf.x[3:6]),
            cusum_value=self.cusum.alarm_threshold,
            residual=hf_sig.overall_drift_magnitude,
        )
        self.bus.send(alert.to_bytes(), 'ALERT')

    def get_directional_recovery_guidance(self) -> Optional[DirectionalRecoveryGuidance]:
        """Return live guidance to the authorised target during GNSS recovery.

        ``None`` means no directional GNSS attack is active, or no authenticated
        mission destination is known.  The guidance vector is recomputed from
        the dead-reckoning / independent estimate on every call.
        """
        if self._directional_recovery is None:
            return None

        trusted_position = self.get_current_position().copy()
        if self._return_to_start_override and self.start_position is not None:
            target_position = self.start_position.copy()
        else:
            target_position = self._directional_recovery.target_position.copy()
        delta_xy = target_position[:2] - trusted_position[:2]
        distance = float(np.linalg.norm(delta_xy))
        direction = delta_xy / distance if distance > 1e-6 else np.zeros(2)
        return DirectionalRecoveryGuidance(
            timestamp=self._directional_recovery.timestamp,
            source_sensor=self._directional_recovery.source_sensor,
            trusted_position=trusted_position,
            target_position=target_position,
            direction_unit_xy=direction,
            distance_to_target_m=distance,
            attack_heading_deg=self._directional_recovery.attack_heading_deg,
        )

    def _start_directional_recovery(self, sensor: str, timestamp: float,
                                    attack_heading_deg: float) -> None:
        """Quarantine a confirmed source and retain the signed mission target."""
        if self._directional_recovery is not None:
            return

        # Do not seed dead reckoning from a state that may already have been
        # pulled by GNSS.  Use the newest healthy non-GNSS estimate instead.
        trusted_state = self.ekf.x.copy()
        trusted_state[0:3] = self._last_independent_position
        saarm_sig = self.saarm.force_isolate(
            sensor, timestamp, trusted_state, self.ekf.x[6],
            np.linalg.norm(self.ekf.x[3:5]),
        )
        self.bus.send(saarm_sig.to_bytes(), 'SAARM')

        alert = self.fleet_alert.create_gps_spoof_alert(
            timestamp=timestamp,
            position=self._last_independent_position,
            heading=self.get_current_heading(),
            speed=np.linalg.norm(self.ekf.x[3:6]),
            cusum_value=self.directional_detector.cumulative_threshold,
            residual=self.directional_detector.cumulative_threshold,
        )
        self.bus.send(alert.to_bytes(), 'ALERT')

        # A safe redirect is possible only toward a destination that passed
        # command authentication before the attack.  Without one, keeping the
        # aircraft in a hold/loiter mode is safer than inventing a target.
        if self.current_destination is None:
            return

        target = np.asarray(self.current_destination, dtype=float).copy()
        trusted_position = self.get_current_position().copy()
        delta_xy = target[:2] - trusted_position[:2]
        distance = float(np.linalg.norm(delta_xy))
        direction = delta_xy / distance if distance > 1e-6 else np.zeros(2)
        self._directional_recovery = DirectionalRecoveryGuidance(
            timestamp=timestamp,
            source_sensor=sensor,
            trusted_position=trusted_position,
            target_position=target,
            direction_unit_xy=direction,
            distance_to_target_m=distance,
            attack_heading_deg=attack_heading_deg,
        )
        
    def _get_sensor_dispatch(self, sensor_type: str) -> tuple:
        """Map string sensor type to EKF model functions."""
        # A simple mapping for orchestrator use
        mapping = {
            'GNSS': (gnss_H, gnss_h, None),
            'IMU': (imu_H, imu_h, None),
            'WHEEL_ENCODER': (wheel_encoder_H, wheel_encoder_h, None),
            'ULTRASONIC': (ultrasonic_H, ultrasonic_h, None),
            'CAMERA': (camera_H, camera_h, camera_residual),
            'LIDAR': (lidar_H, lidar_h, lidar_residual),
            'FIVE_G': (five_g_H, five_g_h, None),
            'ATOMIC_CLOCK': (atomic_clock_H, atomic_clock_h, None),
        }
        return mapping.get(sensor_type, (None, None, None))
        
    def process_sensor_reading(self, reading: SensorReading, R_cov: np.ndarray, *args):
        """
        Process a single incoming sensor measurement through the defense pipeline.
        Returns False if the vehicle is in a Safe Stop state.
        """
        if self.safe_stop.is_stopped:
            return False
            
        t = reading.timestamp
        dt = t - self.last_update_time
        if dt > 1e-6:
            self.ekf.predict(dt, omega_z=self.latest_omega, a_fwd=self.latest_accel)
            # If GPS is isolated, dead-reckoning navigator also steps forward
            if self.saarm.dr_navigator.active:
                vz = self.ekf.x[5]
                self.saarm.dr_navigator.update(dt, omega_z=self.latest_omega, vz=vz)
            self.last_update_time = t
            
            # Update last known good position if not under attack
            if self.safe_stop.state == SafeStopState.NOMINAL and not self.saarm.dr_navigator.active:
                self.safe_stop.update_good_position(self.ekf.x[0:3])

        st_name = reading.sensor_type
        if st_name == 'IMU':
            self.latest_omega = reading.data[0]
            self.latest_accel = reading.data[1]
            
        if self.saarm.is_quarantined(st_name):
            # Ignore data from quarantined sensors
            return True
            
        H_func, h_func, res_func = self._get_sensor_dispatch(st_name)
        if H_func is None:
            return True
            
        # 1. Compute Innovation
        z = np.atleast_1d(reading.data)
        y, S, H, PHT = self.ekf.compute_innovation(z, H_func, h_func, R_cov, args, res_func)
        m = len(z)

        # Directional evidence must see every GNSS innovation, including a
        # large fix that the instantaneous gate will reject below.  Otherwise
        # a fast spoof can continually bypass the sequential detector.
        dir_sig = self.directional_detector.evaluate(st_name, t, y)
        if dir_sig.status == DirectionalStatus.DIRECTIONAL_ATTACK:
            if st_name == 'GNSS':
                self._start_directional_recovery(st_name, t, dir_sig.attack_heading_deg)
            else:
                self.saarm.force_isolate(
                    st_name, t, self.ekf.x, self.ekf.x[6],
                    np.linalg.norm(self.ekf.x[3:5]),
                )

        # Evaluate High-Frequency Cyber Attack & Negative Drift Detector
        cur_pos = self.get_current_position()
        y_vec = np.atleast_1d(y)
        pos_dim = min(2, len(y_vec))
        y_mag = float(np.linalg.norm(y_vec[:pos_dim])) if pos_dim > 0 else 0.0
        is_attack_candidate = (
            dir_sig.status != DirectionalStatus.HEALTHY
            or y_mag > 2.0
        )
        hf_sig = self.hf_detector.record_and_evaluate(
            st_name, t, y, cur_pos, is_attack_sample=is_attack_candidate
        )
        if hf_sig.override_to_start:
            if st_name == 'GNSS':
                self._start_return_to_start_override(st_name, t, hf_sig)
            else:
                self.saarm.force_isolate(
                    st_name, t, self.ekf.x, self.ekf.x[6],
                    np.linalg.norm(self.ekf.x[3:5]),
                )

        if dir_sig.status == DirectionalStatus.DIRECTIONAL_ATTACK or hf_sig.override_to_start:
            return True  # Never fuse an innovation from a confirmed attack.
        
        # 2. Fast Monitor (Spike rejection)
        fast_sig = self.chi_square.evaluate(st_name, t, y, S, m)
        if fast_sig.verdict.value == 1:
            return True # Reject spike, don't fuse
            
        # 3. Slow Monitor (Drift detection)
        slow_sig = self.cusum.evaluate(st_name, t, y, S, m)
        
        # 3b. ML Monitor (Temporal sequence anomaly detection via LSTM-AE)
        ml_sig = self.ml_monitor.evaluate(st_name, t, y, S, m)
        
        # NIS is recorded for SAARM background checks
        try:
            SI = np.linalg.inv(S)
            nis = float(np.dot(y.T, np.dot(SI, y)))
        except:
            nis = float(np.sum(y**2))
        self.saarm.record_residual(st_name, nis)
        
        # If suspicious, sanitize innovation vector (ignore the drift along attack direction)
        if dir_sig.status == DirectionalStatus.SUSPICIOUS:
            # The monitor evaluates horizontal position only, whereas an EKF
            # GNSS innovation also contains altitude and velocity terms.
            # Preserve those dimensions so the Kalman update remains valid.
            y = y.copy()
            y[:len(dir_sig.sanitized_vector)] = dir_sig.sanitized_vector

        # 4. SAARM Isolation Check (triggered by CUSUM alarm OR ML alarm)
        if slow_sig.status.value >= 2 or ml_sig.status == MLStatus.ALARM:
            saarm_sig = self.saarm.process_alarm(
                st_name, t, slow_sig.cusum_value,
                self.ekf.x, self.ekf.x[6], np.linalg.norm(self.ekf.x[3:5])
            )
            self.bus.send(saarm_sig.to_bytes(), 'SAARM')
            
            if saarm_sig.verdict == SaarmVerdict.ISOLATED or saarm_sig.verdict == SaarmVerdict.DEAD_RECKONING:
                # Sensor confirmed compromised (e.g. GPS Spoofing)
                alert = self.fleet_alert.create_gps_spoof_alert(
                    timestamp=t,
                    position=self.get_current_position(),
                    heading=self.get_current_heading(),
                    speed=np.linalg.norm(self.ekf.x[3:6]),
                    cusum_value=slow_sig.cusum_value,
                    residual=saarm_sig.residual_full
                )
                self.bus.send(alert.to_bytes(), 'ALERT')
                # Do not apply update from this compromised sensor
                return True 
                
        # 5. Fuse healthy data (both slow monitor and ML monitor accept)
        if slow_sig.accepted and ml_sig.accepted:
            self.ekf.apply_update(y, S, H, PHT, R_cov)
            if st_name != 'GNSS':
                self._last_independent_position = self.ekf.x[0:3].copy()
            
        return True

    def validate_new_waypoint(self, command: WaypointCommand) -> bool:
        """
        Process a new destination command through the Waypoint Security module (Scenario B).
        Returns True if accepted, False if rejected (and triggers Safe Stop).
        """
        t = command.timestamp
        cur_pos = self.get_current_position()

        # -1. If return-to-start override is active under high-frequency attack, strictly reject new destinations
        if self._return_to_start_override or self.hf_detector.is_override_active():
            stop_sig = self.safe_stop.trigger_safe_stop(
                t, "HIGH_FREQ_CYBER_ATTACK", "New destination rejected: Return-to-start override active under cyber attack"
            )
            self.bus.send(stop_sig.to_bytes(), 'BRAKE')
            return False

        # 0. Check if destination shift aligns with active directional spoof vector
        target_shift = command.destination[:2] - cur_pos[:2]
        if self.directional_detector.is_direction_aligned('GNSS', target_shift):
            stop_sig = self.safe_stop.trigger_safe_stop(
                t, "DIRECTIONAL_SPOOF", "Directional Spoof Destination Hijack Detected"
            )
            self.bus.send(stop_sig.to_bytes(), 'BRAKE')
            return False
        
        # 1. Cryptographic Command Validator
        cmd_sig = self.cmd_validator.validate(command, cur_pos, t)
        self.bus.send(cmd_sig.to_bytes(), 'COMMAND')
        
        if cmd_sig.verdict != CommandVerdict.ACCEPTED:
            self._trigger_hijack_response(t, command.destination, cmd_sig.reason)
            return False
            
        # 2. Geofence Validator
        geo_sig = self.geofence.validate(command.destination, t)
        self.bus.send(geo_sig.to_bytes(), 'GEOFENCE')
        
        if geo_sig.verdict != GeofenceVerdict.AUTHORIZED:
            self._trigger_hijack_response(t, command.destination, geo_sig.reason)
            return False
            
        # Accepted
        self.current_destination = command.destination
        return True
        
    def process_environment_features(self, features: List[DetectedFeature], timestamp: float):
        """Cross-validate physical environment. Call this continuously as vehicle drives."""
        if self.safe_stop.is_stopped or not self.env_matcher._active_route:
            return
            
        env_sig = self.env_matcher.evaluate(self.get_current_position(), features, timestamp)
        
        if self.env_matcher.is_alarm: # Consecutive mismatches
            self._trigger_hijack_response(timestamp, self.current_destination, env_sig.reason)

    def _trigger_hijack_response(self, timestamp: float, tampered_dest: np.ndarray, reason: str):
        """Execute Safe Stop and broadcast Fleet Alert for Waypoint Hijack."""
        # 1. Safe Stop
        stop_sig = self.safe_stop.trigger_safe_stop(timestamp, "WAYPOINT_HIJACK", reason)
        self.bus.send(stop_sig.to_bytes(), 'BRAKE')
        
        # 2. Fleet Alert
        alert = self.fleet_alert.create_waypoint_hijack_alert(
            timestamp=timestamp,
            position=self.get_current_position(),
            heading=self.get_current_heading(),
            speed=np.linalg.norm(self.ekf.x[3:6]),
            original_dest=self.current_destination if self.current_destination is not None else np.zeros(3),
            tampered_dest=tampered_dest,
            rejection_reason=reason
        )
        self.bus.send(alert.to_bytes(), 'ALERT')

    def get_current_position(self) -> np.ndarray:
        if self.saarm.dr_navigator.active:
            return self.saarm.dr_navigator.position
        return self.ekf.x[0:3]
        
    def get_current_heading(self) -> float:
        if self.saarm.dr_navigator.active:
            return self.saarm.dr_navigator.heading
        return self.ekf.x[6]
