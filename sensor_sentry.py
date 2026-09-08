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
from decision_making import ChiSquareGate, CusumMonitor, LSTMAutoEncoderMonitor, MLStatus
from saarm import SaarmFilterBank, SaarmVerdict
from waypoint_security import (
    CommandValidator, WaypointCommand, CommandVerdict,
    GeofenceValidator, GeofenceVerdict,
    EnvironmentMatcher, EnvironmentVerdict, DetectedFeature
)
from response import SafeStopController, SafeStopState, FleetAlertBroadcaster
from hal import HardwareBus, SensorReading


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
        
        # 2. Monitors & SAARM (Scenario A Defense: Fast, Slow, and ML)
        self.chi_square = ChiSquareGate(confidence=0.999)
        self.cusum = CusumMonitor(alarm_threshold=10.0, warning_threshold=5.0, slack=0.5)
        self.ml_monitor = LSTMAutoEncoderMonitor()
        self.saarm = SaarmFilterBank(isolation_threshold=2.0, min_alarm_count=3)
        
        # 3. Waypoint Security (Scenario B Defense)
        self.cmd_validator = CommandValidator()
        self.geofence = GeofenceValidator()
        self.env_matcher = EnvironmentMatcher()
        
        # 4. Response Layer
        self.safe_stop = SafeStopController(alert_to_stop_delay_s=0.0)
        self.fleet_alert = FleetAlertBroadcaster(vehicle_id=vehicle_id)
        
        self.current_destination: Optional[np.ndarray] = None
        
    def initialize_state(self, initial_position: np.ndarray, initial_heading: float):
        """Set starting position before taking off/driving."""
        self.ekf.x[0:3] = initial_position
        self.ekf.x[6] = initial_heading
        self.last_update_time = time.time()
        self.safe_stop.update_good_position(initial_position)
        
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
            
        return True

    def validate_new_waypoint(self, command: WaypointCommand) -> bool:
        """
        Process a new destination command through the Waypoint Security module (Scenario B).
        Returns True if accepted, False if rejected (and triggers Safe Stop).
        """
        t = command.timestamp
        cur_pos = self.get_current_position()
        
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
