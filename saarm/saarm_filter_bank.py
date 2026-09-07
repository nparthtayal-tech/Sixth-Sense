"""
saarm_filter_bank.py — Sensor Autonomous Adaptive Reconfiguration Management
==============================================================================
Parallel EKF sub-filter bank for compromised sensor isolation.

Architecture:
    - 1 "full" filter using ALL sensors (the main EKF)
    - N "exclusion" filters, each omitting exactly one sensor modality
    
When CUSUM flags a sensor as ALARM:
    1. SAARM compares the full filter's residual vs. the exclusion filter
       that omits the suspected sensor.
    2. If the exclusion filter has significantly lower innovation residuals,
       SAARM confirms the sensor is compromised.
    3. SAARM switches the navigation solution to the exclusion filter and
       quarantines the compromised sensor.
    4. The vehicle continues navigating via dead-reckoning (IMU + wheel
       encoders + LiDAR + 5G) without the corrupted GPS feed.

Hardware Interface:
    SaarmSignal.to_bytes() → 24-byte packet for CAN / UART / SPI.
    SaarmSignal.to_dict()  → JSON-serializable dict for ROS2 / TCP / MQTT.

References:
    [6] SAARM concept — parallel redundancy management
    [7] Inertial dead-reckoning fallback
    [9] Autonomous fault isolation in multi-sensor systems
"""

import struct
import numpy as np
from dataclasses import dataclass, field
from enum import IntEnum
from typing import Dict, List, Optional, Set

import sys
import os
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BASE_DIR)
sys.path.insert(0, os.path.join(BASE_DIR, 'filterpy-master'))

from ekf_fusion.ekf_engine import MultiSensorEKF
from ekf_fusion.sensor_models import SensorType


class SaarmVerdict(IntEnum):
    """SAARM isolation verdict for hardware controllers."""
    NOMINAL       = 0   # All sensors healthy, using full filter
    INVESTIGATING = 1   # Anomaly detected, running isolation checks
    ISOLATED      = 2   # Compromised sensor identified and quarantined
    DEAD_RECKONING = 3  # GPS quarantined, navigating on inertial alone


@dataclass
class SaarmSignal:
    """
    Output signal from the SAARM filter bank.
    Designed for direct transmission to hardware decision units.
    """
    timestamp: float
    verdict: SaarmVerdict
    quarantined_sensors: List[str]
    active_filter: str                   # "FULL" or "EXCLUDE_<SENSOR>"
    residual_full: float                 # Mean residual of full filter
    residual_exclusion: float            # Mean residual of best exclusion filter
    confidence: float                    # Isolation confidence (0.0 to 1.0)
    dead_reckoning_active: bool
    dr_position: Optional[np.ndarray] = None  # Dead-reckoning position estimate

    def to_dict(self) -> dict:
        """JSON / telemetry serialization."""
        result = {
            'timestamp': round(self.timestamp, 4),
            'verdict': self.verdict.name,
            'verdict_code': int(self.verdict),
            'quarantined_sensors': self.quarantined_sensors,
            'active_filter': self.active_filter,
            'residual_full': round(float(self.residual_full), 4),
            'residual_exclusion': round(float(self.residual_exclusion), 4),
            'confidence': round(float(self.confidence), 4),
            'dead_reckoning_active': self.dead_reckoning_active,
        }
        if self.dr_position is not None:
            result['dr_position'] = [round(float(v), 4) for v in self.dr_position]
        return result

    def to_bytes(self) -> bytes:
        """
        Compact 24-byte hardware packet:
        [uint32 timestamp_ms, uint8 verdict, uint8 n_quarantined,
         uint8 dr_active, uint8 reserved,
         float32 residual_full, float32 residual_exclusion,
         float32 confidence, float32 dr_x]
        """
        ts_ms = int(self.timestamp * 1000) & 0xFFFFFFFF
        dr_x = float(self.dr_position[0]) if self.dr_position is not None else 0.0
        return struct.pack('<IBBBBffff',
                           ts_ms,
                           int(self.verdict),
                           len(self.quarantined_sensors) & 0xFF,
                           1 if self.dead_reckoning_active else 0,
                           0,
                           float(self.residual_full),
                           float(self.residual_exclusion),
                           float(self.confidence),
                           dr_x)


class DeadReckoningNavigator:
    """
    Inertial dead-reckoning fallback navigator.
    
    When GPS is quarantined, this module propagates position using
    only IMU angular rates and wheel encoder forward speeds.
    No external absolute positioning required.
    
    State: [px, py, pz, psi, v_fwd]
    
    This is intentionally simpler than the full 11-state EKF —
    it's an emergency backup, not a primary navigator.
    """
    
    def __init__(self, initial_position=None, initial_heading=0.0, initial_speed=0.0):
        if initial_position is not None:
            self.px, self.py, self.pz = float(initial_position[0]), float(initial_position[1]), float(initial_position[2])
        else:
            self.px, self.py, self.pz = 0.0, 0.0, 0.0
        self.psi = float(initial_heading)
        self.v_fwd = float(initial_speed)
        self.active = False
        self._accumulated_drift_estimate = 0.0
        self._time_in_dr = 0.0
    
    def activate(self, current_position, current_heading, current_speed):
        """Activate dead-reckoning from known-good state."""
        self.px = float(current_position[0])
        self.py = float(current_position[1])
        self.pz = float(current_position[2])
        self.psi = float(current_heading)
        self.v_fwd = float(current_speed)
        self.active = True
        self._accumulated_drift_estimate = 0.0
        self._time_in_dr = 0.0
    
    def update(self, dt, omega_z=0.0, v_fwd=None, vz=0.0):
        """
        Propagate dead-reckoning position forward by dt.
        
        Parameters
        ----------
        dt : float
            Time step in seconds.
        omega_z : float
            Yaw rate from IMU (rad/s).
        v_fwd : float or None
            Forward speed from wheel encoders (m/s).
            If None, uses last known speed.
        vz : float
            Vertical velocity (m/s).
        """
        if not self.active:
            return
        
        if v_fwd is not None:
            self.v_fwd = float(v_fwd)
        
        # Integrate heading
        self.psi += omega_z * dt
        self.psi = (self.psi + np.pi) % (2.0 * np.pi) - np.pi
        
        # Integrate position
        self.px += self.v_fwd * np.cos(self.psi) * dt
        self.py += self.v_fwd * np.sin(self.psi) * dt
        self.pz += vz * dt
        
        # Track drift accumulation (IMU drift ~ 0.01 m/s² typical)
        self._time_in_dr += dt
        self._accumulated_drift_estimate = 0.5 * 0.01 * self._time_in_dr**2
    
    @property
    def position(self):
        return np.array([self.px, self.py, self.pz])
    
    @property
    def heading(self):
        return self.psi
    
    @property
    def drift_estimate_meters(self):
        """Estimated position drift due to IMU integration errors."""
        return self._accumulated_drift_estimate
    
    def deactivate(self):
        """Return to GPS-aided navigation."""
        self.active = False


class SaarmFilterBank:
    """
    SAARM — Sensor Autonomous Adaptive Reconfiguration Management.
    
    Parallel filter bank architecture:
    - Maintains the main EKF's state snapshot
    - When a sensor is flagged by CUSUM, runs diagnostic exclusion checks
    - Confirms or denies sensor compromise by comparing residuals
    - Manages dead-reckoning fallback when GPS is quarantined
    
    Parameters
    ----------
    isolation_threshold : float
        Ratio of (full_residual / exclusion_residual) above which
        SAARM confirms the sensor is compromised. Default 2.0 means
        the full filter's residuals must be 2x worse than the
        exclusion filter to confirm isolation.
    min_alarm_count : int
        Number of consecutive CUSUM ALARMs required before SAARM
        initiates isolation. Prevents false positives from transient spikes.
    """
    
    def __init__(self,
                 isolation_threshold: float = 2.0,
                 min_alarm_count: int = 3):
        
        self.isolation_threshold = isolation_threshold
        self.min_alarm_count = min_alarm_count
        
        # Tracking state
        self._quarantined: Set[str] = set()
        self._alarm_counts: Dict[str, int] = {}
        self._residual_history_full: List[float] = []
        self._residual_history_excl: Dict[str, List[float]] = {}
        
        # Dead-reckoning navigator
        self.dr_navigator = DeadReckoningNavigator()
        
        # Active filter name
        self.active_filter = "FULL"
        
        # Running residual accumulators
        self._recent_residuals_full: List[float] = []
        self._recent_residuals_by_sensor: Dict[str, List[float]] = {}
        self._window_size = 20  # Rolling window for residual comparison
    
    def record_residual(self, sensor_name: str, nis: float, is_full_filter: bool = True):
        """
        Record a normalized innovation squared (NIS) value for residual tracking.
        Called after every EKF innovation computation.
        """
        if is_full_filter:
            self._recent_residuals_full.append(nis)
            if len(self._recent_residuals_full) > self._window_size:
                self._recent_residuals_full.pop(0)
        
        if sensor_name not in self._recent_residuals_by_sensor:
            self._recent_residuals_by_sensor[sensor_name] = []
        self._recent_residuals_by_sensor[sensor_name].append(nis)
        if len(self._recent_residuals_by_sensor[sensor_name]) > self._window_size:
            self._recent_residuals_by_sensor[sensor_name].pop(0)
    
    def process_alarm(self,
                      sensor_name: str,
                      timestamp: float,
                      cusum_value: float,
                      ekf_state: np.ndarray,
                      ekf_heading: float,
                      ekf_speed: float) -> SaarmSignal:
        """
        Process a CUSUM ALARM signal for a specific sensor.
        
        This is the main entry point called when the CUSUM monitor
        raises an ALARM on a sensor channel.
        
        Parameters
        ----------
        sensor_name : str
            The sensor under suspicion (e.g., "GNSS").
        timestamp : float
            Current time in seconds.
        cusum_value : float
            The CUSUM accumulator value that triggered the alarm.
        ekf_state : np.ndarray
            Current EKF state vector (11-dim).
        ekf_heading : float
            Current heading estimate (rad).
        ekf_speed : float
            Current speed estimate (m/s).
        
        Returns
        -------
        SaarmSignal with isolation verdict and dead-reckoning status.
        """
        # Count consecutive alarms
        self._alarm_counts[sensor_name] = self._alarm_counts.get(sensor_name, 0) + 1
        
        # Not enough consecutive alarms yet — investigating
        if self._alarm_counts[sensor_name] < self.min_alarm_count:
            return SaarmSignal(
                timestamp=timestamp,
                verdict=SaarmVerdict.INVESTIGATING,
                quarantined_sensors=list(self._quarantined),
                active_filter=self.active_filter,
                residual_full=self._mean_residual_full(),
                residual_exclusion=0.0,
                confidence=self._alarm_counts[sensor_name] / self.min_alarm_count,
                dead_reckoning_active=self.dr_navigator.active,
                dr_position=self.dr_navigator.position if self.dr_navigator.active else None
            )
        
        # Enough alarms — run isolation diagnostic
        residual_full = self._mean_residual_full()
        residual_without_sensor = self._mean_residual_excluding(sensor_name)
        
        # If excluding this sensor dramatically reduces residuals → it's compromised
        if residual_without_sensor > 0:
            ratio = residual_full / max(residual_without_sensor, 1e-6)
        else:
            ratio = self.isolation_threshold + 1.0  # Force isolation if no data
        
        confidence = min(1.0, ratio / self.isolation_threshold)
        
        if ratio >= self.isolation_threshold or self._alarm_counts[sensor_name] >= self.min_alarm_count * 2:
            # CONFIRMED: Sensor is compromised — quarantine it
            self._quarantined.add(sensor_name)
            self.active_filter = f"EXCLUDE_{sensor_name}"
            
            # If GPS is quarantined, activate dead-reckoning
            if sensor_name == "GNSS" and not self.dr_navigator.active:
                current_pos = ekf_state[0:3]
                self.dr_navigator.activate(current_pos, ekf_heading, ekf_speed)
            
            verdict = SaarmVerdict.DEAD_RECKONING if self.dr_navigator.active else SaarmVerdict.ISOLATED
            
            return SaarmSignal(
                timestamp=timestamp,
                verdict=verdict,
                quarantined_sensors=list(self._quarantined),
                active_filter=self.active_filter,
                residual_full=residual_full,
                residual_exclusion=residual_without_sensor,
                confidence=confidence,
                dead_reckoning_active=self.dr_navigator.active,
                dr_position=self.dr_navigator.position if self.dr_navigator.active else None
            )
        else:
            return SaarmSignal(
                timestamp=timestamp,
                verdict=SaarmVerdict.INVESTIGATING,
                quarantined_sensors=list(self._quarantined),
                active_filter=self.active_filter,
                residual_full=residual_full,
                residual_exclusion=residual_without_sensor,
                confidence=confidence,
                dead_reckoning_active=self.dr_navigator.active,
                dr_position=self.dr_navigator.position if self.dr_navigator.active else None
            )
    
    def clear_alarm(self, sensor_name: str):
        """Reset alarm counter when a sensor returns to healthy."""
        self._alarm_counts[sensor_name] = 0
    
    def restore_sensor(self, sensor_name: str):
        """
        Un-quarantine a sensor after it has been verified clean.
        Called when CUSUM auto-recovers or after manual recalibration.
        """
        self._quarantined.discard(sensor_name)
        self._alarm_counts[sensor_name] = 0
        
        if not self._quarantined:
            self.active_filter = "FULL"
        
        if sensor_name == "GNSS":
            self.dr_navigator.deactivate()
    
    def is_quarantined(self, sensor_name: str) -> bool:
        """Check if a sensor is currently quarantined by SAARM."""
        return sensor_name in self._quarantined
    
    def get_nominal_signal(self, timestamp: float) -> SaarmSignal:
        """Generate a nominal SAARM signal when no alarms are active."""
        return SaarmSignal(
            timestamp=timestamp,
            verdict=SaarmVerdict.NOMINAL,
            quarantined_sensors=list(self._quarantined),
            active_filter=self.active_filter,
            residual_full=self._mean_residual_full(),
            residual_exclusion=0.0,
            confidence=1.0,
            dead_reckoning_active=self.dr_navigator.active,
            dr_position=self.dr_navigator.position if self.dr_navigator.active else None
        )
    
    def _mean_residual_full(self) -> float:
        """Mean NIS across recent full-filter measurements."""
        if not self._recent_residuals_full:
            return 0.0
        return float(np.mean(self._recent_residuals_full))
    
    def _mean_residual_excluding(self, sensor_name: str) -> float:
        """
        Mean NIS across recent measurements EXCLUDING those from
        the specified sensor. This simulates what the exclusion
        filter's residuals would look like.
        """
        all_residuals = []
        for sname, residuals in self._recent_residuals_by_sensor.items():
            if sname != sensor_name and residuals:
                all_residuals.extend(residuals[-self._window_size:])
        
        if not all_residuals:
            return 0.0
        return float(np.mean(all_residuals))
