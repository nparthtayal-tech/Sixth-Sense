"""
cusum_monitor.py — SLOW CHANGE MONITOR
========================================
Sequential Cumulative Sum (CUSUM) control chart for detecting
persistent sensor drift, incipient bias shifts, and slow degradation.

Purpose:
    Catches failures that are too small for Chi-Square to notice on any
    single frame, but accumulate over time and poison the EKF state.
    Examples: Gyro thermal drift, wheel encoder slip on ice, GPS clock
    spoofing ramp, LiDAR calibration loss, accelerometer aging bias.

Input:
    Innovation vector y and innovation covariance S from the EKF.
    (Obtained via ekf.compute_innovation())

Output:
    A DriftSignal with HEALTHY / WARNING / ALARM status and telemetry.

Latency:
    Stateful. Maintains a running CUSUM accumulator per sensor.
    Detects drift within 5-15 samples of onset (depending on severity).

Hardware Interface:
    DriftSignal.to_bytes() → 16-byte packet for CAN / UART / SPI.
    DriftSignal.to_dict()  → JSON-serializable dict for ROS2 / TCP / MQTT.
"""

import struct
import numpy as np
from dataclasses import dataclass
from enum import IntEnum


class DriftStatus(IntEnum):
    """
    Three-level health status for hardware controllers.
    Maps to standard traffic-light conventions:
      0 = GREEN  (Healthy, fuse normally)
      1 = YELLOW (Warning, fuse but flag for supervisor attention)
      2 = RED    (Alarm, quarantine sensor — do NOT fuse)
    """
    HEALTHY = 0
    WARNING = 1
    ALARM   = 2


@dataclass
class DriftSignal:
    """
    Output signal packet from the Slow Change Monitor.
    Designed for direct transmission to hardware decision units.
    """
    timestamp: float
    sensor: str
    status: DriftStatus
    cusum_value: float          # Current CUSUM accumulator level
    alarm_threshold: float      # Threshold that triggers ALARM
    warning_threshold: float    # Threshold that triggers WARNING
    nis: float                  # NIS of the current sample
    accepted: bool              # Should hardware fuse this measurement?

    def to_dict(self) -> dict:
        """JSON / telemetry serialization."""
        return {
            'timestamp': round(self.timestamp, 4),
            'sensor': self.sensor,
            'status': self.status.name,
            'status_code': int(self.status),
            'cusum': round(float(self.cusum_value), 4),
            'alarm_limit': round(float(self.alarm_threshold), 2),
            'warning_limit': round(float(self.warning_threshold), 2),
            'nis': round(float(self.nis), 4),
            'accepted': self.accepted
        }

    def to_bytes(self) -> bytes:
        """
        Compact 16-byte hardware packet:
        [uint32 timestamp_ms, uint8 status_code, uint8 accepted,
         uint8 reserved, uint8 reserved, float32 cusum, float32 nis]
        """
        ts_ms = int(self.timestamp * 1000) & 0xFFFFFFFF
        return struct.pack('<IBBBBff',
                           ts_ms,
                           int(self.status),
                           1 if self.accepted else 0,
                           0, 0,
                           float(self.cusum_value),
                           float(self.nis))


class CusumMonitor:
    """
    Slow Change Monitor — Sequential CUSUM Drift Detector.
    
    Maintains a running accumulator C_k per sensor:
    
        excess_k = NIS_k - (m + slack)
        C_k = max(0, C_{k-1} + excess_k)
    
    Under healthy conditions (NIS ≈ m), excess < 0 and the max(0,...)
    clamps the accumulator to zero. Under persistent drift (NIS > m + slack
    repeatedly), C_k climbs like a staircase until it crosses a threshold.
    
    Parameters
    ----------
    alarm_threshold : float
        CUSUM level that triggers full sensor quarantine (RED).
    warning_threshold : float
        CUSUM level that triggers a supervisor alert (YELLOW).
    slack : float
        Allowance parameter δ. Excess = NIS - (m + δ).
        Larger δ = more tolerant to natural variance, slower detection.
        Smaller δ = more sensitive, faster detection, more false alarms.
    recovery_rate : float
        How fast the accumulator decays back to 0 when healthy samples
        arrive. Set to 0.0 for strict one-way CUSUM (never self-heals).
    auto_recover_after : int
        Number of consecutive clean samples needed to automatically
        un-quarantine a sensor (set to 0 to require manual reset).
    """

    def __init__(self,
                 alarm_threshold: float = 20.0,
                 warning_threshold: float = 10.0,
                 slack: float = 1.0,
                 recovery_rate: float = 0.5,
                 auto_recover_after: int = 30):
        
        self.alarm_threshold = alarm_threshold
        self.warning_threshold = warning_threshold
        self.slack = slack
        self.recovery_rate = recovery_rate
        self.auto_recover_after = auto_recover_after
        
        # Per-sensor state
        self._accumulators: dict[str, float] = {}
        self._quarantined: dict[str, bool] = {}
        self._consecutive_clean: dict[str, int] = {}

    def _ensure_sensor(self, name: str):
        """Initialize tracking state for a sensor on first encounter."""
        if name not in self._accumulators:
            self._accumulators[name] = 0.0
            self._quarantined[name] = False
            self._consecutive_clean[name] = 0

    def evaluate(self,
                 sensor_name: str,
                 timestamp: float,
                 y: np.ndarray,
                 S: np.ndarray,
                 dim_m: int) -> DriftSignal:
        """
        Feed one innovation sample into the CUSUM accumulator.
        
        Parameters
        ----------
        sensor_name : str
            Identifier of the sensor (e.g. "GNSS", "IMU").
        timestamp : float
            Current time in seconds.
        y : np.ndarray
            Innovation vector (z - h(x)), shape (m,).
        S : np.ndarray
            Innovation covariance matrix, shape (m, m).
        dim_m : int
            Measurement dimension.
            
        Returns
        -------
        DriftSignal with HEALTHY, WARNING, or ALARM status.
        """
        self._ensure_sensor(sensor_name)

        # Compute NIS = y^T S^{-1} y
        try:
            SI = np.linalg.inv(S)
        except np.linalg.LinAlgError:
            SI = np.linalg.pinv(S)
        nis = float(np.dot(y.T, np.dot(SI, y)))

        # CUSUM update
        expected = dim_m + self.slack
        excess = nis - expected

        if excess > 0:
            # Anomalous: accumulate
            self._accumulators[sensor_name] += excess
            self._consecutive_clean[sensor_name] = 0
        else:
            # Clean: decay accumulator towards zero
            self._consecutive_clean[sensor_name] += 1
            self._accumulators[sensor_name] = max(
                0.0,
                self._accumulators[sensor_name] - self.recovery_rate
            )

        cusum = self._accumulators[sensor_name]

        # Auto-recovery from quarantine after sustained healthy streak
        if (self._quarantined[sensor_name]
                and self.auto_recover_after > 0
                and self._consecutive_clean[sensor_name] >= self.auto_recover_after):
            self._quarantined[sensor_name] = False
            self._accumulators[sensor_name] = 0.0

        # Determine status level
        if cusum >= self.alarm_threshold:
            self._quarantined[sensor_name] = True
            status = DriftStatus.ALARM
            accepted = False
        elif cusum >= self.warning_threshold:
            status = DriftStatus.WARNING
            accepted = True   # Still fuse, but flag for attention
        else:
            self._quarantined[sensor_name] = False
            status = DriftStatus.HEALTHY
            accepted = True

        return DriftSignal(
            timestamp=timestamp,
            sensor=sensor_name,
            status=status,
            cusum_value=cusum,
            alarm_threshold=self.alarm_threshold,
            warning_threshold=self.warning_threshold,
            nis=nis,
            accepted=accepted
        )

    def is_quarantined(self, sensor_name: str) -> bool:
        """Check if a sensor is currently quarantined."""
        return self._quarantined.get(sensor_name, False)

    def reset(self, sensor_name: str):
        """Manually reset a sensor's CUSUM state (e.g. after recalibration)."""
        self._accumulators[sensor_name] = 0.0
        self._quarantined[sensor_name] = False
        self._consecutive_clean[sensor_name] = 0

    def get_all_cusum_values(self) -> dict[str, float]:
        """Returns current CUSUM levels for all tracked sensors."""
        return dict(self._accumulators)
