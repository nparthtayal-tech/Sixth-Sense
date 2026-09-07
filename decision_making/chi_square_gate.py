"""
chi_square_gate.py — FAST CHANGE MONITOR
=========================================
Instantaneous single-frame outlier detection using Chi-Square (χ²)
Mahalanobis distance gating on EKF innovation vectors.

Purpose:
    Catches sudden, violent measurement spikes in real-time.
    Examples: GPS multipath reflections off glass buildings, LiDAR echoes
    off dust/rain, ultrasonic false echoes, camera feature mismatches.

Input:
    Innovation vector y and innovation covariance S from the EKF.
    (Obtained via ekf.compute_innovation())

Output:
    A DecisionSignal with ACCEPT / REJECT verdict and numerical telemetry.

Latency:
    Zero memory. Purely memoryless. Decision in 1 sample.

Hardware Interface:
    DecisionSignal.to_bytes() → 12-byte packet for CAN / UART / SPI.
    DecisionSignal.to_dict()  → JSON-serializable dict for ROS2 / TCP / MQTT.
"""

import struct
import numpy as np
from dataclasses import dataclass
from enum import IntEnum


class FastVerdict(IntEnum):
    """Single-byte verdict code for hardware controllers."""
    ACCEPT = 0    # Measurement is clean, fuse into EKF
    REJECT = 1    # Measurement is a spike/outlier, discard it


@dataclass
class DecisionSignal:
    """
    Output signal packet from the Fast Change Monitor.
    Designed for direct transmission to hardware decision units.
    """
    timestamp: float
    sensor: str
    verdict: FastVerdict
    nis: float             # Normalized Innovation Squared (Mahalanobis d²)
    threshold: float       # Chi-Square cutoff used
    dim_m: int             # Measurement dimension

    def to_dict(self) -> dict:
        """JSON / telemetry serialization."""
        return {
            'timestamp': round(self.timestamp, 4),
            'sensor': self.sensor,
            'verdict': self.verdict.name,
            'verdict_code': int(self.verdict),
            'nis': round(float(self.nis), 4),
            'threshold': round(float(self.threshold), 4),
            'dim_m': self.dim_m
        }

    def to_bytes(self) -> bytes:
        """
        Compact 12-byte hardware packet:
        [uint32 timestamp_ms, uint8 verdict, uint8 dim_m,
         uint8 reserved, uint8 reserved, float32 nis]
        """
        ts_ms = int(self.timestamp * 1000) & 0xFFFFFFFF
        return struct.pack('<IBBBBf',
                           ts_ms,
                           int(self.verdict),
                           self.dim_m & 0xFF,
                           0, 0,
                           float(self.nis))


class ChiSquareGate:
    """
    Fast Change Monitor — Memoryless Chi-Square Outlier Gate.
    
    For a measurement of dimension m, the Normalized Innovation Squared
    (NIS = y^T S^{-1} y) follows a Chi-Square distribution with m
    degrees of freedom under healthy conditions.
    
    The gate rejects any measurement whose NIS exceeds the threshold:
        NIS > χ²(m, confidence)
    
    Parameters
    ----------
    confidence : float
        Confidence level for the gate (default 0.999 = 99.9%).
        Higher = more permissive (fewer false rejections).
        Lower  = more aggressive (rejects more borderline readings).
    """

    def __init__(self, confidence: float = 0.999):
        self.confidence = confidence
        
        # Counters for diagnostics
        self.total_evaluated = 0
        self.total_rejected = 0
        self._per_sensor_rejected: dict[str, int] = {}
        self._per_sensor_total: dict[str, int] = {}

    def _chi2_threshold(self, m: int) -> float:
        """
        Approximate χ²(m, 0.999) threshold.
        Uses Wilson-Hilferty normal approximation:
            χ²_p ≈ m * (1 - 2/(9m) + z_p * sqrt(2/(9m)))^3
        where z_p = 3.09 for p = 0.999.
        
        Accurate to <1% for m >= 1.
        """
        if m <= 0:
            return 10.0
        z_p = 3.09  # z-score for 99.9% one-sided
        ratio = 2.0 / (9.0 * m)
        cube = (1.0 - ratio + z_p * np.sqrt(ratio)) ** 3
        return max(m + 3.0, m * cube)

    def evaluate(self,
                 sensor_name: str,
                 timestamp: float,
                 y: np.ndarray,
                 S: np.ndarray,
                 dim_m: int) -> DecisionSignal:
        """
        Evaluate a single measurement against the Chi-Square gate.
        
        Parameters
        ----------
        sensor_name : str
            Identifier of the sensor (e.g. "GNSS", "LIDAR").
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
        DecisionSignal with verdict ACCEPT or REJECT.
        """
        # Compute NIS = y^T S^{-1} y
        try:
            SI = np.linalg.inv(S)
            nis = float(np.dot(y.T, np.dot(SI, y)))
        except np.linalg.LinAlgError:
            nis = float(np.sum(y ** 2))

        threshold = self._chi2_threshold(dim_m)

        # Bookkeeping
        self.total_evaluated += 1
        self._per_sensor_total[sensor_name] = self._per_sensor_total.get(sensor_name, 0) + 1

        if nis > threshold:
            self.total_rejected += 1
            self._per_sensor_rejected[sensor_name] = self._per_sensor_rejected.get(sensor_name, 0) + 1
            verdict = FastVerdict.REJECT
        else:
            verdict = FastVerdict.ACCEPT

        return DecisionSignal(
            timestamp=timestamp,
            sensor=sensor_name,
            verdict=verdict,
            nis=nis,
            threshold=threshold,
            dim_m=dim_m
        )

    def rejection_rate(self, sensor_name: str = None) -> float:
        """Returns fraction of measurements rejected (0.0 to 1.0)."""
        if sensor_name:
            total = self._per_sensor_total.get(sensor_name, 0)
            rejected = self._per_sensor_rejected.get(sensor_name, 0)
        else:
            total = self.total_evaluated
            rejected = self.total_rejected
        return rejected / max(1, total)
