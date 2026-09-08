"""
directional_spoof_detector.py — DIRECTIONAL SPOOF VECTOR MONITOR
==================================================================
Sequential vector accumulator and directional coherence analyzer
for detecting persistent unidirectional GPS/GNSS spoofing attacks.

Purpose:
    When an attacker continuously injects false pseudoranges to drag
    a UAV off-course in a specific direction (e.g., East or toward a trap),
    the individual frame innovations may stay near the threshold, but the
    vectors consistently point in the same direction.
    
    This detector maintains a sliding window of the last N innovation vectors
    (default N=5) and computes:
    
    1. Vector Sum (Constructive vs. Destructive Addition):
       V_accum = sum_{i=1}^N v_i
       Under random zero-mean noise, vectors cancel out: ||V_accum|| ≈ 0.
       Under continuous directional spoofing, vectors add up: ||V_accum|| ≈ sum ||v_i||.
       
    2. Directional Coherence (Cosine Alignment):
       coherence = ||V_accum|| / (sum ||v_i|| + ε) ∈ [0, 1]
       A coherence close to 1.0 proves all recent spoof pushes are in the same direction.
       
    3. Location Dis-trust & Countermeasures:
       - Flags DIRECTIONAL_ATTACK when cumulative drift and coherence breach thresholds.
       - Signals the autopilot to STOP TRUSTING the final/reported GPS location.
       - Projects out the attack vector using orthogonal projection (P_perp = I - u*u^T)
         to mathematically IGNORE any drift along that direction.

Hardware Interface:
    DirectionalSpoofSignal.to_bytes() → 24-byte packet for CAN / UART / SPI.
    DirectionalSpoofSignal.to_dict()  → JSON-serializable dict for ROS2 / TCP / MQTT.
"""

from __future__ import annotations

import math
import struct
from collections import deque
from dataclasses import dataclass, field
from enum import IntEnum
from typing import Deque, Dict, List, Optional, Tuple

import numpy as np


class DirectionalStatus(IntEnum):
    """
    Health classification for directional sensor drift.
      0 = HEALTHY            (Random zero-mean noise, vectors cancel out)
      1 = SUSPICIOUS         (Emerging directional bias, monitoring window)
      2 = DIRECTIONAL_ATTACK (Unidirectional spoof confirmed: vectors aligned & threshold exceeded)
    """
    HEALTHY = 0
    SUSPICIOUS = 1
    DIRECTIONAL_ATTACK = 2


@dataclass
class DirectionalSpoofSignal:
    """
    Output decision packet from the Directional Spoof Vector Monitor.
    Designed for hardware controllers, autopilot governors, and ROS2 buses.
    """
    timestamp: float
    sensor: str
    status: DirectionalStatus
    cumulative_vector: np.ndarray      # Net vector sum [Vx, Vy] (or [Vx, Vy, Vz])
    cumulative_magnitude: float        # ||V_accum|| in meters
    scalar_sum: float                  # sum ||v_i|| in meters
    coherence: float                   # Directional alignment ratio [0.0 to 1.0]
    attack_heading_deg: float          # Heading of the attack vector (0-360 deg)
    window_count: int                  # Current number of samples in window
    stop_trusting_location: bool       # True = Revoke trust in reported GPS location
    sanitized_vector: np.ndarray       # Innovation vector with attack direction projected out

    def to_dict(self) -> dict:
        """JSON / telemetry serialization."""
        return {
            'timestamp': round(self.timestamp, 4),
            'sensor': self.sensor,
            'status': self.status.name,
            'status_code': int(self.status),
            'cumulative_vector': [round(float(x), 4) for x in self.cumulative_vector],
            'cumulative_magnitude_m': round(float(self.cumulative_magnitude), 3),
            'scalar_sum_m': round(float(self.scalar_sum), 3),
            'coherence': round(float(self.coherence), 4),
            'attack_heading_deg': round(float(self.attack_heading_deg), 2),
            'window_count': int(self.window_count),
            'stop_trusting_location': bool(self.stop_trusting_location),
            'sanitized_vector': [round(float(x), 4) for x in self.sanitized_vector],
        }

    def to_bytes(self) -> bytes:
        """
        Compact 24-byte hardware packet:
        [uint32 timestamp_ms, uint8 status_code, uint8 stop_trust, uint8 window_count, uint8 reserved,
         float32 cum_mag, float32 coherence, float32 heading_deg, float32 vx, float32 vy]
        """
        ts_ms = int(self.timestamp * 1000) & 0xFFFFFFFF
        vx = float(self.cumulative_vector[0]) if len(self.cumulative_vector) > 0 else 0.0
        vy = float(self.cumulative_vector[1]) if len(self.cumulative_vector) > 1 else 0.0
        return struct.pack(
            '<IBBBffff',
            ts_ms,
            int(self.status),
            1 if self.stop_trusting_location else 0,
            min(255, self.window_count),
            float(self.cumulative_magnitude),
            float(self.coherence),
            float(self.attack_heading_deg),
            vx,
        )


class DirectionalSpoofDetector:
    """
    Directional Spoof Vector Accumulator & Filtering Subsystem.
    
    Tracks multi-dimensional innovation vectors over a sliding window
    (default N=5) to detect continuous spoof pushes in a common direction.
    
    Parameters
    ----------
    window_size : int
        Number of consecutive vectors to accumulate (default 5).
    cumulative_threshold : float
        Net vector magnitude ||V_accum|| in meters required to trigger alarm (default 15.0m).
    coherence_threshold : float
        Minimum directional coherence ratio required to confirm common direction (default 0.80 = 80%).
    min_vector_norm : float
        Minimum individual vector length in meters to consider (filters negligible noise floor, default 0.2m).
    history_decay : float
        Decay factor for historical vectors when clean signals arrive (default 0.0 = strict FIFO).
    """

    def __init__(
        self,
        window_size: int = 5,
        cumulative_threshold: float = 15.0,
        coherence_threshold: float = 0.80,
        min_vector_norm: float = 0.2,
    ):
        self.window_size = max(2, int(window_size))
        self.cumulative_threshold = float(cumulative_threshold)
        self.coherence_threshold = float(coherence_threshold)
        self.min_vector_norm = float(min_vector_norm)

        # Per-sensor sliding vector window: deque of (timestamp, np.ndarray)
        self._windows: Dict[str, Deque[Tuple[float, np.ndarray]]] = {}
        # Active unit attack direction per sensor: u_hat
        self._attack_directions: Dict[str, Optional[np.ndarray]] = {}
        # A confirmed attack is latched until a trusted recovery procedure
        # explicitly resets it.  A new spoof direction must never make a
        # compromised GNSS source appear healthy again.
        self._attack_latched: Dict[str, bool] = {}

    def reset(self, sensor: Optional[str] = None) -> None:
        """Reset accumulated vector windows."""
        if sensor is None:
            self._windows.clear()
            self._attack_directions.clear()
            self._attack_latched.clear()
        else:
            if sensor in self._windows:
                self._windows[sensor].clear()
            self._attack_directions[sensor] = None
            self._attack_latched[sensor] = False

    def is_attack_latched(self, sensor: str) -> bool:
        """Return whether *sensor* remains quarantined after a confirmed attack.

        Clearing this state is deliberately an explicit operation through
        :meth:`reset`; clean-looking samples immediately after a spoof are not
        enough to re-authorize a navigation source.
        """
        return self._attack_latched.get(sensor, False)

    def attack_direction(self, sensor: str) -> Optional[np.ndarray]:
        """Return a copy of the confirmed attack direction, if one is latched."""
        direction = self._attack_directions.get(sensor)
        return None if direction is None else direction.copy()

    def evaluate(
        self,
        sensor: str,
        timestamp: float,
        innovation_vector: np.ndarray,
    ) -> DirectionalSpoofSignal:
        """
        Evaluate an incoming innovation/spoof vector.
        
        Parameters
        ----------
        sensor : str
            Sensor identifier, e.g. 'GNSS'
        timestamp : float
            Measurement timestamp in seconds
        innovation_vector : np.ndarray
            Measured minus predicted position residual y (e.g. [dx, dy] or [dx, dy, dz])
            
        Returns
        -------
        DirectionalSpoofSignal
            Contains verdict, net accumulated vector, coherence, and sanitized vector.
        """
        vec = np.asarray(innovation_vector, dtype=float).flatten()
        # Restrict to spatial 2D or 3D position component (first 2 or 3 states)
        # Standardize to 2D horizontal navigation plane [dx, dy] for attack direction tracking
        if len(vec) >= 2:
            pos_vec = vec[:2].astype(float)
        else:
            pos_vec = np.pad(vec.astype(float), (0, 2 - len(vec)))

        if sensor not in self._windows:
            self._windows[sensor] = deque(maxlen=self.window_size)
            self._attack_directions[sensor] = None
            self._attack_latched[sensor] = False

        win = self._windows[sensor]

        # Record vector if it meets minimum non-zero threshold
        vec_norm = float(np.linalg.norm(pos_vec))
        if vec_norm >= self.min_vector_norm:
            win.append((timestamp, pos_vec.copy()))
        elif len(win) > 0 and vec_norm < (self.min_vector_norm * 0.5):
            # Optional gradual drainage of window under completely zero noise
            pass

        # If window has insufficient samples, report healthy
        if len(win) == 0:
            zero_v = np.zeros(len(pos_vec))
            return DirectionalSpoofSignal(
                timestamp=timestamp,
                sensor=sensor,
                status=DirectionalStatus.HEALTHY,
                cumulative_vector=zero_v,
                cumulative_magnitude=0.0,
                scalar_sum=0.0,
                coherence=0.0,
                attack_heading_deg=0.0,
                window_count=0,
                stop_trusting_location=False,
                sanitized_vector=pos_vec.copy(),
            )

        # 1. Sum vectors (Vector Addition)
        vectors_list = [v for _, v in win]
        v_accum = np.sum(vectors_list, axis=0)
        cum_mag = float(np.linalg.norm(v_accum))

        # 2. Sum scalar magnitudes
        scalar_sum = float(sum(np.linalg.norm(v) for v in vectors_list))

        # 3. Directional Coherence (Alignment ratio)
        coherence = cum_mag / (scalar_sum + 1e-9)
        coherence = float(np.clip(coherence, 0.0, 1.0))

        # 4. Attack Heading in degrees (0 = North, 90 = East or standard cartesian atan2)
        if len(v_accum) >= 2 and cum_mag > 1e-6:
            # Navigation angle: 0 deg North, 90 deg East
            heading_rad = math.atan2(v_accum[0], v_accum[1]) # dx=East, dy=North
            heading_deg = (math.degrees(heading_rad) + 360.0) % 360.0
            u_hat = v_accum / cum_mag
        else:
            heading_deg = 0.0
            u_hat = None

        # 5. Threshold & Decision Logic
        stop_trusting = False
        if self._attack_latched[sensor]:
            # Keep the first confirmed direction as forensic evidence.  Once
            # GNSS has been declared compromised, a sudden change in spoof
            # direction is further evidence of an attack, not a recovery.
            status = DirectionalStatus.DIRECTIONAL_ATTACK
            stop_trusting = True
        elif len(win) >= min(3, self.window_size):
            if cum_mag >= self.cumulative_threshold and coherence >= self.coherence_threshold:
                status = DirectionalStatus.DIRECTIONAL_ATTACK
                stop_trusting = True
                self._attack_directions[sensor] = u_hat
                self._attack_latched[sensor] = True
            elif coherence >= (self.coherence_threshold * 0.85) and cum_mag >= (self.cumulative_threshold * 0.5):
                status = DirectionalStatus.SUSPICIOUS
                self._attack_directions[sensor] = u_hat
            else:
                status = DirectionalStatus.HEALTHY
                self._attack_directions[sensor] = None
        else:
            status = DirectionalStatus.HEALTHY

        # 6. Sanitize innovation vector (Orthogonal Projection / Vector Ignoring)
        active_direction = self._attack_directions[sensor]
        if active_direction is not None:
            heading_rad = math.atan2(active_direction[0], active_direction[1])
            heading_deg = (math.degrees(heading_rad) + 360.0) % 360.0
        sanitized = self._project_orthogonal(pos_vec, active_direction)

        return DirectionalSpoofSignal(
            timestamp=timestamp,
            sensor=sensor,
            status=status,
            cumulative_vector=v_accum,
            cumulative_magnitude=cum_mag,
            scalar_sum=scalar_sum,
            coherence=coherence,
            attack_heading_deg=heading_deg,
            window_count=len(win),
            stop_trusting_location=stop_trusting,
            sanitized_vector=sanitized,
        )

    def filter_vector(self, sensor: str, vector: np.ndarray) -> np.ndarray:
        """
        Filter out the directional spoof component from an innovation or measurement.
        
        Computes the orthogonal complement of the attack direction:
            P_perp = I - u_hat * u_hat^T
            y_sanitized = P_perp * y
            
        If no attack direction is active, returns the vector unchanged.
        """
        u_hat = self._attack_directions.get(sensor, None)
        if u_hat is None:
            return vector.copy()
        return self._project_orthogonal(vector, u_hat)

    def is_direction_aligned(
        self,
        sensor: str,
        test_vector: np.ndarray,
        angular_tolerance_deg: float = 45.0,
    ) -> bool:
        """
        Check if a given vector (e.g. waypoint delta or destination shift)
        aligns with the active directional spoof attack vector.
        
        Returns True if the angle between test_vector and the attack direction
        is within angular_tolerance_deg.
        """
        u_hat = self._attack_directions.get(sensor, None)
        if u_hat is None:
            return False

        tv = np.asarray(test_vector, dtype=float).flatten()[:len(u_hat)]
        tv_norm = np.linalg.norm(tv)
        if tv_norm < 1e-6:
            return False

        u_test = tv / tv_norm
        cos_sim = float(np.dot(u_hat, u_test))
        cos_thresh = math.cos(math.radians(angular_tolerance_deg))
        return cos_sim >= cos_thresh

    @staticmethod
    def _project_orthogonal(vector: np.ndarray, u_hat: Optional[np.ndarray]) -> np.ndarray:
        """
        Project vector onto the orthogonal subspace perpendicular to u_hat:
            v_perp = v - (v · u_hat) * u_hat
        """
        v = np.asarray(vector, dtype=float).copy()
        if u_hat is None or len(v) == 0:
            return v

        dim = min(len(v), len(u_hat))
        v_sub = v[:dim]
        dot_prod = float(np.dot(v_sub, u_hat[:dim]))
        v[:dim] = v_sub - dot_prod * u_hat[:dim]
        return v
