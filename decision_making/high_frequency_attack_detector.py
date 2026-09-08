"""
high_frequency_attack_detector.py — HIGH-FREQUENCY CYBER ATTACK & DRIFT NEUTRALIZER
====================================================================================
Detects high-frequency continuous cyber attacks, analyzes changing attack patterns,
calculates overall spatial drift from the mission launch point, and computes
the negative drift ("nega drift") vector to counter-balance attack displacement
and command an emergency return to the starting location (rejecting all new destinations).

Purpose:
    When an adversary launches rapid-fire, high-frequency spoofing bursts or
    continuously shifts attack patterns (e.g., alternating headings or modulating offsets),
    individual frame monitors or simple sliding windows may experience continuous drift.
    The drone gets pulled away from its course, and allowing new waypoints is dangerous.

    This subsystem:
    1. Measures attack arrival frequency (f = 1 / mean(dt)) and burst density.
    2. Analyzes the changing pattern of attacks (angular deltas and directional variance).
    3. Calculates overall cumulative drift from the verified start location:
           d_overall = p_current - p_start
    4. Calculates the negative drift ("nega drift") compensation vector:
           v_nega_drift = - d_overall
    5. Flags an emergency override:
           - Overrides navigation to return directly to the starting position.
           - Strictly forbids/rejects any new destination.
           - Emits negative drift guidance to halt displacement dead in its tracks.
"""

from __future__ import annotations

import math
import struct
from collections import deque
from dataclasses import dataclass
from enum import IntEnum
from typing import Deque, Dict, List, Optional, Tuple

import numpy as np


class AttackPattern(IntEnum):
    """Classification of cyber attack patterns across continuous attacks."""
    HEALTHY = 0
    UNIDIRECTIONAL = 1          # Consecutive attacks in a single consistent direction
    RAPID_ALTERNATING = 2       # High-frequency flip-flop attacks (e.g. ±90° to ±180°)
    OSCILLATING = 3             # Cyclic angular modulation across frames
    HIGH_FREQ_BURST = 4         # Rapid dense burst of malicious frames in a short window
    ERRATIC_SCATTER = 5         # High variance multi-directional injection


@dataclass
class HighFrequencySignal:
    """
    Decision and drift telemetry signal produced by HighFrequencyAttackDetector.
    """
    timestamp: float
    sensor: str
    is_high_frequency: bool
    attack_frequency_hz: float
    burst_count: int
    attack_pattern: AttackPattern
    pattern_variance_deg: float
    overall_drift_vector: np.ndarray       # d_overall = p_current - p_start
    overall_drift_magnitude: float         # ||d_overall|| in meters
    negative_drift_vector: np.ndarray      # v_nega_drift = - d_overall
    override_to_start: bool                # True = Stop everything & return to start
    target_start_location: np.ndarray      # Immutable launch coordinates to return to
    guidance_unit_vector: np.ndarray       # Normalized homing vector toward start location

    def to_dict(self) -> dict:
        """JSON / telemetry serialization."""
        return {
            'timestamp': round(self.timestamp, 4),
            'sensor': self.sensor,
            'is_high_frequency': bool(self.is_high_frequency),
            'attack_frequency_hz': round(float(self.attack_frequency_hz), 2),
            'burst_count': int(self.burst_count),
            'attack_pattern': self.attack_pattern.name,
            'attack_pattern_code': int(self.attack_pattern),
            'pattern_variance_deg': round(float(self.pattern_variance_deg), 2),
            'overall_drift_vector': [round(float(x), 4) for x in self.overall_drift_vector],
            'overall_drift_magnitude_m': round(float(self.overall_drift_magnitude), 3),
            'negative_drift_vector': [round(float(x), 4) for x in self.negative_drift_vector],
            'override_to_start': bool(self.override_to_start),
            'target_start_location': [round(float(x), 4) for x in self.target_start_location],
            'guidance_unit_vector': [round(float(x), 4) for x in self.guidance_unit_vector],
        }

    def to_bytes(self) -> bytes:
        """
        Compact 32-byte hardware packet:
        [uint32 timestamp_ms, uint8 is_hf, uint8 pattern, uint8 override, uint8 burst_cnt,
         float32 freq_hz, float32 drift_mag, float32 nega_vx, float32 nega_vy, float32 guide_x, float32 guide_y]
        """
        ts_ms = int(self.timestamp * 1000) & 0xFFFFFFFF
        nega_vx = float(self.negative_drift_vector[0]) if len(self.negative_drift_vector) > 0 else 0.0
        nega_vy = float(self.negative_drift_vector[1]) if len(self.negative_drift_vector) > 1 else 0.0
        guide_x = float(self.guidance_unit_vector[0]) if len(self.guidance_unit_vector) > 0 else 0.0
        guide_y = float(self.guidance_unit_vector[1]) if len(self.guidance_unit_vector) > 1 else 0.0
        return struct.pack(
            '<IBBBBffffff',
            ts_ms,
            1 if self.is_high_frequency else 0,
            int(self.attack_pattern),
            1 if self.override_to_start else 0,
            min(255, self.burst_count),
            float(self.attack_frequency_hz),
            float(self.overall_drift_magnitude),
            nega_vx,
            nega_vy,
            guide_x,
            guide_y,
        )


class HighFrequencyAttackDetector:
    """
    High-Frequency Cyber Attack & Negative Drift Engine.
    
    Parameters
    ----------
    freq_threshold_hz : float
        Minimum attack event frequency (Hz) to declare a high-frequency attack (default 2.0 Hz).
    burst_window_s : float
        Sliding time window (seconds) to measure attack burst counts (default 2.0 s).
    burst_count_threshold : int
        Number of attack events in the burst window required to trigger high frequency status (default 3).
    drift_override_threshold_m : float
        Minimum overall drift (meters) from start position before return-to-start override is triggered (default 5.0m).
    max_history_size : int
        Maximum number of attack events to store per sensor (default 50).
    """

    def __init__(
        self,
        freq_threshold_hz: float = 1.0,
        burst_window_s: float = 5.0,
        burst_count_threshold: int = 5,
        drift_override_threshold_m: float = 4.0,
        max_history_size: int = 50,
    ):
        self.freq_threshold_hz = float(freq_threshold_hz)
        self.burst_window_s = float(burst_window_s)
        self.burst_count_threshold = int(burst_count_threshold)
        self.drift_override_threshold_m = float(drift_override_threshold_m)
        self.max_history_size = max(5, int(max_history_size))

        self._start_position: Optional[np.ndarray] = None
        # Attack events per sensor: deque of (timestamp, attack_vector)
        self._attack_events: Dict[str, Deque[Tuple[float, np.ndarray]]] = {}
        # Latched override status per sensor
        self._override_latched: Dict[str, bool] = {}

    def set_start_position(self, start_position: np.ndarray) -> None:
        """Lock in the immutable launch / start location of the mission."""
        self._start_position = np.asarray(start_position, dtype=float).copy()

    def get_start_position(self) -> Optional[np.ndarray]:
        """Return a safe copy of the mission start location."""
        return None if self._start_position is None else self._start_position.copy()

    def reset(self, sensor: Optional[str] = None) -> None:
        """Reset attack history and override states."""
        if sensor is None:
            self._attack_events.clear()
            self._override_latched.clear()
        else:
            if sensor in self._attack_events:
                self._attack_events[sensor].clear()
            self._override_latched[sensor] = False

    def is_override_active(self, sensor: str = "GNSS") -> bool:
        """Return whether return-to-start override is currently active."""
        return self._override_latched.get(sensor, False)

    def calculate_drift(self, current_position: np.ndarray) -> Tuple[np.ndarray, float, np.ndarray]:
        """
        Calculate overall spatial drift from the start position and its negative drift vector.
        
        Returns
        -------
        overall_drift : np.ndarray
            d_overall = current_position - start_position
        magnitude : float
            ||d_overall||
        negative_drift : np.ndarray
            - d_overall (the restorative counter-drift vector)
        """
        if self._start_position is None:
            zero_vec = np.zeros_like(current_position)
            return zero_vec, 0.0, zero_vec

        cur_2d = np.asarray(current_position, dtype=float)[:2]
        start_2d = self._start_position[:2]
        overall_drift = cur_2d - start_2d
        magnitude = float(np.linalg.norm(overall_drift))
        negative_drift = -overall_drift
        return overall_drift, magnitude, negative_drift

    def record_and_evaluate(
        self,
        sensor: str,
        timestamp: float,
        attack_vector: np.ndarray,
        current_position: np.ndarray,
        is_attack_sample: bool = True,
    ) -> HighFrequencySignal:
        """
        Evaluate sensor reading for high-frequency attack characteristics, changing patterns,
        overall drift, and negative drift homing.
        """
        if sensor not in self._attack_events:
            self._attack_events[sensor] = deque(maxlen=self.max_history_size)
            self._override_latched[sensor] = False

        history = self._attack_events[sensor]

        if is_attack_sample:
            raw_vec = np.atleast_1d(np.asarray(attack_vector, dtype=float))
            if len(raw_vec) >= 2:
                vec_2d = raw_vec[:2].copy()
            elif len(raw_vec) == 1:
                vec_2d = np.array([raw_vec[0], 0.0])
            else:
                vec_2d = np.zeros(2)
            history.append((timestamp, vec_2d))

        # 1. Compute burst count within sliding time window (default 5 seconds)
        recent_events = [
            (t, v) for (t, v) in history
            if (timestamp - t) <= self.burst_window_s and (timestamp - t) >= -1e-6
        ]
        burst_count = len(recent_events)

        # 2. Compute attack arrival frequency (Hz)
        if burst_count >= 2:
            time_span = recent_events[-1][0] - recent_events[0][0]
            if time_span > 1e-4:
                attack_freq = (burst_count - 1) / time_span
            else:
                attack_freq = float(burst_count) * 10.0  # Instantaneous burst
        else:
            attack_freq = 0.0

        # High frequency condition: > 5 spoofing events found in 5 seconds (or >= 3 events at >= freq_threshold_hz)
        is_hf = (burst_count > self.burst_count_threshold) or (burst_count >= 3 and attack_freq >= self.freq_threshold_hz)

        # 3. Analyze Changing Attack Patterns
        pattern, pattern_var_deg = self._classify_pattern(recent_events if len(recent_events) >= 3 else list(history))

        # 4. Calculate Overall Drift & Negative Drift ("Nega Drift")
        overall_drift, drift_mag, negative_drift = self.calculate_drift(current_position)

        # 5. Return-to-Start Override Logic
        # If more than 5 spoofing events in 5 seconds (or high-frequency active):
        # Override destination with starting point! Drone goes ONLY towards starting point.
        if self._override_latched[sensor]:
            override_to_start = True
        elif burst_count > self.burst_count_threshold or is_hf:
            override_to_start = True
            self._override_latched[sensor] = True
        elif burst_count >= self.burst_count_threshold and drift_mag >= self.drift_override_threshold_m:
            override_to_start = True
            self._override_latched[sensor] = True
        else:
            override_to_start = False

        # 6. Compute homing guidance unit vector toward start position
        if self._start_position is not None:
            start_target = self._start_position.copy()
            cur_2d = np.asarray(current_position, dtype=float)[:2]
            delta_to_start = start_target[:2] - cur_2d
            dist_to_start = float(np.linalg.norm(delta_to_start))
            if dist_to_start > 1e-6:
                guidance_unit = delta_to_start / dist_to_start
            else:
                guidance_unit = np.zeros(2)
        else:
            start_target = np.zeros(3)
            guidance_unit = np.zeros(2)

        return HighFrequencySignal(
            timestamp=timestamp,
            sensor=sensor,
            is_high_frequency=is_hf,
            attack_frequency_hz=attack_freq,
            burst_count=burst_count,
            attack_pattern=pattern,
            pattern_variance_deg=pattern_var_deg,
            overall_drift_vector=overall_drift,
            overall_drift_magnitude=drift_mag,
            negative_drift_vector=negative_drift,
            override_to_start=override_to_start,
            target_start_location=start_target,
            guidance_unit_vector=guidance_unit,
        )

    def _classify_pattern(self, events: List[Tuple[float, np.ndarray]]) -> Tuple[AttackPattern, float]:
        """Classify the geometric and temporal pattern of incoming attack vectors."""
        if len(events) < 2:
            return AttackPattern.HEALTHY, 0.0

        headings: List[float] = []
        for _, vec in events:
            if len(vec) >= 2:
                norm = np.linalg.norm(vec)
                if norm > 1e-6:
                    ang = math.atan2(vec[1], vec[0])
                    headings.append(ang)

        if len(headings) < 2:
            return AttackPattern.UNIDIRECTIONAL, 0.0

        # Compute angular differences between consecutive attack vectors
        diffs = []
        for i in range(1, len(headings)):
            diff = (headings[i] - headings[i - 1] + math.pi) % (2 * math.pi) - math.pi
            diffs.append(abs(diff))

        mean_diff_deg = math.degrees(float(np.mean(diffs)))
        std_diff_deg = math.degrees(float(np.std(diffs))) if len(diffs) > 1 else 0.0

        # Pattern classification
        if mean_diff_deg < 25.0 and std_diff_deg < 20.0:
            return AttackPattern.UNIDIRECTIONAL, std_diff_deg
        elif 135.0 <= mean_diff_deg <= 180.0:
            return AttackPattern.RAPID_ALTERNATING, std_diff_deg
        elif 60.0 <= mean_diff_deg < 135.0:
            return AttackPattern.OSCILLATING, std_diff_deg
        elif std_diff_deg > 50.0:
            return AttackPattern.ERRATIC_SCATTER, std_diff_deg
        else:
            return AttackPattern.HIGH_FREQ_BURST, std_diff_deg
