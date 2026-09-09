"""
environment_matcher.py — Physical Sensor Consensus Validator
==============================================================
Cross-validates the physical world against the expected route
using onboard camera, LiDAR, and ultrasonic sensor observations.

Even if the destination command passes cryptographic and geofence checks,
this module catches the case where the vehicle is physically driving
toward the wrong location by comparing what the sensors SEE against
what they SHOULD see along the expected route.

Architecture:
    1. Expected Landmark Database: For each route segment, a list of
       expected physical landmarks (buildings, signs, walls, intersections).
    2. Real-Time Matching: As the vehicle drives, onboard sensors detect
       landmarks. The matcher compares detected vs. expected landmarks.
    3. Mismatch Detection: If the physical environment doesn't match
       the expected route, the system flags an anomaly.

Hardware Interface:
    EnvironmentSignal.to_bytes() → 16-byte packet for CAN / UART / SPI.
    EnvironmentSignal.to_dict()  → JSON-serializable dict for ROS2 / TCP / MQTT.

References:
    [3] Model-based security framework
    [4] Camera, LiDAR, ultrasonic 3D mapping
    [9] Manipulated environmental data detection
"""

import struct
import numpy as np
from dataclasses import dataclass, field
from enum import IntEnum
from typing import List, Optional, Dict, Tuple


class EnvironmentVerdict(IntEnum):
    """Environment matching verdict for hardware controllers."""
    MATCH       = 0   # Physical environment matches expected route
    MISMATCH    = 1   # Environment does NOT match — possible hijack
    DEGRADED    = 2   # Not enough landmarks visible to decide
    UNCHECKED   = 3   # No expected landmarks defined for this segment


@dataclass
class ExpectedLandmark:
    """
    A physical landmark expected along a route segment.
    These come from the secure local map database.
    """
    landmark_id: str
    position: np.ndarray          # Expected [x, y, z] in world frame
    landmark_type: str            # "building", "sign", "wall", "intersection"
    min_detection_range: float    # Min range at which sensor should detect it
    max_detection_range: float    # Max range for detection
    bearing_tolerance_rad: float  # Acceptable bearing error (radians)


@dataclass
class DetectedFeature:
    """
    A physical feature detected by onboard sensors.
    Represents what the camera/LiDAR/ultrasonic actually sees.
    """
    position: np.ndarray       # Detected [x, y, z] in world frame
    sensor_source: str         # "CAMERA", "LIDAR", "ULTRASONIC"
    confidence: float          # Detection confidence (0.0 to 1.0)
    timestamp: float


@dataclass
class EnvironmentSignal:
    """
    Output signal from the Environment Matcher.
    Designed for direct transmission to hardware decision units.
    """
    timestamp: float
    verdict: EnvironmentVerdict
    match_score: float              # 0.0 (total mismatch) to 1.0 (perfect match)
    expected_count: int             # Number of landmarks expected in range
    detected_count: int             # Number of landmarks actually detected
    matched_count: int              # Number of expected landmarks matched
    max_position_error: float       # Worst position error across matched landmarks
    reason: str

    def to_dict(self) -> dict:
        """JSON / telemetry serialization."""
        return {
            'timestamp': round(self.timestamp, 4),
            'verdict': self.verdict.name,
            'verdict_code': int(self.verdict),
            'match_score': round(float(self.match_score), 4),
            'expected_count': self.expected_count,
            'detected_count': self.detected_count,
            'matched_count': self.matched_count,
            'max_position_error': round(float(self.max_position_error), 2),
            'reason': self.reason
        }

    def to_bytes(self) -> bytes:
        """
        Compact 16-byte hardware packet:
        [uint32 timestamp_ms, uint8 verdict, uint8 expected,
         uint8 matched, uint8 detected, float32 match_score, float32 max_error]
        """
        ts_ms = int(self.timestamp * 1000) & 0xFFFFFFFF
        return struct.pack('<IBBBBff',
                           ts_ms,
                           int(self.verdict),
                           self.expected_count & 0xFF,
                           self.matched_count & 0xFF,
                           self.detected_count & 0xFF,
                           float(self.match_score),
                           float(self.max_position_error))


class EnvironmentMatcher:
    """
    Environment Matcher — Physical Sensor Consensus Validator.
    
    Compares what the vehicle's sensors actually see against what
    they should see along the expected route. If the physical world
    doesn't match the expected environment, the route has been hijacked.
    
    Parameters
    ----------
    match_threshold : float
        Minimum match score (0.0 to 1.0) required to consider the
        environment as matching. Below this → MISMATCH. Default 0.5.
    position_tolerance_m : float
        Maximum allowed position error (meters) between an expected
        landmark and a detected feature to count as a match. Default 5.0.
    min_landmarks_for_decision : int
        Minimum number of expected landmarks that must be in range
        before the matcher can make a MATCH/MISMATCH decision.
        If fewer are expected, verdict is DEGRADED. Default 2.
    """
    
    def __init__(self,
                 match_threshold: float = 0.5,
                 position_tolerance_m: float = 5.0,
                 min_landmarks_for_decision: int = 2):
        
        self.match_threshold = match_threshold
        self.position_tolerance_m = position_tolerance_m
        self.min_landmarks_for_decision = min_landmarks_for_decision
        
        # Route landmark database: route_id → list of expected landmarks
        self._route_landmarks: Dict[str, List[ExpectedLandmark]] = {}
        self._active_route: Optional[str] = None
        
        # Running match history for trend detection
        self._match_history: List[float] = []
        self._history_window = 30
        
        # Mismatch counter for consecutive mismatches
        self._consecutive_mismatches = 0
        self._mismatch_alarm_threshold = 5
    
    def define_route(self, route_id: str, landmarks: List[ExpectedLandmark]):
        """Define expected landmarks for a named route."""
        self._route_landmarks[route_id] = landmarks
    
    def set_active_route(self, route_id: str):
        """Activate a route for real-time environment matching."""
        if route_id in self._route_landmarks:
            self._active_route = route_id
            self._consecutive_mismatches = 0
            self._match_history.clear()
        else:
            raise ValueError(f"Unknown route '{route_id}'")
    
    def add_landmark_to_route(self, route_id: str, landmark: ExpectedLandmark):
        """Add a single landmark to an existing route."""
        if route_id not in self._route_landmarks:
            self._route_landmarks[route_id] = []
        self._route_landmarks[route_id].append(landmark)
    
    def evaluate(self,
                 vehicle_position: np.ndarray,
                 detected_features: List[DetectedFeature],
                 timestamp: float) -> EnvironmentSignal:
        """
        Evaluate whether the physical environment matches the expected route.
        
        Parameters
        ----------
        vehicle_position : np.ndarray
            Current vehicle position [x, y, z].
        detected_features : List[DetectedFeature]
            Features currently detected by onboard sensors.
        timestamp : float
            Current time.
        
        Returns
        -------
        EnvironmentSignal with MATCH, MISMATCH, DEGRADED, or UNCHECKED verdict.
        """
        if self._active_route is None or self._active_route not in self._route_landmarks:
            return EnvironmentSignal(
                timestamp=timestamp,
                verdict=EnvironmentVerdict.UNCHECKED,
                match_score=0.0,
                expected_count=0,
                detected_count=len(detected_features),
                matched_count=0,
                max_position_error=0.0,
                reason="No active route defined for environment matching"
            )
        
        landmarks = self._route_landmarks[self._active_route]
        vpos = np.atleast_1d(vehicle_position).astype(float)
        
        # Find expected landmarks within detection range of current position
        expected_in_range = []
        for lm in landmarks:
            dist = np.linalg.norm(vpos[:2] - lm.position[:2])
            if lm.min_detection_range <= dist <= lm.max_detection_range:
                expected_in_range.append(lm)
        
        if len(expected_in_range) < self.min_landmarks_for_decision:
            return EnvironmentSignal(
                timestamp=timestamp,
                verdict=EnvironmentVerdict.DEGRADED,
                match_score=0.5,
                expected_count=len(expected_in_range),
                detected_count=len(detected_features),
                matched_count=0,
                max_position_error=0.0,
                reason=f"Only {len(expected_in_range)} landmarks in range "
                       f"(need {self.min_landmarks_for_decision})"
            )
        
        # Match detected features to expected landmarks
        matched = 0
        max_error = 0.0
        used_detections = set()
        
        for lm in expected_in_range:
            best_match_dist = float('inf')
            best_match_idx = -1
            
            for i, feat in enumerate(detected_features):
                if i in used_detections:
                    continue
                dist = np.linalg.norm(lm.position[:2] - feat.position[:2])
                if dist < best_match_dist:
                    best_match_dist = dist
                    best_match_idx = i
            
            if best_match_idx >= 0 and best_match_dist <= self.position_tolerance_m:
                matched += 1
                used_detections.add(best_match_idx)
                max_error = max(max_error, best_match_dist)
        
        # Calculate match score
        match_score = matched / max(len(expected_in_range), 1)
        
        # Update history
        self._match_history.append(match_score)
        if len(self._match_history) > self._history_window:
            self._match_history.pop(0)
        
        # Determine verdict
        if match_score >= self.match_threshold:
            self._consecutive_mismatches = 0
            verdict = EnvironmentVerdict.MATCH
            reason = f"Environment matches: {matched}/{len(expected_in_range)} landmarks confirmed"
        else:
            self._consecutive_mismatches += 1
            verdict = EnvironmentVerdict.MISMATCH
            reason = (f"ENVIRONMENT MISMATCH: only {matched}/{len(expected_in_range)} "
                      f"landmarks match (score {match_score:.2f} < threshold {self.match_threshold}). "
                      f"Consecutive mismatches: {self._consecutive_mismatches}")
        
        return EnvironmentSignal(
            timestamp=timestamp,
            verdict=verdict,
            match_score=match_score,
            expected_count=len(expected_in_range),
            detected_count=len(detected_features),
            matched_count=matched,
            max_position_error=max_error,
            reason=reason
        )
    
    @property
    def consecutive_mismatches(self) -> int:
        return self._consecutive_mismatches
    
    @property
    def is_alarm(self) -> bool:
        """True if consecutive mismatches exceed the alarm threshold."""
        return self._consecutive_mismatches >= self._mismatch_alarm_threshold
    
    @property
    def average_match_score(self) -> float:
        """Rolling average match score."""
        if not self._match_history:
            return 1.0
        return float(np.mean(self._match_history))
