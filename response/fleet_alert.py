"""
fleet_alert.py — High-Priority Alert Broadcast to Fleet Command Center
========================================================================
Assembles and serializes fleet-level alert messages when SensorSentry
detects a confirmed attack on any vehicle in the fleet.

Output Formats:
    - JSON   (for MQTT / TCP / WebSocket / REST API)
    - Binary (for CAN bus / satellite uplink / LoRa)

Alert Contents:
    - Vehicle ID
    - Attack type (GPS_SPOOF / WAYPOINT_HIJACK)
    - GPS last-known-good position
    - Timestamp (UTC)
    - Severity level
    - Recommended action
    - Sensor evidence summary

References:
    [9] Fleet-wide alert broadcasting
"""

import struct
import json
import time
import numpy as np
from dataclasses import dataclass
from enum import IntEnum
from typing import Optional, List, Dict


class AlertSeverity(IntEnum):
    """Alert severity levels for fleet command center."""
    INFO     = 0   # Informational — no action required
    WARNING  = 1   # Possible threat — monitor closely
    CRITICAL = 2   # Confirmed attack — immediate attention
    EMERGENCY = 3  # Vehicle under active attack — dispatch response


class AttackType(IntEnum):
    """Classification of detected attack vector."""
    NONE            = 0
    GPS_SPOOF       = 1   # Scenario A: Fake satellite signals
    WAYPOINT_HIJACK = 2   # Scenario B: Destination code tampering
    COMBINED        = 3   # Both attacks simultaneously
    UNKNOWN         = 4   # Unclassified anomaly


@dataclass
class FleetAlert:
    """
    Fleet-level alert message for the Command Center.
    """
    alert_id: str                    # Unique alert identifier
    timestamp: float                 # Alert generation time (UTC seconds)
    vehicle_id: str                  # Vehicle / robot identifier
    attack_type: AttackType
    severity: AlertSeverity
    position: Optional[np.ndarray]   # Last known good GPS position [x, y, z]
    heading: float                   # Vehicle heading at time of alert (rad)
    speed: float                     # Vehicle speed at time of alert (m/s)
    evidence: Dict[str, str]         # Key evidence summary
    recommended_action: str          # Human-readable action recommendation
    is_resolved: bool = False        # Has the alert been resolved?

    def to_dict(self) -> dict:
        """JSON serialization for MQTT / TCP / WebSocket."""
        result = {
            'alert_id': self.alert_id,
            'timestamp': round(self.timestamp, 4),
            'timestamp_iso': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime(self.timestamp)),
            'vehicle_id': self.vehicle_id,
            'attack_type': self.attack_type.name,
            'attack_code': int(self.attack_type),
            'severity': self.severity.name,
            'severity_code': int(self.severity),
            'heading_deg': round(float(self.heading) * 180.0 / np.pi, 1),
            'speed_mps': round(float(self.speed), 2),
            'evidence': self.evidence,
            'recommended_action': self.recommended_action,
            'is_resolved': self.is_resolved
        }
        if self.position is not None:
            result['position'] = {
                'x': round(float(self.position[0]), 4),
                'y': round(float(self.position[1]), 4),
                'z': round(float(self.position[2]), 4) if len(self.position) > 2 else 0.0
            }
        return result

    def to_json(self) -> str:
        """JSON string for network transmission."""
        return json.dumps(self.to_dict(), indent=2)

    def to_bytes(self) -> bytes:
        """
        Compact 32-byte binary packet for CAN / satellite uplink:
        [uint32 timestamp_ms, uint8 attack_type, uint8 severity,
         uint8 resolved, uint8 reserved,
         float32 pos_x, float32 pos_y, float32 pos_z,
         float32 heading, float32 speed, uint32 alert_hash]
        """
        ts_ms = int(self.timestamp * 1000) & 0xFFFFFFFF
        pos = self.position if self.position is not None else np.zeros(3)
        # Simple hash of alert_id for tracking
        alert_hash = hash(self.alert_id) & 0xFFFFFFFF
        
        return struct.pack('<IBBBBfffffI',
                           ts_ms,
                           int(self.attack_type),
                           int(self.severity),
                           1 if self.is_resolved else 0,
                           0,
                           float(pos[0]),
                           float(pos[1]),
                           float(pos[2]) if len(pos) > 2 else 0.0,
                           float(self.heading),
                           float(self.speed),
                           alert_hash)


class FleetAlertBroadcaster:
    """
    Fleet Alert Broadcaster — Assembles and manages alert messages
    for the Fleet Security Command Center.
    
    Parameters
    ----------
    vehicle_id : str
        Unique identifier for this vehicle / robot.
    """
    
    def __init__(self, vehicle_id: str = "VEHICLE-001"):
        self.vehicle_id = vehicle_id
        self._alert_counter = 0
        self._active_alerts: List[FleetAlert] = []
        self._alert_history: List[FleetAlert] = []
    
    def create_gps_spoof_alert(self,
                                timestamp: float,
                                position: np.ndarray,
                                heading: float,
                                speed: float,
                                cusum_value: float,
                                residual: float) -> FleetAlert:
        """
        Create a GPS Spoofing (Scenario A) alert.
        """
        self._alert_counter += 1
        alert_id = f"SPOOF-{self.vehicle_id}-{self._alert_counter:04d}"
        
        alert = FleetAlert(
            alert_id=alert_id,
            timestamp=timestamp,
            vehicle_id=self.vehicle_id,
            attack_type=AttackType.GPS_SPOOF,
            severity=AlertSeverity.CRITICAL,
            position=np.array(position, dtype=float),
            heading=heading,
            speed=speed,
            evidence={
                'cusum_level': f"{cusum_value:.2f}",
                'gps_residual': f"{residual:.2f}",
                'detection_method': 'EKF + CUSUM drift detection',
                'mitigation': 'GPS quarantined, SAARM dead-reckoning active'
            },
            recommended_action=(
                "GPS SPOOFING CONFIRMED. Vehicle navigating via inertial dead-reckoning. "
                "Dispatch security to last known good position. "
                "Investigate RF interference source in the area."
            )
        )
        
        self._active_alerts.append(alert)
        self._alert_history.append(alert)
        return alert
    
    def create_waypoint_hijack_alert(self,
                                      timestamp: float,
                                      position: np.ndarray,
                                      heading: float,
                                      speed: float,
                                      original_dest: np.ndarray,
                                      tampered_dest: np.ndarray,
                                      rejection_reason: str) -> FleetAlert:
        """
        Create a Waypoint Hijack (Scenario B) alert.
        """
        self._alert_counter += 1
        alert_id = f"HIJACK-{self.vehicle_id}-{self._alert_counter:04d}"
        
        alert = FleetAlert(
            alert_id=alert_id,
            timestamp=timestamp,
            vehicle_id=self.vehicle_id,
            attack_type=AttackType.WAYPOINT_HIJACK,
            severity=AlertSeverity.EMERGENCY,
            position=np.array(position, dtype=float),
            heading=heading,
            speed=speed,
            evidence={
                'original_destination': f"[{original_dest[0]:.1f}, {original_dest[1]:.1f}]",
                'tampered_destination': f"[{tampered_dest[0]:.1f}, {tampered_dest[1]:.1f}]",
                'rejection_reason': rejection_reason,
                'detection_method': 'Geofence + Command signature + Environment mismatch',
                'mitigation': 'Safe stop executed, vehicle brakes locked'
            },
            recommended_action=(
                "WAYPOINT HIJACK DETECTED. Vehicle safe-stopped at current position. "
                "Mission planner software compromised — initiate forensic investigation. "
                "Do NOT send new waypoints until system integrity verified."
            )
        )
        
        self._active_alerts.append(alert)
        self._alert_history.append(alert)
        return alert
    
    def resolve_alert(self, alert_id: str):
        """Mark an alert as resolved."""
        for alert in self._active_alerts:
            if alert.alert_id == alert_id:
                alert.is_resolved = True
                self._active_alerts.remove(alert)
                break
    
    @property
    def active_alert_count(self) -> int:
        return len(self._active_alerts)
    
    @property
    def all_alerts(self) -> List[FleetAlert]:
        return list(self._alert_history)
    
    def get_active_alerts_json(self) -> str:
        """Get all active alerts as JSON array."""
        return json.dumps([a.to_dict() for a in self._active_alerts], indent=2)
