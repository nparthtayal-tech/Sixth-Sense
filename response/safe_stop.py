"""
safe_stop.py — Emergency Vehicle Immobilization Protocol
==========================================================
State machine that executes immediate brake lockdown when
an attack is confirmed by SensorSentry.

State Machine:
    NOMINAL → ALERT → SAFE_STOP → LOCKDOWN

    NOMINAL:     Normal operation. All systems green.
    ALERT:       Attack suspected. Warning to driver / fleet.
                 Vehicle continues but at reduced authority.
    SAFE_STOP:   Attack confirmed. Brakes engaged immediately.
                 Steering locked. Engine/motor power cut.
    LOCKDOWN:    Vehicle fully stopped and secured. Awaiting
                 manual override or fleet command center clearance.

Hardware Interface:
    SafeStopSignal.to_bytes() → 20-byte packet for CAN / UART / SPI.
    SafeStopSignal.to_dict()  → JSON-serializable dict for ROS2 / TCP / MQTT.

CAN Bus Brake Commands:
    0x00 = Brakes released (normal)
    0x7F = Moderate braking (controlled stop)
    0xFF = Maximum braking (emergency full stop)

References:
    [9] Safe stop protocol upon detection of manipulated data
"""

import struct
import time
import numpy as np
from dataclasses import dataclass
from enum import IntEnum
from typing import Optional, List


class SafeStopState(IntEnum):
    """Vehicle safety state machine levels."""
    NOMINAL   = 0   # All clear — normal operation
    ALERT     = 1   # Threat detected — reduced authority
    SAFE_STOP = 2   # Attack confirmed — brakes engaging
    LOCKDOWN  = 3   # Vehicle secured — awaiting clearance


@dataclass
class SafeStopSignal:
    """
    Output signal from the Safe Stop Controller.
    Directly drives hardware brake and motor controllers.
    """
    timestamp: float
    state: SafeStopState
    brake_command: int             # 0x00 to 0xFF (brake pressure level)
    motor_kill: bool               # True = cut engine/motor power
    steering_lock: bool            # True = lock steering to current angle
    attack_type: str               # "GPS_SPOOF", "WAYPOINT_HIJACK", "UNKNOWN"
    last_known_good_position: Optional[np.ndarray]  # Last trusted GPS position
    reason: str

    def to_dict(self) -> dict:
        """JSON / telemetry serialization."""
        result = {
            'timestamp': round(self.timestamp, 4),
            'state': self.state.name,
            'state_code': int(self.state),
            'brake_command': self.brake_command,
            'brake_hex': f"0x{self.brake_command:02X}",
            'motor_kill': self.motor_kill,
            'steering_lock': self.steering_lock,
            'attack_type': self.attack_type,
            'reason': self.reason
        }
        if self.last_known_good_position is not None:
            result['last_known_good_position'] = [
                round(float(v), 4) for v in self.last_known_good_position
            ]
        return result

    def to_bytes(self) -> bytes:
        """
        Compact 20-byte hardware packet:
        [uint32 timestamp_ms, uint8 state, uint8 brake_cmd,
         uint8 motor_kill, uint8 steering_lock,
         float32 pos_x, float32 pos_y, float32 pos_z]
        """
        ts_ms = int(self.timestamp * 1000) & 0xFFFFFFFF
        pos = self.last_known_good_position if self.last_known_good_position is not None else np.zeros(3)
        return struct.pack('<IBBBBfff',
                           ts_ms,
                           int(self.state),
                           self.brake_command & 0xFF,
                           1 if self.motor_kill else 0,
                           1 if self.steering_lock else 0,
                           float(pos[0]),
                           float(pos[1]),
                           float(pos[2]))


class SafeStopController:
    """
    Safe Stop Controller — Emergency Immobilization Protocol.
    
    Manages the vehicle safety state machine. When SensorSentry
    confirms an attack, this controller immediately:
    1. Engages maximum braking (0xFF)
    2. Kills engine/motor power
    3. Locks steering to current angle
    4. Broadcasts position and alert to fleet
    
    Parameters
    ----------
    alert_to_stop_delay_s : float
        Time delay (seconds) between ALERT and SAFE_STOP states.
        Set to 0.0 for immediate stop on first confirmation.
        Set > 0 to allow a brief warning period.
    require_clearance_for_resume : bool
        If True, vehicle cannot leave LOCKDOWN without explicit
        clearance signal (from fleet command center).
    """
    
    def __init__(self,
                 alert_to_stop_delay_s: float = 0.0,
                 require_clearance_for_resume: bool = True):
        
        self.alert_to_stop_delay = alert_to_stop_delay_s
        self.require_clearance = require_clearance_for_resume
        
        # State machine
        self._state = SafeStopState.NOMINAL
        self._alert_start_time: Optional[float] = None
        self._stop_time: Optional[float] = None
        self._attack_type = "UNKNOWN"
        self._reason = ""
        
        # Last known good position (before attack)
        self._last_good_position: Optional[np.ndarray] = None
        
        # Event log
        self._event_log: List[dict] = []
    
    def update_good_position(self, position: np.ndarray):
        """
        Update the last-known-good GPS position.
        Called continuously during NOMINAL operation.
        Freezes when attack is detected (preserves pre-attack location).
        """
        if self._state == SafeStopState.NOMINAL:
            self._last_good_position = np.array(position, dtype=float).copy()
    
    def trigger_alert(self, timestamp: float, attack_type: str, reason: str) -> SafeStopSignal:
        """
        Escalate to ALERT state. Called when an attack is suspected
        but not yet fully confirmed.
        """
        if self._state == SafeStopState.NOMINAL:
            self._state = SafeStopState.ALERT
            self._alert_start_time = timestamp
            self._attack_type = attack_type
            self._reason = reason
            self._log_event(timestamp, "ALERT", reason)
        
        # Check if we should auto-escalate to SAFE_STOP
        if (self._state == SafeStopState.ALERT and
                self.alert_to_stop_delay <= 0.0):
            return self.trigger_safe_stop(timestamp, attack_type, reason)
        
        return self._build_signal(timestamp)
    
    def trigger_safe_stop(self, timestamp: float, attack_type: str, reason: str) -> SafeStopSignal:
        """
        Escalate to SAFE_STOP state. Called when attack is CONFIRMED.
        Immediately engages emergency braking.
        """
        if self._state.value < SafeStopState.SAFE_STOP.value:
            self._state = SafeStopState.SAFE_STOP
            self._stop_time = timestamp
            self._attack_type = attack_type
            self._reason = reason
            self._log_event(timestamp, "SAFE_STOP", reason)
        
        return self._build_signal(timestamp)
    
    def trigger_lockdown(self, timestamp: float) -> SafeStopSignal:
        """
        Escalate to LOCKDOWN. Vehicle has fully stopped.
        Awaiting clearance from fleet command center.
        """
        if self._state.value < SafeStopState.LOCKDOWN.value:
            self._state = SafeStopState.LOCKDOWN
            self._log_event(timestamp, "LOCKDOWN", "Vehicle secured. Awaiting clearance.")
        
        return self._build_signal(timestamp)
    
    def clear_and_resume(self, timestamp: float, authorization_code: str = "") -> SafeStopSignal:
        """
        Return to NOMINAL state after fleet command center clearance.
        
        Parameters
        ----------
        authorization_code : str
            Fleet command center authorization code (optional).
        """
        if self.require_clearance and not authorization_code:
            self._log_event(timestamp, "RESUME_DENIED", "No authorization code provided")
            return self._build_signal(timestamp)
        
        self._state = SafeStopState.NOMINAL
        self._alert_start_time = None
        self._stop_time = None
        self._attack_type = "UNKNOWN"
        self._reason = ""
        self._log_event(timestamp, "RESUMED", f"Cleared with auth: {authorization_code[:8]}...")
        
        return self._build_signal(timestamp)
    
    def get_status(self, timestamp: float) -> SafeStopSignal:
        """Get current safe stop status without changing state."""
        return self._build_signal(timestamp)
    
    def _build_signal(self, timestamp: float) -> SafeStopSignal:
        """Build output signal based on current state."""
        if self._state == SafeStopState.NOMINAL:
            brake_cmd = 0x00
            motor_kill = False
            steering_lock = False
        elif self._state == SafeStopState.ALERT:
            brake_cmd = 0x40  # Light braking
            motor_kill = False
            steering_lock = False
        elif self._state == SafeStopState.SAFE_STOP:
            brake_cmd = 0xFF  # MAXIMUM BRAKING
            motor_kill = True
            steering_lock = True
        else:  # LOCKDOWN
            brake_cmd = 0xFF
            motor_kill = True
            steering_lock = True
        
        return SafeStopSignal(
            timestamp=timestamp,
            state=self._state,
            brake_command=brake_cmd,
            motor_kill=motor_kill,
            steering_lock=steering_lock,
            attack_type=self._attack_type,
            last_known_good_position=self._last_good_position,
            reason=self._reason
        )
    
    def _log_event(self, timestamp: float, event: str, detail: str):
        """Record event to internal log."""
        self._event_log.append({
            'timestamp': round(timestamp, 4),
            'event': event,
            'detail': detail,
            'attack_type': self._attack_type
        })
    
    @property
    def state(self) -> SafeStopState:
        return self._state
    
    @property
    def is_stopped(self) -> bool:
        return self._state.value >= SafeStopState.SAFE_STOP.value
    
    @property
    def event_log(self) -> List[dict]:
        return list(self._event_log)
