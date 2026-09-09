"""
command_validator.py — Cryptographic Waypoint Command Integrity
================================================================
Validates destination change commands using multiple defense layers:

1. HMAC-SHA256 Signing    — Every waypoint command must carry a valid
                            cryptographic signature. Unsigned or tampered
                            commands are rejected immediately.
2. Sequence Numbering     — Monotonically increasing sequence numbers
                            prevent replay attacks (re-sending old valid
                            commands to redirect the vehicle).
3. Rate Limiting          — No more than N destination changes per minute.
                            Rapid-fire waypoint spam is a DoS indicator.
4. Spatial Continuity     — New destination can't jump more than X km from
                            current position without elevated authorization.
                            Prevents teleportation attacks.
5. Temporal Freshness     — Commands older than T seconds are stale and
                            rejected. Prevents delayed injection.

Hardware Interface:
    CommandSignal.to_bytes() → 20-byte packet for CAN / UART / SPI.
    CommandSignal.to_dict()  → JSON-serializable dict for ROS2 / TCP / MQTT.

References:
    [10] Cryptographic & geometric handshake
"""

import struct
import hashlib
import hmac
import time
import numpy as np
from dataclasses import dataclass
from enum import IntEnum
from typing import Optional
from collections import deque


class CommandVerdict(IntEnum):
    """Waypoint command validation verdict for hardware controllers."""
    ACCEPTED       = 0   # Command is authentic and valid
    BAD_SIGNATURE  = 1   # HMAC signature verification failed
    REPLAY_ATTACK  = 2   # Sequence number already used or out of order
    RATE_LIMITED   = 3   # Too many destination changes in short period
    SPATIAL_JUMP   = 4   # Destination too far from current position
    STALE_COMMAND  = 5   # Command timestamp is too old
    INVALID_FORMAT = 6   # Command structure is malformed


@dataclass
class WaypointCommand:
    """
    A waypoint change command as received from the mission planner.
    In production, this arrives via encrypted CAN / TCP / MQTT.
    """
    destination: np.ndarray     # Target coordinates [x, y] or [x, y, z]
    sequence_number: int        # Monotonic sequence counter
    timestamp: float            # Command creation timestamp
    issuer_id: str              # Identifier of the command issuer
    signature: bytes            # HMAC-SHA256 signature
    priority: int = 0           # 0=normal, 1=urgent, 2=emergency


@dataclass
class CommandSignal:
    """
    Output signal from the Command Validator.
    Designed for direct transmission to hardware decision units.
    """
    timestamp: float
    verdict: CommandVerdict
    sequence_number: int
    issuer_id: str
    destination: np.ndarray
    reason: str
    spatial_distance: float      # Distance from current pos to new dest

    def to_dict(self) -> dict:
        """JSON / telemetry serialization."""
        return {
            'timestamp': round(self.timestamp, 4),
            'verdict': self.verdict.name,
            'verdict_code': int(self.verdict),
            'sequence_number': self.sequence_number,
            'issuer_id': self.issuer_id,
            'destination': [round(float(v), 4) for v in self.destination],
            'reason': self.reason,
            'spatial_distance': round(float(self.spatial_distance), 2)
        }

    def to_bytes(self) -> bytes:
        """
        Compact 20-byte hardware packet:
        [uint32 timestamp_ms, uint8 verdict, uint8 priority,
         uint16 sequence, float32 dest_x, float32 dest_y, float32 distance]
        """
        ts_ms = int(self.timestamp * 1000) & 0xFFFFFFFF
        dest = np.atleast_1d(self.destination)
        return struct.pack('<IBBHfff',
                           ts_ms,
                           int(self.verdict),
                           0,
                           self.sequence_number & 0xFFFF,
                           float(dest[0]),
                           float(dest[1]) if len(dest) > 1 else 0.0,
                           float(self.spatial_distance))


class CommandValidator:
    """
    Cryptographic Waypoint Command Validator.
    
    Multi-layer defense against destination code tampering (Scenario B).
    Every waypoint change must pass ALL checks to be accepted.
    
    Parameters
    ----------
    hmac_key : bytes
        Secret key for HMAC-SHA256 command verification.
    max_commands_per_minute : int
        Rate limit: maximum destination changes per 60-second window.
    max_spatial_jump_m : float
        Maximum allowed distance (meters) between current position and
        new destination without elevated authorization.
    max_command_age_s : float
        Maximum age (seconds) of a command before it is considered stale.
    """
    
    def __init__(self,
                 hmac_key: bytes = b'sensorsentry-command-key',
                 max_commands_per_minute: int = 5,
                 max_spatial_jump_m: float = 50000.0,
                 max_command_age_s: float = 30.0):
        
        self._hmac_key = hmac_key
        self._max_rate = max_commands_per_minute
        self._max_jump = max_spatial_jump_m
        self._max_age = max_command_age_s
        
        # Replay protection
        self._last_sequence = -1
        self._used_sequences: set = set()
        
        # Rate limiting (sliding window of command timestamps)
        self._command_times: deque = deque(maxlen=100)
        
        # Counters
        self.total_validated = 0
        self.total_rejected = 0
        self._rejection_counts: dict = {}
    
    def sign_command(self, destination: np.ndarray, sequence_number: int,
                     timestamp: float, issuer_id: str) -> bytes:
        """
        Generate a valid HMAC-SHA256 signature for a waypoint command.
        Used by authorized mission planners to sign commands.
        
        Parameters
        ----------
        destination : np.ndarray
            Target coordinates.
        sequence_number : int
            Monotonic sequence counter.
        timestamp : float
            Command creation timestamp.
        issuer_id : str
            Identifier of the command issuer.
        
        Returns
        -------
        bytes : HMAC-SHA256 signature.
        """
        payload = self._build_payload(destination, sequence_number, timestamp, issuer_id)
        return hmac.new(self._hmac_key, payload, hashlib.sha256).digest()
    
    def validate(self,
                 command: WaypointCommand,
                 current_position: np.ndarray,
                 current_time: float) -> CommandSignal:
        """
        Validate a waypoint change command through all defense layers.
        
        Parameters
        ----------
        command : WaypointCommand
            The command to validate.
        current_position : np.ndarray
            Vehicle's current position [x, y, z].
        current_time : float
            Current system time.
        
        Returns
        -------
        CommandSignal with ACCEPTED or rejection verdict.
        """
        self.total_validated += 1
        dest = np.atleast_1d(command.destination).astype(float)
        cur_pos = np.atleast_1d(current_position).astype(float)
        
        # Calculate spatial distance
        spatial_dist = float(np.linalg.norm(dest[:2] - cur_pos[:2]))
        
        # Layer 1: HMAC Signature Verification
        expected_payload = self._build_payload(
            command.destination, command.sequence_number,
            command.timestamp, command.issuer_id
        )
        expected_sig = hmac.new(self._hmac_key, expected_payload, hashlib.sha256).digest()
        
        if not hmac.compare_digest(command.signature, expected_sig):
            return self._reject(CommandVerdict.BAD_SIGNATURE, command, dest,
                                spatial_dist, current_time,
                                "HMAC signature verification FAILED — command tampered or unsigned")
        
        # Layer 2: Sequence Number (Replay Attack Prevention)
        if command.sequence_number <= self._last_sequence:
            return self._reject(CommandVerdict.REPLAY_ATTACK, command, dest,
                                spatial_dist, current_time,
                                f"Sequence {command.sequence_number} <= last accepted "
                                f"{self._last_sequence} — possible replay attack")
        
        if command.sequence_number in self._used_sequences:
            return self._reject(CommandVerdict.REPLAY_ATTACK, command, dest,
                                spatial_dist, current_time,
                                f"Sequence {command.sequence_number} already used — replay attack")
        
        # Layer 3: Rate Limiting
        while self._command_times and current_time - self._command_times[0] >= 60.0:
            self._command_times.popleft()
        recent_count = len(self._command_times)
        if recent_count >= self._max_rate:
            return self._reject(CommandVerdict.RATE_LIMITED, command, dest,
                                spatial_dist, current_time,
                                f"{recent_count} commands in last 60s exceeds limit of {self._max_rate}")
        
        # Layer 4: Spatial Continuity
        if spatial_dist > self._max_jump and command.priority < 2:
            return self._reject(CommandVerdict.SPATIAL_JUMP, command, dest,
                                spatial_dist, current_time,
                                f"Destination {spatial_dist:.0f}m from current position "
                                f"exceeds {self._max_jump:.0f}m limit")
        
        # Layer 5: Temporal Freshness
        command_age = abs(current_time - command.timestamp)
        if command_age > self._max_age:
            return self._reject(CommandVerdict.STALE_COMMAND, command, dest,
                                spatial_dist, current_time,
                                f"Command age {command_age:.1f}s exceeds {self._max_age:.1f}s freshness limit")
        
        # ALL CHECKS PASSED — Accept
        self._last_sequence = command.sequence_number
        self._used_sequences.add(command.sequence_number)
        self._command_times.append(current_time)
        
        return CommandSignal(
            timestamp=current_time,
            verdict=CommandVerdict.ACCEPTED,
            sequence_number=command.sequence_number,
            issuer_id=command.issuer_id,
            destination=dest,
            reason="All validation layers passed — command authenticated",
            spatial_distance=spatial_dist
        )
    
    def _reject(self, verdict: CommandVerdict, command: WaypointCommand,
                dest: np.ndarray, spatial_dist: float,
                current_time: float, reason: str) -> CommandSignal:
        """Record a rejection and return the signal."""
        self.total_rejected += 1
        vname = verdict.name
        self._rejection_counts[vname] = self._rejection_counts.get(vname, 0) + 1
        
        return CommandSignal(
            timestamp=current_time,
            verdict=verdict,
            sequence_number=command.sequence_number,
            issuer_id=command.issuer_id,
            destination=dest,
            reason=reason,
            spatial_distance=spatial_dist
        )
    
    @staticmethod
    def _build_payload(destination, sequence_number, timestamp, issuer_id) -> bytes:
        """Build the canonical byte payload for HMAC signing."""
        dest = np.atleast_1d(destination).astype(float)
        parts = [
            struct.pack('<d', float(dest[0])),
            struct.pack('<d', float(dest[1]) if len(dest) > 1 else 0.0),
            struct.pack('<I', int(sequence_number)),
            struct.pack('<d', float(timestamp)),
            issuer_id.encode('utf-8')
        ]
        return b''.join(parts)
    
    @property
    def rejection_summary(self) -> dict:
        """Summary of rejections by type."""
        return dict(self._rejection_counts)
