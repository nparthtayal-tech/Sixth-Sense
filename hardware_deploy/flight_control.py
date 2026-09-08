"""Flight-controller contingency interface.

SensorSentry is advisory by default.  A MAVLink command is emitted only when
the process is explicitly armed with ``--enable-flight-actions`` and an
authenticated, connected flight-control link is available.
"""

from __future__ import annotations

import json
import os
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum
from typing import Optional


class ContingencyAction(str, Enum):
    HOLD = "HOLD"
    RETURN_TO_LAUNCH = "RETURN_TO_LAUNCH"
    LAND = "LAND"


@dataclass(frozen=True)
class ContingencyCommand:
    timestamp: float
    action: ContingencyAction
    reason: str

    def to_bytes(self) -> bytes:
        return json.dumps({
            "schema_version": 1,
            "timestamp": round(self.timestamp, 6),
            "action": self.action.value,
            "reason": self.reason,
        }, sort_keys=True, separators=(",", ":")).encode("utf-8")


class FlightControl(ABC):
    @abstractmethod
    def request(self, action: ContingencyAction, reason: str) -> bool:
        """Request a contingency action. Return True only after controller ACK."""

    def close(self) -> None:
        """Close an optional transport."""


class AdvisoryFlightControl(FlightControl):
    """Safe default: records a request but never controls an aircraft."""

    def __init__(self) -> None:
        self.last_command: Optional[ContingencyCommand] = None

    def request(self, action: ContingencyAction, reason: str) -> bool:
        self.last_command = ContingencyCommand(time.time(), action, reason)
        return False


class MavlinkFlightControl(FlightControl):
    """Minimal command-and-ACK MAVLink contingency client.

    The aircraft's native failsafe and mode policy remain authoritative.  This
    adapter never arms, disarms, or writes mission items; it can request only
    HOLD, RTL, or LAND from a preconfigured autopilot.
    """

    MAVLINK_EPOCH_UNIX_S = 1420070400

    def __init__(self, connection_string: str, signing_key: bytes, signing_state_path: str,
                 heartbeat_timeout_s: float = 10.0, command_timeout_s: float = 3.0):
        if len(signing_key) != 32:
            raise ValueError("MAVLink signing key must be exactly 32 bytes")
        if not signing_state_path:
            raise ValueError("MAVLink signing state path is required")
        self.connection_string = connection_string
        self.signing_key = signing_key
        self.signing_state_path = signing_state_path
        self.heartbeat_timeout_s = heartbeat_timeout_s
        self.command_timeout_s = command_timeout_s
        self._master = None

    def connect(self) -> None:
        try:
            from pymavlink import mavutil
        except ImportError as exc:
            raise RuntimeError("pymavlink is required for --flight-link") from exc
        self._master = mavutil.mavlink_connection(
            self.connection_string, autoreconnect=False, force_mavlink2=True
        )
        # Reject every unsigned or incorrectly signed inbound frame. The runner
        # persists the outbound timestamp on clean shutdown; the host still
        # needs RTC/GNSS time synchronization for crash-recovery safety.
        self._master.setup_signing(
            self.signing_key,
            sign_outgoing=True,
            allow_unsigned_callback=lambda _message_id: False,
            initial_timestamp=max(self._current_signing_timestamp(), self._load_signing_timestamp()),
        )
        heartbeat = self._master.wait_heartbeat(timeout=self.heartbeat_timeout_s)
        if heartbeat is None:
            self.close()
            raise RuntimeError("no MAVLink heartbeat received; refusing to start flight actions")

    def request(self, action: ContingencyAction, reason: str) -> bool:
        if self._master is None:
            return False
        from pymavlink import mavutil
        command = {
            ContingencyAction.HOLD: mavutil.mavlink.MAV_CMD_NAV_LOITER_UNLIM,
            ContingencyAction.RETURN_TO_LAUNCH: mavutil.mavlink.MAV_CMD_NAV_RETURN_TO_LAUNCH,
            ContingencyAction.LAND: mavutil.mavlink.MAV_CMD_NAV_LAND,
        }[action]
        self._master.mav.command_long_send(
            self._master.target_system,
            self._master.target_component,
            command,
            0,
            0, 0, 0, 0, 0, 0, 0,
        )
        deadline = time.monotonic() + self.command_timeout_s
        while time.monotonic() < deadline:
            ack = self._master.recv_match(type="COMMAND_ACK", blocking=True, timeout=0.25)
            if ack is None or getattr(ack, "command", None) != command:
                continue
            return getattr(ack, "result", None) == mavutil.mavlink.MAV_RESULT_ACCEPTED
        return False

    def close(self) -> None:
        if self._master is not None:
            try:
                self._store_signing_timestamp(int(self._master.mav.signing.timestamp))
                self._master.close()
            finally:
                self._master = None

    @classmethod
    def _current_signing_timestamp(cls) -> int:
        """MAVLink timestamp: 10-microsecond ticks since 2015-01-01 UTC."""
        return max(0, int((time.time() - cls.MAVLINK_EPOCH_UNIX_S) * 100_000))

    def _load_signing_timestamp(self) -> int:
        try:
            with open(self.signing_state_path, "rt", encoding="ascii") as state_file:
                value = int(state_file.read().strip())
            if value < 0:
                raise ValueError("negative signing timestamp")
            return value
        except FileNotFoundError:
            return 0
        except (OSError, ValueError) as exc:
            raise RuntimeError(f"cannot load MAVLink signing state: {exc}") from exc

    def _store_signing_timestamp(self, timestamp: int) -> None:
        directory = os.path.dirname(os.path.abspath(self.signing_state_path))
        if not os.path.isdir(directory):
            raise RuntimeError(f"MAVLink signing-state directory does not exist: {directory}")
        temporary_path = f"{self.signing_state_path}.tmp"
        try:
            descriptor = os.open(temporary_path, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
            with os.fdopen(descriptor, "wt", encoding="ascii") as state_file:
                state_file.write(f"{timestamp}\n")
                state_file.flush()
                os.fsync(state_file.fileno())
            os.replace(temporary_path, self.signing_state_path)
        except OSError as exc:
            raise RuntimeError(f"cannot persist MAVLink signing state: {exc}") from exc
