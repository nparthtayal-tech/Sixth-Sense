"""Latching containment policy for a mobile cobot."""

from __future__ import annotations

import hashlib
import hmac
import json
import os
import time
from dataclasses import asdict, dataclass
from enum import Enum
from pathlib import Path
from typing import Optional, Tuple


class SafeStopState(str, Enum):
    NOMINAL = "NOMINAL"
    STOPPED = "STOPPED"
    LOCKDOWN = "LOCKDOWN"


@dataclass(frozen=True)
class SafeStopSignal:
    state: SafeStopState
    reason: str
    position_m: Optional[Tuple[float, float]]

    def to_bytes(self) -> bytes:
        data = asdict(self)
        data["state"] = self.state.value
        return json.dumps(data, sort_keys=True, separators=(",", ":")).encode("utf-8")


class SafeStopController:
    """Stops latch until an authenticated supervisor explicitly clears them."""

    def __init__(
        self,
        robot_id: str = "COBOT",
        require_clearance: bool = True,
        supervisor_key: Optional[bytes] = None,
        persistence_path: Optional[Path | str] = None,
    ) -> None:
        self.robot_id = robot_id
        self.require_clearance = require_clearance
        self.supervisor_key = supervisor_key
        self.persistence_path = Path(persistence_path) if persistence_path else None
        self._state = SafeStopState.NOMINAL
        self._reason = ""
        self._last_position: Optional[Tuple[float, float]] = None

        if self.persistence_path:
            self._load_persisted_state()

    def _load_persisted_state(self) -> None:
        if not self.persistence_path or not self.persistence_path.exists():
            return
        try:
            content = json.loads(self.persistence_path.read_text(encoding="utf-8"))
            persisted_state = content.get("state")
            if persisted_state in (SafeStopState.STOPPED.value, SafeStopState.LOCKDOWN.value):
                self._state = SafeStopState(persisted_state)
                self._reason = str(content.get("reason", "Persisted latch from prior session"))
                pos = content.get("last_position")
                if isinstance(pos, list) and len(pos) == 2:
                    self._last_position = (float(pos[0]), float(pos[1]))
        except Exception:
            # Fall back safely: if state file is corrupt, retain nominal or safe stop
            pass

    def _save_persisted_state(self) -> None:
        if not self.persistence_path:
            return
        try:
            self.persistence_path.parent.mkdir(parents=True, exist_ok=True)
            data = {
                "robot_id": self.robot_id,
                "state": self._state.value,
                "reason": self._reason,
                "last_position": list(self._last_position) if self._last_position else None,
                "updated_at": time.time(),
            }
            tmp_file = self.persistence_path.with_suffix(".tmp")
            tmp_file.write_text(json.dumps(data, indent=2), encoding="utf-8")
            # Atomic replace
            os.replace(tmp_file, self.persistence_path)
        except OSError:
            pass

    def update_position(self, x_m: float, y_m: float) -> None:
        if self._state is SafeStopState.NOMINAL:
            self._last_position = (float(x_m), float(y_m))

    def stop(self, reason: str, lockdown: bool = False) -> SafeStopSignal:
        if self._state is SafeStopState.NOMINAL or lockdown:
            self._state = SafeStopState.LOCKDOWN if lockdown else SafeStopState.STOPPED
            self._reason = reason
            self._save_persisted_state()
        return self.status()

    def verify_clearance_token(self, token: str, allow_lockdown_clear: bool = False) -> bool:
        """Validate whether a token is cryptographically authorized to clear the current stop."""
        if not token or not isinstance(token, str):
            return False

        # If locked down, require explicit lockdown authorization
        if self._state == SafeStopState.LOCKDOWN and not allow_lockdown_clear:
            return False

        if not self.supervisor_key:
            # If no supervisor key configured, any non-empty string is accepted (dev/test mode)
            return True

        # Expected token: HMAC-SHA256(supervisor_key, f"{robot_id}:{state}")
        # Or format: "{timestamp}:{hmac}" where hmac covers f"{robot_id}:{state}:{timestamp}"
        target_message = f"{self.robot_id}:{self._state.value}".encode("utf-8")
        expected_direct = hmac.new(self.supervisor_key, target_message, hashlib.sha256).hexdigest()
        if hmac.compare_digest(token, expected_direct):
            return True

        if ":" in token:
            parts = token.split(":", 1)
            if len(parts) == 2:
                ts_str, sig = parts
                try:
                    ts = float(ts_str)
                    if abs(time.time() - ts) <= 300.0:  # 5-minute validity window
                        msg = f"{self.robot_id}:{self._state.value}:{ts_str}".encode("utf-8")
                        expected_ts = hmac.new(self.supervisor_key, msg, hashlib.sha256).hexdigest()
                        if hmac.compare_digest(sig, expected_ts):
                            return True
                except ValueError:
                    pass

        return False

    def clear(self, clearance_token: str = "", allow_lockdown_clear: bool = False) -> SafeStopSignal:
        if self.require_clearance:
            if not self.verify_clearance_token(clearance_token, allow_lockdown_clear=allow_lockdown_clear):
                return self.status()

        self._state = SafeStopState.NOMINAL
        self._reason = ""
        self._save_persisted_state()
        return self.status()

    def status(self) -> SafeStopSignal:
        return SafeStopSignal(self._state, self._reason, self._last_position)

    @property
    def is_stopped(self) -> bool:
        return self._state is not SafeStopState.NOMINAL

    @property
    def state(self) -> SafeStopState:
        return self._state
