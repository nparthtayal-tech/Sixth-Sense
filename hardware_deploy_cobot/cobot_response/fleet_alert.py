"""Structured fleet alerts for cobot safety containment."""

from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass
from typing import Dict, Optional, Tuple


@dataclass(frozen=True)
class FleetAlert:
    alert_id: str
    timestamp: float
    robot_id: str
    severity: str
    category: str
    reason: str
    position_m: Optional[Tuple[float, float]]
    evidence: Dict[str, str]
    recommended_action: str

    def to_bytes(self) -> bytes:
        return json.dumps(asdict(self), sort_keys=True, separators=(",", ":")).encode("utf-8")


class FleetAlertBroadcaster:
    def __init__(self, robot_id: str) -> None:
        self.robot_id = robot_id
        self._counter = 0
        self.history: list[FleetAlert] = []

    def create(
        self,
        category: str,
        reason: str,
        position_m: Optional[Tuple[float, float]],
        evidence: Optional[Dict[str, str]] = None,
        severity: str = "CRITICAL",
    ) -> FleetAlert:
        self._counter += 1
        alert = FleetAlert(
            alert_id=f"COBOT-{self.robot_id}-{self._counter:04d}",
            timestamp=time.time(),
            robot_id=self.robot_id,
            severity=severity,
            category=category,
            reason=reason,
            position_m=position_m,
            evidence=evidence or {},
            recommended_action="Keep the cobot stopped and require supervisor clearance before resuming.",
        )
        self.history.append(alert)
        return alert
