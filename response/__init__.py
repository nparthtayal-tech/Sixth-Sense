"""
response — Emergency Response Layer
=====================================
Executes immediate vehicle immobilization and fleet alerts when
either attack vector is confirmed.

1. safe_stop.py    — State machine for emergency brake lockdown
2. fleet_alert.py  — High-priority alert broadcast to Fleet Command Center
"""

from .safe_stop import SafeStopController, SafeStopSignal, SafeStopState
from .fleet_alert import FleetAlertBroadcaster, FleetAlert, AlertSeverity, AttackType

__all__ = [
    'SafeStopController', 'SafeStopSignal', 'SafeStopState',
    'FleetAlertBroadcaster', 'FleetAlert', 'AlertSeverity', 'AttackType'
]
