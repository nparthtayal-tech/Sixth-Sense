"""
saarm — Sensor Autonomous Adaptive Reconfiguration Management
==============================================================
Parallel filter bank that detects, isolates, and quarantines
compromised sensor channels (primarily GPS spoofing).

When CUSUM triggers an ALARM on a sensor, SAARM runs N exclusion
sub-filters (each omitting one sensor) and compares their residuals
to identify which sensor is poisoning the EKF state.

Once identified, SAARM switches the navigation solution to the
exclusion filter that omits the compromised sensor, enabling
dead-reckoning via IMU + wheel encoders + LiDAR + 5G.
"""

from .saarm_filter_bank import (
    SaarmFilterBank,
    SaarmSignal,
    SaarmVerdict,
    DeadReckoningNavigator
)

__all__ = [
    'SaarmFilterBank',
    'SaarmSignal',
    'SaarmVerdict',
    'DeadReckoningNavigator'
]
