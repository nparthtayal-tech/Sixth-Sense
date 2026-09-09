"""
waypoint_security — Destination Integrity & Geofence Enforcement
=================================================================
Defends against Scenario B: Waypoint Hijack (Destination Code Tampering).

Three layers of cross-validation before any destination change is accepted:
1. command_validator   — Cryptographic HMAC signing + replay protection
2. geofence_validator  — Spatial bounds checking against authorized zones
3. environment_matcher — Physical landmark consensus from onboard sensors
"""

from .geofence_validator import GeofenceValidator, GeofenceSignal, GeofenceVerdict, GeofenceZone
from .command_validator import CommandValidator, CommandSignal, CommandVerdict, WaypointCommand
from .environment_matcher import EnvironmentMatcher, EnvironmentSignal, EnvironmentVerdict, ExpectedLandmark, DetectedFeature

__all__ = [
    'GeofenceValidator', 'GeofenceSignal', 'GeofenceVerdict', 'GeofenceZone',
    'CommandValidator', 'CommandSignal', 'CommandVerdict', 'WaypointCommand',
    'EnvironmentMatcher', 'EnvironmentSignal', 'EnvironmentVerdict', 'ExpectedLandmark', 'DetectedFeature'
]
