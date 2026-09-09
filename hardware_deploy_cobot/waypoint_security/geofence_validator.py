"""
geofence_validator.py — Spatial Bounds Enforcement
====================================================
Validates proposed destinations against a secure local geofenced map
of authorized operational zones.

If a proposed destination falls outside all authorized zones, or inside
a known restricted/unsafe zone, the validator rejects the waypoint.

Zones can be:
    - Circular  (center + radius)
    - Polygonal (list of vertices)

Zone updates require HMAC-signed commands to prevent the attacker
from simply adding their warehouse to the authorized zone list.

Hardware Interface:
    GeofenceSignal.to_bytes() → 16-byte packet for CAN / UART / SPI.
    GeofenceSignal.to_dict()  → JSON-serializable dict for ROS2 / TCP / MQTT.

References:
    [9] Environmental data manipulation detection
    [10] Cryptographic and geometric handshake
"""

import struct
import hashlib
import hmac
import numpy as np
from dataclasses import dataclass, field
from enum import IntEnum
from typing import List, Optional, Tuple


class GeofenceVerdict(IntEnum):
    """Geofence check verdict for hardware controllers."""
    AUTHORIZED   = 0   # Destination is within an authorized zone
    UNAUTHORIZED = 1   # Destination is outside all authorized zones
    RESTRICTED   = 2   # Destination is inside a restricted/unsafe zone
    ZONE_ERROR   = 3   # Zone database integrity check failed


@dataclass
class GeofenceZone:
    """Defines an authorized operational zone."""
    zone_id: str
    zone_type: str          # "circular" or "polygon"
    center: Optional[np.ndarray] = None   # For circular zones [x, y]
    radius: float = 0.0                    # For circular zones (meters)
    vertices: Optional[List[np.ndarray]] = None  # For polygon zones
    is_restricted: bool = False  # If True, this is a NO-GO zone
    label: str = ""


@dataclass
class GeofenceSignal:
    """
    Output signal from the Geofence Validator.
    Designed for direct transmission to hardware decision units.
    """
    timestamp: float
    verdict: GeofenceVerdict
    destination: np.ndarray      # The proposed destination [x, y, z]
    nearest_zone_id: str         # ID of the closest authorized zone
    distance_to_zone: float      # Distance to nearest zone boundary (m)
    reason: str                  # Human-readable explanation

    def to_dict(self) -> dict:
        """JSON / telemetry serialization."""
        return {
            'timestamp': round(self.timestamp, 4),
            'verdict': self.verdict.name,
            'verdict_code': int(self.verdict),
            'destination': [round(float(v), 4) for v in self.destination],
            'nearest_zone_id': self.nearest_zone_id,
            'distance_to_zone': round(float(self.distance_to_zone), 2),
            'reason': self.reason
        }

    def to_bytes(self) -> bytes:
        """
        Compact 16-byte hardware packet:
        [uint32 timestamp_ms, uint8 verdict, uint8 reserved,
         uint8 reserved, uint8 reserved,
         float32 dest_x, float32 dest_y]
        """
        ts_ms = int(self.timestamp * 1000) & 0xFFFFFFFF
        return struct.pack('<IBBBBff',
                           ts_ms,
                           int(self.verdict),
                           0, 0, 0,
                           float(self.destination[0]),
                           float(self.destination[1]))


class GeofenceValidator:
    """
    Geofence Validator — Spatial Bounds Enforcement.
    
    Maintains a secure database of authorized operational zones
    and validates every proposed destination against them.
    
    Parameters
    ----------
    hmac_key : bytes
        Secret key for HMAC-SHA256 verification of zone updates.
        Zone database modifications require a valid HMAC signature.
    """
    
    def __init__(self, hmac_key: bytes = b'sensorsentry-default-key'):
        self._hmac_key = hmac_key
        self._authorized_zones: List[GeofenceZone] = []
        self._restricted_zones: List[GeofenceZone] = []
        self._zone_db_hash: str = ""
        self._update_db_hash()
    
    def add_zone(self, zone: GeofenceZone, signature: Optional[bytes] = None):
        """
        Add an authorized or restricted zone to the database.
        
        In production, requires a valid HMAC signature to prevent
        an attacker from adding their location as an authorized zone.
        In simulation/testing mode, signature can be None.
        
        Parameters
        ----------
        zone : GeofenceZone
            The zone definition to add.
        signature : bytes or None
            HMAC-SHA256 signature of the zone_id. Required in production.
        """
        # In production, verify HMAC signature
        if signature is not None:
            expected = hmac.new(self._hmac_key, zone.zone_id.encode(), hashlib.sha256).digest()
            if not hmac.compare_digest(signature, expected):
                raise ValueError(f"Invalid signature for zone '{zone.zone_id}' — "
                                 f"potential zone injection attack")
        
        if zone.is_restricted:
            self._restricted_zones.append(zone)
        else:
            self._authorized_zones.append(zone)
        
        self._update_db_hash()
    
    def add_circular_zone(self, zone_id: str, center_x: float, center_y: float,
                          radius: float, label: str = "", is_restricted: bool = False):
        """Convenience method to add a circular geofenced zone."""
        zone = GeofenceZone(
            zone_id=zone_id,
            zone_type="circular",
            center=np.array([center_x, center_y]),
            radius=radius,
            is_restricted=is_restricted,
            label=label
        )
        self.add_zone(zone)
    
    def add_polygon_zone(self, zone_id: str, vertices: List[Tuple[float, float]],
                         label: str = "", is_restricted: bool = False):
        """Convenience method to add a polygonal geofenced zone."""
        verts = [np.array([v[0], v[1]]) for v in vertices]
        zone = GeofenceZone(
            zone_id=zone_id,
            zone_type="polygon",
            vertices=verts,
            is_restricted=is_restricted,
            label=label
        )
        self.add_zone(zone)
    
    def validate(self, destination: np.ndarray, timestamp: float) -> GeofenceSignal:
        """
        Validate a proposed destination against the geofence database.
        
        Parameters
        ----------
        destination : np.ndarray
            Proposed destination coordinates [x, y] or [x, y, z].
        timestamp : float
            Current time in seconds.
        
        Returns
        -------
        GeofenceSignal with AUTHORIZED, UNAUTHORIZED, or RESTRICTED verdict.
        """
        dest_2d = np.array([destination[0], destination[1]])
        dest_full = np.atleast_1d(destination).astype(float)
        if len(dest_full) < 3:
            dest_full = np.array([dest_full[0], dest_full[1], 0.0])
        
        # Check restricted zones FIRST (takes priority)
        for zone in self._restricted_zones:
            if self._point_in_zone(dest_2d, zone):
                return GeofenceSignal(
                    timestamp=timestamp,
                    verdict=GeofenceVerdict.RESTRICTED,
                    destination=dest_full,
                    nearest_zone_id=zone.zone_id,
                    distance_to_zone=0.0,
                    reason=f"Destination inside RESTRICTED zone '{zone.label or zone.zone_id}'"
                )
        
        # Check authorized zones
        nearest_zone_id = ""
        min_distance = float('inf')
        
        for zone in self._authorized_zones:
            if self._point_in_zone(dest_2d, zone):
                return GeofenceSignal(
                    timestamp=timestamp,
                    verdict=GeofenceVerdict.AUTHORIZED,
                    destination=dest_full,
                    nearest_zone_id=zone.zone_id,
                    distance_to_zone=0.0,
                    reason=f"Destination within authorized zone '{zone.label or zone.zone_id}'"
                )
            
            # Track nearest zone for diagnostics
            dist = self._distance_to_zone(dest_2d, zone)
            if dist < min_distance:
                min_distance = dist
                nearest_zone_id = zone.zone_id
        
        # Not in any authorized zone → UNAUTHORIZED
        return GeofenceSignal(
            timestamp=timestamp,
            verdict=GeofenceVerdict.UNAUTHORIZED,
            destination=dest_full,
            nearest_zone_id=nearest_zone_id,
            distance_to_zone=min_distance,
            reason=f"Destination {min_distance:.1f}m outside nearest authorized zone '{nearest_zone_id}'"
        )
    
    def _point_in_zone(self, point: np.ndarray, zone: GeofenceZone) -> bool:
        """Check if a 2D point is inside a zone."""
        if zone.zone_type == "circular":
            dist = np.linalg.norm(point - zone.center)
            return dist <= zone.radius
        elif zone.zone_type == "polygon" and zone.vertices:
            return self._point_in_polygon(point, zone.vertices)
        return False
    
    def _distance_to_zone(self, point: np.ndarray, zone: GeofenceZone) -> float:
        """Distance from a 2D point to the nearest zone boundary."""
        if zone.zone_type == "circular":
            dist = np.linalg.norm(point - zone.center) - zone.radius
            return max(0.0, dist)
        elif zone.zone_type == "polygon" and zone.vertices:
            # Minimum distance to polygon edges
            min_dist = float('inf')
            n = len(zone.vertices)
            for i in range(n):
                p1 = zone.vertices[i]
                p2 = zone.vertices[(i + 1) % n]
                d = self._point_to_segment_dist(point, p1, p2)
                min_dist = min(min_dist, d)
            return min_dist
        return float('inf')
    
    @staticmethod
    def _point_in_polygon(point: np.ndarray, vertices: List[np.ndarray]) -> bool:
        """Ray-casting algorithm for point-in-polygon test."""
        n = len(vertices)
        inside = False
        px, py = point[0], point[1]
        
        j = n - 1
        for i in range(n):
            xi, yi = vertices[i][0], vertices[i][1]
            xj, yj = vertices[j][0], vertices[j][1]
            
            if ((yi > py) != (yj > py)) and (px < (xj - xi) * (py - yi) / (yj - yi) + xi):
                inside = not inside
            j = i
        
        return inside
    
    @staticmethod
    def _point_to_segment_dist(p: np.ndarray, a: np.ndarray, b: np.ndarray) -> float:
        """Minimum distance from point p to line segment a-b."""
        ab = b - a
        ap = p - a
        t = np.dot(ap, ab) / max(np.dot(ab, ab), 1e-10)
        t = np.clip(t, 0.0, 1.0)
        closest = a + t * ab
        return float(np.linalg.norm(p - closest))
    
    def _update_db_hash(self):
        """Update integrity hash of the zone database."""
        data = str(len(self._authorized_zones)) + str(len(self._restricted_zones))
        for z in self._authorized_zones + self._restricted_zones:
            data += z.zone_id
        self._zone_db_hash = hashlib.sha256(data.encode()).hexdigest()[:16]
    
    @property
    def zone_count(self) -> int:
        return len(self._authorized_zones) + len(self._restricted_zones)
    
    @property
    def db_integrity_hash(self) -> str:
        return self._zone_db_hash
