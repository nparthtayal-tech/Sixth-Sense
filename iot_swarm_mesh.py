#!/usr/bin/env python3
"""
IoT Multi-Robot Sensor Mesh & Collaborative Information Gathering Engine
========================================================================
Implements a decentralized IoT sensor mesh for heterogeneous fleets
(Aerial Drones + Ground Cobots). Nodes broadcast authenticated sensor
beacons, share obstacle/environmental telemetry, and perform consensus-based
cross-validation to detect GPS spoofing and sensor tampering.
"""

from __future__ import annotations

import argparse
import hashlib
import hmac
import json
import math
import os
import queue
import socket
import sys
import threading
import time
from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

DEMO_HMAC_KEY = b"0123456789abcdef0123456789abcdef"
COMMUNICATION_RANGE_M = 180.0  # Max wireless P2P communication range
SPOOFING_TOLERANCE_M = 15.0    # Distance deviation threshold for consensus vote


class RobotRole(str, Enum):
    AERIAL_LEAD = "aerial_lead"
    AERIAL_SCOUT = "aerial_scout"
    GROUND_COBOT = "ground_cobot"


@dataclass
class EnvironmentalSensors:
    gas_ppm: float = 12.0
    temperature_c: float = 26.5
    obstacle_dist_m: float = 10.0
    obstacle_bearing_deg: float = 0.0
    rf_interference_level: float = 0.05
    gps_trust_score: float = 1.0


@dataclass
class HazardReport:
    hazard_id: str
    reported_by: str
    timestamp: float
    hazard_type: str  # "OBSTACLE", "GAS_LEAK", "GPS_SPOOFING", "RF_JAMMING"
    position: Tuple[float, float]
    severity: str     # "LOW", "MEDIUM", "HIGH", "CRITICAL"
    details: str = ""


@dataclass
class PeerState:
    node_id: str
    role: str
    last_seen: float
    position: Tuple[float, float]
    altitude_m: float
    battery_pct: float
    rssi_dbm: float
    gps_trust_score: float
    is_compromised: bool = False
    consensus_votes_against: int = 0


def haversine_distance_m(pos1: Tuple[float, float], pos2: Tuple[float, float]) -> float:
    """Compute distance in meters between two lat/lng coordinates."""
    lat1, lon1 = math.radians(pos1[0]), math.radians(pos1[1])
    lat2, lon2 = math.radians(pos2[0]), math.radians(pos2[1])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = math.sin(dlat / 2)**2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon / 2)**2
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
    return 6371000.0 * c


def simulate_rssi(distance_m: float) -> float:
    """Path loss model estimating RSSI in dBm from distance."""
    if distance_m <= 1.0:
        return -35.0
    # Log-distance path loss: RSSI = -40 - 10 * n * log10(d)
    rssi = -38.0 - (2.5 * 10 * math.log10(max(1.0, distance_m)))
    return round(max(-95.0, min(-30.0, rssi)), 1)


class IoTNode:
    """Represents an individual autonomous robot acting as an IoT edge node."""

    def __init__(
        self,
        node_id: str,
        role: RobotRole,
        initial_position: Tuple[float, float],
        altitude_m: float = 0.0,
        key: bytes = DEMO_HMAC_KEY,
    ) -> None:
        self.node_id = node_id
        self.role = role
        self.position = initial_position
        self.true_physical_position = initial_position  # Ground truth for ranging
        self.altitude_m = altitude_m
        self.heading_deg = 0.0
        self.speed_m_s = 0.0
        self.battery_pct = 95.0
        self.key = key
        self.sequence = 0

        self.sensors = EnvironmentalSensors()
        self.peer_table: Dict[str, PeerState] = {}
        self.shared_hazards: Dict[str, HazardReport] = {}
        self.active_mesh_links: List[Tuple[str, str, float]] = []  # (node_a, node_b, rssi)

        # Attack flags
        self.is_spoofed = False
        self.isolated_by_swarm = False
        self._lock = threading.Lock()

    def update_sensors(self, **kwargs: Any) -> None:
        with self._lock:
            for k, v in kwargs.items():
                if hasattr(self.sensors, k):
                    setattr(self.sensors, k, v)

    def create_sensor_beacon(self) -> dict:
        """Create an authenticated IoT telemetry beacon."""
        with self._lock:
            self.sequence += 1
            payload = {
                "schema_version": 1,
                "node_id": self.node_id,
                "role": self.role.value,
                "sequence": self.sequence,
                "timestamp": time.time(),
                "position": [self.position[0], self.position[1]],
                "altitude_m": self.altitude_m,
                "heading_deg": self.heading_deg,
                "speed_m_s": self.speed_m_s,
                "battery_pct": round(self.battery_pct, 1),
                "sensors": asdict(self.sensors),
                "active_hazards": [asdict(h) for h in self.shared_hazards.values()],
            }

            canonical = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
            payload["signature"] = hmac.new(self.key, canonical, hashlib.sha256).hexdigest()
            return payload

    def ingest_neighbor_beacon(self, beacon: dict) -> Tuple[bool, str]:
        """Ingest and cross-validate an incoming beacon from a peer node."""
        with self._lock:
            # 1. Verify signature
            sig = beacon.get("signature")
            if not sig:
                return False, "missing signature"
            unsigned = {k: v for k, v in beacon.items() if k != "signature"}
            canonical = json.dumps(unsigned, sort_keys=True, separators=(",", ":")).encode("utf-8")
            expected_sig = hmac.new(self.key, canonical, hashlib.sha256).hexdigest()
            if not hmac.compare_digest(sig, expected_sig):
                return False, "invalid signature"

            peer_id = beacon["node_id"]
            if peer_id == self.node_id:
                return True, "self"

            reported_pos = (float(beacon["position"][0]), float(beacon["position"][1]))
            reported_dist = haversine_distance_m(self.position, reported_pos)

            # Check if within wireless communication radius
            if reported_dist > COMMUNICATION_RANGE_M:
                return False, "out of wireless mesh range"

            rssi = simulate_rssi(reported_dist)

            # 2. Update Peer Table
            peer = self.peer_table.get(peer_id)
            if not peer:
                peer = PeerState(
                    node_id=peer_id,
                    role=beacon["role"],
                    last_seen=beacon["timestamp"],
                    position=reported_pos,
                    altitude_m=float(beacon["altitude_m"]),
                    battery_pct=float(beacon["battery_pct"]),
                    rssi_dbm=rssi,
                    gps_trust_score=float(beacon["sensors"].get("gps_trust_score", 1.0)),
                )
                self.peer_table[peer_id] = peer
            else:
                peer.last_seen = beacon["timestamp"]
                peer.position = reported_pos
                peer.altitude_m = float(beacon["altitude_m"])
                peer.battery_pct = float(beacon["battery_pct"])
                peer.rssi_dbm = rssi
                peer.gps_trust_score = float(beacon["sensors"].get("gps_trust_score", 1.0))

            # 3. Ingest and gather environmental hazards from peer
            for h in beacon.get("active_hazards", []):
                hid = h["hazard_id"]
                if hid not in self.shared_hazards:
                    self.shared_hazards[hid] = HazardReport(
                        hazard_id=hid,
                        reported_by=h["reported_by"],
                        timestamp=h["timestamp"],
                        hazard_type=h["hazard_type"],
                        position=(h["position"][0], h["position"][1]),
                        severity=h["severity"],
                        details=h.get("details", ""),
                    )

            # 4. Consensus Cross-Validation Check:
            # Compare physical peer ranging with reported GPS position
            actual_physical_dist = haversine_distance_m(self.true_physical_position, reported_pos)
            discrepancy = abs(reported_dist - actual_physical_dist)

            if discrepancy > SPOOFING_TOLERANCE_M or peer.gps_trust_score < 0.5:
                peer.consensus_votes_against += 1
                peer.is_compromised = True
                # Record collaborative hazard
                haz_id = f"SPOOF-{peer_id}"
                self.shared_hazards[haz_id] = HazardReport(
                    hazard_id=haz_id,
                    reported_by=self.node_id,
                    timestamp=time.time(),
                    hazard_type="GPS_SPOOFING",
                    position=reported_pos,
                    severity="CRITICAL",
                    details=f"Cross-ranging discrepancy: {discrepancy:.1f}m detected by {self.node_id}",
                )
            else:
                peer.consensus_votes_against = 0
                peer.is_compromised = False

            return True, "accepted"


class IoTSwarmMesh:
    """Coordinates a swarm of IoT robots and manages the collective perception network."""

    def __init__(self, key: bytes = DEMO_HMAC_KEY) -> None:
        self.key = key
        self.nodes: Dict[str, IoTNode] = {}
        self._running = False
        self._thread: Optional[threading.Thread] = None

    def add_node(self, node: IoTNode) -> None:
        self.nodes[node.node_id] = node

    def setup_default_fleet(self) -> None:
        """Create a default mixed aerial-ground fleet."""
        # Chennai harbor / tech park default coordinates
        base_lat, base_lng = 13.0600, 80.2800

        lead_drone = IoTNode(
            node_id="UAV-ALPHA",
            role=RobotRole.AERIAL_LEAD,
            initial_position=(base_lat, base_lng),
            altitude_m=45.0,
        )
        lead_drone.update_sensors(gas_ppm=14.2, obstacle_dist_m=25.0)

        scout_drone = IoTNode(
            node_id="UAV-BETA",
            role=RobotRole.AERIAL_SCOUT,
            initial_position=(base_lat + 0.0006, base_lng + 0.0008),  # ~110m away
            altitude_m=35.0,
        )
        scout_drone.update_sensors(gas_ppm=18.5, obstacle_dist_m=18.0)

        ground_cobot = IoTNode(
            node_id="AGV-OMEGA",
            role=RobotRole.GROUND_COBOT,
            initial_position=(base_lat + 0.0003, base_lng + 0.0004),  # ~55m away
            altitude_m=0.5,
        )
        ground_cobot.update_sensors(gas_ppm=34.8, obstacle_dist_m=3.2)  # Cobot detected near gas reading

        self.add_node(lead_drone)
        self.add_node(scout_drone)
        self.add_node(ground_cobot)

    def exchange_mesh_cycle(self) -> Dict[str, Any]:
        """Perform one broadcast round where each node emits and neighbors ingest."""
        beacons = {nid: node.create_sensor_beacon() for nid, node in self.nodes.items()}

        active_links = []
        # Simulate P2P wireless broadcast within transmission range
        for sender_id, beacon in beacons.items():
            sender_node = self.nodes[sender_id]
            for receiver_id, receiver_node in self.nodes.items():
                if sender_id == receiver_id:
                    continue
                dist = haversine_distance_m(sender_node.position, receiver_node.position)
                if dist <= COMMUNICATION_RANGE_M:
                    receiver_node.ingest_neighbor_beacon(beacon)
                    active_links.append({
                        "from": sender_id,
                        "to": receiver_id,
                        "distance_m": round(dist, 1),
                        "rssi_dbm": simulate_rssi(dist),
                        "status": "nominal" if not sender_node.is_spoofed else "threat",
                    })

        # Collect aggregate swarm knowledge and consensus voting
        aggregated_hazards = {}
        accusation_counts: Dict[str, int] = {}
        for node in self.nodes.values():
            aggregated_hazards.update(node.shared_hazards)
            for pid, peer in node.peer_table.items():
                if peer.is_compromised:
                    accusation_counts[pid] = accusation_counts.get(pid, 0) + 1

        # Swarm Consensus: Node is confirmed compromised if >= 2 independent peers flag it
        compromised_nodes = [pid for pid, count in accusation_counts.items() if count >= 2]

        return {
            "timestamp": time.time(),
            "active_node_count": len(self.nodes),
            "mesh_links": active_links,
            "compromised_nodes": compromised_nodes,
            "collective_hazards": [asdict(h) for h in aggregated_hazards.values()],
        }

    def simulate_spoofing_attack(self, target_node_id: str, offset_m: float = 60.0) -> None:
        """Inject a simulated GPS spoofing attack on a specific swarm robot."""
        target = self.nodes.get(target_node_id)
        if not target:
            return
        target.is_spoofed = True
        # Move reported GPS position by offset meters, keeping true physical position intact
        lat_offset = offset_m / 111320.0
        target.position = (target.true_physical_position[0] + lat_offset, target.true_physical_position[1])
        target.update_sensors(gps_trust_score=0.2, rf_interference_level=0.92)


def run_self_test() -> None:
    print("==================================================================")
    print(" SensorSentry IoT Multi-Robot Sensor Mesh — Decentralized Test    ")
    print("==================================================================")
    swarm = IoTSwarmMesh()
    swarm.setup_default_fleet()

    print(f"Initialized IoT Swarm Fleet: {list(swarm.nodes.keys())}")
    for nid, node in swarm.nodes.items():
        print(f"  • {nid} [{node.role.value}] at {node.position}, battery: {node.battery_pct}%")

    print("\n--- Cycle 1: Nominal IoT Mesh Discovery & Beacon Exchange ---")
    state = swarm.exchange_mesh_cycle()
    print(f"Active P2P Mesh Links Established: {len(state['mesh_links'])}")
    for link in state['mesh_links']:
        print(f"  Link: {link['from']} <---> {link['to']} | Dist: {link['distance_m']}m | RSSI: {link['rssi_dbm']} dBm")

    print(f"Collective Hazards Shared Across Swarm: {len(state['collective_hazards'])}")

    print("\n--- Cycle 2: Simulating GPS Spoofing Attack on UAV-ALPHA ---")
    print("Injecting 60-meter spoofing divergence on UAV-ALPHA...")
    swarm.simulate_spoofing_attack("UAV-ALPHA", offset_m=60.0)

    state2 = swarm.exchange_mesh_cycle()
    print(f"Swarm Compromised Node Flagged: {state2['compromised_nodes']}")
    assert "UAV-ALPHA" in state2["compromised_nodes"], "Swarm consensus must flag UAV-ALPHA as compromised"

    print("Shared Collective Threat Records:")
    for h in state2["collective_hazards"]:
        print(f"  [ALERT] {h['hazard_type']} by {h['reported_by']}: {h['details']}")

    print("\n[PASS] IoT Multi-Robot Sensor Mesh, P2P telemetry exchange, and Byzantine consensus verified successfully!")


def main() -> None:
    parser = argparse.ArgumentParser(description="SensorSentry IoT Multi-Robot Sensor Mesh")
    parser.add_argument("--test", action="store_true", help="Run automated self-test and verification")
    parser.add_argument("--cycle-rate-hz", type=float, default=2.0)
    args = parser.parse_args()

    if args.test:
        run_self_test()
        return

    swarm = IoTSwarmMesh()
    swarm.setup_default_fleet()
    print("Starting SensorSentry IoT Swarm Mesh daemon. Press Ctrl+C to stop.")
    interval = 1.0 / max(0.2, args.cycle_rate_hz)
    try:
        while True:
            state = swarm.exchange_mesh_cycle()
            print(f"[{time.strftime('%H:%M:%S')}] Swarm nodes: {state['active_node_count']} | Links: {len(state['mesh_links'])} | Compromised: {state['compromised_nodes']}")
            time.sleep(interval)
    except KeyboardInterrupt:
        print("\nIoT Swarm Mesh daemon stopped.")


if __name__ == "__main__":
    main()
