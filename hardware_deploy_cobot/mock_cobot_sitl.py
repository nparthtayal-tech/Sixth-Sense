"""Send a nominal authenticated cobot stream followed by an obstacle event or clear containment."""

from __future__ import annotations

import argparse
import hashlib
import hmac
import json
import socket
import time
import uuid

from telemetry import canonical_packet, load_hmac_key

DEMO_KEY = "MDEyMzQ1Njc4OWFiY2RlZjAxMjM0NTY3ODlhYmNkZWY="


def signed_packet(key: bytes, sequence: int, sensor: str, session_id: str = "", **values: object) -> bytes:
    packet: dict = {
        "schema_version": 1,
        "robot_id": "COBOT-ALPHA-01",
        "sequence": sequence,
        "timestamp": time.time(),
        "sensor": sensor,
        **values,
    }
    if session_id:
        packet["session_id"] = session_id
    packet["signature"] = hmac.new(key, canonical_packet(packet), hashlib.sha256).hexdigest()
    return json.dumps(packet, separators=(",", ":")).encode("utf-8")


def signed_clear_packet(key: bytes, sequence: int, session_id: str = "", allow_lockdown: bool = False) -> bytes:
    robot_id = "COBOT-ALPHA-01"
    # Token matching HMAC over robot_id:STOPPED or robot_id:LOCKDOWN
    target_state = "LOCKDOWN" if allow_lockdown else "STOPPED"
    token = hmac.new(key, f"{robot_id}:{target_state}".encode("utf-8"), hashlib.sha256).hexdigest()
    packet = {
        "schema_version": 1,
        "robot_id": robot_id,
        "sequence": sequence,
        "session_id": session_id or uuid.uuid4().hex[:8],
        "timestamp": time.time(),
        "action": "CLEAR_SAFE_STOP",
        "token": token,
        "allow_lockdown_clear": allow_lockdown,
    }
    packet["signature"] = hmac.new(key, canonical_packet(packet), hashlib.sha256).hexdigest()
    return json.dumps(packet, separators=(",", ":")).encode("utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Cobot authenticated telemetry simulator")
    parser.add_argument("--udp-host", default="127.0.0.1")
    parser.add_argument("--udp-port", type=int, default=5010)
    parser.add_argument("--clear", action="store_true", help="Send an authenticated clearance command")
    parser.add_argument("--allow-lockdown", action="store_true", help="Clear lockdown state")
    args = parser.parse_args()

    key = load_hmac_key(DEMO_KEY)
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    session_id = uuid.uuid4().hex[:8]

    try:
        if args.clear:
            clear_pkt = signed_clear_packet(key, 1, session_id, allow_lockdown=args.allow_lockdown)
            sock.sendto(clear_pkt, (args.udp_host, args.udp_port))
            print("Sent authenticated CLEAR_SAFE_STOP command.")
            return

        for sequence, x_m in enumerate((1.0, 1.2, 1.4), start=1):
            packet = signed_packet(
                key, sequence, "ODOMETRY", session_id=session_id, frame="map",
                position_m=[x_m, 1.0], yaw_rad=0.0, linear_velocity_m_s=0.3, angular_velocity_rad_s=0.0,
            )
            sock.sendto(packet, (args.udp_host, args.udp_port))
            time.sleep(0.1)

        obstacle = signed_packet(key, 4, "LIDAR", session_id=session_id, frame="base_link", min_range_m=0.2)
        sock.sendto(obstacle, (args.udp_host, args.udp_port))
        print("Sent nominal odometry followed by a 0.20 m obstacle event.")
    finally:
        sock.close()


if __name__ == "__main__":
    main()
