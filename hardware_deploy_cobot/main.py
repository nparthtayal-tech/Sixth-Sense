#!/usr/bin/env python3
"""Secure companion-computer runner for a mobile cobot."""

from __future__ import annotations

import argparse
import base64
import hashlib
import hmac
import json
import os
import queue
import socket
import sys
import threading
import time
from pathlib import Path
from typing import Any, Mapping, Optional

BASE_DIR = Path(__file__).resolve().parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from cobot_control import AdvisoryCobotControl, CobotControl, Ros2CobotControl
from cobot_hal import SimulatedBus
from sensor_sentry import CobotSafetyLimits, CobotSentry
from telemetry import (PacketAuthenticator, TelemetryError, canonical_packet,
                       load_hmac_key, parse_clearance_packet, parse_json,
                       parse_packet)
from cobot_waypoint_security import MapBoundary

DEMO_KEY_B64 = "MDEyMzQ1Njc4OWFiY2RlZjAxMjM0NTY3ODlhYmNkZWY="


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="SensorSentry authenticated cobot safety supervisor")
    parser.add_argument("--config", type=Path, default=BASE_DIR / "config.json")
    parser.add_argument("--mode", choices=("udp", "stdin", "test"), default="udp")
    parser.add_argument("--robot-id")
    parser.add_argument("--udp-host", default="0.0.0.0")
    parser.add_argument("--udp-port", type=int)
    parser.add_argument("--hmac-key-env", default="SENSORSENTRY_COBOT_HMAC_KEY")
    parser.add_argument("--supervisor-key-env", default=None)
    parser.add_argument("--state-persistence", type=Path, default=None)
    parser.add_argument("--controller", choices=("advisory", "ros2"), default="advisory")
    parser.add_argument("--enable-robot-actions", action="store_true")
    parser.add_argument("--ros-namespace", default=None)
    parser.add_argument("--trusted-route", type=Path,
                        help="JSON file containing a previously authorised [[x, y], ...] map route")
    return parser.parse_args()


def load_config(path: Path) -> Mapping[str, Any]:
    try:
        config = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"cannot read config {path}: {exc}") from exc
    if not isinstance(config, dict):
        raise RuntimeError("config root must be an object")
    return config


def create_sentry(
    config: Mapping[str, Any],
    robot_id: str,
    control: CobotControl,
    supervisor_key: Optional[bytes] = None,
    persistence_path: Optional[Path] = None,
) -> CobotSentry:
    envelope = config["navigation_envelope"]
    connection = config["connection"]
    monitors = config["security_monitors"]

    limits = CobotSafetyLimits(
        map_boundary=MapBoundary.from_config(envelope["map_bounds_m"]),
        corridor_half_width_m=float(envelope["corridor_half_width_m"]),
        max_linear_speed_m_s=float(envelope["max_linear_speed_m_s"]),
        max_angular_speed_rad_s=float(envelope["max_angular_speed_rad_s"]),
        minimum_obstacle_distance_m=float(envelope["minimum_obstacle_distance_m"]),
        max_localization_jump_m=float(envelope["max_localization_jump_m"]),
        sensor_timeout_s=float(connection["sensor_timeout_s"]),
    )

    if persistence_path is None and monitors.get("state_persistence_file"):
        persistence_path = BASE_DIR / monitors["state_persistence_file"]

    return CobotSentry(
        robot_id=robot_id,
        limits=limits,
        bus=SimulatedBus(),
        control=control,
        require_clearance=bool(monitors.get("require_clearance_to_resume", True)),
        supervisor_key=supervisor_key,
        persistence_path=persistence_path,
    )


def create_control(args: argparse.Namespace, config: Mapping[str, Any]) -> CobotControl:
    if args.controller == "advisory":
        return AdvisoryCobotControl()
    if not args.enable_robot_actions:
        raise RuntimeError("ROS 2 control requires --enable-robot-actions; advisory mode is the default")
    failsafe = config["failsafe_actuation"]
    namespace = args.ros_namespace if args.ros_namespace is not None else config["connection"]["ros_namespace"]
    return Ros2CobotControl(
        namespace=namespace,
        zero_publish_count=int(failsafe["zero_velocity_publish_count"]),
    )


def decode_and_process(data: bytes, sentry: CobotSentry, authenticator: PacketAuthenticator) -> bool:
    packet = parse_json(data)
    authenticator.verify(packet)

    # Check if this is an authenticated supervisor clearance action
    if packet.get("action") == "CLEAR_SAFE_STOP":
        clearance = parse_clearance_packet(packet)
        status = sentry.safe_stop.clear(clearance.token, allow_lockdown_clear=clearance.allow_lockdown_clear)
        if not sentry.safe_stop.is_stopped:
            print(f"[{sentry.robot_id}] Safe stop cleared by supervisor; state={status.state.value}")
            return True
        else:
            print(f"[{sentry.robot_id}] Clearance token rejected or unauthorized for state={status.state.value}", file=sys.stderr)
            return False

    reading = parse_packet(packet)
    return sentry.process_sensor_reading(reading)


def run_udp(host: str, port: int, sentry: CobotSentry, authenticator: PacketAuthenticator) -> None:
    packet_queue: queue.Queue[tuple[bytes, tuple[str, int]]] = queue.Queue(maxsize=1000)
    stop_worker = threading.Event()

    def worker() -> None:
        while not stop_worker.is_set():
            try:
                data, address = packet_queue.get(timeout=0.1)
            except queue.Empty:
                continue
            try:
                healthy = decode_and_process(data, sentry, authenticator)
                print(f"[{address[0]}] telemetry accepted; state={sentry.safe_stop.state.value}; healthy={healthy}")
            except TelemetryError as exc:
                print(f"Rejected telemetry: {exc}", file=sys.stderr)
            finally:
                packet_queue.task_done()

    worker_thread = threading.Thread(target=worker, daemon=True)
    worker_thread.start()

    with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as server:
        server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        server.bind((host, port))
        server.settimeout(0.25)
        print(f"Listening for authenticated cobot telemetry on udp://{host}:{port} (non-blocking worker queue active)")
        try:
            while True:
                try:
                    data, address = server.recvfrom(65_535)
                    try:
                        packet_queue.put_nowait((data, address))
                    except queue.Full:
                        print("Warning: UDP packet queue full, dropping packet", file=sys.stderr)
                except socket.timeout:
                    sentry.check_health()
        finally:
            stop_worker.set()
            worker_thread.join(timeout=1.0)


def run_stdin(sentry: CobotSentry, authenticator: PacketAuthenticator) -> None:
    print("Reading one authenticated JSON telemetry packet per stdin line.")
    for line in sys.stdin:
        if not line.strip():
            continue
        try:
            healthy = decode_and_process(line.encode("utf-8"), sentry, authenticator)
            print(json.dumps({"healthy": healthy, "state": sentry.safe_stop.state.value}))
        except TelemetryError as exc:
            print(json.dumps({"error": str(exc)}))
        sentry.check_health()


def _signed_packet(key: bytes, sequence: int, sensor: str, **values: Any) -> bytes:
    packet: dict[str, Any] = {
        "schema_version": 1,
        "robot_id": "COBOT-SELF-TEST",
        "sequence": sequence,
        "timestamp": time.time(),
        "sensor": sensor,
        **values,
    }
    packet["signature"] = hmac.new(key, canonical_packet(packet), hashlib.sha256).hexdigest()
    return json.dumps(packet).encode("utf-8")


def run_test_mode(config: Mapping[str, Any]) -> None:
    robot_id = "COBOT-SELF-TEST"
    control = AdvisoryCobotControl()
    key = load_hmac_key(DEMO_KEY_B64)
    sentry = create_sentry(config, robot_id, control, supervisor_key=key)
    authenticator = PacketAuthenticator(key, robot_id, max_age_s=5.0, window_size=128)

    # 1. Nominal packet
    nominal = _signed_packet(
        key, 1, "ODOMETRY", frame="map", position_m=[1.0, 1.0], yaw_rad=0.0,
        linear_velocity_m_s=0.2, angular_velocity_rad_s=0.0,
    )
    assert decode_and_process(nominal, sentry, authenticator)

    # 2. Out-of-order packet within sliding window (packet 3 arrives before 2)
    packet_3 = _signed_packet(
        key, 3, "ODOMETRY", frame="map", position_m=[1.2, 1.0], yaw_rad=0.0,
        linear_velocity_m_s=0.2, angular_velocity_rad_s=0.0,
    )
    assert decode_and_process(packet_3, sentry, authenticator)

    packet_2 = _signed_packet(
        key, 2, "ODOMETRY", frame="map", position_m=[1.1, 1.0], yaw_rad=0.0,
        linear_velocity_m_s=0.2, angular_velocity_rad_s=0.0,
    )
    assert decode_and_process(packet_2, sentry, authenticator), "Sliding window should accept out-of-order packet within window"

    # 3. Obstacle triggers latching stop
    obstacle = _signed_packet(key, 4, "LIDAR", frame="base_link", min_range_m=0.2)
    assert not decode_and_process(obstacle, sentry, authenticator)
    assert sentry.safe_stop.is_stopped
    assert control.last_request is not None

    # 4. Authenticated supervisor clearance
    clearance_token = hmac.new(key, f"{robot_id}:{sentry.safe_stop.state.value}".encode("utf-8"), hashlib.sha256).hexdigest()
    clear_pkt = {
        "schema_version": 1,
        "robot_id": robot_id,
        "sequence": 5,
        "timestamp": time.time(),
        "action": "CLEAR_SAFE_STOP",
        "token": clearance_token,
    }
    clear_pkt["signature"] = hmac.new(key, canonical_packet(clear_pkt), hashlib.sha256).hexdigest()
    assert decode_and_process(json.dumps(clear_pkt).encode("utf-8"), sentry, authenticator)
    assert not sentry.safe_stop.is_stopped

    print("[PASS] authenticated cobot telemetry, sliding replay window, containment latch, and supervisor clearance verified.")


def load_trusted_route(sentry: CobotSentry, path: Path) -> None:
    try:
        route = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(route, list):
            raise ValueError("route must be an array")
        sentry.load_trusted_route(route)
    except (OSError, ValueError, TypeError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"cannot load trusted route {path}: {exc}") from exc


def main() -> None:
    args = parse_arguments()
    config = load_config(args.config)
    if args.mode == "test":
        run_test_mode(config)
        return

    robot_id = args.robot_id or config["cobot"]["robot_id"]
    encoded_key = os.getenv(args.hmac_key_env)
    if not encoded_key:
        raise RuntimeError(f"{args.hmac_key_env} must contain a base64-encoded 32-byte HMAC key")

    # Load supervisor clearance key if configured
    sup_env = args.supervisor_key_env or config["security_monitors"].get("supervisor_clearance_key_env")
    supervisor_key = None
    if sup_env and os.getenv(sup_env):
        supervisor_key = load_hmac_key(os.getenv(sup_env, ""))

    window_size = int(config["security_monitors"].get("replay_window_size", 128))
    control = create_control(args, config)

    try:
        sentry = create_sentry(
            config,
            robot_id,
            control,
            supervisor_key=supervisor_key,
            persistence_path=args.state_persistence,
        )
        if args.trusted_route:
            load_trusted_route(sentry, args.trusted_route)

        authenticator = PacketAuthenticator(
            load_hmac_key(encoded_key),
            robot_id,
            float(config["connection"]["telemetry_max_age_s"]),
            window_size=window_size,
        )
        port = args.udp_port if args.udp_port is not None else int(config["connection"]["telemetry_udp_port"])
        if args.mode == "udp":
            run_udp(args.udp_host, port, sentry, authenticator)
        else:
            run_stdin(sentry, authenticator)
    finally:
        control.close()


if __name__ == "__main__":
    main()
