#!/usr/bin/env python3
"""SensorSentry companion-computer runner.

Live modes accept only schema-v1, HMAC-authenticated telemetry and normalize it
to the EKF's local NED measurement contract. Flight actions are off by default
and require both ``--enable-flight-actions`` and a healthy MAVLink connection.
"""

from __future__ import annotations

import argparse
import json
import os
import socket
import sys
import time

import numpy as np

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from ekf_fusion.sensor_models import gnss_default_R, imu_default_R
from flight_control import AdvisoryFlightControl, ContingencyAction, MavlinkFlightControl
from hal import CANBusAdapter, HardwareBus, SensorReading, SimulatedBus, UARTAdapter
from sensor_sentry import SensorSentry
from telemetry import HomeLocation, PacketAuthenticator, TelemetryError, parse_packet


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="SensorSentry secure companion-computer runner")
    parser.add_argument("--mode", choices=["udp", "serial", "test"], default="udp")
    parser.add_argument("--drone-id", default="DRONE-ALPHA-01")
    parser.add_argument("--home-lat-deg", type=float, help="Mission home WGS-84 latitude (degrees)")
    parser.add_argument("--home-lon-deg", type=float, help="Mission home WGS-84 longitude (degrees)")
    parser.add_argument("--home-alt-m", type=float, help="Mission home WGS-84 ellipsoid altitude (metres)")
    parser.add_argument("--udp-port", type=int, default=5000)
    parser.add_argument("--udp-allow-host", action="append", default=[],
                        help="Optional source-IP allowlist; may be supplied more than once")
    parser.add_argument("--input-serial-port", default="/dev/ttyUSB0" if os.name != "nt" else "COM3")
    parser.add_argument("--input-baud", type=int, default=115200)
    parser.add_argument("--hmac-key-env", default="SENSORSENTRY_UDP_HMAC_KEY",
                        help="Environment variable holding a base64 32-byte HMAC key")
    parser.add_argument("--packet-max-age-s", type=float, default=2.0)
    parser.add_argument("--sensor-timeout-s", type=float, default=3.0)
    parser.add_argument("--bus-type", choices=["simulated", "uart", "can"], default="simulated")
    parser.add_argument("--alert-serial-port", default=None,
                        help="Dedicated alert UART. It must not be the input telemetry port.")
    parser.add_argument("--alert-baud", type=int, default=115200)
    parser.add_argument("--can-interface", default="can0")
    parser.add_argument("--flight-link", default=None,
                        help="MAVLink command link, e.g. udpin:0.0.0.0:14550")
    parser.add_argument("--mavlink-signing-key-env", default="SENSORSENTRY_MAVLINK_SIGNING_KEY",
                        help="Environment variable holding the MAVLink-2 signing key")
    parser.add_argument("--mavlink-signing-state-path", default=None,
                        help="Durable, access-controlled path for MAVLink signing timestamp state")
    parser.add_argument("--ml-weights", default=None,
                        help="Path to a trained, aircraft-validated LSTM-AE weights file; disabled if omitted")
    parser.add_argument("--enable-flight-actions", action="store_true",
                        help="Allow only HOLD/RTL/LAND requests after MAVLink ACK; disabled by default")
    return parser.parse_args()


def require_live_configuration(args: argparse.Namespace) -> tuple[HomeLocation, PacketAuthenticator]:
    if args.home_lat_deg is None or args.home_lon_deg is None or args.home_alt_m is None:
        raise SystemExit("live modes require --home-lat-deg, --home-lon-deg, and --home-alt-m")
    secret = os.environ.get(args.hmac_key_env)
    if not secret:
        raise SystemExit(f"live modes require a HMAC key in environment variable {args.hmac_key_env}")
    if args.sensor_timeout_s <= 0:
        raise SystemExit("--sensor-timeout-s must be positive")
    try:
        home = HomeLocation(args.home_lat_deg, args.home_lon_deg, args.home_alt_m)
        authenticator = PacketAuthenticator(PacketAuthenticator.decode_key(secret), args.packet_max_age_s)
    except TelemetryError as exc:
        raise SystemExit(f"invalid live configuration: {exc}") from exc
    return home, authenticator


def create_hardware_bus(args: argparse.Namespace) -> HardwareBus:
    if args.bus_type == "simulated":
        return SimulatedBus(verbose=False)
    if args.bus_type == "uart":
        if not args.alert_serial_port:
            raise SystemExit("--bus-type uart requires a dedicated --alert-serial-port")
        if args.alert_serial_port == args.input_serial_port:
            raise SystemExit("input telemetry and alert output must use different serial ports")
        bus = UARTAdapter(port=args.alert_serial_port, baudrate=args.alert_baud)
    else:
        bus = CANBusAdapter(interface=args.can_interface)
    if not bus.is_connected():
        bus.close()
        raise SystemExit("configured alert bus is unavailable; refusing buffered/degraded output")
    return bus


def create_flight_control(args: argparse.Namespace):
    if not args.enable_flight_actions:
        return AdvisoryFlightControl()
    if not args.flight_link:
        raise SystemExit("--enable-flight-actions requires --flight-link")
    if not args.mavlink_signing_state_path:
        raise SystemExit("--enable-flight-actions requires --mavlink-signing-state-path")
    secret = os.environ.get(args.mavlink_signing_key_env)
    if not secret:
        raise SystemExit(
            f"--enable-flight-actions requires a MAVLink signing key in {args.mavlink_signing_key_env}"
        )
    signing_key = PacketAuthenticator.decode_key(secret)
    if len(signing_key) != 32:
        raise SystemExit("MAVLink signing key must decode to exactly 32 bytes")
    controller = MavlinkFlightControl(args.flight_link, signing_key, args.mavlink_signing_state_path)
    try:
        controller.connect()
    except Exception as exc:
        raise SystemExit(f"flight-control link failed health check: {exc}") from exc
    return controller


def covariance_for(reading: SensorReading) -> np.ndarray:
    if reading.sensor_type == "IMU":
        return imu_default_R()
    if reading.sensor_type == "GNSS":
        return gnss_default_R()
    raise TelemetryError(f"no covariance configured for {reading.sensor_type}")


def decode_and_process(raw: bytes, sentry: SensorSentry, home: HomeLocation,
                       authenticator: PacketAuthenticator, drone_id: str) -> None:
    if len(raw) > 4096:
        raise TelemetryError("packet exceeds 4096-byte limit")
    try:
        packet = json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise TelemetryError("packet is not valid UTF-8 JSON") from exc
    if not isinstance(packet, dict):
        raise TelemetryError("packet must be a JSON object")
    authenticator.verify(packet)
    reading = parse_packet(packet, home, drone_id)
    sentry.process_sensor_reading(reading, covariance_for(reading))


def _print_status(sentry: SensorSentry, packet_count: int, rejected_count: int) -> None:
    position = sentry.get_current_position()
    isolation = "GNSS_QUARANTINED" if sentry.saarm.is_quarantined("GNSS") else "NOMINAL"
    ack = "ACK" if sentry.last_contingency_acknowledged else "NO_ACTION_ACK"
    print(
        f"[STATUS] {isolation} | N={position[0]:+.1f} E={position[1]:+.1f} D={position[2]:+.1f} m "
        f"| packets={packet_count} rejected={rejected_count} | contingency={ack}",
        flush=True,
    )


def run_udp_mode(args: argparse.Namespace, sentry: SensorSentry, home: HomeLocation,
                 authenticator: PacketAuthenticator) -> None:
    allowed_hosts = set(args.udp_allow_host)
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    sock.bind(("0.0.0.0", args.udp_port))
    sock.settimeout(0.5)
    print(f"[READY] Secure UDP telemetry on 0.0.0.0:{args.udp_port}; HMAC and replay checks enabled.")
    last_packet_at = time.monotonic()
    last_status_at = last_packet_at
    packet_count = rejected_count = 0
    stale_handled = False
    try:
        while True:
            try:
                raw, address = sock.recvfrom(4097)
                if allowed_hosts and address[0] not in allowed_hosts:
                    raise TelemetryError(f"source {address[0]} is not in the UDP allowlist")
                decode_and_process(raw, sentry, home, authenticator, args.drone_id)
                packet_count += 1
                last_packet_at = time.monotonic()
                stale_handled = False
            except socket.timeout:
                pass
            except TelemetryError as exc:
                rejected_count += 1
                print(f"[REJECTED] {exc}", flush=True)

            now = time.monotonic()
            if not stale_handled and now - last_packet_at > args.sensor_timeout_s:
                stale_handled = True
                sentry._request_contingency(time.time(), ContingencyAction.HOLD,
                                             "Authenticated navigation telemetry timed out")
                print("[STALE] telemetry timeout; contingency HOLD requested", flush=True)
            if now - last_status_at >= 1.0:
                _print_status(sentry, packet_count, rejected_count)
                packet_count = rejected_count = 0
                last_status_at = now
    except KeyboardInterrupt:
        print("[STOP] UDP listener stopped")
    finally:
        sock.close()


def run_serial_mode(args: argparse.Namespace, sentry: SensorSentry, home: HomeLocation,
                    authenticator: PacketAuthenticator) -> None:
    try:
        import serial
    except ImportError as exc:
        raise SystemExit("pyserial is required for --mode serial") from exc
    try:
        serial_input = serial.Serial(args.input_serial_port, args.input_baud, timeout=0.5)
    except Exception as exc:
        raise SystemExit(f"cannot open input serial port: {exc}") from exc
    print(f"[READY] Secure serial telemetry on {args.input_serial_port}; HMAC and replay checks enabled.")
    last_packet_at = time.monotonic()
    stale_handled = False
    try:
        while True:
            raw = serial_input.readline().strip()
            if raw:
                try:
                    decode_and_process(raw, sentry, home, authenticator, args.drone_id)
                    last_packet_at = time.monotonic()
                    stale_handled = False
                except TelemetryError as exc:
                    print(f"[REJECTED] {exc}", flush=True)
            if not stale_handled and time.monotonic() - last_packet_at > args.sensor_timeout_s:
                stale_handled = True
                sentry._request_contingency(time.time(), ContingencyAction.HOLD,
                                             "Authenticated navigation telemetry timed out")
                print("[STALE] telemetry timeout; contingency HOLD requested", flush=True)
    except KeyboardInterrupt:
        print("[STOP] serial listener stopped")
    finally:
        serial_input.close()


def run_test_mode(sentry: SensorSentry) -> None:
    """Deterministic software test only; it is intentionally not called HIL."""
    print("[TEST] Running deterministic software integrity test (not hardware-in-the-loop).")
    sentry.initialize_state(np.zeros(3), 0.0)
    t, dt = time.time(), 0.02
    for index in range(100):
        t += dt
        sentry.process_sensor_reading(SensorReading(t, np.array([0.0, 0.0]), "IMU", True), imu_default_R())
        if index % 10 == 0:
            sentry.process_sensor_reading(
                SensorReading(t, np.array([index * dt, 0.0, 0.0, 1.0, 0.0, 0.0]), "GNSS", True),
                gnss_default_R(),
            )
    if sentry.saarm.is_quarantined("GNSS"):
        raise RuntimeError("false GNSS quarantine in nominal software test")
    for index in range(150):
        t += dt
        sentry.process_sensor_reading(SensorReading(t, np.array([0.0, 0.0]), "IMU", True), imu_default_R())
        if index % 10 == 0:
            sentry.process_sensor_reading(
                SensorReading(t, np.array([2.0 + index * dt, 25.0, 0.0, 1.0, 5.0, 0.0]), "GNSS", True),
                gnss_default_R(),
            )
        if sentry.saarm.is_quarantined("GNSS"):
            print("[PASS] nominal data accepted; spoofed GNSS quarantined; advisory contingency recorded.")
            return
    raise RuntimeError("spoofed GNSS was not quarantined in software test")


def main() -> None:
    args = parse_arguments()
    bus = create_hardware_bus(args)
    flight_control = create_flight_control(args)
    sentry = SensorSentry(
        args.drone_id, bus, flight_control=flight_control, aircraft_mode=True,
        ml_weights_path=args.ml_weights,
    )
    try:
        if args.mode == "test":
            run_test_mode(sentry)
            return
        home, authenticator = require_live_configuration(args)
        sentry.initialize_state(np.zeros(3), 0.0)
        if args.mode == "udp":
            run_udp_mode(args, sentry, home, authenticator)
        else:
            run_serial_mode(args, sentry, home, authenticator)
    finally:
        flight_control.close()
        bus.close()


if __name__ == "__main__":
    main()
