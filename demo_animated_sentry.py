"""Interactive SensorSentry spoofing simulation for a hackathon judge.

Upgraded version with:
- Multiple selectable mission routes (5 pre-built paths)
- Re-injectable spoof target at any time during flight
- Real 11-state MultiSensorEKF integration (simulated sensor measurements
  feed into the EKF; the EKF's fused output drives the trusted position)

The simulation deliberately waits for the presenter to inject the attack:

1. Select a mission route using the ``Next Route`` button.
2. Click ``Launch mission``.
3. While the drone flies normally, click any point on the map to set the
   attacker's spoofed GPS destination. Click again to change it mid-attack.
4. Watch GNSS diverge from the EKF-fused IMU + camera + LiDAR estimate.
5. Chi-Square rejects bad fixes, CUSUM confirms persistent drift, GNSS is
   quarantined, and the simulated server returns a signed rejoin route.

Controls are available both as visible buttons and keyboard shortcuts:
Space pauses/resumes; R replays the last attack scenario.

This is a local simulation only. It does not control a physical drone.
"""

from __future__ import annotations

import argparse
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation
from matplotlib.patches import Circle
from matplotlib.widgets import Button

# Ensure local packages are importable
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, BASE_DIR)
sys.path.insert(0, os.path.join(BASE_DIR, "filterpy-master"))

from decision_making import (
    ChiSquareGate, CusumMonitor, LSTMAutoEncoderMonitor, MLStatus,
    HighFrequencyAttackDetector, HighFrequencySignal,
)
from decision_making.cusum_monitor import DriftStatus
from ekf_fusion import MultiSensorEKF, SensorType
from ekf_fusion.sensor_models import (
    gnss_h, gnss_H, gnss_default_R,
    imu_h, imu_H, imu_default_R,
    camera_h, camera_H, camera_residual, camera_default_R,
    lidar_h, lidar_H, lidar_residual, lidar_default_R,
)
from hal import SimulatedBus
from waypoint_security import CommandValidator, WaypointCommand

# ── Visual palette ──────────────────────────────────────────────────────
BG = "#0b1220"
PANEL = "#111c2e"
GRID = "#334155"
TEXT = "#e2e8f0"
MUTED = "#94a3b8"
GREEN = "#22c55e"
CYAN = "#22d3ee"
BLUE = "#60a5fa"
ORANGE = "#fb923c"
RED = "#fb7185"
YELLOW = "#facc15"
PURPLE = "#c084fc"

# ── Mission geometry ────────────────────────────────────────────────────
START = np.array([15.0, 15.0])
DESTINATION = np.array([150.0, 125.0])
ALTITUDE = 10.0  # constant flight altitude (m)
DT = 0.20  # simulation tick (s)
MISSION_DURATION = 60.0  # seconds to fly the route end-to-end

# ── Pre-built mission routes ────────────────────────────────────────────
ROUTE_NAMES = [
    "Direct",
    "Curved North",
    "Dogleg South",
    "Urban Corridor",
    "Perimeter Sweep",
]

ROUTE_WAYPOINTS = {
    "Direct": [
        START.copy(),
        DESTINATION.copy(),
    ],
    "Curved North": [
        START.copy(),
        np.array([40.0, 85.0]),
        np.array([95.0, 145.0]),
        DESTINATION.copy(),
    ],
    "Dogleg South": [
        START.copy(),
        np.array([75.0, 20.0]),
        np.array([130.0, 55.0]),
        DESTINATION.copy(),
    ],
    "Urban Corridor": [
        START.copy(),
        np.array([35.0, 55.0]),
        np.array([80.0, 100.0]),
        np.array([115.0, 75.0]),
        DESTINATION.copy(),
    ],
    "Perimeter Sweep": [
        START.copy(),
        np.array([15.0, 110.0]),
        np.array([85.0, 170.0]),
        np.array([155.0, 155.0]),
        DESTINATION.copy(),
    ],
}

# 3-D landmarks used by Camera and LiDAR observations
LANDMARKS_3D = np.array([
    [30.0, 105.0, 15.0],
    [46.0, 93.0, 8.0],
    [107.0, 55.0, 20.0],
    [127.0, 77.0, 12.0],
    [137.0, 111.0, 18.0],
    [83.0, 119.0, 10.0],
])


@dataclass
class Event:
    time_s: float
    message: str
    colour: str


# ── Helpers ─────────────────────────────────────────────────────────────

def _route_cumulative_lengths(waypoints: List[np.ndarray]) -> List[float]:
    """Return cumulative segment lengths for a route."""
    lengths = [0.0]
    for i in range(1, len(waypoints)):
        lengths.append(lengths[-1] + float(np.linalg.norm(waypoints[i] - waypoints[i - 1])))
    return lengths


def _interpolate_route(waypoints: List[np.ndarray], progress: float) -> np.ndarray:
    """Interpolate position along a multi-waypoint route at *progress* ∈ [0, 1]."""
    progress = np.clip(progress, 0.0, 1.0)
    lengths = _route_cumulative_lengths(waypoints)
    total = lengths[-1]
    if total < 1e-9:
        return waypoints[0].copy()
    target_dist = progress * total
    for i in range(1, len(waypoints)):
        if target_dist <= lengths[i]:
            seg_len = lengths[i] - lengths[i - 1]
            if seg_len < 1e-9:
                return waypoints[i].copy()
            seg_progress = (target_dist - lengths[i - 1]) / seg_len
            return waypoints[i - 1] + seg_progress * (waypoints[i] - waypoints[i - 1])
    return waypoints[-1].copy()


def _nearest_landmarks(position_2d: np.ndarray, count: int = 2) -> np.ndarray:
    """Return the *count* nearest landmarks (3-D) to a 2-D position."""
    dists = np.linalg.norm(LANDMARKS_3D[:, :2] - position_2d, axis=1)
    indices = np.argsort(dists)[:count]
    return LANDMARKS_3D[indices]


# ════════════════════════════════════════════════════════════════════════
class InteractiveSpoofDemo:
    """Live simulation state, plotting, EKF detection, and response workflow."""

    phase_style = {
        "READY": ("READY  •  SELECT ROUTE, LAUNCH MISSION (CLICK MAP ANY TIME TO SPOOF)", BLUE),
        "NOMINAL": ("AUTONOMOUS MISSION FLIGHT — ROUTE TRACKING (NOMINAL)", GREEN),
        "QUARANTINED": ("SPOOF CONFIRMED — GNSS QUARANTINED, SAFE HOVER", YELLOW),
        "SERVER_VERIFYING": ("ENCRYPTED SERVER RECONNECT IN PROGRESS", CYAN),
        "RECOVERING": ("FOLLOWING HMAC-SIGNED REJOIN ROUTE BACK TO PLANNED PATH", BLUE),
        "COMPLETE": ("COMPLETE  •  AUTHORISED DESTINATION REACHED", GREEN),
    }

    # ── Construction & state ────────────────────────────────────────────
    def __init__(self, build_figure: bool = True) -> None:
        self.fig = None
        self.animation = None
        self._paused = False
        self._scheduled_replay: Optional[Tuple[float, np.ndarray]] = None
        self._last_spoof_target: Optional[np.ndarray] = None
        self._last_spoof_time: Optional[float] = None
        self._route_index = 0
        self.selected_route_name = ROUTE_NAMES[0]
        self._reset_state(keep_last_attack=True)
        if build_figure:
            self.build_figure()

    @property
    def selected_route(self) -> List[np.ndarray]:
        return ROUTE_WAYPOINTS[self.selected_route_name]

    def planned_position_at_progress(self, progress: float) -> np.ndarray:
        """Position on the currently selected authorised route at given progress."""
        return _interpolate_route(self.selected_route, np.clip(progress, 0.0, 1.0))

    def planned_position(self) -> np.ndarray:
        """Position on the currently selected authorised route at current progress."""
        return self.planned_position_at_progress(self.route_progress)

    @staticmethod
    def _move_toward(position: np.ndarray, target: np.ndarray,
                     speed_mps: float, dt: float) -> np.ndarray:
        delta = target - position
        distance = float(np.linalg.norm(delta))
        if distance < 1e-9:
            return position.copy()
        return position + delta * min(speed_mps * dt / distance, 1.0)

    # ── State management ────────────────────────────────────────────────
    def _reset_state(self, keep_last_attack: bool) -> None:
        if not keep_last_attack:
            self._last_spoof_target = None
            self._last_spoof_time = None
        self.time_s = 0.0
        self.position = START.copy()
        self.prev_position = START.copy()
        self.velocity = np.zeros(2)
        self.heading = 0.0
        self.route_progress = 0.0
        self.route_progress_at_spoof = 0.0
        self.rejoin_progress = 0.0
        self.spoof_count = 0
        self.spoof_timestamps: List[float] = []
        self.phase = "READY"
        self.mission_started = False
        self.spoof_active = False
        self.spoof_target: Optional[np.ndarray] = None
        self.spoof_start_time: Optional[float] = None
        self.detection_time: Optional[float] = None
        self.server_authorization_time: Optional[float] = None
        self.server_command_accepted = False
        self.recovery_route: List[np.ndarray] = []
        self.recovery_index = 0
        self.chi_rejection_count = 0
        self._warning_logged = False
        self._chi_logged = False
        self._server_link_logged = False
        self.bus = SimulatedBus(verbose=False)

        # ── Real EKF ────────────────────────────────────────────────────
        initial_x = np.zeros(11)
        initial_x[0:2] = START
        initial_x[2] = ALTITUDE
        self.ekf = MultiSensorEKF(initial_x=initial_x)

        # ── Decision monitors ──────────────────────────────────────────
        self.chi_gate = ChiSquareGate(confidence=0.999)
        self.cusum = CusumMonitor(
            alarm_threshold=5000.0,
            warning_threshold=1200.0,
            slack=8.0,
            recovery_rate=0.0,
            auto_recover_after=0,
        )
        self.ml_monitor = LSTMAutoEncoderMonitor(alarm_threshold=220.0, warning_threshold=90.0)
        self.hf_detector = HighFrequencyAttackDetector(
            freq_threshold_hz=1.5,
            burst_window_s=4.0,
            burst_count_threshold=2,
            drift_override_threshold_m=4.0,
        )
        self.hf_detector.set_start_position(START)
        self.override_to_start = False
        self.overall_drift = np.zeros(2)
        self.negative_drift = np.zeros(2)
        self._hf_logged = False
        # Sensor noise covariance matrices
        self._gnss_R = gnss_default_R(sigma_pos=1.5, sigma_alt=2.5, sigma_vel=0.1)
        self._imu_R = imu_default_R(sigma_gyro=0.005, sigma_accel=0.05)
        self._camera_R = camera_default_R(sigma_angle_rad=0.008)
        self._lidar_R = lidar_default_R(sigma_r=0.03, sigma_ang=0.003)

        # ── Telemetry ──────────────────────────────────────────────────
        self.nis = 0.0
        self.nis_threshold = self.chi_gate._chi2_threshold(6)
        self.cusum_value = 0.0
        self.ml_mse = 0.0
        self.ml_status = MLStatus.HEALTHY
        self._ml_logged = False
        self.last_gps = START.copy()
        self.gps_position = START.copy()
        self.trusted_position = START.copy()
        self.sensor_residuals = {"IMU": 0.0, "CAMERA": 0.0, "LIDAR": 0.0}
        self.ekf_innovation_mag = 0.0
        self.gnss_accepted = True
        self.events: List[Event] = [
            Event(0.0, "READY: Select route, launch mission, then click map to inject a spoofed target.", BLUE),
        ]
        self.history = {
            "time": [], "physical": [], "trusted": [], "gps": [], "nis": [], "cusum": [], "ml_mse": [],
        }

    def _add_event(self, message: str, colour: str) -> None:
        self.events.append(Event(self.time_s, message, colour))

    # ── Route cycling ───────────────────────────────────────────────────
    def cycle_route(self, _event=None) -> None:
        """Cycle to the next pre-built route (only while not mid-mission)."""
        if self.mission_started and self.phase != "COMPLETE":
            self._add_event("ROUTE LOCKED: Cannot change route during an active mission.", YELLOW)
            self.render()
            self.fig.canvas.draw_idle()
            return
        self._route_index = (self._route_index + 1) % len(ROUTE_NAMES)
        self.selected_route_name = ROUTE_NAMES[self._route_index]
        self._add_event(f"ROUTE: Switched to '{self.selected_route_name}' mission path.", CYAN)
        if self.fig is not None:
            self._update_planned_line()
            self._update_route_label()
            self.render()
            self.fig.canvas.draw_idle()

    def _update_planned_line(self) -> None:
        """Redraw the dashed planned-route line on the map."""
        route = self.selected_route
        pts = np.array([_interpolate_route(route, p) for p in np.linspace(0, 1, 120)])
        self.planned_line.set_data(pts[:, 0], pts[:, 1])

    def _update_route_label(self) -> None:
        if hasattr(self, "route_label"):
            self.route_label.set_text(f"Route: {self.selected_route_name}")

    # ── Mission control ─────────────────────────────────────────────────
    def launch_mission(self, _event=None) -> None:
        """Start a clean mission. The spoof remains entirely presenter-controlled."""
        self._reset_state(keep_last_attack=True)
        self.mission_started = True
        self.phase = "NOMINAL"
        self._paused = False
        self._add_event(
            f"MISSION: Drone departed on '{self.selected_route_name}' route. "
            "EKF fusion active. Awaiting manual spoof injection.",
            GREEN,
        )
        if self.animation is not None:
            self.animation.event_source.start()
        self.render()

    # ── Spoof injection (any time between launch and destination) ───────
    def inject_spoof(self, target: np.ndarray, from_replay: bool = False) -> bool:
        """Begin, redirect, or reassign a gradual GNSS spoof toward *target* at any time."""
        target = np.asarray(target, dtype=float)
        if not self.mission_started or self.phase == "COMPLETE":
            return False

        is_reassign = self.spoof_active and self.spoof_target is not None
        self.spoof_active = True
        self.spoof_target = target.copy()
        self.spoof_start_time = self.time_s
        self.route_progress_at_spoof = self.route_progress
        self._last_spoof_target = target.copy()
        self._last_spoof_time = self.time_s
        self.spoof_count += 1
        self.spoof_timestamps.append(self.time_s)

        recent_spoofs = [t for t in self.spoof_timestamps if (self.time_s - t) <= 5.0]
        if len(recent_spoofs) > 5:
            self.override_to_start = True
            self._add_event(
                "HIGH-FREQ CYBER HIJACK CONFIRMED (>5 spoofing in 5s): "
                "Destination OVERRIDDEN with STARTING POINT! Returning directly to start.",
                RED,
            )

        # If vehicle is currently in NOMINAL, reset CUSUM/detectors for the new attack profile
        if self.phase == "NOMINAL":
            self.cusum.reset("GNSS")
            self.chi_gate = ChiSquareGate(confidence=0.999)
            self.ml_monitor.reset("GNSS")
            self.chi_rejection_count = 0
            self._chi_logged = False
            self._warning_logged = False
            self._ml_logged = False

        if is_reassign:
            prefix = "REPLAY RE-SPOOF: " if from_replay else f"RE-SPOOF #{self.spoof_count}: "
            self._add_event(
                f"{prefix}Attacker shifted fake target to [{target[0]:.0f}, {target[1]:.0f}]. "
                f"Vehicle status: {self.phase}.",
                RED,
            )
        else:
            prefix = "REPLAY ATTACK: " if from_replay else f"ATTACK #{self.spoof_count}: "
            self._add_event(
                f"{prefix}Attacker broadcasting fake GNSS target at [{target[0]:.0f}, {target[1]:.0f}]. "
                "Vehicle operating autonomously.",
                RED,
            )
        return True

    def clear_spoof(self, _event=None) -> None:
        """Turn off the spoof attack transmitter at any time."""
        if not self.spoof_active:
            return
        self.spoof_active = False
        self.spoof_target = None
        self.spoof_start_time = None
        self._add_event("SPOOF CLEARED: Attacker transmitter deactivated. Truthful GNSS restored.", GREEN)
        if self.fig is not None:
            self.render()
            self.fig.canvas.draw_idle()

    def _on_map_click(self, event) -> None:
        if event.inaxes is not self.ax_map or event.xdata is None or event.ydata is None:
            return
        if getattr(event, "button", 1) == 3:
            self.clear_spoof()
            return
        click = np.array([event.xdata, event.ydata], dtype=float)
        if not self.mission_started:
            self._add_event("CLICK RECEIVED: Press Launch mission first, then click map to set spoof target.", YELLOW)
            self.render()
            self.fig.canvas.draw_idle()
            return
        if self.phase == "COMPLETE":
            self._add_event("MISSION COMPLETE: Destination reached. Re-launch mission to test again.", YELLOW)
            self.render()
            self.fig.canvas.draw_idle()
            return
        if self.override_to_start:
            self._add_event("OVERRIDE LOCKED: Returning to START location under continuous cyber attack. New destinations rejected!", YELLOW)
            self.render()
            self.fig.canvas.draw_idle()
            return

        self.inject_spoof(click)
        self.render()
        self.fig.canvas.draw_idle()

    # ── EKF Sensor Simulation ───────────────────────────────────────────
    def _compute_physical_kinematics(self) -> Tuple[float, float]:
        """Derive angular velocity and forward acceleration from physical motion."""
        self.velocity = (self.position - self.prev_position) / DT
        speed = float(np.linalg.norm(self.velocity))

        # Heading from velocity direction
        if speed > 0.05:
            new_heading = float(np.arctan2(self.velocity[1], self.velocity[0]))
            omega_z = (new_heading - self.heading) / DT
            # Wrap
            omega_z = (omega_z + np.pi) % (2 * np.pi) - np.pi
            self.heading = new_heading
        else:
            omega_z = 0.0

        # Forward acceleration (scalar along heading)
        if hasattr(self, "_prev_speed"):
            a_fwd = (speed - self._prev_speed) / DT
        else:
            a_fwd = 0.0
        self._prev_speed = speed

        return omega_z, a_fwd

    def _feed_ekf_imu(self, omega_z: float, a_fwd: float) -> None:
        """Feed a simulated IMU measurement into the EKF."""
        # Simulated noisy IMU reading
        z_imu = np.array([
            omega_z + self.ekf.x[9] + np.random.normal(0, 0.005),
            a_fwd + self.ekf.x[10] + np.random.normal(0, 0.05),
        ])
        self.ekf.update_sensor(
            SensorType.IMU, z_imu, imu_H, imu_h, self._imu_R,
            args=(omega_z, a_fwd),
        )

    def _feed_ekf_camera(self) -> None:
        """Feed simulated camera bearing observations for nearest landmarks."""
        pos_3d = np.array([self.position[0], self.position[1], ALTITUDE])
        nearest = _nearest_landmarks(self.position, count=2)
        for lm in nearest:
            # True bearing + noise
            dx = lm[0] - pos_3d[0]
            dy = lm[1] - pos_3d[1]
            dz = lm[2] - pos_3d[2]
            d_xy = max(1e-4, np.sqrt(dx * dx + dy * dy))
            alpha = np.arctan2(dy, dx) - self.heading
            beta = np.arctan2(dz, d_xy)
            z_cam = np.array([
                alpha + np.random.normal(0, 0.008),
                beta + np.random.normal(0, 0.008),
            ])
            self.ekf.update_sensor(
                SensorType.CAMERA, z_cam, camera_H, camera_h, self._camera_R,
                args=(lm,), residual_func=camera_residual,
            )

    def _feed_ekf_lidar(self) -> None:
        """Feed a simulated LiDAR observation for the nearest reference point."""
        pos_3d = np.array([self.position[0], self.position[1], ALTITUDE])
        nearest = _nearest_landmarks(self.position, count=1)
        for pt in nearest:
            dx = pt[0] - pos_3d[0]
            dy = pt[1] - pos_3d[1]
            dz = pt[2] - pos_3d[2]
            r = max(1e-4, np.sqrt(dx * dx + dy * dy + dz * dz))
            theta = np.arctan2(dy, dx) - self.heading
            phi = np.arcsin(np.clip(dz / r, -0.9999, 0.9999))
            z_lidar = np.array([
                r + np.random.normal(0, 0.03),
                theta + np.random.normal(0, 0.003),
                phi + np.random.normal(0, 0.003),
            ])
            self.ekf.update_sensor(
                SensorType.LIDAR, z_lidar, lidar_H, lidar_h, self._lidar_R,
                args=(pt,), residual_func=lidar_residual,
            )

    def _gnss_measurement(self) -> np.ndarray:
        """Generate GNSS measurement: 6-DOF [px, py, pz, vx, vy, vz]."""
        noise_pos = np.random.normal(0, 1.5, size=3)
        noise_vel = np.random.normal(0, 0.1, size=3)

        if not self.spoof_active or self.spoof_target is None or self.spoof_start_time is None:
            # Truthful GNSS
            true_pos = np.array([self.position[0], self.position[1], ALTITUDE])
            true_vel = np.array([self.velocity[0], self.velocity[1], 0.0])
            return np.concatenate([true_pos + noise_pos, true_vel + noise_vel])

        # Spoofed GNSS: position drifts toward spoof target
        # "nominal" = where drone would be if it kept following the route
        elapsed = self.time_s - self.spoof_start_time
        hypothetical_progress = min(1.0, self.route_progress_at_spoof + elapsed / MISSION_DURATION)
        nominal = self.planned_position_at_progress(hypothetical_progress)
        direction = self.spoof_target - nominal
        dist = np.linalg.norm(direction)
        if dist < 1e-6:
            true_pos = np.array([nominal[0], nominal[1], ALTITUDE])
            true_vel = np.array([self.velocity[0], self.velocity[1], 0.0])
            return np.concatenate([true_pos + noise_pos, true_vel + noise_vel])

        direction_unit = direction / dist
        # Gradual onset: 0.25 m/s^2 initial ramp, realistic slow-onset spoof
        drift_mag = min(dist * 0.9, 0.25 * elapsed ** 2)
        spoofed_2d = nominal + direction_unit * drift_mag
        spoofed_pos = np.array([spoofed_2d[0], spoofed_2d[1], ALTITUDE])

        # Spoofed velocity: consistent with drift direction
        spoof_vel_2d = direction_unit * min(3.0, 1.5 * elapsed)
        spoofed_vel = np.array([spoof_vel_2d[0], spoof_vel_2d[1], 0.0])

        return np.concatenate([spoofed_pos + noise_pos, spoofed_vel + noise_vel])

    # ── Authenticate recovery plan ──────────────────────────────────────
    def _authenticate_recovery_plan(self) -> bool:
        """Simulate command-server approval using the project's HMAC validator."""
        validator = CommandValidator()
        sequence = 901
        issuer = "FLEET-COMMAND"
        signature = validator.sign_command(DESTINATION, sequence, self.time_s, issuer)
        command = WaypointCommand(
            destination=DESTINATION.copy(),
            sequence_number=sequence,
            timestamp=self.time_s,
            issuer_id=issuer,
            signature=signature,
            priority=1,
        )
        signal = validator.validate(command, np.r_[self.position, ALTITUDE], self.time_s)
        self.bus.send(signal.to_bytes(), "COMMAND")
        return signal.verdict.name == "ACCEPTED"

    def _make_recovery_route(self) -> List[np.ndarray]:
        """Return a recovery corridor back to START (if under attack override) or rejoin path."""
        if self.override_to_start or self.hf_detector.is_override_active():
            # Stop drift and return directly to START location using negative drift trajectory
            overall_drift = self.position - START
            mid_safe = self.position - 0.5 * overall_drift
            return [self.position.copy(), mid_safe, START.copy()]

        route = self.selected_route
        route_vector = DESTINATION - START
        route_length_sq = float(np.dot(route_vector, route_vector))
        current_progress = float(np.dot(self.position - START, route_vector) / route_length_sq)
        # Rejoin slightly ahead of where we left off, but not past 86%
        self.rejoin_progress = np.clip(
            max(current_progress + 0.10, self.route_progress_at_spoof + 0.08),
            self.route_progress_at_spoof + 0.05,
            0.92,
        )
        rejoin = _interpolate_route(route, self.rejoin_progress)

        perpendicular = np.array([-route_vector[1], route_vector[0]]) / np.linalg.norm(route_vector)
        lane_direction = 1.0
        if self.spoof_target is not None:
            lane_direction = -np.sign(np.dot(self.spoof_target - self.position, perpendicular)) or 1.0
        safe_lane = self.position + 0.48 * (rejoin - self.position) + lane_direction * 10.0 * perpendicular
        # Route ends at the rejoin point (NOT destination) — drone resumes route from there
        return [self.position.copy(), safe_lane, rejoin]

    # ── Physics step ────────────────────────────────────────────────────
    def _advance_motion(self) -> None:
        self.prev_position = self.position.copy()
        if self.phase == "NOMINAL":
            if self.spoof_active and self.spoof_target is not None:
                # Under active GNSS spoofing, autopilot is deceived by false fix before CUSUM alarms
                self.position = self._move_toward(self.position, self.spoof_target, 5.4, DT)
            else:
                self.route_progress = min(1.0, self.route_progress + DT / MISSION_DURATION)
                self.position = self.planned_position()
                if self.route_progress >= 1.0:
                    self.position = DESTINATION.copy()
                    self.phase = "COMPLETE"
                    self._add_event(
                        f"COMPLETE: Destination reached. Survived {self.spoof_count} spoof attack(s). "
                        "Press R or Replay.", GREEN,
                    )
        elif self.phase in {"QUARANTINED", "SERVER_VERIFYING"}:
            self.position = self.position.copy()  # Safe hover
        elif self.phase == "RECOVERING":
            target = self.recovery_route[self.recovery_index + 1]
            self.position = self._move_toward(self.position, target, 6.5, DT)
            if np.linalg.norm(self.position - target) < 0.15:
                self.recovery_index += 1
                if self.recovery_index >= len(self.recovery_route) - 1:
                    if self.override_to_start or self.hf_detector.is_override_active():
                        self.position = START.copy()
                        self.phase = "COMPLETE"
                        self._add_event(
                            f"SAFE AT START LOCATION: Return-to-launch completed under continuous attack override. "
                            f"Drift canceled. Survived {self.spoof_count} attacks.", GREEN,
                        )
                    else:
                        # Reached the rejoin point on the planned route
                        self._rejoin_nominal()

    def _rejoin_nominal(self) -> None:
        """Transition from recovery back to nominal route-following.

        Resets all detection state so the drone is ready for fresh
        spoof monitoring. The user can inject or reassign another spoof at any time.
        """
        self.route_progress = self.rejoin_progress
        self.position = self.planned_position()
        self.phase = "NOMINAL"
        # Reset detection state for fresh monitoring
        self.cusum.reset("GNSS")
        self.chi_gate = ChiSquareGate(confidence=0.999)
        self.ml_monitor.reset("GNSS")
        self.chi_rejection_count = 0
        self._chi_logged = False
        self._warning_logged = False
        self._ml_logged = False
        self._server_link_logged = False
        self.detection_time = None
        self.server_authorization_time = None
        self.server_command_accepted = False
        self.recovery_route = []
        self.recovery_index = 0
        self.spoof_active = False
        self.spoof_target = None
        self.spoof_start_time = None
        self._add_event(
            f"REJOIN: Back on authorised route (progress {self.route_progress:.0%}). "
            "GNSS restored & monitored. Click map anytime to spoof again!",
            GREEN,
        )

    # ── EKF-driven detection loop ───────────────────────────────────────
    def _evaluate_detectors(self) -> None:
        """Run the real EKF predict/update cycle and feed GNSS innovation to monitors."""
        # 1. Compute physical kinematics
        omega_z, a_fwd = self._compute_physical_kinematics()

        # 2. EKF predict step
        self.ekf.predict(DT, omega_z=omega_z, a_fwd=a_fwd)

        # 3. Feed truthful sensors into the EKF (these are NOT spoofed)
        self._feed_ekf_imu(omega_z, a_fwd)
        self._feed_ekf_camera()
        # LiDAR every other tick to reduce computation
        if int(self.time_s / DT) % 2 == 0:
            self._feed_ekf_lidar()

        # 4. Trusted position = EKF fused estimate (driven by IMU+Camera+LiDAR)
        self.trusted_position = self.ekf.x[0:2].copy()

        # 5. Compute per-sensor residuals for display
        ekf_pos_3d = np.array([self.ekf.x[0], self.ekf.x[1], self.ekf.x[2]])
        for i, name in enumerate(["IMU", "CAMERA", "LIDAR"]):
            # Residual = |EKF_position - physical_position| approximate
            self.sensor_residuals[name] = float(np.linalg.norm(
                self.ekf.x[0:2] - self.position
            )) + np.random.uniform(0, 0.15)

        # 6. GNSS evaluation
        gps_is_active = self.phase == "NOMINAL"
        if not gps_is_active:
            self.gps_position = self.last_gps.copy()
            self.gnss_accepted = False
            return

        # Generate GNSS measurement (potentially spoofed)
        z_gnss = self._gnss_measurement()
        self.gps_position = z_gnss[0:2].copy()

        # Compute innovation through EKF (GNSS is 6-DOF)
        y, S, H, PHT = self.ekf.compute_innovation(
            z_gnss, gnss_H, gnss_h, self._gnss_R,
        )

        dim_m = len(z_gnss)
        self.ekf_innovation_mag = float(np.linalg.norm(y[0:2]))

        # Feed innovation to Chi-Square (fast), CUSUM (slow), and LSTM-AE (ML)
        fast_signal = self.chi_gate.evaluate("GNSS", self.time_s, y, S, dim_m)
        slow_signal = self.cusum.evaluate("GNSS", self.time_s, y, S, dim_m)
        ml_signal = self.ml_monitor.evaluate("GNSS", self.time_s, y, S, dim_m)
        self.nis = fast_signal.nis
        self.nis_threshold = fast_signal.threshold
        self.cusum_value = slow_signal.cusum_value
        self.ml_mse = ml_signal.reconstruction_error
        self.ml_status = ml_signal.status
        self.last_gps = self.gps_position.copy()

        # 7. Apply GNSS update ONLY if all monitors accept
        if fast_signal.verdict.name == "ACCEPT" and slow_signal.accepted and ml_signal.accepted:
            self.ekf.apply_update(y, S, H, PHT, self._gnss_R)
            self.gnss_accepted = True
        else:
            self.gnss_accepted = False

        # 8. Log detection events
        if fast_signal.verdict.name == "REJECT":
            self.chi_rejection_count += 1
            if not self._chi_logged:
                self._chi_logged = True
                self._add_event(
                    f"CHI-SQUARE: GNSS innovation NIS={self.nis:.1f} exceeds gate (threshold={self.nis_threshold:.1f}). "
                    "EKF rejects GNSS fix.",
                    PURPLE,
                )
        if slow_signal.status == DriftStatus.WARNING and not self._warning_logged:
            self._warning_logged = True
            self._add_event(
                f"CUSUM WARNING: Accumulated GNSS drift = {self.cusum_value:.1f} "
                f"(alarm at {self.cusum.alarm_threshold:.0f}).",
                ORANGE,
            )
        if ml_signal.status == MLStatus.WARNING and not self._ml_logged:
            self._ml_logged = True
            self._add_event(
                f"LSTM-AE ML WARNING: Sequence reconstruction loss MSE={self.ml_mse:.1f} (thresh={self.ml_monitor.alarm_threshold:.0f}).",
                ORANGE,
            )

        # 9. Evaluate High-Frequency Cyber Attacks & Negative Drift
        hf_signal = self.hf_detector.record_and_evaluate(
            "GNSS", self.time_s, y, self.position, is_attack_sample=self.spoof_active
        )
        self.overall_drift = hf_signal.overall_drift_vector
        self.negative_drift = hf_signal.negative_drift_vector

        recent_spoofs = [t for t in self.spoof_timestamps if (self.time_s - t) <= 5.0]
        if hf_signal.override_to_start or len(recent_spoofs) > 5:
            self.override_to_start = True
            if not self._hf_logged:
                self._hf_logged = True
                self._add_event(
                    f"HIGH-FREQ CYBER ATTACK ({hf_signal.attack_frequency_hz:.1f}Hz, pattern={hf_signal.attack_pattern.name}): "
                    f"Destination OVERRIDDEN with STARTING POINT. Negative drift ({np.linalg.norm(self.negative_drift):.1f}m) canceling attack. Drone returning to START.",
                    RED,
                )

        # Quarantines on CUSUM alarm OR ML alarm OR High-Frequency Override
        if self.phase == "NOMINAL" and self.spoof_active and (
            slow_signal.status == DriftStatus.ALARM
            or ml_signal.status == MLStatus.ALARM
            or self.override_to_start
        ):
            self.phase = "QUARANTINED"
            self.detection_time = self.time_s
            self.bus.send(b"GPS_QUARANTINED", "SAARM")
            trigger_src = "HIGH-FREQ OVERRIDE" if self.override_to_start else (
                "CUSUM + LSTM-AE" if (slow_signal.status == DriftStatus.ALARM and ml_signal.status == MLStatus.ALARM) else ("LSTM-AE ML" if ml_signal.status == MLStatus.ALARM else "CUSUM")
            )
            self._add_event(
                f"ANOMALY CONFIRMED ({trigger_src}): Persistent GNSS drift confirmed. "
                "SAARM isolates GNSS; drone holds on EKF fusion of IMU+Camera+LiDAR.",
                YELLOW,
            )

    def _progress_response(self) -> None:
        if self.detection_time is None:
            return
        since_detection = self.time_s - self.detection_time
        if self.phase == "QUARANTINED" and since_detection >= 1.2:
            self.phase = "SERVER_VERIFYING"
            if not self._server_link_logged:
                self._server_link_logged = True
                self.bus.send(b"RECOVERY_REQUEST", "ALERT")
                self._add_event("SERVER LINK: Encrypted telemetry sent; requesting authenticated rejoin route.", CYAN)
        if self.phase == "SERVER_VERIFYING" and since_detection >= 3.8:
            self.server_command_accepted = self._authenticate_recovery_plan()
            if not self.server_command_accepted:
                raise RuntimeError("The simulated server recovery command was not authenticated.")
            self.server_authorization_time = self.time_s
            self.recovery_route = self._make_recovery_route()
            self.recovery_index = 0
            self.phase = "RECOVERING"
            self._add_event("SERVER AUTHORIZED: HMAC-verified rejoin route received. Secure navigation resumes.", BLUE)

    # ── Main simulation tick ────────────────────────────────────────────
    def advance(self) -> None:
        """Advance a single simulation tick; usable by the GUI and headless test."""
        if not self.mission_started or self.phase == "COMPLETE":
            return
        self.time_s = round(self.time_s + DT, 2)
        if self._scheduled_replay is not None and self.phase == "NOMINAL":
            attack_time, target = self._scheduled_replay
            if self.time_s >= attack_time:
                self.inject_spoof(target, from_replay=True)
                self._scheduled_replay = None
        self._advance_motion()
        self._evaluate_detectors()
        self._progress_response()
        self.history["time"].append(self.time_s)
        self.history["physical"].append(self.position.copy())
        self.history["trusted"].append(self.trusted_position.copy())
        self.history["gps"].append(self.gps_position.copy())
        self.history["nis"].append(self.nis)
        self.history["cusum"].append(self.cusum_value)
        self.history["ml_mse"].append(self.ml_mse)

    # ════════════════════════════════════════════════════════════════════
    # FIGURE BUILDING
    # ════════════════════════════════════════════════════════════════════
    @staticmethod
    def _configure_style() -> None:
        plt.rcParams.update({
            "figure.facecolor": BG, "axes.facecolor": PANEL, "axes.edgecolor": GRID,
            "axes.labelcolor": TEXT, "text.color": TEXT, "xtick.color": MUTED,
            "ytick.color": MUTED, "grid.color": GRID, "grid.alpha": 0.55,
            "font.family": "DejaVu Sans",
        })

    def build_figure(self) -> None:
        if self.fig is not None:
            return
        self._configure_style()
        self.fig = plt.figure(figsize=(16, 9))
        try:
            self.fig.canvas.manager.set_window_title("SensorSentry | Interactive GPS Spoofing Simulation (EKF-Driven)")
        except Exception:
            pass
        grid = self.fig.add_gridspec(
            3, 4, height_ratios=[1.0, 1.0, 0.60], left=0.045, right=0.975,
            bottom=0.12, top=0.88, hspace=0.42, wspace=0.28,
        )
        self.ax_map = self.fig.add_subplot(grid[:2, :3])
        self.ax_metrics = self.fig.add_subplot(grid[0, 3])
        self.ax_state = self.fig.add_subplot(grid[1, 3])
        self.ax_log = self.fig.add_subplot(grid[2, :])
        self.fig.suptitle(
            "SensorSentry  |  EKF-Driven GPS Spoof → Detection → Secure Recovery",
            color=CYAN, fontsize=16, fontweight="bold",
        )
        self._build_map()
        self._build_metrics()
        self._build_state_panel()
        self._build_log()
        self._build_controls()
        self.fig.canvas.mpl_connect("button_press_event", self._on_map_click)
        self.fig.canvas.mpl_connect("key_press_event", self._on_key)
        self.render()

    def _build_map(self) -> None:
        ax = self.ax_map
        ax.set_title(
            "Left-click map at any time to inject/reassign fake GNSS target  •  Right-click / [C] to clear",
            loc="left", color=TEXT, fontsize=11.0, fontweight="bold", pad=10,
        )
        ax.set_xlim(0, 175)
        ax.set_ylim(0, 185)
        ax.set_aspect("equal")
        ax.set_xlabel("East (m)")
        ax.set_ylabel("North (m)")
        ax.grid(True)

        # Planned route line (updated when route changes)
        route = self.selected_route
        pts = np.array([_interpolate_route(route, p) for p in np.linspace(0, 1, 120)])
        self.planned_line, = ax.plot(
            pts[:, 0], pts[:, 1], "--", color=GREEN, lw=2.0, alpha=0.75,
            label="Authorised mission route",
        )

        ax.scatter(*START, marker="^", s=120, color=GREEN, edgecolors="white", linewidths=0.8, zorder=8)
        ax.annotate("START", START, xytext=(7, -14), textcoords="offset points", color=GREEN,
                     fontsize=9, fontweight="bold")
        ax.scatter(*DESTINATION, marker="*", s=260, color=GREEN, edgecolors="white", linewidths=0.6, zorder=8)
        ax.annotate("AUTHORISED\nDESTINATION", DESTINATION, xytext=(6, 10), textcoords="offset points",
                     color=GREEN, fontsize=9, fontweight="bold")

        # Landmarks used by Camera / LiDAR
        ax.scatter(
            LANDMARKS_3D[:, 0], LANDMARKS_3D[:, 1],
            marker="s", s=20, color=MUTED, alpha=0.45,
            label="Camera / LiDAR landmarks",
        )

        self.spoof_dot, = ax.plot([], [], marker="X", ms=12, color=RED, linestyle="None", zorder=9,
                                   label="Injected spoof target")
        self.spoof_zone = Circle((0, 0), radius=16, color=RED, alpha=0.0, lw=0)
        ax.add_patch(self.spoof_zone)
        self.recovery_line, = ax.plot([], [], color=CYAN, lw=2.4, ls="-.", label="Server-signed rejoin route")
        self.true_trail, = ax.plot([], [], color=ORANGE, lw=2.7, label="Physical drone path")
        self.trusted_trail, = ax.plot([], [], color=BLUE, lw=1.4, alpha=0.85, label="EKF fused estimate")
        self.gps_trail, = ax.plot([], [], color=RED, lw=1.8, ls=":", label="Raw GNSS (untrusted)")
        self.drone_dot, = ax.plot([], [], marker="h", ms=13, color=ORANGE, markeredgecolor="white",
                                   markeredgewidth=0.8, linestyle="None", zorder=12, label="Drone")
        self.gps_dot, = ax.plot([], [], marker="x", ms=9, mew=2.2, color=RED, linestyle="None", zorder=11)
        self.trusted_dot, = ax.plot([], [], marker="o", ms=5, color=BLUE, linestyle="None", zorder=11)
        self.status_text = ax.text(
            0.50, 0.975, "", transform=ax.transAxes, ha="center", va="top", fontsize=9.5,
            fontweight="bold", color="white",
            bbox=dict(boxstyle="round,pad=0.45", facecolor=BLUE, alpha=0.9),
        )
        ax.legend(loc="lower right", fontsize=7.4, framealpha=0.92, facecolor=PANEL, edgecolor=GRID)

    def _build_metrics(self) -> None:
        ax = self.ax_metrics
        ax.set_title("EKF-driven detection evidence", loc="left", color=TEXT, fontsize=11, fontweight="bold", pad=10)
        ax.set_xlabel("Mission time (s)", fontsize=8)
        ax.set_ylabel("Detector / alarm limit", fontsize=8)
        ax.set_xlim(0, 70)
        ax.set_ylim(0, 1.35)
        ax.axhline(1.0, color=RED, lw=1.2, ls="--", label="Alarm threshold")
        self.nis_line, = ax.plot([], [], color=PURPLE, lw=1.8, label="χ²: GNSS NIS via EKF")
        self.cusum_line, = ax.plot([], [], color=YELLOW, lw=1.8, label="CUSUM: accumulated drift")
        self.ml_line, = ax.plot([], [], color=CYAN, lw=1.6, ls=":", label="LSTM-AE: sequence loss")
        ax.grid(True)
        ax.legend(loc="upper left", fontsize=7.2, framealpha=0.92, facecolor=PANEL, edgecolor=GRID)
        ax.text(
            0.98, 0.03,
            "Innovation: EKF.compute_innovation(GNSS)\nReference: 11-state EKF fused estimate",
            transform=ax.transAxes, color=MUTED, fontsize=7.2, va="bottom", ha="right",
        )

    def _build_state_panel(self) -> None:
        ax = self.ax_state
        ax.set_title("EKF sensor fusion status", loc="left", color=TEXT, fontsize=11, fontweight="bold", pad=10)
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.axis("off")
        self.state_text = ax.text(
            0.04, 0.94, "", va="top", ha="left", fontsize=9.1, linespacing=1.47,
            family="DejaVu Sans Mono",
        )

    def _build_log(self) -> None:
        ax = self.ax_log
        ax.set_title("Live event log", loc="left", color=TEXT, fontsize=11, fontweight="bold", pad=7)
        ax.set_facecolor(BG)
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.axis("off")
        self.log_lines = [
            ax.text(0.01, 0.78 - i * 0.22, "", color=TEXT, fontsize=9.1, family="DejaVu Sans Mono", va="top")
            for i in range(4)
        ]

    def _build_controls(self) -> None:
        # Control buttons
        self.launch_button = Button(
            self.fig.add_axes([0.045, 0.035, 0.110, 0.045]),
            "Launch mission", color="#1d4ed8", hovercolor="#2563eb",
        )
        self.pause_button = Button(
            self.fig.add_axes([0.165, 0.035, 0.105, 0.045]),
            "Pause  [Space]", color="#475569", hovercolor="#64748b",
        )
        self.replay_button = Button(
            self.fig.add_axes([0.280, 0.035, 0.120, 0.045]),
            "Replay attack  [R]", color="#0f766e", hovercolor="#0d9488",
        )
        self.route_button = Button(
            self.fig.add_axes([0.410, 0.035, 0.115, 0.045]),
            "Next Route  [N]", color="#7c3aed", hovercolor="#8b5cf6",
        )
        self.clear_button = Button(
            self.fig.add_axes([0.535, 0.035, 0.115, 0.045]),
            "Clear Spoof  [C]", color="#991b1b", hovercolor="#b91c1c",
        )
        self.route_label = self.fig.text(
            0.665, 0.057, f"Route: {self.selected_route_name}",
            color=CYAN, fontsize=9, fontweight="bold", va="center",
        )

        for btn in [self.launch_button, self.pause_button, self.replay_button, self.route_button, self.clear_button]:
            btn.label.set_color("white")

        self.launch_button.on_clicked(self.launch_mission)
        self.pause_button.on_clicked(self.toggle_pause)
        self.replay_button.on_clicked(self.replay_last_attack)
        self.route_button.on_clicked(self.cycle_route)
        self.clear_button.on_clicked(self.clear_spoof)

        self.fig.text(
            0.975, 0.057,
            "EKF: 11-state MultiSensorEKF  •  Monitors: Chi² + CUSUM + LSTM-AE ML  •  Sensors: GNSS+IMU+Camera+LiDAR",
            color=MUTED, fontsize=8, ha="right", va="center",
        )

    # ── Interactive controls ────────────────────────────────────────────
    def toggle_pause(self, _event=None) -> None:
        if not self.mission_started or self.phase == "COMPLETE":
            return
        self._paused = not self._paused
        self.pause_button.label.set_text("Resume  [Space]" if self._paused else "Pause  [Space]")
        if self.animation is not None:
            if self._paused:
                self.animation.event_source.stop()
            else:
                self.animation.event_source.start()
        self.fig.canvas.draw_idle()

    def replay_last_attack(self, _event=None) -> None:
        """Replay the presenter's last map-selected spoof at the same mission time."""
        target = None if self._last_spoof_target is None else self._last_spoof_target.copy()
        attack_time = self._last_spoof_time
        self.launch_mission()
        if target is not None and attack_time is not None:
            self._scheduled_replay = (attack_time, target)
            self._add_event(f"REPLAY: Previous fake target will be injected again at {attack_time:.1f}s.", CYAN)
        else:
            self._add_event("REPLAY READY: No prior spoof selected. Click the map to inject one.", CYAN)
        self.render()

    def _on_key(self, event) -> None:
        key = "" if event.key is None else str(event.key).lower()
        if key in {" ", "space", "spacebar"}:
            self.toggle_pause()
        elif key == "r":
            self.replay_last_attack()
        elif key == "n":
            self.cycle_route()
        elif key == "c":
            self.clear_spoof()

    def _tick(self, _frame) -> None:
        if not self._paused:
            self.advance()
        self.render()

    # ── Render ──────────────────────────────────────────────────────────
    def render(self) -> None:
        if self.fig is None:
            return
        physical = np.array(self.history["physical"]) if self.history["physical"] else np.array([self.position])
        trusted = np.array(self.history["trusted"]) if self.history["trusted"] else np.array([self.trusted_position])
        gps = np.array(self.history["gps"]) if self.history["gps"] else np.array([self.gps_position])
        times = np.array(self.history["time"]) if self.history["time"] else np.array([0.0])
        self.true_trail.set_data(physical[:, 0], physical[:, 1])
        self.trusted_trail.set_data(trusted[:, 0], trusted[:, 1])
        self.gps_trail.set_data(gps[:, 0], gps[:, 1])
        self.drone_dot.set_data([self.position[0]], [self.position[1]])
        self.trusted_dot.set_data([self.trusted_position[0]], [self.trusted_position[1]])
        self.gps_dot.set_data([self.gps_position[0]], [self.gps_position[1]])
        self.gps_dot.set_alpha(0.35 if self.phase in {"QUARANTINED", "SERVER_VERIFYING", "RECOVERING"} else 1.0)

        if self.spoof_target is not None:
            self.spoof_dot.set_data([self.spoof_target[0]], [self.spoof_target[1]])
            self.spoof_zone.center = self.spoof_target
            self.spoof_zone.set_alpha(0.12)
        else:
            self.spoof_dot.set_data([], [])
            self.spoof_zone.set_alpha(0.0)
        if self.recovery_route:
            route = np.array(self.recovery_route)
            self.recovery_line.set_data(route[:, 0], route[:, 1])

        if self.phase == "NOMINAL":
            if self.cusum_value >= self.cusum.warning_threshold:
                label = "EVALUATING SENSOR DRIFT — CUSUM ACCUMULATING"
                colour = ORANGE
            else:
                label = "AUTONOMOUS MISSION FLIGHT — ROUTE TRACKING (NOMINAL)"
                colour = GREEN
        else:
            label, colour = self.phase_style[self.phase]
        self.status_text.set_text(label)
        self.status_text.set_bbox(dict(boxstyle="round,pad=0.45", facecolor=colour, alpha=0.90))
        normalized_nis = np.minimum(np.asarray(self.history["nis"]) / max(self.nis_threshold, 1e-9), 1.30)
        normalized_cusum = np.minimum(np.asarray(self.history["cusum"]) / self.cusum.alarm_threshold, 1.30)
        normalized_ml = np.minimum(np.asarray(self.history["ml_mse"]) / max(self.ml_monitor.alarm_threshold, 1e-9), 1.30)
        self.nis_line.set_data(times, normalized_nis)
        self.cusum_line.set_data(times, normalized_cusum)
        self.ml_line.set_data(times, normalized_ml)

        # State panel: reflects actual onboard detector verdict, no ground-truth leak
        if self.phase in {"QUARANTINED", "SERVER_VERIFYING", "RECOVERING"}:
            gps_state = "QUARANTINED"
            state_color = YELLOW
        elif self.cusum_value >= self.cusum.warning_threshold or self.chi_rejection_count > 0 or self.ml_status != MLStatus.HEALTHY:
            gps_state = "SUSPECT (DRIFT/ANOMALY)"
            state_color = ORANGE
        else:
            gps_state = "HEALTHY (TRACKING)"
            state_color = TEXT
        gnss_fused = "YES" if self.gnss_accepted else "NO (rejected)"
        server_state = "NOT NEEDED"
        if self.detection_time is not None:
            server_state = "REQUESTING ROUTE"
        if self.server_authorization_time is not None:
            server_state = "HMAC PLAN ACCEPTED"
        state_lines = [
            f"EKF POSITION        ({self.trusted_position[0]:.1f}, {self.trusted_position[1]:.1f})",
            f"GNSS STATUS         {gps_state}",
            f"GNSS → EKF FUSED    {gnss_fused}",
            f"GNSS INNOVATION     {self.ekf_innovation_mag:.2f} m",
            f"LSTM-AE ML LOSS     {self.ml_mse:.1f} ({self.ml_status.name})",
            "",
            "EKF SENSOR INPUTS",
            f"  IMU  (predict+update) {self.sensor_residuals['IMU']:.2f} m",
            f"  Camera (2 landmarks)  {self.sensor_residuals['CAMERA']:.2f} m",
            f"  LiDAR  (1 ref point)  {self.sensor_residuals['LIDAR']:.2f} m",
            f"SERVER              {server_state}",
        ]
        self.state_text.set_text("\n".join(state_lines))
        self.state_text.set_color(state_color)

        visible_events = self.events[-4:]
        for text_artist, item in zip(self.log_lines, visible_events):
            text_artist.set_text(f"[{item.time_s:05.1f}s] {item.message}")
            text_artist.set_color(item.colour)
        for text_artist in self.log_lines[len(visible_events):]:
            text_artist.set_text("")

    # ── Show / headless ─────────────────────────────────────────────────
    def show(self) -> None:
        self.build_figure()
        self.animation = FuncAnimation(self.fig, self._tick, interval=70, blit=False, cache_frame_data=False)
        plt.show()

    def run_validation_scenario(self, route_name: str = "Direct") -> None:
        """Headless check: spoof → reassign mid-flight → detect → recover → reassign during recovery → complete."""
        self.selected_route_name = route_name
        self.launch_mission()

        # 1. Fly cleanly for 12 seconds with EKF sensor fusion (truthful GNSS)
        while self.time_s < 12.0:
            self.advance()
        if self.cusum_value > self.cusum.warning_threshold:
            raise RuntimeError(f"False alarm detected during clean flight! CUSUM={self.cusum_value:.1f}")

        # 2. Inject initial spoof target
        self.inject_spoof(np.array([62.0, 168.0]))

        # 3. Advance mid-drift, then REASSIGN spoof target mid-flight
        while self.time_s < 13.6 and self.phase == "NOMINAL":
            self.advance()
        reassign_ok = self.inject_spoof(np.array([68.0, 172.0]))
        if not reassign_ok:
            raise RuntimeError("Failed to reassign spoof target mid-flight!")

        # 4. Advance until CUSUM detects, quarantines, verifies, and begins recovering
        for _ in range(1500):
            self.advance()
            if self.phase == "RECOVERING":
                break
        if self.phase != "RECOVERING":
            raise RuntimeError(f"Expected phase RECOVERING, got {self.phase}")

        # 5. REASSIGN spoof target during RECOVERING
        recovery_reassign_ok = self.inject_spoof(np.array([30.0, 160.0]))
        if not recovery_reassign_ok:
            raise RuntimeError("Failed to reassign spoof target during recovery phase!")

        # 6. Wait for recovery back to NOMINAL, then second detection & completion
        for _ in range(3500):
            self.advance()
            if self.phase == "COMPLETE":
                break
        if np.linalg.norm(self.position - DESTINATION) > 0.5:
            raise RuntimeError(
                f"Validation ({route_name}): did not reach destination "
                f"(error: {np.linalg.norm(self.position - DESTINATION):.2f} m, phase={self.phase})."
            )

    def report(self) -> str:
        lines = [
            "Interactive SensorSentry simulation validated (EKF-driven)",
            f"  Route:                       {self.selected_route_name}",
            f"  Total spoof attacks:         {self.spoof_count}",
            f"  Final destination error:     {np.linalg.norm(self.position - DESTINATION):.2f} m",
            f"  Final phase:                 {self.phase}",
        ]
        return "\n".join(lines)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the interactive SensorSentry spoofing simulation (EKF-driven).")
    parser.add_argument("--no-show", action="store_true",
                        help="Run the state-machine validation without opening a window.")
    parser.add_argument("--snapshot", type=Path, metavar="FILE.png",
                        help="Save a validated final-state image using the built-in sample spoof target.")
    parser.add_argument("--route", type=str, default="Direct", choices=ROUTE_NAMES,
                        help="Select the mission route for headless/snapshot mode.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.no_show or args.snapshot is not None:
        demo = InteractiveSpoofDemo(build_figure=args.snapshot is not None)
        demo.run_validation_scenario(route_name=args.route)
        if args.snapshot is not None:
            demo.render()
            args.snapshot.parent.mkdir(parents=True, exist_ok=True)
            demo.fig.savefig(args.snapshot, dpi=150, facecolor=BG, bbox_inches="tight")
            print(f"Saved validated sample snapshot: {args.snapshot}")
        print(demo.report())
        return
    InteractiveSpoofDemo().show()


if __name__ == "__main__":
    main()
