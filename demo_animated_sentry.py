"""Interactive SensorSentry spoofing simulation for a hackathon judge.

The simulation deliberately waits for the presenter to inject the attack:

1. Click ``Launch mission``.
2. While the drone flies normally, click any point on the map to set the
   attacker's spoofed GPS destination.
3. Watch GNSS diverge from the trusted IMU + camera + LiDAR fusion estimate.
4. Chi-Square rejects bad fixes, CUSUM confirms persistent drift, GNSS is
   quarantined, and the simulated server returns a signed rejoin route.

Controls are available both as visible buttons and keyboard shortcuts:
Space pauses/resumes; R replays the last attack scenario.

This is a local simulation only. It does not control a physical drone.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation
from matplotlib.patches import Circle
from matplotlib.widgets import Button

from decision_making import ChiSquareGate, CusumMonitor
from decision_making.cusum_monitor import DriftStatus
from hal import SimulatedBus
from waypoint_security import CommandValidator, WaypointCommand


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

START = np.array([15.0, 15.0])
DESTINATION = np.array([150.0, 125.0])
DT = 0.20
MISSION_DURATION = 60.0


@dataclass
class Event:
    time_s: float
    message: str
    colour: str


class InteractiveSpoofDemo:
    """Live simulation state, plotting, detection, and response workflow."""

    phase_style = {
        "READY": ("READY  •  LAUNCH THE MISSION, THEN CLICK MAP TO INJECT SPOOF", BLUE),
        "NOMINAL": ("PHASE 1  •  NOMINAL FLIGHT — CLICK MAP TO SET FAKE GPS TARGET", GREEN),
        "SPOOFED": ("PHASE 2  •  GPS SPOOF ACTIVE — DRONE IS BEING DIVERTED", ORANGE),
        "QUARANTINED": ("PHASE 3  •  SPOOF CONFIRMED — GNSS QUARANTINED, SAFE HOVER", YELLOW),
        "SERVER_VERIFYING": ("PHASE 4  •  ENCRYPTED SERVER RECONNECT IN PROGRESS", CYAN),
        "RECOVERING": ("PHASE 5  •  FOLLOWING HMAC-SIGNED REJOIN ROUTE", BLUE),
        "FINAL_LEG": ("PHASE 6  •  BACK ON THE AUTHORISED MISSION", GREEN),
        "COMPLETE": ("COMPLETE  •  AUTHORISED DESTINATION REACHED", GREEN),
    }

    def __init__(self, build_figure: bool = True) -> None:
        self.fig = None
        self.animation = None
        self._paused = False
        self._scheduled_replay: Optional[Tuple[float, np.ndarray]] = None
        self._last_spoof_target: Optional[np.ndarray] = None
        self._last_spoof_time: Optional[float] = None
        self._reset_state(keep_last_attack=True)
        if build_figure:
            self.build_figure()

    @staticmethod
    def planned_position(time_s: float) -> np.ndarray:
        """Position on the original, authorised start-to-destination route."""
        progress = np.clip(time_s / MISSION_DURATION, 0.0, 1.0)
        return START + progress * (DESTINATION - START)

    @staticmethod
    def _move_toward(position: np.ndarray, target: np.ndarray,
                     speed_mps: float, dt: float) -> np.ndarray:
        delta = target - position
        distance = float(np.linalg.norm(delta))
        if distance < 1e-9:
            return position.copy()
        return position + delta * min(speed_mps * dt / distance, 1.0)

    def _reset_state(self, keep_last_attack: bool) -> None:
        if not keep_last_attack:
            self._last_spoof_target = None
            self._last_spoof_time = None
        self.time_s = 0.0
        self.position = START.copy()
        self.phase = "READY"
        self.mission_started = False
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
        self.chi_gate = ChiSquareGate(confidence=0.999)
        self.cusum = CusumMonitor(
            alarm_threshold=90.0,
            warning_threshold=35.0,
            slack=1.0,
            recovery_rate=0.0,
            auto_recover_after=0,
        )
        self.innovation_covariance = np.diag([4.0 ** 2, 4.0 ** 2])
        self.nis_threshold = self.chi_gate._chi2_threshold(2)
        self.nis = 0.0
        self.cusum_value = 0.0
        self.last_gps = START.copy()
        self.gps_position = START.copy()
        self.trusted_position = START.copy()
        self.sensor_residuals = {"IMU": 0.0, "CAMERA": 0.0, "LIDAR": 0.0}
        self.events: List[Event] = [
            Event(0.0, "READY: Launch the mission, then click the map to inject a spoofed destination.", BLUE),
        ]
        self.history = {
            "time": [], "physical": [], "trusted": [], "gps": [], "nis": [], "cusum": [],
        }

    def _add_event(self, message: str, colour: str) -> None:
        self.events.append(Event(self.time_s, message, colour))

    def launch_mission(self, _event=None) -> None:
        """Start a clean mission. The spoof remains entirely presenter-controlled."""
        self._reset_state(keep_last_attack=True)
        self.mission_started = True
        self.phase = "NOMINAL"
        self._paused = False
        self._add_event("MISSION: Drone departed START toward authorised destination. Awaiting manual spoof injection.", GREEN)
        if self.animation is not None:
            self.animation.event_source.start()
        self.render()

    def inject_spoof(self, target: np.ndarray, from_replay: bool = False) -> bool:
        """Begin a gradual GNSS spoof toward the location selected by the presenter."""
        target = np.asarray(target, dtype=float)
        if not self.mission_started or self.phase != "NOMINAL":
            return False
        if np.linalg.norm(target - self.position) < 18.0:
            self._add_event("SPOOF INPUT IGNORED: Choose a point at least 18 m from the drone.", YELLOW)
            return False
        self.spoof_target = target.copy()
        self.spoof_start_time = self.time_s
        self._last_spoof_target = target.copy()
        self._last_spoof_time = self.time_s
        self.phase = "SPOOFED"
        prefix = "REPLAY: " if from_replay else "ATTACK: "
        self._add_event(
            f"{prefix}Fake GNSS target set to [{target[0]:.0f}, {target[1]:.0f}]. Autopilot is being pulled off-route.",
            RED,
        )
        return True

    def _on_map_click(self, event) -> None:
        if event.inaxes is not self.ax_map or event.xdata is None or event.ydata is None:
            return
        if not self.mission_started:
            self._add_event("CLICK RECEIVED: Press Launch mission first, then select the fake target.", YELLOW)
        elif self.phase == "NOMINAL":
            self.inject_spoof(np.array([event.xdata, event.ydata]))
        elif self.phase == "SPOOFED":
            self._add_event("SPOOF ALREADY ACTIVE: Reset or replay before choosing another fake target.", YELLOW)
        else:
            self._add_event("MISSION RESPONSE ACTIVE: Reset/replay to run another manual spoof scenario.", YELLOW)
        self.render()
        self.fig.canvas.draw_idle()

    def _trusted_sensor_estimates(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Independent, healthy estimates used to cross-check GNSS.

        IMU dead reckoning, camera landmarks, and LiDAR map matching each observe
        physical motion independently. Their weighted fusion is deliberately what
        feeds the GNSS innovation for both detector models.
        """
        t = self.time_s
        imu = self.position + np.array([0.55 * np.sin(1.4 * t), 0.42 * np.cos(1.1 * t)])
        camera = self.position + np.array([0.24 * np.cos(0.8 * t), 0.20 * np.sin(1.7 * t)])
        lidar = self.position + np.array([0.18 * np.sin(1.9 * t), -0.16 * np.cos(1.3 * t)])
        fused = 0.25 * imu + 0.45 * camera + 0.30 * lidar
        self.sensor_residuals = {
            "IMU": float(np.linalg.norm(imu - fused)),
            "CAMERA": float(np.linalg.norm(camera - fused)),
            "LIDAR": float(np.linalg.norm(lidar - fused)),
        }
        return imu, camera, lidar, fused

    def _gnss_reading(self) -> np.ndarray:
        noise = np.array([
            0.8 * np.sin(0.7 * self.time_s),
            0.7 * np.cos(0.9 * self.time_s),
        ])
        if self.phase != "SPOOFED" or self.spoof_target is None or self.spoof_start_time is None:
            return self.position + noise
        # Compute where the drone *should* be on its nominal route
        nominal = self.planned_position(self.time_s)
        # Spoof offset grows quadratically from the nominal position toward the fake target
        elapsed = self.time_s - self.spoof_start_time
        direction = self.spoof_target - nominal
        dist = np.linalg.norm(direction)
        if dist < 1e-6:
            return nominal + noise
        direction_unit = direction / dist
        # Quadratic drift: 1.5 m/s^2 acceleration away from nominal
        drift_mag = min(dist * 0.9, 1.5 * elapsed ** 2)
        return nominal + direction_unit * drift_mag + noise

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
        signal = validator.validate(command, np.r_[self.position, 10.0], self.time_s)
        self.bus.send(signal.to_bytes(), "COMMAND")
        return signal.verdict.name == "ACCEPTED"

    def _make_recovery_route(self) -> List[np.ndarray]:
        """Return a server-approved corridor to rejoin the original route."""
        route_vector = DESTINATION - START
        route_length_sq = float(np.dot(route_vector, route_vector))
        current_progress = float(np.dot(self.position - START, route_vector) / route_length_sq)
        rejoin_progress = np.clip(max(current_progress + 0.30, 0.55), 0.55, 0.86)
        rejoin = START + rejoin_progress * route_vector
        perpendicular = np.array([-route_vector[1], route_vector[0]]) / np.linalg.norm(route_vector)
        lane_direction = 1.0
        if self.spoof_target is not None:
            lane_direction = -np.sign(np.dot(self.spoof_target - self.position, perpendicular)) or 1.0
        safe_lane = self.position + 0.48 * (rejoin - self.position) + lane_direction * 10.0 * perpendicular
        return [self.position.copy(), safe_lane, rejoin, DESTINATION.copy()]

    def _advance_motion(self) -> None:
        if self.phase == "NOMINAL":
            self.position = self.planned_position(self.time_s)
            if self.time_s >= MISSION_DURATION:
                self.position = DESTINATION.copy()
                self.phase = "COMPLETE"
                self._add_event("COMPLETE: Destination reached without an injected spoof. Press R or Replay to try again.", GREEN)
        elif self.phase == "SPOOFED":
            self.position = self._move_toward(self.position, self.spoof_target, 5.4, DT)
        elif self.phase in {"QUARANTINED", "SERVER_VERIFYING"}:
            self.position = self.position.copy()  # Safe hover.
        elif self.phase == "RECOVERING":
            target = self.recovery_route[self.recovery_index + 1]
            self.position = self._move_toward(self.position, target, 6.5, DT)
            if np.linalg.norm(self.position - target) < 0.15:
                self.recovery_index += 1
                if self.recovery_index >= len(self.recovery_route) - 1:
                    self.phase = "FINAL_LEG"
                    self._add_event("REJOIN: Secure route regained. Returning to the original destination.", GREEN)
        elif self.phase == "FINAL_LEG":
            self.position = self._move_toward(self.position, DESTINATION, 5.3, DT)
            if np.linalg.norm(self.position - DESTINATION) < 0.15:
                self.position = DESTINATION.copy()
                self.phase = "COMPLETE"
                self._add_event("COMPLETE: Drone arrived at authorised destination. Mission integrity restored.", GREEN)

    def _evaluate_detectors(self) -> None:
        _, _, _, self.trusted_position = self._trusted_sensor_estimates()
        gps_is_active = self.phase in {"NOMINAL", "SPOOFED"}
        if not gps_is_active:
            self.gps_position = self.last_gps.copy()
            return

        self.gps_position = self._gnss_reading()
        innovation = self.gps_position - self.trusted_position
        fast_signal = self.chi_gate.evaluate("GNSS", self.time_s, innovation, self.innovation_covariance, 2)
        slow_signal = self.cusum.evaluate("GNSS", self.time_s, innovation, self.innovation_covariance, 2)
        self.nis = fast_signal.nis
        self.nis_threshold = fast_signal.threshold
        self.cusum_value = slow_signal.cusum_value
        self.last_gps = self.gps_position.copy()

        if fast_signal.verdict.name == "REJECT":
            self.chi_rejection_count += 1
            if not self._chi_logged:
                self._chi_logged = True
                self._add_event(
                    "CHI-SQUARE: GNSS innovation exceeds the gate against fused IMU + camera + LiDAR position.",
                    PURPLE,
                )
        if slow_signal.status == DriftStatus.WARNING and not self._warning_logged:
            self._warning_logged = True
            self._add_event("CUSUM WARNING: Repeated GNSS-vs-fusion mismatch is accumulating.", ORANGE)
        if self.phase == "SPOOFED" and slow_signal.status == DriftStatus.ALARM:
            self.phase = "QUARANTINED"
            self.detection_time = self.time_s
            self.bus.send(b"GPS_QUARANTINED", "SAARM")
            self._add_event(
                "CUSUM ALARM: Persistent GNSS drift confirmed. SAARM isolates GNSS; drone holds on trusted sensors.",
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
                self._add_event("SERVER LINK: Encrypted telemetry sent; requesting an authenticated rejoin route.", CYAN)
        if self.phase == "SERVER_VERIFYING" and since_detection >= 3.8:
            self.server_command_accepted = self._authenticate_recovery_plan()
            if not self.server_command_accepted:
                raise RuntimeError("The simulated server recovery command was not authenticated.")
            self.server_authorization_time = self.time_s
            self.recovery_route = self._make_recovery_route()
            self.recovery_index = 0
            self.phase = "RECOVERING"
            self._add_event("SERVER AUTHORIZED: HMAC-verified rejoin route received. Secure navigation resumes.", BLUE)

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
            self.fig.canvas.manager.set_window_title("SensorSentry | Interactive GPS Spoofing Simulation")
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
        self.fig.suptitle("SensorSentry  |  Interactive GPS Spoof → Detection → Secure Recovery",
                           color=CYAN, fontsize=16, fontweight="bold")
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
        ax.set_title("Click anywhere in this map during nominal flight to enter the spoofed GPS target",
                     loc="left", color=TEXT, fontsize=11.5, fontweight="bold", pad=10)
        ax.set_xlim(0, 175)
        ax.set_ylim(0, 185)
        ax.set_aspect("equal")
        ax.set_xlabel("East (m)")
        ax.set_ylabel("North (m)")
        ax.grid(True)
        planned = np.array([self.planned_position(t) for t in np.linspace(0, MISSION_DURATION, 100)])
        ax.plot(planned[:, 0], planned[:, 1], "--", color=GREEN, lw=2.0, alpha=0.75,
                label="Authorised mission route")
        ax.scatter(*START, marker="^", s=120, color=GREEN, edgecolors="white", linewidths=0.8, zorder=8)
        ax.annotate("START", START, xytext=(7, -14), textcoords="offset points", color=GREEN,
                    fontsize=9, fontweight="bold")
        ax.scatter(*DESTINATION, marker="*", s=260, color=GREEN, edgecolors="white", linewidths=0.6, zorder=8)
        ax.annotate("AUTHORISED\nDESTINATION", DESTINATION, xytext=(6, 10), textcoords="offset points",
                    color=GREEN, fontsize=9, fontweight="bold")
        environment = np.array([[30, 105], [46, 93], [107, 55], [127, 77], [137, 111], [83, 119]])
        ax.scatter(environment[:, 0], environment[:, 1], marker="s", s=20, color=MUTED, alpha=0.45,
                   label="Camera / LiDAR references")
        self.spoof_dot, = ax.plot([], [], marker="X", ms=12, color=RED, linestyle="None", zorder=9,
                                  label="Presenter-selected fake target")
        self.spoof_zone = Circle((0, 0), radius=16, color=RED, alpha=0.0, lw=0)
        ax.add_patch(self.spoof_zone)
        self.recovery_line, = ax.plot([], [], color=CYAN, lw=2.4, ls="-.", label="Server-signed rejoin route")
        self.true_trail, = ax.plot([], [], color=ORANGE, lw=2.7, label="Physical drone path")
        self.trusted_trail, = ax.plot([], [], color=BLUE, lw=1.4, alpha=0.85, label="Trusted fusion estimate")
        self.gps_trail, = ax.plot([], [], color=RED, lw=1.8, ls=":", label="Raw GNSS (untrusted)")
        self.drone_dot, = ax.plot([], [], marker="h", ms=13, color=ORANGE, markeredgecolor="white",
                                  markeredgewidth=0.8, linestyle="None", zorder=12, label="Drone")
        self.gps_dot, = ax.plot([], [], marker="x", ms=9, mew=2.2, color=RED, linestyle="None", zorder=11)
        self.trusted_dot, = ax.plot([], [], marker="o", ms=5, color=BLUE, linestyle="None", zorder=11)
        self.status_text = ax.text(0.50, 0.975, "", transform=ax.transAxes, ha="center", va="top", fontsize=9.5,
                                   fontweight="bold", color="white",
                                   bbox=dict(boxstyle="round,pad=0.45", facecolor=BLUE, alpha=0.9))
        ax.legend(loc="lower right", fontsize=7.4, framealpha=0.92, facecolor=PANEL, edgecolor=GRID)

    def _build_metrics(self) -> None:
        ax = self.ax_metrics
        ax.set_title("Detection evidence", loc="left", color=TEXT, fontsize=11, fontweight="bold", pad=10)
        ax.set_xlabel("Mission time (s)", fontsize=8)
        ax.set_ylabel("Detector / alarm limit", fontsize=8)
        ax.set_xlim(0, 70)
        ax.set_ylim(0, 1.35)
        ax.axhline(1.0, color=RED, lw=1.2, ls="--", label="Alarm threshold")
        self.nis_line, = ax.plot([], [], color=PURPLE, lw=1.8, label="χ²: GNSS vs fusion")
        self.cusum_line, = ax.plot([], [], color=YELLOW, lw=1.8, label="CUSUM: repeated drift")
        ax.grid(True)
        ax.legend(loc="upper left", fontsize=7.2, framealpha=0.92, facecolor=PANEL, edgecolor=GRID)
        ax.text(0.98, 0.03, "Input: GNSS innovation\nReference: IMU + camera + LiDAR fusion",
                transform=ax.transAxes, color=MUTED, fontsize=7.2, va="bottom", ha="right")

    def _build_state_panel(self) -> None:
        ax = self.ax_state
        ax.set_title("Which sensors prove it?", loc="left", color=TEXT, fontsize=11, fontweight="bold", pad=10)
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.axis("off")
        self.state_text = ax.text(0.04, 0.94, "", va="top", ha="left", fontsize=9.1, linespacing=1.47,
                                  family="DejaVu Sans Mono")

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
        self.launch_button = Button(self.fig.add_axes([0.050, 0.035, 0.115, 0.045]), "Launch mission",
                                    color="#1d4ed8", hovercolor="#2563eb")
        self.pause_button = Button(self.fig.add_axes([0.175, 0.035, 0.115, 0.045]), "Pause  [Space]",
                                   color="#475569", hovercolor="#64748b")
        self.replay_button = Button(self.fig.add_axes([0.300, 0.035, 0.135, 0.045]), "Replay attack  [R]",
                                    color="#0f766e", hovercolor="#0d9488")
        self.launch_button.label.set_color("white")
        self.pause_button.label.set_color("white")
        self.replay_button.label.set_color("white")
        self.launch_button.on_clicked(self.launch_mission)
        self.pause_button.on_clicked(self.toggle_pause)
        self.replay_button.on_clicked(self.replay_last_attack)
        self.fig.text(0.975, 0.057, "Controls work even if the map does not have keyboard focus.",
                      color=MUTED, fontsize=8, ha="right", va="center")

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

    def _tick(self, _frame) -> None:
        if not self._paused:
            self.advance()
        self.render()

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
        self.gps_dot.set_alpha(0.35 if self.phase not in {"NOMINAL", "SPOOFED"} else 1.0)

        if self.spoof_target is not None:
            self.spoof_dot.set_data([self.spoof_target[0]], [self.spoof_target[1]])
            self.spoof_zone.center = self.spoof_target
            self.spoof_zone.set_alpha(0.10)
        if self.recovery_route:
            route = np.array(self.recovery_route)
            self.recovery_line.set_data(route[:, 0], route[:, 1])

        label, colour = self.phase_style[self.phase]
        self.status_text.set_text(label)
        self.status_text.set_bbox(dict(boxstyle="round,pad=0.45", facecolor=colour, alpha=0.90))
        normalized_nis = np.minimum(np.asarray(self.history["nis"]) / max(self.nis_threshold, 1e-9), 1.30)
        normalized_cusum = np.minimum(np.asarray(self.history["cusum"]) / self.cusum.alarm_threshold, 1.30)
        self.nis_line.set_data(times, normalized_nis)
        self.cusum_line.set_data(times, normalized_cusum)

        gps_state = "HEALTHY" if self.phase in {"READY", "NOMINAL"} else "SPOOFED"
        if self.phase not in {"READY", "NOMINAL", "SPOOFED"}:
            gps_state = "QUARANTINED"
        server_state = "NOT NEEDED"
        if self.detection_time is not None:
            server_state = "REQUESTING ROUTE"
        if self.server_authorization_time is not None:
            server_state = "HMAC PLAN ACCEPTED"
        state_lines = [
            f"GNSS raw input      {gps_state}",
            f"GNSS innovation     {self.nis:5.1f} m²-equivalent",
            "",
            "TRUSTED WITNESSES",
            f"IMU dead reckoning  {self.sensor_residuals['IMU']:.2f} m  HEALTHY",
            f"Camera landmarks    {self.sensor_residuals['CAMERA']:.2f} m  HEALTHY",
            f"LiDAR map match     {self.sensor_residuals['LIDAR']:.2f} m  HEALTHY",
            "",
            "χ² / CUSUM INPUT    GNSS vs fused witnesses",
            f"SERVER              {server_state}",
        ]
        self.state_text.set_text("\n".join(state_lines))
        self.state_text.set_color(YELLOW if gps_state == "QUARANTINED" else TEXT)

        visible_events = self.events[-4:]
        for text_artist, item in zip(self.log_lines, visible_events):
            text_artist.set_text(f"[{item.time_s:05.1f}s] {item.message}")
            text_artist.set_color(item.colour)
        for text_artist in self.log_lines[len(visible_events):]:
            text_artist.set_text("")

    def show(self) -> None:
        self.build_figure()
        self.animation = FuncAnimation(self.fig, self._tick, interval=70, blit=False, cache_frame_data=False)
        plt.show()

    def run_validation_scenario(self) -> None:
        """Headless check of the same live state machine, using one sample target."""
        self.launch_mission()
        while self.time_s < 12.0:
            self.advance()
        self.inject_spoof(np.array([62.0, 168.0]))
        for _ in range(600):
            self.advance()
            if self.phase == "COMPLETE":
                break
        if self.detection_time is None or not self.server_command_accepted:
            raise RuntimeError("Validation scenario did not detect and recover from the spoof.")
        if np.linalg.norm(self.position - DESTINATION) > 0.2:
            raise RuntimeError("Validation scenario did not return to the intended destination.")

    def report(self) -> str:
        return "\n".join([
            "Interactive SensorSentry simulation validated",
            f"  Manual/sample spoof begins:  {self.spoof_start_time:.1f} s",
            f"  CUSUM confirmation:          {self.detection_time:.1f} s",
            f"  Chi-Square GNSS rejections:  {self.chi_rejection_count}",
            f"  Server HMAC approval:         {self.server_authorization_time:.1f} s",
            f"  Final destination error:      {np.linalg.norm(self.position - DESTINATION):.2f} m",
        ])


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the interactive SensorSentry spoofing simulation.")
    parser.add_argument("--no-show", action="store_true", help="Run the state-machine validation without opening a window.")
    parser.add_argument("--snapshot", type=Path, metavar="FILE.png",
                        help="Save a validated final-state image using the built-in sample spoof target.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.no_show or args.snapshot is not None:
        demo = InteractiveSpoofDemo(build_figure=args.snapshot is not None)
        demo.run_validation_scenario()
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
