"""Cobot safety supervisor with integrated EKF fusion and statistical anomaly monitors."""

from __future__ import annotations

import math
import os
import sys
import time
from dataclasses import dataclass
from math import hypot
from pathlib import Path
from typing import Dict, Iterable, Optional, Tuple

import numpy as np

# Ensure local package modules can be imported
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)
FILTERPY_DIR = os.path.join(BASE_DIR, "filterpy")
if os.path.isdir(FILTERPY_DIR) and FILTERPY_DIR not in sys.path:
    sys.path.insert(0, FILTERPY_DIR)

from cobot_control import CobotControl
from cobot_hal import HardwareBus
from cobot_response import FleetAlertBroadcaster, SafeStopController
from telemetry import SensorReading
from cobot_waypoint_security import MapBoundary, RouteCorridor

from ekf_fusion import MultiSensorEKF
from decision_making import (
    ChiSquareGate,
    CusumMonitor,
    DriftStatus,
    FastVerdict,
)
from saarm import SaarmFilterBank, SaarmVerdict


Point = Tuple[float, float]


@dataclass(frozen=True)
class CobotSafetyLimits:
    map_boundary: MapBoundary
    corridor_half_width_m: float
    max_linear_speed_m_s: float
    max_angular_speed_rad_s: float
    minimum_obstacle_distance_m: float
    max_localization_jump_m: float
    sensor_timeout_s: float
    chi_square_confidence: float = 0.999
    cusum_alarm_threshold: float = 10.0
    cusum_warning_threshold: float = 5.0
    cusum_slack: float = 0.5
    max_consecutive_outliers: int = 3
    enable_statistical_monitors: bool = True


class CobotSentry:
    """Evaluates safety conditions using stochastic EKF fusion, Chi-Square NIS gating, and CUSUM drift detection."""

    def __init__(
        self,
        robot_id: str,
        limits: CobotSafetyLimits,
        bus: HardwareBus,
        control: CobotControl,
        require_clearance: bool = True,
        supervisor_key: Optional[bytes] = None,
        persistence_path: Optional[Path | str] = None,
    ) -> None:
        self.robot_id = robot_id
        self.limits = limits
        self.bus = bus
        self.control = control
        self.safe_stop = SafeStopController(
            robot_id=robot_id,
            require_clearance=require_clearance,
            supervisor_key=supervisor_key,
            persistence_path=persistence_path,
        )
        self.alerts = FleetAlertBroadcaster(robot_id)
        self.corridor: Optional[RouteCorridor] = None
        self.last_sensor_at: Dict[str, float] = {}
        self.last_odometry: Optional[Point] = None
        self.last_localization: Optional[Point] = None

        # 1. State Estimation & Sensor Fusion (11-state Extended Kalman Filter)
        self.ekf = MultiSensorEKF()
        self.ekf_initialized = False
        self.last_update_time = 0.0
        self.latest_omega_z = 0.0
        self.latest_a_fwd = 0.0

        # 2. Statistical Anomaly Detection Suite
        self.chi_square = ChiSquareGate(confidence=self.limits.chi_square_confidence)
        self.cusum = CusumMonitor(
            alarm_threshold=self.limits.cusum_alarm_threshold,
            warning_threshold=self.limits.cusum_warning_threshold,
            slack=self.limits.cusum_slack,
        )
        self.saarm = SaarmFilterBank()
        self._consecutive_localization_outliers = 0

    @property
    def filtered_position(self) -> Point:
        """Returns current EKF-fused 2D position [px, py] in meters."""
        return (float(self.ekf.x[0]), float(self.ekf.x[1]))

    @property
    def filtered_heading(self) -> float:
        """Returns current EKF-fused yaw heading in radians."""
        return float(self.ekf.x[6])

    @property
    def filtered_speed(self) -> float:
        """Returns current EKF-fused planar speed in m/s."""
        return float(math.hypot(self.ekf.x[3], self.ekf.x[4]))

    def load_trusted_route(self, points: Iterable[Point]) -> None:
        """Load a route only from an already-authorised mission planner."""
        self.corridor = RouteCorridor(points, self.limits.corridor_half_width_m)

    def process_sensor_reading(self, reading: SensorReading) -> bool:
        """Process one validated reading through EKF fusion and anomaly monitors.
        
        Returns false when containment occurs.
        """
        t = reading.timestamp
        self.last_sensor_at[reading.sensor] = t
        values = reading.values

        # 0. Temporal prediction step for EKF
        dt = t - self.last_update_time
        if self.last_update_time > 0 and dt > 1e-6:
            self.ekf.predict(dt, omega_z=self.latest_omega_z, a_fwd=self.latest_a_fwd)
            self.last_update_time = t
        elif self.last_update_time == 0:
            self.last_update_time = t

        # Sensor-specific processing
        if reading.sensor == "ESTOP" and values.get("pressed"):
            return self._contain("E_STOP", "physical e-stop telemetry is active", lockdown=True)

        elif reading.sensor == "LIDAR":
            min_range = float(values["min_range_m"])
            if min_range < self.limits.minimum_obstacle_distance_m:
                return self._contain(
                    "OBSTACLE",
                    f"nearest obstacle is {min_range:.2f} m away",
                    evidence={"min_range_m": f"{min_range:.2f}"},
                )

        elif reading.sensor == "IMU":
            self.latest_omega_z = float(values.get("yaw_rate_rad_s", 0.0))
            self.latest_a_fwd = float(values.get("forward_accel_m_s2", 0.0))

        elif reading.sensor == "ODOMETRY":
            point = values["position_m"]
            self.last_odometry = (float(point[0]), float(point[1]))
            yaw = float(values.get("yaw_rad", 0.0))
            linear_v = float(values.get("linear_velocity_m_s", 0.0))
            angular_v = float(values.get("angular_velocity_rad_s", 0.0))

            if linear_v > self.limits.max_linear_speed_m_s:
                return self._contain("SPEED_LIMIT", "linear velocity exceeds configured safety limit")
            if abs(angular_v) > self.limits.max_angular_speed_rad_s:
                return self._contain("SPEED_LIMIT", "angular velocity exceeds configured safety limit")

            if not self.ekf_initialized:
                self.ekf.x[0] = point[0]
                self.ekf.x[1] = point[1]
                self.ekf.x[6] = yaw
                self.ekf.x[3] = linear_v * math.cos(yaw)
                self.ekf.x[4] = linear_v * math.sin(yaw)
                self.ekf_initialized = True
                self.safe_stop.update_position(*point)
                if not self.limits.map_boundary.contains(point):
                    return self._contain("GEOFENCE_BREACH", "odometry is outside the approved map boundary")
                if self.corridor is not None and not self.corridor.contains(point):
                    distance = self.corridor.distance_to(point)
                    return self._contain("CORRIDOR_BREACH", f"robot is {distance:.2f} m outside the route corridor")
                return not self.safe_stop.is_stopped

            if self.limits.enable_statistical_monitors:
                z_odom = np.array([point[0], point[1]], dtype=float)
                R_odom = np.diag([0.05**2, 0.05**2])

                def h_odom(x):
                    return np.array([x[0], x[1]], dtype=float)

                def H_odom(x):
                    H = np.zeros((2, 11), dtype=float)
                    H[0, 0] = 1.0
                    H[1, 1] = 1.0
                    return H

                y, S, H, PHT = self.ekf.compute_innovation(z_odom, H_odom, h_odom, R_odom)
                m = 2

                fast_sig = self.chi_square.evaluate("ODOMETRY", t, y, S, m)
                slow_sig = self.cusum.evaluate("ODOMETRY", t, y, S, m)

                if slow_sig.status == DriftStatus.ALARM:
                    return self._contain(
                        "SENSOR_DRIFT_ATTACK",
                        f"CUSUM drift alarm on ODOMETRY (CUSUM={slow_sig.cusum_value:.2f})",
                        evidence={"cusum": f"{slow_sig.cusum_value:.2f}", "sensor": "ODOMETRY"},
                    )

                if fast_sig.verdict == FastVerdict.ACCEPT and slow_sig.accepted:
                    self.ekf.apply_update(y, S, H, PHT, R_odom)

            filtered_pt = (float(self.ekf.x[0]), float(self.ekf.x[1]))
            self.safe_stop.update_position(*filtered_pt)

            if not self.limits.map_boundary.contains(filtered_pt) or not self.limits.map_boundary.contains(point):
                return self._contain("GEOFENCE_BREACH", "odometry is outside the approved map boundary")
            if self.corridor is not None and not self.corridor.contains(filtered_pt):
                distance = self.corridor.distance_to(filtered_pt)
                return self._contain("CORRIDOR_BREACH", f"robot is {distance:.2f} m outside the route corridor")

        elif reading.sensor == "LOCALIZATION":
            point = values["position_m"]
            self.last_localization = (float(point[0]), float(point[1]))
            yaw = float(values.get("yaw_rad", 0.0))

            # Deterministic cross-check against odometry
            if self.last_odometry is not None:
                disagreement = hypot(point[0] - self.last_odometry[0], point[1] - self.last_odometry[1])
                if disagreement > self.limits.max_localization_jump_m:
                    return self._contain(
                        "LOCALIZATION_ATTACK",
                        f"localization disagreement is {disagreement:.2f} m",
                        evidence={"disagreement_m": f"{disagreement:.2f}"},
                    )

            if not self.ekf_initialized:
                self.ekf.x[0] = point[0]
                self.ekf.x[1] = point[1]
                self.ekf.x[6] = yaw
                self.ekf_initialized = True
                return not self.safe_stop.is_stopped

            if self.limits.enable_statistical_monitors:
                z_loc = np.array([point[0], point[1]], dtype=float)
                R_loc = np.diag([0.15**2, 0.15**2])

                def h_loc(x):
                    return np.array([x[0], x[1]], dtype=float)

                def H_loc(x):
                    H = np.zeros((2, 11), dtype=float)
                    H[0, 0] = 1.0
                    H[1, 1] = 1.0
                    return H

                y, S, H, PHT = self.ekf.compute_innovation(z_loc, H_loc, h_loc, R_loc)
                m = 2

                fast_sig = self.chi_square.evaluate("LOCALIZATION", t, y, S, m)
                slow_sig = self.cusum.evaluate("LOCALIZATION", t, y, S, m)

                self.saarm.record_residual("LOCALIZATION", fast_sig.nis)

                if fast_sig.verdict == FastVerdict.REJECT:
                    self._consecutive_localization_outliers += 1
                    if (
                        self._consecutive_localization_outliers >= self.limits.max_consecutive_outliers
                        or fast_sig.nis > 100.0
                    ):
                        return self._contain(
                            "LOCALIZATION_ATTACK",
                            f"Chi-Square gate rejected localization (NIS={fast_sig.nis:.2f}, limit={fast_sig.threshold:.2f})",
                            evidence={
                                "nis": f"{fast_sig.nis:.2f}",
                                "consecutive_outliers": str(self._consecutive_localization_outliers),
                            },
                        )
                    # Quarantined this single frame from EKF
                    return not self.safe_stop.is_stopped
                else:
                    self._consecutive_localization_outliers = 0

                if slow_sig.status == DriftStatus.ALARM:
                    return self._contain(
                        "SENSOR_DRIFT_ATTACK",
                        f"CUSUM cumulative drift alarm on LOCALIZATION (CUSUM={slow_sig.cusum_value:.2f}, limit={slow_sig.alarm_threshold:.2f})",
                        evidence={"cusum": f"{slow_sig.cusum_value:.2f}"},
                    )

                if fast_sig.verdict == FastVerdict.ACCEPT and slow_sig.accepted:
                    self.ekf.apply_update(y, S, H, PHT, R_loc)

        return not self.safe_stop.is_stopped

    def check_health(self, now: Optional[float] = None) -> bool:
        """Stop if a robot that began moving loses odometry freshness."""
        now = time.time() if now is None else now
        last_odometry_at = self.last_sensor_at.get("ODOMETRY")
        if last_odometry_at is not None and now - last_odometry_at > self.limits.sensor_timeout_s:
            return self._contain("TELEMETRY_LOSS", "odometry telemetry timed out")
        return not self.safe_stop.is_stopped

    def _contain(
        self,
        category: str,
        reason: str,
        lockdown: bool = False,
        evidence: Optional[Dict[str, str]] = None,
    ) -> bool:
        already_stopped = self.safe_stop.is_stopped
        signal = self.safe_stop.stop(reason, lockdown=lockdown)
        if already_stopped:
            return False
        self.bus.send(signal.to_bytes(), "SAFE_STOP")
        action_accepted = self.control.request_safe_stop(reason)
        ev = {"controller_request_accepted": str(action_accepted).lower()}
        if evidence:
            ev.update(evidence)
        alert = self.alerts.create(
            category,
            reason,
            signal.position_m,
            evidence=ev,
            severity="EMERGENCY" if lockdown else "CRITICAL",
        )
        self.bus.send(alert.to_bytes(), "FLEET_ALERT")
        return False
