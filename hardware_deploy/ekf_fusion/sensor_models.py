"""
sensor_models.py
================
Mathematical observation functions h(x), analytical Jacobians H(x),
residual functions, and default noise models for 8 sensor modalities:
1. GNSS / GPS (Absolute 3D Position & Velocity)
2. IMU (Angular Velocity & Body Accelerations)
3. Wheel Encoders / Joint Sensors (Forward Speed & Yaw Rate)
4. Ultrasonic Sensors (Local Obstacle / Ground Distance)
5. Cameras (Visual Landmark Bearing: Azimuth & Elevation)
6. LiDAR (3D Spherical Coordinate Mapping: Range, Azimuth, Elevation)
7. 5G Positioning Nodes (Cellular Timing Pseudo-Ranges / Multilateration)
8. Atomic Clock (Local Frequency & Drift Baseline)

State Vector (dim_x = 11):
  x[0]: px (position x, m)
  x[1]: py (position y, m)
  x[2]: pz (position z, m)
  x[3]: vx (velocity x, m/s)
  x[4]: vy (velocity y, m/s)
  x[5]: vz (velocity z, m/s)
  x[6]: psi (yaw/heading angle, rad)
  x[7]: b_clk (receiver clock bias, equivalent meters = c * delta_t)
  x[8]: d_clk (receiver clock drift, equivalent m/s = c * dot_delta_t)
  x[9]: b_gyro (gyroscope bias, rad/s)
  x[10]: b_accel (accelerometer bias, m/s^2)
"""

import numpy as np
from enum import Enum, auto

class SensorType(Enum):
    GNSS = auto()
    IMU = auto()
    WHEEL_ENCODER = auto()
    ULTRASONIC = auto()
    CAMERA = auto()
    LIDAR = auto()
    FIVE_G = auto()
    ATOMIC_CLOCK = auto()

def normalize_angle(angle):
    """Normalize angle to [-pi, pi]."""
    return (angle + np.pi) % (2.0 * np.pi) - np.pi

def angle_residual(z, hx):
    """Residual computation handling angular wrap-around."""
    res = np.atleast_1d(z - hx).astype(float)
    # If residual vector contains angles, wrap them
    # For general vectors, caller provides specific residual if needed
    for i in range(len(res)):
        res[i] = normalize_angle(res[i])
    return res

# ---------------------------------------------------------------------------
# 1. GNSS / GPS Receivers
# ---------------------------------------------------------------------------
def gnss_h(x):
    """
    Observation function for GNSS receiver.
    Measures 3D position and 3D velocity directly:
      z = [px, py, pz, vx, vy, vz]^T
    """
    return np.array([x[0], x[1], x[2], x[3], x[4], x[5]], dtype=float)

def gnss_H(x):
    """
    Jacobian matrix H for GNSS observation (6 x 11).
    """
    H = np.zeros((6, 11), dtype=float)
    H[0, 0] = 1.0  # px
    H[1, 1] = 1.0  # py
    H[2, 2] = 1.0  # pz
    H[3, 3] = 1.0  # vx
    H[4, 4] = 1.0  # vy
    H[5, 5] = 1.0  # vz
    return H

def gnss_default_R(sigma_pos=1.5, sigma_alt=2.5, sigma_vel=0.1):
    """Default GNSS measurement covariance (6 x 6)."""
    return np.diag([sigma_pos**2, sigma_pos**2, sigma_alt**2,
                    sigma_vel**2, sigma_vel**2, sigma_vel**2])

# ---------------------------------------------------------------------------
# 2. IMU (Inertial Measurement Unit)
# ---------------------------------------------------------------------------
def imu_h(x, cur_omega_z=0.0, cur_accel_body=0.0):
    """
    Observation function for IMU.
    Measures angular rate and forward body acceleration including estimated biases:
      z = [omega_z + b_gyro, a_body + b_accel]^T
    """
    omega_meas = cur_omega_z + x[9]
    accel_meas = cur_accel_body + x[10]
    return np.array([omega_meas, accel_meas], dtype=float)

def imu_H(x, cur_omega_z=0.0, cur_accel_body=0.0):
    """
    Jacobian matrix H for IMU observation (2 x 11).
    """
    H = np.zeros((2, 11), dtype=float)
    H[0, 9] = 1.0   # d(omega_meas)/d(b_gyro)
    H[1, 10] = 1.0  # d(accel_meas)/d(b_accel)
    return H

def imu_default_R(sigma_gyro=0.005, sigma_accel=0.05):
    """Default IMU measurement covariance (2 x 2)."""
    return np.diag([sigma_gyro**2, sigma_accel**2])

# ---------------------------------------------------------------------------
# 3. Wheel Encoders / Joint Sensors
# ---------------------------------------------------------------------------
def wheel_encoder_h(x):
    """
    Observation function for Wheel Encoders.
    Measures forward vehicle speed in heading direction and yaw rate:
      v_fwd = vx * cos(psi) + vy * sin(psi)
      omega = d_psi / dt (or gyro rate)
      z = [v_fwd, (vx^2 + vy^2)^0.5]^T or [v_fwd]
    Here z = [v_fwd] (1 x 1)
    """
    psi = x[6]
    v_fwd = x[3] * np.cos(psi) + x[4] * np.sin(psi)
    return np.array([v_fwd], dtype=float)

def wheel_encoder_H(x):
    """
    Analytical Jacobian matrix H for Wheel Encoder (1 x 11).
      d(v_fwd)/d(vx)  = cos(psi)
      d(v_fwd)/d(vy)  = sin(psi)
      d(v_fwd)/d(psi) = -vx * sin(psi) + vy * cos(psi)
    """
    psi = x[6]
    H = np.zeros((1, 11), dtype=float)
    H[0, 3] = np.cos(psi)
    H[0, 4] = np.sin(psi)
    H[0, 6] = -x[3] * np.sin(psi) + x[4] * np.cos(psi)
    return H

def wheel_encoder_default_R(sigma_v=0.05):
    """Default Wheel Encoder covariance (1 x 1)."""
    return np.array([[sigma_v**2]], dtype=float)

# ---------------------------------------------------------------------------
# 4. Ultrasonic Sensors (Local Range to Wall / Obstacle / Ground)
# ---------------------------------------------------------------------------
def ultrasonic_h(x, landmark_pos):
    """
    Observation function for Ultrasonic Sensor.
    Measures Euclidean distance from vehicle (px, py, pz) to known surface/obstacle:
      d = sqrt((px - x_w)^2 + (py - y_w)^2 + (pz - z_w)^2)
    """
    dx = x[0] - landmark_pos[0]
    dy = x[1] - landmark_pos[1]
    dz = x[2] - landmark_pos[2]
    dist = np.sqrt(dx*dx + dy*dy + dz*dz)
    return np.array([max(1e-4, dist)], dtype=float)

def ultrasonic_H(x, landmark_pos):
    """
    Analytical Jacobian matrix H for Ultrasonic Sensor (1 x 11).
      d(dist)/d(px) = (px - x_w) / dist
      d(dist)/d(py) = (py - y_w) / dist
      d(dist)/d(pz) = (pz - z_w) / dist
    """
    dx = x[0] - landmark_pos[0]
    dy = x[1] - landmark_pos[1]
    dz = x[2] - landmark_pos[2]
    dist = max(1e-4, np.sqrt(dx*dx + dy*dy + dz*dz))
    H = np.zeros((1, 11), dtype=float)
    H[0, 0] = dx / dist
    H[0, 1] = dy / dist
    H[0, 2] = dz / dist
    return H

def ultrasonic_default_R(sigma_dist=0.02):
    """Default Ultrasonic covariance (1 x 1). Accurate up to 3-5m."""
    return np.array([[sigma_dist**2]], dtype=float)

# ---------------------------------------------------------------------------
# 5. Cameras (Visual Landmark Bearing Tracking)
# ---------------------------------------------------------------------------
def camera_h(x, landmark_pos):
    """
    Observation function for Camera / Visual Tracking.
    Measures azimuth (alpha) and elevation (beta) angles to known optical landmark:
      dx = X_L - px,  dy = Y_L - py,  dz = Z_L - pz
      d_xy = sqrt(dx^2 + dy^2)
      alpha = normalize_angle(atan2(dy, dx) - psi)
      beta  = atan2(dz, d_xy)
      z = [alpha, beta]^T
    """
    dx = landmark_pos[0] - x[0]
    dy = landmark_pos[1] - x[1]
    dz = landmark_pos[2] - x[2]
    d_xy = max(1e-4, np.sqrt(dx*dx + dy*dy))
    
    alpha = normalize_angle(np.arctan2(dy, dx) - x[6])
    beta = np.arctan2(dz, d_xy)
    return np.array([alpha, beta], dtype=float)

def camera_H(x, landmark_pos):
    """
    Analytical Jacobian matrix H for Camera Observation (2 x 11).
    """
    dx = landmark_pos[0] - x[0]
    dy = landmark_pos[1] - x[1]
    dz = landmark_pos[2] - x[2]
    d_xy2 = max(1e-6, dx*dx + dy*dy)
    d_xy = np.sqrt(d_xy2)
    d2 = max(1e-6, d_xy2 + dz*dz)
    
    H = np.zeros((2, 11), dtype=float)
    # d(alpha)/d(px) = dy / d_xy^2
    # d(alpha)/d(py) = -dx / d_xy^2
    # d(alpha)/d(psi) = -1
    H[0, 0] = dy / d_xy2
    H[0, 1] = -dx / d_xy2
    H[0, 6] = -1.0
    
    # d(beta)/d(px) = (dx * dz) / (d^2 * d_xy)
    # d(beta)/d(py) = (dy * dz) / (d^2 * d_xy)
    # d(beta)/d(pz) = -d_xy / d^2
    H[1, 0] = (dx * dz) / (d2 * d_xy)
    H[1, 1] = (dy * dz) / (d2 * d_xy)
    H[1, 2] = -d_xy / d2
    return H

def camera_residual(z, hx):
    """Camera residual handling angular wrap on azimuth."""
    res = np.array([z[0] - hx[0], z[1] - hx[1]], dtype=float)
    res[0] = normalize_angle(res[0])
    return res

def camera_default_R(sigma_angle_rad=0.008):
    """Default Camera angular covariance (2 x 2) ~0.45 degrees."""
    return np.diag([sigma_angle_rad**2, sigma_angle_rad**2])

# ---------------------------------------------------------------------------
# 6. LiDAR (3D Spatial Mapping in Spherical Coordinates)
# ---------------------------------------------------------------------------
def lidar_h(x, point_pos):
    """
    Observation function for LiDAR.
    Measures range (r), azimuth (theta), and elevation (phi) to reflected surface:
      dx = X_p - px,  dy = Y_p - py,  dz = Z_p - pz
      r = sqrt(dx^2 + dy^2 + dz^2)
      d_xy = sqrt(dx^2 + dy^2)
      theta = normalize_angle(atan2(dy, dx) - psi)
      phi = asin(dz / r)
      z = [r, theta, phi]^T
    """
    dx = point_pos[0] - x[0]
    dy = point_pos[1] - x[1]
    dz = point_pos[2] - x[2]
    d_xy = max(1e-4, np.sqrt(dx*dx + dy*dy))
    r = max(1e-4, np.sqrt(dx*dx + dy*dy + dz*dz))
    
    theta = normalize_angle(np.arctan2(dy, dx) - x[6])
    phi = np.arcsin(np.clip(dz / r, -0.9999, 0.9999))
    return np.array([r, theta, phi], dtype=float)

def lidar_H(x, point_pos):
    """
    Analytical Jacobian matrix H for 3D LiDAR observation (3 x 11).
    """
    dx = point_pos[0] - x[0]
    dy = point_pos[1] - x[1]
    dz = point_pos[2] - x[2]
    d_xy2 = max(1e-6, dx*dx + dy*dy)
    d_xy = np.sqrt(d_xy2)
    r2 = max(1e-6, d_xy2 + dz*dz)
    r = np.sqrt(r2)
    
    H = np.zeros((3, 11), dtype=float)
    # Row 0: Range r
    H[0, 0] = -dx / r
    H[0, 1] = -dy / r
    H[0, 2] = -dz / r
    
    # Row 1: Azimuth theta
    H[1, 0] = dy / d_xy2
    H[1, 1] = -dx / d_xy2
    H[1, 6] = -1.0
    
    # Row 2: Elevation phi
    H[2, 0] = (dx * dz) / (r2 * d_xy)
    H[2, 1] = (dy * dz) / (r2 * d_xy)
    H[2, 2] = -d_xy / r2
    return H

def lidar_residual(z, hx):
    """LiDAR residual wrapping azimuth."""
    res = np.array([z[0] - hx[0], z[1] - hx[1], z[2] - hx[2]], dtype=float)
    res[1] = normalize_angle(res[1])
    return res

def lidar_default_R(sigma_r=0.03, sigma_ang=0.003):
    """Default LiDAR covariance (3 x 3). 3cm range, 0.17 deg angular."""
    return np.diag([sigma_r**2, sigma_ang**2, sigma_ang**2])

# ---------------------------------------------------------------------------
# 7. 5G Positioning Nodes (Cellular Timing Pseudo-Ranges)
# ---------------------------------------------------------------------------
def five_g_h(x, gnodeb_positions):
    """
    Observation function for 5G Positioning.
    Measures pseudo-range to M known 5G gNodeB base stations:
      rho_i = sqrt((px - x_i)^2 + (py - y_i)^2 + (pz - z_i)^2) + b_clk
    where b_clk is the receiver clock bias in equivalent range meters.
    """
    M = len(gnodeb_positions)
    hx = np.zeros(M, dtype=float)
    for i, gnb in enumerate(gnodeb_positions):
        dx = x[0] - gnb[0]
        dy = x[1] - gnb[1]
        dz = x[2] - gnb[2]
        dist = np.sqrt(dx*dx + dy*dy + dz*dz)
        hx[i] = dist + x[7]  # + b_clk
    return hx

def five_g_H(x, gnodeb_positions):
    """
    Analytical Jacobian matrix H for 5G Positioning (M x 11).
      d(rho_i)/d(px) = (px - x_i) / dist_i
      d(rho_i)/d(py) = (py - y_i) / dist_i
      d(rho_i)/d(pz) = (pz - z_i) / dist_i
      d(rho_i)/d(b_clk) = 1.0
    """
    M = len(gnodeb_positions)
    H = np.zeros((M, 11), dtype=float)
    for i, gnb in enumerate(gnodeb_positions):
        dx = x[0] - gnb[0]
        dy = x[1] - gnb[1]
        dz = x[2] - gnb[2]
        dist = max(1e-4, np.sqrt(dx*dx + dy*dy + dz*dz))
        H[i, 0] = dx / dist
        H[i, 1] = dy / dist
        H[i, 2] = dz / dist
        H[i, 7] = 1.0  # partial with respect to clock bias
    return H

def five_g_default_R(M, sigma_pr=0.8):
    """Default 5G pseudo-range covariance (M x M). Sub-meter timing accuracy."""
    return np.eye(M, dtype=float) * (sigma_pr**2)

# ---------------------------------------------------------------------------
# 8. Atomic Clock (Local High-Precision Timing Baseline)
# ---------------------------------------------------------------------------
def atomic_clock_h(x):
    """
    Observation function for onboard Atomic Clock baseline.
    Directly measures receiver clock drift state:
      z = [d_clk]
    Because atomic frequency standards (e.g. CSAC) provide fractional
    frequency stability ~1e-11 to 1e-12, the clock drift is ultra-tightly
    known and observable.
    """
    return np.array([x[8]], dtype=float)

def atomic_clock_H(x):
    """
    Analytical Jacobian matrix H for Atomic Clock observation (1 x 11).
      d(h)/d(d_clk) = 1.0
    """
    H = np.zeros((1, 11), dtype=float)
    H[0, 8] = 1.0  # d_clk
    return H

def atomic_clock_default_R(sigma_drift=1e-4):
    """
    Default Atomic Clock covariance (1 x 1).
    sigma_drift = 1e-4 m/s corresponds to delta_f/f ~ 3.3e-13.
    """
    return np.array([[sigma_drift**2]], dtype=float)
