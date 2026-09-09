"""
ekf_fusion
==========
Multi-Sensor Asynchronous Extended Kalman Filter (EKF) package.
Fuses 8 heterogeneous sensor modalities using Roger Labbe's FilterPy:
- GNSS / GPS
- IMU
- Wheel Encoders
- Ultrasonic Sensors
- Cameras (Visual Tracking)
- LiDAR (3D Spatial Mapping)
- 5G Positioning Nodes
- Atomic Clock Baseline
"""

from .sensor_models import (
    SensorType,
    normalize_angle,
    angle_residual,
    gnss_h, gnss_H, gnss_default_R,
    imu_h, imu_H, imu_default_R,
    wheel_encoder_h, wheel_encoder_H, wheel_encoder_default_R,
    ultrasonic_h, ultrasonic_H, ultrasonic_default_R,
    camera_h, camera_H, camera_residual, camera_default_R,
    lidar_h, lidar_H, lidar_residual, lidar_default_R,
    five_g_h, five_g_H, five_g_default_R,
    atomic_clock_h, atomic_clock_H, atomic_clock_default_R
)

from .ekf_engine import MultiSensorEKF
from .simulation_generator import TrajectorySimulator, TrajectoryPoint, SensorEvent

__all__ = [
    'MultiSensorEKF',
    'SensorType',
    'TrajectorySimulator',
    'TrajectoryPoint',
    'SensorEvent',
    'normalize_angle',
    'angle_residual'
]
