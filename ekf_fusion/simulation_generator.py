"""
simulation_generator.py
========================
Generates a realistic 3D benchmark scenario:
- Ground-truth 3D platform trajectory with smooth turns, accelerations, and elevation shifts
- Realistic receiver clock bias and drift progression
- Asynchronous timestamped measurements for all 8 sensor modalities
- Urban canyon GPS outage interval (t = 15s to 25s) to benchmark degraded navigation
- Visual landmarks, 5G gNodeB base stations, and reflective LiDAR surfaces
- Realistic Gaussian measurement noise and optional outlier glitches
"""

import numpy as np
from dataclasses import dataclass
from typing import List, Any
from ekf_fusion.sensor_models import SensorType, normalize_angle

@dataclass
class TrajectoryPoint:
    t: float
    p: np.ndarray      # [px, py, pz]
    v: np.ndarray      # [vx, vy, vz]
    a: np.ndarray      # [ax, ay, az]
    psi: float         # heading (rad)
    omega_z: float     # turn rate (rad/s)
    b_clk: float       # clock bias (m)
    d_clk: float       # clock drift (m/s)
    b_gyro: float      # gyro bias (rad/s)
    b_accel: float     # accel bias (m/s^2)

@dataclass
class SensorEvent:
    t: float
    sensor_type: SensorType
    z: np.ndarray
    args: tuple
    R: np.ndarray

class TrajectorySimulator:
    def __init__(self, duration=40.0, dt_sim=0.005, random_seed=42):
        self.duration = duration
        self.dt_sim = dt_sim
        self.rng = np.random.RandomState(random_seed)
        
        # Static landmarks in environment
        self.optical_landmarks = [
            np.array([40.0, 30.0, 6.0]),
            np.array([10.0, 60.0, 8.0]),
            np.array([-30.0, 40.0, 5.0]),
            np.array([-20.0, -20.0, 7.0]),
            np.array([30.0, -30.0, 6.0])
        ]
        
        # 5G gNodeB base station antennas (micro-cell network)
        self.gnodeb_stations = [
            np.array([60.0, 60.0, 20.0]),
            np.array([-60.0, 60.0, 18.0]),
            np.array([70.0, -50.0, 22.0]),
            np.array([-50.0, -60.0, 19.0])
        ]
        
        # Obstacles / reflective walls for Ultrasonic & LiDAR
        self.lidar_features = [
            np.array([25.0, 15.0, 1.5]),
            np.array([0.0, 35.0, 2.0]),
            np.array([-25.0, 10.0, 1.0]),
            np.array([-10.0, -15.0, 2.5]),
            np.array([20.0, -10.0, 1.2])
        ]
        
        self.ground_truth = []
        self._generate_ground_truth()

    def _generate_ground_truth(self):
        """Generates continuous smooth 3D trajectory."""
        n_steps = int(self.duration / self.dt_sim) + 1
        
        # True initial parameters
        px, py, pz = 0.0, 0.0, 1.0
        vx, vy, vz = 5.0, 0.0, 0.0
        psi = 0.0
        
        # Clock parameters
        b_clk = 25.0       # Initial bias: 25 meters (~83 nanoseconds)
        d_clk = 0.015      # Drift: 0.015 m/s (~5e-11 fractional offset)
        
        # Sensor biases
        b_gyro = 0.002     # 0.002 rad/s
        b_accel = 0.04     # 0.04 m/s^2
        
        speed = 6.0
        
        for k in range(n_steps):
            t = k * self.dt_sim
            
            # Trajectory dynamics: smooth sinusoidal turning pattern
            # Figure-8 / S-curve path
            omega_z = 0.18 * np.sin(0.25 * t) + 0.05 * np.cos(0.08 * t)
            psi = normalize_angle(psi + omega_z * self.dt_sim)
            
            # Elevation oscillation
            vz = 0.3 * np.cos(0.3 * t)
            pz = 1.0 + 0.8 * np.sin(0.3 * t)
            
            # Velocity aligned with heading
            vx = speed * np.cos(psi)
            vy = speed * np.sin(psi)
            
            # Position integration
            px += vx * self.dt_sim
            py += vy * self.dt_sim
            
            # Acceleration
            ax = -speed * omega_z * np.sin(psi)
            ay = speed * omega_z * np.cos(psi)
            az = -0.09 * np.sin(0.3 * t)
            
            # Clock evolution (with small random walk in drift)
            d_clk += 1e-6 * self.rng.randn() * np.sqrt(self.dt_sim)
            b_clk += d_clk * self.dt_sim
            
            pt = TrajectoryPoint(
                t=t,
                p=np.array([px, py, pz]),
                v=np.array([vx, vy, vz]),
                a=np.array([ax, ay, az]),
                psi=psi,
                omega_z=omega_z,
                b_clk=b_clk,
                d_clk=d_clk,
                b_gyro=b_gyro,
                b_accel=b_accel
            )
            self.ground_truth.append(pt)

    def generate_sensor_events(self, enable_gps_outage=True, outage_start=15.0, outage_end=25.0) -> List[SensorEvent]:
        """
        Generates realistic asynchronous measurement streams for all 8 sensors.
        Returns a time-sorted list of SensorEvent instances.
        """
        events = []
        
        # Sensor sampling rates
        rates = {
            SensorType.IMU: 100.0,           # 100 Hz
            SensorType.WHEEL_ENCODER: 50.0,  # 50 Hz
            SensorType.ULTRASONIC: 20.0,     # 20 Hz
            SensorType.CAMERA: 20.0,         # 20 Hz
            SensorType.LIDAR: 10.0,          # 10 Hz
            SensorType.FIVE_G: 5.0,          # 5 Hz
            SensorType.GNSS: 5.0,            # 5 Hz
            SensorType.ATOMIC_CLOCK: 1.0     # 1 Hz
        }
        
        # Noise parameters (1-sigma)
        sigma = {
            'gnss_pos': 1.2,
            'gnss_alt': 2.0,
            'gnss_vel': 0.1,
            'imu_gyro': 0.003,
            'imu_acc': 0.04,
            'wheel_v': 0.04,
            'ultra_d': 0.025,
            'cam_ang': 0.006,     # ~0.34 deg
            'lidar_r': 0.03,      # 3 cm
            'lidar_ang': 0.003,   # 0.17 deg
            'five_g_pr': 0.6,     # 60 cm
            'atomic_drift': 1e-4  # 0.1 mm/s
        }
        
        sim_points = {round(pt.t, 4): pt for pt in self.ground_truth}
        
        # 1. IMU events (100 Hz)
        dt_imu = 1.0 / rates[SensorType.IMU]
        t = 0.0
        while t <= self.duration:
            pt = self._interpolate_gt(t)
            # Body forward acceleration: a_fwd = ax * cos(psi) + ay * sin(psi)
            a_body = pt.a[0] * np.cos(pt.psi) + pt.a[1] * np.sin(pt.psi)
            meas_omega = pt.omega_z + pt.b_gyro + sigma['imu_gyro'] * self.rng.randn()
            meas_accel = a_body + pt.b_accel + sigma['imu_acc'] * self.rng.randn()
            
            z = np.array([meas_omega, meas_accel])
            R = np.diag([sigma['imu_gyro']**2, sigma['imu_acc']**2])
            events.append(SensorEvent(t=t, sensor_type=SensorType.IMU, z=z,
                                      args=(pt.omega_z, a_body), R=R))
            t += dt_imu
            
        # 2. Wheel Encoder events (50 Hz)
        dt_enc = 1.0 / rates[SensorType.WHEEL_ENCODER]
        t = 0.0
        while t <= self.duration:
            pt = self._interpolate_gt(t)
            v_fwd = pt.v[0] * np.cos(pt.psi) + pt.v[1] * np.sin(pt.psi)
            meas_v = v_fwd + sigma['wheel_v'] * self.rng.randn()
            z = np.array([meas_v])
            R = np.array([[sigma['wheel_v']**2]])
            events.append(SensorEvent(t=t, sensor_type=SensorType.WHEEL_ENCODER, z=z, args=(), R=R))
            t += dt_enc
            
        # 3. Ultrasonic events (20 Hz)
        dt_ultra = 1.0 / rates[SensorType.ULTRASONIC]
        t = 0.0
        while t <= self.duration:
            pt = self._interpolate_gt(t)
            # Find closest obstacle from lidar_features
            dists = [np.linalg.norm(pt.p - feat) for feat in self.lidar_features]
            closest_idx = int(np.argmin(dists))
            min_dist = dists[closest_idx]
            
            # Ultrasonic has limited range ~ 5 meters
            if min_dist < 6.0:
                meas_dist = min_dist + sigma['ultra_d'] * self.rng.randn()
                z = np.array([max(0.1, meas_dist)])
                R = np.array([[sigma['ultra_d']**2]])
                events.append(SensorEvent(t=t, sensor_type=SensorType.ULTRASONIC, z=z,
                                          args=(self.lidar_features[closest_idx],), R=R))
            t += dt_ultra
            
        # 4. Camera visual tracking events (20 Hz)
        dt_cam = 1.0 / rates[SensorType.CAMERA]
        t = 0.0
        while t <= self.duration:
            pt = self._interpolate_gt(t)
            # Find visible optical landmark in front of camera (FOV 90 deg)
            for lm in self.optical_landmarks:
                dx = lm[0] - pt.p[0]
                dy = lm[1] - pt.p[1]
                dz = lm[2] - pt.p[2]
                dist = np.sqrt(dx*dx + dy*dy + dz*dz)
                if dist < 45.0: # within visual range
                    azimuth = normalize_angle(np.arctan2(dy, dx) - pt.psi)
                    if abs(azimuth) < (np.pi / 4.0): # inside camera FOV (+-45 deg)
                        d_xy = max(1e-4, np.sqrt(dx*dx + dy*dy))
                        elevation = np.arctan2(dz, d_xy)
                        z = np.array([
                            normalize_angle(azimuth + sigma['cam_ang'] * self.rng.randn()),
                            elevation + sigma['cam_ang'] * self.rng.randn()
                        ])
                        R = np.diag([sigma['cam_ang']**2, sigma['cam_ang']**2])
                        events.append(SensorEvent(t=t, sensor_type=SensorType.CAMERA, z=z,
                                                  args=(lm,), R=R))
                        break # Track primary landmark
            t += dt_cam
            
        # 5. LiDAR 3D spatial mapping events (10 Hz)
        dt_lidar = 1.0 / rates[SensorType.LIDAR]
        t = 0.0
        while t <= self.duration:
            pt = self._interpolate_gt(t)
            for feat in self.lidar_features:
                dx = feat[0] - pt.p[0]
                dy = feat[1] - pt.p[1]
                dz = feat[2] - pt.p[2]
                dist = np.sqrt(dx*dx + dy*dy + dz*dz)
                if dist < 30.0:
                    d_xy = max(1e-4, np.sqrt(dx*dx + dy*dy))
                    azimuth = normalize_angle(np.arctan2(dy, dx) - pt.psi)
                    elevation = np.arcsin(np.clip(dz / dist, -0.999, 0.999))
                    
                    z = np.array([
                        dist + sigma['lidar_r'] * self.rng.randn(),
                        normalize_angle(azimuth + sigma['lidar_ang'] * self.rng.randn()),
                        elevation + sigma['lidar_ang'] * self.rng.randn()
                    ])
                    R = np.diag([sigma['lidar_r']**2, sigma['lidar_ang']**2, sigma['lidar_ang']**2])
                    events.append(SensorEvent(t=t, sensor_type=SensorType.LIDAR, z=z,
                                              args=(feat,), R=R))
                    break
            t += dt_lidar
            
        # 6. 5G Positioning Nodes events (5 Hz)
        dt_5g = 1.0 / rates[SensorType.FIVE_G]
        t = 0.0
        M_5g = len(self.gnodeb_stations)
        while t <= self.duration:
            pt = self._interpolate_gt(t)
            pseudo_ranges = []
            for gnb in self.gnodeb_stations:
                dist = np.linalg.norm(pt.p - gnb)
                pr = dist + pt.b_clk + sigma['five_g_pr'] * self.rng.randn()
                pseudo_ranges.append(pr)
            z = np.array(pseudo_ranges)
            R = np.eye(M_5g) * (sigma['five_g_pr']**2)
            events.append(SensorEvent(t=t, sensor_type=SensorType.FIVE_G, z=z,
                                      args=(self.gnodeb_stations,), R=R))
            t += dt_5g
            
        # 7. GNSS / GPS events (5 Hz) with simulated urban canyon outage
        dt_gnss = 1.0 / rates[SensorType.GNSS]
        t = 0.0
        while t <= self.duration:
            # Check GPS outage interval
            in_outage = enable_gps_outage and (outage_start <= t <= outage_end)
            if not in_outage:
                pt = self._interpolate_gt(t)
                z = np.array([
                    pt.p[0] + sigma['gnss_pos'] * self.rng.randn(),
                    pt.p[1] + sigma['gnss_pos'] * self.rng.randn(),
                    pt.p[2] + sigma['gnss_alt'] * self.rng.randn(),
                    pt.v[0] + sigma['gnss_vel'] * self.rng.randn(),
                    pt.v[1] + sigma['gnss_vel'] * self.rng.randn(),
                    pt.v[2] + sigma['gnss_vel'] * self.rng.randn()
                ])
                R = np.diag([
                    sigma['gnss_pos']**2, sigma['gnss_pos']**2, sigma['gnss_alt']**2,
                    sigma['gnss_vel']**2, sigma['gnss_vel']**2, sigma['gnss_vel']**2
                ])
                events.append(SensorEvent(t=t, sensor_type=SensorType.GNSS, z=z, args=(), R=R))
            t += dt_gnss
            
        # 8. Atomic Clock baseline events (1 Hz)
        dt_atomic = 1.0 / rates[SensorType.ATOMIC_CLOCK]
        t = 0.0
        while t <= self.duration:
            pt = self._interpolate_gt(t)
            meas_drift = pt.d_clk + sigma['atomic_drift'] * self.rng.randn()
            z = np.array([meas_drift])
            R = np.array([[sigma['atomic_drift']**2]])
            events.append(SensorEvent(t=t, sensor_type=SensorType.ATOMIC_CLOCK, z=z, args=(), R=R))
            t += dt_atomic
            
        # Sort all events chronologically
        events.sort(key=lambda ev: ev.t)
        return events

    def _interpolate_gt(self, t) -> TrajectoryPoint:
        """Finds closest ground truth point for given timestamp t."""
        idx = int(np.clip(t / self.dt_sim, 0, len(self.ground_truth) - 1))
        return self.ground_truth[idx]
