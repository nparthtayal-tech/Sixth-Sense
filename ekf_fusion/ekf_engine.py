"""
ekf_engine.py
=============
Core MultiSensorEKF implementation wrapping Roger Labbe's FilterPy
ExtendedKalmanFilter. Supports:
- Dynamic measurement dimensions (m x 11) for any sensor
- Kinematic non-linear heading and velocity rotation propagation
- Continuous-to-discrete process noise Q(dt) scaling
- Asynchronous multi-rate event scheduling
- Joseph-form covariance updates for numerical stability
- Exposes raw innovation (y) and covariance (S) for external monitors
"""

import sys
import os
import numpy as np
import scipy.linalg as linalg

# Import Roger Labbe's ExtendedKalmanFilter from local filterpy-master
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(BASE_DIR, 'filterpy-master'))
from filterpy.kalman import ExtendedKalmanFilter

from ekf_fusion.sensor_models import SensorType, normalize_angle

class MultiSensorEKF:
    """
    Unified 11-State Asynchronous Multi-Sensor Extended Kalman Filter.
    Wraps and extends FilterPy's ExtendedKalmanFilter.
    
    Exposes innovation (y) and innovation covariance (S) after each
    update so external decision-making modules can consume them.
    """
    def __init__(self,
                 initial_x=None,
                 initial_P=None,
                 q_accel=0.3,
                 q_gyro=0.015,
                 q_clk_drift=1e-4,
                 q_bias=1e-5):
        
        self.dim_x = 11
        self.ekf = ExtendedKalmanFilter(dim_x=self.dim_x, dim_z=1)
        
        if initial_x is not None:
            self.ekf.x = np.array(initial_x, dtype=float).reshape(self.dim_x)
        else:
            self.ekf.x = np.zeros(self.dim_x, dtype=float)
            
        if initial_P is not None:
            self.ekf.P = np.array(initial_P, dtype=float).reshape((self.dim_x, self.dim_x))
        else:
            self.ekf.P = np.diag([
                4.0, 4.0, 4.0,
                1.0, 1.0, 1.0,
                0.05,
                25.0,
                0.01,
                0.001,
                0.01
            ])
            
        self.q_accel = q_accel
        self.q_gyro = q_gyro
        self.q_clk_drift = q_clk_drift
        self.q_bias = q_bias
        
        self.last_omega_z = 0.0
        self.last_a_fwd = 0.0
        
        self._I = np.eye(self.dim_x, dtype=float)
        self.current_time = 0.0
        
        # Latest innovation data exposed for external monitors
        self.last_innovation_y = None
        self.last_innovation_S = None
        self.last_sensor_name = None
        self.last_dim_m = 0

    @property
    def x(self):
        return self.ekf.x

    @property
    def P(self):
        return self.ekf.P

    def compute_F(self, dt, d_psi=0.0):
        F = np.eye(self.dim_x, dtype=float)
        c = np.cos(d_psi)
        s = np.sin(d_psi)
        F[0, 3] = dt * c
        F[0, 4] = -dt * s
        F[1, 3] = dt * s
        F[1, 4] = dt * c
        F[2, 5] = dt
        F[3, 3] = c
        F[3, 4] = -s
        F[4, 3] = s
        F[4, 4] = c
        F[6, 9] = -dt
        F[7, 8] = dt
        return F

    def compute_Q(self, dt):
        dt = max(1e-6, dt)
        dt2 = dt * dt
        dt3 = dt2 * dt / 3.0
        dt2_2 = dt2 / 2.0
        
        Q = np.zeros((self.dim_x, self.dim_x), dtype=float)
        qa = self.q_accel ** 2
        
        for i in range(3):
            Q[i, i] = dt3 * qa
            Q[i, i+3] = dt2_2 * qa
            Q[i+3, i] = dt2_2 * qa
            Q[i+3, i+3] = dt * qa
            
        Q[6, 6] = (self.q_gyro ** 2) * dt
        
        qc = self.q_clk_drift ** 2
        Q[7, 7] = dt3 * qc + 1e-4 * dt
        Q[7, 8] = dt2_2 * qc
        Q[8, 7] = dt2_2 * qc
        Q[8, 8] = dt * qc
        
        Q[9, 9] = (self.q_bias ** 2) * dt
        Q[10, 10] = (self.q_bias ** 2) * dt
        
        return Q

    def predict(self, dt, omega_z=None, a_fwd=None):
        """Predict step: propagates state and covariance forward by dt."""
        if dt <= 0:
            return
            
        if omega_z is not None:
            self.last_omega_z = omega_z
        if a_fwd is not None:
            self.last_a_fwd = a_fwd
            
        effective_omega = self.last_omega_z - self.ekf.x[9]
        d_psi = effective_omega * dt
        
        new_psi = normalize_angle(self.ekf.x[6] + d_psi)
        self.ekf.x[6] = new_psi
        
        c = np.cos(d_psi)
        s = np.sin(d_psi)
        vx_rot = self.ekf.x[3] * c - self.ekf.x[4] * s
        vy_rot = self.ekf.x[3] * s + self.ekf.x[4] * c
        self.ekf.x[3] = vx_rot
        self.ekf.x[4] = vy_rot
        
        self.ekf.x[0] += self.ekf.x[3] * dt
        self.ekf.x[1] += self.ekf.x[4] * dt
        self.ekf.x[2] += self.ekf.x[5] * dt
        
        self.ekf.x[7] += self.ekf.x[8] * dt
        
        F = self.compute_F(dt, d_psi)
        Q = self.compute_Q(dt)
        self.ekf.P = np.dot(np.dot(F, self.ekf.P), F.T) + Q
        self.ekf.P = 0.5 * (self.ekf.P + self.ekf.P.T)
        
        self.current_time += dt

    def compute_innovation(self, z, H_func, h_func, R, args=(), residual_func=None):
        """
        Computes innovation y and covariance S WITHOUT modifying state.
        External monitors (Chi-Square, CUSUM) call this to inspect before update.
        
        Returns: (y, S, H, PHT)
        """
        z = np.atleast_1d(z).astype(float)
        R = np.atleast_2d(R).astype(float)
        
        x = self.ekf.x
        P = self.ekf.P
        
        H = H_func(x, *args)
        hx = h_func(x, *args)
        
        if residual_func is not None:
            y = residual_func(z, hx)
        else:
            y = z - hx
            
        PHT = np.dot(P, H.T)
        S = np.dot(H, PHT) + R
        
        return y, S, H, PHT

    def apply_update(self, y, S, H, PHT, R):
        """
        Applies Kalman state update using pre-computed innovation.
        Called ONLY when external monitors have approved the measurement.
        """
        R = np.atleast_2d(R).astype(float)
        x = self.ekf.x
        P = self.ekf.P
        
        try:
            SI = linalg.inv(S)
        except linalg.LinAlgError:
            SI = linalg.pinv(S)
            
        K = np.dot(PHT, SI)
        
        self.ekf.x = x + np.dot(K, y)
        self.ekf.x[6] = normalize_angle(self.ekf.x[6])
        
        I_KH = self._I - np.dot(K, H)
        self.ekf.P = np.dot(np.dot(I_KH, P), I_KH.T) + np.dot(np.dot(K, R), K.T)
        self.ekf.P = 0.5 * (self.ekf.P + self.ekf.P.T)

    def update_sensor(self, sensor_type, z, H_func, h_func, R,
                      args=(), residual_func=None):
        """
        Direct update (no external monitors). Computes innovation,
        stores it for external access, and applies update.
        
        Returns: True (always accepted in direct mode)
        """
        z = np.atleast_1d(z).astype(float)
        m = len(z)
        
        y, S, H, PHT = self.compute_innovation(z, H_func, h_func, R, args, residual_func)
        
        st_name = sensor_type.name if isinstance(sensor_type, SensorType) else str(sensor_type)
        self.last_innovation_y = y
        self.last_innovation_S = S
        self.last_sensor_name = st_name
        self.last_dim_m = m
        
        self.apply_update(y, S, H, PHT, R)
        return True
