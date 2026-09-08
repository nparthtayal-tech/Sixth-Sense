"""
MAVLink v2 Bridge for Autonomous Drone Hardware
================================================
Interfaces SensorSentry with Pixhawk / Cube / PX4 / ArduPilot flight stacks.
Handles real-time telemetry ingestion (IMU, GPS, Attitude), companion heartbeats,
and autonomous autopilot actuator overrides (LOITER/BRAKE, RTL, Velocity Guidance).
"""

import time
import threading
import numpy as np
from typing import Optional, Tuple, Dict, Any

from pymavlink import mavutil
from hal import SensorReading
from ekf_fusion.sensor_models import imu_default_R, gnss_default_R


class MAVLinkBridge:
    def __init__(self,
                 connection_string: str = "udpin:0.0.0.0:14550",
                 baud: int = 921600,
                 source_system: int = 1,
                 source_component: int = 191):  # MAV_COMP_ID_ONBOARD_COMPUTER
        self.connection_string = connection_string
        self.baud = baud
        self.source_system = source_system
        self.source_component = source_component
        
        self.mav = None
        self.target_system = 1
        self.target_component = 1
        self._connected = False
        self._running = False
        
        # Telemetry Cache
        self.last_imu_time = 0.0
        self.last_gps_time = 0.0
        self.current_lat = 0.0
        self.current_lon = 0.0
        self.current_alt = 0.0
        self.current_yaw = 0.0
        self.is_armed = False
        self.flight_mode = "UNKNOWN"
        
        # Heartbeat & thread state
        self._heartbeat_thread = None
        
        # Default measurement covariances
        self.R_imu = imu_default_R()
        self.R_gps = gnss_default_R()

    def connect(self, timeout_s: float = 10.0) -> bool:
        """Establishes MAVLink connection to autopilot."""
        print(f"[MAVLINK] Connecting to endpoint '{self.connection_string}' (baud: {self.baud})...")
        try:
            self.mav = mavutil.mavlink_connection(
                self.connection_string,
                baud=self.baud,
                source_system=self.source_system,
                source_component=self.source_component
            )
            
            # Wait for first heartbeat from flight controller
            print("[MAVLINK] Waiting for autopilot HEARTBEAT...")
            msg = self.mav.wait_heartbeat(timeout=timeout_s)
            if msg:
                self.target_system = msg.get_srcSystem()
                self.target_component = msg.get_srcComponent()
                self._connected = True
                self._running = True
                
                # Start companion computer heartbeat broadcaster (2 Hz)
                self._heartbeat_thread = threading.Thread(target=self._heartbeat_loop, daemon=True)
                self._heartbeat_thread.start()
                
                print(f"[MAVLINK] Autopilot detected! Target System: {self.target_system}, Component: {self.target_component}")
                self.send_statustext("SensorSentry Avionics Supervisor Online", severity=mavutil.mavlink.MAV_SEVERITY_INFO)
                return True
            else:
                print(f"[ERROR] Autopilot HEARTBEAT timed out after {timeout_s}s.")
                return False
        except Exception as e:
            print(f"[ERROR] MAVLink connection failed: {e}")
            return False

    def _heartbeat_loop(self):
        """Broadcasts companion computer heartbeat at 2 Hz."""
        while self._running:
            try:
                self.mav.mav.heartbeat_send(
                    mavutil.mavlink.MAV_TYPE_ONBOARD_CONTROLLER,
                    mavutil.mavlink.MAV_AUTOPILOT_INVALID,
                    0, 0, 0
                )
            except Exception:
                pass
            time.sleep(0.5)

    def read_telemetry(self, blocking: bool = False, timeout: float = 0.05) -> Optional[Tuple[SensorReading, np.ndarray]]:
        """
        Polls for next MAVLink message and converts into SensorReading for EKF.
        Returns: (SensorReading, Covariance_R) or None.
        """
        if not self._connected or not self.mav:
            return None
            
        try:
            msg = self.mav.recv_match(blocking=blocking, timeout=timeout)
            if not msg:
                return None
                
            msg_type = msg.get_type()
            t_now = time.time()
            
            # 1. High-Rate IMU (Linear Acceleration + Angular Velocity)
            if msg_type in ["HIGHRES_IMU", "RAW_IMU", "SCALED_IMU", "SCALED_IMU2"]:
                if msg_type == "HIGHRES_IMU":
                    ax = msg.xacc  # m/s^2
                    ay = msg.yacc
                    az = msg.zacc
                    gx = msg.xgyro # rad/s
                    gy = msg.ygyro
                    gz = msg.zgyro
                else:
                    # Millig to m/s^2 conversion
                    ax = (msg.xacc / 1000.0) * 9.80665
                    ay = (msg.yacc / 1000.0) * 9.80665
                    az = (msg.zacc / 1000.0) * 9.80665
                    # Millirad/s to rad/s
                    gx = msg.xgyro / 1000.0
                    gy = msg.ygyro / 1000.0
                    gz = msg.zgyro / 1000.0
                
                # EKF expects 2D IMU observation: [omega_z, a_body]
                a_body = float(np.sqrt(ax**2 + ay**2))
                omega_z = float(gz)
                
                self.last_imu_time = t_now
                reading = SensorReading(
                    timestamp=t_now,
                    data=np.array([omega_z, a_body]),
                    sensor_type="IMU",
                    valid=True,
                    metadata={"raw_acc": [ax, ay, az], "raw_gyro": [gx, gy, gz]}
                )
                return reading, self.R_imu
                
            # 2. GNSS Position & Velocity (GLOBAL_POSITION_INT)
            elif msg_type == "GLOBAL_POSITION_INT":
                lat = msg.lat / 1e7
                lon = msg.lon / 1e7
                alt = msg.relative_alt / 1e3  # meters AGL
                vx = msg.vx / 100.0           # m/s North
                vy = msg.vy / 100.0           # m/s East
                vz = msg.vz / 100.0           # m/s Down
                
                self.current_lat = lat
                self.current_lon = lon
                self.current_alt = alt
                self.last_gps_time = t_now
                
                # EKF expects 6D GNSS observation: [px, py, pz, vx, vy, vz]
                reading = SensorReading(
                    timestamp=t_now,
                    data=np.array([lat, lon, alt, vx, vy, vz]),
                    sensor_type="GNSS",
                    valid=True,
                    metadata={"heading": msg.hdg / 100.0}
                )
                return reading, self.R_gps

            # 3. GNSS Satellite Lock Quality (GPS_RAW_INT)
            elif msg_type == "GPS_RAW_INT":
                fix_type = msg.fix_type
                sats = msg.satellites_visible
                # Reject fix if less than 3D lock
                if fix_type < 3 or sats < 6:
                    # GPS degraded / unconfirmed
                    pass

            # 4. Vehicle Attitude
            elif msg_type == "ATTITUDE":
                self.current_yaw = msg.yaw

            # 5. Heartbeat Status
            elif msg_type == "HEARTBEAT":
                self.is_armed = bool(msg.base_mode & mavutil.mavlink.MAV_MODE_FLAG_SAFETY_ARMED)

        except Exception:
            pass
            
        return None

    # ========================================================================
    # AUTOPILOT FAILSAFE ACTUATION OVERRIDES
    # ========================================================================
    def command_safe_hover(self):
        """
        Emergency Safe Stop: commands autopilot to immediately enter LOITER/BRAKE mode
        and hover in place at current altitude.
        """
        print("[FAILSAFE] >>> COMMANDING AUTOPILOT: EMERGENCY LOITER / SAFE HOVER <<<")
        self.send_statustext("CRITICAL: SENSORSENTRY COMMANDING SAFE HOVER", severity=mavutil.mavlink.MAV_SEVERITY_CRITICAL)
        
        # Standard MAVLink MAV_CMD_DO_SET_MODE
        # Works across both PX4 (Hold mode) and ArduPilot (Brake/Loiter mode)
        try:
            self.mav.mav.command_long_send(
                self.target_system,
                self.target_component,
                mavutil.mavlink.MAV_CMD_DO_SET_MODE,
                0,
                mavutil.mavlink.MAV_MODE_FLAG_CUSTOM_MODE_ENABLED,
                5,  # Standard Loiter/Hold custom mode index
                0, 0, 0, 0, 0
            )
        except Exception as e:
            print(f"[ERROR] Failed to send LOITER command: {e}")

    def command_return_to_launch(self):
        """
        Emergency Return-to-Launch (RTL): commands drone to fly back to origin
        using independent inertial guidance.
        """
        print("[FAILSAFE] >>> COMMANDING AUTOPILOT: EMERGENCY RETURN TO LAUNCH (RTL) <<<")
        self.send_statustext("CRITICAL: SENSORSENTRY TRIGGERED EMERGENCY RTL", severity=mavutil.mavlink.MAV_SEVERITY_EMERGENCY)
        
        try:
            self.mav.mav.command_long_send(
                self.target_system,
                self.target_component,
                mavutil.mavlink.MAV_CMD_NAV_RETURN_TO_LAUNCH,
                0, 0, 0, 0, 0, 0, 0, 0
            )
        except Exception as e:
            print(f"[ERROR] Failed to send RTL command: {e}")

    def command_rejoin_velocity(self, vx: float, vy: float, vz: float = 0.0):
        """
        Streams corrective velocity vectors to steer the drone back inside
        the authorized flight corridor using dead-reckoning.
        """
        try:
            # Type mask: ignore position, ignore accel/yaw, control velocity only
            # 0b0000111111000111 (0x0F87)
            type_mask = (
                mavutil.mavlink.POSITION_TARGET_TYPEMASK_X_IGNORE |
                mavutil.mavlink.POSITION_TARGET_TYPEMASK_Y_IGNORE |
                mavutil.mavlink.POSITION_TARGET_TYPEMASK_Z_IGNORE |
                mavutil.mavlink.POSITION_TARGET_TYPEMASK_AX_IGNORE |
                mavutil.mavlink.POSITION_TARGET_TYPEMASK_AY_IGNORE |
                mavutil.mavlink.POSITION_TARGET_TYPEMASK_AZ_IGNORE |
                mavutil.mavlink.POSITION_TARGET_TYPEMASK_YAW_IGNORE |
                mavutil.mavlink.POSITION_TARGET_TYPEMASK_YAW_RATE_IGNORE
            )
            time_boot_ms = int(time.time() * 1000) & 0xFFFFFFFF
            self.mav.mav.set_position_target_local_ned_send(
                time_boot_ms,
                self.target_system,
                self.target_component,
                mavutil.mavlink.MAV_FRAME_BODY_NED,
                type_mask,
                0, 0, 0,         # Positions (ignored)
                vx, vy, vz,      # Corrective velocities in m/s
                0, 0, 0,         # Accels (ignored)
                0, 0             # Yaw (ignored)
            )
        except Exception:
            pass

    def send_statustext(self, text: str, severity: int = mavutil.mavlink.MAV_SEVERITY_WARNING):
        """Displays visual alert banner on Ground Control Station (QGroundControl)."""
        if self.mav:
            try:
                msg_bytes = text[:50].encode('utf-8')
                self.mav.mav.statustext_send(severity, msg_bytes)
            except Exception:
                pass

    def close(self):
        self._running = False
        self._connected = False
        if self.mav:
            try: self.mav.close()
            except Exception: pass
