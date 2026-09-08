"""
Software-In-The-Loop (SITL) MAVLink Drone Simulator
====================================================
Simulates a real Pixhawk flight controller outputting MAVLink v2
telemetry (IMU, GPS, Heartbeat) over UDP port 14550, and listens
for SensorSentry's autonomous failsafe overrides.
"""

import time
import sys
import os
import numpy as np
from pymavlink import mavutil

def run_sitl_simulator(target_endpoint="udpout:127.0.0.1:14550", spoof_at_seconds=4.0):
    print("=========================================================")
    print("== SENSORSENTRY MAVLINK SITL AUTOPILOT SIMULATOR ==")
    print("=========================================================")
    print(f"Broadcasting realistic Pixhawk telemetry to: {target_endpoint}")
    print(f"GPS Spoofing injection scheduled at: t = {spoof_at_seconds:.1f}s")
    print("---------------------------------------------------------")

    # Connect outgoing stream
    mav = mavutil.mavlink_connection(target_endpoint, source_system=1, source_component=1)

    start_time = time.time()
    last_heartbeat = 0.0
    last_gps = 0.0
    override_received = False

    t = 0.0
    dt_imu = 0.02  # 50 Hz IMU

    lat_base = 13.060421
    lon_base = 80.281054
    alt_base = 25.0

    while True:
        now = time.time()
        t = now - start_time
        
        # 1. Heartbeat at 1 Hz
        if now - last_heartbeat > 1.0:
            mav.mav.heartbeat_send(
                mavutil.mavlink.MAV_TYPE_QUADROTOR,
                mavutil.mavlink.MAV_AUTOPILOT_PX4,
                mavutil.mavlink.MAV_MODE_FLAG_SAFETY_ARMED | mavutil.mavlink.MAV_MODE_FLAG_CUSTOM_MODE_ENABLED,
                4, # Auto mission mode
                mavutil.mavlink.MAV_STATE_ACTIVE
            )
            last_heartbeat = now

        # 2. 50 Hz High-Rate IMU (Level forward flight)
        # Normal forward acceleration = 0.5 m/s^2, gravity = -9.81 m/s^2
        ax = 0.5 + np.random.normal(0, 0.05)
        ay = 0.0 + np.random.normal(0, 0.05)
        az = -9.81 + np.random.normal(0, 0.08)
        gx = np.random.normal(0, 0.002)
        gy = np.random.normal(0, 0.002)
        gz = np.random.normal(0, 0.002)
        
        time_usec = int(now * 1e6)
        mav.mav.highres_imu_send(
            time_usec,
            ax, ay, az,
            gx, gy, gz,
            0.0, 0.0, 0.0,  # mag
            1013.25,        # pressure
            0.0,            # diff pressure
            25.0,           # pressure alt
            24.0,           # temperature
            65535           # fields updated
        )

        # 3. 10 Hz GPS Global Position
        if now - last_gps > 0.1:
            is_spoofing = t >= spoof_at_seconds
            
            # Nominal path moves straight North (lat increments)
            curr_lat = lat_base + (t * 2.0) / 111319.5
            
            # If spoofing: maliciously drag Longitude (lateral East drift)
            if is_spoofing:
                curr_lon = lon_base + (25.0) / 111319.5  # +25 meters lateral deviation
                vx_fake = 1.0
                vy_fake = 8.5
            else:
                curr_lon = lon_base
                vx_fake = 2.0
                vy_fake = 0.0

            time_boot_ms = int(t * 1000) & 0xFFFFFFFF
            mav.mav.global_position_int_send(
                time_boot_ms,
                int(curr_lat * 1e7),
                int(curr_lon * 1e7),
                int((alt_base + 100.0) * 1e3),
                int(alt_base * 1e3),
                int(vx_fake * 100),
                int(vy_fake * 100),
                0,
                0 # heading
            )
            last_gps = now

            if is_spoofing and not override_received and t < spoof_at_seconds + 0.3:
                print(f"\n[SITL {t:.1f}s] >>> ATTACKER ACTIVATED GPS SPOOFING (+25m offset injected) <<<")

        # 4. Check for Incoming Autopilot Overrides from SensorSentry
        msg = mav.recv_match(blocking=False)
        if msg:
            mtype = msg.get_type()
            if mtype == "COMMAND_LONG":
                cmd = msg.command
                if cmd == mavutil.mavlink.MAV_CMD_DO_SET_MODE:
                    print(f"\n[SITL {t:.1f}s] ✅ AUTOPILOT OVERRIDE RECEIVED: MAV_CMD_DO_SET_MODE (Mode: {msg.param2})")
                    print("          Drone successfully engaged emergency safe hover / loiter!")
                    override_received = True
                    break
                elif cmd == mavutil.mavlink.MAV_CMD_NAV_RETURN_TO_LAUNCH:
                    print(f"\n[SITL {t:.1f}s] ✅ AUTOPILOT OVERRIDE RECEIVED: MAV_CMD_NAV_RETURN_TO_LAUNCH")
                    print("          Drone successfully executing autonomous Return to Launch!")
                    override_received = True
                    break
            elif mtype == "STATUSTEXT":
                print(f"[SITL GCS ALERT] {msg.text}")

        time.sleep(dt_imu)

    mav.close()
    if override_received:
        print("\n[SUCCESS] MAVLink SITL Loop Verified 100% Successfully.")
        print("SensorSentry detected the attack and commanded the autopilot override in real-time.\n")
        return True
    return False

if __name__ == "__main__":
    run_sitl_simulator()
