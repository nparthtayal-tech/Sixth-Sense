import streamlit as st
import numpy as np
import socket
import threading
import time
import pandas as pd
import json

# Setup Streamlit page
st.set_page_config(page_title="SensorSentry HIL Dashboard", layout="wide")

# Global state using Streamlit cache to share data between threads and sessions
@st.cache_resource
def get_state():
    return {
        "true_pos": np.array([0.0, 0.0]),
        "true_vel": np.array([0.0, 0.0]),
        "gps_hacked": np.array([0.0, 0.0]),
        "fused_pos": np.array([0.0, 0.0]),
        
        # Phone 3D Tracking & Connection
        "phone_pos_3d": np.array([0.0, 0.0, 0.0]),
        "phone_vel_3d": np.array([0.0, 0.0, 0.0]),
        "raw_accel": np.array([0.0, 0.0, 0.0]),
        "phone_connected": False,
        "last_phone_time": 0.0,
        
        # EKF State
        "X_A": np.zeros(4), # Filter A (GPS+IMU)
        "P_A": np.eye(4),
        "X_B": np.zeros(4), # Filter B (Inertial Only)
        "P_B": np.eye(4),
        
        "chi2": 0.0,
        "cusum": 0.0,
        "attack_active": False,
        "spoofed_detected": False,
        "filter_mode": "FUSED GPS+IMU",
        
        "history": [],
        "last_gps_time": 0.0,
        "lock": threading.Lock()
    }

state = get_state()

# Math Engine & Simulation Thread
@st.cache_resource
def start_engine(_state):
    state = _state
    def engine():
        dt = 0.01 # 100Hz IMU
        
        # EKF Matrices
        F = np.array([
            [1, 0, dt, 0],
            [0, 1, 0, dt],
            [0, 0, 1, 0],
            [0, 0, 0, 1]
        ])
        B = np.array([
            [0.5*dt**2, 0],
            [0, 0.5*dt**2],
            [dt, 0],
            [0, dt]
        ])
        H = np.array([
            [1, 0, 0, 0],
            [0, 1, 0, 0]
        ])
        Q = np.eye(4) * 0.001
        R = np.eye(2) * 2.0
        
        spoof_drift = np.array([0.0, 0.0])
        
        while True:
            with state["lock"]:
                current_time = time.time()
                # Read from Bridge script
                try:
                    import os
                    if os.path.exists("imu_data.json"):
                        with open("imu_data.json", "r") as f:
                            b_data = json.load(f)
                            if current_time - b_data["time"] < 1.0:
                                state["phone_connected"] = True
                                ax, ay, az = b_data["ax"], b_data["ay"], b_data["az"]
                                
                                # Deadband and gravity
                                az_linear = az - 9.81
                                def deadband(val, threshold=0.1): return val if abs(val) > threshold else 0.0
                                accel = np.array([deadband(ax), deadband(ay), deadband(az_linear)])
                                
                                # Integrate
                                state["phone_vel_3d"] += accel * dt
                                state["phone_vel_3d"] *= 0.90
                                state["phone_pos_3d"] += state["phone_vel_3d"] * dt
                                state["raw_accel"] = accel
                            else:
                                state["phone_connected"] = False
                except Exception:
                    state["phone_connected"] = False

                if state["phone_connected"]:
                    # 1. Use REAL Phone Physics (No simulation)
                    state["true_pos"] = state["phone_pos_3d"][:2]
                    imu_reading = state["raw_accel"][:2]
                    
                    # 2. Attack Logic (Slow-Drift Spoofing)
                    if state["attack_active"]:
                        spoof_drift += np.array([0.02, -0.01]) * dt # Millimeters per second drift
                    else:
                        spoof_drift = np.array([0.0, 0.0])
                        
                    # 3. Predict Step (100 Hz) for both filters
                    state["X_A"] = F @ state["X_A"] + B @ imu_reading
                    state["P_A"] = F @ state["P_A"] @ F.T + Q
                    
                    state["X_B"] = F @ state["X_B"] + B @ imu_reading
                    state["P_B"] = F @ state["P_B"] @ F.T + Q
                    
                    # 4. GPS Update Step (1 Hz)
                    if current_time - state["last_gps_time"] >= 1.0:
                        state["last_gps_time"] = current_time
                        
                        # True GPS + Noise + Attack Drift
                        gps_reading = state["true_pos"] + np.random.normal(0, 0.5, 2) + spoof_drift
                        state["gps_hacked"] = gps_reading
                        
                        # Compute Residual
                        y = gps_reading - H @ state["X_A"]
                        S = H @ state["P_A"] @ H.T + R
                        S_inv = np.linalg.inv(S)
                        
                        # Chi-Squared Outlier Gating
                        chi2 = y.T @ S_inv @ y
                        state["chi2"] = chi2
                        
                        # CUSUM Anomaly Detector
                        residual_norm = np.sqrt(chi2)
                        drift_bias = 1.0 # Expected normal deviation
                        state["cusum"] = max(0.0, state["cusum"] + residual_norm - drift_bias)
                        
                        if state["cusum"] > 5.0:
                            state["spoofed_detected"] = True
                            state["filter_mode"] = "INERTIAL ONLY"
                        
                        # Apply GPS update ONLY to Filter A if Chi-Square passes
                        if chi2 < 5.991:
                            K = state["P_A"] @ H.T @ S_inv
                            state["X_A"] = state["X_A"] + K @ y
                            state["P_A"] = (np.eye(4) - K @ H) @ state["P_A"]
                            
                    # 5. Output SAARM Selection
                    if state["spoofed_detected"]:
                        state["fused_pos"] = state["X_B"][:2]
                    else:
                        state["fused_pos"] = state["X_A"][:2]
                        # Sync Filter B to Filter A while system is secure
                        state["X_B"] = state["X_A"].copy()
                        state["P_B"] = state["P_A"].copy()
                        
                    # Log history (downsampled for UI rendering)
                    if np.random.rand() < 0.1: # Save ~10Hz
                        state["history"].append({
                            "time": current_time,
                            "true_x": state["true_pos"][0],
                            "true_y": state["true_pos"][1],
                            "hacked_x": state["gps_hacked"][0],
                            "hacked_y": state["gps_hacked"][1],
                            "fused_x": state["fused_pos"][0],
                            "fused_y": state["fused_pos"][1],
                            "phone_x": state["phone_pos_3d"][0],
                            "phone_y": state["phone_pos_3d"][1],
                            "phone_z": state["phone_pos_3d"][2],
                            "chi2": state["chi2"],
                            "cusum": state["cusum"]
                        })
                        # Keep last 500 samples
                        if len(state["history"]) > 500:
                            state["history"].pop(0)

            time.sleep(dt)

    t = threading.Thread(target=engine, daemon=True)
    t.start()
    return t

start_engine(state)



# --- UI Render ---
st.title("🛡️ SensorSentry: HIL Cyber-Security Dashboard")

# Intruder Panel
with st.sidebar:
    st.header("📱 Hardware Connection")
    active_port = state.get("active_udp_port", 5005)
    if state["phone_connected"]:
        st.success(f"🟢 Phone Connected (Live UDP {active_port})")
    else:
        st.error(f"🔴 Phone Disconnected. Waiting for UDP {active_port}...")
        st.write(f"*Connect HyperIMU to this PC's IP on port {active_port}.*")
    
    st.markdown("---")
    st.header("🏴‍☠️ Intruder Controls")
    st.write("Simulate Remote Hack. Use local server IP to access this button remotely.")
    if st.button("ACTIVATE SLOW-DRIFT SPOOFING"):
        with state["lock"]:
            state["attack_active"] = True
    if st.button("RESET SYSTEM"):
        with state["lock"]:
            state["attack_active"] = False
            state["spoofed_detected"] = False
            state["filter_mode"] = "FUSED GPS+IMU"
            state["cusum"] = 0.0
            state["X_A"] = np.zeros(4)
            state["X_B"] = np.zeros(4)
            state["true_pos"] = np.zeros(2)
            state["phone_pos_3d"] = np.zeros(3)
            state["phone_vel_3d"] = np.zeros(3)
            state["history"] = []

    st.markdown("---")
    st.header("⚙️ Map Settings")
    state["map_scale"] = st.slider("Map Movement Scale", 1, 1000, 100, help="Multiply your hand movement to simulate a drone flying across the city!")

# System HUD
col1, col2, col3 = st.columns(3)
with col1:
    if state["spoofed_detected"]:
        st.error("🚨 SPOOFING DETECTED! ISOLATING GPS!")
    else:
        st.success("✅ SYSTEM SECURE")
        
with col2:
    if state["filter_mode"] == "INERTIAL ONLY":
        st.warning(f"Nav Source: **{state['filter_mode']}**")
    else:
        st.info(f"Nav Source: **{state['filter_mode']}**")

with col3:
    if state["attack_active"]:
        st.warning("⚠️ Attack In Progress")
    else:
        st.write("No Active Attack")

# Live Charts
st.subheader("🗺️ Live Satellite Tracker")
if len(state["history"]) > 0:
    df = pd.DataFrame(state["history"])
    
    # Base location (Chennai, India)
    BASE_LAT = 13.0827
    BASE_LON = 80.2707
    
    LAT_PER_METER = 1.0 / 111111.0
    LON_PER_METER = 1.0 / (111111.0 * np.cos(np.radians(BASE_LAT)))
    
    # Scale physical phone movement to look like a drone flying
    scale = state.get("map_scale", 100)
    
    current_true_x = df['true_x'].iloc[-1] * scale
    current_true_y = df['true_y'].iloc[-1] * scale
    
    current_hacked_x = df['hacked_x'].iloc[-1] * scale
    current_hacked_y = df['hacked_y'].iloc[-1] * scale
    
    current_fused_x = df['fused_x'].iloc[-1] * scale
    current_fused_y = df['fused_y'].iloc[-1] * scale
    
    map_data = pd.DataFrame({
        "lat": [
            BASE_LAT + (current_true_x * LAT_PER_METER),
            BASE_LAT + (current_hacked_x * LAT_PER_METER),
            BASE_LAT + (current_fused_x * LAT_PER_METER)
        ],
        "lon": [
            BASE_LON + (current_true_y * LON_PER_METER),
            BASE_LON + (current_hacked_y * LON_PER_METER),
            BASE_LON + (current_fused_y * LON_PER_METER)
        ],
        "color": [
            "#00ff00", # True (Green)
            "#0000ff", # Hacked (Blue)
            "#ff8c00"  # Fused (Orange)
        ],
        "size": [50, 30, 40]
    })
    
    # This renders a native interactive geographic map (like Google Maps)
    st.map(map_data, latitude="lat", longitude="lon", color="color", size="size", zoom=19)
    st.caption("🟢 True Drone Position (Driven by Phone) | 🔵 Spoofed GPS | 🟠 System Output (Secured)")
    
    st.markdown("### 📍 Live Phone Coordinates")
    st.info(f"**X (Left/Right):** {df['true_y'].iloc[-1]:.3f} m  |  **Y (Forward/Back):** {df['true_x'].iloc[-1]:.3f} m  |  **Z (Altitude):** {df['phone_z'].iloc[-1]:.3f} m")
        
    st.subheader("Statistical Anomalies")
    colA, colB = st.columns(2)
    with colA:
        st.write("Chi-Squared Outlier Gating (Threshold = 5.991)")
        st.line_chart(df["chi2"])
    with colB:
        st.write("CUSUM Drift Accumulator")
        st.line_chart(df["cusum"])

# Auto-refresh UI (~10 FPS for buttery smooth tracking)
time.sleep(0.1)
st.rerun()
