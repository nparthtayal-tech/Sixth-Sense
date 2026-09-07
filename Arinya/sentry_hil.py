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
        "true_vel": np.array([1.0, 1.0]), # Moving straight line
        "gps_hacked": np.array([0.0, 0.0]),
        "fused_pos": np.array([0.0, 0.0]),
        
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
def start_engine(state):
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
                # 1. Simulate "True" Physics (Straight line trajectory)
                accel_true = np.array([0.0, 0.0])
                state["true_pos"] += state["true_vel"] * dt
                
                # Add IMU noise
                imu_reading = accel_true + np.random.normal(0, 0.1, 2)
                
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
                current_time = time.time()
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
                    # (Filter B never gets GPS updates, acting as Dead-Reckoning)
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

# Background UDP Listener for real phone IMU (Port 5000)
@st.cache_resource
def start_udp(state):
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.bind(("0.0.0.0", 5000))
    sock.setblocking(False)
    
    def listen():
        while True:
            try:
                data, addr = sock.recvfrom(1024)
                # This block parses real incoming JSON packets from Phone 1
                # Format expected: {"accel": [x,y,z], "gyro": [x,y,z]}
                payload = json.loads(data.decode())
                # Update logic would inject this into the Math Engine state
            except BlockingIOError:
                pass
            except Exception as e:
                pass
            time.sleep(0.01)
    t = threading.Thread(target=listen, daemon=True)
    t.start()
    return sock

start_udp(state)

# --- UI Render ---
st.title("🛡️ SensorSentry: HIL Cyber-Security Dashboard")

# Intruder Panel
with st.sidebar:
    st.header("🏴‍☠️ Intruder Controls")
    st.write("Simulate Phone 2 Remote Access. Use local server IP to access this button remotely.")
    if st.button("ACTIVATE SLOW-DRIFT SPOOFING", use_container_width=True):
        with state["lock"]:
            state["attack_active"] = True
    if st.button("RESET SYSTEM", use_container_width=True):
        with state["lock"]:
            state["attack_active"] = False
            state["spoofed_detected"] = False
            state["filter_mode"] = "FUSED GPS+IMU"
            state["cusum"] = 0.0
            state["X_A"] = np.zeros(4)
            state["X_B"] = np.zeros(4)
            state["true_pos"] = np.zeros(2)
            state["history"] = []

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
st.subheader("Live 2D Map Trajectory")
if len(state["history"]) > 0:
    df = pd.DataFrame(state["history"])
    
    # Render Trajectory Map
    st.line_chart(df[["true_x", "hacked_x", "fused_x"]], color=["#00ff00", "#0000ff", "#ff8c00"])
    
    st.subheader("Statistical Anomalies")
    colA, colB = st.columns(2)
    with colA:
        st.write("Chi-Squared Outlier Gating (Threshold = 5.991)")
        st.line_chart(df["chi2"])
    with colB:
        st.write("CUSUM Drift Accumulator")
        st.line_chart(df["cusum"])

# Auto-refresh UI (~2 FPS)
time.sleep(0.5)
st.rerun()
