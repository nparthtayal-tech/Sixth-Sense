"""
SensorSentry — Spoofing Detection & Sensor Fusion
PS 18: Intelligent sensor-fusion system that cross-validates sensor data
to detect GPS spoofing, false readings, or manipulated environmental data.

Backend Server: Flask-SocketIO + UDP Listener + EKF Math Engine + Multi-Sensor Cross-Validation
"""

from flask import Flask, send_from_directory
from flask_socketio import SocketIO
import socket
import threading
import json
import time
import numpy as np
import os

# ============================================================================
# Flask App
# ============================================================================
app = Flask(__name__, static_folder='.')
app.config['SECRET_KEY'] = 'sensorsentry-arinya'
socketio = SocketIO(app, cors_allowed_origins="*", async_mode='threading')

# ============================================================================
# Global State
# ============================================================================
class SentryState:
    def __init__(self):
        self.lock = threading.Lock()
        
        # Connection
        self.is_connected = False
        self.last_packet_time = 0.0
        self.packets_received = 0
        
        # Threat flags
        self.spoofing_active = False
        self.imu_injection_active = False
        self.attack_detected = False
        self.attack_type = "none"  # "none", "gps_spoof", "imu_injection", "multi_vector"
        
        # Latest IMU Data (raw from phone)
        self.latest_imu = {
            "ax": 0.0, "ay": 0.0, "az": 0.0,
            "gx": 0.0, "gy": 0.0, "gz": 0.0
        }
        self.prev_imu = {
            "ax": 0.0, "ay": 0.0, "az": 0.0,
            "gx": 0.0, "gy": 0.0, "gz": 0.0
        }
        
        # Navigation State
        self.true_pos_x = 0.0
        self.true_pos_y = 0.0
        self.hacked_gps_x = 0.0
        self.hacked_gps_y = 0.0
        self.fused_pos_x = 0.0
        self.fused_pos_y = 0.0
        
        # Algorithm State
        self.ekf_residual = 0.0
        self.chi_squared = 0.0
        self.cusum = 0.0
        
        # Cross-Validation & Trust Scores
        self.gps_trust = 100.0
        self.imu_trust = 100.0
        self.system_integrity = 100.0
        self.gps_imu_agreement = True  # Do GPS and IMU agree?
        self.imu_anomaly_detected = False
        self.imu_spike_count = 0
        
        # Environmental Sensor (simulated)
        self.baro_altitude = 100.0  # meters
        self.gps_altitude = 100.0
        self.baro_gps_mismatch = False
        self.env_trust = 100.0
        
        # Threat Log
        self.threat_log = []  # [{time, type, severity, message}]
        self.max_log_entries = 50
        
        # Velocity (integrated from accelerometer)
        self.velocity = {"vx": 0.0, "vy": 0.0, "vz": 0.0}
        
        # FFT / Frequency Analysis
        self.accel_buffer_z = []  # Buffer for FFT computation
        self.fft_magnitudes = []  # Frequency bin magnitudes
        self.fft_freqs = []       # Frequency bin labels
        
        # Flight Corridor & Physical Hijack Detection
        self.corridor_half_width = 1.8   # meters allowable mission corridor
        self.phone_heading = 0.0         # degrees (integrated from phone gyro gz)
        self.phone_lateral_pos = 0.0    # meters (integrated from phone lateral accel ay)
        self.phone_lat_vel = 0.0        # m/s
        self.corridor_breached = False
        self.history_envelope_violation = False
        self.corridor_status = "NOMINAL" # "NOMINAL", "WARNING", "BREACHED"
        
        # Hacker Target (Malicious Spoof Destination)
        self.hacker_target_y = 15.0      # Target lateral offset in meters
        self.hacker_target_label = "Mountain Ambush (+15m)"
        
        # Simulation time
        self.sim_time = 0.0
    
    def add_threat(self, threat_type, severity, message):
        entry = {
            "time": round(self.sim_time, 1),
            "type": threat_type,
            "severity": severity,  # "info", "warning", "critical"
            "message": message
        }
        self.threat_log.append(entry)
        if len(self.threat_log) > self.max_log_entries:
            self.threat_log.pop(0)

state = SentryState()

# ============================================================================
# UDP Listener Thread (HyperIMU)
# ============================================================================
def udp_listener():
    UDP_IP = "0.0.0.0"
    UDP_PORT = 5000
    
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        sock.bind((UDP_IP, UDP_PORT))
        sock.settimeout(1.0)
    except Exception as e:
        print(f"[ERROR] Cannot bind UDP port {UDP_PORT}: {e}")
        print("[HINT] Close any other running instances (Streamlit, etc.)")
        return
    
    print(f"[UDP] Listening on port {UDP_PORT}...")
    
    while True:
        try:
            data, addr = sock.recvfrom(4096)
            # Forward copy to local port 5001 for demo_animated_sentry
            try:
                sock.sendto(data, ('127.0.0.1', 5001))
            except Exception:
                pass
            decoded = data.decode('utf-8', errors='ignore').strip()
            if not decoded:
                continue
            
            # Try JSON first
            try:
                packet = json.loads(decoded)
                with state.lock:
                    state.last_packet_time = time.time()
                    state.is_connected = True
                    state.packets_received += 1
                    state.prev_imu = dict(state.latest_imu)
                    for k in state.latest_imu:
                        if k in packet:
                            state.latest_imu[k] = float(packet[k])
                continue
            except (json.JSONDecodeError, ValueError):
                pass
            
            # Try CSV (HyperIMU default format)
            try:
                parts = decoded.replace('\r', '').replace('\n', '').split(',')
                vals = [float(p.strip()) for p in parts if p.strip()]
                
                with state.lock:
                    state.last_packet_time = time.time()
                    state.is_connected = True
                    state.packets_received += 1
                    state.prev_imu = dict(state.latest_imu)
                    
                    if len(vals) >= 6:
                        state.latest_imu["ax"] = vals[-6]
                        state.latest_imu["ay"] = vals[-5]
                        state.latest_imu["az"] = vals[-4]
                        state.latest_imu["gx"] = vals[-3]
                        state.latest_imu["gy"] = vals[-2]
                        state.latest_imu["gz"] = vals[-1]
                    elif len(vals) >= 3:
                        state.latest_imu["ax"] = vals[-3]
                        state.latest_imu["ay"] = vals[-2]
                        state.latest_imu["az"] = vals[-1]
                continue
            except (ValueError, IndexError):
                pass
            
        except socket.timeout:
            with state.lock:
                if state.last_packet_time > 0 and time.time() - state.last_packet_time > 2.0:
                    state.is_connected = False
        except Exception as e:
            print(f"[UDP] Error: {e}")
            break
    
    sock.close()

# ============================================================================
# EKF Math Engine + Cross-Validation + Multi-Attack Detection
# ============================================================================
def math_engine():
    print("[MATH] Engine started.")
    dt = 0.05  # 20 Hz
    
    # EKF State: [x, y, vx, vy]
    x_hat = np.array([0.0, 0.0, 1.0, 0.0])
    P = np.eye(4) * 0.1
    Q = np.eye(4) * 0.005
    R = np.eye(2) * 1.0
    H = np.array([[1, 0, 0, 0], [0, 1, 0, 0]])
    
    # Spoof parameters
    spoof_drift_rate = 0.5
    current_spoof_offset = np.array([0.0, 0.0])
    
    # CUSUM parameters
    cusum_threshold = 20.0
    cusum_drift = 0.0
    
    # IMU filter
    imu_filtered = np.array([0.0, 0.0])
    alpha = 0.3
    
    # Velocity state (integrated from accel)
    vel = np.array([0.0, 0.0, 0.0])
    vel_damping = 0.98  # Slight damping to prevent drift
    
    # Phone physical corridor tracking state
    phone_heading = 0.0
    phone_lat_pos = 0.0
    phone_lat_vel = 0.0
    corridor_breached_logged = False
    
    # FFT counter
    fft_counter = 0
    
    # Cross-validation state
    last_threat_log_time = 0.0
    gps_spoof_logged = False
    imu_inject_logged = False
    saarm_logged = False
    recovery_logged = False
    
    t = 0.0
    
    while True:
        with state.lock:
            connected = state.is_connected
            spoofing = state.spoofing_active
            imu_injection = state.imu_injection_active
            raw_ax = state.latest_imu["ax"]
            raw_ay = state.latest_imu["ay"]
            raw_az = state.latest_imu["az"]
            raw_gx = state.latest_imu["gx"]
            raw_gy = state.latest_imu["gy"]
            raw_gz = state.latest_imu["gz"]
            prev_ax = state.prev_imu["ax"]
            prev_ay = state.prev_imu["ay"]
            prev_az = state.prev_imu["az"]
        
        if not connected:
            time.sleep(dt)
            continue
        
        # ============================================================
        # CROSS-VALIDATION 1: IMU Anomaly Detection
        # ============================================================
        imu_anomaly = False
        imu_spike = False
        
        # Check for physically impossible values (> 80 m/s²)
        accel_magnitude = np.sqrt(raw_ax**2 + raw_ay**2 + raw_az**2)
        if accel_magnitude > 80.0:
            imu_anomaly = True
        
        # Check for sudden spikes (> 30 m/s² change between samples)
        accel_delta = np.sqrt((raw_ax - prev_ax)**2 + (raw_ay - prev_ay)**2 + (raw_az - prev_az)**2)
        if accel_delta > 30.0 and state.packets_received > 10:
            imu_spike = True
        
        # Check for frozen sensor (zero variance = sensor stuck/injected)
        gyro_magnitude = np.sqrt(raw_gx**2 + raw_gy**2 + raw_gz**2)
        
        # Simulated IMU injection attack
        if imu_injection:
            # Override IMU with malicious data
            inject_ax = raw_ax + np.random.normal(0, 5.0)  # Add large noise
            inject_ay = raw_ay + 15.0 * np.sin(t * 2)  # Oscillating false signal
            imu_anomaly = True
            raw_ax = inject_ax
            raw_ay = inject_ay
        
        # Low-pass filter IMU
        imu_filtered[0] = alpha * raw_ax + (1 - alpha) * imu_filtered[0]
        imu_filtered[1] = alpha * raw_ay + (1 - alpha) * imu_filtered[1]
        
        # ============================================================
        # PHYSICAL PHONE CORRIDOR & HEADING TRACKING
        # ============================================================
        # Gyro Z integration for physical heading (yaw in degrees)
        gz_eff = raw_gz if abs(raw_gz) > 0.03 else 0.0
        phone_heading = phone_heading * 0.992 + (gz_eff * 180.0 / np.pi) * dt
        
        # Lateral acceleration Ay integration (lateral drift in meters)
        ay_eff = raw_ay if abs(raw_ay) > 0.15 else 0.0
        phone_lat_vel = phone_lat_vel * 0.94 + ay_eff * dt
        phone_lat_pos = phone_lat_pos * 0.985 + phone_lat_vel * dt
        
        # Corridor Check against pre-authorized mission envelope (+-1.8m)
        is_breached = abs(phone_lat_pos) > state.corridor_half_width or (abs(phone_heading) > 18.0 and abs(phone_lat_pos) > 0.4)
        is_warning = abs(phone_lat_pos) > (state.corridor_half_width * 0.55) or abs(phone_heading) > 10.0
        
        with state.lock:
            state.imu_anomaly_detected = imu_anomaly or imu_spike
            if imu_spike:
                state.imu_spike_count += 1
            
            state.phone_heading = float(phone_heading)
            state.phone_lat_vel = float(phone_lat_vel)
            state.phone_lateral_pos = float(phone_lat_pos)
            state.corridor_breached = is_breached
            state.history_envelope_violation = is_breached
            state.corridor_status = "BREACHED" if is_breached else ("WARNING" if is_warning else "NOMINAL")
            
            if is_breached and not corridor_breached_logged:
                state.add_threat("WRONG_PATH", "critical",
                    f"Physical trajectory diverted! Heading: {phone_heading:+.1f}°, Offset: {phone_lat_pos:+.2f}m. Path violates authorized envelope!")
                corridor_breached_logged = True
            elif not is_breached and corridor_breached_logged:
                corridor_breached_logged = False
            
            # ============================================================
            # NAVIGATION: True Position + Hacked GPS (toward Hacker Target)
            # ============================================================
            state.true_pos_x += 1.0 * dt
            state.true_pos_y += 0.0 * dt
            
            if spoofing:
                target_y = state.hacker_target_y
                diff_y = target_y - current_spoof_offset[1]
                if abs(diff_y) > 0.1:
                    current_spoof_offset[1] += np.sign(diff_y) * spoof_drift_rate * dt
                else:
                    current_spoof_offset[1] = target_y
            else:
                current_spoof_offset *= 0.95
            
            noise_scale = 0.015
            state.hacked_gps_x = state.true_pos_x + current_spoof_offset[0] + np.random.normal(0, noise_scale)
            state.hacked_gps_y = state.true_pos_y + current_spoof_offset[1] + np.random.normal(0, noise_scale)
            
            # ============================================================
            # CROSS-VALIDATION 2: Environmental Sensor Check
            # ============================================================
            # Simulate barometric altitude (true altitude with noise)
            state.baro_altitude = 100.0 + np.random.normal(0, 0.3)
            # GPS altitude (affected by spoofing)
            if spoofing:
                state.gps_altitude = 100.0 + current_spoof_offset[1] * 0.3 + np.random.normal(0, 0.5)
            else:
                state.gps_altitude = 100.0 + np.random.normal(0, 0.5)
            
            baro_gps_diff = abs(state.baro_altitude - state.gps_altitude)
            state.baro_gps_mismatch = baro_gps_diff > 2.0
            
            # ============================================================
            # EKF PREDICT
            # ============================================================
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
            u = imu_filtered
            
            x_hat_minus = F @ x_hat + B @ u
            P_minus = F @ P @ F.T + Q
            
            # ============================================================
            # EKF UPDATE
            # ============================================================
            z = np.array([state.hacked_gps_x, state.hacked_gps_y])
            y_res = z - (H @ x_hat_minus)
            S = H @ P_minus @ H.T + R
            K = P_minus @ H.T @ np.linalg.inv(S)
            
            chi2 = float(y_res.T @ np.linalg.inv(S) @ y_res)
            state.chi_squared = chi2
            state.ekf_residual = float(np.linalg.norm(y_res))
            
            # ============================================================
            # CUSUM ANOMALY DETECTION
            # ============================================================
            if spoofing:
                cusum_drift += state.ekf_residual * dt
            else:
                cusum_drift = max(0, cusum_drift - 0.5 * dt)
            state.cusum = float(cusum_drift)
            
            # ============================================================
            # CROSS-VALIDATION 3: GPS-IMU Agreement
            # ============================================================
            state.gps_imu_agreement = chi2 < 5.991 and not state.baro_gps_mismatch
            
            # ============================================================
            # TRUST SCORES (Cross-Validation Result)
            # ============================================================
            # GPS Trust: degrades with high chi-squared, CUSUM, and baro mismatch
            gps_penalty = min(100, chi2 * 8 + state.cusum * 3 + (20 if state.baro_gps_mismatch else 0))
            target_gps_trust = max(0, 100 - gps_penalty)
            state.gps_trust += (target_gps_trust - state.gps_trust) * 0.1  # Smooth transition
            
            # IMU Trust: degrades with anomalies and spikes
            imu_penalty = 0
            if imu_anomaly:
                imu_penalty += 60
            if imu_spike:
                imu_penalty += 30
            if imu_injection:
                imu_penalty += 80
            target_imu_trust = max(0, 100 - imu_penalty)
            state.imu_trust += (target_imu_trust - state.imu_trust) * 0.1
            
            # Environmental Trust
            env_penalty = 50 if state.baro_gps_mismatch else 0
            target_env_trust = max(0, 100 - env_penalty)
            state.env_trust += (target_env_trust - state.env_trust) * 0.1
            
            # System Integrity: weighted average of all trust scores
            state.system_integrity = state.gps_trust * 0.4 + state.imu_trust * 0.35 + state.env_trust * 0.25
            
            # ============================================================
            # SAARM DECISION LOGIC (Multi-Attack)
            # ============================================================
            gps_attack = chi2 > 5.991 or state.cusum > cusum_threshold
            imu_attack = imu_anomaly and imu_injection
            
            if gps_attack and imu_attack:
                state.attack_detected = True
                state.attack_type = "multi_vector"
            elif gps_attack:
                state.attack_detected = True
                state.attack_type = "gps_spoof"
            elif imu_attack:
                state.attack_detected = True
                state.attack_type = "imu_injection"
            elif state.cusum < cusum_threshold * 0.5 and not spoofing and not imu_injection:
                state.attack_detected = False
                state.attack_type = "none"
            
            if not state.attack_detected:
                x_hat = x_hat_minus + K @ y_res
                P = (np.eye(4) - K @ H) @ P_minus
            else:
                x_hat = x_hat_minus  # Inertial only
                P = P_minus
            
            state.fused_pos_x = float(x_hat[0])
            state.fused_pos_y = float(x_hat[1])
            
            # ============================================================
            # THREAT LOGGING
            # ============================================================
            if spoofing and not gps_spoof_logged and chi2 > 3.0:
                state.add_threat("GPS_ANOMALY", "warning", f"GPS-IMU inconsistency detected (χ²={chi2:.1f})")
                gps_spoof_logged = True
            
            if gps_attack and not saarm_logged:
                state.add_threat("GPS_SPOOF", "critical", f"GPS spoofing confirmed! Chi²={chi2:.1f}, CUSUM={state.cusum:.1f}")
                state.add_threat("SAARM", "info", "SAARM activated — switched to inertial-only navigation")
                saarm_logged = True
            
            if state.baro_gps_mismatch and t - last_threat_log_time > 2.0:
                state.add_threat("ENV_MISMATCH", "warning", f"Barometer-GPS altitude mismatch: Δ={baro_gps_diff:.1f}m")
                last_threat_log_time = t
            
            if imu_injection and not imu_inject_logged:
                state.add_threat("IMU_INJECTION", "critical", "IMU injection attack detected — false accelerometer readings")
                imu_inject_logged = True
            
            if not spoofing and not imu_injection and state.cusum < 1.0 and (gps_spoof_logged or saarm_logged) and not recovery_logged:
                state.add_threat("RECOVERY", "info", "All sensors nominal — system recovered")
                gps_spoof_logged = False
                saarm_logged = False
                imu_inject_logged = False
                recovery_logged = True
            
            if spoofing or imu_injection:
                recovery_logged = False
            
            # ============================================================
            # VELOCITY INTEGRATION
            # ============================================================
            # Subtract gravity from Z, integrate to get velocity
            gravity_compensated = np.array([raw_ax, raw_ay, raw_az - 9.81])
            vel = vel * vel_damping + gravity_compensated * dt
            state.velocity["vx"] = float(vel[0])
            state.velocity["vy"] = float(vel[1])
            state.velocity["vz"] = float(vel[2])
            
            # ============================================================
            # FFT / FREQUENCY SPECTRUM (Octave Analysis)
            # ============================================================
            state.accel_buffer_z.append(raw_az)
            if len(state.accel_buffer_z) > 128:
                state.accel_buffer_z = state.accel_buffer_z[-128:]
            
            fft_counter += 1
            if fft_counter % 4 == 0 and len(state.accel_buffer_z) >= 64:
                signal = np.array(state.accel_buffer_z[-64:])
                signal = signal - np.mean(signal)  # Remove DC offset
                window = np.hanning(len(signal))
                fft_result = np.fft.rfft(signal * window)
                mags = (np.abs(fft_result) / len(signal)).tolist()
                sample_rate = 1.0 / dt  # 20 Hz
                freqs = np.fft.rfftfreq(len(signal), d=dt).tolist()
                state.fft_magnitudes = [round(m, 5) for m in mags[:16]]
                state.fft_freqs = [round(f, 1) for f in freqs[:16]]
            
            t += dt
            state.sim_time = t
        
        time.sleep(dt)

# ============================================================================
# WebSocket Broadcaster
# ============================================================================
def broadcast_loop():
    while True:
        with state.lock:
            payload = {
                'is_connected': state.is_connected,
                'spoofing_active': state.spoofing_active,
                'imu_injection_active': state.imu_injection_active,
                'attack_detected': state.attack_detected,
                'attack_type': state.attack_type,
                'packets': state.packets_received,
                'sim_time': round(state.sim_time, 2),
                'imu': {
                    'ax': round(state.latest_imu['ax'], 4),
                    'ay': round(state.latest_imu['ay'], 4),
                    'az': round(state.latest_imu['az'], 4),
                    'gx': round(state.latest_imu['gx'], 4),
                    'gy': round(state.latest_imu['gy'], 4),
                    'gz': round(state.latest_imu['gz'], 4),
                },
                'true_x': round(state.true_pos_x, 4),
                'true_y': round(state.true_pos_y, 4),
                'hacked_x': round(state.hacked_gps_x, 4),
                'hacked_y': round(state.hacked_gps_y, 4),
                'fused_x': round(state.fused_pos_x, 4),
                'fused_y': round(state.fused_pos_y, 4),
                'ekf_residual': round(state.ekf_residual, 4),
                'chi_squared': round(state.chi_squared, 4),
                'cusum': round(state.cusum, 4),
                # Cross-Validation & Trust
                'gps_trust': round(state.gps_trust, 1),
                'imu_trust': round(state.imu_trust, 1),
                'env_trust': round(state.env_trust, 1),
                'system_integrity': round(state.system_integrity, 1),
                'gps_imu_agreement': state.gps_imu_agreement,
                'imu_anomaly': state.imu_anomaly_detected,
                'baro_gps_mismatch': state.baro_gps_mismatch,
                'baro_alt': round(state.baro_altitude, 1),
                'gps_alt': round(state.gps_altitude, 1),
                # Velocity
                'velocity': {
                    'vx': round(state.velocity['vx'], 4),
                    'vy': round(state.velocity['vy'], 4),
                    'vz': round(state.velocity['vz'], 4),
                },
                # FFT Spectrum
                'fft_magnitudes': state.fft_magnitudes,
                'fft_freqs': state.fft_freqs,
                # Flight Corridor & Physical Hijack
                'corridor': {
                    'half_width': state.corridor_half_width,
                    'heading': round(state.phone_heading, 1),
                    'lateral_pos': round(state.phone_lateral_pos, 2),
                    'lateral_vel': round(state.phone_lat_vel, 2),
                    'breached': state.corridor_breached,
                    'violation': state.history_envelope_violation,
                    'status': state.corridor_status,
                    'target_y': state.hacker_target_y,
                    'target_label': state.hacker_target_label,
                },
                # Threat Log (last 15 entries)
                'threat_log': state.threat_log[-15:],
            }
        socketio.emit('state_update', payload)
        time.sleep(0.1)

# ============================================================================
# Routes & Socket Events
# ============================================================================
@app.route('/')
def index():
    return send_from_directory('.', 'dashboard.html')

@socketio.on('connect')
def handle_connect():
    print("[WS] Client connected")

@socketio.on('toggle_spoofing')
def handle_toggle_spoofing():
    with state.lock:
        state.spoofing_active = not state.spoofing_active
        if state.spoofing_active:
            state.add_threat("ATTACK_SIM", "warning", f"GPS spoofing attack active — dragging navigation toward {state.hacker_target_label}")
        else:
            state.add_threat("ATTACK_SIM", "info", "GPS spoofing attack stopped")
    socketio.emit('spoofing_status', {'active': state.spoofing_active})

@socketio.on('toggle_imu_injection')
def handle_toggle_imu():
    with state.lock:
        state.imu_injection_active = not state.imu_injection_active
        if state.imu_injection_active:
            state.add_threat("ATTACK_SIM", "warning", "IMU injection attack simulation started")
        else:
            state.add_threat("ATTACK_SIM", "info", "IMU injection attack simulation stopped")
    socketio.emit('imu_injection_status', {'active': state.imu_injection_active})

@socketio.on('set_hacker_target')
def handle_set_hacker_target(data):
    with state.lock:
        state.hacker_target_y = float(data.get('y', 15.0))
        state.hacker_target_label = str(data.get('label', f"Custom Spoof ({state.hacker_target_y:+.1f}m)"))
        state.add_threat("HACKER_TARGET", "warning", f"Spoof coordinates targeted at: {state.hacker_target_label}")
    socketio.emit('hacker_target_updated', {
        'target_y': state.hacker_target_y,
        'target_label': state.hacker_target_label
    })

@socketio.on('reset_corridor')
def handle_reset_corridor():
    with state.lock:
        state.phone_heading = 0.0
        state.phone_lateral_pos = 0.0
        state.phone_lat_vel = 0.0
        state.corridor_breached = False
        state.history_envelope_violation = False
        state.corridor_status = "NOMINAL"
        state.add_threat("CORRIDOR", "info", "Corridor tracker calibrated to current orientation")
    socketio.emit('corridor_reset', {})

@socketio.on('reset_simulation')
def handle_reset():
    with state.lock:
        state.spoofing_active = False
        state.imu_injection_active = False
        state.attack_detected = False
        state.attack_type = "none"
        state.true_pos_x = 0.0
        state.true_pos_y = 0.0
        state.hacked_gps_x = 0.0
        state.hacked_gps_y = 0.0
        state.fused_pos_x = 0.0
        state.fused_pos_y = 0.0
        state.ekf_residual = 0.0
        state.chi_squared = 0.0
        state.cusum = 0.0
        state.gps_trust = 100.0
        state.imu_trust = 100.0
        state.env_trust = 100.0
        state.system_integrity = 100.0
        state.sim_time = 0.0
        state.threat_log = []
        state.phone_heading = 0.0
        state.phone_lateral_pos = 0.0
        state.phone_lat_vel = 0.0
        state.corridor_breached = False
        state.history_envelope_violation = False
        state.corridor_status = "NOMINAL"
        state.add_threat("SYSTEM", "info", "Simulation reset — all sensors nominal")
    socketio.emit('simulation_reset', {})

# ============================================================================
# Main
# ============================================================================
if __name__ == '__main__':
    print("=" * 60)
    print("  SENSORSENTRY — Spoofing Detection & Sensor Fusion")
    print("  PS 18 | Hackathon Demo Server")
    print("=" * 60)
    
    threading.Thread(target=udp_listener, daemon=True).start()
    threading.Thread(target=math_engine, daemon=True).start()
    threading.Thread(target=broadcast_loop, daemon=True).start()
    
    print("[SERVER] Dashboard: http://localhost:8080")
    print("[SERVER] UDP listener on port 5000")
    print("=" * 60)
    
    socketio.run(app, host='0.0.0.0', port=8080, allow_unsafe_werkzeug=True)
