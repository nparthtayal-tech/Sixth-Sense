# SensorSentry — Production Autonomous Drone Avionics Package

Production-grade companion avionics system for commercial autonomous delivery drones (Raspberry Pi CM4, NVIDIA Jetson Orin/Nano, or embedded Linux SBC). Directly interfaces with **Pixhawk / Cube Orange / PX4 / ArduPilot** flight controllers over **MAVLink v2** to provide real-time GPS spoofing defense, flight corridor protection, and autonomous failsafe overrides.

---

## 1. System Architecture

```
  ┌────────────────────────────────────────────────────────┐
  │         PIXHAWK / CUBE AUTOPILOT (PX4 / ARDUPILOT)     │
  │  - Hard Real-Time Motor Mixing & Attitude Loops (1kHz) │
  │  - Failsafe Actuation Policy (LOITER / RTL / LAND)     │
  └───────────────────────────┬────────────────────────────┘
                              │ MAVLink v2 (UART / Ethernet / UDP)
                              ▼
  ┌────────────────────────────────────────────────────────┐
  │       COMPANION COMPUTER (RPi CM4 / JETSON ORIN)       │
  │  SENSORSENTRY HARDWARE DEPLOYMENT                      │
  │  - 11-State Multi-Sensor EKF (GNSS + IMU + Baro)       │
  │  - Dual Fast/Slow Anomaly Detection (Chi-Square+CUSUM) │
  │  - SAARM Sensor Quarantine & Dead-Reckoning            │
  │  - Mission Flight Corridor & Geofence Verification     │
  │  - Autonomous Autopilot Actuator Overrides (MAV_CMD)   │
  └────────────────────────────────────────────────────────┘
```

---

## 2. Quick Setup

### Step 1: Install Dependencies
Run on your companion computer:
```bash
pip install -r requirements.txt
```

### Step 2: Configure Drone Parameters
Edit [`config.json`](file:///c:/Users/BSES/OneDrive/Desktop/Resonance/hardware_deploy/config.json) to set your connection string, flight corridor tolerances, and failsafe modes.

---

## 3. Execution Modes

### Mode 1: Production MAVLink Flight Mode
Connects directly to the Pixhawk telemetry port (TELEM1 / TELEM2):
```bash
# Via UART Serial (e.g. Raspberry Pi UART to Pixhawk TELEM2 at 921600 baud):
python main.py --flight-link /dev/ttyAMA0:921600 --enable-flight-actions

# Via UDP (e.g. Ethernet companion port or SITL simulator on port 14550):
python main.py --flight-link udpin:0.0.0.0:14550 --enable-flight-actions
```

### Mode 2: Software-In-The-Loop (SITL) Validation
Run our built-in Pixhawk simulator to test telemetry ingestion, spoofing injection, and autonomous failsafe triggers on your bench without a physical drone:
```bash
python mock_mavlink_sitl.py
```

### Mode 3: Deterministic Hardware Self-Test Benchmark
Validates EKF convergence, Chi-Square innovation gating, CUSUM drift detection, and sensor quarantine offline:
```bash
python main.py --mode test
```

---

## 4. Autonomous Systemd Daemon (Auto-Start on Boot)

To ensure SensorSentry launches automatically as soon as the drone's LiPo battery is plugged in:
```bash
sudo cp sensorsentry.service /etc/systemd/system/
sudo systemctl daemon-reload
sudo systemctl enable sensorsentry.service
sudo systemctl start sensorsentry.service
```

---

## 5. Sensory Inputs Specification

For complete details on coordinate frames, measurement vectors, and MAVLink message IDs, see:
📄 [`sensor_inputs_spec.txt`](file:///c:/Users/BSES/OneDrive/Desktop/Resonance/hardware_deploy/sensor_inputs_spec.txt)
