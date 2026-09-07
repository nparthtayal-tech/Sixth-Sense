# SensorSentry

SensorSentry is a dual-attack GPS defense system designed to protect autonomous vehicles and robots against two primary attack vectors:
1. **Scenario A: GPS Spoofing ("Coordinates Lie" Attack)** - Broadcasting fake satellite radio signals to steer the vehicle off-course.
2. **Scenario B: Waypoint Hijack ("Destination Code Tampering")** - Hacking the mission-planner software to change the target destination coordinates.

## Features

- **Multi-Sensor EKF Fusion:** 11-state Extended Kalman Filter fusing 8 heterogeneous sensor modalities (GNSS, IMU, Wheel Encoders, Ultrasonic, Cameras, LiDAR, 5G, Atomic Clock).
- **Fast & Slow Decision Monitors:** 
  - **Chi-Square Gate:** Instantaneous single-frame outlier detection for violent measurement spikes.
  - **CUSUM Monitor:** Sequential Cumulative Sum control chart for detecting persistent sensor drift.
- **SAARM (Sensor Autonomous Adaptive Reconfiguration Management):** Parallel filter bank that isolates and quarantines compromised sensors, enabling fallback to inertial dead-reckoning.
- **Waypoint Security Module:**
  - **Command Validator:** Cryptographic HMAC signing, sequence numbering, and rate-limiting.
  - **Geofence Validator:** Spatial bounds checking against authorized operational zones.
  - **Environment Matcher:** Cross-validates the physical world against the expected route using onboard sensors.
- **Response Layer:** Safe Stop emergency protocol and Fleet Alert broadcasting.
- **Hardware Abstraction Layer (HAL):** Unified interfaces for CAN, UART, SPI, I2C, and ROS2, making the system deployable to cobots, cars, and embedded platforms.

## Requirements & Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/nparthtayal-tech/Sixth-Sense.git
   cd Sixth-Sense
   ```

2. Create a virtual environment (optional but recommended):
   ```bash
   python -m venv .venv
   source .venv/bin/activate  # On Windows use: .venv\Scripts\activate
   ```

3. Install the dependencies:
   ```bash
   pip install -r requirements.txt
   ```
   *Note: `filterpy` is included locally in the `filterpy-master` directory, so you don't need to install it from pip.*

### Hardware-Specific Dependencies
- **CAN Bus:** Requires `python-can` and compatible hardware (e.g., SocketCAN on Linux).
- **UART/Serial:** Requires `pyserial`.
- **SPI:** Requires `spidev` (Linux/Raspberry Pi only).
- **ROS2:** Requires a valid ROS2 installation (e.g., Humble or Iron) with `rclpy` and `std_msgs`.

## Running the Demonstrations

### 1. EKF Fusion & Monitor Demo
Demonstrates the 8-sensor EKF fusion with Chi-Square and CUSUM monitors during a simulated 3D trajectory with a GPS outage:
```bash
python demo_fusion.py
```
This script will output performance metrics and generate the following plots:
- `fusion_trajectory_3d.png`
- `sensor_fusion_errors.png`
- `sensor_contribution_analysis.png`

### 2. Dual-Attack Demo
*(Ensure you have implemented `demo_dual_attack.py` and `sensor_sentry.py` based on the project roadmap to run this specific demo)*
Simulates both GPS Spoofing and Waypoint Hijacking attacks, showing SensorSentry's detection, isolation, and safe stop protocols in action:
```bash
python demo_dual_attack.py
```

### 3. Judge-ready spoofing recovery animation

Runs a complete automatic flight story for a hackathon presentation: the drone
leaves a marked start point, follows its authorised destination, is diverted by
a gradual spoofed GNSS signal, detects the spoof using the Chi-Square NIS gate
and CUSUM drift monitor, quarantines GNSS, reconnects to the command server,
accepts an HMAC-authenticated rejoin route, and returns to the true destination.

```bash
python demo_animated_sentry.py
```

The replay starts automatically. Press `Space` to pause/resume or `R` to replay.
For a shareable final-state image or animated export:

```bash
python demo_animated_sentry.py --snapshot judge_demo_final.png
python demo_animated_sentry.py --save judge_demo.gif --fps 12
```

To validate the full scenario without opening a window:

```bash
python demo_animated_sentry.py --no-show
```

## System Architecture

The project is structured into several modular packages:
- `ekf_fusion/`: Core EKF engine, sensor observation models, and simulation generator.
- `decision_making/`: Chi-Square and CUSUM health monitors.
- `saarm/`: Parallel filter bank and dead-reckoning navigator.
- `waypoint_security/`: Command validation, geofencing, and environment matching.
- `response/`: Safe stop controller and fleet alert broadcaster.
- `hal/`: Hardware bus adapters and sensor interface stubs.
