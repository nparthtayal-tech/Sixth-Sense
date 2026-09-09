# SensorSentry — Mobile Cobot Hardware Deployment

This package is the ground-robot counterpart to `hardware_deploy`. It runs on a
companion computer or the cobot's ROS 2 computer and protects a **mobile
cobot** (AMR/AGV) from unauthorised navigation targets, localization spoofing,
geofence exits, stale telemetry, e-stop events, and unsafe LiDAR clearances.

It is designed for ROS 2 + Nav2 systems such as TurtleBot-class platforms,
MiR/Fetch-style integrations with an adapter, or custom differential-drive
cobots. It does **not** command motors directly. When explicitly enabled, it
cancels active Nav2 goals and sends only a zero-velocity command on `/cmd_vel`.
The robot's safety PLC, motor controller, bumper chain, and physical E-stop
remain authoritative.

## Architecture

```
 LiDAR / encoders / IMU / localization / E-stop
                    │ authenticated telemetry
                    ▼
 ┌────────────────────────────────────────────────────────────┐
 │  SensorSentry cobot companion                               │
 │  - HMAC + timestamp + sequence replay validation            │
 │  - map boundary / route-corridor validation                 │
 │  - localization jump, stale sensor, speed, obstacle checks  │
 │  - signed navigation-goal validation                         │
 └─────────────────────┬──────────────────────────────────────┘
                       │ only with --enable-robot-actions
                       ▼
       ROS 2 Nav2 cancel goal + repeated zero `/cmd_vel`
```

## Install

Install the ROS 2 distribution on the cobot first (Humble, Jazzy, or the
vendor-supported release), then source its setup script:

```bash
source /opt/ros/humble/setup.bash
python -m pip install -r requirements.txt
```

`rclpy`, `geometry_msgs`, `nav2_msgs`, and `action_msgs` come from the ROS 2
distribution; do not install them from PyPI.

## Configure

Edit [config.json](config.json) before deployment:

- `map_bounds_m`: `[min_x, min_y, max_x, max_y]` for the approved map frame.
- `corridor_half_width_m`: maximum deviation from the active route.
- speed, obstacle and localization-jump limits to match the robot safety case.
- `robot_id` and `map_id` to match the telemetry producer and mission planner.

For live input, set a unique 32-byte HMAC key encoded as base64:

```bash
export SENSORSENTRY_COBOT_HMAC_KEY="<base64-encoded-32-byte-key>"
```

## Run modes

### 1. Deterministic safety self-test

No robot or ROS installation is needed:

```bash
python main.py --mode test
python -m unittest discover -s tests -v
```

### 2. Replay mock authenticated telemetry

In one terminal:

```bash
export SENSORSENTRY_COBOT_HMAC_KEY="MDEyMzQ1Njc4OWFiY2RlZjAxMjM0NTY3ODlhYmNkZWY="
python main.py --mode udp --udp-port 5010
```

In a second terminal:

```bash
python mock_cobot_sitl.py --udp-port 5010
```

### 3. Production ROS 2 safety stop

The default controller is advisory-only. This command is the deliberate opt-in
for ROS 2 navigation cancellation and zero `cmd_vel` publication:

```bash
python main.py --mode udp --udp-port 5010 \
  --controller ros2 --enable-robot-actions --ros-namespace /cobot_01
```

Before enabling this mode, verify the namespace, `/cmd_vel` mux priority,
Nav2 action name, physical E-stop chain, speed limits, and recovery procedure
on a blocked-up bench test. This process never resumes a robot automatically;
clearance must be supplied by the supervisory system.

### 4. Start advisory monitoring on boot

The supplied service intentionally starts in advisory mode. Create
`/etc/sensorsentry/cobot.env` with `SENSORSENTRY_COBOT_HMAC_KEY=...`, then:

```bash
sudo install -m 0644 cobot_sentry.service /etc/systemd/system/
sudo systemctl daemon-reload
sudo systemctl enable --now cobot_sentry.service
```

Switching the service to `--controller ros2 --enable-robot-actions` is a
deployment decision requiring the same bench validation as the production ROS
2 mode above.

## Authenticated telemetry contract

Every packet is compact JSON with `schema_version`, `robot_id`, increasing
`sequence`, Unix `timestamp`, `sensor`, and a lowercase-hex HMAC-SHA256
`signature`. The signature covers canonical JSON excluding `signature`.

Supported sensors are:

- `ODOMETRY`: `frame: "map"`, `position_m: [x, y]`, `yaw_rad`,
  `linear_velocity_m_s`, `angular_velocity_rad_s`.
- `LOCALIZATION`: the same map-frame pose fields, from AMCL/SLAM/VIO.
- `IMU`: `frame: "base_link"`, `yaw_rate_rad_s`, `forward_accel_m_s2`.
- `LIDAR`: `min_range_m` and optional `frame: "base_link"`.
- `BATTERY`: `charge_percent`, `voltage_v`.
- `ESTOP`: `pressed: true|false`.

Navigation commands use `navigation_command` packets validated by
`waypoint_security.command_validator`; targets must be signed, current, inside
the configured map, and within the active corridor when one is loaded.

See [sensor_inputs_spec.txt](sensor_inputs_spec.txt) for exact field rules.

## Files

- `main.py` — secure UDP/stdin runner and hardware self-test.
- `cobot_control.py` — advisory controller plus deliberately armed ROS 2 stop.
- `ros2_bridge.py` — optional ROS 2 telemetry publisher adapter.
- `sensor_sentry.py` — monitor orchestration and containment policy.
- `telemetry.py` — strict telemetry parsing, HMAC and replay protection.
- `waypoint_security/` — map boundary and goal validation.
- `response/` — latching safe-stop and structured fleet alerts.
- `hal/` — a hardware-message boundary and a safe simulated bus.

## Safety boundary

This is companion software, not a certified functional-safety controller. Keep
the safety-rated E-stop, safety scanner, PLC, bumper circuit, and motor-drive
limits independent of this process. Validate with the robot vendor and the
site's safety owner before operating near people.
