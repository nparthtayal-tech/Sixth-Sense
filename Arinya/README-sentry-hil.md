SensorSentry: Two-Phone Hardware-in-the-Loop (HIL) Demonstration Setup
This guide details how to construct and execute the live, physical SensorSentry (PS 18) demonstration. It enables your development environment (or code generation assistant) to build a multi-device system where a smartphone acts as a live navigation unit and a second device acts as a malicious GPS spoofer.

📐 System Architecture
The demonstration is designed as a Hardware-in-the-Loop (HIL) Cyber-Physical Simulation. It runs a real-time software-defined exploit over real physical sensor streams, perfectly mimicking the mathematical signature of an over-the-air radio frequency (RF) attack.

+------------------------------------+
|         Phone 1: TARGET            |
|  Streams Accelerometer & Gyro      |
|  dynamics via UDP (Phyphox/Socket) |
+------------------------------------+
                  |
                  | [Real physical motion via Wi-Fi]
                  v
+------------------------------------+              +------------------------------------+
|    Laptop: SENSORSENTRY ENGINE     |<-------------|        Phone 2: INTRUDER           |
|  - Extended Kalman Filter (EKF)    |              |  Accesses spoofer panel over HTTP  |
|  - Chi-Squared (Outlier Gating)    |              |  Triggers "Slow-Drift Spoof"       |
|  - CUSUM Anomaly Detector          |              +------------------------------------+
|  - Real-Time Streamlit Dashboard   |
+------------------------------------+
                  |
                  v [Real-Time Visuals on Projector]
   [Interactive 2D Live Route Map & Anomaly Charts]
🛠️ Step-by-Step Hardware & App Configurations
1. Configure Phone 1 (The Target Vehicle)
You will use a smartphone's internal IMU (accelerometer/gyroscope) to capture real-world physical dynamics.

App to download: Phyphox (Physical Phone Experiments) or Sensor Logger (both free on Android & iOS).
Phyphox Setup:
Open the app and select the "Acceleration without g" or "IMU" experiment.
Tap the top-right menu icon (three dots) and select "Allow remote access".
Phyphox will generate a local IP address and port (e.g., http://192.168.1.15:8080). Note this down.
(Alternatively, configure the "Network Stream" option to stream raw data as UDP packets to your laptop's IP address on Port 5000).
Sensor Logger Setup:
Toggle Accelerometer and Gyroscope to ON.
Go to Settings -> Pushing (JSON via WebSockets).
Enter your laptop’s local network IP address (e.g., ws://192.168.1.10:8001).
2. Configure Phone 2 (The Intruder)
This phone serves as the remote spoofing transmitter.

Setup: Phone 2 simply opens a web browser and navigates to the spoofer control page hosted on your laptop's local IP address (e.g., http://192.168.1.10:8501/spoofer or a dedicated Streamlit sub-page).
Interface: Features a prominent, styled retro arcade button: "ACTIVATE SNEAKY SPOOFING".
⚙️ The Mathematical Defense Subsystems
Your running code on the laptop must implement these core modules to successfully process the incoming streams:

The Extended Kalman Filter (EKF): Fuses the high-frequency accelerometer ($100\text{ Hz}$) with absolute position inputs ($1\text{ Hz}$) using a 2D state kinematics model: $$x_k = \begin{bmatrix} p_x & p_y & v_x & v_y \end{bmatrix}^T$$
Chi-Squared ($\chi^2$) Outlier Gating: Calculates the Mahalanobis distance of incoming GPS measurements relative to predicted EKF covariance: $$d_M^2 = (y_k - H \hat{x}{k|k-1})^T S_k^{-1} (y_k - H \hat{x}{k|k-1})$$ If $d_M^2$ exceeds the 95% confidence threshold ($\chi^2 > 5.991$ for 2 degrees of freedom), the measurement is flagged as a sudden collision or physical failure and blocked.
CUSUM (Cumulative Sum) Detector: Tracks systematic residuals over time to expose stealthy, slow-drift spoofing: $$S_k = \max(0, S_{k-1} + z_k - \omega)$$ Where $z_k$ is the normalized EKF innovation and $\omega$ is a bias tolerance. If $S_k$ crosses a defined threshold, it exposes a gradual spoofing coordinate deviation.
SAARM (Sensor-Agnostic All-source Residual Monitoring): Runs two parallel Kalman filters:
Filter A (Fused): Processes GPS + IMU inputs.
Filter B (Dead-Reckoning): Excludes GPS, processing only physical IMU integration.
The Switch: The moment CUSUM triggers an alert, SAARM isolates the faulty GPS sensor feed and seamlessly switches active navigation output to the stable dead-reckoning sub-filter, preserving smooth, uninterrupted route tracking.
🚀 Execution & Demo Verification
Launch Server: Execute your web dashboard script on your laptop:
python sentry_live_demo.py
Connect Devices: Ensure both phones and your laptop are on the same local Wi-Fi network. Point Phone 1's sensor stream to the laptop's receiver port.
Run Scenario:
Move Phone 1 along a straight, physical line on the table.
Let a judge tap the "Spoof" button on Phone 2.
Watch the laptop dashboard: the raw GPS coordinates will drift off-path, the CUSUM chart will spike, the alarm will flash, and the estimated position will remain securely pinned to your physical hand movement.SensorSentry: Two-Phone Hardware-in-the-Loop (HIL) Demonstration Setup
This guide details how to construct and execute the live, physical SensorSentry (PS 18) demonstration. It enables your development environment (or code generation assistant) to build a multi-device system where a smartphone acts as a live navigation unit and a second device acts as a malicious GPS spoofer.

📐 System Architecture
The demonstration is designed as a Hardware-in-the-Loop (HIL) Cyber-Physical Simulation. It runs a real-time software-defined exploit over real physical sensor streams, perfectly mimicking the mathematical signature of an over-the-air radio frequency (RF) attack.

+------------------------------------+
|         Phone 1: TARGET            |
|  Streams Accelerometer & Gyro      |
|  dynamics via UDP (Phyphox/Socket) |
+------------------------------------+
                  |
                  | [Real physical motion via Wi-Fi]
                  v
+------------------------------------+              +------------------------------------+
|    Laptop: SENSORSENTRY ENGINE     |<-------------|        Phone 2: INTRUDER           |
|  - Extended Kalman Filter (EKF)    |              |  Accesses spoofer panel over HTTP  |
|  - Chi-Squared (Outlier Gating)    |              |  Triggers "Slow-Drift Spoof"       |
|  - CUSUM Anomaly Detector          |              +------------------------------------+
|  - Real-Time Streamlit Dashboard   |
+------------------------------------+
                  |
                  v [Real-Time Visuals on Projector]
   [Interactive 2D Live Route Map & Anomaly Charts]
🛠️ Step-by-Step Hardware & App Configurations
1. Configure Phone 1 (The Target Vehicle)
You will use a smartphone's internal IMU (accelerometer/gyroscope) to capture real-world physical dynamics.

App to download: Phyphox (Physical Phone Experiments) or Sensor Logger (both free on Android & iOS).
Phyphox Setup:
Open the app and select the "Acceleration without g" or "IMU" experiment.
Tap the top-right menu icon (three dots) and select "Allow remote access".
Phyphox will generate a local IP address and port (e.g., http://192.168.1.15:8080). Note this down.
(Alternatively, configure the "Network Stream" option to stream raw data as UDP packets to your laptop's IP address on Port 5000).
Sensor Logger Setup:
Toggle Accelerometer and Gyroscope to ON.
Go to Settings -> Pushing (JSON via WebSockets).
Enter your laptop’s local network IP address (e.g., ws://192.168.1.10:8001).
2. Configure Phone 2 (The Intruder)
This phone serves as the remote spoofing transmitter.

Setup: Phone 2 simply opens a web browser and navigates to the spoofer control page hosted on your laptop's local IP address (e.g., http://192.168.1.10:8501/spoofer or a dedicated Streamlit sub-page).
Interface: Features a prominent, styled retro arcade button: "ACTIVATE SNEAKY SPOOFING".
⚙️ The Mathematical Defense Subsystems
Your running code on the laptop must implement these core modules to successfully process the incoming streams:

The Extended Kalman Filter (EKF): Fuses the high-frequency accelerometer ($100\text{ Hz}$) with absolute position inputs ($1\text{ Hz}$) using a 2D state kinematics model: $$x_k = \begin{bmatrix} p_x & p_y & v_x & v_y \end{bmatrix}^T$$
Chi-Squared ($\chi^2$) Outlier Gating: Calculates the Mahalanobis distance of incoming GPS measurements relative to predicted EKF covariance: $$d_M^2 = (y_k - H \hat{x}{k|k-1})^T S_k^{-1} (y_k - H \hat{x}{k|k-1})$$ If $d_M^2$ exceeds the 95% confidence threshold ($\chi^2 > 5.991$ for 2 degrees of freedom), the measurement is flagged as a sudden collision or physical failure and blocked.
CUSUM (Cumulative Sum) Detector: Tracks systematic residuals over time to expose stealthy, slow-drift spoofing: $$S_k = \max(0, S_{k-1} + z_k - \omega)$$ Where $z_k$ is the normalized EKF innovation and $\omega$ is a bias tolerance. If $S_k$ crosses a defined threshold, it exposes a gradual spoofing coordinate deviation.
SAARM (Sensor-Agnostic All-source Residual Monitoring): Runs two parallel Kalman filters:
Filter A (Fused): Processes GPS + IMU inputs.
Filter B (Dead-Reckoning): Excludes GPS, processing only physical IMU integration.
The Switch: The moment CUSUM triggers an alert, SAARM isolates the faulty GPS sensor feed and seamlessly switches active navigation output to the stable dead-reckoning sub-filter, preserving smooth, uninterrupted route tracking.
🚀 Execution & Demo Verification
Launch Server: Execute your web dashboard script on your laptop:
python sentry_live_demo.py
Connect Devices: Ensure both phones and your laptop are on the same local Wi-Fi network. Point Phone 1's sensor stream to the laptop's receiver port.
Run Scenario:
Move Phone 1 along a straight, physical line on the table.
Let a judge tap the "Spoof" button on Phone 2.
Watch the laptop dashboard: the raw GPS coordinates will drift off-path, the CUSUM chart will spike, the alarm will flash, and the estimated position will remain securely pinned to your physical hand movement.