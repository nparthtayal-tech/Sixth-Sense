import time
import numpy as np
import matplotlib.pyplot as plt

from sensor_sentry import SensorSentry
from hal import SimulatedBus, SensorReading
from ekf_fusion.simulation_generator import TrajectorySimulator
from ekf_fusion.sensor_models import SensorType

def simulate_drone_hijack():
    print("==================================================")
    print("== SENSORSENTRY DRONE HIJACK SIMULATION ==")
    print("==================================================")
    print("Scenario: A delivery drone is flying to its destination.")
    print("An attacker uses a GPS Spoofer to drag the drone off course.")
    print("SensorSentry will detect the physical mismatch, switch to")
    print("dead-reckoning, and alert the Fleet Command Center.")
    print("--------------------------------------------------\n")

    # 1. Initialize hardware bus (Simulated)
    bus = SimulatedBus(verbose=False)
    
    # 2. Initialize SensorSentry Orchestrator
    sentry = SensorSentry("DRONE-ALPHA-01", bus)
    sentry.cusum.alarm_threshold = 10.0
    sentry.cusum.slack = 1.0
    sentry.saarm.isolation_threshold = 1.5
    
    # 3. Initialize Trajectory Simulator
    sim = TrajectorySimulator(duration=40.0, dt_sim=0.005)
    
    # Generate asynchronous sensor streams with GPS spoofing starting at t=15.0s
    events = sim.generate_sensor_events(enable_gps_spoofing=True, spoofing_start=15.0)
    
    # Set starting position
    sentry.initialize_state(sim.ground_truth[0].p, sim.ground_truth[0].psi)
    
    print("[0.0s] Drone took off. Nominal flight path.")
    
    attack_detected = False
    alert_sent = False
    
    # Arrays for plotting
    x_true = []
    y_true = []
    x_gps = []
    y_gps = []
    x_ekf = []
    y_ekf = []
    
    for ev in events:
        if sentry.safe_stop.is_stopped:
            break
            
        reading = SensorReading(
            timestamp=ev.t,
            data=ev.z,
            sensor_type=ev.sensor_type.name,
            valid=True
        )
        sentry.process_sensor_reading(reading, ev.R, *ev.args)
        
        if ev.sensor_type == SensorType.GNSS:
            gt_pt = sim._interpolate_gt(ev.t)
            x_true.append(gt_pt.p[0])
            y_true.append(gt_pt.p[1])
            x_gps.append(ev.z[0])
            y_gps.append(ev.z[1])
            x_ekf.append(sentry.ekf.x[0])
            y_ekf.append(sentry.ekf.x[1])
            
            # Print Narrative
            if ev.t >= 15.0 and ev.t < 15.2 and not attack_detected:
                print(f"\n[WARNING {ev.t:.1f}s] ATTACKER INITIATES GPS SPOOFING.")
                print(f"[{ev.t:.1f}s] Attacker is broadcasting fake satellite signals.")
                print(f"[{ev.t:.1f}s] Drone's autopilot is unknowingly steering off-course into the trap...")
                
            # Check system state
            if sentry.saarm.dr_navigator.active and not attack_detected:
                attack_detected = True
                print(f"\n[SHIELD {ev.t:.1f}s] SENSORSENTRY TRIGGERED!")
                print(f"[{ev.t:.1f}s] CUSUM Monitor caught the mathematical mismatch between IMU and GPS.")
                print(f"[{ev.t:.1f}s] SAARM isolated the compromised GPS and switched to Dead-Reckoning!")
                
            if attack_detected and not alert_sent:
                # Look for Fleet Alerts on the hardware bus
                alerts = bus.get_messages_by_channel('ALERT')
                if alerts:
                    alert_sent = True
                    print(f"[{ev.t:.1f}s] BROADCASTING ALERT TO SERVER...")
                    print(f"[{ev.t:.1f}s] SERVER RECEIVED: 'ATTACK_GPS_SPOOFING detected on DRONE-ALPHA-01'")
                    print(f"[{ev.t:.1f}s] SERVER RECEIVED: 'Drone is hovering in Safe Stop at Last Known Good Position.'")
                    print(f"[{ev.t:.1f}s] Drone is now completely ignoring the hacker's fake GPS.\n")
                    
    print("==================================================")
    print("Simulation Complete. Generating visual report...")
    print("==================================================")
    
    # Plotting
    plt.figure(figsize=(10, 6))
    plt.plot(x_true, y_true, 'k--', label="Physical Drone Truth (Off-Course)", alpha=0.5)
    plt.plot(x_gps, y_gps, 'rx', label="Spoofed GPS Signal (Lying)", alpha=0.3)
    plt.plot(x_ekf, y_ekf, 'b-', linewidth=2, label="SensorSentry Estimate")
    
    plt.title("Drone Hijack Defense: GPS Spoofing vs SensorSentry")
    plt.xlabel("X Position (m)")
    plt.ylabel("Y Position (m)")
    plt.grid(True)
    plt.legend()
    plt.savefig("drone_hijack_simulation.png", dpi=300)
    print("Saved plot to 'drone_hijack_simulation.png'.")
    plt.show()

if __name__ == "__main__":
    simulate_drone_hijack()
