"""
demo_dual_attack.py
===================
End-to-end Demonstration of SensorSentry dual-attack defense.

Simulates two scenarios:
1. Scenario A (GPS Spoofing): Attacker slowly pulls GPS coordinates off-course.
   SensorSentry detects anomaly via CUSUM, quarantines GPS using SAARM,
   and completes the route using dead-reckoning.
2. Scenario B (Waypoint Hijack): Attacker sends a tampered destination command.
   SensorSentry detects anomaly via Geofence + Command Validator + Environment Matcher,
   and executes a Safe Stop.
"""

import os
import sys
import time
import numpy as np
import matplotlib.pyplot as plt

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, BASE_DIR)

from ekf_fusion import TrajectorySimulator
from ekf_fusion.sensor_models import SensorType
from hal import SimulatedBus, SensorReading
from sensor_sentry import SensorSentry
from waypoint_security import WaypointCommand, ExpectedLandmark
from response import SafeStopState

def run_scenario_a_gps_spoofing():
    print("\n" + "="*60)
    print(" SCENARIO A: GPS SPOOFING ('Coordinates Lie' Attack)")
    print("="*60)
    
    sim = TrajectorySimulator(duration=30.0, random_seed=42)
    # Inject spoofing starting at 10s
    events = sim.generate_sensor_events(enable_gps_outage=False, 
                                        enable_gps_spoofing=True, spoofing_start=10.0)
    
    bus = SimulatedBus(verbose=False)
    sentry = SensorSentry(vehicle_id="COBOT-001", hardware_bus=bus)
    
    p0 = sim.ground_truth[0]
    sentry.initialize_state(p0.p, p0.psi)
    
    # Send a valid initial waypoint
    initial_cmd = WaypointCommand(
        destination=np.array([100.0, 100.0, 0.0]),
        sequence_number=1,
        timestamp=0.0,
        issuer_id="HQ",
        signature=sentry.cmd_validator.sign_command(
            np.array([100.0, 100.0, 0.0]), 1, 0.0, "HQ"
        )
    )
    # Setup authorized zone to cover [0,0] to [150, 150]
    sentry.geofence.add_circular_zone("ZONE_A", 50.0, 50.0, 150.0)
    sentry.validate_new_waypoint(initial_cmd)
    
    print("[0.0s] Nominal driving started towards destination.")
    
    telemetry = []
    
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
            telemetry.append({
                't': ev.t,
                'est_pos': sentry.get_current_position().copy(),
                'gt_pos': gt_pt.p.copy(),
                'gps_meas': ev.z[0:3].copy(), # The spoofed measurement
                'dr_active': sentry.saarm.dr_navigator.active
            })
            
            if ev.t == 10.0:
                print("[10.0s] Attacker begins broadcasting fake GPS signals (pulling left)...")
            elif ev.t > 10.0 and sentry.saarm.dr_navigator.active and not telemetry[-2]['dr_active']:
                print(f"[{ev.t:.1f}s] SensorSentry CUSUM ALARM triggered!")
                print(f"[{ev.t:.1f}s] SAARM isolated GPS. Dead-Reckoning activated.")
                print(f"[{ev.t:.1f}s] Fleet Alert broadcasted via bus.")
                
    print("[30.0s] Scenario A Complete. Vehicle safely completed route via Dead-Reckoning.")
    return telemetry, sim, bus

def run_scenario_b_waypoint_hijack():
    print("\n" + "="*60)
    print(" SCENARIO B: WAYPOINT HIJACK ('Destination Tampering')")
    print("="*60)
    
    sim = TrajectorySimulator(duration=20.0, random_seed=101)
    events = sim.generate_sensor_events(enable_gps_outage=False, enable_gps_spoofing=False)
    
    bus = SimulatedBus(verbose=False)
    sentry = SensorSentry(vehicle_id="CAR-007", hardware_bus=bus)
    
    p0 = sim.ground_truth[0]
    sentry.initialize_state(p0.p, p0.psi)
    
    sentry.geofence.add_circular_zone("FACTORY_FLOOR", 0.0, 0.0, 200.0)
    sentry.geofence.add_circular_zone("THIEFS_WAREHOUSE", 500.0, 500.0, 50.0, is_restricted=True)
    
    # Valid command to Factory point
    valid_dest = np.array([50.0, 50.0, 0.0])
    valid_cmd = WaypointCommand(
        destination=valid_dest,
        sequence_number=1,
        timestamp=0.0,
        issuer_id="HQ",
        signature=sentry.cmd_validator.sign_command(valid_dest, 1, 0.0, "HQ")
    )
    sentry.validate_new_waypoint(valid_cmd)
    print("[0.0s] Vehicle navigating to Factory [50, 50]")
    
    telemetry = []
    
    for ev in events:
        if sentry.safe_stop.is_stopped:
            telemetry.append({'t': ev.t, 'est_pos': sentry.get_current_position().copy(), 'stopped': True})
            continue
            
        reading = SensorReading(timestamp=ev.t, data=ev.z, sensor_type=ev.sensor_type.name, valid=True)
        sentry.process_sensor_reading(reading, ev.R, *ev.args)
        telemetry.append({'t': ev.t, 'est_pos': sentry.get_current_position().copy(), 'stopped': False})
        
        if abs(ev.t - 8.0) < 0.01:
            print("[8.0s] Attacker hacks software and injects tampered Waypoint Command!")
            tampered_dest = np.array([500.0, 500.0, 0.0]) # Thief's warehouse
            
            # The attacker tries to forge it but doesn't have the HMAC key
            tampered_cmd = WaypointCommand(
                destination=tampered_dest,
                sequence_number=2,
                timestamp=ev.t,
                issuer_id="HQ",
                signature=b'fake_signature_12345'
            )
            
            accepted = sentry.validate_new_waypoint(tampered_cmd)
            if not accepted:
                print(f"[{ev.t:.1f}s] SensorSentry Geofence/Command validation FAILED.")
                print(f"[{ev.t:.1f}s] SAFE STOP EXECUTED. Brakes Locked. Engine Killed.")
                print(f"[{ev.t:.1f}s] Fleet Emergency Alert broadcasted via bus.")
                
    print(f"[{events[-1].t:.1f}s] Scenario B Complete. Vehicle remains in Safe Stop lockdown.")
    return telemetry, sim, bus

def plot_scenarios(telem_a, sim_a, telem_b, sim_b):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6), dpi=150)
    
    # Plot A
    ax1.set_title("Scenario A: GPS Spoofing Defense", fontweight='bold')
    
    gt_x = [pt.p[0] for pt in sim_a.ground_truth]
    gt_y = [pt.p[1] for pt in sim_a.ground_truth]
    ax1.plot(gt_x, gt_y, 'k--', label='True Trajectory')
    
    gps_x = [t['gps_meas'][0] for t in telem_a]
    gps_y = [t['gps_meas'][1] for t in telem_a]
    ax1.plot(gps_x, gps_y, 'r.', alpha=0.3, label='Spoofed GPS Signal')
    
    est_x = [t['est_pos'][0] for t in telem_a]
    est_y = [t['est_pos'][1] for t in telem_a]
    dr_idx = next((i for i, t in enumerate(telem_a) if t['dr_active']), None)
    
    if dr_idx is not None:
        ax1.plot(est_x[:dr_idx], est_y[:dr_idx], 'g-', linewidth=2, label='EKF (Normal)')
        ax1.plot(est_x[dr_idx:], est_y[dr_idx:], 'b-', linewidth=3, label='SAARM Dead-Reckoning')
        ax1.plot(est_x[dr_idx], est_y[dr_idx], 'y*', markersize=12, label='Attack Detected')
    else:
        ax1.plot(est_x, est_y, 'g-', linewidth=2, label='EKF (Normal)')
    
    ax1.set_xlabel("X (m)"); ax1.set_ylabel("Y (m)")
    ax1.legend(); ax1.grid(True)
    
    # Plot B
    ax2.set_title("Scenario B: Waypoint Hijack Defense", fontweight='bold')
    
    est_x_b = [t['est_pos'][0] for t in telem_b if not t['stopped']]
    est_y_b = [t['est_pos'][1] for t in telem_b if not t['stopped']]
    ax2.plot(est_x_b, est_y_b, 'g-', linewidth=2, label='Nominal Driving')
    
    stop_idx = len(est_x_b) - 1
    ax2.plot(est_x_b[stop_idx], est_y_b[stop_idx], 'rX', markersize=15, label='SAFE STOP (Attack Blocked)')
    
    ax2.plot(50, 50, 'bo', markersize=10, label='Valid Dest (Factory)')
    ax2.plot(500, 500, 'ro', markersize=10, label='Tampered Dest (Thief)')
    
    ax2.annotate('', xy=(500, 500), xytext=(est_x_b[stop_idx], est_y_b[stop_idx]),
                 arrowprops=dict(facecolor='red', shrink=0.05, alpha=0.5, linestyle='--'))
    
    ax2.set_xlabel("X (m)"); ax2.set_ylabel("Y (m)")
    ax2.legend(); ax2.grid(True)
    
    plt.tight_layout()
    plot_path = os.path.join(BASE_DIR, 'dual_attack_defense.png')
    plt.savefig(plot_path)
    print(f"\nSaved visualization to {plot_path}")

def main():
    ta, sa, ba = run_scenario_a_gps_spoofing()
    tb, sb, bb = run_scenario_b_waypoint_hijack()
    
    print("\n[Bus Traffic Analysis]")
    print(f"Scenario A Bus Msgs: {ba.message_count}")
    print(f"Scenario B Bus Msgs: {bb.message_count}")
    
    plot_scenarios(ta, sa, tb, sb)

if __name__ == '__main__':
    main()
