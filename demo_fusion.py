"""
demo_fusion.py
==============
End-to-end Demonstration of the 8-Sensor EKF with Decision-Making Monitors.

Data Flow:
  Sensor Measurement → EKF.compute_innovation() → [y, S]
       ↓                                            ↓
  Chi-Square Gate (Fast)                     CUSUM Monitor (Slow)
       ↓                                            ↓
  ACCEPT/REJECT signal                       HEALTHY/WARNING/ALARM signal
       ↓                                            ↓
  If both ACCEPT → EKF.apply_update()
  Otherwise      → Discard measurement, log signal for hardware
"""

import os
import sys
import time
import numpy as np
import matplotlib.pyplot as plt

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, BASE_DIR)
sys.path.insert(0, os.path.join(BASE_DIR, 'filterpy-master'))

from ekf_fusion import (
    MultiSensorEKF, SensorType, TrajectorySimulator, normalize_angle
)
from ekf_fusion.sensor_models import (
    gnss_h, gnss_H,
    imu_h, imu_H,
    wheel_encoder_h, wheel_encoder_H,
    ultrasonic_h, ultrasonic_H,
    camera_h, camera_H, camera_residual,
    lidar_h, lidar_H, lidar_residual,
    five_g_h, five_g_H,
    atomic_clock_h, atomic_clock_H
)
from decision_making import ChiSquareGate, CusumMonitor

# Sensor dispatch table: maps SensorType → (H_func, h_func, residual_func, extra_args_key)
SENSOR_DISPATCH = {
    SensorType.GNSS:          (gnss_H, gnss_h, None),
    SensorType.IMU:           (imu_H, imu_h, None),
    SensorType.WHEEL_ENCODER: (wheel_encoder_H, wheel_encoder_h, None),
    SensorType.ULTRASONIC:    (ultrasonic_H, ultrasonic_h, None),
    SensorType.CAMERA:        (camera_H, camera_h, camera_residual),
    SensorType.LIDAR:         (lidar_H, lidar_H, lidar_residual),
    SensorType.FIVE_G:        (five_g_H, five_g_h, None),
    SensorType.ATOMIC_CLOCK:  (atomic_clock_H, atomic_clock_h, None),
}

def get_dispatch(ev):
    """Returns (H_func, h_func, residual_func, args) for a sensor event."""
    st = ev.sensor_type
    if st == SensorType.GNSS:
        return gnss_H, gnss_h, None, ()
    elif st == SensorType.IMU:
        return imu_H, imu_h, None, (ev.args[0], ev.args[1])
    elif st == SensorType.WHEEL_ENCODER:
        return wheel_encoder_H, wheel_encoder_h, None, ()
    elif st == SensorType.ULTRASONIC:
        return ultrasonic_H, ultrasonic_h, None, (ev.args[0],)
    elif st == SensorType.CAMERA:
        return camera_H, camera_h, camera_residual, (ev.args[0],)
    elif st == SensorType.LIDAR:
        return lidar_H, lidar_h, lidar_residual, (ev.args[0],)
    elif st == SensorType.FIVE_G:
        return five_g_H, five_g_h, None, (ev.args[0],)
    elif st == SensorType.ATOMIC_CLOCK:
        return atomic_clock_H, atomic_clock_h, None, ()

def run_fusion_with_monitors(sim, events, use_atomic_clock=True):
    """
    Runs the EKF with external Chi-Square (fast) and CUSUM (slow) monitors.
    
    Data flow per measurement:
      1. EKF computes innovation (y, S) WITHOUT touching state.
      2. Chi-Square gate checks for instant spike → ACCEPT or REJECT.
      3. CUSUM monitor checks for persistent drift → HEALTHY, WARNING, or ALARM.
      4. If BOTH approve → EKF applies the Kalman update.
      5. Otherwise → measurement discarded, hardware signal logged.
    """
    p0 = sim.ground_truth[0]
    initial_x = np.array([
        p0.p[0] + 0.3, p0.p[1] - 0.3, p0.p[2] + 0.1,
        p0.v[0], p0.v[1], p0.v[2],
        p0.psi + 0.02,
        p0.b_clk + 1.0,
        0.0, 0.0, 0.0
    ])
    
    ekf = MultiSensorEKF(initial_x=initial_x, q_accel=0.2, q_gyro=0.01, q_clk_drift=1e-4)
    
    # Instantiate the two independent monitors
    fast_monitor = ChiSquareGate(confidence=0.999)
    slow_monitor = CusumMonitor(alarm_threshold=20.0, warning_threshold=10.0,
                                 slack=1.0, recovery_rate=0.5)
    
    last_t = 0.0
    telemetry = []
    latest_omega = 0.0
    latest_accel = 0.0
    
    # Signal logs for visualization
    fast_signals = []
    slow_signals = []
    rejected_count = 0
    quarantined_count = 0
    
    for ev in events:
        t = ev.t
        dt = t - last_t
        if dt > 1e-6:
            ekf.predict(dt, omega_z=latest_omega, a_fwd=latest_accel)
            last_t = t
            
        st = ev.sensor_type
        
        if st == SensorType.ATOMIC_CLOCK and not use_atomic_clock:
            continue
            
        if st == SensorType.IMU:
            latest_omega = ev.z[0]
            latest_accel = ev.z[1]
            
        H_func, h_func, res_func, args = get_dispatch(ev)
        z = np.atleast_1d(ev.z).astype(float)
        m = len(z)
        st_name = st.name
        
        # Step 1: EKF computes innovation WITHOUT modifying state
        y, S, H, PHT = ekf.compute_innovation(z, H_func, h_func, ev.R, args, res_func)
        
        # Step 2: Fast Change Monitor (Chi-Square)
        fast_sig = fast_monitor.evaluate(st_name, t, y, S, m)
        fast_signals.append(fast_sig)
        
        if fast_sig.verdict.value == 1:  # REJECT
            rejected_count += 1
            gt_pt = sim._interpolate_gt(t)
            telemetry.append({'t': t, 'est_x': ekf.x.copy(),
                              'est_P_diag': np.diag(ekf.P).copy(), 'gt': gt_pt})
            continue
        
        # Step 3: Slow Change Monitor (CUSUM)
        slow_sig = slow_monitor.evaluate(st_name, t, y, S, m)
        slow_signals.append(slow_sig)
        
        if not slow_sig.accepted:  # ALARM → quarantine
            quarantined_count += 1
            gt_pt = sim._interpolate_gt(t)
            telemetry.append({'t': t, 'est_x': ekf.x.copy(),
                              'est_P_diag': np.diag(ekf.P).copy(), 'gt': gt_pt})
            continue
        
        # Step 4: Both monitors approved → Apply Kalman update
        ekf.apply_update(y, S, H, PHT, ev.R)
        
        gt_pt = sim._interpolate_gt(t)
        telemetry.append({'t': t, 'est_x': ekf.x.copy(),
                          'est_P_diag': np.diag(ekf.P).copy(), 'gt': gt_pt})
    
    return telemetry, ekf, fast_monitor, slow_monitor, fast_signals, slow_signals

def compute_metrics(telemetry, t_start=0.0, t_end=40.0):
    pos_errs, vel_errs, psi_errs, clk_errs = [], [], [], []
    for item in telemetry:
        t = item['t']
        if t_start <= t <= t_end:
            est = item['est_x']
            gt = item['gt']
            pos_errs.append(np.linalg.norm(est[0:3] - gt.p))
            vel_errs.append(np.linalg.norm(est[3:6] - gt.v))
            psi_errs.append(abs(normalize_angle(est[6] - gt.psi)) * 180.0 / np.pi)
            clk_errs.append(abs(est[7] - gt.b_clk))
    return {
        'pos_rmse': np.sqrt(np.mean(np.array(pos_errs)**2)),
        'pos_max': np.max(pos_errs),
        'vel_rmse': np.sqrt(np.mean(np.array(vel_errs)**2)),
        'psi_rmse': np.sqrt(np.mean(np.array(psi_errs)**2)),
        'clk_rmse': np.sqrt(np.mean(np.array(clk_errs)**2))
    }

def main():
    print("=" * 78)
    print("   8-SENSOR EKF FUSION + DECISION-MAKING MONITORS BENCHMARK")
    print("   Chi-Square (Fast) + CUSUM (Slow) Change Detection Pipeline")
    print("=" * 78)
    
    duration = 40.0
    outage_start, outage_end = 15.0, 25.0
    
    print(f"\n[1/3] Generating 3D Trajectory & Sensor Streams...")
    sim = TrajectorySimulator(duration=duration, random_seed=101)
    events = sim.generate_sensor_events(True, outage_start, outage_end)
    print(f"      {len(events)} asynchronous events across 8 modalities")
    
    counts = {}
    for ev in events:
        counts[ev.sensor_type.name] = counts.get(ev.sensor_type.name, 0) + 1
    for name, cnt in sorted(counts.items()):
        print(f"       - {name:<15}: {cnt} updates")
    
    print(f"\n[2/3] Running EKF with Chi-Square Gate + CUSUM Monitor...")
    t0 = time.time()
    telem, ekf, fast_mon, slow_mon, fast_sigs, slow_sigs = \
        run_fusion_with_monitors(sim, events, use_atomic_clock=True)
    elapsed = time.time() - t0
    print(f"      Done in {elapsed:.2f}s ({len(events)/elapsed:.0f} events/sec)")
    
    # Print monitor statistics
    print(f"\n      +-----------------------------------------------------------+")
    print(f"      |  FAST MONITOR (Chi-Square)                               |")
    print(f"      |  Total Evaluated : {fast_mon.total_evaluated:<6}                              |")
    print(f"      |  Spikes Rejected : {fast_mon.total_rejected:<6} ({fast_mon.rejection_rate()*100:.1f}%)                       |")
    print(f"      +-----------------------------------------------------------+")
    print(f"      |  SLOW MONITOR (CUSUM)                                    |")
    cusum_vals = slow_mon.get_all_cusum_values()
    for sname, cval in sorted(cusum_vals.items()):
        q_flag = " [QUARANTINED]" if slow_mon.is_quarantined(sname) else ""
        print(f"      |    {sname:<15}: CUSUM = {cval:>6.2f}{q_flag:<20}  |")
    print(f"      +-----------------------------------------------------------+")
    
    print(f"\n[3/3] Computing Metrics & Generating Plots...")
    m_all = compute_metrics(telem, 0.0, duration)
    m_outage = compute_metrics(telem, outage_start, outage_end)
    
    print(f"\n{'Condition':<25} | {'Pos RMSE':<10} | {'Pos Max':<10} | {'Vel RMSE':<10} | {'Heading':<8} | {'Clock'}")
    print("-" * 78)
    print(f"{'Overall (40s)':<25} | {m_all['pos_rmse']:<9.3f}m | {m_all['pos_max']:<9.3f}m | {m_all['vel_rmse']:<9.3f}m/s | {m_all['psi_rmse']:<6.2f}°  | {m_all['clk_rmse']:<.3f}m")
    print(f"{'GPS Outage (15-25s)':<25} | {m_outage['pos_rmse']:<9.3f}m | {m_outage['pos_max']:<9.3f}m | {m_outage['vel_rmse']:<9.3f}m/s | {m_outage['psi_rmse']:<6.2f}°  | {m_outage['clk_rmse']:<.3f}m")
    print("=" * 78)
    
    # ---- PLOT 1: Trajectory ----
    gt_px = [pt.p[0] for pt in sim.ground_truth]
    gt_py = [pt.p[1] for pt in sim.ground_truth]
    est_px = [item['est_x'][0] for item in telem]
    est_py = [item['est_x'][1] for item in telem]
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6), dpi=150)
    ax1.plot(gt_px, gt_py, 'k--', label='Ground Truth', linewidth=2)
    ax1.plot(est_px, est_py, 'g-', label='EKF + Monitors', linewidth=1.5)
    outage_pts = [item for item in telem if outage_start <= item['t'] <= outage_end]
    if outage_pts:
        ox = [item['gt'].p[0] for item in outage_pts]
        oy = [item['gt'].p[1] for item in outage_pts]
        ax1.plot(ox, oy, color='red', linewidth=4, alpha=0.3, label='GPS Outage [15-25s]')
    for gnb in sim.gnodeb_stations:
        ax1.plot(gnb[0], gnb[1], 'm^', markersize=8)
    for lm in sim.optical_landmarks:
        ax1.plot(lm[0], lm[1], 'b*', markersize=8)
    ax1.set_title("2D Ground Track with Decision-Making Monitors", fontweight='bold')
    ax1.set_xlabel("X (m)"); ax1.set_ylabel("Y (m)")
    ax1.legend(fontsize=8); ax1.grid(True, linestyle=':', alpha=0.6)
    
    # ---- PLOT 1 right: Monitor signals timeline ----
    fast_reject_t = [s.timestamp for s in fast_sigs if s.verdict.value == 1]
    slow_warn_t = [s.timestamp for s in slow_sigs if s.status.value == 1]
    slow_alarm_t = [s.timestamp for s in slow_sigs if s.status.value == 2]
    
    ax2.eventplot([fast_reject_t], lineoffsets=[3], linelengths=[0.6],
                  colors=['red'], label='Chi-Square REJECT (spike)')
    ax2.eventplot([slow_warn_t], lineoffsets=[2], linelengths=[0.6],
                  colors=['orange'], label='CUSUM WARNING (drift)')
    ax2.eventplot([slow_alarm_t], lineoffsets=[1], linelengths=[0.6],
                  colors=['darkred'], label='CUSUM ALARM (quarantine)')
    ax2.axvspan(outage_start, outage_end, color='gray', alpha=0.15, label='GPS Outage')
    ax2.set_yticks([1, 2, 3]); ax2.set_yticklabels(['ALARM', 'WARNING', 'REJECT'])
    ax2.set_xlabel("Time (s)")
    ax2.set_title("Decision Monitor Event Timeline", fontweight='bold')
    ax2.legend(fontsize=8, loc='upper right'); ax2.grid(True, linestyle=':', alpha=0.4)
    ax2.set_xlim(0, duration)
    
    plt.tight_layout()
    plot1 = os.path.join(BASE_DIR, 'fusion_trajectory_3d.png')
    plt.savefig(plot1); plt.close()
    print(f"      Saved: {plot1}")
    
    # ---- PLOT 2: Error time-series ----
    times = [item['t'] for item in telem]
    pos_err = [np.linalg.norm(item['est_x'][0:3] - item['gt'].p) for item in telem]
    pos_sig = [np.sqrt(item['est_P_diag'][0]+item['est_P_diag'][1]+item['est_P_diag'][2])*3 for item in telem]
    vel_err = [np.linalg.norm(item['est_x'][3:6] - item['gt'].v) for item in telem]
    psi_err = [abs(normalize_angle(item['est_x'][6]-item['gt'].psi))*180/np.pi for item in telem]
    
    fig, axes = plt.subplots(3, 1, figsize=(12, 8), sharex=True, dpi=150)
    axes[0].plot(times, pos_err, 'b-', linewidth=1.2, label='3D Position Error')
    axes[0].fill_between(times, 0, pos_sig, color='blue', alpha=0.12, label=r'$\pm 3\sigma$')
    axes[0].axvspan(outage_start, outage_end, color='gray', alpha=0.15)
    axes[0].set_ylabel("Pos Error (m)"); axes[0].legend(fontsize=8); axes[0].grid(True,linestyle=':',alpha=0.6)
    axes[0].set_title("EKF Tracking Error with Chi-Square + CUSUM Protection", fontweight='bold')
    axes[1].plot(times, vel_err, 'g-', linewidth=1.2, label='Velocity Error (m/s)')
    axes[1].axvspan(outage_start, outage_end, color='gray', alpha=0.15)
    axes[1].set_ylabel("Vel Error (m/s)"); axes[1].legend(fontsize=8); axes[1].grid(True,linestyle=':',alpha=0.6)
    axes[2].plot(times, psi_err, 'purple', linewidth=1.2, label='Heading Error (deg)')
    axes[2].axvspan(outage_start, outage_end, color='gray', alpha=0.15)
    axes[2].set_ylabel("Heading Err (°)"); axes[2].set_xlabel("Time (s)")
    axes[2].legend(fontsize=8); axes[2].grid(True,linestyle=':',alpha=0.6)
    plt.tight_layout()
    plot2 = os.path.join(BASE_DIR, 'sensor_fusion_errors.png')
    plt.savefig(plot2); plt.close()
    print(f"      Saved: {plot2}")
    
    # ---- PLOT 3: Per-sensor rejection stats + CUSUM levels ----
    fig, (ax_bar, ax_cusum) = plt.subplots(1, 2, figsize=(13, 5), dpi=150)
    
    sensor_names = sorted(counts.keys())
    rej_counts = [fast_mon._per_sensor_rejected.get(s, 0) for s in sensor_names]
    colors = ['#e74c3c' if r > 0 else '#2ecc71' for r in rej_counts]
    ax_bar.barh(sensor_names, rej_counts, color=colors)
    ax_bar.set_xlabel("Chi-Square Spike Rejections")
    ax_bar.set_title("Fast Monitor: Per-Sensor Rejection Count", fontweight='bold')
    ax_bar.grid(True, linestyle=':', alpha=0.6)
    for i, v in enumerate(rej_counts):
        ax_bar.text(v + 0.5, i, str(v), va='center', fontsize=8, fontweight='bold')
    
    cusum_names = sorted(cusum_vals.keys())
    cusum_levels = [cusum_vals[s] for s in cusum_names]
    bar_colors = []
    for s in cusum_names:
        if slow_mon.is_quarantined(s):
            bar_colors.append('#e74c3c')
        elif cusum_vals[s] >= slow_mon.warning_threshold:
            bar_colors.append('#f39c12')
        else:
            bar_colors.append('#2ecc71')
    ax_cusum.barh(cusum_names, cusum_levels, color=bar_colors)
    ax_cusum.axvline(slow_mon.warning_threshold, color='orange', linestyle='--', label=f'Warning ({slow_mon.warning_threshold})')
    ax_cusum.axvline(slow_mon.alarm_threshold, color='red', linestyle='--', label=f'Alarm ({slow_mon.alarm_threshold})')
    ax_cusum.set_xlabel("CUSUM Accumulator Level")
    ax_cusum.set_title("Slow Monitor: Final CUSUM Level by Sensor", fontweight='bold')
    ax_cusum.legend(fontsize=8); ax_cusum.grid(True, linestyle=':', alpha=0.6)
    
    plt.tight_layout()
    plot3 = os.path.join(BASE_DIR, 'sensor_contribution_analysis.png')
    plt.savefig(plot3); plt.close()
    print(f"      Saved: {plot3}")
    
    print("\n" + "=" * 78)
    print("   BENCHMARK COMPLETED SUCCESSFULLY!")
    print("=" * 78)

if __name__ == '__main__':
    main()
