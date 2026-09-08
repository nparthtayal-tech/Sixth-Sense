import os
import sys
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, BASE_DIR)

from ekf_fusion import TrajectorySimulator, MultiSensorEKF
from ekf_fusion.sensor_models import gnss_h, gnss_H, SensorType
from hal import SimulatedBus
from decision_making.ml_monitor import LSTMAutoEncoderMonitor

class ResidualDataset(Dataset):
    def __init__(self, data):
        self.data = data
        
    def __len__(self):
        return self.data.shape[0]
        
    def __getitem__(self, idx):
        return self.data[idx]

def generate_clean_residuals():
    print("Generating clean simulation data...")
    # 200 seconds of normal driving to generate lots of training data
    sim = TrajectorySimulator(duration=200.0, random_seed=99)
    events = sim.generate_sensor_events(enable_gps_outage=False, enable_gps_spoofing=False)
    
    ekf = MultiSensorEKF()
    p0 = sim.ground_truth[0]
    ekf.x[0:3] = p0.p
    ekf.x[6] = p0.psi
    
    last_update_time = 0.0
    latest_omega = 0.0
    latest_accel = 0.0
    
    gnss_nis_values = []
    
    for ev in events:
        dt = ev.t - last_update_time
        if dt > 1e-6:
            ekf.predict(dt, omega_z=latest_omega, a_fwd=latest_accel)
            last_update_time = ev.t
            
        if ev.sensor_type.name == 'IMU':
            latest_omega = ev.z[0]
            latest_accel = ev.z[1]
            
        if ev.sensor_type == SensorType.GNSS:
            # We specifically want GNSS residuals
            y, S, H, PHT = ekf.compute_innovation(ev.z, gnss_H, gnss_h, ev.R, ev.args, None)
            
            try:
                SI = np.linalg.inv(S)
                nis = float(np.dot(y.T, np.dot(SI, y)))
            except:
                nis = float(np.sum(y**2))
                
            gnss_nis_values.append(nis)
            ekf.apply_update(y, S, H, PHT, ev.R)
            
    print(f"Collected {len(gnss_nis_values)} GNSS residuals.")
    return gnss_nis_values

def create_sequences(values, seq_len=10):
    seqs = []
    for i in range(len(values) - seq_len + 1):
        # Shape: (seq_len, 1)
        seq = np.array(values[i:i+seq_len], dtype=np.float32).reshape(-1, 1)
        seqs.append(seq)
    return np.array(seqs)

def train():
    # 1. Generate clean data
    raw_residuals = generate_clean_residuals()
    
    seq_len = 10
    hidden_size = 16
    
    X = create_sequences(raw_residuals, seq_len=seq_len)
    tensor_X = torch.tensor(X)
    print(f"Dataset shape: {tensor_X.shape}") # (num_samples, seq_len, 1)
    
    dataset = ResidualDataset(tensor_X)
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True)
    
    # 2. Setup Model
    print("Initializing LSTM-AE...")
    monitor = LSTMAutoEncoderMonitor(seq_len=seq_len, hidden_size=hidden_size)
    model = monitor.model
    if model is None:
        print("Failed to load model.")
        return
        
    model.train() # Set to train mode
    
    optimizer = optim.Adam(model.parameters(), lr=0.005)
    criterion = nn.MSELoss()
    
    epochs = 50
    print("Starting Training...")
    
    for epoch in range(epochs):
        epoch_loss = 0.0
        for batch_data in dataloader:
            optimizer.zero_grad()
            
            # Autoencoder task: output should reconstruct input
            x_dec = model(batch_data)
            loss = criterion(x_dec, batch_data)
            
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item() * batch_data.size(0)
            
        epoch_loss /= len(dataloader.dataset)
        
        if (epoch + 1) % 10 == 0:
            print(f"Epoch [{epoch+1}/{epochs}], Loss: {epoch_loss:.6f}")
            
    # 3. Save weights
    weights_dir = os.path.join(BASE_DIR, 'decision_making')
    weights_path = os.path.join(weights_dir, 'lstm_ae_weights.pth')
    
    torch.save(model.state_dict(), weights_path)
    print(f"\nTraining Complete! Weights saved to: {weights_path}")
    
    # Calculate a suitable threshold based on the max reconstruction error of clean data
    model.eval()
    with torch.no_grad():
        all_reconstructed = model(tensor_X)
        mses = torch.mean((tensor_X - all_reconstructed)**2, dim=(1,2))
        max_mse = mses.max().item()
        mean_mse = mses.mean().item()
        
    print(f"Clean Data Eval - Mean MSE: {mean_mse:.6f}, Max MSE: {max_mse:.6f}")
    suggested_threshold = max(1.0, max_mse * 2.0)
    print(f"Suggested Anomaly Threshold for ml_monitor.py: {suggested_threshold:.2f}")

if __name__ == "__main__":
    train()
