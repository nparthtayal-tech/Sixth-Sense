"""
ml_monitor.py — Machine Learning Temporal Anomaly Monitor
=========================================================
LSTM Autoencoder (LSTM-AE) sequence-level anomaly detector for EKF
innovation residuals.

Purpose:
    Complements Chi-Square (fast single-frame gate) and CUSUM (linear drift)
    by detecting non-linear, stealthy, oscillating, or multi-step temporal
    anomalies in EKF residuals across sliding time windows.

Input:
    Innovation vector y and covariance S from EKF.compute_innovation().

Output:
    MLSignal with HEALTHY / WARNING / ALARM verdict and reconstruction MSE.

Hardware Interface:
    MLSignal.to_bytes() → 16-byte packet for CAN / UART / SPI.
    MLSignal.to_dict()  → JSON-serializable dict for ROS2 / TCP / MQTT.
"""

import os
import sys
import struct
from collections import deque
from dataclasses import dataclass
from enum import IntEnum
from typing import Dict, Optional

import numpy as np
import torch
import torch.nn as nn

# Ensure LSTM_AutoEncoder-master models can be imported if needed
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LSTM_AE_DIR = os.path.join(BASE_DIR, "LSTM_AutoEncoder-master")
if LSTM_AE_DIR not in sys.path:
    sys.path.insert(0, LSTM_AE_DIR)

try:
    from models.LSTMAE import LSTMAE
except ImportError:
    # Direct fallback implementation matching LSTMAE architecture
    class _Encoder(nn.Module):
        def __init__(self, input_size: int, hidden_size: int, dropout: float, seq_len: int):
            super().__init__()
            self.lstm_enc = nn.LSTM(input_size=input_size, hidden_size=hidden_size, dropout=dropout, batch_first=True)

        def forward(self, x):
            out, (last_h, _) = self.lstm_enc(x)
            x_enc = last_h.squeeze(dim=0).unsqueeze(1).repeat(1, x.shape[1], 1)
            return x_enc, out

    class _Decoder(nn.Module):
        def __init__(self, input_size: int, hidden_size: int, dropout: float, seq_len: int, use_act: bool = False):
            super().__init__()
            self.lstm_dec = nn.LSTM(input_size=hidden_size, hidden_size=hidden_size, dropout=dropout, batch_first=True)
            self.fc = nn.Linear(hidden_size, input_size)
            self.use_act = use_act
            self.act = nn.Sigmoid()

        def forward(self, z):
            dec_out, (h, _) = self.lstm_dec(z)
            dec_out = self.fc(dec_out)
            if self.use_act:
                dec_out = self.act(dec_out)
            return dec_out, h

    class LSTMAE(nn.Module):
        def __init__(self, input_size: int, hidden_size: int, dropout_ratio: float, seq_len: int, use_act: bool = False):
            super().__init__()
            self.encoder = _Encoder(input_size, hidden_size, dropout_ratio, seq_len)
            self.decoder = _Decoder(input_size, hidden_size, dropout_ratio, seq_len, use_act)

        def forward(self, x, return_last_h=False, return_enc_out=False):
            x_enc, enc_out = self.encoder(x)
            x_dec, last_h = self.decoder(x_enc)
            if return_last_h:
                return x_dec, last_h
            if return_enc_out:
                return x_dec, enc_out
            return x_dec


class MLStatus(IntEnum):
    """Traffic-light anomaly status from the Machine Learning Monitor."""
    HEALTHY = 0  # Low reconstruction loss (nominal pattern)
    WARNING = 1  # Moderate reconstruction loss (possible anomaly)
    ALARM = 2    # High reconstruction loss (confirmed anomaly / spoofing)


@dataclass
class MLSignal:
    """Output signal packet from the ML Temporal Anomaly Monitor."""
    timestamp: float
    sensor: str
    status: MLStatus
    reconstruction_error: float
    alarm_threshold: float
    warning_threshold: float
    nis: float
    accepted: bool

    def to_dict(self) -> dict:
        return {
            "timestamp": round(self.timestamp, 4),
            "sensor": self.sensor,
            "status": self.status.name,
            "status_code": int(self.status),
            "reconstruction_error": round(float(self.reconstruction_error), 4),
            "alarm_threshold": round(float(self.alarm_threshold), 4),
            "warning_threshold": round(float(self.warning_threshold), 4),
            "nis": round(float(self.nis), 4),
            "accepted": self.accepted,
        }

    def to_bytes(self) -> bytes:
        """16-byte hardware packet: [uint32 ts_ms, uint8 status, uint8 accepted, float32 error, float32 thresh]."""
        ts_ms = int(self.timestamp * 1000) & 0xFFFFFFFF
        return struct.pack(
            "<IBBff",
            ts_ms,
            int(self.status),
            1 if self.accepted else 0,
            float(self.reconstruction_error),
            float(self.alarm_threshold),
        )


class LSTMAutoEncoderMonitor:
    """Monitors EKF innovation sequences using an LSTM Autoencoder."""

    def __init__(
        self,
        seq_len: int = 10,
        hidden_size: int = 16,
        alarm_threshold: float = 220.0,
        warning_threshold: float = 90.0,
        weights_path: Optional[str] = None,
        use_act: bool = False,
        require_trained_weights: bool = True,
    ):
        self.seq_len = seq_len
        self.hidden_size = hidden_size
        self.alarm_threshold = float(alarm_threshold)
        self.warning_threshold = float(warning_threshold)
        self.use_act = use_act
        self.require_trained_weights = require_trained_weights

        # Rolling window deque per sensor name
        self._buffers: Dict[str, deque] = {}

        # Instantiate PyTorch model
        self.model = LSTMAE(
            input_size=1,
            hidden_size=hidden_size,
            dropout_ratio=0.0,
            seq_len=seq_len,
            use_act=use_act,
        )

        # Loading is explicit. A colocated file is not sufficient evidence that
        # its training set, threshold, and aircraft configuration are approved.
        self.weights_loaded = False
        if weights_path:
            if not os.path.exists(weights_path):
                print(f"[MLMonitor] Warning: weights file does not exist: {weights_path}")
            else:
                try:
                    state_dict = torch.load(weights_path, map_location="cpu")
                    self.model.load_state_dict(state_dict)
                    self.weights_loaded = True
                except Exception as e:
                    print(f"[MLMonitor] Warning: Failed to load weights from {weights_path}: {e}")

        # A random autoencoder is not a detector.  Keep the ML branch disabled
        # until a model specifically validated for this aircraft/configuration
        # has been supplied and loaded successfully.
        self.enabled = self.weights_loaded or not self.require_trained_weights
        if not self.enabled:
            print("[MLMonitor] Disabled: no validated trained weights were loaded.")
        self.model.eval()

    def reset(self, sensor: Optional[str] = None) -> None:
        """Clear residual sequence buffers."""
        if sensor is None:
            self._buffers.clear()
        elif sensor in self._buffers:
            self._buffers[sensor].clear()

    def evaluate(
        self,
        sensor: str,
        timestamp: float,
        y: np.ndarray,
        S: np.ndarray,
        dim_m: int = 6,
    ) -> MLSignal:
        """Evaluate a new EKF innovation measurement against the learned model."""
        # 1. Compute scalar NIS
        try:
            SI = np.linalg.inv(S)
            nis = float(np.dot(y.T, np.dot(SI, y)))
        except Exception:
            nis = float(np.sum(y ** 2))

        if not self.enabled:
            return MLSignal(
                timestamp=timestamp,
                sensor=sensor,
                status=MLStatus.HEALTHY,
                reconstruction_error=0.0,
                alarm_threshold=self.alarm_threshold,
                warning_threshold=self.warning_threshold,
                nis=nis,
                accepted=True,
            )

        # 2. Maintain sliding window
        if sensor not in self._buffers:
            self._buffers[sensor] = deque(maxlen=self.seq_len)
        buf = self._buffers[sensor]
        buf.append(nis)

        # 3. If window is not yet full, return healthy default
        if len(buf) < self.seq_len:
            return MLSignal(
                timestamp=timestamp,
                sensor=sensor,
                status=MLStatus.HEALTHY,
                reconstruction_error=0.0,
                alarm_threshold=self.alarm_threshold,
                warning_threshold=self.warning_threshold,
                nis=nis,
                accepted=True,
            )

        # 4. Neural inference
        seq_array = np.array(buf, dtype=np.float32).reshape(1, self.seq_len, 1)
        inp_tensor = torch.tensor(seq_array)

        with torch.no_grad():
            reconstructed = self.model(inp_tensor)
            mse = float(torch.mean((inp_tensor - reconstructed) ** 2).item())

        # 5. Determine status
        if mse >= self.alarm_threshold:
            status = MLStatus.ALARM
            accepted = False
        elif mse >= self.warning_threshold:
            status = MLStatus.WARNING
            accepted = True
        else:
            status = MLStatus.HEALTHY
            accepted = True

        return MLSignal(
            timestamp=timestamp,
            sensor=sensor,
            status=status,
            reconstruction_error=mse,
            alarm_threshold=self.alarm_threshold,
            warning_threshold=self.warning_threshold,
            nis=nis,
            accepted=accepted,
        )
