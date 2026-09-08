"""
decision_making
================
Sensor Health Decision-Making Subsystem.

Two independent monitors that consume EKF innovation output (y, S)
and produce hardware-ready decision signals:

1. chi_square_gate.py  — Fast Change Monitor (instantaneous spike rejection)
2. cusum_monitor.py    — Slow Change Monitor (persistent drift detection)

Each monitor operates independently. They can be used standalone,
together, or piped into downstream hardware controllers.
"""

from .chi_square_gate import ChiSquareGate, FastVerdict, DecisionSignal
from .cusum_monitor import CusumMonitor, DriftStatus, DriftSignal
from .ml_monitor import LSTMAutoEncoderMonitor, MLStatus, MLSignal

__all__ = [
    'ChiSquareGate', 'FastVerdict', 'DecisionSignal',
    'CusumMonitor', 'DriftStatus', 'DriftSignal',
    'LSTMAutoEncoderMonitor', 'MLStatus', 'MLSignal',
]
