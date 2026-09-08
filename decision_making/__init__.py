"""
decision_making
================
Sensor Health Decision-Making Subsystem.

Independent monitors that consume EKF innovation output (y, S)
and produce hardware-ready decision signals:

1. chi_square_gate.py             — Fast Change Monitor (instantaneous spike rejection)
2. cusum_monitor.py               — Slow Change Monitor (persistent drift detection)
3. ml_monitor.py                  — Temporal Sequence Monitor (LSTM-AutoEncoder loss)
4. directional_spoof_detector.py — Directional Vector Monitor (unidirectional vector accumulator & filter)

Each monitor operates independently. They can be used standalone,
together, or piped into downstream hardware controllers.
"""

from .chi_square_gate import ChiSquareGate, FastVerdict, DecisionSignal
from .cusum_monitor import CusumMonitor, DriftStatus, DriftSignal
from .ml_monitor import LSTMAutoEncoderMonitor, MLStatus, MLSignal
from .directional_spoof_detector import DirectionalSpoofDetector, DirectionalStatus, DirectionalSpoofSignal
from .high_frequency_attack_detector import HighFrequencyAttackDetector, AttackPattern, HighFrequencySignal

__all__ = [
    'ChiSquareGate', 'FastVerdict', 'DecisionSignal',
    'CusumMonitor', 'DriftStatus', 'DriftSignal',
    'LSTMAutoEncoderMonitor', 'MLStatus', 'MLSignal',
    'DirectionalSpoofDetector', 'DirectionalStatus', 'DirectionalSpoofSignal',
    'HighFrequencyAttackDetector', 'AttackPattern', 'HighFrequencySignal',
]
