"""
hal — Hardware Abstraction Layer
==================================
Provides unified interfaces for deploying SensorSentry to
physical hardware platforms:

1. hardware_bus.py     — CAN / UART / SPI / I2C / ROS2 bus adapters
2. sensor_interface.py — Hardware-agnostic sensor reader stubs

All SensorSentry signals route through the HardwareBus for
deployment-ready code on cobots, cars, drones, and embedded platforms.
"""

from .hardware_bus import (
    HardwareBus,
    CANBusAdapter,
    UARTAdapter,
    SPIAdapter,
    ROS2Adapter,
    SimulatedBus
)
from .sensor_interface import (
    SensorInterface,
    GPSReceiver,
    IMUDriver,
    WheelEncoderDriver,
    LiDARDriver,
    CameraDriver,
    UltrasonicDriver
)

__all__ = [
    'HardwareBus', 'CANBusAdapter', 'UARTAdapter', 'SPIAdapter',
    'ROS2Adapter', 'SimulatedBus',
    'SensorInterface', 'GPSReceiver', 'IMUDriver', 'WheelEncoderDriver',
    'LiDARDriver', 'CameraDriver', 'UltrasonicDriver'
]
