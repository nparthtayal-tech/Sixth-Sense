"""
sensor_interface.py — Hardware-Agnostic Sensor Reader Interface
================================================================
Abstract base class and concrete driver stubs for reading sensor
data from physical hardware.

The concrete classes in this module are development scaffolding, not certified
device drivers. ``main.py`` does not instantiate them for live operations; it
accepts only the authenticated schema-v1 telemetry bridge described in the
deployment specification.

Each driver is designed to be swapped with real hardware SDKs:
    - GPSReceiver      → ublox / NMEA parser
    - IMUDriver        → MPU6050 / BNO055 / ICM-20948
    - WheelEncoderDriver → Quadrature decoder / CAN encoder
    - LiDARDriver      → RPLiDAR / Velodyne / Livox
    - CameraDriver     → OpenCV / RealSense / ZED
    - UltrasonicDriver → HC-SR04 / MB1240 / MaxBotix

Usage in production:
    Replace the stub with your hardware-specific driver:
    
    class MyGPS(GPSReceiver):
        def __init__(self):
            super().__init__()
            import serial
            self._serial = serial.Serial('/dev/ttyACM0', 9600)
        
        def read(self):
            nmea_line = self._serial.readline()
            # Parse NMEA sentence...
            return SensorReading(timestamp=time.time(), data=parsed_data,
                                 sensor_type='GNSS', valid=True)
"""

import time
import numpy as np
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional, Dict, Any
from enum import Enum


@dataclass
class SensorReading:
    """
    A single timestamped measurement from any sensor.
    This is the universal data format flowing into SensorSentry.
    """
    timestamp: float              # System time (seconds)
    data: np.ndarray              # Measurement vector
    sensor_type: str              # Sensor modality name
    valid: bool                   # Hardware health flag
    metadata: Optional[Dict[str, Any]] = None  # Extra info (e.g., satellite count for GPS)

    def to_dict(self) -> dict:
        result = {
            'timestamp': round(self.timestamp, 6),
            'sensor_type': self.sensor_type,
            'data': [round(float(v), 6) for v in self.data],
            'valid': self.valid
        }
        if self.metadata:
            result['metadata'] = self.metadata
        return result


class SensorInterface(ABC):
    """
    Abstract base class for hardware sensor drivers.
    
    Implement read() for your specific hardware.
    SensorSentry calls read() at the configured sampling rate.
    """
    
    def __init__(self, sensor_type: str, sampling_rate_hz: float = 10.0):
        self.sensor_type = sensor_type
        self.sampling_rate_hz = sampling_rate_hz
        self._is_initialized = False
        self._read_count = 0
        self._error_count = 0
        self._last_read_time = 0.0
    
    @abstractmethod
    def read(self) -> Optional[SensorReading]:
        """
        Read one measurement from the sensor hardware.
        
        Returns
        -------
        SensorReading or None if no data available.
        """
        pass
    
    @abstractmethod
    def initialize(self) -> bool:
        """
        Initialize the hardware sensor.
        Called once at startup.
        
        Returns
        -------
        bool : True if initialization succeeded.
        """
        pass
    
    def shutdown(self):
        """Release hardware resources. Override if needed."""
        self._is_initialized = False
    
    @property
    def is_ready(self) -> bool:
        return self._is_initialized
    
    @property
    def health_ratio(self) -> float:
        """Fraction of successful reads (0.0 to 1.0)."""
        total = self._read_count + self._error_count
        if total == 0:
            return 1.0
        return self._read_count / total


# ============================================================================
# Concrete Sensor Driver Stubs
# Replace these with real hardware drivers for deployment
# ============================================================================

class GPSReceiver(SensorInterface):
    """
    GPS / GNSS receiver driver stub.
    
    Production replacements:
        - ublox (u-blox M8/F9P via serial/I2C)
        - gpsd (Linux GPS daemon)
        - NMEA 0183 parser
    
    Output: [px, py, pz, vx, vy, vz] in local ENU frame
    """
    
    def __init__(self, port: str = '/dev/ttyACM0', baudrate: int = 9600):
        super().__init__(sensor_type='GNSS', sampling_rate_hz=5.0)
        self.port = port
        self.baudrate = baudrate
        self._serial = None
    
    def initialize(self) -> bool:
        """
        Initialize GPS receiver.
        In production: open serial port, configure receiver.
        """
        try:
            # STUB: In production, uncomment:
            # import serial
            # self._serial = serial.Serial(self.port, self.baudrate, timeout=1)
            self._is_initialized = True
            return True
        except Exception:
            self._is_initialized = False
            return False
    
    def read(self) -> Optional[SensorReading]:
        """
        Read GPS fix.
        In production: parse NMEA sentences from serial.
        """
        if not self._is_initialized:
            return None
        
        # STUB: Return None — real driver parses NMEA/UBX
        # In production:
        # nmea = self._serial.readline().decode()
        # lat, lon, alt = parse_gga(nmea)
        # vn, ve, vd = parse_vtg(nmea)
        # return SensorReading(timestamp=time.time(),
        #                      data=np.array([lon, lat, alt, ve, vn, -vd]),
        #                      sensor_type='GNSS', valid=True,
        #                      metadata={'satellites': sat_count})
        self._read_count += 1
        return None
    
    def shutdown(self):
        if self._serial is not None:
            self._serial.close()
        super().shutdown()


class IMUDriver(SensorInterface):
    """
    IMU (Inertial Measurement Unit) driver stub.
    
    Production replacements:
        - MPU6050 (I2C)
        - BNO055 (I2C, pre-fused)
        - ICM-20948 (SPI/I2C)
        - ADIS16495 (SPI, tactical grade)
    
    Output: [omega_z, a_forward] (yaw rate + body-frame acceleration)
    """
    
    def __init__(self, bus_type: str = 'i2c', address: int = 0x68):
        super().__init__(sensor_type='IMU', sampling_rate_hz=100.0)
        self.bus_type = bus_type
        self.address = address
    
    def initialize(self) -> bool:
        """
        Initialize IMU.
        In production: configure accelerometer/gyro ranges, DLPF.
        """
        # STUB: In production, configure registers
        # import smbus2
        # self._bus = smbus2.SMBus(1)
        # self._bus.write_byte_data(self.address, 0x6B, 0x00)  # Wake up MPU6050
        self._is_initialized = True
        return True
    
    def read(self) -> Optional[SensorReading]:
        if not self._is_initialized:
            return None
        # STUB: Real driver reads I2C/SPI registers
        self._read_count += 1
        return None


class WheelEncoderDriver(SensorInterface):
    """
    Wheel encoder / joint sensor driver stub.
    
    Production replacements:
        - Quadrature encoder via GPIO (Raspberry Pi)
        - CAN-connected wheel speed sensor
        - ROS2 /odom topic
    
    Output: [v_forward] (forward speed in m/s)
    """
    
    def __init__(self, gpio_pin_a: int = 17, gpio_pin_b: int = 18,
                 ticks_per_rev: int = 1024, wheel_radius_m: float = 0.05):
        super().__init__(sensor_type='WHEEL_ENCODER', sampling_rate_hz=50.0)
        self.gpio_pin_a = gpio_pin_a
        self.gpio_pin_b = gpio_pin_b
        self.ticks_per_rev = ticks_per_rev
        self.wheel_radius_m = wheel_radius_m
    
    def initialize(self) -> bool:
        # STUB: Setup GPIO interrupts for quadrature decoding
        self._is_initialized = True
        return True
    
    def read(self) -> Optional[SensorReading]:
        if not self._is_initialized:
            return None
        self._read_count += 1
        return None


class LiDARDriver(SensorInterface):
    """
    LiDAR driver stub.
    
    Production replacements:
        - RPLiDAR A1/A2 (serial)
        - Velodyne VLP-16 (UDP)
        - Livox Mid-40 (UDP)
        - Intel RealSense L515
    
    Output: [range, azimuth, elevation] per point
    """
    
    def __init__(self, port: str = '/dev/ttyUSB0'):
        super().__init__(sensor_type='LIDAR', sampling_rate_hz=10.0)
        self.port = port
    
    def initialize(self) -> bool:
        # STUB: Initialize LiDAR motor spin-up, configure scan rate
        self._is_initialized = True
        return True
    
    def read(self) -> Optional[SensorReading]:
        if not self._is_initialized:
            return None
        self._read_count += 1
        return None


class CameraDriver(SensorInterface):
    """
    Camera / visual tracking driver stub.
    
    Production replacements:
        - OpenCV VideoCapture (USB / CSI cameras)
        - Intel RealSense D435/D455
        - ZED 2i stereo camera
        - ROS2 /camera/image_raw topic
    
    Output: [azimuth, elevation] bearing to tracked landmark
    """
    
    def __init__(self, device_id: int = 0, width: int = 640, height: int = 480):
        super().__init__(sensor_type='CAMERA', sampling_rate_hz=20.0)
        self.device_id = device_id
        self.width = width
        self.height = height
    
    def initialize(self) -> bool:
        # STUB: Open camera, configure resolution / FPS
        # import cv2
        # self._cap = cv2.VideoCapture(self.device_id)
        # self._cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
        # self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
        self._is_initialized = True
        return True
    
    def read(self) -> Optional[SensorReading]:
        if not self._is_initialized:
            return None
        self._read_count += 1
        return None


class UltrasonicDriver(SensorInterface):
    """
    Ultrasonic range sensor driver stub.
    
    Production replacements:
        - HC-SR04 (GPIO trigger/echo)
        - MaxBotix MB1240 (I2C / serial)
        - ROS2 /sonar topic
    
    Output: [distance] in meters
    """
    
    def __init__(self, trigger_pin: int = 23, echo_pin: int = 24):
        super().__init__(sensor_type='ULTRASONIC', sampling_rate_hz=20.0)
        self.trigger_pin = trigger_pin
        self.echo_pin = echo_pin
    
    def initialize(self) -> bool:
        # STUB: Setup GPIO pins for trigger / echo
        self._is_initialized = True
        return True
    
    def read(self) -> Optional[SensorReading]:
        if not self._is_initialized:
            return None
        self._read_count += 1
        return None
