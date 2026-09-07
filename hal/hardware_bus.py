"""
hardware_bus.py — Unified Hardware Communication Bus
======================================================
Abstract interface and concrete adapters for transmitting
SensorSentry signals to hardware controllers.

Supported Buses:
    - CAN Bus    (ISO 11898 / SocketCAN — automotive standard)
    - UART       (RS-232 / TTL serial — Arduino, STM32, Raspberry Pi)
    - SPI        (Serial Peripheral Interface — embedded MCUs)
    - I2C        (Inter-IC — sensor networks)
    - ROS2       (Robot Operating System 2 — cobots, drones)

Each adapter accepts binary packets from SensorSentry signals
(.to_bytes()) and transmits them on the physical bus.

For production deployment, swap the SimulatedBus with the
real adapter matching your hardware.

Usage:
    # Simulation / testing
    bus = SimulatedBus()
    
    # Production CAN bus
    bus = CANBusAdapter(interface='can0', channel='vcan0')
    
    # Production UART serial
    bus = UARTAdapter(port='/dev/ttyUSB0', baudrate=115200)
    
    # All adapters share the same API:
    bus.send(signal.to_bytes(), channel='BRAKE')
"""

import struct
import time
import logging
from abc import ABC, abstractmethod
from typing import Optional, List, Dict
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class BusMessage:
    """A message transmitted on the hardware bus."""
    timestamp: float
    channel: str           # Logical channel (e.g., 'BRAKE', 'ALERT', 'SAARM')
    data: bytes            # Raw binary payload
    bus_type: str          # 'CAN', 'UART', 'SPI', 'ROS2', 'SIMULATED'
    acknowledged: bool = False


class HardwareBus(ABC):
    """
    Abstract base class for hardware communication buses.
    All SensorSentry signals route through this interface.
    """
    
    @abstractmethod
    def send(self, data: bytes, channel: str = 'DEFAULT') -> bool:
        """
        Send binary data on the specified channel.
        
        Parameters
        ----------
        data : bytes
            Binary payload (from signal.to_bytes()).
        channel : str
            Logical channel name for routing.
        
        Returns
        -------
        bool : True if transmission succeeded.
        """
        pass
    
    @abstractmethod
    def receive(self, channel: str = 'DEFAULT', timeout_ms: int = 100) -> Optional[bytes]:
        """
        Receive binary data from the specified channel.
        
        Parameters
        ----------
        channel : str
            Logical channel to listen on.
        timeout_ms : int
            Timeout in milliseconds.
        
        Returns
        -------
        bytes or None : Received data, or None if timeout.
        """
        pass
    
    @abstractmethod
    def is_connected(self) -> bool:
        """Check if the bus is connected and operational."""
        pass
    
    @abstractmethod
    def close(self):
        """Release bus resources."""
        pass


class CANBusAdapter(HardwareBus):
    """
    CAN Bus adapter for automotive / industrial deployment.
    
    Uses SocketCAN on Linux or PCAN / Kvaser on Windows.
    CAN arbitration IDs are mapped from channel names.
    
    Channel → CAN ID Mapping:
        BRAKE   → 0x100
        ALERT   → 0x200
        SAARM   → 0x300
        GEOFENCE → 0x400
        COMMAND → 0x500
        STATUS  → 0x600
    
    Parameters
    ----------
    interface : str
        CAN interface name (e.g., 'can0', 'vcan0').
    bitrate : int
        CAN bus bitrate (default 500000 = 500 kbps).
    """
    
    CAN_ID_MAP = {
        'BRAKE': 0x100,
        'ALERT': 0x200,
        'SAARM': 0x300,
        'GEOFENCE': 0x400,
        'COMMAND': 0x500,
        'STATUS': 0x600,
        'DEFAULT': 0x700,
    }
    
    def __init__(self, interface: str = 'can0', bitrate: int = 500000):
        self.interface = interface
        self.bitrate = bitrate
        self._bus = None
        self._connected = False
        self._tx_count = 0
        self._rx_count = 0
        self._message_log: List[BusMessage] = []
        
        # Try to connect (will fail gracefully on non-CAN systems)
        try:
            import can
            self._bus = can.interface.Bus(channel=interface, interface='socketcan',
                                          bitrate=bitrate)
            self._connected = True
            logger.info(f"CAN bus connected: {interface} @ {bitrate} bps")
        except (ImportError, Exception) as e:
            logger.warning(f"CAN bus unavailable ({e}). Using buffered mode.")
            self._connected = False
    
    def send(self, data: bytes, channel: str = 'DEFAULT') -> bool:
        can_id = self.CAN_ID_MAP.get(channel, 0x700)
        
        msg = BusMessage(
            timestamp=time.time(),
            channel=channel,
            data=data,
            bus_type='CAN',
            acknowledged=False
        )
        
        if self._connected and self._bus is not None:
            try:
                import can
                # CAN frames have max 8 bytes; segment larger payloads
                for i in range(0, len(data), 8):
                    frame = can.Message(
                        arbitration_id=can_id + (i // 8),
                        data=data[i:i+8],
                        is_extended_id=False
                    )
                    self._bus.send(frame)
                msg.acknowledged = True
            except Exception as e:
                logger.error(f"CAN TX error: {e}")
                return False
        
        self._message_log.append(msg)
        self._tx_count += 1
        return True
    
    def receive(self, channel: str = 'DEFAULT', timeout_ms: int = 100) -> Optional[bytes]:
        if not self._connected or self._bus is None:
            return None
        try:
            import can
            msg = self._bus.recv(timeout=timeout_ms / 1000.0)
            if msg is not None:
                self._rx_count += 1
                return bytes(msg.data)
        except Exception:
            pass
        return None
    
    def is_connected(self) -> bool:
        return self._connected
    
    def close(self):
        if self._bus is not None:
            self._bus.shutdown()
            self._connected = False


class UARTAdapter(HardwareBus):
    """
    UART / Serial adapter for embedded deployment.
    
    Compatible with: Arduino, STM32, Raspberry Pi, ESP32.
    
    Protocol:
        [0xAA, 0x55]     — Sync header (2 bytes)
        [channel_id]     — 1 byte channel identifier
        [payload_length] — 2 bytes (big-endian)
        [payload]        — N bytes
        [checksum]       — 1 byte XOR of all payload bytes
    
    Parameters
    ----------
    port : str
        Serial port (e.g., '/dev/ttyUSB0', 'COM3').
    baudrate : int
        Serial baud rate (default 115200).
    """
    
    CHANNEL_IDS = {
        'BRAKE': 0x01,
        'ALERT': 0x02,
        'SAARM': 0x03,
        'GEOFENCE': 0x04,
        'COMMAND': 0x05,
        'STATUS': 0x06,
        'DEFAULT': 0x00,
    }
    
    def __init__(self, port: str = '/dev/ttyUSB0', baudrate: int = 115200):
        self.port = port
        self.baudrate = baudrate
        self._serial = None
        self._connected = False
        self._tx_count = 0
        self._message_log: List[BusMessage] = []
        
        try:
            import serial
            self._serial = serial.Serial(port, baudrate, timeout=0.1)
            self._connected = True
            logger.info(f"UART connected: {port} @ {baudrate} baud")
        except (ImportError, Exception) as e:
            logger.warning(f"UART unavailable ({e}). Using buffered mode.")
            self._connected = False
    
    def send(self, data: bytes, channel: str = 'DEFAULT') -> bool:
        chan_id = self.CHANNEL_IDS.get(channel, 0x00)
        
        # Build framed packet
        header = bytes([0xAA, 0x55, chan_id])
        length = struct.pack('>H', len(data))
        checksum = bytes([0])
        xor = 0
        for b in data:
            xor ^= b
        checksum = bytes([xor])
        
        frame = header + length + data + checksum
        
        msg = BusMessage(
            timestamp=time.time(),
            channel=channel,
            data=frame,
            bus_type='UART'
        )
        
        if self._connected and self._serial is not None:
            try:
                self._serial.write(frame)
                msg.acknowledged = True
            except Exception as e:
                logger.error(f"UART TX error: {e}")
                return False
        
        self._message_log.append(msg)
        self._tx_count += 1
        return True
    
    def receive(self, channel: str = 'DEFAULT', timeout_ms: int = 100) -> Optional[bytes]:
        if not self._connected or self._serial is None:
            return None
        try:
            # Look for sync header
            header = self._serial.read(3)
            if len(header) >= 3 and header[0] == 0xAA and header[1] == 0x55:
                length_bytes = self._serial.read(2)
                if len(length_bytes) == 2:
                    length = struct.unpack('>H', length_bytes)[0]
                    payload = self._serial.read(length)
                    checksum = self._serial.read(1)
                    return bytes(payload)
        except Exception:
            pass
        return None
    
    def is_connected(self) -> bool:
        return self._connected
    
    def close(self):
        if self._serial is not None:
            self._serial.close()
            self._connected = False


class SPIAdapter(HardwareBus):
    """
    SPI adapter for high-speed embedded communication.
    
    Compatible with: STM32, Teensy, FPGA-based controllers.
    
    Parameters
    ----------
    bus_num : int
        SPI bus number (default 0).
    device_num : int
        SPI device / chip select (default 0).
    speed_hz : int
        SPI clock speed (default 1 MHz).
    """
    
    def __init__(self, bus_num: int = 0, device_num: int = 0, speed_hz: int = 1000000):
        self.bus_num = bus_num
        self.device_num = device_num
        self.speed_hz = speed_hz
        self._spi = None
        self._connected = False
        self._tx_count = 0
        self._message_log: List[BusMessage] = []
        
        try:
            import spidev
            self._spi = spidev.SpiDev()
            self._spi.open(bus_num, device_num)
            self._spi.max_speed_hz = speed_hz
            self._connected = True
            logger.info(f"SPI connected: bus={bus_num}, device={device_num}")
        except (ImportError, Exception) as e:
            logger.warning(f"SPI unavailable ({e}). Using buffered mode.")
            self._connected = False
    
    def send(self, data: bytes, channel: str = 'DEFAULT') -> bool:
        msg = BusMessage(
            timestamp=time.time(),
            channel=channel,
            data=data,
            bus_type='SPI'
        )
        
        if self._connected and self._spi is not None:
            try:
                self._spi.xfer2(list(data))
                msg.acknowledged = True
            except Exception as e:
                logger.error(f"SPI TX error: {e}")
                return False
        
        self._message_log.append(msg)
        self._tx_count += 1
        return True
    
    def receive(self, channel: str = 'DEFAULT', timeout_ms: int = 100) -> Optional[bytes]:
        if not self._connected or self._spi is None:
            return None
        try:
            data = self._spi.readbytes(32)
            return bytes(data)
        except Exception:
            return None
    
    def is_connected(self) -> bool:
        return self._connected
    
    def close(self):
        if self._spi is not None:
            self._spi.close()
            self._connected = False


class ROS2Adapter(HardwareBus):
    """
    ROS2 adapter for robotic platforms.
    
    Publishes SensorSentry signals as ROS2 topics.
    Compatible with: TurtleBot, UR5/UR10, Boston Dynamics, custom cobots.
    
    Topic Mapping:
        BRAKE    → /sensorsentry/safe_stop
        ALERT    → /sensorsentry/fleet_alert
        SAARM    → /sensorsentry/saarm_status
        GEOFENCE → /sensorsentry/geofence
        COMMAND  → /sensorsentry/command_validation
        STATUS   → /sensorsentry/system_status
    
    Parameters
    ----------
    node_name : str
        ROS2 node name (default 'sensorsentry_node').
    """
    
    TOPIC_MAP = {
        'BRAKE': '/sensorsentry/safe_stop',
        'ALERT': '/sensorsentry/fleet_alert',
        'SAARM': '/sensorsentry/saarm_status',
        'GEOFENCE': '/sensorsentry/geofence',
        'COMMAND': '/sensorsentry/command_validation',
        'STATUS': '/sensorsentry/system_status',
        'DEFAULT': '/sensorsentry/default',
    }
    
    def __init__(self, node_name: str = 'sensorsentry_node'):
        self.node_name = node_name
        self._node = None
        self._publishers: Dict[str, object] = {}
        self._connected = False
        self._tx_count = 0
        self._message_log: List[BusMessage] = []
        
        try:
            import rclpy
            from std_msgs.msg import ByteMultiArray
            rclpy.init()
            self._node = rclpy.create_node(node_name)
            for channel, topic in self.TOPIC_MAP.items():
                self._publishers[channel] = self._node.create_publisher(
                    ByteMultiArray, topic, 10)
            self._connected = True
            logger.info(f"ROS2 node '{node_name}' initialized")
        except (ImportError, Exception) as e:
            logger.warning(f"ROS2 unavailable ({e}). Using buffered mode.")
            self._connected = False
    
    def send(self, data: bytes, channel: str = 'DEFAULT') -> bool:
        msg = BusMessage(
            timestamp=time.time(),
            channel=channel,
            data=data,
            bus_type='ROS2'
        )
        
        if self._connected and channel in self._publishers:
            try:
                from std_msgs.msg import ByteMultiArray
                ros_msg = ByteMultiArray()
                ros_msg.data = list(data)
                self._publishers[channel].publish(ros_msg)
                msg.acknowledged = True
            except Exception as e:
                logger.error(f"ROS2 publish error: {e}")
                return False
        
        self._message_log.append(msg)
        self._tx_count += 1
        return True
    
    def receive(self, channel: str = 'DEFAULT', timeout_ms: int = 100) -> Optional[bytes]:
        # ROS2 uses callback-based subscription, not polling
        return None
    
    def is_connected(self) -> bool:
        return self._connected
    
    def close(self):
        if self._node is not None:
            self._node.destroy_node()
            try:
                import rclpy
                rclpy.shutdown()
            except Exception:
                pass
            self._connected = False


class SimulatedBus(HardwareBus):
    """
    Simulated hardware bus for testing and development.
    
    Logs all transmitted messages without requiring physical hardware.
    Useful for:
    - Unit testing
    - Simulation runs
    - Development on non-hardware machines
    - Hackathon demos
    """
    
    def __init__(self, verbose: bool = False):
        self.verbose = verbose
        self._connected = True
        self._tx_count = 0
        self._rx_count = 0
        self._message_log: List[BusMessage] = []
        self._rx_buffer: Dict[str, List[bytes]] = {}
    
    def send(self, data: bytes, channel: str = 'DEFAULT') -> bool:
        msg = BusMessage(
            timestamp=time.time(),
            channel=channel,
            data=data,
            bus_type='SIMULATED',
            acknowledged=True
        )
        self._message_log.append(msg)
        self._tx_count += 1
        
        if self.verbose:
            hex_data = data.hex()
            logger.info(f"[SIM-BUS] TX → {channel}: {len(data)} bytes | {hex_data[:40]}{'...' if len(hex_data) > 40 else ''}")
        
        return True
    
    def receive(self, channel: str = 'DEFAULT', timeout_ms: int = 100) -> Optional[bytes]:
        if channel in self._rx_buffer and self._rx_buffer[channel]:
            self._rx_count += 1
            return self._rx_buffer[channel].pop(0)
        return None
    
    def inject_rx(self, data: bytes, channel: str = 'DEFAULT'):
        """Inject data into the receive buffer (for testing)."""
        if channel not in self._rx_buffer:
            self._rx_buffer[channel] = []
        self._rx_buffer[channel].append(data)
    
    def is_connected(self) -> bool:
        return self._connected
    
    def close(self):
        self._connected = False
    
    @property
    def message_count(self) -> int:
        return self._tx_count
    
    @property
    def messages(self) -> List[BusMessage]:
        return list(self._message_log)
    
    def get_messages_by_channel(self, channel: str) -> List[BusMessage]:
        return [m for m in self._message_log if m.channel == channel]
