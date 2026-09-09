"""Small, explicit boundary between safety decisions and hardware transports."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from time import time
from typing import List


@dataclass(frozen=True)
class BusMessage:
    timestamp: float
    channel: str
    payload: bytes


class HardwareBus(ABC):
    """Transport for notifications only; it is not a motor-control interface."""

    @abstractmethod
    def send(self, payload: bytes, channel: str) -> bool:
        """Transmit a safety/alert message."""

    def close(self) -> None:
        """Release optional transport resources."""


class SimulatedBus(HardwareBus):
    """In-memory bus used by tests and dry runs."""

    def __init__(self) -> None:
        self._messages: List[BusMessage] = []

    def send(self, payload: bytes, channel: str) -> bool:
        self._messages.append(BusMessage(time(), channel, bytes(payload)))
        return True

    @property
    def messages(self) -> List[BusMessage]:
        return list(self._messages)

    def messages_for(self, channel: str) -> List[BusMessage]:
        return [message for message in self._messages if message.channel == channel]
