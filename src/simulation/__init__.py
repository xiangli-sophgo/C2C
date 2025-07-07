"""
C2C Simulation Framework

This module provides event-driven simulation capabilities for chip-to-chip communication,
integrating with existing topology and protocol components.
"""

from .engine import C2CSimulationEngine
from .events import SimulationEvent
from .fake_chip import FakeChip
from .stats import SimulationStats

__all__ = [
    "C2CSimulationEngine",
    "SimulationEvent",
    "FakeChip",
    "SimulationStats",
]
