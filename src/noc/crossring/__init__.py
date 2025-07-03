"""
CrossRing NoC topology implementation.

This module provides specific implementations for CrossRing topology
including configuration, validation, and topology construction.
"""

from .config import CrossRingConfig
from .config_factory import CrossRingConfigFactory

__all__ = ['CrossRingConfig', 'CrossRingConfigFactory']