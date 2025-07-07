"""
DEPRECATED: This file has been refactored into multiple modules.
Please use 'from src.utils.components import ...' instead.

This file is kept for backward compatibility and will be removed in future versions.
"""

# Import all classes from the new modular structure for backward compatibility
from .components import (
    Flit, 
    TokenBucket, 
    Node, 
    IPInterface, 
    RingIPInterface, 
    create_ring_ip_interface,
    Network, 
    RingNetwork
)

# Issue deprecation warning when this file is imported
import warnings
warnings.warn(
    "src.utils.component is deprecated. Please use 'from src.utils.components import ...' instead.",
    DeprecationWarning,
    stacklevel=2
)