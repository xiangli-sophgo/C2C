"""
Components package for NoC simulation.
Provides modular components including Flit, Node, IPInterface, Network, and RingNetwork.
"""

# Import all main classes
from .flit import Flit, TokenBucket
from .node import Node
from .ip_interface import IPInterface, RingIPInterface, create_ring_ip_interface
from .network import Network
from .ring_network import RingNetwork

# For backward compatibility, expose all classes at package level
__all__ = [
    'Flit', 
    'TokenBucket', 
    'Node', 
    'IPInterface', 
    'RingIPInterface', 
    'create_ring_ip_interface',
    'Network', 
    'RingNetwork'
]