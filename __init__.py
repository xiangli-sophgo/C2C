# -*- coding: utf-8 -*-
"""
C2C Topology Modeling Framework

A comprehensive framework for modeling SG2260E chip-to-chip communication topologies.
Supports various interconnect topologies including cascade, switch-based, tree, and torus configurations.
"""

__version__ = "1.0.0"
__author__ = "C2C Development Team"
__email__ = "contact@c2c-topology.com"
__description__ = "A topology modeling framework for SG2260E chip-to-chip communication"

# Import core components
from src.c2c.topology.base import BaseNode, BaseLink, BaseTopology
from src.c2c.topology.node import ChipNode, SwitchNode, HostNode
from src.c2c.topology.link import PCIeLink, C2CDirectLink
from src.c2c.topology.graph import TopologyGraph
from src.c2c.topology.builder import TopologyBuilder

# Import protocol components
from src.c2c.protocol.base import BaseProtocol, ProtocolState
from src.c2c.protocol.cdma_system import CDMAProtocol, CDMAMessageType
from src.c2c.protocol.credit import CreditManager
from src.c2c.protocol.address import AddressTranslator
from src.c2c.protocol.router import Router

# Import visualization components
from src.visualization.visualizer import TopologyVisualizer
from src.visualization.comparison import PerformanceComparator

# Import utility components
from config.constants import *
from src.c2c.utils.exceptions import *

# Import configuration
from config.loader import ConfigLoader

__all__ = [
    # Version info
    "__version__",
    "__author__",
    "__email__",
    "__description__",
    # Core topology classes
    "BaseNode",
    "BaseLink",
    "BaseTopology",
    "ChipNode",
    "SwitchNode",
    "HostNode",
    "PCIeLink",
    "C2CDirectLink",
    "TopologyGraph",
    "TopologyBuilder",
    # Protocol classes
    "BaseProtocol",
    "ProtocolState",
    "CDMAProtocol",
    "CDMAMessageType",
    "CreditManager",
    "AddressTranslator",
    "Router",
    # Visualization classes
    "TopologyVisualizer",
    "PerformanceComparator",
    # Utility classes
    "ProtocolError",
    "ConfigLoader",
    # Constants
    "DEFAULT_BANDWIDTH",
    "DEFAULT_LATENCY",
    "PROTOCOL_VERSIONS",
    "MESSAGE_TYPES",
    "NODE_TYPES",
    "LINK_TYPES",
]

# Package metadata
__pkg_info__ = {
    "name": "C2C",
    "version": __version__,
    "description": __description__,
    "author": __author__,
    "author_email": __email__,
    "license": "MIT",
    "keywords": ["topology", "chip-to-chip", "communication", "modeling", "simulation"],
}
