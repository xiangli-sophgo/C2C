"""
C2C 模块 - 芯片间通信核心组件
包含拓扑建模、协议实现和工具类
"""

# 导入子模块
from . import topology
from . import protocol
from . import utils

# 导入主要类
from .topology.base import BaseNode, BaseLink, BaseTopology
from .topology.node import ChipNode, SwitchNode, HostNode
from .topology.link import PCIeLink, C2CDirectLink
from .topology.graph import TopologyGraph
from .topology.builder import TopologyBuilder

from .protocol.base import BaseProtocol, ProtocolState
from .protocol.cdma_system import CDMASystem, CDMASystemState, CDMAMessage
from .protocol.credit import CreditManager
from .protocol.address import AddressTranslator
from .protocol.router import Router

from .utils.exceptions import *

__version__ = "1.0.0"
__author__ = "C2C Development Team"

__all__ = [
    # 子模块
    "topology",
    "protocol",
    "utils",
    # 拓扑类
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
    # 协议类
    "BaseProtocol",
    "ProtocolState",
    "CDMASystem",
    "CDMASystemState",
    "CDMAMessage",
    "CreditManager",
    "AddressTranslator",
    "Router",
    # 异常类
    "C2CException",
    "TopologyError",
    "ProtocolError",
    "CDMAError",
    "AddressError",
    "ShapeCompatibilityError",
]
