"""
C2C 拓扑模块 - 芯片间通信拓扑建模
包含基础拓扑类、节点、链路和构建器
"""

# 导入基础类
from .base import BaseNode, BaseLink, BaseTopology

# 导入节点类
from .node import ChipNode, SwitchNode, HostNode

# 导入链路类
from .link import PCIeLink, C2CDirectLink

# 导入图结构
from .graph import TopologyGraph

# 导入构建器
from .builder import TopologyBuilder

# 导入拓扑逻辑（如果存在）
try:
    from .tree import TreeTopologyLogic, TreeAddressRoutingLogic, TreeConfigGenerationLogic, TreeFaultToleranceLogic
    from .torus import TorusTopologyLogic, TorusRoutingLogic
    from .topology_optimizer import TopologyOptimizer, ApplicationRequirements
except ImportError:
    # 如果某些模块不存在，忽略错误
    pass

__version__ = "1.2.0"

__all__ = [
    # 基础类
    "BaseNode",
    "BaseLink",
    "BaseTopology",
    # 节点类
    "ChipNode",
    "SwitchNode",
    "HostNode",
    # 链路类
    "PCIeLink",
    "C2CDirectLink",
    # 图结构
    "TopologyGraph",
    # 构建器
    "TopologyBuilder",
    # 拓扑逻辑（可选）
    "TreeTopologyLogic",
    "TorusTopologyLogic",
    "TopologyOptimizer",
]
