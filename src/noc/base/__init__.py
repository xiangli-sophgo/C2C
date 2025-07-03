"""
NoC基础抽象类模块。

本模块包含所有NoC组件的基础抽象类，提供统一的接口定义：
- config: 配置管理基类
- topology: 拓扑结构基类
- node: 节点功能基类
- router: 路由器基类（待实现）
- link: 链路基类（待实现）
"""

from .config import (
    BaseNoCConfig,
    create_config_from_type,
)

from .topology import (
    BaseNoCTopology,
)

from .node import (
    BaseNoCNode,
    NodeState,
    BufferStatus,
)

__all__ = [
    # 配置类
    "BaseNoCConfig",
    "CrossRingCompatibleConfig",
    "create_config_from_type",
    # 拓扑类
    "BaseNoCTopology",
    # 节点类
    "BaseNoCNode",
    "NodeState",
    "BufferStatus",
]
