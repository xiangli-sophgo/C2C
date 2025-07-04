"""
NoC（Network-on-Chip）抽象层。

本模块提供统一的NoC拓扑抽象接口，支持不同类型的片内网络拓扑结构，
包括CrossRing、Mesh、Ring等。为后续集成和扩展提供基础框架。

主要组件：
- types: 类型定义和枚举
- base: 基础抽象类
- factory: 拓扑工厂类

使用示例：
    from src.noc import NoCTopologyFactory, TopologyType

    # 创建CrossRing拓扑
    topology = NoCTopologyFactory.create_topology_from_type(
        TopologyType.CROSSRING,
        NUM_NODE=20,
        NUM_COL=2
    )

    # 创建Mesh拓扑
    mesh = NoCTopologyFactory.create_topology_from_type(
        TopologyType.MESH,
        num_nodes=16
    )
"""

from .utils.types import (
    # 枚举类型
    TopologyType,
    RoutingStrategy,
    FlowControlType,
    BufferType,
    TrafficPattern,
    NodeType,
    LinkType,
    Priority,
    SimulationState,
    EventType,
    # 数据类
    NoCMetrics,
    NoCConfiguration,
    LinkMetrics,
    NodeMetrics,
    TrafficFlow,
    SimulationEvent,
    QoSRequirement,
    FaultModel,
    # 类型别名
    NodeId,
    Position,
    Path,
    AdjacencyMatrix,
    LinkId,
    Coordinate,
    Coordinate3D,
    ConfigDict,
    MetricsDict,
    ValidationResult,
    # 常量
    DEFAULT_FLIT_SIZE,
    DEFAULT_PACKET_SIZE,
    DEFAULT_BUFFER_SIZE,
    DEFAULT_LINK_BANDWIDTH,
    DEFAULT_CLOCK_FREQUENCY,
)

from .base.config import (
    BaseNoCConfig,
    create_config_from_type,
)

from .base.topology import (
    BaseNoCTopology,
)

from .base.node import (
    BaseNoCNode,
    NodeState,
    BufferStatus,
)

from .utils.factory import (
    NoCTopologyFactory,
    TopologyRegistryError,
    TopologyCreationError,
    TopologyValidator,
    # 便捷函数
    create_mesh_topology,
    create_ring_topology,
    create_crossring_topology,
    auto_select_topology,
)

# 版本信息
__version__ = "1.0.0"
__author__ = "NoC Development Team"
__description__ = "NoC抽象层 - 统一的片内网络拓扑框架"


# 模块级别的便捷函数
def get_supported_topologies():
    """
    获取支持的拓扑类型列表。

    Returns:
        List[TopologyType]: 支持的拓扑类型
    """
    return NoCTopologyFactory.get_supported_topologies()


def create_topology(topology_type, **kwargs):
    """
    创建拓扑的便捷函数。

    Args:
        topology_type: 拓扑类型（字符串或TopologyType枚举）
        **kwargs: 配置参数

    Returns:
        BaseNoCTopology: 拓扑实例
    """
    if isinstance(topology_type, str):
        # 字符串转换为枚举
        type_map = {
            "crossring": TopologyType.CROSSRING,
            "mesh": TopologyType.MESH,
            "ring": TopologyType.RING,
            "torus": TopologyType.TORUS,
            "tree": TopologyType.TREE,
            "fat_tree": TopologyType.FAT_TREE,
            "butterfly": TopologyType.BUTTERFLY,
            "dragonfly": TopologyType.DRAGONFLY,
        }
        topology_type = type_map.get(topology_type.lower())
        if topology_type is None:
            raise ValueError(f"不支持的拓扑类型: {topology_type}")

    return NoCTopologyFactory.create_topology_from_type(topology_type, **kwargs)


def validate_config(config):
    """
    验证配置的便捷函数。

    Args:
        config: 配置对象

    Returns:
        ValidationResult: 验证结果
    """
    return NoCTopologyFactory.validate_config(config)


def list_topologies():
    """
    列出所有注册的拓扑类型。

    Returns:
        Dict: 拓扑信息字典
    """
    return NoCTopologyFactory.list_topologies()


# 默认导出列表
__all__ = [
    # 类型和枚举
    "TopologyType",
    "RoutingStrategy",
    "FlowControlType",
    "BufferType",
    "TrafficPattern",
    "NodeType",
    "LinkType",
    "Priority",
    "SimulationState",
    "EventType",
    # 数据类
    "NoCMetrics",
    "NoCConfiguration",
    "LinkMetrics",
    "NodeMetrics",
    "TrafficFlow",
    "SimulationEvent",
    "QoSRequirement",
    "FaultModel",
    # 类型别名
    "NodeId",
    "Position",
    "Path",
    "AdjacencyMatrix",
    "LinkId",
    "Coordinate",
    "Coordinate3D",
    "ConfigDict",
    "MetricsDict",
    "ValidationResult",
    # 基础类
    "BaseNoCConfig",
    "CrossRingCompatibleConfig",
    "BaseNoCTopology",
    "BaseNoCNode",
    "NodeState",
    "BufferStatus",
    # 工厂类
    "NoCTopologyFactory",
    "TopologyRegistryError",
    "TopologyCreationError",
    "TopologyValidator",
    # 便捷函数
    "create_topology",
    "create_mesh_topology",
    "create_ring_topology",
    "create_crossring_topology",
    "auto_select_topology",
    "create_config_from_type",
    "get_supported_topologies",
    "validate_config",
    "list_topologies",
    # 常量
    "DEFAULT_FLIT_SIZE",
    "DEFAULT_PACKET_SIZE",
    "DEFAULT_BUFFER_SIZE",
    "DEFAULT_LINK_BANDWIDTH",
    "DEFAULT_CLOCK_FREQUENCY",
]


# 模块信息
def get_module_info():
    """
    获取模块信息。

    Returns:
        Dict: 模块信息
    """
    return {
        "name": "noc",
        "version": __version__,
        "author": __author__,
        "description": __description__,
        "supported_topologies": len(get_supported_topologies()),
        "components": ["types", "base.config", "base.topology", "base.node", "factory"],
    }


# 设置日志
import logging

# 配置NoC模块的日志记录器
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# 如果没有处理器，添加控制台处理器
if not logger.handlers:
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

logger.info(f"NoC抽象层已加载 - 版本 {__version__}")
logger.info(f"支持的拓扑类型: {[t.value for t in get_supported_topologies()]}")
