"""
NoC节点工厂模式实现。

本模块提供节点工厂模式，支持：
- 统一的节点创建接口
- 节点类型注册和管理
- 配置驱动的节点实例化
- 节点依赖注入
- 批量节点创建
"""

from typing import Dict, List, Optional, Any, Type, Callable
from abc import ABC, abstractmethod
import logging

from .node import BaseNoCNode
from .router import RouterNode, RoutingAlgorithm
from .network_interface import NetworkInterface, ProtocolType, QoSClass
from .processing_element import ProcessingElement, WorkloadType, TaskType
from .memory_controller import MemoryController, MemoryType, SchedulingPolicy
from src.noc.utils.types import NodeId, Position, NodeType


class NodeCreationError(Exception):
    """节点创建错误"""

    pass


class NodeConfigurationError(Exception):
    """节点配置错误"""

    pass


class BaseNodeFactory(ABC):
    """节点工厂抽象基类"""

    @abstractmethod
    def create_node(self, node_type: NodeType, node_id: NodeId, position: Position, **kwargs) -> BaseNoCNode:
        """创建节点"""
        pass

    @abstractmethod
    def supports_node_type(self, node_type: NodeType) -> bool:
        """检查是否支持节点类型"""
        pass


class RouterNodeFactory(BaseNodeFactory):
    """路由器节点工厂"""

    def create_node(self, node_type: NodeType, node_id: NodeId, position: Position, **kwargs) -> BaseNoCNode:
        """创建路由器节点"""
        if not self.supports_node_type(node_type):
            raise NodeCreationError(f"不支持的节点类型: {node_type}")

        # 解析路由算法
        routing_alg = kwargs.get("routing_algorithm", "xy")
        if isinstance(routing_alg, str):
            routing_alg = RoutingAlgorithm(routing_alg)
        kwargs.pop("routing_algorithm", None)

        return RouterNode(node_id=node_id, position=position, routing_algorithm=routing_alg, **kwargs)

    def supports_node_type(self, node_type: NodeType) -> bool:
        """检查是否支持路由器节点类型"""
        return node_type == NodeType.ROUTER


class NetworkInterfaceFactory(BaseNodeFactory):
    """网络接口工厂"""

    def create_node(self, node_type: NodeType, node_id: NodeId, position: Position, **kwargs) -> BaseNoCNode:
        """创建网络接口节点"""
        if not self.supports_node_type(node_type):
            raise NodeCreationError(f"不支持的节点类型: {node_type}")

        # 解析协议类型
        protocol = kwargs.get("protocol_type", "memory")
        if isinstance(protocol, str):
            protocol = ProtocolType(protocol)
        kwargs.pop("protocol_type", None)

        return NetworkInterface(node_id=node_id, position=position, protocol_type=protocol, **kwargs)

    def supports_node_type(self, node_type: NodeType) -> bool:
        """检查是否支持网络接口节点类型"""
        return node_type == NodeType.NETWORK_INTERFACE


class ProcessingElementFactory(BaseNodeFactory):
    """处理元素工厂"""

    def create_node(self, node_type: NodeType, node_id: NodeId, position: Position, **kwargs) -> BaseNoCNode:
        """创建处理元素节点"""
        if not self.supports_node_type(node_type):
            raise NodeCreationError(f"不支持的节点类型: {node_type}")

        # 解析工作负载类型
        workload = kwargs.get("workload_type", "synthetic")
        if isinstance(workload, str):
            workload = WorkloadType(workload)
            kwargs["workload_type"] = workload
        return ProcessingElement(node_id=node_id, position=position, **kwargs)

    def supports_node_type(self, node_type: NodeType) -> bool:
        """检查是否支持处理元素节点类型"""
        return node_type == NodeType.PROCESSING_ELEMENT


class MemoryControllerFactory(BaseNodeFactory):
    """内存控制器工厂"""

    def create_node(self, node_type: NodeType, node_id: NodeId, position: Position, **kwargs) -> BaseNoCNode:
        """创建内存控制器节点"""
        if not self.supports_node_type(node_type):
            raise NodeCreationError(f"不支持的节点类型: {node_type}")

        # 解析内存类型
        memory_type = kwargs.get("memory_type", "ddr4")
        if isinstance(memory_type, str):
            memory_type = MemoryType(memory_type)
        kwargs.pop("memory_type", None)

        # 解析调度策略
        scheduling = kwargs.get("scheduling", "frfcfs")
        if isinstance(scheduling, str):
            scheduling = SchedulingPolicy(scheduling)
            kwargs["scheduling"] = scheduling

        return MemoryController(node_id=node_id, position=position, memory_type=memory_type, **kwargs)

    def supports_node_type(self, node_type: NodeType) -> bool:
        """检查是否支持内存控制器节点类型"""
        return node_type == NodeType.MEMORY_CONTROLLER


class NoCNodeFactory:
    """
    NoC节点工厂主类。

    提供统一的节点创建接口，支持：
    1. 多种节点类型的创建
    2. 配置驱动的实例化
    3. 工厂注册和管理
    4. 批量节点创建
    """

    def __init__(self):
        """初始化节点工厂"""
        self.logger = logging.getLogger("NoCNodeFactory")
        if not self.logger.hasHandlers():
            handler = logging.StreamHandler()
            formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
            handler.setFormatter(formatter)
            self.logger.addHandler(handler)
            self.logger.setLevel(logging.INFO)
        self.factories: Dict[NodeType, BaseNodeFactory] = {}
        self.node_configs: Dict[str, Dict[str, Any]] = {}
        self.created_nodes: Dict[NodeId, BaseNoCNode] = {}

        # 注册默认工厂
        self._register_default_factories()

    def _register_default_factories(self) -> None:
        """注册默认工厂"""
        self.register_factory(NodeType.ROUTER, RouterNodeFactory())
        self.register_factory(NodeType.NETWORK_INTERFACE, NetworkInterfaceFactory())
        self.register_factory(NodeType.PROCESSING_ELEMENT, ProcessingElementFactory())
        self.register_factory(NodeType.MEMORY_CONTROLLER, MemoryControllerFactory())

    def register_factory(self, node_type: NodeType, factory: BaseNodeFactory) -> None:
        """
        注册节点工厂。

        Args:
            node_type: 节点类型
            factory: 工厂实例
        """
        self.factories[node_type] = factory
        self.logger.info(f"已注册节点工厂: {node_type.value}")

    def unregister_factory(self, node_type: NodeType) -> None:
        """
        注销节点工厂。

        Args:
            node_type: 节点类型
        """
        if node_type in self.factories:
            del self.factories[node_type]
            self.logger.info(f"已注销节点工厂: {node_type.value}")

    def register_node_config(self, config_name: str, config: Dict[str, Any]) -> None:
        """
        注册节点配置模板。

        Args:
            config_name: 配置名称
            config: 配置字典
        """
        self.node_configs[config_name] = config
        self.logger.info(f"已注册节点配置: {config_name}")

    def create_node(self, node_type: NodeType, node_id: NodeId, position: Position, config_name: Optional[str] = None, **kwargs) -> BaseNoCNode:
        """
        创建节点。

        Args:
            node_type: 节点类型
            node_id: 节点ID
            position: 节点位置
            config_name: 配置模板名称（可选）
            **kwargs: 其他参数

        Returns:
            创建的节点实例

        Raises:
            NodeCreationError: 创建失败时抛出
        """
        try:
            # 检查节点是否已存在
            if node_id in self.created_nodes:
                raise NodeCreationError(f"节点ID {node_id} 已存在")

            # 检查工厂是否存在
            if node_type not in self.factories:
                raise NodeCreationError(f"未注册的节点类型: {node_type}")

            # 合并配置
            final_config = {}
            if config_name and config_name in self.node_configs:
                final_config.update(self.node_configs[config_name])
            final_config.update(kwargs)

            # 验证配置
            self._validate_config(node_type, final_config)

            # 创建节点
            factory = self.factories[node_type]
            node = factory.create_node(node_type, node_id, position, **final_config)

            # 记录创建的节点
            self.created_nodes[node_id] = node

            self.logger.info(f"成功创建节点: {node_type.value}_{node_id} at {position}")

            return node

        except Exception as e:
            error_msg = f"创建节点失败: {node_type.value}_{node_id}, 错误: {e}"
            self.logger.error(error_msg)
            raise NodeCreationError(error_msg) from e

    def _validate_config(self, node_type: NodeType, config: Dict[str, Any]) -> None:
        """
        验证节点配置。

        Args:
            node_type: 节点类型
            config: 配置字典

        Raises:
            NodeConfigurationError: 配置无效时抛出
        """
        # 通用配置验证
        if "input_buffer_size" in config:
            if not isinstance(config["input_buffer_size"], int) or config["input_buffer_size"] <= 0:
                raise NodeConfigurationError("input_buffer_size必须是正整数")

        if "output_buffer_size" in config:
            if not isinstance(config["output_buffer_size"], int) or config["output_buffer_size"] <= 0:
                raise NodeConfigurationError("output_buffer_size必须是正整数")

        # 节点类型特定验证
        if node_type == NodeType.ROUTER:
            self._validate_router_config(config)
        elif node_type == NodeType.NETWORK_INTERFACE:
            self._validate_ni_config(config)
        elif node_type == NodeType.PROCESSING_ELEMENT:
            self._validate_pe_config(config)
        elif node_type == NodeType.MEMORY_CONTROLLER:
            self._validate_mc_config(config)

    def _validate_router_config(self, config: Dict[str, Any]) -> None:
        """验证路由器配置"""
        if "routing_algorithm" in config:
            alg = config["routing_algorithm"]
            if isinstance(alg, str):
                valid_algs = [e.value for e in RoutingAlgorithm]
                if alg not in valid_algs:
                    raise NodeConfigurationError(f"无效的路由算法: {alg}")

    def _validate_ni_config(self, config: Dict[str, Any]) -> None:
        """验证网络接口配置"""
        if "protocol_type" in config:
            protocol = config["protocol_type"]
            if isinstance(protocol, str):
                valid_protocols = [e.value for e in ProtocolType]
                if protocol not in valid_protocols:
                    raise NodeConfigurationError(f"无效的协议类型: {protocol}")

        if "ip_clock_freq" in config:
            freq = config["ip_clock_freq"]
            if not isinstance(freq, (int, float)) or freq <= 0:
                raise NodeConfigurationError("ip_clock_freq必须是正数")

    def _validate_pe_config(self, config: Dict[str, Any]) -> None:
        """验证处理元素配置"""
        if "num_cores" in config:
            cores = config["num_cores"]
            if not isinstance(cores, int) or cores <= 0:
                raise NodeConfigurationError("num_cores必须是正整数")

        if "workload_type" in config:
            workload = config["workload_type"]
            if isinstance(workload, str):
                valid_workloads = [e.value for e in WorkloadType]
                if workload not in valid_workloads:
                    raise NodeConfigurationError(f"无效的工作负载类型: {workload}")

    def _validate_mc_config(self, config: Dict[str, Any]) -> None:
        """验证内存控制器配置"""
        if "memory_capacity" in config:
            capacity = config["memory_capacity"]
            if not isinstance(capacity, int) or capacity <= 0:
                raise NodeConfigurationError("memory_capacity必须是正整数")

        if "memory_type" in config:
            mem_type = config["memory_type"]
            if isinstance(mem_type, str):
                valid_types = [e.value for e in MemoryType]
                if mem_type not in valid_types:
                    raise NodeConfigurationError(f"无效的内存类型: {mem_type}")

    def create_nodes_from_config(self, topology_config: Dict[str, Any]) -> List[BaseNoCNode]:
        """
        从配置文件批量创建节点。

        Args:
            topology_config: 拓扑配置字典

        Returns:
            创建的节点列表
        """
        created_nodes = []

        try:
            nodes_config = topology_config.get("nodes", [])

            for node_config in nodes_config:
                node_type_str = node_config.get("type")
                node_id = node_config.get("id")
                position = tuple(node_config.get("position", (0, 0)))
                config_name = node_config.get("config_template")

                # 验证必需字段
                if not all([node_type_str, node_id is not None]):
                    raise NodeConfigurationError("节点配置缺少必需字段: type, id")

                # 转换节点类型
                node_type = NodeType(node_type_str)

                # 提取其他配置参数
                other_params = {k: v for k, v in node_config.items() if k not in ["type", "id", "position", "config_template"]}

                # 创建节点
                node = self.create_node(node_type=node_type, node_id=node_id, position=position, config_name=config_name, **other_params)

                created_nodes.append(node)

            self.logger.info(f"成功批量创建 {len(created_nodes)} 个节点")

            return created_nodes

        except Exception as e:
            self.logger.error(f"批量创建节点失败: {e}")
            raise NodeCreationError(f"批量创建节点失败: {e}") from e

    def create_mesh_topology(self, width: int, height: int, node_types: Optional[Dict[str, NodeType]] = None) -> List[BaseNoCNode]:
        """
        创建网格拓扑。

        Args:
            width: 网格宽度
            height: 网格高度
            node_types: 节点类型映射（可选）

        Returns:
            创建的节点列表
        """
        nodes = []

        # 默认节点类型分配
        if node_types is None:
            node_types = {"router": NodeType.ROUTER, "pe": NodeType.PROCESSING_ELEMENT, "mc": NodeType.MEMORY_CONTROLLER}

        try:
            for y in range(height):
                for x in range(width):
                    node_id = y * width + x
                    position = (x, y)

                    # 确定节点类型
                    if node_id == 0:
                        # 节点0作为内存控制器
                        node_type = node_types.get("mc", NodeType.MEMORY_CONTROLLER)
                        config_name = "default_mc"
                    elif (x + y) % 2 == 0:
                        # 棋盘模式放置路由器
                        node_type = node_types.get("router", NodeType.ROUTER)
                        config_name = "default_router"
                    else:
                        # 其他位置放置处理元素
                        node_type = node_types.get("pe", NodeType.PROCESSING_ELEMENT)
                        config_name = "default_pe"

                    # 创建节点
                    node = self.create_node(node_type=node_type, node_id=node_id, position=position, config_name=config_name, mesh_width=width, mesh_height=height)

                    nodes.append(node)

            self.logger.info(f"成功创建 {width}x{height} 网格拓扑，共 {len(nodes)} 个节点")

            return nodes

        except Exception as e:
            self.logger.error(f"创建网格拓扑失败: {e}")
            raise NodeCreationError(f"创建网格拓扑失败: {e}") from e

    def get_node(self, node_id: NodeId) -> Optional[BaseNoCNode]:
        """
        获取已创建的节点。

        Args:
            node_id: 节点ID

        Returns:
            节点实例，如果不存在则返回None
        """
        return self.created_nodes.get(node_id)

    def get_nodes_by_type(self, node_type: NodeType) -> List[BaseNoCNode]:
        """
        按类型获取节点。

        Args:
            node_type: 节点类型

        Returns:
            指定类型的节点列表
        """
        return [node for node in self.created_nodes.values() if node.node_type == node_type]

    def get_all_nodes(self) -> List[BaseNoCNode]:
        """
        获取所有已创建的节点。

        Returns:
            所有节点的列表
        """
        return list(self.created_nodes.values())

    def clear_nodes(self) -> None:
        """清除所有已创建的节点"""
        self.created_nodes.clear()
        self.logger.info("已清除所有节点")

    def get_factory_info(self) -> Dict[str, Any]:
        """
        获取工厂信息。

        Returns:
            工厂信息字典
        """
        return {
            "registered_factories": [node_type.value for node_type in self.factories.keys()],
            "registered_configs": list(self.node_configs.keys()),
            "created_nodes": len(self.created_nodes),
            "nodes_by_type": {node_type.value: len(self.get_nodes_by_type(node_type)) for node_type in NodeType},
        }


# 全局工厂实例
default_node_factory = NoCNodeFactory()

# 注册默认配置
default_node_factory.register_node_config("default_router", {"routing_algorithm": "xy", "input_buffer_size": 8, "output_buffer_size": 8, "virtual_channels": 2})

default_node_factory.register_node_config("default_pe", {"num_cores": 1, "workload_type": "synthetic", "clock_frequency": 1.0, "cache_simulation": True, "l1_cache_size": 32, "l2_cache_size": 256})

default_node_factory.register_node_config("default_mc", {"memory_type": "ddr4", "memory_capacity": 16 * 1024 * 1024 * 1024, "channels": 4, "scheduling": "frfcfs", "ecc": True})

default_node_factory.register_node_config("default_ni", {"protocol_type": "memory", "ip_clock_freq": 1.0, "network_clock_freq": 2.0, "qos_enabled": True, "packet_size": 64})


def create_node(node_type: NodeType, node_id: NodeId, position: Position, **kwargs) -> BaseNoCNode:
    """
    便捷函数：创建节点。

    Args:
        node_type: 节点类型
        node_id: 节点ID
        position: 节点位置
        **kwargs: 其他参数

    Returns:
        创建的节点实例
    """
    return default_node_factory.create_node(node_type, node_id, position, **kwargs)


def create_mesh_topology(width: int, height: int) -> List[BaseNoCNode]:
    """
    便捷函数：创建网格拓扑。

    Args:
        width: 网格宽度
        height: 网格高度

    Returns:
        创建的节点列表
    """
    return default_node_factory.create_mesh_topology(width, height)
