"""
NoC拓扑工厂类。

本模块负责根据配置创建相应的拓扑实例，提供统一的创建接口，
支持不同类型的NoC拓扑结构。
"""

from typing import Type, Dict, Any, Optional, List, Callable
import logging
from abc import ABC, abstractmethod

from .types import TopologyType, ValidationResult, ConfigDict
from ..base.topology import BaseNoCTopology
from ..base.config import BaseNoCConfig


class TopologyRegistryError(Exception):
    """拓扑注册相关异常。"""

    pass


class TopologyCreationError(Exception):
    """拓扑创建相关异常。"""

    pass


class TopologyValidator(ABC):
    """拓扑验证器抽象基类。"""

    @abstractmethod
    def validate(self, config: BaseNoCConfig) -> ValidationResult:
        """
        验证配置是否适用于特定拓扑。

        Args:
            config: 配置对象

        Returns:
            验证结果
        """
        pass


class MeshTopologyValidator(TopologyValidator):
    """Mesh拓扑验证器。"""

    def validate(self, config: BaseNoCConfig) -> ValidationResult:
        """验证Mesh拓扑配置。"""
        if config.num_nodes <= 0:
            return False, "节点数必须为正数"

        # 检查是否为完全平方数（对于规则Mesh）
        import math

        sqrt_nodes = int(math.sqrt(config.num_nodes))
        if sqrt_nodes * sqrt_nodes != config.num_nodes:
            return False, f"Mesh拓扑节点数{config.num_nodes}不是完全平方数"

        return True, None


class RingTopologyValidator(TopologyValidator):
    """Ring拓扑验证器。"""

    def validate(self, config: BaseNoCConfig) -> ValidationResult:
        """验证Ring拓扑配置。"""
        if config.num_nodes < 3:
            return False, "Ring拓扑至少需要3个节点"

        return True, None


class CrossRingTopologyValidator(TopologyValidator):
    """CrossRing拓扑验证器。"""

    def validate(self, config: BaseNoCConfig) -> ValidationResult:
        """验证CrossRing拓扑配置。"""

        # 验证CrossRing特定参数
        if config.NUM_NODE != config.NUM_ROW * config.NUM_COL:
            return False, "NUM_NODE必须等于NUM_ROW * NUM_COL"

        if config.NUM_IP > config.NUM_NODE:
            return False, "IP数量不能超过节点数"

        return True, None


class NoCTopologyFactory:
    """
    NoC拓扑工厂类。

    负责根据配置创建相应的拓扑实例，管理拓扑类型注册，
    并提供验证和优化功能。
    """

    # 拓扑类型注册表
    _topology_registry: Dict[TopologyType, Type[BaseNoCTopology]] = {}

    # 配置创建器注册表
    _config_creators: Dict[TopologyType, Callable[..., BaseNoCConfig]] = {}

    # 验证器注册表
    _validators: Dict[TopologyType, TopologyValidator] = {
        TopologyType.MESH: MeshTopologyValidator(),
        TopologyType.RING: RingTopologyValidator(),
        TopologyType.CROSSRING: CrossRingTopologyValidator(),
    }

    # 默认配置参数
    _default_configs: Dict[TopologyType, ConfigDict] = {
        TopologyType.MESH: {"num_nodes": 16, "routing_strategy": "shortest", "buffer_depth": 8, "link_bandwidth": 256.0},
        TopologyType.RING: {"num_nodes": 8, "routing_strategy": "load_balanced", "buffer_depth": 4, "link_bandwidth": 256.0},
        TopologyType.CROSSRING: {"NUM_NODE": 20, "NUM_COL": 2, "NUM_IP": 16, "buffer_depth": 8, "link_bandwidth": 256.0},
    }

    @classmethod
    def register_topology(
        cls,
        topology_type: TopologyType,
        topology_class: Type[BaseNoCTopology],
        config_creator: Optional[Callable[..., BaseNoCConfig]] = None,
        validator: Optional[TopologyValidator] = None,
    ) -> None:
        """
        注册新的拓扑类型。

        Args:
            topology_type: 拓扑类型
            topology_class: 拓扑实现类
            config_creator: 配置创建函数（可选）
            validator: 验证器（可选）
        """
        if not issubclass(topology_class, BaseNoCTopology):
            raise TopologyRegistryError(f"拓扑类{topology_class}必须继承自BaseNoCTopology")

        cls._topology_registry[topology_type] = topology_class

        if config_creator is not None:
            cls._config_creators[topology_type] = config_creator

        if validator is not None:
            cls._validators[topology_type] = validator

        logging.info(f"已注册拓扑类型: {topology_type.value}")

    @classmethod
    def unregister_topology(cls, topology_type: TopologyType) -> None:
        """
        注销拓扑类型。

        Args:
            topology_type: 要注销的拓扑类型
        """
        if topology_type in cls._topology_registry:
            del cls._topology_registry[topology_type]

        if topology_type in cls._config_creators:
            del cls._config_creators[topology_type]

        if topology_type in cls._validators:
            del cls._validators[topology_type]

        logging.info(f"已注销拓扑类型: {topology_type.value}")

    @classmethod
    def create_topology(cls, config: BaseNoCConfig, validate: bool = True) -> BaseNoCTopology:
        """
        根据配置创建拓扑实例。

        Args:
            config: NoC配置对象
            validate: 是否验证配置

        Returns:
            拓扑实例

        Raises:
            TopologyCreationError: 创建失败时抛出
        """
        topology_type = config.topology_type

        # 检查是否支持该拓扑类型
        if topology_type not in cls._topology_registry:
            supported_types = list(cls._topology_registry.keys())
            raise TopologyCreationError(f"不支持的拓扑类型: {topology_type}. " f"支持的类型: {[t.value for t in supported_types]}")

        # 验证配置
        if validate:
            is_valid, error_msg = cls.validate_config(config)
            if not is_valid:
                raise TopologyCreationError(f"配置验证失败: {error_msg}")

        # 创建拓扑实例
        topology_class = cls._topology_registry[topology_type]

        try:
            topology = topology_class(config)
            logging.info(f"成功创建{topology_type.value}拓扑，节点数: {config.num_nodes}")
            return topology
        except Exception as e:
            raise TopologyCreationError(f"创建拓扑失败: {str(e)}") from e

    @classmethod
    def create_config(cls, topology_type: TopologyType, **kwargs) -> BaseNoCConfig:
        """
        创建指定拓扑类型的配置。

        Args:
            topology_type: 拓扑类型
            **kwargs: 配置参数

        Returns:
            配置对象
        """
        # 使用注册的配置创建器
        if topology_type in cls._config_creators:
            return cls._config_creators[topology_type](**kwargs)

        # 使用默认配置创建器
        config = BaseNoCConfig(topology_type)

        # 应用默认配置
        if topology_type in cls._default_configs:
            default_params = cls._default_configs[topology_type]
            for key, value in default_params.items():
                config.set_parameter(key, value)

        # 应用用户参数
        for key, value in kwargs.items():
            config.set_parameter(key, value)

        return config

    @classmethod
    def create_topology_from_type(cls, topology_type: TopologyType, **kwargs) -> BaseNoCTopology:
        """
        根据拓扑类型和参数创建拓扑。

        Args:
            topology_type: 拓扑类型
            **kwargs: 配置参数

        Returns:
            拓扑实例
        """
        config = cls.create_config(topology_type, **kwargs)
        return cls.create_topology(config)

    @classmethod
    def validate_config(cls, config: BaseNoCConfig) -> ValidationResult:
        """
        验证配置。

        Args:
            config: 配置对象

        Returns:
            验证结果
        """
        # 基础验证
        basic_valid, basic_error = config.validate_basic_params()
        if not basic_valid:
            return basic_valid, basic_error

        # 拓扑特定验证
        topology_type = config.topology_type
        if topology_type in cls._validators:
            validator = cls._validators[topology_type]
            return validator.validate(config)

        # 默认验证通过
        return True, None

    @classmethod
    def get_supported_topologies(cls) -> List[TopologyType]:
        """
        获取支持的拓扑类型列表。

        Returns:
            支持的拓扑类型列表
        """
        return list(cls._topology_registry.keys())

    @classmethod
    def is_topology_supported(cls, topology_type: TopologyType) -> bool:
        """
        检查是否支持指定的拓扑类型。

        Args:
            topology_type: 拓扑类型

        Returns:
            是否支持
        """
        return topology_type in cls._topology_registry

    @classmethod
    def get_topology_info(cls, topology_type: TopologyType) -> Dict[str, Any]:
        """
        获取拓扑类型信息。

        Args:
            topology_type: 拓扑类型

        Returns:
            拓扑信息字典
        """
        if topology_type not in cls._topology_registry:
            return {"supported": False}

        topology_class = cls._topology_registry[topology_type]
        default_config = cls._default_configs.get(topology_type, {})

        return {
            "supported": True,
            "class_name": topology_class.__name__,
            "module": topology_class.__module__,
            "default_config": default_config,
            "has_validator": topology_type in cls._validators,
            "has_config_creator": topology_type in cls._config_creators,
        }

    @classmethod
    def list_topologies(cls) -> Dict[str, Dict[str, Any]]:
        """
        列出所有注册的拓扑类型及其信息。

        Returns:
            拓扑信息字典
        """
        return {topology_type.value: cls.get_topology_info(topology_type) for topology_type in cls._topology_registry}

    @classmethod
    def clone_topology(cls, original: BaseNoCTopology, config_updates: Optional[Dict[str, Any]] = None) -> BaseNoCTopology:
        """
        克隆现有拓扑。

        Args:
            original: 原始拓扑
            config_updates: 配置更新（可选）

        Returns:
            克隆的拓扑实例
        """
        # 复制配置
        new_config = original.config.copy()

        # 应用更新
        if config_updates:
            for key, value in config_updates.items():
                new_config.set_parameter(key, value)

        # 创建新拓扑
        return cls.create_topology(new_config)

    @classmethod
    def optimize_topology(cls, config: BaseNoCConfig, optimization_target: str = "latency") -> BaseNoCConfig:
        """
        优化拓扑配置。

        Args:
            config: 原始配置
            optimization_target: 优化目标

        Returns:
            优化后的配置
        """
        optimized_config = config.copy()

        if optimization_target == "latency":
            # 为延迟优化
            optimized_config.routing_strategy = "shortest"
            optimized_config.buffer_depth = min(optimized_config.buffer_depth, 4)
        elif optimization_target == "throughput":
            # 为吞吐量优化
            optimized_config.routing_strategy = "load_balanced"
            optimized_config.buffer_depth = max(optimized_config.buffer_depth, 8)
            optimized_config.virtual_channels = max(getattr(optimized_config, "virtual_channels", 2), 4)
        elif optimization_target == "power":
            # 为功耗优化
            optimized_config.enable_power_management = True
            optimized_config.buffer_depth = min(optimized_config.buffer_depth, 4)

        return optimized_config

    @classmethod
    def compare_topologies(cls, configs: List[BaseNoCConfig]) -> Dict[str, Any]:
        """
        比较多个拓扑配置。

        Args:
            configs: 配置列表

        Returns:
            比较结果
        """
        comparison = {"configs": [], "metrics": {"node_count": [], "estimated_diameter": [], "estimated_latency": [], "estimated_throughput": []}}

        for i, config in enumerate(configs):
            config_info = {
                "index": i,
                "topology_type": config.topology_type.value,
                "num_nodes": config.num_nodes,
                "routing_strategy": config.routing_strategy.value if hasattr(config.routing_strategy, "value") else str(config.routing_strategy),
            }
            comparison["configs"].append(config_info)

            # 添加估算指标
            comparison["metrics"]["node_count"].append(config.num_nodes)

            # 简单的性能估算
            if config.topology_type == TopologyType.MESH:
                import math

                side_length = int(math.sqrt(config.num_nodes))
                estimated_diameter = 2 * (side_length - 1)
            elif config.topology_type == TopologyType.RING:
                estimated_diameter = config.num_nodes // 2
            else:
                estimated_diameter = config.num_nodes  # 保守估算

            comparison["metrics"]["estimated_diameter"].append(estimated_diameter)
            comparison["metrics"]["estimated_latency"].append(estimated_diameter * 2)
            comparison["metrics"]["estimated_throughput"].append(config.link_bandwidth * config.num_nodes)

        return comparison

    @classmethod
    def get_factory_stats(cls) -> Dict[str, Any]:
        """
        获取工厂统计信息。

        Returns:
            统计信息
        """
        return {
            "registered_topologies": len(cls._topology_registry),
            "supported_types": [t.value for t in cls._topology_registry.keys()],
            "validators_count": len(cls._validators),
            "config_creators_count": len(cls._config_creators),
            "default_configs_count": len(cls._default_configs),
        }


# 便捷函数
def create_mesh_topology(width: int, height: int, **kwargs) -> BaseNoCTopology:
    """
    创建Mesh拓扑的便捷函数。

    Args:
        width: 宽度
        height: 高度
        **kwargs: 其他配置参数

    Returns:
        Mesh拓扑实例
    """
    num_nodes = width * height
    return NoCTopologyFactory.create_topology_from_type(TopologyType.MESH, num_nodes=num_nodes, **kwargs)


def create_ring_topology(num_nodes: int, **kwargs) -> BaseNoCTopology:
    """
    创建Ring拓扑的便捷函数。

    Args:
        num_nodes: 节点数
        **kwargs: 其他配置参数

    Returns:
        Ring拓扑实例
    """
    return NoCTopologyFactory.create_topology_from_type(TopologyType.RING, num_nodes=num_nodes, **kwargs)


def create_crossring_topology(num_node: int = 20, num_col: int = 2, num_ip: int = 16, **kwargs) -> BaseNoCTopology:
    """
    创建CrossRing拓扑的便捷函数。

    Args:
        num_node: 节点总数
        num_col: 列数
        num_ip: IP数量
        **kwargs: 其他配置参数

    Returns:
        CrossRing拓扑实例
    """
    return NoCTopologyFactory.create_topology_from_type(TopologyType.CROSSRING, NUM_NODE=num_node, NUM_COL=num_col, NUM_IP=num_ip, **kwargs)


def auto_select_topology(requirements: Dict[str, Any]) -> TopologyType:
    """
    根据需求自动选择合适的拓扑类型。

    Args:
        requirements: 需求字典

    Returns:
        推荐的拓扑类型
    """
    num_nodes = requirements.get("num_nodes", 16)
    latency_critical = requirements.get("latency_critical", False)
    throughput_critical = requirements.get("throughput_critical", False)
    fault_tolerance = requirements.get("fault_tolerance", False)

    # 简单的选择逻辑
    if latency_critical and num_nodes <= 16:
        return TopologyType.MESH
    elif throughput_critical:
        return TopologyType.CROSSRING
    elif fault_tolerance:
        return TopologyType.TORUS
    elif num_nodes <= 8:
        return TopologyType.RING
    else:
        return TopologyType.MESH


# 注册默认拓扑类型
def register_default_topologies():
    """注册默认的拓扑类型。"""
    # 注意：这里只是占位符，实际的拓扑实现类需要在具体实现后注册
    # 例如：
    # from .topologies.mesh import MeshTopology
    # from .topologies.ring import RingTopology
    # from .topologies.crossring import CrossRingTopology
    #
    # NoCTopologyFactory.register_topology(TopologyType.MESH, MeshTopology)
    # NoCTopologyFactory.register_topology(TopologyType.RING, RingTopology)
    # NoCTopologyFactory.register_topology(TopologyType.CROSSRING, CrossRingTopology)

    logging.info("默认拓扑类型将在实现类可用时注册")


# 模块初始化时注册默认拓扑
register_default_topologies()
