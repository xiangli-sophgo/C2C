"""
NoC配置基类。

本模块提供可被特定NoC拓扑实现扩展的基础配置类。
确保与现有CrossRing配置的兼容性，同时提供统一的接口。
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, List, Optional, Tuple, Union
import copy
import json
import os
from pathlib import Path

from src.noc.utils.types import TopologyType, RoutingStrategy, FlowControlType, BufferType, TrafficPattern, NoCConfiguration, ConfigDict, ValidationResult
from typing import Type


class BaseNoCConfig(ABC):
    """
    NoC配置的抽象基类。

    该类定义了所有NoC拓扑配置必须实现的通用接口。
    提供基本的参数验证和配置管理功能。
    """

    def __init__(self, topology_type: TopologyType = TopologyType.MESH):
        """
        初始化基础NoC配置。

        Args:
            topology_type: NoC拓扑类型
        """
        # 核心拓扑参数
        self.topology_type = topology_type
        self.num_nodes = 16
        self.routing_strategy = RoutingStrategy.SHORTEST

        # 性能参数
        self.flit_size = 128  # bits
        self.packet_size = 512  # bits
        self.buffer_depth = 8  # flits
        self.link_bandwidth = 256.0  # GB/s
        self.link_latency = 1  # cycles

        # Tag参数
        self.flow_control = FlowControlType.WORMHOLE
        self.buffer_type = BufferType.SHARED

        # 仿真参数
        self.simulation_cycles = 10000
        self.warmup_cycles = 0
        self.stats_start_cycle = 0

        # 时钟和时序
        self.clock_frequency = 1.0  # GHz
        self.network_frequency = 2.0  # Relative to system clock

        # 流量参数
        self.traffic_pattern = TrafficPattern.UNIFORM_RANDOM
        self.injection_rate = 0.1  # flits per cycle per node

        # 自定义参数存储
        self._custom_params = {}

        # 预设配置存储
        self._presets = {}

        # 初始化拓扑特定参数
        self._init_topology_params()

    def _init_topology_params(self):
        """初始化拓扑特定参数。在子类中重写。"""
        pass

    @abstractmethod
    def validate_config(self) -> ValidationResult:
        """
        验证配置参数。

        Returns:
            返回(是否有效, 错误信息)的元组
        """
        pass

    @abstractmethod
    def get_topology_params(self) -> Dict[str, Any]:
        """
        获取拓扑特定参数。

        Returns:
            包含拓扑特定配置参数的字典
        """
        pass

    def set_parameter(self, key: str, value: Any) -> bool:
        """
        设置配置参数。

        Args:
            key: 参数名称
            value: 参数值

        Returns:
            如果参数设置成功返回True，否则返回False
        """
        if hasattr(self, key):
            # 特殊处理枚举类型
            if key == "topology_type" and isinstance(value, str):
                # 转换字符串回枚举
                try:
                    value = TopologyType(value)
                except ValueError:
                    pass  # 如果转换失败，保持原值
            elif key == "routing_strategy" and isinstance(value, str):
                try:
                    value = RoutingStrategy(value)
                except ValueError:
                    pass
            elif key == "flow_control" and isinstance(value, str):
                try:
                    value = FlowControlType(value)
                except ValueError:
                    pass
            elif key == "buffer_type" and isinstance(value, str):
                try:
                    value = BufferType(value)
                except ValueError:
                    pass
            elif key == "traffic_pattern" and isinstance(value, str):
                try:
                    value = TrafficPattern(value)
                except ValueError:
                    pass

            setattr(self, key, value)
            return True
        else:
            # 存储到自定义参数中
            self._custom_params[key] = value
            return True

    def get_parameter(self, key: str, default: Any = None) -> Any:
        """
        获取配置参数。

        Args:
            key: 参数名称
            default: 如果参数未找到时的默认值

        Returns:
            参数值或默认值
        """
        if hasattr(self, key):
            return getattr(self, key)
        return self._custom_params.get(key, default)

    def to_dict(self) -> ConfigDict:
        """
        Convert configuration to dictionary.

        Returns:
            Dictionary representation of the configuration
        """
        config_dict = {}

        # Add all public attributes
        for key, value in self.__dict__.items():
            if not key.startswith("_"):
                if hasattr(value, "value"):  # Handle Enum types
                    config_dict[key] = value.value
                else:
                    config_dict[key] = value

        # Add custom parameters
        config_dict.update(self._custom_params)

        # Add topology-specific parameters
        config_dict.update(self.get_topology_params())

        return config_dict

    def from_dict(self, config_dict: ConfigDict) -> None:
        """
        Load configuration from dictionary.

        Args:
            config_dict: Configuration dictionary
        """
        for key, value in config_dict.items():
            self.set_parameter(key, value)

    def save_to_file(self, filepath: str) -> None:
        """
        Save configuration to JSON file.

        Args:
            filepath: Path to save configuration file
        """
        config_dict = self.to_dict()

        # Convert enum values to strings for JSON serialization
        def convert_enums(obj):
            if hasattr(obj, "value"):
                return obj.value
            elif isinstance(obj, dict):
                return {k: convert_enums(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert_enums(item) for item in obj]
            return obj

        serializable_dict = convert_enums(config_dict)

        with open(filepath, "w") as f:
            json.dump(serializable_dict, f, indent=2)

    @classmethod
    def load_from_file(cls, filepath: str) -> "BaseNoCConfig":
        """
        Load configuration from JSON file.

        Args:
            filepath: Path to configuration file

        Returns:
            Loaded configuration instance
        """
        with open(filepath, "r") as f:
            config_dict = json.load(f)

        instance = cls()
        instance.from_dict(config_dict)
        return instance

    def copy(self) -> "BaseNoCConfig":
        """
        Create a deep copy of the configuration.

        Returns:
            Copy of the configuration
        """
        new_config = self.__class__()
        new_config.from_dict(self.to_dict())
        return new_config

    def update(self, other_config: Union["BaseNoCConfig", ConfigDict]) -> None:
        """
        Update configuration with parameters from another config or dict.

        Args:
            other_config: Another configuration object or dictionary
        """
        if isinstance(other_config, BaseNoCConfig):
            config_dict = other_config.to_dict()
        else:
            config_dict = other_config

        self.from_dict(config_dict)

    def get_compatibility_info(self) -> Dict[str, Any]:
        """
        Get information about compatibility with other configurations.

        Returns:
            Dictionary containing compatibility information
        """
        return {
            "topology_type": self.topology_type.value,
            "flow_control": self.flow_control.value,
            "buffer_type": self.buffer_type.value,
        }

    def validate_basic_params(self) -> ValidationResult:
        """
        Validate basic configuration parameters.

        Returns:
            Tuple of (is_valid, error_message)
        """
        errors = []

        # Validate basic constraints
        if self.num_nodes <= 0:
            errors.append("节点数必须为正数")

        if self.flit_size <= 0:
            errors.append("Flit大小必须为正数")

        if self.packet_size < self.flit_size:
            errors.append("数据包大小必须 >= Flit大小")

        if self.buffer_depth <= 0:
            errors.append("缓冲区深度必须为正数")

        if self.link_bandwidth <= 0:
            errors.append("链路带宽必须为正数")

        if self.clock_frequency <= 0:
            errors.append("时钟频率必须为正数")

        if self.injection_rate < 0 or self.injection_rate > 1:
            errors.append("注入率必须在0和1之间")

        if self.simulation_cycles <= 0:
            errors.append("仿真周期数必须为正数")

        if self.warmup_cycles < 0:
            errors.append("预热周期数必须为非负数")

        if self.stats_start_cycle < 0:
            errors.append("统计开始周期必须为非负数")

        if errors:
            return False, "; ".join(errors)

        return True, None

    def get_performance_bounds(self) -> Dict[str, float]:
        """
        Calculate theoretical performance bounds for the configuration.

        Returns:
            Dictionary containing performance bounds
        """
        # Calculate theoretical maximum throughput
        max_injection_rate = self.link_bandwidth / (self.packet_size / 8)  # packets/s

        # Calculate minimum latency (single hop)
        min_latency = self.link_latency + (self.packet_size / self.flit_size)

        # Calculate buffer capacity
        total_buffer_capacity = self.num_nodes * self.buffer_depth

        return {
            "max_injection_rate": max_injection_rate,
            "min_latency": min_latency,
            "total_buffer_capacity": total_buffer_capacity,
            "theoretical_bisection_bandwidth": self.link_bandwidth * (self.num_nodes / 2),
        }

    def __str__(self) -> str:
        """String representation of the configuration."""
        return f"{self.__class__.__name__}(topology={self.topology_type.value}, nodes={self.num_nodes})"

    def __repr__(self) -> str:
        """Detailed string representation of the configuration."""
        return f"{self.__class__.__name__}({self.to_dict()})"

    # ========== 预设配置管理 ==========

    def register_preset(self, name: str, config_params: Dict[str, Any]) -> None:
        """
        注册预设配置。

        Args:
            name: 预设配置名称
            config_params: 配置参数字典
        """
        self._presets[name] = config_params.copy()

    def apply_preset(self, preset_name: str) -> None:
        """
        应用预设配置。

        Args:
            preset_name: 预设配置名称

        Raises:
            ValueError: 如果预设配置不存在
        """
        if preset_name not in self._presets:
            available = ", ".join(self._presets.keys())
            raise ValueError(f"未知的预设配置: {preset_name}。可用配置: {available}")

        preset_params = self._presets[preset_name]
        self.from_dict(preset_params)

    def get_preset_info(self, preset_name: str) -> Dict[str, Any]:
        """
        获取预设配置信息。

        Args:
            preset_name: 预设配置名称

        Returns:
            预设配置参数字典

        Raises:
            ValueError: 如果预设配置不存在
        """
        if preset_name not in self._presets:
            available = ", ".join(self._presets.keys())
            raise ValueError(f"未知的预设配置: {preset_name}。可用配置: {available}")

        return self._presets[preset_name].copy()

    def get_registered_presets(self) -> List[str]:
        """
        获取已注册的预设配置列表。

        Returns:
            预设配置名称列表
        """
        return list(self._presets.keys())

    # ========== 抽象工厂方法 ==========

    @classmethod
    @abstractmethod
    def create_preset_config(cls, preset_name: str, **kwargs) -> "BaseNoCConfig":
        """
        创建预设配置。

        Args:
            preset_name: 预设配置名称
            **kwargs: 额外配置参数

        Returns:
            配置实例
        """
        pass

    @classmethod
    @abstractmethod
    def create_custom_config(cls, **kwargs) -> "BaseNoCConfig":
        """
        创建自定义配置。

        Args:
            **kwargs: 配置参数

        Returns:
            配置实例
        """
        pass

    @classmethod
    @abstractmethod
    def load_from_file(cls, file_path: str) -> "BaseNoCConfig":
        """
        从文件加载配置。

        Args:
            file_path: 配置文件路径

        Returns:
            配置实例
        """
        pass

    @abstractmethod
    def get_supported_presets(self) -> List[str]:
        """
        获取支持的预设配置列表。

        Returns:
            预设配置名称列表
        """
        pass


def create_config_from_type(topology_type: TopologyType, **kwargs) -> BaseNoCConfig:
    """
    Factory function to create configuration based on topology type.

    Args:
        topology_type: The topology type
        **kwargs: Additional configuration parameters

    Returns:
        Configuration instance appropriate for the topology type
    """
    if topology_type == TopologyType.CROSSRING:
        from src.noc.crossring.config import CrossRingConfig
        return CrossRingConfig.create_custom_config(**kwargs)
    else:
        raise NotImplementedError(f"暂不支持的拓扑类型: {topology_type}")
