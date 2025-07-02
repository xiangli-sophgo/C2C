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

from ..types import (
    TopologyType, RoutingStrategy, FlowControlType, BufferType,
    TrafficPattern, NoCConfiguration, ConfigDict, ValidationResult
)


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
        self.flit_size = 64  # bits
        self.packet_size = 512  # bits
        self.buffer_depth = 8  # flits
        self.link_bandwidth = 1.0  # GB/s
        self.link_latency = 1  # cycles
        
        # 流控制参数
        self.flow_control = FlowControlType.WORMHOLE
        self.buffer_type = BufferType.SHARED
        self.virtual_channels = 2
        
        # 仿真参数
        self.simulation_cycles = 10000
        self.warmup_cycles = 1000
        self.stats_start_cycle = 1000
        
        # 时钟和时序
        self.clock_frequency = 1.0  # GHz
        self.network_frequency = 1.0  # Relative to system clock
        
        # 流量参数
        self.traffic_pattern = TrafficPattern.UNIFORM_RANDOM
        self.injection_rate = 0.1  # flits per cycle per node
        
        # 服务质量
        self.enable_qos = False
        self.priority_levels = 4
        
        # 高级功能
        self.enable_adaptive_routing = False
        self.enable_multicast = False
        self.enable_power_management = False
        
        # 容错能力
        self.fault_tolerance_enabled = False
        self.redundancy_level = 0
        
        # 自定义参数存储
        self._custom_params = {}
        
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
            if not key.startswith('_'):
                if hasattr(value, 'value'):  # Handle Enum types
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
            if hasattr(obj, 'value'):
                return obj.value
            elif isinstance(obj, dict):
                return {k: convert_enums(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert_enums(item) for item in obj]
            return obj
        
        serializable_dict = convert_enums(config_dict)
        
        with open(filepath, 'w') as f:
            json.dump(serializable_dict, f, indent=2)
    
    @classmethod
    def load_from_file(cls, filepath: str) -> 'BaseNoCConfig':
        """
        Load configuration from JSON file.
        
        Args:
            filepath: Path to configuration file
            
        Returns:
            Loaded configuration instance
        """
        with open(filepath, 'r') as f:
            config_dict = json.load(f)
        
        instance = cls()
        instance.from_dict(config_dict)
        return instance
    
    def copy(self) -> 'BaseNoCConfig':
        """
        Create a deep copy of the configuration.
        
        Returns:
            Copy of the configuration
        """
        new_config = self.__class__()
        new_config.from_dict(self.to_dict())
        return new_config
    
    def update(self, other_config: Union['BaseNoCConfig', ConfigDict]) -> None:
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
            'topology_type': self.topology_type.value,
            'supports_adaptive_routing': self.enable_adaptive_routing,
            'supports_qos': self.enable_qos,
            'supports_multicast': self.enable_multicast,
            'flow_control': self.flow_control.value,
            'buffer_type': self.buffer_type.value
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
            'max_injection_rate': max_injection_rate,
            'min_latency': min_latency,
            'total_buffer_capacity': total_buffer_capacity,
            'theoretical_bisection_bandwidth': self.link_bandwidth * (self.num_nodes / 2)
        }
    
    def optimize_for_workload(self, workload_characteristics: Dict[str, Any]) -> None:
        """
        Optimize configuration parameters for a specific workload.
        
        Args:
            workload_characteristics: Dictionary describing workload properties
        """
        # This is a basic implementation - can be overridden in subclasses
        if 'high_throughput' in workload_characteristics:
            self.buffer_depth = max(self.buffer_depth, 16)
            self.virtual_channels = max(self.virtual_channels, 4)
        
        if 'low_latency' in workload_characteristics:
            self.buffer_depth = min(self.buffer_depth, 4)
            self.enable_adaptive_routing = True
        
        if 'power_sensitive' in workload_characteristics:
            self.enable_power_management = True
    
    def __str__(self) -> str:
        """String representation of the configuration."""
        return f"{self.__class__.__name__}(topology={self.topology_type.value}, nodes={self.num_nodes})"
    
    def __repr__(self) -> str:
        """Detailed string representation of the configuration."""
        return f"{self.__class__.__name__}({self.to_dict()})"


class CrossRingCompatibleConfig(BaseNoCConfig):
    """
    与现有CrossRing实现兼容的配置类。
    
    该类扩展BaseNoCConfig以提供与现有CrossRing配置系统的兼容性，
    同时保持新的抽象接口。
    """
    
    def __init__(self, crossring_config_path: Optional[str] = None):
        """
        初始化CrossRing兼容的配置。
        
        Args:
            crossring_config_path: 现有CrossRing配置文件的路径
        """
        super().__init__(TopologyType.CROSSRING)
        
        # CrossRing特定参数
        self.NUM_NODE = 20
        self.NUM_COL = 2
        self.NUM_ROW = 10
        self.NUM_IP = 16
        self.NUM_RN = 16
        self.NUM_SN = 16
        
        # CrossRing IP配置
        self.NUM_GDMA = 16
        self.NUM_SDMA = 16
        self.NUM_CDMA = 16
        self.NUM_DDR = 16
        self.NUM_L2M = 16
        
        # CrossRing缓冲区配置
        self.SLICE_PER_LINK = 8
        self.RB_IN_FIFO_DEPTH = 16
        self.RB_OUT_FIFO_DEPTH = 8
        self.IQ_OUT_FIFO_DEPTH = 8
        self.EQ_IN_FIFO_DEPTH = 16
        self.IQ_CH_FIFO_DEPTH = 10
        self.EQ_CH_FIFO_DEPTH = 10
        
        # CrossRing ETag配置
        self.ITag_TRIGGER_Th_H = 80
        self.ITag_TRIGGER_Th_V = 80
        self.ITag_MAX_NUM_H = 1
        self.ITag_MAX_NUM_V = 1
        self.ETag_BOTHSIDE_UPGRADE = 0
        
        # CrossRing时序和延迟
        self.BURST = 4
        self.NETWORK_FREQUENCY = 1.0
        self.DDR_R_LATENCY = 100
        self.DDR_R_LATENCY_VAR = 0
        self.DDR_W_LATENCY = 0
        self.L2M_R_LATENCY = 12
        self.L2M_W_LATENCY = 16
        self.SN_TRACKER_RELEASE_LATENCY = 40
        
        # CrossRing跟踪器配置
        self.RN_R_TRACKER_OSTD = 64
        self.RN_W_TRACKER_OSTD = 32
        self.SN_DDR_R_TRACKER_OSTD = 96
        self.SN_DDR_W_TRACKER_OSTD = 48
        self.SN_L2M_R_TRACKER_OSTD = 96
        self.SN_L2M_W_TRACKER_OSTD = 48
        
        # CrossRing缓冲区大小
        self.RN_RDB_SIZE = self.RN_R_TRACKER_OSTD * self.BURST
        self.RN_WDB_SIZE = self.RN_W_TRACKER_OSTD * self.BURST
        self.SN_DDR_RDB_SIZE = self.SN_DDR_R_TRACKER_OSTD * self.BURST
        self.SN_DDR_WDB_SIZE = self.SN_DDR_W_TRACKER_OSTD * self.BURST
        self.SN_L2M_RDB_SIZE = self.SN_L2M_R_TRACKER_OSTD * self.BURST
        self.SN_L2M_WDB_SIZE = self.SN_L2M_W_TRACKER_OSTD * self.BURST
        
        # CrossRing带宽限制
        self.GDMA_BW_LIMIT = 8
        self.SDMA_BW_LIMIT = 8
        self.CDMA_BW_LIMIT = 8
        self.DDR_BW_LIMIT = 80
        self.L2M_BW_LIMIT = 80
        
        # CrossRing ETag UE限制
        self.TL_Etag_T1_UE_MAX = 15
        self.TL_Etag_T2_UE_MAX = 8
        self.TR_Etag_T2_UE_MAX = 8
        self.TU_Etag_T1_UE_MAX = 15
        self.TU_Etag_T2_UE_MAX = 8
        self.TD_Etag_T2_UE_MAX = 8
        
        # CrossRing读写间隔配置
        self.GDMA_RW_GAP = 8
        self.SDMA_RW_GAP = 8
        
        # 通道规格
        self.CHANNEL_SPEC = {
            "gdma": 2,
            "sdma": 2,
            "cdma": 2,
            "ddr": 2,
            "l2m": 2,
        }
        
        # 生成通道名称列表
        self.CH_NAME_LIST = []
        for key in self.CHANNEL_SPEC:
            for idx in range(self.CHANNEL_SPEC[key]):
                self.CH_NAME_LIST.append(f"{key}_{idx}")
        
        # IP位置列表（将根据拓扑更新）
        self.DDR_SEND_POSITION_LIST = []
        self.L2M_SEND_POSITION_LIST = []
        self.GDMA_SEND_POSITION_LIST = []
        self.SDMA_SEND_POSITION_LIST = []
        self.CDMA_SEND_POSITION_LIST = []
        
        # 如果提供了文件则从文件加载
        if crossring_config_path and os.path.exists(crossring_config_path):
            self._load_crossring_config(crossring_config_path)
        else:
            # 设置默认IP位置
            self._update_ip_positions()
    
    def _load_crossring_config(self, config_path: str) -> None:
        """Load configuration from CrossRing config file."""
        try:
            with open(config_path, 'r') as f:
                config_data = json.load(f)
            
            # Map CrossRing parameters to our configuration
            for key, value in config_data.items():
                if hasattr(self, key):
                    setattr(self, key, value)
            
            self._update_ip_positions()
            
        except Exception as e:
            print(f"警告：无法从{config_path}加载CrossRing配置：{e}")
            self._update_ip_positions()
    
    def _update_ip_positions(self) -> None:
        """Update IP position lists based on current topology configuration."""
        # Default CrossRing IP distribution
        ip_positions = [self.NUM_COL * 2 * (x // self.NUM_COL) + self.NUM_COL + x % self.NUM_COL 
                       for x in range(self.NUM_IP)]
        
        self.DDR_SEND_POSITION_LIST = ip_positions.copy()
        self.L2M_SEND_POSITION_LIST = ip_positions.copy()
        self.GDMA_SEND_POSITION_LIST = ip_positions.copy()
        self.SDMA_SEND_POSITION_LIST = ip_positions.copy()
        self.CDMA_SEND_POSITION_LIST = ip_positions.copy()
        
        # Update base class parameters
        self.num_nodes = self.NUM_NODE
        self.flit_size = 64  # Default flit size for CrossRing
    
    def validate_config(self) -> ValidationResult:
        """Validate CrossRing-specific configuration parameters."""
        # First validate basic parameters
        basic_valid, basic_error = self.validate_basic_params()
        if not basic_valid:
            return basic_valid, basic_error
        
        errors = []
        
        # CrossRing-specific validations
        if self.NUM_NODE != self.NUM_ROW * self.NUM_COL:
            errors.append("NUM_NODE must equal NUM_ROW * NUM_COL")
        
        if self.NUM_IP > self.NUM_NODE:
            errors.append("NUM_IP cannot exceed NUM_NODE")
        
        # ETag validations
        if not (self.TL_Etag_T2_UE_MAX < self.TL_Etag_T1_UE_MAX < self.RB_IN_FIFO_DEPTH):
            errors.append("ETag T2 < T1 < RB_IN_FIFO_DEPTH constraint violated for TL")
        
        if not (self.TU_Etag_T2_UE_MAX < self.TU_Etag_T1_UE_MAX < self.EQ_IN_FIFO_DEPTH):
            errors.append("ETag T2 < T1 < EQ_IN_FIFO_DEPTH constraint violated for TU")
        
        # Buffer size validations
        if self.RN_RDB_SIZE != self.RN_R_TRACKER_OSTD * self.BURST:
            errors.append("RN_RDB_SIZE must equal RN_R_TRACKER_OSTD * BURST")
        
        if self.RN_WDB_SIZE != self.RN_W_TRACKER_OSTD * self.BURST:
            errors.append("RN_WDB_SIZE must equal RN_W_TRACKER_OSTD * BURST")
        
        if errors:
            return False, "; ".join(errors)
        
        return True, None
    
    def get_topology_params(self) -> Dict[str, Any]:
        """Get CrossRing-specific topology parameters."""
        return {
            'NUM_NODE': self.NUM_NODE,
            'NUM_COL': self.NUM_COL,
            'NUM_ROW': self.NUM_ROW,
            'NUM_IP': self.NUM_IP,
            'CHANNEL_SPEC': self.CHANNEL_SPEC,
            'CH_NAME_LIST': self.CH_NAME_LIST,
            'DDR_SEND_POSITION_LIST': self.DDR_SEND_POSITION_LIST,
            'L2M_SEND_POSITION_LIST': self.L2M_SEND_POSITION_LIST,
            'GDMA_SEND_POSITION_LIST': self.GDMA_SEND_POSITION_LIST,
            'SDMA_SEND_POSITION_LIST': self.SDMA_SEND_POSITION_LIST,
            'CDMA_SEND_POSITION_LIST': self.CDMA_SEND_POSITION_LIST,
        }
    
    def update_topology(self, topo_type: str) -> None:
        """
        Update configuration for different topology types.
        
        Args:
            topo_type: Topology type string (e.g., "5x2", "5x4")
        """
        if topo_type == "5x2":
            self.NUM_NODE = 20
            self.NUM_COL = 2
            self.NUM_IP = 16
            self.NUM_ROW = self.NUM_NODE // self.NUM_COL
            self._generate_5x2_positions()
        elif topo_type == "5x4":
            self.NUM_NODE = 40
            self.NUM_COL = 4
            self.NUM_IP = 32
            self.NUM_ROW = self.NUM_NODE // self.NUM_COL
            self._generate_5x4_positions()
        else:
            # Default topology
            self._update_ip_positions()
        
        # Update base class parameters
        self.num_nodes = self.NUM_NODE
    
    def _generate_5x2_positions(self) -> None:
        """Generate IP positions for 5x2 topology."""
        active_rows = [i for i in range(self.NUM_ROW) if i % 2 == 0]
        self.DDR_SEND_POSITION_LIST = self._generate_ip_positions(active_rows, [])
        self.L2M_SEND_POSITION_LIST = self._generate_ip_positions(active_rows, [])
        self.GDMA_SEND_POSITION_LIST = self._generate_ip_positions(active_rows, [])
        self.SDMA_SEND_POSITION_LIST = self._generate_ip_positions(active_rows, [])
        self.CDMA_SEND_POSITION_LIST = self._generate_ip_positions(active_rows, [])
    
    def _generate_5x4_positions(self) -> None:
        """Generate IP positions for 5x4 topology."""
        active_rows = [i for i in range(self.NUM_ROW) if i % 2 == 0] + [9]
        self.DDR_SEND_POSITION_LIST = self._generate_ip_positions(active_rows, [])
        self.L2M_SEND_POSITION_LIST = self._generate_ip_positions(active_rows, [])
        self.GDMA_SEND_POSITION_LIST = self._generate_ip_positions(active_rows, [])
        self.SDMA_SEND_POSITION_LIST = self._generate_ip_positions(active_rows, [])
        self.CDMA_SEND_POSITION_LIST = self._generate_ip_positions(
            [i for i in range(self.NUM_ROW - 1)], [])
    
    def _generate_ip_positions(self, zero_rows: List[int] = None, 
                              zero_cols: List[int] = None) -> List[int]:
        """
        Generate IP positions based on row/column constraints.
        
        Args:
            zero_rows: List of rows to exclude (use active rows for CrossRing)
            zero_cols: List of columns to exclude
            
        Returns:
            List of position indices
        """
        # Create matrix with 1s where IPs can be placed
        matrix = [[1 for _ in range(self.NUM_COL)] for _ in range(self.NUM_ROW)]
        
        # Set specified rows to 0 (actually these are active rows in CrossRing context)
        if zero_rows:
            for row in range(self.NUM_ROW):
                if row not in zero_rows:  # Invert logic for CrossRing
                    for col in range(self.NUM_COL):
                        matrix[row][col] = 0
        
        # Set specified columns to 0
        if zero_cols:
            for col in zero_cols:
                if 0 <= col < self.NUM_COL:
                    for row in range(self.NUM_ROW):
                        matrix[row][col] = 0
        
        # Collect all positions with value 1
        positions = []
        for r in range(self.NUM_ROW):
            for c in range(self.NUM_COL):
                if matrix[r][c] == 1:
                    position = r * self.NUM_COL + c
                    positions.append(position)
        
        return positions
    
    def get_crossring_config_dict(self) -> Dict[str, Any]:
        """
        Get configuration in CrossRing-compatible format.
        
        Returns:
            Dictionary compatible with original CrossRing configuration
        """
        return {
            'NUM_NODE': self.NUM_NODE,
            'NUM_COL': self.NUM_COL,
            'NUM_IP': self.NUM_IP,
            'NUM_RN': self.NUM_RN,
            'NUM_SN': self.NUM_SN,
            'NUM_GDMA': self.NUM_GDMA,
            'NUM_SDMA': self.NUM_SDMA,
            'NUM_CDMA': self.NUM_CDMA,
            'NUM_DDR': self.NUM_DDR,
            'NUM_L2M': self.NUM_L2M,
            'FLIT_SIZE': self.flit_size,
            'SLICE_PER_LINK': self.SLICE_PER_LINK,
            'RB_IN_FIFO_DEPTH': self.RB_IN_FIFO_DEPTH,
            'RB_OUT_FIFO_DEPTH': self.RB_OUT_FIFO_DEPTH,
            'IQ_OUT_FIFO_DEPTH': self.IQ_OUT_FIFO_DEPTH,
            'EQ_IN_FIFO_DEPTH': self.EQ_IN_FIFO_DEPTH,
            'IQ_CH_FIFO_DEPTH': self.IQ_CH_FIFO_DEPTH,
            'EQ_CH_FIFO_DEPTH': self.EQ_CH_FIFO_DEPTH,
            'BURST': self.BURST,
            'NETWORK_FREQUENCY': self.NETWORK_FREQUENCY,
            'DDR_R_LATENCY': self.DDR_R_LATENCY,
            'DDR_W_LATENCY': self.DDR_W_LATENCY,
            'L2M_R_LATENCY': self.L2M_R_LATENCY,
            'L2M_W_LATENCY': self.L2M_W_LATENCY,
            # Add all other CrossRing parameters as needed
        }


def create_config_from_type(topology_type: TopologyType, 
                           **kwargs) -> BaseNoCConfig:
    """
    Factory function to create configuration based on topology type.
    
    Args:
        topology_type: The topology type
        **kwargs: Additional configuration parameters
        
    Returns:
        Configuration instance appropriate for the topology type
    """
    if topology_type == TopologyType.CROSSRING:
        return CrossRingCompatibleConfig(**kwargs)
    else:
        # For other topology types, use base configuration
        config = BaseNoCConfig(topology_type)
        for key, value in kwargs.items():
            config.set_parameter(key, value)
        return config