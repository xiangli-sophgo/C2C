"""
Mesh NoC配置类。

本模块提供专门针对Mesh拓扑的配置实现，
继承BaseNoCConfig并添加Mesh特有的配置参数。
"""

from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass, field

from ..base.config import BaseNoCConfig
from src.noc.utils.types import TopologyType, RoutingStrategy, ValidationResult


@dataclass
class BasicConfiguration:
    """Mesh基础配置"""
    
    # 网络频率配置
    network_frequency: float = 2.0  # GHz
    
    # 拓扑参数
    rows: int = 4
    cols: int = 4
    
    # 路由参数
    ENABLE_XY_ROUTING: bool = True
    ENABLE_MINIMAL_ROUTING: bool = True
    
    # 缓冲区配置
    INPUT_BUFFER_DEPTH: int = 8
    OUTPUT_BUFFER_DEPTH: int = 8
    VIRTUAL_CHANNELS: int = 2
    
    # 性能参数
    LINK_WIDTH: int = 64  # bits
    FLIT_SIZE: int = 64   # bits
    
    # 延迟参数
    ROUTER_LATENCY: int = 1  # cycles
    LINK_LATENCY: int = 1    # cycles


@dataclass
class MeshConfiguration:
    """Mesh扩展配置 (deprecated, use BasicConfiguration)"""
    
    # 拓扑参数
    rows: int = 4
    cols: int = 4
    
    # 路由参数
    ENABLE_XY_ROUTING: bool = True
    ENABLE_MINIMAL_ROUTING: bool = True
    
    # 缓冲区配置
    INPUT_BUFFER_DEPTH: int = 8
    OUTPUT_BUFFER_DEPTH: int = 8
    VIRTUAL_CHANNELS: int = 2
    
    # 性能参数
    LINK_WIDTH: int = 64  # bits
    FLIT_SIZE: int = 64   # bits
    
    # 延迟参数
    ROUTER_LATENCY: int = 1  # cycles
    LINK_LATENCY: int = 1    # cycles


class MeshConfig(BaseNoCConfig):
    """
    Mesh拓扑专用配置类。
    
    该类扩展BaseNoCConfig，提供专门针对Mesh拓扑的配置管理，
    包括XY路由、虚拟通道、缓冲区等Mesh特有的参数。
    """
    
    def __init__(self, rows: int = 4, cols: int = 4, config_name: str = "default"):
        """
        初始化Mesh配置。
        
        Args:
            rows: 行数
            cols: 列数
            config_name: 配置名称
        """
        super().__init__(TopologyType.MESH)
        
        # 基本拓扑参数
        self.config_name = config_name
        self.rows = rows
        self.cols = cols
        self.num_nodes = rows * cols
        
        # 使用基础配置
        self.basic_config = BasicConfiguration(rows=rows, cols=cols)
        
        # 使用Mesh配置（向后兼容）
        self.mesh_config = MeshConfiguration(rows=rows, cols=cols)
        
        # 设置路由策略
        self.routing_strategy = RoutingStrategy.XY if self.basic_config.ENABLE_XY_ROUTING else RoutingStrategy.SHORTEST
        
        # 生成节点位置映射
        self._generate_node_positions()
        
        # 生成邻接关系
        self._generate_adjacency()
    
    def _generate_node_positions(self) -> None:
        """生成节点位置映射"""
        self.node_positions = {}
        for row in range(self.rows):
            for col in range(self.cols):
                node_id = row * self.cols + col
                self.node_positions[node_id] = (row, col)
    
    def _generate_adjacency(self) -> None:
        """生成邻接关系"""
        self.adjacency_list = {}
        
        for node_id in range(self.num_nodes):
            row, col = self.node_positions[node_id]
            neighbors = []
            
            # 北邻居
            if row > 0:
                neighbors.append((row - 1) * self.cols + col)
            
            # 南邻居
            if row < self.rows - 1:
                neighbors.append((row + 1) * self.cols + col)
            
            # 西邻居
            if col > 0:
                neighbors.append(row * self.cols + (col - 1))
            
            # 东邻居
            if col < self.cols - 1:
                neighbors.append(row * self.cols + (col + 1))
            
            self.adjacency_list[node_id] = neighbors
    
    def get_node_position(self, node_id: int) -> Tuple[int, int]:
        """
        获取节点位置。
        
        Args:
            node_id: 节点ID
            
        Returns:
            (row, col) 位置元组
        """
        return self.node_positions.get(node_id, (0, 0))
    
    def get_node_id(self, row: int, col: int) -> int:
        """
        根据位置获取节点ID。
        
        Args:
            row: 行
            col: 列
            
        Returns:
            节点ID
        """
        if 0 <= row < self.rows and 0 <= col < self.cols:
            return row * self.cols + col
        return -1
    
    def get_neighbors(self, node_id: int) -> List[int]:
        """
        获取节点的邻居。
        
        Args:
            node_id: 节点ID
            
        Returns:
            邻居节点ID列表
        """
        return self.adjacency_list.get(node_id, [])
    
    def calculate_manhattan_distance(self, src: int, dst: int) -> int:
        """
        计算曼哈顿距离。
        
        Args:
            src: 源节点ID
            dst: 目标节点ID
            
        Returns:
            曼哈顿距离
        """
        src_row, src_col = self.get_node_position(src)
        dst_row, dst_col = self.get_node_position(dst)
        return abs(src_row - dst_row) + abs(src_col - dst_col)
    
    def validate_config(self) -> ValidationResult:
        """
        验证Mesh配置参数。
        
        Returns:
            ValidationResult: (是否有效, 错误信息)
        """
        # 先验证基础参数
        basic_valid, basic_error = self.validate_basic_params()
        if not basic_valid:
            return basic_valid, basic_error
        
        errors = []
        
        # 拓扑参数验证
        if self.rows <= 0 or self.cols <= 0:
            errors.append(f"行数和列数必须为正数 (rows={self.rows}, cols={self.cols})")
        
        if self.num_nodes != self.rows * self.cols:
            errors.append(f"节点数必须等于行数×列数 (num_nodes={self.num_nodes}, rows={self.rows}, cols={self.cols})")
        
        # 缓冲区深度验证
        if self.mesh_config.INPUT_BUFFER_DEPTH <= 0:
            errors.append(f"输入缓冲区深度必须为正数 (INPUT_BUFFER_DEPTH={self.mesh_config.INPUT_BUFFER_DEPTH})")
        
        if self.mesh_config.OUTPUT_BUFFER_DEPTH <= 0:
            errors.append(f"输出缓冲区深度必须为正数 (output_buffer_depth={self.mesh_config.output_buffer_depth})")
        
        # 虚拟通道验证
        if self.mesh_config.VIRTUAL_CHANNELS <= 0:
            errors.append(f"虚拟通道数必须为正数 (VIRTUAL_CHANNELS={self.mesh_config.VIRTUAL_CHANNELS})")
        
        # 链路参数验证
        if self.mesh_config.LINK_WIDTH <= 0:
            errors.append(f"链路宽度必须为正数 (LINK_WIDTH={self.mesh_config.LINK_WIDTH})")
        
        if self.mesh_config.FLIT_SIZE <= 0:
            errors.append(f"Flit大小必须为正数 (FLIT_SIZE={self.mesh_config.FLIT_SIZE})")
        
        # 延迟参数验证
        if self.mesh_config.ROUTER_LATENCY < 0:
            errors.append(f"路由器延迟不能为负数 (ROUTER_LATENCY={self.mesh_config.ROUTER_LATENCY})")
        
        if self.mesh_config.LINK_LATENCY < 0:
            errors.append(f"链路延迟不能为负数 (LINK_LATENCY={self.mesh_config.LINK_LATENCY})")
        
        if errors:
            return False, "; ".join(errors)
        
        return True, None
    
    def get_topology_params(self) -> Dict[str, Any]:
        """
        获取Mesh拓扑参数。
        
        Returns:
            包含拓扑特定参数的字典
        """
        return {
            "topology_type": self.topology_type,
            "rows": self.rows,
            "cols": self.cols,
            "num_nodes": self.num_nodes,
            "routing_strategy": self.routing_strategy,
            "mesh_config": {
                "enable_xy_routing": self.mesh_config.ENABLE_XY_ROUTING,
                "enable_minimal_routing": self.mesh_config.ENABLE_MINIMAL_ROUTING,
                "input_buffer_depth": self.mesh_config.INPUT_BUFFER_DEPTH,
                "output_buffer_depth": self.mesh_config.OUTPUT_BUFFER_DEPTH,
                "virtual_channels": self.mesh_config.VIRTUAL_CHANNELS,
                "link_width": self.mesh_config.LINK_WIDTH,
                "flit_size": self.mesh_config.FLIT_SIZE,
                "router_latency": self.mesh_config.ROUTER_LATENCY,
                "link_latency": self.mesh_config.LINK_LATENCY,
            },
            "node_positions": self.node_positions,
            "adjacency_list": self.adjacency_list,
        }
    
    def update_topology_size(self, rows: int, cols: int) -> None:
        """
        更新拓扑大小。
        
        Args:
            rows: 新的行数
            cols: 新的列数
        """
        self.rows = rows
        self.cols = cols
        self.num_nodes = rows * cols
        self.mesh_config.rows = rows
        self.mesh_config.cols = cols
        
        # 重新生成位置和邻接关系
        self._generate_node_positions()
        self._generate_adjacency()
    
    def enable_xy_routing(self, enable: bool = True) -> None:
        """
        启用/禁用XY路由。
        
        Args:
            enable: 是否启用XY路由
        """
        self.mesh_config.ENABLE_XY_ROUTING = enable
        self.routing_strategy = RoutingStrategy.XY if enable else RoutingStrategy.SHORTEST
    
    def update_buffer_config(self, input_depth: Optional[int] = None, 
                           output_depth: Optional[int] = None,
                           virtual_channels: Optional[int] = None) -> None:
        """
        更新缓冲区配置。
        
        Args:
            input_depth: 输入缓冲区深度
            output_depth: 输出缓冲区深度
            virtual_channels: 虚拟通道数
        """
        if input_depth is not None:
            self.mesh_config.INPUT_BUFFER_DEPTH = input_depth
        if output_depth is not None:
            self.mesh_config.OUTPUT_BUFFER_DEPTH = output_depth
        if virtual_channels is not None:
            self.mesh_config.VIRTUAL_CHANNELS = virtual_channels
    
    def to_dict(self) -> Dict[str, Any]:
        """
        转换为字典格式。
        
        Returns:
            配置的字典表示
        """
        base_dict = super().to_dict()
        
        # 添加Mesh特有参数
        mesh_dict = {
            "config_name": self.config_name,
            "rows": self.rows,
            "cols": self.cols,
            "mesh_config": self.mesh_config.__dict__,
            "node_positions": self.node_positions,
            "adjacency_list": self.adjacency_list,
        }
        
        base_dict.update(mesh_dict)
        return base_dict
    
    def from_dict(self, config_dict: Dict[str, Any]) -> None:
        """
        从字典加载配置。
        
        Args:
            config_dict: 配置字典
        """
        # 先调用父类方法
        super().from_dict(config_dict)
        
        # 加载Mesh特有参数
        if "config_name" in config_dict:
            self.config_name = config_dict["config_name"]
        if "rows" in config_dict:
            self.rows = config_dict["rows"]
        if "cols" in config_dict:
            self.cols = config_dict["cols"]
        
        # 加载mesh配置
        if "mesh_config" in config_dict:
            mesh_cfg = config_dict["mesh_config"]
            if isinstance(mesh_cfg, dict):
                for key, value in mesh_cfg.items():
                    if hasattr(self.mesh_config, key):
                        setattr(self.mesh_config, key, value)
        
        # 加载位置和邻接关系
        if "node_positions" in config_dict:
            self.node_positions = config_dict["node_positions"]
        if "adjacency_list" in config_dict:
            self.adjacency_list = config_dict["adjacency_list"]
        
        # 更新节点数
        self.num_nodes = self.rows * self.cols
    
    def __str__(self) -> str:
        """字符串表示。"""
        return f"MeshConfig({self.config_name}, {self.rows}×{self.cols})"
    
    def __repr__(self) -> str:
        """详细字符串表示。"""
        return f"MeshConfig(name='{self.config_name}', topology={self.rows}×{self.cols})"


# ========== 便捷函数 ==========

def create_mesh_config_2x2() -> MeshConfig:
    """创建2x2 Mesh配置"""
    return MeshConfig(rows=2, cols=2, config_name="2x2_mesh")


def create_mesh_config_4x4() -> MeshConfig:
    """创建4x4 Mesh配置"""
    return MeshConfig(rows=4, cols=4, config_name="4x4_mesh")


def create_mesh_config_8x8() -> MeshConfig:
    """创建8x8 Mesh配置"""
    return MeshConfig(rows=8, cols=8, config_name="8x8_mesh")


def create_mesh_config_custom(rows: int, cols: int, config_name: str = "custom", **kwargs) -> MeshConfig:
    """
    创建自定义Mesh配置
    
    Args:
        rows: 行数
        cols: 列数
        config_name: 配置名称
        **kwargs: 其他配置参数
    
    Returns:
        Mesh配置实例
    """
    config = MeshConfig(rows=rows, cols=cols, config_name=config_name)
    
    # 应用自定义参数
    for key, value in kwargs.items():
        if hasattr(config, key):
            setattr(config, key, value)
        elif hasattr(config.mesh_config, key):
            setattr(config.mesh_config, key, value)
    
    return config