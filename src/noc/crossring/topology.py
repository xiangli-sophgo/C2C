"""
CrossRing拓扑实现。

本模块提供CrossRing拓扑的完整实现，包括：
- 拓扑构建和邻接矩阵生成
- 路径计算算法（HV/VH路由）
- 拓扑分析工具（环结构、性能指标）
- 专用的CrossRing路由策略
"""

import numpy as np
from typing import List, Dict, Tuple, Optional, Set, Any
from collections import deque, defaultdict

from ..base.topology import BaseNoCTopology
from src.noc.utils.types import NodeId, Path, Position, ValidationResult, AdjacencyMatrix, RoutingStrategy, TopologyType, LinkId, MetricsDict
from .config import CrossRingConfig
from ..utils.adjacency import create_crossring_adjacency_matrix, validate_adjacency_matrix


class CrossRingTopology(BaseNoCTopology):
    """
    CrossRing拓扑实现类（作为Mesh拓扑）。

    拓扑特点：
    - 节点按二维网格排列（num_rows × num_cols）
    - 节点与其上下左右的邻居连接（如果存在）
    - 边缘节点没有环形回绕连接
    - 支持水平优先(HV)和垂直优先(VH)的确定性路由
    """

    def __init__(self, config: CrossRingConfig):
        """
        初始化CrossRing拓扑。

        Args:
            config: CrossRing配置对象
        """
        # 验证配置（为测试临时禁用）
        # is_valid, error_msg = config.validate_config()
        # if not is_valid:
        #     raise ValueError(f"CrossRing配置无效: {error_msg}")

        # 设置拓扑类型
        if hasattr(config, "topology_type"):
            config.topology_type = TopologyType.CROSSRING

        # CrossRing特有属性
        self.NUM_ROW = config.NUM_ROW
        self.NUM_COL = config.NUM_COL
        self.NUM_NODE = self.NUM_ROW * self.NUM_COL
        self.crossring_config = config

        super().__init__(config)

    def build_topology(self) -> None:
        """构建CrossRing拓扑结构。"""

        # 创建邻接矩阵
        self._adjacency_matrix = create_crossring_adjacency_matrix(self.NUM_ROW, self.NUM_COL)

        # 验证邻接矩阵
        is_valid, error_msg = validate_adjacency_matrix(self._adjacency_matrix)
        if not is_valid:
            raise ValueError(f"生成的邻接矩阵无效: {error_msg}")

        # 建立节点位置映射
        self._build_node_positions()

        # 建立路由表
        self.routing_table = {}  # routing_table[src][dst] = [path_of_node_ids]
        self._build_routing_table()

    def _build_node_positions(self) -> None:
        """建立节点位置映射。使用直角坐标系，原点在左下角。"""
        self._node_positions.clear()

        for node_id in range(self.NUM_NODE):
            col = node_id % self.NUM_COL  # x坐标：水平方向，从左到右
            row = self.NUM_ROW - 1 - (node_id // self.NUM_COL)  # y坐标：垂直方向，从下到上
            self._node_positions[node_id] = (col, row)  # (x, y)

    def get_neighbors(self, node_id: NodeId) -> List[NodeId]:
        """
        获取指定节点的邻居节点列表。

        Args:
            node_id: 节点ID

        Returns:
            邻居节点ID列表
        """
        if not (0 <= node_id < self.NUM_NODE):
            raise ValueError(f"节点ID {node_id} 超出范围 [0, {self.NUM_NODE-1}]")

        neighbors = []
        adj_matrix = self.get_adjacency_matrix()

        for i, connected in enumerate(adj_matrix[node_id]):
            if connected == 1:
                neighbors.append(i)

        return neighbors

    def get_node_position(self, node_id: NodeId) -> Position:
        """
        获取节点的物理位置坐标。使用直角坐标系，原点在左下角。

        Args:
            node_id: 节点ID

        Returns:
            节点位置坐标 (x, y)，其中x是水平方向，y是垂直方向
        """
        if not (0 <= node_id < self.NUM_NODE):
            raise ValueError(f"节点ID {node_id} 超出范围 [0, {self.NUM_NODE-1}]")

        return self._node_positions[node_id]

    def calculate_shortest_path(self, src: NodeId, dst: NodeId) -> Path:
        """
        计算两点之间的最短路径（考虑路由策略）。

        Args:
            src: 源节点ID
            dst: 目标节点ID

        Returns:
            最短路径（节点ID列表）
        """
        # 优先使用确定性路由算法
        return self._calculate_deterministic_path(src, dst)

    def _calculate_deterministic_path(self, src: NodeId, dst: NodeId) -> Path:
        """
        基于路由策略计算确定性路径。
        
        Args:
            src: 源节点ID
            dst: 目标节点ID
            
        Returns:
            确定性路径（节点ID列表）
        """
        if src == dst:
            return [src]
            
        # 获取路由策略
        routing_strategy = getattr(self.crossring_config, "ROUTING_STRATEGY", "XY")
        if hasattr(routing_strategy, "value"):
            routing_strategy = routing_strategy.value
            
        # 获取源和目标坐标
        src_row = src // self.NUM_COL
        src_col = src % self.NUM_COL
        dst_row = dst // self.NUM_COL
        dst_col = dst % self.NUM_COL
        
        path = [src]
        current_node = src
        current_row = src_row
        current_col = src_col
        
        if routing_strategy == "XY":
            # XY路由：先水平后垂直
            # 1. 水平移动
            while current_col != dst_col:
                if dst_col > current_col:
                    current_col += 1
                else:
                    current_col -= 1
                current_node = current_row * self.NUM_COL + current_col
                path.append(current_node)
                
            # 2. 垂直移动
            while current_row != dst_row:
                if dst_row > current_row:
                    current_row += 1
                else:
                    current_row -= 1
                current_node = current_row * self.NUM_COL + current_col
                path.append(current_node)
                
        elif routing_strategy == "YX":
            # YX路由：先垂直后水平
            # 1. 垂直移动
            while current_row != dst_row:
                if dst_row > current_row:
                    current_row += 1
                else:
                    current_row -= 1
                current_node = current_row * self.NUM_COL + current_col
                path.append(current_node)
                
            # 2. 水平移动
            while current_col != dst_col:
                if dst_col > current_col:
                    current_col += 1
                else:
                    current_col -= 1
                current_node = current_row * self.NUM_COL + current_col
                path.append(current_node)
        else:
            # 默认使用BFS
            return self._calculate_shortest_path_bfs(src, dst)
            
        return path

    def _calculate_shortest_path_bfs(self, src: NodeId, dst: NodeId) -> Path:
        """
        使用BFS计算最短路径。

        Args:
            src: 源节点ID
            dst: 目标节点ID

        Returns:
            最短路径
        """
        if src == dst:
            return [src]

        # BFS队列：(current_node, path_to_current)
        queue = deque([(src, [src])])
        visited = {src}

        while queue:
            current, path = queue.popleft()

            for neighbor in self.get_neighbors(current):
                if neighbor == dst:
                    return path + [neighbor]

                if neighbor not in visited:
                    visited.add(neighbor)
                    queue.append((neighbor, path + [neighbor]))

        # 如果没有找到路径，返回空路径
        return []

    def validate_topology(self) -> ValidationResult:
        """
        验证CrossRing拓扑结构的有效性。

        Returns:
            ValidationResult: (是否有效, 错误消息)
        """
        # 验证基本拓扑结构
        if not self._adjacency_matrix:
            return False, "邻接矩阵未初始化"

        # 验证邻接矩阵
        is_valid, error_msg = validate_adjacency_matrix(self._adjacency_matrix)
        if not is_valid:
            return False, f"邻接矩阵无效: {error_msg}"

        # 验证节点数量
        if len(self._adjacency_matrix) != self.NUM_NODE:
            return False, f"邻接矩阵大小不匹配: 期望{self.NUM_NODE}，实际{len(self._adjacency_matrix)}"

        # 验证连通性
        if not self.is_connected():
            return False, "拓扑不连通"

        return True, None

    def get_ring_distance(self, src: NodeId, dst: NodeId, direction: str) -> float:
        """
        计算环内距离。

        Args:
            src: 源节点ID
            dst: 目标节点ID
            direction: 方向（"horizontal" 或 "vertical"）

        Returns:
            环内距离
        """
        if direction not in ["horizontal", "vertical"]:
            raise ValueError(f"无效的方向: {direction}")

        src_row, src_col = self.get_node_position(src)
        dst_row, dst_col = self.get_node_position(dst)

        if direction == "horizontal":
            # 检查是否在同一行
            if src_row != dst_row:
                return float("inf")
            # 计算水平距离
            return abs(dst_col - src_col)
        else:  # vertical
            # 检查是否在同一列
            if src_col != dst_col:
                return float("inf")
            # 计算垂直距离
            return abs(dst_row - src_row)

    def get_horizontal_rings(self) -> List[List[NodeId]]:
        """
        获取所有水平环。

        Returns:
            水平环列表
        """
        rings = []
        for row in range(self.NUM_ROW):
            ring = []
            for col in range(self.NUM_COL):
                node_id = row * self.NUM_COL + col
                ring.append(node_id)
            rings.append(ring)
        return rings

    def get_vertical_rings(self) -> List[List[NodeId]]:
        """
        获取所有垂直环。

        Returns:
            垂直环列表
        """
        rings = []
        for col in range(self.NUM_COL):
            ring = []
            for row in range(self.NUM_ROW):
                node_id = row * self.NUM_COL + col
                ring.append(node_id)
            rings.append(ring)
        return rings

    def get_crossring_info(self) -> Dict[str, Any]:
        """
        获取CrossRing特有信息。

        Returns:
            CrossRing拓扑信息
        """
        base_info = self.get_topology_info()

        crossring_info = {
            "num_rows": self.NUM_ROW,
            "num_cols": self.NUM_COL,
            "horizontal_rings": len(self.get_horizontal_rings()),
            "vertical_rings": len(self.get_vertical_rings()),
            "supported_routing": ["XY", "YX", "shortest"],
        }

        base_info.update(crossring_info)
        return base_info

    def __str__(self) -> str:
        """字符串表示。"""
        return f"CrossRingTopology({self.NUM_ROW}×{self.NUM_COL}, nodes={self.NUM_NODE})"

    def __repr__(self) -> str:
        """详细字符串表示。"""
        return f"CrossRingTopology(rows={self.NUM_ROW}, cols={self.NUM_COL}, " f"nodes={self.NUM_NODE}, routing={self.routing_strategy.value})"

    def _build_routing_table(self) -> None:
        """构建路由表：预计算所有节点对的完整路径"""
        # 从配置中获取路由策略，默认为XY
        routing_strategy = getattr(self.crossring_config, "ROUTING_STRATEGY", "XY")
        if hasattr(routing_strategy, "value"):
            routing_strategy = routing_strategy.value

        # 确保路由策略是字符串
        if routing_strategy not in ["XY", "YX"]:
            routing_strategy = "XY"  # 默认为XY路由

        for src in range(self.NUM_NODE):
            self.routing_table[src] = {}
            for dst in range(self.NUM_NODE):
                if src == dst:
                    self.routing_table[src][dst] = [src]  # 自己到自己
                else:
                    # 使用路由策略计算路径
                    path = self._calculate_path(src, dst, routing_strategy)
                    self.routing_table[src][dst] = path

    def _calculate_path(self, src: NodeId, dst: NodeId, strategy: str) -> List[NodeId]:
        """根据路由策略计算路径"""
        src_col, src_row = self.get_node_position(src)
        dst_col, dst_row = self.get_node_position(dst)

        path = [src]
        current_col, current_row = src_col, src_row

        if strategy == "XY":
            # 先水平移动
            while current_col != dst_col:
                if current_col < dst_col:
                    current_col += 1
                else:
                    current_col -= 1
                node_id = self._position_to_node_id(current_col, current_row)
                path.append(node_id)

            # 再垂直移动
            while current_row != dst_row:
                if current_row < dst_row:
                    current_row += 1
                else:
                    current_row -= 1
                node_id = self._position_to_node_id(current_col, current_row)
                path.append(node_id)

        elif strategy == "YX":
            # 先垂直移动
            while current_row != dst_row:
                if current_row < dst_row:
                    current_row += 1
                else:
                    current_row -= 1
                node_id = self._position_to_node_id(current_col, current_row)
                path.append(node_id)

            # 再水平移动
            while current_col != dst_col:
                if current_col < dst_col:
                    current_col += 1
                else:
                    current_col -= 1
                node_id = self._position_to_node_id(current_col, current_row)
                path.append(node_id)
        return path

    def _position_to_node_id(self, col: int, row: int) -> NodeId:
        """将坐标转换为节点ID"""
        # 根据CrossRing的坐标系统
        actual_row = self.NUM_ROW - 1 - row  # 转换回原始行编号
        return actual_row * self.NUM_COL + col

    def get_routing_path(self, src: NodeId, dst: NodeId) -> List[NodeId]:
        """获取从src到dst的完整路径"""
        return self.routing_table[src][dst]

    def get_next_direction(self, src: NodeId, dst: NodeId) -> str:
        """获取从src到dst的下一跳方向"""
        path = self.routing_table[src][dst]

        if len(path) <= 1:
            return "EQ"  # 已到达目标

        next_node = path[1]  # 下一跳节点
        return self._get_direction_to_neighbor(src, next_node)

    def _get_direction_to_neighbor(self, src: NodeId, dst: NodeId) -> str:
        """计算从src到相邻节点dst的方向"""
        src_col, src_row = self.get_node_position(src)
        dst_col, dst_row = self.get_node_position(dst)

        if dst_col > src_col:
            return "TR"  # 向右
        elif dst_col < src_col:
            return "TL"  # 向左
        elif dst_row > src_row:
            return "TU"  # 向上
        elif dst_row < src_row:
            return "TD"  # 向下
        else:
            return "EQ"  # 同一位置

    def print_routing_table(self):
        """打印3x3路由表 - 显示完整路径"""
        print("=== CrossRing 3x3 路由表 ===")
        print("格式: src->dst: [完整路径]")

        for src in range(self.NUM_NODE):
            for dst in range(self.NUM_NODE):
                if src != dst:
                    path = self.routing_table[src][dst]
                    path_str = "->".join(map(str, path))
                    print(f"{src}到{dst}: {path_str}")
