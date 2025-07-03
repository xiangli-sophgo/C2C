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
import logging

from ..base.topology import BaseNoCTopology
from ..types import NodeId, Path, Position, ValidationResult, AdjacencyMatrix, RoutingStrategy, TopologyType, LinkId, MetricsDict
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
        # 验证配置
        is_valid, error_msg = config.validate_config()
        if not is_valid:
            raise ValueError(f"CrossRing配置无效: {error_msg}")

        # 设置拓扑类型
        config.topology_type = TopologyType.CROSSRING

        # CrossRing特有属性
        self.num_rows = config.num_row
        self.num_cols = config.num_col
        self.crossring_config = config

        # 初始化日志
        self._logger = logging.getLogger(f"{self.__class__.__name__}[{self.num_rows}×{self.num_cols}]")
        self._logger.info(f"初始化CrossRing拓扑: {self.num_rows}×{self.num_cols}")

        super().__init__(config)

        # 路由策略映射
        self._routing_strategy_map = {
            RoutingStrategy.SHORTEST: self._calculate_shortest_path_bfs,
            RoutingStrategy.DETERMINISTIC: self._calculate_hv_path,
            RoutingStrategy.MINIMAL: self._calculate_vh_path,
            RoutingStrategy.ADAPTIVE: self._calculate_adaptive_crossring_path,
            RoutingStrategy.LOAD_BALANCED: self._calculate_load_balanced_crossring_path,
        }

    def build_topology(self) -> None:
        """构建CrossRing拓扑结构。"""
        self._logger.info("开始构建CrossRing拓扑结构")

        # 创建邻接矩阵
        self._adjacency_matrix = create_crossring_adjacency_matrix(self.num_rows, self.num_cols)

        # 验证邻接矩阵
        is_valid, error_msg = validate_adjacency_matrix(self._adjacency_matrix)
        if not is_valid:
            raise ValueError(f"生成的邻接矩阵无效: {error_msg}")

        # 建立节点位置映射
        self._build_node_positions()

        self._logger.info("CrossRing拓扑构建完成")

    def _build_node_positions(self) -> None:
        """建立节点位置映射。"""
        self._node_positions.clear()

        for node_id in range(self.num_nodes):
            row, col = divmod(node_id, self.num_cols)
            self._node_positions[node_id] = (row, col)

    def get_neighbors(self, node_id: NodeId) -> List[NodeId]:
        """
        获取指定节点的邻居节点列表。

        Args:
            node_id: 节点ID

        Returns:
            邻居节点ID列表
        """
        if not (0 <= node_id < self.num_nodes):
            raise ValueError(f"节点ID {node_id} 超出范围 [0, {self.num_nodes-1}]")

        neighbors = []
        adj_matrix = self.get_adjacency_matrix()

        for i, connected in enumerate(adj_matrix[node_id]):
            if connected == 1:
                neighbors.append(i)

        return neighbors

    def get_node_position(self, node_id: NodeId) -> Position:
        """
        获取节点的物理位置坐标。

        Args:
            node_id: 节点ID

        Returns:
            节点位置坐标 (row, col)
        """
        if not (0 <= node_id < self.num_nodes):
            raise ValueError(f"节点ID {node_id} 超出范围 [0, {self.num_nodes-1}]")

        return self._node_positions[node_id]

    def calculate_shortest_path(self, src: NodeId, dst: NodeId) -> Path:
        """
        计算两点之间的最短路径（使用BFS）。

        Args:
            src: 源节点ID
            dst: 目标节点ID

        Returns:
            最短路径（节点ID列表）
        """
        return self._calculate_shortest_path_bfs(src, dst)

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

    def calculate_hv_path(self, src: NodeId, dst: NodeId) -> Path:
        """
        计算水平优先(HV)路径。

        Args:
            src: 源节点ID
            dst: 目标节点ID

        Returns:
            HV路径
        """
        return self._calculate_hv_path(src, dst)

    def _calculate_hv_path(self, src: NodeId, dst: NodeId) -> Path:
        """
        水平优先(HV)路由：先水平移动到目标列，再垂直移动到目标行。

        Args:
            src: 源节点ID
            dst: 目标节点ID

        Returns:
            HV路径
        """
        if src == dst:
            return [src]

        src_row, src_col = self.get_node_position(src)
        dst_row, dst_col = self.get_node_position(dst)

        path = [src]
        current_row, current_col = src_row, src_col

        # 第一步：水平移动到目标列
        while current_col != dst_col:
            if current_col < dst_col:
                current_col += 1  # 向右移动
            else:
                current_col -= 1  # 向左移动

            next_node = current_row * self.num_cols + current_col
            path.append(next_node)

        # 第二步：垂直移动到目标行
        while current_row != dst_row:
            if current_row < dst_row:
                current_row += 1  # 向下移动
            else:
                current_row -= 1  # 向上移动

            next_node = current_row * self.num_cols + current_col
            path.append(next_node)

        return path

    def calculate_vh_path(self, src: NodeId, dst: NodeId) -> Path:
        """
        计算垂直优先(VH)路径。

        Args:
            src: 源节点ID
            dst: 目标节点ID

        Returns:
            VH路径
        """
        return self._calculate_vh_path(src, dst)

    def _calculate_vh_path(self, src: NodeId, dst: NodeId) -> Path:
        """
        垂直优先(VH)路由：先垂直移动到目标行，再水平移动到目标列。

        Args:
            src: 源节点ID
            dst: 目标节点ID

        Returns:
            VH路径
        """
        if src == dst:
            return [src]

        src_row, src_col = self.get_node_position(src)
        dst_row, dst_col = self.get_node_position(dst)

        path = [src]
        current_row, current_col = src_row, src_col

        # 第一步：垂直移动到目标行
        while current_row != dst_row:
            if current_row < dst_row:
                current_row += 1  # 向下移动
            else:
                current_row -= 1  # 向上移动

            next_node = current_row * self.num_cols + current_col
            path.append(next_node)

        # 第二步：水平移动到目标列
        while current_col != dst_col:
            if current_col < dst_col:
                current_col += 1  # 向右移动
            else:
                current_col -= 1  # 向左移动

            next_node = current_row * self.num_cols + current_col
            path.append(next_node)

        return path

    def _calculate_adaptive_crossring_path(self, src: NodeId, dst: NodeId) -> Path:
        """
        自适应CrossRing路由：根据网络状态选择HV或VH路径。

        Args:
            src: 源节点ID
            dst: 目标节点ID

        Returns:
            自适应路径
        """
        # 计算HV和VH路径
        hv_path = self._calculate_hv_path(src, dst)
        vh_path = self._calculate_vh_path(src, dst)

        # 选择利用率较低的路径
        hv_utilization = self._calculate_path_utilization(hv_path)
        vh_utilization = self._calculate_path_utilization(vh_path)

        if hv_utilization <= vh_utilization:
            self._logger.debug(f"自适应路由选择HV路径: {src} -> {dst}")
            return hv_path
        else:
            self._logger.debug(f"自适应路由选择VH路径: {src} -> {dst}")
            return vh_path

    def _calculate_load_balanced_crossring_path(self, src: NodeId, dst: NodeId) -> Path:
        """
        负载均衡CrossRing路由：选择最低负载的路径。

        Args:
            src: 源节点ID
            dst: 目标节点ID

        Returns:
            负载均衡路径
        """
        # 计算所有可能的最短路径
        all_paths = self._find_all_crossring_paths(src, dst)

        if not all_paths:
            return self._calculate_shortest_path_bfs(src, dst)

        # 选择利用率最低的路径
        best_path = all_paths[0]
        min_utilization = self._calculate_path_utilization(best_path)

        for path in all_paths[1:]:
            utilization = self._calculate_path_utilization(path)
            if utilization < min_utilization:
                min_utilization = utilization
                best_path = path

        return best_path

    def _find_all_crossring_paths(self, src: NodeId, dst: NodeId) -> List[Path]:
        """
        查找所有CrossRing最短路径。

        Args:
            src: 源节点ID
            dst: 目标节点ID

        Returns:
            所有最短路径列表
        """
        if src == dst:
            return [[src]]

        # 对于CrossRing，通常只有HV和VH两种最短路径
        hv_path = self._calculate_hv_path(src, dst)
        vh_path = self._calculate_vh_path(src, dst)

        paths = []
        if hv_path:
            paths.append(hv_path)
        if vh_path and vh_path != hv_path:
            paths.append(vh_path)

        return paths

    def _calculate_path_utilization(self, path: Path) -> float:
        """
        计算路径的平均利用率。

        Args:
            path: 路径

        Returns:
            平均利用率
        """
        if len(path) < 2:
            return 0.0

        total_utilization = 0.0
        link_count = 0

        for i in range(len(path) - 1):
            link = (path[i], path[i + 1])
            utilization = self._get_link_utilization(link)
            total_utilization += utilization
            link_count += 1

        return total_utilization / link_count if link_count > 0 else 0.0

    def get_topology_efficiency(self) -> float:
        """
        计算拓扑效率指标。

        Returns:
            拓扑效率（0.0-1.0）
        """
        if self.num_nodes <= 1:
            return 1.0

        # 计算平均路径长度
        total_distance = 0
        path_count = 0

        for src in range(self.num_nodes):
            for dst in range(self.num_nodes):
                if src != dst:
                    hop_count = self.get_hop_count(src, dst)
                    if hop_count != float("inf"):
                        total_distance += hop_count
                        path_count += 1

        if path_count == 0:
            return 0.0

        average_path_length = total_distance / path_count

        # 效率 = 1 / 平均路径长度
        return 1.0 / average_path_length if average_path_length > 0 else 0.0

    def calculate_load_distribution(self) -> Dict[str, Any]:
        """
        计算负载分布分析。

        Returns:
            负载分布数据
        """
        if not self._link_metrics:
            return {
                "total_links": 0,
                "active_links": 0,
                "load_variance": 0.0,
                "load_distribution": {},
                "utilization_stats": {
                    "mean": 0.0,
                    "std": 0.0,
                    "min": 0.0,
                    "max": 0.0,
                },
            }

        utilizations = [metrics.get("utilization", 0.0) for metrics in self._link_metrics.values()]

        # 计算负载分布统计
        active_links = len([u for u in utilizations if u > 0])
        load_variance = np.var(utilizations) if utilizations else 0.0

        # 负载分布直方图
        load_bins = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
        load_distribution = {}

        for i in range(len(load_bins) - 1):
            bin_name = f"{load_bins[i]:.1f}-{load_bins[i+1]:.1f}"
            count = len([u for u in utilizations if load_bins[i] <= u < load_bins[i + 1]])
            load_distribution[bin_name] = count

        return {
            "total_links": len(self._link_metrics),
            "active_links": active_links,
            "load_variance": load_variance,
            "load_distribution": load_distribution,
            "utilization_stats": {
                "mean": np.mean(utilizations) if utilizations else 0.0,
                "std": np.std(utilizations) if utilizations else 0.0,
                "min": np.min(utilizations) if utilizations else 0.0,
                "max": np.max(utilizations) if utilizations else 0.0,
            },
        }

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
        if len(self._adjacency_matrix) != self.num_nodes:
            return False, f"邻接矩阵大小不匹配: 期望{self.num_nodes}，实际{len(self._adjacency_matrix)}"

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
        for row in range(self.num_rows):
            ring = []
            for col in range(self.num_cols):
                node_id = row * self.num_cols + col
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
        for col in range(self.num_cols):
            ring = []
            for row in range(self.num_rows):
                node_id = row * self.num_cols + col
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
            "num_rows": self.num_rows,
            "num_cols": self.num_cols,
            "horizontal_rings": len(self.get_horizontal_rings()),
            "vertical_rings": len(self.get_vertical_rings()),
            "topology_efficiency": self.get_topology_efficiency(),
            "supported_routing": ["HV", "VH", "adaptive", "load_balanced"],
            "load_distribution": self.calculate_load_distribution(),
        }

        base_info.update(crossring_info)
        return base_info

    def __str__(self) -> str:
        """字符串表示。"""
        return f"CrossRingTopology({self.num_rows}×{self.num_cols}, nodes={self.num_nodes})"

    def __repr__(self) -> str:
        """详细字符串表示。"""
        return f"CrossRingTopology(rows={self.num_rows}, cols={self.num_cols}, " f"nodes={self.num_nodes}, routing={self.routing_strategy.value})"
