"""
Mesh拓扑实现。

本模块实现标准的2D Mesh拓扑，包括：
- XY路由算法
- 最短路径路由
- 拓扑结构管理
- 性能监控
"""

from typing import List, Dict, Tuple, Optional, Set, Any
import math
from collections import deque

from ..base.topology import BaseNoCTopology
from src.noc.utils.types import NodeId, Path, Position, RoutingStrategy, ValidationResult
from .config import MeshConfig


class MeshTopology(BaseNoCTopology):
    """
    2D Mesh拓扑实现。

    实现标准的2D Mesh网络拓扑，支持：
    - XY路由算法（维序路由）
    - 最短路径路由
    - 曼哈顿距离计算
    - 拓扑验证
    """

    def __init__(self, config: MeshConfig):
        """
        初始化Mesh拓扑。

        Args:
            config: Mesh配置对象
        """
        super().__init__(config)
        self.mesh_config = config
        self.rows = config.rows
        self.cols = config.cols

        # 建立拓扑结构
        self._build_topology()

        # 初始化路由表
        self._initialize_routing_tables()

    def _build_topology(self) -> None:
        """构建Mesh拓扑结构"""
        # 构建邻接矩阵
        self._build_adjacency_matrix()

        # 设置节点位置
        self._set_node_positions()

        # 构建链路映射
        self._build_link_map()

    def _build_adjacency_matrix(self) -> None:
        """构建邻接矩阵"""
        n = self.num_nodes
        self._adjacency_matrix = [[0 for _ in range(n)] for _ in range(n)]

        for node_id in range(n):
            neighbors = self.mesh_config.get_neighbors(node_id)
            for neighbor in neighbors:
                self._adjacency_matrix[node_id][neighbor] = 1
                self._adjacency_matrix[neighbor][node_id] = 1

    def _set_node_positions(self) -> None:
        """设置节点位置"""
        self._node_positions = {}
        for node_id in range(self.num_nodes):
            row, col = self.mesh_config.get_node_position(node_id)
            self._node_positions[node_id] = (row, col)

    def _build_link_map(self) -> None:
        """构建链路映射"""
        self._link_map = {}
        link_id = 0

        for node_id in range(self.num_nodes):
            neighbors = self.mesh_config.get_neighbors(node_id)
            for neighbor in neighbors:
                if node_id < neighbor:  # 避免重复链路
                    link_key = f"link_{link_id}"
                    self._link_map[link_key] = {
                        "source": node_id,
                        "destination": neighbor,
                        "bandwidth": self.config.link_bandwidth,
                        "latency": self.mesh_config.mesh_config.LINK_LATENCY,
                        "width": self.mesh_config.mesh_config.LINK_WIDTH,
                    }
                    link_id += 1

    def _initialize_routing_tables(self) -> None:
        """初始化路由表"""
        # XY路由表
        if RoutingStrategy.XY not in self._routing_tables:
            self._routing_tables[RoutingStrategy.XY] = {}

        # 最短路径路由表
        if RoutingStrategy.SHORTEST not in self._routing_tables:
            self._routing_tables[RoutingStrategy.SHORTEST] = {}

        # 为每个源节点计算到所有目标节点的路由
        for src in range(self.num_nodes):
            self._routing_tables[RoutingStrategy.XY][src] = {}
            self._routing_tables[RoutingStrategy.SHORTEST][src] = {}

            for dst in range(self.num_nodes):
                # XY路由
                xy_path = self._calculate_xy_route(src, dst)
                self._routing_tables[RoutingStrategy.XY][src][dst] = xy_path

                # 最短路径路由
                shortest_path = self._calculate_shortest_path(src, dst)
                self._routing_tables[RoutingStrategy.SHORTEST][src][dst] = shortest_path

    def _calculate_xy_route(self, src: NodeId, dst: NodeId) -> Path:
        """
        计算XY路由路径。

        Args:
            src: 源节点ID
            dst: 目标节点ID

        Returns:
            路由路径
        """
        if src == dst:
            return [src]

        path = [src]
        current = src
        src_row, src_col = self.mesh_config.get_node_position(src)
        dst_row, dst_col = self.mesh_config.get_node_position(dst)

        # 首先在X方向移动
        while src_col != dst_col:
            if src_col < dst_col:
                src_col += 1  # 向东移动
            else:
                src_col -= 1  # 向西移动

            current = self.mesh_config.get_node_id(src_row, src_col)
            path.append(current)

        # 然后在Y方向移动
        while src_row != dst_row:
            if src_row < dst_row:
                src_row += 1  # 向南移动
            else:
                src_row -= 1  # 向北移动

            current = self.mesh_config.get_node_id(src_row, src_col)
            path.append(current)

        return path

    def _calculate_shortest_path(self, src: NodeId, dst: NodeId) -> Path:
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

        # BFS队列：(当前节点, 路径)
        queue = deque([(src, [src])])
        visited = {src}

        while queue:
            current, path = queue.popleft()

            # 检查所有邻居
            neighbors = self.mesh_config.get_neighbors(current)
            for neighbor in neighbors:
                if neighbor == dst:
                    return path + [neighbor]

                if neighbor not in visited:
                    visited.add(neighbor)
                    queue.append((neighbor, path + [neighbor]))

        # 如果没有找到路径，返回空路径
        return []

    def get_route(self, src: NodeId, dst: NodeId, strategy: RoutingStrategy = None) -> Path:
        """
        获取从源到目标的路由路径。

        Args:
            src: 源节点ID
            dst: 目标节点ID
            strategy: 路由策略

        Returns:
            路由路径
        """
        if strategy is None:
            strategy = self.routing_strategy

        if strategy in self._routing_tables and src in self._routing_tables[strategy]:
            return self._routing_tables[strategy][src].get(dst, [])

        # 如果没有预计算的路由，动态计算
        if strategy == RoutingStrategy.XY:
            return self._calculate_xy_route(src, dst)
        elif strategy == RoutingStrategy.SHORTEST:
            return self._calculate_shortest_path(src, dst)
        else:
            return self._calculate_xy_route(src, dst)  # 默认使用XY路由

    def get_distance(self, src: NodeId, dst: NodeId) -> int:
        """
        获取两节点间的距离（跳数）。

        Args:
            src: 源节点ID
            dst: 目标节点ID

        Returns:
            距离（跳数）
        """
        return self.mesh_config.calculate_manhattan_distance(src, dst)

    def get_manhattan_distance(self, src: NodeId, dst: NodeId) -> int:
        """
        获取曼哈顿距离。

        Args:
            src: 源节点ID
            dst: 目标节点ID

        Returns:
            曼哈顿距离
        """
        return self.mesh_config.calculate_manhattan_distance(src, dst)

    def get_neighbors(self, node_id: NodeId) -> List[NodeId]:
        """
        获取节点的邻居。

        Args:
            node_id: 节点ID

        Returns:
            邻居节点列表
        """
        return self.mesh_config.get_neighbors(node_id)

    def get_node_position(self, node_id: NodeId) -> Position:
        """
        获取节点位置。

        Args:
            node_id: 节点ID

        Returns:
            节点位置 (row, col)
        """
        return self._node_positions.get(node_id, (0, 0))

    def get_adjacency_matrix(self) -> List[List[int]]:
        """
        获取邻接矩阵。

        Returns:
            邻接矩阵
        """
        return self._adjacency_matrix

    def is_connected(self, src: NodeId, dst: NodeId) -> bool:
        """
        检查两个节点是否连通。

        Args:
            src: 源节点ID
            dst: 目标节点ID

        Returns:
            是否连通
        """
        path = self.get_route(src, dst, RoutingStrategy.SHORTEST)
        return len(path) > 0

    def get_topology_info(self) -> Dict[str, Any]:
        """
        获取拓扑信息。

        Returns:
            拓扑信息字典
        """
        return {
            "topology_type": "Mesh",
            "dimensions": f"{self.rows}x{self.cols}",
            "num_nodes": self.num_nodes,
            "num_links": len(self._link_map),
            "routing_strategies": list(self._routing_tables.keys()),
            "node_positions": self._node_positions,
            "adjacency_matrix": self._adjacency_matrix,
            "link_map": self._link_map,
        }

    def validate_topology(self) -> ValidationResult:
        """
        验证拓扑结构。

        Returns:
            ValidationResult: (是否有效, 错误信息)
        """
        errors = []

        # 检查节点数
        if self.num_nodes != self.rows * self.cols:
            errors.append(f"节点数不匹配: {self.num_nodes} != {self.rows} * {self.cols}")

        # 检查邻接矩阵
        if not self._adjacency_matrix:
            errors.append("邻接矩阵未初始化")
        elif len(self._adjacency_matrix) != self.num_nodes:
            errors.append(f"邻接矩阵大小不匹配: {len(self._adjacency_matrix)} != {self.num_nodes}")

        # 检查连通性
        for src in range(self.num_nodes):
            for dst in range(self.num_nodes):
                if src != dst and not self.is_connected(src, dst):
                    errors.append(f"节点 {src} 和 {dst} 不连通")
                    break
            if errors:
                break

        # 检查节点位置
        if len(self._node_positions) != self.num_nodes:
            errors.append(f"节点位置数不匹配: {len(self._node_positions)} != {self.num_nodes}")

        # 检查路由表
        for strategy in [RoutingStrategy.XY, RoutingStrategy.SHORTEST]:
            if strategy not in self._routing_tables:
                errors.append(f"缺少路由策略 {strategy} 的路由表")

        if errors:
            return False, "; ".join(errors)

        return True, None

    def get_path_length(self, path: Path) -> int:
        """
        获取路径长度（跳数）。

        Args:
            path: 路径

        Returns:
            路径长度
        """
        return max(0, len(path) - 1)

    def get_all_paths(self, src: NodeId, dst: NodeId, max_length: int = None) -> List[Path]:
        """
        获取两个节点间的所有路径。

        Args:
            src: 源节点ID
            dst: 目标节点ID
            max_length: 最大路径长度

        Returns:
            所有路径列表
        """
        if max_length is None:
            max_length = self.num_nodes

        all_paths = []

        def dfs(current: NodeId, target: NodeId, path: Path, visited: Set[NodeId]):
            if len(path) > max_length:
                return

            if current == target:
                all_paths.append(path.copy())
                return

            neighbors = self.get_neighbors(current)
            for neighbor in neighbors:
                if neighbor not in visited:
                    visited.add(neighbor)
                    path.append(neighbor)
                    dfs(neighbor, target, path, visited)
                    path.pop()
                    visited.remove(neighbor)

        visited = {src}
        dfs(src, dst, [src], visited)

        return all_paths

    def get_node_degree(self, node_id: NodeId) -> int:
        """
        获取节点度数。

        Args:
            node_id: 节点ID

        Returns:
            节点度数
        """
        return len(self.get_neighbors(node_id))

    def get_average_degree(self) -> float:
        """
        获取平均节点度数。

        Returns:
            平均度数
        """
        total_degree = sum(self.get_node_degree(node) for node in range(self.num_nodes))
        return total_degree / self.num_nodes if self.num_nodes > 0 else 0.0

    def get_diameter(self) -> int:
        """
        获取网络直径（最大最短路径长度）。

        Returns:
            网络直径
        """
        max_distance = 0

        for src in range(self.num_nodes):
            for dst in range(self.num_nodes):
                if src != dst:
                    distance = self.get_distance(src, dst)
                    max_distance = max(max_distance, distance)

        return max_distance

    # ========== 实现BaseNoCTopology的抽象方法 ==========

    def build_topology(self) -> None:
        """构建拓扑结构（实现抽象方法）"""
        # 在__init__中已经调用了_build_topology
        pass

    def get_neighbors(self, node_id: NodeId) -> List[NodeId]:
        """获取指定节点的邻居节点列表（实现抽象方法）"""
        return self.mesh_config.get_neighbors(node_id)

    def calculate_shortest_path(self, src: NodeId, dst: NodeId) -> Path:
        """计算两点之间的最短路径（实现抽象方法）"""
        return self._calculate_shortest_path(src, dst)
