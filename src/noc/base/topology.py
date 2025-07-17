"""
NoC拓扑抽象基类。

本模块定义了所有NoC拓扑必须实现的核心接口，包括：
- 拓扑结构管理
- 路由计算
- 性能监控
- 配置管理
"""

from abc import ABC, abstractmethod
from typing import List, Dict, Optional, Any
from collections import deque

from src.noc.utils.types import NodeId, Path, Position, RoutingStrategy, TopologyType, ValidationResult, AdjacencyMatrix
from .config import BaseNoCConfig


class BaseNoCTopology(ABC):
    """
    NoC拓扑抽象基类。

    定义所有NoC拓扑必须实现的核心接口，包括：
    - 拓扑结构管理
    - 路由计算
    - 性能监控
    - 配置管理
    """

    def __init__(self, config: BaseNoCConfig):
        """
        初始化NoC拓扑。

        Args:
            config: NoC配置对象
        """
        self.config = config
        self.num_nodes = config.num_nodes
        self.routing_strategy = config.routing_strategy
        self.topology_type = config.topology_type

        # 拓扑结构相关
        self._adjacency_matrix: Optional[AdjacencyMatrix] = None
        self._node_positions: Dict[NodeId, Position] = {}

        # 路由相关
        self._distance_matrix: Optional[List[List[int]]] = None

        # 缓存标志
        self._topology_built = False

        # 初始化拓扑
        self._initialize()

    def _initialize(self) -> None:
        """初始化拓扑结构。"""
        self.build_topology()
        self._topology_built = True

    # ========== 抽象方法 - 必须被子类实现 ==========

    @abstractmethod
    def build_topology(self) -> None:
        """构建拓扑结构（邻接矩阵、连接关系等）。"""
        pass

    @abstractmethod
    def get_neighbors(self, node_id: NodeId) -> List[NodeId]:
        """
        获取指定节点的邻居节点列表。

        Args:
            node_id: 节点ID

        Returns:
            邻居节点ID列表
        """
        pass

    @abstractmethod
    def calculate_shortest_path(self, src: NodeId, dst: NodeId) -> Path:
        """
        计算两点之间的最短路径。

        Args:
            src: 源节点ID
            dst: 目标节点ID

        Returns:
            最短路径（节点ID列表）
        """
        pass

    @abstractmethod
    def get_node_position(self, node_id: NodeId) -> Position:
        """
        获取节点的物理位置坐标。

        Args:
            node_id: 节点ID

        Returns:
            节点位置坐标
        """
        pass

    @abstractmethod
    def validate_topology(self) -> ValidationResult:
        """
        验证拓扑结构的有效性。

        Returns:
            验证结果（是否有效，错误消息）
        """
        pass

    # ========== 路由相关方法 ==========

    def calculate_route(self, src: NodeId, dst: NodeId, strategy: Optional[RoutingStrategy] = None) -> Path:
        """
        根据策略计算路由路径。

        Args:
            src: 源节点ID
            dst: 目标节点ID
            strategy: 路由策略（可选，默认使用配置中的策略）

        Returns:
            路由路径
        """
        if strategy is None:
            strategy = self.routing_strategy

        # 根据策略计算路径
        if strategy == RoutingStrategy.SHORTEST:
            return self.calculate_shortest_path(src, dst)
        elif strategy == RoutingStrategy.DETERMINISTIC:
            return self._calculate_xy_path(src, dst)
        elif strategy == RoutingStrategy.MINIMAL:
            return self._calculate_yx_path(src, dst)
        else:
            # 默认使用最短路径
            return self.calculate_shortest_path(src, dst)


    def _calculate_xy_path(self, src: NodeId, dst: NodeId) -> Path:
        """
        XY路由（先X轴后Y轴）。

        Args:
            src: 源节点ID
            dst: 目标节点ID

        Returns:
            XY路径
        """
        if src == dst:
            return [src]

        src_pos = self.get_node_position(src)
        dst_pos = self.get_node_position(dst)
        
        if len(src_pos) != 2 or len(dst_pos) != 2:
            # 如果不是2D网格，回退到最短路径
            return self.calculate_shortest_path(src, dst)

        src_x, src_y = src_pos
        dst_x, dst_y = dst_pos

        path = [src]
        current_x, current_y = src_x, src_y

        # 第一步：沿X轴移动到目标X坐标
        while current_x != dst_x:
            if current_x < dst_x:
                current_x += 1
            else:
                current_x -= 1
            
            # 查找对应的节点ID
            next_node = self._find_node_by_position((current_x, current_y))
            if next_node is not None:
                path.append(next_node)
            else:
                # 如果找不到对应位置的节点，回退到最短路径
                return self.calculate_shortest_path(src, dst)

        # 第二步：沿Y轴移动到目标Y坐标
        while current_y != dst_y:
            if current_y < dst_y:
                current_y += 1
            else:
                current_y -= 1
            
            # 查找对应的节点ID
            next_node = self._find_node_by_position((current_x, current_y))
            if next_node is not None:
                path.append(next_node)
            else:
                # 如果找不到对应位置的节点，回退到最短路径
                return self.calculate_shortest_path(src, dst)

        return path

    def _calculate_yx_path(self, src: NodeId, dst: NodeId) -> Path:
        """
        YX路由（先Y轴后X轴）。

        Args:
            src: 源节点ID
            dst: 目标节点ID

        Returns:
            YX路径
        """
        if src == dst:
            return [src]

        src_pos = self.get_node_position(src)
        dst_pos = self.get_node_position(dst)
        
        if len(src_pos) != 2 or len(dst_pos) != 2:
            # 如果不是2D网格，回退到最短路径
            return self.calculate_shortest_path(src, dst)

        src_x, src_y = src_pos
        dst_x, dst_y = dst_pos

        path = [src]
        current_x, current_y = src_x, src_y

        # 第一步：沿Y轴移动到目标Y坐标
        while current_y != dst_y:
            if current_y < dst_y:
                current_y += 1
            else:
                current_y -= 1
            
            # 查找对应的节点ID
            next_node = self._find_node_by_position((current_x, current_y))
            if next_node is not None:
                path.append(next_node)
            else:
                # 如果找不到对应位置的节点，回退到最短路径
                return self.calculate_shortest_path(src, dst)

        # 第二步：沿X轴移动到目标X坐标
        while current_x != dst_x:
            if current_x < dst_x:
                current_x += 1
            else:
                current_x -= 1
            
            # 查找对应的节点ID
            next_node = self._find_node_by_position((current_x, current_y))
            if next_node is not None:
                path.append(next_node)
            else:
                # 如果找不到对应位置的节点，回退到最短路径
                return self.calculate_shortest_path(src, dst)

        return path

    def _find_node_by_position(self, position: Position) -> Optional[NodeId]:
        """
        根据位置查找节点ID。

        Args:
            position: 节点位置坐标

        Returns:
            节点ID，如果找不到则返回None
        """
        for node_id, node_pos in self._node_positions.items():
            if node_pos == position:
                return node_id
        return None


    # ========== 拓扑分析方法 ==========

    def get_adjacency_matrix(self) -> AdjacencyMatrix:
        """
        获取邻接矩阵。

        Returns:
            邻接矩阵
        """
        if self._adjacency_matrix is None:
            self.build_topology()
        return self._adjacency_matrix

    def get_distance_matrix(self) -> List[List[int]]:
        """
        获取距离矩阵。

        Returns:
            距离矩阵
        """
        if self._distance_matrix is None:
            self._compute_distance_matrix()
        return self._distance_matrix

    def _compute_distance_matrix(self) -> None:
        """计算距离矩阵（Floyd-Warshall算法）。"""
        n = self.num_nodes
        dist = [[float("inf")] * n for _ in range(n)]

        # 初始化
        for i in range(n):
            dist[i][i] = 0

        adj_matrix = self.get_adjacency_matrix()
        for i in range(n):
            for j in range(n):
                if adj_matrix[i][j] == 1:
                    dist[i][j] = 1

        # Floyd-Warshall
        for k in range(n):
            for i in range(n):
                for j in range(n):
                    if dist[i][k] + dist[k][j] < dist[i][j]:
                        dist[i][j] = dist[i][k] + dist[k][j]

        # 转换为整数矩阵
        self._distance_matrix = [[int(d) if d != float("inf") else -1 for d in row] for row in dist]

    def get_hop_count(self, src: NodeId, dst: NodeId) -> int:
        """
        计算两点之间的跳数。

        Args:
            src: 源节点ID
            dst: 目标节点ID

        Returns:
            跳数
        """
        if src == dst:
            return 0

        dist_matrix = self.get_distance_matrix()
        if 0 <= src < len(dist_matrix) and 0 <= dst < len(dist_matrix[0]):
            return dist_matrix[src][dst] if dist_matrix[src][dst] != -1 else float("inf")

        # 备选方案：计算最短路径长度
        path = self.calculate_shortest_path(src, dst)
        return len(path) - 1 if len(path) > 1 else float("inf")


    def is_connected(self) -> bool:
        """
        检查拓扑是否连通。

        Returns:
            是否连通
        """
        if self.num_nodes == 0:
            return True

        # 使用BFS检查连通性
        visited = [False] * self.num_nodes
        queue = deque([0])
        visited[0] = True
        visited_count = 1

        while queue:
            node = queue.popleft()
            for neighbor in self.get_neighbors(node):
                if not visited[neighbor]:
                    visited[neighbor] = True
                    queue.append(neighbor)
                    visited_count += 1

        return visited_count == self.num_nodes


    # ========== 配置和信息方法 ==========

    def get_topology_info(self) -> Dict[str, Any]:
        """
        获取拓扑基本信息。

        Returns:
            拓扑信息字典
        """
        return {
            "topology_type": self.topology_type.value,
            "num_nodes": self.num_nodes,
            "is_connected": self.is_connected(),
            "routing_strategy": self.routing_strategy.value,
        }



    def __str__(self) -> str:
        """字符串表示。"""
        return f"{self.__class__.__name__}(nodes={self.num_nodes}, type={self.topology_type.value})"

    def __repr__(self) -> str:
        """详细字符串表示。"""
        return f"{self.__class__.__name__}({self.get_topology_info()})"
