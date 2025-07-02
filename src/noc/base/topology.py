"""
NoC拓扑抽象基类。

本模块定义了所有NoC拓扑必须实现的核心接口，包括：
- 拓扑结构管理
- 路由计算
- 性能监控
- 配置管理
"""

from abc import ABC, abstractmethod
from typing import List, Dict, Tuple, Optional, Set, Any, Union
import copy
import heapq
from collections import defaultdict, deque

from ..types import (
    NodeId, Path, Position, NoCMetrics, RoutingStrategy, TopologyType,
    ValidationResult, AdjacencyMatrix, LinkId, MetricsDict, ConfigDict
)
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
        self._link_map: Dict[LinkId, Dict[str, Any]] = {}
        
        # 路由相关
        self._routing_tables: Dict[RoutingStrategy, Dict[NodeId, Dict[NodeId, Path]]] = {}
        self._distance_matrix: Optional[List[List[int]]] = None
        
        # 性能监控相关
        self._metrics: NoCMetrics = NoCMetrics()
        self._link_metrics: Dict[LinkId, MetricsDict] = {}
        self._node_metrics: Dict[NodeId, MetricsDict] = {}
        
        # 缓存标志
        self._topology_built = False
        self._routes_computed = False
        
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
    
    def calculate_route(self, src: NodeId, dst: NodeId, 
                       strategy: Optional[RoutingStrategy] = None) -> Path:
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
        
        # 检查缓存
        if strategy in self._routing_tables and \
           src in self._routing_tables[strategy] and \
           dst in self._routing_tables[strategy][src]:
            return self._routing_tables[strategy][src][dst]
        
        # 根据策略计算路径
        if strategy == RoutingStrategy.SHORTEST:
            path = self.calculate_shortest_path(src, dst)
        elif strategy == RoutingStrategy.LOAD_BALANCED:
            path = self._calculate_load_balanced_path(src, dst)
        elif strategy == RoutingStrategy.ADAPTIVE:
            path = self._calculate_adaptive_path(src, dst)
        elif strategy == RoutingStrategy.DETERMINISTIC:
            path = self._calculate_deterministic_path(src, dst)
        elif strategy == RoutingStrategy.MINIMAL:
            path = self._calculate_minimal_path(src, dst)
        else:
            # 默认使用最短路径
            path = self.calculate_shortest_path(src, dst)
        
        # 缓存结果
        if strategy not in self._routing_tables:
            self._routing_tables[strategy] = {}
        if src not in self._routing_tables[strategy]:
            self._routing_tables[strategy][src] = {}
        self._routing_tables[strategy][src][dst] = path
        
        return path
    
    def _calculate_load_balanced_path(self, src: NodeId, dst: NodeId) -> Path:
        """
        负载均衡路由（默认实现，子类可重写）。
        
        Args:
            src: 源节点ID
            dst: 目标节点ID
            
        Returns:
            负载均衡路径
        """
        # 基础实现：考虑链路利用率选择路径
        all_paths = self._find_all_minimal_paths(src, dst)
        if not all_paths:
            return self.calculate_shortest_path(src, dst)
        
        # 选择利用率最低的路径
        best_path = all_paths[0]
        min_max_utilization = float('inf')
        
        for path in all_paths:
            max_utilization = 0
            for i in range(len(path) - 1):
                link = (path[i], path[i + 1])
                utilization = self._get_link_utilization(link)
                max_utilization = max(max_utilization, utilization)
            
            if max_utilization < min_max_utilization:
                min_max_utilization = max_utilization
                best_path = path
        
        return best_path
    
    def _calculate_adaptive_path(self, src: NodeId, dst: NodeId) -> Path:
        """
        自适应路由（默认实现，子类可重写）。
        
        Args:
            src: 源节点ID
            dst: 目标节点ID
            
        Returns:
            自适应路径
        """
        # 基础实现：根据当前网络状态动态选择路径
        if self._is_congested():
            return self._calculate_load_balanced_path(src, dst)
        else:
            return self.calculate_shortest_path(src, dst)
    
    def _calculate_deterministic_path(self, src: NodeId, dst: NodeId) -> Path:
        """
        确定性路由。
        
        Args:
            src: 源节点ID
            dst: 目标节点ID
            
        Returns:
            确定性路径
        """
        # 总是选择相同的路径（通常是最短路径）
        return self.calculate_shortest_path(src, dst)
    
    def _calculate_minimal_path(self, src: NodeId, dst: NodeId) -> Path:
        """
        最小路径路由。
        
        Args:
            src: 源节点ID
            dst: 目标节点ID
            
        Returns:
            最小路径
        """
        # 与最短路径相同
        return self.calculate_shortest_path(src, dst)
    
    def _find_all_minimal_paths(self, src: NodeId, dst: NodeId) -> List[Path]:
        """
        查找所有最小路径。
        
        Args:
            src: 源节点ID
            dst: 目标节点ID
            
        Returns:
            所有最小路径列表
        """
        if src == dst:
            return [[src]]
        
        shortest_distance = self.get_hop_count(src, dst)
        all_paths = []
        
        def dfs(current: NodeId, target: NodeId, path: Path, remaining_hops: int):
            if current == target and remaining_hops == 0:
                all_paths.append(path.copy())
                return
            
            if remaining_hops <= 0:
                return
            
            for neighbor in self.get_neighbors(current):
                if neighbor not in path:  # 避免环路
                    path.append(neighbor)
                    dfs(neighbor, target, path, remaining_hops - 1)
                    path.pop()
        
        dfs(src, dst, [src], shortest_distance)
        return all_paths
    
    def _get_link_utilization(self, link: LinkId) -> float:
        """
        获取链路利用率。
        
        Args:
            link: 链路ID
            
        Returns:
            链路利用率（0.0-1.0）
        """
        if link in self._link_metrics:
            return self._link_metrics[link].get('utilization', 0.0)
        return 0.0
    
    def _is_congested(self) -> bool:
        """
        检查网络是否拥塞。
        
        Returns:
            是否拥塞
        """
        avg_utilization = self._metrics.average_link_utilization
        return avg_utilization > 0.7  # 阈值可配置
    
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
        dist = [[float('inf')] * n for _ in range(n)]
        
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
        self._distance_matrix = [[int(d) if d != float('inf') else -1 
                                 for d in row] for row in dist]
    
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
            return dist_matrix[src][dst] if dist_matrix[src][dst] != -1 else float('inf')
        
        # 备选方案：计算最短路径长度
        path = self.calculate_shortest_path(src, dst)
        return len(path) - 1 if len(path) > 1 else float('inf')
    
    def get_average_hop_count(self) -> float:
        """
        计算平均跳数。
        
        Returns:
            平均跳数
        """
        total_hops = 0
        total_pairs = 0
        
        for src in range(self.num_nodes):
            for dst in range(self.num_nodes):
                if src != dst:
                    hop_count = self.get_hop_count(src, dst)
                    if hop_count != float('inf'):
                        total_hops += hop_count
                        total_pairs += 1
        
        return total_hops / total_pairs if total_pairs > 0 else 0.0
    
    def get_diameter(self) -> int:
        """
        计算网络直径（最大跳数）。
        
        Returns:
            网络直径
        """
        max_hops = 0
        for src in range(self.num_nodes):
            for dst in range(self.num_nodes):
                if src != dst:
                    hop_count = self.get_hop_count(src, dst)
                    if hop_count != float('inf'):
                        max_hops = max(max_hops, hop_count)
        return max_hops
    
    def get_bisection_bandwidth(self) -> float:
        """
        计算二分带宽。
        
        Returns:
            二分带宽
        """
        # 基础实现：简单估算
        # 子类应该根据具体拓扑重写此方法
        total_links = sum(sum(row) for row in self.get_adjacency_matrix()) // 2
        return total_links * self.config.link_bandwidth / 2
    
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
    
    # ========== 性能监控方法 ==========
    
    def update_metrics(self, new_metrics: Dict[str, float]) -> None:
        """
        更新性能指标。
        
        Args:
            new_metrics: 新的性能指标字典
        """
        for key, value in new_metrics.items():
            if hasattr(self._metrics, key):
                setattr(self._metrics, key, value)
            else:
                # 存储在自定义指标中
                self._metrics.custom_metrics[key] = value
    
    def update_link_metrics(self, link: LinkId, metrics: MetricsDict) -> None:
        """
        更新链路指标。
        
        Args:
            link: 链路ID
            metrics: 指标字典
        """
        if link not in self._link_metrics:
            self._link_metrics[link] = {}
        self._link_metrics[link].update(metrics)
    
    def update_node_metrics(self, node_id: NodeId, metrics: MetricsDict) -> None:
        """
        更新节点指标。
        
        Args:
            node_id: 节点ID
            metrics: 指标字典
        """
        if node_id not in self._node_metrics:
            self._node_metrics[node_id] = {}
        self._node_metrics[node_id].update(metrics)
    
    def get_performance_metrics(self) -> NoCMetrics:
        """
        获取当前性能指标。
        
        Returns:
            性能指标对象
        """
        # 更新基础指标
        self._update_basic_metrics()
        return copy.deepcopy(self._metrics)
    
    def _update_basic_metrics(self) -> None:
        """更新基础性能指标。"""
        # 更新网络拓扑指标
        self._metrics.average_hop_count = self.get_average_hop_count()
        self._metrics.network_diameter = self.get_diameter()
        self._metrics.bisection_bandwidth = self.get_bisection_bandwidth()
        
        # 更新链路利用率指标
        if self._link_metrics:
            utilizations = [metrics.get('utilization', 0.0) 
                          for metrics in self._link_metrics.values()]
            if utilizations:
                self._metrics.average_link_utilization = sum(utilizations) / len(utilizations)
                self._metrics.max_link_utilization = max(utilizations)
    
    def get_link_metrics(self, link: LinkId) -> MetricsDict:
        """
        获取链路指标。
        
        Args:
            link: 链路ID
            
        Returns:
            链路指标字典
        """
        return self._link_metrics.get(link, {})
    
    def get_node_metrics(self, node_id: NodeId) -> MetricsDict:
        """
        获取节点指标。
        
        Args:
            node_id: 节点ID
            
        Returns:
            节点指标字典
        """
        return self._node_metrics.get(node_id, {})
    
    def reset_metrics(self) -> None:
        """重置性能指标。"""
        self._metrics = NoCMetrics()
        self._link_metrics.clear()
        self._node_metrics.clear()
    
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
            "diameter": self.get_diameter(),
            "average_hop_count": self.get_average_hop_count(),
            "is_connected": self.is_connected(),
            "routing_strategy": self.routing_strategy.value,
            "bisection_bandwidth": self.get_bisection_bandwidth()
        }
    
    def get_routing_table(self, strategy: Optional[RoutingStrategy] = None) -> Dict[NodeId, Dict[NodeId, Path]]:
        """
        获取路由表。
        
        Args:
            strategy: 路由策略（可选）
            
        Returns:
            路由表
        """
        if strategy is None:
            strategy = self.routing_strategy
        
        if strategy not in self._routing_tables:
            # 计算所有节点对的路由
            self._compute_all_routes(strategy)
        
        return copy.deepcopy(self._routing_tables[strategy])
    
    def _compute_all_routes(self, strategy: RoutingStrategy) -> None:
        """
        计算所有节点对之间的路由。
        
        Args:
            strategy: 路由策略
        """
        if strategy not in self._routing_tables:
            self._routing_tables[strategy] = {}
        
        for src in range(self.num_nodes):
            if src not in self._routing_tables[strategy]:
                self._routing_tables[strategy][src] = {}
            
            for dst in range(self.num_nodes):
                if dst not in self._routing_tables[strategy][src]:
                    self._routing_tables[strategy][src][dst] = \
                        self.calculate_route(src, dst, strategy)
    
    def export_topology(self, format_type: str = "adjacency_list") -> Dict[str, Any]:
        """
        导出拓扑结构。
        
        Args:
            format_type: 导出格式类型
            
        Returns:
            拓扑数据
        """
        if format_type == "adjacency_matrix":
            return {"adjacency_matrix": self.get_adjacency_matrix()}
        elif format_type == "adjacency_list":
            adj_list = {}
            for node in range(self.num_nodes):
                adj_list[node] = self.get_neighbors(node)
            return {"adjacency_list": adj_list}
        elif format_type == "edge_list":
            edges = []
            adj_matrix = self.get_adjacency_matrix()
            for i in range(self.num_nodes):
                for j in range(self.num_nodes):
                    if adj_matrix[i][j] == 1:
                        edges.append((i, j))
            return {"edge_list": edges}
        else:
            raise ValueError(f"不支持的导出格式: {format_type}")
    
    def get_traffic_distribution(self) -> Dict[str, Any]:
        """
        获取流量分布信息。
        
        Returns:
            流量分布数据
        """
        # 基础实现，子类可以重写提供更详细的信息
        return {
            "total_links": len(self._link_metrics),
            "active_links": len([link for link, metrics in self._link_metrics.items() 
                               if metrics.get('utilization', 0) > 0]),
            "avg_link_utilization": self._metrics.average_link_utilization,
            "max_link_utilization": self._metrics.max_link_utilization
        }
    
    # ========== 优化方法 ==========
    
    def optimize_routing(self, optimization_target: str = "latency") -> None:
        """
        优化路由策略。
        
        Args:
            optimization_target: 优化目标（latency/throughput/energy）
        """
        if optimization_target == "latency":
            # 为延迟优化选择最短路径
            self.routing_strategy = RoutingStrategy.SHORTEST
        elif optimization_target == "throughput":
            # 为吞吐量优化选择负载均衡
            self.routing_strategy = RoutingStrategy.LOAD_BALANCED
        elif optimization_target == "energy":
            # 为能耗优化选择最小路径
            self.routing_strategy = RoutingStrategy.MINIMAL
        
        # 清除旧的路由缓存
        self._routing_tables.clear()
    
    def suggest_configuration_changes(self) -> List[str]:
        """
        基于当前性能建议配置更改。
        
        Returns:
            建议列表
        """
        suggestions = []
        
        if self._metrics.average_link_utilization > 0.8:
            suggestions.append("考虑增加链路带宽或使用负载均衡路由")
        
        if self._metrics.average_latency > self.get_diameter() * 2:
            suggestions.append("考虑优化路由算法或增加缓冲区大小")
        
        if not self.is_connected():
            suggestions.append("拓扑结构不连通，需要添加更多链路")
        
        return suggestions
    
    # ========== 调试和验证方法 ==========
    
    def visualize_topology(self) -> str:
        """
        可视化拓扑结构（简单文本表示）。
        
        Returns:
            拓扑的文本表示
        """
        lines = [f"拓扑类型: {self.topology_type.value}"]
        lines.append(f"节点数: {self.num_nodes}")
        lines.append(f"直径: {self.get_diameter()}")
        lines.append(f"平均跳数: {self.get_average_hop_count():.2f}")
        lines.append(f"连通性: {'是' if self.is_connected() else '否'}")
        
        lines.append("\n邻接关系:")
        for node in range(self.num_nodes):
            neighbors = self.get_neighbors(node)
            lines.append(f"节点 {node}: {neighbors}")
        
        return "\n".join(lines)
    
    def __str__(self) -> str:
        """字符串表示。"""
        return f"{self.__class__.__name__}(nodes={self.num_nodes}, type={self.topology_type.value})"
    
    def __repr__(self) -> str:
        """详细字符串表示。"""
        return f"{self.__class__.__name__}({self.get_topology_info()})"