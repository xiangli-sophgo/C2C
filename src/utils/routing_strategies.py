"""
Ring拓扑路由策略模块
支持多种路由算法，包括最短路径、负载均衡、自适应等
"""

from abc import ABC, abstractmethod
from typing import List, Tuple, Dict, Optional
import logging


class RoutingStrategy(ABC):
    """路由策略基类"""

    def __init__(self, config):
        self.config = config
        self.num_nodes = config.NUM_NODE
        self.strategy_name = self.__class__.__name__

    @abstractmethod
    def get_route_direction(self, source: int, destination: int, **kwargs) -> str:
        """
        获取路由方向

        Args:
            source: 源节点ID
            destination: 目标节点ID
            **kwargs: 额外的上下文信息（如拥塞状态等）

        Returns:
            str: 路由方向 'CW'(顺时针), 'CCW'(逆时针), 'LOCAL'(本地)
        """
        pass

    def calculate_distance(self, source: int, destination: int, direction: str) -> int:
        """计算指定方向的距离"""
        if source == destination:
            return 0

        if direction == "CW":
            return (destination - source) % self.num_nodes
        elif direction == "CCW":
            return (source - destination) % self.num_nodes
        else:
            return 0

    def get_path_info(self, source: int, destination: int, direction: str) -> Dict:
        """获取路径详细信息"""
        distance = self.calculate_distance(source, destination, direction)
        return {"direction": direction, "distance": distance, "hops": distance, "strategy": self.strategy_name}


class ShortestPathRouting(RoutingStrategy):
    """最短路径路由策略"""

    def get_route_direction(self, source: int, destination: int, **kwargs) -> str:
        if source == destination:
            return "LOCAL"

        cw_dist = (destination - source) % self.num_nodes
        ccw_dist = (source - destination) % self.num_nodes

        # 距离相等时优先选择顺时针
        return "CW" if cw_dist <= ccw_dist else "CCW"


class LoadBalancedRouting(RoutingStrategy):
    """
    负载均衡路由策略
    对角传输时：偶数节点走顺时针，奇数节点走逆时针
    其他情况：最短路径
    """

    def __init__(self, config, policy: str = "even_cw_odd_ccw"):
        super().__init__(config)
        self.policy = policy
        self.diagonal_distance = self.num_nodes // 2

    def get_route_direction(self, source: int, destination: int, **kwargs) -> str:
        if source == destination:
            return "LOCAL"

        cw_dist = (destination - source) % self.num_nodes
        ccw_dist = (source - destination) % self.num_nodes

        # 检查是否为对角传输（两个方向距离相等）
        is_diagonal = cw_dist == ccw_dist == self.diagonal_distance

        if is_diagonal:
            return self._get_diagonal_direction(source, destination)
        else:
            # 非对角传输，使用最短路径
            return "CW" if cw_dist < ccw_dist else "CCW"

    def _get_diagonal_direction(self, source: int, destination: int) -> str:
        """对角传输的方向选择策略"""
        if self.policy == "even_cw_odd_ccw":
            # 偶数节点走顺时针，奇数节点走逆时针
            return "CW" if source % 2 == 0 else "CCW"
        elif self.policy == "round_robin":
            # 基于源节点ID的轮询
            return "CW" if (source // 2) % 2 == 0 else "CCW"
        elif self.policy == "hash_based":
            # 基于源-目标对的哈希
            hash_val = hash((source, destination)) % 2
            return "CW" if hash_val == 0 else "CCW"
        else:
            # 默认回退到偶数顺时针策略
            return "CW" if source % 2 == 0 else "CCW"


class AdaptiveRouting(RoutingStrategy):
    """
    自适应路由策略
    基于实时拥塞信息动态选择路径
    """

    def __init__(self, config, congestion_threshold: float = 0.7):
        super().__init__(config)
        self.congestion_threshold = congestion_threshold
        self.fallback_strategy = ShortestPathRouting(config)

    def get_route_direction(self, source: int, destination: int, **kwargs) -> str:
        if source == destination:
            return "LOCAL"

        # 获取拥塞信息
        cw_congestion = kwargs.get("cw_congestion", 0.0)
        ccw_congestion = kwargs.get("ccw_congestion", 0.0)

        # 计算两个方向的距离
        cw_dist = (destination - source) % self.num_nodes
        ccw_dist = (source - destination) % self.num_nodes

        # 如果拥塞差异显著，选择较不拥塞的方向
        congestion_diff = abs(cw_congestion - ccw_congestion)
        if congestion_diff > self.congestion_threshold:
            if cw_congestion < ccw_congestion:
                return "CW"
            else:
                return "CCW"

        # 拥塞差异不大时，考虑距离因素
        # 给较短路径一些优势
        cw_cost = cw_dist * (1 + cw_congestion)
        ccw_cost = ccw_dist * (1 + ccw_congestion)

        return "CW" if cw_cost <= ccw_cost else "CCW"


class CustomRouting(RoutingStrategy):
    """
    完全自定义路由策略
    支持预定义路由表和规则
    """

    def __init__(self, config, custom_routes: Dict[Tuple[int, int], str] = None, fallback_strategy: str = "shortest"):
        super().__init__(config)
        self.custom_routes = custom_routes or {}

        # 设置回退策略
        if fallback_strategy == "shortest":
            self.fallback_strategy = ShortestPathRouting(config)
        elif fallback_strategy == "load_balanced":
            self.fallback_strategy = LoadBalancedRouting(config)
        else:
            self.fallback_strategy = ShortestPathRouting(config)

    def get_route_direction(self, source: int, destination: int, **kwargs) -> str:
        if source == destination:
            return "LOCAL"

        # 检查自定义路由表
        route_key = (source, destination)
        if route_key in self.custom_routes:
            return self.custom_routes[route_key]

        # 回退到默认策略
        return self.fallback_strategy.get_route_direction(source, destination, **kwargs)

    def add_custom_route(self, source: int, destination: int, direction: str):
        """动态添加自定义路由"""
        self.custom_routes[(source, destination)] = direction

    def remove_custom_route(self, source: int, destination: int):
        """移除自定义路由"""
        route_key = (source, destination)
        if route_key in self.custom_routes:
            del self.custom_routes[route_key]


class DeterministicRouting(RoutingStrategy):
    """
    确定性路由策略
    保证相同源-目标对总是选择相同路径，便于调试和分析
    """

    def __init__(self, config, seed: int = 42):
        super().__init__(config)
        self.seed = seed

    def get_route_direction(self, source: int, destination: int, **kwargs) -> str:
        if source == destination:
            return "LOCAL"

        # 使用源-目标对的哈希值确定性地选择方向
        import hashlib

        hash_input = f"{source}-{destination}-{self.seed}".encode()
        hash_val = int(hashlib.md5(hash_input).hexdigest()[:8], 16)

        cw_dist = (destination - source) % self.num_nodes
        ccw_dist = (source - destination) % self.num_nodes

        # 如果距离相等，使用哈希值决定
        if cw_dist == ccw_dist:
            return "CW" if hash_val % 2 == 0 else "CCW"
        else:
            return "CW" if cw_dist < ccw_dist else "CCW"


def create_routing_strategy(config, strategy_name: str, **strategy_params) -> RoutingStrategy:
    """
    路由策略工厂函数

    Args:
        config: 配置对象
        strategy_name: 策略名称
        **strategy_params: 策略特定参数

    Returns:
        RoutingStrategy: 路由策略实例
    """
    strategy_map = {"shortest": ShortestPathRouting, "load_balanced": LoadBalancedRouting, "adaptive": AdaptiveRouting, "custom": CustomRouting, "deterministic": DeterministicRouting}

    if strategy_name not in strategy_map:
        logging.warning(f"Unknown routing strategy: {strategy_name}, using shortest path")
        strategy_name = "shortest"

    strategy_class = strategy_map[strategy_name]
    return strategy_class(config, **strategy_params)


# 预定义的一些有用的自定义路由表示例
def create_balanced_8node_routes() -> Dict[Tuple[int, int], str]:
    """为8节点Ring创建完全负载均衡的路由表"""
    routes = {}

    # 对角路由：0↔4, 1↔5, 2↔6, 3↔7
    diagonal_pairs = [(0, 4), (1, 5), (2, 6), (3, 7)]

    for i, (src, dst) in enumerate(diagonal_pairs):
        # 偶数对走顺时针，奇数对走逆时针
        direction = "CW" if i % 2 == 0 else "CCW"
        routes[(src, dst)] = direction
        routes[(dst, src)] = direction

    return routes


def create_priority_routes(high_priority_pairs: List[Tuple[int, int]], preferred_direction: str = "CW") -> Dict[Tuple[int, int], str]:
    """为高优先级流量创建专用路由"""
    routes = {}

    for src, dst in high_priority_pairs:
        routes[(src, dst)] = preferred_direction
        # 反向路径使用相反方向
        reverse_direction = "CCW" if preferred_direction == "CW" else "CW"
        routes[(dst, src)] = reverse_direction

    return routes
