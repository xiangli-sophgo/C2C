"""
NoC路由器节点实现。

本模块实现了NoC网络中的路由器节点，负责数据包的路由转发。
路由器是NoC网络的核心组件，提供：
- 多端口数据包路由
- 虚拟通道支持
- 流量控制机制
- 缓存管理
- 优先级调度
"""

from typing import Dict, List, Optional, Any, Tuple
from collections import deque
from enum import Enum
import logging

from .node import BaseNoCNode, NodeState, BufferStatus
from .flit import BaseFlit
from src.noc.utils.types import NodeId, Position, Priority, RoutingStrategy


class RoutingAlgorithm(Enum):
    """路由算法类型"""

    XY = "xy"  # XY路由
    YX = "yx"  # YX路由
    WEST_FIRST = "west_first"  # 西优先
    NORTH_LAST = "north_last"  # 北最后
    MINIMAL = "minimal"  # 最小路由
    ADAPTIVE = "adaptive"  # 自适应路由


class PortDirection(Enum):
    """端口方向"""

    NORTH = "north"
    SOUTH = "south"
    EAST = "east"
    WEST = "west"
    LOCAL = "local"


class RouterNode(BaseNoCNode):
    """
    NoC路由器节点实现。

    路由器是NoC网络的核心组件，负责：
    1. 接收来自不同端口的数据包
    2. 根据路由算法计算转发路径
    3. 管理虚拟通道和缓存
    4. 执行流量控制和优先级调度
    5. 监控性能指标
    """

    def __init__(self, node_id: NodeId, position: Position, routing_algorithm: RoutingAlgorithm = RoutingAlgorithm.XY, **kwargs):
        """
        初始化路由器节点。

        Args:
            node_id: 节点ID
            position: 节点位置
            routing_algorithm: 路由算法类型
            **kwargs: 其他配置参数
        """
        from src.noc.utils.types import NodeType

        super().__init__(node_id, position, NodeType.ROUTER)

        self.routing_algorithm = routing_algorithm

        # 路由器特有配置
        self.num_ports = 5  # north, south, east, west, local
        self.port_directions = [PortDirection.NORTH, PortDirection.SOUTH, PortDirection.EAST, PortDirection.WEST, PortDirection.LOCAL]

        # 路由表和缓存
        self.routing_table: Dict[NodeId, PortDirection] = {}
        self.routing_cache: Dict[NodeId, PortDirection] = {}

        # 端口状态管理
        self.port_busy: Dict[PortDirection, bool] = {}
        self.port_credits: Dict[PortDirection, Dict[int, int]] = {}  # {port: {vc: credits}}

        # 仲裁器状态
        self.arbitration_state: Dict[PortDirection, int] = {}  # 轮询仲裁状态

        # 性能监控
        self.router_stats = {
            "packets_routed": 0,
            "packets_dropped": 0,
            "routing_conflicts": 0,
            "port_utilization": {port.value: 0.0 for port in self.port_directions},
            "vc_utilization": {port.value: [0.0] * self.virtual_channels for port in self.port_directions},
            "average_routing_delay": 0.0,
        }

        # 路由延迟统计
        self.routing_delays: List[int] = []

        # 初始化路由器组件
        self._initialize_router_components()

        # 设置日志
        self.logger = logging.getLogger(f"RouterNode_{node_id}")

        self.virtual_channels = kwargs.get("virtual_channels", 2)

    def _initialize_router_components(self) -> None:
        """初始化路由器组件"""
        # 初始化端口状态
        for port in self.port_directions:
            self.port_busy[port] = False
            self.port_credits[port] = {}
            self.arbitration_state[port] = 0

            # 初始化虚拟通道信用
            for vc in range(self.virtual_channels):
                self.port_credits[port][vc] = self.output_buffer_size

        # 初始化路由表（根据网络拓扑设置）
        self._initialize_routing_table()

    def _initialize_routing_table(self) -> None:
        """初始化路由表（基于网络拓扑）"""
        # 默认路由表，实际使用时应根据网络拓扑动态生成
        # 这里提供一个基本的实现框架
        pass

    def process_flit(self, flit: BaseFlit, input_port: str) -> bool:
        """
        处理接收到的flit。

        Args:
            flit: 要处理的flit对象
            input_port: 输入端口名称

        Returns:
            是否成功处理
        """
        try:
            # 检查输入端口是否有效
            if input_port not in [port.value for port in self.port_directions]:
                self.logger.error(f"无效的输入端口: {input_port}")
                return False

            # 更新flit的路由信息
            flit.current_hop = self.node_id
            flit.hop_count += 1

            # 检查是否为本地目标
            if flit.destination == self.node_id:
                return self._handle_local_delivery(flit, input_port)

            # 执行路由决策
            output_port = self.route_flit(flit)
            if output_port is None:
                self.logger.warning(f"无法为flit {flit.packet_id}找到路由")
                self.router_stats["packets_dropped"] += 1
                return False

            # 尝试转发到输出端口
            return self._forward_flit(flit, input_port, output_port)

        except Exception as e:
            self.logger.error(f"处理flit时发生错误: {e}")
            return False

    def route_flit(self, flit: BaseFlit) -> Optional[str]:
        """
        为flit进行路由决策。

        Args:
            flit: 要路由的flit对象

        Returns:
            输出端口名称，如果无法路由则返回None
        """
        destination = flit.destination

        # 检查路由缓存
        if destination in self.routing_cache:
            return self.routing_cache[destination].value

        # 根据路由算法计算路径
        output_port = None

        if self.routing_algorithm == RoutingAlgorithm.XY:
            output_port = self._xy_routing(destination)
        elif self.routing_algorithm == RoutingAlgorithm.YX:
            output_port = self._yx_routing(destination)
        elif self.routing_algorithm == RoutingAlgorithm.ADAPTIVE:
            output_port = self._adaptive_routing(destination, flit)
        else:
            output_port = self._xy_routing(destination)  # 默认使用XY路由

        # 缓存路由结果
        if output_port:
            self.routing_cache[destination] = output_port

        return output_port.value if output_port else None

    def _xy_routing(self, destination: NodeId) -> Optional[PortDirection]:
        """XY路由算法"""
        # 假设节点ID编码了位置信息 (x, y)
        # 这里需要根据实际的节点ID编码方式调整

        if not hasattr(self, "mesh_width"):
            # 如果没有设置网格宽度，使用默认值
            self.mesh_width = 4

        src_x, src_y = divmod(self.node_id, self.mesh_width)
        dst_x, dst_y = divmod(destination, self.mesh_width)

        # 先沿X轴移动
        if src_x < dst_x:
            return PortDirection.EAST
        elif src_x > dst_x:
            return PortDirection.WEST
        # 再沿Y轴移动
        elif src_y < dst_y:
            return PortDirection.NORTH
        elif src_y > dst_y:
            return PortDirection.SOUTH

        return None  # 已到达目标

    def _yx_routing(self, destination: NodeId) -> Optional[PortDirection]:
        """YX路由算法"""
        if not hasattr(self, "mesh_width"):
            self.mesh_width = 4

        src_x, src_y = divmod(self.node_id, self.mesh_width)
        dst_x, dst_y = divmod(destination, self.mesh_width)

        # 先沿Y轴移动
        if src_y < dst_y:
            return PortDirection.NORTH
        elif src_y > dst_y:
            return PortDirection.SOUTH
        # 再沿X轴移动
        elif src_x < dst_x:
            return PortDirection.EAST
        elif src_x > dst_x:
            return PortDirection.WEST

        return None  # 已到达目标

    def _adaptive_routing(self, destination: NodeId, flit: BaseFlit) -> Optional[PortDirection]:
        """自适应路由算法"""
        # 获取所有可能的最小路径
        possible_ports = self._get_minimal_paths(destination)

        if not possible_ports:
            return None

        # 选择负载最轻的端口
        best_port = None
        min_load = float("inf")

        for port in possible_ports:
            load = self._get_port_load(port)
            if load < min_load:
                min_load = load
                best_port = port

        return best_port

    def _get_minimal_paths(self, destination: NodeId) -> List[PortDirection]:
        """获取到目标节点的所有最小路径"""
        if not hasattr(self, "mesh_width"):
            self.mesh_width = 4

        src_x, src_y = divmod(self.node_id, self.mesh_width)
        dst_x, dst_y = divmod(destination, self.mesh_width)

        possible_ports = []

        # 检查X方向
        if src_x < dst_x:
            possible_ports.append(PortDirection.EAST)
        elif src_x > dst_x:
            possible_ports.append(PortDirection.WEST)

        # 检查Y方向
        if src_y < dst_y:
            possible_ports.append(PortDirection.NORTH)
        elif src_y > dst_y:
            possible_ports.append(PortDirection.SOUTH)

        return possible_ports

    def _get_port_load(self, port: PortDirection) -> float:
        """获取端口负载"""
        if port.value not in self.buffer_occupancy:
            return 0.0

        occupancy = self.buffer_occupancy[port.value]
        max_capacity = self.output_buffer_size

        return occupancy / max_capacity if max_capacity > 0 else 0.0

    def _handle_local_delivery(self, flit: BaseFlit, input_port: str) -> bool:
        """处理本地交付"""
        # 将flit放入本地缓冲区
        local_buffer = self.output_buffers.get(PortDirection.LOCAL.value)
        if local_buffer is None:
            return False

        if len(local_buffer) >= self.output_buffer_size:
            return False

        local_buffer.append(flit)
        flit.ejection_time = self.current_cycle

        # 更新统计
        self.router_stats["packets_routed"] += 1

        return True

    def _forward_flit(self, flit: BaseFlit, input_port: str, output_port: str) -> bool:
        """转发flit到输出端口"""
        # 检查输出端口是否可用
        if not self._can_forward_to_port(output_port, flit.priority):
            return False

        # 分配虚拟通道
        vc_id = self.allocate_virtual_channel(output_port, flit.priority)
        if vc_id is None:
            return False

        # 将flit放入输出缓冲区
        output_buffer = self.output_buffers.get(output_port)
        if output_buffer is None:
            return False

        output_buffer.append(flit)
        flit.virtual_channel = vc_id

        # 更新统计
        self.router_stats["packets_routed"] += 1
        self._update_port_utilization(output_port)

        return True

    def _can_forward_to_port(self, port: str, priority: Priority) -> bool:
        """检查是否可以转发到指定端口"""
        # 检查端口是否忙碌
        port_enum = PortDirection(port)
        if self.port_busy.get(port_enum, False):
            return False

        # 检查缓冲区空间
        if self.is_buffer_full(port):
            return False

        # 检查信用
        if not self.has_credit(port):
            return False

        return True

    def _update_port_utilization(self, port: str) -> None:
        """更新端口利用率"""
        if port in self.router_stats["port_utilization"]:
            current_util = self.router_stats["port_utilization"][port]
            self.router_stats["port_utilization"][port] = current_util + 1.0

    def can_accept_flit(self, input_port: str, priority: Priority = Priority.MEDIUM) -> bool:
        """
        检查是否可以接收新的flit。

        Args:
            input_port: 输入端口名称
            priority: flit优先级

        Returns:
            是否可以接收
        """
        # 检查输入缓冲区是否有空间
        if self.is_buffer_full(input_port):
            return False

        # 检查节点是否正常运行
        if not self.is_operational():
            return False

        # 高优先级flit可以抢占资源
        if priority in [Priority.HIGH, Priority.CRITICAL]:
            return True

        # 检查是否有足够的处理能力
        total_occupancy = sum(self.get_buffer_occupancy(port.value) for port in self.port_directions)
        total_capacity = len(self.port_directions) * self.input_buffer_size

        return total_occupancy / total_capacity < 0.8  # 80%阈值

    def step_router(self, cycle: int) -> None:
        """
        执行路由器的一个周期操作。

        Args:
            cycle: 当前周期
        """
        self.current_cycle = cycle

        # 执行仲裁
        self._perform_arbitration()

        # 处理虚拟通道
        self._process_virtual_channels()

        # 更新信用
        self._update_credits()

        # 检查死锁
        self._check_deadlock()

        # 更新统计
        self._update_router_statistics()

    def _perform_arbitration(self) -> None:
        """执行端口仲裁"""
        for port in self.port_directions:
            if port == PortDirection.LOCAL:
                continue

            # 轮询仲裁
            input_buffer = self.input_buffers.get(port.value)
            if input_buffer and input_buffer:
                # 处理缓冲区中的flit
                flit = input_buffer.popleft()
                self.process_flit(flit, port.value)

            # 更新仲裁状态
            self.arbitration_state[port] = (self.arbitration_state[port] + 1) % self.virtual_channels

    def _process_virtual_channels(self) -> None:
        """处理虚拟通道"""
        for port in self.port_directions:
            port_name = port.value
            if port_name in self.virtual_channel_buffers:
                vc_buffers = self.virtual_channel_buffers[port_name]

                for vc_id, vc_buffer in enumerate(vc_buffers):
                    if vc_buffer:
                        # 尝试将flit从虚拟通道转发到输出端口
                        flit = vc_buffer.popleft()
                        output_port = self.route_flit(flit)

                        if output_port:
                            self._forward_flit(flit, port_name, output_port)

    def _update_credits(self) -> None:
        """更新信用计数"""
        for port in self.port_directions:
            for vc_id in range(self.virtual_channels):
                # 检查下游是否有空间
                if self._downstream_has_space(port, vc_id):
                    max_credits = self.port_credits[port].get(vc_id, 0)
                    if max_credits < self.output_buffer_size:
                        self.port_credits[port][vc_id] = max_credits + 1

    def _downstream_has_space(self, port: PortDirection, vc_id: int) -> bool:
        """检查下游是否有缓冲空间"""
        # 这里需要与相邻节点通信，获取其缓冲区状态
        # 暂时返回True，实际实现需要网络级别的协调
        return True

    def _check_deadlock(self) -> None:
        """检查死锁情况"""
        # 简单的死锁检测：检查是否有循环等待
        blocked_ports = []

        for port in self.port_directions:
            if self.port_busy.get(port, False):
                blocked_ports.append(port)

        if len(blocked_ports) >= 3:  # 可能存在死锁
            self.logger.warning(f"可能存在死锁，被阻塞的端口: {blocked_ports}")

    def _update_router_statistics(self) -> None:
        """更新路由器统计信息"""
        # 计算平均路由延迟
        if self.routing_delays:
            self.router_stats["average_routing_delay"] = sum(self.routing_delays) / len(self.routing_delays)

        # 计算端口利用率
        for port in self.port_directions:
            port_name = port.value
            if port_name in self.buffer_occupancy:
                occupancy = self.buffer_occupancy[port_name]
                max_capacity = self.output_buffer_size
                self.router_stats["port_utilization"][port_name] = occupancy / max_capacity

    def get_router_status(self) -> Dict[str, Any]:
        """
        获取路由器状态信息。

        Returns:
            路由器状态字典
        """
        status = self.get_performance_stats()
        status.update(
            {
                "routing_algorithm": self.routing_algorithm.value,
                "router_stats": self.router_stats.copy(),
                "port_credits": {port.value: credits for port, credits in self.port_credits.items()},
                "arbitration_state": {port.value: state for port, state in self.arbitration_state.items()},
                "routing_cache_size": len(self.routing_cache),
            }
        )

        return status

    def clear_routing_cache(self) -> None:
        """清除路由缓存"""
        self.routing_cache.clear()

    def set_routing_algorithm(self, algorithm: RoutingAlgorithm) -> None:
        """
        设置路由算法。

        Args:
            algorithm: 新的路由算法
        """
        self.routing_algorithm = algorithm
        self.clear_routing_cache()
        self.logger.info(f"路由算法已更改为: {algorithm.value}")

    def add_routing_entry(self, destination: NodeId, port: PortDirection) -> None:
        """
        添加路由表条目。

        Args:
            destination: 目标节点ID
            port: 输出端口
        """
        self.routing_table[destination] = port

    def remove_routing_entry(self, destination: NodeId) -> None:
        """
        移除路由表条目。

        Args:
            destination: 目标节点ID
        """
        if destination in self.routing_table:
            del self.routing_table[destination]
        if destination in self.routing_cache:
            del self.routing_cache[destination]
