"""
CrossRing主模型类实现。

基于C2C仓库的架构，提供完整的CrossRing NoC仿真模型，
包括IP接口管理、网络组件和仿真循环控制。
集成真实的环形拓扑、环形桥接和四方向系统。
"""

from __future__ import annotations
from typing import Dict, List, Any, Optional, Tuple
import logging
from collections import defaultdict
import time
from enum import Enum

from .config import CrossRingConfig, RoutingStrategy
from .ip_interface import CrossRingIPInterface
from .flit import CrossRingFlit, get_crossring_flit_pool_stats
from .node import CrossRingNode
from .crossring_link import CrossRingLink
from src.noc.utils.types import NodeId
from src.noc.debug import RequestTracker, RequestState, FlitType
from src.noc.base.model import BaseNoCModel


class RingDirection(Enum):
    """CrossRing方向枚举（简化版本）"""

    TL = "TL"  # Turn Left
    TR = "TR"  # Turn Right
    TU = "TU"  # Turn Up
    TD = "TD"  # Turn Down


class CrossRingModel(BaseNoCModel):
    """
    CrossRing主模型类。

    该类负责：
    1. 整体仿真循环控制
    2. IP接口实例管理
    3. CrossRing网络组件管理（骨架）
    4. 全局状态监控和调试
    5. 性能统计收集
    """

    def __init__(self, config: CrossRingConfig):
        """
        初始化CrossRing模型

        Args:
            config: CrossRing配置实例
        """
        # 调用父类初始化
        super().__init__(config, model_name="CrossRingModel")

        # IP接口管理
        self.ip_interfaces: Dict[str, CrossRingIPInterface] = {}
        self._ip_registry: Dict[str, CrossRingIPInterface] = {}  # 全局debug视图

        # CrossRing网络组件 - 使用新的架构
        self.crossring_nodes: Dict[NodeId, Any] = {}  # {node_id: CrossRingNode}
        self.crossring_links: Dict[str, Any] = {}  # {link_id: CrossRingLink}

        # Tag管理器
        self.tag_managers: Dict[NodeId, Any] = {}  # {node_id: CrossRingTagManager}

        # 性能统计
        self.stats = {
            "total_requests": 0,
            "total_responses": 0,
            "total_data_flits": 0,
            "read_retries": 0,
            "write_retries": 0,
            "average_latency": 0.0,
            "peak_active_requests": 0,
            "current_active_requests": 0,
            "dimension_turns": 0,
            "ring_transmissions": 0,
            "wrap_around_hops": 0,
        }

        # 仿真状态
        self.is_running = False
        self.is_finished = False

        # 日志配置
        self.logger = logging.getLogger(f"CrossRingModel_{id(self)}")

        # 调试和追踪系统
        self.request_tracker = RequestTracker(network_frequency=1)  # 简化为1GHz = 1
        self.debug_enabled = False
        self.trace_packets = set()

        # 初始化所有组件
        self._setup_ip_interfaces()
        self._setup_crossring_networks()

        # 验证CrossRing网络初始化
        if len(self.crossring_nodes) != self.config.num_nodes:
            self.logger.error("CrossRing节点初始化不完整")
            raise RuntimeError("CrossRing网络初始化失败")

        self.logger.info(f"CrossRing模型初始化完成: {config.num_row}x{config.num_col}")

    def _setup_ip_interfaces(self) -> None:
        """设置所有IP接口"""
        ip_type_configs = [
            ("gdma", self.config.gdma_send_position_list),
            ("sdma", self.config.sdma_send_position_list),
            ("cdma", self.config.cdma_send_position_list),
            ("ddr", self.config.ddr_send_position_list),
            ("l2m", self.config.l2m_send_position_list),
        ]

        for ip_type, positions in ip_type_configs:
            for node_id in positions:
                key = f"{ip_type}_{node_id}"
                self.ip_interfaces[key] = CrossRingIPInterface(config=self.config, ip_type=ip_type, node_id=node_id, model=self)

                self.logger.debug(f"创建IP接口: {key} at node {node_id}")

    def _setup_crossring_networks(self) -> None:
        """设置CrossRing网络组件的完整实现 - 真实环形拓扑"""
        # 用CrossRingNode实例替换原有dict结构
        self.crossring_nodes: Dict[NodeId, CrossRingNode] = {}
        for node_id in range(self.config.num_nodes):
            coordinates = self._get_node_coordinates(node_id)
            node = CrossRingNode(node_id=node_id, coordinates=coordinates, config=self.config, logger=self.logger)
            self.crossring_nodes[node_id] = node
        self.logger.info(f"CrossRing网络创建完成: {len(self.crossring_nodes)}个节点")

    def _get_node_coordinates(self, node_id: NodeId) -> Tuple[int, int]:
        """
        获取节点在CrossRing网格中的坐标

        Args:
            node_id: 节点ID

        Returns:
            (x, y)坐标
        """
        x = node_id % self.config.num_col
        y = node_id // self.config.num_col
        return x, y

    def _get_next_node_in_direction(self, node_id: NodeId, direction: RingDirection) -> NodeId:
        """
        获取指定方向的下一个节点（CrossRing特定实现）

        在CrossRing中，边界节点连接到自己，而不是环绕连接

        Args:
            node_id: 当前节点ID
            direction: 移动方向

        Returns:
            下一个节点的ID
        """
        x, y = self._get_node_coordinates(node_id)

        if direction == RingDirection.TL:
            # 向左：如果已经在最左边，连接到自己
            if x == 0:
                next_x = x  # 连接到自己
            else:
                next_x = x - 1
            next_y = y
        elif direction == RingDirection.TR:
            # 向右：如果已经在最右边，连接到自己
            if x == self.config.num_col - 1:
                next_x = x  # 连接到自己
            else:
                next_x = x + 1
            next_y = y
        elif direction == RingDirection.TU:
            # 向上：如果已经在最上边，连接到自己
            if y == 0:
                next_y = y  # 连接到自己
            else:
                next_y = y - 1
            next_x = x
        elif direction == RingDirection.TD:
            # 向下：如果已经在最下边，连接到自己
            if y == self.config.num_row - 1:
                next_y = y  # 连接到自己
            else:
                next_y = y + 1
            next_x = x
        else:
            raise ValueError(f"不支持的方向: {direction}")

        return next_y * self.config.num_col + next_x

    def _get_ring_connections(self, node_id: NodeId) -> Dict[str, NodeId]:
        """
        获取节点的环形连接信息（真实环形拓扑）

        Args:
            node_id: 节点ID

        Returns:
            环形连接字典，包含四个方向的邻居节点
        """
        connections = {}

        # 获取四个方向的邻居节点
        for direction in [RingDirection.TL, RingDirection.TR, RingDirection.TU, RingDirection.TD]:
            neighbor_id = self._get_next_node_in_direction(node_id, direction)
            connections[direction.value] = neighbor_id

        return connections

    def register_ip_interface(self, ip_interface: CrossRingIPInterface) -> None:
        """
        注册IP接口（用于全局debug和管理）

        Args:
            ip_interface: IP接口实例
        """
        key = f"{ip_interface.ip_type}_{ip_interface.node_id}"
        self._ip_registry[key] = ip_interface

        self.logger.debug(f"注册IP接口到全局registry: {key}")

    def step(self) -> None:
        """执行一个仿真周期（使用两阶段执行模型）"""
        self.cycle += 1

        # 阶段1：组合逻辑阶段 - 所有组件计算传输决策
        self._step_compute_phase()

        # 阶段2：时序逻辑阶段 - 所有组件执行传输和状态更新
        self._step_update_phase()

        # 更新统计信息
        self._update_statistics()

        # 调试功能
        self.debug_func()

        # 简化的包完成模拟逻辑（用于demo）
        self._simulate_packet_completion()

        # 定期输出调试信息
        if self.cycle % 1000 == 0:
            self.logger.debug(f"周期 {self.cycle}: {self.get_active_request_count()}个活跃请求")

    def _step_compute_phase(self) -> None:
        """阶段1：组合逻辑阶段 - 所有组件计算传输决策，不修改状态"""
        # 1. 所有IP接口计算阶段
        for ip_interface in self.ip_interfaces.values():
            if hasattr(ip_interface, "step_compute_phase"):
                ip_interface.step_compute_phase(self.cycle)
            else:
                # 兼容性：如果没有两阶段方法，调用compute阶段
                ip_interface._compute_phase(self.cycle)

        # 2. CrossRing网络组件计算阶段
        self._step_crossring_networks_compute_phase()

    def _step_update_phase(self) -> None:
        """阶段2：时序逻辑阶段 - 所有组件执行传输和状态更新"""
        # 1. 所有IP接口更新阶段
        for ip_interface in self.ip_interfaces.values():
            if hasattr(ip_interface, "step_update_phase"):
                ip_interface.step_update_phase(self.cycle)
            else:
                # 兼容性：如果没有两阶段方法，调用sync和update阶段
                ip_interface._sync_phase(self.cycle)
                ip_interface._update_phase(self.cycle)

        # 2. CrossRing网络组件更新阶段
        self._step_crossring_networks_update_phase()

    def _step_crossring_networks_compute_phase(self) -> None:
        """CrossRing网络组件计算阶段"""
        # 所有CrossRing节点计算阶段
        for node in self.crossring_nodes.values():
            if hasattr(node, "step_compute_phase"):
                node.step_compute_phase(self.cycle)

    def _step_crossring_networks_update_phase(self) -> None:
        """CrossRing网络组件更新阶段"""
        # 所有CrossRing节点更新阶段
        for node in self.crossring_nodes.values():
            if hasattr(node, "update_state"):
                node.update_state(self.cycle)

        # 3. 兼容性：如需保留旧接口可在此补充

    def _step_crossring_networks(self) -> None:
        """兼容性接口：使用两阶段执行模型"""
        self._step_crossring_networks_compute_phase()
        self._step_crossring_networks_update_phase()

    def _step_crossring_networks_legacy(self) -> None:
        """
        处理CrossRing网络的完整传输逻辑 - 真实环形拓扑实现

        实现真实的CrossRing网络传输：
        1. 处理环形桥接系统
        2. 重置仲裁状态
        3. inject_queue → ring网络注入（四方向系统）
        4. ring网络内部传输（真实环形连接）
        5. ring网络 → eject_queue提取
        6. ETag/ITag处理和拥塞控制
        """
        # 第零阶段：处理CrossRing网络传输（使用新架构）
        # 在新架构中，这个功能已集成到节点的update_state中

        # 第一阶段：重置每个节点的仲裁状态
        self._reset_arbitration_states()

        # 第二阶段：处理inject队列到ring网络的注入（四方向系统）
        self._process_inject_to_ring_four_directions()

        # 第三阶段：处理ring网络内部传输（真实环形连接）
        self._process_ring_transmission_true_topology()

        # 第四阶段：处理ring网络到eject队列的提取
        self._process_ring_to_eject()

        # 第五阶段：更新ETag/ITag状态
        self._update_etag_itag_status()

    def _reset_arbitration_states(self) -> None:
        """重置所有节点的仲裁状态 - 四方向系统"""
        for node in self.crossring_nodes.values():
            # 重置仲裁优先级为初始状态（四方向系统）
            node.reset_arbitration_state()

    def _process_inject_to_ring_four_directions(self) -> None:
        """处理inject队列到ring网络的注入 - 四方向系统"""
        for node_id, node in self.crossring_nodes.items():
            for channel in ["req", "rsp", "data"]:
                inject_queue = node.inject_queues[channel]

                # 处理inject队列中的每个flit
                while inject_queue:
                    flit = inject_queue[0]  # 查看队首flit

                    # 使用四方向系统确定路由方向
                    ring_directions = self._determine_ring_directions_four_way(node_id, flit.destination)

                    if not ring_directions:
                        # 目标就是当前节点，直接移到eject队列
                        flit = inject_queue.pop(0)
                        node.eject_queues[channel].append(flit)
                        flit.is_arrive = True
                        flit.arrival_network_cycle = self.cycle
                        self.logger.debug(f"周期{self.cycle}：数据包{flit.packet_id}到达目标节点{node_id}")
                        continue

                    # 尝试注入到ring网络（四方向系统）
                    if self._try_inject_to_ring_four_directions(node, flit, channel, ring_directions):
                        inject_queue.pop(0)  # 成功注入，移除flit
                        self.logger.debug(f"周期{self.cycle}：数据包{flit.packet_id}成功注入到环形网络，方向{ring_directions}")
                    else:
                        break  # 注入失败，等待下个周期

    def _determine_ring_directions_four_way(self, source: NodeId, destination: NodeId) -> List[RingDirection]:
        """确定路由方向（四方向系统）- 支持XY和YX路由策略"""
        if source == destination:
            return []

        # 计算路由方向
        src_x, src_y = self._get_node_coordinates(source)
        dst_x, dst_y = self._get_node_coordinates(destination)

        horizontal_direction = None
        vertical_direction = None

        if src_x != dst_x:
            horizontal_direction = RingDirection.TR if src_x < dst_x else RingDirection.TL
        if src_y != dst_y:
            vertical_direction = RingDirection.TD if src_y < dst_y else RingDirection.TU

        directions = []

        # 根据路由策略确定维度顺序
        if self.config.routing_strategy == RoutingStrategy.XY:
            # XY路由：先水平后垂直
            if horizontal_direction:
                directions.append(horizontal_direction)
            if vertical_direction:
                directions.append(vertical_direction)
        elif self.config.routing_strategy == RoutingStrategy.YX:
            # YX路由：先垂直后水平
            if vertical_direction:
                directions.append(vertical_direction)
            if horizontal_direction:
                directions.append(horizontal_direction)
        elif self.config.routing_strategy == RoutingStrategy.ADAPTIVE:
            # 自适应路由：根据拥塞情况选择（未来实现）
            # 目前回退到XY路由
            if horizontal_direction:
                directions.append(horizontal_direction)
            if vertical_direction:
                directions.append(vertical_direction)

        return directions

    def _try_inject_to_ring_four_directions(self, node: CrossRingNode, flit: Any, channel: str, directions: List[RingDirection]) -> bool:
        """尝试将flit注入到ring网络（四方向系统，增强拥塞控制）"""
        if not directions:
            return False

        # 选择第一个方向进行注入（根据路由策略确定）
        target_direction = directions[0]
        direction_str = target_direction.value

        # 1. 检查本地ETag状态（拥塞控制）
        if node.etag_status[direction_str][channel]:
            self._record_congestion_event(node.node_id, "etag_block", direction_str, channel)
            return False  # 本地拥塞，无法注入

        # 2. 检查下游路径的拥塞状态
        if self._check_downstream_path_congestion_four_directions(node.node_id, flit.destination, target_direction, channel):
            self._record_congestion_event(node.node_id, "downstream_congestion", direction_str, channel)
            return False  # 下游路径拥塞

        # 3. 检查ring缓冲区是否有空间
        ring_buffer = node.ring_buffers[direction_str][channel]
        if len(ring_buffer) >= self.config.ring_buffer_depth:
            self._record_congestion_event(node.node_id, "buffer_full", direction_str, channel)
            return False  # 缓冲区满

        # 4. 执行仲裁（四方向系统）
        if not self._arbitrate_ring_access_four_directions(node, target_direction, "inject"):
            self._record_congestion_event(node.node_id, "arbitration_fail", direction_str, channel)
            return False  # 仲裁失败

        # 5. 成功注入到ring
        ring_buffer.append(flit)
        flit.network_entry_cycle = self.cycle
        flit.current_ring_direction = target_direction
        flit.remaining_directions = directions[1:]  # 剩余的路由方向
        self._record_injection_success(node.node_id, direction_str, channel)

        return True

    def _arbitrate_ring_access_four_directions(self, node: CrossRingNode, direction: RingDirection, requester: str) -> bool:
        """四方向系统的环形访问仲裁机制"""
        direction_str = direction.value
        arbitration_state = node.arbitration_state

        # 简化仲裁：轮询机制
        if requester == "inject":
            # 注入请求需要检查是否有其他方向的流量
            for dir_name in ["TL", "TR", "TU", "TD"]:
                if dir_name != direction_str:
                    for channel in ["req", "rsp", "data"]:
                        if node.ring_buffers[dir_name][channel]:
                            # 有其他方向的流量，注入失败
                            return False

            # 没有冲突，允许注入
            arbitration_state["last_winner"] = direction_str
            return True

        else:
            # 环形传输请求总是被允许（优先级高于注入）
            arbitration_state["last_winner"] = direction_str
            return True

    def _process_ring_transmission_true_topology(self) -> None:
        """处理真实环形拓扑的ring网络内部传输"""
        # 处理四个方向的环形传输
        for direction in [RingDirection.TL, RingDirection.TR, RingDirection.TU, RingDirection.TD]:
            self._process_ring_direction_transmission(direction)

    def _process_ring_direction_transmission(self, direction: RingDirection) -> None:
        """处理指定方向的环形传输"""
        direction_str = direction.value

        # 获取该方向上的所有节点
        nodes_in_direction = self._get_nodes_in_ring_direction(direction)

        # 按传输顺序处理每个节点（从后往前避免冲突）
        for i in reversed(range(len(nodes_in_direction))):
            current_node = nodes_in_direction[i]
            next_node = nodes_in_direction[(i + 1) % len(nodes_in_direction)]

            current_node_instance = self.crossring_nodes[current_node]
            next_node_instance = self.crossring_nodes[next_node]

            # 处理每个通道
            for channel in ["req", "rsp", "data"]:
                self._try_move_flit_in_ring_direction(current_node_instance, next_node_instance, direction_str, channel)

    def _get_nodes_in_ring_direction(self, direction: RingDirection) -> List[NodeId]:
        """获取指定方向上的环形节点列表"""
        if direction in [RingDirection.TL, RingDirection.TR]:
            # 水平环：按行组织，但每行独立形成环
            nodes = []
            for row in range(self.config.num_row):
                row_nodes = [row * self.config.num_col + col for col in range(self.config.num_col)]
                if direction == RingDirection.TL:
                    row_nodes.reverse()  # 逆时针
                nodes.extend(row_nodes)
            # 过滤掉超出范围的节点
            return [node for node in nodes if node < self.config.num_nodes]
        else:
            # 垂直环：按列组织，每列独立形成环
            nodes = []
            for col in range(self.config.num_col):
                col_nodes = [row * self.config.num_col + col for row in range(self.config.num_row)]
                if direction == RingDirection.TU:
                    col_nodes.reverse()  # 向上
                nodes.extend(col_nodes)
            # 过滤掉超出范围的节点
            return [node for node in nodes if node < self.config.num_nodes]

    def _try_move_flit_in_ring_direction(self, current_node: CrossRingNode, next_node: CrossRingNode, direction_str: str, channel: str) -> None:
        """尝试在指定方向的环形中移动flit"""
        current_buffer = current_node.ring_buffers[direction_str][channel]
        next_buffer = next_node.ring_buffers[direction_str][channel]

        if not current_buffer:
            return

        flit = current_buffer[0]  # 查看队首flit

        # 检查是否到达最终目标节点
        if self._should_eject_flit_four_directions(next_node.node_id, flit):
            # 尝试移动到eject队列
            if self._try_move_to_eject(next_node, flit, channel):
                current_buffer.pop(0)  # 成功eject，移除flit
                self.stats["ring_transmissions"] += 1
            return

        # 检查是否需要维度转换
        if self._should_transfer_to_another_direction(next_node.node_id, flit, direction_str):
            # 尝试维度转换
            if self._try_dimension_turning(next_node, flit, direction_str, channel):
                current_buffer.pop(0)  # 成功转换，移除flit
                self.stats["dimension_turns"] += 1
            return

        # 检查下一个节点的缓冲区是否有空间
        if len(next_buffer) >= self.config.ring_buffer_depth:
            return  # 缓冲区满，无法移动

        # 移动flit到下一个节点
        flit = current_buffer.pop(0)
        next_buffer.append(flit)
        self.stats["ring_transmissions"] += 1

        # 检查是否是环绕连接
        if self._is_wrap_around_connection(current_node.node_id, next_node.node_id, direction_str):
            self.stats["wrap_around_hops"] += 1
            self.logger.debug(f"周期{self.cycle}：数据包{flit.packet_id}通过环绕连接从节点{current_node.node_id}到节点{next_node.node_id}")

    def _arbitrate_ring_access(self, node: CrossRingNode, direction: str, requester: str) -> bool:
        """Ring访问仲裁机制（简化版：优先级固定）"""
        # 简化仲裁：ring传输优先级高于inject
        # 这样可以确保ring中的flit能够正常流动

        if requester in ["ring_cw", "ring_ccw"]:
            # Ring传输请求总是被允许（除非有冲突）
            return True
        elif requester == "inject":
            # Inject请求在没有ring传输时被允许
            # 检查是否有ring传输正在进行
            ring_buffers = node.ring_buffers[direction]
            has_ring_traffic = False

            for channel in ["req", "rsp", "data"]:
                for ring_dir in ["clockwise", "counter_clockwise"]:
                    if ring_buffers[channel][ring_dir]:
                        has_ring_traffic = True
                        break
                if has_ring_traffic:
                    break

            # 如果没有ring流量，允许inject
            return not has_ring_traffic

        return False

    def _should_eject_flit_four_directions(self, node_id: NodeId, flit: Any) -> bool:
        """判断flit是否应该在当前节点eject（四方向系统）"""
        return node_id == flit.destination

    def _should_transfer_to_another_direction(self, node_id: NodeId, flit: Any, current_direction: str) -> bool:
        """判断是否需要转换到另一个方向（维度转换）- 支持XY和YX路由"""
        if not hasattr(flit, "remaining_directions") or not flit.remaining_directions:
            return False

        # 检查当前方向是否已完成路由
        node_x, node_y = self._get_node_coordinates(node_id)
        dst_x, dst_y = self._get_node_coordinates(flit.destination)

        if current_direction in ["TL", "TR"]:
            # 水平方向：检查是否到达目标列
            # 对于XY路由：到达目标列且需要垂直移动时转换
            # 对于YX路由：不应该在这里转换，因为垂直应该先完成
            return node_x == dst_x and node_y != dst_y
        else:  # 垂直方向 ["TU", "TD"]
            # 垂直方向：检查是否到达目标行
            # 对于XY路由：不应该在这里转换，因为水平应该先完成
            # 对于YX路由：到达目标行且需要水平移动时转换
            return node_y == dst_y and node_x != dst_x

    def _try_dimension_turning(self, node: CrossRingNode, flit: Any, from_direction: str, channel: str) -> bool:
        """尝试维度转换 - 支持XY和YX路由的双向转换"""
        if not hasattr(flit, "remaining_directions") or not flit.remaining_directions:
            return False

        next_direction = flit.remaining_directions[0]
        to_direction = next_direction.value

        # 使用交叉点模块进行维度转换
        cross_point = node.cross_point

        # 确定转换类型（支持双向转换）
        if from_direction in ["TL", "TR"] and to_direction in ["TU", "TD"]:
            # 水平到垂直转换（XY路由）
            success = cross_point.process_dimension_turning(flit, "horizontal", "vertical", self.cycle)
        elif from_direction in ["TU", "TD"] and to_direction in ["TL", "TR"]:
            # 垂直到水平转换（YX路由）
            success = cross_point.process_dimension_turning(flit, "vertical", "horizontal", self.cycle)
        else:
            return False

        if success:
            # 更新flit的方向信息
            flit.current_ring_direction = next_direction
            flit.remaining_directions = flit.remaining_directions[1:]

            # 将flit移动到新方向的缓冲区
            target_buffer = node.ring_buffers[to_direction][channel]
            if len(target_buffer) < self.config.ring_buffer_depth:
                target_buffer.append(flit)
                return True

        return False

    def _is_wrap_around_connection(self, from_node: NodeId, to_node: NodeId, direction: str) -> bool:
        """检查是否是环绕连接"""
        from_x, from_y = self._get_node_coordinates(from_node)
        to_x, to_y = self._get_node_coordinates(to_node)

        if direction in ["TL", "TR"]:
            # 水平方向的环绕
            if direction == "TR":
                # 顺时针：最右边到最左边
                return from_x == self.config.num_col - 1 and to_x == 0 and from_y == to_y
            else:
                # 逆时针：最左边到最右边
                return from_x == 0 and to_x == self.config.num_col - 1 and from_y == to_y
        else:
            # 垂直方向的环绕
            if direction == "TD":
                # 向下：最下面到最上面
                return from_y == self.config.num_row - 1 and to_y == 0 and from_x == to_x
            else:
                # 向上：最上面到最下面
                return from_y == 0 and to_y == self.config.num_row - 1 and from_x == to_x

    def _check_downstream_path_congestion_four_directions(self, source: NodeId, destination: NodeId, direction: RingDirection, channel: str) -> bool:
        """检查四方向系统中到目标节点路径上的拥塞状态"""
        # 获取路径上的关键节点
        path_nodes = self._get_path_nodes_four_directions(source, destination, direction)

        # 检查路径上每个节点的拥塞状态
        for node_id in path_nodes:
            if node_id in self.crossring_nodes:
                node = self.crossring_nodes[node_id]
                direction_str = direction.value

                # 检查该节点的ETag状态
                if node.etag_status[direction_str][channel]:
                    return True

                # 检查ring缓冲区占用率
                buffer = node.ring_buffers[direction_str][channel]
                if len(buffer) >= self.config.ring_buffer_depth * 0.7:  # 70%阈值
                    return True

        return False

    def _get_path_nodes_four_directions(self, source: NodeId, destination: NodeId, direction: RingDirection) -> List[NodeId]:
        """获取四方向系统中从源到目标在指定方向上的路径节点"""
        # 简化实现：计算在指定方向上的路径节点
        direction_nodes = []
        current = source

        # 限制最大跳数避免无限循环
        max_hops = self.config.num_nodes
        hops = 0

        while current != destination and hops < max_hops:
            next_node = self._get_next_node_in_direction(current, direction)
            if next_node == current:  # 到达边界，无法继续
                break
            direction_nodes.append(next_node)
            current = next_node
            hops += 1

        return direction_nodes

    def _process_ring_transmission(self) -> None:
        """处理ring网络内部传输"""
        # 处理水平ring传输
        self._process_horizontal_ring_transmission()

        # 处理垂直ring传输
        self._process_vertical_ring_transmission()

    def _process_horizontal_ring_transmission(self) -> None:
        """处理水平ring的传输"""
        for row in range(self.config.num_row):
            for direction in ["clockwise", "counter_clockwise"]:
                for channel in ["req", "rsp", "data"]:
                    self._advance_ring_flits("horizontal", row, direction, channel)

    def _process_vertical_ring_transmission(self) -> None:
        """处理垂直ring的传输"""
        for col in range(self.config.num_col):
            for direction in ["clockwise", "counter_clockwise"]:
                for channel in ["req", "rsp", "data"]:
                    self._advance_ring_flits("vertical", col, direction, channel)

    def _advance_ring_flits(self, ring_type: str, ring_index: int, direction: str, channel: str) -> None:
        """推进ring中的flits"""
        if ring_type == "horizontal":
            # 水平ring：同一行的所有节点
            nodes_in_ring = [ring_index * self.config.num_col + col for col in range(self.config.num_col)]
        else:  # vertical
            # 垂直ring：同一列的所有节点
            nodes_in_ring = [row * self.config.num_col + ring_index for row in range(self.config.num_row)]

        # 按传输方向排序节点
        if direction == "counter_clockwise":
            nodes_in_ring.reverse()

        # 处理每个节点的ring缓冲区（从后往前处理避免冲突）
        for i in reversed(range(len(nodes_in_ring))):
            node_id = nodes_in_ring[i]
            node_instance = self.crossring_nodes[node_id]
            ring_buffer = node_instance.ring_buffers[ring_type][channel][direction]

            if not ring_buffer:
                continue

            # 获取下一个节点
            next_node_index = (i + 1) % len(nodes_in_ring)
            next_node_id = nodes_in_ring[next_node_index]
            next_node_instance = self.crossring_nodes[next_node_id]

            # 尝试移动flit到下一个节点
            self._try_move_flit_in_ring(node_instance, next_node_instance, ring_type, direction, channel)

    def _try_move_flit_in_ring(self, current_node: CrossRingNode, next_node: CrossRingNode, ring_type: str, direction: str, channel: str) -> None:
        """尝试在ring中移动flit"""
        current_buffer = current_node.ring_buffers[ring_type][channel][direction]
        next_buffer = next_node.ring_buffers[ring_type][channel][direction]

        if not current_buffer:
            return

        flit = current_buffer[0]  # 查看队首flit

        # 检查是否到达最终目标节点
        if self._should_eject_flit(next_node.node_id, flit, ring_type):
            # 尝试移动到eject队列
            if self._try_move_to_eject(next_node, flit, channel):
                current_buffer.pop(0)  # 成功eject，移除flit
            return

        # 检查是否需要转换ring方向（从水平转垂直）
        if self._should_transfer_to_vertical_ring(next_node.node_id, flit, ring_type):
            # 尝试转移到垂直ring
            if self._try_transfer_to_vertical_ring(next_node, flit, channel):
                current_buffer.pop(0)  # 成功转移，移除flit
            return

        # 检查下一个节点的缓冲区是否有空间
        if len(next_buffer) >= self.config.ring_buffer_depth:
            return  # 缓冲区满，无法移动

        # 执行仲裁
        requester = "ring_cw" if direction == "clockwise" else "ring_ccw"
        if not self._arbitrate_ring_access(next_node, ring_type, requester):
            return  # 仲裁失败

        # 移动flit到下一个节点
        flit = current_buffer.pop(0)
        next_buffer.append(flit)

    def _should_eject_flit(self, node_id: NodeId, flit: Any, ring_type: str) -> bool:
        """判断flit是否应该在当前节点eject"""
        node_x, node_y = self._get_node_coordinates(node_id)
        dst_x, dst_y = self._get_node_coordinates(flit.destination)

        if ring_type == "horizontal":
            # 水平ring：当X坐标匹配且Y坐标也匹配时eject（到达最终目标）
            return node_x == dst_x and node_y == dst_y
        else:  # vertical
            # 垂直ring：当Y坐标匹配且X坐标也匹配时eject（到达最终目标）
            return node_x == dst_x and node_y == dst_y

    def _try_move_to_eject(self, node: CrossRingNode, flit: Any, channel: str) -> bool:
        """尝试将flit移动到eject队列"""
        eject_queue = node.eject_queues[channel]

        # 检查eject队列是否有空间
        if len(eject_queue) >= self.config.eject_buffer_depth:
            return False

        # 移动到eject队列
        eject_queue.append(flit)
        flit.is_arrive = True
        flit.arrival_network_cycle = self.cycle

        return True

    def _should_transfer_to_vertical_ring(self, node_id: NodeId, flit: Any, ring_type: str) -> bool:
        """判断是否应该从水平ring转到垂直ring"""
        if ring_type != "horizontal":
            return False  # 只有水平ring可以转到垂直ring

        node_x, node_y = self._get_node_coordinates(node_id)
        dst_x, dst_y = self._get_node_coordinates(flit.destination)

        # 当X坐标匹配但Y坐标不匹配时，需要转到垂直ring
        return node_x == dst_x and node_y != dst_y

    def _try_transfer_to_vertical_ring(self, node: CrossRingNode, flit: Any, channel: str) -> bool:
        """尝试将flit从水平ring转移到垂直ring"""
        # 确定垂直ring的方向
        node_y = self._get_node_coordinates(node.node_id)[1]
        dst_y = self._get_node_coordinates(flit.destination)[1]

        if dst_y > node_y:
            ring_direction = "clockwise"  # 向下
        else:
            ring_direction = "counter_clockwise"  # 向上

        # 检查垂直ring缓冲区是否有空间
        vertical_buffer = node.ring_buffers["vertical"][channel][ring_direction]
        if len(vertical_buffer) >= self.config.ring_buffer_depth:
            return False

        # 执行仲裁
        if not self._arbitrate_ring_access(node, "vertical", "inject"):
            return False

        # 转移到垂直ring
        vertical_buffer.append(flit)
        return True

    def _process_ring_to_eject(self) -> None:
        """处理ring网络到eject队列的提取"""
        # 这个方法主要在_advance_ring_flits中处理
        # 这里可以添加额外的eject逻辑，如优先级处理等
        pass

    def _update_etag_itag_status(self) -> None:
        """更新ETag/ITag状态用于拥塞控制"""
        for node_id, node in self.crossring_nodes.items():
            # 更新ETag状态（Egress Tag - 出口拥塞标记）
            self._update_etag_status(node)

            # 更新ITag状态（Ingress Tag - 入口拥塞标记）
            self._update_itag_status(node)

    def _update_etag_status(self, node: CrossRingNode) -> None:
        """更新ETag状态（四方向系统）"""
        for direction in ["TL", "TR", "TU", "TD"]:
            for channel in ["req", "rsp", "data"]:
                # 检查eject队列的拥塞情况
                eject_queue = node.eject_queues[channel]
                eject_threshold = self.config.eject_buffer_depth * 0.8  # 80%阈值

                # 检查ring缓冲区的拥塞情况
                ring_buffer = node.ring_buffers[direction][channel]
                ring_congestion = False

                buffer_occupancy = len(ring_buffer)
                ring_threshold = self.config.ring_buffer_depth * 0.8
                if buffer_occupancy >= ring_threshold:
                    ring_congestion = True

                # 设置ETag状态
                eject_congestion = len(eject_queue) >= eject_threshold
                node.etag_status[direction][channel] = eject_congestion or ring_congestion

    def _update_itag_status(self, node: CrossRingNode) -> None:
        """更新ITag状态（四方向系统）"""
        for direction in ["TL", "TR", "TU", "TD"]:
            for channel in ["req", "rsp", "data"]:
                # 检查inject队列的拥塞情况
                inject_queue = node.inject_queues[channel]
                inject_threshold = self.config.inject_buffer_depth * 0.8

                # 检查下游节点的ETag状态
                downstream_congestion = self._check_downstream_congestion_four_directions(node.node_id, direction, channel)

                # 设置ITag状态
                inject_congestion = len(inject_queue) >= inject_threshold
                node.itag_status[direction][channel] = inject_congestion or downstream_congestion

    def _check_downstream_congestion_four_directions(self, node_id: NodeId, direction: str, channel: str) -> bool:
        """检查下游节点的拥塞状态"""
        # 获取下游节点列表
        downstream_nodes = self._get_downstream_nodes(node_id, direction)

        # 检查下游节点的ETag状态
        for downstream_node in downstream_nodes:
            if downstream_node in self.crossring_nodes:
                downstream_node_instance = self.crossring_nodes[downstream_node]
                if downstream_node_instance.etag_status[direction][channel]:
                    return True

        return False

    def _get_downstream_nodes(self, node_id: NodeId, direction: str) -> List[NodeId]:
        """获取指定方向的下游节点"""
        x, y = self._get_node_coordinates(node_id)
        downstream_nodes = []

        try:
            # 将方向字符串转换为RingDirection枚举
            ring_direction = RingDirection(direction)

            # 获取下游节点
            downstream_node = self._get_next_node_in_direction(node_id, ring_direction)
            downstream_nodes.append(downstream_node)
        except Exception:
            # 如果转换失败，返回空列表
            pass

        return downstream_nodes

    def _check_downstream_path_congestion(self, source: NodeId, destination: NodeId, direction: str, channel: str) -> bool:
        """检查到目标节点路径上的拥塞状态"""
        # 获取路径上的关键节点
        path_nodes = self._get_path_nodes(source, destination, direction)

        # 检查路径上每个节点的拥塞状态
        for node_id in path_nodes:
            if node_id in self.crossring_nodes:
                node = self.crossring_nodes[node_id]

                # 检查该节点的ETag状态
                if node.etag_status[direction][channel]:
                    return True

                # 检查ring缓冲区占用率
                for ring_dir in ["clockwise", "counter_clockwise"]:
                    buffer = node.ring_buffers[direction][channel][ring_dir]
                    if len(buffer) >= self.config.ring_buffer_depth * 0.7:  # 70%阈值
                        return True

        return False

    def _get_path_nodes(self, source: NodeId, destination: NodeId, direction: str) -> List[NodeId]:
        """获取从源到目标在指定方向上的路径节点"""
        src_x, src_y = self._get_node_coordinates(source)
        dst_x, dst_y = self._get_node_coordinates(destination)
        path_nodes = []

        if direction == "horizontal":
            # 水平路径：同一行的节点
            y = src_y
            if dst_x > src_x:
                # 向右移动
                for x in range(src_x + 1, dst_x + 1):
                    node_id = y * self.config.num_col + x
                    path_nodes.append(node_id)
            else:
                # 向左移动
                for x in range(src_x - 1, dst_x - 1, -1):
                    node_id = y * self.config.num_col + x
                    path_nodes.append(node_id)
        else:  # vertical
            # 垂直路径：同一列的节点
            x = src_x
            if dst_y > src_y:
                # 向下移动
                for y in range(src_y + 1, dst_y + 1):
                    node_id = y * self.config.num_col + x
                    path_nodes.append(node_id)
            else:
                # 向上移动
                for y in range(src_y - 1, dst_y - 1, -1):
                    node_id = y * self.config.num_col + x
                    path_nodes.append(node_id)

        return path_nodes

    def _record_congestion_event(self, node_id: NodeId, event_type: str, direction: str, channel: str) -> None:
        """记录拥塞事件用于统计和调试"""
        if not hasattr(self, "congestion_stats"):
            self.congestion_stats = {}

        key = f"{node_id}_{direction}_{channel}"
        if key not in self.congestion_stats:
            self.congestion_stats[key] = {"etag_block": 0, "downstream_congestion": 0, "buffer_full": 0, "arbitration_fail": 0}

        if event_type in self.congestion_stats[key]:
            self.congestion_stats[key][event_type] += 1

    def _record_injection_success(self, node_id: NodeId, direction: str, channel: str) -> None:
        """记录成功注入事件"""
        if not hasattr(self, "injection_stats"):
            self.injection_stats = {}

        key = f"{node_id}_{direction}_{channel}"
        if key not in self.injection_stats:
            self.injection_stats[key] = 0

        self.injection_stats[key] += 1

    def get_congestion_statistics(self) -> Dict[str, Any]:
        """获取拥塞统计信息"""
        return {
            "congestion_events": getattr(self, "congestion_stats", {}),
            "injection_success": getattr(self, "injection_stats", {}),
            "total_congestion_events": sum(sum(events.values()) for events in getattr(self, "congestion_stats", {}).values()),
            "total_injections": sum(getattr(self, "injection_stats", {}).values()),
        }

    def _update_statistics(self) -> None:
        """更新性能统计"""
        # 更新活跃请求数
        current_active = self.get_active_request_count()
        self.stats["current_active_requests"] = current_active
        if current_active > self.stats["peak_active_requests"]:
            self.stats["peak_active_requests"] = current_active

        # 累计重试数
        total_read_retries = sum(ip.read_retry_num_stat for ip in self._ip_registry.values())
        total_write_retries = sum(ip.write_retry_num_stat for ip in self._ip_registry.values())
        self.stats["read_retries"] = total_read_retries
        self.stats["write_retries"] = total_write_retries

    def run_simulation(self, max_cycles: int = 10000, warmup_cycles: int = 1000, stats_start_cycle: int = 1000) -> Dict[str, Any]:
        """
        运行完整仿真

        Args:
            max_cycles: 最大仿真周期数
            warmup_cycles: 热身周期数
            stats_start_cycle: 统计开始周期

        Returns:
            仿真结果字典
        """
        self.logger.info(f"开始CrossRing仿真: max_cycles={max_cycles}, warmup={warmup_cycles}")

        self.is_running = True
        stats_enabled = False

        try:
            for cycle in range(1, max_cycles + 1):
                self.step()

                # 启用统计收集
                if cycle == stats_start_cycle:
                    stats_enabled = True
                    self._reset_statistics()
                    self.logger.info(f"周期 {cycle}: 开始收集统计数据")

                # 检查仿真结束条件
                if self._should_stop_simulation():
                    self.logger.info(f"周期 {cycle}: 检测到仿真结束条件")
                    break

                # 定期输出进度
                if cycle % 5000 == 0:
                    self.logger.info(f"仿真进度: {cycle}/{max_cycles} 周期")

        except KeyboardInterrupt:
            self.logger.warning("仿真被用户中断")

        except Exception as e:
            self.logger.error(f"仿真过程中发生错误: {e}")
            raise

        finally:
            self.is_running = False
            self.is_finished = True

        # 生成仿真结果
        results = self._generate_simulation_results(stats_start_cycle)
        self.logger.info(f"CrossRing仿真完成: 总周期={self.cycle}")

        return results

    def _should_stop_simulation(self) -> bool:
        """检查是否应该停止仿真"""
        # 简单的停止条件：没有活跃请求
        active_requests = self.get_active_request_count()

        # 如果连续1000个周期没有活跃请求，则停止
        if not hasattr(self, "_idle_cycles"):
            self._idle_cycles = 0

        if active_requests == 0:
            self._idle_cycles += 1
        else:
            self._idle_cycles = 0

        return self._idle_cycles >= 1000

    def _reset_statistics(self) -> None:
        """重置统计计数器"""
        for ip in self._ip_registry.values():
            ip.read_retry_num_stat = 0
            ip.write_retry_num_stat = 0
            # 可以重置更多统计信息

    def _generate_simulation_results(self, stats_start_cycle: int) -> Dict[str, Any]:
        """生成仿真结果"""
        effective_cycles = self.cycle - stats_start_cycle

        # 汇总IP接口统计
        ip_stats = {}
        for key, ip in self._ip_registry.items():
            ip_stats[key] = ip.get_status()

        # 计算平均延迟等指标
        total_transactions = sum(ip.rn_tracker_pointer["read"] + ip.rn_tracker_pointer["write"] for ip in self._ip_registry.values())

        results = {
            "simulation_info": {
                "total_cycles": self.cycle,
                "effective_cycles": effective_cycles,
                "config": self.config.to_dict(),
            },
            "global_stats": self.stats.copy(),
            "ip_interface_stats": ip_stats,
            "network_stats": {
                "total_transactions": total_transactions,
                "peak_active_requests": self.stats["peak_active_requests"],
                "total_read_retries": self.stats["read_retries"],
                "total_write_retries": self.stats["write_retries"],
            },
            "memory_stats": {
                "flit_pool": get_crossring_flit_pool_stats(),
            },
        }

        return results

    def get_active_request_count(self) -> int:
        """获取当前活跃请求总数"""
        total = 0
        for ip in self._ip_registry.values():
            total += len(ip.rn_tracker["read"])
            total += len(ip.rn_tracker["write"])
            total += len(ip.sn_tracker)
        return total

    def get_global_tracker_status(self) -> Dict[str, Any]:
        """
        获取全局tracker状态

        Returns:
            包含所有IP接口tracker状态的字典
        """
        status = {}
        for key, ip_interface in self._ip_registry.items():
            status[key] = {
                "rn_read_active": len(ip_interface.rn_tracker["read"]),
                "rn_write_active": len(ip_interface.rn_tracker["write"]),
                "rn_read_available": ip_interface.rn_tracker_count["read"],
                "rn_write_available": ip_interface.rn_tracker_count["write"],
                "sn_active": len(ip_interface.sn_tracker),
                "sn_ro_available": ip_interface.sn_tracker_count.get("ro", 0),
                "sn_share_available": ip_interface.sn_tracker_count.get("share", 0),
                "read_retries": ip_interface.read_retry_num_stat,
                "write_retries": ip_interface.write_retry_num_stat,
            }
        return status

    def print_debug_status(self) -> None:
        """打印调试状态"""
        status = self.get_global_tracker_status()
        print(f"\n=== CrossRing Model Cycle {self.cycle} Debug Status ===")
        print(f"活跃请求总数: {self.get_active_request_count()}")
        print(f"统计信息: {self.stats}")

        print("\nIP接口状态:")
        for ip_key, ip_status in status.items():
            print(
                f"  {ip_key}: RN({ip_status['rn_read_active']}R+{ip_status['rn_write_active']}W), "
                + f"SN({ip_status['sn_active']}), 重试({ip_status['read_retries']}R+{ip_status['write_retries']}W)"
            )

    def inject_test_traffic(self, source: NodeId, destination: NodeId, req_type: str, count: int = 1, burst_length: int = 4, ip_type: str = None) -> List[str]:
        """
        注入测试流量

        Args:
            source: 源节点
            destination: 目标节点
            req_type: 请求类型
            count: 请求数量
            burst_length: 突发长度
            ip_type: IP类型 (可选，如果未指定则使用第一个找到的IP)

        Returns:
            生成的packet_id列表
        """
        packet_ids = []

        # 找到源节点对应的IP接口
        ip_interface = self._find_ip_interface_for_request(source, req_type, ip_type)
        
        if not ip_interface:
            if ip_type:
                self.logger.warning(f"源节点 {source} 没有找到对应的IP接口 (类型: {ip_type})")
            else:
                self.logger.warning(f"源节点 {source} 没有找到对应的IP接口")
            return packet_ids

        for i in range(count):
            # TODO: packet_id 不需要这么复杂，就用数字就可以了，但是要保证唯一性
            packet_id = f"test_{source}_{destination}_{req_type}_{self.cycle}_{i}"
            success = ip_interface.enqueue_request(source=source, destination=destination, req_type=req_type, burst_length=burst_length, packet_id=packet_id)

            if success:
                packet_ids.append(packet_id)
                self.logger.debug(f"注入测试请求: {packet_id}")
            else:
                self.logger.warning(f"测试请求注入失败: {packet_id}")

        return packet_ids

    def _find_ip_interface_for_request(self, node_id: NodeId, req_type: str, ip_type: str = None) -> Optional[CrossRingIPInterface]:
        """
        为请求查找合适的IP接口

        Args:
            node_id: 节点ID
            req_type: 请求类型 ("read" | "write")
            ip_type: IP类型 (可选)

        Returns:
            找到的IP接口，如果未找到则返回None
        """
        if ip_type:
            # 如果指定了IP类型，则精确匹配
            matching_ips = [ip for key, ip in self._ip_registry.items() if ip.node_id == node_id and ip.ip_type == ip_type]
        else:
            # 如果未指定IP类型，则根据请求类型选择合适的IP
            matching_ips = [ip for key, ip in self._ip_registry.items() if ip.node_id == node_id]
            
            # 优先选择能发起该类型请求的IP
            if req_type == "read":
                # 对于读请求，优先选择DMA类型的IP
                preferred_ips = [ip for ip in matching_ips if ip.ip_type in ["gdma", "sdma", "cdma"]]
                if preferred_ips:
                    matching_ips = preferred_ips
            elif req_type == "write":
                # 对于写请求，优先选择DMA类型的IP
                preferred_ips = [ip for ip in matching_ips if ip.ip_type in ["gdma", "sdma", "cdma"]]
                if preferred_ips:
                    matching_ips = preferred_ips

        if not matching_ips:
            return None

        # 返回第一个匹配的IP接口
        return matching_ips[0]

    def _find_ip_interface_for_response(self, node_id: NodeId, req_type: str, ip_type: str = None) -> Optional[CrossRingIPInterface]:
        """
        为响应查找合适的IP接口

        Args:
            node_id: 节点ID
            req_type: 请求类型 ("read" | "write")
            ip_type: IP类型 (可选)

        Returns:
            找到的IP接口，如果未找到则返回None
        """
        if ip_type:
            # 如果指定了IP类型，则精确匹配
            matching_ips = [ip for key, ip in self._ip_registry.items() if ip.node_id == node_id and ip.ip_type == ip_type]
        else:
            # 如果未指定IP类型，则根据请求类型选择合适的IP
            matching_ips = [ip for key, ip in self._ip_registry.items() if ip.node_id == node_id]
            
            # 对于响应，优先选择能处理该类型请求的SN端IP
            if req_type in ["read", "write"]:
                # 优先选择存储类型的IP (DDR, L2M)
                preferred_ips = [ip for ip in matching_ips if ip.ip_type in ["ddr", "l2m"]]
                if preferred_ips:
                    matching_ips = preferred_ips

        if not matching_ips:
            return None

        # 返回第一个匹配的IP接口
        return matching_ips[0]

    def create_demo_traffic(self, pattern: str = "basic") -> List[Tuple]:
        """
        创建演示流量
        
        Args:
            pattern: 流量模式 ("basic", "stress", "mixed")
            
        Returns:
            流量列表，每个元素为 (源节点, 目标节点, 操作类型, 突发长度, IP类型)
        """
        traffic = []
        
        # 获取IP节点配置
        gdma_nodes = self.config.gdma_send_position_list
        ddr_nodes = self.config.ddr_send_position_list
        l2m_nodes = self.config.l2m_send_position_list
        
        if pattern == "basic":
            # 基础测试：GDMA读写DDR和L2M
            for gdma_node in gdma_nodes:
                for ddr_node in ddr_nodes:
                    traffic.append((gdma_node, ddr_node, "read", 4, "gdma"))
                    traffic.append((gdma_node, ddr_node, "write", 4, "gdma"))
                
                for l2m_node in l2m_nodes:
                    traffic.append((gdma_node, l2m_node, "read", 4, "gdma"))
                    traffic.append((gdma_node, l2m_node, "write", 4, "gdma"))
        
        elif pattern == "stress":
            # 压力测试：更多的请求和更大的突发
            for gdma_node in gdma_nodes:
                for ddr_node in ddr_nodes:
                    for _ in range(3):  # 多次请求
                        traffic.append((gdma_node, ddr_node, "read", 8, "gdma"))
                        traffic.append((gdma_node, ddr_node, "write", 8, "gdma"))
        
        elif pattern == "mixed":
            # 混合测试：不同的IP类型和操作
            # GDMA访问
            for gdma_node in gdma_nodes:
                for target in ddr_nodes + l2m_nodes:
                    traffic.append((gdma_node, target, "read", 4, "gdma"))
                    traffic.append((gdma_node, target, "write", 4, "gdma"))
            
            # SDMA访问
            if self.config.sdma_send_position_list:
                for sdma_node in self.config.sdma_send_position_list:
                    for target in ddr_nodes:
                        traffic.append((sdma_node, target, "read", 4, "sdma"))
            
            # CDMA访问
            if self.config.cdma_send_position_list:
                for cdma_node in self.config.cdma_send_position_list:
                    for target in l2m_nodes:
                        traffic.append((cdma_node, target, "write", 4, "cdma"))
        
        self.logger.info(f"生成 {pattern} 模式流量: {len(traffic)} 个请求")
        return traffic

    def inject_demo_traffic(self, traffic_list: List[Tuple]) -> int:
        """
        注入演示流量
        
        Args:
            traffic_list: 流量列表
            
        Returns:
            成功注入的请求数量
        """
        injected_count = 0
        
        for src, dst, op_type, burst, ip_type in traffic_list:
            # 确保节点在有效范围内
            if src < self.config.num_nodes and dst < self.config.num_nodes:
                packet_ids = self.inject_test_traffic(
                    source=src,
                    destination=dst,
                    req_type=op_type,
                    count=1,
                    burst_length=burst,
                    ip_type=ip_type
                )
                
                if packet_ids:
                    injected_count += len(packet_ids)
                    self.logger.debug(f"注入请求: {src}({ip_type}) -> {dst}, {op_type}, burst={burst}")
        
        self.logger.info(f"成功注入 {injected_count} 个请求")
        return injected_count

    def inject_from_traffic_file(self, traffic_file_path: str, max_requests: int = None) -> int:
        """
        从traffic文件注入流量
        
        Args:
            traffic_file_path: traffic文件路径
            max_requests: 最大请求数（可选）
            
        Returns:
            成功注入的请求数量
        """
        injected_count = 0
        
        try:
            with open(traffic_file_path, 'r') as f:
                for line_num, line in enumerate(f, 1):
                    line = line.strip()
                    if not line or line.startswith('#'):
                        continue
                    
                    # 解析traffic文件格式：cycle src src_type dst dst_type op burst
                    parts = line.split()
                    if len(parts) < 7:
                        continue
                    
                    cycle, src, src_type, dst, dst_type, op, burst = parts[:7]
                    
                    # 转换类型
                    src = int(src)
                    dst = int(dst)
                    burst = int(burst)
                    
                    # 确保节点在有效范围内
                    if src < self.config.num_nodes and dst < self.config.num_nodes:
                        packet_ids = self.inject_test_traffic(
                            source=src,
                            destination=dst,
                            req_type=op.lower(),
                            count=1,
                            burst_length=burst,
                            ip_type=src_type
                        )
                        
                        if packet_ids:
                            injected_count += len(packet_ids)
                    
                    # 检查是否达到最大请求数
                    if max_requests and injected_count >= max_requests:
                        break
                        
        except FileNotFoundError:
            self.logger.error(f"Traffic文件不存在: {traffic_file_path}")
        except Exception as e:
            self.logger.error(f"读取traffic文件失败: {e}")
        
        self.logger.info(f"从文件注入 {injected_count} 个请求")
        return injected_count

    def analyze_simulation_results(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """
        分析仿真结果
        
        Args:
            results: 仿真结果
            
        Returns:
            分析结果
        """
        analysis = {}
        
        # 基础指标
        sim_info = results.get('simulation_info', {})
        network_stats = results.get('network_stats', {})
        
        analysis['basic_metrics'] = {
            'total_cycles': sim_info.get('total_cycles', 0),
            'effective_cycles': sim_info.get('effective_cycles', 0),
            'peak_active_requests': network_stats.get('peak_active_requests', 0),
            'total_read_retries': network_stats.get('total_read_retries', 0),
            'total_write_retries': network_stats.get('total_write_retries', 0),
            'total_transactions': network_stats.get('total_transactions', 0)
        }
        
        # 计算吞吐量
        effective_cycles = analysis['basic_metrics']['effective_cycles']
        total_transactions = analysis['basic_metrics']['total_transactions']
        
        if effective_cycles > 0 and total_transactions > 0:
            analysis['basic_metrics']['throughput'] = total_transactions / effective_cycles
            analysis['basic_metrics']['bandwidth_mbps'] = (total_transactions * 512) / (effective_cycles * 1e6)  # 假设512位/事务
        else:
            analysis['basic_metrics']['throughput'] = 0.0
            analysis['basic_metrics']['bandwidth_mbps'] = 0.0
        
        # IP接口分析
        ip_stats = results.get('ip_interface_stats', {})
        analysis['ip_summary'] = self._analyze_ip_interfaces(ip_stats)
        
        # 拥塞分析
        analysis['congestion_summary'] = self._analyze_congestion()
        
        return analysis

    def _analyze_ip_interfaces(self, ip_stats: Dict[str, Any]) -> Dict[str, Any]:
        """分析IP接口统计"""
        summary = {
            'total_interfaces': len(ip_stats),
            'by_type': {},
            'total_read_transactions': 0,
            'total_write_transactions': 0,
            'total_retries': 0
        }
        
        for ip_key, stats in ip_stats.items():
            ip_type = ip_key.split('_')[0]
            
            if ip_type not in summary['by_type']:
                summary['by_type'][ip_type] = {
                    'count': 0,
                    'read_transactions': 0,
                    'write_transactions': 0,
                    'retries': 0
                }
            
            summary['by_type'][ip_type]['count'] += 1
            summary['by_type'][ip_type]['read_transactions'] += stats.get('rn_read_active', 0)
            summary['by_type'][ip_type]['write_transactions'] += stats.get('rn_write_active', 0)
            summary['by_type'][ip_type]['retries'] += stats.get('read_retries', 0) + stats.get('write_retries', 0)
            
            summary['total_read_transactions'] += stats.get('rn_read_active', 0)
            summary['total_write_transactions'] += stats.get('rn_write_active', 0)
            summary['total_retries'] += stats.get('read_retries', 0) + stats.get('write_retries', 0)
        
        return summary

    def _analyze_congestion(self) -> Dict[str, Any]:
        """分析拥塞情况"""
        congestion_summary = {
            'congestion_detected': False,
            'total_congestion_events': 0,
            'congestion_rate': 0.0
        }
        
        if hasattr(self, 'get_congestion_statistics'):
            congestion_stats = self.get_congestion_statistics()
            total_congestion = congestion_stats.get('total_congestion_events', 0)
            total_injections = congestion_stats.get('total_injections', 1)
            
            congestion_summary['congestion_detected'] = total_congestion > 0
            congestion_summary['total_congestion_events'] = total_congestion
            congestion_summary['congestion_rate'] = total_congestion / total_injections if total_injections > 0 else 0.0
        
        return congestion_summary

    def generate_simulation_report(self, results: Dict[str, Any], analysis: Dict[str, Any] = None) -> str:
        """
        生成仿真报告
        
        Args:
            results: 仿真结果
            analysis: 分析结果（可选，如果未提供则自动分析）
            
        Returns:
            报告文本
        """
        if analysis is None:
            analysis = self.analyze_simulation_results(results)
        
        report = []
        report.append("=" * 60)
        report.append("CrossRing NoC 仿真报告")
        report.append("=" * 60)
        
        # 拓扑信息
        report.append(f"拓扑配置: {self.config.num_row}x{self.config.num_col}")
        report.append(f"总节点数: {self.config.num_nodes}")
        report.append("")
        
        # 基础指标
        basic = analysis.get('basic_metrics', {})
        report.append("性能指标:")
        report.append(f"  仿真周期: {basic.get('total_cycles', 0):,}")
        report.append(f"  有效周期: {basic.get('effective_cycles', 0):,}")
        report.append(f"  总事务数: {basic.get('total_transactions', 0):,}")
        report.append(f"  峰值活跃请求: {basic.get('peak_active_requests', 0)}")
        report.append(f"  吞吐量: {basic.get('throughput', 0):.4f} 事务/周期")
        report.append(f"  带宽: {basic.get('bandwidth_mbps', 0):.2f} Mbps")
        report.append("")
        
        # 重试统计
        report.append("重试统计:")
        report.append(f"  读重试: {basic.get('total_read_retries', 0)}")
        report.append(f"  写重试: {basic.get('total_write_retries', 0)}")
        report.append("")
        
        # IP接口统计
        ip_summary = analysis.get('ip_summary', {})
        report.append("IP接口统计:")
        report.append(f"  总接口数: {ip_summary.get('total_interfaces', 0)}")
        
        by_type = ip_summary.get('by_type', {})
        for ip_type, stats in by_type.items():
            report.append(f"  {ip_type}: {stats['count']}个接口, "
                         f"读事务={stats['read_transactions']}, "
                         f"写事务={stats['write_transactions']}, "
                         f"重试={stats['retries']}")
        
        report.append("")
        
        # 拥塞分析
        congestion = analysis.get('congestion_summary', {})
        if congestion.get('congestion_detected', False):
            report.append("拥塞分析:")
            report.append(f"  拥塞事件: {congestion.get('total_congestion_events', 0)}")
            report.append(f"  拥塞率: {congestion.get('congestion_rate', 0):.2%}")
        else:
            report.append("拥塞分析: 未检测到显著拥塞")
        
        report.append("")
        report.append("=" * 60)
        
        return "\n".join(report)

    def run_demo_simulation(self, 
                           traffic_pattern: str = "basic",
                           max_cycles: int = 1000,
                           warmup_cycles: int = 100,
                           stats_start_cycle: int = 100) -> Dict[str, Any]:
        """
        运行演示仿真
        
        Args:
            traffic_pattern: 流量模式
            max_cycles: 最大仿真周期
            warmup_cycles: 热身周期
            stats_start_cycle: 统计开始周期
            
        Returns:
            包含仿真结果和分析的字典
        """
        self.logger.info(f"开始演示仿真: {traffic_pattern} 模式")
        
        # 生成和注入流量
        traffic = self.create_demo_traffic(traffic_pattern)
        injected_count = self.inject_demo_traffic(traffic)
        
        if injected_count == 0:
            self.logger.warning("没有成功注入任何请求")
            return {"success": False, "message": "No requests injected"}
        
        # 运行仿真
        results = self.run_simulation(
            max_cycles=max_cycles,
            warmup_cycles=warmup_cycles,
            stats_start_cycle=stats_start_cycle
        )
        
        # 分析结果
        analysis = self.analyze_simulation_results(results)
        
        # 生成报告
        report = self.generate_simulation_report(results, analysis)
        
        return {
            "success": True,
            "injected_requests": injected_count,
            "simulation_results": results,
            "analysis": analysis,
            "report": report
        }

    def get_model_summary(self) -> Dict[str, Any]:
        """获取模型摘要信息"""
        return {
            "config_name": self.config.config_name,
            "topology": f"{self.config.num_row}x{self.config.num_col}",
            "total_nodes": self.config.num_nodes,
            "ip_interfaces": len(self.ip_interfaces),
            "current_cycle": self.cycle,
            "active_requests": self.get_active_request_count(),
            "simulation_status": {
                "is_running": self.is_running,
                "is_finished": self.is_finished,
            },
            "networks_ready": len(self.crossring_nodes) > 0,
        }

    def cleanup(self) -> None:
        """清理资源"""
        self.logger.info("开始清理CrossRing模型资源")

        # 清理IP接口
        for ip in self.ip_interfaces.values():
            # 这里可以添加IP接口的清理逻辑
            pass

        # 清理网络组件
        self.crossring_nodes.clear()

        # 清理统计信息
        self.stats.clear()

        self.logger.info("CrossRing模型资源清理完成")

    def __del__(self):
        """析构函数"""
        if hasattr(self, "logger"):
            self.logger.debug("CrossRing模型对象被销毁")

    @property
    def total_active_requests(self) -> int:
        """总活跃请求数（属性访问）"""
        return self.get_active_request_count()

    def __repr__(self) -> str:
        """字符串表示"""
        return (
            f"CrossRingModel({self.config.config_name}, "
            f"{self.config.num_row}x{self.config.num_col}, "
            f"cycle={self.cycle}, "
            f"active_requests={self.get_active_request_count()})"
        )

    # ========== 统一接口方法（用于兼容性） ==========

    def initialize_network(self) -> None:
        """初始化网络（统一接口）"""
        self._setup_ip_interfaces()
        self._setup_crossring_networks()
        print(f"CrossRing网络初始化完成: {self.config.num_row}x{self.config.num_col}")

    def advance_cycle(self) -> None:
        """推进一个周期（统一接口）"""
        self.step()

    def inject_packet(self, src_node: NodeId, dst_node: NodeId, op_type: str = "R", burst_size: int = 4, cycle: int = None, packet_id: str = None) -> bool:
        """注入包（统一接口）"""
        if cycle is None:
            cycle = self.cycle

        # 生成包ID
        if packet_id is None:
            pa, cket_id = f"pkt_{src_node}_{dst_node}_{op_type}_{cycle}"

        # 开始追踪请求
        if self.debug_enabled or packet_id in self.trace_packets:
            self.request_tracker.start_request(packet_id, src_node, dst_node, op_type, burst_size, cycle)

        # 使用现有的inject_test_traffic方法
        packet_ids = self.inject_test_traffic(src_node, dst_node, op_type, 1, burst_size)

        if len(packet_ids) > 0 and self.debug_enabled:
            self.request_tracker.update_request_state(packet_id, RequestState.INJECTED, cycle)

        return len(packet_ids) > 0

    def get_completed_packets(self) -> List[Dict[str, Any]]:
        """获取已完成的包（统一接口）"""
        completed_packets = []

        # 从请求跟踪器中获取已完成的包
        if hasattr(self, "request_tracker") and self.request_tracker:
            for packet_id, lifecycle in self.request_tracker.completed_requests.items():
                if lifecycle.current_state == RequestState.COMPLETED and not lifecycle.reported:
                    completed_packets.append(
                        {
                            "packet_id": packet_id,
                            "source": lifecycle.source,
                            "destination": lifecycle.destination,
                            "op_type": lifecycle.op_type,
                            "burst_size": lifecycle.burst_size,
                            "injected_cycle": lifecycle.injected_cycle,
                            "completed_cycle": lifecycle.completed_cycle,
                            "latency": lifecycle.completed_cycle - lifecycle.injected_cycle if lifecycle.completed_cycle else 0,
                            "data_flit_count": lifecycle.burst_size if lifecycle.op_type == "R" else 0,
                        }
                    )
                    # 标记为已报告
                    lifecycle.reported = True

        return completed_packets

    def _simulate_packet_completion(self):
        """简化的包完成模拟逻辑（用于demo）"""
        if not hasattr(self, "request_tracker") or not self.request_tracker:
            return

        # 模拟延迟：假设包在注入后10-20个周期完成
        # 使用列表拷贝避免在迭代时修改字典
        active_packets = list(self.request_tracker.active_requests.items())
        for packet_id, lifecycle in active_packets:
            if lifecycle.current_state == RequestState.INJECTED:
                latency = self.cycle - lifecycle.injected_cycle

                # 简单的完成条件：延迟达到10-20周期（基于距离和类型）
                expected_latency = 10 + (abs(lifecycle.source - lifecycle.destination) * 2)
                if lifecycle.op_type == "R":
                    expected_latency += 5  # 读操作需要更长时间

                if latency >= expected_latency:
                    # 标记为完成
                    self.request_tracker.update_request_state(packet_id, RequestState.COMPLETED, self.cycle)
                    lifecycle.completed_cycle = self.cycle
                    self.logger.debug(f"包 {packet_id} 在周期 {self.cycle} 完成，延迟 {latency} 周期")

    def get_network_statistics(self) -> Dict[str, Any]:
        """获取网络统计（统一接口）"""
        return {
            "cycle": self.cycle,
            "total_packets_injected": 0,  # 需要真实统计
            "total_packets_completed": 0,
            "active_packets": self.get_active_request_count(),
            "avg_latency": 0.0,
            "avg_hops": 0.0,
            "utilization": 0.0,
            "throughput": 0.0,
        }

    def get_node_count(self) -> int:
        """获取节点数量（统一接口）"""
        return self.config.num_nodes

    # ========== 调试功能接口 ==========

    def enable_debug(self, level: int = 1, trace_packets: List[str] = None):
        """启用调试模式

        Args:
            level: 调试级别 (1-3)
            trace_packets: 要追踪的特定包ID列表
        """
        self.debug_enabled = True
        self.request_tracker.enable_debug(level, trace_packets)

        if trace_packets:
            self.trace_packets.update(trace_packets)

        self.logger.info(f"调试模式已启用，级别: {level}")
        if trace_packets:
            self.logger.info(f"追踪包: {trace_packets}")

    def track_packet(self, packet_id: str):
        """添加要追踪的包"""
        self.trace_packets.add(packet_id)
        self.request_tracker.track_packet(packet_id)

    def debug_func(self):
        """主调试函数，每个周期调用"""
        if not self.debug_enabled:
            return

        # 打印网络状态
        self.request_tracker.print_network_state(self.cycle)

        # 追踪特定包的详细信息
        if self.trace_packets:
            for packet_id in self.trace_packets:
                self._print_packet_trace(packet_id)

    def _print_packet_trace(self, packet_id: str):
        """打印特定包的追踪信息"""
        lifecycle = self.request_tracker.get_request_status(packet_id)
        if not lifecycle:
            return

        print(f"\n=== 包 {packet_id} 追踪信息 (周期 {self.cycle}) ===")
        print(f"状态: {lifecycle.current_state.value}")
        print(f"源: {lifecycle.source} -> 目标: {lifecycle.destination}")
        print(f"操作: {lifecycle.op_type}, 突发: {lifecycle.burst_size}")

        if lifecycle.current_state != RequestState.CREATED:
            print(f"延迟: {self.cycle - lifecycle.injected_cycle} 周期")

        # 显示当前在网络中的位置
        self._print_packet_network_positions(packet_id)
        time.sleep(0.3)

    def _print_packet_network_positions(self, packet_id: str):
        """打印包在网络中的当前位置"""
        found_positions = []

        # 检查所有节点的inject/eject队列和ring缓冲区
        for node_id, node in self.crossring_nodes.items():
            # 检查inject队列
            for channel in ["req", "rsp", "data"]:
                for flit in node.inject_queues[channel]:
                    if hasattr(flit, "packet_id") and flit.packet_id == packet_id:
                        found_positions.append(f"节点{node_id}-inject-{channel}")

                # 检查eject队列
                for flit in node.eject_queues[channel]:
                    if hasattr(flit, "packet_id") and flit.packet_id == packet_id:
                        found_positions.append(f"节点{node_id}-eject-{channel}")

                # 检查ring缓冲区
                for direction in ["TL", "TR", "TU", "TD"]:
                    for flit in node.ring_buffers[direction][channel]:
                        if hasattr(flit, "packet_id") and flit.packet_id == packet_id:
                            found_positions.append(f"节点{node_id}-ring-{direction}-{channel}")

        if found_positions:
            print(f"当前位置: {', '.join(found_positions)}")
        else:
            print("未在网络中找到此包")

    def validate_traffic_correctness(self) -> Dict[str, Any]:
        """验证流量的正确性"""
        stats = self.request_tracker.get_statistics()

        validation_result = {
            "total_requests": stats["total_requests"],
            "completed_requests": stats["completed_requests"],
            "failed_requests": stats["failed_requests"],
            "completion_rate": stats["completed_requests"] / max(1, stats["total_requests"]) * 100,
            "response_errors": stats["response_errors"],
            "data_errors": stats["data_errors"],
            "avg_latency": stats["avg_latency"],
            "max_latency": stats["max_latency"],
            "is_correct": stats["response_errors"] == 0 and stats["data_errors"] == 0,
        }

        return validation_result

    def print_debug_report(self):
        """打印调试报告"""
        if not self.debug_enabled:
            print("调试模式未启用")
            return

        self.request_tracker.print_final_report()

        # 打印验证结果
        validation = self.validate_traffic_correctness()
        print(f"\n流量正确性验证:")
        print(f"  完成率: {validation['completion_rate']:.1f}%")
        print(f"  响应错误: {validation['response_errors']}")
        print(f"  数据错误: {validation['data_errors']}")
        print(f"  结果: {'正确' if validation['is_correct'] else '有错误'}")

    def get_debug_statistics(self) -> Dict[str, Any]:
        """获取调试统计信息"""
        return self.request_tracker.get_statistics()

    # ========== 实现BaseNoCModel抽象方法 ==========

    def _setup_topology_network(self) -> None:
        """设置拓扑网络（拓扑特定）"""
        # CrossRing拓扑已在__init__中设置
        pass

    def _step_topology_network(self) -> None:
        """拓扑网络步进（拓扑特定）"""
        # 使用现有的advance_cycle逻辑
        pass

    def _get_topology_info(self) -> Dict[str, Any]:
        """获取拓扑信息（拓扑特定）"""
        return {
            "topology_type": "CrossRing",
            "num_row": self.config.num_row,
            "num_col": self.config.num_col,
            "total_nodes": self.config.num_nodes,
            "ring_directions": ["TL", "TR", "TU", "TD"],
            "channels": ["req", "rsp", "data"],
        }

    def _calculate_path(self, source: NodeId, destination: NodeId) -> List[NodeId]:
        """计算路径（拓扑特定）"""
        # 使用现有的路径计算逻辑
        return self._calculate_crossring_path(source, destination)


def create_crossring_model(config_name: str = "default", num_row: int = 5, num_col: int = 4, **config_kwargs) -> CrossRingModel:
    """
    创建CrossRing模型的便捷函数

    Args:
        config_name: 配置名称
        num_row: 行数
        num_col: 列数
        **config_kwargs: 其他配置参数

    Returns:
        CrossRing模型实例
    """
    config = CrossRingConfig(num_col=num_col, num_row=num_row, config_name=config_name)

    # 应用额外的配置参数
    if config_kwargs:
        config.from_dict(config_kwargs)

    return CrossRingModel(config)
