"""
CrossRing主模型类实现。

基于C2C仓库的架构，提供完整的CrossRing NoC仿真模型，
包括IP接口管理、网络组件和仿真循环控制。
"""

from __future__ import annotations
from typing import Dict, List, Any, Optional, Tuple
import logging
from collections import defaultdict

from .config import CrossRingConfig
from .ip_interface import CrossRingIPInterface
from .flit import CrossRingFlit, get_crossring_flit_pool_stats
from src.noc.utils.types import NodeId


class CrossRingModel:
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
        self.config = config
        self.cycle = 0

        # IP接口管理
        self.ip_interfaces: Dict[str, CrossRingIPInterface] = {}
        self._ip_registry: Dict[str, CrossRingIPInterface] = {}  # 全局debug视图

        # CrossRing网络组件（暂时为骨架，后续实现）
        self.crossring_pieces: Dict[NodeId, Any] = {}  # {node_id: CrossRingPiece}
        self.networks = {"req": None, "rsp": None, "data": None}  # REQ网络（后续实现）  # RSP网络（后续实现）  # DATA网络（后续实现）

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
        }

        # 仿真状态
        self.is_running = False
        self.is_finished = False

        # 日志配置
        self.logger = logging.getLogger(f"CrossRingModel_{id(self)}")

        # 初始化所有组件
        self._setup_ip_interfaces()
        self._setup_crossring_networks()

        self.logger.info(f"CrossRing模型初始化完成: {config.num_row}x{config.num_col}, {len(self.ip_interfaces)}个IP接口")

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
        """设置CrossRing网络组件的完整实现"""
        # 创建CrossRing网络的真实组件

        for node_id in range(self.config.num_nodes):
            coordinates = self._get_node_coordinates(node_id)

            # 为每个节点创建完整的CrossRing piece
            self.crossring_pieces[node_id] = {
                "node_id": node_id,
                "coordinates": coordinates,
                # Inject/Eject队列（每个通道独立）
                "inject_queues": {"req": [], "rsp": [], "data": []},
                "eject_queues": {"req": [], "rsp": [], "data": []},
                # Ring缓冲区（水平和垂直方向）
                "ring_buffers": {
                    "horizontal": {
                        "req": {"clockwise": [], "counter_clockwise": []},
                        "rsp": {"clockwise": [], "counter_clockwise": []},
                        "data": {"clockwise": [], "counter_clockwise": []},
                    },
                    "vertical": {
                        "req": {"clockwise": [], "counter_clockwise": []},
                        "rsp": {"clockwise": [], "counter_clockwise": []},
                        "data": {"clockwise": [], "counter_clockwise": []},
                    },
                },
                # ETag/ITag状态管理
                "etag_status": {"horizontal": {"req": False, "rsp": False, "data": False}, "vertical": {"req": False, "rsp": False, "data": False}},
                "itag_status": {"horizontal": {"req": False, "rsp": False, "data": False}, "vertical": {"req": False, "rsp": False, "data": False}},
                # 仲裁状态
                "arbitration_state": {
                    "horizontal_priority": "inject",  # inject, ring_cw, ring_ccw
                    "vertical_priority": "inject",
                    "last_arbitration": {"horizontal": 0, "vertical": 0},
                },
            }

        self.logger.info(f"CrossRing网络组件创建完成: {len(self.crossring_pieces)}个节点")

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
        """执行一个仿真周期"""
        self.cycle += 1

        # 处理所有IP接口
        for ip_interface in self.ip_interfaces.values():
            ip_interface.step(self.cycle)

        # 处理CrossRing网络（后续实现）
        self._step_crossring_networks()

        # 更新统计信息
        self._update_statistics()

        # 定期输出调试信息
        if self.cycle % 1000 == 0:
            self.logger.debug(f"周期 {self.cycle}: {self.get_active_request_count()}个活跃请求")

    def _step_crossring_networks(self) -> None:
        """
        处理CrossRing网络的完整传输逻辑

        实现真实的CrossRing网络传输：
        1. 重置仲裁状态
        2. inject_queue → ring网络注入
        3. ring网络内部传输（水平和垂直）
        4. ring网络 → eject_queue提取
        5. ETag/ITag处理和拥塞控制
        """
        # 第零阶段：重置每个节点的仲裁状态
        self._reset_arbitration_states()

        # 第一阶段：处理inject队列到ring网络的注入
        self._process_inject_to_ring()

        # 第二阶段：处理ring网络内部传输
        self._process_ring_transmission()

        # 第三阶段：处理ring网络到eject队列的提取
        self._process_ring_to_eject()

        # 第四阶段：更新ETag/ITag状态
        self._update_etag_itag_status()

    def _reset_arbitration_states(self) -> None:
        """重置所有节点的仲裁状态"""
        for piece in self.crossring_pieces.values():
            # 重置仲裁优先级为初始状态
            piece["arbitration_state"]["horizontal_priority"] = "inject"
            piece["arbitration_state"]["vertical_priority"] = "inject"

    def _process_inject_to_ring(self) -> None:
        """处理inject队列到ring网络的注入"""
        for node_id, piece in self.crossring_pieces.items():
            for channel in ["req", "rsp", "data"]:
                inject_queue = piece["inject_queues"][channel]

                # 处理inject队列中的每个flit
                while inject_queue:
                    flit = inject_queue[0]  # 查看队首flit

                    # 确定路由方向（水平或垂直优先）
                    route_direction = self._determine_route_direction(node_id, flit.destination)

                    if route_direction is None:
                        # 目标就是当前节点，直接移到eject队列
                        flit = inject_queue.pop(0)
                        piece["eject_queues"][channel].append(flit)
                        flit.is_arrive = True
                        flit.arrival_network_cycle = self.cycle
                        continue

                    # 尝试注入到ring网络
                    if self._try_inject_to_ring(piece, flit, channel, route_direction):
                        inject_queue.pop(0)  # 成功注入，移除flit
                    else:
                        break  # 注入失败，等待下个周期

    def _determine_route_direction(self, source: NodeId, destination: NodeId) -> Optional[str]:
        """确定路由方向（水平或垂直优先）"""
        if source == destination:
            return None

        src_x, src_y = self._get_node_coordinates(source)
        dst_x, dst_y = self._get_node_coordinates(destination)

        # CrossRing使用XY路由：先水平后垂直
        if src_x != dst_x:
            return "horizontal"
        elif src_y != dst_y:
            return "vertical"
        else:
            return None

    def _try_inject_to_ring(self, piece: Dict, flit: Any, channel: str, direction: str) -> bool:
        """尝试将flit注入到ring网络（增强拥塞控制）"""
        # 1. 检查本地ETag状态（拥塞控制）
        if piece["etag_status"][direction][channel]:
            self._record_congestion_event(piece["node_id"], "etag_block", direction, channel)
            return False  # 本地拥塞，无法注入

        # 2. 检查下游路径的拥塞状态
        if self._check_downstream_path_congestion(piece["node_id"], flit.destination, direction, channel):
            self._record_congestion_event(piece["node_id"], "downstream_congestion", direction, channel)
            return False  # 下游路径拥塞

        # 3. 确定ring方向（顺时针或逆时针）
        ring_direction = self._determine_ring_direction(piece["node_id"], flit.destination, direction)

        # 4. 检查ring缓冲区是否有空间
        ring_buffer = piece["ring_buffers"][direction][channel][ring_direction]
        if len(ring_buffer) >= self.config.ring_buffer_depth:
            self._record_congestion_event(piece["node_id"], "buffer_full", direction, channel)
            return False  # 缓冲区满

        # 5. 执行仲裁
        if not self._arbitrate_ring_access(piece, direction, "inject"):
            self._record_congestion_event(piece["node_id"], "arbitration_fail", direction, channel)
            return False  # 仲裁失败

        # 6. 成功注入到ring
        ring_buffer.append(flit)
        flit.network_entry_cycle = self.cycle
        self._record_injection_success(piece["node_id"], direction, channel)

        return True

    def _determine_ring_direction(self, source: NodeId, destination: NodeId, direction: str) -> str:
        """确定ring传输方向（顺时针或逆时针）"""
        src_x, src_y = self._get_node_coordinates(source)
        dst_x, dst_y = self._get_node_coordinates(destination)

        if direction == "horizontal":
            # 水平方向：选择最短路径
            if dst_x > src_x:
                # 目标在右侧
                right_distance = dst_x - src_x
                left_distance = src_x + (self.config.num_col - dst_x)
                return "clockwise" if right_distance <= left_distance else "counter_clockwise"
            else:
                # 目标在左侧
                left_distance = src_x - dst_x
                right_distance = dst_x + (self.config.num_col - src_x)
                return "counter_clockwise" if left_distance <= right_distance else "clockwise"
        else:  # vertical
            # 垂直方向：选择最短路径
            if dst_y > src_y:
                # 目标在下方
                down_distance = dst_y - src_y
                up_distance = src_y + (self.config.num_row - dst_y)
                return "clockwise" if down_distance <= up_distance else "counter_clockwise"
            else:
                # 目标在上方
                up_distance = src_y - dst_y
                down_distance = dst_y + (self.config.num_row - src_y)
                return "counter_clockwise" if up_distance <= down_distance else "clockwise"

    def _arbitrate_ring_access(self, piece: Dict, direction: str, requester: str) -> bool:
        """Ring访问仲裁机制（简化版：优先级固定）"""
        # 简化仲裁：ring传输优先级高于inject
        # 这样可以确保ring中的flit能够正常流动

        if requester in ["ring_cw", "ring_ccw"]:
            # Ring传输请求总是被允许（除非有冲突）
            return True
        elif requester == "inject":
            # Inject请求在没有ring传输时被允许
            # 检查是否有ring传输正在进行
            ring_buffers = piece["ring_buffers"][direction]
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
            piece = self.crossring_pieces[node_id]
            ring_buffer = piece["ring_buffers"][ring_type][channel][direction]

            if not ring_buffer:
                continue

            # 获取下一个节点
            next_node_index = (i + 1) % len(nodes_in_ring)
            next_node_id = nodes_in_ring[next_node_index]
            next_piece = self.crossring_pieces[next_node_id]

            # 尝试移动flit到下一个节点
            self._try_move_flit_in_ring(piece, next_piece, ring_type, direction, channel)

    def _try_move_flit_in_ring(self, current_piece: Dict, next_piece: Dict, ring_type: str, direction: str, channel: str) -> None:
        """尝试在ring中移动flit"""
        current_buffer = current_piece["ring_buffers"][ring_type][channel][direction]
        next_buffer = next_piece["ring_buffers"][ring_type][channel][direction]

        if not current_buffer:
            return

        flit = current_buffer[0]  # 查看队首flit

        # 检查是否到达最终目标节点
        if self._should_eject_flit(next_piece["node_id"], flit, ring_type):
            # 尝试移动到eject队列
            if self._try_move_to_eject(next_piece, flit, channel):
                current_buffer.pop(0)  # 成功eject，移除flit
            return

        # 检查是否需要转换ring方向（从水平转垂直）
        if self._should_transfer_to_vertical_ring(next_piece["node_id"], flit, ring_type):
            # 尝试转移到垂直ring
            if self._try_transfer_to_vertical_ring(next_piece, flit, channel):
                current_buffer.pop(0)  # 成功转移，移除flit
            return

        # 检查下一个节点的缓冲区是否有空间
        if len(next_buffer) >= self.config.ring_buffer_depth:
            return  # 缓冲区满，无法移动

        # 执行仲裁
        requester = "ring_cw" if direction == "clockwise" else "ring_ccw"
        if not self._arbitrate_ring_access(next_piece, ring_type, requester):
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

    def _try_move_to_eject(self, piece: Dict, flit: Any, channel: str) -> bool:
        """尝试将flit移动到eject队列"""
        eject_queue = piece["eject_queues"][channel]

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

    def _try_transfer_to_vertical_ring(self, piece: Dict, flit: Any, channel: str) -> bool:
        """尝试将flit从水平ring转移到垂直ring"""
        # 确定垂直ring的方向
        node_y = self._get_node_coordinates(piece["node_id"])[1]
        dst_y = self._get_node_coordinates(flit.destination)[1]

        if dst_y > node_y:
            ring_direction = "clockwise"  # 向下
        else:
            ring_direction = "counter_clockwise"  # 向上

        # 检查垂直ring缓冲区是否有空间
        vertical_buffer = piece["ring_buffers"]["vertical"][channel][ring_direction]
        if len(vertical_buffer) >= self.config.ring_buffer_depth:
            return False

        # 执行仲裁
        if not self._arbitrate_ring_access(piece, "vertical", "inject"):
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
        for node_id, piece in self.crossring_pieces.items():
            # 更新ETag状态（Egress Tag - 出口拥塞标记）
            self._update_etag_status(piece)

            # 更新ITag状态（Ingress Tag - 入口拥塞标记）
            self._update_itag_status(piece)

    def _update_etag_status(self, piece: Dict) -> None:
        """更新ETag状态"""
        for direction in ["horizontal", "vertical"]:
            for channel in ["req", "rsp", "data"]:
                # 检查eject队列的拥塞情况
                eject_queue = piece["eject_queues"][channel]
                eject_threshold = self.config.eject_buffer_depth * 0.8  # 80%阈值

                # 检查ring缓冲区的拥塞情况
                ring_buffers = piece["ring_buffers"][direction][channel]
                ring_congestion = False

                for ring_dir in ["clockwise", "counter_clockwise"]:
                    buffer_occupancy = len(ring_buffers[ring_dir])
                    ring_threshold = self.config.ring_buffer_depth * 0.8
                    if buffer_occupancy >= ring_threshold:
                        ring_congestion = True
                        break

                # 设置ETag状态
                eject_congestion = len(eject_queue) >= eject_threshold
                piece["etag_status"][direction][channel] = eject_congestion or ring_congestion

    def _update_itag_status(self, piece: Dict) -> None:
        """更新ITag状态"""
        for direction in ["horizontal", "vertical"]:
            for channel in ["req", "rsp", "data"]:
                # 检查inject队列的拥塞情况
                inject_queue = piece["inject_queues"][channel]
                inject_threshold = self.config.inject_buffer_depth * 0.8

                # 检查下游节点的ETag状态
                downstream_congestion = self._check_downstream_congestion(piece["node_id"], direction, channel)

                # 设置ITag状态
                inject_congestion = len(inject_queue) >= inject_threshold
                piece["itag_status"][direction][channel] = inject_congestion or downstream_congestion

    def _check_downstream_congestion(self, node_id: NodeId, direction: str, channel: str) -> bool:
        """检查下游节点的拥塞状态"""
        # 获取下游节点列表
        downstream_nodes = self._get_downstream_nodes(node_id, direction)

        # 检查下游节点的ETag状态
        for downstream_node in downstream_nodes:
            if downstream_node in self.crossring_pieces:
                downstream_piece = self.crossring_pieces[downstream_node]
                if downstream_piece["etag_status"][direction][channel]:
                    return True

        return False

    def _get_downstream_nodes(self, node_id: NodeId, direction: str) -> List[NodeId]:
        """获取指定方向的下游节点"""
        x, y = self._get_node_coordinates(node_id)
        downstream_nodes = []

        if direction == "horizontal":
            # 水平方向的下游节点
            for next_x in range(self.config.num_col):
                if next_x != x:
                    next_node = y * self.config.num_col + next_x
                    downstream_nodes.append(next_node)
        else:  # vertical
            # 垂直方向的下游节点
            for next_y in range(self.config.num_row):
                if next_y != y:
                    next_node = next_y * self.config.num_col + x
                    downstream_nodes.append(next_node)

        return downstream_nodes

    def _check_downstream_path_congestion(self, source: NodeId, destination: NodeId, direction: str, channel: str) -> bool:
        """检查到目标节点路径上的拥塞状态"""
        # 获取路径上的关键节点
        path_nodes = self._get_path_nodes(source, destination, direction)

        # 检查路径上每个节点的拥塞状态
        for node_id in path_nodes:
            if node_id in self.crossring_pieces:
                piece = self.crossring_pieces[node_id]

                # 检查该节点的ETag状态
                if piece["etag_status"][direction][channel]:
                    return True

                # 检查ring缓冲区占用率
                for ring_dir in ["clockwise", "counter_clockwise"]:
                    buffer = piece["ring_buffers"][direction][channel][ring_dir]
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
                f"  {ip_key}: RN({ip_status['rn_read_active']}R+{ip_status['rn_write_active']}W), " + f"SN({ip_status['sn_active']}), 重试({ip_status['read_retries']}R+{ip_status['write_retries']}W)"
            )

    def inject_test_traffic(self, source: NodeId, destination: NodeId, req_type: str, count: int = 1, burst_length: int = 4) -> List[str]:
        """
        注入测试流量

        Args:
            source: 源节点
            destination: 目标节点
            req_type: 请求类型
            count: 请求数量
            burst_length: 突发长度

        Returns:
            生成的packet_id列表
        """
        packet_ids = []

        # 找到源节点对应的IP接口
        source_ips = [ip for key, ip in self._ip_registry.items() if ip.node_id == source]

        if not source_ips:
            self.logger.warning(f"源节点 {source} 没有找到对应的IP接口")
            return packet_ids

        # 使用第一个找到的IP接口
        ip_interface = source_ips[0]

        for i in range(count):
            packet_id = f"test_{source}_{destination}_{req_type}_{self.cycle}_{i}"
            success = ip_interface.enqueue_request(source=source, destination=destination, req_type=req_type, burst_length=burst_length, packet_id=packet_id)

            if success:
                packet_ids.append(packet_id)
                self.logger.debug(f"注入测试请求: {packet_id}")
            else:
                self.logger.warning(f"测试请求注入失败: {packet_id}")

        return packet_ids

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
            "networks_ready": {
                "req": self.networks["req"] is not None,
                "rsp": self.networks["rsp"] is not None,
                "data": self.networks["data"] is not None,
            },
        }

    def cleanup(self) -> None:
        """清理资源"""
        self.logger.info("开始清理CrossRing模型资源")

        # 清理IP接口
        for ip in self.ip_interfaces.values():
            # 这里可以添加IP接口的清理逻辑
            pass

        # 清理网络组件
        self.crossring_pieces.clear()

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
        return f"CrossRingModel({self.config.config_name}, " f"{self.config.num_row}x{self.config.num_col}, " f"cycle={self.cycle}, " f"active_requests={self.get_active_request_count()})"


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
