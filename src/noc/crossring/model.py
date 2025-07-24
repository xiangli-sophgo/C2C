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
import numpy as np
import re

from .config import CrossRingConfig, RoutingStrategy
from .topology import CrossRingTopology
from .ip_interface import CrossRingIPInterface
from .flit import CrossRingFlit, get_crossring_flit_pool_stats
from .node import CrossRingNode
from .link import CrossRingLink
from src.noc.utils.types import NodeId
from src.noc.debug import RequestTracker, RequestState, FlitType
from src.noc.base.model import BaseNoCModel
from src.noc.analysis.result_analyzer import ResultAnalyzer
from src.noc.analysis.fifo_analyzer import FIFOStatsCollector


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

    def __init__(self, config: CrossRingConfig, traffic_file_path: str = None):
        """
        初始化CrossRing模型

        Args:
            config: CrossRing配置实例
            traffic_file_path: 可选的traffic文件路径，用于优化IP接口创建
        """
        # 调用父类初始化
        super().__init__(config, model_name="CrossRingModel", traffic_file_path=traffic_file_path)

        # CrossRing网络组件 - 使用新的架构
        self.nodes: Dict[NodeId, Any] = {}  # {node_id: CrossRingNode}
        self.links: Dict[str, Any] = {}  # {link_id: CrossRingLink}

        # Tag管理器
        self.tag_managers: Dict[NodeId, Any] = {}  # {node_id: CrossRingTagManager}

        # CrossRing特有的统计信息
        self.crossring_stats = {
            "dimension_turns": 0,
            "ring_transmissions": 0,
            "wrap_around_hops": 0,
            "crosspoint_arbitrations": 0,
            "tag_upgrades": {"itag": 0, "etag": 0},
        }

        # CrossRing特有的调试信息
        self.crossring_debug = {
            "track_ring_slots": False,
            "track_crosspoint_arbitration": False,
            "track_tag_mechanisms": False,
        }

        # 全局调试配置
        self.debug_enabled = False
        self.debug_packet_ids = set()  # 要跟踪的packet_id集合
        self.debug_sleep_time = 0.0  # 每步的睡眠时间

        # FIFO统计收集器
        self.fifo_stats_collector = FIFOStatsCollector()

        # 可视化配置
        self._viz_config = {"flow_distribution": False, "bandwidth_analysis": False, "save_figures": True, "save_dir": "output"}

        # 初始化模型（不包括IP接口创建，IP接口将在setup_traffic_scheduler中创建）
        self.initialize_model()

        # 初始化完成后注册FIFO统计
        self._register_all_fifos_for_statistics()

        # 验证CrossRing网络初始化
        if len(self.nodes) != self.config.NUM_NODE:
            self.logger.error(f"CrossRing节点初始化不完整: 期望{self.config.NUM_NODE}，实际{len(self.nodes)}")
            self.logger.error("debug: 当前nodes内容: {}".format(list(self.nodes.keys())))
            raise RuntimeError("CrossRing网络初始化失败")

        self.logger.info(f"CrossRing模型初始化完成: {config.NUM_ROW}x{config.NUM_COL}")

    def _create_topology_instance(self, config) -> CrossRingTopology:
        """
        创建CrossRing拓扑实例

        Args:
            config: CrossRing配置对象

        Returns:
            CrossRing拓扑实例
        """
        self.logger.info("创建CrossRing拓扑实例...")
        topology = CrossRingTopology(config)
        self.logger.info(f"CrossRing拓扑实例创建成功: {config.NUM_ROW}x{config.NUM_COL}网格")
        return topology

    def _print_debug_info(self):
        """打印调试信息"""
        if not self.debug_enabled or not hasattr(self, "request_tracker"):
            return

        # 检查所有要跟踪的packet_ids，使用base class的trace_packets
        trace_packets = self.trace_packets if self.trace_packets else self.debug_packet_ids

        # 用于跟踪是否有任何信息需要打印
        printed_info = False
        cycle_header_printed = False
        completed_packets = set()
        flits_to_print = []

        for packet_id in list(trace_packets):
            if self._should_debug_packet(packet_id):
                # 获取lifecycle - 支持整数和字符串形式的packet_id
                lifecycle = self.request_tracker.active_requests.get(packet_id)
                if not lifecycle:
                    lifecycle = self.request_tracker.completed_requests.get(packet_id)
                # 如果字符串形式找不到，尝试整数形式
                if not lifecycle and isinstance(packet_id, str) and packet_id.isdigit():
                    int_packet_id = int(packet_id)
                    lifecycle = self.request_tracker.active_requests.get(int_packet_id)
                    if not lifecycle:
                        lifecycle = self.request_tracker.completed_requests.get(int_packet_id)
                # 如果整数形式找不到，尝试字符串形式
                elif not lifecycle and isinstance(packet_id, int):
                    str_packet_id = str(packet_id)
                    lifecycle = self.request_tracker.active_requests.get(str_packet_id)
                    if not lifecycle:
                        lifecycle = self.request_tracker.completed_requests.get(str_packet_id)

                if lifecycle:
                    # 简化条件：只要有flit就打印，或者状态变化就打印
                    total_flits = len(lifecycle.request_flits) + len(lifecycle.response_flits) + len(lifecycle.data_flits)
                    should_print = total_flits > 0 or lifecycle.current_state != RequestState.CREATED or self.request_tracker.should_print_debug(packet_id)

                    if should_print:
                        # 收集本周期要打印的flit
                        all_flits = lifecycle.request_flits + lifecycle.response_flits + lifecycle.data_flits
                        flits_to_print.extend(all_flits)
                        printed_info = True

                    # 如果完成，标记为已完成
                    if lifecycle.current_state.value == "completed":
                        if not cycle_header_printed:
                            print(f"周期{self.cycle}: ")
                            cycle_header_printed = True
                        print(f"✅ 请求{packet_id}已完成，停止跟踪")
                        completed_packets.add(packet_id)
                        printed_info = True

        # 如果有flit要打印，统一打印在一行
        if flits_to_print:
            if not cycle_header_printed:
                print(f"周期{self.cycle}: ")
                cycle_header_printed = True
            print(f" ", end="")
            for flit in flits_to_print:
                print(f"{flit}", end=" | ")
            print("")

        # 从跟踪列表中移除已完成的请求
        for packet_id in completed_packets:
            self.debug_packet_ids.discard(packet_id)
            self.trace_packets.discard(packet_id)

        # 检查是否所有跟踪的请求都已完成
        remaining_packets = len(self.trace_packets) + len(self.debug_packet_ids)
        if remaining_packets == 0 and self.debug_enabled:
            print(f"🎯 所有跟踪请求已完成，自动关闭debug模式")
            self.disable_debug()
            return

        # 只有在实际打印了信息时才执行sleep
        if printed_info and self.debug_config["sleep_time"] > 0:
            import time

            time.sleep(self.debug_config["sleep_time"])

    def _create_ip_interface(self, node_id: int, ip_type: str, key: str = None) -> bool:
        """
        通用IP接口创建方法

        Args:
            node_id: 节点ID
            ip_type: IP类型
            key: IP接口键名

        Returns:
            创建成功返回True，失败返回False
        """
        if not ip_type or not isinstance(ip_type, str):
            self.logger.warning(f"无效的IP类型: {ip_type} for node {node_id}")
            return False

        if key is None:
            key = ip_type

        try:
            ip_interface = CrossRingIPInterface(config=self.config, ip_type=ip_type, node_id=node_id, model=self)
            self.ip_interfaces[key] = ip_interface
            self._ip_registry[key] = ip_interface

            # 连接IP到对应的节点
            if node_id in self.nodes:
                self.nodes[node_id].connect_ip(key)
                self.logger.debug(f"连接IP接口 {key} 到节点 {node_id}")
            else:
                self.logger.warning(f"节点 {node_id} 不存在，无法连接IP接口 {key}")
                return False

            self.logger.debug(f"创建IP接口: {key} at node {node_id} (ip_type={ip_type})")
            return True

        except Exception as e:
            self.logger.error(f"创建IP接口失败: {key} - {e}")
            return False

    def _setup_all_ip_interfaces(self) -> None:
        """创建所有IP接口（传统模式）"""
        ip_type_configs = [
            ("gdma", self.config.gdma_send_position_list),
            ("sdma", self.config.sdma_send_position_list),
            ("cdma", self.config.cdma_send_position_list),
            ("ddr", self.config.ddr_send_position_list),
            ("l2m", self.config.l2m_send_position_list),
        ]

        for ip_type, positions in ip_type_configs:
            for node_id in positions:
                # 为每个节点创建多个IP通道
                channel_count = self.config.CHANNEL_SPEC.get(ip_type, 2)
                for channel_id in range(channel_count):
                    key = f"{ip_type}_{channel_id}_node{node_id}"
                    ip_interface = CrossRingIPInterface(config=self.config, ip_type=f"{ip_type}_{channel_id}", node_id=node_id, model=self)
                    self.ip_interfaces[key] = ip_interface
                    self._ip_registry[key] = ip_interface

                    # 连接IP到对应的节点，使用简单的ip_type格式
                    simple_ip_key = f"{ip_type}_{channel_id}"
                    if node_id in self.nodes:
                        self.nodes[node_id].connect_ip(simple_ip_key)
                        self.logger.debug(f"连接IP接口 {key} (作为 {simple_ip_key}) 到节点 {node_id}")
                    else:
                        self.logger.warning(f"节点 {node_id} 不存在，无法连接IP接口 {key}")

                    self.logger.debug(f"创建IP接口: {key} at node {node_id}")

    def _create_specific_ip_interfaces(self, required_ips: List[Tuple[int, str]]) -> None:
        """创建特定的IP接口"""
        for node_id, ip_type in required_ips:
            # 验证ip_type格式
            if not ip_type or not isinstance(ip_type, str):
                self.logger.warning(f"无效的IP类型: {ip_type} for node {node_id}")
                continue

            key = ip_type

            # 检查是否已经存在
            if key in self.ip_interfaces:
                self.logger.debug(f"IP接口 {key} 已存在，跳过创建")
                continue

            try:
                ip_interface = CrossRingIPInterface(config=self.config, ip_type=ip_type, node_id=node_id, model=self)
                self.ip_interfaces[key] = ip_interface
                self._ip_registry[key] = ip_interface

                # 连接IP到对应的节点
                if node_id in self.nodes:
                    self.nodes[node_id].connect_ip(key)
                    self.logger.info(f"连接优化IP接口 {key} 到节点 {node_id}")
                else:
                    self.logger.warning(f"节点 {node_id} 不存在，无法连接IP接口 {key}")

                self.logger.info(f"创建优化IP接口: {key} at node {node_id} (ip_type={ip_type})")
            except Exception as e:
                self.logger.error(f"创建IP接口失败: {key} - {e}")
                continue

        # 打印所有创建的IP接口
        self.logger.info(f"总共创建了 {len(self.ip_interfaces)} 个IP接口")
        for key, ip_interface in self.ip_interfaces.items():
            self.logger.info(f"  {key}: node_id={ip_interface.node_id}, ip_type={ip_interface.ip_type}")

    def setup_traffic_scheduler(self, traffic_chains: List[List[str]], traffic_file_path: str = None) -> None:
        """
        设置TrafficScheduler并根据traffic文件动态创建需要的IP接口

        Args:
            traffic_chains: traffic链配置，每个链包含文件名列表
            traffic_file_path: traffic文件路径，默认使用初始化时的路径
        """
        # 先分析traffic文件，获取需要的IP接口
        file_path = traffic_file_path or self.traffic_file_path or "traffic_data"

        try:
            # 分析所有traffic文件中需要的IP接口
            all_required_ips = []
            from src.noc.utils.traffic_scheduler import TrafficFileReader

            for chain in traffic_chains:
                for filename in chain:
                    self.logger.info(f"分析traffic文件: {filename}")
                    traffic_reader = TrafficFileReader(filename=filename, traffic_file_path=file_path, config=self.config, time_offset=0, traffic_id="analysis")

                    ip_info = traffic_reader.get_required_ip_interfaces()
                    required_ips = ip_info["required_ips"]
                    all_required_ips.extend(required_ips)

                    self.logger.info(f"文件 {filename} 需要IP接口: {required_ips}")

            # 去重
            unique_required_ips = list(set(all_required_ips))
            self.logger.info(f"总共需要创建 {len(unique_required_ips)} 个唯一IP接口: {unique_required_ips}")

            # 动态创建需要的IP接口
            self._create_specific_ip_interfaces(unique_required_ips)

        except Exception as e:
            self.logger.warning(f"动态创建IP接口失败: {e}，使用现有IP接口")
            import traceback

            traceback.print_exc()

        # 调用父类方法设置TrafficScheduler
        super().setup_traffic_scheduler(traffic_chains, traffic_file_path)

    def _setup_topology_network(self) -> None:
        """设置拓扑网络（BaseNoCModel抽象方法的实现）"""
        self._setup_crossring_networks()

    def _setup_flit_pools(self) -> None:
        """设置Flit对象池（重写父类方法）"""
        from .flit import CrossRingFlit
        from src.noc.base.flit import FlitPool

        self.flit_pools[CrossRingFlit] = FlitPool(CrossRingFlit)

    def _setup_crossring_networks(self) -> None:
        """设置CrossRing网络组件的完整实现 - 真实环形拓扑"""
        # 用CrossRingNode实例替换原有dict结构
        self.nodes: Dict[NodeId, CrossRingNode] = {}

        # 导入CrossRingNode类
        from .node import CrossRingNode

        for node_id in range(self.config.NUM_NODE):
            coordinates = self._get_node_coordinates(node_id)

            try:
                node = CrossRingNode(node_id=node_id, coordinates=coordinates, config=self.config, logger=self.logger, topology=self.topology)
                self.nodes[node_id] = node
            except Exception as e:
                import traceback

                traceback.print_exc()

        # 创建链接
        self._setup_links()

        # 连接slice到CrossPoint
        self._connect_slices_to_crosspoints()

        # 连接相部链路的slice形成传输链
        self._connect_ring_slices()

    def _setup_links(self) -> None:
        """创建CrossRing链接"""

        # 导入必要的类
        from .link import CrossRingLink
        from .link import Direction

        # 获取slice配置
        normal_slices = getattr(self.config.basic_config, "NORMAL_LINK_SLICES", 8)
        self_slices = getattr(self.config.basic_config, "SELF_LINK_SLICES", 2)

        link_count = 0
        for node_id in range(self.config.NUM_NODE):
            # 获取节点的四个方向连接
            connections = self._get_ring_connections(node_id)

            for direction_str, neighbor_id in connections.items():
                # 确定链接方向
                direction = Direction[direction_str.upper()]

                # 确定slice数量
                if neighbor_id == node_id:
                    # 自连接
                    num_slices = self_slices
                    link_type = "self"
                else:
                    # 正常连接
                    num_slices = normal_slices
                    link_type = "normal"

                # 创建链接ID
                if neighbor_id == node_id:
                    # 自环链路：表示它同时服务于两个相反方向
                    reverse_direction = self.REVERSE_DIRECTION_MAP.get(direction_str, direction_str)
                    link_id = f"link_{node_id}_{direction_str}_{reverse_direction}_{neighbor_id}"
                else:
                    # 普通链路
                    link_id = f"link_{node_id}_{direction_str}_{neighbor_id}"

                # 创建链接
                try:
                    link = CrossRingLink(link_id=link_id, source_node=node_id, dest_node=neighbor_id, direction=direction, config=self.config, num_slices=num_slices, logger=self.logger)
                    self.links[link_id] = link
                    link_count += 1
                except Exception as e:
                    print(f"DEBUG: 创建链接失败 {link_id}: {e}")
                    import traceback

                    traceback.print_exc()

    def _connect_slices_to_crosspoints(self) -> None:
        """连接RingSlice到CrossPoint"""
        # 连接CrossPoint slices（简化输出）
        connected_count = 0
        for node_id, node in self.nodes.items():
            # 处理每个方向
            for direction_str in ["TR", "TL", "TU", "TD"]:
                # 确定CrossPoint方向
                crosspoint_direction = "horizontal" if direction_str in ["TR", "TL"] else "vertical"
                crosspoint = node.get_crosspoint(crosspoint_direction)

                if not crosspoint:
                    continue

                # 获取该方向的出链路（departure）
                out_link = None
                # 获取该方向的邻居节点
                connections = self._get_ring_connections(node_id)
                neighbor_id = connections.get(direction_str)

                if neighbor_id is not None:
                    if neighbor_id == node_id:
                        # 自环链路
                        reverse_direction = self.REVERSE_DIRECTION_MAP.get(direction_str, direction_str)
                        out_link_id = f"link_{node_id}_{direction_str}_{reverse_direction}_{neighbor_id}"
                    else:
                        # 普通链路
                        out_link_id = f"link_{node_id}_{direction_str}_{neighbor_id}"

                    out_link = self.links.get(out_link_id)
                    if out_link:
                        connected_count += 1

                # 连接slice
                for channel in ["req", "rsp", "data"]:  # 处理所有三个通道
                    # 连接departure slice（出链路的第一个slice）
                    if out_link and out_link.ring_slices[channel]:
                        departure_slice = out_link.ring_slices[channel][0]
                        crosspoint.connect_slice(direction_str, "departure", departure_slice)

                    # 连接arrival slice - 需要根据CrossPoint连接规则
                    arrival_slice = None

                    if direction_str == "TR":
                        # TR arrival slice来自其他节点的TR链路，如果没有则来自本节点TL自环
                        found = False
                        for link_id, link in self.links.items():
                            if link.dest_node == node_id and "TR" in link_id and link.source_node != node_id:
                                if link.ring_slices[channel]:
                                    arrival_slice = link.ring_slices[channel][-1]  # 其他节点TR链路的最后slice
                                    found = True
                                break

                        # 如果没有找到其他节点的TR链路，使用本节点TL_TR自环
                        if not found:
                            self_tl_link_id = f"link_{node_id}_TL_TR_{node_id}"
                            self_tl_link = self.links.get(self_tl_link_id)
                            if self_tl_link and self_tl_link.ring_slices[channel] and len(self_tl_link.ring_slices[channel]) > 1:
                                arrival_slice = self_tl_link.ring_slices[channel][1]  # 自环的第1个slice

                    elif direction_str == "TL":
                        # TL arrival slice来自其他节点的TL链路，如果没有则来自本节点TR自环
                        found = False
                        for link_id, link in self.links.items():
                            if link.dest_node == node_id and "TL" in link_id and link.source_node != node_id:
                                if link.ring_slices[channel]:
                                    arrival_slice = link.ring_slices[channel][-1]  # 其他节点TL链路的最后slice
                                    found = True
                                break

                        # 如果没有找到其他节点的TL链路，使用本节点TR_TL自环
                        if not found:
                            self_tr_link_id = f"link_{node_id}_TR_TL_{node_id}"
                            self_tr_link = self.links.get(self_tr_link_id)
                            if self_tr_link and self_tr_link.ring_slices[channel] and len(self_tr_link.ring_slices[channel]) > 1:
                                arrival_slice = self_tr_link.ring_slices[channel][1]  # 自环的第1个slice

                    elif direction_str == "TU":
                        # TU arrival slice来自其他节点的TU链路，如果没有则来自本节点TD自环
                        found = False
                        for link_id, link in self.links.items():
                            if link.dest_node == node_id and "TU" in link_id and link.source_node != node_id:
                                if link.ring_slices[channel]:
                                    arrival_slice = link.ring_slices[channel][-1]  # 其他节点TU链路的最后slice
                                    found = True
                                break

                        # 如果没有找到其他节点的TU链路，使用本节点TD_TU自环
                        if not found:
                            self_td_link_id = f"link_{node_id}_TD_TU_{node_id}"
                            self_td_link = self.links.get(self_td_link_id)
                            if self_td_link and self_td_link.ring_slices[channel] and len(self_td_link.ring_slices[channel]) > 1:
                                arrival_slice = self_td_link.ring_slices[channel][1]  # 自环的第1个slice

                    elif direction_str == "TD":
                        # TD arrival slice来自其他节点的TD链路，如果没有则来自本节点TU自环
                        found = False
                        for link_id, link in self.links.items():
                            if link.dest_node == node_id and "TD" in link_id and link.source_node != node_id:
                                if link.ring_slices[channel]:
                                    arrival_slice = link.ring_slices[channel][-1]  # 其他节点TD链路的最后slice
                                    found = True
                                break

                        # 如果没有找到其他节点的TD链路，使用本节点TU_TD自环
                        if not found:
                            self_tu_link_id = f"link_{node_id}_TU_TD_{node_id}"
                            self_tu_link = self.links.get(self_tu_link_id)
                            if self_tu_link and self_tu_link.ring_slices[channel] and len(self_tu_link.ring_slices[channel]) > 1:
                                arrival_slice = self_tu_link.ring_slices[channel][1]  # 自环的第1个slice

                    if arrival_slice:
                        crosspoint.connect_slice(direction_str, "arrival", arrival_slice)

    def _get_node_links(self, node_id: int) -> Dict[str, Any]:
        """获取节点的所有链接"""
        node_links = {}

        for link_id, link in self.links.items():
            if link.source_node == node_id:
                # 从链接ID中提取方向
                parts = link_id.split("_")
                if len(parts) >= 3:
                    direction_str = parts[2]
                    node_links[direction_str] = link

        return node_links

    def _connect_ring_slices(self) -> None:
        """连接链路的RingSlice形成传输链"""
        # 开始连接RingSlice形成传输链

        connected_count = 0
        for link_id, link in self.links.items():
            for channel in ["req", "rsp", "data"]:
                ring_slices = link.ring_slices[channel]

                # 连接链路内部的slice形成传输链
                for i in range(len(ring_slices) - 1):
                    current_slice = ring_slices[i]
                    next_slice = ring_slices[i + 1]

                    # 设置上下游连接
                    current_slice.downstream_slice = next_slice
                    next_slice.upstream_slice = current_slice

                    connected_count += 1

        # RingSlice连接完成

        # 连接不同链路之间的slice（形成环路）
        self._connect_inter_link_slices()

        # 调试：打印所有连接信息
        # self._print_all_connections()

    def _connect_inter_link_slices(self) -> None:
        """连接不同链路之间的slice形成环路"""
        # 按照CrossRing规范，形成正确的单向环路连接

        for node_id in range(self.config.NUM_NODE):
            connections = self._get_ring_connections(node_id)

            for direction_str, neighbor_id in connections.items():
                # 获取当前节点的出链路
                if neighbor_id == node_id:
                    # 自环链路
                    reverse_direction = self.REVERSE_DIRECTION_MAP.get(direction_str, direction_str)
                    out_link_id = f"link_{node_id}_{direction_str}_{reverse_direction}_{neighbor_id}"
                else:
                    # 普通链路
                    out_link_id = f"link_{node_id}_{direction_str}_{neighbor_id}"
                out_link = self.links.get(out_link_id)

                if not out_link:
                    continue

                # 获取下一个链路
                next_link = None
                next_link_id = None

                if neighbor_id == node_id:
                    # 自环情况：连接到反方向的链路
                    reverse_direction = self.REVERSE_DIRECTION_MAP.get(direction_str, direction_str)
                    next_neighbor_connections = self._get_ring_connections(node_id)
                    next_neighbor_id = next_neighbor_connections.get(reverse_direction)
                    if next_neighbor_id is not None:
                        if next_neighbor_id == node_id:
                            # 下一个也是自环
                            next_reverse = self.REVERSE_DIRECTION_MAP.get(reverse_direction, reverse_direction)
                            next_link_id = f"link_{node_id}_{reverse_direction}_{next_reverse}_{next_neighbor_id}"
                        else:
                            # 下一个是普通链路
                            next_link_id = f"link_{node_id}_{reverse_direction}_{next_neighbor_id}"
                        next_link = self.links.get(next_link_id)
                else:
                    # 非自环情况：继续同方向
                    next_neighbor_connections = self._get_ring_connections(neighbor_id)
                    next_neighbor_id = next_neighbor_connections.get(direction_str)
                    if next_neighbor_id is not None:
                        if next_neighbor_id == neighbor_id:
                            # 下一个是自环
                            reverse_direction = self.REVERSE_DIRECTION_MAP.get(direction_str, direction_str)
                            next_link_id = f"link_{neighbor_id}_{direction_str}_{reverse_direction}_{next_neighbor_id}"
                        else:
                            # 下一个是普通链路
                            next_link_id = f"link_{neighbor_id}_{direction_str}_{next_neighbor_id}"
                        next_link = self.links.get(next_link_id)

                if not next_link:
                    continue

                # 连接两个同方向链路的slice
                for channel in ["req", "rsp", "data"]:
                    out_slices = out_link.ring_slices[channel]
                    next_slices = next_link.ring_slices[channel]

                    if out_slices and next_slices:
                        # 当前链路的最后slice连接到下一个链路的第一个slice
                        last_out_slice = out_slices[-1]
                        first_next_slice = next_slices[0]

                        last_out_slice.downstream_slice = first_next_slice
                        first_next_slice.upstream_slice = last_out_slice

        # 链路间slice连接完成

    def _print_all_connections(self) -> None:
        """打印所有链路连接和CrossPoint连接信息"""
        print("\n" + "=" * 80)
        print("🔗 CrossRing 连接信息调试")
        print("=" * 80)

        # 1. 打印所有链路信息
        print("\n📋 链路列表:")
        for link_id, link in sorted(self.links.items()):
            slice_count = len(link.ring_slices.get("req", []))
            print(f"  {link_id}: {link.source_node}->{link.dest_node}, {slice_count} slices")

        # 2. 打印链路间slice连接
        print("\n🔗 链路间slice连接:")
        for link_id, link in sorted(self.links.items()):
            for channel in ["req"]:  # 只显示req通道
                slices = link.ring_slices.get(channel, [])
                if slices:
                    last_slice = slices[-1]
                    if hasattr(last_slice, "downstream_slice") and last_slice.downstream_slice:
                        downstream_info = f"slice_0"  # 简化显示
                        # 找到downstream slice属于哪个链路
                        for dst_link_id, dst_link in self.links.items():
                            dst_slices = dst_link.ring_slices.get(channel, [])
                            if dst_slices and dst_slices[0] == last_slice.downstream_slice:
                                downstream_info = f"{dst_link_id}:0"
                                break
                        print(f"  {link_id}:{len(slices)-1} -> {downstream_info}")

        # 3. 打印CrossPoint slice连接
        # print("\n🎯 CrossPoint slice连接:")
        for node_id, node in sorted(self.nodes.items()):
            # print(f"\n  节点{node_id} (坐标{node.coordinates}):")

            # 水平CrossPoint
            h_cp = node.get_crosspoint("horizontal")
            if h_cp:
                # print(f"    水平CrossPoint:")
                for direction in ["TR", "TL"]:
                    for slice_type in ["arrival", "departure"]:
                        slice_obj = h_cp.slices.get(direction, {}).get(slice_type)
                        if slice_obj:
                            # 找到这个slice属于哪个链路
                            slice_info = "unknown"
                            for link_id, link in self.links.items():
                                for ch in ["req"]:
                                    slices = link.ring_slices.get(ch, [])
                                    for i, s in enumerate(slices):
                                        if s == slice_obj:
                                            slice_info = f"{link_id}:{i}"
                                            break
                            # print(f"      {direction} {slice_type}: {slice_info}")
                        # else:
                        # print(f"      {direction} {slice_type}: None")

            # 垂直CrossPoint
            v_cp = node.get_crosspoint("vertical")
            if v_cp:
                # print(f"    垂直CrossPoint:")
                for direction in ["TU", "TD"]:
                    for slice_type in ["arrival", "departure"]:
                        slice_obj = v_cp.slices.get(direction, {}).get(slice_type)
                        if slice_obj:
                            # 找到这个slice属于哪个链路
                            slice_info = "unknown"
                            for link_id, link in self.links.items():
                                for ch in ["req"]:
                                    slices = link.ring_slices.get(ch, [])
                                    for i, s in enumerate(slices):
                                        if s == slice_obj:
                                            slice_info = f"{link_id}:{i}"
                                            break
                            # print(f"      {direction} {slice_type}: {slice_info}")
                        # else:
                        # print(f"      {direction} {slice_type}: None")

        print("\n" + "=" * 80)

    # 方向反转映射常量
    REVERSE_DIRECTION_MAP = {"TR": "TL", "TL": "TR", "TU": "TD", "TD": "TU"}

    def _get_node_coordinates(self, node_id: NodeId) -> Tuple[int, int]:
        """获取节点坐标（使用topology实例）"""
        return self.topology.get_node_position(node_id)

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
            if x == self.config.NUM_COL - 1:
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
            if y == self.config.NUM_ROW - 1:
                next_y = y  # 连接到自己
            else:
                next_y = y + 1
            next_x = x
        else:
            raise ValueError(f"不支持的方向: {direction}")

        return next_y * self.config.NUM_COL + next_x

    def _get_ring_connections(self, node_id: NodeId) -> Dict[str, NodeId]:
        """获取节点的环形连接信息"""
        # 计算节点的行列位置
        row = node_id // self.config.NUM_COL
        col = node_id % self.config.NUM_COL

        connections = {}

        # 水平环连接（TR/TL）
        # TR: 向右连接
        if col < self.config.NUM_COL - 1:
            connections["TR"] = row * self.config.NUM_COL + (col + 1)
        else:
            # 边界节点：连接到自己（非环绕设计）
            connections["TR"] = node_id

        # TL: 向左连接
        if col > 0:
            connections["TL"] = row * self.config.NUM_COL + (col - 1)
        else:
            # 边界节点：连接到自己
            connections["TL"] = node_id

        # 垂直环连接（TU/TD）
        # TU: 向上连接
        if row > 0:
            connections["TU"] = (row - 1) * self.config.NUM_COL + col
        else:
            # 边界节点：连接到自己
            connections["TU"] = node_id

        # TD: 向下连接
        if row < self.config.NUM_ROW - 1:
            connections["TD"] = (row + 1) * self.config.NUM_COL + col
        else:
            # 边界节点：连接到自己
            connections["TD"] = node_id

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

    def _sync_global_clock(self) -> None:
        """重写时钟同步阶段：添加CrossRing节点时钟同步"""
        # 调用基类的时钟同步
        super()._sync_global_clock()

        # 额外同步CrossRing节点的时钟
        for node in self.nodes.values():
            if hasattr(node, "current_cycle"):
                node.current_cycle = self.cycle

    def step(self) -> None:
        self.cycle += 1

        # 阶段0.1：TrafficScheduler处理请求注入（如果有配置）
        if hasattr(self, "traffic_scheduler") and self.traffic_scheduler:
            ready_requests = self.traffic_scheduler.get_ready_requests(self.cycle)
            if ready_requests:
                req = ready_requests[0]
                cycle, src, src_type, dst, dst_type, op, burst, traffic_id = req

                # 检查源节点的IP接口
                source_ip = self._find_ip_interface_for_request(src, "read" if op.upper() == "R" else "write", src_type)

                injected = self._inject_traffic_requests(ready_requests)

        # 阶段1：组合逻辑阶段 - 所有组件计算传输决策（现在能看到最新的valid/ready状态）
        self._step_compute_phase()

        # 阶段2：时序逻辑阶段 - 所有组件执行传输和状态更新
        self._step_update_phase()

        # 更新全局统计
        self._update_global_statistics()

        # 调试功能
        if self.debug_enabled:
            self._print_debug_info()
            self.debug_func()

        # 定期输出调试信息
        if self.cycle % self.debug_config["log_interval"] == 0:
            self._log_periodic_status()

        # Debug休眠已移至_print_debug_info中，只有在打印信息时才执行

    def _step_topology_network_compute(self) -> None:
        """CrossRing网络组件计算阶段"""
        # 1. 所有节点的计算阶段
        for node_id, node in self.nodes.items():
            if hasattr(node, "step_compute_phase"):
                node.step_compute_phase(self.cycle)

        # 2. 所有链路的计算阶段
        for link_id, link in self.links.items():
            if hasattr(link, "step_compute_phase"):
                link.step_compute_phase(self.cycle)

    def _step_topology_network_update(self) -> None:
        """CrossRing网络组件更新阶段"""
        # 正确执行顺序：先让Link移动腾空slot，再让CrossPoint注入
        # 1. 先执行链路更新阶段（环路移动，腾空slot[0]位置）
        for link_id, link in self.links.items():
            if hasattr(link, "step_update_phase"):
                link.step_update_phase(self.cycle)

        # 2. 然后执行节点更新阶段（CrossPoint注入到腾空的slot）
        for node_id, node in self.nodes.items():
            if hasattr(node, "step_update_phase"):
                node.step_update_phase(self.cycle)

    def get_congestion_statistics(self) -> Dict[str, Any]:
        """获取拥塞统计信息"""
        return {
            "congestion_events": getattr(self, "congestion_stats", {}),
            "injection_success": getattr(self, "injection_stats", {}),
            "total_congestion_events": sum(sum(events.values()) for events in getattr(self, "congestion_stats", {}).values()),
            "total_injections": sum(getattr(self, "injection_stats", {}).values()),
        }

    def _update_crossring_statistics(self) -> None:
        """更新CrossRing特有的统计信息"""
        # 更新CrossRing特有的统计
        for node in self.nodes.values():
            if hasattr(node, "crossring_stats"):
                node_stats = node.crossring_stats
                self.crossring_stats["dimension_turns"] += node_stats.get("dimension_turns", 0)
                self.crossring_stats["ring_transmissions"] += node_stats.get("ring_transmissions", 0)
                self.crossring_stats["crosspoint_arbitrations"] += node_stats.get("crosspoint_arbitrations", 0)

        # 更新tag统计
        for tag_manager in self.tag_managers.values():
            if hasattr(tag_manager, "stats"):
                tag_stats = tag_manager.stats
                self.crossring_stats["tag_upgrades"]["itag"] += sum(tag_stats.get("itag_triggers", {}).values())
                self.crossring_stats["tag_upgrades"]["etag"] += sum(sum(upgrades.values()) for upgrades in tag_stats.get("etag_upgrades", {}).values())

    def _get_config_summary(self) -> Dict[str, Any]:
        """获取CrossRing配置摘要"""
        return {
            "model_type": self.__class__.__name__,
            "topology_type": "CrossRing",
            "num_row": self.config.NUM_ROW,
            "num_col": self.config.NUM_COL,
            "num_nodes": self.config.NUM_NODE,
            "ring_buffer_depth": getattr(self.config, "RING_BUFFER_DEPTH", 4),
            "routing_strategy": self.config.routing_strategy.value if hasattr(self.config.routing_strategy, "value") else str(self.config.routing_strategy),
            "ip_interface_count": len(self.ip_interfaces),
            "crossring_stats": self.crossring_stats.copy(),
        }

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

    def setup_debug(self, level: int = 1, trace_packets: List[str] = None, sleep_time: float = 0.0) -> None:
        """启用调试模式（CrossRing扩展版本）"""
        # 调用base类的enable_debug
        super().setup_debug(level, trace_packets, sleep_time)

    def setup_result_analysis(self, flow_distribution: bool = False, bandwidth_analysis: bool = False, save_figures: bool = True, save_dir: str = "output") -> None:
        """
        配置结果分析

        Args:
            flow_distribution: 是否生成流量分布图
            bandwidth_analysis: 是否生成带宽分析图
            save_figures: 是否保存图片文件到磁盘
            save_dir: 保存目录
        """
        save_dir = f"{save_dir}{self.traffic_scheduler.get_save_filename()}"
        self._viz_config.update({"flow_distribution": flow_distribution, "bandwidth_analysis": bandwidth_analysis, "save_figures": save_figures, "save_dir": save_dir})
        self.logger.info(f"可视化配置已更新: 流量分布={flow_distribution}, 带宽分析={bandwidth_analysis}, 保存图片={save_figures}, 保存目录={save_dir}")

    def print_debug_status(self) -> None:
        """打印调试状态"""
        # 调用base类的调试状态打印
        super().print_debug_status()

        # 打印CrossRing特有的调试信息
        print(f"\nCrossRing特有统计:")
        print(f"  维度转换: {self.crossring_stats['dimension_turns']}")
        print(f"  环形传输: {self.crossring_stats['ring_transmissions']}")
        print(f"  交叉点仲裁: {self.crossring_stats['crosspoint_arbitrations']}")
        print(f"  Tag升级: I-Tag={self.crossring_stats['tag_upgrades']['itag']}, E-Tag={self.crossring_stats['tag_upgrades']['etag']}")

        if hasattr(self, "get_global_tracker_status"):
            status = self.get_global_tracker_status()
            print("\nIP接口状态:")
            for ip_key, ip_status in status.items():
                print(
                    f"  {ip_key}: RN({ip_status['rn_read_active']}R+{ip_status['rn_write_active']}W), "
                    + f"SN({ip_status['sn_active']}), 重试({ip_status['read_retries']}R+{ip_status['write_retries']}W)"
                )

    def _find_ip_interface(self, node_id: NodeId, req_type: str = None, ip_type: str = None) -> Optional[CrossRingIPInterface]:
        """
        CrossRing特定的IP接口查找方法 (重写base版本)

        Args:
            node_id: 节点ID
            req_type: 请求类型 (可选，此处未使用)
            ip_type: IP类型 (可选)

        Returns:
            找到的IP接口，未找到返回None
        """
        if ip_type:
            # 精确匹配指定IP类型
            target_key = ip_type
            if target_key in self._ip_registry:
                return self._ip_registry[target_key]

            # 精确匹配失败，报错而不是模糊匹配
            self.logger.error(f"未找到指定IP接口: {target_key}")
            return None
        else:
            # 获取该节点所有IP接口
            matching_ips = [ip for key, ip in self._ip_registry.items() if ip.node_id == node_id]
            if not matching_ips:
                self.logger.error(f"节点{node_id}没有任何IP接口")

        return matching_ips[0] if matching_ips else None

    def _find_ip_interface_for_request(self, node_id: NodeId, req_type: str, ip_type: str = None) -> Optional[CrossRingIPInterface]:
        """为请求查找合适的IP接口（优先DMA类）"""
        if ip_type:
            return self._find_ip_interface(node_id, req_type, ip_type)

        # 无指定IP类型时，优先选择DMA类IP (RN端)
        all_ips = [ip for key, ip in self._ip_registry.items() if ip.node_id == node_id]
        if not all_ips:
            self.logger.error(f"节点{node_id}没有任何IP接口可用于请求")
            return None

        preferred_ips = [ip for ip in all_ips if ip.ip_type.startswith(("gdma", "sdma", "cdma"))]
        if preferred_ips:
            return preferred_ips[0]

        # 没有DMA类IP时报警告但仍可使用其他IP
        self.logger.warning(f"节点{node_id}没有适合的DMA类IP用于{req_type}请求，使用{all_ips[0].ip_type}")
        return all_ips[0]

    def _find_ip_interface_for_response(self, node_id: NodeId, req_type: str, ip_type: str = None) -> Optional[CrossRingIPInterface]:
        """为响应查找合适的IP接口（优先存储类）"""
        if ip_type:
            return self._find_ip_interface(node_id, req_type, ip_type)

        # 无指定IP类型时，优先选择存储类IP (SN端)
        all_ips = [ip for key, ip in self._ip_registry.items() if ip.node_id == node_id]
        if not all_ips:
            self.logger.error(f"节点{node_id}没有任何IP接口可用于响应")
            return None

        preferred_ips = [ip for ip in all_ips if ip.ip_type in ["ddr", "l2m"]]
        if preferred_ips:
            return preferred_ips[0]

        # 没有存储类IP时报警告但仍可使用其他IP
        self.logger.warning(f"节点{node_id}没有适合的存储类IP用于{req_type}响应，使用{all_ips[0].ip_type}")
        return all_ips[0]

    def run_file_simulation(
        self, traffic_file_path: str, max_cycles: int = 10000, warmup_cycles: int = 1000, stats_start_cycle: int = 1000, cycle_accurate: bool = False, max_requests: int = None
    ) -> Dict[str, Any]:
        """
        运行基于文件的仿真

        Args:
            traffic_file_path: 流量文件路径
            max_cycles: 最大仿真周期
            warmup_cycles: 热身周期
            stats_start_cycle: 统计开始周期
            cycle_accurate: 是否按照cycle精确注入
            max_requests: 最大请求数限制

        Returns:
            包含仿真结果和分析的字典
        """
        self.logger.info(f"开始基于文件的仿真: {traffic_file_path}")

        # 设置TrafficScheduler
        import os

        traffic_filename = os.path.basename(traffic_file_path)
        traffic_dir = os.path.dirname(traffic_file_path)

        try:
            self.setup_traffic_scheduler([[traffic_filename]], traffic_dir)
            traffic_status = self.get_traffic_status()

            if not traffic_status.get("has_pending", False):
                self.logger.warning("没有成功加载任何请求")
                return {"success": False, "message": "No requests loaded from file"}

            loaded_count = traffic_status.get("active_traffics", 0)
            self.logger.info(f"TrafficScheduler已设置，准备处理traffic文件: {traffic_filename}")

        except Exception as e:
            self.logger.error(f"设置TrafficScheduler失败: {e}")
            return {"success": False, "message": f"Failed to setup TrafficScheduler: {e}"}

        # 运行仿真（TrafficScheduler会自动在合适的周期注入请求）
        results = self.run_simulation(max_cycles=max_cycles, warmup_cycles=warmup_cycles, stats_start_cycle=stats_start_cycle)

        # 分析结果
        analysis = self.analyze_simulation_results(results)
        report = self.generate_simulation_report(results, analysis)

        # 获取最终的traffic统计
        final_traffic_status = self.get_traffic_status()

        return {
            "success": True,
            "traffic_file": traffic_file_path,
            "loaded_requests": loaded_count,
            "simulation_results": results,
            "analysis": analysis,
            "report": report,
            "traffic_status": final_traffic_status,
            "cycle_accurate": cycle_accurate,
        }

    def analyze_simulation_results(self, results: Dict[str, Any], enable_visualization: bool = True, save_results: bool = True, save_dir: str = "output", verbose: bool = True) -> Dict[str, Any]:
        """
        分析仿真结果 - 调用CrossRing专用分析器

        Args:
            results: 仿真结果
            enable_visualization: 是否生成可视化图表
            save_results: 是否保存结果文件
            save_dir: 保存目录
            verbose: 是否打印详细结果

        Returns:
            详细的分析结果
        """
        # 如果使用了可视化配置，则覆盖默认参数
        viz_enabled = False
        save_figures = True

        if hasattr(self, "_viz_config"):
            viz_enabled = self._viz_config["flow_distribution"] or self._viz_config["bandwidth_analysis"]
            if viz_enabled:
                save_figures = self._viz_config["save_figures"]
                save_dir = self._viz_config["save_dir"]
                # 启用可视化，ResultAnalyzer会根据save_figures参数决定保存或显示
                enable_visualization = True

        analyzer = ResultAnalyzer()
        analysis_results = analyzer.analyze_noc_results(self.request_tracker, self.config, self, results, enable_visualization, save_results, save_dir, save_figures, verbose)

        # ResultAnalyzer现在会根据save_figures参数直接处理显示或保存

        return analysis_results

    def _generate_and_display_charts(self, analysis_results: Dict[str, Any]) -> None:
        """生成并显示图表（不保存到文件）"""
        import matplotlib.pyplot as plt

        self.logger.info("生成并显示可视化图表...")

        try:
            # 生成带宽分析图表
            if self._viz_config.get("bandwidth_analysis", False):
                self._show_bandwidth_chart(analysis_results)

            # 生成流量分布图表
            if self._viz_config.get("flow_distribution", False):
                self._show_flow_distribution_chart(analysis_results)

        except Exception as e:
            self.logger.warning(f"生成可视化图表时出错: {e}")
            import traceback

            traceback.print_exc()

    def _show_bandwidth_chart(self, analysis_results: Dict[str, Any]) -> None:
        """显示带宽分析图表"""
        import matplotlib.pyplot as plt

        if "带宽指标" not in analysis_results:
            self.logger.warning("分析结果中没有找到带宽指标数据")
            return

        bandwidth_data = analysis_results["带宽指标"]

        # 创建带宽图表
        fig, ax = plt.subplots(figsize=(10, 6))

        # 绘制总体带宽
        if "总体带宽" in bandwidth_data:
            overall_bw = bandwidth_data["总体带宽"]
            non_weighted = overall_bw.get("非加权带宽_GB/s", 0)
            weighted = overall_bw.get("加权带宽_GB/s", 0)

            categories = ["非加权带宽", "加权带宽"]
            values = [non_weighted, weighted]

            ax.bar(categories, values, color=["skyblue", "lightcoral"])
            ax.set_ylabel("带宽 (GB/s)")
            ax.set_title("CrossRing总体带宽分析")

            # 添加数值标签
            for i, v in enumerate(values):
                ax.text(i, v + max(values) * 0.01, f"{v:.2f}", ha="center", va="bottom")

        plt.tight_layout()
        plt.show()
        print("📊 带宽分析图表已显示")

    def _show_flow_distribution_chart(self, analysis_results: Dict[str, Any]) -> None:
        """显示流量分布图表"""
        import matplotlib.pyplot as plt

        if "延迟指标" not in analysis_results:
            self.logger.warning("分析结果中没有找到延迟指标数据")
            return

        latency_data = analysis_results["延迟指标"]

        # 创建延迟分布图表
        fig, ax = plt.subplots(figsize=(10, 6))

        if "总体延迟" in latency_data:
            overall_lat = latency_data["总体延迟"]
            avg_latency = overall_lat.get("平均延迟_ns", 0)
            max_latency = overall_lat.get("最大延迟_ns", 0)
            min_latency = overall_lat.get("最小延迟_ns", 0)

            categories = ["最小延迟", "平均延迟", "最大延迟"]
            values = [min_latency, avg_latency, max_latency]

            ax.bar(categories, values, color=["lightgreen", "gold", "lightcoral"])
            ax.set_ylabel("延迟 (ns)")
            ax.set_title("CrossRing延迟分布分析")

            # 添加数值标签
            for i, v in enumerate(values):
                ax.text(i, v + max(values) * 0.01, f"{v:.2f}", ha="center", va="bottom")

        plt.tight_layout()
        plt.show()
        print("📊 流量分布图表已显示")

    def _display_visualization_results(self, analysis_results: Dict[str, Any]) -> None:
        """显示可视化结果而不保存到文件"""
        import matplotlib.pyplot as plt

        self.logger.info("显示可视化图表...")

        try:
            # 检查是否有生成的图表文件
            if "可视化文件" in analysis_results and "生成的图表" in analysis_results["可视化文件"]:
                chart_files = analysis_results["可视化文件"]["生成的图表"]

                if chart_files:
                    self.logger.info(f"发现 {len(chart_files)} 个图表文件，正在显示...")

                    # 由于图片已经保存了，我们需要重新生成用于显示
                    # 这里我们可以简单地提示用户图表已生成
                    print("📊 图表已生成，可以在以下文件中查看:")
                    for chart_file in chart_files:
                        print(f"  - {chart_file}")

                    # TODO: 未来可以增加直接显示图片的功能
                    # 需要修改ResultAnalyzer来返回matplotlib figure对象而不仅仅是保存文件

                else:
                    self.logger.info("没有生成图表文件")
            else:
                self.logger.info("分析结果中没有找到可视化文件信息")

        except Exception as e:
            self.logger.warning(f"显示可视化结果时出错: {e}")

    def _analyze_ip_interfaces(self, ip_stats: Dict[str, Any]) -> Dict[str, Any]:
        """分析IP接口统计"""
        summary = {"total_interfaces": len(ip_stats), "by_type": {}, "total_read_transactions": 0, "total_write_transactions": 0, "total_retries": 0}

        for ip_key, stats in ip_stats.items():
            ip_type = ip_key.split("_")[0]

            if ip_type not in summary["by_type"]:
                summary["by_type"][ip_type] = {"count": 0, "read_transactions": 0, "write_transactions": 0, "retries": 0}

            summary["by_type"][ip_type]["count"] += 1
            summary["by_type"][ip_type]["read_transactions"] += stats.get("rn_read_active", 0)
            summary["by_type"][ip_type]["write_transactions"] += stats.get("rn_write_active", 0)
            summary["by_type"][ip_type]["retries"] += stats.get("read_retries", 0) + stats.get("write_retries", 0)

            summary["total_read_transactions"] += stats.get("rn_read_active", 0)
            summary["total_write_transactions"] += stats.get("rn_write_active", 0)
            summary["total_retries"] += stats.get("read_retries", 0) + stats.get("write_retries", 0)

        return summary

    def _analyze_congestion(self) -> Dict[str, Any]:
        """分析拥塞情况"""
        congestion_summary = {"congestion_detected": False, "total_congestion_events": 0, "congestion_rate": 0.0}

        if hasattr(self, "get_congestion_statistics"):
            congestion_stats = self.get_congestion_statistics()
            total_congestion = congestion_stats.get("total_congestion_events", 0)
            total_injections = congestion_stats.get("total_injections", 1)

            congestion_summary["congestion_detected"] = total_congestion > 0
            congestion_summary["total_congestion_events"] = total_congestion
            congestion_summary["congestion_rate"] = total_congestion / total_injections if total_injections > 0 else 0.0

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
        report.append(f"拓扑配置: {self.config.NUM_ROW}x{self.config.NUM_COL}")
        report.append(f"总节点数: {self.config.NUM_NODE}")
        report.append("")

        # 基础指标
        basic = analysis.get("basic_metrics", {})
        report.append("性能指标:")
        report.append(f"  仿真周期: {basic.get('total_cycles', 0):,}")
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
        ip_summary = analysis.get("ip_summary", {})
        report.append("IP接口统计:")
        report.append(f"  总接口数: {ip_summary.get('total_interfaces', 0)}")

        by_type = ip_summary.get("by_type", {})
        for ip_type, stats in by_type.items():
            report.append(f"  {ip_type}: {stats['count']}个接口, " f"读事务={stats['read_transactions']}, " f"写事务={stats['write_transactions']}, " f"重试={stats['retries']}")

        report.append("")

        # 拥塞分析
        congestion = analysis.get("congestion_summary", {})
        if congestion.get("congestion_detected", False):
            report.append("拥塞分析:")
            report.append(f"  拥塞事件: {congestion.get('total_congestion_events', 0)}")
            report.append(f"  拥塞率: {congestion.get('congestion_rate', 0):.2%}")
        else:
            report.append("拥塞分析: 未检测到显著拥塞")

        report.append("")
        report.append("=" * 60)

        return "\n".join(report)

    def _get_ip_type_abbreviation(self, ip_id: str) -> str:
        """获取IP类型缩写"""
        ip_id_lower = ip_id.lower()
        if "gdma" in ip_id_lower:
            # 提取gdma后的数字，如gdma_0 -> G0

            match = re.search(r"gdma[_\-]?(\d+)", ip_id_lower)
            if match:
                return f"G{match.group(1)}"
            return "G0"
        elif "ddr" in ip_id_lower:
            match = re.search(r"ddr[_\-]?(\d+)", ip_id_lower)
            if match:
                return f"D{match.group(1)}"
            return "D0"
        elif "l2m" in ip_id_lower:
            match = re.search(r"l2m[_\-]?(\d+)", ip_id_lower)
            if match:
                return f"L{match.group(1)}"
            return "L0"
        elif "sdma" in ip_id_lower:
            match = re.search(r"sdma[_\-]?(\d+)", ip_id_lower)
            if match:
                return f"S{match.group(1)}"
            return "S0"
        elif "cdma" in ip_id_lower:
            match = re.search(r"cdma[_\-]?(\d+)", ip_id_lower)
            if match:
                return f"C{match.group(1)}"
            return "C0"
        else:
            # 对于其他类型，使用前两个字符加数字
            return f"{ip_id[:2].upper()}0"

    def _register_all_fifos_for_statistics(self) -> None:
        """注册所有FIFO到统计收集器（重写基类方法）"""
        self.logger.info("注册FIFO统计收集...")

        # 注册IP接口的FIFO
        for ip_id, ip_interface in self.ip_interfaces.items():
            node_id = str(ip_interface.node_id)
            ip_abbrev = self._get_ip_type_abbreviation(ip_id)

            # l2h FIFO
            for channel in ["req", "rsp", "data"]:
                if hasattr(ip_interface, "l2h_fifos") and channel in ip_interface.l2h_fifos:
                    fifo = ip_interface.l2h_fifos[channel]
                    if hasattr(fifo, "name") and hasattr(fifo, "stats"):  # 确保是PipelinedFIFO
                        simplified_name = f"{channel}_L2H_{ip_abbrev}"
                        self.fifo_stats_collector.register_fifo(fifo, node_id=node_id, simplified_name=simplified_name)

            # h2l FIFO
            for channel in ["req", "rsp", "data"]:
                if hasattr(ip_interface, "h2l_fifos") and channel in ip_interface.h2l_fifos:
                    fifo = ip_interface.h2l_fifos[channel]
                    if hasattr(fifo, "name") and hasattr(fifo, "stats"):  # 确保是PipelinedFIFO
                        simplified_name = f"{channel}_H2L_{ip_abbrev}"
                        self.fifo_stats_collector.register_fifo(fifo, node_id=node_id, simplified_name=simplified_name)

            # inject FIFO (IP内部注入FIFO)
            for channel in ["req", "rsp", "data"]:
                if hasattr(ip_interface, "inject_fifos") and channel in ip_interface.inject_fifos:
                    fifo = ip_interface.inject_fifos[channel]
                    if hasattr(fifo, "name") and hasattr(fifo, "stats"):  # 确保是PipelinedFIFO
                        simplified_name = f"{channel}_IP_INJ_{ip_abbrev}"
                        self.fifo_stats_collector.register_fifo(fifo, node_id=node_id, simplified_name=simplified_name)

            # ip_processing FIFO (IP内部处理FIFO)
            for channel in ["req", "rsp", "data"]:
                if hasattr(ip_interface, "ip_processing_fifos") and channel in ip_interface.ip_processing_fifos:
                    fifo = ip_interface.ip_processing_fifos[channel]
                    if hasattr(fifo, "name") and hasattr(fifo, "stats"):  # 确保是PipelinedFIFO
                        simplified_name = f"{channel}_IP_PROC_{ip_abbrev}"
                        self.fifo_stats_collector.register_fifo(fifo, node_id=node_id, simplified_name=simplified_name)

        # 注册CrossRing节点的FIFO
        for node_id, node in self.nodes.items():
            node_id_str = str(node_id)

            # 注册inject direction FIFOs (注入队列输出)
            if hasattr(node, "inject_direction_fifos"):
                # 结构: inject_direction_fifos[channel][direction]
                for channel in ["req", "rsp", "data"]:
                    if channel in node.inject_direction_fifos:
                        direction_dict = node.inject_direction_fifos[channel]
                        if isinstance(direction_dict, dict):
                            for direction in ["TR", "TL", "TU", "TD", "EQ"]:
                                if direction in direction_dict:
                                    fifo = direction_dict[direction]
                                    if hasattr(fifo, "name") and hasattr(fifo, "stats"):
                                        simplified_name = f"{channel}_IQ_OUT_{direction}"
                                        self.fifo_stats_collector.register_fifo(fifo, node_id=node_id_str, simplified_name=simplified_name)

            # 注册eject input FIFOs (弹出队列输入)
            if hasattr(node, "eject_input_fifos"):
                # 结构: eject_input_fifos[channel][direction]
                for channel in ["req", "rsp", "data"]:
                    if channel in node.eject_input_fifos:
                        direction_dict = node.eject_input_fifos[channel]
                        if isinstance(direction_dict, dict):
                            for direction in ["TU", "TD", "TR", "TL"]:
                                if direction in direction_dict:
                                    fifo = direction_dict[direction]
                                    if hasattr(fifo, "name") and hasattr(fifo, "stats"):
                                        simplified_name = f"{channel}_EQ_IN_{direction}"
                                        self.fifo_stats_collector.register_fifo(fifo, node_id=node_id_str, simplified_name=simplified_name)

            # 注册ip_inject_channel_buffers (IP注入通道缓冲)
            if hasattr(node, "ip_inject_channel_buffers"):
                for ip_id, channels in node.ip_inject_channel_buffers.items():
                    if isinstance(channels, dict):
                        ip_abbrev = self._get_ip_type_abbreviation(ip_id)
                        for channel, fifo in channels.items():
                            if hasattr(fifo, "name") and hasattr(fifo, "stats"):
                                simplified_name = f"{channel}_IP_CH_{ip_abbrev}"
                                self.fifo_stats_collector.register_fifo(fifo, node_id=node_id_str, simplified_name=simplified_name)

            # 注册ip_eject_channel_buffers (IP弹出通道缓冲)
            if hasattr(node, "ip_eject_channel_buffers"):
                for ip_id, channels in node.ip_eject_channel_buffers.items():
                    if isinstance(channels, dict):
                        ip_abbrev = self._get_ip_type_abbreviation(ip_id)
                        for channel, fifo in channels.items():
                            if hasattr(fifo, "name") and hasattr(fifo, "stats"):
                                simplified_name = f"{channel}_IP_EJECT_{ip_abbrev}"
                                self.fifo_stats_collector.register_fifo(fifo, node_id=node_id_str, simplified_name=simplified_name)

            # 注册ring_bridge input FIFOs (环桥输入)
            if hasattr(node, "ring_bridge_input_fifos"):
                for channel in ["req", "rsp", "data"]:
                    if channel in node.ring_bridge_input_fifos:
                        direction_dict = node.ring_bridge_input_fifos[channel]
                        if isinstance(direction_dict, dict):
                            for direction, fifo in direction_dict.items():
                                if hasattr(fifo, "name") and hasattr(fifo, "stats"):
                                    simplified_name = f"{channel}_RB_IN_{direction}"
                                    self.fifo_stats_collector.register_fifo(fifo, node_id=node_id_str, simplified_name=simplified_name)

            # 注册ring_bridge output FIFOs (环桥输出)
            if hasattr(node, "ring_bridge_output_fifos"):
                for channel in ["req", "rsp", "data"]:
                    if channel in node.ring_bridge_output_fifos:
                        direction_dict = node.ring_bridge_output_fifos[channel]
                        if isinstance(direction_dict, dict):
                            for direction, fifo in direction_dict.items():
                                if hasattr(fifo, "name") and hasattr(fifo, "stats"):
                                    simplified_name = f"{channel}_RB_OUT_{direction}"
                                    self.fifo_stats_collector.register_fifo(fifo, node_id=node_id_str, simplified_name=simplified_name)

        # 统计注册的FIFO数量
        total_fifos = len(self.fifo_stats_collector.fifo_registry)
        self.logger.info(f"已注册 {total_fifos} 个FIFO到统计收集器")

    def export_fifo_statistics(self, filename: str = None, output_dir: str = "results") -> str:
        """
        导出FIFO统计信息到CSV文件

        Args:
            filename: 文件名（不包含扩展名），如果为None则自动生成
            output_dir: 输出目录

        Returns:
            导出的文件路径
        """
        return self.fifo_stats_collector.export_to_csv(filename, output_dir)

    def get_fifo_statistics_summary(self) -> str:
        """获取FIFO统计摘要报告"""
        return self.fifo_stats_collector.get_summary_report()

    def __del__(self):
        """析构函数"""
        if hasattr(self, "logger"):
            self.logger.debug("CrossRing模型对象被销毁")

    # ========== 实现BaseNoCModel抽象方法 ==========

    def __repr__(self) -> str:
        """字符串表示"""
        return f"CrossRingModel({self.config.config_name}, " f"{self.config.NUM_ROW}x{self.config.NUM_COL}, " f"cycle={self.cycle}, " f"active_requests={self.get_total_active_requests()})"

    # ========== 统一接口方法（用于兼容性） ==========

    def initialize_network(self) -> None:
        """初始化网络（统一接口）"""
        self._setup_ip_interfaces()
        self._setup_crossring_networks()
        print(f"CrossRing网络初始化完成: {self.config.NUM_ROW}x{self.config.NUM_COL}")

    def inject_packet(self, src_node: NodeId, dst_node: NodeId, op_type: str = "R", burst_size: int = 4, cycle: int = None, packet_id: str = None) -> bool:
        """注入包（统一接口）"""
        if cycle is None:
            cycle = self.cycle

        # 生成包ID
        if packet_id is None:
            packet_id = f"pkt_{src_node}_{dst_node}_{op_type}_{cycle}"

        # 开始追踪请求
        if self.debug_enabled or packet_id in self.trace_packets:
            self.request_tracker.start_request(packet_id, src_node, dst_node, op_type, burst_size, cycle)

        # 使用现有的inject_test_traffic方法
        packet_ids = self.inject_request(source=src_node, destination=dst_node, req_type=op_type, count=1, burst_length=burst_size)

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

        # ========== 调试功能接口 ==========

        # 打印验证结果
        validation = self.validate_traffic_correctness()
        print(f"\n流量正确性验证:")
        print(f"  完成率: {validation['completion_rate']:.1f}%")
        print(f"  响应错误: {validation['response_errors']}")
        print(f"  数据错误: {validation['data_errors']}")
        print(f"  结果: {'正确' if validation['is_correct'] else '有错误'}")

    def set_debug_sleep_time(self, sleep_time: float):
        """
        设置debug模式下每个周期的休眠时间

        Args:
            sleep_time: 休眠时间（秒），0表示不休眠
        """
        self.debug_config["sleep_time"] = sleep_time
        self.logger.info(f"设置debug休眠时间: {sleep_time}秒/周期")

    # ========== 实现BaseNoCModel抽象方法 ==========


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
