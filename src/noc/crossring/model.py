"""
CrossRing主模型类实现。

基于C2C仓库的架构，提供完整的CrossRing NoC仿真模型，
包括IP接口管理、网络组件和仿真循环控制。
集成真实的环形拓扑、环形桥接和四方向系统。
"""

from __future__ import annotations
from typing import Dict, List, Any, Optional, Tuple
import os
from collections import defaultdict
import time
from datetime import datetime
from enum import Enum
import numpy as np
import re
from pathlib import Path

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
        super().__init__(config, model_name="CrossRing", traffic_file_path=traffic_file_path)

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

        # 等待统计
        self.waiting_stats = {}  # {packet_id: {"start_cycle": int, "total_wait": int, "current_wait": int}}

        # FIFO统计收集器
        self.fifo_stats_collector = FIFOStatsCollector()

        # 可视化配置
        self._viz_config = {"flow_distribution": False, "bandwidth_analysis": False, "save_figures": True, "save_dir": "output"}

        # 实时可视化组件
        self._realtime_visualizer = None
        self._visualization_enabled = False
        self._visualization_initialized = False
        self._visualization_frame_interval = 0.5  # 每帧间隔时间（秒）
        self._visualization_update_interval = 10  # 每多少个周期更新一次可视化
        self._visualization_start_cycle = 0  # 从哪个周期开始可视化
        self._paused = False  # 可视化暂停状态

        # 初始化模型（不包括IP接口创建，IP接口将在setup_traffic_scheduler中创建）
        self.initialize_model()

        # 初始化完成后注册FIFO统计
        self._register_all_fifos_for_statistics()

        # 验证CrossRing网络初始化
        if len(self.nodes) != self.config.NUM_NODE:
            raise RuntimeError(f"CrossRing网络初始化失败: 期望{self.config.NUM_NODE}个节点，实际{len(self.nodes)}个")

    def _create_topology_instance(self, config) -> CrossRingTopology:
        """
        创建CrossRing拓扑实例

        Args:
            config: CrossRing配置对象

        Returns:
            CrossRing拓扑实例
        """
        topology = CrossRingTopology(config)
        return topology

    def _should_skip_waiting_flit(self, flit) -> bool:
        """判断flit是否在等待状态，不需要打印"""
        if hasattr(flit, "departure_cycle") and hasattr(flit, "flit_position"):
            # L2H状态且还未到departure时间 = 等待状态
            if flit.flit_position == "L2H" and flit.departure_cycle > self.cycle:
                return True
            # IP_eject状态且位置没有变化，也算等待状态
            if flit.flit_position == "IP_eject":
                # 检查flit是否有变化，如果没有变化就跳过
                if hasattr(flit, "_last_stable_cycle"):
                    if self.cycle - flit._last_stable_cycle > 2:  # 在IP_eject超过2个周期就跳过
                        return True
                else:
                    flit._last_stable_cycle = self.cycle
        return False

    def _update_waiting_stats(self, packet_id: str, has_active_flit: bool, all_flits: list):
        """更新等待统计（简化版）"""
        waiting_flits = [f for f in all_flits if self._should_skip_waiting_flit(f)]

        # 初始化统计
        if packet_id not in self.waiting_stats:
            self.waiting_stats[packet_id] = {"start_cycle": 0, "total_wait": 0, "is_waiting": False, "resume_printed": False}

        stats = self.waiting_stats[packet_id]

        # 状态转换
        if waiting_flits and not stats["is_waiting"]:
            # 开始等待
            stats["start_cycle"] = self.cycle
            stats["is_waiting"] = True
            stats["resume_printed"] = False
        elif not waiting_flits and stats["is_waiting"]:
            # 等待结束
            wait_duration = self.cycle - stats["start_cycle"]
            stats["total_wait"] += wait_duration
            stats["is_waiting"] = False
            # 标记需要打印等待恢复信息
            if wait_duration > 1:
                stats["resume_printed"] = True
                return wait_duration  # 返回等待时长供调用者打印
        return 0

    def _print_debug_info(self):
        """打印调试信息"""
        if not self.debug_enabled or not hasattr(self, "request_tracker"):
            return

        trace_packets = self.trace_packets if self.trace_packets else self.debug_packet_ids
        cycle_header_printed = False
        completed_packets = set()
        flits_to_print = []

        for packet_id in list(trace_packets):
            if self._should_debug_packet(packet_id):
                lifecycle = self._get_packet_lifecycle(packet_id)
                if lifecycle:
                    all_flits = lifecycle.request_flits + lifecycle.response_flits + lifecycle.data_flits

                    if all_flits or lifecycle.current_state == RequestState.COMPLETED:
                        # 收集活跃的flits
                        active_flits = [flit for flit in all_flits if not self._should_skip_waiting_flit(flit)]

                        # 检查新的DATA flit
                        for flit in all_flits:
                            if hasattr(flit, "flit_type") and flit.flit_type == "data" and hasattr(flit, "flit_position") and flit.flit_position not in ["IP_eject"]:
                                if flit not in active_flits:
                                    active_flits.append(flit)

                        if active_flits:
                            flits_to_print.extend(all_flits)

                        # 处理等待统计
                        wait_duration = self._update_waiting_stats(packet_id, bool(active_flits), all_flits)
                        if wait_duration > 0:
                            if not cycle_header_printed:
                                print(f"周期{self.cycle}: ")
                                cycle_header_printed = True
                            print(f"  📊 请求{packet_id}: 等待{wait_duration}周期后恢复传输")

                    # 处理完成状态
                    if lifecycle.current_state.value == "completed":
                        if not cycle_header_printed:
                            print(f"周期{self.cycle}: ")
                            cycle_header_printed = True

                        total_wait = self.waiting_stats.get(packet_id, {}).get("total_wait", 0)
                        wait_info = f" (总等待: {total_wait}周期)" if total_wait > 0 else ""
                        print(f"✅ 请求{packet_id}已完成，停止跟踪{wait_info}")
                        completed_packets.add(packet_id)

        # 打印结果
        if flits_to_print:
            if not cycle_header_printed:
                print(f"周期{self.cycle}: ")
            print(f" ", end="")
            for flit in flits_to_print:
                print(f"{flit}", end=" | ")
            print("")

        # 清理已完成的packets
        for packet_id in completed_packets:
            self.debug_packet_ids.discard(packet_id)
            self.trace_packets.discard(packet_id)

        if len(self.trace_packets) + len(self.debug_packet_ids) == 0 and self.debug_enabled:
            print(f"🎯 所有跟踪请求已完成，自动关闭debug模式")
            self.disable_debug()

        # debug sleep
        if (flits_to_print or completed_packets) and self.debug_config["sleep_time"] > 0:
            import time

            time.sleep(self.debug_config["sleep_time"])

    def _get_packet_lifecycle(self, packet_id):
        """获取packet的lifecycle，支持整数和字符串形式"""
        # 直接查找
        lifecycle = self.request_tracker.active_requests.get(packet_id) or self.request_tracker.completed_requests.get(packet_id)

        if lifecycle:
            return lifecycle

        # 字符串 -> 整数转换
        if isinstance(packet_id, str) and packet_id.isdigit():
            int_packet_id = int(packet_id)
            return self.request_tracker.active_requests.get(int_packet_id) or self.request_tracker.completed_requests.get(int_packet_id)

        # 整数 -> 字符串转换
        elif isinstance(packet_id, int):
            str_packet_id = str(packet_id)
            return self.request_tracker.active_requests.get(str_packet_id) or self.request_tracker.completed_requests.get(str_packet_id)

        return None

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
            else:
                return False

            return True

        except Exception as e:
            raise RuntimeError(f"创建IP接口失败: {key} - {e}")
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

    def _create_specific_ip_interfaces(self, required_ips: List[Tuple[int, str]]) -> None:
        """创建特定的IP接口"""
        for node_id, ip_type in required_ips:
            # 验证ip_type格式
            if not ip_type or not isinstance(ip_type, str):
                continue

            # 使用多维字典结构 [node_id][ip_type]
            if node_id not in self.ip_interfaces:
                self.ip_interfaces[node_id] = {}

            # 检查该节点是否已有此类型IP
            if ip_type in self.ip_interfaces[node_id]:
                continue

            try:
                ip_interface = CrossRingIPInterface(config=self.config, ip_type=ip_type, node_id=node_id, model=self)
                self.ip_interfaces[node_id][ip_type] = ip_interface
                self._ip_registry[f"{ip_type}_{node_id}"] = ip_interface  # 注册时使用组合键以保证唯一性

                # 连接IP到对应的节点
                if node_id in self.nodes:
                    self.nodes[node_id].connect_ip(ip_type)

            except Exception as e:
                raise RuntimeError(f"创建IP接口失败: {ip_type} - {e}")
                continue

        # 所有IP接口创建完成
        total_ips = sum(len(node_ips) for node_ips in self.ip_interfaces.values())

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
            # 分析所有traffic文件中需要的IP接口，同时收集IP类型用于更新CH_NAME_LIST
            all_required_ips = []
            all_ip_types = set()

            # 直接解析traffic文件，避免重复读取
            for chain in traffic_chains:
                for filename in chain:
                    abs_path = os.path.join(file_path, filename)

                    # 解析文件获取IP类型
                    with open(abs_path, "r") as f:
                        for line in f:
                            line = line.strip()
                            if not line or line.startswith("#"):
                                continue

                            # 支持逗号和空格分隔符
                            if "," in line:
                                parts = line.split(",")
                            else:
                                parts = line.split()

                            if len(parts) < 7:
                                continue

                            try:
                                _, src_node, src_ip, dst_node, dst_ip, _, _ = parts[:7]
                                src_node, dst_node = int(src_node), int(dst_node)

                                # 收集需要的IP接口和类型
                                all_required_ips.append((src_node, src_ip))
                                all_required_ips.append((dst_node, dst_ip))
                                all_ip_types.add(src_ip)
                                all_ip_types.add(dst_ip)

                            except (ValueError, IndexError):
                                continue

            # 直接使用traffic文件中的IP类型更新CH_NAME_LIST
            if all_ip_types:
                traffic_ch_names = sorted(list(all_ip_types))  # 保持一致性
                self.config.update_channel_names(traffic_ch_names)

            # 去重并保持输入文件顺序的稳定性
            # 使用dict.fromkeys保留首次出现顺序，避免set引入的非确定性
            unique_required_ips = list(dict.fromkeys(all_required_ips))

            # 动态创建需要的IP接口
            self._create_specific_ip_interfaces(unique_required_ips)

        except Exception as e:
            import traceback

            traceback.print_exc()

        # 调用父类方法设置TrafficScheduler
        super().setup_traffic_scheduler(traffic_chains, traffic_file_path)

        # 重新注册IP接口的FIFO（因为IP接口是在traffic setup时创建的）
        self._register_ip_fifos_for_statistics()

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
                node = CrossRingNode(node_id, coordinates, self.config, topology=self.topology)
                self.nodes[node_id] = node
            except Exception as e:
                import traceback

                traceback.print_exc()
                raise RuntimeError(f"创建节点{node_id}失败: {e}")

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
        normal_slices = getattr(self.config.basic_config, "SLICE_PER_LINK", 8)
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
                    link = CrossRingLink(link_id, node_id, neighbor_id, direction, self.config, num_slices)
                    self.links[link_id] = link
                    link_count += 1
                except Exception as e:
                    print(f"DEBUG: 创建链接失败 {link_id}: {e}")
                    import traceback

                    traceback.print_exc()
                    raise RuntimeError(f"创建链接{link_id}失败: {e}")

    def _connect_slices_to_crosspoints(self) -> None:
        """连接RingSlice到CrossPoint"""
        connected_count = 0
        for node_id, node in self.nodes.items():
            for direction_str in self.DIRECTIONS:
                crosspoint_direction = self._get_crosspoint_direction(direction_str)
                crosspoint = node.get_crosspoint(crosspoint_direction)

                if not crosspoint:
                    continue

                # 获取出链路
                connections = self._get_ring_connections(node_id)
                neighbor_id = connections.get(direction_str)
                out_link = None

                if neighbor_id is not None:
                    out_link_id = self._create_link_id(node_id, direction_str, neighbor_id)
                    out_link = self.links.get(out_link_id)
                    if out_link:
                        connected_count += 1

                # 连接所有通道的slices
                for channel in self.CHANNELS:
                    # 连接departure slice
                    if out_link and out_link.ring_slices[channel]:
                        departure_slice = out_link.ring_slices[channel][0]
                        crosspoint.connect_slice(direction_str, "departure", departure_slice, channel)

                    # 连接arrival slice
                    arrival_slice = self._find_arrival_slice(node_id, direction_str, channel)
                    if arrival_slice:
                        crosspoint.connect_slice(direction_str, "arrival", arrival_slice, channel)

    def _create_link_id(self, node_id: int, direction_str: str, neighbor_id: int) -> str:
        """创建链路ID"""
        if neighbor_id == node_id:
            # 自环链路
            reverse_direction = self.REVERSE_DIRECTION_MAP.get(direction_str, direction_str)
            return f"link_{node_id}_{direction_str}_{reverse_direction}_{neighbor_id}"
        else:
            # 普通链路
            return f"link_{node_id}_{direction_str}_{neighbor_id}"

    def _find_arrival_slice(self, node_id: int, direction_str: str, channel: str) -> Any:
        """查找arrival slice - 统一的查找逻辑"""
        # 首先查找来自其他节点的同方向链路
        for link_id, link in self.links.items():
            if link.dest_node == node_id and direction_str in link_id and link.source_node != node_id and link.ring_slices[channel]:
                return link.ring_slices[channel][-1]  # 其他节点链路的最后slice

        # 如果没找到，使用本节点的反向自环链路
        reverse_direction = self.REVERSE_DIRECTION_MAP.get(direction_str, direction_str)
        self_loop_link_id = f"link_{node_id}_{reverse_direction}_{direction_str}_{node_id}"

        self_loop_link = self.links.get(self_loop_link_id)
        if self_loop_link and self_loop_link.ring_slices[channel] and len(self_loop_link.ring_slices[channel]) > 1:
            return self_loop_link.ring_slices[channel][1]  # 自环的第1个slice

        return None

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

        # 链路间slice连接完成

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
                    print(f"⚠️  警告：节点{node_id}方向{direction_str}的链路{out_link_id}找不到下一个链路{next_link_id}")
                    print(f"   可用链路：{list(self.links.keys())}")
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

    # 方向反转映射常量
    # 网络连接配置常量
    REVERSE_DIRECTION_MAP = {"TR": "TL", "TL": "TR", "TU": "TD", "TD": "TU"}
    DIRECTIONS = ["TR", "TL", "TU", "TD"]
    CHANNELS = ["req", "rsp", "data"]
    HORIZONTAL_DIRECTIONS = ["TR", "TL"]
    VERTICAL_DIRECTIONS = ["TU", "TD"]

    def _get_node_coordinates(self, node_id: NodeId) -> Tuple[int, int]:
        """获取节点坐标（使用topology实例）"""
        return self.topology.get_node_position(node_id)

    def _get_crosspoint_direction(self, direction_str: str) -> str:
        """根据方向字符串获取CrossPoint类型"""
        return "horizontal" if direction_str in self.HORIZONTAL_DIRECTIONS else "vertical"

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

    def _sync_global_clock(self) -> None:
        """重写时钟同步阶段：添加CrossRing节点时钟同步"""
        # 调用基类的时钟同步
        super()._sync_global_clock()

    def step(self) -> None:
        self.cycle += 1

        # 阶段0.1：TrafficScheduler处理请求注入（如果有配置）
        if hasattr(self, "traffic_scheduler") and self.traffic_scheduler:
            ready_requests = self.traffic_scheduler.get_ready_requests(self.cycle)
            self._inject_traffic_requests(ready_requests)

        for node_interfaces in self.ip_interfaces.values():
            for ip_interface in node_interfaces.values():
                ip_interface.step_compute_phase(self.cycle)
                ip_interface.step_update_phase(self.cycle)

        self._step_node_compute_phase()
        self._step_link_compute_phase()

        self._step_node_update_phase()
        self._step_link_update_phase()

        # 调试功能
        if self.debug_enabled:
            self._print_debug_info()
            self.debug_func()

        # 定期输出调试信息
        if self.cycle % self.debug_config["log_interval"] == 0:
            self._log_periodic_status()

        # 实时可视化更新
        if self._visualization_enabled and self.cycle >= self._visualization_start_cycle:
            if not self._visualization_initialized:
                self._initialize_visualization()

            if self._realtime_visualizer:
                self._update_visualization()

    def _step_link_compute_phase(self) -> None:
        """Link层计算阶段：计算slice移动规划，不实际移动flit"""
        # 所有链路的计算阶段
        for link_id, link in self.links.items():
            if hasattr(link, "step_compute_phase"):
                link.step_compute_phase(self.cycle)

    def _step_link_update_phase(self) -> None:
        """Link层更新阶段：执行slice移动，腾空slot[0]位置"""
        # 所有链路的更新阶段（环路移动，腾空slot[0]位置）
        for link_id, link in self.links.items():
            if hasattr(link, "step_update_phase"):
                link.step_update_phase(self.cycle)

    def _step_node_compute_phase(self) -> None:
        """Node层计算阶段：计算注入/弹出/转发决策，不实际传输flit"""
        # 所有节点的计算阶段
        for node_id, node in self.nodes.items():
            if hasattr(node, "step_compute_phase"):
                node.step_compute_phase(self.cycle)

    def _step_node_update_phase(self) -> None:
        """Node层更新阶段：执行flit传输，包括注入到腾空的slot[0]"""
        # 所有节点的更新阶段（CrossPoint注入到腾空的slot）
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

    def setup_debug(self, trace_packets: List[str] = None, update_interval: float = 0.0) -> None:
        """
        启用调试模式（CrossRing扩展版本）

        Args:
            trace_packets: 要跟踪的请求ID列表，设置后启用请求跟踪功能
            update_interval: 每个周期的暂停时间（用于实时观察）
        """

        # 设置调试参数
        self.debug_enabled = True
        self.debug_config["sleep_time"] = update_interval

        if trace_packets:
            self.debug_packet_ids.update(trace_packets)

        if update_interval > 0:
            self.debug_config["sleep_time"] = update_interval

        # 调用base类的enable_debug，传递level=1作为兼容参数
        super().setup_debug(1, trace_packets, update_interval)

    def setup_result_analysis(
        self,
        # 图片生成控制
        flow_distribution_fig: bool = False,
        bandwidth_analysis_fig: bool = False,
        latency_analysis_fig: bool = False,
        save_figures: bool = True,
        # CSV文件导出控制
        export_request_csv: bool = True,
        export_fifo_csv: bool = True,
        export_link_csv: bool = True,
        # 通用设置
        save_dir: str = "",
    ) -> None:
        """
        配置结果分析

        图片生成控制:
            flow_distribution_fig: 是否生成流量分布图
            bandwidth_analysis_fig: 是否生成带宽分析图
            latency_analysis_fig: 是否生成延迟分析图
            save_figures: 是否保存图片文件到磁盘

        CSV文件导出控制:
            export_request_csv: 是否导出请求统计CSV文件
            export_fifo_csv: 是否导出FIFO统计CSV文件
            export_link_csv: 是否导出链路统计CSV文件

        通用设置:
            save_dir: 保存目录，如果为None或空字符串则不保存任何文件
        """
        # 如果save_dir为None或空字符串，禁用所有保存功能
        if not save_dir:
            save_dir = ""
        else:
            save_dir = f"{save_dir}{self.traffic_scheduler.get_save_filename()}"

        # 图片保存需要同时满足save_dir不为空且save_figures为True
        actual_save_figures = bool(save_dir) and save_figures

        self._viz_config.update(
            {
                # 图片生成控制
                "flow_distribution": flow_distribution_fig,
                "bandwidth_analysis": bandwidth_analysis_fig,
                "latency_analysis": latency_analysis_fig,
                "save_figures": actual_save_figures,
                # CSV导出控制
                "export_request_csv": export_request_csv,
                "export_fifo_csv": export_fifo_csv,
                "export_link_csv": export_link_csv,
                # 通用设置
                "save_dir": save_dir,
            }
        )

    def setup_visualization(self, enable: bool = True, update_interval: int = 1, start_cycle: int = 0) -> None:
        """
        配置实时可视化

        Args:
            enable: 是否启用可视化
            update_interval: 更新间隔（周期数/秒）
            start_cycle: 开始可视化的周期
        """
        self._visualization_enabled = enable
        self._visualization_update_interval = max(update_interval, 0.05) if enable else update_interval
        self._visualization_frame_interval = update_interval  # 兼容性
        self._visualization_start_cycle = start_cycle
        self._visualization_initialized = False

        if enable:
            print(f"✅ 可视化已启用: 更新间隔={update_interval}, 开始周期={start_cycle}")
            print("   提示: 可视化窗口将在仿真开始后自动打开")
        else:
            print("❌ 可视化已禁用")

    def cleanup_visualization(self) -> None:
        """
        清理可视化资源，禁用时间间隔

        用于用户按'q'退出可视化后，让仿真无延迟运行
        """
        if self._visualization_enabled:
            self._visualization_enabled = False
            self._visualization_frame_interval = 0.0  # 禁用时间间隔
            self.debug_config["sleep_time"] = 0.0  # 同时禁用debug模式的延迟
            self.user_interrupted = False  # 重置中断标志，让仿真继续运行

        if self._realtime_visualizer:
            try:
                self._realtime_visualizer = None
                self._visualization_initialized = False
            except Exception as e:
                print(f"⚠️  关闭可视化失败: {e}")

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

    # IP类型偏好配置
    IP_TYPE_PREFERENCES = {"request": ["gdma", "sdma", "cdma"], "response": ["ddr", "l2m"]}  # RN端偏好DMA类  # SN端偏好存储类

    def _find_ip_interface(self, node_id: NodeId, req_type: str = None, ip_type: str = None, preference_type: str = None) -> Optional[CrossRingIPInterface]:
        """
        统一的IP接口查找方法

        Args:
            node_id: 节点ID
            req_type: 请求类型 (可选)
            ip_type: 指定IP类型 (可选)
            preference_type: 偏好类型 ('request'或'response', 可选)

        Returns:
            找到的IP接口，未找到返回None
        """
        # 精确匹配指定IP类型
        if ip_type:
            return self._find_exact_ip_interface(node_id, ip_type)

        # 根据偏好类型查找
        if preference_type and preference_type in self.IP_TYPE_PREFERENCES:
            preferred_types = self.IP_TYPE_PREFERENCES[preference_type]
            return self._find_preferred_ip_interface(node_id, preferred_types, preference_type)

        # 返回任意可用接口
        return self._find_any_ip_interface(node_id)

    def _find_exact_ip_interface(self, node_id: NodeId, ip_type: str) -> Optional[CrossRingIPInterface]:
        """查找精确匹配的IP接口"""
        if node_id in self.ip_interfaces and ip_type in self.ip_interfaces[node_id]:
            return self.ip_interfaces[node_id][ip_type]

        raise ValueError(f"未找到指定IP接口: 节点{node_id}的{ip_type}")

    def _find_preferred_ip_interface(self, node_id: NodeId, preferred_types: list, context: str) -> Optional[CrossRingIPInterface]:
        """根据偏好类型查找IP接口"""
        all_ips = [ip for key, ip in self._ip_registry.items() if ip.node_id == node_id]
        if not all_ips:
            raise ValueError(f"节点{node_id}没有任何IP接口可用于{context}")

        # 查找偏好类型的接口
        for preferred_type in preferred_types:
            preferred_ips = [ip for ip in all_ips if ip.ip_type.startswith(preferred_type) or ip.ip_type == preferred_type]
            if preferred_ips:
                return preferred_ips[0]

        # 没有偏好类型时使用任意接口
        return all_ips[0]

    def _find_any_ip_interface(self, node_id: NodeId) -> Optional[CrossRingIPInterface]:
        """查找任意可用的IP接口"""
        if node_id in self.ip_interfaces:
            node_interfaces = list(self.ip_interfaces[node_id].values())
            if node_interfaces:
                return node_interfaces[0]

        raise ValueError(f"节点{node_id}没有任何IP接口")

    def _find_ip_interface_for_request(self, node_id: NodeId, req_type: str, ip_type: str = None) -> Optional[CrossRingIPInterface]:
        """为请求查找合适的IP接口（优先DMA类）"""
        return self._find_ip_interface(node_id, req_type, ip_type, "request")

    def _find_ip_interface_for_response(self, node_id: NodeId, req_type: str, ip_type: str = None) -> Optional[CrossRingIPInterface]:
        """为响应查找合适的IP接口（优先存储类）"""
        return self._find_ip_interface(node_id, req_type, ip_type, "response")

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
            # 任何一个分析类型启用就启用可视化
            viz_enabled = self._viz_config.get("flow_distribution", False) or self._viz_config.get("bandwidth_analysis", False) or self._viz_config.get("latency_analysis", False)
            # 总是使用配置的save_dir和save_figures，不管是否启用可视化
            save_figures = self._viz_config["save_figures"]
            save_dir = self._viz_config["save_dir"]
            # 只有当用户明确启用可视化功能时才启用
            enable_visualization = viz_enabled

        # 只要save_dir不为空就保存结果文件
        if save_dir:
            # 创建基于时间戳的分析结果目录
            timestamp = int(time.time())
            timestamped_dir = os.path.join(save_dir, f"analysis_{timestamp}")

            # 根据配置决定是否导出链路统计数据到CSV
            link_csv_path = ""
            if self._viz_config.get("export_link_csv", True):
                self._collect_and_export_link_statistics(timestamped_dir, timestamp)
                link_csv_path = os.path.join(timestamped_dir, f"link_bandwidth_{timestamp}.csv")

            # 根据配置决定是否导出FIFO统计数据到CSV
            fifo_csv_path = ""
            if self._viz_config.get("export_fifo_csv", True):
                fifo_csv_path = self.export_fifo_statistics(f"fifo_stats_{timestamp}", timestamped_dir)
        else:
            timestamped_dir = ""
            # 如果save_dir为空，禁用结果保存
            save_results = False

        analyzer = ResultAnalyzer()
        # 传递可视化配置到ResultAnalyzer
        viz_config = getattr(self, "_viz_config", {})

        # 添加CSV文件路径和统计数量到配置中
        if save_dir:
            viz_config["fifo_csv_path"] = fifo_csv_path
            viz_config["link_csv_path"] = link_csv_path
            # 添加数量信息
            if fifo_csv_path:
                viz_config["fifo_count"] = len(self.fifo_stats_collector.fifo_registry)
            if link_csv_path and hasattr(self, "links"):
                viz_config["link_count"] = len(self.links)

        analysis_results = analyzer.analyze_noc_results(self.request_tracker, self.config, self, results, enable_visualization, save_results, timestamped_dir, save_figures, verbose, viz_config)

        # ResultAnalyzer现在会根据save_figures参数直接处理显示或保存

        return analysis_results

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

    # FIFO注册配置
    FIFO_REGISTRY_CONFIG = {
        "node_fifos": [
            {"attr_path": "inject_input_fifos", "name_prefix": "IQ_OUT", "directions": ["TR", "TL", "TU", "TD", "EQ"]},
            {"attr_path": "eject_queue.eject_input_fifos", "name_prefix": "EQ_IN", "directions": ["TU", "TD", "TR", "TL"]},
            {"attr_path": "ring_bridge.ring_bridge_input_fifos", "name_prefix": "RB_IN", "directions": None},  # 使用所有可用方向
            {"attr_path": "ring_bridge.ring_bridge_output_fifos", "name_prefix": "RB_OUT", "directions": None},  # 使用所有可用方向
        ],
        "ip_fifos": [{"attr_path": "l2h_fifos", "name_prefix": "L2H"}, {"attr_path": "h2l_fifos", "name_prefix": "H2L"}],
        "channel_buffers": [{"attr_path": "ip_inject_channel_buffers", "name_prefix": "IP_CH"}, {"attr_path": "ip_eject_channel_buffers", "name_prefix": "IP_EJECT"}],
    }

    def _register_all_fifos_for_statistics(self) -> None:
        """注册所有FIFO到统计收集器（重写基类方法）"""
        # IP接口的FIFO将在setup_traffic_scheduler后单独注册

        # 注册CrossRing节点的FIFO
        for node_id, node in self.nodes.items():
            node_id_str = str(node_id)
            self._register_node_fifos(node, node_id_str)

    def _register_node_fifos(self, node: Any, node_id_str: str) -> None:
        """注册节点的FIFO"""
        channel = "data"  # 只注册data通道

        for fifo_config in self.FIFO_REGISTRY_CONFIG["node_fifos"]:
            self._register_fifo_group(node, node_id_str, channel, fifo_config["attr_path"], fifo_config["name_prefix"], fifo_config.get("directions"))

    def _register_fifo_group(self, obj: Any, node_id_str: str, channel: str, attr_path: str, name_prefix: str, directions: list = None) -> None:
        """注册一组FIFO"""
        # 获取嵌套属性
        fifo_container = self._get_nested_attr(obj, attr_path)
        if not fifo_container or channel not in fifo_container:
            return

        direction_dict = fifo_container[channel]
        if not isinstance(direction_dict, dict):
            return

        # 决定要遍历的方向
        target_directions = directions if directions else direction_dict.keys()

        for direction in target_directions:
            if direction in direction_dict:
                fifo = direction_dict[direction]
                if hasattr(fifo, "name") and hasattr(fifo, "stats"):
                    simplified_name = f"{channel}_{name_prefix}_{direction}"
                    self.fifo_stats_collector.register_fifo(fifo, node_id=node_id_str, simplified_name=simplified_name)

    def _get_nested_attr(self, obj: Any, attr_path: str) -> Any:
        """获取嵌套属性"""
        attrs = attr_path.split(".")
        result = obj

        for attr in attrs:
            if hasattr(result, attr):
                result = getattr(result, attr)
            else:
                return None

        return result

    def _register_ip_fifos_for_statistics(self) -> None:
        """注册IP接口的FIFO到统计收集器"""
        channel = "data"  # 只注册data通道

        # 注册IP接口的FIFO
        for ip_id, ip_interface in self.ip_interfaces.items():
            interface_obj, node_id = self._extract_ip_interface_info(ip_interface)
            if not interface_obj:
                continue

            ip_abbrev = self._get_ip_type_abbreviation(ip_id)

            # 使用配置驱动的方式注册IP FIFO
            for fifo_config in self.FIFO_REGISTRY_CONFIG["ip_fifos"]:
                self._register_ip_fifo_by_config(interface_obj, node_id, channel, ip_abbrev, fifo_config)

        # 注册节点上IP的channel buffer FIFOs
        for node_id, node in self.nodes.items():
            node_id_str = str(node_id)
            self._register_channel_buffer_fifos(node, node_id_str, channel)

    def _extract_ip_interface_info(self, ip_interface) -> tuple:
        """提取IP接口信息"""
        if isinstance(ip_interface, dict):
            node_id = str(ip_interface.get("node_id", "unknown"))
            interface_obj = ip_interface.get("interface")
        else:
            node_id = str(getattr(ip_interface, "node_id", "unknown"))
            interface_obj = ip_interface

        return interface_obj, node_id

    def _register_ip_fifo_by_config(self, interface_obj, node_id, channel, ip_abbrev, fifo_config):
        """根据配置注册IP FIFO"""
        attr_path = fifo_config["attr_path"]
        name_prefix = fifo_config["name_prefix"]

        if hasattr(interface_obj, attr_path.replace("_fifos", "_fifos")):
            fifos = getattr(interface_obj, attr_path)
            if channel in fifos:
                fifo = fifos[channel]
                if hasattr(fifo, "name") and hasattr(fifo, "stats"):
                    simplified_name = f"{channel}_{name_prefix}_{ip_abbrev}"
                    self.fifo_stats_collector.register_fifo(fifo, node_id=node_id, simplified_name=simplified_name)

    def _register_channel_buffer_fifos(self, node, node_id_str, channel):
        """注册channel buffer FIFOs"""
        for buffer_config in self.FIFO_REGISTRY_CONFIG["channel_buffers"]:
            attr_path = buffer_config["attr_path"]
            name_prefix = buffer_config["name_prefix"]

            if hasattr(node, attr_path):
                buffer_dict = getattr(node, attr_path)
                for ip_id, channels in buffer_dict.items():
                    if isinstance(channels, dict) and channel in channels:
                        fifo = channels[channel]
                        if hasattr(fifo, "name") and hasattr(fifo, "stats"):
                            ip_abbrev = self._get_ip_type_abbreviation(ip_id)
                            simplified_name = f"{channel}_{name_prefix}_{ip_abbrev}"
                            self.fifo_stats_collector.register_fifo(fifo, node_id=node_id_str, simplified_name=simplified_name)

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

    # ========== 可视化相关方法 ==========

    def _initialize_visualization(self):
        """初始化可视化组件"""
        if self._visualization_initialized:
            return

        try:
            from src.noc.visualization.link_state_visualizer import LinkStateVisualizer

            # 创建可视化器
            self._realtime_visualizer = LinkStateVisualizer(config=self.config, model=self)

            # 显示可视化窗口（现在使用Dash，不需要matplotlib的交互模式）
            self._realtime_visualizer.show()

            # print(f"🎪 可视化窗口已打开 (周期 {self.cycle})")
            self._visualization_initialized = True

        except ImportError as e:
            print(f"❌ 无法启用可视化: 缺少依赖库 {e}")
            self._visualization_enabled = False
        except Exception as e:
            print(f"❌ 可视化初始化失败: {e}")
            import traceback

            self._visualization_enabled = False
            traceback.print_exc()

    def _update_visualization(self):
        """更新可视化显示"""
        if not self._realtime_visualizer or not self._visualization_enabled:
            return

        try:
            # 更新可视化器状态
            self._realtime_visualizer.update(self, cycle=self.cycle)

            # 使用time.sleep控制帧率，替代matplotlib的pause
            if self._visualization_enabled and self._visualization_frame_interval > 0:
                import time
                time.sleep(self._visualization_frame_interval)

        except KeyboardInterrupt:
            # 捕获Ctrl+C或其他键盘中断
            print("🔑 检测到键盘中断，触发可视化清理...")
            self.cleanup_visualization()
        except Exception as e:
            print(f"⚠️  可视化更新失败 (周期 {self.cycle}): {e}")
            # 出错时也触发清理，避免卡住
            self.cleanup_visualization()

    def __del__(self):
        """析构函数"""

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

    def _collect_and_export_link_statistics(self, save_dir: str, timestamp: int = None) -> str:
        """收集所有链路的带宽统计数据并导出CSV文件

        Returns:
            导出的CSV文件路径
        """
        try:
            import csv
            from pathlib import Path

            # 确保保存目录存在
            Path(save_dir).mkdir(parents=True, exist_ok=True)

            # CSV文件路径（使用传入的时间戳或生成新的）
            if timestamp is None:
                timestamp = int(time.time())
            csv_file_path = os.path.join(save_dir, f"link_bandwidth_{timestamp}.csv")

            # 收集所有链路的统计数据
            all_link_stats = []

            # 遍历模型中的所有链路
            if hasattr(self, "links") and self.links:
                for link in self.links.values():
                    if hasattr(link, "get_link_performance_metrics"):
                        # 获取链路性能指标
                        metrics = link.get_link_performance_metrics()

                        # 只为data通道创建数据行（不需要req和rsp通道的统计）
                        for channel, channel_metrics in metrics.items():
                            if channel == "data":  # 只保存data通道的统计
                                row_data = {
                                    "link_id": link.link_id,
                                    "source_node": link.source_node,
                                    "dest_node": link.dest_node,
                                    "direction": link.direction.value if hasattr(link, "direction") else "unknown",
                                    "channel": channel,
                                    "total_cycles": link.bandwidth_tracker.total_cycles,
                                    "bandwidth_gbps": channel_metrics["bandwidth_gbps"],
                                    "utilization": channel_metrics["utilization"],
                                    "idle_rate": channel_metrics["idle_rate"],
                                    "valid_slots": channel_metrics["valid_slots"],
                                    "empty_slots": channel_metrics["empty_slots"],
                                    "T0_slots": link.bandwidth_tracker.cycle_stats[channel]["T0"],
                                    "T1_slots": link.bandwidth_tracker.cycle_stats[channel]["T1"],
                                    "T2_slots": link.bandwidth_tracker.cycle_stats[channel]["T2"],
                                    "ITag_slots": link.bandwidth_tracker.cycle_stats[channel]["ITag"],
                                    "total_bytes": channel_metrics["total_bytes"],
                                }
                                all_link_stats.append(row_data)

            # 写入CSV文件
            if all_link_stats:
                fieldnames = [
                    "link_id",
                    "source_node",
                    "dest_node",
                    "direction",
                    "channel",
                    "total_cycles",
                    "bandwidth_gbps",
                    "utilization",
                    "idle_rate",
                    "valid_slots",
                    "empty_slots",
                    "T0_slots",
                    "T1_slots",
                    "T2_slots",
                    "ITag_slots",
                    "total_bytes",
                ]

                with open(csv_file_path, "w", newline="", encoding="utf-8") as csvfile:
                    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                    writer.writeheader()
                    writer.writerows(all_link_stats)

                # 输出信息统一在结果总结中显示，这里不重复输出

                # 打印链路带宽汇总
                # self._print_link_bandwidth_summary(all_link_stats)

        except Exception as e:
            print(f"ERROR: 导出链路统计数据失败: {e}")
            import traceback

    def run_simulation(
        self, max_time_ns: float = 5000.0, stats_start_time_ns: float = 0.0, progress_interval_ns: float = 1000.0, results_analysis: bool = False, verbose: bool = True
    ) -> Dict[str, Any]:
        """
        运行完整仿真（CrossRing版本，集成可视化控制）

        Args:
            max_time_ns: 最大仿真时间（纳秒）
            stats_start_time_ns: 统计开始时间（纳秒）
            progress_interval_ns: 进度显示间隔（纳秒）
            results_analysis: 是否在仿真结束后执行结果分析
            verbose: 是否打印详细的模型信息和中间结果

        Returns:
            仿真结果字典
        """
        # 获取网络频率进行ns到cycle的转换
        network_freq = 1.0  # 默认1GHz
        if hasattr(self.config, "basic_config") and hasattr(self.config.basic_config, "NETWORK_FREQUENCY"):
            network_freq = self.config.basic_config.NETWORK_FREQUENCY
        elif hasattr(self.config, "NETWORK_FREQUENCY"):
            network_freq = self.config.NETWORK_FREQUENCY
        elif hasattr(self.config, "clock_frequency"):
            network_freq = self.config.clock_frequency

        # ns转换为cycle：cycle = time_ns * frequency_GHz
        max_cycles = int(max_time_ns * network_freq)
        stats_start_cycle = int(stats_start_time_ns * network_freq)
        progress_interval = int(progress_interval_ns * network_freq)

        cycle_time_ns = 1.0 / network_freq  # 1个周期的纳秒数

        # 如果启用详细模式，打印traffic统计信息
        if verbose and hasattr(self, "traffic_scheduler") and self.traffic_scheduler:
            self._print_traffic_statistics()

        # 初始化可视化（如果已配置）
        if self._visualization_enabled and self.cycle >= self._visualization_start_cycle:
            self._initialize_visualization()

        self.is_running = True
        self.start_time = time.time()

        try:
            for cycle in range(1, max_cycles + 1):
                self.step()

                # 启用统计收集
                if cycle == stats_start_cycle:
                    self._reset_statistics()

                # 可视化更新（在指定周期后开始）
                if self._visualization_enabled and cycle >= self._visualization_start_cycle:
                    if cycle % self._visualization_update_interval == 0:  # 每N个周期更新一次
                        self._update_visualization()

                    # 如果可视化被用户退出，_visualization_enabled会被设为False
                    # 这时仿真继续但不再有延迟

                # 检查仿真结束条件（总是检查）
                if self._should_stop_simulation():
                    break

                # 定期输出进度
                if cycle % progress_interval == 0 and cycle > 0:
                    if verbose:
                        self._print_simulation_progress(cycle, max_cycles, progress_interval)
                    else:
                        active_requests = self.get_total_active_requests()
                        completed_requests = 0
                        if hasattr(self, "request_tracker") and self.request_tracker:
                            completed_requests = len(self.request_tracker.completed_requests)

                        # 计算时间（ns）
                        current_time_ns = cycle * cycle_time_ns

        except KeyboardInterrupt:
            print("🛑 仿真中断...")
            self.cleanup_visualization()  # 清理可视化资源
            self.user_interrupted = True
            # 不重新抛出异常，继续执行结果分析
        except Exception as e:
            self.cleanup_visualization()  # 出错时也清理可视化
            raise

        finally:
            self.is_running = False
            self.is_finished = True
            self.end_time = time.time()

            # 确保可视化资源被清理
            if self._visualization_enabled:
                self.cleanup_visualization()

        # 生成仿真结果
        results = self._generate_simulation_results(stats_start_cycle)

        # 如果启用详细模式，打印最终统计信息
        if verbose:
            self._print_final_statistics()

        # 结果分析（如果启用）
        if results_analysis and hasattr(self, "analyze_simulation_results"):
            try:
                analysis_results = self.analyze_simulation_results(results, enable_visualization=True, save_results=True, verbose=verbose)
                results["analysis"] = analysis_results
            except Exception as e:
                print(f"结果分析过程中出错: {e}")

        return results
