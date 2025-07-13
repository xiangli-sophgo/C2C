"""
CrossRing节点实现。

提供CrossRing网络中节点的详细实现，包括：
- 注入/提取队列管理
- 环形缓冲区管理
- 拥塞控制机制
- 仲裁逻辑
"""

from typing import Dict, List, Any, Tuple, Optional
import logging
from enum import Enum
from dataclasses import dataclass

from src.noc.base.node import BaseNoCNode
from src.noc.base.ip_interface import PipelinedFIFO, FlowControlledTransfer
from ..base.link import PriorityLevel
from .flit import CrossRingFlit
from .config import CrossRingConfig, RoutingStrategy
from .crossring_link import CrossRingSlot, RingSlice  # 导入新的类
from .cross_point import CrossRingCrossPoint, CrossPointDirection





class CrossRingNode:
    """
    CrossRing节点类。

    实现CrossRing节点的内部结构和逻辑，包括：
    1. 注入/提取队列管理
    2. 环形缓冲区管理
    3. ETag/ITag拥塞控制
    4. 仲裁逻辑
    """

    def _create_directional_fifos(self, prefix: str, directions: List[str], depth: int) -> Dict[str, Dict[str, PipelinedFIFO]]:
        """工厂方法：创建方向化FIFO集合减少代码重复"""
        return {
            channel: {
                direction: PipelinedFIFO(f"{prefix}_{channel}_{direction}_{self.node_id}", depth=depth)
                for direction in directions
            }
            for channel in ["req", "rsp", "data"]
        }

    def __init__(self, node_id: int, coordinates: Tuple[int, int], config: CrossRingConfig, logger: logging.Logger):
        """
        初始化CrossRing节点

        Args:
            node_id: 节点ID
            coordinates: 节点坐标 (x, y)
            config: CrossRing配置
            logger: 日志记录器
        """
        self.node_id = node_id
        self.coordinates = coordinates
        self.config = config
        self.logger = logger

        # IP注入缓冲区配置
        # 获取FIFO配置，如果没有则使用默认值
        iq_ch_depth = getattr(config, "iq_ch_depth", 10)
        iq_out_depth = getattr(config, "iq_out_depth", 8)

        # 连接的IP列表（默认每个节点连接一个IP，也可以扩展为多个）
        self.connected_ips = []  # 将存储连接的IP ID列表

        # 每个IP的inject channel_buffer - 结构：ip_inject_channel_buffers[ip_id][channel]
        self.ip_inject_channel_buffers = {}

        # 方向化的注入队列 - 使用工厂方法减少重复代码
        self.inject_direction_fifos = self._create_directional_fifos("inject", ["TR", "TL", "TU", "TD", "EQ"], iq_out_depth)
        # 获取eject相关的FIFO配置
        eq_in_depth = getattr(config, "eq_in_depth", 16)
        eq_ch_depth = getattr(config, "eq_ch_depth", 10)

        # 获取ring_bridge相关的FIFO配置
        rb_in_depth = getattr(config, "rb_in_depth", 16)
        rb_out_depth = getattr(config, "rb_out_depth", 8)

        # 每个IP的eject channel_buffer - 结构：ip_eject_channel_buffers[ip_id][channel]
        self.ip_eject_channel_buffers = {}

        # ring buffer输入的中间FIFO - 使用工厂方法
        self.eject_input_fifos = self._create_directional_fifos("eject_in", ["TU", "TD", "TR", "TL"], eq_in_depth)

        # ring_bridge输入FIFO - 使用工厂方法
        self.ring_bridge_input_fifos = self._create_directional_fifos("ring_bridge_in", ["TR", "TL", "TU", "TD"], rb_in_depth)

        # ring_bridge输出FIFO
        self.ring_bridge_output_fifos = self._create_directional_fifos("ring_bridge_out", ["EQ", "TR", "TL", "TU", "TD"], rb_out_depth)

        # 拥塞控制状态
        self.etag_status = {
            "horizontal": {"req": False, "rsp": False, "data": False},
            "vertical": {"req": False, "rsp": False, "data": False},
        }
        self.itag_status = {
            "horizontal": {"req": False, "rsp": False, "data": False},
            "vertical": {"req": False, "rsp": False, "data": False},
        }

        # 仲裁状态 - 使用更准确的方向优先级
        self.arbitration_state = {
            "horizontal_priority": "inject",  # inject, ring_tr, ring_tl
            "vertical_priority": "inject",  # inject, ring_td, ring_tu
            "last_arbitration": {"horizontal": 0, "vertical": 0},
        }

        # 注入轮询仲裁器状态 - 为每个通道独立的轮询仲裁
        self.inject_arbitration_state = {
            "req": {
                "current_direction": 0,  # 当前轮询位置：0=TR, 1=TL, 2=TU, 3=TD, 4=EQ
                "directions": ["TR", "TL", "TU", "TD", "EQ"],
                "last_served": {"TR": 0, "TL": 0, "TU": 0, "TD": 0, "EQ": 0},
            },
            "rsp": {
                "current_direction": 0,
                "directions": ["TR", "TL", "TU", "TD", "EQ"],
                "last_served": {"TR": 0, "TL": 0, "TU": 0, "TD": 0, "EQ": 0},
            },
            "data": {
                "current_direction": 0,
                "directions": ["TR", "TL", "TU", "TD", "EQ"],
                "last_served": {"TR": 0, "TL": 0, "TU": 0, "TD": 0, "EQ": 0},
            },
        }

        # Eject轮询仲裁器状态 - 为每个通道独立的轮询仲裁
        self.eject_arbitration_state = {
            "req": {
                "current_source": 0,  # 当前输入源位置
                "current_ip": 0,  # 当前IP位置
                "sources": [],  # 动态根据路由策略设置
                "last_served_source": {},
                "last_served_ip": {},
            },
            "rsp": {
                "current_source": 0,
                "current_ip": 0,
                "sources": [],
                "last_served_source": {},
                "last_served_ip": {},
            },
            "data": {
                "current_source": 0,
                "current_ip": 0,
                "sources": [],
                "last_served_source": {},
                "last_served_ip": {},
            },
        }

        # Ring_bridge轮询仲裁器状态 - 为每个通道独立的轮询仲裁
        self.ring_bridge_arbitration_state = {
            "req": {
                "current_input": 0,  # 当前输入源位置
                "current_output": 0,  # 当前输出方向位置
                "input_sources": [],  # 动态根据路由策略设置
                "output_directions": [],  # 动态根据路由策略设置
                "last_served_input": {},
                "last_served_output": {},
            },
            "rsp": {
                "current_input": 0,
                "current_output": 0,
                "input_sources": [],
                "output_directions": [],
                "last_served_input": {},
                "last_served_output": {},
            },
            "data": {
                "current_input": 0,
                "current_output": 0,
                "input_sources": [],
                "output_directions": [],
                "last_served_input": {},
                "last_served_output": {},
            },
        }

        # 性能统计
        self.stats = {
            "injected_flits": {"req": 0, "rsp": 0, "data": 0},
            "ejected_flits": {"req": 0, "rsp": 0, "data": 0},
            "transferred_flits": {"horizontal": 0, "vertical": 0},
            "congestion_events": 0,
        }


        # 存储FIFO配置供后续使用
        self.iq_ch_depth = iq_ch_depth
        self.iq_out_depth = iq_out_depth
        self.eq_in_depth = eq_in_depth
        self.eq_ch_depth = eq_ch_depth
        self.rb_in_depth = rb_in_depth
        self.rb_out_depth = rb_out_depth

        # 初始化CrossPoint实例 - 每个节点有2个CrossPoint
        self.horizontal_crosspoint = CrossRingCrossPoint(
            crosspoint_id=f"node_{node_id}_horizontal",
            node_id=node_id,
            direction=CrossPointDirection.HORIZONTAL,
            config=config,
            coordinates=coordinates,
            parent_node=self,
            logger=logger,
        )

        self.vertical_crosspoint = CrossRingCrossPoint(
            crosspoint_id=f"node_{node_id}_vertical",
            node_id=node_id,
            direction=CrossPointDirection.VERTICAL,
            config=config,
            coordinates=coordinates,
            parent_node=self,
            logger=logger,
        )

        self.logger.debug(f"CrossRing节点初始化: ID={node_id}, 坐标={coordinates}")

    def set_routing_strategy_bias(self, routing_strategy: RoutingStrategy) -> None:
        """
        根据路由策略设置仲裁偏向

        Args:
            routing_strategy: 路由策略
        """
        if routing_strategy == RoutingStrategy.XY:
            # XY路由：稍微偏向水平方向
            self.routing_bias = {"horizontal": 1.2, "vertical": 1.0}
        elif routing_strategy == RoutingStrategy.YX:
            # YX路由：稍微偏向垂直方向
            self.routing_bias = {"horizontal": 1.0, "vertical": 1.2}
        else:
            # 其他策略：均衡
            self.routing_bias = {"horizontal": 1.0, "vertical": 1.0}

        self.logger.debug(f"节点{self.node_id}设置路由偏向: {routing_strategy.value} -> {self.routing_bias}")

    def connect_ip(self, ip_id: str) -> bool:
        """
        连接一个IP到当前节点

        Args:
            ip_id: IP的标识符

        Returns:
            是否成功连接
        """
        if ip_id not in self.connected_ips:
            self.connected_ips.append(ip_id)

            # 为这个IP创建inject channel_buffer
            self.ip_inject_channel_buffers[ip_id] = {
                "req": PipelinedFIFO(f"ip_inject_channel_req_{ip_id}_{self.node_id}", depth=self.iq_ch_depth),
                "rsp": PipelinedFIFO(f"ip_inject_channel_rsp_{ip_id}_{self.node_id}", depth=self.iq_ch_depth),
                "data": PipelinedFIFO(f"ip_inject_channel_data_{ip_id}_{self.node_id}", depth=self.iq_ch_depth),
            }

            # 为这个IP创建eject channel_buffer
            self.ip_eject_channel_buffers[ip_id] = {
                "req": PipelinedFIFO(f"ip_eject_channel_req_{ip_id}_{self.node_id}", depth=self.eq_ch_depth),
                "rsp": PipelinedFIFO(f"ip_eject_channel_rsp_{ip_id}_{self.node_id}", depth=self.eq_ch_depth),
                "data": PipelinedFIFO(f"ip_eject_channel_data_{ip_id}_{self.node_id}", depth=self.eq_ch_depth),
            }

            # 更新eject仲裁状态中的IP列表
            self._update_eject_arbitration_ips()

            self.logger.debug(f"节点{self.node_id}成功连接IP {ip_id}")
            return True
        else:
            self.logger.warning(f"IP {ip_id}已经连接到节点{self.node_id}")
            return False

    def disconnect_ip(self, ip_id: str) -> None:
        """
        断开IP连接

        Args:
            ip_id: IP的标识符
        """
        if ip_id in self.connected_ips:
            self.connected_ips.remove(ip_id)
            del self.ip_inject_channel_buffers[ip_id]
            del self.ip_eject_channel_buffers[ip_id]

            # 更新eject仲裁状态中的IP列表
            self._update_eject_arbitration_ips()

            self.logger.debug(f"节点{self.node_id}断开IP {ip_id}连接")
        else:
            self.logger.warning(f"IP {ip_id}未连接到节点{self.node_id}")

    def get_connected_ips(self) -> List[str]:
        """
        获取连接的IP列表

        Returns:
            连接的IP ID列表
        """
        return self.connected_ips.copy()

    def get_crosspoint(self, direction: str) -> Optional[CrossRingCrossPoint]:
        """
        获取指定方向的CrossPoint

        Args:
            direction: 方向 ("horizontal" 或 "vertical")

        Returns:
            CrossPoint实例，如果不存在则返回None
        """
        if direction == "horizontal":
            return self.horizontal_crosspoint
        elif direction == "vertical":
            return self.vertical_crosspoint
        else:
            return None

    def step_crosspoints(self, cycle: int) -> None:
        """
        执行一个周期的CrossPoint处理

        Args:
            cycle: 当前周期
        """
        # 执行水平CrossPoint处理
        if self.horizontal_crosspoint:
            self.horizontal_crosspoint.step(cycle, self.inject_direction_fifos, self.eject_input_fifos)

        # 执行垂直CrossPoint处理
        if self.vertical_crosspoint:
            self.vertical_crosspoint.step(cycle, self.inject_direction_fifos, self.eject_input_fifos)

    def _get_ring_bridge_config(self) -> Tuple[List[str], List[str]]:
        """
        根据路由策略获取ring_bridge的输入源和输出方向配置

        Returns:
            (输入源列表, 输出方向列表)
        """
        # 获取路由策略
        routing_strategy = getattr(self.config, "routing_strategy", "XY")
        if hasattr(routing_strategy, "value"):
            routing_strategy = routing_strategy.value

        # 根据路由策略配置输入源和输出方向
        if routing_strategy == "XY":
            # XY路由：主要是水平环flit进入ring_bridge，但也要处理可能的垂直环输入
            input_sources = ["IQ_TU", "IQ_TD", "RB_TR", "RB_TL", "RB_TU", "RB_TD"]  # 支持所有输入
            output_directions = ["EQ", "TU", "TD"]  # 垂直环输出
        elif routing_strategy == "YX":
            input_sources = ["IQ_TR", "IQ_TL", "RB_TU", "RB_TD"]
            output_directions = ["EQ", "TR", "TL"]
        else:  # ADAPTIVE 或其他
            input_sources = ["IQ_TU", "IQ_TD", "IQ_TR", "IQ_TL", "RB_TR", "RB_TL", "RB_TU", "RB_TD"]
            output_directions = ["EQ", "TU", "TD", "TR", "TL"]

        return input_sources, output_directions

    def _initialize_ring_bridge_arbitration(self) -> None:
        """初始化ring_bridge仲裁的源和方向列表"""
        input_sources, output_directions = self._get_ring_bridge_config()

        for channel in ["req", "rsp", "data"]:
            arb_state = self.ring_bridge_arbitration_state[channel]
            arb_state["input_sources"] = input_sources.copy()
            arb_state["output_directions"] = output_directions.copy()
            arb_state["last_served_input"] = {source: 0 for source in input_sources}
            arb_state["last_served_output"] = {direction: 0 for direction in output_directions}

    def process_ring_bridge_arbitration(self, cycle: int) -> None:
        """
        处理ring_bridge的轮询仲裁

        Args:
            cycle: 当前周期
        """
        # 首先初始化源和方向列表（如果还没有初始化）
        if not self.ring_bridge_arbitration_state["req"]["input_sources"]:
            self._initialize_ring_bridge_arbitration()

        # 为每个通道处理ring_bridge仲裁
        for channel in ["req", "rsp", "data"]:
            # 只检查req通道的ring_bridge输入FIFO
            if channel == "req":
                has_input = False
                for direction in ["TR", "TL", "TU", "TD"]:
                    rb_fifo = self.ring_bridge_input_fifos[channel][direction]

            self._process_channel_ring_bridge_arbitration(channel, cycle)

    def _process_channel_ring_bridge_arbitration(self, channel: str, cycle: int) -> None:
        """
        处理单个通道的ring_bridge仲裁

        Args:
            channel: 通道类型
            cycle: 当前周期
        """
        arb_state = self.ring_bridge_arbitration_state[channel]
        input_sources = arb_state["input_sources"]

        # 轮询所有输入源
        for input_attempt in range(len(input_sources)):
            current_input_idx = arb_state["current_input"]
            input_source = input_sources[current_input_idx]

            # 获取来自当前输入源的flit
            flit = self._get_flit_from_ring_bridge_input(input_source, channel)
            if flit is not None:
                # 找到flit，现在确定输出方向并分配
                output_direction = self._determine_ring_bridge_output_direction(flit)
                if self._assign_flit_to_ring_bridge_output(flit, output_direction, channel, cycle):
                    # 成功分配，更新输入仲裁状态
                    arb_state["last_served_input"][input_source] = cycle
                    break
                else:
                    print(f"❌ 节点{self.node_id}: flit {flit.packet_id} 分配到ring_bridge输出{output_direction}失败")
            else:
                pass

            # 移动到下一个输入源
            arb_state["current_input"] = (current_input_idx + 1) % len(input_sources)

    def _get_flit_from_ring_bridge_input(self, input_source: str, channel: str) -> Optional[CrossRingFlit]:
        """
        从指定的ring_bridge输入源获取flit

        Args:
            input_source: 输入源名称 (如 "IQ_TU", "RB_TR")
            channel: 通道类型

        Returns:
            获取的flit，如果没有则返回None
        """
        if input_source.startswith("IQ_"):
            # 直接从inject_direction_fifos获取
            direction = input_source[3:]  # 去掉"IQ_"前缀
            iq_fifo = self.inject_direction_fifos[channel][direction]
            if iq_fifo.valid_signal():
                return iq_fifo.read_output()

        elif input_source.startswith("RB_"):
            # 从ring_bridge_input_fifos获取
            direction = input_source[3:]  # 去掉"RB_"前缀
            rb_fifo = self.ring_bridge_input_fifos[channel][direction]
            # 修复：检查实际队列内容而不仅仅依赖valid_signal
            if rb_fifo.valid_signal() or len(rb_fifo.internal_queue) > 0:
                if rb_fifo.valid_signal():
                    return rb_fifo.read_output()
                else:
                    # 直接从内部队列获取（修复FIFO状态不一致问题）
                    if len(rb_fifo.internal_queue) > 0:
                        return rb_fifo.internal_queue.popleft()

        return None

    def _determine_ring_bridge_output_direction(self, flit: CrossRingFlit) -> str:
        """
        确定flit在ring_bridge中的输出方向

        Args:
            flit: 要路由的flit

        Returns:
            输出方向 ("EQ", "TR", "TL", "TU", "TD")
        """
        # 首先检查是否是本地目标
        if self._is_local_destination(flit):
            return "EQ"

        # 否则，根据路由策略和目标位置确定输出方向
        return self._calculate_routing_direction(flit)

    def _assign_flit_to_ring_bridge_output(self, flit: CrossRingFlit, output_direction: str, channel: str, cycle: int) -> bool:
        """
        将flit分配到ring_bridge输出FIFO

        Args:
            flit: 要分配的flit
            output_direction: 输出方向
            channel: 通道类型
            cycle: 当前周期

        Returns:
            是否成功分配
        """
        # 检查输出FIFO是否可用
        output_fifo = self.ring_bridge_output_fifos[channel][output_direction]
        if output_fifo.ready_signal():
            # 更新flit的ring_bridge位置信息
            flit.rb_fifo_name = f"RB_{output_direction}"
            flit.flit_position = f"RB_{output_direction}"  # 同时更新flit_position

            if output_fifo.write_input(flit):
                # 成功分配，更新输出仲裁状态
                arb_state = self.ring_bridge_arbitration_state[channel]
                arb_state["last_served_output"][output_direction] = cycle

                self.logger.debug(f"节点{self.node_id}成功将{channel}通道flit分配到ring_bridge输出{output_direction}")
                return True

        return False

    def add_to_ring_bridge_input(self, flit: CrossRingFlit, direction: str, channel: str) -> bool:
        """
        CrossPoint向ring_bridge输入添加flit

        Args:
            flit: 要添加的flit
            direction: 方向 ("TR", "TL", "TU", "TD")
            channel: 通道类型

        Returns:
            是否成功添加
        """
        input_fifo = self.ring_bridge_input_fifos[channel][direction]
        if input_fifo.ready_signal():
            success = input_fifo.write_input(flit)
            if success:
                self.logger.debug(f"节点{self.node_id}成功添加flit到ring_bridge输入{direction}_{channel}")
            return success
        else:
            self.logger.debug(f"节点{self.node_id}的ring_bridge输入{direction}_{channel}已满")
            return False

    def get_ring_bridge_eq_flit(self, channel: str) -> Optional[CrossRingFlit]:
        """
        从ring_bridge的EQ输出获取flit (为eject队列提供)

        Args:
            channel: 通道类型

        Returns:
            获取的flit，如果没有则返回None
        """
        eq_fifo = self.ring_bridge_output_fifos[channel]["EQ"]
        if eq_fifo.valid_signal():
            return eq_fifo.read_output()
        return None

    def get_ring_bridge_output_flit(self, direction: str, channel: str) -> Optional[CrossRingFlit]:
        """
        从ring_bridge的指定方向输出获取flit

        Args:
            direction: 输出方向 ("TR", "TL", "TU", "TD")
            channel: 通道类型

        Returns:
            获取的flit，如果没有则返回None
        """
        output_fifo = self.ring_bridge_output_fifos[channel][direction]
        if output_fifo.valid_signal():
            return output_fifo.read_output()
        return None

    def _update_eject_arbitration_ips(self) -> None:
        """更新eject仲裁状态中的IP列表"""
        for channel in ["req", "rsp", "data"]:
            arb_state = self.eject_arbitration_state[channel]
            # 重置IP相关的仲裁状态
            arb_state["current_ip"] = 0
            arb_state["last_served_ip"] = {ip_id: 0 for ip_id in self.connected_ips}

    def _get_active_eject_sources(self) -> List[str]:
        """
        根据路由策略获取活跃的eject输入源

        Returns:
            输入源列表
        """
        # 获取路由策略
        routing_strategy = getattr(self.config, "routing_strategy", "XY")
        if hasattr(routing_strategy, "value"):
            routing_strategy = routing_strategy.value

        # 这两个源总是存在
        sources = ["IQ_EQ", "ring_bridge_EQ"]

        if routing_strategy == "XY":
            sources.extend(["TU", "TD"])
        elif routing_strategy == "YX":
            sources.extend(["TR", "TL"])
        else:  # ADAPTIVE 或其他
            sources.extend(["TU", "TD", "TR", "TL"])

        return sources

    def _initialize_eject_arbitration_sources(self) -> None:
        """初始化eject仲裁的源列表"""
        active_sources = self._get_active_eject_sources()

        for channel in ["req", "rsp", "data"]:
            arb_state = self.eject_arbitration_state[channel]
            arb_state["sources"] = active_sources.copy()
            arb_state["last_served_source"] = {source: 0 for source in active_sources}

    def process_eject_arbitration(self, cycle: int) -> None:
        """
        处理eject队列的轮询仲裁

        Args:
            cycle: 当前周期
        """
        # 首先初始化源列表（如果还没有初始化）
        if not self.eject_arbitration_state["req"]["sources"]:
            self._initialize_eject_arbitration_sources()

        # 为每个通道处理eject仲裁
        for channel in ["req", "rsp", "data"]:
            self._process_channel_eject_arbitration(channel, cycle)

    def _process_channel_eject_arbitration(self, channel: str, cycle: int) -> None:
        """
        处理单个通道的eject仲裁

        Args:
            channel: 通道类型
            cycle: 当前周期
        """
        if not self.connected_ips:
            return  # 没有连接的IP

        arb_state = self.eject_arbitration_state[channel]
        sources = arb_state["sources"]

        # 轮询所有输入源
        for source_attempt in range(len(sources)):
            current_source_idx = arb_state["current_source"]
            source = sources[current_source_idx]

            # 获取来自当前源的flit
            flit = self._get_flit_from_eject_source(source, channel)
            if flit is not None:
                # 找到flit，现在轮询分配给IP
                if self._assign_flit_to_ip(flit, channel, cycle):
                    # 成功分配，更新源仲裁状态
                    arb_state["last_served_source"][source] = cycle
                    break

            # 移动到下一个源
            arb_state["current_source"] = (current_source_idx + 1) % len(sources)

    def _get_flit_from_eject_source(self, source: str, channel: str) -> Optional[CrossRingFlit]:
        """
        从指定的eject源获取flit

        Args:
            source: 输入源名称
            channel: 通道类型

        Returns:
            获取的flit，如果没有则返回None
        """
        if source == "IQ_EQ":
            # 直接从inject_direction_fifos的EQ获取
            eq_fifo = self.inject_direction_fifos[channel]["EQ"]
            if eq_fifo.valid_signal():
                return eq_fifo.read_output()

        elif source == "ring_bridge_EQ":
            # 从ring_bridge的EQ输出获取
            return self.get_ring_bridge_eq_flit(channel)

        elif source in ["TU", "TD", "TR", "TL"]:
            # 从eject_input_fifos获取
            input_fifo = self.eject_input_fifos[channel][source]
            if input_fifo.valid_signal():
                return input_fifo.read_output()

        return None

    def _assign_flit_to_ip(self, flit: CrossRingFlit, channel: str, cycle: int) -> bool:
        """
        将flit分配给IP

        Args:
            flit: 要分配的flit
            channel: 通道类型
            cycle: 当前周期

        Returns:
            是否成功分配
        """
        if not self.connected_ips:
            return False

        # 首先尝试根据flit的destination_type匹配对应的IP
        if hasattr(flit, "destination_type") and flit.destination_type:
            target_ips = []

            for ip_id in self.connected_ips:
                # 从IP ID中提取IP类型（例如：ddr_0_node1 -> ddr_0）
                ip_type = "_".join(ip_id.split("_")[:-1])  # 去掉最后的_nodeX部分
                ip_base_type = ip_type.split("_")[0]  # 获取基础类型（例如：ddr）
                
                # 从destination_type中提取基础类型（例如：l2m_2 -> l2m）
                dest_base_type = flit.destination_type.split("_")[0]

                # 修复匹配逻辑：支持多种匹配方式
                # 1. 精确匹配：ip_type == destination_type (例如：l2m_0 == l2m_0)
                # 2. 基础类型匹配：ip_base_type == dest_base_type (例如：l2m == l2m)
                if ip_type == flit.destination_type or ip_base_type == dest_base_type:
                    target_ips.append(ip_id)

            # 如果找到匹配的IP，优先使用它们
            if target_ips:
                for ip_id in target_ips:
                    eject_buffer = self.ip_eject_channel_buffers[ip_id][channel]
                    if eject_buffer.ready_signal():
                        if eject_buffer.write_input(flit):
                            # 成功分配，更新统计
                            self.stats["ejected_flits"][channel] += 1

                            # 更新flit状态
                            flit.flit_position = "EQ_CH"

                            self.logger.debug(f"节点{self.node_id}成功将{channel}通道flit分配给匹配的IP {ip_id} (destination_type={flit.destination_type})")
                            return True

                # 如果匹配的IP都不可用，记录警告
                self.logger.warning(f"节点{self.node_id}: 匹配的IP类型 {flit.destination_type} 都不可用，flit {flit.packet_id} 将被丢弃")
                return False

        # 如果没有destination_type或找不到匹配的IP，使用原来的round-robin逻辑
        arb_state = self.eject_arbitration_state[channel]

        # 轮询所有连接的IP
        for ip_attempt in range(len(self.connected_ips)):
            current_ip_idx = arb_state["current_ip"]
            ip_id = self.connected_ips[current_ip_idx]

            # 检查IP的eject channel buffer是否可用
            eject_buffer = self.ip_eject_channel_buffers[ip_id][channel]
            if eject_buffer.ready_signal():
                # 分配给这个IP
                if eject_buffer.write_input(flit):
                    # 成功分配，更新IP仲裁状态
                    arb_state["last_served_ip"][ip_id] = cycle
                    arb_state["current_ip"] = (current_ip_idx + 1) % len(self.connected_ips)

                    # 更新统计
                    self.stats["ejected_flits"][channel] += 1

                    # 更新flit状态
                    flit.flit_position = "EQ_CH"

                    self.logger.debug(f"节点{self.node_id}成功将{channel}通道flit分配给IP {ip_id}")
                    return True

            # 移动到下一个IP
            arb_state["current_ip"] = (current_ip_idx + 1) % len(self.connected_ips)

        return False

    def get_eject_flit(self, ip_id: str, channel: str) -> Optional[CrossRingFlit]:
        """
        IP从其eject channel buffer获取flit

        Args:
            ip_id: IP标识符
            channel: 通道类型

        Returns:
            获取的flit，如果没有则返回None
        """
        if ip_id not in self.connected_ips:
            self.logger.error(f"IP {ip_id}未连接到节点{self.node_id}")
            return None

        eject_buffer = self.ip_eject_channel_buffers[ip_id][channel]
        if eject_buffer.valid_signal():
            return eject_buffer.read_output()

        return None

    def step_compute_phase(self, cycle: int) -> None:
        """计算阶段：准备数据传输但不执行"""
        # 更新所有FIFO的组合逻辑阶段
        self._step_compute_phase()

        self._compute_inject_arbitration(cycle)

        # 处理CrossPoint的计算阶段
        if hasattr(self.horizontal_crosspoint, "step_compute_phase"):
            self.horizontal_crosspoint.step_compute_phase(cycle, self.inject_direction_fifos, self.eject_input_fifos)
        if hasattr(self.vertical_crosspoint, "step_compute_phase"):
            self.vertical_crosspoint.step_compute_phase(cycle, self.inject_direction_fifos, self.eject_input_fifos)

    def step_update_phase(self, cycle: int) -> None:
        """更新阶段：执行实际的数据传输 - 优化版本，FIFO状态已在预更新阶段处理"""
        # 步骤1：Node执行注入仲裁（从channel_buffer读取并写入inject_direction_fifos）
        # channel_buffer.valid_signal()已在预更新阶段反映了最新数据
        self._execute_inject_arbitration(cycle)

        # 步骤2：CrossPoint执行（从inject_direction_fifos读取数据）
        # 这样CrossPoint能读取到当前周期刚写入的数据，减少1拍延迟
        if hasattr(self.horizontal_crosspoint, "step_update_phase"):
            self.horizontal_crosspoint.step_update_phase(cycle, self.inject_direction_fifos, self.eject_input_fifos)
        else:
            self.horizontal_crosspoint.step(cycle, self.inject_direction_fifos, self.eject_input_fifos)

        if hasattr(self.vertical_crosspoint, "step_update_phase"):
            self.vertical_crosspoint.step_update_phase(cycle, self.inject_direction_fifos, self.eject_input_fifos)
        else:
            self.vertical_crosspoint.step(cycle, self.inject_direction_fifos, self.eject_input_fifos)

        # 处理ring_bridge的轮询仲裁
        self.process_ring_bridge_arbitration(cycle)

        # 处理eject队列的轮询仲裁
        self.process_eject_arbitration(cycle)

        # 更新仲裁状态
        self._update_arbitration_state(cycle)

        # 更新拥塞控制状态
        self._update_congestion_state()

    def _step_compute_phase(self) -> None:
        """更新所有FIFO的组合逻辑阶段"""
        # 更新IP inject channel buffers
        for ip_id in self.connected_ips:
            for channel in ["req", "rsp", "data"]:
                self.ip_inject_channel_buffers[ip_id][channel].step_compute_phase()
                self.ip_eject_channel_buffers[ip_id][channel].step_compute_phase()

        # 更新inject direction FIFOs
        for channel in ["req", "rsp", "data"]:
            for direction in ["TR", "TL", "TU", "TD", "EQ"]:
                self.inject_direction_fifos[channel][direction].step_compute_phase()

        # 更新eject input FIFOs
        for channel in ["req", "rsp", "data"]:
            for direction in ["TU", "TD", "TR", "TL"]:
                self.eject_input_fifos[channel][direction].step_compute_phase()

        # 更新ring_bridge input/output FIFOs
        for channel in ["req", "rsp", "data"]:
            for direction in ["TR", "TL", "TU", "TD"]:
                self.ring_bridge_input_fifos[channel][direction].step_compute_phase()
            for direction in ["EQ", "TR", "TL", "TU", "TD"]:
                self.ring_bridge_output_fifos[channel][direction].step_compute_phase()

    def _step_update_phase(self) -> None:
        """更新所有FIFO的时序逻辑阶段"""
        # 更新IP inject channel buffers
        for ip_id in self.connected_ips:
            for channel in ["req", "rsp", "data"]:
                self.ip_inject_channel_buffers[ip_id][channel].step_update_phase()
                self.ip_eject_channel_buffers[ip_id][channel].step_update_phase()

        # 更新inject direction FIFOs
        for channel in ["req", "rsp", "data"]:
            for direction in ["TR", "TL", "TU", "TD", "EQ"]:
                self.inject_direction_fifos[channel][direction].step_update_phase()

        # 更新eject input FIFOs
        for channel in ["req", "rsp", "data"]:
            for direction in ["TU", "TD", "TR", "TL"]:
                self.eject_input_fifos[channel][direction].step_update_phase()

        # 更新ring_bridge input/output FIFOs
        for channel in ["req", "rsp", "data"]:
            for direction in ["TR", "TL", "TU", "TD"]:
                self.ring_bridge_input_fifos[channel][direction].step_update_phase()
            for direction in ["EQ", "TR", "TL", "TU", "TD"]:
                self.ring_bridge_output_fifos[channel][direction].step_update_phase()

    def _update_arbitration_state(self, cycle: int) -> None:
        """
        更新仲裁状态

        Args:
            cycle: 当前周期
        """
        # 检查是否需要重置仲裁优先级
        for direction in ["horizontal", "vertical"]:
            last_arbitration = self.arbitration_state["last_arbitration"][direction]
            if cycle - last_arbitration > self.config.arbitration_timeout:
                # 重置为默认优先级
                self.arbitration_state[f"{direction}_priority"] = "inject"
                self.logger.debug(f"节点{self.node_id}的{direction}仲裁状态重置为默认")

    def _update_congestion_state(self) -> None:
        """更新拥塞控制状态"""
        # 更新ETag状态
        for direction in ["horizontal", "vertical"]:
            for channel in ["req", "rsp", "data"]:
                # 检查eject input fifos的拥塞情况
                eject_congestion = False
                eject_threshold = self.eq_in_depth * 0.8

                for eject_dir in ["TR", "TL", "TD", "TU"]:
                    eject_fifo = self.eject_input_fifos[channel][eject_dir]
                    buffer_occupancy = len(eject_fifo.internal_queue)
                    if buffer_occupancy >= eject_threshold:
                        eject_congestion = True
                        break

                ring_congestion = False

                # 设置ETag状态
                old_status = self.etag_status[direction][channel]
                new_status = eject_congestion or ring_congestion

                if old_status != new_status:
                    self.etag_status[direction][channel] = new_status
                    if new_status:
                        self.stats["congestion_events"] += 1
                        self.logger.debug(f"节点{self.node_id}的{direction} {channel} ETag状态变为拥塞")
                    else:
                        self.logger.debug(f"节点{self.node_id}的{direction} {channel} ETag状态变为畅通")

    def can_inject_flit(self, channel: str, direction: str) -> bool:
        """
        检查是否可以注入flit

        Args:
            channel: 通道类型 ("req", "rsp", "data")
            direction: 注入方向 ("horizontal", "vertical")

        Returns:
            是否可以注入
        """
        # 检查拥塞状态
        if self.etag_status[direction][channel]:
            return False

        # 检查仲裁状态
        if self.arbitration_state[f"{direction}_priority"] != "inject":
            return False

        return True

    def add_to_inject_queue(self, flit: CrossRingFlit, channel: str, ip_id: str) -> bool:
        """
        特定IP注入flit到其对应的channel_buffer

        Args:
            flit: 要添加的flit
            channel: 通道类型 ("req", "rsp", "data")
            ip_id: IP标识符

        Returns:
            是否成功添加
        """
        # 检查IP是否已连接
        if ip_id not in self.connected_ips:
            self.logger.error(f"IP {ip_id}未连接到节点{self.node_id}")
            return False

        # 获取对应IP的inject channel_buffer
        channel_buffer = self.ip_inject_channel_buffers[ip_id][channel]
        if not channel_buffer.ready_signal():
            self.logger.debug(f"节点{self.node_id}的IP {ip_id} {channel}通道缓冲区已满，无法注入flit")
            return False

        success = channel_buffer.write_input(flit)
        if success:
            self.logger.debug(f"节点{self.node_id}的IP {ip_id}成功注入flit到{channel}通道缓冲区")
        return success

    def _compute_inject_arbitration(self, cycle: int) -> None:
        """
        计算阶段：确定要传输的flit但不执行传输

        Args:
            cycle: 当前周期
        """
        # 初始化传输计划
        if not hasattr(self, "_inject_transfer_plan"):
            self._inject_transfer_plan = []
        self._inject_transfer_plan.clear()

        # 为每个连接的IP和每个通道类型计算仲裁
        for ip_id in self.connected_ips:
            for channel in ["req", "rsp", "data"]:
                # 检查IP的inject channel_buffer是否有数据
                if ip_id not in self.ip_inject_channel_buffers:
                    continue

                channel_buffer = self.ip_inject_channel_buffers[ip_id][channel]
                if not channel_buffer.valid_signal():
                    continue  # 没有数据可传输

                # 获取flit并计算路由方向
                flit = channel_buffer.peek_output()
                if flit is None:
                    continue

                # 计算正确的路由方向
                correct_direction = self._calculate_routing_direction(flit)
                if correct_direction == "INVALID":
                    continue

                # 检查目标inject_direction_fifo是否有空间
                target_fifo = self.inject_direction_fifos[channel][correct_direction]
                if target_fifo.ready_signal():
                    # 规划传输：(ip_id, channel, flit, direction)
                    self._inject_transfer_plan.append((ip_id, channel, flit, correct_direction))

    def _execute_inject_arbitration(self, cycle: int) -> None:
        """
        执行阶段：基于compute阶段的计算执行实际传输

        Args:
            cycle: 当前周期
        """
        if not hasattr(self, "_inject_transfer_plan"):
            return

        # 执行所有计划的传输
        for ip_id, channel, flit, direction in self._inject_transfer_plan:
            # 从channel_buffer读取flit
            channel_buffer = self.ip_inject_channel_buffers[ip_id][channel]
            actual_flit = channel_buffer.read_output()

            # 写入目标inject_direction_fifo
            target_fifo = self.inject_direction_fifos[channel][direction]
            if actual_flit and target_fifo.write_input(actual_flit):
                # 更新flit位置状态
                actual_flit.flit_position = f"IQ_{direction}"
                actual_flit.current_node_id = self.node_id

                # 添加调试信息

                # 更新仲裁状态
                arb_state = self.inject_arbitration_state[channel]
                arb_state["last_served"][direction] = cycle

    def _process_ip_channel_inject_arbitration(self, ip_id: str, channel: str, cycle: int) -> None:
        """
        处理特定IP和通道的注入仲裁

        Args:
            ip_id: IP标识符
            channel: 通道类型
            cycle: 当前周期
        """
        # 检查IP的inject channel_buffer是否有数据
        if ip_id not in self.ip_inject_channel_buffers:
            self.logger.warning(f"节点{self.node_id}: IP {ip_id} 的channel_buffer不存在")
            return

        channel_buffer = self.ip_inject_channel_buffers[ip_id][channel]

        if not channel_buffer.valid_signal():
            return  # 静默处理空buffer

        # 获取当前仲裁状态
        arb_state = self.inject_arbitration_state[channel]

        # 首先peek flit来确定正确的路由方向
        flit = channel_buffer.peek_output()
        if flit is None:
            self.logger.warning(f"节点{self.node_id}: peek_output返回None")
            return

        # 计算正确的路由方向
        correct_direction = self._calculate_routing_direction(flit)

        # Debug路由决策
        if hasattr(flit, "dest_coordinates"):
            dest_x, dest_y = flit.dest_coordinates
            curr_x, curr_y = self.coordinates
            debug_key = f"route_{self.node_id}_{dest_x}_{dest_y}"
            if not hasattr(flit, "_route_debug_count"):
                flit._route_debug_count = {}
            if debug_key not in flit._route_debug_count:
                flit._route_debug_count[debug_key] = 0
            flit._route_debug_count[debug_key] += 1

            # 只显示前几次或异常循环情况
            if flit._route_debug_count[debug_key] <= 2 or flit._route_debug_count[debug_key] % 5 == 0:
                print(f"🧭 节点{self.node_id}({curr_x},{curr_y}) → 目标({dest_x},{dest_y}): 路由方向={correct_direction} [第{flit._route_debug_count[debug_key]}次]")

        # 检查正确方向的FIFO是否可用
        target_fifo = self.inject_direction_fifos[channel][correct_direction]

        if target_fifo.ready_signal():
            # 现在读取并传输flit
            flit = channel_buffer.read_output()

            if flit is not None and target_fifo.write_input(flit):
                # 更新flit位置状态
                flit.flit_position = f"IQ_{correct_direction}"
                flit.current_node_id = self.node_id

                # 添加调试信息
                print(f"🔄 周期{cycle}: channel_buffer->IQ_{correct_direction}: {flit.packet_id}")
                flit.current_position = self.node_id

                # 成功传输，更新仲裁状态
                arb_state["last_served"][correct_direction] = cycle
                print(f"🎉 节点{self.node_id}: 成功将flit {flit.packet_id}仲裁到{correct_direction}方向")
                self.logger.info(f"节点{self.node_id}成功将IP {ip_id} {channel}通道flit仲裁到{correct_direction}方向")
            else:
                self.logger.error(f"节点{self.node_id}: flit读取或写入失败")

    def _should_route_to_direction(self, flit: CrossRingFlit, direction: str) -> bool:
        """
        判断flit是否应该路由到指定方向

        Args:
            flit: 要判断的flit
            direction: 目标方向

        Returns:
            是否应该路由到该方向
        """
        # 如果是EQ方向，检查是否是本地节点
        if direction == "EQ":
            return self._is_local_destination(flit)

        # 对于其他方向，根据路由算法决定
        return self._calculate_routing_direction(flit) == direction

    def _is_local_destination(self, flit: CrossRingFlit) -> bool:
        """
        检查flit是否应该在本地弹出

        Args:
            flit: 要检查的flit

        Returns:
            是否是本地目标
        """
        if hasattr(flit, "destination") and flit.destination == self.node_id:
            return True
        if hasattr(flit, "dest_node_id") and flit.dest_node_id == self.node_id:
            return True
        if hasattr(flit, "dest_coordinates"):
            dest_x, dest_y = flit.dest_coordinates
            curr_x, curr_y = self.coordinates
            if dest_x == curr_x and dest_y == curr_y:
                return True
        return False

    def _calculate_routing_direction(self, flit: CrossRingFlit) -> str:
        """
        根据配置的路由策略计算flit的路由方向
        整合了原来的_apply_routing_strategy和_adaptive_routing_decision函数

        Args:
            flit: 要路由的flit

        Returns:
            路由方向（"TR", "TL", "TU", "TD", "EQ"）
        """
        # 获取目标坐标
        if hasattr(flit, "dest_coordinates"):
            dest_x, dest_y = flit.dest_coordinates
        elif hasattr(flit, "dest_xid") and hasattr(flit, "dest_yid"):
            dest_x, dest_y = flit.dest_xid, flit.dest_yid
        else:
            # 如果没有坐标信息，尝试从destination计算
            num_col = getattr(self.config, "num_col", 3)
            dest_x = flit.destination % num_col
            dest_y = flit.destination // num_col

        curr_x, curr_y = self.coordinates

        # 如果已经到达目标位置
        if dest_x == curr_x and dest_y == curr_y:
            return "EQ"  # 本地

        # 获取路由策略，默认为XY
        routing_strategy = getattr(self.config, "routing_strategy", "XY")
        if hasattr(routing_strategy, "value"):
            routing_strategy = routing_strategy.value

        # 计算移动需求
        need_horizontal = dest_x != curr_x
        need_vertical = dest_y != curr_y

        # 应用路由策略
        if routing_strategy == "XY":
            # XY路由：先水平后垂直
            if need_horizontal:
                return "TR" if dest_x > curr_x else "TL"
            elif need_vertical:
                return "TD" if dest_y > curr_y else "TU"

        elif routing_strategy == "YX":
            # YX路由：先垂直后水平
            if need_vertical:
                return "TD" if dest_y > curr_y else "TU"
            elif need_horizontal:
                return "TR" if dest_x > curr_x else "TL"

        elif routing_strategy == "ADAPTIVE":
            # 自适应路由：根据拥塞状态选择路径
            if need_horizontal and need_vertical:
                # 需要两个维度的移动，选择拥塞较少的维度
                horizontal_congested = self._is_direction_congested("horizontal")
                vertical_congested = self._is_direction_congested("vertical")

                # 根据拥塞情况选择优先维度
                if horizontal_congested and not vertical_congested:
                    # 水平拥塞，优先垂直
                    return "TD" if dest_y > curr_y else "TU"
                elif vertical_congested and not horizontal_congested:
                    # 垂直拥塞，优先水平
                    return "TR" if dest_x > curr_x else "TL"
                else:
                    # 都不拥塞或都拥塞，默认XY路由
                    return "TR" if dest_x > curr_x else "TL"
            elif need_horizontal:
                return "TR" if dest_x > curr_x else "TL"
            elif need_vertical:
                return "TD" if dest_y > curr_y else "TU"
        else:
            # 未知策略，默认使用XY
            self.logger.warning(f"未知路由策略 {routing_strategy}，使用XY路由")
            if need_horizontal:
                return "TR" if dest_x > curr_x else "TL"
            elif need_vertical:
                return "TD" if dest_y > curr_y else "TU"

        return "EQ"  # 本地

    def _is_direction_congested(self, direction: str) -> bool:
        """
        检查指定方向是否拥塞

        Args:
            direction: "horizontal" 或 "vertical"

        Returns:
            是否拥塞
        """
        # 检查对应方向的ETag状态
        for channel in ["req", "rsp", "data"]:
            if self.etag_status[direction][channel]:
                return True
        return False

    def get_inject_direction_status(self) -> Dict[str, Any]:
        """
        获取注入方向队列的状态

        Returns:
            状态信息字典
        """
        status = {}
        for channel in ["req", "rsp", "data"]:
            status[channel] = {}
            for direction in ["TR", "TL", "TU", "TD", "EQ"]:
                fifo = self.inject_direction_fifos[channel][direction]
                status[channel][direction] = {
                    "occupancy": len(fifo),
                    "ready": fifo.ready_signal(),
                    "valid": fifo.valid_signal(),
                }
        return status

    def inject_flit_to_crosspoint(self, flit: CrossRingFlit, direction: str) -> bool:
        """
        将flit注入到指定方向的CrossPoint

        Args:
            flit: 要注入的flit
            direction: 注入方向 ("horizontal", "vertical")

        Returns:
            是否成功注入
        """
        if direction == "horizontal":
            return self.horizontal_crosspoint.try_inject_flit(flit, PriorityLevel.T2)
        elif direction == "vertical":
            return self.vertical_crosspoint.try_inject_flit(flit, PriorityLevel.T2)
        else:
            self.logger.error(f"未知的注入方向: {direction}")
            return False

    def eject_flit_from_crosspoint(self, direction: str, sub_direction: str, target_fifo_occupancy: int, target_fifo_depth: int) -> Optional[CrossRingFlit]:
        """
        从指定方向的CrossPoint下环flit

        Args:
            direction: CrossPoint方向 ("horizontal", "vertical")
            sub_direction: 子方向 ("TR", "TL", "TU", "TD")
            target_fifo_occupancy: 目标FIFO当前占用
            target_fifo_depth: 目标FIFO深度

        Returns:
            下环的flit，如果没有则返回None
        """
        crosspoint = None
        if direction == "horizontal":
            crosspoint = self.horizontal_crosspoint
        elif direction == "vertical":
            crosspoint = self.vertical_crosspoint

        if crosspoint is None:
            return None

        # 查找合适的slot进行下环
        for slot in crosspoint.ring_slots:
            if slot.valid and slot.flit is not None:
                # 检查是否是目标节点
                if self._should_eject_flit(slot.flit):
                    ejected_flit = crosspoint.try_eject_flit(slot, target_fifo_occupancy, target_fifo_depth, sub_direction)
                    if ejected_flit:
                        return ejected_flit

        return None

    def get_crosspoint_status(self) -> Dict[str, Any]:
        """
        获取CrossPoint状态信息

        Returns:
            CrossPoint状态字典
        """
        return {"horizontal": self.horizontal_crosspoint.get_crosspoint_status(), "vertical": self.vertical_crosspoint.get_crosspoint_status()}

    def get_stats(self) -> Dict[str, Any]:
        """
        获取节点统计信息

        Returns:
            统计信息字典
        """
        return {
            "node_id": self.node_id,
            "coordinates": self.coordinates,
            "injected_flits": dict(self.stats["injected_flits"]),
            "ejected_flits": dict(self.stats["ejected_flits"]),
            "transferred_flits": dict(self.stats["transferred_flits"]),
            "congestion_events": self.stats["congestion_events"],
            "buffer_occupancy": {
                "ip_inject_channel_buffers": {ip_id: {k: len(v) for k, v in channels.items()} for ip_id, channels in self.ip_inject_channel_buffers.items()},
                "inject_directions": {k: {d: len(v) for d, v in vv.items()} for k, vv in self.inject_direction_fifos.items()},
                "ip_eject_channel_buffers": {ip_id: {k: len(v) for k, v in channels.items()} for ip_id, channels in self.ip_eject_channel_buffers.items()},
                "eject_input_fifos": {k: {d: len(v) for d, v in vv.items()} for k, vv in self.eject_input_fifos.items()},
                "ring_bridge_input_fifos": {k: {d: len(v) for d, v in vv.items()} for k, vv in self.ring_bridge_input_fifos.items()},
                "ring_bridge_output_fifos": {k: {d: len(v) for d, v in vv.items()} for k, vv in self.ring_bridge_output_fifos.items()},
                # Ring buffers已移除，使用CrossRing架构中的实际缓冲区
                "crosspoints": {
                    "horizontal": self.crosspoints["horizontal"].get_debug_info() if self.crosspoints["horizontal"] else {},
                    "vertical": self.crosspoints["vertical"].get_debug_info() if self.crosspoints["vertical"] else {},
                },
            },
            "congestion_status": {
                "etag": {
                    "horizontal": dict(self.etag_status["horizontal"]),
                    "vertical": dict(self.etag_status["vertical"]),
                },
                "itag": {
                    "horizontal": dict(self.itag_status["horizontal"]),
                    "vertical": dict(self.itag_status["vertical"]),
                },
            },
            "crosspoint_status": self.get_crosspoint_status(),
        }
