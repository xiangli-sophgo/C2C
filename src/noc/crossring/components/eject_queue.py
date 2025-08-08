"""
CrossRing弹出队列管理。

负责处理：
- 弹出输入FIFO管理
- IP弹出缓冲区管理
- 弹出仲裁逻辑
- IP分发决策
"""

from typing import Dict, List, Optional, Tuple

from src.noc.base.ip_interface import PipelinedFIFO
from ..flit import CrossRingFlit
from ..config import CrossRingConfig, RoutingStrategy


class EjectQueue:
    """弹出队列管理类。"""

    def __init__(self, node_id: int, coordinates: Tuple[int, int], config: CrossRingConfig):
        """
        初始化弹出队列管理器。

        Args:
            node_id: 节点ID
            coordinates: 节点坐标
            config: CrossRing配置
        """
        self.node_id = node_id
        self.coordinates = coordinates
        self.config = config
        self.parent_node = None  # 将在节点初始化时设置
        self.current_cycle = 0  # 统一的周期管理

        # 获取FIFO配置
        self.eq_in_depth = config.fifo_config.EQ_IN_FIFO_DEPTH
        self.eq_ch_depth = config.fifo_config.EQ_CH_DEPTH


        # 连接的IP列表
        self.connected_ips = []

        # 每个IP的eject channel_buffer
        self.ip_eject_channel_buffers = {}

        # eject输入FIFO
        self.eject_input_fifos = self._create_eject_input_fifos()

        # Eject轮询仲裁器状态
        self.eject_arbitration_state = {
            "req": {"current_source": 0, "current_ip": 0, "sources": [], "last_served_source": {}, "last_served_ip": {}},
            "rsp": {"current_source": 0, "current_ip": 0, "sources": [], "last_served_source": {}, "last_served_ip": {}},
            "data": {"current_source": 0, "current_ip": 0, "sources": [], "last_served_source": {}, "last_served_ip": {}},
        }

        # 性能统计
        self.stats = {"ejected_flits": {"req": 0, "rsp": 0, "data": 0}}

    def _create_eject_input_fifos(self) -> Dict[str, Dict[str, PipelinedFIFO]]:
        """创建eject输入FIFO集合。"""
        result = {}
        for channel in ["req", "rsp", "data"]:
            result[channel] = {}
            for direction in ["TU", "TD", "TR", "TL"]:
                result[channel][direction] = self._create_fifo(
                    f"eject_in_{channel}_{direction}_{self.node_id}", self.eq_in_depth
                )
        return result

    def connect_ip(self, ip_id: str) -> bool:
        """
        连接一个IP到当前节点。

        Args:
            ip_id: IP标识符

        Returns:
            是否成功连接
        """
        if ip_id not in self.connected_ips:
            self.connected_ips.append(ip_id)

            # 为这个IP创建eject channel_buffer
            self.ip_eject_channel_buffers[ip_id] = {}
            for channel in ["req", "rsp", "data"]:
                self.ip_eject_channel_buffers[ip_id][channel] = self._create_fifo(
                    f"ip_eject_channel_{channel}_{ip_id}_{self.node_id}", self.eq_ch_depth
                )

            # 更新eject仲裁状态中的IP列表
            for channel in ["req", "rsp", "data"]:
                arb_state = self.eject_arbitration_state[channel]
                arb_state["current_ip"] = 0
                arb_state["last_served_ip"] = {ip_id: 0 for ip_id in self.connected_ips}
            return True
        else:
            return False



    def _get_active_eject_sources(self) -> List[str]:
        """根据路由策略获取活跃的eject输入源。"""
        routing_strategy = getattr(self.config, "ROUTING_STRATEGY", "XY")
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


    def compute_arbitration(self, cycle: int, inject_input_fifos: Dict, ring_bridge) -> None:
        """
        计算阶段：确定要传输的flit但不执行传输。

        Args:
            cycle: 当前周期
            inject_input_fifos: 注入方向FIFO
            ring_bridge: RingBridge实例
        """
        # 首先初始化源列表（如果还没有初始化）
        if not self.eject_arbitration_state["req"]["sources"]:
            active_sources = self._get_active_eject_sources()
            for channel in ["req", "rsp", "data"]:
                arb_state = self.eject_arbitration_state[channel]
                arb_state["sources"] = active_sources.copy()
                arb_state["last_served_source"] = {source: 0 for source in active_sources}

        # 存储传输计划
        self._eject_transfer_plan = []

        # 为每个通道计算eject仲裁
        for channel in ["req", "rsp", "data"]:
            self._compute_channel_eject_arbitration(channel, cycle, inject_input_fifos, ring_bridge)

    def _compute_channel_eject_arbitration(self, channel: str, cycle: int, inject_input_fifos: Dict, ring_bridge) -> None:
        """计算单个通道的eject仲裁，支持不同IP的并行传输。"""
        if not self.connected_ips:
            return

        arb_state = self.eject_arbitration_state[channel]
        sources = arb_state["sources"]
        
        # 记录每个IP是否已被占用（同一IP只能接收一个flit）
        ip_used = set()
        first_transfer_source_idx = None  # 记录第一个成功传输的源索引

        # 轮询所有输入源
        for source_attempt in range(len(sources)):
            current_source_idx = (arb_state["current_source"] + source_attempt) % len(sources)
            source = sources[current_source_idx]

            # 获取来自当前源的flit (使用peek，不实际读取)
            flit = self._peek_flit_from_eject_source(source, channel, inject_input_fifos, ring_bridge)
            if flit is not None:
                # 找到flit，现在确定分配给哪个IP
                target_ip = self._find_target_ip_for_flit(flit, channel, cycle)
                if target_ip and target_ip not in ip_used:
                    # 保存传输计划
                    self._eject_transfer_plan.append((source, channel, flit, target_ip))
                    arb_state["last_served_source"][source] = cycle
                    ip_used.add(target_ip)  # 标记该IP已被占用
                    
                    # 记录第一个成功传输的源索引
                    if first_transfer_source_idx is None:
                        first_transfer_source_idx = current_source_idx
                    
                    # 继续检查其他源（不break），允许不同IP的并行传输

        # 更新起始源索引，确保下次从不同源开始
        if first_transfer_source_idx is not None:
            arb_state["current_source"] = (first_transfer_source_idx + 1) % len(sources)
        else:
            # 没有成功传输，也要更新索引确保轮询
            arb_state["current_source"] = (arb_state["current_source"] + 1) % len(sources)

    def execute_arbitration(self, cycle: int, inject_input_fifos: Dict, ring_bridge) -> None:
        """
        执行阶段：基于compute阶段的计算执行实际传输。

        Args:
            cycle: 当前周期
            inject_input_fifos: 注入方向FIFO
            ring_bridge: RingBridge实例
        """
        if not hasattr(self, "_eject_transfer_plan"):
            return

        # 执行所有计划的传输
        for source, channel, flit, target_ip in self._eject_transfer_plan:
            # 从源获取flit（实际取出）
            actual_flit = self._get_flit_from_eject_source(source, channel, inject_input_fifos, ring_bridge)
            if actual_flit:
                if self._assign_flit_to_ip(actual_flit, target_ip, channel, cycle):
                    # 成功传输，更新统计
                    self.stats["ejected_flits"][channel] += 1

    def _peek_flit_from_eject_source(self, source: str, channel: str, inject_input_fifos: Dict, ring_bridge) -> Optional[CrossRingFlit]:
        """从指定的eject源查看flit（不实际读取）。"""
        if source == "IQ_EQ":
            # 直接从inject_input_fifos的EQ查看
            eq_fifo = inject_input_fifos[channel]["EQ"]
            if eq_fifo.valid_signal():
                return eq_fifo.peek_output()

        elif source == "ring_bridge_EQ":
            # 从ring_bridge的EQ输出查看
            # ring_bridge没有peek方法，使用get方法但需要小心
            if ring_bridge and hasattr(ring_bridge, "ring_bridge_output_fifos"):
                eq_fifo = ring_bridge.ring_bridge_output_fifos[channel]["EQ"]
                if eq_fifo.valid_signal():
                    return eq_fifo.peek_output()
            return None

        elif source in ["TU", "TD", "TR", "TL"]:
            # 从eject_input_fifos查看
            input_fifo = self.eject_input_fifos[channel][source]
            fifo_id = id(input_fifo)
            is_valid = input_fifo.valid_signal()
            fifo_len = len(input_fifo.internal_queue)
            output_valid = input_fifo.output_valid
            queue_len = len(input_fifo.internal_queue)
            has_output_reg = input_fifo.output_register is not None
            read_this_cycle = input_fifo.read_this_cycle
            if is_valid:
                return input_fifo.peek_output()

        return None

    def _get_flit_from_eject_source(self, source: str, channel: str, inject_input_fifos: Dict, ring_bridge) -> Optional[CrossRingFlit]:
        """从指定的eject源获取flit。"""
        if source == "IQ_EQ":
            # 直接从inject_input_fifos的EQ获取
            eq_fifo = inject_input_fifos[channel]["EQ"]
            if eq_fifo.valid_signal():
                return eq_fifo.read_output()

        elif source == "ring_bridge_EQ":
            # 从ring_bridge的EQ输出获取
            return ring_bridge.get_eq_output_flit(channel)

        elif source in ["TU", "TD", "TR", "TL"]:
            # 从eject_input_fifos获取
            input_fifo = self.eject_input_fifos[channel][source]
            if input_fifo.valid_signal():
                flit = input_fifo.read_output()
                # 通知entry释放
                if flit and self.parent_node:
                    self._notify_entry_release(flit, channel, source)
                return flit

        return None

    def _notify_entry_release(self, flit, channel: str, direction: str) -> None:
        """通知entry释放"""
        if hasattr(flit, "allocated_entry_info") and flit.allocated_entry_info:
            alloc_info = flit.allocated_entry_info
            alloc_direction = alloc_info.get("direction")
            alloc_priority = alloc_info.get("priority")
            
            if alloc_direction and alloc_priority:
                # 根据方向找到对应的CrossPoint
                crosspoint = None
                if direction in ["TU", "TD"]:
                    crosspoint = self.parent_node.vertical_crosspoint
                elif direction in ["TL", "TR"]:
                    crosspoint = self.parent_node.horizontal_crosspoint
                
                if crosspoint and hasattr(crosspoint, 'etag_entry_managers'):
                    if channel in crosspoint.etag_entry_managers and alloc_direction in crosspoint.etag_entry_managers[channel]:
                        entry_manager = crosspoint.etag_entry_managers[channel][alloc_direction]
                        if entry_manager.release_entry(alloc_priority):
                            crosspoint.stats["entry_releases"][channel][alloc_priority] += 1

    def _find_target_ip_for_flit(self, flit: CrossRingFlit, channel: str, cycle: int) -> Optional[str]:
        """为flit找到目标IP。"""
        if not self.connected_ips:
            return None

        # 首先尝试根据flit的destination_type匹配对应的IP
        if hasattr(flit, "destination_type") and flit.destination_type:
            # 完全匹配优先级最高
            for ip_id in self.connected_ips:
                if ip_id == flit.destination_type:
                    eject_buffer = self.ip_eject_channel_buffers[ip_id][channel]
                    if eject_buffer.ready_signal():
                        return ip_id
            
            # 如果完全匹配的IP buffer不ready，等待而不是fallback到其他IP
            # 这可以避免响应被错误路由到同类型的其他IP
            for ip_id in self.connected_ips:
                if ip_id == flit.destination_type:
                    return None  # 目标IP存在但buffer不ready，等待
            
            # 如果完全匹配的IP不存在，才考虑基础类型匹配（用于兼容性）
            dest_base_type = flit.destination_type.split("_")[0]
            for ip_id in self.connected_ips:
                ip_base_type = ip_id.split("_")[0]
                if ip_base_type == dest_base_type:
                    eject_buffer = self.ip_eject_channel_buffers[ip_id][channel]
                    if eject_buffer.ready_signal():
                        return ip_id

        # 如果没有匹配的IP，使用round-robin逻辑
        arb_state = self.eject_arbitration_state[channel]
        for ip_attempt in range(len(self.connected_ips)):
            current_ip_idx = arb_state["current_ip"]
            ip_id = self.connected_ips[current_ip_idx]

            eject_buffer = self.ip_eject_channel_buffers[ip_id][channel]
            if eject_buffer.ready_signal():
                arb_state["current_ip"] = (current_ip_idx + 1) % len(self.connected_ips)
                return ip_id

            arb_state["current_ip"] = (current_ip_idx + 1) % len(self.connected_ips)

        return None

    def _assign_flit_to_ip(self, flit: CrossRingFlit, ip_id: str, channel: str, cycle: int) -> bool:
        """将flit分配给指定IP。"""
        eject_buffer = self.ip_eject_channel_buffers[ip_id][channel]
        if eject_buffer.write_input(flit):
            # 更新flit状态
            flit.flit_position = "EQ_CH"

            # 更新IP仲裁状态
            arb_state = self.eject_arbitration_state[channel]
            arb_state["last_served_ip"][ip_id] = cycle

            # 释放E-Tag entry（如果flit有allocated_entry_info）
            if hasattr(flit, "allocated_entry_info") and self.parent_node:
                entry_info = flit.allocated_entry_info
                direction = entry_info["direction"]
                priority = entry_info["priority"]

                # 找到对应的CrossPoint和entry管理器
                if direction in ["TR", "TL"]:
                    crosspoint = self.parent_node.horizontal_crosspoint
                else:  # TU, TD
                    crosspoint = self.parent_node.vertical_crosspoint

                if direction in crosspoint.etag_entry_managers:
                    entry_manager = crosspoint.etag_entry_managers[direction]
                    if entry_manager.release_entry(priority):
                        crosspoint.stats["entry_releases"][channel][priority] += 1
                        # 可选：打印调试信息
                        # print(f"🔓 EQ释放entry: 节点{self.node_id} 方向{direction} {priority}级entry")

                # 清除flit的entry信息（已经释放）
                delattr(flit, "allocated_entry_info")

            return True
        else:
            return False

    def get_eject_flit(self, ip_id: str, channel: str) -> Optional[CrossRingFlit]:
        """IP从其eject channel buffer获取flit。"""
        if ip_id not in self.connected_ips:
            raise ValueError(f"IP {ip_id}未连接到节点{self.node_id}")

        eject_buffer = self.ip_eject_channel_buffers[ip_id][channel]
        if eject_buffer.valid_signal():
            return eject_buffer.read_output()
        return None

    def _create_fifo(self, name: str, depth: int) -> PipelinedFIFO:
        """创建FIFO的辅助方法。"""
        fifo = PipelinedFIFO(name, depth=depth)
        fifo._stats_sample_interval = self.config.basic_config.FIFO_STATS_SAMPLE_INTERVAL
        return fifo

    def _step_all_fifos(self, method_name: str, cycle: int = None) -> None:
        """对所有FIFO执行指定方法的辅助函数。"""
        # 更新IP eject channel buffers
        for ip_id in self.connected_ips:
            for channel in ["req", "rsp", "data"]:
                fifo = self.ip_eject_channel_buffers[ip_id][channel]
                if cycle is not None:
                    getattr(fifo, method_name)(cycle)
                else:
                    getattr(fifo, method_name)()

        # 更新eject input FIFOs
        for channel in ["req", "rsp", "data"]:
            for direction in ["TU", "TD", "TR", "TL"]:
                fifo = self.eject_input_fifos[channel][direction]
                if cycle is not None:
                    getattr(fifo, method_name)(cycle)
                else:
                    getattr(fifo, method_name)()

    def step_compute_phase(self, cycle: int) -> None:
        """FIFO组合逻辑更新。"""
        self.current_cycle = cycle  # 更新当前周期
        self._step_all_fifos("step_compute_phase", cycle)

    def step_update_phase(self, cycle: int = 0) -> None:
        """FIFO时序逻辑更新。"""
        # 不需要再次更新current_cycle，已经在compute阶段更新过
        self._step_all_fifos("step_update_phase")


    def get_stats(self) -> Dict:
        """获取统计信息。"""
        return {
            "ejected_flits": dict(self.stats["ejected_flits"]),
            "buffer_occupancy": {
                "ip_eject_channel_buffers": {ip_id: {k: len(v) for k, v in channels.items()} for ip_id, channels in self.ip_eject_channel_buffers.items()},
                "eject_input_fifos": {k: {d: len(v) for d, v in vv.items()} for k, vv in self.eject_input_fifos.items()},
            },
        }
