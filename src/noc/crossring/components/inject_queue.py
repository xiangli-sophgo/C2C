"""
CrossRing注入队列管理。

负责处理：
- IP注入缓冲区管理
- 注入方向队列管理
- 注入仲裁逻辑
- 路由决策
"""

from typing import Dict, List, Optional, Tuple

from src.noc.base.ip_interface import PipelinedFIFO
from ..flit import CrossRingFlit
from ..config import CrossRingConfig, RoutingStrategy


class InjectQueue:
    """注入队列管理类。"""

    def __init__(self, node_id: int, coordinates: Tuple[int, int], config: CrossRingConfig, topology=None):
        """
        初始化注入队列管理器。

        Args:
            node_id: 节点ID
            coordinates: 节点坐标
            config: CrossRing配置
            topology: 拓扑对象（用于路由表查询）
        """
        self.node_id = node_id
        self.coordinates = coordinates
        self.config = config
        self.topology = topology

        # 获取FIFO配置
        self.iq_ch_depth = config.fifo_config.IQ_CH_DEPTH
        self.iq_out_depth = config.fifo_config.IQ_OUT_FIFO_DEPTH

        # 连接的IP列表
        self.connected_ips = []

        # 每个IP的inject channel_buffer
        self.ip_inject_channel_buffers = {}

        # 方向化的注入队列
        self.inject_input_fifos = self._create_direction_fifos()

        # 注入仲裁状态 - 添加IP级别的轮询
        self.inject_arbitration_state = {
            "current_ip_index": 0,  # 当前IP索引，用于IP级别轮询
            "channels": {
                "req": {"current_direction": 0, "directions": ["TR", "TL", "TU", "TD", "EQ"]},
                "rsp": {"current_direction": 0, "directions": ["TR", "TL", "TU", "TD", "EQ"]},
                "data": {"current_direction": 0, "directions": ["TR", "TL", "TU", "TD", "EQ"]},
            }
        }

        # 传输计划（两阶段执行用）
        self._inject_transfer_plan = []

    def _create_direction_fifos(self) -> Dict[str, Dict[str, PipelinedFIFO]]:
        """创建方向化FIFO集合。"""
        return {
            channel: {direction: PipelinedFIFO(f"inject_{channel}_{direction}_{self.node_id}", depth=self.iq_out_depth) for direction in ["TR", "TL", "TU", "TD", "EQ"]}
            for channel in ["req", "rsp", "data"]
        }

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

            # 为这个IP创建inject channel_buffer
            self.ip_inject_channel_buffers[ip_id] = {
                "req": PipelinedFIFO(f"ip_inject_channel_req_{ip_id}_{self.node_id}", depth=self.iq_ch_depth),
                "rsp": PipelinedFIFO(f"ip_inject_channel_rsp_{ip_id}_{self.node_id}", depth=self.iq_ch_depth),
                "data": PipelinedFIFO(f"ip_inject_channel_data_{ip_id}_{self.node_id}", depth=self.iq_ch_depth),
            }

            return True
        else:
            return False

    def disconnect_ip(self, ip_id: str) -> None:
        """断开IP连接。"""
        if ip_id in self.connected_ips:
            self.connected_ips.remove(ip_id)
            del self.ip_inject_channel_buffers[ip_id]

    def add_to_inject_queue(self, flit: CrossRingFlit, channel: str, ip_id: str) -> bool:
        """
        IP注入flit到其对应的channel_buffer。

        Args:
            flit: 要添加的flit
            channel: 通道类型
            ip_id: IP标识符

        Returns:
            是否成功添加
        """
        if ip_id not in self.connected_ips:
            raise ValueError(f"IP {ip_id}未连接到节点{self.node_id}")
            return False

        channel_buffer = self.ip_inject_channel_buffers[ip_id][channel]
        if not channel_buffer.ready_signal():
            return False

        success = channel_buffer.write_input(flit)
        if success:
            pass
        return success

    def compute_arbitration(self, cycle: int) -> None:
        """
        计算阶段：确定要传输的flit但不执行传输。
        使用轮询机制确保公平性。

        Args:
            cycle: 当前周期
        """
        self._inject_transfer_plan.clear()

        # 如果没有连接的IP，直接返回
        if not self.connected_ips:
            return

        # 限制每个周期处理的最大传输数，避免饥饿
        max_transfers_per_cycle = len(self.connected_ips) * 3  # 每个IP最多处理3个通道
        transfers_planned = 0

        # 从当前IP索引开始轮询
        start_ip_index = self.inject_arbitration_state["current_ip_index"]
        
        for ip_offset in range(len(self.connected_ips)):
            if transfers_planned >= max_transfers_per_cycle:
                break
                
            # 计算实际的IP索引
            ip_index = (start_ip_index + ip_offset) % len(self.connected_ips)
            ip_id = self.connected_ips[ip_index]
            
            if ip_id not in self.ip_inject_channel_buffers:
                continue

            # 对每个通道进行轮询
            channels = ["req", "rsp", "data"]
            for channel_offset in range(len(channels)):
                if transfers_planned >= max_transfers_per_cycle:
                    break
                    
                channel = channels[channel_offset]
                channel_buffer = self.ip_inject_channel_buffers[ip_id][channel]
                
                if not channel_buffer.valid_signal():
                    continue

                # 获取flit并计算路由方向
                flit = channel_buffer.peek_output()
                if flit is None:
                    continue

                # 计算正确的路由方向
                correct_direction = self._calculate_routing_direction(flit)
                if correct_direction == "INVALID":
                    continue

                # 检查目标inject_direction_fifo是否有空间
                target_fifo = self.inject_input_fifos[channel][correct_direction]
                if target_fifo.ready_signal():
                    # 规划传输
                    self._inject_transfer_plan.append((ip_id, channel, flit, correct_direction))
                    transfers_planned += 1
                    
                    # 成功规划传输后，继续处理其他通道
                    # 不要break，让每个通道都有机会传输

        # 每轮结束后更新IP索引，确保公平性
        self.inject_arbitration_state["current_ip_index"] = (start_ip_index + 1) % len(self.connected_ips)

    def execute_arbitration(self, cycle: int) -> None:
        """
        执行阶段：基于compute阶段的计算执行实际传输。

        Args:
            cycle: 当前周期
        """
        # 执行所有计划的传输
        for ip_id, channel, flit, direction in self._inject_transfer_plan:
            # 从channel_buffer读取flit
            channel_buffer = self.ip_inject_channel_buffers[ip_id][channel]
            actual_flit = channel_buffer.read_output()

            # 写入目标inject_direction_fifo
            target_fifo = self.inject_input_fifos[channel][direction]
            if actual_flit and target_fifo.write_input(actual_flit):
                # 更新flit位置状态
                actual_flit.flit_position = f"IQ_{direction}"
                actual_flit.current_node_id = self.node_id

                # 更新仲裁状态
                arb_state = self.inject_arbitration_state["channels"][channel]
                if "last_served" not in arb_state:
                    arb_state["last_served"] = {}
                arb_state["last_served"][direction] = cycle

    def step_compute_phase(self, cycle: int) -> None:
        """FIFO组合逻辑更新。"""
        # 更新IP inject channel buffers
        for ip_id in self.connected_ips:
            for channel in ["req", "rsp", "data"]:
                self.ip_inject_channel_buffers[ip_id][channel].step_compute_phase(cycle)

        # 更新inject direction FIFOs
        for channel in ["req", "rsp", "data"]:
            for direction in ["TR", "TL", "TU", "TD", "EQ"]:
                self.inject_input_fifos[channel][direction].step_compute_phase(cycle)

    def step_update_phase(self) -> None:
        """FIFO时序逻辑更新。"""
        # 更新IP inject channel buffers
        for ip_id in self.connected_ips:
            for channel in ["req", "rsp", "data"]:
                self.ip_inject_channel_buffers[ip_id][channel].step_update_phase()

        # 更新inject direction FIFOs
        for channel in ["req", "rsp", "data"]:
            for direction in ["TR", "TL", "TU", "TD", "EQ"]:
                self.inject_input_fifos[channel][direction].step_update_phase()

    def _calculate_routing_direction(self, flit: CrossRingFlit) -> str:
        """
        使用预计算的路由表获取flit的路由方向。

        Args:
            flit: 要路由的flit

        Returns:
            路由方向（"TR", "TL", "TU", "TD", "EQ"）
        """
        # 如果有topology对象，使用路由表
        if self.topology and hasattr(self.topology, "routing_table"):
            return self.topology.get_next_direction(self.node_id, flit.destination)

        # 回退到原始的路由计算方法
        return self._calculate_routing_direction_fallback(flit)

    def _calculate_routing_direction_fallback(self, flit: CrossRingFlit) -> str:
        """
        回退路由计算方法（当路由表不可用时）。

        Args:
            flit: 要路由的flit

        Returns:
            路由方向（"TR", "TL", "TU", "TD", "EQ"）
        """
        # 获取目标坐标
        if hasattr(flit, "dest_coordinates"):
            dest_col, dest_row = flit.dest_coordinates  # (x, y) -> (col, row)
        elif hasattr(flit, "dest_xid") and hasattr(flit, "dest_yid"):
            dest_col, dest_row = flit.dest_xid, flit.dest_yid
        else:
            # 从destination计算
            num_col = getattr(self.config, "NUM_COL", 3)
            dest_col = flit.destination % num_col
            dest_row = flit.destination // num_col

        curr_col, curr_row = self.coordinates  # self.coordinates是(x, y)格式，即(col, row)

        # 如果已经到达目标位置
        if dest_row == curr_row and dest_col == curr_col:
            return "EQ"

        # 获取路由策略
        routing_strategy = getattr(self.config, "ROUTING_STRATEGY", "XY")
        if hasattr(routing_strategy, "value"):
            routing_strategy = routing_strategy.value

        # 计算移动需求
        need_horizontal = dest_col != curr_col
        need_vertical = dest_row != curr_row

        # 应用路由策略
        if routing_strategy == "XY":
            if need_horizontal:
                return "TR" if dest_col > curr_col else "TL"
            elif need_vertical:
                return "TD" if dest_row > curr_row else "TU"
        elif routing_strategy == "YX":
            if need_vertical:
                return "TD" if dest_row > curr_row else "TU"
            elif need_horizontal:
                return "TR" if dest_col > curr_col else "TL"
        else:  # ADAPTIVE
            if need_horizontal and need_vertical:
                # 默认XY路由
                return "TR" if dest_col > curr_col else "TL"
            elif need_horizontal:
                return "TR" if dest_col > curr_col else "TL"
            elif need_vertical:
                return "TD" if dest_row > curr_row else "TU"

        return "EQ"

    def get_inject_direction_status(self) -> Dict:
        """获取注入方向队列的状态。"""
        status = {}
        for channel in ["req", "rsp", "data"]:
            status[channel] = {}
            for direction in ["TR", "TL", "TU", "TD", "EQ"]:
                fifo = self.inject_input_fifos[channel][direction]
                status[channel][direction] = {
                    "occupancy": len(fifo),
                    "ready": fifo.ready_signal(),
                    "valid": fifo.valid_signal(),
                }
        return status

    def get_connected_ips(self) -> List[str]:
        """获取连接的IP列表。"""
        return self.connected_ips.copy()
