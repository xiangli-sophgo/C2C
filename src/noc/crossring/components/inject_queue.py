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
from src.noc.utils.round_robin import RoundRobinScheduler
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
        self.current_cycle = 0  # 统一的周期管理

        # 获取FIFO配置
        self.iq_ch_depth = config.fifo_config.IQ_CH_DEPTH
        self.iq_out_depth = config.fifo_config.IQ_OUT_FIFO_DEPTH

        # 连接的IP列表
        self.connected_ips = []

        # 每个IP的inject channel_buffer
        self.ip_inject_channel_buffers = {}

        # 方向化的注入队列
        self.inject_input_fifos = self._create_direction_fifos()

        # 使用标准轮询调度器替代原有的仲裁状态管理
        self.inject_scheduler = RoundRobinScheduler()

        # 传输计划（两阶段执行用）
        self._inject_transfer_plan = []

    def _create_direction_fifos(self) -> Dict[str, Dict[str, PipelinedFIFO]]:
        """创建方向化FIFO集合。"""
        # 获取统计采样间隔
        sample_interval = self.config.basic_config.FIFO_STATS_SAMPLE_INTERVAL

        result = {}
        for channel in ["req", "rsp", "data"]:
            result[channel] = {}
            for direction in ["TR", "TL", "TU", "TD", "EQ"]:
                fifo = PipelinedFIFO(f"inject_{channel}_{direction}_{self.node_id}", depth=self.iq_out_depth)
                # 设置统计采样间隔
                fifo._stats_sample_interval = sample_interval
                result[channel][direction] = fifo
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

            # 为这个IP创建inject channel_buffer
            # 获取统计采样间隔
            sample_interval = self.config.basic_config.FIFO_STATS_SAMPLE_INTERVAL

            self.ip_inject_channel_buffers[ip_id] = {}
            for channel in ["req", "rsp", "data"]:
                fifo = PipelinedFIFO(f"ip_inject_channel_{channel}_{ip_id}_{self.node_id}", depth=self.iq_ch_depth)
                # 设置统计采样间隔
                fifo._stats_sample_interval = sample_interval
                self.ip_inject_channel_buffers[ip_id][channel] = fifo

            # 注入IP成功（轮询调度器无需显式管理IP列表）

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
        计算阶段：按(channel, direction)二维独立轮询IP。

        Args:
            cycle: 当前周期
        """
        self._inject_transfer_plan.clear()

        # 如果没有连接的IP，直接返回
        if not self.connected_ips:
            return

        # 按(channel, direction)组合进行独立轮询
        for channel in ["req", "rsp", "data"]:
            for direction in ["TR", "TL", "TU", "TD", "EQ"]:
                # 为每个(channel, direction)生成唯一的调度器key
                scheduler_key = f"IQ_{channel}_{direction}_{self.node_id}"
                
                # 检查目标方向FIFO是否可用
                direction_fifo = self.inject_input_fifos[channel][direction]
                if not direction_fifo.ready_signal():
                    continue

                # 定义IP检查函数
                def check_ip_ready(ip_id):
                    if ip_id not in self.ip_inject_channel_buffers:
                        return False
                    
                    # 检查该IP的channel buffer
                    channel_buffer = self.ip_inject_channel_buffers[ip_id][channel]
                    if not channel_buffer.valid_signal():
                        return False
                    
                    flit = channel_buffer.peek_output()
                    if flit is None:
                        return False
                    
                    # 确定flit的目标方向
                    target_direction = self._find_target_direction_for_flit(flit, channel)
                    return target_direction == direction

                # 使用轮询调度器选择IP
                selected_ip, _ = self.inject_scheduler.select(
                    scheduler_key, 
                    self.connected_ips, 
                    check_ip_ready
                )

                if selected_ip:
                    # 获取选中的flit
                    channel_buffer = self.ip_inject_channel_buffers[selected_ip][channel]
                    selected_flit = channel_buffer.peek_output()
                    
                    if selected_flit:
                        # 计划传输
                        self._inject_transfer_plan.append((selected_ip, channel, selected_flit, direction))

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

    def _step_all_fifos(self, method_name: str, cycle: int = None) -> None:
        """对所有FIFO执行指定方法的辅助函数。"""
        # 更新IP inject channel buffers
        for ip_id in self.connected_ips:
            for channel in ["req", "rsp", "data"]:
                fifo = self.ip_inject_channel_buffers[ip_id][channel]
                if cycle is not None:
                    getattr(fifo, method_name)(cycle)
                else:
                    getattr(fifo, method_name)()

        # 更新inject direction FIFOs
        for channel in ["req", "rsp", "data"]:
            for direction in ["TR", "TL", "TU", "TD", "EQ"]:
                fifo = self.inject_input_fifos[channel][direction]
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

    def _find_target_direction_for_flit(self, flit: CrossRingFlit, channel: str) -> str:
        """
        确定flit的目标传输方向。

        Args:
            flit: 要路由的flit
            channel: 通道类型

        Returns:
            目标方向（"TR", "TL", "TU", "TD", "EQ"）
        """
        # 优先使用拓扑路由表
        if self.topology and hasattr(self.topology, "get_next_direction"):
            try:
                direction = self.topology.get_next_direction(self.coordinates, flit.destination, flit)
                if direction:
                    return direction
            except Exception:
                pass  # 回退到默认路由

        # 使用回退路由计算
        return self._calculate_routing_direction_fallback(flit)

    def _get_routing_strategy(self) -> str:
        """获取路由策略的辅助方法。"""
        routing_strategy = getattr(self.config, "ROUTING_STRATEGY", "XY")
        if hasattr(routing_strategy, "value"):
            routing_strategy = routing_strategy.value
        return routing_strategy

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
