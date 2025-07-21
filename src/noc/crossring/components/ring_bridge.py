"""
CrossRing环形桥接管理。

负责处理：
- 维度转换逻辑
- Ring bridge输入/输出FIFO管理
- Ring bridge仲裁逻辑
- 方向路由决策
"""

from typing import Dict, List, Optional, Tuple
import logging

from src.noc.base.ip_interface import PipelinedFIFO
from ..flit import CrossRingFlit
from ..config import CrossRingConfig, RoutingStrategy


class RingBridge:
    """环形桥接管理类。"""

    def __init__(self, node_id: int, coordinates: Tuple[int, int], config: CrossRingConfig, logger: logging.Logger, topology=None):
        """
        初始化环形桥接管理器。

        Args:
            node_id: 节点ID
            coordinates: 节点坐标
            config: CrossRing配置
            logger: 日志记录器
            topology: 拓扑对象（用于路由表查询）
        """
        self.node_id = node_id
        self.coordinates = coordinates
        self.config = config
        self.logger = logger
        self.topology = topology

        # 获取FIFO配置
        self.rb_in_depth = getattr(config, "RB_IN_DEPTH", 16)
        self.rb_out_depth = getattr(config, "RB_OUT_DEPTH", 16)  # 增加输出FIFO深度以缓解死锁

        # ring_bridge输入FIFO
        self.ring_bridge_input_fifos = self._create_input_fifos()

        # ring_bridge输出FIFO
        self.ring_bridge_output_fifos = self._create_output_fifos()

        # Ring_bridge轮询仲裁器状态
        self.ring_bridge_arbitration_state = {
            "req": {"current_input": 0, "current_output": 0, "input_sources": [], "output_directions": [], "last_served_input": {}, "last_served_output": {}},
            "rsp": {"current_input": 0, "current_output": 0, "input_sources": [], "output_directions": [], "last_served_input": {}, "last_served_output": {}},
            "data": {"current_input": 0, "current_output": 0, "input_sources": [], "output_directions": [], "last_served_input": {}, "last_served_output": {}},
        }

        # Ring_bridge仲裁决策缓存（两阶段执行用）
        self.ring_bridge_arbitration_decisions = {
            "req": {"flit": None, "output_direction": None, "input_source": None},
            "rsp": {"flit": None, "output_direction": None, "input_source": None},
            "data": {"flit": None, "output_direction": None, "input_source": None},
        }

    def _create_input_fifos(self) -> Dict[str, Dict[str, PipelinedFIFO]]:
        """创建ring_bridge输入FIFO集合。"""
        return {
            channel: {direction: PipelinedFIFO(f"ring_bridge_in_{channel}_{direction}_{self.node_id}", depth=self.rb_in_depth) for direction in ["TR", "TL", "TU", "TD"]}
            for channel in ["req", "rsp", "data"]
        }

    def _create_output_fifos(self) -> Dict[str, Dict[str, PipelinedFIFO]]:
        """创建ring_bridge输出FIFO集合。"""
        return {
            channel: {direction: PipelinedFIFO(f"ring_bridge_out_{channel}_{direction}_{self.node_id}", depth=self.rb_out_depth) for direction in ["EQ", "TR", "TL", "TU", "TD"]}
            for channel in ["req", "rsp", "data"]
        }

    def _get_ring_bridge_config(self) -> Tuple[List[str], List[str]]:
        """根据路由策略获取ring_bridge的输入源和输出方向配置。"""
        routing_strategy = getattr(self.config, "ROUTING_STRATEGY", "XY")
        if hasattr(routing_strategy, "value"):
            routing_strategy = routing_strategy.value

        # 根据路由策略配置输入源和输出方向
        if routing_strategy == "XY":
            input_sources = ["IQ_TU", "IQ_TD", "RB_TR", "RB_TL"]  # 修正：移除RB_TU, RB_TD
            output_directions = ["EQ", "TU", "TD"]
        elif routing_strategy == "YX":
            input_sources = ["IQ_TR", "IQ_TL", "RB_TU", "RB_TD"]
            output_directions = ["EQ", "TR", "TL"]
        else:  # ADAPTIVE 或其他
            input_sources = ["IQ_TU", "IQ_TD", "IQ_TR", "IQ_TL", "RB_TR", "RB_TL", "RB_TU", "RB_TD"]
            output_directions = ["EQ", "TU", "TD", "TR", "TL"]

        return input_sources, output_directions

    def _initialize_ring_bridge_arbitration(self) -> None:
        """初始化ring_bridge仲裁的源和方向列表。"""
        input_sources, output_directions = self._get_ring_bridge_config()

        for channel in ["req", "rsp", "data"]:
            arb_state = self.ring_bridge_arbitration_state[channel]
            arb_state["input_sources"] = input_sources.copy()
            arb_state["output_directions"] = output_directions.copy()
            arb_state["last_served_input"] = {source: 0 for source in input_sources}
            arb_state["last_served_output"] = {direction: 0 for direction in output_directions}

    def add_to_ring_bridge_input(self, flit: CrossRingFlit, direction: str, channel: str) -> bool:
        """
        CrossPoint向ring_bridge输入添加flit。

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

    def get_eq_output_flit(self, channel: str) -> Optional[CrossRingFlit]:
        """从ring_bridge的EQ输出获取flit (为eject队列提供)。"""
        eq_fifo = self.ring_bridge_output_fifos[channel]["EQ"]
        if eq_fifo.valid_signal():
            return eq_fifo.read_output()
        return None

    def peek_output_flit(self, direction: str, channel: str) -> Optional[CrossRingFlit]:
        """查看ring_bridge的指定方向输出flit（不取出）。"""
        output_fifo = self.ring_bridge_output_fifos[channel][direction]
        if output_fifo.valid_signal():
            return output_fifo.peek_output()
        return None

    def get_output_flit(self, direction: str, channel: str) -> Optional[CrossRingFlit]:
        """从ring_bridge的指定方向输出获取flit。"""
        output_fifo = self.ring_bridge_output_fifos[channel][direction]
        if output_fifo.valid_signal():
            return output_fifo.read_output()
        return None

    def compute_arbitration(self, cycle: int, inject_direction_fifos: Dict) -> None:
        """
        计算ring_bridge仲裁决策（两阶段执行的compute阶段）。

        Args:
            cycle: 当前周期
            inject_direction_fifos: 注入方向FIFO
        """
        # 清空上一周期的决策
        for channel in ["req", "rsp", "data"]:
            self.ring_bridge_arbitration_decisions[channel] = {"flit": None, "output_direction": None, "input_source": None}

        # 首先初始化源和方向列表（如果还没有初始化）
        if not self.ring_bridge_arbitration_state["req"]["input_sources"]:
            self._initialize_ring_bridge_arbitration()

        # 为每个通道计算仲裁决策
        for channel in ["req", "rsp", "data"]:
            self._compute_channel_ring_bridge_arbitration(channel, cycle, inject_direction_fifos)

    def _compute_channel_ring_bridge_arbitration(self, channel: str, cycle: int, inject_direction_fifos: Dict) -> None:
        """计算单个通道的ring_bridge仲裁决策。"""
        arb_state = self.ring_bridge_arbitration_state[channel]
        input_sources = arb_state["input_sources"]

        # 轮询所有输入源，寻找可用的flit
        for input_attempt in range(len(input_sources)):
            current_input_idx = arb_state["current_input"]
            input_source = input_sources[current_input_idx]

            # 检查是否有可用的flit（但不取出）
            flit = self._peek_flit_from_ring_bridge_input(input_source, channel, inject_direction_fifos)
            if flit is not None:
                # 计算输出方向
                output_direction = self._determine_ring_bridge_output_direction(flit)

                # 检查输出FIFO是否可用
                output_fifo = self.ring_bridge_output_fifos[channel][output_direction]
                if output_fifo.ready_signal():
                    # 保存仲裁决策（在update阶段执行）
                    self.ring_bridge_arbitration_decisions[channel] = {"flit": flit, "output_direction": output_direction, "input_source": input_source}
                    # 成功仲裁后，更新current_input到下一个输入源以确保公平性
                    arb_state["current_input"] = (current_input_idx + 1) % len(input_sources)
                    break

            # 移动到下一个输入源
            arb_state["current_input"] = (current_input_idx + 1) % len(input_sources)

    def execute_arbitration(self, cycle: int, inject_direction_fifos: Dict) -> None:
        """
        执行ring_bridge仲裁决策（两阶段执行的update阶段）。

        Args:
            cycle: 当前周期
            inject_direction_fifos: 注入方向FIFO
        """
        for channel in ["req", "rsp", "data"]:
            decision = self.ring_bridge_arbitration_decisions[channel]
            if decision["flit"] is not None:
                # 执行之前计算的仲裁决策
                self._execute_channel_ring_bridge_transfer(channel, decision, cycle, inject_direction_fifos)

    def _execute_channel_ring_bridge_transfer(self, channel: str, decision: dict, cycle: int, inject_direction_fifos: Dict) -> None:
        """执行单个通道的ring_bridge传输。"""
        input_source = decision["input_source"]
        output_direction = decision["output_direction"]

        # 从输入源获取flit（实际取出）
        flit = self._get_flit_from_ring_bridge_input(input_source, channel, inject_direction_fifos)
        if flit is not None:
            # 分配到输出FIFO
            if self._assign_flit_to_ring_bridge_output(flit, output_direction, channel, cycle):
                # 成功传输，更新仲裁状态
                arb_state = self.ring_bridge_arbitration_state[channel]
                arb_state["last_served_input"][input_source] = cycle

                self.logger.debug(f"节点{self.node_id}: ring_bridge成功传输flit {flit.packet_id} 从{input_source}到{output_direction}")

    def _peek_flit_from_ring_bridge_input(self, input_source: str, channel: str, inject_direction_fifos: Dict) -> Optional[CrossRingFlit]:
        """查看ring_bridge输入中的flit（不取出）。"""
        if input_source.startswith("IQ_"):
            direction = input_source[3:]
            iq_fifo = inject_direction_fifos[channel][direction]
            if iq_fifo.valid_signal():
                return iq_fifo.peek_output()

        elif input_source.startswith("RB_"):
            direction = input_source[3:]
            rb_fifo = self.ring_bridge_input_fifos[channel][direction]
            if rb_fifo.valid_signal():
                return rb_fifo.peek_output()

        return None

    def _get_flit_from_ring_bridge_input(self, input_source: str, channel: str, inject_direction_fifos: Dict) -> Optional[CrossRingFlit]:
        """从指定的ring_bridge输入源获取flit。"""
        if input_source.startswith("IQ_"):
            direction = input_source[3:]
            iq_fifo = inject_direction_fifos[channel][direction]
            if iq_fifo.valid_signal():
                return iq_fifo.read_output()

        elif input_source.startswith("RB_"):
            direction = input_source[3:]
            rb_fifo = self.ring_bridge_input_fifos[channel][direction]
            if rb_fifo.valid_signal():
                return rb_fifo.read_output()

        return None

    def _determine_ring_bridge_output_direction(self, flit: CrossRingFlit) -> str:
        """确定flit在ring_bridge中的输出方向。"""
        # 首先检查是否是本地目标
        if self._is_local_destination(flit):
            return "EQ"

        # 否则，根据路由策略和目标位置确定输出方向
        return self._calculate_routing_direction(flit)

    def _is_local_destination(self, flit: CrossRingFlit) -> bool:
        """检查flit是否应该在本地弹出。"""
        if hasattr(flit, "destination") and flit.destination == self.node_id:
            return True
        if hasattr(flit, "dest_node_id") and flit.dest_node_id == self.node_id:
            return True
        if hasattr(flit, "dest_coordinates"):
            dest_col, dest_row = flit.dest_coordinates  # (x, y) -> (col, row)
            curr_col, curr_row = self.coordinates  # self.coordinates是(x, y)格式，即(col, row)
            if dest_row == curr_row and dest_col == curr_col:
                return True
        return False

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

    def _assign_flit_to_ring_bridge_output(self, flit: CrossRingFlit, output_direction: str, channel: str, cycle: int) -> bool:
        """将flit分配到ring_bridge输出FIFO。"""
        output_fifo = self.ring_bridge_output_fifos[channel][output_direction]
        if output_fifo.ready_signal():
            # 更新flit的ring_bridge位置信息
            flit.rb_fifo_name = f"RB_{output_direction}"
            flit.flit_position = f"RB_{output_direction}"

            if output_fifo.write_input(flit):
                # 成功分配，更新输出仲裁状态
                arb_state = self.ring_bridge_arbitration_state[channel]
                arb_state["last_served_output"][output_direction] = cycle

                self.logger.debug(f"节点{self.node_id}成功将{channel}通道flit分配到ring_bridge输出{output_direction}")
                return True

        return False

    def step_compute_phase(self, cycle: int) -> None:
        """FIFO组合逻辑更新。"""
        # 更新ring_bridge input/output FIFOs
        for channel in ["req", "rsp", "data"]:
            for direction in ["TR", "TL", "TU", "TD"]:
                self.ring_bridge_input_fifos[channel][direction].step_compute_phase(cycle)
            for direction in ["EQ", "TR", "TL", "TU", "TD"]:
                self.ring_bridge_output_fifos[channel][direction].step_compute_phase(cycle)

    def step_update_phase(self) -> None:
        """FIFO时序逻辑更新。"""
        # 更新ring_bridge input/output FIFOs
        for channel in ["req", "rsp", "data"]:
            for direction in ["TR", "TL", "TU", "TD"]:
                self.ring_bridge_input_fifos[channel][direction].step_update_phase()
            for direction in ["EQ", "TR", "TL", "TU", "TD"]:
                self.ring_bridge_output_fifos[channel][direction].step_update_phase()

    def get_stats(self) -> Dict:
        """获取统计信息。"""
        return {
            "buffer_occupancy": {
                "ring_bridge_input_fifos": {k: {d: len(v) for d, v in vv.items()} for k, vv in self.ring_bridge_input_fifos.items()},
                "ring_bridge_output_fifos": {k: {d: len(v) for d, v in vv.items()} for k, vv in self.ring_bridge_output_fifos.items()},
            }
        }
