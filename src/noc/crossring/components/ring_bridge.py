"""
CrossRing环形桥接管理。

负责处理：
- 维度转换逻辑
- Ring bridge输入/输出FIFO管理
- Ring bridge仲裁逻辑
- 方向路由决策
"""

from typing import Dict, List, Optional, Tuple

from src.noc.base.ip_interface import PipelinedFIFO
from ..flit import CrossRingFlit
from ..config import CrossRingConfig, RoutingStrategy


class RingBridge:
    """环形桥接管理类。"""

    def __init__(self, node_id: int, coordinates: Tuple[int, int], config: CrossRingConfig, topology=None):
        """
        初始化环形桥接管理器。

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
        self.parent_node = None  # 将在节点初始化时设置

        # 获取FIFO配置
        self.rb_in_depth = config.fifo_config.RB_IN_FIFO_DEPTH
        self.rb_out_depth = config.fifo_config.RB_OUT_FIFO_DEPTH

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
        # 获取统计采样间隔
        sample_interval = self.config.basic_config.FIFO_STATS_SAMPLE_INTERVAL
        
        result = {}
        for channel in ["req", "rsp", "data"]:
            result[channel] = {}
            for direction in ["TR", "TL", "TU", "TD"]:
                fifo = PipelinedFIFO(f"ring_bridge_in_{channel}_{direction}_{self.node_id}", depth=self.rb_in_depth)
                # 设置统计采样间隔
                fifo._stats_sample_interval = sample_interval
                result[channel][direction] = fifo
        return result

    def _create_output_fifos(self) -> Dict[str, Dict[str, PipelinedFIFO]]:
        """创建ring_bridge输出FIFO集合。"""
        # 获取统计采样间隔
        sample_interval = self.config.basic_config.FIFO_STATS_SAMPLE_INTERVAL
        
        result = {}
        for channel in ["req", "rsp", "data"]:
            result[channel] = {}
            for direction in ["EQ", "TR", "TL", "TU", "TD"]:
                fifo = PipelinedFIFO(f"ring_bridge_out_{channel}_{direction}_{self.node_id}", depth=self.rb_out_depth)
                # 设置统计采样间隔
                fifo._stats_sample_interval = sample_interval
                result[channel][direction] = fifo
        return result

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
                pass
            return success
        else:
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

    def compute_arbitration(self, cycle: int, inject_input_fifos: Dict) -> None:
        """
        计算ring_bridge仲裁决策（两阶段执行的compute阶段）。

        Args:
            cycle: 当前周期
            inject_input_fifos: 注入方向FIFO
        """
        # 清空上一周期的决策 - 改为列表以支持多个传输
        self.ring_bridge_arbitration_decisions = {"req": [], "rsp": [], "data": []}

        # 首先初始化源和方向列表（如果还没有初始化）
        if not self.ring_bridge_arbitration_state["req"]["input_sources"]:
            self._initialize_ring_bridge_arbitration()

        # 为每个通道计算仲裁决策（IQ源直接传输到RB输出，无需内部FIFO）
        for channel in ["req", "rsp", "data"]:
            self._compute_channel_ring_bridge_arbitration(channel, cycle, inject_input_fifos)

    def _compute_iq_to_rb_transfers(self, cycle: int, inject_input_fifos: Dict) -> None:
        """计算从IQ到RB内部FIFO的传输（两阶段执行模型第一阶段）。"""
        input_sources, _ = self._get_ring_bridge_config()
        
        # 初始化传输决策
        self.iq_to_rb_transfer_decisions = {"req": {}, "rsp": {}, "data": {}}
        
        for channel in ["req", "rsp", "data"]:
            for input_source in input_sources:
                if input_source.startswith("IQ_"):
                    direction = input_source[3:]  # 从IQ_TD得到TD
                    
                    # 检查IQ FIFO是否有数据
                    iq_fifo = inject_input_fifos[channel][direction]
                    if iq_fifo.valid_signal():
                        # 检查RB内部FIFO是否有空间
                        rb_input_fifo = self.ring_bridge_input_fifos[channel][direction]
                        if rb_input_fifo.ready_signal():
                            # 记录传输决策（在update阶段执行）
                            self.iq_to_rb_transfer_decisions[channel][direction] = True
                        else:
                            self.iq_to_rb_transfer_decisions[channel][direction] = False
                    else:
                        self.iq_to_rb_transfer_decisions[channel][direction] = False

    def _compute_channel_ring_bridge_arbitration(self, channel: str, cycle: int, inject_input_fifos: Dict) -> None:
        """计算单个通道的ring_bridge仲裁决策，支持不同输出方向的并行传输。"""
        arb_state = self.ring_bridge_arbitration_state[channel]
        input_sources = arb_state["input_sources"]
        
        # 记录每个输出方向是否已被占用（同一输出方向只能有一个flit）
        output_direction_used = set()
        first_transfer_source_idx = None  # 记录第一个成功传输的源索引

        # 轮询所有输入源，寻找可用的flit
        for input_attempt in range(len(input_sources)):
            current_input_idx = (arb_state["current_input"] + input_attempt) % len(input_sources)
            input_source = input_sources[current_input_idx]

            # 对于IQ源，直接从inject_input_fifos读取（绕过RB内部FIFO以减少延迟）
            # 对于RB源，从RB内部FIFO读取
            if input_source.startswith("IQ_"):
                direction = input_source[3:]
                iq_fifo = inject_input_fifos[channel][direction]
                if iq_fifo.valid_signal():
                    flit = iq_fifo.peek_output()
                else:
                    flit = None
            else:
                flit = self._peek_flit_from_ring_bridge_input(input_source, channel, inject_input_fifos)
            
            if flit is not None:
                # 计算输出方向
                output_direction = self._determine_ring_bridge_output_direction(flit)
                
                # 检查该输出方向是否已被占用
                if output_direction in output_direction_used:
                    continue  # 该输出方向已有flit，跳过

                # 检查输出FIFO是否可用
                output_fifo = self.ring_bridge_output_fifos[channel][output_direction]
                if output_fifo.ready_signal():
                    # 保存仲裁决策（在update阶段执行）
                    self.ring_bridge_arbitration_decisions[channel].append({
                        "flit": flit, 
                        "output_direction": output_direction, 
                        "input_source": input_source
                    })
                    output_direction_used.add(output_direction)  # 标记该输出方向已被占用
                    
                    # 记录第一个成功传输的源索引
                    if first_transfer_source_idx is None:
                        first_transfer_source_idx = current_input_idx
                    
                    # 继续检查其他源（不break），允许不同输出方向的并行传输

        # 更新起始源索引，确保下次从不同源开始
        if first_transfer_source_idx is not None:
            arb_state["current_input"] = (first_transfer_source_idx + 1) % len(input_sources)
        else:
            # 没有成功传输，也要更新索引确保轮询
            arb_state["current_input"] = (arb_state["current_input"] + 1) % len(input_sources)

    def execute_arbitration(self, cycle: int, inject_input_fifos: Dict) -> None:
        """
        执行ring_bridge仲裁决策（两阶段执行的update阶段）。

        Args:
            cycle: 当前周期
            inject_input_fifos: 注入方向FIFO
        """
        # 执行RB仲裁传输（IQ源直接传输到输出）
        for channel in ["req", "rsp", "data"]:
            decisions = self.ring_bridge_arbitration_decisions[channel]
            # 执行所有计算的仲裁决策
            for decision in decisions:
                if decision["flit"] is not None:
                    self._execute_channel_ring_bridge_transfer(channel, decision, cycle, inject_input_fifos)

    def _execute_iq_to_rb_transfers(self, cycle: int, inject_input_fifos: Dict) -> None:
        """执行从IQ到RB内部FIFO的传输（两阶段执行模型第一阶段）。"""
        if not hasattr(self, 'iq_to_rb_transfer_decisions'):
            return
            
        for channel in ["req", "rsp", "data"]:
            for direction, should_transfer in self.iq_to_rb_transfer_decisions[channel].items():
                if should_transfer:
                    # 从IQ FIFO读取flit
                    iq_fifo = inject_input_fifos[channel][direction]
                    if iq_fifo.valid_signal():
                        flit = iq_fifo.read_output()
                        if flit is not None:
                            # 写入RB内部FIFO
                            rb_input_fifo = self.ring_bridge_input_fifos[channel][direction]
                            if rb_input_fifo.ready_signal():
                                success = rb_input_fifo.write_input(flit)
                                if success:
                                    # 更新flit位置信息 - 简化为RB_direction，避免重复节点ID
                                    flit.flit_position = f"RB_{direction}"
                                    # 可选：打印调试信息
                                    # print(f"🔄 周期{cycle}: 节点{self.node_id} IQ_{direction} -> RB_{direction} 传输成功")

    def _execute_channel_ring_bridge_transfer(self, channel: str, decision: dict, cycle: int, inject_input_fifos: Dict) -> None:
        """执行单个通道的ring_bridge传输。"""
        input_source = decision["input_source"]
        output_direction = decision["output_direction"]

        # 根据输入源类型获取flit
        if input_source.startswith("IQ_"):
            # 直接从IQ FIFO读取（单周期传输）
            direction = input_source[3:]
            iq_fifo = inject_input_fifos[channel][direction]
            if iq_fifo.valid_signal():
                flit = iq_fifo.read_output()
            else:
                flit = None
        else:
            # 从RB内部FIFO读取
            flit = self._get_flit_from_ring_bridge_input(input_source, channel, inject_input_fifos)

        if flit is not None:
            # 分配到输出FIFO
            success = self._assign_flit_to_ring_bridge_output(flit, output_direction, channel, cycle)

            if success:
                # 成功传输，更新仲裁状态
                arb_state = self.ring_bridge_arbitration_state[channel]
                arb_state["last_served_input"][input_source] = cycle

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
                            # print(f"🔓 RB释放entry: 节点{self.node_id} 方向{direction} {priority}级entry")

                    # 清除flit的entry信息（已经释放）
                    delattr(flit, "allocated_entry_info")

    def _peek_flit_from_ring_bridge_input(self, input_source: str, channel: str, inject_input_fifos: Dict) -> Optional[CrossRingFlit]:
        """查看ring_bridge输入中的flit（不取出）。"""
        if input_source.startswith("IQ_"):
            # IQ源现在需要从RB内部FIFO读取（两阶段执行模型）
            direction = input_source[3:]
            rb_fifo = self.ring_bridge_input_fifos[channel][direction]
            if rb_fifo.valid_signal():
                return rb_fifo.peek_output()

        elif input_source.startswith("RB_"):
            direction = input_source[3:]
            rb_fifo = self.ring_bridge_input_fifos[channel][direction]
            if rb_fifo.valid_signal():
                return rb_fifo.peek_output()

        return None

    def _get_flit_from_ring_bridge_input(self, input_source: str, channel: str, inject_input_fifos: Dict) -> Optional[CrossRingFlit]:
        """从指定的ring_bridge输入源获取flit。"""
        if input_source.startswith("IQ_"):
            # IQ源现在需要从RB内部FIFO读取（两阶段执行模型）
            direction = input_source[3:]
            rb_fifo = self.ring_bridge_input_fifos[channel][direction]
            if rb_fifo.valid_signal():
                flit = rb_fifo.read_output()
                # 通知entry释放
                if flit and self.parent_node:
                    self._notify_entry_release(flit, channel, direction)
                return flit

        elif input_source.startswith("RB_"):
            direction = input_source[3:]
            rb_fifo = self.ring_bridge_input_fifos[channel][direction]
            if rb_fifo.valid_signal():
                flit = rb_fifo.read_output()
                # 通知entry释放
                if flit and self.parent_node:
                    self._notify_entry_release(flit, channel, direction)
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
        基于路径信息计算flit的路由方向。

        Args:
            flit: 要路由的flit

        Returns:
            路由方向（"TR", "TL", "TU", "TD", "EQ"）
        """
        current_node = self.node_id

        # 优先使用路径信息
        if hasattr(flit, "path") and flit.path:
            # 检查是否到达最终目标
            if current_node == flit.path[-1]:
                return "EQ"

            # 删除调试信息
            # if hasattr(flit, 'packet_id') and flit.packet_id == 1:
            #     print(f"🎯 RB节点{current_node}: flit {flit.packet_id} 路径={flit.path}, path_index={getattr(flit, 'path_index', '?')}")

            # 查找当前节点在路径中的位置
            try:
                # 首先尝试在路径中找到当前节点
                path_index = flit.path.index(current_node)

                # 如果找到了，检查是否有下一跳
                if path_index < len(flit.path) - 1:
                    next_node = flit.path[path_index + 1]
                    # 更新path_index为当前位置
                    if hasattr(flit, "path_index"):
                        flit.path_index = path_index
                else:
                    # 已经是路径的最后一个节点
                    return "EQ"

                # 根据下一跳计算方向
                direction = self._calculate_direction_to_next_node(current_node, next_node)
                # 删除debug输出
                # if hasattr(flit, 'packet_id') and flit.packet_id == 1:
                #     print(f"   -> 下一跳: 节点{next_node}, 方向: {direction}")
                return direction

            except ValueError:
                # 当前节点不在路径中，可能是特殊情况
                pass

        # 如果有topology对象，使用路由表
        if self.topology and hasattr(self.topology, "routing_table"):
            return self.topology.get_next_direction(self.node_id, flit.destination)

        # 回退到原始的路由计算方法
        return self._calculate_routing_direction_fallback(flit)

    def _calculate_direction_to_next_node(self, current_node: int, next_node: int) -> str:
        """计算从当前节点到下一节点的方向"""
        num_col = getattr(self.config, "NUM_COL", 3)

        curr_row = current_node // num_col
        curr_col = current_node % num_col
        next_row = next_node // num_col
        next_col = next_node % num_col

        # 获取路由策略
        routing_strategy = getattr(self.config, "ROUTING_STRATEGY", "XY")
        if hasattr(routing_strategy, "value"):
            routing_strategy = routing_strategy.value

        # 计算方向
        if routing_strategy == "XY":
            # XY路由：先水平后垂直
            if next_col != curr_col:
                return "TR" if next_col > curr_col else "TL"
            elif next_row != curr_row:
                return "TD" if next_row > curr_row else "TU"
        elif routing_strategy == "YX":
            # YX路由：先垂直后水平
            if next_row != curr_row:
                return "TD" if next_row > curr_row else "TU"
            elif next_col != curr_col:
                return "TR" if next_col > curr_col else "TL"
        else:
            # 默认使用XY路由
            if next_col != curr_col:
                return "TR" if next_col > curr_col else "TL"
            elif next_row != curr_row:
                return "TD" if next_row > curr_row else "TU"

        return "EQ"  # 已到达目标

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
