"""
CrossRing CrossPoint 重构实现 - 统一简化版本

本文件包含完整的CrossPoint功能实现，遵循以下设计原则：
1. 减少类和函数的使用，在单一类中实现所有功能
2. 详细的中文注释和清晰的逻辑
3. 完整实现E-Tag和I-Tag机制，不做简化
4. 统一处理水平和垂直CrossPoint的功能
5. 高效的两阶段执行模型

CrossPoint是CrossRing NoC的核心组件，负责：
- Flit的上环和下环控制
- E-Tag防饿死机制（分层entry管理 + T0全局队列轮询）
- I-Tag预约机制（slot预约 + 回收管理）
- 基于路由策略的下环决策
- 绕环机制处理
"""

from typing import Dict, List, Any, Optional, Tuple, Set
from enum import Enum
from dataclasses import dataclass, field

from ..link import PriorityLevel, CrossRingSlot, RingSlice
from ..flit import CrossRingFlit
from ..config import CrossRingConfig
from ...base.link import BasicDirection


class CrossPointDirection(Enum):
    """CrossPoint管理方向枚举"""

    HORIZONTAL = "horizontal"  # 管理TR/TL方向
    VERTICAL = "vertical"  # 管理TU/TD方向


@dataclass
class EntryAllocationTracker:
    """
    Entry分配跟踪器 - 管理分层entry的分配和占用

    根据CrossRing规范，每个方向有不同的entry层次结构：
    - TL/TU方向：有专用T0、T1、T2 entry
    - TR/TD方向：共享entry池，无专用层次
    """

    # FIFO配置参数
    total_depth: int  # FIFO总深度
    t2_max_entries: int  # T2级可用entry数量
    t1_max_entries: int  # T1级可用entry数量(包含T2)
    has_dedicated_entries: bool  # 是否有专用entry层次

    # 当前占用计数
    t0_occupied: int = 0  # T0级当前占用
    t1_occupied: int = 0  # T1级当前占用
    t2_occupied: int = 0  # T2级当前占用

    def get_t0_dedicated_available(self) -> int:
        """获取T0专用entry的可用数量"""
        if not self.has_dedicated_entries:
            return 0
        t0_dedicated_capacity = self.total_depth - self.t1_max_entries
        return max(0, t0_dedicated_capacity - self.t0_occupied)

    def get_total_occupied(self) -> int:
        """获取总占用数量"""
        return self.t0_occupied + self.t1_occupied + self.t2_occupied

    def can_allocate_entry(self, priority_level: str) -> bool:
        """检查是否可以分配指定优先级的entry"""
        if priority_level == "T2":
            return self.t2_occupied < self.t2_max_entries
        elif priority_level == "T1":
            if self.has_dedicated_entries:
                return (self.t1_occupied + self.t2_occupied) < self.t1_max_entries
            else:
                return self.get_total_occupied() < self.total_depth
        elif priority_level == "T0":
            return self.get_total_occupied() < self.total_depth
        return False

    def allocate_entry(self, priority_level: str) -> bool:
        """分配entry，成功返回True"""
        if not self.can_allocate_entry(priority_level):
            return False

        if priority_level == "T2":
            self.t2_occupied += 1
        elif priority_level == "T1":
            self.t1_occupied += 1
        elif priority_level == "T0":
            self.t0_occupied += 1
        else:
            return False
        return True

    def release_entry(self, priority_level: str) -> bool:
        """释放entry，成功返回True"""
        if priority_level == "T2" and self.t2_occupied > 0:
            self.t2_occupied -= 1
            return True
        elif priority_level == "T1" and self.t1_occupied > 0:
            self.t1_occupied -= 1
            return True
        elif priority_level == "T0" and self.t0_occupied > 0:
            self.t0_occupied -= 1
            return True
        return False


@dataclass
class ITagReservationState:
    """I-Tag预约状态跟踪"""

    active: bool = False  # 预约是否激活
    reserved_slot_id: Optional[str] = None  # 预约的slot ID
    reserver_node_id: Optional[int] = None  # 预约者节点ID
    trigger_cycle: int = 0  # 触发预约的周期
    wait_cycles: int = 0  # 等待周期数
    direction: Optional[str] = None  # 预约方向


class CrossPoint:
    """
    统一的CrossPoint实现 - 集成水平和垂直CrossPoint的完整功能

    核心功能：
    1. 管理4个slice连接（每个方向的arrival和departure slice）
    2. 实现完整的E-Tag机制（分层entry管理 + T0全局队列轮询）
    3. 实现完整的I-Tag预约机制（slot预约 + 回收管理）
    4. 处理上环和下环决策（基于路由策略和坐标）
    5. 两阶段执行模型（compute阶段计算，update阶段执行）

    设计原则：
    - 所有功能集中在一个类中，减少方法调用开销
    - 详细的中文注释说明每个逻辑块
    - 不使用小函数，核心逻辑直接写在主方法内
    - 完整实现所有Tag机制，不做简化
    """

    def __init__(self, crosspoint_id: str, node_id: int, direction: CrossPointDirection, config: CrossRingConfig, coordinates: Tuple[int, int] = (0, 0), parent_node=None):
        """
        初始化CrossPoint

        Args:
            crosspoint_id: CrossPoint标识符
            node_id: 所属节点ID
            direction: CrossPoint方向（水平/垂直）
            config: CrossRing配置
            coordinates: 节点坐标
            parent_node: 父节点引用
        """
        # 基础配置
        self.crosspoint_id = crosspoint_id
        self.node_id = node_id
        self.direction = direction
        self.config = config
        self.coordinates = coordinates
        self.parent_node = parent_node

        # 确定此CrossPoint管理的方向
        if direction == CrossPointDirection.HORIZONTAL:
            self.managed_directions = ["TL", "TR"]  # 水平CrossPoint管理左右方向
        else:  # VERTICAL
            self.managed_directions = ["TU", "TD"]  # 垂直CrossPoint管理上下方向

        # Slice连接管理 - 每个方向每个通道都有arrival和departure两个slice
        # arrival slice: 到达本节点的slice，用于下环判断
        # departure slice: 离开本节点的slice，用于上环操作
        # 结构: slice_connections[direction][channel][slice_type] = RingSlice
        self.slice_connections: Dict[str, Dict[str, Dict[str, Optional[RingSlice]]]] = {}
        for direction_name in self.managed_directions:
            self.slice_connections[direction_name] = {}
            for channel in ["req", "rsp", "data"]:
                self.slice_connections[direction_name][channel] = {"arrival": None, "departure": None}

        # E-Tag机制核心状态 - 分层entry管理
        self.etag_entry_managers: Dict[str, EntryAllocationTracker] = {}
        self._initialize_etag_entry_managers()

        # T0全局队列 - 每个通道独立的轮询队列
        self.t0_global_queues: Dict[str, List[CrossRingSlot]] = {"req": [], "rsp": [], "data": []}

        # I-Tag预约机制状态 - 每个通道每个环路方向独立管理
        self.itag_reservations: Dict[str, Dict[str, ITagReservationState]] = {
            "req": {"horizontal": ITagReservationState(), "vertical": ITagReservationState()},
            "rsp": {"horizontal": ITagReservationState(), "vertical": ITagReservationState()},
            "data": {"horizontal": ITagReservationState(), "vertical": ITagReservationState()},
        }

        # 注入等待队列 - 等待上环的flit及其等待周期数
        self.injection_wait_queues: Dict[str, List[Tuple[CrossRingFlit, int]]] = {"req": [], "rsp": [], "data": []}  # (flit, wait_cycles)

        # 两阶段执行的传输计划
        self.injection_transfer_plans: List[Dict[str, Any]] = []  # compute阶段确定的上环计划
        self.ejection_transfer_plans: List[Dict[str, Any]] = []  # compute阶段确定的下环计划

        # CrossPoint内部缓冲区 - 存储compute阶段下环的flit
        self.ejected_flits_buffer: Dict[str, Dict[str, List[CrossRingFlit]]] = {
            "RB": {"req": [], "rsp": [], "data": []},  # 下环到Ring Bridge的flit
            "EQ": {"req": [], "rsp": [], "data": []},  # 下环到EjectQueue的flit
        }

        # 统计信息 - 用于性能分析和调试
        self.stats = {
            # 基础传输统计
            "flits_injected": {"req": 0, "rsp": 0, "data": 0},
            "flits_ejected": {"req": 0, "rsp": 0, "data": 0},
            "bypass_events": {"req": 0, "rsp": 0, "data": 0},
            # E-Tag机制统计
            "etag_upgrades": {"req": {"T2_to_T1": 0, "T1_to_T0": 0}, "rsp": {"T2_to_T1": 0, "T1_to_T0": 0}, "data": {"T2_to_T1": 0, "T1_to_T0": 0}},
            "t0_queue_operations": {
                "req": {"added": 0, "removed": 0, "arbitrations": 0},
                "rsp": {"added": 0, "removed": 0, "arbitrations": 0},
                "data": {"added": 0, "removed": 0, "arbitrations": 0},
            },
            "entry_allocations": {"req": {"T0": 0, "T1": 0, "T2": 0}, "rsp": {"T0": 0, "T1": 0, "T2": 0}, "data": {"T0": 0, "T1": 0, "T2": 0}},
            "entry_releases": {"req": {"T0": 0, "T1": 0, "T2": 0}, "rsp": {"T0": 0, "T1": 0, "T2": 0}, "data": {"T0": 0, "T1": 0, "T2": 0}},
            # I-Tag机制统计
            "itag_triggers": {"req": 0, "rsp": 0, "data": 0},
            "itag_reservations": {"req": 0, "rsp": 0, "data": 0},
            "slot_recycling_events": {"req": 0, "rsp": 0, "data": 0},
        }

    def _initialize_etag_entry_managers(self) -> None:
        """
        初始化E-Tag的entry管理器

        根据CrossRing规范和路由策略确定每个方向的entry配置：
        - 横向环(TL/TR)在XY路由下下环到RB，在YX路由下下环到EQ
        - 纵向环(TU/TD)在XY路由下下环到EQ，在YX路由下下环到RB
        """
        for sub_direction in self.managed_directions:
            # 根据路由策略确定下环目标FIFO的深度
            routing_strategy = getattr(self.config, "ROUTING_STRATEGY", "XY")
            if hasattr(routing_strategy, "value"):
                routing_strategy = routing_strategy.value

            # 确定目标FIFO深度
            if routing_strategy == "XY":
                if sub_direction in ["TL", "TR"]:  # 横向环下环到RB
                    total_depth = self.config.fifo_config.RB_IN_FIFO_DEPTH
                else:  # TU, TD纵向环下环到EQ
                    total_depth = self.config.fifo_config.EQ_IN_FIFO_DEPTH
            elif routing_strategy == "YX":
                if sub_direction in ["TU", "TD"]:  # 纵向环下环到RB
                    total_depth = self.config.fifo_config.RB_IN_FIFO_DEPTH
                else:  # TL, TR横向环下环到EQ
                    total_depth = self.config.fifo_config.EQ_IN_FIFO_DEPTH
            else:
                # 默认使用较大的深度
                total_depth = max(self.config.fifo_config.RB_IN_FIFO_DEPTH, self.config.fifo_config.EQ_IN_FIFO_DEPTH)

            # 获取该方向的T1/T2配置阈值
            if sub_direction == "TL":
                t2_max = self.config.tag_config.TL_ETAG_T2_UE_MAX
                t1_max = self.config.tag_config.TL_ETAG_T1_UE_MAX
                has_dedicated = True  # TL有T0专用entry
            elif sub_direction == "TR":
                t2_max = self.config.tag_config.TR_ETAG_T2_UE_MAX
                t1_max = self.config.fifo_config.RB_IN_FIFO_DEPTH  # TR的T1_UE_MAX = RB_IN_FIFO_DEPTH
                has_dedicated = False  # TR无T0专用entry
            elif sub_direction == "TU":
                t2_max = self.config.tag_config.TU_ETAG_T2_UE_MAX
                t1_max = self.config.tag_config.TU_ETAG_T1_UE_MAX
                has_dedicated = True  # TU有T0专用entry
            elif sub_direction == "TD":
                t2_max = self.config.tag_config.TD_ETAG_T2_UE_MAX
                t1_max = self.config.fifo_config.EQ_IN_FIFO_DEPTH  # TD的T1_UE_MAX = EQ_IN_FIFO_DEPTH
                has_dedicated = False  # TD无T0专用entry
            else:
                raise ValueError(f"错误的方向{sub_direction}")

            # 创建entry管理器
            self.etag_entry_managers[sub_direction] = EntryAllocationTracker(
                total_depth=total_depth, t2_max_entries=t2_max, t1_max_entries=t1_max, has_dedicated_entries=has_dedicated
            )

    def connect_slice(self, direction: str, slice_type: str, ring_slice: RingSlice, channel: str) -> None:
        """
        连接Ring Slice到CrossPoint

        Args:
            direction: 方向 ("TL", "TR", "TU", "TD")
            slice_type: slice类型 ("arrival"到达, "departure"离开)
            ring_slice: Ring Slice实例
            channel: 通道类型 ("req", "rsp", "data")
        """
        if direction in self.slice_connections:
            if channel in self.slice_connections[direction]:
                if slice_type in self.slice_connections[direction][channel]:
                    self.slice_connections[direction][channel][slice_type] = ring_slice

    def step_compute_phase(self, cycle: int, node_inject_fifos: Dict[str, Dict[str, Any]], node_eject_fifos: Dict[str, Dict[str, Any]]) -> None:
        """
        计算阶段：分析传输可能性并制定传输计划，但不执行实际传输

        这个阶段的核心任务：
        1. 检查所有arrival slice，确定哪些flit可以下环
        2. 检查所有注入源（FIFO + ring_bridge），确定哪些flit可以上环
        3. 更新等待状态和触发Tag机制
        4. 制定详细的传输计划供update阶段执行

        Args:
            cycle: 当前周期
            node_inject_fifos: 节点的inject_input_fifos
            node_eject_fifos: 节点的eject_input_fifos
        """
        # 清空上一周期的传输计划和缓冲区
        self.injection_transfer_plans.clear()
        self.ejection_transfer_plans.clear()

        # 清空上一周期的下环缓冲区
        for eject_target in self.ejected_flits_buffer:
            for channel in self.ejected_flits_buffer[eject_target]:
                self.ejected_flits_buffer[eject_target][channel].clear()

        # ========== 第一部分：下环分析和计划 ==========
        # 遍历所有管理方向的arrival slice，分析下环可能性
        for direction in self.managed_directions:
            # 检查每个通道的当前slot
            for channel in ["req", "rsp", "data"]:
                arrival_slice = self.slice_connections[direction][channel]["arrival"]
                if not arrival_slice:
                    continue
                current_slot = arrival_slice.peek_current_slot(channel)
                if not current_slot or not current_slot.is_occupied:
                    continue

                flit = current_slot.flit
                if not flit:
                    continue

                # 判断是否应该下环以及下环目标
                should_eject, eject_target = self._should_eject_flit(flit, direction)

                if should_eject:
                    if eject_target == "RB":
                        # 下环到Ring Bridge - 同样需要E-Tag机制判断
                        # Ring Bridge的输入FIFO深度配置

                        # 检查Ring Bridge输入FIFO状态
                        if self.parent_node and hasattr(self.parent_node, "ring_bridge"):
                            # 获取对应方向的Ring Bridge输入FIFO占用情况

                            # 使用E-Tag机制判断是否可以下环到RB
                            can_eject = self._can_eject_with_etag_mechanism(current_slot, channel, direction)

                            if can_eject:
                                # compute阶段立即从slot中取出flit并存储到内部缓冲区
                                ejected_flit = current_slot.release_flit()
                                if ejected_flit:
                                    # 存储到RB缓冲区
                                    self.ejected_flits_buffer["RB"][channel].append(ejected_flit)

                                    # 添加下环计划供update阶段执行
                                    self.ejection_transfer_plans.append(
                                        {
                                            "type": "eject_to_ring_bridge",
                                            "direction": direction,
                                            "channel": channel,
                                            "flit": ejected_flit,
                                            "source_direction": direction,  # 记录来源方向用于Ring Bridge输入端口选择
                                            "original_slot": current_slot,  # 保留原slot引用用于清理
                                        }
                                    )
                            else:
                                # 下环失败，触发绕环和E-Tag升级处理
                                self._handle_ejection_failure_in_compute(current_slot, channel, direction, cycle)
                        else:
                            # 如果无法获取Ring Bridge状态，跳过下环
                            self._handle_ejection_failure_in_compute(current_slot, channel, direction, cycle)

                    elif eject_target == "EQ":
                        # 下环到EjectQueue - 需要检查目标FIFO状态
                        if direction in node_eject_fifos[channel]:
                            target_fifo = node_eject_fifos[channel][direction]

                            # 使用E-Tag机制判断是否可以下环
                            can_eject = self._can_eject_with_etag_mechanism(current_slot, channel, direction)

                            if can_eject:
                                # compute阶段立即从slot中取出flit并存储到内部缓冲区
                                ejected_flit = current_slot.release_flit()
                                if ejected_flit:
                                    # 存储到EQ缓冲区
                                    self.ejected_flits_buffer["EQ"][channel].append(ejected_flit)

                                    # 添加下环计划供update阶段执行
                                    self.ejection_transfer_plans.append(
                                        {
                                            "type": "eject_to_eq_fifo",
                                            "direction": direction,
                                            "channel": channel,
                                            "flit": ejected_flit,
                                            "target_fifo": target_fifo,
                                            "original_slot": current_slot,  # 保留原slot引用用于清理
                                        }
                                    )
                            else:
                                # 下环失败，触发绕环和E-Tag升级处理
                                self._handle_ejection_failure_in_compute(current_slot, channel, direction, cycle)

        # ========== 第二部分：上环分析和计划 ==========
        # 按照自然优先级顺序分析上环可能性：ring_bridge输出 > 普通FIFO
        for direction in self.managed_directions:
            for channel in ["req", "rsp", "data"]:
                departure_slice = self.slice_connections[direction][channel]["departure"]
                if not departure_slice:
                    continue
                # 1. 优先检查ring_bridge输出（维度转换后的flit重新注入）
                if self.parent_node and hasattr(self.parent_node, "ring_bridge"):
                    ring_bridge_flit = self.parent_node.ring_bridge.peek_output_flit(direction, channel)
                    if ring_bridge_flit:
                        # 检查departure slice是否可以接受flit
                        if self._can_inject_to_departure_slice(departure_slice, channel, direction):
                            self.injection_transfer_plans.append(
                                {"type": "ring_bridge_reinject", "direction": direction, "channel": channel, "flit": ring_bridge_flit, "priority": 1}
                            )  # 最高优先级
                            continue  # ring_bridge优先级高，如果有就不检查FIFO

                # 2. 检查普通inject_input_fifos
                if direction in node_inject_fifos[channel]:
                    direction_fifo = node_inject_fifos[channel][direction]

                    if direction_fifo.valid_signal():  # FIFO有有效输出
                        flit = direction_fifo.peek_output()
                        if flit:
                            # 检查departure slice是否可以接受
                            if self._can_inject_to_departure_slice(departure_slice, channel, direction):
                                self.injection_transfer_plans.append(
                                    {"type": "fifo_injection", "direction": direction, "channel": channel, "source_fifo": direction_fifo, "flit": flit, "priority": 2}  # 普通优先级
                                )
                            else:
                                # 无法上环，检查是否需要触发I-Tag预约
                                self._check_and_trigger_itag_reservation(flit, direction, channel, cycle)

        # ========== 第三部分：等待状态更新 ==========
        # 更新注入等待队列中flit的等待周期数
        for channel in ["req", "rsp", "data"]:
            updated_queue = []
            for flit, wait_cycles in self.injection_wait_queues[channel]:
                updated_queue.append((flit, wait_cycles + 1))
            self.injection_wait_queues[channel] = updated_queue

        # 更新I-Tag预约的等待周期
        for channel in ["req", "rsp", "data"]:
            for ring_type in ["horizontal", "vertical"]:
                reservation = self.itag_reservations[channel][ring_type]
                if reservation.active:
                    reservation.wait_cycles += 1
                    # 预约超时处理（简化版本，实际应该基于配置）
                    if reservation.wait_cycles > 20:
                        reservation.active = False
                        reservation.wait_cycles = 0

    def step_update_phase(self, cycle: int, node_inject_fifos: Dict[str, Dict[str, Any]], node_eject_fifos: Dict[str, Dict[str, Any]]) -> None:
        """
        更新阶段：执行compute阶段确定的传输计划

        这个阶段严格按照compute阶段的分析结果执行传输，
        不再进行额外的判断，确保两阶段模型的一致性。

        Args:
            cycle: 当前周期
            node_inject_fifos: 节点的inject_input_fifos
            node_eject_fifos: 节点的eject_input_fifos
        """
        # ========== 执行下环传输计划 ==========
        for plan in self.ejection_transfer_plans:
            if plan["type"] == "eject_to_ring_bridge":
                # 下环到Ring Bridge
                success = self._execute_eject_to_ring_bridge(plan)
                if success:
                    self.stats["flits_ejected"][plan["channel"]] += 1

            elif plan["type"] == "eject_to_eq_fifo":
                # 下环到EjectQueue FIFO
                success = self._execute_eject_to_eq_fifo(plan)
                if success:
                    self.stats["flits_ejected"][plan["channel"]] += 1

        # ========== 执行上环传输计划 ==========
        # 按优先级排序执行（ring_bridge优先于FIFO）
        sorted_plans = sorted(self.injection_transfer_plans, key=lambda x: x.get("priority", 999))

        for plan in sorted_plans:
            if plan["type"] == "ring_bridge_reinject":
                # Ring Bridge重新注入
                success = self._execute_ring_bridge_reinject(plan)
                if success:
                    self.stats["flits_injected"][plan["channel"]] += 1

            elif plan["type"] == "fifo_injection":
                # 普通FIFO注入
                success = self._execute_fifo_injection(plan)
                if success:
                    self.stats["flits_injected"][plan["channel"]] += 1
                else:
                    if hasattr(plan["flit"], "packet_id"):
                        raise RuntimeError(f"CrossPoint {self.crosspoint_id} 注入失败 packet {plan['flit'].packet_id}")
                        # 尝试将flit放回FIFO头部
                        if hasattr(plan["source_fifo"], "priority_write"):
                            if not plan["source_fifo"].priority_write(plan["flit"]):
                                raise RuntimeError(f"CrossPoint {self.crosspoint_id} 无法将packet {plan['flit'].packet_id} 放回FIFO，数据可能丢失！")

    def _should_eject_flit(self, flit: CrossRingFlit, arrival_direction: str) -> Tuple[bool, str]:
        """
        基于路径信息的下环决策逻辑

        使用flit的path和current_position来判断是否需要下环

        Args:
            flit: 要判断的flit
            arrival_direction: flit到达的方向（TL/TR/TU/TD）

        Returns:
            (是否下环, 下环目标: "RB"/"EQ"/"")
        """
        if not self.parent_node:
            return False, ""

        current_node = self.parent_node.node_id

        # 基于路径判断
        # 检查是否到达最终目标
        if current_node == flit.path[-1]:  # 路径的最后一个节点是目标
            # 根据来源方向决定下环目标
            if arrival_direction in ["TR", "TL"] and self.direction == CrossPointDirection.HORIZONTAL:
                return True, "RB"  # 水平环到达目标，通过RB下环
            elif arrival_direction in ["TU", "TD"] and self.direction == CrossPointDirection.VERTICAL:
                return True, "EQ"  # 垂直环到达目标，直接下环到IP
            else:
                return True, "EQ"  # Ring Bridge来的，直接到IP

        # 查找当前节点在路径中的位置
        try:
            path_index = flit.path.index(current_node)
            if path_index < len(flit.path) - 1:
                next_node = flit.path[path_index + 1]
                # 更新path_index
                if hasattr(flit, "path_index"):
                    flit.path_index = path_index

                # 判断下一跳是否需要维度转换
                if self.direction == CrossPointDirection.HORIZONTAL:
                    # 水平环：如果下一跳需要垂直移动，下环到RB
                    needs_vertical = self._needs_vertical_move(current_node, next_node)
                    if arrival_direction in ["TR", "TL"] and needs_vertical:
                        return True, "RB"
                elif self.direction == CrossPointDirection.VERTICAL:
                    # 垂直环：如果下一跳需要水平移动，下环到RB
                    needs_horizontal = self._needs_horizontal_move(current_node, next_node)
                    if arrival_direction in ["TU", "TD"] and needs_horizontal:
                        return True, "RB"
        except ValueError:
            # 当前节点不在路径中，可能是绕环情况
            # 对于绕环flit，检查是否到达了路径的目标节点
            if current_node == flit.path[-1]:  # 绕环到达目标节点
                if arrival_direction in ["TR", "TL"] and self.direction == CrossPointDirection.HORIZONTAL:
                    return True, "RB"  # 水平环到达目标，通过RB下环
                elif arrival_direction in ["TU", "TD"] and self.direction == CrossPointDirection.VERTICAL:
                    return True, "EQ"  # 垂直环到达目标，直接下环到IP
                else:
                    return True, "EQ"  # 默认下环到IP
            # 否则继续绕环
            pass

        # 来自Ring Bridge的flit，直接下环到IP
        if arrival_direction not in ["TR", "TL", "TU", "TD"]:
            return True, "EQ"

        # 继续在当前环传输
        return False, ""

    def _needs_vertical_move(self, current_node: int, next_node: int) -> bool:
        """判断从当前节点到下一节点是否需要垂直移动"""
        if not self.parent_node or not hasattr(self.parent_node.config, "NUM_COL"):
            return False
        num_col = self.parent_node.config.NUM_COL
        curr_row = current_node // num_col
        next_row = next_node // num_col
        return curr_row != next_row

    def _needs_horizontal_move(self, current_node: int, next_node: int) -> bool:
        """判断从当前节点到下一节点是否需要水平移动"""
        if not self.parent_node or not hasattr(self.parent_node.config, "NUM_COL"):
            return False
        num_col = self.parent_node.config.NUM_COL
        curr_col = current_node % num_col
        next_col = next_node % num_col
        return curr_col != next_col

    def _can_eject_with_etag_mechanism(self, slot: CrossRingSlot, channel: str, direction: str, is_compute_phase: bool = True) -> bool:
        """
        完整的E-Tag机制下环判断逻辑

        E-Tag分层entry使用规则：
        1. T2级：只能使用T2专用entry
        2. T1级：优先使用T1专用entry，不够时使用T2 entry
        3. T0级：优先使用T0专用entry，然后依次降级使用T1、T2 entry
                只有使用T0专用entry时才需要进行轮询检查

        Args:
            slot: 包含flit的slot
            channel: 通道类型
            direction: 方向
            fifo_occupancy: 目标FIFO当前占用
            fifo_depth: 目标FIFO总深度
            is_compute_phase: 是否为compute阶段（True: 分配entry，False: 只检查已分配）

        Returns:
            是否可以下环
        """
        if not slot.is_occupied:
            return False

        # 在update阶段，如果slot已经有分配的entry信息，直接返回True
        if not is_compute_phase and hasattr(slot, "allocated_entry_info") and slot.allocated_entry_info:
            return True

        # 获取flit的E-Tag优先级
        priority = slot.etag_priority if slot.etag_marked else PriorityLevel.T2

        # 获取该方向的entry管理器
        if direction not in self.etag_entry_managers:
            raise ValueError(f"未找到方向 {direction} 的entry管理器")

        entry_manager = self.etag_entry_managers[direction]

        # 在compute阶段进行entry分配，在update阶段只做检查
        if is_compute_phase:
            # 根据优先级进行分层entry分配判断和实际分配（compute阶段必须分配防止竞争）
            if priority == PriorityLevel.T2:
                # T2级：只能使用T2专用entry
                if entry_manager.can_allocate_entry("T2"):
                    success = entry_manager.allocate_entry("T2")
                    if success:
                        self.stats["entry_allocations"][channel]["T2"] += 1
                        # 记录分配的entry信息到slot和flit中，用于后续释放
                        slot.allocated_entry_info = {"direction": direction, "priority": "T2"}
                        if slot.flit:
                            slot.flit.allocated_entry_info = {"direction": direction, "priority": "T2"}
                    return success
                return False

            elif priority == PriorityLevel.T1:
                # T1级：优先使用T1专用entry，不够时使用T2 entry
                if entry_manager.can_allocate_entry("T1"):
                    success = entry_manager.allocate_entry("T1")
                    if success:
                        self.stats["entry_allocations"][channel]["T1"] += 1
                        # 记录分配的entry信息到slot和flit中，用于后续释放
                        slot.allocated_entry_info = {"direction": direction, "priority": "T1"}
                        if slot.flit:
                            slot.flit.allocated_entry_info = {"direction": direction, "priority": "T1"}
                    return success
                return False

            elif priority == PriorityLevel.T0:
                # T0级：最复杂的分配逻辑
                if not entry_manager.can_allocate_entry("T0"):
                    return False

                # 检查是否有专用entry的方向
                if entry_manager.has_dedicated_entries:
                    # 计算T0专用entry的可用数量
                    t0_dedicated_available = entry_manager.get_t0_dedicated_available()

                    if t0_dedicated_available > 0:
                        # 有T0专用entry可用，需要进行轮询检查
                        is_first_in_queue = self._is_first_in_t0_queue(slot, channel)
                        if is_first_in_queue:
                            success = entry_manager.allocate_entry("T0")
                            if success:
                                self.stats["entry_allocations"][channel]["T0"] += 1
                                # 记录分配的entry信息到slot和flit中
                                slot.allocated_entry_info = {"direction": direction, "priority": "T0"}
                                if slot.flit:
                                    slot.flit.allocated_entry_info = {"direction": direction, "priority": "T0"}
                            return success
                        else:
                            return False
                    else:
                        # 没有T0专用entry，使用其他等级entry，无需轮询检查
                        success = entry_manager.allocate_entry("T0")
                        if success:
                            self.stats["entry_allocations"][channel]["T0"] += 1
                            # 记录分配的entry信息到slot和flit中
                            slot.allocated_entry_info = {"direction": direction, "priority": "T0"}
                            if slot.flit:
                                slot.flit.allocated_entry_info = {"direction": direction, "priority": "T0"}
                        return success
                else:
                    # 没有专用entry的方向（TR/TD），使用共享entry池，无需轮询
                    success = entry_manager.allocate_entry("T0")
                    if success:
                        self.stats["entry_allocations"][channel]["T0"] += 1
                        # 记录分配的entry信息到slot和flit中
                        slot.allocated_entry_info = {"direction": direction, "priority": "T0"}
                        if slot.flit:
                            slot.flit.allocated_entry_info = {"direction": direction, "priority": "T0"}
                    return success
        else:
            # update阶段：不分配entry，只检查是否符合条件
            return False

        return False

    def _is_first_in_t0_queue(self, slot: CrossRingSlot, channel: str) -> bool:
        """
        检查slot是否在T0全局队列的第一位（轮询仲裁）

        这个检查是T0机制的核心，不能简化！
        当多个T0级slot竞争T0专用entry时，只有队列第一位的slot可以使用。

        Args:
            slot: 要检查的slot
            channel: 通道类型

        Returns:
            是否在队列第一位
        """
        if channel not in self.t0_global_queues:
            return False

        queue = self.t0_global_queues[channel]

        # 队列为空或slot不在队列中
        if not queue or slot not in queue:
            return False

        # 检查是否在队列第一位
        is_first = queue[0] == slot

        if is_first:
            self.stats["t0_queue_operations"][channel]["arbitrations"] += 1

        return is_first

    def _add_to_t0_queue(self, slot: CrossRingSlot, channel: str) -> bool:
        """
        将slot加入T0全局队列

        Args:
            slot: 要加入的slot
            channel: 通道类型

        Returns:
            是否成功加入
        """
        if channel not in self.t0_global_queues:
            return False

        queue = self.t0_global_queues[channel]

        # 避免重复添加
        if slot not in queue:
            queue.append(slot)
            self.stats["t0_queue_operations"][channel]["added"] += 1
            return True
        else:
            return False

    def _remove_from_t0_queue(self, slot: CrossRingSlot, channel: str) -> bool:
        """
        从T0全局队列移除slot

        Args:
            slot: 要移除的slot
            channel: 通道类型

        Returns:
            是否成功移除
        """
        if channel not in self.t0_global_queues:
            return False

        queue = self.t0_global_queues[channel]

        if slot in queue:
            queue.remove(slot)
            self.stats["t0_queue_operations"][channel]["removed"] += 1
            return True
        else:
            return False

    def _handle_ejection_failure_in_compute(self, slot: CrossRingSlot, channel: str, direction: str, cycle: int) -> None:
        """
        在compute阶段处理下环失败的情况

        下环失败时的处理：
        1. 记录绕环事件
        2. 检查是否需要E-Tag升级
        3. 如果升级到T0，加入T0全局队列

        Args:
            slot: 下环失败的slot
            channel: 通道类型
            direction: 方向
            cycle: 当前周期
        """
        # 记录绕环事件
        self.stats["bypass_events"][channel] += 1

        # 获取当前优先级
        current_priority = slot.etag_priority if slot.etag_marked else PriorityLevel.T2

        # 检查E-Tag升级条件
        should_upgrade, new_priority = self._should_upgrade_etag(slot, channel, direction)

        if should_upgrade and new_priority != current_priority:
            # 执行E-Tag升级
            slot.mark_etag(new_priority, direction)

            # 更新统计
            if current_priority == PriorityLevel.T2 and new_priority == PriorityLevel.T1:
                self.stats["etag_upgrades"][channel]["T2_to_T1"] += 1
            elif current_priority == PriorityLevel.T1 and new_priority == PriorityLevel.T0:
                self.stats["etag_upgrades"][channel]["T1_to_T0"] += 1

            # 如果升级到T0，加入T0全局队列
            if new_priority == PriorityLevel.T0:
                self._add_to_t0_queue(slot, channel)

        else:
            pass

    def _should_upgrade_etag(self, slot: CrossRingSlot, channel: str, direction: str) -> Tuple[bool, PriorityLevel]:
        """
        检查是否应该升级E-Tag优先级

        升级规则：
        1. T2→T1升级：
           - ETAG_BOTHSIDE_UPGRADE=0: 只有TL和TU能升级
           - ETAG_BOTHSIDE_UPGRADE=1: 所有方向都能升级
        2. T1→T0升级：只有TL和TU能升级，TR和TD永远不能升级到T0

        Args:
            slot: 要检查的slot
            channel: 通道类型
            direction: 方向

        Returns:
            (是否应该升级, 新优先级)
        """
        if not slot.is_occupied:
            return False, slot.etag_priority

        current_priority = slot.etag_priority if slot.etag_marked else PriorityLevel.T2

        # 获取ETAG_BOTHSIDE_UPGRADE配置
        bothside_upgrade = self.config.tag_config.ETAG_BOTHSIDE_UPGRADE

        if current_priority == PriorityLevel.T2:
            # T2 -> T1 升级判断
            if bothside_upgrade == 0:
                # 只有TL和TU能升级到T1
                can_upgrade = direction in ["TL", "TU"]
            else:
                # 所有方向都能升级到T1
                can_upgrade = True

            if can_upgrade:
                return True, PriorityLevel.T1

        elif current_priority == PriorityLevel.T1:
            # T1 -> T0 升级判断：只有TL和TU能升级到T0
            if direction in ["TL", "TU"]:
                return True, PriorityLevel.T0

        return False, current_priority

    def _can_inject_to_departure_slice(self, departure_slice: RingSlice, channel: str, direction: str) -> bool:
        """
        检查是否可以向departure slice注入flit

        注入条件：
        1. departure slice的当前slot为空或可用
        2. 如果有I-Tag预约的slot，预约者有优先权

        Args:
            departure_slice: 目标departure slice
            channel: 通道类型
            direction: 方向

        Returns:
            是否可以注入
        """
        if not departure_slice:
            return False

        # 检查当前slot状态
        current_slot = departure_slice.peek_current_slot(channel)

        # 如果没有slot或slot为空，可以注入
        if current_slot is None or not current_slot.is_occupied:
            return True

        # 如果slot被本节点预约，可以注入
        if current_slot.is_reserved and current_slot.itag_reserver_id == self.node_id:
            return True

        # 其他情况不能注入
        return False

    def _check_and_trigger_itag_reservation(self, flit: CrossRingFlit, direction: str, channel: str, cycle: int) -> None:
        """
        检查是否需要触发I-Tag预约

        I-Tag预约触发条件：
        1. flit等待时间超过配置的阈值
        2. 当前通道在对应环路方向没有活跃的预约
        3. 预约数量未超过最大限制

        Args:
            flit: 等待注入的flit
            direction: 注入方向
            channel: 通道类型
            cycle: 当前周期
        """
        # 确定环路类型
        ring_type = "horizontal" if direction in ["TL", "TR"] else "vertical"

        # 检查flit等待时间
        if not hasattr(flit, "injection_wait_start_cycle"):
            flit.injection_wait_start_cycle = cycle
            return

        wait_cycles = cycle - flit.injection_wait_start_cycle

        # 获取I-Tag触发阈值
        if ring_type == "horizontal":
            trigger_threshold = self.config.tag_config.ITAG_TRIGGER_TH_H
            max_reservations = self.config.tag_config.ITAG_MAX_NUM_H
        else:
            trigger_threshold = self.config.tag_config.ITAG_TRIGGER_TH_V
            max_reservations = self.config.tag_config.ITAG_MAX_NUM_V

        # 检查触发条件
        if wait_cycles < trigger_threshold:
            return

        # 检查是否已有活跃预约
        current_reservation = self.itag_reservations[channel][ring_type]
        if current_reservation.active:
            return

        # 触发I-Tag预约
        departure_slice = self.slice_connections[direction][channel]["departure"]
        if departure_slice:
            # 查找可预约的slot（简化实现）
            success = self._trigger_itag_reservation_internal(channel, ring_type, departure_slice, cycle)
            if success:
                self.stats["itag_triggers"][channel] += 1
                self.stats["itag_reservations"][channel] += 1

    def _trigger_itag_reservation_internal(self, channel: str, ring_type: str, departure_slice: RingSlice, cycle: int) -> bool:
        """
        内部I-Tag预约触发逻辑

        Args:
            channel: 通道类型
            ring_type: 环路类型
            departure_slice: departure slice
            cycle: 当前周期

        Returns:
            是否成功触发预约
        """
        # 简化实现：查找departure slice的当前slot
        current_slot = departure_slice.peek_current_slot(channel)

        # 如果slot可用且未被预约，进行预约
        if current_slot and current_slot.is_available and not current_slot.is_reserved:
            # 预约slot
            success = current_slot.reserve_itag(self.node_id, ring_type)
            if success:
                # 更新预约状态
                self.itag_reservations[channel][ring_type] = ITagReservationState(
                    active=True, reserved_slot_id=current_slot.slot_id, reserver_node_id=self.node_id, trigger_cycle=cycle, wait_cycles=0, direction=ring_type
                )
                return True

        return False

    def _execute_eject_to_ring_bridge(self, plan: Dict[str, Any]) -> bool:
        """
        执行下环到Ring Bridge的传输

        Args:
            plan: 下环计划

        Returns:
            是否执行成功
        """
        flit = plan["flit"]  # compute阶段已从slot中取出的flit
        source_direction = plan["source_direction"]
        channel = plan["channel"]
        original_slot = plan["original_slot"]

        # 更新flit状态
        flit.flit_position = f"RB_{source_direction}"
        flit.current_node_id = self.node_id
        flit.rb_fifo_name = f"RB_{source_direction}"

        # 添加到ring_bridge输入
        if self.parent_node and hasattr(self.parent_node, "add_to_ring_bridge_input"):
            success = self.parent_node.add_to_ring_bridge_input(flit, source_direction, channel)
            if success:
                # 处理成功下环的清理工作
                self._handle_successful_ejection(original_slot, channel, source_direction)
                return True
            else:
                # 添加失败，将flit放回slot
                original_slot.assign_flit(flit)
                return False

        return False

    def _execute_eject_to_eq_fifo(self, plan: Dict[str, Any]) -> bool:
        """
        执行下环到EjectQueue FIFO的传输

        Args:
            plan: 下环计划

        Returns:
            是否执行成功
        """
        flit = plan["flit"]  # compute阶段已从slot中取出的flit
        target_fifo = plan["target_fifo"]
        direction = plan["direction"]
        channel = plan["channel"]
        original_slot = plan["original_slot"]

        # 更新flit状态
        flit.flit_position = f"EQ_{direction}"
        flit.current_node_id = self.node_id

        # 直接写入目标FIFO（compute阶段已经检查过E-Tag机制）
        write_success = target_fifo.write_input(flit)
        if write_success:
            # 处理成功下环的清理工作
            self._handle_successful_ejection(original_slot, channel, direction)
            return True
        else:
            # 写入失败，将flit放回slot
            original_slot.assign_flit(flit)
            return False

    def _execute_ring_bridge_reinject(self, plan: Dict[str, Any]) -> bool:
        """
        执行Ring Bridge重新注入

        Args:
            plan: 注入计划

        Returns:
            是否执行成功
        """
        direction = plan["direction"]
        channel = plan["channel"]

        # 从ring_bridge获取实际flit
        if self.parent_node and hasattr(self.parent_node, "get_ring_bridge_output_flit"):
            actual_flit = self.parent_node.get_ring_bridge_output_flit(direction, channel)
            if actual_flit.packet_id == 6 and actual_flit.flit_id == 0:
                print(actual_flit)
            if actual_flit:
                return self._inject_flit_to_departure_slice(actual_flit, direction, channel)

        return False

    def _execute_fifo_injection(self, plan: Dict[str, Any]) -> bool:
        """
        执行普通FIFO注入

        Args:
            plan: 注入计划

        Returns:
            是否执行成功
        """
        source_fifo = plan["source_fifo"]
        direction = plan["direction"]
        channel = plan["channel"]

        # 从FIFO读取flit
        flit = source_fifo.read_output()
        if flit:
            return self._inject_flit_to_departure_slice(flit, direction, channel)

        return False

    def _inject_flit_to_departure_slice(self, flit: CrossRingFlit, direction: str, channel: str) -> bool:
        """
        将flit注入到departure slice

        Args:
            flit: 要注入的flit
            direction: 注入方向
            channel: 通道类型

        Returns:
            是否注入成功
        """
        departure_slice = self.slice_connections[direction][channel]["departure"]
        if not departure_slice:
            return False

        current_slot = departure_slice.peek_current_slot(channel)

        # 使用预约的slot或空闲slot
        if current_slot.is_reserved and current_slot.itag_reserver_id == self.node_id:
            # 使用预约的slot
            current_slot.assign_flit(flit)
            current_slot.clear_itag()  # 清除预约标记

            # 清除本地预约状态
            ring_type = "horizontal" if direction in ["TL", "TR"] else "vertical"
            self.itag_reservations[channel][ring_type].active = False

        elif not current_slot.is_occupied:
            # 使用空闲slot
            current_slot.assign_flit(flit)

        else:
            # slot被占用，注入失败
            return False

        # 更新flit状态信息
        flit.current_node_id = self.node_id

        target_node = self._calculate_target_node_for_direction(direction)
        if target_node == self.node_id:
            # 自环链路
            reverse_direction = {"TR": "TL", "TL": "TR", "TU": "TD", "TD": "TU"}.get(direction, direction)
            flit.current_link_id = f"link_{self.node_id}_{direction}_{reverse_direction}_{target_node}"
        else:
            # 普通链路
            flit.current_link_id = f"link_{self.node_id}_{direction}_{target_node}"

        flit.current_slice_index = 0
        flit.crosspoint_direction = "departure"
        flit.current_position = self.node_id

        # 设置链路源和目标节点信息
        flit.link_source_node = self.node_id
        flit.link_dest_node = target_node

        # 设置flit位置为标准格式
        flit.flit_position = f"{self.node_id}->{target_node}:0"

        # 设置slot索引（如果有的话）
        if current_slot:
            flit.current_slot_index = current_slot.slot_id

        return True

    def _calculate_target_node_for_direction(self, direction: str) -> int:
        """
        根据方向计算目标节点ID

        Args:
            direction: 方向（TR/TL/TU/TD）

        Returns:
            目标节点ID
        """
        if not self.parent_node or not hasattr(self.parent_node, "config"):
            return self.node_id  # 如果无法获取配置，返回自环

        num_cols = self.parent_node.config.NUM_COL
        num_rows = self.parent_node.config.NUM_ROW

        current_row = self.node_id // num_cols
        current_col = self.node_id % num_cols

        if direction == "TR":  # Turn Right - 向右
            target_col = (current_col + 1) % num_cols if current_col < num_cols - 1 else current_col
            if target_col == current_col:  # 边界情况，自环
                return self.node_id
            return current_row * num_cols + target_col

        elif direction == "TL":  # Turn Left - 向左
            target_col = (current_col - 1) % num_cols if current_col > 0 else current_col
            if target_col == current_col:  # 边界情况，自环
                return self.node_id
            return current_row * num_cols + target_col

        elif direction == "TU":  # Turn Up - 向上
            target_row = (current_row - 1) % num_rows if current_row > 0 else current_row
            if target_row == current_row:  # 边界情况，自环
                return self.node_id
            return target_row * num_cols + current_col

        elif direction == "TD":  # Turn Down - 向下
            target_row = (current_row + 1) % num_rows if current_row < num_rows - 1 else current_row
            if target_row == current_row:  # 边界情况，自环
                return self.node_id
            return target_row * num_cols + current_col

        else:
            return self.node_id  # 未知方向，返回自环

    def _handle_successful_ejection(self, slot: CrossRingSlot, channel: str, direction: str) -> None:
        """
        处理成功下环后的清理工作

        Args:
            slot: 下环的slot
            channel: 通道类型
            direction: 方向
        """
        priority = slot.etag_priority if slot.etag_marked else PriorityLevel.T2

        # 如果是T0级，从所有T0队列中移除
        if priority == PriorityLevel.T0:
            removed_count = 0
            for ch in ["req", "rsp", "data"]:
                if self._remove_from_t0_queue(slot, ch):
                    removed_count += 1

            if removed_count > 0:
                pass

        # 清理slot的E-Tag标记
        slot.clear_etag()

        # 清理allocated_entry_info
        if hasattr(slot, "allocated_entry_info"):
            delattr(slot, "allocated_entry_info")

    # def step(self, cycle: int, node_inject_fifos: Dict[str, Dict[str, Any]], node_eject_fifos: Dict[str, Dict[str, Any]]) -> None:
    #     """
    #     CrossPoint主步进函数 - 执行两阶段处理

    #     Args:
    #         cycle: 当前周期
    #         node_inject_fifos: 节点的inject_input_fifos
    #         node_eject_fifos: 节点的eject_input_fifos
    #     """
    #     # 第一阶段：计算阶段
    #     self.step_compute_phase(cycle, node_inject_fifos, node_eject_fifos)

    #     # 第二阶段：更新阶段
    #     self.step_update_phase(cycle, node_inject_fifos, node_eject_fifos)

    def get_crosspoint_status(self) -> Dict[str, Any]:
        """
        获取CrossPoint详细状态信息

        Returns:
            状态信息字典
        """
        return {
            "crosspoint_id": self.crosspoint_id,
            "node_id": self.node_id,
            "direction": self.direction.value,
            "managed_directions": self.managed_directions,
            # Slice连接状态
            "slice_connections": {
                direction: {slice_type: slice_ref is not None for slice_type, slice_ref in slices.items()} for direction, slices in self.slice_connections.items()
            },
            # E-Tag状态
            "etag_entry_managers": {
                direction: {
                    "total_depth": manager.total_depth,
                    "total_occupied": manager.get_total_occupied(),
                    "t0_occupied": manager.t0_occupied,
                    "t1_occupied": manager.t1_occupied,
                    "t2_occupied": manager.t2_occupied,
                    "t0_dedicated_available": manager.get_t0_dedicated_available(),
                }
                for direction, manager in self.etag_entry_managers.items()
            },
            # T0队列状态
            "t0_global_queues": {channel: {"length": len(queue), "first_slot_id": queue[0].slot_id if queue else None} for channel, queue in self.t0_global_queues.items()},
            # I-Tag预约状态
            "itag_reservations": {
                channel: {
                    ring_type: {"active": reservation.active, "slot_id": reservation.reserved_slot_id, "wait_cycles": reservation.wait_cycles}
                    for ring_type, reservation in reservations.items()
                }
                for channel, reservations in self.itag_reservations.items()
            },
            # 等待队列状态
            "injection_wait_queues": {channel: len(queue) for channel, queue in self.injection_wait_queues.items()},
            # 统计信息
            "stats": self.stats.copy(),
        }
