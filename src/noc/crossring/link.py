"""
CrossRing链路实现，继承BaseLink，实现CrossRing特有的ETag/ITag机制。

本模块实现了CrossRing拓扑中链路的具体逻辑，包括：
- CrossRing特有的ETag升级规则（双向vs单向）
- ITag反饿死机制
- 基于配置的拥塞控制阈值
- Ring Bridge的slice调度逻辑
"""

from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, field
from enum import Enum

from ..base.link import BaseLink, LinkSlot, BasicPriority, BasicDirection
from ..base.flit import BaseFlit
from ..base.ip_interface import PipelinedFIFO
from .config import CrossRingConfig
from .flit import CrossRingFlit


class PriorityLevel(Enum):
    """CrossRing特定的ETag优先级"""

    T0 = "T0"  # 最高优先级
    T1 = "T1"  # 中等优先级
    T2 = "T2"  # 最低优先级


class Direction(Enum):
    """CrossRing特定的传输方向"""

    TR = "TR"  # 向右(To Right)
    TL = "TL"  # 向左(To Left)
    TU = "TU"  # 向上(To Up)
    TD = "TD"  # 向下(To Down)


@dataclass
class LinkBandwidthTracker:
    """链路带宽统计跟踪器 - 在链路末端slice观测点统计slot状态"""

    # 每个通道的cycle统计数据
    cycle_stats: Dict[str, Dict[str, int]] = field(
        default_factory=lambda: {
            "req": {"empty": 0, "valid": 0, "T0": 0, "T1": 0, "T2": 0, "ITag": 0, "bytes": 0},
            "rsp": {"empty": 0, "valid": 0, "T0": 0, "T1": 0, "T2": 0, "ITag": 0, "bytes": 0},
            "data": {"empty": 0, "valid": 0, "T0": 0, "T1": 0, "T2": 0, "ITag": 0, "bytes": 0},
        }
    )

    # 总周期数
    total_cycles: int = 0

    # 观测点信息
    observer_info: Dict[str, str] = field(default_factory=dict)

    def reset_stats(self) -> None:
        """重置统计数据"""
        for channel in ["req", "rsp", "data"]:
            self.cycle_stats[channel] = {"empty": 0, "valid": 0, "T0": 0, "T1": 0, "T2": 0, "ITag": 0, "bytes": 0}
        self.total_cycles = 0
        self.observer_info.clear()

    def record_slot_state(self, channel: str, slot: Optional["CrossRingSlot"]) -> None:
        """记录通过观测点的slot状态"""
        if slot is None:
            self.cycle_stats[channel]["empty"] += 1
        else:
            # 检查slot是否真的包含有效的flit
            if hasattr(slot, "is_occupied") and slot.is_occupied and slot.flit is not None:
                # 有效的flit传输
                self.cycle_stats[channel]["valid"] += 1

                # 统计ETag状态
                if hasattr(slot, "etag_priority") and slot.etag_priority:
                    etag_value = slot.etag_priority.value if hasattr(slot.etag_priority, "value") else str(slot.etag_priority)
                    if etag_value in ["T0", "T1", "T2"]:
                        self.cycle_stats[channel][etag_value] += 1

                # 统计ITag状态
                if hasattr(slot, "itag_reserved") and slot.itag_reserved:
                    self.cycle_stats[channel]["ITag"] += 1

                # 统计字节数 - 每个flit固定128字节
                self.cycle_stats[channel]["bytes"] += 128  # 每个flit固定128字节
            else:
                # 空slot，即使slot对象存在但没有有效flit
                self.cycle_stats[channel]["empty"] += 1

    def increment_cycle(self) -> None:
        """增加周期计数"""
        self.total_cycles += 1


@dataclass
class CrossRingSlot(LinkSlot):
    """
    CrossRing Slot实现，继承LinkSlot

    Slot是环路上传输的基本载体，包含四部分：
    1. Valid位: 标记是否载有有效Flit
    2. I-Tag: 注入预约信息(预约状态、方向、预约者ID)
    3. E-Tag: 弹出优先级信息(T0/T1/T2优先级)
    4. Flit: 实际传输的数据
    """

    # CrossRing特有的槽位内容
    valid: bool = False  # Valid位

    # I-Tag信息 (注入预约机制)
    itag_reserved: bool = False  # 是否被预约
    itag_direction: Optional[str] = None  # 预约方向(TR/TL/TU/TD)
    itag_reserver_id: Optional[int] = None  # 预约者节点ID

    # E-Tag信息 (弹出优先级机制)
    etag_marked: bool = False  # 是否被E-Tag标记
    etag_priority: PriorityLevel = PriorityLevel.T2  # T0/T1/T2优先级
    etag_direction: Optional[str] = None  # 标记方向

    # 额外的计数器
    starvation_counter: int = 0

    # 重写flit类型提示以支持CrossRingFlit
    flit: Optional["CrossRingFlit"] = None

    def __post_init__(self):
        """初始化后处理"""
        # 不调用父类的post_init，避免设置is_occupied字段冲突
        # 在CrossRing中，valid字段控制占用状态
        if self.flit is not None:
            self.valid = True

    @property
    def is_occupied(self) -> bool:
        """检查slot是否被占用 - CrossRing使用valid字段"""
        return self.valid and self.flit is not None

    @is_occupied.setter
    def is_occupied(self, value: bool) -> None:
        """设置占用状态 - 为了与父类兼容"""
        self.valid = value

    @property
    def is_available(self) -> bool:
        """检查slot是否可用(空闲且未被预约)"""
        return not self.is_occupied and not self.itag_reserved

    @property
    def is_reserved(self) -> bool:
        """检查slot是否被I-Tag预约"""
        return self.itag_reserved

    def assign_flit(self, flit: "CrossRingFlit") -> bool:
        """
        分配Flit到空闲Slot，重写父类方法以支持CrossRing特有逻辑

        Args:
            flit: 要分配的flit

        Returns:
            是否成功分配
        """
        if self.is_occupied:
            return False

        # CrossRing特有的valid字段设置
        self.valid = True
        self.flit = flit
        self.wait_cycles = 0
        self.starvation_counter = 0

        # 清除I-Tag预约状态
        self.clear_itag()

        return True

    def release_flit(self) -> Optional["CrossRingFlit"]:
        """
        从Slot释放Flit，重写父类方法以支持CrossRing特有逻辑

        Returns:
            被释放的flit，如果没有则返回None
        """
        if not self.is_occupied:
            return None

        released_flit = self.flit

        # 清空Slot - CrossRing特有的valid字段
        self.valid = False
        self.flit = None
        self.wait_cycles = 0
        self.starvation_counter = 0

        # 清除E-Tag标记
        self.clear_etag()

        return released_flit

    def reserve_itag(self, reserver_id: int, direction: str) -> bool:
        """
        I-Tag预约Slot

        Args:
            reserver_id: 预约者节点ID
            direction: 预约方向

        Returns:
            是否成功预约
        """
        if self.itag_reserved or self.is_occupied:
            return False

        self.itag_reserved = True
        self.itag_reserver_id = reserver_id
        self.itag_direction = direction

        return True

    def clear_itag(self) -> None:
        """清除I-Tag预约"""
        self.itag_reserved = False
        self.itag_reserver_id = None
        self.itag_direction = None

    def mark_etag(self, priority: PriorityLevel, direction: str) -> None:
        """
        设置E-Tag标记

        Args:
            priority: E-Tag优先级
            direction: 标记方向
        """
        self.etag_marked = True
        self.etag_priority = priority
        self.etag_direction = direction

    def clear_etag(self) -> None:
        """清除E-Tag标记"""
        self.etag_marked = False
        self.etag_priority = PriorityLevel.T2
        self.etag_direction = None

    def get_slot_info(self) -> Dict[str, Any]:
        """
        获取Slot完整信息

        Returns:
            Slot状态信息字典
        """
        return {
            "slot_id": self.slot_id,
            "cycle": self.cycle,
            "channel": self.channel,
            "valid": self.valid,
            "occupied": self.is_occupied,
            "available": self.is_available,
            "wait_cycles": self.wait_cycles,
            "starvation_counter": self.starvation_counter,
            "itag_info": {
                "reserved": self.itag_reserved,
                "reserver_id": self.itag_reserver_id,
                "direction": self.itag_direction,
            },
            "etag_info": {
                "marked": self.etag_marked,
                "priority": self.etag_priority.value if self.etag_priority else None,
                "direction": self.etag_direction,
            },
            "flit_info": {
                "flit_id": self.flit.flit_id if self.flit else None,
                "packet_id": self.flit.packet_id if self.flit else None,
                "destination": self.flit.destination if self.flit else None,
            },
        }


class RingSlice:
    """
    Ring Slice组件 - 环路传输的基础单元

    按照Cross Ring Spec v2.0定义，Ring Slice是构成环路的最基本单元，
    实现完美流水线：每拍都必须传输一个slot（可能是空slot），不允许断流。
    使用固定深度2的PipelinedFIFO确保1周期固定延迟。
    """
    
    # 全局slot计数器，用于生成唯一slot_id
    _global_slot_counter = 0

    def __init__(self, slice_id: str, ring_type: str, position: int, num_channels: int = 3):
        """
        初始化Ring Slice

        Args:
            slice_id: Ring Slice标识符
            ring_type: 环路类型 ("horizontal" or "vertical")
            position: 在环路中的位置
            num_channels: 通道数量(req/rsp/data)
        """
        self.slice_id = slice_id
        self.ring_type = ring_type
        self.position = position
        self.num_channels = num_channels

        # 创建PipelinedFIFO用于传输slot对象
        self.internal_pipelines: Dict[str, PipelinedFIFO] = {}
        
        for channel in ["req", "rsp", "data"]:
            # 创建FIFO
            fifo = PipelinedFIFO(f"{slice_id}_{channel}_pipeline", depth=2)
            
            # 为每个通道创建一个初始slot并放入FIFO
            # 使用全局唯一的slot_id，不绑定到特定slice
            RingSlice._global_slot_counter += 1
            slot_id = f"slot_{channel}_{RingSlice._global_slot_counter}"
            direction = Direction.TR if ring_type == "horizontal" else Direction.TU  # 默认方向
            
            initial_slot = CrossRingSlot(
                slot_id=slot_id,
                cycle=0,
                direction=direction,
                channel=channel,
                valid=False,  # 初始为空slot
                flit=None
            )
            
            # 将slot放入FIFO的输出寄存器
            fifo.output_register = initial_slot
            fifo.output_valid = True
            
            self.internal_pipelines[channel] = fifo

        # 上下游连接
        self.upstream_slice: Optional["RingSlice"] = None
        self.downstream_slice: Optional["RingSlice"] = None

        # 统计信息
        self.stats = {
            "slots_received": {"req": 0, "rsp": 0, "data": 0},
            "slots_transmitted": {"req": 0, "rsp": 0, "data": 0},
            "empty_cycles": {"req": 0, "rsp": 0, "data": 0},
            "total_cycles": 0,
        }

    # ========== 完美流水线接口 ==========

    def can_accept_input(self, channel: str) -> bool:
        """
        完美流水线总是能接受输入（强制不断流）

        Args:
            channel: 通道类型 ("req", "rsp", "data")

        Returns:
            总是返回True，保证完美流水线
        """
        return channel in self.internal_pipelines

    def write_input(self, slot: Optional[CrossRingSlot], channel: str) -> bool:
        """
        从上游或CrossPoint写入slot到指定通道（强制写入）
        
        直接传输slot对象，不复制内容

        Args:
            slot: 要写入的slot对象（可能携带flit或为空）
            channel: 通道类型

        Returns:
            总是成功，保证完美流水线
        """
        if channel not in self.internal_pipelines:
            return False

        fifo = self.internal_pipelines[channel]

        # 完美流水线：直接传输slot对象
        if fifo.can_accept_input():
            success = fifo.write_input(slot)
        else:
            # FIFO满时直接写入到内部队列（覆盖最早的数据）
            if len(fifo.internal_queue) >= fifo.internal_queue.maxlen:
                fifo.internal_queue.popleft()
            fifo.internal_queue.append(slot)
            success = True

        # 统计
        if slot and slot.is_occupied:
            self.stats["slots_received"][channel] += 1

        return True

    def can_provide_output(self, channel: str) -> bool:
        """
        完美流水线总是有输出（强制输出，即使是None）

        Args:
            channel: 通道类型

        Returns:
            总是返回True，保证完美流水线
        """
        return channel in self.internal_pipelines

    def peek_output(self, channel: str) -> Optional[CrossRingSlot]:
        """
        查看要输出给下游的slot（不移除）

        Args:
            channel: 通道类型

        Returns:
            输出slot或None
        """
        if channel not in self.internal_pipelines:
            return None
        return self.internal_pipelines[channel].peek_output()

    def read_output(self, channel: str) -> Optional[CrossRingSlot]:
        """
        读取并移除输出slot（给下游slice）

        完美流水线需要绕过PipelinedFIFO的"每周期只能读一次"限制

        Args:
            channel: 通道类型

        Returns:
            输出slot或None
        """
        if channel not in self.internal_pipelines:
            return None

        fifo = self.internal_pipelines[channel]

        # 绕过PipelinedFIFO的read_this_cycle限制，直接读取输出寄存器
        if fifo.output_valid:
            slot = fifo.output_register
            # 手动触发FIFO的读取逻辑，但不受read_this_cycle限制
            fifo.read_this_cycle = True  # 设置标志以便update阶段正确处理

            if slot:
                self.stats["slots_transmitted"][channel] += 1
            else:
                self.stats["empty_cycles"][channel] += 1
            return slot
        else:
            self.stats["empty_cycles"][channel] += 1
            return None

    def peek_current_slot(self, channel: str) -> Optional[CrossRingSlot]:
        """
        查看当前正在处理的slot（给CrossPoint使用）

        Args:
            channel: 通道类型

        Returns:
            当前slot或None
        """
        # 使用PipelinedFIFO的peek_output作为current_slot
        return self.peek_output(channel)

    def can_accept_slot_or_has_reserved_slot(self, channel: str, reserver_node_id: int) -> bool:
        """
        检查是否能接受slot或已有本节点预约的slot（用于I-Tag机制）

        Args:
            channel: 通道类型
            reserver_node_id: 预约者节点ID

        Returns:
            是否可以注入
        """
        # 检查是否有被指定节点预约的slot（优先级最高）
        current_slot = self.peek_current_slot(channel)

        # 完美流水线总是能接受输入
        return True

    def write_slot_or_modify_reserved(self, slot: CrossRingSlot, channel: str, reserver_node_id: int) -> bool:
        """
        写入slot或修改预约的slot（用于I-Tag机制）

        Args:
            slot: 要写入的slot对象
            channel: 通道类型
            reserver_node_id: 预约者节点ID

        Returns:
            是否成功
        """
        # 检查是否有被指定节点预约的slot
        current_slot = self.peek_current_slot(channel)
        
        if current_slot and current_slot.is_reserved and current_slot.itag_reserver_id == reserver_node_id:
            # 直接修改预约slot的内容（不通过FIFO，因为slot位置不变）
            if slot.flit:
                current_slot.assign_flit(slot.flit)
            current_slot.clear_itag()  # 清除预约标记
            return True

        # 使用标准接口写入新slot
        return self.write_input(slot, channel)

    def step_compute_phase(self, cycle: int) -> None:
        """
        计算阶段：更新内部PipelinedFIFO的compute阶段

        这是两阶段执行模型的第一阶段，利用PipelinedFIFO的成熟两阶段逻辑

        Args:
            cycle: 当前周期
        """
        # 更新内部PipelinedFIFO的compute阶段
        for channel in ["req", "rsp", "data"]:
            self.internal_pipelines[channel].step_compute_phase(cycle)

    def step_update_phase(self, cycle: int) -> None:
        """
        更新阶段：执行完美流水线传输

        这是两阶段执行模型的第二阶段，强制每拍都传输slot到下游
        关键：先传输，再更新FIFO状态，避免数据跳跃

        Args:
            cycle: 当前周期
        """
        self.stats["total_cycles"] += 1

        # 1. 先传输到下游（基于当前FIFO输出状态）
        if self.downstream_slice:
            for channel in ["req", "rsp", "data"]:
                fifo = self.internal_pipelines[channel]

                # 基于当前输出状态进行传输
                if fifo.output_valid and fifo.output_register:
                    current_slot = fifo.output_register

                    # 更新slot的位置信息（如果有有效flit）
                    if current_slot.flit:
                        current_slot.flit.current_slice_index = self.downstream_slice.position
                        current_slot.flit.current_position = self.downstream_slice.position
                        current_slot.flit.flit_position = "Ring_slice"

                    # 直接传输slot对象到下游
                    success = self.downstream_slice.write_input(current_slot, channel)
                    if success:
                        if current_slot.is_occupied:
                            self.stats["slots_transmitted"][channel] += 1
                        else:
                            self.stats["empty_cycles"][channel] += 1

                    # 设置读取标志
                    fifo.read_this_cycle = True
                else:
                    # 没有有效输出时传输None保持流水线
                    success = self.downstream_slice.write_input(None, channel)
                    if success:
                        self.stats["empty_cycles"][channel] += 1

        # 2. 然后更新内部PipelinedFIFO状态（这会移动数据）
        for channel in ["req", "rsp", "data"]:
            self.internal_pipelines[channel].step_update_phase()

    def peek_output_slot(self, channel: str) -> Optional[CrossRingSlot]:
        """
        查看输出槽的内容(不移除) - 兼容接口

        Args:
            channel: 通道类型

        Returns:
            输出槽的内容
        """
        # 使用新的peek_output接口
        return self.peek_output(channel)

    def get_ring_slice_status(self) -> Dict[str, Any]:
        """
        获取Ring Slice状态信息，集成PipelinedFIFO的详细状态

        Returns:
            状态信息字典
        """
        return {
            "slice_id": self.slice_id,
            "ring_type": self.ring_type,
            "position": self.position,
            # 使用新的接口获取当前slot信息
            "current_slots": {channel: slot.slot_id if slot else None for channel in ["req", "rsp", "data"] for slot in [self.peek_current_slot(channel)]},
            # 集成统计信息
            "stats": self.get_comprehensive_stats(),
        }

    def get_comprehensive_stats(self) -> Dict[str, Any]:
        """
        获取综合统计信息，包括RingSlice和PipelinedFIFO的统计

        Returns:
            综合统计信息
        """
        return {
            "ring_slice_stats": self.stats.copy(),
            "pipeline_stats": {channel: fifo.get_statistics() for channel, fifo in self.internal_pipelines.items()},
            "current_occupancy": {channel: len(fifo) for channel, fifo in self.internal_pipelines.items()},
            "flow_control_status": {channel: {"can_accept": self.can_accept_input(channel), "can_provide": self.can_provide_output(channel)} for channel in ["req", "rsp", "data"]},
        }

    def reset_stats(self) -> None:
        """重置统计信息，包括PipelinedFIFO的统计"""
        # 重置RingSlice统计
        for channel in ["req", "rsp", "data"]:
            self.stats["slots_received"][channel] = 0
            self.stats["slots_transmitted"][channel] = 0
            self.stats["empty_cycles"][channel] = 0
        self.stats["total_cycles"] = 0

        # 重置PipelinedFIFO统计（如果存在reset方法）
        for channel, fifo in self.internal_pipelines.items():
            if hasattr(fifo, "reset_stats"):
                fifo.reset_stats()
            elif hasattr(fifo, "stats"):
                # 手动重置FIFO统计
                fifo.stats = fifo.stats.__class__()


class CrossRingLink(BaseLink):
    """
    CrossRing链路类 - 继承BaseLink，实现CrossRing特定功能

    职责：
    1. 管理Ring Slice链组成的链路
    2. 实现CrossRing特有的ETag/ITag机制
    3. 提供CrossRing特定的slot管理和仲裁
    4. 与CrossPoint协作处理复杂的anti-starvation逻辑
    """

    def __init__(self, link_id: str, source_node: int, dest_node: int, direction: Direction, config: CrossRingConfig, num_slices: int = 8):
        """
        初始化CrossRing链路

        Args:
            link_id: 链路标识符
            source_node: 源节点ID
            dest_node: 目标节点ID
            direction: 链路方向
            config: CrossRing配置
            num_slices: Ring Slice数量
        """
        # 调用父类构造函数
        super().__init__(link_id, source_node, dest_node, num_slices)

        # CrossRing特有属性
        self.direction = direction
        self.config = config
        self.num_slices = num_slices

        # Ring Slice链 - 构成链路的基础单元
        self.ring_slices: Dict[str, List[RingSlice]] = {"req": [], "rsp": [], "data": []}

        # 初始化Ring Slice链
        self._initialize_ring_slices()

        # 初始化带宽统计跟踪器
        self.bandwidth_tracker = LinkBandwidthTracker()

        # 扩展父类统计信息，添加CrossRing特有的统计
        self.stats.update(
            {
                "utilization": {"req": 0.0, "rsp": 0.0, "data": 0.0},
                "total_cycles": 0,
            }
        )

    def _initialize_ring_slices(self) -> None:
        """初始化Ring Slice链"""
        ring_type = "horizontal" if self.direction in [Direction.TR, Direction.TL] else "vertical"

        for channel in ["req", "rsp", "data"]:
            self.ring_slices[channel] = []
            for i in range(self.num_slices):
                slice_id = f"{self.link_id}_{channel}_slice_{i}"
                ring_slice = RingSlice(slice_id, ring_type, i, 3)
                self.ring_slices[channel].append(ring_slice)

            # 建立下游连接：slice[i] -> slice[i+1]
            for i in range(len(self.ring_slices[channel]) - 1):
                current_slice = self.ring_slices[channel][i]
                next_slice = self.ring_slices[channel][i + 1]
                current_slice.downstream_slice = next_slice
                next_slice.upstream_slice = current_slice

    def get_ring_slice(self, channel: str, position: int) -> Optional[RingSlice]:
        """
        获取指定位置的Ring Slice

        Args:
            channel: 通道类型
            position: 位置索引

        Returns:
            Ring Slice实例，如果不存在则返回None
        """
        if channel not in self.ring_slices:
            return None

        slices = self.ring_slices[channel]
        if 0 <= position < len(slices):
            return slices[position]

        return None

    def step_compute_phase(self, cycle: int) -> None:
        """
        计算阶段：让所有Ring Slice执行compute阶段

        Args:
            cycle: 当前周期
        """
        # 处理每个通道的传输计算
        for channel in ["req", "rsp", "data"]:
            self._step_channel_compute(channel, cycle)

    def step_update_phase(self, cycle: int) -> None:
        """
        更新阶段：让所有Ring Slice执行update阶段

        优化：移除单独的下游传输阶段，传输在slice的update阶段完成

        Args:
            cycle: 当前周期
        """
        self.stats["total_cycles"] += 1

        # 处理每个通道的传输更新（现在包含下游传输）
        for channel in ["req", "rsp", "data"]:
            self._step_channel_update(channel, cycle)

        # 在固定观测点收集带宽统计数据（在处理传输之前）
        self._collect_bandwidth_stats(cycle)

    def _collect_bandwidth_stats(self, cycle: int) -> None:
        """在链路末端观测点收集带宽统计数据"""
        # 增加周期计数
        self.bandwidth_tracker.increment_cycle()

        # 对每个通道的观测点slice进行统计
        for channel in ["data"]:
            slices = self.ring_slices.get(channel, [])
            if not slices:
                continue

            # 使用最后一个slice作为观测点（更能反映链路实际传输情况）
            observer_position = len(slices) - 1
            observer_slice = self.get_ring_slice(channel, observer_position)

            # 记录观测点信息（仅在第一次记录）
            if channel not in self.bandwidth_tracker.observer_info:
                self.bandwidth_tracker.observer_info[channel] = f"slice[{observer_position}]/{len(slices)}"

            if observer_slice is not None:
                # 获取当前cycle通过观测点的slot
                # 观测slice的当前slot状态（实际传输的数据）
                current_slot = observer_slice.peek_current_slot(channel)

                # 记录slot状态到带宽跟踪器
                self.bandwidth_tracker.record_slot_state(channel, current_slot)

    def _step_channel_compute(self, channel: str, cycle: int) -> None:
        """
        处理单个通道的计算阶段

        Args:
            channel: 通道类型
            cycle: 当前周期
        """
        slices = self.ring_slices[channel]

        # 让所有Ring Slice执行compute阶段
        for ring_slice in slices:
            ring_slice.step_compute_phase(cycle)

    def _step_channel_update(self, channel: str, cycle: int) -> None:
        """
        处理单个通道的更新阶段

        Args:
            channel: 通道类型
            cycle: 当前周期
        """
        slices = self.ring_slices[channel]

        # 让所有Ring Slice执行update阶段
        for ring_slice in slices:
            ring_slice.step_update_phase(cycle)

    def get_link_status(self) -> Dict[str, Any]:
        """
        获取链路状态信息

        Returns:
            状态信息字典
        """
        return {
            "link_id": self.link_id,
            "source_node": self.source_node,
            "dest_node": self.dest_node,
            "direction": self.direction.value,
            "num_slices": self.num_slices,
            "ring_slice_status": {channel: [slice.get_ring_slice_status() for slice in slices] for channel, slices in self.ring_slices.items()},
            "stats": self.stats.copy(),
        }

    def get_link_performance_metrics(self) -> Dict[str, Any]:
        """计算链路性能指标"""
        metrics = {}

        for channel in ["req", "rsp", "data"]:
            stats = self.bandwidth_tracker.cycle_stats[channel]
            total_cycles = self.bandwidth_tracker.total_cycles

            if total_cycles > 0:
                # 计算带宽 (GB/s)
                total_time_ns = total_cycles / (self.config.basic_config.NETWORK_FREQUENCY)
                bandwidth_gbps = stats["bytes"] / total_time_ns if total_time_ns > 0 else 0.0

                # 计算利用率
                utilization = stats["valid"] / total_cycles
                idle_rate = stats["empty"] / total_cycles

                # 计算ETag分布
                etag_distribution = {
                    "T0_rate": stats["T0"] / total_cycles,
                    "T1_rate": stats["T1"] / total_cycles,
                    "T2_rate": stats["T2"] / total_cycles,
                    "ITag_rate": stats["ITag"] / total_cycles,
                }

                metrics[channel] = {
                    "bandwidth_gbps": bandwidth_gbps,
                    "utilization": utilization,
                    "idle_rate": idle_rate,
                    "total_bytes": stats["bytes"],
                    "valid_slots": stats["valid"],
                    "empty_slots": stats["empty"],
                    "etag_distribution": etag_distribution,
                }
            else:
                # 没有数据的情况
                metrics[channel] = {
                    "bandwidth_gbps": 0.0,
                    "utilization": 0.0,
                    "idle_rate": 0.0,
                    "total_bytes": 0,
                    "valid_slots": 0,
                    "empty_slots": 0,
                    "etag_distribution": {"T0_rate": 0.0, "T1_rate": 0.0, "T2_rate": 0.0, "ITag_rate": 0.0},
                }

        return metrics

    def print_link_bandwidth_summary(self) -> None:
        """打印链路带宽汇总信息"""
        metrics = self.get_link_performance_metrics()

        print(f"📊 链路 {self.link_id} ({self.source_node}→{self.dest_node}) 带宽统计:")

        # 打印观测点信息
        if hasattr(self.bandwidth_tracker, "observer_info") and self.bandwidth_tracker.observer_info:
            print(f"   观测点信息: {self.bandwidth_tracker.observer_info}")

        for channel, data in metrics.items():
            print(f"  {channel}: {data['bandwidth_gbps']:.2f}GB/s, 利用率{data['utilization']:.1%}, 空载率{data['idle_rate']:.1%}")

    def reset_stats(self) -> None:
        """重置统计信息，重写父类方法以添加CrossRing特有的重置"""
        # 调用父类的reset_stats（如果有的话）
        if hasattr(super(), "reset_stats"):
            super().reset_stats()

        # 重置CrossRing特有的统计
        for channel in ["req", "rsp", "data"]:
            self.stats["utilization"][channel] = 0.0

            # 重置Ring Slice统计
            for ring_slice in self.ring_slices[channel]:
                ring_slice.reset_stats()

        self.stats["total_cycles"] = 0

        # 重置带宽统计跟踪器
        self.bandwidth_tracker.reset_stats()

    def get_slots(self, channel: str) -> List[CrossRingSlot]:
        """
        获取指定通道的所有slots（从所有slice中收集）

        Args:
            channel: 通道类型

        Returns:
            CrossRingSlot列表
        """
        slots = []
        if channel in self.ring_slices:
            for ring_slice in self.ring_slices[channel]:
                # 获取当前正在处理的slot（输出位置）
                current_slot = ring_slice.peek_current_slot(channel)
                if current_slot:
                    slots.append(current_slot)

                # 获取内部队列中的所有slots（使用PipelinedFIFO的接口）
                if channel in ring_slice.internal_pipelines:
                    pipeline = ring_slice.internal_pipelines[channel]
                    # 获取内部队列中的所有slots
                    internal_slots = list(pipeline.internal_queue)
                    slots.extend(internal_slots)
        return slots

    def check_all_slices_have_slots(self) -> Dict[str, Any]:
        """
        检查所有slice中是否有slot的完整报告

        Returns:
            检查结果字典，包含详细信息和统计
        """
        report = {
            "total_slices": 0,
            "slices_with_slots": 0,
            "slices_without_slots": 0,
            "channels": {},
            "missing_slots": [],
            "slot_distribution": {},
            "summary": ""
        }

        for channel in ["req", "rsp", "data"]:
            channel_report = {
                "total_slices": 0,
                "slices_with_slots": 0,
                "slices_without_slots": 0,
                "missing_slice_positions": [],
                "slot_details": []
            }

            if channel in self.ring_slices:
                slices = self.ring_slices[channel]
                channel_report["total_slices"] = len(slices)
                
                for i, ring_slice in enumerate(slices):
                    has_slot = False
                    slot_info = {
                        "slice_id": ring_slice.slice_id,
                        "position": i,
                        "has_output_slot": False,
                        "output_slot_id": None,
                        "internal_queue_slots": 0,
                        "total_slots": 0
                    }

                    # 检查输出寄存器中的slot
                    if channel in ring_slice.internal_pipelines:
                        pipeline = ring_slice.internal_pipelines[channel]
                        
                        # 检查输出寄存器
                        if pipeline.output_valid and pipeline.output_register:
                            slot_info["has_output_slot"] = True
                            slot_info["output_slot_id"] = pipeline.output_register.slot_id
                            has_slot = True
                            slot_info["total_slots"] += 1

                        # 检查内部队列
                        internal_slots = list(pipeline.internal_queue)
                        slot_info["internal_queue_slots"] = len(internal_slots)
                        slot_info["total_slots"] += len(internal_slots)
                        
                        if internal_slots:
                            has_slot = True

                    if has_slot:
                        channel_report["slices_with_slots"] += 1
                    else:
                        channel_report["slices_without_slots"] += 1
                        channel_report["missing_slice_positions"].append(i)
                        report["missing_slots"].append({
                            "channel": channel,
                            "slice_id": ring_slice.slice_id,
                            "position": i
                        })

                    channel_report["slot_details"].append(slot_info)

            report["channels"][channel] = channel_report
            report["total_slices"] += channel_report["total_slices"]
            report["slices_with_slots"] += channel_report["slices_with_slots"]
            report["slices_without_slots"] += channel_report["slices_without_slots"]

        # 生成汇总信息
        if report["slices_without_slots"] == 0:
            report["summary"] = f"✅ 所有{report['total_slices']}个slice都有slot"
        else:
            report["summary"] = f"❌ {report['slices_without_slots']}/{report['total_slices']}个slice缺少slot"

        # 统计slot分布
        for channel in ["req", "rsp", "data"]:
            total_slots = sum(detail["total_slots"] for detail in report["channels"][channel]["slot_details"])
            report["slot_distribution"][channel] = total_slots

        return report

    def print_slot_check_report(self) -> None:
        """打印slot检查报告"""
        report = self.check_all_slices_have_slots()
        
        print(f"📊 链路 {self.link_id} Slot检查报告:")
        print(f"   {report['summary']}")
        print(f"   总Slice数: {report['total_slices']}, 有Slot: {report['slices_with_slots']}, 无Slot: {report['slices_without_slots']}")
        
        # 按通道详细报告
        for channel, channel_data in report["channels"].items():
            print(f"   {channel}通道: {channel_data['slices_with_slots']}/{channel_data['total_slices']}个slice有slot")
            if channel_data["missing_slice_positions"]:
                print(f"     缺失位置: {channel_data['missing_slice_positions']}")
        
        # Slot分布统计
        print(f"   Slot分布: {report['slot_distribution']}")
        
        # 如果有缺失，详细列出
        if report["missing_slots"]:
            print("   缺失详情:")
            for missing in report["missing_slots"]:
                print(f"     - {missing['channel']}通道 {missing['slice_id']} (位置{missing['position']})")

    def verify_slot_continuity(self) -> Dict[str, bool]:
        """
        验证slot的连续性 - 检查环路中是否有断链
        
        Returns:
            每个通道的连续性检查结果
        """
        results = {}
        
        for channel in ["req", "rsp", "data"]:
            is_continuous = True
            
            if channel in self.ring_slices:
                slices = self.ring_slices[channel]
                
                for i, ring_slice in enumerate(slices):
                    # 检查每个slice是否有slot可以传输
                    if not ring_slice.can_provide_output(channel):
                        is_continuous = False
                        print(f"❌ {channel}通道 slice[{i}] 无法提供输出")
                    
                    # 检查下游连接
                    if ring_slice.downstream_slice:
                        if not ring_slice.downstream_slice.can_accept_input(channel):
                            is_continuous = False
                            print(f"❌ {channel}通道 slice[{i}] 下游无法接受输入")
            
            results[channel] = is_continuous
            
        return results

    # ========== BaseLink抽象方法实现 ==========

    def _get_link_direction(self) -> Direction:
        """获取链路方向"""
        return self.direction

    def _process_slot_transmission(self, cycle: int) -> None:
        """
        处理slot传输逻辑

        Args:
            cycle: 当前周期
        """
        # 调用现有的step方法
        self.step(cycle)
