"""
CrossRing链路实现，继承BaseLink，实现CrossRing特有的ETag/ITag机制。

本模块实现了CrossRing拓扑中链路的具体逻辑，包括：
- CrossRing特有的ETag升级规则（双向vs单向）
- ITag反饿死机制
- 基于配置的拥塞控制阈值
- Ring Bridge的slice调度逻辑
"""

from typing import Dict, List, Any, Optional, Tuple
import logging
from dataclasses import dataclass
from enum import Enum

from ..base.link import BaseLink, LinkSlot, BasicPriority, BasicDirection
from ..base.flit import BaseFlit
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

    def increment_wait(self) -> None:
        """增加等待周期计数"""
        if self.is_occupied:
            self.wait_cycles += 1
            if not self.etag_marked:
                self.starvation_counter += 1

    def should_trigger_itag(self, threshold: int) -> bool:
        """
        检查是否应该触发I-Tag预约

        Args:
            threshold: 饿死阈值

        Returns:
            是否应该触发I-Tag
        """
        return self.starvation_counter >= threshold and not self.itag_reserved

    def should_upgrade_etag(self, failed_attempts: int) -> PriorityLevel:
        """
        检查是否应该升级E-Tag优先级

        Args:
            failed_attempts: 下环失败次数

        Returns:
            建议的新优先级
        """
        if failed_attempts == 1 and self.etag_priority == PriorityLevel.T2:
            return PriorityLevel.T1
        elif failed_attempts >= 2 and self.etag_priority == PriorityLevel.T1:
            # 只有TL/TU方向可以升级到T0
            if self.etag_direction in ["TL", "TU"]:
                return PriorityLevel.T0

        return self.etag_priority

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
    本质上是一组寄存器，负责Slot的逐跳传输。
    """

    def __init__(self, slice_id: str, ring_type: str, position: int, num_channels: int = 3, logger: Optional[logging.Logger] = None):
        """
        初始化Ring Slice

        Args:
            slice_id: Ring Slice标识符
            ring_type: 环路类型 ("horizontal" or "vertical")
            position: 在环路中的位置
            num_channels: 通道数量(req/rsp/data)
            logger: 日志记录器
        """
        self.slice_id = slice_id
        self.ring_type = ring_type
        self.position = position
        self.num_channels = num_channels
        self.logger = logger or logging.getLogger(__name__)

        # 当前存储的Slots - 每个通道一个
        self.current_slots: Dict[str, Optional[CrossRingSlot]] = {"req": None, "rsp": None, "data": None}

        # 输入/输出缓存 - 用于流水线传输
        self.input_buffer: Dict[str, Optional[CrossRingSlot]] = {"req": None, "rsp": None, "data": None}
        self.output_buffer: Dict[str, Optional[CrossRingSlot]] = {"req": None, "rsp": None, "data": None}

        # 上下游连接
        self.upstream_slice: Optional["RingSlice"] = None
        self.downstream_slice: Optional["RingSlice"] = None

        # 统计信息
        self.stats = {"slots_received": {"req": 0, "rsp": 0, "data": 0}, "slots_transmitted": {"req": 0, "rsp": 0, "data": 0}, "empty_cycles": {"req": 0, "rsp": 0, "data": 0}, "total_cycles": 0}

    def receive_slot(self, slot: Optional[CrossRingSlot], channel: str) -> bool:
        """
        从上游接收Slot

        Args:
            slot: 接收的Slot，可以为None(空槽)
            channel: 通道类型

        Returns:
            是否成功接收
        """
        if channel not in self.input_buffer:
            return False

        # 检查输入缓存是否已满
        # 修复：允许有效slot覆盖空slot（无flit的slot）
        if self.input_buffer[channel] is not None:
            existing_slot = self.input_buffer[channel]
            # 如果是空slot且新slot有效，允许覆盖
            if (not existing_slot.is_occupied and 
                not hasattr(existing_slot, 'flit') or existing_slot.flit is None) and \
               (slot is not None and slot.is_occupied and slot.flit is not None):
                # 允许覆盖空slot
                pass
            else:
                return False  # 输入缓存已满，无法接收

        self.input_buffer[channel] = slot

        if slot is not None:
            # 更新slot中flit的位置信息
            if slot.flit is not None:
                slot.flit.current_link_id = self.slice_id
                slot.flit.current_slice_index = self.position
                slot.flit.current_slot_index = slot.slot_id
                slot.flit.current_position = self.position

                # 设置链路源和目标节点信息并格式化位置（从slice_id解析）
                # slice_id格式：
                # 普通链路：link_0_TR_1_req_slice_2 (7个部分)
                # 自环链路：link_0_TR_TL_0_req_slice_2 (8个部分)
                try:
                    parts = self.slice_id.split("_")
                    if len(parts) >= 7:
                        source = int(parts[1])  # 源节点总是在第2个位置
                        
                        # 根据parts数量判断链路类型
                        if len(parts) == 8:  # 自环链路：link_0_TR_TL_0_req_slice_2
                            dest = int(parts[4])  # 目标节点在第5个位置
                        elif len(parts) == 7:  # 普通链路：link_0_TR_1_req_slice_2
                            dest = int(parts[3])  # 目标节点在第4个位置
                        else:
                            # 未知格式，尝试从link_id中提取
                            link_parts = parts[:4]  # 取link_id部分
                            dest = int(link_parts[-1])  # 最后一个数字部分作为目标
                        
                        slot.flit.link_source_node = source
                        slot.flit.link_dest_node = dest
                        # 使用位置特定格式：source->dest:slice_index
                        slot.flit.flit_position = f"{source}->{dest}:{self.position}"
                    else:
                        # 如果解析失败，尝试从slice_id中提取节点信息
                        slot.flit.flit_position = f"UNKNOWN_LINK:{self.position}"
                except (ValueError, IndexError) as e:
                    # 解析失败时也尝试提供有意义的位置信息
                    # 调试：显示解析失败的详细信息
                    self.logger.debug(f"slice_id解析失败: {self.slice_id}, 错误: {e}")
                    slot.flit.flit_position = f"PARSE_ERROR:{self.position}"

            self.stats["slots_received"][channel] += 1
            self.logger.debug(f"RingSlice {self.slice_id} 接收到 {channel} 通道的slot {slot.slot_id}")
        else:
            self.stats["empty_cycles"][channel] += 1

        return True

    def transmit_slot(self, channel: str) -> Optional[CrossRingSlot]:
        """
        向下游传输Slot

        Args:
            channel: 通道类型

        Returns:
            传输的Slot，可能为None
        """
        if channel not in self.output_buffer:
            return None

        slot = self.output_buffer[channel]
        self.output_buffer[channel] = None

        if slot is not None:
            self.stats["slots_transmitted"][channel] += 1
            self.logger.debug(f"RingSlice {self.slice_id} 传输 {channel} 通道的slot {slot.slot_id}")

        return slot

    def step_compute_phase(self, cycle: int) -> None:
        """
        计算阶段：确定传输决策，不修改状态
        
        计算哪些slot需要移动，但不执行实际的移动操作
        
        Args:
            cycle: 当前周期
        """
        # 计算传输决策，存储在临时变量中
        # 这里只需要确定传输的可行性，不修改状态
        self._next_cycle = cycle
        
        # 预计算传输决策，但不执行
        self._transfer_plan = {}
        for channel in ["req", "rsp", "data"]:
            self._transfer_plan[channel] = {
                "can_move_to_output": True,  # 当前槽总是可以移动到输出缓存
                "can_move_to_current": True,  # 输入缓存总是可以移动到当前槽
                "can_transmit_downstream": False  # 默认不能传输
            }
            
            # 检查是否可以向下游传输
            # 应该基于current_slots（将要移动到output_buffer的内容）来判断
            if (self.downstream_slice and 
                self.current_slots[channel] is not None and
                self.downstream_slice.input_buffer.get(channel) is None):
                self._transfer_plan[channel]["can_transmit_downstream"] = True

    def step_update_phase(self, cycle: int) -> None:
        """
        更新阶段：基于计算阶段的决策执行状态修改
        
        遵循两阶段执行模型：在单个update周期内完成所有传输
        
        Args:
            cycle: 当前周期
        """
        self.stats["total_cycles"] += 1
        
        # 两阶段模型：同时执行传输和更新操作
        for channel in ["req", "rsp", "data"]:
            current_slot = self.current_slots[channel]
            input_slot = self.input_buffer[channel]
            
            # 第一步：向下游传输当前slot（如果可以）
            downstream_transmitted = False
            if (self._transfer_plan[channel]["can_transmit_downstream"] and
                self.downstream_slice and current_slot is not None):
                
                if self.downstream_slice.receive_slot(current_slot, channel):
                    downstream_transmitted = True
                    self.stats["slots_transmitted"][channel] += 1
            
            # 第二步：同时进行内部移动
            if downstream_transmitted:
                # 当前slot已传输，输入slot移动到当前位置
                self.current_slots[channel] = input_slot
                self.input_buffer[channel] = None
            else:
                # 当前slot未传输，移动到输出缓存，输入slot移动到当前位置
                if current_slot is not None and self._transfer_plan[channel]["can_move_to_output"]:
                    self.output_buffer[channel] = current_slot
                
                if self._transfer_plan[channel]["can_move_to_current"]:
                    self.current_slots[channel] = input_slot
                    self.input_buffer[channel] = None

            # 更新slot的等待时间
            if self.current_slots[channel] is not None:
                self.current_slots[channel].increment_wait()
                self.current_slots[channel].cycle = cycle

    def step_downstream_transmission(self, cycle: int) -> None:
        """
        下游传输阶段：向下游slice传输数据
        
        这个方法应该在所有slice完成update阶段后调用
        
        Args:
            cycle: 当前周期
        """
        for channel in ["req", "rsp", "data"]:
            # Step 4: 向下游传输slot（基于compute阶段的决策）
            if (hasattr(self, '_transfer_plan') and 
                self._transfer_plan[channel]["can_transmit_downstream"] and
                self.downstream_slice and 
                self.output_buffer[channel] is not None):
                
                transmitted_slot = self.output_buffer[channel]
                if self.downstream_slice.receive_slot(transmitted_slot, channel):
                    self.output_buffer[channel] = None
                    self.stats["slots_transmitted"][channel] += 1
                    self.logger.debug(f"RingSlice {self.slice_id} 向下游传输slot {transmitted_slot.slot_id}")


    def peek_current_slot(self, channel: str) -> Optional[CrossRingSlot]:
        """
        查看当前槽的内容(不移除)

        Args:
            channel: 通道类型

        Returns:
            当前槽的内容
        """
        return self.current_slots.get(channel)

    def peek_output_slot(self, channel: str) -> Optional[CrossRingSlot]:
        """
        查看输出槽的内容(不移除)

        Args:
            channel: 通道类型

        Returns:
            输出槽的内容
        """
        return self.output_buffer.get(channel)

    def is_channel_busy(self, channel: str) -> bool:
        """
        检查通道是否繁忙

        Args:
            channel: 通道类型

        Returns:
            通道是否有Slot在传输
        """
        return self.current_slots.get(channel) is not None or self.input_buffer.get(channel) is not None or self.output_buffer.get(channel) is not None

    def get_utilization(self, channel: str) -> float:
        """
        获取通道利用率

        Args:
            channel: 通道类型

        Returns:
            利用率(0.0-1.0)
        """
        if self.stats["total_cycles"] == 0:
            return 0.0

        busy_cycles = (self.stats["slots_received"][channel] + self.stats["slots_transmitted"][channel]) / 2
        return min(1.0, busy_cycles / self.stats["total_cycles"])

    def get_ring_slice_status(self) -> Dict[str, Any]:
        """
        获取Ring Slice状态信息

        Returns:
            状态信息字典
        """
        return {
            "slice_id": self.slice_id,
            "ring_type": self.ring_type,
            "position": self.position,
            "current_slots": {channel: slot.slot_id if slot else None for channel, slot in self.current_slots.items()},
            "channel_busy": {channel: self.is_channel_busy(channel) for channel in ["req", "rsp", "data"]},
            "utilization": {channel: self.get_utilization(channel) for channel in ["req", "rsp", "data"]},
            "stats": self.stats.copy(),
        }

    def reset_stats(self) -> None:
        """重置统计信息"""
        for channel in ["req", "rsp", "data"]:
            self.stats["slots_received"][channel] = 0
            self.stats["slots_transmitted"][channel] = 0
            self.stats["empty_cycles"][channel] = 0
        self.stats["total_cycles"] = 0


class CrossRingLink(BaseLink):
    """
    CrossRing链路类 - 继承BaseLink，实现CrossRing特定功能

    职责：
    1. 管理Ring Slice链组成的链路
    2. 实现CrossRing特有的ETag/ITag机制
    3. 提供CrossRing特定的slot管理和仲裁
    4. 与CrossPoint协作处理复杂的anti-starvation逻辑
    """

    def __init__(self, link_id: str, source_node: int, dest_node: int, direction: Direction, config: CrossRingConfig, num_slices: int = 8, logger: Optional[logging.Logger] = None):
        """
        初始化CrossRing链路

        Args:
            link_id: 链路标识符
            source_node: 源节点ID
            dest_node: 目标节点ID
            direction: 链路方向
            config: CrossRing配置
            num_slices: Ring Slice数量
            logger: 日志记录器
        """
        # 调用父类构造函数
        super().__init__(link_id, source_node, dest_node, num_slices, logger)
        
        # CrossRing特有属性
        self.direction = direction
        self.config = config
        self.num_slices = num_slices

        # Ring Slice链 - 构成链路的基础单元
        self.ring_slices: Dict[str, List[RingSlice]] = {"req": [], "rsp": [], "data": []}

        # 初始化Ring Slice链
        self._initialize_ring_slices()

        # Slot池 - 管理所有的CrossRingSlot
        self.slot_pools: Dict[str, List[CrossRingSlot]] = {"req": [], "rsp": [], "data": []}

        # 初始化Slot池
        self._initialize_slot_pools()

        # 扩展父类统计信息，添加CrossRing特有的统计
        self.stats.update({
            "slots_created": {"req": 0, "rsp": 0, "data": 0},
            "slots_destroyed": {"req": 0, "rsp": 0, "data": 0},
            "utilization": {"req": 0.0, "rsp": 0.0, "data": 0.0},
            "total_cycles": 0,
        })

        self.logger.info(f"CrossRingLink {link_id} 初始化完成: {source_node} -> {dest_node}, 方向: {direction.value}")

    def _initialize_ring_slices(self) -> None:
        """初始化Ring Slice链"""
        ring_type = "horizontal" if self.direction in [Direction.TR, Direction.TL] else "vertical"

        for channel in ["req", "rsp", "data"]:
            self.ring_slices[channel] = []
            for i in range(self.num_slices):
                slice_id = f"{self.link_id}_{channel}_slice_{i}"
                ring_slice = RingSlice(slice_id, ring_type, i, logger=self.logger)
                self.ring_slices[channel].append(ring_slice)

        self.logger.debug(f"初始化 {self.num_slices} 个Ring Slice，类型: {ring_type}")

    def _initialize_slot_pools(self) -> None:
        """初始化Slot池"""
        for channel in ["req", "rsp", "data"]:
            self.slot_pools[channel] = []
            # 预创建一些slot，实际使用时动态分配
            for i in range(self.num_slices * 2):  # 每个slice预分配2个slot
                slot = CrossRingSlot(slot_id=i, cycle=0, direction=BasicDirection.LOCAL, channel=channel)
                self.slot_pools[channel].append(slot)

        self.logger.debug(f"初始化Slot池，每通道 {self.num_slices * 2} 个slot")

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

    def get_available_slot(self, channel: str) -> Optional[CrossRingSlot]:
        """
        获取可用的空闲Slot

        Args:
            channel: 通道类型

        Returns:
            可用的slot，如果没有则返回None
        """
        if channel not in self.slot_pools:
            return None

        for slot in self.slot_pools[channel]:
            if slot.is_available:
                return slot

        # 如果池中没有可用slot，创建新的
        return self._create_new_slot(channel)

    def get_reserved_slot(self, channel: str, reserver_id: int) -> Optional[CrossRingSlot]:
        """
        获取被指定节点预约的Slot

        Args:
            channel: 通道类型
            reserver_id: 预约者ID

        Returns:
            预约的slot，如果没有则返回None
        """
        if channel not in self.slot_pools:
            return None

        for slot in self.slot_pools[channel]:
            if slot.is_reserved and slot.itag_reserver_id == reserver_id:
                return slot

        return None

    def _create_new_slot(self, channel: str) -> CrossRingSlot:
        """
        创建新的Slot

        Args:
            channel: 通道类型

        Returns:
            新创建的slot
        """
        slot_id = len(self.slot_pools[channel])
        new_slot = CrossRingSlot(slot_id=slot_id, cycle=0, channel=channel)

        self.slot_pools[channel].append(new_slot)
        self.stats["slots_created"][channel] += 1

        return new_slot

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
        
        # 注意：下游传输现在在slice的update阶段完成，不需要单独调用

        # 更新利用率统计
        self._update_utilization_stats()

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

    def _step_channel_downstream_transmission(self, channel: str, cycle: int) -> None:
        """
        处理单个通道的下游传输阶段
        
        Args:
            channel: 通道类型
            cycle: 当前周期
        """
        slices = self.ring_slices[channel]
        
        # 让所有Ring Slice执行下游传输
        for ring_slice in slices:
            ring_slice.step_downstream_transmission(cycle)


    def _update_utilization_stats(self) -> None:
        """更新利用率统计"""
        for channel in ["req", "rsp", "data"]:
            slices = self.ring_slices[channel]
            if not slices:
                continue

            # 计算平均利用率
            total_utilization = sum(slice.get_utilization(channel) for slice in slices)
            avg_utilization = total_utilization / len(slices)
            self.stats["utilization"][channel] = avg_utilization

    def inject_slot_to_ring(self, slot: CrossRingSlot, channel: str, position: int = 0) -> bool:
        """
        向环路注入Slot

        Args:
            slot: 要注入的slot
            channel: 通道类型
            position: 注入位置

        Returns:
            是否成功注入
        """
        ring_slice = self.get_ring_slice(channel, position)
        if not ring_slice:
            return False

        return ring_slice.receive_slot(slot, channel)

    def eject_slot_from_ring(self, channel: str, position: int = 0) -> Optional[CrossRingSlot]:
        """
        从环路弹出Slot

        Args:
            channel: 通道类型
            position: 弹出位置

        Returns:
            弹出的slot，如果没有则返回None
        """
        ring_slice = self.get_ring_slice(channel, position)
        if not ring_slice:
            return None

        return ring_slice.transmit_slot(channel)

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
            "slot_pool_status": {
                channel: {
                    "total_slots": len(pools),
                    "available_slots": sum(1 for slot in pools if slot.is_available),
                    "occupied_slots": sum(1 for slot in pools if slot.is_occupied),
                    "reserved_slots": sum(1 for slot in pools if slot.is_reserved),
                }
                for channel, pools in self.slot_pools.items()
            },
            "stats": self.stats.copy(),
        }

    def reset_stats(self) -> None:
        """重置统计信息，重写父类方法以添加CrossRing特有的重置"""
        # 调用父类的reset_stats（如果有的话）
        if hasattr(super(), 'reset_stats'):
            super().reset_stats()
        
        # 重置CrossRing特有的统计
        for channel in ["req", "rsp", "data"]:
            self.stats["slots_created"][channel] = 0
            self.stats["slots_destroyed"][channel] = 0
            self.stats["utilization"][channel] = 0.0

            # 重置Ring Slice统计
            for ring_slice in self.ring_slices[channel]:
                ring_slice.reset_stats()

        self.stats["total_cycles"] = 0

    def get_slots(self, channel: str) -> List[CrossRingSlot]:
        """
        获取指定通道的所有slots（兼容接口）

        Args:
            channel: 通道类型

        Returns:
            CrossRingSlot列表
        """
        return self.slot_pools.get(channel, [])

    # ========== BaseLink抽象方法实现 ==========
    
    def _get_link_direction(self) -> Direction:
        """获取链路方向"""
        return self.direction

    def can_upgrade_etag(self, channel: str, from_level: PriorityLevel, to_level: PriorityLevel) -> bool:
        """
        检查是否可以升级ETag优先级

        Args:
            channel: 通道类型
            from_level: 原优先级
            to_level: 目标优先级

        Returns:
            是否可以升级
        """
        # CrossRing的ETag升级策略
        utilization = self.stats.get("utilization", {}).get(channel, 0.0)
        
        if from_level == PriorityLevel.T2 and to_level == PriorityLevel.T1:
            return utilization > 0.7  # 利用率超过70%可升级T1
        elif from_level == PriorityLevel.T1 and to_level == PriorityLevel.T0:
            return utilization > 0.9  # 利用率超过90%可升级T0
        
        return False

    def should_trigger_itag(self, channel: str, direction: str) -> bool:
        """
        检查是否应该触发ITag

        Args:
            channel: 通道类型
            direction: 方向

        Returns:
            是否应该触发ITag
        """
        # CrossRing的ITag触发策略
        utilization = self.stats.get("utilization", {}).get(channel, 0.0)
        return utilization > 0.8  # 利用率超过80%触发ITag

    def _check_etag_upgrade(self, channel: str, utilization: float, cycle: int) -> None:
        """
        检查ETag升级条件

        Args:
            channel: 通道类型
            utilization: 利用率
            cycle: 当前周期
        """
        # 检查T2到T1升级
        if utilization > 0.7:
            slots = self.slot_pools.get(channel, [])
            for slot in slots:
                if slot.etag_priority == PriorityLevel.T2:
                    slot.etag_priority = PriorityLevel.T1
                    self.stats["etag_upgrades"]["T2_to_T1"] += 1

        # 检查T1到T0升级
        if utilization > 0.9:
            slots = self.slot_pools.get(channel, [])
            for slot in slots:
                if slot.etag_priority == PriorityLevel.T1:
                    slot.etag_priority = PriorityLevel.T0
                    self.stats["etag_upgrades"]["T1_to_T0"] += 1

    def _check_itag_activation(self, channel: str, cycle: int) -> None:
        """
        检查ITag激活条件

        Args:
            channel: 通道类型
            cycle: 当前周期
        """
        utilization = self.stats.get("utilization", {}).get(channel, 0.0)
        if utilization > 0.8:
            direction_key = "horizontal" if self.direction in [Direction.TR, Direction.TL] else "vertical"
            self.stats["itag_activations"][direction_key] += 1

    def _process_slot_transmission(self, cycle: int) -> None:
        """
        处理slot传输逻辑

        Args:
            cycle: 当前周期
        """
        # 调用现有的step方法
        self.step(cycle)
