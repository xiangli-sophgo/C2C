"""
Base Link类定义，提供链路的基础接口和通用统计框架。

本模块定义了NoC中链路的抽象基类，包括：
- 通用的Slot时间片基础管理
- 基础的链路统计和监控框架
- 可扩展的抽象接口供具体拓扑实现
"""

from abc import ABC, abstractmethod
from typing import Dict, List, Any, Optional, Tuple
from enum import Enum
from dataclasses import dataclass, field

from .flit import BaseFlit


class BasicPriority(Enum):
    """通用优先级枚举"""

    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"


class BasicDirection(Enum):
    """通用方向枚举"""

    NORTH = "north"
    SOUTH = "south"
    EAST = "east"
    WEST = "west"
    LOCAL = "local"


@dataclass
class LinkSlot:
    """
    链路时间片(Slot)类。

    基础实现，提供通用的slot功能。
    """

    # 基础信息
    slot_id: int
    cycle: int
    direction: BasicDirection
    channel: str  # "req", "rsp", "data"

    # 承载的flit信息
    flit: Optional[BaseFlit] = None
    is_occupied: bool = False

    # 等待统计
    wait_cycles: int = 0

    def __post_init__(self):
        """初始化后处理"""
        if self.flit is not None:
            self.is_occupied = True

    def assign_flit(self, flit: BaseFlit) -> bool:
        """
        为slot分配flit

        Args:
            flit: 要分配的flit

        Returns:
            是否成功分配
        """
        if self.is_occupied:
            return False

        self.flit = flit
        self.is_occupied = True
        return True

    def release_flit(self) -> Optional[BaseFlit]:
        """
        释放slot中的flit

        Returns:
            被释放的flit，如果没有则返回None
        """
        flit = self.flit
        self.flit = None
        self.is_occupied = False
        self.wait_cycles = 0
        return flit

    def increment_wait(self) -> None:
        """增加等待周期计数"""
        self.wait_cycles += 1


class BaseLink(ABC):
    """
    链路抽象基类。

    定义了NoC中链路的基本接口，包括slot管理、
    流控和仲裁机制的抽象方法。
    """

    def __init__(self, link_id: str, source_node: int, dest_node: int, num_slots: int = 8):
        """
        初始化链路

        Args:
            link_id: 链路标识符
            source_node: 源节点ID
            dest_node: 目标节点ID
            num_slots: slot数量
        """
        self.link_id = link_id
        self.source_node = source_node
        self.dest_node = dest_node
        self.num_slots = num_slots

        # 为每个通道创建slot管理
        self.slots = {"req": [], "rsp": [], "data": []}

        # 初始化slots
        self._initialize_slots()

        # 拥塞控制状态
        self.congestion_state = {
            "req": {"utilization": 0.0, "blocked_cycles": 0},
            "rsp": {"utilization": 0.0, "blocked_cycles": 0},
            "data": {"utilization": 0.0, "blocked_cycles": 0},
        }

        # 统计信息
        self.stats = {
            "flits_transmitted": {"req": 0, "rsp": 0, "data": 0},
            "slots_utilized": {"req": 0, "rsp": 0, "data": 0},
            "congestion_cycles": 0,
            "total_wait_cycles": 0,
            "blocked_transmissions": 0,
        }

    def _initialize_slots(self) -> None:
        """初始化所有通道的slots"""
        for channel in ["req", "rsp", "data"]:
            self.slots[channel] = []
            for i in range(self.num_slots):
                slot = LinkSlot(slot_id=i, cycle=0, direction=BasicDirection.LOCAL, channel=channel)
                self.slots[channel].append(slot)


    def get_available_slot(self, channel: str, priority: BasicPriority = BasicPriority.LOW) -> Optional[LinkSlot]:
        """
        获取可用的slot

        Args:
            channel: 通道类型
            priority: 基础优先级

        Returns:
            可用的slot，如果没有则返回None
        """
        channel_slots = self.slots[channel]

        # 查找可用slots
        available_slots = [slot for slot in channel_slots if not slot.is_occupied]

        if not available_slots:
            return None

        # 简单选择第一个可用的，子类可以重写实现复杂逻辑
        return available_slots[0]

    def transmit_flit(self, flit: BaseFlit, channel: str, cycle: int) -> bool:
        """
        传输flit（基础实现）

        Args:
            flit: 要传输的flit
            channel: 通道类型
            cycle: 当前周期

        Returns:
            是否成功传输
        """
        # 获取可用slot
        slot = self.get_available_slot(channel)
        if slot is None:
            return False

        # 分配flit到slot
        if slot.assign_flit(flit):
            slot.cycle = cycle

            # 更新统计
            self.stats["flits_transmitted"][channel] += 1
            self.stats["slots_utilized"][channel] += 1

            return True

        return False

    def update_congestion_state(self, cycle: int) -> None:
        """
        更新拥塞状态

        Args:
            cycle: 当前周期
        """
        for channel in ["req", "rsp", "data"]:
            self._update_channel_congestion(channel, cycle)

    def _update_channel_congestion(self, channel: str, cycle: int) -> None:
        """
        更新单个通道的拥塞状态

        Args:
            channel: 通道类型
            cycle: 当前周期
        """
        channel_slots = self.slots[channel]

        # 计算占用率
        occupied_count = sum(1 for slot in channel_slots if slot.is_occupied)
        utilization = occupied_count / len(channel_slots)

        # 更新等待周期
        for slot in channel_slots:
            if slot.is_occupied:
                slot.increment_wait()

        # 更新拥塞状态
        self.congestion_state[channel]["utilization"] = utilization
        if utilization > 0.8:  # 高拥塞阈值
            self.congestion_state[channel]["blocked_cycles"] += 1


    def get_slot_status(self, channel: str) -> Dict[str, Any]:
        """
        获取指定通道的slot状态

        Args:
            channel: 通道类型

        Returns:
            slot状态信息
        """
        channel_slots = self.slots[channel]

        return {
            "total_slots": len(channel_slots),
            "occupied_slots": sum(1 for slot in channel_slots if slot.is_occupied),
            "utilization": sum(1 for slot in channel_slots if slot.is_occupied) / len(channel_slots),
            "avg_wait_cycles": sum(slot.wait_cycles for slot in channel_slots) / len(channel_slots),
            "max_wait_cycles": max((slot.wait_cycles for slot in channel_slots), default=0),
        }


    def get_link_stats(self) -> Dict[str, Any]:
        """
        获取链路统计信息

        Returns:
            统计信息字典
        """
        return {
            "link_id": self.link_id,
            "source_node": self.source_node,
            "dest_node": self.dest_node,
            "num_slots": self.num_slots,
            "stats": self.stats.copy(),
            "congestion_state": self.congestion_state.copy(),
            "channel_status": {channel: self.get_slot_status(channel) for channel in ["req", "rsp", "data"]},
        }

    def reset_stats(self) -> None:
        """重置统计信息"""
        self.stats = {
            "flits_transmitted": {"req": 0, "rsp": 0, "data": 0},
            "slots_utilized": {"req": 0, "rsp": 0, "data": 0},
            "congestion_cycles": 0,
            "total_wait_cycles": 0,
            "blocked_transmissions": 0,
        }

    def step(self, cycle: int) -> None:
        """
        执行一个时钟周期的处理

        Args:
            cycle: 当前周期
        """
        # 更新拥塞状态
        self.update_congestion_state(cycle)

        # 处理slot传输（子类实现具体逻辑）
        self._process_slot_transmission(cycle)

    @abstractmethod
    def _process_slot_transmission(self, cycle: int) -> None:
        """
        处理slot传输逻辑

        Args:
            cycle: 当前周期
        """
        pass
