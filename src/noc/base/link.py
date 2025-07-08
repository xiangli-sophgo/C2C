"""
Base Link类定义，提供链路的基础接口和slot管理机制。

本模块定义了NoC中链路的抽象基类，包括：
- 基础的slot时间片管理
- ETag/ITag在slot层面的抽象接口
- 流控和仲裁的基础框架
"""

from abc import ABC, abstractmethod
from typing import Dict, List, Any, Optional, Tuple
from enum import Enum
import logging
from dataclasses import dataclass, field

from .flit import BaseFlit


class PriorityLevel(Enum):
    """优先级级别枚举"""

    T0 = "T0"  # 最高优先级
    T1 = "T1"  # 中等优先级
    T2 = "T2"  # 最低优先级


class Direction(Enum):
    """方向枚举"""

    TR = "TR"  # 向右
    TL = "TL"  # 向左
    TU = "TU"  # 向上
    TD = "TD"  # 向下


@dataclass
class LinkSlot:
    """
    链路时间片(Slot)类。

    基础实现，提供通用的slot功能。
    """

    # 基础信息
    slot_id: int
    cycle: int
    direction: Direction
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

    def __init__(self, link_id: str, source_node: int, dest_node: int, num_slots: int = 8, logger: Optional[logging.Logger] = None):
        """
        初始化链路

        Args:
            link_id: 链路标识符
            source_node: 源节点ID
            dest_node: 目标节点ID
            num_slots: slot数量
            logger: 日志记录器
        """
        self.link_id = link_id
        self.source_node = source_node
        self.dest_node = dest_node
        self.num_slots = num_slots
        self.logger = logger or logging.getLogger(__name__)

        # 为每个通道创建slot管理
        self.slots = {"req": [], "rsp": [], "data": []}

        # 初始化slots
        self._initialize_slots()

        # 拥塞控制状态
        self.congestion_state = {
            "req": {"etag_level": PriorityLevel.T2, "itag_count": 0},
            "rsp": {"etag_level": PriorityLevel.T2, "itag_count": 0},
            "data": {"etag_level": PriorityLevel.T2, "itag_count": 0},
        }

        # 统计信息
        self.stats = {
            "flits_transmitted": {"req": 0, "rsp": 0, "data": 0},
            "slots_utilized": {"req": 0, "rsp": 0, "data": 0},
            "etag_upgrades": {"T2_to_T1": 0, "T1_to_T0": 0},
            "itag_activations": {"horizontal": 0, "vertical": 0},
            "congestion_cycles": 0,
            "total_wait_cycles": 0,
        }

    def _initialize_slots(self) -> None:
        """初始化所有通道的slots"""
        for channel in ["req", "rsp", "data"]:
            self.slots[channel] = []
            for i in range(self.num_slots):
                slot = LinkSlot(slot_id=i, cycle=0, direction=self._get_link_direction(), channel=channel)
                self.slots[channel].append(slot)

    @abstractmethod
    def _get_link_direction(self) -> Direction:
        """获取链路方向，由子类实现"""
        pass

    @abstractmethod
    def can_upgrade_etag(self, channel: str, from_level: PriorityLevel, to_level: PriorityLevel) -> bool:
        """
        检查是否可以升级ETag优先级

        Args:
            channel: 通道类型
            from_level: 源优先级
            to_level: 目标优先级

        Returns:
            是否可以升级
        """
        pass

    @abstractmethod
    def should_trigger_itag(self, channel: str, direction: str) -> bool:
        """
        检查是否应该触发ITag

        Args:
            channel: 通道类型
            direction: 方向("horizontal", "vertical")

        Returns:
            是否应该触发ITag
        """
        pass

    def get_available_slot(self, channel: str, priority: PriorityLevel = PriorityLevel.T2) -> Optional[LinkSlot]:
        """
        获取可用的slot

        Args:
            channel: 通道类型
            priority: 优先级

        Returns:
            可用的slot，如果没有则返回None
        """
        channel_slots = self.slots[channel]

        # 优先分配给高优先级
        available_slots = [slot for slot in channel_slots if not slot.is_occupied]

        if not available_slots:
            return None

        # 根据优先级和ITag状态选择最合适的slot
        best_slot = self._select_best_slot(available_slots, priority)
        return best_slot

    def _select_best_slot(self, available_slots: List[LinkSlot], priority: PriorityLevel) -> LinkSlot:
        """
        从可用slots中选择最佳的slot

        Args:
            available_slots: 可用slot列表
            priority: 请求的优先级

        Returns:
            最佳slot
        """
        # 简单实现：选择第一个可用的
        # 子类可以重写实现更复杂的选择逻辑
        return available_slots[0]

    def transmit_flit(self, flit: BaseFlit, channel: str, cycle: int) -> bool:
        """
        传输flit

        Args:
            flit: 要传输的flit
            channel: 通道类型
            cycle: 当前周期

        Returns:
            是否成功传输
        """
        # 获取flit的优先级（从slot层面确定）
        priority = self._determine_flit_priority(flit, channel)

        # 获取可用slot
        slot = self.get_available_slot(channel, priority)
        if slot is None:
            return False

        # 分配flit到slot
        if slot.assign_flit(flit):
            slot.cycle = cycle
            slot.etag_priority = priority

            # 更新统计
            self.stats["flits_transmitted"][channel] += 1
            self.stats["slots_utilized"][channel] += 1

            self.logger.debug(f"链路{self.link_id}在周期{cycle}成功传输{channel}通道flit到slot{slot.slot_id}")
            return True

        return False

    def _determine_flit_priority(self, flit: BaseFlit, channel: str) -> PriorityLevel:
        """
        确定flit的优先级（在slot层面）

        Args:
            flit: flit对象
            channel: 通道类型

        Returns:
            优先级级别
        """
        # 基础实现：根据通道类型确定默认优先级
        # 子类可以重写实现更复杂的优先级逻辑
        if channel == "req":
            return PriorityLevel.T2  # 请求默认最低优先级
        elif channel == "rsp":
            return PriorityLevel.T1  # 响应默认中等优先级
        else:  # data
            return PriorityLevel.T1  # 数据默认中等优先级

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

        # 检查是否需要ETag升级
        self._check_etag_upgrade(channel, utilization, cycle)

        # 检查是否需要ITag激活
        self._check_itag_activation(channel, cycle)

    @abstractmethod
    def _check_etag_upgrade(self, channel: str, utilization: float, cycle: int) -> None:
        """
        检查ETag升级条件

        Args:
            channel: 通道类型
            utilization: 利用率
            cycle: 当前周期
        """
        pass

    @abstractmethod
    def _check_itag_activation(self, channel: str, cycle: int) -> None:
        """
        检查ITag激活条件

        Args:
            channel: 通道类型
            cycle: 当前周期
        """
        pass

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
            "etag_distribution": self._get_etag_distribution(channel_slots),
            "itag_count": sum(1 for slot in channel_slots if slot.itag_active),
            "avg_wait_cycles": sum(slot.wait_cycles for slot in channel_slots) / len(channel_slots),
            "max_wait_cycles": max((slot.wait_cycles for slot in channel_slots), default=0),
        }

    def _get_etag_distribution(self, slots: List[LinkSlot]) -> Dict[str, int]:
        """获取ETag优先级分布"""
        distribution = {"T0": 0, "T1": 0, "T2": 0}
        for slot in slots:
            if slot.is_occupied:
                distribution[slot.etag_priority.value] += 1
        return distribution

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
            "etag_upgrades": {"T2_to_T1": 0, "T1_to_T0": 0},
            "itag_activations": {"horizontal": 0, "vertical": 0},
            "congestion_cycles": 0,
            "total_wait_cycles": 0,
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
