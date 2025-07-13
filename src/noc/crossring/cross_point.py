"""
CrossRing Tag机制实现 - I-Tag和E-Tag防饿死机制

实现完整的防饿死机制：
- I-Tag: 注入预约机制，解决上环饿死问题
- E-Tag: 弹出优先级机制，解决下环饿死问题
"""

from typing import Dict, List, Any, Optional, Tuple
import logging
from enum import Enum
from dataclasses import dataclass

from ..base.link import PriorityLevel
from .flit import CrossRingFlit
from .crossring_link import CrossRingSlot, RingSlice
from .config import CrossRingConfig

# from .node import CrossRingNode


class CrossPointDirection(Enum):
    """CrossPoint方向枚举"""

    HORIZONTAL = "horizontal"  # 管理TR/TL
    VERTICAL = "vertical"  # 管理TU/TD


class TagTriggerCondition(Enum):
    """Tag触发条件"""

    WAIT_THRESHOLD = "wait_threshold"
    STARVATION = "starvation"
    CONGESTION = "congestion"


@dataclass
class ITagState:
    """I-Tag状态信息"""

    active: bool = False
    reserved_slot_id: Optional[str] = None
    reserver_node_id: Optional[int] = None
    trigger_cycle: int = 0
    wait_cycles: int = 0
    direction: Optional[str] = None


@dataclass
class ETagState:
    """E-Tag状态信息"""

    marked: bool = False
    priority: PriorityLevel = PriorityLevel.T2
    marked_cycle: int = 0
    failed_attempts: int = 0
    direction: Optional[str] = None
    round_robin_index: int = 0


@dataclass
class FifoEntryManager:
    """
    FIFO Entry管理器 - 管理分层entry分配和占用跟踪

    根据CrossRing Spec v2.0，每个方向需要独立管理不同等级的entry占用：
    - T2级：只能使用T2专用entry
    - T1级：优先使用T1专用entry，然后使用T2 entry
    - T0级：优先使用T0专用entry，然后依次降级使用T1、T2 entry

    对于没有专用entry的方向(TR/TD)，所有等级共用一个entry池，但仍然遵循优先级降级使用规则
    """

    # FIFO容量配置 (根据路由策略和方向确定)
    total_depth: int  # rb_in_depth 或 eq_in_depth
    t2_max: int  # T2级最大可用entry
    t1_max: int  # T1级最大可用entry (包含T2)
    has_dedicated_entries: bool = True  # 是否有专用entry (TL/TU=True, TR/TD=False)

    # 当前占用计数
    t2_occupied: int = 0  # T2级当前占用
    t1_occupied: int = 0  # T1级当前占用
    t0_occupied: int = 0  # T0级当前占用

    def can_allocate_entry(self, level: str) -> bool:
        """通用entry分配检查方法"""
        if level == "T2":
            return self.t2_occupied < self.t2_max
        elif level == "T1":
            if self.has_dedicated_entries:
                return (self.t1_occupied + self.t2_occupied) < self.t1_max
            else:
                total_occupied = self.t0_occupied + self.t1_occupied + self.t2_occupied
                return total_occupied < self.total_depth
        elif level == "T0":
            total_occupied = self.t0_occupied + self.t1_occupied + self.t2_occupied
            return total_occupied < self.total_depth
        return False

    def allocate_entry(self, level: str) -> bool:
        """通用entry分配方法"""
        if not self.can_allocate_entry(level):
            return False

        if level == "T2":
            self.t2_occupied += 1
        elif level == "T1":
            self.t1_occupied += 1
        elif level == "T0":
            self.t0_occupied += 1
        else:
            return False
        return True

    def release_entry(self, level: str) -> bool:
        """通用entry释放方法"""
        if level == "T2" and self.t2_occupied > 0:
            self.t2_occupied -= 1
            return True
        elif level == "T1" and self.t1_occupied > 0:
            self.t1_occupied -= 1
            return True
        elif level == "T0" and self.t0_occupied > 0:
            self.t0_occupied -= 1
            return True
        return False

    def get_occupancy_info(self) -> Dict[str, Any]:
        """获取占用情况信息"""
        return {
            "total_depth": self.total_depth,
            "t2_max": self.t2_max,
            "t1_max": self.t1_max,
            "t0_max": self.total_depth,
            "t2_occupied": self.t2_occupied,
            "t1_occupied": self.t1_occupied,
            "t0_occupied": self.t0_occupied,
            "total_occupied": self.t0_occupied + self.t1_occupied + self.t2_occupied,
            "available_for_t2": self.t2_max - self.t2_occupied,
            "available_for_t1": self.t1_max - (self.t1_occupied + self.t2_occupied),
            "available_for_t0": self.total_depth - (self.t0_occupied + self.t1_occupied + self.t2_occupied),
        }


class CrossRingTagManager:
    """
    CrossRing Tag机制管理器

    实现完整的I-Tag和E-Tag防饿死机制，
    按照Cross Ring Spec v2.0的规范进行实现。
    """

    def __init__(self, node_id: int, config: Any, logger: Optional[logging.Logger] = None):
        """
        初始化Tag管理器

        Args:
            node_id: 节点ID
            config: CrossRing配置
            logger: 日志记录器
        """
        self.node_id = node_id
        self.config = config
        self.logger = logger or logging.getLogger(__name__)

        # I-Tag配置参数
        self.itag_config = {
            "horizontal": {"trigger_threshold": getattr(config.tag_config, "itag_trigger_th_h", 80), "max_reservations": getattr(config.tag_config, "itag_max_num_h", 1)},
            "vertical": {"trigger_threshold": getattr(config.tag_config, "itag_trigger_th_v", 80), "max_reservations": getattr(config.tag_config, "itag_max_num_v", 1)},
        }

        # E-Tag配置参数
        # TR和TD的T1最大值应该等于TL和TU的T0容量，而不是无穷大
        tl_t0_capacity = getattr(config.fifo_config, "rb_in_depth", 16)  # TL方向T0容量
        tu_t0_capacity = getattr(config.fifo_config, "eq_in_depth", 16)  # TU方向T0容量

        self.etag_config = {
            "TL": {
                "t2_ue_max": getattr(config.tag_config, "tl_etag_t2_ue_max", 8),
                "t1_ue_max": getattr(config.tag_config, "tl_etag_t1_ue_max", 15),
                "can_upgrade_to_t0": True,
                "has_dedicated_entries": True,
            },
            "TR": {"t2_ue_max": getattr(config.tag_config, "tr_etag_t2_ue_max", 12), "t1_ue_max": tl_t0_capacity, "can_upgrade_to_t0": False, "has_dedicated_entries": False},
            "TU": {
                "t2_ue_max": getattr(config.tag_config, "tu_etag_t2_ue_max", 8),
                "t1_ue_max": getattr(config.tag_config, "tu_etag_t1_ue_max", 15),
                "can_upgrade_to_t0": True,
                "has_dedicated_entries": True,
            },
            "TD": {"t2_ue_max": getattr(config.tag_config, "td_etag_t2_ue_max", 12), "t1_ue_max": tu_t0_capacity, "can_upgrade_to_t0": False, "has_dedicated_entries": False},
        }

        # I-Tag状态管理
        self.itag_states: Dict[str, Dict[str, ITagState]] = {
            "req": {"horizontal": ITagState(), "vertical": ITagState()},
            "rsp": {"horizontal": ITagState(), "vertical": ITagState()},
            "data": {"horizontal": ITagState(), "vertical": ITagState()},
        }

        # E-Tag状态管理
        self.etag_states: Dict[str, Dict[str, ETagState]] = {
            "req": {"TL": ETagState(), "TR": ETagState(), "TU": ETagState(), "TD": ETagState()},
            "rsp": {"TL": ETagState(), "TR": ETagState(), "TU": ETagState(), "TD": ETagState()},
            "data": {"TL": ETagState(), "TR": ETagState(), "TU": ETagState(), "TD": ETagState()},
        }

        # T0 Etag Order FIFO - 全局T0级slot轮询队列
        self.T0_Etag_Order_FIFO: Dict[str, List[Any]] = {"req": [], "rsp": [], "data": []}

        # Entry管理器 - 为每个方向管理分层entry分配
        self.entry_managers: Dict[str, FifoEntryManager] = {}
        self._initialize_entry_managers()

        # 统计信息
        self.stats = {
            "itag_triggers": {"req": 0, "rsp": 0, "data": 0},
            "itag_reservations": {"req": 0, "rsp": 0, "data": 0},
            "etag_upgrades": {"req": {"T2_to_T1": 0, "T1_to_T0": 0}, "rsp": {"T2_to_T1": 0, "T1_to_T0": 0}, "data": {"T2_to_T1": 0, "T1_to_T0": 0}},
            "successful_injections": {"req": 0, "rsp": 0, "data": 0},
            "successful_ejections": {"req": 0, "rsp": 0, "data": 0},
            "t0_queue_operations": {"req": {"added": 0, "removed": 0}, "rsp": {"added": 0, "removed": 0}, "data": {"added": 0, "removed": 0}},
            "entry_allocations": {"req": {"T0": 0, "T1": 0, "T2": 0}, "rsp": {"T0": 0, "T1": 0, "T2": 0}, "data": {"T0": 0, "T1": 0, "T2": 0}},
        }

    def _initialize_entry_managers(self) -> None:
        """初始化每个方向的Entry管理器"""
        for sub_direction in ["TL", "TR", "TU", "TD"]:
            # 根据路由策略和方向确定使用的FIFO容量
            total_depth = self._get_t0_total_capacity(sub_direction)

            # 获取该方向的T1/T2配置
            config = self.etag_config.get(sub_direction, {})
            t2_max = config.get("t2_ue_max", 8)
            t1_max = config.get("t1_ue_max", 15)
            has_dedicated_entries = config.get("has_dedicated_entries", True)

            self.entry_managers[sub_direction] = FifoEntryManager(total_depth=total_depth, t2_max=t2_max, t1_max=t1_max, has_dedicated_entries=has_dedicated_entries)

        self.logger.debug(f"Node {self.node_id} 初始化完成Entry管理器")

    def _get_t0_total_capacity(self, sub_direction: str) -> int:
        """
        根据路由策略和方向确定T0级可用的FIFO容量

        XY路由: 横向环(TL/TR)下环到RB，纵向环(TU/TD)下环到EQ
        YX路由: 纵向环(TU/TD)下环到RB，横向环(TL/TR)下环到EQ

        Args:
            sub_direction: 子方向 (TL/TR/TU/TD)

        Returns:
            T0级可用的FIFO总容量
        """
        # 获取路由策略，默认为XY
        routing_strategy = getattr(self.config, "routing_strategy", "XY")
        if hasattr(routing_strategy, "value"):
            routing_strategy = routing_strategy.value

        # 获取FIFO深度配置
        rb_in_depth = getattr(self.config.fifo_config, "rb_in_depth", 16)
        eq_in_depth = getattr(self.config.fifo_config, "eq_in_depth", 16)

        if routing_strategy == "XY":
            if sub_direction in ["TL", "TR"]:  # 横向环
                return rb_in_depth  # 下环到RB
            else:  # TU, TD 纵向环
                return eq_in_depth  # 下环到EQ

        elif routing_strategy == "YX":
            if sub_direction in ["TU", "TD"]:  # 纵向环
                return rb_in_depth  # 下环到RB
            else:  # TL, TR 横向环
                return eq_in_depth  # 下环到EQ
        else:
            # 默认情况或其他路由策略
            self.logger.warning(f"未知路由策略 {routing_strategy}，使用默认配置")
            return max(rb_in_depth, eq_in_depth)

    def should_trigger_itag(self, channel: str, direction: str, wait_cycles: int) -> bool:
        """
        检查是否应该触发I-Tag预约

        Args:
            channel: 通道类型 (req/rsp/data)
            direction: 方向 (horizontal/vertical)
            wait_cycles: 等待周期数

        Returns:
            是否应该触发I-Tag
        """
        if direction not in self.itag_config:
            return False

        config = self.itag_config[direction]
        current_state = self.itag_states[channel][direction]

        # 检查触发条件
        threshold_met = wait_cycles >= config["trigger_threshold"]
        not_already_active = not current_state.active
        under_max_limit = True  # 简化实现，实际需要检查当前预约数量

        return threshold_met and not_already_active and under_max_limit

    def trigger_itag_reservation(self, channel: str, direction: str, ring_slice: RingSlice, cycle: int) -> bool:
        """
        触发I-Tag预约

        Args:
            channel: 通道类型
            direction: 方向
            ring_slice: Ring Slice实例
            cycle: 当前周期

        Returns:
            是否成功触发预约
        """
        if self.itag_states[channel][direction].active:
            return False

        # 查找可预约的slot
        reserved_slot = self._find_reservable_slot(ring_slice, channel)
        if not reserved_slot:
            return False

        # 激活I-Tag预约
        self.itag_states[channel][direction] = ITagState(
            active=True, reserved_slot_id=reserved_slot.slot_id, reserver_node_id=self.node_id, trigger_cycle=cycle, wait_cycles=0, direction=direction
        )

        # 在slot上设置预约标记
        reserved_slot.reserve_itag(self.node_id, direction)

        self.stats["itag_triggers"][channel] += 1
        self.stats["itag_reservations"][channel] += 1

        self.logger.debug(f"Node {self.node_id} 触发 {channel}:{direction} I-Tag预约，slot {reserved_slot.slot_id}")
        return True

    def cancel_itag_reservation(self, channel: str, direction: str, ring_slice: RingSlice) -> bool:
        """
        取消I-Tag预约

        Args:
            channel: 通道类型
            direction: 方向
            ring_slice: Ring Slice实例

        Returns:
            是否成功取消
        """
        state = self.itag_states[channel][direction]
        if not state.active:
            return False

        # 查找并清除预约的slot
        if state.reserved_slot_id:
            # 这里需要实际的slot查找逻辑
            # 简化实现
            pass

        # 清除I-Tag状态
        self.itag_states[channel][direction] = ITagState()

        self.logger.debug(f"Node {self.node_id} 取消 {channel}:{direction} I-Tag预约")
        return True

    def should_upgrade_etag(self, slot: CrossRingSlot, channel: str, sub_direction: str, failed_attempts: int) -> Optional[PriorityLevel]:
        """
        检查是否应该升级E-Tag优先级

        Args:
            slot: 要检查的slot
            channel: 通道类型
            sub_direction: 子方向 (TL/TR/TU/TD)
            failed_attempts: 下环失败次数

        Returns:
            建议的新优先级，如果不需要升级则返回None
        """
        if not slot.is_occupied:
            return None

        current_priority = slot.etag_priority
        config = self.etag_config.get(sub_direction, {})

        # T2 -> T1 升级
        if current_priority == PriorityLevel.T2 and failed_attempts >= 1:
            if config.get("t1_ue_max", 0) > config.get("t2_ue_max", 0):
                return PriorityLevel.T1

        # T1 -> T0 升级 (仅限TL/TU方向)
        elif current_priority == PriorityLevel.T1 and failed_attempts >= 2:
            if config.get("can_upgrade_to_t0", False):
                return PriorityLevel.T0

        return None

    def upgrade_etag_priority(self, slot: CrossRingSlot, channel: str, sub_direction: str, new_priority: PriorityLevel, cycle: int) -> bool:
        """
        升级E-Tag优先级

        Args:
            slot: 要升级的slot
            channel: 通道类型
            sub_direction: 子方向
            new_priority: 新优先级
            cycle: 当前周期

        Returns:
            是否成功升级
        """
        if not slot.is_occupied:
            return False

        old_priority = slot.etag_priority

        # 更新slot的E-Tag
        slot.mark_etag(new_priority, sub_direction)

        # 更新E-Tag状态
        self.etag_states[channel][sub_direction] = ETagState(
            marked=True, priority=new_priority, marked_cycle=cycle, failed_attempts=self.etag_states[channel][sub_direction].failed_attempts + 1, direction=sub_direction
        )

        # 如果升级到T0级，加入T0全局队列
        if new_priority == PriorityLevel.T0:
            self.add_to_t0_queue(slot, channel)

        # 更新统计
        if old_priority == PriorityLevel.T2 and new_priority == PriorityLevel.T1:
            self.stats["etag_upgrades"][channel]["T2_to_T1"] += 1
        elif old_priority == PriorityLevel.T1 and new_priority == PriorityLevel.T0:
            self.stats["etag_upgrades"][channel]["T1_to_T0"] += 1

        self.logger.debug(f"Node {self.node_id} 升级 {channel}:{sub_direction} E-Tag从 {old_priority.value} 到 {new_priority.value}")
        return True

    def can_eject_with_etag(self, slot: CrossRingSlot, channel: str, sub_direction: str, fifo_occupancy: int, fifo_depth: int) -> bool:
        """
        根据E-Tag检查是否可以下环

        新实现基于分层entry使用逻辑：
        - T2级：只能使用T2专用entry
        - T1级：优先使用T1专用entry，不够用再使用T2 entry
        - T0级：优先使用T0专用entry，然后依次降级使用T1、T2 entry
              只有使用T0专用entry时才需要判断轮询结果，使用其他等级entry时不需要判断轮询

        Args:
            slot: 要检查的slot
            channel: 通道类型
            sub_direction: 子方向
            fifo_occupancy: FIFO当前占用 (已弃用，改用entry管理器)
            fifo_depth: FIFO总深度 (已弃用，改用entry管理器)

        Returns:
            是否可以下环
        """
        if not slot.is_occupied:
            return False

        priority = slot.etag_priority

        # 获取该方向的entry管理器
        if sub_direction not in self.entry_managers:
            self.logger.error(f"未找到方向 {sub_direction} 的entry管理器")
            return False

        entry_manager = self.entry_managers[sub_direction]

        if priority == PriorityLevel.T2:
            # T2级：只能使用T2专用entry
            return entry_manager.can_allocate_entry("T2")

        elif priority == PriorityLevel.T1:
            # T1级：优先使用T1专用entry，不够用再使用T2 entry
            return entry_manager.can_allocate_entry("T1")

        elif priority == PriorityLevel.T0:
            # T0级：优先使用T0专用entry，然后依次降级使用T1、T2 entry
            if not entry_manager.can_allocate_entry("T0"):
                return False

            # 检查是否可以使用T0专用entry
            if entry_manager.has_dedicated_entries:
                # 计算T0专用entry的可用数量
                t0_dedicated_capacity = entry_manager.total_depth - entry_manager.t1_max
                t0_dedicated_available = t0_dedicated_capacity - entry_manager.t0_occupied

                if t0_dedicated_available > 0:
                    # 使用T0专用entry，需要判断轮询结果
                    is_first_in_queue = self._is_first_in_t0_queue(slot, channel)
                    self.logger.debug(f"T0级slot {slot.slot_id} 使用T0专用entry，轮询检查: is_first={is_first_in_queue}")
                    return is_first_in_queue
                else:
                    # 使用其他等级entry，不需要判断轮询结果
                    self.logger.debug(f"T0级slot {slot.slot_id} 使用其他等级entry，无需轮询检查")
                    return True
            else:
                # 没有专用entry的方向，使用共享entry池，不需要轮询检查
                return True

        return False

    def _is_first_in_t0_queue(self, slot: CrossRingSlot, channel: str) -> bool:
        """
        检查slot是否在T0全局队列的第一位

        Args:
            slot: 要检查的slot
            channel: 通道类型

        Returns:
            是否在队列第一位
        """
        if channel not in self.T0_Etag_Order_FIFO:
            return False

        queue = self.T0_Etag_Order_FIFO[channel]
        return len(queue) > 0 and queue[0] == slot

    def add_to_t0_queue(self, slot: CrossRingSlot, channel: str) -> bool:
        """
        将slot加入T0全局队列

        Args:
            slot: 要加入的slot
            channel: 通道类型

        Returns:
            是否成功加入
        """
        if channel not in self.T0_Etag_Order_FIFO:
            self.logger.error(f"无效的通道类型: {channel}")
            return False

        queue = self.T0_Etag_Order_FIFO[channel]

        # 避免重复添加
        if slot not in queue:
            queue.append(slot)
            self.stats["t0_queue_operations"][channel]["added"] += 1
            self.logger.debug(f"Node {self.node_id} 添加slot {slot.slot_id} 到T0队列 {channel}，队列长度: {len(queue)}")
            return True
        else:
            self.logger.debug(f"Slot {slot.slot_id} 已在T0队列 {channel} 中")
            return False

    def remove_from_t0_queue(self, slot: CrossRingSlot, channel: str) -> bool:
        """
        从T0全局队列移除slot

        Args:
            slot: 要移除的slot
            channel: 通道类型

        Returns:
            是否成功移除
        """
        if channel not in self.T0_Etag_Order_FIFO:
            self.logger.error(f"无效的通道类型: {channel}")
            return False

        queue = self.T0_Etag_Order_FIFO[channel]

        if slot in queue:
            queue.remove(slot)
            self.stats["t0_queue_operations"][channel]["removed"] += 1
            self.logger.debug(f"Node {self.node_id} 从T0队列 {channel} 移除slot {slot.slot_id}，队列长度: {len(queue)}")
            return True
        else:
            self.logger.debug(f"Slot {slot.slot_id} 不在T0队列 {channel} 中")
            return False

    def get_t0_queue_status(self, channel: str) -> Dict[str, Any]:
        """
        获取T0队列状态

        Args:
            channel: 通道类型

        Returns:
            队列状态信息
        """
        if channel not in self.T0_Etag_Order_FIFO:
            return {"error": f"无效的通道类型: {channel}"}

        queue = self.T0_Etag_Order_FIFO[channel]
        return {
            "channel": channel,
            "queue_length": len(queue),
            "first_slot_id": queue[0].slot_id if queue else None,
            "all_slot_ids": [slot.slot_id for slot in queue],
            "total_added": self.stats["t0_queue_operations"][channel]["added"],
            "total_removed": self.stats["t0_queue_operations"][channel]["removed"],
        }

    def allocate_entry_for_slot(self, slot: CrossRingSlot, channel: str, sub_direction: str) -> bool:
        """
        为slot分配entry

        Args:
            slot: 要分配entry的slot
            channel: 通道类型
            sub_direction: 子方向

        Returns:
            是否成功分配
        """
        if not slot.is_occupied:
            return False

        priority = slot.etag_priority
        entry_manager = self.entry_managers.get(sub_direction)

        if not entry_manager:
            self.logger.error(f"未找到方向 {sub_direction} 的entry管理器")
            return False

        success = entry_manager.allocate_entry(priority.value)

        if success:
            self.stats["entry_allocations"][channel][priority.value] += 1
            self.logger.debug(f"Node {self.node_id} 为slot {slot.slot_id} 分配{priority.value}级entry")

        return success

    def release_entry_for_slot(self, slot: CrossRingSlot, channel: str, sub_direction: str) -> bool:
        """
        释放slot占用的entry

        Args:
            slot: 要释放entry的slot
            channel: 通道类型
            sub_direction: 子方向

        Returns:
            是否成功释放
        """
        priority = slot.etag_priority
        entry_manager = self.entry_managers.get(sub_direction)

        if not entry_manager:
            self.logger.error(f"未找到方向 {sub_direction} 的entry管理器")
            return False

        success = entry_manager.release_entry(priority.value)

        if success:
            self.logger.debug(f"Node {self.node_id} 释放slot {slot.slot_id} 的{priority.value}级entry")

        return success

    def on_slot_ejected_successfully(self, slot: CrossRingSlot, channel: str, sub_direction: str) -> None:
        """
        slot成功下环时的清理工作

        Args:
            slot: 成功下环的slot
            channel: 通道类型
            sub_direction: 子方向
        """
        priority = slot.etag_priority

        # 释放entry
        self.release_entry_for_slot(slot, channel, sub_direction)

        # 如果是T0级，从所有T0队列中移除该slot
        if priority == PriorityLevel.T0:
            removed_count = 0
            for ch in ["req", "rsp", "data"]:
                if self.remove_from_t0_queue(slot, ch):
                    removed_count += 1

            if removed_count > 0:
                self.logger.debug(f"Node {self.node_id} T0级slot {slot.slot_id} 从 {removed_count} 个T0队列中移除")

        # 更新统计
        self.stats["successful_ejections"][channel] += 1

        self.logger.debug(f"Node {self.node_id} slot {slot.slot_id} 成功下环，清理完成")

    def on_slot_ejection_failed(self, slot: CrossRingSlot, channel: str, sub_direction: str) -> None:
        """
        slot下环失败时的处理

        Args:
            slot: 下环失败的slot
            channel: 通道类型
            sub_direction: 子方向
        """
        # 增加失败计数，可能触发优先级升级
        current_state = self.etag_states[channel][sub_direction]
        current_state.failed_attempts += 1

        # 检查是否需要升级优先级
        new_priority = self.should_upgrade_etag(slot, channel, sub_direction, current_state.failed_attempts)
        if new_priority and new_priority != slot.etag_priority:
            # 升级优先级
            cycle = getattr(slot, "cycle", 0)
            self.upgrade_etag_priority(slot, channel, sub_direction, new_priority, cycle)

        self.logger.debug(f"Node {self.node_id} slot {slot.slot_id} 下环失败，失败次数: {current_state.failed_attempts}")

    def _find_reservable_slot(self, ring_slice: RingSlice, channel: str) -> Optional[CrossRingSlot]:
        """
        查找可预约的slot

        Args:
            ring_slice: Ring Slice实例
            channel: 通道类型

        Returns:
            可预约的slot，如果没有则返回None
        """
        # 简化实现：查看当前slot
        current_slot = ring_slice.peek_current_slot(channel)

        if current_slot and current_slot.is_available and not current_slot.is_reserved:
            return current_slot

        # 实际实现需要遍历整个环路查找合适的slot
        return None

    def update_states(self, cycle: int) -> None:
        """
        更新所有Tag状态

        Args:
            cycle: 当前周期
        """
        # 更新I-Tag状态
        for channel in ["req", "rsp", "data"]:
            for direction in ["horizontal", "vertical"]:
                state = self.itag_states[channel][direction]
                if state.active:
                    state.wait_cycles += 1

                    # 简化：预约在固定周期后过期
                    if state.wait_cycles > 20:
                        self.itag_states[channel][direction] = ITagState()

        # 更新E-Tag状态
        for channel in ["req", "rsp", "data"]:
            for sub_direction in ["TL", "TR", "TU", "TD"]:
                state = self.etag_states[channel][sub_direction]
                if state.marked:
                    # E-Tag状态保持直到成功下环
                    pass

    def get_tag_manager_status(self) -> Dict[str, Any]:
        """
        获取Tag管理器状态

        Returns:
            状态信息字典
        """
        return {
            "node_id": self.node_id,
            "itag_states": {
                channel: {direction: {"active": state.active, "slot_id": state.reserved_slot_id, "wait_cycles": state.wait_cycles} for direction, state in directions.items()}
                for channel, directions in self.itag_states.items()
            },
            "etag_states": {
                channel: {
                    sub_dir: {"marked": state.marked, "priority": state.priority.value if state.priority else None, "failed_attempts": state.failed_attempts}
                    for sub_dir, state in sub_dirs.items()
                }
                for channel, sub_dirs in self.etag_states.items()
            },
            "t0_queue_status": {channel: self.get_t0_queue_status(channel) for channel in ["req", "rsp", "data"]},
            "entry_managers_status": {sub_direction: manager.get_occupancy_info() for sub_direction, manager in self.entry_managers.items()},
            "stats": self.stats.copy(),
        }

    def reset_stats(self) -> None:
        """重置统计信息"""
        for channel in ["req", "rsp", "data"]:
            self.stats["itag_triggers"][channel] = 0
            self.stats["itag_reservations"][channel] = 0
            self.stats["etag_upgrades"][channel]["T2_to_T1"] = 0
            self.stats["etag_upgrades"][channel]["T1_to_T0"] = 0
            self.stats["successful_injections"][channel] = 0
            self.stats["successful_ejections"][channel] = 0


class CrossRingCrossPoint:
    """
    CrossRing CrossPoint实现类 - 按Cross Ring Spec v2.0重新设计

    CrossPoint是交换和控制单元，包含4个slice（每个方向2个）：
    1. 控制Flit的上环和下环
    2. 实现I-Tag和E-Tag防饿死机制
    3. 管理到达slice和离开slice
    4. 处理路由决策和仲裁
    """

    def __init__(
        self,
        crosspoint_id: str,
        node_id: int,
        direction: CrossPointDirection,
        config: CrossRingConfig,
        coordinates: Tuple[int, int] = None,
        parent_node = None,
        logger: Optional[logging.Logger] = None,
    ):
        """
        初始化CrossPoint

        Args:
            crosspoint_id: CrossPoint标识符
            node_id: 所属节点ID
            direction: CrossPoint方向（水平/垂直）
            config: CrossRing配置
            coordinates: 节点坐标
            parent_node: 父Node引用
            logger: 日志记录器
        """
        self.crosspoint_id = crosspoint_id
        self.node_id = node_id
        self.direction = direction
        self.config = config
        self.coordinates = coordinates or (0, 0)
        self.parent_node = parent_node
        self.logger = logger or logging.getLogger(__name__)

        # 获取Tag配置
        self.tag_config = config.tag_config

        # 确定这个CrossPoint管理的方向
        if direction == CrossPointDirection.HORIZONTAL:
            self.managed_directions = ["TL", "TR"]
        else:  # VERTICAL
            self.managed_directions = ["TU", "TD"]

        # 4个slice管理：每个方向2个slice（到达+离开）
        self.slices: Dict[str, Dict[str, Optional[RingSlice]]] = {}
        for dir_name in self.managed_directions:
            self.slices[dir_name] = {"arrival": None, "departure": None}  # 到达本节点的slice（用于下环判断）  # 离开本节点的slice（用于上环判断）

        # 注入等待队列 - 等待上环的flit
        self.injection_queues: Dict[str, List[Tuple[CrossRingFlit, int]]] = {"req": [], "rsp": [], "data": []}  # (flit, wait_cycles)

        # I-Tag预约状态
        self.itag_reservations: Dict[str, Dict[str, Any]] = {
            "req": {"active": False, "slot_id": None, "wait_cycles": 0},
            "rsp": {"active": False, "slot_id": None, "wait_cycles": 0},
            "data": {"active": False, "slot_id": None, "wait_cycles": 0},
        }

        # E-Tag状态管理
        self.etag_states: Dict[str, Dict[str, Any]] = {
            "req": {"t0_round_robin": 0, "failed_ejects": {}},
            "rsp": {"t0_round_robin": 0, "failed_ejects": {}},
            "data": {"t0_round_robin": 0, "failed_ejects": {}},
        }

        # 统计信息
        self.stats = {
            "flits_injected": {"req": 0, "rsp": 0, "data": 0},
            "flits_ejected": {"req": 0, "rsp": 0, "data": 0},
            "itag_triggers": {"req": 0, "rsp": 0, "data": 0},
            "etag_upgrades": {"req": {"T2_to_T1": 0, "T1_to_T0": 0}, "rsp": {"T2_to_T1": 0, "T1_to_T0": 0}, "data": {"T2_to_T1": 0, "T1_to_T0": 0}},
            "t0_arbitrations": {"req": 0, "rsp": 0, "data": 0},
        }

        # 初始化Tag管理器
        self.tag_manager = CrossRingTagManager(node_id, config, logger)

        self.logger.info(f"CrossPoint {crosspoint_id} 初始化完成，方向：{direction.value}，管理方向：{self.managed_directions}")

    def connect_slice(self, direction: str, slice_type: str, ring_slice: RingSlice) -> None:
        """
        连接Ring Slice到CrossPoint

        Args:
            direction: 方向 ("TL", "TR", "TU", "TD")
            slice_type: slice类型 ("arrival"到达, "departure"离开)
            ring_slice: Ring Slice实例
        """
        if direction in self.slices and slice_type in self.slices[direction]:
            self.slices[direction][slice_type] = ring_slice
            self.logger.debug(f"CrossPoint {self.crosspoint_id} 连接{direction}方向的{slice_type} slice")

    def can_inject_flit(self, direction: str, channel: str) -> bool:
        """
        检查是否可以向指定方向注入Flit

        Args:
            direction: 方向 ("TL", "TR", "TU", "TD")
            channel: 通道类型 (req/rsp/data)

        Returns:
            是否可以注入
        """
        if direction not in self.managed_directions:
            return False

        departure_slice = self.slices[direction]["departure"]
        if not departure_slice:
            return False

        # 检查离开slice是否有空闲空间
        current_slot = departure_slice.peek_current_slot(channel)

        # 如果当前没有slot或是空slot，可以注入
        if current_slot is None:
            return True

        # 如果有预约的slot且是本节点预约的，可以注入
        if current_slot and current_slot.is_reserved and current_slot.itag_reserver_id == self.node_id:
            return True

        # 否则不能注入
        return False

    def try_inject_flit(self, direction: str, flit: CrossRingFlit, channel: str) -> bool:
        """
        尝试注入Flit到指定方向的环路（带I-Tag机制）

        Args:
            direction: 方向 ("TL", "TR", "TU", "TD")
            flit: 要注入的flit
            channel: 通道类型

        Returns:
            是否成功注入
        """
        if not self.can_inject_flit(direction, channel):
            return False

        departure_slice = self.slices[direction]["departure"]
        current_slot = departure_slice.peek_current_slot(channel)

        # 创建新的slot或使用预约的slot
        if current_slot is None:
            # 创建新slot
            new_slot = CrossRingSlot(slot_id=len(self.injection_queues[channel]), cycle=0, channel=channel)
            new_slot.assign_flit(flit)
            departure_slice.receive_slot(new_slot, channel)
        else:
            # 使用预约的slot（清除I-Tag预约）
            if current_slot.itag_reserved and current_slot.itag_reserver_id == self.node_id:
                current_slot.assign_flit(flit)
                current_slot.clear_itag()  # 清除I-Tag预约
                self.itag_reservations[channel]["active"] = False
                self.logger.debug(f"CrossPoint {self.crosspoint_id} 使用I-Tag预约的slot注入flit")
            else:
                # 普通slot
                current_slot.assign_flit(flit)

        # 更新flit状态信息
        # flit.flit_position = "LINK"
        flit.current_node_id = self.node_id
        flit.current_link_id = f"link_{self.node_id}_{direction}"
        flit.current_slice_index = 0  # 刚注入到departure slice
        flit.crosspoint_direction = "departure"
        flit.current_position = self.node_id

        self.stats["flits_injected"][channel] += 1
        self.logger.debug(f"CrossPoint {self.crosspoint_id} 成功注入flit {flit.flit_id} 到 {direction}方向{channel}通道")
        return True

    def can_eject_flit(self, slot: CrossRingSlot, channel: str, target_fifo_occupancy: int, target_fifo_depth: int) -> bool:
        """
        检查是否可以下环Flit

        Args:
            slot: 包含flit的slot
            channel: 通道类型
            target_fifo_occupancy: 目标FIFO当前占用
            target_fifo_depth: 目标FIFO深度

        Returns:
            是否可以下环
        """
        if not slot.is_occupied:
            return False

        # 获取子方向
        sub_direction = "TL" if self.direction == CrossPointDirection.HORIZONTAL else "TU"

        # 使用Tag管理器检查是否可以下环
        can_eject = self.tag_manager.can_eject_with_etag(slot, channel, sub_direction, target_fifo_occupancy, target_fifo_depth)

        return can_eject

    def should_eject_flit(self, flit: CrossRingFlit, current_direction: str = None) -> Tuple[bool, str]:
        """
        判断flit的下环决策：下环到IP、ring_bridge或继续在环上
        整合了原来的should_eject_to_ip, should_eject_to_ring_bridge等多个函数

        Args:
            flit: 要判断的flit
            current_direction: 当前到达的方向（可选）

        Returns:
            (是否下环, 下环目标: "IP" 或 "RB" 或 "")
        """
        if not self.parent_node:
            return False, ""

        # 获取坐标信息
        if hasattr(flit, "dest_coordinates"):
            dest_x, dest_y = flit.dest_coordinates
        else:
            # 没有坐标信息，使用节点ID判断
            is_local = (hasattr(flit, "destination") and flit.destination == self.parent_node.node_id) or (
                hasattr(flit, "dest_node_id") and flit.dest_node_id == self.parent_node.node_id
            )
            return (is_local, "IP") if is_local else (False, "")

        curr_x, curr_y = self.parent_node.coordinates

        # 获取路由策略
        routing_strategy = getattr(self.parent_node.config, "routing_strategy", "XY")
        if hasattr(routing_strategy, "value"):
            routing_strategy = routing_strategy.value

        # 根据CrossPoint方向、路由策略和维度匹配判断下环策略
        if current_direction:
            if self.direction == CrossPointDirection.HORIZONTAL and current_direction in ["TR", "TL"]:
                # 水平CrossPoint: X维度到达时判断下环目标
                if dest_x == curr_x:
                    # XY路由：水平环必须通过RB下环
                    if routing_strategy == "XY":
                        return True, "RB"  # XY路由水平环必须走RB
                    else:  # YX路由：水平环可以直接下环到EQ
                        return True, "EQ" if dest_y == curr_y else "RB"

            elif self.direction == CrossPointDirection.VERTICAL and current_direction in ["TU", "TD"]:
                # 垂直CrossPoint: Y维度到达时判断下环目标
                if dest_y == curr_y:
                    # XY路由：垂直环可以通过EQ下环，YX路由：垂直环只能通过RB下环
                    if routing_strategy == "YX":
                        return True, "RB"  # YX路由垂直环必须走RB
                    else:  # XY或其他路由
                        return True, "EQ" if dest_x == curr_x else "RB"

        return False, ""

    def _try_transfer_to_ring_bridge(self, flit: CrossRingFlit, slot: Any, from_direction: str, channel: str) -> bool:
        """
        尝试将flit从当前环转移到ring_bridge

        Args:
            flit: 要转移的flit
            slot: 包含flit的slot
            from_direction: 来源方向（到达slice的方向）
            channel: 通道类型

        Returns:
            是否成功转移
        """
        # 从slot中取出flit
        transferred_flit = slot.release_flit()
        if not transferred_flit:
            return False

        # 计算flit的实际传输方向（而不是到达slice的方向）
        actual_direction = self._get_flit_actual_direction(transferred_flit, from_direction)

        # 更新flit状态，使用实际传输方向
        transferred_flit.flit_position = f"RB_{actual_direction}"
        transferred_flit.current_node_id = self.node_id
        transferred_flit.rb_fifo_name = f"RB_{from_direction}"

        # 添加到ring_bridge输入，使用实际传输方向
        success = self.add_to_ring_bridge_input(transferred_flit, actual_direction, channel)
        if success:
            self.logger.debug(f"CrossPoint {self.crosspoint_id} 成功将flit转移到ring_bridge，实际方向: {actual_direction}")

        return success

    def _get_flit_actual_direction(self, flit: CrossRingFlit, arrival_direction: str) -> str:
        """
        计算flit的实际传输方向（基于其路由目标）

        Args:
            flit: 要分析的flit
            arrival_direction: 到达slice的方向

        Returns:
            flit的实际传输方向
        """
        # 计算flit的下一个路由方向
        next_direction = self.parent_node._calculate_routing_direction(flit) if self.parent_node else "TR"

        # 如果是EQ（本地），则使用到达方向
        if next_direction == "EQ":
            return arrival_direction

        # 否则使用路由计算的方向
        return next_direction

    def add_to_ring_bridge_input(self, flit: CrossRingFlit, from_direction: str, channel: str) -> bool:
        """
        将flit添加到ring_bridge输入

        Args:
            flit: 要添加的flit
            from_direction: 来源方向
            channel: 通道类型

        Returns:
            是否成功添加
        """
        if self.parent_node is None:
            self.logger.error(f"CrossPoint {self.crosspoint_id} 没有parent_node引用，无法访问ring_bridge")
            return False

        # 调用父Node的ring_bridge输入方法
        success = self.parent_node.add_to_ring_bridge_input(flit, from_direction, channel)

        return success

    def try_eject_flit(self, slot: CrossRingSlot, channel: str, target_fifo_occupancy: int, target_fifo_depth: int) -> Optional[CrossRingFlit]:
        """
        尝试从环路下环Flit

        Args:
            slot: 包含flit的slot
            channel: 通道类型
            target_fifo_occupancy: 目标FIFO当前占用
            target_fifo_depth: 目标FIFO深度

        Returns:
            成功下环的flit，失败返回None
        """
        if not self.can_eject_flit(slot, channel, target_fifo_occupancy, target_fifo_depth):
            # 下环失败，考虑E-Tag升级
            self._handle_eject_failure(slot, channel)
            return None

        # 成功下环
        ejected_flit = slot.release_flit()
        if ejected_flit:
            # 更新flit位置状态 - 从arrival slice下环（具体EQ方向由调用者设置）
            ejected_flit.current_node_id = self.node_id
            ejected_flit.crosspoint_direction = "arrival"

            # 使用Tag管理器处理成功下环
            sub_direction = "TL" if self.direction == CrossPointDirection.HORIZONTAL else "TU"
            self.tag_manager.on_slot_ejected_successfully(slot, channel, sub_direction)

            self.stats["flits_ejected"][channel] += 1
            if slot.etag_priority == PriorityLevel.T0:
                self.stats["t0_arbitrations"][channel] += 1

            self.logger.debug(f"CrossPoint {self.crosspoint_id} 成功下环flit {ejected_flit.flit_id} 从 {channel} 通道")

        return ejected_flit

    def process_itag_request(self, flit: CrossRingFlit, channel: str, wait_cycles: int) -> bool:
        """
        处理I-Tag预约请求

        Args:
            flit: 等待的flit
            channel: 通道类型
            wait_cycles: 等待周期数

        Returns:
            是否成功发起预约
        """
        threshold = getattr(self.config.tag_config, "itag_trigger_th_h", 80)

        if wait_cycles < threshold:
            return False

        if self.itag_reservations[channel]["active"]:
            return False  # 已有预约激活

        # 查找可预约的slot
        ring_slice = self.ring_slice_interfaces.get(channel)
        if not ring_slice:
            return False

        # 简化：尝试预约下一个空闲slot
        # 实际实现需要遍历环路查找合适的slot
        self.itag_reservations[channel] = {"active": True, "slot_id": f"reserved_{self.node_id}_{channel}", "wait_cycles": 0}

        self.stats["itag_triggers"][channel] += 1
        self.logger.debug(f"CrossPoint {self.crosspoint_id} 发起 {channel} 通道的I-Tag预约")
        return True

    def process_etag_upgrade(self, slot: CrossRingSlot, channel: str, failed_attempts: int) -> None:
        """
        处理E-Tag优先级提升

        Args:
            slot: 要升级的slot
            channel: 通道类型
            failed_attempts: 下环失败次数
        """
        if not slot.is_occupied:
            return

        new_priority = slot.should_upgrade_etag(failed_attempts)

        if new_priority != slot.etag_priority:
            old_priority = slot.etag_priority
            slot.mark_etag(new_priority, self._get_sub_direction_from_channel(channel))

            # 更新统计
            if old_priority == PriorityLevel.T2 and new_priority == PriorityLevel.T1:
                self.stats["etag_upgrades"][channel]["T2_to_T1"] += 1
            elif old_priority == PriorityLevel.T1 and new_priority == PriorityLevel.T0:
                self.stats["etag_upgrades"][channel]["T1_to_T0"] += 1

            self.logger.debug(f"CrossPoint {self.crosspoint_id} 将 {channel} 通道的slot {slot.slot_id} E-Tag从 {old_priority.value} 升级到 {new_priority.value}")

    def step_compute_phase(self, cycle: int, node_inject_fifos: Dict[str, Dict[str, Any]], node_eject_fifos: Dict[str, Dict[str, Any]]) -> None:
        """
        计算阶段：确定传输可能性但不执行传输
        """
        # 初始化传输计划
        self._injection_transfer_plan = []
        self._ejection_transfer_plan = []

        # 计算下环可能性：检查每个管理方向的到达slice
        for direction in self.managed_directions:
            arrival_slice = self.slices[direction]["arrival"]
            if not arrival_slice:
                continue

            for channel in ["req", "rsp", "data"]:
                current_slot = arrival_slice.peek_current_slot(channel)
                if current_slot and current_slot.is_occupied:
                    flit = current_slot.flit
                    should_eject, eject_target = self.should_eject_flit(flit, direction)

                    if should_eject:
                        if eject_target == "RB":
                            self._ejection_transfer_plan.append({"type": "to_ring_bridge", "direction": direction, "channel": channel, "slot": current_slot, "flit": flit})
                        elif eject_target in ["IP", "EQ"]:
                            if direction in node_eject_fifos[channel]:
                                eject_fifo = node_eject_fifos[channel][direction]
                                fifo_occupancy = len(eject_fifo.internal_queue)
                                fifo_depth = eject_fifo.internal_queue.maxlen

                                if self.can_eject_flit(current_slot, channel, fifo_occupancy, fifo_depth):
                                    self._ejection_transfer_plan.append(
                                        {"type": "to_eject_fifo", "direction": direction, "channel": channel, "slot": current_slot, "flit": flit, "target_fifo": eject_fifo}
                                    )

        # 计算上环可能性：首先处理ring_bridge输出的重新注入
        for direction in self.managed_directions:
            for channel in ["req", "rsp", "data"]:
                if self.parent_node:
                    ring_bridge_flit = self.parent_node.get_ring_bridge_output_flit(direction, channel)
                    if ring_bridge_flit and self.can_inject_flit(direction, channel):
                        self._injection_transfer_plan.append(
                            {"type": "ring_bridge_reinject", "direction": direction, "channel": channel, "flit": ring_bridge_flit, "priority": "high"}
                        )

        # 然后处理正常的inject_direction_fifos（FIFO流水线逻辑）
        for direction in self.managed_directions:
            for channel in ["req", "rsp", "data"]:
                if direction in node_inject_fifos[channel]:
                    direction_fifo = node_inject_fifos[channel][direction]
                    # 只有在FIFO有数据且环路可以接受时才计划传输
                    if direction_fifo.valid_signal() and self.can_inject_flit(direction, channel):
                        self._injection_transfer_plan.append(
                            {"type": "fifo_pipeline_read", "direction": direction, "channel": channel, "source_fifo": direction_fifo, "priority": "normal"}
                        )

        # 更新等待状态（不执行传输）
        for channel in ["req", "rsp", "data"]:
            if self.injection_queues[channel]:
                for i, (flit, wait_cycles) in enumerate(self.injection_queues[channel]):
                    self.injection_queues[channel][i] = (flit, wait_cycles + 1)

    def step_update_phase(self, cycle: int, node_inject_fifos: Dict[str, Dict[str, Any]], node_eject_fifos: Dict[str, Dict[str, Any]]) -> None:
        """
        更新阶段：执行compute阶段确定的传输
        """
        # 执行下环传输
        for transfer in getattr(self, "_ejection_transfer_plan", []):
            if transfer["type"] == "to_ring_bridge":
                success = self._try_transfer_to_ring_bridge(transfer["flit"], transfer["slot"], transfer["direction"], transfer["channel"])
                if success:
                    self.logger.debug(f"CrossPoint {self.crosspoint_id} 成功下环到ring_bridge: {transfer['direction']} {transfer['channel']}")

            elif transfer["type"] == "to_eject_fifo":
                ejected_flit = self.try_eject_flit(
                    transfer["slot"], transfer["channel"], len(transfer["target_fifo"].internal_queue), transfer["target_fifo"].internal_queue.maxlen
                )
                if ejected_flit and transfer["target_fifo"].write_input(ejected_flit):
                    ejected_flit.flit_position = f"EQ_{transfer['direction']}"
                    self.logger.debug(f"CrossPoint {self.crosspoint_id} 成功下环到EQ: {transfer['direction']} {transfer['channel']}")

        # 执行上环传输（按优先级）
        high_priority = [t for t in getattr(self, "_injection_transfer_plan", []) if t.get("priority") == "high"]
        normal_transfers = [t for t in getattr(self, "_injection_transfer_plan", []) if t.get("priority") != "high"]

        for transfer in high_priority + normal_transfers:
            if transfer["type"] == "ring_bridge_reinject":
                flit = transfer["flit"]
                flit.flit_position = "LINK"
                if self.try_inject_flit(transfer["direction"], flit, transfer["channel"]):
                    self.logger.debug(f"CrossPoint {self.crosspoint_id} 从 ring_bridge {transfer['direction']} 重新注入成功")

            elif transfer["type"] == "fifo_pipeline_read":
                # 严格流水线：只有在确保能注入时才读取FIFO
                # compute阶段已经验证了can_inject_flit，这里直接执行
                flit = transfer["source_fifo"].read_output()
                if flit:
                    if self.try_inject_flit(transfer["direction"], flit, transfer["channel"]):
                        self.logger.debug(f"CrossPoint {self.crosspoint_id} 从 {transfer['direction']} FIFO注入到环路成功")
                    else:
                        # 这种情况不应该发生，因为compute阶段已经检查过can_inject_flit
                        self.logger.error(f"CrossPoint {self.crosspoint_id} 严重错误：compute阶段检查通过但update阶段注入失败")
                        # 紧急情况：尝试放回FIFO头部
                        if not transfer["source_fifo"].priority_write(flit):
                            self.logger.error(f"CrossPoint {self.crosspoint_id} 无法将flit放回FIFO，数据丢失风险")

            elif transfer["type"] == "trigger_itag":
                if self._trigger_itag_reservation(transfer["direction"], transfer["channel"], cycle):
                    self.logger.debug(f"CrossPoint {self.crosspoint_id} 为 {transfer['direction']} {transfer['channel']} 触发I-Tag预约")

        # 更新I-Tag预约状态
        for channel in ["req", "rsp", "data"]:
            if self.itag_reservations[channel]["active"]:
                self.itag_reservations[channel]["wait_cycles"] += 1
                if self.itag_reservations[channel]["wait_cycles"] > 10:
                    self.itag_reservations[channel]["active"] = False
                    self.itag_reservations[channel]["wait_cycles"] = 0

    def step(self, cycle: int, node_inject_fifos: Dict[str, Dict[str, Any]], node_eject_fifos: Dict[str, Dict[str, Any]]) -> None:
        """
        CrossPoint步进函数：执行两阶段处理

        Args:
            cycle: 当前周期
            node_inject_fifos: 节点的inject_direction_fifos
            node_eject_fifos: 节点的eject_input_fifos
        """
        self.step_compute_phase(cycle, node_inject_fifos, node_eject_fifos)
        self.step_update_phase(cycle, node_inject_fifos, node_eject_fifos)

    def _trigger_itag_reservation(self, direction: str, channel: str, cycle: int) -> bool:
        """触发I-Tag预约"""
        # 确定环路类型
        ring_type = "horizontal" if direction in ["TL", "TR"] else "vertical"

        # 获取departure slice
        departure_slice = self.slices[direction]["departure"]
        if not departure_slice:
            return False

        # 使用Tag管理器触发预约
        success = self.tag_manager.trigger_itag_reservation(channel, ring_type, departure_slice, cycle)

        if success:
            self.itag_reservations[channel]["active"] = True
            self.itag_reservations[channel]["slot_id"] = f"reserved_{self.node_id}_{channel}"
            self.itag_reservations[channel]["wait_cycles"] = 0

        return success

    def _handle_eject_failure(self, slot: CrossRingSlot, channel: str) -> None:
        """处理下环失败，考虑E-Tag升级"""
        sub_direction = "TL" if self.direction == CrossPointDirection.HORIZONTAL else "TU"

        # 使用Tag管理器处理下环失败
        self.tag_manager.on_slot_ejection_failed(slot, channel, sub_direction)

        # 更新本地统计
        flit_id = slot.flit.flit_id if slot.flit else "unknown"
        if flit_id not in self.etag_states[channel]["failed_ejects"]:
            self.etag_states[channel]["failed_ejects"][flit_id] = 0

        self.etag_states[channel]["failed_ejects"][flit_id] += 1
        failed_count = self.etag_states[channel]["failed_ejects"][flit_id]

        # 检查是否需要E-Tag升级
        new_priority = self.tag_manager.should_upgrade_etag(slot, channel, sub_direction, failed_count)
        if new_priority and new_priority != slot.etag_priority:
            cycle = getattr(slot, "cycle", 0)
            success = self.tag_manager.upgrade_etag_priority(slot, channel, sub_direction, new_priority, cycle)

            if success:
                # 更新统计
                old_priority = slot.etag_priority
                if old_priority == PriorityLevel.T1 and new_priority == PriorityLevel.T0:
                    self.stats["etag_upgrades"][channel]["T1_to_T0"] += 1
                elif old_priority == PriorityLevel.T2 and new_priority == PriorityLevel.T1:
                    self.stats["etag_upgrades"][channel]["T2_to_T1"] += 1

                self.logger.debug(f"CrossPoint {self.crosspoint_id} 升级slot {slot.slot_id} E-Tag从{old_priority.value}到{new_priority.value}")

    def get_crosspoint_status(self) -> Dict[str, Any]:
        """
        获取CrossPoint状态信息

        Returns:
            状态信息字典
        """
        return {
            "crosspoint_id": self.crosspoint_id,
            "node_id": self.node_id,
            "direction": self.direction.value,
            "injection_queue_lengths": {channel: len(queue) for channel, queue in self.injection_queues.items()},
            "itag_reservations": self.itag_reservations.copy(),
            "etag_states": self.etag_states.copy(),
            "stats": self.stats.copy(),
            "ring_slice_connected": {channel: slice is not None for channel, slice in self.ring_slice_interfaces.items()} if hasattr(self, "ring_slice_interfaces") else {},
        }
