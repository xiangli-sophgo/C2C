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
        self.etag_config = {
            "TL": {"t2_ue_max": getattr(config.tag_config, "tl_etag_t2_ue_max", 8), "t1_ue_max": getattr(config.tag_config, "tl_etag_t1_ue_max", 15), "can_upgrade_to_t0": True},
            "TR": {"t2_ue_max": getattr(config.tag_config, "tr_etag_t2_ue_max", 12), "t1_ue_max": float("inf"), "can_upgrade_to_t0": False},
            "TU": {"t2_ue_max": getattr(config.tag_config, "tu_etag_t2_ue_max", 8), "t1_ue_max": getattr(config.tag_config, "tu_etag_t1_ue_max", 15), "can_upgrade_to_t0": True},
            "TD": {"t2_ue_max": getattr(config.tag_config, "td_etag_t2_ue_max", 12), "t1_ue_max": float("inf"), "can_upgrade_to_t0": False},
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

        # 统计信息
        self.stats = {
            "itag_triggers": {"req": 0, "rsp": 0, "data": 0},
            "itag_reservations": {"req": 0, "rsp": 0, "data": 0},
            "etag_upgrades": {"req": {"T2_to_T1": 0, "T1_to_T0": 0}, "rsp": {"T2_to_T1": 0, "T1_to_T0": 0}, "data": {"T2_to_T1": 0, "T1_to_T0": 0}},
            "successful_injections": {"req": 0, "rsp": 0, "data": 0},
            "successful_ejections": {"req": 0, "rsp": 0, "data": 0},
        }

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
        self.itag_states[channel][direction] = ITagState(active=True, reserved_slot_id=reserved_slot.slot_id, reserver_node_id=self.node_id, trigger_cycle=cycle, wait_cycles=0, direction=direction)

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

        Args:
            slot: 要检查的slot
            channel: 通道类型
            sub_direction: 子方向
            fifo_occupancy: FIFO当前占用
            fifo_depth: FIFO总深度

        Returns:
            是否可以下环
        """
        if not slot.is_occupied:
            return False

        priority = slot.etag_priority
        config = self.etag_config.get(sub_direction, {})

        if priority == PriorityLevel.T0:
            # T0级：可使用全部空间，但需要轮询仲裁
            if fifo_occupancy < fifo_depth:
                return self._check_t0_round_robin_grant(slot, channel, sub_direction)
            return False

        elif priority == PriorityLevel.T1:
            # T1级：可使用T1+T2空间
            t1_max = config.get("t1_ue_max", fifo_depth)
            return fifo_occupancy < t1_max

        else:  # PriorityLevel.T2
            # T2级：只能使用T2空间
            t2_max = config.get("t2_ue_max", fifo_depth // 2)
            return fifo_occupancy < t2_max

    def _check_t0_round_robin_grant(self, slot: CrossRingSlot, channel: str, sub_direction: str) -> bool:
        """
        检查T0级轮询仲裁授权

        Args:
            slot: T0级slot
            channel: 通道类型
            sub_direction: 子方向

        Returns:
            是否获得轮询授权
        """
        # TODO: T0级的时候需要先将slot存储到一个轮询表中，然后查看当前slot是不是第一个，
        state = self.etag_states[channel][sub_direction]

        # 简化的轮询实现
        if slot.flit:
            grant = (slot.flit.flit_id + state.round_robin_index) % 2 == 0
            state.round_robin_index = (state.round_robin_index + 1) % 16
            return grant

        return False

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
                    sub_dir: {"marked": state.marked, "priority": state.priority.value if state.priority else None, "failed_attempts": state.failed_attempts} for sub_dir, state in sub_dirs.items()
                }
                for channel, sub_dirs in self.etag_states.items()
            },
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
