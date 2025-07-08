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
    total_depth: int        # rb_in_depth 或 eq_in_depth
    t2_max: int            # T2级最大可用entry
    t1_max: int            # T1级最大可用entry (包含T2)
    has_dedicated_entries: bool = True  # 是否有专用entry (TL/TU=True, TR/TD=False)
    
    # 当前占用计数
    t2_occupied: int = 0   # T2级当前占用
    t1_occupied: int = 0   # T1级当前占用  
    t0_occupied: int = 0   # T0级当前占用
    
    def can_allocate_t2_entry(self) -> bool:
        """检查是否可以为T2级分配entry (T2只能使用T2专用entry)"""
        return self.t2_occupied < self.t2_max
        
    def can_allocate_t1_entry(self) -> bool:
        """检查是否可以为T1级分配entry (T1优先使用T1专用entry，然后使用T2 entry)"""
        if self.has_dedicated_entries:
            # 有专用entry：T1可以使用T1专用 + T2剩余
            return (self.t1_occupied + self.t2_occupied) < self.t1_max
        else:
            # 没有专用entry：所有等级共用一个池
            total_occupied = self.t0_occupied + self.t1_occupied + self.t2_occupied
            return total_occupied < self.total_depth
        
    def can_allocate_t0_entry(self) -> bool:
        """检查是否可以为T0级分配entry (T0优先使用T0专用entry，然后依次降级使用T1、T2 entry)"""
        if self.has_dedicated_entries:
            # 有专用entry：T0可以使用全部容量 - 其他等级占用
            total_occupied = self.t0_occupied + self.t1_occupied + self.t2_occupied
            return total_occupied < self.total_depth
        else:
            # 没有专用entry：所有等级共用一个池
            total_occupied = self.t0_occupied + self.t1_occupied + self.t2_occupied
            return total_occupied < self.total_depth
        
    def allocate_t2_entry(self) -> bool:
        """为T2级分配一个entry (T2只能使用T2专用entry)"""
        if self.can_allocate_t2_entry():
            self.t2_occupied += 1
            return True
        return False
        
    def allocate_t1_entry(self) -> bool:
        """为T1级分配一个entry (T1优先使用T1专用entry，然后使用T2 entry)"""
        if self.can_allocate_t1_entry():
            self.t1_occupied += 1
            return True
        return False
        
    def allocate_t0_entry(self) -> bool:
        """为T0级分配一个entry (T0优先使用T0专用entry，然后依次降级使用T1、T2 entry)"""
        if self.can_allocate_t0_entry():
            self.t0_occupied += 1
            return True
        return False
        
    def release_t2_entry(self) -> bool:
        """释放一个T2级entry"""
        if self.t2_occupied > 0:
            self.t2_occupied -= 1
            return True
        return False
        
    def release_t1_entry(self) -> bool:
        """释放一个T1级entry"""
        if self.t1_occupied > 0:
            self.t1_occupied -= 1
            return True
        return False
        
    def release_t0_entry(self) -> bool:
        """释放一个T0级entry"""
        if self.t0_occupied > 0:
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
            "available_for_t0": self.total_depth - (self.t0_occupied + self.t1_occupied + self.t2_occupied)
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
        tl_t0_capacity = getattr(config.fifo_config, 'rb_in_depth', 16)  # TL方向T0容量
        tu_t0_capacity = getattr(config.fifo_config, 'eq_in_depth', 16)  # TU方向T0容量
        
        self.etag_config = {
            "TL": {"t2_ue_max": getattr(config.tag_config, "tl_etag_t2_ue_max", 8), "t1_ue_max": getattr(config.tag_config, "tl_etag_t1_ue_max", 15), "can_upgrade_to_t0": True, "has_dedicated_entries": True},
            "TR": {"t2_ue_max": getattr(config.tag_config, "tr_etag_t2_ue_max", 12), "t1_ue_max": tl_t0_capacity, "can_upgrade_to_t0": False, "has_dedicated_entries": False},
            "TU": {"t2_ue_max": getattr(config.tag_config, "tu_etag_t2_ue_max", 8), "t1_ue_max": getattr(config.tag_config, "tu_etag_t1_ue_max", 15), "can_upgrade_to_t0": True, "has_dedicated_entries": True},
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
        self.T0_Etag_Order_FIFO: Dict[str, List[Any]] = {
            "req": [],
            "rsp": [],
            "data": []
        }
        
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
            
            self.entry_managers[sub_direction] = FifoEntryManager(
                total_depth=total_depth,
                t2_max=t2_max,
                t1_max=t1_max,
                has_dedicated_entries=has_dedicated_entries
            )
            
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
        routing_strategy = getattr(self.config, 'routing_strategy', 'XY')
        if hasattr(routing_strategy, 'value'):
            routing_strategy = routing_strategy.value
            
        # 获取FIFO深度配置
        rb_in_depth = getattr(self.config.fifo_config, 'rb_in_depth', 16)
        eq_in_depth = getattr(self.config.fifo_config, 'eq_in_depth', 16)
        
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
            return entry_manager.can_allocate_t2_entry()
            
        elif priority == PriorityLevel.T1:
            # T1级：优先使用T1专用entry，不够用再使用T2 entry
            return entry_manager.can_allocate_t1_entry()
            
        elif priority == PriorityLevel.T0:
            # T0级：优先使用T0专用entry，然后依次降级使用T1、T2 entry
            if not entry_manager.can_allocate_t0_entry():
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
            "total_removed": self.stats["t0_queue_operations"][channel]["removed"]
        }

    def _check_t0_round_robin_grant(self, slot: CrossRingSlot, channel: str, sub_direction: str) -> bool:
        """
        检查T0级轮询仲裁授权
        
        新实现基于T0_Etag_Order_FIFO全局队列：
        - 只有在T0队列第一位的slot才能获得授权
        - 实现真正的公平轮询机制

        Args:
            slot: T0级slot
            channel: 通道类型
            sub_direction: 子方向

        Returns:
            是否获得轮询授权
        """
        # 基于T0全局队列的轮询实现
        return self._is_first_in_t0_queue(slot, channel)

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
            
        success = False
        if priority == PriorityLevel.T2:
            success = entry_manager.allocate_t2_entry()
        elif priority == PriorityLevel.T1:
            success = entry_manager.allocate_t1_entry()
        elif priority == PriorityLevel.T0:
            success = entry_manager.allocate_t0_entry()
            
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
            
        success = False
        if priority == PriorityLevel.T2:
            success = entry_manager.release_t2_entry()
        elif priority == PriorityLevel.T1:
            success = entry_manager.release_t1_entry()
        elif priority == PriorityLevel.T0:
            success = entry_manager.release_t0_entry()
            
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
            cycle = getattr(slot, 'cycle', 0)
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
                    sub_dir: {"marked": state.marked, "priority": state.priority.value if state.priority else None, "failed_attempts": state.failed_attempts} for sub_dir, state in sub_dirs.items()
                }
                for channel, sub_dirs in self.etag_states.items()
            },
            "t0_queue_status": {
                channel: self.get_t0_queue_status(channel) for channel in ["req", "rsp", "data"]
            },
            "entry_managers_status": {
                sub_direction: manager.get_occupancy_info() for sub_direction, manager in self.entry_managers.items()
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
