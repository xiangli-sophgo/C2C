"""
CrossRing节点实现。

提供CrossRing网络中节点的详细实现，包括：
- 注入/提取队列管理
- 环形缓冲区管理
- 拥塞控制机制
- 仲裁逻辑
"""

from typing import Dict, List, Any, Tuple, Optional
import logging
from enum import Enum
from dataclasses import dataclass

from src.noc.base.node import BaseNoCNode
from src.noc.base.ip_interface import PipelinedFIFO, FlowControlledTransfer
from ..base.link import PriorityLevel
from .flit import CrossRingFlit
from .config import CrossRingConfig, RoutingStrategy
from .crossring_link import CrossRingSlot, RingSlice  # 导入新的类


class CrossPointDirection(Enum):
    """CrossPoint方向枚举"""

    HORIZONTAL = "horizontal"  # 管理TR/TL
    VERTICAL = "vertical"  # 管理TU/TD


class CrossRingCrossPoint:
    """
    CrossRing CrossPoint实现类 - 按Cross Ring Spec v2.0重新设计
    
    CrossPoint是交换和控制单元，包含4个slice（每个方向2个）：
    1. 控制Flit的上环和下环
    2. 实现I-Tag和E-Tag防饿死机制  
    3. 管理到达slice和离开slice
    4. 处理路由决策和仲裁
    """

    def __init__(self, crosspoint_id: str, node_id: int, direction: CrossPointDirection, 
                 config: CrossRingConfig, logger: Optional[logging.Logger] = None):
        """
        初始化CrossPoint

        Args:
            crosspoint_id: CrossPoint标识符
            node_id: 所属节点ID
            direction: CrossPoint方向（水平/垂直）
            config: CrossRing配置
            logger: 日志记录器
        """
        self.crosspoint_id = crosspoint_id
        self.node_id = node_id
        self.direction = direction
        self.config = config
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
            self.slices[dir_name] = {
                "arrival": None,    # 到达本节点的slice（用于下环判断）
                "departure": None   # 离开本节点的slice（用于上环判断）
            }

        # 注入等待队列 - 等待上环的flit
        self.injection_queues: Dict[str, List[Tuple[CrossRingFlit, int]]] = {
            "req": [],  # (flit, wait_cycles)
            "rsp": [],
            "data": []
        }

        # I-Tag预约状态
        self.itag_reservations: Dict[str, Dict[str, Any]] = {
            "req": {"active": False, "slot_id": None, "wait_cycles": 0},
            "rsp": {"active": False, "slot_id": None, "wait_cycles": 0},
            "data": {"active": False, "slot_id": None, "wait_cycles": 0}
        }

        # E-Tag状态管理
        self.etag_states: Dict[str, Dict[str, Any]] = {
            "req": {"t0_round_robin": 0, "failed_ejects": {}},
            "rsp": {"t0_round_robin": 0, "failed_ejects": {}},
            "data": {"t0_round_robin": 0, "failed_ejects": {}}
        }

        # 统计信息
        self.stats = {
            "flits_injected": {"req": 0, "rsp": 0, "data": 0},
            "flits_ejected": {"req": 0, "rsp": 0, "data": 0},
            "itag_triggers": {"req": 0, "rsp": 0, "data": 0},
            "etag_upgrades": {
                "req": {"T2_to_T1": 0, "T1_to_T0": 0},
                "rsp": {"T2_to_T1": 0, "T1_to_T0": 0},
                "data": {"T2_to_T1": 0, "T1_to_T0": 0}
            },
            "t0_arbitrations": {"req": 0, "rsp": 0, "data": 0}
        }

        # 导入和初始化Tag管理器
        from .tag_mechanism import CrossRingTagManager
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
            new_slot = CrossRingSlot(
                slot_id=len(self.injection_queues[channel]),
                cycle=0,
                channel=channel
            )
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
            
        self.stats["flits_injected"][channel] += 1
        self.logger.debug(f"CrossPoint {self.crosspoint_id} 成功注入flit {flit.flit_id} 到 {direction}方向{channel}通道")
        return True
        
    def process_injection_from_fifos(self, node_fifos: Dict[str, Dict[str, Any]], cycle: int) -> None:
        """
        处理从节点inject_direction_fifos的上环判断（带I-Tag机制）
        
        Args:
            node_fifos: 节点的inject_direction_fifos
            cycle: 当前周期
        """
        # 检查每个管理方向的inject_direction_fifos
        for direction in self.managed_directions:
            for channel in ["req", "rsp", "data"]:
                if direction in node_fifos[channel]:
                    direction_fifo = node_fifos[channel][direction]
                    
                    # 检查是否有flit等待注入
                    if direction_fifo.valid_signal():
                        flit = direction_fifo.peek_output()
                        if flit:
                            # 检查是否可以立即注入
                            if self.can_inject_flit(direction, channel):
                                # 可以注入，读取flit并注入
                                flit = direction_fifo.read_output()
                                if self.try_inject_flit(direction, flit, channel):
                                    self.logger.debug(f"CrossPoint {self.crosspoint_id} 从{direction}方向FIFO成功注入flit到环路")
                                else:
                                    # 注入失败，放回FIFO
                                    direction_fifo.priority_write(flit)
                                    self.logger.debug(f"CrossPoint {self.crosspoint_id} 注入失败，flit返回{direction}方向FIFO")
                            else:
                                # 不能立即注入，检查是否需要触发I-Tag
                                if not self.itag_reservations[channel]["active"]:
                                    # 计算等待时间（简化：从FIFO深度估算）
                                    wait_cycles = len(direction_fifo.internal_queue) * 2  # 简化估算
                                    if wait_cycles >= self._get_itag_threshold():
                                        # 触发I-Tag预约
                                        if self._trigger_itag_reservation(direction, channel, cycle):
                                            self.logger.debug(f"CrossPoint {self.crosspoint_id} 为{direction}方向{channel}通道触发I-Tag预约")
                                        else:
                                            self.logger.debug(f"CrossPoint {self.crosspoint_id} I-Tag预约失败")
                                else:
                                    self.logger.debug(f"CrossPoint {self.crosspoint_id} {channel}通道已有I-Tag预约活跃")

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
        sub_direction = self._get_sub_direction_from_channel(channel)
        
        # 使用Tag管理器检查是否可以下环
        can_eject = self.tag_manager.can_eject_with_etag(
            slot, channel, sub_direction, target_fifo_occupancy, target_fifo_depth
        )
        
        return can_eject
            
    def should_eject_flit(self, flit: CrossRingFlit) -> bool:
        """
        判断flit是否应该在本节点下环
        
        Args:
            flit: 要判断的flit
            
        Returns:
            是否应该下环
        """
        # 检查flit的目标是否是当前节点
        if hasattr(flit, 'destination') and flit.destination == self.node_id:
            return True
        if hasattr(flit, 'dest_node_id') and flit.dest_node_id == self.node_id:
            return True
        return False

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
            # 使用Tag管理器处理成功下环
            sub_direction = self._get_sub_direction_from_channel(channel)
            self.tag_manager.on_slot_ejected_successfully(slot, channel, sub_direction)
            
            self.stats["flits_ejected"][channel] += 1
            if slot.etag_priority == PriorityLevel.T0:
                self.stats["t0_arbitrations"][channel] += 1
                
            self.logger.debug(f"CrossPoint {self.crosspoint_id} 成功下环flit {ejected_flit.flit_id} 从 {channel} 通道")
            
        return ejected_flit
        
    def process_ejection_to_fifos(self, node_fifos: Dict[str, Dict[str, Any]], cycle: int) -> None:
        """
        处理到eject_input_fifos的下环判断
        
        Args:
            node_fifos: 节点的eject_input_fifos
            cycle: 当前周期
        """
        # 检查每个管理方向的到达slice
        for direction in self.managed_directions:
            arrival_slice = self.slices[direction]["arrival"]
            if not arrival_slice:
                continue
                
            for channel in ["req", "rsp", "data"]:
                current_slot = arrival_slice.peek_current_slot(channel)
                if current_slot and current_slot.is_occupied:
                    flit = current_slot.flit
                    
                    # 检查是否应该在本节点下环
                    if self.should_eject_flit(flit):
                        # 检查目标eject_input_fifo是否有空间
                        if direction in node_fifos[channel]:
                            eject_fifo = node_fifos[channel][direction]
                            fifo_occupancy = len(eject_fifo.internal_queue)
                            fifo_depth = eject_fifo.depth
                            
                            # 尝试下环
                            ejected_flit = self.try_eject_flit(current_slot, channel, fifo_occupancy, fifo_depth)
                            if ejected_flit:
                                # 成功下环，写入eject_input_fifo
                                if eject_fifo.write_input(ejected_flit):
                                    self.logger.debug(f"CrossPoint {self.crosspoint_id} 成功下环flit到{direction}方向eject FIFO")
                                else:
                                    self.logger.warning(f"CrossPoint {self.crosspoint_id} 下环成功但写入eject FIFO失败")
                            else:
                                self.logger.debug(f"CrossPoint {self.crosspoint_id} 下环失败，flit继续在环路中传输")

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
        threshold = self._get_itag_threshold()
        
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
        self.itag_reservations[channel] = {
            "active": True,
            "slot_id": f"reserved_{self.node_id}_{channel}",
            "wait_cycles": 0
        }
        
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

    def step(self, cycle: int, node_inject_fifos: Dict[str, Dict[str, Any]], node_eject_fifos: Dict[str, Dict[str, Any]]) -> None:
        """
        执行一个周期的处理
        
        Args:
            cycle: 当前周期
            node_inject_fifos: 节点的inject_direction_fifos
            node_eject_fifos: 节点的eject_input_fifos
        """
        # 处理下环判断：从到达slice到eject_input_fifos
        self.process_ejection_to_fifos(node_eject_fifos, cycle)
        
        # 处理上环判断：从inject_direction_fifos到离开slice
        self.process_injection_from_fifos(node_inject_fifos, cycle)
        
        # 处理各通道的注入等待队列
        for channel in ["req", "rsp", "data"]:
            self._process_injection_queue(channel, cycle)
            
        # 更新I-Tag预约状态
        self._update_itag_reservations(cycle)

    def _process_injection_queue(self, channel: str, cycle: int) -> None:
        """处理注入等待队列"""
        if not self.injection_queues[channel]:
            return
            
        # 更新等待时间
        for i, (flit, wait_cycles) in enumerate(self.injection_queues[channel]):
            self.injection_queues[channel][i] = (flit, wait_cycles + 1)
            
            # 检查是否需要I-Tag预约
            if wait_cycles + 1 >= self._get_itag_threshold() and not self.itag_reservations[channel]["active"]:
                self.process_itag_request(flit, channel, wait_cycles + 1)
                
        # 尝试注入队首flit
        if self.injection_queues[channel]:
            flit, wait_cycles = self.injection_queues[channel][0]
            if self.try_inject_flit(flit, channel):
                self.injection_queues[channel].pop(0)

    def _update_itag_reservations(self, cycle: int) -> None:
        """更新I-Tag预约状态"""
        for channel in ["req", "rsp", "data"]:
            if self.itag_reservations[channel]["active"]:
                self.itag_reservations[channel]["wait_cycles"] += 1
                
                # 简化：假设预约在一定周期后生效
                if self.itag_reservations[channel]["wait_cycles"] > 10:
                    self.itag_reservations[channel]["active"] = False
                    self.itag_reservations[channel]["wait_cycles"] = 0

    def _get_etag_limits(self, sub_direction: str) -> Dict[str, int]:
        """获取E-Tag限制配置"""
        if sub_direction == "TL":
            return {"t2_max": 8, "t1_max": 15, "t0_max": float("inf")}
        elif sub_direction == "TR":
            return {"t2_max": 12, "t1_max": float("inf"), "t0_max": float("inf")}
        elif sub_direction == "TU":
            return {"t2_max": 8, "t1_max": 15, "t0_max": float("inf")}
        elif sub_direction == "TD":
            return {"t2_max": 12, "t1_max": float("inf"), "t0_max": float("inf")}
        else:
            return {"t2_max": 8, "t1_max": 15, "t0_max": float("inf")}

    def _get_itag_threshold(self) -> int:
        """获取I-Tag触发阈值"""
        if self.direction == CrossPointDirection.HORIZONTAL:
            return 80  # 简化配置
        else:
            return 80
            
    def _trigger_itag_reservation(self, direction: str, channel: str, cycle: int) -> bool:
        """触发I-Tag预约"""
        # 确定环路类型
        ring_type = "horizontal" if direction in ["TL", "TR"] else "vertical"
        
        # 获取departure slice
        departure_slice = self.slices[direction]["departure"]
        if not departure_slice:
            return False
            
        # 使用Tag管理器触发预约
        success = self.tag_manager.trigger_itag_reservation(
            channel, ring_type, departure_slice, cycle
        )
        
        if success:
            self.itag_reservations[channel]["active"] = True
            self.itag_reservations[channel]["slot_id"] = f"reserved_{self.node_id}_{channel}"
            self.itag_reservations[channel]["wait_cycles"] = 0
            
        return success

    def _get_sub_direction_from_channel(self, channel: str) -> str:
        """从通道获取子方向"""
        # 简化实现，实际需要根据具体路由策略确定
        if self.direction == CrossPointDirection.HORIZONTAL:
            return "TL"  # 或根据具体情况返回TR
        else:
            return "TU"  # 或根据具体情况返回TD

    def _check_t0_round_robin_grant(self, flit: CrossRingFlit, channel: str) -> bool:
        """检查T0级轮询仲裁授权"""
        current_index = self.etag_states[channel]["t0_round_robin"]
        self.etag_states[channel]["t0_round_robin"] = (current_index + 1) % 16
        return (flit.flit_id + current_index) % 2 == 0

    def _handle_eject_failure(self, slot: CrossRingSlot, channel: str) -> None:
        """处理下环失败，考虑E-Tag升级"""
        sub_direction = self._get_sub_direction_from_channel(channel)
        
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
            cycle = getattr(slot, 'cycle', 0)
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
            "injection_queue_lengths": {
                channel: len(queue) for channel, queue in self.injection_queues.items()
            },
            "itag_reservations": self.itag_reservations.copy(),
            "etag_states": self.etag_states.copy(),
            "stats": self.stats.copy(),
            "ring_slice_connected": {
                channel: slice is not None 
                for channel, slice in self.ring_slice_interfaces.items()
            }
        }


class CrossRingNode:
    """
    CrossRing节点类。

    实现CrossRing节点的内部结构和逻辑，包括：
    1. 注入/提取队列管理
    2. 环形缓冲区管理
    3. ETag/ITag拥塞控制
    4. 仲裁逻辑
    """

    def __init__(self, node_id: int, coordinates: Tuple[int, int], config: CrossRingConfig, logger: logging.Logger):
        """
        初始化CrossRing节点

        Args:
            node_id: 节点ID
            coordinates: 节点坐标 (x, y)
            config: CrossRing配置
            logger: 日志记录器
        """
        self.node_id = node_id
        self.coordinates = coordinates
        self.config = config
        self.logger = logger

        # IP注入缓冲区配置
        # 获取FIFO配置，如果没有则使用默认值
        iq_ch_depth = getattr(config, "iq_ch_depth", 10)
        iq_out_depth = getattr(config, "iq_out_depth", 8)

        # 连接的IP列表（默认每个节点连接一个IP，也可以扩展为多个）
        self.connected_ips = []  # 将存储连接的IP ID列表

        # 每个IP的inject channel_buffer - 结构：ip_inject_channel_buffers[ip_id][channel]
        self.ip_inject_channel_buffers = {}

        # 方向化的注入队列 - 5个方向的PipelinedFIFO，使用iq_out_depth
        self.inject_direction_fifos = {
            "req": {
                "TR": PipelinedFIFO(f"inject_req_TR_{node_id}", depth=iq_out_depth),
                "TL": PipelinedFIFO(f"inject_req_TL_{node_id}", depth=iq_out_depth),
                "TU": PipelinedFIFO(f"inject_req_TU_{node_id}", depth=iq_out_depth),
                "TD": PipelinedFIFO(f"inject_req_TD_{node_id}", depth=iq_out_depth),
                "EQ": PipelinedFIFO(f"inject_req_EQ_{node_id}", depth=iq_out_depth),
            },
            "rsp": {
                "TR": PipelinedFIFO(f"inject_rsp_TR_{node_id}", depth=iq_out_depth),
                "TL": PipelinedFIFO(f"inject_rsp_TL_{node_id}", depth=iq_out_depth),
                "TU": PipelinedFIFO(f"inject_rsp_TU_{node_id}", depth=iq_out_depth),
                "TD": PipelinedFIFO(f"inject_rsp_TD_{node_id}", depth=iq_out_depth),
                "EQ": PipelinedFIFO(f"inject_rsp_EQ_{node_id}", depth=iq_out_depth),
            },
            "data": {
                "TR": PipelinedFIFO(f"inject_data_TR_{node_id}", depth=iq_out_depth),
                "TL": PipelinedFIFO(f"inject_data_TL_{node_id}", depth=iq_out_depth),
                "TU": PipelinedFIFO(f"inject_data_TU_{node_id}", depth=iq_out_depth),
                "TD": PipelinedFIFO(f"inject_data_TD_{node_id}", depth=iq_out_depth),
                "EQ": PipelinedFIFO(f"inject_data_EQ_{node_id}", depth=iq_out_depth),
            },
        }
        # 获取eject相关的FIFO配置
        eq_in_depth = getattr(config, "eq_in_depth", 16)
        eq_ch_depth = getattr(config, "eq_ch_depth", 10)

        # 获取ring_bridge相关的FIFO配置
        rb_in_depth = getattr(config, "rb_in_depth", 16)
        rb_out_depth = getattr(config, "rb_out_depth", 8)

        # 每个IP的eject channel_buffer - 结构：ip_eject_channel_buffers[ip_id][channel]
        self.ip_eject_channel_buffers = {}

        # ring buffer输入的中间FIFO - 仅为ring buffer创建
        self.eject_input_fifos = {
            "req": {
                "TU": PipelinedFIFO(f"eject_in_req_TU_{node_id}", depth=eq_in_depth),
                "TD": PipelinedFIFO(f"eject_in_req_TD_{node_id}", depth=eq_in_depth),
                "TR": PipelinedFIFO(f"eject_in_req_TR_{node_id}", depth=eq_in_depth),
                "TL": PipelinedFIFO(f"eject_in_req_TL_{node_id}", depth=eq_in_depth),
            },
            "rsp": {
                "TU": PipelinedFIFO(f"eject_in_rsp_TU_{node_id}", depth=eq_in_depth),
                "TD": PipelinedFIFO(f"eject_in_rsp_TD_{node_id}", depth=eq_in_depth),
                "TR": PipelinedFIFO(f"eject_in_rsp_TR_{node_id}", depth=eq_in_depth),
                "TL": PipelinedFIFO(f"eject_in_rsp_TL_{node_id}", depth=eq_in_depth),
            },
            "data": {
                "TU": PipelinedFIFO(f"eject_in_data_TU_{node_id}", depth=eq_in_depth),
                "TD": PipelinedFIFO(f"eject_in_data_TD_{node_id}", depth=eq_in_depth),
                "TR": PipelinedFIFO(f"eject_in_data_TR_{node_id}", depth=eq_in_depth),
                "TL": PipelinedFIFO(f"eject_in_data_TL_{node_id}", depth=eq_in_depth),
            },
        }

        # ring_bridge输入FIFO - 为CrossPoint来源的flit创建
        self.ring_bridge_input_fifos = {
            "req": {
                "TR": PipelinedFIFO(f"ring_bridge_in_req_TR_{node_id}", depth=rb_in_depth),
                "TL": PipelinedFIFO(f"ring_bridge_in_req_TL_{node_id}", depth=rb_in_depth),
                "TU": PipelinedFIFO(f"ring_bridge_in_req_TU_{node_id}", depth=rb_in_depth),
                "TD": PipelinedFIFO(f"ring_bridge_in_req_TD_{node_id}", depth=rb_in_depth),
            },
            "rsp": {
                "TR": PipelinedFIFO(f"ring_bridge_in_rsp_TR_{node_id}", depth=rb_in_depth),
                "TL": PipelinedFIFO(f"ring_bridge_in_rsp_TL_{node_id}", depth=rb_in_depth),
                "TU": PipelinedFIFO(f"ring_bridge_in_rsp_TU_{node_id}", depth=rb_in_depth),
                "TD": PipelinedFIFO(f"ring_bridge_in_rsp_TD_{node_id}", depth=rb_in_depth),
            },
            "data": {
                "TR": PipelinedFIFO(f"ring_bridge_in_data_TR_{node_id}", depth=rb_in_depth),
                "TL": PipelinedFIFO(f"ring_bridge_in_data_TL_{node_id}", depth=rb_in_depth),
                "TU": PipelinedFIFO(f"ring_bridge_in_data_TU_{node_id}", depth=rb_in_depth),
                "TD": PipelinedFIFO(f"ring_bridge_in_data_TD_{node_id}", depth=rb_in_depth),
            },
        }

        # ring_bridge输出FIFO
        self.ring_bridge_output_fifos = {
            "req": {
                "EQ": PipelinedFIFO(f"ring_bridge_out_req_EQ_{node_id}", depth=rb_out_depth),
                "TR": PipelinedFIFO(f"ring_bridge_out_req_TR_{node_id}", depth=rb_out_depth),
                "TL": PipelinedFIFO(f"ring_bridge_out_req_TL_{node_id}", depth=rb_out_depth),
                "TU": PipelinedFIFO(f"ring_bridge_out_req_TU_{node_id}", depth=rb_out_depth),
                "TD": PipelinedFIFO(f"ring_bridge_out_req_TD_{node_id}", depth=rb_out_depth),
            },
            "rsp": {
                "EQ": PipelinedFIFO(f"ring_bridge_out_rsp_EQ_{node_id}", depth=rb_out_depth),
                "TR": PipelinedFIFO(f"ring_bridge_out_rsp_TR_{node_id}", depth=rb_out_depth),
                "TL": PipelinedFIFO(f"ring_bridge_out_rsp_TL_{node_id}", depth=rb_out_depth),
                "TU": PipelinedFIFO(f"ring_bridge_out_rsp_TU_{node_id}", depth=rb_out_depth),
                "TD": PipelinedFIFO(f"ring_bridge_out_rsp_TD_{node_id}", depth=rb_out_depth),
            },
            "data": {
                "EQ": PipelinedFIFO(f"ring_bridge_out_data_EQ_{node_id}", depth=rb_out_depth),
                "TR": PipelinedFIFO(f"ring_bridge_out_data_TR_{node_id}", depth=rb_out_depth),
                "TL": PipelinedFIFO(f"ring_bridge_out_data_TL_{node_id}", depth=rb_out_depth),
                "TU": PipelinedFIFO(f"ring_bridge_out_data_TU_{node_id}", depth=rb_out_depth),
                "TD": PipelinedFIFO(f"ring_bridge_out_data_TD_{node_id}", depth=rb_out_depth),
            },
        }

        # 环形缓冲区 - 保持命名但使用PipelinedFIFO
        self.ring_buffers = {
            "horizontal": {
                "req": {
                    "TR": PipelinedFIFO(f"ring_h_req_TR_{node_id}", depth=config.ring_buffer_depth),
                    "TL": PipelinedFIFO(f"ring_h_req_TL_{node_id}", depth=config.ring_buffer_depth),
                    "TD": PipelinedFIFO(f"ring_h_req_TD_{node_id}", depth=config.ring_buffer_depth),
                    "TU": PipelinedFIFO(f"ring_h_req_TU_{node_id}", depth=config.ring_buffer_depth),
                },
                "rsp": {
                    "TR": PipelinedFIFO(f"ring_h_rsp_TR_{node_id}", depth=config.ring_buffer_depth),
                    "TL": PipelinedFIFO(f"ring_h_rsp_TL_{node_id}", depth=config.ring_buffer_depth),
                    "TD": PipelinedFIFO(f"ring_h_rsp_TD_{node_id}", depth=config.ring_buffer_depth),
                    "TU": PipelinedFIFO(f"ring_h_rsp_TU_{node_id}", depth=config.ring_buffer_depth),
                },
                "data": {
                    "TR": PipelinedFIFO(f"ring_h_data_TR_{node_id}", depth=config.ring_buffer_depth),
                    "TL": PipelinedFIFO(f"ring_h_data_TL_{node_id}", depth=config.ring_buffer_depth),
                    "TD": PipelinedFIFO(f"ring_h_data_TD_{node_id}", depth=config.ring_buffer_depth),
                    "TU": PipelinedFIFO(f"ring_h_data_TU_{node_id}", depth=config.ring_buffer_depth),
                },
            },
            "vertical": {
                "req": {
                    "TR": PipelinedFIFO(f"ring_v_req_TR_{node_id}", depth=config.ring_buffer_depth),
                    "TL": PipelinedFIFO(f"ring_v_req_TL_{node_id}", depth=config.ring_buffer_depth),
                    "TD": PipelinedFIFO(f"ring_v_req_TD_{node_id}", depth=config.ring_buffer_depth),
                    "TU": PipelinedFIFO(f"ring_v_req_TU_{node_id}", depth=config.ring_buffer_depth),
                },
                "rsp": {
                    "TR": PipelinedFIFO(f"ring_v_rsp_TR_{node_id}", depth=config.ring_buffer_depth),
                    "TL": PipelinedFIFO(f"ring_v_rsp_TL_{node_id}", depth=config.ring_buffer_depth),
                    "TD": PipelinedFIFO(f"ring_v_rsp_TD_{node_id}", depth=config.ring_buffer_depth),
                    "TU": PipelinedFIFO(f"ring_v_rsp_TU_{node_id}", depth=config.ring_buffer_depth),
                },
                "data": {
                    "TR": PipelinedFIFO(f"ring_v_data_TR_{node_id}", depth=config.ring_buffer_depth),
                    "TL": PipelinedFIFO(f"ring_v_data_TL_{node_id}", depth=config.ring_buffer_depth),
                    "TD": PipelinedFIFO(f"ring_v_data_TD_{node_id}", depth=config.ring_buffer_depth),
                    "TU": PipelinedFIFO(f"ring_v_data_TU_{node_id}", depth=config.ring_buffer_depth),
                },
            },
        }

        # 拥塞控制状态
        self.etag_status = {
            "horizontal": {"req": False, "rsp": False, "data": False},
            "vertical": {"req": False, "rsp": False, "data": False},
        }
        self.itag_status = {
            "horizontal": {"req": False, "rsp": False, "data": False},
            "vertical": {"req": False, "rsp": False, "data": False},
        }

        # 仲裁状态 - 使用更准确的方向优先级
        self.arbitration_state = {
            "horizontal_priority": "inject",  # inject, ring_tr, ring_tl
            "vertical_priority": "inject",  # inject, ring_td, ring_tu
            "last_arbitration": {"horizontal": 0, "vertical": 0},
        }

        # 注入轮询仲裁器状态 - 为每个通道独立的轮询仲裁
        self.inject_arbitration_state = {
            "req": {
                "current_direction": 0,  # 当前轮询位置：0=TR, 1=TL, 2=TU, 3=TD, 4=EQ
                "directions": ["TR", "TL", "TU", "TD", "EQ"],
                "last_served": {"TR": 0, "TL": 0, "TU": 0, "TD": 0, "EQ": 0},
            },
            "rsp": {
                "current_direction": 0,
                "directions": ["TR", "TL", "TU", "TD", "EQ"],
                "last_served": {"TR": 0, "TL": 0, "TU": 0, "TD": 0, "EQ": 0},
            },
            "data": {
                "current_direction": 0,
                "directions": ["TR", "TL", "TU", "TD", "EQ"],
                "last_served": {"TR": 0, "TL": 0, "TU": 0, "TD": 0, "EQ": 0},
            },
        }

        # Eject轮询仲裁器状态 - 为每个通道独立的轮询仲裁
        self.eject_arbitration_state = {
            "req": {
                "current_source": 0,  # 当前输入源位置
                "current_ip": 0,  # 当前IP位置
                "sources": [],  # 动态根据路由策略设置
                "last_served_source": {},
                "last_served_ip": {},
            },
            "rsp": {
                "current_source": 0,
                "current_ip": 0,
                "sources": [],
                "last_served_source": {},
                "last_served_ip": {},
            },
            "data": {
                "current_source": 0,
                "current_ip": 0,
                "sources": [],
                "last_served_source": {},
                "last_served_ip": {},
            },
        }

        # Ring_bridge轮询仲裁器状态 - 为每个通道独立的轮询仲裁
        self.ring_bridge_arbitration_state = {
            "req": {
                "current_input": 0,  # 当前输入源位置
                "current_output": 0,  # 当前输出方向位置
                "input_sources": [],  # 动态根据路由策略设置
                "output_directions": [],  # 动态根据路由策略设置
                "last_served_input": {},
                "last_served_output": {},
            },
            "rsp": {
                "current_input": 0,
                "current_output": 0,
                "input_sources": [],
                "output_directions": [],
                "last_served_input": {},
                "last_served_output": {},
            },
            "data": {
                "current_input": 0,
                "current_output": 0,
                "input_sources": [],
                "output_directions": [],
                "last_served_input": {},
                "last_served_output": {},
            },
        }

        # 性能统计
        self.stats = {
            "injected_flits": {"req": 0, "rsp": 0, "data": 0},
            "ejected_flits": {"req": 0, "rsp": 0, "data": 0},
            "transferred_flits": {"horizontal": 0, "vertical": 0},
            "congestion_events": 0,
        }

        # 存储FIFO配置供后续使用
        self.iq_ch_depth = iq_ch_depth
        self.iq_out_depth = iq_out_depth
        self.eq_in_depth = eq_in_depth
        self.eq_ch_depth = eq_ch_depth
        self.rb_in_depth = rb_in_depth
        self.rb_out_depth = rb_out_depth

        # 初始化CrossPoint实例 - 每个节点有2个CrossPoint
        self.horizontal_crosspoint = CrossRingCrossPoint(
            crosspoint_id=f"node_{node_id}_horizontal", node_id=node_id, direction=CrossPointDirection.HORIZONTAL, config=config, logger=logger
        )

        self.vertical_crosspoint = CrossRingCrossPoint(
            crosspoint_id=f"node_{node_id}_vertical", node_id=node_id, direction=CrossPointDirection.VERTICAL, config=config, logger=logger
        )

        self.logger.debug(f"CrossRing节点初始化: ID={node_id}, 坐标={coordinates}")

    def set_routing_strategy_bias(self, routing_strategy: RoutingStrategy) -> None:
        """
        根据路由策略设置仲裁偏向

        Args:
            routing_strategy: 路由策略
        """
        if routing_strategy == RoutingStrategy.XY:
            # XY路由：稍微偏向水平方向
            self.routing_bias = {"horizontal": 1.2, "vertical": 1.0}
        elif routing_strategy == RoutingStrategy.YX:
            # YX路由：稍微偏向垂直方向
            self.routing_bias = {"horizontal": 1.0, "vertical": 1.2}
        else:
            # 其他策略：均衡
            self.routing_bias = {"horizontal": 1.0, "vertical": 1.0}

        self.logger.debug(f"节点{self.node_id}设置路由偏向: {routing_strategy.value} -> {self.routing_bias}")

    def connect_ip(self, ip_id: str) -> bool:
        """
        连接一个IP到当前节点

        Args:
            ip_id: IP的标识符

        Returns:
            是否成功连接
        """
        if ip_id not in self.connected_ips:
            self.connected_ips.append(ip_id)

            # 为这个IP创建inject channel_buffer
            self.ip_inject_channel_buffers[ip_id] = {
                "req": PipelinedFIFO(f"ip_inject_channel_req_{ip_id}_{self.node_id}", depth=self.iq_ch_depth),
                "rsp": PipelinedFIFO(f"ip_inject_channel_rsp_{ip_id}_{self.node_id}", depth=self.iq_ch_depth),
                "data": PipelinedFIFO(f"ip_inject_channel_data_{ip_id}_{self.node_id}", depth=self.iq_ch_depth),
            }

            # 为这个IP创建eject channel_buffer
            self.ip_eject_channel_buffers[ip_id] = {
                "req": PipelinedFIFO(f"ip_eject_channel_req_{ip_id}_{self.node_id}", depth=self.eq_ch_depth),
                "rsp": PipelinedFIFO(f"ip_eject_channel_rsp_{ip_id}_{self.node_id}", depth=self.eq_ch_depth),
                "data": PipelinedFIFO(f"ip_eject_channel_data_{ip_id}_{self.node_id}", depth=self.eq_ch_depth),
            }

            # 更新eject仲裁状态中的IP列表
            self._update_eject_arbitration_ips()

            self.logger.debug(f"节点{self.node_id}成功连接IP {ip_id}")
            return True
        else:
            self.logger.warning(f"IP {ip_id}已经连接到节点{self.node_id}")
            return False

    def disconnect_ip(self, ip_id: str) -> None:
        """
        断开IP连接

        Args:
            ip_id: IP的标识符
        """
        if ip_id in self.connected_ips:
            self.connected_ips.remove(ip_id)
            del self.ip_inject_channel_buffers[ip_id]
            del self.ip_eject_channel_buffers[ip_id]

            # 更新eject仲裁状态中的IP列表
            self._update_eject_arbitration_ips()

            self.logger.debug(f"节点{self.node_id}断开IP {ip_id}连接")
        else:
            self.logger.warning(f"IP {ip_id}未连接到节点{self.node_id}")

    def get_connected_ips(self) -> List[str]:
        """
        获取连接的IP列表

        Returns:
            连接的IP ID列表
        """
        return self.connected_ips.copy()
        
    def get_crosspoint(self, direction: str) -> Optional[CrossRingCrossPoint]:
        """
        获取指定方向的CrossPoint
        
        Args:
            direction: 方向 ("horizontal" 或 "vertical")
            
        Returns:
            CrossPoint实例，如果不存在则返回None
        """
        if direction == "horizontal":
            return self.horizontal_crosspoint
        elif direction == "vertical":
            return self.vertical_crosspoint
        else:
            return None
            
    def step_crosspoints(self, cycle: int) -> None:
        """
        执行一个周期的CrossPoint处理
        
        Args:
            cycle: 当前周期
        """
        # 执行水平CrossPoint处理
        if self.horizontal_crosspoint:
            self.horizontal_crosspoint.step(
                cycle, 
                self.inject_direction_fifos, 
                self.eject_input_fifos
            )
            
        # 执行垂直CrossPoint处理
        if self.vertical_crosspoint:
            self.vertical_crosspoint.step(
                cycle,
                self.inject_direction_fifos,
                self.eject_input_fifos
            )

    def _get_ring_bridge_input_sources(self) -> List[str]:
        """
        根据路由策略获取ring_bridge的输入源

        Returns:
            输入源列表
        """
        # 获取路由策略
        routing_strategy = getattr(self.config, "routing_strategy", "XY")
        if hasattr(routing_strategy, "value"):
            routing_strategy = routing_strategy.value

        if routing_strategy == "XY":
            return ["IQ_TU", "IQ_TD", "RB_TR", "RB_TL"]
        elif routing_strategy == "YX":
            return ["IQ_TR", "IQ_TL", "RB_TU", "RB_TD"]
        else:  # ADAPTIVE 或其他
            return ["IQ_TU", "IQ_TD", "IQ_TR", "IQ_TL", "RB_TR", "RB_TL", "RB_TU", "RB_TD"]

    def _get_ring_bridge_output_directions(self) -> List[str]:
        """
        根据路由策略获取ring_bridge的输出方向

        Returns:
            输出方向列表
        """
        # 获取路由策略
        routing_strategy = getattr(self.config, "routing_strategy", "XY")
        if hasattr(routing_strategy, "value"):
            routing_strategy = routing_strategy.value

        base = ["EQ"]  # 总是包含EQ
        if routing_strategy == "XY":
            return base + ["TU", "TD"]
        elif routing_strategy == "YX":
            return base + ["TR", "TL"]
        else:  # ADAPTIVE 或其他
            return base + ["TU", "TD", "TR", "TL"]

    def _initialize_ring_bridge_arbitration(self) -> None:
        """初始化ring_bridge仲裁的源和方向列表"""
        input_sources = self._get_ring_bridge_input_sources()
        output_directions = self._get_ring_bridge_output_directions()

        for channel in ["req", "rsp", "data"]:
            arb_state = self.ring_bridge_arbitration_state[channel]
            arb_state["input_sources"] = input_sources.copy()
            arb_state["output_directions"] = output_directions.copy()
            arb_state["last_served_input"] = {source: 0 for source in input_sources}
            arb_state["last_served_output"] = {direction: 0 for direction in output_directions}

    def process_ring_bridge_arbitration(self, cycle: int) -> None:
        """
        处理ring_bridge的轮询仲裁

        Args:
            cycle: 当前周期
        """
        # 首先初始化源和方向列表（如果还没有初始化）
        if not self.ring_bridge_arbitration_state["req"]["input_sources"]:
            self._initialize_ring_bridge_arbitration()

        # 为每个通道处理ring_bridge仲裁
        for channel in ["req", "rsp", "data"]:
            self._process_channel_ring_bridge_arbitration(channel, cycle)

    def _process_channel_ring_bridge_arbitration(self, channel: str, cycle: int) -> None:
        """
        处理单个通道的ring_bridge仲裁

        Args:
            channel: 通道类型
            cycle: 当前周期
        """
        arb_state = self.ring_bridge_arbitration_state[channel]
        input_sources = arb_state["input_sources"]

        # 轮询所有输入源
        for input_attempt in range(len(input_sources)):
            current_input_idx = arb_state["current_input"]
            input_source = input_sources[current_input_idx]

            # 获取来自当前输入源的flit
            flit = self._get_flit_from_ring_bridge_input(input_source, channel)
            if flit is not None:
                # 找到flit，现在确定输出方向并分配
                output_direction = self._determine_ring_bridge_output_direction(flit)
                if self._assign_flit_to_ring_bridge_output(flit, output_direction, channel, cycle):
                    # 成功分配，更新输入仲裁状态
                    arb_state["last_served_input"][input_source] = cycle
                    break

            # 移动到下一个输入源
            arb_state["current_input"] = (current_input_idx + 1) % len(input_sources)

    def _get_flit_from_ring_bridge_input(self, input_source: str, channel: str) -> Optional[CrossRingFlit]:
        """
        从指定的ring_bridge输入源获取flit

        Args:
            input_source: 输入源名称 (如 "IQ_TU", "RB_TR")
            channel: 通道类型

        Returns:
            获取的flit，如果没有则返回None
        """
        if input_source.startswith("IQ_"):
            # 直接从inject_direction_fifos获取
            direction = input_source[3:]  # 去掉"IQ_"前缀
            iq_fifo = self.inject_direction_fifos[channel][direction]
            if iq_fifo.valid_signal():
                return iq_fifo.read_output()

        elif input_source.startswith("RB_"):
            # 从ring_bridge_input_fifos获取
            direction = input_source[3:]  # 去掉"RB_"前缀
            rb_fifo = self.ring_bridge_input_fifos[channel][direction]
            if rb_fifo.valid_signal():
                return rb_fifo.read_output()

        return None

    def _determine_ring_bridge_output_direction(self, flit: CrossRingFlit) -> str:
        """
        确定flit在ring_bridge中的输出方向

        Args:
            flit: 要路由的flit

        Returns:
            输出方向 ("EQ", "TR", "TL", "TU", "TD")
        """
        # 首先检查是否是本地目标
        if self._is_local_destination(flit):
            return "EQ"

        # 否则，根据路由策略和目标位置确定输出方向
        return self._calculate_routing_direction(flit)

    def _assign_flit_to_ring_bridge_output(self, flit: CrossRingFlit, output_direction: str, channel: str, cycle: int) -> bool:
        """
        将flit分配到ring_bridge输出FIFO

        Args:
            flit: 要分配的flit
            output_direction: 输出方向
            channel: 通道类型
            cycle: 当前周期

        Returns:
            是否成功分配
        """
        # 检查输出FIFO是否可用
        output_fifo = self.ring_bridge_output_fifos[channel][output_direction]
        if output_fifo.ready_signal():
            if output_fifo.write_input(flit):
                # 成功分配，更新输出仲裁状态
                arb_state = self.ring_bridge_arbitration_state[channel]
                arb_state["last_served_output"][output_direction] = cycle

                self.logger.debug(f"节点{self.node_id}成功将{channel}通道flit分配到ring_bridge输出{output_direction}")
                return True

        return False

    def add_to_ring_bridge_input(self, flit: CrossRingFlit, direction: str, channel: str) -> bool:
        """
        CrossPoint向ring_bridge输入添加flit

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

    def get_ring_bridge_eq_flit(self, channel: str) -> Optional[CrossRingFlit]:
        """
        从ring_bridge的EQ输出获取flit (为eject队列提供)

        Args:
            channel: 通道类型

        Returns:
            获取的flit，如果没有则返回None
        """
        eq_fifo = self.ring_bridge_output_fifos[channel]["EQ"]
        if eq_fifo.valid_signal():
            return eq_fifo.read_output()
        return None

    def get_ring_bridge_output_flit(self, direction: str, channel: str) -> Optional[CrossRingFlit]:
        """
        从ring_bridge的指定方向输出获取flit

        Args:
            direction: 输出方向 ("TR", "TL", "TU", "TD")
            channel: 通道类型

        Returns:
            获取的flit，如果没有则返回None
        """
        output_fifo = self.ring_bridge_output_fifos[channel][direction]
        if output_fifo.valid_signal():
            return output_fifo.read_output()
        return None

    def _update_eject_arbitration_ips(self) -> None:
        """更新eject仲裁状态中的IP列表"""
        for channel in ["req", "rsp", "data"]:
            arb_state = self.eject_arbitration_state[channel]
            # 重置IP相关的仲裁状态
            arb_state["current_ip"] = 0
            arb_state["last_served_ip"] = {ip_id: 0 for ip_id in self.connected_ips}

    def _get_active_eject_sources(self) -> List[str]:
        """
        根据路由策略获取活跃的eject输入源

        Returns:
            输入源列表
        """
        # 获取路由策略
        routing_strategy = getattr(self.config, "routing_strategy", "XY")
        if hasattr(routing_strategy, "value"):
            routing_strategy = routing_strategy.value

        # 这两个源总是存在
        sources = ["IQ_EQ", "ring_bridge_EQ"]

        if routing_strategy == "XY":
            sources.extend(["TU", "TD"])
        elif routing_strategy == "YX":
            sources.extend(["TR", "TL"])
        else:  # ADAPTIVE 或其他
            sources.extend(["TU", "TD", "TR", "TL"])

        return sources

    def _initialize_eject_arbitration_sources(self) -> None:
        """初始化eject仲裁的源列表"""
        active_sources = self._get_active_eject_sources()

        for channel in ["req", "rsp", "data"]:
            arb_state = self.eject_arbitration_state[channel]
            arb_state["sources"] = active_sources.copy()
            arb_state["last_served_source"] = {source: 0 for source in active_sources}

    def process_eject_arbitration(self, cycle: int) -> None:
        """
        处理eject队列的轮询仲裁

        Args:
            cycle: 当前周期
        """
        # 首先初始化源列表（如果还没有初始化）
        if not self.eject_arbitration_state["req"]["sources"]:
            self._initialize_eject_arbitration_sources()

        # 为每个通道处理eject仲裁
        for channel in ["req", "rsp", "data"]:
            self._process_channel_eject_arbitration(channel, cycle)

    def _process_channel_eject_arbitration(self, channel: str, cycle: int) -> None:
        """
        处理单个通道的eject仲裁

        Args:
            channel: 通道类型
            cycle: 当前周期
        """
        if not self.connected_ips:
            return  # 没有连接的IP

        arb_state = self.eject_arbitration_state[channel]
        sources = arb_state["sources"]

        # 轮询所有输入源
        for source_attempt in range(len(sources)):
            current_source_idx = arb_state["current_source"]
            source = sources[current_source_idx]

            # 获取来自当前源的flit
            flit = self._get_flit_from_eject_source(source, channel)
            if flit is not None:
                # 找到flit，现在轮询分配给IP
                if self._assign_flit_to_ip(flit, channel, cycle):
                    # 成功分配，更新源仲裁状态
                    arb_state["last_served_source"][source] = cycle
                    break

            # 移动到下一个源
            arb_state["current_source"] = (current_source_idx + 1) % len(sources)

    def _get_flit_from_eject_source(self, source: str, channel: str) -> Optional[CrossRingFlit]:
        """
        从指定的eject源获取flit

        Args:
            source: 输入源名称
            channel: 通道类型

        Returns:
            获取的flit，如果没有则返回None
        """
        if source == "IQ_EQ":
            # 直接从inject_direction_fifos的EQ获取
            eq_fifo = self.inject_direction_fifos[channel]["EQ"]
            if eq_fifo.valid_signal():
                return eq_fifo.read_output()

        elif source == "ring_bridge_EQ":
            # 从ring_bridge的EQ输出获取
            return self.get_ring_bridge_eq_flit(channel)

        elif source in ["TU", "TD", "TR", "TL"]:
            # 从eject_input_fifos获取
            input_fifo = self.eject_input_fifos[channel][source]
            if input_fifo.valid_signal():
                return input_fifo.read_output()

        return None

    def _assign_flit_to_ip(self, flit: CrossRingFlit, channel: str, cycle: int) -> bool:
        """
        将flit分配给IP

        Args:
            flit: 要分配的flit
            channel: 通道类型
            cycle: 当前周期

        Returns:
            是否成功分配
        """
        if not self.connected_ips:
            return False

        arb_state = self.eject_arbitration_state[channel]

        # 轮询所有连接的IP
        for ip_attempt in range(len(self.connected_ips)):
            current_ip_idx = arb_state["current_ip"]
            ip_id = self.connected_ips[current_ip_idx]

            # 检查IP的eject channel buffer是否可用
            eject_buffer = self.ip_eject_channel_buffers[ip_id][channel]
            if eject_buffer.ready_signal():
                # 分配给这个IP
                if eject_buffer.write_input(flit):
                    # 成功分配，更新IP仲裁状态
                    arb_state["last_served_ip"][ip_id] = cycle
                    arb_state["current_ip"] = (current_ip_idx + 1) % len(self.connected_ips)

                    # 更新统计
                    self.stats["ejected_flits"][channel] += 1

                    self.logger.debug(f"节点{self.node_id}成功将{channel}通道flit分配给IP {ip_id}")
                    return True

            # 移动到下一个IP
            arb_state["current_ip"] = (current_ip_idx + 1) % len(self.connected_ips)

        return False

    def get_eject_flit(self, ip_id: str, channel: str) -> Optional[CrossRingFlit]:
        """
        IP从其eject channel buffer获取flit

        Args:
            ip_id: IP标识符
            channel: 通道类型

        Returns:
            获取的flit，如果没有则返回None
        """
        if ip_id not in self.connected_ips:
            self.logger.error(f"IP {ip_id}未连接到节点{self.node_id}")
            return None

        eject_buffer = self.ip_eject_channel_buffers[ip_id][channel]
        if eject_buffer.valid_signal():
            return eject_buffer.read_output()

        return None

    def update_state(self, cycle: int) -> None:
        """
        更新节点状态

        Args:
            cycle: 当前周期
        """
        # 阶段1：组合逻辑阶段 - 更新所有FIFO的组合逻辑
        self._step_compute_phase()

        # 处理CrossPoint的slot管理
        self.horizontal_crosspoint.step(cycle, self.inject_direction_fifos, self.eject_input_fifos)
        self.vertical_crosspoint.step(cycle, self.inject_direction_fifos, self.eject_input_fifos)

        # 处理注入队列的轮询仲裁
        self.process_inject_arbitration(cycle)

        # 处理ring_bridge的轮询仲裁
        self.process_ring_bridge_arbitration(cycle)

        # 处理eject队列的轮询仲裁
        self.process_eject_arbitration(cycle)

        # 更新仲裁状态
        self._update_arbitration_state(cycle)

        # 更新拥塞控制状态
        self._update_congestion_state()

        # 阶段2：时序逻辑阶段 - 更新所有FIFO的寄存器
        self._step_update_phase()

    def _step_compute_phase(self) -> None:
        """更新所有FIFO的组合逻辑阶段"""
        # 更新IP inject channel buffers
        for ip_id in self.connected_ips:
            for channel in ["req", "rsp", "data"]:
                self.ip_inject_channel_buffers[ip_id][channel].step_compute_phase()
                self.ip_eject_channel_buffers[ip_id][channel].step_compute_phase()

        # 更新inject direction FIFOs
        for channel in ["req", "rsp", "data"]:
            for direction in ["TR", "TL", "TU", "TD", "EQ"]:
                self.inject_direction_fifos[channel][direction].step_compute_phase()

        # 更新eject input FIFOs
        for channel in ["req", "rsp", "data"]:
            for direction in ["TU", "TD", "TR", "TL"]:
                self.eject_input_fifos[channel][direction].step_compute_phase()

        # 更新ring_bridge input/output FIFOs
        for channel in ["req", "rsp", "data"]:
            for direction in ["TR", "TL", "TU", "TD"]:
                self.ring_bridge_input_fifos[channel][direction].step_compute_phase()
            for direction in ["EQ", "TR", "TL", "TU", "TD"]:
                self.ring_bridge_output_fifos[channel][direction].step_compute_phase()

        # 更新ring buffers
        for direction in ["horizontal", "vertical"]:
            for channel in ["req", "rsp", "data"]:
                for ring_dir in ["TR", "TL", "TD", "TU"]:
                    self.ring_buffers[direction][channel][ring_dir].step_compute_phase()

    def _step_update_phase(self) -> None:
        """更新所有FIFO的时序逻辑阶段"""
        # 更新IP inject channel buffers
        for ip_id in self.connected_ips:
            for channel in ["req", "rsp", "data"]:
                self.ip_inject_channel_buffers[ip_id][channel].step_update_phase()
                self.ip_eject_channel_buffers[ip_id][channel].step_update_phase()

        # 更新inject direction FIFOs
        for channel in ["req", "rsp", "data"]:
            for direction in ["TR", "TL", "TU", "TD", "EQ"]:
                self.inject_direction_fifos[channel][direction].step_update_phase()

        # 更新eject input FIFOs
        for channel in ["req", "rsp", "data"]:
            for direction in ["TU", "TD", "TR", "TL"]:
                self.eject_input_fifos[channel][direction].step_update_phase()

        # 更新ring_bridge input/output FIFOs
        for channel in ["req", "rsp", "data"]:
            for direction in ["TR", "TL", "TU", "TD"]:
                self.ring_bridge_input_fifos[channel][direction].step_update_phase()
            for direction in ["EQ", "TR", "TL", "TU", "TD"]:
                self.ring_bridge_output_fifos[channel][direction].step_update_phase()

        # 更新ring buffers
        for direction in ["horizontal", "vertical"]:
            for channel in ["req", "rsp", "data"]:
                for ring_dir in ["TR", "TL", "TD", "TU"]:
                    self.ring_buffers[direction][channel][ring_dir].step_update_phase()

    def _update_arbitration_state(self, cycle: int) -> None:
        """
        更新仲裁状态

        Args:
            cycle: 当前周期
        """
        # 检查是否需要重置仲裁优先级
        for direction in ["horizontal", "vertical"]:
            last_arbitration = self.arbitration_state["last_arbitration"][direction]
            if cycle - last_arbitration > self.config.arbitration_timeout:
                # 重置为默认优先级
                self.arbitration_state[f"{direction}_priority"] = "inject"
                self.logger.debug(f"节点{self.node_id}的{direction}仲裁状态重置为默认")

    def _update_congestion_state(self) -> None:
        """更新拥塞控制状态"""
        # 更新ETag状态
        for direction in ["horizontal", "vertical"]:
            for channel in ["req", "rsp", "data"]:
                # 检查eject input fifos的拥塞情况
                eject_congestion = False
                eject_threshold = self.eq_in_depth * 0.8

                for eject_dir in ["TR", "TL", "TD", "TU"]:
                    eject_fifo = self.eject_input_fifos[channel][eject_dir]
                    buffer_occupancy = len(eject_fifo.internal_queue)
                    if buffer_occupancy >= eject_threshold:
                        eject_congestion = True
                        break

                # 检查ring缓冲区的拥塞情况
                ring_buffers = self.ring_buffers[direction][channel]
                ring_congestion = False

                for ring_dir in ["TR", "TL", "TD", "TU"]:
                    buffer_occupancy = len(ring_buffers[ring_dir].internal_queue)
                    ring_threshold = self.config.ring_buffer_depth * 0.8
                    if buffer_occupancy >= ring_threshold:
                        ring_congestion = True
                        break

                # 设置ETag状态
                old_status = self.etag_status[direction][channel]
                new_status = eject_congestion or ring_congestion

                if old_status != new_status:
                    self.etag_status[direction][channel] = new_status
                    if new_status:
                        self.stats["congestion_events"] += 1
                        self.logger.debug(f"节点{self.node_id}的{direction} {channel} ETag状态变为拥塞")
                    else:
                        self.logger.debug(f"节点{self.node_id}的{direction} {channel} ETag状态变为畅通")

    def can_inject_flit(self, channel: str, direction: str) -> bool:
        """
        检查是否可以注入flit

        Args:
            channel: 通道类型 ("req", "rsp", "data")
            direction: 注入方向 ("horizontal", "vertical")

        Returns:
            是否可以注入
        """
        # 检查拥塞状态
        if self.etag_status[direction][channel]:
            return False

        # 检查仲裁状态
        if self.arbitration_state[f"{direction}_priority"] != "inject":
            return False

        return True

    def inject_flit(self, flit: CrossRingFlit, channel: str, direction: str, dir_code: str, cycle: int) -> bool:
        """
        注入flit到环形缓冲区

        Args:
            flit: 要注入的flit
            channel: 通道类型 ("req", "rsp", "data")
            direction: 注入方向 ("horizontal", "vertical")
            dir_code: 具体方向代码 ("TR", "TL", "TD", "TU")
            cycle: 当前周期

        Returns:
            是否成功注入
        """
        # 检查环形缓冲区是否有空间
        ring_buffer = self.ring_buffers[direction][channel][dir_code]
        if len(ring_buffer) >= self.config.ring_buffer_depth:
            return False

        # 注入flit
        ring_buffer.append(flit)
        flit.network_entry_cycle = cycle

        # 更新统计
        self.stats["injected_flits"][channel] += 1

        # 更新仲裁状态
        # 将方向代码映射到仲裁优先级
        dir_priority_map = {"TR": "ring_tr", "TL": "ring_tl", "TD": "ring_td", "TU": "ring_tu"}
        self.arbitration_state[f"{direction}_priority"] = dir_priority_map[dir_code]
        self.arbitration_state["last_arbitration"][direction] = cycle

        return True

    def can_transfer_flit(self, direction: str, dir_code: str, channel: str) -> bool:
        """
        检查是否可以传输flit

        Args:
            direction: 传输方向 ("horizontal", "vertical")
            dir_code: 具体方向代码 ("TR", "TL", "TD", "TU")
            channel: 通道类型 ("req", "rsp", "data")

        Returns:
            是否可以传输
        """
        # 检查环形缓冲区是否有flit
        ring_buffer = self.ring_buffers[direction][channel][dir_code]
        if not ring_buffer:
            return False

        # 检查仲裁状态
        # 将方向代码映射到仲裁优先级
        priority_map = {"TR": "ring_tr", "TL": "ring_tl", "TD": "ring_td", "TU": "ring_tu"}
        requester = priority_map[dir_code]

        if self.arbitration_state[f"{direction}_priority"] != requester:
            return False

        return True

    def transfer_flit(self, direction: str, dir_code: str, channel: str, cycle: int) -> Optional[CrossRingFlit]:
        """
        从环形缓冲区传输flit

        Args:
            direction: 传输方向 ("horizontal", "vertical")
            dir_code: 具体方向代码 ("TR", "TL", "TD", "TU")
            channel: 通道类型 ("req", "rsp", "data")
            cycle: 当前周期

        Returns:
            传输的flit，如果没有则返回None
        """
        # 检查环形缓冲区是否有flit
        ring_buffer = self.ring_buffers[direction][channel][dir_code]
        if not ring_buffer:
            return None

        # 传输flit
        flit = ring_buffer.pop(0)

        # 更新统计
        self.stats["transferred_flits"][direction] += 1

        # 更新仲裁状态 - 轮转优先级
        priority_map = {"ring_tr": "ring_tl", "ring_tl": "inject", "ring_td": "ring_tu", "ring_tu": "inject"}

        # 将方向代码映射到仲裁优先级
        dir_priority_map = {"TR": "ring_tr", "TL": "ring_tl", "TD": "ring_td", "TU": "ring_tu"}
        current_priority = dir_priority_map[dir_code]

        # 更新为下一个优先级
        self.arbitration_state[f"{direction}_priority"] = priority_map.get(current_priority, "inject")
        self.arbitration_state["last_arbitration"][direction] = cycle

        return flit

    def receive_flit(self, flit: CrossRingFlit, direction: str, dir_code: str, channel: str) -> bool:
        """
        接收flit到环形缓冲区

        Args:
            flit: 要接收的flit
            direction: 接收方向 ("horizontal", "vertical")
            dir_code: 具体方向代码 ("TR", "TL", "TD", "TU")
            channel: 通道类型 ("req", "rsp", "data")

        Returns:
            是否成功接收
        """
        # 检查环形缓冲区是否有空间
        ring_buffer = self.ring_buffers[direction][channel][dir_code]
        if len(ring_buffer) >= self.config.ring_buffer_depth:
            return False

        # 接收flit
        ring_buffer.append(flit)

        return True

    def add_to_inject_queue(self, flit: CrossRingFlit, channel: str, ip_id: str) -> bool:
        """
        特定IP注入flit到其对应的channel_buffer

        Args:
            flit: 要添加的flit
            channel: 通道类型 ("req", "rsp", "data")
            ip_id: IP标识符

        Returns:
            是否成功添加
        """
        # 检查IP是否已连接
        if ip_id not in self.connected_ips:
            self.logger.error(f"IP {ip_id}未连接到节点{self.node_id}")
            return False

        # 获取对应IP的inject channel_buffer
        channel_buffer = self.ip_inject_channel_buffers[ip_id][channel]
        if not channel_buffer.ready_signal():
            self.logger.debug(f"节点{self.node_id}的IP {ip_id} {channel}通道缓冲区已满，无法注入flit")
            return False

        success = channel_buffer.write_input(flit)
        if success:
            self.logger.debug(f"节点{self.node_id}的IP {ip_id}成功注入flit到{channel}通道缓冲区")
        return success

    def process_inject_arbitration(self, cycle: int) -> None:
        """
        处理注入队列的轮询仲裁

        Args:
            cycle: 当前周期
        """
        # 为每个连接的IP和每个通道类型进行仲裁
        for ip_id in self.connected_ips:
            for channel in ["req", "rsp", "data"]:
                self._process_ip_channel_inject_arbitration(ip_id, channel, cycle)

    def _process_ip_channel_inject_arbitration(self, ip_id: str, channel: str, cycle: int) -> None:
        """
        处理特定IP和通道的注入仲裁

        Args:
            ip_id: IP标识符
            channel: 通道类型
            cycle: 当前周期
        """
        # 检查IP的inject channel_buffer是否有数据
        channel_buffer = self.ip_inject_channel_buffers[ip_id][channel]
        if not channel_buffer.valid_signal():
            return

        # 获取当前仲裁状态
        arb_state = self.inject_arbitration_state[channel]

        # 首先peek flit来确定正确的路由方向
        flit = channel_buffer.peek_output()
        if flit is None:
            return

        # 计算正确的路由方向
        correct_direction = self._calculate_routing_direction(flit)

        # 检查正确方向的FIFO是否可用
        target_fifo = self.inject_direction_fifos[channel][correct_direction]
        if target_fifo.ready_signal():
            # 现在读取并传输flit
            flit = channel_buffer.read_output()
            if flit is not None and target_fifo.write_input(flit):
                # 成功传输，更新仲裁状态
                arb_state["last_served"][correct_direction] = cycle
                self.logger.debug(f"节点{self.node_id}成功将IP {ip_id} {channel}通道flit仲裁到{correct_direction}方向")
        else:
            # 目标方向FIFO不可用，等待下个周期
            self.logger.debug(f"节点{self.node_id} IP {ip_id} {channel}通道目标方向{correct_direction}FIFO不可用，等待")

    def _should_route_to_direction(self, flit: CrossRingFlit, direction: str) -> bool:
        """
        判断flit是否应该路由到指定方向

        Args:
            flit: 要判断的flit
            direction: 目标方向

        Returns:
            是否应该路由到该方向
        """
        # 如果是EQ方向，检查是否是本地节点
        if direction == "EQ":
            return self._is_local_destination(flit)

        # 对于其他方向，根据路由算法决定
        return self._calculate_routing_direction(flit) == direction

    def _is_local_destination(self, flit: CrossRingFlit) -> bool:
        """
        检查flit是否应该在本地弹出

        Args:
            flit: 要检查的flit

        Returns:
            是否是本地目标
        """
        if hasattr(flit, "destination") and flit.destination == self.node_id:
            return True
        if hasattr(flit, "dest_node_id") and flit.dest_node_id == self.node_id:
            return True
        if hasattr(flit, "dest_coordinates"):
            dest_x, dest_y = flit.dest_coordinates
            curr_x, curr_y = self.coordinates
            if dest_x == curr_x and dest_y == curr_y:
                return True
        return False

    def _calculate_routing_direction(self, flit: CrossRingFlit) -> str:
        """
        根据配置的路由策略计算flit的路由方向

        Args:
            flit: 要路由的flit

        Returns:
            路由方向（"TR", "TL", "TU", "TD", "EQ"）
        """
        # 获取目标坐标
        if hasattr(flit, "dest_coordinates"):
            dest_x, dest_y = flit.dest_coordinates
        else:
            # 如果没有坐标信息，默认返回TR
            return "TR"

        curr_x, curr_y = self.coordinates

        # 如果已经到达目标位置
        if dest_x == curr_x and dest_y == curr_y:
            return "EQ"  # 本地

        # 获取路由策略，默认为XY
        routing_strategy = getattr(self.config, "routing_strategy", "XY")
        if hasattr(routing_strategy, "value"):
            routing_strategy = routing_strategy.value

        return self._apply_routing_strategy(curr_x, curr_y, dest_x, dest_y, routing_strategy)

    def _apply_routing_strategy(self, curr_x: int, curr_y: int, dest_x: int, dest_y: int, strategy: str) -> str:
        """
        应用具体的路由策略

        Args:
            curr_x, curr_y: 当前坐标
            dest_x, dest_y: 目标坐标
            strategy: 路由策略 ("XY", "YX", "ADAPTIVE")

        Returns:
            路由方向
        """
        if strategy == "XY":
            # XY路由：先水平后垂直
            if dest_x > curr_x:
                return "TR"  # 向右
            elif dest_x < curr_x:
                return "TL"  # 向左
            elif dest_y > curr_y:
                return "TU"  # 向上
            elif dest_y < curr_y:
                return "TD"  # 向下
            else:
                return "EQ"  # 本地

        elif strategy == "YX":
            # YX路由：先垂直后水平
            if dest_y > curr_y:
                return "TU"  # 向上
            elif dest_y < curr_y:
                return "TD"  # 向下
            elif dest_x > curr_x:
                return "TR"  # 向右
            elif dest_x < curr_x:
                return "TL"  # 向左
            else:
                return "EQ"  # 本地

        elif strategy == "ADAPTIVE":
            # 自适应路由：可以选择较少拥塞的维度
            # 这里可以实现更复杂的逻辑，比如检查拥塞状态
            return self._adaptive_routing_decision(curr_x, curr_y, dest_x, dest_y)

        else:
            # 未知策略，默认使用XY
            self.logger.warning(f"未知路由策略 {strategy}，使用XY路由")
            return self._apply_routing_strategy(curr_x, curr_y, dest_x, dest_y, "XY")

    def _adaptive_routing_decision(self, curr_x: int, curr_y: int, dest_x: int, dest_y: int) -> str:
        """
        自适应路由决策（可以根据拥塞状态选择路径）

        Args:
            curr_x, curr_y: 当前坐标
            dest_x, dest_y: 目标坐标

        Returns:
            路由方向
        """
        # 检查是否需要水平或垂直移动
        need_horizontal = dest_x != curr_x
        need_vertical = dest_y != curr_y

        if need_horizontal and need_vertical:
            # 需要两个维度的移动，选择拥塞较少的维度
            # 检查水平环和垂直环的拥塞状态
            horizontal_congested = self._is_direction_congested("horizontal")
            vertical_congested = self._is_direction_congested("vertical")

            if horizontal_congested and not vertical_congested:
                # 水平拥塞，优先垂直
                return self._apply_routing_strategy(curr_x, curr_y, dest_x, dest_y, "YX")
            elif vertical_congested and not horizontal_congested:
                # 垂直拥塞，优先水平
                return self._apply_routing_strategy(curr_x, curr_y, dest_x, dest_y, "XY")
            else:
                # 都不拥塞或都拥塞，使用默认XY路由
                return self._apply_routing_strategy(curr_x, curr_y, dest_x, dest_y, "XY")
        elif need_horizontal:
            # 只需要水平移动
            return self._apply_routing_strategy(curr_x, curr_y, dest_x, dest_y, "XY")
        elif need_vertical:
            # 只需要垂直移动
            return self._apply_routing_strategy(curr_x, curr_y, dest_x, dest_y, "YX")
        else:
            return "EQ"  # 本地

    def _is_direction_congested(self, direction: str) -> bool:
        """
        检查指定方向是否拥塞

        Args:
            direction: "horizontal" 或 "vertical"

        Returns:
            是否拥塞
        """
        # 检查对应方向的ETag状态
        for channel in ["req", "rsp", "data"]:
            if self.etag_status[direction][channel]:
                return True
        return False

    def get_inject_direction_status(self) -> Dict[str, Any]:
        """
        获取注入方向队列的状态

        Returns:
            状态信息字典
        """
        status = {}
        for channel in ["req", "rsp", "data"]:
            status[channel] = {}
            for direction in ["TR", "TL", "TU", "TD", "EQ"]:
                fifo = self.inject_direction_fifos[channel][direction]
                status[channel][direction] = {
                    "occupancy": len(fifo),
                    "ready": fifo.ready_signal(),
                    "valid": fifo.valid_signal(),
                }
        return status

    def _compute_inject_to_ring_transfers(self, cycle: int) -> None:
        """计算从inject队列到ring缓冲区的传输可能性"""
        for direction in ["horizontal", "vertical"]:
            for channel in ["req", "rsp", "data"]:
                # 检查是否有inject队列中的flit可以传输
                inject_fifo = self.inject_queues[channel]

                # 检查仲裁状态
                if not self.can_inject_flit(channel, direction):
                    self._transfer_decisions["inject_to_ring"][direction][channel] = None
                    continue

                # 找到最合适的方向代码
                best_dir_code = self._find_best_direction_code(direction, channel)
                if best_dir_code:
                    ring_fifo = self.ring_buffers[direction][channel][best_dir_code]
                    if FlowControlledTransfer.can_transfer(inject_fifo, ring_fifo):
                        self._transfer_decisions["inject_to_ring"][direction][channel] = best_dir_code
                    else:
                        self._transfer_decisions["inject_to_ring"][direction][channel] = None
                else:
                    self._transfer_decisions["inject_to_ring"][direction][channel] = None

    def _compute_ring_to_ring_transfers(self, cycle: int) -> None:
        """计算环形缓冲区之间的传输可能性"""
        for direction in ["horizontal", "vertical"]:
            for channel in ["req", "rsp", "data"]:
                self._transfer_decisions["ring_to_ring"][direction][channel] = {}

                for dir_code in ["TR", "TL", "TD", "TU"]:
                    if self.can_transfer_flit(direction, dir_code, channel):
                        # 计算目标方向和节点
                        target_direction, target_dir_code = self._get_ring_transfer_target(direction, dir_code)
                        if target_direction and target_dir_code:
                            source_fifo = self.ring_buffers[direction][channel][dir_code]
                            # 这里假设目标是相邻节点，实际实现中需要获取相邻节点的FIFO
                            if source_fifo.valid_signal():
                                self._transfer_decisions["ring_to_ring"][direction][channel][dir_code] = (target_direction, target_dir_code)
                            else:
                                self._transfer_decisions["ring_to_ring"][direction][channel][dir_code] = None
                        else:
                            self._transfer_decisions["ring_to_ring"][direction][channel][dir_code] = None
                    else:
                        self._transfer_decisions["ring_to_ring"][direction][channel][dir_code] = None

    def _compute_ring_to_eject_transfers(self, cycle: int) -> None:
        """计算从ring缓冲区到eject队列的传输可能性"""
        for channel in ["req", "rsp", "data"]:
            # 检查是否有到达本节点的flit
            eject_fifo = self.eject_queues[channel]
            found_flit = False

            for direction in ["horizontal", "vertical"]:
                for dir_code in ["TR", "TL", "TD", "TU"]:
                    ring_fifo = self.ring_buffers[direction][channel][dir_code]
                    if ring_fifo.valid_signal():
                        flit = ring_fifo.peek_output()
                        if flit and self._should_eject_flit(flit):
                            if FlowControlledTransfer.can_transfer(ring_fifo, eject_fifo):
                                self._transfer_decisions["ring_to_eject"][channel] = (direction, dir_code)
                                found_flit = True
                                break
                if found_flit:
                    break

            if not found_flit:
                self._transfer_decisions["ring_to_eject"][channel] = None

    def _execute_inject_to_ring_transfers(self, cycle: int) -> None:
        """执行从inject队列到ring缓冲区的传输"""
        for direction in ["horizontal", "vertical"]:
            for channel in ["req", "rsp", "data"]:
                decision = self._transfer_decisions["inject_to_ring"][direction][channel]
                if decision:
                    dir_code = decision
                    inject_fifo = self.inject_queues[channel]
                    ring_fifo = self.ring_buffers[direction][channel][dir_code]

                    if FlowControlledTransfer.try_transfer(inject_fifo, ring_fifo):
                        # 更新传输的flit
                        flit = ring_fifo.peek_output()
                        if flit:
                            flit.network_entry_cycle = cycle
                            self.stats["injected_flits"][channel] += 1

                        # 更新仲裁状态
                        dir_priority_map = {"TR": "ring_tr", "TL": "ring_tl", "TD": "ring_td", "TU": "ring_tu"}
                        self.arbitration_state[f"{direction}_priority"] = dir_priority_map[dir_code]
                        self.arbitration_state["last_arbitration"][direction] = cycle

    def _execute_ring_to_ring_transfers(self, cycle: int) -> None:
        """执行环形缓冲区之间的传输"""
        for direction in ["horizontal", "vertical"]:
            for channel in ["req", "rsp", "data"]:
                decisions = self._transfer_decisions["ring_to_ring"][direction][channel]

                for dir_code, decision in decisions.items():
                    if decision:
                        target_direction, target_dir_code = decision
                        source_fifo = self.ring_buffers[direction][channel][dir_code]

                        # 执行读取操作
                        flit = source_fifo.read_output()
                        if flit:
                            self.stats["transferred_flits"][direction] += 1

                            # 更新仲裁状态
                            priority_map = {"ring_tr": "ring_tl", "ring_tl": "inject", "ring_td": "ring_tu", "ring_tu": "inject"}
                            dir_priority_map = {"TR": "ring_tr", "TL": "ring_tl", "TD": "ring_td", "TU": "ring_tu"}
                            current_priority = dir_priority_map[dir_code]
                            self.arbitration_state[f"{direction}_priority"] = priority_map.get(current_priority, "inject")
                            self.arbitration_state["last_arbitration"][direction] = cycle

    def _execute_ring_to_eject_transfers(self, cycle: int) -> None:
        """执行从ring缓冲区到eject队列的传输"""
        for channel in ["req", "rsp", "data"]:
            decision = self._transfer_decisions["ring_to_eject"][channel]
            if decision:
                direction, dir_code = decision
                ring_fifo = self.ring_buffers[direction][channel][dir_code]
                eject_fifo = self.eject_queues[channel]

                if FlowControlledTransfer.try_transfer(ring_fifo, eject_fifo):
                    # 更新ejected flit的状态
                    flit = eject_fifo.peek_output()
                    if flit:
                        flit.is_arrive = True
                        flit.arrival_network_cycle = cycle
                        self.stats["ejected_flits"][channel] += 1

    def _find_best_direction_code(self, direction: str, channel: str) -> Optional[str]:
        """找到最合适的方向代码进行inject"""
        # 简化实现：选择第一个可用的方向
        for dir_code in ["TR", "TL", "TD", "TU"]:
            ring_fifo = self.ring_buffers[direction][channel][dir_code]
            if ring_fifo.ready_signal():
                return dir_code
        return None

    def _get_ring_transfer_target(self, direction: str, dir_code: str) -> Tuple[Optional[str], Optional[str]]:
        """获取环形传输的目标方向和代码"""
        # 简化实现：返回相同的方向和代码
        # 实际实现需要根据拓扑结构计算
        return direction, dir_code

    def _should_eject_flit(self, flit: CrossRingFlit) -> bool:
        """检查是否应该弹出flit"""
        # 检查flit是否到达目标节点
        if hasattr(flit, "destination") and flit.destination == self.node_id:
            return True
        if hasattr(flit, "dest_node_id") and flit.dest_node_id == self.node_id:
            return True
        return False

    def inject_flit_to_crosspoint(self, flit: CrossRingFlit, direction: str) -> bool:
        """
        将flit注入到指定方向的CrossPoint

        Args:
            flit: 要注入的flit
            direction: 注入方向 ("horizontal", "vertical")

        Returns:
            是否成功注入
        """
        if direction == "horizontal":
            return self.horizontal_crosspoint.try_inject_flit(flit, PriorityLevel.T2)
        elif direction == "vertical":
            return self.vertical_crosspoint.try_inject_flit(flit, PriorityLevel.T2)
        else:
            self.logger.error(f"未知的注入方向: {direction}")
            return False

    def eject_flit_from_crosspoint(self, direction: str, sub_direction: str, target_fifo_occupancy: int, target_fifo_depth: int) -> Optional[CrossRingFlit]:
        """
        从指定方向的CrossPoint下环flit

        Args:
            direction: CrossPoint方向 ("horizontal", "vertical")
            sub_direction: 子方向 ("TR", "TL", "TU", "TD")
            target_fifo_occupancy: 目标FIFO当前占用
            target_fifo_depth: 目标FIFO深度

        Returns:
            下环的flit，如果没有则返回None
        """
        crosspoint = None
        if direction == "horizontal":
            crosspoint = self.horizontal_crosspoint
        elif direction == "vertical":
            crosspoint = self.vertical_crosspoint

        if crosspoint is None:
            return None

        # 查找合适的slot进行下环
        for slot in crosspoint.ring_slots:
            if slot.valid and slot.flit is not None:
                # 检查是否是目标节点
                if self._should_eject_flit(slot.flit):
                    ejected_flit = crosspoint.try_eject_flit(slot, target_fifo_occupancy, target_fifo_depth, sub_direction)
                    if ejected_flit:
                        return ejected_flit

        return None

    def get_crosspoint_status(self) -> Dict[str, Any]:
        """
        获取CrossPoint状态信息

        Returns:
            CrossPoint状态字典
        """
        return {"horizontal": self.horizontal_crosspoint.get_crosspoint_status(), "vertical": self.vertical_crosspoint.get_crosspoint_status()}

    def get_stats(self) -> Dict[str, Any]:
        """
        获取节点统计信息

        Returns:
            统计信息字典
        """
        return {
            "node_id": self.node_id,
            "coordinates": self.coordinates,
            "injected_flits": dict(self.stats["injected_flits"]),
            "ejected_flits": dict(self.stats["ejected_flits"]),
            "transferred_flits": dict(self.stats["transferred_flits"]),
            "congestion_events": self.stats["congestion_events"],
            "buffer_occupancy": {
                "ip_inject_channel_buffers": {ip_id: {k: len(v) for k, v in channels.items()} for ip_id, channels in self.ip_inject_channel_buffers.items()},
                "inject_directions": {k: {d: len(v) for d, v in vv.items()} for k, vv in self.inject_direction_fifos.items()},
                "ip_eject_channel_buffers": {ip_id: {k: len(v) for k, v in channels.items()} for ip_id, channels in self.ip_eject_channel_buffers.items()},
                "eject_input_fifos": {k: {d: len(v) for d, v in vv.items()} for k, vv in self.eject_input_fifos.items()},
                "ring_bridge_input_fifos": {k: {d: len(v) for d, v in vv.items()} for k, vv in self.ring_bridge_input_fifos.items()},
                "ring_bridge_output_fifos": {k: {d: len(v) for d, v in vv.items()} for k, vv in self.ring_bridge_output_fifos.items()},
                "horizontal": {k: {d: len(v) for d, v in vv.items()} for k, vv in self.ring_buffers["horizontal"].items()},
                "vertical": {k: {d: len(v) for d, v in vv.items()} for k, vv in self.ring_buffers["vertical"].items()},
            },
            "congestion_status": {
                "etag": {
                    "horizontal": dict(self.etag_status["horizontal"]),
                    "vertical": dict(self.etag_status["vertical"]),
                },
                "itag": {
                    "horizontal": dict(self.itag_status["horizontal"]),
                    "vertical": dict(self.itag_status["vertical"]),
                },
            },
            "crosspoint_status": self.get_crosspoint_status(),
        }
