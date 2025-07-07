"""
仿真事件系统模块
定义了仿真引擎中使用的各种事件类型
"""

from dataclasses import dataclass
from typing import Optional, Any, Dict
from enum import Enum

# 导入现有的CDMA消息类型 - 需要先检查实际的消息结构
from src.c2c.protocol.cdma_system import CDMAPacket


class EventType(Enum):
    """事件类型枚举"""

    CDMA_SEND = "cdma_send"  # CDMA发送事件
    CDMA_RECEIVE = "cdma_receive"  # CDMA接收事件
    LINK_TRANSFER = "link_transfer"  # 链路传输事件
    CHIP_PROCESS = "chip_process"  # 芯片处理事件
    SIMULATION_END = "sim_end"  # 仿真结束事件


@dataclass
class SimulationEvent:
    """
    仿真事件类，扩展现有的CDMA消息系统
    用于事件驱动的仿真引擎
    """

    timestamp_ns: int  # 事件时间戳（纳秒）
    event_type: EventType  # 事件类型
    source_chip_id: str  # 源芯片ID
    target_chip_id: str  # 目标芯片ID
    event_id: str  # 事件唯一标识符

    # 可选字段
    cdma_packet: Optional[CDMAPacket] = None  # CDMA数据包
    data_size: int = 0  # 数据大小（字节）
    priority: int = 0  # 事件优先级
    metadata: Optional[Dict[str, Any]] = None  # 附加元数据

    def __post_init__(self):
        """初始化后处理"""
        if self.metadata is None:
            self.metadata = {}

        # 生成默认事件ID如果未提供
        if not self.event_id:
            self.event_id = f"{self.event_type.value}_{self.source_chip_id}_{self.target_chip_id}_{self.timestamp_ns}"

    def __lt__(self, other):
        """支持优先队列排序 - 按时间戳排序"""
        if not isinstance(other, SimulationEvent):
            return NotImplemented
        return self.timestamp_ns < other.timestamp_ns

    def __str__(self):
        """字符串表示"""
        return f"事件[{self.event_type.value}] " f"时间:{self.timestamp_ns}ns " f"从:{self.source_chip_id} " f"到:{self.target_chip_id}"


class EventFactory:
    """事件工厂类，用于创建标准的仿真事件"""

    @staticmethod
    def create_cdma_send_event(timestamp_ns: int, source_chip_id: str, target_chip_id: str, data_size: int = 1024, priority: int = 0) -> SimulationEvent:
        """创建CDMA发送事件"""
        return SimulationEvent(
            timestamp_ns=timestamp_ns,
            event_type=EventType.CDMA_SEND,
            source_chip_id=source_chip_id,
            target_chip_id=target_chip_id,
            event_id="",  # 将由__post_init__生成
            data_size=data_size,
            priority=priority,
        )

    @staticmethod
    def create_cdma_receive_event(timestamp_ns: int, source_chip_id: str, target_chip_id: str, cdma_packet: Optional[CDMAPacket] = None) -> SimulationEvent:
        """创建CDMA接收事件"""
        return SimulationEvent(timestamp_ns=timestamp_ns, event_type=EventType.CDMA_RECEIVE, source_chip_id=source_chip_id, target_chip_id=target_chip_id, event_id="", cdma_packet=cdma_packet)

    @staticmethod
    def create_link_transfer_event(timestamp_ns: int, source_chip_id: str, target_chip_id: str, data_size: int) -> SimulationEvent:
        """创建链路传输事件"""
        return SimulationEvent(timestamp_ns=timestamp_ns, event_type=EventType.LINK_TRANSFER, source_chip_id=source_chip_id, target_chip_id=target_chip_id, event_id="", data_size=data_size)
