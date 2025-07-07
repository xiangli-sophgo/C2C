"""
FakeChip模型模块
简化的芯片模型，继承现有ChipNode，专用于仿真
"""

from typing import Dict, List, Optional, Any
from src.c2c.topology.node import ChipNode
from src.c2c.protocol.cdma_system import CDMASystem, CDMASystemState
from .events import SimulationEvent, EventType
import logging


class FakeChip(ChipNode):
    """
    简化的芯片模型，继承现有ChipNode，专用于仿真
    提供事件驱动的芯片行为模拟，简化K2K(Kernel-to-Kernel)部分
    """

    def __init__(self, chip_id: str, board_id: str, cdma_engines: int = 10):
        """
        初始化FakeChip

        Args:
            chip_id: 芯片唯一标识符
            board_id: 板卡标识符
            cdma_engines: CDMA引擎数量
        """
        super().__init__(chip_id, board_id, cdma_engines)

        # 初始化CDMA系统
        self.cdma_system = CDMASystem(chip_id)

        # 5个c2c_sys端口，模拟SG2260E的C2C接口
        self.c2c_sys_ports = {f"c2c_sys{i}": None for i in range(5)}

        # 仿真相关状态
        self.current_time_ns = 0
        self.pending_events: List[SimulationEvent] = []
        self.processed_events: List[SimulationEvent] = []

        # 性能统计
        self.sent_messages = 0
        self.received_messages = 0
        self.total_bytes_sent = 0
        self.total_bytes_received = 0

        # 日志记录器
        self.logger = logging.getLogger(f"FakeChip.{chip_id}")

        print(f"初始化FakeChip: {chip_id}, 板卡: {board_id}, CDMA引擎: {cdma_engines}")

    def connect_c2c_port(self, port_index: int, target_chip: "FakeChip"):
        """
        连接C2C端口到目标芯片

        Args:
            port_index: 端口索引 (0-4)
            target_chip: 目标芯片实例
        """
        if 0 <= port_index < 5:
            port_name = f"c2c_sys{port_index}"
            self.c2c_sys_ports[port_name] = target_chip
            print(f"芯片 {self.chip_id} 端口 {port_name} 连接到芯片 {target_chip.chip_id}")
        else:
            raise ValueError(f"无效的端口索引: {port_index}，必须在0-4范围内")

    def process_simulation_event(self, event: SimulationEvent):
        """
        处理仿真事件，简化K2K部分

        Args:
            event: 要处理的仿真事件
        """
        self.current_time_ns = event.timestamp_ns
        self.logger.debug(f"处理事件: {event}")

        try:
            if event.event_type == EventType.CDMA_SEND:
                self._handle_cdma_send_event(event)
            elif event.event_type == EventType.CDMA_RECEIVE:
                self._handle_cdma_receive_event(event)
            elif event.event_type == EventType.LINK_TRANSFER:
                self._handle_link_transfer_event(event)
            elif event.event_type == EventType.CHIP_PROCESS:
                self._handle_chip_process_event(event)
            else:
                print(f"警告: 未知事件类型 {event.event_type}")

            # 记录已处理事件
            self.processed_events.append(event)

        except Exception as e:
            self.logger.error(f"处理事件时发生错误: {e}")
            print(f"错误: 芯片 {self.chip_id} 处理事件失败: {e}")

    def _handle_cdma_send_event(self, event: SimulationEvent):
        """处理CDMA发送事件"""
        print(f"芯片 {self.chip_id} 发送CDMA消息到 {event.target_chip_id}, " f"数据大小: {event.data_size} 字节")

        # 更新统计信息
        self.sent_messages += 1
        self.total_bytes_sent += event.data_size

        # 简化的CDMA发送逻辑 - 在实际实现中会调用CDMA系统
        # 这里模拟发送延迟和处理
        processing_delay_ns = self._calculate_processing_delay(event.data_size)

        # 可以在这里添加更复杂的CDMA协议模拟
        if hasattr(self.cdma_system, "state") and self.cdma_system.state == CDMASystemState.ACTIVE:
            print(f"CDMA系统活跃，处理发送请求")

        print(f"发送处理延迟: {processing_delay_ns} ns")

    def _handle_cdma_receive_event(self, event: SimulationEvent):
        """处理CDMA接收事件"""
        print(f"芯片 {self.chip_id} 接收来自 {event.source_chip_id} 的CDMA消息")

        # 更新统计信息
        self.received_messages += 1
        self.total_bytes_received += event.data_size

        # 简化的接收处理逻辑
        if event.cdma_packet:
            print(f"处理CDMA数据包，大小: {event.data_size} 字节")

        # 模拟接收处理延迟
        processing_delay_ns = self._calculate_processing_delay(event.data_size)
        print(f"接收处理延迟: {processing_delay_ns} ns")

    def _handle_link_transfer_event(self, event: SimulationEvent):
        """处理链路传输事件"""
        print(f"芯片 {self.chip_id} 处理链路传输事件，数据大小: {event.data_size} 字节")

        # 简化的链路处理逻辑
        transfer_delay_ns = self._calculate_transfer_delay(event.data_size)
        print(f"链路传输延迟: {transfer_delay_ns} ns")

    def _handle_chip_process_event(self, event: SimulationEvent):
        """处理芯片内部处理事件"""
        print(f"芯片 {self.chip_id} 执行内部处理任务")

        # 模拟内部处理逻辑
        processing_time_ns = 1000  # 1微秒的基础处理时间
        print(f"内部处理时间: {processing_time_ns} ns")

    def _calculate_processing_delay(self, data_size: int) -> int:
        """
        计算处理延迟（简化模型）

        Args:
            data_size: 数据大小（字节）

        Returns:
            处理延迟（纳秒）
        """
        # 简化的延迟模型：基础延迟 + 数据相关延迟
        base_delay_ns = 100  # 100ns基础延迟
        data_delay_ns = data_size * 0.1  # 每字节0.1ns

        return int(base_delay_ns + data_delay_ns)

    def _calculate_transfer_delay(self, data_size: int) -> int:
        """
        计算传输延迟（简化模型）

        Args:
            data_size: 数据大小（字节）

        Returns:
            传输延迟（纳秒）
        """
        # 假设传输速度为 1GB/s = 1ns/byte
        return data_size

    def get_statistics(self) -> Dict[str, Any]:
        """
        获取芯片统计信息

        Returns:
            包含各种统计数据的字典
        """
        return {
            "chip_id": self.chip_id,
            "board_id": self.board_id,
            "current_time_ns": self.current_time_ns,
            "sent_messages": self.sent_messages,
            "received_messages": self.received_messages,
            "total_bytes_sent": self.total_bytes_sent,
            "total_bytes_received": self.total_bytes_received,
            "processed_events": len(self.processed_events),
            "cdma_engines": self.cdma_engines,
            "c2c_ports_connected": sum(1 for port in self.c2c_sys_ports.values() if port is not None),
        }

    def reset_statistics(self):
        """重置统计信息"""
        self.sent_messages = 0
        self.received_messages = 0
        self.total_bytes_sent = 0
        self.total_bytes_received = 0
        self.processed_events.clear()
        print(f"芯片 {self.chip_id} 统计信息已重置")

    def __str__(self):
        """字符串表示"""
        stats = self.get_statistics()
        return f"FakeChip({self.chip_id}) - " f"发送: {stats['sent_messages']} 消息, " f"接收: {stats['received_messages']} 消息, " f"时间: {stats['current_time_ns']} ns"
