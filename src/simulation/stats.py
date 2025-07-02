"""
仿真统计模块
收集和分析仿真过程中的各种统计数据
"""

from typing import Dict, List, Any, Optional
from dataclasses import dataclass, field
from .events import SimulationEvent, EventType
import json


@dataclass
class ChipStatistics:
    """单个芯片的统计信息"""
    chip_id: str
    sent_messages: int = 0
    received_messages: int = 0
    total_bytes_sent: int = 0
    total_bytes_received: int = 0
    processing_time_ns: int = 0
    idle_time_ns: int = 0
    
    def get_throughput_mbps(self, simulation_time_ns: int) -> float:
        """计算吞吐量（Mbps）"""
        if simulation_time_ns <= 0:
            return 0.0
        
        total_bytes = self.total_bytes_sent + self.total_bytes_received
        simulation_time_s = simulation_time_ns / 1e9
        return (total_bytes * 8) / (simulation_time_s * 1e6)
    
    def get_utilization(self, simulation_time_ns: int) -> float:
        """计算芯片利用率"""
        if simulation_time_ns <= 0:
            return 0.0
        
        return (self.processing_time_ns / simulation_time_ns) * 100


@dataclass
class LinkStatistics:
    """链路统计信息"""
    link_id: str
    total_transfers: int = 0
    total_bytes_transferred: int = 0
    average_latency_ns: float = 0.0
    peak_bandwidth_mbps: float = 0.0
    utilization_percent: float = 0.0


@dataclass 
class TrafficPattern:
    """流量模式统计"""
    source_chip: str
    target_chip: str
    message_count: int = 0
    total_bytes: int = 0
    average_message_size: float = 0.0
    peak_rate_mbps: float = 0.0


class SimulationStats:
    """
    仿真统计收集器
    收集、存储和分析仿真过程中的各种性能指标
    """
    
    def __init__(self):
        """初始化统计收集器"""
        # 基本统计信息
        self.total_simulation_time_ns: int = 0
        self.total_events_processed: int = 0
        self.start_time_ns: int = 0
        self.end_time_ns: int = 0
        
        # 芯片级统计
        self.chip_stats: Dict[str, ChipStatistics] = {}
        
        # 链路级统计  
        self.link_stats: Dict[str, LinkStatistics] = {}
        
        # 流量模式统计
        self.traffic_patterns: Dict[str, TrafficPattern] = {}
        
        # 事件类型统计
        self.event_type_counts: Dict[str, int] = {
            event_type.value: 0 for event_type in EventType
        }
        
        # 性能指标
        self.total_bytes_transferred: int = 0
        self.total_messages_sent: int = 0
        self.average_latency_ns: float = 0.0
        self.peak_throughput_mbps: float = 0.0
        
        # 时间序列数据（用于生成图表）
        self.throughput_timeline: List[tuple] = []  # (时间, 吞吐量)
        self.latency_timeline: List[tuple] = []     # (时间, 延迟)
        
        print("初始化仿真统计收集器")
    
    def reset(self):
        """重置所有统计信息"""
        self.total_simulation_time_ns = 0
        self.total_events_processed = 0
        self.start_time_ns = 0
        self.end_time_ns = 0
        
        self.chip_stats.clear()
        self.link_stats.clear()
        self.traffic_patterns.clear()
        
        for event_type in EventType:
            self.event_type_counts[event_type.value] = 0
        
        self.total_bytes_transferred = 0
        self.total_messages_sent = 0
        self.average_latency_ns = 0.0
        self.peak_throughput_mbps = 0.0
        
        self.throughput_timeline.clear()
        self.latency_timeline.clear()
        
        print("统计信息已重置")
    
    def update_from_event(self, event: SimulationEvent):
        """
        从事件更新统计信息
        
        Args:
            event: 仿真事件
        """
        # 更新事件类型计数
        self.event_type_counts[event.event_type.value] += 1
        
        # 更新时间范围
        if self.start_time_ns == 0 or event.timestamp_ns < self.start_time_ns:
            self.start_time_ns = event.timestamp_ns
        if event.timestamp_ns > self.end_time_ns:
            self.end_time_ns = event.timestamp_ns
        
        # 根据事件类型更新具体统计
        if event.event_type == EventType.CDMA_SEND:
            self._update_send_stats(event)
        elif event.event_type == EventType.CDMA_RECEIVE:
            self._update_receive_stats(event)
        elif event.event_type == EventType.LINK_TRANSFER:
            self._update_link_stats(event)
    
    def _update_send_stats(self, event: SimulationEvent):
        """更新发送统计"""
        source_id = event.source_chip_id
        
        # 更新芯片统计
        if source_id not in self.chip_stats:
            self.chip_stats[source_id] = ChipStatistics(source_id)
        
        self.chip_stats[source_id].sent_messages += 1
        self.chip_stats[source_id].total_bytes_sent += event.data_size
        
        # 更新流量模式
        pattern_key = f"{event.source_chip_id}→{event.target_chip_id}"
        if pattern_key not in self.traffic_patterns:
            self.traffic_patterns[pattern_key] = TrafficPattern(
                event.source_chip_id, 
                event.target_chip_id
            )
        
        pattern = self.traffic_patterns[pattern_key]
        pattern.message_count += 1
        pattern.total_bytes += event.data_size
        pattern.average_message_size = pattern.total_bytes / pattern.message_count
        
        # 更新全局统计
        self.total_messages_sent += 1
        self.total_bytes_transferred += event.data_size
    
    def _update_receive_stats(self, event: SimulationEvent):
        """更新接收统计"""
        target_id = event.target_chip_id
        
        # 更新芯片统计
        if target_id not in self.chip_stats:
            self.chip_stats[target_id] = ChipStatistics(target_id)
        
        self.chip_stats[target_id].received_messages += 1
        self.chip_stats[target_id].total_bytes_received += event.data_size
    
    def _update_link_stats(self, event: SimulationEvent):
        """更新链路统计"""
        link_key = f"{event.source_chip_id}↔{event.target_chip_id}"
        
        if link_key not in self.link_stats:
            self.link_stats[link_key] = LinkStatistics(link_key)
        
        link_stat = self.link_stats[link_key]
        link_stat.total_transfers += 1
        link_stat.total_bytes_transferred += event.data_size
    
    def add_chip_stats(self, chip_id: str, chip_data: Dict[str, Any]):
        """
        添加芯片统计数据
        
        Args:
            chip_id: 芯片ID
            chip_data: 芯片统计数据字典
        """
        if chip_id not in self.chip_stats:
            self.chip_stats[chip_id] = ChipStatistics(chip_id)
        
        stats = self.chip_stats[chip_id]
        stats.sent_messages = chip_data.get('sent_messages', 0)
        stats.received_messages = chip_data.get('received_messages', 0)
        stats.total_bytes_sent = chip_data.get('total_bytes_sent', 0)
        stats.total_bytes_received = chip_data.get('total_bytes_received', 0)
    
    def calculate_overall_metrics(self):
        """计算整体性能指标"""
        if self.total_simulation_time_ns <= 0:
            return
        
        # 计算平均延迟（简化计算）
        if self.total_messages_sent > 0:
            self.average_latency_ns = self.total_simulation_time_ns / self.total_messages_sent
        
        # 计算峰值吞吐量
        simulation_time_s = self.total_simulation_time_ns / 1e9
        if simulation_time_s > 0:
            self.peak_throughput_mbps = (self.total_bytes_transferred * 8) / (simulation_time_s * 1e6)
    
    def get_summary(self) -> Dict[str, Any]:
        """
        获取统计摘要
        
        Returns:
            包含各种统计指标的字典
        """
        self.calculate_overall_metrics()
        
        return {
            "基本信息": {
                "仿真时间_ns": self.total_simulation_time_ns,
                "仿真时间_ms": self.total_simulation_time_ns / 1e6,
                "处理事件数": self.total_events_processed,
                "开始时间_ns": self.start_time_ns,
                "结束时间_ns": self.end_time_ns
            },
            "性能指标": {
                "总字节传输": self.total_bytes_transferred,
                "总消息数": self.total_messages_sent,
                "平均延迟_ns": self.average_latency_ns,
                "峰值吞吐量_Mbps": self.peak_throughput_mbps
            },
            "事件类型统计": self.event_type_counts,
            "芯片数量": len(self.chip_stats),
            "流量模式数量": len(self.traffic_patterns)
        }
    
    def get_chip_summary(self) -> Dict[str, Dict[str, Any]]:
        """获取所有芯片的统计摘要"""
        summary = {}
        
        for chip_id, stats in self.chip_stats.items():
            summary[chip_id] = {
                "发送消息": stats.sent_messages,
                "接收消息": stats.received_messages,
                "发送字节": stats.total_bytes_sent,
                "接收字节": stats.total_bytes_received,
                "吞吐量_Mbps": stats.get_throughput_mbps(self.total_simulation_time_ns),
                "利用率_%": stats.get_utilization(self.total_simulation_time_ns)
            }
        
        return summary
    
    def get_traffic_patterns(self) -> Dict[str, Dict[str, Any]]:
        """获取流量模式统计"""
        patterns = {}
        
        for pattern_key, pattern in self.traffic_patterns.items():
            patterns[pattern_key] = {
                "消息数": pattern.message_count,
                "总字节": pattern.total_bytes,
                "平均消息大小": pattern.average_message_size,
                "峰值速率_Mbps": pattern.peak_rate_mbps
            }
        
        return patterns
    
    def print_summary(self):
        """打印统计摘要"""
        print("\n" + "="*60)
        print("仿真统计摘要")
        print("="*60)
        
        summary = self.get_summary()
        
        # 基本信息
        basic_info = summary["基本信息"]
        print(f"仿真时间: {basic_info['仿真时间_ms']:.2f} ms")
        print(f"处理事件: {basic_info['处理事件数']}")
        
        # 性能指标
        perf_info = summary["性能指标"]
        print(f"总传输量: {perf_info['总字节传输']} 字节")
        print(f"总消息数: {perf_info['总消息数']}")
        print(f"平均延迟: {perf_info['平均延迟_ns']:.2f} ns")
        print(f"峰值吞吐量: {perf_info['峰值吞吐量_Mbps']:.2f} Mbps")
        
        # 芯片统计
        print(f"\n芯片统计 ({len(self.chip_stats)} 个芯片):")
        chip_summary = self.get_chip_summary()
        for chip_id, stats in chip_summary.items():
            print(f"  {chip_id}: "
                  f"发送 {stats['发送消息']}, "
                  f"接收 {stats['接收消息']}, "
                  f"吞吐量 {stats['吞吐量_Mbps']:.2f} Mbps")
        
        # 流量模式
        if self.traffic_patterns:
            print(f"\n流量模式 ({len(self.traffic_patterns)} 个模式):")
            traffic_summary = self.get_traffic_patterns()
            for pattern_key, stats in traffic_summary.items():
                print(f"  {pattern_key}: "
                      f"{stats['消息数']} 消息, "
                      f"{stats['总字节']} 字节, "
                      f"平均大小 {stats['平均消息大小']:.1f} 字节")
        
        print("="*60)
    
    def export_to_json(self, filename: str):
        """
        导出统计数据到JSON文件
        
        Args:
            filename: 输出文件名
        """
        export_data = {
            "summary": self.get_summary(),
            "chip_stats": self.get_chip_summary(),
            "traffic_patterns": self.get_traffic_patterns(),
            "event_timeline": self.event_type_counts
        }
        
        try:
            with open(filename, 'w', encoding='utf-8') as f:
                json.dump(export_data, f, ensure_ascii=False, indent=2)
            print(f"统计数据已导出到: {filename}")
        except Exception as e:
            print(f"导出统计数据失败: {e}")
    
    def get_performance_metrics(self) -> Dict[str, float]:
        """获取关键性能指标"""
        self.calculate_overall_metrics()
        
        return {
            "总吞吐量_Mbps": self.peak_throughput_mbps,
            "平均延迟_μs": self.average_latency_ns / 1000,
            "消息传输率_msg/s": self.total_messages_sent / (self.total_simulation_time_ns / 1e9) if self.total_simulation_time_ns > 0 else 0,
            "平均消息大小_bytes": self.total_bytes_transferred / self.total_messages_sent if self.total_messages_sent > 0 else 0
        }