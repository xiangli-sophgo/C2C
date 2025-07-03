"""
CrossRing队列组件实现。

包含Inject Queue和Eject Queue的核心组件：
- FreqInc/FreqDec：频率转换模块
- Channel FIFO：三通道分离FIFO
- DestSel：方向选择模块
- 仲裁器：Round-Robin仲裁逻辑
"""

from __future__ import annotations
from typing import Dict, List, Optional, Tuple, Any, Deque
from collections import deque, defaultdict
from dataclasses import dataclass, field
import threading
import time
import logging

from .flit import CrossRingFlit
from .config import CrossRingConfig


@dataclass
class FreqConversionState:
    """频率转换状态"""
    last_conversion_cycle: int = 0
    pending_flits: Deque[CrossRingFlit] = field(default_factory=deque)
    conversion_ratio: float = 2.0  # 2GHz / 1GHz
    

class FreqIncModule:
    """
    频率增加模块：1GHz → 2GHz
    
    负责将来自IP的1GHz域Flit转换到网络2GHz域
    """
    
    def __init__(self, name: str, fifo_depth: int = 16):
        self.name = name
        self.fifo_depth = fifo_depth
        
        # 输入输出FIFO
        self.input_fifo: Deque[CrossRingFlit] = deque()
        self.output_fifo: Deque[CrossRingFlit] = deque()
        
        # 频率转换状态
        self.conversion_state = FreqConversionState()
        
        # 时钟域状态
        self.ip_clock_cycle = 0      # 1GHz域时钟
        self.network_clock_cycle = 0  # 2GHz域时钟
        
        # 统计信息
        self.stats = {
            "flits_received": 0,
            "flits_converted": 0,
            "fifo_overflows": 0,
            "conversion_latency_sum": 0,
        }
        
        self.logger = logging.getLogger(f"FreqInc.{name}")
    
    def receive_from_ip(self, flit: CrossRingFlit, cycle: int) -> bool:
        """
        从IP接收Flit（1GHz域）
        
        Args:
            flit: 接收的Flit
            cycle: 当前IP时钟周期
            
        Returns:
            是否成功接收
        """
        if len(self.input_fifo) >= self.fifo_depth:
            self.stats["fifo_overflows"] += 1
            self.logger.warning(f"Input FIFO overflow at cycle {cycle}")
            return False
        
        # 记录接收时间戳
        flit.cmd_entry_noc_from_cake0_cycle = cycle
        flit.departure_inject_cycle = cycle
        
        self.input_fifo.append(flit)
        self.stats["flits_received"] += 1
        self.ip_clock_cycle = cycle
        
        self.logger.debug(f"Received flit {flit.packet_id} from IP at cycle {cycle}")
        return True
    
    def process_conversion(self, network_cycle: int) -> Optional[CrossRingFlit]:
        """
        处理频率转换（2GHz域）
        
        Args:
            network_cycle: 当前网络时钟周期
            
        Returns:
            转换完成的Flit（如果有）
        """
        self.network_clock_cycle = network_cycle
        
        # 检查是否有待转换的Flit
        if not self.input_fifo:
            return None
        
        # 检查输出FIFO是否有空间
        if len(self.output_fifo) >= self.fifo_depth:
            return None
        
        # 取出一个Flit进行转换
        flit = self.input_fifo.popleft()
        
        # 更新Flit的网络入口时间
        flit.cmd_entry_noc_from_cake0_cycle = network_cycle
        flit.departure_network_cycle = network_cycle
        
        # 计算转换延迟
        if flit.departure_inject_cycle > 0:
            conversion_latency = network_cycle - flit.departure_inject_cycle
            self.stats["conversion_latency_sum"] += conversion_latency
        
        self.output_fifo.append(flit)
        self.stats["flits_converted"] += 1
        
        self.logger.debug(f"Converted flit {flit.packet_id} to network domain at cycle {network_cycle}")
        return flit
    
    def get_output_flit(self) -> Optional[CrossRingFlit]:
        """获取转换完成的Flit"""
        if self.output_fifo:
            return self.output_fifo.popleft()
        return None
    
    def is_input_full(self) -> bool:
        """检查输入FIFO是否满"""
        return len(self.input_fifo) >= self.fifo_depth
    
    def is_output_empty(self) -> bool:
        """检查输出FIFO是否空"""
        return len(self.output_fifo) == 0
    
    def get_status(self) -> Dict[str, Any]:
        """获取模块状态"""
        return {
            "name": self.name,
            "input_fifo_occupancy": len(self.input_fifo),
            "output_fifo_occupancy": len(self.output_fifo),
            "fifo_depth": self.fifo_depth,
            "ip_clock_cycle": self.ip_clock_cycle,
            "network_clock_cycle": self.network_clock_cycle,
            "stats": self.stats.copy(),
            "average_conversion_latency": (
                self.stats["conversion_latency_sum"] / max(1, self.stats["flits_converted"])
            )
        }


class FreqDecModule:
    """
    频率减少模块：2GHz → 1GHz
    
    负责将网络2GHz域的Flit转换到IP 1GHz域
    """
    
    def __init__(self, name: str, fifo_depth: int = 16):
        self.name = name
        self.fifo_depth = fifo_depth
        
        # 输入输出FIFO
        self.input_fifo: Deque[CrossRingFlit] = deque()
        self.output_fifo: Deque[CrossRingFlit] = deque()
        
        # 频率转换状态
        self.conversion_state = FreqConversionState()
        self.conversion_state.conversion_ratio = 0.5  # 1GHz / 2GHz
        
        # 时钟域状态
        self.network_clock_cycle = 0  # 2GHz域时钟
        self.ip_clock_cycle = 0      # 1GHz域时钟
        
        # 输出控制
        self.last_output_cycle = -1
        self.output_interval = 2  # 每两个网络周期输出一次
        
        # 统计信息
        self.stats = {
            "flits_received": 0,
            "flits_converted": 0,
            "flits_output": 0,
            "fifo_overflows": 0,
            "conversion_latency_sum": 0,
        }
        
        self.logger = logging.getLogger(f"FreqDec.{name}")
    
    def receive_from_network(self, flit: CrossRingFlit, cycle: int) -> bool:
        """
        从网络接收Flit（2GHz域）
        
        Args:
            flit: 接收的Flit
            cycle: 当前网络时钟周期
            
        Returns:
            是否成功接收
        """
        if len(self.input_fifo) >= self.fifo_depth:
            self.stats["fifo_overflows"] += 1
            self.logger.warning(f"Input FIFO overflow at cycle {cycle}")
            return False
        
        # 记录网络退出时间戳
        flit.arrival_network_cycle = cycle
        
        self.input_fifo.append(flit)
        self.stats["flits_received"] += 1
        self.network_clock_cycle = cycle
        
        self.logger.debug(f"Received flit {flit.packet_id} from network at cycle {cycle}")
        return True
    
    def process_conversion(self, network_cycle: int) -> Optional[CrossRingFlit]:
        """
        处理频率转换（2GHz域触发）
        
        Args:
            network_cycle: 当前网络时钟周期
            
        Returns:
            转换完成的Flit（如果有）
        """
        self.network_clock_cycle = network_cycle
        
        # 检查是否有待转换的Flit
        if not self.input_fifo:
            return None
        
        # 检查输出FIFO是否有空间
        if len(self.output_fifo) >= self.fifo_depth:
            return None
        
        # 按照输出间隔控制转换
        if network_cycle - self.last_output_cycle < self.output_interval:
            return None
        
        # 取出一个Flit进行转换
        flit = self.input_fifo.popleft()
        
        # 更新Flit的转换时间戳
        flit.arrival_eject_cycle = network_cycle
        
        # 计算转换延迟
        if flit.arrival_network_cycle > 0:
            conversion_latency = network_cycle - flit.arrival_network_cycle
            self.stats["conversion_latency_sum"] += conversion_latency
        
        self.output_fifo.append(flit)
        self.stats["flits_converted"] += 1
        self.last_output_cycle = network_cycle
        
        self.logger.debug(f"Converted flit {flit.packet_id} to IP domain at cycle {network_cycle}")
        return flit
    
    def send_to_ip(self, ip_cycle: int) -> Optional[CrossRingFlit]:
        """
        向IP发送Flit（1GHz域）
        
        Args:
            ip_cycle: 当前IP时钟周期
            
        Returns:
            发送的Flit（如果有）
        """
        if not self.output_fifo:
            return None
        
        flit = self.output_fifo.popleft()
        
        # 记录最终的IP接收时间
        flit.arrival_cycle = ip_cycle
        flit.data_received_complete_cycle = ip_cycle
        
        self.stats["flits_output"] += 1
        self.ip_clock_cycle = ip_cycle
        
        self.logger.debug(f"Sent flit {flit.packet_id} to IP at cycle {ip_cycle}")
        return flit
    
    def is_input_full(self) -> bool:
        """检查输入FIFO是否满"""
        return len(self.input_fifo) >= self.fifo_depth
    
    def is_output_empty(self) -> bool:
        """检查输出FIFO是否空"""
        return len(self.output_fifo) == 0
    
    def get_status(self) -> Dict[str, Any]:
        """获取模块状态"""
        return {
            "name": self.name,
            "input_fifo_occupancy": len(self.input_fifo),
            "output_fifo_occupancy": len(self.output_fifo),
            "fifo_depth": self.fifo_depth,
            "network_clock_cycle": self.network_clock_cycle,
            "ip_clock_cycle": self.ip_clock_cycle,
            "output_interval": self.output_interval,
            "last_output_cycle": self.last_output_cycle,
            "stats": self.stats.copy(),
            "average_conversion_latency": (
                self.stats["conversion_latency_sum"] / max(1, self.stats["flits_converted"])
            )
        }


class ChannelFIFO:
    """
    通道FIFO：支持REQ/RSP/DAT三通道分离
    
    每个通道维护独立的FIFO队列，支持独立的流控制
    """
    
    def __init__(self, name: str, channel_depths: Dict[str, int] = None):
        self.name = name
        
        # 默认通道深度配置
        if channel_depths is None:
            channel_depths = {"req": 16, "rsp": 8, "data": 16}
        
        self.channel_depths = channel_depths
        
        # 各通道的FIFO队列
        self.channel_fifos: Dict[str, Deque[CrossRingFlit]] = {
            channel: deque() for channel in channel_depths.keys()
        }
        
        # 通道状态
        self.channel_stats = {
            channel: {
                "flits_received": 0,
                "flits_sent": 0,
                "fifo_overflows": 0,
                "max_occupancy": 0,
            } for channel in channel_depths.keys()
        }
        
        self.logger = logging.getLogger(f"ChannelFIFO.{name}")
    
    def push_flit(self, flit: CrossRingFlit) -> bool:
        """
        向对应通道推送Flit
        
        Args:
            flit: 要推送的Flit
            
        Returns:
            是否成功推送
        """
        channel = flit.channel
        
        if channel not in self.channel_fifos:
            self.logger.error(f"Unknown channel: {channel}")
            return False
        
        fifo = self.channel_fifos[channel]
        max_depth = self.channel_depths[channel]
        
        if len(fifo) >= max_depth:
            self.channel_stats[channel]["fifo_overflows"] += 1
            self.logger.warning(f"Channel {channel} FIFO overflow")
            return False
        
        fifo.append(flit)
        self.channel_stats[channel]["flits_received"] += 1
        
        # 更新最大占用
        current_occupancy = len(fifo)
        if current_occupancy > self.channel_stats[channel]["max_occupancy"]:
            self.channel_stats[channel]["max_occupancy"] = current_occupancy
        
        self.logger.debug(f"Pushed flit {flit.packet_id} to channel {channel}")
        return True
    
    def pop_flit(self, channel: str) -> Optional[CrossRingFlit]:
        """
        从指定通道弹出Flit
        
        Args:
            channel: 通道名称
            
        Returns:
            弹出的Flit（如果有）
        """
        if channel not in self.channel_fifos:
            return None
        
        fifo = self.channel_fifos[channel]
        if not fifo:
            return None
        
        flit = fifo.popleft()
        self.channel_stats[channel]["flits_sent"] += 1
        
        self.logger.debug(f"Popped flit {flit.packet_id} from channel {channel}")
        return flit
    
    def peek_flit(self, channel: str) -> Optional[CrossRingFlit]:
        """
        查看指定通道的队首Flit（不弹出）
        
        Args:
            channel: 通道名称
            
        Returns:
            队首Flit（如果有）
        """
        if channel not in self.channel_fifos:
            return None
        
        fifo = self.channel_fifos[channel]
        if not fifo:
            return None
        
        return fifo[0]
    
    def is_channel_full(self, channel: str) -> bool:
        """检查指定通道是否满"""
        if channel not in self.channel_fifos:
            return True
        
        fifo = self.channel_fifos[channel]
        max_depth = self.channel_depths[channel]
        return len(fifo) >= max_depth
    
    def is_channel_empty(self, channel: str) -> bool:
        """检查指定通道是否空"""
        if channel not in self.channel_fifos:
            return True
        
        fifo = self.channel_fifos[channel]
        return len(fifo) == 0
    
    def get_channel_occupancy(self, channel: str) -> int:
        """获取指定通道的占用数量"""
        if channel not in self.channel_fifos:
            return 0
        
        return len(self.channel_fifos[channel])
    
    def get_non_empty_channels(self) -> List[str]:
        """获取所有非空通道列表"""
        return [
            channel for channel, fifo in self.channel_fifos.items()
            if len(fifo) > 0
        ]
    
    def get_total_occupancy(self) -> int:
        """获取所有通道的总占用数量"""
        return sum(len(fifo) for fifo in self.channel_fifos.values())
    
    def get_status(self) -> Dict[str, Any]:
        """获取FIFO状态"""
        channel_status = {}
        for channel, fifo in self.channel_fifos.items():
            channel_status[channel] = {
                "occupancy": len(fifo),
                "max_depth": self.channel_depths[channel],
                "utilization": len(fifo) / self.channel_depths[channel],
                "is_full": self.is_channel_full(channel),
                "is_empty": self.is_channel_empty(channel),
                "stats": self.channel_stats[channel].copy(),
            }
        
        return {
            "name": self.name,
            "channels": channel_status,
            "total_occupancy": self.get_total_occupancy(),
            "non_empty_channels": self.get_non_empty_channels(),
        }


class RoundRobinArbiter:
    """
    Round-Robin仲裁器
    
    在多个输入源之间进行公平的Round-Robin仲裁
    """
    
    def __init__(self, name: str, sources: List[str]):
        self.name = name
        self.sources = sources
        self.current_index = 0
        
        # 仲裁统计
        self.arbiter_stats = {
            source: {
                "grants": 0,
                "requests": 0,
                "last_grant_cycle": -1,
            } for source in sources
        }
        
        self.total_arbitrations = 0
        self.last_arbitration_cycle = -1
        
        self.logger = logging.getLogger(f"Arbiter.{name}")
    
    def arbitrate(self, requests: Dict[str, bool], cycle: int) -> Optional[str]:
        """
        执行仲裁
        
        Args:
            requests: 各源的请求状态
            cycle: 当前周期
            
        Returns:
            获胜的源名称（如果有）
        """
        # 更新请求统计
        for source, has_request in requests.items():
            if has_request:
                self.arbiter_stats[source]["requests"] += 1
        
        # 寻找有请求的源
        requesting_sources = [source for source, has_request in requests.items() if has_request]
        
        if not requesting_sources:
            return None
        
        # Round-Robin仲裁
        attempts = 0
        start_index = self.current_index
        
        while attempts < len(self.sources):
            current_source = self.sources[self.current_index]
            
            if current_source in requesting_sources:
                # 找到可以授权的源
                self.arbiter_stats[current_source]["grants"] += 1
                self.arbiter_stats[current_source]["last_grant_cycle"] = cycle
                self.total_arbitrations += 1
                self.last_arbitration_cycle = cycle
                
                # 移动到下一个位置
                self.current_index = (self.current_index + 1) % len(self.sources)
                
                self.logger.debug(f"Granted to {current_source} at cycle {cycle}")
                return current_source
            
            # 移动到下一个源
            self.current_index = (self.current_index + 1) % len(self.sources)
            attempts += 1
        
        # 应该不会到达这里，但以防万一
        return None
    
    def get_fairness_metrics(self) -> Dict[str, float]:
        """计算公平性指标"""
        if self.total_arbitrations == 0:
            return {source: 0.0 for source in self.sources}
        
        fairness_metrics = {}
        for source in self.sources:
            grants = self.arbiter_stats[source]["grants"]
            fairness_metrics[source] = grants / self.total_arbitrations
        
        return fairness_metrics
    
    def get_status(self) -> Dict[str, Any]:
        """获取仲裁器状态"""
        return {
            "name": self.name,
            "sources": self.sources,
            "current_index": self.current_index,
            "current_source": self.sources[self.current_index],
            "total_arbitrations": self.total_arbitrations,
            "last_arbitration_cycle": self.last_arbitration_cycle,
            "arbiter_stats": {
                source: stats.copy() for source, stats in self.arbiter_stats.items()
            },
            "fairness_metrics": self.get_fairness_metrics(),
        }