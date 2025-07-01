"""
流控机制模块
实现基于带宽的流控和背压机制，包括滑动窗口和拥塞控制
"""

from typing import Dict, Any, Optional, List, Tuple, Callable
from dataclasses import dataclass, field
from enum import Enum
import time
import threading
from collections import deque, defaultdict
import logging
import math

from utils.exceptions import CDMAError


class FlowState(Enum):
    """流控状态枚举"""
    NORMAL = "normal"           # 正常状态
    CONGESTED = "congested"     # 拥塞状态
    THROTTLED = "throttled"     # 限流状态
    BLOCKED = "blocked"         # 阻塞状态


class WindowState(Enum):
    """窗口状态枚举"""
    OPEN = "open"              # 窗口打开
    HALF_OPEN = "half_open"    # 窗口半开
    CLOSED = "closed"          # 窗口关闭


@dataclass
class FlowMetrics:
    """流控指标"""
    # 带宽指标
    current_bandwidth: float = 0.0      # 当前带宽 (MB/s)
    peak_bandwidth: float = 0.0         # 峰值带宽 (MB/s)
    average_bandwidth: float = 0.0      # 平均带宽 (MB/s)
    
    # 延迟指标
    current_latency: float = 0.0        # 当前延迟 (ms)
    average_latency: float = 0.0        # 平均延迟 (ms)
    
    # 丢包和重传指标
    packet_loss_rate: float = 0.0       # 丢包率
    retransmission_rate: float = 0.0    # 重传率
    
    # 缓冲区指标
    buffer_utilization: float = 0.0     # 缓冲区利用率
    queue_depth: int = 0                # 队列深度
    
    # 窗口指标
    window_size: int = 0                # 当前窗口大小
    congestion_window: int = 0          # 拥塞窗口大小
    
    # 时间戳
    last_update_time: float = field(default_factory=time.time)


@dataclass
class BufferEntry:
    """缓冲区条目"""
    data: bytes
    timestamp: float
    source_id: str
    sequence_number: int
    size: int
    priority: int = 0
    
    def age(self) -> float:
        """获取条目年龄（秒）"""
        return time.time() - self.timestamp


class ReceiveBuffer:
    """接收缓冲区"""
    
    def __init__(self, capacity: int = 1024 * 1024):  # 默认1MB
        self._capacity = capacity
        self._buffer: deque = deque()
        self._size = 0
        self._lock = threading.RLock()
        
        # 统计信息
        self._total_bytes_received = 0
        self._total_entries_received = 0
        self._buffer_overflows = 0
        self._expired_entries = 0
        
        # 配置参数
        self._max_entry_age = 30.0  # 最大条目年龄（秒）
        
        self._logger = logging.getLogger("ReceiveBuffer")
    
    def put(self, entry: BufferEntry) -> bool:
        """
        向缓冲区添加条目
        
        Args:
            entry: 缓冲区条目
            
        Returns:
            bool: 添加是否成功
        """
        with self._lock:
            # 检查容量
            if self._size + entry.size > self._capacity:
                # 尝试清理过期条目
                self._cleanup_expired_entries()
                
                # 仍然没有空间
                if self._size + entry.size > self._capacity:
                    self._buffer_overflows += 1
                    self._logger.warning(f"缓冲区溢出: 当前大小={self._size}, 条目大小={entry.size}, 容量={self._capacity}")
                    return False
            
            # 添加条目
            self._buffer.append(entry)
            self._size += entry.size
            self._total_bytes_received += entry.size
            self._total_entries_received += 1
            
            return True
    
    def get(self) -> Optional[BufferEntry]:
        """
        从缓冲区获取条目（FIFO）
        
        Returns:
            BufferEntry: 缓冲区条目，如果为空则返回None
        """
        with self._lock:
            if not self._buffer:
                return None
            
            entry = self._buffer.popleft()
            self._size -= entry.size
            
            return entry
    
    def peek(self) -> Optional[BufferEntry]:
        """
        查看缓冲区第一个条目但不移除
        
        Returns:
            BufferEntry: 缓冲区条目，如果为空则返回None
        """
        with self._lock:
            if not self._buffer:
                return None
            return self._buffer[0]
    
    def _cleanup_expired_entries(self):
        """清理过期条目"""
        current_time = time.time()
        
        # 从头部开始清理过期条目
        while self._buffer:
            entry = self._buffer[0]
            if entry.age() > self._max_entry_age:
                expired_entry = self._buffer.popleft()
                self._size -= expired_entry.size
                self._expired_entries += 1
                self._logger.debug(f"清理过期条目: age={expired_entry.age():.2f}s")
            else:
                break
    
    def available_space(self) -> int:
        """获取可用空间"""
        with self._lock:
            return self._capacity - self._size
    
    def utilization(self) -> float:
        """获取缓冲区利用率"""
        with self._lock:
            return self._size / self._capacity
    
    def get_statistics(self) -> Dict[str, Any]:
        """获取统计信息"""
        with self._lock:
            return {
                'capacity': self._capacity,
                'current_size': self._size,
                'entry_count': len(self._buffer),
                'utilization': self.utilization(),
                'available_space': self.available_space(),
                'total_bytes_received': self._total_bytes_received,
                'total_entries_received': self._total_entries_received,
                'buffer_overflows': self._buffer_overflows,
                'expired_entries': self._expired_entries
            }


class WindowManager:
    """滑动窗口管理器"""
    
    def __init__(self, initial_window_size: int = 64):
        self._window_size = initial_window_size
        self._max_window_size = 1024
        self._min_window_size = 1
        
        # 窗口状态
        self._state = WindowState.OPEN
        self._outstanding_packets = 0
        self._acknowledged_packets = 0
        
        # 拥塞控制
        self._congestion_window = initial_window_size
        self._slow_start_threshold = initial_window_size // 2
        self._in_slow_start = True
        
        # 时间统计
        self._rtt_samples: deque = deque(maxlen=100)  # 保留最近100个RTT样本
        self._estimated_rtt = 0.1  # 初始RTT估计值（秒）
        self._rtt_variance = 0.05  # RTT方差
        
        self._lock = threading.RLock()
        self._logger = logging.getLogger("WindowManager")
    
    def can_send(self) -> bool:
        """检查是否可以发送新包"""
        with self._lock:
            if self._state == WindowState.CLOSED:
                return False
            
            return self._outstanding_packets < min(self._window_size, self._congestion_window)
    
    def packet_sent(self, sequence_number: int, timestamp: float):
        """记录包发送"""
        with self._lock:
            self._outstanding_packets += 1
    
    def packet_acknowledged(self, sequence_number: int, timestamp: float, send_timestamp: float):
        """记录包确认"""
        with self._lock:
            self._outstanding_packets = max(0, self._outstanding_packets - 1)
            self._acknowledged_packets += 1
            
            # 更新RTT
            rtt = timestamp - send_timestamp
            self._update_rtt(rtt)
            
            # 拥塞控制算法（类似TCP的AIMD）
            self._update_congestion_window(True)
    
    def packet_lost(self, sequence_number: int):
        """记录包丢失"""
        with self._lock:
            self._outstanding_packets = max(0, self._outstanding_packets - 1)
            
            # 拥塞控制：减少窗口
            self._update_congestion_window(False)
            
            # 切换到半开状态
            if self._state == WindowState.OPEN:
                self._state = WindowState.HALF_OPEN
    
    def _update_rtt(self, rtt_sample: float):
        """更新RTT估计"""
        self._rtt_samples.append(rtt_sample)
        
        if len(self._rtt_samples) == 1:
            self._estimated_rtt = rtt_sample
            self._rtt_variance = rtt_sample / 2
        else:
            # 使用指数移动平均
            alpha = 0.125
            beta = 0.25
            
            self._rtt_variance = (1 - beta) * self._rtt_variance + beta * abs(rtt_sample - self._estimated_rtt)
            self._estimated_rtt = (1 - alpha) * self._estimated_rtt + alpha * rtt_sample
    
    def _update_congestion_window(self, packet_acked: bool):
        """更新拥塞窗口"""
        if packet_acked:
            if self._in_slow_start:
                # 慢启动阶段：指数增长
                self._congestion_window += 1
                if self._congestion_window >= self._slow_start_threshold:
                    self._in_slow_start = False
            else:
                # 拥塞避免阶段：线性增长
                self._congestion_window += 1.0 / self._congestion_window
        else:
            # 包丢失：快速重传和快速恢复
            self._slow_start_threshold = max(self._congestion_window // 2, self._min_window_size)
            self._congestion_window = self._slow_start_threshold
            self._in_slow_start = False
        
        # 限制窗口大小范围
        self._congestion_window = max(self._min_window_size, 
                                    min(self._max_window_size, int(self._congestion_window)))
    
    def get_timeout(self) -> float:
        """获取超时时间"""
        with self._lock:
            # RTO = EstimatedRTT + 4 * RTTVariance
            return self._estimated_rtt + 4 * self._rtt_variance
    
    def get_window_info(self) -> Dict[str, Any]:
        """获取窗口信息"""
        with self._lock:
            return {
                'window_size': self._window_size,
                'congestion_window': self._congestion_window,
                'outstanding_packets': self._outstanding_packets,
                'acknowledged_packets': self._acknowledged_packets,
                'state': self._state.value,
                'estimated_rtt': self._estimated_rtt,
                'rtt_variance': self._rtt_variance,
                'timeout': self.get_timeout(),
                'in_slow_start': self._in_slow_start,
                'slow_start_threshold': self._slow_start_threshold
            }


class CongestionControl:
    """拥塞控制器"""
    
    def __init__(self):
        self._state = FlowState.NORMAL
        self._lock = threading.RLock()
        
        # 拥塞检测参数
        self._loss_threshold = 0.01     # 丢包率阈值（1%）
        self._latency_threshold = 0.1   # 延迟阈值（100ms）
        self._buffer_threshold = 0.8    # 缓冲区利用率阈值（80%）
        
        # 历史统计
        self._packet_history: deque = deque(maxlen=1000)  # 最近1000个包的记录
        self._latency_history: deque = deque(maxlen=100)  # 最近100个延迟样本
        
        # 控制参数
        self._backoff_factor = 0.5      # 退避因子
        self._recovery_factor = 1.1     # 恢复因子
        
        self._logger = logging.getLogger("CongestionControl")
    
    def update_packet_result(self, success: bool, latency: float = 0.0):
        """更新包传输结果"""
        with self._lock:
            timestamp = time.time()
            self._packet_history.append({'success': success, 'timestamp': timestamp})
            
            if success and latency > 0:
                self._latency_history.append(latency)
            
            # 检查拥塞状态
            self._check_congestion()
    
    def _check_congestion(self):
        """检查拥塞状态"""
        if len(self._packet_history) < 10:  # 需要足够的样本
            return
        
        # 计算丢包率
        recent_packets = list(self._packet_history)[-50:]  # 最近50个包
        loss_rate = 1.0 - sum(1 for p in recent_packets if p['success']) / len(recent_packets)
        
        # 计算平均延迟
        avg_latency = 0.0
        if self._latency_history:
            avg_latency = sum(self._latency_history) / len(self._latency_history)
        
        # 判断拥塞状态
        old_state = self._state
        
        if loss_rate > self._loss_threshold or avg_latency > self._latency_threshold:
            if self._state == FlowState.NORMAL:
                self._state = FlowState.CONGESTED
            elif self._state == FlowState.CONGESTED:
                self._state = FlowState.THROTTLED
        else:
            if self._state in [FlowState.CONGESTED, FlowState.THROTTLED]:
                self._state = FlowState.NORMAL
        
        if old_state != self._state:
            self._logger.info(f"拥塞状态变更: {old_state.value} -> {self._state.value}")
            self._logger.debug(f"丢包率: {loss_rate:.4f}, 平均延迟: {avg_latency:.4f}s")
    
    def get_sending_rate_factor(self) -> float:
        """获取发送速率因子"""
        with self._lock:
            if self._state == FlowState.NORMAL:
                return 1.0
            elif self._state == FlowState.CONGESTED:
                return 0.7  # 降低到70%
            elif self._state == FlowState.THROTTLED:
                return 0.3  # 降低到30%
            else:  # BLOCKED
                return 0.0
    
    def should_backoff(self) -> bool:
        """是否应该退避"""
        with self._lock:
            return self._state in [FlowState.CONGESTED, FlowState.THROTTLED, FlowState.BLOCKED]
    
    def get_congestion_info(self) -> Dict[str, Any]:
        """获取拥塞信息"""
        with self._lock:
            # 计算当前统计
            loss_rate = 0.0
            avg_latency = 0.0
            
            if len(self._packet_history) >= 10:
                recent_packets = list(self._packet_history)[-50:]
                loss_rate = 1.0 - sum(1 for p in recent_packets if p['success']) / len(recent_packets)
            
            if self._latency_history:
                avg_latency = sum(self._latency_history) / len(self._latency_history)
            
            return {
                'state': self._state.value,
                'loss_rate': loss_rate,
                'average_latency': avg_latency,
                'sending_rate_factor': self.get_sending_rate_factor(),
                'should_backoff': self.should_backoff(),
                'packet_samples': len(self._packet_history),
                'latency_samples': len(self._latency_history)
            }


class FlowController:
    """流控控制器主类"""
    
    def __init__(self, chip_id: str, max_bandwidth: float = 1000.0):  # 默认1GB/s
        self._chip_id = chip_id
        self._max_bandwidth = max_bandwidth  # MB/s
        
        # 核心组件
        self._receive_buffer = ReceiveBuffer()
        self._window_manager = WindowManager()
        self._congestion_control = CongestionControl()
        
        # 流控状态
        self._current_bandwidth = 0.0
        self._target_bandwidth = max_bandwidth
        self._flow_state = FlowState.NORMAL
        
        # 统计信息
        self._metrics = FlowMetrics()
        self._bandwidth_samples: deque = deque(maxlen=100)
        
        # 回调函数
        self._congestion_callback: Optional[Callable[[FlowState], None]] = None
        self._buffer_full_callback: Optional[Callable[[], None]] = None
        
        self._lock = threading.RLock()
        self._logger = logging.getLogger(f"FlowController-{chip_id}")
        
        self._logger.info(f"流控器初始化完成: {chip_id}, 最大带宽: {max_bandwidth} MB/s")
    
    def set_congestion_callback(self, callback: Callable[[FlowState], None]):
        """设置拥塞状态变化回调"""
        self._congestion_callback = callback
    
    def set_buffer_full_callback(self, callback: Callable[[], None]):
        """设置缓冲区满回调"""
        self._buffer_full_callback = callback
    
    def can_send_packet(self, packet_size: int) -> bool:
        """
        检查是否可以发送包
        
        Args:
            packet_size: 包大小（字节）
            
        Returns:
            bool: 是否可以发送
        """
        with self._lock:
            # 检查窗口是否允许
            if not self._window_manager.can_send():
                return False
            
            # 检查拥塞控制
            if self._congestion_control.should_backoff():
                return False
            
            # 检查带宽限制
            rate_factor = self._congestion_control.get_sending_rate_factor()
            allowed_bandwidth = self._target_bandwidth * rate_factor
            
            if self._current_bandwidth > allowed_bandwidth:
                return False
            
            return True
    
    def packet_sent(self, sequence_number: int, packet_size: int) -> bool:
        """
        记录包发送
        
        Args:
            sequence_number: 序列号
            packet_size: 包大小
            
        Returns:
            bool: 记录是否成功
        """
        with self._lock:
            timestamp = time.time()
            
            # 更新窗口管理器
            self._window_manager.packet_sent(sequence_number, timestamp)
            
            # 更新带宽统计
            self._update_bandwidth_stats(packet_size)
            
            return True
    
    def packet_acknowledged(self, sequence_number: int, send_timestamp: float, 
                          receive_timestamp: float) -> bool:
        """
        记录包确认
        
        Args:
            sequence_number: 序列号
            send_timestamp: 发送时间戳
            receive_timestamp: 接收时间戳
            
        Returns:
            bool: 记录是否成功
        """
        with self._lock:
            # 计算延迟
            latency = receive_timestamp - send_timestamp
            
            # 更新窗口管理器
            self._window_manager.packet_acknowledged(sequence_number, receive_timestamp, send_timestamp)
            
            # 更新拥塞控制
            self._congestion_control.update_packet_result(True, latency)
            
            # 更新指标
            self._metrics.current_latency = latency
            self._update_latency_stats(latency)
            
            return True
    
    def packet_lost(self, sequence_number: int) -> bool:
        """
        记录包丢失
        
        Args:
            sequence_number: 序列号
            
        Returns:
            bool: 记录是否成功
        """
        with self._lock:
            # 更新窗口管理器
            self._window_manager.packet_lost(sequence_number)
            
            # 更新拥塞控制
            self._congestion_control.update_packet_result(False)
            
            return True
    
    def receive_packet(self, data: bytes, source_id: str, sequence_number: int) -> bool:
        """
        接收包到缓冲区
        
        Args:
            data: 包数据
            source_id: 源ID
            sequence_number: 序列号
            
        Returns:
            bool: 接收是否成功
        """
        with self._lock:
            entry = BufferEntry(
                data=data,
                timestamp=time.time(),
                source_id=source_id,
                sequence_number=sequence_number,
                size=len(data)
            )
            
            success = self._receive_buffer.put(entry)
            
            if not success and self._buffer_full_callback:
                self._buffer_full_callback()
            
            # 更新缓冲区指标
            self._metrics.buffer_utilization = self._receive_buffer.utilization()
            self._metrics.queue_depth = len(self._receive_buffer._buffer)
            
            return success
    
    def get_received_packet(self) -> Optional[Tuple[bytes, str, int]]:
        """
        获取接收到的包
        
        Returns:
            Tuple[bytes, str, int]: (数据, 源ID, 序列号)，如果没有则返回None
        """
        with self._lock:
            entry = self._receive_buffer.get()
            if entry:
                return (entry.data, entry.source_id, entry.sequence_number)
            return None
    
    def _update_bandwidth_stats(self, packet_size: int):
        """更新带宽统计"""
        current_time = time.time()
        
        # 添加带宽样本
        self._bandwidth_samples.append({
            'size': packet_size,
            'timestamp': current_time
        })
        
        # 计算当前带宽（基于最近1秒的数据）
        recent_samples = [s for s in self._bandwidth_samples 
                         if current_time - s['timestamp'] <= 1.0]
        
        if recent_samples:
            total_bytes = sum(s['size'] for s in recent_samples)
            self._current_bandwidth = total_bytes / (1024 * 1024)  # MB/s
            
            # 更新指标
            self._metrics.current_bandwidth = self._current_bandwidth
            self._metrics.peak_bandwidth = max(self._metrics.peak_bandwidth, self._current_bandwidth)
            
            # 计算平均带宽
            if len(self._bandwidth_samples) > 1:
                total_samples_bytes = sum(s['size'] for s in self._bandwidth_samples)
                time_span = self._bandwidth_samples[-1]['timestamp'] - self._bandwidth_samples[0]['timestamp']
                if time_span > 0:
                    self._metrics.average_bandwidth = total_samples_bytes / (time_span * 1024 * 1024)
    
    def _update_latency_stats(self, latency: float):
        """更新延迟统计"""
        # 简单的移动平均
        if self._metrics.average_latency == 0:
            self._metrics.average_latency = latency
        else:
            alpha = 0.1  # 平滑因子
            self._metrics.average_latency = (1 - alpha) * self._metrics.average_latency + alpha * latency
    
    def get_flow_metrics(self) -> FlowMetrics:
        """获取流控指标"""
        with self._lock:
            # 更新窗口指标
            window_info = self._window_manager.get_window_info()
            self._metrics.window_size = window_info['window_size']
            self._metrics.congestion_window = window_info['congestion_window']
            
            # 更新时间戳
            self._metrics.last_update_time = time.time()
            
            return self._metrics
    
    def get_comprehensive_status(self) -> Dict[str, Any]:
        """获取综合状态信息"""
        with self._lock:
            return {
                'chip_id': self._chip_id,
                'flow_state': self._flow_state.value,
                'current_bandwidth': self._current_bandwidth,
                'target_bandwidth': self._target_bandwidth,
                'max_bandwidth': self._max_bandwidth,
                'metrics': self._metrics,
                'buffer_stats': self._receive_buffer.get_statistics(),
                'window_info': self._window_manager.get_window_info(),
                'congestion_info': self._congestion_control.get_congestion_info()
            }
    
    def adjust_target_bandwidth(self, new_bandwidth: float):
        """调整目标带宽"""
        with self._lock:
            self._target_bandwidth = min(new_bandwidth, self._max_bandwidth)
            self._logger.info(f"调整目标带宽: {self._target_bandwidth} MB/s")
    
    def reset_statistics(self):
        """重置统计信息"""
        with self._lock:
            self._metrics = FlowMetrics()
            self._bandwidth_samples.clear()
            self._logger.info("流控统计信息已重置")
    
    def shutdown(self):
        """关闭流控器"""
        with self._lock:
            self._logger.info(f"关闭流控器: {self._chip_id}")
            # 清理资源
            self._bandwidth_samples.clear()
    
    @property
    def chip_id(self) -> str:
        return self._chip_id