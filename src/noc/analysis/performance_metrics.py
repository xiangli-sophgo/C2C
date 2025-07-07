"""
性能指标数据结构定义
定义各种性能指标的数据结构和计算方法
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Union
from enum import Enum
import numpy as np


class RequestType(Enum):
    """请求类型"""
    READ = "read"
    WRITE = "write"
    ALL = "all"


@dataclass
class RequestMetrics:
    """单个请求的性能指标"""
    packet_id: str
    request_type: RequestType
    source_node: int
    dest_node: int
    burst_size: int
    
    # 时间戳 (单位: ns)
    start_time: int
    end_time: int
    cmd_latency: int
    data_latency: int
    network_latency: int
    
    # 数据量 (单位: bytes)
    total_bytes: int
    
    # 路径信息
    hop_count: int = 0
    path_nodes: List[int] = field(default_factory=list)
    
    @property
    def total_latency(self) -> int:
        """总延迟"""
        return self.end_time - self.start_time
    
    @property
    def bandwidth_bytes_per_ns(self) -> float:
        """带宽 (bytes/ns)"""
        if self.total_latency > 0:
            return self.total_bytes / self.total_latency
        return 0.0
    
    @property
    def bandwidth_gbps(self) -> float:
        """带宽 (GB/s)"""
        return self.bandwidth_bytes_per_ns
    
    @property
    def throughput_requests_per_ns(self) -> float:
        """吞吐量 (requests/ns)"""
        if self.total_latency > 0:
            return 1.0 / self.total_latency
        return 0.0


@dataclass
class WorkingInterval:
    """工作区间指标"""
    start_time: int
    end_time: int
    request_count: int
    total_bytes: int
    
    @property
    def duration(self) -> int:
        """区间持续时间"""
        return self.end_time - self.start_time
    
    @property
    def bandwidth_bytes_per_ns(self) -> float:
        """区间带宽 (bytes/ns)"""
        if self.duration > 0:
            return self.total_bytes / self.duration
        return 0.0
    
    @property
    def bandwidth_gbps(self) -> float:
        """区间带宽 (GB/s)"""
        return self.bandwidth_bytes_per_ns
    
    @property
    def throughput_requests_per_ns(self) -> float:
        """区间吞吐量 (requests/ns)"""
        if self.duration > 0:
            return self.request_count / self.duration
        return 0.0


@dataclass
class BandwidthMetrics:
    """带宽指标"""
    request_type: RequestType
    
    # 基础指标
    total_bytes: int
    total_requests: int
    total_duration: int
    
    # 带宽指标
    average_bandwidth_gbps: float
    peak_bandwidth_gbps: float
    effective_bandwidth_gbps: float  # 基于工作区间计算
    
    # 工作区间统计
    working_intervals: List[WorkingInterval] = field(default_factory=list)
    total_working_time: int = 0
    utilization_ratio: float = 0.0  # 工作时间占比
    
    @property
    def bandwidth_efficiency(self) -> float:
        """带宽效率 (effective/peak)"""
        if self.peak_bandwidth_gbps > 0:
            return self.effective_bandwidth_gbps / self.peak_bandwidth_gbps
        return 0.0


@dataclass
class LatencyMetrics:
    """延迟指标"""
    request_type: RequestType
    
    # 基础统计
    sample_count: int
    
    # 延迟统计 (单位: ns)
    avg_total_latency: float
    min_total_latency: int
    max_total_latency: int
    p50_total_latency: float
    p95_total_latency: float
    p99_total_latency: float
    
    # 细分延迟
    avg_cmd_latency: float
    avg_data_latency: float
    avg_network_latency: float
    
    # 延迟分布
    latency_distribution: Dict[str, int] = field(default_factory=dict)
    
    @classmethod
    def from_requests(cls, requests: List[RequestMetrics], request_type: RequestType) -> 'LatencyMetrics':
        """从请求列表计算延迟指标"""
        filtered_requests = [r for r in requests if request_type == RequestType.ALL or r.request_type == request_type]
        
        if not filtered_requests:
            return cls(
                request_type=request_type,
                sample_count=0,
                avg_total_latency=0,
                min_total_latency=0,
                max_total_latency=0,
                p50_total_latency=0,
                p95_total_latency=0,
                p99_total_latency=0,
                avg_cmd_latency=0,
                avg_data_latency=0,
                avg_network_latency=0
            )
        
        # 收集延迟数据
        total_latencies = [r.total_latency for r in filtered_requests]
        cmd_latencies = [r.cmd_latency for r in filtered_requests]
        data_latencies = [r.data_latency for r in filtered_requests]
        network_latencies = [r.network_latency for r in filtered_requests]
        
        # 计算统计值
        total_latencies_array = np.array(total_latencies)
        
        return cls(
            request_type=request_type,
            sample_count=len(filtered_requests),
            avg_total_latency=float(np.mean(total_latencies_array)),
            min_total_latency=int(np.min(total_latencies_array)),
            max_total_latency=int(np.max(total_latencies_array)),
            p50_total_latency=float(np.percentile(total_latencies_array, 50)),
            p95_total_latency=float(np.percentile(total_latencies_array, 95)),
            p99_total_latency=float(np.percentile(total_latencies_array, 99)),
            avg_cmd_latency=float(np.mean(cmd_latencies)),
            avg_data_latency=float(np.mean(data_latencies)),
            avg_network_latency=float(np.mean(network_latencies))
        )


@dataclass
class ThroughputMetrics:
    """吞吐量指标"""
    request_type: RequestType
    
    # 基础指标
    total_requests: int
    total_duration: int
    
    # 吞吐量指标 (requests/ns)
    average_throughput: float
    peak_throughput: float
    sustained_throughput: float  # 基于工作区间
    
    # 时间窗口统计
    throughput_over_time: List[float] = field(default_factory=list)
    time_windows: List[int] = field(default_factory=list)
    
    @property
    def throughput_requests_per_second(self) -> float:
        """吞吐量 (requests/s)"""
        return self.average_throughput * 1e9
    
    @property
    def throughput_stability(self) -> float:
        """吞吐量稳定性 (std/mean)"""
        if self.throughput_over_time and self.average_throughput > 0:
            return float(np.std(self.throughput_over_time) / self.average_throughput)
        return 0.0


@dataclass
class HotspotMetrics:
    """热点分析指标"""
    node_id: int
    node_type: str
    
    # 流量统计
    incoming_requests: int
    outgoing_requests: int
    total_bytes_received: int
    total_bytes_sent: int
    
    # 拥塞指标
    average_queue_length: float
    max_queue_length: int
    congestion_ratio: float  # 拥塞时间比例
    
    # 性能指标
    average_latency: float
    bandwidth_utilization: float
    
    @property
    def load_balance_ratio(self) -> float:
        """负载平衡比例"""
        total_traffic = self.incoming_requests + self.outgoing_requests
        if total_traffic > 0:
            return min(self.incoming_requests, self.outgoing_requests) / total_traffic
        return 0.0
    
    @property
    def is_hotspot(self) -> bool:
        """是否为热点节点"""
        return self.congestion_ratio > 0.1 or self.bandwidth_utilization > 0.8


@dataclass
class NetworkMetrics:
    """网络整体性能指标"""
    
    # 基础信息
    topology_type: str
    node_count: int
    simulation_duration: int
    
    # 分类性能指标
    read_metrics: Dict[str, Union[BandwidthMetrics, LatencyMetrics, ThroughputMetrics]]
    write_metrics: Dict[str, Union[BandwidthMetrics, LatencyMetrics, ThroughputMetrics]]
    overall_metrics: Dict[str, Union[BandwidthMetrics, LatencyMetrics, ThroughputMetrics]]
    
    # 网络级指标
    network_utilization: float
    average_hop_count: float
    congestion_count: int
    
    # 热点分析
    hotspot_nodes: List[HotspotMetrics] = field(default_factory=list)
    
    # 能耗指标 (如果适用)
    total_energy_consumption: float = 0.0
    energy_efficiency: float = 0.0  # bytes/joule
    
    @property
    def overall_bandwidth_gbps(self) -> float:
        """整体带宽 (GB/s)"""
        if 'bandwidth' in self.overall_metrics:
            return self.overall_metrics['bandwidth'].average_bandwidth_gbps
        return 0.0
    
    @property
    def overall_latency_ns(self) -> float:
        """整体平均延迟 (ns)"""
        if 'latency' in self.overall_metrics:
            return self.overall_metrics['latency'].avg_total_latency
        return 0.0
    
    @property
    def overall_throughput_rps(self) -> float:
        """整体吞吐量 (requests/s)"""
        if 'throughput' in self.overall_metrics:
            return self.overall_metrics['throughput'].throughput_requests_per_second
        return 0.0
    
    def get_performance_summary(self) -> Dict[str, float]:
        """获取性能摘要"""
        return {
            'bandwidth_gbps': self.overall_bandwidth_gbps,
            'latency_ns': self.overall_latency_ns,
            'throughput_rps': self.overall_throughput_rps,
            'network_utilization': self.network_utilization,
            'avg_hop_count': self.average_hop_count,
            'hotspot_count': len(self.hotspot_nodes),
            'energy_efficiency': self.energy_efficiency
        }


@dataclass
class PerformanceMetrics:
    """完整的性能指标集合"""
    
    # 基础请求数据
    requests: List[RequestMetrics] = field(default_factory=list)
    
    # 网络指标
    network_metrics: Optional[NetworkMetrics] = None
    
    # 分析配置
    analysis_config: Dict[str, Union[str, int, float]] = field(default_factory=dict)
    
    def add_request(self, request: RequestMetrics):
        """添加请求指标"""
        self.requests.append(request)
    
    def get_requests_by_type(self, request_type: RequestType) -> List[RequestMetrics]:
        """按类型获取请求"""
        if request_type == RequestType.ALL:
            return self.requests
        return [r for r in self.requests if r.request_type == request_type]
    
    def get_requests_by_node(self, node_id: int, direction: str = 'both') -> List[RequestMetrics]:
        """按节点获取请求"""
        if direction == 'source':
            return [r for r in self.requests if r.source_node == node_id]
        elif direction == 'dest':
            return [r for r in self.requests if r.dest_node == node_id]
        else:  # both
            return [r for r in self.requests if r.source_node == node_id or r.dest_node == node_id]
    
    def calculate_summary_stats(self) -> Dict[str, float]:
        """计算汇总统计"""
        if not self.requests:
            return {}
        
        total_bytes = sum(r.total_bytes for r in self.requests)
        total_requests = len(self.requests)
        total_duration = max(r.end_time for r in self.requests) - min(r.start_time for r in self.requests)
        
        read_requests = [r for r in self.requests if r.request_type == RequestType.READ]
        write_requests = [r for r in self.requests if r.request_type == RequestType.WRITE]
        
        return {
            'total_requests': total_requests,
            'total_bytes': total_bytes,
            'total_duration_ns': total_duration,
            'read_requests': len(read_requests),
            'write_requests': len(write_requests),
            'avg_latency_ns': np.mean([r.total_latency for r in self.requests]),
            'avg_bandwidth_gbps': np.mean([r.bandwidth_gbps for r in self.requests]),
            'avg_hop_count': np.mean([r.hop_count for r in self.requests if r.hop_count > 0])
        }