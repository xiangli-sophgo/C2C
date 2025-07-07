"""
结果处理器 - 现代化的NoC性能分析引擎
整合带宽、延迟、吞吐量等多维度分析功能
"""

from typing import Dict, List, Optional, Tuple, Any, Union
from collections import defaultdict
import numpy as np
import logging
from dataclasses import dataclass
from .performance_metrics import (
    PerformanceMetrics, RequestMetrics, NetworkMetrics, 
    BandwidthMetrics, LatencyMetrics, ThroughputMetrics, 
    WorkingInterval, HotspotMetrics, RequestType
)
from .output_manager import OutputManager


class BandwidthAnalyzer:
    """带宽分析器"""
    
    def __init__(self, gap_threshold_ns: int = 200):
        self.gap_threshold_ns = gap_threshold_ns
        self.logger = logging.getLogger(self.__class__.__name__)
    
    def calculate_working_intervals(self, requests: List[RequestMetrics]) -> List[WorkingInterval]:
        """计算工作区间"""
        if not requests:
            return []
        
        # 按开始时间排序
        sorted_requests = sorted(requests, key=lambda r: r.start_time)
        
        intervals = []
        current_start = sorted_requests[0].start_time
        current_end = sorted_requests[0].end_time
        current_bytes = sorted_requests[0].total_bytes
        current_count = 1
        
        for req in sorted_requests[1:]:
            # 检查是否可以合并到当前区间
            if req.start_time - current_end <= self.gap_threshold_ns:
                current_end = max(current_end, req.end_time)
                current_bytes += req.total_bytes
                current_count += 1
            else:
                # 创建新区间
                intervals.append(WorkingInterval(
                    start_time=current_start,
                    end_time=current_end,
                    request_count=current_count,
                    total_bytes=current_bytes
                ))
                current_start = req.start_time
                current_end = req.end_time
                current_bytes = req.total_bytes
                current_count = 1
        
        # 添加最后一个区间
        intervals.append(WorkingInterval(
            start_time=current_start,
            end_time=current_end,
            request_count=current_count,
            total_bytes=current_bytes
        ))
        
        return intervals
    
    def calculate_bandwidth_metrics(self, requests: List[RequestMetrics], 
                                  request_type: RequestType = RequestType.ALL) -> BandwidthMetrics:
        """计算带宽指标"""
        filtered_requests = [r for r in requests if request_type == RequestType.ALL or r.request_type == request_type]
        
        if not filtered_requests:
            return BandwidthMetrics(
                request_type=request_type,
                total_bytes=0,
                total_requests=0,
                total_duration=0,
                average_bandwidth_gbps=0.0,
                peak_bandwidth_gbps=0.0,
                effective_bandwidth_gbps=0.0
            )
        
        # 基础统计
        total_bytes = sum(r.total_bytes for r in filtered_requests)
        total_requests = len(filtered_requests)
        start_time = min(r.start_time for r in filtered_requests)
        end_time = max(r.end_time for r in filtered_requests)
        total_duration = end_time - start_time
        
        # 计算工作区间
        working_intervals = self.calculate_working_intervals(filtered_requests)
        total_working_time = sum(interval.duration for interval in working_intervals)
        
        # 计算各种带宽指标
        average_bandwidth = total_bytes / total_duration if total_duration > 0 else 0.0
        peak_bandwidth = max(r.bandwidth_gbps for r in filtered_requests) if filtered_requests else 0.0
        effective_bandwidth = total_bytes / total_working_time if total_working_time > 0 else 0.0
        
        utilization_ratio = total_working_time / total_duration if total_duration > 0 else 0.0
        
        return BandwidthMetrics(
            request_type=request_type,
            total_bytes=total_bytes,
            total_requests=total_requests,
            total_duration=total_duration,
            average_bandwidth_gbps=average_bandwidth,
            peak_bandwidth_gbps=peak_bandwidth,
            effective_bandwidth_gbps=effective_bandwidth,
            working_intervals=working_intervals,
            total_working_time=total_working_time,
            utilization_ratio=utilization_ratio
        )


class LatencyAnalyzer:
    """延迟分析器"""
    
    def __init__(self):
        self.logger = logging.getLogger(self.__class__.__name__)
    
    def calculate_latency_metrics(self, requests: List[RequestMetrics],
                                request_type: RequestType = RequestType.ALL) -> LatencyMetrics:
        """计算延迟指标"""
        return LatencyMetrics.from_requests(requests, request_type)
    
    def analyze_latency_by_distance(self, requests: List[RequestMetrics]) -> Dict[int, LatencyMetrics]:
        """按跳数分析延迟"""
        hop_groups = defaultdict(list)
        for req in requests:
            hop_groups[req.hop_count].append(req)
        
        return {
            hop_count: LatencyMetrics.from_requests(reqs, RequestType.ALL)
            for hop_count, reqs in hop_groups.items()
        }
    
    def analyze_latency_by_node_pair(self, requests: List[RequestMetrics]) -> Dict[Tuple[int, int], LatencyMetrics]:
        """按节点对分析延迟"""
        node_pair_groups = defaultdict(list)
        for req in requests:
            pair = (req.source_node, req.dest_node)
            node_pair_groups[pair].append(req)
        
        return {
            pair: LatencyMetrics.from_requests(reqs, RequestType.ALL)
            for pair, reqs in node_pair_groups.items()
        }


class ThroughputAnalyzer:
    """吞吐量分析器"""
    
    def __init__(self, window_size_ns: int = 1000):
        self.window_size_ns = window_size_ns
        self.logger = logging.getLogger(self.__class__.__name__)
    
    def calculate_throughput_metrics(self, requests: List[RequestMetrics],
                                   request_type: RequestType = RequestType.ALL) -> ThroughputMetrics:
        """计算吞吐量指标"""
        filtered_requests = [r for r in requests if request_type == RequestType.ALL or r.request_type == request_type]
        
        if not filtered_requests:
            return ThroughputMetrics(
                request_type=request_type,
                total_requests=0,
                total_duration=0,
                average_throughput=0.0,
                peak_throughput=0.0,
                sustained_throughput=0.0
            )
        
        # 基础统计
        total_requests = len(filtered_requests)
        start_time = min(r.start_time for r in filtered_requests)
        end_time = max(r.end_time for r in filtered_requests)
        total_duration = end_time - start_time
        
        # 计算平均吞吐量
        average_throughput = total_requests / total_duration if total_duration > 0 else 0.0
        
        # 计算时间窗口内的吞吐量
        throughput_over_time, time_windows = self._calculate_windowed_throughput(filtered_requests)
        
        # 峰值和持续吞吐量
        peak_throughput = max(throughput_over_time) if throughput_over_time else 0.0
        sustained_throughput = np.mean(throughput_over_time) if throughput_over_time else 0.0
        
        return ThroughputMetrics(
            request_type=request_type,
            total_requests=total_requests,
            total_duration=total_duration,
            average_throughput=average_throughput,
            peak_throughput=peak_throughput,
            sustained_throughput=sustained_throughput,
            throughput_over_time=throughput_over_time,
            time_windows=time_windows
        )
    
    def _calculate_windowed_throughput(self, requests: List[RequestMetrics]) -> Tuple[List[float], List[int]]:
        """计算时间窗口内的吞吐量"""
        if not requests:
            return [], []
        
        start_time = min(r.start_time for r in requests)
        end_time = max(r.end_time for r in requests)
        
        throughput_over_time = []
        time_windows = []
        
        current_time = start_time
        while current_time < end_time:
            window_end = current_time + self.window_size_ns
            
            # 统计在当前时间窗口内完成的请求
            window_requests = [r for r in requests if current_time <= r.end_time < window_end]
            window_throughput = len(window_requests) / self.window_size_ns
            
            throughput_over_time.append(window_throughput)
            time_windows.append(current_time)
            
            current_time = window_end
        
        return throughput_over_time, time_windows


class HotspotAnalyzer:
    """热点分析器"""
    
    def __init__(self, congestion_threshold: float = 0.8):
        self.congestion_threshold = congestion_threshold
        self.logger = logging.getLogger(self.__class__.__name__)
    
    def analyze_hotspots(self, requests: List[RequestMetrics], 
                        node_types: Optional[Dict[int, str]] = None) -> List[HotspotMetrics]:
        """分析网络热点"""
        if node_types is None:
            node_types = {}
        
        # 统计每个节点的流量
        node_stats = defaultdict(lambda: {
            'incoming': 0, 'outgoing': 0,
            'bytes_received': 0, 'bytes_sent': 0,
            'latencies': []
        })
        
        for req in requests:
            # 源节点统计
            node_stats[req.source_node]['outgoing'] += 1
            node_stats[req.source_node]['bytes_sent'] += req.total_bytes
            
            # 目标节点统计
            node_stats[req.dest_node]['incoming'] += 1
            node_stats[req.dest_node]['bytes_received'] += req.total_bytes
            node_stats[req.dest_node]['latencies'].append(req.total_latency)
        
        # 计算热点指标
        hotspots = []
        for node_id, stats in node_stats.items():
            avg_latency = np.mean(stats['latencies']) if stats['latencies'] else 0.0
            
            # 简化的拥塞和带宽利用率计算
            total_traffic = stats['incoming'] + stats['outgoing']
            congestion_ratio = min(1.0, total_traffic / 100.0)  # 假设阈值为100
            bandwidth_utilization = min(1.0, (stats['bytes_received'] + stats['bytes_sent']) / 1e9)  # 假设1GB/s峰值
            
            hotspot = HotspotMetrics(
                node_id=node_id,
                node_type=node_types.get(node_id, 'unknown'),
                incoming_requests=stats['incoming'],
                outgoing_requests=stats['outgoing'],
                total_bytes_received=stats['bytes_received'],
                total_bytes_sent=stats['bytes_sent'],
                average_queue_length=0.0,  # 需要额外数据
                max_queue_length=0,  # 需要额外数据
                congestion_ratio=congestion_ratio,
                average_latency=avg_latency,
                bandwidth_utilization=bandwidth_utilization
            )
            
            hotspots.append(hotspot)
        
        # 按拥塞程度排序
        hotspots.sort(key=lambda h: h.congestion_ratio, reverse=True)
        
        return hotspots


class ResultProcessor:
    """结果处理器主类"""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None, output_manager: Optional[OutputManager] = None):
        self.config = config or {}
        self.logger = logging.getLogger(self.__class__.__name__)
        self.output_manager = output_manager
        
        # 初始化分析器
        self.bandwidth_analyzer = BandwidthAnalyzer(
            gap_threshold_ns=self.config.get('gap_threshold_ns', 200)
        )
        self.latency_analyzer = LatencyAnalyzer()
        self.throughput_analyzer = ThroughputAnalyzer(
            window_size_ns=self.config.get('window_size_ns', 1000)
        )
        self.hotspot_analyzer = HotspotAnalyzer(
            congestion_threshold=self.config.get('congestion_threshold', 0.8)
        )
        
        # 结果存储
        self.performance_metrics = PerformanceMetrics()
        self.performance_metrics.analysis_config = self.config
    
    def collect_simulation_data(self, simulation_model: Any) -> None:
        """从仿真模型收集数据"""
        self.logger.info("开始收集仿真数据")
        
        # 清空之前的数据
        self.performance_metrics.requests.clear()
        
        # 从仿真模型提取请求数据
        requests = self._extract_requests_from_simulation(simulation_model)
        
        # 添加到性能指标
        for req in requests:
            self.performance_metrics.add_request(req)
        
        self.logger.info(f"收集到 {len(requests)} 个请求")
    
    def _extract_requests_from_simulation(self, simulation_model: Any) -> List[RequestMetrics]:
        """从仿真模型提取请求数据"""
        requests = []
        
        # 假设仿真模型有 completed_requests 属性
        if hasattr(simulation_model, 'completed_requests'):
            for req_data in simulation_model.completed_requests:
                request = RequestMetrics(
                    packet_id=str(req_data.get('packet_id', 0)),
                    request_type=RequestType.READ if req_data.get('type') == 'read' else RequestType.WRITE,
                    source_node=req_data.get('source', 0),
                    dest_node=req_data.get('destination', 0),
                    burst_size=req_data.get('burst_size', 1),
                    start_time=req_data.get('start_time', 0),
                    end_time=req_data.get('end_time', 0),
                    cmd_latency=req_data.get('cmd_latency', 0),
                    data_latency=req_data.get('data_latency', 0),
                    network_latency=req_data.get('network_latency', 0),
                    total_bytes=req_data.get('total_bytes', 0),
                    hop_count=req_data.get('hop_count', 0),
                    path_nodes=req_data.get('path_nodes', [])
                )
                requests.append(request)
        
        return requests
    
    def analyze_performance(self, detailed: bool = True) -> NetworkMetrics:
        """执行完整的性能分析"""
        self.logger.info("开始性能分析")
        
        requests = self.performance_metrics.requests
        if not requests:
            self.logger.warning("没有找到请求数据")
            return self._create_empty_network_metrics()
        
        # 分析各种指标
        read_bandwidth = self.bandwidth_analyzer.calculate_bandwidth_metrics(requests, RequestType.READ)
        write_bandwidth = self.bandwidth_analyzer.calculate_bandwidth_metrics(requests, RequestType.WRITE)
        overall_bandwidth = self.bandwidth_analyzer.calculate_bandwidth_metrics(requests, RequestType.ALL)
        
        read_latency = self.latency_analyzer.calculate_latency_metrics(requests, RequestType.READ)
        write_latency = self.latency_analyzer.calculate_latency_metrics(requests, RequestType.WRITE)
        overall_latency = self.latency_analyzer.calculate_latency_metrics(requests, RequestType.ALL)
        
        read_throughput = self.throughput_analyzer.calculate_throughput_metrics(requests, RequestType.READ)
        write_throughput = self.throughput_analyzer.calculate_throughput_metrics(requests, RequestType.WRITE)
        overall_throughput = self.throughput_analyzer.calculate_throughput_metrics(requests, RequestType.ALL)
        
        # 热点分析
        hotspots = self.hotspot_analyzer.analyze_hotspots(requests)
        
        # 计算网络级指标
        network_utilization = self._calculate_network_utilization(requests)
        average_hop_count = np.mean([r.hop_count for r in requests if r.hop_count > 0])
        
        # 创建网络指标
        network_metrics = NetworkMetrics(
            topology_type=self.config.get('topology_type', 'unknown'),
            node_count=self.config.get('node_count', 0),
            simulation_duration=max(r.end_time for r in requests) - min(r.start_time for r in requests),
            read_metrics={
                'bandwidth': read_bandwidth,
                'latency': read_latency,
                'throughput': read_throughput
            },
            write_metrics={
                'bandwidth': write_bandwidth,
                'latency': write_latency,
                'throughput': write_throughput
            },
            overall_metrics={
                'bandwidth': overall_bandwidth,
                'latency': overall_latency,
                'throughput': overall_throughput
            },
            network_utilization=network_utilization,
            average_hop_count=average_hop_count,
            congestion_count=len([h for h in hotspots if h.is_hotspot]),
            hotspot_nodes=hotspots
        )
        
        self.performance_metrics.network_metrics = network_metrics
        
        self.logger.info("性能分析完成")
        return network_metrics
    
    def _calculate_network_utilization(self, requests: List[RequestMetrics]) -> float:
        """计算网络利用率"""
        if not requests:
            return 0.0
        
        # 简化的网络利用率计算
        total_bytes = sum(r.total_bytes for r in requests)
        total_duration = max(r.end_time for r in requests) - min(r.start_time for r in requests)
        
        # 假设网络峰值带宽为10GB/s
        peak_bandwidth = 10 * 1e9  # bytes/s
        actual_bandwidth = total_bytes / (total_duration / 1e9)  # bytes/s
        
        return min(1.0, actual_bandwidth / peak_bandwidth)
    
    def _create_empty_network_metrics(self) -> NetworkMetrics:
        """创建空的网络指标"""
        empty_bandwidth = BandwidthMetrics(
            request_type=RequestType.ALL,
            total_bytes=0,
            total_requests=0,
            total_duration=0,
            average_bandwidth_gbps=0.0,
            peak_bandwidth_gbps=0.0,
            effective_bandwidth_gbps=0.0
        )
        
        empty_latency = LatencyMetrics(
            request_type=RequestType.ALL,
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
        
        empty_throughput = ThroughputMetrics(
            request_type=RequestType.ALL,
            total_requests=0,
            total_duration=0,
            average_throughput=0.0,
            peak_throughput=0.0,
            sustained_throughput=0.0
        )
        
        return NetworkMetrics(
            topology_type='unknown',
            node_count=0,
            simulation_duration=0,
            read_metrics={'bandwidth': empty_bandwidth, 'latency': empty_latency, 'throughput': empty_throughput},
            write_metrics={'bandwidth': empty_bandwidth, 'latency': empty_latency, 'throughput': empty_throughput},
            overall_metrics={'bandwidth': empty_bandwidth, 'latency': empty_latency, 'throughput': empty_throughput},
            network_utilization=0.0,
            average_hop_count=0.0,
            congestion_count=0
        )
    
    def get_performance_summary(self) -> Dict[str, Any]:
        """获取性能摘要"""
        if not self.performance_metrics.network_metrics:
            return {}
        
        summary = self.performance_metrics.network_metrics.get_performance_summary()
        
        # 添加基础统计
        basic_stats = self.performance_metrics.calculate_summary_stats()
        summary.update(basic_stats)
        
        return summary
    
    def export_results(self, format: str = 'json', filename: Optional[str] = None) -> Optional[str]:
        """导出结果"""
        if filename is None:
            filename = f"performance_results"
        
        if format == 'json':
            return self._export_json(filename)
        elif format == 'csv':
            return self._export_csv(filename)
        elif format == 'excel':
            return self._export_excel(filename)
        else:
            self.logger.error(f"不支持的导出格式: {format}")
            return None
    
    def _export_json(self, filename: str) -> str:
        """导出为JSON格式"""
        # 构建导出数据
        export_data = {
            'summary': self.get_performance_summary(),
            'analysis_config': self.performance_metrics.analysis_config,
            'timestamp': str(np.datetime64('now'))
        }
        
        if self.output_manager:
            return self.output_manager.save_data(export_data, filename, 'json')
        else:
            # 兼容模式，直接保存到当前目录
            import json
            output_path = f"{filename}.json"
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(export_data, f, indent=2, ensure_ascii=False)
            return output_path
    
    def _export_csv(self, filename: str) -> str:
        """导出为CSV格式"""
        import pandas as pd
        
        # 构建请求数据DataFrame
        request_data = []
        for req in self.performance_metrics.requests:
            request_data.append({
                'packet_id': req.packet_id,
                'request_type': req.request_type.value,
                'source_node': req.source_node,
                'dest_node': req.dest_node,
                'burst_size': req.burst_size,
                'start_time': req.start_time,
                'end_time': req.end_time,
                'total_latency': req.total_latency,
                'cmd_latency': req.cmd_latency,
                'data_latency': req.data_latency,
                'network_latency': req.network_latency,
                'total_bytes': req.total_bytes,
                'bandwidth_gbps': req.bandwidth_gbps,
                'hop_count': req.hop_count
            })
        
        df = pd.DataFrame(request_data)
        
        if self.output_manager:
            return self.output_manager.save_data(df, filename, 'csv')
        else:
            # 兼容模式
            output_path = f"{filename}.csv"
            df.to_csv(output_path, index=False)
            return output_path
    
    def _export_excel(self, output_path: Optional[str] = None) -> str:
        """导出为Excel格式"""
        import pandas as pd
        
        # 构建多个工作表
        with pd.ExcelWriter(output_path or 'noc_analysis.xlsx') as writer:
            # 请求详细数据
            request_data = []
            for req in self.performance_metrics.requests:
                request_data.append({
                    'packet_id': req.packet_id,
                    'request_type': req.request_type.value,
                    'source_node': req.source_node,
                    'dest_node': req.dest_node,
                    'burst_size': req.burst_size,
                    'start_time': req.start_time,
                    'end_time': req.end_time,
                    'total_latency': req.total_latency,
                    'cmd_latency': req.cmd_latency,
                    'data_latency': req.data_latency,
                    'network_latency': req.network_latency,
                    'total_bytes': req.total_bytes,
                    'bandwidth_gbps': req.bandwidth_gbps,
                    'hop_count': req.hop_count
                })
            
            df_requests = pd.DataFrame(request_data)
            df_requests.to_excel(writer, sheet_name='Requests', index=False)
            
            # 性能摘要
            summary = self.get_performance_summary()
            df_summary = pd.DataFrame(list(summary.items()), columns=['Metric', 'Value'])
            df_summary.to_excel(writer, sheet_name='Summary', index=False)
            
            # 热点分析
            if self.performance_metrics.network_metrics and self.performance_metrics.network_metrics.hotspot_nodes:
                hotspot_data = []
                for hotspot in self.performance_metrics.network_metrics.hotspot_nodes:
                    hotspot_data.append({
                        'node_id': hotspot.node_id,
                        'node_type': hotspot.node_type,
                        'incoming_requests': hotspot.incoming_requests,
                        'outgoing_requests': hotspot.outgoing_requests,
                        'total_bytes_received': hotspot.total_bytes_received,
                        'total_bytes_sent': hotspot.total_bytes_sent,
                        'congestion_ratio': hotspot.congestion_ratio,
                        'average_latency': hotspot.average_latency,
                        'bandwidth_utilization': hotspot.bandwidth_utilization,
                        'is_hotspot': hotspot.is_hotspot
                    })
                
                df_hotspots = pd.DataFrame(hotspot_data)
                df_hotspots.to_excel(writer, sheet_name='Hotspots', index=False)
        
        return output_path or 'noc_analysis.xlsx'