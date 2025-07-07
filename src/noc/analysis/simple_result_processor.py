"""
简化的结果处理器 - 专注于核心分析功能
减少输出文件，提供最重要的分析结果
"""

from typing import Dict, List, Optional, Any
from collections import defaultdict
import numpy as np
import logging
from .performance_metrics import (
    PerformanceMetrics, RequestMetrics, NetworkMetrics, 
    BandwidthMetrics, LatencyMetrics, ThroughputMetrics, 
    RequestType
)
from .simple_output_manager import SimpleOutputManager


class SimpleBandwidthAnalyzer:
    """简化的带宽分析器"""
    
    def __init__(self, gap_threshold_ns: int = 200):
        self.gap_threshold_ns = gap_threshold_ns
    
    def analyze(self, requests: List[RequestMetrics]) -> Dict[str, float]:
        """返回核心带宽指标"""
        if not requests:
            return {'total_bandwidth_gbps': 0.0, 'utilization': 0.0}
        
        total_bytes = sum(r.total_bytes for r in requests)
        start_time = min(r.start_time for r in requests)
        end_time = max(r.end_time for r in requests)
        duration = end_time - start_time
        
        total_bandwidth = total_bytes / duration if duration > 0 else 0.0
        
        # 简化的利用率计算
        active_time = sum(r.end_time - r.start_time for r in requests)
        utilization = active_time / (duration * len(set(r.source_node for r in requests))) if duration > 0 else 0.0
        
        return {
            'total_bandwidth_gbps': total_bandwidth,
            'utilization': min(1.0, utilization)
        }


class SimpleLatencyAnalyzer:
    """简化的延迟分析器"""
    
    def analyze(self, requests: List[RequestMetrics]) -> Dict[str, float]:
        """返回核心延迟指标"""
        if not requests:
            return {'avg_latency_ns': 0.0, 'p95_latency_ns': 0.0}
        
        latencies = [r.total_latency for r in requests]
        
        return {
            'avg_latency_ns': float(np.mean(latencies)),
            'p95_latency_ns': float(np.percentile(latencies, 95)),
            'max_latency_ns': float(np.max(latencies))
        }


class SimpleThroughputAnalyzer:
    """简化的吞吐量分析器"""
    
    def analyze(self, requests: List[RequestMetrics]) -> Dict[str, float]:
        """返回核心吞吐量指标"""
        if not requests:
            return {'throughput_rps': 0.0}
        
        duration = max(r.end_time for r in requests) - min(r.start_time for r in requests)
        throughput = len(requests) / (duration / 1e9) if duration > 0 else 0.0
        
        return {
            'throughput_rps': throughput,
            'total_requests': len(requests)
        }


class SimpleHotspotAnalyzer:
    """简化的热点分析器"""
    
    def analyze(self, requests: List[RequestMetrics]) -> Dict[str, Any]:
        """返回热点分析结果"""
        node_traffic = defaultdict(int)
        
        for req in requests:
            node_traffic[req.source_node] += req.total_bytes
            node_traffic[req.dest_node] += req.total_bytes
        
        if not node_traffic:
            return {'hotspot_count': 0, 'max_traffic_node': None}
        
        max_traffic = max(node_traffic.values())
        avg_traffic = np.mean(list(node_traffic.values()))
        
        # 简化的热点检测：流量超过平均值2倍的节点
        hotspots = [node_id for node_id, traffic in node_traffic.items() 
                   if traffic > avg_traffic * 2]
        
        return {
            'hotspot_count': len(hotspots),
            'hotspot_nodes': hotspots,
            'max_traffic_node': max(node_traffic.items(), key=lambda x: x[1])[0]
        }


class SimpleResultProcessor:
    """简化的结果处理器 - 专注核心功能"""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None, output_manager: Optional[SimpleOutputManager] = None):
        self.config = config or {}
        self.output_manager = output_manager
        self.logger = logging.getLogger(self.__class__.__name__)
        
        # 简化的分析器
        self.bandwidth_analyzer = SimpleBandwidthAnalyzer(
            gap_threshold_ns=self.config.get('gap_threshold_ns', 200)
        )
        self.latency_analyzer = SimpleLatencyAnalyzer()
        self.throughput_analyzer = SimpleThroughputAnalyzer()
        self.hotspot_analyzer = SimpleHotspotAnalyzer()
        
        # 请求数据
        self.requests: List[RequestMetrics] = []
    
    def add_requests(self, requests: List[RequestMetrics]):
        """添加请求数据"""
        self.requests.extend(requests)
        self.logger.info(f"添加了 {len(requests)} 个请求，总计 {len(self.requests)} 个")
    
    def analyze_performance(self) -> Dict[str, Any]:
        """执行完整的性能分析，返回核心指标"""
        if not self.requests:
            self.logger.warning("没有请求数据")
            return {}
        
        self.logger.info("开始性能分析")
        
        # 执行各项分析
        bandwidth_results = self.bandwidth_analyzer.analyze(self.requests)
        latency_results = self.latency_analyzer.analyze(self.requests)
        throughput_results = self.throughput_analyzer.analyze(self.requests)
        hotspot_results = self.hotspot_analyzer.analyze(self.requests)
        
        # 计算网络级指标
        hop_counts = [r.hop_count for r in self.requests if r.hop_count > 0]
        avg_hop_count = np.mean(hop_counts) if hop_counts else 0.0
        
        # 分类统计
        read_requests = [r for r in self.requests if r.request_type == RequestType.READ]
        write_requests = [r for r in self.requests if r.request_type == RequestType.WRITE]
        
        # 汇总结果
        results = {
            # 基础统计
            'total_requests': len(self.requests),
            'read_requests': len(read_requests),
            'write_requests': len(write_requests),
            'avg_hop_count': avg_hop_count,
            
            # 性能指标
            'bandwidth_gbps': bandwidth_results['total_bandwidth_gbps'],
            'network_utilization': bandwidth_results['utilization'],
            'latency_ns': latency_results['avg_latency_ns'],
            'p95_latency_ns': latency_results['p95_latency_ns'],
            'max_latency_ns': latency_results['max_latency_ns'],
            'throughput_rps': throughput_results['throughput_rps'],
            
            # 热点分析
            'hotspot_count': hotspot_results['hotspot_count'],
            'hotspot_nodes': hotspot_results['hotspot_nodes'],
            'max_traffic_node': hotspot_results['max_traffic_node'],
            
            # 配置信息
            'topology_type': self.config.get('topology_type', 'unknown'),
            'node_count': self.config.get('node_count', 0)
        }
        
        # 保存分析结果
        if self.output_manager:
            self.output_manager.save_data(results, "performance_summary", "json")
            self.output_manager.save_summary_report(results)
        
        self.logger.info("性能分析完成")
        return results
    
    def export_detailed_data(self, filename: str = "data_export") -> Optional[str]:
        """导出详细数据到Excel"""
        if not self.requests or not self.output_manager:
            return None
        
        try:
            import pandas as pd
            
            # 准备请求数据
            request_data = []
            for req in self.requests:
                request_data.append({
                    'packet_id': req.packet_id,
                    'type': req.request_type.value,
                    'source': req.source_node,
                    'dest': req.dest_node,
                    'start_time': req.start_time,
                    'end_time': req.end_time,
                    'latency': req.total_latency,
                    'bytes': req.total_bytes,
                    'bandwidth': req.bandwidth_gbps,
                    'hops': req.hop_count
                })
            
            df = pd.DataFrame(request_data)
            return self.output_manager.save_data(df, filename, "xlsx")
            
        except ImportError:
            self.logger.warning("需要安装pandas和openpyxl来导出Excel文件")
            return None
    
    def get_performance_summary(self) -> Dict[str, Any]:
        """获取性能摘要"""
        return self.analyze_performance()


def create_simple_analysis_session(model_name: str, topology_type: str, config: Dict[str, Any], session_name: Optional[str] = None):
    """创建简化的分析会话"""
    from .simple_output_manager import SimpleSimulationContext
    
    return SimpleSimulationContext(
        model_name=model_name,
        topology_type=topology_type,
        config=config,
        session_name=session_name
    )