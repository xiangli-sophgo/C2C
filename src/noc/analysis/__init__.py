"""
NoC Analysis Module
网络片上系统分析模块

提供带宽、延迟、吞吐量等性能指标的分析和可视化功能
包含完整的输出管理和会话管理功能
"""

from .result_processor import ResultProcessor, BandwidthAnalyzer, LatencyAnalyzer, ThroughputAnalyzer
from .performance_metrics import PerformanceMetrics, RequestMetrics, NetworkMetrics
from .visualization import PerformanceVisualizer, NetworkFlowVisualizer
from .output_manager import OutputManager, SimulationContext

# 简化版本 (推荐日常使用)
from .simple_result_processor import SimpleResultProcessor, create_simple_analysis_session
from .simple_visualizer import SimplePerformanceVisualizer
from .simple_output_manager import SimpleOutputManager, SimpleSimulationContext

__all__ = [
    # 完整版本 (深度分析)
    'ResultProcessor',
    'BandwidthAnalyzer',
    'LatencyAnalyzer', 
    'ThroughputAnalyzer',
    'PerformanceMetrics',
    'RequestMetrics',
    'NetworkMetrics',
    'PerformanceVisualizer',
    'NetworkFlowVisualizer',
    'OutputManager',
    'SimulationContext',
    
    # 简化版本 (快速分析)
    'SimpleResultProcessor',
    'SimplePerformanceVisualizer', 
    'SimpleOutputManager',
    'SimpleSimulationContext',
    'create_simple_analysis_session'
]