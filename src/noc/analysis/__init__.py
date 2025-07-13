"""
NoC分析模块
提供CrossRing专用的性能分析功能
"""

from .crossring_analyzer import CrossRingAnalyzer, RequestType, RequestMetrics

__all__ = ['CrossRingAnalyzer', 'RequestType', 'RequestMetrics']