"""
NoC实用工具模块。

本模块提供NoC实现中常用的工具函数和类。
"""

from .adjacency import create_crossring_adjacency_matrix, validate_adjacency_matrix, check_connectivity

__all__ = [
    "create_crossring_adjacency_matrix",
    "validate_adjacency_matrix", 
    "check_connectivity"
]