"""
NoC通用基础模块。

提供所有NoC拓扑共用的基础类和功能，包括：
- BaseFlit: 通用Flit基类，支持STI协议和重试机制
- BaseIPInterface: 通用IP接口基类，支持时钟域转换和资源管理
- BaseNoCModel: 通用NoC模型基类，支持仿真循环和性能统计
- BaseResourceManager: 通用资源管理器
"""

from .flit import (
    BaseFlit,
    FlitPool,
    create_flit,
)

from .ip_interface import (
    BaseIPInterface,
)

from .model import BaseNoCModel

# 版本信息
__version__ = "1.0.0"
__author__ = "xiang.li"

# 导出所有基础类
__all__ = [
    # Flit相关
    "BaseFlit",
    "FlitPool",
    "create_flit",
    # IP接口相关
    "BaseIPInterface",
    # 模型相关
    "BaseNoCModel",
]


# 模块初始化检查
def _check_base_dependencies():
    """检查基础模块依赖"""
    try:
        import numpy
        import logging
        from collections import deque, defaultdict
        from abc import ABC, abstractmethod
        from dataclasses import dataclass, field
        from typing import Dict, List, Any, Optional, Type, Union
    except ImportError as e:
        raise ImportError(f"NoC基础模块缺少必要依赖: {e}")


# 执行依赖检查
_check_base_dependencies()

# 模块级日志
import logging

_logger = logging.getLogger(__name__)
_logger.info(f"NoC基础模块加载完成 (v{__version__})")
