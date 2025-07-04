"""
C2C 工具模块 - 芯片间通信工具类
包含异常定义、类型定义和通用工具
"""

# 导入异常类
from .exceptions import (
    C2CException,
    TopologyError,
    ProtocolError,
    CDMAError,
    AddressError,
    ShapeCompatibilityError,
    ConfigError,
    ValidationError,
)

# 导入类型定义
from .types import NodeId, Priority

__version__ = "1.0.0"

__all__ = [
    # 异常类
    "C2CException",
    "TopologyError",
    "ProtocolError",
    "CDMAError",
    "AddressError",
    "ShapeCompatibilityError",
    "ConfigError",
    "ValidationError",
    # 类型定义
    "NodeId",
    "Priority",
]
