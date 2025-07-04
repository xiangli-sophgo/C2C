"""
C2C 工具模块 - 芯片间通信工具类
包含异常定义、类型定义和通用工具
"""

# 导入异常类
from src.c2c.utils.exceptions import (
    ProtocolError,
    CDMAError,
    AddressError,
    ShapeCompatibilityError,
)


__version__ = "1.0.0"

__all__ = [
    # 异常类
    "ProtocolError",
    "CDMAError",
    "AddressError",
    "ShapeCompatibilityError",
]
