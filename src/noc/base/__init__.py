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
    BaseResourceManager,
)

from .model import BaseNoCModel

# 版本信息
__version__ = "1.0.0"
__author__ = "C2C NoC Team"

# 导出所有基础类
__all__ = [
    # Flit相关
    "BaseFlit",
    "FlitPool",
    "create_flit",
    # IP接口相关
    "BaseIPInterface",
    "BaseResourceManager",
    # 模型相关
    "BaseNoCModel",
]


# 模块信息
def get_base_module_info() -> dict:
    """获取基础模块信息"""
    return {
        "name": "NoC Base Module",
        "version": __version__,
        "author": __author__,
        "description": "通用NoC基础类库",
        "components": {
            "BaseFlit": "支持STI协议和重试机制的通用Flit类",
            "BaseIPInterface": "支持时钟域转换和资源管理的通用IP接口类",
            "BaseNoCModel": "支持仿真循环和性能统计的通用NoC模型类",
            "BaseResourceManager": "通用资源管理器",
        },
        "features": [
            "STI三通道协议支持",
            "通用重试机制",
            "时钟域转换框架",
            "资源管理框架",
            "性能统计收集",
            "仿真循环控制",
            "对象池优化",
            "调试和监控支持",
        ],
    }


# 便捷工厂函数
def create_basic_flit(source: int, destination: int, req_type: str = "read", **kwargs) -> BaseFlit:
    """
    创建基础Flit的便捷函数

    Args:
        source: 源节点
        destination: 目标节点
        req_type: 请求类型
        **kwargs: 其他参数

    Returns:
        BaseFlit实例
    """
    return BaseFlit(source=source, destination=destination, req_type=req_type, **kwargs)


def validate_base_module() -> bool:
    """验证基础模块功能"""
    try:
        # 测试Flit创建
        flit = create_basic_flit(0, 8, "read")

        # 测试STI协议功能
        response = flit.create_response("ack")
        data_flit = flit.create_data_flit(0)

        # 测试重试机制
        if flit.can_retry():
            flit.prepare_for_retry("test")

        # 测试对象池
        pool = FlitPool(BaseFlit)
        test_flit = pool.get_flit(source=0, destination=1)
        pool.return_flit(test_flit)

        return True
    except Exception as e:
        print(f"基础模块验证失败: {e}")
        return False


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
