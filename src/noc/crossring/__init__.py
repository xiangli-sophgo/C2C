"""
CrossRing NoC模块。

基于C2C仓库架构重新实现的CrossRing NoC支持，
包含STI三通道协议、资源管理和时钟域转换。
"""

from .config import (
    CrossRingConfig,
    BasicConfiguration,
    IPConfiguration,
    FIFOConfiguration,
    TagConfiguration,
    TrackerConfiguration,
    LatencyConfiguration,
)

from .flit import (
    CrossRingFlit,
    create_crossring_flit,
    return_crossring_flit,
    get_crossring_flit_pool_stats,
)

from .ip_interface import CrossRingIPInterface

from .model import (
    CrossRingModel,
    create_crossring_model,
)

# 导入现有的组件
from .link import CrossRingSlot, RingSlice
from .cross_point import CrossRingTagManager, CrossRingCrossPoint, CrossPointDirection

# 版本信息
__version__ = "1.0.0"
__author__ = "C2C CrossRing Team"

# 主要类导出
__all__ = [
    # 配置类
    "CrossRingConfig",
    "BasicConfiguration",
    "IPConfiguration",
    "FIFOConfiguration",
    "TagConfiguration",
    "TrackerConfiguration",
    "LatencyConfiguration",
    # Flit相关
    "CrossRingFlit",
    # IP接口
    "CrossRingIPInterface",
    # 主模型
    "CrossRingModel",
    # CrossRing组件
    "CrossRingSlot",
    "RingSlice", 
    "CrossRingCrossPoint",
    "CrossRingTagManager",
    # 便捷函数
    "create_crossring_flit",
    "return_crossring_flit",
    "get_crossring_flit_pool_stats",
    "create_crossring_model",
]


# 模块级别便捷函数
def quick_start_simulation(config_name: str = "test", max_cycles: int = 10000, num_test_requests: int = 100) -> dict:
    """
    快速启动CrossRing仿真的便捷函数

    Args:
        config_name: 配置名称
        max_cycles: 最大仿真周期
        num_test_requests: 测试请求数量

    Returns:
        仿真结果字典
    """
    # 创建配置
    config = CrossRingConfig(num_row=3, num_col=3, config_name=config_name)

    # 创建模型
    model = CrossRingModel(config)
    
    # 简化的测试结果
    return {
        "config": config.config_name,
        "topology": f"{config.NUM_ROW}x{config.NUM_COL}",
        "status": "架构重构完成"
    }


def get_module_info() -> dict:
    """获取模块信息"""
    return {
        "name": "CrossRing NoC",
        "version": __version__,
        "author": __author__,
        "description": "基于C2C架构的CrossRing NoC实现",
        "components": {
            "config": "CrossRing专用配置管理",
            "flit": "STI三通道协议Flit实现",
            "ip_interface": "IP接口和资源管理",
            "model": "CrossRing主仿真模型",
        },
        "features": [
            "STI三通道协议（REQ/RSP/DAT）",
            "真实环形拓扑（带环绕连接）",
            "四方向系统（TL/TR/TU/TD）",
            "环形桥接和交叉点模块",
            "维度转换（水平↔垂直）",
            "时钟域转换（1GHz ↔ 2GHz）",
            "RN/SN资源管理（tracker, databuffer）",
            "请求重试机制",
            "ETag/ITag优先级控制",
            "XY维度顺序路由",
            "性能统计和调试支持",
        ],
    }


def validate_installation() -> bool:
    """验证模块安装和依赖"""
    try:
        # 测试基本功能
        config = CrossRingConfig(num_row=3, num_col=3, config_name="test")
        model = CrossRingModel(config)

        # 测试基本操作
        test_flit = create_crossring_flit(0, 8, [0, 1, 8])
        return_crossring_flit(test_flit)

        return True

    except Exception as e:
        print(f"CrossRing模块验证失败: {e}")
        return False


# 模块初始化时的检查
def _check_dependencies():
    """检查依赖项"""
    try:
        import numpy
        import logging
        from collections import deque, defaultdict
        from dataclasses import dataclass, field
        from typing import Dict, List, Any, Optional, Tuple
    except ImportError as e:
        raise ImportError(f"CrossRing模块缺少必要依赖: {e}")


# 执行依赖检查
_check_dependencies()

# 模块级日志
import logging

_logger = logging.getLogger(__name__)
_logger.info(f"CrossRing NoC模块加载完成 (v{__version__})")
