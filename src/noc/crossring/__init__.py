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
    create_crossring_config_2260e,
    create_crossring_config_2262,
    create_crossring_config_custom,
    load_crossring_config_from_file,
)

from .flit import (
    CrossRingFlit,
    CrossRingFlitPool,
    create_crossring_flit,
    return_crossring_flit,
    get_crossring_flit_pool_stats,
)

from .ip_interface import CrossRingIPInterface

from .model import (
    CrossRingModel,
    create_crossring_model,
)

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
    "CrossRingFlitPool",
    # IP接口
    "CrossRingIPInterface",
    # 主模型
    "CrossRingModel",
    # 便捷函数
    "create_crossring_config_2260e",
    "create_crossring_config_2262",
    "create_crossring_config_custom",
    "load_crossring_config_from_file",
    "create_crossring_flit",
    "return_crossring_flit",
    "get_crossring_flit_pool_stats",
    "create_crossring_model",
]


# 模块级别便捷函数
def quick_start_simulation(config_name: str = "2262", max_cycles: int = 10000, num_test_requests: int = 100) -> dict:
    """
    快速启动CrossRing仿真的便捷函数

    Args:
        config_name: 配置名称 ("2260E", "2262", "custom")
        max_cycles: 最大仿真周期
        num_test_requests: 测试请求数量

    Returns:
        仿真结果字典
    """
    # 创建配置
    if config_name == "2260E":
        config = create_crossring_config_2260e()
    elif config_name == "2262":
        config = create_crossring_config_2262()
    else:
        config = create_crossring_config_custom(5, 4, config_name)

    # 创建模型
    model = create_crossring_model(config.config_name, config.num_row, config.num_col)

    # 注入测试流量
    import random

    for i in range(num_test_requests):
        source = random.randint(0, config.num_nodes - 1)
        destination = random.randint(0, config.num_nodes - 1)
        if source != destination:
            req_type = random.choice(["read", "write"])
            model.inject_test_traffic(source, destination, req_type)

    # 运行仿真
    recommended_cycles = config.get_recommended_simulation_cycles()
    results = model.run_simulation(max_cycles=max_cycles, warmup_cycles=recommended_cycles["warmup_cycles"], stats_start_cycle=recommended_cycles["stats_start_cycle"])

    # 清理
    model.cleanup()

    return results


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
            "时钟域转换（1GHz ↔ 2GHz）",
            "RN/SN资源管理（tracker, databuffer）",
            "请求重试机制",
            "ETag/ITag优先级控制",
            "性能统计和调试支持",
        ],
    }


def validate_installation() -> bool:
    """验证模块安装和依赖"""
    try:
        # 测试基本功能
        config = create_crossring_config_custom(3, 3, "test")
        model = create_crossring_model("test", 3, 3)

        # 测试基本操作
        test_flit = create_crossring_flit(0, 8, [0, 1, 8])
        return_crossring_flit(test_flit)

        # 清理
        model.cleanup()

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
