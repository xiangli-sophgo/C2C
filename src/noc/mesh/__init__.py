"""
Mesh拓扑NoC实现。

基于通用NoC基类实现的Mesh拓扑，支持：
- XY路由算法
- 最短路径路由
- 虚拟通道
- 性能监控
"""

import logging

# 导入核心组件
from .config import MeshConfig, MeshConfiguration
from .topology import MeshTopology
from .model import MeshModel, MeshPacket, MeshRouter

# 便捷函数
from .config import (
    create_mesh_config_2x2,
    create_mesh_config_4x4,
    create_mesh_config_8x8,
    create_mesh_config_custom,
)

__version__ = "1.0.0"
__author__ = "C2C Mesh Team"

__all__ = [
    # 核心类
    "MeshConfig",
    "MeshConfiguration",
    "MeshTopology",
    "MeshModel",
    "MeshPacket",
    "MeshRouter",
    # 便捷函数
    "create_mesh_config_2x2",
    "create_mesh_config_4x4",
    "create_mesh_config_8x8",
    "create_mesh_config_custom",
]

# 设置日志
logger = logging.getLogger(__name__)
logger.info(f"Mesh NoC模块加载完成 (v{__version__})")

# 模块初始化信息
print(f"Mesh NoC模块已加载 - 版本 {__version__}")
print("支持的功能:")
print("  - 2D Mesh拓扑")
print("  - XY路由算法")
print("  - 最短路径路由")
print("  - 虚拟通道")
print("  - 性能监控")
