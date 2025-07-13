"""
NoC可视化模块

提供网络上芯片(NoC)的实时可视化功能，包括：
- 通用Link状态可视化
- 拓扑特定的Node可视化
- 实时动画和交互控制

支持的拓扑类型：
- CrossRing: 环形拓扑
- Mesh: 网格拓扑（未来支持）
- Torus: 环形网格拓扑（未来支持）
"""

from .link_visualizer import BaseLinkVisualizer
from .crossring_node_visualizer import CrossRingNodeVisualizer
from .realtime_visualizer import RealtimeVisualizer

__all__ = [
    'BaseLinkVisualizer',
    'CrossRingNodeVisualizer', 
    'RealtimeVisualizer'
]

__version__ = "1.0.0"