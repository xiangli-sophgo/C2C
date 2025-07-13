"""
通用Link状态可视化器

提供所有拓扑类型共用的链路可视化功能，包括：
- Slot状态渲染
- 数据流动画
- 拥塞状态指示
- 性能热力图
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, Circle
from matplotlib.collections import LineCollection
from matplotlib.animation import FuncAnimation
from typing import Dict, List, Any, Optional, Tuple, Union
import logging
from abc import ABC, abstractmethod
from enum import Enum
from dataclasses import dataclass
import time

# 中文字体支持
plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False


class SlotState(Enum):
    """Slot状态枚举"""
    EMPTY = "empty"          # 空闲
    OCCUPIED = "occupied"    # 被占用
    RESERVED = "reserved"    # 被预留(I-Tag)
    PRIORITY = "priority"    # 高优先级(E-Tag)


class FlowDirection(Enum):
    """数据流方向枚举"""
    FORWARD = "forward"   # 正向
    BACKWARD = "backward" # 反向
    BIDIRECTIONAL = "bidirectional"  # 双向


@dataclass
class SlotData:
    """Slot数据结构"""
    slot_id: int
    cycle: int
    state: SlotState
    flit_id: Optional[str] = None
    packet_id: Optional[str] = None
    priority: str = "T2"  # T0/T1/T2
    itag: bool = False
    etag: bool = False
    valid: bool = False
    

@dataclass
class LinkStats:
    """链路统计数据"""
    bandwidth_utilization: float = 0.0
    average_latency: float = 0.0
    congestion_level: float = 0.0
    itag_triggers: int = 0
    etag_upgrades: int = 0
    total_flits: int = 0


class BaseLinkVisualizer:
    """
    通用Link状态可视化器
    
    适用于所有拓扑类型的链路可视化，包括：
    - Slot占用状态显示
    - 数据流动画
    - 拥塞控制状态
    - 性能监控
    """

    def __init__(self, ax=None, link_id: str = "link_0", num_slots: int = 8):
        """
        初始化Link可视化器
        
        Args:
            ax: matplotlib轴对象，如果为None则创建新的
            link_id: 链路标识符
            num_slots: slot数量
        """
        self.link_id = link_id
        self.num_slots = num_slots
        self.logger = logging.getLogger(f"LinkVis_{link_id}")
        
        # 图形设置
        if ax is None:
            self.fig, self.ax = plt.subplots(figsize=(12, 6))
        else:
            self.ax = ax
            self.fig = ax.figure
        
        # 可视化参数
        self.slot_width = 0.8
        self.slot_height = 0.4
        self.slot_spacing = 1.0
        self.channel_spacing = 1.2
        
        # 颜色配置
        self.colors = {
            SlotState.EMPTY: 'lightgray',
            SlotState.OCCUPIED: 'lightblue', 
            SlotState.RESERVED: 'yellow',
            SlotState.PRIORITY: 'orange'
        }
        
        self.priority_colors = {
            'T0': 'red',
            'T1': 'orange', 
            'T2': 'lightblue'
        }
        
        # 存储图形元素
        self.slot_patches = {}  # 存储slot矩形
        self.slot_texts = {}    # 存储slot文本
        self.flow_arrows = []   # 存储流向箭头
        self.stats_text = None  # 统计信息文本
        
        # 性能数据
        self.history_stats = []  # 历史统计数据
        self.max_history = 100   # 最大历史记录数
        
        self._init_layout()
        
    def _init_layout(self):
        """初始化布局"""
        self.ax.set_xlim(-0.5, self.num_slots * self.slot_spacing + 0.5)
        self.ax.set_ylim(-2, 4)
        self.ax.set_aspect('equal')
        self.ax.set_title(f'链路状态 - {self.link_id}', fontsize=14, pad=20)
        
        # 绘制通道标签
        channels = ['req', 'rsp', 'data']
        for i, channel in enumerate(channels):
            y_pos = i * self.channel_spacing
            self.ax.text(-0.3, y_pos, channel, fontsize=10, 
                        ha='right', va='center', weight='bold')
        
        # 绘制slot编号
        for i in range(self.num_slots):
            x_pos = i * self.slot_spacing
            self.ax.text(x_pos, -1.5, f'S{i}', fontsize=8,
                        ha='center', va='center')
        
        # 初始化slot patches
        self._create_slot_patches()
        
        # 添加图例
        self._create_legend()
        
        # 添加统计信息显示区域
        self.stats_text = self.ax.text(0.02, 0.98, '', 
                                      transform=self.ax.transAxes,
                                      fontsize=9, va='top', ha='left',
                                      bbox=dict(boxstyle='round', 
                                               facecolor='white', alpha=0.8))
        
    def _create_slot_patches(self):
        """创建slot矩形patches"""
        channels = ['req', 'rsp', 'data']
        
        for ch_idx, channel in enumerate(channels):
            y_pos = ch_idx * self.channel_spacing
            self.slot_patches[channel] = []
            self.slot_texts[channel] = []
            
            for slot_idx in range(self.num_slots):
                x_pos = slot_idx * self.slot_spacing
                
                # 创建矩形patch
                rect = Rectangle((x_pos - self.slot_width/2, y_pos - self.slot_height/2),
                               self.slot_width, self.slot_height,
                               facecolor=self.colors[SlotState.EMPTY],
                               edgecolor='black', linewidth=1)
                self.ax.add_patch(rect)
                self.slot_patches[channel].append(rect)
                
                # 创建文本
                text = self.ax.text(x_pos, y_pos, '', fontsize=7,
                                  ha='center', va='center', weight='bold')
                self.slot_texts[channel].append(text)
    
    def _create_legend(self):
        """创建图例"""
        legend_elements = []
        
        # Slot状态图例
        for state, color in self.colors.items():
            legend_elements.append(Rectangle((0, 0), 1, 1, facecolor=color, 
                                           label=f'{state.value}'))
        
        # 优先级图例  
        for priority, color in self.priority_colors.items():
            legend_elements.append(Rectangle((0, 0), 1, 1, facecolor=color,
                                           label=f'优先级{priority}'))
        
        self.ax.legend(handles=legend_elements, loc='upper right', 
                      bbox_to_anchor=(1.15, 1))
    
    def update_slots(self, slots_data: Dict[str, List[SlotData]]):
        """
        更新slot状态显示
        
        Args:
            slots_data: 按通道组织的slot数据 
                       格式: {'req': [SlotData, ...], 'rsp': [...], 'data': [...]}
        """
        for channel, slot_list in slots_data.items():
            if channel not in self.slot_patches:
                continue
                
            patches = self.slot_patches[channel]
            texts = self.slot_texts[channel]
            
            for i, slot_data in enumerate(slot_list):
                if i >= len(patches):
                    break
                    
                patch = patches[i]
                text = texts[i]
                
                # 设置颜色
                if slot_data.valid and slot_data.flit_id:
                    # 有有效flit，根据优先级着色
                    color = self.priority_colors.get(slot_data.priority, 'lightblue')
                    
                    # 如果有Tag标记，添加特殊效果
                    if slot_data.etag:
                        patch.set_edgecolor('red')
                        patch.set_linewidth(3)
                    elif slot_data.itag:
                        patch.set_edgecolor('yellow')
                        patch.set_linewidth(2)
                    else:
                        patch.set_edgecolor('black')
                        patch.set_linewidth(1)
                        
                    # 设置文本
                    display_text = f"{slot_data.packet_id}\n{slot_data.flit_id}" if slot_data.packet_id else slot_data.flit_id
                    text.set_text(display_text)
                    text.set_visible(True)
                else:
                    # 空slot
                    color = self.colors[SlotState.EMPTY]
                    patch.set_edgecolor('black')
                    patch.set_linewidth(1)
                    text.set_visible(False)
                
                patch.set_facecolor(color)
    
    def add_flow_animation(self, channel: str, start_slot: int, end_slot: int, 
                          packet_id: str, duration: float = 1.0):
        """
        添加数据流动画
        
        Args:
            channel: 通道名称
            start_slot: 起始slot
            end_slot: 结束slot  
            packet_id: 包ID
            duration: 动画持续时间(秒)
        """
        if channel not in self.slot_patches:
            return
        
        ch_idx = ['req', 'rsp', 'data'].index(channel)
        y_pos = ch_idx * self.channel_spacing
        
        start_x = start_slot * self.slot_spacing
        end_x = end_slot * self.slot_spacing
        
        # 创建移动的圆点表示flit
        circle = Circle((start_x, y_pos), 0.1, color='red', alpha=0.8, zorder=10)
        self.ax.add_patch(circle)
        
        # 创建动画函数
        def animate_flow(frame):
            progress = frame / (duration * 30)  # 假设30fps
            if progress <= 1.0:
                current_x = start_x + (end_x - start_x) * progress
                circle.set_center((current_x, y_pos))
            else:
                circle.remove()
                
        # 这里可以集成到主动画循环中
        # 或者使用matplotlib的FuncAnimation
    
    def update_statistics(self, stats: LinkStats):
        """
        更新链路统计信息
        
        Args:
            stats: 链路统计数据
        """
        # 添加到历史记录
        self.history_stats.append(stats)
        if len(self.history_stats) > self.max_history:
            self.history_stats.pop(0)
        
        # 更新显示文本
        stats_text = f"""链路统计 - {self.link_id}
带宽利用率: {stats.bandwidth_utilization:.1%}
平均延迟: {stats.average_latency:.1f} cycles
拥塞级别: {stats.congestion_level:.1%}
I-Tag触发: {stats.itag_triggers}
E-Tag升级: {stats.etag_upgrades}
总flit数: {stats.total_flits}"""
        
        self.stats_text.set_text(stats_text)
    
    def render_congestion_heatmap(self, congestion_data: Dict[str, float]):
        """
        渲染拥塞热力图
        
        Args:
            congestion_data: 按通道的拥塞数据
        """
        channels = ['req', 'rsp', 'data']
        
        for ch_idx, channel in enumerate(channels):
            congestion_level = congestion_data.get(channel, 0.0)
            y_pos = ch_idx * self.channel_spacing
            
            # 在通道旁边绘制拥塞指示条
            bar_width = 0.1
            bar_height = congestion_level * 0.5  # 最大高度0.5
            
            bar_rect = Rectangle(
                (self.num_slots * self.slot_spacing + 0.2, y_pos - bar_height/2),
                bar_width, bar_height,
                facecolor=plt.cm.Reds(congestion_level),  # 使用红色渐变
                edgecolor='black', linewidth=1
            )
            
            # 移除旧的拥塞指示条
            for patch in self.ax.patches:
                if hasattr(patch, '_congestion_bar') and patch._congestion_bar:
                    patch.remove()
            
            # 添加新的指示条
            bar_rect._congestion_bar = True
            self.ax.add_patch(bar_rect)
    
    def highlight_packet(self, packet_id: str, highlight_color: str = 'red'):
        """
        高亮显示特定包的所有flit
        
        Args:
            packet_id: 要高亮的包ID
            highlight_color: 高亮颜色
        """
        # 这个功能需要与上层的数据管理配合
        # 暂时提供接口，具体实现在realtime_visualizer中
        pass
    
    def clear_highlights(self):
        """清除所有高亮显示"""
        for channel in self.slot_patches:
            for patch in self.slot_patches[channel]:
                patch.set_edgecolor('black')
                patch.set_linewidth(1)
    
    def get_performance_trend(self) -> Dict[str, List[float]]:
        """
        获取性能趋势数据
        
        Returns:
            包含各项性能指标历史数据的字典
        """
        if not self.history_stats:
            return {}
        
        return {
            'bandwidth_utilization': [s.bandwidth_utilization for s in self.history_stats],
            'average_latency': [s.average_latency for s in self.history_stats],
            'congestion_level': [s.congestion_level for s in self.history_stats],
            'total_flits': [s.total_flits for s in self.history_stats]
        }
    
    def save_snapshot(self, filename: str):
        """
        保存当前状态快照
        
        Args:
            filename: 保存文件名
        """
        self.fig.savefig(filename, dpi=300, bbox_inches='tight')
        self.logger.info(f"链路状态快照已保存: {filename}")


# 工具函数
def create_demo_slot_data(num_slots: int = 8) -> Dict[str, List[SlotData]]:
    """创建演示用的slot数据"""
    import random
    
    channels = ['req', 'rsp', 'data']
    slots_data = {}
    
    for channel in channels:
        slot_list = []
        for i in range(num_slots):
            # 随机生成一些占用的slot
            if random.random() < 0.3:  # 30%概率被占用
                slot_data = SlotData(
                    slot_id=i,
                    cycle=0,
                    state=SlotState.OCCUPIED,
                    flit_id=f"F{i}",
                    packet_id=f"P{random.randint(1,5)}",
                    priority=random.choice(['T0', 'T1', 'T2']),
                    valid=True,
                    itag=random.random() < 0.1,  # 10%概率有I-Tag
                    etag=random.random() < 0.05  # 5%概率有E-Tag
                )
            else:
                slot_data = SlotData(
                    slot_id=i,
                    cycle=0,
                    state=SlotState.EMPTY
                )
            slot_list.append(slot_data)
        slots_data[channel] = slot_list
    
    return slots_data


if __name__ == "__main__":
    # 简单测试
    visualizer = BaseLinkVisualizer(link_id="test_link", num_slots=8)
    
    # 创建测试数据
    test_data = create_demo_slot_data(8)
    visualizer.update_slots(test_data)
    
    # 更新统计信息
    test_stats = LinkStats(
        bandwidth_utilization=0.65,
        average_latency=12.5,
        congestion_level=0.2,
        itag_triggers=3,
        etag_upgrades=1,
        total_flits=128
    )
    visualizer.update_statistics(test_stats)
    
    plt.show()