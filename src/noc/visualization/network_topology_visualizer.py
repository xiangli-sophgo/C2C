"""
网络拓扑可视化器

基于旧版本Link_State_Visualizer重新设计，主要显示：
1. CrossRing网络的节点网格
2. 节点间的水平和垂直链路
3. 链路上的slot状态
4. 节点选择和高亮
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, Circle, FancyArrowPatch
from matplotlib.collections import LineCollection, PatchCollection
from typing import Dict, List, Any, Optional, Tuple, Callable
import logging
from dataclasses import dataclass
from enum import Enum

# 中文字体支持
plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False


@dataclass
class NodeInfo:
    """节点信息"""
    node_id: int
    row: int
    col: int
    x: float
    y: float
    is_selected: bool = False


@dataclass
class LinkInfo:
    """链路信息"""
    link_id: str
    src_node: int
    dest_node: int
    direction: str  # 'horizontal' or 'vertical'
    slots: List[Any] = None  # slot数据


class NetworkTopologyVisualizer:
    """
    网络拓扑可视化器
    
    显示CrossRing网络的完整拓扑结构，包括：
    - 节点网格布局
    - 水平和垂直链路
    - 链路上的slot状态
    - 节点选择和高亮功能
    """
    
    def __init__(self, config, ax=None, node_click_callback: Optional[Callable] = None):
        """
        初始化网络拓扑可视化器
        
        Args:
            config: CrossRing配置对象
            ax: matplotlib轴对象
            node_click_callback: 节点点击回调函数
        """
        self.config = config
        self.node_click_callback = node_click_callback
        self.logger = logging.getLogger("NetworkTopologyVis")
        
        # 网络参数
        self.num_rows = getattr(config, 'num_row', getattr(config, 'NUM_ROW', 2))
        self.num_cols = getattr(config, 'num_col', getattr(config, 'NUM_COL', 3))
        self.num_nodes = self.num_rows * self.num_cols
        
        # 图形设置
        if ax is None:
            self.fig, self.ax = plt.subplots(figsize=(14, 10))
        else:
            self.ax = ax
            self.fig = ax.figure
        
        # 布局参数（基于旧版本）
        self.node_size = 0.4
        self.node_spacing_x = 2.0
        self.node_spacing_y = 1.5
        self.link_width = 0.1
        self.slot_size = 0.08
        self.slots_per_link = 8  # 每个链路显示的slot数量
        
        # 颜色配置
        self.node_color = 'lightblue'
        self.selected_node_color = 'orange'
        self.link_color = 'blue'
        self.slot_empty_color = 'white'
        self.slot_occupied_color = 'red'
        
        # 调色板
        self._colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
        
        # 数据存储
        self.node_positions = {}  # node_id -> (x, y)
        self.nodes_info = {}      # node_id -> NodeInfo
        self.links_info = {}      # link_id -> LinkInfo
        self.selected_node = 0    # 当前选中的节点
        
        # 图形元素
        self.node_patches = {}    # node_id -> Rectangle
        self.node_texts = {}      # node_id -> Text
        self.link_lines = {}      # link_id -> Line2D
        self.slot_patches = {}    # (link_id, slot_idx) -> Rectangle
        self.selection_box = None # 选择框
        
        # 高亮控制
        self.tracked_packet_id = None
        self.use_packet_highlight = False
        
        self._setup_layout()
        self._draw_static_elements()
        self._connect_events()
    
    def _setup_layout(self):
        """设置布局"""
        # 计算节点位置
        for row in range(self.num_rows):
            for col in range(self.num_cols):
                node_id = row * self.num_cols + col
                x = col * self.node_spacing_x
                y = (self.num_rows - 1 - row) * self.node_spacing_y  # 翻转Y轴
                
                self.node_positions[node_id] = (x, y)
                self.nodes_info[node_id] = NodeInfo(
                    node_id=node_id, row=row, col=col, x=x, y=y
                )
        
        # 生成链路信息
        self._generate_links()
        
        # 设置坐标轴范围
        margin = 1.0
        min_x = min(pos[0] for pos in self.node_positions.values()) - margin
        max_x = max(pos[0] for pos in self.node_positions.values()) + margin
        min_y = min(pos[1] for pos in self.node_positions.values()) - margin
        max_y = max(pos[1] for pos in self.node_positions.values()) + margin
        
        self.ax.set_xlim(min_x, max_x)
        self.ax.set_ylim(min_y, max_y)
        self.ax.set_aspect('equal')
        self.ax.set_title('CrossRing网络拓扑', fontsize=14, fontweight='bold')
        self.ax.axis('off')  # 隐藏坐标轴
    
    def _generate_links(self):
        """生成链路信息（基于CrossRing拓扑）"""
        # 水平链路
        for row in range(self.num_rows):
            for col in range(self.num_cols - 1):
                src_id = row * self.num_cols + col
                dest_id = row * self.num_cols + col + 1
                link_id = f"h_{src_id}_{dest_id}"
                
                self.links_info[link_id] = LinkInfo(
                    link_id=link_id,
                    src_node=src_id,
                    dest_node=dest_id,
                    direction='horizontal'
                )
        
        # 垂直链路
        for row in range(self.num_rows - 1):
            for col in range(self.num_cols):
                src_id = row * self.num_cols + col
                dest_id = (row + 1) * self.num_cols + col
                link_id = f"v_{src_id}_{dest_id}"
                
                self.links_info[link_id] = LinkInfo(
                    link_id=link_id,
                    src_node=src_id,
                    dest_node=dest_id,
                    direction='vertical'
                )
    
    def _draw_static_elements(self):
        """绘制静态元素"""
        # 绘制链路
        self._draw_links()
        
        # 绘制节点
        self._draw_nodes()
        
        # 绘制选择框
        self._draw_selection_box()
    
    def _draw_links(self):
        """绘制链路和slots"""
        for link_id, link_info in self.links_info.items():
            src_pos = self.node_positions[link_info.src_node]
            dest_pos = self.node_positions[link_info.dest_node]
            
            # 绘制主链路线
            if link_info.direction == 'horizontal':
                # 水平链路：在节点上方和下方各画一条线
                y_offset = 0.15
                line_upper = plt.Line2D(
                    [src_pos[0] + self.node_size/2, dest_pos[0] - self.node_size/2],
                    [src_pos[1] + y_offset, dest_pos[1] + y_offset],
                    color=self.link_color, linewidth=2, alpha=0.7
                )
                line_lower = plt.Line2D(
                    [src_pos[0] + self.node_size/2, dest_pos[0] - self.node_size/2],
                    [src_pos[1] - y_offset, dest_pos[1] - y_offset],
                    color=self.link_color, linewidth=2, alpha=0.7
                )
                self.ax.add_line(line_upper)
                self.ax.add_line(line_lower)
                
                # 绘制slots
                self._draw_link_slots(link_id, src_pos, dest_pos, 'horizontal')
                
            else:  # vertical
                # 垂直链路：在节点左侧和右侧各画一条线
                x_offset = 0.15
                line_left = plt.Line2D(
                    [src_pos[0] - x_offset, dest_pos[0] - x_offset],
                    [src_pos[1] - self.node_size/2, dest_pos[1] + self.node_size/2],
                    color=self.link_color, linewidth=2, alpha=0.7
                )
                line_right = plt.Line2D(
                    [src_pos[0] + x_offset, dest_pos[0] + x_offset],
                    [src_pos[1] - self.node_size/2, dest_pos[1] + self.node_size/2],
                    color=self.link_color, linewidth=2, alpha=0.7
                )
                self.ax.add_line(line_left)
                self.ax.add_line(line_right)
                
                # 绘制slots
                self._draw_link_slots(link_id, src_pos, dest_pos, 'vertical')
    
    def _draw_link_slots(self, link_id: str, src_pos: Tuple[float, float], 
                        dest_pos: Tuple[float, float], direction: str):
        """绘制链路上的slots"""
        # 计算slot位置
        if direction == 'horizontal':
            start_x = src_pos[0] + self.node_size/2 + 0.1
            end_x = dest_pos[0] - self.node_size/2 - 0.1
            slot_positions = np.linspace(start_x, end_x, self.slots_per_link)
            
            for i, x in enumerate(slot_positions):
                # 上方slots
                slot_rect_upper = Rectangle(
                    (x - self.slot_size/2, src_pos[1] + 0.05),
                    self.slot_size, self.slot_size,
                    facecolor=self.slot_empty_color,
                    edgecolor='black', linewidth=0.5
                )
                self.ax.add_patch(slot_rect_upper)
                self.slot_patches[(link_id, f"upper_{i}")] = slot_rect_upper
                
                # 下方slots
                slot_rect_lower = Rectangle(
                    (x - self.slot_size/2, src_pos[1] - 0.15),
                    self.slot_size, self.slot_size,
                    facecolor=self.slot_empty_color,
                    edgecolor='black', linewidth=0.5
                )
                self.ax.add_patch(slot_rect_lower)
                self.slot_patches[(link_id, f"lower_{i}")] = slot_rect_lower
                
        else:  # vertical
            start_y = src_pos[1] - self.node_size/2 - 0.1
            end_y = dest_pos[1] + self.node_size/2 + 0.1
            slot_positions = np.linspace(start_y, end_y, self.slots_per_link)
            
            for i, y in enumerate(slot_positions):
                # 左侧slots
                slot_rect_left = Rectangle(
                    (src_pos[0] - 0.25, y - self.slot_size/2),
                    self.slot_size, self.slot_size,
                    facecolor=self.slot_empty_color,
                    edgecolor='black', linewidth=0.5
                )
                self.ax.add_patch(slot_rect_left)
                self.slot_patches[(link_id, f"left_{i}")] = slot_rect_left
                
                # 右侧slots
                slot_rect_right = Rectangle(
                    (src_pos[0] + 0.05, y - self.slot_size/2),
                    self.slot_size, self.slot_size,
                    facecolor=self.slot_empty_color,
                    edgecolor='black', linewidth=0.5
                )
                self.ax.add_patch(slot_rect_right)
                self.slot_patches[(link_id, f"right_{i}")] = slot_rect_right
    
    def _draw_nodes(self):
        """绘制节点"""
        for node_id, node_info in self.nodes_info.items():
            # 节点矩形
            node_rect = Rectangle(
                (node_info.x - self.node_size/2, node_info.y - self.node_size/2),
                self.node_size, self.node_size,
                facecolor=self.node_color,
                edgecolor='black', linewidth=1
            )
            self.ax.add_patch(node_rect)
            self.node_patches[node_id] = node_rect
            
            # 节点编号文本
            node_text = self.ax.text(
                node_info.x, node_info.y, str(node_id),
                fontsize=10, weight='bold',
                ha='center', va='center'
            )
            self.node_texts[node_id] = node_text
    
    def _draw_selection_box(self):
        """绘制选择框"""
        if self.selected_node in self.nodes_info:
            node_info = self.nodes_info[self.selected_node]
            self.selection_box = Rectangle(
                (node_info.x - self.node_size/2 - 0.1, node_info.y - self.node_size/2 - 0.1),
                self.node_size + 0.2, self.node_size + 0.2,
                facecolor='none', edgecolor='red', 
                linewidth=2, linestyle='--'
            )
            self.ax.add_patch(self.selection_box)
    
    def _connect_events(self):
        """连接事件"""
        self.fig.canvas.mpl_connect('button_press_event', self._on_click)
    
    def _on_click(self, event):
        """处理鼠标点击事件"""
        if event.inaxes != self.ax:
            return
        
        # 检查是否点击了节点
        for node_id, node_info in self.nodes_info.items():
            dx = event.xdata - node_info.x
            dy = event.ydata - node_info.y
            distance = np.sqrt(dx*dx + dy*dy)
            
            if distance <= self.node_size/2:
                self._select_node(node_id)
                if self.node_click_callback:
                    self.node_click_callback(node_id)
                break
    
    def _select_node(self, node_id: int):
        """选择节点"""
        if node_id == self.selected_node:
            return
        
        self.selected_node = node_id
        
        # 更新选择框位置
        if self.selection_box:
            self.selection_box.remove()
        
        self._draw_selection_box()
        self.fig.canvas.draw_idle()
        
        self.logger.info(f"选中节点: {node_id}")
    
    def update_network_state(self, network_data: dict):
        """
        更新网络状态
        
        Args:
            network_data: 网络状态数据
                格式: {
                    'links': {
                        link_id: {
                            'slots': [slot_data, ...],
                            'direction': 'req'/'rsp'/'data'
                        }
                    },
                    'nodes': {
                        node_id: node_state_data
                    }
                }
        """
        # 更新链路slot状态
        links_data = network_data.get('links', {})
        for link_id, link_data in links_data.items():
            if link_id in self.links_info:
                self._update_link_slots(link_id, link_data)
        
        # 更新节点状态（如有需要）
        nodes_data = network_data.get('nodes', {})
        for node_id, node_data in nodes_data.items():
            if node_id in self.nodes_info:
                self._update_node_state(node_id, node_data)
    
    def _update_link_slots(self, link_id: str, link_data: dict):
        """更新链路slot状态"""
        slots_data = link_data.get('slots', [])
        direction = link_data.get('direction', 'req')
        
        # 找到该链路的所有slots
        link_slots = [(key, patch) for key, patch in self.slot_patches.items() 
                     if key[0] == link_id]
        
        for i, (slot_key, slot_patch) in enumerate(link_slots):
            if i < len(slots_data):
                slot_data = slots_data[i]
                # 根据slot数据设置颜色
                if hasattr(slot_data, 'valid') and slot_data.valid:
                    if hasattr(slot_data, 'packet_id') and self.tracked_packet_id:
                        if str(slot_data.packet_id) == str(self.tracked_packet_id):
                            color = 'red'  # 高亮tracked packet
                        else:
                            color = 'lightgray' if self.use_packet_highlight else self._get_packet_color(slot_data.packet_id)
                    else:
                        color = self._get_packet_color(getattr(slot_data, 'packet_id', 0))
                else:
                    color = self.slot_empty_color
                    
                slot_patch.set_facecolor(color)
            else:
                slot_patch.set_facecolor(self.slot_empty_color)
    
    def _update_node_state(self, node_id: int, node_data: dict):
        """更新节点状态"""
        # 可以根据需要更新节点的颜色或其他属性
        pass
    
    def _get_packet_color(self, packet_id):
        """根据packet_id获取颜色"""
        try:
            pid = int(packet_id) if isinstance(packet_id, str) and packet_id.isdigit() else hash(str(packet_id))
            return self._colors[abs(pid) % len(self._colors)]
        except:
            return 'lightblue'
    
    def set_packet_highlight(self, packet_id: Optional[str], enable: bool = True):
        """设置包高亮"""
        self.tracked_packet_id = packet_id
        self.use_packet_highlight = enable
        self.logger.info(f"包高亮: {'开启' if enable else '关闭'}, 包ID: {packet_id}")
    
    def get_selected_node(self) -> int:
        """获取当前选中的节点"""
        return self.selected_node
    
    def clear_highlight(self):
        """清除高亮"""
        self.set_packet_highlight(None, False)
        # 重置所有slot颜色
        for slot_patch in self.slot_patches.values():
            slot_patch.set_facecolor(self.slot_empty_color)
        self.fig.canvas.draw_idle()


# 工具函数
def create_demo_network_data(num_nodes: int = 6) -> dict:
    """创建演示用的网络数据"""
    import random
    
    network_data = {
        'links': {},
        'nodes': {}
    }
    
    # 创建一些链路数据
    for i in range(num_nodes - 1):
        link_id = f"h_{i}_{i+1}"
        slots_data = []
        for j in range(8):
            if random.random() < 0.3:  # 30%概率有数据
                from types import SimpleNamespace
                slot = SimpleNamespace()
                slot.valid = True
                slot.packet_id = random.randint(1, 4)
                slot.flit_id = f"F{j}"
                slots_data.append(slot)
            else:
                slot = SimpleNamespace()
                slot.valid = False
                slots_data.append(slot)
        
        network_data['links'][link_id] = {
            'slots': slots_data,
            'direction': 'req'
        }
    
    return network_data


if __name__ == "__main__":
    # 简单测试
    from types import SimpleNamespace
    
    # 创建配置
    config = SimpleNamespace(num_row=3, num_col=4)
    
    # 创建可视化器
    def on_node_click(node_id):
        print(f"点击了节点: {node_id}")
    
    visualizer = NetworkTopologyVisualizer(config, node_click_callback=on_node_click)
    
    # 创建测试数据并更新
    test_data = create_demo_network_data(12)
    visualizer.update_network_state(test_data)
    
    plt.show()