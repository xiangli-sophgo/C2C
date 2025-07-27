"""
CrossRing节点可视化器

基于旧版本Link_State_Visualizer的PieceVisualizer功能，
专门用于CrossRing拓扑的节点内部结构可视化，包括：
- Inject/Eject队列
- Ring Bridge FIFO  
- CrossPoint状态
- Tag机制显示
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, FancyArrowPatch, Circle
from matplotlib.collections import PatchCollection
from typing import Dict, List, Any, Optional, Tuple, Union, Callable
import logging
from dataclasses import dataclass, field
from enum import Enum
import copy

# 中文字体支持
plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False


class FIFOType(Enum):
    """FIFO类型枚举"""
    INJECT_QUEUE = "inject_queue"
    EJECT_QUEUE = "eject_queue"
    RING_BRIDGE = "ring_bridge"
    CROSSPOINT = "crosspoint"
    CHANNEL_BUFFER = "channel_buffer"


@dataclass
class FlitProxy:
    """
    轻量级flit代理类，用于快照渲染
    基于旧版本的_FlitProxy改进
    """
    packet_id: str
    flit_id: str
    etag_priority: str = "T2"  # T0/T1/T2
    itag_h: bool = False       # 水平I-Tag
    itag_v: bool = False       # 垂直I-Tag
    valid: bool = True
    
    def __repr__(self):
        itag = "H" if self.itag_h else ("V" if self.itag_v else "")
        return f"(pid={self.packet_id}, fid={self.flit_id}, ET={self.etag_priority}, IT={itag})"


@dataclass 
class FIFOData:
    """FIFO数据结构"""
    fifo_id: str
    fifo_type: FIFOType
    depth: int
    current_size: int
    flits: List[FlitProxy] = field(default_factory=list)
    orientation: str = "vertical"  # vertical/horizontal
    position: str = "center"       # top/mid/bottom for vertical, left/mid/right for horizontal
    

@dataclass
class CrossPointData:
    """CrossPoint数据结构"""
    cp_id: str
    direction: str  # horizontal/vertical
    slice_data: Dict[str, List[FlitProxy]] = field(default_factory=dict)  # slice_name -> flits
    arbitration_state: str = "idle"
    active_connections: List[Tuple[str, str]] = field(default_factory=list)  # (input, output) pairs


class CrossRingNodeVisualizer:
    """
    CrossRing节点可视化器
    
    基于旧版本PieceVisualizer的功能，专门用于CrossRing节点的内部结构可视化
    """
    
    def __init__(self, config, ax=None, node_id: int = 0, highlight_callback: Optional[Callable] = None, parent_visualizer=None):
        """
        初始化CrossRing节点可视化器
        
        Args:
            config: CrossRing配置对象
            ax: matplotlib轴对象
            node_id: 节点ID
            highlight_callback: 高亮回调函数
            parent_visualizer: 父级link_state_visualizer，用于颜色同步
        """
        self.config = config
        self.node_id = node_id
        self.highlight_callback = highlight_callback
        self.parent_visualizer = parent_visualizer
        self.logger = logging.getLogger(f"CrossRingNodeVis_{node_id}")
        
        # 提取配置参数
        self.cols = getattr(config, 'NUM_COL', 3)
        self.rows = getattr(config, 'NUM_ROW', 2)
        self.iq_depth = getattr(config, 'IQ_OUT_FIFO_DEPTH', 8)
        self.eq_depth = getattr(config, 'EQ_IN_FIFO_DEPTH', 8)
        self.rb_in_depth = getattr(config, 'RB_IN_FIFO_DEPTH', 4)
        self.rb_out_depth = getattr(config, 'RB_OUT_FIFO_DEPTH', 4)
        self.iq_ch_depth = getattr(config, 'IQ_CH_FIFO_DEPTH', 4)
        self.eq_ch_depth = getattr(config, 'EQ_CH_FIFO_DEPTH', 4)
        
        # 获取通道名称列表
        if hasattr(config, 'CH_NAME_LIST'):
            self.ch_names = config.CH_NAME_LIST
        else:
            # 默认通道名称
            self.ch_names = ['gdma', 'ddr', 'l2m']
        
        # 图形设置
        if ax is None:
            self.fig, self.ax = plt.subplots(figsize=(12, 10))
        else:
            self.ax = ax
            self.fig = ax.figure
        
        self.ax.axis("off")
        self.ax.set_aspect("equal")
        
        # 几何参数（压缩版本以适应右侧面板）
        self.square = 0.15     # flit方块边长（缩小）
        self.gap = 0.01        # 相邻槽间距（缩小）
        self.fifo_gap = 0.4    # FIFO间隙（缩小）
        self.fontsize = 6      # 字体缩小
        self.gap_lr = 0.2      # 左右内边距（缩小）
        self.gap_hv = 0.2      # 上下内边距（缩小）
        self.text_gap = 0.05   # 文本间距（缩小）
        self.slot_frame_lw = 0.3
        
        # 默认模块尺寸（会根据FIFO配置动态计算）
        self.inject_module_size = (8, 5)   # (height, width)
        self.eject_module_size = (5, 8)    # (width, height) 
        self.rb_module_size = (8, 8)       # (height, width)
        self.cp_module_size = (2, 5)       # (height, width)
        
        # 调色板（优先使用父级可视化器的颜色）
        if parent_visualizer and hasattr(parent_visualizer, '_colors'):
            self._colors = parent_visualizer._colors
        else:
            self._colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
        
        # 高亮设置
        self.use_highlight = False
        self.highlight_pid = None
        self.highlight_color = "red"
        self.grey_color = "lightgrey"
        
        # 存储图形元素
        self.iq_patches, self.iq_texts = {}, {}
        self.eq_patches, self.eq_texts = {}, {}
        self.rb_patches, self.rb_texts = {}, {}
        self.cph_patches, self.cph_texts = {}, {}  # 水平CrossPoint
        self.cpv_patches, self.cpv_texts = {}, {}  # 垂直CrossPoint
        
        # 点击信息映射
        self.patch_info_map = {}  # patch -> (text_obj, flit_info)
        self.current_highlight_flit = None
        
        # 绘制模块
        self._draw_modules()
        
        # 设置坐标轴范围以确保所有内容都能显示
        self.ax.set_xlim(-4, 3)
        self.ax.set_ylim(-3, 4)
        
        # 连接点击事件
        self.fig.canvas.mpl_connect("button_press_event", self._on_click)
        
        # 信息显示框
        self.info_text = self.fig.text(0.75, 0.02, "", fontsize=12, 
                                      va="bottom", ha="left", wrap=True)
    
    def _draw_modules(self):
        """绘制所有模块（基于旧版本的模块配置）"""
        
        # 根据实际CrossRing架构配置FIFO连接
        # Inject Queue: IP channel buffers + 方向输出FIFO (TR,TL,TU,TD,EQ)
        iq_config = {
            "title": "Inject Queue",
            "lanes": [f"{ch}_ch" for ch in self.ch_names] + ["IQ_TR", "IQ_TL", "IQ_TU", "IQ_TD", "IQ_EQ"],
            "depths": [self.iq_ch_depth] * len(self.ch_names) + [self.iq_depth] * 5,
            "orientations": ["vertical"] * len(self.ch_names) + ["horizontal", "horizontal", "vertical", "vertical", "horizontal"],
            "h_pos": ["top"] * len(self.ch_names) + ["bottom", "bottom", "mid", "mid", "top"],
            "v_pos": ["left"] * len(self.ch_names) + ["left", "right", "left", "right", "mid"],
            "patch_dict": self.iq_patches,
            "text_dict": self.iq_texts,
        }
        
        # Eject Queue: 方向输入FIFO (TU,TD,TR,TL) + IP channel buffers  
        eq_config = {
            "title": "Eject Queue", 
            "lanes": ["EQ_TU", "EQ_TD", "EQ_TR", "EQ_TL"] + [f"{ch}_ch" for ch in self.ch_names],
            "depths": [self.eq_depth] * 4 + [self.eq_ch_depth] * len(self.ch_names),
            "orientations": ["vertical", "vertical", "horizontal", "horizontal"] + ["horizontal"] * len(self.ch_names),
            "h_pos": ["mid", "mid", "top", "bottom"] + ["mid"] * len(self.ch_names),
            "v_pos": ["left", "right", "mid", "mid"] + ["top"] * len(self.ch_names),
            "patch_dict": self.eq_patches,
            "text_dict": self.eq_texts,
        }
        
        # Ring Bridge: 输入FIFO (TR,TL,TU,TD) + 输出FIFO (EQ,TR,TL,TU,TD)
        rb_config = {
            "title": "Ring Bridge",
            "lanes": ["RB_TR_in", "RB_TL_in", "RB_TU_in", "RB_TD_in", "RB_EQ_out", "RB_TR_out", "RB_TL_out", "RB_TU_out", "RB_TD_out"],
            "depths": [self.rb_in_depth] * 4 + [self.rb_out_depth] * 5,
            "orientations": ["horizontal", "horizontal", "vertical", "vertical", "horizontal", "horizontal", "horizontal", "vertical", "vertical"],
            "h_pos": ["top", "bottom", "mid", "mid", "mid", "top", "bottom", "mid", "mid"],
            "v_pos": ["mid", "mid", "left", "right", "left", "left", "left", "right", "right"],
            "patch_dict": self.rb_patches,
            "text_dict": self.rb_texts,
        }
        
        # 绘制三个主要模块（调整间距使布局更紧凑）
        self._draw_module(-2.5, 0.0, self.inject_module_size, iq_config)
        self._draw_module(0.0, 2.5, self.eject_module_size, eq_config)
        self._draw_module(0.0, 0.0, self.rb_module_size, rb_config)
        
        # 绘制CrossPoint（调整位置）
        self._draw_crosspoints()
    
    def _draw_module(self, base_x: float, base_y: float, module_size: Tuple[float, float], config: dict):
        """绘制单个模块（基于src_old的精确模块尺寸计算）"""
        # 临时存储当前配置用于位置计算
        self._current_lanes = config["lanes"]
        self._current_orientations = config["orientations"]
        self._current_h_positions = config["h_pos"]
        self._current_v_positions = config["v_pos"]
        
        # 根据FIFO配置计算实际模块尺寸
        actual_module_size = self._calc_module_size(config)
        module_w, module_h = actual_module_size
        
        # 绘制模块边框
        module_rect = Rectangle((base_x - module_w/2, base_y - module_h/2),
                               module_w, module_h,
                               facecolor='none', edgecolor='black', linewidth=2)
        self.ax.add_patch(module_rect)
        
        # 添加标题（缩小字体和间距）
        self.ax.text(base_x, base_y + module_h/2 + 0.15, config["title"],
                    fontsize=8, weight='bold', ha='center', va='bottom')
        
        # 绘制FIFO lanes
        lanes = config["lanes"]
        depths = config["depths"]
        orientations = config["orientations"]
        h_positions = config["h_pos"]
        v_positions = config["v_pos"]
        patch_dict = config["patch_dict"]
        text_dict = config["text_dict"]
        
        for i, lane in enumerate(lanes):
            depth = depths[i]
            orientation = orientations[i]
            h_pos = h_positions[i]
            v_pos = v_positions[i]
            
            # 计算FIFO位置
            fifo_x, fifo_y = self._calc_fifo_position(base_x, base_y, actual_module_size, 
                                                     i, len(lanes), orientation, h_pos, v_pos)
            
            # 绘制FIFO
            self._draw_fifo(fifo_x, fifo_y, depth, orientation, lane, patch_dict, text_dict)
        
        # 清理临时存储
        delattr(self, '_current_lanes')
        delattr(self, '_current_orientations') 
        delattr(self, '_current_h_positions')
        delattr(self, '_current_v_positions')
    
    def _calc_fifo_position(self, base_x: float, base_y: float, module_size: Tuple[float, float],
                           index: int, total_lanes: int, orientation: str, h_pos: str, v_pos: str) -> Tuple[float, float]:
        """计算FIFO位置（基于src_old的精确算法）"""
        module_w, module_h = module_size
        
        # 根据方向分组计算位置
        if orientation == "vertical":
            # 垂直FIFO的位置计算
            if h_pos == "top":
                y = base_y + module_h/2 - self.gap_hv - self.square/2
            elif h_pos == "bottom": 
                y = base_y - module_h/2 + self.gap_hv + self.square/2
            else:  # mid
                y = base_y
            
            if v_pos == "left":
                x = base_x - module_w/2 + self.gap_lr + self.square/2
            elif v_pos == "right":
                x = base_x + module_w/2 - self.gap_lr - self.square/2
            else:  # mid
                x = base_x
                
        else:  # horizontal
            # 水平FIFO的位置计算
            if h_pos == "top":
                y = base_y + module_h/2 - self.gap_hv - self.square/2
            elif h_pos == "bottom":
                y = base_y - module_h/2 + self.gap_hv + self.square/2
            else:  # mid
                y = base_y
            
            if v_pos == "left":
                x = base_x - module_w/2 + self.gap_lr + self.square/2
            elif v_pos == "right":
                x = base_x + module_w/2 - self.gap_lr - self.square/2
            else:  # mid
                x = base_x
        
        # 同一位置分组的FIFO需要错开排列
        same_pos_fifos = []
        lanes = getattr(self, '_current_lanes', [])
        orientations = getattr(self, '_current_orientations', [])
        h_positions = getattr(self, '_current_h_positions', [])
        v_positions = getattr(self, '_current_v_positions', [])
        
        for i in range(total_lanes):
            if (i < len(orientations) and i < len(h_positions) and i < len(v_positions) and
                orientations[i] == orientation and h_positions[i] == h_pos and v_positions[i] == v_pos):
                same_pos_fifos.append(i)
        
        if len(same_pos_fifos) > 1:
            fifo_index_in_group = same_pos_fifos.index(index)
            group_size = len(same_pos_fifos)
            
            if orientation == "vertical":
                # 垂直FIFO在水平方向错开
                offset = (fifo_index_in_group - (group_size - 1) / 2) * self.fifo_gap
                x += offset
            else:
                # 水平FIFO在垂直方向错开  
                offset = (fifo_index_in_group - (group_size - 1) / 2) * self.fifo_gap
                y += offset
        
        return x, y
    
    def _draw_fifo(self, x: float, y: float, depth: int, orientation: str, 
                   lane: str, patch_dict: dict, text_dict: dict):
        """绘制单个FIFO"""
        patches = []
        texts = []
        
        for i in range(depth):
            if orientation == "vertical":
                slot_x = x
                slot_y = y + (i - depth/2 + 0.5) * (self.square + self.gap)
            else:  # horizontal
                slot_x = x + (i - depth/2 + 0.5) * (self.square + self.gap)
                slot_y = y
            
            # 创建slot矩形
            rect = Rectangle((slot_x - self.square/2, slot_y - self.square/2),
                           self.square, self.square,
                           facecolor='white', edgecolor='black', 
                           linewidth=self.slot_frame_lw)
            self.ax.add_patch(rect)
            patches.append(rect)
            
            # 创建文本
            text = self.ax.text(slot_x, slot_y, '', fontsize=self.fontsize,
                              ha='center', va='center', weight='bold')
            texts.append(text)
        
        # 添加FIFO标签
        if orientation == "vertical":
            label_x = x + self.square/2 + self.text_gap
            label_y = y
        else:
            label_x = x
            label_y = y - self.square/2 - self.text_gap
        
        self.ax.text(label_x, label_y, lane, fontsize=self.fontsize,
                    ha='left' if orientation == "vertical" else 'center',
                    va='center' if orientation == "vertical" else 'top',
                    rotation=0 if orientation == "vertical" else 0)
        
        patch_dict[lane] = patches
        text_dict[lane] = texts
    
    def _calc_module_size(self, config: dict) -> Tuple[float, float]:
        """根据FIFO配置计算模块尺寸（基于src_old算法）"""
        lanes = config["lanes"]
        depths = config["depths"]
        orientations = config["orientations"]
        h_positions = config["h_pos"]
        v_positions = config["v_pos"]
        
        # 按位置分组FIFO
        h_groups = {"top": [], "mid": [], "bottom": []}
        v_groups = {"left": [], "mid": [], "right": []}
        
        for i, (orientation, h_pos, v_pos, depth) in enumerate(zip(orientations, h_positions, v_positions, depths)):
            if orientation == "horizontal":
                h_groups[h_pos].append(depth)
            else:  # vertical
                v_groups[v_pos].append(depth)
        
        # 计算各组的最大深度
        max_h_depths = {pos: max(depths) if depths else 0 for pos, depths in h_groups.items()}
        max_v_depths = {pos: max(depths) if depths else 0 for pos, depths in v_groups.items()}
        
        # 计算模块宽度（基于垂直FIFO）
        width = 0
        for pos in ["left", "mid", "right"]:
            if max_v_depths[pos] > 0:
                group_width = max_v_depths[pos] * (self.square + self.gap) + 2 * self.gap_lr
                width = max(width, group_width)
        
        # 垂直FIFO组之间的间距
        active_v_groups = sum(1 for depth in max_v_depths.values() if depth > 0)
        if active_v_groups > 1:
            width += (active_v_groups - 1) * self.fifo_gap
        
        width = max(width, 3.0)  # 最小宽度
        
        # 计算模块高度（基于水平FIFO）
        height = 0
        for pos in ["top", "mid", "bottom"]:
            if max_h_depths[pos] > 0:
                group_height = max_h_depths[pos] * (self.square + self.gap) + 2 * self.gap_hv
                height = max(height, group_height)
        
        # 水平FIFO组之间的间距
        active_h_groups = sum(1 for depth in max_h_depths.values() if depth > 0)
        if active_h_groups > 1:
            height += (active_h_groups - 1) * self.fifo_gap
        
        height = max(height, 3.0)  # 最小高度
        
        return height, width
    
    def _draw_crosspoints(self):
        """绘制CrossPoint（增强版本，支持slice和数据流显示）"""
        # 水平CrossPoint（调整位置和尺寸）
        cph_x, cph_y = 1.8, -1.5
        self.cp_module_size = (1.5, 3)  # 调整CrossPoint尺寸
        self._draw_enhanced_crosspoint(cph_x, cph_y, "水平CP", "horizontal", self.cph_patches, self.cph_texts)
        
        # 垂直CrossPoint（调整位置和尺寸）
        cpv_x, cpv_y = -1.5, 1.8
        self._draw_enhanced_crosspoint(cpv_x, cpv_y, "垂直CP", "vertical", self.cpv_patches, self.cpv_texts)
    
    def _draw_enhanced_crosspoint(self, x: float, y: float, title: str, direction: str,
                                 patch_dict: dict, text_dict: dict):
        """绘制增强的CrossPoint（支持slice和数据流显示）"""
        # 绘制CrossPoint边框
        cp_w, cp_h = self.cp_module_size
        cp_rect = Rectangle((x - cp_w/2, y - cp_h/2), cp_w, cp_h,
                           facecolor='lightcyan', edgecolor='blue', linewidth=2)
        self.ax.add_patch(cp_rect)
        
        # 添加标题（缩小字体和间距）
        self.ax.text(x, y + cp_h/2 + 0.1, title, fontsize=6, weight='bold',
                    ha='center', va='bottom')
        
        # 为每个方向绘制arrival和departure slice
        slice_names = ["arrival", "departure"] 
        patches = {}
        texts = {}
        
        for i, slice_name in enumerate(slice_names):
            slice_x = x + (i - 0.5) * 0.8  # 缩小间距
            slice_y = y
            
            # 绘制slice容器（缩小）
            slice_rect = Rectangle((slice_x - 0.25, slice_y - 0.25), 0.5, 0.5,
                                 facecolor='white', edgecolor='blue', linewidth=0.8)
            self.ax.add_patch(slice_rect)
            
            # 在slice内部绘制4个flit slot（2x2布局，缩小）
            slice_patches = []
            slice_texts = []
            
            for row in range(2):
                for col in range(2):
                    slot_x = slice_x + (col - 0.5) * 0.15  # 缩小slot间距
                    slot_y = slice_y + (row - 0.5) * 0.15
                    
                    # 小的flit slot（进一步缩小）
                    slot_rect = Rectangle((slot_x - 0.05, slot_y - 0.05), 0.1, 0.1,
                                        facecolor='white', edgecolor='gray', 
                                        linewidth=0.3)
                    self.ax.add_patch(slot_rect)
                    slice_patches.append(slot_rect)
                    
                    # 文本（缩小字体）
                    slot_text = self.ax.text(slot_x, slot_y, '', fontsize=3,
                                           ha='center', va='center')
                    slice_texts.append(slot_text)
            
            # 添加slice标签（缩小字体和间距）
            self.ax.text(slice_x, slice_y - 0.35, slice_name[:3], fontsize=4,
                        ha='center', va='top', weight='bold')
            
            patches[slice_name] = slice_patches
            texts[slice_name] = slice_texts
        
        # 绘制数据流箭头（缩小尺寸）
        if direction == "horizontal":
            # 水平方向：左右连接
            arrow = FancyArrowPatch((x - 0.5, y), (x + 0.5, y),
                                  connectionstyle="arc3", 
                                  arrowstyle='->', mutation_scale=8,
                                  color='lightblue', alpha=0.5, linewidth=1)
        else:
            # 垂直方向：上下连接  
            arrow = FancyArrowPatch((x, y - 0.5), (x, y + 0.5),
                                  connectionstyle="arc3",
                                  arrowstyle='->', mutation_scale=8,
                                  color='lightgreen', alpha=0.5, linewidth=1)
        
        self.ax.add_patch(arrow)
        patches["arrow"] = arrow
        
        # 添加仲裁状态显示（缩小字体）
        status_text = self.ax.text(x, y - cp_h/2 - 0.2, "IDLE", fontsize=5,
                                 ha='center', va='top', weight='bold')
        texts["status"] = status_text
        
        patch_dict.update(patches)
        text_dict.update(texts)
    
    def update_node_state(self, node_data: dict):
        """
        更新节点状态显示
        
        Args:
            node_data: 节点数据，包含各个FIFO的状态
                      格式: {
                          'inject_queues': {lane: [FlitProxy, ...]},
                          'eject_queues': {lane: [FlitProxy, ...]},
                          'ring_bridge': {lane: [FlitProxy, ...]},
                          'crosspoints': {cp_id: CrossPointData}
                      }
        """
        # 清空旧的映射
        self.patch_info_map.clear()
        self.current_highlight_flit = None
        
        # 更新Inject Queues
        iq_data = node_data.get('inject_queues', {})
        self._update_fifo_display(iq_data, self.iq_patches, self.iq_texts)
        
        # 更新Eject Queues
        eq_data = node_data.get('eject_queues', {})
        self._update_fifo_display(eq_data, self.eq_patches, self.eq_texts)
        
        # 更新Ring Bridge
        rb_data = node_data.get('ring_bridge', {})
        self._update_fifo_display(rb_data, self.rb_patches, self.rb_texts)
        
        # 更新CrossPoints
        cp_data = node_data.get('crosspoints', {})
        self._update_crosspoint_display(cp_data)
        
        # 更新信息文本（如果有当前高亮的flit）
        if self.current_highlight_flit:
            self.info_text.set_text(str(self.current_highlight_flit))
    
    def _update_fifo_display(self, fifo_data: Dict[str, List[FlitProxy]], 
                            patches: dict, texts: dict):
        """更新FIFO显示"""
        for lane, patches_list in patches.items():
            texts_list = texts.get(lane, [])
            flits = fifo_data.get(lane, [])
            
            for idx, patch in enumerate(patches_list):
                text = texts_list[idx] if idx < len(texts_list) else None
                
                if idx < len(flits):
                    flit = flits[idx]
                    
                    # 设置颜色和样式
                    face_color, alpha, linewidth, edge_color = self._get_flit_style(flit)
                    patch.set_facecolor(face_color)
                    patch.set_alpha(alpha)
                    patch.set_linewidth(linewidth)
                    patch.set_edgecolor(edge_color)
                    
                    # 设置文本
                    if text:
                        info = f"{flit.packet_id}-{flit.flit_id}"
                        text.set_text(info)
                        text.set_visible(self.use_highlight and flit.packet_id == self.highlight_pid)
                    
                    # 添加到映射
                    self.patch_info_map[patch] = (text, flit)
                    
                    # 检查是否是高亮的flit
                    if self.use_highlight and flit.packet_id == self.highlight_pid:
                        self.current_highlight_flit = flit
                        
                else:
                    # 空slot
                    patch.set_facecolor("white")
                    patch.set_alpha(1.0)
                    patch.set_linewidth(self.slot_frame_lw)
                    patch.set_edgecolor("black")
                    
                    if text:
                        text.set_visible(False)
                    
                    # 移除映射
                    if patch in self.patch_info_map:
                        self.patch_info_map.pop(patch, None)
    
    def _update_crosspoint_display(self, cp_data: Dict[str, CrossPointData]):
        """更新CrossPoint显示（增强版本）"""
        for cp_id, data in cp_data.items():
            if cp_id == "horizontal":
                patches_dict = self.cph_patches
                texts_dict = self.cph_texts
            elif cp_id == "vertical":
                patches_dict = self.cpv_patches
                texts_dict = self.cpv_texts
            else:
                continue
            
            # 更新仲裁状态
            status_text = texts_dict.get("status")
            if status_text:
                status_text.set_text(data.arbitration_state.upper())
                # 根据状态设置颜色
                if data.arbitration_state == "active":
                    status_text.set_color('green')
                elif data.arbitration_state == "arbitrating":
                    status_text.set_color('orange')
                else:
                    status_text.set_color('black')
            
            # 更新slice数据显示
            for slice_name in ["arrival", "departure"]:
                slice_patches = patches_dict.get(slice_name, [])
                slice_texts = texts_dict.get(slice_name, [])
                slice_flits = data.slice_data.get(slice_name, [])
                
                # 更新每个slice内的flit slots
                for idx, (patch, text) in enumerate(zip(slice_patches, slice_texts)):
                    if idx < len(slice_flits):
                        flit = slice_flits[idx]
                        # 设置flit样式
                        face_color, alpha, linewidth, edge_color = self._get_flit_style(flit)
                        patch.set_facecolor(face_color)
                        patch.set_alpha(alpha)
                        patch.set_linewidth(linewidth)
                        patch.set_edgecolor(edge_color)
                        
                        # 设置文本显示
                        text.set_text(f"{flit.packet_id[:1]}")
                        
                        # 添加到映射
                        self.patch_info_map[patch] = (text, flit)
                    else:
                        # 空slot
                        patch.set_facecolor('white')
                        patch.set_alpha(1.0)
                        patch.set_linewidth(0.5)
                        patch.set_edgecolor('gray')
                        text.set_text('')
                        
                        # 移除映射
                        if patch in self.patch_info_map:
                            self.patch_info_map.pop(patch, None)
            
            # 更新数据流箭头状态
            arrow = patches_dict.get("arrow")
            if arrow and len(data.active_connections) > 0:
                arrow.set_alpha(1.0)
                arrow.set_linewidth(3)
            elif arrow:
                arrow.set_alpha(0.3)
                arrow.set_linewidth(1)
    
    def _get_flit_style(self, flit: FlitProxy) -> Tuple[str, float, float, str]:
        """
        获取flit的显示样式
        
        Returns:
            (facecolor, alpha, linewidth, edgecolor)
        """
        # E-Tag优先级样式
        etag_alpha = {"T0": 1.0, "T1": 1.0, "T2": 0.75}
        etag_lw = {"T0": 2.5, "T1": 1.5, "T2": 0.5}
        etag_edge = {"T0": "red", "T1": "orange", "T2": "black"}
        
        # 获取基础颜色（与link_state_visualizer保持一致）
        if self.use_highlight:
            if flit.packet_id == self.highlight_pid:
                face_color = self.highlight_color
            else:
                face_color = self.grey_color
        else:
            # 根据packet_id使用调色板（与link可视化器完全一致）
            try:
                pid_num = int(flit.packet_id) if flit.packet_id.isdigit() else hash(flit.packet_id)
                face_color = self._colors[abs(pid_num) % len(self._colors)]
            except:
                face_color = self._colors[0]  # 默认使用第一个颜色
        
        # 根据E-Tag设置样式
        alpha = etag_alpha.get(flit.etag_priority, 1.0)
        linewidth = etag_lw.get(flit.etag_priority, 1.0)
        edge_color = etag_edge.get(flit.etag_priority, "black")
        
        # I-Tag特殊标识
        if flit.itag_h or flit.itag_v:
            edge_color = "yellow"
            linewidth = max(linewidth, 2.0)
        
        return face_color, alpha, linewidth, edge_color
    
    def _on_click(self, event):
        """处理鼠标点击事件"""
        if event.inaxes != self.ax:
            return
        
        for patch, (text, flit) in self.patch_info_map.items():
            contains, _ = patch.contains(event)
            if contains:
                # 只有在高亮模式下才切换文本可见性
                if self.use_highlight and flit.packet_id == self.highlight_pid:
                    vis = not text.get_visible()
                    text.set_visible(vis)
                    if vis:
                        text.set_zorder(patch.get_zorder() + 1)
                
                # 显示flit信息
                self.info_text.set_text(str(flit))
                self.current_highlight_flit = flit
                
                # 通知父级高亮
                if self.highlight_callback:
                    try:
                        self.highlight_callback(flit.packet_id, flit.flit_id)
                    except Exception as e:
                        self.logger.warning(f"高亮回调失败: {e}")
                
                self.fig.canvas.draw_idle()
                break
        else:
            # 点击空白处清空信息
            self.info_text.set_text("")
    
    def sync_highlight(self, use_highlight: bool, highlight_pid: Optional[str]):
        """同步高亮状态"""
        self.use_highlight = use_highlight
        self.highlight_pid = highlight_pid
        
        # 更新所有patch的文本可见性
        for patch, (text, flit) in self.patch_info_map.items():
            if self.use_highlight and flit.packet_id == self.highlight_pid:
                text.set_visible(True)
            else:
                text.set_visible(False)
        
        if not self.use_highlight:
            self.info_text.set_text("")
    
    def sync_with_parent(self):
        """与父级可视化器同步状态"""
        if self.parent_visualizer:
            # 同步高亮状态
            if hasattr(self.parent_visualizer, 'use_highlight'):
                self.sync_highlight(
                    self.parent_visualizer.use_highlight,
                    getattr(self.parent_visualizer, 'tracked_pid', None)
                )
            
            # 同步选中节点
            if hasattr(self.parent_visualizer, '_selected_node'):
                new_node_id = self.parent_visualizer._selected_node
                if new_node_id != self.node_id:
                    self.node_id = new_node_id
                    return True  # 表示需要重新绘制
        return False
    
    def update_from_model(self, model, selected_node_id: Optional[int] = None):
        """
        从CrossRing模型直接提取并更新节点状态
        
        Args:
            model: CrossRing模型实例
            selected_node_id: 选中的节点ID，如果为None则使用当前节点ID
        """
        # 更新节点ID
        if selected_node_id is not None:
            self.node_id = selected_node_id
        
        # 从模型中提取当前节点数据
        if hasattr(model, 'nodes') and self.node_id < len(model.nodes):
            crossring_node = model.nodes[self.node_id]
            node_data = self.extract_node_state(crossring_node)
            self.update_node_state(node_data)
    
    def save_snapshot(self, filename: str):
        """保存节点状态快照"""
        self.fig.savefig(filename, dpi=300, bbox_inches='tight')
        self.logger.info(f"节点{self.node_id}状态快照已保存: {filename}")
    
    def extract_node_state(self, crossring_node) -> dict:
        """
        从实际的CrossRing节点提取状态数据
        
        Args:
            crossring_node: CrossRingNode实例
            
        Returns:
            标准化的节点状态数据
        """
        node_data = {
            'inject_queues': {},
            'eject_queues': {},
            'ring_bridge': {},
            'crosspoints': {}
        }
        
        # 提取Inject Queue状态
        iq = crossring_node.inject_queue
        for ip_id in iq.connected_ips:
            for channel in ['req', 'rsp', 'data']:
                ch_buffer = iq.ip_inject_channel_buffers[ip_id][channel]
                if hasattr(ch_buffer, 'is_empty') and not ch_buffer.is_empty():
                    lane_name = f"{ip_id}_{channel}_ch"
                    node_data['inject_queues'][lane_name] = [
                        self._convert_flit_to_proxy(flit) for flit in ch_buffer.get_all_flits()
                    ]
                elif hasattr(ch_buffer, 'data') and ch_buffer.data:
                    lane_name = f"{ip_id}_ch"
                    flits = [self._convert_flit_to_proxy(flit) for flit in ch_buffer.data if flit is not None]
                    if flits:
                        if lane_name not in node_data['inject_queues']:
                            node_data['inject_queues'][lane_name] = flits
                        else:
                            node_data['inject_queues'][lane_name].extend(flits)
        
        # 提取IQ方向FIFO - 简化为主要显示的lane名称
        for channel in ['req', 'rsp', 'data']:
            for direction in ['TR', 'TL', 'TU', 'TD', 'EQ']:
                direction_fifo = iq.inject_direction_fifos[channel][direction]
                flits = []
                if hasattr(direction_fifo, 'get_all_flits') and hasattr(direction_fifo, 'is_empty') and not direction_fifo.is_empty():
                    flits = [self._convert_flit_to_proxy(flit) for flit in direction_fifo.get_all_flits()]
                elif hasattr(direction_fifo, 'data') and direction_fifo.data:
                    flits = [self._convert_flit_to_proxy(flit) for flit in direction_fifo.data if flit is not None]
                
                if flits:
                    # 使用简化的lane名称匹配可视化配置
                    lane_name = f"IQ_{direction}"
                    if lane_name not in node_data['inject_queues']:
                        node_data['inject_queues'][lane_name] = flits
                    else:
                        node_data['inject_queues'][lane_name].extend(flits)
        
        # 提取Eject Queue状态
        eq = crossring_node.eject_queue
        for channel in ['req', 'rsp', 'data']:
            for direction in ['TU', 'TD', 'TR', 'TL']:
                eject_fifo = eq.eject_input_fifos[channel][direction]
                flits = []
                if hasattr(eject_fifo, 'get_all_flits') and hasattr(eject_fifo, 'is_empty') and not eject_fifo.is_empty():
                    flits = [self._convert_flit_to_proxy(flit) for flit in eject_fifo.get_all_flits()]
                elif hasattr(eject_fifo, 'data') and eject_fifo.data:
                    flits = [self._convert_flit_to_proxy(flit) for flit in eject_fifo.data if flit is not None]
                
                if flits:
                    lane_name = f"EQ_{direction}"
                    if lane_name not in node_data['eject_queues']:
                        node_data['eject_queues'][lane_name] = flits
                    else:
                        node_data['eject_queues'][lane_name].extend(flits)
        
        # 提取EQ channel buffers  
        for ip_id in eq.connected_ips:
            for channel in ['req', 'rsp', 'data']:
                ch_buffer = eq.ip_eject_channel_buffers[ip_id][channel]
                flits = []
                if hasattr(ch_buffer, 'get_all_flits') and hasattr(ch_buffer, 'is_empty') and not ch_buffer.is_empty():
                    flits = [self._convert_flit_to_proxy(flit) for flit in ch_buffer.get_all_flits()]
                elif hasattr(ch_buffer, 'data') and ch_buffer.data:
                    flits = [self._convert_flit_to_proxy(flit) for flit in ch_buffer.data if flit is not None]
                
                if flits:
                    lane_name = f"{ip_id}_ch"  # 简化为不包含channel的名称
                    if lane_name not in node_data['eject_queues']:
                        node_data['eject_queues'][lane_name] = flits
                    else:
                        node_data['eject_queues'][lane_name].extend(flits)
        
        # 提取Ring Bridge状态
        rb = crossring_node.ring_bridge
        for channel in ['req', 'rsp', 'data']:
            # 输入FIFO
            for direction in ['TR', 'TL', 'TU', 'TD']:
                rb_in_fifo = rb.ring_bridge_input_fifos[channel][direction]
                flits = []
                if hasattr(rb_in_fifo, 'get_all_flits') and hasattr(rb_in_fifo, 'is_empty') and not rb_in_fifo.is_empty():
                    flits = [self._convert_flit_to_proxy(flit) for flit in rb_in_fifo.get_all_flits()]
                elif hasattr(rb_in_fifo, 'data') and rb_in_fifo.data:
                    flits = [self._convert_flit_to_proxy(flit) for flit in rb_in_fifo.data if flit is not None]
                
                if flits:
                    lane_name = f"RB_{direction}_in"
                    if lane_name not in node_data['ring_bridge']:
                        node_data['ring_bridge'][lane_name] = flits
                    else:
                        node_data['ring_bridge'][lane_name].extend(flits)
            
            # 输出FIFO
            for direction in ['EQ', 'TR', 'TL', 'TU', 'TD']:
                rb_out_fifo = rb.ring_bridge_output_fifos[channel][direction]
                flits = []
                if hasattr(rb_out_fifo, 'get_all_flits') and hasattr(rb_out_fifo, 'is_empty') and not rb_out_fifo.is_empty():
                    flits = [self._convert_flit_to_proxy(flit) for flit in rb_out_fifo.get_all_flits()]
                elif hasattr(rb_out_fifo, 'data') and rb_out_fifo.data:
                    flits = [self._convert_flit_to_proxy(flit) for flit in rb_out_fifo.data if flit is not None]
                
                if flits:
                    lane_name = f"RB_{direction}_out"
                    if lane_name not in node_data['ring_bridge']:
                        node_data['ring_bridge'][lane_name] = flits
                    else:
                        node_data['ring_bridge'][lane_name].extend(flits)
        
        # 提取CrossPoint状态
        node_data['crosspoints']['horizontal'] = self._extract_crosspoint_data(
            crossring_node.horizontal_crosspoint, 'horizontal'
        )
        node_data['crosspoints']['vertical'] = self._extract_crosspoint_data(
            crossring_node.vertical_crosspoint, 'vertical'
        )
        
        return node_data
    
    def _convert_flit_to_proxy(self, flit) -> FlitProxy:
        """将实际flit转换为可视化代理"""
        return FlitProxy(
            packet_id=str(flit.packet_id),
            flit_id=str(flit.flit_id),
            etag_priority=getattr(flit, 'etag_priority', 'T2'),
            itag_h=getattr(flit, 'itag_h', False),
            itag_v=getattr(flit, 'itag_v', False)
        )
    
    def _extract_crosspoint_data(self, crosspoint, direction: str) -> CrossPointData:
        """提取CrossPoint状态数据"""
        slice_data = {}
        
        # 提取slice数据
        if hasattr(crosspoint, 'arrival_slice'):
            slice_data['arrival'] = [
                self._convert_flit_to_proxy(flit) for flit in crosspoint.arrival_slice
                if flit is not None
            ]
        
        if hasattr(crosspoint, 'departure_slice'):
            slice_data['departure'] = [
                self._convert_flit_to_proxy(flit) for flit in crosspoint.departure_slice
                if flit is not None
            ]
        
        # 获取仲裁状态
        arb_state = getattr(crosspoint, 'arbitration_state', 'idle')
        active_connections = getattr(crosspoint, 'active_connections', [])
        
        return CrossPointData(
            cp_id=f"{direction}_cp",
            direction=direction,
            slice_data=slice_data,
            arbitration_state=arb_state,
            active_connections=active_connections
        )


# 工具函数
def create_demo_node_data() -> dict:
    """创建演示用的节点数据"""
    import random
    
    # 创建足够的演示flit
    demo_flits = [
        FlitProxy(f"P{i}", f"F{j}", 
                 etag_priority=random.choice(['T0', 'T1', 'T2']),
                 itag_h=random.random() < 0.1,
                 itag_v=random.random() < 0.1)
        for i in range(1, 8) for j in range(3)  # 增加到21个flit
    ]
    
    # 根据实际FIFO名称分配到不同的FIFO
    node_data = {
        'inject_queues': {
            'gdma_ch': demo_flits[:2],
            'ddr_ch': demo_flits[2:3],
            'IQ_TR': demo_flits[3:4],
            'IQ_TL': demo_flits[4:5] if len(demo_flits) > 4 else [],
        },
        'eject_queues': {
            'EQ_TU': demo_flits[5:6] if len(demo_flits) > 5 else [],
            'EQ_TD': demo_flits[6:7] if len(demo_flits) > 6 else [],
            'gdma_ch': demo_flits[7:8] if len(demo_flits) > 7 else [],
        },
        'ring_bridge': {
            'RB_TR_in': demo_flits[8:9] if len(demo_flits) > 8 else [],
            'RB_EQ_out': demo_flits[9:10] if len(demo_flits) > 9 else [],
            'RB_TU_out': demo_flits[10:11] if len(demo_flits) > 10 else [],
        },
        'crosspoints': {
            'horizontal': CrossPointData('h_cp', 'horizontal', 
                                       slice_data={'arrival': demo_flits[11:12] if len(demo_flits) > 11 else []},
                                       arbitration_state='active', 
                                       active_connections=[('TL', 'TR')]),
            'vertical': CrossPointData('v_cp', 'vertical', 
                                     slice_data={'departure': demo_flits[12:13] if len(demo_flits) > 12 else []},
                                     arbitration_state='idle'),
        }
    }
    
    return node_data


if __name__ == "__main__":
    # 简单测试
    from types import SimpleNamespace
    
    # 创建测试配置
    config = SimpleNamespace(
        NUM_COL=3, NUM_ROW=2,
        IQ_OUT_FIFO_DEPTH=8, EQ_IN_FIFO_DEPTH=8,
        RB_IN_FIFO_DEPTH=4, RB_OUT_FIFO_DEPTH=4,
        IQ_CH_FIFO_DEPTH=4, EQ_CH_FIFO_DEPTH=4,
        CH_NAME_LIST=['gdma', 'ddr', 'l2m']
    )
    
    # 创建可视化器
    visualizer = CrossRingNodeVisualizer(config, node_id=0)
    
    # 创建测试数据
    test_data = create_demo_node_data()
    visualizer.update_node_state(test_data)
    
    plt.show()