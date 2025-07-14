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
    
    def __init__(self, config, ax=None, node_id: int = 0, highlight_callback: Optional[Callable] = None):
        """
        初始化CrossRing节点可视化器
        
        Args:
            config: CrossRing配置对象
            ax: matplotlib轴对象
            node_id: 节点ID
            highlight_callback: 高亮回调函数
        """
        self.config = config
        self.node_id = node_id
        self.highlight_callback = highlight_callback
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
        
        # 几何参数（基于旧版本）
        self.square = 0.3      # flit方块边长
        self.gap = 0.02        # 相邻槽间距
        self.fifo_gap = 0.8    # FIFO间隙
        self.fontsize = 8
        self.gap_lr = 0.35     # 左右内边距
        self.gap_hv = 0.35     # 上下内边距
        self.text_gap = 0.1
        self.slot_frame_lw = 0.4
        
        # 模块尺寸
        self.inject_module_size = (8, 5)   # (height, width)
        self.eject_module_size = (5, 8)    # (width, height) 
        self.rb_module_size = (8, 8)       # (height, width)
        self.cp_module_size = (2, 5)       # (height, width)
        
        # 调色板
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
        
        # 连接点击事件
        self.fig.canvas.mpl_connect("button_press_event", self._on_click)
        
        # 信息显示框
        self.info_text = self.fig.text(0.75, 0.02, "", fontsize=12, 
                                      va="bottom", ha="left", wrap=True)
    
    def _draw_modules(self):
        """绘制所有模块（基于旧版本的模块配置）"""
        
        # 统一模块配置
        iq_config = {
            "title": "Inject Queue",
            "lanes": self.ch_names + ["TL", "TR", "TD", "TU", "EQ"],
            "depths": [self.iq_ch_depth] * len(self.ch_names) + [self.iq_depth] * 5,
            "orientations": ["vertical"] * len(self.ch_names) + ["vertical"] * 2 + ["horizontal"] * 3,
            "h_pos": ["top"] * len(self.ch_names) + ["bottom"] * 2 + ["mid"] * 3,
            "v_pos": ["left"] * len(self.ch_names) + ["left"] * 2 + ["right"] * 3,
            "patch_dict": self.iq_patches,
            "text_dict": self.iq_texts,
        }
        
        eq_config = {
            "title": "Eject Queue", 
            "lanes": self.ch_names + ["TU", "TD"],
            "depths": [self.eq_ch_depth] * len(self.ch_names) + [self.eq_depth] * 2,
            "orientations": ["horizontal"] * len(self.ch_names) + ["horizontal"] * 2,
            "h_pos": ["mid"] * len(self.ch_names) + ["mid"] * 2,
            "v_pos": ["top"] * len(self.ch_names) + ["bottom"] * 2,
            "patch_dict": self.eq_patches,
            "text_dict": self.eq_texts,
        }
        
        rb_config = {
            "title": "Ring Bridge",
            "lanes": ["TL_in", "TR_in", "TU_in", "TD_in", "TL_out", "TR_out", "TU_out", "TD_out"],
            "depths": [self.rb_in_depth] * 4 + [self.rb_out_depth] * 4,
            "orientations": ["vertical"] * 8,
            "h_pos": ["top"] * 4 + ["bottom"] * 4,
            "v_pos": ["left"] * 2 + ["right"] * 2 + ["left"] * 2 + ["right"] * 2,
            "patch_dict": self.rb_patches,
            "text_dict": self.rb_texts,
        }
        
        # 绘制三个主要模块
        self._draw_module(-4, 0.0, self.inject_module_size, iq_config)
        self._draw_module(0.0, 4, self.eject_module_size, eq_config)
        self._draw_module(0.0, 0.0, self.rb_module_size, rb_config)
        
        # 绘制CrossPoint（简化版本）
        self._draw_crosspoints()
    
    def _draw_module(self, base_x: float, base_y: float, module_size: Tuple[float, float], config: dict):
        """绘制单个模块"""
        module_w, module_h = module_size
        
        # 绘制模块边框
        module_rect = Rectangle((base_x - module_w/2, base_y - module_h/2),
                               module_w, module_h,
                               facecolor='none', edgecolor='black', linewidth=2)
        self.ax.add_patch(module_rect)
        
        # 添加标题
        self.ax.text(base_x, base_y + module_h/2 + 0.3, config["title"],
                    fontsize=12, weight='bold', ha='center', va='bottom')
        
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
            fifo_x, fifo_y = self._calc_fifo_position(base_x, base_y, module_size, 
                                                     i, len(lanes), orientation, h_pos, v_pos)
            
            # 绘制FIFO
            self._draw_fifo(fifo_x, fifo_y, depth, orientation, lane, patch_dict, text_dict)
    
    def _calc_fifo_position(self, base_x: float, base_y: float, module_size: Tuple[float, float],
                           index: int, total_lanes: int, orientation: str, h_pos: str, v_pos: str) -> Tuple[float, float]:
        """计算FIFO位置"""
        module_w, module_h = module_size
        
        # 简化的位置计算
        if orientation == "vertical":
            if v_pos == "left":
                x = base_x - module_w/3
            elif v_pos == "right":
                x = base_x + module_w/3
            else:  # mid
                x = base_x
            
            if h_pos == "top":
                y = base_y + module_h/4
            elif h_pos == "bottom":
                y = base_y - module_h/4
            else:  # mid
                y = base_y
                
        else:  # horizontal
            if h_pos == "top":
                y = base_y + module_h/4
            elif h_pos == "bottom":
                y = base_y - module_h/4
            else:  # mid
                y = base_y
            
            if v_pos == "left":
                x = base_x - module_w/4
            elif v_pos == "right":
                x = base_x + module_w/4
            else:  # mid
                x = base_x
        
        # 添加一些偏移避免重叠
        x += (index % 3 - 1) * 0.3
        y += (index // 3 - 1) * 0.3
        
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
    
    def _draw_crosspoints(self):
        """绘制CrossPoint（简化版本）"""
        # 水平CrossPoint
        cph_x, cph_y = 2, -2
        self._draw_simple_crosspoint(cph_x, cph_y, "水平CP", self.cph_patches, self.cph_texts)
        
        # 垂直CrossPoint  
        cpv_x, cpv_y = -2, 2
        self._draw_simple_crosspoint(cpv_x, cpv_y, "垂直CP", self.cpv_patches, self.cpv_texts)
    
    def _draw_simple_crosspoint(self, x: float, y: float, title: str, 
                               patch_dict: dict, text_dict: dict):
        """绘制简化的CrossPoint"""
        # 绘制CrossPoint边框
        cp_w, cp_h = self.cp_module_size
        cp_rect = Rectangle((x - cp_w/2, y - cp_h/2), cp_w, cp_h,
                           facecolor='lightcyan', edgecolor='blue', linewidth=2)
        self.ax.add_patch(cp_rect)
        
        # 添加标题
        self.ax.text(x, y + cp_h/2 + 0.2, title, fontsize=10, weight='bold',
                    ha='center', va='bottom')
        
        # 绘制简化的slice显示
        directions = ["arrival", "departure"]
        patches = []
        texts = []
        
        for i, direction in enumerate(directions):
            slice_x = x + (i - 0.5) * 0.8
            slice_y = y
            
            # 小矩形表示slice
            slice_rect = Rectangle((slice_x - 0.3, slice_y - 0.3), 0.6, 0.6,
                                 facecolor='white', edgecolor='blue', linewidth=1)
            self.ax.add_patch(slice_rect)
            patches.append(slice_rect)
            
            # 文本
            text = self.ax.text(slice_x, slice_y, '', fontsize=6,
                              ha='center', va='center')
            texts.append(text)
        
        patch_dict["slices"] = patches
        text_dict["slices"] = texts
    
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
        """更新CrossPoint显示"""
        # 简化的CrossPoint更新
        for cp_id, data in cp_data.items():
            if cp_id == "horizontal":
                patches = self.cph_patches.get("slices", [])
                texts = self.cph_texts.get("slices", [])
            elif cp_id == "vertical":
                patches = self.cpv_patches.get("slices", [])
                texts = self.cpv_texts.get("slices", [])
            else:
                continue
            
            # 显示仲裁状态
            state_text = f"{data.arbitration_state[:4]}"
            if len(texts) > 0:
                texts[0].set_text(state_text)
            
            # 根据活跃连接数设置颜色
            activity_level = len(data.active_connections)
            if activity_level > 0:
                color = 'lightgreen'
            else:
                color = 'white'
            
            for patch in patches:
                patch.set_facecolor(color)
    
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
        
        # 获取基础颜色
        if self.use_highlight:
            if flit.packet_id == self.highlight_pid:
                face_color = self.highlight_color
            else:
                face_color = self.grey_color
        else:
            # 根据packet_id使用调色板
            try:
                pid_num = int(flit.packet_id) if flit.packet_id.isdigit() else hash(flit.packet_id)
                face_color = self._colors[pid_num % len(self._colors)]
            except:
                face_color = 'lightblue'
        
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
    
    def save_snapshot(self, filename: str):
        """保存节点状态快照"""
        self.fig.savefig(filename, dpi=300, bbox_inches='tight')
        self.logger.info(f"节点{self.node_id}状态快照已保存: {filename}")


# 工具函数
def create_demo_node_data() -> dict:
    """创建演示用的节点数据"""
    import random
    
    # 创建一些演示flit
    demo_flits = [
        FlitProxy(f"P{i}", f"F{j}", 
                 priority=random.choice(['T0', 'T1', 'T2']),
                 itag_h=random.random() < 0.1,
                 itag_v=random.random() < 0.1)
        for i in range(1, 4) for j in range(2)
    ]
    
    # 分配到不同的FIFO
    node_data = {
        'inject_queues': {
            'gdma': demo_flits[:2],
            'TL': demo_flits[2:3],
            'TR': demo_flits[3:4],
        },
        'eject_queues': {
            'ddr': demo_flits[4:5],
            'TU': demo_flits[5:6],
        },
        'ring_bridge': {
            'TL_in': demo_flits[6:7] if len(demo_flits) > 6 else [],
            'TR_out': demo_flits[7:8] if len(demo_flits) > 7 else [],
        },
        'crosspoints': {
            'horizontal': CrossPointData('h_cp', 'horizontal', arbitration_state='active', 
                                       active_connections=[('TL', 'TR')]),
            'vertical': CrossPointData('v_cp', 'vertical', arbitration_state='idle'),
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