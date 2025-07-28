#!/usr/bin/env python3
"""
CrossRing Link State Visualizer

基于原版Link_State_Visualizer.py重新实现，保持原有的完整功能：
1. 左侧显示CrossRing网络拓扑
2. 右侧显示选中节点的详细视图（Inject Queue, Eject Queue, Ring Bridge, CrossPoint）
3. 底部控制按钮（REQ/RSP/DATA切换, Clear HL, Show Tags等）
4. 支持点击节点切换详细视图
5. 支持包追踪和高亮
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.patches import Rectangle, Circle, FancyArrowPatch
from matplotlib.widgets import Button, RadioButtons
from matplotlib.lines import Line2D
from collections import defaultdict, deque
import copy
import threading
from matplotlib.widgets import Button
import time
from types import SimpleNamespace
from typing import Dict, List, Any, Optional, Tuple, Union
from src.noc.base.config import BaseNoCConfig
from src.noc.crossring.config import CrossRingConfig
from src.noc.visualization.crossring_node_visualizer import CrossRingNodeVisualizer
from src.noc.base.model import BaseNoCModel

# 移除了logging依赖

# 字体支持 - 中文用SimHei，英文用Times
plt.rcParams["font.sans-serif"] = ["SimHei", "Times New Roman", "Arial Unicode MS"]
plt.rcParams["font.serif"] = ["Times New Roman", "Times", "serif"]
plt.rcParams["axes.unicode_minus"] = False


# ---------- lightweight flit proxy for snapshot rendering ----------
class _FlitProxy:
    __slots__ = ("packet_id", "flit_id", "ETag_priority", "itag_h", "itag_v")

    def __init__(self, pid, fid, etag, ih, iv):
        self.packet_id = pid
        self.flit_id = fid
        self.ETag_priority = etag
        self.itag_h = ih
        self.itag_v = iv

    def __repr__(self):
        itag = "H" if self.itag_h else ("V" if self.itag_v else "")
        return f"(pid={self.packet_id}, fid={self.flit_id}, ET={self.ETag_priority}, IT={itag})"


class LinkStateVisualizer:
    """
    CrossRing Link State Visualizer

    完全基于原版Link_State_Visualizer重新实现，包含：
    - NetworkLinkVisualizer主类
    - CrossRingNodeVisualizer内嵌类
    - 完整的拓扑显示和节点详细视图
    """

    def __init__(self, config, model: BaseNoCModel):
        """
        初始化CrossRing Link State Visualizer

        Args:
            config: CrossRing配置对象
            network: 网络模型对象（可选）
        """
        self.config = config
        self._parent_model = model  # 建立与模型的连接
        # 移除logger，使用简单的调试输出

        # 网络参数
        self.rows = config.NUM_ROW
        self.cols = config.NUM_COL
        self.num_nodes = self.rows * self.cols

        # 调色板
        self._colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]

        # 当前显示的通道
        self.current_channel = "data"  # req/rsp/data，默认显示data通道

        # 高亮控制
        self.tracked_pid = None
        self.use_highlight = False

        # 播放控制状态
        self._is_paused = False
        self._current_speed = 1  # 更新间隔
        self._current_cycle = 0
        self._last_update_time = time.time()

        # 历史回放功能
        from collections import deque

        self.history = deque(maxlen=50)  # 保存最近50个周期的历史
        self._play_idx = None  # 当前回放索引，None表示实时模式

        # 状态显示文本
        self._status_text = None

        # 选中的节点
        self._selected_node = 0

        # 创建图形界面
        self._setup_gui()
        self._setup_controls()
        self._draw_static_elements()

        # 连接事件
        self._connect_events()

    def _setup_gui(self):
        """设置GUI布局"""
        # 创建主窗口 - 增大图形尺寸以容纳更多内容
        self.fig = plt.figure(figsize=(20, 12), constrained_layout=True)
        gs = self.fig.add_gridspec(1, 2, width_ratios=[1.2, 1], left=0.02, right=0.98, top=0.95, bottom=0.08)
        model_name = getattr(self._parent_model, 'model_name', 'CrossRing NoC')
        self.fig.suptitle(f"{model_name} Simulation", fontsize=16, fontweight="bold", family="serif")

        # 左侧：网络拓扑视图
        self.link_ax = self.fig.add_subplot(gs[0])  # 主网络视图

        # 状态显示区域（左上角）
        self._setup_status_display()

        # 右侧：节点详细视图 - 调整尺寸和位置
        self.node_ax = self.fig.add_subplot(gs[1])

        # 创建节点详细视图可视化器
        self.node_vis = CrossRingNodeVisualizer(config=self.config, ax=self.node_ax, highlight_callback=self._on_highlight_callback, parent=self)

        # 设置默认标题
        self._update_network_title()
        self._update_node_title()

    def _setup_status_display(self):
        """设置状态显示"""
        # 在左上角创建状态文本
        self._status_text = self.link_ax.text(
            0.02, 0.98, "", transform=self.link_ax.transAxes, fontsize=10, verticalalignment="top", bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgray", alpha=0.8), family="sans-serif"
        )
        self._update_status_display()

    def _update_status_display(self):
        """更新状态显示"""
        if self._status_text is None:
            return

        # 获取当前状态信息
        current_time = time.time()
        fps = 1.0 / (current_time - self._last_update_time) if current_time > self._last_update_time else 0
        self._last_update_time = current_time

        # 获取模型状态
        paused = getattr(self._parent_model, "_paused", False) if self._parent_model else False
        frame_interval = getattr(self._parent_model, "_visualization_frame_interval", 0.5) if self._parent_model else 0.5
        current_cycle = getattr(self._parent_model, "cycle", 0) if self._parent_model else 0

        # 确定状态和颜色
        if paused and self._play_idx is not None:
            # 重放模式
            status_icon = "[重放]"
            color = "orange"
            if self._play_idx < len(self.history):
                replay_cycle, _ = self.history[self._play_idx]
                display_cycle = f"{replay_cycle} ({self._play_idx+1}/{len(self.history)})"
            else:
                display_cycle = "无历史"
        elif paused:
            # 暂停模式
            status_icon = "[暂停]"
            color = "red"
            display_cycle = current_cycle
        else:
            # 仿真模式
            status_icon = "[仿真]"
            color = "green"
            display_cycle = current_cycle

        # 构建状态文本
        status_text = f"""状态: {status_icon}
周期: {display_cycle}
间隔: {frame_interval:.2f}s
追踪: {self.tracked_pid if self.tracked_pid else '无'}
"""

        self._status_text.set_text(status_text)
        self._status_text.set_color(color)

    def _setup_controls(self):
        """设置控制按钮"""
        # REQ/RSP/DATA 按钮
        req_ax = self.fig.add_axes([0.05, 0.03, 0.05, 0.04])
        rsp_ax = self.fig.add_axes([0.12, 0.03, 0.05, 0.04])
        data_ax = self.fig.add_axes([0.19, 0.03, 0.05, 0.04])

        self.req_btn = Button(req_ax, "REQ")
        self.rsp_btn = Button(rsp_ax, "RSP")
        self.data_btn = Button(data_ax, "DATA")

        # 设置按钮字体为Times
        for btn in [self.req_btn, self.rsp_btn, self.data_btn]:
            btn.label.set_fontfamily("serif")

        self.req_btn.on_clicked(lambda x: self._on_channel_select("req"))
        self.rsp_btn.on_clicked(lambda x: self._on_channel_select("rsp"))
        self.data_btn.on_clicked(lambda x: self._on_channel_select("data"))

        # Clear Highlight 按钮
        clear_ax = self.fig.add_axes([0.28, 0.03, 0.07, 0.04])
        self.clear_btn = Button(clear_ax, "Clear HL")
        self.clear_btn.on_clicked(self._on_clear_highlight)

        # Show Tags 按钮
        tags_ax = self.fig.add_axes([0.37, 0.03, 0.07, 0.04])
        self.tags_btn = Button(tags_ax, "Show Tags")
        self.tags_btn.on_clicked(self._on_toggle_tags)

        # 设置其他按钮字体为Times
        for btn in [self.clear_btn, self.tags_btn]:
            btn.label.set_fontfamily("serif")

    def _draw_static_elements(self):
        """绘制静态元素"""
        # 计算节点位置
        self.node_positions = {}
        node_size = 0.4
        spacing_x = 2.0
        spacing_y = 1.5

        for row in range(self.rows):
            for col in range(self.cols):
                node_id = row * self.cols + col
                x = col * spacing_x
                y = (self.rows - 1 - row) * spacing_y
                self.node_positions[node_id] = (x, y)

        # 绘制节点
        self.node_patches = {}
        self.node_texts = {}

        for node_id, (x, y) in self.node_positions.items():
            # 节点矩形
            node_rect = Rectangle((x - node_size / 2, y - node_size / 2), node_size, node_size, facecolor="lightblue", edgecolor="black", linewidth=1)
            self.link_ax.add_patch(node_rect)
            self.node_patches[node_id] = node_rect

            # 节点编号
            node_text = self.link_ax.text(x, y, str(node_id), fontsize=10, weight="bold", ha="center", va="center", family="serif")
            self.node_texts[node_id] = node_text

        # 绘制链路
        self._draw_links()

        # 绘制选择框
        self._draw_selection_box()

        # 设置坐标轴
        margin = 1.0
        if self.node_positions:
            min_x = min(pos[0] for pos in self.node_positions.values()) - margin
            max_x = max(pos[0] for pos in self.node_positions.values()) + margin
            min_y = min(pos[1] for pos in self.node_positions.values()) - margin
            max_y = max(pos[1] for pos in self.node_positions.values()) + margin

            self.link_ax.set_xlim(min_x, max_x)
            self.link_ax.set_ylim(min_y, max_y)

        # 设置坐标轴比例为相等，确保正方形不被拉伸
        self.link_ax.set_aspect("equal")
        self.link_ax.axis("off")

    def _draw_links(self):
        """绘制链路"""
        # 存储链路和slot信息
        self.link_info = {}
        self.rect_info_map = {}  # slot_rect -> (link_id, flit, slot_idx)
        self.node_pair_slots = {}  # 存储每对节点之间的slot位置信息

        # 根据实际的网络结构动态绘制链路
        if hasattr(self._parent_model, "links"):
            # 从网络中获取实际存在的链路
            for link_id in self._parent_model.links.keys():
                # 解析link_id来确定源和目标节点
                src_id, dest_id = self._parse_link_id(link_id)
                if src_id is not None and dest_id is not None and src_id != dest_id:
                    # 跳过自环链路，只绘制节点间连接
                    self._draw_link_frame(src_id, dest_id, link_id)
        else:
            # 绘制水平链路
            for row in range(self.rows):
                for col in range(self.cols - 1):
                    src_id = row * self.cols + col
                    dest_id = row * self.cols + col + 1
                    link_id = f"link_{src_id}_TR_{dest_id}"
                    self._draw_link_frame(src_id, dest_id, link_id)

            # 绘制垂直链路
            for row in range(self.rows - 1):
                for col in range(self.cols):
                    src_id = row * self.cols + col
                    dest_id = (row + 1) * self.cols + col
                    link_id = f"link_{src_id}_TD_{dest_id}"
                    self._draw_link_frame(src_id, dest_id, link_id)

    def _parse_link_id(self, link_id):
        """解析link_id获取源和目标节点

        处理各种格式：
        - link_0_TR_1 -> (0, 1)
        - link_0_TL_TR_0 -> (0, 0) 自环
        - link_0_TU_TD_0 -> (0, 0) 自环
        - h_0_1 -> (0, 1) demo格式
        - v_0_2 -> (0, 2) demo格式
        """
        try:
            parts = link_id.split("_")
            if len(parts) >= 4 and parts[0] == "link":
                # 标准格式
                src_id = int(parts[1])
                if len(parts) == 4:  # link_0_TR_1
                    dest_id = int(parts[3])
                elif len(parts) == 5:  # link_0_TL_TR_0
                    dest_id = int(parts[4])
                else:
                    return None, None
                return src_id, dest_id
            elif len(parts) == 3:
                # Demo格式: h_0_1, v_0_2
                direction, src_str, dest_str = parts
                if direction in ["h", "v"]:
                    src_id = int(src_str)
                    dest_id = int(dest_str)
                    return src_id, dest_id
        except (ValueError, IndexError):
            pass
        return None, None

    def _convert_demo_link_id(self, demo_link_id):
        """将demo链路ID转换为标准格式
        
        demo格式: h_0_1, h_2_3, v_0_2, v_1_3
        标准格式: link_0_TR_1, link_2_TR_3, link_0_TD_2, link_1_TD_3
        """
        try:
            parts = demo_link_id.split("_")
            if len(parts) == 3:
                direction, src_str, dest_str = parts
                src_id = int(src_str)
                dest_id = int(dest_str)
                
                if direction == "h":  # 水平链路
                    return f"link_{src_id}_TR_{dest_id}"
                elif direction == "v":  # 垂直链路
                    return f"link_{src_id}_TD_{dest_id}"
        except (ValueError, IndexError):
            pass
        
        # 如果转换失败，返回原ID
        return demo_link_id

    def _draw_link_frame(self, src, dest, link_id, slice_num=None):
        """绘制链路框架，包含箭头和slice

        Args:
            src: 源节点ID
            dest: 目标节点ID
            link_id: 链路ID
            slice_num: slice数量（从配置或实际链路获取）
        """
        if slice_num is None:
            # 从实际的链路获取slice数量
            if hasattr(self._parent_model, "links") and link_id in self._parent_model.links:
                link = self._parent_model.links[link_id]
                if hasattr(link, "num_slices"):
                    slice_num = link.num_slices  # 使用链路的实际slice数量
                elif hasattr(link, "ring_slices") and isinstance(link.ring_slices, dict):
                    # ring_slices是字典，获取任一通道的slice数量
                    first_channel = list(link.ring_slices.keys())[0]
                    slice_num = len(link.ring_slices[first_channel])
                else:
                    slice_num = getattr(self.config, "SLICE_PER_LINK", 8)
            else:
                # 根据链路类型确定slice数量
                if src == dest:  # 自环链路
                    slice_num = getattr(self.config, "SELF_LINK_SLICES", 2)
                else:  # 正常链路
                    slice_num = getattr(self.config, "NORMAL_LINK_SLICES", 8)

        src_pos = self.node_positions[src]
        dest_pos = self.node_positions[dest]

        # 计算基本参数
        dx = dest_pos[0] - src_pos[0]
        dy = dest_pos[1] - src_pos[1]
        dist = np.sqrt(dx * dx + dy * dy)

        if dist > 0:
            # 归一化方向向量
            unit_dx = dx / dist
            unit_dy = dy / dist

            # 垂直偏移向量（用于分离双向箭头）
            perp_dx = -unit_dy
            perp_dy = unit_dx

            # 节点边界偏移
            node_radius = 0.2
            arrow_offset = 0.08  # 双向箭头间距

            # 计算箭头起止点（从节点边缘开始）
            start_x = src_pos[0] + unit_dx * node_radius
            start_y = src_pos[1] + unit_dy * node_radius
            end_x = dest_pos[0] - unit_dx * node_radius
            end_y = dest_pos[1] - unit_dy * node_radius

            # 检查是否需要绘制箭头（每对节点只绘制一次）
            node_pair = (min(src, dest), max(src, dest))
            draw_arrows = node_pair not in self.node_pair_slots

            if draw_arrows:
                # 绘制双向箭头
                directions = [("forward", 1, f"{link_id}_fwd"), ("backward", -1, f"{link_id}_bwd")]  # src -> dest  # dest -> src

                for direction_name, offset_sign, arrow_id in directions:
                    # 计算偏移后的起止点
                    offset_start_x = start_x + perp_dx * arrow_offset * offset_sign
                    offset_start_y = start_y + perp_dy * arrow_offset * offset_sign
                    offset_end_x = end_x + perp_dx * arrow_offset * offset_sign
                    offset_end_y = end_y + perp_dy * arrow_offset * offset_sign

                    # 反向箭头需要交换起止点
                    if direction_name == "backward":
                        offset_start_x, offset_end_x = offset_end_x, offset_start_x
                        offset_start_y, offset_end_y = offset_end_y, offset_start_y

                    # 绘制箭头
                    arrow = FancyArrowPatch((offset_start_x, offset_start_y), (offset_end_x, offset_end_y), arrowstyle="-|>", mutation_scale=15, color="black", linewidth=1.5, alpha=0.8, zorder=1)
                    self.link_ax.add_patch(arrow)

            # 绘制slice slots
            self._draw_link_slices(src_pos, dest_pos, link_id, slice_num, unit_dx, unit_dy, perp_dx, perp_dy)

    def _draw_link_slices(self, src_pos, dest_pos, link_id, slice_num, unit_dx, unit_dy, perp_dx, perp_dy):
        """绘制链路上的slice slots，双向链路两侧都显示但对齐"""
        # 计算slice布局参数
        slot_size = 0.08  # slot边长
        slot_spacing = 0.02  # slot间距
        side_offset = 0.15  # 距离箭头的距离

        # 计算slice沿链路方向排列的总长度
        total_length = slice_num * slot_size + (slice_num - 1) * slot_spacing

        # 链路起始和结束位置（考虑节点边界）
        node_radius = 0.2
        start_x = src_pos[0] + unit_dx * node_radius
        start_y = src_pos[1] + unit_dy * node_radius
        end_x = dest_pos[0] - unit_dx * node_radius
        end_y = dest_pos[1] - unit_dy * node_radius

        # 计算slice排列区域的起始点
        link_length = np.sqrt((end_x - start_x) ** 2 + (end_y - start_y) ** 2)
        start_offset = (link_length - total_length) / 2

        # 跳过首尾slice的显示
        visible_slice_num = max(0, slice_num - 2)
        if visible_slice_num <= 0:
            return

        # 解析link_id，确定节点对
        src_id, dest_id = self._parse_link_id(link_id)
        node_pair = (min(src_id, dest_id), max(src_id, dest_id)) if src_id is not None and dest_id is not None else None

        # 检查是否已经为这对节点创建了slice（保证对齐）
        if node_pair and node_pair in self.node_pair_slots:
            # 使用已有的slot位置
            existing_slots = self.node_pair_slots[node_pair]
            for i, (slot_positions, slot_id) in enumerate(existing_slots):
                if i < len(existing_slots):
                    # 为当前链路方向创建slot（关联到已有位置）
                    # 这里不需要重新创建rectangle，只需要更新映射
                    for rect, (rect_link_ids, _, rect_slot_id) in self.rect_info_map.items():
                        if rect_slot_id == slot_id:
                            # 添加当前链路到现有slot的映射中
                            if isinstance(rect_link_ids, list):
                                if link_id not in rect_link_ids:
                                    rect_link_ids.append(link_id)
                            else:
                                rect_link_ids = [rect_link_ids, link_id]
                            self.rect_info_map[rect] = (rect_link_ids, None, rect_slot_id)
                            break
        else:
            # 首次为这对节点创建slice
            slot_positions_list = []

            # 在链路两侧都绘制slice
            for side_name, side_sign in [("side1", 1), ("side2", -1)]:
                for i in range(1, slice_num - 1):  # 跳过i=0和i=slice_num-1
                    # 计算沿链路方向的位置
                    along_link_dist = start_offset + i * (slot_size + slot_spacing)
                    progress = along_link_dist / link_length if link_length > 0 else 0

                    # 沿链路方向的中心点
                    center_x = start_x + progress * (end_x - start_x)
                    center_y = start_y + progress * (end_y - start_y)

                    # 垂直于链路方向的偏移
                    slot_x = center_x + perp_dx * side_offset * side_sign - slot_size / 2
                    slot_y = center_y + perp_dy * side_offset * side_sign - slot_size / 2

                    # 创建slot rectangle（默认为空，虚线边框）
                    slot = Rectangle((slot_x, slot_y), slot_size, slot_size, facecolor="none", edgecolor="gray", linewidth=0.8, linestyle="--", alpha=0.7)
                    self.link_ax.add_patch(slot)

                    # 记录slot信息
                    slot_id = f"{side_name}_{i}"
                    slot_positions_list.append(((slot_x, slot_y), slot_id))
                    self.rect_info_map[slot] = ([link_id], None, slot_id)

            # 记录这对节点的slot位置，供反向链路使用
            if node_pair:
                self.node_pair_slots[node_pair] = slot_positions_list

    def _draw_selection_box(self):
        """绘制选择框"""
        if self._selected_node in self.node_positions:
            node_pos = self.node_positions[self._selected_node]
            self.click_box = Rectangle((node_pos[0] - 0.3, node_pos[1] - 0.3), 0.6, 0.6, facecolor="none", edgecolor="red", linewidth=1.2, linestyle="--")
            self.link_ax.add_patch(self.click_box)

    def _connect_events(self):
        """连接事件"""
        self.fig.canvas.mpl_connect("button_press_event", self._on_click)
        self.fig.canvas.mpl_connect("key_press_event", self._on_key_press)

    def _on_click(self, event):
        """处理点击事件"""
        if event.inaxes != self.link_ax:
            return

        # 检查是否点击了slot
        for rect in self.rect_info_map:
            contains, _ = rect.contains(event)
            if contains:
                link_ids, flit, slot_idx = self.rect_info_map[rect]
                if flit:
                    self._on_flit_click(flit)
                return

        # 检查是否点击了节点
        for node_id, pos in self.node_positions.items():
            dx = event.xdata - pos[0]
            dy = event.ydata - pos[1]
            distance = np.sqrt(dx * dx + dy * dy)

            if distance <= 0.3:  # 节点半径
                self._select_node(node_id)
                break

    def _select_node(self, node_id):
        """选择节点"""
        if node_id == self._selected_node:
            return

        self._selected_node = node_id

        # 更新选择框
        if hasattr(self, "click_box"):
            self.click_box.remove()
        self._draw_selection_box()

        # 更新右侧详细视图 - 使用快照数据
        if self._play_idx is not None and self._play_idx < len(self.history):
            # 回放模式：使用当前回放周期的数据
            replay_cycle, _ = self.history[self._play_idx]
            self.node_vis.render_node_from_snapshot(node_id, replay_cycle)
        elif self.history:
            # 实时模式：使用最新快照数据
            latest_cycle, _ = self.history[-1]
            self.node_vis.render_node_from_snapshot(node_id, latest_cycle)

        # 更新节点标题
        self._update_node_title()

        self.fig.canvas.draw_idle()
        # print(f"选中节点: {node_id}")  # 删除debug输出

    def _on_flit_click(self, flit):
        """处理flit点击"""
        pid = getattr(flit, "packet_id", None)
        if pid is not None:
            self._track_packet(pid)

    def _track_packet(self, packet_id):
        """追踪包"""
        self.tracked_pid = packet_id
        self.use_highlight = True

        # 同步CrossRingNodeVisualizer的高亮状态
        self.node_vis.sync_highlight(self.use_highlight, self.tracked_pid)

    def _on_key_press(self, event):
        """处理键盘事件"""
        if event.key == " ":  # 空格键暂停/继续
            self._toggle_pause()
        elif event.key == "r":  # R键重置视图
            self._reset_view()
        elif event.key == "up":  # 上箭头键加速
            self._change_speed(faster=True)
        elif event.key == "down":  # 下箭头键减速
            self._change_speed(faster=False)
        elif event.key.lower() in ["1", "2", "3"]:  # 数字键切换通道
            channels = ["req", "rsp", "data"]
            if int(event.key) <= len(channels):
                self._on_channel_select(channels[int(event.key) - 1])
        elif event.key == "h" or event.key == "?":  # H键或?键显示帮助
            self._show_help()
        elif event.key == "f":  # F键切换到最快速度
            self._set_max_speed()
        elif event.key == "s":  # S键切换到慢速
            self._set_slow_speed()
        elif event.key == "left":  # 左箭头键：回放上一帧（仅暂停时有效）
            self._replay_previous()
        elif event.key == "right":  # 右箭头键：回放下一帧（仅暂停时有效）
            self._replay_next()

        # 更新状态显示
        self._update_status_display()

    def _replay_previous(self):
        """回放上一帧（仅暂停时有效）"""
        if not hasattr(self, "_parent_model") or not self._parent_model:
            return

        paused = getattr(self._parent_model, "_paused", False)
        if not paused or not self.history:
            return

        # 如果当前在实时模式，切换到最后一帧
        if self._play_idx is None:
            self._play_idx = len(self.history) - 1
        else:
            # 向前回放
            self._play_idx = max(0, self._play_idx - 1)

        # 立即更新显示
        if self._play_idx < len(self.history):
            cycle, snapshot_data = self.history[self._play_idx]
            self._render_from_snapshot(snapshot_data)
            # 同时更新节点显示
            if hasattr(self, 'node_vis') and self.node_vis and self._selected_node is not None:
                self.node_vis.render_node_from_snapshot(self._selected_node, cycle)
            self.fig.canvas.draw_idle()

    def _replay_next(self):
        """回放下一帧（仅暂停时有效）"""
        if not hasattr(self, "_parent_model") or not self._parent_model:
            return

        paused = getattr(self._parent_model, "_paused", False)
        if not paused or not self.history:
            return

        # 如果当前在实时模式，什么都不做
        if self._play_idx is None:
            return

        # 向后回放
        self._play_idx = min(len(self.history) - 1, self._play_idx + 1)

        # 立即更新显示
        if self._play_idx < len(self.history):
            cycle, snapshot_data = self.history[self._play_idx]
            self._render_from_snapshot(snapshot_data)
            # 同时更新节点显示
            if hasattr(self, 'node_vis') and self.node_vis and self._selected_node is not None:
                self.node_vis.render_node_from_snapshot(self._selected_node, cycle)
            self.fig.canvas.draw_idle()

    def _toggle_pause(self):
        """切换暂停状态"""
        if hasattr(self, "_parent_model") and self._parent_model:
            # 创建暂停属性如果不存在
            if not hasattr(self._parent_model, "_paused"):
                self._parent_model._paused = False
            self._parent_model._paused = not self._parent_model._paused

            if self._parent_model._paused:
                # 进入暂停：切换到最新历史帧
                if self.history:
                    self._play_idx = len(self.history) - 1
                    cycle, snapshot_data = self.history[self._play_idx]
                    self._render_from_snapshot(snapshot_data)
                    self.fig.canvas.draw_idle()
                status = "暂停"
            else:
                # 退出暂停：回到实时模式
                self._play_idx = None
                status = "继续"

    def _reset_view(self):
        """重置视图"""
        self.tracked_pid = None
        self.use_highlight = False
        if hasattr(self, "piece_vis"):
            self.node_vis.sync_highlight(False, None)

    def _change_speed(self, faster=True):
        """改变仿真速度"""
        if hasattr(self, "_parent_model") and self._parent_model:
            current_interval = getattr(self._parent_model, "_visualization_frame_interval", 0.5)
            if faster:
                new_interval = max(0.05, current_interval * 0.75)
            else:
                new_interval = min(5.0, current_interval * 1.25)
            self._parent_model._visualization_frame_interval = new_interval
            pass  # print(f"速度调整: 帧间隔 {new_interval}s")

    def _set_max_speed(self):
        """设置最大速度"""
        if hasattr(self, "_parent_model") and self._parent_model:
            self._parent_model._visualization_frame_interval = 0.05
            pass  # print("设置为最大速度")

    def _set_slow_speed(self):
        """设置慢速"""
        if hasattr(self, "_parent_model") and self._parent_model:
            self._parent_model._visualization_frame_interval = 2.0
            pass  # print("设置为慢速")

    def _show_help(self):
        """显示键盘快捷键帮助"""
        help_text = """
CrossRing可视化控制键:
========================================
播放控制:
  空格键  - 暂停/继续仿真
  ←       - 回放上一帧 (暂停时)
  →       - 回放下一帧 (暂停时)
  r       - 重置视图和高亮

速度控制:
  ↑       - 加速 (减少更新间隔)
  ↓       - 减速 (增加更新间隔)
  f       - 最大速度 (间隔=0.05s)
  s       - 慢速 (间隔=2.0s)

视图控制:
  1/2/3   - 切换到REQ/RSP/DATA通道
  h或?    - 显示此帮助信息

交互:
  点击节点 - 查看详细信息
  点击flit - 开始追踪包

状态显示:
  绿色 - 仿真运行中
  红色 - 暂停状态
  橙色 - 历史重放模式
========================================
        """
        print(help_text)

    def _on_channel_select(self, channel):
        """通道选择回调"""
        self.current_channel = channel
        pass  # print(f"切换到通道: {channel}")

        # 更新标题
        self._update_network_title()

        # 重新绘制当前状态
        if self._parent_model:
            self.update(self._parent_model)

        self.fig.canvas.draw_idle()

    def _update_network_title(self):
        """更新网络标题"""
        channel_name = {"req": "请求网络", "rsp": "响应网络", "data": "数据网络"}.get(self.current_channel, f"{self.current_channel.upper()}网络")

        self.link_ax.set_title(channel_name, fontsize=14, family="sans-serif", pad=-10)

        # 移除重复的主标题，只保留axis标题

    def _update_node_title(self):
        """更新节点标题"""
        node_title = f"节点 {self._selected_node}"
        self.node_ax.set_title(node_title, fontsize=14, family="sans-serif", pad=-20)

    def _on_clear_highlight(self, event):
        """清除高亮回调"""
        self.tracked_pid = None
        self.use_highlight = False

        # 同步CrossRingNodeVisualizer
        self.node_vis.sync_highlight(self.use_highlight, self.tracked_pid)

        # 清除所有slot颜色
        for rect in self.rect_info_map:
            rect.set_facecolor("none")

        self.fig.canvas.draw_idle()

    def _on_toggle_tags(self, event):
        """切换标签显示"""
        # TODO
        pass  # print("切换标签显示")

    def _on_highlight_callback(self, packet_id, flit_id):
        """高亮回调"""
        self._track_packet(packet_id)

    def update(self, networks=None, cycle=None, skip_pause=False):
        """更新显示"""
        if networks is None and self._parent_model is None:
            return

        network = networks if networks is not None else self._parent_model

        # 保存历史快照（仅在实时模式下，即非回放状态）
        if self._play_idx is None:
            # 如果没有提供cycle，使用默认值0或历史长度
            effective_cycle = cycle if cycle is not None else len(self.history)
            self._save_history_snapshot(network, effective_cycle)

        # 统一使用快照数据更新显示（无论实时还是回放模式）
        if self._play_idx is not None and len(self.history) > self._play_idx:
            # 回放模式：使用指定历史快照
            replay_cycle, snapshot_data = self.history[self._play_idx]
            self._render_from_snapshot(snapshot_data)
            # 节点视图也使用回放数据
            self.node_vis.render_node_from_snapshot(self._selected_node, replay_cycle)
        else:
            # 实时模式：使用最新保存的快照（刚刚保存的）
            if self.history:
                latest_cycle, latest_snapshot = self.history[-1]
                self._render_from_snapshot(latest_snapshot)
                # 节点视图也从最新快照获取数据
                self.node_vis.render_node_from_snapshot(self._selected_node, latest_cycle)

        # 更新状态显示
        self._update_status_display()

        if not skip_pause:
            self.fig.canvas.draw_idle()

    def _save_history_snapshot(self, model, cycle):
        """保存历史快照 - 完整的链路网络状态"""
        try:
            # 第一步：构建完整的链路快照
            # 链路数据结构: {link_id: {channel: {slice_idx: slice_data}}}
            links_snapshot = {}

            if hasattr(model, "links"):
                for link_id, link in model.links.items():
                    # 支持两种链路格式：真正的CrossRing链路和demo链路
                    if hasattr(link, "get_ring_slice") and hasattr(link, "num_slices"):
                        # 真正的CrossRing链路格式
                        # 为每个链路保存所有通道的完整数据
                        link_data = {}

                        for channel in ["req", "rsp", "data"]:
                            channel_data = {}

                            for slice_idx in range(link.num_slices):
                                try:
                                    slice_obj = link.get_ring_slice(channel, slice_idx)
                                    slice_data = {"slots": {}, "metadata": {}}

                                    # 检查RingSlice的所有pipeline阶段，寻找有效的flit
                                    def extract_flit_from_slot(slot, slot_channel):
                                        """从slot中提取flit信息"""
                                        if slot and hasattr(slot, "flit") and slot.flit:
                                            return {
                                                "valid": getattr(slot, "valid", False),
                                                "flit": {
                                                    "packet_id": getattr(slot.flit, "packet_id", None),
                                                    "flit_id": getattr(slot.flit, "flit_id", None),
                                                    "ETag_priority": getattr(slot.flit, "ETag_priority", None),
                                                    "itag_h": getattr(slot.flit, "itag_h", False),
                                                    "itag_v": getattr(slot.flit, "itag_v", False),
                                                    "current_node_id": getattr(slot.flit, "current_node_id", None),
                                                    "flit_position": getattr(slot.flit, "flit_position", None),
                                                    "channel": slot_channel,
                                                },
                                            }
                                        return None

                                    # 检查所有pipeline阶段：current_slots, input_buffer, output_buffer
                                    pipeline_stages = ["current_slots", "input_buffer", "output_buffer"]
                                    
                                    for slot_channel in ["req", "rsp", "data"]:
                                        slot_info = None
                                        
                                        # 按优先级检查pipeline阶段
                                        for stage in pipeline_stages:
                                            if hasattr(slice_obj, stage):
                                                stage_slots = getattr(slice_obj, stage)
                                                if isinstance(stage_slots, dict) and slot_channel in stage_slots:
                                                    slot_info = extract_flit_from_slot(stage_slots[slot_channel], slot_channel)
                                                    if slot_info:  # 找到有效flit就停止搜索
                                                        break
                                        
                                        slice_data["slots"][slot_channel] = slot_info

                                    # 保存slice元数据
                                    slice_data["metadata"] = {"slice_idx": slice_idx, "channel": channel, "timestamp": cycle}

                                    channel_data[slice_idx] = slice_data

                                except Exception as slice_error:
                                    # 忽略无效slice，但保留结构
                                    channel_data[slice_idx] = {"slots": {}, "metadata": {"error": True}}

                            link_data[channel] = channel_data

                        links_snapshot[link_id] = link_data
                    
                    elif hasattr(link, "slices"):
                        # Demo链路格式：简单的slice列表
                        # 将demo链路ID转换为标准格式
                        standard_link_id = self._convert_demo_link_id(link_id)
                        
                        link_data = {}
                        
                        # demo链路只有一个通道，我们用"req"表示
                        channel_data = {}
                        
                        for slice_idx, slice_obj in enumerate(link.slices):
                            slice_data = {"slots": {}, "metadata": {}}
                            
                            # 从demo slice格式提取数据
                            if hasattr(slice_obj, "slot") and slice_obj.slot:
                                slot = slice_obj.slot
                                slot_info = {
                                    "valid": getattr(slot, "valid", True),
                                    "flit": {
                                        "packet_id": getattr(slot, "packet_id", None),
                                        "flit_id": getattr(slot, "flit_id", None),
                                        "ETag_priority": getattr(slot, "etag_priority", "T2"),
                                        "itag_h": getattr(slot, "itag_h", False),
                                        "itag_v": getattr(slot, "itag_v", False),
                                        "current_node_id": None,
                                        "flit_position": None,
                                        "channel": "req",
                                    },
                                }
                            else:
                                slot_info = None
                            
                            slice_data["slots"]["req"] = slot_info
                            slice_data["metadata"] = {"slice_idx": slice_idx, "channel": "req", "timestamp": cycle}
                            channel_data[slice_idx] = slice_data
                        
                        link_data["req"] = channel_data
                        # 为了保持格式一致，添加空的rsp和data通道
                        link_data["rsp"] = {}
                        link_data["data"] = {}
                        
                        links_snapshot[standard_link_id] = link_data

            # 第二步：让节点可视化器保存自己的历史状态
            if hasattr(model, "nodes") and hasattr(self, "node_vis"):
                self.node_vis.save_history_snapshot(model, cycle)

            # 第三步：保存完整快照
            snapshot_data = {
                "cycle": cycle,
                "timestamp": cycle,
                "links": links_snapshot,
                "metadata": {"total_links": len(links_snapshot), "channels": ["req", "rsp", "data"]},
            }

            self.history.append((cycle, snapshot_data))

        except Exception as e:
            # 静默忽略快照保存错误，但保留基本结构
            fallback_snapshot = {"cycle": cycle, "links": {}, "metadata": {"error": True, "error_msg": str(e)}}
            self.history.append((cycle, fallback_snapshot))

    def _render_from_snapshot(self, snapshot_data):
        """从快照渲染"""
        try:
            # 第一步：重置所有slot为默认状态
            for rect in self.rect_info_map:
                rect.set_facecolor("none")
                rect.set_edgecolor("gray")
                rect.set_linewidth(0.8)
                rect.set_linestyle("--")
                rect.set_alpha(0.7)
                # 清除flit数据
                link_ids, _, slot_id = self.rect_info_map[rect]
                self.rect_info_map[rect] = (link_ids, None, slot_id)

            # 第二步：从完整快照中提取当前通道数据
            current_channel = getattr(self, "current_channel", "data")

            # 直接使用统一的快照格式
            self._render_from_snapshot_data(snapshot_data.get("links", {}), current_channel)

        except Exception as e:
            pass  # 静默忽略渲染错误

    def _render_from_snapshot_data(self, links_snapshot, current_channel):
        """从快照数据渲染"""
        flit_count = 0
        processed_links = 0
        for link_id, link_data in links_snapshot.items():
            processed_links += 1
            # 跳过自环链路
            src_id, dest_id = self._parse_link_id(link_id)
            if src_id is not None and dest_id is not None and src_id == dest_id:
                continue

            # 提取当前通道的数据
            channel_data = link_data.get(current_channel, {})

            for slice_idx, slice_data in channel_data.items():
                if isinstance(slice_idx, (int, str)) and "slots" in slice_data:
                    # 转换为整数用于后续处理
                    slice_idx_int = int(slice_idx) if isinstance(slice_idx, str) else slice_idx
                    slots = slice_data["slots"]

                    # 处理所有slot
                    for slot_key, slot_info in slots.items():
                        if slot_info and slot_info.get("valid", False):
                            flit_data = slot_info.get("flit", {})
                            if flit_data:
                                flit_count += 1
                                
                                # 创建临时flit对象
                                temp_flit = _FlitProxy(
                                    pid=flit_data.get("packet_id"),
                                    fid=flit_data.get("flit_id"),
                                    etag=flit_data.get("ETag_priority", "T2"),
                                    ih=flit_data.get("itag_h", False),
                                    iv=flit_data.get("itag_v", False),
                                )

                                # 添加其他属性（跳过__slots__限制的属性）
                                for attr, value in flit_data.items():
                                    if not hasattr(temp_flit, attr):
                                        try:
                                            setattr(temp_flit, attr, value)
                                        except AttributeError:
                                            # 跳过无法设置的属性（由于__slots__限制）
                                            pass

                                self._update_slot_visual(link_id, slice_idx_int, temp_flit)
                                break  # 每个slice只显示一个flit

    def _link_id_matches(self, link_id, pattern):
        """检查link_id是否匹配带通配符的模式"""
        # pattern格式: link_1_*_0
        # link_id格式: link_1_TR_0 或 link_1_TL_0
        pattern_parts = pattern.split("_")
        link_parts = link_id.split("_")

        if len(pattern_parts) != len(link_parts):
            return False

        for p, l in zip(pattern_parts, link_parts):
            if p != "*" and p != l:
                return False
        return True

    def _update_slot_visual(self, link_id, slice_idx, slot):
        """更新单个slot的视觉效果"""
        # 因为我们跳过了首尾slice（range(1, slice_num-1)），需要调整索引匹配
        # slice_idx=0对应不显示，slice_idx=1对应显示的第0个slot，以此类推
        slice_per_link = getattr(self.config, "SLICE_PER_LINK", 7)
        if slice_idx == 0 or slice_idx >= (slice_per_link - 1):
            return  # 跳过首尾slice

        # 查找对应的slot rectangle
        for rect, (rect_link_ids, _, rect_slot_idx) in self.rect_info_map.items():
            # rect_link_ids可能是字符串（旧格式）或列表（新格式）
            if isinstance(rect_link_ids, str):
                rect_link_ids = [rect_link_ids]

            # 检查link_id是否匹配任何一个方向
            link_matched = False
            for rect_link_id in rect_link_ids:
                if rect_link_id == link_id or ("*" in rect_link_id and self._link_id_matches(link_id, rect_link_id)):
                    link_matched = True
                    break

            if link_matched and str(slice_idx) in rect_slot_idx:
                # 更新flit信息
                self.rect_info_map[rect] = (rect_link_ids, slot, rect_slot_idx)

                # 获取flit样式（颜色、透明度、边框等）
                face_color, alpha, line_width, edge_color = self._get_flit_style(slot, use_highlight=self.use_highlight, expected_packet_id=self.tracked_pid, highlight_color="red")

                # 应用样式到rectangle
                rect.set_facecolor(face_color)
                rect.set_alpha(alpha)
                rect.set_edgecolor(edge_color)
                rect.set_linewidth(max(line_width, 0.8))  # 确保最小线宽
                rect.set_linestyle("-")  # 有flit时使用实线
                break

    def _get_flit_style(self, flit, use_highlight=True, expected_packet_id=None, highlight_color=None):
        """
        返回 (facecolor, alpha, linewidth, edgecolor)
        - facecolor 沿用调色板逻辑（高亮 / 调色板）
        - alpha / linewidth 由 flit.ETag_priority 决定
        """
        # E-Tag样式映射
        _ETAG_ALPHA = {"T0": 1.0, "T1": 0.9, "T2": 0.75}
        _ETAG_LW = {"T0": 2.0, "T1": 1.5, "T2": 1.0}
        _ETAG_EDGE = {"T0": "darkred", "T1": "darkblue", "T2": "black"}

        # 获取基础颜色
        face_color = self._get_flit_color(flit, use_highlight, expected_packet_id, highlight_color)

        # 获取E-Tag优先级
        etag = getattr(flit, "ETag_priority", "T2")  # 缺省视为 T2
        alpha = _ETAG_ALPHA.get(etag, 0.8)
        line_width = _ETAG_LW.get(etag, 1.0)
        edge_color = _ETAG_EDGE.get(etag, "black")

        return face_color, alpha, line_width, edge_color

    def _get_flit_color(self, flit, use_highlight=True, expected_packet_id=None, highlight_color=None):
        """获取flit颜色，支持多种PID格式"""
        # 高亮模式：目标 flit → 指定颜色，其余 → 灰
        if use_highlight and expected_packet_id is not None:
            hl_color = highlight_color or "red"
            flit_pid = getattr(flit, "packet_id", None)
            return hl_color if str(flit_pid) == str(expected_packet_id) else "lightgrey"

        # 普通模式：根据packet_id使用调色板颜色
        pid = getattr(flit, "packet_id", 0)
        if pid is not None:
            return self._colors[int(pid) % len(self._colors)]
        else:
            return "lightblue"  # 默认颜色

    def set_network(self, network):
        """设置网络模型"""
        self._parent_model = network

    def get_selected_node(self):
        """获取当前选中的节点"""
        return self._selected_node

    def show(self):
        """显示可视化界面"""
        plt.show()

    def save_figure(self, filename):
        """保存图片"""
        self.fig.savefig(filename, dpi=300, bbox_inches="tight")
        pass  # print(f"图片已保存到: {filename}")
