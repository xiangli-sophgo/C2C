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
from collections import defaultdict
import logging
from dataclasses import dataclass, field
from enum import Enum
import copy
from src.noc.crossring.config import CrossRingConfig
from src.utils.font_config import configure_matplotlib_fonts

# 配置跨平台字体支持
configure_matplotlib_fonts(verbose=False)


class CrossRingNodeVisualizer:
    """节点详细视图可视化器（右侧面板）"""

    def __init__(self, config: CrossRingConfig, ax, highlight_callback=None, parent=None, gpu_mode=False):
        """
        仅绘制单个节点的 Inject/Eject Queue 和 Ring Bridge FIFO。
        参数:
        - config: 含有 FIFO 深度配置的对象，属性包括 cols, num_nodes, IQ_OUT_FIFO_DEPTH,
            EQ_IN_FIFO_DEPTH, RB_IN_FIFO_DEPTH, RB_OUT_FIFO_DEPTH
        - node_id: 要可视化的节点索引 (0 到 num_nodes-1)
        - gpu_mode: 是否启用GPU加速渲染
        """
        self.highlight_callback = highlight_callback
        self.config = config
        self.cols = config.NUM_COL
        self.rows = config.NUM_ROW
        self.parent = parent
        
        # GPU加速模式支持
        self.gpu_mode = gpu_mode or getattr(config, 'gpu_visualization', False)
        if self.gpu_mode:
            self._init_gpu_rendering()

        # 历史保存功能
        from collections import deque

        self.node_history = deque(maxlen=50)  # 保存最近50个周期的节点状态

        self.IQ_OUT_DEPTH = config.fifo_config.IQ_OUT_FIFO_DEPTH
        self.EQ_IN_DEPTH = config.fifo_config.EQ_IN_FIFO_DEPTH
        self.RB_IN_DEPTH = config.fifo_config.RB_IN_FIFO_DEPTH
        self.RB_OUT_DEPTH = config.fifo_config.RB_OUT_FIFO_DEPTH
        self.IQ_CH_depth = config.fifo_config.IQ_CH_DEPTH
        self.EQ_CH_depth = config.fifo_config.EQ_CH_DEPTH
        self.SLICE_PER_LINK = config.basic_config.SLICE_PER_LINK

        # 固定几何参数
        self.square = 0.3  # flit 方块边长
        self.gap = 0.02  # 相邻槽之间间距
        self.fifo_gap = 0.8  # 相邻fifo之间间隙
        self.fontsize = 8

        # ------- layout tuning parameters (all adjustable) -------
        self.gap_lr = 0.35  # 左右内边距
        self.gap_hv = 0.35  # 上下内边距
        self.min_depth_vis = 4  # 设计最小深度 (=4)
        self.text_gap = 0.1
        # ---------------------------------------------------------

        # line‑width for FIFO slot frames (outer border)
        self.slot_frame_lw = 0.6  # can be tuned externally

        # 初始化图形
        if ax is None:
            self.fig, self.ax = plt.subplots(figsize=(10, 8))  # 增大图形尺寸
        else:
            self.ax = ax
            self.fig = ax.figure

        self.ax.axis("off")
        # 改为自动调整比例，而不是强制相等比例
        self.ax.set_aspect("auto")

        # 调色板
        self._colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]

        # ------ highlight / tracking ------
        self.use_highlight = False  # 是否启用高亮模式
        self.highlight_pid = None  # 被追踪的 packet_id
        self.highlight_color = "red"  # 追踪 flit 颜色
        self.grey_color = "lightgrey"  # 其它 flit 颜色
        self.show_tags_mode = False  # 标签显示模式

        # 存储 patch 和 text
        self.iq_patches, self.iq_texts = {}, {}
        self.eq_patches, self.eq_texts = {}, {}
        self.rb_patches, self.rb_texts = {}, {}
        self.cph_patches, self.cph_texts = {}, {}
        self.cpv_patches, self.cpv_texts = {}, {}

        # 画出三个模块的框和 FIFO 槽
        self._draw_modules()

        # 点击显示 flit 信息
        self.patch_info_map = {}  # patch -> (text_obj, info_str)
        self.fig.canvas.mpl_connect("button_press_event", self._on_click)

        # 全局信息显示框（右下角）
        self.info_text = self.fig.text(0.75, 0.02, "", fontsize=12, va="bottom", ha="left", wrap=True, family="serif")

        # 当前被点击 / 高亮的 flit（用于信息框自动刷新）
        self.current_highlight_flit = None

    # ------------------------------------------------------------------ #
    #  计算模块尺寸 (宽 = X 方向, 高 = Y 方向)                             #
    # ------------------------------------------------------------------ #
    def _calc_module_size(self, fifo_specs):
        """
        fifo_specs: list of tuples (orient, h_group, v_group, depth)
        - orient: 'H' or 'V'
        - h_group: for V → 'T' | 'M' | 'B', else None
        - v_group: for H → 'L' | 'M' | 'R', else None
        - depth: int
        The size is determined by the max depth in each group (per orientation), plus number of orthogonal FIFOs.
        """

        # ----- max depth per slot (L/M/R  and  T/M/B) -----------------
        max_depth = {k: 0 for k in ("L", "M_h", "R", "T", "M_v", "B")}

        # counts per side group
        cnt_H = {"L": 0, "M": 0, "R": 0}  # horizontal fifo counts by v_group
        cnt_V = {"T": 0, "M": 0, "B": 0}  # vertical   fifo counts by h_group

        for o, h_grp, v_grp, d in fifo_specs:
            if o == "H":
                # horizontal -> depth to L/M_h/R & count into cnt_H
                g = v_grp or "M"
                key = "M_h" if g == "M" else g
                max_depth[key] = max(max_depth[key], d)
                cnt_H[g] += 1
            else:  # 'V'
                g = h_grp or "M"
                key = "M_v" if g == "M" else g
                max_depth[key] = max(max_depth[key], d)
                cnt_V[g] += 1

        # take MAX count across side groups (per requirement)
        count_H = max(cnt_H.values())  # horizontal fifo effective count
        count_V = max(cnt_V.values())  # vertical fifo effective count

        width_slots = max_depth["L"] + max_depth["M_h"] + max_depth["R"] + count_V * 2 + 4
        height_slots = max_depth["T"] + max_depth["M_v"] + max_depth["B"] + count_H * 2 + 4

        width = width_slots * (self.square + self.gap) + 4 * self.gap_lr
        height = height_slots * (self.square + self.gap) + 4 * self.gap_hv
        return width, height

    def _draw_modules(self):
        """绘制所有模块"""
        # 获取通道名称
        ch_names = getattr(self.config, "CH_NAME_LIST", None)

        # ------------------- unified module configs ------------------- #
        iq_config = dict(
            title="Inject Queue",
            lanes=ch_names + ["TL", "TR", "TD", "TU", "EQ"],
            depths=[self.IQ_CH_depth] * len(ch_names) + [self.IQ_OUT_DEPTH] * 5,
            orientations=["vertical"] * len(ch_names) + ["vertical"] * 2 + ["horizontal"] * 3,
            h_pos=["top"] * len(ch_names) + ["bottom"] * 2 + ["mid"] * 3,
            v_pos=["left"] * len(ch_names) + ["left"] * 2 + ["right"] * 3,
            patch_dict=self.iq_patches,
            text_dict=self.iq_texts,
        )

        eq_config = dict(
            title="Eject Queue",
            lanes=ch_names + ["TU", "TD"],
            depths=[self.EQ_CH_depth] * len(ch_names) + [self.EQ_IN_DEPTH] * 2,
            orientations=["horizontal"] * len(ch_names) + ["horizontal"] * 2,
            h_pos=["top"] * len(ch_names) + ["bottom"] * 2,
            v_pos=["left"] * len(ch_names) + ["right", "right"],
            patch_dict=self.eq_patches,
            text_dict=self.eq_texts,
        )

        rb_config = dict(
            title="Ring Bridge",
            lanes=["TL_in", "TR_in", "TU_out", "TD_out", "EQ_out"],  # 匹配实际数据格式
            depths=[self.RB_IN_DEPTH] * 2 + [self.RB_OUT_DEPTH] * 3,
            orientations=["vertical", "vertical", "horizontal", "horizontal", "vertical"],
            h_pos=["bottom", "bottom", "top", "top", "top"],
            v_pos=["left", "left", "right", "right", "left"],
            patch_dict=self.rb_patches,
            text_dict=self.rb_texts,
        )

        cross_point_horizontal_config = dict(
            title="CP",
            lanes=["TL_arr", "TL_dep", "TR_arr", "TR_dep"],  # 拆分arrival和departure
            depths=[1, 1, 1, 1],
            orientations=["horizontal", "horizontal", "horizontal", "horizontal"],
            h_pos=["bottom", "bottom", "top", "top"],
            v_pos=["right", "left", "left", "right"],
            patch_dict=self.cph_patches,
            text_dict=self.cph_texts,
        )

        cross_point_vertical_config = dict(
            title="CP",
            lanes=["TU_arr", "TU_dep", "TD_arr", "TD_dep"],  # 拆分arrival和departure
            depths=[1, 1, 1, 1],
            orientations=["vertical", "vertical", "vertical", "vertical"],
            h_pos=["bottom", "top", "top", "bottom"],
            v_pos=["left", "left", "left", "left"],
            patch_dict=self.cpv_patches,
            text_dict=self.cpv_texts,
        )

        # ---------------- compute sizes via fifo specs ---------------- #
        def make_specs(c):
            """
            Build a list of (orient, h_group, v_group, depth) for each fifo lane.
            Each spec tuple is (orient, h_group, v_group, depth), unused group is None.
            """
            specs = []
            for ori, hp, vp, d in zip(c["orientations"], c["h_pos"], c["v_pos"], c["depths"]):
                if ori[0].upper() == "H":
                    v_group = {"left": "L", "right": "R"}.get(vp, "M")
                    h_group = {"top": "T", "bottom": "B"}.get(hp, "M")
                    specs.append(("H", h_group, v_group, d))
                else:  # vertical
                    v_group = {"left": "L", "right": "R"}.get(vp, "M")
                    h_group = {"top": "T", "bottom": "B"}.get(hp, "M")
                    specs.append(("V", h_group, v_group, d))
            return specs

        w_iq, h_iq = self._calc_module_size(make_specs(iq_config))
        w_eq, h_eq = self._calc_module_size(make_specs(eq_config))
        w_rb, h_rb = self._calc_module_size(make_specs(rb_config))
        h_rb = max(h_iq, h_rb)
        w_rb = max(w_eq, w_rb)
        self.inject_module_size = (w_iq, h_rb)
        self.eject_module_size = (w_rb, h_eq)
        self.rb_module_size = (w_rb, h_rb)
        self.cp_module_size = (2.5, 4)

        center_x, center_y = 0, 0
        spacing = 1.5
        RB_x = center_x
        RB_y = center_y
        IQ_x = center_x - self.inject_module_size[0] - spacing
        IQ_y = center_y
        EQ_x = center_x
        EQ_y = center_y + self.rb_module_size[1] + spacing
        CPH_x = center_x - (self.inject_module_size[0] - spacing) / 3
        CPH_y = center_y - self.cp_module_size[1] - spacing / 2
        CPV_x = center_x + self.rb_module_size[0] + spacing
        CPV_y = center_y + (self.rb_module_size[1] + spacing) * 2 / 3

        # 自动调整坐标轴范围以适应所有模块
        self._auto_adjust_axis_limits(IQ_x, IQ_y, RB_x, RB_y, EQ_x, EQ_y, CPH_x, CPH_y, CPV_x, CPV_y)

        # 绘制各个模块
        self._draw_node_module(IQ_x, IQ_y, self.inject_module_size, iq_config)
        self._draw_node_module(EQ_x, EQ_y, self.eject_module_size, eq_config)
        self._draw_node_module(RB_x, RB_y, self.rb_module_size, rb_config)
        self._draw_node_module(CPH_x, CPH_y, self.cp_module_size[::-1], cross_point_horizontal_config)
        self._draw_node_module(CPV_x, CPV_y, self.cp_module_size, cross_point_vertical_config)

    def _auto_adjust_axis_limits(self, IQ_x, IQ_y, RB_x, RB_y, EQ_x, EQ_y, CPH_x, CPH_y, CPV_x, CPV_y):
        """自动调整坐标轴范围以适应所有模块"""
        all_positions = [
            (IQ_x, IQ_y, self.inject_module_size),
            (RB_x, RB_y, self.eject_module_size),
            (EQ_x, EQ_y, self.rb_module_size),
            (CPH_x, CPH_y, self.cp_module_size),
            (CPV_x, CPV_y, self.cp_module_size),
        ]

        # 计算边界
        min_x = min(x for x, y, (h, w) in all_positions)
        max_x = max(x + w for x, y, (h, w) in all_positions)
        min_y = min(y for x, y, (h, w) in all_positions)
        max_y = max(y + h for x, y, (h, w) in all_positions)

        # 添加边距
        margin = 2
        self.ax.set_xlim(min_x - margin, max_x + margin)
        self.ax.set_ylim(min_y - margin * 4, max_y + margin * 0)

    def _draw_node_module(self, x, y, module_size, module_config):
        """绘制节点模块"""
        # 绘制参数
        title = module_config["title"]
        module_width, module_height = module_size
        lanes = module_config["lanes"]
        lane_depths = module_config["depths"]
        orientations = module_config["orientations"]
        h_position = module_config["h_pos"]
        v_position = module_config["v_pos"]
        patch_dict = module_config["patch_dict"]
        text_dict = module_config["text_dict"]

        square = self.square
        gap = self.gap
        fontsize = self.fontsize
        if title == "CP":
            square *= 2
            gap *= 20
            fontsize = 8

        # 处理方向参数
        if orientations is None:
            orientations = ["horizontal"] * len(lanes)
        elif isinstance(orientations, str):
            orientations = [orientations] * len(lanes)

        # 处理 h_position/v_position 支持列表
        if isinstance(h_position, str):
            h_position = [h_position if ori == "horizontal" else None for ori in orientations]
        if isinstance(v_position, str):
            v_position = [v_position if ori == "vertical" else None for ori in orientations]

        if not (len(h_position) == len(v_position) == len(lanes)):
            raise ValueError("h_position, v_position, lanes must have the same length")

        # 处理 depth

        # 绘制模块边框
        box = Rectangle((x, y), module_width, module_height, fill=False, edgecolor="black", linewidth=1.3)
        self.ax.add_patch(box)

        # 模块标题
        title_x = x + module_width / 2
        title_y = y + module_height + 0.05
        self.ax.text(title_x, title_y, title, ha="center", va="bottom", fontweight="bold", family="serif")

        patch_dict.clear()
        text_dict.clear()

        # 分组并组内编号
        group_map = defaultdict(list)
        for i, (ori, hpos, vpos) in enumerate(zip(orientations, h_position, v_position)):
            group_map[(ori, hpos, vpos)].append(i)

        group_idx = {}
        for group, idxs in group_map.items():
            for j, i in enumerate(idxs):
                group_idx[i] = j

        for i, (lane, orient, depth) in enumerate(zip(lanes, orientations, lane_depths)):
            hpos = h_position[i]
            vpos = v_position[i]
            idx_in_group = group_idx[i]
            group_size = len(group_map[(orient, hpos, vpos)])

            if orient == "horizontal":
                # 纵坐标由 hpos 决定
                if hpos == "top":
                    lane_y = y + module_height - ((idx_in_group + 1) * self.fifo_gap) - self.gap_hv
                    text_va = "bottom"
                elif hpos == "bottom":
                    lane_y = y + (idx_in_group * self.fifo_gap) + self.gap_hv
                    text_va = "top"
                elif hpos == "mid":
                    lane_y = y + module_height / 2 + (idx_in_group - 1) * self.fifo_gap
                    text_va = "center"
                else:
                    raise ValueError(f"Unknown h_position: {hpos}")

                # 横坐标由 vpos 决定
                if vpos == "right":
                    lane_x = x + module_width - depth * (square + gap) - self.gap_lr
                    text_x = x + module_width - depth * (square + gap) - self.gap_lr - self.text_gap
                    slot_dir = 1
                    ha = "right"
                elif vpos == "left":
                    lane_x = x + self.gap_lr
                    text_x = x + self.gap_lr + depth * (square + gap) + self.text_gap
                    slot_dir = 1
                    ha = "left"
                elif vpos == "mid" or vpos is None:
                    lane_x = x + module_width / 2 - depth * (square + gap)
                    text_x = x + module_width / 2 - depth * (square + gap) - self.text_gap
                    slot_dir = 1
                    ha = "left"
                else:
                    raise ValueError(f"Unknown v_position: {vpos}")
                # 处理CrossPoint标签 - 每个方向只显示一次标签
                if lane[:2] in ["TL", "TR", "TU", "TD", "EQ"] and title == "CP":
                    # 对于CrossPoint，只在arrival slot显示标签，位置统一
                    if "_arr" in lane:
                        # 为水平方向的CrossPoint调整标签位置，使其居中对齐
                        (label_x, label_y) = (
                            (x + module_width / 2, y + module_height / 2 + square / 2) if lane[:2] in ["TL"] else (x + module_width / 2 - square * 2 / 3, y + module_height / 2 - square)
                        )
                        self.ax.text(label_x, label_y, lane[:2].upper(), ha=ha, va="center", fontsize=fontsize, family="serif")
                elif lane[:2] in ["TL", "TR", "TU", "TD", "EQ"]:
                    # 非CrossPoint的标签正常显示
                    self.ax.text(text_x, lane_y + square / 2, lane[:2].upper(), ha=ha, va="center", fontsize=fontsize, family="serif")
                else:
                    self.ax.text(text_x, lane_y + square / 2, lane[0].upper() + lane[-1], ha=ha, va="center", fontsize=fontsize, family="serif")
                patch_dict[lane] = []
                text_dict[lane] = []

                for s in range(depth):
                    slot_x = lane_x + slot_dir * s * (square + gap)
                    slot_y = lane_y
                    # outer frame (fixed) - use dashed border
                    frame = Rectangle(
                        (slot_x, slot_y),
                        square,
                        square,
                        edgecolor="black",
                        facecolor="none",
                        linewidth=self.slot_frame_lw,
                        linestyle="--",
                    )
                    self.ax.add_patch(frame)

                    # inner patch (dynamic flit) - no border when empty
                    inner = Rectangle(
                        (slot_x, slot_y),
                        square,
                        square,
                        edgecolor="none",
                        facecolor="none",
                        linewidth=0,
                    )
                    self.ax.add_patch(inner)
                    txt = self.ax.text(slot_x, slot_y + (square / 2 + 0.005 if hpos == "top" else -square / 2 - 0.005), "", ha="center", va=text_va, fontsize=fontsize, family="serif")
                    txt.set_visible(False)  # 默认隐藏
                    patch_dict[lane].append(inner)
                    text_dict[lane].append(txt)

            elif orient == "vertical":
                # 横坐标由 vpos 决定
                if vpos == "left":
                    lane_x = x + (idx_in_group * self.fifo_gap) + self.gap_lr
                    text_ha = "right"
                elif vpos == "right":
                    lane_x = x + module_width - (idx_in_group * self.fifo_gap) - self.gap_lr
                    text_ha = "left"
                elif vpos == "mid" or vpos is None:
                    offset = (idx_in_group - (group_size - 1) / 2) * self.fifo_gap
                    lane_x = x + offset
                    text_ha = "center"
                else:
                    raise ValueError(f"Unknown v_position: {vpos}")

                # 纵坐标由 hpos 决定
                if hpos == "top":
                    lane_y = y + module_height - depth * (square + gap) - self.gap_hv
                    text_y = y + module_height - depth * (square + gap) - self.gap_hv - self.text_gap
                    slot_dir = 1
                    va = "top"
                elif hpos == "bottom":
                    lane_y = y + self.gap_hv
                    text_y = y + self.gap_hv + depth * (square + gap) + self.text_gap
                    slot_dir = 1
                    va = "bottom"
                elif hpos == "mid" or hpos is None:
                    lane_y = y + module_height / 2 - (depth / 2) * (square + gap)
                    text_y = y + module_height / 2 - (depth / 2) * (square + gap)
                    slot_dir = 1
                    va = "center"
                else:
                    raise ValueError(f"Unknown h_position: {hpos}")

                # 处理CrossPoint标签 - 每个方向只显示一次标签
                if lane[:2] in ["TL", "TR", "TU", "TD", "EQ"] and title == "CP":
                    # 对于CrossPoint，只在arrival slot显示标签，位置统一
                    if "_arr" in lane:
                        # 为垂直方向的CrossPoint调整标签位置，使其居中对齐
                        (label_x, label_y) = (
                            (x + module_width / 2 - square, y + module_height / 2 - square * 5 / 7) if lane[:2] in ["TU"] else (x + module_width / 2 + square / 3, y + module_height / 2)
                        )
                        self.ax.text(label_x, label_y, lane[:2].upper(), ha="center", va=va, fontsize=fontsize, family="serif")
                elif lane[:2] in ["TL", "TR", "TU", "TD", "EQ"]:
                    # 非CrossPoint的标签正常显示
                    self.ax.text(lane_x + square / 2, text_y, lane[:2].upper(), ha="center", va=va, fontsize=fontsize, family="serif")
                else:
                    self.ax.text(lane_x + square / 2, text_y, lane[0].upper() + lane[-1], ha="center", va=va, fontsize=fontsize, family="serif")
                patch_dict[lane] = []
                text_dict[lane] = []

                for s in range(depth):
                    slot_x = lane_x
                    slot_y = lane_y + slot_dir * s * (square + gap)
                    # outer frame (fixed) - use dashed border
                    frame = Rectangle(
                        (slot_x, slot_y),
                        square,
                        square,
                        edgecolor="black",
                        facecolor="none",
                        linewidth=self.slot_frame_lw,
                        linestyle="--",
                    )
                    self.ax.add_patch(frame)

                    # inner patch (dynamic flit) - no border when empty
                    inner = Rectangle(
                        (slot_x, slot_y),
                        square,
                        square,
                        edgecolor="none",
                        facecolor="none",
                        linewidth=0,
                    )
                    self.ax.add_patch(inner)
                    txt = self.ax.text(slot_x + (square / 2 + 0.005 if vpos == "right" else -square / 2 - 0.005), slot_y, "", ha=text_ha, va="center", fontsize=fontsize, family="serif")
                    txt.set_visible(False)  # 默认隐藏
                    patch_dict[lane].append(inner)
                    text_dict[lane].append(txt)

            else:
                raise ValueError(f"Unknown orientation: {orient}")

    def _calc_fifo_position(self, base_x, base_y, module_size, index, total_lanes, orientation, h_pos, v_pos):
        """计算FIFO位置"""
        module_w, module_h = module_size

        # 简化的位置计算
        if orientation == "vertical":
            if v_pos == "left":
                x = base_x - module_w / 3
            elif v_pos == "right":
                x = base_x + module_w / 3
            else:  # mid
                x = base_x

            if h_pos == "top":
                y = base_y + module_h / 4
            elif h_pos == "bottom":
                y = base_y - module_h / 4
            else:  # mid
                y = base_y

        else:  # horizontal
            if h_pos == "top":
                y = base_y + module_h / 4
            elif h_pos == "bottom":
                y = base_y - module_h / 4
            else:  # mid
                y = base_y

            if v_pos == "left":
                x = base_x - module_w / 4
            elif v_pos == "right":
                x = base_x + module_w / 4
            else:  # mid
                x = base_x

        # 添加一些偏移避免重叠
        x += (index % 3 - 1) * 0.3
        y += (index // 3 - 1) * 0.3

        return x, y

    def _get_flit_style(self, flit, use_highlight=True, expected_packet_id=None, highlight_color=None):
        """
        返回 (facecolor, linewidth, edgecolor)
        - facecolor 包含透明度信息的RGBA颜色（基于flit_id调整透明度）
        - linewidth / edgecolor 由 flit.ETag_priority 决定（tag相关边框属性，不透明）
        """
        import matplotlib.colors as mcolors
        
        # E-Tag样式映射 - 仅控制边框属性，不影响填充透明度
        # 为node中的小方格调整更合适的线宽
        _ETAG_LW = {"T0": 1.2, "T1": 0.9, "T2": 0.6}
        _ETAG_EDGE = {"T0": "darkred", "T1": "darkblue", "T2": "black"}

        # 标签模式下：使用统一的浅色背景，突出显示边框
        if self.show_tags_mode:
            base_color = "lightgray"
        else:
            # 获取基础颜色（不含透明度）
            base_color = self._get_flit_color(flit, use_highlight, expected_packet_id, highlight_color)

        # 获取E-Tag优先级（兼容字典和对象格式）- 仅控制边框样式（边框保持完全不透明）
        if isinstance(flit, dict):
            # 字典格式：优先使用标准化的ETag_priority，然后尝试etag_priority
            etag = flit.get("ETag_priority", flit.get("etag_priority", "T2"))
        else:
            # 对象格式：优先使用etag_priority（CrossRing flit的实际属性名），然后尝试ETag_priority
            etag = getattr(flit, "etag_priority", getattr(flit, "ETag_priority", "T2"))
        line_width = _ETAG_LW.get(etag, 1.0)
        edge_color = _ETAG_EDGE.get(etag, "black")  # 边框颜色保持不透明

        # 根据flit_id调整填充颜色透明度（转换为RGBA格式）
        if self.show_tags_mode:
            # 标签模式下使用固定的中等透明度，便于看清边框
            alpha = 0.3
        else:
            # 正常模式下根据flit_id调整透明度
            if isinstance(flit, dict):
                flit_id = flit.get("flit_id", 0)
            else:
                flit_id = getattr(flit, "flit_id", 0)

            if flit_id is not None:
                # 为同一packet内的不同flit分配不同透明度
                # flit_id=0 -> 1.0倍透明度, flit_id=1 -> 0.8倍, flit_id=2 -> 0.6倍, 等等
                alpha = max(0.4, 1.0 - (int(flit_id) * 0.2))
            else:
                alpha = 1.0  # 默认完全不透明

        # 将基础颜色转换为RGBA格式，嵌入透明度信息
        try:
            # 转换颜色为RGBA元组
            rgba = mcolors.to_rgba(base_color, alpha=alpha)
            face_color_with_alpha = rgba
        except:
            # 如果转换失败，使用默认颜色
            face_color_with_alpha = (0.5, 0.5, 1.0, alpha)  # 浅蓝色

        return face_color_with_alpha, line_width, edge_color

    def _get_flit_color(self, flit, use_highlight=True, expected_packet_id=None, highlight_color=None):
        """获取flit颜色，支持字典和对象两种格式的flit数据"""
        # 兼容字典和对象两种格式获取packet_id
        if isinstance(flit, dict):
            flit_pid = flit.get("packet_id")
        else:
            flit_pid = getattr(flit, "packet_id", None)

        # 高亮模式：目标 flit → 指定颜色，其余 → 灰
        if use_highlight and expected_packet_id is not None:
            hl_color = highlight_color or "red"
            return hl_color if str(flit_pid) == str(expected_packet_id) else "lightgrey"

        # 普通模式：根据packet_id使用调色板颜色
        if flit_pid is not None:
            try:
                # 使用与父类相同的颜色映射
                color_index = int(flit_pid) % len(self.parent._colors)
                selected_color = self.parent._colors[color_index]
                return selected_color
            except Exception as e:
                return "lightblue"
        else:
            return "lightblue"  # 默认颜色

    def _on_click(self, event):
        """处理点击事件"""
        if event.inaxes != self.ax:
            return
        for patch, (txt, flit) in self.patch_info_map.items():
            contains, _ = patch.contains(event)
            if contains:
                # 只有在高亮模式下才允许切换文本可见性
                # 兼容字典和对象两种格式
                if isinstance(flit, dict):
                    pid = flit.get("packet_id", None)
                    fid = flit.get("flit_id", None)
                else:
                    pid = getattr(flit, "packet_id", None)
                    fid = getattr(flit, "flit_id", None)
                if self.use_highlight and pid == self.highlight_pid:
                    vis = not txt.get_visible()
                    txt.set_visible(vis)
                    # 若即将显示，确保在最上层
                    if vis:
                        txt.set_zorder(patch.get_zorder() + 1)
                # 在右下角显示完整 flit 信息
                self.info_text.set_text(self._format_flit_info(flit))
                # 记录当前点击的 flit，方便后续帧仍显示最新信息
                self.current_highlight_flit = flit
                # 通知父级高亮
                if self.highlight_callback:
                    try:
                        self.highlight_callback(int(pid), int(fid))
                    except Exception:
                        pass
                self.fig.canvas.draw_idle()
                break
        else:
            # 点击空白处清空信息
            self.info_text.set_text("")

    def sync_highlight(self, use_highlight, highlight_pid):
        """同步高亮状态"""
        self.use_highlight = use_highlight
        self.highlight_pid = highlight_pid

        # 更新所有patch的颜色和文本可见性
        for patch, (txt, flit) in self.patch_info_map.items():
            # 兼容字典和对象两种格式
            if isinstance(flit, dict):
                pid = flit.get("packet_id", None)
            else:
                pid = getattr(flit, "packet_id", None)

            # 重新计算并应用flit样式（包括颜色）
            if flit:
                face, lw, edge = self._get_flit_style(
                    flit,
                    use_highlight=self.use_highlight,
                    expected_packet_id=self.highlight_pid,
                )
                patch.set_facecolor(face)
                patch.set_linewidth(lw)
                patch.set_edgecolor(edge)

            # 更新文本可见性
            if self.use_highlight and pid == self.highlight_pid:
                txt.set_visible(True)
            else:
                txt.set_visible(False)

        if not self.use_highlight:
            self.info_text.set_text("")

        # 触发重绘
        self.fig.canvas.draw_idle()
    
    def sync_tags_mode(self, show_tags_mode):
        """同步标签显示模式"""
        self.show_tags_mode = show_tags_mode
        
        # 更新所有patch的样式
        for patch, (txt, flit) in self.patch_info_map.items():
            # 重新计算并应用flit样式
            if flit:
                face, lw, edge = self._get_flit_style(
                    flit,
                    use_highlight=self.use_highlight,
                    expected_packet_id=self.highlight_pid,
                )
                patch.set_facecolor(face)
                patch.set_linewidth(lw)
                patch.set_edgecolor(edge)
        
        # 触发重绘
        self.fig.canvas.draw_idle()

    def _format_flit_info(self, flit):
        """Format flit information display - use flit's repr for detailed info"""
        if not flit:
            return "No flit info"

        # 对于字典格式的flit（来自快照），检查是否有保存的repr
        if isinstance(flit, dict):
            # 优先使用保存的repr
            if "flit_repr" in flit:
                return flit["flit_repr"]
            
            # 回退到基本信息显示
            info_lines = []
            packet_id = flit.get("packet_id", None)
            flit_id = flit.get("flit_id", None)
            channel = flit.get("channel", None)
            
            if packet_id is not None:
                info_lines.append(f"Packet ID: {packet_id}")
            if flit_id is not None:
                info_lines.append(f"Flit ID: {flit_id}")
            if channel:
                info_lines.append(f"Channel: {channel}")
                
            return "\n".join(info_lines) if info_lines else "No valid info"
        
        # 对于活动的flit对象，直接使用repr
        try:
            return repr(flit)
        except Exception as e:
            # 如果repr失败，回退到基本信息
            packet_id = getattr(flit, "packet_id", "Unknown")
            flit_id = getattr(flit, "flit_id", "Unknown")
            return f"Packet ID: {packet_id}\nFlit ID: {flit_id}\n(repr failed: {e})"

    def _extract_flit_data(self, flit, channel, direction):
        """提取flit数据的通用方法，包含flit的repr信息"""
        if not flit:
            return None
        
        # 提取基本字段
        # 为ETag_priority添加多种可能的属性名检查，确保兼容性
        # CrossRing flit使用etag_priority（小写），优先检查这个
        etag_priority = getattr(flit, "etag_priority", None)
        if etag_priority is None:
            etag_priority = getattr(flit, "ETag_priority", None)
        if etag_priority is None:
            etag_priority = getattr(flit, "priority", "T2")  # 最后使用默认值
        
        data = {
            "packet_id": getattr(flit, "packet_id", None),
            "flit_id": getattr(flit, "flit_id", None),
            "ETag_priority": etag_priority,
            "itag_h": getattr(flit, "itag_h", False),
            "itag_v": getattr(flit, "itag_v", False),
            "channel": channel,
            "direction": direction,
        }
        
        # 保存flit的完整repr信息
        try:
            data["flit_repr"] = repr(flit)
        except Exception as e:
            data["flit_repr"] = f"repr failed: {e}"
        
        return data

    def _extract_fifo_data(self, fifos, node_id, channels=["req", "rsp", "data"]):
        """提取FIFO数据的通用方法 - 包含internal_queue和output_register"""
        result = {}
        for channel in channels:
            channel_fifos = fifos.get(channel, {})
            channel_data = {}
            for direction, fifo in channel_fifos.items():
                if hasattr(fifo, "internal_queue"):
                    # 提取internal_queue中的flit
                    fifo_data = [self._extract_flit_data(flit, channel, direction) for flit in fifo.internal_queue]

                    # 提取output_register中的flit（如果存在且有效）
                    # 输出寄存器的flit应该在队列第一个位置（下一个要输出的flit）
                    if hasattr(fifo, "output_register") and hasattr(fifo, "output_valid") and fifo.output_valid and fifo.output_register:
                        output_flit_data = self._extract_flit_data(fifo.output_register, channel, direction)
                        fifo_data.insert(0, output_flit_data)  # 插入到队列开头而不是末尾

                    channel_data[direction] = {node_id: fifo_data}
            result[channel] = channel_data
        return result

    def _extract_ip_channel_data(self, ip_channel_buffers, direction_type, channels=["req", "rsp", "data"]):
        """提取IP Channel数据的通用方法 - 包含internal_queue和output_register
        提取当前节点的所有IP接口数据"""
        result = {}
        for channel in channels:
            channel_data = {}
            for ip_id, ip_interface in ip_channel_buffers.items():
                if channel in ip_interface and hasattr(ip_interface[channel], "internal_queue"):
                    # 提取internal_queue中的flit
                    fifo_data = [self._extract_flit_data(flit, channel, direction_type) for flit in ip_interface[channel].internal_queue]

                    # 提取output_register中的flit（如果存在且有效）
                    # 输出寄存器的flit应该在队列第一个位置（下一个要输出的flit）
                    fifo = ip_interface[channel]
                    if hasattr(fifo, "output_register") and hasattr(fifo, "output_valid") and fifo.output_valid and fifo.output_register:
                        output_flit_data = self._extract_flit_data(fifo.output_register, channel, direction_type)
                        fifo_data.insert(0, output_flit_data)  # 插入到队列开头而不是末尾

                    channel_data[ip_id] = fifo_data
            result[channel] = channel_data
        return result

    def save_history_snapshot(self, network, cycle):
        """保存节点历史快照 - 优化版本，减少重复遍历"""
        try:
            nodes_snapshot = {}

            if hasattr(network, "nodes"):
                for node_id, node in network.nodes.items():
                    node_data = {
                        "inject_queues": {},
                        "eject_queues": {},
                        "ring_bridge": {},
                        "iq_channels": {},
                        "eq_channels": {},
                        "crosspoint_h": {},
                        "crosspoint_v": {},
                        "metadata": {"node_id": node_id, "timestamp": cycle},
                    }

                    # 1. 保存Inject Queue数据（使用通用方法）
                    try:
                        if hasattr(node, "inject_queue") and hasattr(node.inject_queue, "inject_input_fifos"):
                            node_data["inject_queues"] = self._extract_fifo_data(node.inject_queue.inject_input_fifos, node_id)
                    except:
                        node_data["inject_queues"] = {}

                    # 2. 保存Eject Queue数据（使用通用方法）
                    try:
                        if hasattr(node, "eject_queue") and hasattr(node.eject_queue, "eject_input_fifos"):
                            node_data["eject_queues"] = self._extract_fifo_data(node.eject_queue.eject_input_fifos, node_id)
                    except:
                        node_data["eject_queues"] = {}

                    # 3. 保存Ring Bridge数据（使用通用方法）
                    try:
                        if hasattr(node, "ring_bridge"):
                            ring_bridge = node.ring_bridge
                            rb_data = {}
                            # 处理input和output FIFO
                            for fifo_type, attr_name in [("_in", "ring_bridge_input_fifos"), ("_out", "ring_bridge_output_fifos")]:
                                if hasattr(ring_bridge, attr_name):
                                    fifos = getattr(ring_bridge, attr_name)
                                    # 使用通用方法提取FIFO数据
                                    extracted_data = self._extract_fifo_data(fifos, node_id)
                                    # 重新组织数据格式以匹配原有的命名约定
                                    for channel, channel_data in extracted_data.items():
                                        if channel not in rb_data:
                                            rb_data[channel] = {}
                                        for direction, data in channel_data.items():
                                            rb_data[channel][f"{direction}{fifo_type}"] = data
                            node_data["ring_bridge"] = rb_data
                    except:
                        node_data["ring_bridge"] = {}

                    # 4. 保存IP Channel数据（使用通用方法，保存当前节点的所有IP接口数据）
                    try:
                        if hasattr(node, "ip_inject_channel_buffers"):
                            node_data["iq_channels"] = self._extract_ip_channel_data(node.ip_inject_channel_buffers, "inject")
                        else:
                            node_data["iq_channels"] = {}

                        if hasattr(node, "ip_eject_channel_buffers"):
                            node_data["eq_channels"] = self._extract_ip_channel_data(node.ip_eject_channel_buffers, "eject")
                        else:
                            node_data["eq_channels"] = {}
                    except:
                        node_data["iq_channels"] = {}
                        node_data["eq_channels"] = {}

                    # 5. 保存CrossPoint数据（通用处理）
                    try:
                        for cp_name, attr_name in [("crosspoint_h", "horizontal_crosspoint"), ("crosspoint_v", "vertical_crosspoint")]:
                            if hasattr(node, attr_name):
                                cp = getattr(node, attr_name)
                                cp_data = {}
                                if hasattr(cp, "slice_connections"):
                                    # 使用当前通道的数据，默认为data
                                    current_channel = getattr(self.parent, "current_channel", "data") if self.parent else "data"

                                    for direction, channels in cp.slice_connections.items():
                                        # CrossPoint数据结构: slice_connections[direction][channel] = {"arrival": RingSlice, "departure": RingSlice}
                                        slices = channels.get(current_channel, {})
                                        arrival_slice = slices.get("arrival")
                                        departure_slice = slices.get("departure")

                                        # 从RingSlice对象中提取slot，使用新的接口
                                        arrival_slot = arrival_slice.peek_current_slot(current_channel) if (arrival_slice and hasattr(arrival_slice, "peek_current_slot")) else None
                                        departure_slot = departure_slice.peek_current_slot(current_channel) if (departure_slice and hasattr(departure_slice, "peek_current_slot")) else None

                                        # 从slot中提取实际的flit数据
                                        arrival_flit = getattr(arrival_slot, "flit", None) if (arrival_slot and getattr(arrival_slot, "valid", False)) else None
                                        departure_flit = getattr(departure_slot, "flit", None) if (departure_slot and getattr(departure_slot, "valid", False)) else None

                                        # 转换为列表格式（单个flit或None转为列表）
                                        arrival_slots = [arrival_flit] if arrival_flit else []
                                        departure_slots = [departure_flit] if departure_flit else []

                                        cp_data[direction] = [arrival_slots, departure_slots]
                                node_data[cp_name] = cp_data
                    except Exception as e:
                        print(f"⚠️ CrossPoint数据保存异常: {e}")
                        node_data["crosspoint_h"] = {}
                        node_data["crosspoint_v"] = {}

                    nodes_snapshot[node_id] = node_data

            # 保存优化后的完整快照
            snapshot_data = {
                "cycle": cycle,
                "timestamp": cycle,
                "nodes": nodes_snapshot,
                "metadata": {"total_nodes": len(nodes_snapshot), "channels": ["req", "rsp", "data"], "optimized": True},
            }

            self.node_history.append((cycle, snapshot_data))

        except Exception as e:
            # 静默忽略快照保存错误，但保留基本结构
            fallback_snapshot = {"cycle": cycle, "nodes": {}, "metadata": {"error": True, "error_msg": str(e)}}
            self.node_history.append((cycle, fallback_snapshot))

    def _init_gpu_rendering(self):
        """初始化GPU渲染组件"""
        try:
            import plotly.graph_objects as go
            self.gpu_available = True
            self.plotly_traces = {}
            self.webgl_config = {
                'toImageButtonOptions': {'format': 'png'},
                'displayModeBar': True,
                'displaylogo': False,
                'scrollZoom': True
            }
            print("🚀 CrossRing节点可视化器GPU加速已启用")
        except ImportError:
            print("⚠️  plotly不可用，节点可视化器降级到CPU模式")
            self.gpu_available = False
            self.gpu_mode = False

    def render_node_state_gpu(self, node_id, node_state):
        """GPU加速的节点状态渲染"""
        if not self.gpu_mode or not hasattr(self, 'gpu_available') or not self.gpu_available:
            # 降级到原有实现
            return self.render_node_from_snapshot(node_id, node_state.get('cycle', 0))
        
        try:
            import plotly.graph_objects as go
            
            fig = go.Figure()
            
            # FIFO队列可视化 - GPU并行渲染
            self._render_fifos_gpu(fig, node_state)
            
            # CrossPoint状态 - GPU渲染
            self._render_crosspoint_gpu(fig, node_state)
            
            # 配置布局
            fig.update_layout(
                title=f"节点 {node_id} 内部状态 (GPU加速)",
                showlegend=True,
                hovermode='closest',
                margin=dict(b=20,l=5,r=5,t=40),
                xaxis=dict(showgrid=True, zeroline=False),
                yaxis=dict(showgrid=True, zeroline=False),
                plot_bgcolor='white',
                paper_bgcolor='white'
            )
            
            return fig
            
        except Exception as e:
            print(f"⚠️  GPU渲染失败，降级到CPU模式: {e}")
            return self.render_node_from_snapshot(node_id, node_state.get('cycle', 0))
    
    def _render_fifos_gpu(self, fig, node_state):
        """GPU加速的FIFO渲染 - 改进版本"""
        fifo_states = node_state.get('fifo_states', {})
        
        # FIFO类型分组
        fifo_groups = {
            'inject': [],
            'eject': [],
            'ring_bridge': [],
            'channel_buffer': []
        }
        
        # 分类FIFO
        for fifo_name, fifo_data in fifo_states.items():
            if 'inject' in fifo_name.lower():
                fifo_groups['inject'].append((fifo_name, fifo_data))
            elif 'eject' in fifo_name.lower():
                fifo_groups['eject'].append((fifo_name, fifo_data))
            elif 'ring_bridge' in fifo_name.lower() or 'rb_' in fifo_name.lower():
                fifo_groups['ring_bridge'].append((fifo_name, fifo_data))
            else:
                fifo_groups['channel_buffer'].append((fifo_name, fifo_data))
        
        # 渲染每个组的FIFO
        y_offset = 0
        group_colors = {
            'inject': '#FF6B6B',      # 红色系 - 注入
            'eject': '#4ECDC4',       # 青色系 - 弹出  
            'ring_bridge': '#45B7D1', # 蓝色系 - 环桥
            'channel_buffer': '#96CEB4' # 绿色系 - 通道缓冲
        }
        
        for group_name, fifos in fifo_groups.items():
            if not fifos:
                continue
                
            for i, (fifo_name, fifo_data) in enumerate(fifos):
                # 计算FIFO利用率
                if isinstance(fifo_data, dict):
                    depth = fifo_data.get('depth', 8)
                    current_count = fifo_data.get('current_count', 0)
                elif isinstance(fifo_data, list):
                    depth = 8  # 默认深度
                    current_count = len(fifo_data)
                else:
                    depth = 8
                    current_count = 0
                
                utilization = current_count / max(depth, 1)
                
                # 可视化FIFO slots
                x_coords = list(range(depth))
                y_coords = [y_offset] * depth
                
                # 根据占用情况设置颜色
                colors = []
                for j in range(depth):
                    if j < current_count:
                        colors.append(group_colors[group_name])  # 占用的slot
                    else:
                        colors.append('lightgray')  # 空的slot
                
                fig.add_trace(go.Scatter(
                    x=x_coords,
                    y=y_coords,
                    mode='markers',
                    marker=dict(
                        size=15,
                        color=colors,
                        line=dict(width=1, color='black'),
                        symbol='square'
                    ),
                    name=f'{fifo_name} ({current_count}/{depth})',
                    hovertemplate=f'{fifo_name}<br>利用率: {utilization:.1%}<br>占用: {current_count}/{depth}<extra></extra>'
                ))
                
                y_offset += 1
    
    def _render_crosspoint_gpu(self, fig, node_state):
        """GPU加速的CrossPoint渲染 - 改进版本"""
        crosspoint_state = node_state.get('crosspoint_state', {})
        
        if crosspoint_state:
            # 环形slice状态可视化
            ring_slices = crosspoint_state.get('ring_slices', {})
            
            # 为不同方向的环路slice创建可视化
            directions = ['horizontal', 'vertical']
            slice_colors = {
                'horizontal': '#FF6B35',  # 橙色 - 水平环
                'vertical': '#7209B7'     # 紫色 - 垂直环
            }
            
            for direction in directions:
                if direction in ring_slices:
                    slice_data = ring_slices[direction]
                    
                    # 每个环路有多个slice (通常8个)
                    slice_count = slice_data.get('slice_count', 8)
                    active_slices = slice_data.get('active_slices', [])
                    
                    # 创建环形布局的slice可视化
                    if direction == 'horizontal':
                        # 水平环 - 线性布局
                        x_coords = list(range(slice_count))
                        y_coords = [10] * slice_count  # 固定y位置
                    else:
                        # 垂直环 - 垂直布局  
                        x_coords = [15] * slice_count  # 固定x位置
                        y_coords = list(range(slice_count))
                    
                    # 根据slice活跃状态设置颜色
                    colors = []
                    for i in range(slice_count):
                        if i in active_slices:
                            colors.append(slice_colors[direction])
                        else:
                            colors.append('lightgray')
                    
                    fig.add_trace(go.Scatter(
                        x=x_coords,
                        y=y_coords,
                        mode='markers',
                        marker=dict(
                            size=12,
                            color=colors,
                            line=dict(width=2, color='black'),
                            symbol='circle'
                        ),
                        name=f'{direction.title()} Ring',
                        hovertemplate=f'{direction} Ring<br>Slice: %{{pointNumber}}<br>Status: Active/Idle<extra></extra>'
                    ))
            
            # 添加CrossPoint仲裁状态指示器
            arbitration_active = crosspoint_state.get('arbitration_active', False)
            connections = crosspoint_state.get('active_connections', 0)
            
            fig.add_trace(go.Scatter(
                x=[10], y=[15],
                mode='markers+text',
                marker=dict(
                    size=25,
                    color='red' if arbitration_active else 'green',
                    symbol='diamond',
                    line=dict(width=3, color='black')
                ),
                text=[f'CP\n{connections}'],
                textposition='middle center',
                name='CrossPoint核心',
                hovertemplate=f'CrossPoint<br>活跃连接: {connections}<br>仲裁状态: {"活跃" if arbitration_active else "空闲"}<extra></extra>'
            ))

    def render_node_from_snapshot(self, node_id, cycle):
        """从快照数据渲染节点"""
        try:
            # 查找对应周期的历史数据
            history_snapshot = None
            for hist_cycle, snapshot_data in self.node_history:
                if hist_cycle == cycle:
                    history_snapshot = snapshot_data
                    break

            if history_snapshot:
                # 直接使用统一格式：从完整快照中提取当前节点和当前通道的数据
                nodes_data = history_snapshot.get("nodes", {})
                node_data = nodes_data.get(node_id)

                if node_data:
                    # 获取当前显示的通道
                    current_channel = getattr(self.parent, "current_channel", "data") if self.parent else "data"
                    # 直接从快照数据渲染节点
                    self._render_from_snapshot_data(node_id, node_data, current_channel)
                else:
                    self._show_no_data_message(node_id, "节点数据不存在")
            else:
                self._show_no_data_message(node_id, "无历史数据")

        except Exception as e:
            self._show_no_data_message(node_id, f"历史数据错误: {str(e)}")

    def _clear_all_components(self, current_channel):
        """清空所有组件的显示"""
        # 清空IQ的所有lanes
        for lane_name, patches in self.iq_patches.items():
            if patches:
                self._clear_fifo_patches(patches, self.iq_texts.get(lane_name, []))

        # 清空EQ的所有lanes
        for lane_name, patches in self.eq_patches.items():
            if patches:
                self._clear_fifo_patches(patches, self.eq_texts.get(lane_name, []))

        # 清空RB的所有lanes
        for lane_name, patches in self.rb_patches.items():
            if patches:
                self._clear_fifo_patches(patches, self.rb_texts.get(lane_name, []))

        # 清空CrossPoint
        for lane_name, patches in self.cph_patches.items():
            if patches:
                self._clear_fifo_patches(patches, self.cph_texts.get(lane_name, []))

        for lane_name, patches in self.cpv_patches.items():
            if patches:
                self._clear_fifo_patches(patches, self.cpv_texts.get(lane_name, []))

    def _render_from_snapshot_data(self, node_id, node_data, current_channel):
        """直接从快照数据渲染节点组件"""
        # 清空旧的 patch->info 映射
        self.patch_info_map.clear()
        # 本帧尚未发现高亮 flit
        self.current_highlight_flit = None

        # 如果轴内无任何图元，说明已被 clear()，需要重新画框架
        if len(self.ax.patches) == 0:
            self._draw_modules()  # 重建 FIFO / RB 边框与槽

        self.node_id = node_id

        # 先清空所有组件的显示（确保没有数据时也能清空）
        self._clear_all_components(current_channel)

        # 直接从快照数据渲染各个组件
        try:
            # 1. 渲染 Inject Queues
            inject_queues = node_data.get("inject_queues", {})
            channel_data = inject_queues.get(current_channel, {})
            self._render_component_from_snapshot("IQ", channel_data, node_id)

            # 2. 渲染 Eject Queues
            eject_queues = node_data.get("eject_queues", {})
            channel_data = eject_queues.get(current_channel, {})
            self._render_component_from_snapshot("EQ", channel_data, node_id)

            # 3. 渲染 Ring Bridge
            ring_bridge = node_data.get("ring_bridge", {})
            channel_data = ring_bridge.get(current_channel, {})
            self._render_component_from_snapshot("RB", channel_data, node_id)

            # 4. 渲染 IP Channels
            iq_channels = node_data.get("iq_channels", {})
            eq_channels = node_data.get("eq_channels", {})

            # 总是调用渲染函数，即使没有数据（函数内部会处理清空）
            self._render_ip_channels_from_snapshot("IQ_Ch", iq_channels.get(current_channel, {}), node_id)
            self._render_ip_channels_from_snapshot("EQ_Ch", eq_channels.get(current_channel, {}), node_id)

            # 5. 渲染 CrossPoint (不区分通道，直接使用原始数据)
            crosspoint_h = node_data.get("crosspoint_h", {})
            crosspoint_v = node_data.get("crosspoint_v", {})

            if crosspoint_h:
                self._render_component_from_snapshot("CP_H", crosspoint_h, node_id)

            if crosspoint_v:
                self._render_component_from_snapshot("CP_V", crosspoint_v, node_id)

        except Exception as e:
            # 渲染失败时显示错误信息
            self._show_no_data_message(node_id, f"渲染错误: {str(e)}")

        # 触发重绘以更新显示
        self.fig.canvas.draw_idle()

    def _render_component_from_snapshot(self, component_type, channel_data, node_id):
        """从快照数据渲染指定组件的所有方向"""
        if not channel_data:
            return

        # 根据组件类型确定需要处理的方向
        if component_type in ["IQ", "EQ"]:
            # 注意：IQ包含方向lanes（如TU、TD）和IP通道lanes（如gdma_0、ddr_0）
            # 这里只处理方向lanes，IP通道lanes由_render_ip_channels_from_snapshot处理
            directions = ["TR", "TL", "TU", "TD", "EQ"]  # 添加EQ方向，因为IQ配置中有
        elif component_type == "RB":
            directions = ["TL_in", "TR_in", "TU_out", "TD_out", "EQ_out"]  # 匹配新的配置
        elif component_type == "CP_H":
            directions = ["TL", "TR"]  # 水平CrossPoint处理TL/TR方向，与实际数据结构一致
        elif component_type == "CP_V":
            directions = ["TU", "TD"]  # 垂直CrossPoint处理TU/TD方向，与实际数据结构一致
        else:
            return

        # 渲染每个方向的数据
        for direction in directions:
            if direction in channel_data:
                direction_data = channel_data[direction]

                # 根据组件类型直接操作patch属性
                if component_type == "IQ":
                    if node_id in direction_data:
                        # 检查这个方向是否存在于iq_patches中
                        if direction in self.iq_patches:
                            self._render_fifo_patches(self.iq_patches, self.iq_texts, direction, direction_data[node_id])
                elif component_type == "EQ":
                    if node_id in direction_data:
                        # 检查这个方向是否存在于eq_patches中
                        if direction in self.eq_patches:
                            self._render_fifo_patches(self.eq_patches, self.eq_texts, direction, direction_data[node_id])
                elif component_type == "RB":
                    if node_id in direction_data:
                        # Ring Bridge现在直接使用完整的direction名称
                        self._render_fifo_patches(self.rb_patches, self.rb_texts, direction, direction_data[node_id])
                elif component_type == "CP_H":
                    # 水平CrossPoint需要将数据映射到新的lane名称
                    self._render_crosspoint_patches_split(self.cph_patches, self.cph_texts, direction, direction_data)
                elif component_type == "CP_V":
                    # 垂直CrossPoint需要将数据映射到新的lane名称
                    self._render_crosspoint_patches_split(self.cpv_patches, self.cpv_texts, direction, direction_data)

    def _render_ip_channels_from_snapshot(self, channel_type, channel_data, node_id):
        """从快照数据渲染IP通道数据"""
        # 获取通道名称配置
        ch_names = getattr(self.config, "CH_NAME_LIST", ["gdma", "ddr"])

        # 注意：清空操作已经在_clear_all_components中完成

        if not channel_data:
            # 即使没有数据也要返回，因为清空操作已经完成
            return

        # IP通道数据：每个节点的数据中包含该节点的IP接口
        # ip_interface_key可能是复合键，如 "0_gdma", "1_ddr" 等
        for ip_interface_key, flit_list in channel_data.items():
            # 尝试从键中提取通道类型
            lane_name = None

            # 检查是否直接匹配配置的通道名称
            for ch_name in ch_names:
                if ch_name in str(ip_interface_key):
                    lane_name = ch_name
                    break

            # 如果没有找到匹配，尝试使用索引映射
            if lane_name is None:
                # 提取节点内的IP索引
                # 键格式可能是 "0", "1" 或 "node0_ip0" 等
                parts = str(ip_interface_key).split("_")
                for part in parts:
                    if part.isdigit():
                        ip_index = int(part)
                        # 计算该节点内的本地IP索引
                        local_index = ip_index % len(ch_names)
                        if local_index < len(ch_names):
                            lane_name = ch_names[local_index]
                            break

            if lane_name:
                if channel_type == "IQ_Ch":
                    self._render_fifo_patches(self.iq_patches, self.iq_texts, lane_name, flit_list)
                elif channel_type == "EQ_Ch":
                    self._render_fifo_patches(self.eq_patches, self.eq_texts, lane_name, flit_list)

    def _clear_fifo_patches(self, patches, texts):
        """清空FIFO patch的显示"""
        for p in patches:
            p.set_facecolor("none")
            p.set_linewidth(0)
            p.set_edgecolor("none")
            # 从patch_info_map中移除
            if hasattr(self, "patch_info_map") and p in self.patch_info_map:
                del self.patch_info_map[p]

        for t in texts:
            t.set_visible(False)

    def _render_fifo_patches(self, patch_dict, text_dict, lane_name, flit_list):
        """渲染FIFO类型patch的flit数据"""
        if lane_name not in patch_dict or lane_name not in text_dict:
            return

        patches = patch_dict[lane_name]
        texts = text_dict[lane_name]

        # 清空所有patch并移除映射
        for p in patches:
            p.set_facecolor("none")
            p.set_linewidth(0)
            p.set_edgecolor("none")
            # 从patch_info_map中移除
            if p in self.patch_info_map:
                del self.patch_info_map[p]

        for t in texts:
            t.set_visible(False)

        # 渲染flit数据
        for idx, flit in enumerate(flit_list):
            if idx >= len(patches):
                break

            p = patches[idx]
            t = texts[idx]

            if flit:
                # 兼容字典和对象两种格式
                if isinstance(flit, dict):
                    packet_id = flit.get("packet_id", None)
                    flit_id = flit.get("flit_id", str(flit))
                else:
                    packet_id = getattr(flit, "packet_id", None)
                    flit_id = getattr(flit, "flit_id", str(flit))

                face, lw, edge = self._get_flit_style(
                    flit,
                    use_highlight=self.use_highlight,
                    expected_packet_id=self.highlight_pid,
                )
                p.set_facecolor(face)
                p.set_linewidth(lw)
                p.set_edgecolor(edge)

                info = f"{packet_id}-{flit_id}"
                t.set_text(info)
                t.set_visible(self.use_highlight and packet_id == self.highlight_pid)
                self.patch_info_map[p] = (t, flit)

                if self.use_highlight and getattr(flit, "packet_id", None) == self.highlight_pid:
                    self.current_highlight_flit = flit
            else:
                if p in self.patch_info_map:
                    self.patch_info_map.pop(p, None)

    def _render_crosspoint_patches(self, patch_dict, text_dict, direction, slice_data):
        """渲染CrossPoint类型patch的slice数据"""
        if direction not in patch_dict or direction not in text_dict:
            print(f"🚫 调试: CrossPoint {direction}方向 patches或texts未找到")
            return

        patches = patch_dict[direction]
        texts = text_dict[direction]

        # CrossPoint数据结构: [arrival_slots, departure_slots]
        if not isinstance(slice_data, list) or len(slice_data) < 2:
            print(f"🚫 调试: CrossPoint {direction}方向 slice_data格式错误: {slice_data}")
            return

        arrival_slots = slice_data[0] if slice_data[0] else []
        departure_slots = slice_data[1] if slice_data[1] else []

        # TR和TU方向需要颠倒顺序：departure在前，arrival在后
        if direction in ["TR", "TU"]:
            all_slots = departure_slots + arrival_slots
        else:
            # TL和TD方向保持原顺序：arrival在前，departure在后
            all_slots = arrival_slots + departure_slots

        print(f"🎯 调试: CrossPoint {direction}方向 arrival_slots={len(arrival_slots)} departure_slots={len(departure_slots)} all_slots={len(all_slots)}")
        for i, flit in enumerate(all_slots):
            if flit:
                print(f"   - slot[{i}]: pid={getattr(flit, 'packet_id', 'N/A')} fid={getattr(flit, 'flit_id', 'N/A')}")

        # 清空所有patch并移除映射
        for p in patches:
            p.set_facecolor("none")
            p.set_linewidth(0)
            p.set_edgecolor("none")
            # 从patch_info_map中移除
            if p in self.patch_info_map:
                del self.patch_info_map[p]

        for t in texts:
            t.set_visible(False)

        # 渲染slot数据
        for idx, flit in enumerate(all_slots):
            if idx >= len(patches):
                break

            p = patches[idx]
            t = texts[idx]

            if flit:
                packet_id = getattr(flit, "packet_id", None)
                flit_id = getattr(flit, "flit_id", str(flit))

                face, lw, edge = self._get_flit_style(
                    flit,
                    use_highlight=self.use_highlight,
                    expected_packet_id=self.highlight_pid,
                )
                p.set_facecolor(face)
                p.set_linewidth(lw)
                p.set_edgecolor(edge)

                info = f"{packet_id}-{flit_id}"
                t.set_text(info)
                t.set_visible(self.use_highlight and packet_id == self.highlight_pid)
                self.patch_info_map[p] = (t, flit)

                if self.use_highlight and getattr(flit, "packet_id", None) == self.highlight_pid:
                    self.current_highlight_flit = flit
            else:
                if p in self.patch_info_map:
                    self.patch_info_map.pop(p, None)

    def _render_crosspoint_patches_split(self, patch_dict, text_dict, direction, slice_data):
        """渲染CrossPoint类型patch的slice数据 - 拆分版本"""
        # CrossPoint数据结构: [arrival_slots, departure_slots]
        if not isinstance(slice_data, list) or len(slice_data) < 2:
            print(f"🚫 调试: CrossPoint {direction}方向 slice_data格式错误: {slice_data}")
            return

        arrival_slots = slice_data[0] if slice_data[0] else []
        departure_slots = slice_data[1] if slice_data[1] else []

        # 将arrival和departure分别渲染到对应的lane
        arr_lane = f"{direction}_arr"
        dep_lane = f"{direction}_dep"

        # 渲染arrival slot
        if arr_lane in patch_dict and arr_lane in text_dict:
            self._render_single_slot(patch_dict[arr_lane], text_dict[arr_lane], arrival_slots)

        # 渲染departure slot
        if dep_lane in patch_dict and dep_lane in text_dict:
            self._render_single_slot(patch_dict[dep_lane], text_dict[dep_lane], departure_slots)

    def _render_single_slot(self, patches, texts, slot_data):
        """渲染单个slot的数据"""
        # 清空所有patch并移除映射
        for p in patches:
            p.set_facecolor("none")
            p.set_linewidth(0)
            p.set_edgecolor("none")
            # 从patch_info_map中移除
            if p in self.patch_info_map:
                del self.patch_info_map[p]

        for t in texts:
            t.set_visible(False)

        # 只渲染第一个slot（因为每个lane现在只有1个深度）
        if slot_data and len(patches) > 0:
            flit = slot_data[0] if slot_data else None
            p = patches[0]
            t = texts[0]

            if flit:
                packet_id = getattr(flit, "packet_id", None)
                flit_id = getattr(flit, "flit_id", str(flit))

                face, lw, edge = self._get_flit_style(
                    flit,
                    use_highlight=self.use_highlight,
                    expected_packet_id=self.highlight_pid,
                )
                p.set_facecolor(face)
                p.set_linewidth(lw)
                p.set_edgecolor(edge)

                info = f"{packet_id}-{flit_id}"
                t.set_text(info)
                t.set_visible(self.use_highlight and packet_id == self.highlight_pid)
                self.patch_info_map[p] = (t, flit)

                if self.use_highlight and getattr(flit, "packet_id", None) == self.highlight_pid:
                    self.current_highlight_flit = flit
            else:
                if p in self.patch_info_map:
                    self.patch_info_map.pop(p, None)

    def _show_no_data_message(self, node_id, message):
        """显示无数据消息"""
        self.ax.clear()
        self.ax.text(0.5, 0.5, f"节点 {node_id}\n{message}", ha="center", va="center", transform=self.ax.transAxes, fontsize=12, family="sans-serif")
