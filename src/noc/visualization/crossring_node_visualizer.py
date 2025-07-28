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

# 字体支持 - 中文用SimHei，英文用Times
plt.rcParams["font.sans-serif"] = ["SimHei", "Times New Roman", "Arial Unicode MS"]
plt.rcParams["font.serif"] = ["Times New Roman", "Times", "serif"]
plt.rcParams["axes.unicode_minus"] = False


class CrossRingNodeVisualizer:
    """节点详细视图可视化器（右侧面板）"""

    def __init__(self, config: CrossRingConfig, ax, highlight_callback=None, parent=None):
        """
        仅绘制单个节点的 Inject/Eject Queue 和 Ring Bridge FIFO。
        参数:
        - config: 含有 FIFO 深度配置的对象，属性包括 cols, num_nodes, IQ_OUT_FIFO_DEPTH,
            EQ_IN_FIFO_DEPTH, RB_IN_FIFO_DEPTH, RB_OUT_FIFO_DEPTH
        - node_id: 要可视化的节点索引 (0 到 num_nodes-1)
        """
        self.highlight_callback = highlight_callback
        self.config = config
        self.cols = config.NUM_COL
        self.rows = config.NUM_ROW
        self.parent = parent

        # 提取深度
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
        ch_names = self.config.CH_NAME_LIST

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
            lanes=["TL", "TR", "TU", "TD", "EQ"],
            depths=[self.RB_IN_DEPTH] * 2 + [self.RB_OUT_DEPTH] * 3,
            orientations=["vertical", "vertical", "horizontal", "horizontal", "vertical"],
            h_pos=["bottom", "bottom", "top", "top", "top"],
            v_pos=["left", "left", "right", "right", "left"],
            patch_dict=self.rb_patches,
            text_dict=self.rb_texts,
        )

        cross_point_horizontal_config = dict(
            title="CP",
            lanes=["TR", "TL"],
            depths=[2, 2],
            orientations=["horizontal", "horizontal"],
            h_pos=["bottom", "bottom"],
            v_pos=["right", "right"],
            patch_dict=self.cph_patches,
            text_dict=self.cph_texts,
        )

        cross_point_vertical_config = dict(
            title="CP",
            lanes=["TD", "TU"],
            depths=[2, 2],
            orientations=["vertical", "vertical"],
            h_pos=["bottom", "bottom"],
            v_pos=["left", "left"],
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
        self.ax.set_ylim(min_y - margin * 3, max_y + margin * 3)

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
                if lane[:2] in ["TL", "TR", "TU", "TD", "EQ"]:
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
                        (slot_x + square * 0.12, slot_y + square * 0.12),
                        square * 0.76,
                        square * 0.76,
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
                    lane_y = y - (depth / 2) * (square + gap)
                    slot_dir = 1
                    va = "center"
                else:
                    raise ValueError(f"Unknown h_position: {hpos}")

                if lane[:2] in ["TL", "TR", "TU", "TD", "EQ"]:
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
                        (slot_x + square * 0.12, slot_y + square * 0.12),
                        square * 0.76,
                        square * 0.76,
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

    def _draw_fifo(self, x, y, depth, orientation, lane, patch_dict, text_dict):
        """绘制单个FIFO"""
        patches = []
        texts = []

        for i in range(depth):
            if orientation == "vertical":
                slot_x = x
                slot_y = y + (i - depth / 2 + 0.5) * (self.square + self.gap)
            else:  # horizontal
                slot_x = x + (i - depth / 2 + 0.5) * (self.square + self.gap)
                slot_y = y

            # 创建slot矩形
            rect = Rectangle((slot_x - self.square / 2, slot_y - self.square / 2), self.square, self.square, facecolor="white", edgecolor="black", linewidth=self.slot_frame_lw)
            self.ax.add_patch(rect)
            patches.append(rect)

            # 创建文本
            text = self.ax.text(slot_x, slot_y, "", fontsize=self.fontsize, ha="center", va="center", weight="bold", family="serif")
            texts.append(text)

        # 添加FIFO标签
        if orientation == "vertical":
            label_x = x + self.square / 2 + self.text_gap
            label_y = y
        else:
            label_x = x
            label_y = y - self.square / 2 - self.text_gap

        self.ax.text(
            label_x,
            label_y,
            lane,
            fontsize=self.fontsize,
            ha="left" if orientation == "vertical" else "center",
            va="center" if orientation == "vertical" else "top",
            rotation=0 if orientation == "vertical" else 0,
            family="serif",
        )

        patch_dict[lane] = patches
        text_dict[lane] = texts

    def draw_node(self, node_id, network):
        """绘制指定节点的详细视图"""
        # 清空旧的 patch->info 映射
        self.patch_info_map.clear()
        # 本帧尚未发现高亮 flit
        self.current_highlight_flit = None

        # 如果轴内无任何图元，说明已被 clear()，需要重新画框架
        if len(self.ax.patches) == 0:
            self._draw_modules()  # 重建 FIFO / RB 边框与槽

        self.node_id = node_id

        # 模拟数据结构 - 这里需要适配实际的network数据结构
        # 原版从network中提取以下数据：
        # IQ = network.inject_queues
        # EQ = network.eject_queues
        # RB = network.ring_bridge
        # IQ_Ch = network.IQ_channel_buffer
        # EQ_Ch = network.EQ_channel_buffer
        # CP_H = network.cross_point["horizontal"]
        # CP_V = network.cross_point["vertical"]

        # 这里我们先用模拟数据，之后可以适配实际的数据结构
        IQ = self._get_inject_queues_data(network, node_id)
        EQ = self._get_eject_queues_data(network, node_id)
        RB = self._get_ring_bridge_data(network, node_id)
        IQ_Ch = self._get_iq_channel_data(network, node_id)
        EQ_Ch = self._get_eq_channel_data(network, node_id)
        CP_H = self._get_crosspoint_data(network, node_id, "horizontal")
        CP_V = self._get_crosspoint_data(network, node_id, "vertical")

        # 更新Inject Queue显示
        for lane, patches in self.iq_patches.items():
            if "_" in lane:
                q = IQ_Ch.get(lane, {}).get(self.node_id, [])
            else:
                q = IQ.get(lane, {}).get(self.node_id, [])

            for idx, p in enumerate(patches):
                t = self.iq_texts[lane][idx]
                if idx < len(q):
                    flit = q[idx]
                    packet_id = getattr(flit, "packet_id", None)
                    flit_id = getattr(flit, "flit_id", str(flit))

                    # 设置颜色（基于packet_id）和显示文本
                    face, alpha, lw, edge = self._get_flit_style(
                        flit,
                        use_highlight=self.use_highlight,
                        expected_packet_id=self.highlight_pid,
                    )
                    p.set_facecolor(face)
                    p.set_alpha(alpha)
                    p.set_linewidth(lw)
                    p.set_edgecolor(edge)

                    info = f"{packet_id}-{flit_id}"
                    t.set_text(info)
                    t.set_visible(self.use_highlight and packet_id == self.highlight_pid)
                    self.patch_info_map[p] = (t, flit)

                    # 若匹配追踪的 packet_id，记录以便结束后刷新 info_text
                    if self.use_highlight and getattr(flit, "packet_id", None) == self.highlight_pid:
                        self.current_highlight_flit = flit
                else:
                    p.set_facecolor("none")
                    t.set_visible(False)
                    if p in self.patch_info_map:
                        self.patch_info_map.pop(p, None)

        # 更新Eject Queue显示（类似逻辑）
        for lane, patches in self.eq_patches.items():
            if "_" in lane:
                q = EQ_Ch.get(lane, {}).get(self.node_id - self.cols, [])
            else:
                q = EQ.get(lane, {}).get(self.node_id - self.cols, [])

            for idx, p in enumerate(patches):
                t = self.eq_texts[lane][idx]
                if idx < len(q):
                    flit = q[idx]
                    packet_id = getattr(flit, "packet_id", None)
                    flit_id = getattr(flit, "flit_id", str(flit))

                    face, alpha, lw, edge = self._get_flit_style(
                        flit,
                        use_highlight=self.use_highlight,
                        expected_packet_id=self.highlight_pid,
                    )
                    p.set_facecolor(face)
                    p.set_alpha(alpha)
                    p.set_linewidth(lw)
                    p.set_edgecolor(edge)

                    info = f"{packet_id}-{flit_id}"
                    t.set_text(info)
                    t.set_visible(self.use_highlight and packet_id == self.highlight_pid)
                    self.patch_info_map[p] = (t, flit)

                    if self.use_highlight and getattr(flit, "packet_id", None) == self.highlight_pid:
                        self.current_highlight_flit = flit
                else:
                    p.set_facecolor("none")
                    t.set_visible(False)
                    if p in self.patch_info_map:
                        self.patch_info_map.pop(p, None)

        # 更新Ring Bridge显示（类似逻辑）
        for lane, patches in self.rb_patches.items():
            q = RB.get(lane, {}).get((self.node_id, self.node_id - self.cols), [])

            for idx, p in enumerate(patches):
                t = self.rb_texts[lane][idx]
                if idx < len(q):
                    flit = q[idx]
                    packet_id = getattr(flit, "packet_id", None)
                    flit_id = getattr(flit, "flit_id", str(flit))

                    face, alpha, lw, edge = self._get_flit_style(
                        flit,
                        use_highlight=self.use_highlight,
                        expected_packet_id=self.highlight_pid,
                    )
                    p.set_facecolor(face)
                    p.set_alpha(alpha)
                    p.set_linewidth(lw)
                    p.set_edgecolor(edge)

                    info = f"{packet_id}-{flit_id}"
                    t.set_text(info)
                    t.set_visible(self.use_highlight and packet_id == self.highlight_pid)
                    self.patch_info_map[p] = (t, flit)

                    if self.use_highlight and getattr(flit, "packet_id", None) == self.highlight_pid:
                        self.current_highlight_flit = flit
                else:
                    p.set_facecolor("none")
                    t.set_visible(False)
                    if p in self.patch_info_map:
                        self.patch_info_map.pop(p, None)

    def _get_inject_queues_data(self, network, node_id):
        """从网络中提取inject queue数据（适配层）"""
        inject_data = {}

        if not network or not hasattr(network, "nodes"):
            return inject_data

        try:
            node = network.nodes.get(node_id)
            if not node:
                return inject_data

            # 提取inject_direction_fifos数据
            if hasattr(node, "inject_direction_fifos"):
                for direction, fifo in node.inject_direction_fifos.items():
                    if hasattr(fifo, "queue") and fifo.queue:
                        inject_data[direction] = {node_id: list(fifo.queue)}

            # 提取channel_buffer数据
            if hasattr(node, "channel_buffer"):
                for channel_name, buffer in node.channel_buffer.items():
                    if hasattr(buffer, "queue") and buffer.queue:
                        inject_data[channel_name] = {node_id: list(buffer.queue)}

        except Exception as e:
            pass  # print(f"警告: 提取inject queue数据失败: {e}")

        return inject_data

    def _get_eject_queues_data(self, network, node_id):
        """从网络中提取eject queue数据（适配层）"""
        eject_data = {}

        if not network or not hasattr(network, "nodes"):
            return eject_data

        try:
            node = network.nodes.get(node_id)
            if not node:
                return eject_data

            # 提取eject_input_fifos数据
            if hasattr(node, "eject_input_fifos"):
                for direction, fifo in node.eject_input_fifos.items():
                    if hasattr(fifo, "queue") and fifo.queue:
                        eject_data[direction] = {node_id: list(fifo.queue)}

            # 提取ip_eject_channel_buffers数据
            if hasattr(node, "ip_eject_channel_buffers"):
                for channel_name, buffer in node.ip_eject_channel_buffers.items():
                    if hasattr(buffer, "queue") and buffer.queue:
                        eject_data[channel_name] = {node_id: list(buffer.queue)}

        except Exception as e:
            pass  # print(f"警告: 提取eject queue数据失败: {e}")

        return eject_data

    def _get_ring_bridge_data(self, network, node_id):
        """从网络中提取ring bridge数据（适配层）"""
        rb_data = {}

        if not network or not hasattr(network, "nodes"):
            return rb_data

        try:
            node = network.nodes.get(node_id)
            if not node or not hasattr(node, "ring_bridge"):
                return rb_data

            ring_bridge = node.ring_bridge

            # 提取ring_bridge input和output数据
            if hasattr(ring_bridge, "ring_bridge_input"):
                for direction, fifo in ring_bridge.ring_bridge_input.items():
                    if hasattr(fifo, "queue") and fifo.queue:
                        rb_data[f"{direction}_in"] = {(node_id, node_id): list(fifo.queue)}

            if hasattr(ring_bridge, "ring_bridge_output"):
                for direction, fifo in ring_bridge.ring_bridge_output.items():
                    if hasattr(fifo, "queue") and fifo.queue:
                        rb_data[f"{direction}_out"] = {(node_id, node_id): list(fifo.queue)}

        except Exception as e:
            pass  # print(f"警告: 提取ring bridge数据失败: {e}")

        return rb_data

    def _get_iq_channel_data(self, network, node_id):
        """从网络中提取IQ channel数据（适配层）"""
        iq_ch_data = {}

        if not network or not hasattr(network, "nodes"):
            return iq_ch_data

        try:
            node = network.nodes.get(node_id)
            if not node:
                return iq_ch_data

            # 从IP接口提取l2h_fifos数据
            if hasattr(node, "ip_interfaces"):
                for ip_name, ip_interface in node.ip_interfaces.items():
                    if hasattr(ip_interface, "l2h_fifos"):
                        for channel_name, fifo in ip_interface.l2h_fifos.items():
                            if hasattr(fifo, "queue") and fifo.queue:
                                full_channel_name = f"{ip_name}_{channel_name}"
                                iq_ch_data[full_channel_name] = {node_id: list(fifo.queue)}

        except Exception as e:
            pass  # print(f"警告: 提取IQ channel数据失败: {e}")

        return iq_ch_data

    def _get_eq_channel_data(self, network, node_id):
        """从网络中提取EQ channel数据（适配层）"""
        eq_ch_data = {}

        if not network or not hasattr(network, "nodes"):
            return eq_ch_data

        try:
            node = network.nodes.get(node_id)
            if not node:
                return eq_ch_data

            # 从IP接口提取h2l_fifos数据
            if hasattr(node, "ip_interfaces"):
                for ip_name, ip_interface in node.ip_interfaces.items():
                    if hasattr(ip_interface, "h2l_fifos"):
                        for channel_name, fifo in ip_interface.h2l_fifos.items():
                            if hasattr(fifo, "queue") and fifo.queue:
                                full_channel_name = f"{ip_name}_{channel_name}"
                                eq_ch_data[full_channel_name] = {node_id: list(fifo.queue)}

        except Exception as e:
            pass  # print(f"警告: 提取EQ channel数据失败: {e}")

        return eq_ch_data

    def _get_crosspoint_data(self, network, node_id, direction):
        """从网络中提取crosspoint数据（适配层）"""
        cp_data = {}

        if not network or not hasattr(network, "nodes"):
            return cp_data

        try:
            node = network.nodes.get(node_id)
            if not node:
                return cp_data

            # 获取对应方向的CrossPoint
            if direction == "horizontal" and hasattr(node, "horizontal_cp"):
                cp = node.horizontal_cp
            elif direction == "vertical" and hasattr(node, "vertical_cp"):
                cp = node.vertical_cp
            else:
                return cp_data

            # 提取CrossPoint状态信息
            cp_data = {
                "arbitration_state": getattr(cp, "arbitration_state", "idle"),
                "active_connections": getattr(cp, "active_connections", []),
                "priority_state": getattr(cp, "priority_state", "normal"),
            }

        except Exception as e:
            pass  # print(f"警告: 提取crosspoint数据失败: {e}")

        return cp_data

    def _get_flit_style(self, flit, use_highlight=True, expected_packet_id=0, highlight_color=None):
        """使用父类的flit样式方法"""
        return self.parent._get_flit_style(flit, use_highlight, expected_packet_id, highlight_color)

    def _get_flit_color(self, flit, use_highlight=True, expected_packet_id=1, highlight_color=None):
        """使用父类的flit颜色方法"""
        return self.parent._get_flit_color(flit, use_highlight, expected_packet_id, highlight_color)

    def _on_click(self, event):
        """处理点击事件"""
        if event.inaxes != self.ax:
            return
        for patch, (txt, flit) in self.patch_info_map.items():
            contains, _ = patch.contains(event)
            if contains:
                # 只有在高亮模式下才允许切换文本可见性
                pid = getattr(flit, "packet_id", None)
                fid = getattr(flit, "flit_id", None)
                if self.use_highlight and pid == self.highlight_pid:
                    vis = not txt.get_visible()
                    txt.set_visible(vis)
                    # 若即将显示，确保在最上层
                    if vis:
                        txt.set_zorder(patch.get_zorder() + 1)
                # 在右下角显示完整 flit 信息
                self.info_text.set_text(str(flit))
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

        # 更新所有patch的文本可见性
        for patch, (txt, flit) in self.patch_info_map.items():
            pid = getattr(flit, "packet_id", None)
            if self.use_highlight and pid == self.highlight_pid:
                txt.set_visible(True)
            else:
                txt.set_visible(False)
        if not self.use_highlight:
            self.info_text.set_text("")
