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

        # 历史保存功能
        from collections import deque

        self.node_history = deque(maxlen=50)  # 保存最近50个周期的节点状态

        # 提取深度 - 兼容不同的配置格式
        if hasattr(config, 'fifo_config'):
            self.IQ_OUT_DEPTH = config.fifo_config.IQ_OUT_FIFO_DEPTH
            self.EQ_IN_DEPTH = config.fifo_config.EQ_IN_FIFO_DEPTH
            self.RB_IN_DEPTH = config.fifo_config.RB_IN_FIFO_DEPTH
            self.RB_OUT_DEPTH = config.fifo_config.RB_OUT_FIFO_DEPTH
            self.IQ_CH_depth = config.fifo_config.IQ_CH_DEPTH
            self.EQ_CH_depth = config.fifo_config.EQ_CH_DEPTH
            self.SLICE_PER_LINK = config.basic_config.SLICE_PER_LINK
        else:
            # demo格式配置
            self.IQ_OUT_DEPTH = getattr(config, 'IQ_OUT_FIFO_DEPTH', 8)
            self.EQ_IN_DEPTH = getattr(config, 'EQ_IN_FIFO_DEPTH', 8)
            self.RB_IN_DEPTH = getattr(config, 'RB_IN_FIFO_DEPTH', 4)
            self.RB_OUT_DEPTH = getattr(config, 'RB_OUT_FIFO_DEPTH', 4)
            self.IQ_CH_depth = getattr(config, 'IQ_CH_FIFO_DEPTH', 4)
            self.EQ_CH_depth = getattr(config, 'EQ_CH_FIFO_DEPTH', 4)
            self.SLICE_PER_LINK = getattr(config, 'SLICE_PER_LINK', 8)

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
        ch_names = getattr(self.config, 'CH_NAME_LIST', ['gdma', 'ddr'])

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
            lanes=["TL", "TR"],  # 修正：与实际slice_connections的键值顺序一致
            depths=[2, 2],
            orientations=["horizontal", "horizontal"],
            h_pos=["bottom", "bottom"],
            v_pos=["right", "right"],
            patch_dict=self.cph_patches,
            text_dict=self.cph_texts,
        )

        cross_point_vertical_config = dict(
            title="CP",
            lanes=["TU", "TD"],  # 修正：与实际slice_connections的键值顺序一致
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

    # def _get_inject_queues_data(self, model, node_id):
    #     """从网络中提取inject queue数据（适配层）"""
    #     inject_data = {}

    #     if not model or not hasattr(model, "nodes"):
    #         return inject_data

    #     try:
    #         node = model.nodes.get(node_id)
    #         if not node:
    #             return inject_data

    #         # 提取IQ_OUT_FIFO数据
    #         for direction, fifo in node.inject_queue.inject_input_fifos[self.parent.current_channel].items():
    #             inject_data[direction] = {node_id: list(fifo.internal_queue)}

    #     except Exception as e:
    #         pass  # print(f"警告: 提取inject queue数据失败: {e}")

    #     return inject_data

    # def _get_eject_queues_data(self, network, node_id):
    #     """从网络中提取eject queue数据（适配层）"""
    #     eject_data = {}

    #     if not network or not hasattr(network, "nodes"):
    #         return eject_data

    #     try:
    #         node = network.nodes.get(node_id)
    #         if not node:
    #             return eject_data

    #         # 提取eject_input_fifos数据
    #         for direction, fifo in node.eject_queue.eject_input_fifos[self.parent.current_channel].items():
    #             eject_data[direction] = {node_id: list(fifo.internal_queue)}

    #     except Exception as e:
    #         pass  # print(f"警告: 提取eject queue数据失败: {e}")

    #     return eject_data

    # def _get_ring_bridge_data(self, network, node_id):
    #     """从网络中提取ring bridge数据（适配层）"""
    #     rb_data = {}

    #     if not network or not hasattr(network, "nodes"):
    #         return rb_data

    #     try:
    #         node = network.nodes.get(node_id)
    #         if not node or not hasattr(node, "ring_bridge"):
    #             return rb_data

    #         ring_bridge = node.ring_bridge

    #         # 提取ring_bridge input和output数据
    #         for direction, fifo in ring_bridge.ring_bridge_input_fifos[self.parent.current_channel].items():
    #             rb_data[f"{direction}_in"] = {(node_id, node_id): list(fifo.internal_queue)}

    #         for direction, fifo in ring_bridge.ring_bridge_output_fifos[self.parent.current_channel].items():
    #             rb_data[f"{direction}_out"] = {(node_id, node_id): list(fifo.internal_queue)}

    #     except Exception as e:
    #         pass  # print(f"警告: 提取ring bridge数据失败: {e}")

    #     return rb_data

    # def _get_iq_channel_data(self, network, node_id):
    #     """从网络中提取IQ channel数据（适配层）"""
    #     iq_ch_data = {}

    #     if not network or not hasattr(network, "nodes"):
    #         return iq_ch_data

    #     try:
    #         node = network.nodes.get(node_id)
    #         if not node:
    #             return iq_ch_data

    #         for ip_name, ip_interface in node.ip_inject_channel_buffers.items():
    #             iq_ch_data[ip_name] = {node_id: list(ip_interface[self.parent.current_channel].internal_queue)}

    #     except Exception as e:
    #         pass  # print(f"警告: 提取IQ channel数据失败: {e}")

    #     return iq_ch_data

    # def _get_eq_channel_data(self, network, node_id):
    #     """从网络中提取EQ channel数据（适配层）"""
    #     eq_ch_data = {}

    #     if not network or not hasattr(network, "nodes"):
    #         return eq_ch_data

    #     try:
    #         node = network.nodes.get(node_id)
    #         if not node:
    #             return eq_ch_data

    #         for ip_name, ip_interface in node.ip_eject_channel_buffers.items():
    #             eq_ch_data[ip_name] = {node_id: list(ip_interface[self.parent.current_channel].internal_queue)}

    #     except Exception as e:
    #         pass  # print(f"警告: 提取EQ channel数据失败: {e}")

    #     return eq_ch_data

    # def _get_crosspoint_data(self, network, node_id, direction):
    #     """从网络中提取crosspoint数据（适配层）"""
    #     cp_data = {}

    #     if not network or not hasattr(network, "nodes"):
    #         return cp_data

    #     try:
    #         node = network.nodes.get(node_id)
    #         if not node:
    #             return cp_data

    #         # 获取对应方向的CrossPoint
    #         if direction == "horizontal":
    #             cp = node.horizontal_crosspoint
    #         elif direction == "vertical":
    #             cp = node.vertical_crosspoint
    #         else:
    #             return cp_data

    #         # 提取CrossPoint状态信息
    #         cp_data = defaultdict(list)
    #         for direction, slices in cp.slice_connections.items():
    #             cp_data[direction] = [slices["arrival"].current_slots[self.parent.current_channel], slices["departure"].current_slots[self.parent.current_channel]]

    #     except Exception as e:
    #         pass  # print(f"警告: 提取crosspoint数据失败: {e}")

    #     return cp_data

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

        # 获取E-Tag优先级（兼容字典和对象格式）
        if isinstance(flit, dict):
            etag = flit.get("ETag_priority", "T2")
        else:
            etag = getattr(flit, "ETag_priority", "T2")  # 缺省视为 T2
        alpha = _ETAG_ALPHA.get(etag, 0.8)
        line_width = _ETAG_LW.get(etag, 1.0)
        edge_color = _ETAG_EDGE.get(etag, "black")

        return face_color, alpha, line_width, edge_color

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
                face, alpha, lw, edge = self._get_flit_style(
                    flit,
                    use_highlight=self.use_highlight,
                    expected_packet_id=self.highlight_pid,
                )
                patch.set_facecolor(face)
                patch.set_alpha(alpha)
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

    def _format_flit_info(self, flit):
        """Format flit information display, consistent with LinkStateVisualizer"""
        if not flit:
            return "No flit info"
        
        info_lines = []
        
        # Basic information
        packet_id = getattr(flit, "packet_id", None) if hasattr(flit, "packet_id") else flit.get("packet_id", None) if isinstance(flit, dict) else None
        flit_id = getattr(flit, "flit_id", None) if hasattr(flit, "flit_id") else flit.get("flit_id", None) if isinstance(flit, dict) else None
        
        if packet_id is not None:
            info_lines.append(f"Packet ID: {packet_id}")
        if flit_id is not None:
            info_lines.append(f"Flit ID: {flit_id}")
        
        # Add flit type information (request/response/data)
        flit_type = getattr(flit, "flit_type", None) if hasattr(flit, "flit_type") else flit.get("flit_type", None) if isinstance(flit, dict) else None
        channel = getattr(flit, "channel", None) if hasattr(flit, "channel") else flit.get("channel", None) if isinstance(flit, dict) else None
        req_type = getattr(flit, "req_type", None) if hasattr(flit, "req_type") else flit.get("req_type", None) if isinstance(flit, dict) else None
        
        if channel:
            channel_name = {"req": "Request", "rsp": "Response", "data": "Data"}.get(channel, channel)
            if flit_type:
                info_lines.append(f"Type: {channel_name}({flit_type})")
            else:
                info_lines.append(f"Type: {channel_name}")
        
        if req_type:
            req_name = {"read": "Read", "write": "Write"}.get(req_type, req_type)
            info_lines.append(f"Request: {req_name}")
        
        # Tag information
        etag = getattr(flit, "ETag_priority", None) if hasattr(flit, "ETag_priority") else flit.get("ETag_priority", None) if isinstance(flit, dict) else None
        if etag:
            info_lines.append(f"E-Tag: {etag}")
        
        itag_h = getattr(flit, "itag_h", False) if hasattr(flit, "itag_h") else flit.get("itag_h", False) if isinstance(flit, dict) else False
        itag_v = getattr(flit, "itag_v", False) if hasattr(flit, "itag_v") else flit.get("itag_v", False) if isinstance(flit, dict) else False
        
        if itag_h:
            info_lines.append("I-Tag: Horizontal")
        elif itag_v:
            info_lines.append("I-Tag: Vertical")
        
        # Position information
        current_pos = getattr(flit, "current_node_id", None) if hasattr(flit, "current_node_id") else flit.get("current_node_id", None) if isinstance(flit, dict) else None
        if current_pos is not None:
            info_lines.append(f"Position: {current_pos}")
        
        # Source-destination information
        src = getattr(flit, "source_ip_type", None) if hasattr(flit, "source_ip_type") else flit.get("source_ip_type", None) if isinstance(flit, dict) else None
        dst = getattr(flit, "dest_ip_type", None) if hasattr(flit, "dest_ip_type") else flit.get("dest_ip_type", None) if isinstance(flit, dict) else None
        
        if src and dst:
            info_lines.append(f"Path: {src}→{dst}")
        
        return "\n".join(info_lines) if info_lines else "No valid info"

    def _extract_flit_data(self, flit, channel, direction):
        """提取flit数据的通用方法"""
        if not flit:
            return None
        return {
            "packet_id": getattr(flit, "packet_id", None),
            "flit_id": getattr(flit, "flit_id", None),
            "ETag_priority": getattr(flit, "ETag_priority", None),
            "itag_h": getattr(flit, "itag_h", False),
            "itag_v": getattr(flit, "itag_v", False),
            "channel": channel,
            "direction": direction,
        }

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
                    if hasattr(fifo, "output_register") and hasattr(fifo, "output_valid") and fifo.output_valid and fifo.output_register:
                        output_flit_data = self._extract_flit_data(fifo.output_register, channel, direction)
                        fifo_data.append(output_flit_data)

                    channel_data[direction] = {node_id: fifo_data}
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

                    # 3. 保存Ring Bridge数据（优化版本）
                    try:
                        if hasattr(node, "ring_bridge"):
                            ring_bridge = node.ring_bridge
                            for channel in ["req", "rsp", "data"]:
                                channel_data = {}
                                # 合并input和output的处理
                                for fifo_type, attr_name in [("_in", "ring_bridge_input_fifos"), ("_out", "ring_bridge_output_fifos")]:
                                    if hasattr(ring_bridge, attr_name):
                                        fifos = getattr(ring_bridge, attr_name).get(channel, {})
                                        for direction, fifo in fifos.items():
                                            if hasattr(fifo, "internal_queue"):
                                                fifo_data = [self._extract_flit_data(flit, channel, direction) for flit in fifo.internal_queue]
                                                channel_data[f"{direction}{fifo_type}"] = {node_id: fifo_data}
                                node_data["ring_bridge"][channel] = channel_data
                    except:
                        node_data["ring_bridge"] = {}

                    # 4. 保存IP Channel数据（合并IQ和EQ处理）
                    try:
                        for channel in ["req", "rsp", "data"]:
                            # IQ channels
                            iq_data = {}
                            if hasattr(node, "ip_inject_channel_buffers"):
                                for ip_id, ip_interface in node.ip_inject_channel_buffers.items():
                                    if channel in ip_interface and hasattr(ip_interface[channel], "internal_queue"):
                                        # 提取internal_queue中的flit
                                        fifo_data = [self._extract_flit_data(flit, channel, "inject") for flit in ip_interface[channel].internal_queue]

                                        # 提取output_register中的flit（如果存在且有效）
                                        fifo = ip_interface[channel]
                                        if hasattr(fifo, "output_register") and hasattr(fifo, "output_valid") and fifo.output_valid and fifo.output_register:
                                            output_flit_data = self._extract_flit_data(fifo.output_register, channel, "inject")
                                            fifo_data.append(output_flit_data)

                                        iq_data[ip_id] = fifo_data
                            node_data["iq_channels"][channel] = iq_data

                            # EQ channels
                            eq_data = {}
                            if hasattr(node, "ip_eject_channel_buffers"):
                                for ip_id, ip_interface in node.ip_eject_channel_buffers.items():
                                    if channel in ip_interface and hasattr(ip_interface[channel], "internal_queue"):
                                        # 提取internal_queue中的flit
                                        fifo_data = [self._extract_flit_data(flit, channel, "eject") for flit in ip_interface[channel].internal_queue]

                                        # 提取output_register中的flit（如果存在且有效）
                                        fifo = ip_interface[channel]
                                        if hasattr(fifo, "output_register") and hasattr(fifo, "output_valid") and fifo.output_valid and fifo.output_register:
                                            output_flit_data = self._extract_flit_data(fifo.output_register, channel, "eject")
                                            fifo_data.append(output_flit_data)

                                        eq_data[ip_id] = fifo_data
                            node_data["eq_channels"][channel] = eq_data
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
                                    
                                    print(f"🔍 调试: 节点{node_id} {cp_name} 有{len(cp.slice_connections)}个方向: {list(cp.slice_connections.keys())}")
                                    
                                    for direction, channels in cp.slice_connections.items():
                                        print(f"🔍 调试: 节点{node_id} {cp_name} {direction}方向 channels: {list(channels.keys())}")
                                        
                                        # CrossPoint数据结构: slice_connections[direction][channel] = {"arrival": RingSlice, "departure": RingSlice}
                                        slices = channels.get(current_channel, {})
                                        arrival_slice = slices.get("arrival")
                                        departure_slice = slices.get("departure")
                                        
                                        print(f"🔍 调试: 节点{node_id} {cp_name} {direction}方向 {current_channel}通道 arrival_slice={arrival_slice is not None} departure_slice={departure_slice is not None}")
                                        
                                        # 调试输出
                                        if arrival_slice and hasattr(arrival_slice, "current_slots"):
                                            arrival_slot = arrival_slice.current_slots.get(current_channel)
                                            print(f"🔍 调试: 节点{node_id} {cp_name} {direction}方向 arrival.current_slots[{current_channel}]={arrival_slot is not None}")
                                            if arrival_slot:
                                                arrival_flit = getattr(arrival_slot, 'flit', None)
                                                slot_valid = getattr(arrival_slot, 'valid', False)
                                                print(f"🔍 调试: 节点{node_id} {cp_name} {direction}方向 arrival slot.valid={slot_valid} slot.flit={arrival_flit is not None}")
                                                if arrival_flit:
                                                    print(f"🔍 调试: 节点{node_id} {cp_name} {direction}方向 arrival发现flit: pid={getattr(arrival_flit, 'packet_id', 'N/A')}")
                                        
                                        if departure_slice and hasattr(departure_slice, "current_slots"):
                                            departure_slot = departure_slice.current_slots.get(current_channel)
                                            print(f"🔍 调试: 节点{node_id} {cp_name} {direction}方向 departure.current_slots[{current_channel}]={departure_slot is not None}")
                                            if departure_slot:
                                                departure_flit = getattr(departure_slot, 'flit', None)
                                                slot_valid = getattr(departure_slot, 'valid', False)
                                                print(f"🔍 调试: 节点{node_id} {cp_name} {direction}方向 departure slot.valid={slot_valid} slot.flit={departure_flit is not None}")
                                                if departure_flit:
                                                    print(f"🔍 调试: 节点{node_id} {cp_name} {direction}方向 departure发现flit: pid={getattr(departure_flit, 'packet_id', 'N/A')}")
                                        
                                        # 从RingSlice对象的current_slots中提取slot，再从slot中提取flit数据
                                        arrival_slot = arrival_slice.current_slots.get(current_channel) if (arrival_slice and hasattr(arrival_slice, "current_slots")) else None
                                        departure_slot = departure_slice.current_slots.get(current_channel) if (departure_slice and hasattr(departure_slice, "current_slots")) else None
                                        
                                        # 从slot中提取实际的flit数据
                                        arrival_flit = getattr(arrival_slot, 'flit', None) if (arrival_slot and getattr(arrival_slot, 'valid', False)) else None
                                        departure_flit = getattr(departure_slot, 'flit', None) if (departure_slot and getattr(departure_slot, 'valid', False)) else None
                                        
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

            if current_channel in iq_channels:
                self._render_ip_channels_from_snapshot("IQ_Ch", iq_channels[current_channel], node_id)

            if current_channel in eq_channels:
                self._render_ip_channels_from_snapshot("EQ_Ch", eq_channels[current_channel], node_id)

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
            directions = ["TR", "TL", "TU", "TD"]
        elif component_type == "RB":
            directions = ["TR_in", "TL_in", "TU_in", "TD_in", "TR_out", "TL_out", "TU_out", "TD_out", "EQ_out"]
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
                        self._render_fifo_patches(self.iq_patches, self.iq_texts, direction, direction_data[node_id])
                elif component_type == "EQ":
                    if node_id in direction_data:
                        self._render_fifo_patches(self.eq_patches, self.eq_texts, direction, direction_data[node_id])
                elif component_type == "RB":
                    if node_id in direction_data:
                        self._render_fifo_patches(self.rb_patches, self.rb_texts, direction, direction_data[node_id])
                elif component_type == "CP_H":
                    self._render_crosspoint_patches(self.cph_patches, self.cph_texts, direction, direction_data)
                elif component_type == "CP_V":
                    self._render_crosspoint_patches(self.cpv_patches, self.cpv_texts, direction, direction_data)

    def _render_ip_channels_from_snapshot(self, channel_type, channel_data, node_id):
        """从快照数据渲染IP通道数据"""
        if not channel_data:
            return

        # IP通道数据使用IP接口名称作为键，需要找到正确的键
        # 通常格式为 "gdma_0", "gdma_1" 等，对应节点0, 1等
        for ip_interface_key, flit_list in channel_data.items():
            # 尝试从IP接口名称提取节点ID
            if f"_{node_id}" in ip_interface_key or ip_interface_key.endswith(f"_{node_id}"):
                # 提取IP类型名称（去掉节点ID后缀）
                ip_type = ip_interface_key.rsplit('_', 1)[0] if '_' in ip_interface_key else ip_interface_key
                
                if channel_type == "IQ_Ch":
                    # IQ通道使用IP类型名称作为lane名称
                    self._render_fifo_patches(self.iq_patches, self.iq_texts, ip_type, flit_list)
                elif channel_type == "EQ_Ch":
                    # EQ通道使用IP类型名称作为lane名称
                    self._render_fifo_patches(self.eq_patches, self.eq_texts, ip_type, flit_list)

    def _render_fifo_patches(self, patch_dict, text_dict, lane_name, flit_list):
        """渲染FIFO类型patch的flit数据"""
        if lane_name not in patch_dict or lane_name not in text_dict:
            return

        patches = patch_dict[lane_name]
        texts = text_dict[lane_name]

        # 清空所有patch
        for p in patches:
            p.set_facecolor("none")
            p.set_alpha(1.0)
            p.set_linewidth(0)
            p.set_edgecolor("none")

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
        all_slots = arrival_slots + departure_slots
        
        print(f"🎯 调试: CrossPoint {direction}方向 arrival_slots={len(arrival_slots)} departure_slots={len(departure_slots)} all_slots={len(all_slots)}")
        for i, flit in enumerate(all_slots):
            if flit:
                print(f"   - slot[{i}]: pid={getattr(flit, 'packet_id', 'N/A')} fid={getattr(flit, 'flit_id', 'N/A')}")

        # 清空所有patch
        for p in patches:
            p.set_facecolor("none")
            p.set_alpha(1.0)
            p.set_linewidth(0)
            p.set_edgecolor("none")

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
                if p in self.patch_info_map:
                    self.patch_info_map.pop(p, None)

    def _show_no_data_message(self, node_id, message):
        """显示无数据消息"""
        self.ax.clear()
        self.ax.text(0.5, 0.5, f"节点 {node_id}\n{message}", ha="center", va="center", transform=self.ax.transAxes, fontsize=12, family="sans-serif")
