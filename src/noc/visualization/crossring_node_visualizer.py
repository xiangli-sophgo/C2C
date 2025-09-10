"""
CrossRing节点可视化器

基于旧版本Link_State_Visualizer的PieceVisualizer功能，
专门用于CrossRing拓扑的节点内部结构可视化，包括：
- Inject/Eject队列
- Ring Bridge FIFO
- CrossPoint状态
- Tag机制显示
"""

import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, FancyArrowPatch
from collections import defaultdict
from src.noc.crossring.config import CrossRingConfig
from src.utils.font_config import configure_matplotlib_fonts
from .color_manager import ColorManager
from .style_manager import VisualizationStyleManager

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

        # 样式管理器
        self.color_manager = ColorManager()
        self.style_manager = VisualizationStyleManager(self.color_manager)

        # ------ highlight / tracking ------
        self.use_highlight = False  # 是否启用高亮模式
        self.highlight_pid = None  # 被追踪的 packet_id
        self.show_tags_mode = False  # 标签显示模式

        # 存储 patch 和 text
        self.iq_patches, self.iq_texts = {}, {}
        self.eq_patches, self.eq_texts = {}, {}
        self.rb_patches, self.rb_texts = {}, {}
        self.cph_patches, self.cph_texts = {}, {}
        self.cpv_patches, self.cpv_texts = {}, {}

        # CP链路可视化相关
        self.cp_link_arrows = {}  # 存储link箭头patches
        self.cp_link_slots = {}  # 存储link的slice patches
        self.cp_link_mapping = {}  # 存储CP link的映射关系: {key: link_id}
        self.cp_link_texts = {}  # 存储link的文本标注
        self.cp_positions = {}  # 存储CP模块位置

        # 预计算所有节点的链路映射关系
        self._precompute_link_mappings()

        # 画出三个模块的框和 FIFO 槽
        self._draw_modules()

        # 初始化CP链路框架（持久化显示）
        self.current_node_id = None
        self._clear_cp_links()  # 确保初始化时清空

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
        def build_specs(orientations, h_pos, v_pos, depths):
            """Build specs for module size calculation"""
            specs = []
            for ori, hp, vp, d in zip(orientations, h_pos, v_pos, depths):
                if ori[0].upper() == "H":
                    v_group = {"left": "L", "right": "R"}.get(vp, "M")
                    h_group = {"top": "T", "bottom": "B"}.get(hp, "M")
                    specs.append(("H", h_group, v_group, d))
                else:  # vertical
                    v_group = {"left": "L", "right": "R"}.get(vp, "M")
                    h_group = {"top": "T", "bottom": "B"}.get(hp, "M")
                    specs.append(("V", h_group, v_group, d))
            return specs

        w_iq, h_iq = self._calc_module_size(build_specs(**{k: iq_config[k] for k in ["orientations", "h_pos", "v_pos", "depths"]}))
        w_eq, h_eq = self._calc_module_size(build_specs(**{k: eq_config[k] for k in ["orientations", "h_pos", "v_pos", "depths"]}))
        w_rb, h_rb = self._calc_module_size(build_specs(**{k: rb_config[k] for k in ["orientations", "h_pos", "v_pos", "depths"]}))
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

        # 记录CP模块位置以便后续绘制link
        self.cp_positions = {"horizontal": (CPH_x, CPH_y, self.cp_module_size[::-1]), "vertical": (CPV_x, CPV_y, self.cp_module_size)}  # (x, y, (width, height))

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

        # 添加边距（考虑到CP links会延伸出去，现在有8个link）
        margin = 4  # 进一步增加边距以容纳更多的link
        self.ax.set_xlim(min_x - margin, max_x + margin)
        self.ax.set_ylim(min_y - margin, max_y + margin)

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
            square *= 1.5  # 调整CP slot大小，使其与link更协调
            gap *= 15  # 调整gap比例
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
                            (x + module_width / 2 + square * 3 / 4, y + module_height / 2 - square * 3 / 2)
                            if lane[:2] in ["TL"]
                            else (x + module_width / 2 - square * 4 / 5, y + module_height / 2 + square / 2)
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
                            (x + module_width / 2 - square * 3 / 2, y + module_height / 2 - square) if lane[:2] in ["TU"] else (x + module_width / 2 + square / 3, y + module_height / 2 + square*1/4)
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

    def _on_click(self, event):
        """处理点击事件"""
        if event.inaxes != self.ax:
            return
        for patch, (txt, flit) in self.patch_info_map.items():
            contains, _ = patch.contains(event)
            if contains:
                # 只有在高亮模式下才允许切换文本可见性
                attrs = self.style_manager._extract_flit_attributes(flit)
                pid = attrs["packet_id"]
                fid = attrs["flit_id"]
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

        # 更新所有patch的颜色和文本可见性（包括CP links）
        all_patches = list(self.patch_info_map.items())

        # 添加CP link slots的patch
        for direction_slots in self.cp_link_slots.values():
            for patch in direction_slots:
                if patch in self.patch_info_map:
                    all_patches.append((patch, self.patch_info_map[patch]))

        for patch, (txt, flit) in all_patches:
            if flit:
                attrs = self.style_manager._extract_flit_attributes(flit)
                pid = attrs["packet_id"]

                # 使用样式管理器应用样式
                self.style_manager.apply_style_to_patch(patch, flit, use_highlight=self.use_highlight, expected_packet_id=self.highlight_pid, show_tags_mode=self.show_tags_mode)

                # 更新文本可见性
                if self.use_highlight and pid == self.highlight_pid:
                    txt.set_visible(True)
                else:
                    txt.set_visible(False)
            else:
                # 应用空样式
                empty_style = self.style_manager.create_empty_patch_style()
                for key, value in empty_style.items():
                    getattr(patch, f"set_{key}")(value)

        if not self.use_highlight:
            self.info_text.set_text("")

        # 触发重绘
        self.fig.canvas.draw_idle()

    def sync_tags_mode(self, show_tags_mode):
        """同步标签显示模式"""
        self.show_tags_mode = show_tags_mode

        # 更新所有patch的样式
        for patch, (txt, flit) in self.patch_info_map.items():
            # 使用样式管理器重新计算并应用样式
            if flit:
                self.style_manager.apply_style_to_patch(patch, flit, use_highlight=self.use_highlight, expected_packet_id=self.highlight_pid, show_tags_mode=self.show_tags_mode)

        # 触发重绘
        self.fig.canvas.draw_idle()

    def _format_flit_info(self, flit):
        """Format flit information display - use flit's repr for detailed info"""
        if not flit:
            return "No flit info"

        # 检查是否是MockFlit对象（CP链路中的flit）
        if hasattr(flit, "__class__") and flit.__class__.__name__ == "MockFlit":
            # 优先使用保存的原始flit repr
            if hasattr(flit, "flit_repr") and flit.flit_repr:
                return flit.flit_repr
            # 否则使用MockFlit的__repr__
            return repr(flit)

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

        # 对于活动的flit对象，优先使用保存的flit_repr
        if hasattr(flit, "flit_repr") and flit.flit_repr:
            return flit.flit_repr

        # 否则直接使用repr
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
        # 直接使用etag_priority属性
        etag_priority = getattr(flit, "etag_priority", "T2")

        data = {
            "packet_id": getattr(flit, "packet_id", None),
            "flit_id": getattr(flit, "flit_id", None),
            "etag_priority": etag_priority,
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
                    # 直接从快照数据渲染节点，传递目标周期信息
                    self._render_from_snapshot_data(node_id, node_data, current_channel, cycle)
                else:
                    self._show_no_data_message(node_id, "节点数据不存在")
            else:
                self._show_no_data_message(node_id, "无历史数据")

        except Exception as e:
            self._show_no_data_message(node_id, f"历史数据错误: {str(e)}")

    def render_node_from_network(self, node_id, network):
        """从实时网络数据渲染节点"""
        try:
            # 保存实时快照
            self.save_history_snapshot(network, getattr(network, "cycle", 0))

            # 从最新快照渲染
            if self.node_history:
                _, latest_snapshot = self.node_history[-1]
                nodes_data = latest_snapshot.get("nodes", {})
                node_data = nodes_data.get(node_id)

                if node_data:
                    current_channel = getattr(self.parent, "current_channel", "data") if self.parent else "data"
                    # 实时模式不需要指定target_cycle，使用最新数据
                    self._render_from_snapshot_data(node_id, node_data, current_channel)
                else:
                    self._show_no_data_message(node_id, "节点数据不存在")
            else:
                self._show_no_data_message(node_id, "无数据")

        except Exception as e:
            self._show_no_data_message(node_id, f"渲染错误: {str(e)}")

    def _clear_all_components(self, current_channel):
        """清空所有组件的显示"""
        # 统一清空所有组件
        for component_dict in [
            (self.iq_patches, self.iq_texts),
            (self.eq_patches, self.eq_texts),
            (self.rb_patches, self.rb_texts),
            (self.cph_patches, self.cph_texts),
            (self.cpv_patches, self.cpv_texts),
        ]:
            patches_dict, texts_dict = component_dict
            for lane_name, patches in patches_dict.items():
                if patches:
                    self._clear_and_render_patches(patches, texts_dict.get(lane_name, []), [])

        # 清空CP link的显示
        for direction_slots in self.cp_link_slots.values():
            for patch in direction_slots:
                patch.set_facecolor("white")
                patch.set_edgecolor("gray")
                patch.set_alpha(0.7)
                if patch in self.patch_info_map:
                    del self.patch_info_map[patch]

    def _render_from_snapshot_data(self, node_id, node_data, current_channel, target_cycle=None):
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

        # 只有当切换节点时才重新绘制CP links框架
        if self.current_node_id != node_id:
            self._clear_cp_links()  # 先清除旧的links
            self._draw_cp_links(node_id)  # 再绘制新的links
            self.current_node_id = node_id

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

            # 6. 更新CP链路slice的flit显示（只更新flit内容，不重绘框架）
            # 使用传递的目标周期参数，确保暂停时使用正确的历史数据
            self._update_all_link_slices(node_id, target_cycle)

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

    def _clear_and_render_patches(self, patches, texts, flit_list):
        """清空并渲染patch的通用方法"""
        # 清空所有patch并移除映射
        for p in patches:
            p.set_facecolor("none")
            p.set_linewidth(0)
            p.set_edgecolor("none")
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
                # 使用样式管理器处理属性和样式
                attrs = self.style_manager._extract_flit_attributes(flit)
                packet_id = attrs["packet_id"]
                flit_id = attrs["flit_id"]

                # 应用样式
                self.style_manager.apply_style_to_patch(p, flit, use_highlight=self.use_highlight, expected_packet_id=self.highlight_pid, show_tags_mode=self.show_tags_mode)

                info = f"{packet_id}-{flit_id}"
                t.set_text(info)
                t.set_visible(self.use_highlight and packet_id == self.highlight_pid)
                self.patch_info_map[p] = (t, flit)

                if self.use_highlight and packet_id == self.highlight_pid:
                    self.current_highlight_flit = flit

    def _render_fifo_patches(self, patch_dict, text_dict, lane_name, flit_list):
        """渲染FIFO类型patch的flit数据"""
        if lane_name not in patch_dict or lane_name not in text_dict:
            return
        self._clear_and_render_patches(patch_dict[lane_name], text_dict[lane_name], flit_list)

    def _render_crosspoint_patches_split(self, patch_dict, text_dict, direction, slice_data):
        """渲染CrossPoint类型patch的slice数据 - 拆分版本"""
        # CrossPoint数据结构: [arrival_slots, departure_slots]
        if not isinstance(slice_data, list) or len(slice_data) < 2:
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
        # 只取第一个slot（因为每个lane现在只有1个深度）
        flit_list = [slot_data[0]] if slot_data else []
        self._clear_and_render_patches(patches, texts, flit_list)

    def _show_no_data_message(self, node_id, message):
        """显示无数据消息"""
        self.ax.clear()
        self.ax.text(0.5, 0.5, f"节点 {node_id}\n{message}", ha="center", va="center", transform=self.ax.transAxes, fontsize=12, family="sans-serif")

    def _should_show_link(self, node_id, direction, slice_type):
        """判断是否应该显示某个方向的link（边缘检测）

        Args:
            node_id: 节点ID
            direction: 方向 ("TL", "TR", "TU", "TD")
            slice_type: slice类型 ("arrival", "departure")
        """
        if node_id is None:
            return False

        row = node_id // self.cols
        col = node_id % self.cols

        # 左边缘节点（col == 0）
        if col == 0:
            if direction == "TL" and slice_type == "departure":
                return False  # 左边缘，TL departure没有（无法向左发送）
            if direction == "TR" and slice_type == "arrival":
                return False  # 左边缘，TR arrival没有（左边没有节点向右发送）

        # 右边缘节点（col == cols-1）
        if col == self.cols - 1:
            if direction == "TR" and slice_type == "departure":
                return False  # 右边缘，TR departure没有（无法向右发送）
            if direction == "TL" and slice_type == "arrival":
                return False  # 右边缘，TL arrival没有（右边没有节点向左发送）

        # 上边缘节点（row == 0）
        if row == 0:
            if direction == "TU" and slice_type == "departure":
                return False  # 上边缘，TU departure没有（无法向上发送）
            if direction == "TD" and slice_type == "arrival":
                return False  # 上边缘，TD arrival没有（上面没有节点向下发送）

        # 下边缘节点（row == rows-1）
        if row == self.rows - 1:
            if direction == "TD" and slice_type == "departure":
                return False  # 下边缘，TD departure没有（无法向下发送）
            if direction == "TU" and slice_type == "arrival":
                return False  # 下边缘，TU arrival没有（下面没有节点向上发送）

        return True

    def _clear_cp_links(self):
        """清除所有CP links的可视化元素"""
        # 清除箭头patches
        for key, patches in self.cp_link_arrows.items():
            if isinstance(patches, list):
                for patch in patches:
                    try:
                        patch.remove()
                    except:
                        pass
            else:
                try:
                    patches.remove()
                except:
                    pass
        self.cp_link_arrows.clear()

        # 清除slice patches
        for key, patches in self.cp_link_slots.items():
            if isinstance(patches, list):
                for patch in patches:
                    try:
                        patch.remove()
                    except:
                        pass
            else:
                try:
                    patches.remove()
                except:
                    pass
        self.cp_link_slots.clear()

        # 清除文本标注
        for key, texts in self.cp_link_texts.items():
            if isinstance(texts, list):
                for text in texts:
                    try:
                        text.remove()
                    except:
                        pass
            else:
                try:
                    texts.remove()
                except:
                    pass
        self.cp_link_texts.clear()

    def _draw_cp_links(self, node_id):
        """绘制CP连接的links和slices

        CrossRing架构说明：
        - 每个节点有2个CP：水平CP和垂直CP
        - 水平CP：管理TL和TR两个方向，每个方向连接2个link（arrival和departure）
        - 垂直CP：管理TU和TD两个方向，每个方向连接2个link（arrival和departure）
        - 总共每个节点有8个link连接
        """
        if node_id is None or not self.cp_positions:
            return

        # 清空之前的link可视化
        self.cp_link_arrows.clear()
        self.cp_link_slots.clear()
        self.cp_link_texts.clear()

        # 获取CP模块位置
        cph_x, cph_y, cph_size = self.cp_positions["horizontal"]  # (x, y, (width, height))
        cpv_x, cpv_y, cpv_size = self.cp_positions["vertical"]

        # 水平CP：绘制TL和TR方向的arrival和departure links
        for direction in ["TL", "TR"]:
            # 绘制departure link（从CP出去）
            if self._should_show_link(node_id, direction, "departure"):
                self._draw_single_cp_link(cph_x, cph_y, cph_size, direction, node_id, "horizontal", "departure")
            # 绘制arrival link（到达CP）
            if self._should_show_link(node_id, direction, "arrival"):
                self._draw_single_cp_link(cph_x, cph_y, cph_size, direction, node_id, "horizontal", "arrival")

        # 垂直CP：绘制TU和TD方向的arrival和departure links
        for direction in ["TU", "TD"]:
            # 绘制departure link（从CP出去）
            if self._should_show_link(node_id, direction, "departure"):
                self._draw_single_cp_link(cpv_x, cpv_y, cpv_size, direction, node_id, "vertical", "departure")
            # 绘制arrival link（到达CP）
            if self._should_show_link(node_id, direction, "arrival"):
                self._draw_single_cp_link(cpv_x, cpv_y, cpv_size, direction, node_id, "vertical", "arrival")

    def _draw_single_cp_link(self, cp_x, cp_y, cp_size, direction, node_id, cp_type, slice_type):
        """绘制单个方向的link和slices - 基于single_cp_visualization.py的正确实现"""

        # 获取slice数量配置
        slice_num = getattr(self.config.basic_config, "SLICE_PER_LINK", 5) - 2
        slice_size = 0.5  # 增大slice尺寸
        slice_gap = 0.1  # 增大slice间距
        slice_offset = 0.3  # 增大slice偏移量
        arrow_length = 7  # 增大箭头长度

        cp_width, cp_height = cp_size

        if cp_type == "horizontal":
            # 基于single_cp_visualization.py的水平CP实现
            self._draw_horizontal_cp_link(cp_x, cp_y, cp_width, cp_height, direction, slice_type, slice_size, slice_gap, slice_num, slice_offset, arrow_length, node_id)
        else:
            # 基于single_vertical_cp_visualization.py的垂直CP实现
            self._draw_vertical_cp_link(cp_x, cp_y, cp_width, cp_height, direction, slice_type, slice_size, slice_gap, slice_num, slice_offset, arrow_length, node_id)

    def _draw_horizontal_cp_link(self, cp_x, cp_y, cp_width, cp_height, direction, slice_type, slice_size, slice_gap, slice_num, slice_offset, arrow_length, node_id):
        """绘制水平CP的link - 直接移植single_cp_visualization.py的逻辑"""

        # 确定link位置（按single_cp_visualization.py的布局）
        if direction == "TL":
            link_y = cp_y + cp_height * 0.3  # TL在上方
        else:  # TR
            link_y = cp_y + cp_height * 0.7  # TR在下方

        # 构建direction参数（兼容single_cp的命名）
        if direction == "TL":
            if slice_type == "departure":
                direction_param = "departure_left"
            else:
                direction_param = "arrival_left"
        else:  # TR
            if slice_type == "departure":
                direction_param = "departure_right"
            else:
                direction_param = "arrival_right"

        # 使用single_cp_visualization.py的核心绘制逻辑
        self._draw_horizontal_link_core(cp_x, link_y, f"{direction}-{slice_type}", direction_param, slice_size, slice_gap, slice_num, slice_offset, arrow_length, node_id, cp_width, cp_height)

    def _draw_horizontal_link_core(self, cp_x, link_y, label, direction, slice_size, slice_gap, slice_num, slice_offset, arrow_length, node_id, cp_width, cp_height):
        """水平链路核心绘制逻辑 - 移植自single_cp_visualization.py"""

        # CP参数 - 使用节点可视化器中的实际CP尺寸
        cp_center_x = cp_x + cp_width / 2
        cp_center_y = link_y

        # 节点半径 - 基于实际CP尺寸
        node_radius = cp_width / 2

        if direction == "departure_left":
            # TL Departure：从CP中心向左
            unit_dx = -1.0
            unit_dy = 0.0
            cp_connection_x = cp_center_x + unit_dx * node_radius
            cp_connection_y = cp_center_y
            arrow_end_x = cp_connection_x + unit_dx * arrow_length
            arrow_end_y = cp_connection_y
            arrow_start = (cp_connection_x, cp_connection_y)
            arrow_end = (arrow_end_x, arrow_end_y)

        elif direction == "departure_right":
            # TR Departure：从CP中心向右
            unit_dx = 1.0
            unit_dy = 0.0
            cp_connection_x = cp_center_x + unit_dx * node_radius
            cp_connection_y = cp_center_y
            arrow_end_x = cp_connection_x + unit_dx * arrow_length
            arrow_end_y = cp_connection_y
            arrow_start = (cp_connection_x, cp_connection_y)
            arrow_end = (arrow_end_x, arrow_end_y)

        elif direction == "arrival_left":
            # TL Arrival：也是向左的箭头
            unit_dx = 1.0
            unit_dy = 0.0
            cp_connection_x = cp_center_x + unit_dx * node_radius
            cp_connection_y = cp_center_y
            arrow_end_x = cp_connection_x + unit_dx * arrow_length
            arrow_end_y = cp_connection_y
            arrow_start = (arrow_end_x, arrow_end_y)
            arrow_end = (cp_connection_x, cp_connection_y)

        elif direction == "arrival_right":
            # TR Arrival：也是向右的箭头
            unit_dx = -1.0
            unit_dy = 0.0
            cp_connection_x = cp_center_x + unit_dx * node_radius
            cp_connection_y = cp_center_y
            arrow_end_x = cp_connection_x + unit_dx * arrow_length
            arrow_end_y = cp_connection_y
            arrow_start = (arrow_end_x, arrow_end_y)
            arrow_end = (cp_connection_x, cp_connection_y)

        # 绘制箭头 - 调整尺寸参数
        arrow = FancyArrowPatch(arrow_start, arrow_end, arrowstyle="->", mutation_scale=12, color="black", linewidth=1.0)  # 减小箭头头部和线宽
        self.ax.add_patch(arrow)

        # 存储箭头以便后续清除
        arrow_key = f"{node_id}_{label}"
        if arrow_key not in self.cp_link_arrows:
            self.cp_link_arrows[arrow_key] = []
        self.cp_link_arrows[arrow_key].append(arrow)

        # 绘制slice序列
        arrow_center_x = (arrow_start[0] + arrow_end[0]) / 2
        self._draw_horizontal_slice_sequence(arrow_center_x, link_y, slice_size, slice_gap, slice_num, slice_offset, label, direction, node_id)

    def _draw_horizontal_slice_sequence(self, center_x, arrow_y, slice_size, slice_gap, slice_num, slice_offset, link_label, direction, node_id):
        """绘制水平link的slice序列 - 移植自single_cp_visualization.py"""

        total_width = slice_num * slice_size + (slice_num - 1) * slice_gap

        # slice的X位置
        if "left" in direction:
            start_x = center_x - total_width / 2
        else:
            start_x = center_x - total_width / 2

        # slice的Y位置
        if "left" in direction:
            # 向右的link，slice在箭头下方
            slice_y = arrow_y - slice_offset * 0.7 - slice_size
        else:
            # 向左的link，slice在箭头上方
            slice_y = arrow_y + slice_offset * 0.7

        # 创建slot key
        slot_key = f"{node_id}_{link_label}"
        self.cp_link_slots[slot_key] = []

        # 绘制slice
        for i in range(slice_num):
            slice_x = start_x + i * (slice_size + slice_gap)
            # 内联绘制单个slice
            outer_rect = Rectangle((slice_x, slice_y), slice_size, slice_size, linewidth=1, edgecolor="gray", facecolor="white", linestyle="--", alpha=0.8)
            self.ax.add_patch(outer_rect)
            self.cp_link_slots[slot_key].append(outer_rect)

    def _draw_vertical_cp_link(self, cp_x, cp_y, cp_width, cp_height, direction, slice_type, slice_size, slice_gap, slice_num, slice_offset, arrow_length, node_id):
        """绘制垂直CP的link - 基于single_vertical_cp_visualization.py的逻辑"""

        # 确定link位置（按single_vertical_cp_visualization.py的布局）
        if direction == "TU":
            link_x = cp_x + cp_width * 0.3  # TU在左方
        else:  # TD
            link_x = cp_x + cp_width * 0.7  # TD在右方

        # 构建direction参数（兼容single_vertical_cp的命名）
        if direction == "TU":
            if slice_type == "departure":
                direction_param = "departure_up"
            else:
                direction_param = "arrival_up"
        else:  # TD
            if slice_type == "departure":
                direction_param = "departure_down"
            else:
                direction_param = "arrival_down"

        # 使用single_vertical_cp_visualization.py的核心绘制逻辑
        self._draw_vertical_link_core(link_x, cp_y, f"{direction}-{slice_type}", direction_param, slice_size, slice_gap, slice_num, slice_offset, arrow_length, node_id, cp_width, cp_height)

    def _draw_vertical_link_core(self, link_x, cp_y, label, direction, slice_size, slice_gap, slice_num, slice_offset, arrow_length, node_id, cp_width, cp_height):
        """垂直链路核心绘制逻辑 - 移植自single_vertical_cp_visualization.py"""

        # CP参数 - 使用节点可视化器中的实际CP尺寸
        cp_center_x = link_x
        cp_center_y = cp_y + cp_height / 2

        # 节点半径 - 基于实际CP尺寸
        node_radius = cp_height / 2

        if direction == "departure_up":
            # TU Departure：从CP中心向上
            unit_dx = 0.0
            unit_dy = 1.0
            cp_connection_x = cp_center_x
            cp_connection_y = cp_center_y + unit_dy * node_radius
            arrow_end_x = cp_connection_x
            arrow_end_y = cp_connection_y + unit_dy * arrow_length
            arrow_start = (cp_connection_x, cp_connection_y)
            arrow_end = (arrow_end_x, arrow_end_y)

        elif direction == "departure_down":
            # TD Departure：从CP中心向下
            unit_dx = 0.0
            unit_dy = -1.0
            cp_connection_x = cp_center_x
            cp_connection_y = cp_center_y + unit_dy * node_radius
            arrow_end_x = cp_connection_x
            arrow_end_y = cp_connection_y + unit_dy * arrow_length
            arrow_start = (cp_connection_x, cp_connection_y)
            arrow_end = (arrow_end_x, arrow_end_y)

        elif direction == "arrival_up":
            # TU Arrival：也是向上的箭头
            unit_dx = 0.0
            unit_dy = -1.0
            cp_connection_x = cp_center_x
            cp_connection_y = cp_center_y + unit_dy * node_radius
            arrow_end_x = cp_connection_x
            arrow_end_y = cp_connection_y + unit_dy * arrow_length
            arrow_start = (arrow_end_x, arrow_end_y)
            arrow_end = (cp_connection_x, cp_connection_y)

        elif direction == "arrival_down":
            # TD Arrival：也是向下的箭头
            unit_dx = 0.0
            unit_dy = 1.0
            cp_connection_x = cp_center_x
            cp_connection_y = cp_center_y + unit_dy * node_radius
            arrow_end_x = cp_connection_x
            arrow_end_y = cp_connection_y + unit_dy * arrow_length
            arrow_start = (arrow_end_x, arrow_end_y)
            arrow_end = (cp_connection_x, cp_connection_y)

        # 绘制箭头 - 调整尺寸参数
        arrow = FancyArrowPatch(arrow_start, arrow_end, arrowstyle="->", mutation_scale=12, color="black", linewidth=1.0)  # 减小箭头头部和线宽
        self.ax.add_patch(arrow)

        # 存储箭头以便后续清除
        arrow_key = f"{node_id}_{label}"
        if arrow_key not in self.cp_link_arrows:
            self.cp_link_arrows[arrow_key] = []
        self.cp_link_arrows[arrow_key].append(arrow)

        # 绘制slice序列
        arrow_center_y = (arrow_start[1] + arrow_end[1]) / 2
        self._draw_vertical_slice_sequence(link_x, arrow_center_y, slice_size, slice_gap, slice_num, slice_offset, label, direction, node_id)

    def _draw_vertical_slice_sequence(self, arrow_x, center_y, slice_size, slice_gap, slice_num, slice_offset, link_label, direction, node_id):
        """绘制垂直link的slice序列 - 移植自single_vertical_cp_visualization.py"""

        total_height = slice_num * slice_size + (slice_num - 1) * slice_gap

        # slice的Y位置
        if "up" in direction:
            start_y = center_y - total_height / 2
        else:
            start_y = center_y - total_height / 2

        # slice的X位置
        if "up" in direction:
            # 向上的link，slice在箭头左边
            slice_x = arrow_x - slice_offset * 0.7 - slice_size
        else:
            # 向下的link，slice在箭头右边
            slice_x = arrow_x + slice_offset * 0.7

        # 创建slot key
        slot_key = f"{node_id}_{link_label}"
        self.cp_link_slots[slot_key] = []

        # 绘制slice
        for i in range(slice_num):
            slice_y = start_y + i * (slice_size + slice_gap)
            # 内联绘制单个slice
            outer_rect = Rectangle((slice_x, slice_y), slice_size, slice_size, linewidth=1, edgecolor="gray", facecolor="white", linestyle="--", alpha=0.8)
            self.ax.add_patch(outer_rect)
            self.cp_link_slots[slot_key].append(outer_rect)

    def _precompute_link_mappings(self):
        """预计算所有节点的CP链路映射关系"""
        for node_id in range(self.rows * self.cols):
            row = node_id // self.cols
            col = node_id % self.cols

            # 为每个方向和类型计算链路ID
            for direction in ["TL", "TR", "TU", "TD"]:
                for link_type in ["arrival", "departure"]:
                    key = f"{node_id}_{direction}-{link_type}"

                    # 计算对应的link_id
                    link_id = self._calculate_link_id(node_id, direction, link_type, row, col)
                    if link_id:
                        self.cp_link_mapping[key] = link_id

    def _calculate_link_id(self, node_id, direction, link_type, row, col):
        """计算特定节点方向的链路ID"""
        if link_type == "departure":
            # Departure链路：从当前节点发出
            if direction == "TL":
                if col == 0:
                    return None  # 左边缘没有向左的链路
                return f"link_{node_id}_TL_{node_id - 1}"
            elif direction == "TR":
                if col == self.cols - 1:
                    return None  # 右边缘没有向右的链路
                return f"link_{node_id}_TR_{node_id + 1}"
            elif direction == "TU":
                if row == 0:
                    return None  # 上边缘没有向上的链路
                return f"link_{node_id}_TU_{node_id - self.cols}"
            elif direction == "TD":
                if row == self.rows - 1:
                    return None  # 下边缘没有向下的链路
                return f"link_{node_id}_TD_{node_id + self.cols}"
        else:  # arrival
            # Arrival链路：从其他节点到当前节点
            if direction == "TL":
                if col == self.cols - 1:
                    return None  # 右边缘没有来自右边的链路
                return f"link_{node_id + 1}_TL_{node_id}"
            elif direction == "TR":
                if col == 0:
                    return None  # 左边缘没有来自左边的链路
                return f"link_{node_id - 1}_TR_{node_id}"
            elif direction == "TU":
                if row == self.rows - 1:
                    return None  # 下边缘没有来自下边的链路
                return f"link_{node_id + self.cols}_TU_{node_id}"
            elif direction == "TD":
                if row == 0:
                    return None  # 上边缘没有来自上边的链路
                return f"link_{node_id - self.cols}_TD_{node_id}"
        return None

    def _update_all_link_slices(self, node_id, target_cycle=None):
        """更新当前节点所有CP链路的slice显示"""
        current_channel = getattr(self.parent, "current_channel", 0) if self.parent else 0

        # 更新所有方向的link slices
        for direction in ["TL", "TR", "TU", "TD"]:
            self._update_link_slices(node_id, direction, current_channel, target_cycle)

    def _update_link_slices(self, node_id, direction, channel, target_cycle=None):
        """更新link中slice的显示状态"""
        # 分别处理arrival和departure链路
        for slice_type in ["arrival", "departure"]:
            key = f"{node_id}_{direction}-{slice_type}"
            if key not in self.cp_link_slots:
                continue

            patches = self.cp_link_slots[key]
            if not patches:
                continue

            # 直接从预计算的映射中获取link_id
            link_id = self.cp_link_mapping.get(key)
            if not link_id:
                # 没有对应链路（如边缘节点），清空显示
                self._clear_link_patches(patches)
                continue

            # 从link snapshot获取数据，支持指定周期
            channel_data = self._get_link_data_by_id(link_id, channel, target_cycle)

            self._update_single_link_patches(patches, channel_data, channel, slice_type, direction)

    def _get_link_data_by_id(self, link_id, channel, target_cycle=None):
        """通过link_id直接获取链路数据"""
        if not hasattr(self.parent, "history") or not self.parent.history:
            return None

        # 确定要使用的快照数据
        snapshot_data = None

        if target_cycle is not None:
            # 查找指定周期的快照数据
            for hist_cycle, hist_snapshot in self.parent.history:
                if hist_cycle == target_cycle:
                    snapshot_data = hist_snapshot
                    break

        if snapshot_data is None:
            # 回退到最新的link snapshot
            _, snapshot_data = self.parent.history[-1]

        links_data = snapshot_data.get("links", {})

        # 将数字通道索引转换为字符串名称
        channel_names = ["req", "rsp", "data"]
        if isinstance(channel, int) and 0 <= channel < len(channel_names):
            channel_name = channel_names[channel]
        elif isinstance(channel, str):
            channel_name = channel
        else:
            return None

        # 获取链路数据
        link_data = links_data.get(link_id, {})
        channel_data = link_data.get(channel_name, {})

        return channel_data

    def _clear_link_patches(self, patches):
        """清空链路patches显示"""
        for patch in patches:
            patch.set_facecolor("white")
            patch.set_edgecolor("gray")
            patch.set_linestyle("--")
            patch.set_alpha(0.7)
            if patch in self.patch_info_map:
                del self.patch_info_map[patch]

    def _update_single_link_patches(self, patches, channel_data, channel, slice_type, direction):
        """更新单个链路的patches显示"""
        # 如果没有链路数据，清空所有slots
        if not channel_data:
            self._clear_link_patches(patches)
            return

        # 获取通道名称用于数据访问
        channel_names = ["req", "rsp", "data"]
        if isinstance(channel, int) and 0 <= channel < len(channel_names):
            channel_name = channel_names[channel]
        elif isinstance(channel, str):
            channel_name = channel
        else:
            return

        # 更新每个slot的显示 - 跳过首尾slice，只显示中间slice
        for i, patch in enumerate(patches):
            # 根据direction调整索引映射：
            # TL: 右→左(反向), TR: 左→右(正向), TU: 下→上(正向), TD: 上→下(反向)
            is_positive_direction = direction in ["TR", "TU"]

            if is_positive_direction:
                # TR/TU：flit向正方向移动（左到右/下到上）
                actual_slice_idx = i + 1  # 正向：跳过slice_idx=0，从slice_idx=1开始
            else:
                # TL/TD：flit向负方向移动（右到左/上到下）
                total_slices = len(patches) + 2  # 加上跳过的首尾slice
                actual_slice_idx = total_slices - 2 - i  # 反向映射

            # 从channel_data中获取对应slice的数据
            slice_data = channel_data.get(actual_slice_idx, {})
            slot_data = slice_data.get("slots", {}).get(channel_name, {})
            # 清除之前的flit显示
            for child in patch.get_children():
                if hasattr(child, "_mock_flit"):
                    child.remove()

            if slot_data and slot_data.get("valid", False) and "flit" in slot_data:
                # 有flit数据，创建一个模拟的flit对象用于样式应用
                flit_data = slot_data["flit"]

                # 创建一个简单的flit对象用于样式管理
                class MockFlit:
                    def __init__(self, data):
                        for key, value in data.items():
                            setattr(self, key, value)

                    def __repr__(self):
                        # 优先使用保存的原始flit repr信息
                        if hasattr(self, "flit_repr") and self.flit_repr:
                            return self.flit_repr
                        # 回退到基本信息（与_FlitProxy保持一致的格式）
                        pid = getattr(self, "packet_id", "N/A")
                        fid = getattr(self, "flit_id", "N/A")
                        etag = getattr(self, "etag_priority", "T2")
                        itag_h = getattr(self, "itag_h", False)
                        itag_v = getattr(self, "itag_v", False)
                        itag = "H" if itag_h else ("V" if itag_v else "")
                        return f"(pid={pid}, fid={fid}, ET={etag}, IT={itag})"

                mock_flit = MockFlit(flit_data)

                # 应用flit样式
                self.style_manager.apply_style_to_patch(patch, mock_flit, use_highlight=self.use_highlight, expected_packet_id=self.highlight_pid, show_tags_mode=self.show_tags_mode)
                # 有flit时设置为实线边框
                patch.set_linestyle("-")
                # 设置为可点击
                text_obj = self.ax.text(0, 0, "", visible=False)
                self.patch_info_map[patch] = (text_obj, mock_flit)
            else:
                # 空slot
                patch.set_facecolor("white")
                patch.set_edgecolor("gray")
                patch.set_linestyle("--")  # 虚线边框
                patch.set_alpha(0.7)
                if patch in self.patch_info_map:
                    del self.patch_info_map[patch]
