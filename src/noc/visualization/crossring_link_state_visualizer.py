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
import logging

# 中文字体支持
plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False


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


class CrossRingLinkStateVisualizer:
    """
    CrossRing Link State Visualizer
    
    完全基于原版Link_State_Visualizer重新实现，包含：
    - NetworkLinkVisualizer主类
    - PieceVisualizer内嵌类  
    - 完整的拓扑显示和节点详细视图
    """
    
    class PieceVisualizer:
        """节点详细视图可视化器（右侧面板）"""
        
        def __init__(self, config, ax, highlight_callback=None, parent=None):
            """
            仅绘制单个节点的 Inject/Eject Queue 和 Ring Bridge FIFO。
            参数:
            - config: 含有 FIFO 深度配置的对象，属性包括 cols, num_nodes, IQ_OUT_FIFO_DEPTH,
              EQ_IN_FIFO_DEPTH, RB_IN_FIFO_DEPTH, RB_OUT_FIFO_DEPTH
            - node_id: 要可视化的节点索引 (0 到 num_nodes-1)
            """
            self.highlight_callback = highlight_callback
            self.config = config
            self.cols = getattr(config, 'NUM_COL', getattr(config, 'num_col', 3))
            self.rows = getattr(config, 'NUM_ROW', getattr(config, 'num_row', 2))
            self.parent = parent
            
            # 提取深度
            self.IQ_depth = getattr(config, 'IQ_OUT_FIFO_DEPTH', 8)
            self.EQ_depth = getattr(config, 'EQ_IN_FIFO_DEPTH', 8)
            self.RB_in_depth = getattr(config, 'RB_IN_FIFO_DEPTH', 4)
            self.RB_out_depth = getattr(config, 'RB_OUT_FIFO_DEPTH', 4)
            self.slice_per_link = getattr(config, 'SLICE_PER_LINK', 8)
            self.IQ_CH_depth = getattr(config, 'IQ_CH_FIFO_DEPTH', 4)
            self.EQ_CH_depth = getattr(config, 'EQ_CH_FIFO_DEPTH', 4)
            
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
            self.slot_frame_lw = 0.4  # can be tuned externally

            height = 8
            weight = 5
            self.inject_module_size = (height, weight)
            self.eject_module_size = (weight, height)
            self.rb_module_size = (height, height)
            self.cp_module_size = (2, 5)
            
            # 初始化图形
            if ax is None:
                self.fig, self.ax = plt.subplots(figsize=(10, 8))  # 增大图形尺寸
            else:
                self.ax = ax
                self.fig = ax.figure
            self.ax.axis("off")
            self.ax.set_aspect("equal")
            
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
            self.info_text = self.fig.text(0.75, 0.02, "", fontsize=12, va="bottom", ha="left", wrap=True)
            
            # 当前被点击 / 高亮的 flit（用于信息框自动刷新）
            self.current_highlight_flit = None

        def _draw_modules(self):
            """绘制所有模块"""
            # 获取通道名称
            ch_names = getattr(self.config, 'CH_NAME_LIST', ['gdma', 'ddr', 'l2m'])
            
            # ------------------- unified module configs ------------------- #
            iq_config = dict(
                title="Inject Queue",
                lanes=ch_names + ["TL", "TR", "TD", "TU", "EQ"],
                depths=[self.IQ_CH_depth] * len(ch_names) + [self.IQ_depth] * 5,
                orientations=["vertical"] * len(ch_names) + ["vertical"] * 2 + ["horizontal"] * 3,
                h_pos=["top"] * len(ch_names) + ["bottom"] * 2 + ["mid"] * 3,
                v_pos=["left"] * len(ch_names) + ["left"] * 2 + ["right"] * 3,
                patch_dict=self.iq_patches,
                text_dict=self.iq_texts,
            )

            eq_config = dict(
                title="Eject Queue",
                lanes=ch_names + ["TU", "TD"],
                depths=[self.EQ_CH_depth] * len(ch_names) + [self.EQ_depth] * 2,
                orientations=["horizontal"] * len(ch_names) + ["horizontal"] * 2,
                h_pos=["top"] * len(ch_names) + ["bottom"] * 2,
                v_pos=["left"] * len(ch_names) + ["right", "right"],
                patch_dict=self.eq_patches,
                text_dict=self.eq_texts,
            )

            rb_config = dict(
                title="Ring Bridge",
                lanes=["TL", "TR", "TU", "TD", "EQ"],
                depths=[self.RB_in_depth] * 2 + [self.RB_out_depth] * 3,
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

            # 绘制各个模块
            self._draw_fifo_module(-4, 0.0, self.inject_module_size, iq_config)
            self._draw_fifo_module(0.0, 4, self.eject_module_size, eq_config)
            self._draw_fifo_module(0.0, 0.0, self.rb_module_size, rb_config)
            self._draw_fifo_module(1, -2, self.cp_module_size, cross_point_horizontal_config)
            self._draw_fifo_module(-1, 2, self.cp_module_size, cross_point_vertical_config)

        def _draw_fifo_module(self, base_x, base_y, module_size, config):
            """绘制FIFO模块"""
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

        def _calc_fifo_position(self, base_x, base_y, module_size, index, total_lanes, 
                               orientation, h_pos, v_pos):
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

        def _draw_fifo(self, x, y, depth, orientation, lane, patch_dict, text_dict):
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

        def draw_piece_for_node(self, node_id, network):
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
            
            if not network or not hasattr(network, 'nodes'):
                return inject_data
            
            try:
                node = network.nodes.get(node_id)
                if not node:
                    return inject_data
                
                # 提取inject_direction_fifos数据
                if hasattr(node, 'inject_direction_fifos'):
                    for direction, fifo in node.inject_direction_fifos.items():
                        if hasattr(fifo, 'queue') and fifo.queue:
                            inject_data[direction] = {node_id: list(fifo.queue)}
                
                # 提取channel_buffer数据
                if hasattr(node, 'channel_buffer'):
                    for channel_name, buffer in node.channel_buffer.items():
                        if hasattr(buffer, 'queue') and buffer.queue:
                            inject_data[channel_name] = {node_id: list(buffer.queue)}
                            
            except Exception as e:
                self.logger.debug(f"提取inject queue数据失败: {e}")
            
            return inject_data

        def _get_eject_queues_data(self, network, node_id):
            """从网络中提取eject queue数据（适配层）"""
            eject_data = {}
            
            if not network or not hasattr(network, 'nodes'):
                return eject_data
            
            try:
                node = network.nodes.get(node_id)
                if not node:
                    return eject_data
                
                # 提取eject_input_fifos数据
                if hasattr(node, 'eject_input_fifos'):
                    for direction, fifo in node.eject_input_fifos.items():
                        if hasattr(fifo, 'queue') and fifo.queue:
                            eject_data[direction] = {node_id: list(fifo.queue)}
                
                # 提取ip_eject_channel_buffers数据
                if hasattr(node, 'ip_eject_channel_buffers'):
                    for channel_name, buffer in node.ip_eject_channel_buffers.items():
                        if hasattr(buffer, 'queue') and buffer.queue:
                            eject_data[channel_name] = {node_id: list(buffer.queue)}
                            
            except Exception as e:
                self.logger.debug(f"提取eject queue数据失败: {e}")
            
            return eject_data

        def _get_ring_bridge_data(self, network, node_id):
            """从网络中提取ring bridge数据（适配层）"""
            rb_data = {}
            
            if not network or not hasattr(network, 'nodes'):
                return rb_data
            
            try:
                node = network.nodes.get(node_id)
                if not node or not hasattr(node, 'ring_bridge'):
                    return rb_data
                
                ring_bridge = node.ring_bridge
                
                # 提取ring_bridge input和output数据
                if hasattr(ring_bridge, 'ring_bridge_input'):
                    for direction, fifo in ring_bridge.ring_bridge_input.items():
                        if hasattr(fifo, 'queue') and fifo.queue:
                            rb_data[f"{direction}_in"] = {(node_id, node_id): list(fifo.queue)}
                
                if hasattr(ring_bridge, 'ring_bridge_output'):
                    for direction, fifo in ring_bridge.ring_bridge_output.items():
                        if hasattr(fifo, 'queue') and fifo.queue:
                            rb_data[f"{direction}_out"] = {(node_id, node_id): list(fifo.queue)}
                            
            except Exception as e:
                self.logger.debug(f"提取ring bridge数据失败: {e}")
            
            return rb_data

        def _get_iq_channel_data(self, network, node_id):
            """从网络中提取IQ channel数据（适配层）"""
            iq_ch_data = {}
            
            if not network or not hasattr(network, 'nodes'):
                return iq_ch_data
            
            try:
                node = network.nodes.get(node_id)
                if not node:
                    return iq_ch_data
                
                # 从IP接口提取l2h_fifos数据
                if hasattr(node, 'ip_interfaces'):
                    for ip_name, ip_interface in node.ip_interfaces.items():
                        if hasattr(ip_interface, 'l2h_fifos'):
                            for channel_name, fifo in ip_interface.l2h_fifos.items():
                                if hasattr(fifo, 'queue') and fifo.queue:
                                    full_channel_name = f"{ip_name}_{channel_name}"
                                    iq_ch_data[full_channel_name] = {node_id: list(fifo.queue)}
                            
            except Exception as e:
                self.logger.debug(f"提取IQ channel数据失败: {e}")
            
            return iq_ch_data

        def _get_eq_channel_data(self, network, node_id):
            """从网络中提取EQ channel数据（适配层）"""
            eq_ch_data = {}
            
            if not network or not hasattr(network, 'nodes'):
                return eq_ch_data
            
            try:
                node = network.nodes.get(node_id)
                if not node:
                    return eq_ch_data
                
                # 从IP接口提取h2l_fifos数据
                if hasattr(node, 'ip_interfaces'):
                    for ip_name, ip_interface in node.ip_interfaces.items():
                        if hasattr(ip_interface, 'h2l_fifos'):
                            for channel_name, fifo in ip_interface.h2l_fifos.items():
                                if hasattr(fifo, 'queue') and fifo.queue:
                                    full_channel_name = f"{ip_name}_{channel_name}"
                                    eq_ch_data[full_channel_name] = {node_id: list(fifo.queue)}
                            
            except Exception as e:
                self.logger.debug(f"提取EQ channel数据失败: {e}")
            
            return eq_ch_data

        def _get_crosspoint_data(self, network, node_id, direction):
            """从网络中提取crosspoint数据（适配层）"""
            cp_data = {}
            
            if not network or not hasattr(network, 'nodes'):
                return cp_data
            
            try:
                node = network.nodes.get(node_id)
                if not node:
                    return cp_data
                
                # 获取对应方向的CrossPoint
                if direction == 'horizontal' and hasattr(node, 'horizontal_cp'):
                    cp = node.horizontal_cp
                elif direction == 'vertical' and hasattr(node, 'vertical_cp'):
                    cp = node.vertical_cp
                else:
                    return cp_data
                
                # 提取CrossPoint状态信息
                cp_data = {
                    'arbitration_state': getattr(cp, 'arbitration_state', 'idle'),
                    'active_connections': getattr(cp, 'active_connections', []),
                    'priority_state': getattr(cp, 'priority_state', 'normal')
                }
                            
            except Exception as e:
                self.logger.debug(f"提取crosspoint数据失败: {e}")
            
            return cp_data

        def _get_flit_style(self, flit, use_highlight=True, expected_packet_id=0, highlight_color=None):
            """
            返回 (facecolor, alpha, linewidth, edgecolor)
            - facecolor 沿用调色板逻辑（高亮 / 调色板）
            - alpha / linewidth 由 flit.ETag_priority 决定
            """
            # E-Tag样式
            _ETAG_ALPHA = {"T0": 1.0, "T1": 1.0, "T2": 0.75}
            _ETAG_LW = {"T0": 2.5, "T1": 1, "T2": 0}
            _ETAG_EDGE = {"T0": "red", "T1": "black", "T2": "black"}
            
            # 获取基础颜色
            face = self._get_flit_color(flit, use_highlight, expected_packet_id, highlight_color)

            etag = getattr(flit, "ETag_priority", "T2")  # 缺省视为 T2
            alpha = _ETAG_ALPHA.get(etag, 1.0)
            lw = _ETAG_LW.get(etag, 0)
            edge_color = _ETAG_EDGE.get(etag, "black")

            return face, alpha, lw, edge_color

        def _get_flit_color(self, flit, use_highlight=True, expected_packet_id=1, highlight_color=None):
            """获取颜色，支持多种PID格式"""
            # 高亮模式：目标 flit → 红，其余 → 灰
            if use_highlight:
                hl = highlight_color or "red"
                return hl if getattr(flit, "packet_id", None) == expected_packet_id else "lightgrey"

            # 普通模式：直接取调色板色
            pid = getattr(flit, "packet_id", 0)
            return self._colors[pid % len(self._colors)]

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

    def __init__(self, config, network=None):
        """
        初始化CrossRing Link State Visualizer
        
        Args:
            config: CrossRing配置对象
            network: 网络模型对象（可选）
        """
        self.config = config
        self.network = network
        self.logger = logging.getLogger("CrossRingLinkStateVis")
        
        # 网络参数
        self.rows = getattr(config, 'NUM_ROW', getattr(config, 'num_row', 2))
        self.cols = getattr(config, 'NUM_COL', getattr(config, 'num_col', 3))
        self.num_nodes = self.rows * self.cols
        
        # 调色板
        self._colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
        
        # 当前显示的通道
        self.current_channel = "req"  # req/rsp/data
        
        # 高亮控制
        self.tracked_pid = None
        self.use_highlight = False
        
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
        # 创建主窗口
        self.fig = plt.figure(figsize=(20, 12))
        self.fig.suptitle('CrossRing Data Network', fontsize=16, fontweight='bold')
        
        # 左侧：网络拓扑视图
        self.ax = self.fig.add_axes([0.05, 0.15, 0.6, 0.8])
        self.ax.set_title('Data Network', fontsize=14)
        self.ax.set_aspect('equal')
        
        # 右侧：节点详细视图
        self.piece_ax = self.fig.add_axes([0.7, 0.15, 0.28, 0.8])
        self.piece_ax.set_title('Node Detail View', fontsize=14)
        
        # 创建PieceVisualizer
        self.piece_vis = self.PieceVisualizer(
            self.config, self.piece_ax, 
            highlight_callback=self._on_highlight_callback,
            parent=self
        )

    def _setup_controls(self):
        """设置控制按钮"""
        # REQ/RSP/DATA 按钮
        req_ax = self.fig.add_axes([0.05, 0.03, 0.05, 0.04])
        rsp_ax = self.fig.add_axes([0.12, 0.03, 0.05, 0.04])
        data_ax = self.fig.add_axes([0.19, 0.03, 0.05, 0.04])
        
        self.req_btn = Button(req_ax, "REQ")
        self.rsp_btn = Button(rsp_ax, "RSP")
        self.data_btn = Button(data_ax, "DATA")
        
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
            node_rect = Rectangle(
                (x - node_size/2, y - node_size/2),
                node_size, node_size,
                facecolor='lightblue', edgecolor='black', linewidth=1
            )
            self.ax.add_patch(node_rect)
            self.node_patches[node_id] = node_rect
            
            # 节点编号
            node_text = self.ax.text(
                x, y, str(node_id),
                fontsize=10, weight='bold', ha='center', va='center'
            )
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
            
            self.ax.set_xlim(min_x, max_x)
            self.ax.set_ylim(min_y, max_y)
        
        self.ax.axis('off')

    def _draw_links(self):
        """绘制链路"""
        # 存储链路和slot信息
        self.link_info = {}
        self.rect_info_map = {}  # slot_rect -> (link_id, flit, slot_idx)
        
        # 绘制水平链路
        for row in range(self.rows):
            for col in range(self.cols - 1):
                src_id = row * self.cols + col
                dest_id = row * self.cols + col + 1
                link_id = f"h_{src_id}_{dest_id}"
                
                self._draw_link_frame(src_id, dest_id, link_id)
        
        # 绘制垂直链路
        for row in range(self.rows - 1):
            for col in range(self.cols):
                src_id = row * self.cols + col
                dest_id = (row + 1) * self.cols + col
                link_id = f"v_{src_id}_{dest_id}"
                
                self._draw_link_frame(src_id, dest_id, link_id)

    def _draw_link_frame(self, src, dest, link_id, queue_fixed_length=1.6, slice_num=7):
        """绘制链路框架（基于原版逻辑）"""
        src_pos = self.node_positions[src]
        dest_pos = self.node_positions[dest]
        
        # 判断方向
        if src_pos[1] == dest_pos[1]:  # 水平链路
            direction = 'horizontal'
            # 计算链路位置
            start_x = src_pos[0] + 0.2
            end_x = dest_pos[0] - 0.2
            center_y = src_pos[1]
            
            # 绘制上下两条链路线
            upper_y = center_y + 0.15
            lower_y = center_y - 0.15
            
            # 上方链路线
            upper_line = Line2D([start_x, end_x], [upper_y, upper_y], 
                               color='blue', linewidth=2, alpha=0.7)
            self.ax.add_line(upper_line)
            
            # 下方链路线
            lower_line = Line2D([start_x, end_x], [lower_y, lower_y], 
                               color='blue', linewidth=2, alpha=0.7)
            self.ax.add_line(lower_line)
            
            # 绘制slots
            slot_width = (end_x - start_x) / slice_num
            for i in range(slice_num):
                slot_x = start_x + i * slot_width + slot_width/2
                
                # 上方slot
                upper_slot = Rectangle(
                    (slot_x - slot_width/4, upper_y - 0.05),
                    slot_width/2, 0.1,
                    facecolor='white', edgecolor='black', linewidth=0.5
                )
                self.ax.add_patch(upper_slot)
                self.rect_info_map[upper_slot] = (link_id, None, f"upper_{i}")
                
                # 下方slot
                lower_slot = Rectangle(
                    (slot_x - slot_width/4, lower_y - 0.05),
                    slot_width/2, 0.1,
                    facecolor='white', edgecolor='black', linewidth=0.5
                )
                self.ax.add_patch(lower_slot)
                self.rect_info_map[lower_slot] = (link_id, None, f"lower_{i}")
                
        else:  # 垂直链路
            direction = 'vertical'
            # 计算链路位置
            start_y = src_pos[1] - 0.2
            end_y = dest_pos[1] + 0.2
            center_x = src_pos[0]
            
            # 绘制左右两条链路线
            left_x = center_x - 0.15
            right_x = center_x + 0.15
            
            # 左侧链路线
            left_line = Line2D([left_x, left_x], [start_y, end_y], 
                              color='blue', linewidth=2, alpha=0.7)
            self.ax.add_line(left_line)
            
            # 右侧链路线
            right_line = Line2D([right_x, right_x], [start_y, end_y], 
                               color='blue', linewidth=2, alpha=0.7)
            self.ax.add_line(right_line)
            
            # 绘制slots
            slot_height = (end_y - start_y) / slice_num
            for i in range(slice_num):
                slot_y = start_y + i * slot_height + slot_height/2
                
                # 左侧slot
                left_slot = Rectangle(
                    (left_x - 0.05, slot_y - slot_height/4),
                    0.1, slot_height/2,
                    facecolor='white', edgecolor='black', linewidth=0.5
                )
                self.ax.add_patch(left_slot)
                self.rect_info_map[left_slot] = (link_id, None, f"left_{i}")
                
                # 右侧slot
                right_slot = Rectangle(
                    (right_x - 0.05, slot_y - slot_height/4),
                    0.1, slot_height/2,
                    facecolor='white', edgecolor='black', linewidth=0.5
                )
                self.ax.add_patch(right_slot)
                self.rect_info_map[right_slot] = (link_id, None, f"right_{i}")

    def _draw_selection_box(self):
        """绘制选择框"""
        if self._selected_node in self.node_positions:
            node_pos = self.node_positions[self._selected_node]
            self.click_box = Rectangle(
                (node_pos[0] - 0.3, node_pos[1] - 0.3),
                0.6, 0.6,
                facecolor="none", edgecolor="red", 
                linewidth=1.2, linestyle="--"
            )
            self.ax.add_patch(self.click_box)

    def _connect_events(self):
        """连接事件"""
        self.fig.canvas.mpl_connect('button_press_event', self._on_click)

    def _on_click(self, event):
        """处理点击事件"""
        if event.inaxes != self.ax:
            return
        
        # 检查是否点击了slot
        for rect in self.rect_info_map:
            contains, _ = rect.contains(event)
            if contains:
                link_id, flit, slot_idx = self.rect_info_map[rect]
                if flit:
                    self._on_flit_click(flit)
                return
        
        # 检查是否点击了节点
        for node_id, pos in self.node_positions.items():
            dx = event.xdata - pos[0]
            dy = event.ydata - pos[1]
            distance = np.sqrt(dx*dx + dy*dy)
            
            if distance <= 0.3:  # 节点半径
                self._select_node(node_id)
                break

    def _select_node(self, node_id):
        """选择节点"""
        if node_id == self._selected_node:
            return
        
        self._selected_node = node_id
        
        # 更新选择框
        if hasattr(self, 'click_box'):
            self.click_box.remove()
        self._draw_selection_box()
        
        # 更新右侧详细视图
        self.piece_vis.draw_piece_for_node(node_id, self.network)
        
        self.fig.canvas.draw_idle()
        self.logger.info(f"选中节点: {node_id}")

    def _on_flit_click(self, flit):
        """处理flit点击"""
        pid = getattr(flit, "packet_id", None)
        if pid is not None:
            self._track_packet(pid)

    def _track_packet(self, packet_id):
        """追踪包"""
        self.tracked_pid = packet_id
        self.use_highlight = True
        
        # 同步PieceVisualizer的高亮状态
        self.piece_vis.sync_highlight(self.use_highlight, self.tracked_pid)
        
        self.logger.info(f"开始追踪包: {packet_id}")

    def _on_channel_select(self, channel):
        """通道选择回调"""
        self.current_channel = channel
        self.logger.info(f"切换到通道: {channel}")
        
        # 重新绘制当前状态
        self.update()

    def _on_clear_highlight(self, event):
        """清除高亮回调"""
        self.tracked_pid = None
        self.use_highlight = False
        
        # 同步PieceVisualizer
        self.piece_vis.sync_highlight(self.use_highlight, self.tracked_pid)
        
        # 清除所有slot颜色
        for rect in self.rect_info_map:
            rect.set_facecolor('white')
        
        self.fig.canvas.draw_idle()
        self.logger.info("清除高亮")

    def _on_toggle_tags(self, event):
        """切换标签显示"""
        self.logger.info("切换标签显示")

    def _on_highlight_callback(self, packet_id, flit_id):
        """高亮回调"""
        self._track_packet(packet_id)

    def update(self, networks=None, cycle=None, skip_pause=False):
        """更新显示"""
        if networks is None and self.network is None:
            return
        
        network = networks if networks is not None else self.network
        
        # 更新链路状态
        self._update_link_state(network)
        
        # 更新右侧节点详细视图
        self.piece_vis.draw_piece_for_node(self._selected_node, network)
        
        if not skip_pause:
            self.fig.canvas.draw_idle()
    
    def _update_link_state(self, network):
        """更新链路状态"""
        if not network or not hasattr(network, 'links'):
            return
        
        try:
            # 重置所有slot颜色
            for rect in self.rect_info_map:
                rect.set_facecolor('white')
                self.rect_info_map[rect] = (self.rect_info_map[rect][0], None, self.rect_info_map[rect][2])
            
            # 更新链路中的slot数据
            for link_id, link in network.links.items():
                if hasattr(link, 'slices'):
                    for slice_idx, slice_obj in enumerate(link.slices):
                        if hasattr(slice_obj, 'slot') and slice_obj.slot:
                            slot = slice_obj.slot
                            if hasattr(slot, 'valid') and slot.valid:
                                self._update_slot_visual(link_id, slice_idx, slot)
                                
        except Exception as e:
            self.logger.debug(f"更新链路状态失败: {e}")
    
    def _update_slot_visual(self, link_id, slice_idx, slot):
        """更新单个slot的视觉效果"""
        # 查找对应的slot rectangle
        for rect, (rect_link_id, _, rect_slot_idx) in self.rect_info_map.items():
            if rect_link_id == link_id and str(slice_idx) in rect_slot_idx:
                # 更新flit信息
                self.rect_info_map[rect] = (rect_link_id, slot, rect_slot_idx)
                
                # 设置颜色
                if self.use_highlight and hasattr(slot, 'packet_id'):
                    if str(slot.packet_id) == str(self.tracked_pid):
                        color = 'red'
                    else:
                        color = 'lightgray'
                else:
                    # 根据packet_id设置颜色
                    pid = getattr(slot, 'packet_id', 0)
                    color = self._colors[int(pid) % len(self._colors)] if pid else 'lightblue'
                
                rect.set_facecolor(color)
                break
    
    def set_network(self, network):
        """设置网络模型"""
        self.network = network
    
    def get_selected_node(self):
        """获取当前选中的节点"""
        return self._selected_node
    
    def show(self):
        """显示可视化界面"""
        plt.show()
    
    def save_figure(self, filename):
        """保存图片"""
        self.fig.savefig(filename, dpi=300, bbox_inches='tight')
        self.logger.info(f"图片已保存到: {filename}")


# 演示函数
def create_demo_network():
    """创建演示网络"""
    from types import SimpleNamespace
    
    # 创建简单的网络结构用于演示
    network = SimpleNamespace()
    network.nodes = {}
    network.links = {}
    
    # 创建演示节点
    for i in range(6):
        node = SimpleNamespace()
        node.inject_direction_fifos = {}
        node.eject_input_fifos = {}
        node.channel_buffer = {}
        node.ip_eject_channel_buffers = {}
        node.ip_interfaces = {}
        node.ring_bridge = SimpleNamespace()
        node.ring_bridge.ring_bridge_input = {}
        node.ring_bridge.ring_bridge_output = {}
        network.nodes[i] = node
    
    return network


if __name__ == "__main__":
    # 简单演示
    from types import SimpleNamespace
    
    # 创建配置
    config = SimpleNamespace(
        NUM_ROW=2, NUM_COL=3,
        IQ_OUT_FIFO_DEPTH=8, EQ_IN_FIFO_DEPTH=8,
        RB_IN_FIFO_DEPTH=4, RB_OUT_FIFO_DEPTH=4,
        IQ_CH_FIFO_DEPTH=4, EQ_CH_FIFO_DEPTH=4,
        SLICE_PER_LINK=8
    )
    
    # 创建演示网络
    demo_network = create_demo_network()
    
    # 创建可视化器
    visualizer = CrossRingLinkStateVisualizer(config, demo_network)
    
    print("🎪 CrossRing Link State Visualizer 演示")
    print("点击节点可切换详细视图")
    print("使用底部按钮控制显示模式")
    
    # 显示
    visualizer.show()