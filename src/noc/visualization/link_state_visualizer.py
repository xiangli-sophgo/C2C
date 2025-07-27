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
# 移除了logging依赖

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
            self.cols = getattr(config, 'NUM_COL', 3)
            self.rows = getattr(config, 'NUM_ROW', 2)
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
            
            # 清理轴并移除任何默认元素（如监控图）
            self.ax.clear()
            self.ax.axis("off")
            self.ax.set_aspect("equal")
            
            # 移除所有可能的线条、网格、tick等
            self.ax.set_xticks([])
            self.ax.set_yticks([])
            self.ax.grid(False)
            
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
                pass  # print(f"警告: 提取inject queue数据失败: {e}")
            
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
                pass  # print(f"警告: 提取eject queue数据失败: {e}")
            
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
                pass  # print(f"警告: 提取ring bridge数据失败: {e}")
            
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
                pass  # print(f"警告: 提取IQ channel数据失败: {e}")
            
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
                pass  # print(f"警告: 提取EQ channel数据失败: {e}")
            
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

    def __init__(self, config, network=None):
        """
        初始化CrossRing Link State Visualizer
        
        Args:
            config: CrossRing配置对象
            network: 网络模型对象（可选）
        """
        self.config = config
        self.network = network
        self._parent_model = network  # 建立与模型的连接
        # 移除logger，使用简单的调试输出
        
        # 网络参数
        self.rows = getattr(config, 'NUM_ROW', 2)
        self.cols = getattr(config, 'NUM_COL', 3)
        self.num_nodes = self.rows * self.cols
        
        # 调色板
        self._colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
        
        # 当前显示的通道
        self.current_channel = "data"  # req/rsp/data，默认显示data通道
        
        # 高亮控制
        self.tracked_pid = None
        self.use_highlight = False
        
        # 播放控制状态
        self._parent_model = None  # 将在初始化时设置
        self._is_paused = False
        self._current_speed = 2  # 更新间隔
        self._current_cycle = 0
        self._last_update_time = time.time()
        
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
        # 创建主窗口
        self.fig = plt.figure(figsize=(14, 10))
        self.fig.suptitle('CrossRing Data Network', fontsize=16, fontweight='bold')
        
        # 左侧：网络拓扑视图
        self.ax = self.fig.add_axes([0.05, 0.15, 0.6, 0.8])
        self.ax.set_title('Data Network', fontsize=14)
        self.ax.set_aspect('equal')
        
        # 状态显示区域（左上角）
        self._setup_status_display()
        
        # 右侧：节点详细视图 - 调整尺寸和位置
        self.piece_ax = self.fig.add_axes([0.68, 0.05, 0.31, 0.9])
        self.piece_ax.set_title('Node Detail View', fontsize=12)
        
        # 移除任何可能的默认图表元素
        self.piece_ax.clear()
        self.piece_ax.set_title('Node Detail View', fontsize=12)
        self.piece_ax.axis('off')
        self.piece_ax.set_aspect('equal')
        
        # 创建节点详细视图可视化器
        self.piece_vis = self.PieceVisualizer(
            config=self.config,
            ax=self.piece_ax,
            highlight_callback=self._on_highlight_callback,
            parent=self
        )
        
    def _setup_status_display(self):
        """设置状态显示"""
        # 在左上角创建状态文本
        self._status_text = self.ax.text(
            0.02, 0.98, '',
            transform=self.ax.transAxes,
            fontsize=10,
            verticalalignment='top',
            fontfamily='monospace',
            bbox=dict(boxstyle='round,pad=0.3', facecolor='lightgray', alpha=0.8)
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
        paused = getattr(self._parent_model, '_paused', False) if self._parent_model else False
        speed = getattr(self._parent_model, '_visualization_update_interval', 2) if self._parent_model else 2
        cycle = getattr(self._parent_model, 'cycle', 0) if self._parent_model else 0
        
        # 播放状态图标
        status_icon = "[PAUSE]" if paused else "[PLAY] "
        
        # 构建状态文本
        status_text = f"""状态: {status_icon}
周期: {cycle}
间隔: {speed} cycles
通道: {self.current_channel.upper()}
追踪: {self.tracked_pid if self.tracked_pid else '无'}"""

        self._status_text.set_text(status_text)

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
        self.node_pair_slots = {}  # 存储每对节点之间的slot位置信息
        
        # 根据实际的网络结构动态绘制链路
        if hasattr(self.network, 'links'):
            # 从网络中获取实际存在的链路
            for link_id in self.network.links.keys():
                # 解析link_id来确定源和目标节点
                src_id, dest_id = self._parse_link_id(link_id)
                if src_id is not None and dest_id is not None and src_id != dest_id:
                    # 跳过自环链路，只绘制节点间连接
                    self._draw_link_frame(src_id, dest_id, link_id)
        else:
            # 回退到基本的网格链路绘制
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
        """
        try:
            parts = link_id.split('_')
            if len(parts) >= 4 and parts[0] == 'link':
                src_id = int(parts[1])
                if len(parts) == 4:  # link_0_TR_1
                    dest_id = int(parts[3])
                elif len(parts) == 5:  # link_0_TL_TR_0
                    dest_id = int(parts[4])
                else:
                    return None, None
                return src_id, dest_id
        except (ValueError, IndexError):
            pass
        return None, None

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
            if hasattr(self.network, 'links') and link_id in self.network.links:
                link = self.network.links[link_id]
                if hasattr(link, 'num_slices'):
                    slice_num = link.num_slices  # 使用链路的实际slice数量
                elif hasattr(link, 'ring_slices') and isinstance(link.ring_slices, dict):
                    # ring_slices是字典，获取任一通道的slice数量
                    first_channel = list(link.ring_slices.keys())[0]
                    slice_num = len(link.ring_slices[first_channel])
                else:
                    slice_num = getattr(self.config.basic_config, 'SLICE_PER_LINK', 8)
            else:
                # 根据链路类型确定slice数量
                if src == dest:  # 自环链路
                    slice_num = getattr(self.config.basic_config, 'SELF_LINK_SLICES', 2)
                else:  # 正常链路
                    slice_num = getattr(self.config.basic_config, 'NORMAL_LINK_SLICES', 8)
            
        src_pos = self.node_positions[src]
        dest_pos = self.node_positions[dest]
        
        # 计算基本参数
        dx = dest_pos[0] - src_pos[0]
        dy = dest_pos[1] - src_pos[1] 
        dist = np.sqrt(dx*dx + dy*dy)
        
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
                directions = [
                    ('forward', 1, f"{link_id}_fwd"),   # src -> dest
                    ('backward', -1, f"{link_id}_bwd")  # dest -> src
                ]
                
                for direction_name, offset_sign, arrow_id in directions:
                    # 计算偏移后的起止点
                    offset_start_x = start_x + perp_dx * arrow_offset * offset_sign
                    offset_start_y = start_y + perp_dy * arrow_offset * offset_sign
                    offset_end_x = end_x + perp_dx * arrow_offset * offset_sign
                    offset_end_y = end_y + perp_dy * arrow_offset * offset_sign
                    
                    # 反向箭头需要交换起止点
                    if direction_name == 'backward':
                        offset_start_x, offset_end_x = offset_end_x, offset_start_x
                        offset_start_y, offset_end_y = offset_end_y, offset_start_y
                    
                    # 绘制箭头
                    arrow = FancyArrowPatch(
                        (offset_start_x, offset_start_y),
                        (offset_end_x, offset_end_y),
                        arrowstyle='-|>',
                        mutation_scale=15,
                        color='black',
                        linewidth=1.5,
                        alpha=0.8,
                        zorder=1
                    )
                    self.ax.add_patch(arrow)
            
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
        link_length = np.sqrt((end_x - start_x)**2 + (end_y - start_y)**2)
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
            for side_name, side_sign in [('side1', 1), ('side2', -1)]:
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
                    slot = Rectangle(
                        (slot_x, slot_y),
                        slot_size, slot_size,
                        facecolor='none',
                        edgecolor='gray',
                        linewidth=0.8,
                        linestyle='--',
                        alpha=0.7
                    )
                    self.ax.add_patch(slot)
                    
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
        self.fig.canvas.mpl_connect('key_press_event', self._on_key_press)

    def _on_click(self, event):
        """处理点击事件"""
        if event.inaxes != self.ax:
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
        
        # 同步PieceVisualizer的高亮状态
        self.piece_vis.sync_highlight(self.use_highlight, self.tracked_pid)
        
        # print(f"开始追踪包: {packet_id}")  # 删除debug输出

    def _on_key_press(self, event):
        """处理键盘事件"""
        if event.key == ' ':  # 空格键暂停/继续
            self._toggle_pause()
        elif event.key == 'r':  # R键重置视图
            self._reset_view()
        elif event.key == 'R':  # Shift+R重放/重启仿真
            self._restart_simulation()
        elif event.key == '+' or event.key == '=':  # 加号键加速
            self._change_speed(faster=True)
        elif event.key == '-':  # 减号键减速
            self._change_speed(faster=False)
        elif event.key.lower() in ['1', '2', '3']:  # 数字键切换通道
            channels = ['req', 'rsp', 'data']
            if int(event.key) <= len(channels):
                self._on_channel_select(channels[int(event.key) - 1])
        elif event.key == 'h' or event.key == '?':  # H键或?键显示帮助
            self._show_help()
        elif event.key == 'f':  # F键切换到最快速度
            self._set_max_speed()
        elif event.key == 's':  # S键切换到慢速
            self._set_slow_speed()
            
        # 更新状态显示
        self._update_status_display()
            
    def _toggle_pause(self):
        """切换暂停状态"""
        if hasattr(self, '_parent_model') and self._parent_model:
            # 创建暂停属性如果不存在
            if not hasattr(self._parent_model, '_paused'):
                self._parent_model._paused = False
            self._parent_model._paused = not self._parent_model._paused
            status = "暂停" if self._parent_model._paused else "继续"
            # print(f"⏯️  仿真{status}")  # 删除debug输出
            
    def _reset_view(self):
        """重置视图"""
        self.tracked_pid = None
        self.use_highlight = False
        if hasattr(self, 'piece_vis'):
            self.piece_vis.sync_highlight(False, None)
        # print("重置视图")  # 删除debug输出
        
    def _restart_simulation(self):
        """重启/重放仿真"""
        if hasattr(self, '_parent_model') and self._parent_model:
            # 重置仿真状态
            if hasattr(self._parent_model, 'cycle'):
                pass  # print(f"重启仿真 (从周期 {self._parent_model.cycle} 重置到 0)")
                # 注意：这里只是示例，实际重启需要模型支持
                # self._parent_model.reset_simulation()  # 如果模型有此方法
            else:
                print("重启仿真")
        
    def _change_speed(self, faster=True):
        """改变仿真速度"""
        if hasattr(self, '_parent_model') and self._parent_model:
            current_interval = getattr(self._parent_model, '_visualization_update_interval', 2)
            if faster:
                new_interval = max(1, current_interval - 1)
            else:
                new_interval = min(50, current_interval + 1)
            self._parent_model._visualization_update_interval = new_interval
            pass  # print(f"速度调整: 间隔 {new_interval}")
            
    def _restart_simulation(self):
        """重启/重放仿真"""
        if hasattr(self, '_parent_model') and self._parent_model:
            # 重置模型状态
            if hasattr(self._parent_model, 'reset'):
                self._parent_model.reset()
                pass  # print("重启仿真")
            else:
                pass  # print("重启功能暂不可用")
                
    def _set_max_speed(self):
        """设置最大速度"""
        if hasattr(self, '_parent_model') and self._parent_model:
            self._parent_model._visualization_update_interval = 1
            pass  # print("设置为最大速度")
            
    def _set_slow_speed(self):
        """设置慢速"""
        if hasattr(self, '_parent_model') and self._parent_model:
            self._parent_model._visualization_update_interval = 10
            pass  # print("设置为慢速")
            
    def _show_help(self):
        """显示键盘快捷键帮助"""
        help_text = """
CrossRing可视化控制键:
========================================
播放控制:
  空格键  - 暂停/继续仿真
  Shift+R - 重启/重放仿真
  r       - 重置视图和高亮

速度控制:
  +/=     - 加速 (减少更新间隔)
  -       - 减速 (增加更新间隔)
  f       - 最大速度 (间隔=1)
  s       - 慢速 (间隔=10)

视图控制:
  1/2/3   - 切换到REQ/RSP/DATA通道
  h或?    - 显示此帮助信息

交互:
  点击节点 - 查看详细信息
  点击flit - 开始追踪包

状态显示在左上角实时更新
========================================
        """
        pass  # print(help_text)

    def _on_channel_select(self, channel):
        """通道选择回调"""
        self.current_channel = channel
        pass  # print(f"切换到通道: {channel}")
        
        # 更新标题
        self._update_network_title()
        
        # 重新绘制当前状态
        if self.network:
            self.update(self.network)
        
        self.fig.canvas.draw_idle()
        
    def _update_network_title(self):
        """更新网络标题"""
        channel_name = {
            'req': 'Request Network',
            'rsp': 'Response Network', 
            'data': 'Data Network'
        }.get(self.current_channel, f'{self.current_channel.upper()} Network')
        
        self.ax.set_title(channel_name, fontsize=14)
        
        # 移除重复的主标题，只保留axis标题

    def _on_clear_highlight(self, event):
        """清除高亮回调"""
        self.tracked_pid = None
        self.use_highlight = False
        
        # 同步PieceVisualizer
        self.piece_vis.sync_highlight(self.use_highlight, self.tracked_pid)
        
        # 清除所有slot颜色
        for rect in self.rect_info_map:
            rect.set_facecolor('none')
        
        self.fig.canvas.draw_idle()
        pass  # print("清除高亮")

    def _on_toggle_tags(self, event):
        """切换标签显示"""
        pass  # print("切换标签显示")

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
        
        # 更新状态显示
        self._update_status_display()
        
        if not skip_pause:
            self.fig.canvas.draw_idle()
    
    def _update_link_state(self, network):
        """更新链路状态"""
        if not network:
            return
        
        try:
            # 重置所有slot为默认状态（虚线边框，无填充）
            reset_count = 0
            for rect in self.rect_info_map:
                rect.set_facecolor('none')
                rect.set_edgecolor('gray')
                rect.set_linewidth(0.8)
                rect.set_linestyle('--')
                rect.set_alpha(0.7)
                # 重置数据绑定，保留link_ids和slot_id，清除flit数据
                link_ids, _, slot_id = self.rect_info_map[rect]
                self.rect_info_map[rect] = (link_ids, None, slot_id)
                reset_count += 1
            
            # print(f"重置了 {reset_count} 个slot为默认状态")  # 删除debug输出
            
            # 简化的网络状态检查（移除debug输出）
            if hasattr(network, 'links'):
                link_count = len(network.links)
                
                # 简化flit检查（移除debug输出）
                found_any_flit = False
                active_flit_count = 0
                for link_id, link in network.links.items():
                    src_id, dest_id = self._parse_link_id(link_id)
                    if src_id is not None and dest_id is not None and src_id != dest_id:
                        if hasattr(link, 'get_ring_slice') and hasattr(link, 'num_slices'):
                            try:
                                for slice_idx in range(min(3, link.num_slices)):
                                    slice_obj = link.get_ring_slice('data', slice_idx)
                                    if hasattr(slice_obj, 'current_slots'):
                                        for ch, slot in slice_obj.current_slots.items():
                                            if slot and hasattr(slot, 'flit') and slot.flit:
                                                active_flit_count += 1
                                                found_any_flit = True
                            except:
                                pass
                
                # if active_flit_count > 0:
                #     print(f"当前传输: {active_flit_count} 个flit")
                            
                # 移除不必要的debug信息
            # else:
            #     print("错误: 网络没有links属性")
            
            # 尝试多种网络数据结构来更新链路状态
            self._update_from_network_links(network)
            self._update_from_network_nodes(network)
                                
        except Exception as e:
            import traceback
            pass  # print(f"错误: 更新链路状态失败: {e}")
            # print(f"详细错误: {traceback.format_exc()}")
            
    def _update_from_network_links(self, network):
        """从network.links更新链路状态"""
        if not hasattr(network, 'links'):
            return
            
        current_channel = getattr(self, 'current_channel', 'req')
        # print(f"更新链路状态，当前通道: {current_channel}")  # 删除debug输出
        
        for link_id, link in network.links.items():
            # 跳过自环链路
            src_id, dest_id = self._parse_link_id(link_id)
            if src_id is not None and dest_id is not None and src_id == dest_id:
                continue  # 跳过自环链路
                
            # 使用正确的CrossRing ring_slices结构
            if hasattr(link, 'get_ring_slice') and hasattr(link, 'num_slices'):
                # 遍历所有通道和slice位置
                for slice_idx in range(link.num_slices):
                    # 检查当前选择的通道
                    try:
                        slice_obj = link.get_ring_slice(current_channel, slice_idx)
                        if hasattr(slice_obj, 'current_slots'):
                            for channel, slot in slice_obj.current_slots.items():
                                if slot and hasattr(slot, 'valid') and slot.valid and hasattr(slot, 'flit') and slot.flit:
                                    # 检查flit是否属于当前选择的通道
                                    if channel == current_channel:
                                        self._update_slot_visual(link_id, slice_idx, slot.flit)
                        elif hasattr(slice_obj, 'slot') and slice_obj.slot:
                            slot = slice_obj.slot
                            if hasattr(slot, 'valid') and slot.valid and hasattr(slot, 'flit') and slot.flit:
                                # 简单检查flit的通道属性
                                if self._should_display_flit(slot.flit, current_channel):
                                    self._update_slot_visual(link_id, slice_idx, slot.flit)
                    except Exception as e:
                        # 忽略无效的slice访问
                        pass
            elif hasattr(link, 'slots'):
                # 直接的slots结构
                for slot_idx, slot in enumerate(link.slots):
                    if slot and hasattr(slot, 'valid') and slot.valid and hasattr(slot, 'flit') and slot.flit:
                        if self._should_display_flit(slot.flit, current_channel):
                            self._update_slot_visual(link_id, slot_idx, slot.flit)
                            
    def _should_display_flit(self, flit, channel):
        """判断是否应该显示该flit（基于通道过滤）"""
        # 检查flit的通道属性（CrossRingFlit使用channel属性）
        flit_channel = getattr(flit, 'channel', None)
        flit_type = getattr(flit, 'flit_type', None)
        
        # 打印调试信息（首次遇到新类型时）
        if not hasattr(self, '_logged_flit_types'):
            self._logged_flit_types = set()
        
        flit_info = f"channel={flit_channel}, type={flit_type}"
        if flit_info not in self._logged_flit_types:
            self._logged_flit_types.add(flit_info)
            # print(f"发现flit: {flit_info}")  # 删除debug输出
        
        # 如果flit有明确的channel属性（CrossRing使用这个）
        if flit_channel:
            return flit_channel.lower() == channel.lower()
            
        # 如果flit有type属性，根据type判断
        if flit_type:
            flit_type_str = str(flit_type).lower()
            if 'req' in flit_type_str or 'request' in flit_type_str:
                return channel.lower() == 'req'
            elif 'rsp' in flit_type_str or 'response' in flit_type_str:
                return channel.lower() == 'rsp' 
            elif 'data' in flit_type_str:
                return channel.lower() == 'data'
        
        # 默认不显示（如果无法确定通道）
        return False
                        
    def _update_from_network_nodes(self, network):
        """从network.nodes的输出缓冲区更新链路状态"""
        if not hasattr(network, 'nodes'):
            return
            
        # 遍历节点，查找输出到链路的数据
        for node_id, node in network.nodes.items():
            if hasattr(node, 'output_buffers'):
                for direction, buffer in node.output_buffers.items():
                    if hasattr(buffer, 'queue') and buffer.queue:
                        # 根据节点和方向构造链路ID
                        link_id = self._get_link_id_from_node_direction(node_id, direction)
                        if link_id:
                            for idx, flit in enumerate(buffer.queue):
                                if flit:
                                    self._update_slot_visual(link_id, idx, flit)
                                    
    def _link_id_matches(self, link_id, pattern):
        """检查link_id是否匹配带通配符的模式"""
        # pattern格式: link_1_*_0
        # link_id格式: link_1_TR_0 或 link_1_TL_0
        pattern_parts = pattern.split('_')
        link_parts = link_id.split('_')
        
        if len(pattern_parts) != len(link_parts):
            return False
            
        for p, l in zip(pattern_parts, link_parts):
            if p != '*' and p != l:
                return False
        return True
    
    def _get_link_id_from_node_direction(self, node_id, direction):
        """根据节点ID和方向获取对应的链路ID"""
        # 根据网络拓扑计算链路ID - 使用CrossRing的实际格式
        if direction in ['TR', 'right', 'east']:
            return f"link_{node_id}_TR_{node_id + 1}"
        elif direction in ['TL', 'left', 'west']:
            return f"link_{node_id - 1}_TR_{node_id}"  
        elif direction in ['TD', 'down', 'south']:
            cols = getattr(self.config, 'NUM_COL', 3)
            return f"link_{node_id}_TD_{node_id + cols}"
        elif direction in ['TU', 'up', 'north']:
            cols = getattr(self.config, 'NUM_COL', 3)
            return f"link_{node_id - cols}_TD_{node_id}"
        return None
    
    def _update_slot_visual(self, link_id, slice_idx, slot):
        """更新单个slot的视觉效果"""
        # 因为我们跳过了首尾slice（range(1, slice_num-1)），需要调整索引匹配
        # slice_idx=0对应不显示，slice_idx=1对应显示的第0个slot，以此类推
        if slice_idx == 0 or slice_idx >= (getattr(self.config, 'SLICE_PER_LINK', 7) - 1):
            return  # 跳过首尾slice
            
        # 查找对应的slot rectangle
        slot_found = False
        for rect, (rect_link_ids, _, rect_slot_idx) in self.rect_info_map.items():
            # rect_link_ids可能是字符串（旧格式）或列表（新格式）
            if isinstance(rect_link_ids, str):
                rect_link_ids = [rect_link_ids]
            
            # 检查link_id是否匹配任何一个方向
            link_matched = False
            for rect_link_id in rect_link_ids:
                if rect_link_id == link_id or ('*' in rect_link_id and self._link_id_matches(link_id, rect_link_id)):
                    link_matched = True
                    break
            
            if link_matched and str(slice_idx) in rect_slot_idx:
                # 更新flit信息
                self.rect_info_map[rect] = (rect_link_ids, slot, rect_slot_idx)
                
                # 获取flit样式（颜色、透明度、边框等）
                face_color, alpha, line_width, edge_color = self._get_flit_style(
                    slot, 
                    use_highlight=self.use_highlight,
                    expected_packet_id=self.tracked_pid,
                    highlight_color='red'
                )
                
                # 应用样式到rectangle
                rect.set_facecolor(face_color)
                rect.set_alpha(alpha)
                rect.set_edgecolor(edge_color)
                rect.set_linewidth(max(line_width, 0.8))  # 确保最小线宽
                rect.set_linestyle('-')  # 有flit时使用实线
                
                slot_found = True
                break
                
        # 移除过于频繁的调试输出
                
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
            return 'lightblue'  # 默认颜色
    
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
        pass  # print(f"图片已保存到: {filename}")


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
    
    pass  # print("CrossRing Link State Visualizer 演示")
    # print("点击节点可切换详细视图")
    # print("使用底部按钮控制显示模式")
    
    # 显示
    visualizer.show()