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
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import plotly.offline as pyo
from dash import Dash, dcc, html, Input, Output, State, callback
import dash_bootstrap_components as dbc
from plotly.colors import qualitative
from collections import defaultdict, deque
import copy
import threading
import time
import webbrowser
from typing import Dict, List, Any, Optional, Tuple, Union
import json
from src.noc.base.config import BaseNoCConfig
from src.noc.crossring.config import CrossRingConfig

# from src.noc.visualization.crossring_node_visualizer import CrossRingNodeVisualizer  # 暂时禁用matplotlib版本
from src.noc.base.model import BaseNoCModel
from .color_manager import ColorManager
from .style_manager import VisualizationStyleManager

# 移除了logging依赖


# ---------- lightweight flit proxy for snapshot rendering ----------
class _FlitProxy:
    __slots__ = ("packet_id", "flit_id", "etag_priority", "itag_h", "itag_v", "flit_repr", "channel", "current_node_id", "flit_position")

    def __init__(self, pid, fid, etag, ih, iv, flit_repr=None, channel=None, current_node_id=None, flit_position=None):
        self.packet_id = pid
        self.flit_id = fid
        self.etag_priority = etag
        self.itag_h = ih
        self.itag_v = iv
        self.flit_repr = flit_repr
        self.channel = channel
        self.current_node_id = current_node_id
        self.flit_position = flit_position

    def __repr__(self):
        itag = "H" if self.itag_h else ("V" if self.itag_v else "")
        return f"(pid={self.packet_id}, fid={self.flit_id}, ET={self.etag_priority}, IT={itag})"


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

        # 样式管理器
        self.color_manager = ColorManager()
        self.style_manager = VisualizationStyleManager(self.color_manager)

        # 当前显示的通道
        self.current_channel = "data"  # req/rsp/data，默认显示data通道

        # 高亮控制
        self.tracked_pid = None
        self.use_highlight = False

        # 标签显示模式
        self.show_tags_mode = False

        # 播放控制状态
        self._is_paused = False
        self._current_speed = 1  # 更新间隔
        self._current_cycle = 0
        self._last_update_time = time.time()

        # 历史回放功能
        self.history = deque(maxlen=50)  # 保存最近50个周期的历史
        self._play_idx = None  # 当前回放索引，None表示实时模式

        # 状态显示文本
        self._status_text = None

        # 选中的节点
        self._selected_node = 0

        # 初始化链路数据结构 - 需要在绘制前初始化
        self.link_info = {}
        self.link_traces = {}  # link_id -> trace indices
        self.slice_traces = {}  # (link_id, slice_idx) -> trace index
        self.flit_data = {}  # (link_id, slice_idx) -> flit info
        
        # 同步控制：用于协调主进程和可视化进程
        self._update_ready = threading.Event()  # 可视化更新完成信号
        self._paused = False  # 暂停状态
        self._step_mode = False  # 单步模式

        # 创建图形界面
        self._setup_gui()
        self._draw_static_elements()

        # 启动服务器标志
        self._server_started = False

    def _setup_gui(self):
        """设置GUI布局"""
        # 创建Dash应用
        self.app = Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])

        # 创建子图：左侧网络拓扑，右侧节点详情
        self.fig = make_subplots(rows=1, cols=2, column_widths=[0.8, 0.3], subplot_titles=["网络拓扑", "节点详情"], specs=[[{"type": "scatter"}, {"type": "scatter"}]])

        model_name = getattr(self._parent_model, "model_name", "NoC")
        self.fig.update_layout(
            height=1000,  # 增加画布高度
            width=1400,  # 增加画布宽度
            showlegend=False,
            margin=dict(l=0, r=0, t=0, b=0),  # 保持原有边距
            plot_bgcolor="white",  # 设置绘图区域背景为白色
            paper_bgcolor="white",  # 设置整个图形背景为白色
        )

        # 根据拓扑行数动态调整子图标题位置
        # 行数越多，网络图越高，标题需要更靠下以避免与节点重叠
        title_y_position = min(0.85 + (self.rows - 3) * 0.05, 1)  # 基准0.85，每增加一行降低0.05，最低0.75

        self.fig.update_annotations(font_size=16, font_color="black", y=title_y_position, showarrow=False)  # 动态计算的标题位置

        # 初始化网络和节点视图
        self._setup_network_view()
        self._setup_node_view()

        # 设置Dash布局
        self._setup_dash_layout()

    def _setup_network_view(self):
        """设置网络拓扑视图"""
        # 初始化空的网络图
        self.fig.add_trace(
            go.Scatter(
                x=[],
                y=[],
                mode="markers+text",
                marker=dict(size=30, color="lightblue", line=dict(color="black", width=2), symbol="square"),
                text=[],
                textposition="middle center",
                hovertemplate="节点 %{text}<extra></extra>",
                name="nodes",
            ),
            row=1,
            col=1,
        )

        # 网络视图设置 - 确保坐标轴比例1:1以显示正方形slice
        self.fig.update_xaxes(showgrid=False, zeroline=False, showticklabels=False, row=1, col=1)
        self.fig.update_yaxes(showgrid=False, zeroline=False, showticklabels=False, scaleanchor="x", scaleratio=1, row=1, col=1)

    def _setup_node_view(self):
        """设置节点详细视图"""
        # 初始化空的节点详情图
        self.fig.add_trace(go.Scatter(x=[], y=[], mode="markers", marker=dict(size=10), name="node_details"), row=1, col=2)

        # 节点视图设置
        self.fig.update_xaxes(showgrid=False, zeroline=False, showticklabels=False, row=1, col=2)
        self.fig.update_yaxes(showgrid=False, zeroline=False, showticklabels=False, row=1, col=2)

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

    def _setup_dash_layout(self):
        """设置Dash应用布局"""
        self.app.layout = dbc.Container(
            [
                dcc.Interval(id="interval-timer", interval=800, n_intervals=0),  # 0.8秒更新间隔
                dbc.Row(
                    [
                        dbc.Col(
                            [
                                dbc.ButtonGroup(
                                    [
                                        dbc.Button("请求网络", id="req-btn", color="primary", outline=True),
                                        dbc.Button("响应网络", id="rsp-btn", color="primary", outline=True),
                                        dbc.Button("数据网络", id="data-btn", color="primary", outline=True, active=True),
                                    ],
                                    className="mb-2",
                                ),
                                dbc.ButtonGroup(
                                    [
                                        dbc.Button("清除高亮", id="clear-btn", color="secondary"),
                                        dbc.Button("显示标签", id="tags-btn", color="info"),
                                    ],
                                    className="mb-2 ms-2",
                                ),
                            ],
                            width=12,
                        )
                    ]
                ),
                dbc.Row([dbc.Col([dcc.Graph(id="main-graph", figure=self.fig, style={"height": "90vh"})], width=12)]),  # 更大的显示区域
                dbc.Row([dbc.Col([html.Small(id="status-display", className="text-muted")], width=12)], style={"height": "1vh"}),  # 使用小字体  # 限制状态显示高度
            ],
            fluid=True,
        )

        # 设置回调函数
        self._setup_callbacks()

    def _setup_callbacks(self):
        """设置Dash回调函数"""

        @self.app.callback(
            Output("main-graph", "figure"),
            [
                Input("req-btn", "n_clicks"),
                Input("rsp-btn", "n_clicks"),
                Input("data-btn", "n_clicks"),
                Input("clear-btn", "n_clicks"),
                Input("tags-btn", "n_clicks"),
                Input("main-graph", "clickData"),
                Input("interval-timer", "n_intervals"),
            ],  # 添加定时器输入
            [State("main-graph", "figure")],
        )
        def update_graph(req_clicks, rsp_clicks, data_clicks, clear_clicks, tags_clicks, click_data, n_intervals, current_fig):
            from dash import callback_context

            ctx = callback_context
            if not ctx.triggered:
                return self.fig

            trigger_id = ctx.triggered[0]["prop_id"].split(".")[0]

            # 处理通道切换
            if trigger_id in ["req-btn", "rsp-btn", "data-btn"]:
                channel_map = {"req-btn": "req", "rsp-btn": "rsp", "data-btn": "data"}
                self.current_channel = channel_map[trigger_id]
                self._update_network_title()

            # 处理高亮控制
            elif trigger_id == "clear-btn":
                self._on_clear_highlight(None)

            elif trigger_id == "tags-btn":
                self._on_toggle_tags(None)

            # 处理节点点击
            elif trigger_id == "main-graph" and click_data:
                self._handle_click_data(click_data)

            # 处理定时器更新 - 获取最新的仿真数据
            elif trigger_id == "interval-timer":
                try:
                    self._refresh_visualization_data()
                except Exception:
                    pass  # 静默处理更新错误

            return self.fig

        @self.app.callback(Output("status-display", "children"), [Input("interval-timer", "n_intervals")])  # 改为定时器触发状态更新
        def update_status(n_intervals):
            try:
                return self._get_status_display()
            except Exception as e:
                return html.Div(f"状态错误: {str(e)}", style={"color": "red", "font-size": "12px"})

    def _refresh_visualization_data(self):
        """刷新可视化数据 - 从仿真模型获取最新状态"""
        try:
            # 如果没有父模型，无法更新
            if not self._parent_model:
                return

            # 获取最新快照
            if hasattr(self._parent_model, "capture_network_snapshot"):
                current_cycle = getattr(self._parent_model, "cycle", 0)
                snapshot_data = self._parent_model.capture_network_snapshot()

                if snapshot_data and len(snapshot_data.get("links", {})) > 0:
                    # 添加到历史记录
                    self.history.append((current_cycle, snapshot_data))

                    # 限制历史记录数量，保留最近100个快照
                    if len(self.history) > 100:
                        self.history.pop(0)

                    # 渲染最新快照
                    self._render_from_snapshot(snapshot_data)

        except Exception as e:
            # 打印错误信息用于调试
            print(f"可视化数据刷新错误: {e}")
            pass

    def _handle_click_data(self, click_data):
        """处理点击事件"""
        if "points" in click_data and len(click_data["points"]) > 0:
            point = click_data["points"][0]
            if "customdata" in point:
                node_id = point["customdata"]
                self._select_node(node_id)

    def _get_status_display(self):
        """获取状态显示内容"""
        try:
            # 获取模型状态
            paused = getattr(self._parent_model, "_paused", False) if self._parent_model else False
            current_cycle = getattr(self._parent_model, "cycle", 0) if self._parent_model else 0

            status_color = "danger" if paused else "success"
            status_text = "暂停" if paused else "运行"

            return dbc.Alert(
                [
                    html.H6(f"状态: {status_text}", className="mb-1"),
                    html.P(f"周期: {current_cycle}", className="mb-1"),
                    html.P(f"追踪: {self.tracked_pid if self.tracked_pid else '无'}", className="mb-0"),
                ],
                color=status_color,
            )
        except Exception as e:
            # 出错时返回简单的错误信息
            return html.Div(f"状态更新错误: {str(e)}", style={"color": "red"})

    def _draw_static_elements(self):
        """绘制静态元素"""
        # 计算节点位置
        self.node_positions = {}

        # 根据SLICE_PER_LINK动态调整节点间距
        slice_per_link = getattr(self.config.basic_config, "SLICE_PER_LINK", 8)
        base_spacing_x = 2.0
        base_spacing_y = 1.5
        spacing_factor = max(1.0, slice_per_link / 8.0 * 0.8 + 0.2)
        spacing_x = base_spacing_x * spacing_factor
        spacing_y = base_spacing_y * spacing_factor

        node_x, node_y, node_text, node_ids = [], [], [], []

        for row in range(self.rows):
            for col in range(self.cols):
                node_id = row * self.cols + col
                x = col * spacing_x
                y = row * spacing_y  # 让第0行在最上面
                self.node_positions[node_id] = (x, y)

                node_x.append(x)
                node_y.append(y)
                node_text.append(str(node_id))
                node_ids.append(node_id)

        # 更新节点trace
        self.fig.data[0].x = node_x
        self.fig.data[0].y = node_y
        self.fig.data[0].text = node_text
        self.fig.data[0].customdata = node_ids

        # 设置坐标轴范围
        if self.node_positions:
            margin = 1.0
            min_x = min(pos[0] for pos in self.node_positions.values()) - margin
            max_x = max(pos[0] for pos in self.node_positions.values()) + margin
            min_y = min(pos[1] for pos in self.node_positions.values()) - margin
            max_y = max(pos[1] for pos in self.node_positions.values()) + margin

            self.fig.update_xaxes(range=[min_x, max_x], row=1, col=1)
            self.fig.update_yaxes(range=[min_y, max_y], row=1, col=1)

        # 绘制链路
        self._draw_links()

    def _draw_links(self):
        """绘制链路为Plotly traces"""
        link_traces = []

        # 根据实际的网络结构动态绘制链路
        if hasattr(self._parent_model, "links"):
            for link_id in self._parent_model.links.keys():
                src_id, dest_id = self._parse_link_id(link_id)
                if src_id is not None and dest_id is not None and src_id != dest_id:
                    trace = self._create_link_trace(src_id, dest_id, link_id)
                    if trace:
                        link_traces.extend(trace)
        else:
            # 绘制水平链路
            for row in range(self.rows):
                for col in range(self.cols - 1):
                    src_id = row * self.cols + col
                    dest_id = row * self.cols + col + 1
                    link_id = f"link_{src_id}_TR_{dest_id}"
                    trace = self._create_link_trace(src_id, dest_id, link_id)
                    if trace:
                        link_traces.extend(trace)

            # 绘制垂直链路
            for row in range(self.rows - 1):
                for col in range(self.cols):
                    src_id = row * self.cols + col
                    dest_id = (row + 1) * self.cols + col
                    link_id = f"link_{src_id}_TD_{dest_id}"
                    trace = self._create_link_trace(src_id, dest_id, link_id)
                    if trace:
                        link_traces.extend(trace)

        # 添加链路 traces到图中
        for trace in link_traces:
            self.fig.add_trace(trace, row=1, col=1)

    def _create_link_trace(self, src_id, dest_id, link_id):
        """创建Plotly链路 trace - 模仿原始matplotlib FancyArrowPatch样式"""
        if src_id not in self.node_positions or dest_id not in self.node_positions:
            return None

        src_pos = self.node_positions[src_id]
        dest_pos = self.node_positions[dest_id]

        # 计算链路方向
        dx = dest_pos[0] - src_pos[0]
        dy = dest_pos[1] - src_pos[1]

        if dx == 0 and dy == 0:
            return []  # 跳过自环

        dist = (dx * dx + dy * dy) ** 0.5
        unit_dx, unit_dy = dx / dist, dy / dist

        # 垂直偏移向量（用于分离双向箭头）
        perp_dx = -unit_dy
        perp_dy = unit_dx

        # 节点边界和箭头偏移参数（与原始版本一致）
        node_radius = 0.2
        arrow_offset = 0.08  # 双向箭头间距

        # 计算箭头起止点（从节点边缘开始）
        start_x = src_pos[0] + unit_dx * node_radius
        start_y = src_pos[1] + unit_dy * node_radius
        end_x = dest_pos[0] - unit_dx * node_radius
        end_y = dest_pos[1] - unit_dy * node_radius

        traces = []

        # 绘制两个方向的箭头（模仿原始的FancyArrowPatch）
        directions = [("forward", 1), ("backward", -1)]

        for direction_name, offset_sign in directions:
            # 计算偏移后的起止点
            offset_start_x = start_x + perp_dx * arrow_offset * offset_sign
            offset_start_y = start_y + perp_dy * arrow_offset * offset_sign
            offset_end_x = end_x + perp_dx * arrow_offset * offset_sign
            offset_end_y = end_y + perp_dy * arrow_offset * offset_sign

            # 反向箭头需要交换起止点
            if direction_name == "backward":
                offset_start_x, offset_end_x = offset_end_x, offset_start_x
                offset_start_y, offset_end_y = offset_end_y, offset_start_y

            # 创建线条
            line_trace = go.Scatter(
                x=[offset_start_x, offset_end_x],
                y=[offset_start_y, offset_end_y],
                mode="lines",
                line=dict(color="black", width=1.5),
                hoverinfo="none",
                showlegend=False,
                name=f"line_{direction_name}_{link_id}",
                opacity=0.8,
            )
            traces.append(line_trace)

            # 根据箭头方向选择合适的符号
            arrow_dx = offset_end_x - offset_start_x
            arrow_dy = offset_end_y - offset_start_y

            # 根据方向选择箭头符号
            if abs(arrow_dx) > abs(arrow_dy):
                # 水平方向
                arrow_symbol = "triangle-right" if arrow_dx > 0 else "triangle-left"
            else:
                # 垂直方向
                arrow_symbol = "triangle-up" if arrow_dy > 0 else "triangle-down"

            # 创建箭头标记（单独的trace）
            arrow_trace = go.Scatter(
                x=[offset_end_x],
                y=[offset_end_y],
                mode="markers",
                marker=dict(symbol=arrow_symbol, size=10, color="black", line=dict(width=1, color="black")),
                hoverinfo="none",
                showlegend=False,
                name=f"arrow_{direction_name}_{link_id}",
                opacity=0.8,
            )
            traces.append(arrow_trace)

        # 创建链路的slice
        self._draw_link_frame(src_id, dest_id, link_id)

        return traces

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

        return demo_link_id

    def _draw_link_frame(self, src, dest, link_id, slice_num=None):
        """绘制链路框架和slice slots - Plotly版本

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
                    slice_num = link.num_slices
                elif hasattr(link, "ring_slices") and isinstance(link.ring_slices, dict):
                    first_channel = list(link.ring_slices.keys())[0]
                    slice_num = len(link.ring_slices[first_channel])
                else:
                    slice_num = getattr(self.config.basic_config, "SLICE_PER_LINK", 8)
            else:
                if src == dest:  # 自环链路
                    slice_num = getattr(self.config.basic_config, "SELF_LINK_SLICES", 2)
                else:  # 正常链路
                    slice_num = getattr(self.config.basic_config, "SLICE_PER_LINK", 8)

        # 为链路创建slice可视化
        self._create_link_slices(src, dest, link_id, slice_num)

    def _create_link_slices(self, src_id, dest_id, link_id, slice_num):
        """为链路创建slice可视化 - 模仿原始matplotlib Rectangle样式"""
        if src_id not in self.node_positions or dest_id not in self.node_positions:
            return

        src_pos = self.node_positions[src_id]
        dest_pos = self.node_positions[dest_id]

        # 计算基本参数（与原始版本一致）
        dx = dest_pos[0] - src_pos[0]
        dy = dest_pos[1] - src_pos[1]
        dist = np.sqrt(dx * dx + dy * dy)

        if dist == 0:
            return

        unit_dx = dx / dist
        unit_dy = dy / dist

        # 垂直偏移向量（用于分离双向箭头）
        perp_dx = -unit_dy
        perp_dy = unit_dx

        # 参数（与原始版本一致）
        slot_size = 0.1
        slot_spacing = 0.00
        side_offset = 0.18  # 距离箭头的距离
        node_radius = 0.2

        # 链路起始和结束位置（考虑节点边界）
        start_x = src_pos[0] + unit_dx * node_radius
        start_y = src_pos[1] + unit_dy * node_radius
        end_x = dest_pos[0] - unit_dx * node_radius
        end_y = dest_pos[1] - unit_dy * node_radius

        # 计算slice排列区域的起始点
        link_length = np.sqrt((end_x - start_x) ** 2 + (end_y - start_y) ** 2)
        total_length = slice_num * slot_size + (slice_num - 1) * slot_spacing
        start_offset = (link_length - total_length) / 2

        # 跳过首尾slice的显示
        visible_slice_num = max(0, slice_num - 2)
        if visible_slice_num <= 0:
            return

        # 在链路两侧都绘制slice（模仿原始的side1和side2）
        for side_name, side_sign in [("side1", 1), ("side2", -1)]:
            slice_x = []
            slice_y = []
            slice_ids = []

            for i in range(1, slice_num - 1):  # 跳过i=0和i=slice_num-1
                # 计算沿链路方向的位置
                along_link_dist = start_offset + i * (slot_size + slot_spacing)
                progress = along_link_dist / link_length if link_length > 0 else 0

                # 沿链路方向的中心点
                center_x = start_x + progress * (end_x - start_x)
                center_y = start_y + progress * (end_y - start_y)

                # 垂直于链路方向的偏移
                slot_x = center_x + perp_dx * side_offset * side_sign
                slot_y = center_y + perp_dy * side_offset * side_sign

                slice_x.append(slot_x)
                slice_y.append(slot_y)
                slice_ids.append((link_id, i))

            # 创建这一侧的slice traces（模仿原始的Rectangle样式）
            if slice_x:
                slice_trace = go.Scatter(
                    x=slice_x,
                    y=slice_y,
                    mode="markers",
                    marker=dict(
                        size=12,  # 调整大小确保正方形显示
                        color="rgba(0,0,0,0)",  # 透明填充（facecolor="none"）
                        line=dict(color="gray", width=0.8),  # 灰色边框
                        symbol="square",
                        sizemode="diameter",  # 确保size指的是直径而不是面积
                    ),
                    hovertemplate="链路: %{customdata[0]}<br>Slice: %{customdata[1]}<br>Side: " + side_name + "<extra></extra>",
                    customdata=slice_ids,
                    name=f"slices_{side_name}_{link_id}",
                    showlegend=False,
                )

                # 添加到图中
                self.fig.add_trace(slice_trace, row=1, col=1)

                # 记录slice traces
                for i, slice_id in enumerate(slice_ids):
                    self.slice_traces[slice_id] = len(self.fig.data) - 1
                    self.flit_data[slice_id] = None

    def _track_packet(self, packet_id):
        """追踪包 - Plotly版本"""
        self.tracked_pid = packet_id
        self.use_highlight = True

        # 重新应用所有flit的样式（Plotly版本）
        self._reapply_all_flit_styles()

    def _reapply_all_flit_styles(self):
        """重新应用所有flit的样式，用于高亮状态改变后 - Plotly版本"""
        # 更新所有slice的颜色和样式
        for slice_id, flit in self.flit_data.items():
            if flit is not None:
                self._update_slice_visual(slice_id, flit)

    def _update_slice_visual(self, slice_id, flit):
        """更新单个slice的可视效果 - Plotly版本"""
        if slice_id not in self.slice_traces:
            return

        trace_idx = self.slice_traces[slice_id]
        if trace_idx >= len(self.fig.data):
            return

        # 获取flit颜色和样式
        face_color, line_width, edge_color = self.style_manager.get_flit_style(
            flit, use_highlight=self.use_highlight, expected_packet_id=self.tracked_pid, highlight_color="red", show_tags_mode=getattr(self, "show_tags_mode", False)
        )

        # 直接更新trace的marker属性（Plotly要求创建新的属性对象）
        trace = self.fig.data[trace_idx]

        if flit:
            # 有flit时显示颜色和包信息
            # 转换RGBA颜色到RGB字符串格式
            if len(face_color) >= 3:
                color_str = f"rgb({int(face_color[0]*255)},{int(face_color[1]*255)},{int(face_color[2]*255)})"
            else:
                color_str = "lightblue"

            # 创建新的marker对象替换原有的
            trace.marker = dict(size=14, color=color_str, symbol="square", line=dict(width=line_width, color=edge_color), sizemode="diameter")
            trace.text = [str(getattr(flit, "packet_id", "?"))]
        else:
            # 无flit时显示为灰色空slot
            trace.marker = dict(size=12, color="rgba(0,0,0,0)", symbol="square", line=dict(width=0.8, color="gray"), sizemode="diameter")  # 透明填充
            trace.text = [""]

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

    # Plotly事件处理已经通过Dash回调实现，不需要此方法
    pass

    def _on_mouse_click(self, event):
        """处理鼠标点击事件"""
        if event.inaxes != self.link_ax:
            return

        # 首先检查flit点击（slots有更高优先级）
        if hasattr(self, "rect_info_map"):
            for rect in self.rect_info_map:
                contains, _ = rect.contains(event)
                if contains:
                    link_ids, flit, slot_idx = self.rect_info_map[rect]
                    if flit:
                        self._on_flit_click(flit)
                    return

        # 然后检查节点点击，使用距离计算
        if hasattr(self, "node_positions"):
            for node_id, pos in self.node_positions.items():
                dx = event.xdata - pos[0]
                dy = event.ydata - pos[1]
                distance = (dx * dx + dy * dy) ** 0.5  # sqrt

                if distance <= 0.3:  # 节点点击半径
                    self._select_node(node_id)
                    break

    def _on_flit_click(self, flit):
        """处理flit点击事件"""
        # 兼容字典和对象两种格式获取packet_id
        if isinstance(flit, dict):
            pid = flit.get("packet_id", None)
        else:
            pid = getattr(flit, "packet_id", None)

        if pid is not None:
            self._track_packet(pid)

        # 显示flit详细信息（使用_format_flit_info支持repr）
        # if hasattr(self, "node_vis") and self.node_vis:
        #     # 格式化flit信息并显示在右下角
        #     flit_info = self._format_flit_info(flit)
        #     self.node_vis.info_text.set_text(flit_info)
        #     self.node_vis.current_highlight_flit = flit

    def _select_node(self, node_id):
        """选择节点并更新右侧详细视图"""
        if node_id == self._selected_node:
            return

        self._selected_node = node_id

        # 更新选择框（红色虚线矩形）
        if hasattr(self, "click_box"):
            self.click_box.remove()
        self._draw_selection_box()

        # 使用快照数据更新右侧详细视图
        if self._play_idx is not None and self._play_idx < len(self.history):
            # 回放模式：使用当前回放周期数据
            replay_cycle, _ = self.history[self._play_idx]
            # self.node_vis.render_node_from_snapshot(node_id, replay_cycle)
        elif self.history:
            # 实时模式：使用最新快照数据
            latest_cycle, _ = self.history[-1]
            # self.node_vis.render_node_from_snapshot(node_id, latest_cycle)

        # 更新节点标题
        self._update_node_title()
        # Plotly不需要手动触发重绘
        pass

    def _draw_selection_box(self):
        """更新选中节点的高亮显示"""
        if hasattr(self, "node_positions") and self._selected_node in self.node_positions:
            # 在Plotly中，我们通过更改节点颜色来显示选中状态
            node_colors = ["red" if i == self._selected_node else "lightblue" for i in range(len(self.node_positions))]

            # 更新节点trace的颜色
            if len(self.fig.data) > 0:
                self.fig.data[0].marker.color = node_colors

    def _on_window_close(self, event):
        """处理窗口关闭事件"""
        if hasattr(self, "_parent_model") and self._parent_model:
            if hasattr(self._parent_model, "cleanup_visualization"):
                self._parent_model.cleanup_visualization()

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
        elif event.key.lower() == "q":  # Q键退出可视化
            self._quit_visualization()

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
            # if hasattr(self, "node_vis") and self.node_vis and self._selected_node is not None:
            #     self.node_vis.render_node_from_snapshot(self._selected_node, cycle)
            # Plotly不需要手动触发重绘

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
            # if hasattr(self, "node_vis") and self.node_vis and self._selected_node is not None:
            #     self.node_vis.render_node_from_snapshot(self._selected_node, cycle)
            # Plotly不需要手动触发重绘

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
                    try:
                        self.fig.canvas.draw_idle()
                    except Exception as e:
                        print(f"警告：matplotlib绘图错误: {e}")
                        pass
                status = "暂停"
            else:
                # 退出暂停：回到实时模式
                self._play_idx = None
                status = "继续"

    def _quit_visualization(self):
        """退出可视化，触发模型的清理方法"""
        if hasattr(self, "_parent_model") and self._parent_model:
            # 调用模型的cleanup_visualization方法
            if hasattr(self._parent_model, "cleanup_visualization"):
                self._parent_model.cleanup_visualization()

    def _reset_view(self):
        """重置视图"""
        self.tracked_pid = None
        self.use_highlight = False
        # if hasattr(self, "piece_vis"):
        #     self.node_vis.sync_highlight(False, None)

    def _change_speed(self, faster=True):
        """改变仿真速度"""
        if hasattr(self, "_parent_model") and self._parent_model:
            current_interval = getattr(self._parent_model, "_visualization_frame_interval", 0.5)
            if faster:
                new_interval = max(0.05, current_interval * 0.75)
            else:
                new_interval = min(5.0, current_interval * 1.25)
            self._parent_model._visualization_frame_interval = new_interval

    def _set_max_speed(self):
        """设置最大速度"""
        if hasattr(self, "_parent_model") and self._parent_model:
            self._parent_model._visualization_frame_interval = 0.05

    def _set_slow_speed(self):
        """设置慢速"""
        if hasattr(self, "_parent_model") and self._parent_model:
            self._parent_model._visualization_frame_interval = 2.0

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

        # 更新标题
        self._update_network_title()

        # 重新绘制当前状态
        if self._parent_model:
            self.update(self._parent_model)

        try:
            self.fig.canvas.draw_idle()
        except Exception as e:
            print(f"警告：matplotlib绘图错误: {e}")
            pass

    def _update_network_title(self):
        """更新网络标题"""
        channel_name = {"req": "请求网络", "rsp": "响应网络", "data": "数据网络"}.get(self.current_channel, f"{self.current_channel.upper()}网络")
        if hasattr(self.fig, "layout") and hasattr(self.fig.layout, "annotations") and len(self.fig.layout.annotations) > 0:
            self.fig.layout.annotations[0].text = channel_name

    def _update_node_title(self):
        """更新节点标题"""
        if hasattr(self.fig, "layout") and hasattr(self.fig.layout, "annotations") and len(self.fig.layout.annotations) > 1:
            self.fig.layout.annotations[1].text = f"节点 {self._selected_node}"

    def _on_clear_highlight(self, event):
        """清除高亮回调"""
        self.tracked_pid = None
        self.use_highlight = False

        # 同步CrossRingNodeVisualizer
        # self.node_vis.sync_highlight(self.use_highlight, self.tracked_pid)

        # 清除右下角信息显示
        # if hasattr(self, "node_vis") and self.node_vis and hasattr(self.node_vis, "info_text"):
        #     self.node_vis.info_text.set_text("")
        #     if hasattr(self.node_vis, "current_highlight_flit"):
        #         self.node_vis.current_highlight_flit = None

        # 立即重新应用所有flit的样式
        self._reapply_all_flit_styles()

        try:
            self.fig.canvas.draw_idle()
        except Exception as e:
            print(f"警告：matplotlib绘图错误: {e}")
            pass

    def _on_toggle_tags(self, event):
        """切换标签显示"""
        # 切换标签模式状态
        self.show_tags_mode = not self.show_tags_mode

        # 更新按钮文本
        if self.show_tags_mode:
            self.tags_btn.label.set_text("隐藏标签")
        else:
            self.tags_btn.label.set_text("显示标签")

        # 同步节点可视化器的标签模式
        # if hasattr(self, "node_vis") and self.node_vis:
        #     self.node_vis.sync_tags_mode(self.show_tags_mode)

        # 立即重新应用所有flit的样式
        self._reapply_all_flit_styles()

        try:
            self.fig.canvas.draw_idle()
        except Exception as e:
            print(f"警告：matplotlib绘图错误: {e}")
            pass

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
            # 优先使用传入的cycle，其次使用模型的cycle，最后使用递增值
            if cycle is not None:
                effective_cycle = cycle
            elif network and hasattr(network, "cycle"):
                effective_cycle = network.cycle
            elif network and hasattr(network, "_current_cycle"):
                effective_cycle = network._current_cycle
            else:
                # 避免cycle重复：如果历史不为空，使用最后一个cycle+1
                effective_cycle = (self.history[-1][0] + 1) if self.history else 0
            self._save_history_snapshot(network, effective_cycle)

        # 统一使用快照数据更新显示（无论实时还是回放模式）
        if self._play_idx is not None and len(self.history) > self._play_idx:
            # 回放模式：使用指定历史快照
            replay_cycle, snapshot_data = self.history[self._play_idx]
            self._render_from_snapshot(snapshot_data)
            # 节点视图也使用回放数据
            # self.node_vis.render_node_from_snapshot(self._selected_node, replay_cycle)
        else:
            # 实时模式：使用最新保存的快照（刚刚保存的）
            if self.history:
                latest_cycle, latest_snapshot = self.history[-1]
                self._render_from_snapshot(latest_snapshot)
                # 节点视图也从最新快照获取数据
                # self.node_vis.render_node_from_snapshot(self._selected_node, latest_cycle)

        # 更新状态显示
        self._update_status_display()

        # Plotly会通过Dash自动更新，不需要手动重绘

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
                                        """从slot中提取flit信息，包含完整repr"""
                                        if slot and hasattr(slot, "flit") and slot.flit:
                                            flit_data = {
                                                "packet_id": getattr(slot.flit, "packet_id", None),
                                                "flit_id": getattr(slot.flit, "flit_id", None),
                                                "etag_priority": getattr(slot.flit, "etag_priority", "T2"),
                                                "itag_h": getattr(slot.flit, "itag_h", False),
                                                "itag_v": getattr(slot.flit, "itag_v", False),
                                                "current_node_id": getattr(slot.flit, "current_node_id", None),
                                                "flit_position": getattr(slot.flit, "flit_position", None),
                                                "channel": slot_channel,
                                            }

                                            # 保存flit的完整repr信息
                                            try:
                                                flit_data["flit_repr"] = repr(slot.flit)
                                            except Exception as e:
                                                flit_data["flit_repr"] = f"repr failed: {e}"

                                            return {
                                                "valid": getattr(slot, "valid", False),
                                                "flit": flit_data,
                                            }
                                        return None

                                    # RingSlice重构后，使用新的接口获取slot数据
                                    slot_info = None

                                    # 方法1：尝试获取当前slot（输出位置）
                                    current_slot = slice_obj.peek_current_slot(channel) if hasattr(slice_obj, "peek_current_slot") else None
                                    if current_slot:
                                        slot_info = extract_flit_from_slot(current_slot, channel)

                                    # 方法2：如果没有找到，尝试从内部pipeline获取
                                    if not slot_info and hasattr(slice_obj, "internal_pipelines"):
                                        pipeline = slice_obj.internal_pipelines.get(channel)
                                        if pipeline:
                                            # 检查output register
                                            if hasattr(pipeline, "output_valid") and pipeline.output_valid and hasattr(pipeline, "output_register"):
                                                slot_info = extract_flit_from_slot(pipeline.output_register, channel)

                                            # 检查internal queue
                                            if not slot_info and hasattr(pipeline, "internal_queue") and len(pipeline.internal_queue) > 0:
                                                # 获取队列中的第一个slot
                                                first_slot = list(pipeline.internal_queue)[0]
                                                slot_info = extract_flit_from_slot(first_slot, channel)

                                    # 只保存当前通道的slot数据
                                    slice_data["slots"][channel] = slot_info

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
                                flit_data = {
                                    "packet_id": getattr(slot, "packet_id", None),
                                    "flit_id": getattr(slot, "flit_id", None),
                                    "etag_priority": getattr(slot, "etag_priority", "T2"),
                                    "itag_h": getattr(slot, "itag_h", False),
                                    "itag_v": getattr(slot, "itag_v", False),
                                    "current_node_id": None,
                                    "flit_position": None,
                                    "channel": "req",
                                }

                                # 保存demo slot的repr信息
                                try:
                                    flit_data["flit_repr"] = repr(slot)
                                except Exception as e:
                                    flit_data["flit_repr"] = f"repr failed: {e}"

                                slot_info = {
                                    "valid": getattr(slot, "valid", True),
                                    "flit": flit_data,
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
            # if hasattr(model, "nodes") and hasattr(self, "node_vis"):
            #     self.node_vis.save_history_snapshot(model, cycle)

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
        """从快照渲染 - Plotly版本"""
        try:
            # 第一步：重置所有slice为默认状态
            for slice_id in self.flit_data:
                self.flit_data[slice_id] = None
                self._update_slice_visual(slice_id, None)

            # 第二步：从完整快照中提取当前通道数据
            current_channel = getattr(self, "current_channel", "data")

            # 直接使用统一的快照格式
            self._render_from_snapshot_data(snapshot_data.get("links", {}), current_channel)

        except Exception as e:
            pass  # 静默忽略渲染错误

    def _render_from_snapshot_data(self, links_snapshot, current_channel):
        """从快照数据渲染 - Plotly版本"""
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

                    # 创建slice_id
                    slice_id = (link_id, slice_idx_int)

                    # 处理所有slot，查找有效的flit
                    flit_found = None
                    for slot_key, slot_info in slots.items():
                        if slot_info and slot_info.get("valid", False):
                            flit_data = slot_info.get("flit", {})
                            if flit_data:
                                flit_count += 1

                                # 创建临时flit对象
                                flit_found = _FlitProxy(
                                    pid=flit_data.get("packet_id"),
                                    fid=flit_data.get("flit_id"),
                                    etag=flit_data.get("etag_priority", "T2"),
                                    ih=flit_data.get("itag_h", False),
                                    iv=flit_data.get("itag_v", False),
                                    flit_repr=flit_data.get("flit_repr"),
                                    channel=flit_data.get("channel"),
                                    current_node_id=flit_data.get("current_node_id"),
                                    flit_position=flit_data.get("flit_position"),
                                )
                                break  # 每个slice只显示一个flit

                    # 更新slice可视化
                    if slice_id in self.flit_data:
                        self.flit_data[slice_id] = flit_found
                        self._update_slice_visual(slice_id, flit_found)

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

    def set_network(self, network):
        """设置网络模型"""
        self._parent_model = network

    def get_selected_node(self):
        """获取当前选中的节点"""
        return self._selected_node

    def show(self, port=8050, debug=False):
        """显示可视化界面"""
        if not self._server_started:
            self._server_started = True
            
            # 配置Flask的日志级别，减少INFO输出
            import logging
            logging.getLogger("werkzeug").setLevel(logging.WARNING)
            
            # 在后台线程中运行Dash服务器，避免阻塞仿真
            def run_server():
                self.app.run(debug=False, port=port, host="localhost", dev_tools_hot_reload=False)
            
            # 启动后台线程
            server_thread = threading.Thread(target=run_server, daemon=True)
            server_thread.start()
            
            # 延迟打开浏览器，确保服务器已启动
            threading.Timer(2.0, lambda: webbrowser.open(f"http://localhost:{port}")).start()
            
            print(f"🌐 Dash应用运行在: http://localhost:{port} (后台线程)")
            print("💡 可视化已准备就绪，仿真可以开始更新")
            
            # 标记可视化已准备好接收更新
            self._update_ready.set()
