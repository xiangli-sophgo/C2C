"""
实时可视化主控器

整合Link和Node可视化器，提供实时动画控制和交互功能，包括：
- 动画播放控制
- 包追踪和高亮
- 性能监控面板
- 状态快照管理
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.widgets import Button, Slider, CheckButtons
import matplotlib.gridspec as gridspec
from typing import Dict, List, Any, Optional, Tuple, Callable, Union
import logging
import time
import threading
from dataclasses import dataclass, field
from enum import Enum
import copy
import sys

from .link_visualizer import BaseLinkVisualizer, SlotData, LinkStats, SlotState
from .crossring_node_visualizer import CrossRingNodeVisualizer, FlitProxy, FIFOData, CrossPointData

# 中文字体支持
plt.rcParams["font.sans-serif"] = ["SimHei", "Arial Unicode MS"]
plt.rcParams["axes.unicode_minus"] = False


class PlaybackState(Enum):
    """播放状态枚举"""

    STOPPED = "stopped"
    PLAYING = "playing"
    PAUSED = "paused"
    STEPPING = "stepping"


@dataclass
class NetworkSnapshot:
    """网络状态快照"""

    cycle: int
    timestamp: float
    nodes_data: Dict[int, dict] = field(default_factory=dict)
    links_data: Dict[str, dict] = field(default_factory=dict)
    stats: Dict[str, Any] = field(default_factory=dict)


class RealtimeVisualizer:
    """
    实时可视化主控器

    集成Link和Node可视化器，提供完整的CrossRing网络实时可视化功能
    """

    def __init__(self, config, model=None, update_interval: float = 0.1):
        """
        初始化实时可视化器

        Args:
            config: CrossRing配置对象
            model: CrossRing模型实例
            update_interval: 更新间隔(秒)
        """
        self.config = config
        self.model = model
        self.update_interval = update_interval
        self.logger = logging.getLogger("RealtimeVisualizer")

        # 网络参数
        self.num_nodes = getattr(config, "NUM_NODES", 6)
        self.num_rows = getattr(config, "NUM_ROW", 2)
        self.num_cols = getattr(config, "NUM_COL", 3)

        # 播放控制
        self.playback_state = PlaybackState.STOPPED
        self.current_cycle = 0
        self.playback_speed = 1.0
        self.auto_advance = True

        # 高亮控制
        self.use_highlight = False
        self.highlight_packet_id = None
        self.highlight_color = "red"

        # 数据存储
        self.snapshots = []  # 历史快照
        self.max_snapshots = 1000
        self.current_snapshot = None

        # 可视化器字典
        self.node_visualizers = {}  # node_id -> CrossRingNodeVisualizer
        self.link_visualizers = {}  # link_id -> BaseLinkVisualizer

        # GUI元素
        self.fig = None
        self.axes = {}
        self.control_widgets = {}
        self.animation = None

        # 性能监控
        self.performance_data = {"bandwidth": [], "latency": [], "congestion": [], "throughput": []}

        self._setup_gui()
        self._setup_animation()

    def _setup_gui(self):
        """设置GUI界面"""
        # 创建主窗口和子图布局
        self.fig = plt.figure(figsize=(16, 12))
        self.fig.suptitle("CrossRing网络实时可视化", fontsize=16, fontweight="bold")

        # 使用GridSpec创建复杂布局
        gs = gridspec.GridSpec(4, 4, figure=self.fig, height_ratios=[3, 3, 1, 1], width_ratios=[3, 3, 2, 2])

        # 节点可视化区域 (2x2网格显示节点)
        node_axes = {}
        for i in range(min(4, self.num_nodes)):  # 最多显示4个节点
            row = i // 2
            col = i % 2
            ax = self.fig.add_subplot(gs[row, col])
            node_axes[i] = ax

        # 链路状态显示区域
        link_ax = self.fig.add_subplot(gs[0:2, 2])

        # 性能监控区域
        perf_ax = self.fig.add_subplot(gs[0:2, 3])

        # 控制面板区域
        control_ax = self.fig.add_subplot(gs[2, :])

        # 信息显示区域
        info_ax = self.fig.add_subplot(gs[3, :])

        self.axes = {"nodes": node_axes, "links": link_ax, "performance": perf_ax, "controls": control_ax, "info": info_ax}

        # 创建可视化器实例
        self._create_visualizers()

        # 创建控制组件
        self._create_controls()

        # 创建性能监控图表
        self._setup_performance_monitor()

    def _create_visualizers(self):
        """创建各种可视化器"""
        # 创建节点可视化器
        for node_id, ax in self.axes["nodes"].items():
            if node_id < self.num_nodes:
                visualizer = CrossRingNodeVisualizer(self.config, ax, node_id, highlight_callback=self._on_highlight_callback)
                self.node_visualizers[node_id] = visualizer

        # 创建链路可视化器（显示主要链路）
        main_link_vis = BaseLinkVisualizer(ax=self.axes["links"], link_id="main_ring", num_slots=8)
        self.link_visualizers["main_ring"] = main_link_vis

    def _create_controls(self):
        """创建控制组件"""
        control_ax = self.axes["controls"]
        control_ax.set_xlim(0, 10)
        control_ax.set_ylim(0, 1)
        control_ax.axis("off")

        # 播放控制按钮
        play_ax = plt.axes([0.1, 0.25, 0.08, 0.05])
        pause_ax = plt.axes([0.2, 0.25, 0.08, 0.05])
        step_ax = plt.axes([0.3, 0.25, 0.08, 0.05])
        reset_ax = plt.axes([0.4, 0.25, 0.08, 0.05])

        self.control_widgets["play"] = Button(play_ax, "播放")
        self.control_widgets["pause"] = Button(pause_ax, "暂停")
        self.control_widgets["step"] = Button(step_ax, "单步")
        self.control_widgets["reset"] = Button(reset_ax, "重置")

        # 连接回调函数
        self.control_widgets["play"].on_clicked(self._on_play)
        self.control_widgets["pause"].on_clicked(self._on_pause)
        self.control_widgets["step"].on_clicked(self._on_step)
        self.control_widgets["reset"].on_clicked(self._on_reset)

        # 速度控制滑块
        speed_ax = plt.axes([0.55, 0.25, 0.2, 0.05])
        self.control_widgets["speed"] = Slider(speed_ax, "速度", 0.1, 5.0, valinit=self.playback_speed, valfmt="%.1fx")
        self.control_widgets["speed"].on_changed(self._on_speed_change)

        # 高亮控制
        highlight_ax = plt.axes([0.8, 0.25, 0.15, 0.05])
        self.control_widgets["highlight"] = CheckButtons(highlight_ax, ["包追踪"])
        self.control_widgets["highlight"].on_clicked(self._on_highlight_toggle)

        # 周期显示
        self.cycle_text = control_ax.text(0.05, 0.7, f"周期: {self.current_cycle}", fontsize=12, weight="bold", transform=control_ax.transAxes)

        # 状态显示
        self.status_text = control_ax.text(0.05, 0.3, f"状态: {self.playback_state.value}", fontsize=10, transform=control_ax.transAxes)

    def _setup_performance_monitor(self):
        """设置性能监控图表"""
        perf_ax = self.axes["performance"]
        perf_ax.set_title("性能监控", fontweight="bold")
        perf_ax.set_xlabel("周期")
        perf_ax.set_ylabel("指标值")

        # 初始化空的性能曲线
        self.perf_lines = {
            "bandwidth": perf_ax.plot([], [], "b-", label="带宽利用率")[0],
            "latency": perf_ax.plot([], [], "r-", label="平均延迟")[0],
            "congestion": perf_ax.plot([], [], "g-", label="拥塞级别")[0],
        }

        perf_ax.legend()
        perf_ax.grid(True, alpha=0.3)
        perf_ax.set_xlim(0, 100)
        perf_ax.set_ylim(0, 1)

    def _setup_animation(self):
        """设置动画"""
        self.animation = FuncAnimation(self.fig, self._animate_frame, interval=int(self.update_interval * 1000), blit=False, cache_frame_data=False)

    def _animate_frame(self, frame):
        """动画帧更新函数"""
        if self.playback_state == PlaybackState.PLAYING and self.model:
            try:
                # 推进模型一步
                if hasattr(self.model, "step"):
                    self.model.step()
                    self.current_cycle = getattr(self.model, "cycle", self.current_cycle + 1)

                # 获取当前状态并更新显示
                self._update_from_model()

                # 更新控制面板显示
                self._update_controls()

            except Exception as e:
                self.logger.error(f"动画更新出错: {e}")
                self.playback_state = PlaybackState.PAUSED

    def _update_from_model(self):
        """从模型更新可视化显示"""
        if not self.model:
            return

        try:
            # 获取模型当前状态
            network_data = self._extract_model_data()

            # 更新各个可视化器
            self._update_node_visualizers(network_data.get("nodes", {}))
            self._update_link_visualizers(network_data.get("links", {}))
            self._update_performance_monitor(network_data.get("stats", {}))

            # 保存快照
            snapshot = NetworkSnapshot(
                cycle=self.current_cycle,
                timestamp=time.time(),
                nodes_data=network_data.get("nodes", {}),
                links_data=network_data.get("links", {}),
                stats=network_data.get("stats", {}),
            )
            self._add_snapshot(snapshot)

        except Exception as e:
            self.logger.error(f"从模型更新可视化失败: {e}")

    def _extract_model_data(self) -> dict:
        """从模型提取可视化数据"""
        if not self.model:
            return {}

        network_data = {"nodes": {}, "links": {}, "stats": {}}

        try:
            # 提取节点数据
            if hasattr(self.model, "nodes"):
                for node_id, node in self.model.nodes.items():
                    network_data["nodes"][node_id] = self._extract_node_data(node)

            # 提取链路数据
            if hasattr(self.model, "links"):
                for link_id, link in self.model.links.items():
                    network_data["links"][link_id] = self._extract_link_data(link)

            # 提取统计数据
            if hasattr(self.model, "get_statistics"):
                network_data["stats"] = self.model.get_statistics()

        except Exception as e:
            self.logger.warning(f"提取模型数据部分失败: {e}")

        return network_data

    def _extract_node_data(self, node) -> dict:
        """从节点提取可视化数据"""
        node_data = {"inject_queues": {}, "eject_queues": {}, "ring_bridge": {}, "crosspoints": {}}

        try:
            # 提取注入队列数据
            if hasattr(node, "inject_direction_fifos"):
                for direction, fifo in node.inject_direction_fifos.items():
                    if hasattr(fifo, "entries"):
                        flits = []
                        for flit in fifo.entries:
                            if flit and hasattr(flit, "packet_id"):
                                proxy = FlitProxy(
                                    packet_id=str(flit.packet_id),
                                    flit_id=str(getattr(flit, "flit_id", "F0")),
                                    etag_priority=getattr(flit, "etag_priority", "T2"),
                                    itag_h=getattr(flit, "itag_h", False),
                                    itag_v=getattr(flit, "itag_v", False),
                                )
                                flits.append(proxy)
                        node_data["inject_queues"][direction] = flits

            # 提取提取队列数据
            if hasattr(node, "ip_eject_channel_buffers"):
                for channel, buffers in node.ip_eject_channel_buffers.items():
                    if isinstance(buffers, dict) and node.node_id in buffers:
                        fifo = buffers[node.node_id]
                        if hasattr(fifo, "entries"):
                            flits = []
                            for flit in fifo.entries:
                                if flit and hasattr(flit, "packet_id"):
                                    proxy = FlitProxy(packet_id=str(flit.packet_id), flit_id=str(getattr(flit, "flit_id", "F0")))
                                    flits.append(proxy)
                            node_data["eject_queues"][channel] = flits

            # 提取CrossPoint数据
            if hasattr(node, "horizontal_crosspoint"):
                cp_data = CrossPointData(cp_id="horizontal", direction="horizontal", arbitration_state=getattr(node.horizontal_crosspoint, "state", "idle"))
                node_data["crosspoints"]["horizontal"] = cp_data

            if hasattr(node, "vertical_crosspoint"):
                cp_data = CrossPointData(cp_id="vertical", direction="vertical", arbitration_state=getattr(node.vertical_crosspoint, "state", "idle"))
                node_data["crosspoints"]["vertical"] = cp_data

        except Exception as e:
            self.logger.warning(f"提取节点{getattr(node, 'node_id', '?')}数据失败: {e}")

        return node_data

    def _extract_link_data(self, link) -> dict:
        """从链路提取可视化数据"""
        link_data = {"slots": {"req": [], "rsp": [], "data": []}, "stats": LinkStats()}

        try:
            # 提取slot数据
            if hasattr(link, "ring_slices") and link.ring_slices:
                for i, slice_obj in enumerate(link.ring_slices):
                    if hasattr(slice_obj, "slots"):
                        for channel in ["req", "rsp", "data"]:
                            slots = getattr(slice_obj.slots, channel, [])
                            for j, slot in enumerate(slots):
                                if hasattr(slot, "valid") and slot.valid and hasattr(slot, "flit"):
                                    flit = slot.flit
                                    slot_data = SlotData(
                                        slot_id=j,
                                        cycle=getattr(slot, "cycle", 0),
                                        state=SlotState.OCCUPIED,
                                        flit_id=str(getattr(flit, "flit_id", "F0")),
                                        packet_id=str(getattr(flit, "packet_id", "P0")),
                                        priority=getattr(flit, "etag_priority", "T2"),
                                        valid=True,
                                        itag=getattr(flit, "itag_h", False) or getattr(flit, "itag_v", False),
                                        etag=getattr(flit, "etag_priority", "T2") != "T2",
                                    )
                                    link_data["slots"][channel].append(slot_data)

            # 提取统计数据
            if hasattr(link, "get_stats"):
                stats = link.get_stats()
                link_data["stats"] = LinkStats(
                    bandwidth_utilization=stats.get("bandwidth_utilization", 0.0),
                    average_latency=stats.get("average_latency", 0.0),
                    congestion_level=stats.get("congestion_level", 0.0),
                    total_flits=stats.get("total_flits", 0),
                )

        except Exception as e:
            self.logger.warning(f"提取链路数据失败: {e}")

        return link_data

    def _update_node_visualizers(self, nodes_data: dict):
        """更新节点可视化器"""
        for node_id, visualizer in self.node_visualizers.items():
            if node_id in nodes_data:
                try:
                    visualizer.update_node_state(nodes_data[node_id])
                    visualizer.sync_highlight(self.use_highlight, self.highlight_packet_id)
                except Exception as e:
                    self.logger.warning(f"更新节点{node_id}可视化失败: {e}")

    def _update_link_visualizers(self, links_data: dict):
        """更新链路可视化器"""
        for link_id, visualizer in self.link_visualizers.items():
            if link_id in links_data or "main_ring" in links_data:
                try:
                    # 使用第一个可用的链路数据
                    data_key = link_id if link_id in links_data else list(links_data.keys())[0] if links_data else None
                    if data_key:
                        link_data = links_data[data_key]
                        visualizer.update_slots(link_data.get("slots", {}))
                        visualizer.update_statistics(link_data.get("stats", LinkStats()))
                except Exception as e:
                    self.logger.warning(f"更新链路{link_id}可视化失败: {e}")

    def _update_performance_monitor(self, stats: dict):
        """更新性能监控图表"""
        try:
            # 添加性能数据点
            current_bandwidth = stats.get("bandwidth_utilization", 0.0)
            current_latency = stats.get("average_latency", 0.0) / 100.0  # 归一化
            current_congestion = stats.get("congestion_level", 0.0)

            self.performance_data["bandwidth"].append(current_bandwidth)
            self.performance_data["latency"].append(current_latency)
            self.performance_data["congestion"].append(current_congestion)

            # 限制历史数据长度
            max_points = 100
            for key in self.performance_data:
                if len(self.performance_data[key]) > max_points:
                    self.performance_data[key] = self.performance_data[key][-max_points:]

            # 更新图表
            x_data = list(range(len(self.performance_data["bandwidth"])))

            self.perf_lines["bandwidth"].set_data(x_data, self.performance_data["bandwidth"])
            self.perf_lines["latency"].set_data(x_data, self.performance_data["latency"])
            self.perf_lines["congestion"].set_data(x_data, self.performance_data["congestion"])

            # 更新坐标轴范围
            if x_data:
                self.axes["performance"].set_xlim(max(0, x_data[-1] - 50), x_data[-1] + 5)

        except Exception as e:
            self.logger.warning(f"更新性能监控失败: {e}")

    def _update_controls(self):
        """更新控制面板显示"""
        self.cycle_text.set_text(f"周期: {self.current_cycle}")
        self.status_text.set_text(f"状态: {self.playback_state.value}")

    def _add_snapshot(self, snapshot: NetworkSnapshot):
        """添加状态快照"""
        self.snapshots.append(snapshot)
        if len(self.snapshots) > self.max_snapshots:
            self.snapshots.pop(0)
        self.current_snapshot = snapshot

    # 控制回调函数
    def _on_play(self, event):
        """播放按钮回调"""
        self.playback_state = PlaybackState.PLAYING
        self.logger.info("开始播放")

    def _on_pause(self, event):
        """暂停按钮回调"""
        self.playback_state = PlaybackState.PAUSED
        self.logger.info("暂停播放")

    def _on_step(self, event):
        """单步按钮回调"""
        if self.model:
            try:
                self.model.step()
                self.current_cycle += 1
                self._update_from_model()
                self._update_controls()
                self.logger.info(f"单步执行到周期 {self.current_cycle}")
            except Exception as e:
                self.logger.error(f"单步执行失败: {e}")

    def _on_reset(self, event):
        """重置按钮回调"""
        self.playback_state = PlaybackState.STOPPED
        self.current_cycle = 0
        self.performance_data = {"bandwidth": [], "latency": [], "congestion": []}
        self.snapshots.clear()
        self.logger.info("重置可视化器")

    def _on_speed_change(self, val):
        """速度滑块回调"""
        self.playback_speed = val
        # 更新动画间隔
        if self.animation:
            self.animation.event_source.interval = int(self.update_interval * 1000 / self.playback_speed)

    def _on_highlight_toggle(self, label):
        """高亮切换回调"""
        if label == "包追踪":
            self.use_highlight = not self.use_highlight
            if not self.use_highlight:
                self.highlight_packet_id = None
            self.logger.info(f"包追踪: {'开启' if self.use_highlight else '关闭'}")

    def _on_highlight_callback(self, packet_id: str, flit_id: str):
        """高亮回调函数"""
        if self.use_highlight:
            self.highlight_packet_id = packet_id
            self.logger.info(f"高亮包: {packet_id}, Flit: {flit_id}")

            # 同步所有可视化器的高亮状态
            for visualizer in self.node_visualizers.values():
                visualizer.sync_highlight(self.use_highlight, self.highlight_packet_id)

    # 公共接口
    def set_model(self, model):
        """设置要可视化的模型"""
        self.model = model
        self.logger.info(f"设置模型: {type(model).__name__}")

    def start_visualization(self):
        """启动可视化"""
        self.playback_state = PlaybackState.PLAYING
        plt.show()

    def save_current_snapshot(self, filename: str):
        """保存当前状态快照"""
        if self.fig:
            self.fig.savefig(filename, dpi=300, bbox_inches="tight")
            self.logger.info(f"快照已保存: {filename}")

    def export_performance_data(self, filename: str):
        """导出性能数据"""
        try:
            import json

            data = {"cycle_range": [0, len(self.performance_data["bandwidth"])], "performance_data": self.performance_data, "snapshots_count": len(self.snapshots)}
            with open(filename, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
            self.logger.info(f"性能数据已导出: {filename}")
        except Exception as e:
            self.logger.error(f"导出性能数据失败: {e}")


# 演示函数
def create_demo_visualizer():
    """创建演示可视化器"""
    from types import SimpleNamespace

    # 创建演示配置
    config = SimpleNamespace(
        num_nodes=4,
        num_row=2,
        num_col=2,
        IQ_OUT_FIFO_DEPTH=8,
        EQ_IN_FIFO_DEPTH=8,
        RB_IN_FIFO_DEPTH=4,
        RB_OUT_FIFO_DEPTH=4,
        IQ_CH_FIFO_DEPTH=4,
        EQ_CH_FIFO_DEPTH=4,
        CH_NAME_LIST=["gdma", "ddr", "l2m"],
    )

    # 创建可视化器
    visualizer = RealtimeVisualizer(config)

    return visualizer


if __name__ == "__main__":
    # 演示用法
    demo_vis = create_demo_visualizer()
    demo_vis.start_visualization()
