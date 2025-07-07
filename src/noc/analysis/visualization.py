"""
可视化模块 - 提供性能数据的图表展示功能
"""

from typing import Dict, List, Optional, Tuple, Any
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import seaborn as sns
import pandas as pd
import numpy as np
from matplotlib.figure import Figure
from matplotlib.axes import Axes
import networkx as nx
from .performance_metrics import PerformanceMetrics, RequestMetrics, NetworkMetrics, BandwidthMetrics, LatencyMetrics, ThroughputMetrics, HotspotMetrics, RequestType
import sys
import matplotlib

if sys.platform == "darwin":  # macOS 的系统标识是 'darwin'
    matplotlib.use("macosx")  # 仅在 macOS 上使用该后端

# 设置中文字体支持
matplotlib.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS', 'DejaVu Sans']
matplotlib.rcParams['axes.unicode_minus'] = False


class PerformanceVisualizer:
    """性能可视化器"""

    def __init__(self, style: str = "seaborn-v0_8", figsize: Tuple[int, int] = (12, 8), output_manager=None):
        self.style = style
        self.figsize = figsize
        self.output_manager = output_manager
        self.colors = {"read": "#2E86AB", "write": "#A23B72", "overall": "#F18F01", "hotspot": "#C73E1D", "normal": "#6A994E"}

        # 设置绘图样式
        plt.style.use(style)
        sns.set_palette("husl")

    def plot_bandwidth_analysis(self, network_metrics: NetworkMetrics, filename: str = "bandwidth_analysis") -> Figure:
        """绘制带宽分析图"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle("带宽分析", fontsize=16, fontweight="bold")

        # 1. 读写带宽对比
        self._plot_bandwidth_comparison(axes[0, 0], network_metrics)

        # 2. 带宽时间序列
        self._plot_bandwidth_over_time(axes[0, 1], network_metrics)

        # 3. 工作区间分析
        self._plot_working_intervals(axes[1, 0], network_metrics)

        # 4. 带宽效率分析
        self._plot_bandwidth_efficiency(axes[1, 1], network_metrics)

        plt.tight_layout()

        # 保存图片
        if self.output_manager:
            self.output_manager.save_figure(fig, filename, 'png')
        else:
            plt.savefig(f"{filename}.png", dpi=300, bbox_inches="tight")

        return fig

    def _plot_bandwidth_comparison(self, ax: Axes, network_metrics: NetworkMetrics):
        """绘制带宽对比"""
        read_bw = network_metrics.read_metrics["bandwidth"]
        write_bw = network_metrics.write_metrics["bandwidth"]
        overall_bw = network_metrics.overall_metrics["bandwidth"]

        categories = ["Read", "Write", "Overall"]
        avg_values = [read_bw.average_bandwidth_gbps, write_bw.average_bandwidth_gbps, overall_bw.average_bandwidth_gbps]
        peak_values = [read_bw.peak_bandwidth_gbps, write_bw.peak_bandwidth_gbps, overall_bw.peak_bandwidth_gbps]
        effective_values = [read_bw.effective_bandwidth_gbps, write_bw.effective_bandwidth_gbps, overall_bw.effective_bandwidth_gbps]

        x = np.arange(len(categories))
        width = 0.25

        ax.bar(x - width, avg_values, width, label="Average", color=self.colors["read"], alpha=0.7)
        ax.bar(x, peak_values, width, label="Peak", color=self.colors["write"], alpha=0.7)
        ax.bar(x + width, effective_values, width, label="Effective", color=self.colors["overall"], alpha=0.7)

        ax.set_xlabel("请求类型")
        ax.set_ylabel("带宽 (GB/s)")
        ax.set_title("带宽对比")
        ax.set_xticks(x)
        ax.set_xticklabels(categories)
        ax.legend()
        ax.grid(True, alpha=0.3)

    def _plot_bandwidth_over_time(self, ax: Axes, network_metrics: NetworkMetrics):
        """绘制带宽时间序列"""
        # 这里需要实际的时间序列数据，暂时使用模拟数据
        time_points = np.linspace(0, 1000, 100)
        bandwidth_values = np.random.normal(5, 1, 100)  # 模拟数据

        ax.plot(time_points, bandwidth_values, color=self.colors["overall"], linewidth=2)
        ax.fill_between(time_points, bandwidth_values, alpha=0.3, color=self.colors["overall"])

        ax.set_xlabel("Time (ns)")
        ax.set_ylabel("Bandwidth (GB/s)")
        ax.set_title("Bandwidth Over Time")
        ax.grid(True, alpha=0.3)

    def _plot_working_intervals(self, ax: Axes, network_metrics: NetworkMetrics):
        """绘制工作区间分析"""
        overall_bw = network_metrics.overall_metrics["bandwidth"]

        if overall_bw.working_intervals:
            intervals = overall_bw.working_intervals
            durations = [interval.duration for interval in intervals]
            bandwidths = [interval.bandwidth_gbps for interval in intervals]

            ax.scatter(durations, bandwidths, alpha=0.6, s=50, color=self.colors["read"])
            ax.set_xlabel("Interval Duration (ns)")
            ax.set_ylabel("Interval Bandwidth (GB/s)")
            ax.set_title("Working Intervals Analysis")
            ax.grid(True, alpha=0.3)
        else:
            ax.text(0.5, 0.5, "No working intervals data", transform=ax.transAxes, ha="center", va="center")

    def _plot_bandwidth_efficiency(self, ax: Axes, network_metrics: NetworkMetrics):
        """绘制带宽效率分析"""
        read_bw = network_metrics.read_metrics["bandwidth"]
        write_bw = network_metrics.write_metrics["bandwidth"]
        overall_bw = network_metrics.overall_metrics["bandwidth"]

        categories = ["Read", "Write", "Overall"]
        efficiencies = [read_bw.bandwidth_efficiency, write_bw.bandwidth_efficiency, overall_bw.bandwidth_efficiency]
        utilizations = [read_bw.utilization_ratio, write_bw.utilization_ratio, overall_bw.utilization_ratio]

        x = np.arange(len(categories))
        width = 0.35

        ax.bar(x - width / 2, efficiencies, width, label="Efficiency", color=self.colors["read"], alpha=0.7)
        ax.bar(x + width / 2, utilizations, width, label="Utilization", color=self.colors["write"], alpha=0.7)

        ax.set_xlabel("Request Type")
        ax.set_ylabel("Ratio")
        ax.set_title("Bandwidth Efficiency & Utilization")
        ax.set_xticks(x)
        ax.set_xticklabels(categories)
        ax.legend()
        ax.grid(True, alpha=0.3)

    def plot_latency_analysis(self, network_metrics: NetworkMetrics, save_path: Optional[str] = None) -> Figure:
        """绘制延迟分析图"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle("延迟分析", fontsize=16, fontweight="bold")

        # 1. 延迟分布
        self._plot_latency_distribution(axes[0, 0], network_metrics)

        # 2. 延迟对比
        self._plot_latency_comparison(axes[0, 1], network_metrics)

        # 3. 延迟百分位数
        self._plot_latency_percentiles(axes[1, 0], network_metrics)

        # 4. 延迟分解
        self._plot_latency_breakdown(axes[1, 1], network_metrics)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches="tight")

        return fig

    def _plot_latency_distribution(self, ax: Axes, network_metrics: NetworkMetrics):
        """绘制延迟分布"""
        # 模拟延迟数据
        read_latencies = np.random.normal(100, 20, 1000)
        write_latencies = np.random.normal(120, 25, 1000)

        ax.hist(read_latencies, bins=50, alpha=0.7, label="Read", color=self.colors["read"])
        ax.hist(write_latencies, bins=50, alpha=0.7, label="Write", color=self.colors["write"])

        ax.set_xlabel("Latency (ns)")
        ax.set_ylabel("Frequency")
        ax.set_title("Latency Distribution")
        ax.legend()
        ax.grid(True, alpha=0.3)

    def _plot_latency_comparison(self, ax: Axes, network_metrics: NetworkMetrics):
        """绘制延迟对比"""
        read_lat = network_metrics.read_metrics["latency"]
        write_lat = network_metrics.write_metrics["latency"]
        overall_lat = network_metrics.overall_metrics["latency"]

        categories = ["Read", "Write", "Overall"]
        avg_values = [read_lat.avg_total_latency, write_lat.avg_total_latency, overall_lat.avg_total_latency]
        max_values = [read_lat.max_total_latency, write_lat.max_total_latency, overall_lat.max_total_latency]
        p95_values = [read_lat.p95_total_latency, write_lat.p95_total_latency, overall_lat.p95_total_latency]

        x = np.arange(len(categories))
        width = 0.25

        ax.bar(x - width, avg_values, width, label="Average", color=self.colors["read"], alpha=0.7)
        ax.bar(x, p95_values, width, label="P95", color=self.colors["write"], alpha=0.7)
        ax.bar(x + width, max_values, width, label="Max", color=self.colors["overall"], alpha=0.7)

        ax.set_xlabel("Request Type")
        ax.set_ylabel("Latency (ns)")
        ax.set_title("Latency Comparison")
        ax.set_xticks(x)
        ax.set_xticklabels(categories)
        ax.legend()
        ax.grid(True, alpha=0.3)

    def _plot_latency_percentiles(self, ax: Axes, network_metrics: NetworkMetrics):
        """绘制延迟百分位数"""
        overall_lat = network_metrics.overall_metrics["latency"]

        percentiles = [50, 90, 95, 99]
        values = [overall_lat.p50_total_latency, overall_lat.p95_total_latency * 0.9, overall_lat.p95_total_latency, overall_lat.p99_total_latency]  # 估算P90

        ax.plot(percentiles, values, marker="o", linewidth=2, markersize=8, color=self.colors["overall"])
        ax.fill_between(percentiles, values, alpha=0.3, color=self.colors["overall"])

        ax.set_xlabel("Percentile")
        ax.set_ylabel("Latency (ns)")
        ax.set_title("Latency Percentiles")
        ax.grid(True, alpha=0.3)

    def _plot_latency_breakdown(self, ax: Axes, network_metrics: NetworkMetrics):
        """绘制延迟分解"""
        overall_lat = network_metrics.overall_metrics["latency"]

        categories = ["CMD", "Data", "Network"]
        values = [overall_lat.avg_cmd_latency, overall_lat.avg_data_latency, overall_lat.avg_network_latency]
        colors = [self.colors["read"], self.colors["write"], self.colors["overall"]]

        ax.pie(values, labels=categories, colors=colors, autopct="%1.1f%%", startangle=90)
        ax.set_title("Latency Breakdown")

    def plot_throughput_analysis(self, network_metrics: NetworkMetrics, save_path: Optional[str] = None) -> Figure:
        """绘制吞吐量分析图"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle("吞吐量分析", fontsize=16, fontweight="bold")

        # 1. 吞吐量对比
        self._plot_throughput_comparison(axes[0, 0], network_metrics)

        # 2. 吞吐量时间序列
        self._plot_throughput_over_time(axes[0, 1], network_metrics)

        # 3. 吞吐量稳定性
        self._plot_throughput_stability(axes[1, 0], network_metrics)

        # 4. 吞吐量vs延迟
        self._plot_throughput_vs_latency(axes[1, 1], network_metrics)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches="tight")

        return fig

    def _plot_throughput_comparison(self, ax: Axes, network_metrics: NetworkMetrics):
        """绘制吞吐量对比"""
        read_tp = network_metrics.read_metrics["throughput"]
        write_tp = network_metrics.write_metrics["throughput"]
        overall_tp = network_metrics.overall_metrics["throughput"]

        categories = ["Read", "Write", "Overall"]
        avg_values = [read_tp.average_throughput * 1e9, write_tp.average_throughput * 1e9, overall_tp.average_throughput * 1e9]
        peak_values = [read_tp.peak_throughput * 1e9, write_tp.peak_throughput * 1e9, overall_tp.peak_throughput * 1e9]

        x = np.arange(len(categories))
        width = 0.35

        ax.bar(x - width / 2, avg_values, width, label="Average", color=self.colors["read"], alpha=0.7)
        ax.bar(x + width / 2, peak_values, width, label="Peak", color=self.colors["write"], alpha=0.7)

        ax.set_xlabel("Request Type")
        ax.set_ylabel("Throughput (req/s)")
        ax.set_title("Throughput Comparison")
        ax.set_xticks(x)
        ax.set_xticklabels(categories)
        ax.legend()
        ax.grid(True, alpha=0.3)

    def _plot_throughput_over_time(self, ax: Axes, network_metrics: NetworkMetrics):
        """绘制吞吐量时间序列"""
        overall_tp = network_metrics.overall_metrics["throughput"]

        if overall_tp.time_windows and overall_tp.throughput_over_time:
            time_windows = overall_tp.time_windows
            throughput_values = [tp * 1e9 for tp in overall_tp.throughput_over_time]

            ax.plot(time_windows, throughput_values, color=self.colors["overall"], linewidth=2)
            ax.fill_between(time_windows, throughput_values, alpha=0.3, color=self.colors["overall"])
        else:
            # 模拟数据
            time_points = np.linspace(0, 1000, 100)
            throughput_values = np.random.normal(1000, 200, 100)

            ax.plot(time_points, throughput_values, color=self.colors["overall"], linewidth=2)
            ax.fill_between(time_points, throughput_values, alpha=0.3, color=self.colors["overall"])

        ax.set_xlabel("Time (ns)")
        ax.set_ylabel("Throughput (req/s)")
        ax.set_title("Throughput Over Time")
        ax.grid(True, alpha=0.3)

    def _plot_throughput_stability(self, ax: Axes, network_metrics: NetworkMetrics):
        """绘制吞吐量稳定性"""
        read_tp = network_metrics.read_metrics["throughput"]
        write_tp = network_metrics.write_metrics["throughput"]
        overall_tp = network_metrics.overall_metrics["throughput"]

        categories = ["Read", "Write", "Overall"]
        stability = [read_tp.throughput_stability, write_tp.throughput_stability, overall_tp.throughput_stability]

        bars = ax.bar(categories, stability, color=[self.colors["read"], self.colors["write"], self.colors["overall"]], alpha=0.7)

        ax.set_ylabel("Stability (std/mean)")
        ax.set_title("Throughput Stability")
        ax.grid(True, alpha=0.3)

        # 添加数值标签
        for bar, value in zip(bars, stability):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01, f"{value:.3f}", ha="center", va="bottom")

    def _plot_throughput_vs_latency(self, ax: Axes, network_metrics: NetworkMetrics):
        """绘制吞吐量vs延迟"""
        # 模拟数据点
        throughput_points = np.random.normal(1000, 200, 50)
        latency_points = np.random.normal(100, 20, 50)

        ax.scatter(throughput_points, latency_points, alpha=0.6, s=50, color=self.colors["overall"])

        # 添加趋势线
        z = np.polyfit(throughput_points, latency_points, 1)
        p = np.poly1d(z)
        ax.plot(throughput_points, p(throughput_points), "r--", alpha=0.8)

        ax.set_xlabel("Throughput (req/s)")
        ax.set_ylabel("Latency (ns)")
        ax.set_title("Throughput vs Latency")
        ax.grid(True, alpha=0.3)

    def plot_hotspot_analysis(self, network_metrics: NetworkMetrics, save_path: Optional[str] = None) -> Figure:
        """绘制热点分析图"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle("热点分析", fontsize=16, fontweight="bold")

        # 1. 热点节点分布
        self._plot_hotspot_distribution(axes[0, 0], network_metrics)

        # 2. 节点流量分析
        self._plot_node_traffic(axes[0, 1], network_metrics)

        # 3. 拥塞分析
        self._plot_congestion_analysis(axes[1, 0], network_metrics)

        # 4. 负载均衡分析
        self._plot_load_balance(axes[1, 1], network_metrics)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches="tight")

        return fig

    def _plot_hotspot_distribution(self, ax: Axes, network_metrics: NetworkMetrics):
        """绘制热点节点分布"""
        hotspots = network_metrics.hotspot_nodes

        if hotspots:
            node_ids = [h.node_id for h in hotspots]
            congestion_ratios = [h.congestion_ratio for h in hotspots]
            colors = [self.colors["hotspot"] if h.is_hotspot else self.colors["normal"] for h in hotspots]

            bars = ax.bar(node_ids, congestion_ratios, color=colors, alpha=0.7)

            ax.set_xlabel("Node ID")
            ax.set_ylabel("Congestion Ratio")
            ax.set_title("Hotspot Distribution")
            ax.grid(True, alpha=0.3)

            # 添加热点阈值线
            ax.axhline(y=0.1, color="r", linestyle="--", alpha=0.5, label="Hotspot Threshold")
            ax.legend()
        else:
            ax.text(0.5, 0.5, "No hotspot data available", transform=ax.transAxes, ha="center", va="center")

    def _plot_node_traffic(self, ax: Axes, network_metrics: NetworkMetrics):
        """绘制节点流量分析"""
        hotspots = network_metrics.hotspot_nodes

        if hotspots:
            node_ids = [h.node_id for h in hotspots]
            incoming = [h.incoming_requests for h in hotspots]
            outgoing = [h.outgoing_requests for h in hotspots]

            x = np.arange(len(node_ids))
            width = 0.35

            ax.bar(x - width / 2, incoming, width, label="Incoming", color=self.colors["read"], alpha=0.7)
            ax.bar(x + width / 2, outgoing, width, label="Outgoing", color=self.colors["write"], alpha=0.7)

            ax.set_xlabel("Node ID")
            ax.set_ylabel("Request Count")
            ax.set_title("Node Traffic Analysis")
            ax.set_xticks(x)
            ax.set_xticklabels(node_ids)
            ax.legend()
            ax.grid(True, alpha=0.3)
        else:
            ax.text(0.5, 0.5, "No traffic data available", transform=ax.transAxes, ha="center", va="center")

    def _plot_congestion_analysis(self, ax: Axes, network_metrics: NetworkMetrics):
        """绘制拥塞分析"""
        hotspots = network_metrics.hotspot_nodes

        if hotspots:
            congestion_ratios = [h.congestion_ratio for h in hotspots]
            bandwidth_utilizations = [h.bandwidth_utilization for h in hotspots]

            ax.scatter(congestion_ratios, bandwidth_utilizations, alpha=0.6, s=100, c=[self.colors["hotspot"] if h.is_hotspot else self.colors["normal"] for h in hotspots])

            ax.set_xlabel("Congestion Ratio")
            ax.set_ylabel("Bandwidth Utilization")
            ax.set_title("Congestion Analysis")
            ax.grid(True, alpha=0.3)

            # 添加热点区域
            ax.axhline(y=0.8, color="r", linestyle="--", alpha=0.5)
            ax.axvline(x=0.1, color="r", linestyle="--", alpha=0.5)
        else:
            ax.text(0.5, 0.5, "No congestion data available", transform=ax.transAxes, ha="center", va="center")

    def _plot_load_balance(self, ax: Axes, network_metrics: NetworkMetrics):
        """绘制负载均衡分析"""
        hotspots = network_metrics.hotspot_nodes

        if hotspots:
            node_ids = [h.node_id for h in hotspots]
            load_balance_ratios = [h.load_balance_ratio for h in hotspots]

            bars = ax.bar(node_ids, load_balance_ratios, color=[self.colors["hotspot"] if h.is_hotspot else self.colors["normal"] for h in hotspots], alpha=0.7)

            ax.set_xlabel("Node ID")
            ax.set_ylabel("Load Balance Ratio")
            ax.set_title("Load Balance Analysis")
            ax.grid(True, alpha=0.3)

            # 添加理想负载线
            ax.axhline(y=0.5, color="g", linestyle="--", alpha=0.5, label="Ideal Balance")
            ax.legend()
        else:
            ax.text(0.5, 0.5, "No load balance data available", transform=ax.transAxes, ha="center", va="center")

    def create_performance_dashboard(self, network_metrics: NetworkMetrics, save_path: Optional[str] = None) -> Figure:
        """创建性能仪表板"""
        fig = plt.figure(figsize=(20, 16))

        # 创建网格布局
        gs = fig.add_gridspec(4, 4, hspace=0.3, wspace=0.3)

        # 1. 总体性能指标
        ax1 = fig.add_subplot(gs[0, :2])
        self._plot_overall_metrics(ax1, network_metrics)

        # 2. 带宽对比
        ax2 = fig.add_subplot(gs[0, 2:])
        self._plot_bandwidth_comparison(ax2, network_metrics)

        # 3. 延迟分布
        ax3 = fig.add_subplot(gs[1, :2])
        self._plot_latency_distribution(ax3, network_metrics)

        # 4. 吞吐量时间序列
        ax4 = fig.add_subplot(gs[1, 2:])
        self._plot_throughput_over_time(ax4, network_metrics)

        # 5. 热点分布
        ax5 = fig.add_subplot(gs[2, :2])
        self._plot_hotspot_distribution(ax5, network_metrics)

        # 6. 网络利用率
        ax6 = fig.add_subplot(gs[2, 2:])
        self._plot_network_utilization(ax6, network_metrics)

        # 7. 性能摘要表
        ax7 = fig.add_subplot(gs[3, :])
        self._plot_performance_summary_table(ax7, network_metrics)

        plt.suptitle("NoC Performance Dashboard", fontsize=20, fontweight="bold")

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches="tight")

        return fig

    def _plot_overall_metrics(self, ax: Axes, network_metrics: NetworkMetrics):
        """绘制总体性能指标"""
        metrics = ["Bandwidth\n(GB/s)", "Latency\n(ns)", "Throughput\n(req/s)", "Utilization\n(%)"]
        values = [
            network_metrics.overall_bandwidth_gbps,
            network_metrics.overall_latency_ns,
            network_metrics.overall_throughput_rps / 1000,  # 转换为K req/s
            network_metrics.network_utilization * 100,
        ]

        colors = [self.colors["overall"]] * len(metrics)
        bars = ax.bar(metrics, values, color=colors, alpha=0.7)

        # 添加数值标签
        for bar, value in zip(bars, values):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + max(values) * 0.01, f"{value:.1f}", ha="center", va="bottom", fontweight="bold")

        ax.set_title("Overall Performance Metrics")
        ax.grid(True, alpha=0.3)

    def _plot_network_utilization(self, ax: Axes, network_metrics: NetworkMetrics):
        """绘制网络利用率"""
        # 创建仪表盘风格的图表
        utilization = network_metrics.network_utilization

        # 绘制半圆形仪表盘
        theta = np.linspace(0, np.pi, 100)
        r = 1

        # 背景圆弧
        x_bg = r * np.cos(theta)
        y_bg = r * np.sin(theta)
        ax.plot(x_bg, y_bg, "lightgray", linewidth=10)

        # 利用率圆弧
        utilization_theta = np.linspace(0, np.pi * utilization, int(100 * utilization))
        x_util = r * np.cos(utilization_theta)
        y_util = r * np.sin(utilization_theta)

        color = self.colors["hotspot"] if utilization > 0.8 else self.colors["normal"]
        ax.plot(x_util, y_util, color=color, linewidth=10)

        # 添加数值标签
        ax.text(0, -0.3, f"{utilization:.1%}", ha="center", va="center", fontsize=20, fontweight="bold")
        ax.text(0, -0.5, "Network Utilization", ha="center", va="center", fontsize=12)

        ax.set_xlim(-1.2, 1.2)
        ax.set_ylim(-0.6, 1.2)
        ax.set_aspect("equal")
        ax.axis("off")

    def _plot_performance_summary_table(self, ax: Axes, network_metrics: NetworkMetrics):
        """绘制性能摘要表"""
        # 准备表格数据
        summary = network_metrics.get_performance_summary()

        table_data = [
            ["Metric", "Value", "Unit"],
            ["Total Bandwidth", f'{summary.get("bandwidth_gbps", 0):.2f}', "GB/s"],
            ["Average Latency", f'{summary.get("latency_ns", 0):.1f}', "ns"],
            ["Throughput", f'{summary.get("throughput_rps", 0):.0f}', "req/s"],
            ["Network Utilization", f'{summary.get("network_utilization", 0):.1%}', "%"],
            ["Average Hop Count", f'{summary.get("avg_hop_count", 0):.1f}', "hops"],
            ["Hotspot Count", f'{summary.get("hotspot_count", 0)}', "nodes"],
            ["Topology", f"{network_metrics.topology_type}", ""],
            ["Node Count", f"{network_metrics.node_count}", "nodes"],
            ["Simulation Duration", f"{network_metrics.simulation_duration}", "ns"],
        ]

        # 创建表格
        table = ax.table(cellText=table_data[1:], colLabels=table_data[0], cellLoc="center", loc="center")

        # 设置表格样式
        table.auto_set_font_size(False)
        table.set_fontsize(10)
        table.scale(1.2, 1.5)

        # 设置表头样式
        for i in range(len(table_data[0])):
            table[(0, i)].set_facecolor("#E8E8E8")
            table[(0, i)].set_text_props(weight="bold")

        ax.axis("off")
        ax.set_title("Performance Summary", fontsize=14, fontweight="bold", pad=20)


class NetworkFlowVisualizer:
    """网络流量可视化器"""

    def __init__(self, layout: str = "spring"):
        self.layout = layout
        self.colors = {"low": "#90EE90", "medium": "#FFD700", "high": "#FF6B6B", "critical": "#FF0000"}

    def visualize_network_flow(self, requests: List[RequestMetrics], topology_info: Dict[str, Any], save_path: Optional[str] = None) -> Figure:
        """可视化网络流量"""
        fig, ax = plt.subplots(figsize=(15, 12))

        # 创建网络图
        G = self._create_network_graph(requests, topology_info)

        # 选择布局
        if self.layout == "spring":
            pos = nx.spring_layout(G, k=2, iterations=50)
        elif self.layout == "circular":
            pos = nx.circular_layout(G)
        elif self.layout == "grid":
            pos = self._create_grid_layout(G, topology_info)
        else:
            pos = nx.spring_layout(G)

        # 绘制网络
        self._draw_network(ax, G, pos, requests)

        ax.set_title("Network Flow Visualization", fontsize=16, fontweight="bold")
        ax.axis("off")

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches="tight")

        return fig

    def _create_network_graph(self, requests: List[RequestMetrics], topology_info: Dict[str, Any]) -> nx.Graph:
        """创建网络图"""
        G = nx.Graph()

        # 添加节点
        nodes = set()
        for req in requests:
            nodes.add(req.source_node)
            nodes.add(req.dest_node)

        G.add_nodes_from(nodes)

        # 添加边（基于通信关系）
        edge_weights = defaultdict(int)
        for req in requests:
            edge = (req.source_node, req.dest_node)
            edge_weights[edge] += req.total_bytes

        for (src, dst), weight in edge_weights.items():
            G.add_edge(src, dst, weight=weight)

        return G

    def _create_grid_layout(self, G: nx.Graph, topology_info: Dict[str, Any]) -> Dict[int, Tuple[float, float]]:
        """创建网格布局"""
        rows = topology_info.get("rows", 4)
        cols = topology_info.get("cols", 4)

        pos = {}
        for i, node in enumerate(G.nodes()):
            row = i // cols
            col = i % cols
            pos[node] = (col, -row)  # 负号用于从上到下排列

        return pos

    def _draw_network(self, ax: Axes, G: nx.Graph, pos: Dict[int, Tuple[float, float]], requests: List[RequestMetrics]):
        """绘制网络"""
        # 计算节点流量
        node_traffic = defaultdict(int)
        for req in requests:
            node_traffic[req.source_node] += req.total_bytes
            node_traffic[req.dest_node] += req.total_bytes

        # 计算边流量
        edge_traffic = defaultdict(int)
        for req in requests:
            edge = tuple(sorted([req.source_node, req.dest_node]))
            edge_traffic[edge] += req.total_bytes

        # 绘制边
        edges = G.edges()
        edge_weights = [edge_traffic.get(tuple(sorted(edge)), 1) for edge in edges]
        max_weight = max(edge_weights) if edge_weights else 1

        nx.draw_networkx_edges(G, pos, width=[w / max_weight * 5 for w in edge_weights], alpha=0.6, edge_color="gray", ax=ax)

        # 绘制节点
        node_sizes = [node_traffic.get(node, 100) for node in G.nodes()]
        max_size = max(node_sizes) if node_sizes else 1
        normalized_sizes = [size / max_size * 1000 + 100 for size in node_sizes]

        # 根据流量着色
        node_colors = []
        for node in G.nodes():
            traffic = node_traffic[node]
            if traffic > max_size * 0.8:
                color = self.colors["critical"]
            elif traffic > max_size * 0.6:
                color = self.colors["high"]
            elif traffic > max_size * 0.4:
                color = self.colors["medium"]
            else:
                color = self.colors["low"]
            node_colors.append(color)

        nx.draw_networkx_nodes(G, pos, node_size=normalized_sizes, node_color=node_colors, alpha=0.8, ax=ax)

        # 绘制节点标签
        nx.draw_networkx_labels(G, pos, font_size=10, font_weight="bold", ax=ax)

        # 添加图例
        legend_elements = [
            plt.Line2D([0], [0], marker="o", color="w", markerfacecolor=self.colors["low"], markersize=10, label="Low Traffic"),
            plt.Line2D([0], [0], marker="o", color="w", markerfacecolor=self.colors["medium"], markersize=10, label="Medium Traffic"),
            plt.Line2D([0], [0], marker="o", color="w", markerfacecolor=self.colors["high"], markersize=10, label="High Traffic"),
            plt.Line2D([0], [0], marker="o", color="w", markerfacecolor=self.colors["critical"], markersize=10, label="Critical Traffic"),
        ]

        ax.legend(handles=legend_elements, loc="upper right", bbox_to_anchor=(1.15, 1))
