"""
简化的可视化模块 - 只生成最重要的图表
专注于提供最有价值的可视化结果
"""

from typing import Dict, List, Optional, Any
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
from matplotlib.figure import Figure
import sys
import matplotlib

if sys.platform == "darwin":
    matplotlib.use("macosx")

# 设置中文字体支持
matplotlib.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS', 'DejaVu Sans']
matplotlib.rcParams['axes.unicode_minus'] = False

from .performance_metrics import RequestMetrics, RequestType
from .simple_output_manager import SimpleOutputManager


class SimplePerformanceVisualizer:
    """简化的性能可视化器 - 只生成核心图表"""
    
    def __init__(self, output_manager: Optional[SimpleOutputManager] = None):
        self.output_manager = output_manager
        self.colors = {
            'primary': '#2E86AB',
            'secondary': '#A23B72', 
            'accent': '#F18F01',
            'success': '#6A994E',
            'warning': '#F2994A',
            'danger': '#E74C3C'
        }
        
        # 设置图表样式
        plt.style.use('seaborn-v0_8-whitegrid')
    
    def create_performance_dashboard(self, performance_data: Dict[str, Any], requests: List[RequestMetrics]) -> Figure:
        """创建综合性能仪表板 - 一个图包含所有关键信息"""
        
        fig = plt.figure(figsize=(16, 10))
        fig.suptitle('NoC 性能分析仪表板', fontsize=20, fontweight='bold')
        
        # 创建网格布局 3x3
        gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)
        
        # 1. 性能指标概览 (左上)
        ax1 = fig.add_subplot(gs[0, 0])
        self._plot_performance_overview(ax1, performance_data)
        
        # 2. 延迟分布 (中上)
        ax2 = fig.add_subplot(gs[0, 1])
        self._plot_latency_distribution(ax2, requests)
        
        # 3. 读写请求对比 (右上)
        ax3 = fig.add_subplot(gs[0, 2])
        self._plot_request_type_comparison(ax3, performance_data)
        
        # 4. 带宽利用率仪表盘 (左中)
        ax4 = fig.add_subplot(gs[1, 0])
        self._plot_utilization_gauge(ax4, performance_data['network_utilization'])
        
        # 5. 热点节点分析 (中中)
        ax5 = fig.add_subplot(gs[1, 1])
        self._plot_hotspot_analysis(ax5, requests, performance_data)
        
        # 6. 性能时间序列 (右中)
        ax6 = fig.add_subplot(gs[1, 2])
        self._plot_performance_timeline(ax6, requests)
        
        # 7. 关键指标汇总表 (底部跨3列)
        ax7 = fig.add_subplot(gs[2, :])
        self._plot_metrics_summary_table(ax7, performance_data)
        
        plt.tight_layout()
        
        # 保存图片
        if self.output_manager:
            self.output_manager.save_figure(fig, "dashboard")
        
        return fig
    
    def _plot_performance_overview(self, ax, data):
        """性能指标概览"""
        metrics = ['带宽\n(GB/s)', '延迟\n(ns)', '吞吐量\n(req/s)', '利用率\n(%)']
        values = [
            data['bandwidth_gbps'],
            data['latency_ns'],
            data['throughput_rps'] / 1000,  # 转换为K req/s
            data['network_utilization'] * 100
        ]
        
        bars = ax.bar(metrics, values, color=[self.colors['primary'], self.colors['secondary'], 
                                            self.colors['accent'], self.colors['success']], alpha=0.8)
        
        # 添加数值标签
        for bar, value in zip(bars, values):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + max(values) * 0.01,
                   f'{value:.1f}', ha='center', va='bottom', fontweight='bold')
        
        ax.set_title('核心性能指标', fontweight='bold')
        ax.grid(True, alpha=0.3)
    
    def _plot_latency_distribution(self, ax, requests):
        """延迟分布图"""
        if not requests:
            ax.text(0.5, 0.5, '无数据', transform=ax.transAxes, ha='center', va='center')
            return
        
        latencies = [r.total_latency for r in requests]
        
        ax.hist(latencies, bins=30, alpha=0.7, color=self.colors['primary'], edgecolor='black')
        ax.axvline(np.mean(latencies), color=self.colors['danger'], linestyle='--', 
                  linewidth=2, label=f'平均值: {np.mean(latencies):.1f}ns')
        ax.axvline(np.percentile(latencies, 95), color=self.colors['warning'], linestyle='--', 
                  linewidth=2, label=f'P95: {np.percentile(latencies, 95):.1f}ns')
        
        ax.set_xlabel('延迟 (ns)')
        ax.set_ylabel('频次')
        ax.set_title('延迟分布')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    def _plot_request_type_comparison(self, ax, data):
        """读写请求对比"""
        labels = ['读请求', '写请求']
        sizes = [data['read_requests'], data['write_requests']]
        colors = [self.colors['primary'], self.colors['secondary']]
        
        wedges, texts, autotexts = ax.pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%', 
                                         startangle=90, textprops={'fontsize': 10})
        
        # 美化文本
        for autotext in autotexts:
            autotext.set_color('white')
            autotext.set_fontweight('bold')
        
        ax.set_title('请求类型分布')
    
    def _plot_utilization_gauge(self, ax, utilization):
        """网络利用率仪表盘"""
        # 绘制半圆形仪表盘
        theta = np.linspace(0, np.pi, 100)
        r = 1
        
        # 背景圆弧
        x_bg = r * np.cos(theta)
        y_bg = r * np.sin(theta)
        ax.plot(x_bg, y_bg, 'lightgray', linewidth=15)
        
        # 利用率圆弧
        utilization_theta = np.linspace(0, np.pi * utilization, int(100 * utilization))
        if len(utilization_theta) > 0:
            x_util = r * np.cos(utilization_theta)
            y_util = r * np.sin(utilization_theta)
            
            # 根据利用率选择颜色
            if utilization > 0.8:
                color = self.colors['danger']
            elif utilization > 0.6:
                color = self.colors['warning']
            else:
                color = self.colors['success']
            
            ax.plot(x_util, y_util, color=color, linewidth=15)
        
        # 添加数值标签
        ax.text(0, -0.3, f'{utilization:.1%}', ha='center', va='center', 
               fontsize=18, fontweight='bold')
        ax.text(0, -0.5, '网络利用率', ha='center', va='center', fontsize=12)
        
        ax.set_xlim(-1.2, 1.2)
        ax.set_ylim(-0.6, 1.2)
        ax.set_aspect('equal')
        ax.axis('off')
    
    def _plot_hotspot_analysis(self, ax, requests, data):
        """热点节点分析"""
        if data['hotspot_count'] == 0:
            ax.text(0.5, 0.5, '无热点节点', transform=ax.transAxes, ha='center', va='center',
                   fontsize=14, color=self.colors['success'])
            ax.set_title('热点分析')
            ax.axis('off')
            return
        
        # 统计节点流量
        from collections import defaultdict
        node_traffic = defaultdict(int)
        for req in requests:
            node_traffic[req.source_node] += req.total_bytes
            node_traffic[req.dest_node] += req.total_bytes
        
        # 显示前5个最高流量节点
        top_nodes = sorted(node_traffic.items(), key=lambda x: x[1], reverse=True)[:5]
        
        if top_nodes:
            nodes, traffic = zip(*top_nodes)
            colors = [self.colors['danger'] if node in data['hotspot_nodes'] 
                     else self.colors['primary'] for node in nodes]
            
            bars = ax.bar(range(len(nodes)), traffic, color=colors, alpha=0.8)
            ax.set_xticks(range(len(nodes)))
            ax.set_xticklabels([f'节点{node}' for node in nodes], rotation=45)
            ax.set_ylabel('流量 (bytes)')
            ax.set_title(f'节点流量 (热点: {data["hotspot_count"]}个)')
        
        ax.grid(True, alpha=0.3)
    
    def _plot_performance_timeline(self, ax, requests):
        """性能时间线"""
        if not requests:
            ax.text(0.5, 0.5, '无数据', transform=ax.transAxes, ha='center', va='center')
            return
        
        # 按时间窗口统计吞吐量
        start_time = min(r.start_time for r in requests)
        end_time = max(r.end_time for r in requests)
        
        window_size = (end_time - start_time) // 20  # 分成20个时间窗口
        if window_size <= 0:
            return
        
        time_windows = []
        throughput_values = []
        
        current_time = start_time
        while current_time < end_time:
            window_end = current_time + window_size
            window_requests = [r for r in requests if current_time <= r.end_time < window_end]
            
            throughput = len(window_requests) / (window_size / 1e9) if window_size > 0 else 0
            time_windows.append(current_time / 1000)  # 转换为μs
            throughput_values.append(throughput)
            
            current_time = window_end
        
        if time_windows and throughput_values:
            ax.plot(time_windows, throughput_values, color=self.colors['accent'], 
                   linewidth=2, marker='o', markersize=4)
            ax.fill_between(time_windows, throughput_values, alpha=0.3, color=self.colors['accent'])
            
            ax.set_xlabel('时间 (μs)')
            ax.set_ylabel('吞吐量 (req/s)')
            ax.set_title('吞吐量时间线')
            ax.grid(True, alpha=0.3)
    
    def _plot_metrics_summary_table(self, ax, data):
        """性能指标汇总表"""
        # 准备表格数据
        table_data = [
            ['指标', '数值', '单位', '状态'],
            ['总带宽', f'{data["bandwidth_gbps"]:.3f}', 'GB/s', self._get_status(data["bandwidth_gbps"], 5.0)],
            ['平均延迟', f'{data["latency_ns"]:.1f}', 'ns', self._get_status(data["latency_ns"], 200.0, reverse=True)],
            ['P95延迟', f'{data["p95_latency_ns"]:.1f}', 'ns', self._get_status(data["p95_latency_ns"], 500.0, reverse=True)],
            ['吞吐量', f'{data["throughput_rps"]:.0f}', 'req/s', self._get_status(data["throughput_rps"], 1000.0)],
            ['网络利用率', f'{data["network_utilization"]:.1%}', '%', self._get_status(data["network_utilization"], 0.8)],
            ['总请求数', f'{data["total_requests"]}', '个', '正常'],
            ['热点节点', f'{data["hotspot_count"]}', '个', '正常' if data["hotspot_count"] == 0 else '注意'],
            ['平均跳数', f'{data["avg_hop_count"]:.1f}', 'hops', self._get_status(data["avg_hop_count"], 3.0, reverse=True)]
        ]
        
        # 创建表格
        table = ax.table(cellText=table_data[1:], colLabels=table_data[0], 
                        cellLoc='center', loc='center', bbox=[0, 0, 1, 1])
        
        # 设置表格样式
        table.auto_set_font_size(False)
        table.set_fontsize(10)
        table.scale(1, 1.8)
        
        # 设置表头样式
        for i in range(len(table_data[0])):
            table[(0, i)].set_facecolor('#E8E8E8')
            table[(0, i)].set_text_props(weight='bold')
        
        # 根据状态设置行颜色
        for i, row in enumerate(table_data[1:], 1):
            status = row[3]
            if status == '优秀':
                color = '#D5EFDF'  # 淡绿色
            elif status == '注意':
                color = '#FFF2CC'  # 淡黄色
            elif status == '警告':
                color = '#FFEBEE'  # 淡红色
            else:
                color = 'white'
            
            for j in range(len(row)):
                table[(i, j)].set_facecolor(color)
        
        ax.axis('off')
        ax.set_title('性能指标汇总', fontsize=14, fontweight='bold', pad=20)
    
    def _get_status(self, value: float, threshold: float, reverse: bool = False) -> str:
        """根据阈值判断状态"""
        if reverse:  # 值越小越好（如延迟）
            if value < threshold * 0.5:
                return '优秀'
            elif value < threshold:
                return '正常'
            elif value < threshold * 1.5:
                return '注意'
            else:
                return '警告'
        else:  # 值越大越好（如带宽）
            if value > threshold * 1.5:
                return '优秀'
            elif value > threshold:
                return '正常'
            elif value > threshold * 0.5:
                return '注意'
            else:
                return '警告'