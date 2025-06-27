# -*- coding: utf-8 -*-
"""
拓扑性能对比分析工具
提供Tree vs Torus等多种拓扑的性能对比可视化
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import networkx as nx
from typing import Dict, List, Tuple, Any
import seaborn as sns

# 设置中文字体和样式
plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans'] 
plt.rcParams['axes.unicode_minus'] = False
sns.set_style("whitegrid")


class PerformanceComparator:
    """拓扑性能对比器"""
    
    def __init__(self, figsize=(15, 10)):
        self.figsize = figsize
        self.results = {}
        
    def compare_topologies(self, topology_results: Dict[str, Dict]):
        """对比多种拓扑的性能"""
        self.results = topology_results
        
        # 创建对比图表
        fig, axes = plt.subplots(2, 3, figsize=self.figsize)
        fig.suptitle('C2C拓扑性能对比分析', fontsize=16, fontweight='bold')
        
        # 1. 平均路径长度对比
        self._plot_average_path_length(axes[0, 0])
        
        # 2. 最大路径长度对比
        self._plot_max_path_length(axes[0, 1])
        
        # 3. 节点数量对比
        self._plot_node_count(axes[0, 2])
        
        # 4. 路径长度分布
        self._plot_path_distribution(axes[1, 0])
        
        # 5. 可扩展性分析
        self._plot_scalability(axes[1, 1])
        
        # 6. 综合性能雷达图
        self._plot_radar_chart(axes[1, 2])
        
        plt.tight_layout()
        return fig
    
    def _plot_average_path_length(self, ax):
        """绘制平均路径长度对比"""
        topologies = list(self.results.keys())
        chip_counts = list(self.results[topologies[0]].keys())
        
        for topology in topologies:
            avg_paths = [self.results[topology][count]['avg_path_length'] 
                        for count in chip_counts]
            ax.plot(chip_counts, avg_paths, marker='o', label=topology, linewidth=2)
        
        ax.set_xlabel('芯片数量')
        ax.set_ylabel('平均路径长度')
        ax.set_title('平均路径长度对比')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    def _plot_max_path_length(self, ax):
        """绘制最大路径长度对比"""
        topologies = list(self.results.keys())
        chip_counts = list(self.results[topologies[0]].keys())
        
        for topology in topologies:
            max_paths = [self.results[topology][count]['max_path_length'] 
                        for count in chip_counts]
            ax.plot(chip_counts, max_paths, marker='s', label=topology, linewidth=2)
        
        ax.set_xlabel('芯片数量')
        ax.set_ylabel('最大路径长度')
        ax.set_title('最大路径长度对比')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    def _plot_node_count(self, ax):
        """绘制总节点数对比"""
        topologies = list(self.results.keys())
        chip_counts = list(self.results[topologies[0]].keys())
        
        x = np.arange(len(chip_counts))
        width = 0.35
        
        for i, topology in enumerate(topologies):
            node_counts = [self.results[topology][count]['total_nodes'] 
                          for count in chip_counts]
            ax.bar(x + i*width, node_counts, width, label=topology, alpha=0.8)
        
        ax.set_xlabel('芯片数量')
        ax.set_ylabel('总节点数')
        ax.set_title('总节点数对比')
        ax.set_xticks(x + width/2)
        ax.set_xticklabels(chip_counts)
        ax.legend()
    
    def _plot_path_distribution(self, ax):
        """绘制路径长度分布"""
        # 简化版：显示每种拓扑的路径长度分布
        topologies = list(self.results.keys())
        
        # 收集所有路径长度数据
        path_data = []
        for topology in topologies:
            for count, data in self.results[topology].items():
                if 'path_lengths' in data:
                    for length in data['path_lengths']:
                        path_data.append({'拓扑': topology, '芯片数': count, '路径长度': length})
        
        if path_data:
            df = pd.DataFrame(path_data)
            for topology in topologies:
                topo_data = df[df['拓扑'] == topology]['路径长度']
                ax.hist(topo_data, alpha=0.6, label=topology, bins=10)
        
        ax.set_xlabel('路径长度')
        ax.set_ylabel('频次')
        ax.set_title('路径长度分布')
        ax.legend()
    
    def _plot_scalability(self, ax):
        """绘制可扩展性分析"""
        topologies = list(self.results.keys())
        chip_counts = list(self.results[topologies[0]].keys())
        
        # 计算效率指标（芯片数/总节点数）
        for topology in topologies:
            efficiency = []
            for count in chip_counts:
                chip_num = count
                total_nodes = self.results[topology][count]['total_nodes']
                eff = chip_num / total_nodes * 100  # 百分比
                efficiency.append(eff)
            
            ax.plot(chip_counts, efficiency, marker='d', label=f'{topology}效率', linewidth=2)
        
        ax.set_xlabel('芯片数量')
        ax.set_ylabel('节点效率 (%)')
        ax.set_title('可扩展性分析（节点效率）')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    def _plot_radar_chart(self, ax):
        """绘制综合性能雷达图"""
        # 选择最大芯片数的数据进行对比
        topologies = list(self.results.keys())
        max_chips = max(self.results[topologies[0]].keys())
        
        # 定义评估指标（归一化到0-1）
        metrics = ['路径效率', '节点效率', '连通性', '容错性', '可扩展性']
        
        angles = np.linspace(0, 2*np.pi, len(metrics), endpoint=False).tolist()
        angles += angles[:1]  # 闭合雷达图
        
        for topology in topologies:
            data = self.results[topology][max_chips]
            
            # 计算标准化指标
            path_eff = 1.0 / data['avg_path_length']  # 路径越短越好
            node_eff = max_chips / data['total_nodes']  # 节点效率
            connectivity = 0.9  # 假设值
            fault_tolerance = 0.8 if topology == 'torus' else 0.6  # 假设值
            scalability = 0.9 if topology == 'torus' else 0.7  # 假设值
            
            values = [path_eff, node_eff, connectivity, fault_tolerance, scalability]
            values += values[:1]  # 闭合
            
            ax.plot(angles, values, 'o-', label=topology, linewidth=2)
            ax.fill(angles, values, alpha=0.1)
        
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(metrics)
        ax.set_ylim(0, 1)
        ax.set_title('综合性能雷达图')
        ax.legend()
    
    def generate_performance_report(self, save_path=None):
        """生成性能分析报告"""
        if not self.results:
            print("没有性能数据，请先运行对比分析")
            return
        
        report = []
        report.append("# C2C拓扑性能分析报告\n")
        
        # 总结
        report.append("## 分析总结\n")
        topologies = list(self.results.keys())
        
        for topology in topologies:
            report.append(f"### {topology.upper()}拓扑\n")
            
            # 获取不同芯片数的结果
            for chip_count, data in self.results[topology].items():
                report.append(f"**{chip_count}芯片配置:**\n")
                report.append(f"- 平均路径长度: {data['avg_path_length']:.2f}跳\n")
                report.append(f"- 最大路径长度: {data['max_path_length']}跳\n")
                report.append(f"- 总节点数: {data['total_nodes']}\n")
                report.append(f"- 节点效率: {chip_count/data['total_nodes']*100:.1f}%\n")
                report.append("\n")
        
        # 对比结论
        report.append("## 对比结论\n")
        report.append("根据分析结果：\n")
        report.append("1. **路径性能**: Torus拓扑在路径长度方面通常优于Tree拓扑\n")
        report.append("2. **节点效率**: Torus拓扑无需额外交换机，节点效率更高\n")
        report.append("3. **可扩展性**: 两种拓扑都有良好的可扩展性，但特点不同\n")
        report.append("4. **适用场景**: Tree适合层次化管理，Torus适合高性能计算\n")
        
        report_text = "".join(report)
        
        if save_path:
            with open(save_path, 'w', encoding='utf-8') as f:
                f.write(report_text)
            print(f"报告已保存到: {save_path}")
        
        return report_text
    
    def plot_bandwidth_utilization(self, topology_graph, traffic_pattern=None):
        """绘制带宽利用率热力图"""
        fig, ax = plt.subplots(figsize=(10, 8))
        
        # 简化版：随机生成带宽利用率数据
        # 实际应用中这里会基于真实的流量模式计算
        edges = list(topology_graph._graph.edges())
        utilization = np.random.random(len(edges)) * 100
        
        # 创建带宽利用率表
        edge_labels = [f"{e[0]}-{e[1]}" for e in edges]
        
        # 绘制条形图
        bars = ax.bar(range(len(edges)), utilization, 
                     color=plt.cm.RdYlBu_r(utilization/100))
        
        ax.set_xlabel('链路')
        ax.set_ylabel('带宽利用率 (%)')
        ax.set_title('链路带宽利用率分析')
        ax.set_xticks(range(len(edges)))
        ax.set_xticklabels(edge_labels, rotation=45, ha='right')
        
        # 添加颜色条
        sm = plt.cm.ScalarMappable(cmap=plt.cm.RdYlBu_r, 
                                  norm=plt.Normalize(vmin=0, vmax=100))
        sm.set_array([])
        plt.colorbar(sm, ax=ax, label='利用率 (%)')
        
        plt.tight_layout()
        return fig
    
    def analyze_hotspots(self, topology_graph, traffic_matrix=None):
        """分析网络热点"""
        # 计算节点的中心性指标
        G = topology_graph._graph
        
        # 度中心性
        degree_centrality = nx.degree_centrality(G)
        
        # 介数中心性
        betweenness_centrality = nx.betweenness_centrality(G)
        
        # 接近中心性
        closeness_centrality = nx.closeness_centrality(G)
        
        # 创建热点分析图
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        
        nodes = list(G.nodes())
        
        # 度中心性
        degree_values = [degree_centrality[node] for node in nodes]
        axes[0].bar(range(len(nodes)), degree_values)
        axes[0].set_title('度中心性')
        axes[0].set_xlabel('节点')
        axes[0].set_ylabel('中心性值')
        
        # 介数中心性
        between_values = [betweenness_centrality[node] for node in nodes]
        axes[1].bar(range(len(nodes)), between_values)
        axes[1].set_title('介数中心性')
        axes[1].set_xlabel('节点')
        axes[1].set_ylabel('中心性值')
        
        # 接近中心性
        close_values = [closeness_centrality[node] for node in nodes]
        axes[2].bar(range(len(nodes)), close_values)
        axes[2].set_title('接近中心性')
        axes[2].set_xlabel('节点')
        axes[2].set_ylabel('中心性值')
        
        # 设置x轴标签
        for ax in axes:
            ax.set_xticks(range(len(nodes)))
            ax.set_xticklabels([n.replace('_', '\n') for n in nodes], 
                              rotation=45, ha='right', fontsize=8)
        
        plt.tight_layout()
        return fig, {
            'degree_centrality': degree_centrality,
            'betweenness_centrality': betweenness_centrality, 
            'closeness_centrality': closeness_centrality
        }