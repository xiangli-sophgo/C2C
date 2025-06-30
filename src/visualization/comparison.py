# -*- coding: utf-8 -*-
"""
拓扑性能对比分析工具
提供Tree vs Torus等多种拓扑的全面性能对比可视化
包含路径分析、带宽效率、延迟分析、成本评估等多维度对比
"""

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd
import networkx as nx
from typing import Dict, List, Tuple, Any, Optional
import seaborn as sns
import math
from dataclasses import dataclass
from collections import defaultdict

# 设置中文字体和样式
plt.rcParams["font.sans-serif"] = ["SimHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False
sns.set_style("whitegrid")


@dataclass
class TopologyMetrics:
    """拓扑性能指标数据类"""

    topology_type: str
    chip_count: int

    # 基础拓扑指标
    total_nodes: int
    total_links: int
    avg_path_length: float
    max_path_length: int
    path_length_std: float

    # 高级性能指标
    bandwidth_efficiency: float  # 理论带宽效率
    latency_factor: float  # 延迟因子
    cost_factor: float  # 成本因子(交换机数量等)
    fault_tolerance: float  # 故障容错指标
    scalability_score: float  # 可扩展性评分

    # 分布统计
    path_length_distribution: Dict[int, int]  # 路径长度分布
    node_degree_distribution: Dict[int, int]  # 节点度分布


class PerformanceComparator:
    """增强的拓扑性能对比器"""

    def __init__(self, figsize=(16, 12)):
        self.figsize = figsize
        self.metrics_data: Dict[str, Dict[int, TopologyMetrics]] = {}
        self.comparison_cache: Dict[str, Any] = {}

        # 性能权重配置
        self.performance_weights = {"path_efficiency": 0.25, "bandwidth_efficiency": 0.20, "cost_efficiency": 0.20, "fault_tolerance": 0.15, "scalability": 0.20}

    def add_topology_data(self, topology_type: str, chip_count: int, topology_structure: Dict, routing_logic=None):
        """添加拓扑数据进行分析"""
        if topology_type not in self.metrics_data:
            self.metrics_data[topology_type] = {}

        # 计算全面的性能指标
        metrics = self._calculate_comprehensive_metrics(topology_type, chip_count, topology_structure, routing_logic)

        self.metrics_data[topology_type][chip_count] = metrics

    def _calculate_comprehensive_metrics(self, topology_type: str, chip_count: int, topology_structure: Dict, routing_logic=None) -> TopologyMetrics:
        """计算全面的拓扑性能指标"""

        if topology_type == "tree":
            return self._calculate_tree_metrics(chip_count, topology_structure)
        elif topology_type == "torus":
            return self._calculate_torus_metrics(chip_count, topology_structure, routing_logic)
        else:
            raise ValueError(f"不支持的拓扑类型: {topology_type}")

    def _calculate_tree_metrics(self, chip_count: int, tree_structure: Dict) -> TopologyMetrics:
        """计算树拓扑的详细指标"""
        tree_root = tree_structure["root"]
        all_nodes = tree_structure["nodes"]

        # 基础指标
        total_nodes = len(all_nodes)
        total_links = sum(len(node.children) for node in all_nodes.values() if hasattr(node, "children"))

        # 计算所有芯片间的路径长度
        chip_nodes = [nid for nid in all_nodes.keys() if nid.startswith("chip_")]
        path_lengths = []
        path_dist = defaultdict(int)

        for i, src in enumerate(chip_nodes):
            for dst in chip_nodes[i + 1 :]:
                path_len = self._calculate_tree_path_length(src, dst, all_nodes)
                path_lengths.append(path_len)
                path_dist[path_len] += 1

        avg_path_length = np.mean(path_lengths) if path_lengths else 0
        max_path_length = max(path_lengths) if path_lengths else 0
        path_length_std = np.std(path_lengths) if path_lengths else 0

        # 高级指标计算
        bandwidth_efficiency = chip_count / total_nodes  # 芯片节点占比
        latency_factor = avg_path_length  # 延迟与平均跳数成正比
        cost_factor = (total_nodes - chip_count) / chip_count  # 每芯片需要的交换机数
        fault_tolerance = self._calculate_tree_fault_tolerance(tree_structure)
        scalability_score = self._calculate_tree_scalability(chip_count, tree_structure)

        # 节点度分布
        degree_dist = defaultdict(int)
        for node in all_nodes.values():
            degree = len(getattr(node, "children", [])) + (1 if hasattr(node, "parent") and node.parent else 0)
            degree_dist[degree] += 1

        return TopologyMetrics(
            topology_type="tree",
            chip_count=chip_count,
            total_nodes=total_nodes,
            total_links=total_links,
            avg_path_length=avg_path_length,
            max_path_length=max_path_length,
            path_length_std=path_length_std,
            bandwidth_efficiency=bandwidth_efficiency,
            latency_factor=latency_factor,
            cost_factor=cost_factor,
            fault_tolerance=fault_tolerance,
            scalability_score=scalability_score,
            path_length_distribution=dict(path_dist),
            node_degree_distribution=dict(degree_dist),
        )

    def _calculate_torus_metrics(self, chip_count: int, torus_structure: Dict, routing_logic) -> TopologyMetrics:
        """计算环面拓扑的详细指标"""
        grid_dims = torus_structure["grid_dimensions"]
        coord_map = torus_structure["coordinate_map"]

        # 基础指标
        total_nodes = chip_count  # Torus中每个芯片都是节点
        dimensions = len(grid_dims)
        total_links = chip_count * dimensions  # 每个节点连接到各维度的邻居

        # 计算所有芯片间的路径长度
        path_lengths = []
        path_dist = defaultdict(int)

        if routing_logic:
            for i in range(min(chip_count, 50)):  # 限制计算量
                for j in range(i + 1, min(chip_count, 50)):
                    src_coord = coord_map[i]
                    dst_coord = coord_map[j]
                    distances = routing_logic.calculate_shortest_distance(src_coord, dst_coord, grid_dims)
                    path_len = distances["total_hops"]
                    path_lengths.append(path_len)
                    path_dist[path_len] += 1

        avg_path_length = np.mean(path_lengths) if path_lengths else self._estimate_torus_avg_path(grid_dims)
        max_path_length = max(path_lengths) if path_lengths else self._estimate_torus_max_path(grid_dims)
        path_length_std = np.std(path_lengths) if path_lengths else 0

        # 高级指标计算
        bandwidth_efficiency = 1.0  # Torus中所有节点都是芯片
        latency_factor = avg_path_length
        cost_factor = 0.0  # Torus无需额外交换机
        fault_tolerance = self._calculate_torus_fault_tolerance(torus_structure)
        scalability_score = self._calculate_torus_scalability(chip_count, torus_structure)

        # 节点度分布 (Torus中所有节点度数相同)
        degree_dist = {dimensions * 2: chip_count}  # 每个节点连接到各维度的两个邻居

        return TopologyMetrics(
            topology_type="torus",
            chip_count=chip_count,
            total_nodes=total_nodes,
            total_links=total_links,
            avg_path_length=avg_path_length,
            max_path_length=max_path_length,
            path_length_std=path_length_std,
            bandwidth_efficiency=bandwidth_efficiency,
            latency_factor=latency_factor,
            cost_factor=cost_factor,
            fault_tolerance=fault_tolerance,
            scalability_score=scalability_score,
            path_length_distribution=dict(path_dist),
            node_degree_distribution=degree_dist,
        )

    def compare_topologies_comprehensive(self) -> plt.Figure:
        """生成全面的拓扑对比分析图表"""
        if not self.metrics_data:
            raise ValueError("没有可用的拓扑数据，请先添加数据")

        # 创建大型对比图表
        fig = plt.figure(figsize=self.figsize)
        gs = fig.add_gridspec(3, 4, hspace=0.3, wspace=0.3)

        fig.suptitle("C2C拓扑全面性能对比分析", fontsize=18, fontweight="bold", y=0.95)

        # 1. 路径性能对比 (第一行左)
        ax1 = fig.add_subplot(gs[0, 0])
        self._plot_path_performance(ax1)

        # 2. 带宽效率对比 (第一行中左)
        ax2 = fig.add_subplot(gs[0, 1])
        self._plot_bandwidth_efficiency(ax2)

        # 3. 成本效率对比 (第一行中右)
        ax3 = fig.add_subplot(gs[0, 2])
        self._plot_cost_efficiency(ax3)

        # 4. 故障容错对比 (第一行右)
        ax4 = fig.add_subplot(gs[0, 3])
        self._plot_fault_tolerance(ax4)

        # 5. 综合性能热力图 (第二行左两列)
        ax5 = fig.add_subplot(gs[1, :2])
        self._plot_performance_heatmap(ax5)

        # 6. 可扩展性趋势 (第二行右两列)
        ax6 = fig.add_subplot(gs[1, 2:])
        self._plot_scalability_trends(ax6)

        # 7. 路径长度分布对比 (第三行左两列)
        ax7 = fig.add_subplot(gs[2, :2])
        self._plot_path_distribution_comparison(ax7)

        # 8. 综合评分雷达图 (第三行右两列)
        ax8 = fig.add_subplot(gs[2, 2:], projection="polar")
        self._plot_comprehensive_radar(ax8)

        return fig

    def _plot_path_performance(self, ax):
        """绘制路径性能对比"""
        topologies = list(self.metrics_data.keys())
        if not topologies:
            return

        chip_counts = sorted(list(self.metrics_data[topologies[0]].keys()))

        for topology in topologies:
            avg_paths = [self.metrics_data[topology][count].avg_path_length for count in chip_counts if count in self.metrics_data[topology]]
            max_paths = [self.metrics_data[topology][count].max_path_length for count in chip_counts if count in self.metrics_data[topology]]

            valid_counts = [count for count in chip_counts if count in self.metrics_data[topology]]

            ax.plot(valid_counts, avg_paths, marker="o", label=f"{topology}-平均", linewidth=2)
            ax.plot(valid_counts, max_paths, marker="s", linestyle="--", label=f"{topology}-最大", alpha=0.7)

        ax.set_xlabel("芯片数量")
        ax.set_ylabel("路径长度(跳数)")
        ax.set_title("路径性能对比")
        ax.legend(bbox_to_anchor=(1.05, 1), loc="upper left")
        ax.grid(True, alpha=0.3)

    def _plot_bandwidth_efficiency(self, ax):
        """绘制带宽效率对比"""
        topologies = list(self.metrics_data.keys())
        if not topologies:
            return

        chip_counts = sorted(list(self.metrics_data[topologies[0]].keys()))

        for topology in topologies:
            efficiencies = [self.metrics_data[topology][count].bandwidth_efficiency * 100 for count in chip_counts if count in self.metrics_data[topology]]
            valid_counts = [count for count in chip_counts if count in self.metrics_data[topology]]

            ax.plot(valid_counts, efficiencies, marker="d", label=topology, linewidth=2)

        ax.set_xlabel("芯片数量")
        ax.set_ylabel("带宽效率 (%)")
        ax.set_title("带宽效率对比")
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.set_ylim(0, 110)

    def _plot_cost_efficiency(self, ax):
        """绘制成本效率对比"""
        topologies = list(self.metrics_data.keys())
        if not topologies:
            return

        chip_counts = sorted(list(self.metrics_data[topologies[0]].keys()))

        x = np.arange(len(chip_counts))
        width = 0.35

        for i, topology in enumerate(topologies):
            cost_factors = [self.metrics_data[topology][count].cost_factor for count in chip_counts if count in self.metrics_data[topology]]
            valid_counts = [count for count in chip_counts if count in self.metrics_data[topology]]
            valid_x = x[: len(valid_counts)]

            ax.bar(valid_x + i * width, cost_factors, width, label=topology, alpha=0.8)

        ax.set_xlabel("芯片数量")
        ax.set_ylabel("成本因子(交换机/芯片)")
        ax.set_title("成本效率对比")
        ax.set_xticks(x + width / 2)
        ax.set_xticklabels(chip_counts)
        ax.legend()

    def _plot_fault_tolerance(self, ax):
        """绘制故障容错能力对比"""
        topologies = list(self.metrics_data.keys())
        if not topologies:
            return

        chip_counts = sorted(list(self.metrics_data[topologies[0]].keys()))

        for topology in topologies:
            fault_scores = [self.metrics_data[topology][count].fault_tolerance for count in chip_counts if count in self.metrics_data[topology]]
            valid_counts = [count for count in chip_counts if count in self.metrics_data[topology]]

            ax.plot(valid_counts, fault_scores, marker="^", label=topology, linewidth=2)

        ax.set_xlabel("芯片数量")
        ax.set_ylabel("故障容错评分")
        ax.set_title("故障容错能力对比")
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.set_ylim(0, 1)

    def _plot_performance_heatmap(self, ax):
        """绘制性能指标热力图"""
        topologies = list(self.metrics_data.keys())
        if not topologies:
            return

        # 收集所有指标数据
        metrics_names = ["路径效率", "带宽效率", "成本效率", "故障容错", "可扩展性"]
        data_matrix = []
        topology_labels = []

        for topology in topologies:
            for chip_count in sorted(self.metrics_data[topology].keys()):
                metrics = self.metrics_data[topology][chip_count]

                # 计算归一化指标
                path_eff = 1.0 / (metrics.avg_path_length + 0.1)  # 路径越短越好
                bandwidth_eff = metrics.bandwidth_efficiency
                cost_eff = 1.0 / (metrics.cost_factor + 0.1)  # 成本越低越好
                fault_tol = metrics.fault_tolerance
                scalability = metrics.scalability_score

                data_matrix.append([path_eff, bandwidth_eff, cost_eff, fault_tol, scalability])
                topology_labels.append(f"{topology}-{chip_count}")

        # 归一化到0-1范围
        data_matrix = np.array(data_matrix)
        for i in range(data_matrix.shape[1]):
            col = data_matrix[:, i]
            if col.max() > col.min():
                data_matrix[:, i] = (col - col.min()) / (col.max() - col.min())

        # 绘制热力图
        im = ax.imshow(data_matrix, cmap="RdYlGn", aspect="auto", vmin=0, vmax=1)

        # 设置坐标轴
        ax.set_xticks(range(len(metrics_names)))
        ax.set_xticklabels(metrics_names, rotation=45, ha="right")
        ax.set_yticks(range(len(topology_labels)))
        ax.set_yticklabels(topology_labels)
        ax.set_title("性能指标热力图")

        # 添加颜色条
        cbar = plt.colorbar(im, ax=ax, shrink=0.8)
        cbar.set_label("性能评分", rotation=270, labelpad=20)

    def _plot_scalability_trends(self, ax):
        """绘制可扩展性趋势分析"""
        topologies = list(self.metrics_data.keys())
        if not topologies:
            return

        for topology in topologies:
            chip_counts = sorted(self.metrics_data[topology].keys())
            scalability_scores = [self.metrics_data[topology][count].scalability_score for count in chip_counts]

            ax.plot(chip_counts, scalability_scores, marker="o", label=f"{topology}", linewidth=2)

        ax.set_xlabel("芯片数量")
        ax.set_ylabel("可扩展性评分")
        ax.set_title("可扩展性趋势分析")
        ax.legend()
        ax.grid(True, alpha=0.3)

    def _plot_path_distribution_comparison(self, ax):
        """绘制路径长度分布对比"""
        topologies = list(self.metrics_data.keys())
        if not topologies:
            return

        # 选择中等规模的数据进行分布对比
        target_chip_counts = []
        for topology in topologies:
            available_counts = list(self.metrics_data[topology].keys())
            if available_counts:
                mid_count = sorted(available_counts)[len(available_counts) // 2]
                target_chip_counts.append(mid_count)

        if not target_chip_counts:
            return

        colors = ["skyblue", "lightcoral", "lightgreen", "gold"]

        for i, topology in enumerate(topologies):
            if i < len(target_chip_counts):
                chip_count = target_chip_counts[i]
                if chip_count in self.metrics_data[topology]:
                    path_dist = self.metrics_data[topology][chip_count].path_length_distribution

                    path_lengths = list(path_dist.keys())
                    frequencies = list(path_dist.values())

                    ax.bar([p + i * 0.3 for p in path_lengths], frequencies, width=0.25, label=f"{topology}({chip_count}芯片)", alpha=0.7, color=colors[i % len(colors)])

        ax.set_xlabel("路径长度(跳数)")
        ax.set_ylabel("路径数量")
        ax.set_title("路径长度分布对比")
        ax.legend()
        ax.grid(True, alpha=0.3)

    def _plot_comprehensive_radar(self, ax):
        """绘制综合评分雷达图"""
        topologies = list(self.metrics_data.keys())
        if not topologies:
            return

        # 选择最大芯片数的数据进行对比
        max_chip_counts = {}
        for topology in topologies:
            max_chip_counts[topology] = max(self.metrics_data[topology].keys())

        # 定义评估指标
        metrics_names = ["路径效率", "带宽效率", "成本效率", "故障容错", "可扩展性"]

        angles = np.linspace(0, 2 * np.pi, len(metrics_names), endpoint=False).tolist()
        angles += angles[:1]  # 闭合雷达图

        colors = ["red", "blue", "green", "orange"]

        for i, topology in enumerate(topologies):
            chip_count = max_chip_counts[topology]
            metrics = self.metrics_data[topology][chip_count]

            # 计算标准化指标
            path_eff = min(1.0, 1.0 / metrics.avg_path_length)  # 路径效率
            bandwidth_eff = metrics.bandwidth_efficiency  # 带宽效率
            cost_eff = min(1.0, 1.0 / (metrics.cost_factor + 0.1))  # 成本效率
            fault_tol = metrics.fault_tolerance  # 故障容错
            scalability = metrics.scalability_score  # 可扩展性

            values = [path_eff, bandwidth_eff, cost_eff, fault_tol, scalability]
            values += values[:1]  # 闭合

            color = colors[i % len(colors)]
            ax.plot(angles, values, "o-", label=f"{topology}({chip_count})", linewidth=2, color=color)
            ax.fill(angles, values, alpha=0.1, color=color)

        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(metrics_names)
        ax.set_ylim(0, 1)
        ax.set_title("综合性能雷达图")
        ax.legend(loc="upper right", bbox_to_anchor=(1.3, 1.0))

    def generate_comprehensive_report(self, save_path: Optional[str] = None) -> str:
        """生成全面的性能分析报告"""
        if not self.metrics_data:
            print("没有性能数据，请先添加拓扑数据")
            return ""

        report = []
        report.append("# C2C拓扑全面性能分析报告\n\n")

        # 执行摘要
        report.append("## 执行摘要\n")
        best_overall = self._find_best_topology_overall()
        report.append(f"**综合最优**: {best_overall['topology']}拓扑在{best_overall['chip_count']}芯片规模下表现最佳\n")
        report.append(f"**综合评分**: {best_overall['score']:.3f}\n\n")

        # 详细分析
        report.append("## 详细性能分析\n")
        topologies = list(self.metrics_data.keys())

        for topology in topologies:
            report.append(f"### {topology.upper()}拓扑\n")

            for chip_count in sorted(self.metrics_data[topology].keys()):
                metrics = self.metrics_data[topology][chip_count]
                report.append(f"**{chip_count}芯片配置:**\n")
                report.append(f"- 平均路径长度: {metrics.avg_path_length:.2f}跳\n")
                report.append(f"- 最大路径长度: {metrics.max_path_length}跳\n")
                report.append(f"- 带宽效率: {metrics.bandwidth_efficiency*100:.1f}%\n")
                report.append(f"- 成本因子: {metrics.cost_factor:.2f}\n")
                report.append(f"- 故障容错: {metrics.fault_tolerance:.2f}\n")
                report.append(f"- 可扩展性: {metrics.scalability_score:.2f}\n")
                report.append("\n")

        # 对比结论
        report.append("## 对比结论与建议\n")
        conclusions = self._generate_conclusions()
        for conclusion in conclusions:
            report.append(f"- {conclusion}\n")

        # 应用建议
        report.append("\n## 应用场景建议\n")
        recommendations = self._generate_recommendations()
        for scenario, recommendation in recommendations.items():
            report.append(f"**{scenario}**: {recommendation}\n")

        report_text = "".join(report)

        if save_path:
            with open(save_path, "w", encoding="utf-8") as f:
                f.write(report_text)
            print(f"报告已保存到: {save_path}")

        return report_text

    # 辅助方法实现
    def _calculate_tree_path_length(self, src: str, dst: str, all_nodes: Dict) -> int:
        """计算树拓扑中两个节点间的路径长度"""
        # 简化实现：使用BFS计算路径长度
        if src == dst:
            return 0

        # 构建邻接关系
        graph = defaultdict(list)
        for node_id, node in all_nodes.items():
            if hasattr(node, "parent") and node.parent:
                graph[node_id].append(node.parent.node_id)
                graph[node.parent.node_id].append(node_id)
            if hasattr(node, "children"):
                for child in node.children:
                    graph[node_id].append(child.node_id)

        # BFS计算最短路径
        from collections import deque

        queue = deque([(src, 0)])
        visited = {src}

        while queue:
            current, dist = queue.popleft()
            if current == dst:
                return dist

            for neighbor in graph[current]:
                if neighbor not in visited:
                    visited.add(neighbor)
                    queue.append((neighbor, dist + 1))

        return float("inf")  # 不可达

    def _estimate_torus_avg_path(self, grid_dims: List[int]) -> float:
        """估算环面拓扑的平均路径长度"""
        # 理论估算：各维度平均距离的和
        avg_path = 0
        for dim in grid_dims:
            avg_path += dim / 4  # 环形拓扑的平均距离约为维度大小的1/4
        return avg_path

    def _estimate_torus_max_path(self, grid_dims: List[int]) -> int:
        """估算环面拓扑的最大路径长度"""
        # 理论最大路径：各维度最大距离的和
        max_path = 0
        for dim in grid_dims:
            max_path += dim // 2  # 环形拓扑的最大距离是维度大小的一半
        return max_path

    def _calculate_tree_fault_tolerance(self, tree_structure: Dict) -> float:
        """计算树拓扑的故障容错能力"""
        # 树拓扑的容错能力较低，单点故障影响较大
        all_nodes = tree_structure["nodes"]
        chip_count = len([n for n in all_nodes.keys() if n.startswith("chip_")])
        switch_count = len([n for n in all_nodes.keys() if n.startswith("switch_")])

        if switch_count == 0:
            return 1.0  # 无交换机的直连情况

        # 容错能力与冗余度相关
        return max(0.1, 1.0 - switch_count / (chip_count + switch_count))

    def _calculate_torus_fault_tolerance(self, torus_structure: Dict) -> float:
        """计算环面拓扑的故障容错能力"""
        # 环面拓扑具有更好的容错能力，多路径冗余
        dimensions = torus_structure["dimensions"]
        # 维度越高，容错能力越强
        return min(0.95, 0.5 + dimensions * 0.15)

    def _calculate_tree_scalability(self, chip_count: int, tree_structure: Dict) -> float:
        """计算树拓扑的可扩展性"""
        all_nodes = tree_structure["nodes"]
        total_nodes = len(all_nodes)

        # 可扩展性与节点效率和结构复杂度相关
        node_efficiency = chip_count / total_nodes
        complexity_penalty = math.log(total_nodes) / math.log(chip_count * 2)  # 复杂度惩罚

        return node_efficiency * (1 - complexity_penalty * 0.3)

    def _calculate_torus_scalability(self, chip_count: int, torus_structure: Dict) -> float:
        """计算环面拓扑的可扩展性"""
        grid_dims = torus_structure["grid_dimensions"]
        dimensions = len(grid_dims)

        # 环面拓扑的可扩展性与网格平衡度相关
        balance_factor = min(grid_dims) / max(grid_dims) if max(grid_dims) > 0 else 1.0
        dimension_bonus = min(0.3, dimensions * 0.1)  # 高维度奖励

        return min(0.95, 0.6 + balance_factor * 0.2 + dimension_bonus)

    def _find_best_topology_overall(self) -> Dict[str, Any]:
        """寻找综合最优的拓扑配置"""
        best_score = -1
        best_config = {"topology": "", "chip_count": 0, "score": 0}

        for topology in self.metrics_data:
            for chip_count in self.metrics_data[topology]:
                metrics = self.metrics_data[topology][chip_count]

                # 计算综合评分
                score = (
                    self.performance_weights["path_efficiency"] * (1.0 / (metrics.avg_path_length + 0.1))
                    + self.performance_weights["bandwidth_efficiency"] * metrics.bandwidth_efficiency
                    + self.performance_weights["cost_efficiency"] * (1.0 / (metrics.cost_factor + 0.1))
                    + self.performance_weights["fault_tolerance"] * metrics.fault_tolerance
                    + self.performance_weights["scalability"] * metrics.scalability_score
                )

                if score > best_score:
                    best_score = score
                    best_config = {"topology": topology, "chip_count": chip_count, "score": score}

        return best_config

    def _generate_conclusions(self) -> List[str]:
        """生成对比结论"""
        conclusions = []

        if "tree" in self.metrics_data and "torus" in self.metrics_data:
            conclusions.extend(
                [
                    "Tree拓扑在小规模系统中具有管理简单的优势，但需要额外的交换机设备",
                    "Torus拓扑在中大规模系统中表现更优，具有更好的路径效率和容错能力",
                    "从成本角度看，Torus拓扑无需额外交换机，硬件成本更低",
                    "从延迟角度看，Torus拓扑的平均路径长度通常更短",
                    "Tree拓扑适合需要集中控制和层次化管理的应用场景",
                    "Torus拓扑适合高性能计算和大规模并行处理场景",
                ]
            )

        return conclusions

    def _generate_recommendations(self) -> Dict[str, str]:
        """生成应用场景建议"""
        recommendations = {
            "小规模系统(4-16芯片)": "Tree拓扑管理简单，适合原型验证和开发测试",
            "中等规模系统(32-64芯片)": "Torus 2D拓扑提供良好的性能和成本平衡",
            "大规模系统(128+芯片)": "Torus 3D拓扑最大化性能，适合高性能计算集群",
            "成本敏感场景": "优先选择Torus拓扑，避免额外交换机成本",
            "高可靠性要求": "Torus拓扑提供更好的容错和多路径冗余",
            "层次化管理需求": "Tree拓扑天然支持分层控制结构",
        }

        return recommendations
