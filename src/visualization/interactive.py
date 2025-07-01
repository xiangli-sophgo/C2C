# -*- coding: utf-8 -*-
"""
增强的C2C拓扑交互式可视化Web应用
集成全面的拓扑对比分析和可视化功能
"""

import streamlit as st
import matplotlib.pyplot as plt
import matplotlib
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import sys
import os
import io
import base64

# 添加父目录到Python路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# 导入项目模块
from src.visualization.comparison import PerformanceComparator
from src.visualization.config import init_visualization_config, get_label, get_color_scheme
from src.topology.topology_optimizer import TopologyOptimizer, ApplicationRequirements
from src.topology.tree import TreeTopologyLogic
from src.topology.torus import TorusTopologyLogic, TorusRoutingLogic
from src.topology.graph import TopologyGraph

# 初始化可视化配置
init_visualization_config()

# 页面配置
st.set_page_config(page_title="C2C拓扑性能对比分析", page_icon="📊", layout="wide", initial_sidebar_state="expanded")

# 自定义CSS样式
st.markdown(
    """
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-box {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #1f77b4;
    }
    .comparison-result {
        background-color: #e8f5e8;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #28a745;
    }
    .warning-box {
        background-color: #fff3cd;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #ffc107;
    }
</style>
""",
    unsafe_allow_html=True,
)


@st.cache_data
def load_topology_data():
    """加载和缓存拓扑数据"""
    tree_logic = TreeTopologyLogic()
    torus_logic = TorusTopologyLogic()
    torus_routing = TorusRoutingLogic()

    return tree_logic, torus_logic, torus_routing


@st.cache_data
def generate_topology_comparison(chip_counts, _tree_logic, _torus_logic, _torus_routing):
    """生成拓扑对比数据"""
    comparator = PerformanceComparator()

    for chip_count in chip_counts:
        # Tree拓扑
        try:
            chip_ids = list(range(chip_count))
            tree_root, tree_nodes = _tree_logic.calculate_tree_structure(chip_ids, switch_capacity=8)
            tree_structure = {"root": tree_root, "nodes": tree_nodes}
            comparator.add_topology_data("tree", chip_count, tree_structure)
        except Exception as e:
            st.warning(f"Tree拓扑({chip_count}芯片)分析失败: {e}")

        # Torus拓扑
        try:
            dimensions = 2 if chip_count <= 64 else 3
            torus_structure = _torus_logic.calculate_torus_structure(chip_count, dimensions=dimensions)
            comparator.add_topology_data("torus", chip_count, torus_structure, _torus_routing)
        except Exception as e:
            st.warning(f"Torus拓扑({chip_count}芯片)分析失败: {e}")

    return comparator


def create_plotly_comparison_charts(comparator):
    """使用Plotly创建交互式对比图表"""
    if not comparator.metrics_data:
        return None

    # 准备数据
    topologies = list(comparator.metrics_data.keys())
    all_chip_counts = set()
    for topology in topologies:
        all_chip_counts.update(comparator.metrics_data[topology].keys())
    chip_counts = sorted(list(all_chip_counts))

    # 创建子图
    fig = make_subplots(
        rows=2,
        cols=3,
        subplot_titles=["路径性能", "带宽效率", "成本分析", "容错能力", "扩展性趋势", "性能雷达图"],
        specs=[[{"type": "scatter"}, {"type": "scatter"}, {"type": "bar"}], [{"type": "scatter"}, {"type": "scatter"}, {"type": "scatterpolar"}]],
    )

    colors = {"tree": "#ff7f0e", "torus": "#1f77b4"}

    for topology in topologies:
        topology_chip_counts = sorted([c for c in chip_counts if c in comparator.metrics_data[topology]])

        if not topology_chip_counts:
            continue

        metrics_list = [comparator.metrics_data[topology][c] for c in topology_chip_counts]

        # 1. 路径性能
        avg_paths = [m.avg_path_length for m in metrics_list]
        max_paths = [m.max_path_length for m in metrics_list]

        fig.add_trace(
            go.Scatter(
                x=topology_chip_counts, y=avg_paths, mode="lines+markers", name=f"{topology}-平均", line=dict(color=colors[topology]), hovertemplate="芯片数: %{x}<br>平均路径: %{y:.2f}<extra></extra>"
            ),
            row=1,
            col=1,
        )

        fig.add_trace(
            go.Scatter(
                x=topology_chip_counts,
                y=max_paths,
                mode="lines+markers",
                name=f"{topology}-最大",
                line=dict(color=colors[topology], dash="dash"),
                hovertemplate="芯片数: %{x}<br>最大路径: %{y}<extra></extra>",
            ),
            row=1,
            col=1,
        )

        # 2. 带宽效率
        bandwidth_effs = [m.bandwidth_efficiency * 100 for m in metrics_list]
        fig.add_trace(
            go.Scatter(
                x=topology_chip_counts,
                y=bandwidth_effs,
                mode="lines+markers",
                name=f"{topology}-带宽",
                line=dict(color=colors[topology]),
                hovertemplate="芯片数: %{x}<br>带宽: %{y:.1f}%<extra></extra>",
            ),
            row=1,
            col=2,
        )

        # 3. 成本分析
        cost_factors = [m.cost_factor for m in metrics_list]
        fig.add_trace(
            go.Bar(
                x=[f"{topology}-{c}" for c in topology_chip_counts],
                y=cost_factors,
                name=f"{topology}-成本",
                marker_color=colors[topology],
                hovertemplate="配置: %{x}<br>成本因子: %{y:.3f}<extra></extra>",
            ),
            row=1,
            col=3,
        )

        # 4. 容错能力
        fault_scores = [m.fault_tolerance for m in metrics_list]
        fig.add_trace(
            go.Scatter(
                x=topology_chip_counts,
                y=fault_scores,
                mode="lines+markers",
                name=f"{topology}-容错",
                line=dict(color=colors[topology]),
                hovertemplate="芯片数: %{x}<br>容错能力: %{y:.3f}<extra></extra>",
            ),
            row=2,
            col=1,
        )

        # 5. 可扩展性
        scalability_scores = [m.scalability_score for m in metrics_list]
        fig.add_trace(
            go.Scatter(
                x=topology_chip_counts,
                y=scalability_scores,
                mode="lines+markers",
                name=f"{topology}-扩展性",
                line=dict(color=colors[topology]),
                hovertemplate="芯片数: %{x}<br>扩展性: %{y:.3f}<extra></extra>",
            ),
            row=2,
            col=2,
        )

    # 6. 雷达图 (使用最大芯片数的数据)
    max_chips = max(chip_counts)
    metrics_names = ["路径效率", "带宽效率", "成本效率", "容错能力", "可扩展性"]

    for topology in topologies:
        if max_chips in comparator.metrics_data[topology]:
            metrics = comparator.metrics_data[topology][max_chips]

            path_eff = min(1.0, 1.0 / (metrics.avg_path_length + 0.1))
            bandwidth_eff = metrics.bandwidth_efficiency
            cost_eff = min(1.0, 1.0 / (metrics.cost_factor + 0.1))
            fault_tol = metrics.fault_tolerance
            scalability = metrics.scalability_score

            values = [path_eff, bandwidth_eff, cost_eff, fault_tol, scalability]

            fig.add_trace(go.Scatterpolar(r=values, theta=metrics_names, fill="toself", name=f"{topology}({max_chips})", line_color=colors[topology]), row=2, col=3)

    # 更新布局
    fig.update_layout(height=800, showlegend=True, title_text="C2C拓扑综合性能对比")

    # 更新各子图的轴标签
    fig.update_xaxes(title_text="芯片数量", row=1, col=1)
    fig.update_yaxes(title_text="路径长度 (跳数)", row=1, col=1)

    fig.update_xaxes(title_text="芯片数量", row=1, col=2)
    fig.update_yaxes(title_text="带宽效率 (%)", row=1, col=2)

    fig.update_xaxes(title_text="配置", row=1, col=3)
    fig.update_yaxes(title_text="成本因子", row=1, col=3)

    fig.update_xaxes(title_text="芯片数量", row=2, col=1)
    fig.update_yaxes(title_text="容错能力", row=2, col=1)

    fig.update_xaxes(title_text="芯片数量", row=2, col=2)
    fig.update_yaxes(title_text="扩展性得分", row=2, col=2)

    return fig


def create_topology_visualization(topology_type, chip_count, tree_logic, torus_logic):
    """创建拓扑结构可视化"""
    fig, ax = plt.subplots(figsize=(12, 9))

    # 获取颜色方案
    colors = get_color_scheme("modern")

    if topology_type == "树状":
        try:
            chip_ids = list(range(chip_count))
            tree_root, tree_nodes = tree_logic.calculate_tree_structure(chip_ids, switch_capacity=8)

            # 构建层次化布局
            levels = {}
            node_positions = {}

            # 计算节点层级
            def assign_levels(node, level=0):
                if node.node_id not in levels:
                    levels[node.node_id] = level
                    if hasattr(node, "children"):
                        for child in node.children:
                            assign_levels(child, level + 1)

            assign_levels(tree_root)

            # 计算节点位置
            max_level = max(levels.values())
            level_nodes = {i: [] for i in range(max_level + 1)}

            for node_id, level in levels.items():
                level_nodes[level].append(node_id)

            for level, nodes in level_nodes.items():
                y = max_level - level
                for i, node_id in enumerate(nodes):
                    x = i - len(nodes) / 2
                    node_positions[node_id] = (x, y)

            # 绘制连接
            for node_id, node in tree_nodes.items():
                if hasattr(node, "children") and node.children:
                    x1, y1 = node_positions[node_id]
                    for child in node.children:
                        x2, y2 = node_positions[child.node_id]
                        ax.plot([x1, x2], [y1, y2], color=colors["default_link"], linewidth=2, alpha=0.7)

            # 绘制节点
            for node_id, (x, y) in node_positions.items():
                if node_id.startswith("chip_"):
                    color = colors["chip"]
                    size = 300
                    marker = "o"
                else:
                    color = colors["switch"]
                    size = 400
                    marker = "s"

                ax.scatter(x, y, s=size, c=color, marker=marker, edgecolors="black", linewidth=2, alpha=0.8, zorder=5)

                # 添加标签
                ax.text(x, y, node_id.replace("_", "\n"), ha="center", va="center", fontsize=8, fontweight="bold", zorder=10)

            # 设置标题
            title = get_label("topology_types", "tree")[0] + f" - {chip_count} " + get_label("ui_elements", "chip_count")[0]
            ax.set_title(title, fontsize=16, fontweight="bold", pad=20)

        except Exception as e:
            error_msg = f"创建树拓扑时出错:\n{str(e)}"
            ax.text(0.5, 0.5, error_msg, ha="center", va="center", transform=ax.transAxes, fontsize=12, bbox=dict(boxstyle="round,pad=0.3", facecolor="lightcoral", alpha=0.7))
            ax.set_title("树拓扑 - 错误", fontsize=14)

    elif topology_type == "环形":
        try:
            dimensions = 2 if chip_count <= 64 else 3
            torus_structure = torus_logic.calculate_torus_structure(chip_count, dimensions=dimensions)

            # 创建2D网格可视化
            grid_dims = torus_structure["grid_dimensions"]
            coord_map = torus_structure["coordinate_map"]

            if len(grid_dims) >= 2:
                rows, cols = grid_dims[0], grid_dims[1]

                # 绘制连接线
                for chip_id, coord in coord_map.items():
                    if len(coord) >= 2:
                        x, y = coord[1], coord[0]

                        # 水平连接
                        next_x = (x + 1) % cols
                        # 查找邻居节点
                        for other_id, other_coord in coord_map.items():
                            if len(other_coord) >= 2 and other_coord[1] == next_x and other_coord[0] == y:
                                # 处理环形连接的视觉效果
                                if abs(x - next_x) == 1:
                                    ax.plot([x, next_x], [y, y], color=colors["c2c"], linewidth=2, alpha=0.6, zorder=1)
                                elif cols > 2:  # 环形连接
                                    # 绘制弯曲的环形连接线
                                    if x == cols - 1 and next_x == 0:
                                        ax.annotate("", xy=(next_x, y), xytext=(x, y), arrowprops=dict(arrowstyle="<->", connectionstyle="arc3,rad=0.3", color=colors["c2c"], lw=2, alpha=0.6))

                        # 垂直连接
                        next_y = (y + 1) % rows
                        for other_id, other_coord in coord_map.items():
                            if len(other_coord) >= 2 and other_coord[0] == next_y and other_coord[1] == x:
                                if abs(y - next_y) == 1:
                                    ax.plot([x, x], [y, next_y], color=colors["c2c"], linewidth=2, alpha=0.6, zorder=1)
                                elif rows > 2:  # 环形连接
                                    if y == rows - 1 and next_y == 0:
                                        ax.annotate("", xy=(x, next_y), xytext=(x, y), arrowprops=dict(arrowstyle="<->", connectionstyle="arc3,rad=0.3", color=colors["c2c"], lw=2, alpha=0.6))

                # 绘制芯片节点
                for chip_id, coord in coord_map.items():
                    if len(coord) >= 2:
                        x, y = coord[1], coord[0]
                        ax.scatter(x, y, s=350, c=colors["chip"], edgecolors="black", linewidth=2, alpha=0.9, zorder=5)
                        ax.text(x, y, str(chip_id), ha="center", va="center", fontweight="bold", fontsize=10, zorder=10)

                ax.set_xlim(-0.8, cols - 0.2)
                ax.set_ylim(-0.8, rows - 0.2)
                ax.set_aspect("equal")

                # 设置标题
                title = get_label("topology_types", "torus")[0] + f" - {chip_count} " + get_label("ui_elements", "chip_count")[0]
                title += f" ({grid_dims[0]}x{grid_dims[1]}"
                if len(grid_dims) > 2:
                    title += f"x{grid_dims[2]})"
                else:
                    title += ")"
                ax.set_title(title, fontsize=16, fontweight="bold", pad=20)

        except Exception as e:
            error_msg = f"创建环面拓扑时出错:\n{str(e)}"
            ax.text(0.5, 0.5, error_msg, ha="center", va="center", transform=ax.transAxes, fontsize=12, bbox=dict(boxstyle="round,pad=0.3", facecolor="lightcoral", alpha=0.7))
            ax.set_title("环面拓扑 - 错误", fontsize=14)

    ax.grid(True, alpha=0.2)
    ax.set_facecolor("#f8f9fa")

    # 添加图例
    if topology_type == "树状":
        chip_patch = plt.Line2D([0], [0], marker="o", color="w", markerfacecolor=colors["chip"], markersize=10, label=get_label("node_types", "chip")[0])
        switch_patch = plt.Line2D([0], [0], marker="s", color="w", markerfacecolor=colors["switch"], markersize=10, label=get_label("node_types", "switch")[0])
        ax.legend(handles=[chip_patch, switch_patch], loc="upper right")
    elif topology_type == "环形":
        chip_patch = plt.Line2D([0], [0], marker="o", color="w", markerfacecolor=colors["chip"], markersize=10, label=get_label("node_types", "chip")[0])
        link_patch = plt.Line2D([0], [0], color=colors["c2c"], linewidth=2, label="C2C链路")
        ax.legend(handles=[chip_patch, link_patch], loc="upper right")

    return fig


def generate_optimization_recommendations(requirements):
    """生成优化建议"""
    optimizer = TopologyOptimizer()
    recommendations = optimizer.analyze_requirements(requirements)

    return recommendations


def main():
    """主应用函数"""

    # 页面标题
    st.markdown('<h1 class="main-header">C2C拓扑性能对比分析</h1>', unsafe_allow_html=True)

    # 侧边栏配置
    st.sidebar.title("配置")

    # 加载数据
    tree_logic, torus_logic, torus_routing = load_topology_data()

    # 分析模式选择
    analysis_mode = st.sidebar.selectbox("分析模式", ["性能对比", "拓扑可视化", "优化顾问"])

    if analysis_mode == "性能对比":
        st.header("📊 综合性能对比")

        # 芯片数量选择
        chip_counts = st.sidebar.multiselect("选择对比的芯片数量", options=[4, 8, 16, 32, 48, 64, 128, 256], default=[8, 16, 32, 64])

        if chip_counts:
            with st.spinner("正在生成拓扑对比..."):
                comparator = generate_topology_comparison(chip_counts, tree_logic, torus_logic, torus_routing)

            if comparator.metrics_data:
                # 创建交互式图表
                fig = create_plotly_comparison_charts(comparator)
                if fig:
                    st.plotly_chart(fig, use_container_width=True)

                # 显示最佳配置
                best_config = comparator._find_best_topology_overall()
                col1, col2, col3 = st.columns(3)

                with col1:
                    st.markdown('<div class="metric-box">', unsafe_allow_html=True)
                    st.metric("最佳拓扑", best_config["topology"].upper())
                    st.markdown("</div>", unsafe_allow_html=True)

                with col2:
                    st.markdown('<div class="metric-box">', unsafe_allow_html=True)
                    st.metric("最佳芯片数", best_config["chip_count"])
                    st.markdown("</div>", unsafe_allow_html=True)

                with col3:
                    st.markdown('<div class="metric-box">', unsafe_allow_html=True)
                    st.metric("综合得分", f"{best_config['score']:.3f}")
                    st.markdown("</div>", unsafe_allow_html=True)

                # 详细分析表格
                st.subheader("📈 详细指标")

                # 创建对比表格
                comparison_data = []
                for topology in comparator.metrics_data:
                    for chip_count in sorted(comparator.metrics_data[topology].keys()):
                        metrics = comparator.metrics_data[topology][chip_count]
                        comparison_data.append(
                            {
                                "拓扑": topology.upper(),
                                "芯片数": chip_count,
                                "平均路径长度": f"{metrics.avg_path_length:.2f}",
                                "最大路径长度": metrics.max_path_length,
                                "带宽效率": f"{metrics.bandwidth_efficiency*100:.1f}%",
                                "成本因子": f"{metrics.cost_factor:.3f}",
                                "容错能力": f"{metrics.fault_tolerance:.3f}",
                                "扩展性": f"{metrics.scalability_score:.3f}",
                            }
                        )

                df = pd.DataFrame(comparison_data)
                st.dataframe(df, use_container_width=True)

                # 生成报告
                if st.button("生成详细报告"):
                    report = comparator.generate_comprehensive_report()
                    st.text_area("分析报告", report, height=400)

                    # 下载报告
                    st.download_button(label="下载报告", data=report, file_name="topology_comparison_report.md", mime="text/markdown")
            else:
                st.warning("没有可用的拓扑数据，请检查配置。")
        else:
            st.info("请选择要对比的芯片数量。")

    elif analysis_mode == "拓扑可视化":
        st.header("🌐 拓扑结构可视化")

        # 拓扑类型选择
        col1, col2 = st.columns(2)

        with col1:
            topology_type = st.selectbox("拓扑类型", ["树状", "环形"])

        with col2:
            chip_count = st.selectbox("芯片数量", [4, 8, 16, 32, 48, 64])

        # 生成可视化
        if st.button("生成可视化"):
            with st.spinner("正在创建拓扑可视化..."):
                fig = create_topology_visualization(topology_type, chip_count, tree_logic, torus_logic)
                st.pyplot(fig)

        # 拓扑信息
        if topology_type and chip_count:
            st.subheader("📋 拓扑信息")

            if topology_type == "树状":
                try:
                    chip_ids = list(range(chip_count))
                    tree_root, tree_nodes = tree_logic.calculate_tree_structure(chip_ids, switch_capacity=8)

                    chip_nodes = len([n for n in tree_nodes.keys() if n.startswith("chip_")])
                    switch_nodes = len([n for n in tree_nodes.keys() if n.startswith("switch_")])

                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("芯片节点", chip_nodes)
                    with col2:
                        st.metric("交换机节点", switch_nodes)
                    with col3:
                        st.metric("总节点", chip_nodes + switch_nodes)

                except Exception as e:
                    st.error(f"分析树状拓扑时出错: {e}")

            elif topology_type == "环形":
                try:
                    dimensions = 2 if chip_count <= 64 else 3
                    torus_structure = torus_logic.calculate_torus_structure(chip_count, dimensions=dimensions)

                    grid_dims = torus_structure["grid_dimensions"]

                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("维度", dimensions)
                    with col2:
                        grid_str = "x".join(map(str, grid_dims))
                        st.metric("网格大小", grid_str)
                    with col3:
                        st.metric("总节点", chip_count)

                except Exception as e:
                    st.error(f"分析环形拓扑时出错: {e}")

    elif analysis_mode == "优化顾问":
        st.header("🎯 拓扑优化顾问")

        st.write("定义您的应用需求，以获取个性化的拓扑建议。")

        # 需求配置
        col1, col2 = st.columns(2)

        with col1:
            chip_count = st.number_input("芯片数量", min_value=4, max_value=512, value=32, step=4)
            budget_constraint = st.slider("预算限制", 0.1, 1.0, 0.8, 0.1)
            latency_req = st.selectbox("延迟要求", ["low", "medium", "high"])
            reliability_req = st.selectbox("可靠性要求", ["low", "medium", "high"])

        with col2:
            scalability_req = st.selectbox("扩展性要求", ["low", "medium", "high"])
            management_complexity = st.selectbox("管理复杂度", ["simple", "moderate", "complex"])
            power_constraint = st.slider("功耗限制", 0.1, 1.0, 0.7, 0.1)

        if st.button("生成建议"):
            requirements = ApplicationRequirements(
                chip_count=chip_count,
                budget_constraint=budget_constraint,
                latency_requirement=latency_req,
                reliability_requirement=reliability_req,
                scalability_requirement=scalability_req,
                management_complexity=management_complexity,
                power_constraint=power_constraint,
            )

            with st.spinner("正在分析需求并生成建议..."):
                recommendations = generate_optimization_recommendations(requirements)

            # 显示推荐结果
            best_topology = max(recommendations.keys(), key=lambda k: recommendations[k].score)
            best_rec = recommendations[best_topology]

            st.markdown(f'<div class="comparison-result">', unsafe_allow_html=True)
            st.subheader(f"🏆 推荐: {best_topology.upper()} 拓扑")
            st.write(f"**适应性得分: {best_rec.score:.3f}**")
            st.markdown("</div>", unsafe_allow_html=True)

            # 配置详情
            col1, col2 = st.columns(2)

            with col1:
                st.subheader("✅ 优点")
                for pro in best_rec.pros:
                    st.write(f"• {pro}")

            with col2:
                st.subheader("⚠️ 局限性")
                for con in best_rec.cons:
                    st.write(f"• {con}")

            # 优化建议
            if best_rec.optimization_tips:
                st.subheader("💡 优化技巧")
                for tip in best_rec.optimization_tips:
                    st.write(f"• {tip}")

            # 性能预测
            st.subheader("📊 性能预测")
            perf_df = pd.DataFrame([{"指标": metric.replace("_", " ").title(), "值": f"{value:.3f}"} for metric, value in best_rec.estimated_performance.items()])
            st.table(perf_df)

            # 对比所有选项
            st.subheader("📋 所有选项对比")
            comparison_df = pd.DataFrame(
                [
                    {
                        "拓扑": topology.upper(),
                        "得分": f"{rec.score:.3f}",
                        "平均路径长度": f"{rec.estimated_performance.get('avg_path_length', 0):.2f}",
                        "成本因子": f"{rec.estimated_performance.get('cost_factor', 0):.3f}",
                        "容错能力": f"{rec.estimated_performance.get('fault_tolerance', 0):.3f}",
                    }
                    for topology, rec in recommendations.items()
                ]
            )
            st.dataframe(comparison_df, use_container_width=True)

    # 页脚
    st.markdown("---")
    st.markdown("**C2C拓扑性能分析工具** - 强大的综合对比功能")


if __name__ == "__main__":
    main()
