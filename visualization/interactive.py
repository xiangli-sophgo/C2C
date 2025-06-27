# -*- coding: utf-8 -*-
"""
Streamlit交互式Web界面
提供C2C拓扑的交互式可视化和分析功能
"""

import streamlit as st
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
import numpy as np
from visualization.visualizer import TopologyVisualizer
from visualization.comparison import PerformanceComparator
from topology.tree import TreeTopologyLogic, validate_tree_topology, evaluate_tree_performance
from topology.torus import TorusTopologyLogic, TorusRoutingLogic, test_torus_connectivity
from topology.node import ChipNode, SwitchNode
from topology.link import C2CDirectLink
from topology.builder import TopologyBuilder


def main():
    """主应用函数"""
    st.set_page_config(page_title="C2C拓扑可视化工具", page_icon="🖥️", layout="wide", initial_sidebar_state="expanded")

    st.title("🖥️ C2C拓扑可视化分析工具")
    st.markdown("---")

    # 侧边栏配置
    with st.sidebar:
        st.header("🛠️ 配置面板")

        # 功能选择
        function_mode = st.selectbox("选择功能模式", ["拓扑可视化", "性能对比", "路径分析", "热点分析"])

        st.markdown("---")

        # 拓扑配置
        st.subheader("拓扑配置")
        topology_type = st.selectbox("拓扑类型", ["Tree", "Torus", "Mixed"])

        chip_count = st.slider("芯片数量", 4, 128, 16, 4)

        if topology_type == "Tree":
            switch_capacity = st.slider("交换机端口数", 4, 16, 8)
        elif topology_type == "Torus":
            dimensions = st.selectbox("维度", [2, 3])

        # 可视化选项
        st.subheader("可视化选项")
        show_labels = st.checkbox("显示节点标签", True)
        show_coordinates = st.checkbox("显示坐标", False)
        layout_type = st.selectbox("布局算法", ["auto", "tree", "spring", "circular"])

    # 主界面内容
    if function_mode == "拓扑可视化":
        topology_visualization_page(topology_type, chip_count, locals())
    elif function_mode == "性能对比":
        performance_comparison_page()
    elif function_mode == "路径分析":
        path_analysis_page(topology_type, chip_count, locals())
    elif function_mode == "热点分析":
        hotspot_analysis_page(topology_type, chip_count, locals())


def topology_visualization_page(topology_type, chip_count, config):
    """拓扑可视化页面"""
    st.header("📊 拓扑结构可视化")

    col1, col2 = st.columns([2, 1])

    with col1:
        st.subheader(f"{topology_type}拓扑 - {chip_count}芯片")

        # 生成拓扑
        if topology_type == "Tree":
            topology_data = generate_tree_topology(chip_count, config.get("switch_capacity", 8))
        elif topology_type == "Torus":
            topology_data = generate_torus_topology(chip_count, config.get("dimensions", 2))
        else:
            st.error("暂不支持的拓扑类型")
            return

        # 可视化
        visualizer = TopologyVisualizer(figsize=(12, 8))

        if topology_type == "Tree":
            fig = visualizer.visualize_tree_topology(topology_data["tree_root"], topology_data["all_nodes"])
        elif topology_type == "Torus":
            fig = visualizer.visualize_torus_topology(topology_data["structure"])

        st.pyplot(fig)

        # 显示拓扑信息
        if topology_type == "Tree":
            display_tree_info(topology_data)
        elif topology_type == "Torus":
            display_torus_info(topology_data)

    with col2:
        st.subheader("📈 性能指标")

        # 计算并显示性能指标
        if topology_type == "Tree":
            perf_metrics = evaluate_tree_performance(topology_data["tree_root"], topology_data["all_nodes"])
            st.metric("平均路径长度", f"{perf_metrics['average_path_length']:.2f}跳")
            st.metric("最大路径长度", f"{perf_metrics['max_path_length']}跳")
            st.metric("总节点数", len(topology_data["all_nodes"]))

        elif topology_type == "Torus":
            # 计算Torus性能指标
            connectivity = test_torus_connectivity(topology_data["structure"])
            st.metric("连通性", "✅ 连通" if connectivity["is_connected"] else "❌ 断开")
            st.metric("网格尺寸", str(topology_data["structure"]["grid_dimensions"]))
            st.metric("芯片数量", topology_data["structure"]["chip_count"])

        # 拓扑特性
        st.subheader("🔍 拓扑特性")
        if topology_type == "Tree":
            st.write("**优点:**")
            st.write("- 层次化管理")
            st.write("- 易于扩展")
            st.write("- 故障隔离好")
            st.write("**缺点:**")
            st.write("- 需要额外交换机")
            st.write("- 根节点压力大")

        elif topology_type == "Torus":
            st.write("**优点:**")
            st.write("- 路径长度短")
            st.write("- 无额外硬件")
            st.write("- 高带宽利用率")
            st.write("**缺点:**")
            st.write("- 布线复杂")
            st.write("- 边界效应")


def performance_comparison_page():
    """性能对比页面"""
    st.header("⚡ 拓扑性能对比分析")

    # 配置对比参数
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("对比配置")
        chip_counts = st.multiselect("选择芯片数量", [8, 16, 32, 64, 128], default=[16, 32, 64])

        topologies = st.multiselect("选择拓扑类型", ["Tree", "Torus"], default=["Tree", "Torus"])

    with col2:
        st.subheader("分析选项")
        analyze_path_length = st.checkbox("路径长度分析", True)
        analyze_node_efficiency = st.checkbox("节点效率分析", True)
        analyze_scalability = st.checkbox("可扩展性分析", True)

    if st.button("🚀 开始对比分析", type="primary"):
        with st.spinner("正在进行性能分析..."):
            # 生成对比数据
            comparison_data = generate_comparison_data(chip_counts, topologies)

            # 创建对比图表
            comparator = PerformanceComparator()
            fig = comparator.compare_topologies(comparison_data)

            st.pyplot(fig)

            # 显示数据表
            st.subheader("📊 详细数据")
            display_comparison_table(comparison_data)


def path_analysis_page(topology_type, chip_count, config):
    """路径分析页面"""
    st.header("🛤️ 路径分析工具")

    # 生成拓扑
    if topology_type == "Tree":
        topology_data = generate_tree_topology(chip_count, config.get("switch_capacity", 8))
    elif topology_type == "Torus":
        topology_data = generate_torus_topology(chip_count, config.get("dimensions", 2))
    else:
        st.error("暂不支持的拓扑类型")
        return

    col1, col2 = st.columns([3, 1])

    with col2:
        st.subheader("路径查找")

        # 获取可用节点列表
        if topology_type == "Tree":
            available_nodes = [nid for nid in topology_data["all_nodes"].keys() if "chip" in nid]
        else:
            available_nodes = [f"chip_{i}" for i in range(chip_count)]

        src_node = st.selectbox("源节点", available_nodes)
        dst_node = st.selectbox("目标节点", available_nodes)

        if st.button("🔍 查找路径"):
            # 计算路径
            if topology_type == "Tree":
                path = find_tree_path(topology_data, src_node, dst_node)
            elif topology_type == "Torus":
                path = find_torus_path(topology_data, src_node, dst_node)

            if path:
                st.success(f"找到路径! 跳数: {len(path)-1}")
                st.write("**路径序列:**")
                st.write(" → ".join(path))
            else:
                st.error("未找到路径")

    with col1:
        # 可视化路径
        visualizer = TopologyVisualizer(figsize=(10, 6))

        if topology_type == "Tree":
            fig = visualizer.visualize_tree_topology(topology_data["tree_root"], topology_data["all_nodes"])
        elif topology_type == "Torus":
            fig = visualizer.visualize_torus_topology(topology_data["structure"])

        st.pyplot(fig)


def hotspot_analysis_page(topology_type, chip_count, config):
    """热点分析页面"""
    st.header("🔥 网络热点分析")

    st.info("热点分析可以帮助识别网络中的瓶颈节点和链路")

    # 生成拓扑和模拟流量
    if topology_type == "Tree":
        topology_data = generate_tree_topology(chip_count, config.get("switch_capacity", 8))
    elif topology_type == "Torus":
        topology_data = generate_torus_topology(chip_count, config.get("dimensions", 2))

    # 创建TopologyGraph对象用于分析
    builder = TopologyBuilder("hotspot_analysis")

    # 添加节点和链路到builder
    # (这里需要根据实际拓扑数据构建)

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("中心性分析")
        # 这里可以添加中心性分析的图表
        st.write("度中心性、介数中心性、接近中心性分析")

    with col2:
        st.subheader("带宽利用率")
        # 模拟带宽利用率数据
        st.write("链路带宽利用率热力图")


# 辅助函数


def generate_tree_topology(chip_count, switch_capacity):
    """生成树状拓扑"""
    topo_logic = TreeTopologyLogic()
    chip_ids = list(range(chip_count))
    tree_root, all_nodes = topo_logic.calculate_tree_structure(chip_ids, switch_capacity)

    return {"tree_root": tree_root, "all_nodes": all_nodes, "chip_count": chip_count, "switch_capacity": switch_capacity}


def generate_torus_topology(chip_count, dimensions):
    """生成环形拓扑"""
    topo_logic = TorusTopologyLogic()
    structure = topo_logic.calculate_torus_structure(chip_count, dimensions)

    return {"structure": structure, "chip_count": chip_count, "dimensions": dimensions}


def display_tree_info(topology_data):
    """显示树拓扑信息"""
    st.subheader("🌳 树拓扑信息")

    # 统计信息
    chip_nodes = [n for nid, n in topology_data["all_nodes"].items() if "chip" in nid]
    switch_nodes = [n for nid, n in topology_data["all_nodes"].items() if "switch" in nid]

    info_data = {"指标": ["芯片节点数", "交换机节点数", "总节点数", "交换机容量"], "数值": [len(chip_nodes), len(switch_nodes), len(topology_data["all_nodes"]), topology_data["switch_capacity"]]}

    df = pd.DataFrame(info_data)
    st.table(df)


def display_torus_info(topology_data):
    """显示环形拓扑信息"""
    st.subheader("🔄 Torus拓扑信息")

    structure = topology_data["structure"]
    grid_dims = structure["grid_dimensions"]

    if len(grid_dims) == 2:
        grid_str = f"{grid_dims[0]} × {grid_dims[1]}"
    else:
        grid_str = f"{grid_dims[0]} × {grid_dims[1]} × {grid_dims[2]}"

    info_data = {"指标": ["芯片数量", "维度", "网格尺寸", "平均度数"], "数值": [structure["chip_count"], structure["dimensions"], grid_str, len(grid_dims) * 2]}

    df = pd.DataFrame(info_data)
    st.table(df)


def generate_comparison_data(chip_counts, topologies):
    """生成对比数据"""
    comparison_data = {}

    for topology in topologies:
        comparison_data[topology] = {}

        for count in chip_counts:
            if topology == "Tree":
                topo_logic = TreeTopologyLogic()
                tree_root, all_nodes = topo_logic.calculate_tree_structure(list(range(count)), 8)
                perf = evaluate_tree_performance(tree_root, all_nodes)

                comparison_data[topology][count] = {"avg_path_length": perf["average_path_length"], "max_path_length": perf["max_path_length"], "total_nodes": len(all_nodes)}

            elif topology == "Torus":
                topo_logic = TorusTopologyLogic()
                structure = topo_logic.calculate_torus_structure(count, 2)

                # 简化的Torus性能计算
                routing_logic = TorusRoutingLogic()
                total_hops = 0
                path_count = 0
                max_hops = 0

                # 采样计算平均路径长度
                sample_size = min(20, count)
                for i in range(sample_size):
                    for j in range(i + 1, sample_size):
                        if i < count and j < count:
                            src_coord = structure["coordinate_map"][i]
                            dst_coord = structure["coordinate_map"][j]
                            distances = routing_logic.calculate_shortest_distance(src_coord, dst_coord, structure["grid_dimensions"])
                            total_hops += distances["total_hops"]
                            max_hops = max(max_hops, distances["total_hops"])
                            path_count += 1

                avg_hops = total_hops / path_count if path_count > 0 else 0

                comparison_data[topology][count] = {"avg_path_length": avg_hops, "max_path_length": max_hops, "total_nodes": count}

    return comparison_data


def display_comparison_table(comparison_data):
    """显示对比数据表格"""
    rows = []
    for topology, data in comparison_data.items():
        for chip_count, metrics in data.items():
            rows.append(
                {
                    "拓扑类型": topology,
                    "芯片数量": chip_count,
                    "平均路径长度": f"{metrics['avg_path_length']:.2f}",
                    "最大路径长度": metrics["max_path_length"],
                    "总节点数": metrics["total_nodes"],
                    "节点效率": f"{chip_count/metrics['total_nodes']*100:.1f}%",
                }
            )

    df = pd.DataFrame(rows)
    st.dataframe(df, use_container_width=True)


def find_tree_path(topology_data, src_node, dst_node):
    """查找树状拓扑中的路径"""
    # 简化实现：使用BFS查找路径
    from collections import deque

    all_nodes = topology_data["all_nodes"]

    if src_node not in all_nodes or dst_node not in all_nodes:
        return None

    # 构建邻接关系
    adjacency = {}
    for node_id, node in all_nodes.items():
        adjacency[node_id] = []
        if hasattr(node, "parent") and node.parent:
            adjacency[node_id].append(node.parent.node_id)
        if hasattr(node, "children"):
            for child in node.children:
                adjacency[node_id].append(child.node_id)

    # BFS查找路径
    queue = deque([(src_node, [src_node])])
    visited = {src_node}

    while queue:
        current, path = queue.popleft()

        if current == dst_node:
            return path

        for neighbor in adjacency.get(current, []):
            if neighbor not in visited:
                visited.add(neighbor)
                queue.append((neighbor, path + [neighbor]))

    return None


def find_torus_path(topology_data, src_node, dst_node):
    """查找Torus拓扑中的路径"""
    structure = topology_data["structure"]
    routing_logic = TorusRoutingLogic()

    # 获取坐标
    src_id = int(src_node.split("_")[1])
    dst_id = int(dst_node.split("_")[1])

    if src_id not in structure["coordinate_map"] or dst_id not in structure["coordinate_map"]:
        return None

    src_coord = structure["coordinate_map"][src_id]
    dst_coord = structure["coordinate_map"][dst_id]

    # 计算路径
    coord_path = routing_logic.dimension_order_routing(src_coord, dst_coord, structure["grid_dimensions"])

    # 转换为节点ID路径
    node_path = []
    for coord in coord_path:
        if coord in structure["id_map"]:
            chip_id = structure["id_map"][coord]
            node_path.append(f"chip_{chip_id}")

    return node_path


if __name__ == "__main__":
    main()
