# -*- coding: utf-8 -*-
"""
C2C拓扑可视化演示脚本
展示所有可视化功能和分析工具
"""

import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import matplotlib.pyplot as plt
from src.visualization.visualizer import TopologyVisualizer
from src.visualization.comparison import PerformanceComparator
from src.visualization.utils import color_manager, graphics_utils
from src.topology.tree import TreeTopologyLogic, evaluate_tree_performance
from src.topology.torus import TorusTopologyLogic, TorusRoutingLogic, test_torus_connectivity
from src.topology.node import ChipNode, SwitchNode, HostNode
from src.topology.link import C2CDirectLink, PCIeLink
from src.topology.builder import TopologyBuilder


def demo_tree_visualization():
    """演示树状拓扑可视化"""
    print("=== 树状拓扑可视化演示 ===")

    # 创建输出目录
    os.makedirs("../output", exist_ok=True)

    # 创建树状拓扑
    topo_logic = TreeTopologyLogic()
    chip_ids = list(range(16))
    tree_root, all_nodes = topo_logic.calculate_tree_structure(chip_ids, switch_capacity=4)

    # 可视化
    visualizer = TopologyVisualizer(figsize=(12, 8))
    fig = visualizer.visualize_tree_topology(tree_root, all_nodes)

    # 保存图片
    visualizer.save_figure("../output/tree_topology_demo.png")

    # 显示性能指标
    perf = evaluate_tree_performance(tree_root, all_nodes)
    print(f"16芯片树拓扑性能:")
    print(f"  平均路径长度: {perf['average_path_length']:.2f}跳")
    print(f"  最大路径长度: {perf['max_path_length']}跳")
    print(f"  总节点数: {len(all_nodes)}")

    visualizer.show()
    visualizer.close()


def demo_torus_visualization():
    """演示环形拓扑可视化"""
    print("\n=== 环形拓扑可视化演示 ===")

    # 创建2D环形拓扑
    topo_logic = TorusTopologyLogic()
    torus_2d = topo_logic.calculate_torus_structure(16, dimensions=2)

    print(f"2D Torus ({torus_2d['grid_dimensions'][0]}x{torus_2d['grid_dimensions'][1]}):")

    # 可视化2D
    visualizer = TopologyVisualizer(figsize=(10, 8))
    fig = visualizer.visualize_torus_topology(torus_2d)

    visualizer.save_figure("../output/torus_2d_demo.png")
    visualizer.show()
    visualizer.close()

    # 创建3D环形拓扑
    torus_3d = topo_logic.calculate_torus_structure(27, dimensions=3)
    print(f"3D Torus ({torus_3d['grid_dimensions'][0]}x{torus_3d['grid_dimensions'][1]}x{torus_3d['grid_dimensions'][2]}):")

    # 可视化3D（投影）
    visualizer = TopologyVisualizer(figsize=(12, 8))
    fig = visualizer.visualize_torus_topology(torus_3d)

    visualizer.save_figure("../output/torus_3d_demo.png")
    visualizer.show()
    visualizer.close()

    # 验证连通性
    connectivity = test_torus_connectivity(torus_2d)
    print(f"2D Torus连通性: {'✓ 连通' if connectivity['is_connected'] else '✗ 断开'}")


def demo_mixed_topology():
    """演示混合拓扑（包含不同节点类型）"""
    print("\n=== 混合拓扑可视化演示 ===")

    # 创建复杂的混合拓扑
    builder = TopologyBuilder("mixed_demo")

    # 创建节点
    chips = [ChipNode(f"chip_{i}", f"board_{i//4}", ["DDR", "HBM"]) for i in range(8)]
    switches = [SwitchNode(f"switch_{i}", port_count=8, bandwidth=128.0) for i in range(3)]
    host = HostNode("host_0", pcie_lanes=16)

    # 添加所有节点
    for chip in chips:
        builder.add_node(chip)
    for switch in switches:
        builder.add_node(switch)
    builder.add_node(host)

    # 创建连接
    # Host到主交换机
    builder.add_link(PCIeLink("link_host_main", host, switches[0], "x16"))

    # 主交换机到子交换机
    builder.add_link(PCIeLink("link_main_sub1", switches[0], switches[1], "x8"))
    builder.add_link(PCIeLink("link_main_sub2", switches[0], switches[2], "x8"))

    # 交换机到芯片
    for i in range(4):
        builder.add_link(PCIeLink(f"link_switch1_chip{i}", switches[1], chips[i], "x8"))
        builder.add_link(PCIeLink(f"link_switch2_chip{i+4}", switches[2], chips[i + 4], "x8"))

    # 芯片间C2C直连
    builder.add_link(C2CDirectLink("link_chip0_chip1", chips[0], chips[1]))
    builder.add_link(C2CDirectLink("link_chip2_chip3", chips[2], chips[3]))
    builder.add_link(C2CDirectLink("link_chip4_chip5", chips[4], chips[5]))
    builder.add_link(C2CDirectLink("link_chip6_chip7", chips[6], chips[7]))

    # 构建拓扑
    topology = builder.build()

    # 可视化
    visualizer = TopologyVisualizer(figsize=(14, 10))
    fig = visualizer.visualize_topology_graph(topology, layout_type="spring")

    visualizer.save_figure("../output/mixed_topology_demo.png")

    # 显示拓扑统计
    stats = topology.get_topology_statistics()
    print(f"混合拓扑统计:")
    print(f"  节点数: {stats['num_nodes']}")
    print(f"  链路数: {stats['num_links']}")
    print(f"  平均度数: {stats['average_degree']:.2f}")
    print(f"  连通性: {'✓' if stats['is_connected'] else '✗'}")

    visualizer.show()
    visualizer.close()


def demo_performance_comparison():
    """演示性能对比功能"""
    print("\n=== 拓扑性能对比演示 ===")

    # 生成对比数据
    chip_counts = [16, 32, 64]
    comparison_data = {}

    # Tree拓扑数据
    comparison_data["Tree"] = {}
    tree_logic = TreeTopologyLogic()

    for count in chip_counts:
        tree_root, all_nodes = tree_logic.calculate_tree_structure(list(range(count)), 8)
        perf = evaluate_tree_performance(tree_root, all_nodes)

        comparison_data["Tree"][count] = {"avg_path_length": perf["average_path_length"], "max_path_length": perf["max_path_length"], "total_nodes": len(all_nodes)}

    # Torus拓扑数据
    comparison_data["Torus"] = {}
    torus_logic = TorusTopologyLogic()
    routing_logic = TorusRoutingLogic()

    for count in chip_counts:
        structure = torus_logic.calculate_torus_structure(count, 2)

        # 计算平均路径长度
        total_hops = 0
        path_count = 0
        max_hops = 0

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

        comparison_data["Torus"][count] = {"avg_path_length": avg_hops, "max_path_length": max_hops, "total_nodes": count}

    # 创建对比图表
    comparator = PerformanceComparator(figsize=(15, 10))
    fig = comparator.compare_topologies(comparison_data)

    # 保存图表
    fig.savefig("../output/performance_comparison_demo.png", dpi=300, bbox_inches="tight")

    # 生成报告
    report = comparator.generate_performance_report("../output/performance_report.md")
    print("性能对比报告已生成")

    plt.show()
    plt.close()


def demo_path_analysis():
    """演示路径分析功能"""
    print("\n=== 路径分析演示 ===")

    # 使用Torus拓扑进行路径分析
    topo_logic = TorusTopologyLogic()
    torus_structure = topo_logic.calculate_torus_structure(16, dimensions=2)
    routing_logic = TorusRoutingLogic()

    print("分析4x4 Torus拓扑中的路径:")

    # 分析几个关键路径
    test_pairs = [(0, 15), (0, 5), (3, 12), (6, 10)]

    for src_id, dst_id in test_pairs:
        src_coord = torus_structure["coordinate_map"][src_id]
        dst_coord = torus_structure["coordinate_map"][dst_id]

        # 计算路径
        path = routing_logic.dimension_order_routing(src_coord, dst_coord, torus_structure["grid_dimensions"])

        # 计算距离信息
        distances = routing_logic.calculate_shortest_distance(src_coord, dst_coord, torus_structure["grid_dimensions"])

        print(f"  芯片{src_id}{src_coord} → 芯片{dst_id}{dst_coord}:")
        print(f"    路径长度: {len(path)-1}跳")
        print(f"    详细距离: {distances}")
        print(f"    路径: {src_coord} → ... → {dst_coord}")


def demo_color_schemes():
    """演示不同颜色方案"""
    print("\n=== 颜色方案演示 ===")

    # 创建简单拓扑用于演示
    topo_logic = TreeTopologyLogic()
    tree_root, all_nodes = topo_logic.calculate_tree_structure(list(range(8)), 4)

    # 测试不同颜色方案
    schemes = ["default", "colorblind", "dark"]

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    for i, scheme in enumerate(schemes):
        color_manager.set_scheme(scheme)

        visualizer = TopologyVisualizer(figsize=(6, 6))
        visualizer.node_colors = color_manager.get_scheme()

        # 需要手动设置ax
        visualizer.ax = axes[i]
        visualizer.fig = fig

        # 简化版可视化（直接在指定ax上绘制）
        axes[i].set_title(f"{scheme.title()}颜色方案")
        axes[i].text(0.5, 0.5, f"{scheme}方案演示", ha="center", va="center", transform=axes[i].transAxes)

        # 显示颜色图例
        colors = color_manager.get_scheme()
        y_pos = 0.8
        for element, color in colors.items():
            axes[i].scatter(0.1, y_pos, c=color, s=100, alpha=0.8)
            axes[i].text(0.2, y_pos, element, va="center", transform=axes[i].transAxes)
            y_pos -= 0.1

        axes[i].set_xlim(0, 1)
        axes[i].set_ylim(0, 1)
        axes[i].set_axis_off()

    plt.tight_layout()
    plt.savefig("../output/color_schemes_demo.png", dpi=300, bbox_inches="tight")
    plt.show()


def main():
    """主演示函数"""
    print("🖥️ C2C拓扑可视化工具演示")
    print("=" * 50)

    # 设置图形样式
    graphics_utils.apply_style("seaborn")

    # 运行各种演示
    try:
        demo_tree_visualization()
        demo_torus_visualization()
        demo_mixed_topology()
        demo_performance_comparison()
        demo_path_analysis()
        demo_color_schemes()

        print("\n🎉 所有演示完成！")
        print("生成的文件保存在 ../output/ 目录:")
        print("  - ../output/tree_topology_demo.png")
        print("  - ../output/torus_2d_demo.png")
        print("  - ../output/torus_3d_demo.png")
        print("  - ../output/mixed_topology_demo.png")
        print("  - ../output/performance_comparison_demo.png")
        print("  - ../output/performance_report.md")
        print("  - ../output/color_schemes_demo.png")

    except Exception as e:
        print(f"❌ 演示过程中发生错误: {str(e)}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    main()
