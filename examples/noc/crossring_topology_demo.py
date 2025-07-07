#!/usr/bin/env python3
"""
CrossRing拓扑使用示例。

本示例演示了CrossRing拓扑的完整使用方法，包括：
- 创建不同规模的CrossRing拓扑
- 基本查询操作
- 路径计算演示
- 拓扑性能分析
- 不同路由策略比较
- 可视化和导出功能

使用方法：
    python examples/crossring_topology_example.py
"""

import sys
import os
import time
import logging
from typing import List, Dict, Tuple, Any

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from src.noc.crossring.topology import CrossRingTopology
from src.noc.crossring.config import CrossRingConfig
from src.noc.utils.types import RoutingStrategy, TopologyType
from src.noc.utils.adjacency import export_adjacency_matrix


def setup_logging():
    """配置日志输出。"""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=[
            logging.StreamHandler(),
        ],
    )


def create_basic_crossring_example():
    """基本CrossRing拓扑创建示例。"""
    print("\n" + "=" * 60)
    print("基本CrossRing拓扑创建示例")
    print("=" * 60)

    # 创建3x3 CrossRing拓扑
    config = CrossRingConfig(num_col=4, num_row=3)

    print(f"创建CrossRing拓扑配置: {config.num_row}×{config.num_col}")

    # 构建拓扑
    start_time = time.time()
    topology = CrossRingTopology(config)
    build_time = time.time() - start_time

    print(f"拓扑构建完成，耗时: {build_time:.4f}秒")
    print(f"拓扑信息: {topology}")

    # 验证拓扑
    is_valid, error_msg = topology.validate_topology()
    print(f"拓扑验证结果: {'有效' if is_valid else '无效'}")
    if error_msg:
        print(f"错误信息: {error_msg}")

    return topology


def demonstrate_basic_queries(topology: CrossRingTopology):
    """演示基本查询功能。"""
    print("\n" + "=" * 60)
    print("基本查询功能演示")
    print("=" * 60)

    # 获取拓扑基本信息
    info = topology.get_crossring_info()
    print(f"拓扑类型: {info['topology_type']}")
    print(f"节点数量: {info['num_nodes']}")
    print(f"网络直径: {info['diameter']}")
    print(f"平均跳数: {info['average_hop_count']:.2f}")
    print(f"拓扑效率: {info['topology_efficiency']:.4f}")
    print(f"支持的路由策略: {info['supported_routing']}")

    # 节点邻居查询
    print(f"\n节点邻居关系:")
    for node_id in range(min(9, topology.num_nodes)):  # 只显示前9个节点
        neighbors = topology.get_neighbors(node_id)
        position = topology.get_node_position(node_id)
        print(f"节点{node_id} {position}: 邻居 {neighbors}")

    # 环结构分析
    h_rings = topology.get_horizontal_rings()
    v_rings = topology.get_vertical_rings()
    print(f"\n水平环结构 ({len(h_rings)}个):")
    for i, ring in enumerate(h_rings):
        print(f"  环{i}: {ring}")

    print(f"\n垂直环结构 ({len(v_rings)}个):")
    for i, ring in enumerate(v_rings):
        print(f"  环{i}: {ring}")


def demonstrate_path_calculation(topology: CrossRingTopology):
    """演示路径计算功能。"""
    print("\n" + "=" * 60)
    print("路径计算功能演示")
    print("=" * 60)

    # 选择测试节点对
    src, dst = 0, topology.num_nodes - 1  # 从左上角到右下角

    print(f"计算节点{src} -> 节点{dst}的路径:")
    print(f"源节点位置: {topology.get_node_position(src)}")
    print(f"目标节点位置: {topology.get_node_position(dst)}")

    # 最短路径
    shortest_path = topology.calculate_shortest_path(src, dst)
    print(f"\n最短路径 (BFS): {shortest_path}")
    print(f"跳数: {len(shortest_path) - 1}")

    # HV路径（水平优先）
    hv_path = topology.calculate_hv_path(src, dst)
    print(f"\nHV路径 (水平优先): {hv_path}")
    print(f"跳数: {len(hv_path) - 1}")

    # VH路径（垂直优先）
    vh_path = topology.calculate_vh_path(src, dst)
    print(f"\nVH路径 (垂直优先): {vh_path}")
    print(f"跳数: {len(vh_path) - 1}")

    # 自适应路径
    adaptive_path = topology.calculate_route(src, dst, RoutingStrategy.ADAPTIVE)
    print(f"\n自适应路径: {adaptive_path}")
    print(f"跳数: {len(adaptive_path) - 1}")

    # 负载均衡路径
    lb_path = topology.calculate_route(src, dst, RoutingStrategy.LOAD_BALANCED)
    print(f"\n负载均衡路径: {lb_path}")
    print(f"跳数: {len(lb_path) - 1}")


def demonstrate_ring_distances(topology: CrossRingTopology):
    """演示环内距离计算。"""
    print("\n" + "=" * 60)
    print("环内距离计算演示")
    print("=" * 60)

    # 选择测试节点
    if topology.num_nodes >= 9:  # 3x3拓扑
        test_pairs = [
            (0, 2),  # 同一行，通过环
            (0, 6),  # 同一列，通过环
            (1, 7),  # 同一列
            (3, 5),  # 同一行
        ]

        for src, dst in test_pairs:
            src_pos = topology.get_node_position(src)
            dst_pos = topology.get_node_position(dst)

            print(f"\n节点{src}{src_pos} -> 节点{dst}{dst_pos}:")

            # 水平环距离
            h_dist = topology.get_ring_distance(src, dst, "horizontal")
            if h_dist != float("inf"):
                print(f"  水平环距离: {h_dist}")
            else:
                print(f"  不在同一水平环")

            # 垂直环距离
            v_dist = topology.get_ring_distance(src, dst, "vertical")
            if v_dist != float("inf"):
                print(f"  垂直环距离: {v_dist}")
            else:
                print(f"  不在同一垂直环")


def performance_analysis_example():
    """性能分析示例。"""
    print("\n" + "=" * 60)
    print("性能分析示例")
    print("=" * 60)

    # 测试不同规模的拓扑
    sizes = [(3, 3), (4, 4), (6, 6), (8, 8)]

    print(f"{'规模':<8} {'节点数':<8} {'直径':<8} {'平均跳数':<12} {'效率':<12} {'构建时间(ms)':<15}")
    print("-" * 75)

    for rows, cols in sizes:
        config = CrossRingConfig(num_row=rows, num_col=cols)

        # 测量构建时间
        start_time = time.time()
        topology = CrossRingTopology(config)
        build_time = (time.time() - start_time) * 1000  # 转换为毫秒

        # 获取性能指标
        diameter = topology.get_diameter()
        avg_hop_count = topology.get_average_hop_count()
        efficiency = topology.get_topology_efficiency()

        print(f"{rows}×{cols:<6} {topology.num_nodes:<8} {diameter:<8} " f"{avg_hop_count:<12.2f} {efficiency:<12.4f} {build_time:<15.2f}")


def routing_strategy_comparison(topology: CrossRingTopology):
    """路由策略比较示例。"""
    print("\n" + "=" * 60)
    print("路由策略比较示例")
    print("=" * 60)

    # 选择多个测试节点对
    test_pairs = []
    if topology.num_nodes >= 9:
        test_pairs = [(0, 8), (1, 7), (2, 6), (3, 5)]
    else:
        test_pairs = [(0, topology.num_nodes - 1)]

    strategies = [RoutingStrategy.SHORTEST, RoutingStrategy.DETERMINISTIC, RoutingStrategy.MINIMAL, RoutingStrategy.ADAPTIVE, RoutingStrategy.LOAD_BALANCED]

    print(f"{'节点对':<10} {'策略':<15} {'路径长度':<10} {'路径'}")
    print("-" * 80)

    for src, dst in test_pairs:
        for strategy in strategies:
            try:
                path = topology.calculate_route(src, dst, strategy)
                path_length = len(path) - 1
                path_str = " -> ".join(map(str, path[:5]))  # 只显示前5个节点
                if len(path) > 5:
                    path_str += " -> ..."

                print(f"{src}->{dst:<7} {strategy.value:<15} {path_length:<10} {path_str}")
            except Exception as e:
                print(f"{src}->{dst:<7} {strategy.value:<15} ERROR     错误: {str(e)[:20]}...")
        print()


def load_distribution_analysis_example(topology: CrossRingTopology):
    """负载分布分析示例。"""
    print("\n" + "=" * 60)
    print("负载分布分析示例")
    print("=" * 60)

    # 模拟一些流量以测试负载分布
    print("模拟网络流量...")

    # 为一些链路添加模拟利用率
    test_links = []
    for node in range(min(topology.num_nodes, 9)):
        neighbors = topology.get_neighbors(node)
        for neighbor in neighbors[:2]:  # 只使用前两个邻居
            link = (node, neighbor)
            test_links.append(link)

    # 设置模拟利用率
    import random

    random.seed(42)  # 固定随机种子以获得可重现的结果

    for i, link in enumerate(test_links):
        utilization = random.uniform(0.1, 0.9)
        topology.update_link_metrics(link, {"utilization": utilization})

    # 分析负载分布
    load_dist = topology.calculate_load_distribution()

    print(f"总链路数: {load_dist['total_links']}")
    print(f"活跃链路数: {load_dist['active_links']}")
    print(f"负载方差: {load_dist['load_variance']:.4f}")

    print(f"\n利用率统计:")
    stats = load_dist["utilization_stats"]
    print(f"  平均值: {stats['mean']:.4f}")
    print(f"  标准差: {stats['std']:.4f}")
    print(f"  最小值: {stats['min']:.4f}")
    print(f"  最大值: {stats['max']:.4f}")

    print(f"\n负载分布:")
    for bin_name, count in load_dist["load_distribution"].items():
        print(f"  {bin_name}: {count} 条链路")


def export_topology_example(topology: CrossRingTopology):
    """拓扑导出示例。"""
    print("\n" + "=" * 60)
    print("拓扑导出示例")
    print("=" * 60)

    # 创建输出目录
    output_dir = "../output/crossring_topology_demo"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"创建输出目录: {output_dir}")

    # 导出邻接矩阵
    adj_matrix = topology.get_adjacency_matrix()

    # 导出为不同格式
    formats = ["csv"]
    for fmt in formats:
        filename = os.path.join(output_dir, f"crossring_{topology.num_rows}x{topology.num_cols}_adjacency.{fmt}")
        try:
            export_adjacency_matrix(adj_matrix, filename, fmt)
            print(f"邻接矩阵已导出为 {fmt.upper()} 格式: {filename}")
        except Exception as e:
            print(f"导出 {fmt.upper()} 格式失败: {e}")

    # 导出拓扑信息
    info_filename = os.path.join(output_dir, f"crossring_{topology.num_rows}x{topology.num_cols}_info.txt")
    with open(info_filename, "w", encoding="utf-8") as f:
        info = topology.get_crossring_info()
        f.write("CrossRing拓扑信息报告\n")
        f.write("=" * 50 + "\n\n")

        f.write(f"拓扑规模: {topology.num_rows}×{topology.num_cols}\n")
        f.write(f"节点总数: {info['num_nodes']}\n")
        f.write(f"网络直径: {info['diameter']}\n")
        f.write(f"平均跳数: {info['average_hop_count']:.4f}\n")
        f.write(f"拓扑效率: {info['topology_efficiency']:.4f}\n")
        f.write(f"水平环数: {info['horizontal_rings']}\n")
        f.write(f"垂直环数: {info['vertical_rings']}\n")
        f.write(f"支持路由: {', '.join(info['supported_routing'])}\n\n")

        # 邻居关系
        f.write("节点邻居关系:\n")
        f.write("-" * 30 + "\n")
        for node_id in range(topology.num_nodes):
            neighbors = topology.get_neighbors(node_id)
            position = topology.get_node_position(node_id)
            f.write(f"节点{node_id} {position}: {neighbors}\n")

    print(f"拓扑信息已导出: {info_filename}")


def visualization_example(topology: CrossRingTopology):
    """可视化示例。"""
    print("\n" + "=" * 60)
    print("可视化示例")
    print("=" * 60)

    # 使用内置的文本可视化
    viz_text = topology.visualize_topology()
    print(viz_text)

    # 创建简单的ASCII网格可视化
    print(f"\nASCII网格可视化 ({topology.num_rows}×{topology.num_cols}):")
    print("-" * (topology.num_cols * 6 + 1))

    for row in range(topology.num_rows):
        # 节点行
        line = "|"
        for col in range(topology.num_cols):
            node_id = row * topology.num_cols + col
            line += f" {node_id:2d}  |"
        print(line)

        # 连接行（除了最后一行）
        if row < topology.num_rows - 1:
            line = "|"
            for col in range(topology.num_cols):
                line += "  |  |"
            print(line)

        print("-" * (topology.num_cols * 6 + 1))


def main():
    """主函数 - 运行所有示例。"""
    print("CrossRing拓扑完整使用示例")
    print("=" * 80)

    # 设置日志
    setup_logging()

    try:
        # 1. 基本创建示例
        topology = create_basic_crossring_example()

        # 2. 基本查询演示
        demonstrate_basic_queries(topology)

        # 3. 路径计算演示
        demonstrate_path_calculation(topology)

        # 4. 环内距离演示
        demonstrate_ring_distances(topology)

        # 5. 性能分析
        performance_analysis_example()

        # 6. 路由策略比较
        routing_strategy_comparison(topology)

        # 7. 负载分布分析
        load_distribution_analysis_example(topology)

        # 8. 拓扑导出
        export_topology_example(topology)

        # 9. 可视化
        visualization_example(topology)

        print("\n" + "=" * 80)
        print("所有示例运行完成！")
        print("=" * 80)

    except Exception as e:
        print(f"运行示例时发生错误: {e}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    main()
