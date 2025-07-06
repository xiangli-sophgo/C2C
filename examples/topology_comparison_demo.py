#!/usr/bin/env python3
"""
Topology Comparison Demo

This example compares the performance of Mesh and CrossRing topologies
using the same traffic patterns.
"""

import sys
import os
import time

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from examples.unified_noc_simulation import UnifiedNoCSimulation, create_sample_traffic_file


def run_comparison(topology_size=(4, 4), max_cycles=1000, traffic_requests=50):
    """运行拓扑对比测试"""
    
    print("NoC拓扑性能对比测试")
    print("=" * 60)
    print(f"拓扑大小: {topology_size[0]}x{topology_size[1]}")
    print(f"仿真周期: {max_cycles}")
    print(f"Traffic请求数: {traffic_requests}")
    print()
    
    # 创建共同的traffic文件
    traffic_dir = "traffic_data"
    if not os.path.exists(traffic_dir):
        os.makedirs(traffic_dir)
    
    comparison_traffic = os.path.join(traffic_dir, "comparison_traffic.txt")
    create_sample_traffic_file(comparison_traffic, traffic_requests)
    
    results = {}
    
    # 测试Mesh拓扑
    print("测试Mesh拓扑...")
    print("-" * 30)
    
    start_time = time.time()
    mesh_sim = UnifiedNoCSimulation(
        topology_type="mesh",
        topology_size=topology_size
    )
    mesh_sim.setup_traffic(traffic_dir, ["comparison_traffic.txt"])
    mesh_sim.run_simulation(max_cycles=max_cycles)
    mesh_time = time.time() - start_time
    
    mesh_stats = mesh_sim.model.get_network_statistics()
    results['mesh'] = {
        'simulation_time': mesh_time,
        'network_stats': mesh_stats,
        'topology_info': mesh_sim.model.get_topology_info()
    }
    
    print()
    
    # 测试CrossRing拓扑
    print("测试CrossRing拓扑...")
    print("-" * 30)
    
    start_time = time.time()
    crossring_sim = UnifiedNoCSimulation(
        topology_type="crossring",
        topology_size=topology_size
    )
    crossring_sim.setup_traffic(traffic_dir, ["comparison_traffic.txt"])
    crossring_sim.run_simulation(max_cycles=max_cycles)
    crossring_time = time.time() - start_time
    
    crossring_stats = crossring_sim.model.get_network_statistics()
    results['crossring'] = {
        'simulation_time': crossring_time,
        'network_stats': crossring_stats,
        'topology_info': crossring_sim.model.get_node_count()
    }
    
    print()
    
    # 打印对比结果
    print_comparison_results(results)
    
    return results


def print_comparison_results(results):
    """打印对比结果"""
    print("=" * 60)
    print("拓扑性能对比结果")
    print("=" * 60)
    
    mesh_stats = results['mesh']['network_stats']
    crossring_stats = results['crossring']['network_stats']
    
    print(f"{'指标':<20} {'Mesh':<15} {'CrossRing':<15} {'对比':<15}")
    print("-" * 70)
    
    # 仿真时间
    mesh_sim_time = results['mesh']['simulation_time']
    crossring_sim_time = results['crossring']['simulation_time']
    time_ratio = crossring_sim_time / mesh_sim_time if mesh_sim_time > 0 else 0
    print(f"{'仿真时间(秒)':<20} {mesh_sim_time:<15.3f} {crossring_sim_time:<15.3f} {time_ratio:<15.2f}")
    
    # 网络利用率
    mesh_util = mesh_stats.get('utilization', 0)
    crossring_util = crossring_stats.get('utilization', 0)
    util_ratio = crossring_util / mesh_util if mesh_util > 0 else 0
    print(f"{'网络利用率(%)':<20} {mesh_util:<15.2f} {crossring_util:<15.2f} {util_ratio:<15.2f}")
    
    # 平均延迟
    mesh_latency = mesh_stats.get('avg_latency', 0)
    crossring_latency = crossring_stats.get('avg_latency', 0)
    latency_ratio = crossring_latency / mesh_latency if mesh_latency > 0 else 0
    print(f"{'平均延迟(周期)':<20} {mesh_latency:<15.2f} {crossring_latency:<15.2f} {latency_ratio:<15.2f}")
    
    # 平均跳数
    mesh_hops = mesh_stats.get('avg_hops', 0)
    crossring_hops = crossring_stats.get('avg_hops', 0)
    hops_ratio = crossring_hops / mesh_hops if mesh_hops > 0 else 0
    print(f"{'平均跳数':<20} {mesh_hops:<15.2f} {crossring_hops:<15.2f} {hops_ratio:<15.2f}")
    
    # 吞吐量
    mesh_throughput = mesh_stats.get('throughput', 0)
    crossring_throughput = crossring_stats.get('throughput', 0)
    throughput_ratio = crossring_throughput / mesh_throughput if mesh_throughput > 0 else 0
    print(f"{'吞吐量(包/周期)':<20} {mesh_throughput:<15.4f} {crossring_throughput:<15.4f} {throughput_ratio:<15.2f}")
    
    # 活跃包数
    mesh_active = mesh_stats.get('active_packets', 0)
    crossring_active = crossring_stats.get('active_packets', 0)
    print(f"{'活跃包数':<20} {mesh_active:<15} {crossring_active:<15} {'-':<15}")
    
    print()
    
    # 拓扑特性
    print("拓扑特性对比:")
    print(f"  Mesh拓扑:")
    mesh_topo = results['mesh']['topology_info']
    if 'network_diameter' in mesh_stats:
        print(f"    - 网络直径: {mesh_stats['network_diameter']}")
    if 'average_degree' in mesh_stats:
        print(f"    - 平均度数: {mesh_stats['average_degree']:.2f}")
    
    print(f"  CrossRing拓扑:")
    print(f"    - 节点数: {results['crossring']['topology_info']}")
    
    print()
    
    # 结论
    print("结论:")
    if mesh_util > crossring_util:
        print("  - Mesh拓扑在此测试中显示出更高的网络利用率")
    elif crossring_util > mesh_util:
        print("  - CrossRing拓扑在此测试中显示出更高的网络利用率")
    else:
        print("  - 两种拓扑的网络利用率相近")
    
    if mesh_latency > 0 and crossring_latency > 0:
        if mesh_latency < crossring_latency:
            print("  - Mesh拓扑显示出更低的平均延迟")
        elif crossring_latency < mesh_latency:
            print("  - CrossRing拓扑显示出更低的平均延迟")
        else:
            print("  - 两种拓扑的延迟性能相近")


def main():
    """主函数"""
    import argparse
    
    parser = argparse.ArgumentParser(description='NoC拓扑性能对比')
    parser.add_argument('--rows', type=int, default=4, help='行数 (默认: 4)')
    parser.add_argument('--cols', type=int, default=4, help='列数 (默认: 4)')
    parser.add_argument('--cycles', type=int, default=1000, help='仿真周期 (默认: 1000)')
    parser.add_argument('--requests', type=int, default=50, help='Traffic请求数 (默认: 50)')
    
    args = parser.parse_args()
    
    # 运行对比测试
    results = run_comparison(
        topology_size=(args.rows, args.cols),
        max_cycles=args.cycles,
        traffic_requests=args.requests
    )
    
    print("\n对比测试完成!")


if __name__ == "__main__":
    main()