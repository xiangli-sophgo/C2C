#!/usr/bin/env python3
"""
NoC 分析演示
展示如何使用新的分析和可视化功能
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.noc.analysis import ResultProcessor, PerformanceVisualizer, NetworkFlowVisualizer
from src.noc.analysis.performance_metrics import RequestMetrics, RequestType
import numpy as np
import matplotlib.pyplot as plt


def create_sample_requests(num_requests: int = 1000) -> list:
    """创建示例请求数据"""
    requests = []
    
    np.random.seed(42)  # 固定随机种子以获得可重复的结果
    
    for i in range(num_requests):
        # 模拟不同类型的请求
        request_type = RequestType.READ if np.random.random() < 0.6 else RequestType.WRITE
        
        # 随机选择源和目标节点
        source_node = np.random.randint(0, 16)
        dest_node = np.random.randint(0, 16)
        while dest_node == source_node:
            dest_node = np.random.randint(0, 16)
        
        # 模拟时间参数
        start_time = np.random.randint(0, 10000)
        cmd_latency = np.random.randint(50, 150)
        data_latency = np.random.randint(100, 300)
        network_latency = np.random.randint(20, 80)
        end_time = start_time + cmd_latency + data_latency + network_latency
        
        # 模拟数据量
        burst_size = np.random.choice([1, 2, 4, 8])
        total_bytes = burst_size * 128  # 假设每个flit 128字节
        
        # 计算跳数（基于曼哈顿距离）
        src_x, src_y = source_node % 4, source_node // 4
        dst_x, dst_y = dest_node % 4, dest_node // 4
        hop_count = abs(src_x - dst_x) + abs(src_y - dst_y)
        
        request = RequestMetrics(
            packet_id=f"req_{i}",
            request_type=request_type,
            source_node=source_node,
            dest_node=dest_node,
            burst_size=burst_size,
            start_time=start_time,
            end_time=end_time,
            cmd_latency=cmd_latency,
            data_latency=data_latency,
            network_latency=network_latency,
            total_bytes=total_bytes,
            hop_count=hop_count,
            path_nodes=list(range(source_node, dest_node + 1))  # 简化路径
        )
        
        requests.append(request)
    
    return requests


def demonstrate_analysis():
    """演示分析功能"""
    print("=" * 60)
    print("NoC 性能分析演示")
    print("=" * 60)
    
    # 创建示例数据
    print("\n1. 创建示例请求数据...")
    requests = create_sample_requests(1000)
    print(f"   创建了 {len(requests)} 个请求")
    
    # 初始化结果处理器
    print("\n2. 初始化结果处理器...")
    config = {
        'topology_type': 'mesh_4x4',
        'node_count': 16,
        'gap_threshold_ns': 200,
        'window_size_ns': 1000,
        'congestion_threshold': 0.8
    }
    
    processor = ResultProcessor(config)
    
    # 添加请求数据
    print("\n3. 添加请求数据到处理器...")
    for req in requests:
        processor.performance_metrics.add_request(req)
    
    # 执行性能分析
    print("\n4. 执行性能分析...")
    network_metrics = processor.analyze_performance()
    
    # 显示分析结果
    print("\n5. 分析结果摘要:")
    print("-" * 40)
    summary = processor.get_performance_summary()
    
    for key, value in summary.items():
        if isinstance(value, float):
            print(f"   {key}: {value:.3f}")
        else:
            print(f"   {key}: {value}")
    
    return processor, network_metrics, requests


def demonstrate_visualization(processor, network_metrics, requests):
    """演示可视化功能"""
    print("\n" + "=" * 60)
    print("NoC 可视化演示")
    print("=" * 60)
    
    # 创建可视化器
    visualizer = PerformanceVisualizer()
    
    # 1. 带宽分析图
    print("\n1. 生成带宽分析图...")
    fig1 = visualizer.plot_bandwidth_analysis(network_metrics, 'bandwidth_analysis.png')
    print("   保存为: bandwidth_analysis.png")
    
    # 2. 延迟分析图
    print("\n2. 生成延迟分析图...")
    fig2 = visualizer.plot_latency_analysis(network_metrics, 'latency_analysis.png')
    print("   保存为: latency_analysis.png")
    
    # 3. 吞吐量分析图
    print("\n3. 生成吞吐量分析图...")
    fig3 = visualizer.plot_throughput_analysis(network_metrics, 'throughput_analysis.png')
    print("   保存为: throughput_analysis.png")
    
    # 4. 热点分析图
    print("\n4. 生成热点分析图...")
    fig4 = visualizer.plot_hotspot_analysis(network_metrics, 'hotspot_analysis.png')
    print("   保存为: hotspot_analysis.png")
    
    # 5. 性能仪表板
    print("\n5. 生成性能仪表板...")
    fig5 = visualizer.create_performance_dashboard(network_metrics, 'performance_dashboard.png')
    print("   保存为: performance_dashboard.png")
    
    # 6. 网络流量可视化
    print("\n6. 生成网络流量可视化...")
    flow_visualizer = NetworkFlowVisualizer(layout='grid')
    topology_info = {'rows': 4, 'cols': 4}
    fig6 = flow_visualizer.visualize_network_flow(requests, topology_info, 'network_flow.png')
    print("   保存为: network_flow.png")
    
    # 显示所有图表
    print("\n7. 显示图表...")
    plt.show()


def demonstrate_export(processor):
    """演示导出功能"""
    print("\n" + "=" * 60)
    print("NoC 结果导出演示")
    print("=" * 60)
    
    # 1. 导出JSON
    print("\n1. 导出JSON格式...")
    json_result = processor.export_results('json', 'noc_results.json')
    print(f"   导出结果: {json_result}")
    
    # 2. 导出CSV
    print("\n2. 导出CSV格式...")
    csv_result = processor.export_results('csv', 'noc_results.csv')
    print(f"   导出结果: {csv_result}")
    
    # 3. 导出Excel
    print("\n3. 导出Excel格式...")
    try:
        excel_result = processor.export_results('excel', 'noc_results.xlsx')
        print(f"   导出结果: {excel_result}")
    except ImportError:
        print("   需要安装 openpyxl 来支持Excel导出")
        print("   运行: pip install openpyxl")


def demonstrate_advanced_analysis(processor):
    """演示高级分析功能"""
    print("\n" + "=" * 60)
    print("NoC 高级分析演示")
    print("=" * 60)
    
    requests = processor.performance_metrics.requests
    
    # 1. 按跳数分析延迟
    print("\n1. 按跳数分析延迟...")
    latency_by_distance = processor.latency_analyzer.analyze_latency_by_distance(requests)
    
    print("   跳数 -> 平均延迟:")
    for hop_count, latency_metrics in sorted(latency_by_distance.items()):
        print(f"     {hop_count} 跳: {latency_metrics.avg_total_latency:.1f} ns")
    
    # 2. 按节点对分析延迟
    print("\n2. 节点对延迟分析（显示前5个）...")
    latency_by_node_pair = processor.latency_analyzer.analyze_latency_by_node_pair(requests)
    
    # 按平均延迟排序
    sorted_pairs = sorted(latency_by_node_pair.items(), 
                         key=lambda x: x[1].avg_total_latency, reverse=True)
    
    print("   节点对 -> 平均延迟:")
    for (src, dst), latency_metrics in sorted_pairs[:5]:
        print(f"     {src} -> {dst}: {latency_metrics.avg_total_latency:.1f} ns")
    
    # 3. 热点分析
    print("\n3. 热点节点分析...")
    hotspots = processor.hotspot_analyzer.analyze_hotspots(requests)
    
    print("   热点节点:")
    for hotspot in hotspots[:5]:  # 显示前5个
        status = "🔥 热点" if hotspot.is_hotspot else "✅ 正常"
        print(f"     节点 {hotspot.node_id}: {status}")
        print(f"       拥塞比例: {hotspot.congestion_ratio:.2f}")
        print(f"       带宽利用率: {hotspot.bandwidth_utilization:.2f}")
        print(f"       负载均衡: {hotspot.load_balance_ratio:.2f}")
    
    # 4. 工作区间分析
    print("\n4. 工作区间分析...")
    overall_bandwidth = processor.bandwidth_analyzer.calculate_bandwidth_metrics(requests)
    
    print(f"   总工作区间数: {len(overall_bandwidth.working_intervals)}")
    print(f"   总工作时间: {overall_bandwidth.total_working_time} ns")
    print(f"   利用率: {overall_bandwidth.utilization_ratio:.2%}")
    print(f"   有效带宽: {overall_bandwidth.effective_bandwidth_gbps:.3f} GB/s")


def print_feature_summary():
    """打印功能总结"""
    print("\n" + "=" * 60)
    print("NoC 分析模块功能总结")
    print("=" * 60)
    
    features = [
        "📊 性能指标分析",
        "   • 带宽分析 (平均/峰值/有效带宽)",
        "   • 延迟分析 (平均/P50/P95/P99延迟)",
        "   • 吞吐量分析 (平均/峰值/持续吞吐量)",
        "   • 工作区间分析",
        "",
        "🔥 热点分析",
        "   • 节点流量统计",
        "   • 拥塞检测",
        "   • 负载均衡分析",
        "   • 热点节点识别",
        "",
        "📈 可视化功能",
        "   • 带宽分析图表",
        "   • 延迟分布图",
        "   • 吞吐量时间序列",
        "   • 热点分析图",
        "   • 性能仪表板",
        "   • 网络流量图",
        "",
        "💾 导出功能",
        "   • JSON 格式导出",
        "   • CSV 格式导出",
        "   • Excel 格式导出",
        "",
        "🔧 高级分析",
        "   • 按跳数分析延迟",
        "   • 按节点对分析",
        "   • 时间窗口分析",
        "   • 网络利用率分析",
        "",
        "📋 与原版的改进",
        "   • 模块化设计，易于扩展",
        "   • 类型安全的数据结构",
        "   • 现代化的可视化界面",
        "   • 支持多种导出格式",
        "   • 详细的性能指标计算",
        "   • 智能的热点检测算法"
    ]
    
    for feature in features:
        print(feature)


def main():
    """主函数"""
    print("NoC 性能分析和可视化演示")
    print("基于新的分析框架，提供全面的性能分析能力")
    
    try:
        # 执行演示
        processor, network_metrics, requests = demonstrate_analysis()
        demonstrate_visualization(processor, network_metrics, requests)
        demonstrate_export(processor)
        demonstrate_advanced_analysis(processor)
        print_feature_summary()
        
        print("\n" + "=" * 60)
        print("演示完成！")
        print("=" * 60)
        print("生成的文件:")
        print("  • bandwidth_analysis.png - 带宽分析图")
        print("  • latency_analysis.png - 延迟分析图")
        print("  • throughput_analysis.png - 吞吐量分析图")
        print("  • hotspot_analysis.png - 热点分析图")
        print("  • performance_dashboard.png - 性能仪表板")
        print("  • network_flow.png - 网络流量图")
        print("  • noc_results.json - JSON格式结果")
        print("  • noc_results.csv - CSV格式结果")
        print("  • noc_results.xlsx - Excel格式结果")
        print("\n要在实际项目中使用，请参考:")
        print("  from src.noc.analysis import ResultProcessor")
        print("  processor = ResultProcessor(config)")
        print("  processor.collect_simulation_data(your_simulation_model)")
        print("  network_metrics = processor.analyze_performance()")
        
    except Exception as e:
        print(f"\n❌ 演示过程中出现错误: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()