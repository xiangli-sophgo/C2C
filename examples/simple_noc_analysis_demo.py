#!/usr/bin/env python3
"""
简化的NoC分析演示
只生成最核心的分析结果和文件
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.noc.analysis.simple_result_processor import SimpleResultProcessor, create_simple_analysis_session
from src.noc.analysis.simple_visualizer import SimplePerformanceVisualizer
from src.noc.analysis.performance_metrics import RequestMetrics, RequestType
import numpy as np
import matplotlib.pyplot as plt
import logging


def create_sample_requests(num_requests: int = 1000) -> list:
    """创建示例请求数据"""
    requests = []
    
    np.random.seed(42)  # 固定随机种子
    
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


def main():
    """主演示函数"""
    print("=" * 60)
    print("简化 NoC 性能分析演示")
    print("=" * 60)
    
    # 设置日志
    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
    
    # 仿真配置
    config = {
        'topology_type': 'mesh_4x4',
        'node_count': 16,
        'gap_threshold_ns': 200,
        'network_frequency': 2.0,  # GHz
        'routing_algorithm': 'xy_routing',
        'packet_size': 128,  # bytes
        'num_requests': 1000
    }
    
    # 使用简化的上下文管理器
    with create_simple_analysis_session(
        model_name="crossring",
        topology_type="4x4_mesh",
        config=config,
        session_name="simple_demo"
    ) as output_manager:
        
        print(f"会话目录: {output_manager.get_session_dir()}")
        
        # 1. 创建示例数据
        print("\n1. 生成仿真数据...")
        requests = create_sample_requests(config['num_requests'])
        print(f"   生成了 {len(requests)} 个请求")
        
        # 2. 执行分析
        print("\n2. 执行性能分析...")
        processor = SimpleResultProcessor(config, output_manager)
        processor.add_requests(requests)
        
        # 执行分析
        performance_data = processor.analyze_performance()
        
        # 3. 显示结果摘要
        print("\n3. 性能分析结果:")
        print("-" * 40)
        print(f"   总带宽: {performance_data['bandwidth_gbps']:.3f} GB/s")
        print(f"   平均延迟: {performance_data['latency_ns']:.1f} ns")
        print(f"   P95延迟: {performance_data['p95_latency_ns']:.1f} ns")
        print(f"   吞吐量: {performance_data['throughput_rps']:.0f} req/s")
        print(f"   网络利用率: {performance_data['network_utilization']:.1%}")
        print(f"   热点节点数: {performance_data['hotspot_count']}")
        print(f"   平均跳数: {performance_data['avg_hop_count']:.1f}")
        
        # 4. 生成可视化
        print("\n4. 生成性能仪表板...")
        visualizer = SimplePerformanceVisualizer(output_manager)
        dashboard_fig = visualizer.create_performance_dashboard(performance_data, requests)
        print("   保存性能仪表板: dashboard.png")
        
        # 5. 导出详细数据
        print("\n5. 导出数据...")
        try:
            excel_file = processor.export_detailed_data("detailed_data")
            if excel_file:
                print("   保存详细数据: detailed_data.xlsx")
        except Exception as e:
            print(f"   Excel导出失败: {e}")
        
        # 6. 显示最终的文件结构
        print(f"\n6. 生成的文件:")
        results_dir = output_manager.get_results_dir()
        if results_dir and results_dir.exists():
            print(f"   📁 {output_manager.get_session_dir().name}/")
            print(f"      📄 config.json")
            print(f"      📄 README.md") 
            print(f"      📁 results/")
            for file in results_dir.iterdir():
                if file.is_file():
                    print(f"         📄 {file.name}")
        
        # 7. 显示简化的文件结构说明
        print(f"\n7. 文件说明:")
        print(f"   config.json - 仿真配置和基本信息")
        print(f"   README.md - 性能分析报告")
        print(f"   results/performance_summary.json - 详细性能数据")
        print(f"   results/dashboard.png - 综合性能仪表板")
        print(f"   results/detailed_data.xlsx - 完整请求数据")
        
        # 关闭图形
        plt.close('all')
        
        print("\n" + "=" * 60)
        print("简化分析完成!")
        print("=" * 60)
        print("优势:")
        print("✅ 只生成5个文件 (vs 之前的20+个文件)")
        print("✅ 一个仪表板包含所有关键信息")
        print("✅ 清晰的文件组织结构")
        print("✅ 完整的性能分析结果")
        print("\n生成的文件位置:")
        print(f"📁 {output_manager.get_session_dir()}")


def compare_with_full_version():
    """与完整版本的对比"""
    print("\n" + "=" * 60)
    print("简化版 vs 完整版对比")
    print("=" * 60)
    
    comparison = """
简化版特点:
✅ 文件数量: 5个核心文件
✅ 目录结构: 2层 (session/results)
✅ 可视化: 1个综合仪表板
✅ 配置: 1个JSON文件
✅ 报告: 1个Markdown报告

完整版特点:
📊 文件数量: 20+个文件
📊 目录结构: 3层 (session/category/files)
📊 可视化: 6个独立图表
📊 配置: 多个YAML文件
📊 报告: 多个专业报告

适用场景:
简化版 → 快速分析、日常使用、演示
完整版 → 深度分析、科研使用、详细记录
"""
    print(comparison)


if __name__ == "__main__":
    try:
        main()
        compare_with_full_version()
        
    except Exception as e:
        print(f"\n❌ 演示过程中出现错误: {e}")
        import traceback
        traceback.print_exc()