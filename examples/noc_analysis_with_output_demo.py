#!/usr/bin/env python3
"""
NoC 分析演示 - 带完整输出管理
展示如何使用新的分析、可视化和输出管理功能
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.noc.analysis import ResultProcessor, PerformanceVisualizer, NetworkFlowVisualizer
from src.noc.analysis.performance_metrics import RequestMetrics, RequestType
from src.noc.analysis.output_manager import OutputManager, SimulationContext
import numpy as np
import matplotlib.pyplot as plt
import logging
import time


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


def demonstrate_complete_workflow():
    """演示完整的工作流程"""
    
    # 仿真配置
    simulation_config = {
        'topology_type': 'mesh_4x4',
        'node_count': 16,
        'gap_threshold_ns': 200,
        'window_size_ns': 1000,
        'congestion_threshold': 0.8,
        'network_frequency': 2.0,  # GHz
        'buffer_size': 64,
        'routing_algorithm': 'xy_routing',
        'packet_size': 128,  # bytes
        'num_requests': 1000
    }
    
    # 使用上下文管理器创建输出管理
    with SimulationContext(
        model_name="crossring",
        topology_type="4x4_mesh", 
        config=simulation_config,
        session_name="demo_run"
    ) as output_manager:
        
        print("=" * 60)
        print("NoC 完整分析演示 - 带输出管理")
        print("=" * 60)
        print(f"会话目录: {output_manager.get_session_dir()}")
        
        # 1. 创建示例数据
        print("\n1. 创建示例请求数据...")
        requests = create_sample_requests(simulation_config['num_requests'])
        print(f"   创建了 {len(requests)} 个请求")
        
        # 保存原始数据
        raw_data = {
            'requests': [
                {
                    'packet_id': req.packet_id,
                    'request_type': req.request_type.value,
                    'source_node': req.source_node,
                    'dest_node': req.dest_node,
                    'start_time': req.start_time,
                    'end_time': req.end_time,
                    'total_bytes': req.total_bytes,
                    'hop_count': req.hop_count
                } for req in requests
            ]
        }
        output_manager.save_data(raw_data, "raw_simulation_data", "json")
        
        # 2. 初始化分析器
        print("\n2. 初始化分析器...")
        processor = ResultProcessor(simulation_config, output_manager)
        
        # 添加请求数据
        for req in requests:
            processor.performance_metrics.add_request(req)
        
        # 3. 执行性能分析
        print("\n3. 执行性能分析...")
        network_metrics = processor.analyze_performance()
        
        # 保存分析结果
        analysis_results = {
            'network_metrics': {
                'topology_type': network_metrics.topology_type,
                'node_count': network_metrics.node_count,
                'simulation_duration': network_metrics.simulation_duration,
                'network_utilization': network_metrics.network_utilization,
                'average_hop_count': network_metrics.average_hop_count,
                'overall_bandwidth_gbps': network_metrics.overall_bandwidth_gbps,
                'overall_latency_ns': network_metrics.overall_latency_ns,
                'overall_throughput_rps': network_metrics.overall_throughput_rps
            }
        }
        output_manager.save_analysis_result(analysis_results, "detailed_analysis_results")
        
        # 4. 生成可视化
        print("\n4. 生成可视化图表...")
        visualizer = PerformanceVisualizer(output_manager=output_manager)
        
        # 带宽分析图
        fig1 = visualizer.plot_bandwidth_analysis(network_metrics, "bandwidth_analysis")
        print("   保存带宽分析图")
        
        # 延迟分析图  
        fig2 = visualizer.plot_latency_analysis(network_metrics, "latency_analysis")
        print("   保存延迟分析图")
        
        # 吞吐量分析图
        fig3 = visualizer.plot_throughput_analysis(network_metrics, "throughput_analysis")
        print("   保存吞吐量分析图")
        
        # 热点分析图
        fig4 = visualizer.plot_hotspot_analysis(network_metrics, "hotspot_analysis")
        print("   保存热点分析图")
        
        # 性能仪表板
        fig5 = visualizer.create_performance_dashboard(network_metrics, "performance_dashboard")
        print("   保存性能仪表板")
        
        # 网络流量图
        print("\n5. 生成网络流量可视化...")
        flow_visualizer = NetworkFlowVisualizer(layout='grid')
        topology_info = {'rows': 4, 'cols': 4}
        fig6 = flow_visualizer.visualize_network_flow(requests, topology_info, "network_flow")
        if output_manager:
            output_manager.save_figure(fig6, "network_flow", 'png')
        print("   保存网络流量图")
        
        # 6. 导出数据
        print("\n6. 导出分析数据...")
        processor.export_results('json', 'performance_summary')
        processor.export_results('csv', 'request_details')
        try:
            processor.export_results('excel', 'complete_analysis')
        except ImportError:
            print("   Excel导出需要安装 openpyxl")
        
        # 7. 生成日志报告
        print("\n7. 生成日志和报告...")
        
        # 保存详细日志
        log_content = f"""
仿真运行日志
===========

运行时间: {time.strftime('%Y-%m-%d %H:%M:%S')}
配置参数: {simulation_config}

分析结果摘要:
- 总请求数: {len(requests)}
- 读请求: {len([r for r in requests if r.request_type == RequestType.READ])}
- 写请求: {len([r for r in requests if r.request_type == RequestType.WRITE])}
- 平均延迟: {network_metrics.overall_latency_ns:.2f} ns
- 总带宽: {network_metrics.overall_bandwidth_gbps:.3f} GB/s
- 网络利用率: {network_metrics.network_utilization:.1%}

热点节点数: {len(network_metrics.hotspot_nodes)}
拥塞节点数: {network_metrics.congestion_count}
"""
        output_manager.save_log(log_content, "simulation_run")
        
        # 生成详细性能报告
        summary = processor.get_performance_summary()
        report_content = f"""
# NoC 性能分析报告

## 仿真配置
- **拓扑类型**: {simulation_config['topology_type']}
- **节点数量**: {simulation_config['node_count']}
- **请求数量**: {simulation_config['num_requests']}
- **路由算法**: {simulation_config['routing_algorithm']}
- **包大小**: {simulation_config['packet_size']} bytes

## 性能指标

### 带宽分析
- **总带宽**: {network_metrics.overall_bandwidth_gbps:.3f} GB/s
- **网络利用率**: {network_metrics.network_utilization:.1%}

### 延迟分析  
- **平均延迟**: {network_metrics.overall_latency_ns:.2f} ns
- **平均跳数**: {network_metrics.average_hop_count:.1f}

### 吞吐量分析
- **总吞吐量**: {network_metrics.overall_throughput_rps:.0f} req/s

### 热点分析
- **热点节点**: {len([h for h in network_metrics.hotspot_nodes if h.is_hotspot])}
- **拥塞节点**: {network_metrics.congestion_count}

## 详细统计
"""
        
        for key, value in summary.items():
            if isinstance(value, float):
                report_content += f"- **{key}**: {value:.3f}\n"
            else:
                report_content += f"- **{key}**: {value}\n"
        
        report_content += f"""

## 生成的文件
此分析会话包含以下文件:
- 配置文件: config/main_config.yaml
- 原始数据: data/raw_simulation_data.json
- 分析结果: analysis/detailed_analysis_results.json
- 图表文件: figures/ 目录中的所有PNG文件
- 详细数据: data/ 目录中的JSON/CSV文件

## 重现实验
使用 config/main_config.yaml 中的配置可以重现此次实验。
"""
        
        output_manager.save_report(report_content, "performance_report", "md")
        
        # 8. 显示结果摘要
        print("\n8. 分析结果摘要:")
        print("-" * 40)
        for key, value in summary.items():
            if isinstance(value, float):
                print(f"   {key}: {value:.3f}")
            else:
                print(f"   {key}: {value}")
        
        # 9. 列出生成的文件
        print(f"\n9. 文件保存位置:")
        print(f"   会话目录: {output_manager.get_session_dir()}")
        print(f"   配置文件: {output_manager.get_subdirectory('config')}")
        print(f"   图表文件: {output_manager.get_subdirectory('figures')}")
        print(f"   数据文件: {output_manager.get_subdirectory('data')}")
        print(f"   分析结果: {output_manager.get_subdirectory('analysis')}")
        print(f"   报告文件: {output_manager.get_subdirectory('reports')}")
        print(f"   日志文件: {output_manager.get_subdirectory('logs')}")
        
        # 关闭所有图形
        plt.close('all')


def demonstrate_session_management():
    """演示会话管理功能"""
    print("\n" + "=" * 60)
    print("会话管理演示")
    print("=" * 60)
    
    # 创建输出管理器
    output_manager = OutputManager()
    
    # 列出历史会话
    print("\n历史会话:")
    sessions = output_manager.list_sessions()
    
    if sessions:
        for i, session in enumerate(sessions[:5], 1):  # 显示最近5个
            print(f"{i}. {session['session_id']}")
            print(f"   模型: {session['model_name']}")
            print(f"   拓扑: {session['topology_type']}")
            print(f"   时间: {session['created_time']}")
            print()
    else:
        print("   没有找到历史会话")
    
    # 清理旧会话 (保留最近3个)
    print("清理旧会话 (保留最近3个)...")
    output_manager.cleanup_old_sessions(keep_count=3)
    
    updated_sessions = output_manager.list_sessions()
    print(f"清理后剩余会话数: {len(updated_sessions)}")


def print_usage_guide():
    """打印使用指南"""
    print("\n" + "=" * 60)
    print("NoC 分析框架使用指南")
    print("=" * 60)
    
    usage_guide = """
## 基本使用方法

### 1. 创建仿真会话
```python
from src.noc.analysis.output_manager import SimulationContext

with SimulationContext(
    model_name="crossring",
    topology_type="4x4_mesh",
    config=your_config
) as output_manager:
    # 进行分析
```

### 2. 性能分析
```python
from src.noc.analysis import ResultProcessor

processor = ResultProcessor(config, output_manager)
processor.collect_simulation_data(simulation_model)
network_metrics = processor.analyze_performance()
```

### 3. 生成可视化
```python
from src.noc.analysis import PerformanceVisualizer

visualizer = PerformanceVisualizer(output_manager=output_manager)
visualizer.plot_bandwidth_analysis(network_metrics)
visualizer.create_performance_dashboard(network_metrics)
```

### 4. 导出结果
```python
processor.export_results('json', 'results')
processor.export_results('excel', 'detailed_analysis')
```

## 输出文件结构

每次仿真会创建一个独立的文件夹:
```
output/
└── crossring_4x4_mesh_demo_run_20231201_143022/
    ├── README.md                 # 会话说明
    ├── session_metadata.json    # 会话元数据
    ├── config/
    │   └── main_config.yaml     # 主配置文件
    ├── logs/
    │   ├── session.log          # 会话日志
    │   └── simulation_run.log   # 仿真运行日志
    ├── figures/
    │   ├── bandwidth_analysis.png
    │   ├── latency_analysis.png
    │   ├── throughput_analysis.png
    │   ├── hotspot_analysis.png
    │   ├── performance_dashboard.png
    │   └── network_flow.png
    ├── data/
    │   ├── raw_simulation_data.json
    │   ├── performance_summary.json
    │   ├── request_details.csv
    │   └── complete_analysis.xlsx
    ├── analysis/
    │   └── detailed_analysis_results.json
    └── reports/
        ├── performance_report.md
        └── session_summary.md
```

## 高级功能

### 1. 会话管理
- 自动创建时间戳命名的会话目录
- 保存完整的配置和元数据
- 支持会话恢复和清理

### 2. 可视化定制
- 支持多种图表类型
- 中文字体支持
- 高分辨率图片输出

### 3. 数据导出
- JSON格式: 结构化数据
- CSV格式: 表格数据
- Excel格式: 多工作表详细数据

### 4. 热点分析
- 自动检测网络热点
- 拥塞分析
- 负载均衡评估
"""
    print(usage_guide)


def main():
    """主函数"""
    print("NoC 性能分析框架 - 完整演示")
    print("包含输出管理、可视化和数据导出功能")
    
    try:
        # 设置日志级别
        logging.basicConfig(level=logging.INFO, 
                          format='%(asctime)s - %(levelname)s - %(message)s')
        
        # 执行完整演示
        demonstrate_complete_workflow()
        
        # 演示会话管理
        demonstrate_session_management()
        
        # 打印使用指南
        print_usage_guide()
        
        print("\n" + "=" * 60)
        print("演示完成！")
        print("=" * 60)
        print("检查 output/ 目录查看生成的文件")
        print("每次运行都会创建一个新的时间戳命名的会话目录")
        
    except Exception as e:
        print(f"\n❌ 演示过程中出现错误: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()