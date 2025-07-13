#!/usr/bin/env python3
"""
简化的CrossRing NoC演示
=====================

最简单的CrossRing仿真演示，只需几行代码：
1. 创建CrossRing模型
2. 从traffic文件注入流量
3. 运行仿真
4. 显示结果

Usage:
    python simple_crossring_demo.py [rows] [cols] [max_cycles]
"""

import sys
import logging
from pathlib import Path

# 添加src路径
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.noc.crossring.model import CrossRingModel
from src.noc.crossring.config import CrossRingConfig


def create_config(rows=2, cols=3, config_name="simple_demo"):
    """创建CrossRing配置"""
    config = CrossRingConfig(num_row=rows, num_col=cols, config_name=config_name)

    # 确保num_nodes正确设置
    config.num_nodes = rows * cols

    # 为所有节点配置IP接口
    all_nodes = list(range(rows * cols))
    config.gdma_send_position_list = all_nodes
    config.ddr_send_position_list = all_nodes
    config.l2m_send_position_list = all_nodes

    return config


def run_crossring_simulation(rows=2, cols=3, max_cycles=10000):
    """运行CrossRing仿真 - 极简版本"""

    print(f"📡 CrossRing仿真开始: {rows}×{cols} 网格, 最大{max_cycles}周期")

    try:
        # 1. 创建配置和模型
        traffic_file = Path(__file__).parent.parent.parent / "traffic_data" / "crossring_traffic.txt"
        if not traffic_file.exists():
            print(f"❌ Traffic文件不存在: {traffic_file}")
            return False

        config = create_config(rows, cols)
        model = CrossRingModel(config, traffic_file_path=str(traffic_file))

        # 2. 注入流量并运行仿真
        injected = model.inject_from_traffic_file(traffic_file_path=str(traffic_file), cycle_accurate=True)  # 使用周期精确模式

        if injected == 0:
            print("❌ 没有成功注入任何请求")
            return False

        print(f"✅ 成功注入 {injected} 个请求")

        # 3. 逐个跟踪请求运行仿真
        # 先运行一小段时间让部分请求注入
        for cycle in range(100):
            model.step()

        # # 选择第一个活跃请求进行详细跟踪
        # if hasattr(model, "request_tracker") and model.request_tracker.active_requests:
        #     first_packet_id = list(model.request_tracker.active_requests.keys())[0]
        #     lifecycle = model.request_tracker.active_requests[first_packet_id]
        #     print(f"\n🔍 开始跟踪第一个请求: {first_packet_id}")
        #     print(f"  源节点: {lifecycle.source} -> 目标节点: {lifecycle.destination}")
        #     print(f"  操作类型: {lifecycle.op_type}, 数据长度: {lifecycle.burst_size}")
        #     print(f"  当前状态: {lifecycle.current_state.value}")

        #     # 启用单个请求的debug跟踪
        #     model.enable_debug([first_packet_id], 0.1)

        #     # 继续运行，跟踪这个请求
        #     for cycle in range(100, max_cycles):
        #         model.step()
        #         if first_packet_id in model.request_tracker.completed_requests:
        #             print(f"\n✅ 请求 {first_packet_id} 已完成，用时 {cycle - lifecycle.created_cycle} 周期")
        #             break
        #         elif cycle % 100 == 0:
        #             print(f"周期 {cycle}: 请求 {first_packet_id} 仍在处理中...")

        #     model.disable_debug()

        # 运行剩余的仿真
        results = model.run_simulation(max_cycles=max_cycles, warmup_cycles=0, stats_start_cycle=0)

        if not results:
            print("❌ 仿真失败")
            return False

        # 4. 分析并显示结果 - 使用新的增强分析功能
        print(f"\n🔬 开始详细性能分析...")
        analysis = model.analyze_simulation_results(results, enable_visualization=True, save_results=True)

        # 打印RequestTracker的详细报告
        if hasattr(model, "request_tracker"):
            print(f"\n🔍 RequestTracker详细信息:")
            print(f"活跃请求数: {len(model.request_tracker.active_requests)}")
            print(f"已完成请求数: {len(model.request_tracker.completed_requests)}")

            if len(model.request_tracker.active_requests) > 0:
                print(f"\n⚠️ 未完成的活跃请求:")
                for packet_id, lifecycle in list(model.request_tracker.active_requests.items())[:10]:  # 只显示前10个
                    print(f"  {packet_id}: {lifecycle.source}->{lifecycle.destination} {lifecycle.op_type}, 状态={lifecycle.current_state.value}")

        print("\n" + "=" * 50)
        print("📊 详细仿真分析结果")
        print("=" * 50)

        # 基础指标
        basic = analysis.get("基础指标", {})
        if basic:
            print(f"✨ 基础指标:")
            for key, value in basic.items():
                print(f"  {key}: {value}")

        # 带宽指标
        bandwidth = analysis.get("带宽指标", {})
        if bandwidth:
            print(f"\n📡 带宽分析:")
            
            overall = bandwidth.get("总体带宽", {})
            if overall:
                print(f"  🔄 总体带宽:")
                for key, value in overall.items():
                    print(f"    {key}: {value}")
            
            read_bw = bandwidth.get("读操作带宽", {})
            if read_bw:
                print(f"  📖 读操作带宽:")
                for key, value in read_bw.items():
                    print(f"    {key}: {value}")
            
            write_bw = bandwidth.get("写操作带宽", {})
            if write_bw:
                print(f"  📝 写操作带宽:")
                for key, value in write_bw.items():
                    print(f"    {key}: {value}")

        # 延迟指标
        latency = analysis.get("延迟指标", {})
        if latency:
            print(f"\n⏱️ 延迟分析:")
            
            overall_latency = latency.get("总体延迟", {})
            if overall_latency:
                print(f"  🔄 总体延迟:")
                for key, value in overall_latency.items():
                    print(f"    {key}: {value}")
            
            read_latency = latency.get("读操作延迟", {})
            if read_latency:
                print(f"  📖 读操作延迟:")
                for key, value in read_latency.items():
                    print(f"    {key}: {value}")
            
            write_latency = latency.get("写操作延迟", {})
            if write_latency:
                print(f"  📝 写操作延迟:")
                for key, value in write_latency.items():
                    print(f"    {key}: {value}")

        # 端口带宽分析
        port_analysis = analysis.get("端口带宽分析", {})
        if port_analysis:
            print(f"\n🚪 端口带宽分析:")
            for port_id, metrics in port_analysis.items():
                print(f"  {port_id}:")
                for key, value in metrics.items():
                    print(f"    {key}: {value}")

        # 工作区间分析
        working_intervals = analysis.get("工作区间分析", {})
        if working_intervals:
            print(f"\n⚡ 工作区间分析:")
            for key, value in working_intervals.items():
                print(f"  {key}: {value}")

        # 可视化和输出文件信息
        viz_files = analysis.get("可视化文件", {})
        if viz_files:
            print(f"\n📈 可视化文件:")
            for key, value in viz_files.items():
                print(f"  {key}: {value}")

        output_files = analysis.get("输出文件", {})
        if output_files:
            print(f"\n💾 输出文件:")
            for key, value in output_files.items():
                print(f"  {key}: {value}")

        print("\n✅ CrossRing仿真分析完成！")
        print("=" * 50)
        return True

    except Exception as e:
        print(f"❌ 仿真异常: {e}")
        import traceback

        traceback.print_exc()
        return False
    finally:
        if "model" in locals():
            model.cleanup()


def main():
    """主函数"""
    # 解析命令行参数
    rows = int(sys.argv[1]) if len(sys.argv) > 1 else 2
    cols = int(sys.argv[2]) if len(sys.argv) > 2 else 3
    max_cycles = int(sys.argv[3]) if len(sys.argv) > 3 else 1000
    max_requests = int(sys.argv[4]) if len(sys.argv) > 4 else 10

    print("=" * 60)
    print("🚀 CrossRing NoC 仿真演示")
    print("=" * 60)
    print("只需几行代码即可完成完整的NoC仿真！")
    print()

    # 核心代码示例
    print("💡 核心代码示例:")
    print("```python")
    print("config = create_config(rows, cols)")
    print("model = CrossRingModel(config, traffic_file_path)")
    print("model.inject_from_traffic_file(traffic_file_path)")
    print("results = model.run_simulation(max_cycles)")
    print("analysis = model.analyze_simulation_results(results)")
    print("```")
    print()

    # 运行仿真
    success = run_crossring_simulation(rows, cols, max_cycles)

    if success:
        print("\n✅ 演示成功完成！")
        print("\n📋 演示功能:")
        print("- ✅ 参数化配置 (支持命令行参数)")
        print("- ✅ 优化的IP接口创建")
        print("- ✅ 周期精确的流量注入")
        print("- ✅ 完整的仿真执行")
        print("- ✅ 自动化结果分析")
        print(f"\n🎯 使用方法: python {Path(__file__).name} [行数] [列数] [最大周期] [最大请求]")
        print(f"📝 当前参数: {rows}×{cols} 网格, {max_cycles}周期, {max_requests}请求")
        return 0
    else:
        print("\n❌ 演示失败")
        return 1


if __name__ == "__main__":
    sys.exit(main())
