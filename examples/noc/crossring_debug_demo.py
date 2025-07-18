#!/usr/bin/env python3
"""
极简版CrossRing调试 - 智能打印控制
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.noc.crossring.model import CrossRingModel
from src.noc.crossring.config import CrossRingConfig
import logging

# 禁用所有日志
logging.disable(logging.CRITICAL)


def track_request_smart():
    """使用新的全局调试控制跟踪请求"""
    # 创建2x2配置 - 使用小规模拓扑减少输出
    config = CrossRingConfig.create_custom_config(num_row=3, num_col=3)

    traffic_file = Path(__file__).parent.parent.parent / "traffic_data" / "sample_traffic.txt"
    if not traffic_file.exists():
        print(f"❌ Traffic文件不存在: {traffic_file}")
        return False

    # 创建模型时重定向详细输出
    import io
    import contextlib

    with contextlib.redirect_stdout(io.StringIO()):
        model = CrossRingModel(config, traffic_file_path=str(traffic_file))

        # IP接口应该由模型自动创建并连接

    # 2. 设置TrafficScheduler并注入流量
    traffic_filename = traffic_file.name
    model.setup_traffic_scheduler([[traffic_filename]], str(traffic_file.parent))

    # 检查注入结果
    traffic_status = model.get_traffic_status()
    print(f"🔍 TrafficScheduler状态: {traffic_status}")

    if not traffic_status.get("has_pending", False):
        print("❌ 流量注入失败")
        # 尝试调试TrafficScheduler
        if hasattr(model, "traffic_scheduler") and model.traffic_scheduler:
            print(f"  - 并行链数量: {len(model.traffic_scheduler.parallel_chains)}")
            for i, chain in enumerate(model.traffic_scheduler.parallel_chains):
                print(f"  - 链{i}: {chain.traffic_files}, has_pending: {chain.has_pending_requests()}")
                if hasattr(chain, "active_traffic"):
                    print(f"    active_traffic: {chain.active_traffic}")
        return False

    print(f"✅ 成功设置TrafficScheduler，准备处理请求")

    # 跟踪的packet_id
    packet_id = 1

    # 启用全局调试模式，跟踪实际的packet_id
    # enable_debug方法签名: enable_debug(level: int = 1, trace_packets: List[str] = None, sleep_time: float = 0.0)
    model.enable_debug(level=2, trace_packets=[str(packet_id)], sleep_time=0.1)

    print("-" * 60)

    # 运行仿真 - 调试信息由模型的全局调试控制自动处理
    for cycle in range(200):
        model.step()

        # 检查是否完成
        if packet_id in model.request_tracker.completed_requests:
            print("-" * 60)
            print("请求完成!")
            break

    # 禁用调试模式
    model.disable_debug()

    # 执行结果分析
    print("-" * 60)
    print("📊 开始结果分析...")

    # 导入结果分析器
    from src.noc.analysis.result_analyzer import ResultAnalyzer

    # 创建分析器实例
    analyzer = ResultAnalyzer()

    # 执行分析
    results = {"simulation_time": model.cycle, "total_requests": len(model.request_tracker.completed_requests), "topology": "CrossRing"}

    analysis = analyzer.analyze_noc_results(request_tracker=model.request_tracker, config=model.config, model=model, results=results, enable_visualization=True, save_results=True)

    # 显示分析结果摘要
    print("\n📈 分析结果摘要:")
    print("=" * 60)

    if "带宽指标" in analysis:
        bw_metrics = analysis["带宽指标"]
        print(f"平均带宽: {bw_metrics.get('平均带宽', 'N/A')}")
        print(f"峰值带宽: {bw_metrics.get('峰值带宽', 'N/A')}")
        print(f"总传输量: {bw_metrics.get('总传输量', 'N/A')}")

    if "延迟指标" in analysis:
        lat_metrics = analysis["延迟指标"]
        print(f"平均延迟: {lat_metrics.get('平均延迟', 'N/A')}")
        print(f"最大延迟: {lat_metrics.get('最大延迟', 'N/A')}")
        print(f"最小延迟: {lat_metrics.get('最小延迟', 'N/A')}")

    if "输出文件" in analysis:
        output_info = analysis["输出文件"]
        print(f"分析结果已保存到: {output_info.get('分析结果文件', 'N/A')}")

    print("=" * 60)
    print("✅ 结果分析完成！")


if __name__ == "__main__":
    track_request_smart()
