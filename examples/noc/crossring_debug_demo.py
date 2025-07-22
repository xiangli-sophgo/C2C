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
import matplotlib

# 配置日志级别以显示关键调试信息
logging.basicConfig(level=logging.INFO, format="%(levelname)s - %(message)s")
# 显示INFO级别信息以查看注入日志
logging.getLogger("src.noc.crossring").setLevel(logging.INFO)

if sys.platform == "darwin":  # macOS 的系统标识是 'darwin'
    matplotlib.use("macosx")  # 仅在 macOS 上使用该后端


def track_request_smart(output_dir: str = None):
    """使用新的全局调试控制跟踪请求

    Args:
        output_dir: 输出目录，默认为 'output/crossring_results'
    """
    # 设置输出目录
    if output_dir is None:
        output_dir = Path(__file__).parent.parent.parent / "output" / "crossring_results"
    else:
        output_dir = Path(output_dir)

    # 确保输出目录存在
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"📁 输出目录: {output_dir}")

    # 创建2x2配置 - 使用小规模拓扑减少输出
    config = CrossRingConfig.create_custom_config(num_row=3, num_col=3)

    traffic_file = Path(__file__).parent.parent.parent / "traffic_data" / "test1.txt"
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

    # 启用调试模式以显示请求处理过程
    packet_ids = [
        # 6,
    ]
    model.enable_debug(level=1, trace_packets=[str(pid) for pid in packet_ids], sleep_time=0.1)

    print("-" * 60)

    # 运行仿真
    for cycle in range(3000):  # 运行足够的周期完成传输
        model.step()
        # if model.cycle > 100:
        # break

    # 执行结果分析
    print("-" * 60)
    print("📊 开始结果分析...")

    # 执行分析
    completed_requests = len(model.request_tracker.completed_requests) if hasattr(model, "request_tracker") and model.request_tracker else 0
    results = {"simulation_time": model.cycle, "total_requests": completed_requests, "topology": "CrossRing"}

    print(f"📊 仿真统计信息:")
    print(f"  - 仿真周期: {model.cycle}")
    print(f"  - 完成请求数: {completed_requests}")
    if hasattr(model, "request_tracker") and model.request_tracker:
        total_requests = len(model.request_tracker.active_requests) + len(model.request_tracker.completed_requests)
        print(f"  - 总请求数: {total_requests}")
        print(f"  - 正在处理: {len(model.request_tracker.active_requests)}")

    # 确保输出目录是绝对路径字符串
    save_dir_str = str(output_dir.resolve())

    try:
        analysis = model.analyze_simulation_results(results, enable_visualization=True, save_results=True, save_dir=save_dir_str)
    except Exception as e:
        print(f"ERROR: 分析失败: {e}")
        analysis = {}

    # 显示分析结果摘要
    print("\n📈 分析结果摘要:")
    print("=" * 60)

    if "带宽指标" in analysis:
        bw_metrics = analysis["带宽指标"]
        if "总体带宽" in bw_metrics:
            overall_bw = bw_metrics["总体带宽"]
            print(f"平均带宽: {overall_bw.get('非加权带宽_GB/s', 'N/A')} GB/s")
            print(f"加权带宽: {overall_bw.get('加权带宽_GB/s', 'N/A')} GB/s")
            print(f"总传输量: {overall_bw.get('总传输字节数', 'N/A')} 字节")

    if "延迟指标" in analysis:
        lat_metrics = analysis["延迟指标"]
        if "总体延迟" in lat_metrics:
            overall_lat = lat_metrics["总体延迟"]
            print(f"平均延迟: {overall_lat.get('平均延迟_ns', 'N/A')} ns")
            print(f"最大延迟: {overall_lat.get('最大延迟_ns', 'N/A')} ns")
            print(f"最小延迟: {overall_lat.get('最小延迟_ns', 'N/A')} ns")

    if "输出文件" in analysis:
        output_info = analysis["输出文件"]
        print(f"分析结果已保存到: {output_info.get('分析结果文件', 'N/A')}")

    print("=" * 60)
    print("✅ 结果分析完成！")

    # 显示生成的文件
    if "可视化文件" in analysis and analysis["可视化文件"]["生成的图表"]:
        print("\n📊 生成的可视化图表:")
        for i, chart_path in enumerate(analysis["可视化文件"]["生成的图表"], 1):
            chart_name = chart_path.split("/")[-1]
            if "bandwidth_curve" in chart_name:
                print(f"  {i}. 带宽时间曲线图: {chart_name}")
            elif "latency_distribution" in chart_name:
                print(f"  {i}. 延迟分布图: {chart_name}")
            elif "port_bandwidth" in chart_name:
                print(f"  {i}. 端口带宽对比图: {chart_name}")
            elif "traffic_distribution" in chart_name:
                print(f"  {i}. 流量分布图: {chart_name}")
            else:
                print(f"  {i}. {chart_name}")

    print(f"\n📁 所有文件保存在: {output_dir}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="CrossRing NoC调试工具")
    parser.add_argument("-o", "--output", type=str, help="输出目录路径")

    args = parser.parse_args()
    track_request_smart(output_dir=args.output)
