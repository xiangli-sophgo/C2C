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

# 配置日志级别以显示关键调试信息
logging.basicConfig(level=logging.ERROR, format="%(levelname)s - %(message)s")
# 只显示错误和重要信息
logging.getLogger("src.noc.crossring").setLevel(logging.ERROR)


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

    # 跟踪的packet_id - 第6和第7个请求
    packet_ids = [6, 7]  # packet_id从0开始，所以5是第6个，6是第7个

    # 启用全局调试模式，跟踪实际的packet_id
    # enable_debug方法签名: enable_debug(level: int = 1, trace_packets: List[str] = None, sleep_time: float = 0.0)
    model.enable_debug(level=2, trace_packets=[str(pid) for pid in packet_ids], sleep_time=0.3)

    # 添加自定义调试：在每个周期检查节点3的inject_direction_fifos状态
    original_step = model.step

    def debug_step():
        original_step()
        # 只在关键周期显示调试信息
        if model.cycle < 40 or model.cycle > 60:
            return
        # 检查节点3的IQ_TR状态
        if hasattr(model, "nodes") and 3 in model.nodes:
            node3 = model.nodes[3]
            if hasattr(node3, "inject_queue") and hasattr(node3.inject_queue, "inject_direction_fifos"):
                req_tr_fifo = node3.inject_queue.inject_direction_fifos["req"]["TR"]
                # 总是显示FIFO状态，不管是否为空
                if hasattr(req_tr_fifo, "internal_queue"):
                    queue_contents = []
                    for i, flit in enumerate(req_tr_fifo.internal_queue):
                        packet_id = getattr(flit, "packet_id", "unknown")
                        queue_contents.append(f"pos{i}:pkt{packet_id}")

                    output_info = "None"
                    if hasattr(req_tr_fifo, "output_register") and req_tr_fifo.output_register:
                        output_packet_id = getattr(req_tr_fifo.output_register, "packet_id", "unknown")
                        output_info = f"pkt{output_packet_id}"

                    print(f"🔍 周期{model.cycle}: N3.IQ_TR内容=[{','.join(queue_contents)}], 输出={output_info}, valid={getattr(req_tr_fifo, 'output_valid', False)}")

    model.step = debug_step

    print("-" * 60)

    # 运行仿真
    for cycle in range(200):  # 减少运行周期以便观察
        model.step()

    # 执行结果分析
    print("-" * 60)
    print("📊 开始结果分析...")

    # 执行分析
    results = {"simulation_time": model.cycle, "total_requests": len(model.request_tracker.completed_requests), "topology": "CrossRing"}

    analysis = analysis = model.analyze_simulation_results(results, enable_visualization=True, save_results=True)

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
