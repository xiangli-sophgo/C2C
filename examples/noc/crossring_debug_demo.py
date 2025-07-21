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

    # 跟踪的packet_id
    packet_id = 1

    # 启用全局调试模式，跟踪实际的packet_id
    # enable_debug方法签名: enable_debug(level: int = 1, trace_packets: List[str] = None, sleep_time: float = 0.0)
    model.enable_debug(level=2, trace_packets=[str(packet_id)], sleep_time=0.1)

    print("-" * 60)

    # 运行仿真 - 调试信息由模型的全局调试控制自动处理
    packet_found_in_iq_tr = False
    for cycle in range(200):  # 减少运行周期以便观察
        model.step()

        # 检查packet_id=6是否在active_requests，如果是，打印详细状态
        if packet_id in model.request_tracker.active_requests and cycle == 50:
            request_info = model.request_tracker.active_requests[packet_id]
            position_str = str(request_info.position) if hasattr(request_info, "position") else "no_position"
            print(f"\n🔍 周期{cycle}: packet_id={packet_id} 的position是: {position_str}")

            # 获取N3节点并打印FIFO状态
            node_3 = model.nodes.get(3)
            if node_3:
                print(f"📌 节点3的inject_direction_fifos状态：")
                for channel in ["req", "rsp", "data"]:
                    for direction in ["TR", "TL", "TU", "TD", "EQ"]:
                        fifo = node_3.inject_direction_fifos[channel][direction]
                        if fifo.valid_signal() or len(fifo) > 0:
                            flit = fifo.peek_output()
                            flit_info = f"flit_id={flit.packet_id}" if flit else "no_flit"
                            print(f"  - {channel}.{direction}: len={len(fifo)}, valid={fifo.valid_signal()}, ready={fifo.ready_signal()}, {flit_info}")

                print(f"📌 节点3的CrossPoint状态：")
                h_crosspoint = node_3.horizontal_crosspoint
                print(f"  - 水平CrossPoint管理方向: {h_crosspoint.managed_directions}")

                # 检查TR方向的slice状态
                if "TR" in h_crosspoint.slices:
                    departure_slice = h_crosspoint.slices["TR"]["departure"]
                    if departure_slice:
                        try:
                            current_slot = departure_slice.peek_current_slot("req")
                            print(f"  - TR departure slice current_slot: {current_slot}")
                            print(f"  - can_inject_flit(TR, req): {h_crosspoint.can_inject_flit('TR', 'req')}")
                        except Exception as e:
                            print(f"  - TR departure slice peek_current_slot错误: {e}")
                    else:
                        print(f"  - TR departure slice: None")

                print()

        # 检查data通道卡在IQ_TL的问题
        if cycle == 80:  # 检查data包卡住的时候
            print(f"\n🔍 周期{cycle}: 检查data包卡在IQ_TL的问题")

            # 获取N4节点并打印data通道FIFO状态
            node_4 = model.nodes.get(4)
            if node_4:
                print(f"📌 节点4的data通道inject_direction_fifos状态：")
                for direction in ["TR", "TL", "TU", "TD", "EQ"]:
                    fifo = node_4.inject_direction_fifos["data"][direction]
                    if fifo.valid_signal() or len(fifo) > 0:
                        flit = fifo.peek_output()
                        flit_info = f"flit_id={flit.packet_id}" if flit else "no_flit"
                        print(f"  - data.{direction}: len={len(fifo)}, valid={fifo.valid_signal()}, ready={fifo.ready_signal()}, {flit_info}")

                print(f"📌 节点4的CrossPoint状态（水平）：")
                h_crosspoint = node_4.horizontal_crosspoint
                print(f"  - 水平CrossPoint管理方向: {h_crosspoint.managed_directions}")

                # 检查TL方向的slice状态（data通道）
                if "TL" in h_crosspoint.slices:
                    departure_slice = h_crosspoint.slices["TL"]["departure"]
                    if departure_slice:
                        try:
                            current_slot = departure_slice.peek_current_slot("data")
                            print(f"  - TL departure slice current_slot (data): {current_slot}")
                            print(f"  - can_inject_flit(TL, data): {h_crosspoint.can_inject_flit('TL', 'data')}")
                        except Exception as e:
                            print(f"  - TL departure slice peek_current_slot错误: {e}")
                    else:
                        print(f"  - TL departure slice: None")

                print()

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
    # from src.noc.analysis.result_analyzer import ResultAnalyzer

    # # 创建分析器实例
    # analyzer = ResultAnalyzer()

    # 执行分析
    results = {"simulation_time": model.cycle, "total_requests": len(model.request_tracker.completed_requests), "topology": "CrossRing"}

    analysis = analysis = model.analyze_simulation_results(results)

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
