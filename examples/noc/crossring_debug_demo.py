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
    config = CrossRingConfig.create_custom_config(num_row=2, num_col=2)

    traffic_file = Path(__file__).parent.parent.parent / "traffic_data" / "sample_traffic.txt"
    if not traffic_file.exists():
        print(f"❌ Traffic文件不存在: {traffic_file}")
        return False

    # 创建模型时重定向详细输出
    import io
    import contextlib

    with contextlib.redirect_stdout(io.StringIO()):
        model = CrossRingModel(config, traffic_file_path=str(traffic_file))
        
        # 手动连接IP接口（避免initialize_network的问题）
        if hasattr(model, 'crossring_nodes') and 0 in model.crossring_nodes and 1 in model.crossring_nodes:
            model.crossring_nodes[0].connect_ip("gdma_0_node0")
            model.crossring_nodes[1].connect_ip("ddr_1_node1")

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

    # 在cycle-accurate模式下，packet_id从pending_file_requests获取
    # 运行几个周期让请求被实际注入
    print("⏳ 运行仿真等待请求注入...")
    for i in range(10):
        print(f"  周期{model.cycle + 1}: 开始step...")
        model.step()
        print(f"  周期{model.cycle}: step完成, active_requests: {len(model.request_tracker.active_requests)}")
        if model.request_tracker.active_requests:
            break

    # 获取实际注入的packet_id列表
    active_packet_ids = list(model.request_tracker.active_requests.keys())
    if not active_packet_ids:
        print("❌ 没有活跃的请求被注入")
        return False

    print(f"📝 活跃的packet_id: {active_packet_ids}")

    # 跟踪第一个packet_id
    packet_id = active_packet_ids[0]

    # 跟踪所有活跃的packet_id，包括可能的数据flit格式
    all_packets = active_packet_ids

    # 启用全局调试模式，跟踪实际的packet_id
    # enable_debug方法签名: enable_debug(level: int = 1, trace_packets: List[str] = None, sleep_time: float = 0.0)
    model.enable_debug(level=2, trace_packets=[str(packet_id)], sleep_time=0.5)

    print("-" * 60)

    # 运行仿真 - 调试信息由模型的全局调试控制自动处理
    for cycle in range(1000):
        model.step()

        # 检查是否完成
        if packet_id in model.request_tracker.completed_requests:
            print("-" * 60)
            print("请求完成!")
            break

    # 禁用调试模式
    model.disable_debug()

    # 导出FIFO统计信息
    print("-" * 60)
    print("📊 导出FIFO统计信息...")

    # 导出CSV文件
    csv_path = model.export_fifo_statistics()
    print(f"✅ FIFO统计信息已导出到: {csv_path}")

    # 显示统计摘要
    summary = model.get_fifo_statistics_summary()
    print("\n📈 FIFO统计摘要:")
    print(summary)


if __name__ == "__main__":
    track_request_smart()
