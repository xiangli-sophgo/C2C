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
    config = CrossRingConfig(num_row=3, num_col=3)
    # config.gdma_send_position_list = [0]
    # config.ddr_send_position_list = [1]
    # config.l2m_send_position_list = [3]  # 节点3需要l2m接口来接收请求
    # # 清空其他不需要的IP
    # config.sdma_send_position_list = []
    # config.cdma_send_position_list = []

    traffic_file = Path(__file__).parent.parent.parent / "traffic_data" / "all_to_all_traffic.txt"
    if not traffic_file.exists():
        print(f"❌ Traffic文件不存在: {traffic_file}")
        return False

    # 创建模型时重定向详细输出
    import io
    import contextlib

    with contextlib.redirect_stdout(io.StringIO()):
        model = CrossRingModel(config, traffic_file_path=str(traffic_file))

    # 2. 注入流量并运行仿真
    injected_count = model.inject_from_traffic_file(traffic_file_path=str(traffic_file), cycle_accurate=True)  # 使用周期精确模式

    # 检查注入结果
    if not injected_count:
        print("❌ 流量注入失败")
        return False

    print(f"✅ 成功加载 {injected_count} 个请求到待处理队列")

    # 在cycle-accurate模式下，packet_id从pending_file_requests获取
    # 运行几个周期让请求被实际注入
    print("⏳ 运行仿真等待请求注入...")
    for _ in range(10):
        model.step()
        if model.request_tracker.active_requests:
            break

    # 获取实际注入的packet_id列表
    active_packet_ids = list(model.request_tracker.active_requests.keys())
    if not active_packet_ids:
        print("❌ 没有活跃的请求被注入")
        return False

    print(f"📝 活跃的packet_id: {active_packet_ids}")

    # 跟踪第一个packet_id
    packet_id = 1

    # 跟踪所有活跃的packet_id，包括可能的数据flit格式
    all_packets = active_packet_ids

    # 同时跟踪可能的数据flit packet_id格式（如果有的话）
    # for pid in active_packet_ids:
    #     for i in range(4):  # burst_length=4
    #         all_packets.append(f"{pid}_data_{i}")

    # print(f"📝 跟踪的packet_id列表: {all_packets}")

    # 启用全局调试模式，跟踪实际的packet_id
    model.enable_debug(packet_id, 0.02)

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
