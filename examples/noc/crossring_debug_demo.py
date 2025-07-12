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
    # 创建2x2配置
    config = CrossRingConfig(num_row=2, num_col=2)
    config.gdma_send_position_list = [0]
    config.ddr_send_position_list = [1]
    # 清空其他不需要的IP
    config.sdma_send_position_list = []
    config.cdma_send_position_list = []
    config.l2m_send_position_list = []

    # 创建模型，重定向输出
    import io
    import contextlib

    with contextlib.redirect_stdout(io.StringIO()):
        with contextlib.redirect_stderr(io.StringIO()):
            model = CrossRingModel(config)

    # 注入请求
    packet_ids = model.inject_request(
        source=0,
        destination=5,
        req_type="read",
        # req_type="write",
        burst_length=4,
        source_type="gdma_0",
        destination_type="l2m_0",
    )

    if not packet_ids:
        print("注入失败")
        return

    packet_id = packet_ids[0]  # 保持原始类型
    print(f"跟踪 packet_id={packet_id}: Read 0->1")

    # 启用全局调试模式，跟踪特定packet_id，设置0.1秒的休眠时间
    model.enable_debug([packet_id], 0.1)

    print("-" * 60)

    # 运行仿真 - 调试信息由模型的全局调试控制自动处理
    for cycle in range(100):
        model.step()

        # 检查是否完成
        if packet_id in model.request_tracker.completed_requests:
            print("-" * 60)
            print("请求完成!")
            break

    # 禁用调试模式
    model.disable_debug()


if __name__ == "__main__":
    track_request_smart()
