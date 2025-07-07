#!/usr/bin/env python3
"""
简化的包完成测试 - 使用继承的flit追踪功能
"""

import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), ".")))
from src.noc.crossring.model import CrossRingModel
from src.noc.crossring.config import CrossRingConfig
from src.noc.crossring.flit import CrossRingFlit


def test_packet_completion():
    print("测试包完成机制 - 使用集成的RequestTracker功能...")

    # 创建配置
    config = CrossRingConfig()
    config.num_row = 4
    config.num_col = 4
    config.basic_config.network_frequency = 1.0

    # 创建模型
    model = CrossRingModel(config=config)
    model.initialize_network()
    model.enable_debug(level=1)
    
    # 启用flit追踪
    model.enable_debug_tracing(trace_flits=True)
    
    print(f"模型初始化完成: {config.num_row}x{config.num_col}")

    # 开始追踪请求并注入包
    packet_id = "test_packet"
    model.start_request_tracking(packet_id, source=0, destination=11, op_type="R", burst_size=4)
    
    success = model.inject_packet(src_node=0, dst_node=11, op_type="R", burst_size=4, cycle=0, packet_id=packet_id)
    print(f"包注入结果: {success}")
    
    # 创建并追踪请求flit
    if success:
        # 更新请求状态为已注入
        from src.noc.debug import RequestState
        model.update_request_state(packet_id, RequestState.INJECTED)
        
        # 创建并追踪请求flit
        req_flit = CrossRingFlit(
            flit_type="req",
            source=0,
            destination=11,
            packet_id=packet_id,
            req_type="read"
        )
        model.track_request_flit(packet_id, req_flit, node_id=0)
        print("请求flit已添加到RequestTracker")

    # 模拟几个周期
    for cycle in range(50):
        model.advance_cycle()

        # 检查完成的包
        completed = model.get_completed_packets()
        if completed:
            print(f"周期 {cycle}: 发现完成的包:")
            for packet in completed:
                print(f"  包ID: {packet['packet_id']}, 延迟: {packet['latency']} 周期")
                
                # 更新请求状态为完成
                model.update_request_state(packet['packet_id'], RequestState.COMPLETED)
                
                # 创建并追踪响应flit
                rsp_flit = CrossRingFlit(
                    flit_type="rsp",
                    source=11,
                    destination=0,
                    packet_id=packet['packet_id'],
                    rsp_type="ack"
                )
                model.track_response_flit(packet['packet_id'], rsp_flit, node_id=11)
                
                # 创建并追踪数据flit
                for i in range(4):  # burst_size=4
                    data_flit = CrossRingFlit(
                        flit_type="data",
                        source=11,
                        destination=0,
                        packet_id=packet['packet_id'],
                        flit_id=i
                    )
                    model.track_data_flit(packet['packet_id'], data_flit, node_id=11)
            break

        # 每5个周期显示状态
        if cycle % 5 == 0:
            print(f"\n===== 周期 {cycle} =====")
            print(f"活跃请求数: {model.get_active_request_count()}")
            
            # 显示详细的flit追踪状态
            model.print_packet_flit_status(packet_id)
            
            # 显示追踪器统计
            tracker_stats = model.get_request_tracker_statistics()
            print(f"RequestTracker统计: {tracker_stats}")
            print("-" * 40)

    # 最终状态显示
    print(f"\n===== 最终状态 =====")
    model.print_packet_flit_status(packet_id)
    
    # 显示所有被追踪的包
    all_packets = model.get_all_tracked_packets()
    print(f"所有被追踪的包: {all_packets}")
    
    # 最终统计
    debug_stats = model.get_debug_statistics()
    print(f"\n最终统计:")
    print(f"  完成请求: {debug_stats.get('completed_requests', 0)}")
    print(f"  活跃请求: {model.get_active_request_count()}")
    
    # 显示RequestTracker完整报告
    print(f"\n===== RequestTracker完整报告 =====")
    model.print_request_tracker_report()


if __name__ == "__main__":
    test_packet_completion()
