#!/usr/bin/env python3
"""测试start_time修改的正确性"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.noc.analysis.result_analyzer import ResultAnalyzer, RequestInfo
from src.noc.debug.request_tracker import RequestTracker, RequestLifecycle, RequestState
from src.noc.crossring.flit import CrossRingFlit
from dataclasses import dataclass
import numpy as np

# 创建模拟配置
@dataclass
class BasicConfig:
    NETWORK_FREQUENCY = 2.0  # GHz
    FLIT_SIZE = 128

@dataclass
class Config:
    basic_config = BasicConfig()

def test_start_time_calculation():
    """测试start_time的计算逻辑"""
    
    # 创建RequestTracker和模拟数据
    tracker = RequestTracker()
    config = Config()
    
    # 创建一个完成的请求
    packet_id = "test_packet_1"
    lifecycle = RequestLifecycle(
        packet_id=packet_id,
        source=0,
        destination=1,
        op_type="write",
        burst_size=4,
        created_cycle=100  # RequestTracker创建时间
    )
    lifecycle.completed_cycle = 200
    
    # 创建一个模拟的flit，设置cmd_entry_cake0_cycle
    flit = CrossRingFlit()
    flit.cmd_entry_cake0_cycle = 105  # 实际进入Cake0的时间（比created_cycle晚）
    flit.source_type = "gdma"
    flit.destination_type = "ddr"
    
    # 添加flit到lifecycle
    lifecycle.request_flits.append(flit)
    
    # 将lifecycle添加到完成的请求中
    tracker.completed_requests[packet_id] = lifecycle
    
    # 使用ResultAnalyzer转换数据
    analyzer = ResultAnalyzer()
    requests = analyzer.convert_tracker_to_request_info(tracker, config)
    
    # 验证结果
    assert len(requests) == 1
    req_info = requests[0]
    
    # 计算期望值
    cycle_time_ns = 1000.0 / (2.0 * 1000)  # 0.5 ns per cycle
    expected_start_time = int(105 * cycle_time_ns)  # 使用cmd_entry_cake0_cycle
    
    print(f"测试结果：")
    print(f"  created_cycle: {lifecycle.created_cycle}")
    print(f"  cmd_entry_cake0_cycle: {flit.cmd_entry_cake0_cycle}")
    print(f"  cycle_time_ns: {cycle_time_ns}")
    print(f"  计算的start_time: {req_info.start_time} ns")
    print(f"  期望的start_time: {expected_start_time} ns")
    
    if req_info.start_time == expected_start_time:
        print("✅ 测试通过：start_time正确使用了cmd_entry_cake0_cycle")
    else:
        print("❌ 测试失败：start_time计算不正确")
        
    # 测试回退逻辑
    print("\n测试回退逻辑：")
    lifecycle2 = RequestLifecycle(
        packet_id="test_packet_2",
        source=0,
        destination=1,
        op_type="read",
        burst_size=4,
        created_cycle=150
    )
    lifecycle2.completed_cycle = 250
    
    # 创建没有cmd_entry_cake0_cycle的flit
    flit2 = CrossRingFlit()
    # 不设置cmd_entry_cake0_cycle，应该回退到created_cycle
    flit2.source_type = "gdma"
    flit2.destination_type = "ddr"
    
    lifecycle2.request_flits.append(flit2)
    tracker.completed_requests["test_packet_2"] = lifecycle2
    
    # 重新转换
    requests2 = analyzer.convert_tracker_to_request_info(tracker, config)
    req_info2 = next(r for r in requests2 if r.packet_id == "test_packet_2")
    
    expected_fallback_time = int(150 * cycle_time_ns)  # 使用created_cycle
    print(f"  created_cycle: {lifecycle2.created_cycle}")
    print(f"  cmd_entry_cake0_cycle: 未设置")
    print(f"  计算的start_time: {req_info2.start_time} ns")
    print(f"  期望的start_time（回退）: {expected_fallback_time} ns")
    
    if req_info2.start_time == expected_fallback_time:
        print("✅ 回退测试通过：当cmd_entry_cake0_cycle不可用时，正确回退到created_cycle")
    else:
        print("❌ 回退测试失败")

if __name__ == "__main__":
    test_start_time_calculation()