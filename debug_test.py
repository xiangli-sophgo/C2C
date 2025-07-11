#!/usr/bin/env python3

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

from src.noc.crossring.model import CrossRingModel
from src.noc.crossring.config import CrossRingConfig

def test_basic_functionality():
    """测试基本功能"""
    print("创建配置...")
    config = CrossRingConfig(num_row=3, num_col=3, config_name="test_3x3")
    config.gdma_send_position_list = [0]
    config.ddr_send_position_list = [4]
    config.debug_enabled = True
    
    print("创建模型...")
    model = CrossRingModel(config)
    
    print("注入单个请求...")
    injected = model.inject_request(source=0, destination=4, req_type="read", 
                                   burst_length=4, source_type="gdma_0", 
                                   destination_type="ddr_0")
    
    print(f"注入了 {injected} 个请求")
    
    # 检查生成的请求详情
    print("生成的请求详情：")
    for packet_id, info in model.packet_id_map.items():
        print(f"  {packet_id}: {info['source']}->{info['destination']}, {info['source_type']}->{info['destination_type']}")
        
    # 检查请求追踪器
    print("RequestTracker状态：")
    tracked_requests = model.request_tracker.get_active_tracked_requests()
    for packet_id, lifecycle in tracked_requests.items():
        print(f"  {packet_id}: {lifecycle.source}->{lifecycle.destination}")
        if lifecycle.request_flits:
            req_flit = lifecycle.request_flits[-1]
            print(f"    Request flit: {req_flit.source}->{req_flit.destination}")
            print(f"    Source type: {req_flit.source_type}, Dest type: {req_flit.destination_type}")
    
    print("运行仿真...")
    for cycle in range(100):  # 增加周期数以观察完整流程
        model.step()
        if cycle % 20 == 0 or cycle > 10:
            print(f"周期 {cycle}")
            # 显示请求追踪器状态
            tracked_requests = model.request_tracker.get_active_tracked_requests()
            print(f"  活跃请求: {len(tracked_requests)}")
            if tracked_requests:
                for packet_id, lifecycle in tracked_requests.items():
                    print(f"    {packet_id}: {lifecycle.current_state.value}")
        
        # 在周期20时检查flit是否到达目标节点
        if cycle == 20:
            print("🔍 周期20检查：请求flit是否到达目标节点4...")
            tracked_requests = model.request_tracker.get_active_tracked_requests()
            for packet_id, lifecycle in tracked_requests.items():
                if lifecycle.request_flits:
                    req_flit = lifecycle.request_flits[-1]
                    print(f"   请求flit {packet_id}: {req_flit.flit_position}")
                    if req_flit.flit_position == "completed":
                        print(f"   ✅ 请求flit {packet_id} 已完成！")
                    else:
                        print(f"   ⏳ 请求flit {packet_id} 仍在传输: {req_flit.flit_position}")
    
    print("完成测试")

if __name__ == "__main__":
    test_basic_functionality()