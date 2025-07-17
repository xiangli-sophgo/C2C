#!/usr/bin/env python3
"""
简化的CrossRing测试，手动设置IP连接
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.noc.crossring.model import CrossRingModel
from src.noc.crossring.config import CrossRingConfig
import logging

# 禁用所有日志
logging.disable(logging.CRITICAL)

def simple_crossring_test():
    """简化的CrossRing测试"""
    print("🔍 简化的CrossRing测试...")
    
    # 创建2x2配置
    config = CrossRingConfig.create_custom_config(num_row=2, num_col=2)
    
    traffic_file = Path(__file__).parent.parent.parent / "traffic_data" / "sample_traffic.txt"
    
    try:
        # 创建模型（不调用initialize_network）
        print("  创建模型...")
        model = CrossRingModel(config, traffic_file_path=str(traffic_file))
        print("  ✅ 模型创建成功")
        
        # 手动连接IP接口
        print("  手动连接IP接口...")
        
        # 从traffic文件读取需要的IP：gdma_0 at node 0, ddr_1 at node 1
        if hasattr(model, 'crossring_nodes') and 0 in model.crossring_nodes and 1 in model.crossring_nodes:
            node0 = model.crossring_nodes[0]
            node1 = model.crossring_nodes[1]
            
            # 手动连接IP
            node0.connect_ip("gdma_0_node0")
            node1.connect_ip("ddr_1_node1")
            
            print(f"  ✅ 节点0连接的IP: {node0.connected_ips}")
            print(f"  ✅ 节点1连接的IP: {node1.connected_ips}")
        else:
            print("  ❌ 节点未正确创建")
            return False
        
        # 设置流量调度器
        print("  设置流量调度器...")
        traffic_filename = traffic_file.name
        model.setup_traffic_scheduler([[traffic_filename]], str(traffic_file.parent))
        print("  ✅ 流量调度器设置成功")
        
        # 启用debug
        model.enable_debug(level=2, trace_packets=["1"], sleep_time=0.0)
        
        # 运行几个周期
        print("  运行仿真...")
        for cycle in range(20):
            model.step()
            
            # 检查是否有完成的请求
            if model.request_tracker.completed_requests:
                packet_id = list(model.request_tracker.completed_requests.keys())[0]
                lifecycle = model.request_tracker.completed_requests[packet_id]
                latency = lifecycle.completed_cycle - lifecycle.created_cycle
                print(f"✅ 请求{packet_id}在周期{lifecycle.completed_cycle}完成，延迟: {latency} 周期")
                return True
        
        print("⚠️ 20个周期内请求未完成")
        return False
        
    except Exception as e:
        print(f"❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    simple_crossring_test()