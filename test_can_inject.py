#!/usr/bin/env python3
"""
测试CrossRing中can_inject_flit方法的行为
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '.'))

from src.noc.crossring.model import CrossRingModel
from src.noc.crossring.config import CrossRingConfig

def test_can_inject_behavior():
    """测试can_inject_flit的行为"""
    print("🔍 分析can_inject_flit方法行为")
    
    # 创建基本的CrossRing配置
    config = CrossRingConfig(
        NUM_NODE=4,
        NUM_ROW=2,
        NUM_COL=2,
        RING_TYPE_LIST=["horizontal", "vertical"],
        SLICE_PER_LINK=8,
        ITAG_ENABLED=True,
        ETAG_ENABLED=True
    )
    
    # 创建模型
    model = CrossRingModel(config)
    
    print(f"✅ 创建CrossRing模型完成: {config.NUM_ROW}x{config.NUM_COL}")
    
    # 获取第一个节点
    node_0 = model.nodes[0]
    
    print(f"📍 节点0坐标: ({node_0.row}, {node_0.col})")
    
    # 获取水平CrossPoint
    horizontal_cp = node_0.horizontal_crosspoint
    vertical_cp = node_0.vertical_crosspoint
    
    print(f"🔄 水平CrossPoint管理方向: {horizontal_cp.managed_directions}")
    print(f"🔄 垂直CrossPoint管理方向: {vertical_cp.managed_directions}")
    
    # 检查初始状态下的can_inject_flit
    for direction in horizontal_cp.managed_directions:
        for channel in ["req", "rsp", "data"]:
            can_inject = horizontal_cp.can_inject_flit(direction, channel)
            slice_info = horizontal_cp.slices[direction]["departure"]
            current_slot = slice_info.peek_current_slot(channel)
            
            print(f"💧 水平CP {direction}-{channel}: can_inject={can_inject}, current_slot={current_slot}")
            
            if current_slot is not None:
                print(f"   槽位详情: valid={current_slot.valid}, reserved={current_slot.is_reserved}")
    
    for direction in vertical_cp.managed_directions:
        for channel in ["req", "rsp", "data"]:
            can_inject = vertical_cp.can_inject_flit(direction, channel)
            slice_info = vertical_cp.slices[direction]["departure"]
            current_slot = slice_info.peek_current_slot(channel)
            
            print(f"🌊 垂直CP {direction}-{channel}: can_inject={can_inject}, current_slot={current_slot}")
            
            if current_slot is not None:
                print(f"   槽位详情: valid={current_slot.valid}, reserved={current_slot.is_reserved}")
    
    # 运行几个周期，看看状态变化
    print("\n" + "="*60)
    print("📊 运行几个周期观察状态变化")
    
    for cycle in range(1, 4):
        print(f"\n--- 周期 {cycle} ---")
        model.step()
        
        # 再次检查can_inject_flit
        direction = "TR"
        channel = "req"
        can_inject = horizontal_cp.can_inject_flit(direction, channel)
        slice_info = horizontal_cp.slices[direction]["departure"]
        current_slot = slice_info.peek_current_slot(channel)
        
        print(f"水平CP {direction}-{channel}: can_inject={can_inject}")
        if current_slot is not None:
            print(f"  current_slot存在: valid={current_slot.valid}, reserved={current_slot.is_reserved}")
            if current_slot.is_reserved:
                print(f"    预约者ID: {current_slot.itag_reserver_id}, 节点ID: {horizontal_cp.node_id}")
        else:
            print(f"  current_slot为None")

if __name__ == "__main__":
    test_can_inject_behavior()