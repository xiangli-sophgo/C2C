#!/usr/bin/env python3
"""
测试简化后的CrossRing链路传递机制
验证基于简单寄存器的环形传递是否工作正常
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from src.noc.crossring.link import CrossRingLink, Direction
from src.noc.crossring.flit import CrossRingFlit
from src.noc.crossring.config import CrossRingConfig

def test_simplified_ring_transmission():
    """测试简化后的环形传递机制"""
    print("🧪 测试简化后的CrossRing环形传递机制")
    
    # 创建配置和链路
    config = CrossRingConfig()
    link = CrossRingLink("test_link", 0, 1, Direction.TR, config, num_slices=6)
    
    print(f"✅ 创建了包含{len(link.ring_slices['req'])}个slice的链路")
    
    # 创建测试flit
    test_flit = CrossRingFlit(
        flit_id=100,
        packet_id="test_packet_100", 
        source=0,
        destination=4,
        flit_type="req",
        flit_size=128,
        req_type="w"
    )
    
    # 在第一个slice中创建一个带flit的slot
    first_slice = link.ring_slices["req"][0]
    
    # 创建slot
    from src.noc.crossring.link import CrossRingSlot
    test_slot = CrossRingSlot(
        slot_id="test_slot_1",
        cycle=0,
        direction=Direction.TR,
        channel="req",
        valid=False,
        flit=None
    )
    test_slot.assign_flit(test_flit)
    
    # 直接设置到第一个slice的current_slots
    first_slice.current_slots["req"] = test_slot
    
    print(f"✅ 在slice[0]中放置了flit {test_flit.flit_id}")
    
    # 运行12个周期，观察环形传递
    print("\n🔄 开始运行环形传递周期:")
    
    for cycle in range(12):
        print(f"\n--- 周期 {cycle} ---")
        
        # 显示当前所有slice的状态
        print("当前状态:")
        for i, slice_obj in enumerate(link.ring_slices["req"]):
            current_slot = slice_obj.current_slots["req"]
            next_slot = slice_obj.next_slots["req"]
            
            current_flit = current_slot.flit.flit_id if current_slot and current_slot.flit else "None"
            next_flit = next_slot.flit.flit_id if next_slot and next_slot.flit else "None"
            
            print(f"  slice[{i}]: current={current_flit}, next={next_flit}")
        
        # 执行compute和update阶段
        link.step_compute_phase(cycle)
        print("compute阶段完成")
        
        # 显示compute后的状态
        print("compute后状态:")
        for i, slice_obj in enumerate(link.ring_slices["req"]):
            current_slot = slice_obj.current_slots["req"]
            next_slot = slice_obj.next_slots["req"]
            
            current_flit = current_slot.flit.flit_id if current_slot and current_slot.flit else "None"
            next_flit = next_slot.flit.flit_id if next_slot and next_slot.flit else "None"
            
            print(f"  slice[{i}]: current={current_flit}, next={next_flit}")
        
        link.step_update_phase(cycle)
        print("update阶段完成")
        
        # 检查flit位置
        flit_position = -1
        for i, slice_obj in enumerate(link.ring_slices["req"]):
            slot = slice_obj.current_slots["req"]
            if slot and slot.is_occupied and slot.flit and slot.flit.flit_id == test_flit.flit_id:
                flit_position = i
                break
        
        print(f"周期{cycle}结束: flit在slice[{flit_position}]")
        
        # 如果flit丢失，停止测试
        if flit_position == -1:
            print("❌ flit丢失!")
            break
    
    print("\n✅ 简化环形传递测试完成")

def test_multiple_flits():
    """测试多个flit的环形传递"""
    print("\n🔄 测试多个flit的环形传递")
    
    config = CrossRingConfig()
    link = CrossRingLink("multi_test", 0, 1, Direction.TR, config, num_slices=4)
    
    # 创建多个flit并放在不同slice中
    from src.noc.crossring.link import CrossRingSlot
    
    flits = []
    for i in range(3):
        flit = CrossRingFlit(
            flit_id=200 + i,
            packet_id=f"packet_{200+i}",
            source=0,
            destination=4,
            flit_type="req",
            flit_size=128
        )
        flits.append(flit)
        
        # 创建slot并放置flit
        slot = CrossRingSlot(
            slot_id=f"slot_{i}",
            cycle=0,
            direction=Direction.TR,
            channel="req",
            valid=False,
            flit=None
        )
        slot.assign_flit(flit)
        
        # 放在不同的slice中
        link.ring_slices["req"][i].current_slots["req"] = slot
        print(f"✅ flit {flit.flit_id} 放在 slice[{i}]")
    
    # 运行几个周期
    print("\n🔄 运行传递周期:")
    for cycle in range(6):
        print(f"\n--- 周期 {cycle} ---")
        
        # 显示所有flit位置
        for i, slice_obj in enumerate(link.ring_slices["req"]):
            slot = slice_obj.current_slots["req"]
            if slot and slot.is_occupied and slot.flit:
                print(f"  slice[{i}]: flit {slot.flit.flit_id}")
            else:
                print(f"  slice[{i}]: empty")
        
        # 执行传递
        link.step_compute_phase(cycle)
        link.step_update_phase(cycle)
    
    print("\n✅ 多flit传递测试完成")

if __name__ == "__main__":
    try:
        test_simplified_ring_transmission()
        test_multiple_flits()
        print("\n🎉 所有测试完成!")
    except Exception as e:
        print(f"\n❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()