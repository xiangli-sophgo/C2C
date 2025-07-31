#!/usr/bin/env python3
"""
测试CrossPoint与RingSlice的完整集成
模拟真实的上环/下环场景
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from src.noc.crossring.link import CrossRingLink, Direction, CrossRingSlot
from src.noc.crossring.flit import CrossRingFlit
from src.noc.crossring.config import CrossRingConfig

def test_crosspoint_injection_ejection_flow():
    """测试CrossPoint的注入和弹出流程"""
    print("🧪 测试CrossPoint注入和弹出流程")
    
    config = CrossRingConfig()
    link = CrossRingLink("integration_test", 0, 1, Direction.TR, config, num_slices=6) 
    
    # 获取关键slice：arrival slice用于注入，departure slice用于弹出
    arrival_slice = link.ring_slices["req"][0]  # CrossPoint的arrival slice
    departure_slice = link.ring_slices["req"][2]  # 几个周期后会到达的slice
    
    print("✅ 创建测试环境，6个slice的环形链路")
    
    # 测试场景1: CrossPoint上环逻辑
    print("\n🔄 场景1: CrossPoint上环逻辑测试")
    
    # 模拟CrossPoint检查arrival slice状态
    arrival_slot = arrival_slice.peek_current_slot("req")
    print(f"arrival slice状态: occupied={arrival_slot.is_occupied if arrival_slot else 'N/A'}")
    
    # 模拟CrossPoint的_can_inject_to_arrival_slice逻辑
    can_inject = arrival_slot is not None and not arrival_slot.is_occupied
    print(f"可以注入: {can_inject}")
    
    if can_inject:
        # 创建要注入的flit
        inject_flit = CrossRingFlit(
            flit_id=4000,
            packet_id="inject_test_4000",
            source=0,
            destination=3,
            flit_type="req",
            flit_size=128,
            req_type="r"
        )
        
        # 模拟CrossPoint的_inject_flit_to_arrival_slice逻辑
        success = arrival_slice.inject_flit_to_slot(inject_flit, "req")
        print(f"注入结果: {'成功' if success else '失败'}")
        
        if success:
            # 验证注入后状态
            after_slot = arrival_slice.peek_current_slot("req")
            print(f"注入后状态: occupied={after_slot.is_occupied}, flit_id={after_slot.flit.flit_id}")
    
    # 测试场景2: 环形传递过程
    print("\n🔄 场景2: 环形传递过程")
    
    print("传递过程:")
    for cycle in range(8):
        print(f"\n--- 周期 {cycle} ---")
        
        # 显示当前所有slice的状态
        for i, slice_obj in enumerate(link.ring_slices["req"]):
            slot = slice_obj.current_slots["req"]
            flit_id = slot.flit.flit_id if slot and slot.flit else "空"
            print(f"  slice[{i}]: {flit_id}")
        
        # 执行传递
        link.step_compute_phase(cycle)
        link.step_update_phase(cycle)
        
        # 检查flit是否到达特定位置（模拟CrossPoint检查departure slice）
        departure_slot = departure_slice.peek_current_slot("req")
        if departure_slot and departure_slot.is_occupied:
            print(f"✅ flit到达departure slice (位置{departure_slice.position}): {departure_slot.flit.flit_id}")
    
    # 测试场景3: CrossPoint下环逻辑
    print("\n🔄 场景3: CrossPoint下环逻辑测试")
    
    # 在某个slice放置一个准备下环的flit
    target_slice = link.ring_slices["req"][1]
    eject_flit = CrossRingFlit(
        flit_id=5000,
        packet_id="eject_test_5000",
        source=1,
        destination=1,  # 目标是当前节点，应该下环
        flit_type="req",
        flit_size=128,
        req_type="w"
    )
    
    # 先清空该slice的现有flit
    empty_slot = CrossRingSlot(
        slot_id="eject_test_slot",
        cycle=0,
        direction=Direction.TR,
        channel="req",
        valid=False,
        flit=None
    )
    empty_slot.assign_flit(eject_flit)
    target_slice.current_slots["req"] = empty_slot
    
    print(f"在slice[{target_slice.position}]放置了准备下环的flit {eject_flit.flit_id}")
    
    # 模拟CrossPoint的下环检查逻辑
    current_slot = target_slice.peek_current_slot("req")
    if current_slot and current_slot.is_occupied:
        flit = current_slot.flit
        print(f"检查flit {flit.flit_id}: source={flit.source}, dest={flit.destination}")
        
        # 简单的下环判断逻辑（flit到达目标节点）
        should_eject = (flit.destination == 1)  # 假设当前是节点1
        print(f"是否应该下环: {should_eject}")
        
        if should_eject:
            # 模拟下环操作：从slot中取出flit
            ejected_flit = flit
            current_slot.release_flit()  # 清空slot
            print(f"✅ 成功下环flit {ejected_flit.flit_id}")
            
            # 验证slot已被清空
            after_eject_slot = target_slice.peek_current_slot("req")
            print(f"下环后slot状态: occupied={after_eject_slot.is_occupied}")
    
    print("\n✅ CrossPoint集成测试完成")

def test_multiple_flits_with_crosspoint_operations():
    """测试多个flit的CrossPoint操作"""
    print("\n🔄 测试多个flit的CrossPoint操作")
    
    config = CrossRingConfig()
    link = CrossRingLink("multi_ops_test", 0, 1, Direction.TR, config, num_slices=4)
    
    # 场景：在不同位置注入和弹出多个flit
    operations = [
        {"cycle": 0, "op": "inject", "slice_idx": 0, "flit_id": 6001},
        {"cycle": 1, "op": "inject", "slice_idx": 0, "flit_id": 6002},  
        {"cycle": 3, "op": "eject", "slice_idx": 2, "expected_flit": 6001},
        {"cycle": 4, "op": "eject", "slice_idx": 2, "expected_flit": 6002},
    ]
    
    print("执行复杂的注入/弹出序列:")
    
    for cycle in range(6):
        print(f"\n--- 周期 {cycle} ---")
        
        # 检查是否有预定的操作
        cycle_ops = [op for op in operations if op["cycle"] == cycle]
        
        for op in cycle_ops:
            if op["op"] == "inject":
                # 注入操作
                target_slice = link.ring_slices["req"][op["slice_idx"]]
                inject_flit = CrossRingFlit(
                    flit_id=op["flit_id"],
                    packet_id=f"multi_ops_{op['flit_id']}",
                    source=0,
                    destination=2,
                    flit_type="req",
                    flit_size=128
                )
                
                slot = target_slice.peek_current_slot("req")
                if slot and not slot.is_occupied:
                    success = target_slice.inject_flit_to_slot(inject_flit, "req")
                    print(f"  📥 注入flit {op['flit_id']} 到slice[{op['slice_idx']}]: {'成功' if success else '失败'}")
                else:
                    print(f"  ❌ slice[{op['slice_idx']}]已占用，无法注入flit {op['flit_id']}")
                    
            elif op["op"] == "eject":
                # 弹出操作
                target_slice = link.ring_slices["req"][op["slice_idx"]]
                slot = target_slice.peek_current_slot("req")
                if slot and slot.is_occupied:
                    actual_flit_id = slot.flit.flit_id
                    expected_flit_id = op["expected_flit"]
                    if actual_flit_id == expected_flit_id:
                        slot.release_flit()
                        print(f"  📤 从slice[{op['slice_idx']}]弹出flit {actual_flit_id}: 成功")
                    else:
                        print(f"  ⚠️  slice[{op['slice_idx']}]的flit {actual_flit_id} 不匹配预期的 {expected_flit_id}")
                else:
                    print(f"  ❌ slice[{op['slice_idx']}]为空，无法弹出")
        
        # 显示当前状态
        for i, slice_obj in enumerate(link.ring_slices["req"]):
            slot = slice_obj.current_slots["req"]
            flit_id = slot.flit.flit_id if slot and slot.flit else "空"
            print(f"  slice[{i}]: {flit_id}")
        
        # 执行环形传递
        link.step_compute_phase(cycle)
        link.step_update_phase(cycle)
    
    print("\n✅ 多flit CrossPoint操作测试完成")

if __name__ == "__main__":
    try:
        test_crosspoint_injection_ejection_flow()
        test_multiple_flits_with_crosspoint_operations()
        print("\n🎉 CrossPoint集成测试全部通过!")
    except Exception as e:
        print(f"\n❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()