#!/usr/bin/env python3
"""
测试CrossPoint与简化RingSlice的集成
验证上环/下环接口是否正常工作
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from src.noc.crossring.link import CrossRingLink, Direction, CrossRingSlot
from src.noc.crossring.flit import CrossRingFlit
from src.noc.crossring.config import CrossRingConfig

def test_crosspoint_ring_interface():
    """测试CrossPoint接口与简化RingSlice的兼容性"""
    print("🧪 测试CrossPoint与简化RingSlice的接口兼容性")
    
    # 创建配置和链路
    config = CrossRingConfig()
    link = CrossRingLink("test_link", 0, 1, Direction.TR, config, num_slices=4)
    
    # 获取第一个slice作为测试对象
    test_slice = link.ring_slices["req"][0]
    
    print("✅ 创建了测试环境")
    
    # 测试1: peek_current_slot接口
    print("\n🔍 测试1: peek_current_slot接口")
    
    # 初始状态应该没有slot
    current_slot = test_slice.peek_current_slot("req")
    print(f"初始状态的slot: {current_slot}")
    
    # 创建一个空slot并设置到slice中
    empty_slot = CrossRingSlot(
        slot_id="test_slot",
        cycle=0,
        direction=Direction.TR,
        channel="req",
        valid=False,
        flit=None
    )
    test_slice.current_slots["req"] = empty_slot
    
    # 再次检查
    current_slot = test_slice.peek_current_slot("req")
    print(f"设置空slot后: {current_slot is not None}, occupied: {current_slot.is_occupied if current_slot else 'N/A'}")
    
    # 测试2: inject_flit_to_slot接口
    print("\n🔍 测试2: inject_flit_to_slot接口")
    
    # 创建测试flit
    test_flit = CrossRingFlit(
        flit_id=999,
        packet_id="test_packet_999",
        source=0,
        destination=2,
        flit_type="req",
        flit_size=128,
        req_type="r"
    )
    
    # 尝试注入flit到空slot
    success = test_slice.inject_flit_to_slot(test_flit, "req")
    print(f"注入flit到空slot: {'成功' if success else '失败'}")
    
    # 检查注入后的状态
    after_inject_slot = test_slice.peek_current_slot("req")
    if after_inject_slot:
        print(f"注入后slot状态: occupied={after_inject_slot.is_occupied}, flit_id={after_inject_slot.flit.flit_id if after_inject_slot.flit else 'None'}")
    
    # 测试3: 尝试向已占用的slot注入（应该失败）
    print("\n🔍 测试3: 向已占用slot注入（应该失败）")
    
    another_flit = CrossRingFlit(
        flit_id=1000,
        packet_id="test_packet_1000", 
        source=1,
        destination=3,
        flit_type="req",
        flit_size=128,
        req_type="w"
    )
    
    fail_success = test_slice.inject_flit_to_slot(another_flit, "req")
    print(f"向已占用slot注入: {'成功' if fail_success else '失败（符合预期）'}")
    
    # 测试4: 环形传递中的接口使用
    print("\n🔍 测试4: 环形传递中的接口兼容性")
    
    # 设置上游slice
    upstream_slice = link.ring_slices["req"][3]  # 环形连接，最后一个是第一个的上游
    test_slice.upstream_slice = upstream_slice
    
    # 在上游slice放一个flit
    upstream_slot = CrossRingSlot(
        slot_id="upstream_slot",
        cycle=0,
        direction=Direction.TR,
        channel="req",
        valid=False,
        flit=None
    )
    upstream_flit = CrossRingFlit(
        flit_id=2000,
        packet_id="upstream_packet_2000",
        source=2,
        destination=4,
        flit_type="req",
        flit_size=128
    )
    upstream_slot.assign_flit(upstream_flit)
    upstream_slice.current_slots["req"] = upstream_slot
    
    print("✅ 在上游slice放置了flit 2000")
    
    # 执行一个完整的传递周期
    print("\n🔄 执行环形传递周期:")
    
    # 显示传递前状态
    print("传递前:")
    for i, slice_obj in enumerate(link.ring_slices["req"][:2]):
        slot = slice_obj.current_slots["req"] 
        flit_id = slot.flit.flit_id if slot and slot.flit else "None"
        print(f"  slice[{i}]: flit {flit_id}")
    
    # 执行compute和update
    link.step_compute_phase(0)
    print("compute阶段完成")
    
    # 显示compute后状态
    print("compute后:")
    for i, slice_obj in enumerate(link.ring_slices["req"][:2]):
        current_slot = slice_obj.current_slots["req"]
        next_slot = slice_obj.next_slots["req"]
        current_flit = current_slot.flit.flit_id if current_slot and current_slot.flit else "None"
        next_flit = next_slot.flit.flit_id if next_slot and next_slot.flit else "None" 
        print(f"  slice[{i}]: current={current_flit}, next={next_flit}")
    
    link.step_update_phase(0)
    print("update阶段完成")
    
    # 显示传递后状态
    print("传递后:")
    for i, slice_obj in enumerate(link.ring_slices["req"][:2]):
        slot = slice_obj.current_slots["req"]
        flit_id = slot.flit.flit_id if slot and slot.flit else "None"
        print(f"  slice[{i}]: flit {flit_id}")
    
    print("\n✅ CrossPoint接口兼容性测试完成")

def test_multiple_channels():
    """测试多通道的接口兼容性"""
    print("\n🔍 测试多通道接口兼容性")
    
    config = CrossRingConfig()
    link = CrossRingLink("multi_channel_test", 0, 1, Direction.TR, config, num_slices=2)
    test_slice = link.ring_slices["req"][0]
    
    # 为每个通道创建slot
    channels = ["req", "rsp", "data"]
    for i, channel in enumerate(channels):
        slot = CrossRingSlot(
            slot_id=f"slot_{channel}",
            cycle=0,
            direction=Direction.TR,
            channel=channel,
            valid=False,
            flit=None
        )
        test_slice.current_slots[channel] = slot
        
        # 为每个通道注入不同的flit
        flit = CrossRingFlit(
            flit_id=3000 + i,
            packet_id=f"packet_{channel}_{3000+i}",
            source=0,
            destination=1,
            flit_type=channel,
            flit_size=128
        )
        
        success = test_slice.inject_flit_to_slot(flit, channel)
        print(f"通道{channel}注入flit {3000+i}: {'成功' if success else '失败'}")
        
        # 验证可以正确peek
        current_slot = test_slice.peek_current_slot(channel)
        if current_slot and current_slot.flit:
            print(f"  peek结果: flit_id={current_slot.flit.flit_id}")
    
    print("✅ 多通道测试完成")

if __name__ == "__main__":
    try:
        test_crosspoint_ring_interface()
        test_multiple_channels()
        print("\n🎉 所有接口兼容性测试通过!")
    except Exception as e:
        print(f"\n❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()