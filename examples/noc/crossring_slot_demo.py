#!/usr/bin/env python3
"""
CrossRing Slot和RingSlice机制演示

本演示专门展示CrossRing的核心创新：
1. CrossRingSlot - 符合Cross Ring Spec v2.0的slot定义
2. RingSlice - 环形传输的基本单元
3. I-Tag/E-Tag机制 - 防饿死机制的完整实现
4. 流水线传输 - Ring Slice的流水线架构

通过具体示例展示这些组件如何协同工作来实现高效的CrossRing通信。
"""

import sys
import os
import logging
from typing import List, Dict, Any

# 添加项目根目录到Python路径
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.noc.crossring.crossring_link import CrossRingSlot, RingSlice
from src.noc.crossring.tag_mechanism import CrossRingTagManager
from src.noc.crossring.flit import CrossRingFlit
from src.noc.crossring.config import create_crossring_config_custom
from src.noc.base.link import PriorityLevel


def setup_logging():
    """配置日志"""
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s", handlers=[logging.StreamHandler()])


def demo_crossring_slot_lifecycle():
    """演示CrossRingSlot的完整生命周期"""
    print("\n" + "=" * 60)
    print("演示 1: CrossRingSlot 生命周期")
    print("=" * 60)

    # 创建一个CrossRingSlot
    slot = CrossRingSlot(slot_id=1, cycle=0, channel="req")
    print(f"创建slot: {slot}")
    print(f"初始状态 - 可用: {slot.is_available}, 占用: {slot.is_occupied}, 预约: {slot.is_reserved}")

    # 阶段1: I-Tag预约
    print("\n阶段1: I-Tag预约机制")
    print("-" * 30)
    success = slot.reserve_itag(reserver_id=5, direction="TR")
    print(f"I-Tag预约结果: {success}")
    print(f"预约后状态 - 可用: {slot.is_available}, 占用: {slot.is_occupied}, 预约: {slot.is_reserved}")
    print(f"预约者ID: {slot.itag_reserver_id}, 预约方向: {slot.itag_direction}")

    # 清除I-Tag，准备分配flit
    slot.clear_itag()
    print("I-Tag预约清除")

    # 阶段2: Flit分配
    print("\n阶段2: Flit分配")
    print("-" * 30)
    flit = CrossRingFlit(packet_id=100, flit_id=1)
    flit.source = 0
    flit.destination = 8
    flit.req_type = "read"

    success = slot.assign_flit(flit)
    print(f"Flit分配结果: {success}")
    print(f"分配后状态 - 可用: {slot.is_available}, 占用: {slot.is_occupied}, 预约: {slot.is_reserved}")
    print(f"携带的flit: packet_id={slot.flit.packet_id}")

    # 阶段3: E-Tag标记
    print("\n阶段3: E-Tag优先级标记")
    print("-" * 30)
    slot.mark_etag(PriorityLevel.T1, "TL")
    print(f"E-Tag标记 - 优先级: {slot.etag_priority}, 方向: {slot.etag_direction}")

    # 测试优先级升级
    new_priority = slot.should_upgrade_etag(failed_attempts=2)
    print(f"尝试升级优先级 (失败2次): {new_priority}")

    if new_priority:
        slot.mark_etag(new_priority, "TL")
        print(f"优先级已升级到: {slot.etag_priority}")

    # 阶段4: Flit释放
    print("\n阶段4: Flit释放")
    print("-" * 30)
    released_flit = slot.release_flit()
    print(f"释放的flit: packet_id={released_flit.packet_id}")
    print(f"释放后状态 - 可用: {slot.is_available}, 占用: {slot.is_occupied}, 预约: {slot.is_reserved}")

    return True


def demo_ring_slice_pipeline():
    """演示RingSlice的流水线传输机制"""
    print("\n" + "=" * 60)
    print("演示 2: RingSlice 流水线传输")
    print("=" * 60)

    # 创建水平和垂直RingSlice
    h_slice = RingSlice("horizontal_slice", "horizontal", 0)
    v_slice = RingSlice("vertical_slice", "vertical", 1)

    print(f"创建水平RingSlice: {h_slice.slice_id}")
    print(f"创建垂直RingSlice: {v_slice.slice_id}")

    # 创建一系列带数据的slots
    print("\n准备测试数据...")
    test_slots = []
    for i in range(5):
        slot = CrossRingSlot(slot_id=i, cycle=i, channel="req")
        flit = CrossRingFlit(packet_id=200 + i, flit_id=1)
        flit.source = 0
        flit.destination = i + 1
        flit.req_type = "read" if i % 2 == 0 else "write"
        slot.assign_flit(flit)

        # 为某些slot添加E-Tag优先级
        if i % 3 == 0:
            slot.mark_etag(PriorityLevel.T1, "TL")
            print(f"Slot {i}: packet_id={200+i}, 优先级=T1")
        else:
            print(f"Slot {i}: packet_id={200+i}, 优先级=T2")

        test_slots.append(slot)

    # 执行流水线传输
    print("\n流水线传输阶段...")
    print("周期 | 输入缓存 | 当前slots | 输出缓存 | 传输slot")
    print("-" * 55)

    transmitted_slots = []
    slots_to_inject = test_slots.copy()
    for cycle in range(len(test_slots) + 10):
        # 每个周期注入一个slot，模拟流水线
        if slots_to_inject:
            slot = slots_to_inject.pop(0)
            h_slice.receive_slot(slot, "req")
        else:
            # 注入空槽以继续推进流水线
            h_slice.receive_slot(None, "req")

        # 执行step操作
        h_slice.step(cycle)

        # 获取状态信息
        input_count = 1 if h_slice.input_buffer.get("req") else 0
        current_count = 1 if h_slice.current_slots.get("req") else 0
        output_count = 1 if h_slice.output_buffer.get("req") else 0

        # 尝试传输
        transmitted_slot = h_slice.transmit_slot("req")
        transmitted_info = f"packet_id={transmitted_slot.flit.packet_id}" if transmitted_slot else "None"

        if transmitted_slot:
            transmitted_slots.append(transmitted_slot)

        print(f"{cycle:4d} |     {input_count:2d}   |     {current_count:2d}    |     {output_count:2d}    | {transmitted_info}")

    # 分析传输结果
    print(f"\n传输完成！总共传输了 {len(transmitted_slots)} 个slots")
    print("传输顺序:")
    for i, slot in enumerate(transmitted_slots):
        priority = slot.etag_priority if slot.etag_marked else "T2(默认)"
        print(f"  {i+1}. packet_id={slot.flit.packet_id}, 优先级={priority}")

    # 水平到垂直的传输
    print("\n水平到垂直传输...")
    if transmitted_slots:
        v_slice.receive_slot(transmitted_slots[0], "req")
        print(f"注入 {transmitted_slots[0].flit.packet_id} 到垂直slice")
        v_slice.step(0)
        v_slice.step(1)
        out_slot = v_slice.transmit_slot("req")
        if out_slot:
            print(f"从垂直slice传出: {out_slot.flit.packet_id}")

    return True


def demo_tag_mechanism_integration():
    """演示完整的Tag机制集成"""
    print("\n" + "=" * 60)
    print("演示 3: I-Tag/E-Tag 防饿死机制")
    print("=" * 60)

    # 创建Tag管理器
    config = create_crossring_config_custom(4, 4, "tag_demo")
    tag_manager = CrossRingTagManager(node_id=0, config=config)

    print(f"创建Tag管理器 for 节点 0")
    print(f"I-Tag配置: {tag_manager.itag_config}")
    print(f"E-Tag配置: {tag_manager.etag_config}")

    # 模拟高注入压力场景
    print("\n场景1: 高注入压力触发I-Tag")
    print("-" * 40)

    waiting_cycles = [10, 50, 80, 120, 150]  # 模拟不同的等待周期
    for cycle in waiting_cycles:
        should_trigger = tag_manager.should_trigger_itag("req", "horizontal", cycle)
        print(f"等待周期 {cycle:3d}: I-Tag触发 = {should_trigger}")

    # 模拟I-Tag预约过程
    print("\n场景2: I-Tag预约机制")
    print("-" * 40)

    # 创建一个RingSlice用于预约
    ring_slice = RingSlice("demo_ring", "horizontal", 0)

    # 添加一些slots到ring slice
    for i in range(3):
        slot = CrossRingSlot(slot_id=10 + i, cycle=0, channel="req")
        ring_slice.receive_slot(slot, "req")

    ring_slice.step(0)  # 让slots进入当前slots

    # 尝试I-Tag预约
    reservation_success = tag_manager.trigger_itag_reservation("req", "horizontal", ring_slice, 150)
    print(f"I-Tag预约结果: {reservation_success}")

    if reservation_success:
        print("I-Tag预约成功，节点获得注入优先权")
        # 取消预约
        cancel_success = tag_manager.cancel_itag_reservation("req", "horizontal", ring_slice)
        print(f"I-Tag预约取消: {cancel_success}")

    # 模拟E-Tag升级场景
    print("\n场景3: E-Tag优先级升级")
    print("-" * 40)

    # 创建不同方向的slots测试E-Tag升级
    directions = ["TL", "TR", "TU", "TD"]

    for direction in directions:
        print(f"\n测试方向 {direction}:")
        slot = CrossRingSlot(slot_id=20, cycle=0, channel="req")
        flit = CrossRingFlit(packet_id=300, flit_id=1)
        slot.assign_flit(flit)

        # 测试不同失败次数的升级
        for failed_attempts in [1, 2, 3]:
            new_priority = tag_manager.should_upgrade_etag(slot, "req", direction, failed_attempts)
            print(f"  失败 {failed_attempts} 次 -> 升级到 {new_priority}")

            if new_priority:
                slot.mark_etag(new_priority, direction)

    # 模拟E-Tag下环控制
    print("\n场景4: E-Tag下环控制")
    print("-" * 40)

    fifo_depths = [5, 8, 12, 15, 16]
    fifo_capacity = 16

    for priority in [PriorityLevel.T2, PriorityLevel.T1, PriorityLevel.T0]:
        print(f"\n优先级 {priority}:")
        slot = CrossRingSlot(slot_id=30, cycle=0, channel="req")
        flit = CrossRingFlit(packet_id=400, flit_id=1)
        slot.assign_flit(flit)
        slot.mark_etag(priority, "TL")

        for depth in fifo_depths:
            can_eject = tag_manager.can_eject_with_etag(slot, "req", "TL", depth, fifo_capacity)
            print(f"  FIFO占用 {depth:2d}/{fifo_capacity}: 可下环 = {can_eject}")

    # 显示最终统计
    print("\n最终统计:")
    print("-" * 20)
    final_status = tag_manager.get_tag_manager_status()
    print(f"I-Tag状态: {len(final_status['itag_states'])} 个通道")
    print(f"E-Tag状态: {len(final_status['etag_states'])} 个通道")
    print(f"统计信息: {final_status['stats']}")

    return True


def main():
    """主函数"""
    print("CrossRing Slot和RingSlice机制演示")
    print("=" * 80)
    print("本演示展示CrossRing的核心创新组件和机制")
    print("=" * 80)

    setup_logging()

    demos = [
        ("CrossRingSlot生命周期", demo_crossring_slot_lifecycle),
        ("RingSlice流水线传输", demo_ring_slice_pipeline),
        ("Tag防饿死机制", demo_tag_mechanism_integration),
    ]

    passed = 0
    total = len(demos)

    for demo_name, demo_func in demos:
        try:
            print(f"\n{'='*80}")
            print(f"开始演示: {demo_name}")
            print("=" * 80)

            if demo_func():
                passed += 1
                print(f"\n✓ {demo_name} 演示完成")
            else:
                print(f"\n✗ {demo_name} 演示失败")

        except Exception as e:
            print(f"\n✗ {demo_name} 演示异常: {e}")
            import traceback

            traceback.print_exc()

    print(f"\n{'='*80}")
    print(f"演示结果: {passed}/{total} 成功")

    if passed == total:
        print("🎉 所有演示完成！")
        print("\nCrossRing新架构特性总结:")
        print("- ✓ CrossRingSlot: 完整的I-Tag/E-Tag支持")
        print("- ✓ RingSlice: 高效的流水线传输架构")
        print("- ✓ I-Tag机制: 防止注入饿死的预约系统")
        print("- ✓ E-Tag机制: 防止下环饿死的优先级系统")
        print("- ✓ 性能提升: 高优先级流量的延迟优化")
    else:
        print(f"❌ {total - passed} 个演示未完成")

    return 0 if passed == total else 1


if __name__ == "__main__":
    sys.exit(main())
