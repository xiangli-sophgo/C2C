#!/usr/bin/env python3
"""
RingSlice重构功能测试脚本

测试重构后的RingSlice基本功能：
1. PipelinedFIFO集成
2. 标准化流控接口
3. 两阶段执行模型
4. I-Tag预约slot处理
"""

import sys
import os

# 添加项目根目录到Python路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.noc.crossring.link import RingSlice, CrossRingSlot
from src.noc.base.link import BasicDirection
from src.noc.crossring.flit import CrossRingFlit

def test_basic_interfaces():
    """测试基本接口功能"""
    print("=== 测试基本接口功能 ===")
    
    # 创建RingSlice实例
    slice1 = RingSlice("test_slice_1", "horizontal", 0)
    print(f"✅ 创建RingSlice: {slice1.slice_id}")
    
    # 测试空状态
    for channel in ["req", "rsp", "data"]:
        assert slice1.can_accept_input(channel), f"新创建的slice应该能接受{channel}输入"
        assert not slice1.can_provide_output(channel), f"新创建的slice不应该有{channel}输出"
        assert slice1.peek_current_slot(channel) is None, f"新创建的slice的{channel}当前slot应该为空"
    
    print("✅ 空状态检查通过")
    
    # 创建测试slot和flit
    test_flit = CrossRingFlit(
        packet_id=1, 
        flit_id=0,
        source=0, 
        destination=1, 
        path=[0, 1],
        channel="req"
    )
    
    test_slot = CrossRingSlot(
        slot_id="test_slot",
        cycle=0,
        direction=BasicDirection.LOCAL,
        channel="req"
    )
    test_slot.assign_flit(test_flit)
    
    # 测试写入
    success = slice1.write_input(test_slot, "req")
    assert success, "写入slot应该成功"
    print("✅ Slot写入成功")
    
    # 测试状态变化
    assert not slice1.can_accept_input("req"), "写入后应该不能再接受req输入"
    # 注意：由于PipelinedFIFO的深度为2，第一次写入后还不会立即有输出
    
    # 执行两阶段
    slice1.step_compute_phase(1)
    slice1.step_update_phase(1)
    
    # 检查输出
    assert slice1.can_provide_output("req"), "执行两阶段后应该有req输出"
    output_slot = slice1.peek_output("req")
    assert output_slot is not None, "应该能peek到输出slot"
    assert output_slot.flit is not None, "输出slot应该包含flit"
    assert output_slot.flit.packet_id == 1, "输出flit的packet_id应该正确"
    
    print("✅ 两阶段执行和输出检查通过")
    
    return True

def test_itag_special_interface():
    """测试I-Tag特殊接口"""
    print("\n=== 测试I-Tag特殊接口 ===")
    
    slice1 = RingSlice("test_slice_2", "horizontal", 0)
    
    # 测试普通情况
    assert slice1.can_accept_slot_or_has_reserved_slot("req", 999), "空slice应该能接受任何节点的slot"
    
    # 创建预约slot
    reserved_slot = CrossRingSlot(
        slot_id="reserved_slot",
        cycle=0,
        direction=BasicDirection.LOCAL,
        channel="req"
    )
    reserved_slot.reserve_itag(123, "horizontal")  # 节点123预约
    
    # 写入预约slot
    success = slice1.write_input(reserved_slot, "req")
    assert success, "写入预约slot应该成功"
    
    # 执行两阶段使slot到达输出位置
    slice1.step_compute_phase(1)
    slice1.step_update_phase(1)
    
    # 测试预约检查
    assert slice1.can_accept_slot_or_has_reserved_slot("req", 123), "应该检测到节点123的预约"
    assert slice1.can_accept_slot_or_has_reserved_slot("req", 456) == False, "不应该检测到节点456的预约"
    
    print("✅ I-Tag预约检测通过")
    
    # 测试修改预约slot
    new_flit = CrossRingFlit(
        packet_id=2,
        flit_id=0, 
        source=0,
        destination=2,
        path=[0, 2],
        channel="req"
    )
    
    new_slot = CrossRingSlot(
        slot_id="new_slot",
        cycle=1,
        direction=BasicDirection.LOCAL,
        channel="req"
    )
    new_slot.assign_flit(new_flit)
    
    # 使用特殊接口修改预约slot
    success = slice1.write_slot_or_modify_reserved(new_slot, "req", 123)
    assert success, "修改预约slot应该成功"
    
    # 验证修改结果
    current_slot = slice1.peek_current_slot("req")
    assert current_slot is not None, "应该有当前slot"
    assert current_slot.flit is not None, "当前slot应该有flit"
    assert current_slot.flit.packet_id == 2, "flit应该被更新"
    assert not current_slot.is_reserved, "预约标记应该被清除"
    
    print("✅ I-Tag预约slot修改通过")
    
    return True

def test_pipeline_statistics():
    """测试统计信息集成"""
    print("\n=== 测试统计信息集成 ===")
    
    slice1 = RingSlice("test_slice_3", "horizontal", 0)
    
    # 获取统计信息
    stats = slice1.get_comprehensive_stats()
    
    assert "ring_slice_stats" in stats, "应该包含RingSlice统计"
    assert "pipeline_stats" in stats, "应该包含Pipeline统计"
    assert "current_occupancy" in stats, "应该包含当前占用"
    assert "flow_control_status" in stats, "应该包含流控状态"
    
    # 检查流控状态
    for channel in ["req", "rsp", "data"]:
        assert channel in stats["flow_control_status"], f"应该包含{channel}通道流控状态"
        assert "can_accept" in stats["flow_control_status"][channel], f"{channel}应该有can_accept状态"
        assert "can_provide" in stats["flow_control_status"][channel], f"{channel}应该有can_provide状态"
    
    print("✅ 统计信息集成检查通过")
    
    return True

def main():
    """主测试函数"""
    print("开始RingSlice重构功能测试...")
    
    try:
        # 基本接口测试
        test_basic_interfaces()
        
        # I-Tag特殊接口测试
        test_itag_special_interface()
        
        # 统计信息测试
        test_pipeline_statistics()
        
        print("\n🎉 所有测试通过！RingSlice重构成功！")
        print("\n📊 重构收益：")
        print("  ✅ 使用PipelinedFIFO统一流控架构")
        print("  ✅ 提供标准化的can_accept_input/write_input接口")
        print("  ✅ 集成成熟的两阶段执行模型")
        print("  ✅ 特殊处理I-Tag预约slot的复杂情况")
        print("  ✅ 集成丰富的统计和调试信息")
        
        return True
        
    except Exception as e:
        print(f"\n❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)