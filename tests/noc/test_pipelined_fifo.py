#!/usr/bin/env python3
"""
测试PipelinedFIFO和新的IPInterface实现
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from src.noc.base.ip_interface import PipelinedFIFO, FlowControlledTransfer
from src.noc.crossring.config import CrossRingConfig
from src.noc.crossring.ip_interface import CrossRingIPInterface

def test_pipelined_fifo():
    """测试PipelinedFIFO基本功能"""
    print("=== 测试PipelinedFIFO ===")
    
    fifo = PipelinedFIFO("test_fifo", depth=4)
    
    # 测试初始状态
    assert fifo.valid_signal() == False
    assert fifo.ready_signal() == True
    assert len(fifo) == 0
    
    # 测试写入
    assert fifo.write_input("data1") == True
    assert len(fifo) == 1
    
    # 测试组合逻辑阶段
    fifo.step_compute_phase()
    assert fifo.next_output_valid == True
    
    # 测试时序逻辑阶段
    fifo.step_update_phase()
    assert fifo.valid_signal() == True
    assert fifo.peek_output() == "data1"
    
    # 测试读取
    data = fifo.read_output()
    assert data == "data1"
    
    # 再次更新，清空输出寄存器
    fifo.step_compute_phase()
    fifo.step_update_phase()
    assert fifo.valid_signal() == False
    
    print("PipelinedFIFO测试通过!")

def test_flow_controlled_transfer():
    """测试FlowControlledTransfer"""
    print("=== 测试FlowControlledTransfer ===")
    
    source = PipelinedFIFO("source", depth=4)
    dest = PipelinedFIFO("dest", depth=4)
    
    # 写入数据到源FIFO
    source.write_input("test_data")
    source.step_compute_phase()
    source.step_update_phase()
    
    # 测试传输
    assert FlowControlledTransfer.can_transfer(source, dest) == True
    assert FlowControlledTransfer.try_transfer(source, dest) == True
    
    # 更新目标FIFO
    dest.step_compute_phase()  
    dest.step_update_phase()
    
    # 验证传输结果
    assert dest.valid_signal() == True
    assert dest.peek_output() == "test_data"
    
    print("FlowControlledTransfer测试通过!")

def test_crossring_interface():
    """测试CrossRingIPInterface基本功能"""
    print("=== 测试CrossRingIPInterface ===")
    
    try:
        # 创建配置
        config = CrossRingConfig(num_row=2, num_col=2)
        
        # 创建IP接口
        ip_interface = CrossRingIPInterface(config, "gdma", 0, None)
        
        # 测试enqueue_request
        success = ip_interface.enqueue_request(
            source=0, 
            destination=1, 
            req_type="read", 
            burst_length=4
        )
        assert success == True
        
        # 测试step执行
        ip_interface.step(0)  # 第一个周期
        ip_interface.step(1)  # 第二个周期
        
        # 检查状态
        status = ip_interface.get_status()
        assert "fifo_status" in status
        assert "inject" in status["fifo_status"]["req"]
        
        print("CrossRingIPInterface基本测试通过!")
        
    except Exception as e:
        print(f"CrossRingIPInterface测试失败: {e}")
        import traceback
        traceback.print_exc()

def main():
    """主测试函数"""
    print("开始测试新的流水线FIFO实现...")
    
    test_pipelined_fifo()
    test_flow_controlled_transfer() 
    test_crossring_interface()
    
    print("\n所有测试完成!")

if __name__ == "__main__":
    main()