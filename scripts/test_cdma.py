#!/usr/bin/env python3
"""
简单CDMA系统测试
"""

import sys
import os
import time

# 添加src目录到Python路径
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from protocol.cdma_system import CDMASystem
from protocol.memory_types import MemoryType

def simple_cdma_test():
    """简单的CDMA测试"""
    print("简单CDMA系统测试")
    print("=" * 40)
    
    # 创建两个芯片系统
    chip_A = CDMASystem("chip_A")
    chip_B = CDMASystem("chip_B")
    
    # 连接芯片
    chip_A.connect_to_chip("chip_B", chip_B)
    chip_B.connect_to_chip("chip_A", chip_A)
    
    print("✓ 创建并连接了两个CDMA系统")
    
    # 执行基本的receive/send配对
    print("\n执行CDMA_receive...")
    receive_result = chip_A.cdma_receive(
        dst_addr=0x1000,
        dst_shape=(1024,),
        dst_mem_type=MemoryType.GMEM,
        src_chip_id="chip_B",
        data_type="float32"
    )
    
    if receive_result.success:
        print(f"✓ CDMA_receive成功，事务ID: {receive_result.transaction_id}")
    else:
        print(f"✗ CDMA_receive失败: {receive_result.error_message}")
        return
    
    time.sleep(0.1)
    
    print("\n执行CDMA_send...")
    send_result = chip_B.cdma_send(
        src_addr=0x2000,
        src_shape=(1024,),
        dst_chip_id="chip_A",
        src_mem_type=MemoryType.GMEM,
        data_type="float32"
    )
    
    if send_result.success:
        print(f"✓ CDMA_send成功，传输了 {send_result.bytes_transferred} 字节")
        print(f"✓ 执行时间: {send_result.execution_time * 1000:.2f} ms")
    else:
        print(f"✗ CDMA_send失败: {send_result.error_message}")
    
    # 关闭系统
    chip_A.shutdown()
    chip_B.shutdown()
    
    print("\n✓ 测试完成")

if __name__ == "__main__":
    simple_cdma_test()