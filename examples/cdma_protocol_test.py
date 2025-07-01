#!/usr/bin/env python3
"""
CDMA协议测试用例
测试基于SG2260E芯片的CDMA发送/接收配对指令协议
"""

import sys
import os
import time
import threading
from typing import List, Tuple

# 添加src目录到Python路径
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from protocol.cdma_system import CDMASystem
from protocol.memory_types import MemoryType


class CDMAProtocolTester:
    """CDMA协议测试器"""
    
    def __init__(self):
        self.test_results: List[Tuple[str, bool, str]] = []
        self.systems = {}  # 简化类型注解以避免导入问题
    
    def setup_test_environment(self):
        """设置测试环境"""
        print("=" * 60)
        print("设置CDMA协议测试环境")
        print("=" * 60)
        
        # 创建3个芯片的CDMA系统
        chip_ids = ["chip_A", "chip_B", "chip_C"]
        
        for chip_id in chip_ids:
            system = CDMASystem(chip_id)
            self.systems[chip_id] = system
            print(f"✓ 创建 {chip_id} CDMA系统")
        
        # 建立芯片间连接（全连接拓扑）
        for i, chip1 in enumerate(chip_ids):
            for j, chip2 in enumerate(chip_ids):
                if i != j:
                    self.systems[chip1].connect_to_chip(chip2, self.systems[chip2])
        
        print("✓ 建立芯片间连接完成")
        print()
    
    def test_basic_send_receive_pair(self):
        """测试1：基本的send/receive配对"""
        print("测试1：基本的send/receive配对")
        print("-" * 40)
        
        try:
            chip_A = self.systems["chip_A"]
            chip_B = self.systems["chip_B"]
            
            # 芯片A执行CDMA_receive
            receive_result = chip_A.cdma_receive(
                dst_addr=0x1000,
                dst_shape=(1024,),
                dst_mem_type=MemoryType.GMEM,
                src_chip_id="chip_B",
                data_type="float32"
            )
            
            if not receive_result.success:
                self.test_results.append(("基本配对测试", False, f"CDMA_receive失败: {receive_result.error_message}"))
                return
            
            print(f"✓ chip_A CDMA_receive完成，事务ID: {receive_result.transaction_id}")
            
            # 等待一小段时间确保Credit传递
            time.sleep(0.1)
            
            # 芯片B执行CDMA_send
            send_result = chip_B.cdma_send(
                src_addr=0x2000,
                src_shape=(1024,),
                dst_chip_id="chip_A",
                src_mem_type=MemoryType.GMEM,
                data_type="float32"
            )
            
            if send_result.success:
                print(f"✓ chip_B CDMA_send完成，传输 {send_result.bytes_transferred} 字节")
                print(f"✓ 总执行时间: {send_result.execution_time * 1000:.2f} ms")
                self.test_results.append(("基本配对测试", True, "成功"))
            else:
                self.test_results.append(("基本配对测试", False, f"CDMA_send失败: {send_result.error_message}"))
            
        except Exception as e:
            self.test_results.append(("基本配对测试", False, f"异常: {str(e)}"))
        
        print()
    
    def test_multiple_parallel_transfers(self):
        """测试2：多引擎并行传输"""
        print("测试2：多引擎并行传输")
        print("-" * 40)
        
        try:
            chip_A = self.systems["chip_A"]
            chip_B = self.systems["chip_B"]
            chip_C = self.systems["chip_C"]
            
            # 芯片A同时接收来自B和C的数据
            receive_results = []
            
            # 第一个接收操作：从chip_B接收到L2M
            result1 = chip_A.cdma_receive(
                dst_addr=0x3000,
                dst_shape=(512,),
                dst_mem_type=MemoryType.L2M,
                src_chip_id="chip_B",
                data_type="float32"
            )
            receive_results.append(("从chip_B接收", result1))
            
            # 第二个接收操作：从chip_C接收到GMEM
            result2 = chip_A.cdma_receive(
                dst_addr=0x4000,
                dst_shape=(256,),
                dst_mem_type=MemoryType.GMEM,
                src_chip_id="chip_C",
                data_type="float16"
            )
            receive_results.append(("从chip_C接收", result2))
            
            # 检查接收操作是否成功
            failed_receives = [desc for desc, result in receive_results if not result.success]
            if failed_receives:
                self.test_results.append(("并行传输测试", False, f"接收操作失败: {failed_receives}"))
                return
            
            print("✓ 两个CDMA_receive操作完成")
            time.sleep(0.1)
            
            # 并行执行发送操作
            send_threads = []
            send_results = []
            
            def send_from_B():
                result = chip_B.cdma_send(
                    src_addr=0x5000,
                    src_shape=(512,),
                    dst_chip_id="chip_A",
                    src_mem_type=MemoryType.GMEM,
                    data_type="float32"
                )
                send_results.append(("chip_B发送", result))
            
            def send_from_C():
                result = chip_C.cdma_send(
                    src_addr=0x6000,
                    src_shape=(256,),
                    dst_chip_id="chip_A",
                    src_mem_type=MemoryType.L2M,
                    data_type="float16"
                )
                send_results.append(("chip_C发送", result))
            
            # 启动并行发送
            thread_B = threading.Thread(target=send_from_B)
            thread_C = threading.Thread(target=send_from_C)
            
            thread_B.start()
            thread_C.start()
            
            thread_B.join()
            thread_C.join()
            
            # 检查发送结果
            failed_sends = [desc for desc, result in send_results if not result.success]
            if failed_sends:
                self.test_results.append(("并行传输测试", False, f"发送操作失败: {failed_sends}"))
                return
            
            total_bytes = sum(result.bytes_transferred for _, result in send_results)
            print(f"✓ 并行传输完成，总传输字节数: {total_bytes}")
            self.test_results.append(("并行传输测试", True, f"成功传输 {total_bytes} 字节"))
            
        except Exception as e:
            self.test_results.append(("并行传输测试", False, f"异常: {str(e)}"))
        
        print()
    
    def test_credit_insufficient_error(self):
        """测试3：Credit不足错误处理"""
        print("测试3：Credit不足错误处理")
        print("-" * 40)
        
        try:
            chip_A = self.systems["chip_A"]
            chip_B = self.systems["chip_B"]
            
            # 直接尝试发送，没有先执行receive
            send_result = chip_B.cdma_send(
                src_addr=0x7000,
                src_shape=(128,),
                dst_chip_id="chip_A",
                src_mem_type=MemoryType.GMEM,
                data_type="float32"
            )
            
            if not send_result.success and "Credit" in send_result.error_message:
                print(f"✓ 正确检测到Credit不足: {send_result.error_message}")
                self.test_results.append(("Credit不足测试", True, "正确处理Credit不足"))
            else:
                self.test_results.append(("Credit不足测试", False, "未正确处理Credit不足"))
            
        except Exception as e:
            self.test_results.append(("Credit不足测试", False, f"异常: {str(e)}"))
        
        print()
    
    def test_shape_mismatch_error(self):
        """测试4：形状不匹配错误处理"""
        print("测试4：形状不匹配错误处理")
        print("-" * 40)
        
        try:
            chip_A = self.systems["chip_A"]
            chip_B = self.systems["chip_B"]
            
            # 芯片A接收1024个元素
            receive_result = chip_A.cdma_receive(
                dst_addr=0x8000,
                dst_shape=(1024,),
                dst_mem_type=MemoryType.GMEM,
                src_chip_id="chip_B",
                data_type="float32"
            )
            
            if not receive_result.success:
                self.test_results.append(("形状不匹配测试", False, f"CDMA_receive失败: {receive_result.error_message}"))
                return
            
            time.sleep(0.1)
            
            # 芯片B发送512个元素（形状不匹配）
            send_result = chip_B.cdma_send(
                src_addr=0x9000,
                src_shape=(512,),  # 不匹配的形状
                dst_chip_id="chip_A",
                src_mem_type=MemoryType.GMEM,
                data_type="float32"
            )
            
            if not send_result.success and "形状" in send_result.error_message:
                print(f"✓ 正确检测到形状不匹配: {send_result.error_message}")
                self.test_results.append(("形状不匹配测试", True, "正确处理形状不匹配"))
            else:
                self.test_results.append(("形状不匹配测试", False, "未正确处理形状不匹配"))
            
        except Exception as e:
            self.test_results.append(("形状不匹配测试", False, f"异常: {str(e)}"))
        
        print()
    
    def test_data_type_mismatch_error(self):
        """测试5：数据类型不匹配错误处理"""
        print("测试5：数据类型不匹配错误处理")
        print("-" * 40)
        
        try:
            chip_A = self.systems["chip_A"]
            chip_B = self.systems["chip_B"]
            
            # 芯片A接收float32类型
            receive_result = chip_A.cdma_receive(
                dst_addr=0xA000,
                dst_shape=(256,),
                dst_mem_type=MemoryType.GMEM,
                src_chip_id="chip_B",
                data_type="float32"
            )
            
            if not receive_result.success:
                self.test_results.append(("数据类型不匹配测试", False, f"CDMA_receive失败: {receive_result.error_message}"))
                return
            
            time.sleep(0.1)
            
            # 芯片B发送int32类型（数据类型不匹配）
            send_result = chip_B.cdma_send(
                src_addr=0xB000,
                src_shape=(256,),
                dst_chip_id="chip_A",
                src_mem_type=MemoryType.GMEM,
                data_type="int32"  # 不匹配的数据类型
            )
            
            if not send_result.success and "数据类型" in send_result.error_message:
                print(f"✓ 正确检测到数据类型不匹配: {send_result.error_message}")
                self.test_results.append(("数据类型不匹配测试", True, "正确处理数据类型不匹配"))
            else:
                self.test_results.append(("数据类型不匹配测试", False, "未正确处理数据类型不匹配"))
            
        except Exception as e:
            self.test_results.append(("数据类型不匹配测试", False, f"异常: {str(e)}"))
        
        print()
    
    def test_sync_message(self):
        """测试6：同步消息机制"""
        print("测试6：同步消息机制")
        print("-" * 40)
        
        try:
            chip_A = self.systems["chip_A"]
            chip_B = self.systems["chip_B"]
            
            # 发送同步消息
            sync_result = chip_A.cdma_sys_send_msg("chip_B", "传输完成同步")
            
            if sync_result.success:
                print(f"✓ 同步消息发送成功，执行时间: {sync_result.execution_time * 1000:.2f} ms")
                self.test_results.append(("同步消息测试", True, "成功"))
            else:
                self.test_results.append(("同步消息测试", False, f"失败: {sync_result.error_message}"))
            
        except Exception as e:
            self.test_results.append(("同步消息测试", False, f"异常: {str(e)}"))
        
        print()
    
    def test_performance_benchmark(self):
        """测试7：性能基准测试"""
        print("测试7：性能基准测试")
        print("-" * 40)
        
        try:
            chip_A = self.systems["chip_A"]
            chip_B = self.systems["chip_B"]
            
            # 测试大数据传输
            large_shape = (1024 * 1024,)  # 1M个float32 = 4MB
            
            print(f"开始传输 {large_shape[0] * 4 / (1024*1024):.1f} MB 数据...")
            
            start_time = time.time()
            
            # 接收操作
            receive_result = chip_A.cdma_receive(
                dst_addr=0x10000000,
                dst_shape=large_shape,
                dst_mem_type=MemoryType.GMEM,
                src_chip_id="chip_B",
                data_type="float32"
            )
            
            if not receive_result.success:
                self.test_results.append(("性能基准测试", False, f"接收失败: {receive_result.error_message}"))
                return
            
            time.sleep(0.01)
            
            # 发送操作
            send_result = chip_B.cdma_send(
                src_addr=0x20000000,
                src_shape=large_shape,
                dst_chip_id="chip_A",
                src_mem_type=MemoryType.GMEM,
                data_type="float32"
            )
            
            total_time = time.time() - start_time
            
            if send_result.success:
                data_size_mb = send_result.bytes_transferred / (1024 * 1024)
                throughput_mbps = data_size_mb / total_time
                
                print(f"✓ 性能测试完成:")
                print(f"  数据大小: {data_size_mb:.1f} MB")
                print(f"  总时间: {total_time * 1000:.2f} ms")
                print(f"  吞吐量: {throughput_mbps:.2f} MB/s")
                
                self.test_results.append(("性能基准测试", True, f"吞吐量: {throughput_mbps:.2f} MB/s"))
            else:
                self.test_results.append(("性能基准测试", False, f"发送失败: {send_result.error_message}"))
            
        except Exception as e:
            self.test_results.append(("性能基准测试", False, f"异常: {str(e)}"))
        
        print()
    
    def print_system_status(self):
        """打印系统状态信息"""
        print("系统状态信息")
        print("-" * 40)
        
        for chip_id, system in self.systems.items():
            status = system.get_system_status()
            print(f"\n{chip_id} 状态:")
            print(f"  系统状态: {status['state']}")
            print(f"  连接的芯片: {status['connected_chips']}")
            print(f"  DMA统计: {status['dma_performance']['total_transfers']} 次传输")
            print(f"  事务统计: {status['transaction_stats']['total_transactions']} 个事务")
        
        print()
    
    def run_all_tests(self):
        """运行所有测试"""
        print("开始CDMA协议完整测试套件")
        print("=" * 60)
        
        # 设置测试环境
        self.setup_test_environment()
        
        # 运行测试
        self.test_basic_send_receive_pair()
        self.test_multiple_parallel_transfers()
        self.test_credit_insufficient_error()
        self.test_shape_mismatch_error()
        self.test_data_type_mismatch_error()
        self.test_sync_message()
        self.test_performance_benchmark()
        
        # 打印系统状态
        self.print_system_status()
        
        # 清理系统
        for system in self.systems.values():
            system.cleanup()
        
        # 打印测试结果汇总
        self.print_test_summary()
    
    def print_test_summary(self):
        """打印测试结果汇总"""
        print("测试结果汇总")
        print("=" * 60)
        
        passed = 0
        failed = 0
        
        for test_name, success, details in self.test_results:
            status = "✓ 通过" if success else "✗ 失败"
            print(f"{status:<8} {test_name:<20} {details}")
            
            if success:
                passed += 1
            else:
                failed += 1
        
        print("-" * 60)
        print(f"总计: {len(self.test_results)} 个测试")
        print(f"通过: {passed} 个")
        print(f"失败: {failed} 个")
        print(f"成功率: {passed / len(self.test_results) * 100:.1f}%")
        
        if failed == 0:
            print("\n🎉 所有测试通过！CDMA协议实现正确。")
        else:
            print(f"\n⚠️ 有 {failed} 个测试失败，请检查实现。")
    
    def cleanup(self):
        """清理测试环境"""
        for system in self.systems.values():
            system.shutdown()
        self.systems.clear()


def main():
    """主函数"""
    tester = CDMAProtocolTester()
    
    try:
        tester.run_all_tests()
    except KeyboardInterrupt:
        print("\n测试被用户中断")
    except Exception as e:
        print(f"\n测试过程中发生异常: {str(e)}")
    finally:
        tester.cleanup()


if __name__ == "__main__":
    main()